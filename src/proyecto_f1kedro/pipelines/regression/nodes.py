from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    random_state: int
    cv_folds: int
    use_time_split: bool
    time_split_year_cutoff: int
    use_log_target: bool


# ----------------------------
# Helpers
# ----------------------------
def _onehot_dense() -> OneHotEncoder:
    """OHE denso para compatibilidad amplia."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Preprocesador:
    - OHE (denso) para IDs
    - StandardScaler para numéricas
    Nota: raceId NO se incluye en features (solo groups/orden).
    """
    cat_cols = ["driverId", "constructorId", "circuitId"]

    num_cols = [
        # base
        "grid", "year", "round",
        # históricos pace (desde model_input_regression)
        "driver_prev_ms_mean", "driver_prev_ms_count",
        "constructor_prev_ms_mean", "constructor_prev_ms_count",
        "circuit_prev_ms_mean", "circuit_prev_ms_count",
        # rolling
        "driver_roll_ms_10", "driver_roll_ms_30",
        "constructor_roll_ms_10", "constructor_roll_ms_30",
        # experiencia
        "driver_log_exp", "constructor_log_exp",
    ]

    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in num_cols if c in df.columns]

    return ColumnTransformer(
        transformers=[
            ("cat", _onehot_dense(), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )


def _get_candidates(random_state: int) -> List[Tuple[str, Any, Dict[str, List[Any]]]]:
    """
    Grids moderados. Nota: param_grid usa prefijo regressor__model__ porque
    envolveremos el Pipeline dentro de TransformedTargetRegressor.
    """
    return [
        ("ridge", Ridge(), {"regressor__model__alpha": [0.1, 1.0, 10.0, 50.0]}),
        ("lasso", Lasso(max_iter=40000, tol=1e-3), {"regressor__model__alpha": [1e-4, 1e-3, 1e-2, 1e-1]}),
        (
            "elasticnet",
            ElasticNet(max_iter=40000, tol=1e-3),
            {"regressor__model__alpha": [1e-3, 1e-2, 1e-1], "regressor__model__l1_ratio": [0.2, 0.5, 0.8]},
        ),
        (
            "rf",
            RandomForestRegressor(random_state=random_state, n_jobs=-1),
            {
                "regressor__model__n_estimators": [400, 800],
                "regressor__model__max_depth": [10, 20, None],
                "regressor__model__min_samples_split": [2, 5, 10],
                "regressor__model__min_samples_leaf": [1, 2, 5],
                "regressor__model__max_features": ["sqrt", "log2"],
            },
        ),
        (
            "extra_trees",
            ExtraTreesRegressor(random_state=random_state, n_jobs=-1),
            {
                "regressor__model__n_estimators": [400, 800],
                "regressor__model__max_depth": [10, 20, None],
                "regressor__model__min_samples_split": [2, 5, 10],
                "regressor__model__min_samples_leaf": [1, 2, 5],
                "regressor__model__max_features": ["sqrt", "log2"],
            },
        ),
        (
            "gbr",
            GradientBoostingRegressor(random_state=random_state),
            {
                "regressor__model__n_estimators": [300, 600],
                "regressor__model__learning_rate": [0.03, 0.05, 0.1],
                "regressor__model__max_depth": [2, 3],
            },
        ),
        (
            "hgb",
            HistGradientBoostingRegressor(random_state=random_state),
            {
                "regressor__model__max_depth": [6, 10, None],
                "regressor__model__learning_rate": [0.03, 0.05, 0.1],
                "regressor__model__max_iter": [300, 600, 900],
                "regressor__model__min_samples_leaf": [20, 50, 100],
                "regressor__model__l2_regularization": [0.0, 0.1, 1.0],
            },
        ),
        (
            "knn",
            KNeighborsRegressor(),
            {"regressor__model__n_neighbors": [7, 15, 31], "regressor__model__weights": ["uniform", "distance"]},
        ),
        (
            "svr",
            SVR(),
            {"regressor__model__C": [1.0, 5.0, 10.0], "regressor__model__epsilon": [0.05, 0.1, 0.2], "regressor__model__kernel": ["rbf"]},
        ),
    ]


def _assert_no_leakage_columns(df: pd.DataFrame) -> None:
    forbidden = {
        "milliseconds", "time", "points", "positionOrder", "position", "laps",
        "fastestLap", "fastestLapTime", "fastestLapSpeed", "rank", "statusId"
    }
    present = forbidden.intersection(set(df.columns))
    present = {c for c in present if c != "target_reg"}
    if present:
        raise ValueError(
            f"Leakage detectado en model_input_regression: {sorted(present)}. "
            "X debe contener solo variables pre-carrera."
        )


def _maybe_time_split(
    X: pd.DataFrame, y: pd.Series, cfg: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    """
    Split temporal (defendible) si year existe y hay datos suficientes.
    """
    if cfg.use_time_split and "year" in X.columns:
        cutoff = cfg.time_split_year_cutoff
        train_mask = X["year"] <= cutoff
        test_mask = X["year"] > cutoff
        if train_mask.sum() >= 200 and test_mask.sum() >= 200:
            return (
                X.loc[train_mask].copy(),
                X.loc[test_mask].copy(),
                y.loc[train_mask].copy(),
                y.loc[test_mask].copy(),
                f"time_split(year <= {cutoff} vs > {cutoff})",
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )
    return X_train, X_test, y_train, y_test, "random_split(train_test_split)"


def _build_cv(X_train: pd.DataFrame, cfg: SplitConfig) -> tuple[Any, str]:
    """
    CV dentro de train:
    - Si existe raceId => GroupKFold por carrera (muy importante en F1).
    - Si no => KFold shuffle.
    Nota: no usamos TimeSeriesSplit aquí porque ya hay split temporal principal.
    """
    if "raceId" in X_train.columns:
        return GroupKFold(n_splits=cfg.cv_folds), "GroupKFold(raceId)"
    return KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state), "KFold(shuffle=True)"


def _ms_to_min(x: float) -> float:
    return float(x) / 60000.0


def _strip_regressor_prefix(grid: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    """
    Si use_log_target=False, no existe TransformedTargetRegressor y los params
    deben apuntar a model__... en el Pipeline.
    """
    out = {}
    for k, v in grid.items():
        out[k.replace("regressor__", "")] = v
    return out


# ----------------------------
# Main node
# ----------------------------
def train_and_evaluate_regression(model_input_regression: pd.DataFrame, modeling: dict):
    """
    Outputs (Kedro):
      1) regression_metrics_table (CSV)
      2) regression_metrics_summary (JSON)
      3) regression_metrics_plot (PNG)
      4) best_model_regression (PKL)
    """
    df = model_input_regression.copy()

    if "target_reg" not in df.columns:
        raise ValueError("model_input_regression debe contener la columna 'target_reg'.")

    _assert_no_leakage_columns(df)

    cfg = SplitConfig(
        test_size=float(modeling.get("test_size", 0.2)),
        random_state=int(modeling.get("random_state", 42)),
        cv_folds=int(modeling.get("cv_folds", 5)),
        use_time_split=bool(modeling.get("use_time_split", True)),
        time_split_year_cutoff=int(modeling.get("time_split_year_cutoff", 2018)),
        use_log_target=bool(modeling.get("use_log_target", True)),
    )
    if cfg.cv_folds < 5:
        raise ValueError("cv_folds debe ser >= 5 para cumplir la rúbrica.")

    X = df.drop(columns=["target_reg"])
    y = pd.to_numeric(df["target_reg"], errors="coerce")

    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    # Split principal
    X_train, X_test, y_train, y_test, split_strategy = _maybe_time_split(X, y, cfg)

    # CV
    cv, cv_strategy = _build_cv(X_train, cfg)

    preprocessor = _build_preprocessor(X_train)

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    refit_metric = "rmse"

    candidates = _get_candidates(cfg.random_state)

    rows = []
    best_estimator = None
    best_name = None
    best_cv_rmse = np.inf

    for name, model, grid in candidates:
        base_pipe = Pipeline([("preprocess", preprocessor), ("model", model)])

        if cfg.use_log_target:
            estimator = TransformedTargetRegressor(
                regressor=base_pipe,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=False,
            )
            param_grid = grid
        else:
            estimator = base_pipe
            param_grid = _strip_regressor_prefix(grid)

        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            refit=refit_metric,
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
            verbose=int(modeling.get("grid_verbose", 0)),
        )

        if cv_strategy.startswith("GroupKFold"):
            groups = X_train["raceId"].values
            gs.fit(X_train, y_train, groups=groups)
        else:
            gs.fit(X_train, y_train)

        best = gs.best_estimator_
        best_idx = gs.best_index_

        def mean_test(metric: str) -> float:
            return float(gs.cv_results_[f"mean_test_{metric}"][best_idx])

        def std_test(metric: str) -> float:
            return float(gs.cv_results_[f"std_test_{metric}"][best_idx])

        cv_rmse_mean = float(-mean_test("rmse"))
        cv_rmse_std = float(std_test("rmse"))
        cv_mae_mean = float(-mean_test("mae"))
        cv_mae_std = float(std_test("mae"))
        cv_r2_mean = float(mean_test("r2"))
        cv_r2_std = float(std_test("r2"))

        # Test
        y_pred = best.predict(X_test)
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        test_mae = float(mean_absolute_error(y_test, y_pred))
        test_r2 = float(r2_score(y_test, y_pred))

        rows.append(
            {
                "model": name,
                "use_log_target": cfg.use_log_target,
                "best_params": str(gs.best_params_),
                "cv_rmse_mean": cv_rmse_mean,
                "cv_rmse_std": cv_rmse_std,
                "cv_mae_mean": cv_mae_mean,
                "cv_mae_std": cv_mae_std,
                "cv_r2_mean": cv_r2_mean,
                "cv_r2_std": cv_r2_std,
                "test_rmse": test_rmse,
                "test_mae": test_mae,
                "test_r2": test_r2,
            }
        )

        if cv_rmse_mean < best_cv_rmse:
            best_cv_rmse = cv_rmse_mean
            best_estimator = best
            best_name = name

    results = pd.DataFrame(rows).sort_values("cv_rmse_mean", ascending=True).reset_index(drop=True)

    # Plot y_true vs y_pred (best model)
    best_pred = best_estimator.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, best_pred, alpha=0.5)
    lo = float(min(y_test.min(), best_pred.min()))
    hi = float(max(y_test.max(), best_pred.max()))
    ax.plot([lo, hi], [lo, hi])
    ax.set_title(f"y_true vs y_pred (Test) - Best: {best_name}")
    ax.set_xlabel("y_true (ms)")
    ax.set_ylabel("y_pred (ms)")
    fig.tight_layout()

    top = results.iloc[0].to_dict()

    summary = {
        "selection_criterion": "cv_rmse_mean (lower is better)",
        "best_model": str(best_name),
        "best_params": str(top["best_params"]),
        "cv": {
            "folds": cfg.cv_folds,
            "strategy": cv_strategy,
            "best_rmse_mean": float(top["cv_rmse_mean"]),
            "best_rmse_std": float(top["cv_rmse_std"]),
            "best_rmse_mean_minutes": _ms_to_min(float(top["cv_rmse_mean"])),
            "best_r2_mean": float(top["cv_r2_mean"]),
            "best_r2_std": float(top["cv_r2_std"]),
        },
        "test": {
            "rmse": float(top["test_rmse"]),
            "mae": float(top["test_mae"]),
            "r2": float(top["test_r2"]),
            "rmse_minutes": _ms_to_min(float(top["test_rmse"])),
            "mae_minutes": _ms_to_min(float(top["test_mae"])),
        },
        "data_split": {
            "strategy": split_strategy,
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "year_min_train": int(X_train["year"].min()) if "year" in X_train.columns else None,
            "year_max_train": int(X_train["year"].max()) if "year" in X_train.columns else None,
            "year_min_test": int(X_test["year"].min()) if "year" in X_test.columns else None,
            "year_max_test": int(X_test["year"].max()) if "year" in X_test.columns else None,
        },
        "models_compared": results["model"].tolist(),
        "notes": [
            "Se bloquean columnas post-carrera para evitar leakage.",
            "Split temporal principal por year si está habilitado (más defendible).",
            "CV dentro de train usa GroupKFold por raceId si existe (evita mezclar la misma carrera).",
            "use_log_target=true estabiliza la varianza del target (ms) y suele mejorar generalización.",
            "La mejora grande de R2 normalmente requiere features de pace pre-carrera (prev/rolling), ya incluidas en model_input.",
        ],
    }

    return results, summary, fig, best_estimator
