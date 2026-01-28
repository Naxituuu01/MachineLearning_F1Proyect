from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    train_test_split,
    KFold,
    GroupKFold,
    GridSearchCV,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


# ----------------------------
# Config (CatBoost)
# ----------------------------
CATBOOST_DIR = Path("data/06_models/catboost_info")
CATBOOST_DIR.mkdir(parents=True, exist_ok=True)


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

    # Segmentación SOLO TRAIN (opcional)
    reg_train_last_n_races: int | None
    reg_max_rows_total: int | None
    reg_pick_recent: bool

    # Muestreo aleatorio SOLO TRAIN por carreras
    reg_train_sample_frac: float | None
    reg_train_sample_n_races: int | None
    reg_train_sample_seed: int
    reg_train_sample_recent_pool_frac: float

    # Fast mode
    reg_fast_mode: bool


# ----------------------------
# Helpers (preprocess)
# ----------------------------
def _onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _dense_transformer() -> FunctionTransformer:
    def _to_dense(x):
        return x.toarray() if hasattr(x, "toarray") else x
    return FunctionTransformer(_to_dense, accept_sparse=True)


def _build_preprocessor(df: pd.DataFrame, kind: str) -> ColumnTransformer:
    """
    Preprocessor robusto:
    - Imputa NaNs en cat y num dentro del pipeline (fit SOLO en train).
    - Usa OneHot+Scaler para modelos lineales; Ordinal+passthrough para árboles.
    """

    # --------
    # Categóricas disponibles
    # --------
    cat_cols = ["driverId", "constructorId", "circuitId"]
    if "country" in df.columns:
        cat_cols.append("country")
    if "driver_constructor" in df.columns:
        cat_cols.append("driver_constructor")
    cat_cols = [c for c in cat_cols if c in df.columns]

    # --------
    # Numéricas candidatas (ALINEADAS con model_input/nodes.py)
    # Nota: solo tomamos las que existan realmente.
    # --------
    candidate_num = [
        # Básicas
        "grid", "year", "round",
        "laps",  # proxy pre-carrera (laps_expected)
        "circuit_baseline_pace",
        "n_drivers_race", "grid_pct_in_race",
        "season_progress", "era_bin",
        "driver_circuit_prev_count", "driver_circuit_log",
        "season_prev_mean_pace", "season_prev_mean_resid",

        # Qualy
        "quali_position", "quali_best_ms", "quali_gap_to_pole_ms",
        "quali_has_q2", "quali_has_q3",
        "quali_gap_pct", "grid_minus_quali",
        "quali_zscore_in_race", "quali_rank_pct_in_race",

        # Standings prev + normalizados
        "drv_points_prev", "drv_pos_prev", "drv_wins_prev",
        "cons_points_prev", "cons_pos_prev", "cons_wins_prev",
        "drv_points_per_round_prev", "drv_wins_per_round_prev",
        "cons_points_per_round_prev", "cons_wins_per_round_prev",

        # Rollings de qualy (según model_input: *_roll_mean_*)
        "drv_quali_roll_mean_5", "drv_quali_roll_mean_10",
        "cons_quali_roll_mean_5", "cons_quali_roll_mean_10",
        "cir_quali_roll_mean_5", "cir_quali_roll_mean_10",

        # Históricos pace (según model_input)
        "driver_prev_pace_mean", "driver_prev_pace_count",
        "constructor_prev_pace_mean", "constructor_prev_pace_count",
        "circuit_prev_pace_mean", "circuit_prev_pace_count",

        "driver_pace_roll_mean_5", "driver_pace_roll_mean_10", "driver_pace_roll_mean_30",
        "constructor_pace_roll_mean_5", "constructor_pace_roll_mean_10", "constructor_pace_roll_mean_30",
        "circuit_pace_roll_mean_5", "circuit_pace_roll_mean_10", "circuit_pace_roll_mean_30",

        "driver_log_exp", "constructor_log_exp", "circuit_log_exp",

        # Históricos resid (según model_input - MEJORA 5)
        "driver_prev_resid_mean", "driver_prev_resid_std",
        "constructor_prev_resid_mean", "constructor_prev_resid_std",
        "circuit_prev_resid_mean", "circuit_prev_resid_std",

        "driver_resid_roll_mean_10", "driver_resid_roll_std_10",
        "constructor_resid_roll_mean_10", "constructor_resid_roll_std_10",
        "circuit_resid_roll_mean_10", "circuit_resid_roll_std_10",

        # meta circuito (opcional)
        "lat", "lng", "alt",
    ]

    num_cols = [c for c in candidate_num if c in df.columns]

    # Árboles/boosting: OneHot (sin scaler) + imputación
    # Motivo: OrdinalEncoder introduce un "orden" artificial y suele bajar mucho el R² en árboles.
    if kind == "tree":
        return ColumnTransformer(
            transformers=[
                ("cat", Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("enc", _onehot()),  # OneHot para categorías
                ]), cat_cols),
                ("num", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                ]), num_cols),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )

    # Lineales/SVR/KNN: OneHot + Scale (con imputación)
    return ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("enc", _onehot()),
            ]), cat_cols),
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


# ----------------------------
# Candidates (models + grids)
# ----------------------------
def _get_candidates(random_state: int, fast_mode: bool):
    if fast_mode:
        return [
            ("ridge", Ridge(), {"model__alpha": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]}, "linear", False),

            ("elasticnet",
             ElasticNet(max_iter=40000, tol=1e-3),
             {"model__alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], "model__l1_ratio": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]},
             "linear",
             False),

            ("gbr",
             GradientBoostingRegressor(random_state=random_state),
             {
                 "model__n_estimators": [300, 600],
                 "model__learning_rate": [0.05, 0.1],
                 "model__max_depth": [2, 3],
             },
             "tree",
             False),

            ("hgb",
             HistGradientBoostingRegressor(random_state=random_state),
             {
                 "model__max_depth": [6, None],
                 "model__learning_rate": [0.05, 0.1],
                 "model__max_iter": [300, 600],
                 "model__min_samples_leaf": [20, 50],
                 "model__l2_regularization": [0.0, 0.1],
             },
             "tree",
             True),

            ("rf",
             RandomForestRegressor(random_state=random_state, n_jobs=-1),
             {
                 "model__n_estimators": [400],
                 "model__max_depth": [12, None],
                 "model__min_samples_leaf": [1, 2],
                 "model__max_features": ["sqrt"],
             },
             "tree",
             False),

            ("extra_trees",
             ExtraTreesRegressor(random_state=random_state, n_jobs=-1),
             {
                 "model__n_estimators": [400],
                 "model__max_depth": [12, None],
                 "model__min_samples_leaf": [1, 2],
                 "model__max_features": ["sqrt"],
             },
             "tree",
             False),

            ("catboost",
             CatBoostRegressor(
                 loss_function="RMSE",
                 random_seed=random_state,
                 verbose=False,
                 allow_writing_files=False,
             ),
             {
                 "model__depth": [6, 8],
                 "model__learning_rate": [0.05, 0.1],
                 "model__iterations": [600],
                 "model__l2_leaf_reg": [3, 10],
             },
             "tree",
             True),
        ]

    return [
        ("ridge", Ridge(), {"model__alpha": [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]}, "linear", False),

        ("elasticnet",
         ElasticNet(max_iter=60000, tol=1e-3),
         {"model__alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], "model__l1_ratio": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]},
         "linear",
         False),

        ("rf",
         RandomForestRegressor(random_state=random_state, n_jobs=-1),
         {
             "model__n_estimators": [600, 1200],
             "model__max_depth": [12, 20, None],
             "model__min_samples_leaf": [1, 2, 5],
             "model__max_features": ["sqrt", "log2"],
         },
         "tree",
         False),

        ("extra_trees",
         ExtraTreesRegressor(random_state=random_state, n_jobs=-1),
         {
             "model__n_estimators": [600, 1200],
             "model__max_depth": [12, 20, None],
             "model__min_samples_leaf": [1, 2, 5],
             "model__max_features": ["sqrt", "log2"],
         },
         "tree",
         False),

        ("gbr",
         GradientBoostingRegressor(random_state=random_state),
         {
             "model__n_estimators": [400, 800],
             "model__learning_rate": [0.03, 0.05, 0.1],
             "model__max_depth": [2, 3],
         },
         "tree",
         False),

        ("hgb",
         HistGradientBoostingRegressor(random_state=random_state),
         {
             "model__max_depth": [6, 10, None],
             "model__learning_rate": [0.03, 0.05, 0.1],
             "model__max_iter": [400, 800],
             "model__min_samples_leaf": [20, 50, 100],
             "model__l2_regularization": [0.0, 0.1, 1.0],
         },
         "tree",
         True),

        ("knn",
         KNeighborsRegressor(),
         {"model__n_neighbors": [7, 15, 31], "model__weights": ["uniform", "distance"]},
         "linear",
         True),

        ("svr",
         SVR(),
         {"model__C": [1.0, 5.0, 10.0], "model__epsilon": [0.05, 0.1], "model__kernel": ["rbf"]},
         "linear",
         True),

        ("catboost",
         CatBoostRegressor(
             loss_function="RMSE",
             random_seed=random_state,
             verbose=False,
             allow_writing_files=False,
         ),
         {
             "model__depth": [6, 8, 10],
             "model__learning_rate": [0.03, 0.05, 0.1],
             "model__iterations": [800, 1500],
             "model__l2_leaf_reg": [1, 3, 10],
         },
         "tree",
         True),
    ]


# ----------------------------
# Validations
# ----------------------------
def _assert_required_columns(df: pd.DataFrame) -> None:
    required = {"target_reg", "year", "round", "raceId"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"model_input_regression debe contener: {sorted(missing)}")


def _assert_no_leakage_columns(df: pd.DataFrame) -> None:
    """
    Aquí validamos que no haya columnas típicas post-carrera que se hayan colado.
    Ojo: en tu model_input sí existen target_ms y laps_actual, pero NO deben usarse como X.
    """
    forbidden = {
        "time", "points", "positionOrder", "position",
        "fastestLap", "fastestLapTime", "fastestLapSpeed", "rank", "statusId",
        "milliseconds",
    }
    present = forbidden.intersection(set(df.columns))
    # target_ms y target_reg son targets/auxiliares, no "leakage" en el dataset; el leakage real se controla al crear X.
    present = {c for c in present if c not in {"target_reg", "target_ms"}}
    if present:
        raise ValueError(
            f"Leakage detectado en model_input_regression: {sorted(present)}. "
            "X debe contener solo variables pre-carrera."
        )


# ----------------------------
# Split + CV
# ----------------------------
def _maybe_time_split(X: pd.DataFrame, y: pd.Series, cfg: SplitConfig):
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


def _build_cv(X_train: pd.DataFrame, cfg: SplitConfig):
    if {"raceId", "year", "round"}.issubset(X_train.columns):
        races = (
            X_train[["raceId", "year", "round"]]
            .drop_duplicates()
            .sort_values(["year", "round", "raceId"])
            .reset_index(drop=True)
        )
        race_ids = races["raceId"].values
        if len(race_ids) >= cfg.cv_folds * 5:
            tscv = TimeSeriesSplit(n_splits=cfg.cv_folds)
            splits = []
            for tr_r, va_r in tscv.split(race_ids):
                tr_set = set(race_ids[tr_r])
                va_set = set(race_ids[va_r])
                tr_idx = np.where(X_train["raceId"].isin(tr_set).values)[0]
                va_idx = np.where(X_train["raceId"].isin(va_set).values)[0]
                splits.append((tr_idx, va_idx))
            return splits, "TimeSeriesSplit(raceId ordered by year/round)"

    if "raceId" in X_train.columns:
        return GroupKFold(n_splits=cfg.cv_folds), "GroupKFold(raceId)"

    return KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state), "KFold(shuffle=True)"


def _ms_to_min(x: float) -> float:
    return float(x) / 60000.0


# ----------------------------
# Segmentation / sampling (TRAIN only)
# (se mantiene igual a tu implementación)
# ----------------------------
def _segment_train_last_n_races(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    last_n_races: int,
    pick_recent: bool,
    min_unique_races: int,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    info = {
        "applied": False,
        "mode": "last_n_races",
        "last_n_races": int(last_n_races),
        "pick_recent": bool(pick_recent),
        "rows_before": int(len(X_train)),
        "rows_after": int(len(X_train)),
        "races_before": int(X_train["raceId"].nunique()) if "raceId" in X_train.columns else None,
        "races_after": int(X_train["raceId"].nunique()) if "raceId" in X_train.columns else None,
    }
    if last_n_races is None or last_n_races <= 0:
        return X_train, y_train, info

    if not {"raceId", "year", "round"}.issubset(X_train.columns):
        return X_train, y_train, info

    races = (
        X_train[["raceId", "year", "round"]]
        .drop_duplicates()
        .sort_values(["year", "round", "raceId"])
        .reset_index(drop=True)
    )
    selected = races["raceId"].tail(last_n_races).tolist() if pick_recent else races["raceId"].head(last_n_races).tolist()

    if len(selected) < min_unique_races:
        return X_train, y_train, info

    X2 = X_train[X_train["raceId"].isin(selected)].copy()
    y2 = y_train.loc[X2.index].copy()

    info.update({"applied": True, "rows_after": int(len(X2)), "races_after": int(X2["raceId"].nunique())})
    return X2, y2, info


def _segment_by_races_max_rows(
    X: pd.DataFrame,
    y: pd.Series,
    max_rows: int,
    pick_recent: bool,
    min_unique_races: int = 30,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    info = {"applied": False, "mode": "max_rows", "rows_before": int(len(X)), "rows_after": int(len(X)), "races_selected": 0}

    if max_rows is None or max_rows <= 0 or len(X) <= max_rows:
        return X, y, info

    if not {"raceId", "year", "round"}.issubset(X.columns):
        X2 = X.tail(max_rows) if pick_recent else X.head(max_rows)
        y2 = y.loc[X2.index]
        info.update({"applied": True, "rows_after": int(len(X2)), "races_selected": int(X2["raceId"].nunique()) if "raceId" in X2.columns else None})
        return X2, y2, info

    races = (
        X[["raceId", "year", "round"]]
        .drop_duplicates()
        .sort_values(["year", "round", "raceId"])
        .reset_index(drop=True)
    )
    if pick_recent:
        races = races.iloc[::-1].reset_index(drop=True)

    counts = X.groupby("raceId").size().rename("n_rows").reset_index()
    races = races.merge(counts, on="raceId", how="left").fillna({"n_rows": 0})

    selected = []
    cum = 0
    for _, row in races.iterrows():
        rid = int(row["raceId"])
        n = int(row["n_rows"])
        if n <= 0:
            continue
        if cum + n > max_rows and len(selected) >= min_unique_races:
            break
        selected.append(rid)
        cum += n
        if cum >= max_rows and len(selected) >= min_unique_races:
            break

    if len(selected) < min_unique_races:
        return X, y, info

    X2 = X[X["raceId"].isin(selected)].copy()
    y2 = y.loc[X2.index].copy()

    info.update({"applied": True, "rows_after": int(len(X2)), "races_selected": int(X2["raceId"].nunique()), "max_rows_target": int(max_rows), "pick_recent": bool(pick_recent)})
    return X2, y2, info


def _sample_train_by_races_random(
    X: pd.DataFrame,
    y: pd.Series,
    frac: float | None,
    n_races: int | None,
    seed: int,
    pick_recent: bool,
    min_unique_races: int,
    recent_pool_frac: float = 0.60,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    info = {
        "applied": False,
        "mode": "random_races",
        "rows_before": int(len(X)),
        "rows_after": int(len(X)),
        "races_before": int(X["raceId"].nunique()) if "raceId" in X.columns else None,
        "races_after": None,
        "frac": frac,
        "n_races": n_races,
        "seed": int(seed),
        "pick_recent": bool(pick_recent),
        "min_unique_races": int(min_unique_races),
        "recent_pool_frac": float(recent_pool_frac),
    }

    if "raceId" not in X.columns or not {"year", "round"}.issubset(X.columns):
        return X, y, info

    races = (
        X[["raceId", "year", "round"]]
        .drop_duplicates()
        .sort_values(["year", "round", "raceId"])
        .reset_index(drop=True)
    )
    all_race_ids = races["raceId"].tolist()
    n_total = len(all_race_ids)

    if n_total < min_unique_races:
        return X, y, info

    pool = all_race_ids
    if pick_recent:
        rp = float(np.clip(recent_pool_frac, 0.10, 1.0))
        k_pool = max(min_unique_races, int(round(rp * n_total)))
        pool = all_race_ids[-k_pool:]

    rng = np.random.default_rng(int(seed))

    if n_races is not None:
        k = int(n_races)
    elif frac is not None:
        k = int(round(float(frac) * len(pool)))
    else:
        return X, y, info

    k = max(min_unique_races, min(k, len(pool)))
    selected = rng.choice(pool, size=k, replace=False).tolist()

    X2 = X[X["raceId"].isin(selected)].copy()
    y2 = y.loc[X2.index].copy()

    X2 = X2.sort_values(["year", "round", "raceId"]).copy()
    y2 = y2.loc[X2.index].copy()

    info.update({"applied": True, "rows_after": int(len(X2)), "races_after": int(X2["raceId"].nunique()), "selected_races": int(len(selected))})
    return X2, y2, info


# ----------------------------
# Feature importance utils
# ----------------------------
def _safe_get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    try:
        names = preprocessor.get_feature_names_out()
        return [str(n) for n in names]
    except Exception:
        return []


def _compute_feature_importances(best_estimator, use_residual_target: bool, top_k: int = 30) -> pd.DataFrame:
    if not isinstance(best_estimator, Pipeline):
        # si algo cambia en el futuro, no rompemos
        return pd.DataFrame(columns=["feature", "importance", "method", "use_residual_target"])

    preprocess = best_estimator.named_steps.get("preprocess")
    model = best_estimator.named_steps.get("model")

    feature_names = _safe_get_feature_names(preprocess) if preprocess is not None else []
    method = None
    importances = None

    if hasattr(model, "coef_"):
        coef = np.asarray(getattr(model, "coef_"))
        importances = np.abs(coef).ravel()
        method = "abs(coef_)"
    elif hasattr(model, "feature_importances_"):
        fi = np.asarray(getattr(model, "feature_importances_"))
        importances = fi.ravel()
        method = "feature_importances_"

    if importances is None or len(importances) == 0:
        return pd.DataFrame(columns=["feature", "importance", "method", "use_residual_target"])

    if not feature_names or len(feature_names) != len(importances):
        feature_names = [f"f_{i}" for i in range(len(importances))]

    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp["importance"] = pd.to_numeric(df_imp["importance"], errors="coerce").fillna(0.0)
    df_imp = df_imp.sort_values("importance", ascending=False).head(int(top_k)).reset_index(drop=True)
    df_imp["method"] = method
    df_imp["use_residual_target"] = bool(use_residual_target)
    return df_imp


def _unwrap_pipeline(estimator):
    """
    Si estimator es TransformedTargetRegressor, retorna su regressor (Pipeline).
    Si no, retorna estimator.
    """
    try:
        from sklearn.compose import TransformedTargetRegressor
        if isinstance(estimator, TransformedTargetRegressor):
            return estimator.regressor
    except Exception:
        pass
    return estimator


def _align_series_like_index(s: pd.Series | None, idx: pd.Index) -> pd.Series | None:
    """Alinea una serie al índice dado si existe."""
    if s is None:
        return None
    return s.loc[idx].copy()


# ----------------------------
# Main node
# ----------------------------
def train_and_evaluate_regression(model_input_regression: pd.DataFrame, modeling: dict):
    df = model_input_regression.copy()

    _assert_required_columns(df)
    _assert_no_leakage_columns(df)

    cfg = SplitConfig(
        test_size=float(modeling.get("test_size", 0.2)),
        random_state=int(modeling.get("random_state", 42)),
        cv_folds=int(modeling.get("cv_folds", 5)),
        use_time_split=bool(modeling.get("use_time_split", True)),
        time_split_year_cutoff=int(modeling.get("time_split_year_cutoff", 2018)),
        use_log_target=bool(modeling.get("use_log_target", False)),

        reg_train_last_n_races=(int(modeling.get("reg_train_last_n_races")) if modeling.get("reg_train_last_n_races") is not None else None),
        reg_max_rows_total=(int(modeling.get("reg_max_rows_total")) if modeling.get("reg_max_rows_total") is not None else None),
        reg_pick_recent=bool(modeling.get("reg_pick_recent", True)),

        reg_train_sample_frac=(float(modeling.get("reg_train_sample_frac")) if modeling.get("reg_train_sample_frac") is not None else None),
        reg_train_sample_n_races=(int(modeling.get("reg_train_sample_n_races")) if modeling.get("reg_train_sample_n_races") is not None else None),
        reg_train_sample_seed=int(modeling.get("reg_train_sample_seed", int(modeling.get("random_state", 42)))),
        reg_train_sample_recent_pool_frac=float(modeling.get("reg_train_sample_recent_pool_frac", 0.60)),

        reg_fast_mode=bool(modeling.get("reg_fast_mode", False)),
    )

    if cfg.cv_folds < 5:
        raise ValueError("cv_folds debe ser >= 5 para cumplir la rúbrica.")

    # y_base (pace)
    y_base = pd.to_numeric(df["target_reg"], errors="coerce")
    mask = y_base.notna()

    # ----------------------------
    # Definir target (CONTROLADO POR PARÁMETRO)
    # ----------------------------
    reg_target_mode = str(modeling.get("reg_target_mode", "pace")).strip().lower()
    if reg_target_mode not in {"pace", "residual"}:
        raise ValueError(f"reg_target_mode inválido: {reg_target_mode}. Usa 'pace' o 'residual'.")

    use_resid = (
        reg_target_mode == "residual"
        and ("target_resid" in df.columns)
        and ("circuit_baseline_pace" in df.columns)
    )

    if use_resid:
        y = pd.to_numeric(df.loc[mask, "target_resid"], errors="coerce")
        use_log = False  # residual puede ser negativo
        baseline_all = pd.to_numeric(df.loc[mask, "circuit_baseline_pace"], errors="coerce")
    else:
        # ENTRENAMOS PACE (target_reg)
        y = y_base.loc[mask].copy()
        use_log = bool(cfg.use_log_target)
        baseline_all = None

    # ----------------------------
    # Drop columns (NO leakage en X)
    # ----------------------------
    drop_cols = ["target_reg"]
    if "target_ms" in df.columns:
        drop_cols.append("target_ms")
    if "target_resid" in df.columns:
        drop_cols.append("target_resid")
    if "laps_actual" in df.columns:
        drop_cols.append("laps_actual")
    for c in ["true_ms", "pred_ms", "milliseconds"]:
        if c in df.columns and c not in drop_cols:
            drop_cols.append(c)

    X = df.drop(columns=drop_cols, errors="ignore").loc[mask].copy()

    # limpieza finitos
    y = pd.to_numeric(y, errors="coerce")
    good_y = np.isfinite(y.values)
    y = y.loc[y.index[good_y]].copy()
    X = X.loc[y.index].copy()
    if baseline_all is not None:
        baseline_all = baseline_all.loc[y.index].copy()

    # ----------------------------
    # Métricas ms: target_ms + laps proxy (pre-carrera)
    # ----------------------------
    y_ms_all = pd.to_numeric(df.loc[y.index, "target_ms"], errors="coerce") if "target_ms" in df.columns else None

    # Prioridad: laps_expected -> laps -> laps_actual (último recurso)
    laps_all = None
    if "laps_expected" in df.columns:
        laps_all = pd.to_numeric(df.loc[y.index, "laps_expected"], errors="coerce")
    elif "laps" in df.columns:
        laps_all = pd.to_numeric(df.loc[y.index, "laps"], errors="coerce")
    elif "laps_actual" in df.columns:
        laps_all = pd.to_numeric(df.loc[y.index, "laps_actual"], errors="coerce")

    # ----------------------------
    # split principal
    # ----------------------------
    X_train, X_test, y_train, y_test, split_strategy = _maybe_time_split(X, y, cfg)

    # segmentación TRAIN only
    seg_info = {"applied": False}
    min_unique_races = cfg.cv_folds * 5

    if (cfg.reg_train_sample_frac is not None) or (cfg.reg_train_sample_n_races is not None):
        X_train, y_train, info_tr = _sample_train_by_races_random(
            X_train, y_train,
            frac=cfg.reg_train_sample_frac,
            n_races=cfg.reg_train_sample_n_races,
            seed=cfg.reg_train_sample_seed,
            pick_recent=cfg.reg_pick_recent,
            min_unique_races=min_unique_races,
            recent_pool_frac=cfg.reg_train_sample_recent_pool_frac,
        )
        seg_info = info_tr
        split_strategy += (
            f" + train_sample_random_races(frac={cfg.reg_train_sample_frac}, "
            f"n_races={cfg.reg_train_sample_n_races}, recent_pool={cfg.reg_train_sample_recent_pool_frac}, "
            f"pick_recent={cfg.reg_pick_recent})"
        )

    elif cfg.reg_train_last_n_races is not None:
        X_train, y_train, info_tr = _segment_train_last_n_races(
            X_train, y_train,
            last_n_races=cfg.reg_train_last_n_races,
            pick_recent=cfg.reg_pick_recent,
            min_unique_races=min_unique_races,
        )
        seg_info = info_tr
        split_strategy += f" + train_segment(last_n_races={cfg.reg_train_last_n_races}, pick_recent={cfg.reg_pick_recent})"

    elif cfg.reg_max_rows_total is not None:
        X_train, y_train, info_tr = _segment_by_races_max_rows(
            X_train, y_train,
            max_rows=int(cfg.reg_max_rows_total),
            pick_recent=cfg.reg_pick_recent,
            min_unique_races=min_unique_races,
        )
        seg_info = info_tr
        split_strategy += f" + train_segment(max_rows={cfg.reg_max_rows_total}, pick_recent={cfg.reg_pick_recent})"

    # IMPORTANTÍSIMO: al segmentar TRAIN, alinea también baseline/laps a los índices segmentados
    baseline_train = _align_series_like_index(baseline_all, X_train.index) if baseline_all is not None else None
    laps_train = _align_series_like_index(laps_all, X_train.index) if laps_all is not None else None

    baseline_test = _align_series_like_index(baseline_all, X_test.index) if baseline_all is not None else None
    y_test_ms = _align_series_like_index(y_ms_all, X_test.index) if y_ms_all is not None else None
    laps_test = _align_series_like_index(laps_all, X_test.index) if laps_all is not None else None

    # Fallbacks robustos (se calculan desde TRAIN ya segmentado)
    if baseline_test is not None:
        fallback = float(np.nanmedian(baseline_train.values)) if baseline_train is not None else float(np.nanmedian(baseline_all.values))
        baseline_test = baseline_test.fillna(fallback)

    if laps_test is not None:
        fallback = float(np.nanmedian(laps_train.values)) if laps_train is not None else float(np.nanmedian(laps_all.values))
        laps_test = laps_test.fillna(fallback)

    # ----------------------------
    # CV
    # ----------------------------
    cv, cv_strategy = _build_cv(X_train, cfg)

    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
    refit_metric = "r2"  # <-- objetivo: maximizar R² en pace

    rows = []
    best_estimators = {}  # guardamos el mejor estimator por modelo (refit por r2)
    best_estimator = None
    best_name = None
    best_cv_r2 = -np.inf

    candidates = _get_candidates(cfg.random_state, fast_mode=cfg.reg_fast_mode)

    for name, model, grid, kind, needs_dense in candidates:
        preprocessor = _build_preprocessor(X_train.drop(columns=["raceId"], errors="ignore"), kind=kind)

        steps = [("preprocess", preprocessor)]
        if needs_dense:
            steps.append(("to_dense", _dense_transformer()))
        steps.append(("model", model))
        base_pipe = Pipeline(steps)

        # log1p solo si corresponde (solo para pace, no residual)
        if use_log:
            if float(np.nanmin(y_train.values)) <= -1.0:
                estimator = base_pipe
                param_grid = grid
                log_used = False
            else:
                from sklearn.compose import TransformedTargetRegressor
                estimator = TransformedTargetRegressor(
                    regressor=base_pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False
                )
                param_grid = {f"regressor__{k}": v for k, v in grid.items()}
                log_used = True
        else:
            estimator = base_pipe
            param_grid = grid
            log_used = False

        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            refit=refit_metric,
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
            verbose=int(modeling.get("grid_verbose", 0)),
            error_score="raise",
        )

        Xtr_fit = X_train.drop(columns=["raceId"], errors="ignore")
        Xte_fit = X_test.drop(columns=["raceId"], errors="ignore")

        if isinstance(cv, GroupKFold):
            groups = X_train["raceId"].values
            gs.fit(Xtr_fit, y_train, groups=groups)
        else:
            gs.fit(Xtr_fit, y_train)

        best = gs.best_estimator_
        best_idx = gs.best_index_

        cv_rmse_mean = float(-gs.cv_results_["mean_test_rmse"][best_idx])
        cv_rmse_std = float(gs.cv_results_["std_test_rmse"][best_idx])
        cv_mae_mean = float(-gs.cv_results_["mean_test_mae"][best_idx])
        cv_mae_std = float(gs.cv_results_["std_test_mae"][best_idx])
        cv_r2_mean = float(gs.cv_results_["mean_test_r2"][best_idx])
        cv_r2_std = float(gs.cv_results_["std_test_r2"][best_idx])

        # predicción en el target entrenado (resid o pace)
        y_pred_target = best.predict(Xte_fit)

        # reconstruir pace real si entrenaste residual
        if use_resid:
            base = baseline_test.values.astype(float)
            y_true_pace = (y_test.values.astype(float) + base)
            y_pred_pace = (np.asarray(y_pred_target).astype(float) + base)
        else:
            y_true_pace = y_test.values.astype(float)
            y_pred_pace = np.asarray(y_pred_target).astype(float)

        valid_pace = np.isfinite(y_true_pace) & np.isfinite(y_pred_pace)
        test_rmse = float(np.sqrt(mean_squared_error(y_true_pace[valid_pace], y_pred_pace[valid_pace])))
        test_mae = float(mean_absolute_error(y_true_pace[valid_pace], y_pred_pace[valid_pace]))
        test_r2 = float(r2_score(y_true_pace[valid_pace], y_pred_pace[valid_pace]))

        # ms reconstruido usando laps proxy pre-carrera
        test_ms_rmse = None
        test_ms_mae = None
        test_ms_r2 = None
        if (y_test_ms is not None) and (laps_test is not None):
            pred_ms = np.asarray(y_pred_pace) * np.asarray(laps_test)
            true_ms = np.asarray(y_test_ms)
            valid_ms = np.isfinite(pred_ms) & np.isfinite(true_ms)
            if int(valid_ms.sum()) > 0:
                test_ms_rmse = float(np.sqrt(mean_squared_error(true_ms[valid_ms], pred_ms[valid_ms])))
                test_ms_mae = float(mean_absolute_error(true_ms[valid_ms], pred_ms[valid_ms]))
                test_ms_r2 = float(r2_score(true_ms[valid_ms], pred_ms[valid_ms]))

        rows.append(
            {
                "model": name,
                "use_log_target": bool(log_used),
                "use_residual_target": bool(use_resid),
                "fast_mode": bool(cfg.reg_fast_mode),
                "best_params": str(gs.best_params_),

                "cv_rmse_mean_pace": cv_rmse_mean,
                "cv_rmse_std_pace": cv_rmse_std,
                "cv_mae_mean_pace": cv_mae_mean,
                "cv_mae_std_pace": cv_mae_std,
                "cv_r2_mean_pace": cv_r2_mean,
                "cv_r2_std_pace": cv_r2_std,

                "test_rmse_pace": test_rmse,
                "test_mae_pace": test_mae,
                "test_r2_pace": test_r2,

                "test_rmse_ms": test_ms_rmse,
                "test_mae_ms": test_ms_mae,
                "test_r2_ms": test_ms_r2,
            }
        )
        # guardar el mejor estimator (ya está refiteado por refit_metric="r2")
        best_estimators[name] = best

        # criterio global: maximizar R² en CV (pace)
        if cv_r2_mean > best_cv_r2:
            best_cv_r2 = cv_r2_mean
            best_estimator = best
            best_name = name

    # ordenar resultados por R² descendente (higher is better)
    results = (
        pd.DataFrame(rows)
        .sort_values("cv_r2_mean_pace", ascending=False)
        .reset_index(drop=True)
)

    # -------------------------------------------------
    # Consistencia final: el mejor es el top-1 por cv_r2_mean_pace
    # -------------------------------------------------
    best_name = str(results.iloc[0]["model"])
    best_estimator = best_estimators[best_name]

    # predicciones del mejor modelo
    Xte_fit = X_test.drop(columns=["raceId"], errors="ignore")
    best_pred_target = best_estimator.predict(Xte_fit)

    if use_resid:
        base = baseline_test.values.astype(float)
        y_true_pace = (y_test.values.astype(float) + base)
        y_pred_pace = (np.asarray(best_pred_target).astype(float) + base)
    else:
        y_true_pace = y_test.values.astype(float)
        y_pred_pace = np.asarray(best_pred_target).astype(float)

    pred_df = pd.DataFrame(
        {
            "raceId": X_test["raceId"].values if "raceId" in X_test.columns else np.nan,
            "year": X_test["year"].values if "year" in X_test.columns else np.nan,
            "round": X_test["round"].values if "round" in X_test.columns else np.nan,
            "driverId": X_test["driverId"].values if "driverId" in X_test.columns else np.nan,
            "constructorId": X_test["constructorId"].values if "constructorId" in X_test.columns else np.nan,
            "circuitId": X_test["circuitId"].values if "circuitId" in X_test.columns else np.nan,
            "use_residual_target": bool(use_resid),
            "y_true_target": y_test.values.astype(float),
            "y_pred_target": np.asarray(best_pred_target).astype(float),
            "circuit_baseline_pace": baseline_test.values.astype(float) if baseline_test is not None else np.nan,
            "y_true_pace": y_true_pace,
            "y_pred_pace": y_pred_pace,
            "laps_used": np.asarray(laps_test).astype(float) if laps_test is not None else np.nan,
            "true_ms": np.asarray(y_test_ms).astype(float) if y_test_ms is not None else np.nan,
        },
        index=X_test.index,
    )

    if laps_test is not None:
        pred_df["pred_ms"] = pred_df["y_pred_pace"] * pred_df["laps_used"]
    else:
        pred_df["pred_ms"] = np.nan

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pred_df["y_true_pace"], pred_df["y_pred_pace"], alpha=0.5)
    lo = float(np.nanmin([pred_df["y_true_pace"].min(), pred_df["y_pred_pace"].min()]))
    hi = float(np.nanmax([pred_df["y_true_pace"].max(), pred_df["y_pred_pace"].max()]))
    ax.plot([lo, hi], [lo, hi])
    ax.set_title(f"y_true vs y_pred (Test, pace ms/lap) - Best: {best_name}")
    ax.set_xlabel("y_true (pace ms/lap)")
    ax.set_ylabel("y_pred (pace ms/lap)")
    fig.tight_layout()

    top = results.iloc[0].to_dict()

    summary = {
        "selection_criterion": "cv_r2_mean_pace (higher is better)",
        "best_model": str(best_name),
        "best_params": str(top["best_params"]),
        "refit_metric": str(refit_metric),
        "cv": {
            "folds": cfg.cv_folds,
            "strategy": cv_strategy,
            "best_rmse_mean_pace": float(top["cv_rmse_mean_pace"]),
            "best_rmse_std_pace": float(top["cv_rmse_std_pace"]),
            "best_r2_mean_pace": float(top["cv_r2_mean_pace"]),
            "best_r2_std_pace": float(top["cv_r2_std_pace"]),
        },
        "test_pace": {
            "rmse_pace": float(top["test_rmse_pace"]),
            "mae_pace": float(top["test_mae_pace"]),
            "r2_pace": float(top["test_r2_pace"]),
        },
        "test_ms_reconstructed": {
            "rmse_ms": top["test_rmse_ms"],
            "mae_ms": top["test_mae_ms"],
            "r2_ms": top["test_r2_ms"],
            "rmse_minutes": _ms_to_min(float(top["test_rmse_ms"])) if top["test_rmse_ms"] is not None else None,
            "mae_minutes": _ms_to_min(float(top["test_mae_ms"])) if top["test_mae_ms"] is not None else None,
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
        "segmentation_train": seg_info,
        "fast_mode": bool(cfg.reg_fast_mode),
        "models_compared": results["model"].tolist(),
        "notes": [
            "Entrenamiento en pace (ms/lap) o residual (pace - baseline).",
            "Target de entrenamiento controlado por parámetro (reg_target_mode): 'pace' o 'residual'. En este run se usa el valor configurado.",
            "Segmentación SOLO TRAIN opcional: random por raceId / últimas N carreras / max_rows (manteniendo test completo).",
            "raceId se usa solo para CV/orden; se elimina antes de alimentar el pipeline.",
            "ms reconstruido usa laps_expected/laps (proxy pre-carrera); laps_actual solo como último recurso.",
        ],
    }

    # Feature importances: soporta Pipeline o TransformedTargetRegressor
    pipe_for_fi = _unwrap_pipeline(best_estimator)
    fi_df = _compute_feature_importances(
        pipe_for_fi,
        use_residual_target=use_resid,
        top_k=int(modeling.get("fi_top_k", 30)),
    )

    return results, summary, fig, best_estimator, pred_df, fi_df

