from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# SMOTE + pipeline que soporta resampling dentro de CV (evita leakage)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ----------------------------
# Config helpers
# ----------------------------

@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    random_state: int
    cv_folds: int
    use_smote: bool
    smote_k_neighbors: int
    grid_verbose: int
    order_by: str
    ohe_min_frequency: int


# ----------------------------
# Validation & Preprocessing
# ----------------------------

def _validate_inputs(df: pd.DataFrame) -> None:
    if "target_cls" not in df.columns:
        raise ValueError("model_input_classification debe contener la columna 'target_cls'.")
    if df["target_cls"].isna().any():
        raise ValueError("target_cls contiene valores nulos. Limpia/imputa antes de modelar.")


def _onehot_encoder(min_freq: int = 10) -> OneHotEncoder:
    """
    OneHot robusto con fallback según versión de scikit-learn.
    min_frequency reduce dimensionalidad cuando hay IDs poco frecuentes.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=min_freq)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore")


def _force_dense_ohe(ohe: OneHotEncoder) -> OneHotEncoder:
    """
    Fuerza salida DENSO del OHE si la versión lo permite.
    Es clave para evitar fallos con SMOTE (que típicamente no soporta sparse bien).
    """
    try:
        ohe.set_params(sparse_output=False)  # sklearn >=1.2
        return ohe
    except (TypeError, ValueError):
        pass

    try:
        ohe.set_params(sparse=False)  # sklearn <1.2
        return ohe
    except (TypeError, ValueError):
        return ohe


def _build_preprocessor(df: pd.DataFrame, ohe_min_frequency: int) -> ColumnTransformer:
    """
    Preprocesador robusto:
    - OneHotEncoder para IDs/categóricas (denso si es posible)
    - StandardScaler para numéricas

    Nota:
    - num_cols incluye base + features históricas/rolling (si existen).
    - Se filtran automáticamente según columnas presentes.
    """
    cat_cols = ["driverId", "constructorId", "circuitId"]

    num_cols = [
        # base
        "grid", "year", "round",
        # históricos (si existen)
        "driver_prev_rate", "driver_prev_count",
        "constructor_prev_rate", "constructor_prev_count",
        "circuit_prev_rate", "circuit_prev_count",
        # rolling (si existen)
        "driver_roll_rate_10", "driver_roll_rate_30",
        "constructor_roll_rate_10", "constructor_roll_rate_30",
        # experiencia transformada (si existe)
        "driver_log_exp", "constructor_log_exp",
    ]

    cat_cols = [c for c in cat_cols if c in df.columns]
    num_cols = [c for c in num_cols if c in df.columns]

    if len(cat_cols) == 0 and len(num_cols) == 0:
        raise ValueError("No se detectaron columnas de features en el dataset de clasificación.")

    ohe = _force_dense_ohe(_onehot_encoder(min_freq=ohe_min_frequency))

    return ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )


# ----------------------------
# Model candidates
# ----------------------------

def _get_candidates(random_state: int) -> List[Tuple[str, Any, Dict[str, List[Any]]]]:
    """
    >=5 modelos para clasificación con grids moderados (runtime razonable).
    Incluye HGB (muy fuerte en tabular) y GB, más clásicos.
    """
    return [
        (
            "hgb",
            HistGradientBoostingClassifier(random_state=random_state),
            {
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [3, 5, None],
                "model__max_iter": [200, 400, 800],
                "model__min_samples_leaf": [20, 50, 100],
                "model__l2_regularization": [0.0, 0.1, 1.0],
            },
        ),
        (
            "gb",
            GradientBoostingClassifier(random_state=random_state),
            {
                "model__n_estimators": [150, 250, 400],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.8, 1.0],
            },
        ),
        (
            "rf",
            RandomForestClassifier(random_state=random_state, class_weight="balanced", n_jobs=-1),
            {
                "model__n_estimators": [300, 600],
                "model__max_depth": [10, 20, None],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2, 5],
                "model__max_features": ["sqrt", "log2"],
            },
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(random_state=random_state, class_weight="balanced", n_jobs=-1),
            {
                "model__n_estimators": [300, 600],
                "model__max_depth": [10, 20, None],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2, 5],
                "model__max_features": ["sqrt", "log2"],
            },
        ),
        (
            "logreg",
            LogisticRegression(max_iter=5000, class_weight="balanced"),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__solver": ["lbfgs"],
            },
        ),
        (
            "svc",
            SVC(class_weight="balanced"),
            {
                "model__C": [0.5, 1.0, 2.0],
                "model__kernel": ["rbf", "linear"],
            },
        ),
        (
            "knn",
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [7, 15, 31],
                "model__weights": ["uniform", "distance"],
            },
        ),
    ]


# ----------------------------
# Main node
# ----------------------------

def train_and_evaluate_classification(
    model_input_classification: pd.DataFrame,
    modeling: dict,
):
    """
    Entrena >=5 modelos con GridSearchCV + StratifiedKFold(k>=5).
    SMOTE opcional, aplicado DENTRO del CV vía imblearn Pipeline (evita leakage),
    y se usa de forma selectiva (por defecto NO en boosting: hgb/gb).

    Outputs:
    1) classification_metrics_table (DataFrame) -> CSV
    2) classification_metrics_summary (dict)   -> JSON
    3) classification_metrics_plot (Figure)    -> PNG
    4) best_model_classification (Pipeline)    -> PKL
    """
    df = model_input_classification.copy()
    _validate_inputs(df)

    cfg = SplitConfig(
        test_size=float(modeling.get("test_size", 0.2)),
        random_state=int(modeling.get("random_state", 42)),
        cv_folds=int(modeling.get("cv_folds", 5)),
        use_smote=bool(modeling.get("use_smote", True)),
        smote_k_neighbors=int(modeling.get("smote_k_neighbors", 5)),
        grid_verbose=int(modeling.get("grid_verbose", 0)),
        order_by=str(modeling.get("order_by", "cv_f1_macro_mean")),
        ohe_min_frequency=int(modeling.get("ohe_min_frequency", 10)),
    )
    if cfg.cv_folds < 5:
        raise ValueError("cv_folds debe ser >= 5 para cumplir la rúbrica.")

    X = df.drop(columns=["target_cls"])
    y = df["target_cls"].astype(int)

    # Hold-out final (test)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)
    preprocessor = _build_preprocessor(X_train, ohe_min_frequency=cfg.ohe_min_frequency)
    candidates = _get_candidates(cfg.random_state)

    # Multi-métrica y refit por f1_macro (criterio principal defendible)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }
    refit_metric = "f1_macro"

    # SMOTE dentro del pipeline (por fold). Se activará selectivamente.
    smote = SMOTE(random_state=cfg.random_state, k_neighbors=cfg.smote_k_neighbors)

    rows: List[Dict[str, Any]] = []

    # Mejor por criterio principal (f1_macro CV)
    best_overall_estimator: Optional[Any] = None
    best_overall_name: Optional[str] = None
    best_overall_params: Optional[Dict[str, Any]] = None
    best_overall_cv_f1: float = -np.inf
    best_overall_used_smote: bool = False

    for name, model, grid in candidates:
        # SMOTE selectivo: típicamente NO mejora boosting y puede empeorarlo.
        use_smote_this = bool(cfg.use_smote) and name not in {"hgb", "gb"}

        steps = [("preprocess", preprocessor)]
        if use_smote_this:
            steps.append(("smote", smote))
        steps.append(("model", model))

        pipe = ImbPipeline(steps=steps)

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=scoring,
            refit=refit_metric,
            cv=cv,
            n_jobs=-1,
            return_train_score=True,
            verbose=cfg.grid_verbose,
        )

        gs.fit(X_train, y_train)

        best = gs.best_estimator_
        best_idx = gs.best_index_

        # Helpers métricas desde cv_results_
        def mean_test(metric: str) -> float:
            return float(gs.cv_results_[f"mean_test_{metric}"][best_idx])

        def std_test(metric: str) -> float:
            return float(gs.cv_results_[f"std_test_{metric}"][best_idx])

        def mean_train(metric: str) -> float:
            return float(gs.cv_results_[f"mean_train_{metric}"][best_idx])

        # CV metrics
        cv_acc_mean, cv_acc_std = mean_test("accuracy"), std_test("accuracy")
        cv_prec_mean, cv_prec_std = mean_test("precision_macro"), std_test("precision_macro")
        cv_rec_mean, cv_rec_std = mean_test("recall_macro"), std_test("recall_macro")
        cv_f1_mean, cv_f1_std = mean_test("f1_macro"), std_test("f1_macro")

        # Overfitting signal (train - cv en f1)
        train_f1_mean = mean_train("f1_macro")
        gen_gap_f1 = float(train_f1_mean - cv_f1_mean)

        # Test metrics
        y_pred = best.predict(X_test)
        test_acc = float(accuracy_score(y_test, y_pred))
        test_prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        test_rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        test_f1m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        rows.append(
            {
                "model": name,
                "use_smote": use_smote_this,
                "best_params": str(gs.best_params_),

                "cv_accuracy_mean": cv_acc_mean,
                "cv_accuracy_std": cv_acc_std,
                "cv_precision_macro_mean": cv_prec_mean,
                "cv_precision_macro_std": cv_prec_std,
                "cv_recall_macro_mean": cv_rec_mean,
                "cv_recall_macro_std": cv_rec_std,
                "cv_f1_macro_mean": cv_f1_mean,
                "cv_f1_macro_std": cv_f1_std,

                "train_f1_macro_mean": float(train_f1_mean),
                "generalization_gap_f1": float(gen_gap_f1),

                "test_accuracy": test_acc,
                "test_precision_macro": test_prec,
                "test_recall_macro": test_rec,
                "test_f1_macro": test_f1m,
            }
        )

        # Selección global por f1_macro CV (criterio principal)
        if cv_f1_mean > best_overall_cv_f1:
            best_overall_cv_f1 = cv_f1_mean
            best_overall_estimator = best
            best_overall_name = name
            best_overall_params = gs.best_params_
            best_overall_used_smote = use_smote_this

    results = pd.DataFrame(rows)

    # Orden configurable desde parameters.yml
    order_by = cfg.order_by
    if order_by not in results.columns:
        order_by = "cv_f1_macro_mean"

    results = results.sort_values(by=order_by, ascending=False).reset_index(drop=True)

    # Confusion matrix (best model seleccionado por f1_macro CV)
    if best_overall_estimator is None or best_overall_name is None:
        raise RuntimeError("No se pudo seleccionar un mejor modelo. Revisa datos/param grids.")

    best_pred = best_overall_estimator.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, best_pred, ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix (Test) - Best: {best_overall_name} | SMOTE={best_overall_used_smote}")
    fig.tight_layout()

    # Summary (reporta top-1 según orden_by y también el best por f1_cv si difiere)
    top = results.iloc[0].to_dict()

    summary = {
        "selection_criterion": order_by,
        "best_model": str(top["model"]),
        "best_params": str(top["best_params"]),
        "cv": {
            "folds": cfg.cv_folds,

            "best_accuracy_mean": float(top["cv_accuracy_mean"]),
            "best_accuracy_std": float(top["cv_accuracy_std"]),
            "best_precision_macro_mean": float(top["cv_precision_macro_mean"]),
            "best_precision_macro_std": float(top["cv_precision_macro_std"]),
            "best_recall_macro_mean": float(top["cv_recall_macro_mean"]),
            "best_recall_macro_std": float(top["cv_recall_macro_std"]),
            "best_f1_macro_mean": float(top["cv_f1_macro_mean"]),
            "best_f1_macro_std": float(top["cv_f1_macro_std"]),
        },
        "test": {
            "accuracy": float(accuracy_score(y_test, best_pred)),
            "precision_macro": float(precision_score(y_test, best_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, best_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, best_pred, average="macro", zero_division=0)),
        },
        "data_split": {
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "class_balance_train": y_train.value_counts(normalize=True).to_dict(),
            "class_balance_test": y_test.value_counts(normalize=True).to_dict(),
        },
        "smote": {
            "enabled_global": cfg.use_smote,
            "k_neighbors": cfg.smote_k_neighbors if cfg.use_smote else None,
            "note": "SMOTE se aplica dentro del pipeline (por fold) y de forma selectiva (no para hgb/gb) para evitar leakage.",
        },
        "models_compared": results["model"].tolist(),
        "notes": [
            "Se usa StratifiedKFold (k>=5) para preservar proporción de clases durante CV.",
            "Se reportan mean±std en CV para accuracy, precision_macro, recall_macro y f1_macro.",
            "Se reporta generalization_gap_f1 = train_f1_macro_mean - cv_f1_macro_mean como señal de overfitting.",
            "El gráfico de matriz de confusión corresponde al mejor modelo por f1_macro CV (refit_metric), no necesariamente al top-1 por order_by.",
        ],
    }

    return results, summary, fig, best_overall_estimator
