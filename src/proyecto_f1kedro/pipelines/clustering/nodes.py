from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


# ----------------------------
# Helpers
# ----------------------------
def _safe_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _get_race_feature_columns(df_race: pd.DataFrame) -> List[str]:
    """Elige columnas numéricas para clustering a nivel carrera."""
    # Excluir identificadores obvios
    drop_like = {"raceId", "year", "round"}
    num_cols = [c for c in df_race.columns if c not in drop_like and pd.api.types.is_numeric_dtype(df_race[c])]
    # Si year/round están y quieres permitirlos, puedes incluirlos; por defecto los dejamos fuera para no “clusterizar por calendario”.
    return num_cols


def _fit_pca(X: np.ndarray, variance_target: float, random_state: int) -> Tuple[PCA, np.ndarray, Dict[str, Any]]:
    pca = PCA(n_components=float(variance_target), random_state=int(random_state))
    Xp = pca.fit_transform(X)
    info = {
        "variance_target": float(variance_target),
        "n_components_": int(pca.n_components_),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return pca, Xp, info


def _k_search_range(params: dict) -> List[int]:
    k_min = int(params.get("k_min", 2))
    k_max = int(params.get("k_max", 12))
    k_min = max(2, k_min)
    k_max = max(k_min, k_max)
    return list(range(k_min, k_max + 1))


# ----------------------------
# Node 1: build race features
# ----------------------------
def build_race_level_features(model_input_regression: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Construye features a nivel de carrera (raceId) desde model_input_regression.

    Nota:
    - Esto es UNSUPERVISED para E3.
    - Para evitar leakage conceptual, puedes excluir targets; aquí se excluyen target_* y target_ms.
    """
    df = model_input_regression.copy()

    group_key = str(params.get("group_key", "raceId"))
    if group_key not in df.columns:
        raise ValueError(f"model_input_regression debe contener '{group_key}' para construir clustering por carrera.")

    # Excluir targets/columnas post-carrera del set de clustering
    forbidden = [c for c in df.columns if c.startswith("target_")]
    forbidden += [c for c in ["target_ms", "laps_actual", "true_ms", "pred_ms"] if c in df.columns]

    keep_cols = [c for c in df.columns if c not in set(forbidden)]
    df = df[keep_cols].copy()

    # Asegurar numéricos típicos (si existen)
    df = _safe_num(df, ["year", "round", group_key])

    # Construir agregaciones por carrera
    agg: Dict[str, List[str]] = {}

    # Agregamos estadísticos sobre numéricas (robusto)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != group_key]
    for c in num_cols:
        agg[c] = ["mean", "std", "median", "min", "max"]

    # Además, tamaño de la parrilla observada en el model_input
    df["_rows"] = 1
    agg["_rows"] = ["sum"]

    race = df.groupby(group_key, sort=False).agg(agg)
    # Flatten columns
    race.columns = [f"{a}__{b}" for a, b in race.columns]
    race = race.reset_index()

    # Features derivadas útiles
    if "grid__mean" in race.columns and "grid__std" in race.columns:
        race["grid_cv"] = (race["grid__std"] / (race["grid__mean"].abs() + 1e-6)).replace([np.inf, -np.inf], np.nan)

    # year/round si venían, los guardamos como referencia (sin usar en clustering por defecto)
    # Si existen en el model_input, a nivel carrera quedarán como mean/median, etc.
    return race


# ----------------------------
# Node 2: fit clustering
# ----------------------------
def fit_clustering(clustering_race_features: pd.DataFrame, params: dict):
    """
    - Escala numéricas (opcional)
    - PCA a varianza objetivo
    - (opcional) búsqueda k con elbow/silhouette/CH/DB para reporte
    - Ajuste FINAL con KMeans k_final
    - Entrega:
      labels_race (raceId, cluster_kmeans_kfinal)
      metrics dict
      elbow fig
      silhouette fig
      bundle (preprocess + modelos)
      cluster_profile (tabla)
      pca2d fig
    """
    df = clustering_race_features.copy()
    group_key = str(params.get("group_key", "raceId"))
    if group_key not in df.columns:
        raise ValueError(f"clustering_race_features debe contener '{group_key}'.")

    # Selección de variables para clustering
    feature_cols = _get_race_feature_columns(df)
    if len(feature_cols) < 3:
        raise ValueError(
            f"Pocas columnas numéricas para clustering ({len(feature_cols)}). "
            f"Revisa clustering_race_features. feature_cols={feature_cols}"
        )

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Imputación simple (pre) antes de scaler: mediana por columna
    X = X.apply(lambda s: s.fillna(s.median()), axis=0)

    scale_numeric = bool(params.get("scale_numeric", True))
    if scale_numeric:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X.values)
    else:
        scaler = None
        Xs = X.values

    # PCA
    pca_var = float(params.get("pca_variance", 0.90))
    pca_rs = int(params.get("pca_random_state", 42))
    pca, Xp, pca_info = _fit_pca(Xs, pca_var, pca_rs)

    # Búsqueda k
    do_search = bool(params.get("do_search", True))
    ks = _k_search_range(params)

    inertia_by_k = []
    sil_by_k = []
    ch_by_k = []
    db_by_k = []
    gmm_bic_by_k = []

    kmeans_rs = int(params.get("kmeans_random_state", 42))
    kmeans_n_init = int(params.get("kmeans_n_init", 20))
    kmeans_max_iter = int(params.get("kmeans_max_iter", 300))
    gmm_rs = int(params.get("gmm_random_state", 42))

    if do_search:
        for k in ks:
            km = KMeans(
                n_clusters=int(k),
                random_state=kmeans_rs,
                n_init=kmeans_n_init,
                max_iter=kmeans_max_iter,
            )
            labels = km.fit_predict(Xp)
            inertia_by_k.append(float(km.inertia_))

            # Silhouette requiere k>=2 y <n_samples
            sil = silhouette_score(Xp, labels) if (k >= 2 and k < len(Xp)) else np.nan
            sil_by_k.append(float(sil))

            ch_by_k.append(float(calinski_harabasz_score(Xp, labels)))
            db_by_k.append(float(davies_bouldin_score(Xp, labels)))

            # GMM BIC (opcional, te da respaldo “no solo KMeans”)
            gmm = GaussianMixture(n_components=int(k), random_state=gmm_rs)
            gmm.fit(Xp)
            gmm_bic_by_k.append(float(gmm.bic(Xp)))

        # plots
        fig_elbow, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(ks, inertia_by_k, marker="o")
        ax1.set_title("Elbow (Inertia) vs K")
        ax1.set_xlabel("K")
        ax1.set_ylabel("Inertia")
        fig_elbow.tight_layout()

        fig_sil, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(ks, sil_by_k, marker="o")
        ax2.set_title("Silhouette vs K")
        ax2.set_xlabel("K")
        ax2.set_ylabel("Silhouette score")
        fig_sil.tight_layout()
    else:
        fig_elbow, fig_sil = None, None

    # K final (lo que tú quieres: K=4)
    method = str(params.get("method", "kmeans")).lower()
    k_final = int(params.get("k_final", 4))
    if method != "kmeans":
        raise ValueError("Por ahora este pipeline está fijado a method='kmeans' para E3 (tu decisión K=4).")

    km_final = KMeans(
        n_clusters=int(k_final),
        random_state=kmeans_rs,
        n_init=kmeans_n_init,
        max_iter=kmeans_max_iter,
    )
    final_labels = km_final.fit_predict(Xp)

    labels_race = pd.DataFrame(
        {
            group_key: df[group_key].values,
            f"cluster_kmeans_k{k_final}": final_labels.astype(int),
        }
    )

    # Perfil de clusters (interpretabilidad)
    tmp = df[[group_key] + feature_cols].merge(labels_race, on=group_key, how="left")
    ccol = f"cluster_kmeans_k{k_final}"
    profile = tmp.groupby(ccol, sort=True)[feature_cols].agg(["count", "mean", "median", "std", "min", "max"])
    profile.columns = [f"{a}__{b}" for a, b in profile.columns]
    profile = profile.reset_index().rename(columns={ccol: "cluster"})

    # PCA2D plot (presentación)
    fig_pca2d, ax = plt.subplots(figsize=(6, 5))
    x0 = Xp[:, 0]
    y0 = Xp[:, 1] if Xp.shape[1] > 1 else np.zeros_like(x0)
    ax.scatter(x0, y0, c=final_labels, alpha=0.7)
    ax.set_title(f"PCA 2D (colored by KMeans K={k_final})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig_pca2d.tight_layout()

    # Métricas finales (sobre k_final)
    sil_final = float(silhouette_score(Xp, final_labels))
    ch_final = float(calinski_harabasz_score(Xp, final_labels))
    db_final = float(davies_bouldin_score(Xp, final_labels))

    metrics = {
        "entity": str(params.get("entity", "race")),
        "n_samples": int(len(df)),
        "features_used": feature_cols,
        "pca": pca_info,
        "kmeans": {
            "k_final": int(k_final),
            "silhouette_k_final": sil_final,
            "calinski_harabasz_k_final": ch_final,
            "davies_bouldin_k_final": db_final,
            "k_search": {
                "k_min": int(params.get("k_min", 2)),
                "k_max": int(params.get("k_max", 12)),
                "ks": ks,
                "inertia_by_k": inertia_by_k if do_search else None,
                "silhouette_by_k": sil_by_k if do_search else None,
                "calinski_harabasz_by_k": ch_by_k if do_search else None,
                "davies_bouldin_by_k": db_by_k if do_search else None,
            },
            "params": {
                "random_state": kmeans_rs,
                "n_init": kmeans_n_init,
                "max_iter": kmeans_max_iter,
                "scale_numeric": scale_numeric,
            },
        },
        "gmm": {
            "bic_by_k": gmm_bic_by_k if do_search else None,
            "random_state": gmm_rs,
        },
    }

    bundle = {
        "feature_cols": feature_cols,
        "scaler": scaler,
        "pca": pca,
        "kmeans": km_final,
        "k_final": int(k_final),
        "group_key": group_key,
    }

    return labels_race, metrics, fig_elbow, fig_sil, bundle, profile, fig_pca2d


# ----------------------------
# Node 3: attach clusters back
# ----------------------------
def attach_race_clusters_to_model_input(
    model_input: pd.DataFrame,
    clustering_labels_race: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    df = model_input.copy()
    group_key = str(params.get("group_key", "raceId"))
    if group_key not in df.columns:
        raise ValueError(f"model_input debe contener '{group_key}'.")

    cluster_cols = [c for c in clustering_labels_race.columns if c != group_key]
    if len(cluster_cols) != 1:
        raise ValueError(f"clustering_labels_race debe contener exactamente 1 columna de cluster + '{group_key}'.")
    ccol = cluster_cols[0]

    out = df.merge(clustering_labels_race[[group_key, ccol]], on=group_key, how="left")

    # Si quedaran NaNs (carreras no presentes), imputamos con moda
    if out[ccol].isna().any():
        mode_val = int(pd.Series(out[ccol].dropna().astype(int)).mode().iloc[0]) if out[ccol].notna().any() else 0
        out[ccol] = out[ccol].fillna(mode_val).astype(int)

    return out
