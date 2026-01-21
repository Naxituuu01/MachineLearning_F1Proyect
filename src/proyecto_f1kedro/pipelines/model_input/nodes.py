import numpy as np
import pandas as pd

# --------------------
# Features base pre-carrera (seguras)
# --------------------
FEATURES_SAFE = [
    "grid",
    "year",
    "round",
    "circuitId",
    "driverId",
    "constructorId",
]

# Para históricos / CV agrupado necesitamos raceId (no como feature final)
FEATURES_SAFE_CLS = FEATURES_SAFE + ["raceId"]
FEATURES_SAFE_REG = FEATURES_SAFE + ["raceId"]


# --------------------
# Históricos Clasificación (se deja por compatibilidad del pipeline)
# --------------------
def _add_historical_features_classification(df_cls: pd.DataFrame) -> pd.DataFrame:
    required = {"year", "round", "raceId", "driverId", "constructorId", "circuitId", "target_cls"}
    missing = required - set(df_cls.columns)
    if missing:
        raise ValueError(f"Faltan columnas para históricos (cls): {sorted(missing)}")

    df_cls = df_cls.sort_values(["year", "round", "raceId"]).reset_index(drop=True)
    global_rate = float(df_cls["target_cls"].mean())

    def prev_rate_and_count(group_key: str):
        g = df_cls.groupby(group_key, sort=False)["target_cls"]
        prev_rate = (
            g.apply(lambda s: s.shift().expanding().mean())
            .reset_index(level=0, drop=True)
        )
        prev_count = g.cumcount()
        return prev_rate.fillna(global_rate), prev_count

    df_cls["driver_prev_rate"], df_cls["driver_prev_count"] = prev_rate_and_count("driverId")
    df_cls["constructor_prev_rate"], df_cls["constructor_prev_count"] = prev_rate_and_count("constructorId")
    df_cls["circuit_prev_rate"], df_cls["circuit_prev_count"] = prev_rate_and_count("circuitId")

    return df_cls


# --------------------
# Históricos Regresión (PACE pre-carrera, sin leakage)
# --------------------
def _rolling_mean_shifted(series: pd.Series, window: int) -> pd.Series:
    # shift() para que la carrera actual NO se use en su propia feature
    return series.shift().rolling(window=window, min_periods=3).mean()


def _add_historical_features_regression(df_reg: pd.DataFrame) -> pd.DataFrame:
    """
    Features históricos PRE-carrera para regresión (pace):
    - medias previas de target_reg por driver/constructor/circuit
    - counts previos (experiencia)
    - rolling means (10 y 30) por driver/constructor (suele mejorar generalización)
    Requiere raceId para orden temporal.
    """
    required = {"year", "round", "raceId", "driverId", "constructorId", "circuitId", "target_reg"}
    missing = required - set(df_reg.columns)
    if missing:
        raise ValueError(f"Faltan columnas para históricos (reg): {sorted(missing)}")

    df_reg = df_reg.sort_values(["year", "round", "raceId"]).reset_index(drop=True)

    global_mean = float(df_reg["target_reg"].mean())

    def prev_mean_and_count(group_key: str):
        g = df_reg.groupby(group_key, sort=False)["target_reg"]
        prev_mean = (
            g.apply(lambda s: s.shift().expanding().mean())
            .reset_index(level=0, drop=True)
        )
        prev_count = g.cumcount()
        return prev_mean.fillna(global_mean), prev_count

    # Expanding mean + counts (experiencia)
    df_reg["driver_prev_ms_mean"], df_reg["driver_prev_ms_count"] = prev_mean_and_count("driverId")
    df_reg["constructor_prev_ms_mean"], df_reg["constructor_prev_ms_count"] = prev_mean_and_count("constructorId")
    df_reg["circuit_prev_ms_mean"], df_reg["circuit_prev_ms_count"] = prev_mean_and_count("circuitId")

    # Rolling means (driver / constructor)
    for key, prefix in [("driverId", "driver"), ("constructorId", "constructor")]:
        g = df_reg.groupby(key, sort=False)["target_reg"]
        df_reg[f"{prefix}_roll_ms_10"] = (
            g.apply(lambda s: _rolling_mean_shifted(s, 10))
            .reset_index(level=0, drop=True)
            .fillna(global_mean)
        )
        df_reg[f"{prefix}_roll_ms_30"] = (
            g.apply(lambda s: _rolling_mean_shifted(s, 30))
            .reset_index(level=0, drop=True)
            .fillna(global_mean)
        )

    # Log experiencia para suavizar escalas
    df_reg["driver_log_exp"] = np.log1p(df_reg["driver_prev_ms_count"])
    df_reg["constructor_log_exp"] = np.log1p(df_reg["constructor_prev_ms_count"])

    return df_reg


# --------------------
# Main builder
# --------------------
def build_model_inputs_from_raw(
    results_raw: pd.DataFrame,
    races_raw: pd.DataFrame,
    targets: dict,
    data_prep: dict,
):
    """
    Construye:
      - model_input_classification (para tu pipeline cls)
      - model_input_regression (para pipeline reg)

    Clasificación:
      target_cls = 1 si points>0 else 0
      + históricos prev_rate (driver/constructor/circuit)

    Regresión:
      target_reg = milliseconds SOLO finished (statusId == finished_id)
      + históricos de pace (prev mean, counts, rolling 10/30, log_exp)
      + raceId se conserva para CV/orden (no es feature del preprocessor)
    """

    min_year = int(data_prep.get("min_year", 2000))

    required_results = {"raceId", "driverId", "constructorId", "grid", "points", "milliseconds", "statusId"}
    required_races = {"raceId", "year", "round", "circuitId"}

    miss_res = required_results - set(results_raw.columns)
    miss_rac = required_races - set(races_raw.columns)
    if miss_res:
        raise ValueError(f"Faltan columnas en results_raw: {sorted(miss_res)}")
    if miss_rac:
        raise ValueError(f"Faltan columnas en races_raw: {sorted(miss_rac)}")

    races_sel = races_raw[["raceId", "year", "round", "circuitId"]].copy()
    df = results_raw.merge(races_sel, on="raceId", how="left")

    df = df[df["year"] >= min_year].copy()

    for col in [
        "raceId", "grid", "points", "milliseconds", "statusId",
        "year", "round", "circuitId", "driverId", "constructorId"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------
    # Clasificación (se mantiene para no romper tu pipeline)
    # --------------------
    df_cls = df[FEATURES_SAFE_CLS + ["points"]].dropna(subset=FEATURES_SAFE_CLS).copy()
    df_cls["target_cls"] = (df_cls["points"] > 0).astype(int)
    df_cls = df_cls.drop(columns=["points"])
    df_cls = _add_historical_features_classification(df_cls)
    df_cls = df_cls.drop(columns=["raceId"])  # raceId solo orden; no feature final

    # --------------------
    # Regresión (Finished only)
    # --------------------
    finished_id = int(targets["regression"]["status_finished_id"])
    target_col = str(targets["regression"]["target_column"])  # "milliseconds"

    df_reg = df.loc[df["statusId"] == finished_id, FEATURES_SAFE_REG + [target_col]].copy()
    df_reg = df_reg.dropna(subset=FEATURES_SAFE_REG + [target_col])
    df_reg = df_reg.rename(columns={target_col: "target_reg"})

    # Históricos de pace sin leakage
    df_reg = _add_historical_features_regression(df_reg)

    # OJO: raceId se mantiene para CV/orden (no lo uses como feature en preprocessor)
    return df_cls, df_reg
