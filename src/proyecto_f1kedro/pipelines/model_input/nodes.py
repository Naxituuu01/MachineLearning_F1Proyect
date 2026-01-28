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

FEATURES_SAFE_CLS = FEATURES_SAFE + ["raceId"]
FEATURES_SAFE_REG = FEATURES_SAFE + ["raceId"]


# --------------------
# Utils
# --------------------
def _to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _parse_time_to_ms(x) -> float:
    """Convierte '1:23.456' o '83.456' a ms. Retorna NaN si no parsea (incluye '\\N')."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s in {"\\N", "nan", "NaN", ""}:
        return np.nan
    try:
        if ":" in s:
            mm, rest = s.split(":")
            sec = float(rest)
            total_sec = float(mm) * 60.0 + sec
        else:
            total_sec = float(s)
        return total_sec * 1000.0
    except Exception:
        return np.nan


def _rolling_mean_shifted(series: pd.Series, window: int) -> pd.Series:
    """Rolling mean con shift(1) para evitar leakage."""
    return series.shift(1).rolling(window=window, min_periods=max(3, window // 3)).mean()


def _rolling_std_shifted(series: pd.Series, window: int) -> pd.Series:
    """Rolling std con shift(1) para evitar leakage."""
    return series.shift(1).rolling(window=window, min_periods=max(3, window // 3)).std()


# --------------------
# Clasificación: históricos (igual que antes)
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
# Regresión: históricos (pace + resid)
# --------------------
def _add_historical_features_regression(df_reg: pd.DataFrame) -> pd.DataFrame:
    """
    Históricos sin leakage:
    - sobre target_reg (pace): prev_mean + counts + rolling mean
    - históricos adicionales sobre target_resid (si existe): prev mean + prev std + roll mean/std
    """
    # Defensa: elimina columnas duplicadas (esto evita bugs de pandas donde df["grid"] devuelve DataFrame)
    df_reg = df_reg.loc[:, ~df_reg.columns.duplicated()].copy()

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

    # Históricos sobre pace (target_reg)
    df_reg["driver_prev_pace_mean"], df_reg["driver_prev_pace_count"] = prev_mean_and_count("driverId")
    df_reg["constructor_prev_pace_mean"], df_reg["constructor_prev_pace_count"] = prev_mean_and_count("constructorId")
    df_reg["circuit_prev_pace_mean"], df_reg["circuit_prev_pace_count"] = prev_mean_and_count("circuitId")

    for key, prefix in [("driverId", "driver"), ("constructorId", "constructor"), ("circuitId", "circuit")]:
        g = df_reg.groupby(key, sort=False)["target_reg"]
        df_reg[f"{prefix}_pace_roll_mean_5"] = (
            g.apply(lambda s: _rolling_mean_shifted(s, 5))
            .reset_index(level=0, drop=True)
            .fillna(global_mean)
        )
        df_reg[f"{prefix}_pace_roll_mean_10"] = (
            g.apply(lambda s: _rolling_mean_shifted(s, 10))
            .reset_index(level=0, drop=True)
            .fillna(global_mean)
        )
        df_reg[f"{prefix}_pace_roll_mean_30"] = (
            g.apply(lambda s: _rolling_mean_shifted(s, 30))
            .reset_index(level=0, drop=True)
            .fillna(global_mean)
        )

    df_reg["driver_log_exp"] = np.log1p(df_reg["driver_prev_pace_count"])
    df_reg["constructor_log_exp"] = np.log1p(df_reg["constructor_prev_pace_count"])
    df_reg["circuit_log_exp"] = np.log1p(df_reg["circuit_prev_pace_count"])

    # -------------------------------------------------------
    # Históricos sobre target_resid (si existe)
    # -------------------------------------------------------
    if "target_resid" in df_reg.columns:
        df_reg["target_resid"] = pd.to_numeric(df_reg["target_resid"], errors="coerce").fillna(0.0)
        global_resid_mean = float(df_reg["target_resid"].mean())

        for key, prefix in [("driverId", "driver"), ("constructorId", "constructor"), ("circuitId", "circuit")]:
            g = df_reg.groupby(key, sort=False)["target_resid"]

            df_reg[f"{prefix}_prev_resid_mean"] = (
                g.apply(lambda s: s.shift().expanding().mean())
                .reset_index(level=0, drop=True)
                .fillna(global_resid_mean)
            )

            df_reg[f"{prefix}_prev_resid_std"] = (
                g.apply(lambda s: s.shift().expanding().std())
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )

            df_reg[f"{prefix}_resid_roll_mean_10"] = (
                g.apply(lambda s: _rolling_mean_shifted(s, 10))
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )
            df_reg[f"{prefix}_resid_roll_std_10"] = (
                g.apply(lambda s: _rolling_std_shifted(s, 10))
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )

    return df_reg


# --------------------
# Qualifying features (pre-carrera)
# --------------------
def _build_qualy_features(qualifying_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna:
      raceId, driverId, quali_position, quali_best_ms, quali_has_q2, quali_has_q3, quali_gap_to_pole_ms
    """
    q = qualifying_raw.copy()
    needed = {"raceId", "driverId", "position", "q1", "q2", "q3"}
    missing = needed - set(q.columns)
    if missing:
        raise ValueError(f"Faltan columnas en qualifying_raw: {sorted(missing)}")

    q = _to_num(q, ["raceId", "driverId", "position"])
    q["q1_ms"] = q["q1"].apply(_parse_time_to_ms)
    q["q2_ms"] = q["q2"].apply(_parse_time_to_ms)
    q["q3_ms"] = q["q3"].apply(_parse_time_to_ms)

    q["quali_has_q2"] = q["q2_ms"].notna().astype(int)
    q["quali_has_q3"] = q["q3_ms"].notna().astype(int)

    q["quali_best_ms"] = q[["q1_ms", "q2_ms", "q3_ms"]].min(axis=1, skipna=True)
    q = q.dropna(subset=["raceId", "driverId"])

    pole = q.groupby("raceId", sort=False)["quali_best_ms"].min().rename("pole_best_ms").reset_index()
    q = q.merge(pole, on="raceId", how="left")
    q["quali_gap_to_pole_ms"] = q["quali_best_ms"] - q["pole_best_ms"]

    out = q[
        ["raceId", "driverId", "position", "quali_best_ms", "quali_has_q2", "quali_has_q3", "quali_gap_to_pole_ms"]
    ].copy()
    out = out.rename(columns={"position": "quali_position"})
    return out


def _add_race_level_qualy_stats(df_reg: pd.DataFrame) -> pd.DataFrame:
    """Stats intra-carrera usando quali_best_ms (pre-carrera)."""
    if "quali_best_ms" not in df_reg.columns:
        return df_reg

    g = df_reg.groupby("raceId", sort=False)["quali_best_ms"]
    mean_q = g.transform("mean")
    std_q = g.transform("std").replace(0, np.nan)
    min_q = g.transform("min")
    max_q = g.transform("max")

    df_reg["quali_zscore_in_race"] = ((df_reg["quali_best_ms"] - mean_q) / std_q).fillna(0.0).clip(-5, 5)
    denom = (max_q - min_q).replace(0, np.nan)
    df_reg["quali_rank_pct_in_race"] = ((df_reg["quali_best_ms"] - min_q) / denom).fillna(0.0).clip(0, 1)
    return df_reg


def _add_qualy_rollings(df_reg: pd.DataFrame) -> pd.DataFrame:
    """Rollings de quali_best_ms (shifted) por driver/constructor/circuit."""
    if "quali_best_ms" not in df_reg.columns:
        return df_reg

    df_reg = df_reg.sort_values(["year", "round", "raceId"]).reset_index(drop=True)
    global_q = float(np.nanmedian(df_reg["quali_best_ms"])) if df_reg["quali_best_ms"].notna().any() else np.nan

    for key, prefix in [("driverId", "drv"), ("constructorId", "cons"), ("circuitId", "cir")]:
        g = df_reg.groupby(key, sort=False)["quali_best_ms"]
        df_reg[f"{prefix}_quali_roll_mean_5"] = (
            g.apply(lambda s: _rolling_mean_shifted(s, 5))
            .reset_index(level=0, drop=True)
            .fillna(global_q)
        )
        df_reg[f"{prefix}_quali_roll_mean_10"] = (
            g.apply(lambda s: _rolling_mean_shifted(s, 10))
            .reset_index(level=0, drop=True)
            .fillna(global_q)
        )
    return df_reg

# --------------------
# Standings PRE-carrera (shift)
# --------------------
def _build_driver_standings_prev(driver_standings_raw: pd.DataFrame, races_raw: pd.DataFrame) -> pd.DataFrame:
    """
    standings suele ser post-carrera => shift(1) para pre-carrera.
    Retorna: raceId, driverId, drv_points_prev, drv_pos_prev, drv_wins_prev
    """
    ds = driver_standings_raw.copy()
    needed = {"raceId", "driverId", "points", "position", "wins"}
    missing = needed - set(ds.columns)
    if missing:
        raise ValueError(f"Faltan columnas en driver_standings_raw: {sorted(missing)}")

    ds = _to_num(ds, ["raceId", "driverId", "points", "position", "wins"])

    race_info = races_raw[["raceId", "year", "round"]].copy()
    race_info = _to_num(race_info, ["raceId", "year", "round"])
    ds = ds.merge(race_info, on="raceId", how="left")

    ds = ds.sort_values(["driverId", "year", "round", "raceId"]).reset_index(drop=True)

    for col, new in [("points", "drv_points_prev"), ("position", "drv_pos_prev"), ("wins", "drv_wins_prev")]:
        ds[new] = ds.groupby("driverId", sort=False)[col].shift(1)

    ds["drv_points_prev"] = ds["drv_points_prev"].fillna(0.0)
    ds["drv_wins_prev"] = ds["drv_wins_prev"].fillna(0.0)
    ds["drv_pos_prev"] = ds["drv_pos_prev"].fillna(ds["drv_pos_prev"].median())

    return ds[["raceId", "driverId", "drv_points_prev", "drv_pos_prev", "drv_wins_prev"]]

def _build_constructor_standings_prev(constructor_standings_raw: pd.DataFrame, races_raw: pd.DataFrame) -> pd.DataFrame:
    """shift(1) por constructorId para pre-carrera."""
    cs = constructor_standings_raw.copy()
    needed = {"raceId", "constructorId", "points", "position", "wins"}
    missing = needed - set(cs.columns)
    if missing:
        raise ValueError(f"Faltan columnas en constructor_standings_raw: {sorted(missing)}")

    cs = _to_num(cs, ["raceId", "constructorId", "points", "position", "wins"])

    race_info = races_raw[["raceId", "year", "round"]].copy()
    race_info = _to_num(race_info, ["raceId", "year", "round"])
    cs = cs.merge(race_info, on="raceId", how="left")

    cs = cs.sort_values(["constructorId", "year", "round", "raceId"]).reset_index(drop=True)

    for col, new in [("points", "cons_points_prev"), ("position", "cons_pos_prev"), ("wins", "cons_wins_prev")]:
        cs[new] = cs.groupby("constructorId", sort=False)[col].shift(1)

    cs["cons_points_prev"] = cs["cons_points_prev"].fillna(0.0)
    cs["cons_wins_prev"] = cs["cons_wins_prev"].fillna(0.0)
    cs["cons_pos_prev"] = cs["cons_pos_prev"].fillna(cs["cons_pos_prev"].median())

    return cs[["raceId", "constructorId", "cons_points_prev", "cons_pos_prev", "cons_wins_prev"]]
# --------------------
# MAIN builder
# --------------------
def build_model_inputs_from_raw(
    results_raw: pd.DataFrame,
    races_raw: pd.DataFrame,
    circuits_raw: pd.DataFrame,
    qualifying_raw: pd.DataFrame,
    driver_standings_raw: pd.DataFrame,
    constructor_standings_raw: pd.DataFrame,
    targets: dict,
    data_prep: dict,
):
    min_year = int(data_prep.get("min_year", 2000))

    required_results = {"raceId", "driverId", "constructorId", "grid", "points", "milliseconds", "statusId", "laps"}
    required_races = {"raceId", "year", "round", "circuitId"}

    miss_res = required_results - set(results_raw.columns)
    miss_rac = required_races - set(races_raw.columns)
    if miss_res:
        raise ValueError(f"Faltan columnas en results_raw: {sorted(miss_res)}")
    if miss_rac:
        raise ValueError(f"Faltan columnas en races_raw: {sorted(miss_rac)}")

    # ---- Merge base results + races ----
    races_sel = races_raw[["raceId", "year", "round", "circuitId"]].copy()
    df = results_raw.merge(races_sel, on="raceId", how="left")

    # circuit meta opcional (si está)
    if circuits_raw is not None and {"circuitId", "lat", "lng", "alt", "country"}.issubset(circuits_raw.columns):
        circuits_sel = circuits_raw[["circuitId", "lat", "lng", "alt", "country"]].copy()
        df = df.merge(circuits_sel, on="circuitId", how="left")

    # filtros y tipos numéricos
    df = df[df["year"] >= min_year].copy()
    df = _to_num(
        df,
        ["raceId", "grid", "points", "milliseconds", "statusId", "year", "round",
         "circuitId", "driverId", "constructorId", "laps"],
    )
    # --------------------
    # Clasificación (intacto)
    # --------------------
    df_cls = df[FEATURES_SAFE_CLS + ["points"]].dropna(subset=FEATURES_SAFE_CLS).copy()
    df_cls["target_cls"] = (df_cls["points"] > 0).astype(int)
    df_cls = df_cls.drop(columns=["points"])
    df_cls = _add_historical_features_classification(df_cls)

    # --------------------
    # Regresión: Finished + target pace (ms/lap)
    # --------------------
    finished_id = int(targets["regression"]["status_finished_id"])
    target_col = str(targets["regression"]["target_column"])  # "milliseconds"
    pace_denominator = str(targets["regression"].get("pace_denominator", "laps_expected")).lower()
    if pace_denominator not in {"laps_expected", "laps_actual"}:
        raise ValueError(f"pace_denominator inválido: {pace_denominator}. Usa 'laps_expected' o 'laps_actual'.")
    
    # IMPORTANTE: no duplicar columnas, y asegurar subset completo
    df_reg = df.loc[df["statusId"] == finished_id, FEATURES_SAFE_REG + [target_col, "laps"]].copy()
    df_reg = df_reg.dropna(subset=FEATURES_SAFE_REG + [target_col, "laps"])

    df_reg = df_reg.rename(columns={target_col: "target_ms", "laps": "laps_actual"})  # laps_actual = observado

    # Limpieza fuerte
    df_reg["target_ms"] = pd.to_numeric(df_reg["target_ms"], errors="coerce")
    df_reg["laps_actual"] = pd.to_numeric(df_reg["laps_actual"], errors="coerce")
    df_reg = df_reg.dropna(subset=["target_ms", "laps_actual"])

    df_reg = df_reg[(df_reg["laps_actual"] >= 10) & (df_reg["laps_actual"] <= 100)]
    df_reg = df_reg[(df_reg["target_ms"] >= 600_000) & (df_reg["target_ms"] <= 20_000_000)]
   
   
    # -------------------------------------------------
    # Pace observado (solo para filtros de calidad)
    # -------------------------------------------------
    df_reg["pace_obs"] = df_reg["target_ms"] / df_reg["laps_actual"]
    df_reg = df_reg[(df_reg["pace_obs"] >= 40_000) & (df_reg["pace_obs"] <= 250_000)]

    # --------------------
    # Merges pre-carrera: QUALY + STANDINGS
    # --------------------
    qualy_feat = _build_qualy_features(qualifying_raw)
    drv_prev = _build_driver_standings_prev(driver_standings_raw, races_raw)
    cons_prev = _build_constructor_standings_prev(constructor_standings_raw, races_raw)

    df_reg = df_reg.merge(qualy_feat, on=["raceId", "driverId"], how="left")
    df_reg = df_reg.merge(drv_prev, on=["raceId", "driverId"], how="left")
    df_reg = df_reg.merge(cons_prev, on=["raceId", "constructorId"], how="left")

    # Defensa: elimina columnas duplicadas después de merges
    df_reg = df_reg.loc[:, ~df_reg.columns.duplicated()].copy()

    # --------------------
    # Orden temporal (para baseline/históricos)
    # --------------------
    df_reg = df_reg.sort_values(["year", "round", "raceId"]).reset_index(drop=True)

    # Asegurar numéricos base (evita mezclas str/int por merges o lecturas)
    df_reg = _to_num(df_reg, ["grid", "year", "round", "raceId", "driverId", "constructorId", "circuitId"])

    # --------------------
    # Contexto de carrera (pre-carrera)
    # --------------------
    n_drivers_race = df_reg.groupby("raceId", sort=False)["driverId"].transform("nunique").replace(0, np.nan)
    df_reg["n_drivers_race"] = pd.to_numeric(n_drivers_race, errors="coerce")
    df_reg["n_drivers_race"] = df_reg["n_drivers_race"].fillna(df_reg["n_drivers_race"].median())

    # grid_pct_in_race (robusto a edge cases)
    df_reg["grid"] = pd.to_numeric(df_reg["grid"], errors="coerce")
    denom = (df_reg["n_drivers_race"] - 1).replace(0, np.nan)
    df_reg["grid_pct_in_race"] = (
        ((df_reg["grid"] - 1) / denom)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.5)
        .clip(0, 1)
    )

    # laps_expected (seguro): mediana histórica por circuito con shift(1)
    g_laps = df_reg.groupby("circuitId", sort=False)["laps_actual"]
    df_reg["laps_expected"] = (
        g_laps.apply(lambda s: s.shift(1).expanding().median())
        .reset_index(level=0, drop=True)
        .fillna(float(np.nanmedian(df_reg["laps_actual"].values)))
    )
    
    # -------------------------------------------------
    # Target pace FINAL (sin leakage):
    # - Si pace_denominator = laps_expected -> target_reg usa laps_expected (pre-carrera)
    # - Si pace_denominator = laps_actual   -> target_reg usa laps_actual (solo si quieres comparar)
    # -------------------------------------------------
    if pace_denominator == "laps_expected":
        denom_pace = pd.to_numeric(df_reg["laps_expected"], errors="coerce")
    else:
        denom_pace = pd.to_numeric(df_reg["laps_actual"], errors="coerce")

    denom_pace = denom_pace.replace(0, np.nan)
    denom_pace = denom_pace.fillna(float(np.nanmedian(df_reg["laps_actual"].values)))

    df_reg["target_reg"] = df_reg["target_ms"] / denom_pace
    df_reg = df_reg[(df_reg["target_reg"] >= 40_000) & (df_reg["target_reg"] <= 250_000)]

    # ya no necesitamos pace_obs
    df_reg = df_reg.drop(columns=["pace_obs"], errors="ignore")


    # Feature pre-carrera para el modelo (proxy): usar laps_expected como "laps"
    df_reg["laps"] = pd.to_numeric(df_reg["laps_expected"], errors="coerce")
    df_reg["laps"] = df_reg["laps"].fillna(df_reg["laps"].median())
    df_reg["laps"] = df_reg["laps"].clip(10, 100)

    # season progress (pre-carrera)
    max_round_year = df_reg.groupby("year", sort=False)["round"].transform("max").replace(0, np.nan)
    df_reg["season_progress"] = (df_reg["round"] / max_round_year).fillna(0.0).clip(0, 1)

    # era bins (captura cambios reglamentarios)
    bins = [1999, 2005, 2010, 2014, 2017, 2021, 2025]
    df_reg["era_bin"] = pd.cut(df_reg["year"], bins=bins, labels=False, include_lowest=True).fillna(0).astype(int)

    # experiencia driver en circuito (sin leakage)
    df_reg["driver_circuit_prev_count"] = df_reg.groupby(["driverId", "circuitId"], sort=False).cumcount().astype(int)
    df_reg["driver_circuit_log"] = np.log1p(df_reg["driver_circuit_prev_count"])

    # --------------------
    # Baseline por circuito (sin leakage) + residual
    # --------------------
    global_mean = float(df_reg["target_reg"].mean())
    g_c = df_reg.groupby("circuitId", sort=False)["target_reg"]

    df_reg["circuit_baseline_pace"] = (
        g_c.apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
        .fillna(global_mean)
    )
    df_reg["target_resid"] = (df_reg["target_reg"] - df_reg["circuit_baseline_pace"]).fillna(0.0)

    # --------------------
    # Tendencia por temporada (drift)
    # --------------------
    race_level = df_reg.groupby(["year", "round", "raceId"], sort=False).agg(
        race_mean_pace=("target_reg", "mean"),
        race_mean_resid=("target_resid", "mean"),
    ).reset_index().sort_values(["year", "round", "raceId"])

    race_level["season_prev_mean_pace"] = (
        race_level.groupby("year", sort=False)["race_mean_pace"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
        .fillna(global_mean)
    )
    race_level["season_prev_mean_resid"] = (
        race_level.groupby("year", sort=False)["race_mean_resid"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    df_reg = df_reg.merge(
        race_level[["raceId", "season_prev_mean_pace", "season_prev_mean_resid"]],
        on="raceId",
        how="left",
    )

    # --------------------
    # Imputaciones seguras
    # --------------------
    for c in ["quali_has_q2", "quali_has_q3"]:
        if c in df_reg.columns:
            df_reg[c] = df_reg[c].fillna(0).astype(int)

    for c in ["drv_points_prev", "drv_wins_prev", "cons_points_prev", "cons_wins_prev"]:
        if c in df_reg.columns:
            df_reg[c] = df_reg[c].fillna(0.0)

    for c in ["drv_pos_prev", "cons_pos_prev"]:
        if c in df_reg.columns:
            df_reg[c] = df_reg[c].fillna(df_reg[c].median())

    # stats intra-carrera de qualy
    df_reg = _add_race_level_qualy_stats(df_reg)

    # features derivadas de qualy (si existe)
    if "quali_gap_to_pole_ms" in df_reg.columns and "quali_best_ms" in df_reg.columns:
        denom = (df_reg["quali_best_ms"].abs() + 1.0)
        df_reg["quali_gap_pct"] = (df_reg["quali_gap_to_pole_ms"] / denom).clip(-1.0, 5.0)

    if "grid" in df_reg.columns and "quali_position" in df_reg.columns:
        df_reg["grid_minus_quali"] = (df_reg["grid"] - df_reg["quali_position"]).clip(-40, 40)

    # standings normalizados por rondas completadas
    rounds_done = (df_reg["round"] - 1).clip(lower=1)
    for col, out in [
        ("drv_points_prev", "drv_points_per_round_prev"),
        ("drv_wins_prev", "drv_wins_per_round_prev"),
        ("cons_points_prev", "cons_points_per_round_prev"),
        ("cons_wins_prev", "cons_wins_per_round_prev"),
    ]:
        if col in df_reg.columns:
            df_reg[out] = (df_reg[col] / rounds_done).fillna(0.0)

    # interacción categórica driver+constructor
    df_reg["driver_constructor"] = (
        df_reg["driverId"].astype("Int64").astype(str) + "_" + df_reg["constructorId"].astype("Int64").astype(str)
    )

    # rollings de qualy (shifted)
    df_reg = _add_qualy_rollings(df_reg)

    # históricos pace + históricos resid
    df_reg = _add_historical_features_regression(df_reg)

    # --------------------
    # IMPORTANTÍSIMO:
    # - 'laps_actual' NO debe usarse como feature en el modelo (es post-carrera).
    # - usar 'laps_expected' y/o 'laps' proxy pre-carrera.
    # --------------------
    return df_cls, df_reg
