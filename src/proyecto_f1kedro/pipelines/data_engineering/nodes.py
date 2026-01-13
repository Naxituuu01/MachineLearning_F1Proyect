import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Integración de datasets
# --------------------------------------------------
def integrate_datasets(
    results: pd.DataFrame,
    races: pd.DataFrame,
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    circuits: pd.DataFrame,
) -> pd.DataFrame:
    """Integra los datasets relacionales de Fórmula 1."""
    races["year"] = pd.to_numeric(races["year"], errors="coerce")

    df = (
        results
        .merge(
            races[["raceId", "year", "circuitId"]],
            on="raceId",
            how="inner"
        )
        .merge(
            drivers[["driverId", "driverRef", "nationality"]],
            on="driverId",
            how="left"
        )
        .merge(
            constructors[["constructorId", "name"]]
            .rename(columns={"name": "constructorName"}),
            on="constructorId",
            how="left"
        )
        .merge(
            circuits[["circuitId", "country"]],
            on="circuitId",
            how="left"
        )
    )

    return df

# --------------------------------------------------
# 2. Feature engineering + targets
# --------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye variables explicativas y targets."""
    # Targets
    df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
    df["is_podium"] = (df["positionOrder"] <= 3).astype(int)

    # Experiencia
    df["driver_experience"] = df.groupby("driverId")["raceId"].transform("count")
    df["constructor_experience"] = df.groupby("constructorId")["raceId"].transform("count")

    # Conversión segura
    numeric_cols = ["grid", "laps", "milliseconds", "fastestLapSpeed"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ritmo promedio
    df["pace_ms_per_lap"] = np.where(
        (df["laps"] > 0) & (df["milliseconds"].notna()),
        df["milliseconds"] / df["laps"],
        np.nan
    )

    return df

# --------------------------------------------------
# 3. Selección final de features
# --------------------------------------------------
def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona variables finales y elimina registros incompletos."""
    features = [
        "grid",
        "laps",
        "fastestLapSpeed",
        "driver_experience",
        "constructor_experience",
        "pace_ms_per_lap"
    ]

    df_model = df[features + ["is_podium", "positionOrder"]].dropna()
    return df_model
