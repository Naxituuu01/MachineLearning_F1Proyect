import pandas as pd

def drop_columns(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Elimina columnas especificadas en parámetros."""
    columns_to_drop = params.get("data_preparation", {}).get("drop_columns", [])
    print(f"🧹 Eliminando columnas innecesarias: {columns_to_drop}")
    df_cleaned = df.drop(columns=columns_to_drop, errors="ignore")
    print(f"✅ Columnas finales: {df_cleaned.columns.tolist()[:10]}...")
    return df_cleaned

def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Llena valores faltantes con mediana/moda."""
    print("🔧 Llenando valores faltantes...")

    # Numéricos → mediana
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Categóricos → moda
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print("✅ Valores faltantes imputados.")
    return df

def create_features(races_filled: pd.DataFrame, drivers: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Integra datasets y genera variables derivadas básicas."""
    print("🚀 Creando nuevas características...")

    df = results.merge(races_filled, on="raceId", how="left")
    df = df.merge(drivers, on="driverId", how="left")

    # Convertir fechas
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Edad del piloto
    df["driver_age"] = df["date"].dt.year - df["dob"].dt.year

    # Puntos acumulados por temporada
    df["season_points"] = df.groupby(["year", "driverId"])["points"].cumsum()

    print(f"✅ Dataset integrado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

import os
import pandas as pd

def save_clean_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Guarda los datos limpios en formato CSV con codificación segura (Windows compatible)."""
    output_path = params.get("data_preparation", {}).get("output_path", "data/03_primary/clean_f1_data.csv")

    print(f"💾 Preparando datos limpios para guardar en: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Guardado seguro con codificación UTF-8-SIG
        df.to_csv(output_path, index=False, encoding="utf-8-sig", errors="replace")
        print("✅ Archivo guardado correctamente en UTF-8-SIG (compatible con Excel y Windows).")
    except UnicodeEncodeError as e:
        print(f"⚠️ Error de codificación detectado: {e}")
        print("🔁 Intentando guardar con codificación ISO-8859-1...")
        df.to_csv(output_path, index=False, encoding="ISO-8859-1", errors="replace")
        print("✅ Guardado exitoso usando ISO-8859-1 como respaldo.")
    except Exception as e:
        print(f"❌ Error inesperado al guardar: {e}")

    return df
