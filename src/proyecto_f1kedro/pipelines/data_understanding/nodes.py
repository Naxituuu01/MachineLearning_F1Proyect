import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Configuración de estilo visual
sns.set_style("whitegrid")
plt.rcParams.update({
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "figure.figsize": (8, 4),
    "axes.edgecolor": "#E0E0E0",
    "axes.linewidth": 1.2
})

# ------------------------------------------------------------------------------
# NODO 1 - Cargar dataset
# ------------------------------------------------------------------------------

def load_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Valida y retorna el dataset cargado desde el catálogo."""
    if df.empty:
        raise ValueError("❌ El dataset está vacío.")
    print(f"✅ Dataset cargado correctamente: {df.shape[0]} filas, {df.shape[1]} columnas.")
    print(f"📋 Columnas: {', '.join(df.columns[:8])}..." if len(df.columns) > 8 else f"📋 Columnas: {', '.join(df.columns)}")
    return df

# ------------------------------------------------------------------------------
# NODO 2 - Resumen estadístico (sin datetime_is_numeric)
# ------------------------------------------------------------------------------

def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Genera y muestra un resumen estadístico textual del dataset."""
    print("\n📈 RESUMEN ESTADÍSTICO GENERAL")
    print("-" * 80)

    # Compatibilidad: sin datetime_is_numeric
    try:
        summary = df.describe(include="all").transpose()
    except Exception:
        summary = df.describe().transpose()

    # Mostrar resumen textual
    for col in summary.index[:10]:
        col_info = summary.loc[col]
        print(f"\n🔹 {col}")

        if pd.api.types.is_numeric_dtype(df[col]):
            mean = col_info.get("mean", "N/A")
            std = col_info.get("std", "N/A")
            print(f"   - Media: {mean:.2f}" if pd.notna(mean) else "   - Media: N/A")
            print(f"   - Desviación estándar: {std:.2f}" if pd.notna(std) else "   - Desviación estándar: N/A")
            print(f"   - Mínimo: {col_info.get('min', 'N/A')}, Máximo: {col_info.get('max', 'N/A')}")
        else:
            unique = col_info.get("unique", "N/A")
            top = col_info.get("top", "N/A")
            freq = col_info.get("freq", "N/A")
            print(f"   - Valores únicos: {unique}")
            print(f"   - Valor más frecuente: {top} ({freq} ocurrencias)")

    print("\n📊 Totales:")
    print(f"   - Filas: {df.shape[0]}")
    print(f"   - Columnas: {df.shape[1]}")
    print(f"   - Numéricas: {len(df.select_dtypes(include='number').columns)}")
    print(f"   - Categóricas: {len(df.select_dtypes(include='object').columns)}")
    print("-" * 80)
    print("✅ Resumen textual completado.\n")

    return summary

# ------------------------------------------------------------------------------
# NODO 3 - Valores faltantes
# ------------------------------------------------------------------------------

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Identifica valores faltantes en el dataset y los resume en texto."""
    missing = df.isnull().sum()
    total_missing = missing[missing > 0]

    print("\n🔍 VERIFICACIÓN DE VALORES FALTANTES")
    print("-" * 80)

    if total_missing.empty:
        print("✅ No se detectaron valores faltantes.")
    else:
        print(f"⚠️ Se detectaron {len(total_missing)} columnas con valores faltantes:\n")
        for col, val in total_missing.items():
            pct = (val / len(df)) * 100
            print(f"   - {col}: {val} valores faltantes ({pct:.2f}%)")

    print("-" * 80)
    return missing

# ------------------------------------------------------------------------------
# NODO 4 - Distribuciones (opcional, sin mostrar gráficos en ejecución normal)
# ------------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame, show_plots: bool = False) -> None:
    """
    Grafica distribuciones de columnas numéricas.
    Por defecto, no muestra gráficos (para ejecución vía kedro run).
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    print("\n📊 ANÁLISIS DE DISTRIBUCIONES NUMÉRICAS")
    print("-" * 80)

    if not numeric_cols:
        print("⚠️ No hay columnas numéricas para graficar.")
        return

    if not show_plots:
        print("ℹ️ Modo silencioso: gráficos no mostrados (ejecuta con show_plots=True para verlos).")
        return

    for col in numeric_cols[:8]:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[col].dropna(), kde=True, color="#007ACC", bins=25)
        plt.title(f"Distribución de {col}", fontsize=14, fontweight="bold")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.gca().xaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
        plt.tight_layout()
        plt.show()
        plt.close()

    print("✅ Distribuciones generadas correctamente.\n")
