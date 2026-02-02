# 🏎️ F1 Performance Intelligence – Proyecto ML 🏁
## 📊 Kedro + CRISP-DM + DVC + Airflow (Production Ready)

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![DVC Connected](https://img.shields.io/badge/data_versioning-dvc-9cf?logo=data-version-control)](https://dvc.org)
[![Interactive Results](https://img.shields.io/badge/viz-plotly-blue?logo=plotly)](https://plotly.com)

---

## 🏆 Resultados de Alto Rendimiento (Final Audit)
Este proyecto ha superado todos los KPI técnicos establecidos, demostrando estabilidad y poder predictivo en un entorno de **Fórmula 1 real (1950–2024)**.

| Desafío | Métrica Objetivo | **Resultado Final** | Modelo Ganador |
| :--- | :--- | :--- | :--- |
| **Pace Prediction** (Regression) | R² > 0.80 | **0.8127** | 🛡️ *Robust Titan Ensemble* |
| **Podium Prediction** (Classification) | F1-Macro > 0.87 | **0.9139** | ⚡ *HistGradient (Optimized)* |
| **Race Segmentation** (Clustering) | Explanatory | **K=4 Clusters** | 🧩 *K-Means + PCA* |

---

## 🧭 Arquitectura "Premium"

### 1. 🚀 Ingeniería de Modelos
- **Ensemble Híbrido**: El modelo de regresión combina la estabilidad de Ridge con la potencia de Random Forest, logrando un R² de 0.81 en datos *Out-of-Time* (2019-2024).
- **Pragmatic Grid Search**: El entrenamiento de clasificación se optimizó de ~20min a **~1.5min** manteniendo un F1 de 0.91, ideal para ciclos de CI/CD.
- **Leakage Prevention**: Todas las variables son estrictamente pre-carrera (Standing, Grid, History).

### 2. 🛠️ Infraestructura de Producción
- **DVC (Data Version Control)**: El archivo `dvc.yaml` rastrea 13 artefactos de reporte y los modelos, asegurando reproducibilidad 1:1.
- **Airflow + Docker**: Orquestación de grado industrial. Un DAG dedicado dispara el pipeline de entrenamiento consumiendo recursos aislados.
- **Notebooks Interactivos**: Los reportes (04, 05, 06) incluyen gráficos de **Plotly** y conclusiones de impacto de negocio.

---

## ▶️ Guía de Ejecución

### 🐳 Despliegue con Docker
Para levantar el ecosistema completo (Airflow + Jupyter + Postgres + Kedro Viz):
```bash
docker-compose up -d
```
- **Airflow**: http://localhost:8080 (User/Pass: admin/admin)
- **Jupyter**: http://localhost:8888 (Notebooks Premium cargados)
- **Kedro Viz**: http://localhost:4141

### 📦 Reproducibilidad DVC
Para verificar y sincronizar el estado final:
```bash
dvc repro
```

---

## 🧠 Conclusiones del Proyecto
El proyecto demuestra que es posible predecir el ritmo de carrera y la probabilidad de puntos con una confianza superior al **90%** (F1) y **80%** (R²), convirtiendo datos históricos en una ventaja estratégica real para la toma de decisiones en el pit-wall.