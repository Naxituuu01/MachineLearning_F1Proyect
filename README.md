# 🏎️ Proyecto de Machine Learning – Fórmula 1 (1950–2024)
## 🧠 Kedro + CRISP-DM + MLOps (DVC · Docker · Airflow)

> Repositorio que almacena el proyecto basado en el dataset de Kaggle: **Formula 1 World Championship (1950–2024)**.

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![DVC](https://img.shields.io/badge/data_versioning-dvc-9cf?logo=data-version-control)](https://dvc.org)
[![Docker](https://img.shields.io/badge/containerized-docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Airflow](https://img.shields.io/badge/orchestrated-airflow-017CEE?logo=apacheairflow&logoColor=white)](https://airflow.apache.org/)

---

## 🎥 Demo / Video del Proyecto (inserta aquí tu enlace)
✅ **Reemplaza este link** por tu video en Google Drive:

🔗 **Video (Google Drive):** [PON_AQUI_TU_ENLACE](https://drive.google.com/)

> Sugerencia: configura el enlace como “**Cualquier persona con el vínculo puede ver**”.

---

## ✨ Resumen Ejecutivo
Este proyecto implementa un **pipeline profesional de Machine Learning** sobre datos históricos de Fórmula 1, usando **Kedro** como framework y la metodología **CRISP-DM** para estructurar el trabajo (desde el entendimiento del negocio hasta evaluación de modelos).

El foco es construir modelos **sin leakage** (solo información **pre-carrera**) para predecir desempeño competitivo:

- ✅ **Clasificación**: predecir si un piloto obtendrá **puntos** → `target_cls = (points > 0)`.
- ✅ **Regresión**: predecir el **pace (ms/lap)** para finishers (status “Finished”) → `target_reg = pace`.
- ✅ **Clustering (no supervisado)**: descubrir **arquetipos latentes** (KMeans) y generar una etiqueta de cluster reutilizable como feature (`cluster_kmeans_k4`).

El proyecto incluye prácticas MLOps: **DVC** para versionado, **Docker** para ejecución reproducible y **Airflow** para orquestación.

---

## 🏆 Resultados (KPIs)
| Tarea | Métrica | Resultado final | Modelo / enfoque |
|---|---:|---:|---|
| 🟦 **Clasificación (points > 0)** | **F1-macro** | **0.9139** | ⚡ HistGradientBoosting (tuned) + CV + SMOTE (por fold) |
| 🟩 **Regresión (PACE ms/lap)** | **R² (test 2019–2024)** | **0.8127** | 🛡️ Ensemble lineal + árbol (robusto OOT) |
| 🟪 **Clustering (KMeans)** | Silhouette (K=4) | **0.097** | 🧩 KMeans + PCA + perfilado por medianas |

> Nota: El clustering prioriza **interpretabilidad y perfilado**, no separación perfecta (alta dimensionalidad + dinámica temporal).

---

## 🎯 Objetivos del Proyecto
### Objetivo general
Desarrollar un proyecto de Machine Learning **reproducible y estructurado** con Kedro, aplicando CRISP-DM sobre datos históricos de Fórmula 1.

### Objetivos específicos
- Integrar datasets relacionales del dominio F1.
- Ejecutar EDA + validación de calidad de datos.
- Preparar datos y features con justificación técnica.
- Definir targets defendibles (clasificación/regresión).
- Entrenar ≥5 modelos por pipeline y comparar métricas.
- Reportar resultados con artefactos (tablas, gráficos, métricas).
- Orquestar ejecución con Airflow y reproducibilidad con DVC + Docker.

---

## 🧭 Metodología — CRISP-DM (entregables)
| Fase | Notebook |
|---|---|
| Business Understanding | `01_business_understanding.ipynb` |
| Data Understanding | `02_data_understanding.ipynb` |
| Data Preparation | `03_data_preparation.ipynb` |
| Modeling & Evaluation (reporting premium) | `04_classification_results.ipynb`, `05_regression_results.ipynb` |
| Unsupervised / Insights | `06_unsupervised_clustering.ipynb` |

---

## 🗂️ Estructura del proyecto
```text
proyecto_f1kedro/
├── airflow/                   # Orquestación
│   └── dags/
│       └── f1_kedro_dag.py
│
├── conf/
│   ├── base/
│   │   ├── catalog.yml
│   │   ├── parameters.yml
│   │   └── parameters_modeling.yml
│   └── local/
│       └── credentials.yml
│
├── data/                      # Capas (versionadas con DVC)
│   ├── 01_raw/
│   ├── 02_intermediate/
│   ├── 05_model_input/
│   ├── 06_models/
│   └── 08_reporting/
│
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_classification_results.ipynb
│   ├── 05_regression_results.ipynb
│   └── 06_unsupervised_clustering.ipynb
│
├── src/proyecto_f1kedro/
│   ├── pipelines/
│   │   ├── data_engineering/
│   │   ├── model_input/
│   │   ├── classification/
│   │   ├── regression/
│   │   └── clustering/
│   ├── pipeline_registry.py
│   └── settings.py
│
├── dvc.yaml
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
