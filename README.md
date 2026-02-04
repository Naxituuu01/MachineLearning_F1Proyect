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
├── airflow/                   # Orquestación de grado industrial
│   └── dags/
│       └── f1_kedro_dag.py   # Automatización del entrenamiento en Docker
│
├── conf/                      # El "Centro de Mando" (Configuración)
│   ├── base/
│   │   ├── catalog.yml       # Definición de datasets (In/Out)
│   │   └── parameters.yml    # Parámetros globales y de pipelines
│   └── local/
│       └── credentials.yml   # Credenciales de BD (Postgres)
│
├── data/                      # Capas de persistencia (Versionado por DVC)
│   ├── 01_raw/               # Datasets originales (Ergast Kaggle)
│   ├── 02_intermediate/      # Limpieza y tipado inicial
│   ├── 03_primary/           # Tablas maestras unificadas
│   ├── 04_feature/           # Features de ingeniería (Stating, Gaps, etc)
│   ├── 05_model_input/       # Tablas listas para modelado (Class/Reg)
│   ├── 06_models/            # Serialización de modelos (.pkl)
│   └── 08_reporting/         # Métricas, matrices y plots de auditoría
│
├── notebooks/                 # Reportes Premium e investigación
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_classification_results.ipynb
│   ├── 05_regression_results.ipynb
│   └── 06_unsupervised_clustering.ipynb
│
├── src/proyecto_f1kedro/      # El "Motor" (Lógica de Ingeniería)
│   ├── pipelines/
│   │   ├── data_engineering/   # Limpieza profunda
│   │   ├── data_understanding/ # EDA programático
│   │   ├── data_preparation/   # Creación de Master Tables
│   │   ├── model_input/        # Feature Engineering avanzado
│   │   ├── classification/     # Predicción de Podios
│   │   ├── regression/         # Predicción de Pace (Lap Time)
│   │   ├── clustering/         # Segmentación de pilotos/constructores
│   │   └── data_science/       # Pipelines de entrenamiento/validación
│   ├── pipeline_registry.py    # Punto de unión de flujos
│   └── settings.py             # Configuración de hooks y core
│
├── docker-compose.yml         # Containerización completa
├── Dockerfile                 # Imagen base Kedro/Airflow
├── .dockerignore              # Exclusiones de construcción Docker
├── .env                       # Variables de entorno y secretos
├── dvc.yaml                   # MLOps: Trazabilidad de datos
├── dvc.lock                   # Estado actual de versionado DVC
├── pyproject.toml             # Configuración central del proyecto
├── requirements.txt           # Dependencias core de producción
├── requirements-dev.txt       # Herramientas de desarrollo y testing
├── requirements-airflow.txt   # Dependencias específicas de orquestación
├── .gitignore                 # Control de versiones git
└── README.md                  # Documentación principal
```

## ⚙️ Requisitos
- Python 3.10+
- Docker + Docker Compose
- DVC configurado si se usa remote

## 🚀 Quickstart (local)
### Crear y activar entorno virtual

```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```
## Instalar dependencias
```bash
pip install -U pip
pip install -r requirements.txt
```

## Ejecutar pipelines Kedro
```bash
kedro run
```

## Ejecutar pipelines por separado:
```bash
kedro run --pipeline clustering
kedro run --pipeline classification
kedro run --pipeline regression
```

## 🧪 Artefactos generados (outputs)
### 📌 Modelos

- data/06_models/best_model_classification.pkl
- data/06_models/best_model_regression.pkl

### 📌 Reporting / auditoría técnica

- data/08_reporting/classification_metrics_summary.json
- data/08_reporting/regression_metrics_summary.json
- Tablas comparativas (CSV) + gráficos (png / html según pipeline)
- Predicciones test y feature importances (cuando aplique)

## 🐳 Ejecución con Docker (ecosistema completo)
### Levanta Airflow + servicios:
```bash
docker-compose up -d
```

## Accesos:

- Airflow: http://localhost:8080
 (admin/admin)
- Kedro Viz: http://localhost:4141
- Jupyter: http://localhost:8888

## ♻️ Reproducibilidad con DVC
### Reproducir stages del pipeline:
```bash
dvc repro
```
Ver estado:
```bash
dvc status
```

## 🧠 Diseño técnico

### ✅ Prevención de leakage
- Features construidos con shift/rolling/expanding y datos pre-carrera (standings, quali, historia).
- Split temporal defendible:
- Train: year <= 2018
- Test: year > 2018 (2019–2024)
- 
### ✅ Clasificación (points > 0)
- CV estratificado + tuning (GridSearchCV) + selección por F1-macro
- Manejo de desbalance con SMOTE dentro del pipeline por fold

###✅ Regresión (PACE ms/lap)
- Target continuo robusto y comparable entre carreras
- Métricas reportadas en pace y reconstrucción aproximada a ms
- Modelo final con excelente generalización OOT

### ✅ Clustering (KMeans)
- Selección de features numéricas sin IDs para clustering real
- K selection (Elbow + Silhouette) y K=4 por interpretabilidad
- Perfilado por medianas y variables discriminantes

## 🧾 Conclusión

### Este repositorio demuestra un flujo completo y reproducible para:
- Clasificación de probabilidad de puntos con desempeño alto (F1-macro ≈ 0.914)
- Regresión de pace con generalización OOT fuerte (R² ≈ 0.813 en 2019–2024)
- Clustering interpretativo para segmentación y feature augmentation
### Todo bajo un enfoque CRISP-DM + Kedro + MLOps, listo para evaluación académica y demostración profesional.
