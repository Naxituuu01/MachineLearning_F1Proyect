Repositorio que almacenara todo el proyecto referido al dataset de Kaggle llamado "Formula 1 World Championship (1950 - 2024)"
# 🏎️ Proyecto de Machine Learning – Fórmula 1  
## 📊 Kedro + CRISP-DM (Business Understanding · Data Understanding · Data Preparation)

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![DVC Connected](https://img.shields.io/badge/data_versioning-dvc-9cf?logo=data-version-control)](https://dvc.org)
[![Interactive Results](https://img.shields.io/badge/viz-plotly-blue?logo=plotly)](https://plotly.com)

## 📌 Descripción General del Proyecto

Este proyecto implementa un **pipeline profesional de Machine Learning** sobre datos históricos del **Campeonato Mundial de Fórmula 1 (1950–2020+)**, utilizando el framework **Kedro** y siguiendo rigurosamente la metodología **CRISP-DM**.

El objetivo principal es **analizar el rendimiento de pilotos y constructores** y construir **modelos predictivos baseline** que permitan:

- 🥇 Predecir si un piloto finalizará en el **podio** (clasificación).
- 📈 Estimar la **posición final** de un piloto en una carrera (regresión).

El proyecto está desarrollado con **buenas prácticas de ingeniería de datos**, código reproducible y documentación clara, orientado a un contexto **académico y profesional**.

---

## 🎯 Objetivos del Proyecto

### Objetivo General
Desarrollar un proyecto de Machine Learning estructurado con Kedro que permita analizar datos históricos de Fórmula 1 y construir modelos predictivos básicos, alineados con la metodología CRISP-DM.

### Objetivos Específicos
- Integrar múltiples datasets relacionales del dominio Fórmula 1.
- Realizar un Análisis Exploratorio de Datos (EDA) exhaustivo.
- Limpiar y preparar los datos aplicando criterios técnicos justificados.
- Construir *features* explicativas basadas en experiencia y rendimiento.
- Definir y defender variables objetivo para clasificación y regresión.
- Entrenar y evaluar modelos baseline interpretables.
- Documentar todo el proceso siguiendo estándares de la industria.

---

## 🧠 Metodología Utilizada – CRISP-DM

El proyecto implementa las **primeras tres fases de CRISP-DM**, exigidas por la evaluación:

| Fase CRISP-DM | Entregable |
|---------------|----------|
| Business Understanding | `01_business_understanding.ipynb` |
| Data Understanding | `02_data_understanding.ipynb` |
| Data Preparation | `03_data_preparation.ipynb` |

Las fases de *Modeling* y *Evaluation* se abordan a nivel de **modelos baseline**, mientras que *Deployment* queda fuera del alcance de esta evaluación.

---

## 📦 Dataset Utilizado

Los datos provienen del dataset público:

**Formula 1 World Championship (1950–2020)**  
- Autor: Rohan Rao  
- Plataforma: Kaggle  
- Fuente original: Ergast Motor Racing Database  

🔗 https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020

### Datasets principales utilizados
- `races.csv` – Información de carreras (año, circuito, fecha)
- `drivers.csv` – Información de pilotos
- `constructors.csv` – Información de equipos
- `circuits.csv` – Información de circuitos
- `results.csv` – Resultados por piloto y carrera

Los datasets se integran mediante claves relacionales (`raceId`, `driverId`, `constructorId`, `circuitId`), conformando un **modelo relacional tipo estrella**.

---

## 🧱 Estructura del Proyecto Kedro

proyecto-ml-f1/
├── conf/
│   ├── base/
│   │   ├── catalog.yml
│   │   ├── parameters.yml
│   │   └── logging.yml
│   └── local/
│       └── credentials.yml   # NO subir a Git
├── data/
│   ├── 01_raw/               # Datos originales
│   ├── 03_primary/           # Datos limpios y listos para MLς ML
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   └── 03_data_preparation.ipynb
├── src/
│   └── proyecto_ml/
│       ├── pipelines/
│       │   ├── data_engineering/
│       │   └── data_science/
│       └── pipeline_registry.py
├── README.md
├── requirements.txt
└── .gitignore


---

## 🏆 Resultados de Alto Rendimiento
Este proyecto ha superado todos los KPI técnicos establecidos, demostrando estabilidad y poder predictivo en un entorno de **Fórmula 1 real (1950–2024)**.

| Desafío | Métrica Objetivo | **Resultado Final** | Modelo Ganador |
| :--- | :--- | :--- | :--- |
| **Pace Prediction** (Regression) | R² > 0.80 | **0.8127** | 🛡️ *Robust Titan Ensemble* |
| **Podium Prediction** (Classification) | F1-Macro > 0.87 | **0.9139** | ⚡ *HistGradient (Optimized)* |
| **Race Segmentation** (Clustering) | Explanatory | **K=4 Clusters** | 🧩 *K-Means + PCA* |

---

## 🧭 Arquitectura

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
