# 🏎️ Proyecto de Machine Learning – Fórmula 1  
## 📊 Kedro + CRISP-DM (Business Understanding · Data Understanding · Data Preparation)

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)

---

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

```text
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
