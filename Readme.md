# BeeHiveBloomDynamics

Repository associated with the manuscript:

**“Characterising Blooming-Phase Dynamics Through Intra-Day Hive-Weight Analysis”**

---

## Overview

This repository contains the datasets and Python scripts used in the study of intra-day hive-weight dynamics associated with blooming phases in honeybee colonies.

The study analyses continuous hive-weight monitoring data collected from honeybee colonies located in the experimental apiary of the University of Córdoba (Spain) during the 2016 and 2017 flowering seasons.

The objective of the work is to characterise temporal hive-weight patterns associated with previously delimited pre-bloom, bloom, and post-bloom periods using non-invasive remote monitoring data.

---

## Repository Structure

```text
BeeHiveBloomDynamics/
│
├── data/       # Raw and processed datasets
│
├── scripts/    # Preprocessing, analysis and figure-generation scripts
│
├── requirements.txt
├── LICENSE
└── README.mdData


---

---

## Dataset Variables

The main dataset (`Temporal_data.csv`) contains continuous hive-monitoring measurements collected during the 2016 and 2017 flowering seasons.

Main variables include:

| Variable | Description |
|---|---|
| `AÑO` | Monitoring year |
| `fecha` | Date of measurement |
| `Colmena2` | Hive identifier |
| `día` | Day index within the monitoring period |
| `PERIODO` | Blooming phase label (1 = Pre-bloom, 2 = Bloom, 3 = Post-bloom) |
| `hora` | Time of measurement |
| `Horario1a24` | Hour of day (1–24 format) |
| `TEMEXT` | External temperature |
| `HUMEXT` | External relative humidity |
| `PESO` | Hive weight (kg) |
| `T1CA`, `T2CA`, `T3CA` | Internal hive temperature sensors |
| `H1CA`, `H2CA`, `H3CA` | Internal hive humidity sensors |

Hive weight was recorded automatically every 5 minutes using the WBee remote monitoring system.
## Scripts Description

### `Comparacion_series.ipynb`
Notebook containing the comparative temporal analysis of intra-day hive-weight dynamics across pre-bloom, bloom, and post-bloom periods. Includes Pearson correlation, Euclidean distance, linear regression analyses, and normalized/non-normalized time-series comparisons.

### `Covarianza_colmenas_tiempo.py`
Script used to analyse inter-colony temporal variability and covariance between individual monitored hives.

### `Covarianza_condicones_ambientales.py`
Script used to evaluate relationships between hive-weight dynamics and environmental variables, including temperature, humidity, precipitation, and solar radiation.

### `Covarianza_Serie_Tiempo.py`
Script used to analyse covariance and temporal relationships between hive-weight time series associated with different blooming periods.

### `Table_data.py`
Script used to generate statistical summary tables and numerical results presented in the manuscript.

### `TestKruskalWallis-cambios.py`
Script used to perform Kruskal–Wallis statistical tests and evaluate significant differences between blooming phases.