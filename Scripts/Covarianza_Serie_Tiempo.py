# -*- coding: utf-8 -*-
"""
Análisis de relación entre variación diaria de peso de colmenas
y variables meteorológicas.

Requisitos:
    - pandas
    - numpy
    - scipy

Autor: (tú)
"""

import pandas as pd
from scipy.stats import pearsonr, spearmanr

# ============================================================
# 1. Rutas a los ficheros CSV
# ============================================================

# CSV con datos de las colmenas (ya lo tienes)
HIVE_CSV = "DATOS_SERIETEM.csv"

# CSV con datos meteorológicos diarios/horarios
# (pon aquí el nombre real de tu CSV meteo)
METEO_CSV = "METEO.csv"

# ============================================================
# 2. Carga y preparación de datos de colmenas
# ============================================================

# Nombres de columnas en el CSV de colmenas
COL_YEAR   = "AÑO"
COL_DAY    = "día"
COL_HIVE   = "Colmena2"
COL_PHASE  = "PERIODO"        # 1 = pre-bloom, 2 = bloom, 3 = post-bloom
COL_HOUR   = "Horario1a24"
COL_WEIGHT = "PESO"

print("Leyendo datos de colmenas...")
hive = pd.read_csv(HIVE_CSV, sep=";", decimal=",")

# Conversión de tipos
hive["PESO_num"] = pd.to_numeric(hive[COL_WEIGHT], errors="coerce")
hive[COL_YEAR]   = pd.to_numeric(hive[COL_YEAR], errors="coerce")
hive[COL_DAY]    = pd.to_numeric(hive[COL_DAY], errors="coerce")
hive[COL_HIVE]   = pd.to_numeric(hive[COL_HIVE], errors="coerce")
hive[COL_PHASE]  = pd.to_numeric(hive[COL_PHASE], errors="coerce")
hive[COL_HOUR]   = pd.to_numeric(hive[COL_HOUR], errors="coerce")

# Filtramos filas con información imprescindible
hive_clean = hive.dropna(subset=[
    "PESO_num", COL_YEAR, COL_DAY, COL_HIVE, COL_PHASE, COL_HOUR
])

print("Filas de colmenas después de limpieza:", len(hive_clean))

# ------------------------------------------------------------
# Cálculo de métricas diarias por colmena y fase
# ------------------------------------------------------------

group_cols = [COL_YEAR, COL_DAY, COL_HIVE, COL_PHASE]

def daily_stats(g):
    """Calcula métricas diarias para un grupo (año, día, colmena, fase)."""
    max_w = g["PESO_num"].max()
    min_w = g["PESO_num"].min()
    idx_max = g["PESO_num"].idxmax()
    idx_min = g["PESO_num"].idxmin()
    hour_max = g.loc[idx_max, COL_HOUR]
    hour_min = g.loc[idx_min, COL_HOUR]
    return pd.Series({
        "daily_max_weight": max_w,
        "daily_min_weight": min_w,
        "amplitude": max_w - min_w,
        "hour_max": hour_max,
        "hour_min": hour_min,
    })

daily = hive_clean.groupby(group_cols).apply(daily_stats).reset_index()

print("Primeras filas de métricas diarias de colmenas:")
print(daily.head())

# ============================================================
# 3. Carga y preparación de datos meteorológicos
# ============================================================

print("\nLeyendo datos meteorológicos...")
# Ajusta sep y decimal según tu CSV meteo;
# si también usa ';' y coma decimal, mantenemos esto:
meteo = pd.read_csv(METEO_CSV, sep=";", decimal=",")

# Nombres de columnas del CSV meteorológico (ajusta a tu fichero)
METEO_YEAR  = "Year"
METEO_MONTH = "Month"
METEO_DAY   = "Day"

# Ejemplo de columnas meteorológicas diarias (ajusta nombres exactos):
COL_TMED    = "Có06TMed"    # temperatura media externa
COL_HUMMED  = "Có06HumMed"  # humedad media externa
COL_RAD     = "Có06Rad"     # radiación
COL_PRECIP  = "Có06Precip"  # precipitación

# Convertimos a numérico año/mes/día
for c in [METEO_YEAR, METEO_MONTH, METEO_DAY]:
    meteo[c] = pd.to_numeric(meteo[c], errors="coerce")

# Convertimos a numérico las variables meteorológicas, si existen
for c in [COL_TMED, COL_HUMMED, COL_RAD, COL_PRECIP]:
    if c in meteo.columns:
        meteo[c] = pd.to_numeric(meteo[c], errors="coerce")

meteo_clean = meteo.dropna(subset=[METEO_YEAR, METEO_DAY])

# ------------------------------------------------------------
# Resumen diario de meteorología
# ------------------------------------------------------------
# Suponiendo que los valores de C006TMed, etc., ya son diarios
# y simplemente se repiten por hora, podemos agrupar por día
# y hacer la media (o el primero, sería equivalente).

agg_dict = {}
if COL_TMED in meteo_clean.columns:
    agg_dict[COL_TMED] = "mean"
if COL_HUMMED in meteo_clean.columns:
    agg_dict[COL_HUMMED] = "mean"
if COL_RAD in meteo_clean.columns:
    agg_dict[COL_RAD] = "mean"
if COL_PRECIP in meteo_clean.columns:
    # si viene por hora, la precipitación diaria suele ser la suma
    agg_dict[COL_PRECIP] = "sum"

meteo_daily = (
    meteo_clean
    .groupby([METEO_YEAR, METEO_MONTH, METEO_DAY], as_index=False)
    .agg(agg_dict)
)

print("\nPrimeras filas de meteorología diaria:")
print(meteo_daily.head())

# ============================================================
# 4. Unión de métricas de colmena con meteorología diaria
# ============================================================

# daily tiene columnas AÑO, día; meteo_daily tiene Year, Day
# Si quieres también controlar por Month, añade la columna de mes a daily.

# En tu CSV de colmenas el mes se puede reconstruir si lo necesitas,
# pero como mínimo unimos por año y día:

merged = pd.merge(
    daily,
    meteo_daily,
    left_on=[COL_YEAR, COL_DAY],
    right_on=[METEO_YEAR, METEO_DAY],
    how="inner"
)

print("\nFilas tras la unión colmenas + meteorología:", len(merged))
print(merged[[COL_YEAR, COL_DAY, "amplitude", COL_TMED, COL_HUMMED]].head())

# ============================================================
# 5. Función auxiliar para imprimir correlaciones
# ============================================================

def corr_report(x, y, label_x, label_y):
    """Imprime correlaciones Pearson y Spearman entre dos series."""
    mask = (~pd.isna(x)) & (~pd.isna(y))
    x2 = x[mask]
    y2 = y[mask]
    n = len(x2)
    if n < 10:
        print(f"\n{label_x} vs {label_y}: muy pocos datos válidos (n={n}), no se calcula.")
        return
    r_p, p_p = pearsonr(x2, y2)
    r_s, p_s = spearmanr(x2, y2)
    print(f"\n{label_x} vs {label_y} (n={n})")
    print(f"  Pearson  r = {r_p:.3f}, p = {p_p:.3e}")
    print(f"  Spearman r = {r_s:.3f}, p = {p_s:.3e}")

# ============================================================
# 6. Correlaciones globales entre amplitud y meteorología
# ============================================================

print("\n=== Correlaciones globales (todas las fases) ===")

if COL_TMED in merged.columns:
    corr_report(merged["amplitude"], merged[COL_TMED],
                "Amplitude", "External Tmean")

if COL_HUMMED in merged.columns:
    corr_report(merged["amplitude"], merged[COL_HUMMED],
                "Amplitude", "External Humidity mean")

if COL_RAD in merged.columns:
    corr_report(merged["amplitude"], merged[COL_RAD],
                "Amplitude", "Radiation")

if COL_PRECIP in merged.columns:
    corr_report(merged["amplitude"], merged[COL_PRECIP],
                "Amplitude", "Precipitation")

# ============================================================
# 7. Correlaciones por fase (PERIODO)
# ============================================================

print("\n=== Correlaciones por fase (PERIODO) ===")
for phase in sorted(merged[COL_PHASE].dropna().unique()):
    sub = merged[merged[COL_PHASE] == phase]
    print(f"\n--- Phase {phase} ---")
    if COL_TMED in merged.columns:
        corr_report(sub["amplitude"], sub[COL_TMED],
                    "Amplitude", "External Tmean")
    if COL_HUMMED in merged.columns:
        corr_report(sub["amplitude"], sub[COL_HUMMED],
                    "Amplitude", "External Humidity mean")
    if COL_RAD in merged.columns:
        corr_report(sub["amplitude"], sub[COL_RAD],
                    "Amplitude", "Radiation")
    if COL_PRECIP in merged.columns:
        corr_report(sub["amplitude"], sub[COL_PRECIP],
                    "Amplitude", "Precipitation")

# ============================================================
# 8. (Opcional) Guardar el dataset combinado para SPSS/R
# ============================================================

merged.to_csv("merged_hive_meteo_daily.csv", index=False)
print("\nArchivo 'merged_hive_meteo_daily.csv' guardado con éxito.")