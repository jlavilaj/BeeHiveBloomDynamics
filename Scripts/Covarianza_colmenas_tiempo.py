#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análisis de la relación entre amplitud diaria de peso de colmenas
y variables meteorológicas diarias descargadas de la API Historical Weather
de Open-Meteo (https://open-meteo.com/en/docs/historical-weather-api).

- Lee DATOS_SERIETEM.csv
- Calcula amplitud diaria por colmena y día
- Descarga meteorología diaria para Córdoba (España)
- Fusiona ambos conjuntos y calcula correlaciones y regresión múltiple
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm

# ---------------------------------------------------------------------
# 1. Parámetros de entrada
# ---------------------------------------------------------------------

# Ruta al CSV de colmenas (ajusta si hace falta)
HIVE_CSV_PATH = "DATOS_SERIETEM.csv"

# Coordenadas aproximadas del apiario de Córdoba (ajustables)
LATITUDE = 37.923  # 37°55' aprox.
LONGITUDE = -4.724  # 4°43' O aprox.

# Zona horaria local
TIMEZONE = "Europe/Madrid"


# ---------------------------------------------------------------------
# 2. Cargar datos de colmenas y calcular amplitud diaria
# ---------------------------------------------------------------------

def load_hive_data(path: str) -> pd.DataFrame:
    """
    Carga DATOS_SERIETEM.csv y calcula, por colmena y día:
    - amplitud diaria (PESO max - PESO min)
    - hora del máximo y mínimo (en Horario1a24)
    """
    # El CSV viene con separador ; y decimales con coma
    df = pd.read_csv(path, sep=";", decimal=",")

    # Parsear fecha (formato tipo '01.05.2016')
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors="coerce")

    # Asegurar tipos numéricos
    df["PESO"] = pd.to_numeric(df["PESO"], errors="coerce")
    df["Horario1a24"] = pd.to_numeric(df["Horario1a24"], errors="coerce")

    # Filtrar filas válidas
    df = df.dropna(subset=["fecha", "PESO", "Horario1a24"])

    # Renombrar algunas columnas para comodidad
    df = df.rename(columns={
        "AÑO": "year",
        "Colmena2": "hive_id",
        "PERIODO": "phase"
    })

    # Cálculo de métricas diarias por colmena y día
    group_cols = ["fecha", "year", "hive_id", "phase"]

    def daily_stats(group: pd.DataFrame) -> pd.Series:
        peso = group["PESO"].values
        hour = group["Horario1a24"].values.astype(float)

        # amplitud diaria
        amp = np.nanmax(peso) - np.nanmin(peso)

        # índice del máximo y mínimo
        idx_max = np.nanargmax(peso)
        idx_min = np.nanargmin(peso)

        hour_max = hour[idx_max]
        hour_min = hour[idx_min]

        return pd.Series({
            "amplitude": amp,
            "hour_max": hour_max,
            "hour_min": hour_min
        })

    daily_df = df.groupby(group_cols, as_index=False).apply(daily_stats)

    return daily_df


# ---------------------------------------------------------------------
# 3. Descargar meteorología diaria desde Open-Meteo
# ---------------------------------------------------------------------

def fetch_daily_weather(start_date: str, end_date: str,
                        latitude: float, longitude: float,
                        timezone: str = "Europe/Madrid") -> pd.DataFrame:
    """
    Descarga datos diarios históricos (reanálisis) desde Open-Meteo
    para el rango [start_date, end_date] (incluidos).

    start_date / end_date en formato 'YYYY-MM-DD'
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"

    daily_vars = [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "shortwave_radiation_sum"
    ]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(daily_vars),
        "timezone": timezone
    }

    resp = requests.get(base_url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    time = daily.get("time", [])
    if not time:
        raise RuntimeError("No daily weather data returned from Open-Meteo.")

    weather_df = pd.DataFrame({"fecha": pd.to_datetime(time)})
    # Añadir cada variable si viene en la respuesta
    for var in daily_vars:
        if var in daily:
            weather_df[var] = daily[var]

    return weather_df


# ---------------------------------------------------------------------
# 4. Fusionar datos y analizar
# ---------------------------------------------------------------------

def run_analysis():
    # 4.1. Cargar datos de colmenas
    hive_daily = load_hive_data(HIVE_CSV_PATH)
    print("Primeras filas de métricas diarias por colmena:")
    print(hive_daily.head())

    # 4.2. Determinar rango de fechas
    start_date = hive_daily["fecha"].min().date().isoformat()
    end_date = hive_daily["fecha"].max().date().isoformat()
    print(f"\nRango de fechas en datos de colmenas: {start_date} a {end_date}")

    # 4.3. Descargar meteorología diaria
    print("\nDescargando meteorología diaria de Open-Meteo...")
    weather_df = fetch_daily_weather(start_date, end_date,
                                     LATITUDE, LONGITUDE, TIMEZONE)
    print("Primeras filas de meteorología diaria:")
    print(weather_df.head())

    # 4.4. Fusionar por fecha
    merged = hive_daily.merge(weather_df, on="fecha", how="left")
    print(f"\nFilas después de fusionar colmenas + meteo: {len(merged)}")

    # 4.5. Eliminar filas sin meteo
    merged = merged.dropna(subset=["temperature_2m_mean"])

    # 4.6. Calcular correlaciones simples
    meteo_cols = [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "shortwave_radiation_sum"
    ]

    print("\nCorrelaciones de Pearson entre amplitud diaria y variables meteorológicas:")
    for col in meteo_cols:
        if col in merged.columns:
            r = merged["amplitude"].corr(merged[col])
            print(f"  amplitude vs {col}: r = {r:.3f}")

    # 4.7. Regresión lineal múltiple (amplitude ~ meteo)
    #    (Opcional: estandarizamos amplitud para interpretar mejor)
    merged["amplitude_z"] = (merged["amplitude"] - merged["amplitude"].mean()) / merged["amplitude"].std()

    X = merged[meteo_cols].copy()
    X = sm.add_constant(X)  # intercepto
    y = merged["amplitude_z"]

    model = sm.OLS(y, X, missing="drop").fit()
    print("\nResumen de regresión lineal múltiple (amplitude_z ~ meteo):")
    print(model.summary())

    # 4.8. Guardar dataset fusionado (por si quieres usarlo en SPSS u otro software)
    merged.to_csv("hive_meteo_merged.csv", index=False)
    print("\nDatos fusionados guardados en 'hive_meteo_merged.csv'.")


if __name__ == "__main__":
    run_analysis()