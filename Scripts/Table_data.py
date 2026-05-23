import pandas as pd

# Ruta a tu CSV
HIVE_PATH = "DATOS_SERIETEM.csv"

# Nombres de columnas
COL_YEAR   = "AÑO"
COL_DAY    = "día"
COL_HIVE   = "Colmena2"
COL_PHASE  = "PERIODO"      # 1=pre-bloom, 2=bloom, 3=post-bloom
COL_HOUR   = "Horario1a24"
COL_WEIGHT = "PESO"

# 1. Cargar datos
df = pd.read_csv(HIVE_PATH, sep=";", decimal=",")

# 2. Limpiar y convertir tipos
df["PESO_num"] = pd.to_numeric(df[COL_WEIGHT], errors="coerce")
df[COL_HOUR]   = pd.to_numeric(df[COL_HOUR], errors="coerce")
df[COL_PHASE]  = pd.to_numeric(df[COL_PHASE], errors="coerce")
df[COL_DAY]    = pd.to_numeric(df[COL_DAY], errors="coerce")
df[COL_HIVE]   = pd.to_numeric(df[COL_HIVE], errors="coerce")
df[COL_YEAR]   = pd.to_numeric(df[COL_YEAR], errors="coerce")

df_clean = df.dropna(subset=["PESO_num", COL_HOUR, COL_PHASE, COL_DAY, COL_HIVE, COL_YEAR])

# 3. Estadísticos diarios por colmena y día
group_cols = [COL_YEAR, COL_DAY, COL_HIVE, COL_PHASE]

def daily_stats(g):
    max_w = g["PESO_num"].max()
    min_w = g["PESO_num"].min()
    # Hora del máximo y del mínimo (primer índice)
    hour_max = g.loc[g["PESO_num"].idxmax(), COL_HOUR]
    hour_min = g.loc[g["PESO_num"].idxmin(), COL_HOUR]
    return pd.Series({
        "daily_max_weight": max_w,
        "daily_min_weight": min_w,
        "hour_max": hour_max,
        "hour_min": hour_min,
        "amplitude": max_w - min_w,
    })

daily = df_clean.groupby(group_cols).apply(daily_stats).reset_index()

# 4. Resumen por fase (todos los años)
summary_phase = daily.groupby(COL_PHASE).agg(
    n_days=("amplitude", "size"),
    mean_amp=("amplitude", "mean"),
    sd_amp=("amplitude", "std"),
    mean_hour_max=("hour_max", "mean"),
    sd_hour_max=("hour_max", "std"),
    mean_hour_min=("hour_min", "mean"),
    sd_hour_min=("hour_min", "std"),
).reset_index()

print(summary_phase)

# 5. Resumen por año y fase (opcional)
summary_year_phase = daily.groupby([COL_YEAR, COL_PHASE]).agg(
    n_days=("amplitude", "size"),
    mean_amp=("amplitude", "mean"),
    sd_amp=("amplitude", "std"),
    mean_hour_max=("hour_max", "mean"),
    sd_hour_max=("hour_max", "std"),
    mean_hour_min=("hour_min", "mean"),
    sd_hour_min=("hour_min", "std"),
).reset_index()

print(summary_year_phase)