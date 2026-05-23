import pandas as pd
import numpy as np

# Si quieres usar tests no paramétricos:
from scipy.stats import kruskal

# =====================================================
# 1. CONFIGURACIÓN BÁSICA
# =====================================================

DATA_PATH = "DATOS_SERIETEM.csv"  # mismo nombre que tu fichero

# Nombres de columnas en tu CSV
COL_YEAR    = "AÑO"
COL_DATE    = "fecha"
COL_HIVE    = "Colmena2"
COL_DAY     = "día"
COL_PHASE   = "PERIODO"       # 1 = pre-bloom, 2 = bloom, 3 = post-bloom
COL_HOUR24  = "Horario1a24"   # 1–24
COL_WEIGHT  = "PESO"          # con coma decimal

PHASE_NAMES = {
    1: "Pre-bloom",
    2: "Bloom",
    3: "Post-bloom",
}

# =====================================================
# 2. LECTURA Y LIMPIEZA DE DATOS
# =====================================================

# El fichero está con separador ';'
df = pd.read_csv(DATA_PATH, sep=';')

# Convertir peso con coma decimal a float
df["PESO_num"] = (
    df[COL_WEIGHT]
    .astype(str)
    .str.replace(",", ".", regex=False)
)

df["PESO_num"] = pd.to_numeric(df["PESO_num"], errors="coerce")

# Asegurarnos de que las columnas clave son numéricas
df[COL_HOUR24] = pd.to_numeric(df[COL_HOUR24], errors="coerce")
df[COL_PHASE]  = pd.to_numeric(df[COL_PHASE],  errors="coerce")
df[COL_DAY]    = pd.to_numeric(df[COL_DAY],    errors="coerce")
df[COL_HIVE]   = pd.to_numeric(df[COL_HIVE],   errors="coerce")

# Eliminar filas sin datos válidos
df_clean = df.dropna(subset=["PESO_num", COL_HOUR24, COL_PHASE, COL_DAY, COL_HIVE])

# =====================================================
# 3. CÁLCULO DE MÁX/MÍN, AMPLITUD Y HORAS
#    POR AÑO–DÍA–COLMENA–FASE
# =====================================================

def hour_at_extreme(sub, col_value, col_hour, func):
    """
    Devuelve la hora (Horario1a24) en la que se alcanza el valor extremo
    (máximo o mínimo) de PESO_num dentro del sub-dataframe.
    """
    # Índice del máximo/mínimo en PESO_num
    idx = sub[col_value].idxmax() if func == "max" else sub[col_value].idxmin()
    # Hora correspondiente
    return sub.loc[idx, col_hour]

# Agrupamos
group_cols = [COL_YEAR, COL_DAY, COL_HIVE, COL_PHASE]

daily = (
    df_clean
    .groupby(group_cols)
    .apply(lambda g: pd.Series({
        "daily_max_weight": g["PESO_num"].max(),
        "daily_min_weight": g["PESO_num"].min(),
        "hour_max": hour_at_extreme(g, "PESO_num", COL_HOUR24, "max"),
        "hour_min": hour_at_extreme(g, "PESO_num", COL_HOUR24, "min"),
    }))
    .reset_index()
)

# Amplitud diaria = max - min
daily["amplitude"] = daily["daily_max_weight"] - daily["daily_min_weight"]

print("Primeras filas del resumen diario:")
print(daily.head())

# =====================================================
# 4. RESUMEN DESCRIPTIVO POR FASE (PERIODO)
# =====================================================

summary = (
    daily
    .groupby(COL_PHASE)
    .agg(
        n_days        = ("amplitude", "size"),
        mean_amp      = ("amplitude", "mean"),
        sd_amp        = ("amplitude", "std"),
        mean_hour_max = ("hour_max", "mean"),
        sd_hour_max   = ("hour_max", "std"),
        mean_hour_min = ("hour_min", "mean"),
        sd_hour_min   = ("hour_min", "std"),
    )
)

# Añadimos nombre de fase para mayor claridad
summary["phase_name"] = summary.index.map(PHASE_NAMES)

print("\nResumen descriptivo por fase (PERIODO):")
print(summary)

# =====================================================
# 5. TEST NO PARAMÉTRICO (KRUSKAL–WALLIS)
#    PARA AMPLITUD ENTRE FASES
# =====================================================

print("\n--- Kruskal–Wallis para amplitud diaria entre fases ---")
phase_values = []
labels = []

for phase_id in sorted(daily[COL_PHASE].dropna().unique()):
    vals = daily.loc[daily[COL_PHASE] == phase_id, "amplitude"].dropna()
    phase_values.append(vals.values)
    labels.append(PHASE_NAMES.get(phase_id, str(phase_id)))
    print(f"Fase {phase_id} ({PHASE_NAMES.get(phase_id)}): n = {len(vals)}")

if len(phase_values) >= 2:
    stat, p = kruskal(*phase_values)
    print(f"\nKruskal–Wallis H = {stat:.3f}, p-value = {p:.3e}")
else:
    print("No hay suficientes fases con datos para aplicar Kruskal–Wallis.")

# =====================================================
# 6. (OPCIONAL) KRUSKAL–WALLIS PARA HORA DE MÁX/MÍN
# =====================================================

print("\n--- Kruskal–Wallis para hora del máximo diario ---")
phase_values_max = []
for phase_id in sorted(daily[COL_PHASE].dropna().unique()):
    vals = daily.loc[daily[COL_PHASE] == phase_id, "hour_max"].dropna()
    phase_values_max.append(vals.values)
    print(f"Fase {phase_id} ({PHASE_NAMES.get(phase_id)}): n = {len(vals)}")

if len(phase_values_max) >= 2:
    stat_max, p_max = kruskal(*phase_values_max)
    print(f"\nKruskal–Wallis (hora max) H = {stat_max:.3f}, p-value = {p_max:.3e}")
else:
    print("No hay suficientes fases con datos para aplicar Kruskal–Wallis en hora_max.")

print("\n--- Kruskal–Wallis para hora del mínimo diario ---")
phase_values_min = []
for phase_id in sorted(daily[COL_PHASE].dropna().unique()):
    vals = daily.loc[daily[COL_PHASE] == phase_id, "hour_min"].dropna()
    phase_values_min.append(vals.values)
    print(f"Fase {phase_id} ({PHASE_NAMES.get(phase_id)}): n = {len(vals)}")

if len(phase_values_min) >= 2:
    stat_min, p_min = kruskal(*phase_values_min)
    print(f"\nKruskal–Wallis (hora min) H = {stat_min:.3f}, p-value = {p_min:.3e}")
else:
    print("No hay suficientes fases con datos para aplicar Kruskal–Wallis en hora_min.")

# =====================================================
# 7. GUARDAR TABLA DE RESULTADOS DIARIOS (OPCIONAL)
# =====================================================

daily.to_csv("daily_metrics_by_hive_day_phase.csv", index=False)
print("\nGuardado 'daily_metrics_by_hive_day_phase.csv' con métricas diarias.")