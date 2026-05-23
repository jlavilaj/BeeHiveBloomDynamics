import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Cargar amplitud diaria
daily = pd.read_csv("daily_metrics_by_hive_day_phase.csv")

# Cargar meteorología diaria
meteo = pd.read_csv("meteo_daily.csv")

# Unir por año y día
df = pd.merge(daily, meteo, on=["AÑO", "día"], how="inner")

# OPCIÓN 1: trabajar a nivel colmena-día (muchos puntos)
x = df["amplitude"].values
y = df["Tmed"].values  # por ejemplo temperatura media diaria

# Pearson
r_pearson, p_pearson = pearsonr(x, y)

# Spearman
r_spear, p_spear = spearmanr(x, y)

print("Pearson r =", r_pearson, "p =", p_pearson)
print("Spearman r =", r_spear, "p =", p_spear)

# OPCIÓN 2: primero promediar amplitud por día (todas las colmenas)
df_day = df.groupby(["AÑO", "día"], as_index=False).agg(
    amp_mean=("amplitude", "mean"),
    Tmed=("Tmed", "mean"),
)

r_pearson_day, p_pearson_day = pearsonr(df_day["amp_mean"], df_day["Tmed"])
print("Pearson (day-aggregated) r =", r_pearson_day, "p =", p_pearson_day)