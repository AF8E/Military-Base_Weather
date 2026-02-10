"""
Individual Base Weather Analysis
=================================
Creates per-base visualizations and summary so you can see
each of the 824 military bases individually.

Outputs:
  - individual_base_heatmap.png     (all 824 bases x 28 features, sorted by cluster)
  - individual_base_profiles.csv    (each base with all features + cluster + rank info)
  - cluster_0_heatmap.png           (warm bases only)
  - cluster_1_heatmap.png           (cool bases only)
  - cluster_2_heatmap.png           (arctic bases only)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
CLUSTERED_CSV = "military_bases_clustered_v2.csv"
DAILY_CSV = "military_bases_daily_weather_v2.csv"

print("Loading clustered data...")
df = pd.read_csv(CLUSTERED_CSV)
print(f"  {len(df)} bases, {len(df.columns)} columns")

# Identify the weather feature columns (same ones used for clustering)
CURRENT_FEATURES = [
    "current_temperature_2m", "current_relative_humidity_2m",
    "current_apparent_temperature", "current_precipitation",
    "current_rain", "current_snowfall", "current_weather_code",
    "current_cloud_cover", "current_pressure_msl", "current_surface_pressure",
    "current_wind_speed_10m", "current_wind_direction_10m", "current_wind_gusts_10m",
]

# Check for daily 7-day avg columns
daily_cols = [c for c in df.columns if "7day_avg" in c]
ALL_FEATURES = [c for c in CURRENT_FEATURES if c in df.columns] + daily_cols
print(f"  Features: {len(ALL_FEATURES)}")

# Short labels for readability
SHORT_LABELS = {
    "current_temperature_2m": "Temp (°F)",
    "current_relative_humidity_2m": "Humidity (%)",
    "current_apparent_temperature": "Feels Like (°F)",
    "current_precipitation": "Precip (in)",
    "current_rain": "Rain (in)",
    "current_snowfall": "Snow (in)",
    "current_weather_code": "Weather Code",
    "current_cloud_cover": "Cloud Cover (%)",
    "current_pressure_msl": "Pressure MSL",
    "current_surface_pressure": "Surface Press.",
    "current_wind_speed_10m": "Wind Speed (mph)",
    "current_wind_direction_10m": "Wind Dir (°)",
    "current_wind_gusts_10m": "Wind Gusts (mph)",
}
for c in daily_cols:
    short = c.replace("daily_", "").replace("_7day_avg", "").replace("_", " ").title()
    SHORT_LABELS[c] = f"{short} (7d)"

# ---------------------------------------------------------------------------
# Standardize features
# ---------------------------------------------------------------------------
X = df[ALL_FEATURES].copy()
X = X.fillna(X.median())
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=ALL_FEATURES, index=df.index)

# Sort by cluster, then by temperature within each cluster
df_sorted = df.copy()
df_sorted["_sort_temp"] = X_scaled["current_temperature_2m"]
df_sorted = df_sorted.sort_values(["Cluster", "_sort_temp"], ascending=[True, False])
X_sorted = X_scaled.loc[df_sorted.index]

# ---------------------------------------------------------------------------
# 1. FULL HEATMAP: All 824 bases
# ---------------------------------------------------------------------------
print("\nGenerating full individual base heatmap (824 x 28)...")

fig, ax = plt.subplots(figsize=(20, 40))

# Create the heatmap
short_cols = [SHORT_LABELS.get(c, c) for c in ALL_FEATURES]
plot_data = X_sorted.copy()
plot_data.columns = short_cols

# Y-axis labels: Base Name (State) 
y_labels = [f"{row['Site_Name']} ({row['State'].upper()})" 
            for _, row in df_sorted.iterrows()]

sns.heatmap(plot_data, ax=ax, cmap="RdYlBu_r", center=0,
            yticklabels=y_labels, xticklabels=short_cols,
            vmin=-3, vmax=3, linewidths=0,
            cbar_kws={"label": "Standardized Value (Z-score)", "shrink": 0.3})

ax.set_title("All 824 Military Bases — Individual Weather Profiles\n(Sorted by Cluster, then Temperature)",
             fontsize=16, fontweight="bold", pad=20)
ax.tick_params(axis="y", labelsize=3)
ax.tick_params(axis="x", labelsize=8, rotation=45)

# Add cluster divider lines
cluster_counts = df_sorted["Cluster"].value_counts().sort_index()
cumsum = 0
cluster_names = {0: "Cluster 0: Warm", 1: "Cluster 1: Cool", 2: "Cluster 2: Arctic"}
for cl in sorted(cluster_counts.index):
    count = cluster_counts[cl]
    # Label at midpoint
    mid = cumsum + count / 2
    ax.text(-1.5, mid, cluster_names.get(cl, f"Cluster {cl}"),
            fontsize=10, fontweight="bold", ha="right", va="center",
            color=["#d32f2f", "#1565c0", "#6a1b9a"][cl % 3])
    cumsum += count
    if cl < max(cluster_counts.index):
        ax.axhline(y=cumsum, color="black", linewidth=2)

plt.tight_layout()
fig.savefig("individual_base_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: individual_base_heatmap.png")

# ---------------------------------------------------------------------------
# 2. PER-CLUSTER HEATMAPS (readable size)
# ---------------------------------------------------------------------------
cluster_names_short = {0: "Warm Climate", 1: "Cool/Temperate", 2: "Arctic/Extreme Cold"}
cluster_colors = {0: "Oranges", 1: "Blues", 2: "Purples"}

for cl in sorted(df["Cluster"].unique()):
    mask = df_sorted["Cluster"] == cl
    cl_data = X_sorted.loc[mask].copy()
    cl_df = df_sorted.loc[mask]
    
    n = len(cl_data)
    fig_h = max(8, n * 0.12 + 3)
    fig, ax = plt.subplots(figsize=(18, fig_h))

    cl_plot = cl_data.copy()
    cl_plot.columns = short_cols
    
    y_labels_cl = [f"{row['Site_Name']} ({row['State'].upper()})" 
                   for _, row in cl_df.iterrows()]

    sns.heatmap(cl_plot, ax=ax, cmap="RdYlBu_r", center=0,
                yticklabels=y_labels_cl, xticklabels=short_cols,
                vmin=-3, vmax=3, linewidths=0.1,
                cbar_kws={"label": "Standardized Z-score", "shrink": 0.5})

    name = cluster_names_short.get(cl, f"Cluster {cl}")
    ax.set_title(f"Cluster {cl}: {name} — {n} Bases (Individual Weather Profiles)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="y", labelsize=5 if n > 100 else 7)
    ax.tick_params(axis="x", labelsize=8, rotation=45)

    plt.tight_layout()
    fname = f"cluster_{cl}_heatmap.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}  ({n} bases)")

# ---------------------------------------------------------------------------
# 3. Per-base profiles CSV with ranks
# ---------------------------------------------------------------------------
print("\nGenerating individual base profiles CSV...")

profiles = df_sorted[["OBJECTID", "Site_Name", "State", "latitude", "longitude", 
                       "Cluster"]].copy()

# Add raw values
for feat in ALL_FEATURES:
    short = SHORT_LABELS.get(feat, feat)
    profiles[short] = df_sorted[feat].values

# Add z-scores
for feat in ALL_FEATURES:
    short = SHORT_LABELS.get(feat, feat)
    profiles[f"{short} (z-score)"] = X_sorted[feat].values

# Add ranks (1 = highest temp, etc.)
for feat in ALL_FEATURES:
    short = SHORT_LABELS.get(feat, feat)
    profiles[f"{short} (rank)"] = df[feat].rank(ascending=False, method="min").loc[df_sorted.index].astype(int).values

profiles.to_csv("individual_base_profiles.csv", index=False)
print(f"  Saved: individual_base_profiles.csv  ({len(profiles)} bases, {len(profiles.columns)} columns)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("  INDIVIDUAL BASE ANALYSIS COMPLETE")
print("=" * 65)
print(f"  Total bases: {len(df)}")
print(f"  Features per base: {len(ALL_FEATURES)}")
print(f"\n  Output files:")
print(f"    1. individual_base_heatmap.png      (ALL 824 bases, sorted by cluster)")
print(f"    2. cluster_0_heatmap.png            (Warm: {(df['Cluster']==0).sum()} bases)")
print(f"    3. cluster_1_heatmap.png            (Cool: {(df['Cluster']==1).sum()} bases)")
print(f"    4. cluster_2_heatmap.png            (Arctic: {(df['Cluster']==2).sum()} bases)")
print(f"    5. individual_base_profiles.csv     (every base with values, z-scores, ranks)")
print("=" * 65)
