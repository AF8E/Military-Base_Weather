"""
K-Means Clustering of Military Bases by Weather Patterns
=========================================================
Reads the weather CSVs produced by get_military_weather.py,
applies K-Means clustering, and identifies weather pattern groups
across all 824 military bases.

Outputs:
  - military_bases_clustered.csv        (base info + cluster label)
  - cluster_summary.csv                 (mean feature values per cluster)
  - elbow_plot.png                      (elbow method for optimal K)
  - cluster_scatter.png                 (PCA 2-D scatter of clusters)
  - cluster_radar.png                   (radar/spider chart per cluster)
  - cluster_heatmap.png                 (heatmap of cluster centroids)
  - cluster_feature_boxplots.png        (boxplots of key features by cluster)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WEATHER_CSV = "military_bases_weather_v2.csv"
DAILY_CSV   = "military_bases_daily_weather_v2.csv"
OUTPUT_CLUSTERED = "military_bases_clustered_v2.csv"
OUTPUT_SUMMARY   = "cluster_summary_v2.csv"

MAX_K = 15          # max clusters to test in elbow method
RANDOM_STATE = 42

# Features to use for clustering (current weather snapshot columns)
CURRENT_FEATURES = [
    "current_temperature_2m",
    "current_relative_humidity_2m",
    "current_apparent_temperature",
    "current_precipitation",
    "current_rain",
    "current_snowfall",
    "current_weather_code",
    "current_cloud_cover",
    "current_pressure_msl",
    "current_surface_pressure",
    "current_wind_speed_10m",
    "current_wind_direction_10m",
    "current_wind_gusts_10m",
]

# Daily aggregate features (we'll compute means per base from the 7-day forecast)
DAILY_FEATURES = [
    "daily_temperature_2m_max",
    "daily_temperature_2m_min",
    "daily_temperature_2m_mean",
    "daily_apparent_temperature_max",
    "daily_apparent_temperature_min",
    "daily_precipitation_sum",
    "daily_rain_sum",
    "daily_snowfall_sum",
    "daily_wind_speed_10m_max",
    "daily_wind_gusts_10m_max",
    "daily_sunshine_duration",
    "daily_uv_index_max",
    "daily_shortwave_radiation_sum",
    "daily_precipitation_hours",
    "daily_et0_fao_evapotranspiration",
]


# ============================= DATA LOADING ==================================
def load_current_weather() -> pd.DataFrame:
    """Load the current weather snapshot CSV."""
    print(f"Loading {WEATHER_CSV}...")
    df = pd.read_csv(WEATHER_CSV)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_daily_weather() -> pd.DataFrame:
    """Load the daily forecast CSV and aggregate to per-base means."""
    print(f"Loading {DAILY_CSV}...")
    import os
    if not os.path.exists(DAILY_CSV) or os.path.getsize(DAILY_CSV) < 10:
        print("  WARNING: Daily weather file is empty or missing. Skipping daily features.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(DAILY_CSV)
    except Exception as e:
        print(f"  WARNING: Could not read daily CSV: {e}")
        return pd.DataFrame()

    print(f"  Loaded {len(df)} rows (base x days)")

    # Aggregate: compute the 7-day mean for each base
    id_cols = ["OBJECTID", "Site_Name", "State", "latitude", "longitude"]
    available_daily = [c for c in DAILY_FEATURES if c in df.columns]

    if not available_daily:
        print("  WARNING: No daily features found!")
        return pd.DataFrame()

    agg_dict = {col: "mean" for col in available_daily}
    grouped = df.groupby("OBJECTID", as_index=False).agg(agg_dict)

    # Rename daily columns to indicate they are 7-day averages
    rename = {col: col + "_7day_avg" for col in available_daily}
    grouped = grouped.rename(columns=rename)

    print(f"  Aggregated to {len(grouped)} bases with {len(available_daily)} daily features")
    return grouped


def prepare_features(current_df: pd.DataFrame, daily_agg: pd.DataFrame) -> tuple:
    """
    Merge current + daily aggregated data and prepare the feature matrix.
    Returns: (full_df, feature_matrix, feature_names, scaler)
    """
    # Start with current weather
    df = current_df.copy()

    # Merge daily aggregates if available
    if len(daily_agg) > 0 and "OBJECTID" in daily_agg.columns:
        daily_cols = [c for c in daily_agg.columns if c != "OBJECTID"]
        df = df.merge(daily_agg, on="OBJECTID", how="left")
        print(f"  Merged daily features: {len(daily_cols)} columns added")

    # Collect all numeric feature columns
    available_current = [c for c in CURRENT_FEATURES if c in df.columns]
    available_daily = [c for c in df.columns if c.endswith("_7day_avg")]
    all_features = available_current + available_daily

    print(f"\n  Total features for clustering: {len(all_features)}")
    for f in all_features:
        print(f"    - {f}")

    # Extract feature matrix
    X = df[all_features].copy()

    # Handle missing values — fill with column median
    missing_before = X.isnull().sum().sum()
    X = X.fillna(X.median())
    print(f"\n  Missing values filled: {missing_before} cells (median imputation)")

    # Drop any rows that are still all-NaN (shouldn't happen, but safety)
    valid_mask = X.notna().any(axis=1)
    df = df[valid_mask].reset_index(drop=True)
    X = X[valid_mask].reset_index(drop=True)

    # Standardize features (zero mean, unit variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"  Final feature matrix: {X_scaled.shape[0]} bases x {X_scaled.shape[1]} features")
    return df, X_scaled, all_features, scaler


# ============================= ELBOW METHOD ==================================
def find_optimal_k(X: np.ndarray, max_k: int = MAX_K) -> int:
    """
    Run K-Means for K=2..max_k, plot inertia (elbow) and silhouette scores.
    Returns the K with the highest silhouette score.
    """
    print(f"\nRunning elbow method (K=2 to {max_k})...")
    inertias = []
    silhouettes = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels)
        silhouettes.append(sil)
        print(f"  K={k:2d}  |  Inertia={km.inertia_:12.1f}  |  Silhouette={sil:.4f}")

    best_k = list(K_range)[np.argmax(silhouettes)]
    print(f"\n  Best K by silhouette score: {best_k} (score={max(silhouettes):.4f})")

    # Plot elbow + silhouette
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(list(K_range), inertias, "bo-", linewidth=2, markersize=6)
    ax1.axvline(x=best_k, color="r", linestyle="--", alpha=0.7, label=f"Best K={best_k}")
    ax1.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
    ax1.set_title("Elbow Method", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(list(K_range), silhouettes, "rs-", linewidth=2, markersize=6)
    ax2.axvline(x=best_k, color="r", linestyle="--", alpha=0.7, label=f"Best K={best_k}")
    ax2.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax2.set_ylabel("Silhouette Score", fontsize=12)
    ax2.set_title("Silhouette Score", fontsize=14, fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("elbow_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: elbow_plot.png")

    return best_k


# ============================= K-MEANS CLUSTERING ============================
def run_kmeans(X: np.ndarray, k: int) -> tuple:
    """Run final K-Means with optimal K. Returns (labels, model)."""
    print(f"\nRunning K-Means with K={k}...")
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20, max_iter=500)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels)
    print(f"  Silhouette Score: {sil:.4f}")

    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n  Cluster distribution:")
    for cluster, count in zip(unique, counts):
        print(f"    Cluster {cluster}: {count} bases ({count/len(labels)*100:.1f}%)")

    return labels, km


# ============================= VISUALIZATION =================================
def plot_pca_scatter(X: np.ndarray, labels: np.ndarray, df: pd.DataFrame):
    """PCA 2-D scatter plot colored by cluster."""
    print("\nGenerating PCA scatter plot...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels, cmap="tab10", alpha=0.7, s=40, edgecolors="white", linewidth=0.5
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=12)
    ax.set_title("Military Bases Weather Clusters (PCA Projection)", fontsize=14, fontweight="bold")

    legend = ax.legend(*scatter.legend_elements(), title="Cluster", loc="upper right")
    ax.add_artist(legend)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("cluster_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: cluster_scatter.png")

    # Also do 3-component PCA for explained variance
    pca3 = PCA(n_components=min(X.shape[1], 10))
    pca3.fit(X)
    cum_var = np.cumsum(pca3.explained_variance_ratio_)
    print(f"  PCA explained variance (cumulative):")
    for i, v in enumerate(cum_var[:5]):
        print(f"    PC{i+1}: {v*100:.1f}%")


def plot_cluster_heatmap(centroids: np.ndarray, feature_names: list, k: int):
    """Heatmap of cluster centroid values (standardized)."""
    print("\nGenerating cluster heatmap...")
    fig, ax = plt.subplots(figsize=(max(14, len(feature_names) * 0.6), max(6, k * 0.8)))

    # Clean feature names for display
    display_names = [f.replace("current_", "").replace("daily_", "").replace("_7day_avg", " (7d)")
                     for f in feature_names]

    sns.heatmap(
        centroids,
        xticklabels=display_names,
        yticklabels=[f"Cluster {i}" for i in range(k)],
        cmap="RdYlBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Cluster Centroids (Standardized Feature Values)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("cluster_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: cluster_heatmap.png")


def plot_radar_chart(centroids: np.ndarray, feature_names: list, k: int):
    """Radar/spider chart showing each cluster's profile."""
    print("\nGenerating radar chart...")
    # Normalize centroids to 0-1 range for radar
    mins = centroids.min(axis=0)
    maxs = centroids.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # avoid division by zero
    normed = (centroids - mins) / ranges

    display_names = [f.replace("current_", "").replace("daily_", "").replace("_7day_avg", " (7d)")
                     for f in feature_names]

    # Limit to top features for readability
    if len(feature_names) > 12:
        # Pick features with highest variance across clusters
        variances = np.var(centroids, axis=0)
        top_idx = np.argsort(variances)[-12:]
        normed = normed[:, top_idx]
        display_names = [display_names[i] for i in top_idx]

    n_features = len(display_names)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, k))

    for i in range(k):
        values = normed[i].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=f"Cluster {i}", color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_names, size=8)
    ax.set_title("Cluster Weather Profiles (Radar Chart)", size=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig("cluster_radar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: cluster_radar.png")


def plot_feature_boxplots(df: pd.DataFrame, feature_names: list):
    """Box plots of key features grouped by cluster."""
    print("\nGenerating feature boxplots...")

    # Select top 6 most discriminating features
    available = [f for f in feature_names if f in df.columns]
    if len(available) == 0:
        print("  No features available for boxplots.")
        return

    # Calculate variance ratio (between-cluster / total) for each feature
    feature_importance = []
    for f in available:
        total_var = df[f].var()
        if total_var > 0:
            between_var = df.groupby("Cluster")[f].mean().var()
            feature_importance.append((f, between_var / total_var))

    feature_importance.sort(key=lambda x: x[1], reverse=True)
    top_features = [f for f, _ in feature_importance[:6]]

    n_plots = len(top_features)
    cols = 3
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for i, feat in enumerate(top_features):
        display = feat.replace("current_", "").replace("daily_", "").replace("_7day_avg", " (7d)")
        sns.boxplot(data=df, x="Cluster", y=feat, palette="tab10", ax=axes[i])
        axes[i].set_title(display, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Cluster")
        axes[i].grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Weather Feature Distribution by Cluster", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("cluster_feature_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: cluster_feature_boxplots.png")


# ============================= CLUSTER ANALYSIS ==============================
def analyze_clusters(df: pd.DataFrame, feature_names: list, k: int):
    """Print a detailed description of each cluster's weather pattern."""
    print("\n" + "=" * 65)
    print("  CLUSTER PATTERN ANALYSIS")
    print("=" * 65)

    available = [f for f in feature_names if f in df.columns]
    overall_means = df[available].mean()
    overall_stds = df[available].std()

    for cluster in range(k):
        cluster_df = df[df["Cluster"] == cluster]
        n = len(cluster_df)
        pct = n / len(df) * 100

        print(f"\n{'─' * 65}")
        print(f"  CLUSTER {cluster}: {n} bases ({pct:.1f}%)")
        print(f"{'─' * 65}")

        # List states
        states = cluster_df["State"].value_counts().head(10)
        print(f"  Top states: {', '.join(f'{s}({c})' for s, c in states.items())}")

        # List a few example bases
        examples = cluster_df["Site_Name"].head(5).tolist()
        print(f"  Example bases: {', '.join(str(e) for e in examples)}")

        # Weather pattern description
        cluster_means = cluster_df[available].mean()
        print(f"\n  Weather characteristics (vs. overall average):")

        for feat in available:
            val = cluster_means[feat]
            avg = overall_means[feat]
            std = overall_stds[feat]
            if std > 0:
                z = (val - avg) / std
                direction = "ABOVE" if z > 0.3 else "BELOW" if z < -0.3 else "NEAR"
                display = feat.replace("current_", "").replace("daily_", "").replace("_7day_avg", " (7d)")
                if abs(z) > 0.3:
                    bar = "+" * min(int(abs(z) * 3), 15) if z > 0 else "-" * min(int(abs(z) * 3), 15)
                    print(f"    {display:40s}  {val:8.2f}  ({direction:5s} avg, z={z:+.2f}) {bar}")

    print(f"\n{'=' * 65}")


def create_cluster_summary(df: pd.DataFrame, feature_names: list, k: int) -> pd.DataFrame:
    """Create and save a summary CSV with mean values per cluster."""
    available = [f for f in feature_names if f in df.columns]
    summary_rows = []

    for cluster in range(k):
        cluster_df = df[df["Cluster"] == cluster]
        row = {"Cluster": cluster, "Num_Bases": len(cluster_df)}

        # Top states
        top_states = cluster_df["State"].value_counts().head(5)
        row["Top_States"] = ", ".join(f"{s}({c})" for s, c in top_states.items())

        # Mean of each feature
        for feat in available:
            display = feat.replace("current_", "").replace("daily_", "").replace("_7day_avg", "_7d_avg")
            row[display] = cluster_df[feat].mean()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)
    print(f"\nSaved: {OUTPUT_SUMMARY}")
    return summary_df


# ============================= MAIN ==========================================
def main():
    print("=" * 65)
    print("  K-MEANS CLUSTERING: MILITARY BASE WEATHER PATTERNS")
    print("=" * 65)

    # --- Load data ---
    current_df = load_current_weather()
    daily_agg = load_daily_weather()

    # --- Prepare features ---
    df, X_scaled, feature_names, scaler = prepare_features(current_df, daily_agg)

    if X_scaled.shape[0] < 10:
        print("ERROR: Not enough bases with valid data for clustering.")
        return

    # --- Find optimal K ---
    best_k = find_optimal_k(X_scaled)

    # --- Run K-Means ---
    labels, km_model = run_kmeans(X_scaled, best_k)

    # --- Add cluster labels to dataframe ---
    df["Cluster"] = labels

    # Save clustered data
    df.to_csv(OUTPUT_CLUSTERED, index=False)
    print(f"\nSaved: {OUTPUT_CLUSTERED}")

    # --- Visualizations ---
    plot_pca_scatter(X_scaled, labels, df)
    plot_cluster_heatmap(km_model.cluster_centers_, feature_names, best_k)
    plot_radar_chart(km_model.cluster_centers_, feature_names, best_k)
    plot_feature_boxplots(df, feature_names)

    # --- Analysis ---
    analyze_clusters(df, feature_names, best_k)
    summary = create_cluster_summary(df, feature_names, best_k)

    # --- Final summary ---
    print("\n" + "=" * 65)
    print("  FINAL SUMMARY")
    print("=" * 65)
    print(f"  Total bases clustered   : {len(df)}")
    print(f"  Number of clusters (K)  : {best_k}")
    print(f"  Features used           : {len(feature_names)}")
    print(f"  Silhouette score        : {silhouette_score(X_scaled, labels):.4f}")
    print(f"\n  Output files:")
    print(f"    1. {OUTPUT_CLUSTERED}      (all bases with cluster labels)")
    print(f"    2. {OUTPUT_SUMMARY}              (cluster centroids summary)")
    print(f"    3. elbow_plot.png                (elbow + silhouette analysis)")
    print(f"    4. cluster_scatter.png           (PCA 2-D scatter plot)")
    print(f"    5. cluster_heatmap.png           (centroid heatmap)")
    print(f"    6. cluster_radar.png             (radar profile chart)")
    print(f"    7. cluster_feature_boxplots.png  (feature distributions)")
    print("=" * 65)


if __name__ == "__main__":
    main()
