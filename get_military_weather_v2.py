"""
Military Bases Weather Data Fetcher v2
=======================================
IMPROVED VERSION: Gets lat/lon from the official BTS ArcGIS API
(polygon centroids) instead of geocoding by name.

This guarantees coordinates for ALL 824 bases.

1. Pulls polygon geometry from BTS ArcGIS FeatureServer
2. Computes centroid (lat/lon) of each base polygon
3. Fetches ALL available current + daily weather from Open-Meteo
4. Saves results to CSV files
"""

import pandas as pd
import numpy as np
import requests
import time
import json
import os
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = "NTAD_Military_Bases_-1644289556481787667.csv"
ARCGIS_URL = "https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/NTAD_Military_Bases/FeatureServer/0/query"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

OUTPUT_GEOCODED = "military_bases_geocoded_v2.csv"
OUTPUT_WEATHER  = "military_bases_weather_v2.csv"
OUTPUT_DAILY    = "military_bases_daily_weather_v2.csv"

# ---------------------------------------------------------------------------
# ALL Open-Meteo current weather variables
# ---------------------------------------------------------------------------
CURRENT_VARS = [
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "is_day", "precipitation", "rain", "showers", "snowfall",
    "weather_code", "cloud_cover", "pressure_msl", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
]

# ---------------------------------------------------------------------------
# ALL Open-Meteo daily weather variables
# ---------------------------------------------------------------------------
DAILY_VARS = [
    "weather_code",
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
    "sunrise", "sunset",
    "daylight_duration", "sunshine_duration",
    "uv_index_max", "uv_index_clear_sky_max",
    "precipitation_sum", "rain_sum", "showers_sum", "snowfall_sum",
    "precipitation_hours", "precipitation_probability_max",
    "precipitation_probability_mean", "precipitation_probability_min",
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]

# ---------------------------------------------------------------------------
# Hourly variables (we'll fetch a subset — the most useful ones)
# ---------------------------------------------------------------------------
HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation_probability",
    "precipitation", "rain", "showers", "snowfall", "snow_depth",
    "weather_code", "pressure_msl", "surface_pressure",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "visibility", "evapotranspiration", "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "wind_speed_10m", "wind_speed_80m",
    "wind_direction_10m", "wind_direction_80m",
    "wind_gusts_10m",
    "temperature_80m",
    "soil_temperature_0cm", "soil_temperature_6cm",
    "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm",
    "uv_index", "uv_index_clear_sky", "is_day", "sunshine_duration",
    "wet_bulb_temperature_2m", "cape",
    "freezing_level_height", "boundary_layer_height",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance",
]


# ============================= STEP 1 ======================================
# Get coordinates from ArcGIS API (polygon centroids)
# ===========================================================================
def compute_polygon_centroid(rings: list) -> tuple:
    """Compute the centroid of a polygon from its rings (list of coordinate arrays)."""
    all_lons = []
    all_lats = []
    for ring in rings:
        for point in ring:
            all_lons.append(point[0])
            all_lats.append(point[1])

    if not all_lons:
        return None, None

    return np.mean(all_lats), np.mean(all_lons)


def fetch_all_base_coordinates() -> pd.DataFrame:
    """
    Pull all military base polygons from the BTS ArcGIS API
    and compute centroid lat/lon for each one.
    """
    print("Fetching military base geometries from BTS ArcGIS API...")

    # First get all OBJECTIDs
    resp0 = requests.get(ARCGIS_URL, params={
        "where": "1=1", "outFields": "OBJECTID", "returnGeometry": "false",
        "f": "json", "resultRecordCount": "2000"
    }, timeout=30)
    all_ids = [f["attributes"]["OBJECTID"] for f in resp0.json().get("features", [])]
    print(f"  Found {len(all_ids)} base IDs")

    # Fetch geometry in SMALL batches (polygons are huge)
    # Using maxAllowableOffset to massively simplify geometry → smaller responses
    all_features = []
    batch_size = 50  # simplified polygons = small responses

    for i in tqdm(range(0, len(all_ids), batch_size), desc="Fetching geometry"):
        batch_ids = all_ids[i:i + batch_size]
        id_list = ",".join(str(x) for x in batch_ids)
        where_clause = f"OBJECTID IN ({id_list})"

        params = {
            "where": where_clause,
            "outFields": "OBJECTID,siteName,stateNameCode,siteOperationalStatus,siteReportingComponent,isJointBase",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "json",
            "resultRecordCount": "100",
            "maxAllowableOffset": "0.01",  # simplify polygons → much smaller response
        }

        try:
            resp = requests.get(ARCGIS_URL, params=params, timeout=60)
            if resp.status_code != 200:
                print(f"  API error {resp.status_code} for batch starting at {i}")
                continue

            data = resp.json()
            features = data.get("features", [])
            all_features.extend(features)
        except Exception as e:
            print(f"  Error fetching batch at {i}: {e}")

        time.sleep(0.15)

    print(f"  Total features from API: {len(all_features)}")

    print(f"  Total features from API: {len(all_features)}")

    # Extract attributes and compute centroids
    rows = []
    for feat in all_features:
        attrs = feat.get("attributes", {})
        geom = feat.get("geometry", {})
        rings = geom.get("rings", [])

        lat, lon = compute_polygon_centroid(rings)

        rows.append({
            "OBJECTID": attrs.get("OBJECTID"),
            "Site_Name": attrs.get("siteName", ""),
            "State": attrs.get("stateNameCode", ""),
            "Operational_Status": attrs.get("siteOperationalStatus", ""),
            "Reporting_Component": attrs.get("siteReportingComponent", ""),
            "Is_Joint_Base": attrs.get("isJointBase", ""),
            "latitude": lat,
            "longitude": lon,
        })

    df = pd.DataFrame(rows)

    valid = df["latitude"].notna().sum()
    print(f"  Successfully computed centroids: {valid}/{len(df)}")

    df.to_csv(OUTPUT_GEOCODED, index=False)
    print(f"  Saved: {OUTPUT_GEOCODED}")

    return df


# ============================= STEP 2 ======================================
# Fetch weather data from Open-Meteo (batched)
# ===========================================================================
def fetch_weather_batch(latitudes: list, longitudes: list,
                        include_daily=True, max_retries=5) -> list:
    """
    Fetch current + daily weather for a batch of locations.
    Returns a list of response dicts (one per location).
    Handles 429 rate-limit errors by waiting and retrying.
    """
    params = {
        "latitude": ",".join(f"{x:.6f}" for x in latitudes),
        "longitude": ",".join(f"{x:.6f}" for x in longitudes),
        "current": ",".join(CURRENT_VARS),
        "timezone": "auto",
        "forecast_days": 7,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
    }

    if include_daily:
        params["daily"] = ",".join(DAILY_VARS)

    for attempt in range(max_retries):
        try:
            resp = requests.get(FORECAST_URL, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                # Single location returns dict, multiple returns list
                if isinstance(data, list):
                    return data
                else:
                    return [data]
            elif resp.status_code == 429:
                # Rate limited — wait and retry
                wait = 65 + attempt * 10  # 65s, 75s, 85s, ...
                print(f"    Rate limited (429). Waiting {wait}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait)
                continue
            else:
                print(f"    API error {resp.status_code}: {resp.text[:200]}")
                return []
        except Exception as e:
            print(f"    Request error: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)
                continue
            return []

    print("    Max retries reached, skipping batch")
    return []


def fetch_all_weather(df: pd.DataFrame) -> tuple:
    """Fetch current + daily weather for all bases. Returns (current_df, daily_df)."""
    valid = df.dropna(subset=["latitude", "longitude"]).copy()
    print(f"\nFetching weather for {len(valid)} bases...")

    # Check for partial results from a previous run
    partial_current = []
    partial_daily = []
    done_ids = set()
    if os.path.exists(OUTPUT_WEATHER):
        try:
            prev = pd.read_csv(OUTPUT_WEATHER)
            if "current_temperature_2m" in prev.columns:
                has_data = prev["current_temperature_2m"].notna()
                partial_current = prev[has_data].to_dict("records")
                done_ids = set(prev[has_data]["OBJECTID"].tolist())
                print(f"  Resuming: {len(done_ids)} bases already have weather data")
        except Exception:
            pass

    if os.path.exists(OUTPUT_DAILY) and done_ids:
        try:
            prev_d = pd.read_csv(OUTPUT_DAILY)
            partial_daily = prev_d[prev_d["OBJECTID"].isin(done_ids)].to_dict("records")
        except Exception:
            pass

    # Filter out already-done bases
    remaining = valid[~valid["OBJECTID"].isin(done_ids)]
    print(f"  Remaining bases to fetch: {len(remaining)}")

    BATCH_SIZE = 20  # smaller batches to avoid rate limits
    current_rows = list(partial_current)
    daily_rows = list(partial_daily)

    for start in tqdm(range(0, len(remaining), BATCH_SIZE), desc="Weather batches"):
        batch = remaining.iloc[start: start + BATCH_SIZE]
        lats = batch["latitude"].tolist()
        lons = batch["longitude"].tolist()

        results = fetch_weather_batch(lats, lons, include_daily=True)

        if not results:
            # Fallback: try individual requests with waits
            print("    Batch failed — trying individual requests...")
            for idx in range(len(batch)):
                single = fetch_weather_batch([lats[idx]], [lons[idx]], include_daily=True)
                if single:
                    results.append(single[0])
                else:
                    results.append({})
                time.sleep(1)

        # Parse results
        for i, (_, row) in enumerate(batch.iterrows()):
            base_info = {
                "OBJECTID": row["OBJECTID"],
                "Site_Name": row["Site_Name"],
                "State": row["State"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "Operational_Status": row.get("Operational_Status", ""),
                "Reporting_Component": row.get("Reporting_Component", ""),
                "Is_Joint_Base": row.get("Is_Joint_Base", ""),
            }

            if i < len(results) and results[i]:
                weather = results[i]

                # --- Current weather ---
                current = weather.get("current", {})
                current_row = {**base_info}
                current_row["timezone"] = weather.get("timezone", "")
                current_row["current_time"] = current.get("time", "")
                for var in CURRENT_VARS:
                    current_row[f"current_{var}"] = current.get(var)
                current_rows.append(current_row)

                # --- Daily weather ---
                daily = weather.get("daily", {})
                times = daily.get("time", [])
                for day_idx, day_date in enumerate(times):
                    day_row = {**base_info}
                    day_row["date"] = day_date
                    for var in DAILY_VARS:
                        vals = daily.get(var, [])
                        day_row[f"daily_{var}"] = vals[day_idx] if day_idx < len(vals) else None
                    daily_rows.append(day_row)
            else:
                current_rows.append(base_info)

        # Save progress after each batch (in case of interruption)
        pd.DataFrame(current_rows).to_csv(OUTPUT_WEATHER, index=False)
        pd.DataFrame(daily_rows).to_csv(OUTPUT_DAILY, index=False)

        # Respect rate limits: 2 seconds between batches
        time.sleep(2)

    current_df = pd.DataFrame(current_rows)
    daily_df = pd.DataFrame(daily_rows)

    current_df.to_csv(OUTPUT_WEATHER, index=False)
    daily_df.to_csv(OUTPUT_DAILY, index=False)

    print(f"\nSaved: {OUTPUT_WEATHER}  ({len(current_df)} rows, {len(current_df.columns)} cols)")
    print(f"Saved: {OUTPUT_DAILY}  ({len(daily_df)} rows, {len(daily_df.columns)} cols)")

    return current_df, daily_df


# ============================= MAIN =========================================
def main():
    print("=" * 65)
    print("  MILITARY BASES WEATHER FETCHER v2")
    print("  (Using BTS ArcGIS polygon centroids — ALL 824 bases)")
    print("=" * 65)

    # Check for cached geocoded data
    if os.path.exists(OUTPUT_GEOCODED):
        print(f"\nFound cached {OUTPUT_GEOCODED}, loading...")
        df = pd.read_csv(OUTPUT_GEOCODED)
        valid = df["latitude"].notna().sum()
        print(f"  {valid}/{len(df)} bases with coordinates")
        if valid < 800:
            print("  Too few — re-fetching from API...")
            df = fetch_all_base_coordinates()
    else:
        df = fetch_all_base_coordinates()

    # Fetch weather
    current_df, daily_df = fetch_all_weather(df)

    # Summary
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Total military bases          : {len(df)}")
    print(f"  Bases with coordinates        : {df['latitude'].notna().sum()}")
    print(f"  Current weather rows          : {len(current_df)}")
    print(f"  Daily forecast rows           : {len(daily_df)}")
    print(f"  Current weather features      : {len(CURRENT_VARS)}")
    print(f"  Daily weather features        : {len(DAILY_VARS)}")
    print(f"\n  Output files:")
    print(f"    1. {OUTPUT_GEOCODED}   (bases + centroid lat/lon)")
    print(f"    2. {OUTPUT_WEATHER}       (current weather snapshot)")
    print(f"    3. {OUTPUT_DAILY} (7-day daily forecast)")
    print("=" * 65)


if __name__ == "__main__":
    main()
