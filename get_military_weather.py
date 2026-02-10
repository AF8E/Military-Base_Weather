"""
Military Bases Weather Data Fetcher
====================================
1. Reads the NTAD Military Bases CSV
2. Geocodes each base using Open-Meteo Geocoding API
3. Fetches ALL available weather features from Open-Meteo Forecast API
4. Saves results to CSV files
"""

import pandas as pd
import requests
import time
import json
import os
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH = "NTAD_Military_Bases_-1644289556481787667.csv"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

OUTPUT_GEOCODED = "military_bases_geocoded.csv"
OUTPUT_WEATHER  = "military_bases_weather.csv"
OUTPUT_DAILY    = "military_bases_daily_weather.csv"

# US state code -> full name mapping (helps geocoding accuracy)
STATE_NAMES = {
    "al": "Alabama", "ak": "Alaska", "az": "Arizona", "ar": "Arkansas",
    "ca": "California", "co": "Colorado", "ct": "Connecticut", "de": "Delaware",
    "fl": "Florida", "ga": "Georgia", "hi": "Hawaii", "id": "Idaho",
    "il": "Illinois", "in": "Indiana", "ia": "Iowa", "ks": "Kansas",
    "ky": "Kentucky", "la": "Louisiana", "me": "Maine", "md": "Maryland",
    "ma": "Massachusetts", "mi": "Michigan", "mn": "Minnesota", "ms": "Mississippi",
    "mo": "Missouri", "mt": "Montana", "ne": "Nebraska", "nv": "Nevada",
    "nh": "New Hampshire", "nj": "New Jersey", "nm": "New Mexico", "ny": "New York",
    "nc": "North Carolina", "nd": "North Dakota", "oh": "Ohio", "ok": "Oklahoma",
    "or": "Oregon", "pa": "Pennsylvania", "ri": "Rhode Island", "sc": "South Carolina",
    "sd": "South Dakota", "tn": "Tennessee", "tx": "Texas", "ut": "Utah",
    "vt": "Vermont", "va": "Virginia", "wa": "Washington", "wv": "West Virginia",
    "wi": "Wisconsin", "wy": "Wyoming", "dc": "District of Columbia",
    "pr": "Puerto Rico", "gu": "Guam", "vi": "Virgin Islands",
    "as": "American Samoa", "mp": "Northern Mariana Islands",
}

# ---------------------------------------------------------------------------
# ALL Open-Meteo hourly weather variables
# ---------------------------------------------------------------------------
HOURLY_VARS = [
    # Basic
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "apparent_temperature", "precipitation_probability",
    "precipitation", "rain", "showers", "snowfall", "snow_depth",
    "weather_code",
    # Pressure
    "pressure_msl", "surface_pressure",
    # Clouds
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    # Visibility
    "visibility",
    # Evapotranspiration
    "evapotranspiration", "et0_fao_evapotranspiration",
    # Vapour
    "vapour_pressure_deficit",
    # Wind
    "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m",
    "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", "wind_direction_180m",
    "wind_gusts_10m",
    # Temperature at height
    "temperature_80m", "temperature_120m", "temperature_180m",
    # Soil temperature
    "soil_temperature_0cm", "soil_temperature_6cm",
    "soil_temperature_18cm", "soil_temperature_54cm",
    # Soil moisture
    "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm",
    "soil_moisture_3_to_9cm", "soil_moisture_9_to_27cm",
    "soil_moisture_27_to_81cm",
    # Additional
    "uv_index", "uv_index_clear_sky", "is_day", "sunshine_duration",
    "wet_bulb_temperature_2m", "cape", "lifted_index",
    "convective_inhibition", "freezing_level_height",
    "boundary_layer_height",
    # Solar radiation
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "direct_normal_irradiance", "global_tilted_irradiance",
    "terrestrial_radiation",
    "shortwave_radiation_instant", "direct_radiation_instant",
    "diffuse_radiation_instant", "direct_normal_irradiance_instant",
    "global_tilted_irradiance_instant", "terrestrial_radiation_instant",
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
    "wind_speed_10m_max", "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]

# ---------------------------------------------------------------------------
# Current weather variables
# ---------------------------------------------------------------------------
CURRENT_VARS = [
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "is_day", "precipitation", "rain", "showers", "snowfall",
    "weather_code", "cloud_cover", "pressure_msl", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
]


# ============================= STEP 1 ======================================
# Read CSV
# ===========================================================================
def load_bases(csv_path: str) -> pd.DataFrame:
    """Load the military bases CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} military bases from CSV.")
    print(f"Columns: {list(df.columns)}")
    return df


# ============================= STEP 2 ======================================
# Geocode each base using Open-Meteo Geocoding API
# ===========================================================================
def geocode_base(site_name: str, state_code: str) -> dict | None:
    """
    Query Open-Meteo Geocoding API and pick the best US result
    matching the state.
    """
    # Clean the name for search — remove common prefixes that hurt search
    search_name = site_name
    for prefix in ["NG ", "NAVWPNSTA ", "NAVPMOSSP ", "CSO ", "NAS ", "NOLF ",
                    "NAVSTA ", "NAF ", "NAWS ", "NSA ", "NSF ", "NSY ", "NWS "]:
        if search_name.startswith(prefix):
            search_name = search_name[len(prefix):]

    state_full = STATE_NAMES.get(state_code.lower(), "")

    # Try multiple query variations — plain name works best with this API
    queries_to_try = [
        search_name,
        site_name,
        f"{search_name} {state_full}",
    ]

    for query in queries_to_try:
        try:
            resp = requests.get(
                GEOCODE_URL,
                params={"name": query, "count": 10, "language": "en", "format": "json"},
                timeout=10,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            results = data.get("results", [])

            if not results:
                continue

            # Prefer results in the United States matching the state
            for r in results:
                country = r.get("country", "").lower()
                admin1 = r.get("admin1", "").lower()
                if "united states" in country or "us" == r.get("country_code", "").lower():
                    if state_full.lower() in admin1 or state_code.lower() in admin1:
                        return {
                            "latitude": r["latitude"],
                            "longitude": r["longitude"],
                            "elevation": r.get("elevation"),
                            "geocoded_name": r.get("name", ""),
                        }

            # Fallback: first US result
            for r in results:
                if "united states" in r.get("country", "").lower():
                    return {
                        "latitude": r["latitude"],
                        "longitude": r["longitude"],
                        "elevation": r.get("elevation"),
                        "geocoded_name": r.get("name", ""),
                    }

            # Last resort: first result
            r = results[0]
            return {
                "latitude": r["latitude"],
                "longitude": r["longitude"],
                "elevation": r.get("elevation"),
                "geocoded_name": r.get("name", ""),
            }

        except Exception:
            continue

    return None


def geocode_all_bases(df: pd.DataFrame) -> pd.DataFrame:
    """Geocode every base and add lat/lon columns."""
    lats, lons, elevs, geo_names = [], [], [], []
    failed = []

    # If a checkpoint file exists, load it to resume
    if os.path.exists(OUTPUT_GEOCODED):
        print(f"Found existing geocoded file '{OUTPUT_GEOCODED}', loading to resume...")
        cached = pd.read_csv(OUTPUT_GEOCODED)
        if len(cached) == len(df) and "latitude" in cached.columns:
            print("  -> All bases already geocoded. Skipping geocoding step.")
            return cached

    print("\nGeocoding 824 military bases (this may take a few minutes)...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Geocoding"):
        site_name = str(row.get("Site Name", row.get("Feature Name", "")))
        state_code = str(row.get("State Name Code", ""))

        result = geocode_base(site_name, state_code)

        if result:
            lats.append(result["latitude"])
            lons.append(result["longitude"])
            elevs.append(result["elevation"])
            geo_names.append(result["geocoded_name"])
        else:
            lats.append(None)
            lons.append(None)
            elevs.append(None)
            geo_names.append(None)
            failed.append(site_name)

        # Be respectful to the free API — small delay
        time.sleep(0.15)

    df = df.copy()
    df["latitude"] = lats
    df["longitude"] = lons
    df["elevation"] = elevs
    df["geocoded_name"] = geo_names

    # Save checkpoint
    df.to_csv(OUTPUT_GEOCODED, index=False)
    print(f"\nGeocoding complete. Saved to '{OUTPUT_GEOCODED}'")

    success = df["latitude"].notna().sum()
    print(f"  Successfully geocoded: {success}/{len(df)}")
    if failed:
        print(f"  Failed to geocode {len(failed)} bases:")
        for name in failed[:20]:
            print(f"    - {name}")
        if len(failed) > 20:
            print(f"    ... and {len(failed) - 20} more")

    return df


# ============================= STEP 3 ======================================
# Fetch weather data from Open-Meteo
# ===========================================================================
def fetch_weather_batch(latitudes: list, longitudes: list) -> dict | None:
    """
    Fetch current + daily weather for a batch of locations.
    Open-Meteo supports multiple coordinates in one request via comma separation.
    """
    params = {
        "latitude": ",".join(str(x) for x in latitudes),
        "longitude": ",".join(str(x) for x in longitudes),
        "current": ",".join(CURRENT_VARS),
        "daily": ",".join(DAILY_VARS),
        "timezone": "auto",
        "forecast_days": 7,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
    }

    try:
        resp = requests.get(FORECAST_URL, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"  API error {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"  Request error: {e}")
        return None


def fetch_all_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch current weather for all geocoded bases."""
    # Filter to only bases with valid coordinates
    valid = df.dropna(subset=["latitude", "longitude"]).copy()
    print(f"\nFetching weather for {len(valid)} bases with valid coordinates...")

    BATCH_SIZE = 50  # Open-Meteo can handle many coords at once
    all_rows = []

    for start in tqdm(range(0, len(valid), BATCH_SIZE), desc="Fetching weather"):
        batch = valid.iloc[start : start + BATCH_SIZE]
        lats = batch["latitude"].tolist()
        lons = batch["longitude"].tolist()

        data = fetch_weather_batch(lats, lons)
        if data is None:
            # Try one-by-one as fallback
            for i, (_, row) in enumerate(batch.iterrows()):
                single = fetch_weather_batch([row["latitude"]], [row["longitude"]])
                if single:
                    data_item = single if not isinstance(single, list) else single[0]
                    row_dict = _extract_current(data_item, row)
                    all_rows.append(row_dict)
                time.sleep(0.3)
            continue

        # Parse the batch response
        # When multiple coords, Open-Meteo returns a list
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "latitude" in data:
            # Single result (or list wrapped in single)
            items = [data]
        else:
            items = data if isinstance(data, list) else [data]

        for i, (_, row) in enumerate(batch.iterrows()):
            if i < len(items):
                row_dict = _extract_current(items[i], row)
            else:
                row_dict = _base_info(row)
            all_rows.append(row_dict)

        time.sleep(0.5)  # rate-limit courtesy

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(OUTPUT_WEATHER, index=False)
    print(f"\nWeather data saved to '{OUTPUT_WEATHER}'")
    print(f"  Total rows: {len(result_df)}")
    print(f"  Total columns: {len(result_df.columns)}")
    return result_df


def _base_info(row) -> dict:
    """Extract base identification fields."""
    return {
        "OBJECTID": row.get("OBJECTID"),
        "Site_Name": row.get("Site Name", row.get("Feature Name", "")),
        "State": row.get("State Name Code", ""),
        "latitude": row.get("latitude"),
        "longitude": row.get("longitude"),
        "elevation": row.get("elevation"),
        "Is_Joint_Base": row.get("Is Joint Base", ""),
        "Operational_Status": row.get("Site Operational Status", ""),
        "Reporting_Component": row.get("Site Reporting Component Code", ""),
    }


def _extract_current(weather_data: dict, row) -> dict:
    """Extract current weather values from API response into a flat dict."""
    info = _base_info(row)

    current = weather_data.get("current", {})
    for var in CURRENT_VARS:
        info[f"current_{var}"] = current.get(var)

    # Also store current time and units
    info["current_time"] = current.get("time")
    info["timezone"] = weather_data.get("timezone", "")

    return info


# ============================= STEP 4 ======================================
# Fetch daily forecasts
# ===========================================================================
def fetch_daily_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch 7-day daily forecast for all geocoded bases."""
    valid = df.dropna(subset=["latitude", "longitude"]).copy()
    print(f"\nFetching 7-day daily forecasts for {len(valid)} bases...")

    BATCH_SIZE = 30
    all_rows = []

    for start in tqdm(range(0, len(valid), BATCH_SIZE), desc="Fetching daily"):
        batch = valid.iloc[start : start + BATCH_SIZE]
        lats = batch["latitude"].tolist()
        lons = batch["longitude"].tolist()

        data = fetch_weather_batch(lats, lons)
        if data is None:
            continue

        items = data if isinstance(data, list) else [data]

        for i, (_, row) in enumerate(batch.iterrows()):
            if i >= len(items):
                continue
            item = items[i]
            daily = item.get("daily", {})
            times = daily.get("time", [])

            for day_idx, day_date in enumerate(times):
                day_row = _base_info(row)
                day_row["date"] = day_date
                for var in DAILY_VARS:
                    vals = daily.get(var, [])
                    day_row[f"daily_{var}"] = vals[day_idx] if day_idx < len(vals) else None
                all_rows.append(day_row)

        time.sleep(0.5)

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(OUTPUT_DAILY, index=False)
    print(f"\nDaily weather data saved to '{OUTPUT_DAILY}'")
    print(f"  Total rows: {len(result_df)} (bases x days)")
    print(f"  Total columns: {len(result_df.columns)}")
    return result_df


# ============================= MAIN =========================================
def main():
    print("=" * 65)
    print("  MILITARY BASES WEATHER DATA FETCHER (Open-Meteo)")
    print("=" * 65)

    # Step 1: Load CSV
    df = load_bases(CSV_PATH)

    # Step 2: Geocode
    df = geocode_all_bases(df)

    # Step 3: Fetch current weather snapshot
    weather_df = fetch_all_weather(df)

    # Step 4: Fetch 7-day daily forecasts
    daily_df = fetch_daily_weather(df)

    # Summary
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Total military bases in CSV : {len(df)}")
    print(f"  Successfully geocoded       : {df['latitude'].notna().sum()}")
    print(f"  Current weather rows        : {len(weather_df)}")
    print(f"  Daily forecast rows         : {len(daily_df)}")
    print(f"\n  Output files:")
    print(f"    1. {OUTPUT_GEOCODED}  (bases + lat/lon)")
    print(f"    2. {OUTPUT_WEATHER}        (current weather snapshot)")
    print(f"    3. {OUTPUT_DAILY}  (7-day daily forecast)")
    print("=" * 65)


if __name__ == "__main__":
    main()
