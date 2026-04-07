import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
MAX_DAYS_PER_REQUEST = 5  # API enforces [1..5] for both NRT and SP


def _fetch_firms_batch(
    bbox: tuple,
    date_str: str,
    days: int,
    map_key: str,
    source: str = "VIIRS_SNPP_NRT",
) -> pd.DataFrame:
    """
    Single request to FIRMS for `days` days ending on `date_str`.

    URL pattern:
        .../api/area/csv/{key}/{source}/{west},{south},{east},{north}/{days}/{date}
    """
    west, south, east, north = bbox
    bbox_str = f"{west},{south},{east},{north}"
    url = f"{FIRMS_BASE}/{map_key}/{source}/{bbox_str}/{days}/{date_str}"

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()

        if not resp.text.strip() or resp.text.startswith("<!"):
            return pd.DataFrame()

        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        return df

    except Exception as e:
        print(f"  [FIRMS] Warning: failed for {date_str} ({days}d): {e}")
        return pd.DataFrame()


def fetch_firms_data(
    bbox: tuple,
    start_date: str,
    end_date: str,
    map_key: str,
    source: str = "VIIRS_SNPP_SP",
) -> pd.DataFrame:
    """
    Fetch fire detections from NASA FIRMS for a bounding box and date range.

    Args:
        bbox:       (west, south, east, north) in decimal degrees
        start_date: "YYYY-MM-DD"
        end_date:   "YYYY-MM-DD"
        map_key:    FIRMS MAP_KEY from .env
        source:     FIRMS data source (default VIIRS_SNPP_NRT)

    Returns:
        DataFrame with columns: latitude, longitude, acq_date (datetime),
        acq_time, frp (float, MW), confidence, bright_ti4
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    total_days = (end_dt - start_dt).days + 1
    n_batches = -(-total_days // MAX_DAYS_PER_REQUEST)
    batches = []
    current = start_dt

    print(f"  Fetching FIRMS in {n_batches} batch(es) using source={source}...")

    while current <= end_dt:
        batch_end = min(current + timedelta(days=MAX_DAYS_PER_REQUEST - 1), end_dt)
        days_in_batch = (batch_end - current).days + 1

        date_str = batch_end.strftime("%Y-%m-%d")

        batch_df = _fetch_firms_batch(bbox, date_str, days_in_batch, map_key, source)
        if not batch_df.empty:
            batches.append(batch_df)

        current = batch_end + timedelta(days=1)
        time.sleep(0.3)

    if not batches:
        print("  [FIRMS] No data returned.")
        return pd.DataFrame(columns=["latitude", "longitude", "acq_date", "acq_time",
                                     "frp", "confidence", "bright_ti4"])

    df = pd.concat(batches, ignore_index=True)

    df.columns = [c.lower() for c in df.columns]

    if "acq_date" in df.columns:
        df["acq_date"] = pd.to_datetime(df["acq_date"])

    if "frp" in df.columns:
        df["frp"] = pd.to_numeric(df["frp"], errors="coerce")

    df = df.dropna(subset=["frp"])
    df = df.drop_duplicates()

    keep = [c for c in ["latitude", "longitude", "acq_date", "acq_time",
                         "frp", "confidence", "bright_ti4"] if c in df.columns]
    return df[keep].reset_index(drop=True)


def check_map_key_status(map_key: str) -> None:
    """Print remaining FIRMS transaction quota for the given MAP_KEY."""
    url = f"https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={map_key}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        status = pd.Series(resp.json())
        print("FIRMS MAP_KEY status:")
        print(status.to_string())
    except Exception as e:
        print(f"Could not retrieve MAP_KEY status: {e}")
        print(f"Try in your browser: {url}")


if __name__ == "__main__":
    map_key = os.environ.get("MAP_KEY", "")
    if not map_key:
        raise EnvironmentError("MAP_KEY not found. Check your .env file.")

    print("=== FIRMS MAP_KEY Status ===")
    check_map_key_status(map_key)

    print("\n=== Sample fetch: N. California, 3 days ===")
    BBOX = (-124.0, 36.0, -119.0, 42.0)
    df = fetch_firms_data(
        bbox=BBOX,
        start_date="2024-07-01",
        end_date="2024-07-05",
        map_key=map_key,
        source="VIIRS_SNPP_SP",
    )
    print(f"Rows: {len(df)}")
    if not df.empty:
        print(df.head())
        print(f"FRP range: {df['frp'].min():.1f} – {df['frp'].max():.1f} MW")
