import os
import time
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def fetch_firms_data(bbox, start_date, end_date, map_key, source="VIIRS_SNPP_SP"):
    west, south, east, north = bbox
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    frames = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=4), end)
        ndays = (chunk_end - cur).days + 1
        date_str = chunk_end.strftime("%Y-%m-%d")

        url = f"{BASE_URL}/{map_key}/{source}/{west},{south},{east},{north}/{ndays}/{date_str}"
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            if r.text.strip() and not r.text.startswith("<!"):
                frames.append(pd.read_csv(StringIO(r.text)))
        except Exception as e:
            print(f"  FIRMS fetch failed ({date_str}): {e}")

        cur = chunk_end + timedelta(days=1)
        time.sleep(0.3)

    if not frames:
        print("  No FIRMS data returned")
        return pd.DataFrame(columns=["latitude", "longitude", "acq_date", "frp", "confidence"])

    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    df["frp"] = pd.to_numeric(df["frp"], errors="coerce")
    df = df.dropna(subset=["frp"]).drop_duplicates()

    keep = [c for c in ["latitude", "longitude", "acq_date", "acq_time",
                         "frp", "confidence", "bright_ti4"] if c in df.columns]
    return df[keep].reset_index(drop=True)


if __name__ == "__main__":
    map_key = os.environ.get("MAP_KEY", "")
    bbox = (-124.0, 36.0, -119.0, 42.0)

    df = fetch_firms_data(bbox, "2024-07-01", "2024-07-05", map_key)
    print(df.shape)
    print(df.head())
