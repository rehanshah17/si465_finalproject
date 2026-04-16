import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = "T2M,RH2M,WS2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN"


def _fetch_cell(lat, lon, start_date, end_date, retries=2):
    for attempt in range(retries + 1):
        try:
            r = requests.get(BASE_URL, params={
                "parameters": PARAMS,
                "community": "AG",
                "longitude": lon,
                "latitude": lat,
                "start": start_date,
                "end": end_date,
                "format": "json",
            }, timeout=60)
            r.raise_for_status()
            param_data = r.json()["properties"]["parameter"]
            rows = []
            dates = list(next(iter(param_data.values())).keys())
            for d in dates:
                row = {"date": pd.to_datetime(d, format="%Y%m%d"), "lat": lat, "lon": lon}
                for p, vals in param_data.items():
                    v = float(vals.get(d, -999))
                    row[p] = np.nan if v == -999.0 else v
                rows.append(row)
            return rows
        except Exception as e:
            if attempt < retries:
                time.sleep(0.5)
            else:
                print(f"  POWER failed ({lat:.1f},{lon:.1f}): {e}")
                return []


def fetch_power_data(bbox, start_date, end_date, resolution=0.5, workers=10):
    west, south, east, north = bbox
    lats = np.arange(south + resolution / 2, north, resolution)
    lons = np.arange(west + resolution / 2, east, resolution)
    cells = list(product(lats, lons))

    print(f"  Pulling POWER for {len(cells)} grid cells ({workers} parallel workers)...")

    all_rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_cell, lat, lon, start_date, end_date): (lat, lon)
            for lat, lon in cells
        }
        done = 0
        for future in as_completed(futures):
            all_rows.extend(future.result())
            done += 1
            if done % 20 == 0:
                print(f"    {done}/{len(cells)} cells done")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.rename(columns={"lat": "latitude", "lon": "longitude"})
    return df.sort_values(["latitude", "longitude", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    bbox = (-124.0, 36.0, -119.0, 42.0)
    df = fetch_power_data(bbox, "20240701", "20240710", resolution=1.0)
    print(df.shape)
    print(df.head())
