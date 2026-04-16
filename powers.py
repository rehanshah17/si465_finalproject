import time
from itertools import product

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = "T2M,RH2M,WS2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN"


def fetch_power_data(bbox, start_date, end_date, resolution=0.5):
    west, south, east, north = bbox
    lats = np.arange(south + resolution / 2, north, resolution)
    lons = np.arange(west + resolution / 2, east, resolution)
    cells = list(product(lats, lons))

    print(f"  Pulling POWER for {len(cells)} grid cells...")

    all_rows = []
    for i, (lat, lon) in enumerate(cells, 1):
        if i % 20 == 0:
            print(f"    {i}/{len(cells)}")

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
        except Exception as e:
            print(f"  POWER failed ({lat:.1f},{lon:.1f}): {e}")
            time.sleep(0.2)
            continue

        dates = list(next(iter(param_data.values())).keys())
        for d in dates:
            row = {"date": pd.to_datetime(d, format="%Y%m%d"), "lat": lat, "lon": lon}
            for p, vals in param_data.items():
                v = float(vals.get(d, -999))
                row[p] = np.nan if v == -999.0 else v
            all_rows.append(row)

        time.sleep(0.2)

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
