import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.enums import Resampling

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"

THUMB = 256

_RENAME = {
    "ndvi_mean": "NDVI_mean",
    "nbr_mean":  "NBR_mean",
    "red_mean":  "B4_mean",
    "nir_mean":  "B5_mean",
    "swir_mean": "B7_mean",
}

os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")


def search_landsat(bbox, start_date, end_date, max_cloud=20, limit=5):
    payload = {
        "collections": ["landsat-c2-l2"],
        "bbox": bbox,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": limit,
        "query": {"eo:cloud_cover": {"lt": max_cloud}}
    }
    resp = requests.post(STAC_SEARCH, json=payload,
                         headers={"Content-Type": "application/json"})
    resp.raise_for_status()
    features = resp.json().get("features", [])
    print(f"Found {len(features)} scene(s)")
    return features


def sign_url(href):
    resp = requests.get(SAS_SIGN, params={"href": href})
    resp.raise_for_status()
    return resp.json()["href"]


def read_band(signed_url, band_name):
    with rasterio.open(signed_url) as ds:
        data = ds.read(
            1,
            out_shape=(THUMB, THUMB),
            resampling=Resampling.average,
        ).astype(np.float32)
    return data


def compute_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-10)


def compute_nbr(nir, swir):
    return (nir - swir) / (nir + swir + 1e-10)


def get_vegetation_features(scene):
    assets = scene["assets"]

    red_url  = sign_url(assets["red"]["href"])
    nir_url  = sign_url(assets["nir08"]["href"])
    swir_url = sign_url(assets["swir16"]["href"])

    # download the 3 bands in parallel
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_red  = ex.submit(read_band, red_url,  "red")
        f_nir  = ex.submit(read_band, nir_url,  "nir08")
        f_swir = ex.submit(read_band, swir_url, "swir16")
        red  = f_red.result()
        nir  = f_nir.result()
        swir = f_swir.result()

    ndvi = compute_ndvi(nir, red)
    nbr  = compute_nbr(nir, swir)

    return {
        "ndvi_mean": round(float(np.nanmean(ndvi)), 4),
        "nbr_mean":  round(float(np.nanmean(nbr)),  4),
        "red_mean":  round(float(np.nanmean(red)),  4),
        "nir_mean":  round(float(np.nanmean(nir)),  4),
        "swir_mean": round(float(np.nanmean(swir)), 4),
    }


def fetch_landsat_features(bbox, start_date, end_date, max_cloud=20, limit=5, resolution=0.5):
    scenes = search_landsat(list(bbox), start_date, end_date, max_cloud, limit)

    rows = []
    for scene in scenes:
        scene_bbox = scene.get("bbox", [])
        if len(scene_bbox) != 4:
            print(f"  Skipped scene {scene.get('id', '?')}: no bbox")
            continue

        dt_str = scene["properties"].get("datetime", "")
        date = pd.to_datetime(dt_str[:10]) if dt_str else None

        try:
            feats = get_vegetation_features(scene)
        except Exception as exc:
            print(f"  Skipped scene {scene.get('id', '?')}: {exc}")
            continue

        west, south, east, north = scene_bbox
        # Clip to the requested bbox so we don't emit cells outside the study area
        west  = max(west,  bbox[0])
        south = max(south, bbox[1])
        east  = min(east,  bbox[2])
        north = min(north, bbox[3])

        # All grid-cell centres inside the (clipped) scene footprint
        lats = np.arange(np.ceil(south / resolution) * resolution,
                         north + resolution * 0.01, resolution)
        lons = np.arange(np.ceil(west  / resolution) * resolution,
                         east  + resolution * 0.01, resolution)

        renamed = {_RENAME.get(k, k): v for k, v in feats.items()}
        for lat in lats:
            for lon in lons:
                row = {"latitude": round(float(lat), 4),
                       "longitude": round(float(lon), 4),
                       "date": date}
                row.update(renamed)
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["latitude", "longitude", "date",
                                     "NDVI_mean", "NBR_mean",
                                     "B4_mean", "B5_mean", "B7_mean"])
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["latitude", "longitude", "date"])


if __name__ == "__main__":
    BBOX = [-122.5, 38.5, -121.5, 39.5]
    START_DATE = "2024-07-01"
    END_DATE = "2024-07-31"

    print(f"Searching Landsat C2 L2 scenes over {BBOX}")
    print(f"Date range : {START_DATE} -> {END_DATE}")
    print(f"Cloud cover: < 20%\n")

    scenes = search_landsat(BBOX, START_DATE, END_DATE)

    all_features = []
    for scene in scenes:
        scene_id = scene["id"]
        date  = scene["properties"].get("datetime", "unknown")
        cloud = scene["properties"].get("eo:cloud_cover", "?")
        print(f"\nScene : {scene_id}")
        print(f"Date  : {date}  |  Cloud cover: {cloud}%")
        try:
            feats = get_vegetation_features(scene)
            feats["scene_id"] = scene_id
            feats["date"] = date
            all_features.append(feats)
            print(f"  NDVI mean : {feats['ndvi_mean']}")
            print(f"  NBR  mean : {feats['nbr_mean']}")
            print(f"  Red  mean : {feats['red_mean']}")
            print(f"  NIR  mean : {feats['nir_mean']}")
            print(f"  SWIR mean : {feats['swir_mean']}")
        except Exception as exc:
            print(f"  Skipped -- {exc}")

    print(f"\n{'='*60}")
    print(f"DONE: {len(all_features)} / {len(scenes)} scenes processed")
    print("=" * 60)
    for f in all_features:
        print(f)
