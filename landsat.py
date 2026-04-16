import requests
import numpy as np
import pandas as pd
from rasterio.io import MemoryFile

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"

_RENAME = {
    "ndvi_mean": "NDVI_mean",
    "nbr_mean":  "NBR_mean",
    "red_mean":  "B4_mean",
    "nir_mean":  "B5_mean",
    "swir_mean": "B7_mean",
}


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


def read_band(href, band_name):
    signed = sign_url(href)
    print(f"  Downloading {band_name} ...", end=" ", flush=True)
    resp = requests.get(signed)
    resp.raise_for_status()
    print(f"{len(resp.content) // 1024} KB")
    with MemoryFile(resp.content) as mf:
        with mf.open() as ds:
            return ds.read(1).astype(np.float32)


def compute_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-10)


def compute_nbr(nir, swir):
    return (nir - swir) / (nir + swir + 1e-10)


def get_vegetation_features(scene):
    assets = scene["assets"]
    red  = read_band(assets["red"]["href"],    "red")
    nir  = read_band(assets["nir08"]["href"],  "nir08")
    swir = read_band(assets["swir16"]["href"], "swir16")

    ndvi = compute_ndvi(nir, red)
    nbr  = compute_nbr(nir, swir)

    return {
        "ndvi_mean": round(float(np.nanmean(ndvi)), 4),
        "nbr_mean":  round(float(np.nanmean(nbr)),  4),
        "red_mean":  round(float(np.nanmean(red)),  4),
        "nir_mean":  round(float(np.nanmean(nir)),  4),
        "swir_mean": round(float(np.nanmean(swir)), 4),
    }


def fetch_landsat_features(bbox, start_date, end_date, max_cloud=20, limit=5):
    scenes = search_landsat(list(bbox), start_date, end_date, max_cloud, limit)

    rows = []
    for scene in scenes:
        scene_bbox = scene.get("bbox", [])
        if len(scene_bbox) == 4:
            lat = (scene_bbox[1] + scene_bbox[3]) / 2
            lon = (scene_bbox[0] + scene_bbox[2]) / 2
        else:
            lat, lon = None, None

        dt_str = scene["properties"].get("datetime", "")
        date = pd.to_datetime(dt_str[:10]) if dt_str else None

        try:
            feats = get_vegetation_features(scene)
        except Exception as exc:
            print(f"  Skipped scene {scene.get('id', '?')}: {exc}")
            continue

        row = {"latitude": lat, "longitude": lon, "date": date}
        row.update({_RENAME.get(k, k): v for k, v in feats.items()})
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["latitude", "longitude", "date",
                                     "NDVI_mean", "NBR_mean",
                                     "B4_mean", "B5_mean", "B7_mean"])
    return pd.DataFrame(rows)


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
