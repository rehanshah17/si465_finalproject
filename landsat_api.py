import requests
import numpy as np
from rasterio.io import MemoryFile

STAC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_SIGN    = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"


def search_landsat(bbox, start_date, end_date, max_cloud=20, limit=5):
    """Search Planetary Computer STAC for Landsat C2 L2 scenes."""
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
    data = resp.json()
    features = data.get("features", [])
    print(f"Found {len(features)} scene(s)")
    return features


def sign_url(href):
    """Get a short-lived signed (SAS) URL for a Planetary Computer asset."""
    resp = requests.get(SAS_SIGN, params={"href": href})
    resp.raise_for_status()
    return resp.json()["href"]


def read_band(href, band_name):
    """Sign, download, and read a single GeoTIFF band into a float array."""
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
    """Return a dict of spectral statistics for one STAC scene."""
    assets = scene["assets"]
    red  = read_band(assets["red"]["href"],    "red")
    nir  = read_band(assets["nir08"]["href"],  "nir08")
    swir = read_band(assets["swir16"]["href"], "swir16")

    ndvi = compute_ndvi(nir, red)
    nbr  = compute_nbr(nir, swir)

    return {
        "ndvi_mean": round(float(np.nanmean(ndvi)), 4),
        "nbr_mean":  round(float(np.nanmean(nbr)),  4),
        "red_mean":  round(float(np.nanmean(red)),   4),
        "nir_mean":  round(float(np.nanmean(nir)),   4),
        "swir_mean": round(float(np.nanmean(swir)),  4),
    }


if __name__ == "__main__":
    BBOX       = [-122.5, 38.5, -121.5, 39.5]
    START_DATE = "2024-07-01"
    END_DATE   = "2024-07-31"

    print(f"Searching Landsat C2 L2 scenes over {BBOX}")
    print(f"Date range : {START_DATE} -> {END_DATE}")
    print(f"Cloud cover: < 20%\n")

    scenes = search_landsat(BBOX, START_DATE, END_DATE)

    all_features = []
    for scene in scenes:
        scene_id = scene["id"]
        date     = scene["properties"].get("datetime", "unknown")
        cloud    = scene["properties"].get("eo:cloud_cover", "?")
        print(f"\nScene : {scene_id}")
        print(f"Date  : {date}  |  Cloud cover: {cloud}%")
        try:
            feats = get_vegetation_features(scene)
            feats["scene_id"] = scene_id
            feats["date"]     = date
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
    print("="*60)
    for f in all_features:
        print(f)
