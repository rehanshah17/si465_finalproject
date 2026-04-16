import os

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from folium.plugins import Draw
from streamlit_folium import st_folium

from firms import fetch_firms_data
from landsat import fetch_landsat_features
from pipeliner import (
    RES, TARGET, WEATHER_COLS,
    agg_firms, agg_power, agg_landsat,
    build_dataset, run_models,
)
from powers import fetch_power_data

load_dotenv()

st.set_page_config(page_title="Wildfire Intensity Predictor", layout="wide")
st.title("Wildfire Intensity Prediction Tool")
st.caption("Predicts weekly Fire Radiative Power using weather + satellite data")

for k, v in [("bbox_west", -124.0), ("bbox_east", -119.0),
             ("bbox_south", 36.0),  ("bbox_north", 42.0)]:
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_data(show_spinner=False, ttl=300)
def verify_map_key(key):
    if not key:
        return False
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv"
           f"/{key}/VIIRS_SNPP_SP/-122,37,-121,38/1/2024-07-01")
    try:
        r = requests.get(url, timeout=10)
        return r.status_code == 200 and not r.text.strip().startswith("<!")
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def cached_firms(bbox, start_str, end_str, map_key):
    return fetch_firms_data(bbox, start_str, end_str, map_key)


@st.cache_data(show_spinner=False)
def cached_power(bbox, start_str, end_str, resolution):
    return fetch_power_data(bbox, start_str, end_str, resolution=resolution)


@st.cache_data(show_spinner=False)
def cached_landsat(bbox, start_str, end_str, max_cloud, limit, resolution):
    return fetch_landsat_features(bbox, start_str, end_str,
                                  max_cloud=max_cloud, limit=limit, resolution=resolution)


def frp_to_hex(frp, vmin, vmax):
    norm = max(0.0, min(1.0, (frp - vmin) / (vmax - vmin + 1e-9)))
    r = int(180 + 75 * norm)
    g = int(180 * (1 - norm))
    return f"#{r:02x}{g:02x}00"


west  = st.session_state.bbox_west
east  = st.session_state.bbox_east
south = st.session_state.bbox_south
north = st.session_state.bbox_north

with st.sidebar:
    st.header("Selected Region")
    st.info(f"W {west}°  E {east}°  S {south}°  N {north}°")
    st.caption("Draw a rectangle on the map to change it.")

    st.header("Date Range")
    start_date = st.date_input("Start date", value=pd.to_datetime("2024-05-01"))
    end_date   = st.date_input("End date",   value=pd.to_datetime("2024-10-31"))

    st.header("Options")
    resolution    = st.selectbox("Grid resolution (degrees)", [0.5, 1.0, 2.0])
    max_cloud     = st.slider("Max Landsat cloud cover %", 0, 100, 50)
    landsat_limit = st.number_input("Max Landsat scenes", value=30, min_value=1, max_value=50)

    st.divider()
    map_key   = os.environ.get("MAP_KEY", "")
    key_valid = verify_map_key(map_key)
    if key_valid:
        st.success("MAP_KEY verified")
    else:
        st.error("MAP_KEY invalid or missing")

    run = st.button("Run Pipeline", type="primary", use_container_width=True, disabled=not key_valid)

mid_lat = (south + north) / 2
mid_lon = (west  + east)  / 2

input_map = folium.Map(location=[mid_lat, mid_lon], zoom_start=5, tiles="CartoDB positron")
folium.Rectangle(
    bounds=[[south, west], [north, east]],
    color="darkorange", weight=2, fill=True,
    fill_color="orange", fill_opacity=0.12,
    tooltip=f"W {west}  E {east}  S {south}  N {north}",
).add_to(input_map)
Draw(
    draw_options={
        "rectangle": {"shapeOptions": {"color": "darkorange"}},
        "polygon": False, "circle": False,
        "marker": False, "circlemarker": False, "polyline": False,
    },
    edit_options={"edit": False},
).add_to(input_map)

map_result = st_folium(input_map, width="100%", height=400,
                       returned_objects=["last_active_drawing"])

if map_result and map_result.get("last_active_drawing"):
    geom = map_result["last_active_drawing"].get("geometry", {})
    if geom.get("type") == "Polygon":
        coords = geom["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        st.session_state.bbox_west  = round(min(lons), 2)
        st.session_state.bbox_east  = round(max(lons), 2)
        st.session_state.bbox_south = round(min(lats), 2)
        st.session_state.bbox_north = round(max(lats), 2)
        st.rerun()

if not run:
    st.info("Draw a rectangle on the map to set your region, then click **Run Pipeline** in the sidebar.")
    st.stop()

bbox      = (west, south, east, north)
start_str = start_date.strftime("%Y-%m-%d")
end_str   = end_date.strftime("%Y-%m-%d")

firms_raw   = pd.DataFrame()
power_raw   = pd.DataFrame()
landsat_raw = pd.DataFrame()

st.header("Step 1 — Fetching Data")
c1, c2, c3 = st.columns(3)

with st.spinner("Fetching FIRMS fire detections..."):
    try:
        firms_raw = cached_firms(bbox, start_str, end_str, map_key)
        c1.metric("Fire Detections", len(firms_raw))
    except Exception as e:
        c1.error(f"FIRMS failed: {e}")

with st.spinner("Fetching NASA POWER weather..."):
    try:
        power_raw = cached_power(bbox,
                                 start_str.replace("-", ""),
                                 end_str.replace("-", ""),
                                 float(resolution))
        c2.metric("Weather Grid-Days", len(power_raw))
    except Exception as e:
        c2.error(f"POWER failed: {e}")

with st.spinner("Fetching Landsat vegetation scenes..."):
    try:
        landsat_raw = cached_landsat(bbox, start_str, end_str, max_cloud, int(landsat_limit), RES)
        c3.metric("Landsat Scenes", len(landsat_raw))
    except Exception as e:
        c3.warning(f"Landsat skipped: {e}")

st.header("Step 2 — Aggregating & Merging")
firms_agg   = agg_firms(firms_raw)
weather_agg = agg_power(power_raw)
landsat_agg = agg_landsat(landsat_raw) if not landsat_raw.empty else None
df          = build_dataset(firms_agg, weather_agg, landsat_agg)

if df.empty:
    st.error("Merge produced no rows. Try a wider region or longer date range.")
    st.stop()

n_weeks = df["week"].nunique()
n_cells = df[["lat_cell", "lon_cell"]].drop_duplicates().shape[0]
st.success(f"Dataset ready: **{len(df)} rows** — {n_weeks} weeks, {n_cells} grid cells")

with st.expander("Preview dataset"):
    preview_cols = [TARGET, "lat_cell", "lon_cell", "week"] + [c for c in WEATHER_COLS if c in df.columns]
    st.dataframe(df[preview_cols].head(30), use_container_width=True)

st.header("Step 3 — PCA + Regression Models")
st.write("**5 feature sets × 3 models**, 5-fold time-series CV. Target = weekly mean FRP per grid cell.")

with st.spinner("Training models..."):
    results = run_models(df)

if results.empty:
    st.warning("Not enough data to run models.")
    st.stop()

results = results.sort_values("mse_mean", ascending=True).reset_index(drop=True)
baseline_mse = float(np.mean((df[TARGET].values - df[TARGET].mean()) ** 2))
best = results.iloc[0]

st.success(
    f"Best: **{best['model']}** on **{best['features']}** "
    f"{'(uses PCA)' if best['uses_pca'] else '(no PCA)'} — "
    f"MSE {best['mse_mean']:.2f} ± {best['mse_std']:.2f}"
)
st.caption(f"Baseline MSE (predict mean FRP): {baseline_mse:.2f}")

tab_table, tab_map = st.tabs(["Table", "Fire Map"])

with tab_table:
    display = results.copy()
    pca_mask = display["uses_pca"].tolist()
    display = display.drop(columns=["uses_pca", "r2_mean", "r2_std"])
    display = display.rename(columns={
        "features": "Feature Set", "model": "Model",
        "n_samples": "N",
        "mse_mean": "MSE", "mse_std": "MSE ±",
    })

    def highlight_pca(row):
        idx = display.index.get_loc(row.name)
        if pca_mask[idx]:
            return ["background-color: #c8e6c9; color: #1b5e20"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display.style
               .apply(highlight_pca, axis=1)
               .format({"MSE": "{:.2f}", "MSE ±": "{:.2f}"}),
        use_container_width=True,
        height=530,
    )
    st.caption("Green rows use PCA on satellite features. Sorted by MSE ascending.")

with tab_map:
    if firms_agg.empty:
        st.info("No fire data to display.")
    else:
        fire_map = folium.Map(location=[mid_lat, mid_lon], zoom_start=6, tiles="CartoDB positron")
        vmin = firms_agg["mean_frp"].quantile(0.1)
        vmax = firms_agg["mean_frp"].quantile(0.9)
        half = RES / 2

        for _, row in firms_agg.iterrows():
            lat, lon, frp = row["lat_cell"], row["lon_cell"], row["mean_frp"]
            color = frp_to_hex(frp, vmin, vmax)
            folium.Rectangle(
                bounds=[[lat - half, lon - half], [lat + half, lon + half]],
                color=color, weight=1,
                fill=True, fill_color=color, fill_opacity=0.65,
                tooltip=(f"Grid ({lat:.1f}, {lon:.1f})<br>"
                         f"Mean FRP: {frp:.1f} MW<br>"
                         f"Max FRP: {row['max_frp']:.1f} MW<br>"
                         f"Detections: {int(row['fire_count'])}"),
            ).add_to(fire_map)

        if not firms_raw.empty:
            for _, row in firms_raw.iterrows():
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=2, color="darkred",
                    fill=True, fill_opacity=0.4,
                    tooltip=f"FRP: {row.get('frp', '?')} MW",
                ).add_to(fire_map)

        st.caption("Grid cells colored yellow→red by mean FRP intensity. Dots = individual detections.")
        st_folium(fire_map, width="100%", height=500, returned_objects=[])
