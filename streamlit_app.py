import os

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv
from folium.plugins import Draw
from streamlit_folium import st_folium

from firms import fetch_firms_data
from landsat import fetch_landsat_features
from pipeliner import (
    RES, TARGET, WEATHER_COLS, SAT_COLS,
    agg_firms, agg_power, agg_landsat,
    build_dataset, run_models,
)
from powers import fetch_power_data

load_dotenv()

st.set_page_config(page_title="Wildfire Intensity Predictor", layout="wide")
st.title("Wildfire Intensity Prediction Tool")
st.caption("SI 465 Final Project — predicts weekly Fire Radiative Power using weather + satellite data")

# use unbound keys so we can freely update them from the map drawing
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
def cached_landsat(bbox, start_str, end_str, max_cloud, limit):
    return fetch_landsat_features(bbox, start_str, end_str, max_cloud=max_cloud, limit=limit)


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
    resolution    = st.selectbox("Grid resolution (degrees)", [1.0, 0.5, 2.0])
    max_cloud     = st.slider("Max Landsat cloud cover %", 0, 100, 20)
    landsat_limit = st.number_input("Max Landsat scenes", value=10, min_value=1, max_value=50)

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
        landsat_raw = cached_landsat(bbox, start_str, end_str, max_cloud, int(landsat_limit))
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

results = results.sort_values("r2_mean", ascending=False).reset_index(drop=True)
baseline_mse = float(np.mean((df[TARGET].values - df[TARGET].mean()) ** 2))
best = results.iloc[0]

st.success(
    f"Best: **{best['model']}** on **{best['features']}** "
    f"{'(uses PCA)' if best['uses_pca'] else '(no PCA)'} — "
    f"R² {best['r2_mean']:.3f},  MSE {best['mse_mean']:.2f}"
)
st.caption(f"Baseline MSE (predict mean FRP): {baseline_mse:.2f}")

tab_table, tab_map, tab_charts = st.tabs(["Table", "Fire Map", "Charts"])

with tab_table:
    display = results.copy()
    display.insert(2, "PCA", display["uses_pca"].map({True: "✓", False: "✗"}))
    display = display.drop(columns=["uses_pca"])
    display = display.rename(columns={
        "features": "Feature Set", "model": "Model",
        "mse_mean": "MSE", "mse_std": "MSE ±",
        "r2_mean": "R²", "r2_std": "R² ±",
    })

    def highlight_pca(row):
        if row["PCA"] == "✓":
            return ["background-color: #e8f5e9"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display.style
               .apply(highlight_pca, axis=1)
               .format({"MSE": "{:.2f}", "MSE ±": "{:.2f}", "R²": "{:.3f}", "R² ±": "{:.3f}"}),
        use_container_width=True,
        height=530,
    )
    st.caption("Green rows use PCA on satellite features.")

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

with tab_charts:
    st.subheader("R² vs MSE — all models")
    st.caption("Top-left is best (high R², low MSE). Dashed line = baseline MSE.")

    scatter_df = results.copy()
    scatter_df["PCA"] = scatter_df["uses_pca"].map({True: "Uses PCA", False: "No PCA"})
    fig1 = px.scatter(
        scatter_df,
        x="mse_mean", y="r2_mean",
        color="features",
        symbol="model",
        error_x="mse_std",
        error_y="r2_std",
        labels={"mse_mean": "MSE (lower is better)", "r2_mean": "R²",
                "features": "Feature Set", "model": "Model"},
        hover_data={"features": True, "model": True,
                    "mse_mean": ":.2f", "r2_mean": ":.3f",
                    "mse_std": ":.2f", "r2_std": ":.3f"},
    )
    fig1.add_vline(x=baseline_mse, line_dash="dash", line_color="grey",
                   annotation_text="baseline MSE", annotation_position="top right")
    fig1.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.5))
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("PCA impact per model")
    st.caption("R² with vs without PCA for the same model and base features.")

    pca_pairs = {"sat_raw": "sat_pca", "weather_sat_raw": "weather_sat_pca"}
    rows = []
    for raw_feat, pca_feat in pca_pairs.items():
        for model_name in results["model"].unique():
            raw_row = results[(results["features"] == raw_feat) & (results["model"] == model_name)]
            pca_row = results[(results["features"] == pca_feat) & (results["model"] == model_name)]
            if raw_row.empty or pca_row.empty:
                continue
            rows.append(dict(
                model=model_name,
                comparison=f"{raw_feat} → {pca_feat}",
                r2_no_pca=float(raw_row["r2_mean"].iloc[0]),
                r2_pca=float(pca_row["r2_mean"].iloc[0]),
                delta=float(pca_row["r2_mean"].iloc[0]) - float(raw_row["r2_mean"].iloc[0]),
            ))

    if rows:
        pca_df = pd.DataFrame(rows)
        melted = pca_df.melt(
            id_vars=["model", "comparison"],
            value_vars=["r2_no_pca", "r2_pca"],
            var_name="version", value_name="R²",
        )
        melted["version"] = melted["version"].map({"r2_no_pca": "Without PCA", "r2_pca": "With PCA"})
        melted["label"] = melted["model"] + " / " + melted["comparison"]

        fig2 = px.bar(
            melted,
            x="label", y="R²",
            color="version",
            barmode="group",
            color_discrete_map={"Without PCA": "#f4a261", "With PCA": "#2a9d8f"},
            labels={"label": "Model / Feature Set"},
            title="R² with vs without PCA (same model, same base features)",
        )
        fig2.update_layout(xaxis_tickangle=-30,
                           legend=dict(orientation="h", yanchor="bottom", y=-0.4))
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("R² gain from adding PCA")
        delta_df = pca_df[["model", "comparison", "delta"]].copy()
        delta_df["delta"] = delta_df["delta"].round(4)
        delta_df.columns = ["Model", "Feature Set Pair", "ΔR² (PCA − Raw)"]
        st.dataframe(delta_df, use_container_width=True)
    else:
        st.info("Need satellite data to compute PCA impact. Landsat scenes may be missing.")
