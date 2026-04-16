import os
import warnings

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from firms import fetch_firms_data
from powers import fetch_power_data
from landsat import fetch_landsat_features

load_dotenv()

BBOX = (-124.0, 36.0, -119.0, 42.0)   # west, south, east, north
START = "2024-05-01"
END   = "2024-10-31"
RES   = 0.5   # grid resolution in degrees

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

WEATHER_COLS  = ["T2M_mean", "RH2M_mean", "WS2M_mean", "PRECTOTCORR_sum", "ALLSKY_mean"]
SAT_COLS      = ["NDVI_mean", "NBR_mean", "B4_mean", "B5_mean", "B7_mean"]
TARGET        = "mean_frp"

#Api's dont return same precision for locations so we need something to numbers up to align for merging datasets.
def snap_to_grid(df, lat_col, lon_col):
    df = df.copy()
    df["lat_cell"] = (df[lat_col] / RES).round() * RES
    df["lon_cell"] = (df[lon_col] / RES).round() * RES
    return df


def add_week(df, date_col):
    df = df.copy()
    d = pd.to_datetime(df[date_col])
    df["week_start"] = (d - pd.to_timedelta(d.dt.dayofweek, unit="D")).dt.normalize()
    df["week"] = df["week_start"].dt.strftime("%Y-W%V")
    return df


def agg_firms(df):
    df = snap_to_grid(df, "latitude", "longitude")
    df = add_week(df, "acq_date")
    return (df.groupby(["lat_cell", "lon_cell", "week", "week_start"])["frp"]
              .agg(mean_frp="mean", max_frp="max", fire_count="count")
              .reset_index())


def agg_power(df):
    df = snap_to_grid(df, "latitude", "longitude")
    df = add_week(df, "date")

    mapping = {
        "T2M": ("T2M_mean", "mean"),
        "RH2M": ("RH2M_mean", "mean"),
        "WS2M": ("WS2M_mean", "mean"),
        "PRECTOTCORR": ("PRECTOTCORR_sum", "sum"),
        "ALLSKY_SFC_SW_DWN": ("ALLSKY_mean", "mean"),
    }
    kwargs = {new: (src, fn) for src, (new, fn) in mapping.items() if src in df.columns}
    return df.groupby(["lat_cell", "lon_cell", "week"]).agg(**kwargs).reset_index()


def build_dataset(firms_agg, weather_agg, landsat_agg):
    df = firms_agg.copy()

    if not weather_agg.empty:
        df = df.merge(weather_agg, on=["lat_cell", "lon_cell", "week"], how="left")

    df = df[df["fire_count"] > 0].copy()

    feat_cols = [c for c in WEATHER_COLS + SAT_COLS if c in df.columns]
    df = df.dropna(subset=[TARGET] + feat_cols)
    df = df.sort_values("week_start").reset_index(drop=True)

    print(f"  dataset: {df.shape[0]} rows, {df[['lat_cell','lon_cell']].drop_duplicates().shape[0]} cells")
    return df


def run_pca(df, cols, n=5):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    present = [c for c in cols if c in df.columns]
    if not present:
        return df, None

    n = min(n, len(present))
    X = StandardScaler().fit_transform(df[present].values)
    pca = PCA(n_components=n, random_state=42).fit(X)
    pcs = pca.transform(X)

    print("  PCA variance explained:")
    cum = 0
    for i, v in enumerate(pca.explained_variance_ratio_, 1):
        cum += v
        print(f"    PC{i}: {v:.3f}  cumulative: {cum:.3f}")

    df = df.copy()
    for i in range(n):
        df[f"PC{i+1}"] = pcs[:, i]
    return df, pca


def cv_score(model, X, y, tscv):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_validate

    pipe = Pipeline([("sc", StandardScaler()), ("m", model)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = cross_validate(pipe, X, y, cv=tscv,
                           scoring=["neg_mean_squared_error", "r2"],
                           error_score=np.nan)
    return {
        "MSE_mean": float(np.nanmean(-s["test_neg_mean_squared_error"])),
        "MSE_std":  float(np.nanstd(-s["test_neg_mean_squared_error"])),
        "R2_mean":  float(np.nanmean(s["test_r2"])),
        "R2_std":   float(np.nanstd(s["test_r2"])),
    }


def tune_ridge(X, y, tscv):
    from sklearn.linear_model import Ridge

    best, best_r2 = RIDGE_ALPHAS[0], -np.inf
    print("  Ridge alpha search:")
    for a in RIDGE_ALPHAS:
        m = cv_score(Ridge(alpha=a), X, y, tscv)
        if m["R2_mean"] > best_r2:
            best_r2 = m["R2_mean"]
            best = a
    return best


def run_comparison(df, n_pca=5, n_splits=5):
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit

    df, _ = run_pca(df, SAT_COLS, n=n_pca)
    pc_cols = [f"PC{i+1}" for i in range(n_pca) if f"PC{i+1}" in df.columns]

    weather = [c for c in WEATHER_COLS if c in df.columns]
    sat_raw = [c for c in SAT_COLS if c in df.columns]

    fsets = {}
    if weather:
        fsets["weather_only"] = weather
    if sat_raw:
        fsets["satellite_only"] = sat_raw
    if pc_cols:
        fsets["sat_pca_weather"] = pc_cols + weather

    y = df[TARGET]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for fs_name, fcols in fsets.items():
        X = df[fcols]
        if len(X) <= n_splits:
            print(f"  skipping {fs_name}: only {len(X)} samples")
            continue

        print(f"\n[{fs_name}]")

        m = cv_score(LinearRegression(), X, y, tscv)
        rows.append({"Model": "LinearRegression", "Feature_Set": fs_name, "Alpha": "-", **m})

        best_a = tune_ridge(X, y, tscv)
        m = cv_score(Ridge(alpha=best_a), X, y, tscv)
        rows.append({"Model": "Ridge", "Feature_Set": fs_name, "Alpha": best_a, **m})

        m = cv_score(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), X, y, tscv)
        rows.append({"Model": "RandomForest", "Feature_Set": fs_name, "Alpha": "-", **m})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    map_key = os.environ.get("MAP_KEY", "").strip()
    if not map_key:
        raise SystemExit("MAP_KEY missing from .env")

    west, south, east, north = BBOX
    grid_lons = np.arange(west + RES / 2, east, RES)
    grid_lats = np.arange(south + RES / 2, north, RES)

    print(f"grid: {len(grid_lons)}x{len(grid_lats)} cells, {START} to {END}\n")

    print("fetching FIRMS...")
    firms_df = fetch_firms_data((west, south, east, north), START, END, map_key)
    print(f"  {len(firms_df)} detections\n")

    print("fetching POWER...")
    weather_df = fetch_power_data(
        (south, west, north, east),
        START.replace("-", ""),
        END.replace("-", ""),
        resolution=RES,
    )
    print(f"  {len(weather_df)} records\n")

    """ print("fetching Landsat...")
    landsat_df = fetch_landsat_features() 
    print(f"  {len(landsat_df)} scene-cell obs\n") """

    firms_agg   = agg_firms(firms_df) if not firms_df.empty else pd.DataFrame()
    weather_agg = agg_power(weather_df) if not weather_df.empty else pd.DataFrame()
    #landsat_agg = agg_landsat(landsat_df)

    if firms_agg.empty:
        raise SystemExit("No fire data -- cannot proceed")

    print("building dataset...")
    dataset = build_dataset(firms_agg, weather_agg, landsat_agg)
    print()

    results = run_comparison(dataset)

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    try:
        from tabulate import tabulate
        print(tabulate(results, headers="keys", tablefmt="github",
                       floatfmt=".4f", showindex=False))
    except ImportError:
        print(results.to_string(index=False))

    print()
    for fs in results["Feature_Set"].unique():
        sub = results[results["Feature_Set"] == fs]
        best = sub.loc[sub["R2_mean"].idxmax()]
        print(f"  {fs}: {best['Model']}  R2={best['R2_mean']:.4f}  alpha={best['Alpha']}")
