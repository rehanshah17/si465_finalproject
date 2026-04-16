import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

from firms import fetch_firms_data
from landsat import fetch_landsat_features
from powers import fetch_power_data

load_dotenv()

BBOX  = (-124.0, 36.0, -119.0, 42.0)
START = "2024-05-01"
END   = "2024-10-31"
RES   = 0.5

WEATHER_COLS = ["T2M_mean", "RH2M_mean", "WS2M_mean", "PRECTOTCORR_sum", "ALLSKY_mean"]
SAT_COLS     = ["NDVI_mean", "NBR_mean", "B4_mean", "B5_mean", "B7_mean"]
TARGET       = "mean_frp"


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
    if df.empty:
        print("  Warning: FIRMS dataframe is empty.")
        return df
    df = snap_to_grid(df, "latitude", "longitude")
    df = add_week(df, "acq_date")
    return (
        df.groupby(["lat_cell", "lon_cell", "week", "week_start"])["frp"]
          .agg(mean_frp="mean", max_frp="max", fire_count="count")
          .reset_index()
    )


def agg_power(df):
    if df.empty:
        print("  Warning: weather dataframe is empty.")
        return df
    df = snap_to_grid(df, "latitude", "longitude")
    df = add_week(df, "date")
    res = df.groupby(["lat_cell", "lon_cell", "week"]).agg(
        T2M_mean=("T2M", "mean"),
        RH2M_mean=("RH2M", "mean"),
        WS2M_mean=("WS2M", "mean"),
        PRECTOTCORR_sum=("PRECTOTCORR", "sum"),
        ALLSKY_mean=("ALLSKY_SFC_SW_DWN", "mean"),
    ).reset_index()
    return res


def agg_landsat(df):
    if df.empty:
        print("  Warning: Landsat dataframe is empty, satellite features will be skipped.")
        return df
    df = snap_to_grid(df, "latitude", "longitude")
    df = add_week(df, "date")
    sat_present = [c for c in SAT_COLS if c in df.columns]
    return df.groupby(["lat_cell", "lon_cell", "week"])[sat_present].mean().reset_index()


def build_dataset(firms_agg, weather_agg, landsat_agg=None):
    merge_keys = ["lat_cell", "lon_cell", "week"]
    if firms_agg.empty or not all(c in firms_agg.columns for c in merge_keys):
        print("  No fire data after aggregation.")
        return pd.DataFrame()
    if weather_agg.empty or not all(c in weather_agg.columns for c in merge_keys):
        print("  No weather data after aggregation.")
        return pd.DataFrame()

    print("Merging fire + weather...")
    df = firms_agg.merge(weather_agg, on=merge_keys, how="inner")
    if df.empty:
        print("  Merge produced no rows, check that bboxes and date ranges overlap.")
        return df
    print(f"  {len(df)} rows after fire x weather merge")

    if landsat_agg is not None and not landsat_agg.empty:
        before = len(df)
        df = df.merge(landsat_agg, on=["lat_cell", "lon_cell", "week"], how="left")
        matched = df[SAT_COLS[0]].notna().sum() if SAT_COLS[0] in df.columns else 0
        print(f"  Landsat join: {matched}/{before} rows have satellite features")

    print(f"  Coverage: lat [{df.lat_cell.min()}, {df.lat_cell.max()}]  "
          f"lon [{df.lon_cell.min()}, {df.lon_cell.max()}]")
    return df


def cv_score(X, y, model, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()
    mses, r2s = [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        mses.append(mean_squared_error(y_te, y_pred))
        r2s.append(r2_score(y_te, y_pred))
    return float(np.mean(mses)), float(np.std(mses)), float(np.mean(r2s)), float(np.std(r2s))


def run_models(df):
    if TARGET not in df.columns:
        print("  Target column missing, skipping models.")
        return pd.DataFrame()

    df = df.sort_values("week").reset_index(drop=True)
    y = df[TARGET].values

    avail_weather = [c for c in WEATHER_COLS if c in df.columns]
    avail_sat     = [c for c in SAT_COLS     if c in df.columns]

    baseline_mse = float(np.mean((y - y.mean()) ** 2))
    print(f"\n  Baseline MSE (mean predictor): {baseline_mse:.2f}")

    models = {
        "linear": LinearRegression(),
        "ridge":  Ridge(alpha=1.0),
        "lasso":  Lasso(alpha=0.1, max_iter=5000),
    }

    results = []

    def record(feat_name, uses_pca, X):
        for name, model in models.items():
            mse_m, mse_s, r2_m, r2_s = cv_score(X, y, model)
            results.append(dict(
                features=feat_name, model=name, uses_pca=uses_pca,
                mse_mean=mse_m, mse_std=mse_s,
                r2_mean=r2_m,  r2_std=r2_s,
            ))

    if avail_weather:
        record("weather_only", False, df[avail_weather].fillna(0).values)
    else:
        print("  No weather columns, skipping weather-only.")

    if avail_sat:
        X_sat_raw = df[avail_sat].fillna(0).values
        record("sat_raw", False, X_sat_raw)

        X_sat_scaled = StandardScaler().fit_transform(X_sat_raw)
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_sat_scaled)
        n_comp = X_pca.shape[1]
        print(f"  PCA: {len(avail_sat)} satellite bands -> {n_comp} components (>=95% variance)")
        record("sat_pca", True, X_pca)

        if avail_weather:
            X_wx = df[avail_weather].fillna(0).values
            record("weather_sat_raw", False, np.hstack([X_wx, X_sat_raw]))
            record("weather_sat_pca", True,  np.hstack([X_wx, X_pca]))
    else:
        print("  No satellite columns, skipping satellite feature sets.")

    return pd.DataFrame(results)


def main():
    map_key = os.environ.get("MAP_KEY", "")
    if not map_key:
        print("Warning: MAP_KEY not set, FIRMS fetch will likely fail.")

    print(f"\n[1/4] Fetching FIRMS fire data ({START} -> {END})...")
    firms_raw = fetch_firms_data(BBOX, START, END, map_key)
    print(f"      {len(firms_raw)} fire detections")

    print(f"\n[2/4] Fetching POWER weather data...")
    power_raw = fetch_power_data(BBOX,
                                 START.replace("-", ""),
                                 END.replace("-", ""),
                                 resolution=RES)
    print(f"      {len(power_raw)} weather grid-day rows")

    print(f"\n[3/4] Fetching Landsat vegetation features...")
    landsat_raw = fetch_landsat_features(BBOX, START, END, max_cloud=20, limit=10)
    print(f"      {len(landsat_raw)} Landsat scenes processed")

    print("\n[4/4] Aggregating and merging...")
    firms_agg = agg_firms(firms_raw)
    weather_agg = agg_power(power_raw)
    landsat_agg = agg_landsat(landsat_raw) if not landsat_raw.empty else None

    df = build_dataset(firms_agg, weather_agg, landsat_agg)
    if df.empty:
        print("\nNo data to model. Exiting.")
        return

    print(f"\nFinal dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df[[TARGET] + [c for c in WEATHER_COLS if c in df.columns][:3]].describe())

    print("\n" + "=" * 60)
    print("MODEL COMPARISON  (5-fold time-series CV)")
    print("=" * 60)
    results = run_models(df)

    if not results.empty:
        results = results.sort_values("mse_mean")
        print("\n" + tabulate(
            results[["features", "model", "mse_mean", "mse_std", "r2_mean", "r2_std"]].round(4),
            headers="keys", tablefmt="github", index=False,
        ))


if __name__ == "__main__":
    main()
