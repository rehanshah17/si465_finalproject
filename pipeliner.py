import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
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
        sat_spatial = (
            landsat_agg.drop(columns=["week"], errors="ignore")
            .groupby(["lat_cell", "lon_cell"])[
                [c for c in SAT_COLS if c in landsat_agg.columns]
            ]
            .mean()
            .reset_index()
        )
        df = df.merge(sat_spatial, on=["lat_cell", "lon_cell"], how="left")
        matched = df[SAT_COLS[0]].notna().sum() if SAT_COLS[0] in df.columns else 0
        print(f"  Landsat join: {matched}/{before} rows have satellite features")

    print(f"  Coverage: lat [{df.lat_cell.min()}, {df.lat_cell.max()}]  "
          f"lon [{df.lon_cell.min()}, {df.lon_cell.max()}]")
    return df


def cv_score(X, y, model, n_splits=5):
    n_splits = min(n_splits, len(y) - 1)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mses, r2s = [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        m = clone(model)
        m.fit(X_tr_s, y_tr)
        y_pred = m.predict(X_te_s)
        mses.append(mean_squared_error(y_te, y_pred))
        r2s.append(r2_score(y_te, y_pred))
    return float(np.mean(mses)), float(np.std(mses)), float(np.mean(r2s)), float(np.std(r2s))


def cv_score_pca(X_sat, X_wx, y, model, n_splits=5):
    n_splits = min(n_splits, len(y) - 1)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mses, r2s = [], []
    for train_idx, test_idx in tscv.split(X_sat):
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler_sat = StandardScaler()
        X_sat_tr = scaler_sat.fit_transform(X_sat[train_idx])
        X_sat_te = scaler_sat.transform(X_sat[test_idx])

        pca = PCA(n_components=0.95)
        X_pca_tr = pca.fit_transform(X_sat_tr)
        X_pca_te = pca.transform(X_sat_te)

        if X_wx is not None:
            scaler_wx = StandardScaler()
            X_wx_tr = scaler_wx.fit_transform(X_wx[train_idx])
            X_wx_te = scaler_wx.transform(X_wx[test_idx])
            X_tr = np.hstack([X_pca_tr, X_wx_tr])
            X_te = np.hstack([X_pca_te, X_wx_te])
        else:
            X_tr, X_te = X_pca_tr, X_pca_te

        m = clone(model)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
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
        "rf":     RandomForestRegressor(n_estimators=100, random_state=42),
    }

    results = []

    def record_no_pca(feat_name, X, y_use=None):
        _y = y_use if y_use is not None else y
        for name, model in models.items():
            mse_m, mse_s, r2_m, r2_s = cv_score(X, _y, model)
            results.append(dict(features=feat_name, model=name, uses_pca=False,
                                n_samples=len(_y),
                                mse_mean=mse_m, mse_std=mse_s, r2_mean=r2_m, r2_std=r2_s))

    def record_pca(feat_name, X_sat, X_wx=None, y_use=None):
        _y = y_use if y_use is not None else y
        for name, model in models.items():
            mse_m, mse_s, r2_m, r2_s = cv_score_pca(X_sat, X_wx, _y, model)
            results.append(dict(features=feat_name, model=name, uses_pca=True,
                                n_samples=len(_y),
                                mse_mean=mse_m, mse_std=mse_s, r2_mean=r2_m, r2_std=r2_s))

    if avail_weather:
        record_no_pca("weather_only", df[avail_weather].fillna(0).values)
    else:
        print("  No weather columns, skipping weather-only.")

    if avail_sat:
        sat_mask = df[avail_sat].notna().any(axis=1)
        sat_coverage = sat_mask.mean()
        sat_count = sat_mask.sum()
        print(f"\n  Satellite coverage: {sat_coverage:.1%} of rows ({sat_count}/{len(df)}) have at least one non-null sat feature")

        if sat_count < 6:
            print("  WARNING: Too few rows with satellite data — skipping sat-based models.")
            print("           Increase limit or reduce max-cloud when fetching Landsat.")
        else:

            df_sat = df[sat_mask].reset_index(drop=True)
            y_sat  = df_sat[TARGET].values
            X_sat  = df_sat[avail_sat].values

            record_no_pca("sat_raw", X_sat, y_sat)
            record_pca("sat_pca",   X_sat, None, y_sat)

            if avail_weather:
                X_wx_sat = df_sat[avail_weather].fillna(0).values
                record_no_pca("weather_sat_raw",  np.hstack([X_wx_sat, X_sat]), y_sat)
                record_pca("weather_sat_pca", X_sat, X_wx_sat, y_sat)
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
    landsat_raw = fetch_landsat_features(BBOX, START, END, max_cloud=20, limit=10, resolution=RES)
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
