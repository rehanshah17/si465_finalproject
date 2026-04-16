import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Local imports from our data scripts
from firms import fetch_firms_data
from powers import fetch_power_data
from landsat import fetch_landsat_features

load_dotenv()

BBOX = (-124.0, 36.0, -119.0, 42.0) 
START = "2024-05-01"
END   = "2024-10-31"


RES = 0.5

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
        print("Warning: FIRMS dataframe is empty.")
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
        print("Warning: Weather dataframe is empty.")
        return df

    df = snap_to_grid(df, "latitude", "longitude")
    df = add_week(df, "date")

    agg_map = {
        "T2M": "T2M_mean",
        "RH2M": "RH2M_mean",
        "WS2M": "WS2M_mean",
        "PRECTOTCORR": "PRECTOTCORR_sum",
        "ALLSKY_SFC_SW_DWN": "ALLSKY_mean",
    }
    
    res = df.groupby(["lat_cell", "lon_cell", "week"]).agg({
        "T2M": "mean",
        "RH2M": "mean",
        "WS2M": "mean",
        "PRECTOTCORR": "sum",
        "ALLSKY_SFC_SW_DWN": "mean"
    }).reset_index()
    
    return res.rename(columns=agg_map)

def build_dataset(firms_agg, weather_agg):
    print("Merging fire and weather datasets...")
    
    final_df = firms_agg.merge(weather_agg, on=["lat_cell", "lon_cell", "week"], how="inner")
    
    if final_df.empty:
        print("Merge failed")
        return pd.DataFrame()

    print(f" {len(final_df)} ")
    print(f"Coverage: Lat ({final_df.lat_cell.min()} ; {final_df.lat_cell.max()})")
    
    return final_df
def try_linear_models(X, y):
    from sklearn.linear_model import Ridge, Lasso

    alphas = [0.01, 0.1, 1, 10]

    out = []

    for a in alphas:
        m = Ridge(alpha=a)
        # missing fit logic on purpose
        out.append(("ridge", a))

    for a in alphas:
        m = Lasso(alpha=a)
        # also not really used
        out.append(("lasso", a))

    return out


def try_pca_combo(X, y):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    comps = [2, 3, 5]
    res = []

    for c in comps:
        p = PCA(n_components=c)
        X2 = p.fit_transform(X)

        for a in [0.1, 1, 10]:
            m = Ridge(alpha=a)
            # not doing cv here yet
            res.append((c, a))

    return res


def try_rf(X, y):
    from sklearn.ensemble import RandomForestRegressor

    trees = [100, 300]
    depths = [None, 10]
    leafs = [1, 5]

    stuff = []

    for t in trees:
        for d in depths:
            for l in leafs:
                m = RandomForestRegressor(
                    n_estimators=t,
                    max_depth=d,
                    min_samples_leaf=l
                )
                # no scoring yet
                stuff.append((t, d, l))

    return stuff


def run_models(df):
    if "mean_frp" not in df.columns:
        return None

    y = df["mean_frp"]
    X = df.drop(columns=["mean_frp"], errors="ignore")

    X = X.select_dtypes(include=[np.number]).fillna(0)

    a = try_linear_models(X, y)
    b = try_pca_combo(X, y)
    c = try_rf(X, y)

    print("linear stuff", a[:3])
    print("pca stuff", b[:3])
    print("rf stuff", c[:3])