import numpy as np
import pandas as pd
from dotenv import load_dotenv

from firms import fetch_firms_data
from powers import fetch_power_data
from landsat import fetch_landsat_features
import pandas as pd
import numpy as np

load_dotenv()

BBOX = (-124.0, 36.0, -119.0, 42.0)
START = "2024-05-01"
END   = "2024-10-31"

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

WEATHER_COLS  = ["T2M_mean", "RH2M_mean", "WS2M_mean", "PRECTOTCORR_sum", "ALLSKY_mean"]
SAT_COLS      = ["NDVI_mean", "NBR_mean", "B4_mean", "B5_mean", "B7_mean"]
TARGET        = "mean_frp"


RES = 0.5

def snap_to_grid(df, lat_col, lon_col):
    df["lat_cell"] = df[lat_col].round() / RES
    df["lon_cell"] = df[lon_col].round() / RES
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
    
    cols = ["T2M", "RH2M", "WS2M"]
    return df.groupby(["lat_cell", "lon_cell", "week"])[cols].mean().reset_index()

def build_dataset(firms_agg, weather_agg):
    df = firms_agg.merge(weather_agg, on=["lat_cell", "lon_cell", "week"], how="inner")
    return df