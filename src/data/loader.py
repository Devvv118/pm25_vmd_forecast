"""Turn a raw PRSA-format station CSV into a clean hourly PM2.5 series."""
import glob
import os

import pandas as pd

from .preprocessing import NUMERIC_COLS, enforce_hourly_index, interpolate_short_gaps


def load_station(csv_path: str) -> pd.DataFrame:
    """Read one station CSV and return a clean, hourly-indexed DataFrame."""
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    df = enforce_hourly_index(df)
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    df = interpolate_short_gaps(df, cols)
    df = df.dropna(subset=["PM2.5"])
    return df


def get_pm25_series(csv_path: str) -> pd.Series:
    """Convenience wrapper: just the PM2.5 series for a station."""
    return load_station(csv_path)["PM2.5"]


def discover_station_csvs(raw_dir: str) -> list:
    """List every station CSV in a directory, sorted for reproducibility."""
    return sorted(glob.glob(os.path.join(raw_dir, "*.csv")))


def station_name_from_path(csv_path: str) -> str:
    return os.path.basename(csv_path).replace(".csv", "")
