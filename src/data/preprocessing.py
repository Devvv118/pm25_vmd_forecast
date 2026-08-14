"""Cleaning steps applied to a raw station dataframe.

Kept separate from loader.py so each step is independently testable:
enforcing a regular hourly grid, and interpolating short gaps.
"""
import pandas as pd

NUMERIC_COLS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]


def enforce_hourly_index(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a strict hourly frequency, inserting NaN rows for gaps."""
    return df.asfreq("h")


def interpolate_short_gaps(df: pd.DataFrame, cols=None, limit: int = 6) -> pd.DataFrame:
    """Time-interpolate missing values, but only across gaps <= `limit` hours.

    Longer gaps are left as NaN on purpose — they shouldn't be silently
    invented by interpolation.
    """
    cols = cols or [c for c in NUMERIC_COLS if c in df.columns]
    df = df.copy()
    df[cols] = df[cols].interpolate(method="time", limit=limit)
    return df
