"""Chronological train/test split and lag-window feature construction."""
import numpy as np
import pandas as pd


def train_test_split_chrono(series: pd.Series, test_frac: float = 0.2):
    """Split a time series chronologically (no shuffling) into train/test."""
    n = len(series)
    split = int(n * (1 - test_frac))
    return series.iloc[:split], series.iloc[split:]


def make_supervised(series: pd.Series, n_lags: int = 24, horizon: int = 1):
    """Turn a 1D series into lag-window (X) / next-value (y) arrays.

    X[i] = [t-n_lags, ..., t-1], y[i] = value at t+horizon-1
    """
    values = series.values.astype(float)
    X, y = [], []
    for i in range(n_lags, len(values) - horizon + 1):
        X.append(values[i - n_lags:i])
        y.append(values[i + horizon - 1])
    return np.array(X), np.array(y)
