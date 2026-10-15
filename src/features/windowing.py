"""Chronological splits and lag-window feature construction."""
import numpy as np
import pandas as pd


def train_test_split_chrono(series: pd.Series, test_frac: float = 0.2):
    """Split a time series chronologically (no shuffling) into train/test."""
    n = len(series)
    split = int(n * (1 - test_frac))
    return series.iloc[:split], series.iloc[split:]


def make_supervised(series: pd.Series, n_lags: int = 24, horizon: int = 1,
                    exog: np.ndarray = None):
    """Turn a 1D series into lag-window (X) / next-value (y) arrays.

    X[i] = [t-n_lags, ..., t-1] (+ optional exogenous columns observed at t-1),
    y[i] = value at t+horizon-1.

    ``exog`` is an (n, e) array aligned with ``series``; its row i-1 is used
    for the sample whose target is i, i.e. only information known at the
    forecast origin.
    """
    values = series.values.astype(float)
    X, y = [], []
    for i in range(n_lags, len(values) - horizon + 1):
        row = values[i - n_lags:i]
        if exog is not None:
            row = np.concatenate([row, np.atleast_1d(exog[i - 1])])
        X.append(row)
        y.append(values[i + horizon - 1])
    return np.array(X), np.array(y)


def split_validation(X: np.ndarray, y: np.ndarray, val_frac: float):
    """Chronological train / validation split of supervised samples (last block = validation)."""
    if not val_frac:
        return X, y, None, None
    n_val = max(int(len(X) * val_frac), 1)
    return X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]
