"""Seasonal statistical benchmarks: seasonal-naive and SARIMA.

Both give genuine one-step-ahead forecasts for every test hour using only
data up to the previous hour. SARIMA parameters are estimated once on the
training segment and then held fixed while the filter runs over the test
segment (no look-ahead, no per-hour refitting).
"""
import warnings

import numpy as np
import pandas as pd


def seasonal_naive(full: pd.Series, test_index, period: int = 24) -> np.ndarray:
    """y_hat(t) = y(t - period)."""
    return full.shift(period).reindex(test_index).values


def sarima_one_step(train: pd.Series, test: pd.Series, order=(2, 0, 1),
                    seasonal_order=(1, 0, 1, 24), max_fit_hours: int = 2000) -> np.ndarray:
    """One-step-ahead SARIMA predictions for each point of `test`."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    fit_part = train.iloc[-max_fit_hours:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(fit_part.values, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        # run the fixed-parameter filter over (tail of train + test)
        context = train.iloc[-max_fit_hours:]
        full = np.concatenate([context.values, test.values])
        filt = res.apply(full)
        pred = filt.get_prediction(start=len(context)).predicted_mean
    return np.asarray(pred)
