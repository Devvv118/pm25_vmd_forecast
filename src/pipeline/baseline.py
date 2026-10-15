"""Non-learning / statistical benchmarks: persistence, seasonal naive, SARIMA."""
import numpy as np
import pandas as pd

from src.evaluation.metrics import persistence_forecast
from src.evaluation.seeds import aggregate_runs
from src.models.statistical import seasonal_naive, sarima_one_step


def _pack(name, y_true, y_pred, timestamps, scale, metrics, artifacts):
    m, a = aggregate_runs(y_true, np.asarray(y_pred)[None, :], scale)
    metrics[name] = m
    artifacts[name] = {"timestamps": timestamps, **a}


def run_baselines(train_series: pd.Series, test_series: pd.Series, n_lags: int,
                  scale: float, cfg):
    """All baselines are scored on test_series[n_lags:], the same hours as the
    lag-window models (persistence used to be scored on one extra hour)."""
    metrics, artifacts = {}, {}
    ts = test_series.index[n_lags:].values
    y_true, y_pred = persistence_forecast(test_series, n_skip=n_lags)
    _pack("Persistence", y_true, y_pred, ts, scale, metrics, artifacts)

    b = cfg.baselines
    full = pd.concat([train_series, test_series])
    if b.get("seasonal_naive"):
        p = seasonal_naive(full, test_series.index, 24)[n_lags:]
        _pack("SeasonalNaive_24", y_true, p, ts, scale, metrics, artifacts)
    if b.get("sarima"):
        p = sarima_one_step(train_series, test_series, tuple(b["sarima_order"]),
                            tuple(b["sarima_seasonal_order"]), b["sarima_max_fit_hours"])[n_lags:]
        _pack("SARIMA", y_true, p, ts, scale, metrics, artifacts)
    return metrics, artifacts
