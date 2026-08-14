"""Persistence baseline: PM2.5(t+1) = PM2.5(t)."""
import pandas as pd

from src.evaluation.metrics import persistence_forecast, regression_metrics


def run_persistence_baseline(test_series: pd.Series):
    """Returns (metrics, artifacts).

    artifacts["Persistence"] holds the timestamps/actual/predicted arrays
    needed to plot this baseline later.
    """
    y_true, y_pred = persistence_forecast(test_series)
    metrics = {"Persistence": regression_metrics(y_true, y_pred)}
    artifacts = {
        "Persistence": {
            "timestamps": test_series.index[1:].values,
            "actual": y_true,
            "predicted": y_pred,
        }
    }
    return metrics, artifacts
