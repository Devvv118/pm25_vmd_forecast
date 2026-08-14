"""Baseline models trained directly on the raw (non-decomposed) PM2.5 series."""
import pandas as pd

from src.evaluation.metrics import regression_metrics
from src.features.windowing import make_supervised


def run_single_models(train_series: pd.Series, test_series: pd.Series,
                       n_lags: int, model_factories: list):
    """Returns (metrics, artifacts) — see baseline.py for the artifacts shape."""
    X_train, y_train = make_supervised(train_series, n_lags)
    X_test, y_test = make_supervised(test_series, n_lags)
    timestamps = test_series.index[n_lags:].values

    metrics, artifacts = {}, {}
    for factory in model_factories:
        model = factory().fit(X_train, y_train)
        preds = model.predict(X_test)
        key = f"Single_{model.name}"
        metrics[key] = regression_metrics(y_test, preds)
        artifacts[key] = {"timestamps": timestamps, "actual": y_test, "predicted": preds}
    return metrics, artifacts
