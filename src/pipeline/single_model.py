"""Models trained directly on the raw (non-decomposed) PM2.5 series:
BP, LSTM (default config), LSTM (validation-tuned) and gradient boosting."""
import numpy as np
import pandas as pd

from src.evaluation.seeds import aggregate_runs
from src.features.windowing import make_supervised
from src.models.registry import build_model_factories
from .common import fit_predict_seeds, tune_lstm


def run_single_models(train_series: pd.Series, test_series: pd.Series, n_lags: int,
                      cfg, scale: float, exog_train=None, exog_test=None):
    X_train, y_train = make_supervised(train_series, n_lags, exog=exog_train)
    X_test, y_test = make_supervised(test_series, n_lags, exog=exog_test)
    timestamps = test_series.index[n_lags:].values

    factories = build_model_factories(cfg, n_lags)
    runs = {f"Single_{n}": f for n, f in factories.items() if n != "GBM"}
    if cfg.baselines.get("gbm"):
        runs["Single_GBM"] = factories["GBM"]

    meta = {}
    if cfg.baselines.get("lstm_tuned"):
        best, log = tune_lstm(cfg, lambda **p: (lambda seed: build_model_factories(cfg, n_lags, **p)["LSTM"](seed)),
                              X_train, y_train)
        runs["Single_LSTM_tuned"] = build_model_factories(cfg, n_lags, **best)["LSTM"]
        meta["lstm_tuning"] = {"best": best, "grid_log": log}

    metrics, artifacts = {}, {}
    for key, factory in runs.items():
        preds, epochs = fit_predict_seeds(factory, cfg, X_train, y_train, X_test)
        m, a = aggregate_runs(y_test, preds, scale)
        m["selected_epochs"] = epochs
        metrics[key] = m
        artifacts[key] = {"timestamps": timestamps, **a}
    return metrics, artifacts, meta
