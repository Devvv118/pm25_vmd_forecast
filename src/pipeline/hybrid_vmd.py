"""Hybrid VMD pipeline: decompose -> one model per IMF -> sum forecasts.

Decomposing train and test separately (rather than the whole series at
once) is what keeps this leakage-safe — see decomposition/vmd.py.
"""
import numpy as np
import pandas as pd

from src.decomposition.vmd import vmd_decompose
from src.evaluation.metrics import regression_metrics
from src.features.windowing import make_supervised


def run_hybrid_vmd(train_series: pd.Series, test_series: pd.Series, k: int,
                    n_lags: int, model_factories: list, vmd_params: dict = None):
    """Returns (metrics, artifacts).

    artifacts contains one entry per fused model ("VMD_BP", "VMD_LSTM", ...)
    plus an "imfs" entry holding the raw train/test decomposition, so it
    can be plotted later without re-running VMD.
    """
    vmd_params = vmd_params or {}
    train_imfs = vmd_decompose(train_series.values, K=k, **vmd_params)
    test_imfs = vmd_decompose(test_series.values, K=k, **vmd_params)

    # vmdpy returns one sample fewer than the input when the input length
    # is odd (an internal mirroring/FFT quirk). Align the series to match
    # whatever length VMD actually produced, so indices stay consistent.
    train_series = train_series.iloc[:train_imfs.shape[1]]
    test_series = test_series.iloc[:test_imfs.shape[1]]

    n_points = len(test_series) - n_lags
    model_names = [factory().name for factory in model_factories]
    fused = {name: np.zeros(n_points) for name in model_names}
    per_imf = {}

    for i in range(k):
        imf_train = pd.Series(train_imfs[i], index=train_series.index)
        imf_test = pd.Series(test_imfs[i], index=test_series.index)

        X_train, y_train = make_supervised(imf_train, n_lags)
        X_test, y_test = make_supervised(imf_test, n_lags)

        for factory in model_factories:
            model = factory().fit(X_train, y_train)
            preds = model.predict(X_test)
            fused[model.name] += preds
            per_imf[f"IMF{i + 1}_{model.name}"] = regression_metrics(y_test, preds)

    y_test_actual = test_series.values[n_lags:]
    timestamps = test_series.index[n_lags:].values

    metrics = {f"VMD_{name}": regression_metrics(y_test_actual, preds_sum)
               for name, preds_sum in fused.items()}
    metrics["per_IMF"] = per_imf

    artifacts = {
        f"VMD_{name}": {"timestamps": timestamps, "actual": y_test_actual, "predicted": preds_sum}
        for name, preds_sum in fused.items()
    }
    artifacts["imfs"] = {
        "train_timestamps": train_series.index.values,
        "test_timestamps": test_series.index.values,
        "train_imfs": train_imfs,
        "test_imfs": test_imfs,
    }
    return metrics, artifacts
