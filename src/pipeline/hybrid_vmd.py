"""Hybrid VMD pipeline: decompose -> one model per mode -> sum the forecasts.

Two regimes are supported (see src/decomposition/vmd.py):
  * retrospective - train and test segments are each decomposed once, as a whole;
  * causal        - rolling VMD, every mode value computed from data up to its own timestamp.
The forecasting step is identical in both; only the mode series differ.
"""
import numpy as np
import pandas as pd

from src.decomposition.vmd import vmd_decompose, dominant_period_hours
from src.decomposition.rolling import rolling_causal_modes
from src.evaluation.seeds import aggregate_runs
from src.features.windowing import make_supervised
from src.models.registry import build_model_factories
from .common import fit_predict_seeds


def _forecast_modes(tag, train_modes, train_idx, test_modes, test_idx, train_exog, test_exog,
                    y_actual, n_lags, cfg, scale):
    """Train one model per mode (and per hybrid model type / seed), sum, and score."""
    factories = build_model_factories(cfg, n_lags)
    names = cfg.hybrid_models
    K = train_modes.shape[0]
    fused = {}          # name -> (n_seeds, n_points)
    per_imf = {}

    for i in range(K):
        imf_train = pd.Series(train_modes[i], index=train_idx)
        imf_test = pd.Series(test_modes[i], index=test_idx)
        X_tr, y_tr = make_supervised(imf_train, n_lags, exog=train_exog)
        X_te, y_te = make_supervised(imf_test, n_lags, exog=test_exog)
        for name in names:
            preds, _ = fit_predict_seeds(factories[name], cfg, X_tr, y_tr, X_te)
            fused[name] = fused.get(name, 0) + preds
            m, _ = aggregate_runs(y_te, preds)
            per_imf[f"IMF{i + 1}_{name}_{tag}"] = m
            print(f"  [{tag}] IMF{i + 1} {name}: MAE={m['MAE']:.3f}")

    metrics, artifacts = {}, {}
    ts = test_idx[n_lags:].values
    for name, preds in fused.items():
        m, a = aggregate_runs(y_actual, preds, scale)
        metrics[f"VMD_{name}_{tag}"] = m
        artifacts[f"VMD_{name}_{tag}"] = {"timestamps": ts, **a}
    return metrics, artifacts, per_imf


def _mode_diagnostics(train_modes, test_modes, train_actual, test_actual):
    """Why train/test modes may differ (reviewer comment): dominant period per mode in
    each segment and the reconstruction error (sum of modes vs the series)."""
    rows = [{"mode": i + 1,
             "train_period_h": dominant_period_hours(train_modes[i]),
             "test_period_h": dominant_period_hours(test_modes[i])} for i in range(train_modes.shape[0])]
    rec = {"train_rmse": float(np.sqrt(np.mean((train_modes.sum(0) - train_actual) ** 2))),
           "test_rmse": float(np.sqrt(np.mean((test_modes.sum(0) - test_actual) ** 2)))}
    return {"mode_periods": rows, "reconstruction": rec}


def run_hybrid_retrospective(train_series, test_series, n_lags, cfg, scale,
                             exog_train=None, exog_test=None):
    k, p = cfg.k, cfg.vmd
    train_imfs = vmd_decompose(train_series.values, K=k, **p)
    test_imfs = vmd_decompose(test_series.values, K=k, **p)
    tr_idx = train_series.index[len(train_series) - train_imfs.shape[1]:]
    te_idx = test_series.index[len(test_series) - test_imfs.shape[1]:]
    off_tr, off_te = len(train_series) - train_imfs.shape[1], len(test_series) - test_imfs.shape[1]
    ex_tr = exog_train[off_tr:] if exog_train is not None else None
    ex_te = exog_test[off_te:] if exog_test is not None else None
    # evaluate on the same hours as every other model: test_series[n_lags:]
    y_actual = test_series.values[n_lags:]
    assert te_idx[n_lags:].equals(test_series.index[n_lags:]), "retrospective test segment must have even length"

    metrics, artifacts, per_imf = _forecast_modes("retrospective", train_imfs, tr_idx, test_imfs, te_idx,
                                                  ex_tr, ex_te, y_actual, n_lags, cfg, scale)
    diag = _mode_diagnostics(train_imfs, test_imfs, train_series.values[off_tr:], test_series.values[off_te:])
    artifacts["imfs"] = {"train_timestamps": tr_idx.values, "test_timestamps": te_idx.values,
                         "train_imfs": train_imfs, "test_imfs": test_imfs}
    return metrics, artifacts, per_imf, diag


def run_hybrid_causal(train_series, test_series, n_lags, cfg, scale,
                      exog_train=None, exog_test=None, cache_dir=None, name="series"):
    full = pd.concat([train_series, test_series])
    W = cfg.causal_window
    cache = f"{cache_dir}/{name}_causal_modes.npy" if cache_dir else None
    modes = rolling_causal_modes(full.values, cfg.k, W, cfg.vmd, cfg.causal_n_jobs, cache)

    split = len(train_series)
    tr_idx, te_idx = full.index[W - 1:split], full.index[split:]
    train_modes, test_modes = modes[:, W - 1:split], modes[:, split:]
    ex_tr = exog_train[W - 1:] if exog_train is not None else None
    y_actual = test_series.values[n_lags:]

    metrics, artifacts, per_imf = _forecast_modes("causal", train_modes, tr_idx, test_modes, te_idx,
                                                  ex_tr, exog_test, y_actual, n_lags, cfg, scale)
    diag = _mode_diagnostics(train_modes, test_modes, full.values[W - 1:split], test_series.values)
    artifacts["imfs_causal"] = {"train_timestamps": tr_idx.values, "test_timestamps": te_idx.values,
                                "train_imfs": train_modes, "test_imfs": test_modes}
    return metrics, artifacts, per_imf, diag
