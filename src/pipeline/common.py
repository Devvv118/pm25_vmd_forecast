"""Shared helpers: seeds, repeated fitting, validation-based tuning."""
import itertools

import numpy as np

from src.features.windowing import split_validation


def seeds_for(cfg, stochastic=True):
    n = cfg.n_seeds if stochastic else 1
    return [cfg.base_seed + i for i in range(n)]


def fit_predict_seeds(factory, cfg, X_train, y_train, X_test):
    """Train `factory(seed)` once per seed (early stopping on a chronological
    validation block carved from the end of the training samples) and return
    an (n_seeds, n_test) array of test predictions, plus the models' chosen epochs."""
    X_tr, y_tr, X_val, y_val = split_validation(X_train, y_train, cfg.val_frac)
    probe = factory(cfg.base_seed)
    seeds = seeds_for(cfg, stochastic=probe.stochastic)
    preds, epochs = [], []
    for seed in seeds:
        model = factory(seed).fit(X_tr, y_tr, X_val, y_val)
        preds.append(model.predict(X_test))
        epochs.append(getattr(model, "best_epoch_", None) or getattr(model, "best_iter_", None))
    return np.stack(preds), epochs


def tune_lstm(cfg, build_factory, X_train, y_train):
    """Grid search on the validation block (never on test). Returns the best param dict."""
    grid = cfg.baselines["tuning_grid"]
    X_tr, y_tr, X_val, y_val = split_validation(X_train, y_train, cfg.val_frac)
    keys = list(grid)
    best, best_params, log = float("inf"), None, []
    for values in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        model = build_factory(**params)(cfg.base_seed).fit(X_tr, y_tr, X_val, y_val)
        log.append({**params, "val_mse": float(model.val_mse_)})
        if model.val_mse_ < best:
            best, best_params = model.val_mse_, params
    return best_params, log
