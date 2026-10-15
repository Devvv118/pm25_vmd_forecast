"""Central place for every pipeline hyperparameter and data-fetch setting.

Load with `load_config("configs/default.yaml")`; any key left out of the
YAML file just falls back to the dataclass default below.
"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


@dataclass
class Config:
    n_lags: int = 24          # hours of history fed to each model
    k: int = 6                # number of VMD modes (IMFs)
    test_frac: float = 0.2    # fraction of the series held out, chronologically
    val_frac: float = 0.15    # last block of the TRAIN samples used to pick epochs / tune (chronological)
    n_seeds: int = 10         # independent training runs per stochastic model (report mean +/- SD)
    base_seed: int = 42
    use_n_stations: bool = True   # feed the per-hour station count as a covariate (if in the CSV)

    # Decomposition regimes to run. "causal" = rolling VMD at every forecast
    # origin (valid); "retrospective" = whole train / whole test segment
    # decomposed at once (leaks future information; kept for comparison).
    run_causal: bool = True
    run_retrospective: bool = True
    causal_window: int = 1440     # hours of history per rolling VMD call (must be even)
    causal_n_jobs: int = -1       # parallel workers for rolling VMD
    hybrid_models: list = field(default_factory=lambda: ["BP", "LSTM"])

    vmd: Dict[str, Any] = field(default_factory=lambda: {
        "alpha": 2000.0, "tau": 0.0, "DC": False, "init": 1, "tol": 1e-7,
    })
    bp: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_layer_sizes": [64, 32], "max_epochs": 200, "patience": 15, "lr": 1e-3,
    })
    lstm: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_size": 32, "num_layers": 1, "epochs": 100, "patience": 10,
        "lr": 1e-3, "batch_size": 64,   # epochs = maximum; early stopping on the validation block
    })
    gbm: Dict[str, Any] = field(default_factory=lambda: {
        "max_iter": 400, "learning_rate": 0.05, "max_leaf_nodes": 31,
    })

    # Extra baselines (comment: baselines were thin).
    baselines: Dict[str, Any] = field(default_factory=lambda: {
        "seasonal_naive": True,
        "sarima": True,
        "sarima_order": [2, 0, 1],
        "sarima_seasonal_order": [1, 0, 1, 24],
        "sarima_max_fit_hours": 2000,
        "gbm": True,
        "lstm_tuned": True,   # LSTM hyper-parameters chosen on the validation block
        "tuning_grid": {"hidden_size": [32, 64, 128], "num_layers": [1, 2], "lr": [0.001, 0.003]},
    })

    # Settings for scripts/fetch_data.py — where to pull data from and how
    # to combine multiple stations into one composite series.
    openaq: Dict[str, Any] = field(default_factory=lambda: {
        "city": "chennai",
        "lat": 13.0827,
        "lon": 80.2707,
        "radius_m": 25000,          # OpenAQ's max allowed radius per request
        "datetime_from": "2025-10-01",   # inclusive, local date (12 months: covers NE monsoon / winter)
        "datetime_to": "2026-09-30",     # inclusive, local date
        "timezone": "Asia/Kolkata",
        "merge_method": "mean",          # "mean" or "median" across stations
        "min_station_coverage": 0.5,     # drop stations with < 50% of window hours
        "min_coverage_pct": 0.0,         # drop hours whose API-reported coverage is lower (0 = off)
        "max_value": 1000.0,             # ug/m3; values above are treated as sensor faults
        "min_stations": 1,               # min stations reporting for a merged hour to be kept
    })

def load_config(path: str = "configs/default.yaml") -> Config:
    if path and os.path.exists(path):
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        cfg = Config()
        # Deep-merge dict sections so a partial YAML block keeps the other defaults.
        for key, val in overrides.items():
            current = getattr(cfg, key)
            setattr(cfg, key, {**current, **val} if isinstance(current, dict) else val)
        return cfg
    return Config()
