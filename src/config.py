"""Central place for every pipeline hyperparameter and data-fetch setting.

Load with `load_config("configs/default.yaml")`; any key left out of the
YAML file just falls back to the dataclass default below.
"""
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict

import yaml


@dataclass
class Config:
    n_lags: int = 24          # hours of history fed to each model
    k: int = 6                # number of VMD modes (IMFs)
    test_frac: float = 0.2    # fraction of the series held out, chronologically

    vmd: Dict[str, Any] = field(default_factory=lambda: {
        "alpha": 2000.0, "tau": 0.0, "DC": False, "init": 1, "tol": 1e-7,
    })
    bp: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_layer_sizes": [64, 32], "max_iter": 500, "random_state": 42,
    })
    lstm: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_size": 32, "epochs": 30, "lr": 1e-3, "batch_size": 64,
    })

    # Settings for scripts/fetch_data.py — where to pull data from and how
    # to combine multiple stations into one composite series.
    openaq: Dict[str, Any] = field(default_factory=lambda: {
        "city": "chennai",
        "lat": 13.0827,
        "lon": 80.2707,
        "radius_m": 25000,          # OpenAQ's max allowed radius per request
        "datetime_from": "2026-04-01",   # inclusive, local date
        "datetime_to": "2026-09-30",     # inclusive, local date
        "timezone": "Asia/Kolkata",
        "merge_method": "mean",          # "mean" or "median" across stations
        "min_station_coverage": 0.5,     # drop stations with < 50% of window hours
        "min_coverage_pct": 0.0,         # drop hours whose API-reported coverage is lower (0 = off)
        "max_value": 1000.0,             # ug/m3; values above are treated as sensor faults
        "min_stations": 1,               # min stations reporting for a merged hour to be kept
    })

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
