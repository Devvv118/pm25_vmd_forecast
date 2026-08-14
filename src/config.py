"""Central place for every pipeline hyperparameter.

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

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str = "configs/default.yaml") -> Config:
    if path and os.path.exists(path):
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        return Config(**overrides)
    return Config()
