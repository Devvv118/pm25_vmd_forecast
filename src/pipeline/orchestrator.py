"""Ties baseline + single-model + hybrid-VMD stages together for one station.

This is the single function both main.py (one station) and
scripts/run_all.py (every station) call, so the two entry points can't
drift out of sync with each other.
"""
from src.config import Config
from src.data.loader import get_pm25_series
from src.features.windowing import train_test_split_chrono
from src.models.registry import build_model_factories
from .baseline import run_persistence_baseline
from .single_model import run_single_models
from .hybrid_vmd import run_hybrid_vmd


def run_pipeline_for_station(csv_path: str, cfg: Config):
    """Returns (results, artifacts).

    results:   {model_name: {MAE, RMSE, MAPE, R2}, ..., "per_IMF": {...}}
    artifacts: {model_name: {timestamps, actual, predicted}, ..., "imfs": {...}}
    Both dicts share the same model-name keys so they're easy to pair up
    later (e.g. in the plotting script).
    """
    series = get_pm25_series(csv_path)
    train_series, test_series = train_test_split_chrono(series, cfg.test_frac)
    model_factories = build_model_factories(cfg)

    results, artifacts = {}, {}

    m, a = run_persistence_baseline(test_series)
    results.update(m)
    artifacts.update(a)

    m, a = run_single_models(train_series, test_series, cfg.n_lags, model_factories)
    results.update(m)
    artifacts.update(a)

    m, a = run_hybrid_vmd(train_series, test_series, cfg.k, cfg.n_lags,
                           model_factories, cfg.vmd)
    results.update(m)
    artifacts.update(a)

    return results, artifacts
