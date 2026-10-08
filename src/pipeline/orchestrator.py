"""Ties baseline + single-model + hybrid-VMD stages together for one
PM2.5 series (the merged Chennai composite, or any single station CSV).
"""
from src.config import Config
from src.data.loader import load_pm25_series
from src.features.windowing import train_test_split_chrono
from src.models.registry import build_model_factories
from .baseline import run_persistence_baseline
from .single_model import run_single_models
from .hybrid_vmd import run_hybrid_vmd


def run_pipeline(csv_path: str, cfg: Config):
    """Returns (results, artifacts).

    results:   {model_name: {MAE, RMSE, MAPE, R2}, ..., "per_IMF": {...}}
    artifacts: {model_name: {timestamps, actual, predicted}, ..., "imfs": {...}}
    """
    series = load_pm25_series(csv_path)
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
