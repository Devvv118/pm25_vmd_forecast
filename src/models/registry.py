"""Builds model factories from config.

A factory is a callable `factory(seed) -> fresh, untrained Forecaster`.
"""
from src.config import Config
from .bp import BPForecaster
from .gbm import GBMForecaster
from .lstm import LSTMForecaster


def build_model_factories(cfg: Config, n_lags: int = None, **lstm_overrides) -> dict:
    n_lags = n_lags or cfg.n_lags
    lstm_params = {**cfg.lstm, **lstm_overrides}
    return {
        "BP": lambda seed: BPForecaster(**cfg.bp, seed=seed),
        "LSTM": lambda seed: LSTMForecaster(n_seq=n_lags, **lstm_params, seed=seed),
        "GBM": lambda seed: GBMForecaster(**cfg.gbm, seed=seed),
    }
