"""Builds the list of model factories the pipeline should run, from config.

A "factory" is just a zero-arg callable that returns a fresh, untrained
Forecaster — the pipeline calls it once per IMF so every model starts
from a clean state.
"""
from src.config import Config
from .bp import BPForecaster
from .lstm import LSTMForecaster


def build_model_factories(cfg: Config) -> list:
    return [
        lambda: BPForecaster(**cfg.bp),
        lambda: LSTMForecaster(**cfg.lstm),
    ]
