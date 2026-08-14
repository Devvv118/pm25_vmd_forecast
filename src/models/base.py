"""Common interface every forecaster (BP, LSTM, ...) implements.

Keeping this interface tiny (fit/predict + a `name`) means the pipeline
code never needs to know which model it's running — it just loops over
a list of forecasters. Adding a new model type later (e.g. GRU) only
means writing one new class here, nothing in pipeline/ changes.
"""
from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    name: str

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "Forecaster":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...
