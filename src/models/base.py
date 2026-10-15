"""Common interface every forecaster (BP, LSTM, GBM, ...) implements.

`fit` optionally receives a chronological validation block, used to pick the
number of training epochs/iterations (early stopping) instead of a fixed count.
`stochastic` tells the pipeline whether repeating with different seeds is
meaningful (True for neural nets; False for deterministic models).
"""
from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    name: str
    stochastic: bool = True

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> "Forecaster":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...
