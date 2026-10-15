"""Backpropagation neural network (MLP) forecaster with validation-based epoch selection."""
import copy

import numpy as np
from sklearn.neural_network import MLPRegressor

from .base import Forecaster


class BPForecaster(Forecaster):
    name = "BP"

    def __init__(self, hidden_layer_sizes=(64, 32), max_epochs=200, patience=15,
                 lr=1e-3, seed=42, **_ignored):
        self.max_epochs = max_epochs
        self.patience = patience
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layer_sizes), activation="relu",
            solver="adam", learning_rate_init=lr, random_state=seed,
        )
        self._mean = self._std = None   # per-column statistics (lags and covariates alike)
        self._ymean, self._ystd = 0.0, 1.0
        self.best_epoch_ = None

    def _scale(self, X):
        return (X - self._mean) / self._std

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self._mean = X_train.mean(0)
        self._std = np.where(X_train.std(0) == 0, 1.0, X_train.std(0))
        self._ymean = float(y_train.mean())
        self._ystd = float(y_train.std()) or 1.0
        Xs, ys = self._scale(X_train), (y_train - self._ymean) / self._ystd

        best, best_model, bad = np.inf, None, 0
        for epoch in range(1, self.max_epochs + 1):
            self.model.partial_fit(Xs, ys)       # one pass over the data
            if X_val is None:
                continue
            val = float(np.mean((self.model.predict(self._scale(X_val))
                                 - (y_val - self._ymean) / self._ystd) ** 2))
            if val < best - 1e-9:
                best, best_model, bad, self.best_epoch_ = val, copy.deepcopy(self.model), 0, epoch
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_model is not None:
            self.model = best_model
        return self

    def predict(self, X):
        return self.model.predict(self._scale(X)) * self._ystd + self._ymean
