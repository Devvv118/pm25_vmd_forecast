"""Gradient-boosting forecaster on lag features (non-neural benchmark)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .base import Forecaster


class GBMForecaster(Forecaster):
    name = "GBM"
    stochastic = False   # deterministic given the data

    def __init__(self, max_iter=400, learning_rate=0.05, max_leaf_nodes=31, seed=42, **_ignored):
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
            early_stopping=False, random_state=seed)
        self.best_iter_ = max_iter

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        if X_val is not None:   # pick the boosting-iteration count on the validation block
            errs = [np.mean((p - y_val) ** 2) for p in self.model.staged_predict(X_val)]
            self.best_iter_ = int(np.argmin(errs)) + 1
        return self

    def predict(self, X):
        for i, p in enumerate(self.model.staged_predict(X), start=1):
            if i == self.best_iter_:
                return p
        return self.model.predict(X)
