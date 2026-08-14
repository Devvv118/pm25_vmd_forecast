"""Backpropagation neural network (MLP) forecaster."""
from sklearn.neural_network import MLPRegressor

from .base import Forecaster


class BPForecaster(Forecaster):
    name = "BP"

    def __init__(self, hidden_layer_sizes=(64, 32), max_iter=500, random_state=42):
        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            activation="relu",
            solver="adam",
            max_iter=max_iter,
            random_state=random_state,
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)
