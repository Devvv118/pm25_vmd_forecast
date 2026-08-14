"""Regression metrics and the naive persistence baseline."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) \
        if nonzero.any() else float("nan")

    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": mape, "R2": float(r2)}


def persistence_forecast(series):
    """Naive baseline: predict PM2.5(t+1) = PM2.5(t)."""
    y_true = series.values[1:]
    y_pred = series.values[:-1]
    return y_true, y_pred
