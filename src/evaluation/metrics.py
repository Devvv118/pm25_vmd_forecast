"""Regression metrics, naive baselines and error diagnostics."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, mase_scale: float = None) -> dict:
    """MAE, RMSE, MAPE, R2 (+ MASE if a scale is given) and residual lag-1 autocorrelation.

    MAPE is computed over non-zero actuals only (PM2.5 series can contain
    zeros); `n_zero_actual` reports how many hours were excluded. MASE has
    no such problem and is the preferred relative-error measure.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) \
        if nonzero.any() else float("nan")

    out = {"MAE": float(mae), "RMSE": float(rmse), "MAPE": mape,
           "R2": float(r2_score(y_true, y_pred)), "n_zero_actual": int((~nonzero).sum())}
    if mase_scale:
        out["MASE"] = float(mae / mase_scale)
    res = y_true - y_pred
    out["resid_acf1"] = float(np.corrcoef(res[:-1], res[1:])[0, 1]) if len(res) > 2 and res.std() > 0 else float("nan")
    return out


def mase_scale(train_values, lag: int = 1) -> float:
    """In-sample MAE of the lag-`lag` naive forecast on the training series."""
    v = np.asarray(train_values, dtype=float)
    return float(np.mean(np.abs(v[lag:] - v[:-lag])))


def top_decile_metrics(y_true, y_pred, q: float = 0.9) -> dict:
    """Errors on the highest-PM2.5 hours only. R2 is deliberately NOT reported:
    within a truncated subset the variance is tiny and R2 is not meaningful."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = y_true >= np.quantile(y_true, q)
    return {"MAE": float(np.mean(np.abs(y_true[m] - y_pred[m]))),
            "RMSE": float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2))),
            "n": int(m.sum())}


def persistence_forecast(series, n_skip: int = 1):
    """Naive baseline: predict PM2.5(t+1) = PM2.5(t), evaluated on series[n_skip:].

    Use n_skip = n_lags so persistence is scored on exactly the same test
    hours as the lag-window models.
    """
    y_true = series.values[n_skip:]
    y_pred = series.values[n_skip - 1:-1]
    return y_true, y_pred
