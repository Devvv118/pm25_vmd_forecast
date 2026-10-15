"""Aggregate repeated (multi-seed) runs into mean +/- SD metrics and prediction bands."""
import numpy as np

from .metrics import regression_metrics, top_decile_metrics

_METRICS = ("MAE", "RMSE", "MAPE", "R2", "MASE", "resid_acf1")


def aggregate_runs(y_true, preds_by_seed: np.ndarray, mase_scale: float = None):
    """preds_by_seed: (n_seeds, n_points).

    Returns (metrics, artifact). metrics holds, for every score, the mean
    over seeds under its own name and the SD under '<name>_std'. The
    reported score is the mean over seeds of per-seed scores (not the score
    of the averaged prediction, which would flatter an ensemble).
    """
    preds_by_seed = np.atleast_2d(preds_by_seed)
    per_seed = [regression_metrics(y_true, p, mase_scale) for p in preds_by_seed]
    out = {"n_seeds": int(len(preds_by_seed)), "n_zero_actual": per_seed[0]["n_zero_actual"]}
    for m in _METRICS:
        if m in per_seed[0]:
            vals = np.array([r[m] for r in per_seed], dtype=float)
            out[m] = float(np.nanmean(vals))
            out[f"{m}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    td = [top_decile_metrics(y_true, p) for p in preds_by_seed]
    out["top_decile"] = {"MAE": float(np.mean([t["MAE"] for t in td])),
                         "RMSE": float(np.mean([t["RMSE"] for t in td])), "n": td[0]["n"]}
    artifact = {"actual": np.asarray(y_true),
                "predicted": preds_by_seed.mean(0),
                "predicted_std": preds_by_seed.std(0, ddof=1) if len(preds_by_seed) > 1 else np.zeros(preds_by_seed.shape[1]),
                "predicted_seeds": preds_by_seed}
    return out, artifact
