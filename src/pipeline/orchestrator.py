"""Ties baselines + single models + hybrid-VMD (causal and retrospective)
together for one PM2.5 series (the merged Chennai composite, or any single
station CSV).
"""
import os

from src.config import Config
from src.data.loader import load_pm25_frame
from src.evaluation.metrics import mase_scale
from src.features.windowing import train_test_split_chrono
from .baseline import run_baselines
from .single_model import run_single_models
from .hybrid_vmd import run_hybrid_retrospective, run_hybrid_causal


def run_pipeline(csv_path: str, cfg: Config, cache_dir: str = "outputs/cache"):
    """Returns (results, artifacts).

    results:   {model: {MAE, MAE_std, RMSE, ..., MASE, top_decile}, "per_IMF": {...}, "_meta": {...}}
    artifacts: {model: {timestamps, actual, predicted, predicted_std, predicted_seeds}, "imfs*": {...}}
    """
    name = os.path.basename(csv_path).replace(".csv", "")
    frame = load_pm25_frame(csv_path)
    series = frame["pm25"]
    exog = frame[["n_stations"]].values if (cfg.use_n_stations and "n_stations" in frame) else None

    train_series, test_series = train_test_split_chrono(series, cfg.test_frac)
    if len(test_series) % 2:            # vmdpy needs even length; drop the first test hour
        test_series = test_series.iloc[1:]
    n_tr, n_te = len(train_series), len(test_series)
    exog_train = exog[:n_tr] if exog is not None else None
    exog_test = exog[len(series) - n_te:] if exog is not None else None

    n_lags = cfg.n_lags
    scale = mase_scale(train_series.values)       # MASE denominator: train naive-1 MAE
    y_test = test_series.values[n_lags:]

    results, artifacts, per_imf, diagnostics = {}, {}, {}, {}

    m, a = run_baselines(train_series, test_series, n_lags, scale, cfg)
    results.update(m); artifacts.update(a)

    m, a, meta = run_single_models(train_series, test_series, n_lags, cfg, scale, exog_train, exog_test)
    results.update(m); artifacts.update(a)

    if cfg.run_retrospective:
        m, a, pi, d = run_hybrid_retrospective(train_series, test_series, n_lags, cfg, scale,
                                                exog_train, exog_test)
        results.update(m); artifacts.update(a); per_imf.update(pi); diagnostics["retrospective"] = d
    if cfg.run_causal:
        m, a, pi, d = run_hybrid_causal(train_series, test_series, n_lags, cfg, scale,
                                         exog_train, exog_test, cache_dir, name)
        results.update(m); artifacts.update(a); per_imf.update(pi); diagnostics["causal"] = d

    results["per_IMF"] = per_imf
    results["_meta"] = {
        "series": name, "n_train": n_tr, "n_test": n_te, "n_eval_hours": int(len(y_test)),
        "n_zero_hours_in_test": int((y_test == 0).sum()),
        "mase_scale_train_naive_mae": scale,
        "n_seeds": cfg.n_seeds, "val_frac": cfg.val_frac,
        "uses_n_stations_covariate": exog is not None,
        "causal_window_h": cfg.causal_window,
        "mode_diagnostics": diagnostics, **meta,
        "note": "Scores are mean over seeds (<metric>_std = SD over seeds); "
                "VMD_*_retrospective decomposes whole segments and leaks future information; "
                "VMD_*_causal uses rolling VMD at every origin.",
    }
    return results, artifacts
