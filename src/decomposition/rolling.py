"""Causal (rolling-origin) VMD.

For each time t the signal y[t-W+1 .. t] (only data available at t) is
decomposed with VMD and the *last* value of every mode is kept:

    c_k(t) = k-th mode of VMD(y[t-W+1..t]) evaluated at t

c_k(t) is a fully causal mode series: nothing after t enters it. Per-mode
forecasters are trained and tested on c_k, so train and test modes are
produced by the identical procedure (same window, same end-point
behaviour) instead of two unrelated retrospective decompositions.
"""
import hashlib
import os

import numpy as np
from joblib import Parallel, delayed

from .vmd import vmd_decompose


def _last_modes(window_values: np.ndarray, K: int, vmd_params: dict) -> np.ndarray:
    return vmd_decompose(window_values, K=K, **vmd_params)[:, -1]


def rolling_causal_modes(values: np.ndarray, K: int, window: int, vmd_params: dict,
                          n_jobs: int = -1, cache_path: str = None) -> np.ndarray:
    """Returns array (K, n); columns t < window-1 are NaN (not enough history).

    ``window`` must be even (vmdpy requirement; an odd window would drop the
    newest sample).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if window % 2:
        raise ValueError("causal VMD window must be even")
    if n < window:
        raise ValueError(f"series ({n}) shorter than causal VMD window ({window})")

    key = hashlib.md5(values.tobytes() + repr((K, window, sorted(vmd_params.items()))).encode()).hexdigest()[:12]
    if cache_path:
        cache_file = cache_path.replace(".npy", f"_{key}.npy")
        if os.path.exists(cache_file):
            print(f"[causal VMD] loaded cache {cache_file}")
            return np.load(cache_file)

    print(f"[causal VMD] {n - window + 1} rolling decompositions "
          f"(window={window}h, K={K}) ...")
    cols = Parallel(n_jobs=n_jobs, batch_size=32)(
        delayed(_last_modes)(values[t - window + 1: t + 1], K, vmd_params)
        for t in range(window - 1, n)
    )
    modes = np.full((K, n), np.nan)
    modes[:, window - 1:] = np.stack(cols, axis=1)

    if cache_path:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.save(cache_file, modes)
    return modes
