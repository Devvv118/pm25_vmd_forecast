"""Variational Mode Decomposition (VMD) wrappers.

Two ways of using VMD are provided, and the distinction matters:

* ``vmd_decompose`` - *retrospective*: decomposes a whole segment at once.
  Every mode value then depends on samples that lie later in time (VMD is
  a global optimisation over the entire window), so a lagged mode value
  already carries information about future hours. Fine for describing a
  signal, NOT valid as a forecasting input. Kept only so the retrospective
  result can be reported next to the causal one.

* ``rolling_causal_modes`` (src/decomposition/rolling.py) - *causal*: for
  every forecast origin t, VMD is run only on data up to and including t.
"""
import numpy as np
from vmdpy import VMD


def vmd_decompose(signal: np.ndarray, K: int = 6, alpha: float = 2000.0,
                   tau: float = 0.0, DC: bool = False, init: int = 1,
                   tol: float = 1e-7) -> np.ndarray:
    """Decompose a 1D signal into K IMFs, ordered by centre frequency (low -> high).

    vmdpy needs an even-length input and silently drops one sample otherwise.
    To keep the *most recent* sample (the one that matters for forecasting),
    an odd-length input loses its FIRST sample instead. The returned array
    therefore has shape (K, len(signal) - len(signal) % 2) and is aligned to
    ``signal[len(signal) % 2:]``.

    Sorting by centre frequency keeps mode i meaning the same frequency band
    in every call (vmdpy does not guarantee this on its own), which is what
    lets a model trained on mode i be applied to mode i of another window.
    """
    signal = np.asarray(signal, dtype=float)
    if len(signal) % 2:
        signal = signal[1:]
    imfs, _, omega = VMD(signal, alpha, tau, K, DC, init, tol)
    order = np.argsort(omega[-1])
    return imfs[order]


def dominant_period_hours(imf: np.ndarray) -> float:
    """Dominant period (hours) of one mode, from its FFT peak (hourly data)."""
    x = np.asarray(imf, dtype=float)
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0)
    if len(spec) < 2:
        return float("nan")
    k = int(np.argmax(spec[1:]) + 1)
    return float(1.0 / freqs[k])
