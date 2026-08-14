"""Variational Mode Decomposition (VMD) wrapper.

Leakage-safe usage: call this separately on the train segment and the test
segment (never on the full series at once). Each call only ever sees data
within its own segment, so no future information leaks into training.
See README.md for the full rationale.
"""
import numpy as np
from vmdpy import VMD


def vmd_decompose(signal: np.ndarray, K: int = 6, alpha: float = 2000.0,
                   tau: float = 0.0, DC: bool = False, init: int = 1,
                   tol: float = 1e-7) -> np.ndarray:
    """Decompose a 1D signal into K intrinsic mode functions (IMFs).

    Returns:
        imfs: array of shape (K, len(signal)).
    """
    signal = np.asarray(signal, dtype=float)
    imfs, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
    return imfs
