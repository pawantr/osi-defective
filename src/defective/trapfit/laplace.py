"""Approximate inverse Laplace transform of an isothermal transient (Laplace DLTS)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from scipy.signal import find_peaks


def laplace_transient(time_s, signal, n_rates=120, alpha=0.01, e_min=None, e_max=None):
    """Tikhonov-regularized NNLS spectrum of emission rates. Approximate; use CONTIN/FTIKREG for production."""
    t = np.asarray(time_s, float)
    y = np.asarray(signal, float)
    if len(t) < 30 or t.shape != y.shape or np.any(t <= 0):
        return {"status": "unresolved", "reason": "need >=30 points with positive times"}
    e_min = e_min or 0.3 / t.max()
    e_max = e_max or 3.0 / t.min()
    e = np.logspace(np.log10(e_min), np.log10(e_max), n_rates)
    A = np.exp(-np.outer(t, e))
    scale = np.max(np.abs(y)) or 1.0
    Aa = np.vstack([A, np.sqrt(alpha) * np.eye(n_rates)])
    ya = np.concatenate([y / scale, np.zeros(n_rates)])
    try:
        x, _ = nnls(Aa, ya, maxiter=20 * n_rates)
    except Exception as exc:
        return {"status": "unresolved", "reason": str(exc)}
    idx, _ = find_peaks(x, height=0.05 * (x.max() or 1.0))
    return {
        "status": "ok",
        "emission_rate_grid_per_s": e.tolist(),
        "spectrum": (x * scale).tolist(),
        "components": [{"emission_rate_per_s": float(e[i]), "relative_amplitude": float(x[i] / x.sum())} for i in idx],
        "flags": ["approximate_regularized_inverse_laplace"],
    }
