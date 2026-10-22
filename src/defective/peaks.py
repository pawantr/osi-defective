"""Digitized trap spectra -> trap energies.

Curve loading, digitized-axis conversion and quick sanity-check estimates live here.
The actual fits (TSC peak deconvolution; DLTS/I-DLTS/PICTS Arrhenius; Laplace DLTS)
are delegated to `trapfit` (vendored from trapfit_pkg), via `trapfit.router.run()`,
which reads `trapfit/algorithms.json` to pick the route, validate required inputs,
and attach quality flags / provenance. trapfit never raises for bad or ambiguous
data: it returns {"status": "ok" | "unresolved" | "refused", ...}. This module
preserves that contract end to end instead of forcing a label.

TSC peaks are the Kitis et al. (1998) analytical glow-peak forms (first-order and
general-order), fitted simultaneously in log10(current) with BIC peak-count
selection -- see trapfit/tsc.py. DLTS/I-DLTS/PICTS use rate-window peak finding,
cross-window linking and a Arrhenius regression -- see trapfit/arrhenius.py.

Units: T in K, Ea in eV, rates in s^-1, beta in K/s, sigma in cm^2.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from scipy import constants as C
from scipy.signal import find_peaks, savgol_filter

from .trapfit import run as trapfit_run
from .trapfit.tsc import peak_first_order, peak_general_order

K_B = C.k / C.e  # Boltzmann constant, eV/K


# ---------------------------------------------------------------- I/O
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def load_curve(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a 2-column (x, y) text/CSV file. Header, comment and non-numeric rows
    are skipped; comma / semicolon / tab / whitespace delimiters. Returns arrays sorted by x."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [p for p in re.split(r"[,;\t ]+", line.strip()) if p]
        if len(parts) < 2 or not all(_NUM.fullmatch(p) for p in parts[:2]):
            continue
        rows.append((float(parts[0]), float(parts[1])))
    a = np.array(rows, dtype=float).reshape(-1, 2)
    a = a[np.isfinite(a).all(axis=1)]
    if len(a) < 3:
        raise ValueError(f"{path}: need >= 3 numeric (x, y) rows, found {len(a)}")
    a = a[np.argsort(a[:, 0], kind="stable")]
    return a[:, 0], a[:, 1]


def to_rate(x, y, x_kind: str = "T", y_kind: str = "rate") -> tuple[np.ndarray, np.ndarray]:
    """Convert digitized Arrhenius axes to (T [K], e [s^-1]).
    x_kind: 'T' | '1000/T' | '1/T'.  y_kind: 'rate' (s^-1) | 'tau' (s) | 'ln(e/T^2)' | 'ln(tau*T^2)'."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    T = {"T": lambda: x, "1000/T": lambda: 1000 / x, "1/T": lambda: 1 / x}[x_kind]()
    e = {"rate": lambda: y, "tau": lambda: 1 / y, "ln(e/T^2)": lambda: T**2 * np.exp(y),
         "ln(tau*T^2)": lambda: T**2 / np.exp(y)}[y_kind]()
    return T, e


def tsc_component(T, peak: dict) -> np.ndarray:
    """Evaluate one fitted TSC peak (a `traps` entry from fit_curve) on a temperature grid,
    for plotting alongside the data. Uses the same Kitis peak model trapfit fitted it with."""
    Im, Tm, E = peak["peak_height_A"], peak["peak_T_K"], peak["energy_eV"]
    if peak.get("order_b", 1.0) and abs(peak.get("order_b", 1.0) - 1.0) > 1e-9:
        return peak_general_order(T, Im, Tm, E, peak["order_b"])
    return peak_first_order(T, Im, Tm, E)


# ---------------------------------------------------------------- quick estimates (TSC, diagnostic only)
def _smooth(y: np.ndarray) -> np.ndarray:
    w = max(5, (len(y) // 25) | 1)
    return savgol_filter(y, w, 2) if len(y) > w else y


def _candidates(T, y, prominence=0.03):
    ys = _smooth(y)
    idx, props = find_peaks(ys, prominence=prominence * np.ptp(ys))
    order = np.argsort(props["prominences"])[::-1]
    return [int(i) for i in idx[order]], ys


def _half_points(T, ys, i):
    half = ys[i] / 2
    lo = i
    while lo > 0 and ys[lo] > half:
        lo -= 1
    hi = i
    while hi < len(ys) - 1 and ys[hi] > half:
        hi += 1
    if ys[lo] > half or ys[hi] > half:
        return None, None
    t1 = np.interp(half, [ys[lo], ys[lo + 1]], [T[lo], T[lo + 1]])
    t2 = np.interp(half, [ys[hi], ys[hi - 1]], [T[hi], T[hi - 1]])
    return float(t1), float(t2)


def initial_rise(T, y, i_peak: int, frac=(0.02, 0.15)) -> dict | None:
    """Ea from the low-T flank: points with frac[0] < I/Im < frac[1], contiguous below the peak."""
    Im = y[i_peak]
    sel = []
    for j in range(i_peak, -1, -1):
        r = y[j] / Im
        if r <= frac[0]:
            break
        if r < frac[1]:
            sel.append(j)
    if len(sel) < 4:
        return None
    p, cov = np.polyfit(1 / T[sel], np.log(y[sel]), 1, cov=True)
    return {"Ea_eV": float(-p[0] * K_B), "Ea_err_eV": float(math.sqrt(cov[0, 0]) * K_B), "n_points": len(sel)}


def quick_estimates(T, y) -> list[dict]:
    """Per resolved peak: Tm, initial-rise Ea and Chen peak-shape Ea (unreliable when peaks overlap).
    Diagnostic cross-check only; the authoritative fit is trapfit's simultaneous multi-peak fit."""
    T, y = np.asarray(T, float), np.asarray(y, float)
    idx, ys = _candidates(T, y)
    out = []
    for i in sorted(idx, key=lambda j: T[j]):
        Tm = float(T[i])
        t1, t2 = _half_points(T, ys, i)
        rec = {"Tm_K": Tm, "initial_rise": initial_rise(T, ys, i)}
        if t1 is not None:
            tau, delta, omega = Tm - t1, t2 - Tm, t2 - t1
            kT2 = K_B * Tm**2
            mu = delta / omega   # 0.42 for an isolated first-order peak
            rec["chen"] = {"E_tau": 1.51 * kT2 / tau - 3.16 * K_B * Tm, "E_delta": 0.976 * kT2 / delta,
                           "E_omega": 2.52 * kT2 / omega - 2 * K_B * Tm, "mu_g": mu,
                           "isolated_first_order_shape": bool(abs(mu - 0.42) < 0.04)}
        out.append(rec)
    return out


# ---------------------------------------------------------------- dispatch (trapfit-backed)
_CANONICAL = {"TSC": "TSC", "DLTS": "DLTS", "CDLTS": "DLTS", "IDLTS": "I-DLTS",
              "PICTS": "PICTS", "LAPLACEDLTS": "LAPLACE_DLTS"}


def fit_curve(path: str | Path, technique: str, heating_rate_K_per_s: float | None = None,
              max_peaks: int = 4, m_eff: float | None = None, x_kind: str = "T",
              y_kind: str = "rate") -> dict:
    """CSV -> trap energies, via trapfit.router.run(). TSC: (T, current) curve, deconvolved into
    simultaneously-fitted Kitis glow peaks. DLTS/I-DLTS/PICTS: digitized Arrhenius pairs (T or Tm,
    emission rate), converted with x_kind/y_kind. LAPLACE_DLTS: (time_s, signal) isothermal transient.

    Never raises for a bad/ambiguous fit: returns trapfit's status dict ("ok" | "unresolved" |
    "refused") instead of forcing a label. Raises only if the CSV itself cannot be parsed."""
    x, y = load_curve(path)
    norm = re.sub(r"[^A-Z0-9]", "", technique.upper())
    tech = _CANONICAL.get(norm, technique.upper())

    if tech == "TSC":
        request = {"technique": "TSC",
                   "data": {"temperature_K": x.tolist(), "current_A": y.tolist()},
                   "metadata": {"heating_rate_K_per_s": heating_rate_K_per_s},
                   "options": {"max_peaks": max_peaks}}
        res = trapfit_run(request)
        if res["status"] == "ok":
            res["quick_estimates"] = quick_estimates(x, y)
        return res

    if tech == "LAPLACE_DLTS":
        request = {"technique": "LAPLACE_DLTS", "data": {"time_s": x.tolist(), "signal": y.tolist()}}
        return trapfit_run(request)

    if tech in {"DLTS", "I-DLTS", "PICTS"}:
        T, e = to_rate(x, y, x_kind, y_kind)
        points = [{"peak_T_K": float(t), "emission_rate_per_s": float(r)} for t, r in zip(T, e)]
        request = {"technique": tech, "data": {"arrhenius_points": points},
                   "metadata": {"effective_mass_ratio": m_eff or 1.0}}
        return trapfit_run(request)

    return {"status": "refused", "reason": f"unknown technique {technique!r}; "
                                            "use TSC, DLTS, I-DLTS, PICTS or LAPLACE_DLTS"}
