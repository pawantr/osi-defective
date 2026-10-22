"""Digitized trap spectra -> trap energies (TSC peak deconvolution, DLTS/PICTS Arrhenius).

TSC (first-order kinetics, Randall-Wilkins / Bube; no retrapping)
    I(T) = n0 q s exp(-Ea/kT) exp[-(s/beta) * F(T)],  F(T) = int_{T0}^{T} exp(-Ea/kT') dT'
  With T0 -> 0, F(T) = T E2(Ea/kT) exactly (E2 = exponential integral of order 2).
  The peak condition  beta Ea / (k Tm^2) = s exp(-Ea/kTm)  is then exact, so each peak is
  fitted as (Ea, Tm, Im) -- far less correlated than (Ea, s) -- and s is derived:
    ln I(T) - ln Im = (xm - x) + g(xm) - (T/Tm)^2 exp(xm - x) g(x),
    x = Ea/kT, xm = Ea/kTm, g(x) = x e^x E2(x)   (g -> 1 - 2/x + 6/x^2 ... for large x).
  The shape depends on (Ea, Tm) only; beta enters through s = beta Ea/(k Tm^2) exp(xm).
  N (number of peaks) is chosen by BIC = n ln(RSS/n) + p ln n over 1..max_peaks.
  Fallbacks (no fit): initial rise (ln I vs 1/T on the low-T flank, slope -Ea/k) and
  Chen's peak-shape formulas for first order (tau, delta, omega = low, high, full half-widths):
    E_tau = 1.51 kTm^2/tau - 3.16 kTm,  E_delta = 0.976 kTm^2/delta,  E_omega = 2.52 kTm^2/omega - 2 kTm.

DLTS / I-DLTS / PICTS (Arrhenius)
    e(T) = sigma_a v_th N_c exp(-Ea/kT),  v_th = sqrt(3kT/m*),  N_c = 2 (2 pi m* k T / h^2)^(3/2)
    => ln(e/T^2) = ln(gamma sigma_a) - Ea/(kT),
       gamma = 2 sqrt(3) (2 pi)^(3/2) k^2 m* / h^3 = 3.256e21 (m*/m0) cm^-2 s^-1 K^-2.
  gamma is an explicit argument (or built from m_eff via gamma_from_mass; v_th convention
  "3kT/m" or "8kT/pi m", the latter gives 3.000e21). sigma_a is the *apparent* cross-section
  (cm^2): it absorbs the entropy factor exp(dS/k) and any T-dependence of sigma.
  At a DLTS/PICTS peak the emission rate equals the rate window, so (Tm, rate window) pairs
  are (T, e) pairs.

Units: T in K, Ea in eV, rates in s^-1, beta in K/s, sigma in cm^2, gamma in cm^-2 s^-1 K^-2.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from scipy import constants as C
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter
from scipy.special import expn

K_B = C.k / C.e  # Boltzmann constant, eV/K


# ---------------------------------------------------------------- I/O
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def load_curve(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a 2-column (T [K], signal) text/CSV file. Header, comment and non-numeric rows
    are skipped; comma / semicolon / tab / whitespace delimiters. Returns arrays sorted by T."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [p for p in re.split(r"[,;\t ]+", line.strip()) if p]
        if len(parts) < 2 or not all(_NUM.fullmatch(p) for p in parts[:2]):
            continue
        rows.append((float(parts[0]), float(parts[1])))
    a = np.array(rows, dtype=float).reshape(-1, 2)
    a = a[np.isfinite(a).all(axis=1)]
    if len(a) < 3:
        raise ValueError(f"{path}: need >= 3 numeric (T, signal) rows, found {len(a)}")
    a = a[np.argsort(a[:, 0], kind="stable")]
    return a[:, 0], a[:, 1]


# ---------------------------------------------------------------- TSC model
def _g(x: np.ndarray) -> np.ndarray:
    """g(x) = x e^x E2(x); asymptotic series beyond x = 500 (error < 1e-11)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    lo = x < 500
    xl = x[lo]
    out[lo] = xl * np.exp(xl) * expn(2, xl)
    xh = x[~lo]
    out[~lo] = 1 - 2 / xh + 6 / xh**2 - 24 / xh**3 + 120 / xh**4
    return out


def tsc_peak(T, Ea, Tm, Im) -> np.ndarray:
    """First-order TSC peak with activation energy Ea [eV], maximum Im at Tm [K]."""
    T = np.asarray(T, dtype=float)
    x, xm = Ea / (K_B * T), Ea / (K_B * Tm)
    arg = np.minimum(xm - x, 700.0)
    lnr = (xm - x) + _g(np.array([xm]))[0] - (T / Tm) ** 2 * np.exp(arg) * _g(x)
    return Im * np.exp(np.maximum(lnr, -745.0))


def prefactor(Ea, Tm, beta):
    """Attempt-to-escape frequency s [s^-1] from the first-order peak condition."""
    return beta * Ea / (K_B * Tm**2) * math.exp(Ea / (K_B * Tm))


def gamma_from_mass(m_eff: float, vth: str = "3kT/m") -> float:
    """gamma in e/T^2 = gamma sigma exp(-Ea/kT), cm^-2 s^-1 K^-2, for m* = m_eff * m0."""
    g = 2 * math.sqrt(3) * (2 * math.pi) ** 1.5 * C.k**2 * C.m_e / C.h**3 * 1e-4  # cm^-2 s^-1 K^-2
    if vth == "8kT/pi m":
        g *= math.sqrt(8 / (3 * math.pi))
    elif vth != "3kT/m":
        raise ValueError("vth must be '3kT/m' or '8kT/pi m'")
    return g * m_eff


# ---------------------------------------------------------------- quick estimates
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
    """Per resolved peak: Tm, initial-rise Ea and Chen peak-shape Ea (unreliable when peaks overlap)."""
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


# ---------------------------------------------------------------- TSC deconvolution
def _model(p, T, n, baseline):
    y = sum(tsc_peak(T, *p[3 * k:3 * k + 3]) for k in range(n))
    return y + (p[-1] if baseline else 0.0)


def _fit_n(T, y, guesses, baseline):
    n = len(guesses)
    tlo, thi = T.min(), T.max()
    p0, lb, ub = [], [], []
    for Ea, Tm, Im in guesses:
        p0 += [np.clip(Ea, 0.02, 2.9), np.clip(Tm, tlo, thi), np.clip(Im, 1e-6, 4.9)]
        lb += [0.01, tlo, 0.0]
        ub += [3.0, thi, 5.0]
    if baseline:
        p0.append(0.0); lb.append(-1.0); ub.append(1.0)
    r = least_squares(lambda p: _model(p, T, n, baseline) - y, p0, bounds=(lb, ub), x_scale="jac",
                      max_nfev=4000)
    return r


def _ea_guess(T, ys, i):
    t1, t2 = _half_points(T, ys, i)
    Tm = T[i]
    if t1 is not None and t2 > t1:
        e = 2.52 * K_B * Tm**2 / (t2 - t1) - 2 * K_B * Tm
        if 0.02 < e < 3:
            return e
    return 25 * K_B * Tm


def fit_tsc(T, I, beta: float | None = None, max_peaks: int = 4, baseline: bool = False,
            m_eff: float | None = None, vth: str = "3kT/m") -> dict:
    """Deconvolve a TSC curve into N first-order peaks, N chosen by BIC.

    beta: heating rate in K/s (needed only for s and sigma). m_eff: m*/m0 to convert s to an
    apparent capture cross-section sigma = s / (gamma Tm^2) (cm^2). Returns peaks sorted by Tm."""
    T, I = np.asarray(T, float), np.asarray(I, float)
    scale = float(np.max(np.abs(I))) or 1.0
    y = I / scale
    n_pts = len(T)
    cand, ys = _candidates(T, y)
    if not cand:
        cand = [int(np.argmax(ys))]
    cand_guess = [(_ea_guess(T, ys, i), T[i], max(ys[i], 1e-3)) for i in cand]
    trials, prev = [], None
    for n in range(1, max_peaks + 1):
        if n + 1 > n_pts // 4:
            break
        starts = []
        if len(cand_guess) >= n:
            starts.append(cand_guess[:n])
        if prev is not None:
            res = y - _model(prev.x, T, n - 1, baseline)
            j = int(np.argmax(_smooth(res)))
            old = [tuple(prev.x[3 * k:3 * k + 3]) for k in range(n - 1)]
            amp = max(res[j], 0.05)
            starts.append(old + [(25 * K_B * T[j], T[j], amp)])
            # also try a new peak on either side of the dominant one
            for k0 in range(n - 1):
                Ea0, Tm0, Im0 = old[k0]
                w = 2.52 * K_B * Tm0**2 / Ea0
                for d in (-w, w):
                    Tn = float(np.clip(Tm0 + d, T.min(), T.max()))
                    starts.append(old + [(25 * K_B * Tn, Tn, 0.3 * Im0)])
        best = None
        for s in starts:
            try:
                r = _fit_n(T, y, s, baseline)
            except ValueError:
                continue
            if best is None or r.cost < best.cost:
                best = r
        if best is None:
            break
        p = 3 * n + baseline
        rss = 2 * best.cost
        bic = n_pts * math.log(max(rss, 1e-300) / n_pts) + p * math.log(n_pts)
        trials.append((n, bic, best))
        prev = best
    if not trials:
        raise ValueError("TSC fit failed")
    n, bic, r = min(trials, key=lambda t: t[1])
    p = len(r.x)
    rss = 2 * r.cost
    dof = max(n_pts - p, 1)
    J = r.jac
    _, sv, VT = np.linalg.svd(J, full_matrices=False)
    tol = np.finfo(float).eps * max(J.shape) * sv[0]
    sv = sv[sv > tol]
    cov = (VT[:len(sv)].T / sv**2) @ VT[:len(sv)] * rss / dof
    err = np.sqrt(np.clip(np.diag(cov), 0, None))
    gam = gamma_from_mass(m_eff, vth) if m_eff else None
    peaks = []
    for k in range(n):
        Ea, Tm, Im = r.x[3 * k:3 * k + 3]
        dEa, dTm, dIm = err[3 * k:3 * k + 3]
        Tg = np.linspace(max(T.min() * 0.5, 1.0), T.max() * 1.5, 4000)
        pk = {"Ea_eV": float(Ea), "Ea_err_eV": float(dEa), "Tm_K": float(Tm), "Tm_err_K": float(dTm),
              "Im": float(Im * scale), "Im_err": float(dIm * scale),
              "area_signal_K": float(np.trapezoid(tsc_peak(Tg, Ea, Tm, Im), Tg) * scale),
              "at_bound": bool(np.isclose(Tm, T.min()) or np.isclose(Tm, T.max()) or Ea < 0.011 or Ea > 2.99)}
        if beta:
            s = float(prefactor(Ea, Tm, beta))
            pk["s_per_s"] = s
            pk["log10_s"] = math.log10(s)
            pk["charge_signal_s"] = pk["area_signal_K"] / beta   # Q = int I dt; C if I in A
            if gam:
                pk["sigma_cm2"] = float(s / (gam * Tm**2))
        peaks.append(pk)
    peaks.sort(key=lambda d: d["Tm_K"])
    resid = y - _model(r.x, T, n, baseline)
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return {
        "technique": "TSC", "model": "first-order (Randall-Wilkins), sum of N peaks",
        "n_peaks": n, "peaks": peaks,
        "baseline": float(r.x[-1] * scale) if baseline else None,
        "fit": {"r2": 1 - float(np.sum(resid**2)) / ss_tot, "rmse": float(np.sqrt(np.mean(resid**2)) * scale),
                "n_points": n_pts, "bic_by_n": {t[0]: round(t[1], 2) for t in trials}},
        "heating_rate_K_per_s": beta, "gamma_cm2_s_K2": gam,
    }


# ---------------------------------------------------------------- DLTS / PICTS
def peak_temperatures(T, S, prominence=0.05) -> list[float]:
    """Peak temperatures of one DLTS/PICTS spectrum (parabolic sub-grid refinement), ascending.
    Collect them across rate windows to build (Tm, e) pairs for fit_arrhenius."""
    T, S = np.asarray(T, float), np.asarray(S, float)
    out = []
    for sign in (1, -1):    # majority (positive) and minority/inverted (negative) peaks
        idx, _ = find_peaks(sign * _smooth(S), prominence=prominence * np.ptp(S))
        for i in idx:
            if 0 < i < len(T) - 1:
                a, b, _c = np.polyfit(T[i - 1:i + 2], S[i - 1:i + 2], 2)
                out.append(float(-b / (2 * a)) if a != 0 else float(T[i]))
            else:
                out.append(float(T[i]))
    return sorted(out)


def to_rate(x, y, x_kind: str = "T", y_kind: str = "rate") -> tuple[np.ndarray, np.ndarray]:
    """Convert digitized Arrhenius axes to (T [K], e [s^-1]).
    x_kind: 'T' | '1000/T' | '1/T'.  y_kind: 'rate' (s^-1) | 'tau' (s) | 'ln(e/T^2)' | 'ln(tau*T^2)'."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    T = {"T": lambda: x, "1000/T": lambda: 1000 / x, "1/T": lambda: 1 / x}[x_kind]()
    e = {"rate": lambda: y, "tau": lambda: 1 / y, "ln(e/T^2)": lambda: T**2 * np.exp(y),
         "ln(tau*T^2)": lambda: T**2 / np.exp(y)}[y_kind]()
    return T, e


def fit_arrhenius(T, rate, gamma: float | None = None, m_eff: float | None = None,
                  vth: str = "3kT/m") -> dict:
    """Fit ln(e/T^2) = ln(gamma sigma) - Ea/(kT). Give gamma (cm^-2 s^-1 K^-2) or m_eff (m*/m0)
    to obtain the apparent cross-section sigma (cm^2); otherwise only Ea and gamma*sigma."""
    T, e = np.asarray(T, float), np.asarray(rate, float)
    if len(T) < 2 or np.any(e <= 0) or np.any(T <= 0):
        raise ValueError("need >= 2 points with T > 0 and rate > 0")
    if gamma is None and m_eff:
        gamma = gamma_from_mass(m_eff, vth)
    x, yy = 1 / (K_B * T), np.log(e / T**2)
    A = np.vstack([np.ones_like(x), -x]).T
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    ln_gs, Ea = coef
    resid = yy - A @ coef
    dof = len(T) - 2
    if dof > 0:
        cov = np.linalg.inv(A.T @ A) * float(resid @ resid) / dof
        d_ln, dEa = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])
    else:
        d_ln = dEa = None
    ss_tot = float(np.sum((yy - yy.mean()) ** 2)) or 1.0
    out = {"Ea_eV": float(Ea), "Ea_err_eV": dEa, "gamma_sigma_per_s_K2": float(math.exp(ln_gs)),
           "ln_gamma_sigma_err": d_ln, "gamma_cm2_s_K2": gamma, "sigma_cm2": None, "sigma_range_cm2": None,
           "fit": {"r2": 1 - float(resid @ resid) / ss_tot, "n_points": len(T),
                   "T_range_K": [float(T.min()), float(T.max())]}}
    if gamma:
        sig = math.exp(ln_gs) / gamma
        out["sigma_cm2"] = sig
        if d_ln is not None:
            out["sigma_range_cm2"] = [sig * math.exp(-d_ln), sig * math.exp(d_ln)]
    return out


def fit_curve(path: str | Path, technique: str, heating_rate_K_per_s: float | None = None,
              max_peaks: int = 4, m_eff: float | None = None, x_kind: str = "T",
              y_kind: str = "rate", baseline: bool = False) -> dict:
    """CSV -> trap energies. TSC: (T, current) curve. DLTS/I-DLTS/PICTS: Arrhenius pairs."""
    x, y = load_curve(path)
    tech = technique.upper().replace("_", "-")
    if tech == "TSC":
        try:
            res = fit_tsc(x, y, heating_rate_K_per_s, max_peaks, baseline, m_eff)
        except (ValueError, np.linalg.LinAlgError) as err:
            res = {"technique": "TSC", "error": str(err), "peaks": []}
        res["quick_estimates"] = quick_estimates(x, y)
        return res
    if tech in {"DLTS", "I-DLTS", "PICTS", "CDLTS"}:
        T, e = to_rate(x, y, x_kind, y_kind)
        return {"technique": tech, **fit_arrhenius(T, e, m_eff=m_eff)}
    raise ValueError(f"unknown technique {technique!r}; use TSC, DLTS, I-DLTS or PICTS")
