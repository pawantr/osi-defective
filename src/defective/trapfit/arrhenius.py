"""Rate-window analysis for DLTS, I-DLTS and PICTS: peak per window, linking, Arrhenius fit."""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter

K_B = 8.617333262e-5  # eV/K
GAMMA_0 = 3.25e21  # cm^-2 s^-1 K^-2 for m*/m0 = 1, degeneracy 1 (apparent cross-section convention)


def rate_window_emission(t1_s: float, t2_s: float) -> float:
    """Emission rate at the peak of a double-boxcar window: ln(t2/t1)/(t2-t1)."""
    if t1_s <= 0 or t2_s <= t1_s:
        raise ValueError("need 0 < t1 < t2")
    return float(np.log(t2_s / t1_s) / (t2_s - t1_s))


def arrhenius_fit(peak_T_K, emission_rate_per_s, effective_mass_ratio=1.0, sigma_T_K=None):
    """Fit ln(e/T^2) = ln(gamma*sigma) - E/(kT). Returns energy, sigma and apparent cross-section."""
    T = np.asarray(peak_T_K, float)
    e = np.asarray(emission_rate_per_s, float)
    if len(T) < 2 or np.any(T <= 0) or np.any(e <= 0):
        return {"status": "unresolved", "reason": "need >=2 valid points"}
    x = 1.0 / (K_B * T)
    y = np.log(e / T**2)
    w = None
    if sigma_T_K is not None:
        s = np.asarray(sigma_T_K, float)
        # dy/dT ~ (E/(kT^2) - 2/T); use E ~ 0.4 eV for weighting only
        sy = np.abs(0.4 / (K_B * T**2) - 2 / T) * s
        w = 1.0 / np.clip(sy, 1e-6, None)
    if len(T) >= 3:
        coef, cov = np.polyfit(x, y, 1, w=w, cov="unscaled" if w is not None else True)
    else:
        coef = np.polyfit(x, y, 1)
        cov = np.full((2, 2), np.nan)
    slope, intercept = coef
    E = -float(slope)
    sE = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else float("nan")
    gamma = GAMMA_0 * effective_mass_ratio
    sigma_cap = float(np.exp(intercept) / gamma)
    resid = y - (slope * x + intercept)
    return {
        "status": "ok",
        "energy_eV": E,
        "energy_sigma_eV": sE,
        "apparent_cross_section_cm2": sigma_cap,
        "intercept_ln": float(intercept),
        "rms_residual_ln": float(np.sqrt(np.mean(resid**2))),
        "n_windows": int(len(T)),
    }


def _refine_peak(T, y, i, half=6):
    lo, hi = max(0, i - half), min(len(T), i + half + 1)
    xs, ys = T[lo:hi], y[lo:hi]
    if len(xs) < 5:
        return float(T[i])
    g = lambda x, a, mu, s, c: a * np.exp(-0.5 * ((x - mu) / s) ** 2) + c
    try:
        p, _ = curve_fit(g, xs, ys, p0=[ys.max() - ys.min(), T[i], max((xs[-1] - xs[0]) / 4, 1.0), ys.min()], maxfev=2000)
        if xs[0] <= p[1] <= xs[-1]:
            return float(p[1])
    except Exception:
        pass
    return float(T[i])


def find_spectrum_peaks(T, signal, smooth_window=7, min_prominence_fraction=0.1, peak_sign="positive"):
    T = np.asarray(T, float)
    y = np.asarray(signal, float)
    if peak_sign == "negative":
        y = -y
    n = len(y)
    win = min(smooth_window, (n // 2) * 2 - 1)
    ys = savgol_filter(y, win, 3) if win >= 5 else y
    prom = min_prominence_fraction * (np.ptp(ys) or 1.0)
    idx, _ = find_peaks(ys, prominence=prom)
    return [_refine_peak(T, ys, int(i)) for i in idx]


def link_peaks(peaks_by_window, max_shift_K=30.0):
    """Link peaks across windows ordered by increasing emission rate. Peak T must not decrease."""
    if not peaks_by_window or not peaks_by_window[0]:
        return []
    tracks = [[(0, p)] for p in peaks_by_window[0]]
    for w in range(1, len(peaks_by_window)):
        used = set()
        for tr in tracks:
            prev = tr[-1][1]
            cands = [(abs(p - prev), j, p) for j, p in enumerate(peaks_by_window[w]) if j not in used and prev - 1.0 <= p <= prev + max_shift_K]
            if cands:
                _, j, p = min(cands)
                used.add(j)
                tr.append((w, p))
    return tracks


def analyze_dlts(spectra=None, arrhenius_points=None, effective_mass_ratio=1.0, max_shift_K=30.0, smooth_window=7, min_prominence_fraction=0.1, peak_sign="positive", min_windows=3):
    """Analyze rate-window spectra (DLTS, I-DLTS, PICTS) or pre-extracted Arrhenius points."""
    traps = []
    if arrhenius_points:
        T = [p["peak_T_K"] for p in arrhenius_points]
        e = [p["emission_rate_per_s"] for p in arrhenius_points]
        if len(T) < min_windows:
            return {"status": "refused", "reason": f"need at least {min_windows} points, got {len(T)}"}
        fit = arrhenius_fit(T, e, effective_mass_ratio)
        if fit["status"] != "ok":
            return {"status": "unresolved", "reason": fit.get("reason", "fit failed")}
        fit["peak_T_K"] = float(np.mean(T))
        fit["flags"] = ["from_reported_points"]
        return {"status": "ok", "traps": [fit]}
    if not spectra:
        return {"status": "refused", "reason": "need data.spectra or data.arrhenius_points"}
    rates = []
    for s in spectra:
        if "emission_rate_per_s" in s:
            rates.append(float(s["emission_rate_per_s"]))
        elif "t1_s" in s and "t2_s" in s:
            rates.append(rate_window_emission(s["t1_s"], s["t2_s"]))
        else:
            return {"status": "refused", "reason": "each spectrum needs emission_rate_per_s or t1_s and t2_s"}
    if len(spectra) < min_windows:
        return {"status": "refused", "reason": f"need at least {min_windows} rate windows, got {len(spectra)}"}
    order = np.argsort(rates)
    peaks_by_window = [find_spectrum_peaks(spectra[i]["temperature_K"], spectra[i]["signal"], smooth_window, min_prominence_fraction, peak_sign) for i in order]
    rates_sorted = [rates[i] for i in order]
    tracks = link_peaks(peaks_by_window, max_shift_K)
    for tr in tracks:
        if len(tr) < min_windows:
            traps.append({"status": "unresolved", "flags": ["peak_not_tracked_in_enough_windows"], "peak_T_K": float(np.mean([p for _, p in tr]))})
            continue
        T = [p for _, p in tr]
        e = [rates_sorted[w] for w, _ in tr]
        fit = arrhenius_fit(T, e, effective_mass_ratio)
        fit["peak_T_K"] = float(np.mean(T))
        fit["flags"] = []
        if fit.get("rms_residual_ln", 0) > 0.3:
            fit["flags"].append("poor_arrhenius_linearity")
        traps.append(fit)
    status = "ok" if any(t.get("status") == "ok" for t in traps) else "unresolved"
    return {"status": status, "traps": traps, "n_windows": len(spectra)}
