"""Simultaneous multi-peak fitting for thermally stimulated current (TSC) curves.

Peak shapes are the Kitis et al. (1998) analytical glow-peak forms, written in
terms of peak height Im, peak temperature Tm and activation energy E.
All peaks are fitted together on log10(current), with greedy peak addition and
BIC model selection.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, savgol_filter

K_B = 8.617333262e-5  # eV/K


def peak_first_order(T, Im, Tm, E):
    """First-order (Randall-Wilkins, Kitis form) glow peak."""
    T = np.asarray(T, float)
    d = 2 * K_B * T / E
    dm = 2 * K_B * Tm / E
    x = np.clip(E / (K_B * T) * (T - Tm) / Tm, -700, 60)
    with np.errstate(over="ignore", under="ignore"):
        arg = 1 + x - (T**2 / Tm**2) * np.exp(x) * (1 - d) - dm
    return Im * np.exp(np.clip(arg, -700, 700))


def peak_general_order(T, Im, Tm, E, b):
    """General-order (May-Partridge, Kitis form) glow peak, 1 < b <= 2."""
    T = np.asarray(T, float)
    d = 2 * K_B * T / E
    dm = 2 * K_B * Tm / E
    x = np.clip(E / (K_B * T) * (T - Tm) / Tm, -700, 60)
    zm = 1 + (b - 1) * dm
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        inner = (b - 1) * (1 - d) * (T**2 / Tm**2) * np.exp(x) + zm
        val = Im * b ** (b / (b - 1)) * np.exp(x) * np.power(np.clip(inner, 1e-300, None), -b / (b - 1))
    return val


def _unpack(p, order):
    n_par = 4 if order == "general_order_kitis" else 3
    return p.reshape(-1, n_par), n_par


def model_sum(T, p, order="first_order_kitis"):
    peaks, _ = _unpack(np.asarray(p, float), order)
    total = np.zeros_like(np.asarray(T, float))
    for row in peaks:
        Im, Tm, E = 10 ** row[0], row[1], row[2]
        if order == "general_order_kitis":
            total += peak_general_order(T, Im, Tm, E, row[3])
        else:
            total += peak_first_order(T, Im, Tm, E)
    return total


def _residuals(p, T, ylog, order, floor):
    m = model_sum(T, p, order)
    return np.log10(np.clip(m, floor, None)) - ylog


def _bounds(n, T, order, ymax_log):
    lo_row = [ymax_log - 8, T.min() + 1.0, 0.05] + ([1.001] if order == "general_order_kitis" else [])
    hi_row = [ymax_log + 0.5, T.max() - 1.0, 1.6] + ([2.0] if order == "general_order_kitis" else [])
    return np.tile(lo_row, n), np.tile(hi_row, n)


def _initial_rows(Tms, T, ylog, order, E0):
    rows = []
    for Tm in Tms:
        h = np.interp(Tm, T, ylog)
        row = [h, Tm, E0]
        if order == "general_order_kitis":
            row.append(1.5)
        rows.append(row)
    return np.array(rows, float).ravel()


def _fit_once(p0, T, ylog, order, floor, lb, ub):
    p0 = np.clip(p0, lb + 1e-9, ub - 1e-9)
    return least_squares(_residuals, p0, args=(T, ylog, order, floor), bounds=(lb, ub), method="trf", x_scale="jac", max_nfev=400)


def _multistart(p0, T, ylog, order, floor, lb, ub, n_starts, rng):
    best = None
    for k in range(max(1, n_starts)):
        pk = p0.copy()
        if k > 0:
            pk = pk.reshape(-1, 4 if order == "general_order_kitis" else 3)
            pk[:, 1] += rng.normal(0, 3.0, len(pk))
            pk[:, 2] *= rng.uniform(0.7, 1.4, len(pk))
            pk = pk.ravel()
        try:
            res = _fit_once(pk, T, ylog, order, floor, lb, ub)
        except Exception:
            continue
        if best is None or res.cost < best.cost:
            best = res
    return best


def _bic(rss, n_obs, n_par):
    return n_obs * np.log(max(rss, 1e-300) / n_obs) + n_par * np.log(n_obs)


def fit_tsc(
    temperature_K,
    current_A,
    dark_current_A=0.0,
    heating_rate_K_per_s=None,
    max_peaks=16,
    peak_model="first_order_kitis",
    criterion="bic",
    n_starts=6,
    floor_fraction=1e-5,
    seed=0,
    config=None,
):
    """Fit all TSC peaks simultaneously. Returns a dict with status, peaks, fit statistics.

    status is "ok" or "unresolved". Never raises for bad data; returns the reason instead.
    """
    T = np.asarray(temperature_K, float)
    I = np.asarray(current_A, float) - np.asarray(dark_current_A, float)
    if T.ndim != 1 or T.shape != I.shape or len(T) < 20:
        return {"status": "unresolved", "reason": "need at least 20 points in matching 1-D arrays"}
    order_idx = np.argsort(T)
    T, I = T[order_idx], I[order_idx]
    if not np.all(np.diff(T) > 0):
        return {"status": "unresolved", "reason": "temperature must be strictly increasing"}
    if np.nanmax(I) <= 0:
        return {"status": "unresolved", "reason": "no positive signal after dark-current subtraction"}

    th = (config or {}).get("quality_thresholds", {})
    floor = floor_fraction * np.nanmax(I)
    good = np.isfinite(I)
    T, I = T[good], I[good]
    ylog = np.log10(np.clip(I, floor, None))
    ymax = float(np.max(ylog))
    rng = np.random.default_rng(seed)
    order = peak_model
    n_obs = len(T)
    n_par_row = 4 if order == "general_order_kitis" else 3

    win = min(11, (len(T) // 2) * 2 - 1)
    ys = savgol_filter(ylog, win, 3) if win >= 5 else ylog
    span = float(np.ptp(ys)) or 1.0
    pk_idx, pk_props = find_peaks(ys, prominence=0.02 * span)
    if len(pk_idx):
        # Most prominent candidates first, capped to max_peaks: the greedy loop below starts
        # from the single most prominent peak and adds candidates in this order, so BIC gets a
        # chance to prefer fewer peaks even when find_peaks over-detects on noisy data.
        top = np.argsort(pk_props["prominences"])[::-1][:max_peaks]
        Tms = [float(t) for t in T[pk_idx[top]]]
    else:
        Tms = [float(T[int(np.argmax(ys))])]

    history = []
    best = None
    stall = 0
    n_try = 1
    while n_try <= max_peaks:
        if n_try <= len(Tms):
            Tm_list = Tms[:n_try]
        else:
            Tm_list = list(best["Tm"])
            resid = ylog - np.log10(np.clip(model_sum(T, best["p"], order), floor, None))
            Tm_list.append(float(T[int(np.argmax(resid))]))
        p0 = _initial_rows(Tm_list, T, ylog, order, E0=0.3)
        lb, ub = _bounds(len(Tm_list), T, order, ymax)
        res = _multistart(p0, T, ylog, order, floor, lb, ub, n_starts, rng)
        if res is None:
            break
        rss = float(np.sum(res.fun**2))
        k = n_par_row * len(Tm_list)
        crit = _bic(rss, n_obs, k) if criterion == "bic" else n_obs * np.log(rss / n_obs) + 2 * k
        cand = {"p": res.x, "res": res, "Tm": [r[1] for r in res.x.reshape(-1, n_par_row)], "crit": crit, "rss": rss, "n": len(Tm_list)}
        history.append({"n_peaks": len(Tm_list), "criterion": float(crit), "rss": rss})
        if best is None or crit < best["crit"] - 2.0:
            best = cand
            stall = 0
        else:
            stall += 1
            if best is not None and cand["crit"] < best["crit"]:
                best = cand
            if stall >= 2:
                break
        n_try = len(Tm_list) + 1

    if best is None:
        return {"status": "unresolved", "reason": "optimizer failed for all starts"}

    res = best["res"]
    k = n_par_row * best["n"]
    dof = max(n_obs - k, 1)
    J = res.jac
    try:
        cov = np.linalg.pinv(J.T @ J) * (best["rss"] / dof)
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    except Exception:
        sig = np.full(k, np.nan)

    rows = best["p"].reshape(-1, n_par_row)
    sig_rows = sig.reshape(-1, n_par_row)
    order_by_T = np.argsort(rows[:, 1])
    heights = np.array([10 ** r[0] for r in rows])
    ymax_lin = heights.max()
    sep_min = th.get("min_peak_separation_K", 8.0)
    weak_frac = th.get("weak_peak_fraction_of_max", 0.05)
    s_abs = th.get("energy_sigma_abs_eV", 0.10)
    s_rel = th.get("energy_sigma_rel", 0.30)
    Tm_sorted = np.sort(rows[:, 1])

    peaks = []
    for j in order_by_T:
        Im, Tm, E = 10 ** rows[j, 0], rows[j, 1], rows[j, 2]
        sE = float(sig_rows[j, 2])
        flags = []
        if not np.isfinite(sE) or sE > s_abs or (E > 0 and sE / E > s_rel):
            flags.append("energy_poorly_constrained")
        nearest = np.min(np.abs(Tm_sorted[Tm_sorted != Tm] - Tm)) if len(Tm_sorted) > 1 else np.inf
        if nearest < sep_min:
            flags.append("overlapped")
        if Im < weak_frac * ymax_lin:
            flags.append("weak_peak")
        if E < 0.06 or E > 1.55:
            flags.append("energy_at_bound")
        d = {
            "peak_T_K": float(Tm),
            "peak_T_sigma_K": float(sig_rows[j, 1]),
            "peak_height_A": float(Im),
            "energy_eV": float(E),
            "energy_sigma_eV": sE,
            "order_b": float(rows[j, 3]) if order == "general_order_kitis" else 1.0,
            "flags": flags,
        }
        if heating_rate_K_per_s:
            d["frequency_factor_per_s"] = float(heating_rate_K_per_s * E / (K_B * Tm**2) * np.exp(E / (K_B * Tm)))
        else:
            d["flags"].append("no_heating_rate_frequency_factor_omitted")
        peaks.append(d)

    fitted = model_sum(T, best["p"], order)
    r = ylog - np.log10(np.clip(fitted, floor, None))
    return {
        "status": "ok",
        "n_peaks": best["n"],
        "peak_model": order,
        "criterion": criterion,
        "criterion_value": float(best["crit"]),
        "rms_log10_residual": float(np.sqrt(np.mean(r**2))),
        "peaks": peaks,
        "history": history,
        "fit_curve_A": fitted.tolist(),
        "temperature_K": T.tolist(),
        "assumptions": ["linear heating ramp", "peaks follow the selected glow-peak model", "peak count chosen by " + criterion],
    }
