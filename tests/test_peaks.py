"""Offline tests for peaks.py on synthetic curves with known ground truth."""
import math

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from defective import peaks
from defective.peaks import K_B

BETA = 0.1  # K/s


def rw_direct(T, Ea, s, n0, beta):
    """Randall-Wilkins by direct numerical integration (independent of the E2 closed form)."""
    Tf = np.linspace(5.0, T.max(), 200001)
    F = cumulative_trapezoid(np.exp(-Ea / (K_B * Tf)), Tf, initial=0)
    I = n0 * s * np.exp(-Ea / (K_B * Tf)) * np.exp(-(s / beta) * F)
    return np.interp(T, Tf, I)


def test_closed_form_matches_numerical_integral():
    T = np.linspace(100, 260, 400)
    Ea, s = 0.40, 1e10
    ref = rw_direct(T, Ea, s, 1.0, BETA)
    Tm = brentq(lambda t: peaks.prefactor(Ea, t, BETA) - s, 100, 260)   # exact peak condition
    assert abs(Tm - T[np.argmax(ref)]) < 0.5
    fit = peaks.tsc_peak(T, Ea, Tm, rw_direct(np.array([Tm]), Ea, s, 1.0, BETA)[0])
    assert np.max(np.abs(fit - ref)) / ref.max() < 2e-3


def _two_peak_csv(tmp_path):
    rng = np.random.default_rng(42)
    T = np.arange(110.0, 240.0, 0.5)
    truth = [(0.35, 1e9, 1.0), (0.42, 1e10, 0.8)]
    I = sum(rw_direct(T, Ea, s, n0, BETA) for Ea, s, n0 in truth)
    I = 1e-12 * I / I.max()                                    # pA-scale current
    I = I + rng.normal(0, 0.01 * I.max(), T.size)               # 1 % noise
    p = tmp_path / "tsc.csv"
    p.write_text("# digitized, Fig. 3\nT (K),I (A)\n" + "\n".join(f"{t:.2f},{i:.6e}" for t, i in zip(T, I)))
    return p, truth


def test_tsc_two_overlapping_peaks(tmp_path):
    path, truth = _two_peak_csv(tmp_path)
    T, I = peaks.load_curve(path)
    assert T.size == 260
    res = peaks.fit_tsc(T, I, beta=BETA, max_peaks=4)
    assert res["n_peaks"] == 2, res["fit"]["bic_by_n"]
    for pk, (Ea, s, _) in zip(res["peaks"], truth):
        assert abs(pk["Ea_eV"] - Ea) < 0.02, pk
        assert 0 < pk["Ea_err_eV"] < 0.02
        assert abs(math.log10(pk["s_per_s"]) - math.log10(s)) < 1.0
    assert res["fit"]["r2"] > 0.99


def test_tsc_single_peak_and_quick_estimates():
    T = np.arange(100.0, 250.0, 0.5)
    I = rw_direct(T, 0.40, 1e10, 1.0, BETA)
    I = I + np.random.default_rng(1).normal(0, 0.005 * I.max(), T.size)
    res = peaks.fit_tsc(T, I, beta=BETA, m_eff=0.2)
    assert res["n_peaks"] == 1
    pk = res["peaks"][0]
    assert abs(pk["Ea_eV"] - 0.40) < 0.01 and pk["sigma_cm2"] > 0
    q = peaks.quick_estimates(T, I)
    assert len(q) == 1
    assert abs(q[0]["chen"]["E_omega"] - 0.40) < 0.03
    assert abs(q[0]["initial_rise"]["Ea_eV"] - 0.40) < 0.04


def test_arrhenius_recovers_ea_and_sigma():
    rng = np.random.default_rng(7)
    Ea, sigma, m_eff = 0.55, 3e-15, 0.26
    gamma = peaks.gamma_from_mass(m_eff)
    assert math.isclose(peaks.gamma_from_mass(1.0), 3.2565e21, rel_tol=1e-3)
    T = np.linspace(230, 300, 8)
    e = gamma * sigma * T**2 * np.exp(-Ea / (K_B * T)) * np.exp(rng.normal(0, 0.03, T.size))
    res = peaks.fit_arrhenius(T, e, m_eff=m_eff)
    assert abs(res["Ea_eV"] - Ea) < 0.01 and res["Ea_err_eV"] < 0.02
    assert 0.5 < res["sigma_cm2"] / sigma < 2.0
    lo, hi = res["sigma_range_cm2"]
    assert lo < res["sigma_cm2"] < hi


def test_dlts_peak_temperatures_feed_arrhenius():
    Ea, sigma, gamma = 0.45, 1e-15, peaks.gamma_from_mass(0.3)
    T = np.arange(120.0, 320.0, 0.25)
    windows = [20.0, 50.0, 100.0, 200.0, 500.0]   # rate windows e_ref (s^-1), t2/t1 = 2
    tm = []
    for w in windows:
        t1 = math.log(2) / w
        e = gamma * sigma * T**2 * np.exp(-Ea / (K_B * T))
        S = np.exp(-e * t1) - np.exp(-2 * e * t1)            # boxcar DLTS, peaks at e = w
        tm.append(peaks.peak_temperatures(T, S)[0])
    res = peaks.fit_arrhenius(tm, windows, gamma=gamma)
    assert abs(res["Ea_eV"] - Ea) < 0.01
    assert 0.5 < res["sigma_cm2"] / sigma < 2.0


def test_fit_curve_and_tool(tmp_path):
    path, _ = _two_peak_csv(tmp_path)
    from defective.tools import Toolbox
    tb = Toolbox.__new__(Toolbox)
    tb.corpus, tb.fns, tb.specs = None, {}, []
    tb._register()
    out, is_err = tb.call("fit_trap_curve", {"csv_path": str(path), "technique": "TSC",
                                             "heating_rate_K_per_s": BETA})
    assert not is_err and '"n_peaks": 2' in out
    arr = tmp_path / "arr.csv"
    T = np.linspace(200, 280, 6)
    g = peaks.gamma_from_mass(1.0)
    e = g * 1e-14 * T**2 * np.exp(-0.5 / (K_B * T))
    arr.write_text("1000/T,ln(e/T^2)\n" + "\n".join(f"{1000/t},{math.log(r / t**2)}" for t, r in zip(T, e)))
    res = peaks.fit_curve(arr, "DLTS", x_kind="1000/T", y_kind="ln(e/T^2)", m_eff=1.0)
    assert abs(res["Ea_eV"] - 0.5) < 1e-6 and math.isclose(res["sigma_cm2"], 1e-14, rel_tol=1e-4)
    with pytest.raises(ValueError):
        peaks.fit_curve(arr, "XPS")
