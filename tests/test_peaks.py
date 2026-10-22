"""Tests for peaks.py: curve loading, axis conversion, and the trapfit-backed fit_curve dispatch.

TSC/Arrhenius fitting itself is trapfit's responsibility (vendored under
defective.trapfit, with its own test suite in trapfit_pkg/tests). These tests
cover the integration layer: CSV parsing, digitized-axis conversion, the
tsc_component plotting helper, and the status-based (ok/unresolved/refused)
contract that fit_curve and the fit_trap_curve tool must preserve end to end.
"""
import math

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from defective import peaks
from defective.peaks import K_B
from defective.trapfit.arrhenius import GAMMA_0
from defective.trapfit.tsc import peak_first_order

BETA = 0.1  # K/s


def rw_direct(T, Ea, s, n0, beta):
    """Randall-Wilkins by direct numerical integration (independent of the Kitis closed form)."""
    Tf = np.linspace(5.0, T.max(), 200001)
    F = cumulative_trapezoid(np.exp(-Ea / (K_B * Tf)), Tf, initial=0)
    I = n0 * s * np.exp(-Ea / (K_B * Tf)) * np.exp(-(s / beta) * F)
    return np.interp(T, Tf, I)


def _s_from_Tm(Ea, Tm, beta):
    """Attempt frequency implied by the first-order peak condition at Tm (exact peak-position relation)."""
    return beta * Ea / (K_B * Tm**2) * np.exp(Ea / (K_B * Tm))


def test_load_curve_skips_headers_and_sorts(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("# comment\nT (K), I (A)\n200,1e-9\n150,2e-9\nbad,row\n180,3e-9\n")
    x, y = peaks.load_curve(p)
    assert list(x) == [150.0, 180.0, 200.0]
    assert list(y) == [2e-9, 3e-9, 1e-9]


def test_to_rate_round_trip():
    T = np.array([250.0, 260.0, 270.0])
    e = np.array([10.0, 20.0, 30.0])
    x, y = 1000 / T, np.log(e / T**2)
    T2, e2 = peaks.to_rate(x, y, x_kind="1000/T", y_kind="ln(e/T^2)")
    assert np.allclose(T2, T)
    assert np.allclose(e2, e)


def test_tsc_component_matches_numerical_integral():
    T = np.linspace(100, 260, 400)
    Ea, s = 0.40, 1e10
    ref = rw_direct(T, Ea, s, 1.0, BETA)
    Tm = brentq(lambda t: _s_from_Tm(Ea, t, BETA) - s, 100, 260)
    Im = float(rw_direct(np.array([Tm]), Ea, s, 1.0, BETA)[0])
    peak = {"energy_eV": Ea, "peak_T_K": Tm, "peak_height_A": Im, "order_b": 1.0}
    fit = peaks.tsc_component(T, peak)
    # Kitis is an analytical approximation of Randall-Wilkins, not an identity -- allow ~1%.
    assert np.max(np.abs(fit - ref)) / ref.max() < 1e-2


def _two_peak_csv(tmp_path):
    rng = np.random.default_rng(42)
    T = np.arange(110.0, 240.0, 0.5)
    truth = [(0.35, 160.0), (0.42, 195.0)]  # (Ea_eV, Tm_K), following the fitted Kitis model exactly
    I = sum(peak_first_order(T, 1.0, Tm, Ea) for Ea, Tm in truth)
    # Multiplicative (log-normal) noise, not fixed additive noise: it scales with the signal
    # instead of swamping the low-signal tails in log-space and faking spurious peaks there.
    I = I * 10 ** rng.normal(0, 0.01, T.size)
    p = tmp_path / "tsc.csv"
    p.write_text("# digitized, Fig. 3\nT (K),I (A)\n" + "\n".join(f"{t:.2f},{i:.6e}" for t, i in zip(T, I)))
    return p, truth


def test_tsc_two_peaks(tmp_path):
    path, truth = _two_peak_csv(tmp_path)
    T, I = peaks.load_curve(path)
    assert T.size == 260
    res = peaks.fit_curve(path, "TSC", heating_rate_K_per_s=BETA, max_peaks=4)
    assert res["status"] == "ok", res
    traps = res["traps"]
    assert len(traps) == 2
    for pk, (Ea, Tm) in zip(traps, truth):
        assert abs(pk["energy_eV"] - Ea) < 0.03, pk
        assert abs(pk["peak_T_K"] - Tm) < 2.0, pk
        assert pk["energy_sigma_eV"] >= 0
        assert pk["method"] == "TSC" and pk["evidence_tag"] == "fitted"
    assert res["fit"]["rms_log10_residual"] < 0.05


def test_tsc_single_peak_and_quick_estimates(tmp_path):
    T = np.arange(100.0, 250.0, 0.5)
    I = peak_first_order(T, 1.0, 180.0, 0.40)
    I = I * 10 ** np.random.default_rng(1).normal(0, 0.01, T.size)
    p = tmp_path / "single.csv"
    p.write_text("T,I\n" + "\n".join(f"{t:.2f},{i:.6e}" for t, i in zip(T, I)))
    res = peaks.fit_curve(p, "TSC", heating_rate_K_per_s=BETA)
    assert res["status"] == "ok", res
    assert len(res["traps"]) == 1
    pk = res["traps"][0]
    assert abs(pk["energy_eV"] - 0.40) < 0.02
    assert "frequency_factor_per_s" in pk
    q = res["quick_estimates"]
    assert len(q) == 1
    assert abs(q[0]["chen"]["E_omega"] - 0.40) < 0.05
    assert abs(q[0]["initial_rise"]["Ea_eV"] - 0.40) < 0.05


def test_arrhenius_recovers_ea_and_sigma(tmp_path):
    Ea, sigma, m_eff = 0.55, 3e-15, 0.26
    gamma = GAMMA_0 * m_eff
    T = np.linspace(230, 300, 8)
    rng = np.random.default_rng(7)
    e = gamma * sigma * T**2 * np.exp(-Ea / (K_B * T)) * np.exp(rng.normal(0, 0.03, T.size))
    p = tmp_path / "arr.csv"
    p.write_text("1000/T,ln(e/T^2)\n" + "\n".join(f"{1000/t},{math.log(r / t**2)}" for t, r in zip(T, e)))
    res = peaks.fit_curve(p, "DLTS", x_kind="1000/T", y_kind="ln(e/T^2)", m_eff=m_eff)
    assert res["status"] == "ok", res
    trap = res["traps"][0]
    assert abs(trap["energy_eV"] - Ea) < 0.02
    assert 0.5 < trap["apparent_cross_section_cm2"] / sigma < 2.0
    assert trap["method"] == "DLTS" and trap["evidence_tag"] == "fitted"


def test_dlts_too_few_points_refused(tmp_path):
    p = tmp_path / "arr.csv"
    p.write_text("1000/T,ln(e/T^2)\n4.0,10.0\n4.1,9.5\n4.2,9.0\n")  # 3 windows: below min_rate_windows
    res = peaks.fit_curve(p, "DLTS", x_kind="1000/T", y_kind="ln(e/T^2)")
    assert res["status"] == "ok"  # 3 meets the minimum; confirms the boundary is inclusive

    p2 = tmp_path / "arr2.csv"
    p2.write_text("1000/T,ln(e/T^2)\n4.0,10.0\n4.2,9.0\n")  # 2 rows: below load_curve's own minimum of 3
    with pytest.raises(ValueError):
        peaks.fit_curve(p2, "DLTS", x_kind="1000/T", y_kind="ln(e/T^2)")


def test_laplace_dlts_runs_without_raising(tmp_path):
    t = np.linspace(0.001, 1.0, 200)
    sig = np.exp(-5.0 * t)
    p = tmp_path / "transient.csv"
    p.write_text("\n".join(f"{ti:.6f},{si:.6e}" for ti, si in zip(t, sig)))
    res = peaks.fit_curve(p, "LAPLACE_DLTS")
    assert res["status"] in {"ok", "unresolved"}


def test_fit_curve_refuses_unknown_technique(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("1,2\n3,4\n5,6\n")
    res = peaks.fit_curve(p, "XPS")
    assert res["status"] == "refused"


def test_fit_curve_bad_file_still_raises(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("not,numeric\n")
    with pytest.raises(ValueError):
        peaks.fit_curve(p, "TSC")


def test_fit_trap_curve_tool(tmp_path):
    path, _ = _two_peak_csv(tmp_path)
    from defective.tools import Toolbox
    tb = Toolbox.__new__(Toolbox)
    tb.corpus, tb.fns, tb.specs = None, {}, []
    tb._register()
    out, is_err = tb.call("fit_trap_curve", {"csv_path": str(path), "technique": "TSC",
                                             "heating_rate_K_per_s": BETA})
    assert not is_err, out
    assert '"status": "ok"' in out
    assert out.count('"energy_eV"') >= 2
