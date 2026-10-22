import pandas as pd

from defective import report

PEAKS = [
    {"peak": 1, "energy_eV": 0.45, "energy_sigma_eV": 0.02, "peak_T_K": 210.0,
     "frequency_factor_per_s": 1e9, "apparent_cross_section_cm2": None, "order_b": 2, "flags": ""},
    {"peak": 2, "energy_eV": 0.80, "energy_sigma_eV": 0.03, "peak_T_K": 260.0,
     "frequency_factor_per_s": 5e8, "apparent_cross_section_cm2": None, "order_b": 2, "flags": "broad"},
]
CANDIDATES_BY_PEAK = {
    1: [{"rank": 1, "defect": "V_Br", "transition": "(0/-)", "level_eV": 0.44, "reference": "Ec",
         "delta_eV": 0.01, "sigma_eV": 0.02, "posterior": 0.9, "source": "doc_a"},
        {"rank": 2, "defect": "V_Cs", "transition": "(+/0)", "level_eV": 0.50, "reference": "Ec",
         "delta_eV": 0.05, "sigma_eV": 0.02, "posterior": 0.1, "source": "doc_b"}],
    # peak 2 intentionally has no candidates
}


def test_build_report_row_per_peak_candidate():
    df = report.build_report("CsPbBr3", "TSC", PEAKS, CANDIDATES_BY_PEAK)
    assert list(df.columns) == report.REPORT_COLUMNS
    assert len(df) == 3  # 2 candidates for peak 1 + 1 blank row for peak 2
    assert set(df["material"]) == {"CsPbBr3"}
    assert set(df["technique"]) == {"TSC"}
    row1 = df[df["peak_peak"] == 1].iloc[0]
    assert row1["defect"] == "V_Br"
    assert row1["posterior"] == 0.9
    row2 = df[df["peak_peak"] == 2].iloc[0]
    assert pd.isna(row2["defect"])
    assert pd.isna(row2["ml_status"])


def test_build_report_includes_ml_results():
    ml = {(1, "V_Br"): {"status": "ok", "formation_energy_eV": 1.23}}
    df = report.build_report("CsPbBr3", "TSC", PEAKS, CANDIDATES_BY_PEAK, ml_results=ml)
    row = df[(df["peak_peak"] == 1) & (df["defect"] == "V_Br")].iloc[0]
    assert row["ml_status"] == "ok"
    assert row["ml_formation_energy_eV"] == 1.23


def test_write_report_creates_csv(tmp_path):
    df = report.build_report("CsPbBr3", "TSC", PEAKS, CANDIDATES_BY_PEAK)
    path = report.write_report(df, reports_dir=tmp_path)
    assert path.exists()
    assert path.parent == tmp_path
    reloaded = pd.read_csv(path)
    assert len(reloaded) == len(df)
