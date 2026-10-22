"""Final CSV report: one row per (peak, candidate defect), with the peak-fit
parameters, the candidate defect's energy/source/posterior, and (when it was
run) the MACE formation-energy verification -- reported alongside the
posterior as supporting evidence, never blended into it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO / "runs" / "reports"

PEAK_FIELDS = ["peak", "energy_eV", "energy_sigma_eV", "peak_T_K", "frequency_factor_per_s",
               "apparent_cross_section_cm2", "order_b", "flags"]
CANDIDATE_FIELDS = ["rank", "defect", "transition", "level_eV", "reference", "delta_eV", "sigma_eV",
                    "posterior", "source"]
ML_FIELDS = ["ml_status", "ml_formation_energy_eV"]
REPORT_COLUMNS = ["material", "technique"] + [f"peak_{f}" for f in PEAK_FIELDS] + CANDIDATE_FIELDS + ML_FIELDS


def build_report(material: str, technique: str, peaks: list[dict], candidates_by_peak: dict,
                  ml_results: dict | None = None) -> pd.DataFrame:
    """One row per peak x candidate defect.

    `peaks`: PEAK_COLS-shaped rows (as produced by ui.fit_table / peaks.fit_curve),
    each with a 1-based "peak" number.
    `candidates_by_peak`: {peak_number: [CAND_COLS-shaped rows, as produced by
    matcher.match]} -- a peak with no candidates still gets one row, with the
    candidate columns left blank.
    `ml_results`: optional {(peak_number, defect): verify_defect_formation() result}.
    """
    ml_results = ml_results or {}
    rows = []
    for peak in peaks:
        pnum = peak.get("peak")
        cands = candidates_by_peak.get(pnum) or [{}]
        for cand in cands:
            row = {"material": material, "technique": technique}
            row.update({f"peak_{f}": peak.get(f) for f in PEAK_FIELDS})
            row.update({f: cand.get(f) for f in CANDIDATE_FIELDS})
            ml = ml_results.get((pnum, cand.get("defect")), {})
            row["ml_status"] = ml.get("status")
            row["ml_formation_energy_eV"] = ml.get("formation_energy_eV")
            rows.append(row)
    return pd.DataFrame(rows, columns=REPORT_COLUMNS)


def write_report(df: pd.DataFrame, reports_dir: Path | None = None) -> Path:
    """Write `df` to runs/reports/<UTC timestamp>.csv and return the path."""
    reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = reports_dir / f"{stamp}.csv"
    df.to_csv(path, index=False)
    return path
