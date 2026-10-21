"""Trap -> candidate-defect matching, and next-experiment selection.

Matcher: Gaussian likelihood on the energy difference between a measured trap
and each DFT transition level referenced to the same band edge, with width
sqrt(sigma_exp^2 + sigma_dft^2). Flat prior; the posterior is normalized over
the candidates that survive the reference check.

Design: an experiment that reads off one attribute of the defect (its
favored chemical-potential condition, donor/acceptor character, ...) partitions
the candidates by that attribute. Its expected information gain is the entropy
of the posterior mass over those partitions.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/extracted"

# Used only when a record carries no harmonized uncertainty of its own.
DEFAULT_SIGMA_DFT_EV = 0.15
DEFAULT_SIGMA_EXP_EV = 0.05


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dft_levels(material: str | None = None, path: Path = DATA / "dft_levels.jsonl") -> list[dict]:
    recs = load_jsonl(path)
    if material:
        m = material.lower().replace(" ", "")
        recs = [r for r in recs if r["material"].lower().replace(" ", "") == m]
    return recs


def _to_reference(rec: dict, ref: str) -> float | None:
    if rec["level_reference"] == ref:
        return rec["level_eV"]
    gap = (rec.get("method") or {}).get("band_gap_eV")
    return None if gap is None else gap - rec["level_eV"]


def match(trap: dict, levels: list[dict]) -> dict:
    """trap: {energy_eV, energy_reference ('Ec'|'Ev'|'unspecified'), energy_err_eV?}."""
    e, ref = trap["energy_eV"], trap.get("energy_reference", "unspecified")
    s_exp = trap.get("energy_err_eV") or DEFAULT_SIGMA_EXP_EV
    cands, skipped = [], []
    for rec in levels:
        refs = ["Ec", "Ev"] if ref == "unspecified" else [ref]
        best = None
        for r in refs:
            lv = _to_reference(rec, r)
            if lv is None:
                continue
            sigma = math.hypot(s_exp, rec.get("uncertainty_eV") or DEFAULT_SIGMA_DFT_EV)
            ll = -0.5 * ((e - lv) / sigma) ** 2 - math.log(sigma)
            if best is None or ll > best[0]:
                best = (ll, r, lv, sigma)
        if best is None:
            skipped.append(rec["record_id"])
            continue
        ll, r, lv, sigma = best
        cands.append({"record_id": rec["record_id"], "defect": rec["defect"],
                      "transition": rec["transition"], "level_eV": round(lv, 3), "reference": r,
                      "delta_eV": round(e - lv, 3), "sigma_eV": round(sigma, 3), "_ll": ll,
                      "source": rec.get("source", {}).get("doc_id")})
    if cands:
        top = max(c["_ll"] for c in cands)
        z = sum(math.exp(c["_ll"] - top) for c in cands)
        for c in cands:
            c["posterior"] = round(math.exp(c.pop("_ll") - top) / z, 4)
        cands.sort(key=lambda c: -c["posterior"])
    return {"candidates": cands, "skipped_no_common_reference": skipped,
            "note": "Flat prior; posterior is relative to the DFT records loaded, not absolute."}


def nearest_energy(trap: dict, levels: list[dict]) -> dict | None:
    """Baseline condition: the single closest level, no uncertainty model."""
    ranked = match(trap, levels)["candidates"]
    return min(ranked, key=lambda c: abs(c["delta_eV"])) if ranked else None


def _entropy(ps) -> float:
    return -sum(p * math.log2(p) for p in ps if p > 0)


def propose_experiments(candidates: list[dict], levels: list[dict],
                        config: Path = REPO / "configs/experiments.yaml") -> list[dict]:
    """Rank experiments by expected information gain (bits) over the candidate posterior."""
    exps = yaml.safe_load(config.read_text())["experiments"]
    by_id = {r["record_id"]: r for r in levels}
    h0 = _entropy(c["posterior"] for c in candidates)
    out = []
    for name, exp in exps.items():
        attr = exp["discriminates_on"]
        groups: dict[str, float] = defaultdict(float)
        members: dict[str, list[str]] = defaultdict(list)
        unknown = 0.0
        for c in candidates:
            val = by_id.get(c["record_id"], {}).get(attr)
            if val in (None, "", "unknown"):
                unknown += c["posterior"]
                val = "unknown"
            groups[str(val)] += c["posterior"]
            members[str(val)].append(f'{c["defect"]}{c["transition"]}')
        out.append({"experiment": name, "description": exp["description"], "cost": exp.get("cost"),
                    "discriminates_on": attr, "expected_info_gain_bits": round(_entropy(groups.values()), 3),
                    "posterior_entropy_bits": round(h0, 3),
                    "mass_with_unknown_attribute": round(unknown, 3),
                    "outcome_groups": dict(members)})
    out.sort(key=lambda x: -x["expected_info_gain_bits"])
    return out
