"""Held-out evaluation: the four comparison conditions from the README.

    nearest   nearest-energy heuristic          (matcher.nearest_energy, then |dE| order)
    llm       LLM alone, no retrieval           (blind trap description only)
    llm_dft   LLM with the retrieved DFT table  (blind trap + matcher.dft_levels(material))
    matcher   probabilistic matcher             (matcher.match)

Every condition sees the same blind view of a trap (`blind_view`): material,
energy (+error, band-edge reference), cross-section, technique and growth. The
authors' assignment, their trap label and the source quote are never shown,
because any of them can carry the answer.

Only records whose `assignment.basis == "independent"` are scored. Agreement
with an assignment that merely cites DFT or the literature would be circular.
Every excluded record is listed with its reason.

Predictions are compared per *defect*, not per transition level. Labels are
reduced to one key by `normalize_defect`, so V_Br, VBr, V$_{Br}$ and "bromine
vacancy" all match. When the matcher returns several levels of one defect, the
defect ranks at its best level and its posterior is the sum over its levels.

Run:  defective eval [--conditions nearest matcher] [--gold data/eval/heldout_traps.jsonl]
Writes runs/eval/<timestamp>/{predictions,summary,excluded}.csv, plus run.json and a metrics figure.
"""
from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path

from . import matcher

REPO = Path(__file__).resolve().parents[2]
DEFAULT_GOLD = REPO / "data/eval/heldout_traps.jsonl"
DEFAULT_DFT = REPO / "data/extracted/dft_levels.jsonl"
RUNS = REPO / "runs/eval"

CONDITIONS = {
    "nearest": "Nearest-energy heuristic",
    "llm": "LLM alone (no retrieval)",
    "llm_dft": "LLM + retrieved DFT table",
    "matcher": "Probabilistic matcher",
}
LLM_CONDITIONS = {"llm", "llm_dft"}
MAX_LLM_LABELS = 5

# ---------------------------------------------------------------- defect labels

# Elements that occur in detector crystals, their dopants and common impurities.
ELEMENT_NAMES = {
    "bromine": "Br", "cadmium": "Cd", "tellurium": "Te", "zinc": "Zn", "selenium": "Se",
    "lead": "Pb", "cesium": "Cs", "caesium": "Cs", "iodine": "I", "chlorine": "Cl",
    "indium": "In", "gallium": "Ga", "aluminum": "Al", "aluminium": "Al", "copper": "Cu",
    "silver": "Ag", "gold": "Au", "sodium": "Na", "lithium": "Li", "potassium": "K",
    "oxygen": "O", "sulfur": "S", "sulphur": "S", "germanium": "Ge", "tin": "Sn",
    "bismuth": "Bi", "iron": "Fe", "nickel": "Ni", "chromium": "Cr", "titanium": "Ti",
    "hydrogen": "H", "nitrogen": "N", "phosphorus": "P", "arsenic": "As", "antimony": "Sb",
    "fluorine": "F", "manganese": "Mn", "cobalt": "Co", "magnesium": "Mg", "calcium": "Ca",
    "rubidium": "Rb", "thallium": "Tl", "mercury": "Hg", "vanadium": "V",
}
SYMBOLS = set(ELEMENT_NAMES.values())
_NAME_RE = "|".join(sorted(ELEMENT_NAMES, key=len, reverse=True))

# Charge annotations that follow a defect: (2-/-), (+/0), ^{2-}, ^+, Kröger-Vink dots/primes.
_TRANSITION = re.compile(r"\(\s*[0-9]*[+\-−0]?\s*/\s*[0-9]*[+\-−0]?\s*\)")
_SUPERSCRIPT = re.compile(r"\^\s*(\{[^}]*\}|[0-9]*[+\-−•'×xX0]+)")
_CHARGE_CHARS = re.compile(r"[•·'′″×⁺⁻⁰¹²³]")


def _words_to_symbols(s: str) -> str:
    """'bromine vacancy' -> V_Br; 'cadmium interstitial' -> Cd_i; 'Te on Cd site' -> Te_Cd."""
    el = rf"({_NAME_RE}|[A-Z][a-z]?)"
    sym = lambda w: ELEMENT_NAMES.get(w.lower(), w)  # noqa: E731
    s = re.sub(rf"\b{el}[\s-]+vacanc(?:y|ies)\b", lambda m: f"V_{sym(m[1])}", s, flags=re.I)
    s = re.sub(rf"\bvacanc(?:y|ies)\s+(?:of|on|at)\s+(?:the\s+)?{el}(?:\s+site)?\b",
               lambda m: f"V_{sym(m[1])}", s, flags=re.I)
    s = re.sub(rf"\b{el}[\s-]+interstitials?\b", lambda m: f"{sym(m[1])}_i", s, flags=re.I)
    s = re.sub(rf"\binterstitial\s+{el}\b", lambda m: f"{sym(m[1])}_i", s, flags=re.I)
    s = re.sub(rf"\b{el}[\s-]+on[\s-]+(?:an?\s+|the\s+)?{el}[\s-]*(?:site|antisite|substitution)?\b",
               lambda m: f"{sym(m[1])}_{sym(m[2])}", s, flags=re.I)
    return s


def _tokens(part: str) -> list[str] | None:
    """Split 'V_Br', 'VBr', 'TeCd', 'Pbi', 'v_br' into [site-occupant, site]; None if not a point defect."""
    if "_" in part:
        toks = [t for t in part.split("_") if t]
    else:
        toks = re.findall(r"[A-Z][a-z]?|i$", part)
        if "".join(toks) != part:
            return None
    if len(toks) != 2:
        return None
    out = []
    for t in toks:
        if t == "i" and out:                 # lowercase i in the site slot = interstitial; I = iodine
            out.append("i")
            continue
        cap = t[:1].upper() + t[1:].lower()
        if cap == "V" or cap in SYMBOLS:
            out.append(cap)
        else:
            return None
    return out


def normalize_defect(label: str | None) -> str:
    """Canonical key for a defect label: 'V_Br', 'Te_Cd', 'Pb_i', complexes as 'Cl_Te+V_Cd'.

    Strips LaTeX, charge states and transition annotations, and maps element
    names and 'vacancy'/'interstitial'/'X on Y' phrasing to symbols. Labels that
    are not point-defect notation (e.g. 'EL2', 'A-center') come back lowercased
    and whitespace-collapsed, so they still compare consistently.
    """
    if not label:
        return ""
    s = str(label).strip()
    s = s.replace("$", "").replace("−", "-")
    s = re.sub(r"\\(?:mathrm|text|rm)\b\s*", "", s)
    s = _TRANSITION.sub("", s)
    s = _SUPERSCRIPT.sub("", s)
    s = _CHARGE_CHARS.sub("", s)
    s = re.sub(r"_\s*\{\s*([^}]*)\s*\}", r"_\1", s)          # V_{Br} -> V_Br
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s*\((?:[0-9]*[+\-0]|neutral)\)\s*$", "", s)  # trailing (2-), (0)
    s = re.sub(r"(?<=[A-Za-z])[0-9]*[+\-]+\s*$", "", s)         # bare trailing charge: V_Cd2-, Cd_i+
    s = _words_to_symbols(s)
    s = re.sub(r"\b(?:defects?|complex(?:es)?|centers?|centres?|related)\b", " ", s, flags=re.I)
    parts = [p for p in re.split(r"\s*(?:\+|–|—|-|&|\band\b|\s)\s*", s.strip("()[] ")) if p]
    keys = []
    for p in parts:
        toks = _tokens(p.strip("()[] ,;"))
        if toks is None:
            return re.sub(r"\s+", " ", str(label).strip().lower())
        keys.append("_".join(toks))
    return "+".join(sorted(set(keys))) if keys else re.sub(r"\s+", " ", str(label).strip().lower())


def gold_keys(defect: str | None) -> set[str]:
    """An assignment may name alternatives ('V_Br or Pb_i'); any of them counts as correct."""
    if not defect:
        return set()
    alts = re.split(r"\s+or\s+|\s*;\s*|\s*,\s*", str(defect))
    return {k for k in (normalize_defect(a) for a in alts) if k}


# ---------------------------------------------------------------- gold records

BLIND_FIELDS = ("material", "energy_eV", "energy_err_eV", "energy_reference",
                "cross_section_cm2", "technique", "growth")


def blind_view(rec: dict) -> dict:
    """What every condition is allowed to see. No assignment, label, or source quote."""
    return {k: rec.get(k) for k in BLIND_FIELDS if rec.get(k) is not None}


def split_gold(records: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    """(scored, excluded). Excluded rows carry record_id and a reason."""
    scored, excluded = [], []
    for r in records:
        a = r.get("assignment") or {}
        basis = a.get("basis") or "missing"
        if basis != "independent":
            why = {"dft_cited": "basis=dft_cited (assignment cites DFT: circular)",
                   "literature_cited": "basis=literature_cited (assignment cites other papers: circular)",
                   "none": "basis=none (authors made no assignment)",
                   "missing": "no assignment.basis"}.get(basis, f"basis={basis}")
            excluded.append({"record_id": r.get("record_id"), "reason": why})
        elif not gold_keys(a.get("defect")):
            excluded.append({"record_id": r.get("record_id"), "reason": "basis=independent but no defect label"})
        else:
            scored.append(r)
    return scored, excluded


# ---------------------------------------------------------------- conditions

def _rank_levels(cands: list[dict], weight: str | None) -> tuple[list[str], dict[str, float]]:
    """Collapse level candidates (already in rank order) to unique defect keys."""
    ranked, mass = [], {}
    for c in cands:
        k = normalize_defect(c["defect"])
        if k not in mass:
            ranked.append(k)
            mass[k] = 0.0
        if weight:
            mass[k] += c.get(weight, 0.0)
    return ranked, mass


def run_nearest(trap: dict, levels: list[dict]) -> dict:
    best = matcher.nearest_energy(trap, levels)
    cands = sorted(matcher.match(trap, levels)["candidates"], key=lambda c: abs(c["delta_eV"]))
    if best is not None:                      # same choice as the baseline, then |dE| order
        cands.remove(next(c for c in cands if c["record_id"] == best["record_id"]))
        cands.insert(0, best)
    ranked, _ = _rank_levels(cands, None)
    return {"ranked": ranked, "raw": [f'{c["defect"]}{c["transition"]}' for c in cands[:MAX_LLM_LABELS]]}


def run_matcher(trap: dict, levels: list[dict]) -> dict:
    cands = matcher.match(trap, levels)["candidates"]
    ranked, post = _rank_levels(cands, "posterior")
    return {"ranked": ranked, "posterior": post,
            "raw": [f'{c["defect"]}{c["transition"]}:{c["posterior"]}' for c in cands[:MAX_LLM_LABELS]]}


LLM_SYSTEM = ("You are an expert in point defects and deep-level spectroscopy of semiconductor "
              "radiation-detector crystals. Answer only with the JSON requested.")

LLM_PROMPT = """A deep-level trap was measured in an experiment. The authors' own assignment is withheld.

TRAP:
{trap}
{table}
Which point defects (vacancies, interstitials, antisites, substitutionals, or complexes) most likely
cause this trap? Return ONLY a JSON array of up to {k} defect labels, most likely first, written in
Kroger-Vink-style site notation without charge states, e.g. ["V_Br", "Pb_i", "Te_Cd", "Cl_Te+V_Cd"]."""

DFT_FIELDS = ("defect", "transition", "level_eV", "level_reference", "uncertainty_eV",
              "character", "favored_condition")


def dft_table(levels: list[dict]) -> str:
    if not levels:
        return "\nDFT TABLE: no computed transition levels are available for this material.\n"
    rows = []
    for r in levels:
        row = {k: r.get(k) for k in DFT_FIELDS if r.get(k) is not None}
        m = r.get("method") or {}
        row |= {k: m[k] for k in ("functional", "band_gap_eV") if m.get(k) is not None}
        rows.append(json.dumps(row, ensure_ascii=False))
    return ("\nDFT TABLE (computed charge-transition levels for this material, one JSON object per line):\n"
            + "\n".join(rows) + "\n")


def parse_labels(text: str) -> list[str]:
    """Pull a ranked label list out of an LLM reply (tolerates prose, code fences, list of objects)."""
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        raise ValueError("no JSON array in reply")
    items = json.loads(m.group(0))
    out = []
    for it in items:
        if isinstance(it, dict):
            it = it.get("defect") or it.get("label") or it.get("name")
        if isinstance(it, str) and it.strip():
            out.append(it.strip())
    return out


def run_llm(trap: dict, levels: list[dict] | None, make_llm: Callable[[str], object]) -> dict:
    """levels=None -> no retrieval. make_llm(system) returns a fresh provider (no tools, no history)."""
    prompt = LLM_PROMPT.format(trap=json.dumps(trap, ensure_ascii=False, indent=1),
                               table="" if levels is None else dft_table(levels), k=MAX_LLM_LABELS)
    reply = make_llm(LLM_SYSTEM).send_user(prompt).text
    labels = parse_labels(reply)[:MAX_LLM_LABELS]
    ranked = []
    for lab in labels:
        k = normalize_defect(lab)
        if k and k not in ranked:
            ranked.append(k)
    return {"ranked": ranked, "raw": labels}


# ---------------------------------------------------------------- scoring

def rank_of(ranked: list[str], gold: set[str]) -> int | None:
    return next((i + 1 for i, k in enumerate(ranked) if k in gold), None)


def summarize(rows: list[dict], conditions: list[str]) -> list[dict]:
    out = []
    for cond in conditions:
        rs = [r for r in rows if r["condition"] == cond]
        n = len(rs)
        ranks = [r["gold_rank"] for r in rs]
        s = {"condition": cond, "description": CONDITIONS[cond], "n_scored": n,
             "n_errors": sum(1 for r in rs if r["error"]),
             "n_no_dft_levels": sum(1 for r in rs if r["n_dft_levels"] == 0),
             "top1": sum(1 for k in ranks if k == 1) / n if n else None,
             "top3": sum(1 for k in ranks if k and k <= 3) / n if n else None,
             "mrr": sum(1 / k for k in ranks if k) / n if n else None,
             "mean_gold_posterior": None, "brier_top1": None}
        if cond == "matcher" and n:
            s["mean_gold_posterior"] = sum(r["gold_posterior"] for r in rs) / n
            s["brier_top1"] = sum((r["top1_posterior"] - (r["gold_rank"] == 1)) ** 2 for r in rs) / n
        out.append(s)
    return out


def evaluate(gold: list[dict], conditions: list[str] | None = None,
             levels_for: Callable[[str], list[dict]] | None = None,
             make_llm: Callable[[str], object] | None = None) -> dict:
    """Score gold records under each condition. Pure (no I/O beyond the callbacks)."""
    conditions = list(conditions or CONDITIONS)
    if unknown := set(conditions) - set(CONDITIONS):
        raise ValueError(f"unknown condition(s) {sorted(unknown)}; choose from {list(CONDITIONS)}")
    if LLM_CONDITIONS & set(conditions) and make_llm is None:
        raise ValueError("LLM conditions need make_llm")
    levels_for = levels_for or (lambda m: matcher.dft_levels(m))
    scored, excluded = split_gold(gold)
    rows = []
    for rec in scored:
        trap = blind_view(rec)
        gold_set = gold_keys(rec["assignment"]["defect"])
        levels = levels_for(rec["material"])
        for cond in conditions:
            err = ""
            try:
                if cond == "nearest":
                    res = run_nearest(trap, levels)
                elif cond == "matcher":
                    res = run_matcher(trap, levels)
                else:
                    res = run_llm(trap, levels if cond == "llm_dft" else None, make_llm)
            except Exception as e:  # one bad reply must not sink the run; it scores as a miss
                res, err = {"ranked": [], "raw": []}, f"{type(e).__name__}: {e}"
            post = res.get("posterior") or {}
            ranked = res["ranked"]
            rows.append({
                "record_id": rec.get("record_id"), "material": rec["material"],
                "energy_eV": rec["energy_eV"], "energy_reference": rec.get("energy_reference"),
                "condition": cond, "gold_defect": rec["assignment"]["defect"],
                "gold_keys": "|".join(sorted(gold_set)), "predicted_ranked": "|".join(ranked),
                "raw_output": json.dumps(res["raw"], ensure_ascii=False),
                "gold_rank": rank_of(ranked, gold_set),
                "gold_posterior": sum(post.get(k, 0.0) for k in gold_set) if cond == "matcher" else None,
                "top1_posterior": post.get(ranked[0], 0.0) if (cond == "matcher" and ranked) else
                                  (0.0 if cond == "matcher" else None),
                "n_dft_levels": len(levels), "error": err,
            })
    return {"rows": rows, "summary": summarize(rows, conditions), "excluded": excluded,
            "n_total": len(gold), "n_scored": len(scored)}


# ---------------------------------------------------------------- I/O

def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def format_summary(result: dict) -> str:
    lines = [f"Gold records: {result['n_total']}   scored (basis=independent): {result['n_scored']}"
             f"   excluded: {len(result['excluded'])}"]
    reasons: dict[str, int] = {}
    for e in result["excluded"]:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    lines += [f"  excluded {n:3d}  {why}" for why, n in sorted(reasons.items(), key=lambda x: -x[1])]
    f = lambda v: "   -  " if v is None else f"{v:6.3f}"  # noqa: E731
    lines += ["", f"{'condition':28s} {'n':>4s} {'top1':>6s} {'top3':>6s} {'MRR':>6s}"
                  f" {'P(gold)':>7s} {'Brier1':>6s} {'errors':>6s}"]
    for s in result["summary"]:
        lines.append(f"{s['description']:28s} {s['n_scored']:4d} {f(s['top1'])} {f(s['top3'])} {f(s['mrr'])}"
                     f"  {f(s['mean_gold_posterior'])} {f(s['brier_top1'])} {s['n_errors']:6d}")
    return "\n".join(lines)


def write_run(result: dict, out_dir: Path, meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "predictions.csv", result["rows"],
               None if result["rows"] else ["record_id", "condition", "gold_rank"])
    _write_csv(out_dir / "summary.csv", result["summary"])
    _write_csv(out_dir / "excluded.csv", result["excluded"], ["record_id", "reason"])
    (out_dir / "run.json").write_text(json.dumps(
        meta | {"n_total": result["n_total"], "n_scored": result["n_scored"],
                "n_excluded": len(result["excluded"])}, indent=2))
    return out_dir


def plot_summary(summary_csv: Path) -> list[Path]:
    """Grouped bars of top-1 / top-3 / MRR per condition, read from summary.csv. Writes pdf+png+svg."""
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = os.path.expanduser("~/.claude/figure-style.mplstyle")
    if os.path.exists(style):
        plt.style.use(style)
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42

    with summary_csv.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["top1"] not in ("", None)]
    if not rows:
        return []
    metrics = [("top1", "Top-1 accuracy"), ("top3", "Top-3 accuracy"), ("mrr", "Mean reciprocal rank")]
    colors = {"nearest": "#2a78d6", "llm": "#eb6834", "llm_dft": "#1baf7a", "matcher": "#eda100"}
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    w = 0.8 / len(rows)
    for j, r in enumerate(rows):
        xs = [i - 0.4 + w * (j + 0.5) for i in range(len(metrics))]
        vals = [float(r[m]) for m, _ in metrics]
        ax.bar(xs, vals, width=w * 0.92, color=colors.get(r["condition"], "#888888"),
               label=f'{r["description"]}', zorder=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color="#52514e")
    ax.set_xticks(range(len(metrics)), [lab for _, lab in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.yaxis.grid(True, color="#e4e3df", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    n = {r["n_scored"] for r in rows}
    ax.set_title(f"Held-out trap assignment (n = {'/'.join(sorted(n))} independent records)", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    base = summary_csv.parent / "metrics"
    paths = [base.with_suffix(".pdf"), base.with_suffix(".png"), base.with_suffix(".svg")]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    fig.savefig(paths[2], format="svg")
    plt.close(fig)
    return paths


def main(gold_path: Path = DEFAULT_GOLD, conditions: list[str] | None = None,
         provider: str | None = None, model: str | None = None, dft_path: Path = DEFAULT_DFT,
         out_dir: Path | None = None, limit: int | None = None, plot: bool = True) -> Path:
    gold_path, dft_path = Path(gold_path), Path(dft_path)
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path} (JSONL of trap records per "
                                "schemas/trap.schema.json, with assignment.basis filled in)")
    conditions = list(conditions or CONDITIONS)
    gold = matcher.load_jsonl(gold_path)[:limit]
    make_llm, llm_id = None, None
    if LLM_CONDITIONS & set(conditions):
        from .providers import make_provider, resolve
        cfg = resolve(provider, model)        # fail fast on a missing key, before any scoring
        llm_id = f"{cfg['provider']}/{cfg['model']}"
        make_llm = lambda system: make_provider(provider, system, [], model)  # noqa: E731
    result = evaluate(gold, conditions, lambda m: matcher.dft_levels(m, dft_path), make_llm)
    out_dir = Path(out_dir or RUNS / datetime.now().strftime("%Y%m%d-%H%M%S"))
    write_run(result, out_dir, {"gold": str(gold_path), "dft_levels": str(dft_path),
                                "conditions": conditions, "llm": llm_id, "limit": limit,
                                "created": datetime.now().isoformat(timespec="seconds")})
    print(format_summary(result))
    if plot and result["n_scored"]:
        for p in plot_summary(out_dir / "summary.csv"):
            print(f"  figure {p}")
    print(f"\nWrote {out_dir}")
    return out_dir
