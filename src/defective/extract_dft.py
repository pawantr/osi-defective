"""LLM extraction of DFT charge-transition levels from the DFT papers.

Each record follows schemas/dft_level.schema.json and keeps the verbatim quote
the number came from. A record is kept only if

- its quote is found verbatim (whitespace-normalized) in one paragraph of the paper,
- the level value itself appears in that quote, and
- it is referenced to a band edge (Ec or Ev).

Everything else is listed in the rejects file with the reason. Kept records are
tagged provenance "llm_extracted_unreviewed" until a person has checked them.

Run:  defective extract-dft [--limit N] [--workers 4]
Writes data/extracted/dft_levels.llm.jsonl and dft_levels.llm.rejected.jsonl.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .corpus import Corpus, Paper
from .providers import make_provider

OUT = Path(__file__).resolve().parents[2] / "data/extracted"
# Host names as the matcher and UI spell them (keys: lowercase, alphanumerics only).
MATERIALS = {"cspbbr3": "CsPbBr3", "cdznte": "CdZnTe", "czt": "CdZnTe", "cdte": "CdTe",
             "cdztse": "CdZnTeSe", "cdzntese": "CdZnTeSe"}

PROMPT = """Extract every point-defect charge-transition level COMPUTED in this paper (DFT or similar).
Return ONLY a JSON object (no prose):
{
 "method": {"functional": str|null, "soc": true|false|null, "correction": str|null,
            "supercell_atoms": int|null, "band_gap_eV": number|null},
 "levels": [
  {"material": str, "defect": str, "transition": str, "level_eV": number,
   "level_reference": "Ec"|"Ev"|"unspecified",
   "defect_class": "vacancy"|"interstitial"|"antisite"|"substitutional"|"complex"|"extended",
   "sublattice": str|null, "character": "donor"|"acceptor"|"amphoteric"|"unknown",
   "favored_condition": str|null, "location": str|null, "quote": str}
 ]
}
Rules:
- "method" is the calculation used for the levels: functional (e.g. "HSE06 (alpha=0.43)"), spin-orbit
  coupling, finite-size correction scheme, supercell size, and the band gap of that calculation.
- defect labels like V_Br, Pb_i, Te_Cd, Cd_i, Cl_Te, V_Cd-Cl_Te; transition like "(0/-)", "(+/0)", "(2-/-)".
- level_eV is the level position measured from the stated band edge: "0.30 eV below the CBM" ->
  0.30, "Ec"; "0.12 eV above the VBM" -> 0.12, "Ev". Use "unspecified" if the edge is not stated.
- Only levels INSIDE the band gap. Skip states resonant in a band ("above the CBM", "above the band
  gap", "below the VBM"): they are not traps.
- "quote" must be copied verbatim from ONE paragraph or table row of the paper and must contain the
  number exactly as printed. Values read only off a figure: skip them.
- Only levels this paper computes itself; skip values it cites from other work and skip measured levels.
- material: the host crystal (CsPbBr3, CdZnTe, CdTe, ...). If none, return {"method": {...}, "levels": []}.

PAPER (doc_id: """


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _json_object(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def _material(name: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "", (name or "").lower().split("(")[0])
    # Alloy compositions (Cd0.9Zn0.1Te) map by element sequence; stoichiometric hosts must match exactly.
    alloy = re.sub(r"[\d]+", "", key)
    return MATERIALS.get(key) or (MATERIALS.get(alloy) if alloy.startswith("cdzn") else None) or name


def _number_in(value: float, quote: str) -> bool:
    nums = {float(x) for x in re.findall(r"(?<![\d.])\d+\.\d+|(?<![\d.])\d+(?![\d.])", quote.replace("−", "-"))}
    return any(abs(n - abs(value)) < 1e-9 for n in nums)


def check(rec: dict, paper: Paper) -> tuple[int | None, str | None]:
    """(paragraph index, None) if the record passes, else (None, reason)."""
    q = _norm(rec.get("quote") or "")
    if len(q) < 15:
        return None, "quote missing or too short"
    para = next((i for i, p in enumerate(paper.paragraphs) if q in _norm(p)), None)
    if para is None:
        return None, "quote not found verbatim in the paper"
    if not isinstance(rec.get("level_eV"), (int, float)) or not _number_in(rec["level_eV"], q):
        return None, "level value not in the quote"
    if rec.get("level_reference") not in ("Ec", "Ev"):
        return None, "no band-edge reference"
    return para, None


def extract_paper(paper: Paper, provider: str | None, model: str | None) -> tuple[list[dict], list[dict]]:
    llm = make_provider(provider, "You extract structured data from scientific papers.", [], model)
    text = "\n\n".join(paper.paragraphs)
    try:
        out = _json_object(llm.send_user(PROMPT + f"{paper.doc_id}):\n{text}").text)
    except Exception as e:   # one bad paper must not stop the run
        return [], [{"doc_id": paper.doc_id, "reason": f"extraction failed: {type(e).__name__}: {e}"}]
    method = {k: (out.get("method") or {}).get(k)
              for k in ("functional", "soc", "correction", "supercell_atoms", "band_gap_eV")}
    by = f"{llm.cfg['provider']}/{llm.cfg['model']}"
    kept, rejected = [], []
    for i, r in enumerate(out.get("levels") or []):
        para, why = check(r, paper)
        if why:
            rejected.append({"doc_id": paper.doc_id, "reason": why, "record": r})
            continue
        kept.append({
            "record_id": f"llm-{Path(paper.doc_id).stem[:40]}#{i}",
            "material": _material(r.get("material")), "defect": r.get("defect"),
            "transition": r.get("transition") or "unspecified", "level_eV": abs(float(r["level_eV"])),
            "level_reference": r["level_reference"], "defect_class": r.get("defect_class"),
            "sublattice": r.get("sublattice"), "character": r.get("character") or "unknown",
            "favored_condition": r.get("favored_condition"), "uncertainty_eV": None, "method": method,
            "source": {"doc_id": paper.doc_id, "paragraph": para, "location": r.get("location"),
                       "quote": _norm(r["quote"]), "doi": None},
            "provenance": "llm_extracted_unreviewed", "extracted_by": by,
        })
    return kept, rejected


def extract_dft(provider: str | None = None, model: str | None = None, limit: int | None = None,
                workers: int = 4) -> Path:
    papers = Corpus.load().visible("dft")[:limit]
    out, rej = OUT / "dft_levels.llm.jsonl", OUT / "dft_levels.llm.rejected.jsonl"
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda p: extract_paper(p, provider, model), papers))
    with out.open("w") as f, rej.open("w") as g:
        for paper, (kept, rejected) in zip(papers, results):
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for r in rejected:
                g.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {len(kept):3d} kept {len(rejected):3d} rejected  {paper.doc_id}")
    return out
