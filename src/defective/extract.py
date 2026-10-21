"""LLM extraction of experimental trap records from Docling markdown.

Fields follow the team's target list: material system, growth method, dopants,
characterization technique, detected defects and their energies — plus the
verbatim quote each number came from, so every record is source-linked.

Run:  defective extract --split dev [--limit N]
Writes data/extracted/traps.<provider>.jsonl (one record per detected level).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .corpus import Corpus
from .providers import make_provider

OUT = Path(__file__).resolve().parents[2] / "data/extracted"

PROMPT = """Extract every deep-level trap / defect state measured in this paper.
Return ONLY a JSON array (no prose). One object per reported level:
{
 "material": str, "label": str|null, "energy_eV": number, "energy_err_eV": number|null,
 "energy_reference": "Ec"|"Ev"|"unspecified", "carrier": "electron"|"hole"|"unknown",
 "cross_section_cm2": number|null, "concentration_cm3": number|null,
 "technique": str, "analysis_method": str|null,
 "growth": {"method": str|null, "dopants": [str], "stoichiometry": str|null, "anneal": str|null, "irradiation": str|null},
 "assignment": {"defect": str|null, "basis": "independent"|"dft_cited"|"literature_cited"|"none", "evidence": str|null},
 "source": {"location": str|null, "quote": str}
}
Rules: copy numbers exactly as printed; "quote" must be verbatim text from the paper containing the number;
"basis" is "independent" only if THIS paper's own doping/stoichiometry/anneal data supports the assignment.
Skip levels from other papers that are merely cited. If none, return [].

PAPER (doc_id: {doc_id}):
{text}"""


def _json_array(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    return json.loads(m.group(0)) if m else []


def extract(split: str = "dev", provider: str | None = None, model: str | None = None,
            limit: int | None = None) -> Path:
    corpus = Corpus.load()
    papers = corpus.visible(split)[:limit]
    probe = make_provider(provider, "", [], model)
    out = OUT / f"traps.{probe.cfg['provider']}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for paper in papers:
            llm = make_provider(provider, "You extract structured data from scientific papers.", [], model)
            text = "\n\n".join(paper.paragraphs)
            try:
                recs = _json_array(llm.send_user(PROMPT.format(doc_id=paper.doc_id, text=text)).text)
            except json.JSONDecodeError as e:
                print(f"  ! {paper.doc_id}: unparseable output ({e})")
                continue
            body = text.replace("\n", " ")
            for i, r in enumerate(recs):
                r["record_id"] = f"{Path(paper.doc_id).stem[:40]}#{i}"
                r.setdefault("source", {})["doc_id"] = paper.doc_id
                # Provenance check: flag records whose quote is not in the paper.
                q = re.sub(r"\s+", " ", r["source"].get("quote", ""))[:80]
                r["source"]["quote_verified"] = bool(q) and q in re.sub(r"\s+", " ", body)
                r["extracted_by"] = f"{llm.cfg['provider']}/{llm.cfg['model']}"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {len(recs):3d} levels  {paper.doc_id}")
    return out
