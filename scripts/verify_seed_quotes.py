"""Check every DFT seed record's quote appears verbatim (whitespace-normalized) in its cited paragraph.

Run:  uv run python scripts/verify_seed_quotes.py   (needs DEFECTIVE_CORPUS_DIR)
"""
import re
import sys

from dotenv import load_dotenv

from defective import matcher
from defective.corpus import Corpus


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def verify(corpus: Corpus, records: list[dict]) -> list[str]:
    bad = []
    for r in records:
        src = r["source"]
        paras = corpus.get(src["doc_id"]).paragraphs   # raises PermissionError for held-out papers
        i = src.get("paragraph")
        text = paras[i] if i is not None and i < len(paras) else ""
        if _norm(src["quote"]) not in _norm(text):
            bad.append(r["record_id"])
    return bad


if __name__ == "__main__":
    load_dotenv(".env")
    recs = matcher.dft_levels()
    bad = verify(Corpus.load(), recs)
    print(f"{len(recs) - len(bad)}/{len(recs)} quotes verified" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)
