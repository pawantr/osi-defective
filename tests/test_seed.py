"""DFT tables (seed + LLM-extracted): every quote is verbatim in its cited paragraph (skipped without the local corpus)."""
import importlib.util
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from defective import matcher
from defective.corpus import Corpus

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
CORPUS = os.environ.get("DEFECTIVE_CORPUS_DIR", "")


@pytest.mark.skipif(not CORPUS or not Path(CORPUS).expanduser().is_dir(), reason="local corpus not available")
def test_seed_quotes_verbatim():
    spec = importlib.util.spec_from_file_location("v", ROOT / "scripts/verify_seed_quotes.py")
    v = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v)
    assert v.verify(Corpus.load(), matcher.dft_levels()) == []


def test_seed_never_cites_heldout():
    assert matcher.dft_levels()
    assert all(not r["source"]["doc_id"].startswith("Test Data/") for r in matcher.dft_levels())
