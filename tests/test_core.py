"""Offline tests: corpus guard, matcher, and design math. No LLM calls."""
import json
import math
from pathlib import Path

import pytest

from defective import matcher
from defective.corpus import Corpus, normalize_title

CFG = Path(__file__).resolve().parents[1] / "configs/corpus.yaml"


@pytest.fixture
def corpus(tmp_path):
    for rel, body in {
        "Experimental Data/Deep levels in CZT.md": "# Deep levels in CZT\n\nA trap at Ec-0.79 eV from TSC.",
        "Experimental Data/Identification of deep trap levels from.md": "# Pavlovic\n\nEL2-like 0.99 eV donor.",
        "Test Data/Identification of deep trap levels from (2).md": "# Pavlovic\n\nheld out copy",
        "DFT Analysis/Vacancies in CsPbBr3.md": "# DFT\n\nV_Br (+/0) level.",
    }.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(body)
    return Corpus.load(tmp_path, CFG)


def test_normalize_title():
    assert normalize_title("Foo Bar (2)") == normalize_title("foo  bar")


def test_heldout_and_duplicates_are_blocked(corpus):
    dup = "Experimental Data/Identification of deep trap levels from.md"
    assert dup in corpus.leaked_duplicates()
    with pytest.raises(PermissionError):
        corpus.get(dup)
    assert all("Pavlovic" not in h["text"] and "EL2" not in h["text"] for h in corpus.search("EL2 0.99 donor"))
    assert {p.split for p in corpus.visible()} == {"dev", "dft"}


def test_search_finds_visible(corpus):
    assert corpus.search("TSC trap")[0]["doc_id"] == "Experimental Data/Deep levels in CZT.md"


LEVELS = [
    {"record_id": "a", "material": "X", "defect": "A", "transition": "(0/-)", "level_eV": 0.80,
     "level_reference": "Ec", "uncertainty_eV": 0.1, "character": "acceptor", "favored_condition": "rich"},
    {"record_id": "b", "material": "X", "defect": "B", "transition": "(+/0)", "level_eV": 0.70,
     "level_reference": "Ec", "uncertainty_eV": 0.1, "character": "donor", "favored_condition": "rich"},
    {"record_id": "c", "material": "X", "defect": "C", "transition": "(+/0)", "level_eV": 0.30,
     "level_reference": "Ev", "uncertainty_eV": 0.1, "character": "donor", "favored_condition": "poor",
     "method": {"band_gap_eV": 1.0}},
    {"record_id": "d", "material": "X", "defect": "D", "transition": "(0/-)", "level_eV": 0.2,
     "level_reference": "Ev", "uncertainty_eV": 0.1},   # no gap: cannot be referenced to Ec
]


def test_match_posterior_and_reference_conversion():
    res = matcher.match({"energy_eV": 0.75, "energy_reference": "Ec", "energy_err_eV": 0.0001}, LEVELS)
    post = {c["defect"]: c["posterior"] for c in res["candidates"]}
    assert res["skipped_no_common_reference"] == ["d"]
    assert math.isclose(sum(post.values()), 1, abs_tol=1e-3)
    assert math.isclose(post["A"], post["B"], abs_tol=1e-3)      # symmetric about 0.75
    c = next(c for c in res["candidates"] if c["defect"] == "C")
    assert c["level_eV"] == 0.7 and c["reference"] == "Ec"         # 1.0 - 0.30


def test_design_prefers_splitting_experiment():
    cands = [{"record_id": "a", "defect": "A", "transition": "", "posterior": 0.5},
             {"record_id": "b", "defect": "B", "transition": "", "posterior": 0.5}]
    ranked = matcher.propose_experiments(cands, LEVELS)
    gains = {e["discriminates_on"]: e["expected_info_gain_bits"] for e in ranked}
    assert gains["character"] == 1.0          # acceptor vs donor: one full bit
    assert gains["favored_condition"] == 0.0  # both "rich": anneal can't separate them
    assert ranked[0]["discriminates_on"] == "character"


def test_seed_dft_table_matches_schema():
    schema = json.loads((CFG.parents[1] / "schemas/dft_level.schema.json").read_text())
    for rec in matcher.dft_levels():
        assert set(schema["required"]) <= rec.keys(), rec["record_id"]


def test_guard_holds_without_test_data(tmp_path):
    (tmp_path / "Experimental Data").mkdir()
    (tmp_path / "Experimental Data/Identification of deep trap levels from.md").write_text("# P\n\nEL2 0.99 eV")
    c = Corpus.load(tmp_path, CFG)
    assert c.leaked_duplicates() == ["Experimental Data/Identification of deep trap levels from.md"]
    assert c.visible() == []
