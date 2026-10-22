"""Offline tests for the held-out evaluation harness. The LLM is a fake object; no network."""
import csv
import json
import math
from pathlib import Path

import pytest

from defective import evaluate, matcher
from defective.evaluate import normalize_defect
from defective.providers import Turn

FIX = Path(__file__).parent / "fixtures"
GOLD = FIX / "eval_gold.jsonl"
DFT = FIX / "eval_dft_levels.jsonl"


@pytest.mark.parametrize("label", [
    "V_Br", "VBr", "V$_{Br}$", "V$_\\mathrm{Br}$", "V_{Br}", "v_br", "bromine vacancy", "Br vacancy",
    "vacancy of bromine", "V_Br(+/0)", "V_{Br}^{+}", "V$_{Br}^{\\bullet}$", "Bromine Vacancies",
])
def test_normalize_bromine_vacancy(label):
    assert normalize_defect(label) == "V_Br"


@pytest.mark.parametrize("label,key", [
    ("Te_Cd", "Te_Cd"), ("TeCd", "Te_Cd"), ("Te$_{Cd}$", "Te_Cd"), ("tellurium on cadmium site", "Te_Cd"),
    ("Pb_i", "Pb_i"), ("Pbi", "Pb_i"), ("Pb$_i$", "Pb_i"), ("lead interstitial", "Pb_i"),
    ("interstitial lead", "Pb_i"), ("Cd_i+", "Cd_i"),
    ("V_Cd(2-/-)", "V_Cd"), ("V_Cd^{2-}", "V_Cd"), ("V_Cd2-", "V_Cd"), ("cadmium vacancy", "V_Cd"),
    ("Cd vacancy (V_Cd)", "V_Cd"),
    ("I_i", "I_i"), ("V_I", "V_I"), ("iodine vacancy", "V_I"),   # iodine is not the interstitial 'i'
    ("Cl_Te+V_Cd", "Cl_Te+V_Cd"), ("V_Cd-Cl_Te complex", "Cl_Te+V_Cd"), ("V_Cd + Cl_Te", "Cl_Te+V_Cd"),
])
def test_normalize_other_defects(label, key):
    assert normalize_defect(label) == key


def test_normalize_distinguishes_and_passes_through():
    assert normalize_defect("V_Br") != normalize_defect("V_Pb")
    assert normalize_defect("Te_Cd") != normalize_defect("Cd_Te")      # site order matters
    assert normalize_defect("EL2") == normalize_defect(" el2 ") == "el2"
    assert normalize_defect(None) == normalize_defect("") == ""
    assert evaluate.gold_keys("V_Br or Pb_i") == {"V_Br", "Pb_i"}


def test_split_gold_scores_only_independent():
    scored, excluded = evaluate.split_gold(matcher.load_jsonl(GOLD))
    assert [r["record_id"] for r in scored] == ["g1", "g2", "g3", "g8"]
    why = {e["record_id"]: e["reason"] for e in excluded}
    assert set(why) == {"g4", "g5", "g6", "g7"}
    assert "dft_cited" in why["g4"] and "literature_cited" in why["g5"]
    assert "none" in why["g6"] and "no assignment" in why["g7"]


def test_blind_view_hides_answer():
    rec = matcher.load_jsonl(GOLD)[0]
    view = json.dumps(evaluate.blind_view(rec))
    for leak in ("assignment", "SECRET", "bromine vacancy", '"T1"', "label"):
        assert leak not in view
    for key in ("material", "energy_eV", "energy_reference", "cross_section_cm2", "technique", "growth"):
        assert key in evaluate.blind_view(rec)


class FakeLLM:
    """Stands in for providers.Provider: records prompts, answers from a script."""
    def __init__(self, log, answer):
        self.log, self.answer = log, answer

    def send_user(self, text):
        self.log.append(text)
        return Turn(self.answer(text))


def _levels(material):
    return matcher.dft_levels(material, DFT)


def test_nearest_and_matcher_metrics():
    res = evaluate.evaluate(matcher.load_jsonl(GOLD), ["nearest", "matcher"], _levels)
    assert res["n_total"] == 8 and res["n_scored"] == 4 and len(res["excluded"]) == 4
    s = {x["condition"]: x for x in res["summary"]}
    # g1, g2, g3 have the gold level nearest; g8 (CdZnTe) has no DFT levels -> miss
    assert s["nearest"]["top1"] == 0.75 and s["nearest"]["top3"] == 0.75
    assert s["nearest"]["n_no_dft_levels"] == 1
    rows = {(r["record_id"], r["condition"]): r for r in res["rows"]}
    assert rows[("g8", "matcher")]["gold_rank"] is None and rows[("g8", "matcher")]["gold_posterior"] == 0
    # V_Br and V$_{Br}$ levels collapse to one defect whose posterior is the sum of both
    g1 = rows[("g1", "matcher")]
    assert g1["predicted_ranked"].split("|")[0] == "V_Br"
    cands = matcher.match(evaluate.blind_view(matcher.load_jsonl(GOLD)[0]), _levels("CsPbBr3"))["candidates"]
    vbr = sum(c["posterior"] for c in cands if normalize_defect(c["defect"]) == "V_Br")
    assert math.isclose(g1["gold_posterior"], vbr)
    # calibration summary is the mean over scored records, computed from the rows
    m = [r for r in res["rows"] if r["condition"] == "matcher"]
    assert math.isclose(s["matcher"]["mean_gold_posterior"], sum(r["gold_posterior"] for r in m) / 4)
    brier = sum((r["top1_posterior"] - (r["gold_rank"] == 1)) ** 2 for r in m) / 4
    assert math.isclose(s["matcher"]["brier_top1"], brier)
    assert s["nearest"]["mean_gold_posterior"] is None


def test_mrr_and_rank():
    assert evaluate.rank_of(["A", "B", "C"], {"C"}) == 3
    assert evaluate.rank_of(["A"], {"Z"}) is None
    rows = [{"condition": "llm", "gold_rank": k, "error": "", "n_dft_levels": 1} for k in (1, 2, None, 4)]
    s = evaluate.summarize(rows, ["llm"])[0]
    assert s["top1"] == 0.25 and s["top3"] == 0.5
    assert math.isclose(s["mrr"], (1 + 0.5 + 0 + 0.25) / 4)


def test_llm_conditions_with_fake_provider():
    log = []
    answers = {"0.32": '```json\n["bromine vacancy", "Pb_i"]\n```',
               "0.58": 'Likely: ["V_Br", "Pb$_{i}$", "V_Pb"]',
               "1.85": '[{"defect": "VPb"}]',
               "0.79": "I cannot tell."}

    def answer(prompt):
        return next(v for k, v in answers.items() if f'"energy_eV": {k}' in prompt)

    res = evaluate.evaluate(matcher.load_jsonl(GOLD), ["llm", "llm_dft"], _levels,
                            lambda system: FakeLLM(log, answer))
    assert len(log) == 8                                  # 4 scored records x 2 conditions
    assert all("SECRET" not in p and "\"assignment\"" not in p and "\"label\"" not in p for p in log)
    plain = [p for p in log if "DFT TABLE" not in p]
    table = [p for p in log if "DFT TABLE" in p]
    assert len(plain) == len(table) == 4
    assert all("dft-vbr" not in p for p in table)          # record ids are not needed by the LLM
    assert any('"defect": "Pb_i"' in p for p in table)
    assert any("no computed transition levels" in p for p in table)   # CdZnTe has none
    s = {x["condition"]: x for x in res["summary"]}
    for cond in ("llm", "llm_dft"):
        assert s[cond]["top1"] == 0.5 and s[cond]["top3"] == 0.75   # g1, g3 top-1; g2 rank 2; g8 miss
        assert math.isclose(s[cond]["mrr"], (1 + 0.5 + 1 + 0) / 4)
        assert s[cond]["n_errors"] == 1                    # g8 reply had no JSON array
        assert s[cond]["brier_top1"] is None


def test_llm_conditions_require_provider():
    with pytest.raises(ValueError):
        evaluate.evaluate([], ["llm"], _levels)
    with pytest.raises(ValueError):
        evaluate.evaluate([], ["bogus"], _levels)


def test_main_writes_csvs_and_figure(tmp_path, capsys):
    out = evaluate.main(GOLD, ["nearest", "matcher"], dft_path=DFT, out_dir=tmp_path / "run")
    text = capsys.readouterr().out
    assert "Probabilistic matcher" in text and "excluded" in text
    with (out / "predictions.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 8
    with (out / "summary.csv").open() as f:
        summary = {r["condition"]: r for r in csv.DictReader(f)}
    assert float(summary["nearest"]["top1"]) == 0.75 and summary["matcher"]["brier_top1"] != ""
    with (out / "excluded.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 4
    assert json.loads((out / "run.json").read_text())["n_excluded"] == 4
    for ext in (".pdf", ".png", ".svg"):
        assert (out / f"metrics{ext}").stat().st_size > 0
    assert "<text" in (out / "metrics.svg").read_text()     # editable text, not glyph paths


def test_main_missing_gold(tmp_path):
    with pytest.raises(FileNotFoundError):
        evaluate.main(tmp_path / "nope.jsonl", ["nearest"])
