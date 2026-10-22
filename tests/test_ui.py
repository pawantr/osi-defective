"""Offline UI tests: the Blocks app builds and the direct matcher tab works. No LLM calls, no launch."""
import pytest

gr = pytest.importorskip("gradio")

from defective import matcher, ui  # noqa: E402

LEVELS = [
    {"record_id": "a", "material": "X", "defect": "A", "transition": "(0/-)", "level_eV": 0.80,
     "level_reference": "Ec", "uncertainty_eV": 0.1, "character": "acceptor"},
    {"record_id": "b", "material": "X", "defect": "B", "transition": "(+/0)", "level_eV": 0.70,
     "level_reference": "Ec", "uncertainty_eV": 0.1, "character": "donor"},
]


def test_build_ui():
    assert isinstance(ui.build_ui("anthropic", "fast"), gr.Blocks)
    assert ui.models_for("argo")[:2] == ["balanced", "fast"]


def test_match_table_empty(monkeypatch):
    monkeypatch.setattr(matcher, "dft_levels", lambda material=None: [])
    msg, cands, exps = ui.match_table("CdZnTe", 0.79, "Ec", 0.05)
    assert "empty" in msg and cands.empty and exps.empty


def test_match_table_ranks(monkeypatch):
    monkeypatch.setattr(matcher, "dft_levels", lambda material=None: LEVELS)
    msg, cands, exps = ui.match_table("X", 0.79, "Ec", 0.01)
    assert list(cands["defect"]) == ["A", "B"] and cands["rank"].tolist() == [1, 2]
    assert exps.iloc[0]["discriminates_on"] == "character"


def test_missing_key_is_reported(monkeypatch, tmp_path):
    (tmp_path / "Experimental Data").mkdir()
    (tmp_path / "Experimental Data/p.md").write_text("# P\n\ntext")
    monkeypatch.setattr(ui, "_corpus", None)
    monkeypatch.setenv("DEFECTIVE_CORPUS_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent, msg = ui.build_agent("openai", "fast")
    assert agent is None and "OPENAI_API_KEY" in msg
