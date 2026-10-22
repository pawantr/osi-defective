"""dft_screen tests: structure generation is pure ASE (always available), while
MACE relaxation itself must degrade to a status dict in this environment
(mace-torch is not installed)."""
from pathlib import Path

import pytest

ase = pytest.importorskip("ase", reason="dft_screen needs the 'dft' extra (ase)")

from defective.dft_screen import defect_builder, screen

STRUCTURES_DIR = Path(__file__).resolve().parents[1] / "data" / "structures"


def _config():
    return screen._load_config()


def test_materials_config_sources_exist():
    config = _config()
    for material in config["materials"]:
        assert (STRUCTURES_DIR / material["source"]).exists(), material["source"]


def test_build_bulk_applies_pre_substitutions():
    config = _config()
    cdznte = next(m for m in config["materials"] if m["name"] == "CdZnTe")
    bulk, meta = defect_builder.build_bulk(cdznte, STRUCTURES_DIR)
    assert meta["pre_substitutions"]
    assert sum(1 for a in bulk if a.symbol == "Zn") == cdznte["pre_substitutions"][0]["count"]


def test_build_defect_vacancy_removes_one_atom():
    config = _config()
    cspbbr3 = next(m for m in config["materials"] if m["name"] == "CsPbBr3")
    bulk, _ = defect_builder.build_bulk(cspbbr3, STRUCTURES_DIR)
    defect_atoms, info = defect_builder.build_defect(bulk, cspbbr3["defects"][0])
    assert len(defect_atoms) == len(bulk) - 1
    assert info["removed_atom"]["element"] == cspbbr3["defects"][0]["element"]


def test_known_defects_lists_configured_labels():
    known = screen.known_defects("CsPbBr3")
    assert known == {"CsPbBr3": ["VBr"]}


def test_verify_unknown_material_reports_cleanly():
    res = screen.verify("Unobtanium", "VX")
    assert res["status"] == "unknown_defect"


def test_verify_degrades_without_mace(tmp_path):
    try:
        import mace  # noqa: F401
        pytest.skip("mace-torch is installed here; this asserts the no-mace path")
    except ImportError:
        pass
    res = screen.verify("CsPbBr3", "VBr", cache_dir=tmp_path)
    assert res["status"] == "mace_unavailable"
