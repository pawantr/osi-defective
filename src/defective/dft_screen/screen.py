"""On-demand MACE-MH-1 defect-formation-energy verification for a single
(material, defect_label) pair named in materials.json.

Builds the bulk and defect structures with `defect_builder`, relaxes both with
MACE-MH-1 + ASE's FIRE optimizer, and reports the formation energy as
defect-minus-bulk. Results are cached by structure name under `cache_dir` so
repeated UI/agent calls for the same pair don't re-relax it.

`mace-torch` is an optional dependency (the `dft` extra in pyproject.toml): if
it isn't installed, or relaxation otherwise fails, `verify` reports that in
`status` instead of raising, so the rest of the pipeline keeps working.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import defect_builder

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_STRUCTURES_DIR = Path(__file__).resolve().parents[3] / "data" / "structures"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "runs" / "dft_screen"
MATERIALS_CONFIG = PACKAGE_DIR / "materials.json"


def _load_config() -> dict:
    return json.loads(MATERIALS_CONFIG.read_text(encoding="utf-8"))


def _material_spec(material: str, config: dict) -> dict:
    for m in config["materials"]:
        if m["name"] == material:
            return m
    known = [m["name"] for m in config["materials"]]
    raise ValueError(f"no dft_screen entry for material {material!r}; known: {known}")


def _defect_spec(material_spec: dict, defect_label: str) -> dict:
    for d in material_spec["defects"]:
        label = d.get("label") or defect_builder._default_label(d)
        if label == defect_label:
            return d
    known = [d.get("label") or defect_builder._default_label(d) for d in material_spec["defects"]]
    raise ValueError(f"no defect {defect_label!r} configured for material "
                      f"{material_spec['name']!r}; known: {known}")


def known_defects(material: str | None = None) -> dict[str, list[str]]:
    """List the defect labels `verify` can screen, per material (or for one material)."""
    config = _load_config()
    out = {}
    for m in config["materials"]:
        if material is not None and m["name"] != material:
            continue
        out[m["name"]] = [d.get("label") or defect_builder._default_label(d) for d in m["defects"]]
    return out


def relax(atoms, name: str, cache_dir: Path, calculator) -> dict:
    """Relax one ASE Atoms object with the given MACE calculator, caching the
    result by `name` under `cache_dir`."""
    from ase.io import write
    from ase.optimize import FIRE

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{name}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    atoms = atoms.copy()
    atoms.calc = calculator
    optimizer = FIRE(atoms, logfile=str(cache_dir / f"{name}.log"))
    converged = optimizer.run(fmax=0.03, steps=500)
    record = {
        "atom_count": len(atoms),
        "energy_eV": atoms.get_potential_energy(),
        "max_force_eV_per_A": float(abs(atoms.get_forces()).max()),
        "steps": optimizer.nsteps,
        "converged": bool(converged),
    }
    write(cache_dir / f"{name}_relaxed.cif", atoms)
    cache_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def verify(material: str, defect_label: str, structures_dir: Path | None = None,
           cache_dir: Path | None = None) -> dict[str, Any]:
    """Build + relax the bulk and the named defect for `material`, and report
    the defect-formation energy (defect energy minus bulk energy).

    Returns {"status": "ok", "formation_energy_eV": ..., ...} on success, or
    {"status": "mace_unavailable" | "unknown_defect" | "mace_error", "reason": ...}
    when verification can't run -- never raises."""
    structures_dir = Path(structures_dir) if structures_dir else DEFAULT_STRUCTURES_DIR
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR

    config = _load_config()
    try:
        material_spec = _material_spec(material, config)
        defect_spec = _defect_spec(material_spec, defect_label)
    except ValueError as e:
        return {"status": "unknown_defect", "reason": str(e)}

    bulk, _ = defect_builder.build_bulk(material_spec, structures_dir)
    defect_atoms, info = defect_builder.build_defect(bulk, defect_spec)
    bulk_label = material_spec.get("bulk_label") or "".join(ch for ch in material if ch.isalnum())

    try:
        from mace.calculators import mace_mp
    except ImportError as e:
        return {"status": "mace_unavailable",
                "reason": f"{e} -- install the 'dft' extra (uv sync --extra dft) to run this verification"}

    try:
        calculator = mace_mp(model="mh-1", device="cpu", default_dtype="float64", head="omat_pbe")
        bulk_record = relax(bulk, f"{bulk_label}_bulk", cache_dir, calculator)
        defect_record = relax(defect_atoms, f"{bulk_label}_{defect_label}", cache_dir, calculator)
    except Exception as e:  # relaxation backends can fail in many vendor-specific ways
        return {"status": "mace_error", "reason": f"{type(e).__name__}: {e}"}

    return {
        "status": "ok",
        "method": "MACE-MH-1",
        "material": material,
        "defect_label": defect_label,
        "formation_energy_eV": defect_record["energy_eV"] - bulk_record["energy_eV"],
        "bulk": bulk_record,
        "defect": defect_record,
        **info,
    }
