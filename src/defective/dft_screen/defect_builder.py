"""Flexible point-defect structure generation for DFT/MLIP prescreening.

A material entry names a source CIF, an optional supercell repeat, an optional
list of "pre_substitutions" (alloy-type species swaps applied before anything
is called "bulk" -- e.g. building a Cd0.9Zn0.1Te supercell from a CdTe-like
primitive cell), and a list of "defects" (vacancy, substitution or
interstitial) each applied independently to that bulk. One engine driven by a
JSON config, so new materials or defect types don't need new code -- see
materials.json for the config that reproduces the original two hardcoded
defects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read, write

DefectSpec = dict[str, Any]
MaterialSpec = dict[str, Any]


def _matching_indices(atoms: Atoms, element: str) -> list[int]:
    return [i for i, atom in enumerate(atoms) if atom.symbol == element]


def select_indices(atoms: Atoms, element: str | None = None, indices: list[int] | None = None,
                    site_index: int = 0, count: int | None = None) -> list[int]:
    """Resolve which atom(s) a defect/substitution spec refers to.

    Explicit `indices` wins. Otherwise atoms matching `element` are listed in
    atom order; `count` takes the first N of them, `site_index` (default 0)
    takes the single Nth one.
    """
    if indices is not None:
        return list(indices)
    if element is None:
        raise ValueError("spec needs 'indices' or 'element' to select a site")
    matches = _matching_indices(atoms, element)
    if not matches:
        raise ValueError(f"no {element!r} atoms found to select a site")
    if count is not None:
        if count > len(matches):
            raise ValueError(f"asked for {count} {element!r} sites but only {len(matches)} exist")
        return matches[:count]
    if site_index >= len(matches):
        raise ValueError(f"site_index {site_index} out of range for {len(matches)} {element!r} atoms")
    return [matches[site_index]]


def apply_vacancy(atoms: Atoms, spec: DefectSpec) -> tuple[Atoms, dict]:
    idx = select_indices(atoms, spec.get("element"), spec.get("indices"), spec.get("site_index", 0))[0]
    out = atoms.copy()
    frac = out.get_scaled_positions()[idx].tolist()
    removed = out.pop(idx)
    return out, {"removed_atom": {"index_zero_based": idx, "element": removed.symbol, "fractional_position": frac}}


def apply_substitution(atoms: Atoms, spec: DefectSpec) -> tuple[Atoms, dict]:
    to_element = spec["to_element"]
    from_element = spec.get("from_element") or spec.get("element")
    idxs = select_indices(atoms, from_element, spec.get("indices"), spec.get("site_index", 0), spec.get("count"))
    out = atoms.copy()
    changed = []
    frac_all = out.get_scaled_positions()
    for idx in idxs:
        from_el = out[idx].symbol
        out[idx].symbol = to_element
        changed.append({"index_zero_based": idx, "from_element": from_el, "to_element": to_element,
                         "fractional_position": frac_all[idx].tolist()})
    return out, {"substituted_atoms": changed}


def apply_interstitial(atoms: Atoms, spec: DefectSpec) -> tuple[Atoms, dict]:
    element = spec["element"]
    frac = list(spec["frac_coord"])
    out = atoms.copy()
    out.append(element)
    positions = out.get_scaled_positions()
    positions[-1] = frac
    out.set_scaled_positions(positions)
    return out, {"added_atom": {"index_zero_based": len(out) - 1, "element": element, "fractional_position": frac}}


_DEFECT_BUILDERS = {
    "vacancy": apply_vacancy,
    "substitution": apply_substitution,
    "interstitial": apply_interstitial,
}


def _default_label(spec: DefectSpec) -> str:
    kind = spec["type"]
    if kind == "vacancy":
        return f"V_{spec.get('element', 'site')}"
    if kind == "substitution":
        to_el = spec["to_element"]
        from_el = spec.get("from_element") or spec.get("element") or "site"
        return f"{to_el}_on_{from_el}"
    if kind == "interstitial":
        return f"{spec['element']}_i"
    return kind


def build_defect(atoms: Atoms, spec: DefectSpec) -> tuple[Atoms, dict]:
    kind = spec["type"]
    if kind not in _DEFECT_BUILDERS:
        raise ValueError(f"unknown defect type {kind!r}; use one of {sorted(_DEFECT_BUILDERS)}")
    return _DEFECT_BUILDERS[kind](atoms, spec)


def build_bulk(material: MaterialSpec, root: Path) -> tuple[Atoms, dict]:
    atoms = read(root / material["source"])
    supercell = tuple(material.get("supercell", (1, 1, 1)))
    if supercell != (1, 1, 1):
        atoms = atoms.repeat(supercell)
    atoms.info = {}
    applied = []
    for sub in material.get("pre_substitutions", []):
        atoms, info = apply_substitution(atoms, sub)
        applied.append({"label": sub.get("label", _default_label({**sub, "type": "substitution"})), **info})
    meta = {
        "source": material["source"],
        "supercell": list(supercell),
        "pre_substitutions": applied,
        "formula": atoms.get_chemical_formula(),
        "atom_count": len(atoms),
    }
    return atoms, meta


def _charge_tag(charge: float) -> str:
    return "neutral" if charge == 0 else f"q{charge:+g}"


def generate_structures(config: dict, root: Path, output: Path) -> dict:
    """Build bulk + defect CIFs for every material/defect `config` lists, write
    them under `output`, and return the combined metadata describing them."""
    output.mkdir(exist_ok=True, parents=True)
    metadata: dict[str, Any] = {
        "calculation_scope": config.get("calculation_scope", "Point-defect MACE-MH-1 prescreening inputs"),
        "relaxation_status": "not_run",
        "materials": {},
    }
    for material in config["materials"]:
        name = material["name"]
        bulk, bulk_meta = build_bulk(material, root)
        bulk_label = material.get("bulk_label") or "".join(ch for ch in name if ch.isalnum())
        bulk_file = f"{bulk_label}_bulk.cif"
        write(output / bulk_file, bulk)

        defects_meta = {}
        for defect in material["defects"]:
            defect_atoms, info = build_defect(bulk, defect)
            label = defect.get("label") or _default_label(defect)
            charge = defect.get("charge", 0)
            defect_file = f"{bulk_label}_{label}_{_charge_tag(charge)}.cif"
            write(output / defect_file, defect_atoms)
            defects_meta[label] = {
                "type": defect["type"],
                "charge": charge,
                "file": defect_file,
                "bulk_file": bulk_file,
                "formula": defect_atoms.get_chemical_formula(),
                "atom_count": len(defect_atoms),
                **info,
            }

        metadata["materials"][name] = {"bulk_file": bulk_file, **bulk_meta, "defects": defects_meta}
    return metadata
