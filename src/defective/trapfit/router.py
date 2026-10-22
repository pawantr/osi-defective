"""Reads algorithms.json, picks the fitting route for a measurement technique, checks inputs, runs it."""
from __future__ import annotations

import json
import re
from importlib import import_module
from pathlib import Path

_DEFAULT = Path(__file__).with_name("algorithms.json")


def load_config(path=None):
    return json.loads(Path(path or _DEFAULT).read_text(encoding="utf-8"))


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def resolve_technique(name, config):
    n = _norm(name)
    for key, spec in config["techniques"].items():
        if n == _norm(key) or n in {_norm(a) for a in spec.get("aliases", [])}:
            return key
    return None


def infer_technique(request):
    """Guess from data shape. Returns None if ambiguous; the router then refuses and asks."""
    d = request.get("data", {})
    if "spectra" in d or "arrhenius_points" in d:
        return None  # DLTS, I-DLTS and PICTS share the data shape, so the technique must be stated
    if "temperature_K" in d and "current_A" in d:
        return "TSC"
    if "time_s" in d and "signal" in d:
        return "LAPLACE_DLTS"
    return None


def _get(request, dotted):
    cur = request
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _missing(request, spec):
    missing = [k for k in spec["required_inputs"] if _get(request, k) is None]
    alt = spec.get("alternative_input", {})
    if missing and any(_get(request, k) is not None for k in alt):
        return []
    return missing


def run(request, config=None):
    """request = {"technique": "TSC", "material": "...", "data": {...}, "metadata": {...}, "options": {...}}"""
    config = config or load_config()
    tech_in = request.get("technique")
    key = resolve_technique(tech_in, config) if tech_in else infer_technique(request)
    prov = {"technique_input": tech_in, "config_version": config.get("schema_version")}
    if key is None:
        if tech_in and _norm(tech_in) in {_norm(x) for x in config["global_policy"].get("excluded_techniques", [])}:
            return {"status": "refused", "reason": f"{tech_in} is excluded from this workflow", "provenance": prov}
        return {"status": "refused", "reason": "technique unknown or ambiguous; state one of " + ", ".join(config["techniques"]), "provenance": prov}
    spec = config["techniques"][key]
    miss = _missing(request, spec)
    if miss:
        return {"status": "refused", "technique": key, "reason": "missing required inputs", "missing": miss, "provenance": prov}

    opts = dict(spec.get("options", {}))
    opts.update(request.get("options", {}))
    data, meta = request.get("data", {}), request.get("metadata", {})
    mod_name, fn_name = spec["function"].rsplit(".", 1)
    try:
        mod = import_module(mod_name)
    except ModuleNotFoundError:
        # algorithms.json names modules as top-level "trapfit.*"; this copy is vendored as a
        # subpackage (e.g. defective.trapfit), so fall back to resolving relative to it.
        mod = import_module(f"{__package__}.{mod_name.split('.', 1)[-1]}")
    fn = getattr(mod, fn_name)
    prov.update({"technique": key, "route": spec["route"], "function": spec["function"], "algorithm": spec["algorithm"]["name"]})

    if key == "TSC":
        out = fn(data["temperature_K"], data["current_A"], dark_current_A=data.get("dark_current_A", 0.0),
                 heating_rate_K_per_s=meta.get("heating_rate_K_per_s"), max_peaks=opts.get("max_peaks", 16),
                 peak_model=opts.get("peak_model", "first_order_kitis"), criterion=opts.get("criterion", "bic"),
                 n_starts=opts.get("n_starts", 6), floor_fraction=opts.get("floor_fraction", 1e-5), config=config)
        traps = out.pop("peaks", []) if out["status"] == "ok" else []
    elif key == "LAPLACE_DLTS":
        out = fn(data["time_s"], data["signal"], n_rates=opts.get("n_rates", 120), alpha=opts.get("alpha", 0.01))
        traps = out.get("components", []) if out["status"] == "ok" else []
    else:
        th = config["quality_thresholds"]
        out = fn(spectra=data.get("spectra"), arrhenius_points=data.get("arrhenius_points"),
                 effective_mass_ratio=meta.get("effective_mass_ratio", 1.0), max_shift_K=opts.get("max_shift_K", 30.0),
                 smooth_window=opts.get("smooth_window", 7), min_prominence_fraction=opts.get("min_prominence_fraction", 0.1),
                 peak_sign=opts.get("peak_sign", "positive"), min_windows=th["min_rate_windows_arrhenius"])
        traps = out.pop("traps", []) if out["status"] == "ok" else []
        if "perovskite" in _norm(request.get("material", "")) or _norm(request.get("material", "")).startswith("cspb"):
            if key == "PICTS":
                for t in traps:
                    t.setdefault("flags", []).append("ionic_artifact_possible")

    for t in traps:
        t["method"] = key
        t["evidence_tag"] = config["global_policy"]["output_tag"]
    result = {"status": out["status"], "technique": key, "material": request.get("material"), "traps": traps, "fit": out, "provenance": prov}
    if out["status"] != "ok":
        result["reason"] = out.get("reason", "fit unresolved; no label forced")
    return result
