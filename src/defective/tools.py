"""Tools exposed to the agent. Plain functions + JSON Schemas, so the same set
can be served to any provider in-process or wrapped as an MCP server later."""
from __future__ import annotations

import json
from typing import Any, Callable

from . import matcher, peaks
from .corpus import Corpus
from .providers import Tool


class Toolbox:
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.fns: dict[str, Callable[..., Any]] = {}
        self.specs: list[Tool] = []
        self._register()

    def tool(self, name: str, description: str, properties: dict, required: list[str] = ()):
        def deco(fn):
            self.fns[name] = fn
            self.specs.append(Tool(name, description, {
                "type": "object", "properties": properties, "required": list(required),
                "additionalProperties": False}))
            return fn
        return deco

    def call(self, name: str, args: dict) -> tuple[str, bool]:
        if name not in self.fns:
            return f"Unknown tool {name!r}", True
        try:
            return json.dumps(self.fns[name](**args), ensure_ascii=False, default=str), False
        except (PermissionError, KeyError, ValueError, TypeError, FileNotFoundError) as e:
            return f"{type(e).__name__}: {e}", True

    def _register(self):
        c = self.corpus
        trap_props = {
            "material": {"type": "string"},
            "energy_eV": {"type": "number"},
            "energy_reference": {"type": "string", "enum": ["Ec", "Ev", "unspecified"]},
            "energy_err_eV": {"type": "number"},
        }

        @self.tool("list_papers", "List visible papers (held-out papers are never listed).",
                   {"split": {"type": "string", "enum": ["dft", "dev"]}})
        def list_papers(split: str | None = None):
            return [{"doc_id": p.doc_id, "split": p.split, "title": p.title} for p in c.visible(split)]

        @self.tool("search_papers", "Keyword (BM25) search over paper paragraphs. Returns doc_id + paragraph index + text.",
                   {"query": {"type": "string"}, "k": {"type": "integer"},
                    "split": {"type": "string", "enum": ["dft", "dev"]}}, ["query"])
        def search_papers(query: str, k: int = 8, split: str | None = None):
            return c.search(query, k, split)

        @self.tool("read_passage", "Read paragraphs [start, start+n) of a paper, to quote evidence.",
                   {"doc_id": {"type": "string"}, "start": {"type": "integer"}, "n": {"type": "integer"}},
                   ["doc_id", "start"])
        def read_passage(doc_id: str, start: int, n: int = 6):
            paras = c.get(doc_id).paragraphs
            return {"doc_id": doc_id, "paragraphs": {i: paras[i] for i in range(start, min(start + n, len(paras)))}}

        @self.tool("query_dft_levels", "Harmonized DFT charge-transition levels for a material, with method metadata.",
                   {"material": {"type": "string"}}, ["material"])
        def query_dft_levels(material: str):
            return matcher.dft_levels(material)

        @self.tool("match_trap", "Rank candidate defects for a measured trap (probabilistic matcher over DFT levels).",
                   trap_props, ["material", "energy_eV"])
        def match_trap(material: str, energy_eV: float, energy_reference: str = "unspecified",
                       energy_err_eV: float | None = None):
            trap = {"energy_eV": energy_eV, "energy_reference": energy_reference, "energy_err_eV": energy_err_eV}
            return matcher.match(trap, matcher.dft_levels(material))

        @self.tool("propose_experiment",
                   "Rank discriminating experiments by expected information gain over the matcher's top candidates.",
                   {**trap_props, "top_k": {"type": "integer"}}, ["material", "energy_eV"])
        def propose_experiment(material: str, energy_eV: float, energy_reference: str = "unspecified",
                               energy_err_eV: float | None = None, top_k: int = 5):
            levels = matcher.dft_levels(material)
            trap = {"energy_eV": energy_eV, "energy_reference": energy_reference, "energy_err_eV": energy_err_eV}
            cands = matcher.match(trap, levels)["candidates"][:top_k]
            z = sum(x["posterior"] for x in cands) or 1.0
            cands = [{**x, "posterior": x["posterior"] / z} for x in cands]
            return {"top_candidates": cands, "experiments": matcher.propose_experiments(cands, levels)}

        @self.tool("fit_trap_curve",
                   "Fit a digitized 2-column CSV to trap energies via trapfit (Kitis-model glow-peak fitting). "
                   "TSC: (T [K], current) curve, simultaneously deconvolved into N peaks (N by BIC) -> "
                   "energy_eV, peak_T_K, frequency_factor_per_s per peak. DLTS/I-DLTS/PICTS: Arrhenius pairs "
                   "(T or Tm, emission rate / rate window) -> energy_eV, apparent_cross_section_cm2. "
                   "LAPLACE_DLTS: one isothermal (time_s, signal) transient -> emission-rate components. "
                   "Returns status ok/unresolved/refused; never forces a label. Feed energy_eV into match_trap.",
                   {"csv_path": {"type": "string"},
                    "technique": {"type": "string", "enum": ["TSC", "DLTS", "I-DLTS", "PICTS", "LAPLACE_DLTS"]},
                    "heating_rate_K_per_s": {"type": "number"}, "max_peaks": {"type": "integer"},
                    "m_eff": {"type": "number", "description": "m*/m0, for capture cross-section"},
                    "x_kind": {"type": "string", "enum": ["T", "1000/T", "1/T"]},
                    "y_kind": {"type": "string", "enum": ["rate", "tau", "ln(e/T^2)", "ln(tau*T^2)"]}},
                   ["csv_path", "technique"])
        def fit_trap_curve(csv_path: str, technique: str, heating_rate_K_per_s: float | None = None,
                           max_peaks: int = 4, m_eff: float | None = None, x_kind: str = "T",
                           y_kind: str = "rate"):
            return peaks.fit_curve(csv_path, technique, heating_rate_K_per_s, max_peaks, m_eff, x_kind, y_kind)

        @self.tool("verify_defect_formation",
                   "Computationally verify a candidate defect by relaxing its structure with MACE-MH-1 "
                   "and reporting the defect-formation energy (defect minus bulk, eV). Only a fixed set "
                   "of (material, defect_label) pairs are configured for screening (see dft_screen/materials.json); "
                   "report this as supporting evidence alongside match_trap's posterior, not blended into it. "
                   "Returns status mace_unavailable if the optional 'dft' extra (mace-torch) isn't installed.",
                   {"material": {"type": "string"}, "defect_label": {"type": "string"}},
                   ["material", "defect_label"])
        def verify_defect_formation(material: str, defect_label: str):
            try:
                from .dft_screen import screen as dft_screen
            except ImportError as e:
                return {"status": "mace_unavailable",
                        "reason": f"{e} -- install the 'dft' extra (uv sync --extra dft) to run this verification"}
            return dft_screen.verify(material, defect_label)
