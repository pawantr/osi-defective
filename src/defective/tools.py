"""Tools exposed to the agent. Plain functions + JSON Schemas, so the same set
can be served to any provider in-process or wrapped as an MCP server later."""
from __future__ import annotations

import json
from typing import Any, Callable

from . import matcher
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
