"""Read-only access to the Docling-converted paper corpus, with a held-out guard.

The guard is enforced here, at the tool boundary, rather than asked of the
model in a prompt: a held-out paper is never searchable or readable, including
duplicate copies that were filed under another split.
"""
from __future__ import annotations

import fnmatch
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
_WORD = re.compile(r"[a-z0-9_.+-]+")


def normalize_title(stem: str) -> str:
    """'Foo Bar (2)' -> 'foo bar'. Docling names files by the first six title words."""
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    return re.sub(r"[^a-z0-9]+", " ", stem.lower()).strip()


@dataclass
class Paper:
    doc_id: str          # path relative to corpus root, e.g. "Experimental Data/Foo.md"
    split: str
    path: Path

    @cached_property
    def paragraphs(self) -> list[str]:
        text = self.path.read_text(encoding="utf-8", errors="replace")
        paras = [p.strip() for p in re.split(r"\n\s*\n", text)]
        return [p for p in paras if p and not p.startswith("![Image]")]

    @property
    def title(self) -> str:
        for p in self.paragraphs[:5]:
            if p.startswith("#"):
                return re.sub(r"[#\[\]]|\(http\S+\)", "", p).strip()
        return self.path.stem


@dataclass
class Corpus:
    root: Path
    papers: dict[str, Paper] = field(default_factory=dict)
    blocked: set[str] = field(default_factory=set)   # doc_ids hidden by the held-out guard

    @classmethod
    def load(cls, root: str | Path | None = None, config: Path = REPO / "configs/corpus.yaml") -> "Corpus":
        cfg = yaml.safe_load(config.read_text())
        root = Path(root or os.environ.get("DEFECTIVE_CORPUS_DIR", "")).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Corpus not found: {root!s}. Set DEFECTIVE_CORPUS_DIR.")
        corpus = cls(root=root)
        for md in sorted(root.rglob("*.md")):
            rel = md.relative_to(root).as_posix()
            split = next((s for s, globs in cfg["splits"].items()
                          if any(fnmatch.fnmatch(rel, g) for g in globs)), None)
            if split and split != "ignore":
                corpus.papers[rel] = Paper(rel, split, md)
        if cfg.get("heldout_guard", True):
            heldout = {normalize_title(Path(d).stem) for d, p in corpus.papers.items() if p.split == "heldout"}
            heldout |= {normalize_title(t) for t in cfg.get("heldout_titles", [])}
            corpus.blocked = {d for d in corpus.papers if normalize_title(Path(d).stem) in heldout}
        return corpus

    def visible(self, split: str | None = None) -> list[Paper]:
        return [p for d, p in self.papers.items()
                if d not in self.blocked and (split is None or p.split == split)]

    def leaked_duplicates(self) -> list[str]:
        """Non-held-out files that duplicate a held-out paper (blocked, but worth fixing at the source)."""
        return sorted(d for d in self.blocked if self.papers[d].split != "heldout")

    def get(self, doc_id: str) -> Paper:
        if doc_id in self.blocked:
            raise PermissionError(f"{doc_id!r} is held out for evaluation and cannot be read.")
        if doc_id not in self.papers:
            raise KeyError(f"Unknown doc_id {doc_id!r}; use list_papers.")
        return self.papers[doc_id]

    # Small BM25 over paragraphs: enough for ~50 papers, no index or embedder needed.
    def search(self, query: str, k: int = 8, split: str | None = None) -> list[dict]:
        docs = [(p, i, para) for p in self.visible(split) for i, para in enumerate(p.paragraphs)]
        if not docs:
            return []
        toks = [_WORD.findall(para.lower()) for _, _, para in docs]
        df = Counter(t for ts in toks for t in set(ts))
        avg = sum(map(len, toks)) / len(toks)
        q = _WORD.findall(query.lower())
        scored = []
        for (paper, i, para), ts in zip(docs, toks):
            tf = Counter(ts)
            s = sum(math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
                    * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(ts) / avg))
                    for t in q if t in tf)
            if s > 0:
                scored.append((s, paper, i, para))
        scored.sort(key=lambda x: -x[0])
        return [{"doc_id": p.doc_id, "split": p.split, "paragraph": i,
                 "score": round(s, 2), "text": para[:1200]} for s, p, i, para in scored[:k]]
