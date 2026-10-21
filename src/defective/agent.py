"""Single-loop agent (APEXA's OrchestratorAgent pattern, minus the beamline parts)."""
from __future__ import annotations

from typing import Callable

from .corpus import Corpus
from .providers import make_provider
from .tools import Toolbox

SYSTEM = """You are DEFECTIVE, an assistant that assigns measured deep-level traps in \
radiation-detector crystals (CsPbBr3, CdZnTe, CdZnTeSe) to candidate point defects.

Work from evidence, not recall:
- Rank candidates with match_trap and report its posterior; do not invent probabilities.
- Every number you state must come from a tool result; cite doc_id (and paragraph) for literature claims.
- An author's assignment that only cites DFT is not independent evidence; say so when you rely on one.
- State the band-edge reference (Ec/Ev) of every energy. If the trap's reference is unknown, say that it widens the candidate set.
- When the top candidates are not clearly separated, call propose_experiment and recommend the experiment with the highest expected information gain, naming what each outcome would imply.
- If the DFT table has no records for the material, say so plainly instead of guessing.
Held-out evaluation papers are blocked by the tools; do not try to work around that."""

MAX_STEPS = 20


class Agent:
    def __init__(self, provider: str | None = None, model: str | None = None, corpus: Corpus | None = None,
                 on_tool: Callable[[str, dict, str, bool], None] | None = None):
        self.toolbox = Toolbox(corpus or Corpus.load())
        self.llm = make_provider(provider, SYSTEM, self.toolbox.specs, model)
        self.on_tool = on_tool

    def ask(self, text: str) -> str:
        turn = self.llm.send_user(text)
        for _ in range(MAX_STEPS):
            if not turn.tool_calls:
                return turn.text
            results = []
            for call in turn.tool_calls:
                out, err = self.toolbox.call(call.name, call.arguments)
                if self.on_tool:
                    self.on_tool(call.name, call.arguments, out, err)
                results.append((call, out, err))
            turn = self.llm.send_tool_results(results)
        return turn.text + f"\n\n[stopped after {MAX_STEPS} tool rounds]"
