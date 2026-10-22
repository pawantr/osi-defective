"""Gradio demo: agent chat with a live tool trace, and a direct (no-LLM) trap matcher."""
from __future__ import annotations

import json
import os
import queue
import threading

import gradio as gr
import pandas as pd

from . import matcher
from .providers import load_model_config

PROVIDERS = ["anthropic", "openai", "gemini", "argo"]
MATERIALS = ["CsPbBr3", "CdZnTe", "CdZnTeSe"]
CAND_COLS = ["rank", "defect", "transition", "level_eV", "reference", "delta_eV", "sigma_eV", "posterior", "source"]
EXP_COLS = ["experiment", "expected_info_gain_bits", "cost", "discriminates_on", "outcome_groups", "description"]
EXAMPLES = ["Which defect explains the 0.79 eV trap in CdZnTe?",
            "A TSC trap at Ec-0.45 eV (±0.03) in CsPbBr3: rank candidates and propose the next experiment."]
_corpus = None


def models_for(provider: str) -> list[str]:
    return list(load_model_config()["providers"][provider]["models"])


def build_agent(provider: str, model: str):
    """Return (agent | None, status markdown). Never raises: missing keys/corpus become a message."""
    global _corpus
    from .agent import Agent
    from .corpus import Corpus
    try:
        _corpus = _corpus or Corpus.load()
        agent = Agent(provider, model or None, _corpus)
    except (RuntimeError, FileNotFoundError, ValueError, ImportError) as e:
        return None, f"**Agent unavailable:** {e}"
    return agent, f"Connected: `{agent.llm.cfg['provider']} / {agent.llm.cfg['model']}`"


def _trace_msg(name: str, args: dict, out: str, err: bool) -> dict:
    try:
        out = json.dumps(json.loads(out), indent=1, ensure_ascii=False)
    except ValueError:
        pass
    out = out if len(out) < 4000 else out[:4000] + "\n... [truncated]"
    return {"role": "assistant", "content": f"```json\n{out}\n```",
            "metadata": {"title": f"{name}({json.dumps(args, ensure_ascii=False)[:100]})" + (" ERROR" if err else ""),
                         "status": "done"}}


def respond(text: str, history: list, trace: list, agent):
    text = (text or "").strip()
    if not text:
        yield history, trace, ""
        return
    history = history + [{"role": "user", "content": text}]
    if agent is None:
        yield history + [{"role": "assistant", "content": "No agent: fix the provider/key shown above."}], trace, ""
        return
    trace = trace + [{"role": "user", "content": text}]
    events: queue.Queue = queue.Queue()
    agent.on_tool = lambda *a: events.put(("tool", a))

    def run():
        try:
            events.put(("done", agent.ask(text)))
        except Exception as e:  # surface API/network errors in the chat instead of killing the session
            events.put(("done", f"**Error:** {type(e).__name__}: {e}"))

    threading.Thread(target=run, daemon=True).start()
    pending = history + [{"role": "assistant", "content": "_working..._"}]
    yield pending, trace, ""
    while True:
        kind, payload = events.get()
        if kind == "tool":
            trace = trace + [_trace_msg(*payload)]
            yield pending, trace, ""
        else:
            yield history + [{"role": "assistant", "content": payload}], trace, ""
            return


def _frame(rows: list[dict], cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in rows], columns=cols)


def match_table(material: str, energy_eV: float | None, reference: str, error: float | None, top_k: int = 5):
    """Direct matcher call: (status markdown, candidates frame, experiments frame)."""
    empty = _frame([], CAND_COLS), _frame([], EXP_COLS)
    if not material or energy_eV is None:
        return "Enter a material and an energy.", *empty
    levels = matcher.dft_levels(material)
    if not levels:
        n = len(matcher.dft_levels())
        why = "the DFT table (data/extracted/dft_levels.jsonl) is empty" if n == 0 else \
            f"the DFT table has {n} records but none for {material!r}"
        return f"**No candidates:** {why}. Nothing to match against.", *empty
    trap = {"energy_eV": float(energy_eV), "energy_reference": reference, "energy_err_eV": error or None}
    res = matcher.match(trap, levels)
    cands = res["candidates"]
    if not cands:
        return (f"**No candidates:** none of the {len(levels)} {material} levels can be put on the "
                f"{reference} reference (missing band gap)."), *empty
    top = cands[:top_k]
    z = sum(c["posterior"] for c in top) or 1.0
    top = [{**c, "posterior": c["posterior"] / z} for c in top]
    exps = matcher.propose_experiments(top, levels)
    for e in exps:
        e["outcome_groups"] = "; ".join(f"{k}: {', '.join(v)}" for k, v in e["outcome_groups"].items())
    ranked = [{**c, "rank": i + 1} for i, c in enumerate(cands)]
    skipped = res["skipped_no_common_reference"]
    msg = (f"{len(cands)} candidates from {len(levels)} DFT levels for {material} "
           f"(trap {energy_eV} eV vs {reference}). Experiments ranked over the top {len(top)} "
           f"(posterior entropy {exps[0]['posterior_entropy_bits'] if exps else 0} bits). {res['note']}"
           + (f" Skipped (no common reference): {', '.join(skipped)}." if skipped else ""))
    return msg, _frame(ranked, CAND_COLS), _frame(exps, EXP_COLS)


def build_ui(provider: str | None = None, model: str | None = None) -> gr.Blocks:
    cfg = load_model_config()
    provider = provider or os.environ.get("DEFECTIVE_PROVIDER") or cfg["default_provider"]
    models = models_for(provider)
    model = model or os.environ.get("DEFECTIVE_MODEL") or models[0]
    materials = sorted({r["material"] for r in matcher.dft_levels()}) or MATERIALS

    with gr.Blocks(title="DEFECTIVE") as demo:
        gr.Markdown("# DEFECTIVE\nAssign measured deep-level traps to candidate defects, with evidence "
                    "and the next discriminating experiment.")
        with gr.Tab("Agent"):
            agent = gr.State(None)
            with gr.Row():
                prov = gr.Dropdown(PROVIDERS, value=provider, label="Provider")
                mod = gr.Dropdown(models, value=model, label="Model", allow_custom_value=True)
            status = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=3):
                    chat = gr.Chatbot(label="Chat", height=520)
                    box = gr.Textbox(placeholder="Ask about a trap...", show_label=False, submit_btn=True)
                    gr.Examples(EXAMPLES, inputs=box)
                    clear = gr.Button("New conversation", size="sm")
                with gr.Column(scale=2):
                    with gr.Accordion("Agent trace", open=True):
                        trace = gr.Chatbot(label="Tool calls", height=560, feedback_options=None)

            def rebuild(p, m):
                a, s = build_agent(p, m)
                return a, s, [], []

            def on_provider(p):
                ms = models_for(p)
                return gr.update(choices=ms, value=ms[0])

            outs = [agent, status, chat, trace]
            # .input (user edits only) so the provider's programmatic model reset doesn't rebuild twice.
            prov.input(on_provider, prov, mod).then(rebuild, [prov, mod], outs)
            mod.input(rebuild, [prov, mod], outs)
            clear.click(rebuild, [prov, mod], outs)
            demo.load(rebuild, [prov, mod], outs)
            box.submit(respond, [box, chat, trace, agent], [chat, trace, box])

        with gr.Tab("Match a trap"):
            gr.Markdown("Direct probabilistic matcher over the DFT table; no LLM involved.")
            with gr.Row():
                mat = gr.Dropdown(materials, value=materials[0], label="Material", allow_custom_value=True)
                en = gr.Number(0.79, label="energy_eV", step=0.01)
                ref = gr.Radio(["Ec", "Ev", "unspecified"], value="unspecified", label="Reference")
                err = gr.Number(matcher.DEFAULT_SIGMA_EXP_EV, label="error (eV)", step=0.01, minimum=0)
            go = gr.Button("Match", variant="primary")
            msg = gr.Markdown()
            cands = gr.Dataframe(headers=CAND_COLS, label="Ranked candidates", interactive=False)
            exps = gr.Dataframe(headers=EXP_COLS, label="Recommended experiments (by expected information gain)",
                                interactive=False, wrap=True)
            go.click(match_table, [mat, en, ref, err], [msg, cands, exps])
    return demo


def launch(provider: str | None = None, model: str | None = None, port: int | None = None):
    build_ui(provider, model).launch(server_port=port)
