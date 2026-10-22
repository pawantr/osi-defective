"""Gradio demo: agent chat with a live tool trace, and a direct (no-LLM) trap matcher."""
from __future__ import annotations

import json
import os
import queue
import threading

import gradio as gr
import numpy as np
import pandas as pd

from . import matcher, peaks, report
from .providers import load_model_config

PROVIDERS = ["anthropic", "openai", "gemini", "argo"]
MATERIALS = ["CsPbBr3", "CdZnTe", "CdTe"]
CAND_COLS = ["rank", "defect", "transition", "level_eV", "reference", "delta_eV", "sigma_eV", "posterior", "source"]
PEAK_COLS = ["peak", "energy_eV", "energy_sigma_eV", "peak_T_K", "frequency_factor_per_s",
             "apparent_cross_section_cm2", "order_b", "flags"]
EXAMPLE_CURVE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "examples", "tsc_synthetic_two_peaks.csv")
EXP_COLS = ["experiment", "info_gain_bits", "cost", "discriminates_on", "outcome_groups", "description"]
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


def _show(v):
    """4 significant figures for display; the tools/JSON keep full precision."""
    return float(f"{v:.4g}") if isinstance(v, float) else v


def _frame(rows: list[dict], cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{c: _show(r.get(c)) for c in cols} for r in rows], columns=cols)


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
        e["info_gain_bits"] = e["expected_info_gain_bits"]
        e["outcome_groups"] = "; ".join(f"{k}: {', '.join(v)}" for k, v in e["outcome_groups"].items())
    ranked = [{**c, "rank": i + 1} for i, c in enumerate(cands)]
    skipped = res["skipped_no_common_reference"]
    msg = (f"{len(cands)} candidates from {len(levels)} DFT levels for {material} "
           f"(trap {energy_eV:.3f} eV vs {reference}). Experiments ranked over the top {len(top)} "
           f"(posterior entropy {abs(exps[0]['posterior_entropy_bits']) if exps else 0:.2f} bits). {res['note']}"
           + (f" Skipped (no common reference): {', '.join(skipped)}." if skipped else ""))
    return msg, _frame(ranked, CAND_COLS), _frame(exps, EXP_COLS)


def fit_table(path: str | None, technique: str, beta: float | None, x_kind: str, y_kind: str):
    """Curve tab: (status markdown, peaks frame, figure, first energy for the matcher)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not path:
        return "Upload a 2-column CSV (T [K], signal).", _frame([], PEAK_COLS), None, None
    try:
        res = peaks.fit_curve(path, technique, beta or None, x_kind=x_kind, y_kind=y_kind)
    except (ValueError, OSError) as e:
        return f"**Fit failed:** {e}", _frame([], PEAK_COLS), None, None
    if res["status"] != "ok":
        reason = res.get("reason", "unresolved")
        missing = f" (missing: {', '.join(res['missing'])})" if res.get("missing") else ""
        return f"**{res['status']}:** {reason}{missing}", _frame([], PEAK_COLS), None, None
    x, y = peaks.load_curve(path)
    traps = res.get("traps", [])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    if technique == "TSC":
        rows = [{**p, "peak": i + 1, "flags": ", ".join(p.get("flags", []))} for i, p in enumerate(traps)]
        ax.plot(x, y, ".", ms=3, color="0.4", label="data")
        if traps:
            T = np.linspace(x.min(), x.max(), 600)
            comps = [peaks.tsc_component(T, p) for p in traps]
            for i, c in enumerate(comps):
                ax.plot(T, c, "--", lw=1, label=f"peak {i + 1}: {traps[i]['energy_eV']:.3f} eV")
            ax.plot(T, np.sum(comps, axis=0), "k-", lw=1.2, label="fit")
        ax.set(xlabel="T (K)", ylabel="signal")
        f = res.get("fit", {})
        msg = (f"{len(traps)} {f.get('peak_model', 'Kitis')} peak(s), "
               f"rms log10 residual = {f.get('rms_log10_residual', float('nan')):.4f}.")
        ea = round(traps[0]["energy_eV"], 3) if traps else None
    else:
        T, e = peaks.to_rate(x, y, x_kind, y_kind)
        rows = [{**p, "peak": i + 1, "flags": ", ".join(p.get("flags", []))} for i, p in enumerate(traps)]
        ax.plot(1000 / T, np.log(e / T**2), "o", label="data")
        trap = traps[0] if traps else {}
        if "intercept_ln" in trap:
            xx = np.linspace(T.min(), T.max(), 50)
            ax.plot(1000 / xx, trap["intercept_ln"] - trap["energy_eV"] / (peaks.K_B * xx), "k-",
                    label=f"Ea = {trap['energy_eV']:.3f} eV")
        ax.set(xlabel="1000/T (1/K)", ylabel="ln(e/T²)")
        msg = (f"Arrhenius fit: Ea = {trap.get('energy_eV', float('nan')):.3f} eV "
               f"over {trap.get('n_windows', len(T))} points." if trap else "No trap resolved.")
        ea = round(trap["energy_eV"], 3) if "energy_eV" in trap else None
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    msg += " Energies have no band-edge reference until you choose one."
    return msg, _frame(rows, PEAK_COLS), fig, ea


def _row_energy(evt: gr.SelectData):
    return round(float(evt.row_value[PEAK_COLS.index("energy_eV")]), 3)


def export_report(material: str, technique: str, peaks_df, reference: str, error: float | None, top_k: int = 5):
    """Match every fitted peak (not just the one sent to the matcher) and write
    a final CSV: one row per peak x candidate defect, with peak-fit params,
    the candidate's energy/source/posterior, and no LLM involved."""
    if peaks_df is None or len(peaks_df) == 0:
        return None, "Fit a curve first -- no peaks to report."
    rows = peaks_df.to_dict("records")
    levels = matcher.dft_levels(material)
    candidates_by_peak = {}
    for row in rows:
        energy = row.get("energy_eV")
        if energy is None or (isinstance(energy, float) and np.isnan(energy)):
            continue
        trap = {"energy_eV": float(energy), "energy_reference": reference, "energy_err_eV": error or None}
        cands = matcher.match(trap, levels)["candidates"][:top_k]
        candidates_by_peak[row.get("peak")] = [{**c, "rank": i + 1} for i, c in enumerate(cands)]
    df = report.build_report(material, technique, rows, candidates_by_peak)
    path = report.write_report(df)
    return str(path), f"Wrote {len(df)} row(s) to `{path.name}`."


def build_ui(provider: str | None = None, model: str | None = None) -> gr.Blocks:
    cfg = load_model_config()
    provider = provider or os.environ.get("DEFECTIVE_PROVIDER") or cfg["default_provider"]
    models = models_for(provider)
    model = model or os.environ.get("DEFECTIVE_MODEL") or models[0]
    materials = MATERIALS
    # Demo default: CsPbBr3 has three competing levels near Ec, so the posterior and experiment ranking are non-trivial.
    default_mat = "CsPbBr3"

    with gr.Blocks(title="DEFECTIVE") as demo:
        gr.Markdown("# DEFECTIVE\nAssign measured deep-level traps to candidate defects, with evidence "
                    "and the next discriminating experiment.")
        with gr.Tabs() as tabs:
            with gr.Tab("Agent", id="agent"):
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
                # Clear the box first and stream into chat/trace only, so the textbox is never
                # an output of the long-running stream (Gradio locks outputs while they stream).
                pending_q = gr.State("")
                box.submit(lambda t: ("", t), box, [box, pending_q], queue=False).then(
                    respond, [pending_q, chat, trace, agent], [chat, trace, pending_q])

            with gr.Tab("Match a trap", id="match"):
                gr.Markdown("Direct probabilistic matcher over the DFT table; no LLM involved.")
                with gr.Row():
                    mat = gr.Dropdown(materials, value=default_mat, label="Material (composition)", allow_custom_value=True)
                    en = gr.Number(0.60, label="energy_eV", step=0.01)
                    ref = gr.Radio(["Ec", "Ev", "unspecified"], value="Ec", label="Reference")
                    err = gr.Number(matcher.DEFAULT_SIGMA_EXP_EV, label="error (eV)", step=0.01, minimum=0)
                go = gr.Button("Match", variant="primary")
                msg = gr.Markdown()
                cands = gr.Dataframe(headers=CAND_COLS, label="Ranked candidates", interactive=False)
                exps = gr.Dataframe(headers=EXP_COLS, label="Recommended experiments (by expected information gain)",
                                    interactive=False, wrap=True)
                go.click(match_table, [mat, en, ref, err], [msg, cands, exps])

            with gr.Tab("Fit a curve", id="fit"):
                gr.Markdown("Upload a digitized curve, state the composition and the measurement technique used -- "
                            "the fitting algorithm (TSC peak deconvolution, DLTS/PICTS Arrhenius regression, or "
                            "Laplace DLTS) is chosen for you from the technique. Then send a peak energy to the "
                            "matcher.")
                with gr.Row():
                    csv = gr.File(label="CSV (2 columns)", file_types=[".csv", ".txt", ".dat"], type="filepath",
                                  value=EXAMPLE_CURVE if os.path.exists(EXAMPLE_CURVE) else None)
                    with gr.Column():
                        fmat = gr.Dropdown(materials, value=default_mat, label="Material (composition)", allow_custom_value=True)
                        tech = gr.Radio(["TSC", "DLTS", "I-DLTS", "PICTS", "LAPLACE_DLTS"], value="TSC",
                                        label="Measurement technique")
                with gr.Accordion("Advanced (axis conventions / heating rate)", open=False):
                    beta = gr.Number(0.1, label="heating rate (K/s, TSC)", minimum=0)
                    xk = gr.Dropdown(["T", "1000/T", "1/T"], value="T", label="x axis (Arrhenius)")
                    yk = gr.Dropdown(["rate", "tau", "ln(e/T^2)", "ln(tau*T^2)"], value="rate",
                                     label="y axis (Arrhenius)")
                fit = gr.Button("Fit", variant="primary")
                fmsg = gr.Markdown()
                plot = gr.Plot(label="Fit")
                ptab = gr.Dataframe(headers=PEAK_COLS, label="Peaks", interactive=False)
                with gr.Row():
                    pick = gr.Number(label="Ea to match (eV)")
                    fref = gr.Radio(["Ec", "Ev", "unspecified"], value="unspecified", label="Reference")
                    send = gr.Button("Match this energy")
                with gr.Row():
                    export = gr.Button("Export CSV (all peaks x candidates)")
                    report_file = gr.File(label="Report", interactive=False)
                expmsg = gr.Markdown()
                fit.click(fit_table, [csv, tech, beta, xk, yk], [fmsg, ptab, plot, pick])
                ptab.select(_row_energy, None, pick)
                send.click(match_table, [fmat, pick, fref, err], [msg, cands, exps]).then(
                    lambda: gr.Tabs(selected="match"), None, tabs)
                export.click(export_report, [fmat, tech, ptab, fref, err], [report_file, expmsg])
    return demo


def launch(provider: str | None = None, model: str | None = None, port: int | None = None):
    build_ui(provider, model).launch(server_port=port)
