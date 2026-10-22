"""`defective` command-line entry point."""
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(prog="defective")
    ap.add_argument("--provider", choices=["anthropic", "openai", "gemini", "argo"])
    ap.add_argument("--model", help="alias from configs/models.yaml (balanced, fast, ...) or a model id")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("corpus", help="Summarize the corpus and the held-out guard")
    sub.add_parser("chat", help="Interactive agent session")
    x = sub.add_parser("extract", help="LLM-extract trap records from a corpus split")
    x.add_argument("--split", default="dev", choices=["dev", "dft"])
    x.add_argument("--limit", type=int)
    xd = sub.add_parser("extract-dft", help="LLM-extract DFT transition levels from the DFT papers")
    xd.add_argument("--limit", type=int)
    xd.add_argument("--workers", type=int, default=4)
    q = sub.add_parser("ask", help="One question to the agent")
    q.add_argument("question")
    u = sub.add_parser("ui", help="Launch the Gradio demo (needs the `ui` extra)")
    u.add_argument("--port", type=int)
    from .evaluate import CONDITIONS, DEFAULT_DFT, DEFAULT_GOLD
    ev = sub.add_parser("eval", help="Score the four comparison conditions on held-out gold traps")
    ev.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS),
                    help="subset to run (default: all four)")
    ev.add_argument("--gold", default=str(DEFAULT_GOLD), help="JSONL of gold trap records")
    ev.add_argument("--dft", default=DEFAULT_DFT, help="JSONL of DFT level records (default: all tables)")
    ev.add_argument("--out", help="output dir (default runs/eval/<timestamp>)")
    ev.add_argument("--limit", type=int, help="score only the first N gold records")
    ev.add_argument("--no-plot", action="store_true", help="skip the metrics figure")
    args = ap.parse_args()

    if args.cmd == "eval":   # needs no corpus: gold records and the DFT table are files
        from .evaluate import main as run_eval
        try:
            run_eval(args.gold, args.conditions, args.provider, args.model, args.dft,
                     args.out, args.limit, plot=not args.no_plot)
        except (FileNotFoundError, RuntimeError, ValueError) as err:
            raise SystemExit(f"defective eval: {err}")
        return

    if args.cmd == "ui":
        try:
            from .ui import launch
        except ImportError as e:
            raise SystemExit(f"{e}\nInstall the UI extra: uv sync --extra ui")
        launch(args.provider, args.model, args.port)
        return

    from .corpus import Corpus
    corpus = Corpus.load()
    if args.cmd == "corpus":
        for split in ("dft", "dev", "heldout"):
            n = sum(1 for p in corpus.papers.values() if p.split == split)
            print(f"{split:8s} {n:3d} papers")
        print(f"blocked  {len(corpus.blocked):3d} files (held-out + duplicates)")
        for d in corpus.leaked_duplicates():
            print(f"  WARNING held-out paper duplicated outside Test Data: {d}")
        return

    if args.cmd == "extract":
        from .extract import extract
        print(extract(args.split, args.provider, args.model, args.limit))
        return

    if args.cmd == "extract-dft":
        from .extract_dft import extract_dft
        print(extract_dft(args.provider, args.model, args.limit, args.workers))
        return

    from .agent import Agent

    def show(name, a, out, err):
        print(f"  -> {name}({json.dumps(a)[:120]}){' ERROR' if err else ''}")

    agent = Agent(args.provider, args.model, corpus, on_tool=show)
    print(f"[{agent.llm.cfg['provider']} / {agent.llm.cfg['model']}]")
    if args.cmd == "ask":
        print(agent.ask(args.question))
        return
    while True:
        try:
            text = input("DEFECTIVE> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in {"quit", "exit"}:
            break
        if text:
            print(agent.ask(text))


if __name__ == "__main__":
    main()
