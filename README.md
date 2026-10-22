# DEFECTIVE

**D**efect **E**vidence **F**rom **E**xperiments, **C**alculations, and **T**ext:
**I**nferring **V**acancies and **E**lectronic levels.
A project for the [OSI LLM Hackathon](https://llmhackathon.github.io/), Oct 21–22, 2026.

## Problem

Deep-level traps control charge collection and stability in radiation-detector
crystals such as CsPbBr3 and CdZnTe. The field still disagrees about what those traps are:

- Experimental papers report activation energies and capture cross-sections.
- DFT papers report charge-transition levels.
- The two sets of numbers are rarely compared systematically.
- Many experimental assignments simply cite DFT.
- The metadata needed to compare numbers (band-edge reference, functional,
  chemical-potential conditions) is buried in the text.

DEFECTIVE builds a source-linked dataset of reported traps and DFT defect levels.
On top of it, an agent matches a measured trap to candidate defects. The agent returns:

- ranked assignments with stated probabilities,
- the evidence behind each assignment,
- the experiment that would best separate the top candidates.

## Team

| Member | Focus |
|---|---|
| Mustafa | Experimental trap schema; independent vs. DFT-inferred labeling; probabilistic matcher; evaluation and calibration; growth-condition predictions |
| Akhil | DFT harmonization (band-edge reference, functional, SOC, correction scheme); per-method uncertainty widths; extracted DFT records |
| Pawan Tripathi | Repo and data layout; agent that proposes the next discriminating experiment; demo interface |
| Yunkai Sun | Docling parsing of the paper corpus |

## How it works

```
Docling markdown (Box) ──► Corpus (BM25 search, held-out guard)
                              │
          extract (LLM) ──────┤──► data/extracted/traps.<provider>.jsonl
                              │
DFT papers ─ extract-dft (LLM)┤──► data/extracted/dft_levels{,.llm}.jsonl
                              ▼
                 Agent (single tool-use loop, any LLM backend)
   list_papers · search_papers · read_passage · query_dft_levels
   match_trap  (Gaussian likelihood, σ = √(σ_exp² + σ_DFT²), Ec/Ev conversion)
   propose_experiment  (expected information gain over candidate attributes)
   fit_trap_curve  (trapfit: TSC deconvolution, DLTS/PICTS Arrhenius, Laplace DLTS)
   verify_defect_formation  (optional MACE-MH-1 formation-energy check)
```

- **Matcher** (`src/defective/matcher.py`):
  - Scores each DFT level against the measured trap energy, using
    `σ = hypot(σ_exp, σ_DFT)`. The defaults are 0.05 and 0.15 eV.
  - Converts between Ec and Ev references using the calculation's band gap.
  - Returns a normalized posterior over the DFT levels.
- **Experiment design**:
  - Each experiment in `configs/experiments.yaml` splits the candidates on one DFT
    attribute: favored growth condition, donor/acceptor character, defect class, or sublattice.
  - The agent recommends the split with the largest expected information gain.
- **Held-out guard**:
  - Papers in `Test Data/` are blocked at the tool boundary for list, search and read.
  - So are their duplicates elsewhere in the corpus, matched by normalized title.
  - The LLM cannot reach the evaluation set.

## Status (Oct 22, 2026)

What works, and what does not yet:

| Piece | State |
|---|---|
| Agent, tools, held-out guard, multi-provider backends | working; offline tests pass in CI |
| Corpus | 25 DFT, 35 dev and 6 held-out papers as Docling markdown (local only, copyrighted) |
| DFT table | **7 hand-checked seed levels** (`dft_levels.jsonl`, quoted in experimental papers) plus **57 LLM-extracted levels from 13 of the 25 DFT papers** (`dft_levels.llm.jsonl`, `defective extract-dft`). Every extracted level is quote-verified against its paper, carries its method (functional, SOC, correction, gap) and is tagged `llm_extracted_unreviewed`. Coverage: mostly CdTe/CdTeSe/ZnTe (49), CsPbBr3 (3), none for CdZnTe itself. Rejected candidates are in `dft_levels.llm.rejected.jsonl`. Not yet reviewed by a DFT expert; uncertainty widths are still the 0.15 eV default. |
| Matcher and experiment design | working, on the seed table above. Its posteriors are only as good as that table. |
| Curve fitting (trapfit) | working; benchmarked on one published TSC fit (see [Benchmarks](#benchmarks)) |
| MACE formation-energy check | working on Python < 3.15 (needs torch) |
| Held-out evaluation | pipeline and metrics are implemented and tested on fixtures. **No held-out results yet:** the gold file `data/eval/heldout_traps.jsonl` has not been built. |

## Setup

Requires Python 3.15 and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:pawantr/osi-defective.git && cd osi-defective
uv sync --all-extras        # uv downloads Python 3.15 if needed
cp .env.example .env        # add your key(s) and the corpus path
uv run pytest               # offline tests, no LLM calls
uv run defective ui         # demo at http://127.0.0.1:7860
```

The demo has to run from a clone. It reads `data/` and `configs/` from the repo, so
`uvx --from git+…` cannot launch it. The repo is private, so ask for collaborator access first.
What each part of the demo needs:

| Part | Needs |
|---|---|
| Match a trap, Fit a curve | nothing beyond `uv sync` (no key, no papers) |
| Agent chat | `DEFECTIVE_CORPUS_DIR` plus one LLM provider (Argo: ANL network/VPN) |
| MACE formation-energy check | the `dft` extra on Python < 3.15 (torch has no 3.15 wheels yet); otherwise it reports `mace_unavailable` |

Set `DEFECTIVE_CORPUS_DIR` to a folder holding `Experimental Data/*.md`, `DFT Papers/*.md`
and `Test Data/*.md` — Docling (or any other PDF/DOCX-to-markdown converter) output, kept
one directory per split exactly like the corresponding folder of source documents. **This
repo does not convert PDFs to markdown yet** — that conversion is a manual pre-step for
now; run your own Docling pass over `Experimental Data/`, `DFT Papers/` and `Test Data/`
and point `DEFECTIVE_CORPUS_DIR` at the resulting output (`corpus/` inside this repo is
gitignored and a convenient place to put it). The held-out titles are also listed in
`configs/corpus.yaml`, so the guard holds even when `Test Data/` is missing from a copy.
Papers are copyrighted, so they stay on Box and are not committed.

## Usage

```bash
uv run defective corpus                         # split sizes + held-out guard report
uv run defective ask "Which defect explains the 0.79 eV trap in CdZnTe?"
uv run defective chat                           # interactive session
uv run defective extract --split dev --limit 3  # LLM-extract trap records
uv run defective extract-dft                    # LLM-extract DFT levels (quote-verified)
uv run defective ui                             # Gradio demo (http://127.0.0.1:7860)
uv run defective eval                           # four-condition evaluation on held-out gold traps
```

### Demo UI

- **Agent:** chat, with a live trace of every tool call (search → read → match → propose experiment).
  Provider and model can be switched from dropdowns.
- **Match a trap:** the probabilistic matcher over the DFT table, with no LLM. It returns ranked
  candidates and the experiments that best separate them.
- **Fit a curve:** upload a digitized TSC curve to deconvolve it into peaks, DLTS/I-DLTS/PICTS
  Arrhenius points, or a Laplace-DLTS transient. Axis conventions and the heating rate are set under
  *Advanced*. Pick a peak and send its energy to the matcher. *Export CSV* writes one row per
  (peak, candidate defect) with the fit, the posterior and any MACE check (`src/defective/report.py`).
  `data/examples/tsc_synthetic_two_peaks.csv` is a **synthetic** two-peak example, written by
  `scripts/make_example_curves.py`.

### Evaluation

`defective eval --gold data/eval/heldout_traps.jsonl` scores nearest-energy, LLM-alone,
LLM + DFT table, and the matcher. That gold file is not built yet (see [Status](#status-oct-22-2026)).
The pipeline is tested on `tests/fixtures/eval_gold.jsonl`.

- **Scored records:** only those with `assignment.basis == "independent"`. The rest are listed
  in `excluded.csv` with the reason.
- **Metrics:** top-1, top-3 and MRR for every condition. The matcher is also scored on the mean
  gold-defect posterior and the Brier score.
- **Output:** written to `runs/eval/<timestamp>/`, including an editable `metrics.{pdf,png,svg}`.

### Curve fitting (`src/defective/trapfit/`, via `peaks.py`)

- **Routing:** `trapfit/algorithms.json` picks the method for each technique and checks the
  required inputs. Every result is `ok`, `unresolved` or `refused`, with quality flags such as
  `overlapped`, `weak_peak` and `energy_poorly_constrained`. The fitter never forces a label.
- **TSC:** Kitis first-order and general-order glow peaks, fitted together in log10(current).
  The number of peaks is chosen by BIC.
- **DLTS / I-DLTS / PICTS:** rate-window peaks are linked across windows, and an Arrhenius
  regression gives Ea and the apparent cross-section.
- **Laplace DLTS:** a Tikhonov-regularized NNLS emission-rate spectrum. It is approximate; use
  CONTIN/FTIKREG for production work.
- **Agent tool:** the same fits are exposed as `fit_trap_curve`.

### Benchmarks

[`benchmarks/tsc_pavlovic2008/`](benchmarks/tsc_pavlovic2008/) compares the TSC fitter with the
published 13-trap SIMPA fit of Pavlović et al. (2008) on Mustafa's digitized envelopes. It tests
the fitter only, with no LLM and no defect labels. Run it with
`uv run python benchmarks/tsc_pavlovic2008/run.py`.

| β (K/s) | traps paired (\|ΔTm\| ≤ 8 K) | median \|ΔEa\|, all paired | median \|ΔEa\|, unflagged |
|---|---|---|---|
| 0.1 | 8 / 13 | 0.105 eV | 0.055 eV |
| 0.5 | 12 / 13 | 0.111 eV | 0.065 eV |

Peak positions are recovered well. Energies in a crowded envelope are not, and most of the bad
fits are flagged. Use the energies of unflagged peaks only.

### MACE defect screening (`src/defective/dft_screen/`)

`materials.json` sets out the supercells and the vacancy, substitution and interstitial defects
for each material, built from `data/structures/*.cif`. `verify_defect_formation` relaxes the bulk
and defect cells with MACE-MH-1 and reports the formation energy as supporting evidence. It is
never blended into the matcher posterior.

## LLM backends

Each vendor offers two efficient models, a `balanced` one and a `fast` one. Switch with
`--provider` / `--model` or with `DEFECTIVE_PROVIDER` / `DEFECTIVE_MODEL`:

| Provider | Key | balanced | fast |
|---|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | claude-sonnet-5-5 | claude-haiku-5-5 |
| `openai` | `OPENAI_API_KEY` | gpt-5.6-terra | gpt-5.6-luna |
| `gemini` | `GEMINI_API_KEY` | gemini-3.5-flash | gemini-3.1-flash-lite |
| `argo` (ANL network) | `ANL_USERNAME` | Claude Sonnet 5 | Claude Haiku 4.5 |

Argo also exposes `gpt-balanced`, `gpt-fast`, `gemini-balanced` and `gemini-fast`.
Example: `uv run defective --provider argo --model gpt-fast chat`.
All model ids live in `configs/models.yaml`.

## Layout

```
configs/    corpus splits, model registry, candidate experiments
schemas/    trap.schema.json (experimental), dft_level.schema.json (DFT) — drafts
src/defective/
  corpus.py     Docling markdown loader, BM25 search, held-out guard
  providers.py  Anthropic / OpenAI / Gemini wire formats (Argo = config)
  matcher.py    probabilistic matcher + information-gain experiment design
  tools.py      tool definitions exposed to the agent
  agent.py      tool-use loop and system prompt
  extract.py    LLM extraction of trap records with quote verification
  peaks.py      curve loading; delegates fits to trapfit/
  trapfit/      TSC, Arrhenius and Laplace-DLTS fitting routes (algorithms.json)
  dft_screen/   defect-cell builder + MACE-MH-1 formation-energy check
  report.py     CSV export (peak x candidate)
  evaluate.py   four-condition evaluation
  ui.py         Gradio demo
  cli.py        `defective` command
scripts/        example-data generator, seed-quote verifier
benchmarks/     fitter benchmarks against published fits
data/structures/ bulk CIFs for MACE screening
data/examples/  synthetic demo curves
data/extracted/ extracted records (JSONL)
tests/          offline tests
```

## Evaluation plan

Each held-out trap is given to the system without the authors' assignment.
The system sees only:

- material
- energy
- cross-section
- technique
- growth conditions

Predictions are scored only against assignments the authors supported independently,
through doping, stoichiometry or annealing evidence. Agreement with an assignment that
only cites DFT would be circular.

The four conditions compared:

1. Nearest-energy heuristic
2. LLM alone, no retrieval
3. LLM with the retrieved DFT table
4. Probabilistic matcher with harmonized uncertainties
