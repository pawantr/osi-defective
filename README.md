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
DFT levels (Akhil) ───────────┤──► data/extracted/dft_levels.jsonl
                              ▼
                 Agent (single tool-use loop, any LLM backend)
   list_papers · search_papers · read_passage · query_dft_levels
   match_trap  (Gaussian likelihood, σ = √(σ_exp² + σ_DFT²), Ec/Ev conversion)
   propose_experiment  (expected information gain over candidate attributes)
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

## Setup

Requires Python 3.15 and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:pawantr/osi-defective.git && cd osi-defective
uv sync --all-extras
cp .env.example .env        # add your key(s) and the corpus path
uv run pytest               # offline tests, no LLM calls
```

Set `DEFECTIVE_CORPUS_DIR` to the Docling output folder
(`Hackathon_202610_ANL/docling_pdf/output` on Box). If Box Drive is slow, copy it locally:
`rsync -a --include='*/' --include='*.md' --exclude='*' <box>/output/ corpus/`
(`corpus/` is gitignored). The held-out titles are also listed in `configs/corpus.yaml`, so the
guard holds even when `Test Data/` is missing from a copy. Papers are copyrighted, so they
stay on Box and are not committed.

## Usage

```bash
uv run defective corpus                         # split sizes + held-out guard report
uv run defective ask "Which defect explains the 0.79 eV trap in CdZnTe?"
uv run defective chat                           # interactive session
uv run defective extract --split dev --limit 3  # LLM-extract trap records
uv run defective ui                             # Gradio demo (http://127.0.0.1:7860)
uv run defective eval                           # four-condition evaluation on held-out gold traps
```

### Demo UI

- **Agent:** chat, with a live trace of every tool call (search → read → match → propose experiment).
  Provider and model can be switched from dropdowns.
- **Match a trap:** the probabilistic matcher over the DFT table, with no LLM. It returns ranked
  candidates and the experiments that best separate them.
- **Fit a curve:** upload a digitized TSC curve to deconvolve it into first-order peaks, or upload
  DLTS/PICTS Arrhenius points. Pick a peak and send its energy to the matcher.
  `data/examples/tsc_synthetic_two_peaks.csv` is a **synthetic** two-peak example, written by
  `scripts/make_example_curves.py`.

### Evaluation

`defective eval --gold data/eval/heldout_traps.jsonl` scores nearest-energy, LLM-alone,
LLM + DFT table, and the matcher.

- **Scored records:** only those with `assignment.basis == "independent"`. The rest are listed
  in `excluded.csv` with the reason.
- **Metrics:** top-1, top-3 and MRR for every condition. The matcher is also scored on the mean
  gold-defect posterior and the Brier score.
- **Output:** written to `runs/eval/<timestamp>/`, including an editable `metrics.{pdf,png,svg}`.

### Curve fitting (`src/defective/peaks.py`)

- **TSC:** a sum of first-order (Randall–Wilkins) peaks, with the number of peaks chosen by BIC.
  Initial-rise and Chen peak-shape estimates serve as fallbacks.
- **DLTS/PICTS:** an Arrhenius fit of ln(e/T²) against 1/T gives Ea and the apparent σ.
  m*/m0 is a parameter.
- **Agent tool:** the same fits are exposed as `fit_trap_curve`.
- **Caveats:** first-order kinetics, no retrapping. Errors are statistical only.
  Convert log-scale digitized currents to linear before fitting.

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
  peaks.py      TSC deconvolution, DLTS/PICTS Arrhenius fits
  evaluate.py   four-condition evaluation
  ui.py         Gradio demo
  cli.py        `defective` command
scripts/        example-data generator, seed-quote verifier
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
