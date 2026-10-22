# TSC fitter benchmark: Pavlović et al. 2008 (CdZnTe)

Scores only the curve fitter (trapfit TSC deconvolution) against a published SIMPA fit. No LLM and no
defect assignments are involved.

- `data/Figure*_envelope_*.csv`: the published total-fit envelopes at β = 0.1 and 0.5 K/s, digitized from
  Figs. 1–2 by Mustafa (pixel tracing; QC overlays on Box under `Defective/TSC_Dataset/`).
- `data/table1_simpa.csv`: Ea and Tm of the 13 published traps (Table I of the paper).
- `uv run python benchmarks/tsc_pavlovic2008/run.py` writes `results.csv` and `fig_ea_parity.{pdf,png,svg}`.

This paper is in the **held-out** evaluation set (Test Data). These files stay outside the agent's corpus
and contain no defect labels, so the held-out trap-assignment evaluation is unaffected. Do not point the
agent at them.

## Result (current trapfit)

| β (K/s) | peaks fitted | published traps paired (\|ΔTm\| ≤ 8 K) | median \|ΔEa\|, all paired | median \|ΔEa\|, unflagged |
|---|---|---|---|---|
| 0.1 | 10 | 8 / 13 | 0.105 eV | 0.055 eV (4 peaks) |
| 0.5 | 12 | 12 / 13 | 0.111 eV | 0.065 eV (5 peaks) |

Peak positions are recovered well. The energies of a 13-peak envelope are not: many sets of overlapping
peaks fit the same envelope, and trapfit flags most of the bad fits itself. On real overlapping TSC data,
use the energies of unflagged peaks and treat flagged peaks as positions only.
