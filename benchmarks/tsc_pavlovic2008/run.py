"""Fitter-only benchmark: trapfit TSC deconvolution vs the published SIMPA fit (Pavlovic et al. 2008, Table I).

Inputs (data/): the two digitized envelope curves (beta = 0.1 and 0.5 K/s) and Table I (Ea, Tm per trap).
Each published trap is paired one-to-one with a fitted peak by Tm (within TM_TOL_K); unpaired traps are
reported as missed. Filled markers: peaks trapfit did not flag; open: flagged (overlapped, weak, poorly
constrained, at bound). No LLM and no defect assignments are involved: this paper is a held-out evaluation
paper, so only the fitter is scored on it, and these files live outside the agent's corpus.
Run:  uv run python benchmarks/tsc_pavlovic2008/run.py
Writes results.csv and fig_ea_parity.{pdf,png,svg} next to this script.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from defective import peaks

HERE = Path(__file__).resolve().parent
CURVES = {0.1: "Figure1_envelope_beta0.1Ks.csv", 0.5: "Figure2_envelope_beta0.5Ks.csv"}
TM_TOL_K = 8.0
MAX_PEAKS = 13  # the published fit uses 13 sub-peaks


def run():
    table = pd.read_csv(HERE / "data" / "table1_simpa.csv")
    rows = []
    for beta, name in CURVES.items():
        res = peaks.fit_curve(HERE / "data" / name, "TSC", beta, max_peaks=MAX_PEAKS)
        fitted = res.get("traps", [])
        Tms = table[f"Tm_K_beta{beta}"].tolist()
        # One-to-one pairing: closest (published, fitted) Tm pairs first, each peak used at most once.
        pairs = sorted((abs(f["peak_T_K"] - Tm), i, j) for i, Tm in enumerate(Tms) for j, f in enumerate(fitted))
        match, used = {}, set()
        for d, i, j in pairs:
            if d <= TM_TOL_K and i not in match and j not in used:
                match[i] = fitted[j]
                used.add(j)
        for i, t in table.iterrows():
            Tm, Ea = t[f"Tm_K_beta{beta}"], t[f"Ea_eV_beta{beta}"]
            near = match.get(i)
            paired = near is not None
            rows.append({"beta_K_per_s": beta, "trap": t["trap"], "Tm_published_K": Tm, "Ea_published_eV": Ea,
                         "Tm_fit_K": near["peak_T_K"] if paired else None,
                         "Ea_fit_eV": near["energy_eV"] if paired else None,
                         "flags": ";".join(near.get("flags", [])) if paired else "missed",
                         "n_fitted_peaks": len(fitted), "fit_status": res["status"]})
    df = pd.DataFrame(rows)
    df["Ea_err_eV"] = df["Ea_fit_eV"] - df["Ea_published_eV"]
    df.to_csv(HERE / "results.csv", index=False)
    return df


def plot(df):
    plt.style.use(os.path.expanduser("~/.claude/figure-style.mplstyle"))
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    lim = (0, 1.7)
    ax.plot(lim, lim, color="0.6", lw=0.8, zorder=0)
    for (beta, g), m in zip(df.dropna(subset=["Ea_fit_eV"]).groupby("beta_K_per_s"), ("o", "s")):
        clean = g["flags"] == ""
        line, = ax.plot(g.loc[clean, "Ea_published_eV"], g.loc[clean, "Ea_fit_eV"], m, ms=4, label=f"β = {beta} K/s")
        ax.plot(g.loc[~clean, "Ea_published_eV"], g.loc[~clean, "Ea_fit_eV"], m, ms=4, mfc="none",
                color=line.get_color(), label=f"β = {beta} K/s, flagged")
    ax.set(xlim=lim, ylim=lim, xlabel="published Ea, SIMPA (eV)", ylabel="fitted Ea, trapfit (eV)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    base = str(HERE / "fig_ea_parity")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".svg", format="svg")


if __name__ == "__main__":
    df = run()
    plot(df)
    for beta, g in df.groupby("beta_K_per_s"):
        ok = g.dropna(subset=["Ea_fit_eV"])
        clean = ok[ok["flags"] == ""]
        print(f"beta={beta}: {g['n_fitted_peaks'].iloc[0]} peaks fitted, {len(ok)}/{len(g)} published traps paired "
              f"(|dTm| <= {TM_TOL_K} K); all paired: median |dEa| = {ok['Ea_err_eV'].abs().median():.3f} eV; "
              f"unflagged ({len(clean)}): median |dEa| = {clean['Ea_err_eV'].abs().median():.3f} eV, "
              f"within 0.05 eV: {(clean['Ea_err_eV'].abs() <= 0.05).sum()}")
