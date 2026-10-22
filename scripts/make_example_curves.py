"""Write the synthetic demo curves in data/examples/ (clearly synthetic, not from any paper).

TSC: two overlapping first-order peaks, the parameters below, 1% noise (seed 0).
Run:  uv run python scripts/make_example_curves.py
"""
from pathlib import Path

import numpy as np

from defective.peaks import tsc_peak

OUT = Path(__file__).resolve().parents[1] / "data/examples"
PEAKS = [(0.35, 168.0, 1.0), (0.42, 190.0, 0.6)]   # (Ea eV, Tm K, relative height)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = np.linspace(110, 260, 301)
    y = sum(tsc_peak(T, *p) for p in PEAKS)
    y = y + np.random.default_rng(0).normal(0, 0.01 * y.max(), T.size)
    path = OUT / "tsc_synthetic_two_peaks.csv"
    with path.open("w") as f:
        f.write("# SYNTHETIC demo curve. Ea_eV,Tm_K,height: " + "; ".join(map(str, PEAKS)) + "\nT_K,current_arb\n")
        f.writelines(f"{t:.2f},{v:.6g}\n" for t, v in zip(T, y))
    print(path)


if __name__ == "__main__":
    main()
