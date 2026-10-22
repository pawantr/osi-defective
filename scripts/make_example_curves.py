"""Write the synthetic demo curves in data/examples/ (clearly synthetic, not from any paper).

TSC: two overlapping first-order (Kitis) peaks, the parameters below, 2% multiplicative noise (seed 0).
Run:  uv run python scripts/make_example_curves.py
"""
from pathlib import Path

import numpy as np

from defective.trapfit.tsc import peak_first_order

OUT = Path(__file__).resolve().parents[1] / "data/examples"
PEAKS = [(0.35, 168.0, 1.0), (0.42, 190.0, 0.6)]   # (Ea eV, Tm K, relative height)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    T = np.linspace(110, 260, 301)
    y = sum(peak_first_order(T, h, Tm, Ea) for Ea, Tm, h in PEAKS)
    # Multiplicative noise keeps the current positive: trapfit fits log10(current).
    y = y * (1 + 0.02 * np.random.default_rng(0).normal(0, 1, T.size))
    path = OUT / "tsc_synthetic_two_peaks.csv"
    with path.open("w") as f:
        f.write("# SYNTHETIC demo curve. Ea_eV,Tm_K,height: " + "; ".join(map(str, PEAKS)) + "\nT_K,current_arb\n")
        f.writelines(f"{t:.2f},{v:.6g}\n" for t, v in zip(T, y))
    print(path)


if __name__ == "__main__":
    main()
