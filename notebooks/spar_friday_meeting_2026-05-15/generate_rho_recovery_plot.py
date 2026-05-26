"""Per-seed posterior P(R = 1 | y) vs truth R, all 4 systems, 10 synthetic seeds.

Reads the per-seed CSV produced by extract_rho_recovery.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(OUT_DIR / "rho_recovery_per_seed.csv")

SYSTEM_ORDER = ["Human", "Chicken", "LLMs (2024)", "ELIZA"]
SYSTEM_TRUTH = {"Human": [1], "Chicken": [0, 1], "LLMs (2024)": [0, 1], "ELIZA": [0]}

fig, axes = plt.subplots(1, 4, figsize=(11, 4.4), sharey=True)
np_rng = np.random.default_rng(0)

for ax, sys_label in zip(axes, SYSTEM_ORDER):
    sub = df[df["system"] == sys_label]
    truth_values = SYSTEM_TRUTH[sys_label]
    for truth_R in truth_values:
        cases = sub[sub["truth_R"] == truth_R]
        rhos = cases["post_mean_rho_collapsed"].values
        x = np.full(len(rhos), truth_R, dtype=float) + np_rng.uniform(-0.08, 0.08, len(rhos))
        colour = "#1f77b4" if truth_R == 1 else "#d62728"
        ax.scatter(x, rhos, s=70, color=colour, alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(sys_label, fontsize=11)
    ax.set_xticks(truth_values)
    ax.set_xticklabels([f"truth R = {r}" for r in truth_values], fontsize=9)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("posterior P(R = 1 | data),  per seed")
fig.suptitle(
    "Recovery on 10 synthetic seeds: posterior P(R = 1 | data) vs true root state, per system\n"
    "(targeted-fix variant; one dot per synthetic seed; horizontal line at 0.5)",
    fontsize=11,
)
fig.tight_layout()

out = OUT_DIR / "rho_recovery_dotplot.png"
fig.savefig(out, dpi=150)
print(f"saved: {out}")
