"""Plot prior-implied q_j vs depth for Human (C=0.999) and ELIZA (C=0.001)
under each candidate prior. Reads eval/analytical_qj_by_prior_depth.csv.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "eval" / "analytical_qj_by_prior_depth.csv"
OUT = HERE / "figs" / "qj_vs_depth.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(CSV)
    priors = list(dict.fromkeys(df["prior"]))
    fig, axes = plt.subplots(1, len(priors), figsize=(4.0 * len(priors), 4.0), sharey=True)
    if len(priors) == 1:
        axes = [axes]
    for ax, prior in zip(axes, priors):
        sub = df[df["prior"] == prior]
        for system, marker, colour in [("Human", "o", "tab:blue"), ("ELIZA", "s", "tab:red")]:
            s = sub[sub["system"] == system].sort_values("depth")
            ax.plot(s["depth"], s["q_mean"], marker=marker, color=colour, label=f"{system} (mean)")
            ax.fill_between(s["depth"], s["q_min"], s["q_max"], color=colour, alpha=0.15)
        ax.set_title(prior, fontsize=10)
        ax.set_xlabel("depth")
        ax.set_xticks([1, 2, 3])
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="grey", lw=0.5, ls=":")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("prior-implied q_j (mean over nodes at depth)")
    axes[-1].legend(loc="center right", fontsize=8)
    fig.suptitle(
        "Prior-implied q_j vs tree depth — does the prior transmit C signal?",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
