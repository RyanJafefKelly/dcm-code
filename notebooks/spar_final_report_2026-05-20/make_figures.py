"""make_figures.py — Generate Section 4.2 figures for the SPAR final report.

Reads posterior medians and 94% credible intervals from the already-computed
``results/all_stances_exact_baseline/summary.json`` (no re-fitting). Outputs
both PDF (for the LaTeX include) and PNG (for inline preview) to ``figures/``.

Outputs
-------
- figures/results_llm_focused_all_stances.{pdf,png}
    Section 4.2 main-text figure. 2024 LLMs across all 13 stances, primary;
    Chicken as a thinner secondary trace for biological comparison. Stances
    ordered by 2024 LLM posterior median.

- figures/results_paired_chicken_llm_all_stances.{pdf,png}
    Alternative / appendix figure. Equal-weight paired Chicken and 2024 LLMs
    across all 13 stances. Stances ordered by signed difference
    delta = median(Chicken) - median(LLMs), so biology-favouring stances appear
    at the top and cognition-favouring at the bottom.

- figures/results_all_stances_credences.{pdf,png}
    Appendix four-panel forest plot, one panel per system, 13 stances per
    panel. Anchor panels (Human, ELIZA) shrunk via ``gridspec`` width_ratios.

- figures/results_gwt_credences.{pdf,png}
    Standalone single-stance forest plot for Global Workspace Theory (4
    systems). Kept for reference / poster / sensitivity use.

Run from the repo root or from this directory:
    python notebooks/spar_final_report_2026-05-20/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SUMMARY_PATH = REPO_ROOT / "results" / "all_stances_exact_baseline" / "summary.json"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SYS_KEYS = ["Human", "Chicken", "LLMs", "ELIZA"]
SYS_DISPLAY = {
    "Human": "Human",
    "Chicken": "Chicken",
    "LLMs": "2024 LLMs",
    "ELIZA": "ELIZA",
}
PRIOR_MEAN_FREE = 1.0 / 6.0  # Beta(1, 5) mean for the free systems

STANCE_ORDER = [
    "Global Workspace Theory",
    "Integrated Information Theory",
    "Higher Order Theory",
    "Recurrent Processing Theory (Perceptual)",
    "Recurrent Processing Theory (Pure)",
    "Attention Schema Theory",
    "Person-like",
    "Cognitive Complexity",
    "Embodied Agency",
    "Field Mechanisms",
    "Simple Valence",
    "Computational Analogy",
    "Biological Analogy",
]

SHORT: Dict[str, str] = {
    "Global Workspace Theory": "GWT",
    "Integrated Information Theory": "IIT",
    "Higher Order Theory": "HOT",
    "Recurrent Processing Theory (Perceptual)": "RPT (Perceptual)",
    "Recurrent Processing Theory (Pure)": "RPT (Pure)",
    "Attention Schema Theory": "AST",
    "Person-like": "Person-like",
    "Cognitive Complexity": "Cognitive Complexity",
    "Embodied Agency": "Embodied Agency",
    "Field Mechanisms": "Field Mechanisms",
    "Simple Valence": "Simple Valence",
    "Computational Analogy": "Comp. Analogy",
    "Biological Analogy": "Biological Analogy",
}

LLM_COLOR = "#2E5F8C"      # primary blue
CHICKEN_COLOR = "#D77B53"  # warm secondary
SECONDARY_GRAY = "#8A8A8A"
PRIOR_LINE_COLOR = "gray"


def load_summary() -> dict:
    return json.loads(SUMMARY_PATH.read_text())


def _get_med_lo_hi(summary: dict, stance: str, sys_key: str) -> Tuple[float, float, float]:
    m, lo, hi = summary[stance]["C_posterior"][sys_key]
    return float(m), float(lo), float(hi)


def _save(fig: Figure, stem: str) -> None:
    """Save PDF at print quality and a lower-DPI PNG for inline preview."""
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# PRIMARY MAIN-TEXT CANDIDATE: LLM-focused all-stances, Chicken secondary
# ----------------------------------------------------------------------
def fig_llm_focused_all_stances(summary: dict) -> None:
    """LLMs primary, Chicken secondary. Stances ordered by LLM median.

    Uses full stance names (no acronyms) and larger text. Wider canvas to
    accommodate the longer y-tick labels (e.g. "Recurrent Processing Theory
    (Perceptual)").
    """
    llm_data = [_get_med_lo_hi(summary, s, "LLMs") for s in STANCE_ORDER]
    chicken_data = [_get_med_lo_hi(summary, s, "Chicken") for s in STANCE_ORDER]

    # Order by LLM posterior median, descending so highest at top after invert.
    order = sorted(range(len(STANCE_ORDER)), key=lambda i: -llm_data[i][0])
    stances_sorted = [STANCE_ORDER[i] for i in order]
    llm_sorted = [llm_data[i] for i in order]
    chicken_sorted = [chicken_data[i] for i in order]

    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    ys = np.arange(len(stances_sorted))
    offset = 0.22

    # Chicken first so LLMs draw on top
    for j, (m, lo, hi) in enumerate(chicken_sorted):
        ax.errorbar(
            [m], [ys[j] + offset],
            xerr=[[m - lo], [hi - m]],
            fmt="o", capsize=3, markersize=5.5,
            color=SECONDARY_GRAY, linewidth=1.3,
            mfc="white", mec=SECONDARY_GRAY, alpha=0.95,
        )
    for j, (m, lo, hi) in enumerate(llm_sorted):
        ax.errorbar(
            [m], [ys[j] - offset],
            xerr=[[m - lo], [hi - m]],
            fmt="o", capsize=4, markersize=8.5,
            color=LLM_COLOR, linewidth=2.1,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels(stances_sorted, fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(r"Posterior $P(C=1 \mid \mathrm{data})$", fontsize=12)
    ax.axvline(
        PRIOR_MEAN_FREE, color=PRIOR_LINE_COLOR, linestyle="--",
        linewidth=1.0, alpha=0.75,
    )

    legend_elements = [
        Line2D([0], [0], marker="o", color=LLM_COLOR, label="2024 LLMs",
               markersize=8.5, linewidth=2.1),
        Line2D([0], [0], marker="o", color=SECONDARY_GRAY, label="Chicken",
               markersize=5.5, linewidth=1.3, mfc="white", mec=SECONDARY_GRAY),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.95)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, "results_llm_focused_all_stances")


# ----------------------------------------------------------------------
# ALTERNATIVE / APPENDIX CANDIDATE: equal-weight paired Chicken + LLM
# ordered by Chicken minus LLM
# ----------------------------------------------------------------------
def fig_paired_chicken_llm_all_stances(summary: dict) -> None:
    """Equal-weight Chicken + LLM, ordered by signed difference (full names)."""
    chicken_data = [_get_med_lo_hi(summary, s, "Chicken") for s in STANCE_ORDER]
    llm_data = [_get_med_lo_hi(summary, s, "LLMs") for s in STANCE_ORDER]

    deltas = [c[0] - lm[0] for c, lm in zip(chicken_data, llm_data)]
    # Highest Chicken-favouring at top, most LLM-favouring at bottom.
    order = sorted(range(len(STANCE_ORDER)), key=lambda i: -deltas[i])
    stances_sorted = [STANCE_ORDER[i] for i in order]
    chicken_sorted = [chicken_data[i] for i in order]
    llm_sorted = [llm_data[i] for i in order]

    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    ys = np.arange(len(stances_sorted))
    offset = 0.22

    for j, (m, lo, hi) in enumerate(chicken_sorted):
        ax.errorbar(
            [m], [ys[j] - offset],
            xerr=[[m - lo], [hi - m]],
            fmt="o", capsize=3, markersize=7,
            color=CHICKEN_COLOR, linewidth=1.7,
        )
    for j, (m, lo, hi) in enumerate(llm_sorted):
        ax.errorbar(
            [m], [ys[j] + offset],
            xerr=[[m - lo], [hi - m]],
            fmt="o", capsize=3, markersize=7,
            color=LLM_COLOR, linewidth=1.7,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels(stances_sorted, fontsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(r"Posterior $P(C=1 \mid \mathrm{data})$", fontsize=12)
    ax.axvline(
        PRIOR_MEAN_FREE, color=PRIOR_LINE_COLOR, linestyle="--",
        linewidth=1.0, alpha=0.75,
    )

    legend_elements = [
        Line2D([0], [0], marker="o", color=CHICKEN_COLOR, label="Chicken",
               markersize=7, linewidth=1.7),
        Line2D([0], [0], marker="o", color=LLM_COLOR, label="2024 LLMs",
               markersize=7, linewidth=1.7),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.95)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _save(fig, "results_paired_chicken_llm_all_stances")


# ----------------------------------------------------------------------
# APPENDIX FALLBACK: four-panel forest plot with shrunk anchor panels
# ----------------------------------------------------------------------
def fig_all_stances_credences(summary: dict) -> None:
    """Four panels (Human, Chicken, 2024 LLMs, ELIZA) with anchor panels shrunk."""
    fig, axes = plt.subplots(
        1, 4, figsize=(13, 5.8), sharey=True,
        gridspec_kw={"width_ratios": [0.55, 1.35, 1.35, 0.55]},
    )
    for sys_i, sys_key in enumerate(SYS_KEYS):
        ax = axes[sys_i]
        for j, stance in enumerate(STANCE_ORDER):
            m, lo, hi = _get_med_lo_hi(summary, stance, sys_key)
            ax.errorbar(
                [m], [j],
                xerr=[[m - lo], [hi - m]],
                fmt="o", capsize=3, markersize=5,
                color=LLM_COLOR, linewidth=1.4,
            )
        ax.set_title(SYS_DISPLAY[sys_key], fontsize=11)
        ax.axvline(PRIOR_MEAN_FREE, color=PRIOR_LINE_COLOR, linestyle="--",
                   linewidth=0.7, alpha=0.5)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, axis="x", alpha=0.3)
        if sys_i == 0:
            ax.set_yticks(range(len(STANCE_ORDER)))
            ax.set_yticklabels([SHORT[s] for s in STANCE_ORDER], fontsize=9)
            ax.invert_yaxis()
    fig.text(
        0.5, 0.02,
        r"Posterior $P(C=1 \mid \mathrm{data})$  (median and 94% CrI)",
        ha="center", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "results_all_stances_credences")


# ----------------------------------------------------------------------
# REFERENCE / POSTER OPTION: GWT-only forest plot (4 systems)
# ----------------------------------------------------------------------
def fig_gwt_credences(summary: dict) -> None:
    """Single-stance forest plot for Global Workspace Theory."""
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    for i, sys_key in enumerate(SYS_KEYS):
        m, lo, hi = _get_med_lo_hi(summary, "Global Workspace Theory", sys_key)
        ax.errorbar(
            [m], [i],
            xerr=[[m - lo], [hi - m]],
            fmt="o", capsize=4, markersize=8,
            color=LLM_COLOR, linewidth=2,
        )
    ax.set_yticks(np.arange(len(SYS_KEYS)))
    ax.set_yticklabels([SYS_DISPLAY[k] for k in SYS_KEYS])
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(r"Posterior $P(C=1 \mid \mathrm{data})$  (median and 94% CrI)")
    ax.axvline(
        PRIOR_MEAN_FREE, color=PRIOR_LINE_COLOR, linestyle="--",
        linewidth=0.9, alpha=0.7, label=r"Beta(1,5) prior mean",
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    _save(fig, "results_gwt_credences")


def main() -> None:
    summary = load_summary()
    fig_llm_focused_all_stances(summary)
    fig_paired_chicken_llm_all_stances(summary)
    fig_all_stances_credences(summary)
    fig_gwt_credences(summary)
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
