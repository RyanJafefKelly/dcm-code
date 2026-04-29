"""Behavioural cutpoint-style figures for the three expert-specific extensions.

Each variant gets a 1x2 figure contrasting:
    LEFT  -- shared baseline (one set of bumps, one set of cutpoints)
    RIGHT -- with the per-expert parameter active (three experts overlaid)

Variants:
    b_e      -- per-expert location shift; bumps slide along the axis,
                cutpoints stay shared.
    kappa_e  -- per-expert ordered cutpoints; bumps shared, cutpoint lines
                vary by expert.
    sigma_e  -- per-expert noise scale; bump centres shared, widths vary.

Run:
    python figures/ordinal_explainer_expert_variants.py            # all three
    python figures/ordinal_explainer_expert_variants.py --which be
    python figures/ordinal_explainer_expert_variants.py --which kappa_e
    python figures/ordinal_explainer_expert_variants.py --which sigma_e

Output:
    report_figures/ordinal_explainer/cutpoints_{be,kappa_e,sigma_e}.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from _style import PALETTE, apply_rc

apply_rc()

OUT = Path(__file__).resolve().parents[1] / "report_figures" / "ordinal_explainer"
OUT.mkdir(parents=True, exist_ok=True)

# Shared schematic params (consistent with the baseline cutpoint figure).
A = 1.6
SIGMA = 1.0
KAPPA = np.array([-1.6, -0.8, -0.25, 0.25, 0.8, 1.6])
ETAS = np.array([0.0, A / 2, A])  # m_j = 0, 1, 2

# Three "experts" with distinct accent colours. Used across all variants.
EXPERT_COLOURS = ["#2a6f97", "#a93f55", "#3a8a3f"]
EXPERT_LABELS = [r"expert $k_1$", r"expert $k_2$", r"expert $k_3$"]

# Per-variant illustrative deviations.
B_E_VALUES     = np.array([-0.6, 0.0, +0.6])           # location shifts
KAPPA_E_OFFSETS = np.array([-0.35, 0.0, +0.35])         # additive cutpoint shift
SIGMA_E_VALUES = np.array([0.6, 1.0, 1.5])              # per-expert sigmas

CAT_BAND_COLOUR = "#cccccc"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_axis_frame(ax, x_lo, x_hi, kappa, *, show_cat_labels=True,
                    title=""):
    edges = np.concatenate([[x_lo], kappa, [x_hi]])
    for c in range(len(edges) - 1):
        # subtle alternating shading
        if c % 2 == 0:
            ax.axvspan(edges[c], edges[c + 1], color=CAT_BAND_COLOUR,
                       alpha=0.08, lw=0)
        if show_cat_labels:
            ax.text((edges[c] + edges[c + 1]) / 2, -0.04,
                    f"{c + 1}", ha="center", va="top", fontsize=8,
                    color="#444", fontweight="bold")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.07, 0.78)
    ax.set_yticks([])
    ax.set_xlabel(r"latent signal  $s_{jk}$")
    ax.set_title(title, fontsize=10)


def plot_baseline_panel(ax, *, title="shared (baseline)"):
    """Single set of bumps + shared cutpoints."""
    x_lo, x_hi = -3.5, 3.5
    draw_axis_frame(ax, x_lo, x_hi, KAPPA, title=title)

    x = np.linspace(x_lo, x_hi, 600)
    bump_colour = PALETTE["m"]
    for eta in ETAS:
        y = norm.pdf(x, loc=eta, scale=SIGMA)
        ax.plot(x, y, color=bump_colour, lw=1.4, alpha=0.85)
        ax.fill_between(x, 0, y, color=bump_colour, alpha=0.10)
        ax.axvline(eta, color=bump_colour, lw=0.4, ls=":")

    for k_idx, k in enumerate(KAPPA):
        ax.axvline(k, color="#666", lw=0.7, ls="--")
        ax.text(k, 0.71, rf"$\kappa_{{{k_idx + 1}}}$",
                ha="center", va="bottom", fontsize=7, color="#555")

    # Marker for the bumps' shared interpretation
    ax.text(0.02, 0.97, r"all experts share $\kappa$, $\sigma$, no shift",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="#555", fontstyle="italic")


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def panel_b_e(ax):
    """Right panel: each expert has a different b_e shift."""
    x_lo, x_hi = -3.5, 3.5
    draw_axis_frame(ax, x_lo, x_hi, KAPPA, title=r"with per-expert $b_e$")

    x = np.linspace(x_lo, x_hi, 600)
    for col, lab, b_e in zip(EXPERT_COLOURS, EXPERT_LABELS, B_E_VALUES):
        for eta in ETAS:
            y = norm.pdf(x, loc=eta + b_e, scale=SIGMA)
            ax.plot(x, y, color=col, lw=1.2, alpha=0.85)
            ax.fill_between(x, 0, y, color=col, alpha=0.06)
        # legend handle (one per expert)
        ax.plot([], [], color=col, lw=1.6, label=f"{lab}  ($b_e={b_e:+.1f}$)")

    for k_idx, k in enumerate(KAPPA):
        ax.axvline(k, color="#666", lw=0.7, ls="--")
        ax.text(k, 0.71, rf"$\kappa_{{{k_idx + 1}}}$",
                ha="center", va="bottom", fontsize=7, color="#555")

    ax.text(0.02, 0.97,
            r"bumps slide $\to$ different baseline biases" + "\n"
            r"cutpoints $\kappa$ shared",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="#555", fontstyle="italic")
    ax.legend(loc="lower right", fontsize=7, frameon=False,
              handlelength=1.2, handletextpad=0.5)


def panel_kappa_e(ax):
    """Right panel: shared bumps, per-expert cutpoints."""
    x_lo, x_hi = -3.5, 3.5
    # No global cutpoints in the band-shading; we'll draw per-expert lines.
    draw_axis_frame(ax, x_lo, x_hi, KAPPA, show_cat_labels=False,
                    title=r"with per-expert $\kappa_e$")

    x = np.linspace(x_lo, x_hi, 600)
    bump_colour = PALETTE["m"]
    for eta in ETAS:
        y = norm.pdf(x, loc=eta, scale=SIGMA)
        ax.plot(x, y, color=bump_colour, lw=1.4, alpha=0.85)
        ax.fill_between(x, 0, y, color=bump_colour, alpha=0.10)

    # Per-expert kappa lines, colour-coded. Each expert's six cutpoints
    # appear as a colour-matched dashed-line family.
    for col, lab, off in zip(EXPERT_COLOURS, EXPERT_LABELS, KAPPA_E_OFFSETS):
        ke = KAPPA + off
        for k in ke:
            ax.axvline(k, color=col, lw=0.7, ls="--", alpha=0.55,
                       ymin=0.0, ymax=0.78)
        ax.plot([], [], color=col, lw=1.6, ls="--",
                label=f"{lab}  ($\\kappa_e = \\kappa{off:+.2f}$)")

    ax.text(0.02, 0.97,
            r"bumps shared, cutpoints $\kappa_e$ vary" + "\n"
            r"$\to$ experts use the scale differently",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="#555", fontstyle="italic")
    ax.legend(loc="lower right", fontsize=7, frameon=False,
              handlelength=1.2, handletextpad=0.5)


def panel_sigma_e(ax):
    """Right panel: shared centres, per-expert sigma."""
    x_lo, x_hi = -3.5, 3.5
    draw_axis_frame(ax, x_lo, x_hi, KAPPA, title=r"with per-expert $\sigma_e$")

    x = np.linspace(x_lo, x_hi, 600)
    for col, lab, sig in zip(EXPERT_COLOURS, EXPERT_LABELS, SIGMA_E_VALUES):
        for eta in ETAS:
            y = norm.pdf(x, loc=eta, scale=sig)
            ax.plot(x, y, color=col, lw=1.2, alpha=0.85)
            ax.fill_between(x, 0, y, color=col, alpha=0.06)
        ax.plot([], [], color=col, lw=1.6,
                label=f"{lab}  ($\\sigma_e={sig:.1f}$)")

    for k_idx, k in enumerate(KAPPA):
        ax.axvline(k, color="#666", lw=0.7, ls="--")
        ax.text(k, 0.71, rf"$\kappa_{{{k_idx + 1}}}$",
                ha="center", va="bottom", fontsize=7, color="#555")

    ax.text(0.02, 0.97,
            r"centres shared, widths vary" + "\n"
            r"$\to$ some experts sharper, others noisier",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="#555", fontstyle="italic")
    ax.legend(loc="lower right", fontsize=7, frameon=False,
              handlelength=1.2, handletextpad=0.5)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

VARIANT_FUNCS = {
    "be":      (panel_b_e,      "cutpoints_be",      r"$b_e$ — per-expert location shift"),
    "kappa_e": (panel_kappa_e,  "cutpoints_kappa_e", r"$\kappa_e$ — per-expert cutpoints"),
    "sigma_e": (panel_sigma_e,  "cutpoints_sigma_e", r"$\sigma_e$ — per-expert noise scale"),
}


def render(which: str):
    panel_fn, stem, title = VARIANT_FUNCS[which]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    plot_baseline_panel(axL)
    panel_fn(axR)
    fig.suptitle(f"Expert-specific extension: {title}",
                 fontsize=11, y=1.00)
    fig.tight_layout()

    fig.savefig(OUT / f"{stem}.png")
    fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)
    print(f"wrote {OUT/f'{stem}.png'}")
    print(f"wrote {OUT/f'{stem}.pdf'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--which", choices=list(VARIANT_FUNCS.keys()) + ["all"],
                   default="all")
    args = p.parse_args()

    if args.which == "all":
        for k in VARIANT_FUNCS:
            render(k)
    else:
        render(args.which)


if __name__ == "__main__":
    main()
