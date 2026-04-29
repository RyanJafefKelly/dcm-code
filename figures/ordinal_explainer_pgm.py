"""DCM structure with the ordinal observation layer made explicit.

Top half mirrors the paper Figure 1: Pr(conscious) -> stance -> feature ->
indicator. Bottom half adds the new ordinal observation layer:

    indicator q_j  ->  m_j  ->  s_{jk}  ->  r_{jk}

Variants (selectable via --variant):
    baseline           -- shared (a, kappa) only; epsilon ~ N(0, 1).
                          Matches dcm_model_ordinal.py defaults
                          (USE_EXPERT_SHIFTS=False, USE_HIERARCHICAL_*=False).
    expert_shifts      -- adds per-expert location shift b_e.
    expert_cutpoints   -- adds per-expert ordered cutpoints kappa_e.
    expert_scales      -- adds per-expert noise scale sigma_e.

Run:
    python figures/ordinal_explainer_pgm.py
    python figures/ordinal_explainer_pgm.py --variant expert_shifts
    python figures/ordinal_explainer_pgm.py --all   # writes all four
Output:
    report_figures/ordinal_explainer/pgm[_<variant>].{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _style import PALETTE, apply_rc

apply_rc()

OUT = Path(__file__).resolve().parents[1] / "report_figures" / "ordinal_explainer"
OUT.mkdir(parents=True, exist_ok=True)

VARIANTS = ("baseline", "expert_shifts", "expert_cutpoints", "expert_scales")


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def box(ax, x, y, w, h, text, fc, tc="white", lw=0.6):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=lw, edgecolor="black", facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", color=tc, fontsize=9)


def arrow(ax, x0, y0, x1, y1, color=None, dotted=False, lw=0.9):
    color = color or PALETTE["edge"]
    style = ":" if dotted else "-"
    ax.annotate(
        "",
        xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=style,
                        shrinkA=4, shrinkB=4),
    )


# ---------------------------------------------------------------------------
# Top-half tree (paper Figure 1)
# ---------------------------------------------------------------------------

def draw_paper_tree(ax):
    # Layer 1 - root
    box(ax, 6, 8.4, 1.5, 0.55, r"Pr(conscious)", PALETTE["root"])

    # Layer 2 - stances
    stance_x = [4.0, 6.0, 8.0]
    for x in stance_x:
        box(ax, x, 7.3, 1.0, 0.45, "stance", PALETTE["stance"])
        arrow(ax, 6, 8.13, x, 7.53, dotted=True, color=PALETTE["edge_dot"])

    # Layer 3 - features
    feature_x = [2.5, 4.0, 5.5, 7.0, 8.5, 10.0]
    for x in feature_x:
        box(ax, x, 6.2, 1.0, 0.45, "feature", PALETTE["feature"])

    # Stance -> feature edges (explicit; each stance fans to ~3 nearby features)
    stance_feature_edges = [
        (4.0, 2.5), (4.0, 4.0), (4.0, 5.5),
        (6.0, 4.0), (6.0, 5.5), (6.0, 7.0), (6.0, 8.5),
        (8.0, 7.0), (8.0, 8.5), (8.0, 10.0),
    ]
    for sx, fx in stance_feature_edges:
        arrow(ax, sx, 7.07, fx, 6.43, color=PALETTE["edge"])

    # Layer 4 - indicators (every indicator linked to >=1 feature)
    ind_x = [1.5, 2.7, 3.9, 5.1, 6.3, 7.5, 8.7, 9.9, 11.0]
    for x in ind_x:
        box(ax, x, 5.1, 0.95, 0.45, "indicator", PALETTE["indicator"])

    feature_indicator_edges = [
        (2.5, 1.5), (2.5, 2.7),
        (4.0, 2.7), (4.0, 3.9),
        (5.5, 3.9), (5.5, 5.1), (5.5, 6.3),
        (7.0, 5.1), (7.0, 6.3), (7.0, 7.5),
        (8.5, 7.5), (8.5, 8.7),
        (10.0, 8.7), (10.0, 9.9), (10.0, 11.0),
    ]
    for fx, ix in feature_indicator_edges:
        arrow(ax, fx, 5.97, ix, 5.33, color=PALETTE["edge"])

    return 6.3  # focus indicator x-coordinate (centre)


# ---------------------------------------------------------------------------
# Bottom-half ordinal layer
# ---------------------------------------------------------------------------

def draw_ordinal_layer(ax, j_x, variant):
    """Draw the q_j -> m_j -> s_{jk} -> r_{jk} layer + shared params.

    Shared-param composition is variant-dependent:
        baseline         : {a, kappa}
        expert_shifts    : {a, kappa, b_e}
        expert_cutpoints : {a, kappa_e}
        expert_scales    : {a, kappa, sigma_e}
    """
    # Marginal q_j (deterministic node)
    box(ax, j_x, 4.0, 0.7, 0.45, r"$q_j$", PALETTE["q"], tc="black")
    arrow(ax, j_x, 4.87, j_x, 4.23, color=PALETTE["edge"])

    # Latent indicator state m_j
    box(ax, j_x, 3.0, 1.15, 0.5, r"$m_j \in \{0,1,2\}$", PALETTE["m"])
    arrow(ax, j_x, 3.77, j_x, 3.25, color=PALETTE["edge"])

    # Expert plate: s_{jk}, r_{jk}
    expert_xs = [4.6, 6.3, 8.0]
    for ek in expert_xs:
        box(ax, ek, 1.9, 0.85, 0.45, r"$s_{jk}$", PALETTE["s"])
        # observed: thicker edge + diagonal-hatched fill for r_{jk}
        patch = FancyBboxPatch(
            (ek - 0.425, 0.625), 0.85, 0.45,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.2, edgecolor="black",
            facecolor=PALETTE["r"], hatch="///",
        )
        ax.add_patch(patch)
        ax.text(ek, 0.85, r"$r_{jk}$", ha="center", va="center",
                color="white", fontsize=9)
        arrow(ax, j_x, 2.77, ek, 2.13, color=PALETTE["edge"])
        arrow(ax, ek, 1.67, ek, 1.08, color=PALETTE["edge"])

    # Expert plate
    plate_e = mpatches.FancyBboxPatch(
        (4.05, 0.40), 4.4, 1.95,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        linewidth=0.8, edgecolor=PALETTE["edge"], facecolor="none",
        linestyle="--",
    )
    ax.add_patch(plate_e)
    ax.text(8.40, 2.40, r"experts  $k$", ha="right", va="bottom",
            fontsize=8, color=PALETTE["edge"])

    # Indicator plate
    plate_j = mpatches.FancyBboxPatch(
        (3.80, 0.20), 4.95, 4.30,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        linewidth=0.8, edgecolor=PALETTE["edge"], facecolor="none",
        linestyle="--",
    )
    ax.add_patch(plate_j)
    ax.text(8.70, 4.55, r"indicators  $j$", ha="right", va="bottom",
            fontsize=8, color=PALETTE["edge"])

    # ---- Shared ordinal parameters ----------------------------------------
    # Always: a (-> s), kappa (-> r). Variant-specific extras layered on top.
    # Param column is at x=10.6, with y-positions chosen by variant.

    # Default minimal placement (baseline): a + kappa only
    if variant == "baseline":
        box(ax, 10.6, 1.95, 0.7, 0.45, r"$a$",     PALETTE["muted"], tc="black")
        box(ax, 10.6, 0.85, 0.7, 0.45, r"$\kappa$", PALETTE["muted"], tc="black")
        ax.text(11.05, 1.95, "discrim.",   ha="left", va="center",
                fontsize=7, color="#888", fontstyle="italic")
        ax.text(11.05, 0.85, "thresholds", ha="left", va="center",
                fontsize=7, color="#888", fontstyle="italic")
        # Shared signal-noise label (sigma fixed = 1)
        ax.text(10.6, 2.55, r"$\varepsilon \sim \mathcal{N}(0,1)$",
                ha="center", va="bottom", fontsize=7.5, color="#444")
        # Edges into plate boundary
        plate_right = 8.45
        arrow(ax, 10.25, 1.95, plate_right + 0.05, 1.95, color=PALETTE["edge"], lw=0.6)
        arrow(ax, 10.25, 0.85, plate_right + 0.05, 0.85, color=PALETTE["edge"], lw=0.6)

    elif variant == "expert_shifts":
        box(ax, 10.6, 2.40, 0.7, 0.45, r"$a$",   PALETTE["muted"], tc="black")
        # b_e drawn INSIDE the expert plate (per-expert), accent colour
        for ek in expert_xs:
            box(ax, ek, 2.55, 0.55, 0.32, r"$b_e$", "#f0d686", tc="black", lw=0.5)
            arrow(ax, ek, 2.39, ek, 2.13, color="#aa9244", lw=0.6)
        box(ax, 10.6, 1.50, 0.7, 0.45, r"$\kappa$", PALETTE["muted"], tc="black")
        ax.text(11.05, 2.40, "discrim.",   ha="left", va="center",
                fontsize=7, color="#888", fontstyle="italic")
        ax.text(11.05, 1.50, "thresholds", ha="left", va="center",
                fontsize=7, color="#888", fontstyle="italic")
        plate_right = 8.45
        arrow(ax, 10.25, 2.40, plate_right + 0.05, 1.95, color=PALETTE["edge"], lw=0.6)
        arrow(ax, 10.25, 1.50, plate_right + 0.05, 0.85, color=PALETTE["edge"], lw=0.6)

    elif variant == "expert_cutpoints":
        box(ax, 10.6, 1.95, 0.7, 0.45, r"$a$", PALETTE["muted"], tc="black")
        # per-expert kappa_e inside the plate
        for ek in expert_xs:
            box(ax, ek, 0.20, 0.7, 0.30, r"$\kappa_e$", "#f0d686",
                tc="black", lw=0.5)
            arrow(ax, ek, 0.36, ek, 0.62, color="#aa9244", lw=0.6)
        ax.text(11.05, 1.95, "discrim.", ha="left", va="center",
                fontsize=7, color="#888", fontstyle="italic")
        ax.text(8.40, 0.05, "per-expert thresholds (hierarchical)",
                ha="right", va="bottom",
                fontsize=7, color="#888", fontstyle="italic")
        plate_right = 8.45
        arrow(ax, 10.25, 1.95, plate_right + 0.05, 1.95, color=PALETTE["edge"], lw=0.6)

    elif variant == "expert_scales":
        box(ax, 10.6, 2.40, 0.7, 0.45, r"$a$",        PALETTE["muted"], tc="black")
        box(ax, 10.6, 1.50, 0.7, 0.45, r"$\kappa$",   PALETTE["muted"], tc="black")
        for ek in expert_xs:
            box(ax, ek, 2.55, 0.55, 0.32, r"$\sigma_e$", "#f0d686",
                tc="black", lw=0.5)
            arrow(ax, ek, 2.39, ek, 2.13, color="#aa9244", lw=0.6)
        ax.text(11.05, 2.40, "discrim.",   ha="left", va="center",
                fontsize=7, color="#888", fontstyle="italic")
        ax.text(11.05, 1.50, "thresholds", ha="left", va="center",
                fontsize=7, color="#888", fontstyle="italic")
        plate_right = 8.45
        arrow(ax, 10.25, 2.40, plate_right + 0.05, 1.95, color=PALETTE["edge"], lw=0.6)
        arrow(ax, 10.25, 1.50, plate_right + 0.05, 0.85, color=PALETTE["edge"], lw=0.6)

    else:
        raise ValueError(f"unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Layer labels and frame
# ---------------------------------------------------------------------------

def draw_frame(ax, variant):
    label_kw = dict(ha="left", va="center", fontsize=8, color="#444",
                    fontstyle="italic")
    ax.text(0.1, 8.4, "system",        **label_kw)
    ax.text(0.1, 7.3, "stance",        **label_kw)
    ax.text(0.1, 6.2, "feature",       **label_kw)
    ax.text(0.1, 5.1, "indicator",     **label_kw)
    ax.text(0.1, 4.0, "marginal $q_j$", **label_kw)
    ax.text(0.1, 3.0, "latent state",  **label_kw)
    ax.text(0.1, 1.9, "latent signal", **label_kw)
    ax.text(0.1, 0.85, "rating (1-7, ordinal)", **label_kw)

    ax.axhline(4.55, xmin=0.06, xmax=0.97, color="#dddddd", lw=0.6, ls=":")
    ax.text(11.7, 4.55, "ordinal layer", ha="right", va="center",
            fontsize=8, color="#888", fontstyle="italic")

    title = {
        "baseline":          "DCM with ordinal observation layer  -  baseline",
        "expert_shifts":     "DCM ordinal layer  -  variant: expert location shifts $b_e$",
        "expert_cutpoints":  "DCM ordinal layer  -  variant: per-expert cutpoints $\\kappa_e$",
        "expert_scales":     "DCM ordinal layer  -  variant: per-expert noise scale $\\sigma_e$",
    }[variant]
    ax.figure.suptitle(title, fontsize=11, y=0.98)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def render(variant: str):
    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    j_x = draw_paper_tree(ax)
    draw_ordinal_layer(ax, j_x, variant)
    draw_frame(ax, variant)

    suffix = "" if variant == "baseline" else f"_{variant}"
    fig.savefig(OUT / f"pgm{suffix}.png")
    fig.savefig(OUT / f"pgm{suffix}.pdf")
    plt.close(fig)
    print(f"wrote {OUT/f'pgm{suffix}.png'}")
    print(f"wrote {OUT/f'pgm{suffix}.pdf'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=VARIANTS, default="baseline")
    p.add_argument("--all", action="store_true",
                   help="render every variant")
    args = p.parse_args()

    if args.all:
        for v in VARIANTS:
            render(v)
    else:
        render(args.variant)


if __name__ == "__main__":
    main()
