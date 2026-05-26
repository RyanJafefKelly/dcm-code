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

def box(ax, x, y, w, h, text, fc, tc="white", lw=0.6, fontsize=11):
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=lw, edgecolor="black", facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", color=tc, fontsize=fontsize)


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
    """Per-stance tree: root C_s, features, subfeatures, indicators.

    Strict tree (each child has one parent within this stance). One example
    edge is annotated with the (beta_pres, beta_abs) transmission parameters
    that the methods text introduces. The model is fit independently per
    stance; this figure shows one such tree.
    """
    # Root: per-stance C_s
    box(ax, 6.0, 8.4, 1.0, 0.55, r"$C_s$", PALETTE["root"])
    ax.text(6.0, 8.83, r"(one of 13 stances per system)",
            ha="center", va="bottom", fontsize=9, color="#777",
            fontstyle="italic")

    # Features (3, evenly spaced, each from C_s)
    feature_x = [3.0, 6.0, 9.0]
    for x in feature_x:
        box(ax, x, 7.3, 1.0, 0.45, "feature", PALETTE["feature"])
        arrow(ax, 6.0, 8.13, x, 7.53, color=PALETTE["edge"])

    # Subfeatures: variable count per feature for visual realism.
    # Feature 3.0 has 1 subfeature directly below; features 6.0 and 9.0
    # have 2 subfeatures each at +/-0.7 offset.
    subfeature_specs = [
        (3.0, (0.0,)),         # 1 subfeature at 3.0
        (6.0, (-0.7, 0.7)),    # 2 subfeatures at 5.3, 6.7
        (9.0, (-0.7, 0.7)),    # 2 subfeatures at 8.3, 9.7
    ]
    for fx, offsets in subfeature_specs:
        for off in offsets:
            sx = fx + off
            box(ax, sx, 6.2, 0.95, 0.4, "subfeature",
                PALETTE["subfeature"], fontsize=9)
            arrow(ax, fx, 7.07, sx, 6.40, color=PALETTE["edge"])

    # Edge-parameter annotation on the feature 6.0 -> subfeature 5.3 edge,
    # drawn as a callout with a leader arrow so the labelled edge is unambiguous.
    ax.annotate(
        r"$\beta^{\mathrm{pres}}_u,\ \beta^{\mathrm{abs}}_u$",
        xy=(5.65, 6.74),       # midpoint of the example edge
        xytext=(4.45, 7.05),   # text position (upper-left of the edge)
        ha="center", va="center", fontsize=9,
        color="#444", fontstyle="italic",
        arrowprops=dict(arrowstyle="->", color="#999", lw=0.6,
                        connectionstyle="arc3,rad=0.2"),
    )

    # Indicators: variable count per subfeature, strict tree, no crossings.
    # Subfeature 3.0 has 2 indicators; the other 4 subfeatures have 1 each.
    # Focus indicator hangs under subfeature 6.7.
    indicator_specs = [
        (3.0, (-0.5, 0.5)),    # 2 indicators at 2.5, 3.5
        (5.3, (0.0,)),         # 1 indicator at 5.3
        (6.7, (0.0,)),         # 1 indicator at 6.7 (focus)
        (8.3, (0.0,)),         # 1 indicator at 8.3
        (9.7, (0.0,)),         # 1 indicator at 9.7
    ]
    for sx, offsets in indicator_specs:
        for off in offsets:
            ix = sx + off
            box(ax, ix, 5.1, 0.95, 0.45, "indicator",
                PALETTE["indicator"], fontsize=9)
            arrow(ax, sx, 5.97, ix, 5.33, color=PALETTE["edge"])

    return 6.7  # focus indicator x-coordinate


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

    # Latent indicator state z_j (three-state baseline)
    box(ax, j_x, 3.0, 1.6, 0.5, r"$z_j \in \{0,\, 1/2,\, 1\}$", PALETTE["m"])
    arrow(ax, j_x, 3.77, j_x, 3.25, color=PALETTE["edge"])

    # Expert plate: s_{ej}, r_{ej}
    expert_xs = [5.0, 6.7, 8.4]
    for ek in expert_xs:
        box(ax, ek, 1.9, 0.85, 0.45, r"$s_{ej}$", PALETTE["s"])
        # observed: thicker edge + diagonal-hatched fill for r_{ej}
        patch = FancyBboxPatch(
            (ek - 0.425, 0.625), 0.85, 0.45,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.2, edgecolor="black",
            facecolor=PALETTE["r"], hatch="///",
        )
        ax.add_patch(patch)
        ax.text(ek, 0.85, r"$r_{ej}$", ha="center", va="center",
                color="white", fontsize=11)
        arrow(ax, j_x, 2.77, ek, 2.13, color=PALETTE["edge"])
        arrow(ax, ek, 1.67, ek, 1.08, color=PALETTE["edge"])

    # Expert plate
    plate_e = mpatches.FancyBboxPatch(
        (4.45, 0.40), 4.4, 1.95,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        linewidth=0.8, edgecolor=PALETTE["edge"], facecolor="none",
        linestyle="--",
    )
    ax.add_patch(plate_e)
    ax.text(8.80, 2.40, r"experts  $e$", ha="right", va="bottom",
            fontsize=10, color=PALETTE["edge"])

    # Indicator plate
    plate_j = mpatches.FancyBboxPatch(
        (4.20, 0.20), 4.95, 4.30,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        linewidth=0.8, edgecolor=PALETTE["edge"], facecolor="none",
        linestyle="--",
    )
    ax.add_patch(plate_j)
    ax.text(9.10, 4.55, r"indicators  $j$", ha="right", va="bottom",
            fontsize=10, color=PALETTE["edge"])

    # ---- Shared ordinal parameters ----------------------------------------
    # Always: a (-> s), kappa (-> r). Variant-specific extras layered on top.
    # Param column is at x=10.6, with y-positions chosen by variant.

    # Default minimal placement (baseline): a + kappa only
    if variant == "baseline":
        box(ax, 10.6, 1.95, 0.7, 0.45, r"$a$",     PALETTE["muted"], tc="black")
        box(ax, 10.6, 0.85, 0.7, 0.45, r"$\kappa$", PALETTE["muted"], tc="black")
        ax.text(11.05, 1.95, "discrim.",   ha="left", va="center",
                fontsize=9, color="#888", fontstyle="italic")
        ax.text(11.05, 0.85, "thresholds", ha="left", va="center",
                fontsize=9, color="#888", fontstyle="italic")
        # Shared signal-noise label (sigma fixed = 1)
        ax.text(10.6, 2.55, r"$\varepsilon \sim \mathcal{N}(0,1)$",
                ha="center", va="bottom", fontsize=9, color="#444")
        # Edges into plate boundary
        plate_right = 8.85
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
        plate_right = 8.85
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
        plate_right = 8.85
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
        plate_right = 8.85
        arrow(ax, 10.25, 2.40, plate_right + 0.05, 1.95, color=PALETTE["edge"], lw=0.6)
        arrow(ax, 10.25, 1.50, plate_right + 0.05, 0.85, color=PALETTE["edge"], lw=0.6)

    else:
        raise ValueError(f"unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Layer labels and frame
# ---------------------------------------------------------------------------

def draw_frame(ax, variant):
    label_kw = dict(ha="left", va="center", fontsize=10, color="#444",
                    fontstyle="italic")
    ax.text(0.1, 8.4, r"root $C_s$",   **label_kw)
    ax.text(0.1, 7.3, "feature",       **label_kw)
    ax.text(0.1, 6.2, "subfeature",    **label_kw)
    ax.text(0.1, 5.1, "indicator",     **label_kw)
    ax.text(0.1, 4.0, r"tree-implied $q_j$", **label_kw)
    ax.text(0.1, 3.0, "latent state",  **label_kw)
    ax.text(0.1, 1.9, "latent signal", **label_kw)
    ax.text(0.1, 0.85, "rating (1-7, ordinal)", **label_kw)

    ax.axhline(4.55, xmin=0.06, xmax=0.97, color="#dddddd", lw=0.6, ls=":")
    ax.text(11.7, 4.55, "ordinal layer", ha="right", va="center",
            fontsize=10, color="#888", fontstyle="italic")

    title = {
        "baseline":          "DCM with ordinal observation layer (baseline)",
        "expert_shifts":     "DCM ordinal layer variant: expert location shifts $b_e$",
        "expert_cutpoints":  "DCM ordinal layer variant: per-expert cutpoints $\\kappa_e$",
        "expert_scales":     "DCM ordinal layer variant: per-expert noise scale $\\sigma_e$",
    }[variant]
    ax.figure.suptitle(title, fontsize=13, y=0.98)


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
