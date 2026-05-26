"""Shared style for ordinal-layer explainer figures.

Palette echoes the paper's Figure 1 (Pr(conscious) green, stance dark teal,
feature ochre, indicator terracotta) and adds three accent colours for the
new ordinal observation layer (latent state, latent signal, observed rating).
"""

from __future__ import annotations

import matplotlib as mpl

PALETTE = {
    "root":       "#3a5a40",  # Pr(conscious) green
    "stance":     "#0f3b3a",  # dark teal
    "feature":    "#b07a3b",  # ochre
    "subfeature": "#c46a3f",  # muted terracotta-orange (between feature and indicator)
    "indicator":  "#b8472b",  # terracotta
    # New ordinal layer
    "q":          "#7a9bb0",  # propagated marginal probability (deterministic)
    "m":          "#9b59b6",  # latent indicator state z_j
    "s":          "#d98c5f",  # latent expert signal s_ej
    "r":          "#5d8aa8",  # observed rating r_ej
    "muted":      "#cccccc",
    "edge":       "#6c757d",
    "edge_dot":   "#9aa4ab",
}

FONT = {
    "family": "serif",
    "size": 10,
}


def apply_rc():
    mpl.rcParams.update({
        "font.family": FONT["family"],
        "font.size": FONT["size"],
        "axes.linewidth": 0.6,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
