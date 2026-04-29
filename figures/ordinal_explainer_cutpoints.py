"""Cutpoint diagram for the ordered-probit observation layer.

Shows the latent signal axis with three Gaussian bumps centred at
eta in {0, a/2, a} (one per latent indicator state m_j in {0,1,2}),
six ordered cutpoints kappa_1..kappa_6 partitioning the axis into
seven Likert bands, and a stacked-bar panel showing the implied
P(r | m_j) for each m_j.

Defaults are SCHEMATIC -- chosen so the bumps and cutpoints are
both visually legible. To use real posterior means from a fit:

    python figures/ordinal_explainer_cutpoints.py \
        --from-fit results/gwt_ordinal/hard_anchor_3s_paper_tree_<TS>/hard_anchor_3s_paper_tree.nc

Run:
    python figures/ordinal_explainer_cutpoints.py            # schematic
    python figures/ordinal_explainer_cutpoints.py --from-fit <PATH.nc>
Output:
    report_figures/ordinal_explainer/cutpoints[_posterior].{png,pdf}
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

# ---------------------------------------------------------------------------
# Schematic defaults (illustrative only -- chosen for legibility, not fit).
# ---------------------------------------------------------------------------
SCHEMATIC_A = 1.6
SCHEMATIC_SIGMA = 1.0
SCHEMATIC_KAPPA = np.array([-1.6, -0.8, -0.25, 0.25, 0.8, 1.6])

M_LABELS = [r"$m_j = 0$", r"$m_j = 1$", r"$m_j = 2$"]
CAT_COLOURS = ["#7e2a1a", "#a73c25", "#c8633b", "#d8a06b",
               "#7faa86", "#3f7f56", "#1b4332"]


def category_probs(eta, kappa, sigma):
    edges = np.concatenate([[-np.inf], kappa, [np.inf]])
    cdfs = norm.cdf(edges, loc=eta, scale=sigma)
    return np.diff(cdfs)


def load_from_fit(path: Path):
    """Pull posterior-mean a, kappa from an arviz NetCDF.

    sigma_e is fixed at 1 in the baseline (no expert scales). If the fit has
    expert-specific kappa or sigma we average / collapse to global values for
    a single illustrative panel.
    """
    import arviz as az
    idata = az.from_netcdf(path)
    post = idata.posterior
    a = float(post["a"].mean(dim=["chain", "draw"]).values)
    kappa = post["kappa"].mean(dim=["chain", "draw"]).values
    if kappa.ndim > 1:
        kappa = kappa.mean(axis=tuple(range(kappa.ndim - 1)))
    sigma = 1.0
    return a, np.asarray(kappa, dtype=float), sigma


def render(a, sigma, kappa, *, source_label: str, suffix: str = ""):
    etas = np.array([0.0, a / 2, a])

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(11, 4.4),
        gridspec_kw={"width_ratios": [2.6, 1.0], "wspace": 0.25},
    )

    # ---- Left panel: latent axis with bands and Gaussians ------------------
    x_lo = float(min(kappa.min(), etas.min()) - 2.0)
    x_hi = float(max(kappa.max(), etas.max()) + 2.0)
    x = np.linspace(x_lo, x_hi, 800)
    edges = np.concatenate([[x_lo], kappa, [x_hi]])
    y_max = 0.50  # density y-limit

    for c in range(7):
        axL.axvspan(edges[c], edges[c + 1], color=CAT_COLOURS[c],
                    alpha=0.10, lw=0)
        axL.text((edges[c] + edges[c + 1]) / 2, -0.04,
                 f"{c + 1}", ha="center", va="top", fontsize=9,
                 color=CAT_COLOURS[c], fontweight="bold")

    for k_idx, k in enumerate(kappa):
        axL.axvline(k, color="#888", lw=0.7, ls="--")
        axL.text(k, y_max - 0.04, rf"$\kappa_{{{k_idx + 1}}}$",
                 ha="center", va="bottom", fontsize=8, color="#555")

    bump_colours = [PALETTE["m"], "#7c498f", "#5a2f6c"]
    for eta, lab, col in zip(etas, M_LABELS, bump_colours):
        y = norm.pdf(x, loc=eta, scale=sigma)
        axL.plot(x, y, color=col, lw=1.6, label=lab)
        axL.fill_between(x, 0, y, color=col, alpha=0.10)
        axL.axvline(eta, color=col, lw=0.5, ls=":")

    axL.set_xlim(x_lo, x_hi)
    axL.set_ylim(-0.07, y_max)
    axL.set_xlabel(r"latent signal  $s_{jk} = \eta_{m_j} + \varepsilon$")
    axL.set_ylabel("density")
    axL.set_yticks([])
    axL.legend(loc="upper left", fontsize=8, frameon=False)
    axL.set_title("Ordered probit: latent signal $\\to$ Likert category",
                  fontsize=10)

    for eta in etas:
        axL.text(eta, y_max - 0.10, rf"$\eta={eta:.2f}$",
                 ha="center", va="bottom", fontsize=7, color="#444")

    # ---- Right panel: stacked bars of P(r | m_j) ---------------------------
    probs = np.stack([category_probs(eta, kappa, sigma) for eta in etas])
    bottoms = np.zeros(3)
    y_pos = np.arange(3)
    for c in range(7):
        axR.barh(y_pos, probs[:, c], left=bottoms,
                 color=CAT_COLOURS[c], edgecolor="white", lw=0.5,
                 label=f"r={c + 1}")
        for i in range(3):
            if probs[i, c] > 0.06:
                axR.text(bottoms[i] + probs[i, c] / 2, i,
                         f"{probs[i, c]:.2f}",
                         ha="center", va="center",
                         fontsize=7, color="white")
        bottoms += probs[:, c]

    axR.set_yticks(y_pos)
    axR.set_yticklabels(M_LABELS)
    axR.invert_yaxis()
    axR.set_xlim(0, 1)
    axR.set_xlabel(r"$P(r_{jk} \mid m_j)$")
    axR.set_title("Implied category probabilities", fontsize=10)
    axR.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
               ncol=7, fontsize=7, frameon=False, handlelength=1.0,
               columnspacing=0.6, handletextpad=0.4)

    fig.text(0.01, -0.02,
             rf"{source_label}: $a={a:.2f}$, $\sigma={sigma:.2f}$, "
             rf"$\kappa=({', '.join(f'{k:.2f}' for k in kappa)})$",
             fontsize=7, color="#666")

    fig.savefig(OUT / f"cutpoints{suffix}.png")
    fig.savefig(OUT / f"cutpoints{suffix}.pdf")
    plt.close(fig)
    print(f"wrote {OUT/f'cutpoints{suffix}.png'}")
    print(f"wrote {OUT/f'cutpoints{suffix}.pdf'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from-fit", type=Path, default=None,
                   help="path to .nc fit; pulls posterior-mean a, kappa")
    args = p.parse_args()

    if args.from_fit is None:
        render(SCHEMATIC_A, SCHEMATIC_SIGMA, SCHEMATIC_KAPPA,
               source_label="schematic")
    else:
        a, kappa, sigma = load_from_fit(args.from_fit)
        render(a, sigma, kappa,
               source_label=f"posterior mean ({args.from_fit.parent.name})",
               suffix="_posterior")


if __name__ == "__main__":
    main()
