"""Visualise how much information each candidate leaf model preserves in the
q_j → rating mapping.

For a sweep of q_j values, plots the predicted 7-category rating distribution
under five leaf models, holding (a, kappa, β_pres, β_abs) fixed at the
production posterior medians. This is the analytical version of the question
"given a q_j, how much does the leaf model compress it before we observe a
rating?".

Five leaf models compared:

  current_3state   m ~ BetaBinomial(2, β_pres, β_abs) → m ∈ {0, 1/2, 1};
                   latent = a · m; rating ~ OrderedProbit(latent, kappa).
                   This is the existing production model.
  direct_q         no latent z; latent = a · q; rating ~ OrderedProbit(latent, kappa).
                   B.1 in next_directions.md.
  logit_normal_z   z ~ Normal(logit(q), tau); m = sigmoid(z); latent = a · m;
                   rating ~ OrderedProbit(latent, kappa). Marginalise z by
                   simple grid quadrature. B.2.
  betabin_K6       m ~ BetaBinomial(K=6, β_pres, β_abs) / 6 → m ∈ {0, 1/6, ..., 1};
                   latent = a · m; rating ~ OrderedProbit(latent, kappa). B.3.
                   Latent state count matches output cardinality (7).
  mixture          rating ~ q · OrderedProbit(α_high, kappa) + (1-q) · OrderedProbit(α_low, kappa).
                   B.4.

Output: figs/leaf_model_compression.png — for q_j ∈ {0, 0.1, ..., 1.0},
predicted 7-category distribution under each model.
Also prints information ceiling per model: H(rating|q) averaged over a
uniform prior on q (lower = more informative; 0 = perfect identification).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SISTER_DIR = REPO_ROOT / "notebooks" / "synthetic_validation_2026-05-06"
SWEEP_DIR = Path(__file__).resolve().parent
for d in (REPO_ROOT, SISTER_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from gwt_oracle_internal_identifiability import ordered_probit_probs  # noqa: E402

OUT = SWEEP_DIR / "figs" / "leaf_model_compression.png"
N_CATS = 7

# Production posterior median nuisance — the same "exact_tree_production_medians"
# Ryan uses for the synthetic DGP. Falls back to plausible defaults if the file
# is missing.
def _load_production_medians() -> Dict:
    sample_truth = SISTER_DIR / "runs/full_exact_recovery"
    if sample_truth.exists():
        for sub in sorted(sample_truth.iterdir()):
            tj = sub / "truth.json"
            if tj.exists():
                t = json.loads(tj.read_text())
                obs = t.get("observation_parameters", {})
                if "a" in obs and "kappa" in obs:
                    return {
                        "a": float(obs["a"]),
                        "kappa": np.asarray(obs["kappa"], dtype=float),
                    }
    return {"a": 4.0, "kappa": np.linspace(-2.4, 2.4, N_CATS - 1)}


def predict_three_state(q: float, a: float, kappa: np.ndarray, beta_pres: float, beta_abs: float) -> np.ndarray:
    p_m_p1 = np.array([(1 - beta_pres) ** 2, 2 * beta_pres * (1 - beta_pres), beta_pres ** 2])
    p_m_p0 = np.array([(1 - beta_abs) ** 2, 2 * beta_abs * (1 - beta_abs), beta_abs ** 2])
    p_m = q * p_m_p1 + (1 - q) * p_m_p0
    out = np.zeros(N_CATS)
    for m_idx, m_val in enumerate((0.0, 0.5, 1.0)):
        out += p_m[m_idx] * ordered_probit_probs(kappa, a * m_val)
    return out


def predict_direct_q(q: float, a: float, kappa: np.ndarray) -> np.ndarray:
    return ordered_probit_probs(kappa, a * q)


def predict_logit_normal_z(q: float, a: float, kappa: np.ndarray, tau: float, n_grid: int = 21) -> np.ndarray:
    eps = 1e-6
    q = min(max(q, eps), 1 - eps)
    logit_q = float(np.log(q / (1 - q)))
    grid = np.linspace(-3.0, 3.0, n_grid)
    z = logit_q + tau * grid
    weights = np.exp(-grid ** 2 / 2)
    weights = weights / weights.sum()
    out = np.zeros(N_CATS)
    for zi, w in zip(z, weights):
        m_val = 1.0 / (1.0 + np.exp(-zi))
        out += w * ordered_probit_probs(kappa, a * m_val)
    return out


def predict_betabin_K(q: float, a: float, kappa: np.ndarray, beta_pres: float, beta_abs: float, K: int) -> np.ndarray:
    """K-state generalisation: m ~ BetaBinomial(K, β_eff)/K, where β_eff = q·β_pres + (1-q)·β_abs."""
    beta_eff = q * beta_pres + (1 - q) * beta_abs
    from math import comb
    p_m = np.array([comb(K, k) * (beta_eff ** k) * ((1 - beta_eff) ** (K - k)) for k in range(K + 1)])
    p_m = p_m / p_m.sum()
    m_vals = np.arange(K + 1) / K
    out = np.zeros(N_CATS)
    for p, m_val in zip(p_m, m_vals):
        out += p * ordered_probit_probs(kappa, a * m_val)
    return out


def predict_mixture(q: float, a: float, kappa: np.ndarray, alpha_low: float = 0.0, alpha_high: float = 1.0) -> np.ndarray:
    return q * ordered_probit_probs(kappa, a * alpha_high) + (1 - q) * ordered_probit_probs(kappa, a * alpha_low)


def info_ceiling_h_rating_given_q(model_fn, q_grid: np.ndarray) -> float:
    """E_q[H(rating | q)] under uniform q. Lower = more informative leaf model."""
    eps = 1e-12
    h_total = 0.0
    for q in q_grid:
        probs = model_fn(q)
        probs = np.clip(probs, eps, 1.0)
        probs = probs / probs.sum()
        h_total += -float(np.sum(probs * np.log(probs)))
    return h_total / len(q_grid)


def main() -> None:
    nuisance = _load_production_medians()
    a = nuisance["a"]
    kappa = nuisance["kappa"]
    beta_pres = 0.65  # paper-mu rough average
    beta_abs = 0.30

    q_panel_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    q_dense = np.linspace(0.001, 0.999, 41)

    models = {
        "current_3state (β=0.65/0.30)": lambda q: predict_three_state(q, a, kappa, beta_pres, beta_abs),
        "current_3state (β=0.90/0.10, override centre)": lambda q: predict_three_state(q, a, kappa, 0.90, 0.10),
        "direct_q (B.1)": lambda q: predict_direct_q(q, a, kappa),
        "logit_normal_z τ=0.5 (B.2)": lambda q: predict_logit_normal_z(q, a, kappa, tau=0.5),
        "logit_normal_z τ=1.0 (B.2)": lambda q: predict_logit_normal_z(q, a, kappa, tau=1.0),
        "betabin K=6 (β=0.90/0.10) (B.3)": lambda q: predict_betabin_K(q, a, kappa, 0.90, 0.10, K=6),
        "mixture α_low=0, α_high=1 (B.4)": lambda q: predict_mixture(q, a, kappa, 0.0, 1.0),
    }

    fig, axes = plt.subplots(len(models), len(q_panel_grid), figsize=(2.2 * len(q_panel_grid), 2.0 * len(models)),
                              sharex=True, sharey=True)
    for row, (label, fn) in enumerate(models.items()):
        for col, q in enumerate(q_panel_grid):
            ax = axes[row, col]
            probs = fn(q)
            ax.bar(np.arange(N_CATS), probs, color="steelblue")
            ax.set_ylim(0, 0.7)
            ax.set_xticks(range(N_CATS))
            if row == 0:
                ax.set_title(f"q_j = {q:.2f}", fontsize=9)
            if col == 0:
                ax.set_ylabel(label.replace(" ", "\n", 1), fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(
        "Predicted 7-category rating distribution under each candidate leaf model "
        "(production-median nuisance: a, kappa)",
        fontsize=11,
    )
    fig.text(0.5, -0.005, "ordinal category (0 = certainly not; 6 = certainly yes)", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")

    print()
    print("Information ceiling: E_q[H(rating | q)] under uniform q in (0, 1)")
    print("(lower = leaf model conveys more info per rating; theoretical floor = 0)")
    for label, fn in models.items():
        h = info_ceiling_h_rating_given_q(fn, q_dense)
        print(f"  {label:<48s}  {h:.4f}")


if __name__ == "__main__":
    main()
