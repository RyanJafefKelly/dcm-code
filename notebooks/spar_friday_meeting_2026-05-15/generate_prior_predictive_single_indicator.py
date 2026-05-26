"""Drill-down: prior-predictive expert ratings for a single GWT indicator,
under 100 prior draws, for Human (R = 1) and ELIZA (R = 0), across baseline
and targeted-fix priors.

For each prior draw we sample (β_pres, β_abs) at every edge along the path
from root to indicator from its logit-Normal prior, propagate q_j down the
path, and compute the implied rating distribution analytically under typical
observation params.

Indicator: Stable Personality (depth 3, under Coherence → Point of View).
Two of its three path edges sit in the targeted-fix override set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
for d in (REPO_ROOT, SCRIPTS):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from dcm_model import ModelConfig, load_data, node_key  # noqa: E402
from main_synthetic_validation_metrics import (  # noqa: E402
    build_edge_beta_profile,
    collect_targeted_override_node_keys,
)

STANCE = "Global Workspace Theory"
INDICATOR_PATH = ["Coherence", "Point of View", "Stable Personality"]

VARIANTS = ["production_unmodified", "targeted_strong_lower_override"]
VARIANT_LABEL = {
    "production_unmodified": "Baseline prior",
    "targeted_strong_lower_override": "With targeted fix",
}
ROOT_STATES = {"R = 1": 0.999, "R = 0": 0.001}
ROOT_COLOUR = {"R = 1": "#1f77b4", "R = 0": "#d62728"}
N_PRIOR_DRAWS = 100
K = 7
SEED = 20260515


def find_path_keys(stance_data, indicator_path):
    keys = []
    current = stance_data
    ancestor_path = (stance_data["name"],)
    for name in indicator_path:
        children = current.get("evidencers", [])
        match = next((c for c in children if c["name"] == name), None)
        if match is None:
            raise ValueError(f"No child '{name}' under {ancestor_path}")
        keys.append(node_key(ancestor_path, name))
        ancestor_path = ancestor_path + (name,)
        current = match
    return keys


def sample_q_at_indicator(
    profile, path_keys, targeted_set, c_value,
    sigma_default, sigma_targeted, n_draws, rng, override_active,
):
    """If override_active is True (targeted variant), edges in targeted_set use
    sigma_targeted; otherwise all edges use sigma_default."""
    q = np.full(n_draws, c_value)
    for key in path_keys:
        mu_p = profile.fitter_prior_mean_pres_by_key[key]
        mu_a = profile.fitter_prior_mean_abs_by_key[key]
        sigma = sigma_targeted if (override_active and key in targeted_set) else sigma_default
        eps = 1e-9
        logit_p = np.log(mu_p / (1 - mu_p + eps)) + sigma * rng.normal(0, 1, n_draws)
        logit_a = np.log(mu_a / (1 - mu_a + eps)) + sigma * rng.normal(0, 1, n_draws)
        beta_p = 1.0 / (1.0 + np.exp(-logit_p))
        beta_a = 1.0 / (1.0 + np.exp(-logit_a))
        q = beta_a + q * (beta_p - beta_a)
    return q


def rating_distribution_per_draw(q_per_draw, a_typical, kappa_typical):
    """Analytical P(rating = k) per draw under three-state leaf + ordinal probit.

    Returns array of shape (n_draws, K).
    """
    n_draws = len(q_per_draw)
    p_m = np.zeros((n_draws, 3))
    p_m[:, 0] = (1 - q_per_draw) ** 2
    p_m[:, 1] = 2 * q_per_draw * (1 - q_per_draw)
    p_m[:, 2] = q_per_draw ** 2

    eta_m = np.array([0.0, a_typical / 2.0, a_typical])
    extended_kappa = np.concatenate([[-np.inf], kappa_typical, [np.inf]])
    p_y_given_m = np.zeros((3, K))
    for m in range(3):
        upper = norm.cdf(extended_kappa[1:] - eta_m[m])
        lower = norm.cdf(extended_kappa[:-1] - eta_m[m])
        p_y_given_m[m] = upper - lower
    return p_m @ p_y_given_m


def main():
    cfg = ModelConfig(INDICATOR_STATE_MODEL="three_state")
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    profiles = {v: build_edge_beta_profile(stance_data, v, cfg) for v in VARIANTS}
    targeted_set = set(collect_targeted_override_node_keys(stance_data))
    path_keys = find_path_keys(stance_data, INDICATOR_PATH)

    sigma_default = float(cfg.LABEL_POOL_SIGMA)
    sigma_targeted = float(
        cfg.BETA_OVERRIDE_SIGMA if cfg.BETA_OVERRIDE_SIGMA is not None else 0.30
    )

    print(f"Indicator path: {' → '.join(['GWT'] + INDICATOR_PATH)}")
    print("Per-edge prior structure:")
    for v in VARIANTS:
        prof = profiles[v]
        override_active = v == "targeted_strong_lower_override"
        print(f"\n  {VARIANT_LABEL[v]}:")
        for key in path_keys:
            mu_p = prof.fitter_prior_mean_pres_by_key[key]
            mu_a = prof.fitter_prior_mean_abs_by_key[key]
            is_in_set = key in targeted_set
            sigma_used = sigma_targeted if (override_active and is_in_set) else sigma_default
            short_key = key.split(" > ")[-1]
            tag = " [override active]" if (override_active and is_in_set) else ""
            print(
                f"    {short_key:30s}  μ_pres={mu_p:.3f}  μ_abs={mu_a:.3f}  "
                f"σ={sigma_used:.2f}{tag}"
            )
    print(
        f"\nUsing σ_default={sigma_default}, σ_targeted={sigma_targeted}, "
        f"{N_PRIOR_DRAWS} prior draws per system per variant"
    )

    a_typical = cfg.A_PRIOR_SIGMA * np.sqrt(2.0 / np.pi)
    kappa_typical = np.linspace(-1.5, 1.5, K - 1)

    rng = np.random.default_rng(SEED)
    rating_dists = {}
    print("\nPrior-implied q_j summary (across 100 draws):")
    for variant in VARIANTS:
        override_active = variant == "targeted_strong_lower_override"
        for r_label, c_val in ROOT_STATES.items():
            q = sample_q_at_indicator(
                profiles[variant], path_keys, targeted_set, c_val,
                sigma_default, sigma_targeted, N_PRIOR_DRAWS, rng,
                override_active=override_active,
            )
            rating_dists[(variant, r_label)] = rating_distribution_per_draw(
                q, a_typical, kappa_typical
            )
            print(
                f"  {VARIANT_LABEL[variant]:24s} {r_label:8s}"
                f"  q median {np.median(q):.3f}  [{np.percentile(q, 5):.3f}, "
                f"{np.percentile(q, 95):.3f}]"
            )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    cat_centres = np.arange(1, K + 1)
    for ax, variant in zip(axes, VARIANTS):
        for r_label in ROOT_STATES:
            dists = rating_dists[(variant, r_label)]
            colour = ROOT_COLOUR[r_label]
            for d in dists:
                ax.plot(cat_centres, d, color=colour, alpha=0.06, linewidth=1.0)
            med = np.median(dists, axis=0)
            ax.plot(cat_centres, med, color=colour, linewidth=2.6, label=f"{r_label} (median of 100 draws)")
        ax.set_title(VARIANT_LABEL[variant], fontsize=12)
        ax.set_xticks(cat_centres)
        ax.set_xlabel("rating category (1 = lowest, 7 = highest)")
        ax.set_ylim(0, None)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("prior-predictive  P(rating = category)")

    full_path = "GWT  →  " + "  →  ".join(INDICATOR_PATH)
    fig.suptitle(
        f"Prior-predictive ratings for a single indicator — Stable Personality\n"
        f"path: {full_path}   |   100 prior draws per system, observation layer fixed",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "prior_predictive_single_indicator.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
