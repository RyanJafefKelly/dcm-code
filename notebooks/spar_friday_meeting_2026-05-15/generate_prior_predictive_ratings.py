"""Forward-simulate prior-predictive expert ratings for Human (R=1) vs ELIZA (R=0)
under the baseline (paper-mu) prior vs the targeted strong-lower-edge override.

No MCMC. Mirrors the production three-state observation layer:
- m_j ~ Binomial(2, q_j); emission centres eta_0=0, eta_1=a/2, eta_2=a
- a ~ HalfNormal(0, A_PRIOR_SIGMA); kappa ~ N(0, KAPPA_PRIOR_SIGMA) ordered
- latent xi ~ N(eta, 1); rating = 1 + #{cutpoints below xi}

q_j is propagated deterministically using each variant's prior-mean betas, so
all variation comes from the observation-layer prior. This isolates the
prior-transmission story (Human vs ELIZA spread) cleanly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
for d in (REPO_ROOT, SCRIPTS):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from dcm_model import ModelConfig, load_data, node_key  # noqa: E402
from main_synthetic_validation_metrics import build_edge_beta_profile  # noqa: E402

STANCE = "Global Workspace Theory"
SYSTEMS = {"Human  (R = 1)": 0.999, "ELIZA  (R = 0)": 0.001}
VARIANTS = ["production_unmodified", "targeted_strong_lower_override"]
VARIANT_TITLE = {
    "production_unmodified": "Baseline prior",
    "targeted_strong_lower_override": "With targeted fix",
}
N_OBS_PER_INDICATOR = 400
K = 7
SEED = 20260515


def collect_indicator_q(stance_data, beta_pres_by_key, beta_abs_by_key, c_value):
    """Walk tree; return list of q_j (one per Indicator) given root c."""
    out = []

    def walk(node, ancestor_path, parent_q):
        key = node_key(ancestor_path, node["name"])
        bp = beta_pres_by_key[key]
        ba = beta_abs_by_key[key]
        q = ba + parent_q * (bp - ba)
        if (node.get("type") or "").lower() == "indicator":
            out.append(float(q))
        new_path = ancestor_path + (node["name"],)
        for child in node.get("evidencers", []):
            walk(child, new_path, q)

    root_path = (stance_data["name"],)
    for child in stance_data.get("evidencers", []):
        walk(child, root_path, c_value)
    return out


def simulate_ratings(q_per_indicator, n_obs_per_indicator, rng, a_typical, kappa_typical):
    """For each indicator's q_j, simulate n_obs ratings using fixed *typical*
    observation-layer params (a_typical, kappa_typical). Variation comes from
    m_j ~ Binomial(2, q_j) and the probit noise xi ~ N(eta, 1)."""
    n_ind = len(q_per_indicator)
    n_total = n_ind * n_obs_per_indicator

    q_repeat = np.repeat(np.asarray(q_per_indicator), n_obs_per_indicator)
    m = rng.binomial(2, q_repeat)
    eta = m * a_typical / 2.0
    xi = rng.normal(eta, 1.0)
    ratings = 1 + np.searchsorted(kappa_typical, xi, side="right")
    return ratings.astype(int)


def main() -> None:
    # build_edge_beta_profile falls back to 0.90 / 0.10 when the cfg fields are None,
    # so we leave them unset to avoid the POOL/TARGETED-required validation.
    cfg = ModelConfig(INDICATOR_STATE_MODEL="three_state")
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    profiles = {v: build_edge_beta_profile(stance_data, v, cfg) for v in VARIANTS}

    # Sanity: report mean q_j per (variant, system) over all indicators
    print("Mean prior-implied q_j across indicators (sanity check):")
    for variant in VARIANTS:
        prof = profiles[variant]
        for sys_name, c_val in SYSTEMS.items():
            qs = collect_indicator_q(
                stance_data,
                prof.fitter_prior_mean_pres_by_key,
                prof.fitter_prior_mean_abs_by_key,
                c_val,
            )
            print(
                f"  {variant:38s} {sys_name:18s}"
                f"  n_ind={len(qs):3d}  q_mean={np.mean(qs):.3f}"
                f"  q_min={np.min(qs):.3f}  q_max={np.max(qs):.3f}"
            )
    print()

    # Typical observation-layer params:
    #   a:  HalfNormal(0, A_PRIOR_SIGMA) -> mean = sigma * sqrt(2/pi)
    #   kappa:  ordered N(0, KAPPA_PRIOR_SIGMA) -> use the model's initval
    #           (linspace(-1.5, 1.5, K-1)) which is the production starting point
    a_typical = cfg.A_PRIOR_SIGMA * np.sqrt(2.0 / np.pi)
    kappa_typical = np.linspace(-1.5, 1.5, K - 1)

    rng = np.random.default_rng(SEED)
    results = {}
    for variant in VARIANTS:
        prof = profiles[variant]
        for sys_name, c_val in SYSTEMS.items():
            qs = collect_indicator_q(
                stance_data,
                prof.fitter_prior_mean_pres_by_key,
                prof.fitter_prior_mean_abs_by_key,
                c_val,
            )
            r = simulate_ratings(qs, N_OBS_PER_INDICATOR, rng, a_typical, kappa_typical)
            results[(variant, sys_name)] = r

    # Two panels: baseline prior vs targeted fix.  In each, overlay Human and ELIZA.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    cat_centres = np.arange(1, K + 1)
    sys_colours = {"Human  (R = 1)": "#1f77b4", "ELIZA  (R = 0)": "#d62728"}
    bar_w = 0.36

    for ax, variant in zip(axes, VARIANTS):
        for offset, sys_name in zip([-bar_w / 2, bar_w / 2], SYSTEMS):
            r = results[(variant, sys_name)]
            h, _ = np.histogram(r, bins=np.arange(0.5, K + 1.5))
            h = h / h.sum()
            ax.bar(
                cat_centres + offset,
                h,
                width=bar_w,
                label=sys_name,
                color=sys_colours[sys_name],
                alpha=0.85,
            )
        ax.set_title(VARIANT_TITLE[variant], fontsize=12)
        ax.set_xticks(cat_centres)
        ax.set_xlabel("rating category (1 = lowest, 7 = highest)")
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("proportion of prior-predictive ratings")

    fig.suptitle(
        "Prior-predictive expert ratings, Human (R = 1) vs ELIZA (R = 0)\n"
        "(forward simulation, no MCMC; GWT tree; obs layer fixed at prior typical values)",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "prior_predictive_ratings.png"
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
