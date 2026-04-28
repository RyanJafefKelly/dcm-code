"""σ_pool prior-predictive sweep for the pool-by-label intervention.

For each σ_pool ∈ {0.25, 0.5, 0.75, 1.0}, sample 10⁴ tree realisations from
the pooling logit-Normal prior (no observations) and propagate q_j through
the GWT tree at C ∈ {0.001, 0.5, 0.999}. Report induced δ_j (depth-stratified),
mean q_j(0.999) − q_j(0.001), and fraction of indicators with sign-flipped δ_j.

Wording precision (per ChatGPT round-2 correction): the pooled prior is
**median-centred** at the paper Beta's prior mean, NOT mean-centred. A
logit-Normal with location logit μ has natural-scale median μ but mean
≠ μ; the natural-scale mean shifts toward 0.5, increasingly so for extreme
μ and large σ_pool.

Output: notebooks/meeting_prep_arvo_2026-04-27/figs_round2/sigma_pool_sweep_summary.csv
        and sigma_pool_sweep_qj.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dcm_model import (
    EvidenceProcessor,
    ModelConfig,
    collect_tree_label_groups,
    load_data,
    node_key,
)
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS
from tree_propagation_diagnostic import (
    IndicatorSpec,
    collect_indicator_path_metadata,
)

OUT_DIR = Path("notebooks/meeting_prep_arvo_2026-04-27/figs_round2")
STANCE = "Global Workspace Theory"
ANCHOR_HIGH = 0.999
ANCHOR_LOW = 0.001
SIGMA_VALUES = (0.25, 0.5, 0.75, 1.0)
N_SAMPLES = 10_000


def _logit(p: float, eps: float = 1e-9) -> float:
    p = max(min(p, 1.0 - eps), eps)
    return float(np.log(p / (1.0 - p)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sample_pool_prior_predictive(
    stance_data: Dict[str, Any],
    evidence: EvidenceProcessor,
    sigma_pool: float,
    n_samples: int,
    C_values: Tuple[float, ...],
    rng_seed: int = 42,
) -> Tuple[Dict[float, np.ndarray], List[str]]:
    """Sample n_samples (β_pres, β_abs) realisations from the pooling
    logit-Normal prior (median-centred at paper means) at every label group,
    propagate q_j through the GWT tree at each C in C_values.

    Returns ({C: ndarray of shape (n_samples, n_indicators)}, indicator_keys).
    """
    pres_groups, abs_groups = collect_tree_label_groups(stance_data)
    rng = np.random.default_rng(rng_seed)

    # Per-group natural-scale draws (n_samples per group)
    bp_by_group: Dict[Tuple[str, str], np.ndarray] = {}
    for (s, d) in pres_groups:
        a_p, b_p, _, _ = evidence.get_beta_parameters(s, d)
        mu = a_p / (a_p + b_p)
        logit_mu = _logit(mu)
        eps = rng.standard_normal(n_samples)
        bp_by_group[(s, d)] = sigmoid(logit_mu + sigma_pool * eps)

    ba_by_group: Dict[str, np.ndarray] = {}
    for d in abs_groups:
        _, _, a_a, b_a = evidence.get_beta_parameters("no bearing", d)
        mu = a_a / (a_a + b_a)
        logit_mu = _logit(mu)
        eps = rng.standard_normal(n_samples)
        ba_by_group[d] = sigmoid(logit_mu + sigma_pool * eps)

    # Walk tree, build per-(C, sample, indicator) q_j
    indicator_keys: List[str] = []
    indicator_alpha: List[np.ndarray] = []
    indicator_slope: List[np.ndarray] = []
    root_path = (stance_data["name"],)

    def walk(node, path, intercept_parent, slope_parent):
        cur = path + (node["name"],)
        s = node.get("support", "no bearing")
        d = node.get("demandingness", "neutral")
        bp = bp_by_group[(s, d)]
        ba = ba_by_group[d]
        delta = bp - ba
        intercept_child = ba + intercept_parent * delta
        slope_child = slope_parent * delta
        if (node.get("type") or "").lower() == "indicator":
            key = node_key(path, node["name"])
            indicator_keys.append(key)
            indicator_alpha.append(intercept_child)
            indicator_slope.append(slope_child)
            return
        for child in node.get("evidencers", []):
            walk(child, cur, intercept_child, slope_child)

    root_intercept = np.zeros(n_samples)
    root_slope = np.ones(n_samples)
    for child in stance_data.get("evidencers", []):
        walk(child, root_path, root_intercept, root_slope)

    alpha = np.column_stack(indicator_alpha)  # (n_samples, n_indicators)
    slope = np.column_stack(indicator_slope)

    out = {C: alpha + slope * C for C in C_values}
    out["__alpha__"] = alpha
    out["__slope__"] = slope
    return out, indicator_keys


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=True,
    )
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    evidence = EvidenceProcessor(cfg)

    # Indicator depth metadata for stratification
    path_meta = collect_indicator_path_metadata(stance_data)

    print(f"σ_pool sweep: {SIGMA_VALUES}, n_samples={N_SAMPLES}")
    print(f"  median-centred logit-Normal pooling priors at each label group")
    print(f"  paper β_pres / β_abs prior means used as natural-scale medians")
    print()

    # Per-σ summary
    summary_rows: List[Dict[str, Any]] = []
    delta_dist_per_sigma: Dict[float, np.ndarray] = {}

    for sigma in SIGMA_VALUES:
        results, indicator_keys = sample_pool_prior_predictive(
            stance_data, evidence, sigma_pool=sigma,
            n_samples=N_SAMPLES,
            C_values=(ANCHOR_LOW, 0.5, ANCHOR_HIGH),
        )
        slope = results["__slope__"]  # (n_samples, n_indicators), this IS δ_j per draw
        alpha = results["__alpha__"]
        delta_dist_per_sigma[sigma] = slope

        # Anchor separation: mean q_high - q_low across draws + indicators
        q_high = results[ANCHOR_HIGH]
        q_low = results[ANCHOR_LOW]
        gap = (q_high - q_low)  # (n_samples, n_indicators)
        gap_mean_across_inds = gap.mean(axis=1)  # (n_samples,)

        # δ_j stats
        delta_med_per_ind = np.median(slope, axis=0)  # (n_indicators,)
        delta_mean_per_ind = slope.mean(axis=0)
        # Sign-flip rate per draw
        n_sign_flip = (slope < 0).sum(axis=1).mean()  # mean across draws

        # Depth stratification
        depths = np.array([path_meta[k].depth for k in indicator_keys])
        for depth in (2, 3):
            mask = depths == depth
            n_d = int(mask.sum())
            if n_d == 0:
                continue
            d_med = float(np.median(slope[:, mask]))
            d_mean = float(slope[:, mask].mean())
            d_q01 = float(np.percentile(slope[:, mask], 1))
            d_q99 = float(np.percentile(slope[:, mask], 99))
            summary_rows.append({
                "sigma_pool": sigma,
                "depth": depth,
                "n_indicators_at_depth": n_d,
                "delta_j_median": d_med,
                "delta_j_mean": d_mean,
                "delta_j_p1": d_q01,
                "delta_j_p99": d_q99,
            })

        # Overall row
        summary_rows.append({
            "sigma_pool": sigma,
            "depth": "ALL",
            "n_indicators_at_depth": len(indicator_keys),
            "delta_j_median": float(np.median(slope)),
            "delta_j_mean": float(slope.mean()),
            "delta_j_p1": float(np.percentile(slope, 1)),
            "delta_j_p99": float(np.percentile(slope, 99)),
            "anchor_separation_mean": float(gap_mean_across_inds.mean()),
            "anchor_separation_p1": float(np.percentile(gap_mean_across_inds, 1)),
            "anchor_separation_p99": float(np.percentile(gap_mean_across_inds, 99)),
            "n_sign_flip_per_draw_mean": float(n_sign_flip),
            "frac_sign_flip": float(n_sign_flip / len(indicator_keys)),
        })
        print(f"σ_pool = {sigma:.2f}:")
        print(f"  mean δ_j (all):           {float(slope.mean()):+.4f}")
        print(f"  mean q(.999)-q(.001):     {float(gap_mean_across_inds.mean()):+.4f}")
        print(f"  fraction δ_j < 0:         {float(n_sign_flip / len(indicator_keys)):.3f}")

    # Write summary
    df = pd.DataFrame(summary_rows)
    out_csv = OUT_DIR / "sigma_pool_sweep_summary.csv"
    df.to_csv(out_csv, index=False)
    print()
    print(f"wrote {out_csv}")

    # Plot — per-σ distribution of δ_j across indicators+draws
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for sigma in SIGMA_VALUES:
        slope = delta_dist_per_sigma[sigma]
        ax.hist(slope.ravel(), bins=80, alpha=0.4, density=True,
                label=f"σ={sigma:.2f}")
    ax.axvline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("δ_j = path-product slope (per draw, per indicator)")
    ax.set_ylabel("density")
    ax.set_title("Pool prior-predictive δ_j distribution by σ_pool")
    ax.legend()

    ax = axes[1]
    sigmas = np.array(SIGMA_VALUES)
    medians = []
    p1s = []
    p99s = []
    anchor_seps = []
    sign_flip_fracs = []
    for sigma in SIGMA_VALUES:
        slope = delta_dist_per_sigma[sigma]
        medians.append(float(np.median(slope)))
        p1s.append(float(np.percentile(slope, 1)))
        p99s.append(float(np.percentile(slope, 99)))
        anchor_seps.append(float(slope.mean() * (ANCHOR_HIGH - ANCHOR_LOW)))
        sign_flip_fracs.append(float((slope < 0).mean()))
    ax.errorbar(sigmas, medians,
                yerr=np.stack([np.array(medians) - np.array(p1s),
                               np.array(p99s) - np.array(medians)]),
                fmt="o-", color="#1f78b4", label="median + 1/99% range")
    ax.set_xlabel("σ_pool (logit-Normal scale, median-centred at paper Beta means)")
    ax.set_ylabel("δ_j across draws + indicators")
    ax.set_title("δ_j sensitivity to σ_pool")
    ax.legend()

    plt.tight_layout()
    fig.savefig(OUT_DIR / "sigma_pool_sweep_qj.png", dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'sigma_pool_sweep_qj.png'}")

    print()
    print(">>> Interpretation guide:")
    print("    σ=0.25 ≈ paper-prior-tight (≤ ×1.6 odds wrong over 95% interval)")
    print("    σ=0.50 ≈ paper-odds-wrong-by-≤×2.7 over 95% interval (current pool_3s)")
    print("    σ=0.75 ≈ ≤ ×4.5 odds over 95% interval")
    print("    σ=1.00 ≈ ≤ ×7.4 odds over 95% interval")


if __name__ == "__main__":
    main()
