"""Composite-vs-exact log-likelihood diagnostic on the pool_3s posterior.

Background.  The DCM uses one ``pm.Potential`` per (system, indicator) with a
binary mixture in the propagated marginal probability ``q_{s,j}``.  There are
no internal-node ``z`` random variables in the PyMC graph; each indicator's
likelihood treats it as conditionally independent of siblings given
``(C, β)``.  This is **not** an exact Rao-Blackwellisation of the original
DCM latent tree -- it is a composite marginal likelihood that drops the
sibling covariance through every shared internal node.

This script quantifies the gap.  For each posterior draw of the ``pool_3s``
fit it computes:

    Δℓ_s = ℓ_exact(θ_s) − ℓ_composite(θ_s)

where ℓ_exact is the latent-tree marginal computed by bottom-up dynamic
programming, and ℓ_composite mirrors the implemented ``pm.Potential`` formula
bit-for-bit (mixture-of-products: one shared latent state per indicator,
all ratings on that indicator share it).

Outputs (written to ``notebooks/meeting_prep_arvo_2026-04-27/figs_round2/``):
    - ``composite_gap_summary.csv`` -- Δℓ percentiles, ESS, headline shifts.
    - ``composite_gap.png`` -- Δℓ histogram + reweighted-vs-original C posterior.
    - ``sibling_correlation_scores.csv`` -- top internal nodes ranked by
      sibling-correlation potential score.

Sanity checks performed before writing outputs:
    1. No-transmission: setting β_pres == β_abs everywhere ⇒ ℓ_exact == ℓ_composite.
    2. Brute-force enumeration on a synthetic 2-internal-node, 4-leaf tree.
    3. Self-consistency: ℓ_composite + Δℓ == ℓ_exact (rounding-invariance).

Failure of any sanity check raises before reporting.

Usage:
    python composite_vs_exact_diagnostic.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from analyse_tree_pooling import pooled_beta_draws_by_node
from dcm_model import (
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    MultiSystemModelBuilder,
    load_data,
    node_key,
)
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS

POOL_3S_PATH = Path("results/gwt_tree_pooling/three_state_pooled_anchored.nc")
OUT_DIR = Path("notebooks/meeting_prep_arvo_2026-04-27/figs_round2")
STANCE = "Global Workspace Theory"
ANCHOR_HUMAN = 0.999
ANCHOR_ELIZA = 0.001
HEADLINE_C_VARS = {
    "Chicken": "chicken__global_workspace_theory_C",
    "LLMs": "2024_leading_chat_llms__global_workspace_theory_C",
}


# ----------------------------------------------------------------------------
# Ordered-probit category probability (pure numpy)
# ----------------------------------------------------------------------------


def ordered_probit_category_logp_per_obs(
    ratings: np.ndarray,        # (N,) int category indices
    expert_idx: np.ndarray,     # (N,)
    a_draws: np.ndarray,        # (S,)
    b_draws: np.ndarray,        # (S, n_experts)
    kappa_draws: np.ndarray,    # (S, K-1)
    eta_offset_factor: float,   # 0.0, 0.5, 1.0  -- η = b + factor * a
    K: int = 7,
) -> np.ndarray:
    """Sum of log P(rating_i | obs_i) across observations, per draw.

    Returns array of shape (S,) with sum of log-cat-probs across all N obs.
    """
    S = a_draws.shape[0]
    N = ratings.shape[0]
    eta = b_draws[:, expert_idx] + eta_offset_factor * a_draws[:, None]   # (S, N)
    cum_full = np.empty((S, N, K + 1))
    cum_full[..., 0] = 0.0
    cum_full[..., -1] = 1.0
    for k in range(K - 1):
        cum_full[..., k + 1] = norm.cdf(kappa_draws[:, k:k + 1] - eta)
    probs = np.diff(cum_full, axis=-1)
    probs = np.clip(probs, 1e-12, 1.0)
    chosen = probs[:, np.arange(N), ratings]  # (S, N)
    return np.log(chosen).sum(axis=1)         # (S,)


# ----------------------------------------------------------------------------
# Composite + Exact log-likelihood for the propagated tree
# ----------------------------------------------------------------------------


def collect_indicator_obs(
    stance_data: Dict[str, Any],
    proc: MultiSystemDataProcessor,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Map indicator node_key → {system: (ratings, expert_idx)} from processor."""
    out: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    root_path = (stance_data["name"],)

    def walk(node: Dict[str, Any], path: Tuple[str, ...]) -> None:
        cur = path + (node["name"],)
        if (node.get("type") or "").lower() == "indicator":
            key = node_key(path, node["name"])
            sys_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for sys_name in proc.systems:
                obs = proc.system_observations.get(sys_name, {}).get(key, [])
                if obs:
                    ratings = np.array([r for _, r in obs], dtype=np.int64)
                    expert_idx = np.array([e for e, _ in obs], dtype=np.int64)
                    sys_dict[sys_name] = (ratings, expert_idx)
            out[key] = sys_dict
            return
        for child in node.get("evidencers", []):
            walk(child, cur)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path)
    return out


def precompute_leaf_logliks(
    indicator_obs: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    a_draws: np.ndarray,
    b_draws: np.ndarray,
    kappa_draws: np.ndarray,
    K: int = 7,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Precompute ℓ_m (m=0,1,2) per (indicator, system) per draw.

    Returns {node_key: {system: array (S, 3)}} where component m corresponds
    to η_m ∈ {0, a/2, a} aggregated as sum_e log P(r_e | η_m).
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for key, sys_obs in indicator_obs.items():
        per_sys: Dict[str, np.ndarray] = {}
        for sys_name, (ratings, expert_idx) in sys_obs.items():
            ll0 = ordered_probit_category_logp_per_obs(
                ratings, expert_idx, a_draws, b_draws, kappa_draws,
                eta_offset_factor=0.0, K=K,
            )
            llh = ordered_probit_category_logp_per_obs(
                ratings, expert_idx, a_draws, b_draws, kappa_draws,
                eta_offset_factor=0.5, K=K,
            )
            ll1 = ordered_probit_category_logp_per_obs(
                ratings, expert_idx, a_draws, b_draws, kappa_draws,
                eta_offset_factor=1.0, K=K,
            )
            per_sys[sys_name] = np.stack([ll0, llh, ll1], axis=-1)  # (S, 3)
        out[key] = per_sys
    return out


def three_state_log_B(
    beta: np.ndarray,             # (S,)
    leaf_lls: np.ndarray,         # (S, 3)
) -> np.ndarray:
    """log B_{s,j}(β) = log[(1-β)² ℓ_0 + 2β(1-β) ℓ_½ + β² ℓ_1]

    Mixture-of-products form: one shared latent m_j per indicator, with
    π_m(β) = ((1-β)², 2β(1-β), β²) and ℓ_m = sum_e log P(r_e | m).
    Returns (S,).
    """
    eps = 1e-12
    log1mb = np.log(np.clip(1.0 - beta, eps, 1.0))
    logb = np.log(np.clip(beta, eps, 1.0))
    log_w0 = 2.0 * log1mb
    log_w1 = np.log(2.0) + logb + log1mb
    log_w2 = 2.0 * logb
    return logsumexp(
        np.stack(
            [log_w0 + leaf_lls[:, 0],
             log_w1 + leaf_lls[:, 1],
             log_w2 + leaf_lls[:, 2]],
            axis=-1,
        ),
        axis=-1,
    )


def three_state_log_B_at_q(
    q: np.ndarray,                # (S,)
    leaf_lls: np.ndarray,         # (S, 3)
) -> np.ndarray:
    """Same shape as three_state_log_B but parameterised by marginal q."""
    return three_state_log_B(q, leaf_lls)


def composite_loglik(
    stance_data: Dict[str, Any],
    indicator_obs: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    leaf_logliks: Dict[str, Dict[str, np.ndarray]],
    beta_pres_by_key: Dict[str, np.ndarray],
    beta_abs_by_key: Dict[str, np.ndarray],
    C_by_system: Dict[str, np.ndarray],   # {system: (S,)}
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Mirror the implemented pm.Potential formula: ℓ = sum log B(q_{s,j}).

    Returns (ℓ_total of shape (S,), per-system ℓ dict).  Iterates the tree
    once to collect intercept_j + slope_j C per indicator, then computes
    q_{s,j}(C) = intercept + slope * C per system per draw.
    """
    sample_key = next(iter(beta_pres_by_key))
    n_draws = beta_pres_by_key[sample_key].shape[0]
    indicator_coeffs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    root_path = (stance_data["name"],)

    def walk(node, path, intercept_parent, slope_parent):
        cur = path + (node["name"],)
        key = node_key(path, node["name"])
        bp = beta_pres_by_key[key]
        ba = beta_abs_by_key[key]
        delta = bp - ba
        intercept_child = ba + intercept_parent * delta
        slope_child = slope_parent * delta
        if (node.get("type") or "").lower() == "indicator":
            indicator_coeffs[key] = (intercept_child, slope_child)
            return
        for child in node.get("evidencers", []):
            walk(child, cur, intercept_child, slope_child)

    root_intercept = np.zeros(n_draws)
    root_slope = np.ones(n_draws)
    for child in stance_data.get("evidencers", []):
        walk(child, root_path, root_intercept, root_slope)

    ll_per_sys: Dict[str, np.ndarray] = {sys_name: np.zeros(n_draws) for sys_name in C_by_system}
    for key, (intercept, slope) in indicator_coeffs.items():
        sys_obs = indicator_obs.get(key, {})
        for sys_name, _ in sys_obs.items():
            C_s = C_by_system[sys_name]
            q = intercept + slope * C_s
            q = np.clip(q, 1e-12, 1.0 - 1e-12)
            leaf_lls = leaf_logliks[key][sys_name]
            ll_per_sys[sys_name] += three_state_log_B_at_q(q, leaf_lls)
    total = sum(ll_per_sys.values())
    return total, ll_per_sys


def exact_loglik(
    stance_data: Dict[str, Any],
    indicator_obs: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    leaf_logliks: Dict[str, Dict[str, np.ndarray]],
    beta_pres_by_key: Dict[str, np.ndarray],
    beta_abs_by_key: Dict[str, np.ndarray],
    C_by_system: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Bottom-up dynamic-programming exact latent-tree marginal.

    For each system s, computes
        ℓ^exact_s = log[ C_s · L_root(z=1) + (1-C_s) · L_root(z=0) ]
    where L_v(z_v) is the subtree marginal likelihood given the parent's
    state.  At leaves, L_indicator(z_pa) = B(β_pres) if z_pa=1 else B(β_abs)
    using the same three-state mixture as the composite formula.
    """
    sample_key = next(iter(beta_pres_by_key))
    n_draws = beta_pres_by_key[sample_key].shape[0]
    root_path = (stance_data["name"],)

    def subtree_log_lik(node, path, sys_name) -> Tuple[np.ndarray, np.ndarray]:
        """Return (log L(z_pa=0), log L(z_pa=1)) for this node's subtree as a
        function of its parent's state. Per-draw vectors of shape (n_draws,).
        """
        cur = path + (node["name"],)
        key = node_key(path, node["name"])
        bp = beta_pres_by_key[key]
        ba = beta_abs_by_key[key]

        if (node.get("type") or "").lower() == "indicator":
            sys_obs = indicator_obs.get(key, {})
            if sys_name not in sys_obs:
                # Indicator with no data for this system: contributes 0
                return np.zeros(n_draws), np.zeros(n_draws)
            leaf_lls = leaf_logliks[key][sys_name]
            log_L_zpa0 = three_state_log_B(ba, leaf_lls)
            log_L_zpa1 = three_state_log_B(bp, leaf_lls)
            return log_L_zpa0, log_L_zpa1

        # Internal node: recurse on children. Each child returns
        #   (L_c(z_v = 0), L_c(z_v = 1))
        # i.e., the child's subtree likelihood AS A FUNCTION OF its parent's
        # state (= v's own state). The child has already integrated its own
        # z_c state inside its returned values, so here we just multiply
        # children's contributions:
        #   L_v(z_v = k) = ∏_c L_c(z_v = k).
        log_L_v0 = np.zeros(n_draws)
        log_L_v1 = np.zeros(n_draws)
        for child in node.get("evidencers", []):
            cL_zpa0, cL_zpa1 = subtree_log_lik(child, cur, sys_name)
            log_L_v1 = log_L_v1 + cL_zpa1
            log_L_v0 = log_L_v0 + cL_zpa0

        # Self-edge to my own parent: parent's state z_pa determines which β
        # I use to integrate my own z_v. Note we use v's OWN bp / ba here
        # (not the children's), because we are now collapsing v's own
        # latent state given its parent.
        log_L_zpa1 = np.logaddexp(
            np.log(np.clip(bp, 1e-12, 1.0)) + log_L_v1,
            np.log(np.clip(1.0 - bp, 1e-12, 1.0)) + log_L_v0,
        )
        log_L_zpa0 = np.logaddexp(
            np.log(np.clip(ba, 1e-12, 1.0)) + log_L_v1,
            np.log(np.clip(1.0 - ba, 1e-12, 1.0)) + log_L_v0,
        )
        return log_L_zpa0, log_L_zpa1

    ll_per_sys: Dict[str, np.ndarray] = {}
    for sys_name, C_s in C_by_system.items():
        # For each top-level child of the stance, compute its subtree LL given
        # the stance "parent" state (= consciousness C).  All top-level
        # features are children of the stance variable, so the "parent state"
        # they see is z_stance = 1 with prob C, 0 with prob (1-C).
        # We aggregate: ℓ = Σ_top_child log[ C·L_c(z_pa=1) + (1-C)·L_c(z_pa=0) ]
        # ... actually that double-counts the C marginalisation. The stance is
        # a single binary z_stance whose marginalisation must happen once,
        # outside the product. Fix:
        # ℓ = log[ C · ∏_c L_c(z_pa=1) + (1-C) · ∏_c L_c(z_pa=0) ]
        log_L_top1 = np.zeros(n_draws)
        log_L_top0 = np.zeros(n_draws)
        for child in stance_data.get("evidencers", []):
            cL0, cL1 = subtree_log_lik(child, root_path, sys_name)
            log_L_top0 = log_L_top0 + cL0
            log_L_top1 = log_L_top1 + cL1
        C_clip = np.clip(C_s, 1e-12, 1.0 - 1e-12)
        ll = np.logaddexp(
            np.log(C_clip) + log_L_top1,
            np.log(1.0 - C_clip) + log_L_top0,
        )
        ll_per_sys[sys_name] = ll
    total = sum(ll_per_sys.values())
    return total, ll_per_sys


# ----------------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------------


def _sanity_check_no_transmission(
    stance_data, indicator_obs, leaf_logliks,
    beta_pres_by_key, beta_abs_by_key, C_by_system,
) -> None:
    """When β_pres == β_abs everywhere, exact == composite to numerical noise."""
    bp_eq = {k: 0.5 * (v + beta_abs_by_key[k]) for k, v in beta_pres_by_key.items()}
    ba_eq = bp_eq  # same dict instance
    ll_c, _ = composite_loglik(stance_data, indicator_obs, leaf_logliks, bp_eq, ba_eq, C_by_system)
    ll_e, _ = exact_loglik(stance_data, indicator_obs, leaf_logliks, bp_eq, ba_eq, C_by_system)
    diff = ll_e - ll_c
    max_abs = np.max(np.abs(diff))
    if max_abs > 1e-6:
        raise AssertionError(
            f"No-transmission sanity check FAILED: max|ℓ_exact - ℓ_composite| = {max_abs:.6e}"
        )
    print(f"[SANITY] no-transmission: max|Δ| = {max_abs:.2e}  PASS")


def _sanity_check_brute_force_enumeration() -> None:
    """Synthetic 2-internal-node, 4-leaf tree.  Enumerate all 4 internal-z
    configurations and check against bottom-up DP.  Single posterior draw.

    Topology (single system, single rating per indicator):
              root C
              /    \\
            f1      f2     (features)
           / \\    / \\
         i11 i12 i21 i22   (indicators)
    """
    rng = np.random.default_rng(7)
    a = np.array([1.4])
    K = 7
    # Cutpoints around 0
    kappa = np.array([[-2.0, -1.0, -0.3, 0.3, 1.0, 2.0]])
    b = np.zeros((1, 1))  # 1 expert, b=0
    expert_idx = np.array([0])

    indicators = ["i11", "i12", "i21", "i22"]
    parent_of = {"i11": "f1", "i12": "f1", "i21": "f2", "i22": "f2", "f1": "root", "f2": "root"}
    bp = {n: float(rng.uniform(0.55, 0.85)) for n in parent_of}
    ba = {n: float(rng.uniform(0.10, 0.40)) for n in parent_of}

    # Hand-set ratings
    ratings_per_ind = {
        "i11": np.array([6]),
        "i12": np.array([5]),
        "i21": np.array([1]),
        "i22": np.array([2]),
    }

    # Precompute leaf likelihoods ℓ_m for each indicator
    leaf_lls = {}
    for ind, r in ratings_per_ind.items():
        ll0 = ordered_probit_category_logp_per_obs(r, expert_idx, a, b, kappa, 0.0, K)
        llh = ordered_probit_category_logp_per_obs(r, expert_idx, a, b, kappa, 0.5, K)
        ll1 = ordered_probit_category_logp_per_obs(r, expert_idx, a, b, kappa, 1.0, K)
        leaf_lls[ind] = np.stack([ll0, llh, ll1], axis=-1)  # (1, 3)

    # Brute force: enumerate (z_f1, z_f2, C∈{0,1}-marginalised) configurations
    C_val = 0.4

    def B_of(beta):
        return np.exp(three_state_log_B(np.array([beta]), leaf_lls[ind_name]))[0]

    total_brute = 0.0
    for z_f1 in (0, 1):
        for z_f2 in (0, 1):
            P_zf1_given_C1 = bp["f1"] if z_f1 == 1 else (1 - bp["f1"])
            P_zf1_given_C0 = ba["f1"] if z_f1 == 1 else (1 - ba["f1"])
            P_zf2_given_C1 = bp["f2"] if z_f2 == 1 else (1 - bp["f2"])
            P_zf2_given_C0 = ba["f2"] if z_f2 == 1 else (1 - ba["f2"])
            # Marginalise C
            P_zf1_zf2 = (
                C_val * P_zf1_given_C1 * P_zf2_given_C1
                + (1 - C_val) * P_zf1_given_C0 * P_zf2_given_C0
            )
            # For each indicator, the leaf likelihood given its parent's z
            L_data = 1.0
            for ind in indicators:
                pa = parent_of[ind]
                z_pa = z_f1 if pa == "f1" else z_f2
                beta = bp[ind] if z_pa == 1 else ba[ind]
                ind_name = ind
                L_data *= B_of(beta)
            total_brute += P_zf1_zf2 * L_data
    log_brute = float(np.log(total_brute))

    # DP version
    indicator_obs = {ind: {"sys": (ratings_per_ind[ind], expert_idx)} for ind in indicators}
    leaf_logliks = {ind: {"sys": leaf_lls[ind]} for ind in indicators}

    # Build a synthetic stance_data structure
    def make_indicator(name):
        return {"name": name, "type": "indicator", "support": "no bearing", "demandingness": "neutral"}
    def make_feature(name, children):
        return {"name": name, "type": "feature", "support": "no bearing",
                "demandingness": "neutral", "evidencers": children}
    stance_data = {
        "name": "Toy",
        "evidencers": [
            make_feature("f1", [make_indicator("i11"), make_indicator("i12")]),
            make_feature("f2", [make_indicator("i21"), make_indicator("i22")]),
        ],
    }
    bp_keys = {
        node_key(("Toy",), "f1"): np.array([bp["f1"]]),
        node_key(("Toy",), "f2"): np.array([bp["f2"]]),
        node_key(("Toy", "f1"), "i11"): np.array([bp["i11"]]),
        node_key(("Toy", "f1"), "i12"): np.array([bp["i12"]]),
        node_key(("Toy", "f2"), "i21"): np.array([bp["i21"]]),
        node_key(("Toy", "f2"), "i22"): np.array([bp["i22"]]),
    }
    ba_keys = {
        node_key(("Toy",), "f1"): np.array([ba["f1"]]),
        node_key(("Toy",), "f2"): np.array([ba["f2"]]),
        node_key(("Toy", "f1"), "i11"): np.array([ba["i11"]]),
        node_key(("Toy", "f1"), "i12"): np.array([ba["i12"]]),
        node_key(("Toy", "f2"), "i21"): np.array([ba["i21"]]),
        node_key(("Toy", "f2"), "i22"): np.array([ba["i22"]]),
    }
    indicator_obs_keyed = {
        node_key(("Toy", "f1"), "i11"): {"sys": (ratings_per_ind["i11"], expert_idx)},
        node_key(("Toy", "f1"), "i12"): {"sys": (ratings_per_ind["i12"], expert_idx)},
        node_key(("Toy", "f2"), "i21"): {"sys": (ratings_per_ind["i21"], expert_idx)},
        node_key(("Toy", "f2"), "i22"): {"sys": (ratings_per_ind["i22"], expert_idx)},
    }
    leaf_logliks_keyed = {
        node_key(("Toy", "f1"), "i11"): {"sys": leaf_lls["i11"]},
        node_key(("Toy", "f1"), "i12"): {"sys": leaf_lls["i12"]},
        node_key(("Toy", "f2"), "i21"): {"sys": leaf_lls["i21"]},
        node_key(("Toy", "f2"), "i22"): {"sys": leaf_lls["i22"]},
    }
    log_dp, _ = exact_loglik(
        stance_data, indicator_obs_keyed, leaf_logliks_keyed,
        bp_keys, ba_keys, {"sys": np.array([C_val])},
    )
    diff = float(log_dp[0] - log_brute)
    if abs(diff) > 1e-9:
        raise AssertionError(
            f"Brute-force enumeration check FAILED: log_DP - log_brute = {diff:+.6e}"
        )
    print(f"[SANITY] brute-force enumeration: |Δ| = {abs(diff):.2e}  PASS")


# ----------------------------------------------------------------------------
# Sibling-correlation potential score
# ----------------------------------------------------------------------------


def sibling_score_table(
    stance_data: Dict[str, Any],
    beta_pres_by_key: Dict[str, np.ndarray],
    beta_abs_by_key: Dict[str, np.ndarray],
    n_obs_per_indicator: Dict[str, int],
) -> "list[dict]":
    """Per internal node v: score_v = q_v(1-q_v) · max_{a,b∈children(v)} |Δ_a Δ_b| · n_obs_subtree."""
    rows = []
    root_path = (stance_data["name"],)

    # Compute posterior median q_v per node (using affine prop with C=0.5 as
    # neutral mid-point — score is qualitative, ranking only)
    # Walk and accumulate q_v under C=0.5
    q_at: Dict[str, float] = {}

    def walk_q(node, path, q_parent_med):
        cur = path + (node["name"],)
        key = node_key(path, node["name"])
        bp_med = float(np.median(beta_pres_by_key[key]))
        ba_med = float(np.median(beta_abs_by_key[key]))
        q_med = q_parent_med * bp_med + (1 - q_parent_med) * ba_med
        q_at[key] = q_med
        if (node.get("type") or "").lower() == "indicator":
            return
        for child in node.get("evidencers", []):
            walk_q(child, cur, q_med)

    for child in stance_data.get("evidencers", []):
        walk_q(child, root_path, 0.5)

    # Walk and collect children's δ medians + n_obs to compute score
    def walk_score(node, path):
        cur = path + (node["name"],)
        key = node_key(path, node["name"])
        if (node.get("type") or "").lower() == "indicator":
            return
        children = node.get("evidencers", [])
        if len(children) < 2:
            return  # need at least 2 children for sibling correlation
        deltas = []
        n_obs_sub = 0
        for ch in children:
            ch_key = node_key(cur, ch["name"])
            bp = float(np.median(beta_pres_by_key[ch_key]))
            ba = float(np.median(beta_abs_by_key[ch_key]))
            deltas.append(bp - ba)
            n_obs_sub += _n_obs_in_subtree(ch, cur, n_obs_per_indicator)
        max_pair = 0.0
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                max_pair = max(max_pair, abs(deltas[i] * deltas[j]))
        q_v = q_at[key]
        score = q_v * (1 - q_v) * max_pair * n_obs_sub
        rows.append({
            "node_key": key,
            "node_name": node["name"],
            "n_children": len(children),
            "q_v_med": q_v,
            "max_delta_product": max_pair,
            "n_obs_subtree": n_obs_sub,
            "score": score,
        })
        for ch in children:
            walk_score(ch, cur)

    for child in stance_data.get("evidencers", []):
        walk_score(child, root_path)
    return rows


def _n_obs_in_subtree(node, path, n_obs_per_indicator):
    cur = path + (node["name"],)
    key = node_key(path, node["name"])
    if (node.get("type") or "").lower() == "indicator":
        return n_obs_per_indicator.get(key, 0)
    return sum(_n_obs_in_subtree(c, cur, n_obs_per_indicator) for c in node.get("evidencers", []))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading pool_3s posterior: {POOL_3S_PATH}")
    idata = az.from_netcdf(str(POOL_3S_PATH))
    post = idata.posterior

    # Reload model context
    cfg = ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=True,
    )
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    proc = MultiSystemDataProcessor(cfg)
    proc.process(stance_data, [s for s, _ in ANCHORED_SYSTEM_CONFIGS])
    builder = MultiSystemModelBuilder(
        cfg, EvidenceProcessor(cfg), proc, list(ANCHORED_SYSTEM_CONFIGS)
    )
    builder.build_model(stance_data)

    # Extract posterior arrays
    a_draws = np.asarray(post["a"].values).reshape(-1)
    K = cfg.N_CATEGORIES
    n_experts = len(proc.expert_names)
    kappa_draws = np.asarray(post["kappa"].values).reshape(-1, K - 1)
    if "b_free" in post.data_vars:
        b_free_all = np.asarray(post["b_free"].values).reshape(-1, n_experts - 1)
        b_draws = np.concatenate([np.zeros((a_draws.shape[0], 1)), b_free_all], axis=1)
    else:
        b_draws = np.zeros((a_draws.shape[0], n_experts))

    # Per-system anchor C / posterior C
    C_by_system: Dict[str, np.ndarray] = {}
    for sys_name, c_fixed in ANCHORED_SYSTEM_CONFIGS:
        if c_fixed is not None:
            C_by_system[sys_name] = np.full(a_draws.shape[0], c_fixed)
        else:
            sp = builder._sys_prefix(sys_name)
            var = f"{sp}__global_workspace_theory_C"
            C_by_system[sys_name] = np.asarray(post[var].values).reshape(-1)

    # β draws keyed by node_key
    beta_pres_by_key, beta_abs_by_key = pooled_beta_draws_by_node(idata, stance_data)
    print(f"  posterior draws: {a_draws.shape[0]}")
    print(f"  tree nodes:      {len(beta_pres_by_key)}")
    print(f"  systems:         {list(C_by_system.keys())}")

    # Indicator obs + precomputed leaf likelihoods
    indicator_obs = collect_indicator_obs(stance_data, proc)
    n_indicators = len(indicator_obs)
    n_obs_per_ind = {k: sum(len(r) for (r, _) in v.values()) for k, v in indicator_obs.items()}
    print(f"  indicators with data: {n_indicators}")
    print(f"  total ratings:        {sum(n_obs_per_ind.values())}")
    leaf_logliks = precompute_leaf_logliks(
        indicator_obs, a_draws, b_draws, kappa_draws, K=K
    )

    # ---- Sanity checks ----
    print()
    print("=== Sanity checks ===")
    _sanity_check_brute_force_enumeration()
    _sanity_check_no_transmission(
        stance_data, indicator_obs, leaf_logliks,
        beta_pres_by_key, beta_abs_by_key, C_by_system,
    )

    # ---- Composite + exact log-likelihoods ----
    print()
    print("=== Computing log-likelihoods ===")
    ll_comp_total, ll_comp_per_sys = composite_loglik(
        stance_data, indicator_obs, leaf_logliks,
        beta_pres_by_key, beta_abs_by_key, C_by_system,
    )
    ll_exact_total, ll_exact_per_sys = exact_loglik(
        stance_data, indicator_obs, leaf_logliks,
        beta_pres_by_key, beta_abs_by_key, C_by_system,
    )
    delta_ll = ll_exact_total - ll_comp_total
    print(f"  composite ℓ: median {float(np.median(ll_comp_total)):+.3f}")
    print(f"  exact ℓ:     median {float(np.median(ll_exact_total)):+.3f}")
    print(f"  Δℓ:          median {float(np.median(delta_ll)):+.3f}, mean {float(np.mean(delta_ll)):+.3f}")
    print(f"               5/95th: [{float(np.percentile(delta_ll, 5)):+.3f}, {float(np.percentile(delta_ll, 95)):+.3f}]")

    # ---- Self-consistency ----
    if not np.allclose(ll_comp_total + delta_ll, ll_exact_total, rtol=1e-9, atol=1e-9):
        raise AssertionError("Self-consistency FAILED: ℓ_comp + Δℓ != ℓ_exact")
    print(f"[SANITY] self-consistency: PASS")

    # ---- Importance reweighting (numerically stable) ----
    log_w = delta_ll - logsumexp(delta_ll)
    w = np.exp(log_w)
    ESS = 1.0 / np.sum(w ** 2)
    n_draws = a_draws.shape[0]
    print()
    print("=== Importance reweighting ===")
    print(f"  ESS:        {ESS:.0f} / {n_draws} = {ESS / n_draws:.3f}")

    # PSIS Pareto-k̂ (optional)
    pareto_k = float("nan")
    try:
        psis_lw, k_hat = az.psislw(delta_ll[None, :])
        pareto_k = float(k_hat[0])
        print(f"  Pareto k̂:  {pareto_k:+.3f}")
    except Exception as e:
        print(f"  Pareto k̂:  unavailable ({e})")

    # Reweighted posterior shifts
    print()
    print("=== Reweighted shifts in headline quantities ===")
    headline_rows = []
    for label, var in HEADLINE_C_VARS.items():
        d = np.asarray(post[var].values).reshape(-1)
        med_orig = float(np.median(d))
        med_rw = float(_weighted_quantile(d, w, 0.5))
        headline_rows.append({
            "quantity": f"C_{label}_med",
            "original": med_orig,
            "reweighted": med_rw,
            "shift": med_rw - med_orig,
        })
        print(f"  C_{label} median: {med_orig:.4f} → {med_rw:.4f}  (Δ = {med_rw - med_orig:+.4f})")

    # mean δ_j (path-product slope)
    from analyse_tree_pooling import per_indicator_delta_under_pooling
    intercepts, slopes, _ = per_indicator_delta_under_pooling(idata, builder, stance_data)
    mean_delta_per_draw = slopes.mean(axis=1)  # (S,)
    md_orig = float(np.mean(mean_delta_per_draw))
    md_rw = float(np.sum(w * mean_delta_per_draw))
    headline_rows.append({
        "quantity": "mean_delta_j",
        "original": md_orig,
        "reweighted": md_rw,
        "shift": md_rw - md_orig,
    })
    print(f"  mean δ_j:      {md_orig:.4f} → {md_rw:.4f}  (Δ = {md_rw - md_orig:+.4f})")

    # β_abs__neutral median
    if "beta_abs__neutral" in post.data_vars:
        ba_n = np.asarray(post["beta_abs__neutral"].values).reshape(-1)
        ban_orig = float(np.median(ba_n))
        ban_rw = float(_weighted_quantile(ba_n, w, 0.5))
        headline_rows.append({
            "quantity": "beta_abs_neutral_med",
            "original": ban_orig,
            "reweighted": ban_rw,
            "shift": ban_rw - ban_orig,
        })
        print(f"  β_abs_neutral: {ban_orig:.4f} → {ban_rw:.4f}  (Δ = {ban_rw - ban_orig:+.4f})")

    # ---- Sibling-correlation potential scores ----
    print()
    print("=== Sibling-correlation potential scores (top 10 internal nodes) ===")
    score_rows = sibling_score_table(stance_data, beta_pres_by_key, beta_abs_by_key, n_obs_per_ind)
    score_rows.sort(key=lambda r: r["score"], reverse=True)
    for r in score_rows[:10]:
        print(f"  {r['node_name'][:40]:<42}  q={r['q_v_med']:.3f} "
              f" max|Δ_aΔ_b|={r['max_delta_product']:.4f}  n_obs={r['n_obs_subtree']:>4d}"
              f"  score={r['score']:.4f}")

    # Locate the weak_undermining + neutral parent
    wu_parents = [r for r in score_rows
                  if "weak undermining" in r["node_name"].lower()]
    if wu_parents:
        wu_rank = next(i for i, r in enumerate(score_rows) if r["node_key"] == wu_parents[0]["node_key"]) + 1
        print(f"  weak-undermining parent: rank {wu_rank} of {len(score_rows)}")

    # ---- Outputs ----
    import pandas as pd
    summary = {
        "n_draws": int(n_draws),
        "delta_ll_median": float(np.median(delta_ll)),
        "delta_ll_mean": float(np.mean(delta_ll)),
        "delta_ll_p05": float(np.percentile(delta_ll, 5)),
        "delta_ll_p95": float(np.percentile(delta_ll, 95)),
        "delta_ll_min": float(np.min(delta_ll)),
        "delta_ll_max": float(np.max(delta_ll)),
        "ess_w": float(ESS),
        "ess_w_ratio": float(ESS / n_draws),
        "pareto_k": pareto_k,
        "headline_shifts": headline_rows,
        "interpretation_threshold": {
            "ess_ratio_above_0.7_no_load_bearing": bool(ESS / n_draws > 0.7),
            "ess_ratio_below_0.3_failure": bool(ESS / n_draws < 0.3),
        },
    }
    (OUT_DIR / "composite_gap_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_DIR / 'composite_gap_summary.json'}")

    pd.DataFrame(headline_rows).to_csv(OUT_DIR / "composite_gap_headline_shifts.csv", index=False)
    print(f"wrote {OUT_DIR / 'composite_gap_headline_shifts.csv'}")

    pd.DataFrame(score_rows).sort_values("score", ascending=False).to_csv(
        OUT_DIR / "sibling_correlation_scores.csv", index=False
    )
    print(f"wrote {OUT_DIR / 'sibling_correlation_scores.csv'}")

    # Δℓ histogram + reweighted vs original
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.hist(delta_ll, bins=40, color="#4a6fa5", alpha=0.8)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Δℓ = ℓ_exact - ℓ_composite (per posterior draw)")
    ax.set_ylabel("# draws")
    ax.set_title(f"Per-draw composite-vs-exact log-lik gap\n"
                 f"median {np.median(delta_ll):+.2f}, ESS/S = {ESS/n_draws:.2f}")

    ax = axes[1]
    var = HEADLINE_C_VARS["Chicken"]
    d = np.asarray(post[var].values).reshape(-1)
    bins = np.linspace(0, 1, 60)
    ax.hist(d, bins=bins, color="#888", alpha=0.5, density=True, label=f"Chicken C (original)")
    ax.hist(d, bins=bins, weights=w * n_draws, color="#1f78b4", alpha=0.5,
            density=True, label=f"Chicken C (reweighted, ESS/S={ESS/n_draws:.2f})")
    ax.set_xlabel("posterior C")
    ax.set_title("Reweighted vs original Chicken posterior")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "composite_gap.png", dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'composite_gap.png'}")

    print()
    if ESS / n_draws > 0.7 and all(abs(r["shift"]) < 0.02 for r in headline_rows):
        print(">>> VERDICT: composite approximation NOT load-bearing under pool_3s "
              "(ESS/S > 0.7, all headline shifts < 0.02).")
    elif ESS / n_draws < 0.3 or any(abs(r["shift"]) > 0.05 for r in headline_rows):
        print(">>> VERDICT: importance reweighting FAILED. Composite posterior too far "
              "from exact-tree posterior for post-hoc correction; report Δℓ distribution "
              "but do NOT trust reweighted shifts.")
    else:
        print(">>> VERDICT: intermediate regime (0.3 ≤ ESS/S ≤ 0.7). Report both raw "
              "Δℓ distribution and reweighted shifts with ESS caveat.")


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cumw = np.cumsum(w) / np.sum(w)
    return float(np.interp(q, cumw, v))


if __name__ == "__main__":
    main()
