"""Proper exact-tree leaf-updated posterior predictive checks.

Computes per-indicator P(m_j | all observed ratings in system, θ) via
two-pass belief propagation (sum-product) on the latent-state tree, then
uses these "exact-tree responsibilities" ρ_m to predict category
distributions at focus cells.

Contrast with the composite-style PPC (in `dcm_ppc.py`), which computes
ρ_m using only the indicator's OWN data — assuming leaves are
conditionally independent given (C, β). The exact-tree PPC properly
accounts for correlations through internal-node z's.

Two passes per (system, draw):
  Forward (bottom-up):
    For leaf j with parent pa:
      log_D_j(z_pa = k) = log B(β_for_zpa_k_j) = log [Σ_m π_m(β) exp(ll_m_j)]
    For internal node v with parent pa and children c:
      log_L_v_at_zv(z_v = k) = Σ_c log_D_c(z_v = k)
      log_D_v(z_pa = k) = logsumexp(log β_for_k_v + log_L_v_at_zv(1),
                                    log(1-β_for_k_v) + log_L_v_at_zv(0))

  Backward (top-down):
    For top-level feature v (parent = stance root):
      log_O_v(z_v = k) = logsumexp(
        log(C) + log P(z_v=k | z_root=1) + (log_L_root_at_zroot(1) - log_D_v(z_pa=1)),
        log(1-C) + log P(z_v=k | z_root=0) + (log_L_root_at_zroot(0) - log_D_v(z_pa=0)),
      )
    For non-top v with internal parent pa:
      log_O_v(z_v = k) = logsumexp_{z_pa} log_O_pa(z_pa) + log P(z_v=k | z_pa)
                                          + (log_L_pa_at_zv(z_pa) - log_D_v(z_pa))

Then for each leaf indicator j:
  log_pa_excl_j(z_pa) = log_L_pa_at_zv(z_pa) - log_D_j(z_pa) + log_O_pa(z_pa)
  P(m_j | y_all) ∝ exp(ll_m_j) * Σ_{z_pa} exp(log_pa_excl_j(z_pa)) * π_m(β_for_zpa)

Sanity checks:
  1. P(m_j | y_all) sums to 1 across m for every (j, draw, system).
  2. log_total_likelihood reconstructed from forward pass alone should match
     the standalone composite_vs_exact_diagnostic.exact_loglik to numerical
     precision.
  3. Sum over m of unnormalised log-mass for one indicator should equal the
     total system log-likelihood (both are Σ over all latent configurations).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

from analyse_tree_pooling import pooled_beta_draws_by_node
from composite_vs_exact_diagnostic import (
    collect_indicator_obs,
    exact_loglik,
    precompute_leaf_logliks,
)
from dcm_model import (
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    load_data,
    node_key,
)
from dcm_model_exact_tree import MultiSystemExactTreeBuilder
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS
from tree_propagation_diagnostic import _EXPERT_ANON_MAP

EXACT_PROD_PATH = Path("results/gwt_exact_tree/three_state_pooled_abs_by_sd_exact_anchored.nc")
OUT_DIR = Path("notebooks/meeting_prep_arvo_2026-04-27/figs_round2")
STANCE = "Global Workspace Theory"

REAL_BY_ANON = {v: k for k, v in dict(_EXPERT_ANON_MAP).items()}

FOCUS_CELLS = [
    ("E_cross", "Human", "right"),
    ("E_cross", "ELIZA", "left"),
    ("E_chickenB", "Chicken", "right"),
    ("E_llmC", "2024 Leading Chat LLMs", "right"),
]


# ---------------------------------------------------------------------------
# Forward pass (bottom-up)
# ---------------------------------------------------------------------------


def _three_state_log_B(beta: np.ndarray, leaf_lls: np.ndarray) -> np.ndarray:
    """log B(β) = log [(1-β)² ℓ_0 + 2β(1-β) ℓ_½ + β² ℓ_1] per draw.

    beta: (S,) array of β values per draw.
    leaf_lls: (S, 3) array of (ll_0, ll_½, ll_1) per draw.
    Returns (S,).
    """
    eps = 1e-12
    b = np.clip(beta, eps, 1.0 - eps)
    log_w0 = 2.0 * np.log(1.0 - b)
    log_w1 = np.log(2.0) + np.log(b) + np.log(1.0 - b)
    log_w2 = 2.0 * np.log(b)
    return logsumexp(
        np.stack([
            log_w0 + leaf_lls[:, 0],
            log_w1 + leaf_lls[:, 1],
            log_w2 + leaf_lls[:, 2],
        ], axis=-1),
        axis=-1,
    )


def forward_pass(
    stance_data: Dict,
    indicator_obs: Dict[str, Dict],
    leaf_logliks: Dict[str, Dict[str, np.ndarray]],
    bp_dict: Dict[str, np.ndarray],
    ba_dict: Dict[str, np.ndarray],
    sys_name: str,
) -> Tuple[
    Dict[str, np.ndarray],   # log_D[node_key] of shape (S, 2)
    Dict[str, np.ndarray],   # log_L_at_zv[node_key] of shape (S, 2) — internals only
    np.ndarray,              # log_L_root_at_zroot of shape (S, 2)
]:
    """Bottom-up forward DP. Stores log_D (downward msg) per node and
    log_L_at_zv (node's internal likelihood given own state) per internal node.
    """
    log_D: Dict[str, np.ndarray] = {}
    log_L_at_zv: Dict[str, np.ndarray] = {}
    eps = 1e-12
    sample_key = next(iter(bp_dict))
    n_draws = bp_dict[sample_key].shape[0]

    root_path = (stance_data["name"],)

    def walk(node, path):
        cur = path + (node["name"],)
        key = node_key(path, node["name"])
        bp = np.clip(bp_dict[key], eps, 1.0 - eps)
        ba = np.clip(ba_dict[key], eps, 1.0 - eps)

        if (node.get("type") or "").lower() == "indicator":
            sys_obs = indicator_obs.get(key, {})
            if sys_name not in sys_obs:
                # No data for this (system, indicator) — treat as 0 contribution
                log_D[key] = np.zeros((n_draws, 2))
                return
            leaf_lls = leaf_logliks[key][sys_name]  # (S, 3)
            log_D_zpa0 = _three_state_log_B(ba, leaf_lls)
            log_D_zpa1 = _three_state_log_B(bp, leaf_lls)
            log_D[key] = np.stack([log_D_zpa0, log_D_zpa1], axis=-1)
            return

        # Internal node: recurse first
        for child in node.get("evidencers", []):
            walk(child, cur)

        # log_L_at_zv(z_v=k) = Σ_c log_D_c(z_v=k)
        log_L_v0 = np.zeros(n_draws)
        log_L_v1 = np.zeros(n_draws)
        for child in node.get("evidencers", []):
            ck = node_key(cur, child["name"])
            log_L_v0 += log_D[ck][:, 0]
            log_L_v1 += log_D[ck][:, 1]
        log_L_at_zv[key] = np.stack([log_L_v0, log_L_v1], axis=-1)

        # log_D_v(z_pa=k): integrate v's own state given parent
        log_D_zpa1 = np.logaddexp(
            np.log(bp) + log_L_v1, np.log(1.0 - bp) + log_L_v0,
        )
        log_D_zpa0 = np.logaddexp(
            np.log(ba) + log_L_v1, np.log(1.0 - ba) + log_L_v0,
        )
        log_D[key] = np.stack([log_D_zpa0, log_D_zpa1], axis=-1)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path)

    # Root-level "L_at_zroot": ∏_top_features D_f(z_root)
    log_L_root_z0 = np.zeros(n_draws)
    log_L_root_z1 = np.zeros(n_draws)
    for child in stance_data.get("evidencers", []):
        ck = node_key(root_path, child["name"])
        log_L_root_z0 += log_D[ck][:, 0]
        log_L_root_z1 += log_D[ck][:, 1]
    log_L_root_at_zroot = np.stack([log_L_root_z0, log_L_root_z1], axis=-1)

    return log_D, log_L_at_zv, log_L_root_at_zroot


# ---------------------------------------------------------------------------
# Backward pass (top-down)
# ---------------------------------------------------------------------------


def backward_pass(
    stance_data: Dict,
    log_D: Dict[str, np.ndarray],
    log_L_at_zv: Dict[str, np.ndarray],
    log_L_root_at_zroot: np.ndarray,
    bp_dict: Dict[str, np.ndarray],
    ba_dict: Dict[str, np.ndarray],
    C_per_draw: np.ndarray,   # (S,)
) -> Dict[str, np.ndarray]:
    """Top-down backward DP. Returns log_O[node_key] of shape (S, 2)
    where log_O_v(z_v=k) = log P(z_v=k, y_outside_v_subtree | C).
    """
    log_O: Dict[str, np.ndarray] = {}
    eps = 1e-12
    n_draws = C_per_draw.shape[0]
    log_C = np.log(np.clip(C_per_draw, eps, 1.0 - eps))
    log_1mC = np.log(np.clip(1.0 - C_per_draw, eps, 1.0 - eps))

    root_path = (stance_data["name"],)

    # Top-level features: parent is the stance root
    # log_O_v(z_v=k) = logsumexp_{z_root} log P(z_root) + log P(z_v=k | z_root)
    #                                    + (log_L_root_at_zroot(z_root) - log_D_v(z_root))
    for child in stance_data.get("evidencers", []):
        ck = node_key(root_path, child["name"])
        bp = np.clip(bp_dict[ck], eps, 1.0 - eps)
        ba = np.clip(ba_dict[ck], eps, 1.0 - eps)

        # sibling_contribution(z_root) = log_L_root_at_zroot(z_root) - log_D_v(z_root)
        sib0 = log_L_root_at_zroot[:, 0] - log_D[ck][:, 0]
        sib1 = log_L_root_at_zroot[:, 1] - log_D[ck][:, 1]

        # log P(z_v=1 | z_root=1) = log(bp); log P(z_v=1 | z_root=0) = log(ba)
        # log P(z_v=0 | z_root=1) = log(1-bp); log P(z_v=0 | z_root=0) = log(1-ba)
        log_O_v_zv1 = np.logaddexp(
            log_C + np.log(bp) + sib1,
            log_1mC + np.log(ba) + sib0,
        )
        log_O_v_zv0 = np.logaddexp(
            log_C + np.log(1.0 - bp) + sib1,
            log_1mC + np.log(1.0 - ba) + sib0,
        )
        log_O[ck] = np.stack([log_O_v_zv0, log_O_v_zv1], axis=-1)

    # Recurse down the tree from each top-level feature
    def walk(node, path):
        cur = path + (node["name"],)
        key = node_key(path, node["name"])
        if (node.get("type") or "").lower() == "indicator":
            return
        # Compute O_v for each child of `node`
        log_L_pa_at_zv = log_L_at_zv[key]   # (S, 2), pa = current node
        log_O_pa = log_O[key]                # (S, 2)
        for child in node.get("evidencers", []):
            ck = node_key(cur, child["name"])
            bp = np.clip(bp_dict[ck], eps, 1.0 - eps)
            ba = np.clip(ba_dict[ck], eps, 1.0 - eps)

            # sibling_contribution(z_pa) = log_L_pa_at_zv(z_pa) - log_D_child(z_pa)
            sib0 = log_L_pa_at_zv[:, 0] - log_D[ck][:, 0]
            sib1 = log_L_pa_at_zv[:, 1] - log_D[ck][:, 1]

            log_O_v_zv1 = np.logaddexp(
                log_O_pa[:, 1] + np.log(bp) + sib1,
                log_O_pa[:, 0] + np.log(ba) + sib0,
            )
            log_O_v_zv0 = np.logaddexp(
                log_O_pa[:, 1] + np.log(1.0 - bp) + sib1,
                log_O_pa[:, 0] + np.log(1.0 - ba) + sib0,
            )
            log_O[ck] = np.stack([log_O_v_zv0, log_O_v_zv1], axis=-1)
            walk(child, cur)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path)

    return log_O


# ---------------------------------------------------------------------------
# Per-indicator P(m | y_all)
# ---------------------------------------------------------------------------


def per_indicator_exact_tree_marginal(
    stance_data: Dict,
    indicator_obs: Dict[str, Dict],
    leaf_logliks: Dict[str, Dict[str, np.ndarray]],
    log_D: Dict[str, np.ndarray],
    log_L_at_zv: Dict[str, np.ndarray],
    log_O: Dict[str, np.ndarray],
    bp_dict: Dict[str, np.ndarray],
    ba_dict: Dict[str, np.ndarray],
    sys_name: str,
) -> Dict[str, np.ndarray]:
    """Compute P(m_j | y_all_in_system, θ) for each leaf indicator with data.

    Returns dict node_key → array of shape (S, 3).
    """
    out: Dict[str, np.ndarray] = {}
    eps = 1e-12
    sample_key = next(iter(bp_dict))
    n_draws = bp_dict[sample_key].shape[0]
    root_path = (stance_data["name"],)

    # First, build a parent map by walking
    parent_of: Dict[str, str] = {}

    def walk_parent(node, path, parent_key):
        cur = path + (node["name"],)
        key = node_key(path, node["name"])
        parent_of[key] = parent_key
        for child in node.get("evidencers", []):
            walk_parent(child, cur, key)

    for child in stance_data.get("evidencers", []):
        walk_parent(child, root_path, "_ROOT_")

    for key, sys_obs in indicator_obs.items():
        if sys_name not in sys_obs:
            continue
        pa_key = parent_of[key]
        if pa_key == "_ROOT_":
            # Indicator directly under the root? Unusual but handle:
            # parent posterior is C / (1-C) directly
            raise NotImplementedError("Root-level indicator not expected for GWT")
        log_L_pa = log_L_at_zv[pa_key]   # (S, 2)
        log_O_pa = log_O[pa_key]          # (S, 2)
        log_D_j = log_D[key]              # (S, 2)
        # log_pa_excl_j(z_pa) = log_L_pa(z_pa) - log_D_j(z_pa) + log_O_pa(z_pa)
        log_pa_excl = log_L_pa - log_D_j + log_O_pa  # (S, 2)
        # Normalise per draw
        log_pa_excl_norm = log_pa_excl - logsumexp(log_pa_excl, axis=-1, keepdims=True)
        pa_excl_post = np.exp(log_pa_excl_norm)  # (S, 2) — P(z_pa | y_all_excluding_j)

        # Per m, π_m(β_for_zpa)
        bp = np.clip(bp_dict[key], eps, 1.0 - eps)
        ba = np.clip(ba_dict[key], eps, 1.0 - eps)
        # π_m(β) = (1-β)², 2β(1-β), β²
        # Stack over (m, z_pa) per draw: (S, 3, 2)
        pi_m_at_zpa = np.stack([
            np.stack([(1 - ba) ** 2, (1 - bp) ** 2], axis=-1),     # m=0
            np.stack([2 * ba * (1 - ba), 2 * bp * (1 - bp)], axis=-1),  # m=1
            np.stack([ba ** 2, bp ** 2], axis=-1),                  # m=2
        ], axis=1)  # (S, 3, 2)

        # weighted_m[s, m] = Σ_zpa pa_excl_post[s, zpa] * pi_m_at_zpa[s, m, zpa]
        weighted_m = np.einsum("sz,smz->sm", pa_excl_post, pi_m_at_zpa)  # (S, 3)
        leaf_lls = leaf_logliks[key][sys_name]   # (S, 3)
        log_post_m_unnorm = leaf_lls + np.log(np.clip(weighted_m, 1e-300, None))
        log_post_m = log_post_m_unnorm - logsumexp(log_post_m_unnorm, axis=-1, keepdims=True)
        out[key] = np.exp(log_post_m)
    return out


# ---------------------------------------------------------------------------
# Sanity check: total log likelihood from forward pass matches exact_loglik
# ---------------------------------------------------------------------------


def total_loglik_from_forward(
    log_L_root_at_zroot: np.ndarray,
    C_per_draw: np.ndarray,
) -> np.ndarray:
    """log P(y_all_in_system | C, θ) = logsumexp(log C + log_L_root(1), log(1-C) + log_L_root(0))."""
    eps = 1e-12
    log_C = np.log(np.clip(C_per_draw, eps, 1.0 - eps))
    log_1mC = np.log(np.clip(1.0 - C_per_draw, eps, 1.0 - eps))
    return np.logaddexp(
        log_C + log_L_root_at_zroot[:, 1],
        log_1mC + log_L_root_at_zroot[:, 0],
    )


# ---------------------------------------------------------------------------
# Focus cell PPC using exact-tree ρ_m
# ---------------------------------------------------------------------------


def _ordered_probit_probs(kappa: np.ndarray, eta: np.ndarray, K: int) -> np.ndarray:
    """(S, K) category probabilities."""
    cum = norm.cdf(kappa - eta[:, None])
    zeros = np.zeros((cum.shape[0], 1))
    ones = np.ones((cum.shape[0], 1))
    cum_full = np.concatenate([zeros, cum, ones], axis=1)
    probs = np.diff(cum_full, axis=1)
    return np.clip(probs, 1e-12, 1.0)


def focus_cell_ppc_exact_tree(
    pm_per_indicator: Dict[str, Dict[str, np.ndarray]],   # {sys: {ikey: (S, 3)}}
    proc,
    a_draws: np.ndarray, b_draws: np.ndarray, kappa_draws: np.ndarray,
    K: int = 7,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """For each (rater, system) cell: predicted category distribution = mean over
    cell's indicators of Σ_m P(m_j | y_all) * P_OP(k | rater, m).

    Returns dict (expert_real, system) → stats.
    """
    n_experts = len(proc.expert_names)
    S = a_draws.shape[0]
    # Per-rater emission components
    pk_components = []  # list (E,) of stacked (3, S, K) arrays
    for e in range(n_experts):
        eta0 = b_draws[:, e]
        eta1 = eta0 + 0.5 * a_draws
        eta2 = eta0 + a_draws
        pk0 = _ordered_probit_probs(kappa_draws, eta0, K)
        pk1 = _ordered_probit_probs(kappa_draws, eta1, K)
        pk2 = _ordered_probit_probs(kappa_draws, eta2, K)
        pk_components.append(np.stack([pk0, pk1, pk2], axis=0))  # (3, S, K)

    # Bucket observations by (expert_idx, system)
    buckets: Dict[Tuple[int, str], List[str]] = defaultdict(list)
    for sys_name, sys_obs in proc.system_observations.items():
        for ikey, obs_list in sys_obs.items():
            if ikey not in pm_per_indicator.get(sys_name, {}):
                continue
            for expert_idx, _rating in obs_list:
                buckets[(expert_idx, sys_name)].append(ikey)

    results = {}
    for (e, sys_name), ikey_list in buckets.items():
        if not ikey_list:
            continue
        # P(m | y_all) per indicator: (N, S, 3)
        pm_stack = np.stack([pm_per_indicator[sys_name][ik] for ik in ikey_list], axis=0)
        # Mean ρ_m across cell indicators per draw: (S, 3)
        rho_mean = pm_stack.mean(axis=0)
        pk_e = pk_components[e]   # (3, S, K)
        # pred[s, k] = Σ_m rho_mean[s, m] * pk_e[m, s, k]
        pred = np.einsum("sm,msk->sk", rho_mean, pk_e)
        pred_left = pred[:, 0]
        pred_right = pred[:, K - 1]
        # Compute observed left/right
        obs_ratings = []
        sys_obs = proc.system_observations.get(sys_name, {})
        for ikey in ikey_list:
            for ex_idx, r in sys_obs.get(ikey, []):
                if ex_idx == e:
                    obs_ratings.append(int(r))
                    break  # one obs per (rater, indicator)
        obs_arr = np.array(obs_ratings, dtype=int)
        obs_left = float(np.mean(obs_arr == 0)) if len(obs_arr) else float("nan")
        obs_right = float(np.mean(obs_arr == K - 1)) if len(obs_arr) else float("nan")
        results[(proc.expert_names[e], sys_name)] = {
            "n_obs": len(ikey_list),
            "obs_left": obs_left,
            "obs_right": obs_right,
            "pred_left_mean": float(pred_left.mean()),
            "pred_left_lo": float(np.percentile(pred_left, 3)),
            "pred_left_hi": float(np.percentile(pred_left, 97)),
            "pred_right_mean": float(pred_right.mean()),
            "pred_right_lo": float(np.percentile(pred_right, 3)),
            "pred_right_hi": float(np.percentile(pred_right, 97)),
            "delta_left": float(pred_left.mean()) - obs_left,
            "delta_right": float(pred_right.mean()) - obs_right,
        }
    return results


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading exact-tree production fit: {EXACT_PROD_PATH}")
    idata = az.from_netcdf(str(EXACT_PROD_PATH))
    post = idata.posterior

    cfg = ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=True,
        BETA_ABS_BY_SUPPORT_DEMAND=True,
    )
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    proc = MultiSystemDataProcessor(cfg)
    proc.process(stance_data, [s for s, _ in ANCHORED_SYSTEM_CONFIGS])
    builder = MultiSystemExactTreeBuilder(
        cfg, EvidenceProcessor(cfg), proc, list(ANCHORED_SYSTEM_CONFIGS)
    )
    builder.build_model(stance_data)

    # Subsample posterior draws (full posterior is 8000)
    n_draws = 500
    rng = np.random.default_rng(42)
    a_all = np.asarray(post["a"].values).reshape(-1)
    S_total = a_all.shape[0]
    if n_draws < S_total:
        idx = rng.choice(S_total, size=n_draws, replace=False)
    else:
        idx = np.arange(S_total)
    print(f"  posterior draws: {S_total}, subsampled to {len(idx)}")

    a_draws = a_all[idx]
    K = cfg.N_CATEGORIES
    n_experts = len(proc.expert_names)
    kappa_draws = np.asarray(post["kappa"].values).reshape(-1, K - 1)[idx]
    if "b_free" in post.data_vars:
        b_free_all = np.asarray(post["b_free"].values).reshape(-1, n_experts - 1)
        b_draws = np.concatenate([np.zeros((len(idx), 1)), b_free_all[idx]], axis=1)
    else:
        b_draws = np.zeros((len(idx), n_experts))

    # β draws (subset)
    bp_full, ba_full = pooled_beta_draws_by_node(idata, stance_data)
    bp_dict = {k: v[idx] for k, v in bp_full.items()}
    ba_dict = {k: v[idx] for k, v in ba_full.items()}

    # Per-system C
    C_by_system: Dict[str, np.ndarray] = {}
    for sys_name, c_fixed in ANCHORED_SYSTEM_CONFIGS:
        if c_fixed is not None:
            C_by_system[sys_name] = np.full(len(idx), c_fixed)
        else:
            sp = builder._sys_prefix(sys_name)
            var = f"{sp}__global_workspace_theory_C"
            C_by_system[sys_name] = np.asarray(post[var].values).reshape(-1)[idx]

    indicator_obs = collect_indicator_obs(stance_data, proc)
    leaf_logliks = precompute_leaf_logliks(
        indicator_obs, a_draws, b_draws, kappa_draws, K=K
    )

    # ---- Forward + backward pass per system ----
    pm_per_indicator: Dict[str, Dict[str, np.ndarray]] = {}
    print()
    print("=== Forward + backward DP per system ===")
    for sys_name in proc.systems:
        log_D, log_L_at_zv, log_L_root = forward_pass(
            stance_data, indicator_obs, leaf_logliks,
            bp_dict, ba_dict, sys_name,
        )
        log_O = backward_pass(
            stance_data, log_D, log_L_at_zv, log_L_root,
            bp_dict, ba_dict, C_by_system[sys_name],
        )
        pm = per_indicator_exact_tree_marginal(
            stance_data, indicator_obs, leaf_logliks,
            log_D, log_L_at_zv, log_O,
            bp_dict, ba_dict, sys_name,
        )
        pm_per_indicator[sys_name] = pm

        # Sanity check 1: P(m | y_all) sums to 1
        for k, p in pm.items():
            row_sums = p.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-9):
                raise AssertionError(
                    f"{sys_name}/{k}: P(m | y_all) doesn't sum to 1; max dev = "
                    f"{float(np.max(np.abs(row_sums - 1.0))):.3e}"
                )

        # Sanity check 2: total system log-lik from forward matches exact_loglik
        log_lik_fwd = total_loglik_from_forward(log_L_root, C_by_system[sys_name])
        # Compare with standalone exact_loglik per-system (recomputed for the same draws)
        # exact_loglik returns per-system; build a single-system slice
        ll_single_total, ll_single_per_sys = exact_loglik(
            stance_data, indicator_obs, leaf_logliks,
            bp_dict, ba_dict, {sys_name: C_by_system[sys_name]},
        )
        diff = log_lik_fwd - ll_single_per_sys[sys_name]
        max_abs = float(np.max(np.abs(diff)))
        if max_abs > 1e-6:
            raise AssertionError(
                f"{sys_name}: forward-pass log-lik does not match exact_loglik; "
                f"max |Δ| = {max_abs:.6e}"
            )
        print(f"  {sys_name:<30} forward log-lik vs exact_loglik: max |Δ| = {max_abs:.2e}  PASS")
        print(f"    indicators with data: {len(pm)}")

    # ---- Focus-cell PPC ----
    print()
    print("=== Focus-cell PPC (exact-tree ρ_m) ===")
    ppc_stats = focus_cell_ppc_exact_tree(
        pm_per_indicator, proc, a_draws, b_draws, kappa_draws, K=K,
    )

    rows = []
    print(f"  {'cell':<35}  {'tail':<6}  {'obs':>6}  {'pred [lo, hi]':<25}  {'Δ':>8}")
    for anon, sys_name, tail in FOCUS_CELLS:
        real = REAL_BY_ANON.get(anon)
        key = (real, sys_name)
        if key not in ppc_stats:
            continue
        s = ppc_stats[key]
        if tail == "left":
            obs = s["obs_left"]
            pred_m = s["pred_left_mean"]
            pred_lo = s["pred_left_lo"]
            pred_hi = s["pred_left_hi"]
            d = s["delta_left"]
        else:
            obs = s["obs_right"]
            pred_m = s["pred_right_mean"]
            pred_lo = s["pred_right_lo"]
            pred_hi = s["pred_right_hi"]
            d = s["delta_right"]
        cell = f"{anon} × {sys_name}"
        pred_str = f"{pred_m:.3f} [{pred_lo:.3f}, {pred_hi:.3f}]"
        print(f"  {cell:<35}  {tail:<6}  {obs:>6.3f}  {pred_str:<25}  {d:>+8.3f}")
        rows.append({
            "fit": "exact_tree_production",
            "ppc_type": "exact_tree",
            "cell": cell,
            "tail": tail,
            "obs": obs,
            "pred_mean": pred_m,
            "pred_lo": pred_lo,
            "pred_hi": pred_hi,
            "delta": d,
        })

    # Compare with composite-style PPC numbers from the previous run
    print()
    print("=== Comparison: composite-style ρ_m vs exact-tree ρ_m on same fit ===")
    composite_ppc_path = OUT_DIR / "exact_tree_production_ppc.csv"
    if composite_ppc_path.exists():
        df_comp = pd.read_csv(composite_ppc_path)
        print(f"  {'cell':<35}  {'tail':<6}  {'composite Δ':>10}  {'exact-tree Δ':>10}")
        for r in rows:
            comp_match = df_comp[(df_comp["cell"] == r["cell"]) & (df_comp["tail"] == r["tail"])]
            if len(comp_match):
                comp_delta = float(comp_match["delta"].iloc[0])
                print(f"  {r['cell']:<35}  {r['tail']:<6}  {comp_delta:>+10.3f}  {r['delta']:>+10.3f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "exact_tree_production_ppc_BP.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'exact_tree_production_ppc_BP.csv'}")


if __name__ == "__main__":
    main()
