"""β_abs__neutral counterfactual decomposition on pool_3s posterior.

Quantitative confirmation of B.3a/b's interpretation: under `pool_3s`, the
apparent `weak_undermining + neutral` sign-flip resolution is ~95% inherited
from a shared `β_abs__neutral` shift (paper 0.50 → posterior 0.29) rather
than from group-specific evidence on weak-undermining.

Four-scenario Shapley-style decomposition on the 5 sign-flip indicators:

    (a) posterior as fitted              — both posterior
    (b) shared β_abs__neutral reset to 0.5  — own β_pres still posterior
    (c) own β_pres reset to paper 0.429   — shared β_abs still posterior
    (d) both reset to paper values        — both at paper

For each scenario, per sign-flip indicator: δ_j (path-product slope),
q_j(C=0.999), q_j(C=0.001), focus-cell PPC predicted right tail at
E_cross × Human, predicted left tail at E_cross × ELIZA.

Critical implementation point (per ChatGPT round-2 correction): we must
re-propagate q_j from primitive β draws under each scenario; cached
*_p Deterministics in the InferenceData were computed during sampling
with the un-patched β values and would silently return stale numbers.

Honest limitation flag: holding β_abs__neutral = 0.5 while reusing the
posterior C draws (jointly fit with the original β_abs) is a partial
counterfactual. For δ_j (path products of β only) the result is exact.
For PPCs (which depend on q_{s,j}(C, β)), predicted tails are computed
conditional on the un-counterfactual C posterior.

Output: notebooks/meeting_prep_arvo_2026-04-27/figs_round2/beta_abs_neutral_counterfactual.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import arviz as az
import numpy as np
import pandas as pd
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
from gwt_reference_recovery_analysis import (
    ANCHORED_SYSTEM_CONFIGS,
    propagate_affine_indicator_coefficients_from_draws,
    build_indicator_index,
)

POOL_3S_PATH = Path("results/gwt_tree_pooling/three_state_pooled_anchored.nc")
OUT_DIR = Path("notebooks/meeting_prep_arvo_2026-04-27/figs_round2")
STANCE = "Global Workspace Theory"

# Five sign-flip indicators (per Stage 1c of notebook 18). All share the
# `weak undermining + neutral` parent subfeature "Autonomous Subparts".
SIGN_FLIP_NODE_KEYS = [
    "Global Workspace Theory > Coherence > Autonomous Subparts > Information Transfer",
    "Global Workspace Theory > Coherence > Autonomous Subparts > Learning Transfer",
    "Global Workspace Theory > Coherence > Autonomous Subparts > Functional Subparts",
    "Global Workspace Theory > Coherence > Autonomous Subparts > Information Transfer Architecture",
    "Global Workspace Theory > Coherence > Autonomous Subparts > Conflicting Subparts",
]
WU_NEUTRAL_NODE_KEY = "Global Workspace Theory > Coherence > Autonomous Subparts"

PAPER_BETA_PRES_WU_NEUTRAL = 0.428571428571428575  # 3/7 ≈ 0.4286
PAPER_BETA_ABS_NEUTRAL = 0.5
ANCHOR_HIGH = 0.999
ANCHOR_LOW = 0.001


def patched_beta_dicts(
    beta_pres_by_key: Dict[str, np.ndarray],
    beta_abs_by_key: Dict[str, np.ndarray],
    stance_data: Dict[str, Any],
    *,
    reset_beta_abs_neutral: bool,
    reset_beta_pres_wu_neutral: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Build patched β dicts under one counterfactual scenario.

    Reset behaviour: replace per-draw arrays of relevant nodes with
    constant arrays at paper values.  Returns NEW dicts (does not mutate
    the originals).
    """
    bp_new = dict(beta_pres_by_key)
    ba_new = dict(beta_abs_by_key)
    sample_key = next(iter(beta_pres_by_key))
    n_draws = beta_pres_by_key[sample_key].shape[0]

    # Identify all nodes whose label is (weak_undermining, neutral) for β_pres
    # and all nodes with demandingness=neutral for β_abs.  Walk tree.
    wu_neutral_nodes: List[str] = []
    neutral_demand_nodes: List[str] = []
    root_path = (stance_data["name"],)

    def walk(node, path):
        cur = path + (node["name"],)
        ntype = (node.get("type") or "").lower()
        if ntype in {"feature", "subfeature", "indicator"}:
            s = node.get("support", "no bearing")
            d = node.get("demandingness", "neutral")
            key = node_key(path, node["name"])
            if d == "neutral":
                neutral_demand_nodes.append(key)
            if s == "weak undermining" and d == "neutral":
                wu_neutral_nodes.append(key)
        for child in node.get("evidencers", []):
            walk(child, cur)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path)

    if reset_beta_abs_neutral:
        for k in neutral_demand_nodes:
            ba_new[k] = np.full(n_draws, PAPER_BETA_ABS_NEUTRAL)

    if reset_beta_pres_wu_neutral:
        for k in wu_neutral_nodes:
            bp_new[k] = np.full(n_draws, PAPER_BETA_PRES_WU_NEUTRAL)

    return bp_new, ba_new


def affine_coeffs_for_keys(
    stance_data,
    node_to_varname,
    bp_dict,
    ba_dict,
    target_keys: Iterable[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute (intercept_j, slope_j) per posterior draw for the requested
    indicator keys, using the propagation pattern from
    propagate_affine_indicator_coefficients_from_draws.
    """
    target = set(target_keys)
    sample_key = next(iter(bp_dict))
    n_draws = bp_dict[sample_key].shape[0]
    coeffs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    root_path = (stance_data["name"],)

    def walk(node, path, intercept_parent, slope_parent):
        cur = path + (node["name"],)
        key = node_key(path, node["name"])
        bp = bp_dict[key]
        ba = ba_dict[key]
        delta = bp - ba
        intercept_child = ba + intercept_parent * delta
        slope_child = slope_parent * delta
        if (node.get("type") or "").lower() == "indicator":
            if key in target:
                coeffs[key] = (intercept_child, slope_child)
            return
        for child in node.get("evidencers", []):
            walk(child, cur, intercept_child, slope_child)

    root_intercept = np.zeros(n_draws)
    root_slope = np.ones(n_draws)
    for child in stance_data.get("evidencers", []):
        walk(child, root_path, root_intercept, root_slope)
    return coeffs


# ----------------------------------------------------------------------------
# Three-state PPC at focus cell, recomputed from primitive β + posterior obs
# ----------------------------------------------------------------------------


def _ordered_probit_probs(kappa, eta, K):
    """(S, K) category probabilities."""
    cum = norm.cdf(kappa - eta[:, None])
    zeros = np.zeros((cum.shape[0], 1))
    ones = np.ones((cum.shape[0], 1))
    cum_full = np.concatenate([zeros, cum, ones], axis=1)
    probs = np.diff(cum_full, axis=1)
    return np.clip(probs, 1e-12, 1.0)


def focus_cell_ppc(
    bp_dict, ba_dict, stance_data, indicator_keys,
    a_draws, b_draws, kappa_draws, expert_idx, C_value, K=7,
) -> Dict[str, float]:
    """Predicted left/right tail mass at the (E_cross, system) focus cell.

    For each indicator in indicator_keys (those observed by E_cross at the
    focus system), compute q_{s,j}(C, β) from primitive β draws via affine
    propagation. Then for each posterior draw form the three-state mixture
    over m ∈ {0, 1, 2} at the focus rater's emission parameters and
    integrate over m + average across indicators.
    """
    coeffs = affine_coeffs_for_keys(
        stance_data, None, bp_dict, ba_dict, indicator_keys
    )
    S = a_draws.shape[0]
    eta0 = b_draws[:, expert_idx]
    eta1 = eta0 + 0.5 * a_draws
    eta2 = eta0 + a_draws
    pk0 = _ordered_probit_probs(kappa_draws, eta0, K)
    pk1 = _ordered_probit_probs(kappa_draws, eta1, K)
    pk2 = _ordered_probit_probs(kappa_draws, eta2, K)
    # Right tail = P(r=K-1), Left tail = P(r=0)
    pred_right_per_ind = []
    pred_left_per_ind = []
    for key in indicator_keys:
        intercept, slope = coeffs[key]
        q = np.clip(intercept + slope * C_value, 1e-12, 1.0 - 1e-12)
        w0 = (1 - q) ** 2
        w1 = 2 * q * (1 - q)
        w2 = q ** 2
        pr_right = w0 * pk0[:, K - 1] + w1 * pk1[:, K - 1] + w2 * pk2[:, K - 1]
        pr_left = w0 * pk0[:, 0] + w1 * pk1[:, 0] + w2 * pk2[:, 0]
        pred_right_per_ind.append(pr_right)
        pred_left_per_ind.append(pr_left)
    pred_right = np.stack(pred_right_per_ind, axis=0).mean(axis=0)
    pred_left = np.stack(pred_left_per_ind, axis=0).mean(axis=0)
    return {
        "pred_right_med": float(np.median(pred_right)),
        "pred_right_mean": float(pred_right.mean()),
        "pred_left_med": float(np.median(pred_left)),
        "pred_left_mean": float(pred_left.mean()),
    }


def collect_e_cross_indicator_keys(stance_data, proc, system_name: str) -> List[str]:
    """Return indicator node_keys observed by the cross-system rater on the
    given system."""
    cross_real = "Sara DeWitte"  # legacy real name; matches data_cache
    # Use processor's mapping if possible
    expert_idx = proc.expert_to_idx.get(cross_real)
    if expert_idx is None:
        # Fallback: pick the expert with max obs at Human (= cross-system rater)
        from collections import Counter
        ctr: Counter = Counter()
        for sys_name in proc.systems:
            for k, obs in proc.system_observations.get(sys_name, {}).items():
                for e, _ in obs:
                    ctr[e] += 1
        expert_idx = ctr.most_common(1)[0][0]
    keys: List[str] = []
    sys_obs = proc.system_observations.get(system_name, {})
    for key, obs in sys_obs.items():
        if any(e == expert_idx for e, _ in obs):
            keys.append(key)
    return keys, expert_idx


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading pool_3s posterior: {POOL_3S_PATH}")
    idata = az.from_netcdf(str(POOL_3S_PATH))
    post = idata.posterior

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

    a_draws = np.asarray(post["a"].values).reshape(-1)
    K = cfg.N_CATEGORIES
    n_experts = len(proc.expert_names)
    kappa_draws = np.asarray(post["kappa"].values).reshape(-1, K - 1)
    if "b_free" in post.data_vars:
        b_free_all = np.asarray(post["b_free"].values).reshape(-1, n_experts - 1)
        b_draws = np.concatenate([np.zeros((a_draws.shape[0], 1)), b_free_all], axis=1)
    else:
        b_draws = np.zeros((a_draws.shape[0], n_experts))

    bp_dict, ba_dict = pooled_beta_draws_by_node(idata, stance_data)

    # For PPCs, we need E_cross's indicator keys at Human and ELIZA.
    e_cross_human_keys, expert_idx = collect_e_cross_indicator_keys(
        stance_data, proc, "Human"
    )
    e_cross_eliza_keys, _ = collect_e_cross_indicator_keys(
        stance_data, proc, "ELIZA"
    )
    print(f"  E_cross indicators at Human: {len(e_cross_human_keys)}")
    print(f"  E_cross indicators at ELIZA: {len(e_cross_eliza_keys)}")
    print(f"  expert_idx = {expert_idx} (= {proc.expert_names[expert_idx]})")

    # Verify the 5 sign-flip indicators exist in our beta dicts
    for k in SIGN_FLIP_NODE_KEYS:
        if k not in bp_dict:
            raise KeyError(f"Sign-flip indicator key not found: {k}")
    print(f"  5 sign-flip indicators verified in tree")

    # Confirm posterior values for context
    bp_wu = float(np.median(bp_dict[WU_NEUTRAL_NODE_KEY]))
    ba_wu_node = float(np.median(ba_dict[WU_NEUTRAL_NODE_KEY]))
    print(f"  posterior medians at weak-undermining + neutral subfeature:")
    print(f"    β_pres = {bp_wu:.4f}  (paper {PAPER_BETA_PRES_WU_NEUTRAL:.4f})")
    print(f"    β_abs  = {ba_wu_node:.4f}  (paper {PAPER_BETA_ABS_NEUTRAL:.4f})")

    # Run all 4 scenarios
    scenarios = [
        ("a_posterior",      False, False),
        ("b_reset_abs",      True,  False),
        ("c_reset_pres",     False, True),
        ("d_reset_both",     True,  True),
    ]
    rows = []
    for scen_name, reset_abs, reset_pres in scenarios:
        print()
        print(f"=== Scenario {scen_name}: reset_abs={reset_abs}, reset_pres={reset_pres} ===")
        bp_s, ba_s = patched_beta_dicts(
            bp_dict, ba_dict, stance_data,
            reset_beta_abs_neutral=reset_abs,
            reset_beta_pres_wu_neutral=reset_pres,
        )

        # Per sign-flip indicator: δ_j, q_j(0.999), q_j(0.001)
        coeffs = affine_coeffs_for_keys(
            stance_data, None, bp_s, ba_s, SIGN_FLIP_NODE_KEYS
        )
        for key in SIGN_FLIP_NODE_KEYS:
            intercept, slope = coeffs[key]
            ind_name = key.split(" > ")[-1]
            delta_j_med = float(np.median(slope))
            q_high = float(np.median(intercept + ANCHOR_HIGH * slope))
            q_low = float(np.median(intercept + ANCHOR_LOW * slope))
            rows.append({
                "scenario": scen_name,
                "indicator": ind_name,
                "delta_j_med": delta_j_med,
                "q_high_med": q_high,
                "q_low_med": q_low,
            })
            print(f"  {ind_name:<40}  δ_j={delta_j_med:+.4f}  q_high={q_high:.3f}  q_low={q_low:.3f}")

        # Focus-cell PPCs (averaged across all 50 E_cross indicators per system,
        # not just the 5 sign-flip ones — this matches B.5 / B.10's framing)
        print(f"  Focus-cell PPC (across all E_cross indicators):")
        for sys_name, c_val, keys_for_sys in [
            ("Human", ANCHOR_HIGH, e_cross_human_keys),
            ("ELIZA", ANCHOR_LOW, e_cross_eliza_keys),
        ]:
            stats = focus_cell_ppc(
                bp_s, ba_s, stance_data, keys_for_sys,
                a_draws, b_draws, kappa_draws, expert_idx, c_val, K=K,
            )
            tail_key = "pred_right_mean" if sys_name == "Human" else "pred_left_mean"
            tail_value = stats[tail_key]
            obs_value = 0.86 if sys_name == "Human" else 0.96
            print(f"    E_cross × {sys_name}: pred {tail_key}={tail_value:.4f}  observed≈{obs_value}")
            # Append a separate row tagged with "_PPC"
            rows.append({
                "scenario": scen_name,
                "indicator": f"PPC_E_cross_{sys_name}_{tail_key}",
                "delta_j_med": float("nan"),
                "q_high_med": float("nan"),
                "q_low_med": float("nan"),
                "ppc_value": tail_value,
                "obs_value": obs_value,
            })

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "beta_abs_neutral_counterfactual.csv"
    df.to_csv(out_csv, index=False)
    print()
    print(f"wrote {out_csv}")

    # Decomposition table for δ_j (per indicator: scenarios a vs b vs c vs d)
    print()
    print("=== δ_j decomposition (per sign-flip indicator) ===")
    pivot = df.dropna(subset=["delta_j_med"]).pivot(
        index="indicator", columns="scenario", values="delta_j_med"
    )
    pivot["shared_abs_effect"] = pivot["a_posterior"] - pivot["b_reset_abs"]
    pivot["own_pres_effect"] = pivot["a_posterior"] - pivot["c_reset_pres"]
    pivot["both_effect"] = pivot["a_posterior"] - pivot["d_reset_both"]
    print(pivot.round(4).to_string())

    pivot.round(4).to_csv(OUT_DIR / "beta_abs_neutral_counterfactual_decomp.csv")
    print(f"wrote {OUT_DIR / 'beta_abs_neutral_counterfactual_decomp.csv'}")

    print()
    print(">>> Interpretation:")
    print("    If shared_abs_effect ≫ own_pres_effect for all 5 sign-flip indicators,")
    print("    the 'sign-flip resolution' under pool_3s is mostly inheritance through")
    print("    the shared β_abs__neutral parameter, not direct evidence on weak-undermining.")
    sa = pivot["shared_abs_effect"].abs().mean()
    op = pivot["own_pres_effect"].abs().mean()
    print(f"    mean |shared_abs_effect|: {sa:+.4f}")
    print(f"    mean |own_pres_effect|:   {op:+.4f}")
    if sa > 3 * op:
        print("    >>> CONFIRMED: shared-β_abs effect dominates by >3×")
    elif sa > op:
        print("    >>> shared effect larger but own-pres is non-negligible")
    else:
        print("    >>> own-pres effect comparable or larger -- B.3a interpretation needs revising")


if __name__ == "__main__":
    main()
