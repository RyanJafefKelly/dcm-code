"""Post-process the production exact-tree fit.

Computes everything needed for meeting-note + diagnostics-writeup updates:
- Headline C posteriors with intervals
- Mean δ_j via affine propagation from primitive β draws
- Focus-cell PPC closure (manual, since exact-tree builder skips *_pz1)
- Sign uncertainty + practical-equivalence on weak-undermining cluster
- All comparison numbers vs baseline_3s, pool_3s, pool_3s_abs_by_sd, exact-smoke

Output: appends results to diagnostics_round2_results.md and meeting note.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import norm

from analyse_tree_pooling import (
    per_indicator_delta_under_pooling,
    pooled_beta_draws_by_node,
)
from dcm_model import (
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    load_data,
)
from dcm_model_exact_tree import MultiSystemExactTreeBuilder
from gwt_reference_recovery_analysis import (
    ANCHORED_SYSTEM_CONFIGS,
    propagate_affine_indicator_coefficients_from_draws,
    build_indicator_index,
)
from tree_propagation_diagnostic import _EXPERT_ANON_MAP

EXACT_PATH = Path("results/gwt_exact_tree/three_state_pooled_abs_by_sd_exact_anchored.nc")
META_PATH = Path("results/gwt_exact_tree/three_state_pooled_abs_by_sd_exact_anchored.meta.json")
OUT_DIR = Path("notebooks/meeting_prep_arvo_2026-04-27/figs_round2")
WRITEUP_PATH = Path("notebooks/meeting_prep_arvo_2026-04-27/diagnostics_round2_results.md")
MEETING_NOTE_PATH = Path("notebooks/meeting_prep_arvo_2026-04-27/meeting_note_arvo_2026-04-27.md")
STANCE = "Global Workspace Theory"

REAL_BY_ANON = {v: k for k, v in dict(_EXPERT_ANON_MAP).items()}

FOCUS_CELLS = [
    ("E_cross", "Human", "right"),
    ("E_cross", "ELIZA", "left"),
    ("E_chickenB", "Chicken", "right"),
    ("E_llmC", "2024 Leading Chat LLMs", "right"),
]

SIGN_FLIP_NODE_KEYS = [
    "Global Workspace Theory > Coherence > Autonomous Subparts > Information Transfer",
    "Global Workspace Theory > Coherence > Autonomous Subparts > Learning Transfer",
    "Global Workspace Theory > Coherence > Autonomous Subparts > Functional Subparts",
    "Global Workspace Theory > Coherence > Autonomous Subparts > Information Transfer Architecture",
    "Global Workspace Theory > Coherence > Autonomous Subparts > Conflicting Subparts",
]


def _ordered_probit_probs(kappa: np.ndarray, eta: np.ndarray, K: int) -> np.ndarray:
    """(S, K) category probabilities."""
    cum = norm.cdf(kappa - eta[:, None])
    zeros = np.zeros((cum.shape[0], 1))
    ones = np.ones((cum.shape[0], 1))
    cum_full = np.concatenate([zeros, cum, ones], axis=1)
    probs = np.diff(cum_full, axis=1)
    return np.clip(probs, 1e-12, 1.0)


def manual_leaf_updated_ppc(
    idata, cfg, stance_data, proc, builder, anchor_C: Dict[str, float],
    n_draws: int = 500, seed: int = 42,
) -> Dict[tuple, Dict[str, Any]]:
    """Compute leaf-updated focus-cell PPC for the exact-tree fit.

    The exact-tree builder doesn't add *_pz1 deterministics, so we
    reconstruct ρ_m manually from posterior β draws + observed ratings.
    For each posterior draw and each (system, indicator) with data, we
    compute the per-indicator latent-state posterior weights ρ_m, then
    use them to compute the leaf-updated predicted category distribution
    at each (rater, system) cell.

    Returns dict keyed by (expert_real_name, system_name) → stats dict.
    """
    post = idata.posterior
    K = cfg.N_CATEGORIES
    n_experts = len(proc.expert_names)

    # Sample subset of draws
    a_all = np.asarray(post["a"].values).reshape(-1)
    S_total = a_all.shape[0]
    rng = np.random.default_rng(seed)
    if n_draws < S_total:
        idx = rng.choice(S_total, size=n_draws, replace=False)
    else:
        idx = np.arange(S_total)
    S = idx.shape[0]

    a_draws = a_all[idx]
    kappa_draws = np.asarray(post["kappa"].values).reshape(-1, K - 1)[idx]
    if "b_free" in post.data_vars:
        b_free_all = np.asarray(post["b_free"].values).reshape(-1, n_experts - 1)
        b_draws = np.concatenate([np.zeros((S, 1)), b_free_all[idx]], axis=1)
    else:
        b_draws = np.zeros((S, n_experts))

    # β draws keyed by node_key (subset to chosen draws)
    bp_dict_full, ba_dict_full = pooled_beta_draws_by_node(idata, stance_data)
    bp_dict = {k: v[idx] for k, v in bp_dict_full.items()}
    ba_dict = {k: v[idx] for k, v in ba_dict_full.items()}

    # Per-indicator affine coefficients (intercept, slope) per draw
    indicators = build_indicator_index(stance_data, builder)
    intercepts, slopes, _ = propagate_affine_indicator_coefficients_from_draws(
        stance_data, builder.node_to_varname, bp_dict, ba_dict, indicator_index=indicators,
    )
    # intercepts, slopes shape: (S, n_indicators)
    indicator_keys = [spec.node_key for spec in indicators]
    key_to_j = {k: j for j, k in enumerate(indicator_keys)}

    # For each (system, indicator) with data and each rater, compute the
    # cell's predicted category distribution:
    #   ρ_m = π_m(q) · L_m(y_j) / B(q)
    #   pred_cat[k] = Σ_m ρ_m · P_OP(k | η_m)  (across the cell's indicators)
    K_arr = np.arange(K)
    # Per-rater emission components (η ∈ {0, a/2, a})
    eta0_per_e = b_draws  # (S, E)
    eta1_per_e = b_draws + 0.5 * a_draws[:, None]
    eta2_per_e = b_draws + a_draws[:, None]

    # Precompute per-rater P_OP for each component: list (E,) of (S, K) arrays
    pk0 = []
    pk1 = []
    pk2 = []
    for e in range(n_experts):
        pk0.append(_ordered_probit_probs(kappa_draws, eta0_per_e[:, e], K))
        pk1.append(_ordered_probit_probs(kappa_draws, eta1_per_e[:, e], K))
        pk2.append(_ordered_probit_probs(kappa_draws, eta2_per_e[:, e], K))

    # Bucket observations by (expert_idx, system_name) -> list of (rating, q_per_draw)
    from collections import defaultdict
    buckets = defaultdict(list)
    for sys_name, sys_obs in proc.system_observations.items():
        C_s = anchor_C.get(sys_name)
        if C_s is None:
            # Free system: pull posterior C at chosen draws
            sp = builder._sys_prefix(sys_name)
            var = f"{sp}__global_workspace_theory_C"
            C_s = np.asarray(post[var].values).reshape(-1)[idx]
        else:
            C_s = np.full(S, C_s)
        for nkey, obs_list in sys_obs.items():
            j = key_to_j.get(nkey)
            if j is None:
                continue
            q_arr = intercepts[:, j] + slopes[:, j] * C_s  # (S,)
            q_arr = np.clip(q_arr, 1e-12, 1.0 - 1e-12)
            for expert_idx, rating in obs_list:
                buckets[(expert_idx, sys_name)].append((int(rating), q_arr))

    results = {}
    for (e, sys_name), entries in buckets.items():
        if not entries:
            continue
        N = len(entries)
        ratings = np.array([r for r, _ in entries], dtype=int)
        q_stack = np.stack([q for _, q in entries], axis=0)  # (N, S)

        # ρ_m per (indicator, draw): π_m(q) · L_m(y_obs) / B(q)
        # L_m for the rater's specific rating
        # P_OP at rater e for each component, per draw, per category
        pk0_e = pk0[e]  # (S, K)
        pk1_e = pk1[e]
        pk2_e = pk2[e]
        # Indexed by observed rating y_obs per indicator
        L0 = pk0_e[:, ratings].T  # (N, S) — actually careful with shape
        L1 = pk1_e[:, ratings].T
        L2 = pk2_e[:, ratings].T
        # π_m(q) per indicator/draw
        w0 = (1 - q_stack) ** 2
        w1 = 2 * q_stack * (1 - q_stack)
        w2 = q_stack ** 2
        unnorm0 = w0 * L0
        unnorm1 = w1 * L1
        unnorm2 = w2 * L2
        Z = unnorm0 + unnorm1 + unnorm2
        Z = np.clip(Z, 1e-300, None)
        rho0 = unnorm0 / Z
        rho1 = unnorm1 / Z
        rho2 = unnorm2 / Z
        # Predicted category mass per draw averaged across indicators
        # mix[s, k] = (1/N) Σ_n [ρ0_n,s · pk0[s,k] + ρ1 · pk1[s,k] + ρ2 · pk2[s,k]]
        # Hmm — but pk0[s, k] is per-rater per-state, doesn't depend on n.
        # So we can factor: mix = (mean_n ρ0)·pk0 + ...
        rho0_mean = rho0.mean(axis=0)  # (S,)
        rho1_mean = rho1.mean(axis=0)
        rho2_mean = rho2.mean(axis=0)
        # pred[s, k] = ρ0_mean[s] · pk0[s, k] + ρ1_mean[s] · pk1[s, k] + ρ2_mean[s] · pk2[s, k]
        pred = (rho0_mean[:, None] * pk0_e + rho1_mean[:, None] * pk1_e
                + rho2_mean[:, None] * pk2_e)  # (S, K)
        # Sample replicated ratings from this mixture (one per (n, s))
        # for histogram intervals — but for left/right tail we just need pred[:, 0] / pred[:, K-1]
        pred_left = pred[:, 0]
        pred_right = pred[:, K - 1]
        obs_left = float(np.mean(ratings == 0))
        obs_right = float(np.mean(ratings == K - 1))

        results[(proc.expert_names[e], sys_name)] = {
            "n_obs": N,
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


def main():
    if not EXACT_PATH.exists():
        print(f"FAIL: production exact-tree fit not found at {EXACT_PATH}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading exact-tree production fit: {EXACT_PATH}")
    idata = az.from_netcdf(str(EXACT_PATH))
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}

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

    print(f"Tier 1: divergences={meta['tier1']['divergences'] if 'tier1' in meta else meta.get('divergences')}, "
          f"max R-hat={meta.get('max_rhat'):.4f}, min ESS bulk={meta.get('min_ess_bulk'):.0f}")

    # ---- Headline C ----
    sys_vars = {
        "Human": "human__global_workspace_theory_C",
        "Chicken": "chicken__global_workspace_theory_C",
        "LLMs": "2024_leading_chat_llms__global_workspace_theory_C",
        "ELIZA": "eliza__global_workspace_theory_C",
    }
    headline_rows = []
    print()
    print("=== Headline C posteriors ===")
    print(f"  {'system':<8}  {'exact-tree (production)':<28}  pool_3s_abs_by_sd  pool_3s")
    pool_abs_post = az.from_netcdf("results/gwt_tree_pooling/three_state_pooled_abs_by_sd_anchored.nc").posterior
    pool_post = az.from_netcdf("results/gwt_tree_pooling/three_state_pooled_anchored.nc").posterior
    for label, v in sys_vars.items():
        d = np.asarray(idata.posterior[v].values).reshape(-1)
        med = float(np.median(d))
        lo = float(np.percentile(d, 3))
        hi = float(np.percentile(d, 97))
        ref_abs = float(np.median(np.asarray(pool_abs_post[v].values).reshape(-1)))
        ref_pool = float(np.median(np.asarray(pool_post[v].values).reshape(-1)))
        print(f"  {label:<8}  {med:.4f} [{lo:.4f}, {hi:.4f}]  {ref_abs:.4f}             {ref_pool:.4f}")
        headline_rows.append({"system": label, "exact_med": med, "exact_lo": lo, "exact_hi": hi,
                              "pool_3s_abs_by_sd_med": ref_abs, "pool_3s_med": ref_pool,
                              "shift_vs_pool_3s": med - ref_pool})

    # ---- Mean δ_j ----
    print()
    print("=== Mean δ_j ===")
    intercepts, slopes, _ = per_indicator_delta_under_pooling(idata, builder, stance_data)
    delta_med_per_ind = np.median(slopes, axis=0)
    mean_dj = float(delta_med_per_ind.mean())
    print(f"  mean δ_j (exact-tree): {mean_dj:.4f}")
    print(f"  mean δ_j (pool_3s_abs_by_sd ref): 0.098")
    print(f"  mean δ_j (pool_3s ref):           0.155")

    # ---- Focus-cell PPC closure ----
    print()
    print("=== Focus-cell PPC closure (leaf-updated, manual) ===")
    anchor_C = {"Human": 0.999, "ELIZA": 0.001}
    ppc_stats = manual_leaf_updated_ppc(
        idata, cfg, stance_data, proc, builder, anchor_C,
        n_draws=500, seed=42,
    )
    ppc_rows = []
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
        ppc_rows.append({"fit": "exact_tree_production", "cell": cell, "tail": tail,
                         "obs": obs, "pred_mean": pred_m, "pred_lo": pred_lo,
                         "pred_hi": pred_hi, "delta": d})

    # Closure vs baseline_3s
    base_lookup = {}
    for r in pd.read_csv(OUT_DIR / "pool_3s_abs_by_sd_ppc_focus_guardrail.csv").to_dict("records"):
        if r["fit"] == "baseline_3s":
            base_lookup[(r["cell"], r["tail"])] = r
    print()
    print("=== Closure (vs baseline_3s) ===")
    for r in ppc_rows:
        base = base_lookup.get((r["cell"], r["tail"]))
        if base is None:
            continue
        base_abs = abs(base["delta"])
        new_abs = abs(r["delta"])
        closure = (base_abs - new_abs) / max(base_abs, 1e-9) * 100
        print(f"  {r['cell']:<35} {r['tail']:<6}  closure = {closure:+.1f}%")

    # ---- Sign uncertainty on cluster ----
    print()
    print("=== Cluster sign uncertainty ===")
    keys = []
    j_indices = []
    indicator_index = build_indicator_index(stance_data, builder)
    for j, spec in enumerate(indicator_index):
        if spec.node_key in SIGN_FLIP_NODE_KEYS:
            keys.append(spec.node_key)
            j_indices.append(j)
    cluster_slopes = slopes[:, j_indices]  # (S, 5)
    cluster_mean = cluster_slopes.mean(axis=1)
    cluster_med = float(np.median(cluster_mean))
    cluster_lo = float(np.percentile(cluster_mean, 3))
    cluster_hi = float(np.percentile(cluster_mean, 97))
    Pr_pos = float(np.mean(cluster_mean > 0))
    Pr_all_pos = float(np.mean((cluster_slopes > 0).all(axis=1)))
    Pr_lt_001 = float(np.mean(np.abs(cluster_mean) < 0.01))
    Pr_lt_002 = float(np.mean(np.abs(cluster_mean) < 0.02))
    print(f"  cluster mean δ̄: {cluster_med:+.4f} [{cluster_lo:+.4f}, {cluster_hi:+.4f}]")
    print(f"    Pr(δ̄ > 0)        = {Pr_pos:.3f}")
    print(f"    Pr(all 5 δ_j > 0) = {Pr_all_pos:.3f}")
    print(f"    Pr(|δ̄| < 0.01)   = {Pr_lt_001:.3f}")
    print(f"    Pr(|δ̄| < 0.02)   = {Pr_lt_002:.3f}")

    # ---- Save tables ----
    pd.DataFrame(headline_rows).to_csv(OUT_DIR / "exact_tree_production_headline.csv", index=False)
    pd.DataFrame(ppc_rows).to_csv(OUT_DIR / "exact_tree_production_ppc.csv", index=False)
    sign_dict = {
        "fit": "exact_tree_production",
        "cluster_med": cluster_med, "cluster_lo": cluster_lo, "cluster_hi": cluster_hi,
        "Pr_pos": Pr_pos, "Pr_all_pos": Pr_all_pos,
        "Pr_lt_001": Pr_lt_001, "Pr_lt_002": Pr_lt_002,
        "mean_delta_j": mean_dj,
    }
    (OUT_DIR / "exact_tree_production_sign.json").write_text(json.dumps(sign_dict, indent=2))
    print(f"\nwrote {OUT_DIR / 'exact_tree_production_headline.csv'}")
    print(f"wrote {OUT_DIR / 'exact_tree_production_ppc.csv'}")
    print(f"wrote {OUT_DIR / 'exact_tree_production_sign.json'}")

    return {
        "headline_rows": headline_rows,
        "mean_dj": mean_dj,
        "ppc_rows": ppc_rows,
        "sign": sign_dict,
        "tier1_divergences": meta.get("divergences"),
        "tier1_max_rhat": meta.get("max_rhat"),
        "tier1_min_ess": meta.get("min_ess_bulk"),
        "elapsed_s_sample": meta.get("elapsed_s_sample"),
    }


if __name__ == "__main__":
    main()
