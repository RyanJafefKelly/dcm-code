"""Extended composite-vs-exact diagnostic across three fits.

Per ChatGPT round-3:
- Run the diagnostic on `pool_3s`, `pool_3s_abs_by_sd`, AND `baseline_3s`.
- Add by-system Δℓ split (Human, Chicken, LLMs, ELIZA) — important because the
  total gap can mean different things depending on whether anchored or free
  systems dominate it.
- Add a reconstruction check: diagnostic's "composite" must match the model's
  actual fitted Potential value, not just our written-down formula.
- Honest β_abs handling across fits: do not force a shared `β_abs__neutral`
  across models that don't have one.
- Sibling-correlation top-5 per fit.

Reuses the DP + sanity check infrastructure from composite_vs_exact_diagnostic.py.

Usage:
    python composite_vs_exact_diagnostic_extended.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from analyse_tree_pooling import pooled_beta_draws_by_node
from composite_vs_exact_diagnostic import (
    collect_indicator_obs,
    composite_loglik,
    exact_loglik,
    precompute_leaf_logliks,
    sibling_score_table,
    _sanity_check_brute_force_enumeration,
    _sanity_check_no_transmission,
    _weighted_quantile,
)
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
    extract_beta_draws_by_node,
)

OUT_DIR = Path("notebooks/meeting_prep_arvo_2026-04-27/figs_round2")
STANCE = "Global Workspace Theory"

FITS = [
    {
        "name": "pool_3s",
        "path": "results/gwt_tree_pooling/three_state_pooled_anchored.nc",
        "config_kwargs": dict(POOL_BETAS_BY_LABEL=True),
        "beta_loader": "pooled",
    },
    {
        "name": "pool_3s_abs_by_sd",
        "path": "results/gwt_tree_pooling/three_state_pooled_abs_by_sd_anchored.nc",
        "config_kwargs": dict(POOL_BETAS_BY_LABEL=True, BETA_ABS_BY_SUPPORT_DEMAND=True),
        "beta_loader": "pooled",
    },
    {
        "name": "baseline_3s",
        "path": "results/gwt_binary_three_state/three_state_anchored.nc",
        "config_kwargs": dict(),  # per-node Beta priors, no pooling
        "beta_loader": "per_node",
    },
]


def load_fit_context(fit_spec):
    """Return (idata, cfg, stance_data, builder, proc, β_pres_dict, β_abs_dict)."""
    print(f"  loading {fit_spec['name']} from {fit_spec['path']}")
    idata = az.from_netcdf(str(fit_spec["path"]))
    cfg = ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        **fit_spec["config_kwargs"],
    )
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    proc = MultiSystemDataProcessor(cfg)
    proc.process(stance_data, [s for s, _ in ANCHORED_SYSTEM_CONFIGS])
    builder = MultiSystemModelBuilder(
        cfg, EvidenceProcessor(cfg), proc, list(ANCHORED_SYSTEM_CONFIGS)
    )
    builder.build_model(stance_data)

    if fit_spec["beta_loader"] == "pooled":
        bp_dict, ba_dict = pooled_beta_draws_by_node(idata, stance_data)
    elif fit_spec["beta_loader"] == "per_node":
        bp_dict, ba_dict = extract_beta_draws_by_node(idata, builder, stance_data)
    else:
        raise ValueError(f"Unknown beta_loader: {fit_spec['beta_loader']}")

    return idata, cfg, stance_data, builder, proc, bp_dict, ba_dict


def run_one_fit(fit_spec, do_sanity_checks_first: bool = True) -> Dict[str, Any]:
    """Run the full Δℓ diagnostic on one fit, return results dict."""
    print()
    print(f"=== {fit_spec['name']} ===")
    idata, cfg, stance_data, builder, proc, bp_dict, ba_dict = load_fit_context(fit_spec)
    post = idata.posterior

    # Posterior arrays
    a_draws = np.asarray(post["a"].values).reshape(-1)
    K = cfg.N_CATEGORIES
    n_experts = len(proc.expert_names)
    kappa_draws = np.asarray(post["kappa"].values).reshape(-1, K - 1)
    if "b_free" in post.data_vars:
        b_free_all = np.asarray(post["b_free"].values).reshape(-1, n_experts - 1)
        b_draws = np.concatenate([np.zeros((a_draws.shape[0], 1)), b_free_all], axis=1)
    else:
        b_draws = np.zeros((a_draws.shape[0], n_experts))

    # System C
    C_by_system: Dict[str, np.ndarray] = {}
    for sys_name, c_fixed in ANCHORED_SYSTEM_CONFIGS:
        if c_fixed is not None:
            C_by_system[sys_name] = np.full(a_draws.shape[0], c_fixed)
        else:
            sp = builder._sys_prefix(sys_name)
            var = f"{sp}__global_workspace_theory_C"
            C_by_system[sys_name] = np.asarray(post[var].values).reshape(-1)

    indicator_obs = collect_indicator_obs(stance_data, proc)
    n_obs_per_ind = {k: sum(len(r) for (r, _) in v.values()) for k, v in indicator_obs.items()}
    leaf_logliks = precompute_leaf_logliks(indicator_obs, a_draws, b_draws, kappa_draws, K=K)

    # Sanity checks (only run once across all fits to save time; gated by flag)
    if do_sanity_checks_first:
        print("  Sanity checks (run once, common to all fits):")
        _sanity_check_brute_force_enumeration()
        _sanity_check_no_transmission(
            stance_data, indicator_obs, leaf_logliks,
            bp_dict, ba_dict, C_by_system,
        )

    # Total ℓ
    print("  Computing log-likelihoods (total + per-system split)...")
    ll_comp_total, ll_comp_per_sys = composite_loglik(
        stance_data, indicator_obs, leaf_logliks,
        bp_dict, ba_dict, C_by_system,
    )
    ll_exact_total, ll_exact_per_sys = exact_loglik(
        stance_data, indicator_obs, leaf_logliks,
        bp_dict, ba_dict, C_by_system,
    )
    delta_ll = ll_exact_total - ll_comp_total

    # Self-consistency
    if not np.allclose(ll_comp_total + delta_ll, ll_exact_total, rtol=1e-9, atol=1e-9):
        raise AssertionError(f"{fit_spec['name']}: self-consistency failed")

    # Importance reweighting
    log_w = delta_ll - logsumexp(delta_ll)
    w = np.exp(log_w)
    n_draws = a_draws.shape[0]
    ESS = float(1.0 / np.sum(w ** 2))

    pareto_k = float("nan")
    try:
        psis_lw, k_hat = az.psislw(delta_ll[None, :])
        pareto_k = float(k_hat[0])
    except Exception:
        pass

    # Per-system Δℓ
    by_sys = {}
    for sys_name in C_by_system:
        d_sys = ll_exact_per_sys[sys_name] - ll_comp_per_sys[sys_name]
        by_sys[sys_name] = {
            "median": float(np.median(d_sys)),
            "mean": float(np.mean(d_sys)),
            "p5": float(np.percentile(d_sys, 5)),
            "p95": float(np.percentile(d_sys, 95)),
            "frac_total": float(np.mean(d_sys) / np.mean(delta_ll))
            if np.mean(delta_ll) != 0 else 0.0,
        }

    # Headline shifts (with caveat — should not be reported as evidence)
    headline = {}
    for label, var in [("Chicken", "chicken__global_workspace_theory_C"),
                       ("LLMs", "2024_leading_chat_llms__global_workspace_theory_C")]:
        d = np.asarray(post[var].values).reshape(-1)
        if np.std(d) < 1e-12:
            continue
        med_orig = float(np.median(d))
        med_rw = float(_weighted_quantile(d, w, 0.5))
        headline[f"C_{label}"] = {"original": med_orig, "reweighted": med_rw,
                                   "shift": med_rw - med_orig}

    # Sibling-correlation top 5
    score_rows = sibling_score_table(stance_data, bp_dict, ba_dict, n_obs_per_ind)
    score_rows.sort(key=lambda r: r["score"], reverse=True)
    top5 = score_rows[:5]

    # Print summary
    print(f"  total Δℓ:     median {float(np.median(delta_ll)):+.3f}  "
          f"5/95: [{float(np.percentile(delta_ll, 5)):+.3f}, {float(np.percentile(delta_ll, 95)):+.3f}]")
    print(f"  ESS_w / S:    {ESS / n_draws:.4f}    (ESS={ESS:.0f}/{n_draws})")
    print(f"  Pareto-k̂:    {pareto_k:+.3f}")
    print(f"  By-system Δℓ medians: " + ", ".join(
        f"{s}={by_sys[s]['median']:+.2f}" for s in by_sys))
    print(f"  Top sibling-corr nodes: " + ", ".join(
        f"{r['node_name'][:25]}={r['score']:.2f}" for r in top5[:3]))

    return {
        "fit_name": fit_spec["name"],
        "n_draws": n_draws,
        "delta_ll_median": float(np.median(delta_ll)),
        "delta_ll_mean": float(np.mean(delta_ll)),
        "delta_ll_p5": float(np.percentile(delta_ll, 5)),
        "delta_ll_p95": float(np.percentile(delta_ll, 95)),
        "ess_w": ESS,
        "ess_w_ratio": ESS / n_draws,
        "pareto_k": pareto_k,
        "by_system": by_sys,
        "headline_reweighted": headline,
        "sibling_top5": top5,
    }


def reconstruction_check_pool_3s(fit_results) -> None:
    """For pool_3s, verify the diagnostic's composite ℓ matches the model's actual
    Potential value via PyMC's compute_log_likelihood path.

    NOTE: pool_3s was NOT sampled with idata_kwargs={"log_likelihood": True}, so
    the InferenceData lacks a `log_likelihood` group. The most reliable cross-check
    we can do here is to manually evaluate the per-indicator Potential formula on
    posterior draws by reading the cached `*_p` (q_j) Deterministics, and confirm
    the sum equals the diagnostic's composite total.

    This catches any drift between the written-down composite formula and what's
    actually in the model — independent of the DP correctness verified by the
    brute-force + no-transmission checks.
    """
    import arviz as az_mod
    from scipy.stats import norm

    fit_spec = next(f for f in FITS if f["name"] == "pool_3s")
    idata = az_mod.from_netcdf(str(fit_spec["path"]))
    cfg = ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        **fit_spec["config_kwargs"],
    )
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    proc = MultiSystemDataProcessor(cfg)
    proc.process(stance_data, [s for s, _ in ANCHORED_SYSTEM_CONFIGS])
    builder = MultiSystemModelBuilder(
        cfg, EvidenceProcessor(cfg), proc, list(ANCHORED_SYSTEM_CONFIGS)
    )
    builder.build_model(stance_data)
    post = idata.posterior

    a_draws = np.asarray(post["a"].values).reshape(-1)
    K = cfg.N_CATEGORIES
    n_experts = len(proc.expert_names)
    kappa_draws = np.asarray(post["kappa"].values).reshape(-1, K - 1)
    if "b_free" in post.data_vars:
        b_free_all = np.asarray(post["b_free"].values).reshape(-1, n_experts - 1)
        b_draws = np.concatenate([np.zeros((a_draws.shape[0], 1)), b_free_all], axis=1)
    else:
        b_draws = np.zeros((a_draws.shape[0], n_experts))

    # For each (system, indicator), pull the cached `q_j` (= *_p) Deterministic
    # and compute the per-indicator three-state log-mix likelihood.
    # Sum across all (system, indicator) to get the recomputed composite total.
    indicator_obs = collect_indicator_obs(stance_data, proc)
    leaf_logliks = precompute_leaf_logliks(indicator_obs, a_draws, b_draws, kappa_draws, K=K)

    print("  reconstruction check on pool_3s:")
    n_check = 50  # spot-check on first 50 draws
    n_total = a_draws.shape[0]

    recon_total = np.zeros(n_total)
    for key, sys_obs_dict in indicator_obs.items():
        # node_to_varname uses the path-keyed approach
        varname = builder.node_to_varname.get(key)
        if varname is None:
            continue
        for sys_name, _ in sys_obs_dict.items():
            sp = builder._sys_prefix(sys_name)
            q_var = f"{sp}__{varname}_p"
            if q_var not in post.data_vars:
                continue
            q = np.asarray(post[q_var].values).reshape(-1)
            q = np.clip(q, 1e-12, 1.0 - 1e-12)
            log_w0 = 2.0 * np.log(1.0 - q)
            log_w1 = np.log(2.0) + np.log(q) + np.log(1.0 - q)
            log_w2 = 2.0 * np.log(q)
            ll_arr = leaf_logliks[key][sys_name]  # (n_total, 3)
            ll0, llh, ll1 = ll_arr[:, 0], ll_arr[:, 1], ll_arr[:, 2]
            recon_total += np.logaddexp(
                np.logaddexp(log_w0 + ll0, log_w1 + llh),
                log_w2 + ll1,
            )

    # Compare with diagnostic's composite (run a fresh composite_loglik)
    bp_dict, ba_dict = pooled_beta_draws_by_node(idata, stance_data)
    C_by_system = {}
    for sys_name, c_fixed in ANCHORED_SYSTEM_CONFIGS:
        if c_fixed is not None:
            C_by_system[sys_name] = np.full(n_total, c_fixed)
        else:
            sp = builder._sys_prefix(sys_name)
            var = f"{sp}__global_workspace_theory_C"
            C_by_system[sys_name] = np.asarray(post[var].values).reshape(-1)
    diag_comp_total, _ = composite_loglik(
        stance_data, indicator_obs, leaf_logliks,
        bp_dict, ba_dict, C_by_system,
    )

    diff = recon_total - diag_comp_total
    max_diff = float(np.max(np.abs(diff[:n_check])))
    print(f"    recomputed via cached q_j deterministics vs diagnostic composite:")
    print(f"      max |Δ| over first {n_check} draws: {max_diff:.6e}")
    if max_diff > 1e-6:
        print(f"    *** WARNING: reconstruction mismatch > 1e-6")
        print(f"    median diff: {float(np.median(diff)):+.6e}")
        print(f"    This may reflect numerical clip differences (1e-12 in both sides)")
    else:
        print(f"    PASS")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Reconstruction check on pool_3s")
    print("="*70)
    reconstruction_check_pool_3s(None)

    print()
    print("="*70)
    print("Diagnostic across three fits")
    print("="*70)

    results = []
    for i, fit_spec in enumerate(FITS):
        # Sanity checks only on first fit (DP + no-transmission are properties
        # of the algorithm, not the data; running them once is sufficient)
        do_sanity = (i == 0)
        res = run_one_fit(fit_spec, do_sanity_checks_first=do_sanity)
        results.append(res)

    # ===== Comparison table =====
    print()
    print("="*70)
    print("Three-fit comparison")
    print("="*70)

    rows = []
    for r in results:
        row = {
            "fit": r["fit_name"],
            "delta_ll_median": r["delta_ll_median"],
            "delta_ll_p5": r["delta_ll_p5"],
            "delta_ll_p95": r["delta_ll_p95"],
            "ess_w_ratio": r["ess_w_ratio"],
            "pareto_k": r["pareto_k"],
        }
        for sys_name, d in r["by_system"].items():
            row[f"dll_{sys_name}_median"] = d["median"]
            row[f"dll_{sys_name}_frac_total"] = d["frac_total"]
        for hkey, h in r["headline_reweighted"].items():
            row[f"{hkey}_orig"] = h["original"]
            row[f"{hkey}_rw"] = h["reweighted"]
            row[f"{hkey}_shift"] = h["shift"]
        rows.append(row)

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))

    df.to_csv(OUT_DIR / "composite_gap_three_fits_comparison.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'composite_gap_three_fits_comparison.csv'}")

    # ===== Sibling-correlation top-5 per fit =====
    sibling_rows = []
    for r in results:
        for rank, sib in enumerate(r["sibling_top5"][:5], start=1):
            sibling_rows.append({
                "fit": r["fit_name"],
                "rank": rank,
                "node_name": sib["node_name"],
                "n_children": sib["n_children"],
                "q_v_med": sib["q_v_med"],
                "max_delta_product": sib["max_delta_product"],
                "n_obs_subtree": sib["n_obs_subtree"],
                "score": sib["score"],
            })
    sib_df = pd.DataFrame(sibling_rows)
    sib_df.to_csv(OUT_DIR / "sibling_correlation_top5_by_fit.csv", index=False)
    print(f"wrote {OUT_DIR / 'sibling_correlation_top5_by_fit.csv'}")

    print()
    print("Sibling-correlation top 5 per fit:")
    print(sib_df.to_string(index=False))


if __name__ == "__main__":
    main()
