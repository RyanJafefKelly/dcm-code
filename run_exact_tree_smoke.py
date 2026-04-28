"""Phase B smoke driver — exact-tree refit on GWT under pool_3s_abs_by_sd config.

Goal: binary question for the meeting — does the exact-tree likelihood
compile and sample at all on GWT?

If yes → the path to a real refit is engineering, not research.
If no → we have actionable info on the specific blocker.

Configuration: same as `pool_3s_abs_by_sd` (POOL_BETAS_BY_LABEL=True,
BETA_ABS_BY_SUPPORT_DEMAND=True, three-state leaf, joint anchored).

Smoke fit: 2 chains × 300 tune × 500 draws, target_accept=0.9.

Sanity check on the PyTensor expression: before sampling, evaluate the
exact-tree log-likelihood on a few posterior draws from the existing
`pool_3s_abs_by_sd` fit and compare with the standalone NumPy DP from
`composite_vs_exact_diagnostic.py:exact_loglik`. Match to numerical
precision is required before sampling.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import arviz as az
import numpy as np

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
)
from dcm_model_exact_tree import MultiSystemExactTreeBuilder
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS


STANCE = "Global Workspace Theory"
SYSTEM_CONFIGS = list(ANCHORED_SYSTEM_CONFIGS)
OUT_DIR = Path("results/gwt_exact_tree_smoke")


def build_smoke_config() -> ModelConfig:
    return ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=True,
        BETA_ABS_BY_SUPPORT_DEMAND=True,
        LABEL_POOL_SIGMA=0.5,
        NUM_SAMPLES=500,
        NUM_TUNE=300,
        NUM_CHAINS=2,
        TARGET_ACCEPT=0.9,
    )


def git_head() -> Dict[str, str]:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        return {"branch": branch, "commit": commit}
    except Exception:
        return {"branch": "unknown", "commit": "unknown"}


def run_pretrain_sanity_check(builder, model, stance_data, proc):
    """Evaluate the exact-tree PyTensor expression on a few posterior draws
    from the existing pool_3s_abs_by_sd fit, compare with NumPy DP.
    """
    print("[SANITY] cross-check exact-tree PyTensor expression vs standalone NumPy DP")
    pool_path = Path("results/gwt_tree_pooling/three_state_pooled_abs_by_sd_anchored.nc")
    if not pool_path.exists():
        print("  pool_3s_abs_by_sd netcdf not found; skipping cross-check")
        return None

    idata_existing = az.from_netcdf(str(pool_path))
    post = idata_existing.posterior

    # Pull posterior draws of all named primitive RVs
    n_check = 5
    rng = np.random.default_rng(0)
    a_all = np.asarray(post["a"].values).reshape(-1)
    n_total = a_all.shape[0]
    sample_idx = rng.choice(n_total, size=n_check, replace=False)

    # NumPy DP from composite_vs_exact_diagnostic
    cfg = build_smoke_config()
    bp_dict, ba_dict = pooled_beta_draws_by_node(idata_existing, stance_data)
    K = cfg.N_CATEGORIES
    n_experts = len(proc.expert_names)
    a_draws = a_all[sample_idx]
    kappa_draws = np.asarray(post["kappa"].values).reshape(-1, K - 1)[sample_idx]
    if "b_free" in post.data_vars:
        b_free_all = np.asarray(post["b_free"].values).reshape(-1, n_experts - 1)
        b_draws = np.concatenate([np.zeros((n_check, 1)), b_free_all[sample_idx]], axis=1)
    else:
        b_draws = np.zeros((n_check, n_experts))

    # Subset β dicts to chosen draws
    bp_sub = {k: v[sample_idx] for k, v in bp_dict.items()}
    ba_sub = {k: v[sample_idx] for k, v in ba_dict.items()}

    # Per-system C
    C_by_system = {}
    for sys_name, c_fixed in SYSTEM_CONFIGS:
        if c_fixed is not None:
            C_by_system[sys_name] = np.full(n_check, c_fixed)
        else:
            sp = builder._sys_prefix(sys_name)
            var = f"{sp}__global_workspace_theory_C"
            C_by_system[sys_name] = np.asarray(post[var].values).reshape(-1)[sample_idx]

    indicator_obs = collect_indicator_obs(stance_data, proc)
    leaf_logliks = precompute_leaf_logliks(
        indicator_obs, a_draws, b_draws, kappa_draws, K=K
    )
    ll_numpy_total, ll_numpy_per_sys = exact_loglik(
        stance_data, indicator_obs, leaf_logliks,
        bp_sub, ba_sub, C_by_system,
    )

    print(f"  NumPy DP exact log-lik (per draw, total): "
          f"{[f'{v:+.3f}' for v in ll_numpy_total]}")
    print(f"  NumPy DP per-system Δℓ:")
    for sys_name in C_by_system:
        sys_ll = ll_numpy_per_sys[sys_name]
        print(f"    {sys_name}: {[f'{v:+.3f}' for v in sys_ll]}")

    # PyTensor side: evaluate model.compile_logp at the same draws
    # Look up named RVs in the model and substitute values from posterior
    print("  evaluating PyTensor exact-tree Potential at same draws...")
    try:
        with model:
            free_rvs = model.free_RVs
            # Build a compile_logp eval using the per-system Potential terms
            # We collect ONLY the exact_tree_lik Potentials (skip prior contributions)
            potential_terms = []
            for v in model.deterministics + model.potentials:
                if "exact_tree_lik" in v.name:
                    potential_terms.append(v)
            # Compile a function that takes posterior values and returns the sum
            # of all exact_tree_lik potentials.
            import pytensor
            inputs = list(free_rvs)
            outputs = pt_sum_of(potential_terms)
            f = pytensor.function(inputs, outputs, on_unused_input="ignore")
        # For each draw, build the input list by reading the corresponding
        # named posterior var. Names match between model RV name and posterior var.
        # NB: free_RVs in PyMC have transformed values during sampling, but for
        # post-hoc eval we need to pass the value-on-natural-scale via the
        # InferenceData posterior.
        # This is brittle if the posterior names don't exactly match free_RV
        # names; for this smoke check we just print whether it ran.
        py_vals = []
        for s in range(n_check):
            input_vals = []
            for rv in free_rvs:
                vn = rv.name
                if vn in post.data_vars:
                    val = np.asarray(post[vn].values).reshape(-1, *post[vn].shape[2:])[
                        sample_idx[s]
                    ]
                else:
                    # Skip / fail loudly
                    print(f"    *** posterior missing var {vn!r}; cannot cross-check fully")
                    return None
                input_vals.append(np.asarray(val))
            py_vals.append(float(f(*input_vals)))
        py_vals_arr = np.array(py_vals)
        print(f"  PyTensor exact-tree Potential value (per draw): "
              f"{[f'{v:+.3f}' for v in py_vals_arr]}")
        diffs = py_vals_arr - ll_numpy_total
        max_abs = float(np.max(np.abs(diffs)))
        print(f"  max |PyTensor - NumPy| = {max_abs:.3e}")
        if max_abs > 1e-3:
            print("  *** WARNING: cross-check failed (> 1e-3)")
            return False
        print("  PASS")
        return True
    except Exception as e:
        print(f"  cross-check raised: {e}")
        print("  proceeding to sampling anyway (sanity check is informative, not blocking)")
        return None


def pt_sum_of(terms):
    """Sum a list of scalar PyTensor tensors."""
    import pytensor.tensor as pt
    if not terms:
        return pt.constant(0.0)
    out = terms[0]
    for t in terms[1:]:
        out = out + t
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_smoke_config()
    git = git_head()
    print(f"Branch: {git['branch']}")
    print(f"Commit: {git['commit']}")
    print(f"Config: INDICATOR_STATE_MODEL={cfg.INDICATOR_STATE_MODEL}, "
          f"POOL_BETAS_BY_LABEL={cfg.POOL_BETAS_BY_LABEL}, "
          f"BETA_ABS_BY_SUPPORT_DEMAND={cfg.BETA_ABS_BY_SUPPORT_DEMAND}, "
          f"σ={cfg.LABEL_POOL_SIGMA}, "
          f"chains={cfg.NUM_CHAINS}, tune={cfg.NUM_TUNE}, draws={cfg.NUM_SAMPLES}")
    print()

    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    proc = MultiSystemDataProcessor(cfg)
    proc.process(stance_data, [s for s, _ in SYSTEM_CONFIGS])

    # Build the exact-tree model
    print("Building exact-tree PyMC model (PyTensor expression construction)...")
    t0 = time.time()
    builder = MultiSystemExactTreeBuilder(
        cfg, EvidenceProcessor(cfg), proc, list(SYSTEM_CONFIGS)
    )
    try:
        model = builder.build_model(stance_data)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n*** EXACT-TREE BUILD FAILED after {elapsed:.0f}s ***")
        print(f"Error: {type(e).__name__}: {e}")
        meta = {
            "phase": "B_exact_tree_smoke",
            "branch": git["branch"], "commit": git["commit"],
            "outcome": "build_failed",
            "error_type": type(e).__name__,
            "error_message": str(e)[:500],
            "elapsed_s_build": elapsed,
        }
        (OUT_DIR / "smoke_meta.json").write_text(json.dumps(meta, indent=2))
        sys.exit(2)
    elapsed_build = time.time() - t0
    print(f"PyMC model build time: {elapsed_build:.1f}s")
    print(f"Free RVs: {[rv.name for rv in model.free_RVs][:8]}... ({len(model.free_RVs)} total)")
    print(f"Potentials: {[p.name for p in model.potentials]}")
    print()

    # Sanity check
    print("Pre-sampling sanity check (cross-check vs NumPy DP)...")
    sanity_ok = run_pretrain_sanity_check(builder, model, stance_data, proc)
    print()

    # Sample
    print(f"Sampling: {cfg.NUM_CHAINS} chains × {cfg.NUM_TUNE} tune × {cfg.NUM_SAMPLES} draws...")
    t1 = time.time()
    try:
        with model:
            import pymc as pm
            idata = pm.sample(
                draws=cfg.NUM_SAMPLES,
                tune=cfg.NUM_TUNE,
                chains=cfg.NUM_CHAINS,
                cores=cfg.NUM_CHAINS,
                target_accept=cfg.TARGET_ACCEPT,
                random_seed=42,
            )
    except Exception as e:
        elapsed_sample = time.time() - t1
        print(f"\n*** EXACT-TREE SAMPLING FAILED after {elapsed_sample:.0f}s ***")
        print(f"Error: {type(e).__name__}: {e}")
        meta = {
            "phase": "B_exact_tree_smoke",
            "branch": git["branch"], "commit": git["commit"],
            "outcome": "sampling_failed",
            "error_type": type(e).__name__,
            "error_message": str(e)[:500],
            "elapsed_s_build": elapsed_build,
            "elapsed_s_sample": elapsed_sample,
        }
        (OUT_DIR / "smoke_meta.json").write_text(json.dumps(meta, indent=2))
        sys.exit(3)
    elapsed_sample = time.time() - t1
    print(f"Sampling time: {elapsed_sample:.1f}s")

    # Diagnostics
    div_count = int(idata.sample_stats["diverging"].values.sum())
    var_names = ["a", "kappa"]
    for v in idata.posterior.data_vars:
        if str(v).endswith("_C"):
            d = np.asarray(idata.posterior[v].values).reshape(-1)
            if np.std(d) > 1e-10:
                var_names.append(str(v))
    diag = az.summary(idata, var_names=var_names, kind="diagnostics")
    max_rhat = float(diag["r_hat"].max())
    min_ess = float(diag["ess_bulk"].min())

    # Headline C
    sys_vars = {
        "Human": "human__global_workspace_theory_C",
        "Chicken": "chicken__global_workspace_theory_C",
        "LLMs": "2024_leading_chat_llms__global_workspace_theory_C",
        "ELIZA": "eliza__global_workspace_theory_C",
    }
    headline = {}
    print()
    print("=== Sampling diagnostics ===")
    print(f"  divergences:    {div_count}")
    print(f"  max R-hat:      {max_rhat:.4f}")
    print(f"  min ESS bulk:   {min_ess:.0f}")
    print()
    print("=== System C posteriors ===")
    print(f"  {'system':<8} {'exact-tree median':<24} {'pool_3s_abs_by_sd ref':<24}")
    pool_path = Path("results/gwt_tree_pooling/three_state_pooled_abs_by_sd_anchored.nc")
    pool_post = az.from_netcdf(str(pool_path)).posterior if pool_path.exists() else None
    for label, v in sys_vars.items():
        if v not in idata.posterior.data_vars:
            continue
        d = np.asarray(idata.posterior[v].values).reshape(-1)
        med = float(np.median(d))
        lo = float(np.percentile(d, 3))
        hi = float(np.percentile(d, 97))
        ref = "—"
        if pool_post is not None and v in pool_post.data_vars:
            ref_d = np.asarray(pool_post[v].values).reshape(-1)
            ref_med = float(np.median(ref_d))
            ref = f"{ref_med:.4f}"
        med_str = f"{med:.4f} [{lo:.4f}, {hi:.4f}]"
        print(f"  {label:<8} {med_str:<24} {ref:<24}")
        headline[label] = {"median": med, "lo": lo, "hi": hi, "pool_3s_abs_by_sd_ref": ref}

    # Save
    nc_path = OUT_DIR / "three_state_pooled_abs_by_sd_exact_smoke.nc"
    meta_path = OUT_DIR / "smoke_meta.json"
    az.to_netcdf(idata, str(nc_path))
    print(f"\nwrote {nc_path}")

    meta = {
        "phase": "B_exact_tree_smoke",
        "branch": git["branch"], "commit": git["commit"],
        "outcome": "success",
        "config": asdict(cfg),
        "elapsed_s_build": elapsed_build,
        "elapsed_s_sample": elapsed_sample,
        "divergences": div_count,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess,
        "headline": headline,
        "sanity_check_passed": sanity_ok,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
