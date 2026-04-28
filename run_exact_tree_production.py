"""Production exact-tree fit on GWT under pool_3s_abs_by_sd config.

After the smoke (Phase B) compiled cleanly in 6.9s and sampled at 8.5 min for
2 chains × 800 iter, the production fit at 4 chains × 3000 iter should
finish in ~35-40 min wall-clock and give convergence-quality samples for
all parameters (the smoke had a "max R-hat > 1.01 for some β_tilde
parameters" warning, expected to resolve with longer chains).

Configuration: identical to the smoke (POOL_BETAS_BY_LABEL=True,
BETA_ABS_BY_SUPPORT_DEMAND=True, three-state leaf, joint anchored,
σ=0.5), but: 4 chains × 1000 tune × 2000 draws, target_accept=0.95.

Saved to results/gwt_exact_tree/three_state_pooled_abs_by_sd_exact_anchored.nc.

Sanity check (cross-check against NumPy DP) is reused from smoke and
re-confirmed before sampling.
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

from dcm_model import (
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    load_data,
)
from dcm_model_exact_tree import MultiSystemExactTreeBuilder
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS
from run_exact_tree_smoke import run_pretrain_sanity_check


STANCE = "Global Workspace Theory"
SYSTEM_CONFIGS = list(ANCHORED_SYSTEM_CONFIGS)
OUT_DIR = Path("results/gwt_exact_tree")


def build_production_config() -> ModelConfig:
    return ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=True,
        BETA_ABS_BY_SUPPORT_DEMAND=True,
        LABEL_POOL_SIGMA=0.5,
        NUM_SAMPLES=2000,
        NUM_TUNE=1000,
        NUM_CHAINS=4,
        TARGET_ACCEPT=0.95,
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_production_config()
    git = git_head()
    print(f"Branch: {git['branch']}")
    print(f"Commit: {git['commit']}")
    print(f"Config: INDICATOR_STATE_MODEL={cfg.INDICATOR_STATE_MODEL}, "
          f"POOL_BETAS_BY_LABEL={cfg.POOL_BETAS_BY_LABEL}, "
          f"BETA_ABS_BY_SUPPORT_DEMAND={cfg.BETA_ABS_BY_SUPPORT_DEMAND}, "
          f"σ={cfg.LABEL_POOL_SIGMA}, "
          f"chains={cfg.NUM_CHAINS}, tune={cfg.NUM_TUNE}, draws={cfg.NUM_SAMPLES}, "
          f"target_accept={cfg.TARGET_ACCEPT}")
    print()

    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    proc = MultiSystemDataProcessor(cfg)
    proc.process(stance_data, [s for s, _ in SYSTEM_CONFIGS])

    print("Building exact-tree PyMC model...")
    t0 = time.time()
    builder = MultiSystemExactTreeBuilder(
        cfg, EvidenceProcessor(cfg), proc, list(SYSTEM_CONFIGS)
    )
    model = builder.build_model(stance_data)
    elapsed_build = time.time() - t0
    print(f"PyMC model build time: {elapsed_build:.1f}s")
    print(f"Free RVs: {len(model.free_RVs)} total")
    print(f"Potentials: {[p.name for p in model.potentials]}")
    print()

    print("Pre-sampling sanity check (cross-check vs NumPy DP)...")
    sanity_ok = run_pretrain_sanity_check(builder, model, stance_data, proc)
    print()

    print(f"Sampling: {cfg.NUM_CHAINS} chains × {cfg.NUM_TUNE} tune × {cfg.NUM_SAMPLES} draws...")
    t1 = time.time()
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
    elapsed_sample = time.time() - t1
    print(f"Sampling time: {elapsed_sample:.1f}s ({elapsed_sample/60:.1f} min)")

    # Diagnostics
    div_count = int(idata.sample_stats["diverging"].values.sum())
    var_names = ["a", "kappa"]
    for v in idata.posterior.data_vars:
        if str(v).endswith("_C"):
            d = np.asarray(idata.posterior[v].values).reshape(-1)
            if np.std(d) > 1e-10:
                var_names.append(str(v))
        if str(v).startswith("beta_pres__") or str(v).startswith("beta_abs__") \
                or str(v).startswith("label_delta__"):
            var_names.append(str(v))
    diag = az.summary(idata, var_names=var_names, kind="diagnostics")
    max_rhat = float(diag["r_hat"].max())
    min_ess = float(diag["ess_bulk"].min())

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
    for label, v in sys_vars.items():
        d = np.asarray(idata.posterior[v].values).reshape(-1)
        med = float(np.median(d))
        lo = float(np.percentile(d, 3))
        hi = float(np.percentile(d, 97))
        print(f"  {label:<8}  {med:.4f} [{lo:.4f}, {hi:.4f}]")
        headline[label] = {"median": med, "lo": lo, "hi": hi}

    nc_path = OUT_DIR / "three_state_pooled_abs_by_sd_exact_anchored.nc"
    meta_path = OUT_DIR / "three_state_pooled_abs_by_sd_exact_anchored.meta.json"
    az.to_netcdf(idata, str(nc_path))
    print(f"\nwrote {nc_path}")

    meta = {
        "phase": "exact_tree_production",
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
