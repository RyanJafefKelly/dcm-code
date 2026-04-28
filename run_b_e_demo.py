"""b_e demonstration fit (sensitivity, NOT a baseline).

For the meeting note's §2: shows what posterior `b_e` looks like under
a narrow prior (EXPERT_SHIFT_SIGMA=0.5), with everything else held at
the baseline (three-state leaf, paper tree, hard anchors, joint
multi-system fit). Used to demonstrate the b_e branch is implemented
and well-behaved, without re-anchoring the meeting around it.

The "real" baseline for the meeting is `baseline_3s` (no b_e); this
fit is a demonstration only.
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
    fit_stance_multisystem,
    load_data,
)
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS

STANCE = "Global Workspace Theory"
OUT_DIR = Path("results/gwt_b_e_demo")


def build_config() -> ModelConfig:
    return ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=True,
        EXPERT_SHIFT_SIGMA=0.5,  # narrow prior — the deliberate choice for this demo
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        USE_EXPERT_SCALES=False,
        POOL_BETAS_BY_LABEL=False,
        NUM_SAMPLES=2000,
        NUM_TUNE=1000,
        NUM_CHAINS=4,
        TARGET_ACCEPT=0.95,
    )


def git_head() -> Dict[str, str]:
    try:
        return {
            "branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        }
    except Exception:
        return {"branch": "unknown", "commit": "unknown"}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = build_config()
    git = git_head()
    print(f"Branch: {git['branch']}")
    print(f"Commit: {git['commit']}")
    print(f"Config: USE_EXPERT_SHIFTS=True, EXPERT_SHIFT_SIGMA={cfg.EXPERT_SHIFT_SIGMA}, "
          f"three_state, paper tree, hard anchors")
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)

    t0 = time.time()
    idata, builder, processor = fit_stance_multisystem(
        stance_data, cfg, list(ANCHORED_SYSTEM_CONFIGS)
    )
    elapsed = time.time() - t0
    print(f"\nelapsed: {elapsed:.0f}s")

    nc_path = OUT_DIR / "three_state_b_e_narrow_anchored.nc"
    meta_path = OUT_DIR / "three_state_b_e_narrow_anchored.meta.json"
    az.to_netcdf(idata, str(nc_path))
    print(f"wrote {nc_path}")

    # Diagnostics
    n_div = int(idata.sample_stats["diverging"].values.sum())
    var_names = ["a", "kappa"]
    for v in idata.posterior.data_vars:
        if str(v).endswith("_C"):
            d = np.asarray(idata.posterior[v].values).reshape(-1)
            if np.std(d) > 1e-10:
                var_names.append(str(v))
    if "b_free" in idata.posterior.data_vars:
        var_names.append("b_free")
    diag = az.summary(idata, var_names=var_names, kind="diagnostics")
    max_rhat = float(diag["r_hat"].max())
    min_ess = float(diag["ess_bulk"].min())

    headline = {}
    sys_vars = {
        "Human": "human__global_workspace_theory_C",
        "Chicken": "chicken__global_workspace_theory_C",
        "LLMs": "2024_leading_chat_llms__global_workspace_theory_C",
        "ELIZA": "eliza__global_workspace_theory_C",
    }
    for label, v in sys_vars.items():
        if v in idata.posterior.data_vars:
            d = np.asarray(idata.posterior[v].values).reshape(-1)
            headline[label] = {
                "median": float(np.median(d)),
                "lo": float(np.percentile(d, 3)),
                "hi": float(np.percentile(d, 97)),
            }

    # b_e summary
    b_e_summary = []
    if "b_free" in idata.posterior.data_vars:
        bf = np.asarray(idata.posterior["b_free"].values).reshape(-1, idata.posterior["b_free"].shape[-1])
        for i in range(bf.shape[1]):
            d = bf[:, i]
            b_e_summary.append({
                "free_idx": i,
                "median": float(np.median(d)),
                "lo": float(np.percentile(d, 3)),
                "hi": float(np.percentile(d, 97)),
            })
    print()
    print("=== Diagnostics ===")
    print(f"  divergences:    {n_div}")
    print(f"  max R-hat:      {max_rhat:.4f}")
    print(f"  min ESS bulk:   {min_ess:.0f}")
    print()
    print("=== Headline C posteriors ===")
    for label, h in headline.items():
        print(f"  {label:<8}  {h['median']:.4f} [{h['lo']:.4f}, {h['hi']:.4f}]")
    print()
    print(f"=== b_e (free experts; first expert is the anchor at 0) ===")
    for be in b_e_summary:
        print(f"  free_idx={be['free_idx']}  median {be['median']:+.3f}  94% [{be['lo']:+.3f}, {be['hi']:+.3f}]")

    meta = {
        "phase": "b_e_demonstration_NOT_a_baseline",
        "branch": git["branch"], "commit": git["commit"],
        "config": asdict(cfg),
        "elapsed_s": elapsed,
        "divergences": n_div,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess,
        "headline": headline,
        "b_e_summary": b_e_summary,
        "expert_names": processor.expert_names,
        "anchor_expert": processor.anchor_expert,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {meta_path}")


if __name__ == "__main__":
    main()
