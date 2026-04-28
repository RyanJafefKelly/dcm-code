"""Fit driver for `POOL_BETAS_BY_LABEL` + `BETA_ABS_BY_SUPPORT_DEMAND`.

Diagnostic refit for round-2 of the Arvo meeting prep: tests whether the
shared-β_abs__neutral inheritance flagged in B.3a/b of notebook 18 is
driving the apparent `weak undermining + neutral` sign-flip resolution
under the current `pool_3s` fit.

Same logit-Normal(σ=0.5) prior family and same paper β_abs prior centre
as `pool_3s`; only the *grouping* of β_abs changes from demandingness-only
to (support, demandingness).  Joint anchored GWT, three-state leaf.

Saved to `results/gwt_tree_pooling/three_state_pooled_abs_by_sd_anchored.nc`.

Runtime: ~20 min wall-clock on an M2 Air (matches `pool_3s`).

Usage:
    python run_tree_prior_pooling_abs_by_sd.py            # full fit
    python run_tree_prior_pooling_abs_by_sd.py --smoke    # build-check only
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
    MultiSystemModelBuilder,
    fit_stance_multisystem,
    load_data,
)


STANCE = "Global Workspace Theory"
SYSTEM_CONFIGS = [
    ("Human", 0.999),
    ("Chicken", None),
    ("2024 Leading Chat LLMs", None),
    ("ELIZA", 0.001),
]

OUT_DIR = Path("results/gwt_tree_pooling")
ANALYSIS_DIR = OUT_DIR / "analysis"


def _paths():
    return (
        OUT_DIR / "three_state_pooled_abs_by_sd_anchored.nc",
        OUT_DIR / "three_state_pooled_abs_by_sd_anchored.meta.json",
    )


def git_head() -> Dict[str, str]:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        branch, commit = "unknown", "unknown"
    return {"branch": branch, "commit": commit}


def build_config(
    num_samples: int = 2000,
    num_tune: int = 1000,
    num_chains: int = 4,
    target_accept: float = 0.95,
    label_pool_sigma: float = 0.5,
) -> ModelConfig:
    return ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=True,
        BETA_ABS_BY_SUPPORT_DEMAND=True,  # the diagnostic
        LABEL_POOL_SIGMA=label_pool_sigma,
        NUM_SAMPLES=num_samples,
        NUM_TUNE=num_tune,
        NUM_CHAINS=num_chains,
        TARGET_ACCEPT=target_accept,
    )


def tier1_diagnostics(idata: Any) -> Dict[str, Any]:
    post = idata.posterior

    var_names = ["a", "kappa"]
    for v in post.data_vars:
        if v.endswith("_C"):
            draws = np.asarray(post[v].values).reshape(-1)
            if np.std(draws) > 1e-10:
                var_names.append(v)
        if v.startswith("beta_pres__") or v.startswith("beta_abs__"):
            var_names.append(v)

    diag = az.summary(idata, var_names=var_names, kind="diagnostics")
    div_counts = int(idata.sample_stats["diverging"].values.sum())

    # Tightened gate on weak_undermining + neutral cluster
    wu_keys = [
        "beta_pres__weak_undermining__neutral",
        "beta_abs__weak_undermining__neutral",
        "label_delta__weak_undermining__neutral",
    ]
    wu_diag = az.summary(
        idata,
        var_names=[v for v in wu_keys if v in post.data_vars],
        kind="diagnostics",
    )
    wu_max_rhat = float(wu_diag["r_hat"].max()) if len(wu_diag) else float("nan")
    wu_min_ess = float(wu_diag["ess_bulk"].min()) if len(wu_diag) else float("nan")

    return {
        "divergences": div_counts,
        "max_rhat": float(diag["r_hat"].max()),
        "min_ess_bulk": float(diag["ess_bulk"].min()),
        "min_ess_tail": float(diag["ess_tail"].min()),
        "wu_max_rhat": wu_max_rhat,
        "wu_min_ess_bulk": wu_min_ess,
        "tightened_gate_pass": (
            div_counts == 0 and wu_max_rhat < 1.005
        ),
        "n_draws": int(idata.posterior.sizes["draw"] * idata.posterior.sizes["chain"]),
    }


def main(smoke: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    idata_path, meta_path = _paths()

    if smoke:
        config = build_config(num_samples=50, num_tune=100, num_chains=2,
                              target_accept=0.9)
        print("[SMOKE] reduced sampling for build-check only")
    else:
        config = build_config()

    meta_git = git_head()
    print(f"Branch: {meta_git['branch']}")
    print(f"Commit: {meta_git['commit']}")
    print(
        "Config: INDICATOR_STATE_MODEL=three_state, POOL_BETAS_BY_LABEL=True, "
        f"BETA_ABS_BY_SUPPORT_DEMAND=True, sigma={config.LABEL_POOL_SIGMA}, "
        f"target_accept={config.TARGET_ACCEPT}, "
        f"tune={config.NUM_TUNE}, samples={config.NUM_SAMPLES}"
    )

    stance_data = next(s for s in load_data(config) if s["name"] == STANCE)
    t0 = time.time()
    idata, builder, processor = fit_stance_multisystem(
        stance_data, config, SYSTEM_CONFIGS
    )
    elapsed = time.time() - t0

    if not smoke:
        az.to_netcdf(idata, str(idata_path))
        print(f"wrote {idata_path}")

    diag = tier1_diagnostics(idata)

    if not smoke:
        meta = {
            "fit": "three_state_pooled_abs_by_sd_anchored",
            "branch": meta_git["branch"],
            "commit": meta_git["commit"],
            "system_configs": [(s, c) for s, c in SYSTEM_CONFIGS],
            "config": asdict(config),
            "elapsed_s": elapsed,
            "tier1": diag,
            "idata_path": str(idata_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"wrote {meta_path}")

    print()
    print(f"elapsed: {elapsed:.0f}s")
    print(f"divergences: {diag['divergences']}")
    print(f"max R-hat: {diag['max_rhat']:.4f}")
    print(f"min ESS bulk: {diag['min_ess_bulk']:.0f}")
    print(f"weak_undermining cluster max R-hat: {diag['wu_max_rhat']:.4f}")
    print(f"weak_undermining cluster min ESS bulk: {diag['wu_min_ess_bulk']:.0f}")
    print(f"tightened gate pass: {diag['tightened_gate_pass']}")


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
