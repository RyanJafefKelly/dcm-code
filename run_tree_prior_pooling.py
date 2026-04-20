"""Fit driver for the `POOL_BETAS_BY_LABEL` intervention (Stage 5, Fit 1).

Joint anchored GWT, binary leaf, with hierarchical label-level hypermeans on
``beta_pres`` (grouped by support x demandingness) and ``beta_abs`` (grouped by
demandingness only).  Non-centred parameterisation internally; see
``dcm_model.build_label_pool_hyperparameters``.

Writes, under ``results/gwt_tree_pooling/``:
- ``binary_pooled_anchored.nc`` (local, gitignored)
- ``binary_pooled_anchored.meta.json`` (git branch/commit, full ModelConfig,
  system configs, wall-clock, output paths)
- ``analysis/tier1_diagnostics.md`` (divergences, R-hat, ESS, label-level
  deterministic summary, first-pass Tier-1 verdict)

Runtime: ~90-120 min wall-clock on an M2 Air.
"""

from __future__ import annotations

import json
import subprocess
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

def _paths(state_model: str):
    tag = "binary" if state_model == "binary" else "three_state"
    return (
        OUT_DIR / f"{tag}_pooled_anchored.nc",
        OUT_DIR / f"{tag}_pooled_anchored.meta.json",
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
    state_model: str = "binary",
    num_samples: int = 2000,
    num_tune: int = 1000,
    num_chains: int = 4,
    target_accept: float = 0.95,
    label_pool_sigma: float = 0.5,
) -> ModelConfig:
    return ModelConfig(
        INDICATOR_STATE_MODEL=state_model,  # type: ignore[arg-type]
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=True,
        LABEL_POOL_SIGMA=label_pool_sigma,
        NUM_SAMPLES=num_samples,
        NUM_TUNE=num_tune,
        NUM_CHAINS=num_chains,
        TARGET_ACCEPT=target_accept,
    )


def tier1_diagnostics(idata: Any) -> Dict[str, Any]:
    """Compute the Tier 1 falsification signals on the posterior."""
    post = idata.posterior

    var_names = ["a", "kappa"]
    for v in post.data_vars:
        if v.endswith("_C"):
            draws = np.asarray(post[v].values).reshape(-1)
            if np.std(draws) > 1e-10:
                var_names.append(v)
        if v.startswith("mu_pres__") or v.startswith("mu_abs__"):
            var_names.append(v)

    diag = az.summary(idata, var_names=var_names, kind="diagnostics")
    div_counts = int(idata.sample_stats["diverging"].values.sum())
    return {
        "divergences": div_counts,
        "max_rhat": float(diag["r_hat"].max()),
        "min_ess_bulk": float(diag["ess_bulk"].min()),
        "min_ess_tail": float(diag["ess_tail"].min()),
        "n_draws": int(idata.posterior.sizes["draw"] * idata.posterior.sizes["chain"]),
    }


def label_posterior_summary(idata: Any) -> Dict[str, Dict[str, float]]:
    """Per-group label-level posterior summaries."""
    post = idata.posterior
    out: Dict[str, Dict[str, float]] = {}
    for v in post.data_vars:
        if v.startswith("mu_pres__") or v.startswith("mu_abs__") or v.startswith("label_delta__"):
            draws = np.asarray(post[v].values).reshape(-1)
            out[str(v)] = {
                "median": float(np.median(draws)),
                "lo": float(np.percentile(draws, 3)),
                "hi": float(np.percentile(draws, 97)),
            }
    return out


def write_tier1_markdown(
    diag: Dict[str, Any],
    label_summary: Dict[str, Dict[str, float]],
    elapsed_s: float,
    config: ModelConfig,
    path: Path,
) -> None:
    lines = [
        f"# Tier-1 diagnostics: binary + POOL_BETAS_BY_LABEL",
        f"",
        f"Generated in {elapsed_s:.0f}s.  LABEL_POOL_SIGMA = {config.LABEL_POOL_SIGMA}, "
        f"TARGET_ACCEPT = {config.TARGET_ACCEPT}, "
        f"NUM_TUNE = {config.NUM_TUNE}, NUM_SAMPLES = {config.NUM_SAMPLES}, "
        f"NUM_CHAINS = {config.NUM_CHAINS}.",
        f"",
        f"## Sampling",
        f"",
        f"| metric | value | target |",
        f"|---|---:|---:|",
        f"| divergences | {diag['divergences']} | 0 ideal; rerun if >50 |",
        f"| max R-hat   | {diag['max_rhat']:.4f} | < 1.01 |",
        f"| min ESS bulk | {diag['min_ess_bulk']:.0f} | > 400 |",
        f"| min ESS tail | {diag['min_ess_tail']:.0f} | > 400 |",
        f"| total draws | {diag['n_draws']} | (across chains) |",
        f"",
        f"## Label-level posteriors (context)",
        f"",
        f"| parameter | median | 3% | 97% |",
        f"|---|---:|---:|---:|",
    ]
    for name in sorted(label_summary):
        s = label_summary[name]
        lines.append(
            f"| {name} | {s['median']:+.3f} | {s['lo']:+.3f} | {s['hi']:+.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main(smoke: bool = False, state_model: str = "binary") -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    idata_path, meta_path = _paths(state_model)

    if smoke:
        config = build_config(state_model=state_model, num_samples=50,
                              num_tune=100, num_chains=2, target_accept=0.9)
        print(f"[SMOKE] reduced sampling for build-check only; state_model={state_model}")
    else:
        config = build_config(state_model=state_model)

    meta_git = git_head()
    print(f"Branch: {meta_git['branch']}")
    print(f"Commit: {meta_git['commit']}")
    print(f"Config: INDICATOR_STATE_MODEL={state_model}, POOL_BETAS_BY_LABEL={config.POOL_BETAS_BY_LABEL}, "
          f"sigma={config.LABEL_POOL_SIGMA}, target_accept={config.TARGET_ACCEPT}, "
          f"tune={config.NUM_TUNE}, samples={config.NUM_SAMPLES}")

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
    label_summary = label_posterior_summary(idata)

    if not smoke:
        tier1_md = ANALYSIS_DIR / f"tier1_diagnostics_{state_model}.md"
        write_tier1_markdown(diag, label_summary, elapsed, config, tier1_md)
        meta = {
            "fit": f"{state_model}_pooled_anchored",
            "branch": meta_git["branch"],
            "commit": meta_git["commit"],
            "system_configs": [(s, c) for s, c in SYSTEM_CONFIGS],
            "config": asdict(config),
            "elapsed_s": elapsed,
            "tier1": diag,
            "idata_path": str(idata_path),
            "tier1_markdown": str(tier1_md),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"wrote {meta_path}")

    print()
    print(f"elapsed: {elapsed:.0f}s")
    print(f"divergences: {diag['divergences']}")
    print(f"max R-hat:   {diag['max_rhat']:.4f}")
    print(f"min ESS bulk: {diag['min_ess_bulk']:.0f}")
    print(f"label parameters: {len(label_summary)}")


if __name__ == "__main__":
    import sys
    state_model = "three_state" if "--three-state" in sys.argv else "binary"
    main(smoke="--smoke" in sys.argv, state_model=state_model)
