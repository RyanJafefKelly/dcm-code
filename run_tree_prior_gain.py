"""Fit driver for the `TRANSMISSION_GAIN` intervention (Stage 5, Fit 2 comparator).

Joint anchored GWT, binary leaf, with symmetric log-odds-gap gain applied
deterministically inside ``EvidenceProcessor.get_beta_parameters``. Default
``gain = 2.5`` chosen from label semantics (strong support + neutral reaches
prior-mean transmission ~0.80).  Does not add PyMC random variables on top of
the paper baseline.

Writes, under ``results/gwt_transmission_gain/``:
- ``binary_gain_g{g}_anchored.nc`` (local, gitignored)
- ``binary_gain_g{g}_anchored.meta.json`` (git branch/commit, full ModelConfig,
  system configs, wall-clock, output paths)
- ``analysis/tier1_diagnostics.md``

Runtime: ~75-90 min wall-clock on an M2 Air (shorter than pooling; no
hierarchical hypermeans).
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
    ModelConfig,
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

OUT_DIR = Path("results/gwt_transmission_gain")
ANALYSIS_DIR = OUT_DIR / "analysis"

DEFAULT_GAIN = 2.5


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
    gain: float = DEFAULT_GAIN,
    state_model: str = "binary",
    safe: bool = False,
    num_samples: int = 2000,
    num_tune: int = 500,
    num_chains: int = 4,
    target_accept: float = 0.95,
) -> ModelConfig:
    return ModelConfig(
        INDICATOR_STATE_MODEL=state_model,  # type: ignore[arg-type]
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        TRANSMISSION_GAIN=gain,
        GAIN_LOGIT_NORMAL=safe,
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
    # Include tree betas for general convergence check
    var_names.extend(sorted(
        v for v in post.data_vars
        if v.endswith("_beta_pres") or v.endswith("_beta_abs")
    )[:20])  # cap to first 20 to keep az.summary fast
    diag = az.summary(idata, var_names=var_names, kind="diagnostics")
    div_counts = int(idata.sample_stats["diverging"].values.sum())
    return {
        "divergences": div_counts,
        "max_rhat": float(diag["r_hat"].max()),
        "min_ess_bulk": float(diag["ess_bulk"].min()),
        "min_ess_tail": float(diag["ess_tail"].min()),
        "n_draws": int(idata.posterior.sizes["draw"] * idata.posterior.sizes["chain"]),
    }


def write_tier1_markdown(
    diag: Dict[str, Any],
    elapsed_s: float,
    config: ModelConfig,
    path: Path,
) -> None:
    lines = [
        f"# Tier-1 diagnostics: binary + TRANSMISSION_GAIN = {config.TRANSMISSION_GAIN}",
        f"",
        f"Generated in {elapsed_s:.0f}s.  "
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
        "",
    ]
    path.write_text("\n".join(lines))


def main(
    smoke: bool = False,
    gain: float = DEFAULT_GAIN,
    state_model: str = "binary",
    safe: bool = False,
    num_tune: int = 500,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if smoke:
        config = build_config(gain=gain, state_model=state_model, safe=safe,
                              num_samples=50, num_tune=100,
                              num_chains=2, target_accept=0.9)
        print(f"[SMOKE] gain={gain}, state_model={state_model}, safe={safe}")
    else:
        config = build_config(gain=gain, state_model=state_model, safe=safe,
                              num_tune=num_tune)

    gain_tag = f"g{gain:g}".replace(".", "p")
    safe_tag = "_safe" if safe else ""
    leaf_tag = state_model  # 'binary' or 'three_state'
    idata_path = OUT_DIR / f"{leaf_tag}_gain_{gain_tag}{safe_tag}_anchored.nc"
    meta_path = OUT_DIR / f"{leaf_tag}_gain_{gain_tag}{safe_tag}_anchored.meta.json"
    tier1_md = ANALYSIS_DIR / f"tier1_diagnostics_{leaf_tag}_{gain_tag}{safe_tag}.md"

    meta_git = git_head()
    print(f"Branch: {meta_git['branch']}")
    print(f"Commit: {meta_git['commit']}")
    print(f"Config: INDICATOR_STATE_MODEL={state_model}, TRANSMISSION_GAIN={config.TRANSMISSION_GAIN}, "
          f"GAIN_LOGIT_NORMAL={config.GAIN_LOGIT_NORMAL}, "
          f"target_accept={config.TARGET_ACCEPT}, "
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

    if not smoke:
        write_tier1_markdown(diag, elapsed, config, tier1_md)
        meta = {
            "fit": idata_path.stem,
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
    print(f"max R-hat:   {diag['max_rhat']:.4f}")
    print(f"min ESS bulk: {diag['min_ess_bulk']:.0f}")


if __name__ == "__main__":
    import sys
    smoke = "--smoke" in sys.argv
    safe = "--safe" in sys.argv
    state_model = "three_state" if "--three-state" in sys.argv else "binary"
    gain = DEFAULT_GAIN
    num_tune = 1000 if safe else 500
    for arg in sys.argv[1:]:
        if arg.startswith("--gain="):
            gain = float(arg.split("=", 1)[1])
        elif arg.startswith("--num-tune="):
            num_tune = int(arg.split("=", 1)[1])
    main(smoke=smoke, gain=gain, state_model=state_model, safe=safe, num_tune=num_tune)
