"""Quick-and-dirty multi-stance pool_3s + soft-anchor fits for exploration.

Not a production driver — drops sampling to NUM_SAMPLES=1000, NUM_TUNE=500 to
keep runtime manageable across multiple stances. Writes idata to
`results/explore_stance/{stance_slug}.nc` (gitignored tree).
"""

from __future__ import annotations
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import arviz as az
import numpy as np

from dcm_model import (
    EvidenceProcessor, ModelConfig, MultiSystemDataProcessor,
    MultiSystemModelBuilder, fit_stance_multisystem, load_data,
)


SOFT_ANCHORS = {"Human": (50.0, 1.0), "ELIZA": (1.0, 50.0)}
SYSTEM_CONFIGS = [
    ("Human", 0.999),
    ("Chicken", None),
    ("2024 Leading Chat LLMs", None),
    ("ELIZA", 0.001),
]
OUT_DIR = Path("results/explore_stance")


def slugify(stance_name: str) -> str:
    return stance_name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def build_quick_config(state_model: str = "three_state") -> ModelConfig:
    return ModelConfig(
        INDICATOR_STATE_MODEL=state_model,
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=True,
        LABEL_POOL_SIGMA=0.5,
        SOFT_REFERENCE_ANCHORS=SOFT_ANCHORS,
        NUM_SAMPLES=1000,
        NUM_TUNE=500,
        NUM_CHAINS=4,
        TARGET_ACCEPT=0.9,
    )


def fit_one(stance_name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_nc = OUT_DIR / f"{slugify(stance_name)}.nc"
    out_meta = OUT_DIR / f"{slugify(stance_name)}.meta.json"

    cfg = build_quick_config()
    stance_data = next((s for s in load_data(cfg) if s["name"] == stance_name), None)
    if stance_data is None:
        raise ValueError(f"stance not found: {stance_name}")

    print(f"fitting {stance_name}...")
    t0 = time.time()
    idata, _bld, _proc = fit_stance_multisystem(stance_data, cfg, SYSTEM_CONFIGS)
    elapsed = time.time() - t0
    az.to_netcdf(idata, str(out_nc))

    div = int(idata.sample_stats["diverging"].values.sum())
    vars_c = [v for v in idata.posterior.data_vars if v.endswith("_C")]
    summ = az.summary(idata, var_names=vars_c + ["a", "kappa"], kind="diagnostics")
    meta = {
        "stance": stance_name,
        "elapsed_s": elapsed,
        "divergences": div,
        "max_rhat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "idata_path": str(out_nc),
        "config": asdict(cfg),
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"  wrote {out_nc}  ({elapsed:.0f}s, {div} divs, R-hat {summ['r_hat'].max():.3f})")


if __name__ == "__main__":
    stance = " ".join(sys.argv[1:]).strip()
    if not stance:
        print("usage: python explore_stance_comparison.py 'Stance Name'")
        sys.exit(2)
    fit_one(stance)
