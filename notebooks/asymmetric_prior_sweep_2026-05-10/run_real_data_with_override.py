"""Real-data exact-tree refit on GWT under asymmetric β_pres / β_abs prior overrides.

Mirrors run_exact_tree_production.py but exposes the three new override knobs
(BETA_PRES_OVERRIDE_MEAN, BETA_ABS_OVERRIDE_MEAN, BETA_OVERRIDE_SIGMA) via CLI
and writes outputs under
  notebooks/asymmetric_prior_sweep_2026-05-10/runs/real_data/<override_tag>/
so the production NetCDF and the sweep variants don't clobber each other.

Usage:
  python notebooks/asymmetric_prior_sweep_2026-05-10/run_real_data_with_override.py \
      --beta-pres-mean 0.90 --beta-abs-mean 0.10 --beta-override-sigma 0.30
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import arviz as az
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcm_model import (  # noqa: E402
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    load_data,
)
from dcm_model_exact_tree import MultiSystemExactTreeBuilder  # noqa: E402
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS  # noqa: E402
from run_exact_tree_smoke import run_pretrain_sanity_check  # noqa: E402


STANCE = "Global Workspace Theory"
SYSTEM_CONFIGS = list(ANCHORED_SYSTEM_CONFIGS)
DEFAULT_OUT_ROOT = REPO_ROOT / "notebooks/asymmetric_prior_sweep_2026-05-10/runs/real_data"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta-pres-mean", type=float, required=True)
    parser.add_argument("--beta-abs-mean", type=float, required=True)
    parser.add_argument("--beta-override-sigma", type=float, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="200 tune × 200 draws × 2 chains; ~2 min compile-and-fit check.")
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--tune", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ModelConfig:
    if args.smoke:
        draws = args.draws or 200
        tune = args.tune or 200
        chains = args.chains or 2
    else:
        draws = args.draws or 2000
        tune = args.tune or 1000
        chains = args.chains or 4
    return ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=True,
        BETA_ABS_BY_SUPPORT_DEMAND=True,
        LABEL_POOL_SIGMA=0.5,
        BETA_PRES_OVERRIDE_MEAN=float(args.beta_pres_mean),
        BETA_ABS_OVERRIDE_MEAN=float(args.beta_abs_mean),
        BETA_OVERRIDE_SIGMA=(
            float(args.beta_override_sigma) if args.beta_override_sigma is not None else None
        ),
        NUM_SAMPLES=draws,
        NUM_TUNE=tune,
        NUM_CHAINS=chains,
        TARGET_ACCEPT=args.target_accept,
    )


def override_tag(args: argparse.Namespace) -> str:
    parts = [
        f"pres{int(round(args.beta_pres_mean * 100)):02d}",
        f"abs{int(round(args.beta_abs_mean * 100)):02d}",
    ]
    if args.beta_override_sigma is not None:
        parts.append(f"sig{int(round(args.beta_override_sigma * 100)):02d}")
    return "_".join(parts)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    tag = override_tag(args) + ("__smoke" if args.smoke else "")
    out_dir: Path = args.out_root / tag
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Run directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    git = git_head()
    print(f"Branch: {git['branch']}")
    print(f"Commit: {git['commit']}")
    print(
        f"Override: pres_mean={cfg.BETA_PRES_OVERRIDE_MEAN}, "
        f"abs_mean={cfg.BETA_ABS_OVERRIDE_MEAN}, "
        f"override_sigma={cfg.BETA_OVERRIDE_SIGMA}"
    )
    print(
        f"Sampling: chains={cfg.NUM_CHAINS}, tune={cfg.NUM_TUNE}, "
        f"draws={cfg.NUM_SAMPLES}, target_accept={cfg.TARGET_ACCEPT}"
    )

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
    print(f"build seconds: {elapsed_build:.1f}")

    print("Pre-sampling sanity check (cross-check vs NumPy DP)...")
    sanity_ok = run_pretrain_sanity_check(builder, model, stance_data, proc)

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
            random_seed=args.seed,
        )
    elapsed_sample = time.time() - t1
    print(f"sample seconds: {elapsed_sample:.1f}")

    div_count = int(idata.sample_stats["diverging"].values.sum())
    var_names = ["a", "kappa"]
    for v in idata.posterior.data_vars:
        name = str(v)
        if name.endswith("_C"):
            d = np.asarray(idata.posterior[v].values).reshape(-1)
            if np.std(d) > 1e-10:
                var_names.append(name)
        if (
            name.startswith("beta_pres__")
            or name.startswith("beta_abs__")
            or name.startswith("label_delta__")
        ):
            var_names.append(name)
    diag = az.summary(idata, var_names=var_names, kind="diagnostics")
    max_rhat = float(diag["r_hat"].max())
    min_ess = float(diag["ess_bulk"].min())

    headline: Dict[str, Dict[str, float]] = {}
    for label, prefix in (
        ("Human", "human"),
        ("Chicken", "chicken"),
        ("LLMs", "2024_leading_chat_llms"),
        ("ELIZA", "eliza"),
    ):
        d = np.asarray(idata.posterior[f"{prefix}__global_workspace_theory_C"].values).reshape(-1)
        headline[label] = {
            "median": float(np.median(d)),
            "p03": float(np.percentile(d, 3)),
            "p97": float(np.percentile(d, 97)),
        }

    nc_path = out_dir / "fit.nc"
    az.to_netcdf(idata, str(nc_path))
    diag.reset_index().to_csv(out_dir / "diagnostics.csv", index=False)

    meta = {
        "phase": "real_data_asymmetric_override",
        "branch": git["branch"], "commit": git["commit"],
        "config": asdict(cfg),
        "elapsed_s_build": elapsed_build,
        "elapsed_s_sample": elapsed_sample,
        "divergences": div_count,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess,
        "headline": headline,
        "sanity_check_passed": sanity_ok,
        "diagnostic_status": (
            "passed" if div_count == 0 and max_rhat <= 1.01 else "exploratory_failed"
        ),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {nc_path}")
    print(f"diag: divergences={div_count}, max_rhat={max_rhat:.4f}, min_ess_bulk={min_ess:.0f}")
    for label, h in headline.items():
        print(f"  {label:<8}  {h['median']:.4f} [{h['p03']:.4f}, {h['p97']:.4f}]")


if __name__ == "__main__":
    main()
