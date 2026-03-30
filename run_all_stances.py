"""Fit the ordinal DCM across all stances and cache results.

Usage
-----
    python run_all_stances.py                      # full run, all 13 stances
    python run_all_stances.py --quick              # fast exploration (~7 min)
    python run_all_stances.py --stances "GWT,RPT"  # subset (substring match)
    python run_all_stances.py --skip-cached        # resume an interrupted run
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import arviz as az
import numpy as np

from dcm_model import ModelConfig, fit_stance, load_data, setup_logging


def sanitise_filename(name: str) -> str:
    """Convert a stance name to a safe filename stem."""
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def extract_summary(
    stance_name: str,
    idata: Any,
    builder: Any,
    processor: Any,
    elapsed: float,
) -> Dict:
    """Extract headline quantities from a fitted stance."""
    post = idata.posterior
    stance_varname = builder.node_to_varname[stance_name]
    draws = post[f"{stance_varname}_beta"].values.flatten()

    rhat = az.rhat(idata)
    ess = az.ess(idata)
    max_rhat = max(float(rhat[v].max()) for v in rhat.data_vars)
    min_ess = min(float(ess[v].min()) for v in ess.data_vars)

    n_experts = len(processor.expert_names)
    a_mean = float(post["a"].mean())

    return {
        "stance": stance_name,
        "p_consciousness_mean": float(draws.mean()),
        "p_consciousness_ci_low": float(np.percentile(draws, 3)),
        "p_consciousness_ci_high": float(np.percentile(draws, 97)),
        "n_indicators": len(processor.observations),
        "n_observations": sum(len(v) for v in processor.observations.values()),
        "n_experts": n_experts,
        "anchor_expert": processor.anchor_expert,
        "discrimination_a": a_mean,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "elapsed_s": round(elapsed, 1),
    }


def print_comparison_table(summaries: List[Dict]) -> None:
    """Print a formatted comparison table to stdout."""
    sorted_s = sorted(summaries, key=lambda s: s["p_consciousness_mean"], reverse=True)

    print(f"\n{'=' * 90}")
    print("P(consciousness = 1 | data) across stances")
    print(f"{'=' * 90}")
    print(
        f"{'Stance':<45} {'P(C=1)':>7} {'94% CI':>16} "
        f"{'Rhat':>6} {'ESS':>6} {'Time':>6}"
    )
    print("-" * 90)
    for s in sorted_s:
        rhat_flag = " *" if s["max_rhat"] > 1.01 else ""
        ess_flag = " *" if s["min_ess"] < 100 else ""
        print(
            f"{s['stance']:<45} {s['p_consciousness_mean']:>7.3f} "
            f"[{s['p_consciousness_ci_low']:.3f}, {s['p_consciousness_ci_high']:.3f}] "
            f"{s['max_rhat']:>6.3f}{rhat_flag}"
            f"{s['min_ess']:>6.0f}{ess_flag}"
            f"{s['elapsed_s']:>5.0f}s"
        )
    print("-" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit ordinal DCM across stances")
    parser.add_argument(
        "--stances",
        type=str,
        default="all",
        help='Comma-separated stance substrings, or "all" (default: all)',
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast exploration: 200 draws, 200 tune, 2 chains",
    )
    parser.add_argument(
        "--skip-cached",
        action="store_true",
        help="Skip stances that already have a cached .nc file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for cached results (default: results/)",
    )
    parser.add_argument(
        "--draws", type=int, default=None, help="Override number of posterior draws"
    )
    parser.add_argument(
        "--tune", type=int, default=None, help="Override number of tuning steps"
    )
    parser.add_argument(
        "--chains", type=int, default=None, help="Override number of chains"
    )
    args = parser.parse_args()

    logger = setup_logging("INFO")

    # Config
    if args.quick:
        config = ModelConfig(NUM_SAMPLES=200, NUM_TUNE=200, NUM_CHAINS=2, TARGET_ACCEPT=0.9)
    else:
        config = ModelConfig(NUM_SAMPLES=1000, NUM_TUNE=1000, NUM_CHAINS=4, TARGET_ACCEPT=0.95)

    if args.draws is not None:
        config.NUM_SAMPLES = args.draws
    if args.tune is not None:
        config.NUM_TUNE = args.tune
    if args.chains is not None:
        config.NUM_CHAINS = args.chains

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    # Load data
    all_data = load_data(config)
    all_stance_names = [item["name"] for item in all_data]

    # Filter stances
    if args.stances.lower() == "all":
        stances_to_fit = all_data
    else:
        query_parts = [q.strip() for q in args.stances.split(",")]
        stances_to_fit = [
            item
            for item in all_data
            if any(q.lower() in item["name"].lower() for q in query_parts)
        ]
        if not stances_to_fit:
            logger.error(
                f"No stances matched: {args.stances}\n"
                f"Available: {all_stance_names}"
            )
            return

    logger.info(
        f"Fitting {len(stances_to_fit)} stance(s): "
        f"{[s['name'] for s in stances_to_fit]}"
    )
    logger.info(
        f"Config: {config.NUM_SAMPLES} draws, {config.NUM_TUNE} tune, "
        f"{config.NUM_CHAINS} chains"
    )

    # Fit loop
    summaries: List[Dict] = []
    total_t0 = time.time()

    for i, stance_data in enumerate(stances_to_fit, 1):
        name = stance_data["name"]
        nc_path = out_dir / f"{sanitise_filename(name)}.nc"

        if args.skip_cached and nc_path.exists():
            logger.info(f"[{i}/{len(stances_to_fit)}] Skipping {name} (cached)")
            # Load summary from existing file if available
            summary_path = out_dir / "summary.json"
            if summary_path.exists():
                existing = json.loads(summary_path.read_text())
                match = [s for s in existing if s["stance"] == name]
                if match:
                    summaries.append(match[0])
            continue

        logger.info(f"[{i}/{len(stances_to_fit)}] Fitting: {name}")
        t0 = time.time()

        idata, builder, processor = fit_stance(stance_data, config)
        elapsed = time.time() - t0

        # Save idata
        idata.to_netcdf(str(nc_path))
        logger.info(f"  Saved to {nc_path} ({elapsed:.0f}s)")

        summary = extract_summary(name, idata, builder, processor, elapsed)
        summaries.append(summary)

        # Incremental save of summary
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summaries, indent=2))

    total_elapsed = time.time() - total_t0
    logger.info(f"Total time: {total_elapsed:.0f}s")

    # Final outputs
    if summaries:
        print_comparison_table(summaries)

        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summaries, indent=2))
        logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
