"""Per-system PSIS-LOO comparison across asymmetric-prior sweep variants + baseline.

LOO unit: per-system (n=4 observations). With n=4 the LOO standard errors are wide;
the comparison is qualitative. See task4_loo.py for the full justification.

Reuses task4_loo.py infrastructure:
  - per_system_log_lik_from_draws (the exact-tree DP, vectorised by chain × draw)
  - extract_prod_arrays (production / sweep arrays)
  - extract_task2_arrays (per-system-obs runs)
  - thin_arrays / make_idata_with_loglik

Usage:
  python notebooks/asymmetric_prior_sweep_2026-05-10/per_system_loo.py
        # automatically picks up baseline + the four sweep run dirs that exist.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import arviz as az
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SISTER_DIR = REPO_ROOT / "notebooks" / "synthetic_validation_2026-05-06"
AUDIT_DIR = SISTER_DIR / "per_system_audit_2026-05-07"
SWEEP_RUNS = REPO_ROOT / "notebooks" / "asymmetric_prior_sweep_2026-05-10" / "runs"

for d in (REPO_ROOT, SISTER_DIR, AUDIT_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from dcm_model import ModelConfig, load_data  # noqa: E402
from gwt_oracle_internal_identifiability import STANCE  # noqa: E402
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS  # noqa: E402
from task4_loo import (  # noqa: E402
    collect_observed_ratings_by_indicator,
    extract_prod_arrays,
    make_idata_with_loglik,
    per_system_log_lik_from_draws,
    thin_arrays,
)


SYSTEMS = [s for s, _ in ANCHORED_SYSTEM_CONFIGS]


def candidate_runs() -> List[Tuple[str, Path]]:
    """Return [(label, fit.nc path), ...] for runs that exist on disk."""
    runs: List[Tuple[str, Path]] = []
    baseline = (
        SISTER_DIR / "runs/full_exact_recovery"
        / "exact_latent_tree__full_exact_tree__three_state_binomial_2__"
          "exact_tree_production_medians__current_gwt_rater_design__seed20260506"
        / "fit.nc"
    )
    if baseline.exists():
        runs.append(("baseline_synthetic_seed06", baseline))
    real_baseline = REPO_ROOT / "results/gwt_exact_tree/three_state_pooled_abs_by_sd_exact_anchored.nc"
    if real_baseline.exists():
        runs.append(("baseline_real_data", real_baseline))
    if SWEEP_RUNS.exists():
        for sub in sorted(SWEEP_RUNS.iterdir()):
            nc = sub / "fit.nc"
            if nc.exists():
                runs.append((sub.name, nc))
        real_dir = SWEEP_RUNS / "real_data"
        if real_dir.exists():
            for sub in sorted(real_dir.iterdir()):
                nc = sub / "fit.nc"
                if nc.exists():
                    runs.append((f"real_data__{sub.name}", nc))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-draws", type=int, default=400)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "notebooks/asymmetric_prior_sweep_2026-05-10/eval",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig(INDICATOR_STATE_MODEL="three_state")
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    observed_by_system = {
        s: collect_observed_ratings_by_indicator(stance_data, s, cfg.ORDINAL_BINS)
        for s in SYSTEMS
    }

    runs = candidate_runs()
    if not runs:
        raise SystemExit("No fits found on disk; run the sweep first.")
    print(f"Found {len(runs)} fits to score.")

    idata_dict: Dict[str, "az.InferenceData"] = {}
    rows: List[Dict] = []
    for label, nc in runs:
        print(f"--- {label} ({nc.relative_to(REPO_ROOT)}) ---")
        _, a, k, c, bp, ba = extract_prod_arrays(nc, stance_data, SYSTEMS)
        a, k, c, bp, ba = thin_arrays(a, k, c, bp, ba, args.keep_draws)
        print(f"  computing log-lik over {a[SYSTEMS[0]].shape}...")
        ll = per_system_log_lik_from_draws(
            stance_data, SYSTEMS, observed_by_system, a, k, c, bp, ba,
        )
        for s_idx, system in enumerate(SYSTEMS):
            mean_ll = float(np.mean(ll[..., s_idx]))
            rows.append({"run": label, "system": system, "mean_log_lik": mean_ll})
        idata_dict[label] = make_idata_with_loglik(ll, SYSTEMS)

    raw = pd.DataFrame(rows)
    raw_path = args.out_dir / "per_system_loglik.csv"
    raw.to_csv(raw_path, index=False)
    print(f"wrote {raw_path.relative_to(REPO_ROOT)}")

    print("--- az.compare (n=4 observations; qualitative only) ---")
    compare = az.compare(idata_dict, ic="loo", method="stacking")
    cmp_path = args.out_dir / "per_system_loo_compare.csv"
    compare.to_csv(cmp_path)
    print(compare)
    print(f"wrote {cmp_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
