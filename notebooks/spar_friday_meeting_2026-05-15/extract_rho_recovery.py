"""Extract per-seed posterior P(R = 1 | y) for the unanchored systems
(Chicken, LLMs) from the 10-seed pilot fits, and compare to the truth root_z.

Posterior P(R = 1 | y) is taken as the posterior mean of `_rho_collapsed`
(equivalently the mean of `_rho`, and equivalently $(α + β + 1) E[C | y] - α$
under the conjugate Beta(α, β) hierarchy). All three give the same posterior
P(R = 1 | y) within Monte Carlo noise; we report `_rho_collapsed`.

Outputs:
  - rho_recovery_per_seed.csv (per-seed, per-system rows)
  - rho_recovery_summary.csv  (per-system summary)
  - prints a markdown table for paste into the meeting notebook
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import arviz as az
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "outputs" / "main_synthetic_validation_pilot_20260511" / "runs" / "targeted_strong_lower_override"
OUT_DIR = Path(__file__).resolve().parent

UNANCHORED_SYSTEMS = {
    "Chicken": "chicken__global_workspace_theory",
    "LLMs (2024)": "2024_leading_chat_llms__global_workspace_theory",
}
ANCHORED_SYSTEMS = {
    "Human": "human__global_workspace_theory",
    "ELIZA": "eliza__global_workspace_theory",
}
ALL_SYSTEMS = {**ANCHORED_SYSTEMS, **UNANCHORED_SYSTEMS}
TRUTH_KEYS = {
    "Human": "Human",
    "ELIZA": "ELIZA",
    "Chicken": "Chicken",
    "LLMs (2024)": "2024 Leading Chat LLMs",
}


def extract_for_seed(seed_dir: Path) -> List[Dict]:
    fit = az.from_netcdf(seed_dir / "fit.nc")
    with open(seed_dir / "truth_payload.json") as fh:
        truth = json.load(fh)
    rows = []
    for sys_label, var_prefix in ALL_SYSTEMS.items():
        truth_key = TRUTH_KEYS[sys_label]
        truth_R = int(truth["root_z_by_system"][truth_key])
        truth_C = float(truth.get("c_true_by_system", {}).get(truth_key, np.nan))
        rho_collapsed = fit.posterior[f"{var_prefix}_rho_collapsed"].values
        rho_full = fit.posterior[f"{var_prefix}_rho"].values
        log_B = fit.posterior[f"{var_prefix}_log_B"].values
        c_post = fit.posterior[f"{var_prefix}_C"].values
        rows.append({
            "seed": int(seed_dir.name.replace("seed_", "")),
            "system": sys_label,
            "truth_R": truth_R,
            "truth_C": truth_C,
            "post_mean_C": float(np.mean(c_post)),
            "post_mean_rho_collapsed": float(np.mean(rho_collapsed)),
            "post_mean_rho_full": float(np.mean(rho_full)),
            "post_mean_log_B": float(np.mean(log_B)),
            "rho_implied_from_meanC": 7.0 * float(np.mean(c_post)) - 1.0,
            "post_sd_C": float(np.std(c_post)),
        })
    return rows


def main() -> None:
    seed_dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    print(f"Found {len(seed_dirs)} seed directories")
    all_rows: List[Dict] = []
    for sd in seed_dirs:
        all_rows.extend(extract_for_seed(sd))
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["system", "seed"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "rho_recovery_per_seed.csv", index=False)
    print(f"\nsaved: {OUT_DIR / 'rho_recovery_per_seed.csv'}")

    # Per-system summary on unanchored systems, split by truth R
    summary_rows = []
    for sys_label in UNANCHORED_SYSTEMS:
        sub = df[df["system"] == sys_label]
        for truth_R in (0, 1):
            cases = sub[sub["truth_R"] == truth_R]
            if len(cases) == 0:
                continue
            summary_rows.append({
                "system": sys_label,
                "truth_R": truth_R,
                "n_cases": len(cases),
                "median_post_rho": float(np.median(cases["post_mean_rho_collapsed"])),
                "min_post_rho": float(np.min(cases["post_mean_rho_collapsed"])),
                "max_post_rho": float(np.max(cases["post_mean_rho_collapsed"])),
                "frac_correct_at_0p5": float(np.mean(
                    cases["post_mean_rho_collapsed"] > 0.5 if truth_R == 1
                    else cases["post_mean_rho_collapsed"] <= 0.5
                )),
                "median_log_BF": float(np.median(cases["post_mean_log_B"])),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "rho_recovery_summary.csv", index=False)
    print(f"saved: {OUT_DIR / 'rho_recovery_summary.csv'}")

    print("\n--- PER-SEED TABLE (unanchored systems only) ---")
    sub = df[df["system"].isin(UNANCHORED_SYSTEMS.keys())]
    print(sub[["seed", "system", "truth_R", "truth_C", "post_mean_C", "post_mean_rho_collapsed", "post_mean_log_B"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n--- SUMMARY (unanchored systems, split by truth) ---")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Also report posterior on anchored systems as sanity
    print("\n--- ANCHORED-SYSTEM SANITY (Human / ELIZA across seeds) ---")
    for sys_label in ANCHORED_SYSTEMS:
        sub = df[df["system"] == sys_label]
        print(
            f"  {sys_label:6s}  "
            f"median post ρ = {np.median(sub['post_mean_rho_collapsed']):.3f}  "
            f"(min {np.min(sub['post_mean_rho_collapsed']):.3f}, "
            f"max {np.max(sub['post_mean_rho_collapsed']):.3f})  "
            f"truth R = {int(np.median(sub['truth_R']))}"
        )


if __name__ == "__main__":
    main()
