"""Aggregate sweep outputs into one wide table per estimand for findings.md.

Walks the sweep run directories, pulls root_recovery.csv (synthetic) /
meta.json (real-data) and per_indicator_ppc.csv (after running per_indicator_ppc.py
against each run), and produces:

  eval/free_c_recovery.csv       — wide table across runs × free systems
  eval/per_indicator_summary.csv — per-run summary of chi2/KL/cov94 by system
  eval/per_indicator_long.csv    — full per-indicator long table for plotting
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

SWEEP_DIR = Path(__file__).resolve().parent
SYNTH_RUNS = SWEEP_DIR / "runs/synthetic"
REAL_RUNS = SWEEP_DIR / "runs/real_data"
PER_SYS_RUNS = SWEEP_DIR / "runs/per_system_noanchors"
EVAL = SWEEP_DIR / "eval"


def collect_synthetic_root_recovery() -> pd.DataFrame:
    rows: List[Dict] = []
    for run_dir in sorted(SYNTH_RUNS.glob("*")):
        rec = run_dir / "root_recovery.csv"
        if not rec.exists():
            continue
        df = pd.read_csv(rec)
        for _, row in df.iterrows():
            rows.append({
                "scope": "synthetic_ref_anchored",
                "run": run_dir.name,
                "system": row["system"],
                "is_hard_anchor": row["is_hard_anchor"],
                "truth": row["truth"],
                "posterior_median": row["posterior_median"],
                "p03": row["posterior_p03"],
                "p97": row["posterior_p97"],
                "signed_error": row["signed_error"],
                "interval_includes_truth": row["interval_includes_truth"],
            })
    if PER_SYS_RUNS.exists():
        for run_dir in sorted(PER_SYS_RUNS.glob("*")):
            rec = run_dir / "root_recovery.csv"
            if not rec.exists():
                continue
            df = pd.read_csv(rec)
            for _, row in df.iterrows():
                rows.append({
                    "scope": "synthetic_no_anchors",
                    "run": run_dir.name,
                    "system": row["system"],
                    "is_hard_anchor": False,
                    "truth": row["truth"],
                    "posterior_median": row["posterior_median"],
                    "p03": row["posterior_p03"],
                    "p97": row["posterior_p97"],
                    "signed_error": row["signed_error"],
                    "interval_includes_truth": row["interval_includes_truth"],
                })
    return pd.DataFrame(rows)


def collect_real_data_headlines() -> pd.DataFrame:
    rows: List[Dict] = []
    if not REAL_RUNS.exists():
        return pd.DataFrame()
    for run_dir in sorted(REAL_RUNS.glob("*")):
        meta = run_dir / "meta.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text())
        head = m.get("headline", {})
        for system, vals in head.items():
            rows.append({
                "scope": "real_data_ref_anchored",
                "run": run_dir.name,
                "system": system,
                "posterior_median": vals.get("median"),
                "p03": vals.get("p03"),
                "p97": vals.get("p97"),
                "diagnostic_status": m.get("diagnostic_status"),
                "divergences": m.get("divergences"),
                "max_rhat": m.get("max_rhat"),
            })
    return pd.DataFrame(rows)


def collect_per_indicator() -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: List[pd.DataFrame] = []
    for base in (SYNTH_RUNS, REAL_RUNS, PER_SYS_RUNS):
        if not base.exists():
            continue
        scope = (
            "synthetic_ref_anchored" if base == SYNTH_RUNS
            else ("real_data_ref_anchored" if base == REAL_RUNS else "synthetic_no_anchors")
        )
        for run_dir in sorted(base.glob("*")):
            csv = run_dir / "per_indicator_ppc.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            df["run"] = run_dir.name
            df["scope"] = scope
            long_rows.append(df)
    long_df = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()
    if long_df.empty:
        return long_df, long_df
    summary = (
        long_df.groupby(["scope", "run", "system"], dropna=False)
        .agg(
            n_indicators=("chi2", "size"),
            mean_chi2=("chi2", "mean"),
            mean_kl=("kl", "mean"),
            cov94_frac=("cov94", "mean"),
        )
        .reset_index()
    )
    return summary, long_df


def main() -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    syn = collect_synthetic_root_recovery()
    real = collect_real_data_headlines()
    summary, long_df = collect_per_indicator()
    if not syn.empty:
        syn.to_csv(EVAL / "free_c_recovery.csv", index=False)
        print(f"wrote {EVAL / 'free_c_recovery.csv'} ({len(syn)} rows)")
    if not real.empty:
        real.to_csv(EVAL / "real_data_headlines.csv", index=False)
        print(f"wrote {EVAL / 'real_data_headlines.csv'} ({len(real)} rows)")
    if not summary.empty:
        summary.to_csv(EVAL / "per_indicator_summary.csv", index=False)
        long_df.to_csv(EVAL / "per_indicator_long.csv", index=False)
        print(f"wrote {EVAL / 'per_indicator_summary.csv'} ({len(summary)} rows)")
        print(f"wrote {EVAL / 'per_indicator_long.csv'} ({len(long_df)} rows)")


if __name__ == "__main__":
    main()
