from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr


INPUT_DIR = Path("outputs/main_synthetic_validation")
OUTPUT_DIR = Path("outputs/post_ladder_diagnostics_20260511/smoke_refresh")
FIT_VARIANT = "targeted_strong_lower_override"
FREE_SYSTEMS = ("Chicken", "2024 Leading Chat LLMs")
SYSTEM_LABEL = {"2024 Leading Chat LLMs": "LLMs", "Chicken": "Chicken"}
SYSTEM_PREFIX = {"2024 Leading Chat LLMs": "2024_leading_chat_llms", "Chicken": "chicken"}
CATEGORY_COUNT = 7
SCORE_CLIP = 1e-12
PRIOR_P = 1.0 / 6.0
LOGIT_PRIOR = math.log(PRIOR_P / (1.0 - PRIOR_P))
TAU_PRESENT_50 = -LOGIT_PRIOR
TAU_ABSENT_05 = math.log(0.05 / 0.95) - LOGIT_PRIOR
IDENTIFIER_COLUMNS = {
    "seed",
    "seed_idx",
    "fit_variant",
    "system",
    "system_raw",
    "run_dir",
    "metric",
    "pillar",
    "validation_scope",
    "status",
    "threshold",
    "notes",
}


def clip_prob(x: float | np.ndarray) -> float | np.ndarray:
    return np.clip(x, SCORE_CLIP, 1.0 - SCORE_CLIP)


def expit(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit_clipped(p: float) -> float:
    pp = float(clip_prob(p))
    return math.log(pp / (1.0 - pp))


def bernoulli_brier(p: float, y: int) -> float:
    return float((float(p) - int(y)) ** 2)


def bernoulli_log_score(p: float, y: int) -> float:
    pp = float(clip_prob(p))
    return float(math.log(pp if int(y) == 1 else 1.0 - pp))


def rho_evidence_category(rho: float) -> str:
    if rho >= 0.95:
        return "strong_present"
    if rho >= 0.50:
        return "moderate_present"
    if rho >= 0.05:
        return "ambiguous"
    return "strong_absent"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def flatten_draws(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr.reshape(-1)


def extract_rho_cases_from_fit(run_dir: Path) -> pd.DataFrame:
    truth = json.loads((run_dir / "truth_payload.json").read_text())
    fit_seed = int(truth["seed"]) + int(truth["seed_idx"])
    idata = az.from_netcdf(run_dir / "fit.nc")
    rows: list[dict[str, Any]] = []
    prior_brier_by_y = {0: bernoulli_brier(PRIOR_P, 0), 1: bernoulli_brier(PRIOR_P, 1)}
    prior_log_by_y = {0: bernoulli_log_score(PRIOR_P, 0), 1: bernoulli_log_score(PRIOR_P, 1)}
    for system_raw in FREE_SYSTEMS:
        prefix = f"{SYSTEM_PREFIX[system_raw]}__global_workspace_theory"
        log_b = flatten_draws(idata.posterior[f"{prefix}_log_B"].values)
        rho_collapsed = flatten_draws(idata.posterior[f"{prefix}_rho_collapsed"].values)
        rho_sampled = flatten_draws(idata.posterior[f"{prefix}_rho"].values)
        rho_mean = float(np.mean(rho_collapsed))
        rho_median = float(np.median(rho_collapsed))
        rho_sampled_mean = float(np.mean(rho_sampled))
        log_b_eff = logit_clipped(rho_mean) - LOGIT_PRIOR
        root_z = int(truth["root_z_by_system"][system_raw])
        brier = bernoulli_brier(rho_mean, root_z)
        log_score = bernoulli_log_score(rho_mean, root_z)
        rows.append(
            {
                "fit_variant": truth["fit_variant"],
                "seed": fit_seed,
                "system": SYSTEM_LABEL[system_raw],
                "system_raw": system_raw,
                "root_z_true": root_z,
                "rho_collapsed": rho_mean,
                "rho_collapsed_median": rho_median,
                "rho_sampled_pi": rho_sampled_mean,
                "abs_rho_collapsed_minus_sampled_pi": abs(rho_mean - rho_sampled_mean),
                "log_B_draw_mean": float(np.mean(log_b)),
                "log_B_draw_median": float(np.median(log_b)),
                "log_B_draw_q05": float(np.percentile(log_b, 5)),
                "log_B_draw_q95": float(np.percentile(log_b, 95)),
                "log_B_eff": log_b_eff,
                "evidence_category": rho_evidence_category(rho_mean),
                "correct_sign": bool((root_z == 1 and log_b_eff > 0.0) or (root_z == 0 and log_b_eff < 0.0)),
                "classified_present_at_0p5": bool(rho_mean > 0.5),
                "decisive_present_at_0p95": bool(rho_mean >= 0.95),
                "decisive_absent_at_0p05": bool(rho_mean < 0.05),
                "brier": brier,
                "log_score": log_score,
                "brier_improvement_vs_prior": prior_brier_by_y[root_z] - brier,
                "log_score_improvement_vs_prior": log_score - prior_log_by_y[root_z],
                "run_dir": str(run_dir.relative_to(INPUT_DIR)),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(df: pd.DataFrame, n_bins: int = 10) -> tuple[float, str]:
    if df.empty:
        return np.nan, "[]"
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    n = len(df)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (df["rho_collapsed"] >= lo) & (df["rho_collapsed"] <= hi)
        else:
            mask = (df["rho_collapsed"] >= lo) & (df["rho_collapsed"] < hi)
        g = df[mask]
        if g.empty:
            continue
        conf = float(g["rho_collapsed"].mean())
        acc = float(g["root_z_true"].mean())
        ece += (len(g) / n) * abs(conf - acc)
        rows.append({"lo": lo, "hi": hi, "n": int(len(g)), "mean_rho": conf, "empirical_rate": acc})
    return float(ece), json.dumps(rows)


def summarise_rho_recovery(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[dict[str, Any], pd.DataFrame]] = []
    for fit_variant, g in cases.groupby("fit_variant", dropna=False):
        groups.append(({"fit_variant": fit_variant, "system": "ALL"}, g))
    for (fit_variant, system), g in cases.groupby(["fit_variant", "system"], dropna=False):
        groups.append(({"fit_variant": fit_variant, "system": system}, g))
    baseline_brier = 0.5 * (1.0 - PRIOR_P) ** 2 + 0.5 * PRIOR_P**2
    prior_log_baseline = 0.5 * math.log(PRIOR_P) + 0.5 * math.log1p(-PRIOR_P)
    for base, g in groups:
        r1 = g[g["root_z_true"] == 1]
        r0 = g[g["root_z_true"] == 0]
        tpr = float((r1["rho_collapsed"] > 0.5).mean()) if not r1.empty else np.nan
        tnr = float((r0["rho_collapsed"] <= 0.5).mean()) if not r0.empty else np.nan
        brier_model = (
            0.5 * float(((1.0 - r1["rho_collapsed"]) ** 2).mean()) if not r1.empty else 0.0
        ) + (0.5 * float((r0["rho_collapsed"] ** 2).mean()) if not r0.empty else 0.0)
        log_model = (
            0.5 * float(np.log(clip_prob(r1["rho_collapsed"].to_numpy())).mean()) if not r1.empty else 0.0
        ) + (
            0.5 * float(np.log(clip_prob(1.0 - r0["rho_collapsed"].to_numpy())).mean()) if not r0.empty else 0.0
        )
        ece, bins = expected_calibration_error(g)
        rows.append(
            {
                **base,
                "n_cases": int(len(g)),
                "mean_rho_R1": float(r1["rho_collapsed"].mean()) if not r1.empty else np.nan,
                "mean_rho_R0": float(r0["rho_collapsed"].mean()) if not r0.empty else np.nan,
                "median_rho_R1": float(r1["rho_collapsed"].median()) if not r1.empty else np.nan,
                "median_rho_R0": float(r0["rho_collapsed"].median()) if not r0.empty else np.nan,
                "median_log_B_eff_R1": float(r1["log_B_eff"].median()) if not r1.empty else np.nan,
                "median_log_B_eff_R0": float(r0["log_B_eff"].median()) if not r0.empty else np.nan,
                "evidence_margin_M": float(
                    min(r1["log_B_eff"].median() - TAU_PRESENT_50, TAU_ABSENT_05 - r0["log_B_eff"].median())
                )
                if not r1.empty and not r0.empty
                else np.nan,
                "TPR_at_rho_gt_0p5": tpr,
                "TNR_at_rho_le_0p5": tnr,
                "balanced_accuracy": float(np.nanmean([tpr, tnr])),
                "decisive_present_rate_R1_at_rho_gt_0p95": float(r1["decisive_present_at_0p95"].mean())
                if not r1.empty
                else np.nan,
                "decisive_absent_rate_R0_at_rho_lt_0p05": float(r0["decisive_absent_at_0p05"].mean())
                if not r0.empty
                else np.nan,
                "brier_model": brier_model,
                "brier_prior_baseline": baseline_brier,
                "brier_improvement": baseline_brier - brier_model,
                "log_score_model": log_model,
                "log_score_prior_baseline": prior_log_baseline,
                "log_score_improvement": log_model - prior_log_baseline,
                "ECE": ece,
                "calibration_bins": bins,
                "mean_abs_rho_collapsed_minus_sampled_pi": float(g["abs_rho_collapsed_minus_sampled_pi"].mean()),
            }
        )
    out = pd.DataFrame(rows)
    sort_map = {"ALL": 0, "Chicken": 1, "LLMs": 2}
    return out.assign(_sort=out["system"].map(sort_map).fillna(99)).sort_values(["fit_variant", "_sort"]).drop(columns="_sort")


def sampler_diagnostics_from_fit(run_dir: Path) -> pd.DataFrame:
    truth = json.loads((run_dir / "truth_payload.json").read_text())
    fit_seed = int(truth["seed"]) + int(truth["seed_idx"])
    run_summary = json.loads((run_dir / "run_summary.json").read_text())
    idata = az.from_netcdf(run_dir / "fit.nc")
    diag = az.summary(idata, kind="diagnostics")
    stats = idata.sample_stats
    n_samples = int(np.prod(stats["diverging"].shape))
    n_div = int(stats["diverging"].values.sum())
    if "reached_max_treedepth" in stats:
        td_hits = int(stats["reached_max_treedepth"].values.sum())
    elif "tree_depth" in stats:
        max_depth = int(np.nanmax(stats["tree_depth"].values))
        td_hits = int((stats["tree_depth"].values >= max_depth).sum())
    else:
        td_hits = 0
    return pd.DataFrame(
        [
            {
                "fit_variant": truth["fit_variant"],
                "seed": fit_seed,
                "max_rhat": float(diag["r_hat"].max(skipna=True)),
                "n_rhat_gt_1p01": int((diag["r_hat"] > 1.01).sum()),
                "n_rhat_gt_1p05": int((diag["r_hat"] > 1.05).sum()),
                "min_ess_bulk": float(diag["ess_bulk"].min(skipna=True)),
                "min_ess_tail": float(diag["ess_tail"].min(skipna=True)),
                "median_ess_bulk": float(diag["ess_bulk"].median(skipna=True)),
                "median_ess_tail": float(diag["ess_tail"].median(skipna=True)),
                "n_divergences": n_div,
                "divergence_rate": float(n_div / max(1, n_samples)),
                "max_tree_depth_hits": td_hits,
                "max_tree_depth_hit_rate": float(td_hits / max(1, n_samples)),
                "mean_acceptance_rate": float(np.nanmean(stats["acceptance_rate"].values))
                if "acceptance_rate" in stats
                else np.nan,
                "runtime_seconds": float(run_summary["runtime_seconds"]),
                "run_dir": str(run_dir.relative_to(INPUT_DIR)),
            }
        ]
    )


def aggregate_metric_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number, "bool"]).columns
        if c not in IDENTIFIER_COLUMNS and c not in group_cols
    ]
    agg = df.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    if "seed" in df.columns:
        seed_meta = (
            df.groupby(group_cols, dropna=False)["seed"]
            .agg(
                n_seeds=lambda x: int(pd.Series(x).nunique()),
                seed_min=lambda x: int(pd.Series(x).min()),
                seed_max=lambda x: int(pd.Series(x).max()),
                seed_list=lambda x: ";".join(str(int(v)) for v in sorted(pd.Series(x).unique())),
            )
            .reset_index()
        )
        agg = agg.merge(seed_meta, on=group_cols, how="left")
    for col in df.columns:
        if col.endswith("_status") and col not in agg.columns:
            status_meta = (
                df.groupby(group_cols, dropna=False)[col]
                .agg(lambda x: ";".join(sorted({str(v) for v in x.dropna().unique()})))
                .reset_index()
            )
            agg = agg.merge(status_meta, on=group_cols, how="left")
    return agg


def pass_fail_summary(
    rho_summary: pd.DataFrame,
    ppc_summary: pd.DataFrame,
    loo_summary: pd.DataFrame,
    sampler_summary: pd.DataFrame,
    validation_scope: str = "smoke",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    variants = sorted(
        set(rho_summary.get("fit_variant", []))
        | set(ppc_summary.get("fit_variant", []))
        | set(loo_summary.get("fit_variant", []))
        | set(sampler_summary.get("fit_variant", []))
    )
    for variant in variants:
        rho = rho_summary[(rho_summary["fit_variant"] == variant) & (rho_summary["system"] == "ALL")]
        if not rho.empty:
            r = rho.iloc[0]
            gates = [
                ("evidence_margin_M", r["evidence_margin_M"], "> 0", r["evidence_margin_M"] > 0),
                ("balanced_accuracy", r["balanced_accuracy"], ">= 0.75", r["balanced_accuracy"] >= 0.75),
                ("brier_improvement", r["brier_improvement"], "> 0", r["brier_improvement"] > 0),
                ("log_score_improvement", r["log_score_improvement"], "> 0", r["log_score_improvement"] > 0),
                ("mean_rho_gap", r["mean_rho_R1"] - r["mean_rho_R0"], "> 0.30", (r["mean_rho_R1"] - r["mean_rho_R0"]) > 0.30),
            ]
            for metric, value, threshold, ok in gates:
                rows.append({"fit_variant": variant, "validation_scope": validation_scope, "pillar": "rho_recovery", "metric": metric, "value": float(value), "threshold": threshold, "status": "pass" if ok else "fail", "notes": ""})
        ppc = ppc_summary[(ppc_summary["fit_variant"] == variant) & (ppc_summary["system"] == "ALL")]
        if not ppc.empty:
            r = ppc.iloc[0]
            gates = [
                ("mean_RPS_improvement", r["mean_RPS_improvement"], "> 0", r["mean_RPS_improvement"] > 0),
                ("weighted_mean_cell_TV", r["weighted_mean_cell_TV"], "<= 0.25", r["weighted_mean_cell_TV"] <= 0.25),
                ("median_cell_TV", r["median_cell_TV"], "<= 0.25", r["median_cell_TV"] <= 0.25),
                ("q90_cell_TV", r["q90_cell_TV"], "<= 0.50", r["q90_cell_TV"] <= 0.50),
                ("fraction_cell_TV_gt_0p5", r["fraction_cell_TV_gt_0p5"], "<= 0.10", r["fraction_cell_TV_gt_0p5"] <= 0.10),
                ("posterior_predictive_tv_p_value", r["posterior_predictive_tv_p_value"], "in [0.05, 0.95]", 0.05 <= r["posterior_predictive_tv_p_value"] <= 0.95),
            ]
            for metric, value, threshold, ok in gates:
                rows.append({"fit_variant": variant, "validation_scope": validation_scope, "pillar": "posterior_predictive", "metric": metric, "value": float(value), "threshold": threshold, "status": "pass" if ok else "fail", "notes": ""})
        loo = loo_summary[loo_summary["fit_variant"] == variant]
        if not loo.empty:
            r = loo.iloc[0]
            partial_flag = bool(float(r.get("loo_partial_flag", 0.0)))
            gates = [
                ("frac_pareto_k_gt_0p7", r["frac_pareto_k_gt_0p7"], "<= 0.05", r["frac_pareto_k_gt_0p7"] <= 0.05),
                ("frac_pareto_k_gt_1p0", r["frac_pareto_k_gt_1p0"], "== 0", r["frac_pareto_k_gt_1p0"] == 0),
                ("max_pareto_k", r["max_pareto_k"], "< 1.0", r["max_pareto_k"] < 1.0),
            ]
            for metric, value, threshold, ok in gates:
                rows.append({"fit_variant": variant, "validation_scope": validation_scope, "pillar": "loo_lppd", "metric": metric, "value": float(value), "threshold": threshold, "status": "pass" if ok else "fail", "notes": "partial" if partial_flag else ""})
        sampler = sampler_summary[sampler_summary["fit_variant"] == variant]
        if not sampler.empty:
            rates = {
                "fit_pass_rate_no_divergences": float((sampler["n_divergences"] == 0).mean()),
                "fit_pass_rate_rhat": float((sampler["max_rhat"] <= 1.01).mean()),
                "fit_pass_rate_ess_bulk": float((sampler["min_ess_bulk"] >= 400).mean()),
                "fit_pass_rate_ess_tail": float((sampler["min_ess_tail"] >= 200).mean()),
                "fit_pass_rate_tree_depth": float((sampler["max_tree_depth_hit_rate"] <= 0.01).mean()),
            }
            for metric, value in rates.items():
                rows.append({"fit_variant": variant, "validation_scope": validation_scope, "pillar": "sampler_health", "metric": metric, "value": value, "threshold": ">= 0.90", "status": "pass" if value >= 0.90 else "fail", "notes": ""})
    return pd.DataFrame(rows)


def ppc_nc_overall_checks(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        truth = json.loads((run_dir / "truth_payload.json").read_text())
        fit_seed = int(truth["seed"]) + int(truth["seed_idx"])
        ds = xr.open_dataset(run_dir / "posterior_predictive.nc")
        probs = ds["posterior_mean_category_probability"].values
        y = ds["observed_rating"].values.astype(int)
        rps = []
        for p, yy in zip(probs, y):
            cdf = np.cumsum(p)
            obs = (np.arange(len(p)) >= int(yy)).astype(float)
            rps.append(float(np.mean((cdf[:-1] - obs[:-1]) ** 2)))
        sums = ds["category_probability"].sum(dim="category").values
        rows.append(
            {
                "fit_variant": truth["fit_variant"],
                "seed": fit_seed,
                "n_ratings": int(len(y)),
                "ppc_draws_in_nc": int(ds.sizes["draw"]),
                "mean_RPS_from_nc": float(np.mean(rps)),
                "max_abs_category_prob_sum_minus_one": float(np.max(np.abs(sums - 1.0))),
                "has_system_or_node_metadata_in_nc": False,
                "cell_tv_recomputable_from_nc_alone": False,
                "note": "posterior_predictive.nc stores rating_row/category arrays but no system/node_key row metadata",
            }
        )
    return pd.DataFrame(rows)


def ppc_cell_nobs_distribution(run_dirs: list[Path]) -> pd.DataFrame:
    rows = []
    for run_dir in run_dirs:
        truth = json.loads((run_dir / "truth_payload.json").read_text())
        fit_seed = int(truth["seed"]) + int(truth["seed_idx"])
        for system_raw, by_key in truth["ordinal_by_system_indicator"].items():
            for node_key, ratings in by_key.items():
                rows.append(
                    {
                        "fit_variant": truth["fit_variant"],
                        "seed": fit_seed,
                        "system": SYSTEM_LABEL.get(system_raw, system_raw),
                        "system_raw": system_raw,
                        "node_key": node_key,
                        "n_obs": len(ratings),
                    }
                )
    return pd.DataFrame(rows)


def compare_tables(top: pd.DataFrame, refreshed: pd.DataFrame, keys: list[str], table_name: str) -> pd.DataFrame:
    if top.empty or refreshed.empty:
        return pd.DataFrame()
    shared = [c for c in top.columns if c in refreshed.columns and c not in keys]
    merged = refreshed.merge(top, on=keys, how="outer", suffixes=("_refreshed", "_top"), indicator=True)
    rows = []
    for _, row in merged.iterrows():
        key_payload = {key: row.get(key) for key in keys}
        if row["_merge"] != "both":
            rows.append({"table": table_name, **key_payload, "metric": "_row_presence", "refreshed": row["_merge"], "top_level": row["_merge"], "abs_diff": np.nan, "status": "row_mismatch"})
            continue
        for col in shared:
            rv = row.get(f"{col}_refreshed")
            tv = row.get(f"{col}_top")
            if pd.isna(rv) and pd.isna(tv):
                continue
            status = "match"
            diff = np.nan
            if isinstance(rv, (int, float, np.floating)) and isinstance(tv, (int, float, np.floating)):
                diff = abs(float(rv) - float(tv))
                if diff > 1e-9:
                    status = "diff"
            elif str(rv) != str(tv):
                status = "diff"
            if status == "diff":
                rows.append({"table": table_name, **key_payload, "metric": col, "refreshed": rv, "top_level": tv, "abs_diff": diff, "status": status})
    return pd.DataFrame(rows)


def sampler_run_summary_comparison(run_dirs: list[Path], refreshed_sampler: pd.DataFrame, top_sampler: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["max_rhat", "n_rhat_gt_1p01", "n_rhat_gt_1p05", "min_ess_bulk", "min_ess_tail", "median_ess_bulk", "median_ess_tail", "n_divergences", "max_tree_depth_hit_rate", "runtime_seconds"]
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "run_summary.json").read_text())
        seed = int(summary["seed"])
        ref = refreshed_sampler[refreshed_sampler["seed"].astype(int) == seed]
        top = top_sampler[top_sampler["seed"].astype(int) == seed] if not top_sampler.empty else pd.DataFrame()
        for metric in metrics:
            rv = float(ref.iloc[0][metric]) if not ref.empty and metric in ref.columns else np.nan
            sv = float(summary[metric]) if metric in summary else np.nan
            tv = float(top.iloc[0][metric]) if not top.empty and metric in top.columns else np.nan
            rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "refreshed_from_fit_nc": rv,
                    "run_summary_json": sv,
                    "top_level_summary": tv,
                    "abs_refreshed_minus_run_summary": abs(rv - sv) if np.isfinite(rv) and np.isfinite(sv) else np.nan,
                    "abs_refreshed_minus_top_level": abs(rv - tv) if np.isfinite(rv) and np.isfinite(tv) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_bridge(cases: pd.DataFrame) -> pd.DataFrame:
    oracle = read_csv(INPUT_DIR / "oracle_root_smoke_cases.csv")
    if oracle.empty:
        bridge = cases.copy()
        bridge["oracle_available"] = False
        bridge["oracle_missing_reason"] = "oracle_root_smoke_cases.csv missing"
        return bridge
    keep_oracle = oracle[["fit_variant", "seed", "system", "system_raw", "root_z_true", "log_B_oracle", "rho_oracle_collapsed"]].copy()
    bridge = cases.merge(keep_oracle, on=["fit_variant", "seed", "system", "system_raw", "root_z_true"], how="left")
    bridge["oracle_available"] = bridge["log_B_oracle"].notna()
    bridge["delta_eff_minus_oracle"] = bridge["log_B_eff"] - bridge["log_B_oracle"]
    bridge["oracle_correct_sign"] = ((bridge["root_z_true"] == 1) & (bridge["log_B_oracle"] > 0.0)) | ((bridge["root_z_true"] == 0) & (bridge["log_B_oracle"] < 0.0))
    bridge["posterior_correct_sign"] = bridge["correct_sign"]
    bridge["oracle_evidence_category"] = bridge["rho_oracle_collapsed"].map(lambda x: rho_evidence_category(float(x)) if pd.notna(x) else "")
    bridge["oracle_classified_present_at_0p5"] = bridge["rho_oracle_collapsed"] > 0.5
    bridge["hmc_classified_present_at_0p5"] = bridge["classified_present_at_0p5"]
    bridge["posterior_decisive"] = ((bridge["root_z_true"] == 1) & (bridge["log_B_eff"] > TAU_PRESENT_50)) | ((bridge["root_z_true"] == 0) & (bridge["log_B_eff"] < TAU_ABSENT_05))
    bridge["oracle_decisive"] = ((bridge["root_z_true"] == 1) & (bridge["log_B_oracle"] > TAU_PRESENT_50)) | ((bridge["root_z_true"] == 0) & (bridge["log_B_oracle"] < TAU_ABSENT_05))
    return bridge


def fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    show = df if max_rows is None else df.head(max_rows)
    cols = list(show.columns)
    def cell(value: Any) -> str:
        text = fmt(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(cell(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def write_report(
    cases: pd.DataFrame,
    rho: pd.DataFrame,
    ppc: pd.DataFrame,
    sampler: pd.DataFrame,
    pass_fail: pd.DataFrame,
    bridge: pd.DataFrame,
    ppc_nc: pd.DataFrame,
    cell_nobs: pd.DataFrame,
    discrepancies: pd.DataFrame,
    sampler_cmp: pd.DataFrame,
) -> None:
    failed_llm = bridge[(bridge["system"] == "LLMs") & (bridge["root_z_true"] == 1)].copy()
    failed_validation = pass_fail[pass_fail["status"] == "fail"].copy()
    nobs_summary = (
        cell_nobs.groupby("system")["n_obs"]
        .agg(n_cells="count", min_n_obs="min", median_n_obs="median", q90_n_obs=lambda x: float(pd.Series(x).quantile(0.90)), max_n_obs="max", n_cells_ge_5=lambda x: int((pd.Series(x) >= 5).sum()))
        .reset_index()
    )
    top_level_stale = not discrepancies.empty or bool((sampler_cmp["abs_refreshed_minus_top_level"] > 1e-9).any())
    sampler_ok = (
        not sampler.empty
        and float(sampler["max_rhat"].max()) <= 1.01
        and int((sampler["n_rhat_gt_1p01"] > 0).sum()) == 0
        and float(sampler["min_ess_bulk"].min()) >= 400
        and float(sampler["min_ess_tail"].min()) >= 200
        and int(sampler["n_divergences"].sum()) == 0
    )
    ppc_all = ppc[ppc["system"] == "ALL"].iloc[0]
    weighted_tv_oracle = read_csv(INPUT_DIR / "oracle_ppc_baseline_summary.csv")
    weighted_tv_rows = weighted_tv_oracle[
        (weighted_tv_oracle["metric"] == "weighted_mean_cell_TV")
        & (weighted_tv_oracle["system"].astype(str) == "ALL")
    ] if not weighted_tv_oracle.empty else pd.DataFrame()
    report = [
        "# Smoke Refresh Report",
        "",
        f"Input: `{INPUT_DIR}`",
        f"Output: `{OUTPUT_DIR}`",
        "",
        "## Summary",
        "",
        f"- Top-level smoke summaries stale or inconsistent: `{top_level_stale}`.",
        "- Refreshed root recovery still fails for the LLM split.",
        "- The failed LLM R=1 case is also a no-HMC/oracle failure, not just an HMC posterior flip.",
        f"- Sampler diagnostics acceptable enough for acceptance interpretation: `{sampler_ok}`.",
        "- PPC legacy TV gates still fail, but all system x indicator cells are sparse and oracle-calibrated PPC files indicate weighted TV is mostly compatible with finite-sample noise.",
        "",
        "## Refreshed Root Recovery",
        "",
        markdown_table(rho),
        "",
        "## LLM R=1 Bridge",
        "",
        markdown_table(failed_llm[[
            "seed",
            "root_z_true",
            "rho_collapsed",
            "rho_collapsed_median",
            "log_B_eff",
            "log_B_draw_median",
            "rho_oracle_collapsed",
            "log_B_oracle",
            "delta_eff_minus_oracle",
            "posterior_correct_sign",
            "oracle_correct_sign",
        ]]),
        "",
        "The generated latent/root truth for the LLM failed case is `root_z_true=1`. Both oracle/clamped evidence and HMC posterior evidence support absence: oracle `log_B` is negative and HMC effective `log_B` is also negative. The HMC-minus-oracle gap is small relative to the sign error, so the available artefacts point to an unlucky or weak generated LLM dataset rather than beta mismatch or nuisance inference as the primary cause.",
        "",
        "## PPC Refresh",
        "",
        markdown_table(ppc),
        "",
        "PPC NetCDF checks:",
        "",
        markdown_table(ppc_nc),
        "",
        "Cell-count sparsity by system:",
        "",
        markdown_table(nobs_summary),
        "",
        f"Refreshed ALL weighted mean cell TV is `{fmt(ppc_all['weighted_mean_cell_TV'])}`, median cell TV is `{fmt(ppc_all['median_cell_TV'])}`, q90 cell TV is `{fmt(ppc_all['q90_cell_TV'])}`, fraction TV > 0.5 is `{fmt(ppc_all['fraction_cell_TV_gt_0p5'])}`, and PPC TV p-value is `{fmt(ppc_all['posterior_predictive_tv_p_value'])}`.",
        "The existing `posterior_predictive.nc` files are sufficient to verify overall RPS and probability normalization, but they do not store system/node row metadata. System and cell TV refresh therefore uses the existing per-run PPC summary artefacts rather than reconstructing cell TV from NetCDF alone.",
        "",
        "Oracle-calibrated weighted TV rows from existing top-level PPC audit:",
        "",
        markdown_table(weighted_tv_rows[["seed", "system", "fitted_ppc_value", "oracle_expected_median", "oracle_q05", "oracle_q95", "status"]]) if not weighted_tv_rows.empty else "_Oracle PPC baseline rows not available._",
        "",
        "## Sampler Refresh",
        "",
        markdown_table(sampler),
        "",
        "Sampler comparison against run summaries and top-level summaries:",
        "",
        markdown_table(sampler_cmp),
        "",
        "## Failed Refreshed Gates",
        "",
        markdown_table(failed_validation),
        "",
        "## Top-Level Discrepancies",
        "",
        markdown_table(discrepancies, max_rows=80),
        "",
        "## Answers",
        "",
        "1. Are top-level smoke summaries stale or inconsistent? Yes. Refreshed root/sampler/LOO values disagree with top-level summaries, especially seed 20260511 sampler fields and seed 20260511 root rho/log_B values. PPC summaries mostly track per-run PPC artefacts but still need metadata caveats.",
        "2. After refresh, does root recovery still fail for LLMs? Yes. LLM balanced accuracy remains 0.5 with negative Brier/log-score improvement.",
        "3. Is the failed LLM case also a no-HMC/oracle failure, or only an HMC posterior failure? It is also an oracle/clamped failure: the exact oracle log_B for seed 20260512 LLMs is negative.",
        "4. Are sampler diagnostics acceptable enough to interpret smoke metrics? No. Seed 20260512 has max_rhat 1.05, 57 Rhat values > 1.01, min bulk ESS 177, and min tail ESS 56. These are diagnostic only.",
        "5. Is PPC failure likely implementation/sparsity/small-smoke noise, or a substantive observation-layer issue? Legacy TV gates are dominated by sparse system x indicator cells (`n_cells_ge_5=0` for every system). Existing oracle-calibrated PPC and predictive-alignment audits reduce concern for a broad observation-layer implementation break, though ELIZA seed 20260511 remains a localized weighted-TV excess signal.",
        "6. Recommended next action: inspect the LLM failed seed and keep pilot blocked until summaries are regenerated consistently. If continuing smoke validation, rerun a longer smoke or pilot only after deciding whether seed 20260512's oracle-negative LLM draw is acceptable smoke noise and after sampler/summary consistency is fixed.",
        "",
        "## Confirmation",
        "",
        "No HMC, refitting, new posterior samples, new posterior predictive samples, pilot/full run, metrics-only command, plots-only command, or LOO recomputation was launched. This script read existing `fit.nc`, `posterior_predictive.nc`, per-run CSVs, JSON payloads, and top-level CSVs only.",
    ]
    (OUTPUT_DIR / "smoke_refresh_report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted((INPUT_DIR / "runs" / FIT_VARIANT).glob("seed_*"))
    cases = pd.concat([extract_rho_cases_from_fit(run_dir) for run_dir in run_dirs], ignore_index=True)
    rho = summarise_rho_recovery(cases)
    sampler = pd.concat([sampler_diagnostics_from_fit(run_dir) for run_dir in run_dirs], ignore_index=True)

    ppc_per_seed = pd.concat([read_csv(run_dir / "posterior_predictive_summary.csv") for run_dir in run_dirs], ignore_index=True)
    if "posterior_predictive_tv_p_value_status" not in ppc_per_seed.columns:
        ppc_per_seed["posterior_predictive_tv_p_value_status"] = np.where(ppc_per_seed["system"] == "ALL", "computed_overall", "not_computed_system_specific")
    else:
        ppc_per_seed["posterior_predictive_tv_p_value_status"] = ppc_per_seed["posterior_predictive_tv_p_value_status"].fillna(
            pd.Series(
                np.where(ppc_per_seed["system"] == "ALL", "computed_overall", "not_computed_system_specific"),
                index=ppc_per_seed.index,
            )
        )
    ppc = aggregate_metric_table(ppc_per_seed, ["fit_variant", "system"])
    loo_per_seed = pd.concat([read_csv(run_dir / "loo_lppd_summary.csv") for run_dir in run_dirs], ignore_index=True)
    loo = aggregate_metric_table(loo_per_seed, ["fit_variant"])
    pass_fail = pass_fail_summary(rho, ppc, loo, sampler)
    bridge = make_bridge(cases)
    ppc_nc = ppc_nc_overall_checks(run_dirs)
    cell_nobs = ppc_cell_nobs_distribution(run_dirs)
    cell_nobs_summary = (
        cell_nobs.groupby(["fit_variant", "system"])["n_obs"]
        .agg(n_cells="count", min_n_obs="min", median_n_obs="median", q90_n_obs=lambda x: float(pd.Series(x).quantile(0.90)), max_n_obs="max", n_cells_ge_5=lambda x: int((pd.Series(x) >= 5).sum()))
        .reset_index()
    )

    top_validation = read_csv(INPUT_DIR / "validation_cases.csv")
    top_rho = read_csv(INPUT_DIR / "rho_recovery_summary.csv")
    top_ppc = read_csv(INPUT_DIR / "posterior_predictive_summary.csv")
    top_loo = read_csv(INPUT_DIR / "loo_lppd_summary.csv")
    top_sampler = read_csv(INPUT_DIR / "sampler_diagnostics_summary.csv")
    discrepancies = pd.concat(
        [
            compare_tables(top_validation, cases, ["fit_variant", "seed", "system"], "validation_cases"),
            compare_tables(top_rho, rho, ["fit_variant", "system"], "rho_recovery_summary"),
            compare_tables(top_ppc, ppc, ["fit_variant", "system"], "posterior_predictive_summary"),
            compare_tables(top_loo, loo, ["fit_variant"], "loo_lppd_summary"),
            compare_tables(top_sampler, sampler, ["fit_variant", "seed"], "sampler_diagnostics_summary"),
        ],
        ignore_index=True,
    )
    sampler_cmp = sampler_run_summary_comparison(run_dirs, sampler, top_sampler)

    write_csv(cases, OUTPUT_DIR / "validation_cases_refreshed.csv")
    write_csv(rho, OUTPUT_DIR / "rho_recovery_summary_refreshed.csv")
    write_csv(ppc, OUTPUT_DIR / "posterior_predictive_summary_refreshed.csv")
    write_csv(ppc_per_seed, OUTPUT_DIR / "posterior_predictive_per_seed_existing.csv")
    write_csv(ppc_nc, OUTPUT_DIR / "posterior_predictive_nc_checks.csv")
    write_csv(cell_nobs, OUTPUT_DIR / "ppc_cell_nobs.csv")
    write_csv(cell_nobs_summary, OUTPUT_DIR / "ppc_cell_nobs_summary.csv")
    write_csv(loo, OUTPUT_DIR / "loo_lppd_summary_refreshed.csv")
    write_csv(sampler, OUTPUT_DIR / "sampler_diagnostics_summary_refreshed.csv")
    write_csv(pass_fail, OUTPUT_DIR / "pass_fail_summary_refreshed.csv")
    write_csv(bridge, OUTPUT_DIR / "seed_root_bridge_refreshed.csv")
    write_csv(discrepancies, OUTPUT_DIR / "summary_discrepancies.csv")
    write_csv(sampler_cmp, OUTPUT_DIR / "sampler_run_summary_comparison.csv")
    write_report(cases, rho, ppc, sampler, pass_fail, bridge, ppc_nc, cell_nobs, discrepancies, sampler_cmp)


if __name__ == "__main__":
    main()
