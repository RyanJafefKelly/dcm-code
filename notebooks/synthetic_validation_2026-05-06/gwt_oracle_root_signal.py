"""Oracle root-signal audit for the GWT exact tree.

This audit asks whether the current GWT rating design contains enough
observed-scale signal to distinguish different root consciousness values C_s,
when tree and ordinal nuisance parameters are fixed to known truth.

The main output is a deterministic marginal-rating KL curve:

    KL_marginal(C_ref || C_alt)

computed by summing ordinal category KL contributions over the current rating
slots.  It is intentionally labelled as marginal because the exact tree also
induces sibling dependence; that dependence is audited separately.  This check
is the fast, transparent root-signal screen.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcm_model import ModelConfig, load_data, node_key  # noqa: E402
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS  # noqa: E402
from gwt_oracle_internal_identifiability import (  # noqa: E402
    EXACT_PROD_PATH,
    LABELS as BASE_LABELS,
    STANCE,
    TRUE_C_BY_SYSTEM,
    is_missing,
    load_oracle_truth,
    ordered_probit_probs,
    write_json,
)


DEFAULT_RUNS_DIR = (
    REPO_ROOT / "notebooks/synthetic_validation_2026-05-06/runs"
    / "oracle_root_signal"
)

LABELS = {
    **BASE_LABELS,
    "fit": "oracle",
}

KL_THRESHOLDS = (1.0, 2.0, 5.0, 10.0)


def iter_indicators(
    stance_data: Dict[str, Any],
) -> Iterable[Tuple[Dict[str, Any], str, Tuple[str, ...], str]]:
    root_path = (stance_data["name"],)

    def walk(
        node: Dict[str, Any],
        ancestor_path: Tuple[str, ...],
        top_feature: str,
    ):
        current_path = ancestor_path + (node["name"],)
        if (node.get("type") or "").lower() == "indicator":
            yield node, node_key(ancestor_path, node["name"]), ancestor_path, top_feature
            return
        for child in node.get("evidencers", []):
            yield from walk(child, current_path, top_feature)

    for child in stance_data.get("evidencers", []):
        yield from walk(child, root_path, child["name"])


def rating_count_for_system(node: Dict[str, Any], system_name: str) -> int:
    obs = node.get("observations", {}).get(system_name)
    if not obs:
        return 0
    return int(sum(not is_missing(v) for v in obs.get("values", [])))


def indicator_design_table(
    stance_data: Dict[str, Any],
    systems: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for node, key, _, top_feature in iter_indicators(stance_data):
        for system_name in systems:
            n = rating_count_for_system(node, system_name)
            if n <= 0:
                continue
            rows.append(
                {
                    "system": system_name,
                    "node_key": key,
                    "indicator": node["name"],
                    "top_feature": top_feature,
                    "rating_count": n,
                }
            )
    return pd.DataFrame(rows)


def parent_state_probabilities_by_indicator(
    stance_data: Dict[str, Any],
    edge_betas: Mapping[str, Mapping[str, float]],
    c_value: float,
) -> Dict[str, float]:
    """Return P(parent z = 1 | C) for each indicator edge."""
    out: Dict[str, float] = {}
    root_path = (stance_data["name"],)

    def walk(
        node: Dict[str, Any],
        ancestor_path: Tuple[str, ...],
        parent_q: float,
    ) -> None:
        key = node_key(ancestor_path, node["name"])
        ntype = (node.get("type") or "").lower()
        if ntype == "indicator":
            out[key] = float(parent_q)
            return

        beta_pres = float(edge_betas[key]["beta_pres"])
        beta_abs = float(edge_betas[key]["beta_abs"])
        q_self = beta_abs + parent_q * (beta_pres - beta_abs)
        current_path = ancestor_path + (node["name"],)
        for child in node.get("evidencers", []):
            walk(child, current_path, q_self)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path, float(c_value))
    return out


def indicator_category_probs(
    parent_q: float,
    beta_pres: float,
    beta_abs: float,
    obs_params: Mapping[str, Any],
) -> np.ndarray:
    """Marginal ordinal category probabilities for one indicator."""
    a = float(obs_params["a"])
    kappa = np.asarray(obs_params["kappa"], dtype=float)
    p_m_given_parent0 = binom.pmf(np.arange(3), 2, beta_abs)
    p_m_given_parent1 = binom.pmf(np.arange(3), 2, beta_pres)
    p_m = (1.0 - parent_q) * p_m_given_parent0 + parent_q * p_m_given_parent1

    out = np.zeros(len(kappa) + 1)
    for m in (0, 1, 2):
        out += p_m[m] * ordered_probit_probs(kappa, a * (m / 2.0))
    out = np.clip(out, 1e-12, 1.0)
    return out / out.sum()


def category_probabilities_for_c(
    stance_data: Dict[str, Any],
    edge_betas: Mapping[str, Mapping[str, float]],
    obs_params: Mapping[str, Any],
    c_value: float,
) -> Dict[str, np.ndarray]:
    parent_probs = parent_state_probabilities_by_indicator(
        stance_data,
        edge_betas,
        c_value,
    )
    out: Dict[str, np.ndarray] = {}
    for _, key, _, _ in iter_indicators(stance_data):
        edge = edge_betas[key]
        out[key] = indicator_category_probs(
            parent_probs[key],
            float(edge["beta_pres"]),
            float(edge["beta_abs"]),
            obs_params,
        )
    return out


def categorical_kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def c_grid(grid_size: int) -> np.ndarray:
    base = np.linspace(0.001, 0.999, grid_size)
    vals = np.concatenate([base, np.asarray(list(TRUE_C_BY_SYSTEM.values()))])
    vals = np.unique(np.round(vals, 6))
    vals.sort()
    return vals


def expected_rating_distribution_rows(
    design: pd.DataFrame,
    probs_by_c: Mapping[float, Mapping[str, np.ndarray]],
    grid: Sequence[float],
    labels: Mapping[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    system_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []

    for c_value in grid:
        probs = probs_by_c[float(c_value)]
        for system_name, sys_df in design.groupby("system"):
            total_n = int(sys_df["rating_count"].sum())
            expected = np.zeros(7)
            for _, row in sys_df.iterrows():
                expected += int(row["rating_count"]) * probs[row["node_key"]]
            for category, value in enumerate(expected):
                system_rows.append(
                    {
                        **labels,
                        "audit": "root_signal",
                        "scope": "system",
                        "system": system_name,
                        "C": float(c_value),
                        "category": category,
                        "expected_count": float(value),
                        "expected_proportion": float(value / total_n),
                        "rating_count": total_n,
                    }
                )

            for top_feature, feat_df in sys_df.groupby("top_feature"):
                feature_n = int(feat_df["rating_count"].sum())
                feature_expected = np.zeros(7)
                for _, row in feat_df.iterrows():
                    feature_expected += int(row["rating_count"]) * probs[row["node_key"]]
                for category, value in enumerate(feature_expected):
                    feature_rows.append(
                        {
                            **labels,
                            "audit": "root_signal",
                            "scope": "top_feature",
                            "system": system_name,
                            "top_feature": top_feature,
                            "C": float(c_value),
                            "category": category,
                            "expected_count": float(value),
                            "expected_proportion": float(value / feature_n),
                            "rating_count": feature_n,
                        }
                    )
    return pd.DataFrame(system_rows), pd.DataFrame(feature_rows)


def kl_rows(
    design: pd.DataFrame,
    probs_by_c: Mapping[float, Mapping[str, np.ndarray]],
    grid: Sequence[float],
    labels: Mapping[str, str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    system_rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, Any]] = []

    for system_name, sys_df in design.groupby("system"):
        ref_c = float(TRUE_C_BY_SYSTEM[system_name])
        ref_probs = probs_by_c[ref_c]
        total_n = int(sys_df["rating_count"].sum())
        for alt_c in grid:
            kl_total = 0.0
            by_feature: Dict[str, float] = {}
            by_feature_n: Dict[str, int] = {}
            alt_probs = probs_by_c[float(alt_c)]
            for _, row in sys_df.iterrows():
                n = int(row["rating_count"])
                contribution = n * categorical_kl(
                    ref_probs[row["node_key"]],
                    alt_probs[row["node_key"]],
                )
                kl_total += contribution
                top_feature = str(row["top_feature"])
                by_feature[top_feature] = by_feature.get(top_feature, 0.0) + contribution
                by_feature_n[top_feature] = by_feature_n.get(top_feature, 0) + n

            system_rows.append(
                {
                    **labels,
                    "audit": "root_signal",
                    "scope": "system",
                    "system": system_name,
                    "reference_C": ref_c,
                    "comparison_C": float(alt_c),
                    "abs_delta_C": float(abs(float(alt_c) - ref_c)),
                    "kl_marginal_nats": float(kl_total),
                    "kl_per_rating": float(kl_total / total_n),
                    "rating_count": total_n,
                }
            )
            for top_feature, contribution in by_feature.items():
                n_feature = by_feature_n[top_feature]
                feature_rows.append(
                    {
                        **labels,
                        "audit": "root_signal",
                        "scope": "top_feature",
                        "system": system_name,
                        "top_feature": top_feature,
                        "reference_C": ref_c,
                        "comparison_C": float(alt_c),
                        "abs_delta_C": float(abs(float(alt_c) - ref_c)),
                        "kl_marginal_nats": float(contribution),
                        "kl_per_rating": float(contribution / n_feature),
                        "rating_count": n_feature,
                    }
                )
    return pd.DataFrame(system_rows), pd.DataFrame(feature_rows)


def truth_pairwise_kl(kl_by_system: pd.DataFrame) -> pd.DataFrame:
    truth_values = sorted(set(float(v) for v in TRUE_C_BY_SYSTEM.values()))
    rows = []
    for _, row in kl_by_system.iterrows():
        if any(abs(float(row["comparison_C"]) - v) < 1e-8 for v in truth_values):
            rows.append(row)
    return pd.DataFrame(rows)


def threshold_crossings(kl_by_system: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for system_name, sys_df in kl_by_system.groupby("system"):
        ref_c = float(sys_df["reference_C"].iloc[0])
        for threshold in KL_THRESHOLDS:
            reached = sys_df[sys_df["kl_marginal_nats"] >= threshold].copy()
            if reached.empty:
                rows.append(
                    {
                        "system": system_name,
                        "reference_C": ref_c,
                        "kl_threshold": threshold,
                        "min_abs_delta_C": math.nan,
                        "nearest_C_at_threshold": math.nan,
                        "reached": False,
                    }
                )
                continue
            nearest = reached.sort_values("abs_delta_C").iloc[0]
            rows.append(
                {
                    "system": system_name,
                    "reference_C": ref_c,
                    "kl_threshold": threshold,
                    "min_abs_delta_C": float(nearest["abs_delta_C"]),
                    "nearest_C_at_threshold": float(nearest["comparison_C"]),
                    "reached": True,
                }
            )
    return pd.DataFrame(rows)


def fmt_float(x: Any, digits: int = 3) -> str:
    if pd.isna(x):
        return "not reached"
    return f"{float(x):.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("|" + "|".join(["---"] * len(columns)) + "|")
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(fmt_float(val))
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return rows


def write_summary(
    out_dir: Path,
    labels: Mapping[str, str],
    grid: Sequence[float],
    design: pd.DataFrame,
    kl_by_system: pd.DataFrame,
    pairwise: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> None:
    rating_counts = (
        design.groupby("system")["rating_count"].sum().reset_index(name="rating_count")
    )
    llm = "2024 Leading Chat LLMs"
    llm_vs_chicken = pairwise[
        (pairwise["system"] == llm)
        & (np.isclose(pairwise["reference_C"], TRUE_C_BY_SYSTEM[llm]))
        & (np.isclose(pairwise["comparison_C"], TRUE_C_BY_SYSTEM["Chicken"]))
    ]
    llm_chicken_kl = (
        float(llm_vs_chicken["kl_marginal_nats"].iloc[0])
        if not llm_vs_chicken.empty
        else math.nan
    )
    chicken_vs_llm = pairwise[
        (pairwise["system"] == "Chicken")
        & (np.isclose(pairwise["reference_C"], TRUE_C_BY_SYSTEM["Chicken"]))
        & (np.isclose(pairwise["comparison_C"], TRUE_C_BY_SYSTEM[llm]))
    ]
    chicken_llm_kl = (
        float(chicken_vs_llm["kl_marginal_nats"].iloc[0])
        if not chicken_vs_llm.empty
        else math.nan
    )

    threshold_display = thresholds.copy()
    threshold_display["reached"] = threshold_display["reached"].map({True: "yes", False: "no"})

    pairwise_display = pairwise[
        pairwise["comparison_C"].isin(sorted(set(TRUE_C_BY_SYSTEM.values())))
    ][
        [
            "system",
            "reference_C",
            "comparison_C",
            "kl_marginal_nats",
            "kl_per_rating",
        ]
    ].sort_values(["system", "comparison_C"])

    lines: List[str] = [
        "# Oracle Root-Signal Summary",
        "",
        "Labels:",
        "",
        f"- DGP: `{labels['dgp']}`",
        f"- Fit: `{labels['fit']}`",
        f"- Leaf: `{labels['leaf']}`",
        f"- Nuisance truth: `{labels['nuisance_truth']}`",
        f"- Design: `{labels['design']}`",
        "",
        f"C grid points: {len(grid)}",
        "",
        "## Plain-English Interpretation",
        "",
        "This audit asks whether the current ordinal ratings would visibly change "
        "if the root consciousness value `C_s` changed, assuming the exact tree "
        "and nuisance parameters are correct.",
        "",
        "The output is a root-signal screen, not a full posterior fit. It fixes "
        "the tree and observation layer, computes the expected category "
        "distribution for each indicator at each candidate `C`, and then asks "
        "how separated those expected ratings are from the reference `C`.",
        "",
        "The KL numbers are expected log-evidence differences, in nats, from "
        "the marginal ordinal rating distributions. Bigger means the design "
        "should more strongly distinguish the two root values. Near zero means "
        "those root values look similar on the observed rating scale.",
        "",
        f"Under the LLM rating design, the KL separation between `C=0.10` and "
        f"`C=0.25` is {fmt_float(llm_chicken_kl)} nats. Under the Chicken "
        f"rating design, the reverse comparison is {fmt_float(chicken_llm_kl)} "
        "nats. This is the practical check for whether LLM-vs-Chicken root "
        "differences are visible in the current data design before fitting.",
        "",
        "How to use this result: if the KL curve is shallow around a system's "
        "reference `C`, then root posterior concentration in a full model will "
        "need strong help from priors, anchors, or shared nuisance learning. If "
        "the curve is steep, the observed ratings themselves contain real root "
        "signal under the exact-tree semantics.",
        "",
        "Caveat: this is a marginal-rating KL, so it intentionally ignores the "
        "extra sibling dependence induced by shared internal states. The "
        "dependence/block audit is the separate check for that structure.",
        "",
        "## Current Rating Counts",
        "",
        *markdown_table(rating_counts, ["system", "rating_count"]),
        "",
        "## KL Threshold Crossings",
        "",
        "Minimum root-C movement needed to reach each marginal KL threshold.",
        "",
        *markdown_table(
            threshold_display,
            [
                "system",
                "reference_C",
                "kl_threshold",
                "min_abs_delta_C",
                "nearest_C_at_threshold",
                "reached",
            ],
        ),
        "",
        "## Pairwise KL At Reference Truths",
        "",
        *markdown_table(
            pairwise_display,
            [
                "system",
                "reference_C",
                "comparison_C",
                "kl_marginal_nats",
                "kl_per_rating",
            ],
        ),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument(
        "--nuisance-truth",
        choices=["exact_tree_production_medians", "paper_mean_tree_transmission"],
        default="exact_tree_production_medians",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grid_size < 5:
        raise ValueError("--grid-size must be at least 5")

    labels = {**LABELS, "nuisance_truth": args.nuisance_truth}
    run_id = args.run_id or (
        f"{labels['dgp']}__{labels['fit']}__{labels['leaf']}__"
        f"{labels['nuisance_truth']}__{labels['design']}"
    )
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else REPO_ROOT / args.runs_dir
    out_dir = runs_dir / run_id
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Run directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=args.nuisance_truth == "exact_tree_production_medians",
        BETA_ABS_BY_SUPPORT_DEMAND=args.nuisance_truth
        == "exact_tree_production_medians",
    )
    systems = [s for s, _ in ANCHORED_SYSTEM_CONFIGS]
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    truth = load_oracle_truth(args.nuisance_truth, stance_data, cfg)
    design = indicator_design_table(stance_data, systems)
    grid = c_grid(args.grid_size)

    probs_by_c = {
        float(c): category_probabilities_for_c(
            stance_data,
            truth.edge_betas,
            truth.obs_params,
            float(c),
        )
        for c in grid
    }

    expected_system, expected_feature = expected_rating_distribution_rows(
        design,
        probs_by_c,
        grid,
        labels,
    )
    kl_system, kl_feature = kl_rows(design, probs_by_c, grid, labels)
    pairwise = truth_pairwise_kl(kl_system)
    thresholds = threshold_crossings(kl_system)

    expected_system.to_csv(out_dir / "expected_rating_distributions.csv", index=False)
    expected_feature.to_csv(
        out_dir / "feature_expected_rating_distributions.csv",
        index=False,
    )
    kl_system.to_csv(out_dir / "kl_by_system.csv", index=False)
    kl_feature.to_csv(out_dir / "kl_by_feature.csv", index=False)
    pairwise.to_csv(out_dir / "truth_pairwise_kl.csv", index=False)
    thresholds.to_csv(out_dir / "kl_threshold_crossings.csv", index=False)

    config_payload = {
        "labels": labels,
        "audit": "root_signal",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stance": STANCE,
        "systems": systems,
        "true_C_by_system": TRUE_C_BY_SYSTEM,
        "grid_size_requested": args.grid_size,
        "grid_size_actual": int(len(grid)),
        "model_config": asdict(cfg),
        "model_config_note": (
            "Used for data loading and oracle semantic metadata. Edge betas are "
            "fixed from edge_beta_source rather than sampled from this config."
        ),
        "production_fit_source": str(EXACT_PROD_PATH.relative_to(REPO_ROOT)),
        "observation_parameters": truth.obs_params,
        "edge_beta_source": args.nuisance_truth,
        "kl_note": (
            "KL values are deterministic marginal ordinal-rating KL sums over "
            "the current rating slots. They do not include sibling dependence."
        ),
        "outputs": {
            "summary": "summary.md",
            "expected_rating_distributions": "expected_rating_distributions.csv",
            "feature_expected_rating_distributions": (
                "feature_expected_rating_distributions.csv"
            ),
            "kl_by_system": "kl_by_system.csv",
            "kl_by_feature": "kl_by_feature.csv",
            "truth_pairwise_kl": "truth_pairwise_kl.csv",
            "kl_threshold_crossings": "kl_threshold_crossings.csv",
        },
    }
    write_json(out_dir / "config.json", config_payload)
    write_summary(out_dir, labels, grid, design, kl_system, pairwise, thresholds)

    print("=== Oracle root-signal audit ===")
    print(f"labels: {labels}")
    print(f"grid points: {len(grid)}")
    print("rating counts:")
    print(design.groupby("system")["rating_count"].sum().to_string())
    print()
    print("KL threshold crossings:")
    print(thresholds.to_string(index=False))
    print(f"\nwrote {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
