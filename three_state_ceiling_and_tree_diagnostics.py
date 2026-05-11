"""Oracle-component emission-ceiling + tree-propagation diagnostics.

Both diagnostics are NumPy-only summaries of the already-persisted idata.
No new fits.

1. Oracle-component ceiling.
   For each model (binary, three-state), compute at posterior draws:
     P(r = 1 | eta = 0, a, kappa)          -- outer-left component ceiling
     P(r = K | eta = a, a, kappa)          -- outer-right component ceiling
     (three-state also: P(r = 1 | eta = a/2) and P(r = K | eta = a/2))
   Compare to observed P(r=1) at Rater_B × ELIZA (= 0.96) and P(r=K) at
   Rater_B × Human (= 0.86). If the ceiling is below observed, the issue is
   not state assignment -- it is emission sharpness. That would justify
   expert-specific scale or cutpoints.

2. Tree-propagation compression.
   Posterior distribution of indicator-level q_j for Human and ELIZA
   (anchored C = 0.999 and 0.001 respectively). Compute summary stats
   of q_j across the indicator set and, crucially, Delta q_j = q_Human - q_ELIZA
   per indicator. If Delta q_j is small (say < 0.3) even though the root
   anchor differs by 0.998, the tree hierarchy is compressing anchor
   credence before the leaves see the ratings.

Outputs
-------
results/gwt_binary_three_state/analysis/ceiling_diagnostic.md
results/gwt_binary_three_state/analysis/tree_propagation_diagnostic.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import arviz as az
import numpy as np
from scipy.stats import norm

from dcm_model import (
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    MultiSystemModelBuilder,
    load_data,
)
from gwt_three_state_indicator_analysis import (
    FOCUS_KEYS,
    STANCE,
    SYSTEM_CONFIGS_VALIDATED,
    SYSTEM_DISPLAY,
)

RESULTS_DIR = Path("results/gwt_binary_three_state")
ANALYSIS_DIR = RESULTS_DIR / "analysis"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_context(state_model: str):
    config = ModelConfig(
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        INDICATOR_STATE_MODEL=state_model,  # type: ignore[arg-type]
    )
    stance_data = next(item for item in load_data(config) if item["name"] == STANCE)
    processor = MultiSystemDataProcessor(config)
    processor.process(stance_data, [s for s, _ in SYSTEM_CONFIGS_VALIDATED])
    builder = MultiSystemModelBuilder(
        config, EvidenceProcessor(config), processor, list(SYSTEM_CONFIGS_VALIDATED)
    )
    builder.build_model(stance_data)
    return stance_data, builder, processor


def op_probs_draws(kappa_draws: np.ndarray, eta_draws: np.ndarray, K: int) -> np.ndarray:
    """Posterior-draw ordered-probit category probabilities.

    kappa_draws: (S, K-1); eta_draws: (S,); returns (S, K).
    """
    # Broadcast eta against the cutpoints
    cum = norm.cdf(kappa_draws - eta_draws[:, None])  # (S, K-1)
    zeros = np.zeros((cum.shape[0], 1))
    ones = np.ones((cum.shape[0], 1))
    cum_full = np.concatenate([zeros, cum, ones], axis=1)  # (S, K+1)
    return np.diff(cum_full, axis=1)


# ---------------------------------------------------------------------------
# Oracle ceiling
# ---------------------------------------------------------------------------


def ceiling_rows_for_model(
    idata: Any, state_model: str
) -> List[Dict[str, Any]]:
    """Posterior summaries of outer-component emission ceilings."""
    K = 7
    post = idata.posterior
    a_draws = np.asarray(post["a"].values).reshape(-1)
    kappa_draws = np.asarray(post["kappa"].values).reshape(-1, K - 1)

    rows: List[Dict[str, Any]] = []

    # Component 0 (always eta = 0) — P(r = 1) ceiling
    eta0 = np.zeros_like(a_draws)
    probs0 = op_probs_draws(kappa_draws, eta0, K)
    p_r1_z0 = probs0[:, 0]
    rows.append(
        {
            "component": "eta=0 (z=0 / m=0)",
            "category": "P(r=1)",
            "median": float(np.median(p_r1_z0)),
            "p03": float(np.percentile(p_r1_z0, 3)),
            "p97": float(np.percentile(p_r1_z0, 97)),
        }
    )

    # Component top (eta = a) — P(r = K) ceiling
    probsA = op_probs_draws(kappa_draws, a_draws, K)
    p_r7_z1 = probsA[:, -1]
    rows.append(
        {
            "component": "eta=a (z=1 / m=2)",
            "category": "P(r=7)",
            "median": float(np.median(p_r7_z1)),
            "p03": float(np.percentile(p_r7_z1, 3)),
            "p97": float(np.percentile(p_r7_z1, 97)),
        }
    )

    if state_model == "three_state":
        # Middle component (eta = a/2) — P(r=1) and P(r=7)
        eta_mid = a_draws * 0.5
        probs_mid = op_probs_draws(kappa_draws, eta_mid, K)
        for cat_idx, cat_label in [(0, "P(r=1)"), (K - 1, "P(r=7)")]:
            vals = probs_mid[:, cat_idx]
            rows.append(
                {
                    "component": "eta=a/2 (m=1)",
                    "category": cat_label,
                    "median": float(np.median(vals)),
                    "p03": float(np.percentile(vals, 3)),
                    "p97": float(np.percentile(vals, 97)),
                }
            )

    return rows


def write_ceiling_report(
    binary_rows: List[Dict[str, Any]],
    three_rows: List[Dict[str, Any]],
    observed: Dict[str, float],
    path: Path,
) -> None:
    lines = []
    lines.append("# Oracle-component emission-ceiling diagnostic\n")
    lines.append(
        "For each model, this is the maximum probability mass the model can "
        "place on an extreme category *under the best possible latent-state "
        "assignment*. Posterior median and 94% interval are reported over "
        "the joint posterior of (a, kappa).\n"
    )
    lines.append("## Observed extreme-category mass at focus reference cells\n")
    lines.append(f"- Rater_B × Human P(r=7): **{observed['human_right']:.3f}**")
    lines.append(f"- Rater_B × ELIZA P(r=1): **{observed['eliza_left']:.3f}**\n")

    def _section(name: str, rows: List[Dict[str, Any]]):
        lines.append(f"## {name}\n")
        lines.append("| Component | Category | Median | 94% interval |")
        lines.append("|---|---|---:|---|")
        for r in rows:
            lines.append(
                f"| {r['component']} | {r['category']} | {r['median']:.3f} "
                f"| [{r['p03']:.3f}, {r['p97']:.3f}] |"
            )
        lines.append("")

    _section("Binary baseline", binary_rows)
    _section("Three-state", three_rows)

    # Interpretation
    def _ceiling(rows: List[Dict[str, Any]], comp: str, cat: str) -> float:
        for r in rows:
            if r["component"] == comp and r["category"] == cat:
                return r["median"]
        return float("nan")

    b_r1 = _ceiling(binary_rows, "eta=0 (z=0 / m=0)", "P(r=1)")
    b_r7 = _ceiling(binary_rows, "eta=a (z=1 / m=2)", "P(r=7)")
    t_r1 = _ceiling(three_rows, "eta=0 (z=0 / m=0)", "P(r=1)")
    t_r7 = _ceiling(three_rows, "eta=a (z=1 / m=2)", "P(r=7)")

    lines.append("## Interpretation\n")
    lines.append("Comparing ceiling to observed extreme-category mass:\n")
    lines.append(f"- **Rater_B × ELIZA** P(r=1): observed = {observed['eliza_left']:.3f}")
    lines.append(
        f"  - Binary ceiling (eta=0): {b_r1:.3f}  -- "
        f"{'below observed by {:.3f}'.format(observed['eliza_left'] - b_r1) if b_r1 < observed['eliza_left'] else 'at or above observed'}"
    )
    lines.append(
        f"  - Three-state ceiling (eta=0): {t_r1:.3f}  -- "
        f"{'below observed by {:.3f}'.format(observed['eliza_left'] - t_r1) if t_r1 < observed['eliza_left'] else 'at or above observed'}"
    )
    lines.append(f"- **Rater_B × Human** P(r=7): observed = {observed['human_right']:.3f}")
    lines.append(
        f"  - Binary ceiling (eta=a): {b_r7:.3f}  -- "
        f"{'below observed by {:.3f}'.format(observed['human_right'] - b_r7) if b_r7 < observed['human_right'] else 'at or above observed'}"
    )
    lines.append(
        f"  - Three-state ceiling (eta=a): {t_r7:.3f}  -- "
        f"{'below observed by {:.3f}'.format(observed['human_right'] - t_r7) if t_r7 < observed['human_right'] else 'at or above observed'}"
    )
    lines.append("")
    lines.append("**Decision rule (from ChatGPT's interpretation table):**")
    lines.append(
        "- If oracle components can reach observed extremes, the residual "
        "focus-cell PPC gap reflects state/tree propagation failure, not "
        "emission sharpness. Work on the tree / reference mechanism."
    )
    lines.append(
        "- If oracle components cannot reach observed extremes (ceiling < obs), "
        "the shared ordered-probit emission is too blunt for Rater_B's rating "
        "style. Expert-specific scale or cutpoints are justified."
    )
    lines.append("")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Tree propagation
# ---------------------------------------------------------------------------


def tree_qj_distribution(
    idata: Any, builder: MultiSystemModelBuilder, processor: MultiSystemDataProcessor,
    system_name: str,
) -> List[Dict[str, Any]]:
    """Posterior q_j summaries for all indicators under one system."""
    sp = builder._sys_prefix(system_name)
    post = idata.posterior
    rows: List[Dict[str, Any]] = []
    for nkey, varname in builder.node_to_varname.items():
        q_name = f"{sp}__{varname}_p"
        if q_name not in post.data_vars:
            continue
        q = np.asarray(post[q_name].values).reshape(-1)
        rows.append(
            {
                "indicator": nkey.split(" > ")[-1],
                "nkey": nkey,
                "median": float(np.median(q)),
                "p03": float(np.percentile(q, 3)),
                "p97": float(np.percentile(q, 97)),
            }
        )
    return rows


def write_tree_propagation_report(
    human_binary: List[Dict[str, Any]],
    eliza_binary: List[Dict[str, Any]],
    human_three: List[Dict[str, Any]],
    eliza_three: List[Dict[str, Any]],
    path: Path,
) -> None:
    lines = []
    lines.append("# Tree-propagation diagnostic\n")
    lines.append(
        "Anchored $C_\\text{Human} = 0.999$, $C_\\text{ELIZA} = 0.001$ "
        "(anchor gap = 0.998). If the tree propagates these anchors strongly "
        "to the indicator layer, we expect tree-implied $q_j$ at Human "
        "indicators to cluster near 1 and at ELIZA indicators to cluster "
        "near 0, with $\\Delta q_j = q_\\text{Human} - q_\\text{ELIZA} "
        "\\approx 1$ for most indicators. Compressed propagation shows up "
        "as $\\Delta q_j$ much smaller than the anchor gap.\n"
    )

    def _summ(rows, label):
        medians = np.array([r["median"] for r in rows])
        return {
            "n": len(rows),
            "mean_q": float(medians.mean()),
            "median_q": float(np.median(medians)),
            "p03": float(np.percentile(medians, 3)),
            "p97": float(np.percentile(medians, 97)),
            "min": float(medians.min()),
            "max": float(medians.max()),
        }

    # Match indicators between Human and ELIZA by nkey for Delta q_j
    def _join(h_rows, e_rows):
        by_key_h = {r["nkey"]: r["median"] for r in h_rows}
        by_key_e = {r["nkey"]: r["median"] for r in e_rows}
        common = set(by_key_h) & set(by_key_e)
        deltas = np.array([by_key_h[k] - by_key_e[k] for k in common])
        return deltas

    deltas_b = _join(human_binary, eliza_binary)
    deltas_t = _join(human_three, eliza_three)
    summ_h_b = _summ(human_binary, "Human binary")
    summ_e_b = _summ(eliza_binary, "ELIZA binary")
    summ_h_t = _summ(human_three, "Human three-state")
    summ_e_t = _summ(eliza_three, "ELIZA three-state")

    lines.append("## Summary of tree-implied q_j medians across indicators\n")
    lines.append("| System | Model | n | mean q_j | median q_j | min | max | 94% interval |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for label, s in [
        ("Human", "binary"),
        ("ELIZA", "binary"),
        ("Human", "three-state"),
        ("ELIZA", "three-state"),
    ]:
        summ = {
            ("Human", "binary"): summ_h_b,
            ("ELIZA", "binary"): summ_e_b,
            ("Human", "three-state"): summ_h_t,
            ("ELIZA", "three-state"): summ_e_t,
        }[(label, s)]
        lines.append(
            f"| {label} | {s} | {summ['n']} | {summ['mean_q']:.3f} | "
            f"{summ['median_q']:.3f} | {summ['min']:.3f} | {summ['max']:.3f} | "
            f"[{summ['p03']:.3f}, {summ['p97']:.3f}] |"
        )
    lines.append("")

    lines.append("## Per-indicator Δq_j = q_Human − q_ELIZA\n")
    lines.append("| Model | n indicators | mean Δq | median Δq | min | max | 94% of Δq |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for label, d in [("binary", deltas_b), ("three-state", deltas_t)]:
        lines.append(
            f"| {label} | {len(d)} | {d.mean():.3f} | {np.median(d):.3f} | "
            f"{d.min():.3f} | {d.max():.3f} | "
            f"[{np.percentile(d, 3):.3f}, {np.percentile(d, 97):.3f}] |"
        )
    lines.append("")

    lines.append("## Interpretation\n")
    max_gap = 0.998
    mean_frac_b = deltas_b.mean() / max_gap
    mean_frac_t = deltas_t.mean() / max_gap
    lines.append(
        f"- Root anchor gap $C_\\text{{Human}} - C_\\text{{ELIZA}} = 0.998$."
    )
    lines.append(
        f"- Under binary tree propagation, indicator-level $\\Delta q_j$ averages "
        f"**{deltas_b.mean():.3f}** (= {100*mean_frac_b:.1f}% of root anchor gap)."
    )
    lines.append(
        f"- Under three-state tree propagation (same tree, different leaf), "
        f"$\\Delta q_j$ averages **{deltas_t.mean():.3f}** "
        f"(= {100*mean_frac_t:.1f}% of root anchor gap)."
    )
    lines.append("")
    lines.append(
        "Tree propagation is a property of the upstream hierarchy "
        "(NODE_CONCENTRATION, support/demandingness mapping, tree depth) "
        "and is essentially identical across the two leaf models. A "
        "$\\Delta q_j$ much below the root anchor gap means the tree "
        "hierarchy compresses anchor credence at each layer -- most of the "
        "anchor work at reference cells is being done by the leaf "
        "likelihood after seeing ratings, not by the tree. This matches "
        "the week-6 reference-recovery finding and is a separate "
        "failure mode from leaf-model choice."
    )
    lines.append("")

    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading saved idatas and contexts...")
    idata_binary = az.from_netcdf(RESULTS_DIR / "binary_anchored.nc")
    idata_three = az.from_netcdf(RESULTS_DIR / "three_state_anchored.nc")
    _, builder_b, proc_b = _build_context("binary")
    _, builder_t, proc_t = _build_context("three_state")

    # --- 1. Ceiling diagnostic ---
    print("\nComputing oracle emission ceilings...")
    binary_rows = ceiling_rows_for_model(idata_binary, "binary")
    three_rows = ceiling_rows_for_model(idata_three, "three_state")
    observed = {
        "human_right": 43.0 / 50.0,  # 0.860
        "eliza_left": 48.0 / 50.0,  # 0.960
    }
    write_ceiling_report(
        binary_rows, three_rows, observed,
        ANALYSIS_DIR / "ceiling_diagnostic.md",
    )
    print(f"  wrote {ANALYSIS_DIR / 'ceiling_diagnostic.md'}")

    # --- 2. Tree-propagation diagnostic ---
    print("\nComputing tree-propagation Delta q_j...")
    human_b = tree_qj_distribution(idata_binary, builder_b, proc_b, "Human")
    eliza_b = tree_qj_distribution(idata_binary, builder_b, proc_b, "ELIZA")
    human_t = tree_qj_distribution(idata_three, builder_t, proc_t, "Human")
    eliza_t = tree_qj_distribution(idata_three, builder_t, proc_t, "ELIZA")
    write_tree_propagation_report(
        human_b, eliza_b, human_t, eliza_t,
        ANALYSIS_DIR / "tree_propagation_diagnostic.md",
    )
    print(f"  wrote {ANALYSIS_DIR / 'tree_propagation_diagnostic.md'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
