"""Apples-vs-apples PPC comparison on the current pool_3s baseline.

Loads the leading three-state + pooled-tree + soft-anchor fit and runs the
per-(expert, system) PPC under both source choices:

  - ``pz1`` (default): leaf-updated, data-conditioned indicator posterior.
  - ``tree_implied`` (= ``q``): upstream tree-implied indicator probability.

The pz1 PPC uses the per-indicator posterior derived from the same ratings
being predicted; it primarily probes whether the ordered-probit emission
layer is flexible enough given a "data-fitted" latent state. The
tree-implied PPC uses the tree's contribution alone; it probes whether the
tree adequately drives cross-system rating differences from C.

Per the arvo (Apr 27) and week-7 reports, the tree-implied PPC is where the
focus-cell misfit shows up — exactly the structural attenuation symptom.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from dcm_model import (
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    MultiSystemModelBuilder,
    load_data,
)
from dcm_ppc import (
    per_expert_system_ppc_multisystem,
    per_expert_system_tree_implied_ppc_multisystem,
    plot_per_expert_system_histograms,
)

STANCE = "Global Workspace Theory"
SYSTEM_CONFIGS = [
    ("Human", 0.999),
    ("Chicken", None),
    ("2024 Leading Chat LLMs", None),
    ("ELIZA", 0.001),
]
SYSTEM_DISPLAY = {
    "Human": "Human",
    "Chicken": "Chicken",
    "2024 Leading Chat LLMs": "LLMs",
    "ELIZA": "ELIZA",
}
SYSTEM_ORDER = ["Human", "Chicken", "2024 Leading Chat LLMs", "ELIZA"]
FOCUS_KEYS = [
    ("Rater_B", "Human"),
    ("Rater_B", "ELIZA"),
    ("Rater_E", "Chicken"),
    ("Rater_D", "2024 Leading Chat LLMs"),
]

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "gwt_tree_pooling"
FIG_DIR = REPO_ROOT / "report_figures"


def make_pool_3s_soft_config() -> ModelConfig:
    """Reconstruct the ModelConfig used for ``three_state_pooled_soft_anchored.nc``."""
    return ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        POOL_BETAS_BY_LABEL=True,
        LABEL_POOL_SIGMA=0.5,
        USE_EXPERT_SHIFTS=False,
        A_PRIOR_SIGMA=2.0,
        KAPPA_PRIOR_SIGMA=2.0,
        SOFT_REFERENCE_ANCHORS={
            "Human": (50.0, 1.0),
            "ELIZA": (1.0, 50.0),
        },
    )


def load_fit_and_rebuild(
    nc_path: Path,
    config: ModelConfig,
    system_configs=SYSTEM_CONFIGS,
    stance_name: str = STANCE,
):
    """Load saved InferenceData and rebuild processor + builder (no sampling)."""
    stance_data = next(s for s in load_data(config) if s["name"] == stance_name)
    processor = MultiSystemDataProcessor(config)
    processor.process(stance_data, [s for s, _ in system_configs])
    evidence_proc = EvidenceProcessor(config)
    builder = MultiSystemModelBuilder(
        config, evidence_proc, processor, system_configs
    )
    # build_model populates builder.node_to_varname as a side effect.
    # No sampling is triggered.
    builder.build_model(stance_data)
    idata = az.from_netcdf(nc_path)
    return idata, builder, processor, stance_data


def obs_weighted_abs_error(
    ppc: Dict[Tuple[str, str], Dict[str, Any]], stat: str
) -> float:
    total_n = sum(v["n_obs"] for v in ppc.values())
    return sum(
        abs(v[f"obs_{stat}"] - v[f"pred_{stat}_mean"]) * v["n_obs"]
        for v in ppc.values()
    ) / total_n


def focus_cell_comparison_table(
    ppc_pz1: Dict[Tuple[str, str], Dict[str, Any]],
    ppc_tree: Dict[Tuple[str, str], Dict[str, Any]],
) -> str:
    """Side-by-side focus-cell table: pz1 vs tree-implied."""
    lines = [
        f"{'Cell':<32s} {'obs_m':>5s} {'pz1_pred_m':>10s} {'tree_pred_m':>11s}  "
        f"{'obs_ex':>6s} {'pz1_pred_ex':>11s} {'tree_pred_ex':>12s}",
        "-" * 100,
    ]
    for expert, system in FOCUS_KEYS:
        key = (expert, system)
        if key not in ppc_pz1 or key not in ppc_tree:
            continue
        rp = ppc_pz1[key]
        rt = ppc_tree[key]
        cell_name = f"{expert} x {SYSTEM_DISPLAY.get(system, system)}"
        lines.append(
            f"{cell_name:<32s} "
            f"{rp['obs_mean']:>5.2f} {rp['pred_mean_mean']:>10.2f} "
            f"{rt['pred_mean_mean']:>11.2f}  "
            f"{rp['obs_extreme']:>6.2f} {rp['pred_extreme_mean']:>11.2f} "
            f"{rt['pred_extreme_mean']:>12.2f}"
        )
    return "\n".join(lines)


def global_metric_table(
    ppc_pz1: Dict[Tuple[str, str], Dict[str, Any]],
    ppc_tree: Dict[Tuple[str, str], Dict[str, Any]],
) -> str:
    """Obs-weighted |Δ| metrics across all (expert, system) cells."""
    lines = [
        f"{'Source':<14s} {'|Δmean|':>10s} {'|Δextreme|':>12s} "
        f"{'|Δleft|':>10s} {'|Δright|':>10s}",
        "-" * 60,
    ]
    for label, ppc in [("pz1", ppc_pz1), ("tree_implied", ppc_tree)]:
        lines.append(
            f"{label:<14s} "
            f"{obs_weighted_abs_error(ppc, 'mean'):>10.3f} "
            f"{obs_weighted_abs_error(ppc, 'extreme'):>12.3f} "
            f"{obs_weighted_abs_error(ppc, 'left'):>10.3f} "
            f"{obs_weighted_abs_error(ppc, 'right'):>10.3f}"
        )
    return "\n".join(lines)


def run(
    nc_filename: str = "three_state_pooled_soft_anchored.nc",
    n_draws: int = 500,
    save_figs: bool = True,
    fig_tag: str = "pool_3s_soft",
) -> Dict[str, Any]:
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
    nc_path = RESULTS_DIR / nc_filename
    print(f"Loading fit: {nc_path}")
    config = make_pool_3s_soft_config()
    idata, builder, processor, stance_data = load_fit_and_rebuild(
        nc_path, config
    )
    print(f"Loaded {nc_path.name}.")
    print(
        f"Posterior shape: {idata.posterior.dims}; "
        f"{len(processor.expert_names)} experts, "
        f"{len(builder.node_to_varname)} nodes."
    )

    print("Computing pz1 (default leaf-updated) PPC...")
    ppc_pz1 = per_expert_system_ppc_multisystem(
        idata, builder, processor, n_draws=n_draws, seed=0
    )
    print("Computing tree-implied (q) PPC...")
    ppc_tree = per_expert_system_tree_implied_ppc_multisystem(
        idata, builder, processor, n_draws=n_draws, seed=0
    )

    print("\n=== Global obs-weighted |Δ| metrics ===")
    print(global_metric_table(ppc_pz1, ppc_tree))
    print("\n=== Focus-cell comparison ===")
    print(focus_cell_comparison_table(ppc_pz1, ppc_tree))

    if save_figs:
        FIG_DIR.mkdir(exist_ok=True)
        for source_label, ppc in [
            ("pz1", ppc_pz1),
            ("tree_implied", ppc_tree),
        ]:
            fig, _ = plot_per_expert_system_histograms(
                ppc,
                K=config.N_CATEGORIES,
                system_display=SYSTEM_DISPLAY,
                system_order=SYSTEM_ORDER,
                expert_order=processor.expert_names,
                title=(
                    f"Stratified {source_label.replace('_', '-')} PPC — "
                    f"{fig_tag.replace('_', ' ')}"
                ),
            )
            out_path = FIG_DIR / f"{fig_tag}_{source_label}_ppc_strat.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {out_path}")

    return {
        "config": config,
        "idata": idata,
        "builder": builder,
        "processor": processor,
        "ppc_pz1": ppc_pz1,
        "ppc_tree": ppc_tree,
        "metrics": {
            source: {
                stat: obs_weighted_abs_error(ppc, stat)
                for stat in ["mean", "extreme", "left", "right"]
            }
            for source, ppc in [("pz1", ppc_pz1), ("tree_implied", ppc_tree)]
        },
    }


if __name__ == "__main__":
    run()
