"""Companion workflow for GWT hierarchical expert-cutpoint comparisons.

Keeps notebook 13 intact and adds a focused comparison for:
  - baseline shared-kappa model
  - wide-prior shared-kappa control
  - hierarchical expert-cutpoint model

The sidecar diagnostic previously described informally as a q_j PPC is named
``tree_implied`` here to avoid implying any prior-predictive interpretation.
It uses the upstream tree-implied indicator probabilities ``..._p`` rather than
the leaf-updated ``..._pz1`` probabilities.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from dcm_model import ModelConfig, fit_stance_multisystem, load_data
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
    ("Derek Shiller", "Human"),
    ("Derek Shiller", "ELIZA"),
    ("Rachael Miller", "Chicken"),
    ("Luhan Mikaelson", "2024 Leading Chat LLMs"),
]
DEFAULT_FIT_OVERRIDES = {
    "NUM_SAMPLES": 2000,
    "NUM_TUNE": 1500,
    "NUM_CHAINS": 4,
    "TARGET_ACCEPT": 0.9,
}


def build_configs(
    fit_overrides: Dict[str, Any] | None = None,
) -> Dict[str, ModelConfig]:
    """Return the three GWT comparison configs with shared fit overrides."""
    common = dict(DEFAULT_FIT_OVERRIDES)
    if fit_overrides:
        common.update(fit_overrides)

    return {
        "baseline": ModelConfig(USE_EXPERT_SHIFTS=False, **common),
        "wide_prior": ModelConfig(
            USE_EXPERT_SHIFTS=False,
            A_PRIOR_SIGMA=5.0,
            KAPPA_PRIOR_SIGMA=4.0,
            **common,
        ),
        "hierarchical_kappa": ModelConfig(
            USE_EXPERT_SHIFTS=False,
            USE_HIERARCHICAL_EXPERT_CUTPOINTS=True,
            HIER_KAPPA_EXPERT_SCALE_SIGMA=0.15,
            **common,
        ),
    }


def extract_c_summary(idata: Any, builder: Any) -> Dict[str, Dict[str, float]]:
    """Posterior median and 94% interval for each system's stance-level C."""
    vn = builder.node_to_varname[STANCE]
    out: Dict[str, Dict[str, float]] = {}
    for sys_name, c_fixed in SYSTEM_CONFIGS:
        key = f"{builder._sys_prefix(sys_name)}__{vn}_C"
        if key in idata.posterior:
            draws = idata.posterior[key].values.reshape(-1)
            out[sys_name] = {
                "median": float(np.median(draws)),
                "lo": float(np.percentile(draws, 3)),
                "hi": float(np.percentile(draws, 97)),
            }
        else:
            out[sys_name] = {
                "median": float(c_fixed),
                "lo": float(c_fixed),
                "hi": float(c_fixed),
            }
    return out


def obs_weighted_abs_error(
    ppc: Dict[Tuple[str, str], Dict[str, Any]],
    stat: str,
) -> float:
    """Obs-weighted mean absolute error for one PPC summary statistic."""
    total_n = sum(v["n_obs"] for v in ppc.values())
    return sum(
        abs(v[f"obs_{stat}"] - v[f"pred_{stat}_mean"]) * v["n_obs"]
        for v in ppc.values()
    ) / total_n


def diagnostic_var_names(idata: Any) -> list[str]:
    """Diagnostics should include the new hierarchical expert parameters."""
    diag_vars = ["a", "kappa"]
    for name in [
        "kappa_expert_gap_scale",
        "kappa_expert_gap_offset",
        "kappa_by_expert",
    ]:
        if name in idata.posterior.data_vars:
            diag_vars.append(name)
    return diag_vars


def diagnostics_overview(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compact convergence overview across the compared models."""
    out: Dict[str, Dict[str, float]] = {}
    for label, data in results.items():
        diag = data["diag"]
        out[label] = {
            "divergences": int(data["idata"].sample_stats["diverging"].values.sum()),
            "max_rhat": float(diag["r_hat"].max()),
            "min_ess": float(diag["ess_bulk"].min()),
            "a_mean": float(data["idata"].posterior["a"].values.mean()),
        }
        if "kappa_expert_gap_scale" in data["idata"].posterior.data_vars:
            out[label]["kappa_expert_gap_scale_mean"] = float(
                data["idata"].posterior["kappa_expert_gap_scale"].values.mean()
            )
    return out


def print_diagnostics_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a short diagnostics table for the compared models."""
    overview = diagnostics_overview(results)
    header = (
        f"{'Model':<18s} {'div':>5s} {'max_rhat':>9s} {'min_ess':>9s} "
        f"{'a_mean':>8s} {'gap_scale':>9s}"
    )
    print(header)
    print("-" * len(header))
    for label, row in overview.items():
        gap_scale = (
            f"{row['kappa_expert_gap_scale_mean']:.3f}"
            if "kappa_expert_gap_scale_mean" in row
            else "-"
        )
        print(
            f"{label:<18s} {row['divergences']:>5d} {row['max_rhat']:>9.3f} "
            f"{row['min_ess']:>9.0f} {row['a_mean']:>8.3f} {gap_scale:>9s}"
        )


def focus_cell_table(
    ppc: Dict[Tuple[str, str], Dict[str, Any]],
    model_label: str,
) -> str:
    """Compact text table for the cells driving the current misfit discussion."""
    lines = [
        f"{'Model':<18s} {'Cell':<28s} {'obs_m':>5s} {'pred_m':>7s} {'obs_ex':>7s} {'pred_ex':>8s}",
        "-" * 82,
    ]
    for expert, system in FOCUS_KEYS:
        key = (expert, system)
        if key not in ppc:
            continue
        row = ppc[key]
        cell_name = f"{expert} x {SYSTEM_DISPLAY.get(system, system)}"
        lines.append(
            f"{model_label:<18s} {cell_name:<28s} "
            f"{row['obs_mean']:>5.2f} {row['pred_mean_mean']:>7.2f} "
            f"{row['obs_extreme']:>7.2f} {row['pred_extreme_mean']:>8.2f}"
        )
    return "\n".join(lines)


def focus_cell_comparison_table(
    results: Dict[str, Dict[str, Any]],
    key: str = "ppc_strat",
) -> str:
    """Side-by-side focus-cell comparison across the three compared models."""
    lines = [
        f"{'Cell':<28s} {'Model':<18s} {'obs_m':>5s} {'pred_m':>7s} {'obs_ex':>7s} {'pred_ex':>8s}",
        "-" * 88,
    ]
    for expert, system in FOCUS_KEYS:
        cell_name = f"{expert} x {SYSTEM_DISPLAY.get(system, system)}"
        for model_label in ["baseline", "wide_prior", "hierarchical_kappa"]:
            ppc = results[model_label][key]
            row = ppc.get((expert, system))
            if row is None:
                continue
            lines.append(
                f"{cell_name:<28s} {model_label:<18s} "
                f"{row['obs_mean']:>5.2f} {row['pred_mean_mean']:>7.2f} "
                f"{row['obs_extreme']:>7.2f} {row['pred_extreme_mean']:>8.2f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def print_c_summary_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Print stance-level C comparisons for the target systems."""
    header = (
        f"{'Model':<18s} {'Chicken':>28s} {'LLMs':>28s}"
    )
    print(header)
    print("-" * len(header))
    for label, data in results.items():
        chick = data["c_summary"]["Chicken"]
        llm = data["c_summary"]["2024 Leading Chat LLMs"]
        chick_str = f"{chick['median']:.3f} [{chick['lo']:.3f}, {chick['hi']:.3f}]"
        llm_str = f"{llm['median']:.3f} [{llm['lo']:.3f}, {llm['hi']:.3f}]"
        print(f"{label:<18s} {chick_str:>28s} {llm_str:>28s}")


def print_ppc_metric_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Print global PPC error metrics for the three models."""
    header = (
        f"{'Model':<18s} {'pz1 |Δmean|':>12s} {'pz1 |Δext|':>12s} "
        f"{'tree |Δmean|':>13s} {'tree |Δext|':>12s}"
    )
    print(header)
    print("-" * len(header))
    for label, data in results.items():
        pz1_mean = f"{data['ppc_metrics']['pz1_mean']:.3f}"
        pz1_ext = f"{data['ppc_metrics']['pz1_extreme']:.3f}"
        tree_mean = f"{data['ppc_metrics']['tree_mean']:.3f}"
        tree_ext = f"{data['ppc_metrics']['tree_extreme']:.3f}"
        print(
            f"{label:<18s} "
            f"{pz1_mean:>12s} "
            f"{pz1_ext:>12s} "
            f"{tree_mean:>13s} "
            f"{tree_ext:>12s}"
        )


def plot_c_comparison(results: Dict[str, Dict[str, Any]], save_dir: Path) -> None:
    """Forest-style comparison for baseline, wide, and hierarchical C posteriors."""
    offsets = {"baseline": -0.16, "wide_prior": 0.0, "hierarchical_kappa": 0.16}
    markers = {"baseline": "o", "wide_prior": "s", "hierarchical_kappa": "^"}
    colours = {
        "Human": "forestgreen",
        "Chicken": "goldenrod",
        "2024 Leading Chat LLMs": "steelblue",
        "ELIZA": "salmon",
    }

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (sys_name, _) in enumerate(SYSTEM_CONFIGS):
        for model_label, y_off in offsets.items():
            summary = results[model_label]["c_summary"][sys_name]
            x = summary["median"]
            lo = summary["lo"]
            hi = summary["hi"]
            y = i + y_off
            if lo == hi:
                ax.plot([x], [y], "D", color=colours[sys_name], markersize=7)
            else:
                ax.errorbar(
                    [x],
                    [y],
                    xerr=[[max(x - lo, 0.0)], [max(hi - x, 0.0)]],
                    fmt=markers[model_label],
                    capsize=3,
                    color=colours[sys_name],
                    markersize=6,
                    label=model_label if i == 0 else None,
                )

    ax.set_yticks(range(len(SYSTEM_CONFIGS)))
    ax.set_yticklabels([SYSTEM_DISPLAY.get(s, s) for s, _ in SYSTEM_CONFIGS])
    ax.invert_yaxis()
    ax.axvline(1 / 6, color="grey", linestyle=":", alpha=0.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(r"$P(C=1 \mid \mathrm{data})$ (posterior median)")
    ax.set_title("GWT joint fit: baseline vs wide prior vs hierarchical cutpoints")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(save_dir / "gwt_joint_c_baseline_wide_hier.png", dpi=150)
    plt.show()


def plot_stratified_ppc_suite(
    results: Dict[str, Dict[str, Any]],
    key: str,
    title_prefix: str,
    save_dir: Path,
) -> None:
    """Plot a stratified PPC grid for each model under one predictive choice."""
    for model_label, data in results.items():
        fig, _ = plot_per_expert_system_histograms(
            data[key],
            K=data["config"].N_CATEGORIES,
            system_display=SYSTEM_DISPLAY,
            system_order=SYSTEM_ORDER,
            expert_order=data["processor"].expert_names,
            title=f"{title_prefix} — {model_label.replace('_', ' ')}",
        )
        fig.savefig(
            save_dir / f"gwt_{model_label}_{key}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()


def build_interpretation_note(results: Dict[str, Dict[str, Any]]) -> str:
    """Return a short run-dependent interpretation note."""
    base = results["baseline"]
    hier = results["hierarchical_kappa"]

    def cell_abs_err(
        ppc: Dict[Tuple[str, str], Dict[str, Any]],
        key: Tuple[str, str],
        stat: str,
    ) -> float:
        row = ppc[key]
        return abs(row[f"obs_{stat}"] - row[f"pred_{stat}_mean"])

    ref_keys = [("Derek Shiller", "Human"), ("Derek Shiller", "ELIZA")]
    mid_keys = [
        ("Rachael Miller", "Chicken"),
        ("Luhan Mikaelson", "2024 Leading Chat LLMs"),
    ]
    ref_delta = np.mean(
        [
            cell_abs_err(hier["ppc_strat"], key, "extreme")
            - cell_abs_err(base["ppc_strat"], key, "extreme")
            for key in ref_keys
            if key in base["ppc_strat"] and key in hier["ppc_strat"]
        ]
    )
    mid_delta = np.mean(
        [
            cell_abs_err(hier["ppc_strat"], key, "mean")
            - cell_abs_err(base["ppc_strat"], key, "mean")
            for key in mid_keys
            if key in base["ppc_strat"] and key in hier["ppc_strat"]
        ]
    )

    base_chicken = base["c_summary"]["Chicken"]["median"]
    hier_chicken = hier["c_summary"]["Chicken"]["median"]
    base_llm = base["c_summary"]["2024 Leading Chat LLMs"]["median"]
    hier_llm = hier["c_summary"]["2024 Leading Chat LLMs"]["median"]
    global_mean_delta = hier["ppc_metrics"]["pz1_mean"] - base["ppc_metrics"]["pz1_mean"]
    global_ext_delta = (
        hier["ppc_metrics"]["pz1_extreme"] - base["ppc_metrics"]["pz1_extreme"]
    )

    lines = ["Interpretation note"]
    if ref_delta < -0.03:
        lines.append(
            "- The hierarchical cutpoints materially improve the Derek/Human-ELIZA reference cells."
        )
    else:
        lines.append(
            "- The hierarchical cutpoints do not visibly improve the Derek/Human-ELIZA reference cells."
        )

    if abs(hier_chicken - base_chicken) <= 0.05 and abs(hier_llm - base_llm) <= 0.05:
        lines.append(
            "- Chicken and LLM stance posteriors remain close to the validated baseline."
        )
    else:
        lines.append(
            "- Chicken and/or LLM stance posteriors move materially relative to baseline; inspect before accepting."
        )

    if mid_delta >= -0.03:
        lines.append(
            "- The middle-concentrated Chicken/LLM cells remain a bottleneck, which points to the binary latent-z structure as the next likely constraint."
        )
    else:
        lines.append(
            "- Some middle-cell misfit improves as well, so the cutpoint rigidity was contributing beyond the reference cells."
        )

    if global_mean_delta > 0.03 or global_ext_delta > 0.03:
        lines.append(
            "- Global stratified PPC gets worse elsewhere, so this may be fixing Derek by smearing other experts."
        )
    else:
        lines.append(
            "- There is no broad sign of fixing Derek by simply degrading the other experts' fit."
        )

    return "\n".join(lines)


def run_all(
    fit_overrides: Dict[str, Any] | None = None,
    ppc_draws: int = 500,
    save_figures: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Run the three-model GWT comparison workflow and return all artifacts."""
    plt.rcParams.update(
        {"figure.dpi": 120, "font.size": 10, "figure.constrained_layout.use": True}
    )

    stance_data = next(s for s in load_data(ModelConfig()) if s["name"] == STANCE)
    configs = build_configs(fit_overrides)
    save_dir = Path("report_figures")
    save_dir.mkdir(exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    for label, cfg in configs.items():
        if verbose:
            print(f"\n=== {label} ===")
        t0 = time.time()
        idata, builder, processor = fit_stance_multisystem(
            stance_data, cfg, SYSTEM_CONFIGS
        )
        elapsed = time.time() - t0
        diag = az.summary(
            idata,
            var_names=diagnostic_var_names(idata),
            kind="diagnostics",
        )
        ppc_strat = per_expert_system_ppc_multisystem(
            idata, builder, processor, n_draws=ppc_draws, seed=0
        )
        ppc_tree = per_expert_system_tree_implied_ppc_multisystem(
            idata, builder, processor, n_draws=ppc_draws, seed=0
        )
        results[label] = {
            "config": cfg,
            "idata": idata,
            "builder": builder,
            "processor": processor,
            "elapsed_s": elapsed,
            "diag": diag,
            "c_summary": extract_c_summary(idata, builder),
            "ppc_strat": ppc_strat,
            "tree_implied_ppc_strat": ppc_tree,
            "ppc_metrics": {
                "pz1_mean": obs_weighted_abs_error(ppc_strat, "mean"),
                "pz1_extreme": obs_weighted_abs_error(ppc_strat, "extreme"),
                "tree_mean": obs_weighted_abs_error(ppc_tree, "mean"),
                "tree_extreme": obs_weighted_abs_error(ppc_tree, "extreme"),
            },
        }

        if verbose:
            print("elapsed_s", round(elapsed, 1))
            print("divergences", int(idata.sample_stats["diverging"].values.sum()))
            print("max_rhat", float(diag["r_hat"].max()))
            print("min_ess", float(diag["ess_bulk"].min()))
            print(focus_cell_table(ppc_strat, label))

    if verbose:
        print("\nPosterior C comparison")
        print_c_summary_table(results)
        print("\nDiagnostics")
        print_diagnostics_table(results)
        print("\nGlobal PPC comparison")
        print_ppc_metric_table(results)
        print("\nFocus cells — pz1-based PPC")
        print(focus_cell_comparison_table(results, key="ppc_strat"))
        print("\nFocus cells — tree-implied PPC")
        print(focus_cell_comparison_table(results, key="tree_implied_ppc_strat"))
        print("\n" + build_interpretation_note(results))

    if save_figures:
        plot_c_comparison(results, save_dir)
        plot_stratified_ppc_suite(
            results,
            "ppc_strat",
            "Stratified pz1-based PPC",
            save_dir,
        )
        plot_stratified_ppc_suite(
            results,
            "tree_implied_ppc_strat",
            "Stratified tree-implied PPC",
            save_dir,
        )

    return results


if __name__ == "__main__":
    run_all()
