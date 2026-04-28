"""Focus-cell diagnostics from persisted binary + three-state GWT idata.

Outputs:
  results/gwt_binary_three_state/analysis/per_cell_utilisation.md
      Per-focus-cell table with per-indicator posterior medians of
      tree-implied q_j, p_m0, p_m1, p_m2, expected_z, and observed rating.

  report_figures/three_state/gwt_ppc_binary.png
  report_figures/three_state/gwt_ppc_three_state.png
      Stratified per-(expert, system) PPC histograms for both models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
    plot_per_expert_system_histograms,
)
from gwt_three_state_indicator_analysis import (
    FOCUS_KEYS,
    STANCE,
    SYSTEM_CONFIGS_VALIDATED,
    SYSTEM_DISPLAY,
    SYSTEM_ORDER,
)

RESULTS_DIR = Path("results/gwt_binary_three_state")
FIG_DIR = Path("report_figures/three_state")


def _build_context(state_model: str) -> Tuple[Any, MultiSystemModelBuilder, MultiSystemDataProcessor]:
    """Rebuild the multi-system context (processor + builder) without sampling."""
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


def per_cell_utilisation_rows(
    idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
    focus_cells: Sequence[Tuple[str, str]] = tuple(FOCUS_KEYS),
) -> List[Dict[str, Any]]:
    """Per-indicator posterior summaries within each focus cell."""
    post = idata.posterior
    rows: List[Dict[str, Any]] = []
    for expert_name, system_name in focus_cells:
        if expert_name not in processor.expert_to_idx:
            continue
        expert_idx = processor.expert_to_idx[expert_name]
        sp = builder._sys_prefix(system_name)
        sys_obs = processor.system_observations.get(system_name, {})
        for nkey, obs_list in sys_obs.items():
            expert_ratings = [r for e, r in obs_list if e == expert_idx]
            if not expert_ratings:
                continue
            varname = builder.node_to_varname.get(nkey)
            if varname is None:
                continue
            q_name = f"{sp}__{varname}_p"
            p_m0_name = f"{sp}__{varname}_p_m0"
            p_m1_name = f"{sp}__{varname}_p_m1"
            p_m2_name = f"{sp}__{varname}_p_m2"
            exp_z_name = f"{sp}__{varname}_expected_z"

            def _med(name: str) -> float | None:
                if name not in post.data_vars:
                    return None
                return float(np.median(np.asarray(post[name].values).reshape(-1)))

            rows.append(
                {
                    "expert": expert_name,
                    "system": system_name,
                    "indicator": nkey.split(" > ")[-1],
                    "n_obs_by_expert": len(expert_ratings),
                    "obs_mean_rating_0idx": float(np.mean(expert_ratings)),
                    "q_median": _med(q_name),
                    "p_m0_median": _med(p_m0_name),
                    "p_m1_median": _med(p_m1_name),
                    "p_m2_median": _med(p_m2_name),
                    "expected_z_median": _med(exp_z_name),
                }
            )
    return rows


def format_per_cell_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Markdown table for the per-cell utilisation output."""
    by_cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        by_cell.setdefault((r["expert"], r["system"]), []).append(r)

    out: List[str] = []
    out.append("# Per-focus-cell component utilisation (three-state)\n")
    out.append(
        "Per-indicator posterior medians from ``three_state_anchored.nc``. "
        "Ratings shown are 0-indexed (0 = strongly absent, 6 = strongly present).\n"
    )

    for cell, cell_rows in by_cell.items():
        expert, sys_name = cell
        sys_label = SYSTEM_DISPLAY.get(sys_name, sys_name)
        out.append(f"\n## {expert} × {sys_label}  ({len(cell_rows)} indicators)\n")
        out.append(
            "| Indicator | obs rating | tree q_j | p_m0 | p_m1 | p_m2 | expected_z |"
        )
        out.append("|---|---:|---:|---:|---:|---:|---:|")
        for r in cell_rows:
            q_s = f"{r['q_median']:.3f}" if r["q_median"] is not None else "—"
            pm0 = f"{r['p_m0_median']:.3f}" if r["p_m0_median"] is not None else "—"
            pm1 = f"{r['p_m1_median']:.3f}" if r["p_m1_median"] is not None else "—"
            pm2 = f"{r['p_m2_median']:.3f}" if r["p_m2_median"] is not None else "—"
            ez = (
                f"{r['expected_z_median']:.3f}"
                if r["expected_z_median"] is not None
                else "—"
            )
            out.append(
                f"| {r['indicator'][:40]} | {r['obs_mean_rating_0idx']:.1f} | "
                f"{q_s} | {pm0} | {pm1} | {pm2} | {ez} |"
            )
        # Cell-level means
        def _m(key: str) -> float:
            vals = [r[key] for r in cell_rows if r[key] is not None]
            return float(np.mean(vals)) if vals else float("nan")

        out.append("")
        out.append(
            f"**Cell means** — tree $q_j$: {_m('q_median'):.3f}, "
            f"$p_{{m=0}}$: {_m('p_m0_median'):.3f}, "
            f"$p_{{m=1}}$: {_m('p_m1_median'):.3f}, "
            f"$p_{{m=2}}$: {_m('p_m2_median'):.3f}, "
            f"$E[z]$: {_m('expected_z_median'):.3f}"
        )
        out.append(
            f"**Mid-dominant indicators** (median $p_{{m=1}} > 0.5$): "
            f"{sum(1 for r in cell_rows if (r['p_m1_median'] or 0) > 0.5)}"
            f" / {len(cell_rows)}"
        )
    return "\n".join(out)


def main() -> None:
    # --- three-state: per-cell utilisation ---
    print("[1/3] Loading three-state idata and building context...")
    idata_three = az.from_netcdf(RESULTS_DIR / "three_state_anchored.nc")
    _, builder_three, processor_three = _build_context("three_state")
    print("[2/3] Computing per-cell utilisation rows...")
    rows = per_cell_utilisation_rows(idata_three, builder_three, processor_three)
    md_path = RESULTS_DIR / "analysis" / "per_cell_utilisation.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(format_per_cell_table(rows))
    print(f"      wrote {md_path}  ({len(rows)} indicator rows)")

    # --- binary: PPC grid ---
    print("[3/3] Building stratified PPC grids for both models...")
    idata_binary = az.from_netcdf(RESULTS_DIR / "binary_anchored.nc")
    _, builder_binary, processor_binary = _build_context("binary")

    ppc_binary = per_expert_system_ppc_multisystem(
        idata_binary, builder_binary, processor_binary, n_draws=500, seed=0
    )
    ppc_three = per_expert_system_ppc_multisystem(
        idata_three, builder_three, processor_three, n_draws=500, seed=0
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_b, _ = plot_per_expert_system_histograms(
        ppc_binary,
        K=7,
        system_display=SYSTEM_DISPLAY,
        system_order=[s for s, _ in SYSTEM_CONFIGS_VALIDATED],
        expert_order=processor_binary.expert_names,
        title="Stratified per-(expert, system) PPC — binary baseline",
    )
    fig_b.savefig(FIG_DIR / "gwt_ppc_binary.png", dpi=150, bbox_inches="tight")
    plt.close(fig_b)
    fig_t, _ = plot_per_expert_system_histograms(
        ppc_three,
        K=7,
        system_display=SYSTEM_DISPLAY,
        system_order=[s for s, _ in SYSTEM_CONFIGS_VALIDATED],
        expert_order=processor_three.expert_names,
        title="Stratified per-(expert, system) PPC — three-state leaf",
    )
    fig_t.savefig(FIG_DIR / "gwt_ppc_three_state.png", dpi=150, bbox_inches="tight")
    plt.close(fig_t)
    print(f"      saved PPC grids to {FIG_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
