"""Constrained-a three-state sensitivity fit.

Re-fits the three-state GWT model with a tighter prior on the discrimination
parameter `a` (HalfNormal(sigma=1.0) instead of the default 2.0). HalfNormal(1.0)
has 94% of its mass in [0, 2.2], roughly matching the binary baseline's
posterior range for `a` (median 1.913, 94% [1.489, 2.302]).

Diagnostic question
-------------------
Do the signed-tail PPC gains at the focus cells survive when `a` is prevented
from inflating to ~4.3? If they do, the three-state leaf's fit improvement
is genuinely about graded indicator presence. If they disappear, the improvement
was primarily latent-scale stretching (response-style absorption).

Outputs
-------
results/gwt_binary_three_state/three_state_a_constrained.nc
results/gwt_binary_three_state/analysis/constrained_a_summary.md
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import arviz as az
import numpy as np

from dcm_ppc import per_expert_system_ppc_multisystem
from gwt_three_state_indicator_analysis import (
    FOCUS_KEYS,
    SYSTEM_CONFIGS_VALIDATED,
    SYSTEM_DISPLAY,
    emission_separation_l1_from_posterior,
    fit_gwt,
)

OUT_DIR = Path("results/gwt_binary_three_state")
ANALYSIS_DIR = OUT_DIR / "analysis"


def focus_cell_ppc_rows(
    idata: Any,
    builder: Any,
    processor: Any,
    n_draws: int = 500,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    ppc = per_expert_system_ppc_multisystem(
        idata, builder, processor, n_draws=n_draws, seed=0
    )
    out = {}
    for key in FOCUS_KEYS:
        r = ppc.get(key)
        if r is None:
            continue
        out[key] = {
            "n_obs": r["n_obs"],
            "obs_mean": r["obs_mean"],
            "pred_mean": r["pred_mean_mean"],
            "obs_left": r["obs_left"],
            "pred_left": r["pred_left_mean"],
            "delta_left": r["delta_left_mean"],
            "obs_right": r["obs_right"],
            "pred_right": r["pred_right_mean"],
            "delta_right": r["delta_right_mean"],
            "delta_mid": r["delta_mid_mean"],
        }
    return out


def main() -> None:
    t0 = time.time()

    print("Fitting three-state with A_PRIOR_SIGMA=1.0 (constrained)...")
    result = fit_gwt(
        state_model="three_state",
        fit_overrides={"A_PRIOR_SIGMA": 1.0},
    )
    az.to_netcdf(result["idata"], OUT_DIR / "three_state_a_constrained.nc")

    idata = result["idata"]
    a_draws = np.asarray(idata.posterior["a"].values).reshape(-1)
    a_median = float(np.median(a_draws))
    a_lo = float(np.percentile(a_draws, 3))
    a_hi = float(np.percentile(a_draws, 97))

    kappa_draws = np.asarray(idata.posterior["kappa"].values).reshape(-1, idata.posterior["kappa"].shape[-1])
    kappa_medians = np.median(kappa_draws, axis=0).tolist()

    ppc_focus = focus_cell_ppc_rows(
        idata, result["builder"], result["processor"]
    )

    # C summary
    c_summary = result["c_summary"]

    # Emission separation
    emis = emission_separation_l1_from_posterior(idata, threshold=0.1)

    # Compare focus-cell to the unconstrained three-state (from persisted idata)
    uncons_path = OUT_DIR / "three_state_anchored.nc"
    ppc_uncons: Dict[Tuple[str, str], Dict[str, float]] = {}
    a_uncons = float("nan")
    if uncons_path.exists():
        from dcm_model import (
            EvidenceProcessor,
            ModelConfig,
            MultiSystemDataProcessor,
            MultiSystemModelBuilder,
            load_data,
        )
        from gwt_three_state_indicator_analysis import STANCE

        cfg_un = ModelConfig(
            USE_EXPERT_SHIFTS=False,
            USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
            INDICATOR_STATE_MODEL="three_state",
        )
        stance_data = next(item for item in load_data(cfg_un) if item["name"] == STANCE)
        proc_un = MultiSystemDataProcessor(cfg_un)
        proc_un.process(stance_data, [s for s, _ in SYSTEM_CONFIGS_VALIDATED])
        bld_un = MultiSystemModelBuilder(cfg_un, EvidenceProcessor(cfg_un), proc_un, list(SYSTEM_CONFIGS_VALIDATED))
        bld_un.build_model(stance_data)
        idata_un = az.from_netcdf(uncons_path)
        a_un_draws = np.asarray(idata_un.posterior["a"].values).reshape(-1)
        a_uncons = float(np.median(a_un_draws))
        ppc_uncons = focus_cell_ppc_rows(idata_un, bld_un, proc_un)

    # Binary reference
    bin_path = OUT_DIR / "binary_anchored.nc"
    ppc_binary: Dict[Tuple[str, str], Dict[str, float]] = {}
    a_binary = float("nan")
    if bin_path.exists():
        from dcm_model import (
            EvidenceProcessor,
            ModelConfig,
            MultiSystemDataProcessor,
            MultiSystemModelBuilder,
            load_data,
        )
        from gwt_three_state_indicator_analysis import STANCE

        cfg_b = ModelConfig(
            USE_EXPERT_SHIFTS=False,
            USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
            INDICATOR_STATE_MODEL="binary",
        )
        stance_data = next(item for item in load_data(cfg_b) if item["name"] == STANCE)
        proc_b = MultiSystemDataProcessor(cfg_b)
        proc_b.process(stance_data, [s for s, _ in SYSTEM_CONFIGS_VALIDATED])
        bld_b = MultiSystemModelBuilder(cfg_b, EvidenceProcessor(cfg_b), proc_b, list(SYSTEM_CONFIGS_VALIDATED))
        bld_b.build_model(stance_data)
        idata_b = az.from_netcdf(bin_path)
        a_b_draws = np.asarray(idata_b.posterior["a"].values).reshape(-1)
        a_binary = float(np.median(a_b_draws))
        ppc_binary = focus_cell_ppc_rows(idata_b, bld_b, proc_b)

    elapsed = time.time() - t0

    # --- write summary markdown ---
    lines = []
    lines.append("# Constrained-a three-state sensitivity\n")
    lines.append(
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}, wall-time {elapsed:.0f}s. "
        f"Prior: `a ~ HalfNormal(sigma=1.0)` (94% mass in ~[0, 2.2]).\n"
    )

    lines.append("## a and kappa summary\n")
    lines.append(
        f"- a (constrained three-state): median {a_median:.3f}, "
        f"94% [{a_lo:.3f}, {a_hi:.3f}]"
    )
    lines.append(f"- a (unconstrained three-state, persisted): median {a_uncons:.3f}")
    lines.append(f"- a (binary baseline, persisted): median {a_binary:.3f}")
    lines.append(
        "- kappa medians (constrained): "
        + ", ".join(f"{k:.3f}" for k in kappa_medians)
    )
    lines.append("")

    lines.append("## System-level C (constrained three-state)\n")
    lines.append("| System | median | 94% interval | fixed |")
    lines.append("|---|---:|---|:---:|")
    for sys_name, _ in SYSTEM_CONFIGS_VALIDATED:
        row = c_summary[sys_name]
        lines.append(
            f"| {SYSTEM_DISPLAY.get(sys_name, sys_name)} | "
            f"{row['median']:.3f} | "
            f"[{row['lo']:.3f}, {row['hi']:.3f}] | "
            f"{'yes' if row['fixed'] else 'no'} |"
        )
    lines.append("")

    lines.append("## Focus-cell signed tail errors\n")
    lines.append(
        "| Cell | Model | pred_left | Δleft | pred_right | Δright | Δmid |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for key in FOCUS_KEYS:
        name = f"{key[0]} × {SYSTEM_DISPLAY.get(key[1], key[1])}"
        for label, src in (
            ("binary", ppc_binary),
            ("3-state (free a)", ppc_uncons),
            ("3-state (a constrained)", ppc_focus),
        ):
            r = src.get(key)
            if r is None:
                continue
            lines.append(
                f"| {name} | {label} | {r['pred_left']:.3f} | "
                f"{r['delta_left']:+.3f} | {r['pred_right']:.3f} | "
                f"{r['delta_right']:+.3f} | {r['delta_mid']:+.3f} |"
            )
    lines.append("")

    lines.append("## Emission-separation L1 (constrained three-state)\n")
    lines.append("```")
    for k, v in emis.items():
        lines.append(f"  {k}: {v}" if not isinstance(v, float) else f"  {k}: {v:.4f}")
    lines.append("```\n")

    lines.append("## Quick read\n")
    delta_a = a_median - a_binary
    if delta_a < 0.5:
        lines.append(
            f"- a stayed close to the binary range (delta={delta_a:+.3f}). "
            "If focus-cell Δleft/Δright improvements are preserved, the three-state "
            "improvement is not primarily about latent-scale inflation."
        )
    else:
        lines.append(
            f"- a still shifted upward (delta={delta_a:+.3f}). "
            "Tightening the prior did not fully suppress the inflation; either the prior "
            "needs to be tighter or the model really wants the extra scale for its "
            "emission components."
        )
    lines.append("")

    (ANALYSIS_DIR / "constrained_a_summary.md").write_text("\n".join(lines))
    print(
        f"\n[done] constrained-a three-state fit complete; summary at "
        f"{ANALYSIS_DIR / 'constrained_a_summary.md'}  (wall-time {elapsed:.0f}s)"
    )


if __name__ == "__main__":
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    main()
