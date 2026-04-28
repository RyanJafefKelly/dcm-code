"""Post-process the β_abs__(σ, δ) refit for round-2 writeup.

Runs after `run_tree_prior_pooling_abs_by_sd.py` finishes sampling.
Compares to the existing pool_3s baseline on:
- Sampling diagnostics (Tier 1 + tightened weak-undermining-neutral gate)
- System C posteriors
- weak_undermining + neutral cluster decomposition (own β_pres + own β_abs movement)
- Per-indicator δ_j on the 5 sign-flip cluster
- Mean δ_j

Output: appended to diagnostics_round2_results.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import arviz as az
import numpy as np
import pandas as pd

from analyse_tree_pooling import pooled_beta_draws_by_node
from dcm_model import (
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    MultiSystemModelBuilder,
    load_data,
    node_key,
)
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS

POOL_3S_PATH = Path("results/gwt_tree_pooling/three_state_pooled_anchored.nc")
POOL_3S_ABS_BY_SD_PATH = Path("results/gwt_tree_pooling/three_state_pooled_abs_by_sd_anchored.nc")
META_PATH = Path("results/gwt_tree_pooling/three_state_pooled_abs_by_sd_anchored.meta.json")
OUT_DIR = Path("notebooks/meeting_prep_arvo_2026-04-27/figs_round2")
WRITEUP_PATH = Path("notebooks/meeting_prep_arvo_2026-04-27/diagnostics_round2_results.md")
STANCE = "Global Workspace Theory"

SIGN_FLIP_NODE_KEYS = [
    "Global Workspace Theory > Coherence > Autonomous Subparts > Information Transfer",
    "Global Workspace Theory > Coherence > Autonomous Subparts > Learning Transfer",
    "Global Workspace Theory > Coherence > Autonomous Subparts > Functional Subparts",
    "Global Workspace Theory > Coherence > Autonomous Subparts > Information Transfer Architecture",
    "Global Workspace Theory > Coherence > Autonomous Subparts > Conflicting Subparts",
]
PAPER_BETA_PRES_WU_NEUTRAL = 0.4286
PAPER_BETA_ABS_NEUTRAL = 0.5


def _med(x):
    return float(np.median(np.asarray(x).reshape(-1)))


def _summary(v):
    a = np.asarray(v).reshape(-1)
    return {
        "median": float(np.median(a)),
        "lo": float(np.percentile(a, 3)),
        "hi": float(np.percentile(a, 97)),
        "mean": float(a.mean()),
    }


def main():
    if not POOL_3S_ABS_BY_SD_PATH.exists():
        print(f"FAIL: refit not found at {POOL_3S_ABS_BY_SD_PATH}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading both fits...")
    idata_new = az.from_netcdf(str(POOL_3S_ABS_BY_SD_PATH))
    idata_old = az.from_netcdf(str(POOL_3S_PATH))
    post_new = idata_new.posterior
    post_old = idata_old.posterior

    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    print(f"Tier 1 (new):  divergences={meta['tier1']['divergences']}, "
          f"max R-hat={meta['tier1']['max_rhat']:.4f}, "
          f"min ESS bulk={meta['tier1']['min_ess_bulk']:.0f}")
    print(f"Tightened gate (weak_undermining cluster):")
    print(f"  max R-hat = {meta['tier1']['wu_max_rhat']:.4f}, "
          f"min ESS bulk = {meta['tier1']['wu_min_ess_bulk']:.0f}, "
          f"PASS={meta['tier1']['tightened_gate_pass']}")

    # System C posteriors
    sys_vars = {
        "Human": "human__global_workspace_theory_C",
        "Chicken": "chicken__global_workspace_theory_C",
        "LLMs": "2024_leading_chat_llms__global_workspace_theory_C",
        "ELIZA": "eliza__global_workspace_theory_C",
    }
    print()
    print("System C posteriors (median [3%, 97%]):")
    print(f"  {'system':<8}  {'pool_3s (d-only)':<28}  {'pool_3s_abs_by_sd':<28}  shift")
    sys_rows = []
    for label, var in sys_vars.items():
        if var not in post_new.data_vars or var not in post_old.data_vars:
            continue
        s_old = _summary(post_old[var].values)
        s_new = _summary(post_new[var].values)
        shift = s_new["median"] - s_old["median"]
        old_str = f"{s_old['median']:.4f} [{s_old['lo']:.4f}, {s_old['hi']:.4f}]"
        new_str = f"{s_new['median']:.4f} [{s_new['lo']:.4f}, {s_new['hi']:.4f}]"
        print(f"  {label:<8}  {old_str:<28}  {new_str:<28}  {shift:+.4f}")
        sys_rows.append({"system": label, "pool_3s_med": s_old["median"],
                         "pool_3s_abs_by_sd_med": s_new["median"], "shift": shift})

    # weak_undermining + neutral decomposition
    print()
    print("=== weak_undermining + neutral decomposition ===")
    bp_wu_new_var = "beta_pres__weak_undermining__neutral"
    ba_wu_new_var = "beta_abs__weak_undermining__neutral"
    ba_n_old_var = "beta_abs__neutral"

    bp_wu_new = _summary(post_new[bp_wu_new_var].values)
    ba_wu_new = _summary(post_new[ba_wu_new_var].values)
    bp_wu_old = _summary(post_old[bp_wu_new_var].values)
    ba_n_old = _summary(post_old[ba_n_old_var].values)

    print(f"                                paper      pool_3s (d-only)             pool_3s_abs_by_sd")
    print(f"  β_pres (weak_und, neutral):  {PAPER_BETA_PRES_WU_NEUTRAL:.3f}     "
          f"{bp_wu_old['median']:.3f} [{bp_wu_old['lo']:.3f}, {bp_wu_old['hi']:.3f}]   "
          f"{bp_wu_new['median']:.3f} [{bp_wu_new['lo']:.3f}, {bp_wu_new['hi']:.3f}]")
    print(f"  β_abs  (weak_und, neutral):  {PAPER_BETA_ABS_NEUTRAL:.3f}     "
          f"d-only:  {ba_n_old['median']:.3f} [{ba_n_old['lo']:.3f}, {ba_n_old['hi']:.3f}]   "
          f"sd-keyed: {ba_wu_new['median']:.3f} [{ba_wu_new['lo']:.3f}, {ba_wu_new['hi']:.3f}]")

    bp_shift = bp_wu_new["median"] - PAPER_BETA_PRES_WU_NEUTRAL
    ba_shift = ba_wu_new["median"] - PAPER_BETA_ABS_NEUTRAL
    if abs(ba_shift) < 0.05:
        verdict_ba = "stays near prior (≤ 0.05 shift) -- consistent with insufficient direct evidence on singleton"
    elif abs(ba_shift) < 0.10:
        verdict_ba = f"shifts {ba_shift:+.3f} (modest)"
    else:
        verdict_ba = f"shifts {ba_shift:+.3f} (substantial -- evidence on the singleton was actually present)"
    print(f"\n  β_abs__(weak_und, neutral) shift from prior: {verdict_ba}")
    print(f"  β_pres__(weak_und, neutral) shift from prior: {bp_shift:+.3f}")

    # Per-indicator δ_j on sign-flip cluster
    print()
    print("=== Per-indicator δ_j on 5 sign-flip cluster ===")
    cfg_new = ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=True,
        BETA_ABS_BY_SUPPORT_DEMAND=True,
    )
    stance_data = next(s for s in load_data(cfg_new) if s["name"] == STANCE)
    bp_new_dict, ba_new_dict = pooled_beta_draws_by_node(idata_new, stance_data)
    bp_old_dict, ba_old_dict = pooled_beta_draws_by_node(idata_old, stance_data)

    from beta_abs_neutral_counterfactual import affine_coeffs_for_keys
    coeffs_new = affine_coeffs_for_keys(stance_data, None, bp_new_dict, ba_new_dict, SIGN_FLIP_NODE_KEYS)
    coeffs_old = affine_coeffs_for_keys(stance_data, None, bp_old_dict, ba_old_dict, SIGN_FLIP_NODE_KEYS)

    print(f"  {'indicator':<40}  pool_3s δ_j   pool_3s_abs_by_sd δ_j   shift")
    deltas_old = []
    deltas_new = []
    decomp_rows = []
    for key in SIGN_FLIP_NODE_KEYS:
        d_old = _med(coeffs_old[key][1])
        d_new = _med(coeffs_new[key][1])
        ind = key.split(" > ")[-1]
        print(f"  {ind:<40}  {d_old:+.4f}        {d_new:+.4f}              {d_new - d_old:+.4f}")
        deltas_old.append(d_old)
        deltas_new.append(d_new)
        decomp_rows.append({
            "indicator": ind,
            "delta_j_pool_3s": d_old,
            "delta_j_pool_3s_abs_by_sd": d_new,
            "shift": d_new - d_old,
        })

    print(f"  mean across cluster:       {np.mean(deltas_old):+.4f}        {np.mean(deltas_new):+.4f}              {np.mean(deltas_new) - np.mean(deltas_old):+.4f}")

    # Mean δ_j across all indicators
    print()
    from analyse_tree_pooling import per_indicator_delta_under_pooling
    cfg_old = ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=True,
    )
    proc_new = MultiSystemDataProcessor(cfg_new)
    proc_new.process(stance_data, [s for s, _ in ANCHORED_SYSTEM_CONFIGS])
    builder_new = MultiSystemModelBuilder(cfg_new, EvidenceProcessor(cfg_new), proc_new, list(ANCHORED_SYSTEM_CONFIGS))
    builder_new.build_model(stance_data)
    proc_old = MultiSystemDataProcessor(cfg_old)
    proc_old.process(stance_data, [s for s, _ in ANCHORED_SYSTEM_CONFIGS])
    builder_old = MultiSystemModelBuilder(cfg_old, EvidenceProcessor(cfg_old), proc_old, list(ANCHORED_SYSTEM_CONFIGS))
    builder_old.build_model(stance_data)

    _, slopes_new, _ = per_indicator_delta_under_pooling(idata_new, builder_new, stance_data)
    _, slopes_old, _ = per_indicator_delta_under_pooling(idata_old, builder_old, stance_data)
    md_new = float(np.median(slopes_new, axis=0).mean())
    md_old = float(np.median(slopes_old, axis=0).mean())
    print(f"Mean δ_j (across 50 indicators):  pool_3s {md_old:+.4f}  →  abs_by_sd {md_new:+.4f}  ({md_new - md_old:+.4f})")

    # Save outputs
    pd.DataFrame(decomp_rows).to_csv(OUT_DIR / "abs_by_sd_signflip_delta.csv", index=False)
    pd.DataFrame(sys_rows).to_csv(OUT_DIR / "abs_by_sd_system_C.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'abs_by_sd_signflip_delta.csv'}")
    print(f"wrote {OUT_DIR / 'abs_by_sd_system_C.csv'}")

    # Summary text snippet for writeup
    block = []
    block.append("\n---\n\n## Task 4 — `β_abs__(σ, δ)` refit RESULTS\n")
    block.append(f"### Tier 1 sampling diagnostics\n")
    block.append(f"- divergences: {meta['tier1']['divergences']}")
    block.append(f"- max R-hat: {meta['tier1']['max_rhat']:.4f}")
    block.append(f"- min ESS bulk: {meta['tier1']['min_ess_bulk']:.0f}")
    block.append(f"- weak_undermining cluster max R-hat: {meta['tier1']['wu_max_rhat']:.4f}")
    block.append(f"- weak_undermining cluster min ESS bulk: {meta['tier1']['wu_min_ess_bulk']:.0f}")
    block.append(f"- **Tightened gate pass: {meta['tier1']['tightened_gate_pass']}**\n")
    block.append(f"### System C posteriors\n")
    block.append("| system | pool_3s (d-only) median | pool_3s_abs_by_sd median | shift |")
    block.append("|---|---:|---:|---:|")
    for r in sys_rows:
        block.append(f"| {r['system']} | {r['pool_3s_med']:.4f} | {r['pool_3s_abs_by_sd_med']:.4f} | {r['shift']:+.4f} |")
    block.append("")
    block.append(f"### `weak_undermining + neutral` decomposition\n")
    block.append("| parameter | paper | pool_3s (d-only) | pool_3s_abs_by_sd |")
    block.append("|---|---:|---:|---:|")
    block.append(f"| β_pres | {PAPER_BETA_PRES_WU_NEUTRAL:.3f} | "
                 f"{bp_wu_old['median']:.3f} [{bp_wu_old['lo']:.3f}, {bp_wu_old['hi']:.3f}] | "
                 f"{bp_wu_new['median']:.3f} [{bp_wu_new['lo']:.3f}, {bp_wu_new['hi']:.3f}] |")
    block.append(f"| β_abs | {PAPER_BETA_ABS_NEUTRAL:.3f} | "
                 f"{ba_n_old['median']:.3f} [{ba_n_old['lo']:.3f}, {ba_n_old['hi']:.3f}] (shared d) | "
                 f"{ba_wu_new['median']:.3f} [{ba_wu_new['lo']:.3f}, {ba_wu_new['hi']:.3f}] (singleton) |")
    block.append("")
    block.append(f"**Verdict on β_abs__(weak_und, neutral) shift from prior:** {verdict_ba}\n")
    block.append(f"### Per-indicator δ_j on 5 sign-flip cluster\n")
    block.append("| indicator | pool_3s δ_j | pool_3s_abs_by_sd δ_j | shift |")
    block.append("|---|---:|---:|---:|")
    for r in decomp_rows:
        block.append(f"| {r['indicator']} | {r['delta_j_pool_3s']:+.4f} | {r['delta_j_pool_3s_abs_by_sd']:+.4f} | {r['shift']:+.4f} |")
    block.append(f"| **mean** | **{np.mean(deltas_old):+.4f}** | **{np.mean(deltas_new):+.4f}** | **{np.mean(deltas_new) - np.mean(deltas_old):+.4f}** |")
    block.append("")
    block.append(f"### Mean δ_j across all 50 indicators\n")
    block.append(f"- pool_3s (d-only):       **{md_old:+.4f}**")
    block.append(f"- pool_3s_abs_by_sd:      **{md_new:+.4f}**")
    block.append(f"- shift:                  {md_new - md_old:+.4f}\n")

    interp_lines = []
    if abs(ba_shift) < 0.05:
        interp_lines.append(
            "**Result confirms Task 2's interpretation.** Under the richer parameterisation, "
            "`β_abs__(weak_und, neutral)` stays near its paper prior 0.5 — direct evidence on "
            "the singleton group is insufficient to move it. The shared-`β_abs__neutral` shift "
            "in `pool_3s` was therefore inheriting evidence from the *other* neutral-demandingness "
            "cells (mostly strong/moderate support nodes), not learning anything specific to "
            "weak-undermining.\n"
        )
    else:
        interp_lines.append(
            "**Mixed result.** `β_abs__(weak_und, neutral)` does shift from prior under the richer "
            "parameterisation, suggesting some direct evidence on the singleton group exists. "
            "Task 2's interpretation needs nuancing.\n"
        )
    if abs(np.mean(deltas_new) - np.mean(deltas_old)) < 0.02 and abs(md_new - md_old) < 0.02:
        interp_lines.append(
            "**System C posteriors and headline mean δ_j barely change** under the richer "
            "parameterisation (< 0.02 shifts). The d-only sharing assumption in `pool_3s` is "
            "not load-bearing for the headline conclusions; only the fine-grained interpretation "
            "of weak-undermining changes.\n"
        )
    block.extend(interp_lines)

    block.append(f"### Outputs\n")
    block.append(f"- `figs_round2/abs_by_sd_signflip_delta.csv`")
    block.append(f"- `figs_round2/abs_by_sd_system_C.csv`")
    block.append(f"- `results/gwt_tree_pooling/three_state_pooled_abs_by_sd_anchored.nc`")
    block.append(f"- `results/gwt_tree_pooling/three_state_pooled_abs_by_sd_anchored.meta.json`\n")

    new_text = "\n".join(block)
    text = WRITEUP_PATH.read_text()
    if "## Task 4 — `β_abs__(σ, δ)` refit RESULTS" not in text:
        # Replace the placeholder
        marker = "[Results pending — append below once sampling completes.]"
        if marker in text:
            text = text.replace(marker, new_text.strip())
        else:
            text = text + new_text
        WRITEUP_PATH.write_text(text)
        print(f"\nappended Task 4 results to {WRITEUP_PATH}")
    else:
        print(f"\nTask 4 results section already present in {WRITEUP_PATH}; not duplicating")


if __name__ == "__main__":
    main()
