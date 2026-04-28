"""A.2 + A.3: PPC closure + sign uncertainty for `pool_3s_abs_by_sd`.

A.2: Leaf-updated PPC closure on focus + guardrail cells (matching B.10).
     With posterior intervals + comparison to baseline_3s and pool_3s.

A.3: Sign uncertainty + practical-equivalence stats for the weak-undermining
     cluster under `pool_3s_abs_by_sd` and `pool_3s`.

Outputs:
- figs_round2/pool_3s_abs_by_sd_ppc_focus_guardrail.csv
- figs_round2/weak_undermining_cluster_sign_uncertainty.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import arviz as az
import numpy as np
import pandas as pd

from analyse_tree_pooling import per_indicator_delta_under_pooling
from dcm_model import (
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    MultiSystemModelBuilder,
    load_data,
)
from dcm_ppc import per_expert_system_ppc_multisystem
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS
from tree_propagation_diagnostic import _EXPERT_ANON_MAP

OUT_DIR = Path("notebooks/meeting_prep_arvo_2026-04-27/figs_round2")
STANCE = "Global Workspace Theory"

FIT_PATHS = {
    "baseline_3s": ("results/gwt_binary_three_state/three_state_anchored.nc", {}),
    "pool_3s": ("results/gwt_tree_pooling/three_state_pooled_anchored.nc",
                {"POOL_BETAS_BY_LABEL": True}),
    "pool_3s_abs_by_sd": ("results/gwt_tree_pooling/three_state_pooled_abs_by_sd_anchored.nc",
                          {"POOL_BETAS_BY_LABEL": True, "BETA_ABS_BY_SUPPORT_DEMAND": True}),
}

# Real names ↔ anonymised codes
ANON = dict(_EXPERT_ANON_MAP)
REAL_BY_ANON = {v: k for k, v in ANON.items()}

FOCUS_CELLS = [
    ("E_cross", "Human", "right"),
    ("E_cross", "ELIZA", "left"),
    ("E_chickenB", "Chicken", "right"),
    ("E_llmC", "2024 Leading Chat LLMs", "right"),
]

SIGN_FLIP_NODE_KEYS = [
    "Global Workspace Theory > Coherence > Autonomous Subparts > Information Transfer",
    "Global Workspace Theory > Coherence > Autonomous Subparts > Learning Transfer",
    "Global Workspace Theory > Coherence > Autonomous Subparts > Functional Subparts",
    "Global Workspace Theory > Coherence > Autonomous Subparts > Information Transfer Architecture",
    "Global Workspace Theory > Coherence > Autonomous Subparts > Conflicting Subparts",
]


def load_fit(name: str):
    path, cfg_kwargs = FIT_PATHS[name]
    idata = az.from_netcdf(path)
    cfg = ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        **cfg_kwargs,
    )
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    proc = MultiSystemDataProcessor(cfg)
    proc.process(stance_data, [s for s, _ in ANCHORED_SYSTEM_CONFIGS])
    builder = MultiSystemModelBuilder(
        cfg, EvidenceProcessor(cfg), proc, list(ANCHORED_SYSTEM_CONFIGS)
    )
    builder.build_model(stance_data)
    return idata, cfg, stance_data, proc, builder


def run_ppc(name: str) -> Dict:
    print(f"\n=== {name} PPC ===")
    idata, cfg, stance_data, proc, builder = load_fit(name)
    stats = per_expert_system_ppc_multisystem(idata, builder, proc, n_draws=500, seed=42)
    rows = []
    for anon, sys_name, tail in FOCUS_CELLS:
        real = REAL_BY_ANON.get(anon)
        key = (real, sys_name) if real else None
        if key is None or key not in stats:
            print(f"  {anon} × {sys_name}: cell missing")
            continue
        s = stats[key]
        if tail == "left":
            obs = s["obs_left"]
            pred_med = s["pred_left_mean"]
            pred_lo = s["pred_left_lo"]
            pred_hi = s["pred_left_hi"]
        else:
            obs = s["obs_right"]
            pred_med = s["pred_right_mean"]
            pred_lo = s["pred_right_lo"]
            pred_hi = s["pred_right_hi"]
        delta = pred_med - obs
        rows.append({
            "fit": name,
            "cell": f"{anon} × {sys_name}",
            "tail": tail,
            "n_obs": s["n_obs"],
            "obs": obs,
            "pred_mean": pred_med,
            "pred_lo": pred_lo,
            "pred_hi": pred_hi,
            "delta": delta,
        })
        print(f"  {anon} × {sys_name} ({tail}): obs={obs:.3f}  "
              f"pred={pred_med:.3f} [{pred_lo:.3f}, {pred_hi:.3f}]  Δ={delta:+.3f}")

    # Headline C posteriors and mean δ_j (already computed elsewhere; recompute for table)
    c_chicken = float(np.median(np.asarray(
        idata.posterior["chicken__global_workspace_theory_C"].values).reshape(-1)))
    c_llms = float(np.median(np.asarray(
        idata.posterior["2024_leading_chat_llms__global_workspace_theory_C"].values).reshape(-1)))
    return {
        "rows": rows,
        "c_chicken": c_chicken,
        "c_llms": c_llms,
    }


def closure_table(ppc_results: Dict[str, Dict]) -> pd.DataFrame:
    """Cross-fit closure table: focus and guardrail cells per fit."""
    all_rows = []
    for fit_name, res in ppc_results.items():
        for r in res["rows"]:
            all_rows.append(r)
    df = pd.DataFrame(all_rows)
    return df


def closure_relative(ppc_results: Dict[str, Dict]) -> pd.DataFrame:
    """For each cell, compute |Δ| closure under pool_3s_abs_by_sd vs baseline_3s."""
    rows = []
    base_lookup = {}
    for r in ppc_results["baseline_3s"]["rows"]:
        base_lookup[(r["cell"], r["tail"])] = r

    for fit_name in ["pool_3s", "pool_3s_abs_by_sd"]:
        for r in ppc_results[fit_name]["rows"]:
            base = base_lookup.get((r["cell"], r["tail"]))
            if base is None:
                continue
            base_abs = abs(base["delta"])
            new_abs = abs(r["delta"])
            closure_pct = (base_abs - new_abs) / max(base_abs, 1e-9) * 100
            rows.append({
                "fit": fit_name,
                "cell": r["cell"],
                "tail": r["tail"],
                "delta": r["delta"],
                "base_delta": base["delta"],
                "abs_closure_pct_vs_baseline_3s": closure_pct,
            })
    return pd.DataFrame(rows)


def sign_uncertainty(name: str) -> Dict:
    """A.3: sign + practical equivalence on the 5 sign-flip cluster."""
    print(f"\n=== {name} sign uncertainty ===")
    idata, cfg, stance_data, proc, builder = load_fit(name)
    intercepts, slopes, indicator_order = per_indicator_delta_under_pooling(
        idata, builder, stance_data
    )
    keys = [spec.node_key for spec in indicator_order]
    cluster_mask = np.array([k in SIGN_FLIP_NODE_KEYS for k in keys])
    cluster_slopes = slopes[:, cluster_mask]   # (S, 5)
    cluster_mean_per_draw = cluster_slopes.mean(axis=1)  # (S,)

    # Per-indicator
    rows = []
    for j, k in enumerate(keys):
        if not cluster_mask[j]:
            continue
        sj = slopes[:, j]
        ind_name = k.split(" > ")[-1]
        rows.append({
            "fit": name,
            "indicator": ind_name,
            "delta_j_med": float(np.median(sj)),
            "delta_j_lo": float(np.percentile(sj, 3)),
            "delta_j_hi": float(np.percentile(sj, 97)),
            "Pr_pos": float(np.mean(sj > 0)),
            "Pr_abs_lt_001": float(np.mean(np.abs(sj) < 0.01)),
            "Pr_abs_lt_002": float(np.mean(np.abs(sj) < 0.02)),
        })

    # Cluster-aggregate
    Pr_all_pos = float(np.mean((cluster_slopes > 0).all(axis=1)))
    cluster_row = {
        "fit": name,
        "indicator": "CLUSTER_MEAN",
        "delta_j_med": float(np.median(cluster_mean_per_draw)),
        "delta_j_lo": float(np.percentile(cluster_mean_per_draw, 3)),
        "delta_j_hi": float(np.percentile(cluster_mean_per_draw, 97)),
        "Pr_pos": float(np.mean(cluster_mean_per_draw > 0)),
        "Pr_abs_lt_001": float(np.mean(np.abs(cluster_mean_per_draw) < 0.01)),
        "Pr_abs_lt_002": float(np.mean(np.abs(cluster_mean_per_draw) < 0.02)),
        "Pr_all_pos": Pr_all_pos,
    }
    rows.append(cluster_row)

    print(f"  cluster mean δ_j: median {cluster_row['delta_j_med']:+.4f}  "
          f"94% [{cluster_row['delta_j_lo']:+.4f}, {cluster_row['delta_j_hi']:+.4f}]")
    print(f"    Pr(δ̄ > 0)        = {cluster_row['Pr_pos']:.3f}")
    print(f"    Pr(all 5 δ_j > 0) = {Pr_all_pos:.3f}")
    print(f"    Pr(|δ̄| < 0.01)   = {cluster_row['Pr_abs_lt_001']:.3f}")
    print(f"    Pr(|δ̄| < 0.02)   = {cluster_row['Pr_abs_lt_002']:.3f}")

    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- A.2 PPC ---
    ppc_results: Dict[str, Dict] = {}
    for name in ["baseline_3s", "pool_3s", "pool_3s_abs_by_sd"]:
        ppc_results[name] = run_ppc(name)

    df_closure = closure_table(ppc_results)
    df_closure.to_csv(OUT_DIR / "pool_3s_abs_by_sd_ppc_focus_guardrail.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'pool_3s_abs_by_sd_ppc_focus_guardrail.csv'}")

    df_rel = closure_relative(ppc_results)
    print()
    print("=== Closure (vs baseline_3s) ===")
    print(df_rel.to_string(index=False))
    df_rel.to_csv(OUT_DIR / "pool_3s_abs_by_sd_ppc_closure_vs_baseline.csv", index=False)
    print(f"wrote {OUT_DIR / 'pool_3s_abs_by_sd_ppc_closure_vs_baseline.csv'}")

    # --- A.3 Sign uncertainty ---
    sign_rows = []
    for name in ["pool_3s", "pool_3s_abs_by_sd"]:
        sign_rows.extend(sign_uncertainty(name))
    df_sign = pd.DataFrame(sign_rows)
    df_sign.to_csv(OUT_DIR / "weak_undermining_cluster_sign_uncertainty.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'weak_undermining_cluster_sign_uncertainty.csv'}")
    print()
    print("=== Sign uncertainty summary ===")
    print(df_sign[df_sign["indicator"] == "CLUSTER_MEAN"].to_string(index=False))


if __name__ == "__main__":
    main()
