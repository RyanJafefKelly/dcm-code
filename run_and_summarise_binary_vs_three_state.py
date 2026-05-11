"""Re-fit binary + three-state GWT and write a full diagnostic summary.

The nbconvert run of notebook 16 completed successfully but lost cell-8's
C-comparison print output to an IOPub timeout, and the notebook does not
currently call the diagnostic / posterior-summary helpers. This script
rectifies both by fitting once more under the same validated configuration
and persisting idata + summary numbers to disk.

Outputs land in ``results/gwt_binary_three_state/``:

    binary_anchored.nc           -- binary baseline posterior (arviz netcdf)
    three_state_anchored.nc      -- three-state leaf posterior
    summary.md                   -- 7-point text summary and the 5 acceptance flags

Run time is comparable to the original (~75 min binary + ~75-90 min three-state).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import arviz as az
import numpy as np

from dcm_ppc import (
    per_expert_system_ppc_multisystem,
    per_expert_system_tree_implied_ppc_multisystem,
)
from gwt_three_state_indicator_analysis import (
    FOCUS_KEYS,
    SYSTEM_CONFIGS_VALIDATED,
    SYSTEM_DISPLAY,
    SYSTEM_ORDER,
    compare_c_summaries,
    emission_separation_l1_from_posterior,
    fit_gwt,
    format_c_comparison_table,
    format_regime_table,
    middle_state_utilisation_per_cell,
    regime_rows_for_focus_cells,
)

OUT_DIR = Path("results/gwt_binary_three_state")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def diagnostics_overview(idata: Any) -> Dict[str, Any]:
    """Divergences, max R-hat, min ESS across core observation / system-C variables."""
    post = idata.posterior
    names = ["a", "kappa"]
    names.extend(
        v
        for v in post.data_vars
        if v.endswith("_C") and not v.endswith("_beta")
    )
    # Exclude anchored (near-zero-variance) C variables to avoid NaN R-hat on constants.
    free_names = []
    for v in names:
        if v in ("a", "kappa"):
            free_names.append(v)
            continue
        draws = np.asarray(post[v].values).reshape(-1)
        if np.std(draws) > 1e-10:
            free_names.append(v)
    diag = az.summary(idata, var_names=free_names, kind="diagnostics")
    return {
        "divergences": int(idata.sample_stats["diverging"].values.sum()),
        "max_rhat": float(diag["r_hat"].max()),
        "min_ess_bulk": float(diag["ess_bulk"].min()),
        "min_ess_tail": float(diag["ess_tail"].min()),
    }


def a_kappa_summary(idata: Any) -> Dict[str, Any]:
    """Posterior median and 94% interval for a and each kappa element."""
    post = idata.posterior
    a = np.asarray(post["a"].values).reshape(-1)
    kappa = np.asarray(post["kappa"].values).reshape(-1, post["kappa"].shape[-1])
    return {
        "a_median": float(np.median(a)),
        "a_lo": float(np.percentile(a, 3)),
        "a_hi": float(np.percentile(a, 97)),
        "kappa_median": kappa.mean(axis=0).tolist(),  # column means/medians close
        "kappa_medians": np.median(kappa, axis=0).tolist(),
        "kappa_lo": np.percentile(kappa, 3, axis=0).tolist(),
        "kappa_hi": np.percentile(kappa, 97, axis=0).tolist(),
    }


def focus_cell_ppc_summary(
    idata: Any,
    builder: Any,
    processor: Any,
    focus_keys: Sequence[Tuple[str, str]] = tuple(FOCUS_KEYS),
    n_draws: int = 500,
) -> Dict[str, Dict[Tuple[str, str], Dict[str, float]]]:
    """Leaf-updated and tree-implied PPCs at focus cells."""
    leaf = per_expert_system_ppc_multisystem(
        idata, builder, processor, n_draws=n_draws, seed=0
    )
    tree = per_expert_system_tree_implied_ppc_multisystem(
        idata, builder, processor, n_draws=n_draws, seed=0
    )

    def _pluck(ppc, key):
        r = ppc.get(key)
        if r is None:
            return None
        return {
            "obs_mean": r["obs_mean"],
            "pred_mean_mean": r["pred_mean_mean"],
            "obs_left": r["obs_left"],
            "pred_left_mean": r["pred_left_mean"],
            "delta_left": r["delta_left_mean"],
            "obs_right": r["obs_right"],
            "pred_right_mean": r["pred_right_mean"],
            "delta_right": r["delta_right_mean"],
            "obs_mid": r["obs_mid"],
            "pred_mid_mean": r["pred_mid_mean"],
            "delta_mid": r["delta_mid_mean"],
            "n_obs": r["n_obs"],
        }

    return {
        "leaf_updated": {key: _pluck(leaf, key) for key in focus_keys},
        "tree_implied": {key: _pluck(tree, key) for key in focus_keys},
    }


# ---------------------------------------------------------------------------
# Flag evaluation
# ---------------------------------------------------------------------------


def evaluate_flags(
    c_compare: Dict[str, Dict[str, float]],
    focus_binary_leaf: Dict[Tuple[str, str], Dict[str, float]],
    focus_three_leaf: Dict[Tuple[str, str], Dict[str, float]],
    a_binary: Dict[str, Any],
    a_three: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Evaluate the five pre-registered flags against computed numbers."""
    flags: List[Dict[str, Any]] = []

    # Flag 1: Chicken / LLM medians stay within ~0.05 of binary baseline.
    dc = abs(c_compare["Chicken"]["delta_median"])
    dl = abs(c_compare["2024 Leading Chat LLMs"]["delta_median"])
    flags.append(
        {
            "name": "Chicken/LLM stability (|delta| <= 0.05)",
            "pass": dc <= 0.05 and dl <= 0.05,
            "detail": f"|delta Chicken|={dc:.3f}, |delta LLM|={dl:.3f}",
        }
    )

    # Flag 2: Rater_B x ELIZA improves via cat-1 (Delta_left towards 0), not cat-7.
    b = focus_binary_leaf.get(("Rater_B", "ELIZA"))
    t = focus_three_leaf.get(("Rater_B", "ELIZA"))
    if b is None or t is None:
        flags.append(
            {"name": "Rater_B x ELIZA cat-1 improvement", "pass": None, "detail": "cell missing"}
        )
    else:
        dL_b = b["delta_left"]
        dL_t = t["delta_left"]
        dR_t = t["delta_right"]
        left_improved = abs(dL_t) < abs(dL_b)  # |delta_left| decreased
        right_stable = abs(dR_t) < 0.1  # model is not cat-7 spiking
        flags.append(
            {
                "name": "Rater_B x ELIZA cat-1 improvement (|delta_left| down, |delta_right| small)",
                "pass": left_improved and right_stable,
                "detail": (
                    f"|delta_left| {abs(dL_b):.3f} -> {abs(dL_t):.3f}; "
                    f"three-state delta_right={dR_t:+.3f}"
                ),
            }
        )

    # Flag 3: Rater_B x Human improves via cat-7 (|Delta_right| decreased), not cat-1.
    b = focus_binary_leaf.get(("Rater_B", "Human"))
    t = focus_three_leaf.get(("Rater_B", "Human"))
    if b is None or t is None:
        flags.append(
            {"name": "Rater_B x Human cat-7 improvement", "pass": None, "detail": "cell missing"}
        )
    else:
        dR_b = b["delta_right"]
        dR_t = t["delta_right"]
        dL_t = t["delta_left"]
        right_improved = abs(dR_t) < abs(dR_b)
        left_stable = abs(dL_t) < 0.1
        flags.append(
            {
                "name": "Rater_B x Human cat-7 improvement (|delta_right| down, |delta_left| small)",
                "pass": right_improved and left_stable,
                "detail": (
                    f"|delta_right| {abs(dR_b):.3f} -> {abs(dR_t):.3f}; "
                    f"three-state delta_left={dL_t:+.3f}"
                ),
            }
        )

    # Flag 4: Rater_E x Chicken and Rater_D x LLMs improve without wrong-tail spikes.
    detail_parts = []
    fail = False
    for cell_key in [("Rater_E", "Chicken"), ("Rater_D", "2024 Leading Chat LLMs")]:
        b = focus_binary_leaf.get(cell_key)
        t = focus_three_leaf.get(cell_key)
        if b is None or t is None:
            detail_parts.append(f"{cell_key[0]} x {cell_key[1]}: missing")
            continue
        # "Improve without wrong-tail" = overall fit better (mid-mass closer) AND no extreme delta_left/right
        mid_improved = abs(t["delta_mid"]) <= abs(b["delta_mid"]) + 0.03
        no_wrong_tail = abs(t["delta_left"]) < 0.2 and abs(t["delta_right"]) < 0.2
        ok = mid_improved and no_wrong_tail
        fail = fail or (not ok)
        detail_parts.append(
            f"{cell_key[0]} x {SYSTEM_DISPLAY.get(cell_key[1], cell_key[1])}: "
            f"delta_mid {b['delta_mid']:+.3f}->{t['delta_mid']:+.3f}, "
            f"three-state delta_left={t['delta_left']:+.3f}, delta_right={t['delta_right']:+.3f}"
        )
    flags.append(
        {
            "name": "Rater_E x Chicken + Rater_D x LLMs improve without wrong-tail spikes",
            "pass": not fail,
            "detail": " | ".join(detail_parts),
        }
    )

    # Flag 5: a inflation flag. Informative, not a hard test.
    a_b = a_binary["a_median"]
    a_t = a_three["a_median"]
    delta_a = a_t - a_b
    inflation_risk = delta_a > 0.3  # heuristic
    flags.append(
        {
            "name": "Three-state `a` inflation (informative; >0.3 rise flags response-style absorption risk)",
            "pass": not inflation_risk,
            "detail": (
                f"a_binary median={a_b:.3f} [{a_binary['a_lo']:.3f}, {a_binary['a_hi']:.3f}]; "
                f"a_three median={a_t:.3f} [{a_three['a_lo']:.3f}, {a_three['a_hi']:.3f}]; "
                f"delta={delta_a:+.3f}"
            ),
        }
    )

    return flags


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()

    print("[1/5] Fitting binary baseline...")
    binary = fit_gwt(state_model="binary")
    az.to_netcdf(binary["idata"], OUT_DIR / "binary_anchored.nc")
    print(f"      saved to {OUT_DIR / 'binary_anchored.nc'}")

    print("\n[2/5] Fitting three-state...")
    three_state = fit_gwt(state_model="three_state")
    az.to_netcdf(three_state["idata"], OUT_DIR / "three_state_anchored.nc")
    print(f"      saved to {OUT_DIR / 'three_state_anchored.nc'}")

    print("\n[3/5] Computing C comparison + diagnostics + a/kappa summaries...")
    c_cmp = compare_c_summaries(binary, three_state)
    diag_b = diagnostics_overview(binary["idata"])
    diag_t = diagnostics_overview(three_state["idata"])
    ak_b = a_kappa_summary(binary["idata"])
    ak_t = a_kappa_summary(three_state["idata"])

    print("\n[4/5] Computing focus-cell PPCs (leaf-updated + tree-implied) for both models...")
    ppc_b = focus_cell_ppc_summary(
        binary["idata"], binary["builder"], binary["processor"]
    )
    ppc_t = focus_cell_ppc_summary(
        three_state["idata"], three_state["builder"], three_state["processor"]
    )

    print("\n[5/5] Computing emission L1 + middle-state utilisation + focus-cell regimes...")
    emis = emission_separation_l1_from_posterior(three_state["idata"], threshold=0.1)
    util = middle_state_utilisation_per_cell(
        three_state["idata"], three_state["builder"], three_state["processor"]
    )
    util_vals = np.array([r["p_m1_mean"] for r in util])
    util_summary = {
        "n_indicators": len(util),
        "mean": float(util_vals.mean()),
        "median": float(np.median(util_vals)),
        "p03": float(np.percentile(util_vals, 3)),
        "p97": float(np.percentile(util_vals, 97)),
    }
    regime_rows_b = regime_rows_for_focus_cells(
        binary["idata"], binary["builder"], binary["processor"]
    )

    flags = evaluate_flags(
        c_cmp,
        ppc_b["leaf_updated"],
        ppc_t["leaf_updated"],
        ak_b,
        ak_t,
    )

    elapsed = time.time() - t0

    # --- Write summary markdown ---
    lines: List[str] = []
    lines.append("# Binary vs three-state GWT comparison\n")
    lines.append(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}, wall-time {elapsed:.0f}s.\n")

    lines.append("## 1. System-level C posteriors\n")
    lines.append("```")
    lines.append(format_c_comparison_table(c_cmp, guardrail=0.05))
    lines.append("```\n")

    lines.append("## 2. Convergence diagnostics\n")
    lines.append("| Model | Divergences | Max R-hat | Min ESS bulk | Min ESS tail |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| Binary | {diag_b['divergences']} | {diag_b['max_rhat']:.3f} | "
        f"{diag_b['min_ess_bulk']:.0f} | {diag_b['min_ess_tail']:.0f} |"
    )
    lines.append(
        f"| Three-state | {diag_t['divergences']} | {diag_t['max_rhat']:.3f} | "
        f"{diag_t['min_ess_bulk']:.0f} | {diag_t['min_ess_tail']:.0f} |"
    )
    lines.append("")

    lines.append("## 3. a and kappa posterior summaries\n")
    lines.append("### a (discrimination)")
    lines.append(
        f"- Binary: median {ak_b['a_median']:.3f}, 94% [{ak_b['a_lo']:.3f}, {ak_b['a_hi']:.3f}]"
    )
    lines.append(
        f"- Three-state: median {ak_t['a_median']:.3f}, 94% [{ak_t['a_lo']:.3f}, {ak_t['a_hi']:.3f}]"
    )
    lines.append("")
    lines.append("### kappa (ordered cutpoints)")
    lines.append("| Cutpoint | Binary median [94%] | Three-state median [94%] |")
    lines.append("|---|---|---|")
    for k in range(len(ak_b["kappa_medians"])):
        lines.append(
            f"| kappa_{k+1} | {ak_b['kappa_medians'][k]:.3f} "
            f"[{ak_b['kappa_lo'][k]:.3f}, {ak_b['kappa_hi'][k]:.3f}] | "
            f"{ak_t['kappa_medians'][k]:.3f} "
            f"[{ak_t['kappa_lo'][k]:.3f}, {ak_t['kappa_hi'][k]:.3f}] |"
        )
    lines.append("")

    lines.append("## 4. Signed tail PPC errors for focus cells (leaf-updated)\n")
    lines.append(
        "| Cell | Model | n | obs_left | pred_left | Delta_left "
        "| obs_right | pred_right | Delta_right | Delta_mid |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cell_key in FOCUS_KEYS:
        for label, ppc in [("binary", ppc_b["leaf_updated"]), ("three_state", ppc_t["leaf_updated"])]:
            r = ppc.get(cell_key)
            if r is None:
                continue
            cell_label = f"{cell_key[0]} × {SYSTEM_DISPLAY.get(cell_key[1], cell_key[1])}"
            lines.append(
                f"| {cell_label} | {label} | {r['n_obs']} | "
                f"{r['obs_left']:.3f} | {r['pred_left_mean']:.3f} | {r['delta_left']:+.3f} | "
                f"{r['obs_right']:.3f} | {r['pred_right_mean']:.3f} | {r['delta_right']:+.3f} | "
                f"{r['delta_mid']:+.3f} |"
            )
    lines.append("")

    lines.append("## 5. Tree-implied vs leaf-updated PPC at focus cells\n")
    lines.append(
        "| Cell | Model | Source | obs_mean | pred_mean | Delta_left | Delta_right | Delta_mid |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for cell_key in FOCUS_KEYS:
        cell_label = f"{cell_key[0]} × {SYSTEM_DISPLAY.get(cell_key[1], cell_key[1])}"
        for label, ppc_pair in [("binary", ppc_b), ("three_state", ppc_t)]:
            for src_name in ("leaf_updated", "tree_implied"):
                r = ppc_pair[src_name].get(cell_key)
                if r is None:
                    continue
                lines.append(
                    f"| {cell_label} | {label} | {src_name} | "
                    f"{r['obs_mean']:.3f} | {r['pred_mean_mean']:.3f} | "
                    f"{r['delta_left']:+.3f} | {r['delta_right']:+.3f} | {r['delta_mid']:+.3f} |"
                )
    lines.append("")

    lines.append("## 6. Emission-separation diagnostics (three-state)\n")
    lines.append("```")
    for k, v in emis.items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.4f}")
        else:
            lines.append(f"  {k}: {v}")
    lines.append("```\n")

    lines.append("## 7. Middle-state utilisation summary\n")
    lines.append(
        f"E[m=1 | ratings] across {util_summary['n_indicators']} indicators: "
        f"mean = {util_summary['mean']:.3f}, median = {util_summary['median']:.3f}, "
        f"94% = [{util_summary['p03']:.3f}, {util_summary['p97']:.3f}]"
    )
    lines.append("")

    lines.append("## Acceptance-criterion flags\n")
    for f in flags:
        status = "PASS" if f["pass"] is True else ("FAIL" if f["pass"] is False else "N/A")
        lines.append(f"- **[{status}] {f['name']}**: {f['detail']}")
    lines.append("")

    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print(f"\n[done] summary written to {OUT_DIR / 'summary.md'}  (wall-time {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
