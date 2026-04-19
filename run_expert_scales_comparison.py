"""Fit three-state + σ_e (primary) on the validated joint GWT configuration.

Optionally also fit binary + σ_e as a secondary comparison (if COMPUTE_SECONDARY
is True). Saves idata and a 7-point diagnostic summary analogous to
``run_and_summarise_binary_vs_three_state.py``, with the expert-scales
acceptance criteria from the plan.

Runtime: primary fit ~90 min CPU; secondary fit another ~90 min.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import arviz as az
import numpy as np
from scipy.stats import norm

from dcm_ppc import (
    per_expert_system_ppc_multisystem,
    per_expert_system_tree_implied_ppc_multisystem,
)
from gwt_three_state_indicator_analysis import (
    FOCUS_KEYS,
    SYSTEM_CONFIGS_VALIDATED,
    SYSTEM_DISPLAY,
    compare_c_summaries,
    fit_gwt,
    format_c_comparison_table,
)

OUT_DIR = Path("results/gwt_expert_scales")
ANALYSIS_DIR = OUT_DIR / "analysis"
BASELINE_DIR = Path("results/gwt_binary_three_state")

COMPUTE_SECONDARY = False  # set True to also fit binary + sigma_e


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------


def diagnostics_overview(idata: Any) -> Dict[str, Any]:
    post = idata.posterior
    names: List[str] = ["a", "kappa"]
    if "tau_sigma" in post.data_vars:
        names.append("tau_sigma")
    if "sigma_by_expert" in post.data_vars:
        names.append("sigma_by_expert")
    # Include free system-C variables (skip anchored / deterministic ones)
    for v in post.data_vars:
        if not v.endswith("_C"):
            continue
        draws = np.asarray(post[v].values).reshape(-1)
        if np.std(draws) > 1e-10:
            names.append(v)
    names.extend(sorted(
        v for v in post.data_vars
        if (v.endswith("_beta_pres") or v.endswith("_beta_abs"))
    ))
    diag = az.summary(idata, var_names=names, kind="diagnostics")
    return {
        "divergences": int(idata.sample_stats["diverging"].values.sum()),
        "max_rhat": float(diag["r_hat"].max()),
        "min_ess_bulk": float(diag["ess_bulk"].min()),
        "min_ess_tail": float(diag["ess_tail"].min()),
    }


def sigma_by_expert_summary(idata: Any, expert_names: Sequence[str]) -> List[Dict[str, Any]]:
    post = idata.posterior
    if "sigma_by_expert" not in post.data_vars:
        return []
    sigmas = np.asarray(post["sigma_by_expert"].values).reshape(-1, len(expert_names))
    tau = np.asarray(post["tau_sigma"].values).reshape(-1) if "tau_sigma" in post.data_vars else None
    rows = []
    for i, name in enumerate(expert_names):
        s = sigmas[:, i]
        rows.append({
            "expert": name,
            "median": float(np.median(s)),
            "lo": float(np.percentile(s, 3)),
            "hi": float(np.percentile(s, 97)),
        })
    tau_row = None
    if tau is not None:
        tau_row = {
            "median": float(np.median(tau)),
            "lo": float(np.percentile(tau, 3)),
            "hi": float(np.percentile(tau, 97)),
        }
    return rows, tau_row


def a_kappa_summary(idata: Any) -> Dict[str, Any]:
    post = idata.posterior
    a = np.asarray(post["a"].values).reshape(-1)
    kappa = np.asarray(post["kappa"].values).reshape(-1, post["kappa"].shape[-1])
    return {
        "a_median": float(np.median(a)),
        "a_lo": float(np.percentile(a, 3)),
        "a_hi": float(np.percentile(a, 97)),
        "kappa_medians": np.median(kappa, axis=0).tolist(),
        "kappa_lo": np.percentile(kappa, 3, axis=0).tolist(),
        "kappa_hi": np.percentile(kappa, 97, axis=0).tolist(),
    }


def focus_ppc_rows(
    idata, builder, processor, n_draws: int = 500,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    leaf = per_expert_system_ppc_multisystem(
        idata, builder, processor, n_draws=n_draws, seed=0
    )
    tree = per_expert_system_tree_implied_ppc_multisystem(
        idata, builder, processor, n_draws=n_draws, seed=0
    )
    rows = {}
    for key in FOCUS_KEYS:
        rL = leaf.get(key)
        rT = tree.get(key)
        if rL is None:
            continue
        rows[key] = {
            "leaf": rL,
            "tree": rT,
        }
    return rows


def all_experts_ppc(
    idata, builder, processor, n_draws: int = 500,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Full stratified (expert, system) PPC for all cells."""
    return per_expert_system_ppc_multisystem(
        idata, builder, processor, n_draws=n_draws, seed=0
    )


def per_expert_emission_ceiling(idata: Any, n_experts: int, K: int) -> np.ndarray:
    """Compute per-expert oracle emission ceiling P(r=1|η=0) and P(r=K|η=a)."""
    post = idata.posterior
    a_draws = np.asarray(post["a"].values).reshape(-1)
    kappa_draws = np.asarray(post["kappa"].values).reshape(-1, K - 1)
    if "sigma_by_expert" in post.data_vars:
        sigmas = np.asarray(post["sigma_by_expert"].values).reshape(-1, n_experts)
    else:
        sigmas = np.ones((a_draws.shape[0], n_experts))

    # Per-expert ceilings
    out = np.zeros((n_experts, 2))  # [expert, (cat1, catK)]
    for e in range(n_experts):
        sigma_e = sigmas[:, e]
        # P(r=1 | eta=0, sigma_e) = Phi(kappa_1 / sigma_e)
        p_r1 = norm.cdf(kappa_draws[:, 0] / sigma_e)
        # P(r=K | eta=a, sigma_e) = 1 - Phi((kappa_{K-1} - a) / sigma_e)
        p_rK = 1.0 - norm.cdf((kappa_draws[:, -1] - a_draws) / sigma_e)
        out[e, 0] = np.median(p_r1)
        out[e, 1] = np.median(p_rK)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("[1/5] Fitting three-state + sigma_e (primary)...")
    three_state_sigma = fit_gwt(
        state_model="three_state",
        fit_overrides={"USE_EXPERT_SCALES": True},
    )
    az.to_netcdf(three_state_sigma["idata"], OUT_DIR / "three_state_sigma_anchored.nc")
    print(f"      saved to {OUT_DIR / 'three_state_sigma_anchored.nc'}")
    t_primary = time.time() - t0

    if COMPUTE_SECONDARY:
        print("\n[2/5] Fitting binary + sigma_e (secondary)...")
        binary_sigma = fit_gwt(
            state_model="binary",
            fit_overrides={"USE_EXPERT_SCALES": True},
        )
        az.to_netcdf(binary_sigma["idata"], OUT_DIR / "binary_sigma_anchored.nc")
        print(f"      saved to {OUT_DIR / 'binary_sigma_anchored.nc'}")
    else:
        binary_sigma = None
        print("\n[2/5] Skipping secondary binary + sigma_e fit (COMPUTE_SECONDARY=False)")

    print("\n[3/5] Computing diagnostics...")
    ts = three_state_sigma
    ts_diag = diagnostics_overview(ts["idata"])
    ts_ak = a_kappa_summary(ts["idata"])
    ts_sigma_rows, ts_tau = sigma_by_expert_summary(ts["idata"], ts["processor"].expert_names)
    ts_focus = focus_ppc_rows(ts["idata"], ts["builder"], ts["processor"])
    ts_strat = all_experts_ppc(ts["idata"], ts["builder"], ts["processor"])
    ts_ceiling = per_expert_emission_ceiling(
        ts["idata"], len(ts["processor"].expert_names),
        ts["builder"].config.N_CATEGORIES,
    )

    # Load three-state baseline (no sigma_e) for comparison
    tsb_path = BASELINE_DIR / "three_state_anchored.nc"
    ts_baseline_idata = az.from_netcdf(tsb_path) if tsb_path.exists() else None
    bin_baseline_path = BASELINE_DIR / "binary_anchored.nc"
    bin_baseline_idata = az.from_netcdf(bin_baseline_path) if bin_baseline_path.exists() else None

    print("\n[4/5] Computing comparison vs three-state baseline...")
    c_cmp = None
    if ts_baseline_idata is not None:
        # Crude C comparison using the existing helpers; needs a "result-dict"-shaped obj
        baseline_result = {
            "c_summary": _build_c_summary_from_idata(
                ts_baseline_idata, ts["builder"], ts["system_configs"],
            ),
        }
        c_cmp = compare_c_summaries(baseline_result, ts)

    print("\n[5/5] Writing summary...")
    _write_summary(
        ts, ts_diag, ts_ak, ts_sigma_rows, ts_tau, ts_focus, ts_strat, ts_ceiling,
        c_cmp, binary_sigma, ts_baseline_idata, bin_baseline_idata,
        elapsed=time.time() - t0,
    )
    print(f"\nDone in {time.time() - t0:.0f}s. Summary at {ANALYSIS_DIR / 'summary.md'}")


def _build_c_summary_from_idata(idata, builder, system_configs):
    """Build a c_summary dict matching the structure used in compare_c_summaries."""
    stance_var = builder.node_to_varname["Global Workspace Theory"]
    summary: Dict[str, Dict[str, float]] = {}
    for sys_name, c_fixed in system_configs:
        key = f"{builder._sys_prefix(sys_name)}__{stance_var}_C"
        if key in idata.posterior.data_vars:
            draws = np.asarray(idata.posterior[key].values).reshape(-1)
            if np.std(draws) < 1e-12:
                summary[sys_name] = {
                    "median": float(c_fixed) if c_fixed is not None else float(draws[0]),
                    "lo": float(c_fixed) if c_fixed is not None else float(draws[0]),
                    "hi": float(c_fixed) if c_fixed is not None else float(draws[0]),
                    "fixed": True,
                }
            else:
                summary[sys_name] = {
                    "median": float(np.median(draws)),
                    "lo": float(np.percentile(draws, 3)),
                    "hi": float(np.percentile(draws, 97)),
                    "fixed": False,
                }
        else:
            summary[sys_name] = {
                "median": float(c_fixed) if c_fixed is not None else float("nan"),
                "lo": float(c_fixed) if c_fixed is not None else float("nan"),
                "hi": float(c_fixed) if c_fixed is not None else float("nan"),
                "fixed": True,
            }
    return summary


def _write_summary(
    ts, ts_diag, ts_ak, ts_sigma_rows, ts_tau, ts_focus, ts_strat, ts_ceiling,
    c_cmp, binary_sigma, ts_baseline_idata, bin_baseline_idata, elapsed,
) -> None:
    lines = []
    lines.append("# Three-state + σ_e (hierarchical partial pooling)\n")
    lines.append(
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}, wall-time {elapsed:.0f}s.\n"
    )

    # 1. C posteriors
    lines.append("## 1. System-level C posteriors\n")
    lines.append("| System | Three-state + σ_e | Three-state baseline | ΔMedian |")
    lines.append("|---|---|---|---:|")
    ts_c = ts["c_summary"]
    baseline_c = _build_c_summary_from_idata(
        ts_baseline_idata, ts["builder"], ts["system_configs"]
    ) if ts_baseline_idata is not None else None
    for sys_name, _ in ts["system_configs"]:
        cur = ts_c[sys_name]
        b = baseline_c[sys_name] if baseline_c else {"median": float("nan"), "lo": float("nan"), "hi": float("nan")}
        label = SYSTEM_DISPLAY.get(sys_name, sys_name)
        cur_str = f"{cur['median']:.3f} [{cur['lo']:.3f}, {cur['hi']:.3f}]"
        base_str = f"{b['median']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}]" if baseline_c else "—"
        delta = cur["median"] - b["median"] if baseline_c and not np.isnan(b["median"]) else float("nan")
        lines.append(f"| {label} | {cur_str} | {base_str} | {delta:+.3f} |")
    lines.append("")

    # 2. Convergence
    lines.append("## 2. Convergence diagnostics\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    for k, v in ts_diag.items():
        lines.append(f"| {k} | {v:.3f}" if isinstance(v, float) else f"| {k} | {v} |")
    lines.append("")

    # 3. a, kappa, sigma
    lines.append("## 3. Observation-layer parameters\n")
    lines.append(
        f"- **a** (discrimination): median {ts_ak['a_median']:.3f}, "
        f"94% [{ts_ak['a_lo']:.3f}, {ts_ak['a_hi']:.3f}]"
    )
    lines.append("- **kappa** (ordered cutpoints):")
    for i, (m, lo, hi) in enumerate(zip(ts_ak["kappa_medians"], ts_ak["kappa_lo"], ts_ak["kappa_hi"])):
        lines.append(f"    - κ_{i+1}: {m:.3f} [{lo:.3f}, {hi:.3f}]")
    if ts_tau is not None:
        lines.append(
            f"- **τ_σ** (pop. log-scale spread): median {ts_tau['median']:.3f}, "
            f"94% [{ts_tau['lo']:.3f}, {ts_tau['hi']:.3f}]  "
            f"(HalfNormal(0.3) prior → 94% upper ~0.6)"
        )
    lines.append("")

    # sigma_by_expert
    lines.append("## 3b. Per-expert σ_e posterior\n")
    lines.append("| Expert | median | 94% interval | Shrunk to 1 (|log σ| < 0.15)? |")
    lines.append("|---|---:|---|:---:|")
    for r in ts_sigma_rows:
        shrunk = "✓" if abs(np.log(r["median"])) < 0.15 else "moved"
        lines.append(
            f"| {r['expert']} | {r['median']:.3f} | "
            f"[{r['lo']:.3f}, {r['hi']:.3f}] | {shrunk} |"
        )
    lines.append("")

    # 4. Focus-cell PPC
    lines.append("## 4. Focus-cell signed-tail + variance PPCs\n")
    lines.append("| Cell | obs_left | pred_left | Δleft | obs_right | pred_right | Δright | obs_var | pred_var | Δvar |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in FOCUS_KEYS:
        if key not in ts_focus:
            continue
        r = ts_focus[key]["leaf"]
        name = f"{key[0]} × {SYSTEM_DISPLAY.get(key[1], key[1])}"
        lines.append(
            f"| {name} | "
            f"{r['obs_left']:.3f} | {r['pred_left_mean']:.3f} | {r['delta_left_mean']:+.3f} | "
            f"{r['obs_right']:.3f} | {r['pred_right_mean']:.3f} | {r['delta_right_mean']:+.3f} | "
            f"{r['obs_var']:.3f} | {r['pred_var_mean']:.3f} | {r['delta_var_mean']:+.3f} |"
        )
    lines.append("")

    # 5. Tree-implied vs leaf-updated for focus cells
    lines.append("## 5. Tree-implied vs leaf-updated PPC (focus cells)\n")
    lines.append("| Cell | Source | pred_mean | Δleft | Δright | Δmid |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for key in FOCUS_KEYS:
        if key not in ts_focus:
            continue
        name = f"{key[0]} × {SYSTEM_DISPLAY.get(key[1], key[1])}"
        for src_name, src_key in [("leaf_updated", "leaf"), ("tree_implied", "tree")]:
            r = ts_focus[key].get(src_key)
            if r is None:
                continue
            lines.append(
                f"| {name} | {src_name} | {r['pred_mean_mean']:.2f} | "
                f"{r['delta_left_mean']:+.3f} | {r['delta_right_mean']:+.3f} | "
                f"{r['delta_mid_mean']:+.3f} |"
            )
    lines.append("")

    # 6. Per-expert emission ceilings
    lines.append("## 6. Per-expert emission ceiling (oracle)\n")
    lines.append("| Expert | P(r=1 \\| η=0) | P(r=7 \\| η=a) |")
    lines.append("|---|---:|---:|")
    for i, name in enumerate(ts["processor"].expert_names):
        lines.append(
            f"| {name} | {ts_ceiling[i, 0]:.3f} | {ts_ceiling[i, 1]:.3f} |"
        )
    lines.append("")

    # 7. Stratified PPC — full expert×system grid summary
    lines.append("## 7. Stratified PPC — all (expert, system) cells\n")
    lines.append("| Expert | System | n | Δleft | Δright | Δmid | Δvar |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for (exp, sys_name), r in sorted(ts_strat.items()):
        lines.append(
            f"| {exp} | {SYSTEM_DISPLAY.get(sys_name, sys_name)} | {r['n_obs']} | "
            f"{r['delta_left_mean']:+.3f} | {r['delta_right_mean']:+.3f} | "
            f"{r['delta_mid_mean']:+.3f} | {r['delta_var_mean']:+.3f} |"
        )
    lines.append("")

    # Acceptance criteria evaluation
    lines.append("## Acceptance criteria evaluation\n")
    # 1. Ordering preserved
    order_ok = (
        ts_c["Human"]["median"] > ts_c["Chicken"]["median"]
        > ts_c["2024 Leading Chat LLMs"]["median"] > ts_c["ELIZA"]["median"]
    )
    lines.append(f"- Ordering Human ≫ Chicken > LLM ≫ ELIZA: {'✓' if order_ok else '✗'}")
    # 2. Stability guardrail
    if baseline_c:
        dc = abs(ts_c["Chicken"]["median"] - baseline_c["Chicken"]["median"])
        dl = abs(ts_c["2024 Leading Chat LLMs"]["median"] - baseline_c["2024 Leading Chat LLMs"]["median"])
        lines.append(
            f"- |ΔChicken|={dc:.3f}, |ΔLLM|={dl:.3f} (guardrail 0.05): "
            f"{'✓' if dc <= 0.05 and dl <= 0.05 else 'flag — articulate reason'}"
        )
    # 3-4. σ_e patterns
    lines.append("- σ_e shrinkage (informational): see 3b table above")
    # 10. Diagnostics
    div_ok = ts_diag["divergences"] == 0
    rhat_ok = ts_diag["max_rhat"] < 1.05
    lines.append(
        f"- Clean diagnostics: divergences={ts_diag['divergences']}, "
        f"max R-hat={ts_diag['max_rhat']:.3f}, "
        f"min ESS bulk={ts_diag['min_ess_bulk']:.0f}: "
        f"{'✓' if div_ok and rhat_ok else 'flag'}"
    )
    lines.append("")

    (ANALYSIS_DIR / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
