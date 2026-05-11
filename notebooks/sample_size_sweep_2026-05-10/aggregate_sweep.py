"""Aggregate the K-sweep fits and emit summary CSV + figures.

For each completed run under runs/<run_id>/, load fit.nc, pull the free
root-C draws for Chicken and LLM, and emit:

  eval/sample_size_sweep_summary.csv         one row per (K, system)
  eval/sample_size_sweep_summary.md          markdown copy of the table
  figs/contraction_vs_sqrtK.png              SD vs sqrt(K) per system + 1/sqrtK ref
  figs/median_vs_K.png                       median + 94% HDI per system vs K

The C prior is Beta(1, 5); SD = sqrt(5 / 252) ~ 0.1409.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS_DIR = HERE / "runs"
EVAL_DIR = HERE / "eval"
FIGS_DIR = HERE / "figs"
EVAL_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)

PRIOR_ALPHA, PRIOR_BETA = 1.0, 5.0
PRIOR_SD = math.sqrt(
    (PRIOR_ALPHA * PRIOR_BETA)
    / ((PRIOR_ALPHA + PRIOR_BETA) ** 2 * (PRIOR_ALPHA + PRIOR_BETA + 1.0))
)

TRUE_C = {"Chicken": 0.25, "LLM": 0.10}
SYSTEM_VARNAMES = {
    "Chicken": "chicken__global_workspace_theory_C",
    "LLM": "2024_leading_chat_llms__global_workspace_theory_C",
}


def find_run_dir(K: int) -> Path:
    suffix = "" if K == 1 else f"__multK{K}"
    name = (
        "exact_latent_tree__full_exact_tree__three_state_binomial_2__"
        "exact_tree_production_medians__current_gwt_rater_design__"
        f"seed20260506{suffix}"
    )
    return RUNS_DIR / name


def summarise_run(K: int) -> List[Dict[str, float]]:
    run_dir = find_run_dir(K)
    nc_path = run_dir / "fit.nc"
    if not nc_path.exists():
        print(f"[skip] K={K}: no fit.nc at {nc_path}")
        return []
    idata = az.from_netcdf(str(nc_path))
    rows = []
    for system, var in SYSTEM_VARNAMES.items():
        draws = np.asarray(idata.posterior[var].values).reshape(-1)
        median = float(np.median(draws))
        sd = float(np.std(draws, ddof=1))
        truth = TRUE_C[system]
        p03 = float(np.percentile(draws, 3))
        p97 = float(np.percentile(draws, 97))
        rows.append(
            {
                "K": int(K),
                "sqrt_K": float(math.sqrt(K)),
                "system": system,
                "truth": truth,
                "posterior_median": median,
                "posterior_sd": sd,
                "contraction_ratio": sd / PRIOR_SD,
                "signed_error": median - truth,
                "abs_error": abs(median - truth),
                "p03": p03,
                "p97": p97,
                "interval_width": p97 - p03,
                "interval_includes_truth": bool(p03 <= truth <= p97),
            }
        )
    return rows


def write_summary(df: pd.DataFrame) -> None:
    df = df.sort_values(["system", "K"]).reset_index(drop=True)
    df.to_csv(EVAL_DIR / "sample_size_sweep_summary.csv", index=False)

    cols = [
        "K",
        "system",
        "posterior_median",
        "posterior_sd",
        "contraction_ratio",
        "signed_error",
        "abs_error",
        "interval_width",
        "interval_includes_truth",
    ]
    md_lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df[cols].iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        md_lines.append("| " + " | ".join(vals) + " |")
    md_lines.append("")
    md_lines.append(
        f"_Prior C ~ Beta({int(PRIOR_ALPHA)}, {int(PRIOR_BETA)});  "
        f"prior SD = {PRIOR_SD:.4f}.  "
        f"Truth: Chicken=0.25, LLM=0.10._"
    )
    (EVAL_DIR / "sample_size_sweep_summary.md").write_text("\n".join(md_lines))


def plot_contraction_vs_sqrtK(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    Ks = sorted(df["K"].unique())
    sqrtKs = [math.sqrt(K) for K in Ks]

    for system in ["Chicken", "LLM"]:
        sub = df[df["system"] == system].sort_values("K")
        ax.plot(
            np.sqrt(sub["K"]),
            sub["posterior_sd"],
            marker="o",
            label=f"{system} (truth={TRUE_C[system]:.2f})",
        )

    K1_sub = df[df["K"] == 1]
    if not K1_sub.empty:
        anchor_sd = float(K1_sub["posterior_sd"].mean())
        x = np.array(sqrtKs)
        ax.plot(
            x,
            anchor_sd / x,
            linestyle="--",
            color="grey",
            alpha=0.7,
            label=r"$\propto 1/\sqrt{K}$ (anchored at K=1)",
        )

    ax.axhline(PRIOR_SD, color="black", linestyle=":", alpha=0.5,
               label=f"prior SD ≈ {PRIOR_SD:.3f}")
    ax.set_xlabel(r"$\sqrt{K}$")
    ax.set_ylabel("Posterior SD on $C_s$")
    ax.set_title("Posterior contraction on $C_s$ vs $\\sqrt{K}$")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "contraction_vs_sqrtK.png", dpi=150)
    plt.close(fig)


def plot_median_vs_K(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    for ax, system in zip(axes, ["Chicken", "LLM"]):
        sub = df[df["system"] == system].sort_values("K")
        K = sub["K"].to_numpy()
        med = sub["posterior_median"].to_numpy()
        lo = sub["p03"].to_numpy()
        hi = sub["p97"].to_numpy()
        ax.errorbar(
            K,
            med,
            yerr=[med - lo, hi - med],
            fmt="o-",
            capsize=4,
            label="posterior median ± 94% HDI",
        )
        ax.axhline(TRUE_C[system], color="red", linestyle="--", alpha=0.6,
                   label=f"truth = {TRUE_C[system]:.2f}")
        ax.set_xscale("log")
        ax.set_xlabel("K (rater multiplier)")
        ax.set_ylabel("$C_s$")
        ax.set_title(system)
        ax.set_xticks(sorted(df["K"].unique()))
        ax.set_xticklabels([str(int(k)) for k in sorted(df["K"].unique())])
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Posterior median + 94% HDI on $C_s$ vs K")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "median_vs_K.png", dpi=150)
    plt.close(fig)


def main() -> None:
    rows: List[Dict[str, float]] = []
    for K in (1, 2, 5, 10, 20):
        rows.extend(summarise_run(K))
    if not rows:
        raise SystemExit("No completed runs found.")
    df = pd.DataFrame(rows)
    write_summary(df)
    plot_contraction_vs_sqrtK(df)
    plot_median_vs_K(df)
    print(f"Wrote {EVAL_DIR / 'sample_size_sweep_summary.csv'}")
    print(f"Wrote {FIGS_DIR / 'contraction_vs_sqrtK.png'}")
    print(f"Wrote {FIGS_DIR / 'median_vs_K.png'}")
    print()
    print(df.sort_values(["system", "K"]).to_string(index=False))


if __name__ == "__main__":
    main()
