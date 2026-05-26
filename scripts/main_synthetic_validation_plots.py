"""Plot regeneration for main synthetic validation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs/main_synthetic_validation")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def save_empty(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ppc_overlay(output_dir: Path, plots_dir: Path) -> None:
    df = read_csv(output_dir / "posterior_predictive_rating_distribution_by_system.csv")
    path = plots_dir / "ppc_rating_distribution_overlay_by_system.png"
    if df.empty:
        save_empty(path, "PPC Rating Distribution Overlay")
        return
    systems = list(df["system"].dropna().unique())
    fig, axes = plt.subplots(len(systems), 1, figsize=(8, max(3, 2.3 * len(systems))), sharex=True)
    if len(systems) == 1:
        axes = [axes]
    for ax, system in zip(axes, systems):
        g = df[df["system"] == system].sort_values("category")
        x = np.arange(len(g))
        ax.bar(x - 0.18, g["observed_proportion"], width=0.36, label="observed", color="#4C78A8")
        ax.bar(x + 0.18, g["predicted_proportion"], width=0.36, label="posterior predictive", color="#F58518")
        ax.set_title(str(system))
        ax.set_ylabel("proportion")
        ax.set_ylim(0, max(0.05, float(max(g["observed_proportion"].max(), g["predicted_proportion"].max())) * 1.25))
    axes[-1].set_xticks(np.arange(7))
    axes[-1].set_xticklabels([str(i + 1) for i in range(7)])
    axes[-1].set_xlabel("rating category")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_ppc_overlay_faceted(output_dir: Path, plots_dir: Path) -> None:
    df = read_csv(output_dir / "posterior_predictive_rating_distribution_by_system.csv")
    path = plots_dir / "ppc_observed_vs_predicted_rating_bars.png"
    if df.empty:
        save_empty(path, "Observed vs Posterior Predictive Ratings")
        return
    systems = [s for s in ["Human", "ELIZA", "Chicken", "LLMs"] if s in set(df["system"])]
    if not systems:
        systems = list(df["system"].dropna().unique())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, system in zip(axes, systems):
        g = df[df["system"] == system].sort_values("category")
        x = np.arange(len(g))
        ax.bar(x - 0.18, g["observed_proportion"], width=0.36, label="observed", color="#4C78A8")
        ax.bar(x + 0.18, g["predicted_proportion"], width=0.36, label="posterior predictive", color="#F58518")
        ax.set_title(str(system))
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels([str(i + 1) for i in range(7)])
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[len(systems):]:
        ax.set_axis_off()
    axes[0].set_ylabel("proportion")
    axes[2].set_ylabel("proportion")
    axes[2].set_xlabel("rating category")
    axes[3].set_xlabel("rating category")
    axes[0].legend(loc="upper right")
    fig.suptitle("Observed Ratings vs Posterior Predictive Mean", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_ppc_rps_improvement(output_dir: Path, plots_dir: Path) -> None:
    df = read_csv(output_dir / "posterior_predictive_summary.csv")
    path = plots_dir / "ppc_rps_improvement_by_system.png"
    if df.empty:
        save_empty(path, "PPC RPS Improvement")
        return
    systems = [s for s in ["ALL", "Human", "ELIZA", "Chicken", "LLMs"] if s in set(df["system"])]
    g = df[df["system"].isin(systems)].set_index("system").loc[systems].reset_index()
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - 0.18, g["mean_RPS_baseline_empirical_marginal"], width=0.36, label="empirical marginal baseline", color="#9D755D")
    ax.bar(x + 0.18, g["mean_RPS"], width=0.36, label="posterior predictive", color="#54A24B")
    for xpos, base, model in zip(x, g["mean_RPS_baseline_empirical_marginal"], g["mean_RPS"]):
        ax.annotate(
            f"{base - model:.3f}",
            xy=(xpos, max(base, model) + 0.01),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel("mean RPS, lower is better")
    ax.set_title("Rating Predictive Score vs Empirical Baseline")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_ppc_weighted_tv_oracle_bands(output_dir: Path, plots_dir: Path) -> None:
    df = read_csv(output_dir / "oracle_ppc_system_baseline_summary.csv")
    path = plots_dir / "ppc_weighted_cell_tv_oracle_bands_by_system.png"
    if df.empty:
        save_empty(path, "Weighted Cell-TV vs Oracle Bands")
        return
    df = df[df["metric"] == "weighted_mean_cell_TV"].copy()
    if df.empty:
        save_empty(path, "Weighted Cell-TV vs Oracle Bands")
        return
    systems = [s for s in ["Human", "ELIZA", "Chicken", "LLMs"] if s in set(df["system"])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.ravel()
    for ax, system in zip(axes, systems):
        g = df[df["system"] == system].sort_values("seed")
        x = np.arange(len(g))
        ax.fill_between(x, g["oracle_q05"], g["oracle_q95"], color="#BAB0AC", alpha=0.35, label="oracle 5-95%")
        ax.plot(x, g["oracle_q50"], color="#666666", linewidth=1.5, label="oracle median")
        status_colors = {
            "compatible_with_finite_sample_noise": "#4C78A8",
            "ppc_excess_misfit": "#E45756",
            "ppc_underdispersed_or_metric_issue": "#B279A2",
        }
        colors = [status_colors.get(str(s), "#4C78A8") for s in g["status"]]
        ax.scatter(x, g["fitted_ppc_value"], color=colors, zorder=3, s=42, label="fitted PPC")
        ax.set_title(system)
        ax.set_xticks(x)
        ax.set_xticklabels(g["seed"].astype(str), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("weighted mean cell-TV")
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[len(systems):]:
        ax.set_axis_off()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:3], labels[:3], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("System Weighted Cell-TV Against Oracle Finite-Sample Bands", y=0.96)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_human_conditioning_audit(output_dir: Path, plots_dir: Path) -> None:
    df = read_csv(output_dir / "human_ppc_target_comparison.csv")
    cells = read_csv(output_dir / "human_oracle_posterior_predictive_cells.csv")
    path = plots_dir / "human_ppc_conditioning_audit.png"
    if df.empty or cells.empty:
        save_empty(path, "Human PPC Conditioning Audit")
        return
    summaries = df[df["row_type"].isin(["seed_summary", "overall_review_summary"])].copy()
    summaries = summaries[summaries["seed"].notna()]
    labels = summaries["seed"].astype(str).str.replace("review_seeds_20260513_20260518", "overall", regex=False)
    x = np.arange(len(summaries))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    ax.bar(x - 0.18, summaries["weighted_mean_tv_fit_vs_current_truth"], width=0.36, label="fit vs realised-latent truth", color="#E45756")
    ax.bar(x + 0.18, summaries["weighted_mean_tv_fit_vs_oracle_ppc"], width=0.36, label="fit vs oracle PPC", color="#54A24B")
    ax.axhline(0.15, color="#777777", linestyle="--", linewidth=1, label="pass threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("weighted mean TV")
    ax.set_title("Human Review Seeds: Target Matters")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1]
    colors = {
        "sparse_observation_noise": "#4C78A8",
        "ppc_diagnostic_artifact": "#F58518",
        "no_specific_issue": "#54A24B",
        "genuine_fit_oracle_mismatch": "#E45756",
        "moderate_fit_oracle_warning": "#B279A2",
    }
    for label, g in cells.groupby("issue_label_revised", dropna=False):
        ax.scatter(
            g["tv_fit_vs_current_truth"],
            g["tv_fit_vs_oracle_ppc"],
            s=34,
            alpha=0.78,
            label=str(label),
            color=colors.get(str(label), "#999999"),
        )
    ax.axhline(0.15, color="#777777", linestyle="--", linewidth=1)
    ax.axhline(0.25, color="#E45756", linestyle=":", linewidth=1)
    ax.set_xlabel("TV: fit vs realised-latent truth")
    ax.set_ylabel("TV: fit vs oracle PPC")
    ax.set_title("Human Cells Reclassified Under PPC Target")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_rho_calibration(output_dir: Path, plots_dir: Path) -> None:
    df = read_csv(output_dir / "validation_cases.csv")
    path = plots_dir / "rho_calibration_curve.png"
    if df.empty:
        save_empty(path, "Rho Calibration")
        return
    bins = np.linspace(0, 1, 6)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (df["rho_collapsed"] >= lo) & (df["rho_collapsed"] < hi if hi < 1 else df["rho_collapsed"] <= hi)
        g = df[mask]
        if not g.empty:
            rows.append((float(g["rho_collapsed"].mean()), float(g["root_z_true"].mean()), len(g)))
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color="#777777", linewidth=1)
    if rows:
        x, y, n = zip(*rows)
        ax.scatter(x, y, s=np.asarray(n) * 18 + 30, color="#4C78A8")
    ax.set_xlabel("mean predicted rho_collapsed")
    ax.set_ylabel("empirical R=1 rate")
    ax.set_title("Rho Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_rho_by_truth(output_dir: Path, plots_dir: Path) -> None:
    df = read_csv(output_dir / "validation_cases.csv")
    path = plots_dir / "rho_by_truth_and_system.png"
    if df.empty:
        save_empty(path, "Rho by Truth and System")
        return
    labels = []
    data = []
    for (system, truth), g in df.groupby(["system", "root_z_true"], dropna=False):
        labels.append(f"{system}\nR={truth}")
        data.append(g["rho_collapsed"].to_numpy())
    fig, ax = plt.subplots(figsize=(max(7, len(data) * 1.1), 4.5))
    ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.axhline(0.5, color="#777777", linewidth=1, linestyle="--")
    ax.axhline(0.05, color="#999999", linewidth=1, linestyle=":")
    ax.axhline(0.95, color="#999999", linewidth=1, linestyle=":")
    ax.set_ylabel("rho_collapsed")
    ax.set_title("Rho by Truth and System")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(output_dir: Path, plots_dir: Path) -> None:
    df = read_csv(output_dir / "validation_cases.csv")
    path = plots_dir / "confusion_matrix_counts.png"
    if df.empty:
        save_empty(path, "Confusion Matrix")
        return
    pred = (df["rho_collapsed"] > 0.5).astype(int)
    mat = np.zeros((2, 2), dtype=int)
    for true, p in zip(df["root_z_true"].astype(int), pred):
        mat[true, int(p)] += 1
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pred 0", "pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["true 0", "true 1"])
    ax.set_title("Confusion Matrix Counts")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pareto(output_dir: Path, plots_dir: Path) -> None:
    df = read_csv(output_dir / "loo_pareto_k.csv")
    path = plots_dir / "pareto_k_distribution.png"
    if df.empty or "pareto_k" not in df:
        save_empty(path, "Pareto-k Distribution")
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["pareto_k"].dropna(), bins=30, color="#54A24B", edgecolor="white")
    ax.axvline(0.7, color="#E45756", linestyle="--", label="0.7")
    ax.axvline(1.0, color="#B279A2", linestyle="--", label="1.0")
    ax.set_xlabel("Pareto k")
    ax.set_ylabel("ratings")
    ax.set_title("Pareto-k Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sampler(output_dir: Path, plots_dir: Path) -> None:
    df = read_csv(output_dir / "sampler_diagnostics_summary.csv")
    path = plots_dir / "sampler_diagnostics_overview.png"
    if df.empty:
        save_empty(path, "Sampler Diagnostics")
        return
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes = axes.ravel()
    metrics = [
        ("max_rhat", 1.01),
        ("min_ess_bulk", 400),
        ("min_ess_tail", 200),
        ("n_divergences", 0),
    ]
    for ax, (metric, threshold) in zip(axes, metrics):
        if metric not in df:
            ax.set_axis_off()
            continue
        ax.bar(np.arange(len(df)), df[metric], color="#4C78A8")
        ax.axhline(threshold, color="#E45756", linestyle="--")
        ax.set_title(metric)
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(df.get("seed", pd.Series(np.arange(len(df)))).astype(str), rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_ppc_overlay(output_dir, plots_dir)
    plot_ppc_overlay_faceted(output_dir, plots_dir)
    plot_ppc_rps_improvement(output_dir, plots_dir)
    plot_ppc_weighted_tv_oracle_bands(output_dir, plots_dir)
    plot_human_conditioning_audit(output_dir, plots_dir)
    plot_rho_calibration(output_dir, plots_dir)
    plot_rho_by_truth(output_dir, plots_dir)
    plot_confusion(output_dir, plots_dir)
    plot_pareto(output_dir, plots_dir)
    plot_sampler(output_dir, plots_dir)
    print(f"[main-validation-plots] wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
