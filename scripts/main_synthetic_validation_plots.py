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
    plot_rho_calibration(output_dir, plots_dir)
    plot_rho_by_truth(output_dir, plots_dir)
    plot_confusion(output_dir, plots_dir)
    plot_pareto(output_dir, plots_dir)
    plot_sampler(output_dir, plots_dir)
    print(f"[main-validation-plots] wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()
