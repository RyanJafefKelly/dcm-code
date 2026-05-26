"""Real expert rating distributions per system, pooled across all stances and indicators.

Reads `data_cache.json` and walks the indicator tree for each stance, collecting
every (rater, indicator) observation per system. Raw values are 0-1 probabilities
(with -1 used as the skip code), binned into 7 equal-width ordinal categories
matching the model's K = 7 rating scale.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data_cache.json"

SYS_DISPLAY = {
    "Human": "Human",
    "Chicken": "Chicken",
    "2024 Leading Chat LLMs": "LLMs (2024)",
    "ELIZA": "ELIZA",
}
SYS_ORDER = ["Human", "Chicken", "2024 Leading Chat LLMs", "ELIZA"]


def collect_per_system_observations(data: list[dict]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {s: [] for s in SYS_ORDER}

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "Indicator":
                obs = node.get("observations", {})
                for sys in SYS_ORDER:
                    if sys in obs:
                        for v in obs[sys].get("values", []):
                            try:
                                fv = float(v)
                                if fv >= 0:  # -1 means skipped
                                    out[sys].append(fv)
                            except (ValueError, TypeError):
                                pass
            for value in node.values():
                if isinstance(value, list):
                    for child in value:
                        walk(child)
                elif isinstance(value, dict):
                    walk(value)

    for stance in data:
        walk(stance)
    return out


def main() -> None:
    with open(DATA_PATH) as fh:
        data = json.load(fh)
    per_system = collect_per_system_observations(data)

    edges = np.linspace(0, 1, 8)
    cat_centres = np.arange(1, 8)

    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    for ax, sys in zip(axes, SYS_ORDER):
        arr = np.asarray(per_system[sys])
        h, _ = np.histogram(arr, bins=edges)
        h = h / h.sum()
        ax.bar(cat_centres, h, color="#4C78A8", width=0.7)
        ax.set_title(f"{SYS_DISPLAY[sys]}  (n = {len(arr)} expert ratings)")
        ax.set_ylabel("proportion")
        ax.set_ylim(0, max(h) * 1.15)
        for i, p in enumerate(h):
            if p > 0.02:
                ax.text(cat_centres[i], p + max(h) * 0.02, f"{p:.2f}", ha="center", fontsize=8)
    axes[-1].set_xlabel("rating category (1 = lowest, 7 = highest)")
    axes[-1].set_xticks(cat_centres)
    fig.suptitle(
        "Real expert rating distributions per system\n"
        "(pooled across all stances and indicators; 0–1 raw → 7-bin ordinal)",
        fontsize=11,
    )
    fig.tight_layout()

    out = Path(__file__).resolve().parent / "real_observed_ratings_per_system.png"
    fig.savefig(out, dpi=150)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
