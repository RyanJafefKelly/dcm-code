"""All-stances pool_3s + soft-anchor batch fit.

Fits every stance with the `pool_3s` config (three-state indicator leaf +
label-tied tree-beta pooling) under soft reference anchors Beta(50, 1) /
Beta(1, 50) on Human / ELIZA. Moderately-reliable sampling: 4 chains × 1200
draws × 600 tune × target_accept=0.95. Skips any stance whose idata already
exists on disk. After each fit, updates a cross-stance root-C summary figure
so progress is visible mid-run.

Outputs (local, `results/` is gitignored):
- `results/all_stances_pool_3s/{stance_slug}.nc` + `.meta.json`
- `results/all_stances_pool_3s/summary.json` (updated per-fit)
- `results/all_stances_pool_3s/root_C_across_stances.png` (regenerated per fit)

Runtime budget: ~15-20 min per fit × 13 stances = ~3.5-4 hours.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcm_model import (
    EvidenceProcessor, ModelConfig, MultiSystemDataProcessor,
    MultiSystemModelBuilder, fit_stance_multisystem, load_data,
)


SOFT_ANCHORS = {"Human": (50.0, 1.0), "ELIZA": (1.0, 50.0)}
SYSTEM_CONFIGS = [
    ("Human", 0.999),
    ("Chicken", None),
    ("2024 Leading Chat LLMs", None),
    ("ELIZA", 0.001),
]
OUT_DIR = Path("results/all_stances_pool_3s")

STANCE_ORDER = [
    "Global Workspace Theory",
    "Integrated Information Theory",
    "Higher Order Theory",
    "Recurrent Processing Theory (Perceptual)",
    "Recurrent Processing Theory (Pure)",
    "Attention Schema Theory",
    "Person-like",
    "Cognitive Complexity",
    "Embodied Agency",
    "Field Mechanisms",
    "Simple Valence",
    "Computational Analogy",
    "Biological Analogy",
]

SYS_KEYS = ["Human", "Chicken", "LLMs", "ELIZA"]


def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").lower()


def slug_file(name: str) -> str:
    # File-system safe; strip parens
    return sanitize(name).replace("(", "").replace(")", "")


def build_config() -> ModelConfig:
    return ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=True,
        LABEL_POOL_SIGMA=0.5,
        SOFT_REFERENCE_ANCHORS=SOFT_ANCHORS,
        NUM_SAMPLES=1200,
        NUM_TUNE=600,
        NUM_CHAINS=4,
        TARGET_ACCEPT=0.95,
    )


def sys_var(stance_name: str, sys_key: str) -> str:
    prefix = {
        "Human":   "human",
        "Chicken": "chicken",
        "LLMs":    "2024_leading_chat_llms",
        "ELIZA":   "eliza",
    }[sys_key]
    return f"{prefix}__{sanitize(stance_name)}_C"


def fit_one(stance_name: str) -> dict | None:
    out_nc = OUT_DIR / f"{slug_file(stance_name)}.nc"
    out_meta = OUT_DIR / f"{slug_file(stance_name)}.meta.json"

    if out_nc.exists():
        print(f"[skip] {stance_name} -> {out_nc} already exists")
        return None

    cfg = build_config()
    stance_data = next((s for s in load_data(cfg) if s["name"] == stance_name), None)
    if stance_data is None:
        print(f"[error] stance not found: {stance_name}")
        return None

    print(f"[fit]  {stance_name} (starting)")
    t0 = time.time()
    try:
        idata, _bld, _proc = fit_stance_multisystem(stance_data, cfg, SYSTEM_CONFIGS)
    except Exception as e:
        print(f"[error] {stance_name} failed: {e}")
        traceback.print_exc()
        return None

    elapsed = time.time() - t0
    az.to_netcdf(idata, str(out_nc))

    div = int(idata.sample_stats["diverging"].values.sum())
    vars_c = [v for v in idata.posterior.data_vars if v.endswith("_C")]
    summ = az.summary(idata, var_names=vars_c + ["a", "kappa"], kind="diagnostics")

    meta = {
        "stance": stance_name,
        "elapsed_s": elapsed,
        "divergences": div,
        "max_rhat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "min_ess_tail": float(summ["ess_tail"].min()),
        "idata_path": str(out_nc),
        "config": asdict(cfg),
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    print(f"[done] {stance_name}  ({elapsed:.0f}s, {div} divs, R-hat {summ['r_hat'].max():.3f})")
    return meta


def load_system_C_posterior(nc_path: Path, stance_name: str) -> dict:
    """Return dict sys_key -> (median, p03, p97)."""
    idata = az.from_netcdf(str(nc_path))
    out = {}
    for sys_key in SYS_KEYS:
        v = sys_var(stance_name, sys_key)
        if v in idata.posterior.data_vars:
            d = np.asarray(idata.posterior[v].values).reshape(-1)
            out[sys_key] = (
                float(np.median(d)),
                float(np.percentile(d, 3)),
                float(np.percentile(d, 97)),
            )
    return out


def regenerate_summary():
    """Rebuild summary.json + root_C plot from whatever idatas are on disk."""
    summary = {}
    for stance in STANCE_ORDER:
        nc_path = OUT_DIR / f"{slug_file(stance)}.nc"
        meta_path = OUT_DIR / f"{slug_file(stance)}.meta.json"
        if not nc_path.exists():
            continue
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        C_summary = load_system_C_posterior(nc_path, stance)
        summary[stance] = {
            "divergences": meta.get("divergences"),
            "max_rhat": meta.get("max_rhat"),
            "min_ess_bulk": meta.get("min_ess_bulk"),
            "elapsed_s": meta.get("elapsed_s"),
            "C_posterior": C_summary,
        }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Plot
    done_stances = [s for s in STANCE_ORDER if s in summary]
    if not done_stances:
        return
    n_done = len(done_stances)
    cmap = plt.get_cmap("tab20")
    colours = [cmap(i % 20) for i in range(n_done)]

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * n_done + 4), 5.5))
    n_sys = len(SYS_KEYS)
    width = 0.8 / n_done

    for i, stance in enumerate(done_stances):
        C = summary[stance]["C_posterior"]
        meds, lows, highs = [], [], []
        for sys_key in SYS_KEYS:
            if sys_key in C:
                m, lo, hi = C[sys_key]
                meds.append(m); lows.append(m - lo); highs.append(hi - m)
            else:
                meds.append(np.nan); lows.append(0); highs.append(0)
        x = np.arange(n_sys) + (i - (n_done - 1) / 2) * width
        ax.errorbar(x, meds, yerr=[lows, highs], fmt="o", capsize=3,
                    color=colours[i], markersize=6, linewidth=0,
                    label=_short(stance))

    ax.set_xticks(np.arange(n_sys))
    ax.set_xticklabels(SYS_KEYS, fontsize=10)
    ax.set_ylabel(r"$P(C=1 \mid \mathrm{data})$  (posterior median + 94% HDI)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.999, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axhline(0.001, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_title(f"System root-credence posterior across {n_done}/13 stances\n"
                 f"(pool_3s + soft anchors; 1200 draws x 4 chains)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "root_C_across_stances.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _short(name: str) -> str:
    return {
        "Global Workspace Theory": "GWT",
        "Integrated Information Theory": "IIT",
        "Higher Order Theory": "HOT",
        "Recurrent Processing Theory (Perceptual)": "RPT-P",
        "Recurrent Processing Theory (Pure)": "RPT-Pure",
        "Attention Schema Theory": "AST",
        "Person-like": "Person-like",
        "Cognitive Complexity": "CogComplex",
        "Embodied Agency": "EmbAgent",
        "Field Mechanisms": "Field",
        "Simple Valence": "SValence",
        "Computational Analogy": "CompAnal",
        "Biological Analogy": "BioAnal",
    }.get(name, name)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Batch-fitting {len(STANCE_ORDER)} stances under pool_3s + soft anchors.")
    print(f"Writing to {OUT_DIR}. Skip-cached: yes.\n")

    total_t0 = time.time()
    for i, stance in enumerate(STANCE_ORDER, 1):
        print(f"=== [{i}/{len(STANCE_ORDER)}] {stance} ===")
        fit_one(stance)
        regenerate_summary()
        elapsed_total = time.time() - total_t0
        print(f"--- cumulative elapsed: {elapsed_total/60:.1f} min ---\n")

    print(f"\nAll done. Total: {(time.time() - total_t0)/60:.1f} min.")
    print(f"See {OUT_DIR}/summary.json and {OUT_DIR}/root_C_across_stances.png.")


if __name__ == "__main__":
    main()
