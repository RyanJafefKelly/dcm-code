"""All-stances exact-tree BASELINE batch fit (SPAR final-report figure).

Fits every stance with the report's baseline model:
  * exact-tree (sum-product / belief-propagation) marginalisation
    -- MultiSystemExactTreeBuilder
  * three-state indicator leaf
  * standard per-edge support/demandingness Beta priors
    -- POOL_BETAS_BY_LABEL=False, no targeted override, no transmission gain
  * soft reference anchors Beta(50,1) / Beta(1,50) on Human / ELIZA
  * Chicken and "2024 Leading Chat LLMs" free under Beta(1, 5)

This deliberately EXCLUDES the pooled tree (pool_3s) and the targeted
asymmetric-prior override -- both are exploratory experiments, not the
baseline. It differs from the poster figure (run_all_stances_pool_3s.py)
in exactly two ways:
  (1) exact-tree builder, not the composite MultiSystemModelBuilder;
  (2) POOL_BETAS_BY_LABEL=False, not True.
USE_EXPERT_SHIFTS=False is kept identical to the poster run.

Outputs (results/ is gitignored):
  results/all_stances_exact_baseline/<stance>.nc + <stance>.meta.json
  results/all_stances_exact_baseline/summary.json
  results/all_stances_exact_baseline/root_C_across_stances.png

Already-fitted stances are skipped, so the run is resumable.

Usage:
  python run_all_stances_exact_baseline.py                                  # full 13-stance run (~12-18 h)
  python run_all_stances_exact_baseline.py --only "Global Workspace Theory"  # selected stance(s)
  python run_all_stances_exact_baseline.py --smoke                           # 2 stances, tiny draws (script check)
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
import pymc as pm

from dcm_model import (
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    load_data,
)
from dcm_model_exact_tree import MultiSystemExactTreeBuilder


SOFT_ANCHORS = {"Human": (50.0, 1.0), "ELIZA": (1.0, 50.0)}
SYSTEM_CONFIGS = [
    ("Human", 0.999),
    ("Chicken", None),
    ("2024 Leading Chat LLMs", None),
    ("ELIZA", 0.001),
]
SYS_KEYS = ["Human", "Chicken", "LLMs", "ELIZA"]

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

# Reassigned in main() when --smoke is passed.
OUT_DIR = Path("results/all_stances_exact_baseline")


def sanitize(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_").lower()


def slug_file(name: str) -> str:
    return sanitize(name).replace("(", "").replace(")", "")


def build_config(smoke: bool = False) -> ModelConfig:
    """The report BASELINE config.

    Every field not set here is the ModelConfig default, and the defaults
    already ARE the baseline (POOL_BETAS_BY_LABEL=False, TRANSMISSION_GAIN=1.0,
    GAIN_LOGIT_NORMAL=False, TARGETED_OVERRIDE_NODE_KEYS=None). The lines below
    are the ones that matter -- read them to confirm what is being fitted.
    """
    return ModelConfig(
        INDICATOR_STATE_MODEL="three_state",   # three-state leaf (matches poster)
        USE_EXPERT_SHIFTS=False,               # matches poster run
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=False,             # NO label pooling -> standard per-edge priors
        SOFT_REFERENCE_ANCHORS=SOFT_ANCHORS,   # soft Human/ELIZA anchors
        NUM_SAMPLES=50 if smoke else 1200,
        NUM_TUNE=50 if smoke else 600,
        NUM_CHAINS=2 if smoke else 4,
        TARGET_ACCEPT=0.95,
    )


def sys_var(stance_name: str, sys_key: str) -> str:
    prefix = {
        "Human": "human",
        "Chicken": "chicken",
        "LLMs": "2024_leading_chat_llms",
        "ELIZA": "eliza",
    }[sys_key]
    return f"{prefix}__{sanitize(stance_name)}_C"


def fit_one(stance_name: str, cfg: ModelConfig) -> dict | None:
    out_nc = OUT_DIR / f"{slug_file(stance_name)}.nc"
    out_meta = OUT_DIR / f"{slug_file(stance_name)}.meta.json"

    if out_nc.exists():
        print(f"[skip] {stance_name} -> {out_nc} already exists")
        return None

    stance_data = next(
        (s for s in load_data(cfg) if s["name"] == stance_name), None
    )
    if stance_data is None:
        print(f"[error] stance not found: {stance_name}")
        return None

    print(f"[fit]  {stance_name} (starting)")
    t0 = time.time()
    try:
        proc = MultiSystemDataProcessor(cfg)
        proc.process(stance_data, [s for s, _ in SYSTEM_CONFIGS])
        builder = MultiSystemExactTreeBuilder(
            cfg, EvidenceProcessor(cfg), proc, list(SYSTEM_CONFIGS)
        )
        model = builder.build_model(stance_data)
        with model:
            idata = pm.sample(
                draws=cfg.NUM_SAMPLES,
                tune=cfg.NUM_TUNE,
                chains=cfg.NUM_CHAINS,
                cores=cfg.NUM_CHAINS,
                target_accept=cfg.TARGET_ACCEPT,
                random_seed=42,
            )
    except Exception as e:  # noqa: BLE001 -- per-stance isolation is intentional
        print(f"[error] {stance_name} failed: {e}")
        traceback.print_exc()
        return None

    elapsed = time.time() - t0
    az.to_netcdf(idata, str(out_nc))

    div = int(idata.sample_stats["diverging"].values.sum())
    vars_c = [
        str(v)
        for v in idata.posterior.data_vars
        if str(v).endswith("_C")
        and float(np.std(np.asarray(idata.posterior[v].values))) > 1e-10
    ]
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
    print(
        f"[done] {stance_name}  ({elapsed:.0f}s, {div} divs, "
        f"max R-hat {summ['r_hat'].max():.3f}, min ESS {summ['ess_bulk'].min():.0f})"
    )
    return meta


def load_system_C_posterior(nc_path: Path, stance_name: str) -> dict:
    """Return dict sys_key -> (median, p3, p97)."""
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


def regenerate_summary(draws_label: str) -> None:
    """Rebuild summary.json + the cross-stance root-C plot from disk."""
    summary = {}
    for stance in STANCE_ORDER:
        nc_path = OUT_DIR / f"{slug_file(stance)}.nc"
        meta_path = OUT_DIR / f"{slug_file(stance)}.meta.json"
        if not nc_path.exists():
            continue
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        summary[stance] = {
            "divergences": meta.get("divergences"),
            "max_rhat": meta.get("max_rhat"),
            "min_ess_bulk": meta.get("min_ess_bulk"),
            "elapsed_s": meta.get("elapsed_s"),
            "C_posterior": load_system_C_posterior(nc_path, stance),
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    done = [s for s in STANCE_ORDER if s in summary]
    if not done:
        return
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(done) + 4), 5.5))
    n_sys = len(SYS_KEYS)
    width = 0.8 / len(done)
    for i, stance in enumerate(done):
        C = summary[stance]["C_posterior"]
        meds, lows, highs = [], [], []
        for sys_key in SYS_KEYS:
            if sys_key in C:
                m, lo, hi = C[sys_key]
                meds.append(m)
                lows.append(m - lo)
                highs.append(hi - m)
            else:
                meds.append(np.nan)
                lows.append(0)
                highs.append(0)
        x = np.arange(n_sys) + (i - (len(done) - 1) / 2) * width
        ax.errorbar(
            x, meds, yerr=[lows, highs], fmt="o", capsize=3,
            color=cmap(i % 20), markersize=6, linewidth=0, label=_short(stance),
        )
    ax.set_xticks(np.arange(n_sys))
    ax.set_xticklabels(SYS_KEYS, fontsize=10)
    ax.set_ylabel(r"$P(C=1 \mid \mathrm{data})$  (posterior median + 94% interval)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.999, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.axhline(0.001, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_title(
        f"System root-credence posterior across {len(done)}/13 stances\n"
        f"(exact-tree baseline + soft anchors; {draws_label})"
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "root_C_across_stances.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global OUT_DIR
    smoke = "--smoke" in sys.argv
    if smoke:
        OUT_DIR = Path("results/all_stances_exact_baseline_smoke")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = build_config(smoke=smoke)
    if smoke:
        stances = STANCE_ORDER[:2]
    elif "--only" in sys.argv:
        requested = sys.argv[sys.argv.index("--only") + 1:]
        stances = [s for s in STANCE_ORDER if s in requested]
        if not stances:
            raise SystemExit(f"--only: no STANCE_ORDER match in {requested!r}")
    else:
        stances = STANCE_ORDER
    draws_label = f"{cfg.NUM_SAMPLES} draws x {cfg.NUM_CHAINS} chains"

    print(f"{'SMOKE ' if smoke else ''}all-stances exact-tree baseline fit")
    print("  builder: MultiSystemExactTreeBuilder (exact-tree / sum-product)")
    print("  leaf: three_state | POOL_BETAS_BY_LABEL: False | USE_EXPERT_SHIFTS: False")
    print(f"  soft anchors: {SOFT_ANCHORS}")
    print(
        f"  sampling: {draws_label}, tune={cfg.NUM_TUNE}, "
        f"target_accept={cfg.TARGET_ACCEPT}"
    )
    print(f"  stances: {len(stances)} | out: {OUT_DIR}\n")

    total_t0 = time.time()
    failed = []
    for i, stance in enumerate(stances, 1):
        print(f"=== [{i}/{len(stances)}] {stance} ===")
        fit_one(stance, cfg)
        if not (OUT_DIR / f"{slug_file(stance)}.nc").exists():
            failed.append(stance)
        regenerate_summary(draws_label)
        print(f"--- cumulative: {(time.time() - total_t0) / 60:.1f} min ---\n")

    n_done = sum((OUT_DIR / f"{slug_file(s)}.nc").exists() for s in stances)
    print(
        f"\nDone. {n_done}/{len(stances)} stances fitted in "
        f"{(time.time() - total_t0) / 60:.1f} min."
    )
    if failed:
        print(f"FAILED stances: {failed}")
    print(
        f"See {OUT_DIR}/summary.json and {OUT_DIR}/root_C_across_stances.png"
    )


if __name__ == "__main__":
    main()
