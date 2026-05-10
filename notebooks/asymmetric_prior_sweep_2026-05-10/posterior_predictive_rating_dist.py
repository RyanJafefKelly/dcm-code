"""Per-system posterior-predictive rating distributions across sweep variants.

For each system, plots:
  - Observed rating distribution (the real-data ratings or the synthetic
    seed's ordinal_by_system_indicator) as bars (green).
  - Posterior-predictive expected category proportions for each prior
    variant on the same fit type, as line+marker overlays.

Mirrors the existing observed_vs_expected_rating_dists.py style but compares
multiple posterior fits side-by-side.

Usage:
  python notebooks/asymmetric_prior_sweep_2026-05-10/posterior_predictive_rating_dist.py \
      --kind synthetic   # the four synthetic seed-06 fits + baseline
  python notebooks/asymmetric_prior_sweep_2026-05-10/posterior_predictive_rating_dist.py \
      --kind real_data
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SISTER_DIR = REPO_ROOT / "notebooks" / "synthetic_validation_2026-05-06"
SWEEP_DIR = Path(__file__).resolve().parent
for d in (REPO_ROOT, SISTER_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from analyse_tree_pooling import pooled_beta_draws_by_node  # noqa: E402
from dcm_model import ModelConfig, MultiSystemDataProcessor, load_data, node_key  # noqa: E402
from gwt_full_exact_recovery import (  # noqa: E402
    indicator_category_probs,
    parent_probs_by_indicator_draw,
    posterior_draws,
)
from gwt_oracle_internal_identifiability import STANCE  # noqa: E402
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS  # noqa: E402

SYSTEMS = [s for s, _ in ANCHORED_SYSTEM_CONFIGS]
SYS_PREFIX = {
    "Human": "human",
    "Chicken": "chicken",
    "2024 Leading Chat LLMs": "2024_leading_chat_llms",
    "ELIZA": "eliza",
}
ORDINAL_BIN_EDGES = (0.05, 0.20, 0.40, 0.60, 0.80, 0.95)
N_CATS = 7


def is_missing(v) -> bool:
    if v is None:
        return True
    return str(v).strip().lower() in {"-1", "-1.0", "none", "", "unsure", "not tested"}


def prob_to_cat(p: float) -> int:
    for k, e in enumerate(ORDINAL_BIN_EDGES):
        if p < e:
            return k
    return N_CATS - 1


def collect_observed_ratings(stance_data: Dict, system: str) -> List[int]:
    out: List[int] = []
    def walk(node):
        if (node.get("type") or "").lower() == "indicator":
            obs = node.get("observations", {}).get(system)
            if obs:
                for v in obs.get("values", []):
                    if not is_missing(v):
                        out.append(prob_to_cat(float(v)))
        for child in node.get("evidencers", []):
            walk(child)
    for child in stance_data.get("evidencers", []):
        walk(child)
    return out


def predicted_distribution(idata, stance_data: Dict, system: str, max_draws: int = 200) -> np.ndarray:
    """Posterior-mean expected rating distribution over all observed indicators
    for this system, marginalised over draws and observed indicators.
    Thins draws uniformly to max_draws to keep wall-clock < ~30s per (fit, system).
    """
    post = idata.posterior
    bp_by_key, ba_by_key = pooled_beta_draws_by_node(idata, stance_data)
    a_draws = posterior_draws(post, "a").reshape(-1)
    kappa_draws = posterior_draws(post, "kappa")
    var = f"{SYS_PREFIX[system]}__global_workspace_theory_C"
    c_draws = posterior_draws(post, var).reshape(-1)
    n_total = a_draws.shape[0]
    if n_total > max_draws:
        sel = np.linspace(0, n_total - 1, max_draws).astype(int)
        a_draws = a_draws[sel]
        kappa_draws = kappa_draws[sel]
        c_draws = c_draws[sel]
        bp_by_key = {k: v[sel] for k, v in bp_by_key.items()}
        ba_by_key = {k: v[sel] for k, v in ba_by_key.items()}
    n_draws = a_draws.shape[0]

    cfg = ModelConfig(INDICATOR_STATE_MODEL="three_state")
    proc = MultiSystemDataProcessor(cfg)
    proc.process(stance_data, SYSTEMS)
    obs_keys = [k for k in proc.system_observations[system]]
    if not obs_keys:
        return np.zeros(N_CATS)

    accum = np.zeros(N_CATS)
    for i in range(n_draws):
        bp = {k: float(v[i]) for k, v in bp_by_key.items()}
        ba = {k: float(v[i]) for k, v in ba_by_key.items()}
        parent_q = parent_probs_by_indicator_draw(stance_data, bp, ba, float(c_draws[i]))
        a_i = float(a_draws[i])
        kappa_i = np.asarray(kappa_draws[i], dtype=float)
        for k in obs_keys:
            probs = indicator_category_probs(parent_q[k], bp[k], ba[k], a_i, kappa_i)
            n_obs = sum(1 for _ in proc.system_observations[system][k])
            accum += probs * n_obs  # weight by rating count for that indicator
    total_obs = sum(len(v) for v in proc.system_observations[system].values())
    if total_obs == 0:
        return accum
    return accum / (n_draws * total_obs)


def load_real_data_stance() -> Dict:
    cfg = ModelConfig(INDICATOR_STATE_MODEL="three_state")
    return next(s for s in load_data(cfg) if s["name"] == STANCE)


def load_synthetic_stance(run_dir: Path) -> Dict:
    return json.loads((run_dir / "synthetic_stance_data.json").read_text())


def synthetic_runs() -> List[Tuple[str, Path]]:
    """Returns (label, run_dir) pairs for synthetic comparisons."""
    base_dir = SISTER_DIR / "runs/full_exact_recovery"
    base = (
        base_dir
        / "exact_latent_tree__full_exact_tree__three_state_binomial_2__"
          "exact_tree_production_medians__current_gwt_rater_design__seed20260506"
    )
    runs: List[Tuple[str, Path]] = []
    if base.exists():
        runs.append(("baseline (paper-mu)", base))
    sweep = SWEEP_DIR / "runs/synthetic"
    if sweep.exists():
        for d in sorted(sweep.iterdir()):
            if "smoke" in d.name:
                continue
            tag = d.name.split("__")[-1]
            runs.append((tag, d))
    return runs


def real_runs() -> List[Tuple[str, Path]]:
    runs: List[Tuple[str, Path]] = []
    base = REPO_ROOT / "results/gwt_exact_tree/three_state_pooled_abs_by_sd_exact_anchored.nc"
    if base.exists():
        runs.append(("baseline (paper-mu, prod)", base))
    sweep = SWEEP_DIR / "runs/real_data"
    if sweep.exists():
        for d in sorted(sweep.iterdir()):
            if (d / "fit.nc").exists():
                runs.append((d.name, d))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["synthetic", "real_data"], default="synthetic")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.kind == "synthetic":
        runs = synthetic_runs()
        out = args.out or (SWEEP_DIR / "figs" / "ppc_rating_dist_synthetic.png")
    else:
        runs = real_runs()
        out = args.out or (SWEEP_DIR / "figs" / "ppc_rating_dist_real_data.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(SYSTEMS), figsize=(5.0 * len(SYSTEMS), 4.5), sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax_i, system in enumerate(SYSTEMS):
        ax = axes[ax_i]
        for ri, (label, path) in enumerate(runs):
            if path.is_dir():
                stance = load_synthetic_stance(path) if args.kind == "synthetic" else load_real_data_stance()
                idata = az.from_netcdf(str(path / "fit.nc"))
            else:
                stance = load_real_data_stance()
                idata = az.from_netcdf(str(path))
            pred = predicted_distribution(idata, stance, system)
            ax.plot(np.arange(N_CATS), pred, marker="o", lw=1.5, color=cmap(ri), label=label)

        # Observed
        if args.kind == "synthetic":
            stance0 = load_synthetic_stance(runs[-1][1])  # use any synthetic stance (same DGP)
        else:
            stance0 = load_real_data_stance()
        obs = collect_observed_ratings(stance0, system)
        counts = Counter(obs)
        obs_props = np.array([counts.get(c, 0) for c in range(N_CATS)], dtype=float)
        if obs_props.sum() > 0:
            obs_props = obs_props / obs_props.sum()
        ax.bar(np.arange(N_CATS), obs_props, color="lightgrey", edgecolor="black", alpha=0.7,
               label=f"observed (n={int(sum(counts.values()))})", zorder=0)
        ax.set_title(system, fontsize=10)
        ax.set_xlabel("ordinal category (0..6)")
        ax.set_xticks(range(N_CATS))
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("proportion of ratings")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Posterior-predictive rating distributions — {args.kind} fits "
        f"(asymmetric β prior sweep + baseline)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
