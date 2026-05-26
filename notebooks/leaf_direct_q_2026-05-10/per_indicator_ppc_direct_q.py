"""Per-indicator PPC adapted for the direct_q leaf.

Mirrors notebooks/asymmetric_prior_sweep_2026-05-10/per_indicator_ppc.py but
substitutes the predicted category-probability function with the direct_q
form (no latent z mixture):

    tilde_q = parent_q * beta_pres + (1 - parent_q) * beta_abs
    probs   = ordered_probit_probs(kappa, a * tilde_q)

Writes per_indicator_ppc.csv inside the run dir.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import arviz as az
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTH_DIR = REPO_ROOT / "notebooks" / "synthetic_validation_2026-05-06"
SWEEP_DIR = REPO_ROOT / "notebooks" / "asymmetric_prior_sweep_2026-05-10"
for d in (REPO_ROOT, SYNTH_DIR, SWEEP_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from analyse_tree_pooling import pooled_beta_draws_by_node  # noqa: E402
from dcm_model import ModelConfig, MultiSystemDataProcessor, load_data  # noqa: E402
from gwt_full_exact_recovery import (  # noqa: E402
    iter_tree_nodes,
    parent_probs_by_indicator_draw,
    posterior_draws,
)
from gwt_oracle_internal_identifiability import STANCE, ordered_probit_probs  # noqa: E402
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS  # noqa: E402

SYSTEM_VAR_PREFIX = {
    "Human": "human",
    "Chicken": "chicken",
    "2024 Leading Chat LLMs": "2024_leading_chat_llms",
    "ELIZA": "eliza",
}


def direct_q_category_probs(
    parent_q: float,
    beta_pres: float,
    beta_abs: float,
    a: float,
    kappa: np.ndarray,
) -> np.ndarray:
    """Predicted 7-category distribution under the direct_q leaf.

    eta_je collapses across experts to eta = a * tilde_q (b_e = 0 in the
    USE_EXPERT_SHIFTS=False configuration this PPC script is run under).
    """
    tilde_q = parent_q * beta_pres + (1.0 - parent_q) * beta_abs
    probs = ordered_probit_probs(kappa, a * tilde_q)
    probs = np.clip(probs, 1e-12, 1.0)
    return probs / probs.sum()


def per_indicator_ppc(run_dir: Path, n_draw_subsample: int = 200, seed: int = 20260510) -> pd.DataFrame:
    idata = az.from_netcdf(str(run_dir / "fit.nc"))
    post = idata.posterior

    syn_path = run_dir / "synthetic_stance_data.json"
    with syn_path.open() as f:
        stance_data = json.load(f)

    cfg = ModelConfig(INDICATOR_STATE_MODEL="direct_q")
    proc = MultiSystemDataProcessor(cfg)
    systems = [s for s, _ in ANCHORED_SYSTEM_CONFIGS]
    proc.process(stance_data, systems)

    beta_pres_by_key, beta_abs_by_key = pooled_beta_draws_by_node(idata, stance_data)
    a_draws = posterior_draws(post, "a").reshape(-1)
    kappa_draws = posterior_draws(post, "kappa")
    n_draws_total = a_draws.shape[0]
    rng = np.random.default_rng(seed)
    if n_draw_subsample is not None and n_draw_subsample < n_draws_total:
        idx = rng.choice(n_draws_total, size=n_draw_subsample, replace=False)
    else:
        idx = np.arange(n_draws_total)
    n_draws = len(idx)

    c_draws_by_system: Dict[str, np.ndarray] = {}
    for system in systems:
        var = f"{SYSTEM_VAR_PREFIX[system]}__global_workspace_theory_C"
        c_draws_by_system[system] = posterior_draws(post, var).reshape(-1)

    indicator_meta: Dict[str, Dict] = {}
    for node, key, _, depth, top_feature in iter_tree_nodes(stance_data):
        if (node.get("type") or "").lower() == "indicator":
            indicator_meta[key] = {
                "indicator": node["name"],
                "top_feature": top_feature,
                "depth": depth,
            }

    obs_counts: Dict[Tuple[str, str], np.ndarray] = {}
    for system, obs_by_key in proc.system_observations.items():
        for key, obs in obs_by_key.items():
            counts = np.zeros(7, dtype=float)
            for _, rating in obs:
                counts[int(rating)] += 1
            obs_counts[(system, key)] = counts

    pred_props_acc: Dict[Tuple[str, str], np.ndarray] = {}
    pred_count_draws: Dict[Tuple[str, str], np.ndarray] = {}

    for di, i in enumerate(idx):
        bp = {k: float(v[i]) for k, v in beta_pres_by_key.items()}
        ba = {k: float(v[i]) for k, v in beta_abs_by_key.items()}
        kappa_i = np.asarray(kappa_draws[i], dtype=float)
        a_i = float(a_draws[i])
        for system in systems:
            parent_q = parent_probs_by_indicator_draw(
                stance_data, bp, ba, float(c_draws_by_system[system][i])
            )
            obs_by_key = proc.system_observations[system]
            for key in obs_by_key:
                if key not in indicator_meta:
                    continue
                probs = direct_q_category_probs(parent_q[key], bp[key], ba[key], a_i, kappa_i)
                gkey = (system, key)
                pred_props_acc.setdefault(gkey, np.zeros(7))[:] += probs
                n_obs = int(obs_counts[gkey].sum())
                if n_obs > 0:
                    sim = rng.multinomial(n_obs, probs)
                    pred_count_draws.setdefault(
                        gkey, np.zeros((n_draws, 7), dtype=float)
                    )[di, :] = sim

    eps = 1e-12
    rows: List[Dict] = []
    for (system, key), props in pred_props_acc.items():
        n_obs = int(obs_counts[(system, key)].sum())
        if n_obs == 0:
            continue
        mean_props = props / n_draws
        mean_props = np.clip(mean_props, eps, 1.0)
        mean_props = mean_props / mean_props.sum()
        observed = obs_counts[(system, key)]
        observed_props = observed / n_obs
        expected_counts = mean_props * n_obs
        chi2 = float(np.sum((observed - expected_counts) ** 2 / np.clip(expected_counts, 1e-9, None)))
        kl = float(
            np.sum(
                observed_props
                * (np.log(np.clip(observed_props, eps, None)) - np.log(np.clip(mean_props, eps, None)))
            )
        )
        sims = pred_count_draws.get((system, key))
        if sims is not None:
            sim_props = sims / max(n_obs, 1)
            band_lo = np.percentile(sim_props, 3, axis=0)
            band_hi = np.percentile(sim_props, 97, axis=0)
            cov94 = bool(np.all((observed_props >= band_lo) & (observed_props <= band_hi)))
        else:
            cov94 = False
        meta = indicator_meta[key]
        rows.append({
            "system": system,
            "node_key": key,
            "indicator": meta["indicator"],
            "top_feature": meta["top_feature"],
            "depth": meta["depth"],
            "n_ratings": n_obs,
            "chi2": chi2,
            "kl": kl,
            "cov94": cov94,
            **{f"obs_p{k}": float(observed_props[k]) for k in range(7)},
            **{f"pred_p{k}": float(mean_props[k]) for k in range(7)},
        })
    return pd.DataFrame(rows).sort_values(["system", "top_feature", "depth", "indicator"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--n-draw-subsample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    df = per_indicator_ppc(args.run_dir, n_draw_subsample=args.n_draw_subsample, seed=args.seed)
    out = args.run_dir / "per_indicator_ppc.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    print(
        f"n_indicator_system_pairs={len(df)} "
        f"chi2_mean={df['chi2'].mean():.2f} "
        f"chi2_median={df['chi2'].median():.2f} "
        f"cov94_frac={df['cov94'].mean():.3f}"
    )


if __name__ == "__main__":
    main()
