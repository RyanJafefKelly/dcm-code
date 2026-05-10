"""Per-indicator posterior predictive check.

For each (system, indicator-with-data) pair in a recovery run:
  1. Walk posterior draws of (β_pres, β_abs, a, kappa, C) for that system.
  2. Compute the predicted 7-category rating distribution per draw via
     parent_probs_by_indicator_draw + indicator_category_probs (reused from
     gwt_full_exact_recovery.py).
  3. Aggregate to posterior mean predicted proportions and a 94% predictive
     band over per-draw simulated counts.
  4. Score against observed rating counts: chi-square (df=6), KL(observed‖predicted),
     and a coverage flag (whether observed proportion vector falls inside the band
     elementwise).

Writes per_indicator_ppc.csv inside the input run directory.

Usage:
  python notebooks/asymmetric_prior_sweep_2026-05-10/per_indicator_ppc.py \
      --run-dir notebooks/asymmetric_prior_sweep_2026-05-10/runs/<run_id>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import arviz as az
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SISTER_DIR = REPO_ROOT / "notebooks" / "synthetic_validation_2026-05-06"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SISTER_DIR) not in sys.path:
    sys.path.insert(0, str(SISTER_DIR))

from analyse_tree_pooling import pooled_beta_draws_by_node  # noqa: E402
from dcm_model import ModelConfig, MultiSystemDataProcessor, load_data, node_key  # noqa: E402
from gwt_full_exact_recovery import (  # noqa: E402
    indicator_category_probs,
    iter_tree_nodes,
    parent_probs_by_indicator_draw,
    posterior_draws,
)
from gwt_oracle_internal_identifiability import STANCE  # noqa: E402
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS  # noqa: E402

SYSTEM_VAR_PREFIX = {
    "Human": "human",
    "Chicken": "chicken",
    "2024 Leading Chat LLMs": "2024_leading_chat_llms",
    "ELIZA": "eliza",
}


def _load_synthetic_or_real_stance(run_dir: Path) -> Tuple[Dict, Dict | None]:
    synth = run_dir / "synthetic_stance_data.json"
    if synth.exists():
        with open(synth) as f:
            stance_data = json.load(f)
        return stance_data, json.loads((run_dir / "truth.json").read_text()) if (run_dir / "truth.json").exists() else None
    cfg = ModelConfig(INDICATOR_STATE_MODEL="three_state")
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    return stance_data, None


def per_indicator_ppc(run_dir: Path, n_simulate_per_draw: int = 1) -> pd.DataFrame:
    idata = az.from_netcdf(str(run_dir / "fit.nc"))
    post = idata.posterior

    stance_data, _ = _load_synthetic_or_real_stance(run_dir)
    cfg = ModelConfig(INDICATOR_STATE_MODEL="three_state")
    proc = MultiSystemDataProcessor(cfg)
    systems = [s for s, _ in ANCHORED_SYSTEM_CONFIGS]
    proc.process(stance_data, systems)

    beta_pres_by_key, beta_abs_by_key = pooled_beta_draws_by_node(idata, stance_data)
    a_draws = posterior_draws(post, "a").reshape(-1)
    kappa_draws = posterior_draws(post, "kappa")
    n_draws = a_draws.shape[0]

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

    rng = np.random.default_rng(20260510)
    rows: List[Dict] = []
    pred_props_acc: Dict[Tuple[str, str], np.ndarray] = {}
    pred_count_draws: Dict[Tuple[str, str], np.ndarray] = {}

    for i in range(n_draws):
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
                probs = indicator_category_probs(parent_q[key], bp[key], ba[key], a_i, kappa_i)
                gkey = (system, key)
                pred_props_acc.setdefault(gkey, np.zeros(7))[:] += probs
                n_obs = int(obs_counts[gkey].sum())
                if n_obs > 0:
                    sim = rng.multinomial(n_obs, probs)
                    pred_count_draws.setdefault(
                        gkey, np.zeros((n_draws, 7), dtype=float)
                    )[i, :] = sim

    eps = 1e-12
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
    parser.add_argument(
        "--out-name",
        default="per_indicator_ppc.csv",
        help="CSV filename written into the run dir.",
    )
    args = parser.parse_args()
    run_dir: Path = args.run_dir
    if not run_dir.is_absolute():
        run_dir = (REPO_ROOT / run_dir).resolve()
    df = per_indicator_ppc(run_dir)
    out = run_dir / args.out_name
    df.to_csv(out, index=False)
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    by_system = df.groupby("system").agg(
        n_indicators=("chi2", "size"),
        mean_chi2=("chi2", "mean"),
        mean_kl=("kl", "mean"),
        cov94_frac=("cov94", "mean"),
    )
    print()
    print("Per-system summary:")
    print(by_system.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
