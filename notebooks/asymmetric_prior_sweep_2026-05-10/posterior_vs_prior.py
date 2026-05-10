"""Posterior-vs-prior diagnostic for the asymmetric-prior sweep.

For a given fit directory, reports:
  1. β_pres / β_abs posterior summaries per (s, d) group, with the prior
     mean (paper or override) and a contraction ratio (posterior SD over
     prior approximate SD in natural space).
  2. Free root C posterior vs prior (Beta(DEFAULT_ALPHA, DEFAULT_BETA)).

Tests the hypothesis: if posterior β_pres has drifted back from the
override mean (e.g. 0.90) to something close to baseline (e.g. 0.65), the
prior signal never reaches C.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import arviz as az
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcm_model import EvidenceProcessor, ModelConfig, _sanitize_label  # noqa: E402

SYSTEM_VAR_PREFIX = {
    "Human": "human",
    "Chicken": "chicken",
    "2024 Leading Chat LLMs": "2024_leading_chat_llms",
    "ELIZA": "eliza",
}


def _load_config(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())["config"]
    meta = run_dir / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text())["config"]
    raise FileNotFoundError(f"No config.json or meta.json in {run_dir}")


def _logit_normal_natural_sd(mu_natural: float, sigma_logit: float, n: int = 5000) -> float:
    """Approximate prior natural-scale SD for a logit-Normal centred at mu."""
    eps = 1e-9
    mu_natural = min(max(mu_natural, eps), 1.0 - eps)
    logit_mu = float(np.log(mu_natural / (1.0 - mu_natural)))
    rng = np.random.default_rng(20260510)
    samples = 1.0 / (1.0 + np.exp(-(logit_mu + sigma_logit * rng.standard_normal(n))))
    return float(np.std(samples))


def beta_posterior_vs_prior(idata: "az.InferenceData", cfg: dict) -> pd.DataFrame:
    label_pool_sigma = float(cfg.get("LABEL_POOL_SIGMA", 0.5))
    override_sigma = cfg.get("BETA_OVERRIDE_SIGMA")
    pres_override = cfg.get("BETA_PRES_OVERRIDE_MEAN")
    abs_override = cfg.get("BETA_ABS_OVERRIDE_MEAN")
    sigma_used = float(override_sigma) if override_sigma is not None else label_pool_sigma

    cfg_obj = ModelConfig(INDICATOR_STATE_MODEL="three_state")
    evidence = EvidenceProcessor(cfg_obj)

    rows: List[Dict] = []
    post = idata.posterior
    for var in post.data_vars:
        name = str(var)
        if not (name.startswith("beta_pres__") or name.startswith("beta_abs__")):
            continue
        if "tilde" in name:
            continue
        kind = "pres" if name.startswith("beta_pres__") else "abs"
        # Names: beta_pres__{support}__{demand} OR beta_abs__{support}__{demand} OR beta_abs__{demand}
        rest = name.split("__", 1)[1]
        parts = rest.split("__")
        if kind == "pres" or len(parts) == 2:
            sup_token, demand_token = parts[0], parts[1]
            support = sup_token.replace("_", " ")
            demand = demand_token.replace("_", " ")
        else:
            support = "no bearing"
            demand = parts[0].replace("_", " ")

        draws = np.asarray(post[var].values).reshape(-1)
        post_mean = float(np.mean(draws))
        post_sd = float(np.std(draws))
        post_p03 = float(np.percentile(draws, 3))
        post_p97 = float(np.percentile(draws, 97))

        if kind == "pres":
            if pres_override is not None:
                prior_mu = float(pres_override)
                this_sigma = sigma_used
            else:
                alpha_p, beta_p, _, _ = evidence.get_beta_parameters(support, demand)
                prior_mu = alpha_p / (alpha_p + beta_p)
                this_sigma = label_pool_sigma
        else:
            if abs_override is not None:
                prior_mu = float(abs_override)
                this_sigma = sigma_used
            else:
                _, _, alpha_a, beta_a = evidence.get_beta_parameters("no bearing", demand)
                prior_mu = alpha_a / (alpha_a + beta_a)
                this_sigma = label_pool_sigma

        prior_sd_nat = _logit_normal_natural_sd(prior_mu, this_sigma)
        contraction = post_sd / prior_sd_nat if prior_sd_nat > 0 else float("nan")
        drift_to_baseline = (
            None if (kind == "pres" and pres_override is None) or (kind == "abs" and abs_override is None)
            else float((post_mean - prior_mu) / max(abs(prior_mu - 0.5), 1e-6))
        )
        rows.append({
            "kind": kind,
            "support": support,
            "demandingness": demand,
            "prior_mu": prior_mu,
            "prior_sigma_logit": this_sigma,
            "prior_sd_natural_approx": prior_sd_nat,
            "post_mean": post_mean,
            "post_sd": post_sd,
            "post_p03": post_p03,
            "post_p97": post_p97,
            "contraction": contraction,
            "fraction_back_to_baseline": drift_to_baseline,
        })
    return pd.DataFrame(rows).sort_values(["kind", "support", "demandingness"])


def root_c_posterior_vs_prior(idata: "az.InferenceData", cfg: dict) -> pd.DataFrame:
    default_alpha = float(cfg.get("DEFAULT_ALPHA", 1))
    default_beta = float(cfg.get("DEFAULT_BETA", 5))
    prior_mean = default_alpha / (default_alpha + default_beta)
    prior_sd = float(np.sqrt(
        default_alpha * default_beta
        / ((default_alpha + default_beta) ** 2 * (default_alpha + default_beta + 1))
    ))
    rows: List[Dict] = []
    post = idata.posterior
    for system, prefix in SYSTEM_VAR_PREFIX.items():
        var = f"{prefix}__global_workspace_theory_C"
        if var not in post.data_vars:
            continue
        draws = np.asarray(post[var].values).reshape(-1)
        post_sd = float(np.std(draws))
        if post_sd < 1e-9:
            continue  # hard-anchored
        rows.append({
            "system": system,
            "prior_mean_beta": prior_mean,
            "prior_sd_beta": prior_sd,
            "post_mean": float(np.mean(draws)),
            "post_median": float(np.median(draws)),
            "post_sd": post_sd,
            "post_p03": float(np.percentile(draws, 3)),
            "post_p97": float(np.percentile(draws, 97)),
            "contraction": post_sd / prior_sd,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir: Path = args.run_dir
    if not run_dir.is_absolute():
        run_dir = (REPO_ROOT / run_dir).resolve()
    cfg = _load_config(run_dir)
    idata = az.from_netcdf(str(run_dir / "fit.nc"))

    beta_df = beta_posterior_vs_prior(idata, cfg)
    beta_df.to_csv(run_dir / "beta_posterior_vs_prior.csv", index=False)
    c_df = root_c_posterior_vs_prior(idata, cfg)
    c_df.to_csv(run_dir / "c_posterior_vs_prior.csv", index=False)

    print(f"=== {run_dir.name} ===")
    print(
        f"prior overrides: pres={cfg.get('BETA_PRES_OVERRIDE_MEAN')}, "
        f"abs={cfg.get('BETA_ABS_OVERRIDE_MEAN')}, "
        f"sigma={cfg.get('BETA_OVERRIDE_SIGMA') or cfg.get('LABEL_POOL_SIGMA')}"
    )
    print()
    print("β posterior vs prior (kind / support / demand / prior_mu → post_mean ± SD ; contraction):")
    for kind in ("pres", "abs"):
        sub = beta_df[beta_df["kind"] == kind]
        if sub.empty:
            continue
        print(f"  -- {kind} --")
        print(f"  prior_mu mean: {sub['prior_mu'].mean():.3f}, "
              f"post_mean mean: {sub['post_mean'].mean():.3f}, "
              f"mean contraction: {sub['contraction'].mean():.3f}")
        worst = sub.iloc[(sub["post_mean"] - sub["prior_mu"]).abs().argsort()[::-1].values[:5]]
        for _, r in worst.iterrows():
            print(
                f"    {r['support']:<22} {r['demandingness']:<22} "
                f"prior {r['prior_mu']:.3f} → post {r['post_mean']:.3f} "
                f"(±{r['post_sd']:.3f}, contract {r['contraction']:.2f})"
            )
    print()
    print("Free root C posterior vs prior:")
    print(c_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
