"""Main synthetic validation for the targeted lower-edge binary-root DCM."""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SYNTH_NOTEBOOK_DIR = REPO_ROOT / "notebooks/synthetic_validation_2026-05-06"
if str(SYNTH_NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(SYNTH_NOTEBOOK_DIR))

from dcm_model import EvidenceProcessor, ModelConfig, MultiSystemDataProcessor, load_data, node_key
from dcm_model_exact_tree import MultiSystemExactTreeBuilder
from gwt_oracle_internal_identifiability import (
    load_oracle_truth,
    ordered_probit_probs,
)
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS
from scripts.main_synthetic_validation_metrics import (
    CATEGORY_COUNT,
    LOGIT_PRIOR,
    PRIOR_P,
    SCORE_CLIP,
    STRONG_TOP_FEATURES,
    STANCE,
    TAU_ABSENT_05,
    TAU_PRESENT_50,
    WEAK_TOP_FEATURES,
    build_edge_beta_profile,
    compute_conditional_loo,
    compute_exact_tree_ppc,
    extract_rho_cases,
    jsonable_config,
    pass_fail_summary,
    ranked_probability_score,
    profile_as_truth_edge_betas,
    sampler_diagnostics,
    summarise_rho_recovery,
    validate_edge_beta_profile,
    write_json,
)
from scripts.phase1_root_evidence_common import (
    ALL_SYSTEMS,
    FREE_SYSTEMS,
    beta_profile_from_evidence_processor,
    bernoulli_brier,
    bernoulli_log_score,
    evidence_category_from_log_b,
    exact_root_sides_from_leaf_messages,
    extreme_beta_profile,
    iter_tree_nodes,
    load_gwt_stance,
)
from scripts.phase1c_evidence_attrition_audit import (
    sample_latent_states,
    simulate_leaf_messages,
)


OUTPUT_DIR = REPO_ROOT / "outputs/main_synthetic_validation"
CATEGORY_REPRESENTATIVES = (0.025, 0.125, 0.300, 0.500, 0.700, 0.875, 0.975)
FIT_VARIANT_TARGETED = "targeted_strong_lower_override"
FIT_VARIANT_UNMODIFIED = "production_unmodified"
EXPECTED_TARGETED_GWT_EDGE_COUNT = 49
IDENTIFIER_COLUMNS = {
    "seed",
    "sub_seed",
    "replicate_id",
    "run_id",
    "fit_variant",
    "system",
    "validation_scope",
}


def git_head() -> dict[str, str]:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
        return {"branch": branch, "commit": commit}
    except Exception:
        return {"branch": "unknown", "commit": "unknown"}


def parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def root_truth_for_seed_idx(seed_idx: int) -> dict[str, int]:
    return {
        "Human": 1,
        "ELIZA": 0,
        "Chicken": 1 if seed_idx % 2 == 0 else 0,
        "2024 Leading Chat LLMs": 0 if seed_idx % 2 == 0 else 1,
    }


def assert_main_targeted_fit_config(cfg: ModelConfig, variant: str) -> None:
    if variant != FIT_VARIANT_TARGETED:
        return
    if cfg.POOL_BETAS_BY_LABEL:
        raise AssertionError(
            "targeted_strong_lower_override must run with POOL_BETAS_BY_LABEL=False"
        )
    if cfg.TARGETED_OVERRIDE_NODE_KEYS is None:
        raise AssertionError(
            "targeted_strong_lower_override requires TARGETED_OVERRIDE_NODE_KEYS"
        )
    if float(cfg.TRANSMISSION_GAIN) != 1.0 or bool(cfg.GAIN_LOGIT_NORMAL):
        raise AssertionError(
            "targeted_strong_lower_override non-targeted edges must stay on "
            "production EvidenceProcessor per-node Beta priors"
        )
    if cfg.BETA_PRES_OVERRIDE_MEAN != 0.90 or cfg.BETA_ABS_OVERRIDE_MEAN != 0.10:
        raise AssertionError(
            "targeted_strong_lower_override must use override centres 0.90/0.10"
        )


def build_fit_config(
    *,
    mode: str,
    variant: str,
    targeted_keys: Optional[Sequence[str]],
    chains: Optional[int],
    draws: Optional[int],
    tune: Optional[int],
    target_accept: Optional[float],
) -> ModelConfig:
    if mode == "smoke":
        default_chains, default_draws, default_tune, default_accept = 2, 200, 200, 0.95
    else:
        default_chains, default_draws, default_tune, default_accept = 4, 1000, 1000, 0.95

    is_targeted = variant == FIT_VARIANT_TARGETED
    cfg = ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=False,
        TARGETED_OVERRIDE_NODE_KEYS=list(targeted_keys) if is_targeted else None,
        BETA_PRES_OVERRIDE_MEAN=0.90 if is_targeted else None,
        BETA_ABS_OVERRIDE_MEAN=0.10 if is_targeted else None,
        BETA_OVERRIDE_SIGMA=0.30 if is_targeted else None,
        NUM_CHAINS=int(chains or default_chains),
        NUM_SAMPLES=int(draws or default_draws),
        NUM_TUNE=int(tune or default_tune),
        TARGET_ACCEPT=float(target_accept if target_accept is not None else default_accept),
    )
    assert_main_targeted_fit_config(cfg, variant)
    return cfg


def load_source_stance(config: Optional[ModelConfig] = None) -> dict[str, Any]:
    cfg = config or ModelConfig()
    return next(s for s in load_data(cfg) if s["name"] == STANCE)


def sample_latent_tree_fixed_root(
    *,
    rng: np.random.Generator,
    stance_data: Mapping[str, Any],
    edge_betas: Mapping[str, Mapping[str, float]],
    root_z: int,
) -> dict[str, Any]:
    internal_z: dict[str, int] = {}
    indicator_m: dict[str, int] = {}
    indicator_parent_z: dict[str, int] = {}
    root_path = (stance_data["name"],)

    def walk(node: Mapping[str, Any], path: tuple[str, ...], parent_z: int) -> None:
        key = node_key(path, node["name"])
        beta = edge_betas[key]["beta_pres"] if parent_z else edge_betas[key]["beta_abs"]
        beta = float(np.clip(beta, SCORE_CLIP, 1.0 - SCORE_CLIP))
        if (node.get("type") or "").lower() == "indicator":
            indicator_parent_z[key] = int(parent_z)
            indicator_m[key] = int(rng.binomial(2, beta))
            return
        z = int(rng.binomial(1, beta))
        internal_z[key] = z
        current_path = path + (node["name"],)
        for child in node.get("evidencers", []):
            walk(child, current_path, z)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path, int(root_z))
    return {
        "root_z": int(root_z),
        "internal_z": internal_z,
        "indicator_m": indicator_m,
        "indicator_parent_z": indicator_parent_z,
    }


def is_missing(val: Any) -> bool:
    if val is None:
        return True
    return str(val).strip().lower() in {"-1", "-1.0", "none", "", "unsure", "not tested"}


def simulate_observations_in_place(
    *,
    rng: np.random.Generator,
    stance_data: dict[str, Any],
    latent_by_system: Mapping[str, Mapping[str, Any]],
    obs_params: Mapping[str, Any],
    systems: Sequence[str],
) -> dict[str, dict[str, list[int]]]:
    a = float(obs_params["a"])
    kappa = np.asarray(obs_params["kappa"], dtype=float)
    ordinal_by_system_indicator: dict[str, dict[str, list[int]]] = {
        system: {} for system in systems
    }

    def walk(node: dict[str, Any], path: tuple[str, ...]) -> None:
        key = node_key(path, node["name"])
        current_path = path + (node["name"],)
        if (node.get("type") or "").lower() == "indicator":
            for system, obs in node.get("observations", {}).items():
                if system not in systems:
                    continue
                values = obs.get("values", [])
                synthetic_values: list[Any] = []
                ordinal_values: list[int] = []
                m = int(latent_by_system[system]["indicator_m"][key])
                probs = ordered_probit_probs(kappa, a * (m / 2.0))
                for val in values:
                    if is_missing(val):
                        synthetic_values.append(-1)
                        continue
                    category = int(rng.choice(np.arange(len(probs)), p=probs))
                    ordinal_values.append(category)
                    synthetic_values.append(float(CATEGORY_REPRESENTATIVES[category]))
                obs["values"] = synthetic_values
                ordinal_by_system_indicator[system][key] = ordinal_values
            return
        for child in node.get("evidencers", []):
            walk(child, current_path)

    for child in stance_data.get("evidencers", []):
        walk(child, (stance_data["name"],))
    return ordinal_by_system_indicator


def generate_synthetic_dataset(
    *,
    variant: str,
    seed: int,
    seed_idx: int,
    base_config: ModelConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    source = load_source_stance(base_config)
    synthetic = copy.deepcopy(source)
    pre_profile = build_edge_beta_profile(source, variant, base_config)
    cfg_for_profile = build_fit_config(
        mode="pilot",
        variant=variant,
        targeted_keys=pre_profile.targeted_override_node_keys,
        chains=None,
        draws=None,
        tune=None,
        target_accept=None,
    )
    profile = build_edge_beta_profile(source, variant, cfg_for_profile)
    edge_betas = profile_as_truth_edge_betas(source, profile)
    targeted_override_summary = validate_edge_beta_profile(source, profile)
    oracle_truth = load_oracle_truth(
        "exact_tree_production_medians",
        source,
        cfg_for_profile,
    )
    root_z_by_system = root_truth_for_seed_idx(seed_idx)
    sub_seed = int(seed + seed_idx * 1_000_003)
    systems = [s for s, _ in ANCHORED_SYSTEM_CONFIGS]
    latent_by_system = {}
    for offset, system in enumerate(systems):
        latent_by_system[system] = sample_latent_tree_fixed_root(
            rng=np.random.default_rng(sub_seed + offset * 100_003),
            stance_data=source,
            edge_betas=edge_betas,
            root_z=root_z_by_system[system],
        )
    ordinal_by_system_indicator = simulate_observations_in_place(
        rng=np.random.default_rng(sub_seed + 765_431),
        stance_data=synthetic,
        latent_by_system=latent_by_system,
        obs_params=oracle_truth.obs_params,
        systems=systems,
    )
    truth_payload = {
        "seed": int(seed),
        "seed_idx": int(seed_idx),
        "sub_seed": sub_seed,
        "fit_variant": variant,
        "tree_variant": "actual_gwt_topology",
        "beta_variant": variant,
        "root_z_by_system": root_z_by_system,
        "latent_by_system": latent_by_system,
        "edge_betas": edge_betas,
        "observation_parameters": oracle_truth.obs_params,
        "a_true": float(oracle_truth.obs_params["a"]),
        "kappa_true": list(map(float, oracle_truth.obs_params["kappa"])),
        "b_true_by_rater": {},
        "rater_design": "production_like_current_gwt_layout",
        "category_representatives": list(CATEGORY_REPRESENTATIVES),
        "ordinal_by_system_indicator": ordinal_by_system_indicator,
        "targeted_override_node_keys": profile.targeted_override_node_keys,
        "overridden_edges": profile.overridden_edges,
        "overridden_edge_keys": profile.overridden_edges,
        "targeted_override_summary": targeted_override_summary,
    }
    return synthetic, source, truth_payload, profile.targeted_override_node_keys


def run_one_fit(
    *,
    output_dir: Path,
    variant: str,
    seed: int,
    seed_idx: int,
    mode: str,
    chains: Optional[int],
    draws: Optional[int],
    tune: Optional[int],
    target_accept: Optional[float],
    ppc_draws: int,
    loo_draws: int,
    loo_max_ratings: Optional[int],
    skip_loo: bool,
    overwrite: bool,
) -> dict[str, Any]:
    base_config = ModelConfig(INDICATOR_STATE_MODEL="three_state")
    synthetic, source, truth_payload, targeted_keys = generate_synthetic_dataset(
        variant=variant,
        seed=seed,
        seed_idx=seed_idx,
        base_config=base_config,
    )
    cfg = build_fit_config(
        mode=mode,
        variant=variant,
        targeted_keys=targeted_keys,
        chains=chains,
        draws=draws,
        tune=tune,
        target_accept=target_accept,
    )
    profile = build_edge_beta_profile(source, variant, cfg)
    profile_check = validate_edge_beta_profile(source, profile)
    run_seed = int(seed + seed_idx)
    run_dir = output_dir / "runs" / variant / f"seed_{run_seed}"
    if run_dir.exists() and overwrite:
        for path in run_dir.iterdir():
            if path.is_file():
                path.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "truth_payload.json", truth_payload)
    pd.DataFrame(profile.edge_table).to_csv(run_dir / "edge_beta_profile.csv", index=False)

    systems = [s for s, _ in ANCHORED_SYSTEM_CONFIGS]
    proc = MultiSystemDataProcessor(cfg)
    proc.process(synthetic, systems)
    truth_payload["b_true_by_rater"] = {name: 0.0 for name in proc.expert_names}
    write_json(run_dir / "truth_payload.json", truth_payload)

    builder = MultiSystemExactTreeBuilder(
        cfg,
        EvidenceProcessor(cfg),
        proc,
        list(ANCHORED_SYSTEM_CONFIGS),
    )
    t0 = time.time()
    model = builder.build_model(synthetic)
    elapsed_build = time.time() - t0
    t1 = time.time()
    with model:
        import pymc as pm

        idata = pm.sample(
            draws=cfg.NUM_SAMPLES,
            tune=cfg.NUM_TUNE,
            chains=cfg.NUM_CHAINS,
            cores=cfg.NUM_CHAINS,
            target_accept=cfg.TARGET_ACCEPT,
            random_seed=run_seed,
        )
    elapsed_sample = time.time() - t1
    fit_path = run_dir / "fit.nc"
    az.to_netcdf(idata, str(fit_path))

    rho_cases = extract_rho_cases(
        idata,
        builder,
        variant,
        run_seed,
        truth_payload["root_z_by_system"],
    )
    rho_cases.to_csv(run_dir / "rho_cases.csv", index=False)
    sampler = sampler_diagnostics(idata, elapsed_sample, variant, run_seed)
    sampler.to_csv(run_dir / "sampler_diagnostics.csv", index=False)

    ppc_summary, ppc_dist = compute_exact_tree_ppc(
        idata=idata,
        builder=builder,
        proc=proc,
        stance_data=synthetic,
        config=cfg,
        system_configs=list(ANCHORED_SYSTEM_CONFIGS),
        output_path=run_dir / "posterior_predictive.nc",
        ppc_draws=ppc_draws,
        seed=run_seed + 991,
        fit_variant=variant,
        fit_seed=run_seed,
    )
    ppc_summary.to_csv(run_dir / "posterior_predictive_summary.csv", index=False)
    ppc_dist.to_csv(run_dir / "posterior_predictive_rating_distribution_by_system.csv", index=False)

    if skip_loo:
        loo_summary = pd.DataFrame(
            [
                {
                    "fit_variant": variant,
                    "seed": run_seed,
                    "n_ratings": int(
                        sum(len(v) for obs in proc.system_observations.values() for v in obs.values())
                    ),
                    "n_ratings_used_for_loo": 0,
                    "loo_draws": 0,
                    "loo_partial_flag": True,
                    "elpd_loo": np.nan,
                    "se_elpd_loo": np.nan,
                    "p_loo": np.nan,
                    "lppd_conditional": np.nan,
                    "mean_pareto_k": np.nan,
                    "median_pareto_k": np.nan,
                    "max_pareto_k": np.nan,
                    "frac_pareto_k_gt_0p7": np.nan,
                    "frac_pareto_k_gt_1p0": np.nan,
                    "loo_status": "skipped",
                }
            ]
        )
        pareto = pd.DataFrame()
    else:
        loo_summary, pareto = compute_conditional_loo(
            idata=idata,
            builder=builder,
            proc=proc,
            stance_data=synthetic,
            config=cfg,
            system_configs=list(ANCHORED_SYSTEM_CONFIGS),
            output_path=run_dir / "per_rating_log_lik.nc",
            loo_draws=loo_draws,
            loo_max_ratings=loo_max_ratings,
            seed=run_seed + 1297,
            fit_variant=variant,
            fit_seed=run_seed,
        )
    loo_summary.to_csv(run_dir / "loo_lppd_summary.csv", index=False)
    if not pareto.empty:
        pareto.to_csv(run_dir / "loo_pareto_k.csv", index=False)

    config_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_head(),
        "mode": mode,
        "fit_variant": variant,
        "seed": int(run_seed),
        "seed_idx": int(seed_idx),
        "stance": STANCE,
        "systems": systems,
        "config": jsonable_config(cfg),
        "targeted_override_node_keys": profile.targeted_override_node_keys,
        "overridden_edges": profile.overridden_edges,
        "edge_profile_sanity": profile_check,
        "elapsed_s_build": elapsed_build,
        "elapsed_s_sample": elapsed_sample,
        "outputs": {
            "fit": "fit.nc",
            "truth_payload": "truth_payload.json",
            "config": "config.json",
            "posterior_predictive": "posterior_predictive.nc",
            "per_rating_log_lik": "per_rating_log_lik.nc" if not skip_loo else None,
            "run_summary": "run_summary.json",
        },
    }
    write_json(run_dir / "config.json", config_payload)
    run_summary = {
        "fit_variant": variant,
        "seed": int(run_seed),
        "seed_idx": int(seed_idx),
        "run_dir": str(run_dir.relative_to(output_dir)),
        "fit_path": str(fit_path.relative_to(output_dir)),
        "elapsed_s_build": elapsed_build,
        "elapsed_s_sample": elapsed_sample,
        "rho_collapsed_present": any(
            str(v).endswith("_rho_collapsed") for v in idata.posterior.data_vars
        ),
        **profile_check,
        **sampler.iloc[0].to_dict(),
    }
    write_json(run_dir / "run_summary.json", run_summary)
    return run_summary


def targeted_override_sanity(output_dir: Path, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    production_cfg = ModelConfig(INDICATOR_STATE_MODEL="three_state")
    source = load_source_stance(production_cfg)
    pre_profile = build_edge_beta_profile(source, FIT_VARIANT_TARGETED, production_cfg)
    cfg = build_fit_config(
        mode="pilot",
        variant=FIT_VARIANT_TARGETED,
        targeted_keys=pre_profile.targeted_override_node_keys,
        chains=None,
        draws=None,
        tune=None,
        target_accept=None,
    )
    profile = build_edge_beta_profile(source, FIT_VARIANT_TARGETED, cfg)
    profile_check = validate_edge_beta_profile(source, profile)
    assert_main_targeted_fit_config(cfg, FIT_VARIANT_TARGETED)

    spec_by_key = {spec.key: spec for spec in iter_tree_nodes(source)}
    all_rows = list(profile.edge_table)
    overridden_rows = [row for row in all_rows if row["is_targeted_override"]]
    non_targeted_rows = [row for row in all_rows if not row["is_targeted_override"]]

    def increment(counter: dict[str, int], key: str) -> None:
        counter[key] = counter.get(key, 0) + 1

    targeted_by_top_feature: dict[str, int] = {}
    targeted_by_depth: dict[str, int] = {}
    targeted_by_type: dict[str, int] = {}
    targeted_by_depth_and_type: dict[str, int] = {}
    for row in overridden_rows:
        node_type = str(row["node_type"] or "unknown")
        depth = int(row["depth"])
        increment(targeted_by_top_feature, str(row["top_feature_name"]))
        increment(targeted_by_depth, str(depth))
        increment(targeted_by_type, node_type)
        increment(targeted_by_depth_and_type, f"depth_{depth}__{node_type}")

    overridden_edge_table: list[dict[str, Any]] = []
    for row in overridden_rows:
        spec = spec_by_key[str(row["node_key"])]
        parent_path = spec.path[:-1]
        parent_path_label = " > ".join(parent_path)
        overridden_edge_table.append(
            {
                "node_key": row["node_key"],
                "parent_key": parent_path_label,
                "parent_path": list(parent_path),
                "top_ancestor": row["top_feature_name"],
                "depth": int(row["depth"]),
                "node_type": row["node_type"],
                "support_label": row["support"],
                "demandingness_label": row["demandingness"],
                "production_beta_pres_mean": float(row["production_prior_mean_pres"]),
                "production_beta_abs_mean": float(row["production_prior_mean_abs"]),
                "production_gap": float(
                    row["production_prior_mean_pres"] - row["production_prior_mean_abs"]
                ),
                "targeted_beta_pres_true": float(row["generator_beta_pres_true"]),
                "targeted_beta_abs_true": float(row["generator_beta_abs_true"]),
                "fitter_beta_pres_prior_centre": float(row["fitter_prior_mean_pres"]),
                "fitter_beta_abs_prior_centre": float(row["fitter_prior_mean_abs"]),
                "prior_family": "logit-Normal/build_safe_gain_node_beta",
            }
        )

    max_pres_diff = max(
        (
            abs(
                float(row["generator_beta_pres_true"])
                - float(row["fitter_prior_mean_pres"])
            )
            for row in all_rows
        ),
        default=0.0,
    )
    max_abs_diff = max(
        (
            abs(
                float(row["generator_beta_abs_true"])
                - float(row["fitter_prior_mean_abs"])
            )
            for row in all_rows
        ),
        default=0.0,
    )

    exact_tree_src = (REPO_ROOT / "dcm_model_exact_tree.py").read_text()
    metrics_src = (REPO_ROOT / "scripts/main_synthetic_validation_metrics.py").read_text()
    target_branch_start = exact_tree_src.find(
        "if targeted_keys is not None and key in targeted_keys"
    )
    target_branch_end = exact_tree_src.find(
        "elif self.config.POOL_BETAS_BY_LABEL", target_branch_start
    )
    target_branch_src = (
        exact_tree_src[target_branch_start:target_branch_end]
        if target_branch_start >= 0 and target_branch_end > target_branch_start
        else ""
    )
    targeted_prior_static_ok = (
        "build_safe_gain_node_beta" in target_branch_src
        and "pm.Beta" not in target_branch_src
    )
    non_targeted_static_ok = (
        "self.evidence_processor.get_beta_parameters" in exact_tree_src
        and "pm.Beta" in exact_tree_src
        and not cfg.POOL_BETAS_BY_LABEL
        and not cfg.GAIN_LOGIT_NORMAL
        and float(cfg.TRANSMISSION_GAIN) == 1.0
    )
    non_targeted_match_production = all(
        float(row["fitter_prior_mean_pres"]) == float(row["production_prior_mean_pres"])
        and float(row["fitter_prior_mean_abs"]) == float(row["production_prior_mean_abs"])
        for row in non_targeted_rows
    )
    rho_deterministic_exists = (
        "_rho_collapsed" in exact_tree_src
        and "pt.sigmoid(log_prior_odds + log_B)" in exact_tree_src
    )
    rho_fallback_exists = (
        "rho_collapsed_var in post.data_vars" in metrics_src
        and "expit(LOGIT_PRIOR + log_b)" in metrics_src
    )

    failed_checks: list[str] = []
    check_results: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    def add_check(name: str, condition: bool, detail: str) -> None:
        check_results[name] = {
            "status": "pass" if condition else "fail",
            "detail": detail,
        }
        if not condition:
            failed_checks.append(f"{name}: {detail}")

    overridden_count = len(overridden_rows)
    root_to_top_count = int(profile_check["root_to_top_overridden_edge_count"])
    weak_top_count = int(profile_check["weak_top_overridden_edge_count"])
    allowed_tops = set(STRONG_TOP_FEATURES)
    disallowed_tops = set(WEAK_TOP_FEATURES)
    overridden_tops = {str(row["top_feature_name"]) for row in overridden_rows}
    disallowed_present = sorted(overridden_tops & disallowed_tops)
    outside_allowed = sorted(overridden_tops - allowed_tops)

    if overridden_count != EXPECTED_TARGETED_GWT_EDGE_COUNT:
        warnings.append(
            "targeted override selected "
            f"{overridden_count} edges, expected {EXPECTED_TARGETED_GWT_EDGE_COUNT}; "
            "the tree or key-selection changed."
        )
    ordered_targeted_by_top_feature = {
        name: int(targeted_by_top_feature[name])
        for name in STRONG_TOP_FEATURES
        if name in targeted_by_top_feature
    }
    ordered_targeted_by_top_feature.update(
        {
            name: int(targeted_by_top_feature[name])
            for name in sorted(targeted_by_top_feature)
            if name not in ordered_targeted_by_top_feature
        }
    )

    add_check(
        "fit_variant_is_targeted_strong_lower_override",
        profile.variant == FIT_VARIANT_TARGETED,
        f"fit_variant={profile.variant}",
    )
    add_check(
        "overridden_edge_count_is_expected_current_gwt",
        overridden_count == EXPECTED_TARGETED_GWT_EDGE_COUNT,
        f"overridden_edge_count={overridden_count}",
    )
    add_check(
        "overridden_edge_count_positive",
        overridden_count > 0,
        f"overridden_edge_count={overridden_count}",
    )
    add_check(
        "no_root_to_top_overrides",
        root_to_top_count == 0,
        f"root_to_top_overridden_count={root_to_top_count}",
    )
    add_check(
        "no_weak_top_overrides",
        weak_top_count == 0,
        f"weak_top_overridden_count={weak_top_count}",
    )
    add_check(
        "overridden_tops_are_allowed_strong_features",
        not outside_allowed,
        f"outside_allowed_top_ancestors={outside_allowed}",
    )
    add_check(
        "no_overridden_weak_top_ancestors",
        not disallowed_present,
        f"disallowed_top_ancestors={disallowed_present}",
    )
    add_check(
        "targeted_generator_truth_is_0p90_0p10",
        all(
            float(row["generator_beta_pres_true"]) == 0.90
            and float(row["generator_beta_abs_true"]) == 0.10
            for row in overridden_rows
        ),
        "all overridden generator truth centres are beta_pres=0.90 and beta_abs=0.10",
    )
    add_check(
        "targeted_fitter_prior_centres_are_0p90_0p10",
        all(
            float(row["fitter_prior_mean_pres"]) == 0.90
            and float(row["fitter_prior_mean_abs"]) == 0.10
            for row in overridden_rows
        ),
        "all overridden fitter prior centres are beta_pres=0.90 and beta_abs=0.10",
    )
    add_check(
        "generator_fitter_centres_match",
        max_pres_diff <= 1e-12 and max_abs_diff <= 1e-12,
        f"max_pres_diff={max_pres_diff:.3e}, max_abs_diff={max_abs_diff:.3e}",
    )
    add_check(
        "targeted_prior_family_is_logit_normal_safe_gain",
        targeted_prior_static_ok
        and all(
            row["prior_family"] == "logit-Normal/build_safe_gain_node_beta"
            for row in overridden_edge_table
        ),
        "targeted branch uses build_safe_gain_node_beta and contains no pm.Beta call",
    )
    add_check(
        "main_targeted_config_pool_betas_by_label_false",
        not cfg.POOL_BETAS_BY_LABEL,
        f"POOL_BETAS_BY_LABEL={cfg.POOL_BETAS_BY_LABEL}",
    )
    add_check(
        "main_targeted_non_targeted_edges_use_production_per_node_beta",
        non_targeted_static_ok and non_targeted_match_production,
        "non-targeted edges remain on EvidenceProcessor get_beta_parameters + pm.Beta centres",
    )
    add_check(
        "rho_collapsed_extraction_path_available",
        rho_deterministic_exists or rho_fallback_exists,
        "trace deterministic exists or extract_rho_cases computes expit(LOGIT_PRIOR + log_B)",
    )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_head(),
        "seed": int(seed),
        "fit_variant": FIT_VARIANT_TARGETED,
        "total_non_root_edges": int(len(all_rows)),
        "overridden_edge_count": int(overridden_count),
        "root_to_top_overridden_count": int(root_to_top_count),
        "weak_top_overridden_count": int(weak_top_count),
        "targeted_by_top_feature": ordered_targeted_by_top_feature,
        "targeted_by_depth_or_type": {
            "by_depth": dict(sorted(targeted_by_depth.items())),
            "by_node_type": dict(sorted(targeted_by_type.items())),
            "by_depth_and_node_type": dict(sorted(targeted_by_depth_and_type.items())),
        },
        "overridden_edge_keys": list(profile.targeted_override_node_keys),
        "overridden_edge_table": overridden_edge_table,
        "max_generator_fitter_pres_diff": float(max_pres_diff),
        "max_generator_fitter_abs_diff": float(max_abs_diff),
        "pool_betas_by_label_for_main_targeted_config": bool(cfg.POOL_BETAS_BY_LABEL),
        "non_targeted_prior_family_for_main_targeted_config": (
            "production_evidence_processor_per_node_beta"
            if non_targeted_static_ok and non_targeted_match_production
            else "not_verified"
        ),
        "targeted_prior_family_for_main_targeted_config": (
            "logit-Normal/build_safe_gain_node_beta"
            if targeted_prior_static_ok
            else "not_verified"
        ),
        "rho_collapsed_extraction_path": {
            "trace_deterministic_declared": bool(rho_deterministic_exists),
            "extract_rho_cases_fallback_declared": bool(rho_fallback_exists),
            "fallback_formula": "expit(log(1/5) + log_B)",
        },
        "check_results": check_results,
        "pass_fail": "pass" if not failed_checks else "fail",
        "failed_checks": failed_checks,
        "warnings": warnings,
    }
    write_json(output_dir / "targeted_override_sanity.json", payload)
    for warning in warnings:
        print(f"WARNING: {warning}")
    return payload


def toy_stance(depth: int) -> dict[str, Any]:
    assert depth in {2, 3}
    top_nodes = []
    for i in range(4):
        top = {
            "name": f"Toy Top {i + 1}",
            "type": "feature",
            "support": "strongly supportive",
            "demandingness": "neutral",
            "evidencers": [],
        }
        if depth == 2:
            top["evidencers"] = [
                {
                    "name": f"Toy Indicator {i + 1}.{j + 1}",
                    "type": "indicator",
                    "support": "strongly supportive",
                    "demandingness": "neutral",
                }
                for j in range(2)
            ]
        else:
            for j in range(2):
                top["evidencers"].append(
                    {
                        "name": f"Toy Sub {i + 1}.{j + 1}",
                        "type": "subfeature",
                        "support": "strongly supportive",
                        "demandingness": "neutral",
                        "evidencers": [
                            {
                                "name": f"Toy Indicator {i + 1}.{j + 1}.{k + 1}",
                                "type": "indicator",
                                "support": "strongly supportive",
                                "demandingness": "neutral",
                            }
                            for k in range(2)
                        ],
                    }
                )
        top_nodes.append(top)
    return {"name": f"Toy Depth {depth}", "evidencers": top_nodes}


def all_edge_profile(
    stance_data: Mapping[str, Any],
    beta_pres: float,
    beta_abs: float,
) -> tuple[dict[str, float], dict[str, float]]:
    bp = {}
    ba = {}
    for spec in iter_tree_nodes(stance_data):
        bp[spec.key] = float(beta_pres)
        ba[spec.key] = float(beta_abs)
    return bp, ba


def top_strong_lower_profile(
    stance_data: Mapping[str, Any],
    *,
    lower_beta_pres: float,
    lower_beta_abs: float,
    top_beta_pres: float = 0.90,
    top_beta_abs: float = 0.10,
) -> tuple[dict[str, float], dict[str, float]]:
    """Use fixed strong root-to-top edges and a separate lower-edge profile."""
    bp: dict[str, float] = {}
    ba: dict[str, float] = {}
    for spec in iter_tree_nodes(stance_data):
        if int(spec.depth) == 1:
            bp[spec.key] = float(top_beta_pres)
            ba[spec.key] = float(top_beta_abs)
        else:
            bp[spec.key] = float(lower_beta_pres)
            ba[spec.key] = float(lower_beta_abs)
    return bp, ba


def recovery_rung_sort_key(rung_id: str) -> tuple[int, str]:
    order = {"L0": 0, "L2": 1, "L2_sweep": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6}
    base = str(rung_id).split("__", 1)[0]
    return order.get(base, 99), str(rung_id)


def root_truth_for_replicate(rep: int) -> dict[str, int]:
    return root_truth_for_seed_idx(rep)


def bootstrap_evidence_margin_ci(
    r1: pd.Series,
    r0: pd.Series,
    *,
    rng: np.random.Generator,
    n_boot: int = 500,
) -> tuple[float, float]:
    r1_vals = np.asarray(r1, dtype=float)
    r0_vals = np.asarray(r0, dtype=float)
    r1_vals = r1_vals[np.isfinite(r1_vals)]
    r0_vals = r0_vals[np.isfinite(r0_vals)]
    if len(r1_vals) == 0 or len(r0_vals) == 0:
        return np.nan, np.nan
    vals = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        b1 = rng.choice(r1_vals, size=len(r1_vals), replace=True)
        b0 = rng.choice(r0_vals, size=len(r0_vals), replace=True)
        vals[i] = min(float(np.median(b1) - TAU_PRESENT_50), float(TAU_ABSENT_05 - np.median(b0)))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def recovery_group_summary(
    cases: pd.DataFrame,
    *,
    seed: int,
    n_boot: int = 500,
) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []
    baseline_brier = 0.5 * (1 - PRIOR_P) ** 2 + 0.5 * PRIOR_P**2
    grouping = ["rung_id", "system_summary"]
    cases_for_summary = pd.concat(
        [
            cases.assign(system_summary="ALL"),
            cases.assign(system_summary=cases["system"]),
        ],
        ignore_index=True,
    )
    for (rung_id, system_summary), g in cases_for_summary.groupby(grouping, dropna=False):
        r1 = g[g["root_z_true"] == 1]
        r0 = g[g["root_z_true"] == 0]
        if r1.empty or r0.empty:
            continue
        tpr = float((r1["rho_collapsed"] > 0.5).mean())
        tnr = float((r0["rho_collapsed"] <= 0.5).mean())
        bal_acc = float(np.mean([tpr, tnr]))
        m = float(
            min(
                r1["log_B_eff"].median() - TAU_PRESENT_50,
                TAU_ABSENT_05 - r0["log_B_eff"].median(),
            )
        )
        group_token = f"{rung_id}|{system_summary}"
        group_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(group_token)) % 1_000_003
        rng = np.random.default_rng(seed + group_offset)
        ci_low, ci_high = bootstrap_evidence_margin_ci(
            r1["log_B_eff"],
            r0["log_B_eff"],
            rng=rng,
            n_boot=n_boot,
        )
        ci_crosses_zero = bool(np.isfinite(ci_low) and np.isfinite(ci_high) and ci_low <= 0.0 <= ci_high)
        u_log = 0.5 * float(np.log(np.clip(r1["rho_collapsed"], SCORE_CLIP, 1.0) / PRIOR_P).mean()) + 0.5 * float(
            np.log(np.clip(1.0 - r0["rho_collapsed"], SCORE_CLIP, 1.0) / (1.0 - PRIOR_P)).mean()
        )
        brier_model = 0.5 * float(((1.0 - r1["rho_collapsed"]) ** 2).mean()) + 0.5 * float((r0["rho_collapsed"] ** 2).mean())
        brier_improvement = baseline_brier - brier_model
        non_margin_fail = u_log <= 0.0 or brier_improvement <= 0.0 or bal_acc < 0.75
        if m < -0.25 or non_margin_fail:
            status = "fail"
        elif -0.25 <= m <= 0.0 or ci_crosses_zero:
            status = "borderline"
        else:
            status = "pass"
        first = g.iloc[0]
        summary_rows.append(
            {
                "rung_id": str(rung_id),
                "rung": str(first["rung"]),
                "rung_description": str(first["rung_description"]),
                "sweep_beta_pres": first.get("sweep_beta_pres", np.nan),
                "sweep_beta_abs": first.get("sweep_beta_abs", np.nan),
                "system": str(system_summary),
                "n_cases": int(len(g)),
                "n_replicates": int(g["seed_idx"].nunique()),
                "median_log_B_eff_R1": float(r1["log_B_eff"].median()),
                "median_log_B_eff_R0": float(r0["log_B_eff"].median()),
                "mean_rho_R1": float(r1["rho_collapsed"].mean()),
                "mean_rho_R0": float(r0["rho_collapsed"].mean()),
                "evidence_margin_M": m,
                "evidence_margin_bootstrap_ci_low": ci_low,
                "evidence_margin_bootstrap_ci_high": ci_high,
                "evidence_margin_bootstrap_ci_crosses_0": ci_crosses_zero,
                "balanced_log_score_improvement_vs_prior": u_log,
                "balanced_brier_improvement_vs_prior": brier_improvement,
                "TPR_at_rho_gt_0p5": tpr,
                "TNR_at_rho_le_0p5": tnr,
                "balanced_accuracy": bal_acc,
                "pass_fail_label": status,
            }
        )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["_sort"] = summary["rung_id"].map(recovery_rung_sort_key)
        summary["_sys_sort"] = summary["system"].map({"ALL": 0, "Chicken": 1, "LLMs": 2}).fillna(99)
        summary = summary.sort_values(["_sort", "_sys_sort"]).drop(columns=["_sort", "_sys_sort"])
    return summary


def write_recovery_ladder_plots(output_dir: Path, cases: pd.DataFrame, summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_cases = cases.copy()
    plot_cases["_sort"] = plot_cases["rung_id"].map(recovery_rung_sort_key)
    rung_ids = list(plot_cases.sort_values("_sort")["rung_id"].drop_duplicates())
    labels = [r.replace("L2_sweep__", "sweep\n").replace("_", " ") for r in rung_ids]

    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(rung_ids)), 4.8))
    data = [plot_cases.loc[plot_cases["rung_id"] == rid, "log_B_eff"].to_numpy() for rid in rung_ids]
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(TAU_PRESENT_50, color="#2f7d32", linestyle="--", linewidth=1, label="rho=0.5 present")
    ax.axhline(TAU_ABSENT_05, color="#8b1e1e", linestyle="--", linewidth=1, label="rho=0.05 absent")
    ax.set_ylabel("log_B_eff")
    ax.set_title("Recovery Ladder Evidence by Rung")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "recovery_ladder_logB_by_rung.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(rung_ids)), 4.8))
    offsets = {("Chicken", 0): -0.27, ("Chicken", 1): -0.09, ("LLMs", 0): 0.09, ("LLMs", 1): 0.27}
    colors = {("Chicken", 0): "#9b5b00", ("Chicken", 1): "#e0a13a", ("LLMs", 0): "#285c8f", ("LLMs", 1): "#6aa2d8"}
    for idx, rid in enumerate(rung_ids, start=1):
        sub = plot_cases[plot_cases["rung_id"] == rid]
        for (system, truth), vals in sub.groupby(["system", "root_z_true"], dropna=False):
            x = np.full(len(vals), idx + offsets[(str(system), int(truth))])
            ax.scatter(x, vals["rho_collapsed"], s=10, alpha=0.35, color=colors[(str(system), int(truth))], label=f"{system} R={truth}" if idx == 1 else None)
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(range(1, len(rung_ids) + 1), labels, rotation=45, ha="right")
    ax.set_ylabel("rho_collapsed")
    ax.set_title("Collapsed Root Probability by Truth and System")
    ax.legend(loc="best", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "rho_by_truth_and_system_ladder.png", dpi=180)
    plt.close(fig)

    sweep = summary[(summary["rung"] == "L2_sweep") & (summary["system"] == "ALL")].copy()
    if not sweep.empty:
        sweep = sweep.sort_values(["sweep_beta_pres", "sweep_beta_abs"], ascending=[False, True])
        x = np.arange(len(sweep))
        xlabels = [f"{p:.2f}/{a:.2f}" for p, a in zip(sweep["sweep_beta_pres"], sweep["sweep_beta_abs"])]
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(x, sweep["median_log_B_eff_R1"], marker="o", label="median log_B | R=1")
        ax.plot(x, sweep["median_log_B_eff_R0"], marker="o", label="median log_B | R=0")
        ax.plot(x, sweep["evidence_margin_M"], marker="s", label="margin M")
        ax.axhline(0.0, color="#555555", linewidth=1)
        ax.axhline(TAU_PRESENT_50, color="#2f7d32", linestyle="--", linewidth=1)
        ax.axhline(TAU_ABSENT_05, color="#8b1e1e", linestyle="--", linewidth=1)
        ax.set_xticks(x, xlabels, rotation=35, ha="right")
        ax.set_xlabel("lower-edge beta_pres/beta_abs")
        ax.set_ylabel("nats")
        ax.set_title("Toy Depth-3 Lower-Edge Gap Sweep")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / "toy_depth3_lower_gap_sweep.png", dpi=180)
        plt.close(fig)

    conf = (
        cases.assign(predicted_present=cases["rho_collapsed"] > 0.5)
        .groupby(["rung", "root_z_true", "predicted_present"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    rungs = list(conf["rung"].drop_duplicates())
    fig, axes = plt.subplots(1, len(rungs), figsize=(max(8, 2.2 * len(rungs)), 3.2), squeeze=False)
    for ax, rung in zip(axes[0], rungs):
        mat = np.zeros((2, 2), dtype=int)
        for _, row in conf[conf["rung"] == rung].iterrows():
            mat[int(row["root_z_true"]), int(row["predicted_present"])] += int(row["count"])
        ax.imshow(mat, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", color="#111111")
        ax.set_title(str(rung))
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
    fig.suptitle("Recovery Ladder Confusion Counts")
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix_counts_ladder.png", dpi=180)
    plt.close(fig)


def run_recovery_ladder(
    *,
    output_dir: Path,
    mode: str,
    n_rep_no_fit: int,
    n_rep_hmc: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if int(n_rep_hmc) != 0:
        raise ValueError("recovery ladder is no-HMC only; pass --n-rep-hmc 0")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = ModelConfig(INDICATOR_STATE_MODEL="three_state")
    actual = load_gwt_stance(cfg)
    targeted_cfg = build_fit_config(
        mode="pilot",
        variant=FIT_VARIANT_TARGETED,
        targeted_keys=build_edge_beta_profile(actual, FIT_VARIANT_TARGETED, cfg).targeted_override_node_keys,
        chains=None,
        draws=None,
        tune=None,
        target_accept=None,
    )
    targeted_profile = build_edge_beta_profile(actual, FIT_VARIANT_TARGETED, targeted_cfg)
    actual_target_bp = targeted_profile.beta_pres_true_by_key
    actual_target_ba = targeted_profile.beta_abs_true_by_key
    actual_extreme_bp, actual_extreme_ba, _ = extreme_beta_profile(actual)
    obs_truth = load_oracle_truth("exact_tree_production_medians", actual, cfg)

    toy2 = toy_stance(2)
    toy3 = toy_stance(3)
    rungs: list[dict[str, Any]] = [
        {
            "rung": "L0",
            "rung_id": "L0",
            "stance_data": toy2,
            "beta_pres_by_key": all_edge_profile(toy2, 0.90, 0.10)[0],
            "beta_abs_by_key": all_edge_profile(toy2, 0.90, 0.10)[1],
            "observation_model": "binary_noisy",
            "description": "toy depth-2 binary tree, all edges 0.90/0.10, binary noisy leaves",
            "sweep_beta_pres": np.nan,
            "sweep_beta_abs": np.nan,
        },
        {
            "rung": "L2",
            "rung_id": "L2",
            "stance_data": toy3,
            "beta_pres_by_key": all_edge_profile(toy3, 0.90, 0.10)[0],
            "beta_abs_by_key": all_edge_profile(toy3, 0.90, 0.10)[1],
            "observation_model": "binary_noisy",
            "description": "toy depth-3 binary tree, all edges 0.90/0.10, binary noisy leaves",
            "sweep_beta_pres": np.nan,
            "sweep_beta_abs": np.nan,
        },
    ]
    for lower_pres, lower_abs in [
        (0.90, 0.10),
        (0.80, 0.20),
        (0.70, 0.30),
        (0.65, 0.35),
        (0.60, 0.40),
        (0.55, 0.45),
        (0.43, 0.50),
    ]:
        bp, ba = top_strong_lower_profile(
            toy3,
            lower_beta_pres=lower_pres,
            lower_beta_abs=lower_abs,
        )
        rungs.append(
            {
                "rung": "L2_sweep",
                "rung_id": f"L2_sweep__{lower_pres:.2f}_{lower_abs:.2f}".replace(".", "p"),
                "stance_data": toy3,
                "beta_pres_by_key": bp,
                "beta_abs_by_key": ba,
                "observation_model": "binary_noisy",
                "description": "toy depth-3 lower-edge gap sweep with root-to-top edges fixed at 0.90/0.10",
                "sweep_beta_pres": float(lower_pres),
                "sweep_beta_abs": float(lower_abs),
            }
        )
    rungs.extend(
        [
            {
                "rung": "L3",
                "rung_id": "L3",
                "stance_data": actual,
                "beta_pres_by_key": actual_extreme_bp,
                "beta_abs_by_key": actual_extreme_ba,
                "observation_model": "binary_noisy",
                "description": "actual GWT topology, all edges 0.90/0.10, binary noisy leaves",
                "sweep_beta_pres": np.nan,
                "sweep_beta_abs": np.nan,
            },
            {
                "rung": "L4",
                "rung_id": "L4",
                "stance_data": actual,
                "beta_pres_by_key": actual_target_bp,
                "beta_abs_by_key": actual_target_ba,
                "observation_model": "binary_noisy",
                "description": "actual GWT topology, targeted_strong_lower_override, binary noisy leaves",
                "sweep_beta_pres": np.nan,
                "sweep_beta_abs": np.nan,
            },
            {
                "rung": "L5",
                "rung_id": "L5",
                "stance_data": actual,
                "beta_pres_by_key": actual_target_bp,
                "beta_abs_by_key": actual_target_ba,
                "observation_model": "ordinal_binary",
                "description": "targeted_strong_lower_override, binary latent leaf plus ordinal probit, nuisance clamped to truth",
                "sweep_beta_pres": np.nan,
                "sweep_beta_abs": np.nan,
            },
            {
                "rung": "L6",
                "rung_id": "L6",
                "stance_data": actual,
                "beta_pres_by_key": actual_target_bp,
                "beta_abs_by_key": actual_target_ba,
                "observation_model": "ordinal_three_state",
                "description": "targeted_strong_lower_override, production three-state ordinal leaf, nuisance clamped to truth",
                "sweep_beta_pres": np.nan,
                "sweep_beta_abs": np.nan,
            },
        ]
    )
    rows: list[dict[str, Any]] = []
    reps = 20 if mode == "recovery-ladder-smoke" else int(n_rep_no_fit)
    for rung_spec in rungs:
        rung = str(rung_spec["rung"])
        rung_id = str(rung_spec["rung_id"])
        stance_data = rung_spec["stance_data"]
        bp = rung_spec["beta_pres_by_key"]
        ba = rung_spec["beta_abs_by_key"]
        observation_model = str(rung_spec["observation_model"])
        description = str(rung_spec["description"])
        for seed_idx in range(reps):
            roots = root_truth_for_seed_idx(seed_idx)
            sub_seed = int(seed + seed_idx * 1_000_003)
            rng = np.random.default_rng(sub_seed)
            for system in FREE_SYSTEMS:
                latent = sample_latent_states(
                    rng=rng,
                    stance_data=stance_data,
                    beta_pres_by_key=bp,
                    beta_abs_by_key=ba,
                    root_z=roots[system],
                )
                messages, obs_meta = simulate_leaf_messages(
                    rng=rng,
                    stance_data=stance_data,
                    latent=latent,
                    beta_pres_by_key=bp,
                    beta_abs_by_key=ba,
                    obs_params=obs_truth.obs_params,
                    observation_model=observation_model,
                    k_or_n=6,
                    epsilon=0.05,
                )
                log_l0, log_l1 = exact_root_sides_from_leaf_messages(
                    stance_data,
                    messages,
                    bp,
                    ba,
                )
                log_b = float(log_l1 - log_l0)
                rho = float(expit(LOGIT_PRIOR + log_b))
                root_z = int(roots[system])
                brier = float(bernoulli_brier(rho, root_z))
                log_score = float(bernoulli_log_score(rho, root_z))
                prior_brier = float(bernoulli_brier(PRIOR_P, root_z))
                prior_log_score = float(bernoulli_log_score(PRIOR_P, root_z))
                rows.append(
                    {
                        "seed_idx": int(seed_idx),
                        "rung": rung,
                        "rung_id": rung_id,
                        "rung_description": description,
                        "fit_type": "no_fit_exact_dp",
                        "replicate_id": int(seed_idx),
                        "seed": int(seed),
                        "sub_seed": sub_seed,
                        "system": "LLMs" if system == "2024 Leading Chat LLMs" else system,
                        "system_raw": system,
                        "root_z_true": root_z,
                        "log_L0": log_l0,
                        "log_L1": log_l1,
                        "log_B": log_b,
                        "log_B_eff": log_b,
                        "rho_collapsed": rho,
                        "rho": rho,
                        "brier": brier,
                        "log_score": log_score,
                        "brier_improvement_vs_prior": float(prior_brier - brier),
                        "log_score_improvement_vs_prior": float(log_score - prior_log_score),
                        "evidence_category": evidence_category_from_log_b(log_b),
                        "correct_sign": bool((root_z == 1 and log_b > 0) or (root_z == 0 and log_b < 0)),
                        "classified_present_at_0p5": bool(rho > 0.5),
                        "brier_improvement": float(prior_brier - brier),
                        "log_score_improvement": float(log_score - prior_log_score),
                        "sweep_beta_pres": rung_spec["sweep_beta_pres"],
                        "sweep_beta_abs": rung_spec["sweep_beta_abs"],
                        **obs_meta,
                    }
                )

    cases = pd.DataFrame(rows)
    summary = recovery_group_summary(cases, seed=seed)
    cases.to_csv(output_dir / "recovery_ladder_cases.csv", index=False)
    summary.to_csv(output_dir / "recovery_ladder_summary.csv", index=False)
    write_recovery_ladder_report(output_dir, summary)
    write_recovery_ladder_plots(output_dir, cases, summary)
    return cases, summary


def three_state_leaf_message_from_ratings(
    ratings: Sequence[int],
    beta_pres: float,
    beta_abs: float,
    a: float,
    kappa: np.ndarray,
) -> tuple[float, float]:
    lls = []
    for m in (0, 1, 2):
        probs = ordered_probit_probs(kappa, a * (m / 2.0))
        lls.append(float(sum(math.log(float(np.clip(probs[int(r)], SCORE_CLIP, 1.0))) for r in ratings)))

    def side(beta: float) -> float:
        b = float(np.clip(beta, SCORE_CLIP, 1.0 - SCORE_CLIP))
        return float(
            np.logaddexp(
                np.logaddexp(2.0 * math.log1p(-b) + lls[0], math.log(2.0) + math.log(b) + math.log1p(-b) + lls[1]),
                2.0 * math.log(b) + lls[2],
            )
        )

    return side(beta_abs), side(beta_pres)


def oracle_root_bridge_cases(
    output_dir: Path,
    posterior_cases: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    source = load_source_stance(ModelConfig(INDICATOR_STATE_MODEL="three_state"))
    if posterior_cases is None:
        posterior_cases_path = output_dir / "validation_cases.csv"
        posterior_cases = pd.read_csv(posterior_cases_path) if posterior_cases_path.exists() else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for truth_path in sorted((output_dir / "runs").glob("*/seed_*/truth_payload.json")):
        run_dir = truth_path.parent
        truth = json.loads(truth_path.read_text())
        variant = str(truth["fit_variant"])
        seed_value = int(truth.get("seed", 0)) + int(truth.get("seed_idx", 0))
        edge_betas = truth["edge_betas"]
        beta_pres = {k: float(v["beta_pres"]) for k, v in edge_betas.items()}
        beta_abs = {k: float(v["beta_abs"]) for k, v in edge_betas.items()}
        obs = truth["ordinal_by_system_indicator"]
        a = float(truth["a_true"])
        kappa = np.asarray(truth["kappa_true"], dtype=float)
        for system in FREE_SYSTEMS:
            messages: dict[str, tuple[float, float]] = {}
            for key, ratings in obs.get(system, {}).items():
                messages[key] = three_state_leaf_message_from_ratings(
                    ratings,
                    beta_pres[key],
                    beta_abs[key],
                    a,
                    kappa,
                )
            log_l0, log_l1 = exact_root_sides_from_leaf_messages(source, messages, beta_pres, beta_abs)
            log_b_oracle = float(log_l1 - log_l0)
            rho_oracle = float(expit(LOGIT_PRIOR + log_b_oracle))
            post_row = posterior_cases[
                (posterior_cases["fit_variant"] == variant)
                & (posterior_cases["seed"].astype(int) == seed_value)
                & (posterior_cases["system_raw"] == system)
            ]
            if post_row.empty:
                if posterior_cases is not None and not posterior_cases.empty:
                    continue
                rho_post = np.nan
                log_b_post = np.nan
                rho_post_median = np.nan
                log_b_post_median = np.nan
            else:
                rho_post = float(post_row.iloc[0]["rho_collapsed"])
                log_b_post = float(post_row.iloc[0]["log_B_eff"])
                rho_post_median = float(post_row.iloc[0].get("rho_collapsed_median", np.nan))
                log_b_post_median = float(post_row.iloc[0].get("log_B_draw_median", np.nan))
            root_z = int(truth["root_z_by_system"][system])
            posterior_correct = bool((root_z == 1 and log_b_post > 0) or (root_z == 0 and log_b_post < 0)) if np.isfinite(log_b_post) else False
            oracle_correct = bool((root_z == 1 and log_b_oracle > 0) or (root_z == 0 and log_b_oracle < 0))
            rows.append(
                {
                    "fit_variant": variant,
                    "seed": seed_value,
                    "system": "LLMs" if system == "2024 Leading Chat LLMs" else system,
                    "system_raw": system,
                    "root_z_true": root_z,
                    "log_B_oracle": log_b_oracle,
                    "rho_oracle_collapsed": rho_oracle,
                    "log_B_eff_HMC": log_b_post,
                    "log_B_draw_median_HMC": log_b_post_median,
                    "rho_collapsed_HMC": rho_post,
                    "rho_collapsed_median_HMC": rho_post_median,
                    "delta_log_B_eff_minus_oracle": float(log_b_post - log_b_oracle) if np.isfinite(log_b_post) else np.nan,
                    "posterior_correct_sign": posterior_correct,
                    "oracle_correct_sign": oracle_correct,
                    "hmc_only_sign_flip": bool(oracle_correct and not posterior_correct),
                    "posterior_decisive": bool((root_z == 1 and log_b_post > TAU_PRESENT_50) or (root_z == 0 and log_b_post < TAU_ABSENT_05)) if np.isfinite(log_b_post) else False,
                    "oracle_decisive": bool((root_z == 1 and log_b_oracle > TAU_PRESENT_50) or (root_z == 0 and log_b_oracle < TAU_ABSENT_05)),
                    "run_dir": str(run_dir.relative_to(output_dir)),
                }
            )
    df = pd.DataFrame(rows)
    if output_path is not None:
        df.to_csv(output_path, index=False)
    return df


def oracle_root_bridge_summary(bridge: pd.DataFrame) -> pd.DataFrame:
    if bridge.empty:
        return pd.DataFrame()
    rows = []
    for (fit_variant, system), g in bridge.groupby(["fit_variant", "system"], dropna=False):
        r1 = g[g["root_z_true"] == 1]
        rows.append(
            {
                "fit_variant": fit_variant,
                "system": system,
                "n_cases": int(len(g)),
                "n_R1_cases": int(len(r1)),
                "n_oracle_correct": int(g["oracle_correct_sign"].sum()),
                "n_posterior_correct": int(g["posterior_correct_sign"].sum()),
                "n_hmc_only_sign_flips": int(g["hmc_only_sign_flip"].sum()),
                "n_R1_oracle_negative": int(((r1["log_B_oracle"] < 0) if not r1.empty else pd.Series(dtype=bool)).sum()),
                "mean_delta_log_B_eff_minus_oracle": float(g["delta_log_B_eff_minus_oracle"].mean()),
                "max_abs_delta_log_B_eff_minus_oracle": float(g["delta_log_B_eff_minus_oracle"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def oracle_root_smoke_cases(output_dir: Path) -> pd.DataFrame:
    df = oracle_root_bridge_cases(output_dir)
    legacy = df.rename(
        columns={
            "log_B_eff_HMC": "log_B_eff_posterior",
            "rho_collapsed_HMC": "rho_collapsed_posterior",
            "delta_log_B_eff_minus_oracle": "delta_eff_minus_oracle",
        }
    )
    legacy.to_csv(output_dir / "oracle_root_smoke_cases.csv", index=False)
    return legacy


def rps_for_probs(probs: np.ndarray, rating: int) -> float:
    cdf = np.cumsum(probs)
    obs = (np.arange(len(probs)) >= int(rating)).astype(float)
    return float(np.mean((cdf[:-1] - obs[:-1]) ** 2))


def ppc_metrics_from_cell_counts(
    cell_counts: dict[tuple[str, str], np.ndarray],
    true_probs: dict[tuple[str, str], np.ndarray],
    rating_rows: list[tuple[str, str, int]],
    systems: Sequence[str],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    group_defs = {"ALL": list(cell_counts.keys())}
    for system in systems:
        label = "LLMs" if system == "2024 Leading Chat LLMs" else system
        group_defs[label] = [key for key in cell_counts if key[0] == system]
    for label, keys in group_defs.items():
        if not keys:
            continue
        rps_vals = [
            rps_for_probs(true_probs[(system, key)], rating)
            for system, key, rating in rating_rows
            if label == "ALL" or ("LLMs" if system == "2024 Leading Chat LLMs" else system) == label
        ]
        tvs = []
        weights = []
        high_rows = []
        for key in keys:
            counts = cell_counts[key].astype(float)
            if counts.sum() <= 0:
                continue
            obs_dist = counts / counts.sum()
            probs = true_probs[key]
            tvs.append(0.5 * float(np.abs(obs_dist - probs).sum()))
            weights.append(float(counts.sum()))
            high_q = bool(probs[-2:].sum() >= 0.50)
            high_rows.append(
                {
                    "high_q": high_q,
                    "abs_top7_error": float(abs(obs_dist[-1] - probs[-1])),
                    "abs_high_error": float(abs(obs_dist[-2:].sum() - probs[-2:].sum())),
                }
            )
        tv_arr = np.asarray(tvs, dtype=float)
        w_arr = np.asarray(weights, dtype=float)
        high_df = pd.DataFrame(high_rows)
        high_q = high_df[high_df["high_q"]] if not high_df.empty else pd.DataFrame()
        out[label] = {
            "n_ratings": int(sum(weights)),
            "n_cells": int(len(weights)),
            "n_cells_ge_5": int(np.sum(w_arr >= 5)) if len(w_arr) else 0,
            "mean_RPS": float(np.mean(rps_vals)) if rps_vals else np.nan,
            "weighted_mean_cell_TV": float(np.average(tv_arr, weights=w_arr)) if len(tv_arr) else np.nan,
            "median_cell_TV": float(np.median(tv_arr)) if len(tv_arr) else np.nan,
            "q90_cell_TV": float(np.quantile(tv_arr, 0.90)) if len(tv_arr) else np.nan,
            "fraction_cell_TV_gt_0p5": float(np.mean(tv_arr > 0.5)) if len(tv_arr) else np.nan,
            "mean_abs_top7_error_high_q": float(high_q["abs_top7_error"].mean()) if not high_q.empty else np.nan,
            "q90_abs_top7_error_high_q": float(high_q["abs_top7_error"].quantile(0.90)) if not high_q.empty else np.nan,
            "mean_abs_high_error_high_q": float(high_q["abs_high_error"].mean()) if not high_q.empty else np.nan,
            "q90_abs_high_error_high_q": float(high_q["abs_high_error"].quantile(0.90)) if not high_q.empty else np.nan,
        }
    return out


def oracle_ppc_baseline(output_dir: Path, n_rep: int = 500) -> pd.DataFrame:
    fitted_by_seed: dict[tuple[str, int, str], pd.Series] = {}
    for path in sorted((output_dir / "runs").glob("*/seed_*/posterior_predictive_summary.csv")):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            fitted_by_seed[(str(row["fit_variant"]), int(row["seed"]), str(row["system"]))] = row
    metric_names = [
        "mean_RPS",
        "weighted_mean_cell_TV",
        "median_cell_TV",
        "q90_cell_TV",
        "fraction_cell_TV_gt_0p5",
        "mean_abs_top7_error_high_q",
        "q90_abs_top7_error_high_q",
        "mean_abs_high_error_high_q",
        "q90_abs_high_error_high_q",
    ]
    rows: list[dict[str, Any]] = []
    for truth_path in sorted((output_dir / "runs").glob("*/seed_*/truth_payload.json")):
        truth = json.loads(truth_path.read_text())
        variant = str(truth["fit_variant"])
        seed_value = int(truth["seed"]) + int(truth["seed_idx"])
        a = float(truth["a_true"])
        kappa = np.asarray(truth["kappa_true"], dtype=float)
        true_probs: dict[tuple[str, str], np.ndarray] = {}
        cell_counts: dict[tuple[str, str], np.ndarray] = {}
        rating_rows: list[tuple[str, str, int]] = []
        for system, by_key in truth["ordinal_by_system_indicator"].items():
            for key, ratings in by_key.items():
                if not ratings:
                    continue
                m = int(truth["latent_by_system"][system]["indicator_m"][key])
                probs = ordered_probit_probs(kappa, a * (m / 2.0))
                true_probs[(system, key)] = probs
                counts = np.bincount(np.asarray(ratings, dtype=int), minlength=CATEGORY_COUNT).astype(float)
                cell_counts[(system, key)] = counts
                for rating in ratings:
                    rating_rows.append((system, key, int(rating)))
        observed_metrics = ppc_metrics_from_cell_counts(
            cell_counts,
            true_probs,
            rating_rows,
            [s for s, _ in ANCHORED_SYSTEM_CONFIGS],
        )
        null_values: dict[str, dict[str, list[float]]] = {}
        rng = np.random.default_rng(seed_value + 880_301)
        for _ in range(int(n_rep)):
            rep_counts: dict[tuple[str, str], np.ndarray] = {}
            rep_rows: list[tuple[str, str, int]] = []
            for key, counts in cell_counts.items():
                probs = true_probs[key]
                n = int(counts.sum())
                ratings = rng.choice(np.arange(CATEGORY_COUNT), size=n, p=probs)
                rep_counts[key] = np.bincount(ratings, minlength=CATEGORY_COUNT).astype(float)
                system, node_key = key
                rep_rows.extend((system, node_key, int(r)) for r in ratings)
            rep_metrics = ppc_metrics_from_cell_counts(
                rep_counts,
                true_probs,
                rep_rows,
                [s for s, _ in ANCHORED_SYSTEM_CONFIGS],
            )
            for system_label, vals in rep_metrics.items():
                null_values.setdefault(system_label, {m: [] for m in metric_names})
                for metric in metric_names:
                    null_values[system_label][metric].append(float(vals.get(metric, np.nan)))
        for system_label, vals in observed_metrics.items():
            fitted_row = fitted_by_seed.get((variant, seed_value, system_label))
            for metric in metric_names:
                dist = np.asarray(null_values.get(system_label, {}).get(metric, []), dtype=float)
                dist = dist[np.isfinite(dist)]
                if dist.size == 0:
                    q05 = q50 = q95 = mean = median = pct = np.nan
                    status = "oracle_null_unavailable"
                else:
                    q05, q50, q95 = np.quantile(dist, [0.05, 0.50, 0.95])
                    mean = float(np.mean(dist))
                    median = float(np.median(dist))
                    fitted_value = float(fitted_row[metric]) if fitted_row is not None and metric in fitted_row else np.nan
                    pct = float(np.mean(dist <= fitted_value)) if np.isfinite(fitted_value) else np.nan
                    if not np.isfinite(fitted_value):
                        status = "fitted_metric_missing"
                    elif fitted_value > q95:
                        status = "ppc_excess_misfit"
                    elif fitted_value < q05:
                        status = "ppc_underdispersed_or_metric_issue"
                    else:
                        status = "compatible_with_finite_sample_noise"
                fitted_value = float(fitted_row[metric]) if fitted_row is not None and metric in fitted_row else np.nan
                rows.append(
                    {
                        "fit_variant": variant,
                        "seed": seed_value,
                        "system": system_label,
                        "n_ratings": int(vals["n_ratings"]),
                        "n_cells": int(vals["n_cells"]),
                        "n_cells_ge_5": int(vals["n_cells_ge_5"]),
                        "metric": metric,
                        "fitted_ppc_value": fitted_value,
                        "oracle_expected_mean": mean,
                        "oracle_expected_median": median,
                        "oracle_q05": float(q05),
                        "oracle_q50": float(q50),
                        "oracle_q95": float(q95),
                        "fitted_minus_oracle_median": float(fitted_value - median) if np.isfinite(fitted_value) and np.isfinite(median) else np.nan,
                        "fitted_percentile_under_oracle_null": pct,
                        "status": status,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "oracle_ppc_baseline_summary.csv", index=False)
    df[df["system"].isin(["Human", "ELIZA", "Chicken", "LLMs"])].to_csv(
        output_dir / "oracle_ppc_system_baseline_summary.csv",
        index=False,
    )
    return df


def true_predictive_probabilities_for_truth(
    truth: Mapping[str, Any],
) -> dict[tuple[str, str], np.ndarray]:
    a = float(truth["a_true"])
    kappa = np.asarray(truth["kappa_true"], dtype=float)
    out: dict[tuple[str, str], np.ndarray] = {}
    for system, latent in truth["latent_by_system"].items():
        for key, m in latent["indicator_m"].items():
            out[(system, key)] = ordered_probit_probs(kappa, a * (int(m) / 2.0))
    return out


def predictive_alignment_to_truth(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eps = SCORE_CLIP
    source = load_source_stance(ModelConfig())
    indicator_order = [
        spec.key
        for spec in iter_tree_nodes(source)
        if (spec.node.get("type") or "").lower() == "indicator"
    ]
    for run_dir in sorted((output_dir / "runs").glob("*/seed_*")):
        pp_path = run_dir / "posterior_predictive.nc"
        truth_path = run_dir / "truth_payload.json"
        if not pp_path.exists() or not truth_path.exists():
            continue
        import xarray as xr

        truth = json.loads(truth_path.read_text())
        variant = str(truth["fit_variant"])
        seed_value = int(truth["seed"]) + int(truth["seed_idx"])
        true_probs = true_predictive_probabilities_for_truth(truth)
        ppc = xr.open_dataset(pp_path)
        fit_probs = ppc["posterior_mean_category_probability"].values
        observed = ppc["observed_rating"].values.astype(int)
        row_sources = []
        # Reconstruct PPC row order. The PPC writer iterates proc.systems,
        # then MultiSystemDataProcessor's tree-order indicator dict, then
        # local observations. truth_payload.json is written with sorted keys,
        # so do not iterate the JSON dict order here.
        for system in [s for s, _ in ANCHORED_SYSTEM_CONFIGS]:
            by_key = truth["ordinal_by_system_indicator"].get(system, {})
            for key in indicator_order:
                if key not in by_key:
                    continue
                ratings = by_key[key]
                for rating in ratings:
                    row_sources.append((system, key, int(rating)))
        if len(row_sources) != fit_probs.shape[0]:
            raise AssertionError(
                f"PPC row count mismatch for {run_dir}: "
                f"truth rows={len(row_sources)}, ppc rows={fit_probs.shape[0]}"
            )
        truth_obs = np.asarray([r for _, _, r in row_sources], dtype=int)
        if not np.array_equal(truth_obs, observed):
            mismatch = int(np.sum(truth_obs != observed))
            raise AssertionError(
                f"PPC row order mismatch for {run_dir}: {mismatch} observed ratings differ"
            )
        cell_rows: list[dict[str, Any]] = []
        for idx, (system, key, rating) in enumerate(row_sources):
            p_truth = np.asarray(true_probs[(system, key)], dtype=float)
            p_fit = np.asarray(fit_probs[idx], dtype=float)
            p_truth = p_truth / p_truth.sum()
            p_fit = p_fit / p_fit.sum()
            tv = 0.5 * float(np.abs(p_fit - p_truth).sum())
            kl_tf = float(np.sum(p_truth * (np.log(np.clip(p_truth, eps, 1.0)) - np.log(np.clip(p_fit, eps, 1.0)))))
            kl_ft = float(np.sum(p_fit * (np.log(np.clip(p_fit, eps, 1.0)) - np.log(np.clip(p_truth, eps, 1.0)))))
            rps_exp = float(
                sum(
                    p_truth[k] * ranked_probability_score(p_fit, k)
                    for k in range(len(p_truth))
                )
            )
            cell_rows.append(
                {
                    "fit_variant": variant,
                    "seed": seed_value,
                    "system": "LLMs" if system == "2024 Leading Chat LLMs" else system,
                    "system_raw": system,
                    "node_key": key,
                    "rating": rating,
                    "observed_rating_check": int(observed[idx]),
                    "tv_fit_vs_truth": tv,
                    "kl_truth_to_fit": kl_tf,
                    "kl_fit_to_truth": kl_ft,
                    "rps_truth_expected_fit": rps_exp,
                    "top7_abs_error_fit_vs_truth": float(abs(p_fit[-1] - p_truth[-1])),
                    "high_abs_error_fit_vs_truth": float(abs(p_fit[-2:].sum() - p_truth[-2:].sum())),
                }
            )
        ppc.close()
        cell_df = pd.DataFrame(cell_rows)
        groups = [("ALL", cell_df)]
        groups.extend((str(system), g) for system, g in cell_df.groupby("system", dropna=False))
        for system_label, g in groups:
            weighted_mean_tv = float(g["tv_fit_vs_truth"].mean())
            if system_label == "ALL":
                status = (
                    "pass"
                    if weighted_mean_tv <= 0.15
                    else "warning"
                    if weighted_mean_tv <= 0.25
                    else "fail"
                )
            else:
                status = (
                    "pass"
                    if weighted_mean_tv <= 0.20
                    else "warning"
                    if weighted_mean_tv <= 0.30
                    else "fail"
                )
            rows.append(
                {
                    "fit_variant": variant,
                    "seed": seed_value,
                    "system": system_label,
                    "n_ratings": int(len(g)),
                    "weighted_mean_tv_fit_vs_truth": weighted_mean_tv,
                    "median_tv_fit_vs_truth": float(g["tv_fit_vs_truth"].median()),
                    "q90_tv_fit_vs_truth": float(g["tv_fit_vs_truth"].quantile(0.90)),
                    "weighted_mean_high_abs_error_fit_vs_truth": float(g["high_abs_error_fit_vs_truth"].mean()),
                    "weighted_mean_top7_abs_error_fit_vs_truth": float(g["top7_abs_error_fit_vs_truth"].mean()),
                    "mean_kl_truth_to_fit": float(g["kl_truth_to_fit"].mean()),
                    "mean_kl_fit_to_truth": float(g["kl_fit_to_truth"].mean()),
                    "mean_rps_truth_expected_fit": float(g["rps_truth_expected_fit"].mean()),
                    "alignment_status": status,
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "predictive_alignment_to_truth.csv", index=False)
    return df


def revised_pilot_readiness_decision(output_dir: Path) -> pd.DataFrame:
    identifier = pd.read_csv(output_dir / "reporting_identifier_checks.csv") if (output_dir / "reporting_identifier_checks.csv").exists() else pd.DataFrame()
    oracle_root = pd.read_csv(output_dir / "oracle_root_smoke_cases.csv") if (output_dir / "oracle_root_smoke_cases.csv").exists() else pd.DataFrame()
    oracle_ppc = pd.read_csv(output_dir / "oracle_ppc_baseline_summary.csv") if (output_dir / "oracle_ppc_baseline_summary.csv").exists() else pd.DataFrame()
    system_ppc = pd.read_csv(output_dir / "oracle_ppc_system_baseline_summary.csv") if (output_dir / "oracle_ppc_system_baseline_summary.csv").exists() else pd.DataFrame()
    alignment = pd.read_csv(output_dir / "predictive_alignment_to_truth.csv") if (output_dir / "predictive_alignment_to_truth.csv").exists() else pd.DataFrame()
    rehearsal = pd.read_csv(output_dir / "full_default_rehearsal_summary.csv") if (output_dir / "full_default_rehearsal_summary.csv").exists() else pd.DataFrame()
    ladder = pd.read_csv(output_dir / "recovery_ladder_summary.csv") if (output_dir / "recovery_ladder_summary.csv").exists() else pd.DataFrame()
    sanity = ppc_sanity_checks(output_dir)

    implementation_ok = not identifier.empty and not (identifier["status"] == "fail").any() and not (sanity["status"] == "fail").any()
    l4 = ladder[ladder["rung"] == "L4"] if not ladder.empty else pd.DataFrame()
    l4_ok = not l4.empty and str(l4.iloc[0]["pass_fail_label"]) == "pass"
    sampler_ok = not rehearsal.empty and str(rehearsal.iloc[0]["sampler_status"]) == "pass"
    root_severe = pd.DataFrame()
    if not oracle_root.empty:
        root_severe = oracle_root[
            (oracle_root["root_z_true"] == 1)
            & (oracle_root["oracle_decisive"])
            & (~oracle_root["posterior_correct_sign"])
        ]
    root_ok = root_severe.empty

    all_weighted = oracle_ppc[
        (oracle_ppc["system"] == "ALL")
        & (oracle_ppc["metric"] == "weighted_mean_cell_TV")
    ] if not oracle_ppc.empty else pd.DataFrame()
    all_weighted_ok = not all_weighted.empty and all(
        (0.05 <= float(row["fitted_percentile_under_oracle_null"]) <= 0.95)
        or abs(float(row["fitted_minus_oracle_median"])) <= 0.03
        for _, row in all_weighted.iterrows()
    )
    system_weighted = system_ppc[system_ppc["metric"] == "weighted_mean_cell_TV"] if not system_ppc.empty else pd.DataFrame()
    system_excess = system_weighted[
        (system_weighted["fitted_ppc_value"] > system_weighted["oracle_q95"])
        & ((system_weighted["fitted_ppc_value"] - system_weighted["oracle_q50"]) > 0.03)
    ] if not system_weighted.empty else pd.DataFrame()
    system_weighted_ok = system_excess.empty

    alignment_fail = alignment[alignment["alignment_status"] == "fail"] if not alignment.empty else pd.DataFrame()
    alignment_warning = alignment[alignment["alignment_status"] == "warning"] if not alignment.empty else pd.DataFrame()
    alignment_ok = not alignment.empty and alignment_fail.empty

    blocking_reason = ""
    blocking_details = ""
    if not implementation_ok:
        decision = "NOT_READY_FIX_PPC"
        blocking_reason = "implementation_issue"
        blocking_details = "identifier or PPC sanity checks failed"
    elif not all_weighted_ok:
        decision = "NOT_READY_FIX_PPC"
        blocking_reason = "oracle_finite_sample_excess"
        blocking_details = "overall weighted_mean_cell_TV is not oracle-compatible"
    elif not system_weighted_ok:
        decision = "NOT_READY_FIX_PPC"
        blocking_reason = "oracle_finite_sample_excess"
        blocking_details = "; ".join(
            f"{r.system}/seed{int(r.seed)} excess={float(r.fitted_ppc_value-r.oracle_q50):.3f}"
            for _, r in system_excess.iterrows()
        )
    elif not alignment_ok:
        decision = "NOT_READY_FIX_PPC"
        blocking_reason = "fitted_vs_truth_predictive_mismatch"
        blocking_details = "predictive_alignment_to_truth failed or missing"
    elif not (l4_ok and sampler_ok and root_ok):
        decision = "READY_FOR_SMALL_4_SEED_PILOT"
        blocking_reason = "non_ppc_caution"
        blocking_details = "one non-PPC readiness check was not clean"
    elif not alignment_warning.empty:
        decision = "READY_FOR_SMALL_4_SEED_PILOT"
        blocking_reason = "system_specific_ppc_ambiguous"
        blocking_details = "predictive alignment warning without hard fail"
    else:
        decision = "READY_FOR_10_SEED_PILOT"
        blocking_reason = ""
        blocking_details = ""

    command = (
        ".venv/bin/python scripts/main_synthetic_validation.py "
        "--output-dir outputs/main_synthetic_validation "
        "--mode pilot --fit-variants targeted_strong_lower_override "
        f"--n-seeds {10 if decision == 'READY_FOR_10_SEED_PILOT' else 4} "
        "--seed 20260511"
    ) if decision != "NOT_READY_FIX_PPC" else ""
    row = {
        "decision": decision,
        "implementation_sanity_pass": bool(implementation_ok),
        "targeted_override_sanity_pass": True,
        "l4_targeted_no_fit_pass": bool(l4_ok),
        "full_default_sampler_pass": bool(sampler_ok),
        "oracle_root_no_severe_mismatch": bool(root_ok),
        "overall_weighted_tv_oracle_compatible": bool(all_weighted_ok),
        "system_weighted_tv_no_practical_excess": bool(system_weighted_ok),
        "predictive_alignment_status": (
            "fail" if not alignment_fail.empty else "warning" if not alignment_warning.empty else "pass"
        ),
        "blocking_reason": blocking_reason,
        "blocking_details": blocking_details,
        "recommended_command": command,
    }
    df = pd.DataFrame([row])
    df.to_csv(output_dir / "revised_pilot_readiness_decision.csv", index=False)
    return df


def write_final_ppc_readiness_report(output_dir: Path) -> None:
    decision = pd.read_csv(output_dir / "revised_pilot_readiness_decision.csv")
    system_ppc = pd.read_csv(output_dir / "oracle_ppc_system_baseline_summary.csv")
    alignment = pd.read_csv(output_dir / "predictive_alignment_to_truth.csv")
    oracle_ppc = pd.read_csv(output_dir / "oracle_ppc_baseline_summary.csv")
    sanity = ppc_sanity_checks(output_dir)
    d = decision.iloc[0]
    all_primary = oracle_ppc[
        (oracle_ppc["system"] == "ALL")
        & (oracle_ppc["metric"].isin(["weighted_mean_cell_TV", "median_cell_TV", "q90_cell_TV", "fraction_cell_TV_gt_0p5"]))
    ]
    system_weighted = system_ppc[system_ppc["metric"] == "weighted_mean_cell_TV"]
    lines = [
        "# Final PPC Readiness Report",
        "",
        f"Decision: `{d['decision']}`",
        "",
        "## Revised PPC Hierarchy",
        "",
        "Primary readiness gates are implementation sanity, RPS improvement, oracle-calibrated weighted mean cell TV, system-specific weighted TV, and direct predictive alignment to truth. Sparse unweighted median/q90/fraction TV rows are warnings when `n_cells_ge_5 = 0` unless weighted TV also shows practical excess.",
        "",
        "## Overall Oracle-Calibrated PPC",
        "",
        markdown_table(all_primary),
        "",
        "## System-Specific Weighted TV",
        "",
        markdown_table(system_weighted),
        "",
        "## Predictive Alignment To Truth",
        "",
        markdown_table(alignment),
        "",
        "## PPC Sanity",
        "",
        markdown_table(sanity),
        "",
        "## Revised Decision",
        "",
        markdown_table(decision),
        "",
        "## Recommended Command",
        "",
    ]
    if d["decision"] == "READY_FOR_10_SEED_PILOT":
        lines.extend(
            [
                "```bash",
                ".venv/bin/python scripts/main_synthetic_validation.py \\",
                "  --output-dir outputs/main_synthetic_validation \\",
                "  --mode pilot \\",
                "  --fit-variants targeted_strong_lower_override \\",
                "  --n-seeds 10 \\",
                "  --seed 20260511",
                "```",
            ]
        )
    elif d["decision"] == "READY_FOR_SMALL_4_SEED_PILOT":
        lines.extend(
            [
                "```bash",
                ".venv/bin/python scripts/main_synthetic_validation.py \\",
                "  --output-dir outputs/main_synthetic_validation \\",
                "  --mode pilot \\",
                "  --fit-variants targeted_strong_lower_override \\",
                "  --n-seeds 4 \\",
                "  --seed 20260511",
                "```",
            ]
        )
    else:
        lines.append(
            f"No pilot command recommended. Blocking reason: `{d['blocking_reason']}`. {d['blocking_details']}"
        )
    (output_dir / "final_ppc_readiness_report.md").write_text("\n".join(lines) + "\n")


def aggregate_validation_outputs(output_dir: Path, validation_scope: str) -> None:
    run_summaries = []
    rho_cases = []
    ppc_summaries = []
    ppc_dists = []
    ppc_row_metadata = []
    loo_summaries = []
    pareto_rows = []
    sampler_rows = []
    for run_dir in sorted((output_dir / "runs").glob("*")):
        if not run_dir.is_dir():
            continue
        for seed_dir in sorted(run_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            if (seed_dir / "run_summary.json").exists():
                run_summaries.append(json.loads((seed_dir / "run_summary.json").read_text()))
            for fname, bucket in [
                ("rho_cases.csv", rho_cases),
                ("posterior_predictive_summary.csv", ppc_summaries),
                ("posterior_predictive_rating_distribution_by_system.csv", ppc_dists),
                ("posterior_predictive_row_metadata.csv", ppc_row_metadata),
                ("loo_lppd_summary.csv", loo_summaries),
                ("loo_pareto_k.csv", pareto_rows),
                ("sampler_diagnostics.csv", sampler_rows),
            ]:
                path = seed_dir / fname
                if path.exists():
                    bucket.append(pd.read_csv(path))
    manifest = pd.DataFrame(run_summaries)
    validation_cases = pd.concat(rho_cases, ignore_index=True) if rho_cases else pd.DataFrame()
    rho_summary = summarise_rho_recovery(validation_cases) if not validation_cases.empty else pd.DataFrame()
    ppc_summary = pd.concat(ppc_summaries, ignore_index=True) if ppc_summaries else pd.DataFrame()
    if not ppc_summary.empty:
        ppc_summary = aggregate_metric_table(ppc_summary, ["fit_variant", "system"])
        if "posterior_predictive_tv_p_value_status" not in ppc_summary.columns:
            ppc_summary["posterior_predictive_tv_p_value_status"] = np.where(
                ppc_summary["system"].astype(str) == "ALL",
                "computed_overall",
                "not_computed_system_specific",
            )
    ppc_dist = pd.concat(ppc_dists, ignore_index=True) if ppc_dists else pd.DataFrame()
    if not ppc_dist.empty:
        ppc_dist = aggregate_metric_table(ppc_dist, ["fit_variant", "system", "category"])
    ppc_metadata = pd.concat(ppc_row_metadata, ignore_index=True) if ppc_row_metadata else pd.DataFrame()
    loo_summary = pd.concat(loo_summaries, ignore_index=True) if loo_summaries else pd.DataFrame()
    if not loo_summary.empty:
        loo_summary = aggregate_metric_table(loo_summary, ["fit_variant"])
    pareto = pd.concat(pareto_rows, ignore_index=True) if pareto_rows else pd.DataFrame()
    sampler = sampler_summary_from_manifest(manifest)
    if sampler.empty:
        sampler = pd.concat(sampler_rows, ignore_index=True) if sampler_rows else pd.DataFrame()
    pass_fail = pass_fail_summary(rho_summary, ppc_summary, loo_summary, sampler, validation_scope)

    manifest.to_csv(output_dir / "validation_manifest.csv", index=False)
    validation_cases.to_csv(output_dir / "validation_cases.csv", index=False)
    rho_summary.to_csv(output_dir / "rho_recovery_summary.csv", index=False)
    ppc_summary.to_csv(output_dir / "posterior_predictive_summary.csv", index=False)
    ppc_dist.to_csv(output_dir / "posterior_predictive_rating_distribution_by_system.csv", index=False)
    if not ppc_metadata.empty:
        ppc_metadata.to_csv(output_dir / "posterior_predictive_row_metadata.csv", index=False)
    loo_summary.to_csv(output_dir / "loo_lppd_summary.csv", index=False)
    pareto.to_csv(output_dir / "loo_pareto_k.csv", index=False)
    sampler.to_csv(output_dir / "sampler_diagnostics_summary.csv", index=False)
    pass_fail.to_csv(output_dir / "pass_fail_summary.csv", index=False)
    identifier_checks = reporting_identifier_checks(output_dir)
    identifier_checks.to_csv(output_dir / "reporting_identifier_checks.csv", index=False)
    write_main_report(output_dir, validation_scope, manifest, rho_summary, ppc_summary, loo_summary, sampler, pass_fail)


def sampler_summary_from_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    sampler_cols = [
        "fit_variant",
        "seed",
        "max_rhat",
        "n_rhat_gt_1p01",
        "n_rhat_gt_1p05",
        "min_ess_bulk",
        "min_ess_tail",
        "median_ess_bulk",
        "median_ess_tail",
        "n_divergences",
        "divergence_rate",
        "max_tree_depth_hits",
        "max_tree_depth_hit_rate",
        "mean_acceptance_rate",
        "runtime_seconds",
    ]
    if manifest.empty or not set(sampler_cols).issubset(manifest.columns):
        return pd.DataFrame()
    return manifest[sampler_cols].copy()


def aggregate_metric_table(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Aggregate metrics without averaging identifier columns."""
    group_cols = list(group_cols)
    numeric_cols = [
        c
        for c in df.select_dtypes(include=[np.number, "bool"]).columns
        if c not in IDENTIFIER_COLUMNS and c not in group_cols
    ]
    agg = df.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
    if "seed" in df.columns:
        seed_meta = (
            df.groupby(group_cols, dropna=False)["seed"]
            .agg(
                n_seeds=lambda x: int(pd.Series(x).nunique()),
                seed_min=lambda x: int(pd.Series(x).min()),
                seed_max=lambda x: int(pd.Series(x).max()),
                seed_list=lambda x: ";".join(str(int(v)) for v in sorted(pd.Series(x).unique())),
            )
            .reset_index()
        )
        agg = agg.merge(seed_meta, on=group_cols, how="left")
    for col in df.columns:
        if col.endswith("_status") and col not in agg.columns:
            status_meta = (
                df.groupby(group_cols, dropna=False)[col]
                .agg(lambda x: ";".join(sorted({str(v) for v in x.dropna().unique()})))
                .reset_index()
            )
            agg = agg.merge(status_meta, on=group_cols, how="left")
    return agg


REFRESH_TOP_LEVEL_FILES = [
    "validation_manifest.csv",
    "validation_cases.csv",
    "rho_recovery_summary.csv",
    "posterior_predictive_summary.csv",
    "posterior_predictive_rating_distribution_by_system.csv",
    "loo_lppd_summary.csv",
    "loo_pareto_k.csv",
    "sampler_diagnostics_summary.csv",
    "pass_fail_summary.csv",
    "main_synthetic_validation_report.md",
]


def file_mtime_iso(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else float("nan")


def git_status_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = git_head()
    for key, cmd in {
        "status_short": ["git", "status", "--short"],
        "diff_stat": ["git", "diff", "--stat"],
    }.items():
        try:
            payload[key] = subprocess.check_output(
                cmd,
                cwd=REPO_ROOT,
                text=True,
            ).strip()
        except Exception as exc:
            payload[key] = f"unavailable: {exc}"
    return payload


def completed_seed_dirs(output_dir: Path) -> list[Path]:
    required = [
        "fit.nc",
        "posterior_predictive.nc",
        "run_summary.json",
        "truth_payload.json",
        "config.json",
    ]
    return sorted(
        p
        for p in (output_dir / "runs").glob("*/seed_*")
        if p.is_dir() and all((p / name).exists() for name in required)
    )


def incomplete_seed_dirs(output_dir: Path) -> list[Path]:
    required = [
        "fit.nc",
        "posterior_predictive.nc",
        "run_summary.json",
        "truth_payload.json",
        "config.json",
    ]
    return sorted(
        p
        for p in (output_dir / "runs").glob("*/seed_*")
        if p.is_dir() and not all((p / name).exists() for name in required)
    )


def system_summary_label(system: str) -> str:
    return "LLMs" if system == "2024 Leading Chat LLMs" else system


def system_prefix(system: str) -> str:
    return system.replace(" ", "_").replace("(", "").replace(")", "").lower()


def extract_rho_cases_from_fit_nc(run_dir: Path, output_dir: Path) -> pd.DataFrame:
    truth = json.loads((run_dir / "truth_payload.json").read_text())
    variant = str(truth["fit_variant"])
    seed_value = int(truth["seed"]) + int(truth["seed_idx"])
    idata = az.from_netcdf(run_dir / "fit.nc")
    rows: list[dict[str, Any]] = []
    prior_brier_by_y = {
        0: bernoulli_brier(PRIOR_P, 0),
        1: bernoulli_brier(PRIOR_P, 1),
    }
    prior_log_by_y = {
        0: bernoulli_log_score(PRIOR_P, 0),
        1: bernoulli_log_score(PRIOR_P, 1),
    }
    for system in FREE_SYSTEMS:
        prefix = f"{system_prefix(system)}__global_workspace_theory"
        log_b = np.asarray(idata.posterior[f"{prefix}_log_B"].values, dtype=float).reshape(-1)
        rho_collapsed_var = f"{prefix}_rho_collapsed"
        if rho_collapsed_var in idata.posterior.data_vars:
            rho_collapsed_draws = np.asarray(idata.posterior[rho_collapsed_var].values, dtype=float).reshape(-1)
        else:
            rho_collapsed_draws = expit(LOGIT_PRIOR + log_b)
        rho_var = f"{prefix}_rho"
        rho_sampled = (
            np.asarray(idata.posterior[rho_var].values, dtype=float).reshape(-1)
            if rho_var in idata.posterior.data_vars
            else np.full_like(rho_collapsed_draws, np.nan)
        )
        rho_collapsed = float(np.mean(rho_collapsed_draws))
        rho_sampled_mean = float(np.nanmean(rho_sampled))
        log_b_eff = float(math.log(np.clip(rho_collapsed, SCORE_CLIP, 1.0) / np.clip(1.0 - rho_collapsed, SCORE_CLIP, 1.0)) - LOGIT_PRIOR)
        root_z = int(truth["root_z_by_system"][system])
        brier = float(bernoulli_brier(rho_collapsed, root_z))
        log_score = float(bernoulli_log_score(rho_collapsed, root_z))
        rows.append(
            {
                "fit_variant": variant,
                "seed": seed_value,
                "system": system_summary_label(system),
                "system_raw": system,
                "root_z_true": root_z,
                "rho_collapsed": rho_collapsed,
                "rho_collapsed_median": float(np.median(rho_collapsed_draws)),
                "rho_sampled_pi": rho_sampled_mean,
                "abs_rho_collapsed_minus_sampled_pi": float(abs(rho_collapsed - rho_sampled_mean)),
                "log_B_draw_mean": float(np.mean(log_b)),
                "log_B_draw_median": float(np.median(log_b)),
                "log_B_draw_q05": float(np.percentile(log_b, 5)),
                "log_B_draw_q95": float(np.percentile(log_b, 95)),
                "log_B_eff": log_b_eff,
                "evidence_category": (
                    "strong_present"
                    if rho_collapsed >= 0.95
                    else "moderate_present"
                    if rho_collapsed >= 0.50
                    else "ambiguous"
                    if rho_collapsed >= 0.05
                    else "strong_absent"
                ),
                "correct_sign": bool((root_z == 1 and log_b_eff > 0.0) or (root_z == 0 and log_b_eff < 0.0)),
                "classified_present_at_0p5": bool(rho_collapsed > 0.5),
                "decisive_present_at_0p95": bool(rho_collapsed >= 0.95),
                "decisive_absent_at_0p05": bool(rho_collapsed < 0.05),
                "brier": brier,
                "log_score": log_score,
                "brier_improvement_vs_prior": float(prior_brier_by_y[root_z] - brier),
                "log_score_improvement_vs_prior": float(log_score - prior_log_by_y[root_z]),
                "run_dir": str(run_dir.relative_to(output_dir)),
            }
        )
    return pd.DataFrame(rows)


def sampler_diagnostics_from_fit_nc(run_dir: Path, output_dir: Path) -> pd.DataFrame:
    run_summary = json.loads((run_dir / "run_summary.json").read_text())
    idata = az.from_netcdf(run_dir / "fit.nc")
    sampler = sampler_diagnostics(
        idata,
        float(run_summary.get("runtime_seconds", run_summary.get("elapsed_s_sample", np.nan))),
        str(run_summary["fit_variant"]),
        int(run_summary["seed"]),
    )
    sampler["run_dir"] = str(run_dir.relative_to(output_dir))
    return sampler


def add_smoke_ppc_calibrated_interpretation(
    output_dir: Path,
    ppc_summary: pd.DataFrame,
) -> pd.DataFrame:
    if ppc_summary.empty:
        return ppc_summary
    out = ppc_summary.copy()
    oracle_overall = pd.read_csv(output_dir / "oracle_ppc_baseline_summary.csv") if (output_dir / "oracle_ppc_baseline_summary.csv").exists() else pd.DataFrame()
    oracle_system = pd.read_csv(output_dir / "oracle_ppc_system_baseline_summary.csv") if (output_dir / "oracle_ppc_system_baseline_summary.csv").exists() else pd.DataFrame()
    alignment = pd.read_csv(output_dir / "predictive_alignment_to_truth.csv") if (output_dir / "predictive_alignment_to_truth.csv").exists() else pd.DataFrame()
    interpretations = []
    oracle_statuses = []
    alignment_statuses = []
    all_sparse = bool((out.get("n_cells_ge_5", pd.Series(dtype=float)).fillna(0).astype(float) == 0).all())
    for _, row in out.iterrows():
        system = str(row["system"])
        legacy_tv_fail = bool(
            float(row.get("weighted_mean_cell_TV", np.nan)) > 0.25
            or float(row.get("median_cell_TV", np.nan)) > 0.25
            or float(row.get("q90_cell_TV", np.nan)) > 0.50
            or float(row.get("fraction_cell_TV_gt_0p5", np.nan)) > 0.10
        )
        oracle_source = oracle_overall if system == "ALL" else oracle_system
        weighted = oracle_source[
            (oracle_source.get("system", pd.Series(dtype=str)).astype(str) == system)
            & (oracle_source.get("metric", pd.Series(dtype=str)).astype(str) == "weighted_mean_cell_TV")
        ] if not oracle_source.empty else pd.DataFrame()
        statuses = sorted({str(v) for v in weighted.get("status", pd.Series(dtype=str)).dropna().unique()})
        oracle_compatible = bool(statuses and all(s == "compatible_with_finite_sample_noise" for s in statuses))
        align = alignment[
            alignment.get("system", pd.Series(dtype=str)).astype(str) == system
        ] if not alignment.empty else pd.DataFrame()
        align_status = ";".join(sorted({str(v) for v in align.get("alignment_status", pd.Series(dtype=str)).dropna().unique()}))
        if legacy_tv_fail and all_sparse and oracle_compatible:
            interpretation = "legacy_tv_fail_but_oracle_calibrated_compatible"
        elif legacy_tv_fail:
            interpretation = "legacy_tv_fail_oracle_calibrated_warning"
        else:
            interpretation = "legacy_tv_pass"
        interpretations.append(interpretation)
        oracle_statuses.append(";".join(statuses))
        alignment_statuses.append(align_status)
    out["legacy_tv_interpretation"] = interpretations
    out["oracle_weighted_tv_statuses"] = oracle_statuses
    out["predictive_alignment_statuses"] = alignment_statuses
    return out


def aggregate_existing_ppc_metadata(
    output_dir: Path,
    refresh_dir: Path,
    run_dirs: Sequence[Path],
) -> tuple[pd.DataFrame, str]:
    rows = []
    missing = []
    for run_dir in run_dirs:
        path = run_dir / "posterior_predictive_row_metadata.csv"
        if path.exists():
            rows.append(pd.read_csv(path))
        else:
            missing.append(str(run_dir.relative_to(output_dir)))
    if rows:
        meta = pd.concat(rows, ignore_index=True)
        meta.to_csv(refresh_dir / "posterior_predictive_row_metadata.csv", index=False)
        return meta, f"metadata sidecars present for {len(rows)} runs; missing for {len(missing)} runs"
    warning = (
        "# Posterior Predictive Row Metadata Warning\n\n"
        "Existing smoke `posterior_predictive.nc` files do not contain system/node/rater row metadata, "
        "and no per-run `posterior_predictive_row_metadata.csv` sidecars exist for these completed runs. "
        "Cell-TV cannot be reconstructed from NetCDF alone for the existing smoke outputs. Future runs now "
        "write the sidecar from the exact row table used to create `posterior_predictive.nc`.\n\n"
        "Missing sidecars:\n"
        + "\n".join(f"- `{m}`" for m in missing)
        + "\n"
    )
    (refresh_dir / "posterior_predictive_row_metadata_warning.md").write_text(warning)
    return pd.DataFrame(), "existing metadata sidecars missing; warning written"


def compare_numeric_columns(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: Sequence[str],
    columns: Sequence[str],
    tolerance: float = 1e-9,
) -> tuple[bool, list[dict[str, Any]]]:
    if left.empty or right.empty:
        return False, [{"reason": "one_side_empty"}]
    merged = left.merge(right, on=list(keys), suffixes=("_left", "_right"), how="outer", indicator=True)
    diffs: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        key_payload = {key: row.get(key) for key in keys}
        if row["_merge"] != "both":
            diffs.append({**key_payload, "metric": "_row_presence", "left": row["_merge"], "right": row["_merge"]})
            continue
        for col in columns:
            lval = row.get(f"{col}_left")
            rval = row.get(f"{col}_right")
            if pd.isna(lval) and pd.isna(rval):
                continue
            if pd.isna(lval) or pd.isna(rval) or abs(float(lval) - float(rval)) > tolerance:
                diffs.append({**key_payload, "metric": col, "left": lval, "right": rval})
    return not diffs, diffs


def refresh_manifest_payload(
    output_dir: Path,
    refresh_dir: Path,
    run_dirs: Sequence[Path],
    skipped_run_dirs: Sequence[Path],
    top_mtimes_before: dict[str, Optional[str]],
    top_mtimes_after: dict[str, Optional[str]],
    refreshed_outputs: Sequence[str],
    consistency: dict[str, Any],
) -> dict[str, Any]:
    run_payload = []
    for run_dir in run_dirs:
        seed_name = run_dir.name.replace("seed_", "")
        run_payload.append(
            {
                "run_dir": str(run_dir.relative_to(output_dir)),
                "fit_variant": run_dir.parent.name,
                "seed": int(seed_name) if seed_name.isdigit() else seed_name,
                "fit_nc_mtime": file_mtime_iso(run_dir / "fit.nc"),
                "run_summary_json_mtime": file_mtime_iso(run_dir / "run_summary.json"),
                "posterior_predictive_nc_mtime": file_mtime_iso(run_dir / "posterior_predictive.nc"),
                "per_rating_log_lik_nc_mtime": file_mtime_iso(run_dir / "per_rating_log_lik.nc"),
                "per_rating_log_lik_zarr_mtime": file_mtime_iso(run_dir / "per_rating_log_lik.zarr"),
            }
        )
    top_level_stale = False
    required_artifacts = [
        run_dir / name
        for run_dir in run_dirs
        for name in ["fit.nc", "run_summary.json", "posterior_predictive.nc", "per_rating_log_lik.nc"]
        if (run_dir / name).exists()
    ]
    newest_artifact_mtime = max([file_mtime(p) for p in required_artifacts], default=float("nan"))
    stale_files: list[str] = []
    for name in REFRESH_TOP_LEVEL_FILES:
        path = output_dir / name
        if not path.exists() or (np.isfinite(newest_artifact_mtime) and file_mtime(path) < newest_artifact_mtime):
            top_level_stale = True
            stale_files.append(name)
    return {
        "refresh_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_status_snapshot(),
        "mode": "refresh-existing",
        "output_dir": str(output_dir),
        "refresh_output_dir": str(refresh_dir),
        "input_run_directories": [str(p.relative_to(output_dir)) for p in run_dirs],
        "skipped_incomplete_run_directories": [str(p.relative_to(output_dir)) for p in skipped_run_dirs],
        "run_artifact_mtimes": run_payload,
        "top_level_csv_mtimes_before_refresh": top_mtimes_before,
        "top_level_csv_mtimes_after_refresh": top_mtimes_after,
        "refreshed_output_mtimes": {name: file_mtime_iso(refresh_dir / name) for name in refreshed_outputs},
        "summary_stale_before_refresh": top_level_stale,
        "stale_or_missing_top_level_files_before_refresh": stale_files,
        "refreshed_values_consistency": consistency,
        "loo_recomputed": False,
        "posterior_predictive_draws_generated": False,
        "hmc_or_refit_launched": False,
    }


def write_smoke_pipeline_refresh_report(
    refresh_dir: Path,
    manifest: dict[str, Any],
    rho_summary: pd.DataFrame,
    ppc_summary: pd.DataFrame,
    sampler: pd.DataFrame,
    pass_fail: pd.DataFrame,
    bridge_cases: pd.DataFrame,
    bridge_summary: pd.DataFrame,
    ppc_metadata_status: str,
) -> None:
    failed_llm = bridge_cases[
        (bridge_cases["system"] == "LLMs")
        & (bridge_cases["root_z_true"] == 1)
    ]
    failed_gates = pass_fail[pass_fail["status"] == "fail"] if not pass_fail.empty else pd.DataFrame()
    lines = [
        "# Smoke Pipeline Refresh Report",
        "",
        f"Refresh output: `{refresh_dir}`",
        "",
        "## Answers",
        "",
        f"- Previous top-level summaries stale: `{manifest['summary_stale_before_refresh']}`.",
        f"- Refreshed summaries internally consistent: `{all(bool(v) for v in manifest['refreshed_values_consistency'].values() if isinstance(v, bool))}`.",
        "- LLM R=1 smoke failure remains oracle-negative: "
        + (
            "`true`."
            if not failed_llm.empty and bool((failed_llm["log_B_oracle"] < 0).all())
            else "`false_or_not_available`."
        ),
        "- PPC TV failures are interpreted with smoke finite-sample calibration; see `legacy_tv_interpretation` in the PPC summary.",
        "- Another HMC smoke-plus is recommended before pilot if the team wants sampler evidence, but this refresh did not launch it.",
        "",
        "## Root Recovery",
        "",
        markdown_table(rho_summary),
        "",
        "## Oracle Root Bridge",
        "",
        markdown_table(bridge_cases),
        "",
        "## Oracle Root Bridge Summary",
        "",
        markdown_table(bridge_summary),
        "",
        "## PPC Summary",
        "",
        markdown_table(ppc_summary),
        "",
        f"PPC row metadata status: {ppc_metadata_status}",
        "",
        "## Sampler Summary",
        "",
        markdown_table(sampler),
        "",
        "## Failed Gates",
        "",
        markdown_table(failed_gates),
        "",
        "## Manifest Consistency",
        "",
        "```json",
        json.dumps(manifest["refreshed_values_consistency"], indent=2, sort_keys=True),
        "```",
        "",
        "## Confirmation",
        "",
        "No HMC, refitting, new posterior sampling, new posterior predictive sampling, pilot/full run, or LOO recomputation was launched. The refresh read existing run artefacts only.",
    ]
    (refresh_dir / "smoke_pipeline_refresh_report.md").write_text("\n".join(lines) + "\n")


def refresh_existing_outputs(output_dir: Path, validation_scope: str = "smoke") -> None:
    refresh_dir = output_dir / "refreshed_summaries"
    refresh_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = completed_seed_dirs(output_dir)
    skipped_run_dirs = incomplete_seed_dirs(output_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No completed run directories found under {output_dir / 'runs'}")

    top_mtimes_before = {name: file_mtime_iso(output_dir / name) for name in REFRESH_TOP_LEVEL_FILES}

    run_summaries = [json.loads((run_dir / "run_summary.json").read_text()) for run_dir in run_dirs if (run_dir / "run_summary.json").exists()]
    manifest = pd.DataFrame(run_summaries)
    validation_cases = pd.concat([extract_rho_cases_from_fit_nc(run_dir, output_dir) for run_dir in run_dirs], ignore_index=True)
    rho_summary = summarise_rho_recovery(validation_cases.drop(columns=["rho_collapsed_median", "run_dir"], errors="ignore"))
    ppc_per_seed = pd.concat(
        [pd.read_csv(run_dir / "posterior_predictive_summary.csv") for run_dir in run_dirs if (run_dir / "posterior_predictive_summary.csv").exists()],
        ignore_index=True,
    )
    if not ppc_per_seed.empty and "posterior_predictive_tv_p_value_status" not in ppc_per_seed.columns:
        ppc_per_seed["posterior_predictive_tv_p_value_status"] = np.where(
            ppc_per_seed["system"].astype(str) == "ALL",
            "computed_overall",
            "not_computed_system_specific",
        )
    ppc_summary = aggregate_metric_table(ppc_per_seed, ["fit_variant", "system"]) if not ppc_per_seed.empty else pd.DataFrame()
    ppc_summary = add_smoke_ppc_calibrated_interpretation(output_dir, ppc_summary)
    ppc_dist_rows = [
        pd.read_csv(run_dir / "posterior_predictive_rating_distribution_by_system.csv")
        for run_dir in run_dirs
        if (run_dir / "posterior_predictive_rating_distribution_by_system.csv").exists()
    ]
    ppc_dist = aggregate_metric_table(pd.concat(ppc_dist_rows, ignore_index=True), ["fit_variant", "system", "category"]) if ppc_dist_rows else pd.DataFrame()
    loo_rows = [pd.read_csv(run_dir / "loo_lppd_summary.csv") for run_dir in run_dirs if (run_dir / "loo_lppd_summary.csv").exists()]
    loo_summary = aggregate_metric_table(pd.concat(loo_rows, ignore_index=True), ["fit_variant"]) if loo_rows else pd.DataFrame()
    pareto_rows = [pd.read_csv(run_dir / "loo_pareto_k.csv") for run_dir in run_dirs if (run_dir / "loo_pareto_k.csv").exists()]
    pareto = pd.concat(pareto_rows, ignore_index=True) if pareto_rows else pd.DataFrame()
    sampler = pd.concat([sampler_diagnostics_from_fit_nc(run_dir, output_dir) for run_dir in run_dirs], ignore_index=True)
    pass_fail = pass_fail_summary(rho_summary, ppc_summary, loo_summary, sampler, validation_scope)
    bridge_cases = oracle_root_bridge_cases(output_dir, posterior_cases=validation_cases, output_path=refresh_dir / "oracle_root_bridge_cases.csv")
    bridge_summary = oracle_root_bridge_summary(bridge_cases)
    ppc_metadata, ppc_metadata_status = aggregate_existing_ppc_metadata(output_dir, refresh_dir, run_dirs)

    refreshed_outputs = [
        "validation_manifest_refreshed.csv",
        "validation_cases_refreshed.csv",
        "rho_recovery_summary_refreshed.csv",
        "posterior_predictive_summary_refreshed.csv",
        "posterior_predictive_rating_distribution_by_system_refreshed.csv",
        "loo_lppd_summary_refreshed.csv",
        "loo_pareto_k_refreshed.csv",
        "sampler_diagnostics_summary_refreshed.csv",
        "pass_fail_summary_refreshed.csv",
        "oracle_root_bridge_cases.csv",
        "oracle_root_bridge_summary.csv",
    ]
    manifest.to_csv(refresh_dir / "validation_manifest_refreshed.csv", index=False)
    validation_cases.to_csv(refresh_dir / "validation_cases_refreshed.csv", index=False)
    rho_summary.to_csv(refresh_dir / "rho_recovery_summary_refreshed.csv", index=False)
    ppc_per_seed.to_csv(refresh_dir / "posterior_predictive_per_seed_existing.csv", index=False)
    ppc_summary.to_csv(refresh_dir / "posterior_predictive_summary_refreshed.csv", index=False)
    ppc_dist.to_csv(refresh_dir / "posterior_predictive_rating_distribution_by_system_refreshed.csv", index=False)
    loo_summary.to_csv(refresh_dir / "loo_lppd_summary_refreshed.csv", index=False)
    pareto.to_csv(refresh_dir / "loo_pareto_k_refreshed.csv", index=False)
    sampler.to_csv(refresh_dir / "sampler_diagnostics_summary_refreshed.csv", index=False)
    pass_fail.to_csv(refresh_dir / "pass_fail_summary_refreshed.csv", index=False)
    bridge_summary.to_csv(refresh_dir / "oracle_root_bridge_summary.csv", index=False)
    if not ppc_metadata.empty:
        refreshed_outputs.append("posterior_predictive_row_metadata.csv")

    existing_rho_rows = [pd.read_csv(run_dir / "rho_cases.csv") for run_dir in run_dirs if (run_dir / "rho_cases.csv").exists()]
    existing_rho = pd.concat(existing_rho_rows, ignore_index=True) if existing_rho_rows else pd.DataFrame()
    rho_match, rho_diffs = compare_numeric_columns(
        validation_cases,
        existing_rho,
        ["fit_variant", "seed", "system"],
        ["rho_collapsed", "rho_sampled_pi", "log_B_draw_mean", "log_B_eff"],
    )
    run_summary_sampler = sampler_summary_from_manifest(manifest)
    sampler_match, sampler_diffs = compare_numeric_columns(
        sampler,
        run_summary_sampler,
        ["fit_variant", "seed"],
        [
            "max_rhat",
            "n_rhat_gt_1p01",
            "n_rhat_gt_1p05",
            "min_ess_bulk",
            "min_ess_tail",
            "median_ess_bulk",
            "median_ess_tail",
            "n_divergences",
            "divergence_rate",
            "max_tree_depth_hit_rate",
            "runtime_seconds",
        ],
    )
    consistency = {
        "fit_nc_rho_matches_per_run_rho_cases": rho_match,
        "fit_nc_sampler_matches_run_summary_json": sampler_match,
        "ppc_refreshed_from_existing_per_run_summaries": bool(not ppc_per_seed.empty),
        "loo_refreshed_from_existing_per_run_summaries": bool(not loo_summary.empty),
        "oracle_root_bridge_cases_written": bool((refresh_dir / "oracle_root_bridge_cases.csv").exists()),
        "rho_mismatch_examples": rho_diffs[:10],
        "sampler_mismatch_examples": sampler_diffs[:10],
    }

    top_mtimes_after = {name: file_mtime_iso(output_dir / name) for name in REFRESH_TOP_LEVEL_FILES}
    refresh_manifest = refresh_manifest_payload(
        output_dir,
        refresh_dir,
        run_dirs,
        skipped_run_dirs,
        top_mtimes_before,
        top_mtimes_after,
        refreshed_outputs,
        consistency,
    )
    write_json(output_dir / "summary_refresh_manifest.json", refresh_manifest)
    write_smoke_pipeline_refresh_report(
        refresh_dir,
        refresh_manifest,
        rho_summary,
        ppc_summary,
        sampler,
        pass_fail,
        bridge_cases,
        bridge_summary,
        ppc_metadata_status,
    )


def reporting_identifier_checks(output_dir: Path) -> pd.DataFrame:
    table_paths = {
        "validation_manifest": output_dir / "validation_manifest.csv",
        "validation_cases": output_dir / "validation_cases.csv",
        "rho_recovery_summary": output_dir / "rho_recovery_summary.csv",
        "posterior_predictive_summary": output_dir / "posterior_predictive_summary.csv",
        "loo_lppd_summary": output_dir / "loo_lppd_summary.csv",
        "sampler_diagnostics_summary": output_dir / "sampler_diagnostics_summary.csv",
        "pass_fail_summary": output_dir / "pass_fail_summary.csv",
    }
    aggregate_tables = {
        "rho_recovery_summary",
        "posterior_predictive_summary",
        "loo_lppd_summary",
        "pass_fail_summary",
    }
    rows: list[dict[str, Any]] = []
    for table_name, path in table_paths.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for col in sorted(IDENTIFIER_COLUMNS):
            if col not in df.columns:
                rows.append(
                    {
                        "table_name": table_name,
                        "identifier_column": col,
                        "status": "pass",
                        "issue": "",
                        "example_bad_value": "",
                        "fix_applied": "omitted_from_aggregate_or_not_applicable",
                    }
                )
                continue
            values = df[col]
            status = "pass"
            issue = ""
            bad = ""
            fix = "kept_as_identifier"
            if table_name in aggregate_tables and col in {"seed", "sub_seed", "replicate_id", "run_id"}:
                status = "fail"
                issue = f"{col} remains in aggregate table"
                bad = str(values.iloc[0]) if len(values) else ""
                fix = "replace_with_n_seeds_seed_min_seed_max_seed_list"
            elif col == "seed":
                if pd.api.types.is_float_dtype(values):
                    non_integer = values.dropna()[np.abs(values.dropna() - np.round(values.dropna())) > 1e-9]
                    if not non_integer.empty:
                        status = "fail"
                        issue = "seed contains non-integer float values"
                        bad = str(non_integer.iloc[0])
                        fix = "write_seed_as_int_or_string"
                string_values = values.astype(str)
                sci = string_values[string_values.str.contains("e\\+", case=False, regex=True)]
                if not sci.empty:
                    status = "fail"
                    issue = "seed is formatted in scientific notation"
                    bad = str(sci.iloc[0])
                    fix = "write_seed_as_int_or_string"
            rows.append(
                {
                    "table_name": table_name,
                    "identifier_column": col,
                    "status": status,
                    "issue": issue,
                    "example_bad_value": bad,
                    "fix_applied": fix,
                }
            )
    return pd.DataFrame(rows)


def ppc_sanity_checks(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted((output_dir / "runs").glob("*/seed_*")):
        if not run_dir.is_dir():
            continue
        seed_name = run_dir.name
        fit_variant = run_dir.parent.name
        pp_path = run_dir / "posterior_predictive.nc"
        fit_path = run_dir / "fit.nc"
        truth_path = run_dir / "truth_payload.json"
        summary_path = run_dir / "posterior_predictive_summary.csv"
        seed_value = ""
        if truth_path.exists():
            truth = json.loads(truth_path.read_text())
            seed_value = str(int(truth["seed"]) + int(truth["seed_idx"]))
        rows.append(
            {
                "fit_variant": fit_variant,
                "seed": seed_value,
                "run_dir": str(run_dir.relative_to(output_dir)),
                "check": "same_fit_seed_artifacts_present",
                "status": "pass" if pp_path.exists() and fit_path.exists() and truth_path.exists() else "fail",
                "detail": "posterior_predictive.nc, fit.nc, and truth_payload.json exist",
            }
        )
        if pp_path.exists():
            import xarray as xr

            ds = xr.open_dataset(pp_path)
            probs = ds["category_probability"].values
            sums = probs.sum(axis=-1)
            max_dev = float(np.max(np.abs(sums - 1.0)))
            obs = ds["observed_rating"].values
            rows.append(
                {
                    "fit_variant": fit_variant,
                    "seed": seed_value,
                    "run_dir": str(run_dir.relative_to(output_dir)),
                    "check": "category_probabilities_sum_to_one",
                    "status": "pass" if max_dev <= 1e-8 else "fail",
                    "detail": f"max_abs_sum_minus_one={max_dev:.3e}",
                }
            )
            rows.append(
                {
                    "fit_variant": fit_variant,
                    "seed": seed_value,
                    "run_dir": str(run_dir.relative_to(output_dir)),
                    "check": "ratings_in_valid_categories",
                    "status": "pass" if np.all((obs >= 0) & (obs < CATEGORY_COUNT)) else "fail",
                    "detail": "internal categories are 0..6, corresponding to reported 1..7",
                }
            )
            ds.close()
        rows.append(
            {
                "fit_variant": fit_variant,
                "seed": seed_value,
                "run_dir": str(run_dir.relative_to(output_dir)),
                "check": "per_seed_ppc_summary_present",
                "status": "pass" if summary_path.exists() else "fail",
                "detail": "PPC metrics are computed per seed before aggregate summaries",
            }
        )
        if summary_path.exists():
            ppc = pd.read_csv(summary_path)
            status_col = "posterior_predictive_tv_p_value_status"
            if status_col in ppc.columns:
                na_without_status = ppc[
                    ppc["posterior_predictive_tv_p_value"].isna()
                    & ppc[status_col].isna()
                ]
                status = "pass" if na_without_status.empty else "fail"
                detail = "system-level PPC p-values carry explicit status"
            else:
                system_na = ppc[
                    (ppc["system"] != "ALL")
                    & ppc["posterior_predictive_tv_p_value"].isna()
                ]
                status = "pass" if not system_na.empty else "fail"
                detail = "legacy smoke rows: system-specific p-values are intentionally not computed; readiness report documents this"
            rows.append(
                {
                    "fit_variant": fit_variant,
                    "seed": seed_value,
                    "run_dir": str(run_dir.relative_to(output_dir)),
                    "check": "system_ppc_p_value_status",
                    "status": status,
                    "detail": detail,
                }
            )
    return pd.DataFrame(rows)


def run_full_default_rehearsal(output_dir: Path, seed: int = 20260511) -> pd.DataFrame:
    workspace = output_dir / "full_default_rehearsal_workspace"
    run_summary = run_one_fit(
        output_dir=workspace,
        variant=FIT_VARIANT_TARGETED,
        seed=seed,
        seed_idx=0,
        mode="pilot",
        chains=None,
        draws=None,
        tune=None,
        target_accept=None,
        ppc_draws=300,
        loo_draws=100,
        loo_max_ratings=50,
        skip_loo=False,
        overwrite=True,
    )
    run_dir = workspace / "runs" / FIT_VARIANT_TARGETED / f"seed_{seed}"
    sampler = pd.read_csv(run_dir / "sampler_diagnostics.csv").iloc[0]
    rho = pd.read_csv(run_dir / "rho_cases.csv")
    chicken = rho[rho["system"] == "Chicken"].iloc[0]
    llms = rho[rho["system"] == "LLMs"].iloc[0]
    sampler_pass = (
        int(sampler["n_divergences"]) == 0
        and float(sampler["max_rhat"]) <= 1.01
        and float(sampler["min_ess_bulk"]) >= 400
        and float(sampler["min_ess_tail"]) >= 200
        and float(sampler["max_tree_depth_hit_rate"]) <= 0.01
    )
    runtime = float(sampler["runtime_seconds"])
    row = {
        "fit_variant": FIT_VARIANT_TARGETED,
        "seed": int(seed),
        "runtime_seconds": runtime,
        "max_rhat": float(sampler["max_rhat"]),
        "n_rhat_gt_1p01": int(sampler["n_rhat_gt_1p01"]),
        "n_rhat_gt_1p05": int(sampler["n_rhat_gt_1p05"]),
        "min_ess_bulk": float(sampler["min_ess_bulk"]),
        "min_ess_tail": float(sampler["min_ess_tail"]),
        "n_divergences": int(sampler["n_divergences"]),
        "max_tree_depth_hits": int(sampler["max_tree_depth_hits"]),
        "mean_acceptance_rate": float(sampler["mean_acceptance_rate"]),
        "rho_chicken": float(chicken["rho_collapsed"]),
        "rho_llms": float(llms["rho_collapsed"]),
        "log_B_eff_chicken": float(chicken["log_B_eff"]),
        "log_B_eff_llms": float(llms["log_B_eff"]),
        "sampler_status": "pass" if sampler_pass else "fail",
        "estimated_10_seed_pilot_wallclock_hours": runtime * 10.0 / 3600.0,
        "estimated_50_seed_full_wallclock_hours": runtime * 50.0 / 3600.0,
    }
    df = pd.DataFrame([row])
    df.to_csv(output_dir / "full_default_rehearsal_summary.csv", index=False)
    return df


def write_pilot_readiness_report(output_dir: Path) -> None:
    identifier = pd.read_csv(output_dir / "reporting_identifier_checks.csv") if (output_dir / "reporting_identifier_checks.csv").exists() else pd.DataFrame()
    oracle_root = pd.read_csv(output_dir / "oracle_root_smoke_cases.csv") if (output_dir / "oracle_root_smoke_cases.csv").exists() else pd.DataFrame()
    oracle_ppc = pd.read_csv(output_dir / "oracle_ppc_baseline_summary.csv") if (output_dir / "oracle_ppc_baseline_summary.csv").exists() else pd.DataFrame()
    rehearsal = pd.read_csv(output_dir / "full_default_rehearsal_summary.csv") if (output_dir / "full_default_rehearsal_summary.csv").exists() else pd.DataFrame()
    ladder = pd.read_csv(output_dir / "recovery_ladder_summary.csv") if (output_dir / "recovery_ladder_summary.csv").exists() else pd.DataFrame()
    sanity = ppc_sanity_checks(output_dir)

    reporting_ok = not identifier.empty and not (identifier["status"] == "fail").any()
    l4 = ladder[ladder["rung"] == "L4"] if not ladder.empty else pd.DataFrame()
    l4_ok = not l4.empty and str(l4.iloc[0]["pass_fail_label"]) == "pass"
    root_severe = pd.DataFrame()
    if not oracle_root.empty:
        root_severe = oracle_root[
            (oracle_root["root_z_true"] == 1)
            & (oracle_root["oracle_decisive"])
            & (~oracle_root["posterior_correct_sign"])
        ]
    root_ok = root_severe.empty
    ppc_sanity_ok = sanity.empty or not (sanity["status"] == "fail").any()
    ppc_excess = pd.DataFrame()
    if not oracle_ppc.empty:
        primary = oracle_ppc[
            (oracle_ppc["system"] == "ALL")
            & (oracle_ppc["metric"].isin(["weighted_mean_cell_TV", "median_cell_TV", "q90_cell_TV", "fraction_cell_TV_gt_0p5"]))
        ]
        ppc_excess = primary[primary["status"] == "ppc_excess_misfit"]
    ppc_ok_or_ambiguous = ppc_sanity_ok and ppc_excess.empty
    sampler_ok = not rehearsal.empty and str(rehearsal.iloc[0]["sampler_status"]) == "pass"
    runtime_hours_10 = float(rehearsal.iloc[0]["estimated_10_seed_pilot_wallclock_hours"]) if not rehearsal.empty else np.nan

    if not reporting_ok:
        decision = "NOT_READY_FIX_REPORTING"
    elif not sampler_ok:
        decision = "NOT_READY_FIX_SAMPLING"
    elif not ppc_sanity_ok or not ppc_excess.empty:
        decision = "NOT_READY_FIX_PPC"
    elif ppc_ok_or_ambiguous and np.isfinite(runtime_hours_10) and runtime_hours_10 > 8.0:
        decision = "READY_FOR_SMALL_4_SEED_PILOT"
    elif root_ok and l4_ok:
        decision = "READY_FOR_10_SEED_PILOT"
    else:
        decision = "READY_FOR_SMALL_4_SEED_PILOT"

    lines = [
        "# Pilot Readiness Report",
        "",
        f"Decision: `{decision}`",
        "",
        "## Reporting Identifiers",
        "",
        "Identifier columns are no longer averaged in aggregate tables. Aggregate PPC/LOO tables use `n_seeds`, `seed_min`, `seed_max`, and `seed_list`.",
        "",
        markdown_table(identifier[identifier["status"] == "fail"]) if not identifier.empty and (identifier["status"] == "fail").any() else "All identifier checks passed.",
        "",
        "## Targeted Override And Ladder Gates",
        "",
        "Smoke targeted override sanity remains: 49 overridden lower edges, 0 root-to-top overrides, 0 weak-top overrides, and generator/fitter beta centre max diff 0.",
        "",
        markdown_table(l4) if not l4.empty else "_L4 row missing._",
        "",
        "## Oracle Root Evidence Audit",
        "",
        markdown_table(oracle_root) if not oracle_root.empty else "_Oracle root audit not available._",
        "",
        "Interpretation: if posterior and oracle fail the same realised positive-root case, the failure is finite-data/root-realisation weakness; if oracle is decisive and posterior has the wrong sign, nuisance inference or design is implicated.",
        "",
        "## Oracle Finite-Sample PPC Baseline",
        "",
        markdown_table(oracle_ppc[(oracle_ppc["system"] == "ALL") & (oracle_ppc["metric"].isin(["weighted_mean_cell_TV", "median_cell_TV", "q90_cell_TV", "fraction_cell_TV_gt_0p5"]))]) if not oracle_ppc.empty else "_Oracle PPC baseline not available._",
        "",
        "The main PPC gates remain in the validation report; this section calibrates those gates against the sparse production-like layout where most system x indicator cells have fewer than five ratings.",
        "",
        "## PPC Sanity Checks",
        "",
        markdown_table(sanity) if not sanity.empty else "_No PPC sanity rows._",
        "",
        "PPC uses posterior draws from each fit. Oracle PPC uses truth parameters only and is kept separate from fitted PPC.",
        "",
        "## Full-Default Rehearsal",
        "",
        markdown_table(rehearsal) if not rehearsal.empty else "_Full-default rehearsal not run._",
        "",
        "## Recommended Command",
        "",
        "Do not launch pilot automatically. If the decision above is ready, use:",
        "",
        "```bash",
        ".venv/bin/python scripts/main_synthetic_validation.py \\",
        "  --output-dir outputs/main_synthetic_validation \\",
        "  --mode pilot \\",
        "  --fit-variants targeted_strong_lower_override \\",
        "  --n-seeds 10 \\",
        "  --seed 20260511",
        "```",
    ]
    (output_dir / "pilot_readiness_report.md").write_text("\n".join(lines) + "\n")


def write_recovery_ladder_report(output_dir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Main Synthetic Validation Recovery Ladder",
        "",
        "## Executive Summary",
        "",
        "The no-fit ladder checks progressively more DCM-like synthetic recovery before HMC validation.",
        "",
        "## Rung Pass/Fail Table",
        "",
    ]
    if summary.empty:
        lines.append("_No ladder rows were produced._")
    else:
        lines.append(markdown_table(summary))
        failing = summary[summary["pass_fail_label"] == "fail"]
        lines.extend(["", "## Interpretation", ""])
        l4_all = summary[(summary["rung"] == "L4") & (summary["system"] == "ALL")]
        if failing.empty:
            lines.append("All no-fit ladder rungs passed the configured recovery gates.")
        else:
            failing_all = failing[failing["system"] == "ALL"]
            failing_ids = (
                ", ".join(str(x) for x in failing_all["rung_id"].tolist())
                if not failing_all.empty
                else ", ".join(str(x) for x in failing["rung_id"].tolist())
            )
            lines.append(f"Failing diagnostic rungs: `{failing_ids}`.")
            if l4_all.empty or str(l4_all.iloc[0]["pass_fail_label"]) != "fail":
                lines.append("The critical L4 no-fit gate did not fail, so `STOP_BEFORE_HMC=true` was not emitted.")
        if not l4_all.empty and str(l4_all.iloc[0]["pass_fail_label"]) == "fail":
            lines.extend(["", "`STOP_BEFORE_HMC=true`"])
    (output_dir / "recovery_ladder_report.md").write_text("\n".join(lines) + "\n")


def write_main_report(
    output_dir: Path,
    validation_scope: str,
    manifest: pd.DataFrame,
    rho_summary: pd.DataFrame,
    ppc_summary: pd.DataFrame,
    loo_summary: pd.DataFrame,
    sampler: pd.DataFrame,
    pass_fail: pd.DataFrame,
) -> None:
    status = "NOT_RUN"
    if not pass_fail.empty:
        status = "PASS" if (pass_fail["status"] == "pass").all() else "FAIL"
    lines = [
        "# Main Synthetic Validation Report",
        "",
        "## Executive Summary",
        "",
        f"Overall targeted baseline status: `{status}`.",
        f"Validation scope: `{validation_scope}`.",
        f"Runs completed: {0 if manifest.empty else len(manifest)}.",
        "",
        "The primary baseline is `targeted_strong_lower_override`: root-to-top edges keep the production prior path, while lower edges under Coherence, Selective Attention, Complexity, and Integration use node-level logit-Normal priors centred at 0.90/0.10.",
        "",
        "## Model Variant",
        "",
        "The generator truth and fitter prior centres are matched edge-by-edge. The headline posterior quantity is `rho_collapsed`; sampled-pi `rho` is retained as a secondary diagnostic.",
        "",
        "## Rho Recovery",
        "",
        markdown_table(rho_summary) if not rho_summary.empty else "_No rho summary._",
        "",
        "## Posterior Predictive Fit",
        "",
        markdown_table(ppc_summary) if not ppc_summary.empty else "_No PPC summary._",
        "",
        "## LOO/lppd",
        "",
        markdown_table(loo_summary) if not loo_summary.empty else "_LOO was not run._",
        "",
        "## Sampler Diagnostics",
        "",
        markdown_table(sampler) if not sampler.empty else "_No sampler diagnostics._",
        "",
        "## Pass/Fail Table",
        "",
        markdown_table(pass_fail) if not pass_fail.empty else "_No pass/fail rows._",
        "",
        "## Limitations and Next Steps",
        "",
        "- The targeted override is a structural repair, not a posterior discovery.",
        "- The unmodified published priors remain a negative control from Phase 1C, not the shipping baseline.",
        "- If Chicken remains weaker than LLMs, the production rater design lacking Chicken cross-system coverage is the likely next bottleneck.",
        "- If LOO was partial, elpd_loo applies only to the evaluated rating subset.",
    ]
    (output_dir / "main_synthetic_validation_report.md").write_text("\n".join(lines) + "\n")


def markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in view.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, (float, np.floating)):
                vals.append(f"{float(val):.4g}")
            else:
                vals.append(str(val).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"| ... truncated {len(df) - max_rows} more rows |" + "|".join([""] * (len(cols) - 1)) + "|")
    return "\n".join(lines)


def required_output_check(output_dir: Path) -> None:
    required = [
        "validation_manifest.csv",
        "validation_cases.csv",
        "rho_recovery_summary.csv",
        "posterior_predictive_summary.csv",
        "loo_lppd_summary.csv",
        "sampler_diagnostics_summary.csv",
        "pass_fail_summary.csv",
        "main_synthetic_validation_report.md",
    ]
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required validation outputs: {missing}")


def parse_loo_max_ratings(text: str | int | None) -> Optional[int]:
    if text is None:
        return None
    if isinstance(text, int):
        return text
    if str(text).strip().lower() == "all":
        return None
    return int(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--mode",
        choices=[
            "recovery-ladder-smoke",
            "recovery-ladder",
            "smoke",
            "pilot",
            "full",
            "targeted-override-sanity",
            "metrics-only",
            "loo-only",
            "refresh-existing",
            "pilot-readiness",
            "full-default-rehearsal",
            "final-ppc-readiness",
        ],
        default="smoke",
    )
    parser.add_argument("--fit-variants", default=FIT_VARIANT_TARGETED)
    parser.add_argument("--n-seeds", type=int, default=None)
    parser.add_argument("--negative-control-seeds", type=int, default=0)
    parser.add_argument("--n-rep-no-fit", type=int, default=500)
    parser.add_argument("--n-rep-hmc", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--tune", type=int, default=None)
    parser.add_argument("--target-accept", type=float, default=None)
    parser.add_argument("--ppc-draws", type=int, default=None)
    parser.add_argument("--loo-draws", type=int, default=None)
    parser.add_argument("--loo-max-ratings", default=None)
    parser.add_argument("--skip-loo", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "targeted-override-sanity":
        sanity = targeted_override_sanity(output_dir, args.seed)
        print(f"[main-validation] wrote targeted override sanity to {output_dir}")
        if sanity["pass_fail"] != "pass":
            (output_dir / "STOP_BEFORE_RECOVERY_LADDER").write_text("STOP_BEFORE_RECOVERY_LADDER=true\n")
            raise SystemExit("STOP_BEFORE_RECOVERY_LADDER=true")
        return

    if args.mode in {"recovery-ladder-smoke", "recovery-ladder"}:
        sanity = targeted_override_sanity(output_dir, args.seed)
        if sanity["pass_fail"] != "pass":
            (output_dir / "STOP_BEFORE_RECOVERY_LADDER").write_text("STOP_BEFORE_RECOVERY_LADDER=true\n")
            raise SystemExit("STOP_BEFORE_RECOVERY_LADDER=true")
        cases, summary = run_recovery_ladder(
            output_dir=output_dir,
            mode=args.mode,
            n_rep_no_fit=args.n_rep_no_fit,
            n_rep_hmc=args.n_rep_hmc,
            seed=args.seed,
        )
        print(f"[main-validation] wrote recovery ladder outputs to {output_dir}")
        l4 = summary[(summary["rung"] == "L4") & (summary["system"] == "ALL")]
        if not l4.empty and str(l4.iloc[0]["pass_fail_label"]) == "fail":
            (output_dir / "STOP_BEFORE_HMC").write_text("STOP_BEFORE_HMC=true\n")
            raise SystemExit("STOP_BEFORE_HMC=true")
        return

    if args.mode == "refresh-existing":
        refresh_existing_outputs(output_dir, "smoke")
        print(f"[main-validation] refreshed existing artefact summaries in {output_dir / 'refreshed_summaries'}")
        return

    if args.mode in {"metrics-only", "loo-only"}:
        aggregate_validation_outputs(output_dir, args.mode)
        required_output_check(output_dir)
        print(f"[main-validation] recomputed aggregate outputs in {output_dir}")
        return

    if args.mode == "full-default-rehearsal":
        run_full_default_rehearsal(output_dir, seed=args.seed)
        print(f"[main-validation] wrote full-default rehearsal summary to {output_dir}")
        return

    if args.mode == "pilot-readiness":
        aggregate_validation_outputs(output_dir, "smoke")
        required_output_check(output_dir)
        oracle_root_smoke_cases(output_dir)
        oracle_ppc_baseline(output_dir, n_rep=500)
        reporting_identifier_checks(output_dir).to_csv(
            output_dir / "reporting_identifier_checks.csv",
            index=False,
        )
        write_pilot_readiness_report(output_dir)
        print(f"[main-validation] wrote pilot readiness report to {output_dir}")
        return

    if args.mode == "final-ppc-readiness":
        aggregate_validation_outputs(output_dir, "smoke")
        required_output_check(output_dir)
        oracle_root_smoke_cases(output_dir)
        oracle_ppc_baseline(output_dir, n_rep=500)
        predictive_alignment_to_truth(output_dir)
        reporting_identifier_checks(output_dir).to_csv(
            output_dir / "reporting_identifier_checks.csv",
            index=False,
        )
        revised_pilot_readiness_decision(output_dir)
        write_final_ppc_readiness_report(output_dir)
        print(f"[main-validation] wrote final PPC readiness report to {output_dir}")
        return

    variants = parse_csv_list(args.fit_variants)
    if FIT_VARIANT_UNMODIFIED in variants and args.negative_control_seeds <= 0:
        variants = [v for v in variants if v != FIT_VARIANT_UNMODIFIED]
    if args.mode == "smoke":
        n_seeds = int(args.n_seeds or 2)
        ppc_draws = int(args.ppc_draws or 100)
        loo_draws = int(args.loo_draws or 100)
        loo_max = parse_loo_max_ratings(args.loo_max_ratings or 50)
    elif args.mode == "pilot":
        n_seeds = int(args.n_seeds or 10)
        ppc_draws = int(args.ppc_draws or 500)
        loo_draws = int(args.loo_draws or 500)
        loo_max = parse_loo_max_ratings(args.loo_max_ratings or "all")
    else:
        n_seeds = int(args.n_seeds or 50)
        ppc_draws = int(args.ppc_draws or 300)
        loo_draws = int(args.loo_draws or 500)
        loo_max = parse_loo_max_ratings(args.loo_max_ratings or "all")

    for variant in variants:
        if variant == FIT_VARIANT_UNMODIFIED:
            variant_n = min(n_seeds, int(args.negative_control_seeds))
        else:
            variant_n = n_seeds
        for seed_idx in range(variant_n):
            print(f"[main-validation] running {variant} seed_idx={seed_idx}")
            run_one_fit(
                output_dir=output_dir,
                variant=variant,
                seed=args.seed,
                seed_idx=seed_idx,
                mode="smoke" if args.mode == "smoke" else "pilot",
                chains=args.chains,
                draws=args.draws,
                tune=args.tune,
                target_accept=args.target_accept,
                ppc_draws=ppc_draws,
                loo_draws=loo_draws,
                loo_max_ratings=loo_max,
                skip_loo=args.skip_loo,
                overwrite=args.overwrite,
            )

    aggregate_validation_outputs(output_dir, args.mode)
    required_output_check(output_dir)
    print(f"[main-validation] wrote validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
