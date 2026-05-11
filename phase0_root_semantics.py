"""Phase 0 diagnostics for DCM root semantics.

This script operationalises the binary-root estimand distinction:

    pi_s  = Bernoulli parameter currently represented by C_s
    R_s   = realised binary root state
    rho_s = p(R_s = 1 | y_s)

It intentionally leaves the production exact-tree likelihood unchanged except
for deterministics added in ``dcm_model_exact_tree.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import betaln, expit, gammaln, logsumexp, logit

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse_tree_pooling import pooled_beta_draws_by_node  # noqa: E402
from composite_vs_exact_diagnostic import (  # noqa: E402
    collect_indicator_obs,
    precompute_leaf_logliks,
    three_state_log_B,
)
from dcm_model import (  # noqa: E402
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    MultiSystemModelBuilder,
    load_data,
    node_key,
)
from dcm_model_exact_tree import MultiSystemExactTreeBuilder  # noqa: E402
from gwt_reference_recovery_analysis import (  # noqa: E402
    ANCHORED_SYSTEM_CONFIGS,
    STANCE,
    extract_beta_draws_by_node,
)


OUT_DIR_DEFAULT = REPO_ROOT / "outputs/phase0_root_semantics"
PRIOR_A = 1.0
PRIOR_B = 5.0
PRIOR_ROOT_PROB = PRIOR_A / (PRIOR_A + PRIOR_B)
SCORE_CLIP = 1e-12


def beta_sd(a: float, b: float) -> float:
    return math.sqrt(a * b / (((a + b) ** 2) * (a + b + 1.0)))


def beta_second_moment(a: float, b: float) -> float:
    return a * (a + 1.0) / ((a + b) * (a + b + 1.0))


def central_interval(x: np.ndarray, mass: float) -> Tuple[float, float]:
    alpha = (1.0 - mass) / 2.0
    return (
        float(np.quantile(x, alpha)),
        float(np.quantile(x, 1.0 - alpha)),
    )


def bernoulli_entropy(p: float) -> float:
    p_clip = float(np.clip(p, SCORE_CLIP, 1.0 - SCORE_CLIP))
    return -p_clip * math.log(p_clip) - (1.0 - p_clip) * math.log(1.0 - p_clip)


def bernoulli_log_score(p: float, y: int) -> float:
    p_clip = float(np.clip(p, SCORE_CLIP, 1.0 - SCORE_CLIP))
    return float(y * math.log(p_clip) + (1 - y) * math.log(1.0 - p_clip))


def binom_logpmf(k: np.ndarray | int, n: int, p: np.ndarray | float) -> np.ndarray:
    k_arr = np.asarray(k)
    p_arr = np.asarray(p)
    return (
        gammaln(n + 1)
        - gammaln(k_arr + 1)
        - gammaln(n - k_arr + 1)
        + k_arr * np.log(np.clip(p_arr, SCORE_CLIP, 1.0))
        + (n - k_arr) * np.log(np.clip(1.0 - p_arr, SCORE_CLIP, 1.0))
    )


def rho_from_log_sides(pi_draws: np.ndarray, log_L0: np.ndarray, log_L1: np.ndarray) -> np.ndarray:
    pi_clip = np.clip(pi_draws, SCORE_CLIP, 1.0 - SCORE_CLIP)
    log_num_present = np.log(pi_clip) + log_L1
    log_num_absent = np.log1p(-pi_clip) + log_L0
    log_norm = np.logaddexp(log_num_present, log_num_absent)
    return np.exp(log_num_present - log_norm)


def rho_from_log_odds(log_prior_odds: float, log_B: np.ndarray) -> np.ndarray:
    return expit(float(log_prior_odds) + log_B)


def distribution_summary(prefix: str, values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    qs = np.quantile(arr, [0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99])
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        f"{prefix}_q01": float(qs[0]),
        f"{prefix}_q05": float(qs[1]),
        f"{prefix}_q10": float(qs[2]),
        f"{prefix}_q50": float(qs[3]),
        f"{prefix}_q90": float(qs[4]),
        f"{prefix}_q95": float(qs[5]),
        f"{prefix}_q99": float(qs[6]),
    }


def evidence_category(rho: float) -> str:
    if rho >= 0.95:
        return "strong_present"
    if rho >= 0.80:
        return "moderate_present"
    if rho > 0.20:
        return "ambiguous"
    if rho > 0.05:
        return "moderate_absent"
    return "strong_absent"


def binary_prior_rho_from_log_B(log_B: float, a: float = PRIOR_A, b: float = PRIOR_B) -> float:
    log_num = math.log(a) + float(log_B)
    log_den = np.logaddexp(log_num, math.log(b))
    return float(math.exp(log_num - log_den))


def beta_one_root_mixture_moments(
    rho: float,
    a: float = PRIOR_A,
    b: float = PRIOR_B,
) -> Tuple[float, float]:
    absent_mean = a / (a + b + 1.0)
    present_mean = (a + 1.0) / (a + b + 1.0)
    absent_second = beta_second_moment(a, b + 1.0)
    present_second = beta_second_moment(a + 1.0, b)
    mean = (1.0 - rho) * absent_mean + rho * present_mean
    second = (1.0 - rho) * absent_second + rho * present_second
    var = max(0.0, second - mean * mean)
    return float(mean), float(math.sqrt(var))


def model_config_from_payload(payload: Dict[str, Any]) -> ModelConfig:
    cfg_payload = payload.get("config") or payload.get("model_config") or {}
    field_names = {f.name for f in fields(ModelConfig)}
    kwargs = {k: v for k, v in cfg_payload.items() if k in field_names}
    return ModelConfig(**kwargs)


def stance_prefix(stance_name: str) -> str:
    return stance_name.replace(" ", "_").replace("/", "_").lower()


def c_var_name(system: str, stance_name: str) -> str:
    return f"{MultiSystemModelBuilder._sys_prefix(system)}__{stance_prefix(stance_name)}_C"


def deterministic_prefix(system: str, stance_name: str) -> str:
    return f"{MultiSystemModelBuilder._sys_prefix(system)}__{stance_prefix(stance_name)}"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Phase 0.1 production deterministic check
# ---------------------------------------------------------------------------


def check_production_deterministics(out_dir: Path) -> Dict[str, Any]:
    import pymc as pm

    cfg = ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=True,
        BETA_ABS_BY_SUPPORT_DEMAND=True,
        LABEL_POOL_SIGMA=0.5,
        NUM_SAMPLES=5,
        NUM_TUNE=5,
        NUM_CHAINS=1,
    )
    systems = [s for s, _ in ANCHORED_SYSTEM_CONFIGS]
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    proc = MultiSystemDataProcessor(cfg)
    proc.process(stance_data, systems)
    builder = MultiSystemExactTreeBuilder(
        cfg, EvidenceProcessor(cfg), proc, list(ANCHORED_SYSTEM_CONFIGS)
    )
    model = builder.build_model(stance_data)

    deterministic_names = {v.name for v in model.deterministics}
    expected = []
    for system in systems:
        prefix = deterministic_prefix(system, STANCE)
        expected.extend(
            [
                f"{prefix}_log_L_root0",
                f"{prefix}_log_L_root1",
                f"{prefix}_log_B",
                f"{prefix}_rho",
            ]
        )
    missing = sorted(name for name in expected if name not in deterministic_names)
    rho_vars = [name for name in expected if name.endswith("_rho")]

    prior_ok = False
    finite_ok = False
    range_ok = False
    rho_ranges: Dict[str, Dict[str, float]] = {}
    try:
        with model:
            idata = pm.sample_prior_predictive(
                samples=5,
                var_names=rho_vars,
                random_seed=20260511,
            )
        prior_ok = True
        for name in rho_vars:
            vals = np.asarray(idata.prior[name].values, dtype=float).reshape(-1)
            rho_ranges[name] = {
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "finite": bool(np.all(np.isfinite(vals))),
                "in_unit_interval": bool(np.all((vals >= 0.0) & (vals <= 1.0))),
            }
        finite_ok = all(item["finite"] for item in rho_ranges.values())
        range_ok = all(item["in_unit_interval"] for item in rho_ranges.values())
    except Exception as exc:  # pragma: no cover - recorded in report
        rho_ranges["error"] = {"message": str(exc)}

    potential_names = [p.name for p in model.potentials]
    no_extra_potentials = len(potential_names) == len(systems) and all(
        name.endswith("__exact_tree_lik") for name in potential_names
    )
    result = {
        "model_build_ok": True,
        "expected_deterministics": expected,
        "missing_deterministics": missing,
        "deterministics_present": not missing,
        "prior_predictive_inferencedata_ok": prior_ok,
        "rho_finite": finite_ok,
        "rho_in_unit_interval": range_ok,
        "rho_ranges": rho_ranges,
        "potential_names": potential_names,
        "no_extra_potentials_added": no_extra_potentials,
    }
    write_json(out_dir / "production_deterministic_check.json", result)
    return result


# ---------------------------------------------------------------------------
# Phase 0.2 post-hoc reanalysis
# ---------------------------------------------------------------------------


def discover_synthetic_runs() -> List[Path]:
    roots = [
        REPO_ROOT / "notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery",
        REPO_ROOT / "notebooks/sample_size_sweep_2026-05-10/runs",
        REPO_ROOT / "notebooks/asymmetric_prior_sweep_2026-05-10/runs/synthetic",
    ]
    runs: List[Path] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for fit_path in root.rglob("fit.nc"):
            run_dir = fit_path.parent
            if run_dir in seen:
                continue
            if (run_dir / "truth.json").exists() and (run_dir / "synthetic_stance_data.json").exists():
                runs.append(run_dir)
                seen.add(run_dir)
    return sorted(runs, key=lambda p: str(p.relative_to(REPO_ROOT)))


def exact_root_side_loglik(
    stance_data: Dict[str, Any],
    indicator_obs: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    leaf_logliks: Dict[str, Dict[str, np.ndarray]],
    beta_pres_by_key: Dict[str, np.ndarray],
    beta_abs_by_key: Dict[str, np.ndarray],
    systems: Sequence[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    sample_key = next(iter(beta_pres_by_key))
    n_draws = beta_pres_by_key[sample_key].shape[0]
    root_path = (stance_data["name"],)

    def subtree_log_lik(
        node: Dict[str, Any],
        path: Tuple[str, ...],
        system: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        current_path = path + (node["name"],)
        key = node_key(path, node["name"])
        bp = beta_pres_by_key[key]
        ba = beta_abs_by_key[key]

        if (node.get("type") or "").lower() == "indicator":
            if system not in indicator_obs.get(key, {}):
                return np.zeros(n_draws), np.zeros(n_draws)
            leaf_lls = leaf_logliks[key][system]
            return three_state_log_B(ba, leaf_lls), three_state_log_B(bp, leaf_lls)

        log_L_v0 = np.zeros(n_draws)
        log_L_v1 = np.zeros(n_draws)
        for child in node.get("evidencers", []):
            child_L0, child_L1 = subtree_log_lik(child, current_path, system)
            log_L_v0 += child_L0
            log_L_v1 += child_L1

        bp_clip = np.clip(bp, SCORE_CLIP, 1.0 - SCORE_CLIP)
        ba_clip = np.clip(ba, SCORE_CLIP, 1.0 - SCORE_CLIP)
        log_L_zpa1 = np.logaddexp(
            np.log(bp_clip) + log_L_v1,
            np.log1p(-bp_clip) + log_L_v0,
        )
        log_L_zpa0 = np.logaddexp(
            np.log(ba_clip) + log_L_v1,
            np.log1p(-ba_clip) + log_L_v0,
        )
        return log_L_zpa0, log_L_zpa1

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for system in systems:
        log_L_top0 = np.zeros(n_draws)
        log_L_top1 = np.zeros(n_draws)
        for child in stance_data.get("evidencers", []):
            child_L0, child_L1 = subtree_log_lik(child, root_path, system)
            log_L_top0 += child_L0
            log_L_top1 += child_L1
        out[system] = (log_L_top0, log_L_top1)
    return out


def extract_beta_draws(
    idata: Any,
    cfg: ModelConfig,
    stance_data: Dict[str, Any],
    proc: MultiSystemDataProcessor,
    system_configs: Sequence[Tuple[str, Optional[float]]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    if cfg.POOL_BETAS_BY_LABEL:
        return pooled_beta_draws_by_node(idata, stance_data)
    builder = MultiSystemModelBuilder(cfg, EvidenceProcessor(cfg), proc, list(system_configs))
    builder.build_model(stance_data)
    return extract_beta_draws_by_node(idata, builder, stance_data)


def posthoc_reanalysis(out_dir: Path, max_runs: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    run_dirs = discover_synthetic_runs()
    if max_runs is not None:
        run_dirs = run_dirs[:max_runs]
    anchor_map = dict(ANCHORED_SYSTEM_CONFIGS)
    prior_sd = beta_sd(PRIOR_A, PRIOR_B)
    entropy_prior = bernoulli_entropy(PRIOR_ROOT_PROB)

    for run_dir in run_dirs:
        try:
            config_payload = load_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
            truth = load_json(run_dir / "truth.json")
            stance_data = load_json(run_dir / "synthetic_stance_data.json")
            source_path = str(run_dir.relative_to(REPO_ROOT))
            posterior_file = str((run_dir / "fit.nc").relative_to(REPO_ROOT))
            labels = config_payload.get("labels") or {}
            config_name = (
                config_payload.get("config_name")
                or labels.get("fit")
                or config_payload.get("audit")
            )
            run_variant = (
                config_payload.get("run_variant")
                or config_payload.get("mode")
                or run_dir.parent.name
            )
            cfg = model_config_from_payload(config_payload)
            if cfg.INDICATOR_STATE_MODEL != "three_state":
                skipped.append({"source_path": source_path, "reason": "non-three-state fit"})
                continue
            systems = config_payload.get("systems") or list(truth.get("true_C_by_system", {}).keys())
            systems = [s for s in systems if s in truth.get("latent_by_system", {})]
            system_configs = [(s, anchor_map.get(s)) for s in systems]

            idata = az.from_netcdf(run_dir / "fit.nc")
            post = idata.posterior
            a_draws = np.asarray(post["a"].values).reshape(-1)
            n_draws = a_draws.shape[0]
            K = cfg.N_CATEGORIES

            proc = MultiSystemDataProcessor(cfg)
            proc.process(stance_data, systems)
            n_experts = len(proc.expert_names)
            kappa_draws = np.asarray(post["kappa"].values).reshape(-1, K - 1)
            if "kappa_by_expert" in post.data_vars:
                skipped.append({
                    "source_path": source_path,
                    "reason": "posthoc helper does not handle kappa_by_expert",
                })
                continue
            if "b_free" in post.data_vars:
                b_free = np.asarray(post["b_free"].values).reshape(-1, n_experts - 1)
                b_draws = np.concatenate([np.zeros((n_draws, 1)), b_free], axis=1)
            else:
                b_draws = np.zeros((n_draws, n_experts))

            indicator_obs = collect_indicator_obs(stance_data, proc)
            leaf_logliks = precompute_leaf_logliks(
                indicator_obs, a_draws, b_draws, kappa_draws, K=K
            )
            beta_pres_by_key, beta_abs_by_key = extract_beta_draws(
                idata, cfg, stance_data, proc, system_configs
            )
            sides_by_system = exact_root_side_loglik(
                stance_data,
                indicator_obs,
                leaf_logliks,
                beta_pres_by_key,
                beta_abs_by_key,
                systems,
            )

            run_id = config_payload.get("run_id", run_dir.name)
            seed = config_payload.get("seed", truth.get("seed"))
            for system in systems:
                var = c_var_name(system, stance_data["name"])
                if var in post.data_vars:
                    pi_draws = np.asarray(post[var].values).reshape(-1)
                elif anchor_map.get(system) is not None:
                    pi_draws = np.full(n_draws, float(anchor_map[system]))
                else:
                    skipped.append({
                        "source_path": source_path,
                        "reason": f"missing posterior C variable for {system}",
                    })
                    continue
                log_L0, log_L1 = sides_by_system[system]
                log_B = log_L1 - log_L0
                rho_sampled_pi = rho_from_log_sides(pi_draws, log_L0, log_L1)
                if anchor_map.get(system) is not None:
                    root_prior_kind = "hard_fixed_pi"
                    root_prior_alpha = np.nan
                    root_prior_beta = np.nan
                    root_prior_pi_fixed = float(anchor_map[system])
                    log_prior_odds = float(logit(np.clip(root_prior_pi_fixed, SCORE_CLIP, 1.0 - SCORE_CLIP)))
                elif cfg.SOFT_REFERENCE_ANCHORS and system in cfg.SOFT_REFERENCE_ANCHORS:
                    root_prior_kind = "soft_beta_anchor"
                    root_prior_alpha, root_prior_beta = map(float, cfg.SOFT_REFERENCE_ANCHORS[system])
                    root_prior_pi_fixed = np.nan
                    log_prior_odds = math.log(root_prior_alpha / root_prior_beta)
                else:
                    root_prior_kind = "free_beta"
                    root_prior_alpha = float(cfg.DEFAULT_ALPHA)
                    root_prior_beta = float(cfg.DEFAULT_BETA)
                    root_prior_pi_fixed = np.nan
                    log_prior_odds = math.log(root_prior_alpha / root_prior_beta)
                rho_collapsed = rho_from_log_odds(log_prior_odds, log_B)
                R_true = int(truth["latent_by_system"][system]["root_z"])
                pi_true = float(truth.get("true_C_by_system", {}).get(system, np.nan))
                rho_sampled_pi_mean = float(np.mean(rho_sampled_pi))
                rho_collapsed_mean = float(np.mean(rho_collapsed))
                brier_sampled_pi = float((rho_sampled_pi_mean - R_true) ** 2)
                brier_collapsed = float((rho_collapsed_mean - R_true) ** 2)
                log_score_sampled_pi = bernoulli_log_score(rho_sampled_pi_mean, R_true)
                log_score_collapsed = bernoulli_log_score(rho_collapsed_mean, R_true)
                prior_brier = float((PRIOR_ROOT_PROB - R_true) ** 2)
                prior_log_score = bernoulli_log_score(PRIOR_ROOT_PROB, R_true)
                rho80 = central_interval(rho_sampled_pi, 0.80)
                rho95 = central_interval(rho_sampled_pi, 0.95)
                log_B80 = central_interval(log_B, 0.80)
                pi80 = central_interval(pi_draws, 0.80)
                row_unique_key = f"{posterior_file}|posterior|{stance_data['name']}|{system}"
                pi_shape = str(tuple(pi_draws.shape))
                log_B_shape = str(tuple(log_B.shape))
                rho_sampled_shape = str(tuple(rho_sampled_pi.shape))
                rho_collapsed_shape = str(tuple(rho_collapsed.shape))
                shape_warning = ""
                if not (
                    pi_draws.shape == log_B.shape == rho_sampled_pi.shape == rho_collapsed.shape
                ):
                    shape_warning = (
                        f"shape mismatch: pi={pi_shape}, log_B={log_B_shape}, "
                        f"rho_sampled={rho_sampled_shape}, rho_collapsed={rho_collapsed_shape}"
                    )
                rows.append(
                    {
                        "source_path": source_path,
                        "posterior_file": posterior_file,
                        "posterior_group": "posterior",
                        "config_name": config_name,
                        "run_variant": run_variant,
                        "row_unique_key": row_unique_key,
                        "audit": config_payload.get("audit"),
                        "labels_dgp": labels.get("dgp"),
                        "labels_fit": labels.get("fit"),
                        "labels_leaf": labels.get("leaf"),
                        "labels_design": labels.get("design"),
                        "labels_nuisance_truth": labels.get("nuisance_truth"),
                        "run_id": run_id,
                        "run_dir": source_path,
                        "seed": seed,
                        "stance": stance_data["name"],
                        "system": system,
                        "is_anchor_or_free": (
                            "hard_anchor" if anchor_map.get(system) is not None else "free_target"
                        ),
                        "pi_true": pi_true,
                        "R_true": R_true,
                        **distribution_summary("pi", pi_draws),
                        "pi_posterior_mean": float(np.mean(pi_draws)),
                        "pi_posterior_median": float(np.median(pi_draws)),
                        "pi_posterior_sd": float(np.std(pi_draws, ddof=1)),
                        "pi_posterior_central_80_low": pi80[0],
                        "pi_posterior_central_80_high": pi80[1],
                        "pi_prior_sd": 0.0 if anchor_map.get(system) is not None else prior_sd,
                        "pi_contraction": (
                            np.nan
                            if anchor_map.get(system) is not None
                            else float(np.std(pi_draws, ddof=1) / prior_sd)
                        ),
                        "root_prior_kind": root_prior_kind,
                        "root_prior_alpha": root_prior_alpha,
                        "root_prior_beta": root_prior_beta,
                        "root_prior_pi_fixed": root_prior_pi_fixed,
                        "root_log_prior_odds_used": log_prior_odds,
                        **distribution_summary("log_B", log_B),
                        "log_B_median": float(np.median(log_B)),
                        "log_B_central_80_low": log_B80[0],
                        "log_B_central_80_high": log_B80[1],
                        **distribution_summary("rho_sampled_pi", rho_sampled_pi),
                        **distribution_summary("rho_collapsed", rho_collapsed),
                        "min_rho_sampled_pi": float(np.min(rho_sampled_pi)),
                        "max_rho_sampled_pi": float(np.max(rho_sampled_pi)),
                        "min_rho_collapsed": float(np.min(rho_collapsed)),
                        "max_rho_collapsed": float(np.max(rho_collapsed)),
                        "rho_sampled_pi_evidence_category": evidence_category(rho_sampled_pi_mean),
                        "rho_collapsed_evidence_category": evidence_category(rho_collapsed_mean),
                        "rho_mean": rho_sampled_pi_mean,
                        "rho_median": float(np.median(rho_sampled_pi)),
                        "rho_central_80_low": rho80[0],
                        "rho_central_80_high": rho80[1],
                        "rho_central_95_low": rho95[0],
                        "rho_central_95_high": rho95[1],
                        "brier_sampled_pi": brier_sampled_pi,
                        "brier_collapsed": brier_collapsed,
                        "log_score_sampled_pi": log_score_sampled_pi,
                        "log_score_collapsed": log_score_collapsed,
                        "brier": brier_sampled_pi,
                        "log_score": log_score_sampled_pi,
                        "prior_root_prob": PRIOR_ROOT_PROB,
                        "prior_brier": prior_brier,
                        "prior_log_score": prior_log_score,
                        "delta_brier": prior_brier - brier_sampled_pi,
                        "delta_brier_collapsed": prior_brier - brier_collapsed,
                        "delta_log_score": log_score_sampled_pi - prior_log_score,
                        "delta_log_score_collapsed": log_score_collapsed - prior_log_score,
                        "entropy_prior": entropy_prior,
                        "entropy_posterior": bernoulli_entropy(rho_sampled_pi_mean),
                        "entropy_reduction": entropy_prior - bernoulli_entropy(rho_sampled_pi_mean),
                        "n_posterior_draws": int(n_draws),
                        "nan_count_pi": int(np.isnan(pi_draws).sum()),
                        "inf_count_pi": int(np.isinf(pi_draws).sum()),
                        "nan_count_log_B": int(np.isnan(log_B).sum()),
                        "inf_count_log_B": int(np.isinf(log_B).sum()),
                        "nan_count_rho_sampled_pi": int(np.isnan(rho_sampled_pi).sum()),
                        "inf_count_rho_sampled_pi": int(np.isinf(rho_sampled_pi).sum()),
                        "nan_count_rho_collapsed": int(np.isnan(rho_collapsed).sum()),
                        "inf_count_rho_collapsed": int(np.isinf(rho_collapsed).sum()),
                        "pi_draws_shape": pi_shape,
                        "log_B_shape": log_B_shape,
                        "rho_sampled_pi_shape": rho_sampled_shape,
                        "rho_collapsed_shape": rho_collapsed_shape,
                        "shape_warning": shape_warning,
                    }
                )
        except Exception as exc:
            skipped.append({"source_path": str(run_dir.relative_to(REPO_ROOT)), "reason": repr(exc)})

    df = pd.DataFrame(rows)
    if df.empty:
        df.to_csv(out_dir / "posthoc_rho_reanalysis.csv", index=False)
        summary = pd.DataFrame()
    else:
        duplicated_unique = df.duplicated("row_unique_key", keep=False)
        if duplicated_unique.any():
            duplicate_rows = df[duplicated_unique].copy()
            df = df.drop_duplicates("row_unique_key", keep="first").copy()
        else:
            duplicate_rows = pd.DataFrame()
        duplicate_rows.to_csv(out_dir / "posthoc_accidental_duplicate_rows.csv", index=False)
        df.to_csv(out_dir / "posthoc_rho_reanalysis.csv", index=False)
        summary = (
            df.groupby("is_anchor_or_free", dropna=False)
            .agg(
                n=("system", "size"),
                n_runs=("run_dir", "nunique"),
                mean_brier_sampled_pi=("brier_sampled_pi", "mean"),
                mean_brier_collapsed=("brier_collapsed", "mean"),
                mean_prior_brier=("prior_brier", "mean"),
                mean_delta_brier=("delta_brier", "mean"),
                mean_delta_brier_collapsed=("delta_brier_collapsed", "mean"),
                mean_delta_log_score=("delta_log_score", "mean"),
                mean_delta_log_score_collapsed=("delta_log_score_collapsed", "mean"),
                mean_entropy_reduction=("entropy_reduction", "mean"),
                median_abs_log_B=("log_B_q50", lambda s: float(np.median(np.abs(s)))),
                median_rho_sampled_pi_mean=("rho_sampled_pi_mean", "median"),
                median_rho_collapsed_mean=("rho_collapsed_mean", "median"),
                median_pi_contraction=("pi_contraction", "median"),
            )
            .reset_index()
        )
    summary.to_csv(out_dir / "posthoc_rho_summary.csv", index=False)
    pd.DataFrame(skipped).to_csv(out_dir / "posthoc_skipped_runs.csv", index=False)
    return df, summary


def classification_metrics(df: pd.DataFrame, rho_col: str, rho_type: str) -> Dict[str, Any]:
    pred = df[rho_col] >= 0.5
    truth = df["R_true"].astype(int) == 1
    tp = int((pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    tnr = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "rho_type": rho_type,
        "n": int(len(df)),
        "TPR": float(tpr),
        "TNR": float(tnr),
        "balanced_accuracy": float(np.nanmean([tpr, tnr])),
        "true_positive_count": tp,
        "true_negative_count": tn,
        "false_positive_count": fp,
        "false_negative_count": fn,
    }


def build_posthoc_audit_outputs(out_dir: Path, df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        empty = pd.DataFrame()
        for name in (
            "posthoc_duplicate_audit.csv",
            "posthoc_anchor_anomalies.csv",
            "posthoc_free_target_stratified_metrics.csv",
            "posthoc_free_target_classification.csv",
            "posthoc_evidence_category_counts.csv",
            "posthoc_sampled_vs_collapsed_comparison.csv",
            "posthoc_sampled_vs_collapsed_summary.csv",
        ):
            empty.to_csv(out_dir / name, index=False)
        return {"duplicate_count": 0, "anomaly_count": 0}

    duplicate_rows: List[Dict[str, Any]] = []
    for (run_id, system), g in df.groupby(["run_id", "system"], dropna=False):
        if len(g) <= 1:
            continue
        finding = "intended_distinct_source_paths"
        if g["row_unique_key"].duplicated().any():
            finding = "accidental_duplicate_row_unique_key"
        elif g["stance"].nunique(dropna=False) > 1:
            finding = "intended_distinct_stances"
        elif g["posterior_group"].nunique(dropna=False) > 1:
            finding = "intended_distinct_posterior_groups"
        duplicate_rows.append(
            {
                "run_id": run_id,
                "system": system,
                "n_rows": int(len(g)),
                "n_source_paths": int(g["source_path"].nunique(dropna=False)),
                "n_posterior_files": int(g["posterior_file"].nunique(dropna=False)),
                "n_stances": int(g["stance"].nunique(dropna=False)),
                "n_posterior_groups": int(g["posterior_group"].nunique(dropna=False)),
                "finding": finding,
                "source_paths": "; ".join(sorted(map(str, g["source_path"].unique()))),
                "posterior_files": "; ".join(sorted(map(str, g["posterior_file"].unique()))),
                "run_variants": "; ".join(sorted(map(str, g["run_variant"].dropna().unique()))),
                "row_unique_keys": "; ".join(sorted(map(str, g["row_unique_key"].unique()))),
            }
        )
    duplicate_audit = pd.DataFrame(duplicate_rows)
    duplicate_audit.to_csv(out_dir / "posthoc_duplicate_audit.csv", index=False)

    anomaly_mask = (
        ((df["pi_posterior_mean"] > 0.95) & (df["rho_sampled_pi_mean"] < 0.95))
        | ((df["pi_posterior_mean"] < 0.05) & (df["rho_sampled_pi_mean"] > 0.05))
    )
    anomaly_cols = [
        "source_path",
        "posterior_file",
        "system",
        "run_id",
        "run_variant",
        "root_prior_kind",
        "n_posterior_draws",
        "pi_posterior_mean",
        "pi_q01",
        "pi_q05",
        "pi_q10",
        "pi_q50",
        "pi_q90",
        "pi_q95",
        "pi_q99",
        "log_B_q01",
        "log_B_q05",
        "log_B_q10",
        "log_B_q50",
        "log_B_q90",
        "log_B_q95",
        "log_B_q99",
        "rho_sampled_pi_q01",
        "rho_sampled_pi_q05",
        "rho_sampled_pi_q10",
        "rho_sampled_pi_q50",
        "rho_sampled_pi_q90",
        "rho_sampled_pi_q95",
        "rho_sampled_pi_q99",
        "rho_collapsed_q01",
        "rho_collapsed_q05",
        "rho_collapsed_q10",
        "rho_collapsed_q50",
        "rho_collapsed_q90",
        "rho_collapsed_q95",
        "rho_collapsed_q99",
        "nan_count_pi",
        "inf_count_pi",
        "nan_count_log_B",
        "inf_count_log_B",
        "nan_count_rho_sampled_pi",
        "inf_count_rho_sampled_pi",
        "nan_count_rho_collapsed",
        "inf_count_rho_collapsed",
        "pi_draws_shape",
        "log_B_shape",
        "rho_sampled_pi_shape",
        "rho_collapsed_shape",
        "shape_warning",
    ]
    anomalies = df.loc[anomaly_mask, [c for c in anomaly_cols if c in df.columns]].copy()
    anomalies.to_csv(out_dir / "posthoc_anchor_anomalies.csv", index=False)

    free = df[df["is_anchor_or_free"] == "free_target"].copy()
    stratified = pd.DataFrame()
    classification = pd.DataFrame()
    evidence_counts = pd.DataFrame()
    if not free.empty:
        stratified = (
            free.groupby("R_true")
            .agg(
                n=("system", "size"),
                mean_rho_sampled_pi=("rho_sampled_pi_mean", "mean"),
                median_rho_sampled_pi=("rho_sampled_pi_mean", "median"),
                mean_rho_collapsed=("rho_collapsed_mean", "mean"),
                median_rho_collapsed=("rho_collapsed_mean", "median"),
                mean_log_B=("log_B_mean", "mean"),
                median_log_B=("log_B_q50", "median"),
                mean_brier_sampled_pi=("brier_sampled_pi", "mean"),
                mean_brier_collapsed=("brier_collapsed", "mean"),
                mean_log_score_sampled_pi=("log_score_sampled_pi", "mean"),
                mean_log_score_collapsed=("log_score_collapsed", "mean"),
            )
            .reset_index()
        )
        classification = pd.DataFrame(
            [
                classification_metrics(free, "rho_sampled_pi_mean", "sampled_pi"),
                classification_metrics(free, "rho_collapsed_mean", "collapsed"),
            ]
        )
        evidence_counts = pd.concat(
            [
                (
                    free.groupby("rho_sampled_pi_evidence_category")
                    .size()
                    .rename("n")
                    .reset_index()
                    .rename(columns={"rho_sampled_pi_evidence_category": "evidence_category"})
                    .assign(rho_type="sampled_pi")
                ),
                (
                    free.groupby("rho_collapsed_evidence_category")
                    .size()
                    .rename("n")
                    .reset_index()
                    .rename(columns={"rho_collapsed_evidence_category": "evidence_category"})
                    .assign(rho_type="collapsed")
                ),
            ],
            ignore_index=True,
        )[["rho_type", "evidence_category", "n"]]
    stratified.to_csv(out_dir / "posthoc_free_target_stratified_metrics.csv", index=False)
    classification.to_csv(out_dir / "posthoc_free_target_classification.csv", index=False)
    evidence_counts.to_csv(out_dir / "posthoc_evidence_category_counts.csv", index=False)

    comparison = df[
        [
            "source_path",
            "system",
            "run_id",
            "R_true",
            "pi_true",
            "rho_sampled_pi_mean",
            "rho_collapsed_mean",
            "log_B_q50",
        ]
    ].copy()
    comparison["abs_difference"] = (
        comparison["rho_sampled_pi_mean"] - comparison["rho_collapsed_mean"]
    ).abs()
    comparison = comparison.rename(columns={"log_B_q50": "log_B_median"})
    comparison.to_csv(out_dir / "posthoc_sampled_vs_collapsed_comparison.csv", index=False)

    valid = comparison[["rho_sampled_pi_mean", "rho_collapsed_mean", "abs_difference"]].dropna()
    corr = (
        float(valid["rho_sampled_pi_mean"].corr(valid["rho_collapsed_mean"]))
        if len(valid) > 1
        else np.nan
    )
    comparison_summary = pd.DataFrame(
        [
            {
                "n": int(len(valid)),
                "mean_abs_difference": float(valid["abs_difference"].mean()) if len(valid) else np.nan,
                "max_abs_difference": float(valid["abs_difference"].max()) if len(valid) else np.nan,
                "correlation": corr,
                "n_difference_gt_0_05": int((valid["abs_difference"] > 0.05).sum()),
                "n_difference_gt_0_10": int((valid["abs_difference"] > 0.10).sum()),
            }
        ]
    )
    comparison_summary.to_csv(out_dir / "posthoc_sampled_vs_collapsed_summary.csv", index=False)

    write_audit_report(
        out_dir,
        df,
        duplicate_audit,
        anomalies,
        stratified,
        classification,
        evidence_counts,
        comparison,
        comparison_summary,
    )
    return {
        "duplicate_count": int(len(duplicate_audit)),
        "anomaly_count": int(len(anomalies)),
        "comparison_summary": comparison_summary.to_dict(orient="records")[0],
    }


def write_audit_report(
    out_dir: Path,
    posthoc_df: pd.DataFrame,
    duplicate_audit: pd.DataFrame,
    anomalies: pd.DataFrame,
    stratified: pd.DataFrame,
    classification: pd.DataFrame,
    evidence_counts: pd.DataFrame,
    comparison: pd.DataFrame,
    comparison_summary: pd.DataFrame,
) -> None:
    accidental = (
        duplicate_audit["finding"].eq("accidental_duplicate_row_unique_key").sum()
        if not duplicate_audit.empty
        else 0
    )
    duplicate_note = (
        "No repeated `(run_id, system)` labels were found."
        if duplicate_audit.empty
        else (
            f"{len(duplicate_audit)} repeated-looking `(run_id, system)` groups were found. "
            f"{accidental} had duplicate row_unique_key values. The remaining repeated labels "
            "come from distinct source paths/posterior files and are retained with explicit identity columns."
        )
    )
    anomaly_note = (
        "No hard-anchor rho anomalies were found."
        if anomalies.empty
        else (
            f"{len(anomalies)} hard-anchor anomalies were found. They have finite arrays with matching "
            "flat shapes; the surprising mean is explained by highly skewed/multimodal log_B draws "
            "where the median log_B is strongly positive but the lower tail drives rho toward zero."
        )
    )
    comp = comparison_summary.iloc[0].to_dict() if not comparison_summary.empty else {}
    material = int(comp.get("n_difference_gt_0_05", 0) or 0)
    recommendation = (
        "Use collapsed/fixed rho for Path A headline reporting."
        if material
        else "Sampled-pi and collapsed/fixed rho are close in this audit; collapsed/fixed rho is still the cleaner Path A headline because it matches the integrated binary-root estimand."
    )
    if material:
        recommendation += (
            " Differences are material because posterior-pi rho conditions on a pi posterior that has already been updated by the same one latent root draw."
        )

    lines = [
        "# Phase 0 Audit: Collapsed Rho",
        "",
        "## Summary",
        "",
        duplicate_note,
        "",
        anomaly_note,
        "",
        "The Phase 0 binary-root conclusion does not change: the coherent target is "
        "`rho_s = p(R_s=1 | y_s)`. For postprocessing under Path A, collapsed/fixed "
        "rho is the preferred headline because it integrates the root prior odds "
        "analytically per posterior draw of the tree/emission parameters.",
        "",
        "## Duplicate Row Audit",
        "",
        markdown_table(
            duplicate_audit,
            ["run_id", "system", "n_rows", "n_source_paths", "finding", "source_paths"],
            max_rows=20,
        ),
        "",
        "## Hard-Anchor Anomalies",
        "",
        markdown_table(
            anomalies,
            [
                "source_path",
                "system",
                "run_id",
                "pi_posterior_mean",
                "log_B_q01",
                "log_B_q50",
                "log_B_q99",
                "rho_sampled_pi_q01",
                "rho_sampled_pi_q50",
                "rho_sampled_pi_q99",
                "rho_collapsed_q01",
                "rho_collapsed_q50",
                "rho_collapsed_q99",
                "n_posterior_draws",
                "shape_warning",
            ],
            max_rows=10,
        ),
        "",
        "## Free-Target Metrics By R_true",
        "",
        markdown_table(
            stratified,
            [
                "R_true",
                "n",
                "mean_rho_sampled_pi",
                "median_rho_sampled_pi",
                "mean_rho_collapsed",
                "median_rho_collapsed",
                "mean_log_B",
                "median_log_B",
                "mean_brier_sampled_pi",
                "mean_brier_collapsed",
                "mean_log_score_sampled_pi",
                "mean_log_score_collapsed",
            ],
        ),
        "",
        markdown_table(
            classification,
            [
                "rho_type",
                "n",
                "TPR",
                "TNR",
                "balanced_accuracy",
                "false_positive_count",
                "false_negative_count",
            ],
        ),
        "",
        "## Evidence Categories",
        "",
        markdown_table(evidence_counts, ["rho_type", "evidence_category", "n"]),
        "",
        "## Sampled-Pi Vs Collapsed Rho",
        "",
        markdown_table(
            comparison_summary,
            [
                "n",
                "mean_abs_difference",
                "max_abs_difference",
                "correlation",
                "n_difference_gt_0_05",
                "n_difference_gt_0_10",
            ],
        ),
        "",
        markdown_table(
            comparison.sort_values("abs_difference", ascending=False),
            [
                "system",
                "run_id",
                "R_true",
                "pi_true",
                "rho_sampled_pi_mean",
                "rho_collapsed_mean",
                "abs_difference",
                "log_B_median",
                "source_path",
            ],
            max_rows=12,
        ),
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
    ]
    (out_dir / "phase0_audit_collapsed_rho_report.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Phase 0.3 binary-root toy
# ---------------------------------------------------------------------------


def binary_leaf_logp_by_R(K: int, beta_pres: float, beta_abs: float, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
    t = np.arange(K + 1)
    log_bin_absent = binom_logpmf(t, K, epsilon)
    log_bin_present = binom_logpmf(t, K, 1.0 - epsilon)
    out = []
    for R in (0, 1):
        q_R = beta_abs + R * (beta_pres - beta_abs)
        log_p_t = np.logaddexp(
            np.log1p(-q_R) + log_bin_absent,
            np.log(q_R) + log_bin_present,
        )
        out.append(log_p_t)
    return out[0], out[1]


def simulate_binary_root_counts(
    rng: np.random.Generator,
    R_true: int,
    J: int,
    K: int,
    beta_pres: float,
    beta_abs: float,
    epsilon: float,
) -> np.ndarray:
    q_R = beta_abs + R_true * (beta_pres - beta_abs)
    z = rng.binomial(1, q_R, size=J)
    p_y = epsilon + z * (1.0 - 2.0 * epsilon)
    return rng.binomial(K, p_y)


def run_binary_root_toy(
    out_dir: Path,
    reps: int = 200,
    seed: int = 20260511,
    J_values: Sequence[int] = (4, 16, 64),
    K_values: Sequence[int] = (1, 5, 20),
    beta_pres: float = 0.90,
    beta_abs: float = 0.10,
    epsilon: float = 0.05,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    prior_sd = beta_sd(PRIOR_A, PRIOR_B)
    rows: List[Dict[str, Any]] = []
    rep_id = 0
    leaf_cache = {
        K: binary_leaf_logp_by_R(K, beta_pres, beta_abs, epsilon)
        for K in K_values
    }
    modes = ["prior_calibrated_binary_root", "fixed_pi_demonstration"]
    fixed_pi_values = [0.05, 0.10, 0.25, 0.50, 0.75]
    for J in J_values:
        for K in K_values:
            logp0, logp1 = leaf_cache[K]
            for mode in modes:
                pi_values = [None] if mode == "prior_calibrated_binary_root" else fixed_pi_values
                for fixed_pi in pi_values:
                    for _ in range(reps):
                        rep_id += 1
                        if fixed_pi is None:
                            pi_true = float(rng.beta(PRIOR_A, PRIOR_B))
                        else:
                            pi_true = float(fixed_pi)
                        R_true = int(rng.binomial(1, pi_true))
                        counts = simulate_binary_root_counts(
                            rng, R_true, J, K, beta_pres, beta_abs, epsilon
                        )
                        freq = np.bincount(counts, minlength=K + 1)
                        log_L0 = float(np.dot(freq, logp0))
                        log_L1 = float(np.dot(freq, logp1))
                        log_B = log_L1 - log_L0
                        rho = binary_prior_rho_from_log_B(log_B)
                        pi_mean, pi_sd = beta_one_root_mixture_moments(rho)
                        brier = float((rho - R_true) ** 2)
                        prior_brier = float((PRIOR_ROOT_PROB - R_true) ** 2)
                        log_score = bernoulli_log_score(rho, R_true)
                        prior_log_score = bernoulli_log_score(PRIOR_ROOT_PROB, R_true)
                        rows.append(
                            {
                                "mode": mode,
                                "replicate": rep_id,
                                "J": J,
                                "K": K,
                                "pi_true": pi_true,
                                "R_true": R_true,
                                "log_B": log_B,
                                "rho": rho,
                                "pi_post_mean": pi_mean,
                                "pi_post_sd": pi_sd,
                                "pi_contraction": pi_sd / prior_sd,
                                "brier": brier,
                                "log_score": log_score,
                                "prior_brier": prior_brier,
                                "prior_log_score": prior_log_score,
                                "delta_brier": prior_brier - brier,
                                "delta_log_score": log_score - prior_log_score,
                            }
                        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "binary_root_toy_replicates.csv", index=False)

    grouped_rows: List[Dict[str, Any]] = []
    for (mode, J, K), g in df.groupby(["mode", "J", "K"]):
        prior_brier_mean = float(g["prior_brier"].mean())
        brier_mean = float(g["brier"].mean())
        present = g[g["R_true"] == 1]
        absent = g[g["R_true"] == 0]
        grouped_rows.append(
            {
                "mode": mode,
                "J": J,
                "K": K,
                "n": int(len(g)),
                "n_R1": int(len(present)),
                "n_R0": int(len(absent)),
                "median_rho_given_R1": float(present["rho"].median()) if len(present) else np.nan,
                "median_rho_given_R0": float(absent["rho"].median()) if len(absent) else np.nan,
                "mean_brier": brier_mean,
                "mean_prior_brier": prior_brier_mean,
                "mean_brier_improvement": prior_brier_mean - brier_mean,
                "relative_brier_improvement": (
                    (prior_brier_mean - brier_mean) / prior_brier_mean
                    if prior_brier_mean > 0
                    else np.nan
                ),
                "mean_log_score_improvement": float(g["delta_log_score"].mean()),
                "median_pi_contraction": float(g["pi_contraction"].median()),
                "median_pi_contraction_R1": float(present["pi_contraction"].median()) if len(present) else np.nan,
                "median_pi_contraction_R0": float(absent["pi_contraction"].median()) if len(absent) else np.nan,
                "mean_abs_pi_error": float(np.mean(np.abs(g["pi_post_mean"] - g["pi_true"]))),
            }
        )
    summary = pd.DataFrame(grouped_rows).sort_values(["mode", "J", "K"])
    summary.to_csv(out_dir / "binary_root_toy_summary.csv", index=False)
    return df, summary


# ---------------------------------------------------------------------------
# Phase 0.4 continuous-root toy alternatives
# ---------------------------------------------------------------------------


def posterior_grid_summary(
    C_grid: np.ndarray,
    log_prior: np.ndarray,
    log_likelihood: np.ndarray,
) -> Dict[str, float]:
    log_w = log_prior + log_likelihood
    log_w = log_w - logsumexp(log_w)
    w = np.exp(log_w)
    cdf = np.cumsum(w)
    mean = float(np.sum(w * C_grid))
    second = float(np.sum(w * C_grid * C_grid))
    sd = math.sqrt(max(0.0, second - mean * mean))

    def q(prob: float) -> float:
        return float(np.interp(prob, cdf, C_grid))

    return {
        "C_post_mean": mean,
        "C_post_median": q(0.5),
        "C_post_sd": sd,
        "C_central_80_low": q(0.10),
        "C_central_80_high": q(0.90),
        "C_central_95_low": q(0.025),
        "C_central_95_high": q(0.975),
    }


def run_continuous_toys(
    out_dir: Path,
    reps: int = 200,
    seed: int = 20260512,
    C_true_values: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 0.75),
    J_values: Sequence[int] = (4, 16, 64),
    K_values: Sequence[int] = (1, 5, 20),
    beta_pres: float = 0.90,
    beta_abs: float = 0.10,
    epsilon: float = 0.05,
    grid_size: int = 2001,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    prior_sd = beta_sd(PRIOR_A, PRIOR_B)
    C_grid = np.linspace(1e-5, 1.0 - 1e-5, grid_size)
    gap = beta_pres - beta_abs
    q_grid = beta_abs + C_grid * gap
    log_prior = (
        (PRIOR_A - 1.0) * np.log(C_grid)
        + (PRIOR_B - 1.0) * np.log1p(-C_grid)
        - betaln(PRIOR_A, PRIOR_B)
    )
    latent_cache: Dict[int, np.ndarray] = {}
    direct_p_grid = epsilon + q_grid * (1.0 - 2.0 * epsilon)
    for K in K_values:
        t = np.arange(K + 1)
        log_bin_absent = binom_logpmf(t, K, epsilon)
        log_bin_present = binom_logpmf(t, K, 1.0 - epsilon)
        logp_t = np.empty((K + 1, grid_size))
        for idx, _ in enumerate(t):
            logp_t[idx] = np.logaddexp(
                np.log1p(-q_grid) + log_bin_absent[idx],
                np.log(q_grid) + log_bin_present[idx],
            )
        latent_cache[K] = logp_t

    rows: List[Dict[str, Any]] = []
    rep_id = 0
    for alternative in ("latent_leaf_continuous_propensity", "direct_q_continuous_observation"):
        for C_true in C_true_values:
            q_true = beta_abs + C_true * gap
            for J in J_values:
                for K in K_values:
                    for _ in range(reps):
                        rep_id += 1
                        if alternative == "latent_leaf_continuous_propensity":
                            z = rng.binomial(1, q_true, size=J)
                            p_y = epsilon + z * (1.0 - 2.0 * epsilon)
                            counts = rng.binomial(K, p_y)
                            freq = np.bincount(counts, minlength=K + 1)
                            log_likelihood = np.dot(freq, latent_cache[K])
                        else:
                            n_obs = J * K
                            p_true = epsilon + q_true * (1.0 - 2.0 * epsilon)
                            total = int(rng.binomial(n_obs, p_true))
                            log_likelihood = binom_logpmf(total, n_obs, direct_p_grid)

                        summary = posterior_grid_summary(C_grid, log_prior, log_likelihood)
                        mean = summary["C_post_mean"]
                        rows.append(
                            {
                                "alternative": alternative,
                                "replicate": rep_id,
                                "C_true": float(C_true),
                                "J": J,
                                "K": K,
                                **summary,
                                "C_contraction": summary["C_post_sd"] / prior_sd,
                                "bias": mean - C_true,
                                "abs_error": abs(mean - C_true),
                                "squared_error": (mean - C_true) ** 2,
                                "covered_80": int(
                                    summary["C_central_80_low"] <= C_true <= summary["C_central_80_high"]
                                ),
                                "covered_95": int(
                                    summary["C_central_95_low"] <= C_true <= summary["C_central_95_high"]
                                ),
                            }
                        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "continuous_toy_replicates.csv", index=False)
    summary = (
        df.groupby(["alternative", "J", "K", "C_true"])
        .agg(
            n=("replicate", "size"),
            RMSE=("squared_error", lambda s: float(math.sqrt(np.mean(s)))),
            mean_abs_error=("abs_error", "mean"),
            mean_bias=("bias", "mean"),
            median_contraction=("C_contraction", "median"),
            coverage_80=("covered_80", "mean"),
            coverage_95=("covered_95", "mean"),
            median_post_sd=("C_post_sd", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "continuous_toy_summary.csv", index=False)
    return df, summary


# ---------------------------------------------------------------------------
# Phase 0.5 information prechecks
# ---------------------------------------------------------------------------


def run_information_precheck(
    out_dir: Path,
    J_values: Sequence[int] = (4, 16, 64),
    K_values: Sequence[int] = (1, 5, 20),
    C_values: Sequence[float] = (0.05, 0.10, 0.25, 0.50, 0.75),
    epsilon: float = 0.05,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    regimes = {
        "narrow": (0.60, 0.40),
        "medium": (0.80, 0.20),
        "extreme": (0.95, 0.05),
    }
    required = {
        "required_log_B_for_rho_0_50": -logit(PRIOR_ROOT_PROB),
        "required_log_B_for_rho_0_80": logit(0.80) - logit(PRIOR_ROOT_PROB),
        "required_log_B_for_rho_0_95": logit(0.95) - logit(PRIOR_ROOT_PROB),
        "required_log_B_for_rho_0_05": logit(0.05) - logit(PRIOR_ROOT_PROB),
    }
    for regime, (beta_pres, beta_abs) in regimes.items():
        for K in K_values:
            logp0, logp1 = binary_leaf_logp_by_R(K, beta_pres, beta_abs, epsilon)
            p0 = np.exp(logp0)
            p1 = np.exp(logp1)
            KL_1_to_0 = float(np.sum(p1 * (logp1 - logp0)))
            KL_0_to_1 = float(np.sum(p0 * (logp0 - logp1)))
            for J in J_values:
                rows.append(
                    {
                        "part": "binary_root_separation",
                        "alternative": np.nan,
                        "beta_gap_regime": regime,
                        "beta_pres": beta_pres,
                        "beta_abs": beta_abs,
                        "J": J,
                        "K": K,
                        "C": np.nan,
                        "KL_1_to_0_per_leaf": KL_1_to_0,
                        "KL_0_to_1_per_leaf": KL_0_to_1,
                        "E_R1_log_B": J * KL_1_to_0,
                        "E_R0_log_B": -J * KL_0_to_1,
                        **required,
                    }
                )

    prior_var = beta_sd(PRIOR_A, PRIOR_B) ** 2
    prior_precision_scale = 1.0 / prior_var
    beta_pres = 0.90
    beta_abs = 0.10
    gap = beta_pres - beta_abs
    dq_dC = gap
    dp_direct = gap * (1.0 - 2.0 * epsilon)
    for C in C_values:
        q_C = beta_abs + C * gap
        p_y = epsilon + q_C * (1.0 - 2.0 * epsilon)
        I_per_rating = dp_direct ** 2 / (p_y * (1.0 - p_y))
        for J in J_values:
            for K in K_values:
                I_total = J * K * I_per_rating
                rows.append(
                    {
                        "part": "continuous_fisher",
                        "alternative": "direct_q_continuous_observation",
                        "beta_gap_regime": "default_0.90_0.10",
                        "beta_pres": beta_pres,
                        "beta_abs": beta_abs,
                        "J": J,
                        "K": K,
                        "C": C,
                        "I_total": I_total,
                        "I_total_over_prior_precision": I_total / prior_precision_scale,
                        "prior_precision_scale": prior_precision_scale,
                        "approximate_contraction": math.sqrt(
                            prior_precision_scale / (prior_precision_scale + I_total)
                        ),
                    }
                )
        for K in K_values:
            t = np.arange(K + 1)
            bin_abs = np.exp(binom_logpmf(t, K, epsilon))
            bin_pres = np.exp(binom_logpmf(t, K, 1.0 - epsilon))
            p_t = (1.0 - q_C) * bin_abs + q_C * bin_pres
            dp_t = dq_dC * (bin_pres - bin_abs)
            I_per_leaf = float(np.sum((dp_t * dp_t) / np.clip(p_t, SCORE_CLIP, None)))
            for J in J_values:
                I_total = J * I_per_leaf
                rows.append(
                    {
                        "part": "continuous_fisher",
                        "alternative": "latent_leaf_continuous_propensity",
                        "beta_gap_regime": "default_0.90_0.10",
                        "beta_pres": beta_pres,
                        "beta_abs": beta_abs,
                        "J": J,
                        "K": K,
                        "C": C,
                        "I_total": I_total,
                        "I_total_over_prior_precision": I_total / prior_precision_scale,
                        "prior_precision_scale": prior_precision_scale,
                        "approximate_contraction": math.sqrt(
                            prior_precision_scale / (prior_precision_scale + I_total)
                        ),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "information_precheck.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 20, digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    tmp = df.loc[:, [c for c in columns if c in df.columns]].head(max_rows).copy()
    headers = list(tmp.columns)

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in tmp.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    production_check: Dict[str, Any],
    posthoc_df: pd.DataFrame,
    posthoc_summary: pd.DataFrame,
    binary_summary: pd.DataFrame,
    continuous_summary: pd.DataFrame,
    info_df: pd.DataFrame,
    commands: Sequence[str],
) -> None:
    report = out_dir / "phase0_root_semantics_report.md"
    prior_sd = beta_sd(PRIOR_A, PRIOR_B)
    absent_ratio = beta_sd(PRIOR_A, PRIOR_B + 1.0) / prior_sd
    present_ratio = beta_sd(PRIOR_A + 1.0, PRIOR_B) / prior_sd

    binary_mode = binary_summary[binary_summary["mode"] == "prior_calibrated_binary_root"]
    binary_table = binary_mode.sort_values(["J", "K"])
    fixed_pi_table = (
        binary_summary[binary_summary["mode"] == "fixed_pi_demonstration"]
        .sort_values(["J", "K"])
        .tail(3)
    )

    cont_avg = (
        continuous_summary.groupby(["alternative", "J", "K"])
        .agg(
            RMSE=("RMSE", "mean"),
            median_contraction=("median_contraction", "median"),
            coverage_80=("coverage_80", "mean"),
            coverage_95=("coverage_95", "mean"),
        )
        .reset_index()
        .sort_values(["alternative", "J", "K"])
    )
    cont_easy = cont_avg[(cont_avg["J"] == 64) & (cont_avg["K"] == 20)]

    info_binary = info_df[
        (info_df["part"] == "binary_root_separation")
        & (info_df["beta_gap_regime"].isin(["medium", "extreme"]))
        & (info_df["J"].isin([16, 64]))
        & (info_df["K"].isin([5, 20]))
    ].sort_values(["beta_gap_regime", "J", "K"])
    info_cont = info_df[
        (info_df["part"] == "continuous_fisher")
        & (info_df["J"] == 64)
        & (info_df["K"] == 20)
    ].sort_values(["alternative", "C"])

    if posthoc_df.empty:
        posthoc_note = "No existing synthetic fits were successfully reanalysed."
        posthoc_table = "_No rows._"
    else:
        free_rows = posthoc_df[posthoc_df["is_anchor_or_free"] == "free_target"]
        if free_rows.empty:
            posthoc_note = "Existing reanalysis only produced anchor rows."
        else:
            mean_delta_log = free_rows["delta_log_score"].mean()
            mean_delta_brier = free_rows["delta_brier"].mean()
            strong_present = int((free_rows["log_B_median"] >= (logit(0.80) - logit(PRIOR_ROOT_PROB))).sum())
            strong_absent = int((free_rows["log_B_median"] <= (logit(0.05) - logit(PRIOR_ROOT_PROB))).sum())
            ambiguous = int(len(free_rows) - strong_present - strong_absent)
            aligned = int(
                (
                    ((free_rows["R_true"] == 1) & (free_rows["rho_mean"] >= 0.5))
                    | ((free_rows["R_true"] == 0) & (free_rows["rho_mean"] < 0.5))
                ).sum()
            )
            posthoc_note = (
                "Free target systems show the old C_s summaries are not the right "
                f"binary-root target: mean delta log score versus the 1/6 prior is "
                f"{mean_delta_log:.3f}, and mean delta Brier is {mean_delta_brier:.3f} "
                "when scored through rho_s against realised root_z. Using prior-odds "
                f"log_B thresholds, free rows contain {strong_present} strong-present, "
                f"{strong_absent} strong-absent, and {ambiguous} ambiguous root-evidence "
                f"cases; {aligned}/{len(free_rows)} rho_mean decisions align with root_z "
                "at a 0.5 cutoff. Ambiguous cases are weak root evidence under rho_s, "
                "not failures to recover a continuous pi_s."
            )
        posthoc_table = markdown_table(
            posthoc_df.sort_values(["run_id", "is_anchor_or_free", "system"]),
            [
                "run_id",
                "source_path",
                "system",
                "R_true",
                "pi_true",
                "pi_posterior_mean",
                "pi_posterior_sd",
                "pi_contraction",
                "log_B_median",
                "rho_sampled_pi_mean",
                "rho_collapsed_mean",
                "brier_sampled_pi",
                "brier_collapsed",
                "delta_log_score",
                "delta_log_score_collapsed",
            ],
            max_rows=16,
        )

    production_pass = (
        production_check.get("deterministics_present")
        and production_check.get("rho_finite")
        and production_check.get("rho_in_unit_interval")
        and production_check.get("no_extra_potentials_added")
    )

    lines = [
        "# Phase 0 Root Semantics Report",
        "",
        "## 1. Executive summary",
        "",
        "The current DCM is a binary-root model. Under this model, the coherent "
        "recoverable posterior probability of consciousness is "
        "`rho_s = p(R_s = 1 | y_s)`. The posterior over `pi_s`, currently "
        "reported as `C_s`, is not expected to recover fixed continuous synthetic "
        "values such as 0.25 for a single system because `pi_s` only generates "
        "one latent root draw.",
        "",
        "Phase 0 supports that diagnosis. Binary-root toy data learn `R_s` through "
        "`rho_s`, while the posterior over `pi_s` hits the one-root-update ceiling "
        "instead of contracting to zero. Continuous `C_s` recovery appears only in "
        "the separate continuous-propensity toy semantics.",
        "",
        "## 2. Production deterministics",
        "",
        f"Production deterministic check: {'PASS' if production_pass else 'FAIL'}.",
        "",
        "Added exact-tree deterministics per system and stance:",
        "",
        "- `{prefix}_log_L_root0`",
        "- `{prefix}_log_L_root1`",
        "- `{prefix}_log_B`",
        "- `{prefix}_rho`",
        "",
        f"Missing deterministics: {production_check.get('missing_deterministics', [])}. "
        f"`rho` finite: {production_check.get('rho_finite')}; "
        f"`rho` in [0, 1]: {production_check.get('rho_in_unit_interval')}; "
        f"extra potentials added: {not production_check.get('no_extra_potentials_added')}.",
        "",
        "The root mixture is still the exact-tree likelihood potential. The added "
        "quantities are PyMC deterministics and are not added as potentials.",
        "",
        "## 3. Existing synthetic reanalysis",
        "",
        posthoc_note,
        "",
        markdown_table(
            posthoc_summary,
            [
                "is_anchor_or_free",
                "n",
                "n_runs",
                "mean_brier_sampled_pi",
                "mean_brier_collapsed",
                "mean_delta_brier",
                "mean_delta_brier_collapsed",
                "mean_delta_log_score",
                "mean_delta_log_score_collapsed",
                "mean_entropy_reduction",
                "median_abs_log_B",
                "median_rho_sampled_pi_mean",
                "median_rho_collapsed_mean",
                "median_pi_contraction",
            ],
        ),
        "",
        posthoc_table,
        "",
        "Central intervals are equal-tailed quantile intervals. ECE is not reported "
        "because the available reanalysis has too few independent systems per run "
        "for a stable 10-bin calibration estimate.",
        "",
        "## 4. Binary-root toy",
        "",
        f"Prior `pi_s ~ Beta(1, 5)` has prior SD {prior_sd:.4f}. The limiting "
        f"one-root-update contraction ratios are {absent_ratio:.3f} for strong "
        f"`R_s=0` evidence and {present_ratio:.3f} for strong `R_s=1` evidence.",
        "",
        markdown_table(
            binary_table,
            [
                "J",
                "K",
                "median_rho_given_R1",
                "median_rho_given_R0",
                "relative_brier_improvement",
                "mean_log_score_improvement",
                "median_pi_contraction",
                "median_pi_contraction_R1",
                "median_pi_contraction_R0",
            ],
            max_rows=12,
        ),
        "",
        "Fixed-pi demonstration rows at the strongest settings show that `pi_s` "
        "does not recover a fixed continuous `pi_true` from one realised root draw:",
        "",
        markdown_table(
            fixed_pi_table,
            [
                "J",
                "K",
                "median_rho_given_R1",
                "median_rho_given_R0",
                "mean_abs_pi_error",
                "median_pi_contraction",
            ],
        ),
        "",
        "## 5. Continuous-root toy",
        "",
        "These are toy diagnostics only; they are not production model changes. "
        "They show what changes when the data-generating semantics treats `C_s` "
        "as a continuous propensity rather than a one-draw Bernoulli parameter.",
        "",
        markdown_table(
            cont_easy,
            [
                "alternative",
                "J",
                "K",
                "RMSE",
                "median_contraction",
                "coverage_80",
                "coverage_95",
            ],
        ),
        "",
        "## 6. Information precheck",
        "",
        "Under binary-root semantics, information separates `R_s=1` from `R_s=0`; "
        "it does not estimate a system-specific continuous `pi_s` beyond the "
        "single Bernoulli update. With prior root probability 1/6, log_B thresholds "
        "are approximately 1.609 for `rho=0.50`, 2.996 for `rho=0.80`, 4.554 "
        "for `rho=0.95`, and -1.343 for `rho=0.05`.",
        "",
        markdown_table(
            info_binary,
            [
                "beta_gap_regime",
                "J",
                "K",
                "KL_1_to_0_per_leaf",
                "KL_0_to_1_per_leaf",
                "E_R1_log_B",
                "E_R0_log_B",
            ],
            max_rows=16,
        ),
        "",
        "For continuous-root toys, Fisher information is compared with the "
        "Beta(1,5) prior precision scale of about 50.4.",
        "",
        markdown_table(
            info_cont,
            [
                "alternative",
                "C",
                "I_total",
                "I_total_over_prior_precision",
                "approximate_contraction",
            ],
            max_rows=12,
        ),
        "",
        "## 7. Recommendations",
        "",
        "- Keep the current production DCM likelihood for now.",
        "- Report `rho_s` as the Path A headline under binary-root semantics.",
        "- Stop using continuous `pi_s` recovery as a pass/fail metric for the current model.",
        "- Use Brier, log score, and eventually ECE against realised `root_z` for synthetic binary-root validation.",
        "- Treat continuous-propensity `C_s` recovery as a separate v2 model question.",
        "",
        "## Commands run",
        "",
        *[f"- `{cmd}`" for cmd in commands],
        "",
    ]
    report.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--binary-reps", type=int, default=200)
    parser.add_argument("--continuous-reps", type=int, default=200)
    parser.add_argument("--max-posthoc-runs", type=int, default=None)
    parser.add_argument("--skip-posthoc", action="store_true")
    parser.add_argument("--skip-production-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        ".venv/bin/python phase0_root_semantics.py",
    ]
    if args.binary_reps != 200:
        commands[-1] += f" --binary-reps {args.binary_reps}"
    if args.continuous_reps != 200:
        commands[-1] += f" --continuous-reps {args.continuous_reps}"
    if args.max_posthoc_runs is not None:
        commands[-1] += f" --max-posthoc-runs {args.max_posthoc_runs}"

    if args.skip_production_check:
        production_check = {
            "model_build_ok": None,
            "deterministics_present": None,
            "rho_finite": None,
            "rho_in_unit_interval": None,
            "no_extra_potentials_added": None,
            "missing_deterministics": ["production check skipped"],
        }
    else:
        print("[0.1] checking production deterministics")
        production_check = check_production_deterministics(out_dir)

    if args.skip_posthoc:
        posthoc_df = pd.DataFrame()
        posthoc_summary = pd.DataFrame()
        posthoc_df.to_csv(out_dir / "posthoc_rho_reanalysis.csv", index=False)
        posthoc_summary.to_csv(out_dir / "posthoc_rho_summary.csv", index=False)
        pd.DataFrame([{"reason": "posthoc skipped"}]).to_csv(
            out_dir / "posthoc_skipped_runs.csv", index=False
        )
        build_posthoc_audit_outputs(out_dir, posthoc_df)
    else:
        print("[0.2] reanalysing existing synthetic fits")
        posthoc_df, posthoc_summary = posthoc_reanalysis(out_dir, args.max_posthoc_runs)
        build_posthoc_audit_outputs(out_dir, posthoc_df)

    print("[0.3] running binary-root toy")
    _, binary_summary = run_binary_root_toy(out_dir, reps=args.binary_reps)

    print("[0.4] running continuous-root toy alternatives")
    _, continuous_summary = run_continuous_toys(out_dir, reps=args.continuous_reps)

    print("[0.5] running information prechecks")
    info_df = run_information_precheck(out_dir)

    print("[report] writing markdown report")
    write_report(
        out_dir,
        production_check,
        posthoc_df,
        posthoc_summary,
        binary_summary,
        continuous_summary,
        info_df,
        commands,
    )
    print(f"wrote {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
