"""Metrics and shared utilities for the main DCM synthetic validation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit, logsumexp
from scipy.stats import norm

from composite_vs_exact_diagnostic import (
    collect_indicator_obs,
    exact_loglik,
    precompute_leaf_logliks,
)
from dcm_model import ModelConfig, MultiSystemDataProcessor, node_key
from exact_tree_ppc import (
    backward_pass,
    forward_pass,
    per_indicator_exact_tree_marginal,
)
from scripts.phase1_root_evidence_common import (
    FREE_SYSTEMS,
    LOGIT_PRIOR,
    PRIOR_P,
    SCORE_CLIP,
    STANCE,
    TAU_ABSENT_05,
    TAU_PRESENT_50,
    beta_profile_from_evidence_processor,
    bernoulli_brier,
    bernoulli_log_score,
    evidence_category_from_log_b,
    iter_tree_nodes,
)


STRONG_TOP_FEATURES = (
    "Coherence",
    "Selective Attention",
    "Complexity",
    "Integration",
)
WEAK_TOP_FEATURES = (
    "Representationality",
    "Hierarchical Organization",
    "Modularity",
)
CATEGORY_COUNT = 7


@dataclass(frozen=True)
class EdgeBetaProfile:
    variant: str
    beta_pres_true_by_key: dict[str, float]
    beta_abs_true_by_key: dict[str, float]
    fitter_prior_mean_pres_by_key: dict[str, float]
    fitter_prior_mean_abs_by_key: dict[str, float]
    targeted_override_node_keys: list[str]
    overridden_edges: list[str]
    edge_table: list[dict[str, Any]]


def clip_prob(x: float | np.ndarray, eps: float = SCORE_CLIP) -> float | np.ndarray:
    return np.clip(x, eps, 1.0 - eps)


def logit_clipped(p: float) -> float:
    p = float(clip_prob(p))
    return float(math.log(p / (1.0 - p)))


def rho_evidence_category(rho: float) -> str:
    if rho >= 0.95:
        return "strong_present"
    if rho >= 0.50:
        return "moderate_present"
    if rho >= 0.05:
        return "ambiguous"
    return "strong_absent"


def collect_targeted_override_node_keys(
    stance_data: Mapping[str, Any],
    strong_top_feature_names: Sequence[str] = STRONG_TOP_FEATURES,
) -> list[str]:
    """Collect lower-edge node keys under the selected strong top features."""
    strong = set(strong_top_feature_names)
    keys: list[str] = []
    for spec in iter_tree_nodes(stance_data):
        if spec.top_feature_name in strong and spec.depth > 1:
            keys.append(spec.key)
    return sorted(keys)


def build_edge_beta_profile(
    stance_data: Mapping[str, Any],
    variant: str,
    config: ModelConfig,
) -> EdgeBetaProfile:
    """Build matched generator truth and fitter prior centres for one variant."""
    base_pres, base_abs, base_meta = beta_profile_from_evidence_processor(
        stance_data,
        config,
    )
    truth_pres = dict(base_pres)
    truth_abs = dict(base_abs)
    fit_pres = dict(base_pres)
    fit_abs = dict(base_abs)
    targeted_keys: list[str] = []

    if variant == "targeted_strong_lower_override":
        targeted_keys = collect_targeted_override_node_keys(stance_data)
        mu_p = (
            float(config.BETA_PRES_OVERRIDE_MEAN)
            if config.BETA_PRES_OVERRIDE_MEAN is not None
            else 0.90
        )
        mu_a = (
            float(config.BETA_ABS_OVERRIDE_MEAN)
            if config.BETA_ABS_OVERRIDE_MEAN is not None
            else 0.10
        )
        for key in targeted_keys:
            truth_pres[key] = mu_p
            truth_abs[key] = mu_a
            fit_pres[key] = mu_p
            fit_abs[key] = mu_a
    elif variant == "production_unmodified":
        targeted_keys = []
    else:
        raise ValueError(f"Unknown fit variant {variant!r}")

    edge_rows: list[dict[str, Any]] = []
    for spec in iter_tree_nodes(stance_data):
        edge_rows.append(
            {
                "variant": variant,
                "node_key": spec.key,
                "node_name": spec.node["name"],
                "node_type": (spec.node.get("type") or "").lower(),
                "depth": int(spec.depth),
                "top_feature_name": spec.top_feature_name,
                "support": spec.node.get("support", "no bearing"),
                "demandingness": spec.node.get("demandingness", "neutral"),
                "is_targeted_override": spec.key in targeted_keys,
                "generator_beta_pres_true": float(truth_pres[spec.key]),
                "generator_beta_abs_true": float(truth_abs[spec.key]),
                "fitter_prior_mean_pres": float(fit_pres[spec.key]),
                "fitter_prior_mean_abs": float(fit_abs[spec.key]),
                "production_prior_mean_pres": float(base_pres[spec.key]),
                "production_prior_mean_abs": float(base_abs[spec.key]),
                **{
                    k: v
                    for k, v in base_meta.get(spec.key, {}).items()
                    if k not in {"beta_pres", "beta_abs", "node_name", "node_type"}
                },
            }
        )

    profile = EdgeBetaProfile(
        variant=variant,
        beta_pres_true_by_key=truth_pres,
        beta_abs_true_by_key=truth_abs,
        fitter_prior_mean_pres_by_key=fit_pres,
        fitter_prior_mean_abs_by_key=fit_abs,
        targeted_override_node_keys=targeted_keys,
        overridden_edges=[
            f"{row['node_key']}"
            for row in edge_rows
            if row["is_targeted_override"]
        ],
        edge_table=edge_rows,
    )
    validate_edge_beta_profile(stance_data, profile)
    return profile


def validate_edge_beta_profile(
    stance_data: Mapping[str, Any],
    profile: EdgeBetaProfile,
    tol: float = 1e-12,
) -> dict[str, Any]:
    spec_by_key = {spec.key: spec for spec in iter_tree_nodes(stance_data)}
    targeted = set(profile.targeted_override_node_keys)
    if profile.variant == "targeted_strong_lower_override" and not targeted:
        raise AssertionError("targeted override selected zero edges")
    root_to_top = [k for k in targeted if spec_by_key[k].depth == 1]
    weak = [k for k in targeted if spec_by_key[k].top_feature_name in WEAK_TOP_FEATURES]
    non_strong = [
        k for k in targeted if spec_by_key[k].top_feature_name not in STRONG_TOP_FEATURES
    ]
    if root_to_top:
        raise AssertionError(f"targeted override includes root-to-top edges: {root_to_top}")
    if weak:
        raise AssertionError(f"targeted override includes weak-top edges: {weak}")
    if non_strong:
        raise AssertionError(f"targeted override includes non-strong edges: {non_strong}")

    max_pres = max(
        abs(
            profile.beta_pres_true_by_key[k]
            - profile.fitter_prior_mean_pres_by_key[k]
        )
        for k in profile.beta_pres_true_by_key
    )
    max_abs = max(
        abs(
            profile.beta_abs_true_by_key[k]
            - profile.fitter_prior_mean_abs_by_key[k]
        )
        for k in profile.beta_abs_true_by_key
    )
    if max_pres > tol or max_abs > tol:
        raise AssertionError(
            "generator/fitter beta centres differ: "
            f"max_pres={max_pres:.3e}, max_abs={max_abs:.3e}"
        )
    return {
        "overridden_edge_count": len(targeted),
        "root_to_top_overridden_edge_count": len(root_to_top),
        "weak_top_overridden_edge_count": len(weak),
        "max_abs_generator_fitter_pres_diff": float(max_pres),
        "max_abs_generator_fitter_abs_diff": float(max_abs),
        "targeted_override_prior_family": (
            "node_level_logit_normal" if targeted else "not_applicable"
        ),
    }


def profile_as_truth_edge_betas(
    stance_data: Mapping[str, Any],
    profile: EdgeBetaProfile,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for spec in iter_tree_nodes(stance_data):
        bp = float(profile.beta_pres_true_by_key[spec.key])
        ba = float(profile.beta_abs_true_by_key[spec.key])
        rows[spec.key] = {
            "node_name": spec.node["name"],
            "node_type": (spec.node.get("type") or "").lower(),
            "top_feature_name": spec.top_feature_name,
            "depth": int(spec.depth),
            "support": spec.node.get("support", "no bearing"),
            "demandingness": spec.node.get("demandingness", "neutral"),
            "beta_pres": bp,
            "beta_abs": ba,
            "delta": bp - ba,
            "is_targeted_override": spec.key in profile.targeted_override_node_keys,
        }
    return rows


def flatten_draws(post: Any, var: str, draw_idx: Optional[np.ndarray] = None) -> np.ndarray:
    arr = np.asarray(post[var].values)
    flat = arr.reshape((-1,) + arr.shape[2:])
    if draw_idx is not None:
        flat = flat[draw_idx]
    return flat


def beta_draws_by_node(
    idata: Any,
    stance_data: Mapping[str, Any],
    node_to_varname: Mapping[str, str],
    draw_idx: Optional[np.ndarray] = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    post = idata.posterior
    bp: dict[str, np.ndarray] = {}
    ba: dict[str, np.ndarray] = {}
    for spec in iter_tree_nodes(stance_data):
        var_prefix = node_to_varname[spec.key]
        bp_var = f"{var_prefix}_beta_pres"
        ba_var = f"{var_prefix}_beta_abs"
        if bp_var not in post.data_vars or ba_var not in post.data_vars:
            raise KeyError(
                f"Missing per-node beta vars for {spec.key}: {bp_var}, {ba_var}"
            )
        bp[spec.key] = flatten_draws(post, bp_var, draw_idx).reshape(-1)
        ba[spec.key] = flatten_draws(post, ba_var, draw_idx).reshape(-1)
    return bp, ba


def posterior_observation_arrays(
    idata: Any,
    proc: MultiSystemDataProcessor,
    config: ModelConfig,
    draw_idx: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    post = idata.posterior
    a_draws = flatten_draws(post, "a", draw_idx).reshape(-1)
    kappa_draws = flatten_draws(post, "kappa", draw_idx).reshape(-1, config.N_CATEGORIES - 1)
    n_experts = len(proc.expert_names)
    if "b_free" in post.data_vars:
        b_free = flatten_draws(post, "b_free", draw_idx).reshape(-1, n_experts - 1)
        b_draws = np.concatenate([np.zeros((b_free.shape[0], 1)), b_free], axis=1)
    elif "b" in post.data_vars:
        b_draws = flatten_draws(post, "b", draw_idx).reshape(-1, n_experts)
    else:
        b_draws = np.zeros((a_draws.shape[0], n_experts))
    return a_draws, b_draws, kappa_draws


def system_c_draws(
    idata: Any,
    builder: Any,
    system_configs: Sequence[tuple[str, Optional[float]]],
    draw_idx: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    post = idata.posterior
    out: dict[str, np.ndarray] = {}
    n_draws = flatten_draws(post, "a", draw_idx).shape[0]
    for sys_name, c_fixed in system_configs:
        var = f"{builder._sys_prefix(sys_name)}__global_workspace_theory_C"
        if var in post.data_vars:
            out[sys_name] = flatten_draws(post, var, draw_idx).reshape(-1)
        elif c_fixed is not None:
            out[sys_name] = np.full(n_draws, float(c_fixed))
        else:
            raise KeyError(f"Missing root C posterior var {var!r}")
    return out


def choose_draw_idx(n_total: int, n_draws: Optional[int], seed: int) -> Optional[np.ndarray]:
    if n_draws is None or n_draws >= n_total:
        return None
    if n_draws <= 0:
        raise ValueError("draw count must be positive or None")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(n_total), size=int(n_draws), replace=False))


def extract_rho_cases(
    idata: Any,
    builder: Any,
    fit_variant: str,
    seed: int,
    root_z_by_system: Mapping[str, int],
    free_systems: Sequence[str] = FREE_SYSTEMS,
) -> pd.DataFrame:
    post = idata.posterior
    rows: list[dict[str, Any]] = []
    prior_brier_by_y = {
        0: bernoulli_brier(PRIOR_P, 0),
        1: bernoulli_brier(PRIOR_P, 1),
    }
    prior_log_by_y = {
        0: bernoulli_log_score(PRIOR_P, 0),
        1: bernoulli_log_score(PRIOR_P, 1),
    }
    for system in free_systems:
        prefix = f"{builder._sys_prefix(system)}__global_workspace_theory"
        log_b_var = f"{prefix}_log_B"
        if log_b_var not in post.data_vars:
            raise KeyError(f"Missing root evidence deterministic {log_b_var}")
        log_b = flatten_draws(post, log_b_var).reshape(-1)
        rho_collapsed_var = f"{prefix}_rho_collapsed"
        if rho_collapsed_var in post.data_vars:
            rho_collapsed_draws = flatten_draws(post, rho_collapsed_var).reshape(-1)
        else:
            rho_collapsed_draws = expit(LOGIT_PRIOR + log_b)
        rho_sampled_var = f"{prefix}_rho"
        rho_sampled_draws = (
            flatten_draws(post, rho_sampled_var).reshape(-1)
            if rho_sampled_var in post.data_vars
            else np.full_like(rho_collapsed_draws, np.nan)
        )
        rho_collapsed = float(np.mean(rho_collapsed_draws))
        rho_sampled = float(np.nanmean(rho_sampled_draws))
        log_b_eff = logit_clipped(rho_collapsed) - LOGIT_PRIOR
        root_z = int(root_z_by_system[system])
        brier = bernoulli_brier(rho_collapsed, root_z)
        log_score = bernoulli_log_score(rho_collapsed, root_z)
        rows.append(
            {
                "fit_variant": fit_variant,
                "seed": int(seed),
                "system": "LLMs" if system == "2024 Leading Chat LLMs" else system,
                "system_raw": system,
                "root_z_true": root_z,
                "rho_collapsed": rho_collapsed,
                "rho_sampled_pi": rho_sampled,
                "abs_rho_collapsed_minus_sampled_pi": float(
                    abs(rho_collapsed - rho_sampled)
                ),
                "log_B_draw_mean": float(np.mean(log_b)),
                "log_B_draw_median": float(np.median(log_b)),
                "log_B_draw_q05": float(np.percentile(log_b, 5)),
                "log_B_draw_q95": float(np.percentile(log_b, 95)),
                "log_B_eff": float(log_b_eff),
                "evidence_category": rho_evidence_category(rho_collapsed),
                "correct_sign": bool(
                    (root_z == 1 and log_b_eff > 0.0)
                    or (root_z == 0 and log_b_eff < 0.0)
                ),
                "classified_present_at_0p5": bool(rho_collapsed > 0.5),
                "decisive_present_at_0p95": bool(rho_collapsed >= 0.95),
                "decisive_absent_at_0p05": bool(rho_collapsed < 0.05),
                "brier": float(brier),
                "log_score": float(log_score),
                "brier_improvement_vs_prior": float(
                    prior_brier_by_y[root_z] - brier
                ),
                "log_score_improvement_vs_prior": float(
                    log_score - prior_log_by_y[root_z]
                ),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(df: pd.DataFrame, n_bins: int = 10) -> tuple[float, str]:
    if df.empty:
        return np.nan, "[]"
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    n = len(df)
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (df["rho_collapsed"] >= lo) & (df["rho_collapsed"] <= hi)
        else:
            mask = (df["rho_collapsed"] >= lo) & (df["rho_collapsed"] < hi)
        g = df[mask]
        if g.empty:
            continue
        conf = float(g["rho_collapsed"].mean())
        acc = float(g["root_z_true"].mean())
        weight = len(g) / n
        ece += weight * abs(conf - acc)
        rows.append({"lo": lo, "hi": hi, "n": int(len(g)), "mean_rho": conf, "empirical_rate": acc})
    return float(ece), json.dumps(rows)


def summarise_rho_recovery(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return pd.DataFrame()
    group_frames: list[tuple[tuple[Any, ...], pd.DataFrame, list[str]]] = []
    for key, g in cases.groupby(["fit_variant"], dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        group_frames.append((key_tuple, g, ["fit_variant"]))
    for key, g in cases.groupby(["fit_variant", "system"], dropna=False):
        group_frames.append((key if isinstance(key, tuple) else (key,), g, ["fit_variant", "system"]))

    rows: list[dict[str, Any]] = []
    baseline_brier = 0.5 * (1 - PRIOR_P) ** 2 + 0.5 * PRIOR_P**2
    prior_log_baseline = 0.5 * math.log(PRIOR_P) + 0.5 * math.log1p(-PRIOR_P)
    for key, g, cols in group_frames:
        row = {col: val for col, val in zip(cols, key)}
        row.setdefault("system", "ALL")
        r1 = g[g["root_z_true"] == 1]
        r0 = g[g["root_z_true"] == 0]
        tpr = float((r1["rho_collapsed"] > 0.5).mean()) if not r1.empty else np.nan
        tnr = float((r0["rho_collapsed"] <= 0.5).mean()) if not r0.empty else np.nan
        bal_acc = float(np.nanmean([tpr, tnr]))
        m = (
            float(
                min(
                    r1["log_B_eff"].median() - TAU_PRESENT_50,
                    TAU_ABSENT_05 - r0["log_B_eff"].median(),
                )
            )
            if not r1.empty and not r0.empty
            else np.nan
        )
        brier_model = (
            0.5 * float(((1.0 - r1["rho_collapsed"]) ** 2).mean())
            if not r1.empty
            else 0.0
        ) + (
            0.5 * float((r0["rho_collapsed"] ** 2).mean())
            if not r0.empty
            else 0.0
        )
        log_model = (
            0.5 * float(np.log(clip_prob(r1["rho_collapsed"].to_numpy())).mean())
            if not r1.empty
            else 0.0
        ) + (
            0.5
            * float(np.log(clip_prob(1.0 - r0["rho_collapsed"].to_numpy())).mean())
            if not r0.empty
            else 0.0
        )
        ece, bins = expected_calibration_error(g)
        rows.append(
            {
                **row,
                "n_cases": int(len(g)),
                "mean_rho_R1": float(r1["rho_collapsed"].mean()) if not r1.empty else np.nan,
                "mean_rho_R0": float(r0["rho_collapsed"].mean()) if not r0.empty else np.nan,
                "median_rho_R1": float(r1["rho_collapsed"].median()) if not r1.empty else np.nan,
                "median_rho_R0": float(r0["rho_collapsed"].median()) if not r0.empty else np.nan,
                "median_log_B_eff_R1": float(r1["log_B_eff"].median()) if not r1.empty else np.nan,
                "median_log_B_eff_R0": float(r0["log_B_eff"].median()) if not r0.empty else np.nan,
                "evidence_margin_M": m,
                "TPR_at_rho_gt_0p5": tpr,
                "TNR_at_rho_le_0p5": tnr,
                "balanced_accuracy": bal_acc,
                "decisive_present_rate_R1_at_rho_gt_0p95": float(
                    r1["decisive_present_at_0p95"].mean()
                ) if not r1.empty else np.nan,
                "decisive_absent_rate_R0_at_rho_lt_0p05": float(
                    r0["decisive_absent_at_0p05"].mean()
                ) if not r0.empty else np.nan,
                "brier_model": float(brier_model),
                "brier_prior_baseline": float(baseline_brier),
                "brier_improvement": float(baseline_brier - brier_model),
                "log_score_model": float(log_model),
                "log_score_prior_baseline": float(prior_log_baseline),
                "log_score_improvement": float(log_model - prior_log_baseline),
                "ECE": ece,
                "calibration_bins": bins,
                "mean_abs_rho_collapsed_minus_sampled_pi": float(
                    g["abs_rho_collapsed_minus_sampled_pi"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def category_probs_for_draws(
    kappa_draws: np.ndarray,
    eta: np.ndarray,
    K: int = CATEGORY_COUNT,
) -> np.ndarray:
    cum = norm.cdf(kappa_draws - eta[:, None])
    cum_full = np.concatenate(
        [np.zeros((cum.shape[0], 1)), cum, np.ones((cum.shape[0], 1))],
        axis=1,
    )
    probs = np.diff(cum_full, axis=1)
    probs = np.clip(probs, SCORE_CLIP, 1.0)
    return probs / probs.sum(axis=1, keepdims=True)


def ranked_probability_score(probs: np.ndarray, y: int) -> float:
    cdf = np.cumsum(np.asarray(probs, dtype=float))
    obs = (np.arange(len(probs)) >= int(y)).astype(float)
    return float(np.mean((cdf[:-1] - obs[:-1]) ** 2))


def compute_exact_tree_ppc(
    *,
    idata: Any,
    builder: Any,
    proc: MultiSystemDataProcessor,
    stance_data: Mapping[str, Any],
    config: ModelConfig,
    system_configs: Sequence[tuple[str, Optional[float]]],
    output_path: Path,
    ppc_draws: int,
    seed: int,
    fit_variant: str,
    fit_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    post = idata.posterior
    n_total = flatten_draws(post, "a").shape[0]
    draw_idx = choose_draw_idx(n_total, ppc_draws, seed)
    a_draws, b_draws, kappa_draws = posterior_observation_arrays(
        idata, proc, config, draw_idx
    )
    bp_dict, ba_dict = beta_draws_by_node(
        idata, stance_data, builder.node_to_varname, draw_idx
    )
    c_by_system = system_c_draws(idata, builder, system_configs, draw_idx)
    indicator_obs = collect_indicator_obs(dict(stance_data), proc)
    leaf_logliks = precompute_leaf_logliks(
        indicator_obs, a_draws, b_draws, kappa_draws, K=config.N_CATEGORIES
    )

    pm_per_indicator: dict[str, dict[str, np.ndarray]] = {}
    for sys_name in proc.systems:
        log_d, log_l_at_zv, log_l_root = forward_pass(
            dict(stance_data),
            indicator_obs,
            leaf_logliks,
            bp_dict,
            ba_dict,
            sys_name,
        )
        log_o = backward_pass(
            dict(stance_data),
            log_d,
            log_l_at_zv,
            log_l_root,
            bp_dict,
            ba_dict,
            c_by_system[sys_name],
        )
        pm_per_indicator[sys_name] = per_indicator_exact_tree_marginal(
            dict(stance_data),
            indicator_obs,
            leaf_logliks,
            log_d,
            log_l_at_zv,
            log_o,
            bp_dict,
            ba_dict,
            sys_name,
        )

    rating_rows: list[dict[str, Any]] = []
    draw_probs: list[np.ndarray] = []
    for sys_name in proc.systems:
        for ikey, obs in proc.system_observations.get(sys_name, {}).items():
            if ikey not in pm_per_indicator.get(sys_name, {}):
                continue
            pm = pm_per_indicator[sys_name][ikey]
            for local_idx, (expert_idx, rating) in enumerate(obs):
                pk = []
                for m in (0, 1, 2):
                    eta = b_draws[:, expert_idx] + (m / 2.0) * a_draws
                    pk.append(category_probs_for_draws(kappa_draws, eta, config.N_CATEGORIES))
                pk_stack = np.stack(pk, axis=1)  # (S, 3, K)
                probs_draw = np.einsum("sm,smk->sk", pm, pk_stack)
                probs_mean = probs_draw.mean(axis=0)
                draw_probs.append(probs_draw)
                rating_rows.append(
                    {
                        "row_id": len(rating_rows),
                        "system": "LLMs" if sys_name == "2024 Leading Chat LLMs" else sys_name,
                        "system_raw": sys_name,
                        "indicator_key": ikey,
                        "node_key": ikey,
                        "indicator_label": str(ikey).split(" > ")[-1],
                        "local_rating_index": int(local_idx),
                        "expert_idx": int(expert_idx),
                        "rater_id": (
                            proc.expert_names[int(expert_idx)]
                            if int(expert_idx) < len(proc.expert_names)
                            else str(expert_idx)
                        ),
                        "rating": int(rating),
                        "observed_rating": int(rating),
                        "RPS": ranked_probability_score(probs_mean, int(rating)),
                        "pred_top7_mass": float(probs_mean[-1]),
                        "pred_high_mass": float(probs_mean[-2:].sum()),
                    }
                )

    if not rating_rows:
        raise RuntimeError("No rating rows available for PPC")
    row_df = pd.DataFrame(rating_rows)
    probs = np.stack(draw_probs, axis=1)  # (S, N, K)
    pred_mean = probs.mean(axis=0)
    obs_ratings = row_df["rating"].to_numpy(dtype=int)
    baseline_counts = np.bincount(obs_ratings, minlength=config.N_CATEGORIES).astype(float)
    baseline_probs = baseline_counts / baseline_counts.sum()
    row_df["RPS_baseline_empirical_marginal"] = [
        ranked_probability_score(baseline_probs, int(y)) for y in obs_ratings
    ]
    row_df["RPS_improvement"] = row_df["RPS_baseline_empirical_marginal"] - row_df["RPS"]

    xr.Dataset(
        {
            "category_probability": (("draw", "rating_row", "category"), probs),
            "posterior_mean_category_probability": (("rating_row", "category"), pred_mean),
            "observed_rating": (("rating_row",), obs_ratings),
        },
        coords={
            "draw": np.arange(probs.shape[0]),
            "rating_row": row_df["row_id"].to_numpy(),
            "category": np.arange(config.N_CATEGORIES),
        },
    ).to_netcdf(output_path)

    row_metadata = row_df[
        [
            "row_id",
            "system",
            "indicator_key",
            "node_key",
            "indicator_label",
            "rater_id",
            "observed_rating",
        ]
    ].rename(columns={"row_id": "rating_row"})
    row_metadata.insert(0, "fit_variant", fit_variant)
    row_metadata.insert(1, "seed", int(fit_seed))
    row_metadata.to_csv(output_path.parent / "posterior_predictive_row_metadata.csv", index=False)

    cell_rows: list[dict[str, Any]] = []
    for (sys_name, ikey), g in row_df.groupby(["system_raw", "node_key"], dropna=False):
        idx = g["row_id"].to_numpy(dtype=int)
        obs_counts = np.bincount(g["rating"].to_numpy(dtype=int), minlength=config.N_CATEGORIES).astype(float)
        obs_dist = obs_counts / obs_counts.sum()
        pred_dist = pred_mean[idx].mean(axis=0)
        tv = 0.5 * float(np.abs(obs_dist - pred_dist).sum())
        cell_rows.append(
            {
                "fit_variant": fit_variant,
                "seed": int(fit_seed),
                "system": "LLMs" if sys_name == "2024 Leading Chat LLMs" else sys_name,
                "system_raw": sys_name,
                "node_key": ikey,
                "n_obs": int(len(g)),
                "cell_TV": tv,
                "obs_top7_mass": float(obs_dist[-1]),
                "ppc_top7_mass": float(pred_dist[-1]),
                "obs_high_mass": float(obs_dist[-2:].sum()),
                "ppc_high_mass": float(pred_dist[-2:].sum()),
                "abs_top7_error": float(abs(obs_dist[-1] - pred_dist[-1])),
                "abs_high_error": float(abs(obs_dist[-2:].sum() - pred_dist[-2:].sum())),
                "high_q_cell": bool(pred_dist[-2:].sum() >= 0.50),
            }
        )
    cell_df = pd.DataFrame(cell_rows)

    observed_tv_weighted = float(np.average(cell_df["cell_TV"], weights=cell_df["n_obs"]))
    replicated_stats = []
    rng = np.random.default_rng(seed + 17)
    for draw in range(probs.shape[0]):
        rep_rating = np.array(
            [rng.choice(np.arange(config.N_CATEGORIES), p=probs[draw, i]) for i in range(probs.shape[1])],
            dtype=int,
        )
        rep_tvs = []
        rep_weights = []
        for (_, _), g in row_df.groupby(["system_raw", "node_key"], dropna=False):
            idx = g["row_id"].to_numpy(dtype=int)
            counts = np.bincount(rep_rating[idx], minlength=config.N_CATEGORIES).astype(float)
            dist = counts / counts.sum()
            ref = pred_mean[idx].mean(axis=0)
            rep_tvs.append(0.5 * float(np.abs(dist - ref).sum()))
            rep_weights.append(len(idx))
        replicated_stats.append(float(np.average(rep_tvs, weights=rep_weights)))
    p_value = float(np.mean(np.asarray(replicated_stats) >= observed_tv_weighted))

    summary_rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame, pd.DataFrame]] = [
        ("ALL", row_df, cell_df),
    ]
    for sys_name, rg in row_df.groupby("system", dropna=False):
        groups.append((str(sys_name), rg, cell_df[cell_df["system"] == sys_name]))
    for sys_name, rg, cg in groups:
        high_q = cg[cg["high_q_cell"]]
        summary_rows.append(
            {
                "fit_variant": fit_variant,
                "seed": int(fit_seed),
                "system": sys_name,
                "n_ratings": int(len(rg)),
                "mean_RPS": float(rg["RPS"].mean()),
                "median_RPS": float(rg["RPS"].median()),
                "q90_RPS": float(rg["RPS"].quantile(0.90)),
                "mean_RPS_baseline_empirical_marginal": float(
                    rg["RPS_baseline_empirical_marginal"].mean()
                ),
                "mean_RPS_improvement": float(rg["RPS_improvement"].mean()),
                "weighted_mean_cell_TV": float(
                    np.average(cg["cell_TV"], weights=cg["n_obs"])
                ) if not cg.empty else np.nan,
                "median_cell_TV": float(cg["cell_TV"].median()) if not cg.empty else np.nan,
                "q90_cell_TV": float(cg["cell_TV"].quantile(0.90)) if not cg.empty else np.nan,
                "max_cell_TV": float(cg["cell_TV"].max()) if not cg.empty else np.nan,
                "fraction_cell_TV_gt_0p3": float((cg["cell_TV"] > 0.3).mean()) if not cg.empty else np.nan,
                "fraction_cell_TV_gt_0p5": float((cg["cell_TV"] > 0.5).mean()) if not cg.empty else np.nan,
                "n_cells": int(len(cg)),
                "n_cells_ge_5": int((cg["n_obs"] >= 5).sum()) if not cg.empty else 0,
                "mean_abs_top7_error_high_q": float(high_q["abs_top7_error"].mean()) if not high_q.empty else np.nan,
                "q90_abs_top7_error_high_q": float(high_q["abs_top7_error"].quantile(0.90)) if not high_q.empty else np.nan,
                "mean_abs_high_error_high_q": float(high_q["abs_high_error"].mean()) if not high_q.empty else np.nan,
                "q90_abs_high_error_high_q": float(high_q["abs_high_error"].quantile(0.90)) if not high_q.empty else np.nan,
                "posterior_predictive_tv_p_value": p_value if sys_name == "ALL" else np.nan,
                "posterior_predictive_tv_p_value_status": (
                    "computed_overall"
                    if sys_name == "ALL"
                    else "not_computed_system_specific"
                ),
                "ppc_draws": int(probs.shape[0]),
            }
        )

    dist_rows: list[dict[str, Any]] = []
    for sys_name, g in row_df.groupby("system", dropna=False):
        idx = g["row_id"].to_numpy(dtype=int)
        obs_counts = np.bincount(g["rating"].to_numpy(dtype=int), minlength=config.N_CATEGORIES).astype(float)
        obs_dist = obs_counts / obs_counts.sum()
        pred_dist = pred_mean[idx].mean(axis=0)
        for cat in range(config.N_CATEGORIES):
            dist_rows.append(
                {
                    "fit_variant": fit_variant,
                    "seed": int(fit_seed),
                    "system": sys_name,
                    "category": int(cat),
                    "observed_proportion": float(obs_dist[cat]),
                    "predicted_proportion": float(pred_dist[cat]),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(dist_rows)


def per_rating_category_logp(
    rating: int,
    expert_idx: int,
    a_draws: np.ndarray,
    b_draws: np.ndarray,
    kappa_draws: np.ndarray,
    eta_offset_factor: float,
    K: int,
) -> np.ndarray:
    eta = b_draws[:, expert_idx] + eta_offset_factor * a_draws
    cum = norm.cdf(kappa_draws - eta[:, None])
    cum_full = np.concatenate(
        [np.zeros((cum.shape[0], 1)), cum, np.ones((cum.shape[0], 1))],
        axis=1,
    )
    probs = np.clip(np.diff(cum_full, axis=1), SCORE_CLIP, 1.0)
    return np.log(probs[:, int(rating)])


def stratified_rating_rows(
    proc: MultiSystemDataProcessor,
    max_ratings: Optional[int],
    seed: int,
) -> pd.DataFrame:
    rows = []
    for system, obs_by_key in proc.system_observations.items():
        for key, obs in obs_by_key.items():
            for local_idx, (expert_idx, rating) in enumerate(obs):
                rows.append(
                    {
                        "rating_row": len(rows),
                        "system_raw": system,
                        "system": "LLMs" if system == "2024 Leading Chat LLMs" else system,
                        "node_key": key,
                        "local_rating_index": int(local_idx),
                        "expert_idx": int(expert_idx),
                        "rating": int(rating),
                    }
                )
    df = pd.DataFrame(rows)
    if max_ratings is None or max_ratings >= len(df):
        return df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    parts = []
    per_system = max(1, int(math.ceil(max_ratings / max(1, df["system_raw"].nunique()))))
    for _, g in df.groupby("system_raw", dropna=False):
        n = min(per_system, len(g))
        parts.append(g.sample(n=n, random_state=int(rng.integers(0, 2**31 - 1))))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > max_ratings:
        out = out.sample(n=max_ratings, random_state=int(rng.integers(0, 2**31 - 1)))
    return out.sort_values("rating_row").reset_index(drop=True)


def compute_conditional_loo(
    *,
    idata: Any,
    builder: Any,
    proc: MultiSystemDataProcessor,
    stance_data: Mapping[str, Any],
    config: ModelConfig,
    system_configs: Sequence[tuple[str, Optional[float]]],
    output_path: Path,
    loo_draws: int,
    loo_max_ratings: Optional[int],
    seed: int,
    fit_variant: str,
    fit_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    post = idata.posterior
    n_total = flatten_draws(post, "a").shape[0]
    draw_idx = choose_draw_idx(n_total, loo_draws, seed)
    a_draws, b_draws, kappa_draws = posterior_observation_arrays(
        idata, proc, config, draw_idx
    )
    bp_dict, ba_dict = beta_draws_by_node(
        idata, stance_data, builder.node_to_varname, draw_idx
    )
    c_by_system = system_c_draws(idata, builder, system_configs, draw_idx)
    indicator_obs = collect_indicator_obs(dict(stance_data), proc)
    leaf_logliks = precompute_leaf_logliks(
        indicator_obs, a_draws, b_draws, kappa_draws, K=config.N_CATEGORIES
    )
    rows = stratified_rating_rows(proc, loo_max_ratings, seed)
    n_draws = a_draws.shape[0]
    log_lik = np.zeros((n_draws, len(rows)), dtype=float)

    full_by_system: dict[str, np.ndarray] = {}
    for sys_name in proc.systems:
        _, per = exact_loglik(
            dict(stance_data),
            indicator_obs,
            leaf_logliks,
            bp_dict,
            ba_dict,
            {sys_name: c_by_system[sys_name]},
        )
        full_by_system[sys_name] = per[sys_name]

    for col, row in rows.iterrows():
        sys_name = row["system_raw"]
        key = row["node_key"]
        contrib = np.stack(
            [
                per_rating_category_logp(
                    int(row["rating"]),
                    int(row["expert_idx"]),
                    a_draws,
                    b_draws,
                    kappa_draws,
                    factor,
                    config.N_CATEGORIES,
                )
                for factor in (0.0, 0.5, 1.0)
            ],
            axis=1,
        )
        leaf_minus = {k: dict(v) for k, v in leaf_logliks.items()}
        modified = np.array(leaf_minus[key][sys_name], copy=True)
        modified -= contrib
        leaf_minus[key][sys_name] = modified
        _, per_minus = exact_loglik(
            dict(stance_data),
            indicator_obs,
            leaf_minus,
            bp_dict,
            ba_dict,
            {sys_name: c_by_system[sys_name]},
        )
        log_lik[:, col] = full_by_system[sys_name] - per_minus[sys_name]

    xr.Dataset(
        {
            "log_likelihood": (("draw", "rating_row"), log_lik),
            "rating_row_source_index": (("rating_row",), rows["rating_row"].to_numpy(dtype=int)),
        },
        coords={"draw": np.arange(n_draws), "rating_row": np.arange(len(rows))},
    ).to_netcdf(output_path)

    loo_idata = az.from_dict(
        posterior={"dummy": np.zeros((1, n_draws))},
        log_likelihood={"rating": log_lik[None, :, :]},
    )
    loo_res = az.loo(loo_idata, pointwise=True)
    pareto_k = np.asarray(loo_res.pareto_k.values).reshape(-1)
    lppd = float(np.sum(logsumexp(log_lik, axis=0) - math.log(n_draws)))
    summary = pd.DataFrame(
        [
            {
                "fit_variant": fit_variant,
                "seed": int(fit_seed),
                "n_ratings": int(sum(len(v) for obs in proc.system_observations.values() for v in obs.values())),
                "n_ratings_used_for_loo": int(len(rows)),
                "loo_draws": int(n_draws),
                "loo_partial_flag": bool(len(rows) < sum(len(v) for obs in proc.system_observations.values() for v in obs.values())),
                "elpd_loo": float(loo_res.elpd_loo),
                "se_elpd_loo": float(loo_res.se),
                "p_loo": float(loo_res.p_loo),
                "lppd_conditional": lppd,
                "mean_pareto_k": float(np.mean(pareto_k)),
                "median_pareto_k": float(np.median(pareto_k)),
                "max_pareto_k": float(np.max(pareto_k)),
                "frac_pareto_k_gt_0p7": float(np.mean(pareto_k > 0.7)),
                "frac_pareto_k_gt_1p0": float(np.mean(pareto_k > 1.0)),
            }
        ]
    )
    pareto = rows.copy()
    pareto["pareto_k"] = pareto_k
    pareto["fit_variant"] = fit_variant
    pareto["seed"] = int(fit_seed)
    return summary, pareto


def sampler_diagnostics(
    idata: Any,
    runtime_seconds: float,
    fit_variant: str,
    seed: int,
) -> pd.DataFrame:
    diag = az.summary(idata, kind="diagnostics")
    sample_stats = idata.sample_stats
    n_samples = int(np.prod(sample_stats["diverging"].shape))
    div = int(sample_stats["diverging"].values.sum())
    if "reached_max_treedepth" in sample_stats:
        td_hits = int(sample_stats["reached_max_treedepth"].values.sum())
    elif "tree_depth" in sample_stats:
        max_depth = int(np.nanmax(sample_stats["tree_depth"].values))
        td_hits = int((sample_stats["tree_depth"].values >= max_depth).sum())
    else:
        td_hits = 0
    accept = (
        float(np.nanmean(sample_stats["acceptance_rate"].values))
        if "acceptance_rate" in sample_stats
        else np.nan
    )
    return pd.DataFrame(
        [
            {
                "fit_variant": fit_variant,
                "seed": int(seed),
                "max_rhat": float(diag["r_hat"].max(skipna=True)),
                "n_rhat_gt_1p01": int((diag["r_hat"] > 1.01).sum()),
                "n_rhat_gt_1p05": int((diag["r_hat"] > 1.05).sum()),
                "min_ess_bulk": float(diag["ess_bulk"].min(skipna=True)),
                "min_ess_tail": float(diag["ess_tail"].min(skipna=True)),
                "median_ess_bulk": float(diag["ess_bulk"].median(skipna=True)),
                "median_ess_tail": float(diag["ess_tail"].median(skipna=True)),
                "n_divergences": div,
                "divergence_rate": float(div / max(1, n_samples)),
                "max_tree_depth_hits": td_hits,
                "max_tree_depth_hit_rate": float(td_hits / max(1, n_samples)),
                "mean_acceptance_rate": accept,
                "runtime_seconds": float(runtime_seconds),
            }
        ]
    )


def pass_fail_summary(
    rho_summary: pd.DataFrame,
    ppc_summary: pd.DataFrame,
    loo_summary: pd.DataFrame,
    sampler_summary: pd.DataFrame,
    validation_scope: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    variants = sorted(
        set(rho_summary.get("fit_variant", []))
        | set(ppc_summary.get("fit_variant", []))
        | set(loo_summary.get("fit_variant", []))
        | set(sampler_summary.get("fit_variant", []))
    )
    for variant in variants:
        rho = rho_summary[
            (rho_summary["fit_variant"] == variant)
            & (rho_summary["system"] == "ALL")
        ]
        if not rho.empty:
            r = rho.iloc[0]
            gates = [
                ("evidence_margin_M", r["evidence_margin_M"], "> 0", r["evidence_margin_M"] > 0),
                ("balanced_accuracy", r["balanced_accuracy"], ">= 0.75", r["balanced_accuracy"] >= 0.75),
                ("brier_improvement", r["brier_improvement"], "> 0", r["brier_improvement"] > 0),
                ("log_score_improvement", r["log_score_improvement"], "> 0", r["log_score_improvement"] > 0),
                ("mean_rho_gap", r["mean_rho_R1"] - r["mean_rho_R0"], "> 0.30", (r["mean_rho_R1"] - r["mean_rho_R0"]) > 0.30),
            ]
            for metric, value, threshold, ok in gates:
                rows.append(
                    {
                        "fit_variant": variant,
                        "validation_scope": validation_scope,
                        "pillar": "rho_recovery",
                        "metric": metric,
                        "value": float(value),
                        "threshold": threshold,
                        "status": "pass" if ok else "fail",
                        "notes": "",
                    }
                )
        ppc = ppc_summary[
            (ppc_summary["fit_variant"] == variant)
            & (ppc_summary["system"] == "ALL")
        ]
        if not ppc.empty:
            r = ppc.iloc[0]
            gates = [
                ("mean_RPS_improvement", r["mean_RPS_improvement"], "> 0", r["mean_RPS_improvement"] > 0),
                ("weighted_mean_cell_TV", r["weighted_mean_cell_TV"], "<= 0.25", r["weighted_mean_cell_TV"] <= 0.25),
                ("median_cell_TV", r["median_cell_TV"], "<= 0.25", r["median_cell_TV"] <= 0.25),
                ("q90_cell_TV", r["q90_cell_TV"], "<= 0.50", r["q90_cell_TV"] <= 0.50),
                ("fraction_cell_TV_gt_0p5", r["fraction_cell_TV_gt_0p5"], "<= 0.10", r["fraction_cell_TV_gt_0p5"] <= 0.10),
                ("posterior_predictive_tv_p_value", r["posterior_predictive_tv_p_value"], "in [0.05, 0.95]", 0.05 <= r["posterior_predictive_tv_p_value"] <= 0.95),
            ]
            for metric, value, threshold, ok in gates:
                rows.append(
                    {
                        "fit_variant": variant,
                        "validation_scope": validation_scope,
                        "pillar": "posterior_predictive",
                        "metric": metric,
                        "value": float(value),
                        "threshold": threshold,
                        "status": "pass" if ok else "fail",
                        "notes": "",
                    }
                )
        loo = loo_summary[loo_summary["fit_variant"] == variant]
        if not loo.empty:
            r = loo.iloc[0]
            gates = [
                ("frac_pareto_k_gt_0p7", r["frac_pareto_k_gt_0p7"], "<= 0.05", r["frac_pareto_k_gt_0p7"] <= 0.05),
                ("frac_pareto_k_gt_1p0", r["frac_pareto_k_gt_1p0"], "== 0", r["frac_pareto_k_gt_1p0"] == 0),
                ("max_pareto_k", r["max_pareto_k"], "< 1.0", r["max_pareto_k"] < 1.0),
            ]
            for metric, value, threshold, ok in gates:
                rows.append(
                    {
                        "fit_variant": variant,
                        "validation_scope": validation_scope,
                        "pillar": "loo_lppd",
                        "metric": metric,
                        "value": float(value),
                        "threshold": threshold,
                        "status": "pass" if ok else "fail",
                        "notes": "partial" if bool(r.get("loo_partial_flag", False)) else "",
                    }
                )
        sampler = sampler_summary[sampler_summary["fit_variant"] == variant]
        if not sampler.empty:
            rates = {
                "fit_pass_rate_no_divergences": float((sampler["n_divergences"] == 0).mean()),
                "fit_pass_rate_rhat": float((sampler["max_rhat"] <= 1.01).mean()),
                "fit_pass_rate_ess_bulk": float((sampler["min_ess_bulk"] >= 400).mean()),
                "fit_pass_rate_ess_tail": float((sampler["min_ess_tail"] >= 200).mean()),
                "fit_pass_rate_tree_depth": float((sampler["max_tree_depth_hit_rate"] <= 0.01).mean()),
            }
            for metric, value in rates.items():
                rows.append(
                    {
                        "fit_variant": variant,
                        "validation_scope": validation_scope,
                        "pillar": "sampler_health",
                        "metric": metric,
                        "value": value,
                        "threshold": ">= 0.90",
                        "status": "pass" if value >= 0.90 else "fail",
                        "notes": "",
                    }
                )
    return pd.DataFrame(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def jsonable_config(config: ModelConfig) -> dict[str, Any]:
    data = asdict(config)
    if data.get("TARGETED_OVERRIDE_NODE_KEYS") is not None:
        data["TARGETED_OVERRIDE_NODE_KEYS"] = list(data["TARGETED_OVERRIDE_NODE_KEYS"])
    return data
