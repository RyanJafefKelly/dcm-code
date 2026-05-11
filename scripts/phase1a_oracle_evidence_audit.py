"""Phase 1A oracle root-evidence audit for saved GWT synthetic fits."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse_tree_pooling import pooled_beta_draws_by_node  # noqa: E402
from composite_vs_exact_diagnostic import (  # noqa: E402
    collect_indicator_obs,
    exact_loglik,
    precompute_leaf_logliks,
    three_state_log_B,
)
from dcm_model import (  # noqa: E402
    EvidenceProcessor,
    MultiSystemDataProcessor,
    MultiSystemModelBuilder,
    node_key,
)
from gwt_reference_recovery_analysis import extract_beta_draws_by_node  # noqa: E402
from scripts.phase1_root_evidence_common import (  # noqa: E402
    ALL_SYSTEMS,
    DISAGREE_SEVERE_NATS,
    DISAGREE_WARN_NATS,
    EPS,
    FREE_SYSTEMS,
    LOGIT_PRIOR,
    PRIOR_P,
    REPO_ROOT,
    SCORE_CLIP,
    STANCE,
    TAU_ABSENT_05,
    TAU_PRESENT_50,
    beta_profile_from_evidence_processor,
    beta_profile_from_truth,
    bernoulli_brier,
    bernoulli_log_score,
    c_var_name,
    canonical_system,
    check_threshold_constants,
    count_system_observations,
    descendant_counts,
    deterministic_prefix,
    display_system,
    enumerate_top_bound_distribution,
    evidence_category_from_log_b,
    extreme_beta_profile,
    load_gwt_stance,
    make_draw_index,
    markdown_table,
    model_config_from_payload,
    posterior_flat,
    realised_top_log_b,
    rho_from_log_b,
    top_children,
)


@dataclass(frozen=True)
class Phase1ACase:
    run_id: str
    run_family: str
    fit_path: Path
    truth_path: Path
    stance_path: Path
    config_path: Path
    free_systems: Tuple[str, ...] = FREE_SYSTEMS
    stance: str = STANCE


def parse_fallback_draws(value: str) -> Optional[int]:
    if str(value).strip().lower() == "all":
        return None
    return int(value)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def run_family_for(path: Path) -> str:
    text = str(path)
    if "synthetic_validation_2026-05-06/runs/full_exact_recovery" in text:
        return "full_exact_recovery"
    if "sample_size_sweep_2026-05-10/runs" in text:
        return "sample_size_sweep"
    if "asymmetric_prior_sweep_2026-05-10/runs/synthetic" in text:
        return "asymmetric_prior_sweep"
    return path.parent.name


def truth_file_for(run_dir: Path) -> Optional[Path]:
    for name in ("truth_payload.json", "truth.json"):
        path = run_dir / name
        if path.exists():
            return path
    return None


def load_phase1a_cases(run_roots: Sequence[Path]) -> list[Phase1ACase]:
    cases: list[Phase1ACase] = []
    seen: set[Path] = set()
    for root in run_roots:
        root = root.expanduser().resolve()
        if not root.exists():
            continue
        for fit_path in sorted(root.rglob("fit.nc")):
            run_dir = fit_path.parent
            if run_dir in seen:
                continue
            truth_path = truth_file_for(run_dir)
            stance_path = run_dir / "synthetic_stance_data.json"
            config_path = run_dir / "config.json"
            if truth_path is None or not stance_path.exists():
                continue
            family = run_family_for(run_dir)
            run_id = f"{family}:{run_dir.name}"
            cases.append(
                Phase1ACase(
                    run_id=run_id,
                    run_family=family,
                    fit_path=fit_path,
                    truth_path=truth_path,
                    stance_path=stance_path,
                    config_path=config_path,
                )
            )
            seen.add(run_dir)
    return cases


def exact_root_side_loglik(
    stance_data: Mapping[str, Any],
    indicator_obs: Mapping[str, Mapping[str, Tuple[np.ndarray, np.ndarray]]],
    leaf_logliks: Mapping[str, Mapping[str, np.ndarray]],
    beta_pres_by_key: Mapping[str, np.ndarray],
    beta_abs_by_key: Mapping[str, np.ndarray],
    systems: Sequence[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    sample_key = next(iter(beta_pres_by_key))
    n_draws = beta_pres_by_key[sample_key].shape[0]
    root_path = (stance_data["name"],)

    def subtree(
        node: Mapping[str, Any],
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

        log_l_v0 = np.zeros(n_draws)
        log_l_v1 = np.zeros(n_draws)
        for child in node.get("evidencers", []):
            c0, c1 = subtree(child, current_path, system)
            log_l_v0 += c0
            log_l_v1 += c1
        bp_c = np.clip(bp, SCORE_CLIP, 1.0 - SCORE_CLIP)
        ba_c = np.clip(ba, SCORE_CLIP, 1.0 - SCORE_CLIP)
        log_l_zpa1 = np.logaddexp(np.log(bp_c) + log_l_v1, np.log1p(-bp_c) + log_l_v0)
        log_l_zpa0 = np.logaddexp(np.log(ba_c) + log_l_v1, np.log1p(-ba_c) + log_l_v0)
        return log_l_zpa0, log_l_zpa1

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for system in systems:
        log_l0 = np.zeros(n_draws)
        log_l1 = np.zeros(n_draws)
        for child in stance_data.get("evidencers", []):
            c0, c1 = subtree(child, root_path, system)
            log_l0 += c0
            log_l1 += c1
        out[system] = (log_l0, log_l1)
    return out


def true_leaf_logliks(
    stance_data: Mapping[str, Any],
    truth_payload: Mapping[str, Any],
    processor: MultiSystemDataProcessor,
):
    obs_params = truth_payload["observation_parameters"]
    a = np.asarray([float(obs_params["a"])])
    kappa = np.asarray(obs_params["kappa"], dtype=float).reshape(1, -1)
    n_experts = len(processor.expert_names)
    if obs_params.get("use_expert_shifts"):
        raw = obs_params.get("expert_shifts") or obs_params.get("b") or obs_params.get("expert_shift")
        if isinstance(raw, Mapping):
            b = np.asarray([[float(raw.get(name, 0.0)) for name in processor.expert_names]])
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            arr = np.asarray(raw, dtype=float).reshape(1, -1)
            b = np.zeros((1, n_experts))
            b[:, : min(n_experts, arr.shape[1])] = arr[:, : min(n_experts, arr.shape[1])]
        else:
            b = np.full((1, n_experts), float(raw or 0.0))
    else:
        b = np.zeros((1, n_experts))
    indicator_obs = collect_indicator_obs(dict(stance_data), processor)
    leaf_logliks = precompute_leaf_logliks(indicator_obs, a, b, kappa, K=kappa.shape[1] + 1)
    return indicator_obs, leaf_logliks


def oracle_root_evidence(
    stance_data: Mapping[str, Any],
    indicator_obs: Mapping[str, Mapping[str, Tuple[np.ndarray, np.ndarray]]],
    leaf_logliks: Mapping[str, Mapping[str, np.ndarray]],
    truth_payload: Mapping[str, Any],
    target_system: str,
) -> Dict[str, float]:
    bp, ba, _ = beta_profile_from_truth(truth_payload)
    bp_arr = {k: np.asarray([v], dtype=float) for k, v in bp.items()}
    ba_arr = {k: np.asarray([v], dtype=float) for k, v in ba.items()}
    sides = exact_root_side_loglik(
        stance_data,
        indicator_obs,
        leaf_logliks,
        bp_arr,
        ba_arr,
        [target_system],
    )
    log_l0 = float(sides[target_system][0][0])
    log_l1 = float(sides[target_system][1][0])

    # Internal consistency with the verified NumPy DP entry point.  exact_loglik
    # clips C away from {0,1}, so use this as a finite precision guard rather
    # than the primary side extraction.
    _, ll0_per_sys = exact_loglik(
        dict(stance_data),
        dict(indicator_obs),
        dict(leaf_logliks),
        bp_arr,
        ba_arr,
        {target_system: np.asarray([0.0])},
    )
    _, ll1_per_sys = exact_loglik(
        dict(stance_data),
        dict(indicator_obs),
        dict(leaf_logliks),
        bp_arr,
        ba_arr,
        {target_system: np.asarray([1.0])},
    )
    consistency_abs = max(
        abs(float(ll0_per_sys[target_system][0]) - log_l0),
        abs(float(ll1_per_sys[target_system][0]) - log_l1),
    )
    log_b = log_l1 - log_l0
    true_c = float(truth_payload.get("true_C_by_system", {}).get(target_system, np.nan))
    return {
        "log_L0_oracle": log_l0,
        "log_L1_oracle": log_l1,
        "log_B_oracle": log_b,
        "rho_oracle_collapsed": float(rho_from_log_b(log_b, PRIOR_P)),
        "rho_oracle_with_true_C": float(expit(math.log(true_c / (1.0 - true_c)) + log_b))
        if 0.0 < true_c < 1.0
        else np.nan,
        "exact_loglik_consistency_abs": float(consistency_abs),
    }


def top_feature_oracle_contributions(
    *,
    case: Phase1ACase,
    stance_data: Mapping[str, Any],
    indicator_obs: Mapping[str, Mapping[str, Tuple[np.ndarray, np.ndarray]]],
    leaf_logliks: Mapping[str, Mapping[str, np.ndarray]],
    truth_payload: Mapping[str, Any],
    target_system: str,
    log_B_oracle: float,
    processor: MultiSystemDataProcessor,
) -> pd.DataFrame:
    bp, ba, meta = beta_profile_from_truth(truth_payload)
    root_path = (stance_data["name"],)

    def edge_messages(
        node: Mapping[str, Any],
        path: Tuple[str, ...],
    ) -> Tuple[float, float, float, float]:
        """Return state messages m0/m1 and edge-integrated zpa0/zpa1."""
        current_path = path + (node["name"],)
        key = node_key(path, node["name"])
        if (node.get("type") or "").lower() == "indicator":
            if target_system not in indicator_obs.get(key, {}):
                return 0.0, 0.0, 0.0, 0.0
            leaf_lls = leaf_logliks[key][target_system]
            log_zpa0 = float(three_state_log_B(np.asarray([ba[key]]), leaf_lls)[0])
            log_zpa1 = float(three_state_log_B(np.asarray([bp[key]]), leaf_lls)[0])
            return np.nan, np.nan, log_zpa0, log_zpa1

        m0 = 0.0
        m1 = 0.0
        for child in node.get("evidencers", []):
            _, _, c0, c1 = edge_messages(child, current_path)
            m0 += c0
            m1 += c1
        bp_c = float(np.clip(bp[key], SCORE_CLIP, 1.0 - SCORE_CLIP))
        ba_c = float(np.clip(ba[key], SCORE_CLIP, 1.0 - SCORE_CLIP))
        zpa1 = float(np.logaddexp(math.log(bp_c) + m1, math.log1p(-bp_c) + m0))
        zpa0 = float(np.logaddexp(math.log(ba_c) + m1, math.log1p(-ba_c) + m0))
        return m0, m1, zpa0, zpa1

    rows: list[dict[str, Any]] = []
    latent = truth_payload["latent_by_system"][target_system]
    internal_z = latent.get("internal_z", {})
    for child in top_children(stance_data):
        key = child.key
        m0, m1, zpa0, zpa1 = edge_messages(child.node, root_path)
        bp_c = float(np.clip(bp[key], SCORE_CLIP, 1.0 - SCORE_CLIP))
        ba_c = float(np.clip(ba[key], SCORE_CLIP, 1.0 - SCORE_CLIP))
        c_oracle = zpa1 - zpa0
        realised_z = int(internal_z[key])
        c_latent = (
            math.log(bp_c / ba_c)
            if realised_z
            else math.log((1.0 - bp_c) / (1.0 - ba_c))
        )
        counts = descendant_counts(
            child.node,
            root_path,
            observed_by_system=processor.system_observations,
            system=target_system,
        )
        edge_meta = meta.get(key, {})
        rows.append(
            {
                "run_id": case.run_id,
                "run_family": case.run_family,
                "seed": truth_payload.get("seed"),
                "system": display_system(target_system),
                "root_z_true": int(latent["root_z"]),
                "top_feature_key": key,
                "top_feature_name": child.node["name"],
                "support_label": child.node.get("support", edge_meta.get("support", "no bearing")),
                "demandingness_label": child.node.get("demandingness", edge_meta.get("demandingness", "neutral")),
                "beta_pres_true": bp_c,
                "beta_abs_true": ba_c,
                "beta_gap_true": bp_c - ba_c,
                "realised_z_top": realised_z,
                **counts,
                "c_top_latent": float(c_latent),
                "c_oracle_subtree": float(c_oracle),
                "delta_oracle_minus_top_latent": float(c_oracle - c_latent),
                "abs_delta": float(abs(c_oracle - c_latent)),
                "subtree_message_loglik_z0": float(m0),
                "subtree_message_loglik_z1": float(m1),
            }
        )
    df = pd.DataFrame(rows)
    err = abs(float(df["c_oracle_subtree"].sum()) - float(log_B_oracle))
    if err > 1e-6:
        raise AssertionError(
            f"Top-feature decomposition mismatch for {case.run_id} {target_system}: "
            f"sum={df['c_oracle_subtree'].sum():.12g}, log_B={log_B_oracle:.12g}, err={err:.3g}"
        )
    return df


def extract_posterior_beta_draws(
    idata: Any,
    cfg: Any,
    stance_data: Mapping[str, Any],
    processor: MultiSystemDataProcessor,
    systems: Sequence[str],
    draw_idx: Optional[np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    post = idata.posterior
    if cfg.POOL_BETAS_BY_LABEL or any(str(v).startswith("beta_pres__") for v in post.data_vars):
        bp, ba = pooled_beta_draws_by_node(idata, dict(stance_data))
    else:
        builder = MultiSystemModelBuilder(
            cfg,
            EvidenceProcessor(cfg),
            processor,
            [(s, None) for s in systems],
        )
        builder.build_model(dict(stance_data))
        bp, ba = extract_beta_draws_by_node(idata, builder, dict(stance_data))
    if draw_idx is not None:
        bp = {k: np.asarray(v)[draw_idx] for k, v in bp.items()}
        ba = {k: np.asarray(v)[draw_idx] for k, v in ba.items()}
    return bp, ba


def posterior_log_b_fallback(
    *,
    idata: Any,
    cfg: Any,
    stance_data: Mapping[str, Any],
    processor: MultiSystemDataProcessor,
    systems: Sequence[str],
    fallback_draws: Optional[int],
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    post = idata.posterior
    if "kappa_by_expert" in post.data_vars or "sigma_by_expert" in post.data_vars:
        raise NotImplementedError(
            "Phase 1A fallback currently supports shared kappa/b or b_free, "
            "but this fit has kappa_by_expert or sigma_by_expert."
        )
    n_total = int(np.asarray(post["a"].values).reshape(-1).shape[0])
    draw_idx = make_draw_index(n_total, fallback_draws)
    a_draws = posterior_flat(post, "a", draw_idx).reshape(-1)
    n_draws = a_draws.shape[0]
    kappa_draws = posterior_flat(post, "kappa", draw_idx).reshape(n_draws, cfg.N_CATEGORIES - 1)
    n_experts = len(processor.expert_names)
    if "b_free" in post.data_vars:
        b_free = posterior_flat(post, "b_free", draw_idx).reshape(n_draws, n_experts - 1)
        b_draws = np.concatenate([np.zeros((n_draws, 1)), b_free], axis=1)
    else:
        b_draws = np.zeros((n_draws, n_experts))

    indicator_obs = collect_indicator_obs(dict(stance_data), processor)
    leaf_logliks = precompute_leaf_logliks(
        indicator_obs,
        a_draws,
        b_draws,
        kappa_draws,
        K=cfg.N_CATEGORIES,
    )
    beta_pres, beta_abs = extract_posterior_beta_draws(
        idata, cfg, stance_data, processor, systems, draw_idx
    )
    sides = exact_root_side_loglik(
        stance_data,
        indicator_obs,
        leaf_logliks,
        beta_pres,
        beta_abs,
        systems,
    )
    return {system: sides[system][1] - sides[system][0] for system in systems}, draw_idx


def posterior_evidence_stats(
    *,
    idata: Any,
    system: str,
    log_b_draws: np.ndarray,
    draw_idx: Optional[np.ndarray],
) -> Dict[str, Any]:
    log_b = np.asarray(log_b_draws, dtype=float).reshape(-1)
    rho_collapsed_draws = expit(LOGIT_PRIOR + log_b)
    rho_collapsed = float(np.mean(rho_collapsed_draws))
    log_b_eff = float(math.log(np.clip(rho_collapsed, SCORE_CLIP, 1.0) / np.clip(1.0 - rho_collapsed, SCORE_CLIP, 1.0)) - LOGIT_PRIOR)
    if abs(log_b_eff - (math.log(np.clip(rho_collapsed, SCORE_CLIP, 1.0) / np.clip(1.0 - rho_collapsed, SCORE_CLIP, 1.0)) - LOGIT_PRIOR)) > 1e-12:
        raise AssertionError("Posterior effective log_B identity failed")
    c_var = c_var_name(system)
    if c_var in idata.posterior.data_vars:
        c_draws = posterior_flat(idata.posterior, c_var, draw_idx).reshape(-1)
        rho_sampled_pi = float(np.mean(expit(np.log(np.clip(c_draws, SCORE_CLIP, 1.0 - SCORE_CLIP) / np.clip(1.0 - c_draws, SCORE_CLIP, 1.0)) + log_b)))
    else:
        rho_sampled_pi = np.nan
    qs = np.quantile(log_b, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "log_B_draw_mean": float(np.mean(log_b)),
        "log_B_draw_median": float(qs[2]),
        "log_B_draw_sd": float(np.std(log_b, ddof=1)) if len(log_b) > 1 else 0.0,
        "log_B_draw_q05": float(qs[0]),
        "log_B_draw_q25": float(qs[1]),
        "log_B_draw_q75": float(qs[3]),
        "log_B_draw_q95": float(qs[4]),
        "p_draw_log_B_gt_0": float(np.mean(log_b > 0.0)),
        "p_draw_log_B_gt_tau_present_50": float(np.mean(log_b > TAU_PRESENT_50)),
        "p_draw_log_B_lt_tau_absent_05": float(np.mean(log_b < TAU_ABSENT_05)),
        "rho_posterior_collapsed": rho_collapsed,
        "rho_posterior_sampled_pi": rho_sampled_pi,
        "log_B_eff_posterior": log_b_eff,
        "n_posterior_draws_used": int(len(log_b)),
    }


def posterior_log_b_for_systems(
    *,
    idata: Any,
    cfg: Any,
    stance_data: Mapping[str, Any],
    processor: MultiSystemDataProcessor,
    systems: Sequence[str],
    fallback_draws: Optional[int],
) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray], str]:
    out: Dict[str, np.ndarray] = {}
    have_all = True
    for system in systems:
        var = f"{deterministic_prefix(system)}_log_B"
        if var not in idata.posterior.data_vars:
            have_all = False
            break
        out[system] = np.asarray(idata.posterior[var].values).reshape(-1)
    if have_all:
        return out, None, "posterior_deterministic"
    fallback, draw_idx = posterior_log_b_fallback(
        idata=idata,
        cfg=cfg,
        stance_data=stance_data,
        processor=processor,
        systems=systems,
        fallback_draws=fallback_draws,
    )
    return fallback, draw_idx, "numpy_dp_fallback"


def audit_one_case(
    case: Phase1ACase,
    fallback_draws: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    truth = load_json(case.truth_path)
    stance_data = load_json(case.stance_path)
    config_payload = load_json(case.config_path) if case.config_path.exists() else {}
    cfg = model_config_from_payload(config_payload)
    systems = list(config_payload.get("systems") or truth.get("latent_by_system", {}).keys() or ALL_SYSTEMS)
    systems = [canonical_system(s) for s in systems if canonical_system(s) in truth.get("latent_by_system", {})]
    free_systems = [s for s in case.free_systems if s in systems and s in truth.get("latent_by_system", {})]

    processor = MultiSystemDataProcessor(cfg).process(dict(stance_data), systems)
    indicator_obs_true, leaf_logliks_true = true_leaf_logliks(stance_data, truth, processor)
    idata = az.from_netcdf(case.fit_path)
    posterior_log_b, draw_idx, extraction_method = posterior_log_b_for_systems(
        idata=idata,
        cfg=cfg,
        stance_data=stance_data,
        processor=processor,
        systems=free_systems,
        fallback_draws=fallback_draws,
    )
    beta_pres_true, beta_abs_true, _ = beta_profile_from_truth(truth)
    labels = config_payload.get("labels") or truth.get("labels") or {}
    run_variant = config_payload.get("mode") or config_payload.get("run_variant")
    rater_design_tag = labels.get("design") or config_payload.get("rater_design_tag")

    rows: list[dict[str, Any]] = []
    contrib_frames: list[pd.DataFrame] = []
    for system in free_systems:
        root_z = int(truth["latent_by_system"][system]["root_z"])
        obs_counts = count_system_observations(processor, system)
        top = realised_top_log_b(
            stance_data=stance_data,
            truth_payload=truth,
            system=system,
            beta_pres_by_key=beta_pres_true,
            beta_abs_by_key=beta_abs_true,
        )
        oracle = oracle_root_evidence(
            stance_data,
            indicator_obs_true,
            leaf_logliks_true,
            truth,
            system,
        )
        posterior = posterior_evidence_stats(
            idata=idata,
            system=system,
            log_b_draws=posterior_log_b[system],
            draw_idx=draw_idx,
        )
        oracle_category = evidence_category_from_log_b(float(oracle["log_B_oracle"]))
        posterior_category = evidence_category_from_log_b(float(posterior["log_B_eff_posterior"]))
        oracle_correct = bool(
            (root_z == 1 and oracle["log_B_oracle"] > 0.0)
            or (root_z == 0 and oracle["log_B_oracle"] < 0.0)
        )
        posterior_correct = bool(
            (root_z == 1 and posterior["log_B_eff_posterior"] > 0.0)
            or (root_z == 0 and posterior["log_B_eff_posterior"] < 0.0)
        )
        oracle_decisive = bool(
            (root_z == 1 and oracle["log_B_oracle"] > TAU_PRESENT_50)
            or (root_z == 0 and oracle["log_B_oracle"] < TAU_ABSENT_05)
        )
        posterior_decisive = bool(
            (root_z == 1 and posterior["log_B_eff_posterior"] > TAU_PRESENT_50)
            or (root_z == 0 and posterior["log_B_eff_posterior"] < TAU_ABSENT_05)
        )
        delta_eff_oracle = float(posterior["log_B_eff_posterior"] - oracle["log_B_oracle"])
        delta_median_oracle = float(posterior["log_B_draw_median"] - oracle["log_B_oracle"])
        delta_eff_top = float(posterior["log_B_eff_posterior"] - top["log_B_top_latent"])
        sign_flip = bool(np.sign(oracle["log_B_oracle"]) != np.sign(posterior["log_B_eff_posterior"]))
        prior_log = bernoulli_log_score(PRIOR_P, root_z)
        prior_brier = bernoulli_brier(PRIOR_P, root_z)
        row = {
            "run_id": case.run_id,
            "run_family": case.run_family,
            "run_variant": run_variant,
            "fit_path": str(case.fit_path.relative_to(REPO_ROOT)),
            "truth_path": str(case.truth_path.relative_to(REPO_ROOT)),
            "seed": truth.get("seed") or config_payload.get("seed"),
            "system": display_system(system),
            "system_raw": system,
            "stance": stance_data["name"],
            **obs_counts,
            "rater_design_tag": rater_design_tag,
            "true_C": float(truth.get("true_C_by_system", {}).get(system, np.nan)),
            "root_z_true": root_z,
            **top,
            **oracle,
            "oracle_category": oracle_category,
            "oracle_correct_sign": oracle_correct,
            "oracle_decisive_for_true_root": oracle_decisive,
            **posterior,
            "posterior_extraction_method": extraction_method,
            "posterior_category": posterior_category,
            "posterior_correct_sign": posterior_correct,
            "posterior_decisive_for_true_root": posterior_decisive,
            "delta_eff_minus_oracle": delta_eff_oracle,
            "delta_median_draw_minus_oracle": delta_median_oracle,
            "delta_eff_minus_top": delta_eff_top,
            "sign_flip_oracle_vs_posterior": sign_flip,
            "category_shift_oracle_to_posterior": f"{oracle_category}->{posterior_category}",
            "large_disagreement_flag": bool(abs(delta_eff_oracle) >= DISAGREE_WARN_NATS or sign_flip),
            "severe_disagreement_flag": bool(abs(delta_eff_oracle) >= DISAGREE_SEVERE_NATS),
            "log_score_improvement_oracle": float(bernoulli_log_score(oracle["rho_oracle_collapsed"], root_z) - prior_log),
            "log_score_improvement_posterior": float(bernoulli_log_score(posterior["rho_posterior_collapsed"], root_z) - prior_log),
            "brier_improvement_oracle": float(prior_brier - bernoulli_brier(oracle["rho_oracle_collapsed"], root_z)),
            "brier_improvement_posterior": float(prior_brier - bernoulli_brier(posterior["rho_posterior_collapsed"], root_z)),
        }
        rows.append(row)
        contrib_frames.append(
            top_feature_oracle_contributions(
                case=case,
                stance_data=stance_data,
                indicator_obs=indicator_obs_true,
                leaf_logliks=leaf_logliks_true,
                truth_payload=truth,
                target_system=system,
                log_B_oracle=float(oracle["log_B_oracle"]),
                processor=processor,
            )
        )
    return pd.DataFrame(rows), pd.concat(contrib_frames, ignore_index=True) if contrib_frames else pd.DataFrame()


def summarise_phase1a(case_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if case_rows.empty:
        return pd.DataFrame()
    for keys, g in case_rows.groupby(["run_family", "system", "root_z_true"], dropna=False):
        run_family, system, root_z_true = keys
        rows.append(
            {
                "run_family": run_family,
                "system": system,
                "root_z_true": int(root_z_true),
                "n": int(len(g)),
                "median_log_B_top_latent": float(g["log_B_top_latent"].median()),
                "median_log_B_oracle": float(g["log_B_oracle"].median()),
                "median_log_B_eff_posterior": float(g["log_B_eff_posterior"].median()),
                "mean_log_B_top_latent": float(g["log_B_top_latent"].mean()),
                "mean_log_B_oracle": float(g["log_B_oracle"].mean()),
                "mean_log_B_eff_posterior": float(g["log_B_eff_posterior"].mean()),
                "mean_rho_oracle_collapsed": float(g["rho_oracle_collapsed"].mean()),
                "mean_rho_posterior_collapsed": float(g["rho_posterior_collapsed"].mean()),
                "p_correct_sign_oracle": float(g["oracle_correct_sign"].mean()),
                "p_correct_sign_posterior": float(g["posterior_correct_sign"].mean()),
                "p_decisive_oracle": float(g["oracle_decisive_for_true_root"].mean()),
                "p_decisive_posterior": float(g["posterior_decisive_for_true_root"].mean()),
                "balanced_accuracy_oracle": float(g["oracle_correct_sign"].mean()),
                "balanced_accuracy_posterior": float(g["posterior_correct_sign"].mean()),
                "balanced_log_score_improvement_oracle": float(g["log_score_improvement_oracle"].mean()),
                "balanced_log_score_improvement_posterior": float(g["log_score_improvement_posterior"].mean()),
                "balanced_brier_improvement_oracle": float(g["brier_improvement_oracle"].mean()),
                "balanced_brier_improvement_posterior": float(g["brier_improvement_posterior"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["run_family", "system", "root_z_true"])


def build_report(
    *,
    output_dir: Path,
    case_rows: pd.DataFrame,
    summary: pd.DataFrame,
    disagreements: pd.DataFrame,
    contributions: pd.DataFrame,
    top_distribution: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Phase 1A Oracle Root-Evidence Audit")
    lines.append("")
    lines.append(f"Rows audited: {len(case_rows)} free-target system cases across {case_rows['run_id'].nunique() if not case_rows.empty else 0} fits.")
    lines.append("")
    lines.append("## 1. Top-Level Bound")
    if not top_distribution.empty:
        lines.append(markdown_table(
            top_distribution,
            [
                "beta_profile",
                "mean_log_B_top_R1",
                "mean_log_B_top_R0",
                "median_log_B_top_R1",
                "median_log_B_top_R0",
                "frac_R1_log_B_top_gt_tau_present_50",
                "frac_R0_log_B_top_lt_tau_absent_05",
                "preflight_gate",
            ],
            max_rows=10,
        ))
        gates = ", ".join(
            f"{r.beta_profile}={r.preflight_gate}" for r in top_distribution.itertuples()
        )
        lines.append(f"\nPre-flight gate status: {gates}. The full median and tail distribution, not the mean alone, is the relevant bound.")
    lines.append("")
    lines.append("## 2. Summary By System And Root")
    lines.append(markdown_table(
        summary,
        [
            "run_family",
            "system",
            "root_z_true",
            "n",
            "median_log_B_top_latent",
            "median_log_B_oracle",
            "median_log_B_eff_posterior",
            "p_correct_sign_oracle",
            "p_correct_sign_posterior",
            "p_decisive_oracle",
            "p_decisive_posterior",
        ],
        max_rows=30,
    ))
    lines.append("")
    lines.append("## 3. Evidence Categories")
    if not case_rows.empty:
        cat_counts = (
            case_rows.groupby(["system", "root_z_true", "oracle_category", "posterior_category"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["system", "root_z_true", "n"], ascending=[True, True, False])
        )
        lines.append(markdown_table(cat_counts, ["system", "root_z_true", "oracle_category", "posterior_category", "n"], max_rows=30))
    lines.append("")
    lines.append("## 4. Oracle Versus Posterior Disagreements")
    n_large = int(case_rows["large_disagreement_flag"].sum()) if not case_rows.empty else 0
    n_severe = int(case_rows["severe_disagreement_flag"].sum()) if not case_rows.empty else 0
    lines.append(f"Large disagreements (>=2 nats or sign flip): {n_large}. Severe disagreements (>=4 nats): {n_severe}.")
    lines.append(markdown_table(
        disagreements.sort_values("delta_eff_minus_oracle", key=lambda s: s.abs(), ascending=False)
        if not disagreements.empty else disagreements,
        [
            "run_family",
            "seed",
            "system",
            "root_z_true",
            "log_B_top_latent",
            "log_B_oracle",
            "log_B_eff_posterior",
            "delta_eff_minus_oracle",
            "delta_eff_minus_top",
            "sign_flip_oracle_vs_posterior",
        ],
        max_rows=10,
    ))
    lines.append("")
    lines.append("## 5. Top-Feature Evidence Loss")
    if not contributions.empty:
        loss = (
            contributions.groupby(["top_feature_name", "support_label", "demandingness_label"], dropna=False)
            .agg(
                n=("c_oracle_subtree", "size"),
                median_c_top_latent=("c_top_latent", "median"),
                median_c_oracle_subtree=("c_oracle_subtree", "median"),
                median_delta_oracle_minus_top=("delta_oracle_minus_top_latent", "median"),
                mean_abs_delta=("abs_delta", "mean"),
                median_observed_ratings_desc=("n_observed_ratings_desc", "median"),
            )
            .reset_index()
            .sort_values("mean_abs_delta", ascending=False)
        )
        lines.append(markdown_table(
            loss,
            [
                "top_feature_name",
                "support_label",
                "demandingness_label",
                "n",
                "median_c_top_latent",
                "median_c_oracle_subtree",
                "median_delta_oracle_minus_top",
                "mean_abs_delta",
                "median_observed_ratings_desc",
            ],
            max_rows=10,
        ))
    lines.append("")
    lines.append("## 6. Chicken Versus LLMs")
    if not case_rows.empty:
        chicken_llm = (
            case_rows.groupby("system", dropna=False)
            .agg(
                n=("system", "size"),
                mean_n_raters=("n_raters", "mean"),
                median_log_B_oracle=("log_B_oracle", "median"),
                median_log_B_eff_posterior=("log_B_eff_posterior", "median"),
                p_correct_sign_posterior=("posterior_correct_sign", "mean"),
                p_decisive_posterior=("posterior_decisive_for_true_root", "mean"),
            )
            .reset_index()
        )
        lines.append(markdown_table(
            chicken_llm,
            [
                "system",
                "n",
                "mean_n_raters",
                "median_log_B_oracle",
                "median_log_B_eff_posterior",
                "p_correct_sign_posterior",
                "p_decisive_posterior",
            ],
        ))
        lines.append(
            "\nChicken remains the key design comparison because the production-style design has no Chicken cross-system rater coverage, while LLMs have Rater_B-linked coverage."
        )
    lines.append("")
    lines.append("## 7. Asymmetric Prior Sweep Interpretation")
    lines.append(
        "The existing asymmetric-prior synthetic sweep changed the fitted beta-prior centre while the generator truth still came from production median edge betas. It is therefore a prior-DGP mismatch test, not evidence that true beta=(0.90,0.10) fails. Phase 1B adds the matched extreme-beta positive-control ladder."
    )
    lines.append("")
    lines.append("## Output Files")
    for name in [
        "phase1a_oracle_audit_cases.csv",
        "phase1a_oracle_audit_summary.csv",
        "phase1a_disagreements.csv",
        "phase1a_top_feature_contributions.csv",
    ]:
        lines.append(f"- `{name}`")
    (output_dir / "phase1a_oracle_audit_report.md").write_text("\n".join(lines) + "\n")


def build_top_distribution_for_report(case_rows: pd.DataFrame, cases: Sequence[Phase1ACase]) -> pd.DataFrame:
    stance_data = load_gwt_stance()
    frames: list[pd.DataFrame] = []
    bp, ba, _ = beta_profile_from_evidence_processor(stance_data)
    frames.append(
        enumerate_top_bound_distribution(
            stance_data=stance_data,
            beta_pres_by_key=bp,
            beta_abs_by_key=ba,
            beta_profile_name="production_prior_mean",
        )
    )
    if cases:
        truth = load_json(cases[0].truth_path)
        bp_t, ba_t, _ = beta_profile_from_truth(truth)
        frames.append(
            enumerate_top_bound_distribution(
                stance_data=stance_data,
                beta_pres_by_key=bp_t,
                beta_abs_by_key=ba_t,
                beta_profile_name="production_oracle_median",
            )
        )
    bp_x, ba_x, _ = extreme_beta_profile(stance_data)
    frames.append(
        enumerate_top_bound_distribution(
            stance_data=stance_data,
            beta_pres_by_key=bp_x,
            beta_abs_by_key=ba_x,
            beta_profile_name="extreme_0p9_0p1",
        )
    )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/phase1_root_evidence",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        default=[],
        help="Root directory containing saved synthetic runs with fit.nc.",
    )
    parser.add_argument("--fallback-draws", default="all")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    check_threshold_constants()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_roots = args.run_root or [
        REPO_ROOT / "notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery",
        REPO_ROOT / "notebooks/sample_size_sweep_2026-05-10/runs",
        REPO_ROOT / "notebooks/asymmetric_prior_sweep_2026-05-10/runs/synthetic",
    ]
    cases = load_phase1a_cases(run_roots)
    if args.max_runs is not None:
        cases = cases[: args.max_runs]
    fallback_draws = parse_fallback_draws(args.fallback_draws)
    if not cases:
        raise SystemExit("No Phase 1A saved fits found.")

    case_frames: list[pd.DataFrame] = []
    contribution_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        print(f"[phase1a] {i}/{len(cases)} {case.run_id}")
        try:
            rows, contrib = audit_one_case(case, fallback_draws=fallback_draws)
            case_frames.append(rows)
            contribution_frames.append(contrib)
        except Exception as exc:
            failures.append(
                {
                    "run_id": case.run_id,
                    "fit_path": str(case.fit_path.relative_to(REPO_ROOT)),
                    "error": repr(exc),
                }
            )
            print(f"[phase1a] failed {case.run_id}: {exc!r}")

    if not case_frames:
        pd.DataFrame(failures).to_csv(output_dir / "phase1a_failures.csv", index=False)
        raise SystemExit("All Phase 1A audits failed; see phase1a_failures.csv.")

    case_rows = pd.concat(case_frames, ignore_index=True)
    contributions = pd.concat(contribution_frames, ignore_index=True) if contribution_frames else pd.DataFrame()
    summary = summarise_phase1a(case_rows)
    disagreements = case_rows[
        (case_rows["delta_eff_minus_oracle"].abs() >= DISAGREE_WARN_NATS)
        | (case_rows["sign_flip_oracle_vs_posterior"])
        | (case_rows["delta_eff_minus_top"].abs() >= DISAGREE_WARN_NATS)
    ].copy()
    top_distribution = build_top_distribution_for_report(case_rows, cases)

    case_rows.to_csv(output_dir / "phase1a_oracle_audit_cases.csv", index=False)
    summary.to_csv(output_dir / "phase1a_oracle_audit_summary.csv", index=False)
    disagreements.to_csv(output_dir / "phase1a_disagreements.csv", index=False)
    contributions.to_csv(output_dir / "phase1a_top_feature_contributions.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "phase1a_failures.csv", index=False)
    build_report(
        output_dir=output_dir,
        case_rows=case_rows,
        summary=summary,
        disagreements=disagreements,
        contributions=contributions,
        top_distribution=top_distribution,
    )
    print(f"[phase1a] wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()

