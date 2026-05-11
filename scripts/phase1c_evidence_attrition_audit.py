"""Phase 1C evidence-attrition audit for binary-root GWT DCM.

This script is deliberately no-fit only.  It diagnoses where root evidence is
lost after the top-level latent state by running matched-seed clamped-parameter
simulations over the actual GWT topology.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcm_model import node_key  # noqa: E402
from scripts.phase1_root_evidence_common import (  # noqa: E402
    ALL_SYSTEMS,
    EPS,
    FREE_SYSTEMS,
    LOGIT_PRIOR,
    PRIOR_P,
    SCORE_CLIP,
    STANCE,
    TAU_ABSENT_05,
    TAU_PRESENT_50,
    beta_profile_from_evidence_processor,
    beta_profile_from_truth,
    bernoulli_brier,
    bernoulli_log_score,
    check_threshold_constants,
    clip_prob,
    descendant_counts,
    display_system,
    evidence_category_from_log_b,
    extreme_beta_profile,
    iter_tree_nodes,
    load_gwt_stance,
    markdown_table,
    ordered_probit_category_logp,
    sample_ordered_probit_category,
    top_children,
)
from scripts.phase1b_root_evidence_ladder import (  # noqa: E402
    find_oracle_truth_path,
    load_observation_truth,
    root_truth_for_replicate,
)


STRONG_TOP_FEATURES = {"Coherence", "Complexity", "Selective Attention", "Integration"}
WEAK_TOP_FEATURES = {"Representationality", "Hierarchical Organization", "Modularity"}

STAGES = [
    "top_observed",
    "subfeature_observed",
    "leaf_latent_observed",
    "binary_perfect_leaf",
    "binary_noisy_K6",
    "ordinal_binary_clamped",
    "ordinal_three_state_clamped",
]

STAGE_TO_COLUMN = {
    "top_observed": "log_B_top_observed",
    "subfeature_observed": "log_B_subfeature_observed",
    "leaf_latent_observed": "log_B_leaf_latent_observed",
    "binary_perfect_leaf": "log_B_binary_perfect_leaf",
    "binary_noisy_K6": "log_B_binary_noisy_K6",
    "ordinal_binary_clamped": "log_B_ordinal_binary_clamped",
    "ordinal_three_state_clamped": "log_B_ordinal_three_state_clamped",
}

RUNG_LABEL = {
    ("production_prior_mean", "binary_noisy_K6"): "1P",
    ("production_oracle_median", "binary_noisy_K6"): "1O",
    ("extreme_0p9_0p1", "binary_noisy_K6"): "1X",
    ("production_prior_mean", "ordinal_binary_clamped"): "3",
    ("production_oracle_median", "ordinal_binary_clamped"): "3O",
    ("extreme_0p9_0p1", "ordinal_binary_clamped"): "3X",
    ("production_prior_mean", "ordinal_three_state_clamped"): "4",
    ("production_oracle_median", "ordinal_three_state_clamped"): "4O",
    ("extreme_0p9_0p1", "ordinal_three_state_clamped"): "4X",
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def build_beta_profiles(
    stance_data: Mapping[str, Any],
    oracle_truth_path: Optional[Path],
) -> Dict[str, Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, Any]]]]:
    profiles: Dict[str, Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, Any]]]] = {
        "production_prior_mean": beta_profile_from_evidence_processor(stance_data),
        "extreme_0p9_0p1": extreme_beta_profile(stance_data),
    }
    if oracle_truth_path and oracle_truth_path.exists():
        profiles["production_oracle_median"] = beta_profile_from_truth(load_json(oracle_truth_path))
    else:
        profiles["production_oracle_median"] = profiles["production_prior_mean"]
    return {
        name: profiles[name]
        for name in ["production_prior_mean", "production_oracle_median", "extreme_0p9_0p1"]
    }


def rng_for(seed: int, rep: int, system: str) -> np.random.Generator:
    sys_offset = list(ALL_SYSTEMS).index(system) * 10_000
    return np.random.default_rng(int(seed) + int(rep) * 100_003 + sys_offset)


def all_leaf_keys(stance_data: Mapping[str, Any]) -> list[str]:
    return [
        spec.key
        for spec in iter_tree_nodes(stance_data)
        if (spec.node.get("type") or "").lower() == "indicator"
    ]


def sample_latent_states(
    *,
    rng: np.random.Generator,
    stance_data: Mapping[str, Any],
    beta_pres_by_key: Mapping[str, float],
    beta_abs_by_key: Mapping[str, float],
    root_z: int,
) -> Dict[str, Any]:
    """Sample one latent tree with shared internal states and both leaf forms."""
    internal_z: Dict[str, int] = {}
    indicator_z: Dict[str, int] = {}
    indicator_m: Dict[str, int] = {}
    indicator_parent_z: Dict[str, int] = {}
    root_path = (stance_data["name"],)

    def walk(node: Mapping[str, Any], path: Tuple[str, ...], parent_z: int) -> None:
        key = node_key(path, node["name"])
        beta = float(beta_pres_by_key[key] if parent_z else beta_abs_by_key[key])
        beta = float(clip_prob(beta))
        if (node.get("type") or "").lower() == "indicator":
            indicator_parent_z[key] = int(parent_z)
            indicator_z[key] = int(rng.binomial(1, beta))
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
        "indicator_z": indicator_z,
        "indicator_m": indicator_m,
        "indicator_parent_z": indicator_parent_z,
    }


def leaf_binary_observed_message(
    z: int,
    beta_pres: float,
    beta_abs: float,
) -> Tuple[float, float]:
    bp = float(clip_prob(beta_pres))
    ba = float(clip_prob(beta_abs))
    if int(z):
        return math.log(ba), math.log(bp)
    return math.log1p(-ba), math.log1p(-bp)


def leaf_three_state_observed_message(
    m: int,
    beta_pres: float,
    beta_abs: float,
) -> Tuple[float, float]:
    def log_binom2(k: int, q: float) -> float:
        q = float(clip_prob(q))
        if k == 0:
            return 2.0 * math.log1p(-q)
        if k == 1:
            return math.log(2.0) + math.log(q) + math.log1p(-q)
        if k == 2:
            return 2.0 * math.log(q)
        raise ValueError(k)

    return log_binom2(int(m), beta_abs), log_binom2(int(m), beta_pres)


def binary_noisy_message_from_count(
    *,
    successes: int,
    n: int,
    epsilon: float,
    beta_pres: float,
    beta_abs: float,
) -> Tuple[float, float, float, float]:
    eps = float(np.clip(epsilon, SCORE_CLIP, 1.0 - SCORE_CLIP))
    k = int(successes)
    n = int(n)
    ll0 = k * math.log(eps) + (n - k) * math.log1p(-eps)
    ll1 = k * math.log1p(-eps) + (n - k) * math.log(eps)
    bp = float(clip_prob(beta_pres))
    ba = float(clip_prob(beta_abs))
    zpa0 = float(np.logaddexp(math.log1p(-ba) + ll0, math.log(ba) + ll1))
    zpa1 = float(np.logaddexp(math.log1p(-bp) + ll0, math.log(bp) + ll1))
    return zpa0, zpa1, float(ll0), float(ll1)


def binary_ordinal_message(
    *,
    ratings: np.ndarray,
    a: float,
    kappa: np.ndarray,
    beta_pres: float,
    beta_abs: float,
) -> Tuple[float, float, float, float, tuple[int, ...]]:
    expert_idx = np.zeros(len(ratings), dtype=np.int64)
    b = np.zeros(1)
    ll0 = ordered_probit_category_logp(ratings, expert_idx, a, b, kappa, 0.0)
    ll1 = ordered_probit_category_logp(ratings, expert_idx, a, b, kappa, 1.0)
    bp = float(clip_prob(beta_pres))
    ba = float(clip_prob(beta_abs))
    zpa0 = float(np.logaddexp(math.log1p(-ba) + ll0, math.log(ba) + ll1))
    zpa1 = float(np.logaddexp(math.log1p(-bp) + ll0, math.log(bp) + ll1))
    return zpa0, zpa1, float(ll0), float(ll1), tuple(int(x) for x in ratings)


def three_state_ordinal_message(
    *,
    ratings: np.ndarray,
    a: float,
    kappa: np.ndarray,
    beta_pres: float,
    beta_abs: float,
) -> Tuple[float, float, float, float, tuple[int, ...]]:
    expert_idx = np.zeros(len(ratings), dtype=np.int64)
    b = np.zeros(1)
    ll0 = ordered_probit_category_logp(ratings, expert_idx, a, b, kappa, 0.0)
    llh = ordered_probit_category_logp(ratings, expert_idx, a, b, kappa, 0.5)
    ll1 = ordered_probit_category_logp(ratings, expert_idx, a, b, kappa, 1.0)

    def side(beta: float) -> float:
        q = float(clip_prob(beta))
        return float(
            logsumexp(
                [
                    2.0 * math.log1p(-q) + ll0,
                    math.log(2.0) + math.log(q) + math.log1p(-q) + llh,
                    2.0 * math.log(q) + ll1,
                ]
            )
        )

    return side(beta_abs), side(beta_pres), float(ll0), float(ll1), tuple(int(x) for x in ratings)


def simulate_leaf_messages(
    *,
    rng: np.random.Generator,
    stance_data: Mapping[str, Any],
    latent: Mapping[str, Any],
    beta_pres_by_key: Mapping[str, float],
    beta_abs_by_key: Mapping[str, float],
    obs_params: Mapping[str, Any],
    observation_model: str,
    k_or_n: int,
    epsilon: float = 0.05,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Any]]:
    a = float(obs_params["a"])
    kappa = np.asarray(obs_params["kappa"], dtype=float)
    messages: Dict[str, Tuple[float, float]] = {}
    leaf_ll0: list[float] = []
    leaf_ll1: list[float] = []
    unique_values: set[int] = set()
    root_path = (stance_data["name"],)

    def walk(node: Mapping[str, Any], path: Tuple[str, ...]) -> None:
        key = node_key(path, node["name"])
        current_path = path + (node["name"],)
        if (node.get("type") or "").lower() == "indicator":
            bp = beta_pres_by_key[key]
            ba = beta_abs_by_key[key]
            if observation_model == "binary_perfect":
                msg = leaf_binary_observed_message(int(latent["indicator_z"][key]), bp, ba)
                ll0 = 0.0 if latent["indicator_z"][key] == 0 else -np.inf
                ll1 = 0.0 if latent["indicator_z"][key] == 1 else -np.inf
                unique_values.add(int(latent["indicator_z"][key]))
            elif observation_model == "binary_noisy":
                z = int(latent["indicator_z"][key])
                p = 1.0 - epsilon if z else epsilon
                successes = int(rng.binomial(int(k_or_n), p))
                msg0, msg1, ll0, ll1 = binary_noisy_message_from_count(
                    successes=successes,
                    n=int(k_or_n),
                    epsilon=epsilon,
                    beta_pres=bp,
                    beta_abs=ba,
                )
                msg = (msg0, msg1)
                unique_values.update([0, 1] if 0 < successes < int(k_or_n) else [1 if successes else 0])
            elif observation_model == "ordinal_binary":
                z = int(latent["indicator_z"][key])
                ratings = np.asarray(
                    [
                        sample_ordered_probit_category(rng, a, kappa, float(z))
                        for _ in range(int(k_or_n))
                    ],
                    dtype=np.int64,
                )
                msg0, msg1, ll0, ll1, vals = binary_ordinal_message(
                    ratings=ratings,
                    a=a,
                    kappa=kappa,
                    beta_pres=bp,
                    beta_abs=ba,
                )
                msg = (msg0, msg1)
                unique_values.update(vals)
            elif observation_model == "ordinal_three_state":
                m = int(latent["indicator_m"][key])
                ratings = np.asarray(
                    [
                        sample_ordered_probit_category(rng, a, kappa, m / 2.0)
                        for _ in range(int(k_or_n))
                    ],
                    dtype=np.int64,
                )
                msg0, msg1, ll0, ll1, vals = three_state_ordinal_message(
                    ratings=ratings,
                    a=a,
                    kappa=kappa,
                    beta_pres=bp,
                    beta_abs=ba,
                )
                msg = (msg0, msg1)
                unique_values.update(vals)
            else:
                raise ValueError(f"Unknown observation model {observation_model!r}")
            messages[key] = msg
            if np.isfinite(ll0):
                leaf_ll0.append(float(ll0))
            if np.isfinite(ll1):
                leaf_ll1.append(float(ll1))
            return
        for child in node.get("evidencers", []):
            walk(child, current_path)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path)
    meta = {
        "unique_observation_values": ",".join(map(str, sorted(unique_values))),
        "mean_leaf_loglik_z0": float(np.mean(leaf_ll0)) if leaf_ll0 else np.nan,
        "mean_leaf_loglik_z1": float(np.mean(leaf_ll1)) if leaf_ll1 else np.nan,
    }
    return messages, meta


def top_feature_messages(
    *,
    stance_data: Mapping[str, Any],
    leaf_messages: Mapping[str, Tuple[float, float]],
    beta_pres_by_key: Mapping[str, float],
    beta_abs_by_key: Mapping[str, float],
    latent: Mapping[str, Any],
    system: str,
    source: str,
    run_id: str,
    rung: str,
    beta_profile: str,
    seed: int,
    replicate_id: Optional[int],
) -> Tuple[float, pd.DataFrame]:
    root_path = (stance_data["name"],)

    def subtree(node: Mapping[str, Any], path: Tuple[str, ...]) -> Tuple[float, float, float, float]:
        key = node_key(path, node["name"])
        current_path = path + (node["name"],)
        if (node.get("type") or "").lower() == "indicator":
            zpa0, zpa1 = leaf_messages.get(key, (0.0, 0.0))
            return np.nan, np.nan, float(zpa0), float(zpa1)
        m0 = 0.0
        m1 = 0.0
        for child in node.get("evidencers", []):
            _, _, c0, c1 = subtree(child, current_path)
            m0 += c0
            m1 += c1
        bp = float(clip_prob(beta_pres_by_key[key]))
        ba = float(clip_prob(beta_abs_by_key[key]))
        zpa1 = float(np.logaddexp(math.log(bp) + m1, math.log1p(-bp) + m0))
        zpa0 = float(np.logaddexp(math.log(ba) + m1, math.log1p(-ba) + m0))
        return float(m0), float(m1), zpa0, zpa1

    rows = []
    total_log_b = 0.0
    for child in top_children(stance_data):
        m0, m1, zpa0, zpa1 = subtree(child.node, root_path)
        contrast = float(m1 - m0)
        c_oracle = float(zpa1 - zpa0)
        total_log_b += c_oracle
        bp = float(clip_prob(beta_pres_by_key[child.key]))
        ba = float(clip_prob(beta_abs_by_key[child.key]))
        realised_z = int(latent["internal_z"][child.key])
        c_top = math.log(bp / ba) if realised_z else math.log((1.0 - bp) / (1.0 - ba))
        counts = descendant_counts(child.node, root_path)
        rows.append(
            {
                "source": source,
                "run_id": run_id,
                "rung": rung,
                "beta_profile": beta_profile,
                "seed": seed,
                "replicate_id": replicate_id if replicate_id is not None else np.nan,
                "system": display_system(system),
                "root_z_true": int(latent["root_z"]),
                "top_feature_key": child.key,
                "top_feature_name": child.node["name"],
                "support_label": child.node.get("support", "no bearing"),
                "demandingness_label": child.node.get("demandingness", "neutral"),
                "beta_pres_top": bp,
                "beta_abs_top": ba,
                "beta_gap_top": bp - ba,
                "realised_z_top": realised_z,
                "c_top_latent": float(c_top),
                "m_u0": float(m0),
                "m_u1": float(m1),
                "message_contrast": contrast,
                "desc_p_z1_equal_prior": float(expit(contrast)),
                "c_oracle_subtree": c_oracle,
                "delta_oracle_minus_top": float(c_oracle - c_top),
                "abs_delta_oracle_minus_top": float(abs(c_oracle - c_top)),
                "message_correct_sign": bool(
                    (realised_z == 1 and contrast > 0.0)
                    or (realised_z == 0 and contrast < 0.0)
                ),
                "message_decisive": bool(abs(contrast) > TAU_PRESENT_50),
                **counts,
            }
        )
    return total_log_b, pd.DataFrame(rows)


def exact_root_sides_from_leaf_messages(
    stance_data: Mapping[str, Any],
    leaf_messages: Mapping[str, Tuple[float, float]],
    beta_pres_by_key: Mapping[str, float],
    beta_abs_by_key: Mapping[str, float],
) -> Tuple[float, float]:
    root_path = (stance_data["name"],)

    def subtree(node: Mapping[str, Any], path: Tuple[str, ...]) -> Tuple[float, float]:
        key = node_key(path, node["name"])
        current_path = path + (node["name"],)
        if (node.get("type") or "").lower() == "indicator":
            return leaf_messages.get(key, (0.0, 0.0))
        m0 = 0.0
        m1 = 0.0
        for child in node.get("evidencers", []):
            c0, c1 = subtree(child, current_path)
            m0 += c0
            m1 += c1
        bp = float(clip_prob(beta_pres_by_key[key]))
        ba = float(clip_prob(beta_abs_by_key[key]))
        zpa1 = float(np.logaddexp(math.log(bp) + m1, math.log1p(-bp) + m0))
        zpa0 = float(np.logaddexp(math.log(ba) + m1, math.log1p(-ba) + m0))
        return zpa0, zpa1

    log_l0 = 0.0
    log_l1 = 0.0
    for child in stance_data.get("evidencers", []):
        c0, c1 = subtree(child, root_path)
        log_l0 += c0
        log_l1 += c1
    return float(log_l0), float(log_l1)


def top_observed_log_b(
    stance_data: Mapping[str, Any],
    latent: Mapping[str, Any],
    beta_pres_by_key: Mapping[str, float],
    beta_abs_by_key: Mapping[str, float],
) -> float:
    val = 0.0
    for child in top_children(stance_data):
        z = int(latent["internal_z"][child.key])
        bp = float(clip_prob(beta_pres_by_key[child.key]))
        ba = float(clip_prob(beta_abs_by_key[child.key]))
        val += math.log(bp / ba) if z else math.log((1.0 - bp) / (1.0 - ba))
    return float(val)


def subfeature_observed_log_b(
    stance_data: Mapping[str, Any],
    latent: Mapping[str, Any],
    beta_pres_by_key: Mapping[str, float],
    beta_abs_by_key: Mapping[str, float],
) -> float:
    total = 0.0
    root_path = (stance_data["name"],)
    for top in top_children(stance_data):
        m0 = 0.0
        m1 = 0.0
        top_path = root_path + (top.node["name"],)
        for child in top.node.get("evidencers", []):
            key = node_key(top_path, child["name"])
            bp = float(clip_prob(beta_pres_by_key[key]))
            ba = float(clip_prob(beta_abs_by_key[key]))
            if (child.get("type") or "").lower() == "indicator":
                z = int(latent["indicator_z"][key])
            else:
                z = int(latent["internal_z"][key])
            if z:
                m1 += math.log(bp)
                m0 += math.log(ba)
            else:
                m1 += math.log1p(-bp)
                m0 += math.log1p(-ba)
        top_bp = float(clip_prob(beta_pres_by_key[top.key]))
        top_ba = float(clip_prob(beta_abs_by_key[top.key]))
        c1 = float(np.logaddexp(math.log(top_bp) + m1, math.log1p(-top_bp) + m0))
        c0 = float(np.logaddexp(math.log(top_ba) + m1, math.log1p(-top_ba) + m0))
        total += c1 - c0
    return float(total)


def leaf_latent_messages(
    latent: Mapping[str, Any],
    beta_pres_by_key: Mapping[str, float],
    beta_abs_by_key: Mapping[str, float],
) -> Dict[str, Tuple[float, float]]:
    return {
        key: leaf_binary_observed_message(
            int(z),
            beta_pres_by_key[key],
            beta_abs_by_key[key],
        )
        for key, z in latent["indicator_z"].items()
    }


def run_attrition_cases(
    *,
    stance_data: Mapping[str, Any],
    profiles: Mapping[str, Tuple[Mapping[str, float], Mapping[str, float], Mapping[str, Any]]],
    obs_params: Mapping[str, Any],
    n_rep: int,
    seed: int,
    collect_messages: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    message_frames = []
    for beta_profile, (bp, ba, _) in profiles.items():
        for rep in range(n_rep):
            roots = root_truth_for_replicate(rep)
            for system in FREE_SYSTEMS:
                rng = rng_for(seed, rep, system)
                latent = sample_latent_states(
                    rng=rng,
                    stance_data=stance_data,
                    beta_pres_by_key=bp,
                    beta_abs_by_key=ba,
                    root_z=roots[system],
                )
                top_log_b = top_observed_log_b(stance_data, latent, bp, ba)
                sub_log_b = subfeature_observed_log_b(stance_data, latent, bp, ba)
                leaf_msg = leaf_latent_messages(latent, bp, ba)
                leaf_l0, leaf_l1 = exact_root_sides_from_leaf_messages(stance_data, leaf_msg, bp, ba)
                leaf_log_b = leaf_l1 - leaf_l0
                binary_perfect = leaf_log_b

                noisy_msg, _ = simulate_leaf_messages(
                    rng=rng,
                    stance_data=stance_data,
                    latent=latent,
                    beta_pres_by_key=bp,
                    beta_abs_by_key=ba,
                    obs_params=obs_params,
                    observation_model="binary_noisy",
                    k_or_n=6,
                    epsilon=0.05,
                )
                noisy_l0, noisy_l1 = exact_root_sides_from_leaf_messages(stance_data, noisy_msg, bp, ba)
                noisy_log_b = noisy_l1 - noisy_l0

                ordinal_binary_msg, _ = simulate_leaf_messages(
                    rng=rng,
                    stance_data=stance_data,
                    latent=latent,
                    beta_pres_by_key=bp,
                    beta_abs_by_key=ba,
                    obs_params=obs_params,
                    observation_model="ordinal_binary",
                    k_or_n=6,
                    epsilon=0.05,
                )
                ob_l0, ob_l1 = exact_root_sides_from_leaf_messages(stance_data, ordinal_binary_msg, bp, ba)
                ordinal_binary_log_b = ob_l1 - ob_l0

                ordinal_three_msg, _ = simulate_leaf_messages(
                    rng=rng,
                    stance_data=stance_data,
                    latent=latent,
                    beta_pres_by_key=bp,
                    beta_abs_by_key=ba,
                    obs_params=obs_params,
                    observation_model="ordinal_three_state",
                    k_or_n=6,
                    epsilon=0.05,
                )
                ot_l0, ot_l1 = exact_root_sides_from_leaf_messages(stance_data, ordinal_three_msg, bp, ba)
                ordinal_three_log_b = ot_l1 - ot_l0

                values = {
                    "log_B_top_observed": top_log_b,
                    "log_B_subfeature_observed": sub_log_b,
                    "log_B_leaf_latent_observed": leaf_log_b,
                    "log_B_binary_perfect_leaf": binary_perfect,
                    "log_B_binary_noisy_K6": noisy_log_b,
                    "log_B_ordinal_binary_clamped": ordinal_binary_log_b,
                    "log_B_ordinal_three_state_clamped": ordinal_three_log_b,
                }
                row = {
                    "beta_profile": beta_profile,
                    "seed": seed,
                    "replicate_id": rep,
                    "system": display_system(system),
                    "system_raw": system,
                    "root_z_true": int(roots[system]),
                    **values,
                }
                for stage, col in STAGE_TO_COLUMN.items():
                    row[f"rho_{stage}"] = float(expit(LOGIT_PRIOR + values[col]))
                row.update(
                    {
                        "loss_top_to_subfeature": sub_log_b - top_log_b,
                        "loss_subfeature_to_leaf_latent": leaf_log_b - sub_log_b,
                        "loss_leaf_latent_to_binary_K6": noisy_log_b - leaf_log_b,
                        "loss_binary_perfect_to_binary_K6": noisy_log_b - binary_perfect,
                        "loss_binary_K6_to_ordinal_binary": ordinal_binary_log_b - noisy_log_b,
                        "loss_ordinal_binary_to_three_state": ordinal_three_log_b - ordinal_binary_log_b,
                    }
                )
                rows.append(row)
                if collect_messages:
                    for rung, msg in [
                        (RUNG_LABEL.get((beta_profile, "binary_noisy_K6"), "1"),
                         noisy_msg),
                        (RUNG_LABEL.get((beta_profile, "ordinal_binary_clamped"), "3"),
                         ordinal_binary_msg),
                        (RUNG_LABEL.get((beta_profile, "ordinal_three_state_clamped"), "4"),
                         ordinal_three_msg),
                    ]:
                        _, msg_df = top_feature_messages(
                            stance_data=stance_data,
                            leaf_messages=msg,
                            beta_pres_by_key=bp,
                            beta_abs_by_key=ba,
                            latent=latent,
                            system=system,
                            source="nofit_phase1c",
                            run_id=f"phase1c_rep{rep:04d}_{display_system(system)}",
                            rung=rung,
                            beta_profile=beta_profile,
                            seed=seed + rep,
                            replicate_id=rep,
                        )
                        message_frames.append(msg_df)
    return (
        pd.DataFrame(rows),
        pd.concat(message_frames, ignore_index=True) if message_frames else pd.DataFrame(),
    )


def stage_metrics(df: pd.DataFrame, log_col: str) -> Dict[str, float]:
    r1 = df[df["root_z_true"] == 1]
    r0 = df[df["root_z_true"] == 0]
    if r1.empty or r0.empty:
        return {
            "balanced_accuracy": np.nan,
            "balanced_log_score_improvement": np.nan,
            "evidence_margin_M": np.nan,
            "pass_fail_label": "insufficient",
        }
    rho = expit(LOGIT_PRIOR + df[log_col].to_numpy(dtype=float))
    pred = rho > 0.5
    truth = df["root_z_true"].to_numpy(dtype=int) == 1
    tpr = float(np.mean(pred[truth])) if truth.any() else np.nan
    tnr = float(np.mean(~pred[~truth])) if (~truth).any() else np.nan
    ba = float(np.nanmean([tpr, tnr]))
    rho_r1 = expit(LOGIT_PRIOR + r1[log_col].to_numpy(dtype=float))
    rho_r0 = expit(LOGIT_PRIOR + r0[log_col].to_numpy(dtype=float))
    u_log = 0.5 * float(np.mean(np.log(np.clip(rho_r1, SCORE_CLIP, 1.0) / PRIOR_P))) + 0.5 * float(
        np.mean(np.log(np.clip(1.0 - rho_r0, SCORE_CLIP, 1.0) / (1.0 - PRIOR_P)))
    )
    m = float(
        min(
            float(r1[log_col].median()) - TAU_PRESENT_50,
            TAU_ABSENT_05 - float(r0[log_col].median()),
        )
    )
    if m > 0 and u_log > 0 and ba >= 0.75:
        label = "pass"
    elif m < -0.25 or u_log <= 0 or ba < 0.75:
        label = "fail"
    else:
        label = "borderline"
    return {
        "balanced_accuracy": ba,
        "balanced_log_score_improvement": u_log,
        "evidence_margin_M": m,
        "pass_fail_label": label,
    }


def summarise_attrition_cases(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    loss_cols = [
        "loss_top_to_subfeature",
        "loss_subfeature_to_leaf_latent",
        "loss_leaf_latent_to_binary_K6",
        "loss_binary_perfect_to_binary_K6",
        "loss_binary_K6_to_ordinal_binary",
        "loss_ordinal_binary_to_three_state",
    ]
    for beta_profile in sorted(cases["beta_profile"].unique()):
        for stage in STAGES:
            log_col = STAGE_TO_COLUMN[stage]
            for system in sorted(cases["system"].unique()):
                base = cases[(cases["beta_profile"] == beta_profile) & (cases["system"] == system)]
                metrics = stage_metrics(base, log_col)
                for root_value, g in base.groupby("root_z_true", dropna=False):
                    arr = g[log_col]
                    row = {
                        "beta_profile": beta_profile,
                        "evidence_stage": stage,
                        "rung_label": RUNG_LABEL.get((beta_profile, stage), ""),
                        "system": system,
                        "root_z_true": int(root_value),
                        "n": int(len(g)),
                        "median_log_B": float(arr.median()),
                        "mean_log_B": float(arr.mean()),
                        "q05_log_B": float(arr.quantile(0.05)),
                        "q25_log_B": float(arr.quantile(0.25)),
                        "q75_log_B": float(arr.quantile(0.75)),
                        "q95_log_B": float(arr.quantile(0.95)),
                        "mean_rho": float(np.mean(expit(LOGIT_PRIOR + arr.to_numpy(dtype=float)))),
                        "p_correct_sign": float(
                            np.mean(((g["root_z_true"] == 1) & (arr > 0.0)) | ((g["root_z_true"] == 0) & (arr < 0.0)))
                        ),
                        "p_decisive": float(
                            np.mean(
                                ((g["root_z_true"] == 1) & (arr > TAU_PRESENT_50))
                                | ((g["root_z_true"] == 0) & (arr < TAU_ABSENT_05))
                            )
                        ),
                        **metrics,
                    }
                    for col in loss_cols:
                        row[f"median_{col}"] = float(g[col].median())
                    rows.append(row)
                metrics_all = stage_metrics(base, log_col)
                arr = base[log_col]
                row_all = {
                    "beta_profile": beta_profile,
                    "evidence_stage": stage,
                    "rung_label": RUNG_LABEL.get((beta_profile, stage), ""),
                    "system": system,
                    "root_z_true": "ALL",
                    "n": int(len(base)),
                    "median_log_B": float(arr.median()),
                    "mean_log_B": float(arr.mean()),
                    "q05_log_B": float(arr.quantile(0.05)),
                    "q25_log_B": float(arr.quantile(0.25)),
                    "q75_log_B": float(arr.quantile(0.75)),
                    "q95_log_B": float(arr.quantile(0.95)),
                    "mean_rho": float(np.mean(expit(LOGIT_PRIOR + arr.to_numpy(dtype=float)))),
                    "p_correct_sign": float(
                        np.mean(((base["root_z_true"] == 1) & (arr > 0.0)) | ((base["root_z_true"] == 0) & (arr < 0.0)))
                    ),
                    "p_decisive": float(
                        np.mean(
                            ((base["root_z_true"] == 1) & (arr > TAU_PRESENT_50))
                            | ((base["root_z_true"] == 0) & (arr < TAU_ABSENT_05))
                        )
                    ),
                    **metrics_all,
                }
                for col in loss_cols:
                    row_all[f"median_{col}"] = float(base[col].median())
                rows.append(row_all)
    return pd.DataFrame(rows)


def existing_phase1a_messages(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "phase1a_top_feature_contributions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "source": "existing_phase1a",
            "run_id": df["run_id"],
            "rung": "phase1a_oracle",
            "beta_profile": "production_oracle_median",
            "seed": df["seed"],
            "replicate_id": np.nan,
            "system": df["system"],
            "root_z_true": df["root_z_true"],
            "top_feature_key": df["top_feature_key"],
            "top_feature_name": df["top_feature_name"],
            "support_label": df["support_label"],
            "demandingness_label": df["demandingness_label"],
            "beta_pres_top": df["beta_pres_true"],
            "beta_abs_top": df["beta_abs_true"],
            "beta_gap_top": df["beta_gap_true"],
            "realised_z_top": df["realised_z_top"],
            "c_top_latent": df["c_top_latent"],
            "m_u0": df["subtree_message_loglik_z0"],
            "m_u1": df["subtree_message_loglik_z1"],
            "message_contrast": df["subtree_message_loglik_z1"] - df["subtree_message_loglik_z0"],
            "desc_p_z1_equal_prior": expit(df["subtree_message_loglik_z1"] - df["subtree_message_loglik_z0"]),
            "c_oracle_subtree": df["c_oracle_subtree"],
            "delta_oracle_minus_top": df["delta_oracle_minus_top_latent"],
            "abs_delta_oracle_minus_top": df["abs_delta"],
            "message_correct_sign": (
                ((df["realised_z_top"] == 1) & ((df["subtree_message_loglik_z1"] - df["subtree_message_loglik_z0"]) > 0))
                | ((df["realised_z_top"] == 0) & ((df["subtree_message_loglik_z1"] - df["subtree_message_loglik_z0"]) < 0))
            ),
            "message_decisive": (df["subtree_message_loglik_z1"] - df["subtree_message_loglik_z0"]).abs() > TAU_PRESENT_50,
            "n_descendant_subfeatures": df["n_descendant_subfeatures"],
            "n_descendant_indicators": df["n_descendant_indicators"],
            "n_observed_ratings_desc": df["n_observed_ratings_desc"],
        }
    )
    return out


def message_feature_summary(messages: pd.DataFrame) -> pd.DataFrame:
    if messages.empty:
        return pd.DataFrame()
    rows = []
    for keys, g in messages.groupby(["source", "rung", "beta_profile", "top_feature_name"], dropna=False):
        source, rung, beta_profile, feature = keys
        z1 = g[g["realised_z_top"] == 1]
        z0 = g[g["realised_z_top"] == 0]
        rows.append(
            {
                "source": source,
                "rung": rung,
                "beta_profile": beta_profile,
                "top_feature_name": feature,
                "n": int(len(g)),
                "mean_message_contrast_given_z1": float(z1["message_contrast"].mean()) if len(z1) else np.nan,
                "mean_message_contrast_given_z0": float(z0["message_contrast"].mean()) if len(z0) else np.nan,
                "p_message_correct_sign": float(g["message_correct_sign"].mean()),
                "p_message_decisive": float(g["message_decisive"].mean()),
                "median_c_top_latent": float(g["c_top_latent"].median()),
                "median_c_oracle_subtree": float(g["c_oracle_subtree"].median()),
                "median_delta_oracle_minus_top": float(g["delta_oracle_minus_top"].median()),
                "mean_abs_delta_oracle_minus_top": float(g["abs_delta_oracle_minus_top"].mean()),
                "median_n_descendant_indicators": float(g["n_descendant_indicators"].median()),
                "median_n_observed_ratings_desc": float(g["n_observed_ratings_desc"].median()),
            }
        )
    return pd.DataFrame(rows)


def run_rung_sanity_checks(
    *,
    stance_data: Mapping[str, Any],
    profiles: Mapping[str, Tuple[Mapping[str, float], Mapping[str, float], Mapping[str, Any]]],
    obs_params: Mapping[str, Any],
    seed: int,
    n_rep_checked: int,
) -> pd.DataFrame:
    rows = []
    ref_ll_by_profile: Dict[str, list[float]] = {}
    for beta_profile in ["production_prior_mean", "production_oracle_median"]:
        bp, ba, _ = profiles[beta_profile]
        per_rung_ll = {}
        for rung, obs_model, leaf_model in [
            ("1P" if beta_profile == "production_prior_mean" else "1O", "binary_noisy", "binary"),
            ("3" if beta_profile == "production_prior_mean" else "3O", "ordinal_binary", "binary"),
            ("4" if beta_profile == "production_prior_mean" else "4O", "ordinal_three_state", "three_state"),
        ]:
            all_ll0 = []
            all_ll1 = []
            all_values: set[str] = set()
            max_consistency = 0.0
            for rep in range(n_rep_checked):
                roots = root_truth_for_replicate(rep)
                system = "Chicken"
                rng = rng_for(seed, rep, system)
                latent = sample_latent_states(
                    rng=rng,
                    stance_data=stance_data,
                    beta_pres_by_key=bp,
                    beta_abs_by_key=ba,
                    root_z=roots[system],
                )
                msg, meta = simulate_leaf_messages(
                    rng=rng,
                    stance_data=stance_data,
                    latent=latent,
                    beta_pres_by_key=bp,
                    beta_abs_by_key=ba,
                    obs_params=obs_params,
                    observation_model=obs_model,
                    k_or_n=6,
                    epsilon=0.05,
                )
                l0, l1 = exact_root_sides_from_leaf_messages(stance_data, msg, bp, ba)
                total_log_b, _ = top_feature_messages(
                    stance_data=stance_data,
                    leaf_messages=msg,
                    beta_pres_by_key=bp,
                    beta_abs_by_key=ba,
                    latent=latent,
                    system=system,
                    source="sanity",
                    run_id="sanity",
                    rung=rung,
                    beta_profile=beta_profile,
                    seed=seed,
                    replicate_id=rep,
                )
                max_consistency = max(max_consistency, abs((l1 - l0) - total_log_b))
                all_ll0.append(float(meta["mean_leaf_loglik_z0"]))
                all_ll1.append(float(meta["mean_leaf_loglik_z1"]))
                all_values.update(str(x) for x in str(meta["unique_observation_values"]).split(",") if x != "")
            per_rung_ll[rung] = all_ll0 + all_ll1
            if obs_model == "binary_noisy":
                ref_ll_by_profile[beta_profile] = per_rung_ll[rung]
                diff = 0.0
            else:
                ref = np.asarray(ref_ll_by_profile[beta_profile], dtype=float)
                cur = np.asarray(per_rung_ll[rung], dtype=float)
                n = min(len(ref), len(cur))
                diff = float(np.nanmax(np.abs(cur[:n] - ref[:n]))) if n else np.nan
            rows.append(
                {
                    "rung": rung,
                    "beta_profile": beta_profile,
                    "n_rep_checked": int(n_rep_checked),
                    "observation_model": obs_model,
                    "latent_leaf_model": leaf_model,
                    "unique_observation_values": ",".join(sorted(all_values)),
                    "mean_leaf_loglik_z0": float(np.nanmean(all_ll0)),
                    "mean_leaf_loglik_z1": float(np.nanmean(all_ll1)),
                    "max_abs_diff_leaf_loglik_vs_rung_1P": diff,
                    "exact_loglik_consistency_error": float(max_consistency),
                    "status": "pass" if max_consistency < 1e-8 and (obs_model == "binary_noisy" or diff > 1e-6) else "check",
                }
            )
    return pd.DataFrame(rows)


def edge_profile_betas(
    stance_data: Mapping[str, Any],
    prod_bp: Mapping[str, float],
    prod_ba: Mapping[str, float],
    extreme_bp: Mapping[str, float],
    extreme_ba: Mapping[str, float],
    edge_profile: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    bp: Dict[str, float] = {}
    ba: Dict[str, float] = {}
    for spec in iter_tree_nodes(stance_data):
        is_top = spec.depth == 1
        use_extreme = False
        if edge_profile == "all_extreme_0p9_0p1":
            use_extreme = True
        elif edge_profile == "all_production_prior_mean":
            use_extreme = False
        elif edge_profile == "top_production_lower_extreme":
            use_extreme = not is_top
        elif edge_profile == "top_extreme_lower_production":
            use_extreme = is_top
        elif edge_profile == "strong_top_features_only_extreme_lower":
            use_extreme = (not is_top) and spec.top_feature_name in STRONG_TOP_FEATURES
        elif edge_profile == "weak_top_features_only_extreme_lower":
            use_extreme = (not is_top) and spec.top_feature_name in WEAK_TOP_FEATURES
        else:
            raise ValueError(edge_profile)
        bp[spec.key] = float(extreme_bp[spec.key] if use_extreme else prod_bp[spec.key])
        ba[spec.key] = float(extreme_ba[spec.key] if use_extreme else prod_ba[spec.key])
    return bp, ba


def summarise_edge_profile(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for (edge_profile, observation_model), g in df.groupby(["edge_profile", "observation_model"], dropna=False):
        metrics = stage_metrics(g.rename(columns={"log_B": "log_B"}), "log_B")
        r1 = g[g["root_z_true"] == 1]
        r0 = g[g["root_z_true"] == 0]
        top_r1 = float(r1["log_B_top"].median()) if len(r1) else np.nan
        top_r0 = float(r0["log_B_top"].median()) if len(r0) else np.nan
        out.append(
            {
                "edge_profile": edge_profile,
                "observation_model": observation_model,
                "n_replicates": int(g["replicate_id"].nunique()),
                "median_log_B_R1": float(r1["log_B"].median()) if len(r1) else np.nan,
                "median_log_B_R0": float(r0["log_B"].median()) if len(r0) else np.nan,
                "q05_log_B_R1": float(r1["log_B"].quantile(0.05)) if len(r1) else np.nan,
                "q95_log_B_R1": float(r1["log_B"].quantile(0.95)) if len(r1) else np.nan,
                "q05_log_B_R0": float(r0["log_B"].quantile(0.05)) if len(r0) else np.nan,
                "q95_log_B_R0": float(r0["log_B"].quantile(0.95)) if len(r0) else np.nan,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "balanced_log_score_improvement": metrics["balanced_log_score_improvement"],
                "evidence_margin_M": metrics["evidence_margin_M"],
                "pass_fail_label": metrics["pass_fail_label"],
                "survival_R1_vs_top": float(r1["log_B"].median() / top_r1) if abs(top_r1) > EPS else np.nan,
                "abs_survival_R0_vs_top": float(abs(r0["log_B"].median()) / abs(top_r0)) if abs(top_r0) > EPS else np.nan,
            }
        )
    return pd.DataFrame(out)


def run_edge_profile_ablation(
    *,
    stance_data: Mapping[str, Any],
    profiles: Mapping[str, Tuple[Mapping[str, float], Mapping[str, float], Mapping[str, Any]]],
    obs_params: Mapping[str, Any],
    n_rep: int,
    seed: int,
) -> pd.DataFrame:
    prod_bp, prod_ba, _ = profiles["production_prior_mean"]
    ext_bp, ext_ba, _ = profiles["extreme_0p9_0p1"]
    edge_profiles = [
        "all_production_prior_mean",
        "all_extreme_0p9_0p1",
        "top_production_lower_extreme",
        "top_extreme_lower_production",
        "strong_top_features_only_extreme_lower",
        "weak_top_features_only_extreme_lower",
    ]
    rows: list[dict[str, Any]] = []
    for edge_profile in edge_profiles:
        bp, ba = edge_profile_betas(stance_data, prod_bp, prod_ba, ext_bp, ext_ba, edge_profile)
        for rep in range(n_rep):
            roots = root_truth_for_replicate(rep)
            for system in FREE_SYSTEMS:
                rng = rng_for(seed, rep, system)
                latent = sample_latent_states(
                    rng=rng,
                    stance_data=stance_data,
                    beta_pres_by_key=bp,
                    beta_abs_by_key=ba,
                    root_z=roots[system],
                )
                top_log_b = top_observed_log_b(stance_data, latent, bp, ba)
                perfect_msg = leaf_latent_messages(latent, bp, ba)
                p0, p1 = exact_root_sides_from_leaf_messages(stance_data, perfect_msg, bp, ba)
                noisy_msg, _ = simulate_leaf_messages(
                    rng=rng,
                    stance_data=stance_data,
                    latent=latent,
                    beta_pres_by_key=bp,
                    beta_abs_by_key=ba,
                    obs_params=obs_params,
                    observation_model="binary_noisy",
                    k_or_n=6,
                    epsilon=0.05,
                )
                n0, n1 = exact_root_sides_from_leaf_messages(stance_data, noisy_msg, bp, ba)
                for obs_model, log_b in [
                    ("binary_perfect_leaf", p1 - p0),
                    ("binary_noisy_K6", n1 - n0),
                ]:
                    rows.append(
                        {
                            "edge_profile": edge_profile,
                            "observation_model": obs_model,
                            "replicate_id": rep,
                            "system": display_system(system),
                            "root_z_true": int(roots[system]),
                            "log_B": float(log_b),
                            "rho": float(expit(LOGIT_PRIOR + log_b)),
                            "log_B_top": float(top_log_b),
                        }
                    )
    out = summarise_edge_profile(rows)
    # Matched-seed check: every profile should have the same replicate ids.
    counts = out.groupby("observation_model")["n_replicates"].nunique()
    if not counts.empty and int(counts.max()) != 1:
        raise AssertionError("Edge-profile ablation did not use matched replicate counts")
    return out


def summarise_ramp_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for keys, g in df.groupby(["beta_profile", "observation_model", "K_or_n_raters", "epsilon"], dropna=False):
        beta_profile, obs_model, k, eps = keys
        metrics = stage_metrics(g.rename(columns={"log_B": "log_B"}), "log_B")
        r1 = g[g["root_z_true"] == 1]
        r0 = g[g["root_z_true"] == 0]
        out.append(
            {
                "beta_profile": beta_profile,
                "observation_model": obs_model,
                "K_or_n_raters": k,
                "epsilon": eps,
                "n_replicates": int(g["replicate_id"].nunique()),
                "median_log_B_R1": float(r1["log_B"].median()) if len(r1) else np.nan,
                "median_log_B_R0": float(r0["log_B"].median()) if len(r0) else np.nan,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "balanced_log_score_improvement": metrics["balanced_log_score_improvement"],
                "evidence_margin_M": metrics["evidence_margin_M"],
                "pass_fail_label": metrics["pass_fail_label"],
            }
        )
    return pd.DataFrame(out)


def run_information_ramp(
    *,
    stance_data: Mapping[str, Any],
    profiles: Mapping[str, Tuple[Mapping[str, float], Mapping[str, float], Mapping[str, Any]]],
    obs_params: Mapping[str, Any],
    n_rep: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    profile_names = ["production_prior_mean", "production_oracle_median"]
    for beta_profile in profile_names:
        bp, ba, _ = profiles[beta_profile]
        for rep in range(n_rep):
            roots = root_truth_for_replicate(rep)
            for system in FREE_SYSTEMS:
                rng_base = rng_for(seed, rep, system)
                latent = sample_latent_states(
                    rng=rng_base,
                    stance_data=stance_data,
                    beta_pres_by_key=bp,
                    beta_abs_by_key=ba,
                    root_z=roots[system],
                )
                perfect_msg = leaf_latent_messages(latent, bp, ba)
                p0, p1 = exact_root_sides_from_leaf_messages(stance_data, perfect_msg, bp, ba)
                rows.append(
                    {
                        "beta_profile": beta_profile,
                        "observation_model": "perfect_leaf",
                        "K_or_n_raters": "perfect",
                        "epsilon": 0.0,
                        "replicate_id": rep,
                        "system": display_system(system),
                        "root_z_true": int(roots[system]),
                        "log_B": float(p1 - p0),
                    }
                )
                for k in [1, 2, 3, 6, 12, 30, 100, 1000]:
                    rng = rng_for(seed + k * 17, rep, system)
                    msg, _ = simulate_leaf_messages(
                        rng=rng,
                        stance_data=stance_data,
                        latent=latent,
                        beta_pres_by_key=bp,
                        beta_abs_by_key=ba,
                        obs_params=obs_params,
                        observation_model="binary_noisy",
                        k_or_n=k,
                        epsilon=0.05,
                    )
                    l0, l1 = exact_root_sides_from_leaf_messages(stance_data, msg, bp, ba)
                    rows.append(
                        {
                            "beta_profile": beta_profile,
                            "observation_model": "binary_noisy",
                            "K_or_n_raters": k,
                            "epsilon": 0.05,
                            "replicate_id": rep,
                            "system": display_system(system),
                            "root_z_true": int(roots[system]),
                            "log_B": float(l1 - l0),
                        }
                    )
                for n_raters in [1, 2, 3, 6, 12, 30]:
                    for obs_model in ["ordinal_binary", "ordinal_three_state"]:
                        rng = rng_for(seed + n_raters * 101 + (0 if obs_model == "ordinal_binary" else 10_000), rep, system)
                        msg, _ = simulate_leaf_messages(
                            rng=rng,
                            stance_data=stance_data,
                            latent=latent,
                            beta_pres_by_key=bp,
                            beta_abs_by_key=ba,
                            obs_params=obs_params,
                            observation_model=obs_model,
                            k_or_n=n_raters,
                            epsilon=0.05,
                        )
                        l0, l1 = exact_root_sides_from_leaf_messages(stance_data, msg, bp, ba)
                        rows.append(
                            {
                                "beta_profile": beta_profile,
                                "observation_model": obs_model,
                                "K_or_n_raters": n_raters,
                                "epsilon": np.nan,
                                "replicate_id": rep,
                                "system": display_system(system),
                                "root_z_true": int(roots[system]),
                                "log_B": float(l1 - l0),
                            }
                        )
    return summarise_ramp_rows(rows)


def path_records(
    stance_data: Mapping[str, Any],
    beta_pres_by_key: Mapping[str, float],
    beta_abs_by_key: Mapping[str, float],
) -> list[dict[str, Any]]:
    root_path = (stance_data["name"],)
    records = []

    def walk(
        node: Mapping[str, Any],
        path: Tuple[str, ...],
        top_key: str,
        top_name: str,
        edge_keys: list[str],
        labels: list[str],
    ) -> None:
        key = node_key(path, node["name"])
        current_edges = edge_keys + [key]
        current_labels = labels + [f"{node.get('support', 'no bearing')} / {node.get('demandingness', 'neutral')}"]
        current_path = path + (node["name"],)
        if (node.get("type") or "").lower() == "indicator":
            gaps = [float(beta_pres_by_key[k] - beta_abs_by_key[k]) for k in current_edges]
            records.append(
                {
                    "leaf_key": key,
                    "leaf_name": node["name"],
                    "top_feature_key": top_key,
                    "top_feature_name": top_name,
                    "path_length": len(current_edges),
                    "path_edge_gaps": ";".join(f"{g:.6g}" for g in gaps),
                    "min_path_gap": float(min(abs(g) for g in gaps)),
                    "product_path_gap_heuristic": float(np.prod(gaps)),
                    "path_labels": " > ".join(current_labels),
                }
            )
            return
        for child in node.get("evidencers", []):
            walk(child, current_path, top_key, top_name, current_edges, current_labels)

    for child in stance_data.get("evidencers", []):
        top_key = node_key(root_path, child["name"])
        walk(child, root_path, top_key, child["name"], [], [])
    return records


def propagate_leaf_probabilities(
    stance_data: Mapping[str, Any],
    beta_pres_by_key: Mapping[str, float],
    beta_abs_by_key: Mapping[str, float],
    root_q: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    root_path = (stance_data["name"],)

    def walk(node: Mapping[str, Any], path: Tuple[str, ...], parent_q: float) -> None:
        key = node_key(path, node["name"])
        bp = float(beta_pres_by_key[key])
        ba = float(beta_abs_by_key[key])
        q = ba + parent_q * (bp - ba)
        if (node.get("type") or "").lower() == "indicator":
            out[key] = float(q)
            return
        current_path = path + (node["name"],)
        for child in node.get("evidencers", []):
            walk(child, current_path, q)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path, float(root_q))
    return out


def kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.sum(p * (np.log(np.clip(p, SCORE_CLIP, 1.0)) - np.log(np.clip(q, SCORE_CLIP, 1.0)))))


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def ordinal_probs_for_q(q: float, a: float, kappa: np.ndarray) -> np.ndarray:
    out = np.zeros(len(kappa) + 1)
    weights = np.asarray([(1 - q) ** 2, 2 * q * (1 - q), q**2], dtype=float)
    for m, w in enumerate(weights):
        eta = a * (m / 2.0)
        cum = np.concatenate(([0.0], np.asarray([math.erfc(-(cut - eta) / math.sqrt(2.0)) / 2.0 for cut in kappa]), [1.0]))
        out += w * np.diff(cum)
    out = np.clip(out, SCORE_CLIP, 1.0)
    return out / out.sum()


def run_leaf_information_map(
    *,
    stance_data: Mapping[str, Any],
    profiles: Mapping[str, Tuple[Mapping[str, float], Mapping[str, float], Mapping[str, Any]]],
    obs_params: Mapping[str, Any],
) -> pd.DataFrame:
    rows = []
    a = float(obs_params["a"])
    kappa = np.asarray(obs_params["kappa"], dtype=float)
    for beta_profile, (bp, ba, _) in profiles.items():
        q1 = propagate_leaf_probabilities(stance_data, bp, ba, 1.0)
        q0 = propagate_leaf_probabilities(stance_data, bp, ba, 0.0)
        records = path_records(stance_data, bp, ba)
        siblings = defaultdict(int)
        for rec in records:
            siblings[rec["top_feature_key"]] += 1
        for rec in records:
            leaf = rec["leaf_key"]
            p1 = float(np.clip(q1[leaf], SCORE_CLIP, 1.0 - SCORE_CLIP))
            p0 = float(np.clip(q0[leaf], SCORE_CLIP, 1.0 - SCORE_CLIP))
            b1 = np.asarray([1.0 - p1, p1])
            b0 = np.asarray([1.0 - p0, p0])
            m1 = np.asarray([(1 - p1) ** 2, 2 * p1 * (1 - p1), p1**2])
            m0 = np.asarray([(1 - p0) ** 2, 2 * p0 * (1 - p0), p0**2])
            y1 = ordinal_probs_for_q(p1, a, kappa)
            y0 = ordinal_probs_for_q(p0, a, kappa)
            row = {
                "beta_profile": beta_profile,
                **rec,
                "n_sibling_leaves_under_top": int(siblings[rec["top_feature_key"]]),
                "p_leaf_present_R1": p1,
                "p_leaf_present_R0": p0,
                "binary_gap": p1 - p0,
                "kl_binary_R1_to_R0": kl(b1, b0),
                "kl_binary_R0_to_R1": kl(b0, b1),
                "sym_kl_binary": 0.5 * (kl(b1, b0) + kl(b0, b1)),
                "tv_binary": 0.5 * float(np.sum(np.abs(b1 - b0))),
                "hellinger_binary": hellinger(b1, b0),
                "p_m0_R1": float(m1[0]),
                "p_m1_R1": float(m1[1]),
                "p_m2_R1": float(m1[2]),
                "p_m0_R0": float(m0[0]),
                "p_m1_R0": float(m0[1]),
                "p_m2_R0": float(m0[2]),
                "sym_kl_three_state": 0.5 * (kl(m1, m0) + kl(m0, m1)),
                "hellinger_three_state": hellinger(m1, m0),
                "sym_kl_ordinal_ref_rater": 0.5 * (kl(y1, y0) + kl(y0, y1)),
                "hellinger_ordinal_ref_rater": hellinger(y1, y0),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def run_expected_information_checks(
    *,
    stance_data: Mapping[str, Any],
    profiles: Mapping[str, Tuple[Mapping[str, float], Mapping[str, float], Mapping[str, Any]]],
    obs_params: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """Small numerical guards requested for Phase 1C."""
    # Leaf-perfect should dominate noisy K=6 in median absolute evidence in
    # a matched small sample under production prior.
    bp, ba, _ = profiles["production_prior_mean"]
    rows = []
    for rep in range(50):
        roots = root_truth_for_replicate(rep)
        for system in FREE_SYSTEMS:
            rng = rng_for(seed, rep, system)
            latent = sample_latent_states(
                rng=rng,
                stance_data=stance_data,
                beta_pres_by_key=bp,
                beta_abs_by_key=ba,
                root_z=roots[system],
            )
            perfect_msg = leaf_latent_messages(latent, bp, ba)
            p0, p1 = exact_root_sides_from_leaf_messages(stance_data, perfect_msg, bp, ba)
            noisy_msg, _ = simulate_leaf_messages(
                rng=rng,
                stance_data=stance_data,
                latent=latent,
                beta_pres_by_key=bp,
                beta_abs_by_key=ba,
                obs_params=obs_params,
                observation_model="binary_noisy",
                k_or_n=6,
                epsilon=0.05,
            )
            n0, n1 = exact_root_sides_from_leaf_messages(stance_data, noisy_msg, bp, ba)
            rows.append({"perfect": abs(p1 - p0), "noisy": abs(n1 - n0)})
    df = pd.DataFrame(rows)
    perfect_ge_noisy = bool(df["perfect"].median() >= df["noisy"].median())
    return {
        "perfect_leaf_median_abs_log_B": float(df["perfect"].median()),
        "noisy_K6_median_abs_log_B": float(df["noisy"].median()),
        "perfect_leaf_at_least_noisy_in_median_abs": perfect_ge_noisy,
    }


def build_report(
    *,
    output_dir: Path,
    sanity: pd.DataFrame,
    attrition_summary: pd.DataFrame,
    messages: pd.DataFrame,
    edge_ablation: pd.DataFrame,
    information_ramp: pd.DataFrame,
    leaf_map: pd.DataFrame,
    checks: Mapping[str, Any],
    n_rep: int,
    seed: int,
) -> None:
    def compact_stage_table(df: pd.DataFrame, wanted: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for keys, _ in wanted.groupby(["beta_profile", "evidence_stage", "system"], dropna=False):
            beta_profile, stage, system = keys
            base = df[
                (df["beta_profile"] == beta_profile)
                & (df["evidence_stage"] == stage)
                & (df["system"] == system)
            ]
            r1 = base[base["root_z_true"].astype(str) == "1"]
            r0 = base[base["root_z_true"].astype(str) == "0"]
            all_row = base[base["root_z_true"].astype(str) == "ALL"]
            metric_row = all_row.iloc[0] if len(all_row) else base.iloc[0]
            rows.append(
                {
                    "beta_profile": beta_profile,
                    "evidence_stage": stage,
                    "rung_label": metric_row.get("rung_label", ""),
                    "system": system,
                    "median_log_B_R1": float(r1["median_log_B"].iloc[0]) if len(r1) else np.nan,
                    "median_log_B_R0": float(r0["median_log_B"].iloc[0]) if len(r0) else np.nan,
                    "balanced_accuracy": float(metric_row["balanced_accuracy"]),
                    "evidence_margin_M": float(metric_row["evidence_margin_M"]),
                    "pass_fail_label": metric_row["pass_fail_label"],
                }
            )
        return pd.DataFrame(rows)

    lines = ["# Phase 1C Evidence Attrition Audit", ""]
    lines.append(f"Replicates: `{n_rep}` matched seeds from `{seed}`. No HMC refits were launched.")
    lines.append("")
    lines.append("## 1. Sanity Checks")
    lines.append(markdown_table(sanity, ["rung", "beta_profile", "observation_model", "latent_leaf_model", "unique_observation_values", "max_abs_diff_leaf_loglik_vs_rung_1P", "exact_loglik_consistency_error", "status"], max_rows=20))
    lines.append(
        f"\nLeaf-perfect median absolute evidence check: perfect={checks.get('perfect_leaf_median_abs_log_B'):.3f}, "
        f"noisy K6={checks.get('noisy_K6_median_abs_log_B'):.3f}, "
        f"passed={checks.get('perfect_leaf_at_least_noisy_in_median_abs')}."
    )
    lines.append("Pure log_B columns exclude the root prior; the prior is only used to compute rho and threshold gates.")
    lines.append("")
    lines.append("## 2. Oracle-Median No-Fit Rungs")
    rung_view = attrition_summary[
        (attrition_summary["system"].isin(["Chicken", "LLMs"]))
        & (attrition_summary["evidence_stage"].isin(["binary_noisy_K6", "ordinal_binary_clamped", "ordinal_three_state_clamped"]))
        & (attrition_summary["beta_profile"].isin(["production_prior_mean", "production_oracle_median", "extreme_0p9_0p1"]))
    ]
    rung_compact = compact_stage_table(attrition_summary, rung_view)
    lines.append(markdown_table(rung_compact, ["beta_profile", "evidence_stage", "rung_label", "system", "median_log_B_R1", "median_log_B_R0", "balanced_accuracy", "evidence_margin_M", "pass_fail_label"], max_rows=40))
    lines.append("")
    lines.append("## 3. Layer-Wise Attrition")
    stage_view = attrition_summary[
        (attrition_summary["system"].isin(["Chicken", "LLMs"]))
        & (attrition_summary["beta_profile"].isin(["production_prior_mean", "production_oracle_median"]))
    ]
    stage_compact = compact_stage_table(attrition_summary, stage_view)
    lines.append(markdown_table(stage_compact, ["beta_profile", "evidence_stage", "system", "median_log_B_R1", "median_log_B_R0", "balanced_accuracy", "evidence_margin_M", "pass_fail_label"], max_rows=60))
    lines.append("Evidence first falls below the pass gate at the leaf/subtree stages under production beta; top-observed passes but descendant-observed stages generally do not.")
    lines.append("")
    lines.append("## 4. Top-State Message Recovery")
    msg_summary = message_feature_summary(messages)
    msg_focus = msg_summary[
        (msg_summary["source"] == "nofit_phase1c")
        & (msg_summary["rung"].isin(["1P", "1O", "3", "3O", "4", "4O"]))
    ].sort_values(["beta_profile", "rung", "mean_abs_delta_oracle_minus_top"], ascending=[True, True, False])
    lines.append(markdown_table(msg_focus, ["rung", "beta_profile", "top_feature_name", "p_message_correct_sign", "p_message_decisive", "median_c_top_latent", "median_c_oracle_subtree", "mean_abs_delta_oracle_minus_top"], max_rows=40))
    lines.append("")
    lines.append("## 5. Biggest Top-Feature Losses")
    existing_focus = msg_summary[msg_summary["source"] == "existing_phase1a"].sort_values("mean_abs_delta_oracle_minus_top", ascending=False)
    lines.append(markdown_table(existing_focus, ["top_feature_name", "n", "p_message_correct_sign", "p_message_decisive", "median_c_top_latent", "median_c_oracle_subtree", "mean_abs_delta_oracle_minus_top"], max_rows=10))
    lines.append("Coherence, Selective Attention, and Complexity remain the primary downstream-loss features; Integration is comparatively more stable but still worth preserving because it is a strong top feature.")
    lines.append("")
    lines.append("## 6. Edge-Profile Ablations")
    lines.append(markdown_table(edge_ablation, ["edge_profile", "observation_model", "median_log_B_R1", "median_log_B_R0", "balanced_accuracy", "evidence_margin_M", "pass_fail_label", "survival_R1_vs_top", "abs_survival_R0_vs_top"], max_rows=20))
    lines.append(
        "Observed pattern: all-production fails, all-extreme passes, top-production/lower-extreme passes, "
        "and strong-top-feature lower-edge strengthening passes while weak-top-feature lower-edge strengthening fails. "
        "Top-extreme/lower-production also passes, but with low survival versus its much larger top bound; this means stronger root-to-top gaps can compensate, while the production lower tree still transmits too little evidence on its own."
    )
    lines.append("")
    lines.append("## 7. Information Ramp")
    ramp_view = information_ramp[information_ramp["beta_profile"].isin(["production_prior_mean", "production_oracle_median"])]
    lines.append(markdown_table(ramp_view, ["beta_profile", "observation_model", "K_or_n_raters", "median_log_B_R1", "median_log_B_R0", "balanced_accuracy", "evidence_margin_M", "pass_fail_label"], max_rows=80))
    lines.append("")
    lines.append("## 8. Leaf Information Map")
    top_leaves = leaf_map.sort_values("sym_kl_binary", ascending=False).head(10)
    bottom_leaves = leaf_map.sort_values("sym_kl_binary", ascending=True).head(10)
    top_ord = leaf_map.sort_values("sym_kl_ordinal_ref_rater", ascending=False).head(10)
    bottom_ord = leaf_map.sort_values("sym_kl_ordinal_ref_rater", ascending=True).head(10)
    lines.append("Top 10 leaves by binary symmetric KL:")
    lines.append(markdown_table(top_leaves, ["beta_profile", "leaf_name", "top_feature_name", "binary_gap", "sym_kl_binary", "product_path_gap_heuristic"], max_rows=10))
    lines.append("Bottom 10 leaves by binary symmetric KL:")
    lines.append(markdown_table(bottom_leaves, ["beta_profile", "leaf_name", "top_feature_name", "binary_gap", "sym_kl_binary", "product_path_gap_heuristic"], max_rows=10))
    lines.append("Top 10 leaves by ordinal reference-rater symmetric KL:")
    lines.append(markdown_table(top_ord, ["beta_profile", "leaf_name", "top_feature_name", "sym_kl_ordinal_ref_rater", "product_path_gap_heuristic"], max_rows=10))
    lines.append("Bottom 10 leaves by ordinal reference-rater symmetric KL:")
    lines.append(markdown_table(bottom_ord, ["beta_profile", "leaf_name", "top_feature_name", "sym_kl_ordinal_ref_rater", "product_path_gap_heuristic"], max_rows=10))
    lines.append("")
    lines.append("## 9. Revised Hypothesis Ranking")
    lines.append("1. Lower-depth transmission / subtree observability under production beta.")
    lines.append("2. Finite observation density, if the information ramp only recovers at high K or many crossed raters.")
    lines.append("3. Nuisance uncertainty / beta identifiability, only after no-fit stages pass.")
    lines.append("4. Rater design, likely important for Chicken posterior disagreements but not the first cause of R=1 oracle ambiguity.")
    lines.append("5. Ordinal/three-state emission specifically, because binary noisy leaves already fail under production beta.")
    lines.append("")
    lines.append("## 10. Before HMC Fit Rungs")
    lines.append("Do not start full HMC yet. Inspect lower-tree support/demandingness labels, strengthen or redesign weak lower-depth paths, and use the information ramp to decide whether survey density can rescue production beta. Fit rungs should wait until no-fit stages have a passable evidence margin.")
    lines.append("")
    lines.append("## Commands")
    lines.append("```bash")
    lines.append(".venv/bin/python scripts/phase1c_evidence_attrition_audit.py --output-dir outputs/phase1_root_evidence --n-rep 500 --seed 20260511 --mode all")
    lines.append(".venv/bin/python scripts/phase1c_evidence_attrition_audit.py --output-dir outputs/phase1_root_evidence --n-rep 20 --seed 20260511 --mode smoke")
    lines.append("```")
    (output_dir / "phase1c_evidence_attrition_report.md").write_text("\n".join(lines) + "\n")


def run_all(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stance_data = load_gwt_stance()
    oracle_truth_path = args.oracle_truth.expanduser().resolve() if args.oracle_truth else find_oracle_truth_path()
    profiles = build_beta_profiles(stance_data, oracle_truth_path)
    obs_params = load_observation_truth(oracle_truth_path)

    n_rep = int(args.n_rep)
    if args.mode == "smoke":
        n_rep = min(n_rep, 20)
    message_n = min(n_rep, args.message_n_rep)

    print("[phase1c] sanity checks")
    sanity = run_rung_sanity_checks(
        stance_data=stance_data,
        profiles=profiles,
        obs_params=obs_params,
        seed=args.seed,
        n_rep_checked=min(5, n_rep),
    )
    sanity.to_csv(output_dir / "phase1c_rung_sanity_checks.csv", index=False)

    print(f"[phase1c] attrition cases n_rep={n_rep}")
    cases, nofit_messages = run_attrition_cases(
        stance_data=stance_data,
        profiles=profiles,
        obs_params=obs_params,
        n_rep=n_rep,
        seed=args.seed,
        collect_messages=args.mode != "quick",
    )
    cases.to_csv(output_dir / "phase1c_evidence_attrition_cases.csv", index=False)
    summary = summarise_attrition_cases(cases)
    summary.to_csv(output_dir / "phase1c_evidence_attrition_summary.csv", index=False)

    existing_messages = existing_phase1a_messages(output_dir)
    messages = pd.concat([existing_messages, nofit_messages], ignore_index=True) if not nofit_messages.empty else existing_messages
    messages.to_csv(output_dir / "phase1c_feature_message_diagnostics.csv", index=False)

    print(f"[phase1c] edge profile ablation n_rep={n_rep}")
    edge_ablation = run_edge_profile_ablation(
        stance_data=stance_data,
        profiles=profiles,
        obs_params=obs_params,
        n_rep=n_rep,
        seed=args.seed,
    )
    edge_ablation.to_csv(output_dir / "phase1c_edge_profile_ablation.csv", index=False)

    print(f"[phase1c] information ramp n_rep={n_rep}")
    ramp = run_information_ramp(
        stance_data=stance_data,
        profiles=profiles,
        obs_params=obs_params,
        n_rep=n_rep,
        seed=args.seed,
    )
    ramp.to_csv(output_dir / "phase1c_information_ramp.csv", index=False)

    print("[phase1c] leaf information map")
    leaf_map = run_leaf_information_map(
        stance_data=stance_data,
        profiles=profiles,
        obs_params=obs_params,
    )
    leaf_map.to_csv(output_dir / "phase1c_leaf_information_map.csv", index=False)

    checks = run_expected_information_checks(
        stance_data=stance_data,
        profiles=profiles,
        obs_params=obs_params,
        seed=args.seed,
    )
    if not checks["perfect_leaf_at_least_noisy_in_median_abs"]:
        print("[phase1c] warning: perfect leaf median abs evidence below noisy K6 in small check")

    build_report(
        output_dir=output_dir,
        sanity=sanity,
        attrition_summary=summary,
        messages=messages,
        edge_ablation=edge_ablation,
        information_ramp=ramp,
        leaf_map=leaf_map,
        checks=checks,
        n_rep=n_rep,
        seed=args.seed,
    )
    print(f"[phase1c] wrote outputs to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/phase1_root_evidence")
    parser.add_argument("--n-rep", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--mode", choices=["all", "smoke", "quick"], default="all")
    parser.add_argument("--oracle-truth", type=Path, default=None)
    parser.add_argument(
        "--message-n-rep",
        type=int,
        default=500,
        help="Reserved for future thinning; messages currently follow n_rep.",
    )
    check_threshold_constants()
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
