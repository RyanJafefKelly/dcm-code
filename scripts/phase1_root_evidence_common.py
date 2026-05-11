"""Shared helpers for Phase 1 root-evidence diagnostics.

The helpers in this module intentionally stay close to the existing exact-tree
diagnostic code.  They provide stable top-bound enumeration, root-evidence
scoring, and small reporting utilities without changing model semantics.
"""

from __future__ import annotations

import itertools
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit, logit, logsumexp
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcm_model import (  # noqa: E402
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    MultiSystemModelBuilder,
    load_data,
    node_key,
)

PRIOR_P = 1.0 / 6.0
LOGIT_PRIOR = math.log(PRIOR_P / (1.0 - PRIOR_P))

TAU_PRESENT_50 = math.log(5.0)
TAU_ABSENT_05 = -1.3350010667
TAU_PRESENT_95 = 4.5538768916
TAU_ABSENT_01 = -2.9856819377

DISAGREE_WARN_NATS = 2.0
DISAGREE_SEVERE_NATS = 4.0
EPS = 1e-9
SCORE_CLIP = 1e-12

STANCE = "Global Workspace Theory"
FREE_SYSTEMS = ("Chicken", "2024 Leading Chat LLMs")
SYSTEM_LABELS = {"2024 Leading Chat LLMs": "LLMs"}
SYSTEM_ALIASES = {"LLMs": "2024 Leading Chat LLMs", "Chicken": "Chicken"}
ALL_SYSTEMS = ("Human", "Chicken", "2024 Leading Chat LLMs", "ELIZA")


@dataclass(frozen=True)
class TreeNodeSpec:
    node: Mapping[str, Any]
    key: str
    path: Tuple[str, ...]
    depth: int
    top_feature_key: str
    top_feature_name: str


def check_threshold_constants() -> None:
    """Lightweight guard against threshold drift."""
    if abs(expit(LOGIT_PRIOR + TAU_PRESENT_50) - 0.5) > 1e-10:
        raise AssertionError("TAU_PRESENT_50 no longer maps to rho=0.5")
    if abs(expit(LOGIT_PRIOR + TAU_ABSENT_05) - 0.05) > 1e-9:
        raise AssertionError("TAU_ABSENT_05 no longer maps to rho=0.05")
    if abs(expit(LOGIT_PRIOR + TAU_PRESENT_95) - 0.95) > 1e-9:
        raise AssertionError("TAU_PRESENT_95 no longer maps to rho=0.95")
    if abs(expit(LOGIT_PRIOR + TAU_ABSENT_01) - 0.01) > 1e-9:
        raise AssertionError("TAU_ABSENT_01 no longer maps to rho=0.01")


def clip_prob(x: float | np.ndarray, eps: float = SCORE_CLIP) -> float | np.ndarray:
    return np.clip(x, eps, 1.0 - eps)


def logit_clipped(p: float) -> float:
    p_clip = float(clip_prob(p))
    return float(math.log(p_clip / (1.0 - p_clip)))


def rho_from_log_b(log_b: float | np.ndarray, prior_p: float = PRIOR_P) -> float | np.ndarray:
    return expit(logit_clipped(prior_p) + np.asarray(log_b))


def evidence_category_from_log_b(log_b: float) -> str:
    if log_b >= TAU_PRESENT_95:
        return "strong_present"
    if log_b >= TAU_PRESENT_50:
        return "moderate_present"
    if log_b >= TAU_ABSENT_05:
        return "ambiguous"
    if log_b > TAU_ABSENT_01:
        return "moderate_absent"
    return "strong_absent"


def bernoulli_log_score(p: float, y: int) -> float:
    p_clip = float(clip_prob(p))
    return float(y * math.log(p_clip) + (1 - y) * math.log1p(-p_clip))


def bernoulli_brier(p: float, y: int) -> float:
    return float((float(p) - int(y)) ** 2)


def display_system(system: str) -> str:
    return SYSTEM_LABELS.get(system, system)


def canonical_system(system: str) -> str:
    return SYSTEM_ALIASES.get(system, system)


def stance_prefix(stance_name: str = STANCE) -> str:
    return stance_name.replace(" ", "_").replace("/", "_").lower()


def c_var_name(system: str, stance_name: str = STANCE) -> str:
    return f"{MultiSystemModelBuilder._sys_prefix(system)}__{stance_prefix(stance_name)}_C"


def deterministic_prefix(system: str, stance_name: str = STANCE) -> str:
    return f"{MultiSystemModelBuilder._sys_prefix(system)}__{stance_prefix(stance_name)}"


def load_gwt_stance(config: Optional[ModelConfig] = None) -> Dict[str, Any]:
    cfg = config or ModelConfig()
    return next(s for s in load_data(cfg) if s["name"] == STANCE)


def model_config_from_payload(payload: Mapping[str, Any]) -> ModelConfig:
    cfg_payload = payload.get("config") or payload.get("model_config") or {}
    field_names = set(ModelConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    return ModelConfig(**{k: v for k, v in cfg_payload.items() if k in field_names})


def iter_tree_nodes(stance_data: Mapping[str, Any]) -> Iterable[TreeNodeSpec]:
    root_path = (stance_data["name"],)

    def walk(
        node: Mapping[str, Any],
        ancestor_path: Tuple[str, ...],
        depth: int,
        top_key: str,
        top_name: str,
    ) -> Iterable[TreeNodeSpec]:
        key = node_key(ancestor_path, node["name"])
        current_path = ancestor_path + (node["name"],)
        yield TreeNodeSpec(
            node=node,
            key=key,
            path=current_path,
            depth=depth,
            top_feature_key=top_key,
            top_feature_name=top_name,
        )
        for child in node.get("evidencers", []):
            yield from walk(child, current_path, depth + 1, top_key, top_name)

    for child in stance_data.get("evidencers", []):
        top_key = node_key(root_path, child["name"])
        yield from walk(child, root_path, 1, top_key, child["name"])


def top_children(stance_data: Mapping[str, Any]) -> list[TreeNodeSpec]:
    root_path = (stance_data["name"],)
    return [
        TreeNodeSpec(
            node=child,
            key=node_key(root_path, child["name"]),
            path=root_path + (child["name"],),
            depth=1,
            top_feature_key=node_key(root_path, child["name"]),
            top_feature_name=child["name"],
        )
        for child in stance_data.get("evidencers", [])
    ]


def descendant_counts(
    node: Mapping[str, Any],
    ancestor_path: Tuple[str, ...],
    observed_by_system: Optional[Mapping[str, Mapping[str, Sequence[Tuple[int, int]]]]] = None,
    system: Optional[str] = None,
) -> Dict[str, int]:
    counts = {
        "n_descendant_subfeatures": 0,
        "n_descendant_indicators": 0,
        "n_observed_ratings_desc": 0,
    }

    def walk(cur_node: Mapping[str, Any], path: Tuple[str, ...]) -> None:
        current_path = path + (cur_node["name"],)
        ntype = (cur_node.get("type") or "").lower()
        key = node_key(path, cur_node["name"])
        if ntype in {"feature", "subfeature"} and current_path != ancestor_path + (node["name"],):
            counts["n_descendant_subfeatures"] += 1
        if ntype == "indicator":
            counts["n_descendant_indicators"] += 1
            if observed_by_system is not None and system is not None:
                counts["n_observed_ratings_desc"] += len(
                    observed_by_system.get(system, {}).get(key, [])
                )
        for child in cur_node.get("evidencers", []):
            walk(child, current_path)

    walk(node, ancestor_path)
    return counts


def count_system_observations(
    processor: MultiSystemDataProcessor,
    system: str,
) -> Dict[str, int]:
    obs = processor.system_observations.get(system, {})
    raters = {expert for rows in obs.values() for expert, _ in rows}
    return {
        "n_ratings": int(sum(len(rows) for rows in obs.values())),
        "n_leaves_observed": int(len(obs)),
        "n_raters": int(len(raters)),
    }


def beta_profile_from_evidence_processor(
    stance_data: Mapping[str, Any],
    config: Optional[ModelConfig] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, Any]]]:
    cfg = config or ModelConfig()
    ep = EvidenceProcessor(cfg)
    beta_pres: Dict[str, float] = {}
    beta_abs: Dict[str, float] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for spec in iter_tree_nodes(stance_data):
        support = spec.node.get("support", "no bearing")
        demand = spec.node.get("demandingness", "neutral")
        ap, bp, aa, ba = ep.get_beta_parameters(support, demand)
        beta_pres[spec.key] = float(ap / (ap + bp))
        beta_abs[spec.key] = float(aa / (aa + ba))
        meta[spec.key] = {
            "node_name": spec.node["name"],
            "node_type": (spec.node.get("type") or "").lower(),
            "support": support,
            "demandingness": demand,
            "beta_pres": beta_pres[spec.key],
            "beta_abs": beta_abs[spec.key],
            "delta": beta_pres[spec.key] - beta_abs[spec.key],
        }
    return beta_pres, beta_abs, meta


def beta_profile_from_truth(
    truth_payload: Mapping[str, Any],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, Any]]]:
    edge_betas = truth_payload["edge_betas"]
    beta_pres = {k: float(v["beta_pres"]) for k, v in edge_betas.items()}
    beta_abs = {k: float(v["beta_abs"]) for k, v in edge_betas.items()}
    meta = {k: dict(v) for k, v in edge_betas.items()}
    return beta_pres, beta_abs, meta


def extreme_beta_profile(
    stance_data: Mapping[str, Any],
    beta_pres_value: float = 0.90,
    beta_abs_value: float = 0.10,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, Any]]]:
    beta_pres: Dict[str, float] = {}
    beta_abs: Dict[str, float] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for spec in iter_tree_nodes(stance_data):
        beta_pres[spec.key] = float(beta_pres_value)
        beta_abs[spec.key] = float(beta_abs_value)
        meta[spec.key] = {
            "node_name": spec.node["name"],
            "node_type": (spec.node.get("type") or "").lower(),
            "support": spec.node.get("support", "no bearing"),
            "demandingness": spec.node.get("demandingness", "neutral"),
            "beta_pres": float(beta_pres_value),
            "beta_abs": float(beta_abs_value),
            "delta": float(beta_pres_value - beta_abs_value),
        }
    return beta_pres, beta_abs, meta


def top_feature_table(
    stance_data: Mapping[str, Any],
    beta_pres_by_key: Mapping[str, float],
    beta_abs_by_key: Mapping[str, float],
    beta_profile_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in top_children(stance_data):
        bp = float(beta_pres_by_key[spec.key])
        ba = float(beta_abs_by_key[spec.key])
        present_contrib = math.log(clip_prob(bp) / clip_prob(ba))
        absent_contrib = math.log(clip_prob(1.0 - bp) / clip_prob(1.0 - ba))
        counts = descendant_counts(spec.node, (stance_data["name"],))
        rows.append(
            {
                "beta_profile": beta_profile_name,
                "top_feature_key": spec.key,
                "top_feature_name": spec.node["name"],
                "support_label": spec.node.get("support", "no bearing"),
                "demandingness_label": spec.node.get("demandingness", "neutral"),
                "beta_pres": bp,
                "beta_abs": ba,
                "beta_gap": bp - ba,
                "contribution_if_top_present": present_contrib,
                "contribution_if_top_absent": absent_contrib,
                **counts,
            }
        )
    return pd.DataFrame(rows)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probs: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    total = float(np.sum(w))
    if total <= 0:
        return np.full(len(probs), np.nan)
    cdf = np.cumsum(w) / total
    return np.asarray([v[min(np.searchsorted(cdf, p, side="left"), len(v) - 1)] for p in probs])


def enumerate_top_bound_distribution(
    *,
    stance_data: Mapping[str, Any],
    beta_pres_by_key: Mapping[str, float],
    beta_abs_by_key: Mapping[str, float],
    beta_profile_name: str,
) -> pd.DataFrame:
    children = top_children(stance_data)
    log_bs: list[float] = []
    w_r1: list[float] = []
    w_r0: list[float] = []
    state_strings: list[str] = []
    for state in itertools.product([0, 1], repeat=len(children)):
        log_b = 0.0
        p1 = 1.0
        p0 = 1.0
        for z, child in zip(state, children):
            bp = float(beta_pres_by_key[child.key])
            ba = float(beta_abs_by_key[child.key])
            bp_c = float(clip_prob(bp))
            ba_c = float(clip_prob(ba))
            if z:
                log_b += math.log(bp_c / ba_c)
                p1 *= bp_c
                p0 *= ba_c
            else:
                log_b += math.log((1.0 - bp_c) / (1.0 - ba_c))
                p1 *= 1.0 - bp_c
                p0 *= 1.0 - ba_c
        log_bs.append(log_b)
        w_r1.append(p1)
        w_r0.append(p0)
        state_strings.append("".join(str(x) for x in state))

    values = np.asarray(log_bs, dtype=float)
    weights_r1 = np.asarray(w_r1, dtype=float)
    weights_r0 = np.asarray(w_r0, dtype=float)
    if abs(weights_r1.sum() - 1.0) > 1e-8 or abs(weights_r0.sum() - 1.0) > 1e-8:
        raise AssertionError("Top-state enumeration probabilities do not sum to one")

    probs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    q_r1 = weighted_quantile(values, weights_r1, probs)
    q_r0 = weighted_quantile(values, weights_r0, probs)
    mean_r1 = float(np.sum(weights_r1 * values))
    mean_r0 = float(np.sum(weights_r0 * values))
    # Explicit consistency assertion for the requested top-bound test.
    if abs(mean_r1 - float(np.dot(weights_r1, values))) > 1e-12:
        raise AssertionError("Weighted top-bound mean consistency failed")

    median_r1 = float(q_r1[4])
    median_r0 = float(q_r0[4])
    frac_r1_gt_50 = float(np.sum(weights_r1[values > TAU_PRESENT_50]))
    frac_r0_lt_05 = float(np.sum(weights_r0[values < TAU_ABSENT_05]))
    if median_r1 <= TAU_PRESENT_50 or median_r0 >= TAU_ABSENT_05:
        gate = "fail"
    elif frac_r1_gt_50 < 0.60 or frac_r0_lt_05 < 0.60:
        gate = "borderline"
    else:
        gate = "pass"

    row: dict[str, Any] = {
        "beta_profile": beta_profile_name,
        "n_top_children": len(children),
        "n_states": len(values),
        "state_order": ";".join(child.node["name"] for child in children),
        "mean_log_B_top_R1": mean_r1,
        "mean_log_B_top_R0": mean_r0,
        "median_log_B_top_R1": median_r1,
        "median_log_B_top_R0": median_r0,
        "frac_R1_log_B_top_gt_0": float(np.sum(weights_r1[values > 0.0])),
        "frac_R1_log_B_top_gt_tau_present_50": frac_r1_gt_50,
        "frac_R1_log_B_top_gt_tau_present_95": float(np.sum(weights_r1[values > TAU_PRESENT_95])),
        "frac_R0_log_B_top_lt_0": float(np.sum(weights_r0[values < 0.0])),
        "frac_R0_log_B_top_lt_tau_absent_05": frac_r0_lt_05,
        "frac_R0_log_B_top_lt_tau_absent_01": float(np.sum(weights_r0[values < TAU_ABSENT_01])),
        "preflight_gate": gate,
        "min_log_B_top": float(np.min(values)),
        "max_log_B_top": float(np.max(values)),
    }
    q_names = ["q01", "q05", "q10", "q25", "q50", "q75", "q90", "q95", "q99"]
    for name, val in zip(q_names, q_r1):
        row[f"log_B_top_R1_{name}"] = float(val)
    for name, val in zip(q_names, q_r0):
        row[f"log_B_top_R0_{name}"] = float(val)
    row["enumerated_states"] = ";".join(
        f"{s}:{lb:.6g}:R1={p1:.6g}:R0={p0:.6g}"
        for s, lb, p1, p0 in zip(state_strings, values, weights_r1, weights_r0)
    )
    return pd.DataFrame([row])


def realised_top_log_b(
    *,
    stance_data: Mapping[str, Any],
    truth_payload: Mapping[str, Any],
    system: str,
    beta_pres_by_key: Mapping[str, float],
    beta_abs_by_key: Mapping[str, float],
) -> Dict[str, Any]:
    latent = truth_payload["latent_by_system"][system]
    internal_z = latent.get("internal_z", {})
    contributions = []
    n_present = 0
    min_total = 0.0
    max_total = 0.0
    for child in top_children(stance_data):
        z = int(internal_z[child.key])
        bp = float(clip_prob(beta_pres_by_key[child.key]))
        ba = float(clip_prob(beta_abs_by_key[child.key]))
        c_present = math.log(bp / ba)
        c_absent = math.log((1.0 - bp) / (1.0 - ba))
        contribution = c_present if z else c_absent
        contributions.append(contribution)
        n_present += z
        min_total += min(c_present, c_absent)
        max_total += max(c_present, c_absent)
    log_b = float(np.sum(contributions))
    return {
        "n_top_children": len(contributions),
        "n_top_present": int(n_present),
        "log_B_top_latent": log_b,
        "log_B_top_max": float(max_total),
        "log_B_top_min": float(min_total),
        "top_latent_category": evidence_category_from_log_b(log_b),
        "top_decisive_present_possible": bool(max_total >= TAU_PRESENT_50),
        "top_decisive_absent_possible": bool(min_total <= TAU_ABSENT_05),
    }


def posterior_flat(post: Any, var: str, draw_idx: Optional[np.ndarray] = None) -> np.ndarray:
    arr = np.asarray(post[var].values)
    flat = arr.reshape((-1,) + arr.shape[2:])
    if draw_idx is not None:
        flat = flat[draw_idx]
    return flat


def make_draw_index(n_draws: int, fallback_draws: Optional[int]) -> Optional[np.ndarray]:
    if fallback_draws is None or fallback_draws >= n_draws:
        return None
    if fallback_draws <= 0:
        raise ValueError("--fallback-draws must be 'all' or a positive integer")
    return np.unique(np.linspace(0, n_draws - 1, fallback_draws).astype(int))


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, [c for c in columns if c in df.columns]].head(max_rows).copy()
    cols = list(view.columns)
    if not cols:
        return "_No columns._"
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in view.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append("nan" if pd.isna(val) else f"{float(val):.3f}")
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def format_float(x: Any, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "nan"
    return f"{float(x):.{digits}f}"


def ordered_probit_category_logp(
    ratings: np.ndarray,
    expert_idx: np.ndarray,
    a: float,
    b: np.ndarray,
    kappa: np.ndarray,
    eta_offset_factor: float,
) -> float:
    if ratings.size == 0:
        return 0.0
    eta = b[expert_idx] + eta_offset_factor * float(a)
    cum_full = np.empty((ratings.size, len(kappa) + 2))
    cum_full[:, 0] = 0.0
    cum_full[:, -1] = 1.0
    for k, cut in enumerate(kappa):
        cum_full[:, k + 1] = norm.cdf(cut - eta)
    probs = np.clip(np.diff(cum_full, axis=1), SCORE_CLIP, 1.0)
    chosen = probs[np.arange(ratings.size), ratings]
    return float(np.log(chosen).sum())


def sample_ordered_probit_category(
    rng: np.random.Generator,
    a: float,
    kappa: np.ndarray,
    eta_offset_factor: float,
) -> int:
    eta = float(a) * float(eta_offset_factor)
    cum = np.concatenate(([0.0], norm.cdf(kappa - eta), [1.0]))
    probs = np.clip(np.diff(cum), SCORE_CLIP, 1.0)
    probs = probs / probs.sum()
    return int(rng.choice(np.arange(len(probs)), p=probs))


def exact_root_sides_from_leaf_messages(
    stance_data: Mapping[str, Any],
    leaf_edge_messages: Mapping[str, Tuple[float, float]],
    beta_pres_by_key: Mapping[str, float],
    beta_abs_by_key: Mapping[str, float],
) -> Tuple[float, float]:
    """Return root-side log likelihoods (R=0, R=1) from leaf edge messages.

    ``leaf_edge_messages[key]`` is ``(log L_leaf(z_parent=0),
    log L_leaf(z_parent=1))`` for indicator key.
    """
    root_path = (stance_data["name"],)

    def subtree(node: Mapping[str, Any], path: Tuple[str, ...]) -> Tuple[float, float]:
        current_path = path + (node["name"],)
        key = node_key(path, node["name"])
        if (node.get("type") or "").lower() == "indicator":
            return leaf_edge_messages.get(key, (0.0, 0.0))

        log_l_v0 = 0.0
        log_l_v1 = 0.0
        for child in node.get("evidencers", []):
            c0, c1 = subtree(child, current_path)
            log_l_v0 += c0
            log_l_v1 += c1
        bp = float(clip_prob(beta_pres_by_key[key]))
        ba = float(clip_prob(beta_abs_by_key[key]))
        log_l_zpa1 = float(np.logaddexp(math.log(bp) + log_l_v1, math.log1p(-bp) + log_l_v0))
        log_l_zpa0 = float(np.logaddexp(math.log(ba) + log_l_v1, math.log1p(-ba) + log_l_v0))
        return log_l_zpa0, log_l_zpa1

    log_l0 = 0.0
    log_l1 = 0.0
    for child in stance_data.get("evidencers", []):
        c0, c1 = subtree(child, root_path)
        log_l0 += c0
        log_l1 += c1
    return float(log_l0), float(log_l1)
