"""Oracle internal-node identifiability audit for the GWT exact tree.

This script targets Arvo's question directly:

    Are internal feature/subfeature binary states practically identifiable
    under the current GWT rating design?

It is an M-closed oracle check.  The DGP is the exact latent-state tree, the
leaf is the three-state indicator model, and nuisance parameters are fixed to
known truth.  For each simulated dataset it computes
Pr(z_sv = 1 | y, theta*) by exact clamped dynamic programming, then scores
those probabilities against the simulated latent states.

No PyMC sampling is run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse_tree_pooling import pooled_beta_draws_by_node  # noqa: E402
from dcm_model import EvidenceProcessor, ModelConfig, load_data, node_key  # noqa: E402
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS  # noqa: E402


STANCE = "Global Workspace Theory"
EXACT_PROD_PATH = (
    REPO_ROOT / "results/gwt_exact_tree/three_state_pooled_abs_by_sd_exact_anchored.nc"
)
DEFAULT_RUNS_DIR = (
    REPO_ROOT / "notebooks/synthetic_validation_2026-05-06/runs"
    / "oracle_internal_identifiability"
)

LABELS = {
    "dgp": "exact_latent_tree",
    "fit": "oracle",
    "leaf": "three_state_binomial_2",
    "design": "current_gwt_rater_design",
}

TRUE_C_BY_SYSTEM = {
    "Human": 0.999,
    "Chicken": 0.25,
    "2024 Leading Chat LLMs": 0.10,
    "ELIZA": 0.001,
}

N_SYSTEM_LABEL = {
    "Human": "Human",
    "Chicken": "Chicken",
    "2024 Leading Chat LLMs": "LLMs",
    "ELIZA": "ELIZA",
}


@dataclass(frozen=True)
class NodeMeta:
    node_key: str
    node_name: str
    node_type: str
    depth: int
    fanout: int
    subtree_indicator_count: int
    subtree_rating_count_by_system: Dict[str, int]
    top_feature: str


@dataclass(frozen=True)
class OracleTruth:
    nuisance_truth: str
    edge_betas: Dict[str, Dict[str, Any]]
    obs_params: Dict[str, Any]


def is_missing(val: Any) -> bool:
    if val is None:
        return True
    return str(val).strip().lower() in {"-1", "-1.0", "none", "", "unsure", "not tested"}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def clipped_prob(p: float, eps: float = 1e-12) -> float:
    return float(min(max(p, eps), 1.0 - eps))


def binary_entropy(p: float) -> float:
    p = clipped_prob(p)
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def ordered_probit_probs(kappa: np.ndarray, eta: float) -> np.ndarray:
    cum = np.concatenate(([0.0], norm.cdf(kappa - eta), [1.0]))
    probs = np.diff(cum)
    probs = np.clip(probs, 1e-12, 1.0)
    return probs / probs.sum()


def sample_rating_category(
    rng: np.random.Generator,
    kappa: np.ndarray,
    a: float,
    m: int,
) -> int:
    probs = ordered_probit_probs(kappa, a * (m / 2.0))
    return int(rng.choice(np.arange(len(probs)), p=probs))


def iter_tree_nodes(
    stance_data: Dict[str, Any],
) -> Iterable[Tuple[Dict[str, Any], str, Tuple[str, ...], int]]:
    """Yield every non-root node with key, ancestor path, and depth."""
    root_path = (stance_data["name"],)

    def walk(node: Dict[str, Any], ancestor_path: Tuple[str, ...], depth: int):
        key = node_key(ancestor_path, node["name"])
        yield node, key, ancestor_path, depth
        current_path = ancestor_path + (node["name"],)
        for child in node.get("evidencers", []):
            yield from walk(child, current_path, depth + 1)

    for child in stance_data.get("evidencers", []):
        yield from walk(child, root_path, 1)


def subtree_counts(
    node: Dict[str, Any],
    systems: Sequence[str],
) -> Tuple[int, Dict[str, int]]:
    """Return descendant indicator count and current-design rating counts."""
    ntype = (node.get("type") or "").lower()
    if ntype == "indicator":
        by_system: Dict[str, int] = {}
        for sys_name in systems:
            obs = node.get("observations", {}).get(sys_name)
            by_system[sys_name] = (
                0
                if not obs
                else sum(not is_missing(v) for v in obs.get("values", []))
            )
        return 1, by_system

    total_indicators = 0
    by_system = {sys_name: 0 for sys_name in systems}
    for child in node.get("evidencers", []):
        child_indicators, child_counts = subtree_counts(child, systems)
        total_indicators += child_indicators
        for sys_name, n in child_counts.items():
            by_system[sys_name] += n
    return total_indicators, by_system


def collect_internal_node_meta(
    stance_data: Dict[str, Any],
    systems: Sequence[str],
) -> Dict[str, NodeMeta]:
    meta: Dict[str, NodeMeta] = {}
    root_path = (stance_data["name"],)

    def walk(
        node: Dict[str, Any],
        ancestor_path: Tuple[str, ...],
        depth: int,
        top_feature: str,
    ) -> None:
        current_path = ancestor_path + (node["name"],)
        key = node_key(ancestor_path, node["name"])
        ntype = (node.get("type") or "").lower()
        if ntype in {"feature", "subfeature"}:
            n_indicators, rating_counts = subtree_counts(node, systems)
            children = node.get("evidencers", [])
            meta[key] = NodeMeta(
                node_key=key,
                node_name=node["name"],
                node_type=ntype,
                depth=depth,
                fanout=len(children),
                subtree_indicator_count=n_indicators,
                subtree_rating_count_by_system=rating_counts,
                top_feature=top_feature,
            )
        for child in node.get("evidencers", []):
            walk(child, current_path, depth + 1, top_feature)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path, 1, child["name"])
    return meta


def collect_paper_mean_edge_betas(
    stance_data: Dict[str, Any],
    evidence_processor: EvidenceProcessor,
) -> Dict[str, Dict[str, Any]]:
    edge_betas: Dict[str, Dict[str, Any]] = {}
    for node, key, _, _ in iter_tree_nodes(stance_data):
        support = node.get("support", "no bearing")
        demandingness = node.get("demandingness", "neutral")
        alpha_p, beta_p, alpha_a, beta_a = evidence_processor.get_beta_parameters(
            support,
            demandingness,
        )
        bp = alpha_p / (alpha_p + beta_p)
        ba = alpha_a / (alpha_a + beta_a)
        edge_betas[key] = {
            "node_name": node["name"],
            "node_type": (node.get("type") or "").lower(),
            "support": support,
            "demandingness": demandingness,
            "beta_pres": float(bp),
            "beta_abs": float(ba),
            "delta": float(bp - ba),
        }
    return edge_betas


def collect_production_median_edge_betas(
    idata: Any,
    stance_data: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    beta_pres_by_key, beta_abs_by_key = pooled_beta_draws_by_node(idata, stance_data)
    edge_betas: Dict[str, Dict[str, Any]] = {}
    for node, key, _, _ in iter_tree_nodes(stance_data):
        bp = float(np.median(beta_pres_by_key[key]))
        ba = float(np.median(beta_abs_by_key[key]))
        edge_betas[key] = {
            "node_name": node["name"],
            "node_type": (node.get("type") or "").lower(),
            "support": node.get("support", "no bearing"),
            "demandingness": node.get("demandingness", "neutral"),
            "beta_pres": bp,
            "beta_abs": ba,
            "delta": float(bp - ba),
        }
    return edge_betas


def load_observation_medians(idata: Any, k: int) -> Dict[str, Any]:
    post = idata.posterior
    a = float(np.median(np.asarray(post["a"].values).reshape(-1)))
    kappa = np.median(
        np.asarray(post["kappa"].values).reshape(-1, k - 1),
        axis=0,
    )
    return {
        "source": str(EXACT_PROD_PATH.relative_to(REPO_ROOT)),
        "a": a,
        "kappa": [float(x) for x in kappa],
        "use_expert_shifts": False,
        "expert_shift": 0.0,
    }


def load_oracle_truth(
    nuisance_truth: str,
    stance_data: Dict[str, Any],
    cfg: ModelConfig,
) -> OracleTruth:
    if not EXACT_PROD_PATH.exists():
        raise FileNotFoundError(f"Missing exact production posterior: {EXACT_PROD_PATH}")
    idata = az.from_netcdf(EXACT_PROD_PATH)
    obs_params = load_observation_medians(idata, cfg.N_CATEGORIES)

    if nuisance_truth == "exact_tree_production_medians":
        edge_betas = collect_production_median_edge_betas(idata, stance_data)
    elif nuisance_truth == "paper_mean_tree_transmission":
        edge_betas = collect_paper_mean_edge_betas(
            stance_data,
            EvidenceProcessor(cfg),
        )
    else:
        raise ValueError(f"Unsupported nuisance truth: {nuisance_truth!r}")

    return OracleTruth(
        nuisance_truth=nuisance_truth,
        edge_betas=edge_betas,
        obs_params=obs_params,
    )


def sample_latent_tree_for_system(
    rng: np.random.Generator,
    stance_data: Dict[str, Any],
    edge_betas: Mapping[str, Mapping[str, float]],
    true_c: float,
) -> Dict[str, Any]:
    root_z = int(rng.binomial(1, true_c))
    internal_z: Dict[str, int] = {}
    indicator_m: Dict[str, int] = {}
    root_path = (stance_data["name"],)

    def walk(node: Dict[str, Any], ancestor_path: Tuple[str, ...], parent_z: int) -> None:
        key = node_key(ancestor_path, node["name"])
        beta = (
            edge_betas[key]["beta_pres"]
            if parent_z
            else edge_betas[key]["beta_abs"]
        )
        ntype = (node.get("type") or "").lower()
        if ntype == "indicator":
            indicator_m[key] = int(rng.binomial(2, beta))
            return

        z = int(rng.binomial(1, beta))
        internal_z[key] = z
        current_path = ancestor_path + (node["name"],)
        for child in node.get("evidencers", []):
            walk(child, current_path, z)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path, root_z)

    return {
        "root_z": root_z,
        "internal_z": internal_z,
        "indicator_m": indicator_m,
    }


def simulate_ratings(
    rng: np.random.Generator,
    stance_data: Dict[str, Any],
    latent_by_system: Mapping[str, Mapping[str, Any]],
    obs_params: Mapping[str, Any],
    systems: Sequence[str],
) -> Dict[str, Dict[str, List[int]]]:
    a = float(obs_params["a"])
    kappa = np.asarray(obs_params["kappa"], dtype=float)
    ratings_by_system: Dict[str, Dict[str, List[int]]] = {
        sys_name: {} for sys_name in systems
    }

    def walk(node: Dict[str, Any], ancestor_path: Tuple[str, ...]) -> None:
        key = node_key(ancestor_path, node["name"])
        current_path = ancestor_path + (node["name"],)
        if (node.get("type") or "").lower() == "indicator":
            for sys_name in systems:
                obs = node.get("observations", {}).get(sys_name)
                if not obs:
                    continue
                m = int(latent_by_system[sys_name]["indicator_m"][key])
                values = obs.get("values", [])
                ratings = [
                    sample_rating_category(rng, kappa, a, m)
                    for val in values
                    if not is_missing(val)
                ]
                if ratings:
                    ratings_by_system[sys_name][key] = ratings
            return
        for child in node.get("evidencers", []):
            walk(child, current_path)

    for child in stance_data.get("evidencers", []):
        walk(child, (stance_data["name"],))
    return ratings_by_system


def leaf_log_likelihood_terms(
    ratings: Sequence[int],
    obs_params: Mapping[str, Any],
) -> np.ndarray:
    """Return log P(y_j | m) for m=0,1,2 under shared ordered probit."""
    if not ratings:
        return np.zeros(3)
    a = float(obs_params["a"])
    kappa = np.asarray(obs_params["kappa"], dtype=float)
    out = np.zeros(3)
    for m in (0, 1, 2):
        probs = ordered_probit_probs(kappa, a * (m / 2.0))
        out[m] = float(sum(math.log(probs[int(r)]) for r in ratings))
    return out


def three_state_log_mix(beta: float, leaf_ll: np.ndarray) -> float:
    beta = clipped_prob(beta)
    log_w = np.array(
        [
            2.0 * math.log(1.0 - beta),
            math.log(2.0) + math.log(beta) + math.log(1.0 - beta),
            2.0 * math.log(beta),
        ]
    )
    return float(logsumexp(log_w + leaf_ll))


def exact_log_evidence_with_optional_clamp(
    stance_data: Dict[str, Any],
    system_name: str,
    ratings_by_indicator: Mapping[str, Sequence[int]],
    edge_betas: Mapping[str, Mapping[str, float]],
    obs_params: Mapping[str, Any],
    true_c: float,
    target_key: Optional[str] = None,
    target_state: Optional[int] = None,
) -> float:
    """Compute log P(y, optional z_target=state | theta*) exactly."""
    if (target_key is None) != (target_state is None):
        raise ValueError("target_key and target_state must be supplied together")

    root_path = (stance_data["name"],)

    def subtree_message(
        node: Dict[str, Any],
        ancestor_path: Tuple[str, ...],
    ) -> Tuple[float, float]:
        """Return log messages as a function of parent state: (parent=0, parent=1)."""
        key = node_key(ancestor_path, node["name"])
        ntype = (node.get("type") or "").lower()
        bp = float(edge_betas[key]["beta_pres"])
        ba = float(edge_betas[key]["beta_abs"])

        if ntype == "indicator":
            leaf_ll = leaf_log_likelihood_terms(
                ratings_by_indicator.get(key, []),
                obs_params,
            )
            return (
                three_state_log_mix(ba, leaf_ll),
                three_state_log_mix(bp, leaf_ll),
            )

        current_path = ancestor_path + (node["name"],)
        log_u0 = 0.0
        log_u1 = 0.0
        for child in node.get("evidencers", []):
            child_parent0, child_parent1 = subtree_message(child, current_path)
            log_u0 += child_parent0
            log_u1 += child_parent1

        if key == target_key:
            if target_state == 1:
                return (
                    math.log(clipped_prob(ba)) + log_u1,
                    math.log(clipped_prob(bp)) + log_u1,
                )
            return (
                math.log(clipped_prob(1.0 - ba)) + log_u0,
                math.log(clipped_prob(1.0 - bp)) + log_u0,
            )

        log_msg_parent0 = float(
            logsumexp(
                [
                    math.log(clipped_prob(1.0 - ba)) + log_u0,
                    math.log(clipped_prob(ba)) + log_u1,
                ]
            )
        )
        log_msg_parent1 = float(
            logsumexp(
                [
                    math.log(clipped_prob(1.0 - bp)) + log_u0,
                    math.log(clipped_prob(bp)) + log_u1,
                ]
            )
        )
        return log_msg_parent0, log_msg_parent1

    log_top0 = 0.0
    log_top1 = 0.0
    for child in stance_data.get("evidencers", []):
        child_parent0, child_parent1 = subtree_message(child, root_path)
        log_top0 += child_parent0
        log_top1 += child_parent1

    true_c = clipped_prob(true_c)
    return float(
        logsumexp(
            [
                math.log(1.0 - true_c) + log_top0,
                math.log(true_c) + log_top1,
            ]
        )
    )


def posterior_internal_probability(
    stance_data: Dict[str, Any],
    system_name: str,
    ratings_by_indicator: Mapping[str, Sequence[int]],
    edge_betas: Mapping[str, Mapping[str, float]],
    obs_params: Mapping[str, Any],
    true_c: float,
    target_key: str,
    log_evidence: Optional[float] = None,
) -> float:
    log_y = (
        exact_log_evidence_with_optional_clamp(
            stance_data,
            system_name,
            ratings_by_indicator,
            edge_betas,
            obs_params,
            true_c,
        )
        if log_evidence is None
        else log_evidence
    )
    log_joint1 = exact_log_evidence_with_optional_clamp(
        stance_data,
        system_name,
        ratings_by_indicator,
        edge_betas,
        obs_params,
        true_c,
        target_key=target_key,
        target_state=1,
    )
    return float(np.exp(log_joint1 - log_y))


def prior_internal_probabilities(
    stance_data: Dict[str, Any],
    edge_betas: Mapping[str, Mapping[str, float]],
    true_c: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    root_path = (stance_data["name"],)

    def walk(node: Dict[str, Any], ancestor_path: Tuple[str, ...], parent_q: float) -> None:
        key = node_key(ancestor_path, node["name"])
        bp = float(edge_betas[key]["beta_pres"])
        ba = float(edge_betas[key]["beta_abs"])
        q = ba + parent_q * (bp - ba)
        if (node.get("type") or "").lower() in {"feature", "subfeature"}:
            out[key] = float(q)
        current_path = ancestor_path + (node["name"],)
        for child in node.get("evidencers", []):
            walk(child, current_path, q)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path, true_c)
    return out


def subtree_rating_bin(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    if n <= 10:
        return "6-10"
    if n <= 25:
        return "11-25"
    if n <= 50:
        return "26-50"
    return "51+"


def simulate_and_score_once(
    rng: np.random.Generator,
    sim_id: int,
    stance_data: Dict[str, Any],
    systems: Sequence[str],
    internal_meta: Mapping[str, NodeMeta],
    truth: OracleTruth,
) -> List[Dict[str, Any]]:
    latent_by_system = {
        sys_name: sample_latent_tree_for_system(
            rng,
            stance_data,
            truth.edge_betas,
            TRUE_C_BY_SYSTEM[sys_name],
        )
        for sys_name in systems
    }
    ratings_by_system = simulate_ratings(
        rng,
        stance_data,
        latent_by_system,
        truth.obs_params,
        systems,
    )

    rows: List[Dict[str, Any]] = []
    for sys_name in systems:
        prior_probs = prior_internal_probabilities(
            stance_data,
            truth.edge_betas,
            TRUE_C_BY_SYSTEM[sys_name],
        )
        log_evidence = exact_log_evidence_with_optional_clamp(
            stance_data,
            sys_name,
            ratings_by_system[sys_name],
            truth.edge_betas,
            truth.obs_params,
            TRUE_C_BY_SYSTEM[sys_name],
        )
        for key, meta in internal_meta.items():
            p_post = posterior_internal_probability(
                stance_data,
                sys_name,
                ratings_by_system[sys_name],
                truth.edge_betas,
                truth.obs_params,
                TRUE_C_BY_SYSTEM[sys_name],
                key,
                log_evidence=log_evidence,
            )
            p_post = clipped_prob(p_post)
            p_prior = clipped_prob(prior_probs[key])
            z_true = int(latent_by_system[sys_name]["internal_z"][key])
            prior_h = binary_entropy(p_prior)
            post_h = binary_entropy(p_post)
            entropy_reduction = prior_h - post_h
            rel_entropy_reduction = (
                entropy_reduction / prior_h if prior_h > 1e-12 else math.nan
            )
            log_score = math.log(p_post if z_true else 1.0 - p_post)
            subtree_n = meta.subtree_rating_count_by_system[sys_name]
            rows.append(
                {
                    **LABELS,
                    "nuisance_truth": truth.nuisance_truth,
                    "audit": "internal_node_identifiability",
                    "simulation": sim_id,
                    "system": sys_name,
                    "system_label": N_SYSTEM_LABEL.get(sys_name, sys_name),
                    "true_C": TRUE_C_BY_SYSTEM[sys_name],
                    "root_z_true": int(latent_by_system[sys_name]["root_z"]),
                    "node_key": key,
                    "node_name": meta.node_name,
                    "node_type": meta.node_type,
                    "top_feature": meta.top_feature,
                    "depth": meta.depth,
                    "fanout": meta.fanout,
                    "subtree_indicator_count": meta.subtree_indicator_count,
                    "subtree_rating_count": subtree_n,
                    "subtree_rating_count_bin": subtree_rating_bin(subtree_n),
                    "z_true": z_true,
                    "prior_p_z1": p_prior,
                    "posterior_p_z1": p_post,
                    "prior_entropy": prior_h,
                    "posterior_entropy": post_h,
                    "entropy_reduction": entropy_reduction,
                    "relative_entropy_reduction": rel_entropy_reduction,
                    "brier": float((p_post - z_true) ** 2),
                    "log_score": float(log_score),
                    "neg_log_score": float(-log_score),
                }
            )
    return rows


def summarise_group(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    grouped = df.groupby(list(group_cols), dropna=False)
    out = grouped.agg(
        n=("posterior_p_z1", "size"),
        mean_posterior_p=("posterior_p_z1", "mean"),
        empirical_z_rate=("z_true", "mean"),
        mean_prior_entropy=("prior_entropy", "mean"),
        mean_posterior_entropy=("posterior_entropy", "mean"),
        mean_entropy_reduction=("entropy_reduction", "mean"),
        mean_relative_entropy_reduction=("relative_entropy_reduction", "mean"),
        mean_brier=("brier", "mean"),
        mean_neg_log_score=("neg_log_score", "mean"),
    ).reset_index()
    for key, value in LABELS.items():
        out[key] = value
    out["audit"] = "internal_node_identifiability"
    if "nuisance_truth" not in out:
        out["nuisance_truth"] = str(df["nuisance_truth"].iloc[0])
    return out


def build_stratified_summary(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for col in [
        "system",
        "depth",
        "fanout",
        "subtree_rating_count_bin",
        "top_feature",
    ]:
        summary = summarise_group(df, [col])
        summary.insert(0, "grouping", col)
        summary = summary.rename(columns={col: "group_value"})
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def build_calibration_table(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # include 1.0 in the last bin while keeping labels stable
    df_cal = df.copy()
    df_cal["prob_bin"] = pd.cut(
        df_cal["posterior_p_z1"],
        bins=bins,
        include_lowest=True,
        right=False,
    )
    df_cal.loc[df_cal["posterior_p_z1"] >= 1.0, "prob_bin"] = df_cal[
        "prob_bin"
    ].cat.categories[-1]
    out = summarise_group(df_cal, ["prob_bin"])
    out["prob_bin"] = out["prob_bin"].astype(str)
    return out


def fmt_float(x: Any, digits: int = 3) -> str:
    if pd.isna(x):
        return "nan"
    return f"{float(x):.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("|" + "|".join(["---"] * len(columns)) + "|")
    for _, row in df.iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(fmt_float(val))
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return rows


def metric_value(df: pd.DataFrame, grouping: str, group_value: Any, column: str) -> float:
    mask = (df["grouping"] == grouping) & (df["group_value"].astype(str) == str(group_value))
    if not mask.any():
        return math.nan
    return float(df.loc[mask, column].iloc[0])


def plain_language_interpretation(
    labels: Mapping[str, str],
    overall: pd.DataFrame,
    stratified: pd.DataFrame,
) -> List[str]:
    overall_rel = float(overall["mean_relative_entropy_reduction"].iloc[0])
    overall_brier = float(overall["mean_brier"].iloc[0])
    depth3_rel = metric_value(
        stratified,
        "depth",
        3,
        "mean_relative_entropy_reduction",
    )
    zero_rating_rel = metric_value(
        stratified,
        "subtree_rating_count_bin",
        "0",
        "mean_relative_entropy_reduction",
    )
    fanout6_rel = metric_value(
        stratified,
        "fanout",
        6,
        "mean_relative_entropy_reduction",
    )

    if labels["nuisance_truth"] == "paper_mean_tree_transmission":
        nuisance_sentence = (
            "Using paper-mean tree transmission makes the internal states much "
            "harder to learn. This is the conservative stress case: if these "
            "edge strengths are closer to reality, many middle-layer posteriors "
            "will be only weakly data-driven."
        )
    else:
        nuisance_sentence = (
            "Using production-median nuisance truth is a best-case version of "
            "the current exact-tree model: the observation layer and tree "
            "transmission are set to values the real-data fit already found "
            "plausible."
        )

    return [
        "## Plain-English Interpretation",
        "",
        "This audit asks a practical question: if the exact binary GWT tree were "
        "really how expert ratings are generated, would the current rating "
        "design let us learn the hidden feature/subfeature states?",
        "",
        f"Short answer: the answer is mixed. On average the ratings remove about "
        f"{fmt_float(overall_rel)} of the prior uncertainty about internal "
        f"states, with mean Brier score {fmt_float(overall_brier)}. That means "
        "the internal probabilities are not purely arbitrary, but they are also "
        "not uniformly strong across the tree.",
        "",
        nuisance_sentence,
        "",
        "For Arvo's question, the important point is that internal states should "
        "not be interpreted as recovered true/false labels. They are posterior "
        "probabilities. Some feature blocks have enough descendant ratings and "
        "tree transmission to move those probabilities substantially; other "
        "nodes mostly inherit information from ancestors and priors.",
        "",
        f"The clearest weak spot is the deep/empty part of the tree: depth-3 "
        f"nodes have mean relative entropy reduction {fmt_float(depth3_rel)}, "
        f"and nodes with zero subtree ratings have {fmt_float(zero_rating_rel)}. "
        "Those nodes should not support strong scientific claims in the current "
        "design.",
        "",
        f"The clearest strong spot is broad, connected structure: fanout-6 nodes "
        f"have mean relative entropy reduction {fmt_float(fanout6_rel)}. These "
        "are the kinds of internal summaries where the exact-tree model has "
        "actual observed-scale leverage.",
        "",
        "What this means for the model results: root `C_s` summaries and "
        "well-observed feature-block probabilities are more defensible than "
        "hard claims about every individual subfeature. The right reporting "
        "style is probability plus uncertainty/entropy by node, not a table of "
        "binary recovered states.",
        "",
        "How to read the metrics:",
        "",
        "- Relative entropy reduction is the fraction of uncertainty removed by "
        "the ratings: 0 means no learning beyond the prior/tree, 1 means near "
        "complete resolution.",
        "- Brier score is a probability-error score: lower is better; around "
        "0.25 is what a non-informative 50/50 probability gets for balanced "
        "binary states.",
        "- Calibration compares probability bins to the simulated truth rate. "
        "Good calibration means a bin around 0.8 contains true `z=1` states "
        "about 80% of the time.",
        "",
        "This is a best-case information check, not a proof that the real world "
        "is an exact binary tree. If a node is weak here, a full real-data fit "
        "cannot make it strongly data-driven without relying on priors or model "
        "structure. If a node is strong here, it is at least identifiable in "
        "principle under the current design.",
    ]


def write_markdown_summary(
    out_dir: Path,
    labels: Mapping[str, str],
    n_sims: int,
    n_internal_nodes: int,
    overall: pd.DataFrame,
    stratified: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    metric_cols = [
        "n",
        "mean_relative_entropy_reduction",
        "mean_brier",
        "mean_neg_log_score",
    ]
    lines: List[str] = [
        "# Oracle Internal-Node Identifiability Summary",
        "",
        "Labels:",
        "",
        f"- DGP: `{labels['dgp']}`",
        f"- Fit: `{labels['fit']}`",
        f"- Leaf: `{labels['leaf']}`",
        f"- Nuisance truth: `{labels['nuisance_truth']}`",
        f"- Design: `{labels['design']}`",
        "",
        f"Simulations: {n_sims}",
        f"Internal nodes per system: {n_internal_nodes}",
        "",
        *plain_language_interpretation(labels, overall, stratified),
        "",
        "## Overall",
        "",
        *markdown_table(overall, ["nuisance_truth", *metric_cols]),
        "",
        "## By System",
        "",
    ]

    by_system = stratified[stratified["grouping"] == "system"][
        ["group_value", *metric_cols]
    ]
    lines.extend(markdown_table(by_system, ["group_value", *metric_cols]))

    lines.extend(["", "## By Depth", ""])
    by_depth = stratified[stratified["grouping"] == "depth"][
        ["group_value", *metric_cols]
    ]
    lines.extend(markdown_table(by_depth, ["group_value", *metric_cols]))

    lines.extend(["", "## By Fanout", ""])
    by_fanout = stratified[stratified["grouping"] == "fanout"][
        ["group_value", *metric_cols]
    ]
    lines.extend(markdown_table(by_fanout, ["group_value", *metric_cols]))

    lines.extend(["", "## By Subtree Rating Count", ""])
    by_subtree = stratified[stratified["grouping"] == "subtree_rating_count_bin"][
        ["group_value", *metric_cols]
    ]
    lines.extend(markdown_table(by_subtree, ["group_value", *metric_cols]))

    lines.extend(["", "## Calibration", ""])
    cal_cols = ["prob_bin", "n", "mean_posterior_p", "empirical_z_rate", "mean_brier"]
    lines.extend(markdown_table(calibration, cal_cols))
    lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--n-sims", type=int, default=50)
    parser.add_argument(
        "--nuisance-truth",
        choices=["exact_tree_production_medians", "paper_mean_tree_transmission"],
        default="exact_tree_production_medians",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_sims < 1:
        raise ValueError("--n-sims must be at least 1")

    labels = {**LABELS, "nuisance_truth": args.nuisance_truth}
    run_id = args.run_id or (
        f"{labels['dgp']}__{labels['fit']}__{labels['leaf']}__"
        f"{labels['nuisance_truth']}__{labels['design']}__seed{args.seed}"
    )
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else REPO_ROOT / args.runs_dir
    out_dir = runs_dir / run_id
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Run directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=args.nuisance_truth == "exact_tree_production_medians",
        BETA_ABS_BY_SUPPORT_DEMAND=args.nuisance_truth
        == "exact_tree_production_medians",
    )
    systems = [s for s, _ in ANCHORED_SYSTEM_CONFIGS]
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    internal_meta = collect_internal_node_meta(stance_data, systems)
    truth = load_oracle_truth(args.nuisance_truth, stance_data, cfg)
    rng = np.random.default_rng(args.seed)

    rows: List[Dict[str, Any]] = []
    for sim_id in range(args.n_sims):
        rows.extend(
            simulate_and_score_once(
                rng,
                sim_id,
                stance_data,
                systems,
                internal_meta,
                truth,
            )
        )

    node_df = pd.DataFrame(rows)
    stratified = build_stratified_summary(node_df)
    calibration = build_calibration_table(node_df, args.calibration_bins)
    overall = summarise_group(node_df, ["nuisance_truth"])

    node_df.to_csv(out_dir / "node_probabilities.csv", index=False)
    stratified.to_csv(out_dir / "stratified_summary.csv", index=False)
    calibration.to_csv(out_dir / "calibration.csv", index=False)
    overall.to_csv(out_dir / "overall_summary.csv", index=False)

    config_payload = {
        "labels": labels,
        "audit": "internal_node_identifiability",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_simulations": args.n_sims,
        "stance": STANCE,
        "systems": systems,
        "true_C_by_system": TRUE_C_BY_SYSTEM,
        "model_config": asdict(cfg),
        "model_config_note": (
            "Used for data loading and oracle semantic metadata. Edge betas are "
            "fixed from edge_beta_source rather than sampled from this config."
        ),
        "production_fit_source": str(EXACT_PROD_PATH.relative_to(REPO_ROOT)),
        "observation_parameters": truth.obs_params,
        "edge_beta_source": args.nuisance_truth,
        "outputs": {
            "node_probabilities": "node_probabilities.csv",
            "stratified_summary": "stratified_summary.csv",
            "calibration": "calibration.csv",
            "overall_summary": "overall_summary.csv",
            "summary": "summary.md",
        },
    }
    write_json(out_dir / "config.json", config_payload)
    write_markdown_summary(
        out_dir,
        labels,
        args.n_sims,
        len(internal_meta),
        overall,
        stratified,
        calibration,
    )

    print("=== Oracle internal-node identifiability audit ===")
    print(f"labels: {labels}")
    print(f"simulations: {args.n_sims}")
    print(f"internal nodes: {len(internal_meta)}")
    print(f"rows: {len(node_df)}")
    print()
    print("overall:")
    display_cols = [
        "n",
        "mean_relative_entropy_reduction",
        "mean_brier",
        "mean_neg_log_score",
    ]
    print(overall[display_cols].to_string(index=False))
    print()
    print("by system:")
    by_system = stratified[stratified["grouping"] == "system"][
        [
            "group_value",
            "n",
            "mean_relative_entropy_reduction",
            "mean_brier",
            "mean_neg_log_score",
        ]
    ]
    print(by_system.to_string(index=False))
    print(f"\nwrote {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
