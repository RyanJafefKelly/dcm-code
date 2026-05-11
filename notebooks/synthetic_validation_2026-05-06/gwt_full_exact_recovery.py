"""Full-GWT exact/exact fake-data recovery pilot.

This is the one-seed M-closed recovery run:

1. Generate synthetic GWT ratings from the exact latent-state tree.
2. Fit the same exact-tree model to those synthetic ratings.
3. Score recovery of scientific estimands and observed-scale summaries.

The run is intentionally not a final calibration study.  It builds the full
fake-data recovery pipeline and gives one realistic exact/exact pilot before
deciding whether a multi-seed study is worth the compute.
"""

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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import logsumexp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analyse_tree_pooling import (  # noqa: E402
    _beta_abs_var_name,
    _beta_pres_var_name,
    pooled_beta_draws_by_node,
)
from dcm_model import (  # noqa: E402
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    load_data,
    node_key,
)
from dcm_model_exact_tree import MultiSystemExactTreeBuilder  # noqa: E402
from gwt_reference_recovery_analysis import (  # noqa: E402
    ANCHORED_SYSTEM_CONFIGS,
    propagate_affine_indicator_coefficients_from_draws,
)
from gwt_oracle_internal_identifiability import (  # noqa: E402
    EXACT_PROD_PATH,
    STANCE,
    TRUE_C_BY_SYSTEM,
    binary_entropy,
    clipped_prob,
    collect_internal_node_meta,
    is_missing,
    load_oracle_truth,
    ordered_probit_probs,
    sample_latent_tree_for_system,
    sample_rating_category,
    write_json,
)


DEFAULT_RUNS_DIR = (
    REPO_ROOT / "notebooks/synthetic_validation_2026-05-06/runs"
    / "full_exact_recovery"
)
CATEGORY_REPRESENTATIVES = (0.025, 0.125, 0.300, 0.500, 0.700, 0.875, 0.975)
LABELS = {
    "dgp": "exact_latent_tree",
    "fit": "full_exact_tree",
    "leaf": "three_state_binomial_2",
    "nuisance_truth": "exact_tree_production_medians",
    "design": "current_gwt_rater_design",
}


def git_head() -> Dict[str, str]:
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
        ).strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return {"branch": branch, "commit": commit}
    except Exception:
        return {"branch": "unknown", "commit": "unknown"}


def iter_tree_nodes(
    stance_data: Dict[str, Any],
) -> Iterable[Tuple[Dict[str, Any], str, Tuple[str, ...], int, str]]:
    root_path = (stance_data["name"],)

    def walk(
        node: Dict[str, Any],
        ancestor_path: Tuple[str, ...],
        depth: int,
        top_feature: str,
    ):
        key = node_key(ancestor_path, node["name"])
        yield node, key, ancestor_path, depth, top_feature
        current_path = ancestor_path + (node["name"],)
        for child in node.get("evidencers", []):
            yield from walk(child, current_path, depth + 1, top_feature)

    for child in stance_data.get("evidencers", []):
        yield from walk(child, root_path, 1, child["name"])


def indicator_design_table(stance_data: Dict[str, Any], systems: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for node, key, _, depth, top_feature in iter_tree_nodes(stance_data):
        if (node.get("type") or "").lower() != "indicator":
            continue
        for system in systems:
            obs = node.get("observations", {}).get(system)
            if not obs:
                continue
            n = int(sum(not is_missing(v) for v in obs.get("values", [])))
            if n <= 0:
                continue
            rows.append(
                {
                    "system": system,
                    "node_key": key,
                    "indicator": node["name"],
                    "top_feature": top_feature,
                    "depth": depth,
                    "rating_count": n,
                }
            )
    return pd.DataFrame(rows)


def simulate_observations_in_place(
    rng: np.random.Generator,
    stance_data: Dict[str, Any],
    latent_by_system: Mapping[str, Mapping[str, Any]],
    obs_params: Mapping[str, Any],
    systems: Sequence[str],
    rater_multiplier: int = 1,
) -> Dict[str, Dict[str, List[int]]]:
    """Mutate stance_data observations to synthetic category representatives.

    If ``rater_multiplier`` K > 1, each existing (rater, indicator) rating slot
    is replicated K times — the rater identity is preserved, and K independent
    categories are drawn from the posterior-predictive given the indicator's
    latent state.  Missing slots are also replicated K times (still missing).
    """
    K = int(rater_multiplier)
    if K < 1:
        raise ValueError(f"rater_multiplier must be >= 1; got {K}")
    a = float(obs_params["a"])
    kappa = np.asarray(obs_params["kappa"], dtype=float)
    ordinal_by_system_indicator: Dict[str, Dict[str, List[int]]] = {
        system: {} for system in systems
    }

    def walk(node: Dict[str, Any], ancestor_path: Tuple[str, ...]) -> None:
        key = node_key(ancestor_path, node["name"])
        current_path = ancestor_path + (node["name"],)
        if (node.get("type") or "").lower() == "indicator":
            for system, obs in node.get("observations", {}).items():
                if system not in systems:
                    continue
                values = obs.get("values", [])
                names = obs.get("names", [])
                synthetic_values: List[Any] = []
                synthetic_names: List[Any] = []
                ordinal_values: List[int] = []
                m = int(latent_by_system[system]["indicator_m"][key])
                for i, val in enumerate(values):
                    rater_name = names[i] if i < len(names) else None
                    if is_missing(val):
                        for _ in range(K):
                            synthetic_values.append(-1)
                            synthetic_names.append(rater_name)
                        continue
                    for _ in range(K):
                        category = sample_rating_category(rng, kappa, a, m)
                        ordinal_values.append(category)
                        synthetic_values.append(float(CATEGORY_REPRESENTATIVES[category]))
                        synthetic_names.append(rater_name)
                obs["values"] = synthetic_values
                if names:
                    obs["names"] = synthetic_names
                ordinal_by_system_indicator[system][key] = ordinal_values
            return
        for child in node.get("evidencers", []):
            walk(child, current_path)

    for child in stance_data.get("evidencers", []):
        walk(child, (stance_data["name"],))
    return ordinal_by_system_indicator


def generate_synthetic_dataset(
    seed: int,
    cfg: ModelConfig,
    rater_multiplier: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    systems = [s for s, _ in ANCHORED_SYSTEM_CONFIGS]
    source_stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    synthetic_stance_data = copy.deepcopy(source_stance_data)
    truth = load_oracle_truth("exact_tree_production_medians", source_stance_data, cfg)

    rng = np.random.default_rng(seed)
    latent_by_system = {
        system: sample_latent_tree_for_system(
            rng,
            source_stance_data,
            truth.edge_betas,
            TRUE_C_BY_SYSTEM[system],
        )
        for system in systems
    }
    ordinal_by_system_indicator = simulate_observations_in_place(
        rng,
        synthetic_stance_data,
        latent_by_system,
        truth.obs_params,
        systems,
        rater_multiplier=rater_multiplier,
    )

    truth_payload = {
        "labels": LABELS,
        "seed": seed,
        "rater_multiplier": int(rater_multiplier),
        "true_C_by_system": TRUE_C_BY_SYSTEM,
        "latent_by_system": latent_by_system,
        "edge_betas": truth.edge_betas,
        "observation_parameters": truth.obs_params,
        "category_representatives": list(CATEGORY_REPRESENTATIVES),
        "ordinal_by_system_indicator": ordinal_by_system_indicator,
    }
    return synthetic_stance_data, source_stance_data, truth_payload


def build_fit_config(args: argparse.Namespace) -> ModelConfig:
    if args.smoke:
        draws = args.draws if args.draws is not None else 80
        tune = args.tune if args.tune is not None else 80
        chains = args.chains if args.chains is not None else 2
        target_accept = args.target_accept if args.target_accept is not None else 0.9
    else:
        draws = args.draws if args.draws is not None else 1000
        tune = args.tune if args.tune is not None else 1000
        chains = args.chains if args.chains is not None else 4
        target_accept = args.target_accept if args.target_accept is not None else 0.95

    return ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=True,
        BETA_ABS_BY_SUPPORT_DEMAND=True,
        LABEL_POOL_SIGMA=0.5,
        BETA_PRES_OVERRIDE_MEAN=(
            float(args.beta_pres_mean) if args.beta_pres_mean is not None else None
        ),
        BETA_ABS_OVERRIDE_MEAN=(
            float(args.beta_abs_mean) if args.beta_abs_mean is not None else None
        ),
        BETA_OVERRIDE_SIGMA=(
            float(args.beta_override_sigma) if args.beta_override_sigma is not None else None
        ),
        NUM_SAMPLES=draws,
        NUM_TUNE=tune,
        NUM_CHAINS=chains,
        TARGET_ACCEPT=target_accept,
    )


def posterior_draws(post: Any, var: str) -> np.ndarray:
    arr = np.asarray(post[var].values)
    return arr.reshape((-1,) + arr.shape[2:])


def summarize_draws(draws: np.ndarray, truth: float) -> Dict[str, float]:
    draws = np.asarray(draws, dtype=float).reshape(-1)
    mean = float(np.mean(draws))
    median = float(np.median(draws))
    lo = float(np.percentile(draws, 3))
    hi = float(np.percentile(draws, 97))
    signed_error = median - float(truth)
    return {
        "truth": float(truth),
        "posterior_mean": mean,
        "posterior_median": median,
        "posterior_p03": lo,
        "posterior_p97": hi,
        "signed_error": signed_error,
        "abs_error": abs(signed_error),
        "squared_error": signed_error**2,
        "interval_includes_truth": bool(lo <= float(truth) <= hi),
    }


def recovery_aggregate(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "n": int(len(g)),
                "mean_signed_error": float(g["signed_error"].mean()),
                "mae": float(g["abs_error"].mean()),
                "rmse": float(np.sqrt(g["squared_error"].mean())),
                "coverage_94": float(g["interval_includes_truth"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def root_recovery(
    idata: Any,
    builder: MultiSystemExactTreeBuilder,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    post = idata.posterior
    for system, _ in ANCHORED_SYSTEM_CONFIGS:
        var = f"{builder._sys_prefix(system)}__global_workspace_theory_C"
        draws = posterior_draws(post, var)
        row = {
            **LABELS,
            "estimand": "root_C",
            "system": system,
            "is_hard_anchor": system in {"Human", "ELIZA"},
            **summarize_draws(draws, TRUE_C_BY_SYSTEM[system]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def label_truth_from_edges(edge_betas: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    seen: Dict[str, Dict[str, Any]] = {}
    for edge in edge_betas.values():
        support = str(edge["support"])
        demand = str(edge["demandingness"])
        pres_var = _beta_pres_var_name(support, demand)
        abs_var = _beta_abs_var_name(demand, support)
        seen[pres_var] = {
            "parameter": pres_var,
            "kind": "beta_pres",
            "support": support,
            "demandingness": demand,
            "truth": float(edge["beta_pres"]),
        }
        seen[abs_var] = {
            "parameter": abs_var,
            "kind": "beta_abs",
            "support": support,
            "demandingness": demand,
            "truth": float(edge["beta_abs"]),
        }
    return pd.DataFrame(seen.values()).sort_values(["kind", "support", "demandingness"])


def label_recovery(idata: Any, edge_betas: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    post = idata.posterior
    rows = []
    for _, truth_row in label_truth_from_edges(edge_betas).iterrows():
        parameter = str(truth_row["parameter"])
        if parameter not in post.data_vars:
            continue
        rows.append(
            {
                **LABELS,
                "estimand": "label_beta",
                "parameter": parameter,
                "kind": truth_row["kind"],
                "support": truth_row["support"],
                "demandingness": truth_row["demandingness"],
                **summarize_draws(posterior_draws(post, parameter), float(truth_row["truth"])),
            }
        )
    return pd.DataFrame(rows)


def edge_recovery(
    idata: Any,
    stance_data: Dict[str, Any],
    edge_betas_truth: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    beta_pres_by_key, beta_abs_by_key = pooled_beta_draws_by_node(idata, stance_data)
    rows: List[Dict[str, Any]] = []
    for node, key, _, depth, top_feature in iter_tree_nodes(stance_data):
        if key not in edge_betas_truth:
            continue
        truth = edge_betas_truth[key]
        for kind, draws, true_val in [
            ("beta_pres", beta_pres_by_key[key], truth["beta_pres"]),
            ("beta_abs", beta_abs_by_key[key], truth["beta_abs"]),
        ]:
            rows.append(
                {
                    **LABELS,
                    "estimand": "edge_beta",
                    "node_key": key,
                    "node_name": node["name"],
                    "node_type": (node.get("type") or "").lower(),
                    "top_feature": top_feature,
                    "depth": depth,
                    "kind": kind,
                    "support": truth["support"],
                    "demandingness": truth["demandingness"],
                    **summarize_draws(draws, float(true_val)),
                }
            )
    return pd.DataFrame(rows)


def path_recovery(
    idata: Any,
    builder: MultiSystemExactTreeBuilder,
    stance_data: Dict[str, Any],
    edge_betas_truth: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    beta_pres_by_key, beta_abs_by_key = pooled_beta_draws_by_node(idata, stance_data)
    intercepts, slopes, indicators = propagate_affine_indicator_coefficients_from_draws(
        stance_data,
        builder.node_to_varname,
        beta_pres_by_key,
        beta_abs_by_key,
    )
    truth_bp = {k: np.asarray([v["beta_pres"]]) for k, v in edge_betas_truth.items()}
    truth_ba = {k: np.asarray([v["beta_abs"]]) for k, v in edge_betas_truth.items()}
    truth_intercepts, truth_slopes, truth_indicators = (
        propagate_affine_indicator_coefficients_from_draws(
            stance_data,
            builder.node_to_varname,
            truth_bp,
            truth_ba,
        )
    )
    truth_by_key = {
        spec.node_key: {
            "delta_j": float(truth_slopes[0, j]),
            "q_gap_999_001": float(truth_slopes[0, j] * (0.999 - 0.001)),
        }
        for j, spec in enumerate(truth_indicators)
    }

    meta_by_key = {}
    for node, key, _, depth, top_feature in iter_tree_nodes(stance_data):
        if (node.get("type") or "").lower() == "indicator":
            meta_by_key[key] = {
                "indicator": node["name"],
                "top_feature": top_feature,
                "depth": depth,
            }

    rows: List[Dict[str, Any]] = []
    for j, spec in enumerate(indicators):
        meta = meta_by_key[spec.node_key]
        for estimand, draws, truth in [
            ("delta_j", slopes[:, j], truth_by_key[spec.node_key]["delta_j"]),
            (
                "q_gap_999_001",
                slopes[:, j] * (0.999 - 0.001),
                truth_by_key[spec.node_key]["q_gap_999_001"],
            ),
        ]:
            rows.append(
                {
                    **LABELS,
                    "estimand": estimand,
                    "node_key": spec.node_key,
                    "indicator": meta["indicator"],
                    "top_feature": meta["top_feature"],
                    "depth": meta["depth"],
                    **summarize_draws(draws, truth),
                }
            )
    return pd.DataFrame(rows)


def leaf_log_likelihood_terms(
    ratings: Sequence[int],
    a: float,
    kappa: np.ndarray,
) -> np.ndarray:
    if not ratings:
        return np.zeros(3)
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


def internal_probs_one_draw(
    stance_data: Dict[str, Any],
    obs_by_key: Mapping[str, Sequence[Tuple[int, int]]],
    beta_pres: Mapping[str, float],
    beta_abs: Mapping[str, float],
    c_value: float,
    a: float,
    kappa: np.ndarray,
) -> Dict[str, float]:
    messages: Dict[str, Tuple[float, float]] = {}
    utilities: Dict[str, Tuple[float, float]] = {}
    root_path = (stance_data["name"],)

    def upward(node: Dict[str, Any], ancestor_path: Tuple[str, ...]) -> Tuple[float, float]:
        key = node_key(ancestor_path, node["name"])
        ntype = (node.get("type") or "").lower()
        if ntype == "indicator":
            ratings = [rating for _, rating in obs_by_key.get(key, [])]
            leaf_ll = leaf_log_likelihood_terms(ratings, a, kappa)
            msg = (
                three_state_log_mix(beta_abs[key], leaf_ll),
                three_state_log_mix(beta_pres[key], leaf_ll),
            )
            messages[key] = msg
            return msg

        current_path = ancestor_path + (node["name"],)
        log_u0 = 0.0
        log_u1 = 0.0
        for child in node.get("evidencers", []):
            child_msg0, child_msg1 = upward(child, current_path)
            log_u0 += child_msg0
            log_u1 += child_msg1
        utilities[key] = (log_u0, log_u1)
        bp = clipped_prob(beta_pres[key])
        ba = clipped_prob(beta_abs[key])
        msg0 = float(
            logsumexp(
                [
                    math.log(1.0 - ba) + log_u0,
                    math.log(ba) + log_u1,
                ]
            )
        )
        msg1 = float(
            logsumexp(
                [
                    math.log(1.0 - bp) + log_u0,
                    math.log(bp) + log_u1,
                ]
            )
        )
        messages[key] = (msg0, msg1)
        return msg0, msg1

    for child in stance_data.get("evidencers", []):
        upward(child, root_path)

    root_u0 = 0.0
    root_u1 = 0.0
    for child in stance_data.get("evidencers", []):
        key = node_key(root_path, child["name"])
        root_u0 += messages[key][0]
        root_u1 += messages[key][1]
    c = clipped_prob(c_value)
    log_evidence = float(
        logsumexp([math.log(1.0 - c) + root_u0, math.log(c) + root_u1])
    )

    post_probs: Dict[str, float] = {}

    def recurse_down(
        node: Dict[str, Any],
        ancestor_path: Tuple[str, ...],
        down: Tuple[float, float],
    ) -> None:
        key = node_key(ancestor_path, node["name"])
        ntype = (node.get("type") or "").lower()
        if ntype not in {"feature", "subfeature"}:
            return
        log_u0, log_u1 = utilities[key]
        post_probs[key] = float(np.exp(down[1] + log_u1 - log_evidence))

        current_path = ancestor_path + (node["name"],)
        children = node.get("evidencers", [])
        for child in children:
            child_key = node_key(current_path, child["name"])
            if (child.get("type") or "").lower() == "indicator":
                continue
            sibling0 = 0.0
            sibling1 = 0.0
            for sibling in children:
                sibling_key = node_key(current_path, sibling["name"])
                if sibling_key == child_key:
                    continue
                sibling0 += messages[sibling_key][0]
                sibling1 += messages[sibling_key][1]
            bp = clipped_prob(beta_pres[child_key])
            ba = clipped_prob(beta_abs[child_key])
            child_down0 = float(
                logsumexp(
                    [
                        down[0] + sibling0 + math.log(1.0 - ba),
                        down[1] + sibling1 + math.log(1.0 - bp),
                    ]
                )
            )
            child_down1 = float(
                logsumexp(
                    [
                        down[0] + sibling0 + math.log(ba),
                        down[1] + sibling1 + math.log(bp),
                    ]
                )
            )
            recurse_down(child, current_path, (child_down0, child_down1))

    for child in stance_data.get("evidencers", []):
        child_key = node_key(root_path, child["name"])
        sibling0 = 0.0
        sibling1 = 0.0
        for sibling in stance_data.get("evidencers", []):
            sibling_key = node_key(root_path, sibling["name"])
            if sibling_key == child_key:
                continue
            sibling0 += messages[sibling_key][0]
            sibling1 += messages[sibling_key][1]
        bp = clipped_prob(beta_pres[child_key])
        ba = clipped_prob(beta_abs[child_key])
        down0 = float(
            logsumexp(
                [
                    math.log(1.0 - c) + sibling0 + math.log(1.0 - ba),
                    math.log(c) + sibling1 + math.log(1.0 - bp),
                ]
            )
        )
        down1 = float(
            logsumexp(
                [
                    math.log(1.0 - c) + sibling0 + math.log(ba),
                    math.log(c) + sibling1 + math.log(bp),
                ]
            )
        )
        recurse_down(child, root_path, (down0, down1))

    return post_probs


def prior_internal_probabilities(
    stance_data: Dict[str, Any],
    edge_betas: Mapping[str, Mapping[str, Any]],
    c_value: float,
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
        walk(child, root_path, c_value)
    return out


def internal_state_recovery(
    idata: Any,
    builder: MultiSystemExactTreeBuilder,
    processor: MultiSystemDataProcessor,
    stance_data: Dict[str, Any],
    source_stance_data: Dict[str, Any],
    truth_payload: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    post = idata.posterior
    beta_pres_by_key, beta_abs_by_key = pooled_beta_draws_by_node(idata, stance_data)
    a_draws = posterior_draws(post, "a").reshape(-1)
    kappa_draws = posterior_draws(post, "kappa")
    n_draws = a_draws.shape[0]
    c_draws_by_system: Dict[str, np.ndarray] = {}
    for system, _ in ANCHORED_SYSTEM_CONFIGS:
        var = f"{builder._sys_prefix(system)}__global_workspace_theory_C"
        c_draws_by_system[system] = posterior_draws(post, var).reshape(-1)

    systems = [s for s, _ in ANCHORED_SYSTEM_CONFIGS]
    meta = collect_internal_node_meta(source_stance_data, systems)
    edge_truth = truth_payload["edge_betas"]
    latent_by_system = truth_payload["latent_by_system"]
    rows: List[Dict[str, Any]] = []

    for system in systems:
        obs_by_key = processor.system_observations[system]
        posterior_sums = {key: 0.0 for key in meta}
        for i in range(n_draws):
            bp = {k: float(v[i]) for k, v in beta_pres_by_key.items()}
            ba = {k: float(v[i]) for k, v in beta_abs_by_key.items()}
            probs = internal_probs_one_draw(
                stance_data,
                obs_by_key,
                bp,
                ba,
                float(c_draws_by_system[system][i]),
                float(a_draws[i]),
                np.asarray(kappa_draws[i], dtype=float),
            )
            for key, value in probs.items():
                posterior_sums[key] += float(value)

        prior_probs = prior_internal_probabilities(
            source_stance_data,
            edge_truth,
            TRUE_C_BY_SYSTEM[system],
        )
        for key, node_meta in meta.items():
            p_post = clipped_prob(posterior_sums[key] / n_draws)
            p_prior = clipped_prob(prior_probs[key])
            z_true = int(latent_by_system[system]["internal_z"][key])
            prior_h = binary_entropy(p_prior)
            post_h = binary_entropy(p_post)
            log_score = math.log(p_post if z_true else 1.0 - p_post)
            rows.append(
                {
                    **LABELS,
                    "estimand": "internal_z_probability",
                    "system": system,
                    "node_key": key,
                    "node_name": node_meta.node_name,
                    "node_type": node_meta.node_type,
                    "top_feature": node_meta.top_feature,
                    "depth": node_meta.depth,
                    "fanout": node_meta.fanout,
                    "subtree_indicator_count": node_meta.subtree_indicator_count,
                    "subtree_rating_count": node_meta.subtree_rating_count_by_system[system],
                    "z_true": z_true,
                    "prior_p_z1": p_prior,
                    "posterior_p_z1": p_post,
                    "prior_entropy": prior_h,
                    "posterior_entropy": post_h,
                    "entropy_reduction": prior_h - post_h,
                    "relative_entropy_reduction": (
                        (prior_h - post_h) / prior_h if prior_h > 1e-12 else math.nan
                    ),
                    "brier": float((p_post - z_true) ** 2),
                    "log_score": float(log_score),
                    "neg_log_score": float(-log_score),
                }
            )

    df = pd.DataFrame(rows)
    cal = df.copy()
    cal["prob_bin"] = pd.cut(
        cal["posterior_p_z1"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
        right=False,
    ).astype(str)
    calibration = (
        cal.groupby("prob_bin", dropna=False)
        .agg(
            n=("posterior_p_z1", "size"),
            mean_posterior_p=("posterior_p_z1", "mean"),
            empirical_z_rate=("z_true", "mean"),
            mean_brier=("brier", "mean"),
            mean_neg_log_score=("neg_log_score", "mean"),
        )
        .reset_index()
    )
    return df, calibration


def parent_probs_by_indicator_draw(
    stance_data: Dict[str, Any],
    beta_pres: Mapping[str, float],
    beta_abs: Mapping[str, float],
    c_value: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    root_path = (stance_data["name"],)

    def walk(node: Dict[str, Any], ancestor_path: Tuple[str, ...], parent_q: float) -> None:
        key = node_key(ancestor_path, node["name"])
        if (node.get("type") or "").lower() == "indicator":
            out[key] = float(parent_q)
            return
        q_self = beta_abs[key] + parent_q * (beta_pres[key] - beta_abs[key])
        current_path = ancestor_path + (node["name"],)
        for child in node.get("evidencers", []):
            walk(child, current_path, q_self)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path, c_value)
    return out


def indicator_category_probs(
    parent_q: float,
    beta_pres: float,
    beta_abs: float,
    a: float,
    kappa: np.ndarray,
) -> np.ndarray:
    p_m_parent0 = np.asarray(
        [(1 - beta_abs) ** 2, 2 * beta_abs * (1 - beta_abs), beta_abs**2]
    )
    p_m_parent1 = np.asarray(
        [(1 - beta_pres) ** 2, 2 * beta_pres * (1 - beta_pres), beta_pres**2]
    )
    p_m = (1.0 - parent_q) * p_m_parent0 + parent_q * p_m_parent1
    out = np.zeros(len(kappa) + 1)
    for m in (0, 1, 2):
        out += p_m[m] * ordered_probit_probs(kappa, a * (m / 2.0))
    out = np.clip(out, 1e-12, 1.0)
    return out / out.sum()


def observed_counts_by_group(
    processor: MultiSystemDataProcessor,
    design: pd.DataFrame,
) -> Dict[Tuple[str, str], np.ndarray]:
    group_counts: Dict[Tuple[str, str], np.ndarray] = {}
    top_by_key = dict(zip(design["node_key"], design["top_feature"]))
    for system, obs_by_key in processor.system_observations.items():
        for key, obs in obs_by_key.items():
            top = top_by_key[key]
            group_key = (system, top)
            group_counts.setdefault(group_key, np.zeros(7))
            for _, rating in obs:
                group_counts[group_key][int(rating)] += 1
    return group_counts


def observed_scale_check(
    idata: Any,
    builder: MultiSystemExactTreeBuilder,
    processor: MultiSystemDataProcessor,
    stance_data: Dict[str, Any],
    design: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    post = idata.posterior
    beta_pres_by_key, beta_abs_by_key = pooled_beta_draws_by_node(idata, stance_data)
    a_draws = posterior_draws(post, "a").reshape(-1)
    kappa_draws = posterior_draws(post, "kappa")
    n_draws = a_draws.shape[0]
    c_draws_by_system = {
        system: posterior_draws(
            post,
            f"{builder._sys_prefix(system)}__global_workspace_theory_C",
        ).reshape(-1)
        for system, _ in ANCHORED_SYSTEM_CONFIGS
    }
    group_keys = sorted(set(zip(design["system"], design["top_feature"])))
    predictive = {group: np.zeros((n_draws, 7)) for group in group_keys}

    for i in range(n_draws):
        bp = {k: float(v[i]) for k, v in beta_pres_by_key.items()}
        ba = {k: float(v[i]) for k, v in beta_abs_by_key.items()}
        for system, sys_design in design.groupby("system"):
            parent_probs = parent_probs_by_indicator_draw(
                stance_data,
                bp,
                ba,
                float(c_draws_by_system[system][i]),
            )
            for _, row in sys_design.iterrows():
                probs = indicator_category_probs(
                    parent_probs[row["node_key"]],
                    bp[row["node_key"]],
                    ba[row["node_key"]],
                    float(a_draws[i]),
                    np.asarray(kappa_draws[i], dtype=float),
                )
                predictive[(system, row["top_feature"])][i, :] += (
                    rng.multinomial(int(row["rating_count"]), probs)
                )

    obs_counts = observed_counts_by_group(processor, design)
    rows: List[Dict[str, Any]] = []
    for group, draws in predictive.items():
        system, top_feature = group
        total_n = float(draws.sum(axis=1)[0])
        obs = obs_counts[group]
        for category in range(7):
            pred_counts = draws[:, category]
            pred_props = pred_counts / total_n
            obs_count = float(obs[category])
            obs_prop = obs_count / total_n
            rows.append(
                {
                    **LABELS,
                    "estimand": "observed_scale_feature_block",
                    "system": system,
                    "top_feature": top_feature,
                    "category": category,
                    "rating_count": int(total_n),
                    "observed_count": obs_count,
                    "observed_proportion": obs_prop,
                    "pred_count_mean": float(np.mean(pred_counts)),
                    "pred_count_p03": float(np.percentile(pred_counts, 3)),
                    "pred_count_p97": float(np.percentile(pred_counts, 97)),
                    "pred_prop_mean": float(np.mean(pred_props)),
                    "pred_prop_p03": float(np.percentile(pred_props, 3)),
                    "pred_prop_p97": float(np.percentile(pred_props, 97)),
                    "prop_error": float(np.mean(pred_props) - obs_prop),
                    "interval_includes_observed_prop": bool(
                        np.percentile(pred_props, 3)
                        <= obs_prop
                        <= np.percentile(pred_props, 97)
                    ),
                }
            )
    return pd.DataFrame(rows)


def diagnostics_table(idata: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    div_count = int(idata.sample_stats["diverging"].values.sum())
    var_names = ["a", "kappa"]
    for v in idata.posterior.data_vars:
        name = str(v)
        if name.endswith("_C"):
            d = np.asarray(idata.posterior[v].values).reshape(-1)
            if np.std(d) > 1e-10:
                var_names.append(name)
        if (
            name.startswith("beta_pres__")
            or name.startswith("beta_abs__")
            or name.startswith("label_delta__")
        ):
            var_names.append(name)
    diag = az.summary(idata, var_names=var_names, kind="diagnostics").reset_index()
    max_rhat = float(diag["r_hat"].max(skipna=True))
    min_ess_bulk = float(diag["ess_bulk"].min(skipna=True))
    status = {
        "divergences": div_count,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "diagnostic_status": (
            "passed" if div_count == 0 and max_rhat <= 1.01 else "exploratory_failed"
        ),
    }
    return diag, status


def format_metric(x: Any, digits: int = 3) -> str:
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
                vals.append(format_metric(val))
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return rows


def write_summary(
    out_dir: Path,
    smoke: bool,
    elapsed_build: float,
    elapsed_sample: float,
    diagnostic_status: Mapping[str, Any],
    root_df: pd.DataFrame,
    label_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    path_df: pd.DataFrame,
    internal_df: pd.DataFrame,
    observed_df: pd.DataFrame,
) -> None:
    free_root = root_df[~root_df["is_hard_anchor"]][
        [
            "system",
            "truth",
            "posterior_median",
            "posterior_p03",
            "posterior_p97",
            "signed_error",
            "interval_includes_truth",
        ]
    ]
    label_agg = recovery_aggregate(label_df, ["kind"])
    edge_agg = recovery_aggregate(edge_df, ["kind"])
    path_agg = recovery_aggregate(path_df, ["estimand"])
    internal_overall = pd.DataFrame(
        [
            {
                "n": len(internal_df),
                "mean_relative_entropy_reduction": internal_df[
                    "relative_entropy_reduction"
                ].mean(),
                "mean_brier": internal_df["brier"].mean(),
                "mean_neg_log_score": internal_df["neg_log_score"].mean(),
            }
        ]
    )
    observed_overall = pd.DataFrame(
        [
            {
                "n": len(observed_df),
                "mean_abs_prop_error": observed_df["prop_error"].abs().mean(),
                "coverage_94": observed_df["interval_includes_observed_prop"].mean(),
            }
        ]
    )
    lines = [
        "# Full-GWT Exact/Exact Fake-Data Recovery Pilot",
        "",
        "Labels:",
        "",
        f"- DGP: `{LABELS['dgp']}`",
        f"- Fit: `{LABELS['fit']}`",
        f"- Leaf: `{LABELS['leaf']}`",
        f"- Nuisance truth: `{LABELS['nuisance_truth']}`",
        f"- Design: `{LABELS['design']}`",
        "",
        f"Mode: {'smoke' if smoke else 'full'}",
        f"Diagnostic status: `{diagnostic_status['diagnostic_status']}`",
        f"Elapsed build seconds: {elapsed_build:.1f}",
        f"Elapsed sample seconds: {elapsed_sample:.1f}",
        f"Divergences: {diagnostic_status['divergences']}",
        f"Max R-hat: {diagnostic_status['max_rhat']:.4f}",
        f"Min bulk ESS: {diagnostic_status['min_ess_bulk']:.0f}",
        "",
        "## Plain-English Interpretation",
        "",
        "This is the first full fake-data recovery check: data were generated "
        "from the exact GWT tree and then fit with the same exact-tree model. "
        "Unlike the oracle audits, nuisance parameters are learned rather than "
        "held fixed.",
        "",
        "Because this is one synthetic seed, signed errors are recovery errors "
        "for this pilot, not Monte Carlo bias estimates. Multi-seed runs are "
        "needed before treating RMSE or coverage as stable calibration claims.",
        "",
        "If diagnostics fail, read the numerical recovery tables as exploratory "
        "engineering output rather than as a model validation result.",
        "",
        "Observed-scale PPC intervals include posterior uncertainty and synthetic "
        "ordinal rating noise. They are intentionally wider than intervals over "
        "posterior expected proportions alone.",
        "",
        "## Free Root C Recovery",
        "",
        *markdown_table(
            free_root,
            [
                "system",
                "truth",
                "posterior_median",
                "posterior_p03",
                "posterior_p97",
                "signed_error",
                "interval_includes_truth",
            ],
        ),
        "",
        "## Label Beta Recovery",
        "",
        *markdown_table(label_agg, ["kind", "n", "mean_signed_error", "mae", "rmse", "coverage_94"]),
        "",
        "## Edge Beta Recovery",
        "",
        *markdown_table(edge_agg, ["kind", "n", "mean_signed_error", "mae", "rmse", "coverage_94"]),
        "",
        "## Path Summary Recovery",
        "",
        *markdown_table(path_agg, ["estimand", "n", "mean_signed_error", "mae", "rmse", "coverage_94"]),
        "",
        "## Internal State Probability Scores",
        "",
        *markdown_table(
            internal_overall,
            ["n", "mean_relative_entropy_reduction", "mean_brier", "mean_neg_log_score"],
        ),
        "",
        "## Observed-Scale Feature-Block PPC",
        "",
        *markdown_table(observed_overall, ["n", "mean_abs_prop_error", "coverage_94"]),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--tune", type=int, default=None)
    parser.add_argument("--chains", type=int, default=None)
    parser.add_argument("--target-accept", type=float, default=None)
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Reuse an existing run directory with fit.nc and regenerate recovery outputs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rater-multiplier",
        type=int,
        default=1,
        help=(
            "Replicate every (rater, indicator) rating slot K times in the "
            "synthetic dataset; K independent draws from the same posterior-"
            "predictive distribution given the latent state.  Default 1."
        ),
    )
    parser.add_argument("--beta-pres-mean", type=float, default=None)
    parser.add_argument("--beta-abs-mean", type=float, default=None)
    parser.add_argument("--beta-override-sigma", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_fit_config(args)
    labels = dict(LABELS)
    K = int(args.rater_multiplier)
    run_id = args.run_id or (
        f"{labels['dgp']}__{labels['fit']}__{labels['leaf']}__"
        f"{labels['nuisance_truth']}__{labels['design']}__seed{args.seed}"
        + (f"__multK{K}" if K != 1 else "")
        + ("__smoke" if args.smoke else "")
    )
    runs_dir = args.runs_dir if args.runs_dir.is_absolute() else REPO_ROOT / args.runs_dir
    out_dir = runs_dir / run_id
    if out_dir.exists() and not args.overwrite and not args.postprocess_only:
        raise FileExistsError(f"Run directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    git = git_head()
    systems = [s for s, _ in ANCHORED_SYSTEM_CONFIGS]
    print("=== Full-GWT exact/exact fake-data recovery ===")
    print(f"mode: {'smoke' if args.smoke else 'full'}")
    print(f"run_id: {run_id}")
    print(
        "sampling: "
        f"chains={cfg.NUM_CHAINS}, tune={cfg.NUM_TUNE}, draws={cfg.NUM_SAMPLES}, "
        f"target_accept={cfg.TARGET_ACCEPT}"
    )

    if args.postprocess_only:
        if not (out_dir / "fit.nc").exists():
            raise FileNotFoundError(f"Missing existing fit.nc in {out_dir}")
        with open(out_dir / "synthetic_stance_data.json") as f:
            synthetic_stance_data = json.load(f)
        source_stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
        with open(out_dir / "truth.json") as f:
            truth_payload = json.load(f)
        existing_config = {}
        if (out_dir / "config.json").exists():
            with open(out_dir / "config.json") as f:
                existing_config = json.load(f)
    else:
        synthetic_stance_data, source_stance_data, truth_payload = generate_synthetic_dataset(
            args.seed,
            cfg,
            rater_multiplier=K,
        )
        write_json(out_dir / "truth.json", truth_payload)
        write_json(out_dir / "synthetic_stance_data.json", synthetic_stance_data)
        existing_config = {}
    design = indicator_design_table(source_stance_data, systems)

    proc = MultiSystemDataProcessor(cfg)
    proc.process(synthetic_stance_data, systems)

    print("Building exact-tree model on synthetic data...")
    t0 = time.time()
    builder = MultiSystemExactTreeBuilder(cfg, EvidenceProcessor(cfg), proc, list(ANCHORED_SYSTEM_CONFIGS))
    model = builder.build_model(synthetic_stance_data)
    elapsed_build_current = time.time() - t0
    elapsed_build = float(existing_config.get("elapsed_s_build", elapsed_build_current))
    print(f"build seconds: {elapsed_build:.1f}")

    nc_path = out_dir / "fit.nc"
    if args.postprocess_only:
        print("Loading existing fit.nc for postprocess-only run...")
        idata = az.from_netcdf(str(nc_path))
        elapsed_sample = float(existing_config.get("elapsed_s_sample", 0.0))
    else:
        print("Sampling synthetic exact-tree fit...")
        t1 = time.time()
        with model:
            import pymc as pm

            idata = pm.sample(
                draws=cfg.NUM_SAMPLES,
                tune=cfg.NUM_TUNE,
                chains=cfg.NUM_CHAINS,
                cores=cfg.NUM_CHAINS,
                target_accept=cfg.TARGET_ACCEPT,
                random_seed=args.seed,
            )
        elapsed_sample = time.time() - t1
        print(f"sample seconds: {elapsed_sample:.1f}")

        if nc_path.exists():
            nc_path.unlink()
        az.to_netcdf(idata, str(nc_path))

    print("Postprocessing recovery diagnostics...")
    diag_df, diag_status = diagnostics_table(idata)
    root_df = root_recovery(idata, builder)
    label_df = label_recovery(idata, truth_payload["edge_betas"])
    edge_df = edge_recovery(idata, synthetic_stance_data, truth_payload["edge_betas"])
    path_df = path_recovery(idata, builder, synthetic_stance_data, truth_payload["edge_betas"])
    internal_df, internal_calibration = internal_state_recovery(
        idata,
        builder,
        proc,
        synthetic_stance_data,
        source_stance_data,
        truth_payload,
    )
    observed_df = observed_scale_check(
        idata,
        builder,
        proc,
        synthetic_stance_data,
        design,
        np.random.default_rng(args.seed + 1_000_003),
    )

    diag_df.to_csv(out_dir / "diagnostics.csv", index=False)
    root_df.to_csv(out_dir / "root_recovery.csv", index=False)
    label_df.to_csv(out_dir / "label_beta_recovery.csv", index=False)
    edge_df.to_csv(out_dir / "edge_beta_recovery.csv", index=False)
    path_df.to_csv(out_dir / "path_summary_recovery.csv", index=False)
    internal_df.to_csv(out_dir / "internal_state_probability_recovery.csv", index=False)
    internal_calibration.to_csv(out_dir / "internal_state_calibration.csv", index=False)
    observed_df.to_csv(out_dir / "observed_scale_feature_block_ppc.csv", index=False)

    config_payload = {
        "labels": labels,
        "audit": "full_exact_fake_data_recovery",
        "mode": "smoke" if args.smoke else "full",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "rater_multiplier": K,
        "git": git,
        "stance": STANCE,
        "systems": systems,
        "true_C_by_system": TRUE_C_BY_SYSTEM,
        "production_fit_source": str(EXACT_PROD_PATH.relative_to(REPO_ROOT)),
        "config": asdict(cfg),
        "elapsed_s_build": elapsed_build,
        "elapsed_s_sample": elapsed_sample,
        **diag_status,
        "outputs": {
            "truth": "truth.json",
            "synthetic_stance_data": "synthetic_stance_data.json",
            "fit": "fit.nc",
            "diagnostics": "diagnostics.csv",
            "root_recovery": "root_recovery.csv",
            "label_beta_recovery": "label_beta_recovery.csv",
            "edge_beta_recovery": "edge_beta_recovery.csv",
            "path_summary_recovery": "path_summary_recovery.csv",
            "internal_state_probability_recovery": (
                "internal_state_probability_recovery.csv"
            ),
            "internal_state_calibration": "internal_state_calibration.csv",
            "observed_scale_feature_block_ppc": "observed_scale_feature_block_ppc.csv",
            "summary": "summary.md",
        },
    }
    write_json(out_dir / "config.json", config_payload)
    write_summary(
        out_dir,
        args.smoke,
        elapsed_build,
        elapsed_sample,
        diag_status,
        root_df,
        label_df,
        edge_df,
        path_df,
        internal_df,
        observed_df,
    )

    print("=== Recovery pilot complete ===")
    print(f"diagnostic status: {diag_status['diagnostic_status']}")
    print(
        f"divergences={diag_status['divergences']}, "
        f"max_rhat={diag_status['max_rhat']:.4f}, "
        f"min_ess_bulk={diag_status['min_ess_bulk']:.0f}"
    )
    print("free root recovery:")
    print(
        root_df[~root_df["is_hard_anchor"]][
            ["system", "truth", "posterior_median", "posterior_p03", "posterior_p97", "signed_error"]
        ].to_string(index=False)
    )
    print(f"\nwrote {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
