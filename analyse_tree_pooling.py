"""Post-fit analysis for the POOL_BETAS_BY_LABEL branch.

Under complete-pooling-within-label the tree has no per-node
``{varname}_beta_pres`` / ``_beta_abs`` RVs; instead, each tree node's beta
is a shared group-level variable ``beta_pres__{support}__{demandingness}`` /
``beta_abs__{demandingness}``.  Standard helpers like
``gwt_reference_recovery_analysis.extract_beta_draws_by_node`` expect
per-node variables, so this module provides pooling-aware analogues.

Functions:
- ``pooled_beta_draws_by_node``: walks the stance tree and for each node
  fetches the relevant group-level beta draws from the idata, returning
  the same (beta_pres_by_key, beta_abs_by_key) shape that
  ``propagate_affine_indicator_coefficients_from_draws`` consumes.
- ``per_indicator_delta_under_pooling``: convenience wrapper that calls
  the affine-propagation helper with pooling-aware beta draws.
- ``label_level_summary``: per-group posterior summary of beta_pres,
  beta_abs, and label_delta.

All functions operate on any idata with ``POOL_BETAS_BY_LABEL=True``
outputs.  No MCMC.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import arviz as az
import numpy as np
import pandas as pd

from dcm_model import _sanitize_label, node_key


def _beta_pres_var_name(support: str, demand: str) -> str:
    return f"beta_pres__{_sanitize_label(support)}__{_sanitize_label(demand)}"


def _beta_abs_var_name(demand: str) -> str:
    return f"beta_abs__{_sanitize_label(demand)}"


def _label_delta_var_name(support: str, demand: str) -> str:
    return f"label_delta__{_sanitize_label(support)}__{_sanitize_label(demand)}"


def pooled_beta_draws_by_node(
    idata: Any,
    stance_data: Dict[str, Any],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Per-node beta draws under complete-pooling-within-label.

    Walks the stance tree and for each internal/leaf node returns the
    group-level beta draws corresponding to its (support, demandingness)
    labels.  Returned dicts have the same shape as
    ``gwt_reference_recovery_analysis.extract_beta_draws_by_node`` but with
    nodes sharing the same label also sharing the same underlying draw
    array (shape: (n_draws,)).
    """
    post = idata.posterior
    beta_pres_by_key: Dict[str, np.ndarray] = {}
    beta_abs_by_key: Dict[str, np.ndarray] = {}

    # Cache group-level fetches to avoid repeat reshapes.
    pres_cache: Dict[Tuple[str, str], np.ndarray] = {}
    abs_cache: Dict[str, np.ndarray] = {}

    def fetch_pres(support: str, demand: str) -> np.ndarray:
        key = (support, demand)
        if key not in pres_cache:
            var = _beta_pres_var_name(support, demand)
            pres_cache[key] = np.asarray(post[var].values).reshape(-1)
        return pres_cache[key]

    def fetch_abs(demand: str) -> np.ndarray:
        if demand not in abs_cache:
            var = _beta_abs_var_name(demand)
            abs_cache[demand] = np.asarray(post[var].values).reshape(-1)
        return abs_cache[demand]

    root_path = (stance_data["name"],)

    def walk(node: Dict[str, Any], ancestor_path: Tuple[str, ...]) -> None:
        current_path = ancestor_path + (node["name"],)
        ntype = (node.get("type") or "").lower()
        if ntype in {"feature", "subfeature", "indicator"}:
            s = node.get("support", "no bearing")
            d = node.get("demandingness", "neutral")
            key = node_key(ancestor_path, node["name"])
            beta_pres_by_key[key] = fetch_pres(s, d)
            beta_abs_by_key[key] = fetch_abs(d)
        for child in node.get("evidencers", []):
            walk(child, current_path)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path)
    return beta_pres_by_key, beta_abs_by_key


def per_indicator_delta_under_pooling(
    idata: Any,
    builder: Any,
    stance_data: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """Compute affine propagation coefficients (alpha_j, delta_j) per
    indicator under the pooled tree."""
    from gwt_reference_recovery_analysis import (
        build_indicator_index,
        propagate_affine_indicator_coefficients_from_draws,
    )

    beta_pres_by_key, beta_abs_by_key = pooled_beta_draws_by_node(idata, stance_data)
    indicators = build_indicator_index(stance_data, builder)
    return propagate_affine_indicator_coefficients_from_draws(
        stance_data,
        builder.node_to_varname,
        beta_pres_by_key,
        beta_abs_by_key,
        indicator_index=indicators,
    )


def label_level_summary(idata: Any) -> pd.DataFrame:
    """Per-group posterior summary of beta_pres, beta_abs, label_delta."""
    post = idata.posterior
    rows: List[Dict[str, Any]] = []
    for v in post.data_vars:
        name = str(v)
        if name.startswith("beta_pres__") or name.startswith("beta_abs__") \
                or name.startswith("label_delta__"):
            draws = np.asarray(post[v].values).reshape(-1)
            rows.append(
                {
                    "parameter": name,
                    "kind": name.split("__", 1)[0],
                    "median": float(np.median(draws)),
                    "lo": float(np.percentile(draws, 3)),
                    "hi": float(np.percentile(draws, 97)),
                    "std": float(np.std(draws)),
                }
            )
    return pd.DataFrame(rows).sort_values(["kind", "parameter"]).reset_index(drop=True)


def sign_flip_resolution_summary(
    idata: Any,
    builder: Any,
    stance_data: Dict[str, Any],
    diagnostic_table_csv: str = "results/gwt_tree_propagation/analysis/"
    "per_indicator_propagation.csv",
) -> pd.DataFrame:
    """For the 5 sign-flip indicators flagged in Stage 1c, report posterior
    delta_j under pooling vs the baseline (from disk) and under the paper
    prior (from disk).
    """
    base = pd.read_csv(diagnostic_table_csv)
    log_r_sign = np.sign(base["delta_post_med"]) != np.sign(base["delta_prior"])
    flip_rows = base[log_r_sign & (np.abs(base["delta_prior"]) >= 1e-3)].copy()
    flip_keys = set(flip_rows["node_key"])

    intercepts, slopes, indicator_order = per_indicator_delta_under_pooling(
        idata, builder, stance_data
    )
    out: List[Dict[str, Any]] = []
    for j, spec in enumerate(indicator_order):
        if spec.node_key not in flip_keys:
            continue
        delta_draws = slopes[:, j]
        row = {
            "indicator": spec.display_name,
            "node_key": spec.node_key,
            "delta_prior_paper": float(
                flip_rows.loc[flip_rows["node_key"] == spec.node_key, "delta_prior"].iloc[0]
            ),
            "delta_post_baseline": float(
                flip_rows.loc[flip_rows["node_key"] == spec.node_key, "delta_post_med"].iloc[0]
            ),
            "delta_post_pooling_med": float(np.median(delta_draws)),
            "delta_post_pooling_lo": float(np.percentile(delta_draws, 3)),
            "delta_post_pooling_hi": float(np.percentile(delta_draws, 97)),
        }
        out.append(row)
    return pd.DataFrame(out)
