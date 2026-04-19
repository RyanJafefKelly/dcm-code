"""Smoke tests for the GWT reference-recovery workflow.

These checks are intentionally structural:
  - numerical recovery-curve helpers normalize correctly
  - affine q_j(c) propagation matches direct tree evaluation
  - anchored / one-anchor configs build and sample on a tiny synthetic tree
  - the anchored build still exposes tree-implied ``..._p`` and leaf-updated
    ``..._pz1`` variables for observed indicators
"""

from __future__ import annotations

import copy

import numpy as np

from gwt_reference_recovery_analysis import (
    ANCHORED_SYSTEM_CONFIGS,
    DEFAULT_GRID_EPS,
    ELIZA_FREE_SYSTEM_CONFIGS,
    HUMAN_FREE_SYSTEM_CONFIGS,
    IndicatorSpec,
    build_indicator_index,
    compare_system_medians,
    evaluate_recovery_curve_from_terms,
    extract_beta_draws_by_node,
    extract_free_c_draws,
    fit_reference_model,
    normalize_log_weights,
    propagate_affine_indicator_coefficients,
    summarise_grid_density,
)
from dcm_model import node_key
from test_ordinal_smoke import build_synthetic_tree


def _relabel_system(tree: dict, old_system: str, new_system: str) -> dict:
    out = copy.deepcopy(tree)

    def walk(node: dict) -> None:
        if node.get("type", "").lower() == "indicator":
            obs = node.get("observations", {})
            if old_system in obs:
                obs[new_system] = obs.pop(old_system)
            stats = node.get("observation_stats", {})
            if old_system in stats:
                stats[new_system] = stats.pop(old_system)
        for child in node.get("evidencers", []):
            walk(child)

    walk(out)
    return out


def _merge_system_tree(base: dict, other: dict, system_name: str) -> dict:
    merged = copy.deepcopy(base)

    def walk(node_a: dict, node_b: dict) -> None:
        if node_a.get("type", "").lower() == "indicator":
            node_a.setdefault("observations", {})[system_name] = node_b["observations"][
                system_name
            ]
            node_a.setdefault("observation_stats", {})[system_name] = node_b[
                "observation_stats"
            ][system_name]
        for child_a, child_b in zip(
            node_a.get("evidencers", []),
            node_b.get("evidencers", []),
        ):
            walk(child_a, child_b)

    walk(merged, other)
    return merged


def build_four_system_tree() -> dict:
    system_names = ["Human", "Chicken", "2024 Leading Chat LLMs", "ELIZA"]
    seeds = [1, 7, 13, 21]
    base_tree = None
    old_system = None
    for system_name, seed in zip(system_names, seeds):
        tree, _, generated_system = build_synthetic_tree(
            n_features=2,
            indicators_per_feature=2,
            n_experts=3,
            true_a=1.7,
            seed=seed,
        )
        relabelled = _relabel_system(tree, generated_system, system_name)
        if base_tree is None:
            base_tree = relabelled
            old_system = system_name
        else:
            base_tree = _merge_system_tree(base_tree, relabelled, system_name)
    assert base_tree is not None
    return base_tree


def _direct_indicator_probs(
    stance_data: dict,
    indicator_index: list[IndicatorSpec],
    beta_pres_by_key: dict[str, np.ndarray],
    beta_abs_by_key: dict[str, np.ndarray],
    c_value: float,
) -> np.ndarray:
    out: dict[str, np.ndarray] = {}
    root_path = (stance_data["name"],)

    def walk(node: dict, ancestor_path: tuple[str, ...], parent_q: np.ndarray) -> None:
        current_path = ancestor_path + (node["name"],)
        key = node_key(ancestor_path, node["name"])
        q = beta_abs_by_key[key] + parent_q * (beta_pres_by_key[key] - beta_abs_by_key[key])
        if node.get("type", "").lower() == "indicator":
            out[key] = q
            return
        for child in node.get("evidencers", []):
            walk(child, current_path, q)

    root_q = np.full(next(iter(beta_pres_by_key.values())).shape[0], c_value)
    for child in stance_data.get("evidencers", []):
        walk(child, root_path, root_q)
    return np.column_stack([out[spec.node_key] for spec in indicator_index])


def numerical_helper_checks() -> None:
    log_weights = np.log(np.array([1.0, 2.0, 3.0, 2.0, 1.0]))
    weights = normalize_log_weights(log_weights)
    assert np.isclose(weights.sum(), 1.0)

    grid = np.linspace(DEFAULT_GRID_EPS, 1.0 - DEFAULT_GRID_EPS, 5)
    summary = summarise_grid_density(grid, weights)
    assert np.isfinite(summary["median"])
    assert summary["lo"] <= summary["median"] <= summary["hi"]
    assert 0.0 <= summary["p_lt_005"] <= 1.0
    assert 0.0 <= summary["p_gt_095"] <= 1.0

    ll_z0 = np.log(
        np.array(
            [
                [0.91, 0.82],
                [0.88, 0.79],
                [0.90, 0.81],
            ]
        )
    )
    ll_z1 = np.log(
        np.array(
            [
                [0.31, 0.42],
                [0.28, 0.45],
                [0.33, 0.39],
            ]
        )
    )
    intercepts = np.array(
        [
            [0.08, 0.17],
            [0.11, 0.20],
            [0.09, 0.18],
        ]
    )
    slopes = np.array(
        [
            [0.74, 0.61],
            [0.70, 0.58],
            [0.72, 0.60],
        ]
    )
    curve = evaluate_recovery_curve_from_terms(
        np.linspace(DEFAULT_GRID_EPS, 1.0 - DEFAULT_GRID_EPS, 51),
        ll_z0,
        ll_z1,
        intercepts,
        slopes,
        prior_alpha=1.0,
        prior_beta=5.0,
        chunk_size=2,
    )
    assert np.isclose(curve["likelihood_density"].sum(), 1.0)
    assert np.isclose(curve["posterior_density"].sum(), 1.0)
    print("numerical helper checks passed")


def end_to_end_smoke() -> None:
    stance_data = build_four_system_tree()
    fit_overrides = {
        "NUM_SAMPLES": 80,
        "NUM_TUNE": 120,
        "NUM_CHAINS": 2,
        "TARGET_ACCEPT": 0.9,
    }

    anchored = fit_reference_model(
        stance_data,
        ANCHORED_SYSTEM_CONFIGS,
        fit_overrides=fit_overrides,
        verbose=False,
    )
    human_free = fit_reference_model(
        stance_data,
        HUMAN_FREE_SYSTEM_CONFIGS,
        fit_overrides=fit_overrides,
        verbose=False,
    )
    eliza_free = fit_reference_model(
        stance_data,
        ELIZA_FREE_SYSTEM_CONFIGS,
        fit_overrides=fit_overrides,
        verbose=False,
    )

    assert extract_free_c_draws(anchored["idata"], anchored["builder"], "Human") is None
    assert extract_free_c_draws(anchored["idata"], anchored["builder"], "ELIZA") is None
    assert (
        extract_free_c_draws(anchored["idata"], anchored["builder"], "Chicken")
        is not None
    )
    assert (
        extract_free_c_draws(
            anchored["idata"],
            anchored["builder"],
            "2024 Leading Chat LLMs",
        )
        is not None
    )
    assert extract_free_c_draws(human_free["idata"], human_free["builder"], "Human") is not None
    assert extract_free_c_draws(eliza_free["idata"], eliza_free["builder"], "ELIZA") is not None

    indicator_index = build_indicator_index(stance_data, anchored["builder"])
    first_indicator = indicator_index[0]
    for system_name, _ in ANCHORED_SYSTEM_CONFIGS:
        sp = anchored["builder"]._sys_prefix(system_name)
        assert f"{sp}__{first_indicator.varname}_p" in anchored["idata"].posterior.data_vars
        assert (
            f"{sp}__{first_indicator.varname}_pz1"
            in anchored["idata"].posterior.data_vars
        )

    beta_pres_by_key, beta_abs_by_key = extract_beta_draws_by_node(
        anchored["idata"],
        anchored["builder"],
        stance_data,
    )
    intercepts, slopes, affine_index = propagate_affine_indicator_coefficients(
        anchored["idata"],
        anchored["builder"],
        stance_data,
        indicator_index=indicator_index,
    )
    assert [spec.node_key for spec in affine_index] == [
        spec.node_key for spec in indicator_index
    ]

    for c_value in (0.2, 0.5, 0.8):
        affine_q = intercepts + slopes * c_value
        direct_q = _direct_indicator_probs(
            stance_data,
            indicator_index,
            beta_pres_by_key,
            beta_abs_by_key,
            c_value,
        )
        assert np.allclose(affine_q, direct_q, atol=1e-8)

    deltas_h = compare_system_medians(
        anchored["c_summary"],
        human_free["c_summary"],
    )
    deltas_e = compare_system_medians(
        anchored["c_summary"],
        eliza_free["c_summary"],
    )
    assert set(deltas_h) == {"Chicken", "2024 Leading Chat LLMs"}
    assert set(deltas_e) == {"Chicken", "2024 Leading Chat LLMs"}
    print("end-to-end reference-recovery smoke passed")


def main() -> None:
    numerical_helper_checks()
    end_to_end_smoke()


if __name__ == "__main__":
    main()
