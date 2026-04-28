"""Smoke test for the three-state leaf-indicator branch.

Structural checks only (build + short sample + posterior shape):
  - Binary baseline still builds and samples with the new config flag present.
  - Three-state branch builds and samples on a tiny synthetic tree.
  - Per-indicator deterministics match the expected schema for each branch:
      binary:       ``_pz1``
      three-state:  ``_p_m0``, ``_p_m1``, ``_p_m2``, ``_expected_z``
  - Three-state component probabilities sum to 1 across draws.
  - ``expected_z`` equals ``0.5 * p_m1 + p_m2`` within numerical tolerance.
  - Multi-system builder also dispatches correctly under both state models.
"""

from __future__ import annotations

import copy

import numpy as np

import pytensor.tensor as pt
from scipy.special import logsumexp
from scipy.stats import norm

from dcm_model_ordinal import (
    BayesianModelBuilder,
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    MultiSystemModelBuilder,
    OrdinalDataProcessor,
    pt_three_state_ll_terms,
    three_state_log_weights,
)
from dcm_ppc_ordinal import (
    inspect_indicator_state_for_cell,
    inspect_pz1_for_cell,
    per_expert_ppc_multisystem,
    per_expert_system_ppc_multisystem,
    per_expert_system_tree_implied_ppc_multisystem,
)
from test_ordinal_smoke import build_synthetic_tree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def per_indicator_marginalisation_check() -> None:
    """Hand-calculation check that three-state LL marginalises over m at the indicator level.

    For a single indicator with multiple ratings {r_1, ..., r_n} and prior
    m ~ Binomial(2, q), the correct marginal log-likelihood is

        log P(r_1, ..., r_n | q, theta)
          = logsumexp_{m in {0,1,2}} [log w_m(q) + sum_i log p(r_i | eta = m/2 a, theta)]

    A product of per-rating mixtures would be wrong -- it would allow each
    rating to have its own latent m. For n > 1 the two quantities differ,
    so we test: (a) the NumPy per-indicator reference and the NumPy
    per-rating computation disagree, and (b) the PyTensor helper chain
    (pt_three_state_ll_terms + three_state_log_weights + logaddexp) matches
    the per-indicator reference.
    """
    print("\n=== hand-calculation: per-indicator marginalisation ===")
    kappa_vals = np.array([-1.5, -0.8, -0.1, 0.4, 1.0, 1.6])
    a_val = 1.7
    q_val = 0.3
    ratings = np.array([1, 3, 5], dtype=np.int64)
    expert_idx = np.zeros_like(ratings, dtype=np.int64)

    # --- NumPy reference ---
    def op_probs(eta: float) -> np.ndarray:
        cuts = np.concatenate([[-np.inf], kappa_vals, [np.inf]])
        return np.diff(norm.cdf(cuts - eta))

    p_components = {
        0: op_probs(0.0),
        1: op_probs(a_val * 0.5),
        2: op_probs(a_val),
    }
    L_per_m = {
        m: sum(float(np.log(p_components[m][r])) for r in ratings)
        for m in (0, 1, 2)
    }
    log_w0 = 2.0 * np.log(1.0 - q_val)
    log_wmid = np.log(2.0) + np.log(q_val) + np.log(1.0 - q_val)
    log_w1 = 2.0 * np.log(q_val)
    expected_per_indicator = float(
        logsumexp(
            [log_w0 + L_per_m[0], log_wmid + L_per_m[1], log_w1 + L_per_m[2]]
        )
    )
    expected_per_rating = float(
        sum(
            logsumexp(
                [
                    log_w0 + np.log(p_components[0][r]),
                    log_wmid + np.log(p_components[1][r]),
                    log_w1 + np.log(p_components[2][r]),
                ]
            )
            for r in ratings
        )
    )

    # (a) Test setup must make the two quantities differ.
    assert not np.isclose(expected_per_indicator, expected_per_rating), (
        "Test setup degenerate: per-indicator and per-rating must differ for n > 1"
    )

    # --- PyTensor helper chain ---
    a_sym = pt.as_tensor_variable(float(a_val))
    kappa_sym = pt.as_tensor_variable(kappa_vals.astype(np.float64))
    b_sym = pt.zeros(1, dtype="float64")
    ll_0, ll_half, ll_1 = pt_three_state_ll_terms(
        ratings, expert_idx, a_sym, kappa_sym, b_sym, None
    )
    q_sym = pt.as_tensor_variable(float(q_val))
    lw0, lwmid, lw1 = three_state_log_weights(q_sym)
    log_mix_sym = pt.logaddexp(
        pt.logaddexp(lw0 + ll_0, lwmid + ll_half), lw1 + ll_1
    )
    actual = float(log_mix_sym.eval())

    assert np.isclose(actual, expected_per_indicator, atol=1e-6), (
        f"Helper marginal LL = {actual:.6f}\n"
        f"  per-indicator (expected) = {expected_per_indicator:.6f}\n"
        f"  per-rating    (wrong)    = {expected_per_rating:.6f}\n"
        f"  n_ratings = {len(ratings)}, q = {q_val}, a = {a_val}"
    )
    print(
        f"  per-indicator LL    = {expected_per_indicator:+.6f}\n"
        f"  per-rating LL       = {expected_per_rating:+.6f}  (different; good)\n"
        f"  PyTensor helper LL  = {actual:+.6f}  (matches per-indicator)"
    )


def _fast_config(state_model: str) -> ModelConfig:
    return ModelConfig(
        NUM_SAMPLES=120,
        NUM_TUNE=150,
        NUM_CHAINS=2,
        TARGET_ACCEPT=0.9,
        USE_EXPERT_SHIFTS=False,
        INDICATOR_STATE_MODEL=state_model,  # type: ignore[arg-type]
    )


def _relabel_system(tree: dict, old: str, new: str) -> dict:
    out = copy.deepcopy(tree)

    def walk(node: dict) -> None:
        if node.get("type", "").lower() == "indicator":
            obs = node.get("observations", {})
            if old in obs:
                obs[new] = obs.pop(old)
            stats = node.get("observation_stats", {})
            if old in stats:
                stats[new] = stats.pop(old)
        for child in node.get("evidencers", []):
            walk(child)

    walk(out)
    return out


def _merge_system_tree(base: dict, other: dict, system_name: str) -> dict:
    merged = copy.deepcopy(base)

    def walk(node_a: dict, node_b: dict) -> None:
        if node_a.get("type", "").lower() == "indicator":
            node_a.setdefault("observations", {})[system_name] = node_b[
                "observations"
            ][system_name]
            node_a.setdefault("observation_stats", {})[system_name] = node_b[
                "observation_stats"
            ][system_name]
        for child_a, child_b in zip(
            node_a.get("evidencers", []), node_b.get("evidencers", [])
        ):
            walk(child_a, child_b)

    walk(merged, other)
    return merged


def _collect_indicator_varnames(builder) -> list[str]:
    """Return sanitised variable names for indicators only (not features)."""
    return [
        varname
        for key, varname in builder.node_to_varname.items()
        # Indicators are the leaves: we know them by presence of '_pz1' / '_p_m0' vars
        # but here we can't peek at the model; instead return all names and let
        # the caller filter on existence in idata.
    ] or list(builder.node_to_varname.values())


# ---------------------------------------------------------------------------
# Single-system checks
# ---------------------------------------------------------------------------


def single_system_check(state_model: str) -> None:
    print(f"\n=== single-system: state_model={state_model!r} ===")
    tree, _, system = build_synthetic_tree(
        n_features=2, indicators_per_feature=2, n_experts=3, true_a=1.6, seed=11
    )
    config = _fast_config(state_model)

    processor = OrdinalDataProcessor(config)
    processor.process(tree, system)
    evidence_proc = EvidenceProcessor(config)
    builder = BayesianModelBuilder(config, evidence_proc, processor)
    model = builder.build_model(tree)
    idata = builder.sample(model)

    post = idata.posterior
    indicator_varnames = list(builder.node_to_varname.values())

    if state_model == "binary":
        pz1_names = [vn for vn in post.data_vars if vn.endswith("_pz1")]
        assert pz1_names, "binary branch should expose _pz1 deterministics"
        for vn in pz1_names:
            vals = np.asarray(post[vn].values).reshape(-1)
            assert (vals >= 0).all() and (vals <= 1).all(), (
                f"{vn} out of [0,1]"
            )
        # Three-state-specific variables must NOT be present
        assert not any(vn.endswith("_p_m0") for vn in post.data_vars), (
            "binary branch should not expose _p_m0"
        )
        assert not any(vn.endswith("_expected_z") for vn in post.data_vars), (
            "binary branch should not expose _expected_z"
        )
        print(
            f"  binary OK: {len(pz1_names)} pz1 deterministics; "
            f"ranges in [0, 1]"
        )
        return

    if state_model == "three_state":
        p_m0_names = sorted(vn for vn in post.data_vars if vn.endswith("_p_m0"))
        p_m1_names = sorted(vn for vn in post.data_vars if vn.endswith("_p_m1"))
        p_m2_names = sorted(vn for vn in post.data_vars if vn.endswith("_p_m2"))
        ez_names = sorted(
            vn for vn in post.data_vars if vn.endswith("_expected_z")
        )
        assert p_m0_names, "three_state branch should expose _p_m0 deterministics"
        assert len(p_m0_names) == len(p_m1_names) == len(p_m2_names) == len(
            ez_names
        ), "mismatched count across _p_m0 / _p_m1 / _p_m2 / _expected_z"
        # Binary-specific name must be absent
        assert not any(vn.endswith("_pz1") for vn in post.data_vars), (
            "three_state branch should not expose _pz1"
        )

        for pm0, pm1, pm2, ez in zip(
            p_m0_names, p_m1_names, p_m2_names, ez_names
        ):
            v0 = np.asarray(post[pm0].values).reshape(-1)
            v1 = np.asarray(post[pm1].values).reshape(-1)
            v2 = np.asarray(post[pm2].values).reshape(-1)
            vz = np.asarray(post[ez].values).reshape(-1)
            assert (v0 >= 0).all() and (v1 >= 0).all() and (v2 >= 0).all()
            s = v0 + v1 + v2
            assert np.allclose(s, 1.0, atol=1e-4), (
                f"{pm0.split('_p_m0')[0]} component probs do not sum to 1 "
                f"(max deviation {np.max(np.abs(s - 1.0)):.2e})"
            )
            expected = 0.5 * v1 + v2
            assert np.allclose(vz, expected, atol=1e-6), (
                f"{ez} disagrees with 0.5 * p_m1 + p_m2"
            )
            assert (vz >= 0).all() and (vz <= 1).all()
        print(
            f"  three_state OK: {len(p_m0_names)} indicators; "
            f"components normalise to 1; expected_z matches 0.5 p_m1 + p_m2"
        )
        return

    raise ValueError(f"unexpected state_model: {state_model!r}")


# ---------------------------------------------------------------------------
# Multi-system checks
# ---------------------------------------------------------------------------


def _ppc_checks(
    idata,
    builder,
    processor,
    *,
    state_model: str,
    sys_a: str,
    sys_b: str,
    K: int,
) -> None:
    """Assert shape + signed-tail invariants for pooled and stratified PPC."""
    pooled = per_expert_ppc_multisystem(
        idata, builder, processor, n_draws=60, seed=0
    )
    strat = per_expert_system_ppc_multisystem(
        idata, builder, processor, n_draws=60, seed=0
    )
    tree_strat = per_expert_system_tree_implied_ppc_multisystem(
        idata, builder, processor, n_draws=60, seed=0
    )

    for label, results in [("pooled", pooled), ("strat", strat), ("tree_strat", tree_strat)]:
        assert results, f"{state_model}/{label} PPC returned empty dict"
        for key, r in results.items():
            assert r["obs_hist"].shape == (K,)
            assert r["pred_hist_mean"].shape == (K,)
            assert np.isclose(r["obs_hist"].sum(), 1.0, atol=1e-6)
            assert np.isclose(r["pred_hist_mean"].sum(), 1.0, atol=1e-4)
            assert (r["pred_hist_mean"] >= 0).all()
            # New signed tail / middle-mass invariants
            for fld in (
                "obs_left", "pred_left_mean", "delta_left_mean",
                "obs_right", "pred_right_mean", "delta_right_mean",
                "obs_mid", "pred_mid_mean", "delta_mid_mean",
            ):
                assert fld in r, f"{state_model}/{label}/{key}: missing field {fld}"
            # Sanity checks on ranges
            for fld in ("pred_left_mean", "pred_right_mean", "pred_mid_mean"):
                assert 0.0 - 1e-9 <= r[fld] <= 1.0 + 1e-9, (
                    f"{state_model}/{label}/{key}: {fld}={r[fld]} out of range"
                )
            # delta = pred - obs, should match
            assert np.isclose(
                r["delta_left_mean"], r["pred_left_mean"] - r["obs_left"], atol=1e-9
            )
            assert np.isclose(
                r["delta_right_mean"], r["pred_right_mean"] - r["obs_right"], atol=1e-9
            )
    print(f"  PPC checks OK ({state_model}): pooled, stratified, tree-implied all valid")

    # inspect_indicator_state_for_cell dispatches on state model
    expert_names = processor.expert_names
    e_name = expert_names[0]
    rows = inspect_indicator_state_for_cell(
        idata, builder, processor, e_name, sys_a
    )
    assert rows, (
        f"inspect_indicator_state_for_cell ({state_model}) returned no rows for "
        f"{e_name} x {sys_a}"
    )
    for row in rows:
        assert 0.0 <= row["median"] <= 1.0
        assert row["p03"] <= row["median"] <= row["p97"]
    print(
        f"  inspect_indicator_state_for_cell ({state_model}) OK: {len(rows)} "
        f"indicator rows under {'pz1' if state_model == 'binary' else 'expected_z'}"
    )
    # Backwards-compatible alias still works.
    legacy_rows = inspect_pz1_for_cell(idata, builder, processor, e_name, sys_a)
    assert legacy_rows == rows, "inspect_pz1_for_cell alias diverged"


def multi_system_check(state_model: str) -> None:
    print(f"\n=== multi-system: state_model={state_model!r} ===")
    tree_a, _, sys_a = build_synthetic_tree(
        n_features=2, indicators_per_feature=2, n_experts=3, true_a=1.6, seed=3
    )
    tree_b_raw, _, gen_sys_b = build_synthetic_tree(
        n_features=2, indicators_per_feature=2, n_experts=3, true_a=1.6, seed=17
    )
    sys_b = "TestSystemB"
    tree_b = _relabel_system(tree_b_raw, gen_sys_b, sys_b)
    merged = _merge_system_tree(tree_a, tree_b, sys_b)
    system_configs = [(sys_a, None), (sys_b, None)]

    config = _fast_config(state_model)
    processor = MultiSystemDataProcessor(config)
    processor.process(merged, [sys_a, sys_b])
    evidence_proc = EvidenceProcessor(config)
    builder = MultiSystemModelBuilder(config, evidence_proc, processor, system_configs)
    model = builder.build_model(merged)
    idata = builder.sample(model)

    post = idata.posterior
    sp_a = builder._sys_prefix(sys_a)
    sp_b = builder._sys_prefix(sys_b)

    if state_model == "binary":
        any_pz1 = [vn for vn in post.data_vars if vn.endswith("_pz1")]
        a_pz1 = [vn for vn in any_pz1 if vn.startswith(f"{sp_a}__")]
        b_pz1 = [vn for vn in any_pz1 if vn.startswith(f"{sp_b}__")]
        assert a_pz1 and b_pz1, (
            f"expected per-system _pz1 deterministics under both prefixes "
            f"({sp_a}__, {sp_b}__), got {any_pz1}"
        )
        print(
            f"  binary multi-system OK: {len(a_pz1)} + {len(b_pz1)} pz1 "
            f"deterministics across two systems"
        )
        _ppc_checks(
            idata, builder, processor,
            state_model=state_model, sys_a=sys_a, sys_b=sys_b,
            K=config.N_CATEGORIES,
        )
        return

    if state_model == "three_state":
        a_p_m0 = [
            vn for vn in post.data_vars
            if vn.startswith(f"{sp_a}__") and vn.endswith("_p_m0")
        ]
        b_p_m0 = [
            vn for vn in post.data_vars
            if vn.startswith(f"{sp_b}__") and vn.endswith("_p_m0")
        ]
        assert a_p_m0 and b_p_m0, (
            f"expected per-system _p_m0 deterministics under {sp_a}__ / {sp_b}__"
        )
        # Spot-check one cell sums to 1
        base = a_p_m0[0][: -len("_p_m0")]
        v0 = np.asarray(post[f"{base}_p_m0"].values).reshape(-1)
        v1 = np.asarray(post[f"{base}_p_m1"].values).reshape(-1)
        v2 = np.asarray(post[f"{base}_p_m2"].values).reshape(-1)
        assert np.allclose(v0 + v1 + v2, 1.0, atol=1e-4)
        print(
            f"  three_state multi-system OK: {len(a_p_m0)} + {len(b_p_m0)} "
            f"indicators; normalisation holds on spot-checked cell"
        )
        _ppc_checks(
            idata, builder, processor,
            state_model=state_model, sys_a=sys_a, sys_b=sys_b,
            K=config.N_CATEGORIES,
        )
        return

    raise ValueError(f"unexpected state_model: {state_model!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Three-state leaf-indicator smoke test")
    print("=" * 60)
    per_indicator_marginalisation_check()
    single_system_check("binary")
    single_system_check("three_state")
    multi_system_check("binary")
    multi_system_check("three_state")
    print("\nAll structural checks passed.")


if __name__ == "__main__":
    main()
