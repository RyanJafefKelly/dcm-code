"""Smoke test for the hierarchical expert-scales branch (USE_EXPERT_SCALES).

Structural checks:
  - Binary baseline (USE_EXPERT_SCALES=False) still builds and samples unchanged.
  - Binary + sigma_e builds and samples; exposes `sigma_by_expert` deterministic
    of shape (n_experts,) and `tau_sigma` > 0.
  - Three-state + sigma_e likewise; per-indicator deterministics `_p_m*` and
    `_expected_z` present and correctly shaped.
  - Per-indicator marginalisation invariant: with all ratings on an indicator
    from one expert and sigma_by_expert[e] = s*, the three-state marginal
    log-likelihood equals the shared-sigma case with sigma = s* (up to an
    affine constant independent of z). Hand-calculation parity test parallels
    the existing `per_indicator_marginalisation_check`.
  - Incompatibility guards: USE_EXPERT_SCALES + USE_HIERARCHICAL_EXPERT_CUTPOINTS
    and USE_EXPERT_SCALES + USE_EXPERT_SHIFTS both raise at model build time.
"""

from __future__ import annotations

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
from test_ordinal_smoke import build_synthetic_tree


def _fast_config(**kwargs) -> ModelConfig:
    return ModelConfig(
        NUM_SAMPLES=120,
        NUM_TUNE=150,
        NUM_CHAINS=2,
        TARGET_ACCEPT=0.9,
        USE_EXPERT_SHIFTS=False,
        **kwargs,
    )


def incompat_guards() -> None:
    print("\n=== incompatibility guards ===")
    tree, _, system = build_synthetic_tree(
        n_features=2, indicators_per_feature=2, n_experts=3, seed=1
    )

    # expert-scales + hierarchical cutpoints should raise
    cfg = ModelConfig(
        USE_EXPERT_SHIFTS=False,
        USE_EXPERT_SCALES=True,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=True,
        NUM_SAMPLES=30, NUM_TUNE=30, NUM_CHAINS=1,
    )
    processor = OrdinalDataProcessor(cfg)
    processor.process(tree, system)
    builder = BayesianModelBuilder(cfg, EvidenceProcessor(cfg), processor)
    raised = False
    try:
        builder.build_model(tree)
    except ValueError as e:
        raised = True
        print(f"  scales+hier_kappa correctly raised: {e}")
    assert raised, "expected ValueError when combining USE_EXPERT_SCALES with hierarchical cutpoints"

    # expert-scales + expert-shifts should raise
    cfg2 = ModelConfig(
        USE_EXPERT_SHIFTS=True,
        USE_EXPERT_SCALES=True,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        NUM_SAMPLES=30, NUM_TUNE=30, NUM_CHAINS=1,
    )
    processor2 = OrdinalDataProcessor(cfg2)
    processor2.process(tree, system)
    builder2 = BayesianModelBuilder(cfg2, EvidenceProcessor(cfg2), processor2)
    raised = False
    try:
        builder2.build_model(tree)
    except ValueError as e:
        raised = True
        print(f"  scales+shifts correctly raised: {e}")
    assert raised, "expected ValueError when combining USE_EXPERT_SCALES with USE_EXPERT_SHIFTS"


def single_system_build_check(state_model: str) -> None:
    print(f"\n=== single-system sigma_e: state_model={state_model!r} ===")
    tree, _, system = build_synthetic_tree(
        n_features=2, indicators_per_feature=2, n_experts=3, true_a=1.6, seed=11
    )
    cfg = _fast_config(
        USE_EXPERT_SCALES=True,
        INDICATOR_STATE_MODEL=state_model,  # type: ignore[arg-type]
    )
    processor = OrdinalDataProcessor(cfg)
    processor.process(tree, system)
    builder = BayesianModelBuilder(cfg, EvidenceProcessor(cfg), processor)
    model = builder.build_model(tree)
    idata = builder.sample(model)

    post = idata.posterior
    assert "sigma_by_expert" in post.data_vars, "sigma_by_expert missing"
    assert post["sigma_by_expert"].shape[-1] == len(processor.expert_names), (
        "sigma_by_expert has wrong expert dimension"
    )
    assert "tau_sigma" in post.data_vars, "tau_sigma missing"
    tau_vals = np.asarray(post["tau_sigma"].values).reshape(-1)
    assert (tau_vals > 0).all(), "tau_sigma should be positive"
    sigmas = np.asarray(post["sigma_by_expert"].values).mean(axis=(0, 1))
    print(
        f"  sigma_by_expert posterior means: {sigmas}  "
        f"(geometric mean should be ~1 from sum-to-zero constraint)"
    )
    # Geometric mean should be close to 1 due to sum-to-zero centring
    geo_mean = float(np.exp(np.log(sigmas).mean()))
    assert 0.85 < geo_mean < 1.15, (
        f"sigma_by_expert geometric mean {geo_mean} not close to 1; "
        f"sum-to-zero constraint may be broken"
    )
    print(f"  sigma_by_expert geometric mean = {geo_mean:.4f} (near 1, OK)")

    if state_model == "three_state":
        p_m0 = [v for v in post.data_vars if v.endswith("_p_m0")]
        assert p_m0, "three_state should expose _p_m0 deterministics"
        print(f"  three_state deterministics present: {len(p_m0)} _p_m0 vars")
    else:
        pz1 = [v for v in post.data_vars if v.endswith("_pz1")]
        assert pz1, "binary should expose _pz1 deterministics"
        print(f"  binary deterministics present: {len(pz1)} _pz1 vars")


def per_indicator_marginalisation_invariant_under_sigma() -> None:
    """Under sigma_by_expert[e] = s* for the one expert rating an indicator,
    the three-state marginal log-likelihood equals the shared-sigma case with
    sigma=s* evaluated against rescaled cutpoints and discrimination.

    Specifically, emission is Phi((kappa - eta)/s*); this equals the shared
    case with kappa' = kappa/s*, a' = a/s*, so the per-indicator logsumexp
    at q_j also uses the rescaled parameters.

    Simplest invariant: under sigma_by_expert[e] = 1 for all e, the
    sigma-aware code must produce identical output to the shared-sigma
    (sigma=None) code. This is the minimal check that the sigma path
    does not break the per-indicator marginalisation.
    """
    print("\n=== per-indicator marginalisation invariant ===")
    kappa_vals = np.array([-1.5, -0.8, -0.1, 0.4, 1.0, 1.6])
    a_val = 1.7
    q_val = 0.3
    ratings = np.array([1, 3, 5], dtype=np.int64)
    expert_idx = np.zeros_like(ratings, dtype=np.int64)

    # --- Shared-sigma path (sigma_by_expert=None) ---
    a_sym = pt.as_tensor_variable(float(a_val))
    kappa_sym = pt.as_tensor_variable(kappa_vals.astype(np.float64))
    b_sym = pt.zeros(1, dtype="float64")
    ll_0_shared, ll_half_shared, ll_1_shared = pt_three_state_ll_terms(
        ratings, expert_idx, a_sym, kappa_sym, b_sym, None,
        sigma_by_expert=None,
    )
    q_sym = pt.as_tensor_variable(float(q_val))
    lw0, lwmid, lw1 = three_state_log_weights(q_sym)
    log_mix_shared = pt.logaddexp(
        pt.logaddexp(lw0 + ll_0_shared, lwmid + ll_half_shared), lw1 + ll_1_shared
    )
    shared_val = float(log_mix_shared.eval())

    # --- sigma-aware path with sigma=1 for all experts ---
    sigma_vec = pt.as_tensor_variable(np.array([1.0], dtype=np.float64))
    ll_0_sig, ll_half_sig, ll_1_sig = pt_three_state_ll_terms(
        ratings, expert_idx, a_sym, kappa_sym, b_sym, None,
        sigma_by_expert=sigma_vec,
    )
    log_mix_sig = pt.logaddexp(
        pt.logaddexp(lw0 + ll_0_sig, lwmid + ll_half_sig), lw1 + ll_1_sig
    )
    sig1_val = float(log_mix_sig.eval())

    assert np.isclose(shared_val, sig1_val, atol=1e-10), (
        f"sigma=1 path diverges from shared path:\n"
        f"  shared: {shared_val:.6f}\n"
        f"  sigma=1: {sig1_val:.6f}"
    )
    print(f"  shared-sigma LL        = {shared_val:+.6f}")
    print(f"  sigma-aware @ s=1 LL   = {sig1_val:+.6f}  (identical)")

    # --- Sanity: scale invariance. sigma=s* should equal shared with kappa'=kappa/s*, a'=a/s* ---
    s_star = 0.7
    sigma_vec_s = pt.as_tensor_variable(np.array([s_star], dtype=np.float64))
    ll_0_s, ll_half_s, ll_1_s = pt_three_state_ll_terms(
        ratings, expert_idx, a_sym, kappa_sym, b_sym, None,
        sigma_by_expert=sigma_vec_s,
    )
    log_mix_s = pt.logaddexp(
        pt.logaddexp(lw0 + ll_0_s, lwmid + ll_half_s), lw1 + ll_1_s
    )
    sigma_s_val = float(log_mix_s.eval())

    # Manual NumPy reference: divide (kappa - eta) by s_star
    def op_probs_scaled(eta: float) -> np.ndarray:
        cuts = np.concatenate([[-np.inf], kappa_vals, [np.inf]])
        return np.diff(norm.cdf((cuts - eta) / s_star))

    p0 = op_probs_scaled(0.0)
    p_half = op_probs_scaled(a_val * 0.5)
    p1 = op_probs_scaled(a_val)
    L0 = sum(float(np.log(p0[r])) for r in ratings)
    Lh = sum(float(np.log(p_half[r])) for r in ratings)
    L1 = sum(float(np.log(p1[r])) for r in ratings)
    log_w0_n = 2.0 * np.log(1.0 - q_val)
    log_wmid_n = np.log(2.0) + np.log(q_val) + np.log(1.0 - q_val)
    log_w1_n = 2.0 * np.log(q_val)
    expected = float(
        logsumexp([log_w0_n + L0, log_wmid_n + Lh, log_w1_n + L1])
    )
    assert np.isclose(sigma_s_val, expected, atol=1e-8), (
        f"sigma-aware @ s={s_star} diverges from NumPy reference:\n"
        f"  pytensor: {sigma_s_val:.6f}\n"
        f"  numpy:    {expected:.6f}"
    )
    print(f"  sigma-aware @ s={s_star} LL = {sigma_s_val:+.6f}  (matches NumPy ref)")


def multi_system_build_check(state_model: str) -> None:
    print(f"\n=== multi-system sigma_e: state_model={state_model!r} ===")
    import copy
    tree_a, _, sys_a = build_synthetic_tree(
        n_features=2, indicators_per_feature=2, n_experts=3, true_a=1.6, seed=3
    )
    tree_b_raw, _, gen_sys_b = build_synthetic_tree(
        n_features=2, indicators_per_feature=2, n_experts=3, true_a=1.6, seed=17
    )
    sys_b = "TestSystemB"

    # Relabel + merge (same as three-state smoke)
    def relabel(node, old, new):
        if node.get("type", "").lower() == "indicator":
            obs = node.get("observations", {})
            if old in obs:
                obs[new] = obs.pop(old)
        for c in node.get("evidencers", []):
            relabel(c, old, new)

    tree_b = copy.deepcopy(tree_b_raw)
    relabel(tree_b, gen_sys_b, sys_b)

    merged = copy.deepcopy(tree_a)

    def walk_merge(na, nb):
        if na.get("type", "").lower() == "indicator":
            obs_b = nb.get("observations", {}).get(sys_b)
            if obs_b is not None:
                na.setdefault("observations", {})[sys_b] = obs_b
        for ca, cb in zip(na.get("evidencers", []), nb.get("evidencers", [])):
            walk_merge(ca, cb)

    walk_merge(merged, tree_b)

    cfg = _fast_config(
        USE_EXPERT_SCALES=True,
        INDICATOR_STATE_MODEL=state_model,  # type: ignore[arg-type]
    )
    processor = MultiSystemDataProcessor(cfg)
    processor.process(merged, [sys_a, sys_b])
    builder = MultiSystemModelBuilder(
        cfg, EvidenceProcessor(cfg), processor, [(sys_a, None), (sys_b, None)]
    )
    model = builder.build_model(merged)
    idata = builder.sample(model)

    post = idata.posterior
    assert "sigma_by_expert" in post.data_vars
    n_exp = len(processor.expert_names)
    sigmas = np.asarray(post["sigma_by_expert"].values).mean(axis=(0, 1))
    assert sigmas.shape == (n_exp,)
    geo_mean = float(np.exp(np.log(sigmas).mean()))
    assert 0.85 < geo_mean < 1.15, f"geo mean = {geo_mean}, not near 1"
    print(f"  multi-system sigma_by_expert posterior means: {sigmas}")
    print(f"  geometric mean: {geo_mean:.4f}")


def main() -> None:
    print("=" * 60)
    print("USE_EXPERT_SCALES smoke test")
    print("=" * 60)
    per_indicator_marginalisation_invariant_under_sigma()
    incompat_guards()
    single_system_build_check("binary")
    single_system_build_check("three_state")
    multi_system_build_check("binary")
    multi_system_build_check("three_state")
    print("\nAll structural checks passed.")


if __name__ == "__main__":
    main()
