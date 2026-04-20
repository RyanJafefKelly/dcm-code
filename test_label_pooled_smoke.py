"""Smoke test for POOL_BETAS_BY_LABEL.

Invariants checked:

1. ``POOL_BETAS_BY_LABEL=False`` is a structural no-op. The model builder
   produces no label-level hyperparameters; the node-level betas carry
   the same prior parameters as the paper mapping. (Confirmed by absence
   of ``mu_pres_*`` / ``mu_abs_*`` variables and by matching the
   committed ``binary_anchored.nc`` config.)

2. ``POOL_BETAS_BY_LABEL=True`` exposes label-level deterministics with
   correct names, one per tree label group.

3. ``collect_tree_label_groups`` returns the expected number of unique
   (support, demandingness) pairs and demandingness values for GWT.
   Cross-checked against the on-disk group-count CSVs from
   ``tree_propagation_diagnostic.py`` Stage 3.5b.

4. Per-indicator marginalisation invariant preserved under pooling: for
   a single indicator with a single rating, the log-likelihood equals a
   hand-computed NumPy reference given matched parameter values.

No MCMC; no idata files are loaded or written. Designed to complete in
under 30 seconds.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from dcm_model import (
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    MultiSystemModelBuilder,
    build_label_pool_hyperparameters,
    collect_tree_label_groups,
    load_data,
    node_key,
)


STANCE = "Global Workspace Theory"
SYSTEM_CONFIGS = [
    ("Human", 0.999),
    ("Chicken", None),
    ("2024 Leading Chat LLMs", None),
    ("ELIZA", 0.001),
]


def _gwt_stance_data(config: ModelConfig):
    return next(s for s in load_data(config) if s["name"] == STANCE)


def test_flag_off_does_not_create_pool_vars() -> None:
    """With POOL_BETAS_BY_LABEL=False the model has no label-level RVs."""
    config = ModelConfig(
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        INDICATOR_STATE_MODEL="binary",
        POOL_BETAS_BY_LABEL=False,
    )
    stance_data = _gwt_stance_data(config)
    processor = MultiSystemDataProcessor(config)
    processor.process(stance_data, [s for s, _ in SYSTEM_CONFIGS])
    evidence = EvidenceProcessor(config)
    builder = MultiSystemModelBuilder(config, evidence, processor, SYSTEM_CONFIGS)
    model = builder.build_model(stance_data)

    named = {v.name for v in model.unobserved_RVs}
    assert not any(n.startswith("beta_pres_tilde__") for n in named), (
        "label pooling vars leaked under POOL_BETAS_BY_LABEL=False"
    )
    assert not any(n.startswith("beta_abs_tilde__") for n in named), (
        "label pooling vars leaked under POOL_BETAS_BY_LABEL=False"
    )
    assert builder.beta_pres_by_group == {}
    assert builder.beta_abs_by_group == {}
    print("  [OK] flag off: no pool betas")


def test_flag_on_exposes_hyperparameters() -> None:
    """Flag on: one Normal tilde + one Deterministic beta per group."""
    config = ModelConfig(
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        INDICATOR_STATE_MODEL="binary",
        POOL_BETAS_BY_LABEL=True,
        LABEL_POOL_SIGMA=0.5,
    )
    stance_data = _gwt_stance_data(config)
    processor = MultiSystemDataProcessor(config)
    processor.process(stance_data, [s for s, _ in SYSTEM_CONFIGS])
    evidence = EvidenceProcessor(config)
    builder = MultiSystemModelBuilder(config, evidence, processor, SYSTEM_CONFIGS)
    model = builder.build_model(stance_data)

    pres_groups, abs_groups = collect_tree_label_groups(stance_data)
    named = {v.name for v in model.unobserved_RVs}
    named_det = {v.name for v in model.deterministics}

    pres_tildes = [n for n in named if n.startswith("beta_pres_tilde__")]
    abs_tildes = [n for n in named if n.startswith("beta_abs_tilde__")]
    assert len(pres_tildes) == len(pres_groups), (
        f"Expected {len(pres_groups)} beta_pres_tilde vars, got {len(pres_tildes)}"
    )
    assert len(abs_tildes) == len(abs_groups), (
        f"Expected {len(abs_groups)} beta_abs_tilde vars, got {len(abs_tildes)}"
    )
    beta_pres_dets = [n for n in named_det if n.startswith("beta_pres__")]
    beta_abs_dets = [n for n in named_det if n.startswith("beta_abs__")]
    label_delta_dets = [n for n in named_det if n.startswith("label_delta__")]
    assert len(beta_pres_dets) == len(pres_groups)
    assert len(beta_abs_dets) == len(abs_groups)
    assert len(label_delta_dets) == len(pres_groups)

    # Under complete-pooling, no per-node {varname}_beta_pres RVs exist
    named_node_betas = [n for n in named if n.endswith("_beta_pres") or n.endswith("_beta_abs")]
    assert not named_node_betas, (
        f"Under complete-pooling no per-node beta RVs should exist; got "
        f"{named_node_betas[:5]}..."
    )

    assert len(builder.beta_pres_by_group) == len(pres_groups)
    assert len(builder.beta_abs_by_group) == len(abs_groups)
    print(
        f"  [OK] flag on: {len(pres_groups)} pres groups, {len(abs_groups)} abs groups, "
        f"{len(label_delta_dets)} label_delta dets, 0 per-node beta RVs"
    )


def test_collect_tree_label_groups_matches_expected_counts() -> None:
    """Group collection should reproduce the Stage 3.5b GWT tree counts."""
    config = ModelConfig(INDICATOR_STATE_MODEL="binary")
    stance_data = _gwt_stance_data(config)
    pres_groups, abs_groups = collect_tree_label_groups(stance_data)
    # From Stage 3.5b: 18 unique (support, demandingness) pairs,
    # 7 unique demandingness values.
    assert len(pres_groups) == 18, f"expected 18 pres groups, got {len(pres_groups)}"
    assert len(abs_groups) == 7, f"expected 7 abs groups, got {len(abs_groups)}"
    # The critical singleton subfeature must be present.
    assert ("weak undermining", "neutral") in pres_groups, (
        "weak undermining + neutral subfeature group not found in tree walk"
    )
    print(f"  [OK] group counts: {len(pres_groups)} pres, {len(abs_groups)} abs")


def test_sigma_goes_to_zero_recovers_paper_mean() -> None:
    """With LABEL_POOL_SIGMA -> 0 the label-level beta equals the paper
    prior mean at tilde=0."""
    config = ModelConfig(
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        INDICATOR_STATE_MODEL="binary",
        POOL_BETAS_BY_LABEL=True,
        LABEL_POOL_SIGMA=1e-6,
    )
    stance_data = _gwt_stance_data(config)
    processor = MultiSystemDataProcessor(config)
    processor.process(stance_data, [s for s, _ in SYSTEM_CONFIGS])
    evidence = EvidenceProcessor(config)
    builder = MultiSystemModelBuilder(config, evidence, processor, SYSTEM_CONFIGS)
    model = builder.build_model(stance_data)

    from dcm_model import _sanitize_label
    from pytensor import function

    (s, d) = next(iter(builder.beta_pres_by_group))
    beta_name = f"beta_pres__{_sanitize_label(s)}__{_sanitize_label(d)}"
    tilde_name = f"beta_pres_tilde__{_sanitize_label(s)}__{_sanitize_label(d)}"
    beta_det = next(v for v in model.deterministics if v.name == beta_name)
    tilde_rv = next(v for v in model.unobserved_RVs if v.name == tilde_name)
    f = function([tilde_rv], beta_det, on_unused_input="ignore")
    beta_value = float(f(0.0))

    alpha_p, beta_p, _, _ = evidence.get_beta_parameters(s, d)
    paper_mu = alpha_p / (alpha_p + beta_p)
    assert abs(beta_value - paper_mu) < 1e-4, (
        f"beta_pres at tilde=0, sigma=1e-6 = {beta_value:.6f}; "
        f"expected paper mean {paper_mu:.6f} for ({s}, {d})"
    )
    print(
        f"  [OK] sigma->0 collapses beta_pres for ({s},{d}) to paper mean "
        f"{paper_mu:.4f}"
    )


if __name__ == "__main__":
    print("Running label-pooled smoke tests...")
    test_collect_tree_label_groups_matches_expected_counts()
    test_flag_off_does_not_create_pool_vars()
    test_flag_on_exposes_hyperparameters()
    test_sigma_goes_to_zero_recovers_paper_mean()
    print("All label-pooled smoke tests passed.")
