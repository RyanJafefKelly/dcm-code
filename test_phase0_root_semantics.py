from __future__ import annotations

import math

import numpy as np
from scipy.special import expit, logit

from dcm_model import EvidenceProcessor, ModelConfig, MultiSystemDataProcessor, load_data
from dcm_model_exact_tree import MultiSystemExactTreeBuilder
from gwt_reference_recovery_analysis import ANCHORED_SYSTEM_CONFIGS, STANCE
from phase0_root_semantics import (
    PRIOR_A,
    PRIOR_B,
    beta_one_root_mixture_moments,
    beta_sd,
    deterministic_prefix,
    rho_from_log_sides,
)


def test_rho_logsum_matches_logit_bayes_factor_identity() -> None:
    pi = np.array([0.05, 0.2, 0.8])
    log_L0 = np.array([-10.0, -4.0, -2.0])
    log_B = np.array([-3.0, 0.5, 4.0])
    log_L1 = log_L0 + log_B

    rho = rho_from_log_sides(pi, log_L0, log_L1)
    expected = expit(logit(pi) + log_B)

    np.testing.assert_allclose(rho, expected, rtol=1e-12, atol=1e-12)
    assert np.all((rho >= 0.0) & (rho <= 1.0))


def test_pi_mixture_hits_one_root_update_sd_limits() -> None:
    prior_sd = beta_sd(PRIOR_A, PRIOR_B)
    _, absent_sd = beta_one_root_mixture_moments(0.0)
    _, present_sd = beta_one_root_mixture_moments(1.0)

    assert math.isclose(absent_sd / prior_sd, beta_sd(PRIOR_A, PRIOR_B + 1.0) / prior_sd)
    assert math.isclose(present_sd / prior_sd, 1.133893, rel_tol=1e-5)


def test_exact_tree_builder_exposes_root_semantic_deterministics() -> None:
    cfg = ModelConfig(
        INDICATOR_STATE_MODEL="three_state",
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        POOL_BETAS_BY_LABEL=True,
        BETA_ABS_BY_SUPPORT_DEMAND=True,
        LABEL_POOL_SIGMA=0.5,
        NUM_SAMPLES=5,
        NUM_TUNE=5,
        NUM_CHAINS=1,
    )
    systems = [s for s, _ in ANCHORED_SYSTEM_CONFIGS]
    stance_data = next(s for s in load_data(cfg) if s["name"] == STANCE)
    proc = MultiSystemDataProcessor(cfg)
    proc.process(stance_data, systems)
    builder = MultiSystemExactTreeBuilder(
        cfg, EvidenceProcessor(cfg), proc, list(ANCHORED_SYSTEM_CONFIGS)
    )
    model = builder.build_model(stance_data)
    names = {v.name for v in model.deterministics}

    for system in systems:
        prefix = deterministic_prefix(system, STANCE)
        assert f"{prefix}_log_L_root0" in names
        assert f"{prefix}_log_L_root1" in names
        assert f"{prefix}_log_B" in names
        assert f"{prefix}_rho" in names
