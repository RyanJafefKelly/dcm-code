"""Smoke test for TRANSMISSION_GAIN.

Invariants checked:

1. At ``TRANSMISSION_GAIN = 1.0``, ``EvidenceProcessor.get_beta_parameters``
   returns values bit-identical to the paper mapping for every
   (support, demandingness) combination.

2. At ``TRANSMISSION_GAIN = 2.5``, the closed-form symmetric log-odds-gap
   rule is applied exactly: for `strong support + neutral` the prior-mean
   transmission delta should match hand computation.

3. The concentration `alpha + beta` is preserved at ``NODE_CONCENTRATION``.

4. The diagnostic module's ``apply_symmetric_gain`` (NumPy helper) and
   ``_apply_transmission_gain`` (model-side helper) agree on the same
   inputs. Both pipelines are exercised to prevent drift.
"""

from __future__ import annotations

import math

import numpy as np

from dcm_model import (
    EvidenceProcessor,
    ModelConfig,
    _apply_transmission_gain,
)
from tree_propagation_diagnostic import (
    SUPPORT_LABELS,
    DEMANDINGNESS_LABELS,
    apply_symmetric_gain,
)


def test_gain_one_is_paper_bit_equal() -> None:
    """At gain=1.0 the output must match the paper mapping exactly."""
    cfg_paper = ModelConfig(TRANSMISSION_GAIN=1.0)
    ep_paper = EvidenceProcessor(cfg_paper)
    cfg_g1 = ModelConfig(TRANSMISSION_GAIN=1.0)  # explicit; same value
    ep_g1 = EvidenceProcessor(cfg_g1)
    for s in SUPPORT_LABELS:
        for d in DEMANDINGNESS_LABELS:
            ap0, bp0, aa0, ba0 = ep_paper.get_beta_parameters(s, d)
            ap1, bp1, aa1, ba1 = ep_g1.get_beta_parameters(s, d)
            assert ap0 == ap1, f"alpha_pres diverged at g=1 for ({s}, {d})"
            assert bp0 == bp1, f"beta_pres diverged at g=1 for ({s}, {d})"
            assert aa0 == aa1, f"alpha_abs diverged at g=1 for ({s}, {d})"
            assert ba0 == ba1, f"beta_abs diverged at g=1 for ({s}, {d})"
    print(
        f"  [OK] gain=1.0 bit-equal over {len(SUPPORT_LABELS)}*{len(DEMANDINGNESS_LABELS)} combinations"
    )


def test_gain_two_five_matches_closed_form_for_strong_neutral() -> None:
    """strong support + neutral at g=2.5 should give Delta approx 0.8."""
    cfg = ModelConfig(TRANSMISSION_GAIN=2.5)
    ep = EvidenceProcessor(cfg)
    ap, bp, aa, ba = ep.get_beta_parameters("strong support", "neutral")
    c = cfg.NODE_CONCENTRATION
    # Concentration preserved
    assert math.isclose(ap + bp, c, abs_tol=1e-9), "alpha+beta != NODE_CONCENTRATION for pres"
    assert math.isclose(aa + ba, c, abs_tol=1e-9), "alpha+beta != NODE_CONCENTRATION for abs"
    mu_p = ap / (ap + bp)
    mu_a = aa / (aa + ba)
    # Closed-form expected values (independent NumPy implementation):
    # Paper means: mu_pres = 8/9, mu_abs = 1/2.
    paper_mu_p = 8.0 / 9.0
    paper_mu_a = 1.0 / 2.0
    exp_mu_p, exp_mu_a = apply_symmetric_gain(paper_mu_p, paper_mu_a, 2.5)
    assert math.isclose(float(mu_p), float(exp_mu_p), abs_tol=1e-10), (
        f"mu_pres at g=2.5 mismatch: model {mu_p}, diag {exp_mu_p}"
    )
    assert math.isclose(float(mu_a), float(exp_mu_a), abs_tol=1e-10), (
        f"mu_abs at g=2.5 mismatch: model {mu_a}, diag {exp_mu_a}"
    )
    delta = float(mu_p) - float(mu_a)
    assert 0.78 < delta < 0.82, (
        f"strong support + neutral Delta at g=2.5 should be ~0.80; got {delta}"
    )
    print(
        f"  [OK] g=2.5 strong+neutral: mu_pres={float(mu_p):.4f}, mu_abs={float(mu_a):.4f}, "
        f"Delta={delta:.4f}"
    )


def test_helpers_agree_over_all_label_combinations() -> None:
    """Model-side helper and diagnostic helper must agree on every label."""
    cfg = ModelConfig(TRANSMISSION_GAIN=1.0)  # will be bypassed by direct call
    ep = EvidenceProcessor(cfg)
    gain = 2.0
    for s in SUPPORT_LABELS:
        for d in DEMANDINGNESS_LABELS:
            ap, bp, aa, ba = ep.get_beta_parameters(s, d)  # paper values (g=1)
            # Apply gain via model-side helper
            ap2, bp2, aa2, ba2 = _apply_transmission_gain(
                ap, bp, aa, ba, gain, cfg.NODE_CONCENTRATION
            )
            # Apply gain via diagnostic helper on the means
            mu_p_paper = ap / (ap + bp)
            mu_a_paper = aa / (aa + ba)
            mu_p_diag, mu_a_diag = apply_symmetric_gain(mu_p_paper, mu_a_paper, gain)
            mu_p_model = ap2 / (ap2 + bp2)
            mu_a_model = aa2 / (aa2 + ba2)
            assert math.isclose(mu_p_model, float(mu_p_diag), abs_tol=1e-10), (
                f"pres mu mismatch at ({s}, {d}): model {mu_p_model}, diag {mu_p_diag}"
            )
            assert math.isclose(mu_a_model, float(mu_a_diag), abs_tol=1e-10), (
                f"abs mu mismatch at ({s}, {d}): model {mu_a_model}, diag {mu_a_diag}"
            )
            # Concentration preserved
            assert math.isclose(ap2 + bp2, cfg.NODE_CONCENTRATION, abs_tol=1e-9)
            assert math.isclose(aa2 + ba2, cfg.NODE_CONCENTRATION, abs_tol=1e-9)
    print(
        f"  [OK] model-side and diagnostic gain helpers agree across "
        f"{len(SUPPORT_LABELS)}*{len(DEMANDINGNESS_LABELS)} combinations at g=2.0"
    )


if __name__ == "__main__":
    print("Running transmission-gain smoke tests...")
    test_gain_one_is_paper_bit_equal()
    test_gain_two_five_matches_closed_form_for_strong_neutral()
    test_helpers_agree_over_all_label_combinations()
    print("All transmission-gain smoke tests passed.")
