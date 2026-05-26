"""Ordinal observation model for the Digital Consciousness Model (DCM).

Replaces the binary-collapse observation layer with a constrained ordered probit
(rating-SDT) model for 7-point ordinal expert responses. All discrete latent
variables (stance, feature, subfeature presence) are marginalised analytically,
yielding a fully continuous model that NUTS can sample without Metropolis steps.

Current limitations
-------------------
- Likert-only core; legacy probability data uses a temporary adapter
- Shared discrimination parameter (a)
- Shared cutpoints (kappa)
- Expert-specific location shifts (b_e) with one anchored at zero

References
----------
- Betancourt, M. (2019). Ordinal Regression.
  https://betanalpha.github.io/assets/case_studies/ordinal_regression.html
- Rethink Priorities DCM specification (scheme 133).
"""

from __future__ import annotations

import os

import pytensor

try:
    pytensor.config.gcc__cxxflags = "-fbracket-depth=4096"
    pytensor.config.compiledir = f"/tmp/pytensor_cache_{os.getpid()}"
except Exception:
    pass  # already initialised (e.g. imported from notebook)

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration parameters for the Bayesian model."""

    # Evidential strength constants (preserved from original DCM)
    BASE: int = 3
    OVERWHELMING: int = 50
    STRONG: int = 8
    MODERATE: int = 3
    WEAK: float = 1.5

    # Sampling parameters
    NUM_SAMPLES: int = 3000
    NUM_TUNE: int = 500
    NUM_CHAINS: int = 4
    TARGET_ACCEPT: float = 0.9

    # Prior parameters (stance Beta prior)
    DEFAULT_ALPHA: int = 1
    DEFAULT_BETA: int = 5

    # Concentration for every internal tree node's Beta(alpha, beta).
    # alpha + beta is rescaled to this value; the ratio alpha:beta is
    # set by the (support, demandingness) mapping. Original DCM uses 10.
    NODE_CONCENTRATION: float = 10.0

    # Transmission-gain sensitivity. Symmetric log-odds-gap rescaling of
    # the paper's per-label Beta prior means: for each node with paper
    # means (mu_pres, mu_abs), let m = (logit mu_pres + logit mu_abs) / 2
    # and h = (logit mu_pres - logit mu_abs) / 2; under gain g the means
    # become mu_pres^(g) = sigmoid(m + g*h), mu_abs^(g) = sigmoid(m - g*h).
    # At g=1.0 this is bit-identical to the paper mapping. For g>1 the
    # edge's discriminativeness sharpens on both sides. Applied deterministically
    # inside EvidenceProcessor.get_beta_parameters; no PyMC RVs added.
    TRANSMISSION_GAIN: float = 1.0

    # Indicator latent-state model. Selects the leaf family without touching the rest
    # of the validated baseline. Members of the growing model library:
    #   "binary"      — z_j ∈ {0,1},   z_j ~ Bernoulli(q_j)        (original baseline)
    #   "three_state" — m_j ∈ {0,1,2}, m_j ~ Binomial(2, q_j);
    #                   emission centres at η ∈ {0, a/2, a};
    #                   z_j = m_j/2 gives expected_z = q_j
    #   "direct_q"    — no latent indicator state; emission centre is the
    #                   indicator-effective probability itself,
    #                   η_je = a · q̃_j + b_e. q̃_j = β_abs + q_j(β_pres - β_abs)
    #                   in the composite path (already supplied by the upstream
    #                   propagation) and = β_pres / β_abs (conditional on parent
    #                   binary state) in the exact-tree path.
    # Marginalisation is per-indicator for binary / three_state (m_j or z_j is
    # shared across the indicator's ratings), analytic in all cases, so NUTS
    # sees a fully continuous model.
    INDICATOR_STATE_MODEL: Literal["binary", "three_state", "direct_q"] = "binary"

    # Ordinal observation model
    USE_EXPERT_SHIFTS: bool = True  # If False, all experts share b=0
    # Prior scale on free expert location shifts b_free ~ Normal(0, sigma).
    # 2.0 is the original diffuse default; a tight value (e.g. 0.3) lets the
    # model express small expert calibration differences without absorbing
    # system-level signal when expert pools are non-overlapping.
    EXPERT_SHIFT_SIGMA: float = 2.0
    # Prior scale on shared discrimination a ~ HalfNormal(sigma). Controls
    # how far the latent signal shifts between indicator-absent and
    # indicator-present. Larger sigma lets the posterior push `a` higher if
    # the extreme-category data (e.g. Human 7s / ELIZA 1s) demand it.
    A_PRIOR_SIGMA: float = 2.0
    # Prior scale on shared ordered cutpoints kappa ~ Normal(0, sigma).
    # Larger sigma lets the outer cutpoints stretch further to accommodate
    # extreme-category observations.
    KAPPA_PRIOR_SIGMA: float = 2.0
    # Optional strongly regularised hierarchical expert-specific cutpoints.
    # When enabled, expert cutpoints vary through positive inter-cutpoint gaps
    # that are pooled hierarchically and then centered to mean zero per expert.
    # This allows spacing / shape heterogeneity without reintroducing a free
    # expert-specific location shift.
    USE_HIERARCHICAL_EXPERT_CUTPOINTS: bool = False
    HIER_KAPPA_GLOBAL_LOG_GAP_MU: float = -0.5
    HIER_KAPPA_GLOBAL_LOG_GAP_SIGMA: float = 0.35
    HIER_KAPPA_EXPERT_SCALE_SIGMA: float = 0.15
    # Hierarchical per-expert noise scale sigma_e. Partial-pooling pattern
    # analogous to USE_HIERARCHICAL_EXPERT_CUTPOINTS but for the scale of the
    # latent noise component. Under the sum-to-zero / geometric-mean-one
    # parameterisation:
    #   log sigma_e = tau_sigma * (u_e - mean(u)),
    #   u_e ~ Normal(0, 1),  tau_sigma ~ HalfNormal(EXPERT_SCALE_TAU_SIGMA)
    # This lets experts differ in sharpness while keeping the global probit
    # scale anchored (prevents sigma_e from trading off with a and kappa).
    # Single-system experts' sigma_e remains near 1 under partial pooling;
    # experts with cross-system coverage shift away from 1
    # when their rating behaviour supports it.
    # Cannot be combined with USE_HIERARCHICAL_EXPERT_CUTPOINTS.
    USE_EXPERT_SCALES: bool = False
    EXPERT_SCALE_TAU_SIGMA: float = 0.3
    # Label-level pooling on tree node transmission (complete-pooling-
    # within-label).  When True, each (support, demandingness) group gets
    # a single logit-Normal beta_pres parameter; each demandingness group
    # gets a single logit-Normal beta_abs parameter.  All tree nodes with
    # the same label share that group-level beta value.  Centred on the
    # paper's fixed prior means in logit space with scale
    # LABEL_POOL_SIGMA.  Non-centred parameterisation (a standard-Normal
    # tilde RV per group).  At paper-mean + tilde=0, recovers the paper
    # baseline exactly at that point; LABEL_POOL_SIGMA -> 0 collapses to
    # the paper baseline everywhere.
    #
    # The "pooling" here is complete pooling within each label: same-
    # label nodes are not allowed to have independent beta values.  This
    # matches the paper baseline semantics (same-label nodes have
    # identical Beta priors), but upgrades the fixed prior to a random
    # variable that data can move.  For groups with many nodes this is
    # genuine cross-node information sharing (all same-label node data
    # contributes to one label-level posterior).  For singleton groups it
    # is prior-softening on that single node.
    #
    # Partial-pooling (label mean + independent node-level deviations) was
    # implemented in an earlier draft but produced a pytensor graph too
    # large for the C++ backend to compile in reasonable time.
    POOL_BETAS_BY_LABEL: bool = False
    LABEL_POOL_SIGMA: float = 0.5
    # When POOL_BETAS_BY_LABEL is True, key beta_abs by (support,
    # demandingness) instead of demandingness only.  Used as a structural
    # diagnostic against the shared-beta_abs__neutral inheritance noted
    # in B.3a/b (the apparent "weak undermining + neutral" sign-flip
    # resolution being driven by the shared neutral-demandingness
    # absence baseline rather than group-specific evidence).
    #
    # Under this flag, each (support, demandingness) group gets its
    # OWN beta_abs RV (centred on the paper's d-only β_abs prior mean).
    # Singleton (s, d) groups will likely be weakly identified — that's
    # the diagnostic point: if β_abs__(weak_und, neutral) stays near
    # the prior 0.5 under the new fit, the singleton has no direct
    # evidence and the inheritance interpretation is confirmed.
    BETA_ABS_BY_SUPPORT_DEMAND: bool = False
    # Asymmetric prior overrides for the label-pool path.  When set, replace
    # paper_mu(s,d) with the override value for ALL pres / abs groups, keeping
    # the logit-Normal parameterisation.  Motivated by the 2026-05-07 finding
    # that the paper-centred prior implies a small δ = β_pres − β_abs at every
    # label and so transmits very little C signal down the tree (multiplicative
    # collapse via q_child = β_abs + parent_q · (β_pres − β_abs)).  Setting
    # β_pres → 1 and β_abs → 0 makes "this is a real evidencer" the prior
    # default, restoring transmission.  Sigma override (defaults to
    # LABEL_POOL_SIGMA) lets the override path be tighter than the data-pooled
    # baseline, since the override values already encode strong prior belief.
    BETA_PRES_OVERRIDE_MEAN: Optional[float] = None
    BETA_ABS_OVERRIDE_MEAN: Optional[float] = None
    BETA_OVERRIDE_SIGMA: Optional[float] = None
    # Soft reference-system anchors.  Per-system (alpha, beta) parameters for
    # a Beta prior on root C; overrides the hard anchor (c_fixed) from
    # system_configs for any system listed.  Used for anchor-defensibility
    # diagnostics: how far does Human / ELIZA posterior drift under
    # Beta(50, 1) / Beta(1, 50) relative to the hard 0.999 / 0.001?  None
    # (default) preserves the existing hard-anchor behaviour.
    SOFT_REFERENCE_ANCHORS: Optional[Dict[str, Tuple[float, float]]] = None
    # "Safe gain": logit-Normal tree betas centred at the gained means.
    # When TRANSMISSION_GAIN != 1.0 AND GAIN_LOGIT_NORMAL=True, replace
    # the per-node Beta(alpha, beta) priors with
    #   eta_n_pres ~ N(logit(mu_pres_gain), sigma^2),  beta_n_pres = sigmoid(eta_n_pres)
    # and similarly for beta_abs.  This avoids the Beta-boundary
    # singularities that cause NUTS divergences when the gained means
    # approach 0 or 1.  Node-level (independent per node; no label
    # pooling).  Not combinable with POOL_BETAS_BY_LABEL (semantics of
    # combined pool+gain not settled — see plan's out-of-scope notes).
    GAIN_LOGIT_NORMAL: bool = False
    GAIN_LOGIT_NORMAL_SIGMA: float = 0.5
    N_CATEGORIES: int = 7
    # Bin edges for legacy probability -> ordinal conversion.
    # Category k is assigned when bins[k-1] <= p < bins[k].
    ORDINAL_BINS: Tuple[float, ...] = (0.05, 0.20, 0.40, 0.60, 0.80, 0.95)

    # Default targets
    TARGET_STANCE: str = "Global Workspace Theory"
    TARGET_SYSTEM: str = "2024 Leading Chat LLMs"

    # Data source
    DATA_CACHE_PATH: str = "data_cache.json"

    def __post_init__(self) -> None:
        """Mutual-exclusion guards for tree-prior intervention flags."""
        if self.POOL_BETAS_BY_LABEL and self.TRANSMISSION_GAIN != 1.0:
            raise ValueError(
                "POOL_BETAS_BY_LABEL and TRANSMISSION_GAIN != 1.0 cannot both "
                "be enabled.  They are orthogonal tree-prior interventions — "
                "hierarchical pooling lets data calibrate label transmission; "
                "transmission-gain hand-sets a global sharper semantic prior. "
                "Run them as separate fits and compare in the library."
            )
        if self.POOL_BETAS_BY_LABEL and self.GAIN_LOGIT_NORMAL:
            raise ValueError(
                "POOL_BETAS_BY_LABEL and GAIN_LOGIT_NORMAL cannot both be "
                "enabled.  The symmetric gain transform makes mu_abs depend "
                "on support as well as demandingness, which conflicts with "
                "the pooling branch's demandingness-only abs grouping.  "
                "A combined pool+gain model needs a semantic choice that "
                "has not been made yet (see plan)."
            )
        if self.BETA_ABS_BY_SUPPORT_DEMAND and not self.POOL_BETAS_BY_LABEL:
            raise ValueError(
                "BETA_ABS_BY_SUPPORT_DEMAND requires POOL_BETAS_BY_LABEL=True. "
                "It is a refinement of the pooling branch's abs grouping, not "
                "a standalone parameterisation."
            )
        for name, val in (
            ("BETA_PRES_OVERRIDE_MEAN", self.BETA_PRES_OVERRIDE_MEAN),
            ("BETA_ABS_OVERRIDE_MEAN", self.BETA_ABS_OVERRIDE_MEAN),
        ):
            if val is not None:
                if not self.POOL_BETAS_BY_LABEL:
                    raise ValueError(
                        f"{name} requires POOL_BETAS_BY_LABEL=True; the "
                        "override is implemented in the label-pool path only."
                    )
                if not (0.0 < float(val) < 1.0):
                    raise ValueError(
                        f"{name} must be in the open interval (0, 1); got {val!r}."
                    )
        if self.BETA_OVERRIDE_SIGMA is not None and float(self.BETA_OVERRIDE_SIGMA) <= 0.0:
            raise ValueError(
                f"BETA_OVERRIDE_SIGMA must be positive; got {self.BETA_OVERRIDE_SIGMA!r}."
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def node_key(ancestor_path: Tuple[str, ...], name: str) -> str:
    """Deterministic path-based key for a node in the DCM tree.

    Concatenates the ancestor path with the current node name using ' > '
    as delimiter, giving a unique identifier even when names repeat
    in different branches (e.g. 'Stable Personality' appears twice in GWT).
    """
    return " > ".join(ancestor_path + (name,))


def legacy_probability_to_ordinal(p: float, bins: Tuple[float, ...]) -> int:
    """Convert a [0, 1] probability to a 0-indexed ordinal category.

    TEMPORARY ADAPTER for legacy probability-valued observations.
    The core model expects integer ordinal data (0 .. K-1).

    Bin edges define the boundaries between adjacent categories:
      p < bins[0]              -> 0
      bins[0] <= p < bins[1]   -> 1
      ...
      p >= bins[-1]            -> len(bins)

    Parameters
    ----------
    p : float in [0, 1]
    bins : tuple of floats, strictly increasing, defining K-1 boundaries
           for K+1 categories (though we only use K = len(bins)+1 = 7 by default).
    """
    for i, edge in enumerate(bins):
        if p < edge:
            return i
    return len(bins)


def centered_cutpoints_from_gaps_np(gaps: np.ndarray) -> np.ndarray:
    """Construct ordered cutpoints from positive gaps and center to mean zero."""
    raw = np.concatenate([np.zeros(1), np.cumsum(gaps)])
    return raw - raw.mean()


def pt_centered_cutpoints_from_positive_gaps(
    gaps: pt.TensorVariable,
) -> pt.TensorVariable:
    """PyTensor version of centered cutpoint reconstruction from positive gaps."""
    if gaps.ndim == 1:
        raw = pt.concatenate([pt.zeros(1, dtype=gaps.dtype), pt.cumsum(gaps)])
        return raw - pt.mean(raw)
    if gaps.ndim == 2:
        zeros = pt.zeros((gaps.shape[0], 1), dtype=gaps.dtype)
        raw = pt.concatenate([zeros, pt.cumsum(gaps, axis=1)], axis=1)
        return raw - pt.mean(raw, axis=1, keepdims=True)
    raise ValueError("gaps must be a vector or matrix")


# ---------------------------------------------------------------------------
# Ordinal log-likelihood (PyTensor graph ops)
# ---------------------------------------------------------------------------


def pt_ordinal_logp(
    ratings: np.ndarray,
    expert_idx: np.ndarray,
    kappa: pt.TensorVariable,
    b: pt.TensorVariable,
    eta_shift: pt.TensorVariable,
    sigma_by_expert: Optional[pt.TensorVariable] = None,
) -> pt.TensorVariable:
    """Vectorised log P(ratings | params, eta_shift) under ordered probit.

    Implements the cumulative-normal parameterisation:
        P(r = k) = Phi((kappa_k - eta) / sigma_e) - Phi((kappa_{k-1} - eta) / sigma_e)
    with eta = b[expert] + eta_shift  (eta_shift = 0 for z=0, a for z=1),
    and sigma_e = sigma_by_expert[expert] if provided, else 1 (shared scale).

    Parameters
    ----------
    ratings : (N,) int array -- observed ordinal categories, 0-indexed.
    expert_idx : (N,) int array -- expert index for each observation.
    kappa : (K-1,) pytensor -- ordered cutpoints.
    b : (E,) pytensor -- expert location shifts.
    eta_shift : scalar pytensor -- additional shift (0 or a).
    sigma_by_expert : (E,) pytensor, optional -- per-expert noise scale.

    Returns
    -------
    Scalar pytensor: sum of log-probabilities across all observations.
    """
    n_obs = ratings.shape[0]
    eta = b[expert_idx] + eta_shift  # (N,)
    c_minus_eta = kappa[None, :] - eta[:, None]  # (N, K-1)
    if sigma_by_expert is not None:
        sigma_obs = sigma_by_expert[expert_idx][:, None]  # (N, 1)
        c_minus_eta = c_minus_eta / sigma_obs
    cum_probs = pt.erfc(-c_minus_eta / pt.sqrt(2.0)) / 2.0  # Phi((c - eta)/sigma)
    zeros = pt.zeros((cum_probs.shape[0], 1))
    ones = pt.ones((cum_probs.shape[0], 1))
    cum_full = pt.concatenate([zeros, cum_probs, ones], axis=1)  # (N, K+1)
    cat_probs = cum_full[:, 1:] - cum_full[:, :-1]  # (N, K)
    cat_probs = pt.clip(cat_probs, 1e-12, 1.0)
    log_p = pt.log(cat_probs[pt.arange(n_obs), ratings])  # (N,)
    return pt.sum(log_p)


def pt_ordinal_logp_expert_kappa(
    ratings: np.ndarray,
    expert_idx: np.ndarray,
    kappa_by_expert: pt.TensorVariable,
    eta_shift: pt.TensorVariable,
    sigma_by_expert: Optional[pt.TensorVariable] = None,
) -> pt.TensorVariable:
    """Ordered-probit logp with expert-specific cutpoints and no free location.

    Optionally accepts per-expert sigma_by_expert; note that the hierarchical
    expert-cutpoint branch and expert-scales branch are mutually exclusive at
    the config level (see build_ordinal_observation_layer), so sigma_by_expert
    is expected to be None here in practice.
    """
    n_obs = ratings.shape[0]
    kappa_obs = kappa_by_expert[expert_idx]  # (N, K-1)
    c_minus_eta = kappa_obs - eta_shift
    if sigma_by_expert is not None:
        sigma_obs = sigma_by_expert[expert_idx][:, None]  # (N, 1)
        c_minus_eta = c_minus_eta / sigma_obs
    cum_probs = pt.erfc(-c_minus_eta / pt.sqrt(2.0)) / 2.0
    zeros = pt.zeros((cum_probs.shape[0], 1), dtype=cum_probs.dtype)
    ones = pt.ones((cum_probs.shape[0], 1), dtype=cum_probs.dtype)
    cum_full = pt.concatenate([zeros, cum_probs, ones], axis=1)
    cat_probs = cum_full[:, 1:] - cum_full[:, :-1]
    cat_probs = pt.clip(cat_probs, 1e-12, 1.0)
    log_p = pt.log(cat_probs[pt.arange(n_obs), ratings])
    return pt.sum(log_p)


def pt_indicator_logps(
    ratings: np.ndarray,
    expert_idx: np.ndarray,
    a: pt.TensorVariable,
    kappa: pt.TensorVariable,
    b: pt.TensorVariable,
    kappa_by_expert: Optional[pt.TensorVariable] = None,
    sigma_by_expert: Optional[pt.TensorVariable] = None,
) -> Tuple[pt.TensorVariable, pt.TensorVariable]:
    """Return log-likelihood terms for z=0 and z=1 under the active obs layer."""
    if kappa_by_expert is not None:
        return (
            pt_ordinal_logp_expert_kappa(
                ratings, expert_idx, kappa_by_expert, pt.constant(0.0),
                sigma_by_expert=sigma_by_expert,
            ),
            pt_ordinal_logp_expert_kappa(
                ratings, expert_idx, kappa_by_expert, a,
                sigma_by_expert=sigma_by_expert,
            ),
        )
    return (
        pt_ordinal_logp(
            ratings, expert_idx, kappa, b, pt.constant(0.0),
            sigma_by_expert=sigma_by_expert,
        ),
        pt_ordinal_logp(
            ratings, expert_idx, kappa, b, a,
            sigma_by_expert=sigma_by_expert,
        ),
    )


# ---------------------------------------------------------------------------
# Three-state latent indicator helpers
# ---------------------------------------------------------------------------
#
# Under the three-state leaf model we take m_j ~ Binomial(2, q_j) with
# z_j = m_j/2 in {0, 0.5, 1} and ordered-probit emission centres at
# eta in {0, a/2, a}. The latent m_j is shared across all ratings for a
# single indicator, so marginalisation is per-indicator:
#
#     log P(r_{1j}, ..., r_{n_j j} | q_j, theta) =
#         logsumexp_{m in {0,1,2}} [ log w_m(q_j) + sum_i log P_OP(r_ij | eta_m, theta) ]
#
# with log w_0 = 2 log(1-q), log w_1 = log 2 + log q + log(1-q), log w_2 = 2 log q.
# A product of per-rating mixtures would incorrectly let each rating within the
# same indicator have its own latent m; hence the per-indicator aggregation.


def three_state_log_weights(
    q_j: pt.TensorVariable,
) -> Tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]:
    """Log prior weights for (m=0, m=1, m=2) under m_j ~ Binomial(2, q_j)."""
    log_1mq = pt.log(1.0 - q_j + 1e-12)
    log_q = pt.log(q_j + 1e-12)
    return 2.0 * log_1mq, pt.log(2.0) + log_q + log_1mq, 2.0 * log_q


def pt_three_state_ll_terms(
    ratings: np.ndarray,
    expert_idx: np.ndarray,
    a: pt.TensorVariable,
    kappa: pt.TensorVariable,
    b: pt.TensorVariable,
    kappa_by_expert: Optional[pt.TensorVariable] = None,
    sigma_by_expert: Optional[pt.TensorVariable] = None,
) -> Tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]:
    """Aggregated per-indicator log-likelihoods for m in {0, 1, 2}.

    Each returned component is ``sum_i log P_OP(r_ij | eta=eta_m, theta)``
    for emission centres ``eta_0 = 0``, ``eta_1 = a/2``, ``eta_2 = a``.
    The marginalisation over m_j is then performed by the caller using
    ``three_state_log_weights(q_j)`` and pairwise ``pt.logaddexp``.

    Each per-rating emission uses the rater's own sigma via
    ``sigma_by_expert`` (when provided), so the per-indicator
    aggregation ``sum_i log P_OP`` correctly accounts for
    expert-heterogeneous sharpness without breaking the shared-latent-m
    invariant.
    """
    if kappa_by_expert is not None:
        return (
            pt_ordinal_logp_expert_kappa(
                ratings, expert_idx, kappa_by_expert, pt.constant(0.0),
                sigma_by_expert=sigma_by_expert,
            ),
            pt_ordinal_logp_expert_kappa(
                ratings, expert_idx, kappa_by_expert, a * 0.5,
                sigma_by_expert=sigma_by_expert,
            ),
            pt_ordinal_logp_expert_kappa(
                ratings, expert_idx, kappa_by_expert, a,
                sigma_by_expert=sigma_by_expert,
            ),
        )
    return (
        pt_ordinal_logp(
            ratings, expert_idx, kappa, b, pt.constant(0.0),
            sigma_by_expert=sigma_by_expert,
        ),
        pt_ordinal_logp(
            ratings, expert_idx, kappa, b, a * 0.5,
            sigma_by_expert=sigma_by_expert,
        ),
        pt_ordinal_logp(
            ratings, expert_idx, kappa, b, a,
            sigma_by_expert=sigma_by_expert,
        ),
    )


def pt_direct_q_ll(
    ratings: np.ndarray,
    expert_idx: np.ndarray,
    q_j_eff: pt.TensorVariable,
    a: pt.TensorVariable,
    kappa: pt.TensorVariable,
    b: pt.TensorVariable,
    kappa_by_expert: Optional[pt.TensorVariable] = None,
    sigma_by_expert: Optional[pt.TensorVariable] = None,
) -> pt.TensorVariable:
    """Direct-q leaf log-likelihood: η_je = a · q_j_eff + b_e.

    No latent indicator state. ``q_j_eff`` is the **indicator-effective**
    probability $\\tilde q_j$ — caller's responsibility to compute as
    $\\beta^{\\rm abs}_j + q_j(\\beta^{\\rm pres}_j - \\beta^{\\rm abs}_j)$ in
    the composite path, or as the indicator's own β (β_pres or β_abs) when
    conditioning on the parent's binary state in the exact-tree path.
    Returns the scalar sum of log P_OP across this indicator's ratings.
    """
    eta_shift = a * q_j_eff
    if kappa_by_expert is not None:
        return pt_ordinal_logp_expert_kappa(
            ratings, expert_idx, kappa_by_expert, eta_shift,
            sigma_by_expert=sigma_by_expert,
        )
    return pt_ordinal_logp(
        ratings, expert_idx, kappa, b, eta_shift,
        sigma_by_expert=sigma_by_expert,
    )


def add_indicator_marginal_likelihood(
    name: str,
    potential_name: str,
    q_j: pt.TensorVariable,
    ratings: np.ndarray,
    expert_indices: np.ndarray,
    a: pt.TensorVariable,
    kappa: pt.TensorVariable,
    b: pt.TensorVariable,
    kappa_by_expert: Optional[pt.TensorVariable],
    state_model: str,
    sigma_by_expert: Optional[pt.TensorVariable] = None,
) -> None:
    """Add the marginalised ordinal likelihood + deterministics for one indicator.

    Dispatches on ``state_model``:
      - "binary":      z_j ~ Bernoulli(q_j). Exposes ``{name}_pz1``.
      - "three_state": m_j ~ Binomial(2, q_j). Exposes ``{name}_p_m0``,
        ``{name}_p_m1``, ``{name}_p_m2``, and ``{name}_expected_z``.

    ``ratings`` / ``expert_indices`` must contain every rating for the
    single indicator being added (per-indicator marginalisation).

    ``sigma_by_expert`` routes through to the ordered-probit helpers so
    each rating is evaluated under its rater's noise scale; per-indicator
    marginalisation (sharing m_j across all ratings of the indicator) is
    preserved.
    """
    if state_model == "binary":
        ll_z0, ll_z1 = pt_indicator_logps(
            ratings, expert_indices, a, kappa, b, kappa_by_expert,
            sigma_by_expert=sigma_by_expert,
        )
        log_mix = pt.logaddexp(
            pt.log(1.0 - q_j + 1e-12) + ll_z0,
            pt.log(q_j + 1e-12) + ll_z1,
        )
        pm.Potential(potential_name, log_mix)
        logit_q = pt.log(q_j + 1e-12) - pt.log(1.0 - q_j + 1e-12)
        pz1 = pt.sigmoid(logit_q + ll_z1 - ll_z0)
        pm.Deterministic(f"{name}_pz1", pz1)
        return
    if state_model == "three_state":
        ll_0, ll_half, ll_1 = pt_three_state_ll_terms(
            ratings, expert_indices, a, kappa, b, kappa_by_expert,
            sigma_by_expert=sigma_by_expert,
        )
        log_w0, log_wmid, log_w1 = three_state_log_weights(q_j)
        log_mix = pt.logaddexp(
            pt.logaddexp(log_w0 + ll_0, log_wmid + ll_half),
            log_w1 + ll_1,
        )
        pm.Potential(potential_name, log_mix)
        p_m0 = pt.exp(log_w0 + ll_0 - log_mix)
        p_m1 = pt.exp(log_wmid + ll_half - log_mix)
        p_m2 = pt.exp(log_w1 + ll_1 - log_mix)
        pm.Deterministic(f"{name}_p_m0", p_m0)
        pm.Deterministic(f"{name}_p_m1", p_m1)
        pm.Deterministic(f"{name}_p_m2", p_m2)
        pm.Deterministic(f"{name}_expected_z", 0.5 * p_m1 + p_m2)
        return
    if state_model == "direct_q":
        ll = pt_direct_q_ll(
            ratings, expert_indices, q_j, a, kappa, b,
            kappa_by_expert=kappa_by_expert,
            sigma_by_expert=sigma_by_expert,
        )
        pm.Potential(potential_name, ll)
        pm.Deterministic(f"{name}_q_eff", q_j)
        return
    raise ValueError(
        f"Unknown INDICATOR_STATE_MODEL: {state_model!r}. "
        f"Expected 'binary', 'three_state', or 'direct_q'."
    )


def add_no_data_prior_deterministics(
    name: str,
    q_j: pt.TensorVariable,
    state_model: str,
) -> None:
    """Expose prior-only indicator deterministics when there are no ratings.

    Keeps downstream consumers (PPC helpers, summaries) free of
    data-availability branching: they can always read ``{name}_pz1`` (binary)
    or ``{name}_p_m*`` / ``{name}_expected_z`` (three-state).
    """
    if state_model == "binary":
        pm.Deterministic(f"{name}_pz1", q_j)
        return
    if state_model == "three_state":
        p_m0_prior = (1.0 - q_j) ** 2
        p_m1_prior = 2.0 * q_j * (1.0 - q_j)
        p_m2_prior = q_j ** 2
        pm.Deterministic(f"{name}_p_m0", p_m0_prior)
        pm.Deterministic(f"{name}_p_m1", p_m1_prior)
        pm.Deterministic(f"{name}_p_m2", p_m2_prior)
        pm.Deterministic(f"{name}_expected_z", 0.5 * p_m1_prior + p_m2_prior)
        return
    if state_model == "direct_q":
        pm.Deterministic(f"{name}_q_eff", q_j)
        return
    raise ValueError(
        f"Unknown INDICATOR_STATE_MODEL: {state_model!r}. "
        f"Expected 'binary', 'three_state', or 'direct_q'."
    )


def build_ordinal_observation_layer(
    config: ModelConfig,
    n_experts: int,
    K: int,
) -> Tuple[
    pt.TensorVariable,
    pt.TensorVariable,
    pt.TensorVariable,
    Optional[pt.TensorVariable],
    Optional[pt.TensorVariable],
]:
    """Build the shared ordinal observation layer for single- or multi-system fits.

    Returns (a, b, kappa, kappa_by_expert, sigma_by_expert) where
    ``kappa_by_expert`` is non-None under ``USE_HIERARCHICAL_EXPERT_CUTPOINTS``
    and ``sigma_by_expert`` is non-None under ``USE_EXPERT_SCALES``. These two
    flags are mutually exclusive. ``USE_EXPERT_SCALES`` is also mutually
    exclusive with ``USE_EXPERT_SHIFTS`` (combining shift + scale under the
    current non-overlapping expert pools would worsen attribution and was
    explicitly scoped out of this branch).
    """
    if config.USE_HIERARCHICAL_EXPERT_CUTPOINTS and config.USE_EXPERT_SHIFTS:
        raise ValueError(
            "Hierarchical expert cutpoints and expert shifts cannot both be enabled"
        )
    if config.USE_EXPERT_SCALES and config.USE_HIERARCHICAL_EXPERT_CUTPOINTS:
        raise ValueError(
            "USE_EXPERT_SCALES and USE_HIERARCHICAL_EXPERT_CUTPOINTS cannot both "
            "be enabled: an expert-specific noise scale plus expert-specific "
            "cutpoint spacings is over-parameterised for the current data."
        )
    if config.USE_EXPERT_SCALES and config.USE_EXPERT_SHIFTS:
        raise ValueError(
            "USE_EXPERT_SCALES and USE_EXPERT_SHIFTS cannot both be enabled in "
            "this branch: expert shift + scale combined under non-overlapping "
            "expert pools worsens attribution. See plan 'Out of scope'."
        )

    a = pm.HalfNormal("a", sigma=config.A_PRIOR_SIGMA)

    # --- Optional hierarchical per-expert sigma_e ---
    sigma_by_expert: Optional[pt.TensorVariable] = None
    if config.USE_EXPERT_SCALES:
        if n_experts < 2:
            raise ValueError(
                "USE_EXPERT_SCALES requires at least 2 experts to be meaningful"
            )
        tau_sigma = pm.HalfNormal(
            "tau_sigma",
            sigma=config.EXPERT_SCALE_TAU_SIGMA,
            initval=config.EXPERT_SCALE_TAU_SIGMA / 2,
        )
        u_raw = pm.Normal(
            "sigma_u",
            mu=0.0,
            sigma=1.0,
            shape=n_experts,
            initval=np.zeros(n_experts),
        )
        # Sum-to-zero / geometric-mean-one via centring
        u_centered = u_raw - pt.mean(u_raw)
        sigma_by_expert = pm.Deterministic(
            "sigma_by_expert",
            pt.exp(tau_sigma * u_centered),
        )

    if config.USE_HIERARCHICAL_EXPERT_CUTPOINTS:
        n_gaps = K - 2
        if n_gaps < 1:
            raise ValueError(
                "Hierarchical cutpoints require at least 3 ordinal categories"
            )
        init_gaps = np.full(n_gaps, 0.6)
        log_gap_loc = pm.Normal(
            "kappa_log_gap_loc",
            mu=config.HIER_KAPPA_GLOBAL_LOG_GAP_MU,
            sigma=config.HIER_KAPPA_GLOBAL_LOG_GAP_SIGMA,
            shape=n_gaps,
            initval=np.log(init_gaps),
        )
        gap_scale = pm.HalfNormal(
            "kappa_expert_gap_scale",
            sigma=config.HIER_KAPPA_EXPERT_SCALE_SIGMA,
            initval=max(config.HIER_KAPPA_EXPERT_SCALE_SIGMA / 2, 1e-3),
        )
        gap_offset = pm.Normal(
            "kappa_expert_gap_offset",
            mu=0.0,
            sigma=1.0,
            shape=(n_experts, n_gaps),
            initval=np.zeros((n_experts, n_gaps)),
        )
        pop_gaps = pt.exp(log_gap_loc)
        expert_gaps = pt.exp(log_gap_loc[None, :] + gap_scale * gap_offset)
        kappa = pm.Deterministic(
            "kappa",
            pt_centered_cutpoints_from_positive_gaps(pop_gaps),
        )
        kappa_by_expert = pm.Deterministic(
            "kappa_by_expert",
            pt_centered_cutpoints_from_positive_gaps(expert_gaps),
        )
        b = pt.zeros(n_experts)
        return a, b, kappa, kappa_by_expert, sigma_by_expert

    if config.USE_EXPERT_SHIFTS and n_experts > 1:
        b_free = pm.Normal(
            "b_free",
            mu=0.0,
            sigma=config.EXPERT_SHIFT_SIGMA,
            shape=n_experts - 1,
        )
        b = pt.concatenate([pt.zeros(1), b_free])
    else:
        b = pt.zeros(n_experts)

    kappa = pm.Normal(
        "kappa",
        mu=0.0,
        sigma=config.KAPPA_PRIOR_SIGMA,
        shape=K - 1,
        transform=pm.distributions.transforms.ordered,
        initval=np.linspace(-1.5, 1.5, K - 1),
    )
    return a, b, kappa, None, sigma_by_expert


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------


class OrdinalDataProcessor:
    """Preprocesses DCM tree data into ordinal observations.

    Core interface is ordinal-native (0 .. K-1 integer categories).
    Legacy probability-valued data is converted via an explicit adapter
    whose bin edges live in ``ModelConfig.ORDINAL_BINS``.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        # {node_key: [(expert_idx, ordinal_rating), ...]}
        self.observations: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.expert_names: List[str] = []
        self.expert_to_idx: Dict[str, int] = {}
        self.anchor_expert: Optional[str] = None

    def process(self, stance_data: Dict, system: str) -> "OrdinalDataProcessor":
        """Process a stance tree for a given system.  Returns self."""
        self.logger.info(f"Processing observations for system: {system}")
        self.observations.clear()

        # --- First pass: count valid observations per expert ---
        expert_counts: Dict[str, int] = defaultdict(int)
        self._count_expert_observations(stance_data, system, expert_counts)
        if not expert_counts:
            self.logger.warning("No valid observations found for this system")
            return self

        # Anchor expert = most observations (deterministic tie-break: alphabetical)
        self.anchor_expert = max(
            sorted(expert_counts.keys()),
            key=lambda e: expert_counts[e],
        )
        self.logger.info(
            f"Anchor expert: {self.anchor_expert} "
            f"({expert_counts[self.anchor_expert]} observations)"
        )

        # Build canonical expert index: anchor at 0, rest alphabetical
        other_experts = sorted(e for e in expert_counts if e != self.anchor_expert)
        self.expert_names = [self.anchor_expert] + other_experts
        self.expert_to_idx = {name: i for i, name in enumerate(self.expert_names)}
        self.logger.info(f"Expert mapping: {self.expert_to_idx}")

        # --- Second pass: collect ordinal observations keyed by node path ---
        self._collect_observations(stance_data, system, ancestor_path=())

        n_obs = sum(len(v) for v in self.observations.values())
        n_ind = len(self.observations)
        self.logger.info(f"Collected {n_obs} observations across {n_ind} indicators")
        return self

    # -- internal helpers --------------------------------------------------

    def _is_missing(self, val: Any) -> bool:
        """Return True if the observation value should be treated as missing."""
        if val is None:
            return True
        s = str(val).strip().lower()
        return s in ("-1", "-1.0", "none", "", "unsure", "not tested")

    def _count_expert_observations(
        self, node: Dict, system: str, counts: Dict[str, int]
    ) -> None:
        if node.get("type", "").lower() == "indicator":
            obs = node.get("observations", {}).get(system)
            if obs:
                for i, val in enumerate(obs["values"]):
                    if not self._is_missing(val):
                        counts[obs["names"][i]] += 1
        for child in node.get("evidencers", []):
            self._count_expert_observations(child, system, counts)

    def _collect_observations(
        self,
        node: Dict,
        system: str,
        ancestor_path: Tuple[str, ...],
    ) -> None:
        current_path = ancestor_path + (node["name"],)

        if node.get("type", "").lower() == "indicator":
            obs = node.get("observations", {}).get(system)
            if obs:
                key = node_key(ancestor_path, node["name"])
                for i, val in enumerate(obs["values"]):
                    if self._is_missing(val):
                        continue
                    expert = obs["names"][i]
                    ordinal = legacy_probability_to_ordinal(
                        float(val), self.config.ORDINAL_BINS
                    )
                    self.observations[key].append((self.expert_to_idx[expert], ordinal))

        for child in node.get("evidencers", []):
            self._collect_observations(child, system, ancestor_path=current_path)


# ---------------------------------------------------------------------------
# Evidence processor (support / demandingness -> Beta parameters)
# ---------------------------------------------------------------------------


def _apply_transmission_gain(
    alpha_p: float,
    beta_p: float,
    alpha_a: float,
    beta_a: float,
    gain: float,
    concentration: float,
) -> Tuple[float, float, float, float]:
    """Symmetric log-odds-gap rescaling of Beta prior means.

    Given paper Beta parameters (alpha_s, beta_s) for s in {pres, abs}
    (with alpha_s + beta_s == concentration), compute means mu_s, recentre
    each mu_s in logit space around the midpoint of (logit mu_pres,
    logit mu_abs), and rescale the signed offset by ``gain``.  Returns
    new (alpha_p, beta_p, alpha_a, beta_a) with the same concentration.
    At gain == 1.0 the output is numerically identical to the input.
    """
    eps = 1e-9
    mu_p = alpha_p / (alpha_p + beta_p)
    mu_a = alpha_a / (alpha_a + beta_a)
    mu_p = min(max(mu_p, eps), 1.0 - eps)
    mu_a = min(max(mu_a, eps), 1.0 - eps)
    logit_p = np.log(mu_p / (1.0 - mu_p))
    logit_a = np.log(mu_a / (1.0 - mu_a))
    m = 0.5 * (logit_p + logit_a)
    h = 0.5 * (logit_p - logit_a)
    new_logit_p = m + gain * h
    new_logit_a = m - gain * h
    new_mu_p = 1.0 / (1.0 + np.exp(-new_logit_p))
    new_mu_a = 1.0 / (1.0 + np.exp(-new_logit_a))
    c = concentration
    return (
        float(c * new_mu_p),
        float(c * (1.0 - new_mu_p)),
        float(c * new_mu_a),
        float(c * (1.0 - new_mu_a)),
    )


class EvidenceProcessor:
    """Maps support/demandingness labels to Beta distribution parameters.

    Preserved from the original DCM implementation.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_beta_parameters(
        self, support: str, demandingness: str
    ) -> Tuple[float, float, float, float]:
        """Beta parameters for parent-present and parent-absent states.

        At the default ``TRANSMISSION_GAIN = 1.0`` the output is
        bit-identical to the paper mapping.  For gain != 1 the prior means
        are rescaled in logit space via the symmetric log-odds-gap rule
        (see ``_apply_transmission_gain``).  Concentration ``alpha + beta``
        is preserved at ``NODE_CONCENTRATION``.

        Returns
        -------
        (alpha_present, beta_present, alpha_absent, beta_absent)
        """
        absence_alpha, absence_beta = self._get_demandingness_parameters(demandingness)
        support_factor = self._get_support_factor(support, demandingness)

        presence_alpha = int(absence_alpha * support_factor[0])
        presence_beta = int(absence_beta * support_factor[1])

        c = self.config.NODE_CONCENTRATION
        alpha_p = presence_alpha * c / (presence_alpha + presence_beta)
        beta_p = presence_beta * c / (presence_alpha + presence_beta)
        alpha_a = absence_alpha * c / (absence_alpha + absence_beta)
        beta_a = absence_beta * c / (absence_alpha + absence_beta)

        gain = self.config.TRANSMISSION_GAIN
        if gain != 1.0:
            alpha_p, beta_p, alpha_a, beta_a = _apply_transmission_gain(
                alpha_p, beta_p, alpha_a, beta_a, gain, c
            )
        return alpha_p, beta_p, alpha_a, beta_a

    def get_gained_means(
        self, support: str, demandingness: str
    ) -> Tuple[float, float]:
        """Return (mu_pres, mu_abs) after TRANSMISSION_GAIN is applied.

        At gain=1.0 these are the paper's prior means.  At gain != 1 they
        are the symmetric log-odds-gap-rescaled means.  Useful for the
        safe-gain (logit-Normal tree betas) path, which centres a
        logit-Normal prior on logit(mu_gain) rather than drawing a Beta
        with potentially near-boundary shape parameters.
        """
        alpha_p, beta_p, alpha_a, beta_a = self.get_beta_parameters(support, demandingness)
        return alpha_p / (alpha_p + beta_p), alpha_a / (alpha_a + beta_a)

    def _get_demandingness_parameters(self, demandingness: str) -> Tuple[int, int]:
        base = self.config.BASE
        demandingness_map = {
            "overwhelmingly demanding": (base, int(base * self.config.OVERWHELMING)),
            "strongly demanding": (base, int(base * self.config.STRONG)),
            "moderately demanding": (base, int(base * self.config.MODERATE)),
            "weakly demanding": (base, int(base * self.config.WEAK)),
            "neutral": (base, base),
            "weakly undemanding": (int(base * self.config.WEAK), base),
            "moderately undemanding": (int(base * self.config.MODERATE), base),
            "strongly undemanding": (int(base * self.config.STRONG), base),
            "overwhelmingly undemanding": (
                int(base * self.config.OVERWHELMING),
                base,
            ),
        }
        if demandingness not in demandingness_map:
            raise ValueError(f"Unrecognised demandingness: {demandingness!r}")
        return demandingness_map[demandingness]

    def _get_support_factor(
        self, support: str, demandingness: str
    ) -> Tuple[float, float]:
        demandingness_factor = {
            "overwhelmingly demanding": self.config.STRONG,
            "strongly demanding": self.config.MODERATE,
            "moderately demanding": self.config.WEAK,
            "weakly demanding": 1,
        }.get(demandingness, 1)

        support_map = {
            "overwhelming support": (
                self.config.OVERWHELMING * demandingness_factor,
                1,
            ),
            "strong support": (
                self.config.STRONG * demandingness_factor,
                1,
            ),
            "moderate support": (
                self.config.MODERATE * demandingness_factor,
                1,
            ),
            "weak support": (
                self.config.WEAK * demandingness_factor,
                1,
            ),
            "no bearing": (1, 1),
            "overwhelming undermining": (1, self.config.OVERWHELMING),
            "strong undermining": (1, self.config.STRONG),
            "moderate undermining": (1, self.config.MODERATE),
            "weak undermining": (1, self.config.WEAK),
        }
        if support not in support_map:
            raise ValueError(f"Unrecognised support: {support!r}")
        return support_map[support]


# ---------------------------------------------------------------------------
# Label-level pooling helpers
# ---------------------------------------------------------------------------


def _logit_np(p: float, eps: float = 1e-9) -> float:
    p = max(min(p, 1.0 - eps), eps)
    return float(np.log(p / (1.0 - p)))


def _sanitize_label(s: str) -> str:
    return s.replace(" ", "_").replace("/", "_").lower()


def collect_tree_label_groups(
    stance_data: Dict,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Walk the stance tree and collect unique (support, demandingness) and
    demandingness-only label groups that appear on feature/subfeature/indicator
    nodes. Used to decide which hyperparameters to instantiate under
    POOL_BETAS_BY_LABEL.

    Returns (pres_groups, abs_groups) with pres_groups a list of (support,
    demandingness) tuples and abs_groups a list of demandingness strings.
    Both sorted deterministically.
    """
    pres_set: set = set()
    abs_set: set = set()

    def walk(node: Dict) -> None:
        s = node.get("support")
        d = node.get("demandingness")
        ntype = (node.get("type") or "").lower()
        if ntype in {"feature", "subfeature", "indicator"} and s is not None and d is not None:
            pres_set.add((s, d))
            abs_set.add(d)
        for child in node.get("evidencers", []):
            walk(child)

    for child in stance_data.get("evidencers", []):
        walk(child)
    return sorted(pres_set), sorted(abs_set)


def build_safe_gain_node_beta(
    name: str,
    kind: str,  # "pres" or "abs"
    mu_gain: float,
    sigma: float,
) -> pt.TensorVariable:
    """Build a safe-gain node-level beta as logit-Normal centred at logit(mu_gain).

    Non-centred parameterisation.  Sidesteps Beta boundary singularities
    when the gained mean approaches 0 or 1.
    """
    suffix = {"pres": "_beta_pres", "abs": "_beta_abs"}[kind]
    tilde = pm.Normal(f"{name}{suffix}_lnbeta_tilde", mu=0.0, sigma=1.0)
    logit_beta = pt.constant(_logit_np(mu_gain)) + sigma * tilde
    return pm.Deterministic(f"{name}{suffix}", pt.sigmoid(logit_beta))


def build_label_pool_hyperparameters(
    config: ModelConfig,
    evidence_processor: "EvidenceProcessor",
    stance_data: Dict,
) -> Tuple[
    Dict[Tuple[str, str], pt.TensorVariable],
    Dict[Any, pt.TensorVariable],
]:
    """Instantiate complete-pooling label-level betas (non-centred logit-Normal).

    For each (support, demandingness) group in the tree, creates a single
    ``beta_pres`` random variable.

    For ``beta_abs``, the grouping depends on
    ``config.BETA_ABS_BY_SUPPORT_DEMAND``:
      - False (default): one ``beta_abs`` RV per demandingness group; all
        same-demandingness nodes share it.  ``beta_abs_by_group`` keys are
        demandingness strings.  PyMC variable name: ``beta_abs__{demand}``.
      - True: one ``beta_abs`` RV per (support, demandingness) group;
        same paper β_abs prior centre as default (which depends only on
        demandingness), but the *grouping* is finer.  ``beta_abs_by_group``
        keys are ``(support, demandingness)`` tuples.  PyMC variable name:
        ``beta_abs__{support}__{demand}``.

    Returns ``(beta_pres_by_group, beta_abs_by_group)``.  Also exposes
    ``label_delta__{support}__{demand}`` = beta_pres - beta_abs per pres
    group; under the (s, d) abs grouping the matching abs group has the
    same key, so label_delta is per-(s, d).
    """
    pres_groups, abs_groups = collect_tree_label_groups(stance_data)
    sigma = config.LABEL_POOL_SIGMA
    override_sigma = (
        float(config.BETA_OVERRIDE_SIGMA)
        if config.BETA_OVERRIDE_SIGMA is not None
        else sigma
    )
    pres_override = config.BETA_PRES_OVERRIDE_MEAN
    abs_override = config.BETA_ABS_OVERRIDE_MEAN

    beta_pres_by_group: Dict[Tuple[str, str], pt.TensorVariable] = {}
    for (s, d) in pres_groups:
        if pres_override is not None:
            mu = float(pres_override)
            sigma_used = override_sigma
        else:
            alpha_p, beta_p, _, _ = evidence_processor.get_beta_parameters(s, d)
            mu = alpha_p / (alpha_p + beta_p)
            sigma_used = sigma
        raw = pm.Normal(
            f"beta_pres_tilde__{_sanitize_label(s)}__{_sanitize_label(d)}",
            mu=0.0,
            sigma=1.0,
        )
        logit_beta = pt.constant(_logit_np(mu)) + sigma_used * raw
        beta = pm.Deterministic(
            f"beta_pres__{_sanitize_label(s)}__{_sanitize_label(d)}",
            pt.sigmoid(logit_beta),
        )
        beta_pres_by_group[(s, d)] = beta

    beta_abs_by_group: Dict[Any, pt.TensorVariable] = {}
    if config.BETA_ABS_BY_SUPPORT_DEMAND:
        # One β_abs per (s, d) group present in the tree.  Prior centre is
        # still the paper's d-only β_abs mean (paper β_abs has no
        # support-dependence).  Only the grouping changes.
        for (s, d) in pres_groups:
            if abs_override is not None:
                mu = float(abs_override)
                sigma_used = override_sigma
            else:
                _, _, alpha_a, beta_a = evidence_processor.get_beta_parameters(
                    "no bearing", d
                )
                mu = alpha_a / (alpha_a + beta_a)
                sigma_used = sigma
            raw = pm.Normal(
                f"beta_abs_tilde__{_sanitize_label(s)}__{_sanitize_label(d)}",
                mu=0.0,
                sigma=1.0,
            )
            logit_beta = pt.constant(_logit_np(mu)) + sigma_used * raw
            beta = pm.Deterministic(
                f"beta_abs__{_sanitize_label(s)}__{_sanitize_label(d)}",
                pt.sigmoid(logit_beta),
            )
            beta_abs_by_group[(s, d)] = beta
    else:
        for d in abs_groups:
            if abs_override is not None:
                mu = float(abs_override)
                sigma_used = override_sigma
            else:
                _, _, alpha_a, beta_a = evidence_processor.get_beta_parameters(
                    "no bearing", d
                )
                mu = alpha_a / (alpha_a + beta_a)
                sigma_used = sigma
            raw = pm.Normal(
                f"beta_abs_tilde__{_sanitize_label(d)}",
                mu=0.0,
                sigma=1.0,
            )
            logit_beta = pt.constant(_logit_np(mu)) + sigma_used * raw
            beta = pm.Deterministic(
                f"beta_abs__{_sanitize_label(d)}",
                pt.sigmoid(logit_beta),
            )
            beta_abs_by_group[d] = beta

    for (s, d), beta_p in beta_pres_by_group.items():
        beta_a_key: Any = (s, d) if config.BETA_ABS_BY_SUPPORT_DEMAND else d
        beta_a = beta_abs_by_group[beta_a_key]
        pm.Deterministic(
            f"label_delta__{_sanitize_label(s)}__{_sanitize_label(d)}",
            beta_p - beta_a,
        )

    return beta_pres_by_group, beta_abs_by_group


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


class BayesianModelBuilder:
    """Builds a PyMC model with ordinal observation layer.

    Hierarchy above indicators (stance -> features -> subfeatures) uses the
    standard Beta/Bernoulli structure.  At indicator leaves, expert ratings
    are modelled via a constrained ordered probit with the latent indicator
    state z_j marginalised analytically using the parent-implied probability
    q_j = P(z_j = 1 | tree above).
    """

    def __init__(
        self,
        config: ModelConfig,
        evidence_processor: EvidenceProcessor,
        ordinal_data: OrdinalDataProcessor,
    ):
        self.config = config
        self.evidence_processor = evidence_processor
        self.ordinal_data = ordinal_data
        self.variable_names: List[str] = []
        self.logger = logging.getLogger(self.__class__.__name__)

        # Populated during build_model
        self.a: Optional[pt.TensorVariable] = None
        self.b: Optional[pt.TensorVariable] = None
        self.kappa: Optional[pt.TensorVariable] = None
        self.kappa_by_expert: Optional[pt.TensorVariable] = None
        self.sigma_by_expert: Optional[pt.TensorVariable] = None
        # Maps node_key -> sanitised PyMC variable prefix
        self.node_to_varname: Dict[str, str] = {}
        # Label-pooling betas (populated only if POOL_BETAS_BY_LABEL)
        self.beta_pres_by_group: Dict[Tuple[str, str], pt.TensorVariable] = {}
        self.beta_abs_by_group: Dict[str, pt.TensorVariable] = {}

    def sanitize_name(self, name: str) -> str:
        """Create a valid PyMC variable name, handling duplicates."""
        sanitized = name.replace(" ", "_").replace("/", "_").lower()
        if sanitized not in self.variable_names:
            self.variable_names.append(sanitized)
            return sanitized
        counter = len([x for x in self.variable_names if x.startswith(sanitized)])
        unique_name = f"{sanitized}_{counter}"
        self.variable_names.append(unique_name)
        return unique_name

    # -- tree construction -------------------------------------------------

    def _create_node_variable(
        self,
        evidencer: Dict,
        parent_prob: pt.TensorVariable,
        ancestor_path: Tuple[str, ...],
    ) -> Optional[pt.TensorVariable]:
        """Create a hierarchy node with the parent's discrete state marginalised.

        Instead of sampling a discrete Bernoulli for each node, we propagate
        the continuous probability P(parent=1) downward:
            q_j = P(parent=1) * beta_present + P(parent=0) * beta_absent
        This is analytically equivalent (Rao-Blackwell) and keeps the entire
        model continuous so NUTS can sample without Metropolis steps.

        Returns
        -------
        Continuous P(node=1) tensor for features/subfeatures (passed to children).
        None for indicators (likelihood added via Potential).
        """
        name = self.sanitize_name(evidencer["name"])
        key = node_key(ancestor_path, evidencer["name"])
        self.node_to_varname[key] = name

        support = evidencer.get("support", "no bearing")
        demand = evidencer.get("demandingness", "neutral")

        if self.config.POOL_BETAS_BY_LABEL:
            # Complete-pooling-within-label: reuse the single group-level
            # beta.  beta_abs key depends on BETA_ABS_BY_SUPPORT_DEMAND.
            beta_present = self.beta_pres_by_group[(support, demand)]
            abs_key: Any = (
                (support, demand)
                if self.config.BETA_ABS_BY_SUPPORT_DEMAND
                else demand
            )
            beta_absent = self.beta_abs_by_group[abs_key]
        elif self.config.GAIN_LOGIT_NORMAL and self.config.TRANSMISSION_GAIN != 1.0:
            # Safe gain: node-level logit-Normal centred at gained means
            mu_p, mu_a = self.evidence_processor.get_gained_means(support, demand)
            sigma = self.config.GAIN_LOGIT_NORMAL_SIGMA
            beta_present = build_safe_gain_node_beta(name, "pres", mu_p, sigma)
            beta_absent = build_safe_gain_node_beta(name, "abs", mu_a, sigma)
        else:
            alpha_pres, beta_pres, alpha_abs, beta_abs = (
                self.evidence_processor.get_beta_parameters(support, demand)
            )
            beta_present = pm.Beta(f"{name}_beta_pres", alpha=alpha_pres, beta=beta_pres)
            beta_absent = pm.Beta(f"{name}_beta_abs", alpha=alpha_abs, beta=beta_abs)

        q_j = pm.Deterministic(
            f"{name}_p",
            parent_prob * beta_present + (1 - parent_prob) * beta_absent,
        )

        if evidencer["type"].lower() == "indicator":
            self._add_indicator_ordinal_likelihood(evidencer, name, q_j, ancestor_path)
            return None
        else:
            pm.Deterministic(f"{name}_bern", q_j)
            return q_j

    def _add_indicator_ordinal_likelihood(
        self,
        evidencer: Dict,
        name: str,
        q_j: pt.TensorVariable,
        ancestor_path: Tuple[str, ...],
    ) -> None:
        """Add marginalised ordinal likelihood + per-indicator deterministics.

        Dispatches on ``config.INDICATOR_STATE_MODEL`` -- binary or three-state.
        If no valid observations exist, exposes prior-only deterministics.
        """
        key = node_key(ancestor_path, evidencer["name"])
        obs_data = self.ordinal_data.observations.get(key, [])
        state_model = self.config.INDICATOR_STATE_MODEL

        if not obs_data:
            self.logger.debug(f"No observations for {key} -- prior deterministics only")
            add_no_data_prior_deterministics(name, q_j, state_model)
            return

        ratings = np.array([r for _, r in obs_data], dtype=np.int64)
        expert_indices = np.array([e for e, _ in obs_data], dtype=np.int64)

        add_indicator_marginal_likelihood(
            name=name,
            potential_name=f"{name}_ordinal_lik",
            q_j=q_j,
            ratings=ratings,
            expert_indices=expert_indices,
            a=self.a,
            kappa=self.kappa,
            b=self.b,
            kappa_by_expert=self.kappa_by_expert,
            state_model=state_model,
            sigma_by_expert=self.sigma_by_expert,
        )

    def _add_evidencer(
        self,
        parent_prob: pt.TensorVariable,
        evidencer: Dict,
        ancestor_path: Tuple[str, ...],
    ) -> None:
        """Add a single evidencer node to the model."""
        current_path = ancestor_path + (evidencer["name"],)

        try:
            var = self._create_node_variable(evidencer, parent_prob, ancestor_path)
        except Exception as e:
            self.logger.warning(f"Failed to add evidencer {evidencer['name']}: {e}")
            return

        # Recurse into children for features / subfeatures
        if evidencer["type"].lower() in {"feature", "subfeature"} and var is not None:
            for child in evidencer.get("evidencers", []):
                self._add_evidencer(var, child, ancestor_path=current_path)

    def _add_evidencers(
        self,
        parent_prob: pt.TensorVariable,
        evidencers: List[Dict],
        ancestor_path: Tuple[str, ...],
    ) -> None:
        """Add multiple evidencer nodes to the model."""
        self.logger.info(f"Adding {len(evidencers)} evidencers")
        for evidencer in evidencers:
            self._add_evidencer(parent_prob, evidencer, ancestor_path)

    # -- public API --------------------------------------------------------

    def build_model(self, stance_data: Dict) -> pm.Model:
        """Build the complete PyMC model with ordinal observation layer.

        Does not sample -- call ``sample()`` separately.
        """
        n_experts = len(self.ordinal_data.expert_names)
        K = self.config.N_CATEGORIES
        stance_name_raw = stance_data["name"]
        stance_name = self.sanitize_name(stance_name_raw)
        self.node_to_varname[stance_name_raw] = stance_name

        self.logger.info(
            f"Building ordinal DCM: stance={stance_name_raw}, "
            f"{n_experts} experts, K={K}"
        )

        model = pm.Model()
        with model:
            # --- Stance prior (marginalised: no discrete Bernoulli) ---
            stance_p = pm.Beta(
                f"{stance_name}_beta",
                alpha=self.config.DEFAULT_ALPHA,
                beta=self.config.DEFAULT_BETA,
            )
            pm.Deterministic(f"{stance_name}_bern", stance_p)

            # --- Ordinal observation parameters (shared, defined once) ---
            (
                self.a,
                self.b,
                self.kappa,
                self.kappa_by_expert,
                self.sigma_by_expert,
            ) = build_ordinal_observation_layer(self.config, n_experts, K)

            # --- Optional label-level beta pooling (POOL_BETAS_BY_LABEL) ---
            if self.config.POOL_BETAS_BY_LABEL:
                (
                    self.beta_pres_by_group,
                    self.beta_abs_by_group,
                ) = build_label_pool_hyperparameters(
                    self.config, self.evidence_processor, stance_data
                )

            # --- Hierarchy ---
            evidencers = stance_data.get("evidencers", [])
            ancestor_path = (stance_name_raw,)
            self._add_evidencers(stance_p, evidencers, ancestor_path)

        return model

    def sample(self, model: pm.Model) -> Any:
        """Run MCMC sampling on the built model."""
        self.logger.info(
            f"Sampling: {self.config.NUM_SAMPLES} draws, "
            f"{self.config.NUM_TUNE} tune, {self.config.NUM_CHAINS} chains"
        )
        start = time.time()
        with model:
            idata = pm.sample(
                draws=self.config.NUM_SAMPLES,
                tune=self.config.NUM_TUNE,
                chains=self.config.NUM_CHAINS,
                cores=self.config.NUM_CHAINS,
                target_accept=self.config.TARGET_ACCEPT,
                random_seed=42,
            )
        elapsed = time.time() - start
        self.logger.info(f"Sampling completed in {elapsed:.1f}s")
        return idata


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class ResultsManager:
    """Extracts and displays posterior results from the ordinal DCM."""

    def __init__(self, config: ModelConfig, node_to_varname: Dict[str, str]):
        self.config = config
        self.node_to_varname = node_to_varname
        self.logger = logging.getLogger(self.__class__.__name__)

    def summarise(self, idata: Any, stance_data: Dict) -> None:
        """Print hierarchical posterior summary."""
        self._print_node(idata, stance_data, indent=0, ancestor_path=())

    def _get_var_mean(self, idata: Any, var_name: str) -> Optional[float]:
        try:
            return float(idata.posterior[var_name].mean())
        except (KeyError, AttributeError):
            return None

    def _print_node(
        self,
        idata: Any,
        node: Dict,
        indent: int,
        ancestor_path: Tuple[str, ...],
    ) -> None:
        prefix = "  " * indent
        name = node["name"]
        current_path = ancestor_path + (name,)
        key = node_key(ancestor_path, name)
        varname = self.node_to_varname.get(key)
        node_type = node.get("type", "").lower()

        print(f"{prefix}{name}")

        if varname:
            if node_type == "indicator":
                pz1 = self._get_var_mean(idata, f"{varname}_pz1")
                if pz1 is not None:
                    print(f"{prefix}  P(present | data): {pz1:.4f}")
                else:
                    print(f"{prefix}  P(present | data): N/A")
            else:
                bern = self._get_var_mean(idata, f"{varname}_bern")
                if bern is not None:
                    print(f"{prefix}  Posterior: {bern:.4f}")

        if "support" in node:
            print(f"{prefix}  Support: {node['support']}")
        if "demandingness" in node:
            print(f"{prefix}  Demandingness: {node['demandingness']}")

        for child in node.get("evidencers", []):
            self._print_node(idata, child, indent + 1, ancestor_path=current_path)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(config: ModelConfig) -> List[Dict]:
    """Load DCM data from local cache (no network dependency)."""
    path = Path(config.DATA_CACHE_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Data cache not found at {path}. "
            f"Place data_cache.json in the working directory."
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fit a single stance (reusable entry point)
# ---------------------------------------------------------------------------


def fit_stance(
    stance_data: Dict,
    config: ModelConfig,
    system: Optional[str] = None,
) -> Tuple[Any, BayesianModelBuilder, OrdinalDataProcessor]:
    """Fit the ordinal DCM for a single stance.

    Parameters
    ----------
    stance_data : dict
        One element of the list returned by ``load_data()``.
    config : ModelConfig
        Sampling and prior configuration.
    system : str, optional
        Target system name.  Defaults to ``config.TARGET_SYSTEM``.

    Returns
    -------
    (idata, builder, processor)
    """
    system = system or config.TARGET_SYSTEM
    logger = logging.getLogger(__name__)
    logger.info(f"Fitting stance: {stance_data['name']} | system: {system}")

    processor = OrdinalDataProcessor(config)
    processor.process(stance_data, system)

    evidence_proc = EvidenceProcessor(config)
    builder = BayesianModelBuilder(config, evidence_proc, processor)
    model = builder.build_model(stance_data)
    idata = builder.sample(model)

    return idata, builder, processor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logger = setup_logging("INFO")
    logger.info("Starting DCM ordinal model analysis")

    config = ModelConfig()

    all_data = load_data(config)
    stance_data = next(
        (item for item in all_data if item["name"] == config.TARGET_STANCE),
        None,
    )
    if not stance_data:
        raise ValueError(f"Stance not found: {config.TARGET_STANCE}")

    idata, builder, processor = fit_stance(stance_data, config)

    # Report
    results = ResultsManager(config, builder.node_to_varname)
    print(f"\n{'=' * 60}")
    print(f"System: {config.TARGET_SYSTEM}")
    print(f"Stance: {config.TARGET_STANCE}")
    print(f"{'=' * 60}\n")
    results.summarise(idata, stance_data)

    print(f"\n{'=' * 60}")
    print("Observation model parameters")
    print(f"{'=' * 60}")
    print(f"  a (discrimination): {float(idata.posterior['a'].mean()):.3f}")
    n_experts = len(processor.expert_names)
    if "b_free" in idata.posterior.data_vars:
        b_free = idata.posterior["b_free"].mean(dim=("chain", "draw")).values
        print(
            f"  b (expert shifts):  "
            f"[0.000 (anchor: {processor.anchor_expert}), "
            + ", ".join(f"{v:.3f}" for v in b_free)
            + "]"
        )
    kappa_vals = idata.posterior["kappa"].mean(dim=("chain", "draw")).values
    print(f"  kappa (cutpoints):  [{', '.join(f'{v:.3f}' for v in kappa_vals)}]")

    logger.info("Analysis complete")


# ---------------------------------------------------------------------------
# Multi-system joint fit (reference systems)
# ---------------------------------------------------------------------------


class MultiSystemDataProcessor:
    """Processes ordinal observations for multiple systems simultaneously.

    Builds a merged expert pool across all systems so that experts who
    rate multiple systems share a single index and hence a single
    location-shift parameter.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.expert_names: List[str] = []
        self.expert_to_idx: Dict[str, int] = {}
        self.anchor_expert: Optional[str] = None
        self.systems: List[str] = []
        # {system: {node_key: [(global_expert_idx, ordinal_rating), ...]}}
        self.system_observations: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}

    def process(
        self, stance_data: Dict, systems: List[str]
    ) -> "MultiSystemDataProcessor":
        self.systems = list(systems)

        # --- Pass 1: count observations per expert across ALL systems ---
        expert_system_counts: Dict[str, int] = defaultdict(int)
        for system in systems:
            per_sys: Dict[str, int] = defaultdict(int)
            self._count_expert_obs(stance_data, system, per_sys)
            for exp, cnt in per_sys.items():
                expert_system_counts[exp] += cnt

        if not expert_system_counts:
            self.logger.warning("No observations found across any system")
            return self

        # Anchor = expert with most total observations (deterministic tie-break)
        self.anchor_expert = max(
            sorted(expert_system_counts.keys()),
            key=lambda e: expert_system_counts[e],
        )
        other_experts = sorted(
            e for e in expert_system_counts if e != self.anchor_expert
        )
        self.expert_names = [self.anchor_expert] + other_experts
        self.expert_to_idx = {name: i for i, name in enumerate(self.expert_names)}
        self.logger.info(
            f"Merged expert pool ({len(self.expert_names)}): "
            f"{self.expert_names}  anchor={self.anchor_expert}"
        )

        # --- Pass 2: collect ordinal observations per system ---
        for system in systems:
            obs: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
            self._collect_obs(stance_data, system, obs, ancestor_path=())
            self.system_observations[system] = dict(obs)
            n_obs = sum(len(v) for v in obs.values())
            self.logger.info(f"  {system}: {n_obs} obs across {len(obs)} indicators")

        return self

    # -- helpers (mirror OrdinalDataProcessor but write to external dicts) --

    @staticmethod
    def _is_missing(val) -> bool:
        if val is None:
            return True
        s = str(val).strip().lower()
        return s in ("-1", "-1.0", "none", "", "unsure", "not tested")

    def _count_expert_obs(
        self, node: Dict, system: str, counts: Dict[str, int]
    ) -> None:
        if node.get("type", "").lower() == "indicator":
            obs = node.get("observations", {}).get(system)
            if obs:
                for i, val in enumerate(obs["values"]):
                    if not self._is_missing(val):
                        counts[obs["names"][i]] += 1
        for child in node.get("evidencers", []):
            self._count_expert_obs(child, system, counts)

    def _collect_obs(
        self,
        node: Dict,
        system: str,
        obs_dict: Dict[str, List[Tuple[int, int]]],
        ancestor_path: Tuple[str, ...],
    ) -> None:
        current_path = ancestor_path + (node["name"],)
        if node.get("type", "").lower() == "indicator":
            obs = node.get("observations", {}).get(system)
            if obs:
                key = node_key(ancestor_path, node["name"])
                for i, val in enumerate(obs["values"]):
                    if self._is_missing(val):
                        continue
                    expert = obs["names"][i]
                    ordinal = legacy_probability_to_ordinal(
                        float(val), self.config.ORDINAL_BINS
                    )
                    obs_dict[key].append((self.expert_to_idx[expert], ordinal))
        for child in node.get("evidencers", []):
            self._collect_obs(child, system, obs_dict, ancestor_path=current_path)


class MultiSystemModelBuilder:
    """Builds a joint PyMC model across multiple systems.

    Tree parameters (beta_pres, beta_abs) are shared.  Observation-layer
    parameters (a, kappa, b_e) are shared.  Each system gets its own
    stance-level C and per-node q_j propagation.
    """

    def __init__(
        self,
        config: ModelConfig,
        evidence_processor: EvidenceProcessor,
        multi_data: MultiSystemDataProcessor,
        system_configs: List[Tuple[str, Optional[float]]],
    ):
        self.config = config
        self.evidence_processor = evidence_processor
        self.multi_data = multi_data
        self.system_configs = system_configs
        self.variable_names: List[str] = []
        self.node_to_varname: Dict[str, str] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

        self.a: Optional[pt.TensorVariable] = None
        self.b: Optional[pt.TensorVariable] = None
        self.kappa: Optional[pt.TensorVariable] = None
        self.kappa_by_expert: Optional[pt.TensorVariable] = None
        self.sigma_by_expert: Optional[pt.TensorVariable] = None
        self.beta_pres_by_group: Dict[Tuple[str, str], pt.TensorVariable] = {}
        self.beta_abs_by_group: Dict[str, pt.TensorVariable] = {}

    def sanitize_name(self, name: str) -> str:
        sanitized = name.replace(" ", "_").replace("/", "_").lower()
        if sanitized not in self.variable_names:
            self.variable_names.append(sanitized)
            return sanitized
        counter = len([x for x in self.variable_names if x.startswith(sanitized)])
        unique_name = f"{sanitized}_{counter}"
        self.variable_names.append(unique_name)
        return unique_name

    @staticmethod
    def _sys_prefix(system: str) -> str:
        return system.replace(" ", "_").replace("(", "").replace(")", "").lower()

    # -- tree construction with per-system q propagation -------------------

    def _create_shared_node(
        self,
        evidencer: Dict,
        parent_qs: Dict[str, pt.TensorVariable],
        ancestor_path: Tuple[str, ...],
    ) -> Optional[Dict[str, pt.TensorVariable]]:
        """Create shared beta params + per-system q_j for one tree node."""
        name = self.sanitize_name(evidencer["name"])
        key = node_key(ancestor_path, evidencer["name"])
        self.node_to_varname[key] = name

        support = evidencer.get("support", "no bearing")
        demand = evidencer.get("demandingness", "neutral")

        if self.config.POOL_BETAS_BY_LABEL:
            # Complete-pooling-within-label: reuse the single group-level
            # beta.  beta_abs key depends on BETA_ABS_BY_SUPPORT_DEMAND.
            bp = self.beta_pres_by_group[(support, demand)]
            abs_key: Any = (
                (support, demand)
                if self.config.BETA_ABS_BY_SUPPORT_DEMAND
                else demand
            )
            ba = self.beta_abs_by_group[abs_key]
        elif self.config.GAIN_LOGIT_NORMAL and self.config.TRANSMISSION_GAIN != 1.0:
            # Safe gain: node-level logit-Normal centred at gained means,
            # shared across systems (same as the paper Beta case).
            mu_p, mu_a = self.evidence_processor.get_gained_means(support, demand)
            sigma = self.config.GAIN_LOGIT_NORMAL_SIGMA
            bp = build_safe_gain_node_beta(name, "pres", mu_p, sigma)
            ba = build_safe_gain_node_beta(name, "abs", mu_a, sigma)
        else:
            alpha_pres, beta_pres, alpha_abs, beta_abs = (
                self.evidence_processor.get_beta_parameters(support, demand)
            )
            # Shared across systems
            bp = pm.Beta(f"{name}_beta_pres", alpha=alpha_pres, beta=beta_pres)
            ba = pm.Beta(f"{name}_beta_abs", alpha=alpha_abs, beta=beta_abs)

        # Per-system q
        child_qs: Dict[str, pt.TensorVariable] = {}
        for sys_name, parent_p in parent_qs.items():
            sp = self._sys_prefix(sys_name)
            q = pm.Deterministic(
                f"{sp}__{name}_p",
                parent_p * bp + (1 - parent_p) * ba,
            )
            child_qs[sys_name] = q

        if evidencer["type"].lower() == "indicator":
            self._add_multisystem_indicator(evidencer, name, child_qs, ancestor_path)
            return None
        else:
            # Expose per-system bern for features/subfeatures
            for sys_name, q in child_qs.items():
                sp = self._sys_prefix(sys_name)
                pm.Deterministic(f"{sp}__{name}_bern", q)
            return child_qs

    def _add_multisystem_indicator(
        self,
        evidencer: Dict,
        name: str,
        q_by_sys: Dict[str, pt.TensorVariable],
        ancestor_path: Tuple[str, ...],
    ) -> None:
        """Add marginalised ordinal likelihood for each system that has data.

        Dispatches on ``config.INDICATOR_STATE_MODEL`` so binary and
        three-state branches share the same tree-propagation logic.
        """
        key = node_key(ancestor_path, evidencer["name"])
        state_model = self.config.INDICATOR_STATE_MODEL
        for sys_name, q_j in q_by_sys.items():
            sp = self._sys_prefix(sys_name)
            obs_data = self.multi_data.system_observations.get(sys_name, {}).get(
                key, []
            )

            if not obs_data:
                add_no_data_prior_deterministics(f"{sp}__{name}", q_j, state_model)
                continue

            ratings = np.array([r for _, r in obs_data], dtype=np.int64)
            expert_indices = np.array([e for e, _ in obs_data], dtype=np.int64)

            add_indicator_marginal_likelihood(
                name=f"{sp}__{name}",
                potential_name=f"{sp}__{name}_lik",
                q_j=q_j,
                ratings=ratings,
                expert_indices=expert_indices,
                a=self.a,
                kappa=self.kappa,
                b=self.b,
                kappa_by_expert=self.kappa_by_expert,
                state_model=state_model,
                sigma_by_expert=self.sigma_by_expert,
            )

    def _add_shared_evidencer(
        self,
        parent_qs: Dict[str, pt.TensorVariable],
        evidencer: Dict,
        ancestor_path: Tuple[str, ...],
    ) -> None:
        current_path = ancestor_path + (evidencer["name"],)
        try:
            child_qs = self._create_shared_node(evidencer, parent_qs, ancestor_path)
        except Exception as e:
            self.logger.warning(f"Failed to add evidencer {evidencer['name']}: {e}")
            return
        if (
            evidencer["type"].lower() in {"feature", "subfeature"}
            and child_qs is not None
        ):
            for child in evidencer.get("evidencers", []):
                self._add_shared_evidencer(child_qs, child, ancestor_path=current_path)

    # -- public API --------------------------------------------------------

    def build_model(self, stance_data: Dict) -> pm.Model:
        n_experts = len(self.multi_data.expert_names)
        K = self.config.N_CATEGORIES
        stance_name_raw = stance_data["name"]
        stance_name = self.sanitize_name(stance_name_raw)
        self.node_to_varname[stance_name_raw] = stance_name

        self.logger.info(
            f"Building multi-system ordinal DCM: stance={stance_name_raw}, "
            f"{len(self.system_configs)} systems, {n_experts} experts, K={K}"
        )

        model = pm.Model()
        with model:
            # --- Per-system stance C ---
            soft = self.config.SOFT_REFERENCE_ANCHORS or {}
            stance_qs: Dict[str, pt.TensorVariable] = {}
            for sys_name, c_fixed in self.system_configs:
                sp = self._sys_prefix(sys_name)
                if sys_name in soft:
                    alpha_s, beta_s = soft[sys_name]
                    c_var = pm.Beta(
                        f"{sp}__{stance_name}_C",
                        alpha=alpha_s,
                        beta=beta_s,
                    )
                    stance_qs[sys_name] = c_var
                elif c_fixed is not None:
                    c_val = pt.constant(c_fixed, dtype="floatX")
                    pm.Deterministic(f"{sp}__{stance_name}_C", c_val)
                    stance_qs[sys_name] = c_val
                else:
                    c_var = pm.Beta(
                        f"{sp}__{stance_name}_C",
                        alpha=self.config.DEFAULT_ALPHA,
                        beta=self.config.DEFAULT_BETA,
                    )
                    stance_qs[sys_name] = c_var

            # --- Shared observation layer ---
            (
                self.a,
                self.b,
                self.kappa,
                self.kappa_by_expert,
                self.sigma_by_expert,
            ) = build_ordinal_observation_layer(self.config, n_experts, K)

            # --- Optional label-level beta pooling (POOL_BETAS_BY_LABEL) ---
            if self.config.POOL_BETAS_BY_LABEL:
                (
                    self.beta_pres_by_group,
                    self.beta_abs_by_group,
                ) = build_label_pool_hyperparameters(
                    self.config, self.evidence_processor, stance_data
                )

            # --- Shared hierarchy with per-system q propagation ---
            evidencers = stance_data.get("evidencers", [])
            ancestor_path = (stance_name_raw,)
            for ev in evidencers:
                self._add_shared_evidencer(stance_qs, ev, ancestor_path=ancestor_path)

        return model

    def sample(self, model: pm.Model) -> Any:
        self.logger.info(
            f"Sampling: {self.config.NUM_SAMPLES} draws, "
            f"{self.config.NUM_TUNE} tune, {self.config.NUM_CHAINS} chains"
        )
        start = time.time()
        with model:
            idata = pm.sample(
                draws=self.config.NUM_SAMPLES,
                tune=self.config.NUM_TUNE,
                chains=self.config.NUM_CHAINS,
                cores=self.config.NUM_CHAINS,
                target_accept=self.config.TARGET_ACCEPT,
                random_seed=42,
            )
        elapsed = time.time() - start
        self.logger.info(f"Sampling completed in {elapsed:.1f}s")
        return idata


def fit_stance_multisystem(
    stance_data: Dict,
    config: ModelConfig,
    system_configs: List[Tuple[str, Optional[float]]],
) -> Tuple[Any, MultiSystemModelBuilder, MultiSystemDataProcessor]:
    """Fit the ordinal DCM jointly across multiple systems.

    Parameters
    ----------
    stance_data : dict
        One element of the list returned by ``load_data()``.
    config : ModelConfig
        Sampling and prior configuration.
    system_configs : list of (system_name, fixed_c_or_None)
        For reference systems supply a float (e.g. 0.99 for Human);
        for target systems supply None (gets a Beta(1,5) prior).

    Returns
    -------
    (idata, builder, processor)
    """
    logger = logging.getLogger(__name__)
    systems = [s for s, _ in system_configs]
    logger.info(f"Multi-system fit: stance={stance_data['name']} | systems={systems}")

    processor = MultiSystemDataProcessor(config)
    processor.process(stance_data, systems)

    evidence_proc = EvidenceProcessor(config)
    builder = MultiSystemModelBuilder(config, evidence_proc, processor, system_configs)
    model = builder.build_model(stance_data)
    idata = builder.sample(model)

    return idata, builder, processor


if __name__ == "__main__":
    main()
