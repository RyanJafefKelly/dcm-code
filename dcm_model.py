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

    # Indicator latent-state model. Selects the leaf family without touching the rest
    # of the validated baseline. Members of the growing model library:
    #   "binary"      — z_j ∈ {0,1},   z_j ~ Bernoulli(q_j)        (original baseline)
    #   "three_state" — m_j ∈ {0,1,2}, m_j ~ Binomial(2, q_j);
    #                   emission centres at η ∈ {0, a/2, a};
    #                   z_j = m_j/2 gives expected_z = q_j
    # Marginalisation is per-indicator in both cases (m_j or z_j is shared across the
    # indicator's ratings), analytic in both cases, so NUTS sees a fully continuous model.
    INDICATOR_STATE_MODEL: Literal["binary", "three_state"] = "binary"

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
    N_CATEGORIES: int = 7
    # Bin edges for legacy probability -> ordinal conversion.
    # Category k is assigned when bins[k-1] <= p < bins[k].
    ORDINAL_BINS: Tuple[float, ...] = (0.05, 0.20, 0.40, 0.60, 0.80, 0.95)

    # Default targets
    TARGET_STANCE: str = "Global Workspace Theory"
    TARGET_SYSTEM: str = "2024 Leading Chat LLMs"

    # Data source
    DATA_CACHE_PATH: str = "data_cache.json"


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
) -> pt.TensorVariable:
    """Vectorised log P(ratings | params, eta_shift) under ordered probit.

    Implements the cumulative-normal parameterisation:
        P(r = k) = Phi(kappa_k - eta) - Phi(kappa_{k-1} - eta)
    with eta = b[expert] + eta_shift  (eta_shift = 0 for z=0, a for z=1).

    Parameters
    ----------
    ratings : (N,) int array -- observed ordinal categories, 0-indexed.
    expert_idx : (N,) int array -- expert index for each observation.
    kappa : (K-1,) pytensor -- ordered cutpoints.
    b : (E,) pytensor -- expert location shifts.
    eta_shift : scalar pytensor -- additional shift (0 or a).

    Returns
    -------
    Scalar pytensor: sum of log-probabilities across all observations.
    """
    n_obs = ratings.shape[0]
    eta = b[expert_idx] + eta_shift  # (N,)
    c_minus_eta = kappa[None, :] - eta[:, None]  # (N, K-1)
    cum_probs = pt.erfc(-c_minus_eta / pt.sqrt(2.0)) / 2.0  # Phi(c - eta)
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
) -> pt.TensorVariable:
    """Ordered-probit logp with expert-specific cutpoints and no free location."""
    n_obs = ratings.shape[0]
    kappa_obs = kappa_by_expert[expert_idx]  # (N, K-1)
    c_minus_eta = kappa_obs - eta_shift
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
) -> Tuple[pt.TensorVariable, pt.TensorVariable]:
    """Return log-likelihood terms for z=0 and z=1 under the active obs layer."""
    if kappa_by_expert is not None:
        return (
            pt_ordinal_logp_expert_kappa(
                ratings, expert_idx, kappa_by_expert, pt.constant(0.0)
            ),
            pt_ordinal_logp_expert_kappa(ratings, expert_idx, kappa_by_expert, a),
        )
    return (
        pt_ordinal_logp(ratings, expert_idx, kappa, b, pt.constant(0.0)),
        pt_ordinal_logp(ratings, expert_idx, kappa, b, a),
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
) -> Tuple[pt.TensorVariable, pt.TensorVariable, pt.TensorVariable]:
    """Aggregated per-indicator log-likelihoods for m in {0, 1, 2}.

    Each returned component is ``sum_i log P_OP(r_ij | eta=eta_m, theta)``
    for emission centres ``eta_0 = 0``, ``eta_1 = a/2``, ``eta_2 = a``.
    The marginalisation over m_j is then performed by the caller using
    ``three_state_log_weights(q_j)`` and pairwise ``pt.logaddexp``.
    """
    if kappa_by_expert is not None:
        return (
            pt_ordinal_logp_expert_kappa(
                ratings, expert_idx, kappa_by_expert, pt.constant(0.0)
            ),
            pt_ordinal_logp_expert_kappa(
                ratings, expert_idx, kappa_by_expert, a * 0.5
            ),
            pt_ordinal_logp_expert_kappa(ratings, expert_idx, kappa_by_expert, a),
        )
    return (
        pt_ordinal_logp(ratings, expert_idx, kappa, b, pt.constant(0.0)),
        pt_ordinal_logp(ratings, expert_idx, kappa, b, a * 0.5),
        pt_ordinal_logp(ratings, expert_idx, kappa, b, a),
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
) -> None:
    """Add the marginalised ordinal likelihood + deterministics for one indicator.

    Dispatches on ``state_model``:
      - "binary":      z_j ~ Bernoulli(q_j). Exposes ``{name}_pz1``.
      - "three_state": m_j ~ Binomial(2, q_j). Exposes ``{name}_p_m0``,
        ``{name}_p_m1``, ``{name}_p_m2``, and ``{name}_expected_z``.

    ``ratings`` / ``expert_indices`` must contain every rating for the
    single indicator being added (per-indicator marginalisation).
    """
    if state_model == "binary":
        ll_z0, ll_z1 = pt_indicator_logps(
            ratings, expert_indices, a, kappa, b, kappa_by_expert
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
            ratings, expert_indices, a, kappa, b, kappa_by_expert
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
    raise ValueError(
        f"Unknown INDICATOR_STATE_MODEL: {state_model!r}. "
        f"Expected 'binary' or 'three_state'."
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
    raise ValueError(
        f"Unknown INDICATOR_STATE_MODEL: {state_model!r}. "
        f"Expected 'binary' or 'three_state'."
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
]:
    """Build the shared ordinal observation layer for single- or multi-system fits."""
    if config.USE_HIERARCHICAL_EXPERT_CUTPOINTS and config.USE_EXPERT_SHIFTS:
        raise ValueError(
            "Hierarchical expert cutpoints and expert shifts cannot both be enabled"
        )

    a = pm.HalfNormal("a", sigma=config.A_PRIOR_SIGMA)

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
        return a, b, kappa, kappa_by_expert

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
    return a, b, kappa, None


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

        Returns
        -------
        (alpha_present, beta_present, alpha_absent, beta_absent)
        """
        absence_alpha, absence_beta = self._get_demandingness_parameters(demandingness)
        support_factor = self._get_support_factor(support, demandingness)

        presence_alpha = int(absence_alpha * support_factor[0])
        presence_beta = int(absence_beta * support_factor[1])

        c = self.config.NODE_CONCENTRATION
        return (
            presence_alpha * c / (presence_alpha + presence_beta),
            presence_beta * c / (presence_alpha + presence_beta),
            absence_alpha * c / (absence_alpha + absence_beta),
            absence_beta * c / (absence_alpha + absence_beta),
        )

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
        # Maps node_key -> sanitised PyMC variable prefix
        self.node_to_varname: Dict[str, str] = {}

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

        alpha_pres, beta_pres, alpha_abs, beta_abs = (
            self.evidence_processor.get_beta_parameters(
                evidencer.get("support", "no bearing"),
                evidencer.get("demandingness", "neutral"),
            )
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
            ) = build_ordinal_observation_layer(self.config, n_experts, K)

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

        alpha_pres, beta_pres, alpha_abs, beta_abs = (
            self.evidence_processor.get_beta_parameters(
                evidencer.get("support", "no bearing"),
                evidencer.get("demandingness", "neutral"),
            )
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
            stance_qs: Dict[str, pt.TensorVariable] = {}
            for sys_name, c_fixed in self.system_configs:
                sp = self._sys_prefix(sys_name)
                if c_fixed is not None:
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
            ) = build_ordinal_observation_layer(self.config, n_experts, K)

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
