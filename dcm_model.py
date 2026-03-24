"""Ordinal observation model for the Digital Consciousness Model (DCM).

Replaces the binary-collapse observation layer with a constrained ordered probit
(rating-SDT) model for 7-point ordinal expert responses. All discrete latent
variables (stance, feature, subfeature presence) are marginalised analytically,
yielding a fully continuous model that NUTS can sample without Metropolis steps.

Current limitations
-------------------
- GWT stance only (other stances require dcm_model_binary_legacy.py)
- Likert-only core; legacy probability data uses a temporary adapter
- Shared discrimination parameter (a)
- Shared cutpoints (kappa)
- Expert-specific location shifts (b_e) with one anchored at zero
- No rigorous sensitivity analysis or PPC yet

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

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

    # Ordinal observation model
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


def legacy_probability_to_ordinal(
    p: float, bins: Tuple[float, ...]
) -> int:
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
    eta = b[expert_idx] + eta_shift                              # (N,)
    c_minus_eta = kappa[None, :] - eta[:, None]                  # (N, K-1)
    cum_probs = pt.erfc(-c_minus_eta / pt.sqrt(2.0)) / 2.0      # Phi(c - eta)
    zeros = pt.zeros((cum_probs.shape[0], 1))
    ones = pt.ones((cum_probs.shape[0], 1))
    cum_full = pt.concatenate([zeros, cum_probs, ones], axis=1)  # (N, K+1)
    cat_probs = cum_full[:, 1:] - cum_full[:, :-1]              # (N, K)
    cat_probs = pt.clip(cat_probs, 1e-12, 1.0)
    log_p = pt.log(cat_probs[pt.arange(n_obs), ratings])         # (N,)
    return pt.sum(log_p)


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

    def process(
        self, stance_data: Dict, system: str
    ) -> "OrdinalDataProcessor":
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
        other_experts = sorted(
            e for e in expert_counts if e != self.anchor_expert
        )
        self.expert_names = [self.anchor_expert] + other_experts
        self.expert_to_idx = {
            name: i for i, name in enumerate(self.expert_names)
        }
        self.logger.info(f"Expert mapping: {self.expert_to_idx}")

        # --- Second pass: collect ordinal observations keyed by node path ---
        self._collect_observations(stance_data, system, ancestor_path=())

        n_obs = sum(len(v) for v in self.observations.values())
        n_ind = len(self.observations)
        self.logger.info(
            f"Collected {n_obs} observations across {n_ind} indicators"
        )
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
                    self.observations[key].append(
                        (self.expert_to_idx[expert], ordinal)
                    )

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
        absence_alpha, absence_beta = self._get_demandingness_parameters(
            demandingness
        )
        support_factor = self._get_support_factor(support, demandingness)

        presence_alpha = int(absence_alpha * support_factor[0])
        presence_beta = int(absence_beta * support_factor[1])

        return (
            presence_alpha * 10 / (presence_alpha + presence_beta),
            presence_beta * 10 / (presence_alpha + presence_beta),
            absence_alpha * 10 / (absence_alpha + absence_beta),
            absence_beta * 10 / (absence_alpha + absence_beta),
        )

    def _get_demandingness_parameters(
        self, demandingness: str
    ) -> Tuple[int, int]:
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
                self.config.OVERWHELMING * demandingness_factor, 1,
            ),
            "strong support": (
                self.config.STRONG * demandingness_factor, 1,
            ),
            "moderate support": (
                self.config.MODERATE * demandingness_factor, 1,
            ),
            "weak support": (
                self.config.WEAK * demandingness_factor, 1,
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
        # Maps node_key -> sanitised PyMC variable prefix
        self.node_to_varname: Dict[str, str] = {}

    def sanitize_name(self, name: str) -> str:
        """Create a valid PyMC variable name, handling duplicates."""
        sanitized = name.replace(" ", "_").lower()
        if sanitized not in self.variable_names:
            self.variable_names.append(sanitized)
            return sanitized
        counter = len(
            [x for x in self.variable_names if x.startswith(sanitized)]
        )
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

        beta_present = pm.Beta(
            f"{name}_beta_pres", alpha=alpha_pres, beta=beta_pres
        )
        beta_absent = pm.Beta(
            f"{name}_beta_abs", alpha=alpha_abs, beta=beta_abs
        )
        q_j = pm.Deterministic(
            f"{name}_p",
            parent_prob * beta_present + (1 - parent_prob) * beta_absent,
        )

        if evidencer["type"].lower() == "indicator":
            self._add_indicator_ordinal_likelihood(
                evidencer, name, q_j, ancestor_path
            )
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
        """Add marginalised ordinal likelihood and posterior P(z_j=1) for one indicator.

        If no valid observations exist, exposes pz1 = q_j (prior only).
        """
        key = node_key(ancestor_path, evidencer["name"])
        obs_data = self.ordinal_data.observations.get(key, [])

        if not obs_data:
            self.logger.debug(f"No observations for {key} -- pz1 = q_j")
            pm.Deterministic(f"{name}_pz1", q_j)
            return

        ratings = np.array([r for _, r in obs_data], dtype=np.int64)
        expert_indices = np.array([e for e, _ in obs_data], dtype=np.int64)

        # Log-likelihood under z=0 (indicator absent) and z=1 (present)
        ll_z0 = pt_ordinal_logp(
            ratings, expert_indices, self.kappa, self.b, pt.constant(0.0)
        )
        ll_z1 = pt_ordinal_logp(
            ratings, expert_indices, self.kappa, self.b, self.a
        )

        # Marginalised likelihood: log[(1-q)*L0 + q*L1]
        log_mix = pt.logaddexp(
            pt.log(1 - q_j + 1e-12) + ll_z0,
            pt.log(q_j + 1e-12) + ll_z1,
        )
        pm.Potential(f"{name}_ordinal_lik", log_mix)

        # Posterior P(z_j=1 | ratings, q_j, theta) via sigmoid form:
        #   sigmoid(logit(q_j) + ll_z1 - ll_z0)
        logit_q = pt.log(q_j + 1e-12) - pt.log(1 - q_j + 1e-12)
        pz1 = pt.sigmoid(logit_q + ll_z1 - ll_z0)
        pm.Deterministic(f"{name}_pz1", pz1)

    def _add_evidencer(
        self,
        parent_prob: pt.TensorVariable,
        evidencer: Dict,
        ancestor_path: Tuple[str, ...],
    ) -> None:
        """Add a single evidencer node to the model."""
        current_path = ancestor_path + (evidencer["name"],)

        try:
            var = self._create_node_variable(
                evidencer, parent_prob, ancestor_path
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to add evidencer {evidencer['name']}: {e}"
            )
            return

        # Recurse into children for features / subfeatures
        if (
            evidencer["type"].lower() in {"feature", "subfeature"}
            and var is not None
        ):
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
            self.a = pm.HalfNormal("a", sigma=2.0)

            if n_experts > 1:
                b_free = pm.Normal(
                    "b_free", mu=0.0, sigma=2.0, shape=n_experts - 1
                )
                self.b = pt.concatenate([pt.zeros(1), b_free])
            else:
                self.b = pt.zeros(1)

            self.kappa = pm.Normal(
                "kappa",
                mu=0.0,
                sigma=2.0,
                shape=K - 1,
                transform=pm.distributions.transforms.ordered,
                initval=np.linspace(-1.5, 1.5, K - 1),
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

    def __init__(
        self, config: ModelConfig, node_to_varname: Dict[str, str]
    ):
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger = setup_logging("INFO")
    logger.info("Starting DCM ordinal model analysis")

    config = ModelConfig()

    # Load data (local cache, no API)
    all_data = load_data(config)
    stance_data = next(
        (item for item in all_data if item["name"] == config.TARGET_STANCE),
        None,
    )
    if not stance_data:
        raise ValueError(f"Stance not found: {config.TARGET_STANCE}")

    # Preprocess observations
    processor = OrdinalDataProcessor(config)
    processor.process(stance_data, config.TARGET_SYSTEM)

    # Build model
    evidence_proc = EvidenceProcessor(config)
    builder = BayesianModelBuilder(config, evidence_proc, processor)
    model = builder.build_model(stance_data)

    # Sample
    idata = builder.sample(model)

    # Report
    results = ResultsManager(config, builder.node_to_varname)
    print(f"\n{'=' * 60}")
    print(f"System: {config.TARGET_SYSTEM}")
    print(f"Stance: {config.TARGET_STANCE}")
    print(f"{'=' * 60}\n")
    results.summarise(idata, stance_data)

    # Observation model parameters
    print(f"\n{'=' * 60}")
    print("Observation model parameters")
    print(f"{'=' * 60}")
    print(f"  a (discrimination): {float(idata.posterior['a'].mean()):.3f}")
    n_experts = len(processor.expert_names)
    if n_experts > 1:
        b_free = idata.posterior["b_free"].mean(dim=("chain", "draw")).values
        print(
            f"  b (expert shifts):  "
            f"[0.000 (anchor: {processor.anchor_expert}), "
            + ", ".join(f"{v:.3f}" for v in b_free)
            + "]"
        )
    kappa_vals = idata.posterior["kappa"].mean(dim=("chain", "draw")).values
    print(
        f"  kappa (cutpoints):  "
        f"[{', '.join(f'{v:.3f}' for v in kappa_vals)}]"
    )

    logger.info("Analysis complete")


if __name__ == "__main__":
    main()
