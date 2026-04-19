"""Reference-system recovery diagnostics for the joint GWT baseline.

This module adds two staged diagnostics around the current validated
multi-system Global Workspace Theory fit:

1. A cheap post-hoc free-C recovery curve that keeps the anchored baseline
   posterior over lower-layer parameters fixed.
2. Stronger one-anchor-at-a-time refits that free either Human or ELIZA while
   keeping the rest of the validated baseline unchanged.

The stage-1 recovery curve intentionally uses the tree-implied indicator
probabilities q_j(c), not the leaf-updated posterior probabilities pz1_j.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import betaln, logsumexp
from scipy.stats import norm

from dcm_model import (
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    MultiSystemModelBuilder,
    fit_stance_multisystem,
    load_data,
    node_key,
)

STANCE = "Global Workspace Theory"
SYSTEM_ORDER = ["Human", "Chicken", "2024 Leading Chat LLMs", "ELIZA"]
SYSTEM_DISPLAY = {
    "Human": "Human",
    "Chicken": "Chicken",
    "2024 Leading Chat LLMs": "LLMs",
    "ELIZA": "ELIZA",
}
ANCHOR_VALUES = {"Human": 0.999, "ELIZA": 0.001}
ANCHORED_SYSTEM_CONFIGS = [
    ("Human", 0.999),
    ("Chicken", None),
    ("2024 Leading Chat LLMs", None),
    ("ELIZA", 0.001),
]
HUMAN_FREE_SYSTEM_CONFIGS = [
    ("Human", None),
    ("Chicken", None),
    ("2024 Leading Chat LLMs", None),
    ("ELIZA", 0.001),
]
ELIZA_FREE_SYSTEM_CONFIGS = [
    ("Human", 0.999),
    ("Chicken", None),
    ("2024 Leading Chat LLMs", None),
    ("ELIZA", None),
]
DEFAULT_FIT_OVERRIDES = {
    "NUM_SAMPLES": 2000,
    "NUM_TUNE": 1500,
    "NUM_CHAINS": 4,
    "TARGET_ACCEPT": 0.9,
}
DEFAULT_SMOKE_FIT_OVERRIDES = {
    "NUM_SAMPLES": 120,
    "NUM_TUNE": 120,
    "NUM_CHAINS": 2,
    "TARGET_ACCEPT": 0.9,
}
DEFAULT_GRID_SIZE = 401
DEFAULT_GRID_EPS = 1e-3
DEFAULT_CHUNK_SIZE = 256


@dataclass(frozen=True)
class IndicatorSpec:
    """Metadata for one indicator in deterministic tree order."""

    node_key: str
    varname: str
    display_name: str


@dataclass(frozen=True)
class IndicatorLogLikelihoodTerms:
    """Per-indicator observation-layer log-likelihood terms for one system."""

    indicators: tuple[IndicatorSpec, ...]
    ll_z0: np.ndarray  # (S, J)
    ll_z1: np.ndarray  # (S, J)
    obs_counts: np.ndarray  # (J,)


@dataclass(frozen=True)
class RecoveryCurveResult:
    """Post-hoc free-C recovery result for one system."""

    system_name: str
    grid: np.ndarray
    likelihood_density: np.ndarray
    prior_density: np.ndarray
    posterior_density: np.ndarray
    log_mean_likelihood: np.ndarray
    likelihood_summary: Dict[str, float]
    posterior_summary: Dict[str, float]
    n_draws: int
    n_indicators: int
    anchor_value: Optional[float]
    prior_alpha: float
    prior_beta: float


def _build_config(
    fit_overrides: Optional[Dict[str, Any]] = None,
) -> ModelConfig:
    """Return the validated baseline config with optional fit overrides."""
    common = dict(DEFAULT_FIT_OVERRIDES)
    if fit_overrides:
        common.update(fit_overrides)
    return ModelConfig(
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        **common,
    )


def get_gwt_stance_data(config: Optional[ModelConfig] = None) -> Dict[str, Any]:
    """Load the GWT stance tree from the local data cache."""
    cfg = config or ModelConfig()
    return next(item for item in load_data(cfg) if item["name"] == STANCE)


def build_metadata_context(
    stance_data: Dict[str, Any],
    config: ModelConfig,
    system_configs: Sequence[Tuple[str, Optional[float]]],
) -> Tuple[MultiSystemModelBuilder, MultiSystemDataProcessor]:
    """Build processor + builder without sampling, for metadata/indexing use."""
    processor = MultiSystemDataProcessor(config)
    processor.process(stance_data, [sys_name for sys_name, _ in system_configs])
    evidence_proc = EvidenceProcessor(config)
    builder = MultiSystemModelBuilder(config, evidence_proc, processor, list(system_configs))
    builder.build_model(stance_data)
    return builder, processor


def validate_current_baseline_idata(idata: Any) -> None:
    """Ensure an idata object matches the validated baseline structure."""
    post = idata.posterior
    if "b_free" in post.data_vars:
        raise ValueError("Expected baseline idata with USE_EXPERT_SHIFTS=False")
    if "kappa_by_expert" in post.data_vars:
        raise ValueError(
            "Expected baseline idata with USE_HIERARCHICAL_EXPERT_CUTPOINTS=False"
        )
    stance_var = "global_workspace_theory"
    human_key = f"human__{stance_var}_C"
    eliza_key = f"eliza__{stance_var}_C"
    chicken_key = f"chicken__{stance_var}_C"
    llm_key = f"2024_leading_chat_llms__{stance_var}_C"
    if human_key in post.data_vars or eliza_key in post.data_vars:
        raise ValueError("Expected anchored baseline idata with Human and ELIZA fixed")
    if chicken_key not in post.data_vars or llm_key not in post.data_vars:
        raise ValueError(
            "Expected anchored baseline idata with Chicken and LLMs free"
        )


def load_anchored_baseline_idata(path: str | Path) -> Any:
    """Load an anchored baseline idata from disk and validate its structure."""
    idata = az.from_netcdf(Path(path))
    validate_current_baseline_idata(idata)
    return idata


def prepare_anchored_baseline(
    idata_path: str | Path | None = None,
    fit_overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fit or load the anchored baseline and rebuild metadata context."""
    config = _build_config(fit_overrides)
    stance_data = get_gwt_stance_data(config)

    if idata_path is None:
        if verbose:
            print("Fitting anchored GWT baseline...")
        t0 = time.time()
        idata, builder, processor = fit_stance_multisystem(
            stance_data,
            config,
            ANCHORED_SYSTEM_CONFIGS,
        )
        elapsed = time.time() - t0
        if verbose:
            print(f"anchored baseline elapsed_s={elapsed:.1f}")
    else:
        if verbose:
            print(f"Loading anchored GWT baseline from {idata_path}")
        idata = load_anchored_baseline_idata(idata_path)
        builder, processor = build_metadata_context(
            stance_data,
            config,
            ANCHORED_SYSTEM_CONFIGS,
        )
        elapsed = None

    return {
        "config": config,
        "stance_data": stance_data,
        "idata": idata,
        "builder": builder,
        "processor": processor,
        "system_configs": list(ANCHORED_SYSTEM_CONFIGS),
        "elapsed_s": elapsed,
        "c_summary": summarise_system_posteriors(idata, builder, ANCHORED_SYSTEM_CONFIGS),
        "diagnostics": diagnostics_overview(idata, builder, ANCHORED_SYSTEM_CONFIGS),
    }


def build_indicator_index(
    stance_data: Dict[str, Any],
    builder: MultiSystemModelBuilder,
) -> List[IndicatorSpec]:
    """Return indicators in deterministic tree order."""
    indicators: List[IndicatorSpec] = []
    root_path = (stance_data["name"],)

    def walk(node: Dict[str, Any], ancestor_path: Tuple[str, ...]) -> None:
        current_path = ancestor_path + (node["name"],)
        if node.get("type", "").lower() == "indicator":
            key = node_key(ancestor_path, node["name"])
            varname = builder.node_to_varname.get(key)
            if varname is None:
                raise KeyError(f"Missing varname for indicator {key!r}")
            indicators.append(
                IndicatorSpec(node_key=key, varname=varname, display_name=node["name"])
            )
            return
        for child in node.get("evidencers", []):
            walk(child, current_path)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path)
    return indicators


def _ordered_probit_probs(
    kappa: np.ndarray,
    eta: np.ndarray,
    K: int,
) -> np.ndarray:
    """Ordered-probit category probabilities for draw-wise parameters."""
    cum = norm.cdf(kappa - eta[:, None])
    zeros = np.zeros((cum.shape[0], 1))
    ones = np.ones((cum.shape[0], 1))
    probs = np.diff(np.concatenate([zeros, cum, ones], axis=1), axis=1)
    return np.clip(probs, 1e-12, 1.0)


def _extract_obs_layer_draws(
    post: Any,
    n_experts: int,
    K: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract shared observation-layer draws in flattened draw order."""
    a_draws = np.asarray(post["a"].values).reshape(-1)
    n_draws = a_draws.shape[0]

    if "b_free" in post.data_vars:
        b_free = np.asarray(post["b_free"].values).reshape(-1, n_experts - 1)
        b_draws = np.concatenate([np.zeros((n_draws, 1)), b_free], axis=1)
    else:
        b_draws = np.zeros((n_draws, n_experts))

    kappa_draws = np.asarray(post["kappa"].values).reshape(-1, K - 1)
    kappa_by_expert_draws: Optional[np.ndarray]
    if "kappa_by_expert" in post.data_vars:
        kappa_by_expert_draws = np.asarray(post["kappa_by_expert"].values).reshape(
            -1, n_experts, K - 1
        )
    else:
        kappa_by_expert_draws = None

    return a_draws, b_draws, kappa_draws, kappa_by_expert_draws


def _indicator_loglik_for_draws(
    ratings: np.ndarray,
    expert_idx: np.ndarray,
    a_draws: np.ndarray,
    b_draws: np.ndarray,
    kappa_draws: np.ndarray,
    K: int,
    kappa_by_expert_draws: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute draw-wise log-likelihood terms under z=0 and z=1."""
    n_draws = a_draws.shape[0]
    ll_z0 = np.zeros(n_draws)
    ll_z1 = np.zeros(n_draws)

    for rating, expert in zip(ratings, expert_idx):
        if kappa_by_expert_draws is None:
            kappa_obs = kappa_draws
        else:
            kappa_obs = kappa_by_expert_draws[:, expert, :]

        eta_z0 = b_draws[:, expert]
        eta_z1 = eta_z0 + a_draws
        probs_z0 = _ordered_probit_probs(kappa_obs, eta_z0, K)
        probs_z1 = _ordered_probit_probs(kappa_obs, eta_z1, K)
        ll_z0 += np.log(probs_z0[:, rating])
        ll_z1 += np.log(probs_z1[:, rating])

    return ll_z0, ll_z1


def extract_system_indicator_log_lik_terms(
    idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
    system_name: str,
    stance_data: Dict[str, Any],
    indicator_index: Optional[Sequence[IndicatorSpec]] = None,
) -> IndicatorLogLikelihoodTerms:
    """Extract draw-wise L0_j / L1_j terms for one system in tree order."""
    indicators = list(indicator_index or build_indicator_index(stance_data, builder))
    obs_by_key = processor.system_observations.get(system_name, {})
    K = builder.config.N_CATEGORIES
    n_experts = len(processor.expert_names)
    a_draws, b_draws, kappa_draws, kappa_by_expert_draws = _extract_obs_layer_draws(
        idata.posterior,
        n_experts,
        K,
    )
    selected: List[IndicatorSpec] = []
    ll0_cols: List[np.ndarray] = []
    ll1_cols: List[np.ndarray] = []
    obs_counts: List[int] = []

    for spec in indicators:
        obs_list = obs_by_key.get(spec.node_key, [])
        if not obs_list:
            continue
        ratings = np.asarray([rating for _, rating in obs_list], dtype=np.int64)
        expert_idx = np.asarray([expert for expert, _ in obs_list], dtype=np.int64)
        ll_z0, ll_z1 = _indicator_loglik_for_draws(
            ratings,
            expert_idx,
            a_draws,
            b_draws,
            kappa_draws,
            K,
            kappa_by_expert_draws=kappa_by_expert_draws,
        )
        selected.append(spec)
        ll0_cols.append(ll_z0)
        ll1_cols.append(ll_z1)
        obs_counts.append(len(obs_list))

    if not selected:
        raise ValueError(f"No observed indicators found for system {system_name!r}")

    ll0 = np.column_stack(ll0_cols)
    ll1 = np.column_stack(ll1_cols)
    return IndicatorLogLikelihoodTerms(
        indicators=tuple(selected),
        ll_z0=ll0,
        ll_z1=ll1,
        obs_counts=np.asarray(obs_counts, dtype=np.int64),
    )


def extract_beta_draws_by_node(
    idata: Any,
    builder: MultiSystemModelBuilder,
    stance_data: Dict[str, Any],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Return draw arrays for each shared beta parameter keyed by node_key."""
    post = idata.posterior
    beta_pres_by_key: Dict[str, np.ndarray] = {}
    beta_abs_by_key: Dict[str, np.ndarray] = {}
    root_path = (stance_data["name"],)

    def walk(node: Dict[str, Any], ancestor_path: Tuple[str, ...]) -> None:
        current_path = ancestor_path + (node["name"],)
        key = node_key(ancestor_path, node["name"])
        varname = builder.node_to_varname.get(key)
        if varname is None:
            raise KeyError(f"Missing varname for node {key!r}")
        beta_pres_by_key[key] = np.asarray(post[f"{varname}_beta_pres"].values).reshape(-1)
        beta_abs_by_key[key] = np.asarray(post[f"{varname}_beta_abs"].values).reshape(-1)
        for child in node.get("evidencers", []):
            walk(child, current_path)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path)
    return beta_pres_by_key, beta_abs_by_key


def propagate_affine_indicator_coefficients_from_draws(
    stance_data: Dict[str, Any],
    node_to_varname: Mapping[str, str],
    beta_pres_by_key: Mapping[str, np.ndarray],
    beta_abs_by_key: Mapping[str, np.ndarray],
    indicator_index: Optional[Sequence[IndicatorSpec]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[IndicatorSpec]]:
    """Propagate q_j(c) = intercept_j + slope_j * c through the tree."""
    indicators = list(indicator_index or [])
    if not indicators:
        root_path = (stance_data["name"],)
        tmp: List[IndicatorSpec] = []

        def gather(node: Dict[str, Any], ancestor_path: Tuple[str, ...]) -> None:
            current_path = ancestor_path + (node["name"],)
            if node.get("type", "").lower() == "indicator":
                key = node_key(ancestor_path, node["name"])
                tmp.append(
                    IndicatorSpec(
                        node_key=key,
                        varname=node_to_varname[key],
                        display_name=node["name"],
                    )
                )
                return
            for child in node.get("evidencers", []):
                gather(child, current_path)

        for child in stance_data.get("evidencers", []):
            gather(child, root_path)
        indicators = tmp

    indicator_keys = {spec.node_key for spec in indicators}
    sample_key = next(iter(beta_pres_by_key))
    n_draws = beta_pres_by_key[sample_key].shape[0]
    coeffs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    root_path = (stance_data["name"],)

    def walk(
        node: Dict[str, Any],
        ancestor_path: Tuple[str, ...],
        intercept_parent: np.ndarray,
        slope_parent: np.ndarray,
    ) -> None:
        current_path = ancestor_path + (node["name"],)
        key = node_key(ancestor_path, node["name"])
        beta_pres = beta_pres_by_key[key]
        beta_abs = beta_abs_by_key[key]
        delta = beta_pres - beta_abs
        intercept_child = beta_abs + intercept_parent * delta
        slope_child = slope_parent * delta

        if node.get("type", "").lower() == "indicator":
            if key in indicator_keys:
                coeffs[key] = (intercept_child, slope_child)
            return

        for child in node.get("evidencers", []):
            walk(child, current_path, intercept_child, slope_child)

    root_intercept = np.zeros(n_draws)
    root_slope = np.ones(n_draws)
    for child in stance_data.get("evidencers", []):
        walk(child, root_path, root_intercept, root_slope)

    intercepts = np.column_stack([coeffs[spec.node_key][0] for spec in indicators])
    slopes = np.column_stack([coeffs[spec.node_key][1] for spec in indicators])
    return intercepts, slopes, indicators


def propagate_affine_indicator_coefficients(
    idata: Any,
    builder: MultiSystemModelBuilder,
    stance_data: Dict[str, Any],
    indicator_index: Optional[Sequence[IndicatorSpec]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[IndicatorSpec]]:
    """Wrapper that extracts beta draws from idata before affine propagation."""
    beta_pres_by_key, beta_abs_by_key = extract_beta_draws_by_node(
        idata,
        builder,
        stance_data,
    )
    return propagate_affine_indicator_coefficients_from_draws(
        stance_data,
        builder.node_to_varname,
        beta_pres_by_key,
        beta_abs_by_key,
        indicator_index=indicator_index,
    )


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """Normalize log-weights into a probability vector."""
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    return weights / weights.sum()


def _beta_logpdf_grid(grid: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Beta log-density evaluated on a clipped grid."""
    return (
        (alpha - 1.0) * np.log(grid)
        + (beta - 1.0) * np.log1p(-grid)
        - betaln(alpha, beta)
    )


def _logmeanexp(x: np.ndarray, axis: int) -> np.ndarray:
    """Stable log-mean-exp along one axis."""
    return logsumexp(x, axis=axis) - np.log(x.shape[axis])


def summarise_grid_density(grid: np.ndarray, density: np.ndarray) -> Dict[str, float]:
    """Return weighted median, interval, mode, and tail probabilities."""
    weights = np.asarray(density, dtype=float)
    weights = weights / weights.sum()
    cdf = np.cumsum(weights)
    summary = {
        "mean": float(np.sum(grid * weights)),
        "median": float(np.interp(0.5, cdf, grid)),
        "lo": float(np.interp(0.03, cdf, grid)),
        "hi": float(np.interp(0.97, cdf, grid)),
        "mode": float(grid[np.argmax(weights)]),
        "p_lt_005": float(weights[grid < 0.05].sum()),
        "p_gt_095": float(weights[grid > 0.95].sum()),
    }
    return summary


def evaluate_recovery_curve_from_terms(
    grid: np.ndarray,
    ll_z0: np.ndarray,
    ll_z1: np.ndarray,
    intercepts: np.ndarray,
    slopes: np.ndarray,
    prior_alpha: float,
    prior_beta: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Evaluate the post-hoc free-C recovery curve from precomputed terms."""
    if ll_z0.shape != ll_z1.shape:
        raise ValueError("ll_z0 and ll_z1 must have matching shapes")
    if intercepts.shape != ll_z0.shape or slopes.shape != ll_z0.shape:
        raise ValueError("Affine coefficients and log-likelihood terms must align")

    n_draws = ll_z0.shape[0]
    chunk_logliks: List[np.ndarray] = []
    for start in range(0, n_draws, chunk_size):
        stop = min(start + chunk_size, n_draws)
        q = intercepts[start:stop, :, None] + slopes[start:stop, :, None] * grid[None, None, :]
        q = np.clip(q, eps, 1.0 - eps)
        log_mix = np.logaddexp(
            np.log1p(-q) + ll_z0[start:stop, :, None],
            np.log(q) + ll_z1[start:stop, :, None],
        )
        chunk_logliks.append(log_mix.sum(axis=1))

    draw_loglik = np.concatenate(chunk_logliks, axis=0)
    log_mean_likelihood = _logmeanexp(draw_loglik, axis=0)
    log_prior = _beta_logpdf_grid(grid, prior_alpha, prior_beta)
    likelihood_density = normalize_log_weights(log_mean_likelihood)
    prior_density = normalize_log_weights(log_prior)
    posterior_density = normalize_log_weights(log_mean_likelihood + log_prior)

    return {
        "grid": grid,
        "log_mean_likelihood": log_mean_likelihood,
        "likelihood_density": likelihood_density,
        "prior_density": prior_density,
        "posterior_density": posterior_density,
        "likelihood_summary": summarise_grid_density(grid, likelihood_density),
        "posterior_summary": summarise_grid_density(grid, posterior_density),
    }


def run_posthoc_reference_recovery(
    idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
    stance_data: Dict[str, Any],
    system_name: str,
    grid_size: int = DEFAULT_GRID_SIZE,
    grid_eps: float = DEFAULT_GRID_EPS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> RecoveryCurveResult:
    """Run the stage-1 post-hoc free-C recovery curve for one system."""
    indicator_index = build_indicator_index(stance_data, builder)
    ll_terms = extract_system_indicator_log_lik_terms(
        idata,
        builder,
        processor,
        system_name,
        stance_data,
        indicator_index=indicator_index,
    )
    intercepts, slopes, affine_indicators = propagate_affine_indicator_coefficients(
        idata,
        builder,
        stance_data,
        indicator_index=ll_terms.indicators,
    )
    if [spec.node_key for spec in affine_indicators] != [
        spec.node_key for spec in ll_terms.indicators
    ]:
        raise ValueError("Affine indicator order does not match log-likelihood order")

    grid = np.linspace(grid_eps, 1.0 - grid_eps, grid_size)
    curve = evaluate_recovery_curve_from_terms(
        grid,
        ll_terms.ll_z0,
        ll_terms.ll_z1,
        intercepts,
        slopes,
        prior_alpha=builder.config.DEFAULT_ALPHA,
        prior_beta=builder.config.DEFAULT_BETA,
        chunk_size=chunk_size,
    )
    return RecoveryCurveResult(
        system_name=system_name,
        grid=curve["grid"],
        likelihood_density=curve["likelihood_density"],
        prior_density=curve["prior_density"],
        posterior_density=curve["posterior_density"],
        log_mean_likelihood=curve["log_mean_likelihood"],
        likelihood_summary=curve["likelihood_summary"],
        posterior_summary=curve["posterior_summary"],
        n_draws=ll_terms.ll_z0.shape[0],
        n_indicators=ll_terms.ll_z0.shape[1],
        anchor_value=ANCHOR_VALUES.get(system_name),
        prior_alpha=float(builder.config.DEFAULT_ALPHA),
        prior_beta=float(builder.config.DEFAULT_BETA),
    )


def summarise_system_posteriors(
    idata: Any,
    builder: MultiSystemModelBuilder,
    system_configs: Sequence[Tuple[str, Optional[float]]],
) -> Dict[str, Dict[str, float]]:
    """Posterior median and 94% interval for each system's stance-level C."""
    stance_var = builder.node_to_varname[STANCE]
    summary: Dict[str, Dict[str, float]] = {}
    for system_name, c_fixed in system_configs:
        key = f"{builder._sys_prefix(system_name)}__{stance_var}_C"
        if key in idata.posterior:
            draws = np.asarray(idata.posterior[key].values).reshape(-1)
            summary[system_name] = {
                "median": float(np.median(draws)),
                "lo": float(np.percentile(draws, 3)),
                "hi": float(np.percentile(draws, 97)),
                "fixed": False,
            }
        else:
            summary[system_name] = {
                "median": float(c_fixed),
                "lo": float(c_fixed),
                "hi": float(c_fixed),
                "fixed": True,
            }
    return summary


def diagnostic_var_names(
    idata: Any,
    builder: MultiSystemModelBuilder,
    system_configs: Sequence[Tuple[str, Optional[float]]],
) -> List[str]:
    """Variables to include in convergence diagnostics for this workflow."""
    names = ["a", "kappa"]
    if "kappa_by_expert" in idata.posterior.data_vars:
        names.append("kappa_by_expert")
    names.extend(
        sorted(
            name
            for name in idata.posterior.data_vars
            if name.endswith("_beta_pres") or name.endswith("_beta_abs")
        )
    )
    stance_var = builder.node_to_varname[STANCE]
    for system_name, c_fixed in system_configs:
        if c_fixed is None:
            c_name = f"{builder._sys_prefix(system_name)}__{stance_var}_C"
            if c_name in idata.posterior.data_vars:
                names.append(c_name)
    return names


def diagnostics_overview(
    idata: Any,
    builder: MultiSystemModelBuilder,
    system_configs: Sequence[Tuple[str, Optional[float]]],
) -> Dict[str, float]:
    """Compact diagnostics summary for one fit."""
    diag = az.summary(
        idata,
        var_names=diagnostic_var_names(idata, builder, system_configs),
        kind="diagnostics",
    )
    return {
        "divergences": int(idata.sample_stats["diverging"].values.sum()),
        "max_rhat": float(diag["r_hat"].max()),
        "min_ess": float(diag["ess_bulk"].min()),
    }


def extract_free_c_draws(
    idata: Any,
    builder: MultiSystemModelBuilder,
    system_name: str,
) -> Optional[np.ndarray]:
    """Return posterior draws for a free system's C, or None if fixed."""
    key = f"{builder._sys_prefix(system_name)}__{builder.node_to_varname[STANCE]}_C"
    if key not in idata.posterior.data_vars:
        return None
    return np.asarray(idata.posterior[key].values).reshape(-1)


def compare_system_medians(
    baseline_summary: Mapping[str, Mapping[str, float]],
    candidate_summary: Mapping[str, Mapping[str, float]],
    systems: Sequence[str] = ("Chicken", "2024 Leading Chat LLMs"),
) -> Dict[str, float]:
    """Return median deltas versus the anchored baseline for selected systems."""
    return {
        system_name: float(candidate_summary[system_name]["median"] - baseline_summary[system_name]["median"])
        for system_name in systems
    }


def fit_reference_model(
    stance_data: Dict[str, Any],
    system_configs: Sequence[Tuple[str, Optional[float]]],
    fit_overrides: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fit one reference-recovery model under the validated baseline structure."""
    config = _build_config(fit_overrides)
    if verbose:
        print(f"Fitting systems={system_configs}")
    t0 = time.time()
    idata, builder, processor = fit_stance_multisystem(
        stance_data,
        config,
        list(system_configs),
    )
    elapsed = time.time() - t0
    summary = summarise_system_posteriors(idata, builder, system_configs)
    diagnostics = diagnostics_overview(idata, builder, system_configs)
    return {
        "config": config,
        "system_configs": list(system_configs),
        "idata": idata,
        "builder": builder,
        "processor": processor,
        "elapsed_s": elapsed,
        "c_summary": summary,
        "diagnostics": diagnostics,
    }


def run_one_anchor_refits(
    stance_data: Optional[Dict[str, Any]] = None,
    fit_overrides: Optional[Dict[str, Any]] = None,
    include_anchored: bool = False,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Run the one-anchor refits, optionally including a fresh anchored fit."""
    stance = stance_data or get_gwt_stance_data(_build_config(fit_overrides))
    runs: Dict[str, Dict[str, Any]] = {}
    if include_anchored:
        runs["anchored"] = fit_reference_model(
            stance,
            ANCHORED_SYSTEM_CONFIGS,
            fit_overrides=fit_overrides,
            verbose=verbose,
        )
    runs["human_free"] = fit_reference_model(
        stance,
        HUMAN_FREE_SYSTEM_CONFIGS,
        fit_overrides=fit_overrides,
        verbose=verbose,
    )
    runs["eliza_free"] = fit_reference_model(
        stance,
        ELIZA_FREE_SYSTEM_CONFIGS,
        fit_overrides=fit_overrides,
        verbose=verbose,
    )
    return runs


def format_system_summary_table(
    summary: Mapping[str, Mapping[str, float]],
    system_configs: Sequence[Tuple[str, Optional[float]]],
) -> str:
    """Format per-system posterior summaries as a compact table."""
    lines = [
        f"{'System':<28s} {'median':>8s} {'94% interval':>20s} {'status':>8s}",
        "-" * 70,
    ]
    for system_name, _ in system_configs:
        row = summary[system_name]
        status = "fixed" if row["fixed"] else "free"
        lines.append(
            f"{SYSTEM_DISPLAY.get(system_name, system_name):<28s} "
            f"{row['median']:>8.3f} "
            f"[{row['lo']:.3f}, {row['hi']:.3f}]".rjust(20)
            + f" {status:>8s}"
        )
    return "\n".join(lines)


def format_refit_delta_table(
    baseline_summary: Mapping[str, Mapping[str, float]],
    candidate_summary: Mapping[str, Mapping[str, float]],
) -> str:
    """Format Chicken / LLM median shifts versus the anchored baseline."""
    deltas = compare_system_medians(baseline_summary, candidate_summary)
    lines = [
        f"{'System':<16s} {'baseline':>10s} {'candidate':>10s} {'delta':>10s}",
        "-" * 50,
    ]
    for system_name in ("Chicken", "2024 Leading Chat LLMs"):
        lines.append(
            f"{SYSTEM_DISPLAY.get(system_name, system_name):<16s} "
            f"{baseline_summary[system_name]['median']:>10.3f} "
            f"{candidate_summary[system_name]['median']:>10.3f} "
            f"{deltas[system_name]:>10.3f}"
        )
    return "\n".join(lines)


def summarise_recovery_curve(result: RecoveryCurveResult) -> str:
    """Return a compact text summary for one post-hoc recovery curve."""
    post = result.posterior_summary
    like = result.likelihood_summary
    tail_label = "P(C > 0.95)" if result.system_name == "Human" else "P(C < 0.05)"
    tail_value = post["p_gt_095"] if result.system_name == "Human" else post["p_lt_005"]
    lines = [
        f"{result.system_name} post-hoc free-C recovery",
        (
            "  likelihood-only: "
            f"median={like['median']:.3f}  "
            f"[{like['lo']:.3f}, {like['hi']:.3f}]  "
            f"mode={like['mode']:.3f}"
        ),
        (
            f"  Beta({result.prior_alpha:.0f}, {result.prior_beta:.0f}) posterior: "
            f"median={post['median']:.3f}  "
            f"[{post['lo']:.3f}, {post['hi']:.3f}]  "
            f"mode={post['mode']:.3f}  "
            f"{tail_label}={tail_value:.3f}"
        ),
    ]
    if result.anchor_value is not None:
        lines.append(f"  anchor={result.anchor_value:.3f}")
    return "\n".join(lines)


def plot_recovery_curve(
    result: RecoveryCurveResult,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """Plot likelihood-only and model-consistent posterior recovery curves."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(
        result.grid,
        result.likelihood_density,
        color="steelblue",
        linewidth=2,
        label="likelihood / uniform-prior posterior",
    )
    ax.plot(
        result.grid,
        result.posterior_density,
        color="darkorange",
        linewidth=2,
        label=f"Beta({int(result.prior_alpha)}, {int(result.prior_beta)}) posterior",
    )
    if result.anchor_value is not None:
        ax.axvline(
            result.anchor_value,
            color="black",
            linestyle="--",
            alpha=0.7,
            label=f"current anchor = {result.anchor_value:.3f}",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("candidate root C")
    ax.set_ylabel("normalised density")
    ax.set_title(title or f"{result.system_name}: post-hoc free-C recovery")
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, ax


def plot_recovery_overlay(
    result: RecoveryCurveResult,
    refit_draws: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """Overlay the stage-1 posterior curve with a one-anchor refit posterior."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(
        result.grid,
        result.posterior_density,
        color="darkorange",
        linewidth=2,
        label="post-hoc recovery posterior",
    )
    if result.anchor_value is not None:
        ax.axvline(
            result.anchor_value,
            color="black",
            linestyle="--",
            alpha=0.7,
            label=f"anchored baseline = {result.anchor_value:.3f}",
        )
    if refit_draws is not None:
        ax.hist(
            refit_draws,
            bins=35,
            density=True,
            histtype="step",
            linewidth=1.8,
            color="forestgreen",
            label="one-anchor refit posterior",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("root C")
    ax.set_ylabel("density")
    ax.set_title(title or f"{result.system_name}: post-hoc vs one-anchor refit")
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, ax


def save_recovery_figures(
    curves: Mapping[str, RecoveryCurveResult],
    refit_results: Optional[Mapping[str, Dict[str, Any]]] = None,
    save_dir: Path = Path("report_figures"),
) -> None:
    """Save the standard Human / ELIZA recovery-curve figures."""
    save_dir.mkdir(exist_ok=True)
    for system_name in ("Human", "ELIZA"):
        if system_name not in curves:
            continue
        curve = curves[system_name]
        stem = system_name.lower()
        plot_recovery_curve(
            curve,
            save_path=save_dir / f"gwt_reference_recovery_{stem}_curve.png",
        )
        refit_draws = None
        if refit_results is not None:
            label = "human_free" if system_name == "Human" else "eliza_free"
            if label in refit_results:
                refit_draws = extract_free_c_draws(
                    refit_results[label]["idata"],
                    refit_results[label]["builder"],
                    system_name,
                )
        plot_recovery_overlay(
            curve,
            refit_draws=refit_draws,
            save_path=save_dir / f"gwt_reference_recovery_{stem}_overlay.png",
        )


def build_interpretation_note(
    curves: Mapping[str, RecoveryCurveResult],
    refit_results: Optional[Mapping[str, Dict[str, Any]]] = None,
    baseline_summary: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> str:
    """Build a short notebook/report interpretation note."""
    lines = [
        "Stage 1 is a favorable-but-not-final check: theta was learned under fixed Human / ELIZA anchors, so failure here would be especially concerning, while success is only reassuring rather than decisive.",
    ]
    for system_name in ("Human", "ELIZA"):
        if system_name not in curves:
            continue
        post = curves[system_name].posterior_summary
        if system_name == "Human":
            tail = f"P(C > 0.95) = {post['p_gt_095']:.3f}"
        else:
            tail = f"P(C < 0.05) = {post['p_lt_005']:.3f}"
        lines.append(
            f"{system_name} post-hoc: median {post['median']:.3f} [{post['lo']:.3f}, {post['hi']:.3f}], {tail}."
        )

    if refit_results is not None and baseline_summary is not None:
        for label, system_name in (("human_free", "Human"), ("eliza_free", "ELIZA")):
            if label not in refit_results:
                continue
            summary = refit_results[label]["c_summary"][system_name]
            lines.append(
                f"{label}: freed {system_name} posterior median {summary['median']:.3f} [{summary['lo']:.3f}, {summary['hi']:.3f}]."
            )
            deltas = compare_system_medians(
                baseline_summary,
                refit_results[label]["c_summary"],
            )
            lines.append(
                f"{label}: Chicken delta {deltas['Chicken']:+.3f}, LLM delta {deltas['2024 Leading Chat LLMs']:+.3f} versus anchored baseline."
            )
    lines.append(
        "Human and ELIZA remain single-expert cells in the current GWT data, so weak recovery is a substantive result to report plainly rather than an automatic implementation failure."
    )
    return "\n".join(lines)
