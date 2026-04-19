"""Companion workflow for the three-state latent indicator branch.

Packages the Stage 0–5a helpers that surround the validated joint GWT
baseline when the three-state leaf model is engaged. Structured to mirror
``gwt_hierarchical_cutpoints_analysis.py`` and
``gwt_reference_recovery_analysis.py`` so notebook 16 can drive the full
comparison via a small number of function calls.

Stages packaged here
--------------------
0.  Read-only q_j regime inspection for focus cells (wrapper over the
    PPC helpers in ``dcm_ppc``).
1.  NumPy-only predictive-shape sweep (binary vs three-state leaf) at
    fixed (a, kappa), with signed tail and middle-mass curves.
1b. Frozen-theta counterfactual PPC: use a binary baseline's posterior
    draws of (a, kappa, q_j) but evaluate the three-state predictive.
2.  Response-style synthetic data generator (the "required gate" in
    Stage 2 of the plan): draws ratings with expert-specific cutpoint
    jitter and scale so the three-state model should *not* appear to
    "fix" that misspecified process.
5a. Fit drivers for binary and three-state real-data GWT comparison
    plus the emission-separation L1 identification diagnostic.

Stage 5b (Beta-latent quadrature fit + PSIS-LOO comparison) is
intentionally not implemented in this module yet — it is sequenced
after Stage 5a in the plan and is explicitly non-blocking for the
Stage 5a write-up.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
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
from dcm_ppc import (
    _config_state_model,
    _draw_cell_predictive_stats,
    _expert_component_predictives,
    _extract_component_weights_for_indicator,
    _extract_obs_layer_draws,
    _sample_draw_indices,
    classify_pz1_regime,
    inspect_pz1_for_cell,
)

# ---------------------------------------------------------------------------
# Constants mirroring the reference-recovery module
# ---------------------------------------------------------------------------

STANCE = "Global Workspace Theory"
SYSTEM_CONFIGS_VALIDATED = [
    ("Human", 0.999),
    ("Chicken", None),
    ("2024 Leading Chat LLMs", None),
    ("ELIZA", 0.001),
]
SYSTEM_DISPLAY = {
    "Human": "Human",
    "Chicken": "Chicken",
    "2024 Leading Chat LLMs": "LLMs",
    "ELIZA": "ELIZA",
}
SYSTEM_ORDER = ["Human", "Chicken", "2024 Leading Chat LLMs", "ELIZA"]

# Focus cells that drove the week-6 diagnosis; reused across Stage 0 inspections.
FOCUS_KEYS: List[Tuple[str, str]] = [
    ("Derek Shiller", "Human"),
    ("Derek Shiller", "ELIZA"),
    ("Rachael Miller", "Chicken"),
    ("Luhan Mikaelson", "2024 Leading Chat LLMs"),
]

DEFAULT_FIT_OVERRIDES: Dict[str, Any] = {
    "NUM_SAMPLES": 2000,
    "NUM_TUNE": 1500,
    "NUM_CHAINS": 4,
    "TARGET_ACCEPT": 0.9,
}
DEFAULT_SMOKE_FIT_OVERRIDES: Dict[str, Any] = {
    "NUM_SAMPLES": 120,
    "NUM_TUNE": 150,
    "NUM_CHAINS": 2,
    "TARGET_ACCEPT": 0.9,
}


# ---------------------------------------------------------------------------
# Stage 0: q_j regime inspection for focus cells
# ---------------------------------------------------------------------------


def regime_rows_for_focus_cells(
    idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
    focus_cells: Sequence[Tuple[str, str]] = tuple(FOCUS_KEYS),
    low: float = 0.1,
    high: float = 0.9,
) -> List[Dict[str, Any]]:
    """Flatten ``inspect_pz1_for_cell`` + ``classify_pz1_regime`` across focus cells.

    Returns one row per (focus cell, indicator) with cell metadata, the
    expert × system observation count, the posterior median / 94% CI
    of ``pz1`` (binary) or ``expected_z`` (three-state), and a
    ``regime`` label in {"near_0", "moderate", "near_1"}.
    """
    rows: List[Dict[str, Any]] = []
    for expert_name, system_name in focus_cells:
        cell_rows = inspect_pz1_for_cell(
            idata, builder, processor, expert_name, system_name
        )
        if not cell_rows:
            continue
        summary = classify_pz1_regime(cell_rows, low=low, high=high)
        for row in cell_rows:
            median = row["median"]
            if median < low:
                regime = "near_0"
            elif median > high:
                regime = "near_1"
            else:
                regime = "moderate"
            rows.append(
                {
                    "expert": expert_name,
                    "system": system_name,
                    "indicator": row["indicator"],
                    "n_obs_by_expert": row["n_obs_by_expert"],
                    "median": median,
                    "p03": row["p03"],
                    "p97": row["p97"],
                    "regime": regime,
                    "cell_frac_moderate": summary["frac_moderate"],
                }
            )
    return rows


def format_regime_table(rows: Sequence[Dict[str, Any]]) -> str:
    """Format ``regime_rows_for_focus_cells`` output as a compact text table."""
    header = (
        f"{'Expert':<18s} {'System':<24s} {'Indicator':<38s} "
        f"{'n':>3s}  {'med':>5s}  {'3%':>5s}  {'97%':>5s}  {'regime':<9s}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        sys_label = SYSTEM_DISPLAY.get(row["system"], row["system"])
        lines.append(
            f"{row['expert']:<18s} {sys_label:<24s} "
            f"{row['indicator'][:38]:<38s} "
            f"{row['n_obs_by_expert']:>3d}  "
            f"{row['median']:>5.2f}  {row['p03']:>5.2f}  {row['p97']:>5.2f}  "
            f"{row['regime']:<9s}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage 1: NumPy-only predictive-shape check
# ---------------------------------------------------------------------------


def ordered_probit_probs_np(kappa: np.ndarray, eta: float) -> np.ndarray:
    """Ordered-probit category probabilities at scalar eta (NumPy)."""
    cuts = np.concatenate([[-np.inf], kappa, [np.inf]])
    return np.diff(norm.cdf(cuts - eta))


def binary_leaf_category_probs(
    q: float, a: float, kappa: np.ndarray
) -> np.ndarray:
    """Binary leaf predictive category probabilities for one q."""
    return (1.0 - q) * ordered_probit_probs_np(kappa, 0.0) + q * ordered_probit_probs_np(
        kappa, a
    )


def three_state_leaf_category_probs(
    q: float, a: float, kappa: np.ndarray
) -> np.ndarray:
    """Three-state leaf predictive category probabilities for one q."""
    w0 = (1.0 - q) ** 2
    w_mid = 2.0 * q * (1.0 - q)
    w1 = q ** 2
    return (
        w0 * ordered_probit_probs_np(kappa, 0.0)
        + w_mid * ordered_probit_probs_np(kappa, a * 0.5)
        + w1 * ordered_probit_probs_np(kappa, a)
    )


def _mid_band_indices(K: int) -> np.ndarray:
    """Indices of the middle band {K//2-1, K//2, K//2+1}; for K=7 -> {2, 3, 4}."""
    return np.arange(K // 2 - 1, K // 2 + 2)


@dataclass(frozen=True)
class ShapeSweep:
    """Stage 1 output: predictive shape summaries across a grid of q values."""

    q_grid: np.ndarray
    a: float
    kappa: np.ndarray
    binary_probs: np.ndarray  # (n_q, K)
    three_state_probs: np.ndarray  # (n_q, K)

    @property
    def K(self) -> int:
        return self.binary_probs.shape[1]

    def tail_curves(self) -> Dict[str, np.ndarray]:
        """Signed tail and middle-mass curves for both leaf models."""
        mid = _mid_band_indices(self.K)
        return {
            "binary_left": self.binary_probs[:, 0],
            "binary_right": self.binary_probs[:, -1],
            "binary_mid": self.binary_probs[:, mid].sum(axis=1),
            "three_state_left": self.three_state_probs[:, 0],
            "three_state_right": self.three_state_probs[:, -1],
            "three_state_mid": self.three_state_probs[:, mid].sum(axis=1),
        }


def abstract_shape_sweep(
    a: float,
    kappa: np.ndarray,
    q_grid: Optional[np.ndarray] = None,
) -> ShapeSweep:
    """Evaluate binary and three-state predictive categories over q.

    Defaults to a dense grid plus the plan's focus values {0.01, 0.1,
    0.25, 0.5, 0.75, 0.9, 0.99}. Used in Stage 1 to inspect whether the
    three-state leaf can produce strong extreme mass at small / large q
    and middle-heavy predictions at moderate q.
    """
    if q_grid is None:
        q_grid = np.array(
            [0.01, 0.05, 0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 0.95, 0.99]
        )
    binary = np.stack([binary_leaf_category_probs(q, a, kappa) for q in q_grid])
    three = np.stack(
        [three_state_leaf_category_probs(q, a, kappa) for q in q_grid]
    )
    return ShapeSweep(
        q_grid=np.asarray(q_grid, dtype=float),
        a=float(a),
        kappa=np.asarray(kappa, dtype=float),
        binary_probs=binary,
        three_state_probs=three,
    )


def posterior_mean_a_kappa(idata: Any) -> Tuple[float, np.ndarray]:
    """Convenience: pull posterior-mean ``a`` and ``kappa`` from an idata."""
    a = float(np.asarray(idata.posterior["a"].values).mean())
    kappa = np.asarray(idata.posterior["kappa"].values).mean(axis=(0, 1))
    return a, np.asarray(kappa, dtype=float)


# ---------------------------------------------------------------------------
# Stage 1b: Frozen-theta counterfactual PPC
# ---------------------------------------------------------------------------


def frozen_theta_three_state_ppc(
    idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
    n_draws: int = 500,
    seed: int = 42,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """PPC using a BINARY fit's posterior but the three-state predictive mixture.

    This isolates the effect of switching the likelihood family from the
    effect of the posterior reconfiguring around the new family. Reads
    tree-implied ``..._p`` from the binary posterior, maps it to
    three-state prior weights ((1-q)^2, 2q(1-q), q^2), and evaluates the
    three-component emission mixture at ``eta ∈ {0, a/2, a}``.

    Raises
    ------
    ValueError if the supplied ``idata``/``builder`` pair is not a
    binary-state fit (guard against accidentally running on a
    three-state fit, which would make the diagnostic meaningless).
    """
    state_model = _config_state_model(builder)
    if state_model != "binary":
        raise ValueError(
            f"frozen_theta_three_state_ppc expects a binary source fit; "
            f"got state_model={state_model!r}"
        )

    K = builder.config.N_CATEGORIES
    post = idata.posterior
    n_experts = len(processor.expert_names)
    S_total = np.asarray(post["a"].values).reshape(-1).shape[0]
    idx = _sample_draw_indices(S_total, n_draws, seed)
    a_draws, b_draws, kappa_draws, kappa_by_expert_draws = _extract_obs_layer_draws(
        post, idx, n_experts, K
    )
    rng = np.random.default_rng(seed)

    # For each indicator, compute three-state prior weights from binary posterior's q.
    # Reuse the PPC helper with state_model="three_state" and source="tree_implied" to
    # produce correctly-shaped weight arrays.
    from collections import defaultdict

    buckets: Dict[Tuple[int, str], List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for sys_name, sys_obs in processor.system_observations.items():
        sp = builder._sys_prefix(sys_name)
        for nkey, obs_list in sys_obs.items():
            varname = builder.node_to_varname.get(nkey)
            if varname is None:
                continue
            weights = _extract_component_weights_for_indicator(
                post,
                idx,
                sp,
                varname,
                state_model="three_state",
                indicator_prob_source="tree_implied",
            )
            if weights is None:
                continue
            for expert_idx, rating in obs_list:
                buckets[(expert_idx, sys_name)].append((int(rating), weights))

    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (e, sys_name), entries in buckets.items():
        obs = np.asarray([r for r, _ in entries], dtype=int)
        weights_stack = np.stack([w for _, w in entries], axis=1)
        pk_components = _expert_component_predictives(
            a_draws,
            b_draws,
            kappa_draws,
            kappa_by_expert_draws,
            e,
            K,
            state_model="three_state",
        )
        results[(processor.expert_names[e], sys_name)] = _draw_cell_predictive_stats(
            weights_stack, pk_components, obs, rng, K
        )
    return results


# ---------------------------------------------------------------------------
# Stage 2: Response-style synthetic data generator
# ---------------------------------------------------------------------------


_ORDINAL_TO_PROB = [0.00, 0.10, 0.30, 0.50, 0.70, 0.90, 1.00]


def generate_response_style_synthetic_tree(
    *,
    n_features: int = 3,
    indicators_per_feature: int = 4,
    n_experts: int = 4,
    true_a: float = 1.8,
    cutpoint_loc_sigma: float = 0.35,
    cutpoint_scale_sigma: float = 0.2,
    generator_model: str = "binary",
    seed: int = 123,
) -> Tuple[Dict[str, Any], Dict[str, int], str, Dict[str, Any]]:
    """Synthetic GWT-shaped tree with expert-specific cutpoint heterogeneity.

    Parameters
    ----------
    generator_model : {"binary", "three_state"}
        Distribution of the latent indicator state used to draw ratings.
        Under "binary" every indicator is purely present or absent; under
        "three_state" z_j is in {0, 0.5, 1} with midpoint frequency
        2q(1-q) (but we fix a single true q per indicator for simplicity).
    cutpoint_loc_sigma, cutpoint_scale_sigma : float
        Expert-specific jitter on a shared population cutpoint vector:
        ``kappa_e = loc_e + scale_e * kappa_pop`` with ``loc_e ~ N(0, loc_sigma)``
        and ``scale_e = exp(N(0, scale_sigma))``. This is the "response
        style" heterogeneity that the Stage 2 gate asks whether the
        three-state model silently absorbs.
    """
    rng = np.random.default_rng(seed)
    K = 7
    kappa_pop = np.array([-1.5, -0.8, -0.1, 0.4, 1.0, 1.6])
    expert_locs = rng.normal(0.0, cutpoint_loc_sigma, size=n_experts)
    expert_scales = np.exp(rng.normal(0.0, cutpoint_scale_sigma, size=n_experts))
    # First expert is anchored to the population values for identifiability.
    expert_locs[0] = 0.0
    expert_scales[0] = 1.0
    expert_kappas = expert_locs[:, None] + expert_scales[:, None] * kappa_pop[None, :]
    expert_names = [f"Expert_{i}" for i in range(n_experts)]
    system_name = "SyntheticResponseStyle"

    true_states: Dict[str, int] = {}

    features = []
    for f_idx in range(n_features):
        indicators = []
        for ind_idx in range(indicators_per_feature):
            ind_name = f"Indicator_F{f_idx}_I{ind_idx}"
            # Target q for this indicator: spread across {0.1, 0.3, 0.5, 0.7, 0.9}
            target_q = 0.1 + 0.2 * ((f_idx * indicators_per_feature + ind_idx) % 5)
            if generator_model == "binary":
                z = int(rng.random() < target_q)
                z_eff = [float(z)]
            elif generator_model == "three_state":
                m = int(rng.binomial(2, target_q))
                z_eff = [m / 2.0]
            else:
                raise ValueError(f"unknown generator_model: {generator_model!r}")
            true_states[ind_name] = float(z_eff[0])

            values, names = [], []
            for e in range(n_experts):
                eta = true_a * z_eff[0]
                # Per-expert cutpoints
                kappa_e = expert_kappas[e]
                probs = np.diff(
                    norm.cdf(np.concatenate([[-np.inf], kappa_e, [np.inf]]) - eta)
                )
                r = rng.choice(K, p=probs)
                values.append(str(_ORDINAL_TO_PROB[r]))
                names.append(expert_names[e])

            indicators.append(
                {
                    "name": ind_name,
                    "type": "Indicator",
                    "support": "moderate support",
                    "demandingness": "neutral",
                    "observations": {
                        system_name: {"values": values, "names": names}
                    },
                    "observation_stats": {
                        system_name: {
                            "average": str(
                                np.mean([float(v) for v in values])
                            ),
                            "std": 0,
                        }
                    },
                }
            )

        features.append(
            {
                "name": f"Feature_{f_idx}",
                "type": "Feature",
                "support": "moderate support",
                "demandingness": "neutral",
                "evidencers": indicators,
            }
        )

    tree = {
        "name": "Global Workspace Theory",
        "type": "Stance",
        "evidencers": features,
    }
    metadata = {
        "true_a": float(true_a),
        "kappa_pop": kappa_pop,
        "expert_locs": expert_locs,
        "expert_scales": expert_scales,
        "expert_kappas": expert_kappas,
        "generator_model": generator_model,
    }
    return tree, true_states, system_name, metadata


# ---------------------------------------------------------------------------
# Stage 5a: Emission-separation L1 diagnostic + fit drivers
# ---------------------------------------------------------------------------


def emission_separation_l1(a: float, kappa: np.ndarray) -> Dict[str, float]:
    """L1 distances between emission vectors g(0), g(a/2), g(a)."""
    g0 = ordered_probit_probs_np(kappa, 0.0)
    g_half = ordered_probit_probs_np(kappa, a * 0.5)
    g1 = ordered_probit_probs_np(kappa, a)
    return {
        "l1_g0_g_half": float(np.sum(np.abs(g0 - g_half))),
        "l1_g_half_g1": float(np.sum(np.abs(g_half - g1))),
        "l1_g0_g1": float(np.sum(np.abs(g0 - g1))),
    }


def emission_separation_l1_from_posterior(
    idata: Any, threshold: float = 0.1
) -> Dict[str, Any]:
    """Posterior-mean L1 separations plus a pass/fail against ``threshold``.

    The plan's acceptance criterion flags a three-state fit as effectively
    binary if the middle emission lies too close to either endpoint; the
    default threshold is 0.1 as discussed in the plan.
    """
    a, kappa = posterior_mean_a_kappa(idata)
    l1 = emission_separation_l1(a, kappa)
    return {
        **l1,
        "threshold": threshold,
        "middle_distinct_low": l1["l1_g0_g_half"] >= threshold,
        "middle_distinct_high": l1["l1_g_half_g1"] >= threshold,
    }


def get_gwt_stance_data(config: Optional[ModelConfig] = None) -> Dict[str, Any]:
    """Load the GWT stance tree from the local cache (same helper as recovery module)."""
    cfg = config or ModelConfig()
    return next(item for item in load_data(cfg) if item["name"] == STANCE)


def _build_validated_config(
    state_model: str,
    fit_overrides: Optional[Dict[str, Any]] = None,
) -> ModelConfig:
    """Construct the validated joint-GWT config with a chosen state model."""
    common = dict(DEFAULT_FIT_OVERRIDES)
    if fit_overrides:
        common.update(fit_overrides)
    return ModelConfig(
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        INDICATOR_STATE_MODEL=state_model,  # type: ignore[arg-type]
        **common,
    )


def fit_gwt(
    state_model: str = "binary",
    fit_overrides: Optional[Dict[str, Any]] = None,
    system_configs: Optional[Sequence[Tuple[str, Optional[float]]]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fit the validated joint GWT configuration with the given state model.

    Defaults to the anchored baseline config (Human=0.999, Chicken free,
    LLMs free, ELIZA=0.001; no expert shifts; no hierarchical cutpoints).
    """
    system_configs = list(system_configs or SYSTEM_CONFIGS_VALIDATED)
    config = _build_validated_config(state_model, fit_overrides)
    stance_data = get_gwt_stance_data(config)
    if verbose:
        print(
            f"Fitting GWT state_model={state_model!r} | "
            f"{len(system_configs)} systems | samples={config.NUM_SAMPLES}"
        )
    t0 = time.time()
    idata, builder, processor = fit_stance_multisystem(
        stance_data, config, system_configs
    )
    elapsed = time.time() - t0
    if verbose:
        print(f"  elapsed_s={elapsed:.1f}")
    summary = summarise_system_posteriors(idata, builder, system_configs)
    return {
        "state_model": state_model,
        "config": config,
        "system_configs": system_configs,
        "stance_data": stance_data,
        "idata": idata,
        "builder": builder,
        "processor": processor,
        "elapsed_s": elapsed,
        "c_summary": summary,
    }


def summarise_system_posteriors(
    idata: Any,
    builder: MultiSystemModelBuilder,
    system_configs: Sequence[Tuple[str, Optional[float]]],
) -> Dict[str, Dict[str, float]]:
    """Posterior median + 94% interval for each system's stance-level C."""
    stance_var = builder.node_to_varname[STANCE]
    summary: Dict[str, Dict[str, float]] = {}
    for system_name, c_fixed in system_configs:
        key = f"{builder._sys_prefix(system_name)}__{stance_var}_C"
        if key in idata.posterior.data_vars:
            draws = np.asarray(idata.posterior[key].values).reshape(-1)
            if np.std(draws) < 1e-12:
                # Fixed anchor: all draws equal to c_fixed
                summary[system_name] = {
                    "median": float(c_fixed) if c_fixed is not None else float(draws[0]),
                    "lo": float(c_fixed) if c_fixed is not None else float(draws[0]),
                    "hi": float(c_fixed) if c_fixed is not None else float(draws[0]),
                    "fixed": True,
                }
            else:
                summary[system_name] = {
                    "median": float(np.median(draws)),
                    "lo": float(np.percentile(draws, 3)),
                    "hi": float(np.percentile(draws, 97)),
                    "fixed": False,
                }
        else:
            summary[system_name] = {
                "median": float(c_fixed) if c_fixed is not None else float("nan"),
                "lo": float(c_fixed) if c_fixed is not None else float("nan"),
                "hi": float(c_fixed) if c_fixed is not None else float("nan"),
                "fixed": True,
            }
    return summary


def compare_c_summaries(
    binary_result: Mapping[str, Any],
    three_state_result: Mapping[str, Any],
    systems: Sequence[str] = ("Chicken", "2024 Leading Chat LLMs"),
) -> Dict[str, Dict[str, float]]:
    """Return posterior medians, 94% intervals, and deltas for key systems."""
    out: Dict[str, Dict[str, float]] = {}
    for sys_name in systems:
        b = binary_result["c_summary"][sys_name]
        t = three_state_result["c_summary"][sys_name]
        out[sys_name] = {
            "binary_median": b["median"],
            "binary_lo": b["lo"],
            "binary_hi": b["hi"],
            "three_state_median": t["median"],
            "three_state_lo": t["lo"],
            "three_state_hi": t["hi"],
            "delta_median": t["median"] - b["median"],
        }
    return out


def format_c_comparison_table(
    comparison: Mapping[str, Mapping[str, float]],
    guardrail: float = 0.05,
) -> str:
    """Compact table: per-system posterior median + binary delta."""
    header = (
        f"{'System':<26s} {'Binary':>22s} {'Three-state':>22s} "
        f"{'Δ':>7s} {'guard':>6s}"
    )
    lines = [header, "-" * len(header)]
    for sys_name, row in comparison.items():
        status = "ok" if abs(row["delta_median"]) <= guardrail else "flag"
        lines.append(
            f"{SYSTEM_DISPLAY.get(sys_name, sys_name):<26s} "
            f"{row['binary_median']:>6.3f} [{row['binary_lo']:.3f}, {row['binary_hi']:.3f}] "
            f"{row['three_state_median']:>6.3f} [{row['three_state_lo']:.3f}, {row['three_state_hi']:.3f}] "
            f"{row['delta_median']:>7.3f} {status:>6s}"
        )
    return "\n".join(lines)


def middle_state_utilisation_per_cell(
    idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
) -> List[Dict[str, Any]]:
    """Posterior expected middle-state probability E[m=1 | ratings] per cell.

    Pre-registered diagnostic in the plan. If ``mean_p_m1`` activates
    uniformly across cells regardless of observed rating concentration,
    the three-state branch is absorbing generic misspecification rather
    than tracking real graded presence.
    """
    state_model = _config_state_model(builder)
    if state_model != "three_state":
        raise ValueError(
            "middle_state_utilisation_per_cell requires a three-state fit; "
            f"got state_model={state_model!r}"
        )
    post = idata.posterior
    rows: List[Dict[str, Any]] = []
    for sys_name, sys_obs in processor.system_observations.items():
        sp = builder._sys_prefix(sys_name)
        for nkey, obs_list in sys_obs.items():
            varname = builder.node_to_varname.get(nkey)
            if varname is None:
                continue
            p_m1_name = f"{sp}__{varname}_p_m1"
            if p_m1_name not in post.data_vars:
                continue
            p_m1 = np.asarray(post[p_m1_name].values).reshape(-1)
            rows.append(
                {
                    "system": sys_name,
                    "indicator": nkey.split(" > ")[-1],
                    "n_obs": len(obs_list),
                    "p_m1_mean": float(p_m1.mean()),
                    "p_m1_median": float(np.median(p_m1)),
                    "p_m1_lo": float(np.percentile(p_m1, 3)),
                    "p_m1_hi": float(np.percentile(p_m1, 97)),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Verbose plotting helpers (matplotlib; imported lazily inside the functions)
# ---------------------------------------------------------------------------


def plot_shape_sweep(
    sweep: ShapeSweep,
    title: Optional[str] = None,
):
    """Plot category probabilities and signed tail curves for both leaf models."""
    import matplotlib.pyplot as plt

    K = sweep.K
    tails = sweep.tail_curves()

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))

    ax = axes[0, 0]
    cats = np.arange(1, K + 1)
    for q_label in (0.01, 0.5, 0.99):
        if q_label in sweep.q_grid:
            i = np.argmin(np.abs(sweep.q_grid - q_label))
            ax.plot(cats, sweep.binary_probs[i], "o-", label=f"binary q={sweep.q_grid[i]:.2f}")
    ax.set_xlabel("rating category")
    ax.set_ylabel("P(r | q)")
    ax.set_title("binary leaf predictive")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for q_label in (0.01, 0.5, 0.99):
        if q_label in sweep.q_grid:
            i = np.argmin(np.abs(sweep.q_grid - q_label))
            ax.plot(
                cats,
                sweep.three_state_probs[i],
                "s--",
                label=f"3-state q={sweep.q_grid[i]:.2f}",
            )
    ax.set_xlabel("rating category")
    ax.set_ylabel("P(r | q)")
    ax.set_title("three-state leaf predictive")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(sweep.q_grid, tails["binary_left"], "o-", label="binary P(r=1)")
    ax.plot(sweep.q_grid, tails["three_state_left"], "s--", label="3-state P(r=1)")
    ax.plot(sweep.q_grid, tails["binary_right"], "o-", color="salmon", label=f"binary P(r={K})")
    ax.plot(
        sweep.q_grid,
        tails["three_state_right"],
        "s--",
        color="darkorange",
        label=f"3-state P(r={K})",
    )
    ax.set_xlabel("q")
    ax.set_ylabel("tail probability")
    ax.set_title("signed tail curves")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(sweep.q_grid, tails["binary_mid"], "o-", label="binary middle-mass")
    ax.plot(sweep.q_grid, tails["three_state_mid"], "s--", label="3-state middle-mass")
    ax.set_xlabel("q")
    ax.set_ylabel(f"P(r in {{{K//2}, {K//2+1}, {K//2+2}}})")
    ax.set_title("middle-mass vs q")
    ax.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96) if title else None)
    return fig, axes
