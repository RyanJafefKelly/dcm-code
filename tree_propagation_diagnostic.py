"""Tree-propagation diagnostic + counterfactual gain screen.

Produces, from the already-persisted anchored binary baseline idata, the
artefacts needed to decide whether the tree's compression of the reference-
anchor gap (mean Delta q_j approx 0.13 vs root gap 0.998) is structural
(label-mapping means) or data-induced (posterior shrinkage under sparse
reference coverage), and to screen a candidate transmission-gain fix
before committing a refit.

Expert anonymisation: expert identities are replaced with role-based codes
(``E_cross``, ``E_llmA/B/C``, ``E_chickenA/B``) in all output artefacts.
The mapping is module-private and is used only for data-frame lookups.

Artefacts
---------
1. Prior-predictive label calibration table.
   For every (support, demandingness) combination in
   ``EvidenceProcessor.get_beta_parameters``: prior means mu_pres, mu_abs,
   transmission coefficient Delta = mu_pres - mu_abs, and one-layer
   propagation at C in {0.001, 0.5, 0.999}.  Also tabulated under a
   symmetric log-odds-gap gain g.

2. Per-indicator posterior propagation table.
   Uses ``propagate_affine_indicator_coefficients_from_draws`` against
   ``binary_anchored.nc``.  For each GWT indicator: posterior median and
   94% interval of alpha_j, delta_j, q_j(0.999), q_j(0.001), Delta q_j;
   plus depth, ordered path (support, demandingness) labels, the
   cross-system rater's observed Human/ELIZA ratings where available, and
   leaf-updated E[z_j].  Sorted by delta_j ascending.

3. Robust structural-vs-data-induced statistic.
   log r_j = log(|delta_j^post| + eps) - log(|delta_j^prior| + eps)
   plus a sign-flip indicator.  Indicators on no-bearing paths
   (|delta_j^prior| < eps) are bucketed separately as structural-zero.

4. Cross-expert-vs-others propagation scatter data.
   q_j(0.999), q_j(0.001), plus a flag for indicators present in the
   cross-system rater's Human/ELIZA cells.  Saved as CSV for notebook
   rendering.

5. Counterfactual gain screen (Stage 2).
   For candidate g values, computes a fixed-emission transformed-tree
   counterfactual under the three-state emission at the focus reference
   cells.  Uses the symmetric log-odds-gap gain

       logit mu_pres^(g) = m + g * h,  logit mu_abs^(g) = m - g * h,
       m = (logit mu_pres + logit mu_abs) / 2,
       h = (logit mu_pres - logit mu_abs) / 2,

   which recovers the paper mapping at g=1 and sharpens the edge's
   discriminativeness on both sides for g>1.  This is NOT a
   prior-predictive check nor an approximate refit; it is a heuristic
   veto.  g* is selected from the label table alone; PPC acts only as a
   falsification screen.

No fits.  No model-code changes.  All outputs derived from
``results/gwt_binary_three_state/binary_anchored.nc`` +
``results/gwt_binary_three_state/three_state_anchored.nc`` (three-state
needed only for the counterfactual screen's emission parameters).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import norm

from dcm_model import (
    EvidenceProcessor,
    ModelConfig,
    MultiSystemModelBuilder,
    MultiSystemDataProcessor,
    node_key,
)
from gwt_reference_recovery_analysis import (
    ANCHORED_SYSTEM_CONFIGS,
    IndicatorSpec,
    build_indicator_index,
    build_metadata_context,
    extract_beta_draws_by_node,
    get_gwt_stance_data,
    propagate_affine_indicator_coefficients_from_draws,
)

BINARY_IDATA_PATH = Path("results/gwt_binary_three_state/binary_anchored.nc")
THREE_STATE_IDATA_PATH = Path("results/gwt_binary_three_state/three_state_anchored.nc")
OUTPUT_DIR = Path("results/gwt_tree_propagation/analysis")

HUMAN_SYSTEM = "Human"
ELIZA_SYSTEM = "ELIZA"

# --- Expert anonymisation ---
# Real names appear in data_cache.json; they are meant to be anonymous.  We
# keep a module-private mapping to role-based codes; real names never leave
# this module in any tabular or markdown output.  ``_CROSS_SYSTEM_EXPERT``
# is the only data-lookup constant that retains the raw string (used inside
# ``processor.expert_to_idx``).  All downstream display uses
# ``anonymise_expert``.
_EXPERT_ANON_MAP: Dict[str, str] = {
    "Rater_B": "E_cross",
    "Rater_A": "E_llmA",
    "Rater_C": "E_llmB",
    "Rater_D": "E_llmC",
    "Hayley chickens": "E_chickenA",
    "Rater_E": "E_chickenB",
}
_CROSS_SYSTEM_EXPERT = "Rater_B"  # data_cache.json lookup key


def anonymise_expert(name: str) -> str:
    """Return the role-based anonymised identifier for an expert name."""
    return _EXPERT_ANON_MAP.get(name, name)


CROSS_SYSTEM_CODE = anonymise_expert(_CROSS_SYSTEM_EXPERT)  # "E_cross"

ANCHOR_HIGH = 0.999
ANCHOR_LOW = 0.001

RATIO_EPS = 1e-3  # regime bucketing threshold for |delta_j^prior|
LOG_EPS = 1e-6    # numerical floor inside log |.|

K_CATEGORIES = 7


# ---------------------------------------------------------------------------
# Symmetric log-odds-gap gain (Stage 2a parameterisation)
# ---------------------------------------------------------------------------


def _logit(p: np.ndarray | float, eps: float = 1e-9) -> np.ndarray | float:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _expit(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def apply_symmetric_gain(
    mu_pres: np.ndarray | float,
    mu_abs: np.ndarray | float,
    gain: float,
) -> Tuple[np.ndarray | float, np.ndarray | float]:
    """Apply the symmetric log-odds-gap gain on prior means.

    At gain=1 this returns (mu_pres, mu_abs) unchanged.  For gain>1 the
    edge's discriminativeness is sharpened on both sides.
    """
    lp = _logit(mu_pres)
    la = _logit(mu_abs)
    m = 0.5 * (lp + la)
    h = 0.5 * (lp - la)
    return _expit(m + gain * h), _expit(m - gain * h)


# ---------------------------------------------------------------------------
# Label calibration table (Stage 1a)
# ---------------------------------------------------------------------------


SUPPORT_LABELS = (
    "overwhelming support",
    "strong support",
    "moderate support",
    "weak support",
    "no bearing",
    "weak undermining",
    "moderate undermining",
    "strong undermining",
    "overwhelming undermining",
)

DEMANDINGNESS_LABELS = (
    "overwhelmingly demanding",
    "strongly demanding",
    "moderately demanding",
    "weakly demanding",
    "neutral",
    "weakly undemanding",
    "moderately undemanding",
    "strongly undemanding",
    "overwhelmingly undemanding",
)


def label_calibration_rows(
    evidence: EvidenceProcessor,
    gains: Sequence[float] = (1.0,),
) -> pd.DataFrame:
    """One row per (support, demandingness, gain) combination."""
    rows: List[Dict[str, Any]] = []
    for support in SUPPORT_LABELS:
        for demandingness in DEMANDINGNESS_LABELS:
            alpha_p, beta_p, alpha_a, beta_a = evidence.get_beta_parameters(
                support, demandingness
            )
            mu_pres = alpha_p / (alpha_p + beta_p)
            mu_abs = alpha_a / (alpha_a + beta_a)
            for g in gains:
                mu_pg, mu_ag = apply_symmetric_gain(mu_pres, mu_abs, g)
                mu_pg_f = float(mu_pg)
                mu_ag_f = float(mu_ag)
                delta_prior = mu_pg_f - mu_ag_f
                rows.append(
                    {
                        "support": support,
                        "demandingness": demandingness,
                        "gain": g,
                        "mu_pres": mu_pg_f,
                        "mu_abs": mu_ag_f,
                        "transmission_delta": delta_prior,
                        "q_at_C_0.001": mu_ag_f + ANCHOR_LOW * delta_prior,
                        "q_at_C_0.5": mu_ag_f + 0.5 * delta_prior,
                        "q_at_C_0.999": mu_ag_f + ANCHOR_HIGH * delta_prior,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-indicator posterior propagation table (Stage 1b)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndicatorPathMetadata:
    """Support/demandingness path for a single indicator."""

    node_key: str
    display_name: str
    depth: int
    support_path: Tuple[str, ...]
    demandingness_path: Tuple[str, ...]


def collect_indicator_path_metadata(
    stance_data: Dict[str, Any],
) -> Dict[str, IndicatorPathMetadata]:
    """Walk the stance tree and record each indicator's ancestor label path."""
    out: Dict[str, IndicatorPathMetadata] = {}
    root_path = (stance_data["name"],)

    def walk(
        node: Dict[str, Any],
        ancestor_path: Tuple[str, ...],
        support_stack: Tuple[str, ...],
        demand_stack: Tuple[str, ...],
    ) -> None:
        current_support = support_stack + (node.get("support", "no bearing"),)
        current_demand = demand_stack + (node.get("demandingness", "neutral"),)
        current_path = ancestor_path + (node["name"],)
        if node.get("type", "").lower() == "indicator":
            key = node_key(ancestor_path, node["name"])
            out[key] = IndicatorPathMetadata(
                node_key=key,
                display_name=node["name"],
                depth=len(current_support),
                support_path=current_support,
                demandingness_path=current_demand,
            )
            return
        for child in node.get("evidencers", []):
            walk(child, current_path, current_support, current_demand)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path, (), ())
    return out


def propagate_prior_mean_coefficients(
    stance_data: Dict[str, Any],
    evidence: EvidenceProcessor,
    indicators: Sequence[IndicatorSpec],
    gain: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute prior-mean affine coefficients (alpha_j, delta_j) per indicator.

    Returns (intercepts, slopes), each of shape (n_indicators,).  Uses the
    symmetric log-odds-gap gain on every node.  At gain=1 this is just
    propagation through the paper's per-label prior means.
    """
    root_path = (stance_data["name"],)
    indicator_keys = {spec.node_key for spec in indicators}
    alpha_by_key: Dict[str, float] = {}
    delta_by_key: Dict[str, float] = {}

    def walk(
        node: Dict[str, Any],
        ancestor_path: Tuple[str, ...],
        intercept_parent: float,
        slope_parent: float,
    ) -> None:
        current_path = ancestor_path + (node["name"],)
        alpha_p, beta_p, alpha_a, beta_a = evidence.get_beta_parameters(
            node.get("support", "no bearing"),
            node.get("demandingness", "neutral"),
        )
        mu_pres = alpha_p / (alpha_p + beta_p)
        mu_abs = alpha_a / (alpha_a + beta_a)
        mu_pres_g, mu_abs_g = apply_symmetric_gain(mu_pres, mu_abs, gain)
        delta = float(mu_pres_g - mu_abs_g)
        intercept_child = float(mu_abs_g) + intercept_parent * delta
        slope_child = slope_parent * delta

        if node.get("type", "").lower() == "indicator":
            key = node_key(ancestor_path, node["name"])
            if key in indicator_keys:
                alpha_by_key[key] = intercept_child
                delta_by_key[key] = slope_child
            return
        for child in node.get("evidencers", []):
            walk(child, current_path, intercept_child, slope_child)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path, 0.0, 1.0)

    alphas = np.array([alpha_by_key[spec.node_key] for spec in indicators])
    deltas = np.array([delta_by_key[spec.node_key] for spec in indicators])
    return alphas, deltas


def cross_expert_ratings_for_indicator(
    processor: MultiSystemDataProcessor,
    node_key_str: str,
    system: str,
) -> List[int]:
    """Return the cross-system rater's 0-indexed ordinal ratings for one cell."""
    expert_idx = processor.expert_to_idx.get(_CROSS_SYSTEM_EXPERT)
    if expert_idx is None:
        return []
    system_obs = processor.system_observations.get(system, {})
    obs_list = system_obs.get(node_key_str, [])
    return [rating for expert, rating in obs_list if expert == expert_idx]


def _leaf_updated_expected_z(
    idata: Any, builder: MultiSystemModelBuilder, system: str,
    indicators: Sequence[IndicatorSpec],
) -> np.ndarray:
    """Posterior-median leaf-updated E[z_j] per indicator for one system."""
    sp = builder._sys_prefix(system)
    post = idata.posterior
    out = np.full(len(indicators), np.nan)
    for i, spec in enumerate(indicators):
        var = f"{sp}__{spec.varname}_pz1"
        if var in post.data_vars:
            out[i] = float(np.median(np.asarray(post[var].values)))
    return out


def per_indicator_posterior_table(
    idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
    stance_data: Dict[str, Any],
    evidence: EvidenceProcessor,
) -> pd.DataFrame:
    """Per-indicator posterior affine + prior-mean comparison table."""
    indicators = build_indicator_index(stance_data, builder)
    beta_pres_by_key, beta_abs_by_key = extract_beta_draws_by_node(
        idata, builder, stance_data
    )
    intercepts, slopes, indicator_order = (
        propagate_affine_indicator_coefficients_from_draws(
            stance_data,
            builder.node_to_varname,
            beta_pres_by_key,
            beta_abs_by_key,
            indicator_index=indicators,
        )
    )
    # q_j(0.999), q_j(0.001) per draw, then posterior medians
    q_high = intercepts + ANCHOR_HIGH * slopes  # (S, J)
    q_low = intercepts + ANCHOR_LOW * slopes
    dq = q_high - q_low

    alpha_med = np.median(intercepts, axis=0)
    alpha_lo = np.percentile(intercepts, 3, axis=0)
    alpha_hi = np.percentile(intercepts, 97, axis=0)
    delta_med = np.median(slopes, axis=0)
    delta_lo = np.percentile(slopes, 3, axis=0)
    delta_hi = np.percentile(slopes, 97, axis=0)
    q_high_med = np.median(q_high, axis=0)
    q_low_med = np.median(q_low, axis=0)
    dq_med = np.median(dq, axis=0)

    # Prior-mean counterfactual (gain=1.0, paper mapping)
    alpha_prior, delta_prior = propagate_prior_mean_coefficients(
        stance_data, evidence, indicator_order, gain=1.0
    )

    # Path metadata
    path_meta = collect_indicator_path_metadata(stance_data)

    # Leaf-updated E[z] at Human/ELIZA
    ez_human = _leaf_updated_expected_z(idata, builder, HUMAN_SYSTEM, indicator_order)
    ez_eliza = _leaf_updated_expected_z(idata, builder, ELIZA_SYSTEM, indicator_order)

    rows: List[Dict[str, Any]] = []
    for i, spec in enumerate(indicator_order):
        meta = path_meta[spec.node_key]
        cross_h = cross_expert_ratings_for_indicator(
            processor, spec.node_key, HUMAN_SYSTEM
        )
        cross_e = cross_expert_ratings_for_indicator(
            processor, spec.node_key, ELIZA_SYSTEM
        )
        rows.append(
            {
                "node_key": spec.node_key,
                "indicator": spec.display_name,
                "depth": meta.depth,
                "support_path": " | ".join(meta.support_path),
                "demandingness_path": " | ".join(meta.demandingness_path),
                "leading_support": meta.support_path[0] if meta.support_path else "",
                "leading_demand": meta.demandingness_path[0] if meta.demandingness_path else "",
                "alpha_post_med": float(alpha_med[i]),
                "alpha_post_lo": float(alpha_lo[i]),
                "alpha_post_hi": float(alpha_hi[i]),
                "delta_post_med": float(delta_med[i]),
                "delta_post_lo": float(delta_lo[i]),
                "delta_post_hi": float(delta_hi[i]),
                "delta_prior": float(delta_prior[i]),
                "alpha_prior": float(alpha_prior[i]),
                "q_high_post": float(q_high_med[i]),
                "q_low_post": float(q_low_med[i]),
                "delta_q_post": float(dq_med[i]),
                "q_high_prior": float(alpha_prior[i] + ANCHOR_HIGH * delta_prior[i]),
                "q_low_prior": float(alpha_prior[i] + ANCHOR_LOW * delta_prior[i]),
                "delta_q_prior": float(delta_prior[i] * (ANCHOR_HIGH - ANCHOR_LOW)),
                "leaf_updated_ez_human": float(ez_human[i]),
                "leaf_updated_ez_eliza": float(ez_eliza[i]),
                f"{CROSS_SYSTEM_CODE}_human_ratings": cross_h,
                f"{CROSS_SYSTEM_CODE}_eliza_ratings": cross_e,
            }
        )
    return pd.DataFrame(rows).sort_values("delta_post_med").reset_index(drop=True)


def robust_log_ratio_and_flag(
    delta_post: np.ndarray, delta_prior: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robust log ratio + sign-flip indicator + structural-zero mask.

    Returns (log_r, sign_flip, structural_zero_mask).  Entries where
    |delta_prior| < RATIO_EPS are masked out (structural zero); their
    log_r is NaN.
    """
    mask_zero = np.abs(delta_prior) < RATIO_EPS
    log_r = np.full_like(delta_post, np.nan, dtype=float)
    valid = ~mask_zero
    log_r[valid] = (
        np.log(np.abs(delta_post[valid]) + LOG_EPS)
        - np.log(np.abs(delta_prior[valid]) + LOG_EPS)
    )
    sign_flip = (np.sign(delta_post) != np.sign(delta_prior)) & valid
    return log_r, sign_flip, mask_zero


# ---------------------------------------------------------------------------
# Stage 3.5a: prior-predictive Monte Carlo over the tree
# ---------------------------------------------------------------------------


def sample_tree_prior_predictive(
    stance_data: Dict[str, Any],
    evidence: EvidenceProcessor,
    indicators: Sequence[IndicatorSpec],
    n_samples: int = 10_000,
    C_values: Tuple[float, ...] = (ANCHOR_LOW, 0.5, ANCHOR_HIGH),
    gain: float = 1.0,
    rng_seed: int = 42,
) -> Dict[float, np.ndarray]:
    """Sample q_j from the tree prior (no observations) at fixed C values.

    For each node the prior is ``Beta(c*mu, c*(1-mu))`` with c =
    NODE_CONCENTRATION and mu set by the label mapping (optionally under
    the symmetric log-odds-gap gain).  Beta draws propagate additively
    through the tree per the recursion
        q_child = q_parent * beta_pres + (1 - q_parent) * beta_abs.

    Returns {C: ndarray of shape (n_samples, n_indicators)}.
    """
    c_conc = 10.0  # NODE_CONCENTRATION; imported constant is inside ModelConfig
    rng = np.random.default_rng(rng_seed)
    indicator_keys = [spec.node_key for spec in indicators]
    out: Dict[float, np.ndarray] = {C: np.zeros((n_samples, len(indicators))) for C in C_values}

    root_path = (stance_data["name"],)

    for s in range(n_samples):
        q_at: Dict[float, Dict[str, float]] = {C: {} for C in C_values}

        def walk(
            node: Dict[str, Any],
            ancestor_path: Tuple[str, ...],
            q_parent: Dict[float, float],
        ) -> None:
            current_path = ancestor_path + (node["name"],)
            alpha_p, beta_p, alpha_a, beta_a = evidence.get_beta_parameters(
                node.get("support", "no bearing"),
                node.get("demandingness", "neutral"),
            )
            mu_pres = alpha_p / (alpha_p + beta_p)
            mu_abs = alpha_a / (alpha_a + beta_a)
            mu_pres_g, mu_abs_g = apply_symmetric_gain(mu_pres, mu_abs, gain)
            # Draw beta from Beta(c*mu, c*(1-mu)) per node
            a_p = float(c_conc * mu_pres_g)
            b_p = float(c_conc * (1 - mu_pres_g))
            a_a = float(c_conc * mu_abs_g)
            b_a = float(c_conc * (1 - mu_abs_g))
            b_pres_draw = rng.beta(max(a_p, 1e-6), max(b_p, 1e-6))
            b_abs_draw = rng.beta(max(a_a, 1e-6), max(b_a, 1e-6))
            q_child: Dict[float, float] = {}
            for C in C_values:
                q_child[C] = q_parent[C] * b_pres_draw + (1 - q_parent[C]) * b_abs_draw

            if node.get("type", "").lower() == "indicator":
                key = node_key(ancestor_path, node["name"])
                for C in C_values:
                    q_at[C][key] = q_child[C]
                return
            for child in node.get("evidencers", []):
                walk(child, current_path, q_child)

        # Stance-level C is fixed per column in C_values; tree walk uses it
        # as the root q (stance is binary at C, so q_parent of its children
        # is just C).
        for child in stance_data.get("evidencers", []):
            walk(child, root_path, {C: C for C in C_values})

        for C in C_values:
            for j, key in enumerate(indicator_keys):
                out[C][s, j] = q_at[C].get(key, np.nan)

    return out


def summarise_prior_predictive(
    samples_by_C: Dict[float, np.ndarray],
    indicators: Sequence[IndicatorSpec],
) -> pd.DataFrame:
    """Per-indicator prior-predictive medians + 94% CIs at each C."""
    rows: List[Dict[str, Any]] = []
    for C, samples in samples_by_C.items():
        medians = np.median(samples, axis=0)
        lo = np.percentile(samples, 3, axis=0)
        hi = np.percentile(samples, 97, axis=0)
        for j, spec in enumerate(indicators):
            rows.append(
                {
                    "C": C,
                    "indicator": spec.display_name,
                    "node_key": spec.node_key,
                    "q_median": float(medians[j]),
                    "q_lo": float(lo[j]),
                    "q_hi": float(hi[j]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 3.5b: label-group count table
# ---------------------------------------------------------------------------


def label_group_counts(
    stance_data: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Count (support, demandingness) combinations appearing in the tree.

    Returns two DataFrames:
    - pres_groups: rows keyed by (support, demandingness), count of nodes
      with that pairing. Relevant for pooling ``beta_pres``.
    - abs_groups: rows keyed by demandingness alone, count of nodes with
      that demandingness. Relevant for pooling ``beta_abs``.
    """
    pres_counts: Dict[Tuple[str, str], int] = {}
    abs_counts: Dict[str, int] = {}

    def walk(node: Dict[str, Any]) -> None:
        s = node.get("support", "no bearing")
        d = node.get("demandingness", "neutral")
        # Stance root has no support/demandingness; skip it.
        if "type" in node and node["type"].lower() in {"feature", "subfeature", "indicator"}:
            pres_counts[(s, d)] = pres_counts.get((s, d), 0) + 1
            abs_counts[d] = abs_counts.get(d, 0) + 1
        for child in node.get("evidencers", []):
            walk(child)

    for child in stance_data.get("evidencers", []):
        walk(child)

    pres_df = pd.DataFrame(
        [{"support": s, "demandingness": d, "count": c} for (s, d), c in pres_counts.items()]
    ).sort_values(["count", "support", "demandingness"], ascending=[False, True, True]).reset_index(drop=True)
    abs_df = pd.DataFrame(
        [{"demandingness": d, "count": c} for d, c in abs_counts.items()]
    ).sort_values("count", ascending=False).reset_index(drop=True)
    return pres_df, abs_df


def write_prior_predictive_markdown(
    summary_df: pd.DataFrame,
    pres_groups: pd.DataFrame,
    abs_groups: pd.DataFrame,
    path: Path,
) -> None:
    """Markdown report summarising Stage 3.5a + 3.5b."""
    lines: List[str] = [
        "# Tree prior-predictive Monte Carlo + label-group counts (Stage 3.5)",
        "",
        "## 3.5a — Prior-predictive distribution of q_j at reference anchors",
        "",
        "Summary statistics (median of per-indicator medians, across indicators):",
        "",
        "| C | n_ind | mean q_median | median q_median | min | max | mean CI-width |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for C in sorted(summary_df["C"].unique()):
        sub = summary_df[summary_df["C"] == C]
        widths = sub["q_hi"] - sub["q_lo"]
        lines.append(
            f"| {C} | {len(sub)} | {sub['q_median'].mean():.3f} | "
            f"{sub['q_median'].median():.3f} | {sub['q_median'].min():.3f} | "
            f"{sub['q_median'].max():.3f} | {widths.mean():.3f} |"
        )
    lines.append("")
    lines.append(
        "**Reading.** At $C = 0.999$ (Human anchor), if $q_j$ medians "
        "cluster near $0.6$ with little mass near $1.0$, the paper's "
        "prior structurally cannot support strong anchor transmission — "
        "the observed ~$0.13$ average $\\Delta q_j$ in the posterior is "
        "not posterior shrinkage, it is the prior's own ceiling."
    )
    lines.append("")
    lines.append("## 3.5b — (support × demandingness) group counts for β_pres pooling")
    lines.append("")
    lines.append("| support | demandingness | count |")
    lines.append("|---|---|---:|")
    for _, r in pres_groups.iterrows():
        lines.append(f"| {r['support']} | {r['demandingness']} | {r['count']} |")
    lines.append("")
    lines.append("## 3.5b — demandingness group counts for β_abs pooling")
    lines.append("")
    lines.append("| demandingness | count |")
    lines.append("|---|---:|")
    for _, r in abs_groups.iterrows():
        lines.append(f"| {r['demandingness']} | {r['count']} |")
    lines.append("")
    lines.append(
        "**Reading.** Groups with count $\\geq 3$ are usefully poolable: "
        "pooling lets same-label nodes share evidence about their common "
        "transmission. Groups with count $\\leq 2$ are effectively "
        "re-parameterised but not much learned. Of particular interest: "
        "the count for `weak undermining + neutral` (the subfeature behind "
        "the 5 sign-flip indicators)."
    )
    lines.append("")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Cross-expert-vs-others propagation scatter (Stage 1d)
# ---------------------------------------------------------------------------


def cross_expert_scatter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Compact frame for cross-expert-vs-others scatter plot."""
    h_col = f"{CROSS_SYSTEM_CODE}_human_ratings"
    e_col = f"{CROSS_SYSTEM_CODE}_eliza_ratings"
    has_h = f"has_{CROSS_SYSTEM_CODE}_human"
    has_e = f"has_{CROSS_SYSTEM_CODE}_eliza"
    return df.assign(
        **{
            has_h: df[h_col].apply(bool),
            has_e: df[e_col].apply(bool),
        }
    )[
        [
            "indicator",
            "depth",
            "q_high_post",
            "q_low_post",
            "delta_q_post",
            "delta_q_prior",
            has_h,
            has_e,
        ]
    ]


# ---------------------------------------------------------------------------
# Counterfactual gain screen (Stage 2c)
# ---------------------------------------------------------------------------


def transformed_beta_draws(
    beta_pres_by_key: Dict[str, np.ndarray],
    beta_abs_by_key: Dict[str, np.ndarray],
    evidence: EvidenceProcessor,
    stance_data: Dict[str, Any],
    gain: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Logit-space residual-preserving transform of posterior beta draws.

    For each node with paper prior means (mu_p, mu_a), compute g-rescaled
    means (mu_p^g, mu_a^g) and shift each beta draw in logit space by
    (logit mu_s^g - logit mu_s^paper).  This is a counterfactual
    translation under the new prior; not a refit.
    """
    out_pres: Dict[str, np.ndarray] = {}
    out_abs: Dict[str, np.ndarray] = {}
    root_path = (stance_data["name"],)

    def walk(node: Dict[str, Any], ancestor_path: Tuple[str, ...]) -> None:
        current_path = ancestor_path + (node["name"],)
        key = node_key(ancestor_path, node["name"])
        alpha_p, beta_p, alpha_a, beta_a = evidence.get_beta_parameters(
            node.get("support", "no bearing"),
            node.get("demandingness", "neutral"),
        )
        mu_p = alpha_p / (alpha_p + beta_p)
        mu_a = alpha_a / (alpha_a + beta_a)
        mu_p_g, mu_a_g = apply_symmetric_gain(mu_p, mu_a, gain)
        shift_p = _logit(mu_p_g) - _logit(mu_p)
        shift_a = _logit(mu_a_g) - _logit(mu_a)
        logit_pres = _logit(beta_pres_by_key[key]) + shift_p
        logit_abs = _logit(beta_abs_by_key[key]) + shift_a
        out_pres[key] = np.clip(_expit(logit_pres), 1e-9, 1.0 - 1e-9)
        out_abs[key] = np.clip(_expit(logit_abs), 1e-9, 1.0 - 1e-9)
        for child in node.get("evidencers", []):
            walk(child, current_path)

    for child in stance_data.get("evidencers", []):
        walk(child, root_path)
    return out_pres, out_abs


def _three_state_cell_prediction(
    q_draws: np.ndarray,
    a_draws: np.ndarray,
    kappa_draws: np.ndarray,
    expert_ratings: List[int],
) -> Dict[str, float]:
    """Predicted signed-tail stats for a (expert, system, indicator-set) cell.

    q_draws: (S, J) posterior q_j draws (after any counterfactual transform).
    a_draws: (S,) discrimination draws.
    kappa_draws: (S, K-1) shared cutpoint draws.
    expert_ratings: flat list of observed ratings for this (expert, system)
        cell, used only to compute observed-category counts.
    """
    K = K_CATEGORIES
    # Three-state weights per indicator
    w0 = (1.0 - q_draws) ** 2
    w1 = 2.0 * q_draws * (1.0 - q_draws)
    w2 = q_draws ** 2

    # Ordered-probit category probs at eta in {0, a/2, a}
    def op_probs(eta):
        cum = norm.cdf(kappa_draws - eta[:, None])
        zeros = np.zeros((cum.shape[0], 1))
        ones = np.ones((cum.shape[0], 1))
        return np.diff(np.concatenate([zeros, cum, ones], axis=1), axis=1)

    zeros_eta = np.zeros_like(a_draws)
    p0 = op_probs(zeros_eta)         # (S, K)
    phalf = op_probs(a_draws * 0.5)
    p2 = op_probs(a_draws)

    # Per-indicator predicted category distribution then averaged across indicators
    # shape w*: (S, J); p*: (S, K)
    # predicted per indicator = w0[:,j,None]*p0 + w1[:,j,None]*phalf + w2[:,j,None]*p2
    pred = (
        w0[:, :, None] * p0[:, None, :]
        + w1[:, :, None] * phalf[:, None, :]
        + w2[:, :, None] * p2[:, None, :]
    )  # (S, J, K)

    # Average over indicators for each draw -> (S, K)
    pred_mean = pred.mean(axis=1)
    pred_left = pred_mean[:, 0]
    pred_right = pred_mean[:, -1]
    pred_mid = pred_mean[:, 2:5].sum(axis=1)
    pred_cat_mean = (pred_mean * np.arange(K)).sum(axis=1)

    # Observed stats
    if expert_ratings:
        arr = np.asarray(expert_ratings)
        obs_left = float(np.mean(arr == 0))
        obs_right = float(np.mean(arr == K - 1))
        obs_mid = float(np.mean((arr >= 2) & (arr <= 4)))
        obs_mean = float(arr.mean())
    else:
        obs_left = obs_right = obs_mid = obs_mean = float("nan")

    return {
        "pred_left": float(np.median(pred_left)),
        "pred_right": float(np.median(pred_right)),
        "pred_mid": float(np.median(pred_mid)),
        "pred_mean": float(np.median(pred_cat_mean)),
        "delta_left": float(np.median(pred_left)) - obs_left,
        "delta_right": float(np.median(pred_right)) - obs_right,
        "delta_mid": float(np.median(pred_mid)) - obs_mid,
        "obs_left": obs_left,
        "obs_right": obs_right,
        "obs_mid": obs_mid,
        "obs_mean": obs_mean,
    }


def _cell_q_draws_for_cross_expert_cell(
    idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
    stance_data: Dict[str, Any],
    beta_pres_by_key: Dict[str, np.ndarray],
    beta_abs_by_key: Dict[str, np.ndarray],
    system: str,
) -> Tuple[np.ndarray, List[int]]:
    """Return (q_draws, rating_list) restricted to cross-expert indicators for system.

    q_draws has shape (S, J) where J is the number of indicators the
    cross-system rater observed in this system.  rating_list is the flat
    list of that rater's ratings across those indicators.
    """
    indicators = build_indicator_index(stance_data, builder)
    intercepts, slopes, indicator_order = (
        propagate_affine_indicator_coefficients_from_draws(
            stance_data,
            builder.node_to_varname,
            beta_pres_by_key,
            beta_abs_by_key,
            indicator_index=indicators,
        )
    )
    anchor = ANCHOR_HIGH if system == HUMAN_SYSTEM else ANCHOR_LOW
    q_all = intercepts + anchor * slopes  # (S, J_all)

    expert_idx = processor.expert_to_idx[_CROSS_SYSTEM_EXPERT]
    sys_obs = processor.system_observations.get(system, {})
    cross_indicator_idx: List[int] = []
    rating_list: List[int] = []
    for j, spec in enumerate(indicator_order):
        obs_list = sys_obs.get(spec.node_key, [])
        cross_here = [rating for expert, rating in obs_list if expert == expert_idx]
        if cross_here:
            cross_indicator_idx.append(j)
            rating_list.extend(cross_here)

    if not cross_indicator_idx:
        return q_all[:, :0], []
    return q_all[:, cross_indicator_idx], rating_list


def counterfactual_screen_rows(
    binary_idata: Any,
    three_state_idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
    stance_data: Dict[str, Any],
    evidence: EvidenceProcessor,
    gains: Sequence[float],
) -> pd.DataFrame:
    """Transformed-tree counterfactual under three-state emission, per gain.

    The tree-shape beta draws come from binary_idata (the reporting baseline),
    transformed by ``transformed_beta_draws`` at each gain.  The emission
    parameters (a, kappa) come from the three-state anchored idata (best
    leaf ceiling); the screen therefore reports what would happen if the
    tree-side prior mapping were revised but the three-state emission were
    refit unchanged.  Clearly a heuristic: no interaction between the two
    sides is modelled.  Focus cells are ``(E_cross, Human)`` and
    ``(E_cross, ELIZA)`` — the cross-system rater at the reference anchors.
    """
    beta_pres_by_key, beta_abs_by_key = extract_beta_draws_by_node(
        binary_idata, builder, stance_data
    )
    # Common indicator length
    n_draws_binary = next(iter(beta_pres_by_key.values())).shape[0]

    post_three = three_state_idata.posterior
    a_draws = np.asarray(post_three["a"].values).reshape(-1)
    kappa_draws = np.asarray(post_three["kappa"].values).reshape(-1, K_CATEGORIES - 1)

    # Align S by resampling/truncating to common length
    n_common = min(n_draws_binary, a_draws.shape[0])
    a_draws = a_draws[:n_common]
    kappa_draws = kappa_draws[:n_common]

    rows: List[Dict[str, Any]] = []
    for gain in gains:
        if np.isclose(gain, 1.0):
            bp_g, ba_g = beta_pres_by_key, beta_abs_by_key
        else:
            bp_g, ba_g = transformed_beta_draws(
                beta_pres_by_key, beta_abs_by_key, evidence, stance_data, gain
            )
        # Truncate to n_common for consistency
        bp_g = {k: v[:n_common] for k, v in bp_g.items()}
        ba_g = {k: v[:n_common] for k, v in ba_g.items()}
        for system in (HUMAN_SYSTEM, ELIZA_SYSTEM):
            q_draws, ratings = _cell_q_draws_for_cross_expert_cell(
                binary_idata, builder, processor, stance_data,
                bp_g, ba_g, system,
            )
            # q_draws: (n_binary, J). Truncate to n_common.
            q_draws = q_draws[:n_common]
            if q_draws.shape[1] == 0:
                continue
            stats = _three_state_cell_prediction(
                q_draws, a_draws, kappa_draws, ratings
            )
            stats.update({"gain": gain, "system": system, "n_ratings": len(ratings)})
            rows.append(stats)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-level driver + markdown rendering
# ---------------------------------------------------------------------------


def _load_contexts() -> Dict[str, Any]:
    """Load the binary (and three-state) idatas plus metadata contexts."""
    config = ModelConfig(
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        INDICATOR_STATE_MODEL="binary",
    )
    stance_data = get_gwt_stance_data(config)
    builder, processor = build_metadata_context(
        stance_data, config, ANCHORED_SYSTEM_CONFIGS
    )
    binary_idata = az.from_netcdf(BINARY_IDATA_PATH)
    three_idata = az.from_netcdf(THREE_STATE_IDATA_PATH)
    evidence = EvidenceProcessor(config)
    return {
        "config": config,
        "stance_data": stance_data,
        "builder": builder,
        "processor": processor,
        "binary_idata": binary_idata,
        "three_state_idata": three_idata,
        "evidence": evidence,
    }


def _fmt_f(v: float, w: int = 6, prec: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan".rjust(w)
    return f"{v:{w}.{prec}f}"


def write_label_calibration_markdown(df: pd.DataFrame, path: Path) -> None:
    """Render the label calibration table(s) grouped by gain."""
    lines: List[str] = ["# Prior-predictive label calibration table\n"]
    lines.append(
        "For each (support, demandingness) combination in "
        "``EvidenceProcessor.get_beta_parameters``, prior-mean "
        "$\\beta_\\text{pres}$, $\\beta_\\text{abs}$, transmission "
        "$\\Delta = \\beta_\\text{pres} - \\beta_\\text{abs}$, and one-layer "
        "propagation at $C \\in \\{0.001, 0.5, 0.999\\}$.  Multiple gain "
        "values correspond to the symmetric log-odds-gap transform; "
        "$g = 1$ is the paper mapping.\n"
    )
    for gain in sorted(df["gain"].unique()):
        sub = df[df["gain"] == gain]
        lines.append(f"## gain = {gain}\n")
        lines.append(
            "| support | demandingness | mu_pres | mu_abs | delta | q(0.001) | q(0.5) | q(0.999) |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['support']} | {r['demandingness']} | "
                f"{_fmt_f(r['mu_pres'])} | {_fmt_f(r['mu_abs'])} | "
                f"{_fmt_f(r['transmission_delta'])} | "
                f"{_fmt_f(r['q_at_C_0.001'])} | "
                f"{_fmt_f(r['q_at_C_0.5'])} | "
                f"{_fmt_f(r['q_at_C_0.999'])} |"
            )
        lines.append("")
    path.write_text("\n".join(lines))


def write_per_indicator_markdown(df: pd.DataFrame, path: Path) -> None:
    """Render the per-indicator posterior propagation table."""
    lines: List[str] = [
        "# Per-indicator posterior propagation table",
        "",
        "Sorted by posterior-median $\\delta_j$ ascending.  Prior-mean "
        "counterfactual (paper mapping at $g=1$) shown alongside for the "
        "structural-vs-data-induced diagnostic.  Ratings are 0-indexed "
        "ordinal (0 = strongly absent, 6 = strongly present).  "
        f"`{CROSS_SYSTEM_CODE}_H` and `{CROSS_SYSTEM_CODE}_E` are the "
        "cross-system rater's Human and ELIZA observations respectively.",
        "",
    ]
    # Robust log r_j
    log_r, flip, mask_zero = robust_log_ratio_and_flag(
        df["delta_post_med"].to_numpy(), df["delta_prior"].to_numpy()
    )
    df = df.assign(log_r=log_r, sign_flip=flip, structural_zero=mask_zero)

    h_col = f"{CROSS_SYSTEM_CODE}_human_ratings"
    e_col = f"{CROSS_SYSTEM_CODE}_eliza_ratings"
    cols_header = (
        "| indicator | depth | leading_support | leading_demand | "
        "delta_prior | delta_post_med | log r_j | sign_flip | struct_0 | "
        "q_high_post | q_low_post | delta_q_post | "
        f"ez_human | ez_eliza | {CROSS_SYSTEM_CODE}_H | {CROSS_SYSTEM_CODE}_E |"
    )
    align = "|---|---:|---|---|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|---|---|"
    lines.append(cols_header)
    lines.append(align)
    for _, r in df.iterrows():
        lines.append(
            f"| {r['indicator']} | {r['depth']} | {r['leading_support']} | "
            f"{r['leading_demand']} | {_fmt_f(r['delta_prior'])} | "
            f"{_fmt_f(r['delta_post_med'])} | {_fmt_f(r['log_r'])} | "
            f"{'Y' if r['sign_flip'] else ''} | "
            f"{'Y' if r['structural_zero'] else ''} | "
            f"{_fmt_f(r['q_high_post'])} | {_fmt_f(r['q_low_post'])} | "
            f"{_fmt_f(r['delta_q_post'])} | "
            f"{_fmt_f(r['leaf_updated_ez_human'])} | "
            f"{_fmt_f(r['leaf_updated_ez_eliza'])} | "
            f"{r[h_col]} | {r[e_col]} |"
        )
    lines.append("")

    # Summary statistics
    lines.append("## Summary\n")
    non_zero = df[~df["structural_zero"]]
    if len(non_zero):
        lines.append(
            f"- Non structural-zero indicators: {len(non_zero)} / {len(df)}"
        )
        lines.append(
            f"- Posterior delta_j: mean={non_zero['delta_post_med'].mean():.3f}, "
            f"median={non_zero['delta_post_med'].median():.3f}, "
            f"range=[{non_zero['delta_post_med'].min():.3f}, "
            f"{non_zero['delta_post_med'].max():.3f}]"
        )
        lines.append(
            f"- Prior (g=1) delta_j:   mean={non_zero['delta_prior'].mean():.3f}, "
            f"median={non_zero['delta_prior'].median():.3f}, "
            f"range=[{non_zero['delta_prior'].min():.3f}, "
            f"{non_zero['delta_prior'].max():.3f}]"
        )
        lines.append(
            f"- log r_j (post/prior):   mean={non_zero['log_r'].mean():.3f}, "
            f"median={non_zero['log_r'].median():.3f}, "
            f"range=[{non_zero['log_r'].min():.3f}, {non_zero['log_r'].max():.3f}]"
        )
        lines.append(
            f"- Sign flips: {int(non_zero['sign_flip'].sum())} / {len(non_zero)}"
        )
    n_zero = int(df["structural_zero"].sum())
    lines.append(f"- Structural-zero (|prior delta| < {RATIO_EPS}): {n_zero} / {len(df)}")
    lines.append("")

    # Depth stratification
    lines.append("## By depth\n")
    lines.append("| depth | n | mean delta_prior | mean delta_post | mean log r_j |")
    lines.append("|---:|---:|---:|---:|---:|")
    for depth, sub in df.groupby("depth"):
        sub_nz = sub[~sub["structural_zero"]]
        lines.append(
            f"| {depth} | {len(sub)} | {sub['delta_prior'].mean():.3f} | "
            f"{sub['delta_post_med'].mean():.3f} | "
            f"{sub_nz['log_r'].mean() if len(sub_nz) else float('nan'):.3f} |"
        )
    lines.append("")

    path.write_text("\n".join(lines))


def write_counterfactual_markdown(df: pd.DataFrame, path: Path) -> None:
    """Render the counterfactual screen output."""
    lines: List[str] = [
        "# Transformed-tree counterfactual screen (Stage 2c)",
        "",
        "Heuristic veto only; NOT a prior-predictive or approximate refit.",
        "Tree-side beta draws are taken from `binary_anchored.nc` and "
        "residual-preservingly translated in logit space under the "
        "symmetric log-odds-gap gain at each candidate $g$.  Emission "
        "parameters $(a, \\kappa)$ come from `three_state_anchored.nc` "
        "unchanged.  Predictions averaged over the cross-system rater's "
        "observed indicators per reference cell.",
        "",
        "| gain | system | n_rat | pred_left | obs_left | delta_left | "
        "pred_right | obs_right | delta_right | pred_mid | delta_mid | "
        "pred_mean |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['gain']} | {r['system']} | {r['n_ratings']} | "
            f"{_fmt_f(r['pred_left'])} | {_fmt_f(r['obs_left'])} | "
            f"{_fmt_f(r['delta_left'])} | {_fmt_f(r['pred_right'])} | "
            f"{_fmt_f(r['obs_right'])} | {_fmt_f(r['delta_right'])} | "
            f"{_fmt_f(r['pred_mid'])} | {_fmt_f(r['delta_mid'])} | "
            f"{_fmt_f(r['pred_mean'])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = _load_contexts()
    evidence: EvidenceProcessor = ctx["evidence"]
    stance_data = ctx["stance_data"]
    builder = ctx["builder"]
    processor = ctx["processor"]
    binary_idata = ctx["binary_idata"]
    three_state_idata = ctx["three_state_idata"]

    # --- Stage 1a: label calibration ---
    print("Stage 1a: label calibration table...")
    gains_for_table = (1.0, 1.5, 2.0, 2.5, 3.0)
    label_df = label_calibration_rows(evidence, gains=gains_for_table)
    label_df.to_csv(OUTPUT_DIR / "label_calibration.csv", index=False)
    write_label_calibration_markdown(
        label_df, OUTPUT_DIR / "label_calibration.md"
    )
    print(f"  wrote {OUTPUT_DIR / 'label_calibration.md'}")

    # --- Stage 1b + 1c: per-indicator posterior table + log r_j ---
    print("Stage 1b/1c: per-indicator posterior propagation table...")
    per_ind_df = per_indicator_posterior_table(
        binary_idata, builder, processor, stance_data, evidence
    )
    per_ind_df.to_csv(OUTPUT_DIR / "per_indicator_propagation.csv", index=False)
    write_per_indicator_markdown(
        per_ind_df, OUTPUT_DIR / "per_indicator_propagation.md"
    )
    print(f"  wrote {OUTPUT_DIR / 'per_indicator_propagation.md'}")

    # --- Stage 1d: cross-expert scatter data ---
    print("Stage 1d: cross-expert-vs-others scatter dataframe...")
    scatter_df = cross_expert_scatter_dataframe(per_ind_df)
    scatter_df.to_csv(OUTPUT_DIR / "cross_expert_scatter.csv", index=False)
    print(f"  wrote {OUTPUT_DIR / 'cross_expert_scatter.csv'}")

    # --- Stage 3.5a: prior-predictive Monte Carlo ---
    print("Stage 3.5a: prior-predictive Monte Carlo over the tree...")
    indicators = build_indicator_index(stance_data, builder)
    samples_by_C = sample_tree_prior_predictive(
        stance_data, evidence, indicators, n_samples=10_000, gain=1.0
    )
    pp_summary = summarise_prior_predictive(samples_by_C, indicators)
    pp_summary.to_csv(OUTPUT_DIR / "prior_predictive_qj.csv", index=False)
    np.save(OUTPUT_DIR / "prior_predictive_samples_Chigh.npy", samples_by_C[ANCHOR_HIGH])
    np.save(OUTPUT_DIR / "prior_predictive_samples_Cmid.npy", samples_by_C[0.5])
    np.save(OUTPUT_DIR / "prior_predictive_samples_Clow.npy", samples_by_C[ANCHOR_LOW])

    # --- Stage 3.5b: label-group counts ---
    print("Stage 3.5b: label-group count tables...")
    pres_groups, abs_groups = label_group_counts(stance_data)
    pres_groups.to_csv(OUTPUT_DIR / "label_pres_group_counts.csv", index=False)
    abs_groups.to_csv(OUTPUT_DIR / "label_abs_group_counts.csv", index=False)
    write_prior_predictive_markdown(
        pp_summary, pres_groups, abs_groups,
        OUTPUT_DIR / "prior_predictive_and_group_counts.md",
    )
    print(f"  wrote {OUTPUT_DIR / 'prior_predictive_and_group_counts.md'}")

    # --- Stage 2c: counterfactual gain screen ---
    print("Stage 2c: counterfactual gain screen...")
    screen_gains = (1.0, 1.5, 2.0, 2.5, 3.0)
    screen_df = counterfactual_screen_rows(
        binary_idata, three_state_idata, builder, processor, stance_data,
        evidence, gains=screen_gains,
    )
    screen_df.to_csv(OUTPUT_DIR / "counterfactual_screen.csv", index=False)
    write_counterfactual_markdown(
        screen_df, OUTPUT_DIR / "counterfactual_screen.md"
    )
    print(f"  wrote {OUTPUT_DIR / 'counterfactual_screen.md'}")

    print("Done.")


if __name__ == "__main__":
    main()
