"""Information-theoretic diagnostics for the leaf likelihood / C identification.

These replace the older ``H(Y | q)`` "information ceiling" diagnostic from
``leaf_model_compression.py`` with the quantities ChatGPT 5.5 Pro flagged in
the 2026-05-10 review:

  * ``mutual_information_q_y(leaf_fn, q_grid)``
        I(Q; Y) under uniform Q over q_grid (in nats).
        Upper bound is ``log K`` (≈ 1.95 nats for K = 7); the existing
        ``H(Y | q)`` quantity is ``log K - I(Q;Y)``.

  * ``expected_fisher_c_per_rating(leaf_fn, c_value, alpha_j, delta_j, eps)``
        Per-rating expected Fisher information for C, contributed by one
        indicator at a single design row, given the affine
        propagation slope δ_j.  Uses a central-difference approximation of
        ∂ log p / ∂ C and integrates (·)² over y under p(y | C).

  * ``full_vector_kl(leaf_fn, alpha, delta, c_hi, c_lo, n_per_indicator)``
        ∑_j n_j · D_KL( p(y | C_hi) ‖ p(y | C_lo) ) per system / design.
        This is the *upper bound* of the joint per-system KL (treats the
        indicator-level latent m_j as if independent across raters within
        the same indicator, which only over-counts info; the true joint
        is no larger).  Useful as a "how distinguishable are these two C
        values at this design?" summary.

  * ``sensitivity_decomposition(leaf_fn, c_value, alpha, delta, eps)``
        Per indicator returns (tilde_q, delta_j, leaf_score_var,
        c_fisher_per_rating).  Lets you spot indicators where the slope
        δ_j is large but the leaf is flat in tilde_q (so even though C
        moves tilde_q, the leaf doesn't translate that into rating
        information).

The leaf-model adapter signature is intentionally minimal:

    leaf_fn: Callable[[float], np.ndarray]
        Maps an indicator-effective probability tilde_q in [0, 1] to a
        K-vector of category probabilities.

So the diagnostics are leaf-model agnostic — they apply equally to
``three_state``, ``direct_q``, ``mixture``, etc.  Concrete leaf adapters
for the current production model (three_state with shared a, κ) are
provided as ``three_state_leaf_fn`` below.

CLI usage:

    python info_diagnostics.py --truth-json <path>

reads observation-layer + edge-betas truth from a synthetic-fit
``truth.json``, runs the diagnostic suite for the current GWT design, and
writes a CSV table to ``info_diagnostic_table.csv`` next to the truth
file.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[2]
SISTER_DIR = REPO_ROOT / "notebooks" / "synthetic_validation_2026-05-06"
for d in (REPO_ROOT, SISTER_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))


# ---------------------------------------------------------------------------
# Leaf adapters (numpy-only; mirror the PyTensor leaf in dcm_model.py at the
# production median nuisance).
# ---------------------------------------------------------------------------


def ordered_probit_probs(
    kappa: np.ndarray,
    eta: float,
    n_categories: int = 7,
) -> np.ndarray:
    """P(Y = k | eta, kappa) under shared-scale ordered probit.

    Returns a length-K vector that sums to 1.  K is inferred from kappa
    (length K - 1).
    """
    cum = np.concatenate(([0.0], norm.cdf(np.asarray(kappa, dtype=float) - float(eta)), [1.0]))
    probs = np.diff(cum)
    probs = np.clip(probs, 1e-12, 1.0)
    return probs / probs.sum()


def three_state_leaf_fn(
    a: float,
    kappa: np.ndarray,
) -> Callable[[float], np.ndarray]:
    """Three-state leaf at production-median (a, kappa).

    Returns a callable ``tilde_q -> P(Y | tilde_q)``.

    Construction: ``m ~ Binomial(2, tilde_q)``, eta_m = a · m / 2, then
    ordered-probit emission.  This is the analytical marginalisation over
    the latent ``m`` for a single rating.
    """
    a = float(a)
    kappa = np.asarray(kappa, dtype=float)
    op_probs_at_m = np.stack(
        [ordered_probit_probs(kappa, a * m / 2.0) for m in (0, 1, 2)],
        axis=0,
    )

    def fn(tilde_q: float) -> np.ndarray:
        q = float(np.clip(tilde_q, 1e-12, 1.0 - 1e-12))
        weights = np.array([(1 - q) ** 2, 2 * q * (1 - q), q ** 2])
        return weights @ op_probs_at_m

    return fn


def direct_q_leaf_fn(
    a: float,
    kappa: np.ndarray,
) -> Callable[[float], np.ndarray]:
    """Direct-q leaf (Pro's B.1 ablation): no latent z, eta = a · tilde_q."""
    a = float(a)
    kappa = np.asarray(kappa, dtype=float)

    def fn(tilde_q: float) -> np.ndarray:
        return ordered_probit_probs(kappa, a * float(tilde_q))

    return fn


def mixture_leaf_fn(
    a: float,
    kappa: np.ndarray,
    eta_low: float = 0.0,
    eta_high: float = None,
) -> Callable[[float], np.ndarray]:
    """Two-component mixture leaf (Pro's B.4 / production candidate).

    P(Y | tilde_q) = (1 - tilde_q) · OP(eta_low) + tilde_q · OP(eta_high).

    Defaults to eta_high = a (so the centred-endpoint parameterisation
    mu = a/2, delta = a is implicit).
    """
    a = float(a)
    kappa = np.asarray(kappa, dtype=float)
    if eta_high is None:
        eta_high = a
    op_low = ordered_probit_probs(kappa, eta_low)
    op_high = ordered_probit_probs(kappa, eta_high)

    def fn(tilde_q: float) -> np.ndarray:
        q = float(np.clip(tilde_q, 1e-12, 1.0 - 1e-12))
        return (1 - q) * op_low + q * op_high

    return fn


# ---------------------------------------------------------------------------
# Information-theoretic primitives
# ---------------------------------------------------------------------------


def categorical_entropy(probs: np.ndarray) -> float:
    """H(p) in nats.  ``probs`` need not be exactly normalised."""
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def categorical_kl(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(p ‖ q) in nats.  Both must be probability vectors over same K."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = p > 0
    p_ = p[mask]
    q_ = np.clip(q[mask], 1e-30, 1.0)
    return float(np.sum(p_ * (np.log(p_) - np.log(q_))))


def conditional_entropy_y_given_q(
    leaf_fn: Callable[[float], np.ndarray],
    q_grid: np.ndarray,
) -> float:
    """E_q[H(Y | q)] under uniform Q over ``q_grid`` (in nats).

    Lower = more informative leaf.  This is the legacy diagnostic
    ``info_ceiling_h_rating_given_q`` from leaf_model_compression.py;
    kept here for backward comparability.
    """
    return float(np.mean([categorical_entropy(leaf_fn(q)) for q in q_grid]))


def mutual_information_q_y(
    leaf_fn: Callable[[float], np.ndarray],
    q_grid: np.ndarray,
) -> float:
    """I(Q; Y) under uniform Q over ``q_grid`` (in nats).

    I(Q;Y) = H(Y) - E_q[H(Y | q)].  Upper bound is log K.  This is the
    raw amount of category-distribution information the leaf can
    transmit about q under a uniform prior on q.
    """
    py_at_q = np.array([leaf_fn(q) for q in q_grid])  # (n_q, K)
    py_marg = py_at_q.mean(axis=0)
    h_y = categorical_entropy(py_marg)
    h_y_given_q = float(np.mean([categorical_entropy(p) for p in py_at_q]))
    return float(h_y - h_y_given_q)


# ---------------------------------------------------------------------------
# C-level diagnostics (use the affine tree propagation tilde_q_j = alpha_j + delta_j · C)
# ---------------------------------------------------------------------------


def _tilde_q_at_c(alpha: float, delta: float, c: float) -> float:
    return float(np.clip(alpha + delta * c, 1e-12, 1.0 - 1e-12))


def expected_fisher_c_per_rating(
    leaf_fn: Callable[[float], np.ndarray],
    c_value: float,
    alpha_j: float,
    delta_j: float,
    eps: float = 1e-3,
) -> float:
    """Per-rating expected Fisher info I_F(C) at one indicator and one rating.

    Numerical formulation:  central-difference d log p / d C over y, then
    expectation under p(y | C).  Since log p depends on C only through
    tilde_q_j = alpha_j + delta_j · C, this is equivalent to
    ``delta_j**2 · E_y[(d log p / d tilde_q)^2]`` evaluated at
    tilde_q_j(C).

    Returns nats^2 (so that summing across n ratings of this indicator
    gives the indicator's contribution to total I_F(C)).
    """
    q_lo = _tilde_q_at_c(alpha_j, delta_j, c_value - eps)
    q_hi = _tilde_q_at_c(alpha_j, delta_j, c_value + eps)
    q_at = _tilde_q_at_c(alpha_j, delta_j, c_value)
    p_lo = np.clip(leaf_fn(q_lo), 1e-30, 1.0)
    p_hi = np.clip(leaf_fn(q_hi), 1e-30, 1.0)
    p_at = np.clip(leaf_fn(q_at), 1e-30, 1.0)
    score = (np.log(p_hi) - np.log(p_lo)) / (2.0 * eps)
    return float(np.sum(p_at * score ** 2))


def full_vector_kl(
    leaf_fn: Callable[[float], np.ndarray],
    alpha: Sequence[float],
    delta: Sequence[float],
    c_hi: float,
    c_lo: float,
    n_per_indicator: Sequence[int],
) -> float:
    """Upper bound on D_KL( p(Y | C_hi) ‖ p(Y | C_lo) ) for one system / design.

    Treats each rating as marginally independent and sums per-rating KLs
    weighted by the design (rater count per indicator).  This is an
    over-estimate of the true joint KL (which would account for the
    shared latent m_j across raters within an indicator), but the
    direction of comparison across leaf models / designs is preserved.
    """
    if not (len(alpha) == len(delta) == len(n_per_indicator)):
        raise ValueError("alpha, delta, n_per_indicator must have the same length")
    total = 0.0
    for a_j, d_j, n_j in zip(alpha, delta, n_per_indicator):
        if n_j <= 0:
            continue
        q_hi = _tilde_q_at_c(float(a_j), float(d_j), float(c_hi))
        q_lo = _tilde_q_at_c(float(a_j), float(d_j), float(c_lo))
        p_hi = leaf_fn(q_hi)
        p_lo = leaf_fn(q_lo)
        total += float(n_j) * categorical_kl(p_hi, p_lo)
    return float(total)


def sensitivity_decomposition(
    leaf_fn: Callable[[float], np.ndarray],
    c_value: float,
    alpha: Sequence[float],
    delta: Sequence[float],
    n_per_indicator: Sequence[int],
    indicator_keys: Sequence[str],
    eps: float = 1e-3,
) -> pd.DataFrame:
    """Per-indicator decomposition of the C-level Fisher info contribution.

    Columns:
      ``indicator``         : indicator key (e.g. node_path).
      ``alpha_j``, ``delta_j`` : affine coefficients of tilde_q_j(C).
      ``tilde_q_at_c``      : tilde_q_j evaluated at c_value.
      ``leaf_score_var``    : E_y[(d log p / d tilde_q)^2] per rating.
      ``c_fisher_per_rating`` : delta_j^2 · leaf_score_var.
      ``c_fisher_indicator`` : n_j · c_fisher_per_rating.

    Sum of c_fisher_indicator over indicators = total I_F(C) for one
    system at this design.  Indicators with large delta_j but small
    leaf_score_var are the "wide-but-flat" cases Pro flagged: they
    *separate* C endpoints in tilde_q-space but the leaf doesn't
    translate that separation into rating information.
    """
    rows: List[Dict[str, float]] = []
    for key, a_j, d_j, n_j in zip(indicator_keys, alpha, delta, n_per_indicator):
        a_j = float(a_j)
        d_j = float(d_j)
        n_j = int(n_j)
        q_at = _tilde_q_at_c(a_j, d_j, c_value)
        # Score d log p / d tilde_q (central-difference, k-vector)
        q_l = float(np.clip(q_at - eps, 1e-12, 1.0 - 1e-12))
        q_h = float(np.clip(q_at + eps, 1e-12, 1.0 - 1e-12))
        p_at = np.clip(leaf_fn(q_at), 1e-30, 1.0)
        p_l = np.clip(leaf_fn(q_l), 1e-30, 1.0)
        p_h = np.clip(leaf_fn(q_h), 1e-30, 1.0)
        score = (np.log(p_h) - np.log(p_l)) / (2.0 * (q_h - q_l))
        leaf_score_var = float(np.sum(p_at * score ** 2))
        c_fisher_per_rating = d_j ** 2 * leaf_score_var
        rows.append(
            {
                "indicator": key,
                "alpha_j": a_j,
                "delta_j": d_j,
                "tilde_q_at_c": q_at,
                "leaf_score_var": leaf_score_var,
                "c_fisher_per_rating": c_fisher_per_rating,
                "c_fisher_indicator": float(n_j) * c_fisher_per_rating,
                "n_ratings": n_j,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver: read truth.json and produce a diagnostic table
# ---------------------------------------------------------------------------


def _affine_propagate_from_truth(
    truth_payload: Mapping,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Compute (intercepts alpha_j, slopes delta_j) per indicator from truth.

    Uses the propagation helper from gwt_reference_recovery_analysis but
    feeds it a degenerate single-draw set of beta_pres / beta_abs arrays
    derived from truth_payload['edge_betas'].
    """
    from gwt_reference_recovery_analysis import (  # noqa: E402
        propagate_affine_indicator_coefficients_from_draws,
    )
    from dcm_model import load_data, ModelConfig  # noqa: E402

    cfg = ModelConfig()
    stance_name = next(iter(truth_payload.get("true_C_by_system", {".": 0}).keys()), None)
    # We need the source stance tree for the propagation walk.  Fetch from
    # the data cache via the ModelConfig defaults; the truth was generated
    # from this same data cache, so labels match.
    stance_data = next(
        s for s in load_data(cfg) if s["name"] == cfg.TARGET_STANCE
    )
    edge_betas = truth_payload["edge_betas"]
    # Build the {node_key: ndarray of one draw} dicts the helper expects.
    bp_by_key = {k: np.asarray([float(v["beta_pres"])]) for k, v in edge_betas.items()}
    ba_by_key = {k: np.asarray([float(v["beta_abs"])]) for k, v in edge_betas.items()}
    # The helper also needs node_to_varname for log/diagnostic naming;
    # we don't actually use that output, but the function expects it.
    node_to_varname: Dict[str, str] = {k: k for k in edge_betas.keys()}
    intercepts, slopes, indicators = propagate_affine_indicator_coefficients_from_draws(
        stance_data,
        node_to_varname,
        bp_by_key,
        ba_by_key,
    )
    indicator_keys = [spec.node_key for spec in indicators]
    return indicator_keys, intercepts[0], slopes[0]


def _design_table(
    truth_payload: Mapping,
    indicator_keys: Sequence[str],
    system: str,
) -> List[int]:
    """Per-indicator rater counts for one system from the simulated ratings."""
    obs_by_indicator = (
        truth_payload.get("ordinal_by_system_indicator", {}).get(system, {})
    )
    counts: List[int] = []
    for key in indicator_keys:
        ratings = obs_by_indicator.get(key, [])
        counts.append(int(len(ratings)))
    return counts


def diagnostic_table(
    truth_payload: Mapping,
    leaf_fn_factory: Callable[[float, np.ndarray], Callable[[float], np.ndarray]] = three_state_leaf_fn,
    leaf_label: str = "three_state",
    q_grid_n: int = 41,
    c_grid: Sequence[float] = (0.001, 0.1, 0.25, 0.5, 0.999),
) -> pd.DataFrame:
    """Run the full diagnostic suite and return one summary row per (system, c_query).

    Columns:
      leaf, system, c_value, mutual_info_q_y, h_y_given_q,
      fisher_c_total (sum over indicators · n_j),
      kl_full_vector_vs_c01_c09,
      kl_full_vector_vs_chicken_llm.

    The KL columns compare two C values for the same system.
    """
    obs = truth_payload["observation_parameters"]
    a = float(obs["a"])
    kappa = np.asarray(obs["kappa"], dtype=float)
    leaf_fn = leaf_fn_factory(a, kappa)

    indicator_keys, alpha, delta = _affine_propagate_from_truth(truth_payload)

    q_grid = np.linspace(1e-3, 1 - 1e-3, q_grid_n)
    mi = mutual_information_q_y(leaf_fn, q_grid)
    h_yq = conditional_entropy_y_given_q(leaf_fn, q_grid)

    rows: List[Dict[str, float]] = []
    systems = list(truth_payload["true_C_by_system"].keys())
    for system in systems:
        n_per_ind = _design_table(truth_payload, indicator_keys, system)
        true_c = float(truth_payload["true_C_by_system"][system])
        # Fisher info at the system's true C.
        fisher_per_ind = [
            expected_fisher_c_per_rating(leaf_fn, true_c, a_j, d_j)
            for a_j, d_j in zip(alpha, delta)
        ]
        fisher_total = float(
            np.sum([n * f for n, f in zip(n_per_ind, fisher_per_ind)])
        )
        # KL between c=0.9 and c=0.1 for this system at this design — measures
        # raw distinguishability of two endpoint C values for the design.
        kl_endpoints = full_vector_kl(
            leaf_fn, alpha, delta, c_hi=0.9, c_lo=0.1, n_per_indicator=n_per_ind
        )
        rows.append(
            {
                "leaf": leaf_label,
                "system": system,
                "true_c": true_c,
                "n_total_ratings": int(sum(n_per_ind)),
                "mutual_info_q_y_nats": mi,
                "h_y_given_q_nats": h_yq,
                "fisher_c_at_true_c": fisher_total,
                "kl_full_vector_c09_vs_c01": kl_endpoints,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--truth-json",
        type=Path,
        required=True,
        help="Path to a synthetic-fit truth.json. Provides obs params + edge_betas + design.",
    )
    parser.add_argument(
        "--leaf",
        type=str,
        default="three_state",
        choices=["three_state", "direct_q", "mixture"],
        help="Leaf model adapter to use for the diagnostic.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: alongside truth.json as info_diagnostic_table.csv).",
    )
    parser.add_argument(
        "--per-indicator-csv",
        type=Path,
        default=None,
        help="If set, also write the per-indicator sensitivity decomposition for "
             "each system at its true C value.",
    )
    args = parser.parse_args()

    with args.truth_json.open() as f:
        truth_payload = json.load(f)
    factory_by_name = {
        "three_state": three_state_leaf_fn,
        "direct_q": direct_q_leaf_fn,
        "mixture": mixture_leaf_fn,
    }
    factory = factory_by_name[args.leaf]
    df = diagnostic_table(truth_payload, leaf_fn_factory=factory, leaf_label=args.leaf)
    out_csv = args.out_csv or (args.truth_json.parent / "info_diagnostic_table.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    print(df.to_string(index=False))

    if args.per_indicator_csv is not None:
        a = float(truth_payload["observation_parameters"]["a"])
        kappa = np.asarray(truth_payload["observation_parameters"]["kappa"], dtype=float)
        leaf_fn = factory(a, kappa)
        indicator_keys, alpha, delta = _affine_propagate_from_truth(truth_payload)
        rows: List[pd.DataFrame] = []
        for system in truth_payload["true_C_by_system"]:
            n_per_ind = _design_table(truth_payload, indicator_keys, system)
            sub = sensitivity_decomposition(
                leaf_fn,
                c_value=float(truth_payload["true_C_by_system"][system]),
                alpha=alpha,
                delta=delta,
                n_per_indicator=n_per_ind,
                indicator_keys=indicator_keys,
            )
            sub.insert(0, "system", system)
            sub.insert(0, "leaf", args.leaf)
            rows.append(sub)
        per_ind = pd.concat(rows, ignore_index=True)
        per_ind.to_csv(args.per_indicator_csv, index=False)
        print(f"wrote {args.per_indicator_csv}")


if __name__ == "__main__":
    main()
