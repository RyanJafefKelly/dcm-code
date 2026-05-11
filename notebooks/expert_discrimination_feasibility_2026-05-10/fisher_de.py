"""Analytical Fisher information for per-expert discrimination $d_e$.

Goal: at $d_e = 1$ (production median), compute the expected Fisher
information that each expert's set of ratings contributes about that
expert's discrimination parameter, under two designs:

  * realistic: GWT seed-06 design — most experts only rate one system,
    Rater_B is the only cross-system expert.
  * idealised: every expert rates every (system, indicator) cell.

Model assumed for the extension (production-median three-state leaf):

    tilde_q_j         = alpha_j + delta_j * C_s   (no d_e dependence)
    m | tilde_q_j     ~ Binomial(2, tilde_q_j)
    eta_{e, m}        = a * d_e * m / 2            (d_e enters here)
    Y | eta, kappa    ~ ordered probit (shared sigma=1)

So d_e only acts as a multiplicative scale on the latent rating, leaving
the upstream tree (q_j and its propagation through the DCM) untouched.
This mirrors the simplest "per-rater discrimination" extension Arvo
asked about, and is the cheapest version to test for identifiability:
if it doesn't work here, more elaborate factorisations won't help.

The per-rating expected Fisher info at $d_e = 1$ on indicator $j$ of
system $s$ depends on (a, kappa, alpha_j, delta_j, C_s) only — not on
which expert performs the rating — so per-expert Fisher is a sum of
indicator-level Fishers weighted by how many ratings the expert
contributes to each (system, indicator) cell.  The whole calculation is
analytical / numpy; no sampling required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gwt_reference_recovery_analysis import (  # noqa: E402
    propagate_affine_indicator_coefficients_from_draws,
)
from dcm_model import ModelConfig, load_data, node_key  # noqa: E402

TRUTH_JSON = (
    REPO_ROOT
    / "notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery"
    / "exact_latent_tree__full_exact_tree__three_state_binomial_2"
      "__exact_tree_production_medians__current_gwt_rater_design__seed20260506"
    / "truth.json"
)

OUT_DIR = REPO_ROOT / "notebooks/expert_discrimination_feasibility_2026-05-10"


# --------------------------------------------------------------------------
# Leaf likelihood + Fisher info for d_e
# --------------------------------------------------------------------------

def ordered_probit_probs(kappa: np.ndarray, eta: float) -> np.ndarray:
    """P(Y = k | eta, kappa) under shared-scale ordered probit.

    Returns a length-K vector that sums to 1.  K = len(kappa) + 1.
    """
    cum = np.concatenate([[0.0], norm.cdf(np.asarray(kappa, dtype=float) - float(eta)), [1.0]])
    probs = np.diff(cum)
    probs = np.clip(probs, 1e-12, 1.0)
    return probs / probs.sum()


def three_state_leaf_probs(tilde_q: float, a: float, kappa: np.ndarray) -> np.ndarray:
    """P(Y | tilde_q) under the three-state production-median leaf.

    m ~ Binomial(2, tilde_q), eta_m = a * m / 2, ordered-probit emission.
    """
    q = float(np.clip(tilde_q, 1e-12, 1.0 - 1e-12))
    weights = np.array([(1 - q) ** 2, 2 * q * (1 - q), q ** 2])
    op_probs = np.stack([
        ordered_probit_probs(kappa, a * m / 2.0) for m in (0, 1, 2)
    ])  # (3, K)
    return weights @ op_probs  # (K,)


def three_state_leaf_probs_with_de(
    tilde_q: float, a: float, kappa: np.ndarray, d_e: float
) -> np.ndarray:
    """Same as `three_state_leaf_probs` but with d_e scaling eta.

    eta_m = a * d_e * m / 2.  At d_e = 1 reduces to the production model.
    """
    return three_state_leaf_probs(tilde_q, a * d_e, kappa)


def expected_fisher_de_per_rating(
    a: float,
    kappa: np.ndarray,
    alpha_j: float,
    delta_j: float,
    c_value: float,
    eps: float = 1e-3,
) -> float:
    """Per-rating expected Fisher info for d_e at d_e = 1.

    Computes E_y[(d log p(y | d_e) / d d_e)^2] using a central
    difference w.r.t. d_e on the log leaf likelihood.  Returns nats^2.
    Summing across ratings of an indicator gives that indicator's
    contribution to total Fisher info on this rater's d_e.
    """
    tilde_q = float(np.clip(alpha_j + delta_j * c_value, 1e-12, 1.0 - 1e-12))
    p_lo = three_state_leaf_probs_with_de(tilde_q, a, kappa, 1.0 - eps)
    p_hi = three_state_leaf_probs_with_de(tilde_q, a, kappa, 1.0 + eps)
    p_at = three_state_leaf_probs_with_de(tilde_q, a, kappa, 1.0)
    p_lo = np.clip(p_lo, 1e-30, 1.0)
    p_hi = np.clip(p_hi, 1e-30, 1.0)
    score = (np.log(p_hi) - np.log(p_lo)) / (2.0 * eps)
    return float(np.sum(p_at * score ** 2))


# --------------------------------------------------------------------------
# Affine propagation from truth.json (single-draw)
# --------------------------------------------------------------------------

def affine_from_truth(truth: Dict) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Return (indicator_keys, alpha_j, delta_j) from truth.json edge_betas."""
    config = ModelConfig()
    stances = load_data(config)
    stance_data = next(iter(s for s in stances if s["name"] == "Global Workspace Theory"))
    edge_betas = truth["edge_betas"]
    bp = {k: np.asarray([float(v["beta_pres"])]) for k, v in edge_betas.items()}
    ba = {k: np.asarray([float(v["beta_abs"])]) for k, v in edge_betas.items()}
    # Build node_to_varname identity-like mapping (we only need the dict's keys)
    node_to_varname = {k: k for k in edge_betas.keys()}
    intercepts, slopes, indicators = propagate_affine_indicator_coefficients_from_draws(
        stance_data, node_to_varname, bp, ba,
    )
    keys = [spec.node_key for spec in indicators]
    return keys, intercepts[0], slopes[0]


# --------------------------------------------------------------------------
# GWT design — walk data_cache.json to recover (expert, system, indicator) cells
# --------------------------------------------------------------------------

def _walk_indicators(node: Dict, ancestor: Tuple[str, ...]) -> List[Tuple[str, Dict]]:
    """Yield (node_key, indicator_node) for every Indicator-typed node."""
    out: List[Tuple[str, Dict]] = []
    here = ancestor + (node["name"],)
    if node.get("type", "").lower() == "indicator":
        out.append((node_key(ancestor, node["name"]), node))
        return out
    for child in node.get("evidencers", []):
        out.extend(_walk_indicators(child, here))
    return out


def realistic_design(
    data_cache: List[Dict],
) -> Dict[Tuple[str, str, str], int]:
    """Map (expert_name, system_name, indicator_key) -> n_ratings (= 1 here).

    Mirrors the actual seed-06 GWT design: a rater appears in a (system,
    indicator) cell iff data_cache lists their name with rating != -1.
    """
    gwt = next(s for s in data_cache if s["name"] == "Global Workspace Theory")
    ancestor = (gwt["name"],)
    cells: Dict[Tuple[str, str, str], int] = {}
    for child in gwt.get("evidencers", []):
        for ind_key, ind_node in _walk_indicators(child, ancestor):
            obs = ind_node.get("observations", {})
            for system, payload in obs.items():
                names = payload.get("names", [])
                values = payload.get("values", [])
                for name, val in zip(names, values):
                    try:
                        if float(val) < 0:
                            continue
                    except (TypeError, ValueError):
                        continue
                    cells[(name, system, ind_key)] = cells.get((name, system, ind_key), 0) + 1
    return cells


def idealised_design(
    realistic_cells: Dict[Tuple[str, str, str], int],
) -> Dict[Tuple[str, str, str], int]:
    """Every expert rates every (system, indicator) cell exactly once."""
    experts = sorted({e for (e, _, _) in realistic_cells})
    systems = sorted({s for (_, s, _) in realistic_cells})
    indicators = sorted({i for (_, _, i) in realistic_cells})
    return {(e, s, i): 1 for e in experts for s in systems for i in indicators}


# --------------------------------------------------------------------------
# Per-expert Fisher
# --------------------------------------------------------------------------

def per_expert_fisher(
    cells: Dict[Tuple[str, str, str], int],
    indicator_keys: List[str],
    alpha: np.ndarray,
    delta: np.ndarray,
    a: float,
    kappa: np.ndarray,
    true_C_by_system: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per_expert_summary, per_expert_per_system_long)."""
    alpha_by_key = dict(zip(indicator_keys, alpha))
    delta_by_key = dict(zip(indicator_keys, delta))

    # Pre-compute per-rating Fisher per (system, indicator) cell
    cell_fisher: Dict[Tuple[str, str], float] = {}
    for system, c in true_C_by_system.items():
        for key in indicator_keys:
            f = expected_fisher_de_per_rating(
                a, kappa, alpha_by_key[key], delta_by_key[key], c,
            )
            cell_fisher[(system, key)] = f

    rows: List[Dict] = []
    long_rows: List[Dict] = []
    by_expert: Dict[str, float] = {}
    by_expert_systems: Dict[str, set] = {}
    by_expert_n: Dict[str, int] = {}
    by_expert_system_F: Dict[Tuple[str, str], float] = {}
    by_expert_system_n: Dict[Tuple[str, str], int] = {}

    for (expert, system, key), n in cells.items():
        f = cell_fisher[(system, key)] * n
        by_expert[expert] = by_expert.get(expert, 0.0) + f
        by_expert_systems.setdefault(expert, set()).add(system)
        by_expert_n[expert] = by_expert_n.get(expert, 0) + n
        ks = (expert, system)
        by_expert_system_F[ks] = by_expert_system_F.get(ks, 0.0) + f
        by_expert_system_n[ks] = by_expert_system_n.get(ks, 0) + n

    for expert in sorted(by_expert):
        rows.append({
            "expert": expert,
            "n_systems_rated": len(by_expert_systems[expert]),
            "n_ratings": by_expert_n[expert],
            "fisher_de_total_nats2": by_expert[expert],
            "fisher_de_per_rating_nats2": by_expert[expert] / max(by_expert_n[expert], 1),
        })
    for (expert, system), F in sorted(by_expert_system_F.items()):
        long_rows.append({
            "expert": expert,
            "system": system,
            "n_ratings": by_expert_system_n[(expert, system)],
            "fisher_de_nats2": F,
        })

    return pd.DataFrame(rows), pd.DataFrame(long_rows)


def main() -> None:
    truth = json.loads(TRUTH_JSON.read_text())
    obs = truth["observation_parameters"]
    a = float(obs["a"])
    kappa = np.asarray(obs["kappa"], dtype=float)
    true_C = truth["true_C_by_system"]

    data_cache = json.loads((REPO_ROOT / "data_cache.json").read_text())

    indicator_keys, alpha, delta = affine_from_truth(truth)

    realistic = realistic_design(data_cache)
    idealised = idealised_design(realistic)

    print("=" * 70)
    print(f"a = {a:.4f}, kappa = {np.array2string(kappa, precision=3)}")
    print(f"true C = {true_C}")
    print(f"n_indicators = {len(indicator_keys)}")
    print(f"realistic cells: {len(realistic)}, idealised cells: {len(idealised)}")
    print("=" * 70)

    real_summary, real_long = per_expert_fisher(
        realistic, indicator_keys, alpha, delta, a, kappa, true_C,
    )
    ideal_summary, ideal_long = per_expert_fisher(
        idealised, indicator_keys, alpha, delta, a, kappa, true_C,
    )

    print("\nREALISTIC:")
    print(real_summary.to_string(index=False))
    print("\nIDEALISED (every expert rates every cell):")
    print(ideal_summary.to_string(index=False))

    real_summary.to_csv(OUT_DIR / "fisher_de_realistic.csv", index=False)
    ideal_summary.to_csv(OUT_DIR / "fisher_de_idealised.csv", index=False)
    real_long.to_csv(OUT_DIR / "fisher_de_realistic_per_system.csv", index=False)
    ideal_long.to_csv(OUT_DIR / "fisher_de_idealised_per_system.csv", index=False)
    print(f"\nWrote CSVs to {OUT_DIR}")


if __name__ == "__main__":
    main()
