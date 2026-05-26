"""Sanity unit check for direct_q leaf: at a→0, predicted category distribution
should collapse to a kappa-only ordered probit independent of q_eff.

Compare pt_direct_q_ll evaluated at q_eff=0.0 and q_eff=1.0 with a tiny `a`.
The two log-likelihoods should agree to high numerical precision.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytensor
import pytensor.tensor as pt

from dcm_model import pt_direct_q_ll


def main() -> None:
    rng = np.random.default_rng(42)
    n_obs = 30
    n_experts = 5
    K = 7
    ratings = rng.integers(0, K, size=n_obs).astype(np.int64)
    expert_idx = rng.integers(0, n_experts, size=n_obs).astype(np.int64)
    kappa_vals = np.linspace(-2.0, 2.0, K - 1).astype(np.float64)
    b_vals = rng.normal(0.0, 0.3, size=n_experts).astype(np.float64)

    a_sym = pt.scalar("a")
    q_sym = pt.scalar("q")
    kappa_sym = pt.constant(kappa_vals, dtype="floatX")
    b_sym = pt.constant(b_vals, dtype="floatX")

    ll = pt_direct_q_ll(
        ratings, expert_idx, q_sym, a_sym, kappa_sym, b_sym,
        kappa_by_expert=None, sigma_by_expert=None,
    )
    f = pytensor.function([a_sym, q_sym], ll)

    # At a -> 0, ll should be invariant in q_eff.
    a_tiny = 1e-9
    ll_q0 = float(f(a_tiny, 0.0))
    ll_q1 = float(f(a_tiny, 1.0))
    ll_qmid = float(f(a_tiny, 0.5))
    diff = max(abs(ll_q1 - ll_q0), abs(ll_qmid - ll_q0))
    print(f"a={a_tiny:.0e}: ll(q=0)={ll_q0:.8f}, ll(q=0.5)={ll_qmid:.8f}, "
          f"ll(q=1)={ll_q1:.8f}, max|Δ|={diff:.2e}")
    assert diff < 1e-6, (
        f"At a→0 the direct_q log-likelihood should be q-invariant; got max|Δ|={diff:.2e}"
    )

    # Sanity: at a=2.0, ll should differ between q=0 and q=1.
    ll_q0_big = float(f(2.0, 0.0))
    ll_q1_big = float(f(2.0, 1.0))
    sep = abs(ll_q1_big - ll_q0_big)
    print(f"a=2.0:  ll(q=0)={ll_q0_big:.4f}, ll(q=1)={ll_q1_big:.4f}, |Δ|={sep:.4f}")
    assert sep > 1e-3, "At a=2 we should see q dependence."

    print("PASS: direct_q collapses to kappa-only ordered probit at a=0.")


if __name__ == "__main__":
    main()
