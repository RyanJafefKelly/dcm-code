"""Corrected frozen-theta sigma_Derek sweep (recomputes p_m responsibilities).

Unlike `sigma_derek_frozen_theta_sweep.py` (which reused persisted p_m values
from the sigma=1 fit), this version recomputes posterior component
responsibilities under each candidate sigma:

    p_tilde_m(sigma) ∝ w_m(q_j) * prod_{i in I_j} P_OP(r_i | eta_m, kappa, sigma_{e(i,j)})

where q_j is tree-implied (from the saved ``..._p`` deterministic), sigma_{e(i,j)}
is 1 for all experts except Derek (who gets the candidate sigma_D), and the
product runs over *all* ratings of indicator j (not just Derek's). The predictive
mixture for Derek's rating of indicator j is then

    P(r=k) = sum_m p_tilde_m_j(sigma) * P_OP(k | eta_m, kappa, sigma_D).

Output: results/gwt_binary_three_state/analysis/sigma_derek_sweep_corrected.md
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import arviz as az
import numpy as np
from scipy.stats import norm

from dcm_model import (
    EvidenceProcessor,
    ModelConfig,
    MultiSystemDataProcessor,
    MultiSystemModelBuilder,
    load_data,
)
from dcm_ppc import _extract_obs_layer_draws, _sample_draw_indices
from gwt_three_state_indicator_analysis import (
    STANCE,
    SYSTEM_CONFIGS_VALIDATED,
    SYSTEM_DISPLAY,
)

RESULTS_DIR = Path("results/gwt_binary_three_state")
ANALYSIS_DIR = RESULTS_DIR / "analysis"
DEREK = "Derek Shiller"
DEREK_SYSTEMS = ("Human", "2024 Leading Chat LLMs", "ELIZA")
SIGMA_GRID = (0.5, 0.6, 0.7, 0.8, 1.0)


def _op_probs_sigma(kappa: np.ndarray, eta: np.ndarray, sigma: np.ndarray, K: int) -> np.ndarray:
    """Ordered-probit category probabilities.

    kappa: (S, K-1); eta: (S,); sigma: (S,) -- per-draw.
    Returns (S, K).
    """
    # (S, K-1): (kappa_k - eta_s) / sigma_s
    cum = norm.cdf((kappa - eta[:, None]) / sigma[:, None])
    zeros = np.zeros((cum.shape[0], 1))
    ones = np.ones((cum.shape[0], 1))
    return np.diff(np.concatenate([zeros, cum, ones], axis=1), axis=1)


def _op_probs_sigma_per_obs(
    kappa: np.ndarray,  # (S, K-1)
    eta: np.ndarray,  # (S,) or scalar
    sigma_by_obs: np.ndarray,  # (N_obs, S) -- per-observation, per-draw
    K: int,
) -> np.ndarray:
    """Returns (N_obs, S, K)."""
    # Broadcast: (N_obs, S, K-1)
    eta_arr = np.asarray(eta)
    if eta_arr.ndim == 0:
        eta_broad = np.broadcast_to(eta_arr, (kappa.shape[0],))
    else:
        eta_broad = eta_arr
    # cum shape (N_obs, S, K-1): kappa[None, S, K-1] - eta[None, S, None] --> (1, S, K-1); divide by sigma (N, S, 1)
    kappa_minus_eta = kappa[None, :, :] - eta_broad[None, :, None]  # (1, S, K-1)
    kappa_minus_eta_scaled = kappa_minus_eta / sigma_by_obs[:, :, None]  # (N_obs, S, K-1)
    cum = norm.cdf(kappa_minus_eta_scaled)
    zeros = np.zeros((cum.shape[0], cum.shape[1], 1))
    ones = np.ones((cum.shape[0], cum.shape[1], 1))
    cum_full = np.concatenate([zeros, cum, ones], axis=-1)
    return np.diff(cum_full, axis=-1)


def run_corrected_sweep(
    idata, builder, processor,
    expert_name: str,
    system_name: str,
    sigma_grid: Tuple[float, ...],
    n_draws: int = 300,
    seed: int = 0,
) -> List[Dict]:
    K = builder.config.N_CATEGORIES
    post = idata.posterior
    n_experts = len(processor.expert_names)
    expert_idx = processor.expert_to_idx[expert_name]
    sp = builder._sys_prefix(system_name)
    sys_obs = processor.system_observations.get(system_name, {})

    S_total = np.asarray(post["a"].values).reshape(-1).shape[0]
    idx = _sample_draw_indices(S_total, n_draws, seed)
    a_draws, b_draws, kappa_draws, _ = _extract_obs_layer_draws(post, idx, n_experts, K)
    S = len(idx)

    # Per-indicator: collect full rating list + tree-implied q draws + flag Derek's rating
    indicator_rows: List[Dict] = []
    for nkey, obs_list in sys_obs.items():
        varname = builder.node_to_varname.get(nkey)
        if varname is None:
            continue
        q_name = f"{sp}__{varname}_p"
        if q_name not in post.data_vars:
            continue
        q_draws = np.asarray(post[q_name].values).reshape(-1)[idx]  # (S,)
        ratings_all = np.array([r for _, r in obs_list], dtype=np.int64)
        experts_all = np.array([e for e, _ in obs_list], dtype=np.int64)
        derek_ratings = [r for e, r in obs_list if e == expert_idx]
        if not derek_ratings:
            continue
        indicator_rows.append(
            {
                "nkey": nkey,
                "ratings_all": ratings_all,
                "experts_all": experts_all,
                "q_draws": q_draws,
                "derek_ratings": np.array(derek_ratings, dtype=np.int64),
            }
        )

    if not indicator_rows:
        return []

    all_derek_obs = np.concatenate([r["derek_ratings"] for r in indicator_rows])

    results: List[Dict] = []
    mid_lo, mid_hi = K // 2 - 1, K // 2 + 1

    for sigma_D in sigma_grid:
        # Build per-draw sigma_by_expert (n_experts, S)
        sigma_by_expert = np.ones((n_experts, S))
        sigma_by_expert[expert_idx, :] = sigma_D

        # Per draw: kappa (S, K-1), a (S,). Eta values for m=0,1,2: 0, a/2, a
        eta_values = [np.zeros(S), a_draws * 0.5, a_draws]

        # Aggregate predictive histogram across all of Derek's ratings
        pred_hist_total = np.zeros(K)
        total_n = 0

        for ind in indicator_rows:
            q = ind["q_draws"]  # (S,)
            ratings_all = ind["ratings_all"]  # (N_obs,)
            experts_all = ind["experts_all"]  # (N_obs,)
            n_derek = len(ind["derek_ratings"])
            # Per-observation sigma draws (N_obs, S)
            sigma_obs = sigma_by_expert[experts_all, :]  # (N_obs, S)

            # Compute L_m_j_s = sum_i log P_OP(r_i | eta_m, kappa, sigma_obs_i)
            # Shape: (S,)
            log_L = np.zeros((3, S))
            for m_idx, eta_m in enumerate(eta_values):
                # (N_obs, S, K) category probs per observation/draw
                pk = _op_probs_sigma_per_obs(
                    kappa_draws, eta_m, sigma_obs, K
                )
                # P(r_i) per observation per draw: pick the observed category
                # pk[n, s, ratings_all[n]]
                pk_r = pk[np.arange(len(ratings_all)), :, ratings_all]  # (N_obs, S)
                pk_r = np.clip(pk_r, 1e-20, 1.0)
                log_L[m_idx] = np.log(pk_r).sum(axis=0)  # (S,)

            # Log prior weights per m, per draw
            log_w = np.stack([
                2.0 * np.log(np.clip(1 - q, 1e-12, 1)),
                np.log(2.0) + np.log(np.clip(q, 1e-12, 1)) + np.log(np.clip(1 - q, 1e-12, 1)),
                2.0 * np.log(np.clip(q, 1e-12, 1)),
            ], axis=0)  # (3, S)

            # Posterior responsibilities per draw
            log_unnorm = log_w + log_L  # (3, S)
            log_norm = np.logaddexp(np.logaddexp(log_unnorm[0], log_unnorm[1]), log_unnorm[2])  # (S,)
            p_m = np.exp(log_unnorm - log_norm[None, :])  # (3, S)

            # Derek's predictive category distribution per draw
            # P(r=k | Derek, indicator j, draw s, sigma_D)
            #   = sum_m p_m_j_s * P_OP(k | eta_m, kappa_s, sigma_D)
            sigma_derek_draws = np.full(S, sigma_D)
            pred_per_draw = np.zeros((S, K))
            for m_idx, eta_m in enumerate(eta_values):
                pk_m = _op_probs_sigma(kappa_draws, eta_m, sigma_derek_draws, K)  # (S, K)
                pred_per_draw += p_m[m_idx][:, None] * pk_m

            # Aggregate over draws (mean) for this indicator, then aggregate across Derek's
            # multiple ratings of this same indicator (if any)
            mean_pred_k = pred_per_draw.mean(axis=0)  # (K,)
            pred_hist_total += n_derek * mean_pred_k
            total_n += n_derek

        pred_hist = pred_hist_total / max(total_n, 1)

        # Observed
        obs_hist = np.bincount(all_derek_obs, minlength=K) / len(all_derek_obs)
        obs_left = float(obs_hist[0])
        obs_right = float(obs_hist[-1])
        obs_mid = float(obs_hist[mid_lo:mid_hi + 1].sum())
        obs_mean = float(all_derek_obs.mean())

        pred_left = float(pred_hist[0])
        pred_right = float(pred_hist[-1])
        pred_mid = float(pred_hist[mid_lo:mid_hi + 1].sum())
        pred_mean = float(np.sum(pred_hist * np.arange(K)))

        results.append({
            "sigma_derek": sigma_D,
            "n_obs": len(all_derek_obs),
            "obs_hist": obs_hist,
            "pred_hist": pred_hist,
            "obs_left": obs_left,
            "obs_right": obs_right,
            "obs_mid": obs_mid,
            "obs_mean": obs_mean,
            "pred_left": pred_left,
            "pred_right": pred_right,
            "pred_mid": pred_mid,
            "pred_mean": pred_mean,
            "delta_left": pred_left - obs_left,
            "delta_right": pred_right - obs_right,
            "delta_mid": pred_mid - obs_mid,
        })

    return results


def main() -> None:
    print("Loading three-state idata and context...")
    config = ModelConfig(
        USE_EXPERT_SHIFTS=False,
        USE_HIERARCHICAL_EXPERT_CUTPOINTS=False,
        INDICATOR_STATE_MODEL="three_state",
    )
    stance_data = next(item for item in load_data(config) if item["name"] == STANCE)
    processor = MultiSystemDataProcessor(config)
    processor.process(stance_data, [s for s, _ in SYSTEM_CONFIGS_VALIDATED])
    builder = MultiSystemModelBuilder(
        config, EvidenceProcessor(config), processor, list(SYSTEM_CONFIGS_VALIDATED)
    )
    builder.build_model(stance_data)
    idata = az.from_netcdf(RESULTS_DIR / "three_state_anchored.nc")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    results_by_sys: Dict[str, List[Dict]] = {}
    print("Running corrected sweep across Derek's three systems...")
    for sys_name in DEREK_SYSTEMS:
        print(f"  - {sys_name}...")
        results_by_sys[sys_name] = run_corrected_sweep(
            idata, builder, processor, DEREK, sys_name, SIGMA_GRID,
            n_draws=300, seed=0,
        )

    lines = []
    lines.append("# Corrected frozen-θ σ_Derek sweep\n")
    lines.append(
        "**Corrected vs original:** component responsibilities $\\tilde p_m(\\sigma)$ "
        "are recomputed under each candidate σ using tree-implied $q_j$ + the full "
        "rating list for each indicator, rather than reusing persisted $p_m$ from the "
        "σ=1 fit.\n"
    )
    lines.append(
        "Derek's σ varies; all other experts held at σ=1. $(a, \\kappa, q_j)$ frozen "
        "at the three-state posterior.\n"
    )

    for sys_name in DEREK_SYSTEMS:
        rows = results_by_sys.get(sys_name, [])
        if not rows:
            continue
        sys_label = SYSTEM_DISPLAY.get(sys_name, sys_name)
        r0 = rows[0]
        lines.append(
            f"\n## Derek × {sys_label}  (n = {r0['n_obs']})\n"
        )
        lines.append(
            f"Observed: mean = {r0['obs_mean']:.2f}, "
            f"P(r=1) = {r0['obs_left']:.3f}, P(r=7) = {r0['obs_right']:.3f}, "
            f"P(r ∈ {{3,4,5}}) = {r0['obs_mid']:.3f}\n"
        )
        lines.append("| σ_D | pred_left | Δleft | pred_right | Δright | pred_mid | Δmid | pred_mean |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                f"| {r['sigma_derek']:.1f} | "
                f"{r['pred_left']:.3f} | {r['delta_left']:+.3f} | "
                f"{r['pred_right']:.3f} | {r['delta_right']:+.3f} | "
                f"{r['pred_mid']:.3f} | {r['delta_mid']:+.3f} | "
                f"{r['pred_mean']:.2f} |"
            )

    # Interpretation
    lines.append("\n## Interpretation (relative to uncorrected sweep)\n")
    lines.append(
        "Under the corrected sweep, smaller σ_Derek sharpens Derek's emissions AND "
        "pulls the posterior state assignment toward whichever $m$-component gives "
        "the sharpest match to the observed extremes. That is the effect the "
        "uncorrected sweep missed.\n"
    )
    for sys_name in DEREK_SYSTEMS:
        rows = results_by_sys.get(sys_name, [])
        if not rows:
            continue
        # Find best sigma by |delta_left| + |delta_right| + |delta_mid|
        best = min(
            rows,
            key=lambda r: abs(r["delta_left"]) + abs(r["delta_right"]) + abs(r["delta_mid"]),
        )
        lines.append(
            f"- **Derek × {SYSTEM_DISPLAY.get(sys_name, sys_name)}**: "
            f"best σ_D ≈ {best['sigma_derek']:.1f} "
            f"(Δleft {best['delta_left']:+.3f}, "
            f"Δright {best['delta_right']:+.3f}, "
            f"Δmid {best['delta_mid']:+.3f})"
        )

    out = ANALYSIS_DIR / "sigma_derek_sweep_corrected.md"
    out.write_text("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
