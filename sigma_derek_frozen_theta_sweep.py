"""Frozen-theta sigma_Derek sweep against the persisted three-state idata.

Pre-fit mechanistic check. Uses the three_state_anchored posterior draws of
(a, kappa, q_j) and evaluates the PPC at Derek's focus cells (Human / ELIZA)
and at Derek × LLMs under several fixed values of sigma_Derek. Other experts
remain at sigma = 1.

The emission CDF becomes
    P(r <= k | eta, sigma) = Phi((kappa_k - eta) / sigma),
so we reuse all the existing component-weight machinery and only substitute
the ordered-probit category probabilities.

Output:
    results/gwt_binary_three_state/analysis/sigma_derek_sweep.md
        Signed tail + histogram summary for each candidate sigma.
"""

from __future__ import annotations

from collections import defaultdict
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
from dcm_ppc import (
    _extract_component_weights_for_indicator,
    _extract_obs_layer_draws,
    _sample_draw_indices,
)
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


def _op_probs(kappa: np.ndarray, eta: np.ndarray, sigma: float, K: int) -> np.ndarray:
    """Ordered-probit category probabilities with scale sigma. (S, K)."""
    cum = norm.cdf((kappa - eta[:, None]) / sigma)
    zeros = np.zeros((cum.shape[0], 1))
    ones = np.ones((cum.shape[0], 1))
    cum_full = np.concatenate([zeros, cum, ones], axis=1)
    return np.diff(cum_full, axis=1)


def _expert_component_predictives_sigma(
    a_draws, b_draws, kappa_draws, expert_idx, K, sigma,
) -> np.ndarray:
    """(3, S, K) component emission probabilities under expert-specific sigma."""
    eta_base = b_draws[:, expert_idx]
    kappa_e = kappa_draws
    eta_values = [eta_base, eta_base + 0.5 * a_draws, eta_base + a_draws]
    return np.stack([_op_probs(kappa_e, eta, sigma, K) for eta in eta_values], axis=0)


def _sample_cell(
    weights_stack: np.ndarray,
    pk_components: np.ndarray,
    rng: np.random.Generator,
    K: int,
    n_obs: int,
) -> Dict[str, np.ndarray]:
    """Sample predictive ratings per draw and return shape diagnostics."""
    n_components, _, S = weights_stack.shape
    pred_hist = np.zeros((S, K))
    pred_left = np.zeros(S)
    pred_right = np.zeros(S)
    pred_mid = np.zeros(S)
    mid_lo, mid_hi = K // 2 - 1, K // 2 + 1
    for s in range(S):
        mix = np.zeros((n_obs, K))
        for c in range(n_components):
            mix += weights_stack[c, :, s, None] * pk_components[c, s, None, :]
        mix = mix / np.clip(mix.sum(axis=1, keepdims=True), 1e-12, None)
        cum = np.cumsum(mix, axis=1)
        u = rng.random(n_obs)[:, None]
        sampled = (u < cum).argmax(axis=1)
        pred_hist[s] = np.bincount(sampled, minlength=K) / n_obs
        pred_left[s] = float(np.mean(sampled == 0))
        pred_right[s] = float(np.mean(sampled == K - 1))
        pred_mid[s] = float(np.mean((sampled >= mid_lo) & (sampled <= mid_hi)))
    return {
        "pred_hist_mean": pred_hist.mean(axis=0),
        "pred_hist_lo": np.percentile(pred_hist, 3, axis=0),
        "pred_hist_hi": np.percentile(pred_hist, 97, axis=0),
        "pred_left_mean": float(pred_left.mean()),
        "pred_left_lo": float(np.percentile(pred_left, 3)),
        "pred_left_hi": float(np.percentile(pred_left, 97)),
        "pred_right_mean": float(pred_right.mean()),
        "pred_right_lo": float(np.percentile(pred_right, 3)),
        "pred_right_hi": float(np.percentile(pred_right, 97)),
        "pred_mid_mean": float(pred_mid.mean()),
    }


def run_sweep_for_cell(
    idata,
    builder,
    processor,
    expert_name: str,
    system_name: str,
    sigma_derek_grid,
    n_draws: int = 500,
    seed: int = 0,
) -> List[Dict]:
    """Per-sigma sweep for one (expert, system) cell."""
    K = builder.config.N_CATEGORIES
    post = idata.posterior
    n_experts = len(processor.expert_names)
    expert_idx = processor.expert_to_idx[expert_name]
    sp = builder._sys_prefix(system_name)
    sys_obs = processor.system_observations.get(system_name, {})

    S_total = np.asarray(post["a"].values).reshape(-1).shape[0]
    idx = _sample_draw_indices(S_total, n_draws, seed)
    a_draws, b_draws, kappa_draws, _ = _extract_obs_layer_draws(post, idx, n_experts, K)
    rng = np.random.default_rng(seed)

    # Collect observations + weights for this expert in this system
    entries: List[Tuple[int, np.ndarray]] = []
    for nkey, obs_list in sys_obs.items():
        varname = builder.node_to_varname.get(nkey)
        if varname is None:
            continue
        weights = _extract_component_weights_for_indicator(
            post, idx, sp, varname, state_model="three_state",
            indicator_prob_source="pz1",
        )
        if weights is None:
            continue
        for e_idx, rating in obs_list:
            if e_idx == expert_idx:
                entries.append((int(rating), weights))
    if not entries:
        return []
    obs = np.asarray([r for r, _ in entries], dtype=int)
    weights_stack = np.stack([w for _, w in entries], axis=1)

    # Observed summaries
    obs_hist = np.bincount(obs, minlength=K) / len(obs)
    mid_lo, mid_hi = K // 2 - 1, K // 2 + 1
    obs_stats = {
        "obs_left": float(np.mean(obs == 0)),
        "obs_right": float(np.mean(obs == K - 1)),
        "obs_mid": float(np.mean((obs >= mid_lo) & (obs <= mid_hi))),
        "obs_mean": float(obs.mean()),
        "obs_hist": obs_hist,
        "n_obs": len(obs),
    }

    results: List[Dict] = []
    for sigma in sigma_derek_grid:
        pk_components = _expert_component_predictives_sigma(
            a_draws, b_draws, kappa_draws, expert_idx, K, sigma
        )
        stats = _sample_cell(weights_stack, pk_components, rng, K, len(obs))
        row = {"sigma_derek": sigma, **obs_stats, **stats}
        row["delta_left"] = stats["pred_left_mean"] - obs_stats["obs_left"]
        row["delta_right"] = stats["pred_right_mean"] - obs_stats["obs_right"]
        row["delta_mid"] = stats["pred_mid_mean"] - obs_stats["obs_mid"]
        results.append(row)
    return results


def main() -> None:
    print("Loading three-state idata and building context...")
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

    print("Running sigma_Derek sweep across Derek's three systems...")
    all_results: Dict[str, List[Dict]] = {}
    for sys_name in DEREK_SYSTEMS:
        all_results[sys_name] = run_sweep_for_cell(
            idata, builder, processor, DEREK, sys_name, SIGMA_GRID
        )

    # Write markdown
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Frozen-theta sigma_Derek sweep\n")
    lines.append(
        "Using the persisted three-state idata, we hold all parameters fixed "
        "and vary only sigma_Derek -- the noise scale of Derek's ordered-probit "
        "emission. Other experts remain at sigma = 1. The table below reports "
        "signed tail and middle-mass errors for each candidate sigma.\n"
    )
    lines.append(
        "**Goal:** verify that sigma_Derek < 1 moves Derek x Human cat-7 and "
        "Derek x ELIZA cat-1 PPCs toward observed, without degrading Derek x "
        "LLMs. If this mechanism is credible at fixed theta, the full "
        "USE_EXPERT_SCALES refit is worth doing.\n"
    )

    for sys_name in DEREK_SYSTEMS:
        sys_label = SYSTEM_DISPLAY.get(sys_name, sys_name)
        rows = all_results[sys_name]
        if not rows:
            lines.append(f"\n## Derek x {sys_label}\n\n(no data)\n")
            continue
        obs_left = rows[0]["obs_left"]
        obs_right = rows[0]["obs_right"]
        obs_mid = rows[0]["obs_mid"]
        obs_mean = rows[0]["obs_mean"]
        n = rows[0]["n_obs"]
        lines.append(
            f"\n## Derek × {sys_label}  (n = {n})\n"
        )
        lines.append(
            f"Observed: mean rating = {obs_mean:.2f}, "
            f"P(r=1) = {obs_left:.3f}, P(r=7) = {obs_right:.3f}, "
            f"P(r in {{3,4,5}}) = {obs_mid:.3f}\n"
        )
        lines.append(
            "| sigma_D | pred_left | Δleft | pred_right | Δright | pred_mid | Δmid |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                f"| {r['sigma_derek']:.1f} | "
                f"{r['pred_left_mean']:.3f} | {r['delta_left']:+.3f} | "
                f"{r['pred_right_mean']:.3f} | {r['delta_right']:+.3f} | "
                f"{r['pred_mid_mean']:.3f} | {r['delta_mid']:+.3f} |"
            )

    # Per-sigma rating histogram for Derek x LLMs
    llm_rows = all_results.get("2024 Leading Chat LLMs", [])
    if llm_rows:
        lines.append("\n## Derek × LLMs rating histograms (per sigma_D)\n")
        lines.append(
            f"Observed histogram (categories 1..7): "
            f"{[f'{v:.2f}' for v in llm_rows[0]['obs_hist']]}\n"
        )
        for r in llm_rows:
            lines.append(
                f"- sigma_D = {r['sigma_derek']:.1f}: "
                f"{[f'{v:.2f}' for v in r['pred_hist_mean']]}"
            )

    # Short interpretation
    lines.append("\n## Quick interpretation\n")
    best_by_cell = {}
    for sys_name in DEREK_SYSTEMS:
        rows = all_results.get(sys_name, [])
        if not rows:
            continue
        # Best sigma = minimises sum of |delta_left| + |delta_right| + |delta_mid|
        scored = [
            (r, abs(r["delta_left"]) + abs(r["delta_right"]) + abs(r["delta_mid"]))
            for r in rows
        ]
        best = min(scored, key=lambda x: x[1])[0]
        best_by_cell[sys_name] = best["sigma_derek"]
        lines.append(
            f"- **Derek × {SYSTEM_DISPLAY.get(sys_name, sys_name)}**: "
            f"best sigma ≈ {best['sigma_derek']:.1f} "
            f"(Δleft {best['delta_left']:+.3f}, "
            f"Δright {best['delta_right']:+.3f}, "
            f"Δmid {best['delta_mid']:+.3f})"
        )
    lines.append("")
    consistent = len(set(best_by_cell.values())) == 1
    if consistent:
        s = next(iter(best_by_cell.values()))
        lines.append(
            f"All three Derek cells prefer sigma_D ≈ {s}. That is strong "
            "evidence the sharpness mechanism is consistent across Derek's "
            "three systems, not just fitting the reference cells at the "
            "expense of the LLM spread. Worth proceeding with the full "
            "USE_EXPERT_SCALES refit."
        )
    else:
        lines.append(
            "Best sigma differs across Derek's three cells. That means a single "
            "sigma_D parameter cannot fully reconcile Derek's Human / ELIZA "
            "extremes with his LLM spread -- there is a genuine tension that "
            "a one-parameter scale adjustment cannot resolve. Worth inspecting "
            "the discrepancies before committing to a refit."
        )

    out = ANALYSIS_DIR / "sigma_derek_sweep.md"
    out.write_text("\n".join(lines))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
