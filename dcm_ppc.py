"""Per-expert posterior predictive checks for the joint multi-system DCM.

Because the ordinal likelihood is added via ``pm.Potential`` (with the latent
indicator state marginalised analytically), PyMC's generic
``sample_posterior_predictive`` cannot replicate ratings directly. This module
implements a small custom PPC that uses the deterministics the joint builder
already exposes:

    {system_prefix}__{indicator_varname}_pz1

which is the data-conditioned posterior probability P(z_j=1 | ratings, theta)
for each indicator in each system.

For each posterior draw s and each observation (indicator j, system, expert e,
rating r_obs) the predictive distribution is the mixture

    P(r = k | s) = pz1_j^(s) * P_OP(k | b_e^(s) + a^(s), kappa^(s))
                 + (1 - pz1_j^(s)) * P_OP(k | b_e^(s),            kappa^(s))

where P_OP is the ordered-probit category probability. Replicated ratings are
sampled from this categorical and aggregated by expert into:
  - 7-bin rating histograms
  - mean rating
  - proportion of extreme ratings (categories 1 and K, i.e. indices 0 and K-1)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from dcm_model import MultiSystemDataProcessor, MultiSystemModelBuilder


def _ordered_probit_probs(
    kappa: np.ndarray, eta: np.ndarray, K: int
) -> np.ndarray:
    """Ordered-probit category probabilities.

    kappa : (S, K-1) cutpoints.
    eta   : (S,) linear predictors.
    Returns (S, K).
    """
    # (S, K-1) cumulative Phi(kappa_k - eta)
    cum = norm.cdf(kappa - eta[:, None])
    zeros = np.zeros((cum.shape[0], 1))
    ones = np.ones((cum.shape[0], 1))
    cum_full = np.concatenate([zeros, cum, ones], axis=1)  # (S, K+1)
    probs = np.diff(cum_full, axis=1)  # (S, K)
    return np.clip(probs, 1e-12, 1.0)


def _resolve_indicator_prob_suffix(indicator_prob_source: str) -> str:
    source = indicator_prob_source.lower()
    if source in {"pz1", "posterior_z"}:
        return "pz1"
    if source in {"q", "q_j", "p", "tree", "tree_implied", "tree-implied"}:
        return "p"
    raise ValueError(
        "indicator_prob_source must be one of {'pz1', 'q', 'tree_implied'}"
    )


def _sample_draw_indices(S_total: int, n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n_draws >= S_total:
        return np.arange(S_total)
    return rng.choice(S_total, size=n_draws, replace=False)


def _extract_obs_layer_draws(
    post: Any,
    idx: np.ndarray,
    n_experts: int,
    K: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    a_all = np.asarray(post["a"].values).reshape(-1)
    a_draws = a_all[idx]
    S = idx.shape[0]

    if "kappa_by_expert" in post.data_vars:
        kappa_expert_all = np.asarray(post["kappa_by_expert"].values).reshape(
            -1, n_experts, K - 1
        )
        kappa_draws = np.asarray(post["kappa"].values).reshape(-1, K - 1)[idx]
        kappa_by_expert_draws = kappa_expert_all[idx]
    else:
        kappa_draws = np.asarray(post["kappa"].values).reshape(-1, K - 1)[idx]
        kappa_by_expert_draws = None

    if "b_free" in post.data_vars:
        b_free_all = np.asarray(post["b_free"].values).reshape(-1, n_experts - 1)
        b_draws = np.concatenate([np.zeros((S, 1)), b_free_all[idx]], axis=1)
    else:
        b_draws = np.zeros((S, n_experts))

    return a_draws, b_draws, kappa_draws, kappa_by_expert_draws


def _expert_predictive_components(
    a_draws: np.ndarray,
    b_draws: np.ndarray,
    kappa_draws: np.ndarray,
    kappa_by_expert_draws: Optional[np.ndarray],
    expert_idx: int,
    K: int,
) -> Tuple[np.ndarray, np.ndarray]:
    eta_z0 = b_draws[:, expert_idx]
    eta_z1 = eta_z0 + a_draws
    if kappa_by_expert_draws is None:
        kappa_e = kappa_draws
    else:
        kappa_e = kappa_by_expert_draws[:, expert_idx, :]
    pk_z0 = _ordered_probit_probs(kappa_e, eta_z0, K)
    pk_z1 = _ordered_probit_probs(kappa_e, eta_z1, K)
    return pk_z0, pk_z1


def per_expert_ppc_multisystem(
    idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
    n_draws: int = 500,
    seed: int = 42,
    indicator_prob_source: str = "pz1",
) -> Dict[str, Dict[str, Any]]:
    """Compute per-expert posterior predictive summaries for a joint DCM fit.

    Returns
    -------
    {expert_name: {
        "n_obs": int,
        "obs_hist": (K,), "pred_hist_mean": (K,), "pred_hist_lo": (K,), "pred_hist_hi": (K,),
        "obs_mean": float, "pred_mean_mean": float, "pred_mean_lo": float, "pred_mean_hi": float,
        "obs_extreme": float, "pred_extreme_mean": float, "pred_extreme_lo": float, "pred_extreme_hi": float,
    }}

    ``indicator_prob_source="pz1"`` uses the data-conditioned posterior
    indicator probabilities. ``indicator_prob_source="q"`` uses the upstream
    tree-implied indicator probabilities ``q_j`` exposed as ``..._p``.
    """
    K = builder.config.N_CATEGORIES
    post = idata.posterior
    n_experts = len(processor.expert_names)
    S_total = np.asarray(post["a"].values).reshape(-1).shape[0]
    idx = _sample_draw_indices(S_total, n_draws, seed)
    S = idx.shape[0]
    rng = np.random.default_rng(seed)
    (
        a_draws,
        b_draws,
        kappa_draws,
        kappa_by_expert_draws,
    ) = _extract_obs_layer_draws(post, idx, n_experts, K)
    prob_suffix = _resolve_indicator_prob_suffix(indicator_prob_source)

    # --- Collect (expert_idx, rating, indicator_prob_draws) across systems ---
    per_expert_obs: List[List[int]] = [[] for _ in range(n_experts)]
    per_expert_indicator_probs: List[List[np.ndarray]] = [[] for _ in range(n_experts)]

    for sys_name, sys_obs in processor.system_observations.items():
        sp = builder._sys_prefix(sys_name)
        for nkey, obs_list in sys_obs.items():
            varname = builder.node_to_varname.get(nkey)
            if varname is None:
                continue
            indicator_prob_name = f"{sp}__{varname}_{prob_suffix}"
            if indicator_prob_name not in post.data_vars:
                continue
            indicator_prob_all = np.asarray(post[indicator_prob_name].values).reshape(-1)
            indicator_prob_draws = indicator_prob_all[idx]
            for expert_idx, rating in obs_list:
                per_expert_obs[expert_idx].append(int(rating))
                per_expert_indicator_probs[expert_idx].append(indicator_prob_draws)

    results: Dict[str, Dict[str, Any]] = {}
    for e in range(n_experts):
        if not per_expert_obs[e]:
            continue
        obs = np.asarray(per_expert_obs[e], dtype=int)  # (N_e,)
        indicator_prob_mat = np.stack(per_expert_indicator_probs[e], axis=0)
        N_e = obs.shape[0]
        pk_z0, pk_z1 = _expert_predictive_components(
            a_draws,
            b_draws,
            kappa_draws,
            kappa_by_expert_draws,
            e,
            K,
        )

        pred_hist = np.zeros((S, K))
        pred_mean_rating = np.zeros(S)
        pred_extreme_prop = np.zeros(S)

        for s in range(S):
            mix = (
                indicator_prob_mat[:, s, None] * pk_z1[s, None, :]
                + (1.0 - indicator_prob_mat[:, s, None]) * pk_z0[s, None, :]
            )
            row_sums = mix.sum(axis=1, keepdims=True)
            mix = mix / row_sums

            cum = np.cumsum(mix, axis=1)
            u = rng.random(N_e)[:, None]
            sampled = (u < cum).argmax(axis=1)  # (N_e,)

            pred_hist[s] = np.bincount(sampled, minlength=K) / N_e
            pred_mean_rating[s] = float(sampled.mean())
            pred_extreme_prop[s] = float(
                np.mean((sampled == 0) | (sampled == K - 1))
            )

        obs_hist = np.bincount(obs, minlength=K) / N_e
        obs_mean = float(obs.mean())
        obs_extreme = float(np.mean((obs == 0) | (obs == K - 1)))

        results[processor.expert_names[e]] = {
            "n_obs": int(N_e),
            "obs_hist": obs_hist,
            "pred_hist_mean": pred_hist.mean(axis=0),
            "pred_hist_lo": np.percentile(pred_hist, 3, axis=0),
            "pred_hist_hi": np.percentile(pred_hist, 97, axis=0),
            "obs_mean": obs_mean,
            "pred_mean_mean": float(pred_mean_rating.mean()),
            "pred_mean_lo": float(np.percentile(pred_mean_rating, 3)),
            "pred_mean_hi": float(np.percentile(pred_mean_rating, 97)),
            "obs_extreme": obs_extreme,
            "pred_extreme_mean": float(pred_extreme_prop.mean()),
            "pred_extreme_lo": float(np.percentile(pred_extreme_prop, 3)),
            "pred_extreme_hi": float(np.percentile(pred_extreme_prop, 97)),
        }

    return results


def per_expert_tree_implied_ppc_multisystem(
    idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
    n_draws: int = 500,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Per-expert PPC using tree-implied q_j instead of data-conditioned pz1."""
    return per_expert_ppc_multisystem(
        idata,
        builder,
        processor,
        n_draws=n_draws,
        seed=seed,
        indicator_prob_source="tree_implied",
    )


def per_expert_q_ppc_multisystem(
    idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
    n_draws: int = 500,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Backward-compatible alias for the tree-implied per-expert PPC."""
    return per_expert_tree_implied_ppc_multisystem(
        idata, builder, processor, n_draws=n_draws, seed=seed
    )


def plot_per_expert_ppc(
    results: Dict[str, Dict[str, Any]],
    K: int = 7,
    expert_display: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
):
    """Small convenience plot: one row per expert, three columns.

    col 0: observed vs predicted 7-bin histogram
    col 1: observed vs predicted mean rating (with CI)
    col 2: observed vs predicted extreme-rating proportion (with CI)

    Returns the (fig, axes) pair so callers can further tweak or save.
    """
    import matplotlib.pyplot as plt

    experts = list(results.keys())
    n_e = len(experts)
    if n_e == 0:
        raise ValueError("per_expert_ppc results are empty")

    fig, axes = plt.subplots(
        n_e, 3, figsize=(10, max(2.2, 1.8 * n_e)), squeeze=False
    )
    cats = np.arange(K)
    bin_labels = [str(k + 1) for k in range(K)]
    width = 0.38

    for i, name in enumerate(experts):
        r = results[name]
        disp = (expert_display or {}).get(name, name)

        # --- histogram ---
        ax = axes[i, 0]
        ax.bar(
            cats - width / 2,
            r["obs_hist"],
            width,
            label="obs",
            color="steelblue",
            alpha=0.85,
        )
        ax.bar(
            cats + width / 2,
            r["pred_hist_mean"],
            width,
            label="pred",
            color="salmon",
            alpha=0.85,
        )
        ax.errorbar(
            cats + width / 2,
            r["pred_hist_mean"],
            yerr=[
                np.clip(r["pred_hist_mean"] - r["pred_hist_lo"], 0, None),
                np.clip(r["pred_hist_hi"] - r["pred_hist_mean"], 0, None),
            ],
            fmt="none",
            color="black",
            capsize=2,
            linewidth=0.8,
        )
        ax.set_xticks(cats)
        ax.set_xticklabels(bin_labels, fontsize=8)
        ax.set_ylabel(f"{disp}\n(n={r['n_obs']})", fontsize=9)
        if i == 0:
            ax.set_title("rating histogram", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")

        # --- mean rating (plotted on 1..K display scale) ---
        ax = axes[i, 1]
        ax.errorbar(
            [r["pred_mean_mean"] + 1],
            [0.2],
            xerr=[
                [max(r["pred_mean_mean"] - r["pred_mean_lo"], 0)],
                [max(r["pred_mean_hi"] - r["pred_mean_mean"], 0)],
            ],
            fmt="o",
            color="salmon",
            capsize=3,
            label="pred",
        )
        ax.plot(
            [r["obs_mean"] + 1],
            [-0.2],
            "D",
            color="steelblue",
            markersize=7,
            label="obs",
        )
        ax.set_xlim(1, K)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.set_xticks(np.arange(1, K + 1))
        ax.set_xticklabels([str(k) for k in range(1, K + 1)], fontsize=8)
        if i == 0:
            ax.set_title("mean rating", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")

        # --- extreme proportion ---
        ax = axes[i, 2]
        ax.errorbar(
            [r["pred_extreme_mean"]],
            [0.2],
            xerr=[
                [max(r["pred_extreme_mean"] - r["pred_extreme_lo"], 0)],
                [max(r["pred_extreme_hi"] - r["pred_extreme_mean"], 0)],
            ],
            fmt="o",
            color="salmon",
            capsize=3,
            label="pred",
        )
        ax.plot(
            [r["obs_extreme"]],
            [-0.2],
            "D",
            color="steelblue",
            markersize=7,
            label="obs",
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        if i == 0:
            ax.set_title("P(rating in {1, 7})", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig, axes


def summarise_per_expert_ppc(
    results: Dict[str, Dict[str, Any]], K: int = 7
) -> str:
    """Return a compact text summary of per-expert PPC stats."""
    lines = []
    header = (
        f"{'Expert':<20s} {'n':>4s}  "
        f"{'obs_mean':>8s}  {'pred_mean':>12s}  "
        f"{'obs_extr':>8s}  {'pred_extr':>12s}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for name, r in results.items():
        pred_m = (
            f"{r['pred_mean_mean']:.2f} "
            f"[{r['pred_mean_lo']:.2f},{r['pred_mean_hi']:.2f}]"
        )
        pred_e = (
            f"{r['pred_extreme_mean']:.2f} "
            f"[{r['pred_extreme_lo']:.2f},{r['pred_extreme_hi']:.2f}]"
        )
        lines.append(
            f"{name:<20s} {r['n_obs']:>4d}  "
            f"{r['obs_mean']:>8.2f}  {pred_m:>12s}  "
            f"{r['obs_extreme']:>8.2f}  {pred_e:>12s}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stratified per-(expert, system) PPC
# ---------------------------------------------------------------------------
#
# Motivation: the pooled per-expert PPC confounds "this expert rates
# extremely" with "this expert happened to rate mostly systems at the ends
# of the consciousness scale (e.g. Human + ELIZA), where extreme ratings are
# expected a priori". Stratifying by (expert, system) holds the system's
# indicator-presence distribution fixed within each cell, so any residual
# misfit is a real calibration issue for that expert on that system rather
# than a coverage artefact.


def per_expert_system_ppc_multisystem(
    idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
    n_draws: int = 500,
    seed: int = 42,
    indicator_prob_source: str = "pz1",
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Per-(expert, system) posterior predictive summaries.

    Uses the same mixture as ``per_expert_ppc_multisystem`` but aggregates
    observations by (expert_name, system_name) tuples. Returns an empty dict
    entry for any (expert, system) combination that has no observations.
    """
    K = builder.config.N_CATEGORIES
    post = idata.posterior
    n_experts = len(processor.expert_names)
    S_total = np.asarray(post["a"].values).reshape(-1).shape[0]
    idx = _sample_draw_indices(S_total, n_draws, seed)
    S = idx.shape[0]
    rng = np.random.default_rng(seed)
    (
        a_draws,
        b_draws,
        kappa_draws,
        kappa_by_expert_draws,
    ) = _extract_obs_layer_draws(post, idx, n_experts, K)
    prob_suffix = _resolve_indicator_prob_suffix(indicator_prob_source)

    # Bucket observations by (expert_idx, system_name) -> list of (rating, prob_draws)
    buckets: Dict[Tuple[int, str], List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for sys_name, sys_obs in processor.system_observations.items():
        sp = builder._sys_prefix(sys_name)
        for nkey, obs_list in sys_obs.items():
            varname = builder.node_to_varname.get(nkey)
            if varname is None:
                continue
            indicator_prob_name = f"{sp}__{varname}_{prob_suffix}"
            if indicator_prob_name not in post.data_vars:
                continue
            indicator_prob_all = np.asarray(post[indicator_prob_name].values).reshape(-1)
            indicator_prob_draws = indicator_prob_all[idx]
            for expert_idx, rating in obs_list:
                buckets[(expert_idx, sys_name)].append(
                    (int(rating), indicator_prob_draws)
                )

    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (e, sys_name), entries in buckets.items():
        if not entries:
            continue
        obs = np.asarray([r for r, _ in entries], dtype=int)
        indicator_prob_mat = np.stack([p for _, p in entries], axis=0)
        N = obs.shape[0]
        pk_z0, pk_z1 = _expert_predictive_components(
            a_draws,
            b_draws,
            kappa_draws,
            kappa_by_expert_draws,
            e,
            K,
        )

        pred_hist = np.zeros((S, K))
        pred_mean_rating = np.zeros(S)
        pred_extreme_prop = np.zeros(S)

        for s in range(S):
            mix = (
                indicator_prob_mat[:, s, None] * pk_z1[s, None, :]
                + (1.0 - indicator_prob_mat[:, s, None]) * pk_z0[s, None, :]
            )
            mix = mix / mix.sum(axis=1, keepdims=True)
            cum = np.cumsum(mix, axis=1)
            u = rng.random(N)[:, None]
            sampled = (u < cum).argmax(axis=1)
            pred_hist[s] = np.bincount(sampled, minlength=K) / N
            pred_mean_rating[s] = float(sampled.mean())
            pred_extreme_prop[s] = float(
                np.mean((sampled == 0) | (sampled == K - 1))
            )

        obs_hist = np.bincount(obs, minlength=K) / N
        results[(processor.expert_names[e], sys_name)] = {
            "n_obs": int(N),
            "obs_hist": obs_hist,
            "pred_hist_mean": pred_hist.mean(axis=0),
            "pred_hist_lo": np.percentile(pred_hist, 3, axis=0),
            "pred_hist_hi": np.percentile(pred_hist, 97, axis=0),
            "obs_mean": float(obs.mean()),
            "pred_mean_mean": float(pred_mean_rating.mean()),
            "pred_mean_lo": float(np.percentile(pred_mean_rating, 3)),
            "pred_mean_hi": float(np.percentile(pred_mean_rating, 97)),
            "obs_extreme": float(np.mean((obs == 0) | (obs == K - 1))),
            "pred_extreme_mean": float(pred_extreme_prop.mean()),
            "pred_extreme_lo": float(np.percentile(pred_extreme_prop, 3)),
            "pred_extreme_hi": float(np.percentile(pred_extreme_prop, 97)),
        }

    return results


def per_expert_system_tree_implied_ppc_multisystem(
    idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
    n_draws: int = 500,
    seed: int = 42,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Stratified PPC using tree-implied q_j instead of data-conditioned pz1."""
    return per_expert_system_ppc_multisystem(
        idata,
        builder,
        processor,
        n_draws=n_draws,
        seed=seed,
        indicator_prob_source="tree_implied",
    )


def per_expert_system_q_ppc_multisystem(
    idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
    n_draws: int = 500,
    seed: int = 42,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Backward-compatible alias for the tree-implied stratified PPC."""
    return per_expert_system_tree_implied_ppc_multisystem(
        idata, builder, processor, n_draws=n_draws, seed=seed
    )


def plot_per_expert_system_histograms(
    results: Dict[Tuple[str, str], Dict[str, Any]],
    K: int = 7,
    expert_display: Optional[Dict[str, str]] = None,
    system_display: Optional[Dict[str, str]] = None,
    system_order: Optional[List[str]] = None,
    expert_order: Optional[List[str]] = None,
    title: Optional[str] = None,
):
    """Grid of (expert x system) mini-histograms: observed vs predicted.

    A cell is blank if that expert did not rate that system. One row per
    expert, one column per system.
    """
    import matplotlib.pyplot as plt

    experts = expert_order or sorted({k[0] for k in results})
    systems = system_order or sorted({k[1] for k in results})
    n_e = len(experts)
    n_s = len(systems)

    fig, axes = plt.subplots(
        n_e,
        n_s,
        figsize=(1.9 * n_s + 1.4, 1.55 * n_e + 0.8),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    cats = np.arange(K)
    width = 0.38

    for i, exp in enumerate(experts):
        for j, sys in enumerate(systems):
            ax = axes[i, j]
            r = results.get((exp, sys))
            if r is None:
                ax.set_axis_off()
                continue
            ax.bar(
                cats - width / 2,
                r["obs_hist"],
                width,
                color="steelblue",
                alpha=0.85,
                label="obs" if (i == 0 and j == 0) else None,
            )
            ax.bar(
                cats + width / 2,
                r["pred_hist_mean"],
                width,
                color="salmon",
                alpha=0.85,
                label="pred" if (i == 0 and j == 0) else None,
            )
            ax.errorbar(
                cats + width / 2,
                r["pred_hist_mean"],
                yerr=[
                    np.clip(r["pred_hist_mean"] - r["pred_hist_lo"], 0, None),
                    np.clip(r["pred_hist_hi"] - r["pred_hist_mean"], 0, None),
                ],
                fmt="none",
                color="black",
                capsize=1.3,
                linewidth=0.6,
            )
            ax.set_xticks(cats)
            ax.set_xticklabels([str(k + 1) for k in range(K)], fontsize=7)
            ax.tick_params(axis="y", labelsize=7)
            if i == 0:
                sys_lbl = (system_display or {}).get(sys, sys)
                ax.set_title(sys_lbl, fontsize=8)
            if j == 0:
                exp_lbl = (expert_display or {}).get(exp, exp)
                ax.set_ylabel(
                    f"{exp_lbl}\n(n={r['n_obs']})", fontsize=8
                )
            else:
                # annotate n per-cell in the corner
                ax.text(
                    0.98,
                    0.92,
                    f"n={r['n_obs']}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6,
                    color="grey",
                )

    # Global legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper right", fontsize=8, framealpha=0.9
        )
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96) if title else None)
    return fig, axes


def inspect_pz1_for_cell(
    idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
    expert_name: str,
    system_name: str,
) -> List[Dict[str, Any]]:
    """Per-indicator pz1 posterior summary for one (expert, system) cell.

    Returns one dict per indicator rated by ``expert_name`` in ``system_name``,
    with keys ``indicator``, ``n_obs_by_expert``, ``median``, ``p03``, ``p97``.

    Use together with ``classify_pz1_regime`` to decide whether a cell is in
    a single-component regime (pz1 near 0 or 1) or genuinely mixed (pz1
    moderate). This prerequisite check is needed before considering a
    per-expert noise scale sigma_e: sigma_e is most plausibly useful in the
    single-component regime, not in the moderate regime.
    """
    post = idata.posterior
    if expert_name not in processor.expert_to_idx:
        raise ValueError(f"expert {expert_name!r} not in processor")
    expert_idx = processor.expert_to_idx[expert_name]
    sp = builder._sys_prefix(system_name)
    sys_obs = processor.system_observations.get(system_name, {})
    rows: List[Dict[str, Any]] = []
    for nkey, obs_list in sys_obs.items():
        expert_count = sum(1 for e, _ in obs_list if e == expert_idx)
        if expert_count == 0:
            continue
        varname = builder.node_to_varname.get(nkey)
        if varname is None:
            continue
        pz1_name = f"{sp}__{varname}_pz1"
        if pz1_name not in post.data_vars:
            continue
        pz1 = np.asarray(post[pz1_name].values).reshape(-1)
        rows.append(
            {
                "indicator": nkey.split(" > ")[-1],
                "n_obs_by_expert": int(expert_count),
                "median": float(np.median(pz1)),
                "p03": float(np.percentile(pz1, 3)),
                "p97": float(np.percentile(pz1, 97)),
            }
        )
    return rows


def classify_pz1_regime(
    rows: List[Dict[str, Any]],
    low: float = 0.1,
    high: float = 0.9,
) -> Dict[str, Any]:
    """Classify a set of per-indicator pz1 summaries into regime buckets.

    ``near_0``: median < low
    ``near_1``: median > high
    ``moderate``: everything in between
    """
    n = len(rows)
    near_0 = sum(1 for r in rows if r["median"] < low)
    near_1 = sum(1 for r in rows if r["median"] > high)
    moderate = n - near_0 - near_1
    return {
        "n_indicators": n,
        "near_0": near_0,
        "near_1": near_1,
        "moderate": moderate,
        "frac_single_component": (near_0 + near_1) / n if n else 0.0,
        "frac_moderate": moderate / n if n else 0.0,
    }


def summarise_per_expert_system_ppc(
    results: Dict[Tuple[str, str], Dict[str, Any]],
) -> str:
    """Compact text summary for stratified (expert, system) PPC stats."""
    lines = []
    header = (
        f"{'Expert':<18s} {'System':<26s} {'n':>4s}  "
        f"{'obs_m':>5s}  {'pr_m':>13s}  "
        f"{'obs_ex':>6s}  {'pr_ex':>13s}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for (exp, sys), r in sorted(results.items()):
        pm = (
            f"{r['pred_mean_mean']:.2f}"
            f"[{r['pred_mean_lo']:.2f},{r['pred_mean_hi']:.2f}]"
        )
        pe = (
            f"{r['pred_extreme_mean']:.2f}"
            f"[{r['pred_extreme_lo']:.2f},{r['pred_extreme_hi']:.2f}]"
        )
        lines.append(
            f"{exp:<18s} {sys:<26s} {r['n_obs']:>4d}  "
            f"{r['obs_mean']:>5.2f}  {pm:>13s}  "
            f"{r['obs_extreme']:>6.2f}  {pe:>13s}"
        )
    return "\n".join(lines)
