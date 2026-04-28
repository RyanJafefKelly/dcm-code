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

from dcm_model_ordinal import MultiSystemDataProcessor, MultiSystemModelBuilder


def _expert_label(processor: MultiSystemDataProcessor, expert_idx: int) -> str:
    if hasattr(processor, "expert_label"):
        return processor.expert_label(expert_idx)
    return f"Expert {expert_idx + 1}"


def _resolve_expert_idx(processor: MultiSystemDataProcessor, expert_name: str) -> int:
    if expert_name in processor.expert_to_idx:
        return processor.expert_to_idx[expert_name]
    display = getattr(processor, "expert_display_names", {})
    for raw_name, label in display.items():
        if expert_name == label:
            return processor.expert_to_idx[raw_name]
    raise ValueError(f"expert {expert_name!r} not in processor")


def _ordered_probit_probs(
    kappa: np.ndarray, eta: np.ndarray, K: int, sigma: Optional[np.ndarray] = None
) -> np.ndarray:
    """Ordered-probit category probabilities.

    kappa : (S, K-1) cutpoints.
    eta   : (S,) linear predictors.
    Returns (S, K).
    """
    # (S, K-1) cumulative Phi(kappa_k - eta)
    scaled = kappa - eta[:, None]
    if sigma is not None:
        scaled = scaled / sigma[:, None]
    cum = norm.cdf(scaled)
    zeros = np.zeros((cum.shape[0], 1))
    ones = np.ones((cum.shape[0], 1))
    cum_full = np.concatenate([zeros, cum, ones], axis=1)  # (S, K+1)
    probs = np.diff(cum_full, axis=1)  # (S, K)
    return np.clip(probs, 1e-12, 1.0)


def _resolve_indicator_prob_suffix(indicator_prob_source: str) -> str:
    """Legacy helper: resolve source name to a binary-only variable suffix.

    Kept for backwards compatibility with external callers. New code should
    prefer ``_extract_component_weights_for_indicator``, which understands
    both binary and three-state state models.
    """
    source = indicator_prob_source.lower()
    if source in {"pz1", "leaf_updated", "posterior_z"}:
        return "pz1"
    if source in {"q", "q_j", "p", "tree", "tree_implied", "tree-implied"}:
        return "p"
    raise ValueError(
        "indicator_prob_source must be one of "
        "{'pz1', 'leaf_updated', 'tree_implied', 'q'}"
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
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
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

    if "sigma_by_expert" in post.data_vars:
        sigma_all = np.asarray(post["sigma_by_expert"].values).reshape(
            -1, n_experts
        )
        sigma_by_expert_draws = sigma_all[idx]
    else:
        sigma_by_expert_draws = None

    return (
        a_draws,
        b_draws,
        kappa_draws,
        kappa_by_expert_draws,
        sigma_by_expert_draws,
    )


def _expert_predictive_components(
    a_draws: np.ndarray,
    b_draws: np.ndarray,
    kappa_draws: np.ndarray,
    kappa_by_expert_draws: Optional[np.ndarray],
    sigma_by_expert_draws: Optional[np.ndarray],
    expert_idx: int,
    K: int,
) -> Tuple[np.ndarray, np.ndarray]:
    eta_z0 = b_draws[:, expert_idx]
    eta_z1 = eta_z0 + a_draws
    if kappa_by_expert_draws is None:
        kappa_e = kappa_draws
    else:
        kappa_e = kappa_by_expert_draws[:, expert_idx, :]
    sigma_e = (
        None
        if sigma_by_expert_draws is None
        else sigma_by_expert_draws[:, expert_idx]
    )
    pk_z0 = _ordered_probit_probs(kappa_e, eta_z0, K, sigma_e)
    pk_z1 = _ordered_probit_probs(kappa_e, eta_z1, K, sigma_e)
    return pk_z0, pk_z1


# ---------------------------------------------------------------------------
# State-model-aware helpers (support both "binary" and "three_state")
# ---------------------------------------------------------------------------


def _config_state_model(builder: Any) -> str:
    """Read INDICATOR_STATE_MODEL off the builder config with a safe default."""
    return getattr(builder.config, "INDICATOR_STATE_MODEL", "binary")


def _is_leaf_updated(indicator_prob_source: str) -> bool:
    source = indicator_prob_source.lower()
    if source in {"pz1", "leaf_updated", "posterior_z"}:
        return True
    if source in {"q", "q_j", "p", "tree", "tree_implied", "tree-implied"}:
        return False
    raise ValueError(
        "indicator_prob_source must be one of {'pz1', 'leaf_updated', "
        "'tree_implied', 'q'}; got "
        f"{indicator_prob_source!r}"
    )


def _extract_component_weights_for_indicator(
    post: Any,
    idx: np.ndarray,
    sys_prefix: str,
    varname: str,
    state_model: str,
    indicator_prob_source: str,
) -> Optional[np.ndarray]:
    """Per-draw latent-component weight matrix for one indicator.

    Returns shape (n_components, S) array with draws at the sampled indices.
    Returns ``None`` if the required deterministics are missing from the
    posterior (caller skips the indicator).

    Mapping
    -------
    binary + leaf_updated   : (1 - pz1, pz1)
    binary + tree_implied   : (1 - q, q)
    three_state + leaf_updated : (p_m0, p_m1, p_m2)
    three_state + tree_implied : ((1-q)^2, 2q(1-q), q^2)
    """
    leaf_updated = _is_leaf_updated(indicator_prob_source)

    if state_model == "binary":
        suffix = "pz1" if leaf_updated else "p"
        vname = f"{sys_prefix}__{varname}_{suffix}"
        if vname not in post.data_vars:
            return None
        p = np.asarray(post[vname].values).reshape(-1)[idx]
        return np.stack([1.0 - p, p], axis=0)

    if state_model == "three_state":
        if leaf_updated:
            names = [f"{sys_prefix}__{varname}_p_m{i}" for i in range(3)]
            if not all(n in post.data_vars for n in names):
                return None
            ws = [np.asarray(post[n].values).reshape(-1)[idx] for n in names]
            return np.stack(ws, axis=0)
        vname = f"{sys_prefix}__{varname}_p"
        if vname not in post.data_vars:
            return None
        q = np.asarray(post[vname].values).reshape(-1)[idx]
        return np.stack(
            [(1.0 - q) ** 2, 2.0 * q * (1.0 - q), q ** 2], axis=0
        )

    raise ValueError(f"Unknown state_model: {state_model!r}")


def _expert_component_predictives(
    a_draws: np.ndarray,
    b_draws: np.ndarray,
    kappa_draws: np.ndarray,
    kappa_by_expert_draws: Optional[np.ndarray],
    sigma_by_expert_draws: Optional[np.ndarray],
    expert_idx: int,
    K: int,
    state_model: str,
) -> np.ndarray:
    """Component emission category probabilities for one expert.

    Shape (n_components, S, K).  binary: eta in {0, a}; three_state:
    eta in {0, a/2, a}.
    """
    eta_base = b_draws[:, expert_idx]
    if kappa_by_expert_draws is None:
        kappa_e = kappa_draws
    else:
        kappa_e = kappa_by_expert_draws[:, expert_idx, :]
    sigma_e = (
        None
        if sigma_by_expert_draws is None
        else sigma_by_expert_draws[:, expert_idx]
    )
    if state_model == "binary":
        eta_values = [eta_base, eta_base + a_draws]
    elif state_model == "three_state":
        eta_values = [eta_base, eta_base + 0.5 * a_draws, eta_base + a_draws]
    else:
        raise ValueError(f"Unknown state_model: {state_model!r}")
    return np.stack(
        [_ordered_probit_probs(kappa_e, eta_c, K, sigma_e) for eta_c in eta_values],
        axis=0,
    )


def _draw_cell_predictive_stats(
    weights_mat: np.ndarray,
    pk_components: np.ndarray,
    obs: np.ndarray,
    rng: np.random.Generator,
    K: int,
) -> Dict[str, Any]:
    """Sample from per-draw mixture; aggregate shape + signed-tail diagnostics.

    Parameters
    ----------
    weights_mat : (C, N_obs, S) array of latent-component weights.
    pk_components : (C, S, K) array of component emission probabilities.
    obs : (N_obs,) observed ratings (0-indexed categories).
    """
    n_components, N_obs, S = weights_mat.shape
    mid_lo = K // 2 - 1
    mid_hi = K // 2 + 1  # middle band = {K//2-1, K//2, K//2+1}; for K=7 -> {2,3,4}

    pred_hist = np.zeros((S, K))
    pred_mean = np.zeros(S)
    pred_var = np.zeros(S)
    pred_left = np.zeros(S)
    pred_right = np.zeros(S)
    pred_extreme = np.zeros(S)
    pred_mid = np.zeros(S)

    for s in range(S):
        mix = np.zeros((N_obs, K))
        for c in range(n_components):
            mix += weights_mat[c, :, s, None] * pk_components[c, s, None, :]
        row_sums = mix.sum(axis=1, keepdims=True)
        mix = mix / np.clip(row_sums, 1e-12, None)
        cum = np.cumsum(mix, axis=1)
        u = rng.random(N_obs)[:, None]
        sampled = (u < cum).argmax(axis=1)
        pred_hist[s] = np.bincount(sampled, minlength=K) / N_obs
        pred_mean[s] = float(sampled.mean())
        # Betancourt-style per-cell rating variance; N_obs denominator (not
        # N_obs - 1) so the statistic is well-defined when N_obs = 1.
        pred_var[s] = float(sampled.var())
        pred_left[s] = float(np.mean(sampled == 0))
        pred_right[s] = float(np.mean(sampled == K - 1))
        pred_extreme[s] = pred_left[s] + pred_right[s]
        pred_mid[s] = float(np.mean((sampled >= mid_lo) & (sampled <= mid_hi)))

    obs_hist = np.bincount(obs, minlength=K) / N_obs
    obs_left = float(np.mean(obs == 0))
    obs_right = float(np.mean(obs == K - 1))
    obs_extreme = obs_left + obs_right
    obs_mid = float(np.mean((obs >= mid_lo) & (obs <= mid_hi)))
    obs_var = float(obs.var())

    def _p3_p97(arr: np.ndarray) -> Tuple[float, float]:
        return float(np.percentile(arr, 3)), float(np.percentile(arr, 97))

    pm_lo, pm_hi = _p3_p97(pred_mean)
    pl_lo, pl_hi = _p3_p97(pred_left)
    pr_lo, pr_hi = _p3_p97(pred_right)
    pe_lo, pe_hi = _p3_p97(pred_extreme)
    pmid_lo, pmid_hi = _p3_p97(pred_mid)
    pvar_lo, pvar_hi = _p3_p97(pred_var)
    pred_left_m = float(pred_left.mean())
    pred_right_m = float(pred_right.mean())
    pred_mid_m = float(pred_mid.mean())
    pred_var_m = float(pred_var.mean())

    return {
        "n_obs": int(N_obs),
        "obs_hist": obs_hist,
        "pred_hist_mean": pred_hist.mean(axis=0),
        "pred_hist_lo": np.percentile(pred_hist, 3, axis=0),
        "pred_hist_hi": np.percentile(pred_hist, 97, axis=0),
        "obs_mean": float(obs.mean()),
        "pred_mean_mean": float(pred_mean.mean()),
        "pred_mean_lo": pm_lo,
        "pred_mean_hi": pm_hi,
        "obs_extreme": obs_extreme,
        "pred_extreme_mean": float(pred_extreme.mean()),
        "pred_extreme_lo": pe_lo,
        "pred_extreme_hi": pe_hi,
        # Signed tail and middle-mass diagnostics (new)
        "obs_left": obs_left,
        "pred_left_mean": pred_left_m,
        "pred_left_lo": pl_lo,
        "pred_left_hi": pl_hi,
        "delta_left_mean": pred_left_m - obs_left,
        "obs_right": obs_right,
        "pred_right_mean": pred_right_m,
        "pred_right_lo": pr_lo,
        "pred_right_hi": pr_hi,
        "delta_right_mean": pred_right_m - obs_right,
        "obs_mid": obs_mid,
        "pred_mid_mean": pred_mid_m,
        "pred_mid_lo": pmid_lo,
        "pred_mid_hi": pmid_hi,
        "delta_mid_mean": pred_mid_m - obs_mid,
        # Betancourt-style per-cell rating variance (Bae's Theorem §2)
        "obs_var": obs_var,
        "pred_var_mean": pred_var_m,
        "pred_var_lo": pvar_lo,
        "pred_var_hi": pvar_hi,
        "delta_var_mean": pred_var_m - obs_var,
    }


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
    state_model = _config_state_model(builder)
    post = idata.posterior
    n_experts = len(processor.expert_names)
    S_total = np.asarray(post["a"].values).reshape(-1).shape[0]
    idx = _sample_draw_indices(S_total, n_draws, seed)
    rng = np.random.default_rng(seed)
    (
        a_draws,
        b_draws,
        kappa_draws,
        kappa_by_expert_draws,
        sigma_by_expert_draws,
    ) = _extract_obs_layer_draws(post, idx, n_experts, K)

    # --- Collect (expert_idx, rating, component_weights) across systems ---
    # component_weights has shape (C, S) with C=2 (binary) or C=3 (three_state)
    per_expert_obs: List[List[int]] = [[] for _ in range(n_experts)]
    per_expert_weights: List[List[np.ndarray]] = [[] for _ in range(n_experts)]

    for sys_name, sys_obs in processor.system_observations.items():
        sp = builder._sys_prefix(sys_name)
        for nkey, obs_list in sys_obs.items():
            varname = builder.node_to_varname.get(nkey)
            if varname is None:
                continue
            weights = _extract_component_weights_for_indicator(
                post, idx, sp, varname, state_model, indicator_prob_source
            )
            if weights is None:
                continue
            for expert_idx, rating in obs_list:
                per_expert_obs[expert_idx].append(int(rating))
                per_expert_weights[expert_idx].append(weights)

    results: Dict[str, Dict[str, Any]] = {}
    for e in range(n_experts):
        if not per_expert_obs[e]:
            continue
        obs = np.asarray(per_expert_obs[e], dtype=int)
        # weights_stack: (C, N_e, S)
        weights_stack = np.stack(per_expert_weights[e], axis=1)
        pk_components = _expert_component_predictives(
            a_draws,
            b_draws,
            kappa_draws,
            kappa_by_expert_draws,
            sigma_by_expert_draws,
            e,
            K,
            state_model,
        )
        results[_expert_label(processor, e)] = _draw_cell_predictive_stats(
            weights_stack, pk_components, obs, rng, K
        )

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
    state_model = _config_state_model(builder)
    post = idata.posterior
    n_experts = len(processor.expert_names)
    S_total = np.asarray(post["a"].values).reshape(-1).shape[0]
    idx = _sample_draw_indices(S_total, n_draws, seed)
    rng = np.random.default_rng(seed)
    (
        a_draws,
        b_draws,
        kappa_draws,
        kappa_by_expert_draws,
        sigma_by_expert_draws,
    ) = _extract_obs_layer_draws(post, idx, n_experts, K)

    # Bucket observations by (expert_idx, system_name) -> list of (rating, component_weights).
    # component_weights has shape (C, S), with C=2 (binary) or C=3 (three_state).
    buckets: Dict[Tuple[int, str], List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for sys_name, sys_obs in processor.system_observations.items():
        sp = builder._sys_prefix(sys_name)
        for nkey, obs_list in sys_obs.items():
            varname = builder.node_to_varname.get(nkey)
            if varname is None:
                continue
            weights = _extract_component_weights_for_indicator(
                post, idx, sp, varname, state_model, indicator_prob_source
            )
            if weights is None:
                continue
            for expert_idx, rating in obs_list:
                buckets[(expert_idx, sys_name)].append((int(rating), weights))

    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (e, sys_name), entries in buckets.items():
        if not entries:
            continue
        obs = np.asarray([r for r, _ in entries], dtype=int)
        # weights_stack: (C, N_e, S)
        weights_stack = np.stack([w for _, w in entries], axis=1)
        pk_components = _expert_component_predictives(
            a_draws,
            b_draws,
            kappa_draws,
            kappa_by_expert_draws,
            sigma_by_expert_draws,
            e,
            K,
            state_model,
        )
        results[(_expert_label(processor, e), sys_name)] = _draw_cell_predictive_stats(
            weights_stack, pk_components, obs, rng, K
        )

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


def inspect_indicator_state_for_cell(
    idata: Any,
    builder: MultiSystemModelBuilder,
    processor: MultiSystemDataProcessor,
    expert_name: str,
    system_name: str,
) -> List[Dict[str, Any]]:
    """Per-indicator posterior expected-presence summary for one (expert, system) cell.

    Under the binary state model, reads ``{sys}__{var}_pz1`` directly.
    Under the three-state model, reads ``{sys}__{var}_expected_z`` (the
    posterior expected indicator strength in [0, 1], which plays the same
    regime-classification role as ``pz1`` but is not itself ``pz1``).

    Returns one dict per indicator rated by ``expert_name`` in
    ``system_name`` with keys ``indicator``, ``n_obs_by_expert``,
    ``median``, ``p03``, ``p97``. Pair with ``classify_pz1_regime`` to
    bucket cells into single-component (near 0 / near 1) vs moderate
    regimes.
    """
    post = idata.posterior
    expert_idx = _resolve_expert_idx(processor, expert_name)
    sp = builder._sys_prefix(system_name)
    sys_obs = processor.system_observations.get(system_name, {})
    state_model = _config_state_model(builder)
    suffix = "pz1" if state_model == "binary" else "expected_z"
    rows: List[Dict[str, Any]] = []
    for nkey, obs_list in sys_obs.items():
        expert_count = sum(1 for e, _ in obs_list if e == expert_idx)
        if expert_count == 0:
            continue
        varname = builder.node_to_varname.get(nkey)
        if varname is None:
            continue
        var_name = f"{sp}__{varname}_{suffix}"
        if var_name not in post.data_vars:
            continue
        pz1 = np.asarray(post[var_name].values).reshape(-1)
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


# Backwards-compatible alias. The original name implied a binary-only
# quantity (pz1), but under three-state the reported value is expected_z.
# New callers should prefer ``inspect_indicator_state_for_cell``.
inspect_pz1_for_cell = inspect_indicator_state_for_cell


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
