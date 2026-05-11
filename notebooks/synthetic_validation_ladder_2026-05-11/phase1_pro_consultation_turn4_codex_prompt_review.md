# Phase 1 Pro consultation, turn 4: amendments to your Codex prompt before sending

I reviewed your Codex prompt against the actual repo. The structure and scope are right; the three critical things below need amending before the prompt is sent to Codex, because each is a place where Codex will either implement something inconsistent with the existing infrastructure (introducing fragility) or hit a wall it can't solve from the prompt alone. There are also eight smaller cleanups worth tucking in.

Please amend your Codex prompt with the changes below and we'll send the revised version.

## Critical amendment 1: targeted-override implementation must use logit-Normal, not Beta(18, 2), and needs a new `ModelConfig` flag

### What's wrong with the current spec

Your prompt says:

> Matching fitter priors: `beta_pres ~ Beta(18, 2)`, `beta_abs ~ Beta(2, 18)`

But the existing override infrastructure in `ModelConfig` and `MultiSystemExactTreeBuilder` uses **logit-Normal**, not Beta:

- `BETA_PRES_OVERRIDE_MEAN = 0.90` (mean in probability space)
- `BETA_OVERRIDE_SIGMA = 0.30` (default scale in logit space)
- Plumbing: `η ~ Normal(logit(μ), σ²)`, then `β = sigmoid(η)`

Plus the existing override applies **uniformly across all edges** — there is no per-edge selection mechanism. So Codex needs new infrastructure for per-edge selective override.

The natural pattern to follow already exists in the repo: `dcm_model.py:build_safe_gain_node_beta` constructs a per-node logit-Normal Beta variable. The existing `GAIN_LOGIT_NORMAL` branch in `_collect_tree_betas` (line ~165 of `dcm_model_exact_tree.py`) shows exactly how per-node logit-Normal priors plumb through.

### Proposed replacement text for "Chosen SPAR baseline > targeted_strong_lower_override"

Replace the "Matching fitter priors" lines and the "Implementation details > Targeted override utility" section with:

```text
Fitter prior on overridden edges: per-edge logit-Normal centred at the override mean.

    For overridden edges (lower edges under Coherence, Selective Attention,
    Complexity, Integration):

        eta_pres_e ~ Normal(logit(0.90), sigma_override^2)
        beta_pres_e = sigmoid(eta_pres_e)

        eta_abs_e ~ Normal(logit(0.10), sigma_override^2)
        beta_abs_e = sigmoid(eta_abs_e)

        sigma_override = 0.30 by default, matching existing
        BETA_OVERRIDE_SIGMA convention.

    For non-overridden edges:

        Use the existing per-node Beta priors derived from the
        EvidenceProcessor support/demandingness mapping
        (i.e., the production prior path).

    Do NOT use Beta(18, 2) / Beta(2, 18). The repo's existing override
    convention is logit-Normal, and using Beta would create a divergent
    second prior family that breaks consistency with
    BETA_PRES_OVERRIDE_MEAN / BETA_OVERRIDE_SIGMA infrastructure.

Implementation routing:

    1. Add a new ModelConfig field:

           TARGETED_OVERRIDE_NODE_KEYS: Optional[Sequence[str]] = None

       Each entry is a node_key (as produced by node_key(path, node["name"]))
       for an edge (parent v -> child u) whose β should be overridden.

    2. Modify _collect_tree_betas in dcm_model_exact_tree.py:136 with a
       new branch that runs BEFORE the existing GAIN_LOGIT_NORMAL and
       per-node Beta branches:

           if (config.TARGETED_OVERRIDE_NODE_KEYS is not None
               and key in set(config.TARGETED_OVERRIDE_NODE_KEYS)):
               # Per-edge logit-Normal at the override mean.
               mu_p = config.BETA_PRES_OVERRIDE_MEAN  # default 0.90
               mu_a = config.BETA_ABS_OVERRIDE_MEAN   # default 0.10
               sigma = (config.BETA_OVERRIDE_SIGMA
                        if config.BETA_OVERRIDE_SIGMA is not None
                        else config.LABEL_POOL_SIGMA)
               bp = build_safe_gain_node_beta(name, "pres", mu_p, sigma)
               ba = build_safe_gain_node_beta(name, "abs", mu_a, sigma)
           elif config.POOL_BETAS_BY_LABEL:
               # ... existing code ...

    3. The targeted override utility populates TARGETED_OVERRIDE_NODE_KEYS
       at config-build time, by walking the tree once and selecting edges
       whose top-level ancestor is one of the strong top features (and
       whose parent is not the root). Save the resulting list to
       config.json for verification.

Sanity-check assertions to add inside the targeted override utility:

    - len(TARGETED_OVERRIDE_NODE_KEYS) > 0
    - no key in TARGETED_OVERRIDE_NODE_KEYS corresponds to a root->top edge
    - every key in TARGETED_OVERRIDE_NODE_KEYS has a strong top-feature
      ancestor (Coherence / Selective Attention / Complexity / Integration)
    - no key has a weak top-feature ancestor (Representationality /
      Hierarchical Organization / Modularity)
```

This makes the implementation route concrete and consistent with the existing codebase. The `build_safe_gain_node_beta` helper already does the logit-Normal-with-sigmoid pattern; Codex just dispatches per edge.

## Critical amendment 2: add `rho_collapsed` as a first-class deterministic

### What's wrong

Your prompt assumes `{prefix}_rho` gives the collapsed-prior form. The Phase 0 deterministic added to `MultiSystemExactTreeBuilder._exact_tree_log_likelihood` is actually:

```python
pm.Deterministic(
    f"{sp}__{stance_name}_rho",
    pt.sigmoid(pt.logit(c_var) + (log_L_top1 - log_L_top0)),
)
```

This is the **sampled-π form** (uses `c_var` = π_s draw), not collapsed. Phase 0 audit showed sampled-π and collapsed agree to mean abs diff 0.001 in practice, but for clean validation reporting Codex should expose both.

### Proposed addition under "Implementation details > Root evidence extraction"

Add this paragraph:

```text
The existing {prefix}_rho deterministic in MultiSystemExactTreeBuilder uses
the sampled-pi form: sigmoid(logit(c_var) + log_B). For Path A reporting,
add a second deterministic that uses the collapsed-prior form:

    import math
    LOG_PRIOR_ODDS = math.log(
        config.DEFAULT_ALPHA / config.DEFAULT_BETA
    )  # log(1/5) = -1.609 for the default Beta(1, 5)

    pm.Deterministic(
        f"{sp}__{stance_name}_rho_collapsed",
        pt.sigmoid(LOG_PRIOR_ODDS + (log_L_top1 - log_L_top0)),
    )

Use rho_collapsed as the headline reporting quantity throughout the
validation. The existing {prefix}_rho (sampled-pi form) is retained for
backward compatibility and reported as a secondary diagnostic
(comparison with rho_collapsed gives a cheap check that the sampled-pi
posterior closely tracks the integrated odds form, as Phase 0 found).

If for some reason rho_collapsed cannot be added as a deterministic at
model-build time (e.g., backward-compatibility concerns with downstream
consumers of the existing schema), compute it post-hoc from log_B
posterior draws as:

    rho_collapsed_d = sigmoid(LOG_PRIOR_ODDS + log_B_d)
    rho_collapsed_mean = mean over d of rho_collapsed_d
```

## Critical amendment 3: LOO implementation cost guidance

### What's wrong

Your prompt correctly defines the integrated leave-one-rating-out form:

```
log_lik_i_d = log p(y_s | theta_d) - log p(y_{s, -i} | theta_d)
```

But the implementation involves recomputing leaf log-likelihoods for every (rating-dropped, posterior-draw) pair. With 379 ratings × 500 draws = ~190K DP evaluations per fit × 50 fits = ~10M DP evaluations. This is doable but Codex needs guidance on how to make it efficient and what to cut if it's too slow.

### Proposed replacement / addition under "PSIS-LOO / lppd > Implementation route"

Replace the four-step implementation route with:

```text
Implementation route:

    1. For each posterior draw theta_d, precompute the per-(system,
       indicator) three-state (or binary) leaf log-likelihoods using the
       existing precompute_leaf_logliks pattern from
       composite_vs_exact_diagnostic.py:139. Store as a dict keyed by
       (system, indicator_node_key) -> (ll_0, ll_half, ll_1) tuples.

    2. For each draw and system, run the full exact_loglik DP once to
       get full_ll_s_d.

    3. For each rating i = (system_i, indicator_i, expert_idx_i,
       rating_value_i), do a leave-one-out DP evaluation:

         a. Take the cached per-indicator leaf log-likelihoods for system_i.

         b. For indicator_i specifically, recompute (ll_0, ll_half, ll_1)
            with the i-th (expert_idx, rating) entry removed. This is a
            small modification: subtract the per-rating contribution from
            the indicator's aggregated log-likelihood, which can be done
            in O(1) if per-rating contributions are stored separately.

         c. Substitute the modified indicator's leaf log-likelihoods into
            the cached dict.

         d. Re-run exact_loglik for system_i only (other systems unchanged).

         e. log_lik_i_d = full_ll_{system_i}_d - minus_i_ll_{system_i}_d.

    4. Build the LOO log-likelihood matrix of shape (n_draws, n_ratings)
       and pass to arviz.loo via az.from_dict + az.loo or by constructing
       an InferenceData object with a "log_likelihood" group.

Performance and caching:

    - Caching unchanged-indicator leaf log-likelihoods is essential.
      Without caching, each LOO evaluation re-walks every leaf in the
      system — orders of magnitude slower.
    - Storing per-rating contributions separately during the initial
      precompute_leaf_logliks step makes step (3b) O(1) instead of
      O(n_ratings_per_indicator) per leave-one-out.
    - Vectorise across draws where possible: the DP is purely numerical
      so it can be evaluated for all draws of (beta, a, kappa, b)
      simultaneously by broadcasting.
    - Estimate cost: with 379 ratings, 500 draws, vectorised over draws,
      and per-system DPs that vectorise cheaply over draws, the LOO
      post-processing should run in roughly tens of minutes per fit
      (vs hours if naively implemented per-(rating, draw)).

Defaults for run modes:

    smoke mode:    loo_draws = 100, loo_max_ratings = 50 (sub-sample)
    pilot mode:    loo_draws = 500, loo_max_ratings = all
    full mode:     loo_draws = 500, loo_max_ratings = all (target);
                   if wall-clock pressure mounts, fall back to
                   loo_draws = 300 with all ratings, OR
                   loo_draws = 500 with a stratified sub-sample
                   (e.g. 100 ratings per system, balanced across
                   indicator and expert).

Skip rules:

    - --skip-loo flag bypasses all LOO computation and explicitly marks
      the LOO pillar as "not run" (NOT "passed") in the report.
    - --loo-max-ratings N caps the per-system LOO evaluation to N
      ratings (stratified sample) for time-bounded runs.
    - If LOO is partial in full validation, the report must clearly
      label it as such and note that elpd_loo is computed on the
      subsample, not the full data.
```

## Smaller cleanups (insert as appropriate)

These don't need long elaboration — quick edits within Pro's existing structure.

1. **`mean_cell_TV <= 0.25` is too strict for sparse cells.** Add: "Aggregate TV at the (system, indicator) level across all raters before averaging — per-cell TV with 2-3 raters per cell is sampling-noise-dominated. If finer granularity is wanted, weight TV contributions by `n_obs_in_cell` to deflate the contribution of sparse cells, or filter to cells with `n_obs >= 5`."

2. **PPC p-value formulation.** Replace the bespoke "TV against predictive mean" formulation with: "Use a standard Bayesian posterior predictive p-value on the mean cell-TV statistic. Either Pro's formulation or the standard arviz form is acceptable; the exact form is less important than reporting where it falls. arviz.plot_ppc and arviz.bayes_p_value are reasonable references."

3. **Toy tree HMC L1 — strengthen the skip recommendation.** Replace "If the production builder cannot easily consume the toy tree, document and skip HMC for L1" with: "Skip L1 HMC by default unless L0 (no-fit toy) shows unexpected behaviour that requires HMC to diagnose. The toy-tree → MultiSystemExactTreeBuilder integration requires constructing a synthetic stance_data dict in the DCM API JSON format plus a MultiSystemData object — fragile infrastructure that's not worth building for a sanity-check rung."

4. **Build vs reuse — be more explicit about extending `gwt_full_exact_recovery.py`.** Replace "Build: a main validation harness" with: "Build by extending `notebooks/synthetic_validation_2026-05-06/gwt_full_exact_recovery.py`, which already implements the generate → fit → summarise per-seed loop. The new validation script adds the targeted-override variant, the new metric pillars (PPC, LOO), and the recovery ladder. Avoid rewriting from scratch."

5. **Negative-control seeds.** Add: "When running the negative control (`production_unmodified`), use the **same seeds** as the primary variant (the first N seeds where N = `--negative-control-seeds`). This is required for `delta_elpd_loo` to be meaningful — paired comparison on identical synthetic datasets."

6. **PPC sampling cost.** Add to the PPC section: "Posterior predictive sampling on the marginalised tree DP requires re-evaluating the per-rating predictive distribution per draw. The existing `exact_tree_ppc.py` likely provides the right primitives — Codex should read that file's API before implementing fresh PPC sampling. Reuse rather than rebuild."

7. **Sub-seeds per replicate.** Add to "Synthetic data specification": "Use distinct sub-seeds per replicate to avoid spurious correlation across runs. Pattern: `rng_per_seed_idx = np.random.default_rng(BASE_SEED + seed_idx * 1_000_003)`. Document the sub-seed scheme in `truth_payload.json`."

8. **Memory and disk budget.** Add to "Compute budget": "Disk: 50 fits × ~50MB per `fit.nc` = ~2.5GB; per-rating LOO log-lik tensors (n_ratings × n_draws) per fit add 1-10MB. Use NetCDF compression (`compression='zlib'` in xarray.to_netcdf) for `fit.nc` and chunked Zarr for the LOO log-lik tensors if sizes balloon."

9. **Pre-validation smoke fit.** Add to "Quality gates" as item 7: "Before launching pilot/full mode, run **one end-to-end smoke HMC fit** and verify: (a) the targeted override actually fires (overridden edges have logit-Normal priors, others have production Beta priors); (b) the generator/fitter beta-mean match assertion passes; (c) the new `rho_collapsed` deterministic is exposed in the trace; (d) one fit completes with no errors; (e) the LOO computation produces a valid log-lik matrix on the smoke fit. Only after this passes should pilot/full mode be launched."

## Closing

With these amendments, the prompt should be ~95% ready. Codex will still hit unexpected issues during implementation (it always does), but the three critical gaps and eight smaller cleanups remove the foreseeable ones.

If you'd prefer to consolidate the targeted-override implementation guidance into a separate `dcm_model_exact_tree.py` patch spec rather than embedding it in the Codex prompt, that's also a defensible structure — but the patch is small enough to keep inline.

Send me the amended Codex prompt when ready and I'll do one more pass before it goes to the coding agent.
