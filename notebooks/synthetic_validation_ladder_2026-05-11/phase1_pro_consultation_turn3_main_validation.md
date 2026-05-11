# Phase 1 Pro consultation, turn 3: brief for writing the Codex prompt for the main synthetic validation task

The diagnostic ladder is done. Phase 1A/B/C have localised the failure mechanism. Now I want to actually do the original synthetic-validation task that motivated all of this — and I want you to write the Codex-ready prompt for it. I will review your Codex prompt before sending it.

## Where we are after Phase 1C

Headlines, in case you want to refer back:

- **Phase 0** settled the estimand: binary-root semantics; $\rho_s = p(R_s=1 \mid y_s)$ is the headline; production deterministics (`log_L_root0`, `log_L_root1`, `log_B`, `rho`) added to `MultiSystemExactTreeBuilder`.
- **Phase 1A** (oracle audit): existing synthetic fits show $R = 1$ free-target failures even with posterior-median nuisance.
- **Phase 1B** (no-fit ladder): failure occurs even with perfectly clamped nuisance under production $\beta$. The extreme-$\beta$ positive control passes decisively.
- **Phase 1C** (attrition audit): the bottleneck is **lower-edge transmission under specifically the strong top features**.

Specific Phase 1C numbers that anchor what comes next:

- Top-level analytical bound (production prior mean): $\mathbb{E}[\log B \mid R=1] = +2.81$, $\mathbb{E}[\log B \mid R=0] = -2.81$. (Top is fine.)
- Subfeature-observed under production prior: median $R=1$ = $+1.02$, $R=0$ = $-1.06$ (already below pass threshold).
- Leaf-perfect or noisy K=6 or K=1000: all converge to ~$+0.56 / -0.63$ — **the information ramp asymptotes; more data does not help under production priors**.
- Per-top-feature loss (median $c_{\text{top latent}} \to$ median $c_{\text{oracle subtree}}$): Coherence loses 1.45 nats; Selective Attention 0.94; Complexity 0.68; Integration ~0; Modularity / Representationality / Hierarchical Org ~0 (those have weak top contributions to begin with).
- Edge-profile ablation: `strong_top_features_only_extreme_lower` PASSES (M = +0.90); `weak_top_features_only_extreme_lower` FAILS. Strengthening must target lower edges under the **strong** top features.

The fix is structural — re-labelling lower edges, or asymmetric prior overrides applied selectively, or both. No amount of nuisance handling, ordinal extension, or rater-design improvement rescues the published model.

## The original validation task

This is the goal that motivated all the diagnostics. The user wants:

> A baseline DCM that, on synthetic data, demonstrates:
> 1. **Adequate $\rho_s$ recovery** for free systems — calibrated against realised $R_s$, decisive on $R_s = 1$ vs $R_s = 0$, beating the prior baseline on Brier and log-score.
> 2. **Adequate posterior-predictive fit** on rating distributions — RPS, TV per system × indicator cell, high-$q$ top-category mass error within bounds.
> 3. **Acceptable out-of-sample fit** — PSIS-LOO (elpd_loo), lppd, Pareto-$\hat{k}$ diagnostics passing.
> 4. **Clean sampler diagnostics** — $\hat R$, ESS, divergences.

This validated baseline is the SPAR final-result candidate and the model that gets handed off to Dawn (Anthropic data scientist) and Arvo for continuation.

## What I want you to do

**Write a Codex-ready markdown prompt** that specifies this comprehensive synthetic validation experiment end-to-end. The Codex prompt should be self-contained: the coding agent will implement, run, and report from it without further consultation. I will review your Codex prompt before sending it.

The prompt needs to make the following decisions concrete:

### A. Which model variant(s) to validate

The Phase 1C finding implies the unmodified published tree+priors cannot pass validation. Some "fix" must be applied. Three candidates, in increasing surgical effort:

1. **Uniform asymmetric $\beta$ override** — apply `BETA_PRES_OVERRIDE_MEAN = 0.90, BETA_ABS_OVERRIDE_MEAN = 0.10` to all edges via the existing `ModelConfig` flag. Easy, but blunt; over-rides label-aware semantics.
2. **Targeted re-labelling** of subfeatures under the strong top features (Coherence, Selective Attention, Complexity, Integration) to stronger support/demandingness labels. More surgical, defensible to the modeller, but requires modifying the GWT tree JSON.
3. **Hybrid**: keep the tree intact but apply asymmetric prior overrides selectively to lower edges under strong top features only. Requires a new `ModelConfig` flag (e.g. `STRONG_TOP_FEATURE_LOWER_OVERRIDE_NAMES`).

**Recommend one as the SPAR-shipping baseline** and justify briefly. Optionally include a tightly scoped comparison against the unmodified production model as a negative control. Don't sprawl into a 4-variant sweep — the point is to ship a defensible single baseline.

### B. Synthetic data design

- Use **balanced deterministic roots** for free targets (Chicken_R = i % 2; LLMs_R = 1 − i % 2). Anchored Human (R=1) and ELIZA (R=0).
- **Generator $\beta$ truth must match what the fitter expects** — this avoids the prior–DGP mismatch that contaminated the original asymmetric prior sweep. If the fix is "asymmetric override", the generator must use 0.90 / 0.10 truth; if "targeted re-labelling", the generator must use the relabelled-tree's prior means as truth. Be explicit.
- Specify number of seeds given compute budget (see F).

### C. Fit specification

- Full HMC via `MultiSystemExactTreeBuilder` (production class; deterministics already added).
- Production rater design (5 single-system + 1 cross-system rater = Rater_B, who covers Human/LLM/ELIZA but **not Chicken** — this asymmetry matters and is worth flagging in the prompt for Codex's attention).
- Production sampler defaults: 4 chains × 1000 draws × 1000 tune, target_accept=0.95. Deviate only if you have reason.

### D. Metric set + pass thresholds

- **$\rho_s$ recovery**: Brier improvement vs prior baseline (1/6), log-score improvement, ECE on aggregated free-target × seed predictions, balanced accuracy at $\rho > 0.5$, decisive-rate at $\rho > 0.95$ for $R = 1$ and $\rho < 0.05$ for $R = 0$, posterior calibration curve data.
- **Posterior predictive**: per-rating RPS, per-(system, indicator)-cell TV distance, high-$q$ top-category mass error (per Pro turn-3 of original consultation), posterior predictive $p$-value on TV statistic.
- **LOO/lppd**: elpd_loo via PSIS-LOO with **ordinal-Likert pointwise log-likelihoods extracted per rating per draw** (this is non-trivial — see F); Pareto-$\hat k$ diagnostics (% > 0.7, % > 1.0).
- **Sampler health**: max $\hat R$, min ESS bulk, divergence count, max tree depth.

Define pass/fail thresholds for each. They need not be publication-grade — they're "validation passes" gates. Suggest defaults that are strict enough to be meaningful but achievable for a model that genuinely works.

### E. Output structure

- Per-(seed, system, fit_variant) CSV for fine-grained inspection.
- Aggregated summary CSV per (fit_variant, R_true, system).
- Final markdown report in the style of `outputs/phase1_root_evidence/phase1c_evidence_attrition_report.md` (executive summary; per-pillar sections; pass/fail table at top).
- Plots: at minimum (a) PPC rating-distribution overlay per system, (b) $\rho_s$ calibration curve aggregated across seeds, (c) per-system $\rho$ vs realised $R_s$ confusion-matrix data.

### F. Compute budget

Each HMC fit on the production GWT model takes roughly 10–30 minutes wall clock. Rough scopes:

- Pilot: 10 seeds × 1 fit variant ≈ 2–5 hours
- Production: 50 seeds × 1 fit variant ≈ 10–25 hours
- Stretch: 50 × 2 variants (recommended baseline + negative control) ≈ 20–50 hours

Recommend a feasible scope given a two-week budget where validation is one of several priorities. Specify smoke mode (~1–2 seeds) and full mode explicitly. Per-rating LOO log-lik computation may add wall-clock — note this explicitly to Codex.

### G. Implementation considerations to flag for Codex

- **Reuse vs build:** the Phase 0/1 scaffolding is mature. Reuse `MultiSystemExactTreeBuilder`, the deterministics, `composite_vs_exact_diagnostic.py:exact_loglik`, `gwt_full_exact_recovery.py` patterns, and the `scripts/phase1_root_evidence_common.py` helpers. Build: probably a comprehensive scoring/PPC/LOO module that composes existing ingredients into one validation harness.
- **PSIS-LOO over the marginalised tree-DP likelihood is non-trivial.** PSIS-LOO needs per-rating pointwise log-likelihoods, not the per-system marginal. Codex will need to either (a) re-evaluate the leaf-likelihood mixture per rating per draw post-hoc using `exact_loglik` infrastructure, or (b) add per-rating deterministics to the model build. Recommend the route in your prompt.
- **PPC at the rating-distribution level** has existing infrastructure (`exact_tree_ppc.py`, `notebooks/asymmetric_prior_sweep_2026-05-10/posterior_predictive_rating_dist.py`). Recommend whether to extend or rewrite.
- **The targeted re-labelling fix (variant 2)** requires programmatically modifying the GWT tree JSON before fitting. The fitter consumes whatever tree it's given via `load_data` / `data_cache.json`. The generator and fitter must use the same modified tree.

## Recommended structure for your Codex prompt

Roughly this shape (you can deviate if better):

```text
# Codex task: main synthetic validation of [chosen variant] DCM

## Goal
[1 paragraph: ship validated baseline]

## Reuse vs build
[explicit list]

## Model variant + justification
[which fix; why]

## Synthetic data spec
[generator config, balanced roots, n_seeds, n_replicates]

## Fit spec
[chains, draws, sampler, model class]

## Metrics + pass thresholds
[four pillars × thresholds]

## Outputs
[CSVs, report, plots]

## Sanity / quality gates
[smoke mode, py_compile, consistency checks]

## Compute budget
[smoke vs full, expected wall clock]

## Commands
[python scripts/main_synthetic_validation.py ...]
```

## Constraints

- Hand-off to Dawn happens this week — anything she sees should be defensible.
- Don't propose new infrastructure unless necessary.
- Push back on premises if you think the "fix" choice is wrong, or the validation goal itself should be re-scoped given Phase 1C. In particular: if you think the right move is *not* to ship a fixed-variant baseline but instead to ship the unmodified production model with a clear "here's why it doesn't recover; here's what would" finding, say so — that's a defensible alternative.
- I (Claude, the user's pair) will review your Codex prompt for repo-specific gotchas, then the user will send it to Codex.
