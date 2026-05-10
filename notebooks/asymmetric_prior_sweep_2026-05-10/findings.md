# Asymmetric β_pres / β_abs prior sweep — findings

**Branch:** `asymmetric-beta-prior-sweep` off `synthetic-validation-checks`.
**Date:** 2026-05-10. **Stance:** GWT only.

## Bottom line

The asymmetric β prior change (`β_pres → ~1`, `β_abs → ~0`) **structurally
fixes the prior-implied transmission gap** that the 2026-05-07 work flagged
— q_j(Human) and q_j(ELIZA) now stay separated all the way down the tree
(see Pre-flight section). But in posterior fits (synthetic seed 06 + real
data), it **does not produce the expected improvement in C recovery or
rating-distribution fit**. The diagnostic for that is now clear: the rating
likelihood is nearly flat in C, and the per-indicator PPCs are bad across
*all* prior choices including the production baseline. The leaf model
(3-state z + ordinal probit) is the next thing to fix, not the prior.

Treat cross-prior comparisons in this doc as exploratory. The structural
prior-implied q_j gap is the only result that can stand alone right now;
the rest will need re-running once the leaf model is doing its job.

## Pre-flight: analytical prior-implied q_j vs depth

Walk the GWT tree analytically (no MCMC) using prior *mean* β values and
propagate `q_child = β_abs + parent_q · (β_pres − β_abs)` under each
candidate prior. Direct test of "does the prior transmit C signal?".

`eval/analytical_qj_by_prior_depth.csv`. Headline gap `q̄_Human − q̄_ELIZA`:

| prior | depth 1 | depth 2 | depth 3 |
|---|---|---|---|
| baseline (paper means) | 0.332 | 0.113 | **0.038** |
| `pres0.85_abs0.15` | 0.699 | 0.489 | 0.342 |
| `pres0.90_abs0.10` | 0.798 | 0.639 | **0.511** |
| `pres0.95_abs0.05` | 0.898 | 0.808 | 0.728 |

`figs/qj_vs_depth.png` shows the same data graphically. Under baseline the
prior-implied q_j collapses Human ↔ ELIZA at depth 3; under the new prior
they stay separated.

This is **not** a result that depends on data, sampler, or any other
modelling choice. It is a property of the β-prior centre alone.

## Sweep design

| # | scope | run_id suffix | β_pres | β_abs | sigma | status |
|---|---|---|---|---|---|---|
| 1 | synthetic, ref-anchored, seed 06 | `pres95_abs05_sig30` | 0.95 | 0.05 | 0.30 | ✓ |
| 2 | synthetic, ref-anchored, seed 06 | `pres90_abs10_sig30` | 0.90 | 0.10 | 0.30 | ✓ |
| 3 | synthetic, ref-anchored, seed 06 | `pres85_abs15_sig50` | 0.85 | 0.15 | 0.50 | ✓ |
| 4 | real data, ref-anchored | `pres90_abs10_sig30` | 0.90 | 0.10 | 0.30 | ✓ |
| 5 | synthetic, no-anchors, per-system | `pres90_abs10_sig30` | 0.90 | 0.10 | 0.30 | ✓ (8 div) |

Existing baselines on disk (no re-fit):
- Synthetic: `synthetic_validation_2026-05-06/runs/full_exact_recovery/...seed20260506/`.
- Real data: `results/gwt_exact_tree/three_state_pooled_abs_by_sd_exact_anchored.nc`.

## Results — synthetic seed 06 (free C recovery)

Truth: Human 0.999 (anchored), Chicken 0.25, LLM 0.10, ELIZA 0.001 (anchored).

| prior | Chicken median | LLM median | LLM signed bias | order |
|---|---|---|---|---|
| baseline | 0.113 [0.006, 0.439] | 0.159 [0.010, 0.546] | +0.059 | flipped |
| pres85/abs15/sig50 | 0.133 [0.007, 0.496] | 0.138 [0.007, 0.525] | +0.038 | flipped |
| pres90/abs10/sig30 | 0.118 [0.006, 0.460] | 0.123 [0.007, 0.484] | +0.023 | flipped |
| pres95/abs05/sig30 | 0.108 [0.005, 0.443] | 0.111 [0.007, 0.431] | +0.011 | tied |

LLM signed bias improves monotonically with prior strength. Chicken bias
roughly stable around −0.13. Coverage holds across all (intervals contain
truth). **C posterior contraction ≈ 1 across all priors** (`eval/per_run_*_c_posterior_vs_prior.csv`)
— posterior SD ≈ prior SD, i.e. the data adds essentially no information
about C beyond the prior.

## Results — real data (free C posteriors)

| prior | Chicken | LLM |
|---|---|---|
| production (paper-mu) | 0.252 [0.037, 0.613] | **0.111 [0.005, 0.446]** |
| centre (pres90/abs10/sig30) | 0.263 [0.044, 0.628] | **0.263 [0.044, 0.624]** |

**LLM P(C=1) shifts 0.111 → 0.263** under the new prior. Chicken barely
moves. Mechanism: the strong-transmission prior amplifies the modest
mid-range "yes" evidence in LLM ratings, pulling LLM up to chicken level.
Whether this direction is defensible is a research call — *not* a methods
verdict. The posterior moved on real data because the real-data likelihood
*does* have C information; on synthetic seed 06 it didn't.

## Results — fit 5: per-system, no-anchors, centre prior (synthetic seed 06)

This is the cleanest test of "does the new prior alone identify C?". Drop
the hard anchors on Human (truth 0.999) and ELIZA (truth 0.001). Each
system gets its own (a_s, kappa_s). Fit on the same synthetic seed-06
data with `pres0.90 / abs0.10 / sig0.30`.

| system | truth | post median | [p03, p97] | err |
|---|---|---|---|---|
| Human | 0.999 | **0.215** | [0.018, 0.589] | **−0.78** |
| Chicken | 0.250 | 0.109 | [0.004, 0.463] | −0.14 |
| LLM | 0.100 | 0.196 | [0.012, 0.588] | +0.10 |
| ELIZA | 0.001 | **0.122** | [0.008, 0.456] | **+0.12** |

Without anchors, **every system's C posterior collapses to ≈ the C prior
mean** (Beta(1,5) prior mean = 0.167). β posterior contractions ~1.03
(pres) / 1.04 (abs); C posterior contractions 0.91–1.14. The 8 divergences
are mild but the qualitative result is unambiguous.

**Implication:** the reference anchors were carrying *all* of the
per-system distinguishability. With per-system kappa absorbing the
per-system rating-distribution differences (Human-clusters-at-6,
ELIZA-clusters-at-0), nothing is left to identify C. Even the new
asymmetric prior (which transmits q_j cleanly down the tree) cannot
overcome this.

This sharpens the case for the next-experiment branches considerably:
- D (oracle nuisance): if pinning kappa at truth restores C
  identifiability, the per-system kappa flexibility was the dominant
  culprit (and B becomes second-order).
- B (better q_j → rating leaf): without pinned nuisance, the leaf model
  may not be enough either — but it is the only structural lever left.

## Results — per-indicator PPC chi² (mean across indicators)

| prior | Human | Chicken | LLM | ELIZA | mean |
|---|---|---|---|---|---|
| baseline | 6.47 | 7.41 | 9.35 | 5.50 | 7.18 |
| pres85/abs15/sig50 | 6.27 | 7.17 | 9.26 | 6.05 | 7.19 |
| pres90/abs10/sig30 | 7.63 | 7.08 | 9.21 | 5.85 | 7.44 |
| pres95/abs05/sig30 | 9.56 | 7.14 | 9.27 | 5.68 | 7.94 |

Aggregate chi² is roughly the same across baseline and the wider 85/15
prior; gets modestly worse with stronger priors. **All values are large in
absolute terms** — see `figs/ppc_rating_dist_synthetic.png` and
`figs/ppc_rating_dist_real_data.png` for the actual predicted vs observed
rating distributions. The model's predicted distributions look qualitatively
wrong across the board, so per-prior chi² differences are not an
appropriate ranking signal at this stage.

## Posterior-vs-prior diagnostic (all fits)

`eval/per_run_*_beta_posterior_vs_prior.csv` and `c_posterior_vs_prior.csv`.

- **β posteriors** (any prior, synthetic or real): mean over groups,
  posterior_mean ≈ prior_mu within ~0.01–0.04. Posterior SD ≈ prior SD
  (contraction ≈ 1). The β prior is essentially the β posterior — data
  doesn't move it. (Sigma=0.3 on the override is tight enough that it
  wouldn't move much anyway, but even at sigma=0.5 the posterior tracks the
  prior centre.)
- **C posteriors on synthetic data:** contraction ≈ 1. C posterior ≈ C
  prior. *Likelihood is flat in C at this design.*
- **C posteriors on real data:** contraction ≈ 1.13–1.14, but the *mean*
  moves substantially (0.167 prior → 0.285 posterior). The real data
  carries C information that the synthetic doesn't (or it carries different
  information).

## Read of the experiment

1. The β-prior change is a **structural fix to a real problem** (q_j
   transmission). It belongs in the model.
2. It is **not sufficient on its own** to produce good C recovery or
   rating-distribution fit. The leaf-model side (3-state z → ordinal probit
   → 7 categories) is the next bottleneck — `q_j → rating` compression is
   plausibly why the C-likelihood is flat.
3. Cross-prior comparisons of C medians, PPC chi², and similar are not a
   reliable basis for prior selection right now. Re-evaluate after fixing
   the leaf model.
4. Concrete next-experiment plan in `next_directions.md` — recommended
   order is D (oracle nuisance) → A (sample-size sweep) → B (continuous z /
   better q_j → rating) → E (soft anchors).

## Reproducibility

- Branch `asymmetric-beta-prior-sweep` off `synthetic-validation-checks`.
- Sweep driver: `notebooks/asymmetric_prior_sweep_2026-05-10/run_sweep.sh`.
- Per-fit logs: `notebooks/asymmetric_prior_sweep_2026-05-10/logs/sweep.log`.
- Per-fit outputs: `runs/synthetic/<run_id>/`, `runs/real_data/<override_tag>/`,
  `runs/per_system_noanchors/<run_id>/`.
- Eval: `aggregate_results.py` (cross-fit roll-up), `per_indicator_ppc.py`,
  `per_system_loo.py`, `posterior_vs_prior.py`, `posterior_predictive_rating_dist.py`.
- Figures: `figs/qj_vs_depth.png`, `figs/ppc_rating_dist_synthetic.png`,
  `figs/ppc_rating_dist_real_data.png`.

## Implementation note (for future branches)

The asymmetric prior knobs are now in `dcm_model.py`:

- `BETA_PRES_OVERRIDE_MEAN`, `BETA_ABS_OVERRIDE_MEAN`, `BETA_OVERRIDE_SIGMA`
  on `ModelConfig`.
- Wired into `build_label_pool_hyperparameters` (label-pool path only;
  override means require `POOL_BETAS_BY_LABEL=True`).
- CLI flags `--beta-pres-mean / --beta-abs-mean / --beta-override-sigma`
  on `gwt_full_exact_recovery.py` and
  `task5_recovery_per_system_obs.py`.
- Run-id suffix `__pres{p}_abs{a}_sig{s}` so existing baselines aren't
  clobbered.

Test invariants (`test_label_pooled_smoke.py`) still pass — defaults
preserved exactly.
