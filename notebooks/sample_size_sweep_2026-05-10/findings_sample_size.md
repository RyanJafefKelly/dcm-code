# Branch 6 — C-identifiability vs rater-multiplier K (sample-size sweep)

**Branch:** `c-id-sample-size-sweep` (off `asymmetric-beta-prior-sweep` @ 929063b)
**Date:** 2026-05-10
**Driver:** `notebooks/synthetic_validation_2026-05-06/gwt_full_exact_recovery.py --rater-multiplier K`
**DGP:** `exact_latent_tree`, seed 20260506
**Fit:** `full_exact_tree` (three_state leaf, label-pooled β, no per-system kappa/a/b).
**Prior:** centre β override (β_pres=0.90, β_abs=0.10, σ=0.30); root C ~ Beta(1, 5) on free systems.
**Free systems:** Chicken (truth 0.25), LLM (truth 0.10).  Human / ELIZA hard-anchored.
**Prior SD on C:** $\sqrt{5/252} \approx 0.1409$.

## What this run isolates

Holding everything else fixed (DGP, leaf, β prior, observation layer, anchor design),
each existing (rater, indicator) rating slot is replicated K times.  Each replicate
is an independent draw from the posterior-predictive given the indicator's latent
state; rater identities are preserved.  This grows rating count per indicator by K
without changing the rater pool or the design.

Decomposes "C is poorly identified because the leaf is too compressive" from
"C is poorly identified because we just don't have enough ratings":

- Posterior SD on $C$ shrinks at the parametric $1/\sqrt K$ rate → simply data-limited.
- Posterior SD asymptotes well above zero → structural ceiling in the leaf.

## Run table

| K | system | posterior_median | posterior_sd | contraction_ratio | signed_error | abs_error | interval_width | interval_includes_truth |
|---|---|---|---|---|---|---|---|---|
| 1 | Chicken | 0.1178 | 0.1306 | 0.927 | -0.1322 | 0.1322 | 0.4542 | True |
| 2 | Chicken | 0.1286 | 0.1414 | 1.004 | -0.1214 | 0.1214 | 0.4973 | True |
| 5 | Chicken | 0.1312 | 0.1432 | 1.016 | -0.1188 | 0.1188 | 0.5119 | True |
| 10 | Chicken | 0.1335 | 0.1422 | 1.010 | -0.1165 | 0.1165 | 0.5074 | True |
| 20 | Chicken | 0.1387 | 0.1421 | 1.009 | -0.1113 | 0.1113 | 0.5111 | True |
| 1 | LLM | 0.1226 | 0.1348 | 0.957 | +0.0226 | 0.0226 | 0.4771 | True |
| 2 | LLM | 0.1782 | 0.1570 | 1.114 | +0.0782 | 0.0782 | 0.5524 | True |
| 5 | LLM | 0.1818 | 0.1597 | 1.134 | +0.0818 | 0.0818 | 0.5648 | True |
| 10 | LLM | 0.1787 | 0.1598 | 1.134 | +0.0787 | 0.0787 | 0.5695 | True |
| 20 | LLM | 0.1808 | 0.1568 | 1.113 | +0.0808 | 0.0808 | 0.5565 | True |

Full table in `eval/sample_size_sweep_summary.csv`; markdown copy in
`eval/sample_size_sweep_summary.md`.

### Diagnostics

| K | divergences | max_rhat (any var) | min_ess_bulk (any var) | r_hat on C_chicken | r_hat on C_llm | ess_bulk C_llm |
|---|---|---|---|---|---|---|
| 1 | 0 | 1.01 | 2900 | 1.00 | 1.00 | 5918 |
| 2 | 0 | 1.00 | 2101 | 1.00 | 1.00 | 3994 |
| 5 | 0 | 1.53 | 7 | 1.01 | 1.05 | 60 |
| 10 | 0 | 1.53 | 7 | 1.01 | 1.04 | 68 |
| 20 | 0 | 1.53 | 7 | 1.01 | 1.05 | 55 |

The headline `max_rhat=1.53 / min_ess=7` from K=5 onwards comes from a label-pool
β-tilde singleton, not from C.  Per-system C draws have R-hat ≤ 1.05 and ≥55
ESS at every K, so the posterior SD on $C$ is estimated reliably.  Sampler
slowness is itself a flat-likelihood signal: NUTS step size adapts to a
posterior whose C-dimension is essentially the prior, so the chains drift
slowly across that dimension.

## Plots

- `figs/contraction_vs_sqrtK.png` — posterior SD on $C_s$ vs $\sqrt K$, with a
  $\propto 1/\sqrt K$ reference anchored at K=1.  Both Chicken and LLM lines
  are flat and sit *above* the reference for every K > 1.
- `figs/median_vs_K.png` — posterior median + 94% HDI vs K (log axis).  Truth
  (red) is inside the 94% HDI in every cell, but the median sits near the prior
  mean (≈ 0.17) and barely moves with K.

## Bottom line

**No K in {1, 2, 5, 10, 20} drives Chicken contraction below 0.5 — at K=20 it
is 1.01, statistically indistinguishable from the prior.**  Posterior SD on
$C$ asymptotes at the prior SD, not at zero, so shrinkage is **not parametric**:
the leaf likelihood at this design is approximately *uninformative* about
$C_s$ for the free systems, and 20× more raters reproduces the prior rather
than the truth.  **For Tuesday: more rater budget on the current design is the
wrong lever — fix the leaf first** (Branch 3 mixture is the next test, with
Branch 1 oracle-C ladder confirming it).

## Reproducibility

```bash
git checkout c-id-sample-size-sweep
bash notebooks/sample_size_sweep_2026-05-10/run_sweep.sh
python notebooks/sample_size_sweep_2026-05-10/aggregate_sweep.py
```
