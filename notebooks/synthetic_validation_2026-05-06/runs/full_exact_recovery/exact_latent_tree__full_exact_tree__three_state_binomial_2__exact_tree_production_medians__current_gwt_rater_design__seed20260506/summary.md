# Full-GWT Exact/Exact Fake-Data Recovery Pilot

Labels:

- DGP: `exact_latent_tree`
- Fit: `full_exact_tree`
- Leaf: `three_state_binomial_2`
- Nuisance truth: `exact_tree_production_medians`
- Design: `current_gwt_rater_design`

Mode: full
Diagnostic status: `passed`
Elapsed build seconds: 6.2
Elapsed sample seconds: 1005.0
Divergences: 0
Max R-hat: 1.0000
Min bulk ESS: 3181

## Plain-English Interpretation

This is the first full fake-data recovery check: data were generated from the exact GWT tree and then fit with the same exact-tree model. Unlike the oracle audits, nuisance parameters are learned rather than held fixed.

Because this is one synthetic seed, signed errors are recovery errors for this pilot, not Monte Carlo bias estimates. Multi-seed runs are needed before treating RMSE or coverage as stable calibration claims.

If diagnostics fail, read the numerical recovery tables as exploratory engineering output rather than as a model validation result.

Observed-scale PPC intervals include posterior uncertainty and synthetic ordinal rating noise. They are intentionally wider than intervals over posterior expected proportions alone.

## Free Root C Recovery

| system | truth | posterior_median | posterior_p03 | posterior_p97 | signed_error | interval_includes_truth |
|---|---|---|---|---|---|---|
| Chicken | 0.250 | 0.113 | 0.006 | 0.439 | -0.137 | True |
| 2024 Leading Chat LLMs | 0.100 | 0.159 | 0.010 | 0.546 | 0.059 | True |

## Label Beta Recovery

| kind | n | mean_signed_error | mae | rmse | coverage_94 |
|---|---|---|---|---|---|
| beta_abs | 18 | 0.040 | 0.054 | 0.064 | 1.000 |
| beta_pres | 18 | -0.039 | 0.039 | 0.053 | 1.000 |

## Edge Beta Recovery

| kind | n | mean_signed_error | mae | rmse | coverage_94 |
|---|---|---|---|---|---|
| beta_abs | 75 | 0.035 | 0.062 | 0.071 | 1.000 |
| beta_pres | 75 | -0.051 | 0.051 | 0.065 | 1.000 |

## Path Summary Recovery

| estimand | n | mean_signed_error | mae | rmse | coverage_94 |
|---|---|---|---|---|---|
| delta_j | 50 | -0.054 | 0.054 | 0.062 | 0.920 |
| q_gap_999_001 | 50 | -0.054 | 0.054 | 0.062 | 0.920 |

## Internal State Probability Scores

| n | mean_relative_entropy_reduction | mean_brier | mean_neg_log_score |
|---|---|---|---|
| 100.000 | 0.333 | 0.111 | 0.343 |

## Observed-Scale Feature-Block PPC

| n | mean_abs_prop_error | coverage_94 |
|---|---|---|
| 196.000 | 0.116 | 0.990 |
