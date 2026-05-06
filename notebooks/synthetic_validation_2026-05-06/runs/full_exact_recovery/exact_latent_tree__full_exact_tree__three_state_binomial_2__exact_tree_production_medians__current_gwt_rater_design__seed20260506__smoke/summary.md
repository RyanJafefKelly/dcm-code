# Full-GWT Exact/Exact Fake-Data Recovery Pilot

Labels:

- DGP: `exact_latent_tree`
- Fit: `full_exact_tree`
- Leaf: `three_state_binomial_2`
- Nuisance truth: `exact_tree_production_medians`
- Design: `current_gwt_rater_design`

Mode: smoke
Diagnostic status: `exploratory_failed`
Elapsed build seconds: 6.4
Elapsed sample seconds: 556.1
Divergences: 0
Max R-hat: 1.0900
Min bulk ESS: 76

## Plain-English Interpretation

This is the first full fake-data recovery check: data were generated from the exact GWT tree and then fit with the same exact-tree model. Unlike the oracle audits, nuisance parameters are learned rather than held fixed.

Because this is one synthetic seed, signed errors are recovery errors for this pilot, not Monte Carlo bias estimates. Multi-seed runs are needed before treating RMSE or coverage as stable calibration claims.

If diagnostics fail, read the numerical recovery tables as exploratory engineering output rather than as a model validation result.

## Free Root C Recovery

| system | truth | posterior_median | posterior_p03 | posterior_p97 | signed_error | interval_includes_truth |
|---|---|---|---|---|---|---|
| Chicken | 0.250 | 0.118 | 0.006 | 0.407 | -0.132 | True |
| 2024 Leading Chat LLMs | 0.100 | 0.152 | 0.014 | 0.479 | 0.052 | True |

## Label Beta Recovery

| kind | n | mean_signed_error | mae | rmse | coverage_94 |
|---|---|---|---|---|---|
| beta_abs | 18 | 0.040 | 0.056 | 0.065 | 1.000 |
| beta_pres | 18 | -0.039 | 0.040 | 0.053 | 0.944 |

## Edge Beta Recovery

| kind | n | mean_signed_error | mae | rmse | coverage_94 |
|---|---|---|---|---|---|
| beta_abs | 75 | 0.034 | 0.064 | 0.072 | 1.000 |
| beta_pres | 75 | -0.051 | 0.052 | 0.067 | 0.853 |

## Path Summary Recovery

| estimand | n | mean_signed_error | mae | rmse | coverage_94 |
|---|---|---|---|---|---|
| delta_j | 50 | -0.054 | 0.054 | 0.061 | 0.900 |
| q_gap_999_001 | 50 | -0.053 | 0.054 | 0.061 | 0.900 |

## Internal State Probability Scores

| n | mean_relative_entropy_reduction | mean_brier | mean_neg_log_score |
|---|---|---|---|
| 100.000 | 0.329 | 0.111 | 0.343 |

## Observed-Scale Feature-Block PPC

| n | mean_abs_prop_error | coverage_94 |
|---|---|---|
| 196.000 | 0.116 | 0.199 |
