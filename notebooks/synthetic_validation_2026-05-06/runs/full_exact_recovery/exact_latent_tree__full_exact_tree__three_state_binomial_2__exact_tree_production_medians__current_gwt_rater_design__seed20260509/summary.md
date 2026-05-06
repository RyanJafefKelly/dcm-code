# Full-GWT Exact/Exact Fake-Data Recovery Pilot

Labels:

- DGP: `exact_latent_tree`
- Fit: `full_exact_tree`
- Leaf: `three_state_binomial_2`
- Nuisance truth: `exact_tree_production_medians`
- Design: `current_gwt_rater_design`

Mode: full
Diagnostic status: `passed`
Elapsed build seconds: 6.7
Elapsed sample seconds: 965.5
Divergences: 0
Max R-hat: 1.0000
Min bulk ESS: 2552

## Plain-English Interpretation

This is the first full fake-data recovery check: data were generated from the exact GWT tree and then fit with the same exact-tree model. Unlike the oracle audits, nuisance parameters are learned rather than held fixed.

Because this is one synthetic seed, signed errors are recovery errors for this pilot, not Monte Carlo bias estimates. Multi-seed runs are needed before treating RMSE or coverage as stable calibration claims.

If diagnostics fail, read the numerical recovery tables as exploratory engineering output rather than as a model validation result.

Observed-scale PPC intervals include posterior uncertainty and synthetic ordinal rating noise. They are intentionally wider than intervals over posterior expected proportions alone.

## Free Root C Recovery

| system | truth | posterior_median | posterior_p03 | posterior_p97 | signed_error | interval_includes_truth |
|---|---|---|---|---|---|---|
| Chicken | 0.250 | 0.121 | 0.005 | 0.499 | -0.129 | True |
| 2024 Leading Chat LLMs | 0.100 | 0.111 | 0.005 | 0.474 | 0.011 | True |

## Label Beta Recovery

| kind | n | mean_signed_error | mae | rmse | coverage_94 |
|---|---|---|---|---|---|
| beta_abs | 18 | 0.052 | 0.069 | 0.086 | 1.000 |
| beta_pres | 18 | -0.052 | 0.054 | 0.082 | 0.944 |

## Edge Beta Recovery

| kind | n | mean_signed_error | mae | rmse | coverage_94 |
|---|---|---|---|---|---|
| beta_abs | 75 | 0.044 | 0.066 | 0.083 | 1.000 |
| beta_pres | 75 | -0.078 | 0.078 | 0.108 | 0.853 |

## Path Summary Recovery

| estimand | n | mean_signed_error | mae | rmse | coverage_94 |
|---|---|---|---|---|---|
| delta_j | 50 | -0.068 | 0.070 | 0.081 | 0.600 |
| q_gap_999_001 | 50 | -0.068 | 0.070 | 0.081 | 0.600 |

## Internal State Probability Scores

| n | mean_relative_entropy_reduction | mean_brier | mean_neg_log_score |
|---|---|---|---|
| 100.000 | 0.237 | 0.109 | 0.362 |

## Observed-Scale Feature-Block PPC

| n | mean_abs_prop_error | coverage_94 |
|---|---|---|
| 196.000 | 0.115 | 0.944 |
