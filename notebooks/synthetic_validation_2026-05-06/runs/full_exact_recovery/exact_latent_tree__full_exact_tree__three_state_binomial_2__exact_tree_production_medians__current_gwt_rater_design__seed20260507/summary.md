# Full-GWT Exact/Exact Fake-Data Recovery Pilot

Labels:

- DGP: `exact_latent_tree`
- Fit: `full_exact_tree`
- Leaf: `three_state_binomial_2`
- Nuisance truth: `exact_tree_production_medians`
- Design: `current_gwt_rater_design`

Mode: full
Diagnostic status: `passed`
Elapsed build seconds: 5.9
Elapsed sample seconds: 991.2
Divergences: 0
Max R-hat: 1.0000
Min bulk ESS: 2008

## Plain-English Interpretation

This is the first full fake-data recovery check: data were generated from the exact GWT tree and then fit with the same exact-tree model. Unlike the oracle audits, nuisance parameters are learned rather than held fixed.

Because this is one synthetic seed, signed errors are recovery errors for this pilot, not Monte Carlo bias estimates. Multi-seed runs are needed before treating RMSE or coverage as stable calibration claims.

If diagnostics fail, read the numerical recovery tables as exploratory engineering output rather than as a model validation result.

Observed-scale PPC intervals include posterior uncertainty and synthetic ordinal rating noise. They are intentionally wider than intervals over posterior expected proportions alone.

## Free Root C Recovery

| system | truth | posterior_median | posterior_p03 | posterior_p97 | signed_error | interval_includes_truth |
|---|---|---|---|---|---|---|
| Chicken | 0.250 | 0.112 | 0.005 | 0.461 | -0.138 | True |
| 2024 Leading Chat LLMs | 0.100 | 0.141 | 0.007 | 0.529 | 0.041 | True |

## Label Beta Recovery

| kind | n | mean_signed_error | mae | rmse | coverage_94 |
|---|---|---|---|---|---|
| beta_abs | 18 | 0.061 | 0.064 | 0.080 | 1.000 |
| beta_pres | 18 | -0.023 | 0.029 | 0.040 | 1.000 |

## Edge Beta Recovery

| kind | n | mean_signed_error | mae | rmse | coverage_94 |
|---|---|---|---|---|---|
| beta_abs | 75 | 0.057 | 0.058 | 0.074 | 1.000 |
| beta_pres | 75 | -0.029 | 0.033 | 0.041 | 1.000 |

## Path Summary Recovery

| estimand | n | mean_signed_error | mae | rmse | coverage_94 |
|---|---|---|---|---|---|
| delta_j | 50 | -0.055 | 0.056 | 0.065 | 0.880 |
| q_gap_999_001 | 50 | -0.055 | 0.055 | 0.065 | 0.880 |

## Internal State Probability Scores

| n | mean_relative_entropy_reduction | mean_brier | mean_neg_log_score |
|---|---|---|---|
| 100.000 | 0.350 | 0.085 | 0.286 |

## Observed-Scale Feature-Block PPC

| n | mean_abs_prop_error | coverage_94 |
|---|---|---|
| 196.000 | 0.124 | 0.959 |
