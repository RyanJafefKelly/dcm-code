# Full-GWT Exact/Exact Fake-Data Recovery Pilot

Labels:

- DGP: `exact_latent_tree`
- Fit: `full_exact_tree`
- Leaf: `three_state_binomial_2`
- Nuisance truth: `exact_tree_production_medians`
- Design: `current_gwt_rater_design`

Mode: full
Diagnostic status: `passed`
Elapsed build seconds: 3.9
Elapsed sample seconds: 434.2
Divergences: 0
Max R-hat: 1.0000
Min bulk ESS: 5248

## Plain-English Interpretation

This is the first full fake-data recovery check: data were generated from the exact GWT tree and then fit with the same exact-tree model. Unlike the oracle audits, nuisance parameters are learned rather than held fixed.

Because this is one synthetic seed, signed errors are recovery errors for this pilot, not Monte Carlo bias estimates. Multi-seed runs are needed before treating RMSE or coverage as stable calibration claims.

If diagnostics fail, read the numerical recovery tables as exploratory engineering output rather than as a model validation result.

Observed-scale PPC intervals include posterior uncertainty and synthetic ordinal rating noise. They are intentionally wider than intervals over posterior expected proportions alone.

## Free Root C Recovery

| system | truth | posterior_median | posterior_p03 | posterior_p97 | signed_error | interval_includes_truth |
|---|---|---|---|---|---|---|
| Chicken | 0.250 | 0.115 | 0.006 | 0.460 | -0.135 | True |
| 2024 Leading Chat LLMs | 0.100 | 0.127 | 0.007 | 0.493 | 0.027 | True |

## Label Beta Recovery

| kind | n | mean_signed_error | mae | rmse | coverage_94 |
|---|---|---|---|---|---|
| beta_abs | 18 | -0.307 | 0.308 | 0.360 | 0.111 |
| beta_pres | 18 | 0.114 | 0.126 | 0.175 | 0.389 |

## Edge Beta Recovery

| kind | n | mean_signed_error | mae | rmse | coverage_94 |
|---|---|---|---|---|---|
| beta_abs | 75 | -0.243 | 0.243 | 0.281 | 0.080 |
| beta_pres | 75 | 0.084 | 0.090 | 0.121 | 0.440 |

## Path Summary Recovery

| estimand | n | mean_signed_error | mae | rmse | coverage_94 |
|---|---|---|---|---|---|
| delta_j | 50 | 0.392 | 0.392 | 0.401 | 0.000 |
| q_gap_999_001 | 50 | 0.391 | 0.391 | 0.400 | 0.000 |

## Internal State Probability Scores

| n | mean_relative_entropy_reduction | mean_brier | mean_neg_log_score |
|---|---|---|---|
| 100.000 | 0.478 | 0.211 | 0.620 |

## Observed-Scale Feature-Block PPC

| n | mean_abs_prop_error | coverage_94 |
|---|---|---|
| 196.000 | 0.118 | 0.985 |
