# Final PPC Readiness Report

Decision: `NOT_READY_FIX_PPC`

## Revised PPC Hierarchy

Primary readiness gates are implementation sanity, RPS improvement, oracle-calibrated weighted mean cell TV, system-specific weighted TV, and direct predictive alignment to truth. Sparse unweighted median/q90/fraction TV rows are warnings when `n_cells_ge_5 = 0` unless weighted TV also shows practical excess.

## Overall Oracle-Calibrated PPC

| fit_variant | seed | system | n_ratings | n_cells | n_cells_ge_5 | metric | fitted_ppc_value | oracle_expected_mean | oracle_expected_median | oracle_q05 | oracle_q50 | oracle_q95 | fitted_minus_oracle_median | fitted_percentile_under_oracle_null | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | 20260511 | ALL | 379 | 199 | 0 | weighted_mean_cell_TV | 0.424 | 0.4134 | 0.4131 | 0.3954 | 0.4131 | 0.4328 | 0.01089 | 0.836 | compatible_with_finite_sample_noise |
| targeted_strong_lower_override | 20260511 | ALL | 379 | 199 | 0 | median_cell_TV | 0.4309 | 0.3739 | 0.3728 | 0.3728 | 0.3728 | 0.3729 | 0.05812 | 1 | ppc_excess_misfit |
| targeted_strong_lower_override | 20260511 | ALL | 379 | 199 | 0 | q90_cell_TV | 0.8388 | 0.8742 | 0.877 | 0.8511 | 0.877 | 0.8811 | -0.03824 | 0.012 | ppc_underdispersed_or_metric_issue |
| targeted_strong_lower_override | 20260511 | ALL | 379 | 199 | 0 | fraction_cell_TV_gt_0p5 | 0.3668 | 0.3782 | 0.3769 | 0.3417 | 0.3769 | 0.4221 | -0.01005 | 0.29 | compatible_with_finite_sample_noise |
| targeted_strong_lower_override | 20260512 | ALL | 379 | 199 | 0 | weighted_mean_cell_TV | 0.3645 | 0.366 | 0.3653 | 0.3469 | 0.3653 | 0.3858 | -0.0007432 | 0.474 | compatible_with_finite_sample_noise |
| targeted_strong_lower_override | 20260512 | ALL | 379 | 199 | 0 | median_cell_TV | 0.3207 | 0.3718 | 0.3728 | 0.3728 | 0.3728 | 0.3728 | -0.05211 | 0.012 | ppc_underdispersed_or_metric_issue |
| targeted_strong_lower_override | 20260512 | ALL | 379 | 199 | 0 | q90_cell_TV | 0.8352 | 0.8565 | 0.8763 | 0.7887 | 0.8763 | 0.8843 | -0.04106 | 0.178 | compatible_with_finite_sample_noise |
| targeted_strong_lower_override | 20260512 | ALL | 379 | 199 | 0 | fraction_cell_TV_gt_0p5 | 0.2462 | 0.2895 | 0.2889 | 0.2513 | 0.2889 | 0.3317 | -0.04271 | 0.024 | ppc_underdispersed_or_metric_issue |

## System-Specific Weighted TV

| fit_variant | seed | system | n_ratings | n_cells | n_cells_ge_5 | metric | fitted_ppc_value | oracle_expected_mean | oracle_expected_median | oracle_q05 | oracle_q50 | oracle_q95 | fitted_minus_oracle_median | fitted_percentile_under_oracle_null | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | 20260511 | Human | 50 | 50 | 0 | weighted_mean_cell_TV | 0.6157 | 0.6057 | 0.6046 | 0.5599 | 0.6046 | 0.6496 | 0.01113 | 0.658 | compatible_with_finite_sample_noise |
| targeted_strong_lower_override | 20260511 | Chicken | 93 | 49 | 0 | weighted_mean_cell_TV | 0.4457 | 0.4448 | 0.4437 | 0.4101 | 0.4437 | 0.4795 | 0.002023 | 0.544 | compatible_with_finite_sample_noise |
| targeted_strong_lower_override | 20260511 | LLMs | 186 | 50 | 0 | weighted_mean_cell_TV | 0.3212 | 0.321 | 0.3208 | 0.2966 | 0.3208 | 0.3488 | 0.000434 | 0.514 | compatible_with_finite_sample_noise |
| targeted_strong_lower_override | 20260511 | ELIZA | 50 | 50 | 0 | weighted_mean_cell_TV | 0.5744 | 0.5062 | 0.5063 | 0.4555 | 0.5063 | 0.5614 | 0.06804 | 0.982 | ppc_excess_misfit |
| targeted_strong_lower_override | 20260512 | Human | 50 | 50 | 0 | weighted_mean_cell_TV | 0.4821 | 0.5398 | 0.5385 | 0.4902 | 0.5385 | 0.5924 | -0.05632 | 0.028 | ppc_underdispersed_or_metric_issue |
| targeted_strong_lower_override | 20260512 | Chicken | 93 | 49 | 0 | weighted_mean_cell_TV | 0.3596 | 0.342 | 0.3408 | 0.3076 | 0.3408 | 0.3798 | 0.0188 | 0.774 | compatible_with_finite_sample_noise |
| targeted_strong_lower_override | 20260512 | LLMs | 186 | 50 | 0 | weighted_mean_cell_TV | 0.3133 | 0.3152 | 0.3141 | 0.2861 | 0.3141 | 0.3437 | -0.0007835 | 0.49 | compatible_with_finite_sample_noise |
| targeted_strong_lower_override | 20260512 | ELIZA | 50 | 50 | 0 | weighted_mean_cell_TV | 0.4465 | 0.4258 | 0.4221 | 0.3747 | 0.4221 | 0.4892 | 0.02443 | 0.742 | compatible_with_finite_sample_noise |

## Predictive Alignment To Truth

| fit_variant | seed | system | n_ratings | weighted_mean_tv_fit_vs_truth | median_tv_fit_vs_truth | q90_tv_fit_vs_truth | weighted_mean_high_abs_error_fit_vs_truth | weighted_mean_top7_abs_error_fit_vs_truth | mean_kl_truth_to_fit | mean_kl_fit_to_truth | mean_rps_truth_expected_fit | alignment_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | 20260511 | ALL | 379 | 0.1208 | 0.06957 | 0.2894 | 0.06174 | 0.05476 | 0.09191 | 0.1292 | 0.09929 | pass |
| targeted_strong_lower_override | 20260511 | Chicken | 93 | 0.1349 | 0.05444 | 0.3645 | 0.08835 | 0.09329 | 0.131 | 0.1918 | 0.1014 | pass |
| targeted_strong_lower_override | 20260511 | ELIZA | 50 | 0.1889 | 0.1184 | 0.3966 | 0.08133 | 0.0603 | 0.1471 | 0.2236 | 0.1081 | pass |
| targeted_strong_lower_override | 20260511 | Human | 50 | 0.1898 | 0.1137 | 0.4571 | 0.1511 | 0.1379 | 0.2067 | 0.2654 | 0.1251 | pass |
| targeted_strong_lower_override | 20260511 | LLMs | 186 | 0.07693 | 0.06881 | 0.08861 | 0.01913 | 0.01165 | 0.02663 | 0.03601 | 0.08895 | pass |
| targeted_strong_lower_override | 20260512 | ALL | 379 | 0.1168 | 0.06883 | 0.2847 | 0.04586 | 0.04706 | 0.09531 | 0.1469 | 0.08946 | pass |
| targeted_strong_lower_override | 20260512 | Chicken | 93 | 0.108 | 0.06507 | 0.2332 | 0.04522 | 0.04089 | 0.08413 | 0.132 | 0.07358 | pass |
| targeted_strong_lower_override | 20260512 | ELIZA | 50 | 0.1749 | 0.06249 | 0.4453 | 0.09387 | 0.06398 | 0.1915 | 0.3226 | 0.1004 | pass |
| targeted_strong_lower_override | 20260512 | Human | 50 | 0.1717 | 0.06381 | 0.4786 | 0.09553 | 0.09546 | 0.1873 | 0.3054 | 0.1093 | pass |
| targeted_strong_lower_override | 20260512 | LLMs | 186 | 0.09076 | 0.06954 | 0.1204 | 0.01992 | 0.03258 | 0.0503 | 0.06459 | 0.0891 | pass |

## PPC Sanity

| fit_variant | seed | run_dir | check | status | detail |
|---|---|---|---|---|---|
| targeted_strong_lower_override | 20260511 | runs/targeted_strong_lower_override/seed_20260511 | same_fit_seed_artifacts_present | pass | posterior_predictive.nc, fit.nc, and truth_payload.json exist |
| targeted_strong_lower_override | 20260511 | runs/targeted_strong_lower_override/seed_20260511 | category_probabilities_sum_to_one | pass | max_abs_sum_minus_one=1.110e-15 |
| targeted_strong_lower_override | 20260511 | runs/targeted_strong_lower_override/seed_20260511 | ratings_in_valid_categories | pass | internal categories are 0..6, corresponding to reported 1..7 |
| targeted_strong_lower_override | 20260511 | runs/targeted_strong_lower_override/seed_20260511 | per_seed_ppc_summary_present | pass | PPC metrics are computed per seed before aggregate summaries |
| targeted_strong_lower_override | 20260511 | runs/targeted_strong_lower_override/seed_20260511 | system_ppc_p_value_status | pass | legacy smoke rows: system-specific p-values are intentionally not computed; readiness report documents this |
| targeted_strong_lower_override | 20260512 | runs/targeted_strong_lower_override/seed_20260512 | same_fit_seed_artifacts_present | pass | posterior_predictive.nc, fit.nc, and truth_payload.json exist |
| targeted_strong_lower_override | 20260512 | runs/targeted_strong_lower_override/seed_20260512 | category_probabilities_sum_to_one | pass | max_abs_sum_minus_one=9.992e-16 |
| targeted_strong_lower_override | 20260512 | runs/targeted_strong_lower_override/seed_20260512 | ratings_in_valid_categories | pass | internal categories are 0..6, corresponding to reported 1..7 |
| targeted_strong_lower_override | 20260512 | runs/targeted_strong_lower_override/seed_20260512 | per_seed_ppc_summary_present | pass | PPC metrics are computed per seed before aggregate summaries |
| targeted_strong_lower_override | 20260512 | runs/targeted_strong_lower_override/seed_20260512 | system_ppc_p_value_status | pass | legacy smoke rows: system-specific p-values are intentionally not computed; readiness report documents this |

## Revised Decision

| decision | implementation_sanity_pass | targeted_override_sanity_pass | l4_targeted_no_fit_pass | full_default_sampler_pass | oracle_root_no_severe_mismatch | overall_weighted_tv_oracle_compatible | system_weighted_tv_no_practical_excess | predictive_alignment_status | blocking_reason | blocking_details | recommended_command |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NOT_READY_FIX_PPC | True | True | True | True | True | True | False | pass | oracle_finite_sample_excess | ELIZA/seed20260511 excess=0.068 |  |

## Recommended Command

No pilot command recommended. Blocking reason: `oracle_finite_sample_excess`. ELIZA/seed20260511 excess=0.068
