# Smoke Refresh Report

Input: `outputs/main_synthetic_validation`
Output: `outputs/post_ladder_diagnostics_20260511/smoke_refresh`

## Summary

- Top-level smoke summaries stale or inconsistent: `True`.
- Refreshed root recovery still fails for the LLM split.
- The failed LLM R=1 case is also a no-HMC/oracle failure, not just an HMC posterior flip.
- Sampler diagnostics acceptable enough for acceptance interpretation: `False`.
- PPC legacy TV gates still fail, but all system x indicator cells are sparse and oracle-calibrated PPC files indicate weighted TV is mostly compatible with finite-sample noise.

## Refreshed Root Recovery

| fit_variant | system | n_cases | mean_rho_R1 | mean_rho_R0 | median_rho_R1 | median_rho_R0 | median_log_B_eff_R1 | median_log_B_eff_R0 | evidence_margin_M | TPR_at_rho_gt_0p5 | TNR_at_rho_le_0p5 | balanced_accuracy | decisive_present_rate_R1_at_rho_gt_0p95 | decisive_absent_rate_R0_at_rho_lt_0p05 | brier_model | brier_prior_baseline | brier_improvement | log_score_model | log_score_prior_baseline | log_score_improvement | ECE | calibration_bins | mean_abs_rho_collapsed_minus_sampled_pi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| targeted_strong_lower_override | ALL | 4 | 0.509804 | 0.00214514 | 0.509804 | 0.00214514 | 2.0573 | -4.78848 | 0.447859 | 0.5 | 1 | 0.75 | 0.5 | 1 | 0.23353 | 0.361111 | 0.127581 | -0.85284 | -0.987041 | 0.1342 | 0.244026 | [{"lo": 0.0, "hi": 0.1, "n": 3, "mean_rho": 0.01263305361874046, "empirical_rate": 0.3333333333333333}, {"lo": 0.9, "hi": 1.0, "n": 1, "mean_rho": 0.985998572588115, "empirical_rate": 1.0}] | 0.00179575 |
| targeted_strong_lower_override | Chicken | 2 | 0.985999 | 0.00350204 | 0.985999 | 0.00350204 | 5.86393 | -4.04146 | 2.70646 | 1 | 1 | 1 | 1 | 1 | 0.000104152 | 0.361111 | 0.361007 | -0.00880428 | -0.987041 | 0.978236 | 0.00875173 | [{"lo": 0.0, "hi": 0.1, "n": 1, "mean_rho": 0.0035020373865153488, "empirical_rate": 0.0}, {"lo": 0.9, "hi": 1.0, "n": 1, "mean_rho": 0.985998572588115, "empirical_rate": 1.0}] | 0.000311525 |
| targeted_strong_lower_override | LLMs | 2 | 0.0336089 | 0.000788234 | 0.0336089 | 0.000788234 | -1.74934 | -5.53549 | -3.35878 | 0 | 1 | 0.5 | 0 | 1 | 0.466956 | 0.361111 | -0.105845 | -1.69688 | -0.987041 | -0.709836 | 0.482801 | [{"lo": 0.0, "hi": 0.1, "n": 2, "mean_rho": 0.017198561734853016, "empirical_rate": 0.5}] | 0.00327997 |

## LLM R=1 Bridge

| seed | root_z_true | rho_collapsed | rho_collapsed_median | log_B_eff | log_B_draw_median | rho_oracle_collapsed | log_B_oracle | delta_eff_minus_oracle | posterior_correct_sign | oracle_correct_sign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260512 | 1 | 0.0336089 | 0.0200211 | -1.74934 | -2.28131 | 0.0400138 | -1.56826 | -0.181085 | False | False |

The generated latent/root truth for the LLM failed case is `root_z_true=1`. Both oracle/clamped evidence and HMC posterior evidence support absence: oracle `log_B` is negative and HMC effective `log_B` is also negative. The HMC-minus-oracle gap is small relative to the sign error, so the available artefacts point to an unlucky or weak generated LLM dataset rather than beta mismatch or nuisance inference as the primary cause.

## PPC Refresh

| fit_variant | system | n_ratings | mean_RPS | median_RPS | q90_RPS | mean_RPS_baseline_empirical_marginal | mean_RPS_improvement | weighted_mean_cell_TV | median_cell_TV | q90_cell_TV | max_cell_TV | fraction_cell_TV_gt_0p3 | fraction_cell_TV_gt_0p5 | n_cells | n_cells_ge_5 | mean_abs_top7_error_high_q | q90_abs_top7_error_high_q | mean_abs_high_error_high_q | q90_abs_high_error_high_q | posterior_predictive_tv_p_value | ppc_draws | n_seeds | seed_min | seed_max | seed_list | posterior_predictive_tv_p_value_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| targeted_strong_lower_override | ALL | 379 | 0.0720782 | 0.0380735 | 0.195095 | 0.230902 | 0.158824 | 0.393942 | 0.373152 | 0.837273 | 0.941326 | 0.650754 | 0.30402 | 199 | 0 | 0.323843 | 0.468365 | 0.1995 | 0.312854 | 1 | 300 | 2 | 20260511 | 20260512 | 20260511;20260512 | computed_overall |
| targeted_strong_lower_override | Chicken | 93 | 0.0616851 | 0.028643 | 0.134954 | 0.228917 | 0.167232 | 0.401697 | 0.368909 | 0.706533 | 0.853268 | 0.693878 | 0.22449 | 49 | 0 | 0.246395 | 0.363343 | 0.181127 | 0.315033 | NA | 300 | 2 | 20260511 | 20260512 | 20260511;20260512 | not_computed_system_specific |
| targeted_strong_lower_override | ELIZA | 50 | 0.0594732 | 0.0340767 | 0.149112 | 0.217128 | 0.157655 | 0.510154 | 0.415364 | 0.899378 | 0.941326 | 0.53 | 0.42 | 50 | 0 | 0.437193 | 0.551058 | 0.299671 | 0.54767 | NA | 300 | 2 | 20260511 | 20260512 | 20260511;20260512 | not_computed_system_specific |
| targeted_strong_lower_override | Human | 50 | 0.0612447 | 0.0383561 | 0.12919 | 0.247361 | 0.186116 | 0.548124 | 0.499414 | 0.89237 | 0.935032 | 0.93 | 0.42 | 50 | 0 | 0.398073 | 0.494354 | 0.219396 | 0.419247 | NA | 300 | 2 | 20260511 | 20260512 | 20260511;20260512 | not_computed_system_specific |
| targeted_strong_lower_override | LLMs | 186 | 0.0835753 | 0.0396444 | 0.247807 | 0.231173 | 0.147597 | 0.317377 | 0.285038 | 0.563066 | 0.742553 | 0.45 | 0.15 | 50 | 0 | 0.174962 | 0.286976 | 0.13765 | 0.14326 | NA | 300 | 2 | 20260511 | 20260512 | 20260511;20260512 | not_computed_system_specific |

PPC NetCDF checks:

| fit_variant | seed | n_ratings | ppc_draws_in_nc | mean_RPS_from_nc | max_abs_category_prob_sum_minus_one | has_system_or_node_metadata_in_nc | cell_tv_recomputable_from_nc_alone | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| targeted_strong_lower_override | 20260511 | 379 | 500 | 0.079265 | 1.11022e-15 | False | False | posterior_predictive.nc stores rating_row/category arrays but no system/node_key row metadata |
| targeted_strong_lower_override | 20260512 | 379 | 100 | 0.0648914 | 9.99201e-16 | False | False | posterior_predictive.nc stores rating_row/category arrays but no system/node_key row metadata |

Cell-count sparsity by system:

| system | n_cells | min_n_obs | median_n_obs | q90_n_obs | max_n_obs | n_cells_ge_5 |
| --- | --- | --- | --- | --- | --- | --- |
| Chicken | 100 | 0 | 2 | 2 | 2 | 0 |
| ELIZA | 100 | 1 | 1 | 1 | 1 | 0 |
| Human | 100 | 1 | 1 | 1 | 1 | 0 |
| LLMs | 100 | 3 | 4 | 4 | 4 | 0 |

Refreshed ALL weighted mean cell TV is `0.393942`, median cell TV is `0.373152`, q90 cell TV is `0.837273`, fraction TV > 0.5 is `0.30402`, and PPC TV p-value is `1`.
The existing `posterior_predictive.nc` files are sufficient to verify overall RPS and probability normalization, but they do not store system/node row metadata. System and cell TV refresh therefore uses the existing per-run PPC summary artefacts rather than reconstructing cell TV from NetCDF alone.

Oracle-calibrated weighted TV rows from existing top-level PPC audit:

| seed | system | fitted_ppc_value | oracle_expected_median | oracle_q05 | oracle_q95 | status |
| --- | --- | --- | --- | --- | --- | --- |
| 20260511 | ALL | 0.424033 | 0.413146 | 0.395414 | 0.432759 | compatible_with_finite_sample_noise |
| 20260512 | ALL | 0.364533 | 0.365276 | 0.346946 | 0.385819 | compatible_with_finite_sample_noise |

## Sampler Refresh

| fit_variant | seed | max_rhat | n_rhat_gt_1p01 | n_rhat_gt_1p05 | min_ess_bulk | min_ess_tail | median_ess_bulk | median_ess_tail | n_divergences | divergence_rate | max_tree_depth_hits | max_tree_depth_hit_rate | mean_acceptance_rate | runtime_seconds | run_dir |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| targeted_strong_lower_override | 20260511 | 1.01 | 0 | 0 | 2679 | 1569 | 7631 | 2784 | 0 | 0 | 0 | 0 | 0.948107 | 2210.09 | runs/targeted_strong_lower_override/seed_20260511 |
| targeted_strong_lower_override | 20260512 | 1.05 | 57 | 0 | 177 | 56 | 992 | 267 | 0 | 0 | 0 | 0 | 0.973089 | 900.704 | runs/targeted_strong_lower_override/seed_20260512 |

Sampler comparison against run summaries and top-level summaries:

| seed | metric | refreshed_from_fit_nc | run_summary_json | top_level_summary | abs_refreshed_minus_run_summary | abs_refreshed_minus_top_level |
| --- | --- | --- | --- | --- | --- | --- |
| 20260511 | max_rhat | 1.01 | 1.01 | 1.04 | 0 | 0.03 |
| 20260511 | n_rhat_gt_1p01 | 0 | 0 | 38 | 0 | 38 |
| 20260511 | n_rhat_gt_1p05 | 0 | 0 | 0 | 0 | 0 |
| 20260511 | min_ess_bulk | 2679 | 2679 | 247 | 0 | 2432 |
| 20260511 | min_ess_tail | 1569 | 1569 | 80 | 0 | 1489 |
| 20260511 | median_ess_bulk | 7631 | 7631 | 761 | 0 | 6870 |
| 20260511 | median_ess_tail | 2784 | 2784 | 265 | 0 | 2519 |
| 20260511 | n_divergences | 0 | 0 | 0 | 0 | 0 |
| 20260511 | max_tree_depth_hit_rate | 0 | 0 | 0 | 0 | 0 |
| 20260511 | runtime_seconds | 2210.09 | 2210.09 | 1044.33 | 0 | 1165.76 |
| 20260512 | max_rhat | 1.05 | 1.05 | 1.05 | 0 | 0 |
| 20260512 | n_rhat_gt_1p01 | 57 | 57 | 57 | 0 | 0 |
| 20260512 | n_rhat_gt_1p05 | 0 | 0 | 0 | 0 | 0 |
| 20260512 | min_ess_bulk | 177 | 177 | 177 | 0 | 0 |
| 20260512 | min_ess_tail | 56 | 56 | 56 | 0 | 0 |
| 20260512 | median_ess_bulk | 992 | 992 | 992 | 0 | 0 |
| 20260512 | median_ess_tail | 267 | 267 | 267 | 0 | 0 |
| 20260512 | n_divergences | 0 | 0 | 0 | 0 | 0 |
| 20260512 | max_tree_depth_hit_rate | 0 | 0 | 0 | 0 | 0 |
| 20260512 | runtime_seconds | 900.704 | 900.704 | 900.704 | 0 | 0 |

## Failed Refreshed Gates

| fit_variant | validation_scope | pillar | metric | value | threshold | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| targeted_strong_lower_override | smoke | posterior_predictive | weighted_mean_cell_TV | 0.393942 | <= 0.25 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | median_cell_TV | 0.373152 | <= 0.25 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | q90_cell_TV | 0.837273 | <= 0.50 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | fraction_cell_TV_gt_0p5 | 0.30402 | <= 0.10 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | posterior_predictive_tv_p_value | 1 | in [0.05, 0.95] | fail |  |
| targeted_strong_lower_override | smoke | sampler_health | fit_pass_rate_rhat | 0.5 | >= 0.90 | fail |  |
| targeted_strong_lower_override | smoke | sampler_health | fit_pass_rate_ess_bulk | 0.5 | >= 0.90 | fail |  |
| targeted_strong_lower_override | smoke | sampler_health | fit_pass_rate_ess_tail | 0.5 | >= 0.90 | fail |  |

## Top-Level Discrepancies

| table | fit_variant | seed | system | metric | refreshed | top_level | abs_diff | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | Chicken | rho_collapsed | 0.985999 | 0.986858 | 0.000859128 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | Chicken | rho_sampled_pi | 0.985856 | 0.986432 | 0.000576122 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | Chicken | abs_rho_collapsed_minus_sampled_pi | 0.000142602 | 0.000425608 | 0.000283006 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | Chicken | log_B_draw_mean | 7.03743 | 7.06219 | 0.0247598 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | Chicken | log_B_draw_median | 6.92853 | 6.9353 | 0.00677169 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | Chicken | log_B_draw_q05 | 4.41682 | 4.30803 | 0.108784 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | Chicken | log_B_draw_q95 | 10.1247 | 9.72056 | 0.404147 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | Chicken | log_B_eff | 5.86393 | 5.92813 | 0.0641942 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | Chicken | brier | 0.00019604 | 0.00017272 | 2.33199e-05 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | Chicken | log_score | -0.0141004 | -0.0132294 | 0.000870948 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | Chicken | brier_improvement_vs_prior | 0.694248 | 0.694272 | 2.33199e-05 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | Chicken | log_score_improvement_vs_prior | 1.77766 | 1.77853 | 0.000870948 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | LLMs | rho_collapsed | 0.000788234 | 0.000649487 | 0.000138747 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | LLMs | rho_sampled_pi | 0.000748635 | 0.0006738 | 7.48355e-05 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | LLMs | abs_rho_collapsed_minus_sampled_pi | 3.95989e-05 | 2.43126e-05 | 1.52863e-05 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | LLMs | log_B_draw_mean | -6.80489 | -6.81465 | 0.00976323 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | LLMs | log_B_draw_median | -6.69513 | -6.67847 | 0.0166587 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | LLMs | log_B_draw_q05 | -9.78664 | -9.64716 | 0.139485 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | LLMs | log_B_draw_q95 | -4.09471 | -4.20883 | 0.114116 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | LLMs | log_B_eff | -5.53549 | -5.72924 | 0.193751 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | LLMs | brier | 6.21313e-07 | 4.21833e-07 | 1.9948e-07 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | LLMs | log_score | -0.000788545 | -0.000649698 | 0.000138847 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | LLMs | brier_improvement_vs_prior | 0.0277772 | 0.0277774 | 1.9948e-07 | diff |
| validation_cases | targeted_strong_lower_override | 2.02605e+07 | LLMs | log_score_improvement_vs_prior | 0.181533 | 0.181672 | 0.000138847 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | ALL | mean_rho_R1 | 0.509804 | 0.510233 | 0.000429564 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | ALL | mean_rho_R0 | 0.00214514 | 0.00207576 | 6.93735e-05 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | ALL | median_rho_R1 | 0.509804 | 0.510233 | 0.000429564 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | ALL | median_rho_R0 | 0.00214514 | 0.00207576 | 6.93735e-05 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | ALL | median_log_B_eff_R1 | 2.0573 | 2.08939 | 0.0320971 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | ALL | median_log_B_eff_R0 | -4.78848 | -4.88535 | 0.0968755 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | ALL | evidence_margin_M | 0.447859 | 0.479956 | 0.0320971 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | ALL | brier_model | 0.23353 | 0.233524 | 5.87985e-06 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | ALL | brier_improvement | 0.127581 | 0.127587 | 5.87985e-06 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | ALL | log_score_model | -0.85284 | -0.852588 | 0.000252449 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | ALL | log_score_improvement | 0.1342 | 0.134453 | 0.000252449 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | ALL | ECE | 0.244026 | 0.243845 | 0.000180095 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | ALL | calibration_bins | [{"lo": 0.0, "hi": 0.1, "n": 3, "mean_rho": 0.01263305361874046, "empirical_rate": 0.3333333333333333}, {"lo": 0.9, "hi": 1.0, "n": 1, "mean_rho": 0.985998572588115, "empirical_rate": 1.0}] | [{"lo": 0.0, "hi": 0.1, "n": 3, "mean_rho": 0.0125868046135501, "empirical_rate": 0.3333333333333333}, {"lo": 0.9, "hi": 1.0, "n": 1, "mean_rho": 0.986857700329246, "empirical_rate": 1.0}] | NA | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | ALL | mean_abs_rho_collapsed_minus_sampled_pi | 0.00179575 | 0.00186268 | 6.69299e-05 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | Chicken | mean_rho_R1 | 0.985999 | 0.986858 | 0.000859128 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | Chicken | median_rho_R1 | 0.985999 | 0.986858 | 0.000859128 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | Chicken | median_log_B_eff_R1 | 5.86393 | 5.92813 | 0.0641942 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | Chicken | brier_model | 0.000104152 | 9.24922e-05 | 1.166e-05 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | Chicken | brier_improvement | 0.361007 | 0.361019 | 1.166e-05 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | Chicken | log_score_model | -0.00880428 | -0.0083688 | 0.000435474 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | Chicken | log_score_improvement | 0.978236 | 0.978672 | 0.000435474 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | Chicken | ECE | 0.00875173 | 0.00832217 | 0.000429564 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | Chicken | calibration_bins | [{"lo": 0.0, "hi": 0.1, "n": 1, "mean_rho": 0.0035020373865153488, "empirical_rate": 0.0}, {"lo": 0.9, "hi": 1.0, "n": 1, "mean_rho": 0.985998572588115, "empirical_rate": 1.0}] | [{"lo": 0.0, "hi": 0.1, "n": 1, "mean_rho": 0.0035020373865153, "empirical_rate": 0.0}, {"lo": 0.9, "hi": 1.0, "n": 1, "mean_rho": 0.986857700329246, "empirical_rate": 1.0}] | NA | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | Chicken | mean_abs_rho_collapsed_minus_sampled_pi | 0.000311525 | 0.000453028 | 0.000141503 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | LLMs | mean_rho_R0 | 0.000788234 | 0.000649487 | 0.000138747 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | LLMs | median_rho_R0 | 0.000788234 | 0.000649487 | 0.000138747 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | LLMs | median_log_B_eff_R0 | -5.53549 | -5.72924 | 0.193751 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | LLMs | brier_model | 0.466956 | 0.466956 | 9.97398e-08 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | LLMs | brier_improvement | -0.105845 | -0.105845 | 9.97398e-08 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | LLMs | log_score_model | -1.69688 | -1.69681 | 6.94234e-05 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | LLMs | log_score_improvement | -0.709836 | -0.709767 | 6.94234e-05 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | LLMs | ECE | 0.482801 | 0.482871 | 6.93735e-05 | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | LLMs | calibration_bins | [{"lo": 0.0, "hi": 0.1, "n": 2, "mean_rho": 0.017198561734853016, "empirical_rate": 0.5}] | [{"lo": 0.0, "hi": 0.1, "n": 2, "mean_rho": 0.0171291882270675, "empirical_rate": 0.5}] | NA | diff |
| rho_recovery_summary | targeted_strong_lower_override | NA | LLMs | mean_abs_rho_collapsed_minus_sampled_pi | 0.00327997 | 0.00327233 | 7.64315e-06 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | ALL | mean_RPS | 0.0720782 | 0.0721754 | 9.72318e-05 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | ALL | median_RPS | 0.0380735 | 0.0379302 | 0.000143267 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | ALL | q90_RPS | 0.195095 | 0.194768 | 0.000326621 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | ALL | mean_RPS_improvement | 0.158824 | 0.158727 | 9.72318e-05 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | ALL | weighted_mean_cell_TV | 0.393942 | 0.394283 | 0.000341492 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | ALL | median_cell_TV | 0.373152 | 0.375778 | 0.00262658 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | ALL | q90_cell_TV | 0.837273 | 0.83702 | 0.000252199 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | ALL | max_cell_TV | 0.941326 | 0.940479 | 0.000846342 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | ALL | fraction_cell_TV_gt_0p5 | 0.30402 | 0.306533 | 0.00251256 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | ALL | mean_abs_top7_error_high_q | 0.323843 | 0.324437 | 0.000593954 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | ALL | q90_abs_top7_error_high_q | 0.468365 | 0.46752 | 0.000844212 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | ALL | mean_abs_high_error_high_q | 0.1995 | 0.201102 | 0.00160193 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | ALL | q90_abs_high_error_high_q | 0.312854 | 0.311912 | 0.000942941 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | ALL | ppc_draws | 300 | 100 | 200 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | Chicken | mean_RPS | 0.0616851 | 0.0619204 | 0.000235379 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | Chicken | median_RPS | 0.028643 | 0.0290708 | 0.000427773 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | Chicken | q90_RPS | 0.134954 | 0.134408 | 0.000546491 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | Chicken | mean_RPS_improvement | 0.167232 | 0.166997 | 0.000235379 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | Chicken | weighted_mean_cell_TV | 0.401697 | 0.402682 | 0.00098495 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | Chicken | median_cell_TV | 0.368909 | 0.3717 | 0.00279149 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | Chicken | q90_cell_TV | 0.706533 | 0.706871 | 0.000337854 | diff |
| posterior_predictive_summary | targeted_strong_lower_override | NA | Chicken | max_cell_TV | 0.853268 | 0.85274 | 0.000528247 | diff |

## Answers

1. Are top-level smoke summaries stale or inconsistent? Yes. Refreshed root/sampler/LOO values disagree with top-level summaries, especially seed 20260511 sampler fields and seed 20260511 root rho/log_B values. PPC summaries mostly track per-run PPC artefacts but still need metadata caveats.
2. After refresh, does root recovery still fail for LLMs? Yes. LLM balanced accuracy remains 0.5 with negative Brier/log-score improvement.
3. Is the failed LLM case also a no-HMC/oracle failure, or only an HMC posterior failure? It is also an oracle/clamped failure: the exact oracle log_B for seed 20260512 LLMs is negative.
4. Are sampler diagnostics acceptable enough to interpret smoke metrics? No. Seed 20260512 has max_rhat 1.05, 57 Rhat values > 1.01, min bulk ESS 177, and min tail ESS 56. These are diagnostic only.
5. Is PPC failure likely implementation/sparsity/small-smoke noise, or a substantive observation-layer issue? Legacy TV gates are dominated by sparse system x indicator cells (`n_cells_ge_5=0` for every system). Existing oracle-calibrated PPC and predictive-alignment audits reduce concern for a broad observation-layer implementation break, though ELIZA seed 20260511 remains a localized weighted-TV excess signal.
6. Recommended next action: inspect the LLM failed seed and keep pilot blocked until summaries are regenerated consistently. If continuing smoke validation, rerun a longer smoke or pilot only after deciding whether seed 20260512's oracle-negative LLM draw is acceptable smoke noise and after sampler/summary consistency is fixed.

## Confirmation

No HMC, refitting, new posterior samples, new posterior predictive samples, pilot/full run, metrics-only command, plots-only command, or LOO recomputation was launched. This script read existing `fit.nc`, `posterior_predictive.nc`, per-run CSVs, JSON payloads, and top-level CSVs only.
