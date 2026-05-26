# Smoke Pipeline Refresh Report

Refresh output: `/Users/ryankelly/ryan-code/research/dcm-code/outputs/main_synthetic_validation_pilot_20260511/refreshed_summaries`

## Answers

- Previous top-level summaries stale: `False`.
- Refreshed summaries internally consistent: `True`.
- LLM R=1 smoke failure remains oracle-negative: `false_or_not_available`.
- PPC TV failures are interpreted with smoke finite-sample calibration; see `legacy_tv_interpretation` in the PPC summary.
- Another HMC smoke-plus is recommended before pilot if the team wants sampler evidence, but this refresh did not launch it.

## Root Recovery

| fit_variant | system | n_cases | mean_rho_R1 | mean_rho_R0 | median_rho_R1 | median_rho_R0 | median_log_B_eff_R1 | median_log_B_eff_R0 | evidence_margin_M | TPR_at_rho_gt_0p5 | TNR_at_rho_le_0p5 | balanced_accuracy | decisive_present_rate_R1_at_rho_gt_0p95 | decisive_absent_rate_R0_at_rho_lt_0p05 | brier_model | brier_prior_baseline | brier_improvement | log_score_model | log_score_prior_baseline | log_score_improvement | ECE | calibration_bins | mean_abs_rho_collapsed_minus_sampled_pi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | ALL | 20 | 0.6006 | 0.01054 | 0.7406 | 0.004074 | 3.087 | -3.9 | 1.478 | 0.6 | 1 | 0.8 | 0.4 | 0.9 | 0.1585 | 0.3611 | 0.2027 | -0.5211 | -0.987 | 0.466 | 0.1944 | [{"lo": 0.0, "hi": 0.1, "n": 13, "mean_rho": 0.021107048954298734, "empirical_rate": 0.23076923076923078}, {"lo": 0.4, "hi": 0.5, "n": 1, "mean_rho": 0.4778223304489019, "empirical_rate": 1.0}, {"lo": 0.5, "hi": 0.6000000000000001, "n": 1, "mean_rho": 0.5383601013574584, "empirical_rate": 1.0}, {"lo": 0.9, "hi": 1.0, "n": 5, "mean_rho": 0.9642244426067187, "empirical_rate": 1.0}] | 0.0005164 |
| targeted_strong_lower_override | Chicken | 10 | 0.866 | 0.01908 | 0.9528 | 0.01557 | 4.615 | -2.538 | 1.203 | 0.8 | 1 | 0.9 | 0.6 | 0.8 | 0.02826 | 0.3611 | 0.3329 | -0.09871 | -0.987 | 0.8883 | 0.07656 | [{"lo": 0.0, "hi": 0.1, "n": 5, "mean_rho": 0.019079644714673626, "empirical_rate": 0.0}, {"lo": 0.4, "hi": 0.5, "n": 1, "mean_rho": 0.4778223304489019, "empirical_rate": 1.0}, {"lo": 0.9, "hi": 1.0, "n": 4, "mean_rho": 0.9629921195857705, "empirical_rate": 1.0}] | 0.0005797 |
| targeted_strong_lower_override | LLMs | 10 | 0.3353 | 0.001998 | 0.07887 | 0.001402 | -0.8484 | -4.959 | -2.458 | 0.4 | 1 | 0.7 | 0.2 | 1 | 0.2887 | 0.3611 | 0.07246 | -0.9434 | -0.987 | 0.04361 | 0.3313 | [{"lo": 0.0, "hi": 0.1, "n": 8, "mean_rho": 0.022374176604064426, "empirical_rate": 0.375}, {"lo": 0.5, "hi": 0.6000000000000001, "n": 1, "mean_rho": 0.5383601013574584, "empirical_rate": 1.0}, {"lo": 0.9, "hi": 1.0, "n": 1, "mean_rho": 0.9691537346905117, "empirical_rate": 1.0}] | 0.000453 |

## Oracle Root Bridge

| fit_variant | seed | system | system_raw | root_z_true | log_B_oracle | rho_oracle_collapsed | log_B_eff_HMC | log_B_draw_median_HMC | rho_collapsed_HMC | rho_collapsed_median_HMC | delta_log_B_eff_minus_oracle | posterior_correct_sign | oracle_correct_sign | hmc_only_sign_flip | posterior_decisive | oracle_decisive | run_dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | 20260511 | Chicken | Chicken | 1 | 4.958 | 0.9661 | 5.864 | 6.929 | 0.986 | 0.9951 | 0.9056 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260511 |
| targeted_strong_lower_override | 20260511 | LLMs | 2024 Leading Chat LLMs | 0 | -4.432 | 0.002374 | -5.535 | -6.695 | 0.0007882 | 0.0002473 | -1.104 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260511 |
| targeted_strong_lower_override | 20260512 | Chicken | Chicken | 0 | -3.864 | 0.004181 | -4.047 | -4.799 | 0.003482 | 0.001645 | -0.1836 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260512 |
| targeted_strong_lower_override | 20260512 | LLMs | 2024 Leading Chat LLMs | 1 | -1.568 | 0.04001 | -1.656 | -2.292 | 0.03677 | 0.01982 | -0.08806 | False | False | False | False | False | runs/targeted_strong_lower_override/seed_20260512 |
| targeted_strong_lower_override | 20260513 | Chicken | Chicken | 1 | 1.383 | 0.4436 | 1.521 | 1.493 | 0.4778 | 0.4708 | 0.1379 | True | True | False | False | False | runs/targeted_strong_lower_override/seed_20260513 |
| targeted_strong_lower_override | 20260513 | LLMs | 2024 Leading Chat LLMs | 0 | -5.117 | 0.001197 | -4.959 | -6.145 | 0.001402 | 0.0004286 | 0.1582 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260513 |
| targeted_strong_lower_override | 20260514 | Chicken | Chicken | 0 | -1.141 | 0.06004 | -1.294 | -1.815 | 0.05199 | 0.03154 | -0.1524 | True | True | False | False | False | runs/targeted_strong_lower_override/seed_20260514 |
| targeted_strong_lower_override | 20260514 | LLMs | 2024 Leading Chat LLMs | 1 | -1.09 | 0.063 | -0.8484 | -1.308 | 0.07887 | 0.05131 | 0.2417 | False | False | False | False | False | runs/targeted_strong_lower_override/seed_20260514 |
| targeted_strong_lower_override | 20260515 | Chicken | Chicken | 1 | 4.67 | 0.9552 | 5.098 | 5.908 | 0.9704 | 0.9866 | 0.4284 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260515 |
| targeted_strong_lower_override | 20260515 | LLMs | 2024 Leading Chat LLMs | 0 | -5.582 | 0.0007526 | -5.938 | -7.138 | 0.0005273 | 0.0001588 | -0.3559 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260515 |
| targeted_strong_lower_override | 20260516 | Chicken | Chicken | 0 | -2.582 | 0.01489 | -2.538 | -3.425 | 0.01557 | 0.006465 | 0.04476 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260516 |
| targeted_strong_lower_override | 20260516 | LLMs | 2024 Leading Chat LLMs | 1 | 5.05 | 0.9689 | 5.057 | 5.897 | 0.9692 | 0.9864 | 0.007043 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260516 |
| targeted_strong_lower_override | 20260517 | Chicken | Chicken | 1 | 4.668 | 0.9552 | 4.615 | 5.154 | 0.9528 | 0.9719 | -0.05307 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260517 |
| targeted_strong_lower_override | 20260517 | LLMs | 2024 Leading Chat LLMs | 0 | -3.69 | 0.004971 | -4.338 | -5.116 | 0.002607 | 0.001199 | -0.648 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260517 |
| targeted_strong_lower_override | 20260518 | Chicken | Chicken | 0 | -3.383 | 0.006742 | -3.449 | -4.107 | 0.006316 | 0.00328 | -0.06557 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260518 |
| targeted_strong_lower_override | 20260518 | LLMs | 2024 Leading Chat LLMs | 1 | 2.342 | 0.6753 | 1.763 | 1.784 | 0.5384 | 0.5435 | -0.5786 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260518 |
| targeted_strong_lower_override | 20260519 | Chicken | Chicken | 1 | 4.334 | 0.9384 | 4.411 | 5.198 | 0.9428 | 0.9731 | 0.07754 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260519 |
| targeted_strong_lower_override | 20260519 | LLMs | 2024 Leading Chat LLMs | 0 | -3.249 | 0.007704 | -3.753 | -4.275 | 0.004667 | 0.002774 | -0.5043 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260519 |
| targeted_strong_lower_override | 20260520 | Chicken | Chicken | 0 | -2.488 | 0.01634 | -2.387 | -2.916 | 0.01804 | 0.01071 | 0.1008 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260520 |
| targeted_strong_lower_override | 20260520 | LLMs | 2024 Leading Chat LLMs | 1 | -1.26 | 0.05367 | -1.266 | -1.678 | 0.05337 | 0.03601 | -0.006008 | False | False | False | False | False | runs/targeted_strong_lower_override/seed_20260520 |

## Oracle Root Bridge Summary

| fit_variant | system | n_cases | n_R1_cases | n_oracle_correct | n_posterior_correct | n_hmc_only_sign_flips | n_R1_oracle_negative | mean_delta_log_B_eff_minus_oracle | max_abs_delta_log_B_eff_minus_oracle |
|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | Chicken | 10 | 5 | 10 | 10 | 0 | 0 | 0.124 | 0.9056 |
| targeted_strong_lower_override | LLMs | 10 | 5 | 7 | 7 | 0 | 3 | -0.2878 | 1.104 |

## PPC Summary

| fit_variant | system | n_ratings | mean_RPS | median_RPS | q90_RPS | mean_RPS_baseline_empirical_marginal | mean_RPS_improvement | weighted_mean_cell_TV | median_cell_TV | q90_cell_TV | max_cell_TV | fraction_cell_TV_gt_0p3 | fraction_cell_TV_gt_0p5 | n_cells | n_cells_ge_5 | mean_abs_top7_error_high_q | q90_abs_top7_error_high_q | mean_abs_high_error_high_q | q90_abs_high_error_high_q | posterior_predictive_tv_p_value | ppc_draws | n_seeds | seed_min | seed_max | seed_list | posterior_predictive_tv_p_value_status | legacy_tv_interpretation | oracle_weighted_tv_statuses | predictive_alignment_statuses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | ALL | 379 | 0.06713 | 0.03168 | 0.1696 | 0.2264 | 0.1592 | 0.3876 | 0.376 | 0.8281 | 0.9359 | 0.6523 | 0.3221 | 199 | 0 | 0.3404 | 0.5385 | 0.2084 | 0.362 | 0.9952 | 500 | 10 | 20260511 | 20260520 | 20260511;20260512;20260513;20260514;20260515;20260516;20260517;20260518;20260519;20260520 | computed_overall | legacy_tv_fail_oracle_calibrated_warning |  |  |
| targeted_strong_lower_override | Chicken | 93 | 0.06561 | 0.03407 | 0.1498 | 0.2343 | 0.1687 | 0.4082 | 0.3695 | 0.7279 | 0.8576 | 0.6714 | 0.2571 | 49 | 0 | 0.2664 | 0.44 | 0.1853 | 0.3039 |  | 500 | 10 | 20260511 | 20260520 | 20260511;20260512;20260513;20260514;20260515;20260516;20260517;20260518;20260519;20260520 | not_computed_system_specific | legacy_tv_fail_oracle_calibrated_warning |  |  |
| targeted_strong_lower_override | ELIZA | 50 | 0.05431 | 0.03368 | 0.1347 | 0.2329 | 0.1786 | 0.4926 | 0.3968 | 0.8666 | 0.9298 | 0.634 | 0.404 | 50 | 0 | 0.433 | 0.544 | 0.2486 | 0.3959 |  | 500 | 10 | 20260511 | 20260520 | 20260511;20260512;20260513;20260514;20260515;20260516;20260517;20260518;20260519;20260520 | not_computed_system_specific | legacy_tv_fail_oracle_calibrated_warning |  |  |
| targeted_strong_lower_override | Human | 50 | 0.06224 | 0.04101 | 0.1211 | 0.2323 | 0.17 | 0.5716 | 0.5159 | 0.8684 | 0.929 | 0.904 | 0.478 | 50 | 0 | 0.4197 | 0.5418 | 0.2435 | 0.5125 |  | 500 | 10 | 20260511 | 20260520 | 20260511;20260512;20260513;20260514;20260515;20260516;20260517;20260518;20260519;20260520 | not_computed_system_specific | legacy_tv_fail_oracle_calibrated_warning |  |  |
| targeted_strong_lower_override | LLMs | 186 | 0.07264 | 0.02758 | 0.2019 | 0.219 | 0.1464 | 0.2996 | 0.2541 | 0.5284 | 0.6708 | 0.4 | 0.148 | 50 | 0 | 0.1754 | 0.3205 | 0.1446 | 0.1804 |  | 500 | 10 | 20260511 | 20260520 | 20260511;20260512;20260513;20260514;20260515;20260516;20260517;20260518;20260519;20260520 | not_computed_system_specific | legacy_tv_fail_oracle_calibrated_warning |  |  |

PPC row metadata status: metadata sidecars present for 10 runs; missing for 0 runs

## Sampler Summary

| fit_variant | seed | max_rhat | n_rhat_gt_1p01 | n_rhat_gt_1p05 | min_ess_bulk | min_ess_tail | median_ess_bulk | median_ess_tail | n_divergences | divergence_rate | max_tree_depth_hits | max_tree_depth_hit_rate | mean_acceptance_rate | runtime_seconds | run_dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | 20260511 | 1.01 | 0 | 0 | 2679 | 1569 | 7631 | 2784 | 0 | 0 | 0 | 0 | 0.9481 | 2219 | runs/targeted_strong_lower_override/seed_20260511 |
| targeted_strong_lower_override | 20260512 | 1.01 | 0 | 0 | 2105 | 882 | 5473 | 2823 | 0 | 0 | 0 | 0 | 0.947 | 1943 | runs/targeted_strong_lower_override/seed_20260512 |
| targeted_strong_lower_override | 20260513 | 1 | 0 | 0 | 2621 | 1494 | 5170 | 2870 | 0 | 0 | 0 | 0 | 0.9477 | 1982 | runs/targeted_strong_lower_override/seed_20260513 |
| targeted_strong_lower_override | 20260514 | 1.01 | 0 | 0 | 1904 | 1509 | 5113 | 2865 | 0 | 0 | 0 | 0 | 0.9457 | 2078 | runs/targeted_strong_lower_override/seed_20260514 |
| targeted_strong_lower_override | 20260515 | 1 | 0 | 0 | 2421 | 1284 | 6040 | 2848 | 0 | 0 | 0 | 0 | 0.9417 | 2439 | runs/targeted_strong_lower_override/seed_20260515 |
| targeted_strong_lower_override | 20260516 | 1.01 | 0 | 0 | 3208 | 1515 | 8436 | 2738 | 0 | 0 | 0 | 0 | 0.9423 | 2523 | runs/targeted_strong_lower_override/seed_20260516 |
| targeted_strong_lower_override | 20260517 | 1.01 | 0 | 0 | 2889 | 1596 | 5456 | 2812 | 0 | 0 | 0 | 0 | 0.9451 | 2239 | runs/targeted_strong_lower_override/seed_20260517 |
| targeted_strong_lower_override | 20260518 | 1.01 | 0 | 0 | 2962 | 1875 | 9482 | 2710 | 0 | 0 | 0 | 0 | 0.9448 | 2418 | runs/targeted_strong_lower_override/seed_20260518 |
| targeted_strong_lower_override | 20260519 | 1.01 | 0 | 0 | 2763 | 1587 | 9900 | 2734 | 0 | 0 | 0 | 0 | 0.9452 | 2348 | runs/targeted_strong_lower_override/seed_20260519 |
| targeted_strong_lower_override | 20260520 | 1 | 0 | 0 | 2211 | 1327 | 4786 | 2984 | 0 | 0 | 0 | 0 | 0.945 | 1646 | runs/targeted_strong_lower_override/seed_20260520 |

## Failed Gates

| fit_variant | validation_scope | pillar | metric | value | threshold | status | notes |
|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | smoke | posterior_predictive | weighted_mean_cell_TV | 0.3876 | <= 0.25 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | median_cell_TV | 0.376 | <= 0.25 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | q90_cell_TV | 0.8281 | <= 0.50 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | fraction_cell_TV_gt_0p5 | 0.3221 | <= 0.10 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | posterior_predictive_tv_p_value | 0.9952 | in [0.05, 0.95] | fail |  |

## Manifest Consistency

```json
{
  "fit_nc_rho_matches_per_run_rho_cases": true,
  "fit_nc_sampler_matches_run_summary_json": true,
  "loo_refreshed_from_existing_per_run_summaries": true,
  "oracle_root_bridge_cases_written": true,
  "ppc_refreshed_from_existing_per_run_summaries": true,
  "rho_mismatch_examples": [],
  "sampler_mismatch_examples": []
}
```

## Confirmation

No HMC, refitting, new posterior sampling, new posterior predictive sampling, pilot/full run, or LOO recomputation was launched. The refresh read existing run artefacts only.
