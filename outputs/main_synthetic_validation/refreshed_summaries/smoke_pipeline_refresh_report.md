# Smoke Pipeline Refresh Report

Refresh output: `/Users/ryankelly/ryan-code/research/dcm-code/outputs/main_synthetic_validation/refreshed_summaries`

## Answers

- Previous top-level summaries stale: `True`.
- Refreshed summaries internally consistent: `True`.
- LLM R=1 smoke failure remains oracle-negative: `true`.
- PPC TV failures are interpreted with smoke finite-sample calibration; see `legacy_tv_interpretation` in the PPC summary.
- Another HMC smoke-plus is recommended before pilot if the team wants sampler evidence, but this refresh did not launch it.

## Root Recovery

| fit_variant | system | n_cases | mean_rho_R1 | mean_rho_R0 | median_rho_R1 | median_rho_R0 | median_log_B_eff_R1 | median_log_B_eff_R0 | evidence_margin_M | TPR_at_rho_gt_0p5 | TNR_at_rho_le_0p5 | balanced_accuracy | decisive_present_rate_R1_at_rho_gt_0p95 | decisive_absent_rate_R0_at_rho_lt_0p05 | brier_model | brier_prior_baseline | brier_improvement | log_score_model | log_score_prior_baseline | log_score_improvement | ECE | calibration_bins | mean_abs_rho_collapsed_minus_sampled_pi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | ALL | 4 | 0.5114 | 0.002135 | 0.5114 | 0.002135 | 2.104 | -4.791 | 0.4944 | 0.5 | 1 | 0.75 | 0.5 | 1 | 0.232 | 0.3611 | 0.1291 | -0.8304 | -0.987 | 0.1566 | 0.2432 | [{"lo": 0.0, "hi": 0.1, "n": 3, "mean_rho": 0.013678387685094833, "empirical_rate": 0.3333333333333333}, {"lo": 0.9, "hi": 1.0, "n": 1, "mean_rho": 0.985998572588115, "empirical_rate": 1.0}] | 0.0001487 |
| targeted_strong_lower_override | Chicken | 2 | 0.986 | 0.003482 | 0.986 | 0.003482 | 5.864 | -4.047 | 2.712 | 1 | 1 | 1 | 1 | 1 | 0.0001041 | 0.3611 | 0.361 | -0.008794 | -0.987 | 0.9782 | 0.008742 | [{"lo": 0.0, "hi": 0.1, "n": 1, "mean_rho": 0.003481888810227063, "empirical_rate": 0.0}, {"lo": 0.9, "hi": 1.0, "n": 1, "mean_rho": 0.985998572588115, "empirical_rate": 1.0}] | 0.0002039 |
| targeted_strong_lower_override | LLMs | 2 | 0.03677 | 0.0007882 | 0.03677 | 0.0007882 | -1.656 | -5.535 | -3.266 | 0 | 1 | 0.5 | 0 | 1 | 0.4639 | 0.3611 | -0.1028 | -1.652 | -0.987 | -0.665 | 0.4812 | [{"lo": 0.0, "hi": 0.1, "n": 2, "mean_rho": 0.01877663712252872, "empirical_rate": 0.5}] | 9.345e-05 |

## Oracle Root Bridge

| fit_variant | seed | system | system_raw | root_z_true | log_B_oracle | rho_oracle_collapsed | log_B_eff_HMC | log_B_draw_median_HMC | rho_collapsed_HMC | rho_collapsed_median_HMC | delta_log_B_eff_minus_oracle | posterior_correct_sign | oracle_correct_sign | hmc_only_sign_flip | posterior_decisive | oracle_decisive | run_dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | 20260511 | Chicken | Chicken | 1 | 4.958 | 0.9661 | 5.864 | 6.929 | 0.986 | 0.9951 | 0.9056 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260511 |
| targeted_strong_lower_override | 20260511 | LLMs | 2024 Leading Chat LLMs | 0 | -4.432 | 0.002374 | -5.535 | -6.695 | 0.0007882 | 0.0002473 | -1.104 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260511 |
| targeted_strong_lower_override | 20260512 | Chicken | Chicken | 0 | -3.864 | 0.004181 | -4.047 | -4.799 | 0.003482 | 0.001645 | -0.1836 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260512 |
| targeted_strong_lower_override | 20260512 | LLMs | 2024 Leading Chat LLMs | 1 | -1.568 | 0.04001 | -1.656 | -2.292 | 0.03677 | 0.01982 | -0.08806 | False | False | False | False | False | runs/targeted_strong_lower_override/seed_20260512 |

## Oracle Root Bridge Summary

| fit_variant | system | n_cases | n_R1_cases | n_oracle_correct | n_posterior_correct | n_hmc_only_sign_flips | n_R1_oracle_negative | mean_delta_log_B_eff_minus_oracle | max_abs_delta_log_B_eff_minus_oracle |
|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | Chicken | 2 | 1 | 2 | 2 | 0 | 0 | 0.361 | 0.9056 |
| targeted_strong_lower_override | LLMs | 2 | 1 | 1 | 1 | 0 | 1 | -0.596 | 1.104 |

## PPC Summary

| fit_variant | system | n_ratings | mean_RPS | median_RPS | q90_RPS | mean_RPS_baseline_empirical_marginal | mean_RPS_improvement | weighted_mean_cell_TV | median_cell_TV | q90_cell_TV | max_cell_TV | fraction_cell_TV_gt_0p3 | fraction_cell_TV_gt_0p5 | n_cells | n_cells_ge_5 | mean_abs_top7_error_high_q | q90_abs_top7_error_high_q | mean_abs_high_error_high_q | q90_abs_high_error_high_q | posterior_predictive_tv_p_value | ppc_draws | n_seeds | seed_min | seed_max | seed_list | posterior_predictive_tv_p_value_status | legacy_tv_interpretation | oracle_weighted_tv_statuses | predictive_alignment_statuses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | ALL | 379 | 0.072 | 0.03745 | 0.1951 | 0.2309 | 0.1589 | 0.3931 | 0.3676 | 0.8382 | 0.941 | 0.6281 | 0.304 | 199 | 0 | 0.3212 | 0.4709 | 0.1985 | 0.3166 | 0.996 | 500 | 2 | 20260511 | 20260512 | 20260511;20260512 | computed_overall | legacy_tv_fail_but_oracle_calibrated_compatible | compatible_with_finite_sample_noise | pass |
| targeted_strong_lower_override | Chicken | 93 | 0.06172 | 0.02797 | 0.1363 | 0.2289 | 0.1672 | 0.4018 | 0.3727 | 0.706 | 0.8537 | 0.6633 | 0.2245 | 49 | 0 | 0.2483 | 0.3577 | 0.1809 | 0.3155 |  | 500 | 2 | 20260511 | 20260512 | 20260511;20260512 | not_computed_system_specific | legacy_tv_fail_but_oracle_calibrated_compatible | compatible_with_finite_sample_noise | pass |
| targeted_strong_lower_override | ELIZA | 50 | 0.05937 | 0.03418 | 0.1489 | 0.2171 | 0.1578 | 0.5095 | 0.4159 | 0.8996 | 0.941 | 0.53 | 0.42 | 50 | 0 | 0.4343 | 0.5504 | 0.2987 | 0.5448 |  | 500 | 2 | 20260511 | 20260512 | 20260511;20260512 | not_computed_system_specific | legacy_tv_fail_oracle_calibrated_warning | compatible_with_finite_sample_noise;ppc_excess_misfit | pass |
| targeted_strong_lower_override | Human | 50 | 0.06091 | 0.0377 | 0.1267 | 0.2474 | 0.1865 | 0.5451 | 0.4941 | 0.8931 | 0.9348 | 0.93 | 0.42 | 50 | 0 | 0.3937 | 0.4944 | 0.2179 | 0.4182 |  | 500 | 2 | 20260511 | 20260512 | 20260511;20260512 | not_computed_system_specific | legacy_tv_fail_oracle_calibrated_warning | compatible_with_finite_sample_noise;ppc_underdispersed_or_metric_issue | pass |
| targeted_strong_lower_override | LLMs | 186 | 0.08352 | 0.03898 | 0.248 | 0.2312 | 0.1477 | 0.3166 | 0.2842 | 0.5663 | 0.7436 | 0.39 | 0.15 | 50 | 0 | 0.1722 | 0.2841 | 0.1369 | 0.1424 |  | 500 | 2 | 20260511 | 20260512 | 20260511;20260512 | not_computed_system_specific | legacy_tv_fail_but_oracle_calibrated_compatible | compatible_with_finite_sample_noise | pass |

PPC row metadata status: existing metadata sidecars missing; warning written

## Sampler Summary

| fit_variant | seed | max_rhat | n_rhat_gt_1p01 | n_rhat_gt_1p05 | min_ess_bulk | min_ess_tail | median_ess_bulk | median_ess_tail | n_divergences | divergence_rate | max_tree_depth_hits | max_tree_depth_hit_rate | mean_acceptance_rate | runtime_seconds | run_dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | 20260511 | 1.01 | 0 | 0 | 2679 | 1569 | 7631 | 2784 | 0 | 0 | 0 | 0 | 0.9481 | 2210 | runs/targeted_strong_lower_override/seed_20260511 |
| targeted_strong_lower_override | 20260512 | 1.01 | 0 | 0 | 2105 | 882 | 5473 | 2823 | 0 | 0 | 0 | 0 | 0.947 | 1977 | runs/targeted_strong_lower_override/seed_20260512 |

## Failed Gates

| fit_variant | validation_scope | pillar | metric | value | threshold | status | notes |
|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | smoke | posterior_predictive | weighted_mean_cell_TV | 0.3931 | <= 0.25 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | median_cell_TV | 0.3676 | <= 0.25 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | q90_cell_TV | 0.8382 | <= 0.50 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | fraction_cell_TV_gt_0p5 | 0.304 | <= 0.10 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | posterior_predictive_tv_p_value | 0.996 | in [0.05, 0.95] | fail |  |

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
