# Smoke Pipeline Refresh Report

Refresh output: `/Users/ryankelly/ryan-code/research/dcm-code/outputs/main_synthetic_validation_smoke_postpatch_20260511/refreshed_summaries`

## Answers

- Previous top-level summaries stale: `False`.
- Refreshed summaries internally consistent: `True`.
- LLM R=1 smoke failure remains oracle-negative: `false_or_not_available`.
- PPC TV failures are interpreted with smoke finite-sample calibration; see `legacy_tv_interpretation` in the PPC summary.
- Another HMC smoke-plus is recommended before pilot if the team wants sampler evidence, but this refresh did not launch it.

## Root Recovery

| fit_variant | system | n_cases | mean_rho_R1 | mean_rho_R0 | median_rho_R1 | median_rho_R0 | median_log_B_eff_R1 | median_log_B_eff_R0 | evidence_margin_M | TPR_at_rho_gt_0p5 | TNR_at_rho_le_0p5 | balanced_accuracy | decisive_present_rate_R1_at_rho_gt_0p95 | decisive_absent_rate_R0_at_rho_lt_0p05 | brier_model | brier_prior_baseline | brier_improvement | log_score_model | log_score_prior_baseline | log_score_improvement | ECE | calibration_bins | mean_abs_rho_collapsed_minus_sampled_pi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | ALL | 2 | 0.9869 | 0.0006495 | 0.9869 | 0.0006495 | 5.928 | -5.729 | 4.319 | 1 | 1 | 1 | 1 | 1 | 8.657e-05 | 0.3611 | 0.361 | -0.00694 | -0.987 | 0.9801 | 0.006896 | [{"lo": 0.0, "hi": 0.1, "n": 1, "mean_rho": 0.0006494870405678088, "empirical_rate": 0.0}, {"lo": 0.9, "hi": 1.0, "n": 1, "mean_rho": 0.9868577003292461, "empirical_rate": 1.0}] | 0.000225 |
| targeted_strong_lower_override | Chicken | 1 | 0.9869 |  | 0.9869 |  | 5.928 |  |  | 1 |  | 1 | 1 |  | 8.636e-05 | 0.3611 | 0.361 | -0.006615 | -0.987 | 0.9804 | 0.01314 | [{"lo": 0.9, "hi": 1.0, "n": 1, "mean_rho": 0.9868577003292461, "empirical_rate": 1.0}] | 0.0004256 |
| targeted_strong_lower_override | LLMs | 1 |  | 0.0006495 |  | 0.0006495 |  | -5.729 |  |  | 1 | 1 |  | 1 | 2.109e-07 | 0.3611 | 0.3611 | -0.0003248 | -0.987 | 0.9867 | 0.0006495 | [{"lo": 0.0, "hi": 0.1, "n": 1, "mean_rho": 0.0006494870405678088, "empirical_rate": 0.0}] | 2.431e-05 |

## Oracle Root Bridge

| fit_variant | seed | system | system_raw | root_z_true | log_B_oracle | rho_oracle_collapsed | log_B_eff_HMC | log_B_draw_median_HMC | rho_collapsed_HMC | rho_collapsed_median_HMC | delta_log_B_eff_minus_oracle | posterior_correct_sign | oracle_correct_sign | hmc_only_sign_flip | posterior_decisive | oracle_decisive | run_dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | 20260511 | Chicken | Chicken | 1 | 4.958 | 0.9661 | 5.928 | 6.935 | 0.9869 | 0.9952 | 0.9698 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260511 |
| targeted_strong_lower_override | 20260511 | LLMs | 2024 Leading Chat LLMs | 0 | -4.432 | 0.002374 | -5.729 | -6.678 | 0.0006495 | 0.0002515 | -1.298 | True | True | False | True | True | runs/targeted_strong_lower_override/seed_20260511 |

## Oracle Root Bridge Summary

| fit_variant | system | n_cases | n_R1_cases | n_oracle_correct | n_posterior_correct | n_hmc_only_sign_flips | n_R1_oracle_negative | mean_delta_log_B_eff_minus_oracle | max_abs_delta_log_B_eff_minus_oracle |
|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | Chicken | 1 | 1 | 1 | 1 | 0 | 0 | 0.9698 | 0.9698 |
| targeted_strong_lower_override | LLMs | 1 | 0 | 1 | 1 | 0 | 0 | -1.298 | 1.298 |

## PPC Summary

| fit_variant | system | n_ratings | mean_RPS | median_RPS | q90_RPS | mean_RPS_baseline_empirical_marginal | mean_RPS_improvement | weighted_mean_cell_TV | median_cell_TV | q90_cell_TV | max_cell_TV | fraction_cell_TV_gt_0p3 | fraction_cell_TV_gt_0p5 | n_cells | n_cells_ge_5 | mean_abs_top7_error_high_q | q90_abs_top7_error_high_q | mean_abs_high_error_high_q | q90_abs_high_error_high_q | posterior_predictive_tv_p_value | ppc_draws | n_seeds | seed_min | seed_max | seed_list | posterior_predictive_tv_p_value_status | legacy_tv_interpretation | oracle_weighted_tv_statuses | predictive_alignment_statuses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | ALL | 379 | 0.07946 | 0.05548 | 0.204 | 0.2273 | 0.1478 | 0.424 | 0.4309 | 0.8388 | 0.9347 | 0.6884 | 0.3668 | 199 | 0 | 0.3464 | 0.4886 | 0.2244 | 0.3383 | 1 | 100 | 1 | 20260511 | 20260511 | 20260511 | computed_overall | legacy_tv_fail_oracle_calibrated_warning |  |  |
| targeted_strong_lower_override | Chicken | 93 | 0.06903 | 0.03908 | 0.1543 | 0.2308 | 0.1618 | 0.4457 | 0.4219 | 0.7504 | 0.8335 | 0.7551 | 0.2857 | 49 | 0 | 0.267 | 0.4223 | 0.1735 | 0.2911 |  | 100 | 1 | 20260511 | 20260511 | 20260511 | not_computed_system_specific | legacy_tv_fail_oracle_calibrated_warning |  |  |
| targeted_strong_lower_override | ELIZA | 50 | 0.0671 | 0.05549 | 0.151 | 0.2213 | 0.1542 | 0.5744 | 0.6167 | 0.9058 | 0.9347 | 0.64 | 0.5 | 50 | 0 | 0.4395 | 0.5107 | 0.301 | 0.562 |  | 100 | 1 | 20260511 | 20260511 | 20260511 | not_computed_system_specific | legacy_tv_fail_oracle_calibrated_warning |  |  |
| targeted_strong_lower_override | Human | 50 | 0.07062 | 0.05527 | 0.1329 | 0.2273 | 0.1567 | 0.6157 | 0.6767 | 0.9051 | 0.9237 | 0.94 | 0.52 | 50 | 0 | 0.44 | 0.5144 | 0.2671 | 0.6095 |  | 100 | 1 | 20260511 | 20260511 | 20260511 | not_computed_system_specific | legacy_tv_fail_oracle_calibrated_warning |  |  |
| targeted_strong_lower_override | LLMs | 186 | 0.09037 | 0.05987 | 0.2917 | 0.2271 | 0.1367 | 0.3212 | 0.2765 | 0.5522 | 0.7434 | 0.42 | 0.16 | 50 | 0 | 0.1455 | 0.2539 | 0.1535 | 0.1606 |  | 100 | 1 | 20260511 | 20260511 | 20260511 | not_computed_system_specific | legacy_tv_fail_oracle_calibrated_warning |  |  |

PPC row metadata status: metadata sidecars present for 1 runs; missing for 0 runs

## Sampler Summary

| fit_variant | seed | max_rhat | n_rhat_gt_1p01 | n_rhat_gt_1p05 | min_ess_bulk | min_ess_tail | median_ess_bulk | median_ess_tail | n_divergences | divergence_rate | max_tree_depth_hits | max_tree_depth_hit_rate | mean_acceptance_rate | runtime_seconds | run_dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | 20260511 | 1.04 | 38 | 0 | 247 | 80 | 761 | 265 | 0 | 0 | 0 | 0 | 0.9729 | 1518 | runs/targeted_strong_lower_override/seed_20260511 |

## Failed Gates

| fit_variant | validation_scope | pillar | metric | value | threshold | status | notes |
|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | smoke | posterior_predictive | weighted_mean_cell_TV | 0.424 | <= 0.25 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | median_cell_TV | 0.4309 | <= 0.25 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | q90_cell_TV | 0.8388 | <= 0.50 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | fraction_cell_TV_gt_0p5 | 0.3668 | <= 0.10 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | posterior_predictive_tv_p_value | 1 | in [0.05, 0.95] | fail |  |
| targeted_strong_lower_override | smoke | sampler_health | fit_pass_rate_rhat | 0 | >= 0.90 | fail |  |
| targeted_strong_lower_override | smoke | sampler_health | fit_pass_rate_ess_bulk | 0 | >= 0.90 | fail |  |
| targeted_strong_lower_override | smoke | sampler_health | fit_pass_rate_ess_tail | 0 | >= 0.90 | fail |  |

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
