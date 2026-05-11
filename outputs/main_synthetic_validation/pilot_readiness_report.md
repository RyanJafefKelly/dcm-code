# Pilot Readiness Report

Decision: `NOT_READY_FIX_PPC`

## Reporting Identifiers

Identifier columns are no longer averaged in aggregate tables. Aggregate PPC/LOO tables use `n_seeds`, `seed_min`, `seed_max`, and `seed_list`.

All identifier checks passed.

## Targeted Override And Ladder Gates

Smoke targeted override sanity remains: 49 overridden lower edges, 0 root-to-top overrides, 0 weak-top overrides, and generator/fitter beta centre max diff 0.

| rung | rung_description | n_cases | n_replicates | median_log_B_eff_R1 | median_log_B_eff_R0 | mean_rho_R1 | mean_rho_R0 | evidence_margin_M | balanced_log_score_improvement | balanced_brier_improvement | TPR_at_rho_gt_0p5 | TNR_at_rho_le_0p5 | balanced_accuracy | pass_fail_label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L4 | actual GWT targeted strong-lower beta | 40 | 20 | 3.009 | -2.624 | 0.7194 | 0.09111 | 1.289 | 0.6465 | 0.2598 | 0.8 | 0.95 | 0.875 | pass |

## Oracle Root Evidence Audit

| fit_variant | seed | system | system_raw | root_z_true | log_B_oracle | rho_oracle_collapsed | log_B_eff_posterior | rho_collapsed_posterior | delta_eff_minus_oracle | posterior_correct_sign | oracle_correct_sign | posterior_decisive | oracle_decisive | run_dir |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | 20260511 | Chicken | Chicken | 1 | 4.958 | 0.9661 | 5.928 | 0.9869 | 0.9698 | True | True | True | True | runs/targeted_strong_lower_override/seed_20260511 |
| targeted_strong_lower_override | 20260511 | LLMs | 2024 Leading Chat LLMs | 0 | -4.432 | 0.002374 | -5.729 | 0.0006495 | -1.298 | True | True | True | True | runs/targeted_strong_lower_override/seed_20260511 |
| targeted_strong_lower_override | 20260512 | Chicken | Chicken | 0 | -3.864 | 0.004181 | -4.041 | 0.003502 | -0.1778 | True | True | True | True | runs/targeted_strong_lower_override/seed_20260512 |
| targeted_strong_lower_override | 20260512 | LLMs | 2024 Leading Chat LLMs | 1 | -1.568 | 0.04001 | -1.749 | 0.03361 | -0.1811 | False | False | False | False | runs/targeted_strong_lower_override/seed_20260512 |

Interpretation: if posterior and oracle fail the same realised positive-root case, the failure is finite-data/root-realisation weakness; if oracle is decisive and posterior has the wrong sign, nuisance inference or design is implicated.

## Oracle Finite-Sample PPC Baseline

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

The main PPC gates remain in the validation report; this section calibrates those gates against the sparse production-like layout where most system x indicator cells have fewer than five ratings.

## PPC Sanity Checks

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

PPC uses posterior draws from each fit. Oracle PPC uses truth parameters only and is kept separate from fitted PPC.

## Full-Default Rehearsal

| fit_variant | seed | runtime_seconds | max_rhat | n_rhat_gt_1p01 | n_rhat_gt_1p05 | min_ess_bulk | min_ess_tail | n_divergences | max_tree_depth_hits | mean_acceptance_rate | rho_chicken | rho_llms | log_B_eff_chicken | log_B_eff_llms | sampler_status | estimated_10_seed_pilot_wallclock_hours | estimated_50_seed_full_wallclock_hours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | 20260511 | 2236 | 1.01 | 0 | 0 | 2679 | 1569 | 0 | 0 | 0.9481 | 0.986 | 0.0007882 | 5.864 | -5.535 | pass | 6.211 | 31.06 |

## Recommended Command

Do not launch pilot automatically. If the decision above is ready, use:

```bash
.venv/bin/python scripts/main_synthetic_validation.py \
  --output-dir outputs/main_synthetic_validation \
  --mode pilot \
  --fit-variants targeted_strong_lower_override \
  --n-seeds 10 \
  --seed 20260511
```
