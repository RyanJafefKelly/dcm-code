# Phase 0 Audit: Collapsed Rho

## Summary

4 repeated-looking `(run_id, system)` groups were found. 0 had duplicate row_unique_key values. The remaining repeated labels come from distinct source paths/posterior files and are retained with explicit identity columns.

3 hard-anchor anomalies were found. They have finite arrays with matching flat shapes; the surprising mean is explained by highly skewed/multimodal log_B draws where the median log_B is strongly positive but the lower tail drives rho toward zero.

The Phase 0 binary-root conclusion does not change: the coherent target is `rho_s = p(R_s=1 | y_s)`. For postprocessing under Path A, collapsed/fixed rho is the preferred headline because it integrates the root prior odds analytically per posterior draw of the tree/emission parameters.

## Duplicate Row Audit

| run_id | system | n_rows | n_source_paths | finding | source_paths |
| --- | --- | --- | --- | --- | --- |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | 2024 Leading Chat LLMs | 2 | 2 | intended_distinct_source_paths | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506; notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | Chicken | 2 | 2 | intended_distinct_source_paths | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506; notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | ELIZA | 2 | 2 | intended_distinct_source_paths | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506; notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | Human | 2 | 2 | intended_distinct_source_paths | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506; notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 |

## Hard-Anchor Anomalies

| source_path | system | run_id | pi_posterior_mean | log_B_q01 | log_B_q50 | log_B_q99 | rho_sampled_pi_q01 | rho_sampled_pi_q50 | rho_sampled_pi_q99 | rho_collapsed_q01 | rho_collapsed_q50 | rho_collapsed_q99 | n_posterior_draws | shape_warning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK10 | Human | exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK10 | 0.999 | -13.514 | 11.645 | 13.692 | 0.001 | 1.000 | 1.000 | 0.001 | 1.000 | 1.000 | 4000 |  |
| notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK20 | Human | exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK20 | 0.999 | -14.392 | 11.648 | 13.783 | 0.001 | 1.000 | 1.000 | 0.001 | 1.000 | 1.000 | 4000 |  |
| notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK5 | Human | exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK5 | 0.999 | -11.487 | 11.764 | 13.776 | 0.010 | 1.000 | 1.000 | 0.010 | 1.000 | 1.000 | 4000 |  |

## Free-Target Metrics By R_true

| R_true | n | mean_rho_sampled_pi | median_rho_sampled_pi | mean_rho_collapsed | median_rho_collapsed | mean_log_B | median_log_B | mean_brier_sampled_pi | mean_brier_collapsed | mean_log_score_sampled_pi | mean_log_score_collapsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000 | 16.000 | 0.112 | 0.118 | 0.111 | 0.117 | -1.650 | -0.574 | 0.018 | 0.018 | -0.122 | -0.121 |
| 1.000 | 10.000 | 0.314 | 0.361 | 0.315 | 0.369 | -1.024 | 0.986 | 0.502 | 0.500 | -1.529 | -1.526 |

| rho_type | n | TPR | TNR | balanced_accuracy | false_positive_count | false_negative_count |
| --- | --- | --- | --- | --- | --- | --- |
| sampled_pi | 26 | 0.000 | 1.000 | 0.500 | 0 | 10 |
| collapsed | 26 | 0.000 | 1.000 | 0.500 | 0 | 10 |

## Evidence Categories

| rho_type | evidence_category | n |
| --- | --- | --- |
| sampled_pi | ambiguous | 10 |
| sampled_pi | moderate_absent | 10 |
| sampled_pi | strong_absent | 6 |
| collapsed | ambiguous | 10 |
| collapsed | moderate_absent | 10 |
| collapsed | strong_absent | 6 |

## Sampled-Pi Vs Collapsed Rho

| n | mean_abs_difference | max_abs_difference | correlation | n_difference_gt_0_05 | n_difference_gt_0_10 |
| --- | --- | --- | --- | --- | --- |
| 52.000 | 0.001 | 0.016 | 1.000 | 0.000 | 0.000 |

| system | run_id | R_true | pi_true | rho_sampled_pi_mean | rho_collapsed_mean | abs_difference | log_B_median | source_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2024 Leading Chat LLMs | exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__smoke | 1 | 0.100 | 0.361 | 0.377 | 0.016 | 1.012 | notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__smoke |
| Chicken | exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__pres85_abs15_sig50 | 0 | 0.250 | 0.158 | 0.154 | 0.004 | -0.519 | notebooks/asymmetric_prior_sweep_2026-05-10/runs/synthetic/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__pres85_abs15_sig50 |
| 2024 Leading Chat LLMs | exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK20 | 1 | 0.100 | 0.495 | 0.492 | 0.003 | 1.930 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK20 |
| Chicken | exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260508 | 0 | 0.250 | 0.119 | 0.116 | 0.003 | -0.568 | notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260508 |
| Chicken | exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK2 | 0 | 0.250 | 0.158 | 0.155 | 0.003 | -0.406 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK2 |
| Chicken | exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK10 | 0 | 0.250 | 0.206 | 0.208 | 0.002 | -0.008 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK10 |
| 2024 Leading Chat LLMs | exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK5 | 1 | 0.100 | 0.496 | 0.494 | 0.002 | 1.921 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK5 |
| 2024 Leading Chat LLMs | exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK2 | 1 | 0.100 | 0.467 | 0.465 | 0.002 | 1.425 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK2 |
| Chicken | exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__smoke | 0 | 0.250 | 0.039 | 0.038 | 0.002 | -1.891 | notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__smoke |
| Chicken | exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK20 | 0 | 0.250 | 0.214 | 0.213 | 0.002 | 0.035 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK20 |
| 2024 Leading Chat LLMs | exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__pres85_abs15_sig50 | 1 | 0.100 | 0.244 | 0.246 | 0.002 | 0.032 | notebooks/asymmetric_prior_sweep_2026-05-10/runs/synthetic/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__pres85_abs15_sig50 |
| Chicken | exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK5 | 0 | 0.250 | 0.188 | 0.187 | 0.001 | -0.142 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK5 |

## Recommendation

Sampled-pi and collapsed/fixed rho are close in this audit; collapsed/fixed rho is still the cleaner Path A headline because it matches the integrated binary-root estimand.
