# Phase 0 Root Semantics Report

## 1. Executive summary

The current DCM is a binary-root model. Under this model, the coherent recoverable posterior probability of consciousness is `rho_s = p(R_s = 1 | y_s)`. The posterior over `pi_s`, currently reported as `C_s`, is not expected to recover fixed continuous synthetic values such as 0.25 for a single system because `pi_s` only generates one latent root draw.

Phase 0 supports that diagnosis. Binary-root toy data learn `R_s` through `rho_s`, while the posterior over `pi_s` hits the one-root-update ceiling instead of contracting to zero. Continuous `C_s` recovery appears only in the separate continuous-propensity toy semantics.

## 2. Production deterministics

Production deterministic check: PASS.

Added exact-tree deterministics per system and stance:

- `{prefix}_log_L_root0`
- `{prefix}_log_L_root1`
- `{prefix}_log_B`
- `{prefix}_rho`

Missing deterministics: []. `rho` finite: True; `rho` in [0, 1]: True; extra potentials added: False.

The root mixture is still the exact-tree likelihood potential. The added quantities are PyMC deterministics and are not added as potentials.

## 3. Existing synthetic reanalysis

Free target systems show the old C_s summaries are not the right binary-root target: mean delta log score versus the 1/6 prior is 0.138, and mean delta Brier is 0.080 when scored through rho_s against realised root_z. Using prior-odds log_B thresholds, free rows contain 0 strong-present, 10 strong-absent, and 16 ambiguous root-evidence cases; 16/26 rho_mean decisions align with root_z at a 0.5 cutoff. Ambiguous cases are weak root evidence under rho_s, not failures to recover a continuous pi_s.

| is_anchor_or_free | n | n_runs | mean_brier_sampled_pi | mean_brier_collapsed | mean_delta_brier | mean_delta_brier_collapsed | mean_delta_log_score | mean_delta_log_score_collapsed | mean_entropy_reduction | median_abs_log_B | median_rho_sampled_pi_mean | median_rho_collapsed_mean | median_pi_contraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| free_target | 26 | 13 | 0.204 | 0.203 | 0.080 | 0.081 | 0.138 | 0.140 | 0.048 | 1.572 | 0.139 | 0.137 | 0.995 |
| hard_anchor | 26 | 13 | 0.007 | 0.007 | 0.354 | 0.354 | 0.955 | 0.955 | 0.385 | 3.203 | 0.377 | 0.377 |  |

| run_id | source_path | system | R_true | pi_true | pi_posterior_mean | pi_posterior_sd | pi_contraction | log_B_median | rho_sampled_pi_mean | rho_collapsed_mean | brier_sampled_pi | brier_collapsed | delta_log_score | delta_log_score_collapsed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | 2024 Leading Chat LLMs | 1 | 0.100 | 0.159 | 0.135 | 0.957 | -2.347 | 0.105 | 0.104 | 0.802 | 0.803 | -0.465 | -0.470 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | 2024 Leading Chat LLMs | 1 | 0.100 | 0.195 | 0.153 | 1.088 | 0.961 | 0.362 | 0.361 | 0.407 | 0.408 | 0.775 | 0.773 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | Chicken | 0 | 0.250 | 0.153 | 0.131 | 0.927 | -1.718 | 0.071 | 0.071 | 0.005 | 0.005 | 0.109 | 0.109 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | Chicken | 0 | 0.250 | 0.147 | 0.126 | 0.897 | -2.007 | 0.032 | 0.032 | 0.001 | 0.001 | 0.150 | 0.150 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | ELIZA | 0 | 0.001 | 0.001 | 0.000 |  | -5.279 | 0.000 | 0.000 | 0.000 | 0.000 | 0.182 | 0.182 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | ELIZA | 0 | 0.001 | 0.001 | 0.000 |  | -1.696 | 0.000 | 0.000 | 0.000 | 0.000 | 0.182 | 0.182 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | Human | 1 | 0.999 | 0.999 | 0.000 |  | 6.921 | 1.000 | 1.000 | 0.000 | 0.000 | 1.792 | 1.792 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | notebooks/synthetic_validation_2026-05-06/runs/full_exact_recovery/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506 | Human | 1 | 0.999 | 0.999 | 0.000 |  | 3.130 | 1.000 | 1.000 | 0.000 | 0.000 | 1.792 | 1.792 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK10 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK10 | 2024 Leading Chat LLMs | 1 | 0.100 | 0.214 | 0.160 | 1.134 | 1.921 | 0.493 | 0.493 | 0.257 | 0.257 | 1.084 | 1.084 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK10 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK10 | Chicken | 0 | 0.250 | 0.171 | 0.142 | 1.010 | -0.008 | 0.206 | 0.208 | 0.042 | 0.043 | -0.048 | -0.051 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK10 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK10 | ELIZA | 0 | 0.001 | 0.001 | 0.000 |  | -2.820 | 0.000 | 0.000 | 0.000 | 0.000 | 0.182 | 0.182 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK10 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK10 | Human | 1 | 0.999 | 0.999 | 0.000 |  | 11.645 | 0.756 | 0.756 | 0.059 | 0.059 | 1.512 | 1.512 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK2 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK2 | 2024 Leading Chat LLMs | 1 | 0.100 | 0.210 | 0.157 | 1.114 | 1.425 | 0.467 | 0.465 | 0.284 | 0.286 | 1.030 | 1.026 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK2 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK2 | Chicken | 0 | 0.250 | 0.166 | 0.141 | 1.004 | -0.406 | 0.158 | 0.155 | 0.025 | 0.024 | 0.010 | 0.013 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK2 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK2 | ELIZA | 0 | 0.001 | 0.001 | 0.000 |  | -2.491 | 0.000 | 0.000 | 0.000 | 0.000 | 0.182 | 0.182 |
| exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK2 | notebooks/sample_size_sweep_2026-05-10/runs/exact_latent_tree__full_exact_tree__three_state_binomial_2__exact_tree_production_medians__current_gwt_rater_design__seed20260506__multK2 | Human | 1 | 0.999 | 0.999 | 0.000 |  | 10.279 | 1.000 | 1.000 | 0.000 | 0.000 | 1.792 | 1.792 |

Central intervals are equal-tailed quantile intervals. ECE is not reported because the available reanalysis has too few independent systems per run for a stable 10-bin calibration estimate.

## 4. Binary-root toy

Prior `pi_s ~ Beta(1, 5)` has prior SD 0.1409. The limiting one-root-update contraction ratios are 0.878 for strong `R_s=0` evidence and 1.134 for strong `R_s=1` evidence.

| J | K | median_rho_given_R1 | median_rho_given_R0 | relative_brier_improvement | mean_log_score_improvement | median_pi_contraction | median_pi_contraction_R1 | median_pi_contraction_R0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.000 | 1.000 | 0.883 | 0.000 | 0.783 | 0.339 | 0.883 | 1.135 | 0.878 |
| 4.000 | 5.000 | 0.999 | 0.000 | 0.927 | 0.355 | 0.878 | 1.134 | 0.878 |
| 4.000 | 20.000 | 0.999 | 0.000 | 0.842 | 0.350 | 0.878 | 1.134 | 0.878 |
| 16.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.440 | 0.878 | 1.134 | 0.878 |
| 16.000 | 5.000 | 1.000 | 0.000 | 1.000 | 0.472 | 0.878 | 1.134 | 0.878 |
| 16.000 | 20.000 | 1.000 | 0.000 | 1.000 | 0.383 | 0.878 | 1.134 | 0.878 |
| 64.000 | 1.000 | 1.000 | 0.000 | 1.000 | 0.375 | 0.878 | 1.134 | 0.878 |
| 64.000 | 5.000 | 1.000 | 0.000 | 1.000 | 0.472 | 0.878 | 1.134 | 0.878 |
| 64.000 | 20.000 | 1.000 | 0.000 | 1.000 | 0.416 | 0.878 | 1.134 | 0.878 |

Fixed-pi demonstration rows at the strongest settings show that `pi_s` does not recover a fixed continuous `pi_true` from one realised root draw:

| J | K | median_rho_given_R1 | median_rho_given_R0 | mean_abs_pi_error | median_pi_contraction |
| --- | --- | --- | --- | --- | --- |
| 64.000 | 1.000 | 1.000 | 0.000 | 0.205 | 0.878 |
| 64.000 | 5.000 | 1.000 | 0.000 | 0.206 | 0.878 |
| 64.000 | 20.000 | 1.000 | 0.000 | 0.206 | 0.878 |

## 5. Continuous-root toy

These are toy diagnostics only; they are not production model changes. They show what changes when the data-generating semantics treats `C_s` as a continuous propensity rather than a one-draw Bernoulli parameter.

| alternative | J | K | RMSE | median_contraction | coverage_80 | coverage_95 |
| --- | --- | --- | --- | --- | --- | --- |
| direct_q_continuous_observation | 64 | 20 | 0.018 | 0.128 | 0.801 | 0.943 |
| latent_leaf_continuous_propensity | 64 | 20 | 0.065 | 0.471 | 0.743 | 0.927 |

## 6. Information precheck

Under binary-root semantics, information separates `R_s=1` from `R_s=0`; it does not estimate a system-specific continuous `pi_s` beyond the single Bernoulli update. With prior root probability 1/6, log_B thresholds are approximately 1.609 for `rho=0.50`, 2.996 for `rho=0.80`, 4.554 for `rho=0.95`, and -1.343 for `rho=0.05`.

| beta_gap_regime | J | K | KL_1_to_0_per_leaf | KL_0_to_1_per_leaf | E_R1_log_B | E_R0_log_B |
| --- | --- | --- | --- | --- | --- | --- |
| extreme | 16 | 5 | 2.631 | 2.631 | 42.092 | -42.092 |
| extreme | 16 | 20 | 2.650 | 2.650 | 42.400 | -42.400 |
| extreme | 64 | 5 | 2.631 | 2.631 | 168.367 | -168.367 |
| extreme | 64 | 20 | 2.650 | 2.650 | 169.600 | -169.600 |
| medium | 16 | 5 | 0.828 | 0.828 | 13.242 | -13.242 |
| medium | 16 | 20 | 0.832 | 0.832 | 13.308 | -13.308 |
| medium | 64 | 5 | 0.828 | 0.828 | 52.967 | -52.967 |
| medium | 64 | 20 | 0.832 | 0.832 | 53.234 | -53.234 |

For continuous-root toys, Fisher information is compared with the Beta(1,5) prior precision scale of about 50.4.

| alternative | C | I_total | I_total_over_prior_precision | approximate_contraction |
| --- | --- | --- | --- | --- |
| direct_q_continuous_observation | 0.050 | 4575.463 | 90.783 | 0.104 |
| direct_q_continuous_observation | 0.100 | 3972.033 | 78.810 | 0.112 |
| direct_q_continuous_observation | 0.250 | 3049.412 | 60.504 | 0.128 |
| direct_q_continuous_observation | 0.500 | 2654.208 | 52.663 | 0.137 |
| direct_q_continuous_observation | 0.750 | 3049.412 | 60.504 | 0.128 |
| latent_leaf_continuous_propensity | 0.050 | 340.199 | 6.750 | 0.359 |
| latent_leaf_continuous_propensity | 0.100 | 277.507 | 5.506 | 0.392 |
| latent_leaf_continuous_propensity | 0.250 | 195.048 | 3.870 | 0.453 |
| latent_leaf_continuous_propensity | 0.500 | 163.840 | 3.251 | 0.485 |
| latent_leaf_continuous_propensity | 0.750 | 195.048 | 3.870 | 0.453 |

## 7. Recommendations

- Keep the current production DCM likelihood for now.
- Report `rho_s` as the Path A headline under binary-root semantics.
- Stop using continuous `pi_s` recovery as a pass/fail metric for the current model.
- Use Brier, log score, and eventually ECE against realised `root_z` for synthetic binary-root validation.
- Treat continuous-propensity `C_s` recovery as a separate v2 model question.

## Commands run

- `.venv/bin/python phase0_root_semantics.py`
