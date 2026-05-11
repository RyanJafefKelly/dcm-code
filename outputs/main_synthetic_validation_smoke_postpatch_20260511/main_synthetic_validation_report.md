# Main Synthetic Validation Report

## Executive Summary

Overall targeted baseline status: `FAIL`.
Validation scope: `smoke`.
Runs completed: 1.

The primary baseline is `targeted_strong_lower_override`: root-to-top edges keep the production prior path, while lower edges under Coherence, Selective Attention, Complexity, and Integration use node-level logit-Normal priors centred at 0.90/0.10.

## Model Variant

The generator truth and fitter prior centres are matched edge-by-edge. The headline posterior quantity is `rho_collapsed`; sampled-pi `rho` is retained as a secondary diagnostic.

## Rho Recovery

| fit_variant | system | n_cases | mean_rho_R1 | mean_rho_R0 | median_rho_R1 | median_rho_R0 | median_log_B_eff_R1 | median_log_B_eff_R0 | evidence_margin_M | TPR_at_rho_gt_0p5 | TNR_at_rho_le_0p5 | balanced_accuracy | decisive_present_rate_R1_at_rho_gt_0p95 | decisive_absent_rate_R0_at_rho_lt_0p05 | brier_model | brier_prior_baseline | brier_improvement | log_score_model | log_score_prior_baseline | log_score_improvement | ECE | calibration_bins | mean_abs_rho_collapsed_minus_sampled_pi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | ALL | 2 | 0.9869 | 0.0006495 | 0.9869 | 0.0006495 | 5.928 | -5.729 | 4.319 | 1 | 1 | 1 | 1 | 1 | 8.657e-05 | 0.3611 | 0.361 | -0.00694 | -0.987 | 0.9801 | 0.006896 | [{"lo": 0.0, "hi": 0.1, "n": 1, "mean_rho": 0.0006494870405678, "empirical_rate": 0.0}, {"lo": 0.9, "hi": 1.0, "n": 1, "mean_rho": 0.986857700329246, "empirical_rate": 1.0}] | 0.000225 |
| targeted_strong_lower_override | Chicken | 1 | 0.9869 |  | 0.9869 |  | 5.928 |  |  | 1 |  | 1 | 1 |  | 8.636e-05 | 0.3611 | 0.361 | -0.006615 | -0.987 | 0.9804 | 0.01314 | [{"lo": 0.9, "hi": 1.0, "n": 1, "mean_rho": 0.986857700329246, "empirical_rate": 1.0}] | 0.0004256 |
| targeted_strong_lower_override | LLMs | 1 |  | 0.0006495 |  | 0.0006495 |  | -5.729 |  |  | 1 | 1 |  | 1 | 2.109e-07 | 0.3611 | 0.3611 | -0.0003248 | -0.987 | 0.9867 | 0.0006495 | [{"lo": 0.0, "hi": 0.1, "n": 1, "mean_rho": 0.0006494870405678, "empirical_rate": 0.0}] | 2.431e-05 |

## Posterior Predictive Fit

| fit_variant | system | n_ratings | mean_RPS | median_RPS | q90_RPS | mean_RPS_baseline_empirical_marginal | mean_RPS_improvement | weighted_mean_cell_TV | median_cell_TV | q90_cell_TV | max_cell_TV | fraction_cell_TV_gt_0p3 | fraction_cell_TV_gt_0p5 | n_cells | n_cells_ge_5 | mean_abs_top7_error_high_q | q90_abs_top7_error_high_q | mean_abs_high_error_high_q | q90_abs_high_error_high_q | posterior_predictive_tv_p_value | ppc_draws | n_seeds | seed_min | seed_max | seed_list | posterior_predictive_tv_p_value_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | ALL | 379 | 0.07946 | 0.05548 | 0.204 | 0.2273 | 0.1478 | 0.424 | 0.4309 | 0.8388 | 0.9347 | 0.6884 | 0.3668 | 199 | 0 | 0.3464 | 0.4886 | 0.2244 | 0.3383 | 1 | 100 | 1 | 20260511 | 20260511 | 20260511 | computed_overall |
| targeted_strong_lower_override | Chicken | 93 | 0.06903 | 0.03908 | 0.1543 | 0.2308 | 0.1618 | 0.4457 | 0.4219 | 0.7504 | 0.8335 | 0.7551 | 0.2857 | 49 | 0 | 0.267 | 0.4223 | 0.1735 | 0.2911 |  | 100 | 1 | 20260511 | 20260511 | 20260511 | not_computed_system_specific |
| targeted_strong_lower_override | ELIZA | 50 | 0.0671 | 0.05549 | 0.151 | 0.2213 | 0.1542 | 0.5744 | 0.6167 | 0.9058 | 0.9347 | 0.64 | 0.5 | 50 | 0 | 0.4395 | 0.5107 | 0.301 | 0.562 |  | 100 | 1 | 20260511 | 20260511 | 20260511 | not_computed_system_specific |
| targeted_strong_lower_override | Human | 50 | 0.07062 | 0.05527 | 0.1329 | 0.2273 | 0.1567 | 0.6157 | 0.6767 | 0.9051 | 0.9237 | 0.94 | 0.52 | 50 | 0 | 0.44 | 0.5144 | 0.2671 | 0.6095 |  | 100 | 1 | 20260511 | 20260511 | 20260511 | not_computed_system_specific |
| targeted_strong_lower_override | LLMs | 186 | 0.09037 | 0.05987 | 0.2917 | 0.2271 | 0.1367 | 0.3212 | 0.2765 | 0.5522 | 0.7434 | 0.42 | 0.16 | 50 | 0 | 0.1455 | 0.2539 | 0.1535 | 0.1606 |  | 100 | 1 | 20260511 | 20260511 | 20260511 | not_computed_system_specific |

## LOO/lppd

| fit_variant | n_ratings | n_ratings_used_for_loo | loo_draws | loo_partial_flag | elpd_loo | se_elpd_loo | p_loo | lppd_conditional | mean_pareto_k | median_pareto_k | max_pareto_k | frac_pareto_k_gt_0p7 | frac_pareto_k_gt_1p0 | n_seeds | seed_min | seed_max | seed_list |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | 379 | 50 | 100 | 1 | -62.9 | 5.585 | 1.797 | -61.1 | 0.188 | 0.1852 | 0.5713 | 0 | 0 | 1 | 20260511 | 20260511 | 20260511 |

## Sampler Diagnostics

| fit_variant | seed | max_rhat | n_rhat_gt_1p01 | n_rhat_gt_1p05 | min_ess_bulk | min_ess_tail | median_ess_bulk | median_ess_tail | n_divergences | divergence_rate | max_tree_depth_hits | max_tree_depth_hit_rate | mean_acceptance_rate | runtime_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | 20260511 | 1.04 | 38 | 0 | 247 | 80 | 761 | 265 | 0 | 0 | 0 | 0 | 0.9729 | 1518 |

## Pass/Fail Table

| fit_variant | validation_scope | pillar | metric | value | threshold | status | notes |
|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | smoke | rho_recovery | evidence_margin_M | 4.319 | > 0 | pass |  |
| targeted_strong_lower_override | smoke | rho_recovery | balanced_accuracy | 1 | >= 0.75 | pass |  |
| targeted_strong_lower_override | smoke | rho_recovery | brier_improvement | 0.361 | > 0 | pass |  |
| targeted_strong_lower_override | smoke | rho_recovery | log_score_improvement | 0.9801 | > 0 | pass |  |
| targeted_strong_lower_override | smoke | rho_recovery | mean_rho_gap | 0.9862 | > 0.30 | pass |  |
| targeted_strong_lower_override | smoke | posterior_predictive | mean_RPS_improvement | 0.1478 | > 0 | pass |  |
| targeted_strong_lower_override | smoke | posterior_predictive | weighted_mean_cell_TV | 0.424 | <= 0.25 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | median_cell_TV | 0.4309 | <= 0.25 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | q90_cell_TV | 0.8388 | <= 0.50 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | fraction_cell_TV_gt_0p5 | 0.3668 | <= 0.10 | fail |  |
| targeted_strong_lower_override | smoke | posterior_predictive | posterior_predictive_tv_p_value | 1 | in [0.05, 0.95] | fail |  |
| targeted_strong_lower_override | smoke | loo_lppd | frac_pareto_k_gt_0p7 | 0 | <= 0.05 | pass | partial |
| targeted_strong_lower_override | smoke | loo_lppd | frac_pareto_k_gt_1p0 | 0 | == 0 | pass | partial |
| targeted_strong_lower_override | smoke | loo_lppd | max_pareto_k | 0.5713 | < 1.0 | pass | partial |
| targeted_strong_lower_override | smoke | sampler_health | fit_pass_rate_no_divergences | 1 | >= 0.90 | pass |  |
| targeted_strong_lower_override | smoke | sampler_health | fit_pass_rate_rhat | 0 | >= 0.90 | fail |  |
| targeted_strong_lower_override | smoke | sampler_health | fit_pass_rate_ess_bulk | 0 | >= 0.90 | fail |  |
| targeted_strong_lower_override | smoke | sampler_health | fit_pass_rate_ess_tail | 0 | >= 0.90 | fail |  |
| targeted_strong_lower_override | smoke | sampler_health | fit_pass_rate_tree_depth | 1 | >= 0.90 | pass |  |

## Limitations and Next Steps

- The targeted override is a structural repair, not a posterior discovery.
- The unmodified published priors remain a negative control from Phase 1C, not the shipping baseline.
- If Chicken remains weaker than LLMs, the production rater design lacking Chicken cross-system coverage is the likely next bottleneck.
- If LOO was partial, elpd_loo applies only to the evaluated rating subset.
