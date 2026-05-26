# Main Synthetic Validation Report

## Executive Summary

Overall targeted baseline status: `FAIL`.
Validation scope: `pilot`.
Runs completed: 10.

The primary baseline is `targeted_strong_lower_override`: root-to-top edges keep the production prior path, while lower edges under Coherence, Selective Attention, Complexity, and Integration use node-level logit-Normal priors centred at 0.90/0.10.

## Model Variant

The generator truth and fitter prior centres are matched edge-by-edge. The headline posterior quantity is `rho_collapsed`; sampled-pi `rho` is retained as a secondary diagnostic.

## Rho Recovery

| fit_variant | system | n_cases | mean_rho_R1 | mean_rho_R0 | median_rho_R1 | median_rho_R0 | median_log_B_eff_R1 | median_log_B_eff_R0 | evidence_margin_M | TPR_at_rho_gt_0p5 | TNR_at_rho_le_0p5 | balanced_accuracy | decisive_present_rate_R1_at_rho_gt_0p95 | decisive_absent_rate_R0_at_rho_lt_0p05 | brier_model | brier_prior_baseline | brier_improvement | log_score_model | log_score_prior_baseline | log_score_improvement | ECE | calibration_bins | mean_abs_rho_collapsed_minus_sampled_pi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | ALL | 20 | 0.6006 | 0.01054 | 0.7406 | 0.004074 | 3.087 | -3.9 | 1.478 | 0.6 | 1 | 0.8 | 0.4 | 0.9 | 0.1585 | 0.3611 | 0.2027 | -0.5211 | -0.987 | 0.466 | 0.1944 | [{"lo": 0.0, "hi": 0.1, "n": 13, "mean_rho": 0.021107048954298678, "empirical_rate": 0.23076923076923078}, {"lo": 0.4, "hi": 0.5, "n": 1, "mean_rho": 0.4778223304489019, "empirical_rate": 1.0}, {"lo": 0.5, "hi": 0.6000000000000001, "n": 1, "mean_rho": 0.5383601013574584, "empirical_rate": 1.0}, {"lo": 0.9, "hi": 1.0, "n": 5, "mean_rho": 0.9642244426067187, "empirical_rate": 1.0}] | 0.0005164 |
| targeted_strong_lower_override | Chicken | 10 | 0.866 | 0.01908 | 0.9528 | 0.01557 | 4.615 | -2.538 | 1.203 | 0.8 | 1 | 0.9 | 0.6 | 0.8 | 0.02826 | 0.3611 | 0.3329 | -0.09871 | -0.987 | 0.8883 | 0.07656 | [{"lo": 0.0, "hi": 0.1, "n": 5, "mean_rho": 0.01907964471467356, "empirical_rate": 0.0}, {"lo": 0.4, "hi": 0.5, "n": 1, "mean_rho": 0.4778223304489019, "empirical_rate": 1.0}, {"lo": 0.9, "hi": 1.0, "n": 4, "mean_rho": 0.9629921195857705, "empirical_rate": 1.0}] | 0.0005797 |
| targeted_strong_lower_override | LLMs | 10 | 0.3353 | 0.001998 | 0.07887 | 0.001402 | -0.8484 | -4.959 | -2.458 | 0.4 | 1 | 0.7 | 0.2 | 1 | 0.2887 | 0.3611 | 0.07246 | -0.9434 | -0.987 | 0.04361 | 0.3313 | [{"lo": 0.0, "hi": 0.1, "n": 8, "mean_rho": 0.022374176604064377, "empirical_rate": 0.375}, {"lo": 0.5, "hi": 0.6000000000000001, "n": 1, "mean_rho": 0.5383601013574584, "empirical_rate": 1.0}, {"lo": 0.9, "hi": 1.0, "n": 1, "mean_rho": 0.9691537346905116, "empirical_rate": 1.0}] | 0.000453 |

## Posterior Predictive Fit

| fit_variant | system | n_ratings | mean_RPS | median_RPS | q90_RPS | mean_RPS_baseline_empirical_marginal | mean_RPS_improvement | weighted_mean_cell_TV | median_cell_TV | q90_cell_TV | max_cell_TV | fraction_cell_TV_gt_0p3 | fraction_cell_TV_gt_0p5 | n_cells | n_cells_ge_5 | mean_abs_top7_error_high_q | q90_abs_top7_error_high_q | mean_abs_high_error_high_q | q90_abs_high_error_high_q | posterior_predictive_tv_p_value | ppc_draws | n_seeds | seed_min | seed_max | seed_list | posterior_predictive_tv_p_value_status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | ALL | 379 | 0.06713 | 0.03168 | 0.1696 | 0.2264 | 0.1592 | 0.3876 | 0.376 | 0.8281 | 0.9359 | 0.6523 | 0.3221 | 199 | 0 | 0.3404 | 0.5385 | 0.2084 | 0.362 | 0.9952 | 500 | 10 | 20260511 | 20260520 | 20260511;20260512;20260513;20260514;20260515;20260516;20260517;20260518;20260519;20260520 | computed_overall |
| targeted_strong_lower_override | Chicken | 93 | 0.06561 | 0.03407 | 0.1498 | 0.2343 | 0.1687 | 0.4082 | 0.3695 | 0.7279 | 0.8576 | 0.6714 | 0.2571 | 49 | 0 | 0.2664 | 0.44 | 0.1853 | 0.3039 |  | 500 | 10 | 20260511 | 20260520 | 20260511;20260512;20260513;20260514;20260515;20260516;20260517;20260518;20260519;20260520 | not_computed_system_specific |
| targeted_strong_lower_override | ELIZA | 50 | 0.05431 | 0.03368 | 0.1347 | 0.2329 | 0.1786 | 0.4926 | 0.3968 | 0.8666 | 0.9298 | 0.634 | 0.404 | 50 | 0 | 0.433 | 0.544 | 0.2486 | 0.3959 |  | 500 | 10 | 20260511 | 20260520 | 20260511;20260512;20260513;20260514;20260515;20260516;20260517;20260518;20260519;20260520 | not_computed_system_specific |
| targeted_strong_lower_override | Human | 50 | 0.06224 | 0.04101 | 0.1211 | 0.2323 | 0.17 | 0.5716 | 0.5159 | 0.8684 | 0.929 | 0.904 | 0.478 | 50 | 0 | 0.4197 | 0.5418 | 0.2435 | 0.5125 |  | 500 | 10 | 20260511 | 20260520 | 20260511;20260512;20260513;20260514;20260515;20260516;20260517;20260518;20260519;20260520 | not_computed_system_specific |
| targeted_strong_lower_override | LLMs | 186 | 0.07264 | 0.02758 | 0.2019 | 0.219 | 0.1464 | 0.2996 | 0.2541 | 0.5284 | 0.6708 | 0.4 | 0.148 | 50 | 0 | 0.1754 | 0.3205 | 0.1446 | 0.1804 |  | 500 | 10 | 20260511 | 20260520 | 20260511;20260512;20260513;20260514;20260515;20260516;20260517;20260518;20260519;20260520 | not_computed_system_specific |

## LOO/lppd

| fit_variant | n_ratings | n_ratings_used_for_loo | loo_draws | loo_partial_flag | elpd_loo | se_elpd_loo | p_loo | lppd_conditional | mean_pareto_k | median_pareto_k | max_pareto_k | frac_pareto_k_gt_0p7 | frac_pareto_k_gt_1p0 | n_seeds | seed_min | seed_max | seed_list |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | 379 | 379 | 500 | 0 | -462.4 | 19.69 | 12.32 | -450.1 | 0.04259 | 0.0376 | 0.4903 | 0 | 0 | 10 | 20260511 | 20260520 | 20260511;20260512;20260513;20260514;20260515;20260516;20260517;20260518;20260519;20260520 |

## Sampler Diagnostics

| fit_variant | seed | max_rhat | n_rhat_gt_1p01 | n_rhat_gt_1p05 | min_ess_bulk | min_ess_tail | median_ess_bulk | median_ess_tail | n_divergences | divergence_rate | max_tree_depth_hits | max_tree_depth_hit_rate | mean_acceptance_rate | runtime_seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | 20260511 | 1.01 | 0 | 0 | 2679 | 1569 | 7631 | 2784 | 0 | 0 | 0 | 0 | 0.9481 | 2219 |
| targeted_strong_lower_override | 20260512 | 1.01 | 0 | 0 | 2105 | 882 | 5473 | 2823 | 0 | 0 | 0 | 0 | 0.947 | 1943 |
| targeted_strong_lower_override | 20260513 | 1 | 0 | 0 | 2621 | 1494 | 5170 | 2870 | 0 | 0 | 0 | 0 | 0.9477 | 1982 |
| targeted_strong_lower_override | 20260514 | 1.01 | 0 | 0 | 1904 | 1509 | 5113 | 2865 | 0 | 0 | 0 | 0 | 0.9457 | 2078 |
| targeted_strong_lower_override | 20260515 | 1 | 0 | 0 | 2421 | 1284 | 6040 | 2848 | 0 | 0 | 0 | 0 | 0.9417 | 2439 |
| targeted_strong_lower_override | 20260516 | 1.01 | 0 | 0 | 3208 | 1515 | 8436 | 2738 | 0 | 0 | 0 | 0 | 0.9423 | 2523 |
| targeted_strong_lower_override | 20260517 | 1.01 | 0 | 0 | 2889 | 1596 | 5456 | 2812 | 0 | 0 | 0 | 0 | 0.9451 | 2239 |
| targeted_strong_lower_override | 20260518 | 1.01 | 0 | 0 | 2962 | 1875 | 9482 | 2710 | 0 | 0 | 0 | 0 | 0.9448 | 2418 |
| targeted_strong_lower_override | 20260519 | 1.01 | 0 | 0 | 2763 | 1587 | 9900 | 2734 | 0 | 0 | 0 | 0 | 0.9452 | 2348 |
| targeted_strong_lower_override | 20260520 | 1 | 0 | 0 | 2211 | 1327 | 4786 | 2984 | 0 | 0 | 0 | 0 | 0.945 | 1646 |

## Pass/Fail Table

| fit_variant | validation_scope | pillar | metric | value | threshold | status | notes |
|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | pilot | rho_recovery | evidence_margin_M | 1.478 | > 0 | pass |  |
| targeted_strong_lower_override | pilot | rho_recovery | balanced_accuracy | 0.8 | >= 0.75 | pass |  |
| targeted_strong_lower_override | pilot | rho_recovery | brier_improvement | 0.2027 | > 0 | pass |  |
| targeted_strong_lower_override | pilot | rho_recovery | log_score_improvement | 0.466 | > 0 | pass |  |
| targeted_strong_lower_override | pilot | rho_recovery | mean_rho_gap | 0.5901 | > 0.30 | pass |  |
| targeted_strong_lower_override | pilot | posterior_predictive | mean_RPS_improvement | 0.1592 | > 0 | pass |  |
| targeted_strong_lower_override | pilot | posterior_predictive | weighted_mean_cell_TV | 0.3876 | <= 0.25 | fail |  |
| targeted_strong_lower_override | pilot | posterior_predictive | median_cell_TV | 0.376 | <= 0.25 | fail |  |
| targeted_strong_lower_override | pilot | posterior_predictive | q90_cell_TV | 0.8281 | <= 0.50 | fail |  |
| targeted_strong_lower_override | pilot | posterior_predictive | fraction_cell_TV_gt_0p5 | 0.3221 | <= 0.10 | fail |  |
| targeted_strong_lower_override | pilot | posterior_predictive | posterior_predictive_tv_p_value | 0.9952 | in [0.05, 0.95] | fail |  |
| targeted_strong_lower_override | pilot | loo_lppd | frac_pareto_k_gt_0p7 | 0 | <= 0.05 | pass |  |
| targeted_strong_lower_override | pilot | loo_lppd | frac_pareto_k_gt_1p0 | 0 | == 0 | pass |  |
| targeted_strong_lower_override | pilot | loo_lppd | max_pareto_k | 0.4903 | < 1.0 | pass |  |
| targeted_strong_lower_override | pilot | sampler_health | fit_pass_rate_no_divergences | 1 | >= 0.90 | pass |  |
| targeted_strong_lower_override | pilot | sampler_health | fit_pass_rate_rhat | 1 | >= 0.90 | pass |  |
| targeted_strong_lower_override | pilot | sampler_health | fit_pass_rate_ess_bulk | 1 | >= 0.90 | pass |  |
| targeted_strong_lower_override | pilot | sampler_health | fit_pass_rate_ess_tail | 1 | >= 0.90 | pass |  |
| targeted_strong_lower_override | pilot | sampler_health | fit_pass_rate_tree_depth | 1 | >= 0.90 | pass |  |

## Limitations and Next Steps

- The targeted override is a structural repair, not a posterior discovery.
- The unmodified published priors remain a negative control from Phase 1C, not the shipping baseline.
- If Chicken remains weaker than LLMs, the production rater design lacking Chicken cross-system coverage is the likely next bottleneck.
- If LOO was partial, elpd_loo applies only to the evaluated rating subset.
