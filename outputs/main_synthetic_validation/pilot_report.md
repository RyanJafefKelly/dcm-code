# Pilot Report

Decision: `NOT_READY_FOR_50_SEED_FULL_VALIDATION_PPC_REVIEW`

## Executive Summary

The 10-seed `targeted_strong_lower_override` pilot completed end-to-end with full-default HMC, 500-draw PPC, and full all-rating 500-draw LOO for every seed. Rho recovery passes overall. Chicken recovery is strong; LLM recovery passes the no-collapse balanced-accuracy gate but remains weak in evidence decisiveness.

Sampler and LOO diagnostics pass cleanly across all 10 fits. Overall oracle-calibrated weighted PPC is not the blocker, and fitted-vs-truth predictive alignment has no failures. The original ELIZA excess did not recur. However, Human system-specific weighted cell-TV shows repeated practical oracle excess in two seeds, so under the pilot rule this should be treated as a PPC review/block before launching the 50-seed full validation.

## Rho Recovery

| system | n_cases | evidence_margin_M | balanced_accuracy | TPR_at_rho_gt_0p5 | TNR_at_rho_le_0p5 | brier_improvement | log_score_improvement | rho_status | rho_status_notes |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 20 | 1.478 | 0.8 | 0.6 | 1 | 0.2027 | 0.466 | pass | overall hard rho recovery gates |
| Chicken | 10 | 1.203 | 0.9 | 0.8 | 1 | 0.3329 | 0.8883 | pass | system-specific recovery is decisive enough for pilot |
| LLMs | 10 | -2.458 | 0.7 | 0.4 | 1 | 0.07246 | 0.04361 | warning | passes no-collapse BA gate but evidence margin/TPR is weak |

Answer: overall rho recovery passes. Chicken passes strongly. LLMs passes the system no-collapse BA gate (`0.70 >= 0.65`) but has negative evidence margin and low TPR, so it is a warning rather than a decisive per-system pass.

## Oracle-Calibrated Weighted PPC

| system | n_seeds | n_practical_excess | mean_fitted_ppc_value | max_fitted_ppc_value | mean_fitted_minus_oracle_median | max_practical_excess_over_oracle_q50 | practical_excess_seed_list | status |
|---|---|---|---|---|---|---|---|---|
| ALL | 10 | 0 | 0.3876 | 0.4234 | -0.003517 | 0 |  | pass |
| Chicken | 10 | 0 | 0.4082 | 0.4818 | -0.006325 | 0 |  | pass |
| ELIZA | 10 | 1 | 0.4926 | 0.5738 | 0.01145 | 0.06742 | 20260511 | pass |
| Human | 10 | 2 | 0.5716 | 0.6481 | 0.01112 | 0.1167 | 20260513;20260518 | fail_repeated_system_excess |
| LLMs | 10 | 0 | 0.2996 | 0.3254 | -0.009378 | 0 |  | pass |

Answer: ELIZA excess is isolated to seed `20260511`. Human has repeated practical excess at seeds `20260513` and `20260518`; this is the remaining PPC issue under the user-specified pilot rule. Overall weighted TV remains acceptable under the oracle-calibrated interpretation.

Sparse unweighted median/q90/fraction metrics remain warnings only because every production-like system x indicator layout has `n_cells_ge_5 = 0`.

## Predictive Alignment To Truth

| system | n_seeds | mean_weighted_mean_tv_fit_vs_truth | max_weighted_mean_tv_fit_vs_truth | n_warning | n_fail | warning_seed_list | status |
|---|---|---|---|---|---|---|---|
| ALL | 10 | 0.1099 | 0.1396 | 0 | 0 |  | pass |
| Chicken | 10 | 0.1188 | 0.1626 | 0 | 0 |  | pass |
| ELIZA | 10 | 0.1588 | 0.1932 | 0 | 0 |  | pass |
| Human | 10 | 0.1788 | 0.2456 | 3 | 0 | 20260513;20260517;20260518 | warning |
| LLMs | 10 | 0.07375 | 0.1144 | 0 | 0 |  | pass |

Answer: fitted-vs-truth predictive alignment does not fail overall or for any system. Human has warnings in three seeds, matching the system-specific PPC concern, but the warning range does not cross the readiness fail threshold.

## Sampler Diagnostics

| seed | max_rhat | min_ess_bulk | min_ess_tail | n_divergences | max_tree_depth_hits | mean_acceptance_rate | runtime_seconds | sampler_status |
|---|---|---|---|---|---|---|---|---|
| ALL | 1.01 | 1904 | 882 | 0 | 0 | 0.9452 | 2.151e+04 | pass |

Answer: sampler diagnostics pass under full defaults: no divergences, max Rhat <= 1.01, ESS thresholds pass, and no tree-depth hits.

## LOO Diagnostics

| seed | n_ratings_used_for_loo | loo_draws | loo_partial_flag | elpd_loo | se_elpd_loo | max_pareto_k | frac_pareto_k_gt_0p7 | frac_pareto_k_gt_1p0 | loo_status |
|---|---|---|---|---|---|---|---|---|---|
| ALL | 379 | 500 | 0 | -462.4 | 19.69 | 0.4903 | 0 | 0 | pass |

Answer: LOO diagnostics pass. LOO used all 379 ratings and 500 draws for every seed; max Pareto-k is below 0.7 in the aggregate summary and no seed had k > 0.7.

## Pass/Fail Summary

| fit_variant | validation_scope | pillar | metric | value | threshold | status | notes |
|---|---|---|---|---|---|---|---|
| targeted_strong_lower_override | pilot | rho_recovery | overall_status | pass | overall hard gates | pass | M=1.478, BA=0.800 |
| targeted_strong_lower_override | pilot | rho_recovery | chicken_status | pass | BA>=0.65 no-collapse; M>0 for strong pass | pass | M=1.203, BA=0.900 |
| targeted_strong_lower_override | pilot | rho_recovery | llms_status | warning | BA>=0.65 no-collapse; M>0 for strong pass | warning | M=-2.458, BA=0.700, TPR=0.400 |
| targeted_strong_lower_override | pilot | posterior_predictive | overall_oracle_weighted_tv | 0 | no repeated overall practical excess | pass | mean fitted TV=0.388; practical excess seeds= |
| targeted_strong_lower_override | pilot | posterior_predictive | eliza_weighted_tv_recurrence | 1 | not repeated across seeds | pass | ELIZA practical excess seeds=20260511 |
| targeted_strong_lower_override | pilot | posterior_predictive | system_weighted_tv_repeated_excess | Human:2 | no system with >1 practical excess seed | fail | Repeated practical excess is a PPC block under the pilot rule. |
| targeted_strong_lower_override | pilot | predictive_alignment | overall_alignment_status | pass | overall not fail | pass | mean TV fit-vs-truth=0.110 |
| targeted_strong_lower_override | pilot | predictive_alignment | any_system_alignment_fail | 0 | 0 system failures | pass | Warnings are present for Human but no system fails. |
| targeted_strong_lower_override | pilot | sampler_health | overall_sampler_status | pass | >=90% fits pass and no divergences | pass | total divergences=0, max Rhat=1.01, min ESS bulk=1904, min ESS tail=882 |
| targeted_strong_lower_override | pilot | loo_lppd | overall_loo_status | pass | Pareto-k diagnostic pass | pass | max Pareto-k=0.490, frac k>0.7=0.000 |
| targeted_strong_lower_override | pilot | overall | pilot_decision | NOT_READY_FOR_50_SEED_FULL_VALIDATION_PPC_REVIEW | ready only if no hard/recurrent PPC block | warning | Pilot completed; final decision follows the user-specified PPC recurrence rule. |

## Readiness For 50-Seed Full Validation

Not ready to launch the 50-seed full validation automatically under the stated pilot interpretation rule, because Human shows repeated practically large system-specific oracle-calibrated weighted TV excess. If that Human-only sparse-cell behaviour is accepted as non-blocking, the remaining pillars support proceeding: rho recovery passes overall, ELIZA does not recur, predictive alignment has no failures, sampler diagnostics pass, and LOO diagnostics pass.
