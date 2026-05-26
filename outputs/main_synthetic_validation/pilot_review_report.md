# Pilot Review Report

Decision: `NOT_READY_FIX_HUMAN_PPC`

## Executive Summary

This review uses only the existing 10-seed pilot artefacts. No model code was changed, no new HMC was run, and the 50-seed validation was not launched.

The LLM root-recovery warning is resolved as realised-data ambiguity rather than posterior washout: all 3 LLM R=1 posterior failures are also weak/absent under the true-parameter oracle, and there are 0 cases where oracle evidence is present but posterior evidence is absent.

The Human PPC recurrence itself is not rare under the oracle finite-sample baseline: P(Human >= 2 practical excess seeds out of 10) = 0.086; family-level P(any system >= 2) = 0.278. Recurrence alone therefore does not block the full validation.

The remaining blocker is the Human cell decomposition. In the two Human review seeds, 44/100 Human cells are labelled `fitted_truth_mismatch`, including 16/20 of the largest weighted-TV contributors. Those mismatch-labelled cells account for 53.0% of the review-seed weighted-TV contribution. That is enough fitted-vs-truth predictive mismatch to keep the 50-seed run blocked pending Human PPC investigation.

## Human PPC Cell Decomposition

Human has 50 one-rating system x indicator cells per seed in this production-like layout. The review seeds are 20260513 and 20260518. Weighted TV is therefore highly sensitive to individual realised categories, but direct fitted-vs-truth probability mismatch is still informative because it does not depend on the observed category.

Top Human review-seed cell contributions:

| seed | indicator_name | n_obs_in_cell | cell_TV_obs_vs_fit | cell_TV_obs_vs_truth | cell_TV_fit_vs_truth | weighted_cell_TV_contribution | high_abs_error_fit_vs_obs | high_abs_error_fit_vs_truth | oracle_cell_TV_q50 | oracle_cell_TV_q95 | fitted_cell_percentile_under_oracle | ppc_issue_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260513 | Stable Personality | 1 | 0.9374 | 0.9636 | 0.584 | 0.01875 | 0.3125 | 0.3113 | 0.1521 | 0.9636 | 0.9238 | fitted_truth_mismatch |
| 20260518 | Feature Binding | 1 | 0.9347 | 0.8811 | 0.276 | 0.01869 | 0.1674 | 0.003183 | 0.8511 | 0.8811 | 0.962 | fitted_truth_mismatch |
| 20260518 | Dedicated Sensory Systems | 1 | 0.9325 | 0.8811 | 0.1534 | 0.01865 | 0.2115 | 0.04721 | 0.8511 | 0.8811 | 0.9566 | no_specific_issue |
| 20260518 | Stable Personality | 1 | 0.929 | 0.8801 | 0.1725 | 0.01858 | 0.3108 | 0.1465 | 0.8511 | 0.8811 | 0.9562 | no_specific_issue |
| 20260513 | Probe-Detectable Themes | 1 | 0.9244 | 0.8801 | 0.2269 | 0.01849 | 0.1797 | 0.01547 | 0.8511 | 0.8811 | 0.965 | fitted_truth_mismatch |
| 20260513 | Optimized Componentry | 1 | 0.9134 | 0.9748 | 0.3783 | 0.01827 | 0.4842 | 0.3783 | 0.3728 | 0.897 | 0.9676 | fitted_truth_mismatch |
| 20260513 | Performance Degradation | 1 | 0.9134 | 0.9748 | 0.379 | 0.01827 | 0.4836 | 0.379 | 0.3728 | 0.897 | 0.959 | fitted_truth_mismatch |
| 20260513 | Informational Bottleneck | 1 | 0.9128 | 0.9748 | 0.3848 | 0.01826 | 0.4777 | 0.3848 | 0.3728 | 0.897 | 0.9672 | fitted_truth_mismatch |
| 20260513 | Threshold Activation | 1 | 0.9016 | 0.8811 | 0.4118 | 0.01803 | 0.1175 | 0.04678 | 0.8511 | 0.8811 | 0.9574 | fitted_truth_mismatch |
| 20260513 | Perspective-relative Representations | 1 | 0.8927 | 0.9216 | 0.5403 | 0.01785 | 0.2488 | 0.2476 | 0.1521 | 0.9636 | 0.8508 | fitted_truth_mismatch |
| 20260513 | Functional Subparts | 1 | 0.8903 | 0.8811 | 0.1941 | 0.01781 | 0.3141 | 0.1498 | 0.8511 | 0.8811 | 0.9574 | no_specific_issue |
| 20260513 | Poorly Intraconnected Networks | 1 | 0.8901 | 0.9729 | 0.46 | 0.0178 | 0.1361 | 0.135 | 0.1521 | 0.9636 | 0.8426 | fitted_truth_mismatch |
| 20260518 | Adaptive Learning | 1 | 0.8791 | 0.9748 | 0.5976 | 0.01758 | 0.2649 | 0.5976 | 0.3728 | 0.897 | 0.8614 | fitted_truth_mismatch |
| 20260518 | Stable Social Interactions | 1 | 0.8709 | 0.7887 | 0.2386 | 0.01742 | 0.3957 | 0.2314 | 0.8511 | 0.8811 | 0.5994 | fitted_truth_mismatch |
| 20260518 | Priming Enhancement | 1 | 0.8703 | 0.9748 | 0.4647 | 0.01741 | 0.3978 | 0.4647 | 0.3728 | 0.897 | 0.8536 | fitted_truth_mismatch |

Human review-seed issue-label contribution summary:

| ppc_issue_label | n_cells | weighted_contribution | weighted_contribution_share | mean_fit_vs_truth |
| --- | --- | --- | --- | --- |
| fitted_truth_mismatch | 44 | 0.6792 | 0.5298 | 0.385 |
| sparse_cell_unstable | 44 | 0.4074 | 0.3177 | 0.09357 |
| compatible_sparse_noise | 8 | 0.1234 | 0.09627 | 0.08435 |
| no_specific_issue | 4 | 0.07211 | 0.05625 | 0.172 |

Human review-seed fit-vs-truth TV summary:

| seed | mean | median | max |
| --- | --- | --- | --- |
| 20260513 | 0.2456 | 0.1708 | 0.584 |
| 20260518 | 0.2028 | 0.1434 | 0.6686 |

Interpretation: the observed recurrence is statistically plausible under oracle sparse-cell noise, but the review seed cell decomposition does not fully clear PPC. The largest Human weighted-TV contributors include many cells where posterior predictive probabilities are not close to the true generator probabilities, especially in high-category mass. This is a focused Human PPC issue rather than a global baseline failure.

## Human PPC Recurrence Calibration

| system | per_seed_excess_probability_under_oracle | observed_n_excess_seeds | n_seeds | prob_ge_observed_excess_under_oracle | family_prob_any_system_ge_observed_excess | interpretation |
| --- | --- | --- | --- | --- | --- | --- |
| Human | 0.05 | 2 | 10 | 0.08614 | 0.2776 | compatible_with_sparse_finite_sample_noise |
| ELIZA | 0.05 | 1 | 10 | 0.4013 | 0.2776 | single_excess_not_recurrent |
| Chicken | 0.05 | 0 | 10 | 1 | 0.2776 | no_observed_excess |
| LLMs | 0.03818 | 0 | 10 | 1 | 0.2776 | no_observed_excess |

Interpretation: Human recurrence probability is not small. ELIZA's earlier excess remains isolated. The recurrence test alone supports treating repeated system-specific oracle-tail hits as plausible finite-sample behaviour in this sparse layout.

## Predictive Alignment Context

| fit_variant | system | n_seeds | mean_weighted_mean_tv_fit_vs_truth | max_weighted_mean_tv_fit_vs_truth | n_pass | n_warning | n_fail | warning_seed_list | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| targeted_strong_lower_override | ALL | 10 | 0.1099 | 0.1396 | 10 | 0 | 0 |  | pass |
| targeted_strong_lower_override | Chicken | 10 | 0.1188 | 0.1626 | 10 | 0 | 0 |  | pass |
| targeted_strong_lower_override | ELIZA | 10 | 0.1588 | 0.1932 | 10 | 0 | 0 |  | pass |
| targeted_strong_lower_override | Human | 10 | 0.1788 | 0.2456 | 7 | 3 | 0 | 20260513;20260517;20260518 | warning |
| targeted_strong_lower_override | LLMs | 10 | 0.07375 | 0.1144 | 10 | 0 | 0 |  | pass |

Interpretation: fitted-vs-truth predictive alignment passes overall and has no system failures. Human is the only warning system, with warning seeds 20260513, 20260517, and 20260518. This keeps the issue narrow but real enough to review before a 50-seed run.

## LLM Oracle Root Audit

| group | n_cases | posterior_present_rate | oracle_present_rate | posterior_correct_rate | oracle_correct_rate | median_log_B_oracle | median_log_B_posterior |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LLM R=0 | 5 | 0 | 0 | 1 | 1 | -4.432 | -4.959 |
| LLM R=1 | 5 | 0.4 | 0.4 | 0.4 | 0.4 | -1.09 | -0.8484 |

LLM R=1 posterior failures:

| seed | root_z_true | log_B_oracle | rho_oracle_collapsed | log_B_eff_posterior | rho_collapsed_posterior | oracle_classified_present_at_0p5 | posterior_classified_present_at_0p5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20260512 | 1 | -1.568 | 0.04001 | -1.656 | 0.03677 | False | False |
| 20260514 | 1 | -1.09 | 0.063 | -0.8484 | 0.07887 | False | False |
| 20260520 | 1 | -1.26 | 0.05367 | -1.266 | 0.05337 | False | False |

Interpretation: LLM absent-root cases remain cleanly absent. LLM present-root misses are already weak or negative under the true-parameter oracle, so the LLM warning is not evidence of nuisance inference losing strong root evidence.

## Decision

| decision | human_ppc_recurrence_compatible_with_oracle | human_observed_excess_seeds | human_prob_ge_observed_excess_under_oracle | family_prob_any_system_ge_observed_excess | human_fit_truth_mismatch_cells_review_seeds | human_fit_truth_mismatch_cells_top20 | human_fit_truth_mismatch_weight_share_review_seeds | human_seed_20260513_mean_cell_TV_fit_vs_truth | human_seed_20260518_mean_cell_TV_fit_vs_truth | human_predictive_alignment_status | overall_predictive_alignment_status | llm_r1_cases | llm_r1_posterior_failures | llm_r1_failures_oracle_weak | llm_r1_strong_oracle_posterior_failures | sampler_pass | loo_pass | recommended_command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NOT_READY_FIX_HUMAN_PPC | True | 2 | 0.08614 | 0.2776 | 44 | 16 | 0.5298 | 0.2456 | 0.2028 | warning | pass | 5 | 3 | 3 | 0 | True | True |  |

No launch command is recommended for this decision.

Recommended next step: investigate the Human PPC layer using the existing decomposition table first. The likely target is not the binary-root structural repair; it is why several Human one-rating cells have posterior predictive category probabilities materially displaced from the true generator probabilities in the review seeds.
