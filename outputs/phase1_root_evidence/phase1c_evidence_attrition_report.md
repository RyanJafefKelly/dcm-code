# Phase 1C Evidence Attrition Audit

Replicates: `500` matched seeds from `20260511`. No HMC refits were launched.

## 1. Sanity Checks
| rung | beta_profile | observation_model | latent_leaf_model | unique_observation_values | max_abs_diff_leaf_loglik_vs_rung_1P | exact_loglik_consistency_error | status |
|---|---|---|---|---|---|---|---|
| 1P | production_prior_mean | binary_noisy | binary | 0,1 | 0.000 | 0.000 | pass |
| 3 | production_prior_mean | ordinal_binary | binary | 0,1,2,3,4,5,6 | 22.194 | 0.000 | pass |
| 4 | production_prior_mean | ordinal_three_state | three_state | 0,1,2,3,4,5,6 | 21.371 | 0.000 | pass |
| 1O | production_oracle_median | binary_noisy | binary | 0,1 | 0.000 | 0.000 | pass |
| 3O | production_oracle_median | ordinal_binary | binary | 0,1,2,3,4,5,6 | 27.736 | 0.000 | pass |
| 4O | production_oracle_median | ordinal_three_state | three_state | 0,1,2,3,4,5,6 | 26.195 | 0.000 | pass |

Leaf-perfect median absolute evidence check: perfect=0.865, noisy K6=0.865, passed=True.
Pure log_B columns exclude the root prior; the prior is only used to compute rho and threshold gates.

## 2. Oracle-Median No-Fit Rungs
| beta_profile | evidence_stage | rung_label | system | median_log_B_R1 | median_log_B_R0 | balanced_accuracy | evidence_margin_M | pass_fail_label |
|---|---|---|---|---|---|---|---|---|
| extreme_0p9_0p1 | binary_noisy_K6 | 1X | Chicken | 10.592 | -10.275 | 0.990 | 8.940 | pass |
| extreme_0p9_0p1 | binary_noisy_K6 | 1X | LLMs | 10.280 | -10.270 | 0.990 | 8.671 | pass |
| extreme_0p9_0p1 | ordinal_binary_clamped | 3X | Chicken | 10.592 | -10.275 | 0.990 | 8.940 | pass |
| extreme_0p9_0p1 | ordinal_binary_clamped | 3X | LLMs | 10.280 | -10.294 | 0.990 | 8.671 | pass |
| extreme_0p9_0p1 | ordinal_three_state_clamped | 4X | Chicken | 11.495 | -10.965 | 0.992 | 9.630 | pass |
| extreme_0p9_0p1 | ordinal_three_state_clamped | 4X | LLMs | 11.472 | -10.863 | 0.996 | 9.528 | pass |
| production_oracle_median | binary_noisy_K6 | 1O | Chicken | 1.756 | -2.038 | 0.758 | 0.146 | pass |
| production_oracle_median | binary_noisy_K6 | 1O | LLMs | 1.547 | -1.408 | 0.722 | -0.062 | fail |
| production_oracle_median | ordinal_binary_clamped | 3O | Chicken | 1.756 | -2.033 | 0.758 | 0.147 | pass |
| production_oracle_median | ordinal_binary_clamped | 3O | LLMs | 1.541 | -1.448 | 0.720 | -0.069 | fail |
| production_oracle_median | ordinal_three_state_clamped | 4O | Chicken | 2.317 | -2.292 | 0.796 | 0.708 | pass |
| production_oracle_median | ordinal_three_state_clamped | 4O | LLMs | 2.423 | -2.108 | 0.798 | 0.773 | pass |
| production_prior_mean | binary_noisy_K6 | 1P | Chicken | 0.562 | -0.669 | 0.546 | -1.047 | fail |
| production_prior_mean | binary_noisy_K6 | 1P | LLMs | 0.558 | -0.591 | 0.558 | -1.052 | fail |
| production_prior_mean | ordinal_binary_clamped | 3 | Chicken | 0.568 | -0.669 | 0.544 | -1.042 | fail |
| production_prior_mean | ordinal_binary_clamped | 3 | LLMs | 0.561 | -0.586 | 0.560 | -1.049 | fail |
| production_prior_mean | ordinal_three_state_clamped | 4 | Chicken | 0.824 | -0.682 | 0.598 | -0.786 | fail |
| production_prior_mean | ordinal_three_state_clamped | 4 | LLMs | 0.702 | -0.637 | 0.610 | -0.907 | fail |

## 3. Layer-Wise Attrition
| beta_profile | evidence_stage | system | median_log_B_R1 | median_log_B_R0 | balanced_accuracy | evidence_margin_M | pass_fail_label |
|---|---|---|---|---|---|---|---|
| production_oracle_median | binary_noisy_K6 | Chicken | 1.756 | -2.038 | 0.758 | 0.146 | pass |
| production_oracle_median | binary_noisy_K6 | LLMs | 1.547 | -1.408 | 0.722 | -0.062 | fail |
| production_oracle_median | binary_perfect_leaf | Chicken | 1.756 | -2.033 | 0.758 | 0.147 | pass |
| production_oracle_median | binary_perfect_leaf | LLMs | 1.541 | -1.448 | 0.720 | -0.069 | fail |
| production_oracle_median | leaf_latent_observed | Chicken | 1.756 | -2.033 | 0.758 | 0.147 | pass |
| production_oracle_median | leaf_latent_observed | LLMs | 1.541 | -1.448 | 0.720 | -0.069 | fail |
| production_oracle_median | ordinal_binary_clamped | Chicken | 1.756 | -2.033 | 0.758 | 0.147 | pass |
| production_oracle_median | ordinal_binary_clamped | LLMs | 1.541 | -1.448 | 0.720 | -0.069 | fail |
| production_oracle_median | ordinal_three_state_clamped | Chicken | 2.317 | -2.292 | 0.796 | 0.708 | pass |
| production_oracle_median | ordinal_three_state_clamped | LLMs | 2.423 | -2.108 | 0.798 | 0.773 | pass |
| production_oracle_median | subfeature_observed | Chicken | 2.637 | -2.853 | 0.816 | 1.027 | pass |
| production_oracle_median | subfeature_observed | LLMs | 2.629 | -2.386 | 0.814 | 1.019 | pass |
| production_oracle_median | top_observed | Chicken | 5.458 | -4.395 | 0.926 | 3.060 | pass |
| production_oracle_median | top_observed | LLMs | 4.839 | -4.329 | 0.938 | 2.994 | pass |
| production_prior_mean | binary_noisy_K6 | Chicken | 0.562 | -0.669 | 0.546 | -1.047 | fail |
| production_prior_mean | binary_noisy_K6 | LLMs | 0.558 | -0.591 | 0.558 | -1.052 | fail |
| production_prior_mean | binary_perfect_leaf | Chicken | 0.568 | -0.669 | 0.544 | -1.042 | fail |
| production_prior_mean | binary_perfect_leaf | LLMs | 0.561 | -0.586 | 0.560 | -1.049 | fail |
| production_prior_mean | leaf_latent_observed | Chicken | 0.568 | -0.669 | 0.544 | -1.042 | fail |
| production_prior_mean | leaf_latent_observed | LLMs | 0.561 | -0.586 | 0.560 | -1.049 | fail |
| production_prior_mean | ordinal_binary_clamped | Chicken | 0.568 | -0.669 | 0.544 | -1.042 | fail |
| production_prior_mean | ordinal_binary_clamped | LLMs | 0.561 | -0.586 | 0.560 | -1.049 | fail |
| production_prior_mean | ordinal_three_state_clamped | Chicken | 0.824 | -0.682 | 0.598 | -0.786 | fail |
| production_prior_mean | ordinal_three_state_clamped | LLMs | 0.702 | -0.637 | 0.610 | -0.907 | fail |
| production_prior_mean | subfeature_observed | Chicken | 1.020 | -1.059 | 0.640 | -0.590 | fail |
| production_prior_mean | subfeature_observed | LLMs | 1.213 | -0.865 | 0.664 | -0.470 | fail |
| production_prior_mean | top_observed | Chicken | 2.809 | -3.510 | 0.880 | 1.199 | pass |
| production_prior_mean | top_observed | LLMs | 2.809 | -2.854 | 0.856 | 1.199 | pass |
Evidence first falls below the pass gate at the leaf/subtree stages under production beta; top-observed passes but descendant-observed stages generally do not.

## 4. Top-State Message Recovery
| rung | beta_profile | top_feature_name | p_message_correct_sign | p_message_decisive | median_c_top_latent | median_c_oracle_subtree | mean_abs_delta_oracle_minus_top |
|---|---|---|---|---|---|---|---|
| 1O | production_oracle_median | Coherence | 0.693 | 0.203 | -1.492 | 0.012 | 1.422 |
| 1O | production_oracle_median | Selective Attention | 0.782 | 0.476 | 1.388 | 0.093 | 0.961 |
| 1O | production_oracle_median | Complexity | 0.768 | 0.462 | 0.841 | 0.182 | 0.857 |
| 1O | production_oracle_median | Integration | 0.852 | 0.687 | 1.388 | -0.417 | 0.731 |
| 1O | production_oracle_median | Hierarchical Organization | 0.798 | 0.243 | 0.272 | 0.126 | 0.307 |
| 1O | production_oracle_median | Modularity | 0.923 | 0.879 | -0.555 | -0.444 | 0.160 |
| 1O | production_oracle_median | Representationality | 0.812 | 0.452 | 0.062 | 0.044 | 0.081 |
| 3O | production_oracle_median | Coherence | 0.692 | 0.202 | -1.492 | 0.008 | 1.423 |
| 3O | production_oracle_median | Selective Attention | 0.784 | 0.476 | 1.388 | 0.093 | 0.959 |
| 3O | production_oracle_median | Complexity | 0.769 | 0.462 | 0.841 | 0.182 | 0.858 |
| 3O | production_oracle_median | Integration | 0.852 | 0.690 | 1.388 | -0.417 | 0.729 |
| 3O | production_oracle_median | Hierarchical Organization | 0.800 | 0.251 | 0.272 | 0.126 | 0.307 |
| 3O | production_oracle_median | Modularity | 0.923 | 0.880 | -0.555 | -0.446 | 0.161 |
| 3O | production_oracle_median | Representationality | 0.813 | 0.455 | 0.062 | 0.044 | 0.081 |
| 4O | production_oracle_median | Coherence | 0.728 | 0.335 | -1.492 | 0.149 | 1.299 |
| 4O | production_oracle_median | Selective Attention | 0.848 | 0.633 | 1.388 | 0.154 | 0.725 |
| 4O | production_oracle_median | Complexity | 0.819 | 0.570 | 0.841 | 0.324 | 0.721 |
| 4O | production_oracle_median | Integration | 0.930 | 0.689 | 1.388 | 0.746 | 0.415 |
| 4O | production_oracle_median | Hierarchical Organization | 0.793 | 0.102 | 0.272 | 0.200 | 0.249 |
| 4O | production_oracle_median | Representationality | 0.845 | 0.628 | 0.062 | 0.051 | 0.070 |
| 4O | production_oracle_median | Modularity | 0.971 | 0.950 | -0.555 | -0.544 | 0.062 |
| 1P | production_prior_mean | Coherence | 0.650 | 0.000 | -1.269 | -0.036 | 1.450 |
| 1P | production_prior_mean | Selective Attention | 0.650 | 0.055 | 1.163 | -0.015 | 1.079 |
| 1P | production_prior_mean | Integration | 0.803 | 0.627 | 1.163 | -0.245 | 0.763 |
| 1P | production_prior_mean | Complexity | 0.670 | 0.099 | 0.470 | 0.090 | 0.673 |
| 1P | production_prior_mean | Modularity | 0.857 | 0.613 | -0.223 | -0.160 | 0.139 |
| 1P | production_prior_mean | Hierarchical Organization | 0.799 | 0.000 | 0.080 | 0.030 | 0.105 |
| 1P | production_prior_mean | Representationality | 0.747 | 0.213 | 0.038 | 0.015 | 0.059 |
| 3 | production_prior_mean | Coherence | 0.648 | 0.000 | -1.269 | -0.036 | 1.451 |
| 3 | production_prior_mean | Selective Attention | 0.651 | 0.054 | 1.163 | -0.014 | 1.079 |
| 3 | production_prior_mean | Integration | 0.803 | 0.629 | 1.163 | -0.245 | 0.763 |
| 3 | production_prior_mean | Complexity | 0.670 | 0.100 | 0.470 | 0.090 | 0.673 |
| 3 | production_prior_mean | Modularity | 0.857 | 0.614 | -0.223 | -0.160 | 0.139 |
| 3 | production_prior_mean | Hierarchical Organization | 0.801 | 0.000 | 0.080 | 0.030 | 0.105 |
| 3 | production_prior_mean | Representationality | 0.748 | 0.215 | 0.038 | 0.015 | 0.059 |
| 4 | production_prior_mean | Coherence | 0.623 | 0.024 | -1.269 | 0.014 | 1.428 |
| 4 | production_prior_mean | Selective Attention | 0.696 | 0.169 | 1.163 | 0.005 | 0.994 |
| 4 | production_prior_mean | Complexity | 0.724 | 0.259 | 0.470 | 0.141 | 0.599 |
| 4 | production_prior_mean | Integration | 0.850 | 0.584 | 1.163 | 0.407 | 0.571 |
| 4 | production_prior_mean | Hierarchical Organization | 0.778 | 0.059 | 0.080 | 0.050 | 0.091 |

## 5. Biggest Top-Feature Losses
| top_feature_name | n | p_message_correct_sign | p_message_decisive | median_c_top_latent | median_c_oracle_subtree | mean_abs_delta_oracle_minus_top |
|---|---|---|---|---|---|---|
| Coherence | 26 | 0.577 | 0.423 | -1.492 | -0.134 | 1.296 |
| Selective Attention | 26 | 0.615 | 0.577 | -1.761 | -1.300 | 0.944 |
| Complexity | 26 | 1.000 | 0.538 | 0.841 | 0.200 | 0.683 |
| Hierarchical Organization | 26 | 0.538 | 0.000 | 0.272 | 0.089 | 0.281 |
| Integration | 26 | 1.000 | 0.885 | 1.388 | 1.348 | 0.217 |
| Representationality | 26 | 0.923 | 0.500 | 0.062 | 0.052 | 0.039 |
| Modularity | 26 | 1.000 | 1.000 | -0.555 | -0.555 | 0.000 |
Coherence, Selective Attention, and Complexity remain the primary downstream-loss features; Integration is comparatively more stable but still worth preserving because it is a strong top feature.

## 6. Edge-Profile Ablations
| edge_profile | observation_model | median_log_B_R1 | median_log_B_R0 | balanced_accuracy | evidence_margin_M | pass_fail_label | survival_R1_vs_top | abs_survival_R0_vs_top |
|---|---|---|---|---|---|---|---|---|
| all_extreme_0p9_0p1 | binary_noisy_K6 | 10.398 | -10.275 | 0.990 | 8.788 | pass | 0.676 | 0.779 |
| all_extreme_0p9_0p1 | binary_perfect_leaf | 10.397 | -10.284 | 0.990 | 8.788 | pass | 0.676 | 0.780 |
| all_production_prior_mean | binary_noisy_K6 | 0.560 | -0.638 | 0.552 | -1.049 | fail | 0.199 | 0.203 |
| all_production_prior_mean | binary_perfect_leaf | 0.564 | -0.634 | 0.552 | -1.046 | fail | 0.201 | 0.202 |
| strong_top_features_only_extreme_lower | binary_noisy_K6 | 2.509 | -2.460 | 0.831 | 0.900 | pass | 0.893 | 0.783 |
| strong_top_features_only_extreme_lower | binary_perfect_leaf | 2.509 | -2.460 | 0.831 | 0.900 | pass | 0.893 | 0.783 |
| top_extreme_lower_production | binary_noisy_K6 | 2.798 | -3.004 | 0.836 | 1.188 | pass | 0.182 | 0.228 |
| top_extreme_lower_production | binary_perfect_leaf | 2.811 | -3.005 | 0.833 | 1.201 | pass | 0.183 | 0.228 |
| top_production_lower_extreme | binary_noisy_K6 | 2.702 | -2.530 | 0.828 | 1.093 | pass | 0.962 | 0.805 |
| top_production_lower_extreme | binary_perfect_leaf | 2.694 | -2.529 | 0.828 | 1.085 | pass | 0.959 | 0.805 |
| weak_top_features_only_extreme_lower | binary_noisy_K6 | 0.570 | -0.594 | 0.555 | -1.039 | fail | 0.203 | 0.189 |
| weak_top_features_only_extreme_lower | binary_perfect_leaf | 0.570 | -0.593 | 0.554 | -1.039 | fail | 0.203 | 0.189 |
Observed pattern: all-production fails, all-extreme passes, top-production/lower-extreme passes, and strong-top-feature lower-edge strengthening passes while weak-top-feature lower-edge strengthening fails. Top-extreme/lower-production also passes, but with low survival versus its much larger top bound; this means stronger root-to-top gaps can compensate, while the production lower tree still transmits too little evidence on its own.

## 7. Information Ramp
| beta_profile | observation_model | K_or_n_raters | median_log_B_R1 | median_log_B_R0 | balanced_accuracy | evidence_margin_M | pass_fail_label |
|---|---|---|---|---|---|---|---|
| production_oracle_median | binary_noisy | 1 | 1.374 | -1.494 | 0.701 | -0.236 | fail |
| production_oracle_median | binary_noisy | 2 | 1.549 | -1.674 | 0.721 | -0.061 | fail |
| production_oracle_median | binary_noisy | 3 | 1.618 | -1.729 | 0.732 | 0.008 | fail |
| production_oracle_median | binary_noisy | 6 | 1.678 | -1.813 | 0.739 | 0.069 | fail |
| production_oracle_median | binary_noisy | 12 | 1.678 | -1.813 | 0.739 | 0.069 | fail |
| production_oracle_median | binary_noisy | 30 | 1.678 | -1.813 | 0.739 | 0.069 | fail |
| production_oracle_median | binary_noisy | 100 | 1.678 | -1.813 | 0.739 | 0.069 | fail |
| production_oracle_median | binary_noisy | 1000 | 1.678 | -1.813 | 0.739 | 0.069 | fail |
| production_oracle_median | ordinal_binary | 1 | 1.571 | -1.722 | 0.723 | -0.039 | fail |
| production_oracle_median | ordinal_binary | 2 | 1.691 | -1.815 | 0.738 | 0.081 | fail |
| production_oracle_median | ordinal_binary | 3 | 1.678 | -1.810 | 0.739 | 0.069 | fail |
| production_oracle_median | ordinal_binary | 6 | 1.678 | -1.813 | 0.739 | 0.069 | fail |
| production_oracle_median | ordinal_binary | 12 | 1.678 | -1.813 | 0.739 | 0.069 | fail |
| production_oracle_median | ordinal_binary | 30 | 1.678 | -1.813 | 0.739 | 0.069 | fail |
| production_oracle_median | ordinal_three_state | 1 | 1.938 | -1.852 | 0.754 | 0.329 | pass |
| production_oracle_median | ordinal_three_state | 2 | 2.245 | -2.088 | 0.775 | 0.636 | pass |
| production_oracle_median | ordinal_three_state | 3 | 2.350 | -2.060 | 0.789 | 0.725 | pass |
| production_oracle_median | ordinal_three_state | 6 | 2.372 | -2.265 | 0.803 | 0.763 | pass |
| production_oracle_median | ordinal_three_state | 12 | 2.396 | -2.296 | 0.800 | 0.787 | pass |
| production_oracle_median | ordinal_three_state | 30 | 2.401 | -2.295 | 0.798 | 0.791 | pass |
| production_oracle_median | perfect_leaf | perfect | 1.678 | -1.813 | 0.739 | 0.069 | fail |
| production_prior_mean | binary_noisy | 1 | 0.454 | -0.485 | 0.529 | -1.156 | fail |
| production_prior_mean | binary_noisy | 2 | 0.490 | -0.543 | 0.539 | -1.120 | fail |
| production_prior_mean | binary_noisy | 3 | 0.559 | -0.616 | 0.544 | -1.051 | fail |
| production_prior_mean | binary_noisy | 6 | 0.560 | -0.633 | 0.552 | -1.050 | fail |
| production_prior_mean | binary_noisy | 12 | 0.564 | -0.634 | 0.552 | -1.046 | fail |
| production_prior_mean | binary_noisy | 30 | 0.564 | -0.634 | 0.552 | -1.046 | fail |
| production_prior_mean | binary_noisy | 100 | 0.564 | -0.634 | 0.552 | -1.046 | fail |
| production_prior_mean | binary_noisy | 1000 | 0.564 | -0.634 | 0.552 | -1.046 | fail |
| production_prior_mean | ordinal_binary | 1 | 0.551 | -0.594 | 0.544 | -1.059 | fail |
| production_prior_mean | ordinal_binary | 2 | 0.563 | -0.630 | 0.549 | -1.047 | fail |
| production_prior_mean | ordinal_binary | 3 | 0.561 | -0.634 | 0.552 | -1.049 | fail |
| production_prior_mean | ordinal_binary | 6 | 0.564 | -0.634 | 0.552 | -1.046 | fail |
| production_prior_mean | ordinal_binary | 12 | 0.564 | -0.634 | 0.552 | -1.046 | fail |
| production_prior_mean | ordinal_binary | 30 | 0.564 | -0.634 | 0.552 | -1.046 | fail |
| production_prior_mean | ordinal_three_state | 1 | 0.550 | -0.493 | 0.566 | -1.060 | fail |
| production_prior_mean | ordinal_three_state | 2 | 0.621 | -0.585 | 0.589 | -0.988 | fail |
| production_prior_mean | ordinal_three_state | 3 | 0.765 | -0.604 | 0.596 | -0.845 | fail |
| production_prior_mean | ordinal_three_state | 6 | 0.820 | -0.682 | 0.608 | -0.789 | fail |
| production_prior_mean | ordinal_three_state | 12 | 0.791 | -0.676 | 0.605 | -0.818 | fail |
| production_prior_mean | ordinal_three_state | 30 | 0.794 | -0.675 | 0.606 | -0.816 | fail |
| production_prior_mean | perfect_leaf | perfect | 0.564 | -0.634 | 0.552 | -1.046 | fail |

## 8. Leaf Information Map
Top 10 leaves by binary symmetric KL:
| beta_profile | leaf_name | top_feature_name | binary_gap | sym_kl_binary | product_path_gap_heuristic |
|---|---|---|---|---|---|
| extreme_0p9_0p1 | Holistic Dependency | Integration | 0.640 | 0.970 | 0.640 |
| extreme_0p9_0p1 | Concrete-Abstract Separation | Hierarchical Organization | 0.640 | 0.970 | 0.640 |
| extreme_0p9_0p1 | Selective Competence Disruption | Modularity | 0.640 | 0.970 | 0.640 |
| extreme_0p9_0p1 | Poorly Intraconnected Networks | Modularity | 0.640 | 0.970 | 0.640 |
| extreme_0p9_0p1 | Dedicated Speech Systems | Modularity | 0.640 | 0.970 | 0.640 |
| extreme_0p9_0p1 | Dedicated Action Systems | Modularity | 0.640 | 0.970 | 0.640 |
| extreme_0p9_0p1 | Dedicated Sensory Systems | Modularity | 0.640 | 0.970 | 0.640 |
| extreme_0p9_0p1 | Cross-Modal Learning Deficits | Modularity | 0.640 | 0.970 | 0.640 |
| extreme_0p9_0p1 | Wiring Convergence | Integration | 0.640 | 0.970 | 0.640 |
| extreme_0p9_0p1 | Broad Predictive Training | Representationality | 0.512 | 0.579 | 0.512 |
Bottom 10 leaves by binary symmetric KL:
| beta_profile | leaf_name | top_feature_name | binary_gap | sym_kl_binary | product_path_gap_heuristic |
|---|---|---|---|---|---|
| production_prior_mean | Broad Predictive Training | Representationality | 0.001 | 0.000 | 0.001 |
| production_prior_mean | Somatotopic Neural Maps | Representationality | 0.002 | 0.000 | 0.002 |
| production_prior_mean | Has Retinotopic Neural Maps | Representationality | 0.003 | 0.000 | 0.003 |
| production_prior_mean | Activation Steering Effects | Representationality | 0.004 | 0.000 | 0.004 |
| production_oracle_median | Information Transfer | Coherence | -0.005 | 0.000 | -0.005 |
| production_prior_mean | Probe-Detectable Themes | Representationality | 0.006 | 0.000 | 0.006 |
| production_prior_mean | Variability of Responses to Stimuli | Complexity | 0.006 | 0.000 | 0.006 |
| production_oracle_median | Functional Subparts | Coherence | -0.008 | 0.000 | -0.008 |
| production_prior_mean | Information Transfer | Coherence | -0.007 | 0.000 | -0.007 |
| production_oracle_median | Broad Predictive Training | Representationality | 0.008 | 0.000 | 0.008 |
Top 10 leaves by ordinal reference-rater symmetric KL:
| beta_profile | leaf_name | top_feature_name | sym_kl_ordinal_ref_rater | product_path_gap_heuristic |
|---|---|---|---|---|
| extreme_0p9_0p1 | Holistic Dependency | Integration | 1.252 | 0.640 |
| extreme_0p9_0p1 | Concrete-Abstract Separation | Hierarchical Organization | 1.252 | 0.640 |
| extreme_0p9_0p1 | Selective Competence Disruption | Modularity | 1.252 | 0.640 |
| extreme_0p9_0p1 | Poorly Intraconnected Networks | Modularity | 1.252 | 0.640 |
| extreme_0p9_0p1 | Dedicated Speech Systems | Modularity | 1.252 | 0.640 |
| extreme_0p9_0p1 | Dedicated Action Systems | Modularity | 1.252 | 0.640 |
| extreme_0p9_0p1 | Dedicated Sensory Systems | Modularity | 1.252 | 0.640 |
| extreme_0p9_0p1 | Cross-Modal Learning Deficits | Modularity | 1.252 | 0.640 |
| extreme_0p9_0p1 | Wiring Convergence | Integration | 1.252 | 0.640 |
| extreme_0p9_0p1 | Broad Predictive Training | Representationality | 0.757 | 0.512 |
Bottom 10 leaves by ordinal reference-rater symmetric KL:
| beta_profile | leaf_name | top_feature_name | sym_kl_ordinal_ref_rater | product_path_gap_heuristic |
|---|---|---|---|---|
| production_prior_mean | Broad Predictive Training | Representationality | 0.000 | 0.001 |
| production_prior_mean | Somatotopic Neural Maps | Representationality | 0.000 | 0.002 |
| production_prior_mean | Has Retinotopic Neural Maps | Representationality | 0.000 | 0.003 |
| production_prior_mean | Activation Steering Effects | Representationality | 0.000 | 0.004 |
| production_oracle_median | Information Transfer | Coherence | 0.000 | -0.005 |
| production_prior_mean | Probe-Detectable Themes | Representationality | 0.000 | 0.006 |
| production_prior_mean | Variability of Responses to Stimuli | Complexity | 0.000 | 0.006 |
| production_oracle_median | Functional Subparts | Coherence | 0.000 | -0.008 |
| production_prior_mean | Information Transfer | Coherence | 0.000 | -0.007 |
| production_oracle_median | Broad Predictive Training | Representationality | 0.000 | 0.008 |

## 9. Revised Hypothesis Ranking
1. Lower-depth transmission / subtree observability under production beta.
2. Finite observation density, if the information ramp only recovers at high K or many crossed raters.
3. Nuisance uncertainty / beta identifiability, only after no-fit stages pass.
4. Rater design, likely important for Chicken posterior disagreements but not the first cause of R=1 oracle ambiguity.
5. Ordinal/three-state emission specifically, because binary noisy leaves already fail under production beta.

## 10. Before HMC Fit Rungs
Do not start full HMC yet. Inspect lower-tree support/demandingness labels, strengthen or redesign weak lower-depth paths, and use the information ramp to decide whether survey density can rescue production beta. Fit rungs should wait until no-fit stages have a passable evidence margin.

## Commands
```bash
.venv/bin/python scripts/phase1c_evidence_attrition_audit.py --output-dir outputs/phase1_root_evidence --n-rep 500 --seed 20260511 --mode all
.venv/bin/python scripts/phase1c_evidence_attrition_audit.py --output-dir outputs/phase1_root_evidence --n-rep 20 --seed 20260511 --mode smoke
```
