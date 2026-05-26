# Human PPC Conditioning Audit Report

Decision: `READY_FOR_50_SEED_FULL_VALIDATION`

## Executive Summary

This audit used only existing 10-seed pilot artefacts. No model code was changed, no HMC was run, and the 50-seed validation was not launched.

The prior `truth_generator_probs_1_to_7` diagnostic was not a posterior predictive target. It used true continuous parameters plus the realised synthetic indicator state, so it was a `realised_latent_predictive` probability. It did not condition on the observed ratings and did not integrate tree latents. Comparing fitted PPC probabilities directly to that quantity can overstate Human PPC mismatch.

The conditioning-correct oracle PPC target was computed for Human review seeds 20260513 and 20260518 by exact DP ratios: `p(y_obs plus y_rep=k | theta*) / p(y_obs | theta*)`, with Human root fixed present and true continuous parameters/edge betas. Against this target, the review-seed weighted mean TV falls from 0.224 versus current truth to 0.070 versus oracle PPC.

The Human blocker is therefore a PPC diagnostic conditioning artefact rather than a confirmed model failure. The revised fit-vs-oracle status is `pass`. Human nuisance recovery shows no severe a/kappa/b issue in the review seeds; rater shifts are fixed at zero in this fit because no `b`/`b_free` posterior variable is present.

## Probability Semantics

| probability_source | conditioning_set | uses_observed_y | uses_realised_latents | uses_true_continuous_params | integrates_tree_latents | comparable_to_fitted_ppc | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| truth_generator_probs_1_to_7 | true continuous parameters plus realised synthetic leaf state m_j | False | True | True | False | False | Computed from truth_payload latent_by_system[system][indicator_m] via ordered probit; this is realised-latent predictive, not PPC-conditioned. |
| fitted_ppc_probs_1_to_7 | observed ratings and posterior draws from fitted model | True | False | False | True | True | Posterior predictive mean category probability from posterior_predictive.nc. |
| p_oracle_ppc_1_to_7 | observed ratings, true continuous parameters, true edge betas, Human root fixed present | True | False | True | True | True | Computed by exact DP ratio p(y_obs plus pseudo replicated category k given theta*) divided by p(y_obs given theta*) and normalized. |

## Fit vs Oracle PPC Summary

| row_type | seed | n_cells | old_n_fitted_truth_mismatch | revised_n_genuine_fit_oracle_mismatch | revised_n_moderate_fit_oracle_warning | revised_n_ppc_diagnostic_artifact | revised_n_sparse_observation_noise | weighted_mean_tv_fit_vs_current_truth | weighted_mean_tv_fit_vs_oracle_ppc | median_tv_fit_vs_oracle_ppc | q90_tv_fit_vs_oracle_ppc | max_tv_fit_vs_oracle_ppc | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seed_summary | 20260513 | 50 | 23 | 0 | 0 | 2 | 30 | 0.2456 | 0.09202 | 0.09674 | 0.1184 | 0.1452 | pass |
| seed_summary | 20260518 | 50 | 21 | 0 | 0 | 3 | 30 | 0.2028 | 0.04822 | 0.04586 | 0.09548 | 0.1226 | pass |
| overall_review_summary | review_seeds_20260513_20260518 | 100 | 44 | 0 | 0 | 5 | 60 | 0.2242 | 0.07012 | 0.07314 | 0.1183 | 0.1452 | pass |

Revised cell labels:

| issue_label_revised | n_cells | mean_tv_fit_vs_oracle_ppc | mean_tv_fit_vs_current_truth |
| --- | --- | --- | --- |
| sparse_observation_noise | 60 | 0.07846 | 0.3014 |
| no_specific_issue | 35 | 0.05896 | 0.07627 |
| ppc_diagnostic_artifact | 5 | 0.0482 | 0.3335 |

Largest fit-vs-oracle PPC mismatches:

| seed | indicator_name | observed_rating | tv_fit_vs_current_truth | tv_fit_vs_oracle_ppc | tv_current_truth_vs_oracle_ppc | high_fit_P_y_ge_6 | high_current_truth_P_y_ge_6 | high_oracle_ppc_P_y_ge_6 | issue_label_old | issue_label_revised |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260513 | Poorly Intraconnected Networks | 4 | 0.46 | 0.1452 | 0.5908 | 0.1361 | 0.00116 | 0.15 | fitted_truth_mismatch | sparse_observation_noise |
| 20260513 | Functional Subparts | 2 | 0.1941 | 0.1306 | 0.0659 | 0.3141 | 0.1643 | 0.2207 | no_specific_issue | sparse_observation_noise |
| 20260513 | Perspective-relative Representations | 2 | 0.5403 | 0.1276 | 0.4696 | 0.2488 | 0.00116 | 0.1443 | fitted_truth_mismatch | sparse_observation_noise |
| 20260513 | Stable Personality | 3 | 0.584 | 0.124 | 0.5356 | 0.3125 | 0.00116 | 0.1923 | fitted_truth_mismatch | sparse_observation_noise |
| 20260518 | Adaptive Learning | 4 | 0.5976 | 0.1226 | 0.5453 | 0.2649 | 0.8625 | 0.3172 | fitted_truth_mismatch | sparse_observation_noise |
| 20260513 | Diversity of Tasks | 7 | 0.1271 | 0.1184 | 0.01033 | 0.7672 | 0.8625 | 0.8522 | sparse_cell_unstable | no_specific_issue |
| 20260513 | Information Transfer | 7 | 0.1267 | 0.1184 | 0.009892 | 0.7678 | 0.8625 | 0.8526 | sparse_cell_unstable | no_specific_issue |
| 20260513 | Stable Personality | 7 | 0.1267 | 0.1183 | 0.009895 | 0.7678 | 0.8625 | 0.8526 | sparse_cell_unstable | no_specific_issue |
| 20260513 | Conflicting Subparts | 7 | 0.1266 | 0.1183 | 0.009892 | 0.7679 | 0.8625 | 0.8526 | sparse_cell_unstable | no_specific_issue |
| 20260513 | Unified Egocentric Representations | 7 | 0.1268 | 0.1183 | 0.01009 | 0.7677 | 0.8625 | 0.8524 | sparse_cell_unstable | no_specific_issue |
| 20260513 | Persistence Seeking | 7 | 0.1266 | 0.1182 | 0.009895 | 0.768 | 0.8625 | 0.8526 | sparse_cell_unstable | no_specific_issue |
| 20260513 | Number of Nodes | 7 | 0.1265 | 0.1182 | 0.009892 | 0.7681 | 0.8625 | 0.8526 | sparse_cell_unstable | no_specific_issue |
| 20260513 | Number of Connections | 7 | 0.1265 | 0.1182 | 0.009892 | 0.7681 | 0.8625 | 0.8526 | sparse_cell_unstable | no_specific_issue |
| 20260513 | Functional Specialization | 7 | 0.1265 | 0.1182 | 0.009892 | 0.7681 | 0.8625 | 0.8526 | sparse_cell_unstable | no_specific_issue |
| 20260513 | Goal Focus Shifts | 7 | 0.1264 | 0.1181 | 0.009895 | 0.7681 | 0.8625 | 0.8526 | sparse_cell_unstable | no_specific_issue |

## Human Nuisance Recovery

Compact review-seed nuisance summary:

| seed | a_error | max_kappa_abs_error | max_abs_b_error_for_human_raters | mean_abs_b_error_for_human_raters | n_human_rater_truth_outside_90pct_interval | n_key_beta_truth_outside_90pct_interval | n_key_beta_audited | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20260513 | 0.4893 | 0.1897 | 0 | 0 | 0 | 0 | 100 | fixed_zero_no_b_posterior_variable |
| 20260518 | 0.2656 | 0.4246 | 0 | 0 | 0 | 0 | 100 | fixed_zero_no_b_posterior_variable |

Interpretation: there is no severe Human nuisance recovery signal in the continuous observation layer. The beta rows are best-effort edge checks and are not used as a hard decision gate here. The fitted-vs-oracle PPC comparison passes despite the old fitted-vs-realised-latent truth mismatch labels.

## Decision

| decision | current_truth_probability_semantics | current_truth_comparable_to_fitted_ppc | review_weighted_mean_tv_fit_vs_current_truth | review_weighted_mean_tv_fit_vs_oracle_ppc | review_median_tv_fit_vs_oracle_ppc | review_q90_tv_fit_vs_oracle_ppc | review_max_tv_fit_vs_oracle_ppc | review_revised_n_genuine_fit_oracle_mismatch | review_revised_n_moderate_fit_oracle_warning | review_revised_n_sparse_observation_noise | fit_vs_oracle_ppc_status | human_nuisance_recovery_severe_issue | sampler_pass_from_pilot | loo_pass_from_pilot | recommended_command |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| READY_FOR_50_SEED_FULL_VALIDATION | realised_latent_predictive | False | 0.2242 | 0.07012 | 0.07314 | 0.1183 | 0.1452 | 0 | 0 | 60 | pass | False | True | True | .venv/bin/python scripts/main_synthetic_validation.py \
  --output-dir outputs/main_synthetic_validation \
  --mode full \
  --fit-variants targeted_strong_lower_override \
  --n-seeds 50 \
  --seed 20260511 |

Recommended command:

```bash
.venv/bin/python scripts/main_synthetic_validation.py \
  --output-dir outputs/main_synthetic_validation \
  --mode full \
  --fit-variants targeted_strong_lower_override \
  --n-seeds 50 \
  --seed 20260511
```
