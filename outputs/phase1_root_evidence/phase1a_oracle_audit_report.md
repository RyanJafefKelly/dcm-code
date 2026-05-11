# Phase 1A Oracle Root-Evidence Audit

Rows audited: 26 free-target system cases across 13 fits.

## 1. Top-Level Bound
| beta_profile | mean_log_B_top_R1 | mean_log_B_top_R0 | median_log_B_top_R1 | median_log_B_top_R0 | frac_R1_log_B_top_gt_tau_present_50 | frac_R0_log_B_top_lt_tau_absent_05 | preflight_gate |
|---|---|---|---|---|---|---|---|
| production_prior_mean | 2.811 | -2.807 | 2.809 | -3.142 | 0.755 | 0.718 | pass |
| production_oracle_median | 4.240 | -4.532 | 4.839 | -4.395 | 0.877 | 0.840 | pass |
| extreme_0p9_0p1 | 12.304 | -12.304 | 10.986 | -10.986 | 0.997 | 0.997 | pass |

Pre-flight gate status: production_prior_mean=pass, production_oracle_median=pass, extreme_0p9_0p1=pass. The full median and tail distribution, not the mean alone, is the relevant bound.

## 2. Summary By System And Root
| run_family | system | root_z_true | n | median_log_B_top_latent | median_log_B_oracle | median_log_B_eff_posterior | p_correct_sign_oracle | p_correct_sign_posterior | p_decisive_oracle | p_decisive_posterior |
|---|---|---|---|---|---|---|---|---|---|---|
| asymmetric_prior_sweep | Chicken | 0 | 3 | -4.113 | -3.011 | -0.964 | 1.000 | 1.000 | 1.000 | 0.333 |
| asymmetric_prior_sweep | LLMs | 1 | 3 | 5.458 | 0.174 | -0.543 | 1.000 | 0.333 | 0.000 | 0.000 |
| full_exact_recovery | Chicken | 0 | 5 | -4.113 | -3.011 | -1.635 | 1.000 | 1.000 | 0.800 | 0.600 |
| full_exact_recovery | LLMs | 0 | 3 | -2.964 | -0.838 | -0.408 | 0.667 | 0.667 | 0.333 | 0.333 |
| full_exact_recovery | LLMs | 1 | 2 | 5.458 | 0.174 | 1.073 | 1.000 | 1.000 | 0.000 | 0.000 |
| sample_size_sweep | Chicken | 0 | 5 | -4.113 | -2.977 | 0.137 | 1.000 | 0.400 | 1.000 | 0.000 |
| sample_size_sweep | LLMs | 1 | 5 | 5.458 | -0.004 | 1.577 | 0.200 | 0.800 | 0.000 | 0.000 |

## 3. Evidence Categories
| system | root_z_true | oracle_category | posterior_category | n |
|---|---|---|---|---|
| Chicken | 0 | moderate_absent | ambiguous | 4 |
| Chicken | 0 | strong_absent | ambiguous | 4 |
| Chicken | 0 | strong_absent | moderate_absent | 2 |
| Chicken | 0 | strong_absent | strong_absent | 2 |
| Chicken | 0 | ambiguous | ambiguous | 1 |
| LLMs | 0 | ambiguous | ambiguous | 2 |
| LLMs | 0 | strong_absent | moderate_absent | 1 |
| LLMs | 1 | ambiguous | ambiguous | 9 |
| LLMs | 1 | ambiguous | moderate_absent | 1 |

## 4. Oracle Versus Posterior Disagreements
Large disagreements (>=2 nats or sign flip): 15. Severe disagreements (>=4 nats): 0.
| run_family | seed | system | root_z_true | log_B_top_latent | log_B_oracle | log_B_eff_posterior | delta_eff_minus_oracle | delta_eff_minus_top | sign_flip_oracle_vs_posterior |
|---|---|---|---|---|---|---|---|---|---|
| sample_size_sweep | 20260506 | Chicken | 0 | -4.113 | -2.960 | 0.300 | 3.260 | 4.413 | True |
| sample_size_sweep | 20260506 | Chicken | 0 | -4.113 | -2.960 | 0.272 | 3.232 | 4.386 | True |
| sample_size_sweep | 20260506 | Chicken | 0 | -4.113 | -2.977 | 0.137 | 3.114 | 4.250 | True |
| sample_size_sweep | 20260506 | Chicken | 0 | -4.113 | -3.095 | -0.083 | 3.012 | 4.030 | False |
| asymmetric_prior_sweep | 20260506 | LLMs | 1 | 5.458 | 0.174 | -2.831 | -3.005 | -8.289 | True |
| asymmetric_prior_sweep | 20260506 | Chicken | 0 | -4.113 | -3.011 | -0.097 | 2.914 | 4.016 | False |
| full_exact_recovery | 20260509 | LLMs | 0 | -7.263 | -4.231 | -1.688 | 2.543 | 5.575 | False |
| asymmetric_prior_sweep | 20260506 | Chicken | 0 | -4.113 | -3.011 | -0.964 | 2.047 | 3.149 | False |
| sample_size_sweep | 20260506 | Chicken | 0 | -4.113 | -3.011 | -0.964 | 2.047 | 3.149 | False |
| full_exact_recovery | 20260507 | Chicken | 0 | -5.832 | -4.657 | -3.038 | 1.619 | 2.794 | False |

## 5. Top-Feature Evidence Loss
| top_feature_name | support_label | demandingness_label | n | median_c_top_latent | median_c_oracle_subtree | median_delta_oracle_minus_top | mean_abs_delta | median_observed_ratings_desc |
|---|---|---|---|---|---|---|---|---|
| Coherence | strong support | strongly demanding | 26 | -1.492 | -0.134 | 0.202 | 1.296 | 48.000 |
| Selective Attention | strong support | moderately demanding | 26 | -1.761 | -1.300 | 0.071 | 0.944 | 43.000 |
| Complexity | strong support | weakly undemanding | 26 | 0.841 | 0.200 | -0.131 | 0.683 | 30.000 |
| Hierarchical Organization | weak support | moderately undemanding | 26 | 0.272 | 0.089 | -0.086 | 0.281 | 3.000 |
| Integration | strong support | moderately demanding | 26 | 1.388 | 1.348 | -0.040 | 0.217 | 8.000 |
| Representationality | weak support | strongly undemanding | 26 | 0.062 | 0.052 | -0.011 | 0.039 | 30.000 |
| Modularity | weak support | moderately demanding | 26 | -0.555 | -0.555 | 0.000 | 0.000 | 24.000 |

## 6. Chicken Versus LLMs
| system | n | mean_n_raters | median_log_B_oracle | median_log_B_eff_posterior | p_correct_sign_posterior | p_decisive_posterior |
|---|---|---|---|---|---|---|
| Chicken | 13 | 2.000 | -3.011 | -0.421 | 0.769 | 0.308 |
| LLMs | 13 | 4.000 | 0.174 | 0.489 | 0.692 | 0.077 |

Chicken remains the key design comparison because the production-style design has no Chicken cross-system rater coverage, while LLMs have Rater_B-linked coverage.

## 7. Asymmetric Prior Sweep Interpretation
The existing asymmetric-prior synthetic sweep changed the fitted beta-prior centre while the generator truth still came from production median edge betas. It is therefore a prior-DGP mismatch test, not evidence that true beta=(0.90,0.10) fails. Phase 1B adds the matched extreme-beta positive-control ladder.

## Output Files
- `phase1a_oracle_audit_cases.csv`
- `phase1a_oracle_audit_summary.csv`
- `phase1a_disagreements.csv`
- `phase1a_top_feature_contributions.csv`
