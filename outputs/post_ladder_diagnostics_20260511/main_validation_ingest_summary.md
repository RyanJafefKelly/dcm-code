# Main Validation Ingest Summary

Input directory: `outputs/main_synthetic_validation`
Output directory: `outputs/post_ladder_diagnostics_20260511`

## Detected Status

- detected_status: `smoke_complete`
- validation_scope: `smoke`
- overall targeted baseline status in report: `FAIL`
- fit directories: `2`; completed artefact sets: `2`; failed/partial fit artefact sets: `0`
- fit variants: `targeted_strong_lower_override`
- seeds: `20260511, 20260512`
- missing required outputs: `False`

No error logs, traceback files, or explicit failed/partial run marker files were found. `pass_fail_summary.csv` exists and contains failed validation checks; LOO is explicitly partial.

## Summary Files

- `validation_manifest.csv`: `True`
- `validation_cases.csv`: `True`
- `rho_recovery_summary.csv`: `True`
- `posterior_predictive_summary.csv`: `True`
- `loo_lppd_summary.csv`: `True`
- `sampler_diagnostics_summary.csv`: `True`
- `pass_fail_summary.csv`: `True`
- `main_synthetic_validation_report.md`: `True`

## Run Artefacts

- `outputs/main_synthetic_validation/runs/targeted_strong_lower_override/seed_20260511`: fit.nc=True, posterior_predictive.nc=True, per_rating_log_lik=nc, run_summary=True, config=True, truth_payload=True
- `outputs/main_synthetic_validation/runs/targeted_strong_lower_override/seed_20260512`: fit.nc=True, posterior_predictive.nc=True, per_rating_log_lik=nc, run_summary=True, config=True, truth_payload=True

## Root Recovery

| system | n_cases | evidence_margin_M | balanced_accuracy | brier_improvement | log_score_improvement | mean_rho_R1 | mean_rho_R0 | ECE | decisive_present_R1 | decisive_absent_R0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 4 | 0.479956 | 0.75 | 0.127587 | 0.134453 | 0.510233 | 0.00207576 | 0.243845 | 0.5 | 1 |
| Chicken | 2 | 2.70646 | 1 | 0.361019 | 0.978672 | 0.986858 | 0.00350204 | 0.00832217 | 1 | 1 |
| LLMs | 2 | -3.35878 | 0.5 | -0.105845 | -0.709767 | 0.0336089 | 0.000649487 | 0.482871 | 0 | 1 |

Chicken split passes root recovery on this smoke sample. LLMs split fails root recovery diagnostics: balanced accuracy is 0.5, evidence_margin_M is negative, and Brier/log-score improvements are negative.
Decisive rates are diagnostic only because this is a four-case smoke run.

## Posterior Predictive

| system | n_ratings | mean_RPS | empirical_baseline | RPS_improvement | weighted_mean_cell_TV | median_cell_TV | q90_cell_TV | frac_TV_gt_0.5 | PPC_TV_p | mean_top7_high_q_err | q90_top7_high_q_err | mean_high_q_err | q90_high_q_err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | 379 | 0.0721754 | 0.230902 | 0.158727 | 0.394283 | 0.375778 | 0.83702 | 0.306533 | 1 | 0.324437 | 0.46752 | 0.201102 | 0.311912 |
| Chicken | 93 | 0.0619204 | 0.228917 | 0.166997 | 0.402682 | 0.3717 | 0.706871 | 0.22449 | missing | 0.246608 | 0.36579 | 0.183097 | 0.311724 |
| ELIZA | 50 | 0.0596317 | 0.217128 | 0.157496 | 0.510462 | 0.419114 | 0.899735 | 0.42 | missing | 0.438516 | 0.549202 | 0.300774 | 0.546272 |
| Human | 50 | 0.0613442 | 0.247361 | 0.186017 | 0.548937 | 0.499353 | 0.892634 | 0.42 | missing | 0.399079 | 0.493697 | 0.220692 | 0.418048 |
| LLMs | 186 | 0.0835864 | 0.231173 | 0.147586 | 0.317279 | 0.285449 | 0.562144 | 0.16 | missing | 0.174264 | 0.285651 | 0.13965 | 0.145303 |

Overall PPC fails TV diagnostics despite positive mean RPS improvement. The ALL row fails weighted_mean_cell_TV, median_cell_TV, q90_cell_TV, fraction_cell_TV_gt_0p5, and PPC TV p-value thresholds. System-specific p-values were not computed; system TV diagnostics are high for Chicken, ELIZA, Human, and LLMs.

## LOO/lppd

- availability: `partial` (`loo_partial_flag=1`)
- n_ratings_used: `50` of `379`
- loo_draws: `100`
- elpd_loo: `-61.9538`; p_loo: `1.84721`
- Pareto k mean/median/max: `0.156901` / `0.15363` / `0.586075`
- frac_pareto_k_gt_0p7: `0`; frac_pareto_k_gt_1p0: `0`
- reliability: Pareto-k diagnostics are acceptable for the evaluated subset, but LOO is not a reliable full-run estimate because it used only 50 of 379 ratings.

## Sampler

- max_rhat: `1.05`
- n_rhat_gt_1p01: `95`; n_rhat_gt_1p05: `0`
- min_ess_bulk: `177`; min_ess_tail: `56`
- n_divergences: `0`
- max_tree_depth_hit_rate: `0`
- runtime_seconds from top-level sampler summary: `1945.03`
- posterior summaries trustworthy for acceptance: `False`

Sampler failures are Rhat/ESS-related, not divergence or tree-depth related. These posterior summaries should be treated as diagnostics, not acceptance evidence.

## Failed Validation Checks

- `posterior_predictive` / `weighted_mean_cell_TV`: value `0.39428308282569136` vs threshold `<= 0.25`
- `posterior_predictive` / `median_cell_TV`: value `0.37577814703853885` vs threshold `<= 0.25`
- `posterior_predictive` / `q90_cell_TV`: value `0.8370203540019212` vs threshold `<= 0.50`
- `posterior_predictive` / `fraction_cell_TV_gt_0p5`: value `0.30653266331658285` vs threshold `<= 0.10`
- `posterior_predictive` / `posterior_predictive_tv_p_value`: value `1.0` vs threshold `in [0.05, 0.95]`
- `sampler_health` / `fit_pass_rate_rhat`: value `0.0` vs threshold `>= 0.90`
- `sampler_health` / `fit_pass_rate_ess_bulk`: value `0.0` vs threshold `>= 0.90`
- `sampler_health` / `fit_pass_rate_ess_tail`: value `0.0` vs threshold `>= 0.90`

## Metadata Consistency Caveat

- seed `20260511`: max_rhat: run_summary=1.01, top_summary=1.04; n_rhat_gt_1p01: run_summary=0.0, top_summary=38.0; min_ess_bulk: run_summary=2679.0, top_summary=247.0; min_ess_tail: run_summary=1569.0, top_summary=80.0; runtime_seconds: run_summary=2210.085331916809, top_summary=1044.3279082775116

The seed_20260511 run_summary/fit artefacts are newer than the top-level summaries and disagree with top-level sampler fields, so a later non-sampling summary refresh appears to be the next summarisation step if the current outputs need to be made internally consistent.

## Safe Later Actions From Existing Artefacts

- metrics-only later: feasible from existing artefacts, but not run now.
- plots-only later: feasible and plots already exist, but not run now.
- loo-only later: feasible only if configured to reuse `per_rating_log_lik.nc` without sampling/refitting; not run now.

## Confirmation

No HMC, sampling, refitting, metrics-only, plots-only, or LOO recomputation was launched by this ingest pass.
