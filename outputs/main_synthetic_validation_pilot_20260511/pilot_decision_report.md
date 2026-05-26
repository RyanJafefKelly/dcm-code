# Targeted 10-Seed Pilot Decision Report

Output directory: `/Users/ryankelly/ryan-code/research/dcm-code/outputs/main_synthetic_validation_pilot_20260511`

## Decision

The 10-seed `targeted_strong_lower_override` pilot completed all requested fits. Root recovery and sampler diagnostics pass the pilot gates; LOO is full and reliable on the evaluated diagnostics. PPC legacy TV gates fail materially despite positive RPS improvement, so the recommended next action is `debug_ppc_observation_layer` before any full validation run.

Recommended next action: `debug_ppc_observation_layer`.

Do not proceed directly to full targeted validation until the PPC failure is understood or explicitly accepted as a finite-sample/cell-sparsity artefact with a stronger calibration check.

## Required Answers

- **1. Did the 10-seed targeted pilot complete?** yes
- **2. How many fits completed, failed, or were skipped as incomplete?** 10 completed, 0 failed, 0 skipped/incomplete
- **3. Did targeted override sanity pass?** yes; pass_fail=pass, overridden_edges=49, root_to_top=0, weak_top=0, generator/fitter diffs=0.0
- **4. Did root recovery pass overall?** yes; M=1.478, balanced_accuracy=0.800, Brier improvement=0.203, log-score improvement=0.466, mean rho gap=0.590
- **5. Did Chicken pass separately?** yes; balanced_accuracy=0.900
- **6. Did LLMs pass separately?** yes by the specified split gate; balanced_accuracy=0.700. It remains weak: median_log_B_eff_R1=-0.848 and evidence_margin_M=-2.458.
- **7. Were any false negatives oracle-negative rather than HMC-only sign flips?** yes; 3 false negatives, all oracle-negative; HMC-only sign flips=0.
- **8. Did sampler diagnostics pass?** yes; total divergences=0, max_rhat=1.01, min_ess_bulk=1904, min_ess_tail=882, max_tree_depth_hit_rate=0.000
- **9. Did PPC pass, warn, or fail?** fail on legacy TV gates; mean_RPS improves by 0.159 vs empirical marginal, but weighted_mean_cell_TV=0.388, median_cell_TV=0.376, q90_cell_TV=0.828, fraction_cell_TV_gt_0p5=0.322, p-value=0.9952.
- **10. Is LOO full, partial, skipped, or failed?** full; n_ratings_used=379/379, loo_draws=500, max_pareto_k=0.490, frac_k_gt_0p7=0.000, frac_k_gt_1p0=0.000
- **11. Are refreshed summaries internally consistent?** yes; refresh consistency checks all true, summary_stale_before_refresh=False.
- **12. Recommended next action?** `debug_ppc_observation_layer`

## Root Recovery

| System | n | mean_rho_R1 | mean_rho_R0 | M | balanced accuracy | Brier improvement | log-score improvement | Pass note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ALL | 20 | 0.601 | 0.011 | 1.478 | 0.800 | 0.203 | 0.466 | passes hard overall gates |
| Chicken | 10 | 0.866 | 0.019 | 1.203 | 0.900 | 0.333 | 0.888 | passes split gate |
| LLMs | 10 | 0.335 | 0.002 | -2.458 | 0.700 | 0.072 | 0.044 | passes split gate, weak positive recovery |

## Oracle/HMC Bridge

There were no HMC-only sign flips. The three LLM false negatives were also oracle-negative synthetic draws.

| Seed | System | root_z_true | log_B_oracle | rho_oracle | log_B_eff_HMC | rho_HMC | oracle_correct | posterior_correct | HMC-only flip |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| 20260512 | LLMs | 1 | -1.568 | 0.040 | -1.656 | 0.037 | False | False | False |
| 20260514 | LLMs | 1 | -1.090 | 0.063 | -0.848 | 0.079 | False | False | False |
| 20260520 | LLMs | 1 | -1.260 | 0.054 | -1.266 | 0.053 | False | False | False |

Bridge summary:

| System | cases | R1 cases | oracle correct | posterior correct | HMC-only flips | R1 oracle-negative | max abs delta log_B |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chicken | 10 | 5 | 10 | 10 | 0 | 0 | 0.906 |
| LLMs | 10 | 5 | 7 | 7 | 0 | 3 | 1.104 |

## Sampler

Sampler hard pass: `true`.

| Metric | Value | Gate |
|---|---:|---|
| completed fits | 10 | 10 |
| total divergences | 0 | == 0 |
| max R-hat across fits | 1.01 | <= 1.01 |
| min ESS bulk across fits | 1904 | >= 400 |
| min ESS tail across fits | 882 | >= 200 |
| max tree-depth hit rate | 0.000 | <= 0.01 |

## PPC

PPC fails the legacy TV gates. RPS improves over the empirical-marginal baseline for all systems, but the cell-TV metrics are well above gate thresholds and every system has `n_cells_ge_5=0`, so the failure is entangled with sparse cells. The pilot output does not contain oracle-calibrated finite-sample fields beyond the warning label; do not suppress this failure.

| System | mean_RPS | empirical baseline | RPS improvement | weighted TV | median TV | q90 TV | frac TV > 0.5 | n_cells_ge_5 | interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ALL | 0.067 | 0.226 | 0.159 | 0.388 | 0.376 | 0.828 | 0.322 | 0 | legacy_tv_fail_oracle_calibrated_warning; p=0.9952 |
| Chicken | 0.066 | 0.234 | 0.169 | 0.408 | 0.370 | 0.728 | 0.257 | 0 | legacy_tv_fail_oracle_calibrated_warning |
| ELIZA | 0.054 | 0.233 | 0.179 | 0.493 | 0.397 | 0.867 | 0.404 | 0 | legacy_tv_fail_oracle_calibrated_warning |
| Human | 0.062 | 0.232 | 0.170 | 0.572 | 0.516 | 0.868 | 0.478 | 0 | legacy_tv_fail_oracle_calibrated_warning |
| LLMs | 0.073 | 0.219 | 0.146 | 0.300 | 0.254 | 0.528 | 0.148 | 0 | legacy_tv_fail_oracle_calibrated_warning |

Failed PPC gates:

- `weighted_mean_cell_TV` = 0.387592; threshold `<= 0.25`
- `median_cell_TV` = 0.375954; threshold `<= 0.25`
- `q90_cell_TV` = 0.828053; threshold `<= 0.50`
- `fraction_cell_TV_gt_0p5` = 0.322111; threshold `<= 0.10`
- `posterior_predictive_tv_p_value` = 0.9952; threshold `in [0.05, 0.95]`

## LOO

LOO is full, not partial: 379/379 ratings, 500 LOO draws.

| Metric | Value |
|---|---:|
| elpd_loo | -462.442 |
| p_loo | 12.3153 |
| mean_pareto_k | 0.0425892 |
| median_pareto_k | 0.0376036 |
| max_pareto_k | 0.490256 |
| frac_pareto_k_gt_0p7 | 0 |
| frac_pareto_k_gt_1p0 | 0 |

## Artefact Completeness

- Completed run directories: `10`
- Failed/incomplete/skipped run directories: `0`
- Required per-seed artefacts: present for all completed seeds
- Sidecar row metadata: present per seed and concatenated top-level/refreshed
- Top-level sidecar rows: `3790` data rows plus header
- Refreshed sidecar rows: `3790` data rows plus header
- `code_diff_before_pilot.patch`: written before pilot launch; size was `0` bytes because preflight `git status --short` was clean

## Reproducibility Caveat

The preflight `git status --short` was clean before pilot launch. The post-run refresh manifest captured unrelated working-tree changes outside the pilot directory, including files under `outputs/main_synthetic_validation` and `scripts/main_synthetic_validation_plots.py`. I did not write to `outputs/main_synthetic_validation`, `outputs/main_synthetic_validation_smoke_postpatch_20260511`, or `outputs/recovery_ladder_20260511`; the pilot command wrote to `outputs/main_synthetic_validation_pilot_20260511`.

## Command Ledger

```zsh
ps -eo pid,ppid,etime,command | rg 'main_synthetic_validation|main_validation|pymc|pytensor|python'
find outputs/main_synthetic_validation_pilot_20260511 -maxdepth 3 -mindepth 1 -print 2>/dev/null | head -n 80
find outputs -maxdepth 2 -type d -name '*pilot*' -print 2>/dev/null | sort
pwd
mkdir -p outputs/main_synthetic_validation_pilot_20260511
git status --short
git rev-parse HEAD
rg -n "def run_one_fit|NUM_CHAINS|NUM_SAMPLES|NUM_TUNE|TARGET_ACCEPT|chains|draws|tune|target_accept" scripts/main_synthetic_validation.py dcm_model.py
git diff --binary --no-ext-diff > outputs/main_synthetic_validation_pilot_20260511/code_diff_before_pilot.patch && ls -l outputs/main_synthetic_validation_pilot_20260511/code_diff_before_pilot.patch && wc -c outputs/main_synthetic_validation_pilot_20260511/code_diff_before_pilot.patch
.venv/bin/python -m py_compile scripts/main_synthetic_validation.py scripts/main_synthetic_validation_metrics.py
sed -n '140,176p' scripts/main_synthetic_validation.py && sed -n '3204,3216p' scripts/main_synthetic_validation.py
.venv/bin/python - <<'PY'  # attempted old validation_config helper import; failed, then reran with build_fit_config
rg -n "def .*targeted|TARGETED_OVERRIDE|targeted_keys|build_fit_config\(" scripts/main_synthetic_validation.py | head -n 80
sed -n '80,145p' scripts/main_synthetic_validation.py
.venv/bin/python - <<'PY'  # build_fit_config pilot sampler assertion: 4 chains, 1000 draws, 1000 tune, target_accept 0.95
.venv/bin/python scripts/main_synthetic_validation.py --output-dir outputs/main_synthetic_validation_pilot_20260511 --mode targeted-override-sanity --seed 20260511
.venv/bin/python - <<'PY'  # targeted_override_sanity.json assertions
find outputs/main_synthetic_validation_pilot_20260511 -maxdepth 2 -type f -print | sort
.venv/bin/python scripts/main_synthetic_validation.py --output-dir outputs/main_synthetic_validation_pilot_20260511 --mode pilot --fit-variants targeted_strong_lower_override --n-seeds 10 --seed 20260511
.venv/bin/python scripts/main_synthetic_validation.py --output-dir outputs/main_synthetic_validation_pilot_20260511 --mode refresh-existing
find outputs/main_synthetic_validation_pilot_20260511 -maxdepth 3 -type f -print | sort
cat outputs/main_synthetic_validation_pilot_20260511/summary_refresh_manifest.json
cat outputs/main_synthetic_validation_pilot_20260511/refreshed_summaries/rho_recovery_summary_refreshed.csv
cat outputs/main_synthetic_validation_pilot_20260511/refreshed_summaries/sampler_diagnostics_summary_refreshed.csv
cat outputs/main_synthetic_validation_pilot_20260511/refreshed_summaries/pass_fail_summary_refreshed.csv
cat outputs/main_synthetic_validation_pilot_20260511/refreshed_summaries/oracle_root_bridge_summary.csv
cat outputs/main_synthetic_validation_pilot_20260511/refreshed_summaries/posterior_predictive_summary_refreshed.csv
cat outputs/main_synthetic_validation_pilot_20260511/refreshed_summaries/loo_lppd_summary_refreshed.csv
cat outputs/main_synthetic_validation_pilot_20260511/refreshed_summaries/oracle_root_bridge_cases.csv
cat outputs/main_synthetic_validation_pilot_20260511/validation_manifest.csv
wc -l outputs/main_synthetic_validation_pilot_20260511/posterior_predictive_row_metadata.csv outputs/main_synthetic_validation_pilot_20260511/refreshed_summaries/posterior_predictive_row_metadata.csv
find outputs/main_synthetic_validation_pilot_20260511/runs/targeted_strong_lower_override -mindepth 1 -maxdepth 2 -type f \( -name 'fit.nc' -o -name 'posterior_predictive.nc' -o -name 'posterior_predictive_row_metadata.csv' -o -name 'per_rating_log_lik.nc' -o -name 'per_rating_log_lik.zarr' -o -name 'truth_payload.json' -o -name 'config.json' -o -name 'run_summary.json' -o -name 'sampler_diagnostics.csv' -o -name 'posterior_predictive_summary.csv' -o -name 'loo_lppd_summary.csv' \) -print | sort
.venv/bin/python - <<'PY'  # structured threshold and artefact check
```

## Scope Confirmation

Launched exactly the requested targeted-only 10-seed pilot and the requested non-sampling refresh. Did not launch `production_unmodified`, a negative control, a full validation run, or extra variant sweeps. Did not automatically proceed beyond pilot.
