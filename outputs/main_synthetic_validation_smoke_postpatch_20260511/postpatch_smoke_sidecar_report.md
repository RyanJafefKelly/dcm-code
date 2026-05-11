# Post-Patch Smoke Sidecar Report

Output directory: `/Users/ryankelly/ryan-code/research/dcm-code/outputs/main_synthetic_validation_smoke_postpatch_20260511`

## Decision

The post-patch smoke sidecar check completed. The hardened refresh pipeline can regenerate internally consistent summaries from the fresh smoke artefacts, and the new PPC row-metadata sidecars were written and validated.

It is safe to launch the targeted-only 10-seed pilot in a separate fresh directory, with the normal caveat that this one-seed smoke is not an acceptance-quality sampler run. The smoke fit had no divergences and no tree-depth hits, but did report R-hat/ESS warnings under 2 chains x 200 draws.

## Answers

- Did the post-patch smoke run complete? `yes`
- Were PPC row-metadata sidecars written? `yes`
- Can `refresh-existing` regenerate consistent summaries? `yes`
- Are there any stale-summary warnings? `no`; `summary_stale_before_refresh=false`
- Are there any blocking runtime or sampler failures? `no blocking runtime failure`; sampler warnings are present but expected for this short smoke.
- Is it safe to launch the 10-seed targeted pilot? `yes`, in a fresh pilot directory, without reusing this smoke directory or `outputs/main_synthetic_validation`.

## Required Checks

| Check | Result |
|---|---|
| Fresh output directory used | pass |
| `git status --short` printed before run | pass |
| `py_compile` on touched Python files | pass |
| Targeted override sanity | pass |
| Generator/fitter beta matching | pass; max pres diff `0.0`, max abs diff `0.0` |
| `fit.nc` exists | pass |
| `posterior_predictive.nc` exists | pass |
| `per_rating_log_lik.nc` exists | pass |
| `truth_payload.json` exists | pass |
| `config.json` exists | pass |
| `run_summary.json` exists | pass |
| Per-run `posterior_predictive_row_metadata.csv` exists | pass |
| Top-level `posterior_predictive_row_metadata.csv` exists | pass |
| Refreshed `posterior_predictive_row_metadata.csv` exists | pass |
| Sidecar row count matches PPC rating rows | pass; `379` rows vs `379` `rating_row`s |
| `summary_refresh_manifest.json` exists | pass |
| `refreshed_summaries/smoke_pipeline_refresh_report.md` exists | pass |
| `oracle_root_bridge_cases.csv` exists | pass |
| `oracle_root_bridge_summary.csv` exists | pass |
| `sampler_diagnostics_summary_refreshed.csv` exists | pass |
| `posterior_predictive_summary_refreshed.csv` exists | pass |
| `pass_fail_summary_refreshed.csv` exists | pass |
| Manifest internal consistency | pass |
| HMC-only root sign flips | pass; `0` |
| Divergences | pass; `0` |
| Tree-depth hits | pass; `0` |

## Targeted Override Sanity

`targeted_override_sanity.json` reports `pass_fail=pass`.

- overridden edge count: `49`
- root-to-top overridden edge count: `0`
- weak-top overridden edge count: `0`
- generator/fitter pres diff: `0.0`
- generator/fitter abs diff: `0.0`
- targeted prior family: `logit-Normal/build_safe_gain_node_beta`
- non-targeted prior family: `production_evidence_processor_per_node_beta`

## Refresh Consistency

`summary_refresh_manifest.json` reports:

- `summary_stale_before_refresh=false`
- `fit_nc_rho_matches_per_run_rho_cases=true`
- `fit_nc_sampler_matches_run_summary_json=true`
- `ppc_refreshed_from_existing_per_run_summaries=true`
- `loo_refreshed_from_existing_per_run_summaries=true`
- `oracle_root_bridge_cases_written=true`
- `hmc_or_refit_launched=false` for the refresh step
- `posterior_predictive_draws_generated=false` for the refresh step
- `loo_recomputed=false`

## Root Bridge

The oracle/HMC bridge contains two system rows:

| System | root_z_true | log_B_oracle | log_B_eff_HMC | rho_collapsed_HMC | posterior_correct_sign | oracle_correct_sign | hmc_only_sign_flip |
|---|---:|---:|---:|---:|---|---|---|
| Chicken | 1 | 4.958 | 5.928 | 0.9869 | True | True | False |
| LLMs | 0 | -4.432 | -5.729 | 0.000649 | True | True | False |

There are no HMC-only root sign flips.

## Sampler Notes

This smoke fit used 2 chains, 200 tune, and 200 draw iterations. PyMC recommended at least 4 chains for robust convergence diagnostics.

Sampler summary:

- runtime: `1517.85` seconds
- max R-hat: `1.04`
- `n_rhat_gt_1p01`: `38`
- `n_rhat_gt_1p05`: `0`
- min ESS bulk: `247`
- min ESS tail: `80`
- divergences: `0`
- tree-depth hits: `0`
- mean acceptance rate: `0.9729`

These are non-blocking for this pipeline sidecar check, but the pilot should use the pilot sampler settings and enforce the pilot sampler gates.

## PPC Notes

PPC artefacts were produced and the sidecar metadata was written. The sidecar row count matches the PPC NetCDF and PPC summary:

- per-run sidecar rows: `379`
- top-level sidecar rows: `379`
- refreshed sidecar rows: `379`
- `posterior_predictive.nc` `rating_row` dimension: `379`
- PPC summary `ALL.n_ratings`: `379`

Legacy PPC TV gates still fail in this one-seed smoke, with `n_cells_ge_5=0`; this is a smoke diagnostic warning, not a pilot decision.

## Commands Run

```zsh
pwd
git status --short
find outputs/main_synthetic_validation_smoke_postpatch_20260511 -maxdepth 2 -mindepth 1 -print 2>/dev/null | head -n 40
rg -n -- "--n-seeds|--seed|fit-variants|mode.*smoke|refresh-existing|pilot|smoke" scripts/main_synthetic_validation.py
.venv/bin/python -m py_compile scripts/main_synthetic_validation.py scripts/main_synthetic_validation_metrics.py
sed -n '3090,3245p' scripts/main_synthetic_validation.py
.venv/bin/python scripts/main_synthetic_validation.py --output-dir outputs/main_synthetic_validation_smoke_postpatch_20260511 --mode smoke --fit-variants targeted_strong_lower_override --n-seeds 1 --seed 20260511
.venv/bin/python scripts/main_synthetic_validation.py --output-dir outputs/main_synthetic_validation_smoke_postpatch_20260511 --mode refresh-existing
find outputs/main_synthetic_validation_smoke_postpatch_20260511 -maxdepth 4 -type f -print | sort
cat outputs/main_synthetic_validation_smoke_postpatch_20260511/summary_refresh_manifest.json
cat outputs/main_synthetic_validation_smoke_postpatch_20260511/refreshed_summaries/sampler_diagnostics_summary_refreshed.csv
cat outputs/main_synthetic_validation_smoke_postpatch_20260511/refreshed_summaries/oracle_root_bridge_cases.csv
cat outputs/main_synthetic_validation_smoke_postpatch_20260511/refreshed_summaries/pass_fail_summary_refreshed.csv
cat outputs/main_synthetic_validation_smoke_postpatch_20260511/refreshed_summaries/posterior_predictive_summary_refreshed.csv
cat outputs/main_synthetic_validation_smoke_postpatch_20260511/validation_manifest.csv
cat outputs/main_synthetic_validation_smoke_postpatch_20260511/runs/targeted_strong_lower_override/seed_20260511/run_summary.json
cat outputs/main_synthetic_validation_smoke_postpatch_20260511/runs/targeted_strong_lower_override/seed_20260511/config.json
head -n 5 outputs/main_synthetic_validation_smoke_postpatch_20260511/runs/targeted_strong_lower_override/seed_20260511/posterior_predictive_row_metadata.csv && wc -l outputs/main_synthetic_validation_smoke_postpatch_20260511/runs/targeted_strong_lower_override/seed_20260511/posterior_predictive_row_metadata.csv outputs/main_synthetic_validation_smoke_postpatch_20260511/refreshed_summaries/posterior_predictive_row_metadata.csv outputs/main_synthetic_validation_smoke_postpatch_20260511/posterior_predictive_row_metadata.csv
cat outputs/main_synthetic_validation_smoke_postpatch_20260511/refreshed_summaries/rho_recovery_summary_refreshed.csv
head -n 160 outputs/main_synthetic_validation_smoke_postpatch_20260511/refreshed_summaries/smoke_pipeline_refresh_report.md
.venv/bin/python scripts/main_synthetic_validation.py --output-dir outputs/main_synthetic_validation_smoke_postpatch_20260511 --mode targeted-override-sanity --seed 20260511
cat outputs/main_synthetic_validation_smoke_postpatch_20260511/targeted_override_sanity.json
.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd
import xarray as xr
base = Path('outputs/main_synthetic_validation_smoke_postpatch_20260511')
run = base/'runs/targeted_strong_lower_override/seed_20260511'
meta = pd.read_csv(run/'posterior_predictive_row_metadata.csv')
ppc = xr.open_dataset(run/'posterior_predictive.nc')
print('sidecar_rows', len(meta))
print('posterior_predictive_dims', dict(ppc.sizes))
for name in ppc.data_vars:
    print('data_var', name, tuple(ppc[name].dims), tuple(ppc[name].shape))
print('n_ratings_summary', pd.read_csv(base/'refreshed_summaries/posterior_predictive_summary_refreshed.csv').query("system == 'ALL'")['n_ratings'].iloc[0])
PY
find outputs/main_synthetic_validation_smoke_postpatch_20260511/runs/targeted_strong_lower_override/seed_20260511 -maxdepth 1 \( -name '*error*' -o -name '*fail*' -o -name '*partial*' -o -name '*.log' \) -print | sort
git status --short scripts/main_synthetic_validation.py scripts/main_synthetic_validation_metrics.py outputs/main_synthetic_validation_smoke_postpatch_20260511
```

## Scope Confirmation

This task did launch exactly one requested targeted smoke HMC fit in the fresh post-patch directory. It did not write to `outputs/main_synthetic_validation`, did not launch pilot/full, did not run `production_unmodified`, did not run a negative control, and did not automatically proceed beyond the sidecar smoke check.
