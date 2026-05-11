# Phase 1B Root-Evidence Ladder Plan

Current invocation mode: `fit-smoke`. Requested rungs: `2,5M,5X,6,7`. Seed: `20260511`.

## Top-Bound Status
| beta_profile | median_log_B_top_R1 | median_log_B_top_R0 | frac_R1_log_B_top_gt_tau_present_50 | frac_R0_log_B_top_lt_tau_absent_05 | preflight_gate |
|---|---|---|---|---|---|
| production_prior_mean | 2.809 | -3.142 | 0.755 | 0.718 | pass |
| production_oracle_median | 4.839 | -4.395 | 0.877 | 0.840 | pass |
| extreme_0p9_0p1 | 10.986 | -10.986 | 0.997 | 0.997 | pass |

## No-Fit Rung Status
| rung | beta_profile | n_replicates | median_log_B_eff_R1 | median_log_B_eff_R0 | balanced_accuracy | balanced_log_score_improvement | evidence_margin_M | pass_fail_label | survival_R1 | abs_survival_R0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1P | production_prior_mean | 500 | 0.501 | -0.518 | 0.546 | 0.154 | -1.109 | fail | 0.178 | 0.165 |
| 1X | extreme_0p9_0p1 | 500 | 10.198 | -10.318 | 0.983 | 0.937 | 8.589 | pass | 0.928 | 0.939 |
| 3 | production_prior_mean | 500 | 0.501 | -0.516 | 0.546 | 0.153 | -1.108 | fail | 0.179 | 0.164 |
| 4 | production_prior_mean | 500 | 0.812 | -0.658 | 0.610 | 0.221 | -0.797 | fail | 0.289 | 0.209 |

## Fit-Smoke Status
| rung | rung_description | n_rep_smoke | status |
|---|---|---|---|
| 2 | binary noisy leaves with inferred beta and matching production priors | 2 | config_validated_not_sampled |
| 5M | matched-prior nuisance washout, fully crossed | 2 | config_validated_not_sampled |
| 5X | extreme-beta nuisance positive control, fully crossed | 2 | config_validated_not_sampled |
| 6 | production-like rater design | 2 | config_validated_not_sampled |
| 7 | improved crossed design | 2 | config_validated_not_sampled |

## Exact Commands
Top-bound pre-flight:
```bash
.venv/bin/python scripts/phase1b_root_evidence_ladder.py --output-dir outputs/phase1_root_evidence --mode top-bound
```
No-fit full run:
```bash
.venv/bin/python scripts/phase1b_root_evidence_ladder.py --output-dir outputs/phase1_root_evidence --mode no-fit --rungs 1P,1X,3,4 --n-rep 500 --seed 20260511
```
No-fit smoke run:
```bash
.venv/bin/python scripts/phase1b_root_evidence_ladder.py --output-dir outputs/phase1_root_evidence --mode no-fit --rungs 1P,1X,3,4 --n-rep 20 --seed 20260511
```
Fit-rung smoke/config validation:
```bash
.venv/bin/python scripts/phase1b_root_evidence_ladder.py --output-dir outputs/phase1_root_evidence --mode fit-smoke --rungs 2,5M,5X,6,7 --n-rep 2 --seed 20260511
```
Full fit rungs require HMC refits through the exact-tree model. Use the same rung definitions in this script and run screening batches first:
```bash
.venv/bin/python scripts/phase1b_root_evidence_ladder.py --output-dir outputs/phase1_root_evidence --mode fit-smoke --rungs 2,5M,6,7 --n-rep 32 --seed 20260511
# Then run the generated configs with MultiSystemExactTreeBuilder/NUTS in batches; confirm 64 replicates if evidence_margin_M is borderline.
```

## Decision Logic
- If top-bound fails, inspect heterogeneous strong-feature firing patterns before broad validation.
- If top-bound passes but no-fit rungs fail, evidence is being lost downstream of top-level latent structure.
- If no-fit rungs pass but fit rungs fail, nuisance uncertainty, beta identifiability, or rater design is the bottleneck.
- If Rung 6 fails but Rung 7 passes, improved cross-system rater coverage, especially Chicken coverage, is a sufficient design fix.

Recommended next action: inspect failing no-fit rungs before spending HMC on inferred nuisance rungs.
