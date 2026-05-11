# Main Synthetic Validation Recovery Ladder

## Executive Summary

The no-fit ladder checks progressively more DCM-like synthetic recovery before HMC validation.

## Rung Pass/Fail Table

| rung | rung_description | n_cases | n_replicates | median_log_B_eff_R1 | median_log_B_eff_R0 | mean_rho_R1 | mean_rho_R0 | evidence_margin_M | balanced_log_score_improvement | balanced_brier_improvement | TPR_at_rho_gt_0p5 | TNR_at_rho_le_0p5 | balanced_accuracy | pass_fail_label |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L0 | minimal binary-tree oracle | 40 | 20 | 6.28 | -6.278 | 0.9295 | 0.001356 | 4.67 | 0.882 | 0.3367 | 0.95 | 1 | 0.975 | pass |
| L2 | depth-3 toy tree | 40 | 20 | 3.832 | -5.302 | 0.6598 | 0.03328 | 2.223 | 0.6221 | 0.2328 | 0.65 | 1 | 0.825 | pass |
| L3 | actual GWT all-edge extreme beta | 40 | 20 | 10.32 | -11.35 | 0.9959 | 7.773e-05 | 8.71 | 0.9849 | 0.3611 | 1 | 1 | 1 | pass |
| L4 | actual GWT targeted strong-lower beta | 40 | 20 | 3.009 | -2.624 | 0.7194 | 0.09111 | 1.289 | 0.6465 | 0.2598 | 0.8 | 0.95 | 0.875 | pass |
| L5 | targeted strong-lower ordinal binary leaf | 40 | 20 | 3.009 | -2.615 | 0.719 | 0.09112 | 1.28 | 0.6463 | 0.2597 | 0.8 | 0.95 | 0.875 | pass |
| L6 | targeted strong-lower three-state ordinal leaf | 40 | 20 | 3.044 | -3.574 | 0.7388 | 0.08335 | 1.435 | 0.6839 | 0.2705 | 0.85 | 0.95 | 0.9 | pass |

## Interpretation

All no-fit ladder rungs passed the configured recovery gates.
