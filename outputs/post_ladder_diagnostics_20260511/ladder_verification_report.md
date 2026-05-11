# Ladder Output Verification Report

Input directory: `outputs/recovery_ladder_20260511`
Output directory: `outputs/post_ladder_diagnostics_20260511`

## Required Checks

- targeted_override_sanity pass_fail: `pass`.
- L4/L5/L6 pass check: `pass` (pass_rows=9/9; system_ALL_pass=True).
- Case-count consistency: `pass` (no mismatches).
- Chicken/LLMs threshold flags: `fail`.

## System-Specific Threshold Flags

- `L2_sweep__0p43_0p50`: Chicken(balanced_accuracy<0.65;mean_rho_R1<=mean_rho_R0+0.20); LLMs(balanced_accuracy<0.65;mean_rho_R1<=mean_rho_R0+0.20)
- `L2_sweep__0p55_0p45`: Chicken(balanced_accuracy<0.65;mean_rho_R1<=mean_rho_R0+0.20); LLMs(balanced_accuracy<0.65;mean_rho_R1<=mean_rho_R0+0.20)
- `L2_sweep__0p60_0p40`: Chicken(balanced_accuracy<0.65;mean_rho_R1<=mean_rho_R0+0.20); LLMs(balanced_accuracy<0.65;mean_rho_R1<=mean_rho_R0+0.20)
- `L2_sweep__0p65_0p35`: Chicken(balanced_accuracy<0.65;mean_rho_R1<=mean_rho_R0+0.20); LLMs(balanced_accuracy<0.65;mean_rho_R1<=mean_rho_R0+0.20)
- `L2_sweep__0p70_0p30`: Chicken(balanced_accuracy<0.65;mean_rho_R1<=mean_rho_R0+0.20); LLMs(balanced_accuracy<0.65;mean_rho_R1<=mean_rho_R0+0.20)

Thresholds applied per Chicken/LLMs row: balanced accuracy >= 0.65, Brier improvement > 0, log-score improvement > 0, and mean_rho_R1 > mean_rho_R0 + 0.20.

## L2_sweep Lower-Edge-Gap Diagnostic

- First gap where both Chicken and LLMs clear the requested metric thresholds: `0.6`.
- First gap where both Chicken and LLMs have source pass_fail_label=`pass`: `0.8`.
- gap -0.07: threshold_status={'Chicken': 'fail', 'LLMs': 'fail'}, source_labels={'Chicken': 'fail', 'LLMs': 'fail'}
- gap 0.10: threshold_status={'Chicken': 'fail', 'LLMs': 'fail'}, source_labels={'Chicken': 'fail', 'LLMs': 'fail'}
- gap 0.20: threshold_status={'Chicken': 'fail', 'LLMs': 'fail'}, source_labels={'Chicken': 'fail', 'LLMs': 'fail'}
- gap 0.30: threshold_status={'Chicken': 'fail', 'LLMs': 'fail'}, source_labels={'Chicken': 'fail', 'LLMs': 'fail'}
- gap 0.40: threshold_status={'Chicken': 'fail', 'LLMs': 'fail'}, source_labels={'Chicken': 'fail', 'LLMs': 'fail'}
- gap 0.60: threshold_status={'Chicken': 'pass', 'LLMs': 'pass'}, source_labels={'Chicken': 'borderline', 'LLMs': 'borderline'}
- gap 0.80: threshold_status={'Chicken': 'pass', 'LLMs': 'pass'}, source_labels={'Chicken': 'pass', 'LLMs': 'pass'}

Interpretation: the requested metric thresholds fail through lower-edge gap 0.40 and first clear at gap 0.60 (sweep 0.80/0.20). The source summary still labels Chicken and LLMs as borderline at that gap because the evidence-margin bootstrap CI crosses zero; both systems are source-pass by gap 0.80 (sweep 0.90/0.10).

## L4 Structural Repair Gate

L4 clears the structural repair gate.
This is based on targeted override sanity passing, all L4 source pass_fail labels being pass, and both Chicken and LLMs clearing the requested system thresholds for L4.

## HMC/Sampling Confirmation

No HMC or sampling was launched. This diagnostics run only read the specified ladder JSON/CSV/Markdown inputs and wrote derived CSV/Markdown summaries.
