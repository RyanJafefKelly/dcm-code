#!/usr/bin/env bash
# Asymmetric β_pres / β_abs prior sweep — driver.
#
# Runs four fits in series: three synthetic recoveries on seed 20260506 +
# one real-data refit at the centre prior. Each fit ≈30 min on this machine.
#
# Stretch: per-system / no-anchors fit at the centre prior, gated on
# RUN_PER_SYSTEM_STRETCH=1.
#
# Each invocation produces stdout/stderr in the run dir's sample.log. The
# whole script writes a top-level sweep.log so you can `tail -f` it.
#
# Usage:
#   nohup bash notebooks/asymmetric_prior_sweep_2026-05-10/run_sweep.sh > \
#       notebooks/asymmetric_prior_sweep_2026-05-10/sweep.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

REPO_ROOT="$(pwd)"
SWEEP_DIR="$REPO_ROOT/notebooks/asymmetric_prior_sweep_2026-05-10"
RECOVERY_DRIVER="$REPO_ROOT/notebooks/synthetic_validation_2026-05-06/gwt_full_exact_recovery.py"
PER_SYSTEM_DRIVER="$REPO_ROOT/notebooks/synthetic_validation_2026-05-06/per_system_audit_2026-05-07/task5_recovery_per_system_obs.py"
REAL_DATA_DRIVER="$SWEEP_DIR/run_real_data_with_override.py"
SYNTH_RUNS_DIR="$SWEEP_DIR/runs/synthetic"
mkdir -p "$SYNTH_RUNS_DIR"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_synth() {
  local pres="$1"; local abs_="$2"; local sigma="$3"
  local tag="pres$(printf "%02d" $(python -c "print(int(round($pres*100)))"))_abs$(printf "%02d" $(python -c "print(int(round($abs_*100)))"))_sig$(printf "%02d" $(python -c "print(int(round($sigma*100)))"))"
  echo "[$(ts)] === synthetic recovery: $tag ==="
  python "$RECOVERY_DRIVER" \
      --seed 20260506 \
      --runs-dir "$SYNTH_RUNS_DIR" \
      --beta-pres-mean "$pres" --beta-abs-mean "$abs_" --beta-override-sigma "$sigma" \
      --overwrite
}

run_real() {
  local pres="$1"; local abs_="$2"; local sigma="$3"
  echo "[$(ts)] === real-data refit: pres=$pres abs=$abs_ sigma=$sigma ==="
  python "$REAL_DATA_DRIVER" \
      --beta-pres-mean "$pres" --beta-abs-mean "$abs_" --beta-override-sigma "$sigma" \
      --overwrite
}

run_per_system() {
  local pres="$1"; local abs_="$2"; local sigma="$3"
  echo "[$(ts)] === per-system / no-anchors stretch: pres=$pres abs=$abs_ sigma=$sigma ==="
  python "$PER_SYSTEM_DRIVER" \
      --seed 20260506 --no-anchors \
      --runs-dir "$SWEEP_DIR/runs/per_system_noanchors" \
      --beta-pres-mean "$pres" --beta-abs-mean "$abs_" --beta-override-sigma "$sigma" \
      --overwrite
}

START=$(date +%s)
echo "[$(ts)] sweep starting (branch=$(git rev-parse --abbrev-ref HEAD), commit=$(git rev-parse HEAD))"

run_synth 0.95 0.05 0.30
run_synth 0.90 0.10 0.30
run_synth 0.85 0.15 0.50
run_real  0.90 0.10 0.30

if [ "${RUN_PER_SYSTEM_STRETCH:-0}" = "1" ]; then
  run_per_system 0.90 0.10 0.30
else
  echo "[$(ts)] (skipping per-system stretch; set RUN_PER_SYSTEM_STRETCH=1 to enable)"
fi

END=$(date +%s)
echo "[$(ts)] sweep complete; total elapsed $((END-START))s"
