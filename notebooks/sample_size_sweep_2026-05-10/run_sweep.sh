#!/usr/bin/env bash
# Run the K-sweep for Branch 6 (c-id-sample-size-sweep).
# Centre β prior, synthetic seed 20260506, three_state baseline.

set -euo pipefail

cd "$(dirname "$0")/../.."
HERE="notebooks/sample_size_sweep_2026-05-10"
RUNS="${HERE}/runs"
LOGS="${HERE}/logs"
mkdir -p "${LOGS}"

for K in 1 2 5 10 20; do
  log="${LOGS}/k${K}.log"
  echo "[$(date -u +%FT%TZ)] === starting K=${K} ===" | tee -a "${log}"
  /usr/bin/time -p python notebooks/synthetic_validation_2026-05-06/gwt_full_exact_recovery.py \
    --seed 20260506 \
    --rater-multiplier "${K}" \
    --runs-dir "${RUNS}" \
    --beta-pres-mean 0.90 \
    --beta-abs-mean 0.10 \
    --beta-override-sigma 0.30 \
    --overwrite \
    >> "${log}" 2>&1
  echo "[$(date -u +%FT%TZ)] === finished K=${K} ===" | tee -a "${log}"
done

echo "[$(date -u +%FT%TZ)] === all K done ===" | tee -a "${LOGS}/_summary.log"
