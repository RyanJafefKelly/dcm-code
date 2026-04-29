#!/usr/bin/env bash
# Quick rebuild loop for the ordinal explainer figures.
# Usage:
#   bash figures/build.sh           # build both
#   bash figures/build.sh pgm       # build PGM only
#   bash figures/build.sh cut       # build cutpoint diagram only
#   bash figures/build.sh open      # rebuild both then open the PNGs

set -euo pipefail
cd "$(dirname "$0")/.."

# Find a Python with daft-pgm. Prefer local venv, fall back to the
# sibling dcm-code worktree's venv (where daft-pgm was installed), then $PATH.
if [ -n "${PY:-}" ]; then
    :
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif [ -x "../dcm-code/.venv/bin/python" ]; then
    PY="../dcm-code/.venv/bin/python"
else
    PY="python"
fi

target="${1:-all}"

run_pgm()      { "$PY" figures/ordinal_explainer_pgm.py; }
run_pgm_all()  { "$PY" figures/ordinal_explainer_pgm.py --all; }
run_cut()      { "$PY" figures/ordinal_explainer_cutpoints.py; }
run_variants() { "$PY" figures/ordinal_explainer_expert_variants.py; }

case "$target" in
  pgm)        run_pgm ;;
  cut)        run_cut ;;
  variants)   run_variants ;;
  pgm-all)    run_pgm_all ;;     # keep for if you ever want PGM variants
  open)
    run_pgm; run_cut; run_variants
    open report_figures/ordinal_explainer/pgm.png \
         report_figures/ordinal_explainer/cutpoints.png \
         report_figures/ordinal_explainer/cutpoints_be.png \
         report_figures/ordinal_explainer/cutpoints_kappa_e.png \
         report_figures/ordinal_explainer/cutpoints_sigma_e.png
    ;;
  all|*)
    run_pgm; run_cut; run_variants
    ;;
esac
