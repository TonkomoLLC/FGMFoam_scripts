#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(pwd -P)"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPTS_DIR="${FGM_SCRIPTS_DIR:-$HERE/scripts}"
CASE_DIR="${CASE_DIR:-$RUN_DIR/03_testcase}"

[[ -d "$CASE_DIR/constant/tables" ]] || { echo "[FAIL] Missing installed tables: $CASE_DIR/constant/tables" >&2; exit 1; }

python3 "$SCRIPTS_DIR/upgrade_existing_zero_variance_tables.py" \
    --case "$CASE_DIR" --enable-progress-variable-variance

python3 "$SCRIPTS_DIR/validate_of7_premixed_tables.py" \
    --case "$CASE_DIR" --require-species CH4 O2 CO2 H2O OH N2 \
    --check-z 0.04293 0.1559 --pilot-z 0.04293 --require-zero-input-variances

echo "[DONE] Working premixed tables upgraded without rerunning Cantera."
echo "[NEXT] Move/remove nonzero time directories and run FGMFoam from time 0."
