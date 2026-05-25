#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(pwd -P)"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPTS_DIR="${FGM_SCRIPTS_DIR:-$HERE/scripts}"
CASE_DIR="${CASE_DIR:-$RUN_DIR/03_testcase}"
FLAMELET_DIR="${FLAMELET_DIR:-$RUN_DIR/1DPremixedFlameletFiles}"
MECH="${MECH:-gri30.yaml}"
Z_MAX="${Z_MAX:-0.1559}"
PILOT_Z="${PILOT_Z:-0.04293}"
NZ="${NZ:-51}"
NC="${NC:-51}"
TIN="${TIN:-294.0}"

[[ -d "$CASE_DIR/constant" ]] || { echo "[FAIL] Missing case: $CASE_DIR" >&2; exit 1; }
mkdir -p "$FLAMELET_DIR"
rm -f "$FLAMELET_DIR"/premixed_Z_*.yaml "$FLAMELET_DIR"/post_premixed_Z_*.csv "$FLAMELET_DIR"/premixed_manifest.csv

python3 "$SCRIPTS_DIR/generatePremixedFlamelets.py" \
    --mech "$MECH" --out-dir "$FLAMELET_DIR" \
    --z-max "$Z_MAX" --nz "$NZ" --extra-z "$PILOT_Z" \
    --tin "$TIN" --overwrite

CASE_DIR="$CASE_DIR" FLAMELET_DIR="$FLAMELET_DIR" MECH="$MECH" Z_MAX="$Z_MAX" PILOT_Z="$PILOT_Z" NZ="$NZ" NC="$NC" TIN="$TIN" \
    "$HERE/make_03_testcase_tables_from_existing_premixed_flamelets.sh"
