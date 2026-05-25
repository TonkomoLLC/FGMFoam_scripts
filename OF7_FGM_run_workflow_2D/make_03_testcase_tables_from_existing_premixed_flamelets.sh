#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="$(pwd -P)"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPTS_DIR="${FGM_SCRIPTS_DIR:-$HERE/scripts}"
CASE_DIR="${CASE_DIR:-$RUN_DIR/03_testcase}"
FLAMELET_DIR="${FLAMELET_DIR:-$RUN_DIR/1DPremixedFlameletFiles}"
WORK_DIR="${WORK_DIR:-$RUN_DIR/FGMTableBuild_OF7_premixed}"
MECH="${MECH:-gri30.yaml}"
Z_MAX="${Z_MAX:-0.1559}"
PILOT_Z="${PILOT_Z:-0.04293}"
NZ="${NZ:-51}"
NC="${NC:-51}"
TIN="${TIN:-294.0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="$WORK_DIR/02_tables_of7_premixed.tar.xz"
REQUIRED_SPECIES=(CH4 O2 CO2 H2O OH N2)

fail() { printf '[FAIL] %s\n' "$*" >&2; exit 1; }
backup_path() { if [[ -e "$1" ]]; then mv "$1" "$1.before_premixed_FGM_${STAMP}"; echo "[BACKUP] $1 -> $1.before_premixed_FGM_${STAMP}"; fi; }
[[ -d "$CASE_DIR/constant" ]] || fail "Missing case: $CASE_DIR"
[[ -f "$FLAMELET_DIR/premixed_manifest.csv" ]] || fail "Missing premixed manifest: $FLAMELET_DIR/premixed_manifest.csv. Diffusion flamelets cannot be used for this workflow."
compgen -G "$FLAMELET_DIR/premixed_Z_*.yaml" >/dev/null || fail "No premixed YAML flamelets in $FLAMELET_DIR"
mkdir -p "$WORK_DIR"
rm -f "$FLAMELET_DIR"/post_premixed_Z_*.csv
backup_path "$ARCHIVE"

python3 "$SCRIPTS_DIR/organizePremixedData.py" \
    --mech "$MECH" --manifest "$FLAMELET_DIR/premixed_manifest.csv" \
    --glob "$FLAMELET_DIR/premixed_Z_*.yaml"
python3 "$SCRIPTS_DIR/buildPremixedFGMTables.py" \
    --mech "$MECH" --glob "$FLAMELET_DIR/post_premixed_Z_*.csv" \
    --out "$ARCHIVE" --z-max "$Z_MAX" --nz "$NZ" --nc "$NC" --tin "$TIN" --report-z "$PILOT_Z"

backup_path "$CASE_DIR/constant/tables"
backup_path "$CASE_DIR/constant/tableProperties"
backup_path "$CASE_DIR/constant/PVtableProperties"
python3 "$SCRIPTS_DIR/csv2of_tables.py" --in "$ARCHIVE" --case "$CASE_DIR" --table-dir constant/tables
python3 "$SCRIPTS_DIR/validate_of7_premixed_tables.py" --case "$CASE_DIR" \
    --require-species "${REQUIRED_SPECIES[@]}" --check-z "$PILOT_Z" "$Z_MAX" --pilot-z "$PILOT_Z"

COMB="$CASE_DIR/constant/combustionProperties"
if [[ -f "$COMB" ]]; then
    cp -p "$COMB" "$COMB.before_premixed_variance_patch_${STAMP}"
    python3 - "$COMB" <<'PY'
from pathlib import Path
import re, sys
p = Path(sys.argv[1]); s = p.read_text()
for key in ('useProgressVariableVariance', 'useMixtureFractionVariance'):
    pat = rf'(\b{key}\s+)(true|yes|on|false|no|off)(\s*;)'
    s, n = re.subn(pat, rf'\g<1>false\g<3>', s)
    if n == 0:
        print(f'[WARN] {key} not found in {p}; add {key} false; in FGMModelCoeffs')
p.write_text(s)
print(f'[OK] Set FGM variance switches false where present in {p}')
PY
else
    echo "[WARN] Missing $COMB; set both FGM variance switches false manually."
fi

echo "[DONE] Premixed OpenFOAM-7 tables installed at: $CASE_DIR/constant/tables"
echo "[DONE] Z axis is 0 to $Z_MAX and brackets pilot Z=$PILOT_Z and main-inlet Z=$Z_MAX."
echo "[NEXT] Remove nonzero time directories and run from the 0 directory before comparing to the original table case."
