#!/usr/bin/env python3
'''Add the six zero-variance progress-variable auxiliary tables to installed v5 tables.'''
from __future__ import annotations
import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path
import numpy as np
from csv2of_tables import write_fgm_table, write_pv_table
from validate_of7_premixed_tables import parse_list, read_param


def set_progress_true(path: Path) -> None:
    text = path.read_text(errors="replace")
    pattern = r"(?m)^(\s*)(useProgressVariableVariance\s+)(true|false|yes|no|on|off)(\s*;)"
    changed, count = re.subn(pattern, r"\1\2true\4", text)
    if count == 0:
        raise SystemExit(f"[FAIL] Active useProgressVariableVariance entry not found in {path}.")
    path.write_text(changed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--table-dir", default="constant/tables")
    p.add_argument("--enable-progress-variable-variance", action="store_true")
    args = p.parse_args()
    case = Path(args.case).resolve()
    table_dir = case / args.table_dir
    props = (case / "constant/tableProperties").read_text(errors="replace")
    varpv = read_param(props, "varPV_param")
    varz = read_param(props, "varZ_param")
    c_axis = read_param(props, "PV_param")
    z_axis = read_param(props, "Z_param")
    ipv0 = int(np.argmin(np.abs(varpv)))
    izeta0 = int(np.argmin(np.abs(varz)))
    if abs(varpv[ipv0]) > 1e-12 or abs(varz[izeta0]) > 1e-12:
        raise SystemExit("[FAIL] Existing table axes do not contain zero variance coordinates.")

    source4 = parse_list(table_dir / "SourcePV_table", "SourcePV_table")
    pvmin2 = parse_list(table_dir / "PVmin_table", "PVmin_table")
    pvmax2 = parse_list(table_dir / "PVmax_table", "PVmax_table")
    expected_shape = (len(varpv), len(c_axis), len(varz), len(z_axis))
    if source4.shape != expected_shape:
        raise SystemExit(f"[FAIL] SourcePV shape {source4.shape}; expected {expected_shape}.")
    source = source4[ipv0, :, izeta0, :]
    pvmin = pvmin2[izeta0, :]
    pvmax = pvmax2[izeta0, :]
    spread = max(
        float(np.max(np.abs(source4 - source[None, :, None, :]))),
        float(np.max(np.abs(pvmin2 - pvmin[None, :]))),
        float(np.max(np.abs(pvmax2 - pvmax[None, :])))
    )
    if spread > 1e-12:
        raise SystemExit(f"[FAIL] Existing tables are not replicated zero-variance tables; spread={spread:.6e}.")

    pv = pvmin[None, :] + c_axis[:, None]*(pvmax - pvmin)[None, :]
    auxiliary_fields_cz = {
        "YWI": pv*source,
        "YuWI": pvmin[None, :]*source,
        "YbWI": pvmax[None, :]*source,
    }
    auxiliary_bounds = {
        "Yu2I": pvmin*pvmin,
        "YuYbI": pvmin*pvmax,
        "Yb2I": pvmax*pvmax,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = table_dir.parent / f"{table_dir.name}.before_zeroVarPV_upgrade_{stamp}"
    shutil.copytree(table_dir, backup)
    for name, array_cz in auxiliary_fields_cz.items():
        write_fgm_table(table_dir / f"{name}_table", name, array_cz.T, varpv, varz)
    for name, array_z in auxiliary_bounds.items():
        write_pv_table(table_dir / f"{name}_table", name, array_z, varz)
    if args.enable_progress_variable_variance:
        combustion = case / "constant/combustionProperties"
        shutil.copy2(combustion, combustion.with_name(combustion.name + f".before_zeroVarPV_upgrade_{stamp}"))
        set_progress_true(combustion)
    print(f"[BACKUP] Existing tables copied to {backup}")
    print("[OK] Wrote YWI, YuWI, YbWI, Yu2I, YuYbI, and Yb2I.")
    if args.enable_progress_variable_variance:
        print("[OK] Set useProgressVariableVariance true.")
    print("[LIMITATION] Valid only for zero varPV/varZ; variance slices remain replicated.")


if __name__ == "__main__":
    main()
