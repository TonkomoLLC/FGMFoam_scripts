#!/usr/bin/env python3
"""Validate OpenFOAM-7 nested FGM tables, constructor names, and case Z coverage."""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np


def parse_list(path: Path, key: str) -> np.ndarray:
    text = path.read_text().split(key, 1)[1]
    toks = re.findall(r'\(|\)|;|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    pos = 0
    def get():
        nonlocal pos
        n = int(float(toks[pos])); pos += 1
        if toks[pos] != '(':
            raise ValueError(f'{path}: expected list after length')
        pos += 1; result = []
        for _ in range(n):
            if pos + 1 < len(toks) and toks[pos + 1] == '(':
                result.append(get())
            else:
                result.append(float(toks[pos])); pos += 1
        if toks[pos] != ')':
            raise ValueError(f'{path}: unterminated list')
        pos += 1
        return result
    return np.asarray(get(), dtype=float)


def read_param(text: str, key: str) -> np.ndarray:
    match = re.search(rf'^\s*{key}\s*\(([^;]*)\)\s*;', text, re.S | re.M)
    if not match:
        raise ValueError(f'Missing {key}')
    return np.asarray([float(token) for token in match.group(1).split()])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--case', required=True)
    p.add_argument('--require-species', nargs='*', default=[])
    p.add_argument('--check-z', nargs='*', type=float, default=[0.04293, 0.1559],
                   help='Z boundary values which must lie within Z_param bounds')
    p.add_argument('--pilot-z', type=float, default=0.04293)
    args = p.parse_args()
    case = Path(args.case)
    props = (case / 'constant/tableProperties').read_text(errors='replace')
    pvprops = (case / 'constant/PVtableProperties').read_text(errors='replace')
    if not re.search(r'\binterpolationType\s+linearInterpolation\s*;', props):
        raise SystemExit('[FAIL] constant/tableProperties must contain: interpolationType linearInterpolation;')
    if not re.search(r'\binterpolationType\s+PVlinearInterpolation\s*;', pvprops):
        raise SystemExit('[FAIL] constant/PVtableProperties must contain: interpolationType PVlinearInterpolation;')
    match = re.search(r'tablePath\s+"?([^";]+)"?\s*;', props)
    if not match:
        raise SystemExit('[FAIL] tablePath is missing from constant/tableProperties')
    tables = case / match.group(1)
    axes = {key: read_param(props, key) for key in ('varPV_param', 'PV_param', 'varZ_param', 'Z_param')}
    pv_axes = {key: read_param(pvprops, key) for key in ('varZ_param', 'Z_param')}
    dims4 = tuple(len(axes[key]) for key in ('varPV_param', 'PV_param', 'varZ_param', 'Z_param'))
    dims2 = tuple(len(pv_axes[key]) for key in ('varZ_param', 'Z_param'))
    required4 = ['T', 'psi', 'mu', 'alpha', 'SourcePV', *args.require_species]
    for name in required4:
        f = tables / f'{name}_table'
        if not f.exists(): raise SystemExit(f'[FAIL] Missing required table: {f}')
        arr = parse_list(f, f'{name}_table')
        if arr.shape != dims4: raise SystemExit(f'[FAIL] {name}_table shape {arr.shape}, expected {dims4}')
    bounds = {}
    for name in ('PVmin', 'PVmax'):
        arr = parse_list(tables / f'{name}_table', f'{name}_table')
        if arr.shape != dims2: raise SystemExit(f'[FAIL] {name}_table shape {arr.shape}, expected {dims2}')
        bounds[name] = arr
    z_axis = axes['Z_param']; c_axis = axes['PV_param']
    for z in args.check_z:
        if z < z_axis[0] - 1e-12 or z > z_axis[-1] + 1e-12:
            raise SystemExit(f'[FAIL] Boundary/requested Z={z:g} is outside Z_param [{z_axis[0]:g}, {z_axis[-1]:g}]')
    source = parse_list(tables / 'SourcePV_table', 'SourcePV_table')
    iz = int(np.argmin(abs(z_axis - args.pilot_z)))
    source_row = source[0, :, 0, iz]
    jc = int(np.argmax(source_row))
    print(f'[OK] Table directory: {tables}')
    print(f'[OK] FGM field topology: {dims4}; PV bounds topology: {dims2}')
    print(f'[OK] Z_param coverage: {z_axis[0]:.8g} to {z_axis[-1]:.8g}; requested case Z values are bracketed.')
    print(f'[OK] Nearest pilot row: Z={z_axis[iz]:.8g}, max SourcePV={source_row[jc]:.6e} at scaledPV={c_axis[jc]:.5g}')
    print('[OK] OpenFOAM-7 constructor names and nested tables are consistent.')

if __name__ == '__main__':
    main()
