#!/usr/bin/env python3
"""Write OpenFOAM-7 FGM nested-list tables from 02_tables_of7.tar.xz.

Output topology used by the supplied solver:
  FGM fields: (varPV_param, PV_param, varZ_param, Z_param)
  PV bounds:  (varZ_param, Z_param)
The emitted default is a physically meaningful variance-neutral 2D manifold replicated
over safe variance axes; `useProgressVariableVariance` must be `false`.
"""
from __future__ import annotations
import argparse, io, json, tarfile
from pathlib import Path
import numpy as np

BANNER = r'''/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \      /  F ield          | OpenFOAM: The Open Source CFD Toolbox           |
|  \    /   O peration      | Version:  7                                     |
|   \  /    A nd            | Web:      http://www.OpenFOAM.com               |
|    \/     M anipulation   |                                                 |
\*---------------------------------------------------------------------------*/'''
MANDATORY = ["T", "psi", "mu", "alpha", "SourcePV", "PVmin", "PVmax"]

def read_csv(tf, suffix):
    for m in tf.getmembers():
        if m.isfile() and m.name.replace('\\','/').endswith(suffix):
            return np.genfromtxt(io.StringIO(tf.extractfile(m).read().decode()), delimiter=',')
    raise FileNotFoundError(suffix)

def optional_csv(tf, suffix):
    try: return read_csv(tf, suffix)
    except FileNotFoundError: return None

def fmt_list(a): return "( " + " ".join(f"{float(x):.12g}" for x in a) + " )"

def header(keyword):
    return BANNER + '''
FoamFile
{   version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      LDMtable;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

''' + keyword + '\n'

def list_open(out, n): out.write(f"{int(n)}\n(\n\n")
def list_close(out): out.write(")\n\n")
def write_leaf(out, row):
    list_open(out, len(row))
    for value in row: out.write(f" {float(value):.8e}\n")
    list_close(out)

def write_fgm_table(path, name, array_zc, varpv, varz):
    nz, nc = array_zc.shape
    with path.open('w') as out:
        out.write(header(f"{name}_table")); list_open(out, len(varpv))
        for _ in varpv:
            list_open(out, nc)
            for jc in range(nc):
                list_open(out, len(varz))
                for _ in varz: write_leaf(out, array_zc[:, jc])
                list_close(out)
            list_close(out)
        list_close(out); out.write(';\n')

def write_pv_table(path, name, row_z, varz):
    row_z = np.asarray(row_z).reshape(-1)
    with path.open('w') as out:
        out.write(header(f"{name}_table")); list_open(out, len(varz))
        for _ in varz: write_leaf(out, row_z)
        list_close(out); out.write(';\n')

def write_properties(const, table_path, z, c, varz, varpv):
    props = BANNER + f'''\nFoamFile\n{{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object tableProperties;\n}}\n\ntablePath           "{table_path}";\ninterpolationType   linearInterpolation;\nvarPV_param         {fmt_list(varpv)};\nPV_param            {fmt_list(c)};\nvarZ_param          {fmt_list(varz)};\nZ_param             {fmt_list(z)};\n'''
    pvprops = BANNER + f'''\nFoamFile\n{{\n    version 2.0;\n    format ascii;\n    class dictionary;\n    object PVtableProperties;\n}}\n\ntablePath           "{table_path}";\ninterpolationType   PVlinearInterpolation;\nvarZ_param          {fmt_list(varz)};\nZ_param             {fmt_list(z)};\n'''
    (const/'tableProperties').write_text(props); (const/'PVtableProperties').write_text(pvprops)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, help='02_tables_of7.tar.xz')
    ap.add_argument('--case', required=True, help='OpenFOAM case directory')
    ap.add_argument('--table-dir', default='constant/tables', help='Path relative to the case')
    ap.add_argument('--varz', nargs='+', type=float, default=[0.0, 0.25, 0.50, 0.75, 0.99],
                    help='Safe replicated varZ axis; do not interpret as PDF-integrated data')
    ap.add_argument('--varpv', nargs='+', type=float, default=[0.0, 0.25, 0.50, 0.75, 0.99],
                    help='Safe replicated varPV axis; used only because the OF7 interpolator requires adjacent entries')
    args = ap.parse_args()
    case = Path(args.case).resolve(); const = case/'constant'; const.mkdir(parents=True, exist_ok=True)
    table_dir = case/args.table_dir; table_dir.mkdir(parents=True, exist_ok=True)
    varz = np.array(args.varz); varpv = np.array(args.varpv)
    if len(varz) < 2 or len(varpv) < 2:
        raise SystemExit('The supplied OpenFOAM-7 interpolators require at least two entries in both variance axes. Use the defaults and set useProgressVariableVariance false.')
    if np.any(np.diff(varz) <= 0) or np.any(np.diff(varpv) <= 0):
        raise SystemExit('Variance axes must be strictly increasing.')
    with tarfile.open(args.inp, 'r:*') as tf:
        z = np.asarray(read_csv(tf, 'axes/Z.csv')).reshape(-1); c = np.asarray(read_csv(tf, 'axes/PV.csv')).reshape(-1)
        table_sources = {'T':'thermo/T.csv', 'psi':'thermo/psi.csv', 'mu':'thermo/mu.csv', 'alpha':'thermo/alpha.csv',
                         'Cps':'thermo/Cps.csv', 'rho':'thermo/rho.csv', 'SourcePV':'extras/SourcePV.csv'}
        tables = {n: optional_csv(tf, suffix) for n, suffix in table_sources.items()}
        pvmin = read_csv(tf, 'extras/PVmin.csv').reshape(-1); pvmax = read_csv(tf, 'extras/PVmax.csv').reshape(-1)
        species = {}
        for m in tf.getmembers():
            stem = Path(m.name).stem
            if m.isfile() and '/species/' in m.name.replace('\\','/') and stem.startswith('Y_'):
                species[stem[2:]] = np.genfromtxt(io.StringIO(tf.extractfile(m).read().decode()), delimiter=',')
    for n in ['T','psi','mu','alpha','SourcePV']:
        if tables[n] is None: raise SystemExit(f'Missing mandatory field table {n}')
    for n, arr in {**{k:v for k,v in tables.items() if v is not None}, **species}.items():
        if np.asarray(arr).shape != (len(z), len(c)):
            raise SystemExit(f'{n} shape {np.asarray(arr).shape} does not match ({len(z)}, {len(c)})')
    write_properties(const, args.table_dir, z, c, varz, varpv)
    for n, arr in tables.items():
        if arr is not None: write_fgm_table(table_dir/f'{n}_table', n, np.asarray(arr), varpv, varz)
    for n, arr in species.items(): write_fgm_table(table_dir/f'{n}_table', n, np.asarray(arr), varpv, varz)
    write_pv_table(table_dir/'PVmin_table', 'PVmin', pvmin, varz); write_pv_table(table_dir/'PVmax_table', 'PVmax', pvmax, varz)
    print(f'[OK] Wrote OpenFOAM-7 variance-neutral nested tables to {table_dir}')
    print(f'[OK] Required runtime tables present: {", ".join(MANDATORY)}')
    print('[REQUIRED] In FGMModelCoeffs, set useProgressVariableVariance false for these tables.')

if __name__ == '__main__': main()
