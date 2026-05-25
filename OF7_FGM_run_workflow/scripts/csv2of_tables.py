#!/usr/bin/env python3
'''Write full OpenFOAM-7 premixed FGM nested-list tables.

Topology required by the supplied solver:
  FGM fields: (PVeta/varPV_param, scaledPV/PV_param, Zeta/varZ_param, Z_param)
  PV family:  (Zeta/varZ_param, Z_param)

The additional tables required by useProgressVariableVariance=true are included.
At this development stage, all variance-axis slices are replicated: the table set is
valid for varPV=0 and varZ=0, not for a nonzero presumed-PDF calculation.
'''
from __future__ import annotations
import argparse
import io
import tarfile
from pathlib import Path
import numpy as np

BANNER = r'''/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \      /  F ield          | OpenFOAM: The Open Source CFD Toolbox           |
|  \    /   O peration      | Version:  7                                     |
|   \  /    A nd            | Web:      http://www.OpenFOAM.com               |
|    \/     M anipulation   |                                                 |
\*---------------------------------------------------------------------------*/'''
FIELD_MANDATORY = ["T", "psi", "mu", "alpha", "SourcePV", "YWI", "YuWI", "YbWI"]
PV_MANDATORY = ["PVmin", "PVmax", "Yu2I", "YuYbI", "Yb2I"]


def read_csv(tf: tarfile.TarFile, suffix: str) -> np.ndarray:
    for member in tf.getmembers():
        if member.isfile() and member.name.replace("\\", "/").endswith(suffix):
            return np.genfromtxt(io.StringIO(tf.extractfile(member).read().decode()), delimiter=",")
    raise FileNotFoundError(suffix)


def optional_csv(tf: tarfile.TarFile, suffix: str) -> np.ndarray | None:
    try:
        return read_csv(tf, suffix)
    except FileNotFoundError:
        return None


def fmt_list(values: np.ndarray) -> str:
    return "( " + " ".join(f"{float(x):.12g}" for x in values) + " )"


def header(keyword: str) -> str:
    return BANNER + '''
FoamFile
{   version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      LDMtable;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

''' + keyword + "\n"


def list_open(out, n: int) -> None:
    out.write(f"{int(n)}\n(\n\n")


def list_close(out) -> None:
    out.write(")\n\n")


def write_leaf(out, row: np.ndarray) -> None:
    list_open(out, len(row))
    for value in row:
        out.write(f" {float(value):.8e}\n")
    list_close(out)


def write_fgm_table(path: Path, name: str, array_zc: np.ndarray, varpv: np.ndarray, varz: np.ndarray) -> None:
    array_zc = np.asarray(array_zc, dtype=float)
    _, nc = array_zc.shape
    with path.open("w") as out:
        out.write(header(f"{name}_table"))
        list_open(out, len(varpv))
        for _ in varpv:
            list_open(out, nc)
            for jc in range(nc):
                list_open(out, len(varz))
                for _ in varz:
                    write_leaf(out, array_zc[:, jc])
                list_close(out)
            list_close(out)
        list_close(out)
        out.write(";\n")


def write_pv_table(path: Path, name: str, row_z: np.ndarray, varz: np.ndarray) -> None:
    row_z = np.asarray(row_z, dtype=float).reshape(-1)
    with path.open("w") as out:
        out.write(header(f"{name}_table"))
        list_open(out, len(varz))
        for _ in varz:
            write_leaf(out, row_z)
        list_close(out)
        out.write(";\n")


def write_properties(const: Path, table_path: str, z: np.ndarray, c: np.ndarray, varz: np.ndarray, varpv: np.ndarray) -> None:
    props = BANNER + f'''
FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object tableProperties;
}}

tablePath           "{table_path}";
interpolationType   linearInterpolation;
varPV_param         {fmt_list(varpv)};
PV_param            {fmt_list(c)};
varZ_param          {fmt_list(varz)};
Z_param             {fmt_list(z)};
'''
    pvprops = BANNER + f'''
FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object PVtableProperties;
}}

tablePath           "{table_path}";
interpolationType   PVlinearInterpolation;
varZ_param          {fmt_list(varz)};
Z_param             {fmt_list(z)};
'''
    (const / "tableProperties").write_text(props)
    (const / "PVtableProperties").write_text(pvprops)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--table-dir", default="constant/tables")
    ap.add_argument("--varz", nargs="+", type=float, default=[0.0, 0.25, 0.50, 0.75, 0.99])
    ap.add_argument("--varpv", nargs="+", type=float, default=[0.0, 0.25, 0.50, 0.75, 0.99])
    args = ap.parse_args()
    case = Path(args.case).resolve()
    const = case / "constant"
    const.mkdir(parents=True, exist_ok=True)
    table_dir = case / args.table_dir
    table_dir.mkdir(parents=True, exist_ok=True)
    varz = np.asarray(args.varz, dtype=float)
    varpv = np.asarray(args.varpv, dtype=float)
    if len(varz) < 2 or len(varpv) < 2:
        raise SystemExit("The supplied OpenFOAM-7 interpolators require at least two entries on both variance axes.")
    if np.any(np.diff(varz) <= 0) or np.any(np.diff(varpv) <= 0):
        raise SystemExit("Variance axes must be strictly increasing.")

    with tarfile.open(args.inp, "r:*") as tf:
        z = np.asarray(read_csv(tf, "axes/Z.csv")).reshape(-1)
        c = np.asarray(read_csv(tf, "axes/PV.csv")).reshape(-1)
        sources = {
            "T": "thermo/T.csv", "psi": "thermo/psi.csv", "mu": "thermo/mu.csv",
            "alpha": "thermo/alpha.csv", "Cps": "thermo/Cps.csv", "rho": "thermo/rho.csv",
            "SourcePV": "extras/SourcePV.csv", "YWI": "extras/YWI.csv",
            "YuWI": "extras/YuWI.csv", "YbWI": "extras/YbWI.csv",
        }
        fields = {name: optional_csv(tf, suffix) for name, suffix in sources.items()}
        pv_sources = {
            "PVmin": "extras/PVmin.csv", "PVmax": "extras/PVmax.csv",
            "Yu2I": "extras/Yu2I.csv", "YuYbI": "extras/YuYbI.csv", "Yb2I": "extras/Yb2I.csv",
        }
        pvtables = {name: read_csv(tf, suffix).reshape(-1) for name, suffix in pv_sources.items()}
        species = {}
        for member in tf.getmembers():
            stem = Path(member.name).stem
            if member.isfile() and "/species/" in member.name.replace("\\", "/") and stem.startswith("Y_"):
                species[stem[2:]] = np.genfromtxt(io.StringIO(tf.extractfile(member).read().decode()), delimiter=",")

    for name in FIELD_MANDATORY:
        if fields[name] is None:
            raise SystemExit(f"Missing mandatory table {name}; rebuild the archive with the v6 builder.")
    for name, arr in {**{k: v for k, v in fields.items() if v is not None}, **species}.items():
        if np.asarray(arr).shape != (len(z), len(c)):
            raise SystemExit(f"{name} shape {np.asarray(arr).shape} does not match ({len(z)}, {len(c)})")
    for name in PV_MANDATORY:
        if np.asarray(pvtables[name]).shape != (len(z),):
            raise SystemExit(f"{name} shape {np.asarray(pvtables[name]).shape} does not match ({len(z)},)")

    write_properties(const, args.table_dir, z, c, varz, varpv)
    for name, arr in fields.items():
        if arr is not None:
            write_fgm_table(table_dir / f"{name}_table", name, np.asarray(arr), varpv, varz)
    for name, arr in species.items():
        write_fgm_table(table_dir / f"{name}_table", name, np.asarray(arr), varpv, varz)
    for name, arr in pvtables.items():
        write_pv_table(table_dir / f"{name}_table", name, arr, varz)
    print(f"[OK] Wrote complete OpenFOAM-7 zero-variance progress-variable-variance table set to {table_dir}")
    print("[REQUIRED] useProgressVariableVariance may be true only while varPV=0 and varZ=0 for this replicated-variance table set.")


if __name__ == "__main__":
    main()
