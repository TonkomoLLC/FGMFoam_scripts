#!/usr/bin/env python3
import os, io, tarfile, argparse
from pathlib import Path
import numpy as np

BANNER = r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield        | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration    | Version:  7                                     |
|   \\  /    A nd          | Web:      http://www.OpenFOAM.com               |
|    \\/     M anipulation |                                                 |
\*---------------------------------------------------------------------------*/"""

def _tar_exists(tf, relpath: str) -> bool:
    rel = relpath.replace("\\","/")
    for m in tf.getmembers():
        if not m.isfile(): continue
        if m.name.replace("\\","/").endswith(rel):
            return True
    return False

def _tar_read(tf, relpath: str) -> bytes:
    rel = relpath.replace("\\","/")
    for m in tf.getmembers():
        if not m.isfile(): continue
        if m.name.replace("\\","/").endswith(rel):
            return tf.extractfile(m).read()
    raise FileNotFoundError(relpath)

def _dir_read_any(root: Path, relpath: str) -> bytes:
    rel_norm = relpath.replace("\\","/")
    for p in root.rglob("*"):
        if p.is_file() and p.as_posix().endswith(rel_norm):
            return p.read_bytes()
    raise FileNotFoundError(f"{relpath} under {root}")

def load_tables(input_path: Path):
    data = {'axes': {}, 'fields': {}, 'extras': {}}

    def read_csv_from_bytes(b: bytes) -> np.ndarray:
        s = b.decode('utf-8')
        arr = np.genfromtxt(io.StringIO(s), delimiter=',')
        return arr

    is_tar = input_path.is_file()
    tf = None
    if is_tar:
        tf = tarfile.open(input_path, mode="r:*")

    # Axes (required)
    if is_tar:
        Zb  = _tar_read(tf, "axes/Z.csv")
        PVb = _tar_read(tf, "axes/PV.csv")
    else:
        Zb  = _dir_read_any(input_path, "axes/Z.csv")
        PVb = _dir_read_any(input_path, "axes/PV.csv")
    data['axes']['Z']  = np.atleast_1d(read_csv_from_bytes(Zb)).astype(float)
    data['axes']['PV'] = np.atleast_1d(read_csv_from_bytes(PVb)).astype(float)

    # Thermo (required)
    if is_tar:
        Tb   = _tar_read(tf, "thermo/T.csv")
        rhob = _tar_read(tf, "thermo/rho.csv")
    else:
        Tb   = _dir_read_any(input_path, "thermo/T.csv")
        rhob = _dir_read_any(input_path, "thermo/rho.csv")
    data['fields']['T']   = np.atleast_2d(read_csv_from_bytes(Tb)).astype(float)
    data['fields']['rho'] = np.atleast_2d(read_csv_from_bytes(rhob)).astype(float)

    # Species (optional)
    if is_tar:
        names = []
        for m in tf.getmembers():
            if not m.isfile(): continue
            name = m.name.replace("\\","/")
            if "/species/" in name and name.endswith(".csv") and Path(name).name.startswith("Y_"):
                names.append(Path(name).stem)
        names = sorted(set(names))
        for stem in names:
            b = _tar_read(tf, f"species/{stem}.csv")
            data['fields'][stem] = np.atleast_2d(read_csv_from_bytes(b)).astype(float)
    else:
        for p in input_path.rglob("species/Y_*.csv"):
            stem = p.stem
            data['fields'][stem] = np.atleast_2d(np.genfromtxt(p, delimiter=',')).astype(float)

    # Optional extras
    def maybe_read_extra(key, rels):
        for rel in rels:
            try:
                if is_tar:
                    if _tar_exists(tf, rel):
                        b = _tar_read(tf, rel)
                        data['extras'][key] = np.atleast_2d(read_csv_from_bytes(b)).astype(float)
                        return
                else:
                    b = _dir_read_any(input_path, rel)
                    data['extras'][key] = np.atleast_2d(read_csv_from_bytes(b)).astype(float)
                    return
            except FileNotFoundError:
                pass

    extra_map = {
        'psi':     ["thermo/psi.csv",     "extras/psi.csv"],
        'mu':      ["thermo/mu.csv",      "extras/mu.csv"],
        'Cps':     ["thermo/Cps.csv",     "extras/Cps.csv"],
        'alpha':   ["thermo/alpha.csv",   "extras/alpha.csv"],
        'SourcePV':["thermo/SourcePV.csv","extras/SourcePV.csv"],
        'PVmin':   ["thermo/PVmin.csv",   "extras/PVmin.csv"],
        'PVmax':   ["thermo/PVmax.csv",   "extras/PVmax.csv"],
    }
    for key, rels in extra_map.items():
        maybe_read_extra(key, rels)

    if tf is not None:
        tf.close()

    return data

def format_scalar_list(lst):
    return "( " + " ".join(f"{x:.12g}" for x in lst) + " )"

def write_table_dict(out_path: Path, object_name: str, dims: str, axes_names, axes_sizes, array2d: np.ndarray):
    out = io.StringIO()
    print(BANNER, file=out); print("", file=out)
    print("FoamFile\n{", file=out)
    print("    version     2.0;", file=out)
    print("    format      ascii;", file=out)
    print("    class       dictionary;", file=out)
    print(f"    object      {object_name};", file=out)
    print("}\n", file=out)

    if dims:
        print(f"dimensions      {dims};\n", file=out)

    print("axes            ( " + " ".join(axes_names) + " );", file=out)
    print("size            ( " + " ".join(str(int(s)) for s in axes_sizes) + " );\n", file=out)

    print("data", file=out)
    print("(", file=out)
    NZ, NPV = array2d.shape
    for i in range(NZ):
        row = " ".join(f"{array2d[i, j]:.12g}" for j in range(NPV))
        print(f"    ( {row} )", file=out)
    print(");\n", file=out)

    out_path.write_text(out.getvalue())

def emit_properties(out_root: Path, Z, PV, tablePath="tables"):
    tp = io.StringIO()
    print(BANNER, file=tp); print("", file=tp)
    print("FoamFile\n{", file=tp)
    print("    version     2.0;", file=tp)
    print("    format      ascii;", file=tp)
    print("    class       dictionary;", file=tp)
    print("    object      tableProperties;", file=tp)
    print("}\n", file=tp)
    print(f'tablePath       "{tablePath}";', file=tp)
    print("interpolationType    linear;", file=tp)
    print(f"varPV_param     ( 0 );", file=tp)
    print("PV_param        " + format_scalar_list(PV) + ";", file=tp)
    print(f"varZ_param      ( 0 );", file=tp)
    print("Z_param         " + format_scalar_list(Z) + ";", file=tp)
    (out_root/"tableProperties").write_text(tp.getvalue())

    pv = io.StringIO()
    print(BANNER, file=pv); print("", file=pv)
    print("FoamFile\n{", file=pv)
    print("    version     2.0;", file=pv)
    print("    format      ascii;", file=pv)
    print("    class       dictionary;", file=pv)
    print("    object      PVtableProperties;", file=pv)
    print("}\n", file=pv)
    print(f'tablePath       "{tablePath}";', file=pv)
    print("interpolationType    linear;", file=pv)
    print(f"varZ_param      ( 0 );", file=pv)
    print("Z_param         " + format_scalar_list(Z) + ";", file=pv)
    (out_root/"PVtableProperties").write_text(pv.getvalue())

def maybe_compute_thermo_extras(fields, P, mech):
    if mech is None:
        return {}
    try:
        import cantera as ct
    except Exception:
        print("[WARN] Cantera not available; skipping thermo extras.")
        return {}

    gas = ct.Solution(mech)
    sp_names = gas.species_names

    T = fields['T']; rho = fields['rho']
    NZ, NPV = T.shape
    Ystack = np.zeros((NZ, NPV, len(sp_names)), dtype=float)
    for name in list(fields.keys()):
        if not name.startswith("Y_"): continue
        sp = name[2:]
        if sp not in sp_names:
            print(f"[WARN] Species {sp} not in mechanism; skipping thermo extras."); return {}
        j = sp_names.index(sp)
        Ystack[:,:,j] = fields[name]

    mu = np.zeros_like(T); cp = np.zeros_like(T); k = np.zeros_like(T); psi = np.zeros_like(T)
    for i in range(NZ):
        for j in range(NPV):
            gas.TPY = float(T[i,j]), float(P), Ystack[i,j,:]
            mu[i,j]  = gas.viscosity if hasattr(gas, 'viscosity') else gas.viscosity_mole
            cp[i,j]  = gas.cp_mass
            k[i,j]   = gas.thermal_conductivity
            psi[i,j] = gas.density/ P
    alpha = k/ np.maximum(rho*cp, 1e-300)
    return {'mu': mu, 'Cps': cp, 'alpha': alpha, 'psi': psi}

def main():
    ap = argparse.ArgumentParser(description="Convert CSV-based FGM tables to OpenFOAM dictionary tables.")
    ap.add_argument("--in", dest="inp", required=True, help="Input 02_tables folder (or its parent) or 02_tables.tar.xz")
    ap.add_argument("--out", dest="out", required=True, help="Output case directory (writes to OUT/constant/...)")
    ap.add_argument("--nz", type=int, default=None, help="Override NZ if needed (infer by default)")
    ap.add_argument("--npv", type=int, default=None, help="Override NPV if needed (infer by default)")
    ap.add_argument("--pressure", type=float, default=101325.0, help="Pressure [Pa] for psi/alpha computations (default 101325)")
    ap.add_argument("--mech", type=str, default=None, help="Optional Cantera mechanism to compute mu, Cps, alpha, psi")
    ap.add_argument("--emit-pv-bounds", action="store_true", help="Also emit PVmin_table=0 and PVmax_table=1")
    ap.add_argument("--emit-zero-sourcepv", action="store_true", help="Emit SourcePV_table with zeros")
    args = ap.parse_args()

    inpath = Path(args.inp).resolve()
    out_case = Path(args.out).resolve()
    const_dir = out_case/"constant"
    tables_dir = const_dir/"tables"; tables_dir.mkdir(parents=True, exist_ok=True)

    data = load_tables(inpath)

    Z = np.array(data['axes']['Z'], dtype=float).reshape(-1)
    PV = np.array(data['axes']['PV'], dtype=float).reshape(-1)

    NZ = args.nz or data['fields']['T'].shape[0]
    NPV = args.npv or data['fields']['T'].shape[1]

    for name, arr in data['fields'].items():
        if arr.shape != (NZ, NPV):
            raise SystemExit(f"Field '{name}' has shape {arr.shape}, expected {(NZ, NPV)}")

    emit_properties(const_dir, Z, PV, tablePath="tables")

    dims = {
        'T':        "[0 0 0 1 0 0 0]",
        'rho':      "[1 -3 0 0 0 0 0]",
        'Y':        "[0 0 0 0 0 0 0]",
        'psi':      "[0 -2 2 0 0 0 0]",
        'mu':       "[1 -1 -1 0 0 0 0]",
        'Cps':      "[0 2 -2 -1 0 0 0]",
        'alpha':    "[0 2 -1 0 0 0 0]",
        'SourcePV': "[0 0 -1 0 0 0 0]",
        'PV':       "[0 0 0 0 0 0 0]",
    }
    axes_names = ("Z","PV")
    axes_sizes = (NZ, NPV)

    write_table_dict(tables_dir/"T_table",   "T_table",   dims['T'],   axes_names, axes_sizes, data['fields']['T'])
    write_table_dict(tables_dir/"rho_table", "rho_table", dims['rho'], axes_names, axes_sizes, data['fields']['rho'])

    for name, arr in data['fields'].items():
        if name in ("T","rho"): continue
        obj = f"{name}_table"
        write_table_dict(tables_dir/obj, obj, dims['Y'], axes_names, axes_sizes, arr)

    for key, arr in data['extras'].items():
        obj = f"{key}_table" if not key.endswith("_table") else key
        dim_key = key if key in dims else 'PV'
        write_table_dict(tables_dir/f"{obj}", f"{obj}", dims[dim_key], axes_names, axes_sizes, arr)

    if args.emit_pv_bounds:
        write_table_dict(tables_dir/"PVmin_table", "PVmin_table", dims['PV'], axes_names, axes_sizes, np.zeros((NZ,NPV)))
        write_table_dict(tables_dir/"PVmax_table", "PVmax_table", dims['PV'], axes_names, axes_sizes, np.ones((NZ,NPV)))
    if args.emit_zero_sourcepv:
        write_table_dict(tables_dir/"SourcePV_table", "SourcePV_table", dims['SourcePV'], axes_names, axes_sizes, np.zeros((NZ,NPV)))

    thermo_extras = maybe_compute_thermo_extras(data['fields'], args.pressure, args.mech)
    for key, arr in thermo_extras.items():
        obj = f"{key}_table"
        write_table_dict(tables_dir/obj, obj, dims[key], axes_names, axes_sizes, arr)

    print(f"[OK] Wrote OpenFOAM tables to: {tables_dir}")
    print(f"[OK] Wrote: {const_dir/'tableProperties'} and {const_dir/'PVtableProperties'}")
    print("If your model expects SourcePV_table, PVmin_table, PVmax_table, psi_table, mu_table, Cps_table, alpha_table,")
    print("use --emit-zero-sourcepv, --emit-pv-bounds, and/or --mech to create them.")

if __name__ == "__main__":
    main()
