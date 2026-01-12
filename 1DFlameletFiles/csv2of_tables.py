#!/usr/bin/env python3
"""
csv2of_tables.py

Convert CSV-based FGM tables (02_tables) into OpenFOAM dictionary tables for FGMFoam (OF7).

Integrated tables (tableSolver): T, rho, psi, mu, Cps, alpha, SourcePV, species Y_*
  List<List<List<scalarList>>> indexed [varPV][PV][varZ][Z]

PV tables (PVtableSolver): PVmin_table, PVmax_table
  List<scalarList> indexed [varZ][Z]

Robustness:
- varPV_param and varZ_param must have size >= 2 (defaults: 0 1).
- PV bounds handling:
    * prefer axes/PVmin.csv and axes/PVmax.csv (physical PV units, length NZ)
    * else fall back to 0/1 (stock-like), optionally forced by --emit-pv-bounds
- Writes tables to the directory implied by --tablePath without creating accidental nesting.
- Guarantees mu_table/alpha_table exist (Cantera if possible; else constant fallbacks).
- Guarantees SourcePV_table exists:
    * uses thermo/SourcePV.csv (or extras/SourcePV.csv) if present
    * else writes zeros and WARNs (otherwise PV won’t self-increase).

"""

import io
import tarfile
import argparse
from pathlib import Path
import numpy as np


BANNER = r"""/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield        | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration    | Version:  7                                     |
|   \\  /    A nd          | Web:      http://www.OpenFOAM.com               |
|    \\/     M anipulation |                                                 |
\*---------------------------------------------------------------------------*/"""


def _tar_read(tf: tarfile.TarFile, relpath: str) -> bytes:
    rel = relpath.replace("\\", "/")
    for m in tf.getmembers():
        if not m.isfile():
            continue
        name = m.name.replace("\\", "/")
        if name.endswith(rel):
            return tf.extractfile(m).read()
    raise FileNotFoundError(relpath)


def _tar_list(tf: tarfile.TarFile):
    return [m.name.replace("\\", "/") for m in tf.getmembers() if m.isfile()]


def _dir_read_any(root: Path, relpath: str) -> bytes:
    rel_norm = relpath.replace("\\", "/")
    for p in root.rglob("*"):
        if p.is_file() and p.as_posix().endswith(rel_norm):
            return p.read_bytes()
    raise FileNotFoundError(f"{relpath} under {root}")


def _read_csv_bytes(b: bytes) -> np.ndarray:
    s = b.decode("utf-8")
    return np.genfromtxt(io.StringIO(s), delimiter=",")


def load_tables(input_path: Path):
    data = {"axes": {}, "fields": {}, "extras": {}}

    is_tar = input_path.is_file()
    tf = None
    names = None
    if is_tar:
        tf = tarfile.open(input_path, mode="r:*")
        names = _tar_list(tf)

    def read_required(rel):
        if is_tar:
            return _read_csv_bytes(_tar_read(tf, rel))
        return _read_csv_bytes(_dir_read_any(input_path, rel))

    def read_optional(rel):
        try:
            return read_required(rel)
        except FileNotFoundError:
            return None

    # axes
    Z = np.atleast_1d(read_required("axes/Z.csv")).astype(float).squeeze()
    PV = np.atleast_1d(read_required("axes/PV.csv")).astype(float).squeeze()

    if Z.ndim != 1 or PV.ndim != 1:
        raise SystemExit("axes/Z.csv and axes/PV.csv must be 1D arrays (NZx1 or NZ).")

    data["axes"]["Z"] = Z
    data["axes"]["PV"] = PV

    # optional PV bounds
    PVmin = read_optional("axes/PVmin.csv")
    PVmax = read_optional("axes/PVmax.csv")
    if PVmin is not None and PVmax is not None:
        data["axes"]["PVmin"] = np.atleast_1d(PVmin).astype(float).squeeze()
        data["axes"]["PVmax"] = np.atleast_1d(PVmax).astype(float).squeeze()

    # required thermo
    data["fields"]["T"] = np.atleast_2d(read_required("thermo/T.csv")).astype(float)
    data["fields"]["rho"] = np.atleast_2d(read_required("thermo/rho.csv")).astype(float)

    # species
    if is_tar:
        stems = []
        for n in names:
            if "/species/" in n and n.endswith(".csv"):
                base = Path(n).name
                if base.startswith("Y_"):
                    stems.append(Path(base).stem)
        for stem in sorted(set(stems)):
            b = _tar_read(tf, f"species/{stem}.csv")
            data["fields"][stem] = np.atleast_2d(_read_csv_bytes(b)).astype(float)
    else:
        for p in input_path.rglob("species/Y_*.csv"):
            data["fields"][p.stem] = np.atleast_2d(np.genfromtxt(p, delimiter=",")).astype(float)

    # optional extras / additional thermo
    extra_map = {
        "psi":      ["thermo/psi.csv",      "extras/psi.csv"],
        "mu":       ["thermo/mu.csv",       "extras/mu.csv"],
        "Cps":      ["thermo/Cps.csv",      "extras/Cps.csv"],
        "alpha":    ["thermo/alpha.csv",    "extras/alpha.csv"],
        "SourcePV": ["thermo/SourcePV.csv", "extras/SourcePV.csv"],  # IMPORTANT
    }

    def maybe_read_extra(key, rels):
        for rel in rels:
            try:
                if is_tar:
                    if not any(n.endswith(rel.replace("\\", "/")) for n in names):
                        continue
                    b = _tar_read(tf, rel)
                else:
                    b = _dir_read_any(input_path, rel)
                data["extras"][key] = np.atleast_2d(_read_csv_bytes(b)).astype(float)
                return
            except FileNotFoundError:
                continue

    for k, rels in extra_map.items():
        maybe_read_extra(k, rels)

    if tf is not None:
        tf.close()

    return data


def write_integrated_table_4d(out_path: Path, table_name: str, dims: str,
                             Z: np.ndarray, PV: np.ndarray,
                             varPV_param: np.ndarray, varZ_param: np.ndarray,
                             arr_ZxPV: np.ndarray):
    arr = np.asarray(arr_ZxPV, dtype=float)
    NZ, NPV = arr.shape

    if NZ != len(Z) or NPV != len(PV):
        raise SystemExit(f"{table_name}: shape {arr.shape} != (NZ,NPV)=({len(Z)},{len(PV)})")

    nVarPV = len(varPV_param)
    nVarZ = len(varZ_param)
    if nVarPV < 2 or nVarZ < 2:
        raise SystemExit(f"{table_name}: varPV_param and varZ_param must have size >= 2 (got {nVarPV},{nVarZ})")

    out = io.StringIO()
    print(BANNER, file=out); print("", file=out)
    print("FoamFile\n{", file=out)
    print("    version     2.0;", file=out)
    print("    format      ascii;", file=out)
    print("    class       dictionary;", file=out)
    print(f"    object      {table_name};", file=out)
    print("}\n", file=out)

    if dims:
        print(f"dimensions      {dims};\n", file=out)

    print(f"{table_name}", file=out)
    print(f"{nVarPV}", file=out)
    print("(", file=out)

    for _ipv in range(nVarPV):
        print(f"  {NPV}", file=out)
        print("  (", file=out)

        for j in range(NPV):
            z_list = arr[:, j]

            print(f"    {nVarZ}", file=out)
            print("    (", file=out)

            for _iz in range(nVarZ):
                print(f"      {NZ}", file=out)
                print("      (", file=out)
                for val in z_list:
                    print(f"        {float(val):.16g}", file=out)
                print("      )", file=out)

            print("    )", file=out)

        print("  )", file=out)

    print(")", file=out)
    print(";\n", file=out)
    out_path.write_text(out.getvalue())


def write_pv_table_2d(out_path: Path, table_name: str, dims: str,
                      Z: np.ndarray, varZ_param: np.ndarray,
                      values_Z: np.ndarray):
    v = np.asarray(values_Z, dtype=float).reshape(-1)
    NZ = len(Z)

    nVarZ = len(varZ_param)
    if nVarZ < 2:
        raise SystemExit(f"{table_name}: varZ_param must have size >= 2 (got {nVarZ})")
    if v.size != NZ:
        raise SystemExit(f"{table_name}: must be length NZ={NZ}, got {v.size}")

    out = io.StringIO()
    print(BANNER, file=out); print("", file=out)
    print("FoamFile\n{", file=out)
    print("    version     2.0;", file=out)
    print("    format      ascii;", file=out)
    print("    class       dictionary;", file=out)
    print(f"    object      {table_name};", file=out)
    print("}\n", file=out)

    if dims:
        print(f"dimensions      {dims};\n", file=out)

    print(f"{table_name}", file=out)
    print(f"{nVarZ}", file=out)
    print("(", file=out)

    for _ in range(nVarZ):
        print(f"  {NZ}", file=out)
        print("  (", file=out)
        for val in v:
            print(f"    {float(val):.16g}", file=out)
        print("  )", file=out)

    print(")", file=out)
    print(";\n", file=out)

    out_path.write_text(out.getvalue())


def format_scalar_list(lst):
    lst = np.asarray(lst, dtype=float).reshape(-1)
    return "( " + " ".join(f"{x:.12g}" for x in lst) + " )"


def emit_properties(const_dir: Path, Z, PV, varPV_param, varZ_param, tablePath_string: str):
    const_dir.mkdir(parents=True, exist_ok=True)

    tp = io.StringIO()
    print(BANNER, file=tp); print("", file=tp)
    print("FoamFile\n{", file=tp)
    print("    version     2.0;", file=tp)
    print("    format      ascii;", file=tp)
    print("    class       dictionary;", file=tp)
    print("    object      tableProperties;", file=tp)
    print("}\n", file=tp)

    tp_string = tablePath_string
    if tp_string.strip().lstrip("./") == "tables":
        tp_string = "constant/tables"
    print(f'tablePath       "{tp_string}";', file=tp)
    print("interpolationType    linearInterpolation;", file=tp)
    print("varPV_param     " + format_scalar_list(varPV_param) + ";", file=tp)
    print("PV_param        " + format_scalar_list(PV) + ";", file=tp)
    print("varZ_param      " + format_scalar_list(varZ_param) + ";", file=tp)
    print("Z_param         " + format_scalar_list(Z) + ";", file=tp)
    (const_dir / "tableProperties").write_text(tp.getvalue())

    pv = io.StringIO()
    print(BANNER, file=pv); print("", file=pv)
    print("FoamFile\n{", file=pv)
    print("    version     2.0;", file=pv)
    print("    format      ascii;", file=pv)
    print("    class       dictionary;", file=pv)
    print("    object      PVtableProperties;", file=pv)
    print("}\n", file=pv)

    print(f'tablePath       "{tablePath_string}";', file=pv)
    print("interpolationType    PVlinearInterpolation;", file=pv)
    print("varZ_param      " + format_scalar_list(varZ_param) + ";", file=pv)
    print("Z_param         " + format_scalar_list(Z) + ";", file=pv)
    (const_dir / "PVtableProperties").write_text(pv.getvalue())


def maybe_compute_thermo_extras(fields, P, mech):
    if mech is None:
        return {}
    try:
        import cantera as ct
    except Exception:
        print("[WARN] Cantera not available; skipping mu/Cps/alpha from Cantera.")
        return {}

    mech_path = Path(mech).expanduser().resolve()
    if not mech_path.exists():
        print(f"[WARN] Mechanism '{mech}' not found; skipping mu/Cps/alpha from Cantera.")
        return {}

    gas = ct.Solution(str(mech_path))
    gas.transport_model = "mixture-averaged"   # or "multicomponent" 
    sp_names = gas.species_names

    T = np.asarray(fields["T"], dtype=float)
    rho = np.asarray(fields["rho"], dtype=float)
    NZ, NPV = T.shape

    Ystack = np.zeros((NZ, NPV, len(sp_names)), dtype=float)
    found_any = False
    for key in list(fields.keys()):
        if not key.startswith("Y_"):
            continue
        sp = key[2:]
        if sp not in sp_names:
            continue
        idx = sp_names.index(sp)
        Ystack[:, :, idx] = np.asarray(fields[key], dtype=float)
        found_any = True

    if not found_any:
        print("[WARN] No Y_<SPEC> matched mechanism species list; skipping mu/Cps/alpha from Cantera.")
        return {}

    mu = np.zeros_like(T)
    cp = np.zeros_like(T)
    k = np.zeros_like(T)

    for i in range(NZ):
        for j in range(NPV):
            gas.TPY = float(T[i, j]), float(P), Ystack[i, j, :]
            mu[i, j] = float(gas.viscosity)
            cp[i, j] = float(gas.cp_mass)
            k[i, j] = float(gas.thermal_conductivity)

    alpha = k / np.maximum(rho * cp, 1e-300)
    return {"mu": mu, "Cps": cp, "alpha": alpha}


def compute_fallback_mu_Cps_alpha(fields):
    T = np.asarray(fields["T"], dtype=float)
    mu_const = 1.8e-5
    Cps_const = 1100.0
    alpha_const = 2.0e-5
    print("[WARN] Writing FALLBACK mu/Cps/alpha as constants (Cantera unusable).")
    return {
        "mu": np.full_like(T, mu_const, dtype=float),
        "Cps": np.full_like(T, Cps_const, dtype=float),
        "alpha": np.full_like(T, alpha_const, dtype=float),
    }


def resolve_tables_dir(out_case: Path, tablePath: str) -> Path:
    p = Path(tablePath)
    if p.is_absolute():
        return p

    norm = p.as_posix().lstrip("./")
    if norm == "tables":
        return out_case / "constant" / "tables"
    if norm.startswith("constant/"):
        return out_case / norm
    return out_case / norm


def main():
    ap = argparse.ArgumentParser(description="Convert CSV-based FGM tables to OpenFOAM tables (OF7 / FGMFoam).")
    ap.add_argument("--in", dest="inp", required=True, help="Input 02_tables folder OR 02_tables.tar.xz")
    ap.add_argument("--out", dest="out", required=True, help="Output case directory")
    ap.add_argument("--pressure", type=float, default=101325.0, help="Pressure [Pa] for psi=rho/P and Cantera")
    ap.add_argument("--mech", type=str, default=None, help="Cantera mechanism (yaml). Optional.")
    ap.add_argument("--emit-pv-bounds", action="store_true",
                    help="Force PVmin_table=0 and PVmax_table=1 (normalized PV bounds).")
    ap.add_argument("--tablePath", type=str, default="tables",
                    help='tablePath written into (PV)tableProperties. Recommended: "tables" (default).')
    ap.add_argument("--varPV", type=float, nargs="+", default=[0.0, 1.0],
                    help="varPV_param values (>=2). Default: 0 1")
    ap.add_argument("--varZ", type=float, nargs="+", default=[0.0, 1.0],
                    help="varZ_param values (>=2). Default: 0 1")
    ap.add_argument("--rebuild-thermo", action="store_true",
                help="Recompute and overwrite psi/mu/Cps/alpha tables even if files already exist.")

    args = ap.parse_args()

    inpath = Path(args.inp).resolve()
    out_case = Path(args.out).resolve()
    tables_dir = resolve_tables_dir(out_case, args.tablePath)
    tables_dir.mkdir(parents=True, exist_ok=True)

    const_dir = out_case / "constant"
    const_dir.mkdir(parents=True, exist_ok=True)

    mech = args.mech
    if mech is None and Path("gri30.yaml").exists():
        mech = "gri30.yaml"

    data = load_tables(inpath)

    Z = np.asarray(data["axes"]["Z"], dtype=float).reshape(-1)
    PV = np.asarray(data["axes"]["PV"], dtype=float).reshape(-1)
    varPV_param = np.asarray(args.varPV, dtype=float).reshape(-1)
    varZ_param = np.asarray(args.varZ, dtype=float).reshape(-1)
    if len(varPV_param) < 2 or len(varZ_param) < 2:
        raise SystemExit("ERROR: --varPV and --varZ must each provide at least 2 values (or accept defaults 0 1).")

    T = np.asarray(data["fields"]["T"], dtype=float)
    rho = np.asarray(data["fields"]["rho"], dtype=float)
    NZ, NPV = T.shape
    if rho.shape != (NZ, NPV):
        raise SystemExit(f"rho shape {rho.shape} != T shape {(NZ, NPV)}")

    for name, arr in data["fields"].items():
        a = np.asarray(arr)
        if a.ndim != 2 or a.shape != (NZ, NPV):
            raise SystemExit(f"Field '{name}' has shape {a.shape}, expected {(NZ, NPV)}")

    emit_properties(const_dir, Z, PV, varPV_param, varZ_param, tablePath_string=args.tablePath)

    dims = {
        "T_table":        "[0 0 0 1 0 0 0]",
        "rho_table":      "[1 -3 0 0 0 0 0]",
        "Y_table":        "[0 0 0 0 0 0 0]",
        "psi_table":      "[0 -2 2 0 0 0 0]",
        "mu_table":       "[1 -1 -1 0 0 0 0]",
        "Cps_table":      "[0 2 -2 -1 0 0 0]",
        "alpha_table":    "[0 2 -1 0 0 0 0]",
        "SourcePV_table": "[0 0 0 -1 0 0 0]",
        "PV_table":       "[0 0 0 0 0 0 0]",
    }

    # Required integrated tables
    write_integrated_table_4d(tables_dir / "T_table", "T_table", dims["T_table"],
                              Z, PV, varPV_param, varZ_param, data["fields"]["T"])
    write_integrated_table_4d(tables_dir / "rho_table", "rho_table", dims["rho_table"],
                              Z, PV, varPV_param, varZ_param, data["fields"]["rho"])

    for name, arr in data["fields"].items():
        if name.startswith("Y_"):
            write_integrated_table_4d(tables_dir / f"{name}_table", f"{name}_table", dims["Y_table"],
                                      Z, PV, varPV_param, varZ_param, arr)

    # Ensure mu/Cps/alpha exist: prefer Cantera, else constants
    force = bool(args.rebuild_thermo)
    
    # Optional extras from input
    for key in ("psi", "mu", "Cps", "alpha", "SourcePV"):
        if force and key in ("psi", "mu", "Cps", "alpha"):
            continue  # skip importing these; we will rebuild them
        if key in data["extras"]:
            write_integrated_table_4d(tables_dir / f"{key}_table", f"{key}_table",
                                      dims.get(f"{key}_table", dims["Y_table"]),
                                      Z, PV, varPV_param, varZ_param, data["extras"][key])

    need_mu    = force or not (tables_dir / "mu_table").exists()
    need_cps   = force or not (tables_dir / "Cps_table").exists()
    need_alpha = force or not (tables_dir / "alpha_table").exists()

    need_psi = force or not (tables_dir / "psi_table").exists()

    if need_mu or need_cps or need_alpha:
        thermo_extras = maybe_compute_thermo_extras(data["fields"], args.pressure, mech)
        if not thermo_extras:
            thermo_extras = compute_fallback_mu_Cps_alpha(data["fields"])

        if need_mu:
            write_integrated_table_4d(tables_dir / "mu_table", "mu_table", dims["mu_table"],
                                      Z, PV, varPV_param, varZ_param, thermo_extras["mu"])
            print("[INFO] Wrote mu_table.")
        if need_cps:
            write_integrated_table_4d(tables_dir / "Cps_table", "Cps_table", dims["Cps_table"],
                                      Z, PV, varPV_param, varZ_param, thermo_extras["Cps"])
            print("[INFO] Wrote Cps_table.")
        if need_alpha:
            write_integrated_table_4d(tables_dir / "alpha_table", "alpha_table", dims["alpha_table"],
                                      Z, PV, varPV_param, varZ_param, thermo_extras["alpha"])
            print("[INFO] Wrote alpha_table.")

    psi_path = tables_dir / "psi_table"
    if need_psi:
        psi = np.asarray(data["fields"]["rho"], dtype=float) / max(float(args.pressure), 1e-300)
        write_integrated_table_4d(psi_path, "psi_table", dims["psi_table"], Z, PV, varPV_param, varZ_param, psi)
        print("[INFO] Wrote psi_table as rho/P (no Cantera required).")

    # Ensure SourcePV_table exists (this is crucial for PV evolution / heat release)
    spv_path = tables_dir / "SourcePV_table"
    if not spv_path.exists():
        print("[WARN] No SourcePV provided in input tables. Writing SourcePV_table = 0 everywhere.")
        print("[WARN] This means PV will not self-increase (no chemistry progress unless imposed by BCs).")
        src = np.zeros((NZ, NPV), dtype=float)
        write_integrated_table_4d(spv_path, "SourcePV_table", dims["SourcePV_table"],
                                  Z, PV, varPV_param, varZ_param, src)

    # PV bounds tables
    if args.emit_pv_bounds:
        pvmin = np.zeros((len(Z),), dtype=float)
        pvmax = np.ones((len(Z),), dtype=float)
        print("[INFO] Forcing PVmin/PVmax bounds to 0/1 due to --emit-pv-bounds.")
    elif ("PVmin" in data["axes"]) and ("PVmax" in data["axes"]):
        pvmin = np.asarray(data["axes"]["PVmin"], dtype=float).reshape(-1)
        pvmax = np.asarray(data["axes"]["PVmax"], dtype=float).reshape(-1)
        if pvmin.size != len(Z) or pvmax.size != len(Z):
            raise SystemExit(f"axes/PVmin.csv and axes/PVmax.csv must be length NZ={len(Z)} "
                             f"(got {pvmin.size}, {pvmax.size})")
        print("[INFO] Using axes/PVmin.csv and axes/PVmax.csv for PV bounds (physical PV units).")
    else:
        pvmin = np.zeros((len(Z),), dtype=float)
        pvmax = np.ones((len(Z),), dtype=float)
        print("[WARN] No axes/PVmin.csv and axes/PVmax.csv found; falling back to PVmin=0, PVmax=1.")

    write_pv_table_2d(tables_dir / "PVmin_table", "PVmin_table", dims["PV_table"], Z, varZ_param, pvmin)
    write_pv_table_2d(tables_dir / "PVmax_table", "PVmax_table", dims["PV_table"], Z, varZ_param, pvmax)

    print(f"[OK] Wrote OpenFOAM tables to: {tables_dir}")
    print(f"[OK] tableProperties: {const_dir/'tableProperties'}")
    print(f"[OK] PVtableProperties: {const_dir/'PVtableProperties'}")
    print(f"[OK] tablePath written as: {args.tablePath}")
    print("[HINT] FGMFoam expects files like:")
    print(f"       {tables_dir}/T_table")
    print(f"       {tables_dir}/mu_table")
    print(f"       {tables_dir}/alpha_table")


if __name__ == "__main__":
    main()

