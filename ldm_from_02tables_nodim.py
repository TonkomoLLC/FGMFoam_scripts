#!/usr/bin/env python3
import io, argparse, tarfile, numpy as np, re, sys
from pathlib import Path

def parse_list(txt, key):
    m = re.search(rf"^\s*{re.escape(key)}\s+(.*?);", txt, flags=re.M|re.S)
    if not m: return []
    inner = re.search(r"\((.*?)\)", m.group(1), flags=re.S)
    if not inner: return []
    toks = inner.group(1).replace("\n"," ").split()
    out = []
    for t in toks:
        try: out.append(float(t))
        except: pass
    return out

BANNER = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield        | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration    | Version:  7                                     |
|   \\\\  /    A nd          | Web:      http://www.OpenFOAM.com               |
|    \\\\/     M anipulation |                                                 |
\\*---------------------------------------------------------------------------*/"""

def open_tables(inpath: Path):
    is_tar = inpath.is_file()
    tf = tarfile.open(inpath, "r:*") if is_tar else None
    def read(rel):
        reln = rel.replace("\\","/")
        if is_tar:
            for m in tf.getmembers():
                if not m.isfile(): continue
                if m.name.replace("\\","/").endswith(reln):
                    return tf.extractfile(m).read().decode("utf-8")
            raise FileNotFoundError(f"Could not find '{reln}' inside tarball: {inpath}")
        else:
            p = inpath / reln
            if p.exists():
                return p.read_text()
            for q in inpath.rglob(Path(reln).name):
                if q.as_posix().endswith(reln):
                    return q.read_text()
            raise FileNotFoundError(f"Could not find '{reln}' under folder: {inpath}")
    return read, (tf if is_tar else None)

def load_02tables(inpath: Path):
    read, tf = open_tables(inpath)
    Z  = np.loadtxt(io.StringIO(read("axes/Z.csv")), delimiter=",").reshape(-1)
    PV = np.loadtxt(io.StringIO(read("axes/PV.csv")), delimiter=",").reshape(-1)
    T  = np.loadtxt(io.StringIO(read("thermo/T.csv")), delimiter=",")
    rho= np.loadtxt(io.StringIO(read("thermo/rho.csv")), delimiter=",")
    species = {}
    if tf:
        names = []
        for m in tf.getmembers():
            n = m.name.replace("\\","/")
            if "/species/" in n and n.endswith(".csv"):
                nm = Path(n).stem
                names.append(nm)
        names = sorted(set(names))
        for nm in names:
            species[nm] = np.loadtxt(io.StringIO(read(f"species/{nm}.csv")), delimiter=",")
        tf.close()
    else:
        spdir = inpath / "species"
        if spdir.exists():
            for p in spdir.glob("Y_*.csv"):
                species[p.stem] = np.loadtxt(p, delimiter=",")
        else:
            for p in inpath.rglob("Y_*.csv"):
                species[p.stem] = np.loadtxt(p, delimiter=",")
    return Z, PV, T, rho, species

def write_FOAM_sized_list(out, count):
    out.write(f"{int(count)}\n(\n")

def write_scalar_list_lines(out, arr):
    for v in arr:
        out.write(f" {v:.8e}\n")
    out.write(")\n")

def write_nested_T(out_path: Path, name: str, Z, PV, varZ, varPV, field2d):
    NZ, NPV = field2d.shape
    with open(out_path, "w") as f:
        f.write(BANNER + "\n\n")
        f.write("FoamFile\n{\n")
        f.write("    version     2.0;\n")
        f.write("    format      ascii;\n")
        f.write("    class       dictionary;\n")
        f.write(f"    object      {name}_table;\n")
        f.write("}\n\n")
        f.write(f"{name}_table\n")
        varPV_list = varPV if (varPV and len(varPV)>0) else [0.0]
        write_FOAM_sized_list(f, len(varPV_list))
        for _vp in varPV_list:
            write_FOAM_sized_list(f, len(PV))
            for j in range(len(PV)):
                varZ_list = varZ if (varZ and len(varZ)>0) else [0.0]
                write_FOAM_sized_list(f, len(varZ_list))
                for _vz in varZ_list:
                    write_FOAM_sized_list(f, len(Z))
                    write_scalar_list_lines(f, field2d[:, j])
                f.write(")\n")
            f.write(")\n")
        f.write(");\n")

def write_nested_PV(out_path: Path, name: str, Z, varZ, rowvec):
    NZ = len(Z)
    varZ_list = varZ if (varZ and len(varZ)>0) else [0.0]
    with open(out_path, "w") as f:
        f.write(BANNER + "\n\n")
        f.write("FoamFile\n{\n")
        f.write("    version     2.0;\n")
        f.write("    format      ascii;\n")
        f.write("    class       dictionary;\n")
        f.write(f"    object      {name}_table;\n")
        f.write("}\n\n")
        f.write(f"{name}_table\n")
        write_FOAM_sized_list(f, len(varZ_list))
        for _ in varZ_list:
            write_FOAM_sized_list(f, NZ)
            write_scalar_list_lines(f, rowvec)
        f.write(");\n")

def main():
    ap = argparse.ArgumentParser(description="Emit nested OpenFOAM tables without 'dimensions' entries.")
    ap.add_argument("--inp", required=True, help="02_tables folder or tar.xz")
    ap.add_argument("--case", required=True, help="OpenFOAM case (has constant/...)")
    ap.add_argument("--tablePath", default="tables", help="Folder (relative to case) where files go, e.g. ./table/ or ./../02_table/")
    ap.add_argument("--emit", nargs="+", default=["T","rho","Y_all","SourcePV","PVmin","PVmax","YWI","YuWI","YbWI","Yu2I","YuYbI","Yb2I"],
                    help="What to emit")
    args = ap.parse_args()

    case = Path(args.case).resolve()
    const = case/"constant"
    tp = (const/"tableProperties").read_text()
    pv = (const/"PVtableProperties").read_text()

    Z_param   = parse_list(tp, "Z_param")
    PV_param  = parse_list(tp, "PV_param")
    varZ_par  = parse_list(tp, "varZ_param") or [0.0]
    varPV_par = parse_list(tp, "varPV_param") or [0.0]
    varZ_PV   = parse_list(pv, "varZ_param") or [0.0]
    Z_PV      = parse_list(pv, "Z_param") or Z_param

    inpath = Path(args.inp).resolve()
    Z, PV, T, rho, species = load_02tables(inpath)

    out_dir = (case/args.tablePath).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    want = set(args.emit)

    # FGM data fields
    if "T" in want:    write_nested_T(out_dir/"T_table", "T", Z, PV, varZ_par, varPV_par, T)
    if "rho" in want:  write_nested_T(out_dir/"rho_table", "rho", Z, PV, varZ_par, varPV_par, rho)
    if "Y_all" in want:
        for nm, arr in species.items():
            write_nested_T(out_dir/f"{nm}_table", nm, Z, PV, varZ_par, varPV_par, arr)
    else:
        for token in list(want):
            if token.startswith("Y_") and token in species:
                write_nested_T(out_dir/f"{token}_table", token, Z, PV, varZ_par, varPV_par, species[token])

    # Zero-filled FGM extras
    zeros2 = np.zeros_like(T)
    for k in ["SourcePV","YWI","YuWI","YbWI"]:
        if k in want:
            write_nested_T(out_dir/f"{k}_table", k, Z, PV, varZ_par, varPV_par, zeros2)

    # PV-family
    if "PVmin" in want:
        write_nested_PV(out_dir/"PVmin_table", "PVmin", Z_PV, varZ_PV, np.zeros(len(Z_PV)))
    if "PVmax" in want:
        write_nested_PV(out_dir/"PVmax_table", "PVmax", Z_PV, varZ_PV, np.ones(len(Z_PV)))

    # Zero-filled PV extras
    for k in ["Yu2I","YuYbI","Yb2I"]:
        if k in want:
            write_nested_PV(out_dir/f"{k}_table", k, Z_PV, varZ_PV, np.zeros(len(Z_PV)))

    print(f"[OK] Wrote tables (no 'dimensions' entries) to: {out_dir}")

if __name__ == "__main__":
    main()
