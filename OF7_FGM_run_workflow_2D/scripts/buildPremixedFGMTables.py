#!/usr/bin/env python3
"""Build an OpenFOAM-7 premixed FGM table archive on axes (Z, scaled PV).

Each converged Cantera FreeFlame supplies the c direction at its premixed inlet Z.
The output Z coordinate is the unburned CH4/air fuel mass fraction used in the case
(Z=0.1559 at the main jet and Z=0.04293 at the pilot), not a normalized pure-fuel
stream coordinate extending to one.
"""
from __future__ import annotations
import argparse
import glob
import io
import json
import tarfile
from pathlib import Path
import cantera as ct
import numpy as np
import pandas as pd
from fgm_common import MixtureConfig, endpoint_state, sorted_unique, stoichiometric_z, thermo_values, validate_progress_species

THERMO_FIELDS = ["T", "rho", "psi", "mu", "Cps", "alpha", "SourcePV"]


def add_csv(tf: tarfile.TarFile, name: str, arr: np.ndarray) -> None:
    buff = io.BytesIO(); np.savetxt(buff, np.asarray(arr), delimiter=",", fmt="%.12e")
    payload = buff.getvalue(); info = tarfile.TarInfo(name); info.size = len(payload); tf.addfile(info, io.BytesIO(payload))


def add_text(tf: tarfile.TarFile, name: str, text: str) -> None:
    payload = text.encode(); info = tarfile.TarInfo(name); info.size = len(payload); tf.addfile(info, io.BytesIO(payload))


def unique_average(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x); x = x[order]; y = y[order]
    ux, inv = np.unique(x, return_inverse=True)
    sums = np.zeros(ux.size); counts = np.zeros(ux.size)
    for i, group in enumerate(inv):
        sums[group] += y[i]; counts[group] += 1.0
    return ux, sums / counts


def interp_curve(craw: np.ndarray, values: np.ndarray, cgrid: np.ndarray, start: float, end: float) -> np.ndarray:
    x = np.concatenate(([0.0], np.clip(craw, 0.0, 1.0), [1.0]))
    y = np.concatenate(([start], values, [end]))
    x, y = unique_average(x, y)
    return np.interp(cgrid, x, y)


def synthetic_branch(gas: ct.Solution, z: float, cgrid: np.ndarray, cfg: MixtureConfig, species: list[str]) -> dict[str, np.ndarray]:
    u = endpoint_state(gas, z, cfg, burned=False); b = endpoint_state(gas, z, cfg, burned=True)
    result = {name: np.zeros(cgrid.size) for name in THERMO_FIELDS + species}
    Yu = np.asarray(u["Y"]); Yb = np.asarray(b["Y"])
    for j, c in enumerate(cgrid):
        gas.TPY = (1.0-c)*float(u["T"]) + c*float(b["T"]), cfg.pressure, (1.0-c)*Yu + c*Yb
        vals = thermo_values(gas)
        for field in THERMO_FIELDS:
            result[field][j] = 0.0 if field == "SourcePV" else float(vals[field])
        for isp, sp in enumerate(species):
            result[sp][j] = gas.Y[gas.species_index(sp)]
    return result


def flame_curve(df: pd.DataFrame, gas: ct.Solution, z: float, cgrid: np.ndarray, cfg: MixtureConfig, species: list[str]) -> dict[str, np.ndarray]:
    u = endpoint_state(gas, z, cfg, burned=False); b = endpoint_state(gas, z, cfg, burned=True)
    pvmin, pvmax = float(u["PV"]), float(b["PV"])
    span = pvmax - pvmin
    if span <= 1e-12:
        return synthetic_branch(gas, z, cgrid, cfg, species)
    c = (df["PV"].to_numpy(float) - pvmin) / span
    curve: dict[str, np.ndarray] = {}
    for field in THERMO_FIELDS:
        start = 0.0 if field == "SourcePV" else float(u[field])
        end = 0.0 if field == "SourcePV" else float(b[field])
        curve[field] = interp_curve(c, df[field].to_numpy(float), cgrid, start, end)
    for sp in species:
        col = f"Y.{sp}"
        curve[sp] = interp_curve(c, df[col].to_numpy(float), cgrid,
                                 float(np.asarray(u["Y"])[gas.species_index(sp)]),
                                 float(np.asarray(b["Y"])[gas.species_index(sp)]))
    return curve


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mech", default="gri30.yaml")
    p.add_argument("--glob", dest="input_glob", default="1DPremixedFlameletFiles/post_premixed_Z_*.csv")
    p.add_argument("--out", default="FGMTableBuild_OF7_premixed/02_tables_of7_premixed.tar.xz")
    p.add_argument("--fuel", default="CH4:1")
    p.add_argument("--oxidizer", default="O2:0.21, N2:0.79")
    p.add_argument("--tin", type=float, default=294.0)
    p.add_argument("--pressure", type=float, default=101325.0)
    p.add_argument("--z-max", type=float, default=0.1559)
    p.add_argument("--nz", type=int, default=51)
    p.add_argument("--nc", type=int, default=51)
    p.add_argument("--report-z", nargs="*", type=float, default=[0.04293])
    args = p.parse_args()
    cfg = MixtureConfig(args.mech, args.fuel, args.oxidizer, args.tin, args.pressure)
    gas = ct.Solution(args.mech); validate_progress_species(gas)
    species = list(gas.species_names)
    files = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not files:
        raise SystemExit(f"No processed premixed flame files match {args.input_glob}")
    data: list[tuple[float, pd.DataFrame]] = []
    for path in files:
        df = pd.read_csv(path)
        required = set(THERMO_FIELDS + ["Z", "PV"] + [f"Y.{s}" for s in species])
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"{path}: missing fields {sorted(missing)[:10]}")
        data.append((float(df.Z.iloc[0]), df))
    data.sort(key=lambda item: item[0])
    solved_z = np.array([item[0] for item in data], dtype=float)
    if solved_z.size < 3:
        raise SystemExit("At least three solved premixed profiles are required.")
    zgrid = np.linspace(0.0, args.z_max, args.nz)
    cgrid = np.linspace(0.0, 1.0, args.nc)
    fields = {field: np.zeros((args.nz, args.nc)) for field in THERMO_FIELDS + species}
    pvmin = np.zeros(args.nz); pvmax = np.zeros(args.nz)
    solved_curves = {z: flame_curve(df, gas, z, cgrid, cfg, species) for z, df in data}
    solved_stack = {field: np.vstack([solved_curves[z][field] for z in solved_z]) for field in fields}
    for iz, z in enumerate(zgrid):
        u = endpoint_state(gas, z, cfg, burned=False); b = endpoint_state(gas, z, cfg, burned=True)
        pvmin[iz], pvmax[iz] = float(u["PV"]), float(b["PV"])
        if solved_z[0] <= z <= solved_z[-1]:
            for field in fields:
                for jc in range(args.nc):
                    fields[field][iz, jc] = np.interp(z, solved_z, solved_stack[field][:, jc])
        else:
            synth = synthetic_branch(gas, z, cgrid, cfg, species)
            for field in fields:
                fields[field][iz, :] = synth[field]
        # Always impose exact unburned/burned end states at c=0 and c=1.
        for field in THERMO_FIELDS:
            fields[field][iz, 0] = 0.0 if field == "SourcePV" else float(u[field])
            fields[field][iz, -1] = 0.0 if field == "SourcePV" else float(b[field])
        for sp in species:
            idx = gas.species_index(sp)
            fields[sp][iz, 0] = np.asarray(u["Y"])[idx]
            fields[sp][iz, -1] = np.asarray(b["Y"])[idx]
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": "OpenFOAM7-premixed-FGM-Z-scaledPV",
        "coordinate_definition": "Z is kg CH4 fuel stream / kg unburned CH4-air mixture; for pure CH4/air this equals unburned Y_CH4",
        "Z_max": args.z_max,
        "Z_stoichiometric": stoichiometric_z(gas, cfg),
        "source": "Cantera FreeFlame premixed profiles; nonflammable extensions are zero-source endpoint blends",
        "variance": "Writer replicates 2D data over variance axes; useProgressVariableVariance false and useMixtureFractionVariance false required",
        "solved_Z": solved_z.tolist(),
    }
    with tarfile.open(output, "w:xz") as tf:
        add_csv(tf, "02_tables/axes/Z.csv", zgrid.reshape(-1, 1)); add_csv(tf, "02_tables/axes/PV.csv", cgrid.reshape(-1, 1))
        for field in THERMO_FIELDS:
            folder = "extras" if field == "SourcePV" else "thermo"
            add_csv(tf, f"02_tables/{folder}/{field}.csv", fields[field])
        for sp in species:
            add_csv(tf, f"02_tables/species/Y_{sp}.csv", fields[sp])
        add_csv(tf, "02_tables/extras/PVmin.csv", pvmin.reshape(1, -1)); add_csv(tf, "02_tables/extras/PVmax.csv", pvmax.reshape(1, -1))
        add_text(tf, "02_tables/metadata.json", json.dumps(meta, indent=2))
    print(f"[OK] Wrote premixed table archive: {output}")
    print(f"[OK] Z_param range: 0 to {args.z_max:.8g}; stoichiometric Z={meta['Z_stoichiometric']:.8g}")
    print(f"[OK] PV range: {pvmin.min():.6e} to {pvmax.max():.6e}; SourcePV max={fields['SourcePV'].max():.6e}")
    for zq in args.report_z:
        iz = int(np.argmin(abs(zgrid-zq))); jc = int(np.argmax(fields["SourcePV"][iz, :]))
        print(f"[REPORT] Near Z={zgrid[iz]:.8g} (requested {zq:.8g}): max SourcePV={fields['SourcePV'][iz,jc]:.6e} at scaledPV={cgrid[jc]:.4f}, T={fields['T'][iz,jc]:.2f} K")

if __name__ == "__main__":
    main()
