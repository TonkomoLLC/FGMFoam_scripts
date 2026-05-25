#!/usr/bin/env python3
"""Extract premixed FreeFlame profiles and compute OpenFOAM-7 FGM quantities."""
from __future__ import annotations
import argparse
import glob
from pathlib import Path
import cantera as ct
import numpy as np
import pandas as pd
from fgm_common import MixtureConfig, progress_variable, source_progress_variable, thermo_values, validate_progress_species


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mech", default="gri30.yaml")
    p.add_argument("--manifest", default="1DPremixedFlameletFiles/premixed_manifest.csv")
    p.add_argument("--glob", dest="input_glob", default="1DPremixedFlameletFiles/premixed_Z_*.yaml")
    p.add_argument("--group", default="premixed")
    p.add_argument("--width", type=float, default=0.04)
    p.add_argument("--out-prefix", default="post_")
    args = p.parse_args()
    gas = ct.Solution(args.mech)
    validate_progress_species(gas)
    manifest = pd.read_csv(args.manifest)
    solved = manifest[manifest.status == "solved"].copy()
    z_by_file = {str(row.file): float(row.Z) for row in solved.itertuples()}
    files = [Path(path) for path in sorted(glob.glob(args.input_glob)) if Path(path).name in z_by_file]
    if not files:
        raise SystemExit(f"No solved premixed flame YAML files match {args.input_glob}")
    for path in files:
        target_z = z_by_file[path.name]
        flame = ct.FreeFlame(gas, width=args.width)
        flame.restore(str(path), name=args.group)
        states = flame.to_array()
        Y = np.asarray(states.Y, dtype=float)
        T = np.asarray(states.T, dtype=float)
        P = np.full_like(T, float(flame.P))
        grid = np.asarray(flame.grid, dtype=float)
        out: dict[str, np.ndarray] = {"x": grid, "Z": np.full_like(T, target_z), "T": T}
        vals = {key: np.empty_like(T) for key in ("rho", "PV", "SourcePV", "psi", "mu", "Cps", "alpha")}
        for i in range(T.size):
            gas.TPY = T[i], P[i], Y[i, :]
            local = thermo_values(gas)
            for key in vals:
                vals[key][i] = float(local[key])
        out.update(vals)
        for j, sp in enumerate(gas.species_names):
            out[f"Y.{sp}"] = Y[:, j]
        result = pd.DataFrame(out)
        outfile = path.with_name(args.out_prefix + path.stem + ".csv")
        result.to_csv(outfile, index=False)
        print(f"[OK] {outfile.name}: Z={target_z:.8g}, points={len(result)}, T=[{T.min():.2f},{T.max():.2f}] K, PV=[{result.PV.min():.6e},{result.PV.max():.6e}], max SourcePV={result.SourcePV.max():.6e}")
    print(f"[DONE] Processed {len(files)} premixed flamelet file(s).")

if __name__ == "__main__":
    main()
