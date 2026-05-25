#!/usr/bin/env python3
"""Generate premixed CH4/air FreeFlame solutions for the OpenFOAM-7 FGM table.

This replaces the earlier CounterflowDiffusionFlame generator. The supplied good
OpenFOAM-7 database is a premixed CH4/air FGM database, and the case coordinate Z
is the unburned fuel mass fraction, with main-jet boundary Z=0.1559.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import cantera as ct
import numpy as np
from fgm_common import MixtureConfig, equivalence_ratio_at_z, set_unburned_state, stoichiometric_z, sorted_unique


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mech", default="gri30.yaml")
    p.add_argument("--out-dir", default="1DPremixedFlameletFiles")
    p.add_argument("--fuel", default="CH4:1")
    p.add_argument("--oxidizer", default="O2:0.21, N2:0.79")
    p.add_argument("--tin", type=float, default=294.0)
    p.add_argument("--pressure", type=float, default=101325.0)
    p.add_argument("--z-max", type=float, default=0.1559,
                   help="Maximum table coordinate; the supplied case main inlet is Z=0.1559")
    p.add_argument("--nz", type=int, default=51)
    p.add_argument("--extra-z", nargs="*", type=float, default=[0.04293],
                   help="Additional solved coordinates, e.g. the pilot mixture Z")
    p.add_argument("--width", type=float, default=0.04)
    p.add_argument("--transport", default="mixture-averaged")
    p.add_argument("--phi-min", type=float, default=0.30,
                   help="Skip FreeFlame solve below this equivalence ratio")
    p.add_argument("--phi-max", type=float, default=2.40,
                   help="Skip FreeFlame solve above this equivalence ratio")
    p.add_argument("--lit-delta-T", type=float, default=150.0)
    p.add_argument("--loglevel", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.nz < 3 or not (0 < args.z_max <= 1.0):
        raise SystemExit("Require --nz >= 3 and 0 < --z-max <= 1.")
    cfg = MixtureConfig(args.mech, args.fuel, args.oxidizer, args.tin, args.pressure)
    gas = ct.Solution(args.mech)
    gas.transport_model = args.transport
    zst = stoichiometric_z(gas, cfg)
    z_axis = np.linspace(0.0, args.z_max, args.nz)
    requested = sorted_unique([*z_axis.tolist(), *args.extra_z, zst])
    solve_order = sorted(requested.tolist(), key=lambda z: abs(z - zst))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "premixed_manifest.csv"
    rows: list[dict[str, object]] = []

    print(f"[SETUP] Premixed CH4/air FreeFlame manifold; Cantera={ct.__version__}")
    print(f"[SETUP] Z axis: 0 <= Z <= {args.z_max:.7g}; stoichiometric Z={zst:.8g}")
    print(f"[SETUP] Solving flame profiles for {len(requested)} candidate Z values")

    for iz, z in enumerate(solve_order):
        phi = equivalence_ratio_at_z(gas, z, cfg) if z > 0 else 0.0
        name = f"premixed_Z_{z:.8f}.yaml".replace(".", "p")
        # restore .yaml extension after decimal-safe stem substitution
        name = name.replace("pyaml", ".yaml")
        path = out / name
        row: dict[str, object] = {"Z": z, "phi": phi, "file": path.name, "status": "", "Tmax": "", "flame_speed": ""}
        if z <= 0.0 or phi < args.phi_min or phi > args.phi_max:
            row["status"] = "not_solved_outside_phi_range"
            rows.append(row)
            print(f"[SKIP] Z={z:.8g}, phi={phi:.5g}: outside solve interval")
            continue
        try:
            set_unburned_state(gas, z, cfg)
            f = ct.FreeFlame(gas, width=args.width)
            f.transport_model = args.transport
            f.set_refine_criteria(ratio=3.0, slope=0.10, curve=0.18, prune=0.02)
            f.solve(loglevel=args.loglevel, auto=True)
            Tmax = float(np.max(f.T))
            if Tmax < args.tin + args.lit_delta_T:
                row.update(status="cold_solution_rejected", Tmax=Tmax)
                rows.append(row)
                print(f"[SKIP] Z={z:.8g}, phi={phi:.5g}: cold solution Tmax={Tmax:.2f} K")
                continue
            if path.exists() and not args.overwrite:
                path.unlink()
            f.save(str(path), name="premixed", description=f"Premixed FreeFlame Z={z:.12g}, phi={phi:.12g}", overwrite=True)
            speed = float(f.velocity[0])
            row.update(status="solved", Tmax=Tmax, flame_speed=speed)
            rows.append(row)
            print(f"[OK] Z={z:.8g}, phi={phi:.5g}, Tmax={Tmax:.2f} K, Su={speed:.6g} m/s -> {path.name}")
        except ct.CanteraError as exc:
            row["status"] = "solve_failed"
            rows.append(row)
            print(f"[WARN] Z={z:.8g}, phi={phi:.5g}: FreeFlame solve failed: {exc}")

    rows.sort(key=lambda r: float(r["Z"]))
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Z", "phi", "file", "status", "Tmax", "flame_speed"])
        writer.writeheader(); writer.writerows(rows)
    solved = sum(r["status"] == "solved" for r in rows)
    if solved < 3:
        raise SystemExit(f"Only {solved} premixed flames converged; at least 3 are required. Review mechanism/inlet range.")
    print(f"[DONE] Solved {solved} premixed flamelet(s); manifest: {manifest}")
    print("[NOTE] Unsolved nonflammable/rich endpoints are filled as zero-source thermochemical branches by the table builder.")

if __name__ == "__main__":
    main()
