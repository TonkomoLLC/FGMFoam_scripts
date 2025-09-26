#!/usr/bin/env python3
# build_fgm_tables.py — assemble FGM tables from postprocessed flamelet points
# Inputs:  post_strain_loop_*.csv (from organizeData.py)
# Output:  02_tables.tar.xz with axes/, thermo/, species/, metadata.yaml

import os, glob, tarfile, io, json
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------- user knobs ----------------
POST_GLOB = "post_strain_loop_*.csv"
# choose table resolution (typical 101x101; adjust as you like)
NZ = 101
NPV = 101
# minimum points per bin to accept (else NaN -> nearest fill)
MIN_BIN_COUNT = 6
# -------------------------------------------

def _favre_bin_2d(Z, PV, rho, field, Zgrid, PVgrid, min_count=6):
    """Mass-weighted bin-average field(Z,PV), returning table shape (NZ, NPV)."""
    NZ, NPV = len(Zgrid), len(PVgrid)
    table = np.full((NZ, NPV), np.nan, float)
    counts = np.zeros((NZ, NPV), int)

    # bin edges midpoints
    Zb = (Zgrid[:-1] + Zgrid[1:]) / 2.0
    PVb = (PVgrid[:-1] + PVgrid[1:]) / 2.0

    Zi = np.searchsorted(Zb, Z)
    PVi = np.searchsorted(PVb, PV)

    # clip to table interior
    Zi = np.clip(Zi, 0, NZ-1)
    PVi = np.clip(PVi, 0, NPV-1)

    # accumulate mass-weighted sums
    num = np.zeros((NZ, NPV), float)
    den = np.zeros((NZ, NPV), float)
    for k in range(len(field)):
        i, j = Zi[k], PVi[k]
        w = float(rho[k])
        num[i, j] += w * float(field[k])
        den[i, j] += w
        counts[i, j] += 1

    # compute averages
    mask = (counts >= min_count) & (den > 0.0)
    table[mask] = num[mask] / den[mask]
    return table, counts

def _nearest_fill(table):
    """Simple nearest-neighbor fill for NaNs to keep table complete."""
    nz, npv = table.shape
    filled = table.copy()
    nan_idx = np.argwhere(np.isnan(filled))
    if nan_idx.size == 0:
        return filled
    known = np.argwhere(~np.isnan(filled))
    if known.size == 0:
        return filled
    known_vals = filled[~np.isnan(filled)]
    for (i, j) in nan_idx:
        d2 = (known[:,0]-i)**2 + (known[:,1]-j)**2
        k = np.argmin(d2)
        filled[i, j] = known_vals[k]
    return filled

def main():
    files = sorted(glob.glob(POST_GLOB))
    if not files:
        raise SystemExit(f"No files match {POST_GLOB}")

    # collect all points
    cols_basic = ["Z","PV","rho","T"]
    df_all = []
    species_cols = None
    for fn in files:
        df = pd.read_csv(fn)
        if species_cols is None:
            species_cols = [c for c in df.columns if c.startswith("Y.")]
        take = cols_basic + species_cols
        df_all.append(df[take].copy())
    df = pd.concat(df_all, ignore_index=True)

    # sanitize/normalize
    Z = np.clip(df["Z"].to_numpy(dtype=float), 0.0, 1.0)
    PV_raw = df["PV"].to_numpy(dtype=float)
    # robust PV scaling: map min->0, 95th percentile->1 (limit overshoots)
    pv_lo = float(np.min(PV_raw))
    pv_hi = float(np.percentile(PV_raw, 95.0))
    pv_span = max(1e-30, pv_hi - pv_lo)
    PV = np.clip((PV_raw - pv_lo)/pv_span, 0.0, 1.0)

    rho = np.maximum(1e-12, df["rho"].to_numpy(dtype=float))
    T   = df["T"].to_numpy(dtype=float)

    # grids
    Zgrid  = np.linspace(0.0, 1.0, NZ)
    PVgrid = np.linspace(0.0, 1.0, NPV)

    # make output dirs in memory
    out_root = Path("02_tables")
    axes_dir = out_root/"axes"
    thermo_dir = out_root/"thermo"
    species_dir = out_root/"species"

    # prepare tables
    T_tbl, _ = _favre_bin_2d(Z, PV, rho, T, Zgrid, PVgrid, MIN_BIN_COUNT)
    T_tbl = _nearest_fill(T_tbl)

    rho_tbl, _ = _favre_bin_2d(Z, PV, rho, rho, Zgrid, PVgrid, MIN_BIN_COUNT)
    rho_tbl = _nearest_fill(rho_tbl)

    # species
    species_tables = {}
    for yc in species_cols:
        Yk = np.clip(df[yc].to_numpy(dtype=float), 0.0, 1.0)
        Y_tbl, _ = _favre_bin_2d(Z, PV, rho, Yk, Zgrid, PVgrid, MIN_BIN_COUNT)
        species_tables[yc[2:]] = _nearest_fill(Y_tbl)

    # write into a tar.xz (no temp files needed)
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w:xz") as tf:
        def add_csv(path, array2d):
            data = io.BytesIO()
            np.savetxt(data, array2d, delimiter=",")
            info = tarfile.TarInfo(name=str(path))
            data.seek(0)
            info.size = len(data.getvalue())
            tf.addfile(tarinfo=info, fileobj=data)

        def add_text(path, text):
            data = io.BytesIO(text.encode("utf-8"))
            info = tarfile.TarInfo(name=str(path))
            info.size = len(data.getvalue())
            tf.addfile(tarinfo=info, fileobj=data)

        # axes
        add_csv(axes_dir/"Z.csv", Zgrid.reshape(-1,1))
        add_csv(axes_dir/"PV.csv", PVgrid.reshape(-1,1))

        # thermo
        add_csv(thermo_dir/"T.csv",  T_tbl)
        add_csv(thermo_dir/"rho.csv", rho_tbl)

        # species
        for name, tbl in species_tables.items():
            add_csv(species_dir/f"Y_{name}.csv", tbl)

        # metadata
        meta = {
            "schema": "FGM-2D-Z-PV",
            "shape": [NZ, NPV],
            "axes": {"Z":"axes/Z.csv", "PV":"axes/PV.csv"},
            "thermo": {"T":"thermo/T.csv", "rho":"thermo/rho.csv"},
            "species": {f"Y_{k}": f"species/Y_{k}.csv" for k in species_tables.keys()},
            "notes": [
                "Favre (rho-weighted) averages within (Z,PV) bins.",
                "PV normalized to [0,1] using min..95th percentile of collected points.",
                "NaN bins filled by nearest-neighbor to ensure a complete table."
            ],
            "source": sorted(files),
        }
        add_text(out_root/"metadata.yaml", json.dumps(meta, indent=2))

    # write tarball
    with open("02_tables.tar.xz", "wb") as f:
        f.write(tar_bytes.getvalue())

    print("Wrote 02_tables.tar.xz with axes/, thermo/, species/, metadata.yaml")

if __name__ == "__main__":
    main()
