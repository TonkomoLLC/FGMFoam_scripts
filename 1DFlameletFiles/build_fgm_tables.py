#!/usr/bin/env python3
# build_fgm_tables.py — assemble FGM tables from postprocessed flamelet points
#
# Inputs:  post_strain_loop_*.csv (from organizeData.py)
# Output:  02_tables.tar.xz containing:
#          axes/Z.csv, axes/PV.csv
#          thermo/T.csv, thermo/rho.csv, thermo/SourcePV.csv   (NEW)
#          species/Y_<name>.csv
#          metadata.yaml (JSON content)
#
# Key robustness fixes (to avoid FGMFoam segfaults in linearInterpolation):
#   1) PAD the Z and PV axes slightly beyond [0,1] so upperBound() never yields ub==0 or ub==N.
#   2) Fill NaN bins, then CONSTANT-EXTRAPOLATE into padded rows/cols for ALL tables.
#   3) Optional: use a higher PV percentile (e.g. 99.5) to reduce clipping-induced flat spots.
#
# NEW:
#   4) Build SourcePV table from a PV-source column if available (e.g. SourcePV, S_n, omegaPV, etc.)
#      and write thermo/SourcePV.csv. If not available, write zeros and warn.

import glob
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- user knobs ----------------
POST_GLOB = "post_strain_loop_*.csv"

# Final table resolution (including padding cells). Typical: 101x101
NZ = 101
NPV = 101

# Minimum points per bin to accept (else NaN -> filled)
MIN_BIN_COUNT = 6

# Padding epsilon for axes.
PAD_EPS = 1.0e-6

# PV normalization: map pv_lo -> 0, pv_hi -> 1, clip to [0,1]
PV_HI_PERCENTILE = 99.5

# Small floor to prevent pathological weights
RHO_FLOOR = 1e-12

# Output tarball name
OUT_TAR = "02_tables.tar.xz"

# Candidate column names for PV source term (progress-variable reaction source)
# We’ll choose the first one found.
SOURCEPV_CANDIDATES = [
    "SourcePV", "sourcePV", "omegaPV", "OmegaPV", "PVdot", "dPVdt",
    "S_n", "Sn", "S_n_PV", "S_PV"
]
# -------------------------------------------


def _build_padded_axis(n_total: int, pad_eps: float) -> np.ndarray:
    """Axis length n_total with 2 padding cells: [-eps, interior 0..1, 1+eps]."""
    if n_total < 3:
        raise ValueError(f"Need axis length >= 3 to include padding; got {n_total}")
    n_int = n_total - 2
    interior = np.linspace(0.0, 1.0, n_int)
    return np.concatenate(([-pad_eps], interior, [1.0 + pad_eps]))


def _favre_bin_2d(Z, PV, rho, field, Zgrid, PVgrid, min_count=6):
    """
    Mass-weighted (Favre) bin-average field(Z,PV).
    Returns table shape (len(Zgrid), len(PVgrid)).
    """
    NZg, NPVg = len(Zgrid), len(PVgrid)
    table = np.full((NZg, NPVg), np.nan, float)

    Zb = (Zgrid[:-1] + Zgrid[1:]) / 2.0
    PVb = (PVgrid[:-1] + PVgrid[1:]) / 2.0

    Zi = np.searchsorted(Zb, Z, side="left")
    PVi = np.searchsorted(PVb, PV, side="left")

    Zi = np.clip(Zi, 0, NZg - 1)
    PVi = np.clip(PVi, 0, NPVg - 1)

    flat_idx = Zi * NPVg + PVi
    w = rho.astype(float)
    f = field.astype(float)

    den = np.bincount(flat_idx, weights=w, minlength=NZg * NPVg).reshape(NZg, NPVg)
    num = np.bincount(flat_idx, weights=w * f, minlength=NZg * NPVg).reshape(NZg, NPVg)
    counts = np.bincount(flat_idx, minlength=NZg * NPVg).reshape(NZg, NPVg)

    mask = (counts >= int(min_count)) & (den > 0.0)
    table[mask] = num[mask] / den[mask]
    return table, counts


def _nearest_fill(table: np.ndarray) -> np.ndarray:
    """Nearest-neighbor fill for NaNs (index-space)."""
    filled = table.copy()
    nan_idx = np.argwhere(np.isnan(filled))
    if nan_idx.size == 0:
        return filled

    known = np.argwhere(~np.isnan(filled))
    if known.size == 0:
        return filled

    known_vals = filled[~np.isnan(filled)]
    for (i, j) in nan_idx:
        d2 = (known[:, 0] - i) ** 2 + (known[:, 1] - j) ** 2
        k = int(np.argmin(d2))
        filled[i, j] = float(known_vals[k])
    return filled


def _pad_extrapolate_const(tbl: np.ndarray) -> np.ndarray:
    """Constant-extrapolate into padded boundary rows/cols."""
    out = tbl.copy()
    out[0, :] = out[1, :]
    out[-1, :] = out[-2, :]
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out


def _add_csv(tf: tarfile.TarFile, path: Path, array: np.ndarray):
    bio = io.BytesIO()
    np.savetxt(bio, array, delimiter=",", fmt="%.16e")
    data = bio.getvalue()
    info = tarfile.TarInfo(name=str(path))
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _add_text(tf: tarfile.TarFile, path: Path, text: str):
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name=str(path))
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _pick_sourcepv_column(df: pd.DataFrame):
    for c in SOURCEPV_CANDIDATES:
        if c in df.columns:
            return c
    return None


def main():
    files = sorted(glob.glob(POST_GLOB))
    if not files:
        raise SystemExit(f"No files match {POST_GLOB}")

    # -----------------------------
    # PASS 1: determine consistent columns across ALL files
    # -----------------------------
    required = {"Z", "PV", "rho", "T"}

    first_species_order = None
    species_intersection = None

    # For SourcePV: find a candidate column that exists in ALL files
    sourcepv_intersection = None

    for fn in files:
        dfi = pd.read_csv(fn)

        missing = [c for c in required if c not in dfi.columns]
        if missing:
            raise SystemExit(f"{fn}: missing required columns: {missing}")

        # species columns: use intersection but preserve the order from the first file
        sp_cols_here = [c for c in dfi.columns if c.startswith("Y.")]
        if not sp_cols_here:
            raise SystemExit(
                f"{fn}: no species columns found with prefix 'Y.' "
                f"(expected columns like Y.O2, Y.N2, ...)"
            )

        if first_species_order is None:
            first_species_order = sp_cols_here[:]
            species_intersection = set(sp_cols_here)
        else:
            species_intersection &= set(sp_cols_here)

        # SourcePV candidates present in this file
        cand_here = {c for c in SOURCEPV_CANDIDATES if c in dfi.columns}
        if sourcepv_intersection is None:
            sourcepv_intersection = cand_here
        else:
            sourcepv_intersection &= cand_here

    # finalize species columns (ordered by first file)
    species_cols = [c for c in first_species_order if c in species_intersection]
    if not species_cols:
        raise SystemExit("No common species columns across all files after intersection.")

    # finalize SourcePV column (first match in priority list that exists in ALL files)
    sourcepv_col = None
    if sourcepv_intersection:
        for c in SOURCEPV_CANDIDATES:
            if c in sourcepv_intersection:
                sourcepv_col = c
                break

    have_sourcepv = sourcepv_col is not None

    # -----------------------------
    # PASS 2: read only needed columns and concatenate
    # -----------------------------
    df_all = []
    base_cols = ["Z", "PV", "rho", "T"]
    for fn in files:
        use_cols = base_cols + species_cols
        if have_sourcepv:
            use_cols = use_cols + [sourcepv_col]
        dfi = pd.read_csv(fn, usecols=use_cols)
        df_all.append(dfi)

    df = pd.concat(df_all, ignore_index=True)

    # -----------------------------
    # sanitize / normalize inputs
    # -----------------------------
    Z = np.clip(df["Z"].to_numpy(dtype=float), 0.0, 1.0)

    PV_raw = df["PV"].to_numpy(dtype=float)
    pv_lo = float(np.nanmin(PV_raw))
    pv_hi = float(np.nanpercentile(PV_raw, PV_HI_PERCENTILE))
    pv_span = pv_hi - pv_lo
    if not np.isfinite(pv_span) or pv_span <= 0.0:
        PV = np.zeros_like(PV_raw, dtype=float)
    else:
        PV = np.clip((PV_raw - pv_lo) / max(1e-30, pv_span), 0.0, 1.0)

    rho = df["rho"].to_numpy(dtype=float)
    rho = np.where(np.isfinite(rho), rho, RHO_FLOOR)
    rho = np.maximum(RHO_FLOOR, rho)

    T = df["T"].to_numpy(dtype=float)

    # optional SourcePV
    if have_sourcepv:
        SourcePV = df[sourcepv_col].to_numpy(dtype=float)
        SourcePV = np.where(np.isfinite(SourcePV), SourcePV, 0.0)
    else:
        SourcePV = None

    # -----------------------------
    # grids (padded)
    # -----------------------------
    Zgrid = _build_padded_axis(NZ, PAD_EPS)
    PVgrid = _build_padded_axis(NPV, PAD_EPS)

    # PV bounds vectors aligned with Zgrid (physical PV units)
    PVmin_vec = np.full((len(Zgrid),), pv_lo, dtype=float)
    PVmax_vec = np.full((len(Zgrid),), pv_hi, dtype=float)

    # -----------------------------
    # build thermo tables
    # -----------------------------
    T_tbl, _ = _favre_bin_2d(Z, PV, rho, T, Zgrid, PVgrid, MIN_BIN_COUNT)
    T_tbl = _pad_extrapolate_const(_nearest_fill(T_tbl))

    rho_tbl, _ = _favre_bin_2d(Z, PV, rho, rho, Zgrid, PVgrid, MIN_BIN_COUNT)
    rho_tbl = _pad_extrapolate_const(_nearest_fill(rho_tbl))

    if have_sourcepv:
        spv_tbl, _ = _favre_bin_2d(Z, PV, rho, SourcePV, Zgrid, PVgrid, MIN_BIN_COUNT)
        spv_tbl = _pad_extrapolate_const(_nearest_fill(spv_tbl))
        print(f"[INFO] Using PV source column '{sourcepv_col}' to build thermo/SourcePV.csv")
    else:
        spv_tbl = np.zeros((len(Zgrid), len(PVgrid)), dtype=float)
        print("[WARN] No PV source column found in ALL files (tried: " + ", ".join(SOURCEPV_CANDIDATES) + ").")
        print("[WARN] Writing thermo/SourcePV.csv as ZEROS.")

    # -----------------------------
    # species tables
    # -----------------------------
    species_tables = {}
    for yc in species_cols:
        sp_name = yc[2:]  # drop "Y."
        Yk = df[yc].to_numpy(dtype=float)
        Yk = np.where(np.isfinite(Yk), Yk, 0.0)
        Yk = np.clip(Yk, 0.0, 1.0)

        Y_tbl, _ = _favre_bin_2d(Z, PV, rho, Yk, Zgrid, PVgrid, MIN_BIN_COUNT)
        Y_tbl = _pad_extrapolate_const(_nearest_fill(Y_tbl))
        species_tables[sp_name] = Y_tbl

    # -----------------------------
    # metadata
    # -----------------------------
    out_root = Path("02_tables")
    axes_dir = out_root / "axes"
    thermo_dir = out_root / "thermo"
    species_dir = out_root / "species"

    meta = {
        "schema": "FGM-2D-Z-PV",
        "shape": [int(len(Zgrid)), int(len(PVgrid))],
        "axes": {
            "Z": "axes/Z.csv",
            "PV": "axes/PV.csv",
            "PVmin": "axes/PVmin.csv",
            "PVmax": "axes/PVmax.csv",
        },
        "thermo": {
            "T": "thermo/T.csv",
            "rho": "thermo/rho.csv",
            "SourcePV": "thermo/SourcePV.csv",
        },
        "species": {f"Y_{k}": f"species/Y_{k}.csv" for k in sorted(species_tables.keys())},
        "build": {
            "MIN_BIN_COUNT": int(MIN_BIN_COUNT),
            "PAD_EPS": float(PAD_EPS),
            "PV_HI_PERCENTILE": float(PV_HI_PERCENTILE),
            "rho_floor": float(RHO_FLOOR),
            "pv_raw_min": float(pv_lo),
            "pv_raw_hi_percentile": float(pv_hi),
            "sourcepv_col": sourcepv_col if have_sourcepv else None,
        },
        "source": files,
    }

    # -----------------------------
    # write tarball
    # -----------------------------
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w:xz") as tf:
        _add_csv(tf, axes_dir / "Z.csv", Zgrid.reshape(-1, 1))
        _add_csv(tf, axes_dir / "PV.csv", PVgrid.reshape(-1, 1))
        _add_csv(tf, axes_dir / "PVmin.csv", PVmin_vec.reshape(-1, 1))
        _add_csv(tf, axes_dir / "PVmax.csv", PVmax_vec.reshape(-1, 1))

        _add_csv(tf, thermo_dir / "T.csv", T_tbl)
        _add_csv(tf, thermo_dir / "rho.csv", rho_tbl)
        _add_csv(tf, thermo_dir / "SourcePV.csv", spv_tbl)

        for sp_name in sorted(species_tables.keys()):
            _add_csv(tf, species_dir / f"Y_{sp_name}.csv", species_tables[sp_name])

        _add_text(tf, out_root / "metadata.yaml", json.dumps(meta, indent=2))

    with open(OUT_TAR, "wb") as f:
        f.write(tar_bytes.getvalue())

    print(f"Wrote {OUT_TAR}")
    print(f"  axes:     NZ={len(Zgrid)}  NPV={len(PVgrid)}  (padded eps={PAD_EPS:g})")
    print(f"  PV norm:  pv_lo={pv_lo:.6e}, pv_hi(p{PV_HI_PERCENTILE:g})={pv_hi:.6e}")
    print(f"  SourcePV: {'from ' + sourcepv_col if have_sourcepv else 'ZEROS'}")
    print(f"  species:  {len(species_tables)} tables")



if __name__ == "__main__":
    main()

