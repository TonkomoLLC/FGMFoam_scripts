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

# PV orientation / PV-T diagnostics quantiles
PV_Q_LO = 0.20
PV_Q_HI = 0.80

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

    # Bin boundaries (midpoints) in axis-space
    Zb = (Zgrid[:-1] + Zgrid[1:]) / 2.0
    PVb = (PVgrid[:-1] + PVgrid[1:]) / 2.0

    Zi = np.searchsorted(Zb, Z, side="left")
    PVi = np.searchsorted(PVb, PV, side="left")

    Zi = np.clip(Zi, 0, NZg - 1)
    PVi = np.clip(PVi, 0, NPVg - 1)

    flat_idx = Zi * NPVg + PVi

    w = rho.astype(float)
    f = field.astype(float)

    # IMPORTANT: filter non-finite values BEFORE bincount (prevents NaN poisoning)
    good = np.isfinite(flat_idx) & np.isfinite(w) & np.isfinite(f) & np.isfinite(Z) & np.isfinite(PV)
    flat_idx = flat_idx[good].astype(np.int64, copy=False)
    w = w[good]
    f = f[good]

    den = np.bincount(flat_idx, weights=w, minlength=NZg * NPVg).reshape(NZg, NPVg)
    num = np.bincount(flat_idx, weights=w * f, minlength=NZg * NPVg).reshape(NZg, NPVg)
    counts = np.bincount(flat_idx, minlength=NZg * NPVg).reshape(NZg, NPVg)

    mask = (counts >= int(min_count)) & (den > 0.0)
    table[mask] = num[mask] / den[mask]
    return table, counts

def _nearest_fill(table: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbor fill for NaNs in a 2D array, using Euclidean distance
    in (i,j) index-space.

    NOTE: O(N_nan * N_known). For 101x101 tables this is fine.
    """
    filled = table.copy()

    nan_ij = np.argwhere(~np.isfinite(filled))
    if nan_ij.size == 0:
        return filled

    known_ij = np.argwhere(np.isfinite(filled))
    if known_ij.size == 0:
        # nothing to fill from
        return filled

    # Keep values aligned with known_ij ordering
    known_vals = filled[np.isfinite(filled)]

    for (i, j) in nan_ij:
        d2 = (known_ij[:, 0] - i) ** 2 + (known_ij[:, 1] - j) ** 2
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


def _copy_sparse_rows_from_nearest_good(
    tbl: np.ndarray,
    counts: np.ndarray,
    min_bins_nonempty: int = 25,
    min_count_per_bin: int = 6,
    protect_rows: set[int] | None = None,
):
    """
    Copy entire sparse rows from nearest "good" donor row.

    - "Supported" bin: finite(tbl) AND counts >= min_count_per_bin
    - Donor rows: have >= min_bins_nonempty supported bins AND are NOT padding-adjacent
      (exclude rows 0,1 and -2,-1).
    - protect_rows: rows that should never be overwritten (optional).
    """
    out = tbl.copy()
    if counts.shape != out.shape:
        raise ValueError("counts and tbl must have same shape")
    protect_rows = protect_rows or set()

    supported = np.isfinite(out) & (counts >= int(min_count_per_bin))
    n_supported = supported.sum(axis=1)

    donor = np.where(n_supported >= int(min_bins_nonempty))[0]
    donor = donor[(donor >= 2) & (donor <= out.shape[0] - 3)]
    donor = np.array([r for r in donor if r not in protect_rows], dtype=int)

    if donor.size == 0:
        return out

    dmin = int(donor.min())
    dmax = int(donor.max())

    bad = np.where(n_supported < int(min_bins_nonempty))[0]
    for r in bad:
        if r in protect_rows:
            continue
        if r <= dmin:
            g = dmin
        elif r >= dmax:
            g = dmax
        else:
            g = donor[np.argmin(np.abs(donor - r))]
        out[r, :] = out[g, :]

    return out

def _enforce_Z_boundary_rows_from_data(
    Z: np.ndarray,
    PV: np.ndarray,
    T: np.ndarray,
    Zgrid: np.ndarray,
    PVgrid: np.ndarray,
    T_tbl: np.ndarray,
    counts_T: np.ndarray,
    z_frac: float = 0.02,
    pv_tail: float = 0.05,
    min_samples: int = 200,
    fallback_T: float = 300.0,
):
    """
    Fix the exact pathology you are seeing: Z≈0 / Z≈1 rows get filled with interior hot values.

    We compute boundary temperatures directly from the original cloud:
      - low-Z region: Z <= Z_lo + z_frac*(Z_hi-Z_lo)
      - high-Z region: Z >= Z_hi - z_frac*(Z_hi-Z_lo)

    For each boundary side, we compute a representative T_lowPV and T_highPV using PV tails.
    Then we overwrite a small band of Z rows near each boundary with a smooth interpolation
    between those two values along PVgrid.

    This prevents nonsense like T(Z≈0,PV≈0)=2000 K.
    """
    Zlo = float(Zgrid[1])
    Zhi = float(Zgrid[-2])
    z_band = float(z_frac) * max(1e-30, (Zhi - Zlo))

    low_mask = np.isfinite(Z) & np.isfinite(T) & np.isfinite(PV) & (Z <= (Zlo + z_band))
    hi_mask  = np.isfinite(Z) & np.isfinite(T) & np.isfinite(PV) & (Z >= (Zhi - z_band))

    def _estimate_T_tail(mask):
        if np.count_nonzero(mask) < min_samples:
            return None
        m0 = mask & (PV <= pv_tail)
        m1 = mask & (PV >= 1.0 - pv_tail)
        if np.count_nonzero(m0) < (min_samples // 10) or np.count_nonzero(m1) < (min_samples // 10):
            return None
        t0 = float(np.nanmean(T[m0]))
        t1 = float(np.nanmean(T[m1]))
        if not (np.isfinite(t0) and np.isfinite(t1)):
            return None
        return (t0, t1)

    low_tail = _estimate_T_tail(low_mask)
    hi_tail  = _estimate_T_tail(hi_mask)

    # PV interior indices (exclude padded PV columns)
    pv_int = PVgrid[1:-1]
    pv_int = np.clip(pv_int, 0.0, 1.0)

    def _pv_profile(t0, t1):
        # linear profile along PV (you can replace with tanh if desired)
        return (t0 + (t1 - t0) * pv_int)

    # Determine which Z rows are "boundary bands" (include padding-adjacent row 1 / -2 too)
    Zint = Zgrid[1:-1]
    low_rows = np.where(Zint <= (Zlo + z_band))[0] + 1  # shift to table row indices
    hi_rows  = np.where(Zint >= (Zhi - z_band))[0] + 1

    # If we cannot estimate from data, set to fallback (usually inlet temperature)
    if low_tail is None:
        low_t0 = low_t1 = float(fallback_T)
    else:
        low_t0, low_t1 = low_tail

    if hi_tail is None:
        hi_t0 = hi_t1 = float(fallback_T)
    else:
        hi_t0, hi_t1 = hi_tail

    # Overwrite low-Z band
    prof_low = _pv_profile(low_t0, low_t1)
    for r in low_rows:
        # overwrite interior PV columns; keep padded columns consistent later via pad_extrapolate_const
        T_tbl[r, 1:-1] = prof_low

    # Overwrite high-Z band
    prof_hi = _pv_profile(hi_t0, hi_t1)
    for r in hi_rows:
        T_tbl[r, 1:-1] = prof_hi

    # Optional: if boundary bands are extremely sparse, also force counts sanity (not required)
    # (We leave counts_T unchanged since it is diagnostic; table values are now physically sane.)

    return T_tbl

def _robust_pv_autoflip_quantile(
    Z: np.ndarray,
    PV: np.ndarray,
    T: np.ndarray,
    PV_raw: np.ndarray,
    Zgrid: np.ndarray,
    have_sourcepv: bool,
    SourcePV: np.ndarray | None,
    min_rows: int = 10,
    min_samples_row: int = 60,
    q_lo: float = 0.20,
    q_hi: float = 0.80,
    pv_raw_span_min: float = 1e-12,
    flip_if_frac_negative_gt: float = 0.5,
    exclude_z_edge_rows: int = 2,
):
    """
    Decide PV orientation robustly using per-Z-row *quantiles* instead of PV tails.

    For each Z-row:
      dT_row = mean(T | PV <= quantile(PV,q_lo)) -vs- mean(T | PV >= quantile(PV,q_hi))
      (implemented as mean(high) - mean(low))

    If most rows have dT_row < 0, flip PV -> 1-PV.
    """
    # Bin samples into Z rows
    Zb = 0.5 * (Zgrid[:-1] + Zgrid[1:])
    Zi = np.searchsorted(Zb, Z, side="left")
    Zi = np.clip(Zi, 0, len(Zgrid) - 1)

    dT_rows = []

    z_start = int(max(1, exclude_z_edge_rows))
    z_stop  = int(min(len(Zgrid) - 1, len(Zgrid) - exclude_z_edge_rows))

    for zi in range(z_start, z_stop):
        m = (Zi == zi) & np.isfinite(T) & np.isfinite(PV)
        if np.count_nonzero(m) < int(min_samples_row):
            continue

        mraw = (Zi == zi) & np.isfinite(PV_raw)
        if np.count_nonzero(mraw) < int(min_samples_row):
            continue
        if (np.nanmax(PV_raw[mraw]) - np.nanmin(PV_raw[mraw])) < float(pv_raw_span_min):
            continue

        pv_row = PV[m]
        t_row  = T[m]

        # Quantile thresholds inside this Z-row
        lo_thr = float(np.nanquantile(pv_row, q_lo))
        hi_thr = float(np.nanquantile(pv_row, q_hi))
        if not (np.isfinite(lo_thr) and np.isfinite(hi_thr)) or (hi_thr <= lo_thr):
            continue

        m0 = m & (PV <= lo_thr)
        m1 = m & (PV >= hi_thr)

        # Require some samples in each quantile group
        if np.count_nonzero(m0) < max(10, int(0.1 * min_samples_row)):
            continue
        if np.count_nonzero(m1) < max(10, int(0.1 * min_samples_row)):
            continue

        t0 = float(np.nanmean(T[m0]))
        t1 = float(np.nanmean(T[m1]))
        if not (np.isfinite(t0) and np.isfinite(t1)):
            continue

        dT_rows.append(t1 - t0)

    flipped = False
    frac_negative = np.nan

    if len(dT_rows) >= int(min_rows):
        frac_negative = float(np.mean(np.array(dT_rows) < 0.0))
        print(f"[DIAG] PV orientation check (quantile): frac(dT<0) over Z-rows = {frac_negative:.2f}")

        if frac_negative > float(flip_if_frac_negative_gt):
            print("[WARN] PV appears reversed for most Z slices; flipping PV axis.")
            PV = 1.0 - PV
            PV = np.clip(PV, 0.0, 1.0)

            if have_sourcepv and (SourcePV is not None):
                SourcePV = -SourcePV

            flipped = True
    else:
        print(f"[WARN] PV orientation check (quantile): only {len(dT_rows)} usable Z-rows; not flipping.")

    return PV, SourcePV, flipped, dT_rows, frac_negative


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


def _robust_percentile(x: np.ndarray, q: float):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    return float(np.percentile(x, q))


def _compute_pv_bounds_per_Z(
    Z: np.ndarray,
    PV_raw: np.ndarray,
    Zgrid: np.ndarray,
    pv_lo_q: float = 0.0,
    pv_hi_q: float = 99.5,
    min_per_bin: int = 20,
    span_floor: float = 1e-30,
):
    """
    Compute PVmin(Z) and PVmax(Z) vectors aligned with Zgrid (including padded rows).

    Strategy:
      - Use interior Z bins (exclude padding endpoints).
      - For each Z bin, compute PV percentiles [pv_lo_q, pv_hi_q] from samples in that bin.
      - Fill missing bins by nearest-in-index fill.
      - Enforce pvmax > pvmin everywhere (span_floor).
      - Constant-extrapolate into padded Z rows (Zgrid[0], Zgrid[-1]).
    """
    NZg = len(Zgrid)
    if NZg < 3:
        raise ValueError("Zgrid must include padding (len>=3).")

    # interior bin edges: use the interior axis (Zgrid[1:-1]) which is [0..1]
    Z_int = Zgrid[1:-1]
    n_int = len(Z_int)

    # define Z-bin edges halfway between interior points
    Z_edges = np.empty(n_int + 1, dtype=float)
    Z_edges[1:-1] = 0.5 * (Z_int[:-1] + Z_int[1:])
    Z_edges[0] = -np.inf
    Z_edges[-1] = np.inf

    pvmin_int = np.full(n_int, np.nan, dtype=float)
    pvmax_int = np.full(n_int, np.nan, dtype=float)

    # bin index for each sample into interior bins
    Zi = np.searchsorted(Z_edges, Z, side="right") - 1
    Zi = np.clip(Zi, 0, n_int - 1)

    for i in range(n_int):
        mask = (Zi == i) & np.isfinite(PV_raw)
        if np.count_nonzero(mask) < int(min_per_bin):
            continue
        pvl = _robust_percentile(PV_raw[mask], pv_lo_q)
        pvh = _robust_percentile(PV_raw[mask], pv_hi_q)
        pvmin_int[i] = pvl
        pvmax_int[i] = pvh

    # Fill missing bins (nearest in index space)
    def _nearest_1d_fill(v):
        out = v.copy()
        nan = ~np.isfinite(out)
        if not np.any(nan):
            return out
        known_idx = np.where(np.isfinite(out))[0]
        if known_idx.size == 0:
            return out
        for j in np.where(nan)[0]:
            k = known_idx[np.argmin((known_idx - j) ** 2)]
            out[j] = out[k]
        return out

    pvmin_int = _nearest_1d_fill(pvmin_int)
    pvmax_int = _nearest_1d_fill(pvmax_int)

    # If still all-NaN (e.g. PV_raw all NaN), fall back safely
    if not np.isfinite(pvmin_int).any() or not np.isfinite(pvmax_int).any():
        pv_global = PV_raw[np.isfinite(PV_raw)]
        if pv_global.size == 0:
            pv_lo = 0.0
            pv_hi = 1.0
        else:
            pv_lo = float(np.min(pv_global))
            pv_hi = float(np.percentile(pv_global, pv_hi_q))
        pvmin_int[:] = pv_lo
        pvmax_int[:] = pv_hi

    # Enforce strictly positive span everywhere
    span = pvmax_int - pvmin_int
    bad = (~np.isfinite(span)) | (span <= span_floor)
    if np.any(bad):
        # widen around pvmin_int
        pvmax_int[bad] = pvmin_int[bad] + span_floor

    # Build full vectors aligned with Zgrid (including padding)
    pvmin_vec = np.empty(NZg, dtype=float)
    pvmax_vec = np.empty(NZg, dtype=float)
    pvmin_vec[1:-1] = pvmin_int
    pvmax_vec[1:-1] = pvmax_int

    # constant-extrapolate into padded rows
    pvmin_vec[0] = pvmin_vec[1]
    pvmin_vec[-1] = pvmin_vec[-2]
    pvmax_vec[0] = pvmax_vec[1]
    pvmax_vec[-1] = pvmax_vec[-2]

    # Final safety
    pvmax_vec = np.maximum(pvmax_vec, pvmin_vec + span_floor)
    return pvmin_vec, pvmax_vec

def _diag_pvT_quantiles_byZ(
    Z: np.ndarray,
    PV: np.ndarray,
    T: np.ndarray,
    PV_raw: np.ndarray,
    Zgrid: np.ndarray,
    q_lo: float = 0.2,
    q_hi: float = 0.8,
    min_samples_row: int = 60,
    pv_raw_span_min: float = 1e-12,
    exclude_z_edge_rows: int = 2,
):
    # Bin samples into Z rows
    Zb = 0.5 * (Zgrid[:-1] + Zgrid[1:])
    Zi = np.searchsorted(Zb, Z, side="left")
    Zi = np.clip(Zi, 0, len(Zgrid) - 1)

    z_start = int(max(1, exclude_z_edge_rows))
    z_stop  = int(min(len(Zgrid) - 1, len(Zgrid) - exclude_z_edge_rows))

    corrs = []
    tlo_list = []
    thi_list = []
    used = 0

    for zi in range(z_start, z_stop):
        m = (Zi == zi) & np.isfinite(T) & np.isfinite(PV) & np.isfinite(PV_raw)
        if np.count_nonzero(m) < int(min_samples_row):
            continue

        # Require real PV_raw span in this row
        pvraw = PV_raw[m]
        if (np.nanmax(pvraw) - np.nanmin(pvraw)) < float(pv_raw_span_min):
            continue

        pv = PV[m]
        tt = T[m]

        lo_thr = float(np.nanquantile(pv, q_lo))
        hi_thr = float(np.nanquantile(pv, q_hi))
        if not (np.isfinite(lo_thr) and np.isfinite(hi_thr)) or (hi_thr <= lo_thr):
            continue

        m0 = pv <= lo_thr
        m1 = pv >= hi_thr
        if np.count_nonzero(m0) < 10 or np.count_nonzero(m1) < 10:
            continue

        tlo = float(np.nanmean(tt[m0]))
        thi = float(np.nanmean(tt[m1]))
        if not (np.isfinite(tlo) and np.isfinite(thi)):
            continue

        # Correlation only within this Z-row
        corr = float(np.corrcoef(pv, tt)[0, 1])
        if np.isfinite(corr):
            corrs.append(corr)
            tlo_list.append(tlo)
            thi_list.append(thi)
            used += 1

    if used == 0:
        print("[DIAG] PV-T byZ quantile diag: no usable Z-rows")
        return np.nan, np.nan, np.nan, 0

    # Robust summary (median is better than mean here)
    corr_med = float(np.median(corrs))
    tlo_med  = float(np.median(tlo_list))
    thi_med  = float(np.median(thi_list))

    print(f"[DIAG] PV-T byZ (quantile): median corr={corr_med:+.4f} | "
          f"median mean(T|PV<=q{int(q_lo*100)})={tlo_med:.2f}  "
          f"median mean(T|PV>=q{int(q_hi*100)})={thi_med:.2f}  "
          f"(rows used={used})")

    return corr_med, tlo_med, thi_med, used

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
    sourcepv_intersection = None  # candidates present in ALL files

    for fn in files:
        dfi = pd.read_csv(fn)

        missing = [c for c in required if c not in dfi.columns]
        if missing:
            raise SystemExit(f"{fn}: missing required columns: {missing}")

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

        cand_here = {c for c in SOURCEPV_CANDIDATES if c in dfi.columns}
        if sourcepv_intersection is None:
            sourcepv_intersection = cand_here
        else:
            sourcepv_intersection &= cand_here

    species_cols = [c for c in first_species_order if c in species_intersection]
    if not species_cols:
        raise SystemExit("No common species columns across all files after intersection.")

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
        use_cols = base_cols + species_cols + ([sourcepv_col] if have_sourcepv else [])
        df_all.append(pd.read_csv(fn, usecols=use_cols))
    df = pd.concat(df_all, ignore_index=True)

    # -----------------------------
    # sanitize inputs (DO THIS BEFORE BUILDING AXES)
    # -----------------------------
    Z = df["Z"].to_numpy(dtype=float)
    Z = np.where(np.isfinite(Z), Z, np.nan)

    rho = df["rho"].to_numpy(dtype=float)
    rho = np.where(np.isfinite(rho), rho, RHO_FLOOR)
    rho = np.maximum(RHO_FLOOR, rho)

    T = df["T"].to_numpy(dtype=float)  # allow NaNs; handled in binning
    PV_raw = df["PV"].to_numpy(dtype=float)

    SourcePV = None
    if have_sourcepv:
        SourcePV = df[sourcepv_col].to_numpy(dtype=float)
        SourcePV = np.where(np.isfinite(SourcePV), SourcePV, 0.0)

    # -----------------------------
    # Build axes
    #   - Z axis from actual data range (prevents huge empty Z space)
    #   - PV axis stays [0..1] after normalization
    # -----------------------------
    Z_finite = Z[np.isfinite(Z)]
    if Z_finite.size == 0:
        raise SystemExit("All Z are non-finite; cannot build axis.")

    Z_lo = float(np.min(Z_finite))
    Z_hi = float(np.max(Z_finite))
    if not np.isfinite(Z_lo) or not np.isfinite(Z_hi) or (Z_hi <= Z_lo):
        raise SystemExit(f"Bad Z range: Z_lo={Z_lo}, Z_hi={Z_hi}")

    def _build_padded_axis_range(n_total: int, lo: float, hi: float, pad_eps: float) -> np.ndarray:
        if n_total < 3:
            raise ValueError("Need n_total>=3 for padding.")
        n_int = n_total - 2
        interior = np.linspace(lo, hi, n_int)
        pad = pad_eps * max(1.0, abs(hi - lo))
        return np.concatenate(([lo - pad], interior, [hi + pad]))

    Zgrid = _build_padded_axis_range(NZ, Z_lo, Z_hi, PAD_EPS)
    PVgrid = _build_padded_axis(NPV, PAD_EPS)

    # Clip Z into interior axis
    Z = np.where(np.isfinite(Z), Z, Zgrid[1])
    Z = np.clip(Z, Zgrid[1], Zgrid[-2])

    print(f"[DIAG] Z axis range: data [{Z_lo:.6g}, {Z_hi:.6g}]  |  axis [{Zgrid[0]:.6g}, {Zgrid[-1]:.6g}]")

    # -----------------------------
    # PVmin/PVmax(Z) in physical units (WINDOWED), then normalize PV -> [0,1]
    #   NOTE: min_per_bin for bounds should scale with total samples and NZ
    # -----------------------------
    n_total = int(np.count_nonzero(np.isfinite(PV_raw) & np.isfinite(Z)))
    n_int = len(Zgrid) - 2
    # aim for "a few hundred" if possible, but don't demand more than exists
    min_per_bin_bounds = int(np.clip(n_total // max(1, n_int), 50, 300))

    PVmin_vec, PVmax_vec = _compute_pv_bounds_per_Z(
        Z=Z,
        PV_raw=PV_raw,
        Zgrid=Zgrid,
        pv_lo_q=0.0,
        pv_hi_q=PV_HI_PERCENTILE,
        min_per_bin=min_per_bin_bounds,
        span_floor=1e-30,
    )

    Z_int = Zgrid[1:-1]
    pvmin_at_Z = np.interp(Z, Z_int, PVmin_vec[1:-1], left=PVmin_vec[1], right=PVmin_vec[-2])
    pvmax_at_Z = np.interp(Z, Z_int, PVmax_vec[1:-1], left=PVmax_vec[1], right=PVmax_vec[-2])
    span_at_Z = np.maximum(1e-30, pvmax_at_Z - pvmin_at_Z)

    PV = (PV_raw - pvmin_at_Z) / span_at_Z
    PV = np.where(np.isfinite(PV), PV, 0.0)
    PV = np.clip(PV, 0.0, 1.0)

    PV, SourcePV, flipped, dT_rows, frac_negative = _robust_pv_autoflip_quantile(
        Z=Z, PV=PV, T=T, PV_raw=PV_raw, Zgrid=Zgrid,
        have_sourcepv=have_sourcepv, SourcePV=SourcePV,
        min_rows=10,
        min_samples_row=60,   # <- your rows have ~696; safe to lower
        q_lo=PV_Q_LO, q_hi=PV_Q_HI,
        exclude_z_edge_rows=2
    )

    corr_byZ, tlo_q, thi_q, nrows_used = _diag_pvT_quantiles_byZ(
        Z=Z, PV=PV, T=T, PV_raw=PV_raw, Zgrid=Zgrid,
        q_lo=PV_Q_LO, q_hi=PV_Q_HI,
        min_samples_row=60,
        pv_raw_span_min=1e-12,
        exclude_z_edge_rows=2
    )
    
    # -----------------------------
    # DIAGNOSTIC #1: Per-Z cloud diagnostics (adaptive indices)
    # -----------------------------
    Zb = 0.5 * (Zgrid[:-1] + Zgrid[1:])
    Zi = np.searchsorted(Zb, Z, side="left")
    Zi = np.clip(Zi, 0, len(Zgrid) - 1)

    clip0 = PV <= 1e-12
    clip1 = PV >= 1.0 - 1e-12

    def _pick_zi_samples(nz):
        # interior rows are [1 .. nz-2]
        if nz <= 4:
            return [1, nz - 2]
        # avoid edge-adjacent interior rows (1 and nz-2); pick 2 and nz-3 instead
        cand = [
            2,
            2 + int(round(0.25 * (nz - 5))),
            2 + int(round(0.50 * (nz - 5))),
            2 + int(round(0.75 * (nz - 5))),
            nz - 3,
        ]
        # keep only valid + unique + sorted
        cand = sorted({int(np.clip(z, 2, nz - 3)) for z in cand})
        return cand

    zi_check = _pick_zi_samples(len(Zgrid))

    print("[DIAG] --- Per-Z cloud diagnostics (selected Z indices) ---")
    for zi in zi_check:
        m = (Zi == zi) & np.isfinite(PV) & np.isfinite(T) & np.isfinite(PV_raw)
        n = int(np.count_nonzero(m))
        if n == 0:
            print(f"  zi={zi:3d}  Z~{float(Zgrid[zi]):.6g}  n=0 (no samples)")
            continue

        corr_zi = np.nan
        if n > 5:
            corr_zi = float(np.corrcoef(PV[m], T[m])[0, 1])

        frac_clip = float(np.mean((clip0[m] | clip1[m]).astype(float)))
        pvmin_raw = float(np.min(PV_raw[m]))
        pvmax_raw = float(np.max(PV_raw[m]))

        lo_m = m & (PV <= 0.05)
        hi_m = m & (PV >= 0.95)
        tlo_zi = float(np.mean(T[lo_m])) if np.count_nonzero(lo_m) > 10 else np.nan
        thi_zi = float(np.mean(T[hi_m])) if np.count_nonzero(hi_m) > 10 else np.nan

        print(
            f"  zi={zi:3d}  Z~{float(Zgrid[zi]):.6g}  n={n:6d}  corr={corr_zi:+.3f}  clipFrac={frac_clip:.2f}  "
            f"PVraw[min,max]=[{pvmin_raw:.3e},{pvmax_raw:.3e}]  PVmin_vec={float(PVmin_vec[zi]):.3e} PVmax_vec={float(PVmax_vec[zi]):.3e}  "
            f"Tlo={tlo_zi if np.isfinite(tlo_zi) else np.nan:.1f} Thi={thi_zi if np.isfinite(thi_zi) else np.nan:.1f}"
        )

    row_n = np.bincount(Zi, minlength=len(Zgrid))
    print(f"[DIAG] Z-row sample counts: min={row_n.min()}, p10={np.percentile(row_n,10):.0f}, median={np.median(row_n):.0f}, p90={np.percentile(row_n,90):.0f}, max={row_n.max()}")


    # PV raw stats for metadata
    pv_finite = PV_raw[np.isfinite(PV_raw)]
    if pv_finite.size == 0:
        pv_lo = 0.0
        pv_hi = 1.0
    else:
        pv_lo = float(np.min(pv_finite))
        pv_hi = float(np.percentile(pv_finite, PV_HI_PERCENTILE))

    # -----------------------------
    # build thermo tables (copy sparse rows BEFORE nearest-fill)
    # -----------------------------
    T_tbl_raw, counts_T = _favre_bin_2d(Z, PV, rho, T, Zgrid, PVgrid, MIN_BIN_COUNT)

    T_tbl_raw = _copy_sparse_rows_from_nearest_good(
        T_tbl_raw, counts_T,
        min_bins_nonempty=40,
        min_count_per_bin=MIN_BIN_COUNT
    )

    T_tbl = _nearest_fill(T_tbl_raw)

    # NEW: boundary repair (fixes your zi=1 issue)
    T_tbl = _enforce_Z_boundary_rows_from_data(
        Z=Z, PV=PV, T=T,
        Zgrid=Zgrid, PVgrid=PVgrid,
        T_tbl=T_tbl, counts_T=counts_T,
        z_frac=0.02, pv_tail=0.05,
        min_samples=200, fallback_T=300.0,
    )

    T_tbl = _pad_extrapolate_const(T_tbl)


    rho_tbl_raw, counts_rho = _favre_bin_2d(Z, PV, rho, rho, Zgrid, PVgrid, MIN_BIN_COUNT)
    rho_tbl_raw = _copy_sparse_rows_from_nearest_good(rho_tbl_raw, counts_rho, min_bins_nonempty=25)
    rho_tbl = _pad_extrapolate_const(_nearest_fill(rho_tbl_raw))

    if have_sourcepv and (SourcePV is not None):
        spv_tbl_raw, counts_spv = _favre_bin_2d(Z, PV, rho, SourcePV, Zgrid, PVgrid, MIN_BIN_COUNT)
        spv_tbl_raw = _copy_sparse_rows_from_nearest_good(spv_tbl_raw, counts_spv, min_bins_nonempty=25)
        spv_tbl = _pad_extrapolate_const(_nearest_fill(spv_tbl_raw))
        print(f"[INFO] Using PV source column '{sourcepv_col}' to build thermo/SourcePV.csv"
              f"{' (sign flipped)' if flipped else ''}")
    else:
        spv_tbl = np.zeros((len(Zgrid), len(PVgrid)), dtype=float)
        print("[WARN] No PV source column found in ALL files (tried: " + ", ".join(SOURCEPV_CANDIDATES) + ").")
        print("[WARN] Writing thermo/SourcePV.csv as ZEROS.")

    # -----------------------------
    # DIAGNOSTIC #2: table row diagnostics (T_tbl + counts)
    # -----------------------------
    print("[DIAG] --- Table row diagnostics (T_tbl + counts) ---")
    for zi in zi_check:
        zi = int(np.clip(zi, 0, len(Zgrid) - 1))
        t0 = float(T_tbl[zi, 1])
        t1 = float(T_tbl[zi, -2])
        dT = t1 - t0

        c_row = counts_T[zi, :]
        cmin = int(np.min(c_row))
        cmax = int(np.max(c_row))
        n0 = int(np.count_nonzero(c_row == 0))
        nlt = int(np.count_nonzero(c_row < int(MIN_BIN_COUNT)))
        print(
            f"  zi={zi:3d}  Z~{float(Zgrid[zi]):.6g}  T(PV~0)={t0:.2f}  T(PV~1)={t1:.2f}  dT={dT:+.2f}  "
            f"counts[min,max]=[{cmin},{cmax}]  bins(count=0)={n0}  bins(count<MIN)={nlt}"
        )

    # -----------------------------
    # species tables
    # -----------------------------
    species_tables = {}
    for yc in species_cols:
        sp_name = yc[2:]  # drop "Y."
        Yk = df[yc].to_numpy(dtype=float)
        Yk = np.where(np.isfinite(Yk), Yk, 0.0)
        Yk = np.clip(Yk, 0.0, 1.0)

        Y_tbl_raw, counts_Y = _favre_bin_2d(Z, PV, rho, Yk, Zgrid, PVgrid, MIN_BIN_COUNT)
        Y_tbl_raw = _copy_sparse_rows_from_nearest_good(Y_tbl_raw, counts_Y, min_bins_nonempty=25)
        Y_tbl = _pad_extrapolate_const(_nearest_fill(Y_tbl_raw))
        species_tables[sp_name] = Y_tbl

    # -----------------------------
    # metadata + write tarball
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
            "NZ": int(NZ),
            "NPV": int(NPV),
            "Z_data_min": float(Z_lo),
            "Z_data_max": float(Z_hi),
            "MIN_BIN_COUNT": int(MIN_BIN_COUNT),
            "PAD_EPS": float(PAD_EPS),
            "PV_HI_PERCENTILE": float(PV_HI_PERCENTILE),
            "rho_floor": float(RHO_FLOOR),
            "pv_raw_min": float(pv_lo),
            "pv_raw_hi_percentile": float(pv_hi),
            "pvmin_vec_min": float(np.min(PVmin_vec)),
            "pvmax_vec_max": float(np.max(PVmax_vec)),
            "pv_axis_flipped": bool(flipped),
            "pvT_method": "byZ_median_quantiles",
            "pvT_rows_used": int(nrows_used),
            "pvT_q_lo": float(PV_Q_LO),
            "pvT_q_hi": float(PV_Q_HI),
            "pvT_corr_used": float(corr_byZ) if np.isfinite(corr_byZ) else None,
            "pvT_Tmean_qlo": float(tlo_q) if np.isfinite(tlo_q) else None,
            "pvT_Tmean_qhi": float(thi_q) if np.isfinite(thi_q) else None,
            "sourcepv_col": sourcepv_col if have_sourcepv else None,
            "pv_bounds_method": "perZ",
            "pv_bounds_min_per_bin": int(min_per_bin_bounds),
            "sparse_row_copy_min_bins_nonempty": 5,
        },
        "source": files,
    }

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
    print(f"  PV norm:  pv_lo={pv_lo:.6e}, pv_hi(p{PV_HI_PERCENTILE:g})={pv_hi:.6e}  flipped={flipped}")
    print(f"  SourcePV: {'from ' + sourcepv_col if (have_sourcepv and SourcePV is not None) else 'ZEROS'}")
    print(f"  species:  {len(species_tables)} tables")



if __name__ == "__main__":
    main()

