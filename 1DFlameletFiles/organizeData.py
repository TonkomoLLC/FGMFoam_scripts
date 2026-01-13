#!/usr/bin/env python3
# organizeData.py (Cantera 3.1–3.2) — robust + UNIFORM-Z RESAMPLING
# Reads flamelets saved by generate1DFlamelets.py and writes post_*.csv for tabulation.
#
# Key fix vs your version:
#   - compute Z, PV, SourcePV on the native flame grid
#   - then RESAMPLE EVERYTHING onto a uniform Z grid [0..1]
#     (prevents huge clustering at Z~0 and Z~1 and sparse mid-Z)
#
# Output columns:
#   z,u,V,T,rho,Z,PV,SourcePV,chiZ,S_n,Y.<species...>
#
# Notes:
#   - "z" in the output becomes a *pseudo-coordinate* monotonically increasing with Z
#     (we set it equal to Z after resampling). This is intentional for tabulation.

import cantera as ct
import numpy as np
import glob, os, errno

# --------- user knobs ----------
mech = 'gri30.yaml'
search_glob = 'strain_loop_*.yaml'   # you can also point this at *.csv
out_prefix = 'post_'
P_default = 101325.0                 # Pa

# Uniform-Z resampling (THIS is the important part)
RESAMPLE_ON_Z = True
NZ_RESAMPLE = 1200                   # 400–2000 typical; higher gives smoother tables
Z_EPS = 1e-12                        # guard for duplicates / monotonicity
# -------------------------------

def load_solution_array(gas, fn, name='diff1D'):
    """
    Robust loader:
      1) try restore(fn, name)
      2) if empty, try restore(fn) with no name
      3) if empty, try CSV fallback <basename>.csv
    Returns (sa, fmt_str)
    """
    sa = ct.SolutionArray(gas)
    ext = os.path.splitext(fn)[1].lower()

    if ext in ('.yaml', '.yml', '.h5', '.hdf5', '.hdf'):
        # 1) named group
        try:
            sa.restore(fn, name=name)
        except Exception:
            pass

        # 2) default group
        if sa.shape[0] == 0:
            try:
                sa2 = ct.SolutionArray(gas)
                sa2.restore(fn)
                if sa2.shape[0] > 0:
                    sa = sa2
            except Exception:
                pass

        # 3) CSV fallback
        if sa.shape[0] == 0:
            csv = os.path.splitext(fn)[0] + '.csv'
            if os.path.exists(csv):
                sa = ct.SolutionArray(gas)
                sa.read_csv(csv)
                return sa, 'csv'
            raise ValueError(f"Empty container in {os.path.basename(fn)} and no CSV fallback found.")

        return sa, 'yaml'

    elif ext == '.csv':
        sa.read_csv(fn)
        return sa, 'csv'

    else:
        raise ValueError(f"Unsupported file type: {ext}")

def pick_col(df, *names):
    """Return numpy column by first matching name present in df, else None."""
    if df is None:
        return None
    for n in names:
        if n in df.columns:
            return df[n].to_numpy()
    return None

def derivative_uniform_grid(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dydx = np.zeros_like(y, dtype=float)
    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("Grid must be strictly increasing.")
    dy = np.diff(y)
    dydx[:-1] = dy / dx
    dydx[-1]  = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return dydx

def bilger_b_from_state(gas, T, P, Y):
    gas.TPY = T, P, Y
    YC = gas.elemental_mass_fraction('C')
    YH = gas.elemental_mass_fraction('H')
    YO = gas.elemental_mass_fraction('O')
    M_C = gas.atomic_weight('C')
    M_H = gas.atomic_weight('H')
    M_O = gas.atomic_weight('O')
    return 2.0*YC/M_C + 0.5*YH/M_H - YO/M_O

def progress_variable(Y, mw, k_H2O, k_CO2, k_H2, k_CO):
    n_H2O = Y[k_H2O] / mw[k_H2O]
    n_CO2 = Y[k_CO2] / mw[k_CO2]
    n_H2  = Y[k_H2]  / mw[k_H2]
    n_CO  = Y[k_CO]  / mw[k_CO]
    return 4.0*n_H2O + 2.0*n_CO2 + 0.5*n_H2 + 1.0*n_CO

def stream_b(gas, X_str, T=300.0, P=101325.0):
    gas.TPX = T, P, X_str
    YC = gas.elemental_mass_fraction('C')
    YH = gas.elemental_mass_fraction('H')
    YO = gas.elemental_mass_fraction('O')
    M_C = gas.atomic_weight('C')
    M_H = gas.atomic_weight('H')
    M_O = gas.atomic_weight('O')
    return 2.0*YC/M_C + 0.5*YH/M_H - YO/M_O

def _interp_vs_Z(Z_in, y_in, Z_out, fill_left=None, fill_right=None):
    """
    1D interpolation y(Z) onto Z_out using np.interp.
    Requires Z_in strictly increasing; we enforce that upstream.
    """
    if fill_left is None:
        fill_left = float(y_in[0])
    if fill_right is None:
        fill_right = float(y_in[-1])
    return np.interp(Z_out, Z_in, y_in, left=fill_left, right=fill_right)

def _make_monotone_unique(Z, *arrays):
    """
    Sort by Z, drop non-finite, clip to [0,1], then enforce strictly increasing Z
    by removing duplicates and (if needed) nudging by Z_EPS.
    Returns (Zs, arrays_sorted...)
    """
    Z = np.asarray(Z, dtype=float)
    m = np.isfinite(Z)
    for a in arrays:
        m &= np.isfinite(np.asarray(a)).all(axis=-1) if np.asarray(a).ndim > 1 else np.isfinite(np.asarray(a))
    if np.count_nonzero(m) < 3:
        return None

    Zm = np.clip(Z[m], 0.0, 1.0)
    order = np.argsort(Zm)
    Zs = Zm[order]

    out = []
    for a in arrays:
        a = np.asarray(a)
        out.append(a[m][order])

    # Remove duplicates (Z not strictly increasing)
    # Strategy: keep first occurrence after sorting, then nudge any remaining non-increasing points.
    # First, collapse exact duplicates via unique on rounded Z
    Zr = np.round(Zs / max(Z_EPS, 1e-15)).astype(np.int64)
    _, keep_idx = np.unique(Zr, return_index=True)
    keep_idx.sort()
    Zs = Zs[keep_idx]
    out = [a[keep_idx] for a in out]

    # Ensure strictly increasing by nudging (rare, but safe)
    for i in range(1, len(Zs)):
        if Zs[i] <= Zs[i-1]:
            Zs[i] = min(1.0, Zs[i-1] + Z_EPS)

    # Ensure we include endpoints (helps tabulation stability)
    if Zs[0] > 0.0:
        Zs = np.insert(Zs, 0, 0.0)
        out = [np.insert(a, 0, a[0], axis=0) for a in out]
    if Zs[-1] < 1.0:
        Zs = np.append(Zs, 1.0)
        out = [np.append(a, [a[-1]], axis=0) for a in out]

    return (Zs, *out)

def main():
    gas = ct.Solution(mech)

    # Reference streams (match generator)
    X_fuel = 'CH4:1'
    X_ox   = 'O2:0.21, N2:0.79'
    b_f = stream_b(gas, X_fuel)
    b_o = stream_b(gas, X_ox)

    k_CO2 = gas.species_index('CO2')
    k_H2O = gas.species_index('H2O')
    k_CO  = gas.species_index('CO')
    k_H2  = gas.species_index('H2')
    mw = gas.molecular_weights
    ns = gas.n_species

    files = sorted(glob.glob(search_glob))
    if not files:
        raise FileNotFoundError(f"No files match {search_glob}")

    for fn in files:
        sa, fmt = load_solution_array(gas, fn, name='diff1D')
        if sa.shape[0] == 0:
            print(f"[SKIP] {os.path.basename(fn)}: empty container.")
            continue

        # DataFrame view (may include extra columns like grid/velocity in some formats)
        try:
            df = sa.to_pandas()
        except Exception:
            df = None

        # --- state ---
        T = np.asarray(sa.T, dtype=float)
        Y = np.asarray(sa.Y, dtype=float)            # (n, nspecies)

        # Pressure: optional/constant
        try:
            P_arr = np.asarray(sa.P, dtype=float)
            P = float(P_arr[0]) if P_arr.size > 0 else P_default
        except Exception:
            P = P_default

        # Density: from SA if available else compute
        try:
            rho = np.asarray(sa.density, dtype=float)
            if rho.size == 0:
                raise ValueError
        except Exception:
            rho = np.empty(len(T), dtype=float)
            for i in range(len(T)):
                gas.TPY = T[i], P, Y[i, :]
                rho[i] = gas.density

        # --- geometry/flow from DataFrame columns, with aliases ---
        z = pick_col(df, 'grid', 'z', 'x', 'position', 'distance')
        u = pick_col(df, 'velocity', 'u', 'axial_velocity')
        V = pick_col(df, 'spread_rate', 'V', 'radial_velocity', 'transverse_velocity')

        # fallbacks
        if z is None:
            z = np.linspace(0.0, 1.0, len(T))
            print(f"[WARN] {os.path.basename(fn)}: no grid column; using normalized surrogate [0,1].")
        if u is None:
            u = np.zeros_like(T)
        if V is None:
            V = np.zeros_like(T)

        z = np.asarray(z, dtype=float)
        u = np.asarray(u, dtype=float)
        V = np.asarray(V, dtype=float)

        n = len(T)
        if len(z) != n or Y.shape[0] != n:
            print(f"[SKIP] {os.path.basename(fn)}: inconsistent lengths (T={n}, z={len(z)}, Y={Y.shape[0]}).")
            continue

        # Ensure z is increasing for derivatives (Counterflow grid should be increasing; guard anyway)
        order_z = np.argsort(z)
        z = z[order_z]
        u = u[order_z]
        V = V[order_z]
        T = T[order_z]
        rho = rho[order_z]
        Y = Y[order_z, :]

        # --- derived scalars on native grid ---
        Z_raw = np.empty(n, dtype=float)
        PV = np.empty(n, dtype=float)

        for i in range(n):
            b_local = bilger_b_from_state(gas, T[i], P, Y[i, :])
            Z_raw[i] = (b_local - b_o) / (b_f - b_o)
            PV[i] = progress_variable(Y[i, :], mw, k_H2O, k_CO2, k_H2, k_CO)

        # Compute dudz, chiZ etc on native grid (optional / proxies as in your script)
        try:
            dudz = derivative_uniform_grid(z, u)
        except Exception:
            dudz = np.gradient(u, z, edge_order=1)

        # IMPORTANT: For chiZ proxy we need dZ/dz; use unclipped Z for derivative, then clip after.
        Z_for_grad = np.where(np.isfinite(Z_raw), Z_raw, 0.0)
        try:
            dZdz = derivative_uniform_grid(z, Z_for_grad)
        except Exception:
            dZdz = np.gradient(Z_for_grad, z, edge_order=1)

        S_n  = dudz
        chiZ = 2.0 * (dZdz ** 2)  # still your proxy

        # --- chemistry source term for PV on native grid ---
        SourcePV = np.empty(n, dtype=float)
        for i in range(n):
            gas.TPY = T[i], P, Y[i, :]
            wdot = gas.net_production_rates  # kmol/m^3/s

            rho_i = max(float(rho[i]), 1e-12)
            val = (
                4.0 * wdot[k_H2O]
                + 2.0 * wdot[k_CO2]
                + 0.5 * wdot[k_H2]
                + 1.0 * wdot[k_CO]
            ) / rho_i
            SourcePV[i] = val if np.isfinite(val) else 0.0
        SourcePV = np.where(np.isfinite(SourcePV), SourcePV, 0.0)

        # Clip mixture fraction for tabulation range
        Z = np.clip(Z_raw, 0.0, 1.0)

        # ==========================
        # UNIFORM-Z RESAMPLING (fix)
        # ==========================
        if RESAMPLE_ON_Z:
            packed = _make_monotone_unique(
                Z,
                z, u, V, T, rho, PV, SourcePV, chiZ, S_n, Y
            )
            if packed is None:
                print(f"[SKIP] {os.path.basename(fn)}: insufficient finite/unique Z points to resample.")
                continue

            Zs, z_s, u_s, V_s, T_s, rho_s, PV_s, SPV_s, chiZ_s, S_n_s, Y_s = packed

            # Uniform Z grid
            Z_out = np.linspace(0.0, 1.0, int(NZ_RESAMPLE))

            # Interpolate scalar fields vs Z
            u_out     = _interp_vs_Z(Zs, u_s,     Z_out)
            V_out     = _interp_vs_Z(Zs, V_s,     Z_out)
            T_out     = _interp_vs_Z(Zs, T_s,     Z_out)
            rho_out   = _interp_vs_Z(Zs, rho_s,   Z_out)
            PV_out    = _interp_vs_Z(Zs, PV_s,    Z_out)
            SPV_out   = _interp_vs_Z(Zs, SPV_s,   Z_out)
            chiZ_out  = _interp_vs_Z(Zs, chiZ_s,  Z_out)
            S_n_out   = _interp_vs_Z(Zs, S_n_s,   Z_out)

            # Interpolate each species mass fraction vs Z
            Y_out = np.empty((len(Z_out), ns), dtype=float)
            for k in range(ns):
                Y_out[:, k] = _interp_vs_Z(Zs, Y_s[:, k], Z_out)
            Y_out = np.clip(Y_out, 0.0, 1.0)

            # IMPORTANT: make 'z' a monotone coordinate for downstream derivatives/users.
            # We set it equal to Z_out (so it's uniform and consistent).
            z_out = Z_out.copy()

            # Recompute any derivatives on the resampled grid if you want (optional).
            # Here we keep your chiZ/S_n already interpolated.

            Z_final = Z_out
            out_mat = np.column_stack([z_out, u_out, V_out, T_out, rho_out, Z_final, PV_out, SPV_out, chiZ_out, S_n_out, Y_out])
            n_out = len(Z_out)
        else:
            # write native grid (your original behavior)
            out_mat = np.column_stack([z, u, V, T, rho, Z, PV, SourcePV, chiZ, S_n, Y])
            n_out = len(z)

        # --- write output ---
        header = ['z','u','V','T','rho','Z','PV','SourcePV','chiZ','S_n'] + [f'Y.{s}' for s in gas.species_names]

        out_name = os.path.join(
            os.path.dirname(fn),
            out_prefix + os.path.basename(fn)
            .replace('.yaml','.csv').replace('.yml','.csv')
            .replace('.h5','.csv').replace('.hdf5','.csv').replace('.hdf','.csv')
        )

        np.savetxt(out_name, out_mat, delimiter=',', header=",".join(header), comments='')
        print(f"[OK] Wrote {out_name}  (n={n_out})  resample_on_Z={RESAMPLE_ON_Z}")

if __name__ == "__main__":
    try:
        main()
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

