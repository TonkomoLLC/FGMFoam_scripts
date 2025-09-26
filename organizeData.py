#!/usr/bin/env python3
# organizeData.py (Cantera 3.1 – DataFrame-based, robust)
# Reads flamelets saved by generate1DFlamelets.py and writes post_*.csv for tabulation.

import cantera as ct
import numpy as np
import glob, os, errno

# --------- user knobs ----------
mech = 'gri30.yaml'
search_glob = 'strain_loop_*.yaml'   # you can also point this at *.csv
out_prefix = 'post_'
P_default = 101325.0                 # Pa
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

    files = sorted(glob.glob(search_glob))
    if not files:
        raise FileNotFoundError(f"No files match {search_glob}")

    for fn in files:
        sa, fmt = load_solution_array(gas, fn, name='diff1D')
        if sa.shape[0] == 0:
            print(f"[SKIP] {os.path.basename(fn)}: empty container.")
            continue

        # DataFrame view (extras appear as columns here)
        try:
            df = sa.to_pandas()
        except Exception:
            df = None

        # --- state ---
        # Prefer SA properties for thermo/species (reliable across formats)
        T   = np.asarray(sa.T)
        Y   = np.asarray(sa.Y)              # shape (n, nspecies)

        # Pressure: optional/constant
        try:
            P_arr = np.asarray(sa.P)
            P = float(P_arr[0]) if P_arr.size > 0 else P_default
        except Exception:
            P = P_default

        # Density: from SA if available else compute
        try:
            rho = np.asarray(sa.density)
            if rho.size == 0:
                raise ValueError
        except Exception:
            rho = np.empty(len(T))
            for i in range(len(T)):
                gas.TPY = T[i], P, Y[i, :]
                rho[i] = gas.density

        # --- geometry/flow from DataFrame columns, with aliases ---
        z = pick_col(df, 'grid', 'x', 'z', 'position', 'distance')
        u = pick_col(df, 'velocity', 'u', 'axial_velocity')
        V = pick_col(df, 'spread_rate', 'V', 'radial_velocity', 'transverse_velocity')

        # fallbacks
        if z is None:
            npts = len(T)
            z = np.linspace(0.0, 1.0, npts)
            print(f"[WARN] {os.path.basename(fn)}: no grid column; using normalized surrogate [0,1].")
        if u is None:
            u = np.zeros_like(T)
        if V is None:
            V = np.zeros_like(T)

        n = len(z)

        # --- derived scalars ---
        Z  = np.empty(n)
        PV = np.empty(n)
        for i in range(n):
            b_local = bilger_b_from_state(gas, T[i], P, Y[i, :])
            Z[i] = (b_local - b_o) / (b_f - b_o)
            PV[i] = progress_variable(Y[i, :], mw, k_H2O, k_CO2, k_H2, k_CO)

        Z = np.clip(Z, 0.0, 1.0)

        dudz = derivative_uniform_grid(z, u)
        dZdz = derivative_uniform_grid(z, Z)
        S_n  = dudz
        chiZ = 2.0 * (dZdz ** 2)  # proxy; replace if you have a specific chi_Z

        # --- write output ---
        header = ['z','u','V','T','rho','Z','PV','chiZ','S_n'] \
                 + [f'Y.{s}' for s in gas.species_names]
        out = np.column_stack([z, u, V, T, rho, Z, PV, chiZ, S_n, Y])

        out_name = os.path.join(os.path.dirname(fn),
                                out_prefix + os.path.basename(fn)
                                .replace('.yaml','.csv').replace('.yml','.csv')
                                .replace('.h5','.csv').replace('.hdf5','.csv').replace('.hdf','.csv'))
        np.savetxt(out_name, out, delimiter=',', header=",".join(header), comments='')
        print(f"[OK] Wrote {out_name}  (n={n})")

if __name__ == "__main__":
    try:
        main()
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise

