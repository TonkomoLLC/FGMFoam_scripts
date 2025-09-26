#!/usr/bin/env python3
# generate1DFlamelets.py — Cantera 3.1 (v2.0)
# Robust ignition + recenter line-search + adaptive strain stepping (until extinction)

import cantera as ct
import numpy as np
import os

# ---------------- user inputs ----------------
air = 'O2:0.21, N2:0.79'
fuel = 'CH4:1'
mech = 'gri30.yaml'

width = 0.03                 # [m] domain width (0.02–0.05 OK; larger helps early "thick flame" warnings)
P_atm = 101325.0             # [Pa]

# Asymmetric inlets keep the flame off boundaries; OX > FUEL is typical
mdot_fuel = 0.010            # [kg/m^2/s]
mdot_ox   = 0.025            # [kg/m^2/s]  (ratio ~2.5)

T_f_in    = 300.0            # [K]
T_o_in    = 300.0            # [K]

transport_model = 'mixture-averaged'  # or 'multicomponent'
out_dir = './'               # where to save strain_loop_XX.{yaml,csv}

# continuation controls
strain_factor = 1.10         # nominal step in total mass flux (strain proxy)
max_steps = 200              # safety cap (loop runs until extinction or this limit)
loglevel = 1                 # 0=silent .. 2=verbose

# ignition / extinction logic
T_lit_threshold = 800.0      # flame considered "lit" if Tmax > this [K]
min_T_ok = 1200.0            # for accepting a step as clearly lit

# refinement & mesh control
max_grid_points = 4000
ref_ratio, ref_slope, ref_curve, ref_prune = 3.0, 0.12, 0.25, 0.03
# ------------------------------------------------


# ---------------- helpers ----------------
def is_lit(f):
    return float(np.max(f.T)) > T_lit_threshold

def unit_positions(f):
    """Return positions normalized to exactly [0,1] for set_profile()."""
    z = f.grid.astype(float)
    pos = (z - z[0]) / (z[-1] - z[0])
    pos[0] = 0.0
    pos[-1] = 1.0
    return pos

def seed_hotspot(f, T_peak=1700.0, center=0.55, halfwidth=0.12):
    """
    Seed a Gaussian temperature bump. 'center' and 'halfwidth' are in [0,1].
    """
    xi = unit_positions(f)
    T0 = f.oxidizer_inlet.T
    prof = T0 + (T_peak - T0) * np.exp(-((xi - center) / halfwidth) ** 2)
    f.set_profile('T', xi, prof)

def save_yaml_csv(f, path_yaml):
    f.save(path_yaml, name='diff1D', overwrite=True)
    f.save(path_yaml.replace('.yaml', '.csv'), name='diff1D', overwrite=True)

def xi_peak(f):
    z = f.grid
    return (z[np.argmax(f.T)] - z[0]) / (z[-1] - z[0] + 1e-30)

def recenter_line_search(f, target=0.55, tol=0.08, step=1.06, max_iters=5):
    """
    Move Tmax location toward 'target' by adjusting ox:fuel mdot ratio
    while keeping total mass flux constant. No orientation assumptions.
    """
    for _ in range(max_iters):
        xi0 = xi_peak(f)
        if abs(xi0 - target) <= tol:
            return xi0

        M  = f.oxidizer_inlet.mdot + f.fuel_inlet.mdot
        r0 = f.oxidizer_inlet.mdot / max(f.fuel_inlet.mdot, 1e-30)

        # Try ratio up
        r_up = max(0.05, min(20.0, r0 * step))
        f.oxidizer_inlet.mdot = M * r_up / (1.0 + r_up)
        f.fuel_inlet.mdot     = M / (1.0 + r_up)
        try:
            f.solve(loglevel=0, auto=False)
        except ct.CanteraError:
            f.solve(loglevel=0, auto=True)
        xi_up = xi_peak(f)

        # Restore and try ratio down
        f.oxidizer_inlet.mdot = M * r0 / (1.0 + r0)
        f.fuel_inlet.mdot     = M / (1.0 + r0)
        try:
            f.solve(loglevel=0, auto=False)
        except ct.CanteraError:
            pass

        r_dn = max(0.05, min(20.0, r0 / step))
        f.oxidizer_inlet.mdot = M * r_dn / (1.0 + r_dn)
        f.fuel_inlet.mdot     = M / (1.0 + r_dn)
        try:
            f.solve(loglevel=0, auto=False)
        except ct.CanteraError:
            f.solve(loglevel=0, auto=True)
        xi_dn = xi_peak(f)

        # Choose the direction that helps more
        if abs(xi_up - target) < abs(xi_dn - target):
            # keep r_up (already applied)
            pass
        else:
            # keep r_dn (already applied)
            pass

        # tighten step if neither helped much
        if (abs(xi_up - target) < abs(xi0 - target)) or (abs(xi_dn - target) < abs(xi0 - target)):
            continue
        step = 1.0 + (step - 1.0) * 0.6

    return xi_peak(f)

def adaptive_step_solve(f, step_target, tries=(1.0, 0.8, 0.65, 0.5, 0.35), min_T_accept=1200.0):
    """
    Advance total mdot by 'step_target' while preserving current ox:fuel ratio.
    Backtrack if solve ends cold/fails. Returns (ok_lit, used_step_factor).
    """
    M0 = f.oxidizer_inlet.mdot + f.fuel_inlet.mdot
    r0 = f.oxidizer_inlet.mdot / max(f.fuel_inlet.mdot, 1e-15)

    for frac in tries:
        sf = 1.0 + (step_target - 1.0) * frac   # e.g. 1.10, then 1.08, 1.06, ...
        M = M0 * sf
        f.oxidizer_inlet.mdot = M * r0 / (1.0 + r0)
        f.fuel_inlet.mdot     = M / (1.0 + r0)

        try:
            f.set_refine_criteria(ratio=ref_ratio, slope=ref_slope, curve=ref_curve, prune=ref_prune)
            f.solve(loglevel=loglevel, auto=False)
        except ct.CanteraError:
            try:
                f.solve(loglevel=loglevel, auto=True)
            except ct.CanteraError:
                # try smaller step
                continue

        if float(np.max(f.T)) > min_T_accept:
            return True, sf

        # else: try smaller step on next iteration

    # revert to last good M0
    f.oxidizer_inlet.mdot = M0 * r0 / (1.0 + r0)
    f.fuel_inlet.mdot     = M0 / (1.0 + r0)
    try:
        f.solve(loglevel=0, auto=False)
    except ct.CanteraError:
        pass
    return False, 1.0

def ignite_base_flame(f):
    """
    Try to obtain a lit flame at cold inlets using a hotspot, else briefly preheat oxidizer.
    """
    seed_hotspot(f, T_peak=1700.0, center=0.55, halfwidth=0.12)
    try:
        f.solve(loglevel=loglevel, auto=True)
    except ct.CanteraError:
        pass
    if is_lit(f):
        return True

    # quick preheat ladder then cool
    for Tox in (900, 700, 500, 300):
        f.oxidizer_inlet.T = Tox
        seed_hotspot(f, T_peak=max(1800.0, Tox + 600.0), center=0.55, halfwidth=0.14)
        try:
            f.solve(loglevel=loglevel, auto=True)
        except ct.CanteraError:
            continue
        if is_lit(f):
            if Tox != 300.0:
                f.oxidizer_inlet.T = 300.0
                f.solve(loglevel=loglevel, auto=False)
            return True
    return False
# ------------------------------------------------


def main():
    os.makedirs(out_dir, exist_ok=True)

    gas = ct.Solution(mech)
    gas.transport_model = transport_model

    f = ct.CounterflowDiffusionFlame(gas, width=width)
    f.transport_model = transport_model
    f.P = P_atm

    # inlets
    f.fuel_inlet.mdot = mdot_fuel
    f.fuel_inlet.T = T_f_in
    f.fuel_inlet.X = fuel

    f.oxidizer_inlet.mdot = mdot_ox
    f.oxidizer_inlet.T = T_o_in
    f.oxidizer_inlet.X = air

    f.set_refine_criteria(ratio=ref_ratio, slope=ref_slope, curve=ref_curve, prune=ref_prune)
    try:
        f.flame.set_max_grid_points(max_grid_points)
    except Exception:
        pass

    # --- Base ignition ---
    if not ignite_base_flame(f):
        # reduce strain once and retry
        M = (f.oxidizer_inlet.mdot + f.fuel_inlet.mdot) * 0.6
        r = f.oxidizer_inlet.mdot / max(f.fuel_inlet.mdot, 1e-12)
        f.oxidizer_inlet.mdot = M * r / (1.0 + r)
        f.fuel_inlet.mdot     = M / (1.0 + r)
        if not ignite_base_flame(f):
            print("[INFO] Could not obtain a lit base flamelet; stopping.")
            return

    xi = recenter_line_search(f, target=0.55, tol=0.08, step=1.06, max_iters=5)
    y0 = os.path.join(out_dir, 'strain_loop_00.yaml')
    save_yaml_csv(f, y0)
    print(f"[OK] Saved {y0}  (points={len(f.grid)}, Tmax={np.max(f.T):.1f} K, xi={xi:.3f})")

    last_lit = 0
    n = 1
    while n <= max_steps:
        # primary attempt
        ok, sf = adaptive_step_solve(f, step_target=strain_factor,
                                     tries=(1.0, 0.8, 0.65, 0.5, 0.35),
                                     min_T_accept=min_T_ok)
        if not ok:
            # fine ladder near the limit
            got_fine = False
            for fine_sf in (1.06, 1.04, 1.03, 1.02):
                ok2, _ = adaptive_step_solve(f, step_target=fine_sf,
                                             tries=(1.0, 0.8, 0.6),
                                             min_T_accept=min_T_ok)
                if ok2 and is_lit(f):
                    got_fine = True
                    sf = fine_sf
                    break
            if not got_fine:
                print(f"[INFO] Extinction around step {n}. Last lit: {last_lit}. Stopping.")
                break

        # recentre (preserving total mdot) and check lit
        xi = recenter_line_search(f, target=0.55, tol=0.08, step=1.06, max_iters=3)
        if not is_lit(f):
            # save once for diagnostics and stop
            yx = os.path.join(out_dir, f'strain_loop_{n:02d}.yaml')
            save_yaml_csv(f, yx)
            print(f"[EXTINCT] Saved {yx}  (points={len(f.grid)}, Tmax={np.max(f.T):.1f} K, xi={xi:.3f}, sf={sf:.3f})")
            print(f"[INFO] Extinction at step {n}. Last lit: {last_lit}. Stopping.")
            break

        # lit: save and continue
        yx = os.path.join(out_dir, f'strain_loop_{n:02d}.yaml')
        save_yaml_csv(f, yx)
        last_lit = n
        print(f"[OK] Saved {yx}  (points={len(f.grid)}, Tmax={np.max(f.T):.1f} K, xi={xi:.3f}, sf={sf:.3f})")
        n += 1

    if last_lit == 0:
        print("[RESULT] Only base flamelet is lit.")
    else:
        print(f"[RESULT] Last lit flamelet: strain_loop_{last_lit:02d}.yaml")

if __name__ == "__main__":
    main()

