#!/usr/bin/env python3
'''Validate full OpenFOAM-7 premixed FGM tables in the zero-variance PVeta limit.'''
from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np

REQUIRED4 = ["T", "psi", "mu", "alpha", "SourcePV", "YWI", "YuWI", "YbWI"]
REQUIRED2 = ["PVmin", "PVmax", "Yu2I", "YuYbI", "Yb2I"]


def parse_list(path: Path, key: str) -> np.ndarray:
    text = path.read_text(errors="replace")
    start = text.find(key)
    if start < 0:
        raise ValueError(f"{path}: missing keyword {key}")
    toks = re.findall(r"\(|\)|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", text[start + len(key):])
    pos = 0
    def get():
        nonlocal pos
        n = int(float(toks[pos])); pos += 1
        if toks[pos] != "(":
            raise ValueError(f"{path}: expected list after length")
        pos += 1
        values = []
        for _ in range(n):
            if pos + 1 < len(toks) and toks[pos + 1] == "(":
                values.append(get())
            else:
                values.append(float(toks[pos])); pos += 1
        if toks[pos] != ")":
            raise ValueError(f"{path}: unterminated list")
        pos += 1
        return values
    return np.asarray(get(), dtype=float)


def read_param(text: str, key: str) -> np.ndarray:
    match = re.search(rf"^\s*{key}\s*\(([^;]*)\)\s*;", text, re.S | re.M)
    if not match:
        raise ValueError(f"Missing {key}")
    return np.asarray([float(token) for token in match.group(1).split()], dtype=float)


def relative_error(actual: np.ndarray, expected: np.ndarray, floor: float) -> float:
    return float(np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), floor)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", required=True)
    p.add_argument("--require-species", nargs="*", default=[])
    p.add_argument("--check-z", nargs="*", type=float, default=[0.04293, 0.1559])
    p.add_argument("--pilot-z", type=float, default=0.04293)
    p.add_argument("--require-zero-input-variances", action="store_true")
    args = p.parse_args()
    case = Path(args.case)
    props = (case / "constant/tableProperties").read_text(errors="replace")
    pvprops = (case / "constant/PVtableProperties").read_text(errors="replace")
    if not re.search(r"\binterpolationType\s+linearInterpolation\s*;", props):
        raise SystemExit("[FAIL] tableProperties must use linearInterpolation.")
    if not re.search(r"\binterpolationType\s+PVlinearInterpolation\s*;", pvprops):
        raise SystemExit("[FAIL] PVtableProperties must use PVlinearInterpolation.")
    match = re.search(r'tablePath\s+"?([^";]+)"?\s*;', props)
    if not match:
        raise SystemExit("[FAIL] tablePath missing from tableProperties.")
    tables = case / match.group(1)
    axes = {key: read_param(props, key) for key in ("varPV_param", "PV_param", "varZ_param", "Z_param")}
    pv_axes = {key: read_param(pvprops, key) for key in ("varZ_param", "Z_param")}
    dims4 = tuple(len(axes[key]) for key in ("varPV_param", "PV_param", "varZ_param", "Z_param"))
    dims2 = tuple(len(pv_axes[key]) for key in ("varZ_param", "Z_param"))

    data4 = {}
    for name in REQUIRED4 + args.require_species:
        f = tables / f"{name}_table"
        if not f.exists():
            raise SystemExit(f"[FAIL] Missing {f}.")
        arr = parse_list(f, f"{name}_table")
        if arr.shape != dims4:
            raise SystemExit(f"[FAIL] {name}_table shape {arr.shape}; expected {dims4}.")
        data4[name] = arr
    data2 = {}
    for name in REQUIRED2:
        f = tables / f"{name}_table"
        if not f.exists():
            raise SystemExit(f"[FAIL] Missing {f}.")
        arr = parse_list(f, f"{name}_table")
        if arr.shape != dims2:
            raise SystemExit(f"[FAIL] {name}_table shape {arr.shape}; expected {dims2}.")
        data2[name] = arr

    c_axis, z_axis = axes["PV_param"], axes["Z_param"]
    for value in args.check_z:
        if value < z_axis[0] - 1e-12 or value > z_axis[-1] + 1e-12:
            raise SystemExit(f"[FAIL] Requested Z={value:g} outside Z_param [{z_axis[0]:g}, {z_axis[-1]:g}].")
    ipv0 = int(np.argmin(np.abs(axes["varPV_param"])))
    izeta0 = int(np.argmin(np.abs(axes["varZ_param"])))
    if abs(axes["varPV_param"][ipv0]) > 1e-12 or abs(axes["varZ_param"][izeta0]) > 1e-12:
        raise SystemExit("[FAIL] Zero is absent from the replicated variance axes.")

    source = data4["SourcePV"][ipv0, :, izeta0, :]
    pvmin = data2["PVmin"][izeta0, :]
    pvmax = data2["PVmax"][izeta0, :]
    pv = pvmin[None, :] + c_axis[:, None]*(pvmax - pvmin)[None, :]
    source_expected = {
        "YWI": pv*source,
        "YuWI": pvmin[None, :]*source,
        "YbWI": pvmax[None, :]*source,
    }
    moments_expected = {
        "Yu2I": pvmin*pvmin,
        "YuYbI": pvmin*pvmax,
        "Yb2I": pvmax*pvmax,
    }
    source_err = max(relative_error(data4[name][ipv0, :, izeta0, :], target, 1e-18)
                     for name, target in source_expected.items())
    moment_err = max(relative_error(data2[name][izeta0, :], target, 1e-20)
                     for name, target in moments_expected.items())

    gc = data2["YuYbI"][izeta0, :] - data2["Yu2I"][izeta0, :]
    hc = data2["Yb2I"][izeta0, :] - 2.0*data2["YuYbI"][izeta0, :] + data2["Yu2I"][izeta0, :]
    valid = hc > 1e-14
    residual = np.zeros_like(pv)
    if np.any(valid):
        residual[:, valid] = ((pv[:, valid]**2 - data2["Yu2I"][izeta0, valid][None, :]
                              - 2.0*gc[valid][None, :]*c_axis[:, None]) / hc[valid][None, :]
                              - c_axis[:, None]**2)
    scaled_var_residual = float(np.max(np.abs(residual[:, valid]))) if np.any(valid) else 0.0
    if source_err > 5e-6 or moment_err > 5e-6 or scaled_var_residual > 5e-6:
        raise SystemExit(f"[FAIL] Zero-variance identities fail: source={source_err:.3e}, moments={moment_err:.3e}, scaledVarPV={scaled_var_residual:.3e}")

    spread = 0.0
    for arr in data4.values():
        spread = max(spread, float(np.max(np.abs(arr - arr[ipv0, :, izeta0, :][None, :, None, :]))))
    for arr in data2.values():
        spread = max(spread, float(np.max(np.abs(arr - arr[izeta0, :][None, :]))))
    if spread > 1e-12:
        raise SystemExit(f"[FAIL] Tables are not replicated zero-variance slices; maximum spread={spread:.6e}.")

    if args.require_zero_input_variances:
        for name in ("varPV", "varZ"):
            f = case / "0" / name
            if not f.exists():
                raise SystemExit(f"[FAIL] Missing initial field {f}.")
            text = f.read_text(errors="replace")
            if not re.search(r"\binternalField\s+uniform\s+0(?:\.0*)?\s*;", text):
                raise SystemExit(f"[FAIL] {f} internalField is not uniform zero.")

    iz = int(np.argmin(np.abs(z_axis - args.pilot_z)))
    jc = int(np.argmax(source[:, iz]))
    print(f"[OK] Table directory: {tables}")
    print(f"[OK] FGM field topology: {dims4}; PV family topology: {dims2}")
    print("[OK] Complete useProgressVariableVariance table interface is present.")
    print(f"[OK] Zero-variance identities: source relerr={source_err:.3e}, moments relerr={moment_err:.3e}, scaledVarPV residual={scaled_var_residual:.3e}")
    print(f"[OK] Replicated variance slices: maximum spread={spread:.3e}")
    print(f"[OK] Near pilot Z={z_axis[iz]:.8g}: max SourcePV={source[jc, iz]:.6e} at scaledPV={c_axis[jc]:.5g}")
    print("[LIMITATION] Validated for varPV=0 and varZ=0 only; not a nonzero presumed-PDF database.")


if __name__ == "__main__":
    main()
