#!/usr/bin/env python3
import argparse
import io
import os
import tarfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def _load_csv_bytes(b: bytes) -> np.ndarray:
    return np.loadtxt(io.BytesIO(b), delimiter=",")

def _load_from_tar(tar_path: Path, member: str) -> np.ndarray:
    with tarfile.open(tar_path, "r:*") as tf:
        m = tf.getmember(member)
        b = tf.extractfile(m).read()
    return _load_csv_bytes(b)

def _load_from_dir(root: Path, rel: str) -> np.ndarray:
    return np.loadtxt(root / rel, delimiter=",")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=str, default="02_tables.tar.xz",
                    help="Either 02_tables.tar.xz or an unpacked 02_tables/ directory")
    ap.add_argument("--out", type=str, default="thermo_T.png", help="output PNG filename")
    ap.add_argument("--vmin", type=float, default=None, help="optional fixed color min")
    ap.add_argument("--vmax", type=float, default=None, help="optional fixed color max")
    args = ap.parse_args()

    p = Path(args.tables)

    if p.is_file() and p.suffixes[-2:] == [".tar", ".xz"] or p.suffix == ".xz":
        Z = _load_from_tar(p, "02_tables/axes/Z.csv").ravel()
        PV = _load_from_tar(p, "02_tables/axes/PV.csv").ravel()
        T = _load_from_tar(p, "02_tables/thermo/T.csv")
    else:
        root = p
        Z = _load_from_dir(root, "axes/Z.csv").ravel()
        PV = _load_from_dir(root, "axes/PV.csv").ravel()
        T = _load_from_dir(root, "thermo/T.csv")

    # interior (unpadded)
    Zi = Z[1:-1]
    PVi = PV[1:-1]
    Ti = T[1:-1, 1:-1]

    fig = plt.figure()
    ax = plt.gca()

    im = ax.imshow(
        Ti,
        origin="lower",
        aspect="auto",
        extent=[float(PVi[0]), float(PVi[-1]), float(Zi[0]), float(Zi[-1])],
        vmin=args.vmin,
        vmax=args.vmax,
    )
    ax.set_xlabel("PV")
    ax.set_ylabel("Z")
    ax.set_title("FGM table: thermo/T.csv (interior, unpadded)")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("T [K]")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.show()

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()

