#!/usr/bin/env python3
import argparse
import io
import tarfile

import numpy as np
import matplotlib.pyplot as plt


def read_csv_from_tar(tar_path: str, member: str) -> np.ndarray:
    with tarfile.open(tar_path, mode="r:*") as tf:
        try:
            m = tf.getmember(member)
        except KeyError as e:
            raise FileNotFoundError(f"{member} not found in {tar_path}") from e
        f = tf.extractfile(m)
        if f is None:
            raise FileNotFoundError(f"Could not extract {member} from {tar_path}")
        data = f.read()
    return np.loadtxt(io.BytesIO(data), delimiter=",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tar", help="Path to 02_tables.tar.xz")
    ap.add_argument("--field", default="T", choices=["T", "rho", "SourcePV"], help="Thermo field to plot")
    ap.add_argument("--out", default=None, help="Output PNG filename (default: <field>_table_plot.png)")
    ap.add_argument("--no-show", action="store_true", help="Do not display interactively")
    ap.add_argument("--full", action="store_true", help="Plot full table including padded rows/cols")
    ap.add_argument("--vmin", type=float, default=None, help="Colorbar min")
    ap.add_argument("--vmax", type=float, default=None, help="Colorbar max")
    ap.add_argument("--log", action="store_true", help="Use log10 color scale (positive-only fields)")
    args = ap.parse_args()

    out = args.out or f"{args.field}_table_plot.png"

    Z = read_csv_from_tar(args.tar, "02_tables/axes/Z.csv").reshape(-1)
    PV = read_csv_from_tar(args.tar, "02_tables/axes/PV.csv").reshape(-1)
    F = read_csv_from_tar(args.tar, f"02_tables/thermo/{args.field}.csv")

    if not args.full:
        # interior (unpadded)
        Zp = Z[1:-1]
        PVp = PV[1:-1]
        Fp = F[1:-1, 1:-1]
    else:
        Zp, PVp, Fp = Z, PV, F

    # Optional log scaling (useful for SourcePV sometimes)
    plot_data = Fp
    cbar_label = args.field
    if args.field == "T":
        cbar_label = "T [K]"
    elif args.field == "rho":
        cbar_label = r"$\rho$ [kg/m$^3$]"
    elif args.field == "SourcePV":
        cbar_label = "SourcePV [units as input]"

    if args.log:
        # Guard: log only works for strictly positive values
        pos = plot_data[np.isfinite(plot_data) & (plot_data > 0)]
        if pos.size == 0:
            raise SystemExit("[ERROR] --log requested but no positive finite values found.")
        plot_data = np.log10(np.maximum(plot_data, np.min(pos)))
        cbar_label = "log10(" + cbar_label + ")"

    fig, ax = plt.subplots(figsize=(7.0, 5.6))

    # pcolormesh uses axis vectors directly (more faithful than imshow extent)
    # shading="auto" handles dimensions correctly.
    im = ax.pcolormesh(PVp, Zp, plot_data, shading="auto", vmin=args.vmin, vmax=args.vmax)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(cbar_label)

    ax.set_xlabel("PV")
    ax.set_ylabel("Z")
    ax.set_title(f"FGM table: thermo/{args.field}.csv ({'full' if args.full else 'interior, unpadded'})")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"Wrote {out}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
