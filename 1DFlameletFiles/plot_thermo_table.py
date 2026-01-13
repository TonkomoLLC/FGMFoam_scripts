#!/usr/bin/env python3
import argparse
import io
import tarfile

import numpy as np
import matplotlib.pyplot as plt


def read_csv_from_tar(tar_path: str, member: str) -> np.ndarray:
    with tarfile.open(tar_path, mode="r:*") as tf:
        m = tf.getmember(member)
        f = tf.extractfile(m)
        if f is None:
            raise FileNotFoundError(f"Could not extract {member} from {tar_path}")
        data = f.read()
    return np.loadtxt(io.BytesIO(data), delimiter=",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tar", help="Path to 02_tables.tar.xz")
    ap.add_argument("--field", default="T", choices=["T", "rho", "SourcePV"],
                    help="Thermo field to plot from 02_tables/thermo/<field>.csv")
    ap.add_argument("--out", default=None, help="Output PNG filename (default: <field>_table.png)")
    ap.add_argument("--no-show", action="store_true", help="Do not display interactively")
    ap.add_argument("--full", action="store_true", help="Plot full table including padded rows/cols")
    ap.add_argument("--log10-abs", action="store_true",
                    help="Plot log10(abs(field)+eps). Useful for SourcePV.")
    ap.add_argument("--eps", type=float, default=1e-30,
                    help="Epsilon for --log10-abs (default 1e-30)")
    args = ap.parse_args()

    out = args.out or f"{args.field}_table.png"

    Z = read_csv_from_tar(args.tar, "02_tables/axes/Z.csv").reshape(-1)
    PV = read_csv_from_tar(args.tar, "02_tables/axes/PV.csv").reshape(-1)
    F = read_csv_from_tar(args.tar, f"02_tables/thermo/{args.field}.csv")

    if not args.full:
        Zp = Z[1:-1]
        PVp = PV[1:-1]
        Fp = F[1:-1, 1:-1]
        subtitle = "interior, unpadded"
    else:
        Zp, PVp, Fp = Z, PV, F
        subtitle = "full (padded)"

    plot_data = Fp
    cbar_label = args.field

    if args.log10_abs:
        plot_data = np.log10(np.abs(plot_data) + float(args.eps))
        cbar_label = f"log10(abs({args.field}) + eps)"

    fig = plt.figure(figsize=(7.0, 5.6))
    ax = plt.gca()

    im = ax.imshow(
        plot_data,
        origin="lower",
        aspect="auto",
        extent=[PVp[0], PVp[-1], Zp[0], Zp[-1]],
    )
    cb = plt.colorbar(im, ax=ax)
    cb.set_label(cbar_label)

    ax.set_xlabel("PV")
    ax.set_ylabel("Z")
    ax.set_title(f"FGM table: thermo/{args.field}.csv ({subtitle})")

    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"Wrote {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

