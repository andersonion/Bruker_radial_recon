#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

try:
    import scipy.ndimage as ndi
    from scipy.optimize import minimize
except Exception:
    ndi = None


# ----------------------------- utilities ----------------------------- #

def fit_line(t, y):
    A = np.vstack([t, np.ones_like(t)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m), float(b)


def mean_window(y, n, head=True):
    return float(np.mean(y[:n] if head else y[-n:]))


def slope_window(t, y, n, head=True):
    if head:
        tseg, yseg = t[:n], y[:n]
    else:
        tseg, yseg = t[-n:], y[-n:]
    m, _ = fit_line(tseg, yseg)
    return m


# ----------------------------- main ----------------------------- #

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("img1")
    parser.add_argument("img2", nargs="?")
    parser.add_argument("--img2-baseline")

    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--center", nargs=3, type=float)

    parser.add_argument("--norm-method",
                        choices=["none", "affine2boundline"],
                        default="none")

    parser.add_argument("--stable-n", type=int, default=8)
    parser.add_argument("--norm-n", type=int, default=5)

    parser.add_argument("--affine-source",
                        choices=["roi", "global", "nonzero"],
                        default="roi")

    parser.add_argument("--line-weight-left", type=float, default=0.1)
    parser.add_argument("--line-weight-right", type=float, default=0.1)

    parser.add_argument("--a-bound-scale", type=float, default=2.0)

    args = parser.parse_args()

    # ---------------- Load images ---------------- #

    img1 = nib.load(args.img1)
    data1 = img1.get_fdata()

    img2 = nib.load(args.img2) if args.img2 else None
    data2 = img2.get_fdata() if img2 else None

    baseline = nib.load(args.img2_baseline) if args.img2_baseline else None
    datab = baseline.get_fdata() if baseline else None

    nx, ny, nz = data1.shape[:3]

    # ROI mask
    if args.center:
        cx, cy, cz = args.center
    else:
        cx, cy, cz = (nx-1)/2, (ny-1)/2, (nz-1)/2

    X, Y, Z = np.meshgrid(np.arange(nx),
                          np.arange(ny),
                          np.arange(nz),
                          indexing="ij")

    mask = ((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2) <= args.radius**2

    def roi_ts(data):
        return np.mean(data[mask], axis=0)

    y1_roi = roi_ts(data1)
    y2_roi = roi_ts(data2) if data2 is not None else None
    yb_roi = roi_ts(datab) if datab is not None else None

    t1 = np.arange(len(y1_roi))
    t2 = np.arange(len(y2_roi)) if y2_roi is not None else None

    # ---------------- Affine2BoundLine ---------------- #

    if args.norm_method == "affine2boundline":

        # Means
        mu1L = mean_window(y1_roi, args.stable_n, head=True)
        mu1R = mean_window(y1_roi, args.norm_n, head=False)

        muBL = float(np.mean(yb_roi)) if yb_roi is not None else mu1L
        mu2R = mean_window(y2_roi, args.norm_n, head=True)

        # Slopes
        m1L = slope_window(t1, y1_roi, args.stable_n, head=True)
        m1R = slope_window(t1, y1_roi, args.norm_n, head=False)

        mBL = 0.0
        if yb_roi is not None:
            tb = np.arange(len(yb_roi))
            mBL = slope_window(tb, yb_roi, len(yb_roi), head=True)

        m2R = slope_window(t2, y2_roi, args.norm_n, head=True)

        # Expected gain from NPro
        def get_npro(path):
            method = Path(path).with_suffix(".method")
            if not method.exists():
                return None
            txt = method.read_text(errors="ignore")
            m = re.search(r"##\$NPro\s*=\s*(\d+)", txt)
            return float(m.group(1)) if m else None

        n1 = get_npro(args.img1)
        n2 = get_npro(args.img2) if args.img2 else None

        a_expected = (n2 / n1) if (n1 and n2) else 1.0

        a_min = a_expected / args.a_bound_scale
        a_max = a_expected * args.a_bound_scale

        # Objective
        def objective(x):
            a, c = x

            valL = (a*mu1L + c - muBL)**2
            valR = (a*mu1R + c - mu2R)**2

            slopeL = (a*m1L - mBL)**2
            slopeR = (a*m1R - m2R)**2

            return (
                valL
                + valR
                + args.line_weight_left * slopeL
                + args.line_weight_right * slopeR
            )

        bounds = [(a_min, a_max), (None, None)]

        res = minimize(objective,
                       x0=[a_expected, 0.0],
                       bounds=bounds)

        a_opt, c_opt = res.x

        y1_roi = a_opt * y1_roi + c_opt

        print("[affine2boundline]")
        print(f"  a_expected={a_expected:.6g}")
        print(f"  a_bounds=[{a_min:.6g}, {a_max:.6g}]")
        print(f"  a_opt={a_opt:.6g}")
        print(f"  c_opt={c_opt:.6g}")

    # ---------------- Plot ---------------- #

    plt.plot(y1_roi, label="img1")
    if yb_roi is not None:
        plt.plot(yb_roi, label="baseline")
    if y2_roi is not None:
        plt.plot(y2_roi, label="img2")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()