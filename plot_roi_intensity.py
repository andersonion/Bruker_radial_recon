#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# ----------------------------- helpers: filenames ----------------------------- #

def _strip_nii_extensions(p: Path) -> str:
    name = p.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return p.stem


def find_method_sidecar(img_path: Path) -> Path | None:
    base = _strip_nii_extensions(img_path)
    cand = img_path.parent / f"{base}.method"
    return cand if cand.exists() else None


# ----------------------------- helpers: method parsing ----------------------------- #

def _parse_method_value(method_text: str, key: str, method_path: Path) -> float:
    m = re.search(rf"^{re.escape(key)}\s*=\s*(.*)$", method_text, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"Could not find {key} in {method_path}")
    val = m.group(1).strip()

    if val.startswith("("):
        after = method_text[m.end():]
        m2 = re.search(r"([-+]?\d+(\.\d+)?)", after)
        if not m2:
            raise ValueError(f"Found {key} but could not parse numeric value in {method_path}")
        return float(m2.group(1))

    m3 = re.search(r"([-+]?\d+(\.\d+)?)", val)
    if not m3:
        raise ValueError(f"Found {key} but could not parse numeric value on the same line in {method_path}")
    return float(m3.group(1))


def parse_bruker_method_for_tr_seconds(method_path: Path) -> float:
    text = method_path.read_text(errors="ignore")
    scan_ms = _parse_method_value(text, "##$PVM_ScanTime", method_path)
    nrep = _parse_method_value(text, "##$PVM_NRepetitions", method_path)
    if nrep <= 0:
        raise ValueError(f"Invalid ##$PVM_NRepetitions={nrep} in {method_path}")
    tr_sec = (scan_ms / 1000.0) / nrep
    if tr_sec <= 0:
        raise ValueError(f"Computed TR <= 0 from {method_path}: TR={tr_sec}")
    return tr_sec


def parse_bruker_method_for_npro(method_path: Path) -> float:
    text = method_path.read_text(errors="ignore")
    npro = _parse_method_value(text, "##$NPro", method_path)
    if npro <= 0:
        raise ValueError(f"Invalid ##$NPro={npro} in {method_path}")
    return npro


def get_tr_seconds(img_path: Path, tr_arg: float | None) -> float:
    if tr_arg is not None:
        if tr_arg <= 0:
            raise ValueError(f"TR must be > 0, got {tr_arg}")
        return tr_arg
    method_path = find_method_sidecar(img_path)
    if method_path is None:
        raise FileNotFoundError(
            f"No TR provided and no method sidecar found for {img_path}. "
            f"Expected {_strip_nii_extensions(img_path)}.method next to it, or pass --tr*."
        )
    return parse_bruker_method_for_tr_seconds(method_path)


def get_npro(img_path: Path) -> float:
    method_path = find_method_sidecar(img_path)
    if method_path is None:
        raise FileNotFoundError(
            f"Projection normalization requires method sidecars. "
            f"Could not find {_strip_nii_extensions(img_path)}.method next to {img_path}."
        )
    return parse_bruker_method_for_npro(method_path)


# ----------------------------- helpers: ROI & stats ----------------------------- #

def make_spherical_mask(shape_xyz, radius_vox: float, center_xyz: tuple[float, float, float]):
    nx, ny, nz = shape_xyz
    cx, cy, cz = center_xyz
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    dist2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    return (dist2 <= radius_vox ** 2).astype(np.uint8)


def mean_roi_timeseries(data_4d: np.ndarray, mask_3d: np.ndarray) -> np.ndarray:
    if data_4d.ndim != 4:
        raise ValueError(f"Expected 4D data, got shape {data_4d.shape}")
    if mask_3d.shape != data_4d.shape[:3]:
        raise ValueError(f"Mask shape {mask_3d.shape} != data spatial shape {data_4d.shape[:3]}")
    if mask_3d.sum() == 0:
        raise RuntimeError("ROI mask is empty.")
    roi = data_4d[mask_3d == 1, :]
    return roi.mean(axis=0)


def fit_line(t: np.ndarray, y: np.ndarray):
    A = np.vstack([t, np.ones_like(t)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def affine_match_img1_tail_to_img2_head(t1, y1, t2, y2, n: int):
    """
    Compute alpha,beta so that the fitted line to img2 head matches
    the extrapolated tail line of img1 over img2 head times.
    """
    if n < 2:
        raise ValueError("--norm-n must be >= 2")
    if len(y1) < n or len(y2) < n:
        raise ValueError(f"Not enough points for n={n}: len(y1)={len(y1)}, len(y2)={len(y2)}")

    t1_tail = t1[-n:]
    y1_tail = y1[-n:]
    t2_head = t2[:n]
    y2_head = y2[:n]

    a1, b1 = fit_line(t1_tail, y1_tail)
    a2, b2 = fit_line(t2_head, y2_head)

    L1 = a1 * t2_head + b1
    L2 = a2 * t2_head + b2

    M = np.vstack([L2, np.ones_like(L2)]).T
    alpha, beta = np.linalg.lstsq(M, L1, rcond=None)[0]
    alpha = float(alpha)
    beta = float(beta)

    dbg = {
        "t1_tail": t1_tail,
        "y1_fit_tail": a1 * t1_tail + b1,
        "t2_head": t2_head,
        "L1_target": L1,
        "L2_pre": L2,
        "L2_post": alpha * L2 + beta,
        "alpha": alpha,
        "beta": beta,
    }
    return alpha, beta, dbg


# ----------------------------------- main ----------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Plot mean ROI intensity from one 4D NIfTI (QA) or stitch runs with baseline-assisted scaling."
    )
    parser.add_argument("img_1", help="4D NIfTI (main run)")
    parser.add_argument("img_2", nargs="?", default=None, help="Second 4D NIfTI (optional)")

    parser.add_argument("--img2-baseline", default=None,
                        help="Baseline 4D NIfTI acquired with same params as img_2 (and before img_1). "
                             "Used ONLY to compute a stable mean scale (img2-space -> img1-space).")

    parser.add_argument("--radius", type=float, required=True, help="ROI radius in voxels")
    parser.add_argument("--center", nargs=3, type=float, default=None,
                        metavar=("X", "Y", "Z"),
                        help="ROI center in voxel indices (0-based). If omitted, uses image center.")

    parser.add_argument("--tr1", type=float, default=None, help="TR seconds for img_1 (optional if method exists)")
    parser.add_argument("--tr2", type=float, default=None, help="TR seconds for img_2 (optional if method exists)")
    parser.add_argument("--trb", type=float, default=None, help="TR seconds for --img2-baseline (optional if method exists)")

    parser.add_argument("--stable-n", type=int, default=8,
                        help="Number of initial points of img_1 assumed stable for baseline scaling (default 8).")

    parser.add_argument("--norm-method", choices=["none", "projections", "slope", "both"], default="none",
                        help="Normalization applied at the img1->img2 boundary (after baseline scaling, if any).")
    parser.add_argument("--norm-n", type=int, default=5, help="Window size for slope/both at img1 tail / img2 head.")

    parser.add_argument("--time-unit", choices=["s", "min"], default="s",
                        help="X axis units for plotting (and CSV if written). Default seconds.")

    parser.add_argument("--mask-out", default=None, help="Output ROI mask NIfTI (default auto-named)")
    parser.add_argument("--out", default=None, help="Output plot PNG (default auto-named)")
    parser.add_argument("--csv-out", default=None, help="Optional CSV output")

    args = parser.parse_args()

    img1_path = Path(args.img_1)
    img1 = nib.load(str(img1_path))
    data1 = img1.get_fdata()
    if data1.ndim != 4:
        raise ValueError(f"img_1 must be 4D, got shape {data1.shape}")

    nx, ny, nz = data1.shape[:3]

    # Center
    if args.center is None:
        cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
        center_label = "center"
    else:
        cx, cy, cz = args.center
        if not (0 <= cx < nx and 0 <= cy < ny and 0 <= cz < nz):
            raise ValueError(f"Center {args.center} out of bounds: [0..{nx-1}]x[0..{ny-1}]x[0..{nz-1}]")
        center_label = f"{int(round(cx))}_{int(round(cy))}_{int(round(cz))}"

    roi_tag = f"xyz_{center_label}_r{args.radius:g}"
    out_plot = args.out or f"roi_intensity_{roi_tag}.png"
    mask_out = args.mask_out or f"roi_mask_{roi_tag}.nii.gz"

    mask = make_spherical_mask((nx, ny, nz), radius_vox=args.radius, center_xyz=(cx, cy, cz))
    nib.save(nib.Nifti1Image(mask, img1.affine), mask_out)

    y1 = mean_roi_timeseries(data1, mask)
    tr1 = get_tr_seconds(img1_path, args.tr1)
    t1 = np.arange(len(y1), dtype=float) * tr1

    x_scale = 60.0 if args.time_unit == "min" else 1.0
    x_label = "Time (min)" if args.time_unit == "min" else "Time (s)"

    # Load img2 if present
    y2_raw = None
    tr2 = None
    img2_path = Path(args.img_2) if args.img_2 else None
    if img2_path is not None:
        img2 = nib.load(str(img2_path))
        data2 = img2.get_fdata()
        if data2.ndim != 4:
            raise ValueError(f"img_2 must be 4D, got shape {data2.shape}")
        if data2.shape[:3] != data1.shape[:3]:
            raise ValueError(f"Spatial shapes differ: img_1 {data1.shape[:3]} vs img_2 {data2.shape[:3]}")
        y2_raw = mean_roi_timeseries(data2, mask)
        tr2 = get_tr_seconds(img2_path, args.tr2)

    # Load baseline if provided
    baseline_scale = 1.0
    yb_scaled = None
    baseline_path = Path(args.img2_baseline) if args.img2_baseline else None
    if baseline_path is not None:
        bimg = nib.load(str(baseline_path))
        bdata = bimg.get_fdata()
        if bdata.ndim != 4:
            raise ValueError(f"--img2-baseline must be 4D, got shape {bdata.shape}")
        if bdata.shape[:3] != data1.shape[:3]:
            raise ValueError(f"--img2-baseline spatial shape differs: {bdata.shape[:3]} vs {data1.shape[:3]}")
        yb = mean_roi_timeseries(bdata, mask)
        _ = get_tr_seconds(baseline_path, args.trb)  # just for validation/printing

        if len(yb) < 1:
            raise ValueError("Baseline has no timepoints?")
        if args.stable_n < 1 or args.stable_n > len(y1):
            raise ValueError(f"--stable-n must be in [1..len(img1)] = [1..{len(y1)}], got {args.stable_n}")

        mu_b = float(np.mean(yb))
        mu_1 = float(np.mean(y1[:args.stable_n]))
        if abs(mu_b) < 1e-12:
            raise ValueError("Baseline mean is ~0; cannot compute baseline scale.")
        baseline_scale = mu_1 / mu_b

        yb_scaled = yb * baseline_scale

    # SINGLE IMAGE MODE (img1 only) — still allowed
    if img2_path is None:
        t_plot = t1 / x_scale
        plt.figure()
        plt.plot(t_plot, y1, label="img1")
        if yb_scaled is not None:
            # put baseline before img1 for visualization
            trb = get_tr_seconds(baseline_path, args.trb)
            tb = np.arange(len(yb_scaled), dtype=float) * trb
            t1_offset = tb[-1] + trb
            plt.plot(tb / x_scale, yb_scaled, label=f"img2_baseline × {baseline_scale:.6g}")
            plt.plot((t1_offset + t1) / x_scale, y1, label="img1 (shifted)")
            plt.axvline(t1_offset / x_scale, linestyle="--")
        plt.xlabel(x_label)
        plt.ylabel("Mean ROI intensity")
        plt.title(f"ROI mean intensity (r={args.radius:g}, center={center_label})")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_plot, dpi=150)

        print("[mode] single-image (plus optional baseline scaling preview)")
        if baseline_path is not None:
            print(f"[baseline-scale] scale(img2-space->img1)={baseline_scale:.6g} using mean(baseline) and mean(img1[:{args.stable_n}])")
        print(f"[out] mask: {mask_out}")
        print(f"[out] plot: {out_plot}")
        return

    # TWO IMAGE MODE: apply baseline scale to img2-space runs (baseline and img2)
    y2 = y2_raw.copy()
    if baseline_path is not None:
        y2 *= baseline_scale

    # Build absolute timeline: if baseline exists, plot it before img1
    segments_t = []
    segments_y = []
    labels = []

    t_cursor = 0.0

    if baseline_path is not None:
        trb = get_tr_seconds(baseline_path, args.trb)
        tb = t_cursor + np.arange(len(yb_scaled), dtype=float) * trb
        segments_t.append(tb)
        segments_y.append(yb_scaled)
        labels.append("baseline(scaled)")
        t_cursor = tb[-1] + trb

    t1_abs = t_cursor + t1
    segments_t.append(t1_abs)
    segments_y.append(y1)
    labels.append("img1")
    t_cursor = t1_abs[-1] + tr1

    t2_abs = t_cursor + np.arange(len(y2), dtype=float) * tr2
    labels.append("img2(scaled)" if baseline_path is not None else "img2")
    segments_t.append(t2_abs)
    segments_y.append(y2)

    # Boundary normalization (img1 -> img2) after baseline scale
    scale_proj = 1.0
    alpha = 1.0
    beta = 0.0
    dbg = None

    if args.norm_method in ("projections", "both"):
        scale_proj = float(get_npro(img1_path) / get_npro(img2_path))
        segments_y[-1] = segments_y[-1] * scale_proj  # img2 only
        y2 = segments_y[-1]

    if args.norm_method in ("slope", "both"):
        # Use img1 tail (absolute time) and img2 head (absolute time)
        alpha, beta, dbg = affine_match_img1_tail_to_img2_head(t1_abs, y1, t2_abs, y2, n=args.norm_n)
        segments_y[-1] = alpha * segments_y[-1] + beta
        y2 = segments_y[-1]

    # Concatenate final
    t_all = np.concatenate(segments_t, axis=0)
    y_all = np.concatenate(segments_y, axis=0)

    # Plot
    plt.figure()
    plt.plot(t_all / x_scale, y_all)

    # Mark boundaries
    if baseline_path is not None:
        plt.axvline((segments_t[1][0]) / x_scale, linestyle="--")  # baseline -> img1
    plt.axvline((t2_abs[0]) / x_scale, linestyle="--")            # img1 -> img2

    # Debug overlays for slope/both at img1->img2
    if args.norm_method in ("slope", "both") and dbg is not None:
        plt.plot(dbg["t1_tail"] / x_scale, dbg["y1_fit_tail"], linestyle="--")
        plt.plot(dbg["t2_head"] / x_scale, dbg["L1_target"], linestyle="--")
        plt.plot(dbg["t2_head"] / x_scale, dbg["L2_pre"], linestyle=":")
        plt.plot(dbg["t2_head"] / x_scale, dbg["L2_post"], linestyle=":")
        plt.plot(dbg["t2_head"] / x_scale, y2[:args.norm_n], "o:")

    plt.xlabel(x_label)
    plt.ylabel("Mean ROI intensity")

    title = f"ROI mean intensity (r={args.radius:g}, center={center_label})"
    if baseline_path is not None:
        title += f"\nbaseline scale={baseline_scale:.6g} using mean(baseline) and mean(img1[:{args.stable_n}])"
    if args.norm_method == "projections":
        title += f"\nimg2 scaled by {scale_proj:.6g} (projections)"
    elif args.norm_method == "slope":
        title += f"\nimg2 affine: y2={alpha:.6g}*y2+{beta:.6g} (slope, n={args.norm_n})"
    elif args.norm_method == "both":
        title += f"\nimg2 proj={scale_proj:.6g}, then affine y2={alpha:.6g}*y2+{beta:.6g} (both, n={args.norm_n})"

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)

    # Optional CSV
    if args.csv_out:
        with open(args.csv_out, "w") as f:
            f.write(f"time_{args.time_unit},mean_intensity\n")
            for ti, yi in zip(t_all / x_scale, y_all):
                f.write(f"{ti:.9g},{yi:.9g}\n")

    # Print summary
    print("[mode] baseline->img1->img2" if baseline_path is not None else "[mode] img1->img2")
    print(f"       segments: {labels}")
    if baseline_path is not None:
        print(f"[baseline-scale] scale(img2-space->img1)={baseline_scale:.6g} using mean(baseline) and mean(img1[:{args.stable_n}])")
    print(f"[norm] boundary method={args.norm_method}")
    if args.norm_method in ("projections", "both"):
        print(f"       projections scale={scale_proj:.6g}")
    if args.norm_method in ("slope", "both"):
        print(f"       affine alpha={alpha:.6g}, beta={beta:.6g} (norm-n={args.norm_n})")
    print(f"[out] mask: {mask_out}")
    print(f"[out] plot: {out_plot}")
    if args.csv_out:
        print(f"[out] csv : {args.csv_out}")


if __name__ == "__main__":
    main()
