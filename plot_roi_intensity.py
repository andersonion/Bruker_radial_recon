#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Optional, but typically available; used for morphology + connected components
try:
    import scipy.ndimage as ndi
except Exception:
    ndi = None


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


# ----------------------------- ROI + global signals ----------------------------- #

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
    return np.nanmean(roi, axis=0)


def global_timeseries(data_4d: np.ndarray, global_mask_3d: np.ndarray | None, mode: str) -> np.ndarray:
    if data_4d.ndim != 4:
        raise ValueError(f"Expected 4D data, got shape {data_4d.shape}")

    if global_mask_3d is not None:
        if global_mask_3d.shape != data_4d.shape[:3]:
            raise ValueError(f"Global mask shape {global_mask_3d.shape} != data spatial shape {data_4d.shape[:3]}")
        vox = data_4d[global_mask_3d > 0, :]
    else:
        vox = data_4d.reshape(-1, data_4d.shape[3])

    if vox.size == 0:
        raise RuntimeError("Global mask selected 0 voxels.")

    if mode == "mean":
        return np.nanmean(vox, axis=0)
    if mode == "median":
        return np.nanmedian(vox, axis=0)

    raise ValueError(f"Unknown global mode: {mode}")


# ----------------------------- auto brain mask (global scaling) ----------------------------- #

def otsu_threshold(vol: np.ndarray, nbins: int = 256) -> float:
    """
    Otsu threshold on a 3D volume (ignores NaNs, flattens).
    Returns threshold in the same intensity units as vol.
    """
    x = vol[np.isfinite(vol)].ravel()
    if x.size == 0:
        raise ValueError("Otsu: volume has no finite values.")
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        # degenerate
        return x_min

    hist, bin_edges = np.histogram(x, bins=nbins, range=(x_min, x_max))
    hist = hist.astype(np.float64)

    prob = hist / np.sum(hist)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (bin_edges[:-1] + bin_edges[1:]) * 0.5)
    mu_t = mu[-1]

    # between-class variance
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom

    idx = int(np.nanargmax(sigma_b2))
    thr = float((bin_edges[idx] + bin_edges[idx + 1]) * 0.5)
    return thr


def _require_scipy():
    if ndi is None:
        raise RuntimeError(
            "This script needs scipy for the requested morphology/CC steps. "
            "On your cluster/venv, install scipy or run in an environment that has it."
        )


def generate_auto_brain_mask_from_mean(data_4d: np.ndarray, *, erosions: int = 3, dilations: int = 3,
                                      nbins: int = 256) -> np.ndarray:
    """
    Build a quick brain-ish mask from the time-mean volume:
      - Otsu threshold
      - binary erosion (erosions times)
      - keep largest connected component
      - binary dilation (dilations times)
    Returns uint8 mask (0/1).
    """
    _require_scipy()

    mean_vol = np.nanmean(data_4d, axis=3)
    thr = otsu_threshold(mean_vol, nbins=nbins)
    m = mean_vol > thr

    # Use 3D 26-connected struct
    struct = ndi.generate_binary_structure(3, 2)

    if erosions > 0:
        m = ndi.binary_erosion(m, structure=struct, iterations=erosions)

    # Keep largest blob
    lbl, n = ndi.label(m, structure=struct)
    if n < 1:
        # Fallback: if erosion killed everything, relax by skipping erosion
        m = mean_vol > thr
        lbl, n = ndi.label(m, structure=struct)
        if n < 1:
            raise RuntimeError("Auto mask failed: no components found after thresholding.")

    counts = np.bincount(lbl.ravel())
    counts[0] = 0  # ignore background
    keep = int(np.argmax(counts))
    m = (lbl == keep)

    if dilations > 0:
        m = ndi.binary_dilation(m, structure=struct, iterations=dilations)

    return m.astype(np.uint8)


# ----------------------------- boundary normalization (ROI) ----------------------------- #

def fit_line(t: np.ndarray, y: np.ndarray):
    A = np.vstack([t, np.ones_like(t)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def affine_match_img1_tail_to_img2_head(t1, y1, t2, y2, n: int):
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
        description="Plot ROI intensity while computing baseline scaling from GLOBAL signal. "
                    "Global mask can be provided, or auto-generated from mean(img2)."
    )
    parser.add_argument("img_1", help="4D NIfTI (main run)")
    parser.add_argument("img_2", nargs="?", default=None, help="Second 4D NIfTI (optional)")

    parser.add_argument("--img2-baseline", default=None,
                        help="Baseline 4D NIfTI acquired with same params as img_2 (and before img_1). "
                             "Used ONLY to compute stable scaling (img2-space -> img1-space) using GLOBAL signal.")

    parser.add_argument("--radius", type=float, required=True, help="ROI radius in voxels")
    parser.add_argument("--center", nargs=3, type=float, default=None,
                        metavar=("X", "Y", "Z"),
                        help="ROI center in voxel indices (0-based). If omitted, uses image center.")

    parser.add_argument("--tr1", type=float, default=None, help="TR seconds for img_1 (optional if method exists)")
    parser.add_argument("--tr2", type=float, default=None, help="TR seconds for img_2 (optional if method exists)")
    parser.add_argument("--trb", type=float, default=None, help="TR seconds for --img2-baseline (optional if method exists)")

    parser.add_argument("--stable-n", type=int, default=8,
                        help="Number of initial points of img_1 assumed stable for baseline scaling (default 8).")
    parser.add_argument(
        "--title-tag",
        default=None,
        help="Optional custom tag to include in the plot title (e.g., subject/run/sequence label).",
    )
    parser.add_argument(
        "--title-mode",
        choices=["full", "compact"],
        default="full",
        help="Title style. 'full' includes verbose normalization/global details. "
             "'compact' shows only --title-tag (or default) and a small second line '(x,y,z;r=R)'.",
    )

    # Global signal controls
    parser.add_argument("--global-mode", choices=["mean", "median"], default="mean",
                        help="How to compute global signal per timepoint (default mean).")

    # global mask can be a file path OR literal 'auto'
    parser.add_argument("--global-mask", default=None,
                        help="3D mask NIfTI for global signal, OR 'auto' to generate from mean(img2). "
                             "If omitted, uses ALL voxels (usually not recommended).")

    # auto-mask params
    parser.add_argument("--auto-mask-source", choices=["img2", "img2_baseline", "img1"], default="img2",
                        help="If --global-mask auto, which image to build it from (default img2).")
    parser.add_argument("--auto-mask-erosions", type=int, default=3, help="Erosion iterations for auto mask (default 3)")
    parser.add_argument("--auto-mask-dilations", type=int, default=3, help="Dilation iterations for auto mask (default 3)")
    parser.add_argument("--auto-mask-hist-bins", type=int, default=256, help="Histogram bins for Otsu (default 256)")
    parser.add_argument("--auto-mask-out", default=None,
                        help="Where to write the auto-generated global mask (default auto name). "
                             "Set to 'none' to skip writing.")

    # Boundary normalization (ROI-based) at img1 -> img2
    parser.add_argument(
        "--norm-method",
        choices=["none", "lastfirst", "projections", "slope", "both"],
        default="none",
        help="Normalization at img1->img2 boundary (ROI-based), after any baseline scaling. "
             "'lastfirst' can also use baseline (if provided) by averaging scale implied by "
             "baseline->img1 and img1->img2 joins.",
    )
    parser.add_argument("--norm-n", type=int, default=5, help="Window size for slope/both at img1 tail / img2 head.")

    parser.add_argument("--time-unit", choices=["s", "min"], default="s",
                        help="X axis units for plotting (and CSV if written). Default seconds.")

    parser.add_argument("--mask-out", default=None, help="Output ROI mask NIfTI (default auto-named)")
    parser.add_argument("--out", default=None, help="Output plot PNG (default auto-named)")
    parser.add_argument("--csv-out", default=None, help="Optional CSV output")

    args = parser.parse_args()


    # ---------------- Filename prefixing from title tag ----------------
    def _slugify(s: str) -> str:
        # Safe filename prefix: keep alnum, dash, underscore; collapse others to underscore
        out = []
        for ch in s.strip():
            if ch.isalnum() or ch in ("-", "_"):
                out.append(ch)
            else:
                out.append("_")
        slug = "".join(out)
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug.strip("_")

    file_prefix = _slugify(args.title_tag.strip()) + "_" if (args.title_tag is not None and args.title_tag.strip() != "") else ""
    _prefix = file_prefix
    
    def _prefix_path(p: str | None) -> str | None:
        if p is None:
            return None
        pp = Path(p)
        if pp.name.startswith(file_prefix) or file_prefix == "":
            return str(pp)
        return str(pp.with_name(file_prefix + pp.name))



    # Load img1
    img1_path = Path(args.img_1)
    img1 = nib.load(str(img1_path))
    data1 = img1.get_fdata()
    if data1.ndim != 4:
        raise ValueError(f"img_1 must be 4D, got shape {data1.shape}")
    nx, ny, nz = data1.shape[:3]

    # Load img2 (optional)
    img2_path = Path(args.img_2) if args.img_2 else None
    data2 = None
    if img2_path is not None:
        img2 = nib.load(str(img2_path))
        data2 = img2.get_fdata()
        if data2.ndim != 4:
            raise ValueError(f"img_2 must be 4D, got shape {data2.shape}")
        if data2.shape[:3] != (nx, ny, nz):
            raise ValueError(f"Spatial shapes differ: img_2 {data2.shape[:3]} vs img_1 {(nx, ny, nz)}")

    # Load baseline (optional) - allow 3D or 4D
    baseline_path = Path(args.img2_baseline) if args.img2_baseline else None
    datab = None
    bimg = None
    baseline_is_3d = False

    if baseline_path is not None:
        bimg = nib.load(str(baseline_path))
        datab = bimg.get_fdata()

        if datab.ndim == 3:
            # Promote 3D baseline to single-timepoint 4D
            datab = datab[..., np.newaxis]
            baseline_is_3d = True
        elif datab.ndim != 4:
            raise ValueError(f"--img2-baseline must be 3D or 4D, got shape {datab.shape}")

        if datab.shape[:3] != (nx, ny, nz):
            raise ValueError(
                f"--img2-baseline spatial shape differs: {datab.shape[:3]} vs {(nx, ny, nz)}"
            )


    # ROI center
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

    out_plot = _prefix_path(out_plot)
    mask_out = _prefix_path(mask_out)
    if args.csv_out is not None:
        args.csv_out = _prefix_path(args.csv_out)
    if args.auto_mask_out is not None and str(args.auto_mask_out).lower() not in ("none",):
        args.auto_mask_out = _prefix_path(args.auto_mask_out)


    # ROI mask
    roi_mask = make_spherical_mask((nx, ny, nz), radius_vox=args.radius, center_xyz=(cx, cy, cz))
    nib.save(nib.Nifti1Image(roi_mask, img1.affine), mask_out)

    # Decide / build global mask
    global_mask = None
    global_mask_note = "OFF"

    if args.global_mask is not None:
        if args.global_mask.lower() == "auto":
            # Choose source volume for auto mask generation
            if args.auto_mask_source == "img2":
                if data2 is None:
                    raise ValueError("--global-mask auto with --auto-mask-source img2 requires img_2 to be provided.")
                src_data = data2
                src_aff = img2.affine
                src_name = "img2"
                src_path = img2_path
            elif args.auto_mask_source == "img2_baseline":
                if datab is None:
                    raise ValueError("--global-mask auto with --auto-mask-source img2_baseline requires --img2-baseline.")
                src_data = datab
                src_aff = bimg.affine
                src_name = "img2_baseline"
                src_path = baseline_path
            else:
                src_data = data1
                src_aff = img1.affine
                src_name = "img1"
                src_path = img1_path

            global_mask = generate_auto_brain_mask_from_mean(
                src_data,
                erosions=args.auto_mask_erosions,
                dilations=args.auto_mask_dilations,
                nbins=args.auto_mask_hist_bins,
            ).astype(bool)

            global_mask_note = f"AUTO({src_name})"

            # Write it (unless disabled)
            if args.auto_mask_out is None:
                auto_out = src_path.parent / f"{_prefix}auto_global_mask_from_{_strip_nii_extensions(src_path)}.nii.gz"
            elif str(args.auto_mask_out).lower() == "none":
                auto_out = None
            else:
                auto_out = Path(_prefix_path(str(args.auto_mask_out)))
            if auto_out is not None:
                nib.save(nib.Nifti1Image(global_mask.astype(np.uint8), src_aff), str(auto_out))
        else:
            gm_path = Path(args.global_mask)
            gm_img = nib.load(str(gm_path))
            gm = gm_img.get_fdata()
            if gm.ndim != 3:
                raise ValueError(f"--global-mask must be 3D, got shape {gm.shape}")
            if gm.shape != (nx, ny, nz):
                raise ValueError(f"--global-mask shape {gm.shape} != data shape {(nx, ny, nz)}")
            global_mask = (gm > 0)
            global_mask_note = f"FILE({gm_path.name})"

    # Signals for img1
    y1_roi = mean_roi_timeseries(data1, roi_mask)
    y1_glob = global_timeseries(data1, global_mask, args.global_mode)

    tr1 = get_tr_seconds(img1_path, args.tr1)
    t1 = np.arange(len(y1_roi), dtype=float) * tr1

    x_scale = 60.0 if args.time_unit == "min" else 1.0
    x_label = "Time (min)" if args.time_unit == "min" else "Time (s)"

    # ROI for img2 (optional)
    y2_roi = None
    tr2 = None
    if data2 is not None:
        y2_roi = mean_roi_timeseries(data2, roi_mask)
        tr2 = get_tr_seconds(img2_path, args.tr2)

    # ---------------- Scaling: baseline/img2 fixed, scale img1 only ----------------
    img1_scale = 1.0
    scale_lastfirst_left = 1.0
    scale_lastfirst_right = 1.0
    scale_lastfirst_final = 1.0

    # Optional baseline signals (kept UNCHANGED)
    yb_roi = None
    yb_glob = None
    trb = None

    # Optional img2 global (kept UNCHANGED) for right-boundary scaling with global signal
    y2_glob = None

    if datab is not None:
        yb_roi = mean_roi_timeseries(datab, roi_mask)
        yb_glob = global_timeseries(datab, global_mask, args.global_mode)
        trb = get_tr_seconds(baseline_path, args.trb) if not baseline_is_3d else tr1

        if args.stable_n < 1 or args.stable_n > len(y1_glob):
            raise ValueError(f"--stable-n must be in [1..len(img1)] = [1..{len(y1_glob)}], got {args.stable_n}")

        # Left boundary scale (GLOBAL): make early img1 match baseline (baseline is stable)
        mu_b = float(np.mean(yb_glob))
        mu_1 = float(np.mean(y1_glob[:args.stable_n]))
        if abs(mu_1) < 1e-12:
            raise ValueError("Stable img1 global mean is ~0; cannot compute scale.")
        scale_lastfirst_left = mu_b / mu_1

    if data2 is not None:
        y2_glob = global_timeseries(data2, global_mask, args.global_mode)

        # Right boundary scale (GLOBAL): enforce last(img1) == first(img2)
        mu_2_first = float(y2_glob[0])
        mu_1_last = float(y1_glob[-1])
        if abs(mu_1_last) < 1e-12:
            raise ValueError("Last img1 global value is ~0; cannot compute scale.")
        scale_lastfirst_right = mu_2_first / mu_1_last

    # Choose how to scale img1
    if args.norm_method == "lastfirst":
        if datab is not None and data2 is not None:
            # Use BOTH boundaries (baseline->img1 and img1->img2)
            scale_lastfirst_final = 0.5 * (scale_lastfirst_left + scale_lastfirst_right)
            img1_scale = scale_lastfirst_final
        elif datab is not None and data2 is None:
            # Only baseline boundary available
            img1_scale = scale_lastfirst_left
        elif datab is None and data2 is not None:
            # Only img1->img2 boundary available
            img1_scale = scale_lastfirst_right
        else:
            img1_scale = 1.0
    else:
        # For other norm methods, keep your existing behavior (img1 unscaled here)
        img1_scale = 1.0

    # Apply scaling to img1 ONLY (both ROI + global)
    if img1_scale != 1.0:
        y1_roi = y1_roi * img1_scale
        y1_glob = y1_glob * img1_scale
    # If only img1 (and maybe baseline), we can still plot baseline->img1 QA
    segments_t = []
    segments_y = []
    labels = []
    t_cursor = 0.0

    if datab is not None:
        tb = t_cursor + np.arange(len(yb_roi), dtype=float) * trb
        segments_t.append(tb)
        segments_y.append(yb_roi)
        labels.append("baseline_ROI")
        t_cursor = tb[-1] + trb

    t1_abs = t_cursor + t1
    segments_t.append(t1_abs)
    segments_y.append(y1_roi)
    if img1_scale != 1.0:
        labels.append(f"img1_ROI×{img1_scale:.6g}")
    else:
        labels.append("img1_ROI")
    t_cursor = t1_abs[-1] + tr1

    dbg = None
    scale_proj = 1.0
    alpha = 1.0
    beta = 0.0

    if data2 is not None:
        t2_abs = t_cursor + np.arange(len(y2_roi), dtype=float) * tr2
        segments_t.append(t2_abs)
        segments_y.append(y2_roi)
        labels.append("img2_ROI" + ("(scaled)" if datab is not None else ""))

        # Boundary normalization on ROI only (optional)
        scale_lastfirst = 1.0

        if args.norm_method == "lastfirst" and datab is None:
            y1_last = float(y1_roi[-1])
            y2_first = float(segments_y[-1][0])
            if abs(y2_first) < 1e-12:
                raise ValueError("lastfirst normalization: img2 first ROI value is ~0; cannot scale.")
            scale_lastfirst = y1_last / y2_first
            segments_y[-1] = segments_y[-1] * scale_lastfirst
            
        if args.norm_method in ("projections", "both"):
            scale_proj = float(get_npro(img1_path) / get_npro(img2_path))
            segments_y[-1] = segments_y[-1] * scale_proj

        if args.norm_method in ("slope", "both"):
            alpha, beta, dbg = affine_match_img1_tail_to_img2_head(
                t1_abs, y1_roi, t2_abs, segments_y[-1], n=args.norm_n
            )
            segments_y[-1] = alpha * segments_y[-1] + beta

    # Concatenate + plot
    t_all = np.concatenate(segments_t, axis=0)
    y_all = np.concatenate(segments_y, axis=0)

    plt.figure()
    plt.plot(t_all / x_scale, y_all)

    # boundaries
    if datab is not None:
        plt.axvline((t1_abs[0]) / x_scale, linestyle="--")  # baseline->img1
    if data2 is not None:
        plt.axvline((segments_t[-1][0]) / x_scale, linestyle="--")  # img1->img2

    # slope debug overlays (img1->img2)
    if data2 is not None and args.norm_method in ("slope", "both") and dbg is not None:
        # img1 tail fit (reference)
        plt.plot(dbg["t1_tail"] / x_scale, dbg["y1_fit_tail"], linestyle="--", alpha=0.7)

        # img2 post-normalization fit ONLY (for continuity QA)
        plt.plot(dbg["t2_head"] / x_scale, dbg["L2_post"], linestyle="--", alpha=0.7)
        
    plt.xlabel(x_label)
    plt.ylabel("Mean ROI intensity")

    # ---------------- Title handling ----------------
    # Title tag (user override) or default label
    if args.title_tag is not None and args.title_tag.strip() != "":
        title_tag = args.title_tag.strip()
    else:
        title_tag = "ROI intensity"

    # Compact coordinate line requested format: "($x,$y,$z;r=$radius)"
    if args.center is None:
        # If center not specified, we used image center; keep label consistent
        coord_line = f"({center_label};r={args.radius:g})"
    else:
        coord_line = f"({int(round(cx))},{int(round(cy))},{int(round(cz))};r={args.radius:g})"

    ax = plt.gca()

    if args.title_mode == "compact":
        # Main title is just the tag (or default) with coords appended on the same line
        ax.set_title(f"{title_tag} {coord_line}", fontsize=12)

    else:
        # FULL title mode: keep verbose details, but put coords on the FIRST line after the tag
        title = f"{title_tag} {coord_line}\n" + f"ROI mean intensity (r={args.radius:g}, center={center_label})"

        # (keep all your existing title += ... lines below this point)
        if baseline_path is not None:
            title += (
                f"\nimg1 scale (left)={scale_lastfirst_left:.6g} using mean(baseline_global)/mean(img1_global[:{args.stable_n}])"
            )
            if data2 is not None:
                title += f"\nimg1 scale (right)={scale_lastfirst_right:.6g} using img2_global_first/img1_global_last"
            if args.norm_method == "lastfirst" and datab is not None and data2 is not None:
                title += f"\nimg1 scale (final avg)={img1_scale:.6g}"
            title += f"\n(global-mode={args.global_mode}, global-mask={global_mask_note})"
            
        if data2 is not None:
            if args.norm_method == "lastfirst" and datab is not None:
                title += (
                    f"\nlastfirst avg scale={scale_lastfirst_final:.6g} "
                    f"(left={scale_lastfirst_left:.6g}, right={scale_lastfirst_right:.6g})"
                )
            elif args.norm_method == "lastfirst" and datab is None:
                title += f"\nimg2 ROI scaled by {scale_lastfirst_final:.6g} (lastfirst)"
            elif args.norm_method == "projections":
                title += f"\nimg2 ROI scaled by {scale_proj:.6g} (projections)"
            elif args.norm_method == "slope":
                title += f"\nimg2 ROI affine: y2={alpha:.6g}*y2+{beta:.6g} (slope, n={args.norm_n})"
            elif args.norm_method == "both":
                title += f"\nimg2 ROI proj={scale_proj:.6g}, then affine y2={alpha:.6g}*y2+{beta:.6g} (both, n={args.norm_n})"

        ax.set_title(title, fontsize=12)   
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)

    # CSV
    if args.csv_out:
        with open(args.csv_out, "w") as f:
            f.write(f"time_{args.time_unit},roi_mean_intensity\n")
            for ti, yi in zip(t_all / x_scale, y_all):
                f.write(f"{ti:.9g},{yi:.9g}\n")

    # Print useful QA numbers (so you can see if the mismatch is ROI-specific)
    print("[mode] ROI plot with GLOBAL scaling (optional auto mask)")
    print(f"       segments: {labels}")
    print(f"[roi] center={center_label}, radius={args.radius:g}, voxels={int(roi_mask.sum())}")
    print(f"[global] mode={args.global_mode}, mask={global_mask_note}")
    if datab is not None:
        # Global means used for scale
        yb_glob = global_timeseries(datab, global_mask, args.global_mode)
        mu_b = float(np.mean(yb_glob))
        mu_1 = float(np.mean(y1_glob[:args.stable_n]))
        print(f"[img1-scale-left]  mu_baseline_global={mu_b:.6g}  mu_img1_global_stable={mu_1:.6g}  scale={scale_lastfirst_left:.6g}")
        if data2 is not None:
            mu_2_first = float(y2_glob[0])
            mu_1_last = float((y1_glob / img1_scale)[-1]) if img1_scale != 0 else float("nan")
            print(f"[img1-scale-right] img2_global_first={mu_2_first:.6g}  img1_global_last_raw={mu_1_last:.6g}  scale={scale_lastfirst_right:.6g}")
        if args.norm_method == "lastfirst" and datab is not None and data2 is not None:
            print(f"[img1-scale-final] avg(left,right)={img1_scale:.6g}")
            
        # ROI means (helps explain your “ROI baselines nearly match if scaled differently” observation)
        mu_b_roi = float(np.mean(yb_roi))
        mu_1_roi = float(np.mean(y1_roi[:args.stable_n]))
        if abs(mu_b_roi) < 1e-12:
            ratio = float("nan")
        else:
            ratio = mu_1_roi / mu_b_roi
        print(f"[QA ROI means] mean(baseline_ROI)={mu_b_roi:.6g}  mean(img1_ROI_stable_scaled)={mu_1_roi:.6g}  ratio(img1/baseline)={ratio:.6g}")
        
    if data2 is not None:
        print(f"[norm] img1->img2 method={args.norm_method}")
        if args.norm_method in ("projections", "both"):
            print(f"       projections scale={scale_proj:.6g}")
        if args.norm_method in ("slope", "both"):
            print(f"       affine alpha={alpha:.6g}, beta={beta:.6g} (norm-n={args.norm_n})")

    print(f"[out] roi-mask: {mask_out}")
    print(f"[out] plot    : {out_plot}")
    if args.csv_out:
        print(f"[out] csv     : {args.csv_out}")


if __name__ == "__main__":
    main()
