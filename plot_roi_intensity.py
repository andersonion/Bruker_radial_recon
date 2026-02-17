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


def _slugify(s: str) -> str:
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


def _prefix_path(p: str | None, file_prefix: str) -> str | None:
    if p is None:
        return None
    pp = Path(p)
    if file_prefix == "" or pp.name.startswith(file_prefix):
        return str(pp)
    return str(pp.with_name(file_prefix + pp.name))


# ----------------------------- helpers: method parsing ----------------------------- #

def _parse_method_value(method_text: str, key: str, method_path: Path) -> float:
    m = re.search(rf"^{re.escape(key)}\s*=\s*(.*)$", method_text, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"Could not find {key} in {method_path}")
    val = m.group(1).strip()

    # Bruker style: key=( ... ) then values on next line(s)
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
            "Projection normalization requires method sidecars. "
            f"Could not find {_strip_nii_extensions(img_path)}.method next to {img_path}."
        )
    return parse_bruker_method_for_npro(method_path)


# ----------------------------- ROI + signal extraction ----------------------------- #

def make_spherical_mask(shape_xyz, radius_vox: float, center_xyz: tuple[float, float, float]) -> np.ndarray:
    nx, ny, nz = shape_xyz
    cx, cy, cz = center_xyz
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    dist2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    return (dist2 <= radius_vox ** 2).astype(np.uint8)


def _reduce_timeseries(vox_by_t: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return np.nanmean(vox_by_t, axis=0)
    if mode == "median":
        return np.nanmedian(vox_by_t, axis=0)
    raise ValueError(f"Unknown mode: {mode}")


def timeseries_from_source(
    data_4d: np.ndarray,
    *,
    source: str,
    roi_mask_3d: np.ndarray,
    global_mask_3d: np.ndarray | None,
    global_mode: str,
    nonzero_eps: float
) -> np.ndarray:
    if data_4d.ndim != 4:
        raise ValueError(f"Expected 4D data, got shape {data_4d.shape}")

    if source == "roi":
        if roi_mask_3d.shape != data_4d.shape[:3]:
            raise ValueError("ROI mask shape mismatch.")
        vox = data_4d[roi_mask_3d > 0, :]
        if vox.size == 0:
            raise RuntimeError("ROI mask selected 0 voxels.")
        return _reduce_timeseries(vox, global_mode)

    if source == "global":
        vox = data_4d.reshape(-1, data_4d.shape[3])
        if vox.size == 0:
            raise RuntimeError("Global selected 0 voxels.")
        return _reduce_timeseries(vox, global_mode)

    if source == "nonzero":
        mean_vol = np.nanmean(data_4d, axis=3)
        m = np.isfinite(mean_vol) & (mean_vol > nonzero_eps)
        vox = data_4d[m, :]
        if vox.size == 0:
            raise RuntimeError(f"Nonzero mask selected 0 voxels (eps={nonzero_eps}).")
        return _reduce_timeseries(vox, global_mode)

    if source == "mask":
        if global_mask_3d is None:
            raise ValueError("source='mask' requested but global_mask_3d is None.")
        if global_mask_3d.shape != data_4d.shape[:3]:
            raise ValueError("Global mask shape mismatch.")
        vox = data_4d[global_mask_3d > 0, :]
        if vox.size == 0:
            raise RuntimeError("Global mask selected 0 voxels.")
        return _reduce_timeseries(vox, global_mode)

    raise ValueError(f"Unknown source: {source}")


# ----------------------------- auto brain mask (optional) ----------------------------- #

def _require_scipy():
    if ndi is None:
        raise RuntimeError(
            "This script needs scipy for the requested morphology/CC steps. "
            "Install scipy or run in an environment that has it."
        )


def otsu_threshold(vol: np.ndarray, nbins: int = 256) -> float:
    x = vol[np.isfinite(vol)].ravel()
    if x.size == 0:
        raise ValueError("Otsu: volume has no finite values.")
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return x_min

    hist, bin_edges = np.histogram(x, bins=nbins, range=(x_min, x_max))
    hist = hist.astype(np.float64)
    prob = hist / np.sum(hist)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * (bin_edges[:-1] + bin_edges[1:]) * 0.5)
    mu_t = mu[-1]

    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom

    idx = int(np.nanargmax(sigma_b2))
    thr = float((bin_edges[idx] + bin_edges[idx + 1]) * 0.5)
    return thr


def generate_auto_brain_mask_from_mean(
    data_4d: np.ndarray,
    *,
    erosions: int = 3,
    dilations: int = 3,
    nbins: int = 256
) -> np.ndarray:
    _require_scipy()

    mean_vol = np.nanmean(data_4d, axis=3)
    thr = otsu_threshold(mean_vol, nbins=nbins)
    m = mean_vol > thr

    struct = ndi.generate_binary_structure(3, 2)  # 26-connected

    if erosions > 0:
        m = ndi.binary_erosion(m, structure=struct, iterations=erosions)

    lbl, n = ndi.label(m, structure=struct)
    if n < 1:
        # fallback: remove erosions
        m = mean_vol > thr
        lbl, n = ndi.label(m, structure=struct)
        if n < 1:
            raise RuntimeError("Auto mask failed: no components found after thresholding.")

    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    m = (lbl == keep)

    if dilations > 0:
        m = ndi.binary_dilation(m, structure=struct, iterations=dilations)

    return m.astype(np.uint8)


# ----------------------------- boundary normalization ----------------------------- #

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
        "L2_post": alpha * (a2 * t2_head + b2) + beta,
        "alpha": alpha,
        "beta": beta,
    }
    return alpha, beta, dbg


def _window_mean(y: np.ndarray, start: int, stop: int) -> float:
    if stop <= start:
        raise ValueError("Invalid window for mean.")
    return float(np.mean(y[start:stop]))


def solve_affine_two_boundaries(mu_left_target: float, mu_right_target: float, mu_left_src: float, mu_right_src: float):
    denom = (mu_left_src - mu_right_src)
    if abs(denom) < 1e-12:
        raise ValueError(
            "Affine2bound degenerate: mu_left_src == mu_right_src (or too close). "
            "Try different windows or use lastfirst."
        )
    a = (mu_left_target - mu_right_target) / denom
    c = mu_left_target - a * mu_left_src
    return float(a), float(c)


# ----------------------------------- main ----------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Plot ROI mean intensity across baseline/img1/img2. "
                    "Baseline and img2 are held fixed; normalization is applied to img1 only."
    )
    parser.add_argument("img_1", help="4D NIfTI (main run, img1)")
    parser.add_argument("img_2", nargs="?", default=None, help="Second 4D NIfTI (img2, optional)")

    parser.add_argument(
        "--img2-baseline",
        default=None,
        help="Baseline NIfTI acquired with same params as img2 (before img1). 3D or 4D allowed."
    )

    parser.add_argument("--radius", type=float, required=True, help="ROI radius in voxels")
    parser.add_argument(
        "--center",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="ROI center in voxel indices (0-based). If omitted, uses image center."
    )

    parser.add_argument("--tr1", type=float, default=None, help="TR seconds for img1 (optional if .method exists)")
    parser.add_argument("--tr2", type=float, default=None, help="TR seconds for img2 (optional if .method exists)")
    parser.add_argument("--trb", type=float, default=None, help="TR seconds for baseline (optional if .method exists)")

    parser.add_argument("--stable-n", type=int, default=8, help="Initial points of img1 assumed baseline-like (default 8)")
    parser.add_argument("--norm-n", type=int, default=5, help="Tail/head window size for boundary (default 5)")

    parser.add_argument(
        "--norm-method",
        choices=["none", "lastfirst", "affine2bound", "projections", "slope", "both"],
        default="none",
        help="Normalization method. Baseline+img2 held constant; normalization is applied to img1 only.\n"
             "  none        : no scaling\n"
             "  lastfirst   : multiplicative img1 scaling; uses BOTH boundaries when possible\n"
             "  affine2bound: affine img1 transform (a*y + c) solving to match BOTH boundaries (window means)\n"
             "  projections/slope/both: apply ONLY at img1->img2 (for legacy behavior); these do NOT change baseline->img1"
    )

    # Affine/scale source controls
    parser.add_argument(
        "--affine-source",
        choices=["roi", "global", "nonzero", "mask"],
        default="roi",
        help="Where to compute boundary statistics used for lastfirst/affine2bound. "
             "roi=ROI mean, global=all voxels, nonzero=voxels with mean>eps, mask=--global-mask (file/auto)."
    )
    parser.add_argument("--global-mode", choices=["mean", "median"], default="mean",
                        help="Reducer for global/nonzero/mask sources (default mean).")
    parser.add_argument("--nonzero-eps", type=float, default=0.0,
                        help="Threshold for --affine-source nonzero (mean volume > eps). Default 0.0")

    # Optional global mask for source=mask
    parser.add_argument("--global-mask", default=None,
                        help="3D mask NIfTI for affine-source=mask, OR 'auto' to generate from a mean volume.")
    parser.add_argument("--auto-mask-source", choices=["img2", "img2_baseline", "img1"], default="img2",
                        help="If --global-mask auto, which image to build it from (default img2).")
    parser.add_argument("--auto-mask-erosions", type=int, default=3, help="Erosion iterations for auto mask (default 3)")
    parser.add_argument("--auto-mask-dilations", type=int, default=3, help="Dilation iterations for auto mask (default 3)")
    parser.add_argument("--auto-mask-hist-bins", type=int, default=256, help="Histogram bins for Otsu (default 256)")
    parser.add_argument("--auto-mask-out", default=None,
                        help="Where to write the auto-generated mask (default auto name). Set to 'none' to skip writing.")

    # Plot controls
    parser.add_argument("--time-unit", choices=["s", "min"], default="s", help="X axis units. Default seconds.")
    parser.add_argument("--mask-out", default=None, help="Output ROI mask NIfTI (default auto-named)")
    parser.add_argument("--out", default=None, help="Output plot PNG (default auto-named)")
    parser.add_argument("--csv-out", default=None, help="Optional CSV output")

    # Title controls
    parser.add_argument("--title-tag", default=None, help="Custom tag prepended to title and filenames.")
    parser.add_argument("--title-mode", choices=["full", "compact"], default="full",
                        help="compact: 'tag (x,y,z;r=R)'; full: verbose normalization info too.")

    args = parser.parse_args()

    file_prefix = _slugify(args.title_tag.strip()) + "_" if (args.title_tag is not None and args.title_tag.strip() != "") else ""

    # ---------------- Load images ----------------

    img1_path = Path(args.img_1)
    img1 = nib.load(str(img1_path))
    data1 = img1.get_fdata()
    if data1.ndim != 4:
        raise ValueError(f"img_1 must be 4D, got shape {data1.shape}")
    nx, ny, nz = data1.shape[:3]

    img2_path = Path(args.img_2) if args.img_2 else None
    data2 = None
    img2 = None
    if img2_path is not None:
        img2 = nib.load(str(img2_path))
        data2 = img2.get_fdata()
        if data2.ndim != 4:
            raise ValueError(f"img_2 must be 4D, got shape {data2.shape}")
        if data2.shape[:3] != (nx, ny, nz):
            raise ValueError(f"Spatial shapes differ: img_2 {data2.shape[:3]} vs img_1 {(nx, ny, nz)}")

    baseline_path = Path(args.img2_baseline) if args.img2_baseline else None
    datab = None
    bimg = None
    baseline_is_3d = False
    if baseline_path is not None:
        bimg = nib.load(str(baseline_path))
        datab = bimg.get_fdata()
        if datab.ndim == 3:
            datab = datab[..., np.newaxis]
            baseline_is_3d = True
        elif datab.ndim != 4:
            raise ValueError(f"--img2-baseline must be 3D or 4D, got shape {datab.shape}")
        if datab.shape[:3] != (nx, ny, nz):
            raise ValueError(f"--img2-baseline spatial shape differs: {datab.shape[:3]} vs {(nx, ny, nz)}")

    # ---------------- ROI mask + outputs ----------------

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

    out_plot = _prefix_path(out_plot, file_prefix)
    mask_out = _prefix_path(mask_out, file_prefix)
    if args.csv_out is not None:
        args.csv_out = _prefix_path(args.csv_out, file_prefix)
    if args.auto_mask_out is not None and str(args.auto_mask_out).lower() not in ("none",):
        args.auto_mask_out = _prefix_path(args.auto_mask_out, file_prefix)

    roi_mask = make_spherical_mask((nx, ny, nz), radius_vox=args.radius, center_xyz=(cx, cy, cz))
    nib.save(nib.Nifti1Image(roi_mask, img1.affine), mask_out)

    # ---------------- Optional global mask (for affine-source=mask) ----------------

    global_mask = None
    global_mask_note = "OFF"

    if args.global_mask is not None:
        if args.global_mask.lower() == "auto":
            # pick source
            if args.auto_mask_source == "img2":
                if data2 is None:
                    raise ValueError("--global-mask auto requires img_2 when --auto-mask-source img2.")
                src_data = data2
                src_aff = img2.affine
                src_name = "img2"
                src_path = img2_path
            elif args.auto_mask_source == "img2_baseline":
                if datab is None:
                    raise ValueError("--global-mask auto with img2_baseline requires --img2-baseline.")
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

            # write it unless disabled
            if args.auto_mask_out is None:
                auto_out = src_path.parent / f"{file_prefix}auto_global_mask_from_{_strip_nii_extensions(src_path)}.nii.gz"
            elif str(args.auto_mask_out).lower() == "none":
                auto_out = None
            else:
                auto_out = Path(str(args.auto_mask_out))
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

    # ---------------- Signals (ROI plot always uses ROI) ----------------

    y1_roi = timeseries_from_source(
        data1,
        source="roi",
        roi_mask_3d=roi_mask,
        global_mask_3d=global_mask,
        global_mode=args.global_mode,
        nonzero_eps=args.nonzero_eps,
    )

    y2_roi = None
    if data2 is not None:
        y2_roi = timeseries_from_source(
            data2,
            source="roi",
            roi_mask_3d=roi_mask,
            global_mask_3d=global_mask,
            global_mode=args.global_mode,
            nonzero_eps=args.nonzero_eps,
        )

    yb_roi = None
    if datab is not None:
        yb_roi = timeseries_from_source(
            datab,
            source="roi",
            roi_mask_3d=roi_mask,
            global_mask_3d=global_mask,
            global_mode=args.global_mode,
            nonzero_eps=args.nonzero_eps,
        )

    # ---------------- Timing ----------------

    tr1 = get_tr_seconds(img1_path, args.tr1)
    t1 = np.arange(len(y1_roi), dtype=float) * tr1

    tr2 = None
    if data2 is not None:
        tr2 = get_tr_seconds(img2_path, args.tr2)

    if datab is not None:
        if baseline_is_3d:
            trb = tr1
        else:
            trb = get_tr_seconds(baseline_path, args.trb)
    else:
        trb = None

    x_scale = 60.0 if args.time_unit == "min" else 1.0
    x_label = "Time (min)" if args.time_unit == "min" else "Time (s)"

    # ---------------- Boundary stats source ----------------

    affine_source = args.affine_source
    if affine_source == "mask" and global_mask is None:
        raise ValueError("--affine-source mask requires --global-mask (file or auto).")

    y1_src = timeseries_from_source(
        data1,
        source=("mask" if affine_source == "mask" else affine_source),
        roi_mask_3d=roi_mask,
        global_mask_3d=global_mask,
        global_mode=args.global_mode,
        nonzero_eps=args.nonzero_eps,
    )

    y2_src = None
    if data2 is not None:
        y2_src = timeseries_from_source(
            data2,
            source=("mask" if affine_source == "mask" else affine_source),
            roi_mask_3d=roi_mask,
            global_mask_3d=global_mask,
            global_mode=args.global_mode,
            nonzero_eps=args.nonzero_eps,
        )

    yb_src = None
    if datab is not None:
        yb_src = timeseries_from_source(
            datab,
            source=("mask" if affine_source == "mask" else affine_source),
            roi_mask_3d=roi_mask,
            global_mask_3d=global_mask,
            global_mode=args.global_mode,
            nonzero_eps=args.nonzero_eps,
        )

    # ---------------- Compute img1 transform (baseline/img2 fixed) ----------------

    img1_a = 1.0
    img1_c = 0.0
    dbg_norm = {}

    if args.stable_n < 1 or args.stable_n > len(y1_src):
        raise ValueError(f"--stable-n must be in [1..{len(y1_src)}], got {args.stable_n}")

    if args.norm_n < 1:
        raise ValueError("--norm-n must be >= 1")

    have_left = (yb_src is not None)
    have_right = (y2_src is not None)

    if args.norm_method == "lastfirst":
        # multiplicative only: y' = s*y
        scales = []

        if have_left:
            mu_b = float(np.mean(yb_src))
            mu_1L = float(np.mean(y1_src[:args.stable_n]))
            if abs(mu_1L) < 1e-12:
                raise ValueError("lastfirst: img1 left window mean is ~0; cannot scale.")
            s_left = mu_b / mu_1L
            scales.append(s_left)
            dbg_norm["s_left"] = s_left

        if have_right:
            nR = min(args.norm_n, len(y1_src), len(y2_src))
            mu_2 = float(np.mean(y2_src[:nR]))
            mu_1R = float(np.mean(y1_src[-nR:]))
            if abs(mu_1R) < 1e-12:
                raise ValueError("lastfirst: img1 right window mean is ~0; cannot scale.")
            s_right = mu_2 / mu_1R
            scales.append(s_right)
            dbg_norm["s_right"] = s_right

        if len(scales) == 0:
            img1_a = 1.0
            img1_c = 0.0
        else:
            img1_a = float(np.mean(scales))
            img1_c = 0.0
        dbg_norm["img1_a"] = img1_a
        dbg_norm["img1_c"] = img1_c

    elif args.norm_method == "affine2bound":
        # affine: y' = a*y + c, match BOTH boundaries in window-mean sense
        if not have_left or not have_right:
            raise ValueError("affine2bound requires BOTH --img2-baseline and img_2.")

        nR = min(args.norm_n, len(y1_src), len(y2_src))
        mu_b = float(np.mean(yb_src))
        mu_2 = float(np.mean(y2_src[:nR]))
        mu_1L = float(np.mean(y1_src[:args.stable_n]))
        mu_1R = float(np.mean(y1_src[-nR:]))

        img1_a, img1_c = solve_affine_two_boundaries(mu_b, mu_2, mu_1L, mu_1R)

        dbg_norm["mu_b"] = mu_b
        dbg_norm["mu_2"] = mu_2
        dbg_norm["mu_1L"] = mu_1L
        dbg_norm["mu_1R"] = mu_1R
        dbg_norm["img1_a"] = img1_a
        dbg_norm["img1_c"] = img1_c

    else:
        img1_a = 1.0
        img1_c = 0.0

    # Apply img1 transform to ROI plot series (and also to y1_src for reporting)
    if img1_a != 1.0 or img1_c != 0.0:
        y1_roi = img1_a * y1_roi + img1_c
        y1_src_post = img1_a * y1_src + img1_c
    else:
        y1_src_post = y1_src.copy()

    # ---------------- Build segments for plot ----------------

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
    if img1_a != 1.0 or img1_c != 0.0:
        labels.append(f"img1_ROI→a*y+c (a={img1_a:.6g}, c={img1_c:.6g})")
    else:
        labels.append("img1_ROI")
    t_cursor = t1_abs[-1] + tr1

    dbg_slope = None
    scale_proj = 1.0
    alpha = 1.0
    beta = 0.0

    if data2 is not None:
        t2_abs = t_cursor + np.arange(len(y2_roi), dtype=float) * tr2
        segments_t.append(t2_abs)
        segments_y.append(y2_roi)
        labels.append("img2_ROI")

        # Legacy boundary normalizations on img2 ONLY (optional)
        if args.norm_method in ("projections", "both"):
            scale_proj = float(get_npro(img1_path) / get_npro(img2_path))
            segments_y[-1] = segments_y[-1] * scale_proj

        if args.norm_method in ("slope", "both"):
            alpha, beta, dbg_slope = affine_match_img1_tail_to_img2_head(
                t1_abs, segments_y[1 if datab is not None else 0], t2_abs, segments_y[-1], n=max(2, args.norm_n)
            )
            segments_y[-1] = alpha * segments_y[-1] + beta

    # ---------------- Plot ----------------

    t_all = np.concatenate(segments_t, axis=0)
    y_all = np.concatenate(segments_y, axis=0)

    plt.figure()
    plt.plot(t_all / x_scale, y_all)

    # boundary lines
    if datab is not None:
        plt.axvline((t1_abs[0]) / x_scale, linestyle="--")
    if data2 is not None:
        plt.axvline((segments_t[-1][0]) / x_scale, linestyle="--")

    # slope debug overlays (post-normalization only)
    if data2 is not None and args.norm_method in ("slope", "both") and dbg_slope is not None:
        plt.plot(dbg_slope["t1_tail"] / x_scale, dbg_slope["y1_fit_tail"], linestyle="--", alpha=0.7)
        plt.plot(dbg_slope["t2_head"] / x_scale, dbg_slope["L2_post"], linestyle="--", alpha=0.7)

    plt.xlabel(x_label)
    plt.ylabel("Mean ROI intensity")

    # ---------------- Title handling ----------------

    if args.title_tag is not None and args.title_tag.strip() != "":
        title_tag = args.title_tag.strip()
    else:
        title_tag = "ROI intensity"

    coord_line = f"({int(round(cx))},{int(round(cy))},{int(round(cz))};r={args.radius:g})"

    ax = plt.gca()

    if args.title_mode == "compact":
        ax.set_title(f"{title_tag} {coord_line}", fontsize=12)
    else:
        title = f"{title_tag} {coord_line}\nROI mean intensity (r={args.radius:g}, center={center_label})"
        title += f"\n(affine-source={args.affine_source}, reducer={args.global_mode}, global-mask={global_mask_note}, nonzero-eps={args.nonzero_eps:g})"
        title += f"\nimg1 transform: y'={img1_a:.6g}*y + {img1_c:.6g} (method={args.norm_method})"

        if args.norm_method in ("lastfirst", "affine2bound"):
            if have_left:
                title += f"\nleft match uses baseline mean vs img1[:{args.stable_n}]"
            if have_right and y2_src is not None:
                nR = min(args.norm_n, len(y1_src), len(y2_src))
                title += f"\nright match uses img2[:{nR}] vs img1[-{nR}:]"

        if data2 is not None:
            if args.norm_method == "projections":
                title += f"\nimg2 scaled by projections={scale_proj:.6g}"
            elif args.norm_method == "slope":
                title += f"\nimg2 affine: alpha={alpha:.6g}, beta={beta:.6g} (n={args.norm_n})"
            elif args.norm_method == "both":
                title += f"\nimg2 proj={scale_proj:.6g}, then affine alpha={alpha:.6g}, beta={beta:.6g} (n={args.norm_n})"

        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)

    # CSV
    if args.csv_out:
        with open(args.csv_out, "w") as f:
            f.write(f"time_{args.time_unit},roi_mean_intensity\n")
            for ti, yi in zip(t_all / x_scale, y_all):
                f.write(f"{ti:.9g},{yi:.9g}\n")

    # ---------------- QA prints ----------------

    print("[mode] ROI plot; baseline+img2 fixed; img1 transformed only")
    print(f"       segments: {labels}")
    print(f"[roi] center={center_label}, radius={args.radius:g}, voxels={int(roi_mask.sum())}")

    print(f"[affine-source] {args.affine_source}  (reducer={args.global_mode}, nonzero-eps={args.nonzero_eps:g}, global-mask={global_mask_note})")
    print(f"[img1-transform] method={args.norm_method}  a={img1_a:.9g}  c={img1_c:.9g}")

    # show boundary residuals in the SOURCE domain (not ROI), since that’s what drove the transform
    if have_left:
        mu_b = float(np.mean(yb_src))
        mu_1L_pre = float(np.mean(y1_src[:args.stable_n]))
        mu_1L_post = float(np.mean(y1_src_post[:args.stable_n]))
        print(f"[left]  baseline_mean={mu_b:.6g}  img1_stable_pre={mu_1L_pre:.6g}  img1_stable_post={mu_1L_post:.6g}  resid(post-baseline)={mu_1L_post - mu_b:.6g}")

    if have_right:
        nR = min(args.norm_n, len(y1_src), len(y2_src))
        mu_2 = float(np.mean(y2_src[:nR]))
        mu_1R_pre = float(np.mean(y1_src[-nR:]))
        mu_1R_post = float(np.mean(y1_src_post[-nR:]))
        print(f"[right] img2_head_mean={mu_2:.6g}  img1_tail_pre={mu_1R_pre:.6g}  img1_tail_post={mu_1R_post:.6g}  resid(post-img2)={mu_1R_post - mu_2:.6g}")

    print(f"[out] roi-mask: {mask_out}")
    print(f"[out] plot    : {out_plot}")
    if args.csv_out:
        print(f"[out] csv     : {args.csv_out}")


if __name__ == "__main__":
    main()