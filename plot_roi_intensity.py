#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

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

    # Bruker style arrays: "##$X=( 2 )" then values on following lines
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


# ----------------------------- ROI + reducers ----------------------------- #

def make_spherical_mask(shape_xyz, radius_vox: float, center_xyz: tuple[float, float, float]) -> np.ndarray:
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


def reduce_timeseries_from_mask(data_4d: np.ndarray, mask_3d: np.ndarray | None, reducer: str) -> np.ndarray:
    if data_4d.ndim != 4:
        raise ValueError(f"Expected 4D data, got shape {data_4d.shape}")

    if mask_3d is not None:
        if mask_3d.shape != data_4d.shape[:3]:
            raise ValueError(f"Mask shape {mask_3d.shape} != data spatial shape {data_4d.shape[:3]}")
        vox = data_4d[mask_3d > 0, :]
    else:
        vox = data_4d.reshape(-1, data_4d.shape[3])

    if vox.size == 0:
        raise RuntimeError("Reducer selected 0 voxels.")

    if reducer == "mean":
        return np.nanmean(vox, axis=0)
    if reducer == "median":
        return np.nanmedian(vox, axis=0)

    raise ValueError(f"Unknown reducer: {reducer}")


def reduce_timeseries_from_nonzero(data_4d: np.ndarray, reducer: str, nonzero_eps: float) -> np.ndarray:
    if data_4d.ndim != 4:
        raise ValueError(f"Expected 4D data, got shape {data_4d.shape}")
    out = np.zeros((data_4d.shape[3],), dtype=np.float64)
    flat = data_4d.reshape(-1, data_4d.shape[3])
    for i in range(data_4d.shape[3]):
        v = flat[:, i]
        v = v[np.isfinite(v)]
        v = v[v > nonzero_eps]
        if v.size == 0:
            out[i] = np.nan
        else:
            out[i] = np.nanmean(v) if reducer == "mean" else np.nanmedian(v)
    return out


# ----------------------------- auto brain mask (global) ----------------------------- #

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
    return float((bin_edges[idx] + bin_edges[idx + 1]) * 0.5)


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
    struct = ndi.generate_binary_structure(3, 2)
    if erosions > 0:
        m = ndi.binary_erosion(m, structure=struct, iterations=erosions)
    lbl, n = ndi.label(m, structure=struct)
    if n < 1:
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


# ----------------------------- line fitting + affine solve ----------------------------- #

def fit_line_or_constant(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if t.size != y.size:
        raise ValueError("fit_line: t and y must be same length")
    if t.size < 1:
        raise ValueError("fit_line: need >=1 point")
    if t.size == 1:
        return 0.0, float(y[0])
    m, b = np.polyfit(t.astype(np.float64), y.astype(np.float64), 1)
    return float(m), float(b)


def eval_line(m: float, b: float, t: float) -> float:
    return float(m * t + b)


def solve_affine_from_two_constraints(y1L_hat: float, yB_hat: float, y1R_hat: float, y2_hat: float) -> tuple[float, float]:
    denom = (y1R_hat - y1L_hat)
    if abs(denom) < 1e-12:
        raise ValueError("Affine solve ill-conditioned: y1R_hat ~ y1L_hat (denominator ~ 0)")
    a = (y2_hat - yB_hat) / denom
    c = yB_hat - a * y1L_hat
    return float(a), float(c)


def solve_affine_match_line_right(m1R: float, b1R: float, m2: float, b2: float) -> tuple[float, float]:
    if abs(m1R) < 1e-12:
        raise ValueError("Right-only solve ill-conditioned: m1R ~ 0 (flat img1 tail).")
    a = m2 / m1R
    c = b2 - a * b1R
    return float(a), float(c)


def robust_std(x: np.ndarray) -> float:
    """MAD-based robust std estimate."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad)  # normal-consistent


# ----------------------------------- main ----------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Plot ROI intensity across baseline + img1 + img2. "
                    "Supports affine transform of img1 only, with C0 continuity constraints from fitted lines, "
                    "optionally penalizing variance blow-up and negative values."
    )
    parser.add_argument("img_1", help="4D NIfTI (block1 / main run)")
    parser.add_argument("img_2", nargs="?", default=None, help="Second 4D NIfTI (block2)")

    parser.add_argument("--img2-baseline", default=None,
                        help="Baseline 3D/4D NIfTI acquired with same params as img_2 (before img_1).")

    parser.add_argument("--radius", type=float, required=True, help="ROI radius in voxels")
    parser.add_argument("--center", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                        help="ROI center in voxel indices (0-based). If omitted, uses image center.")

    parser.add_argument("--tr1", type=float, default=None, help="TR seconds for img_1 (optional if method exists)")
    parser.add_argument("--tr2", type=float, default=None, help="TR seconds for img_2 (optional if method exists)")
    parser.add_argument("--trb", type=float, default=None, help="TR seconds for baseline (optional if method exists)")

    parser.add_argument("--title-tag", default=None,
                        help="Optional custom tag to include in the plot title.")
    parser.add_argument("--title-mode", choices=["full", "compact"], default="full",
                        help="Title style.")
    parser.add_argument("--time-unit", choices=["s", "min"], default="s",
                        help="X axis units for plotting (and CSV if written).")

    parser.add_argument("--mask-out", default=None, help="Output ROI mask NIfTI (default auto-named)")
    parser.add_argument("--out", default=None, help="Output plot PNG (default auto-named)")
    parser.add_argument("--csv-out", default=None, help="Optional CSV output")

    # Line windows
    parser.add_argument("--stable-n", type=int, default=8,
                        help="Number of initial img1 points for left-line fit (default 8).")
    parser.add_argument("--tail-n", type=int, default=8,
                        help="Number of final img1 points for right-line fit (default 8).")
    parser.add_argument("--base-tail-n", type=int, default=0,
                        help="Number of final baseline points for baseline-line fit. "
                             "If 0, uses all baseline points (even if that is 1).")
    parser.add_argument("--img2-head-n", type=int, default=8,
                        help="Number of initial img2 points for img2-line fit (default 8).")

    # Transform choice
    parser.add_argument("--norm-method",
                        choices=["none", "affine2boundline"],
                        default="none",
                        help="Transform applied to img1 only. "
                             "'affine2boundline' enforces continuity from fitted lines. "
                             "If baseline is missing, it enforces continuity at the right boundary only.")

    # Affine source signal (for computing a,c) can be ROI or global-like
    parser.add_argument("--affine-source",
                        choices=["roi", "global", "nonzero"],
                        default="roi",
                        help="Signal used to compute the affine transform parameters (a,c). "
                             "roi=ROI mean timeseries; global=masked/global reducer timeseries; "
                             "nonzero=reducer over >eps voxels for each frame.")

    parser.add_argument("--reducer", choices=["mean", "median"], default="mean",
                        help="Reducer for global/nonzero sources (default mean).")
    parser.add_argument("--nonzero-eps", type=float, default=0.0,
                        help="Threshold for nonzero reducer: keep voxels > eps (default 0).")

    # Global mask controls (used if affine-source=global)
    parser.add_argument("--global-mask", default=None,
                        help="3D mask NIfTI for global reducer OR 'auto' to generate from mean(volume). "
                             "If omitted, uses ALL voxels.")
    parser.add_argument("--auto-mask-source", choices=["img2", "img2_baseline", "img1"], default="img2",
                        help="If --global-mask auto, which image to build it from (default img2).")
    parser.add_argument("--auto-mask-erosions", type=int, default=3)
    parser.add_argument("--auto-mask-dilations", type=int, default=3)
    parser.add_argument("--auto-mask-hist-bins", type=int, default=256)
    parser.add_argument("--auto-mask-out", default=None,
                        help="Write auto global mask to this path (default auto name). Use 'none' to skip writing.")

    # Solver mode + penalties
    parser.add_argument("--affine-solver",
                        choices=["exact", "robust"],
                        default="robust",
                        help="How to solve for (a,c). "
                             "exact = your previous exact continuity solve; "
                             "robust = soft continuity + penalties (std blowup + negatives).")

    parser.add_argument("--std-max-ratio", type=float, default=1.7,
                        help="Desired upper bound on std(img1_transformed_window) / std_target. Default 1.7 (~70%% higher).")
    parser.add_argument("--std-penalty", type=float, default=1e6,
                        help="Weight for std blow-up penalty. Larger -> stronger suppression.")
    parser.add_argument("--neg-penalty", type=float, default=1e6,
                        help="Weight for negative-value penalty (hinge^2). Larger -> pushes transform to keep signal >= 0.")
    parser.add_argument("--boundary-penalty", type=float, default=1e9,
                        help="Weight for boundary continuity penalty. Larger -> closer to exact continuity.")
    parser.add_argument("--use-robust-std", action="store_true",
                        help="Use MAD-based robust std in penalties instead of classic std.")

    # Constraints / priors on a
    parser.add_argument("--a-positive", action="store_true",
                        help="Constrain a >= 0 (recommended).")
    parser.add_argument("--c-nonneg", action="store_true",
                        help="Constrain c >= 0 (optional; can be too strict if you truly need an offset).")

    parser.add_argument("--a-expected-from-npro", action="store_true",
                        help="Compute a_expected = NPro(img2)/NPro(img1) (requires method sidecars).")
    parser.add_argument("--a-expected", type=float, default=None,
                        help="Override expected a (e.g., 3.0).")
    parser.add_argument("--a-bound-scale", type=float, default=0.0,
                        help="If >0, constrain a to [a_expected/scale, a_expected*scale]. "
                             "0 disables bounds.")
    parser.add_argument("--a-prior-penalty", type=float, default=0.0,
                        help="If >0, add penalty a_prior_penalty * ((a-a_expected)/a_expected)^2.")

    args = parser.parse_args()

    # ---------------- Filename prefixing from title tag ----------------
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

    file_prefix = _slugify(args.title_tag.strip()) + "_" if (args.title_tag and args.title_tag.strip()) else ""
    def _prefix_path(p: str | None) -> str | None:
        if p is None:
            return None
        pp = Path(p)
        if file_prefix == "" or pp.name.startswith(file_prefix):
            return str(pp)
        return str(pp.with_name(file_prefix + pp.name))

    # Load img1
    img1_path = Path(args.img_1)
    img1 = nib.load(str(img1_path))
    data1 = img1.get_fdata()
    if data1.ndim != 4:
        raise ValueError(f"img_1 must be 4D, got shape {data1.shape}")
    nx, ny, nz = data1.shape[:3]

    # Load img2
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

    # Load baseline (optional) - allow 3D or 4D
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

    # ROI center
    if args.center is None:
        cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
        center_label = "center"
    else:
        cx, cy, cz = args.center
        if not (0 <= cx < nx and 0 <= cy < ny and 0 <= cz < nz):
            raise ValueError(f"Center {args.center} out of bounds.")
        center_label = f"{int(round(cx))}_{int(round(cy))}_{int(round(cz))}"

    roi_tag = f"xyz_{center_label}_r{args.radius:g}"
    out_plot = _prefix_path(args.out or f"roi_intensity_{roi_tag}.png")
    mask_out = _prefix_path(args.mask_out or f"roi_mask_{roi_tag}.nii.gz")
    csv_out = _prefix_path(args.csv_out) if args.csv_out else None

    # ROI mask
    roi_mask = make_spherical_mask((nx, ny, nz), radius_vox=args.radius, center_xyz=(cx, cy, cz))
    nib.save(nib.Nifti1Image(roi_mask, img1.affine), str(mask_out))

    # Build global mask if requested (used if affine-source=global)
    global_mask = None
    global_mask_note = "OFF"

    if args.global_mask is not None:
        if args.global_mask.lower() == "auto":
            if args.auto_mask_source == "img2":
                if data2 is None:
                    raise ValueError("--global-mask auto with --auto-mask-source img2 requires img_2.")
                src_data, src_aff, src_name, src_path = data2, img2.affine, "img2", img2_path
            elif args.auto_mask_source == "img2_baseline":
                if datab is None:
                    raise ValueError("--global-mask auto with --auto-mask-source img2_baseline requires --img2-baseline.")
                src_data, src_aff, src_name, src_path = datab, bimg.affine, "img2_baseline", baseline_path
            else:
                src_data, src_aff, src_name, src_path = data1, img1.affine, "img1", img1_path

            global_mask = generate_auto_brain_mask_from_mean(
                src_data,
                erosions=args.auto_mask_erosions,
                dilations=args.auto_mask_dilations,
                nbins=args.auto_mask_hist_bins,
            ).astype(bool)
            global_mask_note = f"AUTO({src_name})"

            if args.auto_mask_out is None:
                auto_out = src_path.parent / f"{file_prefix}auto_global_mask_from_{_strip_nii_extensions(src_path)}.nii.gz"
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

    # TRs + time axes (segment-local)
    tr1 = get_tr_seconds(img1_path, args.tr1)
    t1 = np.arange(data1.shape[3], dtype=float) * tr1

    tr2 = None
    t2 = None
    if data2 is not None:
        tr2 = get_tr_seconds(img2_path, args.tr2)
        t2 = np.arange(data2.shape[3], dtype=float) * tr2

    trb = None
    tb = None
    if datab is not None:
        trb = get_tr_seconds(baseline_path, args.trb) if not baseline_is_3d else tr1
        tb = np.arange(datab.shape[3], dtype=float) * trb

    # Plot time scaling
    x_scale = 60.0 if args.time_unit == "min" else 1.0
    x_label = "Time (min)" if args.time_unit == "min" else "Time (s)"

    # ROI timeseries (always plotted)
    y1_roi = mean_roi_timeseries(data1, roi_mask)
    y2_roi = mean_roi_timeseries(data2, roi_mask) if data2 is not None else None
    yb_roi = mean_roi_timeseries(datab, roi_mask) if datab is not None else None

    # Build absolute time axes for concatenation
    segments_t = []
    segments_y = []
    labels = []
    t_cursor = 0.0

    tb_abs = None
    if datab is not None:
        tb_abs = t_cursor + tb
        segments_t.append(tb_abs)
        segments_y.append(yb_roi.copy())
        labels.append("baseline_ROI")
        t_cursor = tb_abs[-1] + trb

    t1_abs = t_cursor + t1
    segments_t.append(t1_abs)
    segments_y.append(y1_roi.copy())
    labels.append("img1_ROI")
    t_cursor = t1_abs[-1] + tr1

    t2_abs = None
    if data2 is not None:
        t2_abs = t_cursor + t2
        segments_t.append(t2_abs)
        segments_y.append(y2_roi.copy())
        labels.append("img2_ROI")

    # ---------------- affine source signals (for computing a,c) ---------------- #
    def get_affine_signal(data_4d: np.ndarray) -> np.ndarray:
        if args.affine_source == "roi":
            return mean_roi_timeseries(data_4d, roi_mask)
        if args.affine_source == "global":
            return reduce_timeseries_from_mask(data_4d, global_mask, args.reducer)
        if args.affine_source == "nonzero":
            return reduce_timeseries_from_nonzero(data_4d, args.reducer, args.nonzero_eps)
        raise ValueError("Unknown affine source")

    # ---------------- Transform: affine2boundline ---------------- #
    a = 1.0
    c = 0.0
    ok = True
    msg = "none"
    left_resid = None
    right_resid = None
    penalty_info = {}

    if args.norm_method == "affine2boundline":
        if data2 is None:
            raise ValueError("affine2boundline requires img_2 (block2) to be provided.")

        # affine-source signals
        s1 = get_affine_signal(data1)
        s2 = get_affine_signal(data2)
        sB = get_affine_signal(datab) if datab is not None else None

        stable_n = int(args.stable_n)
        tail_n = int(args.tail_n)
        head2_n = int(args.img2_head_n)
        base_tail_n = int(args.base_tail_n)

        if stable_n < 2 or stable_n > s1.size:
            raise ValueError(f"--stable-n must be in [2..len(img1)] got {stable_n}")
        if tail_n < 2 or tail_n > s1.size:
            raise ValueError(f"--tail-n must be in [2..len(img1)] got {tail_n}")
        if head2_n < 2 or head2_n > s2.size:
            raise ValueError(f"--img2-head-n must be in [2..len(img2)] got {head2_n}")

        if sB is not None:
            if base_tail_n <= 0:
                base_tail_n = int(sB.size)
            if base_tail_n < 1 or base_tail_n > sB.size:
                raise ValueError(f"--base-tail-n must be in [1..len(baseline)] got {base_tail_n}")

        # boundary times in the concatenated plot
        tL = float(t1_abs[0])          # baseline -> img1 boundary time (if baseline exists)
        tR = float(t2_abs[0])          # img1 -> img2 boundary time

        # fit lines
        m1L, b1L = fit_line_or_constant(t1_abs[:stable_n], s1[:stable_n])
        m1R, b1R = fit_line_or_constant(t1_abs[-tail_n:], s1[-tail_n:])
        m2,  b2  = fit_line_or_constant(t2_abs[:head2_n], s2[:head2_n])

        # expected a (optional)
        a_expected = None
        if args.a_expected is not None:
            a_expected = float(args.a_expected)
        elif args.a_expected_from_npro:
            a_expected = float(get_npro(img2_path) / get_npro(img1_path))

        a_lo = None
        a_hi = None
        if a_expected is not None and args.a_bound_scale and args.a_bound_scale > 0:
            a_lo = a_expected / float(args.a_bound_scale)
            a_hi = a_expected * float(args.a_bound_scale)

        # std target (baseline tail + img2 head)
        def _std_fn(x):
            return robust_std(x) if args.use_robust_std else float(np.nanstd(x))

        std2 = _std_fn(s2[:head2_n])
        if sB is not None:
            stdB = _std_fn(sB[-base_tail_n:])
            std_target = float(np.nanmean([stdB, std2]))
        else:
            stdB = None
            std_target = float(std2)

        if not np.isfinite(std_target) or std_target <= 0:
            std_target = 1.0

        # boundary hats for two-boundary mode
        if sB is not None:
            mB, bB = fit_line_or_constant(tb_abs[-base_tail_n:], sB[-base_tail_n:])
            yB_hat  = eval_line(mB,  bB,  tL)
            y1L_hat = eval_line(m1L, b1L, tL)
            y2_hat  = eval_line(m2,  b2,  tR)
            y1R_hat = eval_line(m1R, b1R, tR)

        # objective in terms of a only; choose c(a) analytically
        def c_from_a(a_val: float) -> float:
            # Weighted least squares for c to satisfy boundary constraints as much as possible.
            wL = float(args.boundary_penalty)
            wR = float(args.boundary_penalty)
            if sB is not None:
                # two constraints: a*y1L+c ~ yB and a*y1R+c ~ y2
                num = wL * (yB_hat - a_val * y1L_hat) + wR * (y2_hat - a_val * y1R_hat)
                den = (wL + wR)
                return float(num / den)
            else:
                # no baseline: match right-boundary line intercept at any t (C0/C1 already in a)
                # best c is from matching intercepts: a*b1R + c = b2
                return float(b2 - a_val * b1R)

        def objective(a_val: float) -> float:
            # constraints
            if args.a_positive and a_val < 0:
                return 1e30
            if a_lo is not None and a_val < a_lo:
                return 1e30
            if a_hi is not None and a_val > a_hi:
                return 1e30

            c_val = c_from_a(a_val)
            if args.c_nonneg and c_val < 0:
                return 1e30

            # boundary residuals
            loss = 0.0
            if sB is not None:
                rL = (a_val * y1L_hat + c_val) - yB_hat
                rR = (a_val * y1R_hat + c_val) - y2_hat
                loss += float(args.boundary_penalty) * (rL * rL + rR * rR)

            # std blow-up penalty computed on transformed s1 windows (affine-source signal)
            # (we penalize the *larger* of the head/tail stds)
            s1_head = a_val * s1[:stable_n] + c_val
            s1_tail = a_val * s1[-tail_n:] + c_val
            std_head = _std_fn(s1_head)
            std_tail = _std_fn(s1_tail)
            std_use = float(np.nanmax([std_head, std_tail]))

            ratio = std_use / std_target if std_target > 0 else float("inf")
            excess = max(0.0, ratio - float(args.std_max_ratio))
            loss += float(args.std_penalty) * (excess * excess)

            # negative penalty computed on transformed *source* windows
            neg = np.concatenate([s1_head, s1_tail], axis=0)
            neg_amount = np.clip(-neg, 0.0, None)
            loss += float(args.neg_penalty) * float(np.mean(neg_amount * neg_amount))

            # optional a prior
            if a_expected is not None and args.a_prior_penalty and args.a_prior_penalty > 0:
                rel = (a_val - a_expected) / a_expected if a_expected != 0 else (a_val - a_expected)
                loss += float(args.a_prior_penalty) * float(rel * rel)

            # stash some debug-ish numbers
            penalty_info["std_target"] = std_target
            penalty_info["stdB"] = stdB
            penalty_info["std2"] = std2
            penalty_info["std_head"] = std_head
            penalty_info["std_tail"] = std_tail
            penalty_info["std_ratio_maxwin"] = ratio
            penalty_info["c_from_a"] = c_val

            return float(loss)

        # Solve for a
        if args.affine_solver == "exact":
            if sB is not None:
                a, c = solve_affine_from_two_constraints(y1L_hat, yB_hat, y1R_hat, y2_hat)
                msg = "exact_two_boundary_C0"
            else:
                a, c = solve_affine_match_line_right(m1R, b1R, m2, b2)
                msg = "exact_right_line_match"
        else:
            # robust 1D search over a
            # pick a search bracket
            if a_lo is not None and a_hi is not None:
                lo, hi = float(a_lo), float(a_hi)
            elif a_expected is not None:
                lo, hi = max(0.0, a_expected / 10.0), a_expected * 10.0
            else:
                # broad but not insane
                lo, hi = 0.0, 50.0

            # coarse grid to find a good start
            grid = np.linspace(lo, hi, 401, dtype=np.float64)
            vals = np.array([objective(float(av)) for av in grid], dtype=np.float64)
            k = int(np.nanargmin(vals))
            a0 = float(grid[k])

            # refine with golden-section style search
            def golden_section_min(f, aL, aU, iters=80):
                gr = (np.sqrt(5) + 1) / 2
                c1 = aU - (aU - aL) / gr
                c2 = aL + (aU - aL) / gr
                f1 = f(c1)
                f2 = f(c2)
                for _ in range(iters):
                    if f1 > f2:
                        aL = c1
                        c1 = c2
                        f1 = f2
                        c2 = aL + (aU - aL) / gr
                        f2 = f(c2)
                    else:
                        aU = c2
                        c2 = c1
                        f2 = f1
                        c1 = aU - (aU - aL) / gr
                        f1 = f(c1)
                return (aL + aU) / 2.0

            span = hi - lo
            aL = max(lo, a0 - 0.1 * span)
            aU = min(hi, a0 + 0.1 * span)
            a = float(golden_section_min(objective, aL, aU, iters=100))
            c = float(c_from_a(a))
            msg = "robust_penalized_1D"

            if args.a_positive and a < 0:
                ok = False
            if args.c_nonneg and c < 0:
                ok = False

        # QA residuals on the actual line constraints if baseline exists
        if sB is not None:
            left_resid = (a * y1L_hat + c) - yB_hat
            right_resid = (a * y1R_hat + c) - y2_hat

        # Apply affine to plotted img1 ROI curve ONLY
        y1_roi_xform = a * y1_roi + c
        segments_y[0 if datab is None else 1] = y1_roi_xform
        labels[0 if datab is None else 1] = f"img1_ROI→a*y+c (a={a:.6g}, c={c:.6g})"

    # Concatenate + plot
    t_all = np.concatenate(segments_t, axis=0)
    y_all = np.concatenate(segments_y, axis=0)

    plt.figure()
    plt.plot(t_all / x_scale, y_all)

    # boundaries
    if datab is not None:
        plt.axvline((t1_abs[0]) / x_scale, linestyle="--")
    if data2 is not None:
        plt.axvline((t2_abs[0]) / x_scale, linestyle="--")

    plt.xlabel(x_label)
    plt.ylabel("Mean ROI intensity")

    # Title
    if args.title_tag is not None and args.title_tag.strip() != "":
        title_tag = args.title_tag.strip()
    else:
        title_tag = "ROI intensity"

    coord_line = f"({int(round(cx))},{int(round(cy))},{int(round(cz))};r={args.radius:g})" if args.center is not None else f"({center_label};r={args.radius:g})"
    ax = plt.gca()
    if args.title_mode == "compact":
        ax.set_title(f"{title_tag} {coord_line}", fontsize=12)
    else:
        title = f"{title_tag} {coord_line}\nROI mean intensity"
        title += f"\n(mode={args.norm_method}, solver={args.affine_solver}, affine-source={args.affine_source}, reducer={args.reducer}, global-mask={global_mask_note})"
        if args.norm_method == "affine2boundline":
            title += f"\na={a:.6g}, c={c:.6g}  ok={ok}  msg={msg}"
            if left_resid is not None:
                title += f"\nline-C0 left_resid={left_resid:.3g}  right_resid={right_resid:.3g}"
            if penalty_info:
                title += f"\nstd_target={penalty_info.get('std_target', float('nan')):.3g}  std_ratio_maxwin={penalty_info.get('std_ratio_maxwin', float('nan')):.3g}"
        ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(str(out_plot), dpi=150)

    # CSV
    if csv_out:
        with open(csv_out, "w") as f:
            f.write(f"time_{args.time_unit},roi_mean_intensity\n")
            for ti, yi in zip(t_all / x_scale, y_all):
                f.write(f"{ti:.9g},{yi:.9g}\n")

    # ---------------- prints ----------------
    if args.norm_method == "affine2boundline":
        print("[mode] ROI plot; baseline+img2 fixed; img1 transformed only" if datab is not None else "[mode] ROI plot; img2 fixed; img1 transformed only (no baseline)")
    else:
        print("[mode] ROI plot (no transform)")

    print(f"       segments: {labels}")
    print(f"[roi] center={center_label}, radius={args.radius:g}, voxels={int(roi_mask.sum())}")
    print(f"[affine-source] {args.affine_source}  (reducer={args.reducer}, nonzero-eps={args.nonzero_eps:g}, global-mask={global_mask_note})")

    if args.norm_method == "affine2boundline":
        print(f"[img1-transform] method=affine2boundline  solver={args.affine_solver}  a={a:.10g}  c={c:.10g}  ok={ok}  msg={msg}")
        if left_resid is not None:
            print(f"[line-C0] left_resid={left_resid:.6g}")
            print(f"[line-C0] right_resid={right_resid:.6g}")
        if penalty_info:
            print(f"[penalty] std_target={penalty_info.get('std_target', float('nan')):.6g}  std2={penalty_info.get('std2', float('nan')):.6g}  stdB={penalty_info.get('stdB', float('nan')):.6g}")
            print(f"[penalty] std_head={penalty_info.get('std_head', float('nan')):.6g}  std_tail={penalty_info.get('std_tail', float('nan')):.6g}  std_ratio_maxwin={penalty_info.get('std_ratio_maxwin', float('nan')):.6g}")

    # quick sanity warning if ROI itself has negatives BEFORE transform (this is a data problem, not just fitting)
    if np.nanmin(y1_roi) < 0:
        frac_neg = float(np.mean(y1_roi < 0))
        print(f"[WARN] img1 ROI timeseries has negatives pre-transform: min={np.nanmin(y1_roi):.6g}, frac_neg={frac_neg:.3%}. "
              f"If these are magnitude images, that suggests a reconstruction/scaling issue upstream.")

    print(f"[out] roi-mask: {mask_out}")
    print(f"[out] plot    : {out_plot}")
    if csv_out:
        print(f"[out] csv     : {csv_out}")


if __name__ == "__main__":
    main()