#!/usr/bin/env python3
"""
plot_roi_intensity.py

Plot ROI mean intensity across (optional) baseline + img1 + (optional) img2,
and (optionally) apply an affine transform to img1 only:

    y1' = a * y1 + c

Normalization mode implemented:

  --norm-method affine2boundline

“Version 2: continuous line well-fit across boundary” (C0 using fitted lines)

- If BOTH baseline and img2 exist:
    Fit a line to baseline tail and img1 head; fit a line to img1 tail and img2 head.
    Evaluate both at their respective boundary times and solve for (a,c) such that
    img1' hits the baseline line at left boundary and hits img2 line at right boundary.
    This yields an exact C0 solution (a_exact, c_exact).

    If exact solution violates constraints (bounds / std / positivity), fall back to
    a robust constrained search that minimizes BOTH boundary residuals, with c(a)
    chosen by least squares across both boundaries.

- If ONLY baseline exists (no img2):
    One boundary only: anchor on left, choose c(a)=yB_hat - a*y1L_hat.

- If ONLY img2 exists (no baseline):
    One boundary only: anchor on right, choose c(a)=y2_hat - a*y1R_hat.

STD constraint (TRUE pooling):
- std_target is computed from pooled samples of BOTH baseline window and img2 head window
  (whichever are present) using the *same affine-source* signal.
- If the pooled set has <2 samples, std_target is NaN and the std constraint is disabled.

Affined-source for solving (a,c) can be:
- roi       : ROI mean
- global    : reducer over a mask (file mask or auto mask); if no mask, uses all voxels
- nonzero   : reducer over voxels > eps for each frame

Plot always displays ROI mean curves. The affine parameters (a,c) can be computed
from ROI or global/nonzero signals depending on --affine-source.
"""

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

def fit_line(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if t.size != y.size:
        raise ValueError("fit_line: t and y must be same length")
    if t.size < 2:
        raise ValueError("fit_line: need >=2 points")
    m, b = np.polyfit(t.astype(np.float64), y.astype(np.float64), 1)
    return float(m), float(b)


def eval_line(m: float, b: float, t: float) -> float:
    return float(m * t + b)


def safe_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1))


def pooled_std(*arrays: np.ndarray) -> float:
    xs = []
    for a in arrays:
        if a is None:
            continue
        v = np.asarray(a, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size > 0:
            xs.append(v)
    if not xs:
        return float("nan")
    x = np.concatenate(xs, axis=0)
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1))


def solve_affine_from_two_constraints(
    y1L_hat: float,
    yB_hat: float,
    y1R_hat: float,
    y2_hat: float
) -> tuple[float, float]:
    denom = (y1R_hat - y1L_hat)
    if abs(denom) < 1e-12:
        raise ValueError("Affine solve ill-conditioned: y1R_hat ~ y1L_hat (denominator ~ 0)")
    a = (y2_hat - yB_hat) / denom
    c = yB_hat - a * y1L_hat
    return float(a), float(c)


def c_least_squares_two_boundaries(a: float, y1L_hat: float, yB_hat: float, y1R_hat: float, y2_hat: float) -> float:
    """
    Given 'a', choose 'c' to minimize:
      (a*y1L_hat + c - yB_hat)^2 + (a*y1R_hat + c - y2_hat)^2
    Closed form:
      c = mean([yB_hat - a*y1L_hat, y2_hat - a*y1R_hat])
    """
    return float(0.5 * ((yB_hat - a * y1L_hat) + (y2_hat - a * y1R_hat)))


# ----------------------------------- main ----------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Plot ROI intensity across baseline + img1 + img2, with optional affine transform on img1."
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
                        help="Optional custom tag to include in the plot title (prefixes output filenames too).")
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
                        help="Number of final baseline points for baseline-line fit. If 0, uses all baseline points. "
                             "NOTE: if baseline has only 1 volume, we use a constant at the boundary.")
    parser.add_argument("--img2-head-n", type=int, default=8,
                        help="Number of initial img2 points for img2-line fit (default 8).")

    # Transform choice
    parser.add_argument("--norm-method",
                        choices=["none", "affine2boundline"],
                        default="none",
                        help="Transform applied to img1 only. "
                             "'affine2boundline' enforces C0 continuity (using fitted lines) at available boundaries.")

    # Affine source signal (for computing a,c)
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

    # Expected/bounded a
    parser.add_argument("--a-expected-from-npro", action="store_true",
                        help="Compute a_expected = NPro(img2)/NPro(img1) (requires method sidecars).")
    parser.add_argument("--a-expected", type=float, default=None,
                        help="Override expected a (e.g., 3.0).")
    parser.add_argument("--a-bound-scale", type=float, default=25.0,
                        help="Bounds factor for a around expected: [a_expected/scale, a_expected*scale]. "
                             "Only applied if a_expected is available. Default 25.0 (very loose).")

    # Solver controls
    parser.add_argument("--affine-solver", choices=["exact", "robust"], default="exact",
                        help="exact: use exact C0 solution when possible; "
                             "robust: constrained grid search if exact violates constraints.")
    parser.add_argument("--robust-grid-n", type=int, default=401,
                        help="Number of grid points for robust search over a (default 401).")

    # HARD std constraint (feasibility)
    parser.add_argument("--std-max-ratio", type=float, default=None,
                        help="HARD constraint: reject candidate if std(transformed img1)/std_target > this. "
                             "std_target is TRUE-pooled from baseline_tail + img2_head (as available).")
    parser.add_argument("--std-use", choices=["all", "headtail", "all+headtail"], default="all+headtail",
                        help="Which transformed img1 region(s) to test against std constraint.")

    # Physical-ish constraints
    parser.add_argument("--a-positive", action="store_true",
                        help="Require a > 0 (reject otherwise).")
    parser.add_argument("--y-positive", action="store_true",
                        help="Require transformed ROI curve y1' to be >= --y-min for all timepoints.")
    parser.add_argument("--y-min", type=float, default=0.0,
                        help="Minimum allowed transformed ROI value when --y-positive is set (default 0).")

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

    # ---------------- ROI mask ----------------
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

    roi_mask = make_spherical_mask((nx, ny, nz), radius_vox=args.radius, center_xyz=(cx, cy, cz))
    nib.save(nib.Nifti1Image(roi_mask, img1.affine), str(mask_out))

    # ---------------- Global mask (for affine-source=global) ----------------
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

    # ---------------- TRs + time axes ----------------
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

    x_scale = 60.0 if args.time_unit == "min" else 1.0
    x_label = "Time (min)" if args.time_unit == "min" else "Time (s)"

    # ---------------- ROI signals (always plotted) ----------------
    y1_roi = mean_roi_timeseries(data1, roi_mask)
    y2_roi = mean_roi_timeseries(data2, roi_mask) if data2 is not None else None
    yb_roi = mean_roi_timeseries(datab, roi_mask) if datab is not None else None

    # ---------------- Concatenated time axes for plotting ----------------
    segments_t = []
    segments_y = []
    labels = []
    t_cursor = 0.0

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

    # ---------------- affine source signals ----------------
    def get_affine_signal(data_4d: np.ndarray) -> np.ndarray:
        if args.affine_source == "roi":
            return mean_roi_timeseries(data_4d, roi_mask)
        if args.affine_source == "global":
            return reduce_timeseries_from_mask(data_4d, global_mask, args.reducer)
        if args.affine_source == "nonzero":
            return reduce_timeseries_from_nonzero(data_4d, args.reducer, args.nonzero_eps)
        raise ValueError("Unknown affine source")

    # ---------------- Version 2 affine2boundline ----------------
    a = 1.0
    c = 0.0
    ok = True
    msg = "none"

    a_expected = None
    a_bounds = None
    a_exact = None
    c_exact = None

    left_resid = None
    right_resid = None

    std_target = None
    std_all = None
    std_head = None
    std_tail = None
    std_ratio_maxwin = None

    if args.norm_method == "affine2boundline":
        has_left = datab is not None
        has_right = data2 is not None
        if not (has_left or has_right):
            raise ValueError("affine2boundline requires at least one of: --img2-baseline or img_2")

        s1 = get_affine_signal(data1)
        sB = get_affine_signal(datab) if has_left else None
        s2 = get_affine_signal(data2) if has_right else None

        stable_n = int(args.stable_n)
        tail_n = int(args.tail_n)
        head2_n = int(args.img2_head_n)
        base_tail_n = int(args.base_tail_n)

        if stable_n < 2 or stable_n > s1.size:
            raise ValueError(f"--stable-n must be in [2..len(img1)] got {stable_n}")
        if tail_n < 2 or tail_n > s1.size:
            raise ValueError(f"--tail-n must be in [2..len(img1)] got {tail_n}")
        if has_right:
            if head2_n < 2 or head2_n > s2.size:
                raise ValueError(f"--img2-head-n must be in [2..len(img2)] got {head2_n}")
        if has_left:
            if base_tail_n <= 0:
                base_tail_n = int(sB.size)
            if base_tail_n > sB.size or base_tail_n < 1:
                raise ValueError(f"--base-tail-n must be in [1..len(baseline)] got {base_tail_n}")

        # concatenated time arrays
        if has_left:
            tb_abs_fit = segments_t[0]
            t1_abs_fit = segments_t[1]
            tL = float(t1_abs_fit[0])
        else:
            tb_abs_fit = None
            t1_abs_fit = segments_t[0]
            tL = None

        if has_right:
            idx2 = 2 if has_left else 1
            t2_abs_fit = segments_t[idx2]
            tR = float(t2_abs_fit[0])
        else:
            t2_abs_fit = None
            tR = None

        # left boundary
        if has_left:
            tB_win = tb_abs_fit[-base_tail_n:]
            yB_win = sB[-base_tail_n:]
            if tB_win.size >= 2:
                mB, bB = fit_line(tB_win, yB_win)
                yB_hat = eval_line(mB, bB, tL)
                left_line_desc = f"line_fit(n={tB_win.size})"
            else:
                mB, bB = 0.0, float(yB_win[-1])
                yB_hat = float(yB_win[-1])
                left_line_desc = "single_point_const"

            t1L_win = t1_abs_fit[:stable_n]
            y1L_win = s1[:stable_n]
            m1L, b1L = fit_line(t1L_win, y1L_win)
            y1L_hat = eval_line(m1L, b1L, tL)
        else:
            yB_hat = None
            y1L_hat = None
            left_line_desc = None

        # right boundary
        if has_right:
            t1R_win = t1_abs_fit[-tail_n:]
            y1R_win = s1[-tail_n:]
            m1R, b1R = fit_line(t1R_win, y1R_win)
            y1R_hat = eval_line(m1R, b1R, tR)

            t2_win = t2_abs_fit[:head2_n]
            y2_win = s2[:head2_n]
            m2, b2 = fit_line(t2_win, y2_win)
            y2_hat = eval_line(m2, b2, tR)
        else:
            y1R_hat = None
            y2_hat = None

        # a_expected / bounds
        if args.a_expected is not None:
            a_expected = float(args.a_expected)
        elif args.a_expected_from_npro and has_right:
            a_expected = float(get_npro(img2_path) / get_npro(img1_path))

        if a_expected is not None and args.a_bound_scale and args.a_bound_scale > 0:
            a_bounds = (a_expected / float(args.a_bound_scale), a_expected * float(args.a_bound_scale))

        # TRUE pooled std_target from target-space windows
        pool_parts = []
        if has_left:
            pool_parts.append(yB_win.copy())
        if has_right:
            pool_parts.append(y2_win.copy())
        std_target = pooled_std(*pool_parts)
        std_constraint_active = (args.std_max_ratio is not None) and np.isfinite(std_target) and (std_target > 0)

        # helpers for std tests on transformed affine-source signal
        s1_head = s1[:stable_n]
        s1_tail = s1[-tail_n:]

        def compute_std_ratio_maxwin(a_try: float, c_try: float) -> tuple[float, float, float, float]:
            y_all1 = a_try * s1 + c_try
            sh = safe_std(a_try * s1_head + c_try)
            st = safe_std(a_try * s1_tail + c_try)
            sa = safe_std(y_all1)

            ratios = []
            if std_target is not None and np.isfinite(std_target) and std_target > 0:
                if args.std_use in ("all", "all+headtail"):
                    if np.isfinite(sa):
                        ratios.append(sa / std_target)
                if args.std_use in ("headtail", "all+headtail"):
                    if np.isfinite(sh):
                        ratios.append(sh / std_target)
                    if np.isfinite(st):
                        ratios.append(st / std_target)
            rmax = max(ratios) if ratios else float("nan")
            return sa, sh, st, rmax

        def passes_constraints(a_try: float, c_try: float) -> tuple[bool, str, float, float, float, float]:
            if args.a_positive and a_try <= 0:
                return False, "a_nonpositive", float("nan"), float("nan"), float("nan"), float("nan")

            if args.y_positive:
                y1_roi_x = a_try * y1_roi + c_try
                if np.nanmin(y1_roi_x) < float(args.y_min):
                    return False, "y_min_violation", float("nan"), float("nan"), float("nan"), float("nan")

            sa, sh, st, rmax = compute_std_ratio_maxwin(a_try, c_try)
            if std_constraint_active:
                if not np.isfinite(rmax):
                    return False, "std_ratio_nan", sa, sh, st, rmax
                if rmax > float(args.std_max_ratio):
                    return False, "std_ratio_exceeded", sa, sh, st, rmax

            return True, "ok", sa, sh, st, rmax

        # --- Step 1: exact solution when both boundaries exist ---
        if has_left and has_right:
            a_exact, c_exact = solve_affine_from_two_constraints(y1L_hat, yB_hat, y1R_hat, y2_hat)
            feasible, why, sa, sh, st, rmax = passes_constraints(a_exact, c_exact)

            # enforce bounds if specified (treat bounds as a constraint)
            if feasible and (a_bounds is not None):
                lo, hi = a_bounds
                if not (lo <= a_exact <= hi):
                    feasible = False
                    why = "a_out_of_bounds"

            if args.affine_solver == "exact":
                a, c = a_exact, c_exact
                ok = True
                msg = "exact_C0_from_lines_both"
                if args.a_positive and a <= 0:
                    ok = False
                    msg = "exact_produced_nonpositive_a"
            else:
                if feasible:
                    a, c = a_exact, c_exact
                    ok = True
                    msg = "robust_used_exact_C0_both"
                    std_all, std_head, std_tail, std_ratio_maxwin = sa, sh, st, rmax
                else:
                    # --- Step 2: robust constrained search minimizing BOTH residuals ---
                    # a-grid
                    if a_bounds is not None:
                        a_lo, a_hi = float(a_bounds[0]), float(a_bounds[1])
                    else:
                        if a_expected is not None and np.isfinite(a_expected) and a_expected > 0:
                            a_lo, a_hi = a_expected / 25.0, a_expected * 25.0
                        else:
                            a_lo, a_hi = 0.05, 50.0
                    if args.a_positive:
                        a_lo = max(a_lo, 1e-6)

                    ngrid = int(args.robust_grid_n)
                    if ngrid < 11:
                        raise ValueError("--robust-grid-n must be >= 11")
                    grid = np.linspace(a_lo, a_hi, ngrid, dtype=np.float64)

                    best = None  # (cost, a, c, lres, rres, sa, sh, st, rmax, why)
                    for a_try in grid:
                        c_try = c_least_squares_two_boundaries(a_try, y1L_hat, yB_hat, y1R_hat, y2_hat)
                        feasible2, why2, sa2, sh2, st2, rmax2 = passes_constraints(a_try, c_try)
                        if not feasible2:
                            continue

                        lres = (a_try * y1L_hat + c_try) - yB_hat
                        rres = (a_try * y1R_hat + c_try) - y2_hat
                        cost = float(lres) ** 2 + float(rres) ** 2

                        # light preference near a_expected (won't override feasibility)
                        if a_expected is not None and np.isfinite(a_expected) and a_expected > 0:
                            cost += 1e-6 * ((a_try - a_expected) / a_expected) ** 2

                        if best is None or cost < best[0]:
                            best = (cost, float(a_try), float(c_try), float(lres), float(rres), sa2, sh2, st2, rmax2, why2)

                    if best is None:
                        # last-resort fallback: clip exact a into bounds (if any), and balance c
                        ok = False
                        msg = f"robust_no_feasible_solution (exact infeasible: {why})"
                        if a_bounds is not None:
                            lo, hi = a_bounds
                            a = float(np.clip(a_exact, lo, hi))
                        else:
                            a = float(a_exact)
                        c = c_least_squares_two_boundaries(a, y1L_hat, yB_hat, y1R_hat, y2_hat)
                    else:
                        ok = True
                        _, a, c, lres, rres, sa2, sh2, st2, rmax2, why2 = best
                        msg = f"robust_constrained_min_both_resid (exact infeasible: {why})"
                        std_all, std_head, std_tail, std_ratio_maxwin = sa2, sh2, st2, rmax2

        # --- One-boundary cases ---
        else:
            if args.affine_solver == "exact":
                a = 1.0
                if has_left:
                    c = yB_hat - a * y1L_hat
                    msg = f"exact_left_only_anchor_{left_line_desc}"
                else:
                    c = y2_hat - a * y1R_hat
                    msg = "exact_right_only_anchor"
                ok = True
            else:
                # robust grid over a, anchor boundary exactly; minimize nothing else (only constraints)
                if a_bounds is not None:
                    a_lo, a_hi = float(a_bounds[0]), float(a_bounds[1])
                else:
                    if a_expected is not None and np.isfinite(a_expected) and a_expected > 0:
                        a_lo, a_hi = a_expected / 25.0, a_expected * 25.0
                    else:
                        a_lo, a_hi = 0.05, 50.0
                if args.a_positive:
                    a_lo = max(a_lo, 1e-6)

                grid = np.linspace(a_lo, a_hi, int(args.robust_grid_n), dtype=np.float64)
                best = None
                for a_try in grid:
                    if has_left:
                        c_try = yB_hat - a_try * y1L_hat
                    else:
                        c_try = y2_hat - a_try * y1R_hat
                    feasible2, why2, sa2, sh2, st2, rmax2 = passes_constraints(a_try, c_try)
                    if not feasible2:
                        continue
                    # cost just mild pull to a_expected if present; otherwise 0
                    cost = 0.0
                    if a_expected is not None and np.isfinite(a_expected) and a_expected > 0:
                        cost += 1e-6 * ((a_try - a_expected) / a_expected) ** 2
                    if best is None or cost < best[0]:
                        best = (cost, float(a_try), float(c_try), sa2, sh2, st2, rmax2, why2)

                if best is None:
                    ok = False
                    msg = "robust_no_feasible_one_boundary"
                    a = 1.0
                    if has_left:
                        c = yB_hat - a * y1L_hat
                    else:
                        c = y2_hat - a * y1R_hat
                else:
                    ok = True
                    _, a, c, sa2, sh2, st2, rmax2, why2 = best
                    msg = "robust_one_boundary_feasible"
                    std_all, std_head, std_tail, std_ratio_maxwin = sa2, sh2, st2, rmax2

        # apply affine to ROI curve for plotting
        y1_roi_xform = a * y1_roi + c
        idx_img1 = 1 if has_left else 0
        segments_y[idx_img1] = y1_roi_xform
        labels[idx_img1] = f"img1_ROI→a*y+c (a={a:.6g}, c={c:.6g})"

        # residuals on actual line-C0 constraints
        if has_left:
            left_resid = (a * y1L_hat + c) - yB_hat
        if has_right:
            right_resid = (a * y1R_hat + c) - y2_hat

    # ---------------- Plot ----------------
    t_all = np.concatenate(segments_t, axis=0)
    y_all = np.concatenate(segments_y, axis=0)

    plt.figure()
    plt.plot(t_all / x_scale, y_all)

    if datab is not None:
        plt.axvline((t1_abs[0]) / x_scale, linestyle="--")
    if data2 is not None:
        plt.axvline((t2_abs[0]) / x_scale, linestyle="--")

    plt.xlabel(x_label)
    plt.ylabel("Mean ROI intensity")

    # Title
    title_tag = args.title_tag.strip() if (args.title_tag and args.title_tag.strip()) else "ROI intensity"
    if args.center is None:
        coord_line = f"({center_label};r={args.radius:g})"
    else:
        coord_line = f"({int(round(cx))},{int(round(cy))},{int(round(cz))};r={args.radius:g})"

    ax = plt.gca()
    if args.title_mode == "compact":
        ax.set_title(f"{title_tag} {coord_line}", fontsize=12)
    else:
        title = f"{title_tag} {coord_line}\nROI mean intensity"
        title += f"\n(mode={args.norm_method}, solver={args.affine_solver}, affine-source={args.affine_source}, reducer={args.reducer}, global-mask={global_mask_note})"
        if args.norm_method == "affine2boundline":
            title += f"\na={a:.6g}, c={c:.6g}, ok={ok}, msg={msg}"
            if a_exact is not None and c_exact is not None:
                title += f"\na_exact={a_exact:.6g}, c_exact={c_exact:.6g}"
            if a_expected is not None:
                title += f"\na_expected={a_expected:.6g}"
            if a_bounds is not None:
                title += f"  a_bounds=[{a_bounds[0]:.6g},{a_bounds[1]:.6g}]"
            if left_resid is not None:
                title += f"\nleft_resid(line-C0)={left_resid:.6g}"
            if right_resid is not None:
                title += f"\nright_resid(line-C0)={right_resid:.6g}"
            if std_target is not None and np.isfinite(std_target):
                title += f"\nstd_target(pooled)={std_target:.6g}"
            if std_ratio_maxwin is not None and np.isfinite(std_ratio_maxwin):
                title += f"  std_ratio_maxwin={std_ratio_maxwin:.6g}"
        ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(str(out_plot), dpi=150)

    # CSV
    if csv_out:
        with open(csv_out, "w") as f:
            f.write(f"time_{args.time_unit},roi_mean_intensity\n")
            for ti, yi in zip(t_all / x_scale, y_all):
                f.write(f"{ti:.9g},{yi:.9g}\n")

    # ---------------- prints ----------------
    if args.norm_method == "none":
        print("[mode] ROI plot (no transform)")
    else:
        print("[mode] ROI plot; baseline/img2 fixed; img1 transformed only")

    print(f"       segments: {labels}")
    print(f"[roi] center={center_label}, radius={args.radius:g}, voxels={int(roi_mask.sum())}")
    print(f"[affine-source] {args.affine_source}  (reducer={args.reducer}, nonzero-eps={args.nonzero_eps:g}, global-mask={global_mask_note})")

    if args.norm_method == "affine2boundline":
        if a_expected is not None and a_bounds is not None:
            print(f"[a] a_expected={a_expected:.10g}  a_bounds=[{a_bounds[0]:.10g},{a_bounds[1]:.10g}]")
        elif a_expected is not None:
            print(f"[a] a_expected={a_expected:.10g}")

        if a_exact is not None and c_exact is not None:
            print(f"[exact] a_exact={a_exact:.10g}  c_exact={c_exact:.10g}")

        print(f"[img1-transform] method=affine2boundline  solver={args.affine_solver}")
        print(f"                a={a:.10g}  c={c:.10g}  ok={ok}  msg={msg}")

        if left_resid is not None:
            print(f"[line-C0] left_resid(a*y1Lhat+c - yBhat)={left_resid:.6g}")
        if right_resid is not None:
            print(f"[line-C0] right_resid(a*y1Rhat+c - y2hat)={right_resid:.6g}")

        if std_target is not None and np.isfinite(std_target):
            print(f"[std] std_target_pooled={std_target:.6g}  (pooled baseline_tail + img2_head where available)")

        if std_all is not None and np.isfinite(std_all):
            print(f"[std] std_img1_all_post={std_all:.6g}")
        if std_head is not None and np.isfinite(std_head):
            print(f"[std] std_img1_head_post={std_head:.6g} (stable-n={args.stable_n})")
        if std_tail is not None and np.isfinite(std_tail):
            print(f"[std] std_img1_tail_post={std_tail:.6g} (tail-n={args.tail_n})")
        if std_ratio_maxwin is not None and np.isfinite(std_ratio_maxwin):
            print(f"[std] std_ratio_maxwin={std_ratio_maxwin:.6g}  (std-use={args.std_use}, std-max-ratio={args.std_max_ratio})")
        elif args.std_max_ratio is not None:
            if std_target is None or not np.isfinite(std_target) or std_target <= 0:
                print("[std] NOTE: std-max-ratio requested but std_target could not be computed (pooled sample count <2). Constraint disabled.")

        if baseline_is_3d:
            print("[baseline] NOTE: baseline is 3D (single volume) - left boundary uses single_point_const if base-tail-n=1.")

    print(f"[out] roi-mask: {mask_out}")
    print(f"[out] plot    : {out_plot}")
    if csv_out:
        print(f"[out] csv     : {csv_out}")


if __name__ == "__main__":
    main()