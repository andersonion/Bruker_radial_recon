#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Optional scipy (morphology + connected components + bounded optimization)
try:
    import scipy.ndimage as ndi
    from scipy.optimize import minimize
except Exception:
    ndi = None
    minimize = None


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


# ----------------------------- helpers: method parsing ----------------------------- #

def _parse_method_value(method_text: str, key: str, method_path: Path) -> float:
    m = re.search(rf"^{re.escape(key)}\s*=\s*(.*)$", method_text, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"Could not find {key} in {method_path}")
    val = m.group(1).strip()

    # Bruker sometimes uses "##$KEY=( 1 ) <value>" style
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
            f"NPro requires method sidecars. "
            f"Could not find {_strip_nii_extensions(img_path)}.method next to {img_path}."
        )
    return parse_bruker_method_for_npro(method_path)


# ----------------------------- ROI + global signals ----------------------------- #

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


def _reduce_ts(vox_by_t: np.ndarray, mode: str) -> np.ndarray:
    if mode == "mean":
        return np.nanmean(vox_by_t, axis=0)
    if mode == "median":
        return np.nanmedian(vox_by_t, axis=0)
    raise ValueError(f"Unknown reducer: {mode}")


def global_timeseries(data_4d: np.ndarray,
                      global_mask_3d: np.ndarray | None,
                      mode: str) -> np.ndarray:
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
    return _reduce_ts(vox, mode)


def nonzero_timeseries(data_4d: np.ndarray,
                       eps: float,
                       global_mask_3d: np.ndarray | None,
                       mode: str) -> np.ndarray:
    """
    Build a time series using voxels that are > eps (per-voxel threshold applied on time-mean),
    optionally intersected with global_mask_3d.
    """
    if data_4d.ndim != 4:
        raise ValueError(f"Expected 4D data, got shape {data_4d.shape}")

    mean_vol = np.nanmean(data_4d, axis=3)
    nz_mask = mean_vol > eps
    if global_mask_3d is not None:
        if global_mask_3d.shape != data_4d.shape[:3]:
            raise ValueError(f"Global mask shape {global_mask_3d.shape} != data spatial shape {data_4d.shape[:3]}")
        nz_mask = nz_mask & (global_mask_3d > 0)

    vox = data_4d[nz_mask, :]
    if vox.size == 0:
        raise RuntimeError("nonzero mask selected 0 voxels; try lowering --nonzero-eps or disable masking.")
    return _reduce_ts(vox, mode)


# ----------------------------- auto brain mask (global scaling) ----------------------------- #

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


def generate_auto_brain_mask_from_mean(data_4d: np.ndarray,
                                      *,
                                      erosions: int = 3,
                                      dilations: int = 3,
                                      nbins: int = 256) -> np.ndarray:
    _require_scipy()

    mean_vol = np.nanmean(data_4d, axis=3)
    thr = otsu_threshold(mean_vol, nbins=nbins)
    m = mean_vol > thr

    struct = ndi.generate_binary_structure(3, 2)

    if erosions > 0:
        m = ndi.binary_erosion(m, structure=struct, iterations=erosions)

    lbl, n = ndi.label(m, structure=struct)
    if n < 1:
        # fallback: skip erosion
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


# ----------------------------- affine (Version 2): continuity + slope penalty ----------------------------- #

def fit_line(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    A = np.vstack([t, np.ones_like(t)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


def window_slice(n: int, head: bool, length: int) -> slice:
    if n < 1:
        raise ValueError("Window n must be >= 1")
    if n > length:
        raise ValueError(f"Window n={n} > length={length}")
    return slice(0, n) if head else slice(length - n, length)


def affine2boundline_solve(
    *,
    t1: np.ndarray, y1_src: np.ndarray,
    tb: np.ndarray, yb_src: np.ndarray,
    t2: np.ndarray, y2_src: np.ndarray,
    stable_n: int,
    tail_n: int,
    head2_n: int,
    a_expected: float,
    a_bound_scale: float,
    w_slope_left: float,
    w_slope_right: float,
) -> tuple[float, float, dict]:
    """
    Solve for y1' = a*y1 + c such that:
      - hard(ish) continuity in means at both boundaries
      - soft slope matching via penalties
      - a bounded around a_expected

    We implement as bounded minimization of:
      (a*mu1L+c - muB)^2 + (a*mu1R+c - mu2)^2
      + wL*(a*m1L - mB)^2 + wR*(a*m1R - m2)^2
    """
    # windows
    sL = window_slice(stable_n, True, len(y1_src))
    sR = window_slice(tail_n, False, len(y1_src))
    s2 = window_slice(head2_n, True, len(y2_src))

    # boundary means (targets)
    muB = float(np.mean(yb_src))  # baseline assumed stable across its available points
    mu1L = float(np.mean(y1_src[sL]))
    mu1R = float(np.mean(y1_src[sR]))
    mu2 = float(np.mean(y2_src[s2]))

    # slopes (line fits)
    m1L, _ = fit_line(t1[sL], y1_src[sL])
    m1R, _ = fit_line(t1[sR], y1_src[sR])

    # baseline slope: fit across baseline window (all baseline points)
    mB, _ = fit_line(tb, yb_src)

    # img2 head slope
    m2, _ = fit_line(t2[s2], y2_src[s2])

    a_min = a_expected / a_bound_scale
    a_max = a_expected * a_bound_scale
    if a_min <= 0:
        a_min = 1e-12

    # objective
    def obj(x):
        a, c = float(x[0]), float(x[1])
        valL = (a * mu1L + c - muB) ** 2
        valR = (a * mu1R + c - mu2) ** 2
        slopeL = (a * m1L - mB) ** 2
        slopeR = (a * m1R - m2) ** 2
        return valL + valR + w_slope_left * slopeL + w_slope_right * slopeR

    # Initial guess: enforce left mean exactly with expected a
    c0 = muB - a_expected * mu1L
    x0 = np.array([a_expected, c0], dtype=float)

    if minimize is not None:
        res = minimize(
            obj,
            x0=x0,
            method="L-BFGS-B",
            bounds=[(a_min, a_max), (None, None)],
        )
        a_opt, c_opt = float(res.x[0]), float(res.x[1])
        ok = bool(res.success)
        msg = str(res.message)
    else:
        # Fallback: brute 1D grid in a; solve best c analytically (least squares) for each a
        # c(a) = avg( (muB - a*mu1L), (mu2 - a*mu1R) )
        ok = True
        msg = "scipy unavailable; used grid search over a"
        grid = np.linspace(a_min, a_max, 1001)
        best = None
        for a in grid:
            c = 0.5 * ((muB - a * mu1L) + (mu2 - a * mu1R))
            f = obj((a, c))
            if best is None or f < best[0]:
                best = (f, a, c)
        _, a_opt, c_opt = best

    dbg = {
        "muB": muB,
        "mu1L": mu1L,
        "mu1R": mu1R,
        "mu2": mu2,
        "mB": mB,
        "m1L": m1L,
        "m1R": m1R,
        "m2": m2,
        "a_expected": a_expected,
        "a_bounds": (a_min, a_max),
        "ok": ok,
        "msg": msg,
        "stable_n": stable_n,
        "tail_n": tail_n,
        "head2_n": head2_n,
    }
    return a_opt, c_opt, dbg


# ----------------------------------- main ----------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot ROI intensity (baseline + img2 fixed; img1 transformed only for certain methods). "
            "Supports TR from Bruker .method sidecars and optional auto global mask creation."
        )
    )

    parser.add_argument("img_1", help="4D NIfTI (main run)")
    parser.add_argument("img_2", nargs="?", default=None, help="Second 4D NIfTI (optional)")

    parser.add_argument("--img2-baseline", default=None,
                        help="Baseline NIfTI acquired with same params as img_2 (before img_1). "
                             "Can be 3D or 4D. Used as stable baseline anchor.")

    parser.add_argument("--radius", type=float, required=True, help="ROI radius in voxels")
    parser.add_argument("--center", nargs=3, type=float, default=None,
                        metavar=("X", "Y", "Z"),
                        help="ROI center in voxel indices (0-based). If omitted, uses image center.")

    parser.add_argument("--tr1", type=float, default=None, help="TR seconds for img_1 (optional if method exists)")
    parser.add_argument("--tr2", type=float, default=None, help="TR seconds for img_2 (optional if method exists)")
    parser.add_argument("--trb", type=float, default=None, help="TR seconds for --img2-baseline (optional if method exists)")

    parser.add_argument("--stable-n", type=int, default=8,
                        help="Number of initial points of img_1 assumed stable (default 8).")
    parser.add_argument("--norm-n", type=int, default=5,
                        help="Window size for tail/head operations (default 5).")

    parser.add_argument("--time-unit", choices=["s", "min"], default="s",
                        help="X axis units for plotting (and CSV if written). Default seconds.")

    parser.add_argument(
        "--title-tag",
        default=None,
        help="Optional custom tag prepended to output filenames and included in title.",
    )
    parser.add_argument(
        "--title-mode",
        choices=["full", "compact"],
        default="full",
        help="Title style. 'compact' shows only tag + (x,y,z;r=R). 'full' adds method/debug lines.",
    )

    # Global signal controls (used by affine-source and other reporting)
    parser.add_argument("--global-mode", choices=["mean", "median"], default="mean",
                        help="Reducer for global/nonzero signals (default mean).")
    parser.add_argument("--global-mask", default=None,
                        help="3D mask NIfTI for global/nonzero signals, OR 'auto' to generate from selected image. "
                             "If omitted, mask is OFF (global uses all voxels; nonzero uses nonzero-only).")
    parser.add_argument("--auto-mask-source", choices=["img2", "img2_baseline", "img1"], default="img2",
                        help="If --global-mask auto, which image to build it from (default img2).")
    parser.add_argument("--auto-mask-erosions", type=int, default=3, help="Erosion iterations for auto mask (default 3)")
    parser.add_argument("--auto-mask-dilations", type=int, default=3, help="Dilation iterations for auto mask (default 3)")
    parser.add_argument("--auto-mask-hist-bins", type=int, default=256, help="Histogram bins for Otsu (default 256)")
    parser.add_argument("--auto-mask-out", default=None,
                        help="Where to write auto-generated global mask. Default auto name. Set to 'none' to skip.")

    # Nonzero options
    parser.add_argument("--nonzero-eps", type=float, default=0.0,
                        help="Threshold on time-mean volume for nonzero affine-source (default 0).")

    # Normalization / transform options
    parser.add_argument(
        "--norm-method",
        choices=["none", "lastfirst", "projections", "slope", "both", "affine2bound", "affine2boundline"],
        default="none",
        help=(
            "How to handle img1/img2 scaling. "
            "affine2bound and affine2boundline transform img1 only while keeping baseline+img2 fixed."
        ),
    )
    parser.add_argument(
        "--affine-source",
        choices=["roi", "global", "nonzero"],
        default="roi",
        help="Which signal to use to compute affine parameters for img1 transforms.",
    )
    parser.add_argument("--a-bound-scale", type=float, default=2.0,
                        help="Bound 'a' around NPro-expected: [a_exp/scale, a_exp*scale]. Default 2.0.")
    parser.add_argument("--line-weight-left", type=float, default=0.1,
                        help="Slope penalty weight on left boundary (default 0.1).")
    parser.add_argument("--line-weight-right", type=float, default=0.1,
                        help="Slope penalty weight on right boundary (default 0.1).")

    # Outputs
    parser.add_argument("--mask-out", default=None, help="Output ROI mask NIfTI (default auto-named)")
    parser.add_argument("--out", default=None, help="Output plot PNG (default auto-named)")
    parser.add_argument("--csv-out", default=None, help="Optional CSV output")

    args = parser.parse_args()

    # ---------------- Filename prefixing from title tag ----------------
    title_tag = args.title_tag.strip() if (args.title_tag is not None and args.title_tag.strip() != "") else ""
    file_prefix = (_slugify(title_tag) + "_") if title_tag != "" else ""

    def _prefix_path(p: str | None) -> str | None:
        if p is None:
            return None
        pp = Path(p)
        if file_prefix == "" or pp.name.startswith(file_prefix):
            return str(pp)
        return str(pp.with_name(file_prefix + pp.name))

    # ---------------- Load img1 ----------------
    img1_path = Path(args.img_1)
    img1 = nib.load(str(img1_path))
    data1 = img1.get_fdata()
    if data1.ndim != 4:
        raise ValueError(f"img_1 must be 4D, got shape {data1.shape}")
    nx, ny, nz = data1.shape[:3]

    # ---------------- Load img2 (optional) ----------------
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

    # ---------------- Load baseline (optional; allow 3D or 4D) ----------------
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

    # ---------------- ROI center ----------------
    if args.center is None:
        cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
        center_label = "center"
    else:
        cx, cy, cz = args.center
        if not (0 <= cx < nx and 0 <= cy < ny and 0 <= cz < nz):
            raise ValueError(f"Center {args.center} out of bounds: [0..{nx-1}]x[0..{ny-1}]x[0..{nz-1}]")
        center_label = f"{int(round(cx))}_{int(round(cy))}_{int(round(cz))}"

    roi_tag = f"xyz_{center_label}_r{args.radius:g}"
    out_plot = _prefix_path(args.out or f"roi_intensity_{roi_tag}.png")
    mask_out = _prefix_path(args.mask_out or f"roi_mask_{roi_tag}.nii.gz")
    if args.csv_out is not None:
        args.csv_out = _prefix_path(args.csv_out)

    # ---------------- ROI mask output ----------------
    roi_mask = make_spherical_mask((nx, ny, nz), radius_vox=args.radius, center_xyz=(cx, cy, cz))
    nib.save(nib.Nifti1Image(roi_mask, img1.affine), mask_out)

    # ---------------- Decide/build global mask (used by affine-source global/nonzero) ----------------
    global_mask = None
    global_mask_note = "OFF"
    auto_mask_written_path = None

    if args.global_mask is not None:
        if args.global_mask.lower() == "auto":
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

            # Write mask unless disabled
            if args.auto_mask_out is None:
                auto_out = src_path.parent / f"{file_prefix}auto_global_mask_from_{_strip_nii_extensions(src_path)}.nii.gz"
            elif str(args.auto_mask_out).lower() == "none":
                auto_out = None
            else:
                auto_out = Path(_prefix_path(str(args.auto_mask_out)))

            if auto_out is not None:
                nib.save(nib.Nifti1Image(global_mask.astype(np.uint8), src_aff), str(auto_out))
                auto_mask_written_path = str(auto_out)

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

    # ---------------- Time axes (TR) ----------------
    tr1 = get_tr_seconds(img1_path, args.tr1)
    t1 = np.arange(data1.shape[3], dtype=float) * tr1

    tr2 = None
    if data2 is not None:
        tr2 = get_tr_seconds(img2_path, args.tr2)
        t2 = np.arange(data2.shape[3], dtype=float) * tr2
    else:
        t2 = None

    # baseline time
    if datab is not None:
        if baseline_is_3d:
            trb = tr1
        else:
            trb = get_tr_seconds(baseline_path, args.trb)
        tb = np.arange(datab.shape[3], dtype=float) * trb
    else:
        trb = None
        tb = None

    x_scale = 60.0 if args.time_unit == "min" else 1.0
    x_label = "Time (min)" if args.time_unit == "min" else "Time (s)"

    # ---------------- ROI signals (what we always plot) ----------------
    y1_roi = mean_roi_timeseries(data1, roi_mask)
    y2_roi = mean_roi_timeseries(data2, roi_mask) if data2 is not None else None
    yb_roi = mean_roi_timeseries(datab, roi_mask) if datab is not None else None

    # ---------------- Affine-source signals (what we use to fit transforms) ----------------
    def get_affine_source_ts(data_4d: np.ndarray, which: str) -> np.ndarray:
        if which == "roi":
            return mean_roi_timeseries(data_4d, roi_mask)
        if which == "global":
            return global_timeseries(data_4d, global_mask, args.global_mode)
        if which == "nonzero":
            return nonzero_timeseries(data_4d, args.nonzero_eps, global_mask, args.global_mode)
        raise ValueError(f"Unknown affine-source: {which}")

    y1_src = get_affine_source_ts(data1, args.affine_source)
    y2_src = get_affine_source_ts(data2, args.affine_source) if data2 is not None else None
    yb_src = get_affine_source_ts(datab, args.affine_source) if datab is not None else None

    # ---------------- Apply normalization / transforms ----------------
    # baseline + img2 are ALWAYS held fixed for affine methods; only img1 is transformed.
    img1_a = 1.0
    img1_c = 0.0
    affine_dbg = None

    # Helper: NPro expected 'a' (baseline/img2 vs img1)
    def expected_a_from_npro() -> float:
        # If method files exist, use NPro(img2)/NPro(img1); else fall back to 1.0
        if img2_path is None:
            return 1.0
        try:
            n1 = get_npro(img1_path)
            n2 = get_npro(img2_path)
            if n1 > 0 and n2 > 0:
                return float(n2 / n1)
        except Exception:
            pass
        return 1.0

    if args.norm_method == "affine2boundline":
        if datab is None or data2 is None:
            raise ValueError("affine2boundline requires BOTH --img2-baseline and img_2.")
        if args.stable_n < 2:
            raise ValueError("--stable-n must be >= 2 for affine2boundline.")
        if args.norm_n < 2:
            raise ValueError("--norm-n must be >= 2 for affine2boundline.")

        a_exp = expected_a_from_npro()
        img1_a, img1_c, affine_dbg = affine2boundline_solve(
            t1=t1, y1_src=y1_src,
            tb=tb, yb_src=yb_src,
            t2=t2, y2_src=y2_src,
            stable_n=args.stable_n,
            tail_n=args.norm_n,
            head2_n=args.norm_n,
            a_expected=a_exp,
            a_bound_scale=args.a_bound_scale,
            w_slope_left=args.line_weight_left,
            w_slope_right=args.line_weight_right,
        )

        # Apply transform to ROI trace for plotting
        y1_roi = img1_a * y1_roi + img1_c

    elif args.norm_method == "affine2bound":
        # Simple (unconstrained) mean-matching across both boundaries on chosen affine-source,
        # then apply to ROI. This is kept for backward compatibility.
        if datab is None or data2 is None:
            raise ValueError("affine2bound requires BOTH --img2-baseline and img_2.")
        if args.stable_n < 1 or args.stable_n > len(y1_src):
            raise ValueError(f"--stable-n must be in [1..len(img1)] = [1..{len(y1_src)}], got {args.stable_n}")
        if args.norm_n < 1 or args.norm_n > len(y1_src) or args.norm_n > len(y2_src):
            raise ValueError("--norm-n invalid for available timepoints.")

        muB = float(np.mean(yb_src))
        mu1L = float(np.mean(y1_src[:args.stable_n]))
        mu2 = float(np.mean(y2_src[:args.norm_n]))
        mu1R = float(np.mean(y1_src[-args.norm_n:]))

        # Solve exact 2-equation system if possible
        denom = (mu1L - mu1R)
        if abs(denom) < 1e-12:
            # fallback: use expected a and left mean
            img1_a = expected_a_from_npro()
            img1_c = muB - img1_a * mu1L
        else:
            img1_a = (muB - mu2) / denom
            img1_c = muB - img1_a * mu1L

        affine_dbg = {
            "muB": muB, "mu1L": mu1L, "mu1R": mu1R, "mu2": mu2,
            "a_expected": expected_a_from_npro(),
            "note": "affine2bound (unconstrained)",
        }

        y1_roi = img1_a * y1_roi + img1_c

    elif args.norm_method == "lastfirst":
        # ROI-based lastfirst, baseline/img2 fixed; scale img1 only so that:
        # (baseline mean) matches (img1 stable mean) AND (img2 head mean) matches (img1 tail mean),
        # then average the two implied scales.
        if args.stable_n < 1 or args.stable_n > len(y1_roi):
            raise ValueError(f"--stable-n must be in [1..len(img1)] = [1..{len(y1_roi)}], got {args.stable_n}")

        scale_left = 1.0
        scale_right = 1.0

        if yb_roi is not None:
            mu_b = float(np.mean(yb_roi))
            mu_1 = float(np.mean(y1_roi[:args.stable_n]))
            if abs(mu_1) < 1e-12:
                raise ValueError("Stable img1 ROI mean is ~0; cannot compute lastfirst.")
            scale_left = mu_b / mu_1

        if y2_roi is not None:
            mu_2 = float(np.mean(y2_roi[:max(1, min(args.norm_n, len(y2_roi)))]))
            mu_1t = float(np.mean(y1_roi[-max(1, min(args.norm_n, len(y1_roi))):]))
            if abs(mu_1t) < 1e-12:
                raise ValueError("Tail img1 ROI mean is ~0; cannot compute lastfirst.")
            scale_right = mu_2 / mu_1t

        if yb_roi is not None and y2_roi is not None:
            img1_a = 0.5 * (scale_left + scale_right)
        elif yb_roi is not None:
            img1_a = scale_left
        elif y2_roi is not None:
            img1_a = scale_right
        else:
            img1_a = 1.0

        y1_roi = img1_a * y1_roi
        img1_c = 0.0

    else:
        # Other older methods kept: projections/slope/both operate on img2 (legacy behavior).
        # We leave them intact, and they do not affect baseline/img1 for now.
        pass

    # ---------------- Build concatenated segments for plotting ----------------
    segments_t = []
    segments_y = []
    labels = []
    t_cursor = 0.0

    # Baseline segment
    if datab is not None:
        tb_abs = t_cursor + tb
        segments_t.append(tb_abs)
        segments_y.append(yb_roi)
        labels.append("baseline_ROI")
        t_cursor = tb_abs[-1] + trb

    # img1 segment (possibly transformed)
    t1_abs = t_cursor + t1
    segments_t.append(t1_abs)
    segments_y.append(y1_roi)
    if args.norm_method in ("affine2bound", "affine2boundline"):
        labels.append(f"img1_ROI→a*y+c (a={img1_a:.6g}, c={img1_c:.6g})")
    elif args.norm_method == "lastfirst" and abs(img1_a - 1.0) > 1e-12:
        labels.append(f"img1_ROI×{img1_a:.6g}")
    else:
        labels.append("img1_ROI")
    t_cursor = t1_abs[-1] + tr1

    # img2 segment
    if data2 is not None:
        t2_abs = t_cursor + t2
        segments_t.append(t2_abs)
        segments_y.append(y2_roi)
        labels.append("img2_ROI")

    t_all = np.concatenate(segments_t, axis=0)
    y_all = np.concatenate(segments_y, axis=0)

    # ---------------- Plot ----------------
    plt.figure()
    plt.plot(t_all / x_scale, y_all)

    # boundaries
    if datab is not None:
        plt.axvline((t1_abs[0]) / x_scale, linestyle="--")  # baseline->img1
    if data2 is not None:
        plt.axvline((segments_t[-1][0]) / x_scale, linestyle="--")  # img1->img2

    plt.xlabel(x_label)
    plt.ylabel("Mean ROI intensity")

    # Title formatting
    if title_tag != "":
        main_title = title_tag
    else:
        main_title = "ROI intensity"

    coord_line = f"({int(round(cx))},{int(round(cy))},{int(round(cz))};r={args.radius:g})" if args.center is not None else f"({center_label};r={args.radius:g})"
    ax = plt.gca()

    if args.title_mode == "compact":
        ax.set_title(f"{main_title} {coord_line}", fontsize=12)
    else:
        title = f"{main_title} {coord_line}\nROI mean intensity (center={center_label}, r={args.radius:g})"
        title += f"\nsegments: {', '.join(labels)}"
        title += f"\n[affine-source] {args.affine_source}  (reducer={args.global_mode}, nonzero-eps={args.nonzero_eps:g}, global-mask={global_mask_note})"
        if auto_mask_written_path is not None:
            title += f"\n[auto-mask] wrote: {Path(auto_mask_written_path).name}"
        if args.norm_method == "affine2boundline" and affine_dbg is not None:
            lo, hi = affine_dbg["a_bounds"]
            title += (
                f"\n[affine2boundline] a_expected={affine_dbg['a_expected']:.6g}, bounds=[{lo:.6g},{hi:.6g}], "
                f"a={img1_a:.6g}, c={img1_c:.6g}, ok={affine_dbg['ok']} ({affine_dbg['msg']})"
            )
            title += (
                f"\n[left means] baseline={affine_dbg['muB']:.6g}  img1_stable_pre={affine_dbg['mu1L']:.6g}  "
                f"img1_stable_post={(img1_a*affine_dbg['mu1L']+img1_c):.6g}"
            )
            title += (
                f"\n[right means] img2_head={affine_dbg['mu2']:.6g}  img1_tail_pre={affine_dbg['mu1R']:.6g}  "
                f"img1_tail_post={(img1_a*affine_dbg['mu1R']+img1_c):.6g}"
            )
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)

    # CSV
    if args.csv_out:
        with open(args.csv_out, "w") as f:
            f.write(f"time_{args.time_unit},roi_mean_intensity\n")
            for ti, yi in zip(t_all / x_scale, y_all):
                f.write(f"{ti:.9g},{yi:.9g}\n")

    # ---------------- Prints / QA ----------------
    if args.norm_method in ("affine2bound", "affine2boundline"):
        print("[mode] ROI plot; baseline+img2 fixed; img1 transformed only")
    else:
        print("[mode] ROI plot")

    print(f"       segments: {labels}")
    print(f"[roi] center={center_label}, radius={args.radius:g}, voxels={int(roi_mask.sum())}")
    print(f"[affine-source] {args.affine_source}  (reducer={args.global_mode}, nonzero-eps={args.nonzero_eps:g}, global-mask={global_mask_note})")
    if auto_mask_written_path is not None:
        print(f"[auto-mask] wrote: {auto_mask_written_path}")

    if args.norm_method == "affine2boundline" and affine_dbg is not None:
        lo, hi = affine_dbg["a_bounds"]
        print(f"[img1-transform] method=affine2boundline  a_expected={affine_dbg['a_expected']:.8g}  a_bounds=[{lo:.8g},{hi:.8g}]")
        print(f"                a={img1_a:.8g}  c={img1_c:.8g}  ok={affine_dbg['ok']}  msg={affine_dbg['msg']}")
        print(f"[left]  baseline_mean={affine_dbg['muB']:.6g}  img1_stable_pre={affine_dbg['mu1L']:.6g}  img1_stable_post={(img1_a*affine_dbg['mu1L']+img1_c):.6g}  resid={(img1_a*affine_dbg['mu1L']+img1_c)-affine_dbg['muB']:.6g}")
        print(f"[right] img2_head_mean={affine_dbg['mu2']:.6g}  img1_tail_pre={affine_dbg['mu1R']:.6g}  img1_tail_post={(img1_a*affine_dbg['mu1R']+img1_c):.6g}  resid={(img1_a*affine_dbg['mu1R']+img1_c)-affine_dbg['mu2']:.6g}")

    if args.norm_method == "affine2bound" and affine_dbg is not None:
        print(f"[img1-transform] method=affine2bound  a={img1_a:.8g}  c={img1_c:.8g}  (note: {affine_dbg.get('note','')})")

    print(f"[out] roi-mask: {mask_out}")
    print(f"[out] plot    : {out_plot}")
    if args.csv_out:
        print(f"[out] csv     : {args.csv_out}")


if __name__ == "__main__":
    main()