#!/usr/bin/env python3
"""
plot_roi_intensity.py

Plot ROI mean intensity across (optional) baseline + img1 + (optional) img2,
with optional affine transform applied to img1 ONLY.

Key features (what you’ve been iterating toward):
- ROI spherical mask from voxel center + radius
- Optional GLOBAL mask (FILE or AUTO via Otsu+CC+morphpy) used for global reducers
- Affine transform modes:
  - none
  - affine2boundline:
      Version 2 ("continuous line well-fit" / C0 at both boundaries using fitted lines)
      Supports:
        * With baseline + img2: constrain C0 at left (baseline->img1) and right (img1->img2)
        * WITHOUT baseline: constrain only right boundary (img1->img2) and choose c by one of:
            - c = 0 (default)
            - c = match_left_mean (use img1 head mean after scaling to match itself; effectively no-op for c)
            - c = preserve_min (choose c so transformed img1 stays nonnegative; requires --nonneg)
- Optional “robust” selection of (a,c) when bounds or penalties are enabled:
  - exact: solve constraints exactly (preferred)
  - robust: 1D search over 'a' within bounds, choose c that satisfies left constraint (if baseline)
            or c policy (if no baseline), and penalize:
              * boundary residuals
              * excessive std inflation relative to TRUE pooled std of (baseline-tail + img2-head)
                or (img2-head only if no baseline)
              * negativity if --nonneg is enabled
- The “TRUE POOLING (A)” change: std_target is computed from pooled samples
  (baseline tail + img2 head) together when baseline exists, not separately.

Examples:
  # Normal (baseline + block1 + block2), ROI-based affine source:
  python plot_roi_intensity.py block1.nii.gz block2.nii.gz \
      --img2-baseline baseline.nii.gz \
      --radius 2.5 --center 57 38 49 \
      --norm-method affine2boundline --affine-source roi \
      --title-tag "P26012003 CSF" --title-mode compact --time-unit min

  # Same, but affine params derived from NONZERO global reducer (no mask):
  python plot_roi_intensity.py block1.nii.gz block2.nii.gz \
      --img2-baseline baseline.nii.gz \
      --radius 2.5 --center 57 38 49 \
      --norm-method affine2boundline --affine-source nonzero --reducer mean --nonzero-eps 0 \
      --title-tag "P26012003 CSF" --title-mode compact --time-unit min

  # affine2boundline WITHOUT baseline (right boundary only), keep c=0, enforce a>0 and nonneg:
  python plot_roi_intensity.py block1.nii.gz block2.nii.gz \
      --radius 2.5 --center 57 38 49 \
      --norm-method affine2boundline --affine-source roi \
      --no-baseline-ok --c-policy zero --a-positive --nonneg \
      --title-tag "NoBaseline" --title-mode compact --time-unit min
"""

import argparse
import re
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Optional scipy for auto-mask
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

    # Bruker arrays: "##$X=( 2 )" then values on following lines
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


def solve_affine_from_two_constraints(y1L_hat: float, yB_hat: float, y1R_hat: float, y2_hat: float) -> tuple[float, float]:
    denom = (y1R_hat - y1L_hat)
    if abs(denom) < 1e-12:
        raise ValueError("Affine solve ill-conditioned: y1R_hat ~ y1L_hat (denominator ~ 0)")
    a = (y2_hat - yB_hat) / denom
    c = yB_hat - a * y1L_hat
    return float(a), float(c)


def pooled_std(values_list: list[np.ndarray]) -> float:
    # TRUE POOLING (A): one pooled distribution, not separate stds.
    x = np.concatenate([v.ravel() for v in values_list if v is not None], axis=0)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1))


def safe_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    return float(np.std(x, ddof=1))


def frac_negative(y: np.ndarray) -> float:
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0
    return float(np.mean(y < 0))


# ----------------------------------- main ----------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Plot ROI intensity across baseline + img1 + img2, with optional affine transform on img1 only."
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

    # Title/time options (you explicitly wanted these preserved)
    parser.add_argument("--title-tag", default=None, help="Optional custom tag to include in the plot title.")
    parser.add_argument("--title-mode", choices=["full", "compact"], default="full", help="Title style.")
    parser.add_argument("--time-unit", choices=["s", "min"], default="s", help="X axis units (seconds or minutes).")

    parser.add_argument("--mask-out", default=None, help="Output ROI mask NIfTI (default auto-named)")
    parser.add_argument("--out", default=None, help="Output plot PNG (default auto-named)")
    parser.add_argument("--csv-out", default=None, help="Optional CSV output")

    # Windows for line fits
    parser.add_argument("--stable-n", type=int, default=8, help="img1 head window for left-line fit (default 8).")
    parser.add_argument("--tail-n", type=int, default=8, help="img1 tail window for right-line fit (default 8).")
    parser.add_argument("--base-tail-n", type=int, default=0,
                        help="Baseline tail window for baseline-line fit. If 0, uses all baseline points.")
    parser.add_argument("--img2-head-n", type=int, default=8, help="img2 head window for img2-line fit (default 8).")

    # Transform choice
    parser.add_argument("--norm-method", choices=["none", "affine2boundline"], default="none",
                        help="Transform applied to img1 only. affine2boundline enforces boundary continuity using fitted lines.")

    # Allow affine2boundline without baseline
    parser.add_argument("--no-baseline-ok", action="store_true",
                        help="Allow affine2boundline with no baseline: enforce only right boundary. (Still requires img2.)")
    parser.add_argument("--c-policy", choices=["zero", "match_left_mean", "preserve_min"], default="zero",
                        help="When baseline is absent, how to choose c after selecting a. "
                             "zero: c=0. "
                             "match_left_mean: choose c so mean(img1 head) stays unchanged after scaling. "
                             "preserve_min: choose c so min(a*y1) maps to 0 (only meaningful with --nonneg).")

    # Affine source signal
    parser.add_argument("--affine-source", choices=["roi", "global", "nonzero"], default="roi",
                        help="Signal used to compute (a,c). roi=ROI mean; global=masked/global reducer; nonzero=reducer over >eps voxels.")
    parser.add_argument("--reducer", choices=["mean", "median"], default="mean", help="Reducer for global/nonzero sources.")
    parser.add_argument("--nonzero-eps", type=float, default=0.0, help="nonzero reducer: keep voxels > eps (default 0).")

    # Global mask controls (used when affine-source=global). You asked for AUTO explicitly.
    parser.add_argument("--global-mask", default=None,
                        help="3D mask NIfTI for global reducer OR 'auto' to generate from mean(volume). If omitted, uses ALL voxels.")
    parser.add_argument("--auto-mask-source", choices=["img2", "img2_baseline", "img1"], default="img2",
                        help="If --global-mask auto, which image to build it from (default img2).")
    parser.add_argument("--auto-mask-erosions", type=int, default=3)
    parser.add_argument("--auto-mask-dilations", type=int, default=3)
    parser.add_argument("--auto-mask-hist-bins", type=int, default=256)
    parser.add_argument("--auto-mask-out", default=None,
                        help="Write auto global mask to this path (default auto name). Use 'none' to skip writing.")

    # Expected/bounds for a (from NPro or explicit)
    parser.add_argument("--a-expected-from-npro", action="store_true",
                        help="Compute a_expected = NPro(img2)/NPro(img1) (requires method sidecars).")
    parser.add_argument("--a-expected", type=float, default=None, help="Override expected a (e.g., 3.0).")
    parser.add_argument("--a-bound-scale", type=float, default=0.0,
                        help="If >0 and a_expected known: enforce a in [a_expected/scale, a_expected*scale]. "
                             "Use something like 2-4. Default 0 disables bounds.")

    # Constraints
    parser.add_argument("--a-positive", action="store_true", help="Force a > 0.")
    parser.add_argument("--nonneg", action="store_true",
                        help="Penalize/avoid negative transformed values. "
                             "If baseline exists, solver will still satisfy left constraint; "
                             "for no-baseline mode, --c-policy preserve_min is recommended.")

    # Solver
    parser.add_argument("--affine-solver", choices=["exact", "robust"], default="exact",
                        help="exact: solve constraints directly. robust: bounded 1D search on a with penalties.")

    # Penalties (robust solver only)
    parser.add_argument("--std-max-ratio", type=float, default=1.7,
                        help="Penalty threshold for std inflation: std(img1_trans_head/tail)/std_target <= ratio. (default 1.7)")
    parser.add_argument("--penalty-w-resid", type=float, default=1.0, help="Weight for boundary residual penalty.")
    parser.add_argument("--penalty-w-std", type=float, default=10.0, help="Weight for std inflation penalty.")
    parser.add_argument("--penalty-w-neg", type=float, default=100.0, help="Weight for negativity penalty.")
    parser.add_argument("--robust-a-grid", type=int, default=2001,
                        help="Number of grid samples for robust a search (default 2001).")

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

    # Load img2 (optional but required for affine2boundline)
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

    # Load baseline (optional; allow 3D or 4D)
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

    # Build global mask if requested
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

    # ---------------- Transform: affine2boundline (Version 2) ---------------- #
    a = 1.0
    c = 0.0
    a_expected = None
    a_bounds = None
    ok = True
    msg = "none"

    # QA (line-based residuals)
    left_resid = None
    right_resid = None

    # penalty logging
    std_target = None
    stdB = None
    std2 = None
    std_head = None
    std_tail = None
    std_ratio_maxwin = None

    if args.norm_method == "affine2boundline":
        if data2 is None:
            raise ValueError("affine2boundline requires img_2.")
        if datab is None and not args.no_baseline_ok:
            raise ValueError("affine2boundline requires --img2-baseline unless --no-baseline-ok is set.")

        # affine-source signals
        s1 = get_affine_signal(data1)
        s2 = get_affine_signal(data2)
        sB = get_affine_signal(datab) if datab is not None else None

        # windows
        stable_n = int(args.stable_n)
        tail_n = int(args.tail_n)
        head2_n = int(args.img2_head_n)

        if stable_n < 2 or stable_n > s1.size:
            raise ValueError(f"--stable-n must be in [2..len(img1)] got {stable_n}")
        if tail_n < 2 or tail_n > s1.size:
            raise ValueError(f"--tail-n must be in [2..len(img1)] got {tail_n}")
        if head2_n < 2 or head2_n > s2.size:
            raise ValueError(f"--img2-head-n must be in [2..len(img2)] got {head2_n}")

        base_tail_n = int(args.base_tail_n)
        if datab is not None:
            if base_tail_n <= 0:
                base_tail_n = int(sB.size)
            if base_tail_n < 2 or base_tail_n > sB.size:
                raise ValueError(f"--base-tail-n must be in [2..len(baseline)] got {base_tail_n}")

        # absolute times for fitting
        if datab is not None:
            tb_abs_fit = segments_t[0]
            t1_abs_fit = segments_t[1]
            t2_abs_fit = segments_t[2]
            tL = float(t1_abs_fit[0])
            tR = float(t2_abs_fit[0])
        else:
            # no baseline segment
            t1_abs_fit = segments_t[0]
            t2_abs_fit = segments_t[1]
            tL = float(t1_abs_fit[0])
            tR = float(t2_abs_fit[0])

        # fit img1 head line and tail line (always)
        t1L_win = t1_abs_fit[:stable_n]
        y1L_win = s1[:stable_n]
        m1L, b1L = fit_line(t1L_win, y1L_win)

        t1R_win = t1_abs_fit[-tail_n:]
        y1R_win = s1[-tail_n:]
        m1R, b1R = fit_line(t1R_win, y1R_win)

        # fit img2 head line
        t2_win = t2_abs_fit[:head2_n]
        y2_win = s2[:head2_n]
        m2, b2 = fit_line(t2_win, y2_win)

        # eval at boundary times
        y1L_hat = eval_line(m1L, b1L, tL)
        y1R_hat = eval_line(m1R, b1R, tR)
        y2_hat = eval_line(m2, b2, tR)

        # baseline line only if baseline exists
        if datab is not None:
            tB_win = tb_abs_fit[-base_tail_n:]
            yB_win = sB[-base_tail_n:]
            mB, bB = fit_line(tB_win, yB_win)
            yB_hat = eval_line(mB, bB, tL)
        else:
            yB_win = None
            yB_hat = None

        # expected a from NPro if requested
        if args.a_expected is not None:
            a_expected = float(args.a_expected)
        elif args.a_expected_from_npro:
            a_expected = float(get_npro(img2_path) / get_npro(img1_path))

        if a_expected is not None and args.a_bound_scale and args.a_bound_scale > 0:
            a_bounds = (a_expected / float(args.a_bound_scale), a_expected * float(args.a_bound_scale))

        if args.a_positive:
            if a_bounds is None:
                a_bounds = (1e-12, float("inf"))
            else:
                a_bounds = (max(a_bounds[0], 1e-12), a_bounds[1])

        # TRUE POOLING (A) for std_target (baseline tail + img2 head together, if baseline exists)
        std2 = safe_std(y2_win)
        stdB = safe_std(yB_win) if yB_win is not None else float("nan")
        std_target = pooled_std([yB_win, y2_win]) if yB_win is not None else safe_std(y2_win)
        if not np.isfinite(std_target):
            # last-ditch fallback
            std_target = std2 if np.isfinite(std2) else (stdB if np.isfinite(stdB) else float("nan"))

        # Helper to choose c when baseline absent
        def choose_c_no_baseline(a_try: float) -> float:
            if args.c_policy == "zero":
                return 0.0
            if args.c_policy == "match_left_mean":
                # keep mean of img1 head window unchanged after scaling
                mu = float(np.nanmean(y1L_win))
                return mu - a_try * mu
            if args.c_policy == "preserve_min":
                # shift so minimum of transformed img1 is 0 (only makes sense if --nonneg)
                y_min = float(np.nanmin(s1))
                return -a_try * y_min
            raise ValueError("Unknown c-policy")

        # Exact solve
        def exact_solution() -> tuple[float, float, str]:
            if datab is not None:
                a0, c0 = solve_affine_from_two_constraints(y1L_hat, yB_hat, y1R_hat, y2_hat)
                return a0, c0, "exact_C0_from_lines"
            # no baseline: enforce right boundary only -> choose a from lines, then choose c by policy
            denom = y1R_hat
            if abs(denom) < 1e-12:
                raise ValueError("No-baseline exact: y1R_hat ~ 0; cannot determine a from right boundary.")
            a0 = y2_hat / y1R_hat
            c0 = choose_c_no_baseline(a0)
            return float(a0), float(c0), "exact_right_only"

        # Robust search
        def robust_solution() -> tuple[float, float, bool, str]:
            # Build bounds for search
            if a_bounds is not None:
                lo, hi = a_bounds
                if not np.isfinite(lo):
                    lo = 1e-12
                if not np.isfinite(hi):
                    hi = max(lo * 100.0, 100.0)
            else:
                # default broad range centered near expected if available, else [0.1, 20]
                if a_expected is not None and np.isfinite(a_expected):
                    lo = max(1e-12, a_expected / 10.0)
                    hi = a_expected * 10.0
                else:
                    lo, hi = (1e-3, 50.0)

            if args.a_positive:
                lo = max(lo, 1e-12)

            if hi <= lo:
                return lo, (yB_hat - lo * y1L_hat) if datab is not None else choose_c_no_baseline(lo), False, "bad_bounds"

            # Candidate grid
            grid_n = int(args.robust_a_grid)
            grid_n = max(101, grid_n)
            a_grid = np.linspace(lo, hi, grid_n, dtype=np.float64)

            best = None  # (cost, a, c, msg)
            for a_try in a_grid:
                # choose c to satisfy left constraint if baseline exists
                if datab is not None:
                    c_try = yB_hat - a_try * y1L_hat
                else:
                    c_try = choose_c_no_baseline(a_try)

                # boundary residuals (line-based)
                if datab is not None:
                    lr = (a_try * y1L_hat + c_try) - yB_hat  # should be 0 by construction
                else:
                    lr = 0.0

                rr = (a_try * y1R_hat + c_try) - y2_hat

                # std inflation check on affine-source windows (head/tail), transformed
                y_head = a_try * y1L_win + c_try
                y_tail = a_try * y1R_win + c_try
                sh = safe_std(y_head)
                st = safe_std(y_tail)

                # compute ratio vs pooled std_target
                ratios = []
                if np.isfinite(sh) and np.isfinite(std_target) and std_target > 0:
                    ratios.append(sh / std_target)
                if np.isfinite(st) and np.isfinite(std_target) and std_target > 0:
                    ratios.append(st / std_target)
                rmax = max(ratios) if ratios else 1.0

                # penalties
                cost = 0.0
                cost += args.penalty_w_resid * (rr ** 2)
                # penalize std inflation above threshold
                if np.isfinite(rmax) and rmax > args.std_max_ratio:
                    cost += args.penalty_w_std * ((rmax - args.std_max_ratio) ** 2)

                # penalize negativity if requested
                if args.nonneg:
                    # use transformed img1 ROI (what you actually plot) as the negativity proxy
                    y_plot = a_try * y1_roi + c_try
                    neg_frac = frac_negative(y_plot)
                    if neg_frac > 0:
                        cost += args.penalty_w_neg * (neg_frac ** 2)

                # keep a near expected (soft) if provided
                if a_expected is not None and np.isfinite(a_expected) and a_expected > 0:
                    # very light regularizer (so it doesn't fight actual continuity)
                    cost += 0.01 * ((a_try - a_expected) / a_expected) ** 2

                if best is None or cost < best[0]:
                    best = (cost, float(a_try), float(c_try), rmax, rr, lr, sh, st)

            if best is None:
                return 1.0, 0.0, False, "robust_failed"

            _, a_best, c_best, rmax, rr, lr, sh, st = best
            # declare ok if residual small-ish and no-neg if requested
            ok_local = True
            if abs(rr) > 1e-6 * max(1.0, abs(y2_hat)):
                ok_local = False
            if args.nonneg:
                if frac_negative(a_best * y1_roi + c_best) > 0.0:
                    ok_local = False

            # stash std logs for printing
            nonlocal std_head, std_tail, std_ratio_maxwin
            std_head = sh
            std_tail = st
            std_ratio_maxwin = rmax

            return a_best, c_best, ok_local, "robust_penalized_1D"

        # Choose solver
        if args.affine_solver == "exact":
            a, c, msg = exact_solution()
            ok = True
        else:
            a, c, ok, msg = robust_solution()

        # Apply additional hard bounds if specified and exact violated them
        if a_bounds is not None:
            lo, hi = a_bounds
            if a < lo:
                a = lo
                if datab is not None:
                    c = yB_hat - a * y1L_hat
                else:
                    c = choose_c_no_baseline(a)
                ok = False
                msg = "a_hit_lower_bound"
            if a > hi:
                a = hi
                if datab is not None:
                    c = yB_hat - a * y1L_hat
                else:
                    c = choose_c_no_baseline(a)
                ok = False
                msg = "a_hit_upper_bound"

        if args.a_positive and a <= 0:
            # force positivity
            a = max(a, 1e-12)
            if datab is not None:
                c = yB_hat - a * y1L_hat
            else:
                c = choose_c_no_baseline(a)
            ok = False
            msg = "a_forced_positive"

        # Optional “nonneg” hard adjustment in no-baseline mode only (since baseline constraint could force negatives)
        if args.nonneg and datab is None:
            y_try = a * y1_roi + c
            if np.nanmin(y_try) < 0:
                # shift up to make min==0
                c = c - float(np.nanmin(y_try))
                ok = False
                msg = msg + "+shift_nonneg"

        # Apply affine transform to img1 ROI segment for plotting
        y1_roi_xform = a * y1_roi + c
        # replace plotted img1 segment (index depends on baseline presence)
        idx_img1 = 1 if datab is not None else 0
        segments_y[idx_img1] = y1_roi_xform
        labels[idx_img1] = f"img1_ROI→a*y+c (a={a:.6g}, c={c:.6g})"

        # Line-based residuals (the actual constraints)
        if datab is not None:
            left_resid = (a * y1L_hat + c) - yB_hat
        else:
            left_resid = None
        right_resid = (a * y1R_hat + c) - y2_hat

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

    if args.center is None:
        coord_line = f"({center_label};r={args.radius:g})"
    else:
        coord_line = f"({int(round(cx))},{int(round(cy))},{int(round(cz))};r={args.radius:g})"

    ax = plt.gca()
    if args.title_mode == "compact":
        ax.set_title(f"{title_tag} {coord_line}", fontsize=12)
    else:
        title = f"{title_tag} {coord_line}\nROI mean intensity"
        title += f"\n(mode={args.norm_method}, affine-source={args.affine_source}, reducer={args.reducer}, global-mask={global_mask_note})"
        if args.norm_method == "affine2boundline":
            title += f"\nsolver={args.affine_solver}  a={a:.6g}, c={c:.6g}"
            if a_expected is not None:
                title += f"  a_expected={a_expected:.6g}"
            if a_bounds is not None:
                title += f"  a_bounds=[{a_bounds[0]:.6g},{a_bounds[1]:.6g}]"
            title += f"\nstatus ok={ok} msg={msg}"
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
    print("[mode] ROI plot; baseline+img2 fixed; img1 transformed only" if args.norm_method != "none" else "[mode] ROI plot (no transform)")
    print(f"       segments: {labels}")
    print(f"[roi] center={center_label}, radius={args.radius:g}, voxels={int(roi_mask.sum())}")
    print(f"[affine-source] {args.affine_source}  (reducer={args.reducer}, nonzero-eps={args.nonzero_eps:g}, global-mask={global_mask_note})")

    if args.norm_method == "affine2boundline":
        if a_expected is not None:
            if a_bounds is not None:
                print(f"[img1-transform] method=affine2boundline  solver={args.affine_solver}  a_expected={a_expected:.10g}  a_bounds=[{a_bounds[0]:.10g},{a_bounds[1]:.10g}]")
            else:
                print(f"[img1-transform] method=affine2boundline  solver={args.affine_solver}  a_expected={a_expected:.10g}")
        else:
            print(f"[img1-transform] method=affine2boundline  solver={args.affine_solver}")
        print(f"                a={a:.10g}  c={c:.10g}  ok={ok}  msg={msg}")

        if left_resid is not None:
            print(f"[line-C0] left_resid(a*y1Lhat+c - yBhat)={left_resid:.6g}")
        else:
            print(f"[line-C0] left_resid=N/A (no baseline)")
        print(f"[line-C0] right_resid(a*y1Rhat+c - y2hat)={right_resid:.6g}")

        # penalty diagnostics (pooled)
        if std_target is not None:
            # compute head/tail std on transformed affine-source windows if exact solver used (for logging)
            if args.affine_solver == "exact":
                # we can approximate with ROI-based windows for logging only
                std_head = std_head if std_head is not None else float("nan")
                std_tail = std_tail if std_tail is not None else float("nan")
            print(f"[penalty] std_target(pooled)={std_target:.6g}  std2={std2:.6g}  stdB={stdB:.6g}")

    print(f"[out] roi-mask: {mask_out}")
    print(f"[out] plot    : {out_plot}")
    if csv_out:
        print(f"[out] csv     : {csv_out}")


if __name__ == "__main__":
    main()