#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from pathlib import Path


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def read_runnos_from_file(path: Path):
    runnos = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                runnos.append(s)
    return runnos


def discover_runnos(base_dir: Path):
    out = []
    for d in sorted(base_dir.glob("z*")):
        if d.is_dir():
            runno = d.name[1:]
            if runno:
                out.append(runno)
    return out


def unique_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def sanitize_job_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def submit_job(script_path: Path):
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def write_headmask_helper(helper_path: Path):
    helper_code = r'''#!/usr/bin/env python3

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    binary_opening,
    binary_propagation,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
    label,
)


def largest_component(mask: np.ndarray, structure=None) -> np.ndarray:
    if structure is None:
        structure = generate_binary_structure(3, 2)
    lab, nlab = label(mask, structure=structure)
    if nlab == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    best = counts.argmax()
    return lab == best


def remove_small_components(mask: np.ndarray, min_voxels: int, structure=None) -> np.ndarray:
    if structure is None:
        structure = generate_binary_structure(3, 2)
    if min_voxels <= 1:
        return mask.copy()
    lab, nlab = label(mask, structure=structure)
    if nlab == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(lab.ravel())
    keep = np.zeros(nlab + 1, dtype=bool)
    for k in range(1, nlab + 1):
        if counts[k] >= min_voxels:
            keep[k] = True
    return keep[lab]


def robust_normalize(vol: np.ndarray, mask: np.ndarray, low_pct=2.0, high_pct=98.0) -> np.ndarray:
    vals = vol[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise RuntimeError("No finite values inside mask for normalization.")
    lo = np.percentile(vals, low_pct)
    hi = np.percentile(vals, high_pct)
    if hi <= lo:
        raise RuntimeError(f"Bad normalization range: lo={lo}, hi={hi}")
    out = (vol - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def boundary_seed(mask: np.ndarray) -> np.ndarray:
    seed = np.zeros_like(mask, dtype=bool)
    seed[0, :, :] |= mask[0, :, :]
    seed[-1, :, :] |= mask[-1, :, :]
    seed[:, 0, :] |= mask[:, 0, :]
    seed[:, -1, :] |= mask[:, -1, :]
    seed[:, :, 0] |= mask[:, :, 0]
    seed[:, :, -1] |= mask[:, :, -1]
    return seed


def save_like(path: Path, data: np.ndarray, ref_img: nib.Nifti1Image):
    img = nib.Nifti1Image(data, ref_img.affine, ref_img.header)
    nib.save(img, str(path))


def main():
    ap = argparse.ArgumentParser(
        description="Refine a whole-head mask from BFC T2 using outer dark-background flood fill plus bright-shell protection."
    )
    ap.add_argument("--bfc", required=True)
    ap.add_argument("--support_mask", required=True)
    ap.add_argument("--out_mask", required=True)
    ap.add_argument("--out_masked_bfc", required=True)
    ap.add_argument("--debug_prefix", default=None)

    ap.add_argument("--smooth_sigma", type=float, default=1.0)

    # Dark exterior/background
    ap.add_argument("--dark_bg_threshold", type=float, default=0.20,
                    help="Normalized intensity threshold for exterior/background candidates.")
    ap.add_argument("--min_bg_volume_mm3", type=float, default=0.03)
    ap.add_argument("--bg_close_iters", type=int, default=1)

    # Bright shell
    ap.add_argument("--shell_threshold", type=float, default=0.55,
                    help="Normalized intensity threshold for outer bright shell candidates.")
    ap.add_argument("--min_shell_volume_mm3", type=float, default=0.03)
    ap.add_argument("--shell_close_iters", type=int, default=1)
    ap.add_argument("--shell_protect_mm", type=float, default=0.40,
                    help="Distance around bright shell to protect from exterior subtraction.")

    # Head cleanup / completion
    ap.add_argument("--head_close_iters", type=int, default=2)
    ap.add_argument("--head_open_iters", type=int, default=0)
    ap.add_argument("--head_dilate_iters", type=int, default=0)
    ap.add_argument("--head_fill_holes", action="store_true")

    ap.add_argument("--tight_mask", action="store_true")
    ap.add_argument("--tight_erode_iters", type=int, default=1)

    args = ap.parse_args()

    bfc_img = nib.load(args.bfc)
    sup_img = nib.load(args.support_mask)

    bfc = bfc_img.get_fdata(dtype=np.float32)
    support = sup_img.get_fdata() > 0

    if bfc.shape != support.shape:
        raise RuntimeError("BFC image and support mask shapes do not match.")
    if not np.any(support):
        raise RuntimeError("Support mask is empty.")

    zooms = bfc_img.header.get_zooms()[:3]
    voxel_volume_mm3 = float(zooms[0] * zooms[1] * zooms[2])
    if voxel_volume_mm3 <= 0:
        raise RuntimeError(f"Bad voxel volume from header: {voxel_volume_mm3}")

    mean_spacing = float(np.mean(zooms))
    shell_protect_vox = max(1.0, args.shell_protect_mm / max(mean_spacing, 1e-6))
    min_bg_vox = max(1, int(round(args.min_bg_volume_mm3 / voxel_volume_mm3)))
    min_shell_vox = max(1, int(round(args.min_shell_volume_mm3 / voxel_volume_mm3)))

    structure = generate_binary_structure(3, 2)

    bfc_smooth = gaussian_filter(bfc, sigma=args.smooth_sigma)
    inten_norm = robust_normalize(bfc_smooth, support, low_pct=2.0, high_pct=98.0)

    # Bright shell candidates
    shell_raw = (inten_norm >= args.shell_threshold) & support
    shell = remove_small_components(shell_raw, min_shell_vox, structure=structure)
    if args.shell_close_iters > 0 and np.any(shell):
        shell = binary_closing(shell, structure=structure, iterations=args.shell_close_iters)

    dist_to_shell = distance_transform_edt(~shell)
    shell_protect = support & (dist_to_shell <= shell_protect_vox)

    # Dark background candidates
    bg_raw = (inten_norm <= args.dark_bg_threshold) & support
    bg = remove_small_components(bg_raw, min_bg_vox, structure=structure)
    if args.bg_close_iters > 0 and np.any(bg):
        bg = binary_closing(bg, structure=structure, iterations=args.bg_close_iters)

    # Exterior dark background: only what is connected to support boundary
    bg_open = bg & (~shell_protect)
    seeds = boundary_seed(bg_open)
    exterior_bg = binary_propagation(seeds, structure=structure, mask=bg_open)

    # Head mask = support minus exterior background
    head = support & (~exterior_bg)

    # Add protected shell back explicitly, just in case
    head |= shell_protect

    head &= support
    head = largest_component(head, structure=structure)

    if args.head_fill_holes:
        head = binary_fill_holes(head)

    if args.head_close_iters > 0:
        head = binary_closing(head, structure=structure, iterations=args.head_close_iters)

    if args.head_open_iters > 0:
        head = binary_opening(head, structure=structure, iterations=args.head_open_iters)

    if args.head_dilate_iters > 0:
        head = binary_dilation(head, structure=structure, iterations=args.head_dilate_iters)

    head &= support
    head = largest_component(head, structure=structure)
    head = binary_fill_holes(head)
    head = largest_component(head, structure=structure)

    if args.tight_mask:
        head = binary_erosion(head, structure=structure, iterations=args.tight_erode_iters)
        head = largest_component(head, structure=structure)
        head = binary_fill_holes(head)

    out_mask = head.astype(np.uint8)
    out_masked_bfc = np.where(out_mask > 0, bfc, 0.0).astype(np.float32)

    save_like(Path(args.out_mask), out_mask, bfc_img)
    save_like(Path(args.out_masked_bfc), out_masked_bfc, bfc_img)

    if args.debug_prefix:
        prefix = Path(args.debug_prefix)
        save_like(prefix.with_name(prefix.name + "_inten_norm.nii.gz"), inten_norm.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_shell_raw.nii.gz"), shell_raw.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_shell.nii.gz"), shell.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_dist_to_shell.nii.gz"), dist_to_shell.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_shell_protect.nii.gz"), shell_protect.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_bg_raw.nii.gz"), bg_raw.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_bg.nii.gz"), bg.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_bg_open.nii.gz"), bg_open.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_exterior_bg.nii.gz"), exterior_bg.astype(np.uint8), bfc_img)


if __name__ == "__main__":
    main()
'''
    helper_path.write_text(helper_code)
    helper_path.chmod(0o755)


def build_job_script(
    runno: str,
    in_nii: Path,
    out_nii: Path,
    in_method: Path,
    out_method: Path,
    premask_nii: Path,
    bias_nii: Path,
    headmask_nii: Path,
    head_t2_nii: Path,
    diff_nii: Path | None,
    tmp_otsu_pre_nii: Path,
    helper_py: Path,
    python_path: str,
    n4_path: str,
    thresholdimage_path: str,
    imagemath_path: str,
    printheader_path: str | None,
    dimension: int,
    shrink_factor: int,
    convergence: str,
    bspline: str,
    histogram_sharpening: str,
    pre_otsu_keep_low: int,
    pre_otsu_keep_high: int,
    pre_close_radius: int,
    pre_dilate_radius_pre_glc: int,
    pre_fill_holes_radius: int,
    pre_dilate_radius_final: int,
    head_smooth_sigma: float,
    head_dark_bg_threshold: float,
    head_min_bg_volume_mm3: float,
    head_bg_close_iters: int,
    head_shell_threshold: float,
    head_min_shell_volume_mm3: float,
    head_shell_close_iters: int,
    head_shell_protect_mm: float,
    head_close_iters: int,
    head_open_iters: int,
    head_dilate_iters: int,
    head_fill_holes: bool,
    tight_mask: bool,
    tight_mask_erode_iters: int,
    save_head_debug: bool,
    threads: int,
    job_name: str,
    log_out: Path,
    log_err: Path,
    partition: str | None,
    time_str: str,
    mem_gb: int,
    cpus: int,
    overwrite: bool,
):
    sbatch_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_out}",
        f"#SBATCH --error={log_err}",
        f"#SBATCH --time={time_str}",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH --cpus-per-task={cpus}",
    ]
    if partition:
        sbatch_lines.append(f"#SBATCH --partition={partition}")

    script = "\n".join(sbatch_lines) + "\n\n"

    script += r"""set -euo pipefail

echo "===== JOB START ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "RUNNO=""" + runno + r""""

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=""" + str(threads) + r"""

in_nii=""" + shell_quote(str(in_nii)) + r"""
out_nii=""" + shell_quote(str(out_nii)) + r"""
in_method=""" + shell_quote(str(in_method)) + r"""
out_method=""" + shell_quote(str(out_method)) + r"""
premask_nii=""" + shell_quote(str(premask_nii)) + r"""
bias_nii=""" + shell_quote(str(bias_nii)) + r"""
headmask_nii=""" + shell_quote(str(headmask_nii)) + r"""
head_t2_nii=""" + shell_quote(str(head_t2_nii)) + r"""
tmp_otsu_pre_nii=""" + shell_quote(str(tmp_otsu_pre_nii)) + r"""
helper_py=""" + shell_quote(str(helper_py)) + r"""
python_exe=""" + shell_quote(str(python_path)) + r"""
overwrite_flag=""" + ("1" if overwrite else "0") + r"""
tight_mask_flag=""" + ("1" if tight_mask else "0") + r"""
save_head_debug_flag=""" + ("1" if save_head_debug else "0") + r"""
head_fill_holes_flag=""" + ("1" if head_fill_holes else "0") + r"""
"""

    if diff_nii is not None:
        script += r"""diff_nii=""" + shell_quote(str(diff_nii)) + r"""
"""
    else:
        script += r"""diff_nii=""
"""

    script += r"""
n4_exe=""" + shell_quote(str(n4_path)) + r"""
threshold_exe=""" + shell_quote(str(thresholdimage_path)) + r"""
imagemath_exe=""" + shell_quote(str(imagemath_path)) + r"""
"""
    if printheader_path:
        script += r"""printheader_exe=""" + shell_quote(str(printheader_path)) + r"""
"""
    else:
        script += r"""printheader_exe=""
"""

    script += r"""
mkdir -p "$(dirname "$out_nii")"

if [[ ! -f "$in_nii" ]]; then
    echo "ERROR: Input not found: $in_nii" >&2
    exit 1
fi

if [[ -n "$printheader_exe" ]]; then
    echo
    echo "Header / spacing check:"
    "$printheader_exe" "$in_nii" 1 || true
fi

need_n4=0
if [[ "$overwrite_flag" == "1" ]]; then
    need_n4=1
elif [[ ! -f "$out_nii" || ! -f "$bias_nii" || ! -f "$premask_nii" ]]; then
    need_n4=1
fi

need_headmask=0
if [[ "$overwrite_flag" == "1" ]]; then
    need_headmask=1
elif [[ ! -f "$headmask_nii" || ! -f "$head_t2_nii" ]]; then
    need_headmask=1
fi

if [[ -n "$diff_nii" ]]; then
    need_diff=0
    if [[ "$overwrite_flag" == "1" ]]; then
        need_diff=1
    elif [[ ! -f "$diff_nii" ]]; then
        need_diff=1
    fi
else
    need_diff=0
fi

make_support_mask() {
    local src_nii="$1"
    local tmp_otsu="$2"
    local dst_mask="$3"

    rm -f "$tmp_otsu" "$dst_mask"

    "$threshold_exe" """ + str(dimension) + r""" "$src_nii" "$tmp_otsu" Otsu 4
    "$threshold_exe" """ + str(dimension) + r""" "$tmp_otsu" "$dst_mask" """ + str(pre_otsu_keep_low) + r""" """ + str(pre_otsu_keep_high) + r""" 1 0
    "$imagemath_exe" """ + str(dimension) + r""" "$dst_mask" MC "$dst_mask" """ + str(pre_close_radius) + r"""
    "$imagemath_exe" """ + str(dimension) + r""" "$dst_mask" MD "$dst_mask" """ + str(pre_dilate_radius_pre_glc) + r"""
    "$imagemath_exe" """ + str(dimension) + r""" "$dst_mask" FillHoles "$dst_mask" """ + str(pre_fill_holes_radius) + r"""
    "$imagemath_exe" """ + str(dimension) + r""" "$dst_mask" GetLargestComponent "$dst_mask"
    "$imagemath_exe" """ + str(dimension) + r""" "$dst_mask" MD "$dst_mask" """ + str(pre_dilate_radius_final) + r"""
}

if [[ "$need_n4" == "1" ]]; then
    echo
    echo "Creating pre-N4 support mask..."
    make_support_mask "$in_nii" "$tmp_otsu_pre_nii" "$premask_nii"

    echo
    echo "Running N4BiasFieldCorrection..."
    cmd=(
        "$n4_exe"
        -d """ + str(dimension) + r"""
        -i "$in_nii"
        -x "$premask_nii"
        -s """ + str(shrink_factor) + r"""
        -c """ + shell_quote(convergence) + r"""
        -b """ + shell_quote(bspline) + r"""
        -t """ + shell_quote(histogram_sharpening) + r"""
        -r 1
        -o "[""$out_nii"",""$bias_nii""]"
    )
    printf '  %q' "${cmd[@]}"
    echo
    "${cmd[@]}"

    if [[ ! -f "$out_nii" || ! -f "$bias_nii" ]]; then
        echo "ERROR: N4 outputs missing." >&2
        exit 1
    fi

    if [[ -f "$in_method" ]]; then
        cp -f "$in_method" "$out_method"
    fi
else
    echo
    echo "Skipping N4 stage; existing outputs found and overwrite is off."
fi

if [[ "$need_headmask" == "1" ]]; then
    if [[ ! -f "$out_nii" ]]; then
        echo "ERROR: Missing corrected image for head mask: $out_nii" >&2
        exit 1
    fi
    if [[ ! -f "$premask_nii" ]]; then
        echo "ERROR: Missing support mask for head mask: $premask_nii" >&2
        exit 1
    fi

    echo
    echo "Creating refined whole-head mask..."
    head_cmd=(
        "$python_exe" "$helper_py"
        --bfc "$out_nii"
        --support_mask "$premask_nii"
        --out_mask "$headmask_nii"
        --out_masked_bfc "$head_t2_nii"
        --smooth_sigma """ + str(head_smooth_sigma) + r"""
        --dark_bg_threshold """ + str(head_dark_bg_threshold) + r"""
        --min_bg_volume_mm3 """ + str(head_min_bg_volume_mm3) + r"""
        --bg_close_iters """ + str(head_bg_close_iters) + r"""
        --shell_threshold """ + str(head_shell_threshold) + r"""
        --min_shell_volume_mm3 """ + str(head_min_shell_volume_mm3) + r"""
        --shell_close_iters """ + str(head_shell_close_iters) + r"""
        --shell_protect_mm """ + str(head_shell_protect_mm) + r"""
        --head_close_iters """ + str(head_close_iters) + r"""
        --head_open_iters """ + str(head_open_iters) + r"""
        --head_dilate_iters """ + str(head_dilate_iters) + r"""
        --tight_erode_iters """ + str(tight_mask_erode_iters) + r"""
    )

    if [[ "$head_fill_holes_flag" == "1" ]]; then
        head_cmd+=( --head_fill_holes )
    fi

    if [[ "$tight_mask_flag" == "1" ]]; then
        head_cmd+=( --tight_mask )
    fi

    if [[ "$save_head_debug_flag" == "1" ]]; then
        head_cmd+=( --debug_prefix "$(dirname "$headmask_nii")/""" + runno + r"""_bfc_head_dbg" )
    fi

    printf '  %q' "${head_cmd[@]}"
    echo
    "${head_cmd[@]}"

    if [[ ! -f "$headmask_nii" || ! -f "$head_t2_nii" ]]; then
        echo "ERROR: Head-mask outputs missing." >&2
        exit 1
    fi
else
    echo
    echo "Skipping head-mask stage; outputs exist and overwrite is off."
fi

if [[ "$need_diff" == "1" ]]; then
    "$imagemath_exe" """ + str(dimension) + r""" "$diff_nii" - "$out_nii" "$in_nii"
fi

rm -f "$tmp_otsu_pre_nii"

echo "===== JOB END ====="
date
"""
    return script


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", required=True)

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--runnos", nargs="+")
    group.add_argument("--runno_file")
    group.add_argument("--discover", action="store_true")

    p.add_argument("--sbatch_dir", default=None)

    p.add_argument("--python_path", default="python3")
    p.add_argument("--n4_path", default="N4BiasFieldCorrection")
    p.add_argument("--thresholdimage_path", default="ThresholdImage")
    p.add_argument("--imagemath_path", default="ImageMath")
    p.add_argument("--printheader_path", default="PrintHeader")

    p.add_argument("--dimension", type=int, default=3, choices=[2, 3, 4])

    p.add_argument("--shrink_factor", type=int, default=1)
    p.add_argument("--convergence", default="[200x200x100x50,1e-8]")
    p.add_argument("--bspline", default="[8]")
    p.add_argument("--histogram_sharpening", default="[0.15,0.01,200]")

    p.add_argument("--pre_otsu_keep_low", type=int, default=2)
    p.add_argument("--pre_otsu_keep_high", type=int, default=4)
    p.add_argument("--pre_close_radius", type=int, default=2)
    p.add_argument("--pre_dilate_radius_pre_glc", type=int, default=1)
    p.add_argument("--pre_fill_holes_radius", type=int, default=2)
    p.add_argument("--pre_dilate_radius_final", type=int, default=2)

    # Whole-head refinement parameters
    p.add_argument("--head_smooth_sigma", type=float, default=1.0)

    p.add_argument("--head_dark_bg_threshold", type=float, default=0.20)
    p.add_argument("--head_min_bg_volume_mm3", type=float, default=0.03)
    p.add_argument("--head_bg_close_iters", type=int, default=1)

    p.add_argument("--head_shell_threshold", type=float, default=0.55)
    p.add_argument("--head_min_shell_volume_mm3", type=float, default=0.03)
    p.add_argument("--head_shell_close_iters", type=int, default=1)
    p.add_argument("--head_shell_protect_mm", type=float, default=0.40)

    p.add_argument("--head_close_iters", type=int, default=2)
    p.add_argument("--head_open_iters", type=int, default=0)
    p.add_argument("--head_dilate_iters", type=int, default=0)
    p.add_argument("--head_fill_holes", action="store_true")

    p.add_argument("--tight_mask", action="store_true")
    p.add_argument("--tight_mask_erode_iters", type=int, default=1)

    p.add_argument("--save_head_debug", action="store_true")

    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--mem_gb", type=int, default=16)
    p.add_argument("--time", default="04:00:00")
    p.add_argument("--partition", default=None)

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_diff", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    args = p.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    if not base_dir.exists():
        eprint(f"[ERROR] base_dir does not exist: {base_dir}")
        return 1

    if args.runnos:
        runnos = [str(x).strip() for x in args.runnos if str(x).strip()]
    elif args.runno_file:
        runnos = read_runnos_from_file(Path(args.runno_file).expanduser().resolve())
    else:
        runnos = discover_runnos(base_dir)

    runnos = unique_preserve_order(runnos)
    if not runnos:
        eprint("[ERROR] No runnos found.")
        return 1

    sbatch_dir = Path(args.sbatch_dir).expanduser().resolve() if args.sbatch_dir else (base_dir / "n4_t2_sbatch")
    scripts_dir = sbatch_dir / "scripts"
    logs_dir = sbatch_dir / "logs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    helper_py = sbatch_dir / "make_head_mask_from_bfc.py"
    write_headmask_helper(helper_py)

    print(f"[INFO] base_dir    : {base_dir}")
    print(f"[INFO] sbatch_dir  : {sbatch_dir}")
    print(f"[INFO] runno count : {len(runnos)}")

    n_prepared = 0
    n_submitted = 0
    n_skipped = 0
    n_missing = 0
    n_failed_submit = 0

    for runno in runnos:
        run_dir = base_dir / f"z{runno}"
        in_nii = run_dir / f"{runno}_T2.nii.gz"
        out_nii = run_dir / f"{runno}_bfc_T2.nii.gz"
        in_method = run_dir / f"{runno}_T2.method"
        out_method = run_dir / f"{runno}_bfc_T2.method"
        premask_nii = run_dir / f"{runno}_bfc_mask.nii.gz"
        bias_nii = run_dir / f"{runno}_bfc_biasfield.nii.gz"
        headmask_nii = run_dir / f"{runno}_bfc_head_mask.nii.gz"
        head_t2_nii = run_dir / f"{runno}_bfc_head_T2.nii.gz"
        tmp_otsu_pre_nii = run_dir / f"{runno}_bfc_otsu_pre_tmp.nii.gz"
        diff_nii = run_dir / f"{runno}_bfc_diff_T2.nii.gz" if args.save_diff else None

        if not in_nii.exists():
            eprint(f"[MISSING] Input NIfTI not found for runno {runno}: {in_nii}")
            n_missing += 1
            continue

        if not args.overwrite:
            have_core = out_nii.exists() and bias_nii.exists() and premask_nii.exists()
            have_head = headmask_nii.exists() and head_t2_nii.exists()
            have_diff = True if diff_nii is None else diff_nii.exists()
            if have_core and have_head and have_diff:
                print(f"[SKIP] All requested outputs already exist for runno {runno}")
                n_skipped += 1
                continue

        job_name = sanitize_job_name(f"n4_t2_{runno}")
        script_path = scripts_dir / f"{job_name}.sbatch"
        log_out = logs_dir / f"{job_name}.%j.out"
        log_err = logs_dir / f"{job_name}.%j.err"

        script_text = build_job_script(
            runno=runno,
            in_nii=in_nii,
            out_nii=out_nii,
            in_method=in_method,
            out_method=out_method,
            premask_nii=premask_nii,
            bias_nii=bias_nii,
            headmask_nii=headmask_nii,
            head_t2_nii=head_t2_nii,
            diff_nii=diff_nii,
            tmp_otsu_pre_nii=tmp_otsu_pre_nii,
            helper_py=helper_py,
            python_path=args.python_path,
            n4_path=args.n4_path,
            thresholdimage_path=args.thresholdimage_path,
            imagemath_path=args.imagemath_path,
            printheader_path=args.printheader_path,
            dimension=args.dimension,
            shrink_factor=args.shrink_factor,
            convergence=args.convergence,
            bspline=args.bspline,
            histogram_sharpening=args.histogram_sharpening,
            pre_otsu_keep_low=args.pre_otsu_keep_low,
            pre_otsu_keep_high=args.pre_otsu_keep_high,
            pre_close_radius=args.pre_close_radius,
            pre_dilate_radius_pre_glc=args.pre_dilate_radius_pre_glc,
            pre_fill_holes_radius=args.pre_fill_holes_radius,
            pre_dilate_radius_final=args.pre_dilate_radius_final,
            head_smooth_sigma=args.head_smooth_sigma,
            head_dark_bg_threshold=args.head_dark_bg_threshold,
            head_min_bg_volume_mm3=args.head_min_bg_volume_mm3,
            head_bg_close_iters=args.head_bg_close_iters,
            head_shell_threshold=args.head_shell_threshold,
            head_min_shell_volume_mm3=args.head_min_shell_volume_mm3,
            head_shell_close_iters=args.head_shell_close_iters,
            head_shell_protect_mm=args.head_shell_protect_mm,
            head_close_iters=args.head_close_iters,
            head_open_iters=args.head_open_iters,
            head_dilate_iters=args.head_dilate_iters,
            head_fill_holes=args.head_fill_holes,
            tight_mask=args.tight_mask,
            tight_mask_erode_iters=args.tight_mask_erode_iters,
            save_head_debug=args.save_head_debug,
            threads=args.threads,
            job_name=job_name,
            log_out=log_out,
            log_err=log_err,
            partition=args.partition,
            time_str=args.time,
            mem_gb=args.mem_gb,
            cpus=args.cpus,
            overwrite=args.overwrite,
        )

        script_path.write_text(script_text)
        n_prepared += 1
        print(f"[PREPARED] {script_path}")

        if args.dry_run:
            continue

        rc, stdout, stderr = submit_job(script_path)
        if rc == 0:
            print(f"[SUBMITTED] runno {runno}: {stdout}")
            n_submitted += 1
        else:
            eprint(f"[SUBMIT FAIL] runno {runno}")
            if stdout:
                eprint(stdout)
            if stderr:
                eprint(stderr)
            n_failed_submit += 1

    print("\n[SUMMARY]")
    print(f"  Prepared     : {n_prepared}")
    print(f"  Submitted    : {n_submitted}")
    print(f"  Skipped      : {n_skipped}")
    print(f"  Missing input: {n_missing}")
    print(f"  Submit failed: {n_failed_submit}")

    return 0 if n_failed_submit == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())