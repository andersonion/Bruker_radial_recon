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


def write_brainmask_helper(helper_path: Path):
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
    binary_propagation,
    gaussian_filter,
    gaussian_gradient_magnitude,
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


def boundary_seed(mask: np.ndarray) -> np.ndarray:
    seed = np.zeros_like(mask, dtype=bool)
    if mask.ndim != 3:
        raise RuntimeError("Expected 3D mask.")
    # all six faces
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
        description="Build a brain mask from BFC T2 using a gradient shell / enclosed-cavity strategy."
    )
    ap.add_argument("--bfc", required=True)
    ap.add_argument("--support_mask", required=True)
    ap.add_argument("--out_mask", required=True)
    ap.add_argument("--out_masked_bfc", required=True)
    ap.add_argument("--debug_prefix", default=None)

    ap.add_argument("--smooth_sigma", type=float, default=1.0)
    ap.add_argument("--grad_sigma", type=float, default=1.0)
    ap.add_argument("--grad_threshold", type=float, default=0.10)

    # physical cleanup thresholds, in mm^3
    ap.add_argument("--min_barrier_volume_mm3", type=float, default=0.05,
                    help="Remove tiny white edge blobs smaller than this physical volume.")
    ap.add_argument("--min_enclosed_volume_mm3", type=float, default=1.0,
                    help="Remove tiny enclosed black regions smaller than this physical volume.")

    ap.add_argument("--close_barrier_iters", type=int, default=1,
                    help="Closing iterations on binary barrier shell.")
    ap.add_argument("--brain_close_iters", type=int, default=2)
    ap.add_argument("--brain_dilate_iters", type=int, default=0)

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

    min_barrier_vox = max(1, int(round(args.min_barrier_volume_mm3 / voxel_volume_mm3)))
    min_enclosed_vox = max(1, int(round(args.min_enclosed_volume_mm3 / voxel_volume_mm3)))

    structure = generate_binary_structure(3, 2)

    # Smooth BFC and compute gradient
    bfc_smooth = gaussian_filter(bfc, sigma=args.smooth_sigma)
    grad = gaussian_gradient_magnitude(bfc_smooth, sigma=args.grad_sigma)
    grad_norm = robust_normalize(grad, support, low_pct=2.0, high_pct=98.0)

    # Binary shell/barrier
    barrier = (grad_norm >= args.grad_threshold) & support
    barrier = remove_small_components(barrier, min_barrier_vox, structure=structure)

    if args.close_barrier_iters > 0 and np.any(barrier):
        barrier = binary_closing(barrier, structure=structure, iterations=args.close_barrier_iters)

    # Open space inside support that the flood fill can occupy
    open_space = support & (~barrier)

    # Flood fill "outside" from the support-mask boundary, constrained to open_space
    seeds = boundary_seed(open_space)
    outside = binary_propagation(seeds, structure=structure, mask=open_space)

    # Enclosed cavity is whatever in open_space is NOT reachable from boundary
    enclosed = open_space & (~outside)

    # Remove tiny enclosed junk
    enclosed = remove_small_components(enclosed, min_enclosed_vox, structure=structure)

    # Keep the largest enclosed component as brain candidate
    brain = largest_component(enclosed, structure=structure)
    if not np.any(brain):
        raise RuntimeError("No enclosed brain candidate found. Try lowering grad threshold or barrier cleanup.")

    # Cleanup
    brain = binary_fill_holes(brain)
    if args.brain_close_iters > 0:
        brain = binary_closing(brain, structure=structure, iterations=args.brain_close_iters)
    if args.brain_dilate_iters > 0:
        brain = binary_dilation(brain, structure=structure, iterations=args.brain_dilate_iters)

    brain &= support
    brain = largest_component(brain, structure=structure)

    if args.tight_mask:
        brain = binary_erosion(brain, structure=structure, iterations=args.tight_erode_iters)
        brain = largest_component(brain, structure=structure)

    out_mask = brain.astype(np.uint8)
    out_masked_bfc = np.where(out_mask > 0, bfc, 0.0).astype(np.float32)

    save_like(Path(args.out_mask), out_mask, bfc_img)
    save_like(Path(args.out_masked_bfc), out_masked_bfc, bfc_img)

    if args.debug_prefix:
        prefix = Path(args.debug_prefix)
        save_like(prefix.with_name(prefix.name + "_grad.nii.gz"), grad.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_grad_norm.nii.gz"), grad_norm.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_barrier_raw.nii.gz"),
                  ((grad_norm >= args.grad_threshold) & support).astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_barrier.nii.gz"), barrier.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_open_space.nii.gz"), open_space.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_outside.nii.gz"), outside.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_enclosed.nii.gz"), enclosed.astype(np.uint8), bfc_img)


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
    brainmask_nii: Path,
    brain_t2_nii: Path,
    diff_nii: Path | None,
    tmp_otsu_pre_nii: Path,
    helper_py: Path,
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
    brain_smooth_sigma: float,
    brain_grad_sigma: float,
    brain_grad_threshold: float,
    brain_min_barrier_volume_mm3: float,
    brain_min_enclosed_volume_mm3: float,
    brain_close_barrier_iters: int,
    brain_close_iters: int,
    brain_dilate_iters: int,
    tight_mask: bool,
    tight_mask_erode_iters: int,
    save_brain_debug: bool,
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
brainmask_nii=""" + shell_quote(str(brainmask_nii)) + r"""
brain_t2_nii=""" + shell_quote(str(brain_t2_nii)) + r"""
tmp_otsu_pre_nii=""" + shell_quote(str(tmp_otsu_pre_nii)) + r"""
helper_py=""" + shell_quote(str(helper_py)) + r"""
overwrite_flag=""" + ("1" if overwrite else "0") + r"""
tight_mask_flag=""" + ("1" if tight_mask else "0") + r"""
save_brain_debug_flag=""" + ("1" if save_brain_debug else "0") + r"""
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

need_brainmask=0
if [[ "$overwrite_flag" == "1" ]]; then
    need_brainmask=1
elif [[ ! -f "$brainmask_nii" || ! -f "$brain_t2_nii" ]]; then
    need_brainmask=1
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

if [[ "$need_brainmask" == "1" ]]; then
    if [[ ! -f "$out_nii" ]]; then
        echo "ERROR: Missing corrected image for brain mask: $out_nii" >&2
        exit 1
    fi
    if [[ ! -f "$premask_nii" ]]; then
        echo "ERROR: Missing support mask for brain mask: $premask_nii" >&2
        exit 1
    fi

    echo
    echo "Creating shell-based brain mask..."
    brain_cmd=(
        python3 "$helper_py"
        --bfc "$out_nii"
        --support_mask "$premask_nii"
        --out_mask "$brainmask_nii"
        --out_masked_bfc "$brain_t2_nii"
        --smooth_sigma """ + str(brain_smooth_sigma) + r"""
        --grad_sigma """ + str(brain_grad_sigma) + r"""
        --grad_threshold """ + str(brain_grad_threshold) + r"""
        --min_barrier_volume_mm3 """ + str(brain_min_barrier_volume_mm3) + r"""
        --min_enclosed_volume_mm3 """ + str(brain_min_enclosed_volume_mm3) + r"""
        --close_barrier_iters """ + str(brain_close_barrier_iters) + r"""
        --brain_close_iters """ + str(brain_close_iters) + r"""
        --brain_dilate_iters """ + str(brain_dilate_iters) + r"""
        --tight_erode_iters """ + str(tight_mask_erode_iters) + r"""
    )

    if [[ "$tight_mask_flag" == "1" ]]; then
        brain_cmd+=( --tight_mask )
    fi

    if [[ "$save_brain_debug_flag" == "1" ]]; then
        brain_cmd+=( --debug_prefix "$(dirname "$brainmask_nii")/""" + runno + r"""_bfc_brain_dbg" )
    fi

    printf '  %q' "${brain_cmd[@]}"
    echo
    "${brain_cmd[@]}"

    if [[ ! -f "$brainmask_nii" || ! -f "$brain_t2_nii" ]]; then
        echo "ERROR: Brain-mask outputs missing." >&2
        exit 1
    fi
else
    echo
    echo "Skipping brain-mask stage; outputs exist and overwrite is off."
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

    p.add_argument("--n4_path", default="N4BiasFieldCorrection")
    p.add_argument("--thresholdimage_path", default="ThresholdImage")
    p.add_argument("--imagemath_path", default="ImageMath")
    p.add_argument("--printheader_path", default="PrintHeader")

    p.add_argument("--dimension", type=int, default=3, choices=[2, 3, 4])

    # Mouse-scale N4 defaults
    p.add_argument("--shrink_factor", type=int, default=1)
    p.add_argument("--convergence", default="[200x200x100x50,1e-8]")
    p.add_argument("--bspline", default="[8]")
    p.add_argument("--histogram_sharpening", default="[0.15,0.01,200]")

    # Pre-N4 support mask
    p.add_argument("--pre_otsu_keep_low", type=int, default=2)
    p.add_argument("--pre_otsu_keep_high", type=int, default=4)
    p.add_argument("--pre_close_radius", type=int, default=2)
    p.add_argument("--pre_dilate_radius_pre_glc", type=int, default=1)
    p.add_argument("--pre_fill_holes_radius", type=int, default=2)
    p.add_argument("--pre_dilate_radius_final", type=int, default=2)

    # Shell-based brain-mask parameters
    p.add_argument("--brain_smooth_sigma", type=float, default=1.0)
    p.add_argument("--brain_grad_sigma", type=float, default=1.0)
    p.add_argument("--brain_grad_threshold", type=float, default=0.10)
    p.add_argument("--brain_min_barrier_volume_mm3", type=float, default=0.05)
    p.add_argument("--brain_min_enclosed_volume_mm3", type=float, default=1.0)
    p.add_argument("--brain_close_barrier_iters", type=int, default=1)
    p.add_argument("--brain_close_iters", type=int, default=2)
    p.add_argument("--brain_dilate_iters", type=int, default=0)

    p.add_argument("--tight_mask", action="store_true")
    p.add_argument("--tight_mask_erode_iters", type=int, default=1)

    p.add_argument("--save_brain_debug", action="store_true")

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

    helper_py = sbatch_dir / "make_brain_mask_from_bfc.py"
    write_brainmask_helper(helper_py)

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
        brainmask_nii = run_dir / f"{runno}_bfc_brain_mask.nii.gz"
        brain_t2_nii = run_dir / f"{runno}_bfc_brain_T2.nii.gz"
        tmp_otsu_pre_nii = run_dir / f"{runno}_bfc_otsu_pre_tmp.nii.gz"
        diff_nii = run_dir / f"{runno}_bfc_diff_T2.nii.gz" if args.save_diff else None

        if not in_nii.exists():
            eprint(f"[MISSING] Input NIfTI not found for runno {runno}: {in_nii}")
            n_missing += 1
            continue

        if not args.overwrite:
            have_core = out_nii.exists() and bias_nii.exists() and premask_nii.exists()
            have_brain = brainmask_nii.exists() and brain_t2_nii.exists()
            have_diff = True if diff_nii is None else diff_nii.exists()
            if have_core and have_brain and have_diff:
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
            brainmask_nii=brainmask_nii,
            brain_t2_nii=brain_t2_nii,
            diff_nii=diff_nii,
            tmp_otsu_pre_nii=tmp_otsu_pre_nii,
            helper_py=helper_py,
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
            brain_smooth_sigma=args.brain_smooth_sigma,
            brain_grad_sigma=args.brain_grad_sigma,
            brain_grad_threshold=args.brain_grad_threshold,
            brain_min_barrier_volume_mm3=args.brain_min_barrier_volume_mm3,
            brain_min_enclosed_volume_mm3=args.brain_min_enclosed_volume_mm3,
            brain_close_barrier_iters=args.brain_close_barrier_iters,
            brain_close_iters=args.brain_close_iters,
            brain_dilate_iters=args.brain_dilate_iters,
            tight_mask=args.tight_mask,
            tight_mask_erode_iters=args.tight_mask_erode_iters,
            save_brain_debug=args.save_brain_debug,
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