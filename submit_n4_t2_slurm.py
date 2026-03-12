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
    binary_opening,
    distance_transform_edt,
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


def constrained_region_grow(seed: np.ndarray, allowed: np.ndarray, structure, max_iters: int) -> np.ndarray:
    grown = seed.copy()
    for _ in range(max_iters):
        nxt = binary_dilation(grown, structure=structure) & allowed
        if np.array_equal(nxt, grown):
            break
        grown = nxt
    return grown


def save_like(path: Path, data: np.ndarray, ref_img: nib.Nifti1Image):
    img = nib.Nifti1Image(data, ref_img.affine, ref_img.header)
    nib.save(img, str(path))


def main():
    ap = argparse.ArgumentParser(
        description="Build a brain mask from BFC T2 using a dark-moat exclusion model plus optional targeted brainstem rescue."
    )
    ap.add_argument("--bfc", required=True)
    ap.add_argument("--support_mask", required=True)
    ap.add_argument("--out_mask", required=True)
    ap.add_argument("--out_masked_bfc", required=True)
    ap.add_argument("--debug_prefix", default=None)

    ap.add_argument("--smooth_sigma", type=float, default=1.0)
    ap.add_argument("--grad_sigma", type=float, default=1.0)

    # Dark moat
    ap.add_argument("--dark_moat_threshold", type=float, default=0.20)
    ap.add_argument("--min_moat_volume_mm3", type=float, default=0.03)
    ap.add_argument("--moat_close_iters", type=int, default=1)

    # Main brain seed / growth
    ap.add_argument("--seed_min_distance_mm", type=float, default=0.8,
                    help="Minimum distance from support boundary for central seed search.")
    ap.add_argument("--seed_intensity_min", type=float, default=0.35)
    ap.add_argument("--grow_intensity_min", type=float, default=0.08)
    ap.add_argument("--grow_gradient_max", type=float, default=0.80)
    ap.add_argument("--grow_iters", type=int, default=200)

    # Optional slight shell add-back after moat-constrained growth
    ap.add_argument("--shell_add_mm", type=float, default=0.10)
    ap.add_argument("--shell_add_intensity_min", type=float, default=0.06)

    # Brainstem-only rescue
    ap.add_argument("--do_brainstem_rescue", action="store_true")
    ap.add_argument("--brainstem_seed_slice_frac", type=float, default=0.18)
    ap.add_argument("--brainstem_corridor_radius_mm", type=float, default=1.0)
    ap.add_argument("--brainstem_max_inferior_mm", type=float, default=3.0)
    ap.add_argument("--brainstem_intensity_min", type=float, default=0.04)
    ap.add_argument("--brainstem_gradient_max", type=float, default=0.80)
    ap.add_argument("--brainstem_iters", type=int, default=50)

    # Final morphology
    ap.add_argument("--brain_close_iters", type=int, default=2)
    ap.add_argument("--brain_open_iters", type=int, default=0)
    ap.add_argument("--brain_dilate_iters", type=int, default=0)
    ap.add_argument("--brain_fill_holes", action="store_true")

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
    seed_min_distance_vox = max(1.0, args.seed_min_distance_mm / max(mean_spacing, 1e-6))
    shell_add_vox = max(1.0, args.shell_add_mm / max(mean_spacing, 1e-6))
    brainstem_corridor_radius_vox = max(1.0, args.brainstem_corridor_radius_mm / max(mean_spacing, 1e-6))
    brainstem_max_inferior_vox = max(1.0, args.brainstem_max_inferior_mm / max(mean_spacing, 1e-6))
    min_moat_vox = max(1, int(round(args.min_moat_volume_mm3 / voxel_volume_mm3)))

    structure = generate_binary_structure(3, 2)

    bfc_smooth = gaussian_filter(bfc, sigma=args.smooth_sigma)
    inten_norm = robust_normalize(bfc_smooth, support, low_pct=2.0, high_pct=98.0)

    grad = gaussian_gradient_magnitude(bfc_smooth, sigma=args.grad_sigma)
    grad_norm = robust_normalize(grad, support, low_pct=2.0, high_pct=98.0)

    # 1) Dark moat
    moat_raw = (inten_norm <= args.dark_moat_threshold) & support
    moat = remove_small_components(moat_raw, min_moat_vox, structure=structure)
    if args.moat_close_iters > 0 and np.any(moat):
        moat = binary_closing(moat, structure=structure, iterations=args.moat_close_iters)

    # 2) Central bright seed
    dist_to_support_edge = distance_transform_edt(support)
    seed_region = support & (dist_to_support_edge >= seed_min_distance_vox) & (inten_norm >= args.seed_intensity_min)
    seed_region &= (~moat)
    seed = largest_component(seed_region, structure=structure)

    if not np.any(seed):
        # fallback: relax seed threshold
        seed_region = support & (dist_to_support_edge >= max(1.0, 0.5 * seed_min_distance_vox)) & (inten_norm >= max(0.2, args.seed_intensity_min * 0.7))
        seed_region &= (~moat)
        seed = largest_component(seed_region, structure=structure)

    if not np.any(seed):
        raise RuntimeError("No valid central seed found. Try lowering seed threshold.")

    # 3) Main growth constrained by support and NOT moat
    allowed = support & (~moat)
    grow_gate = (inten_norm >= args.grow_intensity_min) & (grad_norm <= args.grow_gradient_max)
    allowed &= grow_gate
    allowed |= seed

    brain = constrained_region_grow(seed, allowed, structure, args.grow_iters)
    brain = largest_component(brain, structure=structure)

    # 4) Small shell add-back near grown brain only
    dist_to_brain = distance_transform_edt(~brain)
    shell_add = support & (dist_to_brain <= shell_add_vox) & (inten_norm >= args.shell_add_intensity_min) & (~moat)
    brain |= shell_add
    brain = largest_component(brain, structure=structure)

    # 5) Targeted brainstem rescue
    brainstem_seed = np.zeros_like(brain, dtype=bool)
    brainstem_allowed = np.zeros_like(brain, dtype=bool)

    if args.do_brainstem_rescue and np.any(brain):
        coords = np.argwhere(brain)
        zmin = coords[:, 2].min()
        zmax = coords[:, 2].max()
        zspan = max(1, zmax - zmin + 1)
        seed_depth = max(1, int(round(args.brainstem_seed_slice_frac * zspan)))
        z_cut = zmax - seed_depth + 1

        inferior_band = brain & (np.indices(brain.shape)[2] >= z_cut)
        inferior_band = largest_component(inferior_band, structure=structure)
        brainstem_seed = inferior_band.copy()

        if np.any(brainstem_seed):
            seed_coords = np.argwhere(brainstem_seed)
            cy, cx, cz = seed_coords.mean(axis=0)

            yy, xx, zz = np.indices(brain.shape)
            rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

            inferior_limit = zz <= (seed_coords[:, 2].max() + brainstem_max_inferior_vox)
            corridor = support & (rr <= brainstem_corridor_radius_vox) & inferior_limit

            tissue_gate = (inten_norm >= args.brainstem_intensity_min) & (grad_norm <= args.brainstem_gradient_max) & (~moat)
            brainstem_allowed = corridor & tissue_gate
            brainstem_allowed |= brainstem_seed

            brainstem_grown = constrained_region_grow(brainstem_seed, brainstem_allowed, structure, args.brainstem_iters)
            brain |= brainstem_grown
            brain = largest_component(brain, structure=structure)

    # 6) Cleanup
    if args.brain_fill_holes:
        brain = binary_fill_holes(brain)

    if args.brain_close_iters > 0:
        brain = binary_closing(brain, structure=structure, iterations=args.brain_close_iters)

    if args.brain_open_iters > 0:
        brain = binary_opening(brain, structure=structure, iterations=args.brain_open_iters)

    if args.brain_dilate_iters > 0:
        brain = binary_dilation(brain, structure=structure, iterations=args.brain_dilate_iters)

    brain &= support
    brain = largest_component(brain, structure=structure)
    brain = binary_fill_holes(brain)
    brain = largest_component(brain, structure=structure)

    if args.tight_mask:
        brain = binary_erosion(brain, structure=structure, iterations=args.tight_erode_iters)
        brain = largest_component(brain, structure=structure)
        brain = binary_fill_holes(brain)

    out_mask = brain.astype(np.uint8)
    out_masked_bfc = np.where(out_mask > 0, bfc, 0.0).astype(np.float32)

    save_like(Path(args.out_mask), out_mask, bfc_img)
    save_like(Path(args.out_masked_bfc), out_masked_bfc, bfc_img)

    if args.debug_prefix:
        prefix = Path(args.debug_prefix)
        save_like(prefix.with_name(prefix.name + "_inten_norm.nii.gz"), inten_norm.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_grad_norm.nii.gz"), grad_norm.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_moat_raw.nii.gz"), moat_raw.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_moat.nii.gz"), moat.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_seed_region.nii.gz"), seed_region.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_seed.nii.gz"), seed.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_allowed.nii.gz"), allowed.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_dist_to_support_edge.nii.gz"), dist_to_support_edge.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_dist_to_brain.nii.gz"), dist_to_brain.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_shell_add.nii.gz"), shell_add.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_brainstem_seed.nii.gz"), brainstem_seed.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_brainstem_allowed.nii.gz"), brainstem_allowed.astype(np.uint8), bfc_img)


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
    brain_smooth_sigma: float,
    brain_grad_sigma: float,
    brain_dark_moat_threshold: float,
    brain_min_moat_volume_mm3: float,
    brain_moat_close_iters: int,
    brain_seed_min_distance_mm: float,
    brain_seed_intensity_min: float,
    brain_grow_intensity_min: float,
    brain_grow_gradient_max: float,
    brain_grow_iters: int,
    brain_shell_add_mm: float,
    brain_shell_add_intensity_min: float,
    do_brainstem_rescue: bool,
    brainstem_seed_slice_frac: float,
    brainstem_corridor_radius_mm: float,
    brainstem_max_inferior_mm: float,
    brainstem_intensity_min: float,
    brainstem_gradient_max: float,
    brainstem_iters: int,
    brain_close_iters: int,
    brain_open_iters: int,
    brain_dilate_iters: int,
    brain_fill_holes: bool,
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
python_exe=""" + shell_quote(str(python_path)) + r"""
overwrite_flag=""" + ("1" if overwrite else "0") + r"""
tight_mask_flag=""" + ("1" if tight_mask else "0") + r"""
save_brain_debug_flag=""" + ("1" if save_brain_debug else "0") + r"""
brain_fill_holes_flag=""" + ("1" if brain_fill_holes else "0") + r"""
do_brainstem_rescue_flag=""" + ("1" if do_brainstem_rescue else "0") + r"""
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
    echo "Creating dark-moat brain mask..."
    brain_cmd=(
        "$python_exe" "$helper_py"
        --bfc "$out_nii"
        --support_mask "$premask_nii"
        --out_mask "$brainmask_nii"
        --out_masked_bfc "$brain_t2_nii"
        --smooth_sigma """ + str(brain_smooth_sigma) + r"""
        --grad_sigma """ + str(brain_grad_sigma) + r"""
        --dark_moat_threshold """ + str(brain_dark_moat_threshold) + r"""
        --min_moat_volume_mm3 """ + str(brain_min_moat_volume_mm3) + r"""
        --moat_close_iters """ + str(brain_moat_close_iters) + r"""
        --seed_min_distance_mm """ + str(brain_seed_min_distance_mm) + r"""
        --seed_intensity_min """ + str(brain_seed_intensity_min) + r"""
        --grow_intensity_min """ + str(brain_grow_intensity_min) + r"""
        --grow_gradient_max """ + str(brain_grow_gradient_max) + r"""
        --grow_iters """ + str(brain_grow_iters) + r"""
        --shell_add_mm """ + str(brain_shell_add_mm) + r"""
        --shell_add_intensity_min """ + str(brain_shell_add_intensity_min) + r"""
        --brainstem_seed_slice_frac """ + str(brainstem_seed_slice_frac) + r"""
        --brainstem_corridor_radius_mm """ + str(brainstem_corridor_radius_mm) + r"""
        --brainstem_max_inferior_mm """ + str(brainstem_max_inferior_mm) + r"""
        --brainstem_intensity_min """ + str(brainstem_intensity_min) + r"""
        --brainstem_gradient_max """ + str(brainstem_gradient_max) + r"""
        --brainstem_iters """ + str(brainstem_iters) + r"""
        --brain_close_iters """ + str(brain_close_iters) + r"""
        --brain_open_iters """ + str(brain_open_iters) + r"""
        --brain_dilate_iters """ + str(brain_dilate_iters) + r"""
        --tight_erode_iters """ + str(tight_mask_erode_iters) + r"""
    )

    if [[ "$brain_fill_holes_flag" == "1" ]]; then
        brain_cmd+=( --brain_fill_holes )
    fi

    if [[ "$do_brainstem_rescue_flag" == "1" ]]; then
        brain_cmd+=( --do_brainstem_rescue )
    fi

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

    # Dark-moat mask parameters
    p.add_argument("--brain_smooth_sigma", type=float, default=1.0)
    p.add_argument("--brain_grad_sigma", type=float, default=1.0)
    p.add_argument("--brain_dark_moat_threshold", type=float, default=0.20)
    p.add_argument("--brain_min_moat_volume_mm3", type=float, default=0.03)
    p.add_argument("--brain_moat_close_iters", type=int, default=1)

    p.add_argument("--brain_seed_min_distance_mm", type=float, default=0.8)
    p.add_argument("--brain_seed_intensity_min", type=float, default=0.35)
    p.add_argument("--brain_grow_intensity_min", type=float, default=0.08)
    p.add_argument("--brain_grow_gradient_max", type=float, default=0.80)
    p.add_argument("--brain_grow_iters", type=int, default=200)

    p.add_argument("--brain_shell_add_mm", type=float, default=0.10)
    p.add_argument("--brain_shell_add_intensity_min", type=float, default=0.06)

    p.add_argument("--do_brainstem_rescue", action="store_true")
    p.add_argument("--brainstem_seed_slice_frac", type=float, default=0.18)
    p.add_argument("--brainstem_corridor_radius_mm", type=float, default=1.0)
    p.add_argument("--brainstem_max_inferior_mm", type=float, default=3.0)
    p.add_argument("--brainstem_intensity_min", type=float, default=0.04)
    p.add_argument("--brainstem_gradient_max", type=float, default=0.80)
    p.add_argument("--brainstem_iters", type=int, default=50)

    p.add_argument("--brain_close_iters", type=int, default=2)
    p.add_argument("--brain_open_iters", type=int, default=0)
    p.add_argument("--brain_dilate_iters", type=int, default=0)
    p.add_argument("--brain_fill_holes", action="store_true")

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
            brain_smooth_sigma=args.brain_smooth_sigma,
            brain_grad_sigma=args.brain_grad_sigma,
            brain_dark_moat_threshold=args.brain_dark_moat_threshold,
            brain_min_moat_volume_mm3=args.brain_min_moat_volume_mm3,
            brain_moat_close_iters=args.brain_moat_close_iters,
            brain_seed_min_distance_mm=args.brain_seed_min_distance_mm,
            brain_seed_intensity_min=args.brain_seed_intensity_min,
            brain_grow_intensity_min=args.brain_grow_intensity_min,
            brain_grow_gradient_max=args.brain_grow_gradient_max,
            brain_grow_iters=args.brain_grow_iters,
            brain_shell_add_mm=args.brain_shell_add_mm,
            brain_shell_add_intensity_min=args.brain_shell_add_intensity_min,
            do_brainstem_rescue=args.do_brainstem_rescue,
            brainstem_seed_slice_frac=args.brainstem_seed_slice_frac,
            brainstem_corridor_radius_mm=args.brainstem_corridor_radius_mm,
            brainstem_max_inferior_mm=args.brainstem_max_inferior_mm,
            brainstem_intensity_min=args.brainstem_intensity_min,
            brainstem_gradient_max=args.brainstem_gradient_max,
            brainstem_iters=args.brainstem_iters,
            brain_close_iters=args.brain_close_iters,
            brain_open_iters=args.brain_open_iters,
            brain_dilate_iters=args.brain_dilate_iters,
            brain_fill_holes=args.brain_fill_holes,
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