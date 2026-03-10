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


def write_prodmask_helper(helper_path: Path):
    helper_code = r'''#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_dilation,
    binary_erosion,
    center_of_mass,
    gaussian_filter,
    label,
    generate_binary_structure,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bfc", required=True)
    p.add_argument("--support_mask", required=True)
    p.add_argument("--out_mask", required=True)
    p.add_argument("--out_masked_bfc", required=True)
    p.add_argument("--norm_low_pct", type=float, default=2.0)
    p.add_argument("--norm_high_pct", type=float, default=98.0)
    p.add_argument("--threshold_frac", type=float, default=0.12)
    p.add_argument("--smooth_sigma", type=float, default=0.5)
    p.add_argument("--close_iters", type=int, default=2)
    p.add_argument("--dilate_iters", type=int, default=1)
    p.add_argument("--tight_mask", action="store_true")
    p.add_argument("--tight_erode_iters", type=int, default=1)
    args = p.parse_args()

    bfc_img = nib.load(args.bfc)
    sup_img = nib.load(args.support_mask)

    bfc = bfc_img.get_fdata(dtype=np.float32)
    support = sup_img.get_fdata() > 0

    if not np.any(support):
        raise RuntimeError("Support mask is empty.")

    masked_bfc = np.where(support, bfc, 0.0).astype(np.float32)

    smoothed = gaussian_filter(masked_bfc, sigma=args.smooth_sigma)

    vals = smoothed[support]
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        raise RuntimeError("No finite values inside support mask.")

    p_lo = np.percentile(vals, args.norm_low_pct)
    p_hi = np.percentile(vals, args.norm_high_pct)

    if p_hi <= p_lo:
        raise RuntimeError(f"Bad percentile range: low={p_lo}, high={p_hi}")

    norm = (smoothed - p_lo) / (p_hi - p_lo)
    norm = np.clip(norm, 0.0, 1.0)

    # Threshold only inside support mask
    cand = (norm >= args.threshold_frac) & support

    if not np.any(cand):
        # Fallback: just use support mask if thresholding somehow collapses
        cand = support.copy()

    struct = generate_binary_structure(3, 2)
    lab, nlab = label(cand, structure=struct)

    if nlab == 0:
        final_mask = support.copy()
    else:
        com = center_of_mass(support.astype(np.uint8))
        com_idx = tuple(int(round(x)) for x in com)
        com_idx = tuple(
            min(max(0, com_idx[i]), lab.shape[i] - 1) for i in range(3)
        )

        chosen_label = lab[com_idx]

        if chosen_label == 0:
            # If COM falls in a gap, choose the labeled component with greatest overlap
            # with the support mask center neighborhood.
            overlaps = []
            for k in range(1, nlab + 1):
                comp = (lab == k)
                overlaps.append((np.count_nonzero(comp & support), k))
            overlaps.sort(reverse=True)
            chosen_label = overlaps[0][1]

        final_mask = (lab == chosen_label)

    final_mask = binary_closing(final_mask, structure=struct, iterations=args.close_iters)
    final_mask = binary_fill_holes(final_mask)
    final_mask = binary_dilation(final_mask, structure=struct, iterations=args.dilate_iters)

    # Always constrain back to support mask
    final_mask = final_mask & support

    if args.tight_mask:
        final_mask = binary_erosion(
            final_mask, structure=struct, iterations=args.tight_erode_iters
        )

    out_mask = final_mask.astype(np.uint8)
    out_masked_bfc = np.where(out_mask > 0, bfc, 0.0).astype(np.float32)

    nib.save(nib.Nifti1Image(out_mask, bfc_img.affine, bfc_img.header), args.out_mask)
    nib.save(nib.Nifti1Image(out_masked_bfc, bfc_img.affine, bfc_img.header), args.out_masked_bfc)


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
    prodmask_nii: Path,
    masked_bfc_nii: Path,
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
    prod_norm_low_pct: float,
    prod_norm_high_pct: float,
    prod_threshold_frac: float,
    prod_smooth_sigma: float,
    prod_close_iters: int,
    prod_dilate_iters: int,
    tight_mask: bool,
    tight_mask_erode_iters: int,
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
prodmask_nii=""" + shell_quote(str(prodmask_nii)) + r"""
masked_bfc_nii=""" + shell_quote(str(masked_bfc_nii)) + r"""
tmp_otsu_pre_nii=""" + shell_quote(str(tmp_otsu_pre_nii)) + r"""
helper_py=""" + shell_quote(str(helper_py)) + r"""
overwrite_flag=""" + ("1" if overwrite else "0") + r"""
tight_mask_flag=""" + ("1" if tight_mask else "0") + r"""
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

need_prodmask=0
if [[ "$overwrite_flag" == "1" ]]; then
    need_prodmask=1
elif [[ ! -f "$prodmask_nii" || ! -f "$masked_bfc_nii" ]]; then
    need_prodmask=1
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

if [[ "$need_prodmask" == "1" ]]; then
    if [[ ! -f "$out_nii" ]]; then
        echo "ERROR: Missing corrected image for production mask: $out_nii" >&2
        exit 1
    fi
    if [[ ! -f "$premask_nii" ]]; then
        echo "ERROR: Missing support mask for production mask: $premask_nii" >&2
        exit 1
    fi

    echo
    echo "Creating seeded/support-constrained production mask..."
    prod_cmd=(
        python3 "$helper_py"
        --bfc "$out_nii"
        --support_mask "$premask_nii"
        --out_mask "$prodmask_nii"
        --out_masked_bfc "$masked_bfc_nii"
        --norm_low_pct """ + str(prod_norm_low_pct) + r"""
        --norm_high_pct """ + str(prod_norm_high_pct) + r"""
        --threshold_frac """ + str(prod_threshold_frac) + r"""
        --smooth_sigma """ + str(prod_smooth_sigma) + r"""
        --close_iters """ + str(prod_close_iters) + r"""
        --dilate_iters """ + str(prod_dilate_iters) + r"""
        --tight_erode_iters """ + str(tight_mask_erode_iters) + r"""
    )

    if [[ "$tight_mask_flag" == "1" ]]; then
        prod_cmd+=( --tight_mask )
    fi

    printf '  %q' "${prod_cmd[@]}"
    echo
    "${prod_cmd[@]}"

    if [[ ! -f "$prodmask_nii" ]]; then
        echo "ERROR: Production mask was not created." >&2
        exit 1
    fi
else
    echo
    echo "Skipping production mask stage; outputs exist and overwrite is off."
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

    # Pre-mask
    p.add_argument("--pre_otsu_keep_low", type=int, default=2)
    p.add_argument("--pre_otsu_keep_high", type=int, default=4)
    p.add_argument("--pre_close_radius", type=int, default=2)
    p.add_argument("--pre_dilate_radius_pre_glc", type=int, default=1)
    p.add_argument("--pre_fill_holes_radius", type=int, default=2)
    p.add_argument("--pre_dilate_radius_final", type=int, default=2)

    # Production mask refinement
    p.add_argument("--prod_norm_low_pct", type=float, default=2.0)
    p.add_argument("--prod_norm_high_pct", type=float, default=98.0)
    p.add_argument("--prod_threshold_frac", type=float, default=0.12)
    p.add_argument("--prod_smooth_sigma", type=float, default=0.5)
    p.add_argument("--prod_close_iters", type=int, default=2)
    p.add_argument("--prod_dilate_iters", type=int, default=1)

    p.add_argument("--tight_mask", action="store_true")
    p.add_argument("--tight_mask_erode_iters", type=int, default=1)

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

    helper_py = sbatch_dir / "make_prod_mask.py"
    write_prodmask_helper(helper_py)

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
        prodmask_nii = run_dir / f"{runno}_bfc_T2_mask.nii.gz"
        masked_bfc_nii = run_dir / f"{runno}_bfc_T2_masked.nii.gz"
        tmp_otsu_pre_nii = run_dir / f"{runno}_bfc_otsu_pre_tmp.nii.gz"
        diff_nii = run_dir / f"{runno}_bfc_diff_T2.nii.gz" if args.save_diff else None

        if not in_nii.exists():
            eprint(f"[MISSING] Input NIfTI not found for runno {runno}: {in_nii}")
            n_missing += 1
            continue

        if not args.overwrite:
            have_core = out_nii.exists() and bias_nii.exists() and premask_nii.exists()
            have_prodmask = prodmask_nii.exists() and masked_bfc_nii.exists()
            have_diff = True if diff_nii is None else diff_nii.exists()
            if have_core and have_prodmask and have_diff:
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
            prodmask_nii=prodmask_nii,
            masked_bfc_nii=masked_bfc_nii,
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
            prod_norm_low_pct=args.prod_norm_low_pct,
            prod_norm_high_pct=args.prod_norm_high_pct,
            prod_threshold_frac=args.prod_threshold_frac,
            prod_smooth_sigma=args.prod_smooth_sigma,
            prod_close_iters=args.prod_close_iters,
            prod_dilate_iters=args.prod_dilate_iters,
            tight_mask=args.tight_mask,
            tight_mask_erode_iters=args.tight_mask_erode_iters,
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