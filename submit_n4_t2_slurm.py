#!/usr/bin/env python3
"""
submit_n4_t2_slurm.py

Submit one Slurm job per runno to run ANTs N4 bias field correction on:

    all_niis/z${runno}/${runno}_T2.nii.gz

Outputs in the same folder:
    ${runno}_bfc_T2.nii.gz
    ${runno}_bfc_T2.method
    ${runno}_bfc_mask.nii.gz
    ${runno}_bfc_biasfield.nii.gz
    ${runno}_bfc_diff_T2.nii.gz   (optional)

This version:
- builds a more robust foreground mask
- uses mouse-scale N4 defaults
- saves the N4 bias field
- submits one Slurm job per runno
"""

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
            if not s or s.startswith("#"):
                continue
            runnos.append(s)
    return runnos


def discover_runnos(base_dir: Path):
    runnos = []
    for d in sorted(base_dir.glob("z*")):
        if d.is_dir():
            runno = d.name[1:]
            if runno:
                runnos.append(runno)
    return runnos


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


def build_job_script(
    runno: str,
    in_nii: Path,
    out_nii: Path,
    in_method: Path,
    out_method: Path,
    mask_nii: Path,
    bias_nii: Path,
    diff_nii: Path | None,
    tmp_otsu_nii: Path,
    n4_path: str,
    thresholdimage_path: str,
    imagemath_path: str,
    printheader_path: str | None,
    dimension: int,
    shrink_factor: int,
    convergence: str,
    bspline: str,
    histogram_sharpening: str,
    otsu_keep_low: int,
    otsu_keep_high: int,
    close_radius: int,
    dilate_radius_pre_glc: int,
    dilate_radius_final: int,
    fill_holes_radius: int,
    threads: int,
    job_name: str,
    log_out: Path,
    log_err: Path,
    partition: str | None,
    time_str: str,
    mem_gb: int,
    cpus: int,
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
mask_nii=""" + shell_quote(str(mask_nii)) + r"""
bias_nii=""" + shell_quote(str(bias_nii)) + r"""
tmp_otsu_nii=""" + shell_quote(str(tmp_otsu_nii)) + r"""
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

echo "Input      : $in_nii"
echo "Output     : $out_nii"
echo "Mask       : $mask_nii"
echo "Bias field : $bias_nii"
if [[ -n "$diff_nii" ]]; then
    echo "Diff image : $diff_nii"
fi

if [[ ! -f "$in_nii" ]]; then
    echo "ERROR: Input not found: $in_nii" >&2
    exit 1
fi

echo
echo "Header / spacing check:"
if [[ -n "$printheader_exe" ]]; then
    "$printheader_exe" "$in_nii" 1 || true
else
    echo "PrintHeader not provided; skipping explicit header dump."
fi

echo
echo "Creating robust foreground mask..."
rm -f "$tmp_otsu_nii" "$mask_nii"

# 1) Otsu label map
"$threshold_exe" """ + str(dimension) + r""" "$in_nii" "$tmp_otsu_nii" Otsu 4

# 2) Keep upper Otsu classes (foreground-ish)
"$threshold_exe" """ + str(dimension) + r""" "$tmp_otsu_nii" "$mask_nii" """ + str(otsu_keep_low) + r""" """ + str(otsu_keep_high) + r""" 1 0

# 3) Morphology to preserve low-signal anterior structures and smooth the mask
"$imagemath_exe" """ + str(dimension) + r""" "$mask_nii" MC "$mask_nii" """ + str(close_radius) + r"""
"$imagemath_exe" """ + str(dimension) + r""" "$mask_nii" MD "$mask_nii" """ + str(dilate_radius_pre_glc) + r"""

# 4) Fill holes
"$imagemath_exe" """ + str(dimension) + r""" "$mask_nii" FillHoles "$mask_nii" """ + str(fill_holes_radius) + r"""

# 5) Keep largest connected component
"$imagemath_exe" """ + str(dimension) + r""" "$mask_nii" GetLargestComponent "$mask_nii"

# 6) Final dilation to avoid clipping bulb / edge tissue
"$imagemath_exe" """ + str(dimension) + r""" "$mask_nii" MD "$mask_nii" """ + str(dilate_radius_final) + r"""

if [[ ! -f "$mask_nii" ]]; then
    echo "ERROR: Mask creation failed: $mask_nii" >&2
    exit 1
fi

echo
echo "Running N4BiasFieldCorrection..."
cmd=(
    "$n4_exe"
    -d """ + str(dimension) + r"""
    -i "$in_nii"
    -x "$mask_nii"
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

if [[ ! -f "$out_nii" ]]; then
    echo "ERROR: Expected corrected output not found: $out_nii" >&2
    exit 1
fi

if [[ ! -f "$bias_nii" ]]; then
    echo "ERROR: Expected bias field not found: $bias_nii" >&2
    exit 1
fi

if [[ -f "$in_method" ]]; then
    cp -f "$in_method" "$out_method"
    echo "Copied method file: $out_method"
else
    echo "WARNING: Method file not found, skipping copy: $in_method" >&2
fi

if [[ -n "$diff_nii" ]]; then
    echo
    echo "Creating difference image..."
    "$imagemath_exe" """ + str(dimension) + r""" "$diff_nii" - "$out_nii" "$in_nii"
fi

rm -f "$tmp_otsu_nii"

echo "===== JOB END ====="
date
"""
    return script


def main():
    parser = argparse.ArgumentParser(
        description="Submit one Slurm job per runno for N4 bias field correction of T2 images."
    )
    parser.add_argument(
        "--base_dir",
        required=True,
        help="Path to all_niis directory."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--runnos", nargs="+", help="One or more runnos.")
    group.add_argument("--runno_file", help="Text file containing one runno per line.")
    group.add_argument("--discover", action="store_true", help="Discover runnos from z* directories under base_dir.")

    parser.add_argument(
        "--sbatch_dir",
        default=None,
        help="Directory to store sbatch scripts and logs. Default: <base_dir>/n4_t2_sbatch"
    )

    parser.add_argument("--n4_path", default="N4BiasFieldCorrection", help="Path to N4BiasFieldCorrection executable.")
    parser.add_argument("--thresholdimage_path", default="ThresholdImage", help="Path to ThresholdImage executable.")
    parser.add_argument("--imagemath_path", default="ImageMath", help="Path to ImageMath executable.")
    parser.add_argument("--printheader_path", default="PrintHeader", help="Path to PrintHeader executable.")

    parser.add_argument("--dimension", type=int, default=3, choices=[2, 3, 4], help="Image dimension. Default: 3")

    # Mouse-scale defaults
    parser.add_argument("--shrink_factor", type=int, default=1, help="N4 shrink factor. Default: 1")
    parser.add_argument("--convergence", default="[200x200x100x50,1e-8]", help='N4 convergence string.')
    parser.add_argument("--bspline", default="[8]", help='N4 bspline distance in physical units. Mouse-scale default: "[8]"')
    parser.add_argument("--histogram_sharpening", default="[0.15,0.01,200]", help='N4 histogram sharpening string.')

    # Mask settings
    parser.add_argument("--otsu_keep_low", type=int, default=2, help="Lowest Otsu class to keep. Default: 2")
    parser.add_argument("--otsu_keep_high", type=int, default=4, help="Highest Otsu class to keep. Default: 4")
    parser.add_argument("--close_radius", type=int, default=2, help="Morphological closing radius. Default: 2")
    parser.add_argument("--dilate_radius_pre_glc", type=int, default=1, help="Pre-GLC dilation radius. Default: 1")
    parser.add_argument("--fill_holes_radius", type=int, default=2, help="Fill-holes radius. Default: 2")
    parser.add_argument("--dilate_radius_final", type=int, default=2, help="Final dilation radius. Default: 2")

    parser.add_argument("--threads", type=int, default=4, help="ITK thread count inside each job.")
    parser.add_argument("--cpus", type=int, default=4, help="Slurm cpus-per-task.")
    parser.add_argument("--mem_gb", type=int, default=16, help="Slurm memory in GB.")
    parser.add_argument("--time", default="04:00:00", help='Slurm time limit.')
    parser.add_argument("--partition", default=None, help="Optional Slurm partition.")

    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--save_diff", action="store_true", help="Also save corrected-minus-original difference image.")
    parser.add_argument("--dry_run", action="store_true", help="Write sbatch scripts but do not submit.")

    args = parser.parse_args()

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

    print(f"[INFO] base_dir    : {base_dir}")
    print(f"[INFO] sbatch_dir  : {sbatch_dir}")
    print(f"[INFO] runno count : {len(runnos)}")
    print(f"[INFO] Mouse-scale N4 defaults:")
    print(f"       shrink      = {args.shrink_factor}")
    print(f"       convergence = {args.convergence}")
    print(f"       bspline     = {args.bspline}")

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
        mask_nii = run_dir / f"{runno}_bfc_mask.nii.gz"
        bias_nii = run_dir / f"{runno}_bfc_biasfield.nii.gz"
        tmp_otsu_nii = run_dir / f"{runno}_bfc_otsu_tmp.nii.gz"
        diff_nii = run_dir / f"{runno}_bfc_diff_T2.nii.gz" if args.save_diff else None

        if not in_nii.exists():
            eprint(f"[MISSING] Input NIfTI not found for runno {runno}: {in_nii}")
            n_missing += 1
            continue

        outputs_to_check = [out_nii, mask_nii, bias_nii]
        if diff_nii is not None:
            outputs_to_check.append(diff_nii)

        if all(p.exists() for p in outputs_to_check) and not args.overwrite:
            print(f"[SKIP] Outputs already exist for runno {runno}")
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
            mask_nii=mask_nii,
            bias_nii=bias_nii,
            diff_nii=diff_nii,
            tmp_otsu_nii=tmp_otsu_nii,
            n4_path=args.n4_path,
            thresholdimage_path=args.thresholdimage_path,
            imagemath_path=args.imagemath_path,
            printheader_path=args.printheader_path,
            dimension=args.dimension,
            shrink_factor=args.shrink_factor,
            convergence=args.convergence,
            bspline=args.bspline,
            histogram_sharpening=args.histogram_sharpening,
            otsu_keep_low=args.otsu_keep_low,
            otsu_keep_high=args.otsu_keep_high,
            close_radius=args.close_radius,
            dilate_radius_pre_glc=args.dilate_radius_pre_glc,
            dilate_radius_final=args.dilate_radius_final,
            fill_holes_radius=args.fill_holes_radius,
            threads=args.threads,
            job_name=job_name,
            log_out=log_out,
            log_err=log_err,
            partition=args.partition,
            time_str=args.time,
            mem_gb=args.mem_gb,
            cpus=args.cpus,
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