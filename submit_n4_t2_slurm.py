#!/usr/bin/env python3
"""
submit_n4_t2_slurm.py

Submit one Slurm job per runno to run ANTs N4 bias field correction on:

    all_niis/z${runno}/${runno}_T2.nii.gz

Output:
    all_niis/z${runno}/${runno}_bfc_T2.nii.gz

Also copies:
    all_niis/z${runno}/${runno}_T2.method
to:
    all_niis/z${runno}/${runno}_bfc_T2.method

Examples
--------
# Submit specific runnos
python submit_n4_t2_slurm.py \
    --base_dir /mnt/newStor/paros/paros_MRI/DennisTurner/all_niis \
    --runnos 12345 12346 12347

# Submit from a file with one runno per line
python submit_n4_t2_slurm.py \
    --base_dir /mnt/newStor/paros/paros_MRI/DennisTurner/all_niis \
    --runno_file runnos.txt

# Auto-discover runnos from all_niis/z*/
python submit_n4_t2_slurm.py \
    --base_dir /mnt/newStor/paros/paros_MRI/DennisTurner/all_niis \
    --discover

# Dry run
python submit_n4_t2_slurm.py \
    --base_dir /mnt/newStor/paros/paros_MRI/DennisTurner/all_niis \
    --discover \
    --dry_run
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
        if not d.is_dir():
            continue
        runno = d.name[1:]  # strip leading z
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


def build_job_script(
    runno: str,
    in_nii: Path,
    out_nii: Path,
    in_method: Path,
    out_method: Path,
    n4_path: str,
    dimension: int,
    shrink_factor: int,
    convergence: str,
    bspline: str,
    histogram_sharpening: str,
    mask_path: str | None,
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
n4_exe=""" + shell_quote(str(n4_path)) + r"""

mkdir -p "$(dirname "$out_nii")"

cmd=(
    "$n4_exe"
    -d """ + str(dimension) + r"""
    -i "$in_nii"
    -s """ + str(shrink_factor) + r"""
    -c """ + shell_quote(convergence) + r"""
    -b """ + shell_quote(bspline) + r"""
    -t """ + shell_quote(histogram_sharpening) + r"""
    -o "$out_nii"
)
"""

    if mask_path:
        script += 'cmd+=( -x ' + shell_quote(mask_path) + ' )\n'

    script += r'''
echo "Running N4BiasFieldCorrection..."
printf '  %q' "${cmd[@]}"
echo
"${cmd[@]}"

if [[ ! -f "$out_nii" ]]; then
    echo "ERROR: Expected output not found: $out_nii" >&2
    exit 1
fi

if [[ -f "$in_method" ]]; then
    cp -f "$in_method" "$out_method"
    echo "Copied method file: $out_method"
else
    echo "WARNING: Method file not found, skipping copy: $in_method" >&2
fi

echo "===== JOB END ====="
date
'''
    return script


def shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def submit_job(script_path: Path):
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


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
    group.add_argument(
        "--runnos",
        nargs="+",
        help="One or more runnos."
    )
    group.add_argument(
        "--runno_file",
        help="Text file containing one runno per line."
    )
    group.add_argument(
        "--discover",
        action="store_true",
        help="Discover runnos from z* directories under base_dir."
    )

    parser.add_argument(
        "--sbatch_dir",
        default=None,
        help="Directory to store sbatch scripts and Slurm logs. Default: <base_dir>/n4_t2_sbatch"
    )
    parser.add_argument(
        "--n4_path",
        default="N4BiasFieldCorrection",
        help="Path to N4BiasFieldCorrection executable."
    )
    parser.add_argument(
        "--mask",
        default=None,
        help="Optional mask image to use for all jobs."
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=3,
        choices=[2, 3, 4],
        help="Image dimension. Default: 3"
    )
    parser.add_argument(
        "--shrink_factor",
        type=int,
        default=4,
        help="N4 shrink factor. Default: 4"
    )
    parser.add_argument(
        "--convergence",
        default="[50x50x50x50,1e-7]",
        help='N4 convergence string. Default: "[50x50x50x50,1e-7]"'
    )
    parser.add_argument(
        "--bspline",
        default="[200]",
        help='N4 bspline string. Default: "[200]"'
    )
    parser.add_argument(
        "--histogram_sharpening",
        default="[0.15,0.01,200]",
        help='N4 histogram sharpening string. Default: "[0.15,0.01,200]"'
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="ITK thread count inside each job. Default: 4"
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="Slurm cpus-per-task. Default: 4"
    )
    parser.add_argument(
        "--mem_gb",
        type=int,
        default=16,
        help="Slurm memory in GB. Default: 16"
    )
    parser.add_argument(
        "--time",
        default="04:00:00",
        help='Slurm time limit. Default: "04:00:00"'
    )
    parser.add_argument(
        "--partition",
        default=None,
        help="Optional Slurm partition."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Write sbatch scripts but do not submit."
    )

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

    print(f"[INFO] base_dir   : {base_dir}")
    print(f"[INFO] sbatch_dir : {sbatch_dir}")
    print(f"[INFO] runnos     : {len(runnos)}")

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

        if not in_nii.exists():
            eprint(f"[MISSING] Input NIfTI not found for runno {runno}: {in_nii}")
            n_missing += 1
            continue

        if out_nii.exists() and not args.overwrite:
            print(f"[SKIP] Output exists for runno {runno}: {out_nii}")
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
            n4_path=args.n4_path,
            dimension=args.dimension,
            shrink_factor=args.shrink_factor,
            convergence=args.convergence,
            bspline=args.bspline,
            histogram_sharpening=args.histogram_sharpening,
            mask_path=args.mask,
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