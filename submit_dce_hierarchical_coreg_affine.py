#!/usr/bin/env python3
"""
Submit hierarchical per-run DCE coregistration jobs directly to Slurm.

This launcher pairs with:
    dce_hierarchical_coreg_affine.py

Workflow per runno:
    prep (single job per run)
    for each DCE nifti:
        local_reg array      : one task per volume in that nifti
        local_mean single    : depends on local_reg array
        mean_to_global single: depends on local_mean
        final_apply array    : one task per volume in that nifti
    finalize (single job per run): depends on all final_apply arrays

Input layout:
    all_niis/z<runno>/
        <runno>_DCE_baseline.nii.gz
        <runno>_DCE_block1.nii.gz
        <runno>_DCE_block2.nii.gz
        <runno>_T2.nii.gz

Output layout:
    all_niis/<runno>_coregistered/
        sbatch/
        work/
        <runno>_meanDCE_coregistered_allvols.nii.gz
        <runno>_T2.nii.gz
        <same-basename DCE outputs as inputs>

Usage example:
    python3 submit_dce_hierarchical_coreg_affine.py \
        --all-niis-dir /mnt/newStor/paros/paros_MRI/DennisTurner/all_niis \
        --pipeline /mnt/newStor/paros/paros_MRI/DennisTurner/dce_hierarchical_coreg_affine.py \
        --run-glob 'z*' \
        --mem 12000M \
        --partition normal \
        --affine-step 0.05 \
        --conv-iters '100x100x100x20' \
        --conv-thresh 1e-7 \
        --conv-window 15 \
        --shrink-factors '8x4x2x1' \
        --smoothing-sigmas '3x2x1x0vox'

Optional environment setup:
    --python /full/path/to/python3
    --env-setup 'source /path/to/conda.sh && conda activate myenv'

Notes:
- Uses direct sbatch submission.
- Writes all sbatch scripts into <runno>_coregistered/sbatch/
- Writes Slurm logs there as well.
- Uses named arguments and validates inputs before submission.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


DCE_SUFFIXES_DEFAULT = ["DCE_baseline", "DCE_block1", "DCE_block2"]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Submit hierarchical DCE coregistration jobs directly to Slurm."
    )

    ap.add_argument(
        "--all-niis-dir",
        required=True,
        help="Path to all_niis directory containing z<runno>/ input folders.",
    )
    ap.add_argument(
        "--pipeline",
        required=True,
        help="Path to dce_hierarchical_coreg_affine.py",
    )
    ap.add_argument(
        "--run-glob",
        default="z*",
        help="Glob for run directories under all_niis_dir (default: %(default)s)",
    )
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use on compute nodes (default: current interpreter)",
    )
    ap.add_argument(
        "--env-setup",
        default="",
        help="Optional shell snippet to run before python commands on compute nodes, "
             "e.g. 'source /path/to/conda.sh && conda activate myenv'",
    )

    # Slurm controls
    ap.add_argument(
        "--partition",
        default="normal",
        help="Slurm partition (default: %(default)s)",
    )
    ap.add_argument(
        "--mem",
        default="12000M",
        help="Memory request for all jobs (default: %(default)s)",
    )
    ap.add_argument(
        "--cpus-per-task",
        type=int,
        default=1,
        help="CPUs per task (default: %(default)s)",
    )
    ap.add_argument(
        "--time",
        default="0",
        help="Optional Slurm time limit, e.g. 08:00:00. Use 0 to omit (default: %(default)s)",
    )
    ap.add_argument(
        "--mail-user",
        default=os.environ.get("NOTIFICATION_EMAIL", ""),
        help="Email for Slurm notifications. Default: NOTIFICATION_EMAIL env var if set, else omitted.",
    )
    ap.add_argument(
        "--mail-type",
        default="END,FAIL",
        help="Slurm mail types if --mail-user is set (default: %(default)s)",
    )

    # Registration params
    ap.add_argument(
        "--affine-step",
        type=float,
        default=0.05,
        help="Affine step size for Affine[x.xx] (default: %(default)s)",
    )
    ap.add_argument(
        "--conv-iters",
        default="100x100x100x20",
        help="Convergence iterations per level (default: %(default)s)",
    )
    ap.add_argument(
        "--conv-thresh",
        type=float,
        default=1.0e-7,
        help="Convergence threshold (default: %(default)s)",
    )
    ap.add_argument(
        "--conv-window",
        type=int,
        default=15,
        help="Convergence window size (default: %(default)s)",
    )
    ap.add_argument(
        "--shrink-factors",
        default="8x4x2x1",
        help="Shrink factors schedule (default: %(default)s)",
    )
    ap.add_argument(
        "--smoothing-sigmas",
        default="3x2x1x0vox",
        help="Smoothing sigmas schedule (default: %(default)s)",
    )
    ap.add_argument(
        "--winsorize-lower",
        type=float,
        default=0.005,
        help="Lower winsorization bound (default: %(default)s)",
    )
    ap.add_argument(
        "--winsorize-upper",
        type=float,
        default=0.995,
        help="Upper winsorization bound (default: %(default)s)",
    )
    ap.add_argument(
        "--hist-match",
        type=int,
        default=0,
        choices=[0, 1],
        help="Pass -u to antsRegistration (default: %(default)s)",
    )

    # Discovery / behavior
    ap.add_argument(
        "--dce-suffixes",
        default=",".join(DCE_SUFFIXES_DEFAULT),
        help="Comma-separated DCE suffixes to look for (default: %(default)s)",
    )
    ap.add_argument(
        "--t2-suffix",
        default="T2",
        help="T2 suffix to look for (default: %(default)s)",
    )
    ap.add_argument(
        "--skip-complete-runs",
        action="store_true",
        help="Skip runs that already appear complete.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass through to pipeline stages if you want them to overwrite outputs.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Write sbatch scripts and print planned submissions, but do not submit.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose launcher logging and pass --verbose to pipeline.",
    )

    return ap.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    all_niis_dir = Path(args.all_niis_dir).resolve()
    pipeline = Path(args.pipeline).resolve()

    if not all_niis_dir.is_dir():
        raise SystemExit(f"--all-niis-dir is not a directory: {all_niis_dir}")
    if not pipeline.is_file():
        raise SystemExit(f"--pipeline not found: {pipeline}")

    if args.cpus_per_task < 1:
        raise SystemExit("--cpus-per-task must be >= 1")
    if args.conv_window < 1:
        raise SystemExit("--conv-window must be >= 1")
    if args.affine_step <= 0:
        raise SystemExit("--affine-step must be > 0")
    if not (0.0 <= args.winsorize_lower < args.winsorize_upper <= 1.0):
        raise SystemExit("--winsorize bounds must satisfy 0 <= lower < upper <= 1")

    for name, value in [
        ("conv-iters", args.conv_iters),
        ("shrink-factors", args.shrink_factors),
        ("smoothing-sigmas", args.smoothing_sigmas),
    ]:
        if not value or not isinstance(value, str):
            raise SystemExit(f"--{name} must be a non-empty string")


def discover_run_dirs(all_niis_dir: Path, run_glob: str) -> List[Path]:
    runs = sorted([p for p in all_niis_dir.glob(run_glob) if p.is_dir()])
    return runs


def parse_runno(run_dir: Path) -> str:
    bn = run_dir.name
    if bn.startswith("z") and len(bn) > 1:
        return bn[1:]
    return bn


def discover_dce_files(run_dir: Path, runno: str, dce_suffixes: List[str]) -> List[Path]:
    dce_files: List[Path] = []

    for suf in dce_suffixes:
        exact = run_dir / f"{runno}_{suf}.nii.gz"
        if exact.is_file():
            dce_files.append(exact)
            continue

        # fallback if naming is slightly off but suffix matches
        matches = sorted(run_dir.glob(f"*_{suf}.nii.gz"))
        if len(matches) == 1:
            dce_files.append(matches[0])
        elif len(matches) > 1:
            raise RuntimeError(
                f"Ambiguous DCE matches in {run_dir} for suffix '{suf}':\n  " +
                "\n  ".join(str(m) for m in matches)
            )

    return dce_files


def discover_t2_file(run_dir: Path, runno: str, t2_suffix: str) -> Optional[Path]:
    exact = run_dir / f"{runno}_{t2_suffix}.nii.gz"
    if exact.is_file():
        return exact

    matches = sorted(run_dir.glob(f"*_{t2_suffix}.nii.gz"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous T2 matches in {run_dir} for suffix '{t2_suffix}':\n  " +
            "\n  ".join(str(m) for m in matches)
        )
    return None


def count_4d_volumes(python_exe: str, nifti_path: Path) -> int:
    code = (
        "import nibabel as nib; "
        f"img = nib.load(r'''{str(nifti_path)}'''); "
        "shape = img.shape; "
        "assert len(shape) == 4, f'Expected 4D, got {shape}'; "
        "print(shape[3])"
    )
    proc = subprocess.run(
        [python_exe, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to read 4D volume count for {nifti_path}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return int(proc.stdout.strip())


def shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)


def build_pipeline_args(args: argparse.Namespace, run_dir: Path) -> List[str]:
    out = [
        "--run_dir", str(run_dir),
        "--dce_suffixes", args.dce_suffixes,
        "--t2_suffix", args.t2_suffix,
        "--affine_step", str(args.affine_step),
        "--conv_iters", args.conv_iters,
        "--conv_thresh", str(args.conv_thresh),
        "--conv_window", str(args.conv_window),
        "--shrink_factors", args.shrink_factors,
        "--smoothing_sigmas", args.smoothing_sigmas,
        "--winsorize_lower", str(args.winsorize_lower),
        "--winsorize_upper", str(args.winsorize_upper),
        "--hist_match", str(args.hist_match),
    ]
    if args.overwrite:
        out.append("--overwrite")
    if args.verbose:
        out.append("--verbose")
    return out


def make_job_script_text(
    *,
    job_name: str,
    partition: str,
    mem: str,
    cpus_per_task: int,
    output_path: Path,
    command_body: str,
    time_limit: str = "0",
    mail_user: str = "",
    mail_type: str = "END,FAIL",
    dependency: Optional[str] = None,
    array_spec: Optional[str] = None,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --output={output_path}",
        f"#SBATCH --error={output_path}",
    ]

    if time_limit and time_limit != "0":
        lines.append(f"#SBATCH --time={time_limit}")
    if dependency:
        lines.append(f"#SBATCH --dependency={dependency}")
    if array_spec:
        lines.append(f"#SBATCH --array={array_spec}")
    if mail_user:
        lines.append(f"#SBATCH --mail-user={mail_user}")
        lines.append(f"#SBATCH --mail-type={mail_type}")

    lines += [
        "",
        "set -euo pipefail",
        "",
        command_body.strip(),
        "",
    ]
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o775)


def extract_job_id(sbatch_stdout: str) -> str:
    # Typical: "Submitted batch job 123456"
    m = re.search(r"Submitted batch job (\d+)", sbatch_stdout)
    if not m:
        raise RuntimeError(f"Could not parse job ID from sbatch output:\n{sbatch_stdout}")
    return m.group(1)


def submit_script(script_path: Path, dry_run: bool = False) -> str:
    if dry_run:
        return "DRYRUN"

    proc = subprocess.run(
        ["sbatch", str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sbatch failed for {script_path}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return extract_job_id(proc.stdout)


def run_looks_complete(out_dir: Path, runno: str) -> bool:
    final_mean = out_dir / f"{runno}_meanDCE_coregistered_allvols.nii.gz"
    t2_out = out_dir / f"{runno}_T2.nii.gz"
    return final_mean.exists() and t2_out.exists()


def build_command_body(env_setup: str, cmd_parts: List[str]) -> str:
    pieces: List[str] = []
    if env_setup.strip():
        pieces.append(env_setup.strip())
    pieces.append(shell_join(cmd_parts))
    return "\n".join(pieces)


def main() -> int:
    args = parse_args()
    validate_args(args)

    all_niis_dir = Path(args.all_niis_dir).resolve()
    pipeline = Path(args.pipeline).resolve()
    python_exe = str(Path(args.python).resolve())
    dce_suffixes = [s.strip() for s in args.dce_suffixes.split(",") if s.strip()]

    run_dirs = discover_run_dirs(all_niis_dir, args.run_glob)
    if not run_dirs:
        raise SystemExit(f"No run directories matched: {all_niis_dir / args.run_glob}")

    print("Resolved configuration:")
    print(f"  all_niis_dir      : {all_niis_dir}")
    print(f"  pipeline          : {pipeline}")
    print(f"  python            : {python_exe}")
    print(f"  run_glob          : {args.run_glob}")
    print(f"  partition         : {args.partition}")
    print(f"  mem               : {args.mem}")
    print(f"  cpus_per_task     : {args.cpus_per_task}")
    print(f"  time              : {args.time}")
    print(f"  affine_step       : {args.affine_step}")
    print(f"  conv_iters        : {args.conv_iters}")
    print(f"  conv_thresh       : {args.conv_thresh}")
    print(f"  conv_window       : {args.conv_window}")
    print(f"  shrink_factors    : {args.shrink_factors}")
    print(f"  smoothing_sigmas  : {args.smoothing_sigmas}")
    print(f"  winsorize_lower   : {args.winsorize_lower}")
    print(f"  winsorize_upper   : {args.winsorize_upper}")
    print(f"  hist_match        : {args.hist_match}")
    print(f"  dce_suffixes      : {','.join(dce_suffixes)}")
    print(f"  t2_suffix         : {args.t2_suffix}")
    print(f"  dry_run           : {args.dry_run}")
    print(f"  overwrite         : {args.overwrite}")
    print(f"  skip_complete_runs: {args.skip_complete_runs}")
    if args.env_setup:
        print(f"  env_setup         : {args.env_setup}")

    submitted_runs = 0
    skipped_runs = 0

    for run_dir in run_dirs:
        runno = parse_runno(run_dir)
        out_dir = all_niis_dir / f"{runno}_coregistered"
        sbatch_dir = out_dir / "sbatch"
        work_dir = out_dir / "work"
        sbatch_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        dce_files = discover_dce_files(run_dir, runno, dce_suffixes)
        t2_file = discover_t2_file(run_dir, runno, args.t2_suffix)

        if not dce_files:
            print(f"\nSKIP {run_dir.name}: no DCE files found")
            skipped_runs += 1
            continue

        if args.skip_complete_runs and run_looks_complete(out_dir, runno):
            print(f"\nSKIP {run_dir.name}: already looks complete")
            skipped_runs += 1
            continue

        print(f"\n=== SUBMIT {run_dir.name} (runno={runno}) ===")
        print(f"  DCE files:")
        for d in dce_files:
            print(f"    {d.name}")
        print(f"  T2: {t2_file.name if t2_file else 'not found'}")

        common_pipeline_args = build_pipeline_args(args, run_dir)

        # Stage 1: prep
        prep_script = sbatch_dir / f"{runno}_prep.bash"
        prep_job_name = f"DCEprep_{runno}"
        prep_output = sbatch_dir / "slurm-%j.out"
        prep_cmd = [python_exe, str(pipeline), "prep"] + common_pipeline_args
        prep_body = build_command_body(args.env_setup, prep_cmd)
        prep_text = make_job_script_text(
            job_name=prep_job_name,
            partition=args.partition,
            mem=args.mem,
            cpus_per_task=args.cpus_per_task,
            output_path=prep_output,
            command_body=prep_body,
            time_limit=args.time,
            mail_user=args.mail_user,
            mail_type=args.mail_type,
        )
        write_text(prep_script, prep_text)
        prep_jobid = submit_script(prep_script, dry_run=args.dry_run)
        print(f"  prep jobid: {prep_jobid}")

        final_array_jobids: List[str] = []

        for dce_nifti in dce_files:
            n_tasks = count_4d_volumes(python_exe, dce_nifti)
            array_spec = f"0-{n_tasks - 1}"
            dce_tag = dce_nifti.name.replace(".nii.gz", "")

            print(f"  {dce_nifti.name}: {n_tasks} volumes")

            # local_reg array
            local_reg_script = sbatch_dir / f"{dce_tag}_local_reg_array.bash"
            local_reg_job_name = f"LocReg_{runno}"
            local_reg_output = sbatch_dir / "slurm-%A_%a.out"
            local_reg_cmd = (
                [python_exe, str(pipeline), "local_reg"] +
                common_pipeline_args +
                ["--dce_nifti", str(dce_nifti)]
            )
            local_reg_body = build_command_body(args.env_setup, local_reg_cmd)
            local_reg_text = make_job_script_text(
                job_name=local_reg_job_name,
                partition=args.partition,
                mem=args.mem,
                cpus_per_task=args.cpus_per_task,
                output_path=local_reg_output,
                command_body=local_reg_body,
                time_limit=args.time,
                mail_user=args.mail_user,
                mail_type=args.mail_type,
                dependency=None if prep_jobid == "DRYRUN" else f"afterok:{prep_jobid}",
                array_spec=array_spec,
            )
            write_text(local_reg_script, local_reg_text)
            local_reg_jobid = submit_script(local_reg_script, dry_run=args.dry_run)
            print(f"    local_reg array jobid: {local_reg_jobid}")

            # local_mean single
            local_mean_script = sbatch_dir / f"{dce_tag}_local_mean.bash"
            local_mean_job_name = f"LocMean_{runno}"
            local_mean_output = sbatch_dir / "slurm-%j.out"
            local_mean_cmd = (
                [python_exe, str(pipeline), "local_mean"] +
                common_pipeline_args +
                ["--dce_nifti", str(dce_nifti)]
            )
            local_mean_body = build_command_body(args.env_setup, local_mean_cmd)
            local_mean_text = make_job_script_text(
                job_name=local_mean_job_name,
                partition=args.partition,
                mem=args.mem,
                cpus_per_task=args.cpus_per_task,
                output_path=local_mean_output,
                command_body=local_mean_body,
                time_limit=args.time,
                mail_user=args.mail_user,
                mail_type=args.mail_type,
                dependency=None if local_reg_jobid == "DRYRUN" else f"afterok:{local_reg_jobid}",
            )
            write_text(local_mean_script, local_mean_text)
            local_mean_jobid = submit_script(local_mean_script, dry_run=args.dry_run)
            print(f"    local_mean jobid: {local_mean_jobid}")

            # mean_to_global single
            mean_to_global_script = sbatch_dir / f"{dce_tag}_mean_to_global.bash"
            mean_to_global_job_name = f"Mean2Glob_{runno}"
            mean_to_global_output = sbatch_dir / "slurm-%j.out"
            mean_to_global_cmd = (
                [python_exe, str(pipeline), "mean_to_global"] +
                common_pipeline_args +
                ["--dce_nifti", str(dce_nifti)]
            )
            mean_to_global_body = build_command_body(args.env_setup, mean_to_global_cmd)
            mean_to_global_text = make_job_script_text(
                job_name=mean_to_global_job_name,
                partition=args.partition,
                mem=args.mem,
                cpus_per_task=args.cpus_per_task,
                output_path=mean_to_global_output,
                command_body=mean_to_global_body,
                time_limit=args.time,
                mail_user=args.mail_user,
                mail_type=args.mail_type,
                dependency=None if local_mean_jobid == "DRYRUN" else f"afterok:{local_mean_jobid}",
            )
            write_text(mean_to_global_script, mean_to_global_text)
            mean_to_global_jobid = submit_script(mean_to_global_script, dry_run=args.dry_run)
            print(f"    mean_to_global jobid: {mean_to_global_jobid}")

            # final_apply array
            final_apply_script = sbatch_dir / f"{dce_tag}_final_apply_array.bash"
            final_apply_job_name = f"FinalApply_{runno}"
            final_apply_output = sbatch_dir / "slurm-%A_%a.out"
            final_apply_cmd = (
                [python_exe, str(pipeline), "final_apply"] +
                common_pipeline_args +
                ["--dce_nifti", str(dce_nifti)]
            )
            final_apply_body = build_command_body(args.env_setup, final_apply_cmd)
            final_apply_text = make_job_script_text(
                job_name=final_apply_job_name,
                partition=args.partition,
                mem=args.mem,
                cpus_per_task=args.cpus_per_task,
                output_path=final_apply_output,
                command_body=final_apply_body,
                time_limit=args.time,
                mail_user=args.mail_user,
                mail_type=args.mail_type,
                dependency=None if mean_to_global_jobid == "DRYRUN" else f"afterok:{mean_to_global_jobid}",
                array_spec=array_spec,
            )
            write_text(final_apply_script, final_apply_text)
            final_apply_jobid = submit_script(final_apply_script, dry_run=args.dry_run)
            print(f"    final_apply array jobid: {final_apply_jobid}")

            final_array_jobids.append(final_apply_jobid)

        if not final_array_jobids:
            print(f"  No final_apply jobs submitted for {run_dir.name}; skipping finalize")
            skipped_runs += 1
            continue

        finalize_dep = None
        if final_array_jobids[0] != "DRYRUN":
            finalize_dep = "afterok:" + ":".join(final_array_jobids)

        finalize_script = sbatch_dir / f"{runno}_finalize.bash"
        finalize_job_name = f"DCEfinal_{runno}"
        finalize_output = sbatch_dir / "slurm-%j.out"
        finalize_cmd = [python_exe, str(pipeline), "finalize"] + common_pipeline_args
        finalize_body = build_command_body(args.env_setup, finalize_cmd)
        finalize_text = make_job_script_text(
            job_name=finalize_job_name,
            partition=args.partition,
            mem=args.mem,
            cpus_per_task=args.cpus_per_task,
            output_path=finalize_output,
            command_body=finalize_body,
            time_limit=args.time,
            mail_user=args.mail_user,
            mail_type=args.mail_type,
            dependency=finalize_dep,
        )
        write_text(finalize_script, finalize_text)
        finalize_jobid = submit_script(finalize_script, dry_run=args.dry_run)
        print(f"  finalize jobid: {finalize_jobid}")

        submitted_runs += 1

    print("\nSummary:")
    print(f"  Submitted runs: {submitted_runs}")
    print(f"  Skipped runs:   {skipped_runs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())