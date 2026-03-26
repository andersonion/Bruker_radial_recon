#!/usr/bin/env python3

import argparse
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="Submit hierarchical DCE coreg jobs")

    ap.add_argument(
        "--all-niis-dir",
        required=True,
        help="Path to all_niis directory containing z<runno>/ folders",
    )
    ap.add_argument(
        "--pipeline",
        default=None,
        help="Path to pipeline script (default: sibling of this launcher; prefers vanilla first)",
    )
    ap.add_argument("--run-glob", default="z*")
    ap.add_argument("--python", default=sys.executable)

    ap.add_argument("--partition", default="normal")
    ap.add_argument("--mem", default="12000M")
    ap.add_argument("--cpus-per-task", type=int, default=1)
    ap.add_argument("--time", default=None)

    ap.add_argument("--affine-step", type=float, default=0.05)
    ap.add_argument("--conv-iters", default="100x100x100x20")
    ap.add_argument("--conv-thresh", type=float, default=1e-7)
    ap.add_argument("--conv-window", type=int, default=15)
    ap.add_argument("--shrink-factors", default="8x4x2x1")
    ap.add_argument("--smoothing-sigmas", default="3x2x1x0vox")

    # Optional pass-through knobs for the updated T2 gradient-refinement pipeline
    ap.add_argument("--t2-grad-affine-step", type=float, default=0.02)
    ap.add_argument("--t2-grad-conv-iters", default="100x100x100x20")
    ap.add_argument("--t2-grad-conv-thresh", type=float, default=1e-7)
    ap.add_argument("--t2-grad-conv-window", type=int, default=15)
    ap.add_argument("--t2-grad-shrink-factors", default="8x4x2x1")
    ap.add_argument("--t2-grad-smoothing-sigmas", default="3x2x1x0vox")
    ap.add_argument("--t2-grad-smooth-passes", type=int, default=1)

    ap.add_argument("--skip-complete-runs", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    ap.add_argument(
        "--submit-retries",
        type=int,
        default=8,
        help="Number of sbatch retry attempts on temporary submission failures",
    )
    ap.add_argument(
        "--submit-sleep-seconds",
        type=float,
        default=10.0,
        help="Initial sleep before retrying sbatch after temporary failure",
    )
    ap.add_argument(
        "--inter-submit-delay",
        type=float,
        default=0.25,
        help="Delay after each successful sbatch so we do not hammer Slurm",
    )

    return ap.parse_args()


def resolve_pipeline(args):
    launcher_dir = Path(__file__).resolve().parent

    if args.pipeline is None:
        vanilla = launcher_dir / "dce_hierarchical_coreg.py"
        affine = launcher_dir / "dce_hierarchical_coreg_affine.py"

        if vanilla.is_file():
            pipeline = vanilla
        elif affine.is_file():
            pipeline = affine
        else:
            raise SystemExit(
                "Could not auto-find pipeline next to launcher.\n"
                f"Tried:\n  {vanilla}\n  {affine}"
            )
    else:
        pipeline = Path(args.pipeline).resolve()

    if not pipeline.is_file():
        raise SystemExit(f"Pipeline not found: {pipeline}")

    return pipeline


def run_cmd(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def sbatch_submit(script_path: Path, retries: int, sleep_seconds: float, inter_submit_delay: float, verbose: bool):
    last_stdout = ""
    last_stderr = ""

    for attempt in range(1, retries + 1):
        rc, out, err = run_cmd(["sbatch", str(script_path)])
        last_stdout = out
        last_stderr = err

        if rc == 0:
            m = re.search(r"Submitted batch job (\d+)", out)
            if not m:
                raise RuntimeError(f"Could not parse sbatch output:\n{out}")
            job_id = m.group(1)
            if inter_submit_delay > 0:
                time.sleep(inter_submit_delay)
            return job_id

        err_l = err.lower()
        temporary = (
            "temporarily unable to accept job" in err_l
            or "resource temporarily unavailable" in err_l
            or "socket timed out" in err_l
            or "slurm temporarily unable to accept job" in err_l
        )

        if not temporary or attempt == retries:
            raise RuntimeError(
                f"Command failed:\n"
                f"  sbatch {script_path}\n\n"
                f"STDOUT:\n{out}\n\n"
                f"STDERR:\n{err}"
            )

        delay = sleep_seconds * (2 ** (attempt - 1))
        if verbose:
            print(
                f"      sbatch temporary failure on attempt {attempt}/{retries}; "
                f"sleeping {delay:.1f}s then retrying"
            )
        time.sleep(delay)

    raise RuntimeError(
        f"sbatch failed after {retries} attempts for {script_path}\n\n"
        f"STDOUT:\n{last_stdout}\n\nSTDERR:\n{last_stderr}"
    )


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o775)


def get_nvols(python_exe: str, nii_path: Path) -> int:
    code = (
        "import nibabel as nib; "
        f"img=nib.load(r'''{str(nii_path)}'''); "
        "shape=img.shape; "
        "assert len(shape)==4, f'Expected 4D, got {shape}'; "
        "print(shape[3])"
    )
    rc, out, err = run_cmd([python_exe, "-c", code])
    if rc != 0:
        raise RuntimeError(
            f"Failed to read number of volumes from {nii_path}\n\nSTDOUT:\n{out}\n\nSTDERR:\n{err}"
        )
    return int(out.strip())


def parse_runno(run_dir: Path) -> str:
    bn = run_dir.name
    if bn.startswith("z") and len(bn) > 1:
        return bn[1:]
    return bn


def strict_dce_files(run_dir: Path, runno: str):
    out = []
    for suffix in ("DCE_baseline", "DCE_block1", "DCE_block2"):
        p = run_dir / f"{runno}_{suffix}.nii.gz"
        if p.is_file():
            out.append(p)
    return out


def shell_join(parts):
    return " ".join(shlex.quote(str(x)) for x in parts)


def build_job_script(
    *,
    job_name: str,
    partition: str,
    mem: str,
    cpus_per_task: int,
    stdout_path: Path,
    cmd: str,
    time_limit: str = None,
    dependency: str = None,
    array_spec: str = None,
):
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --output={stdout_path}",
        f"#SBATCH --error={stdout_path}",
    ]

    if time_limit:
        lines.append(f"#SBATCH --time={time_limit}")
    if dependency:
        lines.append(f"#SBATCH --dependency={dependency}")
    if array_spec:
        lines.append(f"#SBATCH --array={array_spec}")

    lines += [
        "",
        "set -euo pipefail",
        'echo "Running on node: $(hostname)"',
        "",
        cmd,
        "",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()

    all_niis_dir = Path(args.all_niis_dir).resolve()
    if not all_niis_dir.is_dir():
        raise SystemExit(f"Bad all_niis_dir: {all_niis_dir}")

    pipeline = resolve_pipeline(args)
    python_exe = str(Path(args.python).resolve())

    run_dirs = sorted([p for p in all_niis_dir.glob(args.run_glob) if p.is_dir()])
    if not run_dirs:
        raise SystemExit(f"No runs matched {all_niis_dir / args.run_glob}")

    common_args = [
        "--affine_step", str(args.affine_step),
        "--conv_iters", args.conv_iters,
        "--conv_thresh", str(args.conv_thresh),
        "--conv_window", str(args.conv_window),
        "--shrink_factors", args.shrink_factors,
        "--smoothing_sigmas", args.smoothing_sigmas,
        "--t2_grad_affine_step", str(args.t2_grad_affine_step),
        "--t2_grad_conv_iters", args.t2_grad_conv_iters,
        "--t2_grad_conv_thresh", str(args.t2_grad_conv_thresh),
        "--t2_grad_conv_window", str(args.t2_grad_conv_window),
        "--t2_grad_shrink_factors", args.t2_grad_shrink_factors,
        "--t2_grad_smoothing_sigmas", args.t2_grad_smoothing_sigmas,
        "--t2_grad_smooth_passes", str(args.t2_grad_smooth_passes),
    ]
    if args.verbose:
        common_args.append("--verbose")

    submitted = 0
    skipped = 0
    failed = 0

    for run_dir in run_dirs:
        runno = parse_runno(run_dir)
        out_dir = all_niis_dir / f"{runno}_coregistered"
        sbatch_dir = out_dir / "sbatch"
        work_dir = out_dir / "work"

        sbatch_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        final_mean = out_dir / f"{runno}_meanDCE_coregistered_allvols.nii.gz"
        t2_out = out_dir / f"{runno}_T2.nii.gz"

        if args.skip_complete_runs and final_mean.exists() and t2_out.exists():
            print(f"SKIP {runno}")
            skipped += 1
            continue

        dce_files = strict_dce_files(run_dir, runno)
        if not dce_files:
            print(f"SKIP {runno}: no strict DCE files found")
            skipped += 1
            continue

        print(f"\n=== {runno} ===")

        try:
            # prep
            prep_cmd = shell_join(
                [python_exe, str(pipeline), "prep", "--run_dir", str(run_dir)] + common_args
            )
            prep_script = sbatch_dir / f"{runno}_prep.bash"
            prep_log = sbatch_dir / "slurm-%j.out"
            write_text(
                prep_script,
                build_job_script(
                    job_name=f"DCEprep_{runno}",
                    partition=args.partition,
                    mem=args.mem,
                    cpus_per_task=args.cpus_per_task,
                    stdout_path=prep_log,
                    cmd=prep_cmd,
                    time_limit=args.time,
                ),
            )
            prep_id = (
                "DRY" if args.dry_run else
                sbatch_submit(
                    prep_script,
                    retries=args.submit_retries,
                    sleep_seconds=args.submit_sleep_seconds,
                    inter_submit_delay=args.inter_submit_delay,
                    verbose=args.verbose,
                )
            )
            print(f"prep: {prep_id}")

            final_apply_ids = []

            for dce in dce_files:
                nvols = get_nvols(python_exe, dce)
                tag = dce.name.replace(".nii.gz", "")
                print(f"  {dce.name}: {nvols} vols")

                # local_reg array
                local_reg_cmd = shell_join(
                    [python_exe, str(pipeline), "local_reg", "--run_dir", str(run_dir), "--dce_nifti", str(dce)] + common_args
                )
                local_reg_script = sbatch_dir / f"{tag}_local_reg_array.bash"
                local_reg_log = sbatch_dir / "slurm-%A_%a.out"
                write_text(
                    local_reg_script,
                    build_job_script(
                        job_name=f"LocReg_{runno}",
                        partition=args.partition,
                        mem=args.mem,
                        cpus_per_task=args.cpus_per_task,
                        stdout_path=local_reg_log,
                        cmd=local_reg_cmd,
                        time_limit=args.time,
                        dependency=None if prep_id == "DRY" else f"afterany:{prep_id}",
                        array_spec=f"0-{nvols - 1}",
                    ),
                )
                local_reg_id = (
                    "DRY" if args.dry_run else
                    sbatch_submit(
                        local_reg_script,
                        retries=args.submit_retries,
                        sleep_seconds=args.submit_sleep_seconds,
                        inter_submit_delay=args.inter_submit_delay,
                        verbose=args.verbose,
                    )
                )
                print(f"    local_reg: {local_reg_id}")

                # local_mean
                local_mean_cmd = shell_join(
                    [python_exe, str(pipeline), "local_mean", "--run_dir", str(run_dir), "--dce_nifti", str(dce)] + common_args
                )
                local_mean_script = sbatch_dir / f"{tag}_local_mean.bash"
                local_mean_log = sbatch_dir / "slurm-%j.out"
                write_text(
                    local_mean_script,
                    build_job_script(
                        job_name=f"LocMean_{runno}",
                        partition=args.partition,
                        mem=args.mem,
                        cpus_per_task=args.cpus_per_task,
                        stdout_path=local_mean_log,
                        cmd=local_mean_cmd,
                        time_limit=args.time,
                        dependency=None if local_reg_id == "DRY" else f"afterany:{local_reg_id}",
                    ),
                )
                local_mean_id = (
                    "DRY" if args.dry_run else
                    sbatch_submit(
                        local_mean_script,
                        retries=args.submit_retries,
                        sleep_seconds=args.submit_sleep_seconds,
                        inter_submit_delay=args.inter_submit_delay,
                        verbose=args.verbose,
                    )
                )
                print(f"    local_mean: {local_mean_id}")

                # mean_to_global
                mean_to_global_cmd = shell_join(
                    [python_exe, str(pipeline), "mean_to_global", "--run_dir", str(run_dir), "--dce_nifti", str(dce)] + common_args
                )
                mean_to_global_script = sbatch_dir / f"{tag}_mean_to_global.bash"
                mean_to_global_log = sbatch_dir / "slurm-%j.out"
                write_text(
                    mean_to_global_script,
                    build_job_script(
                        job_name=f"Mean2Glob_{runno}",
                        partition=args.partition,
                        mem=args.mem,
                        cpus_per_task=args.cpus_per_task,
                        stdout_path=mean_to_global_log,
                        cmd=mean_to_global_cmd,
                        time_limit=args.time,
                        dependency=None if local_mean_id == "DRY" else f"afterok:{local_mean_id}",
                    ),
                )
                mean_to_global_id = (
                    "DRY" if args.dry_run else
                    sbatch_submit(
                        mean_to_global_script,
                        retries=args.submit_retries,
                        sleep_seconds=args.submit_sleep_seconds,
                        inter_submit_delay=args.inter_submit_delay,
                        verbose=args.verbose,
                    )
                )
                print(f"    mean_to_global: {mean_to_global_id}")

                # final_apply array
                final_apply_cmd = shell_join(
                    [python_exe, str(pipeline), "final_apply", "--run_dir", str(run_dir), "--dce_nifti", str(dce)] + common_args
                )
                final_apply_script = sbatch_dir / f"{tag}_final_apply_array.bash"
                final_apply_log = sbatch_dir / "slurm-%A_%a.out"
                write_text(
                    final_apply_script,
                    build_job_script(
                        job_name=f"FinalApply_{runno}",
                        partition=args.partition,
                        mem=args.mem,
                        cpus_per_task=args.cpus_per_task,
                        stdout_path=final_apply_log,
                        cmd=final_apply_cmd,
                        time_limit=args.time,
                        dependency=None if mean_to_global_id == "DRY" else f"afterok:{mean_to_global_id}",
                        array_spec=f"0-{nvols - 1}",
                    ),
                )
                final_apply_id = (
                    "DRY" if args.dry_run else
                    sbatch_submit(
                        final_apply_script,
                        retries=args.submit_retries,
                        sleep_seconds=args.submit_sleep_seconds,
                        inter_submit_delay=args.inter_submit_delay,
                        verbose=args.verbose,
                    )
                )
                print(f"    final_apply: {final_apply_id}")
                final_apply_ids.append(final_apply_id)

            # finalize
            finalize_cmd = shell_join(
                [python_exe, str(pipeline), "finalize", "--run_dir", str(run_dir)] + common_args
            )
            finalize_script = sbatch_dir / f"{runno}_finalize.bash"
            finalize_log = sbatch_dir / "slurm-%j.out"
            finalize_dep = None if (args.dry_run or not final_apply_ids) else "afterany:" + ":".join(final_apply_ids)
            write_text(
                finalize_script,
                build_job_script(
                    job_name=f"DCEfinal_{runno}",
                    partition=args.partition,
                    mem=args.mem,
                    cpus_per_task=args.cpus_per_task,
                    stdout_path=finalize_log,
                    cmd=finalize_cmd,
                    time_limit=args.time,
                    dependency=finalize_dep,
                ),
            )
            finalize_id = (
                "DRY" if args.dry_run else
                sbatch_submit(
                    finalize_script,
                    retries=args.submit_retries,
                    sleep_seconds=args.submit_sleep_seconds,
                    inter_submit_delay=args.inter_submit_delay,
                    verbose=args.verbose,
                )
            )
            print(f"finalize: {finalize_id}")

            submitted += 1

        except Exception as exc:
            failed += 1
            print(f"FAILED {runno}: {exc}")

    print(f"\nSubmitted runs: {submitted}")
    print(f"Skipped runs:   {skipped}")
    print(f"Failed runs:    {failed}")


if __name__ == "__main__":
    main()