#!/usr/bin/env python3

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description="Submit hierarchical DCE coreg jobs")

    ap.add_argument("--all-niis-dir", required=True)

    # Optional now
    ap.add_argument("--pipeline", default=None)

    ap.add_argument("--run-glob", default="z*")
    ap.add_argument("--python", default=sys.executable)

    ap.add_argument("--partition", default="normal")
    ap.add_argument("--mem", default="12000M")
    ap.add_argument("--cpus-per-task", type=int, default=1)

    ap.add_argument("--affine-step", type=float, default=0.05)
    ap.add_argument("--conv-iters", default="100x100x100x20")
    ap.add_argument("--conv-thresh", type=float, default=1e-7)
    ap.add_argument("--conv-window", type=int, default=15)
    ap.add_argument("--shrink-factors", default="8x4x2x1")
    ap.add_argument("--smoothing-sigmas", default="3x2x1x0vox")

    ap.add_argument("--skip-complete-runs", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    return ap.parse_args()


def resolve_pipeline(args):
    launcher_dir = Path(__file__).resolve().parent

    if args.pipeline is None:
        pipeline = launcher_dir / "dce_hierarchical_coreg.py"
    else:
        pipeline = Path(args.pipeline).resolve()

    if not pipeline.is_file():
        raise SystemExit(f"Pipeline not found: {pipeline}")

    return pipeline


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)
    return p.stdout


def sbatch(script):
    out = run(["sbatch", script])
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise RuntimeError(out)
    return m.group(1)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o775)


def get_nvols(python, nii):
    code = f"import nibabel as nib;print(nib.load(r'{nii}').shape[3])"
    return int(run([python, "-c", code]).strip())


def main():
    args = parse_args()

    all_niis = Path(args.all_niis_dir).resolve()
    if not all_niis.is_dir():
        raise SystemExit("Bad all_niis_dir")

    pipeline = resolve_pipeline(args)
    python = str(Path(args.python).resolve())

    runs = sorted([p for p in all_niis.glob(args.run_glob) if p.is_dir()])

    for run_dir in runs:
        runno = run_dir.name[1:] if run_dir.name.startswith("z") else run_dir.name

        out_dir = all_niis / f"{runno}_coregistered"
        sbatch_dir = out_dir / "sbatch"
        work_dir = out_dir / "work"

        sbatch_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        final_mean = out_dir / f"{runno}_meanDCE_coregistered_allvols.nii.gz"
        t2_out = out_dir / f"{runno}_T2.nii.gz"

        if args.skip_complete_runs and final_mean.exists() and t2_out.exists():
            print(f"SKIP {runno}")
            continue

        print(f"\n=== {runno} ===")

        dce_files = list(run_dir.glob("*_DCE_*.nii.gz"))
        if not dce_files:
            print("No DCE files")
            continue

        common = f"--run_dir {run_dir} --affine_step {args.affine_step} " \
                 f"--conv_iters {args.conv_iters} --conv_thresh {args.conv_thresh} " \
                 f"--conv_window {args.conv_window} --shrink_factors {args.shrink_factors} " \
                 f"--smoothing_sigmas {args.smoothing_sigmas}"

        # PREP
        prep_script = sbatch_dir / "prep.sh"
        write(prep_script, f"""#!/bin/bash
#SBATCH --mem={args.mem}
#SBATCH --partition={args.partition}
python3 {pipeline} prep {common}
""")

        prep_id = "DRY" if args.dry_run else sbatch(prep_script)
        print("prep:", prep_id)

        final_ids = []

        for dce in dce_files:
            n = get_nvols(python, dce)
            print(f"{dce.name}: {n} vols")

            tag = dce.stem

            # LOCAL REG ARRAY
            s = sbatch_dir / f"{tag}_local.sh"
            write(s, f"""#!/bin/bash
#SBATCH --mem={args.mem}
#SBATCH --partition={args.partition}
#SBATCH --array=0-{n-1}
#SBATCH --dependency=afterok:{prep_id}
python3 {pipeline} local_reg {common} --dce_nifti {dce}
""")
            jid1 = "DRY" if args.dry_run else sbatch(s)

            # LOCAL MEAN
            s = sbatch_dir / f"{tag}_mean.sh"
            write(s, f"""#!/bin/bash
#SBATCH --mem={args.mem}
#SBATCH --dependency=afterok:{jid1}
python3 {pipeline} local_mean {common} --dce_nifti {dce}
""")
            jid2 = "DRY" if args.dry_run else sbatch(s)

            # GLOBAL REG
            s = sbatch_dir / f"{tag}_glob.sh"
            write(s, f"""#!/bin/bash
#SBATCH --mem={args.mem}
#SBATCH --dependency=afterok:{jid2}
python3 {pipeline} mean_to_global {common} --dce_nifti {dce}
""")
            jid3 = "DRY" if args.dry_run else sbatch(s)

            # FINAL APPLY ARRAY
            s = sbatch_dir / f"{tag}_final.sh"
            write(s, f"""#!/bin/bash
#SBATCH --mem={args.mem}
#SBATCH --array=0-{n-1}
#SBATCH --dependency=afterok:{jid3}
python3 {pipeline} final_apply {common} --dce_nifti {dce}
""")
            jid4 = "DRY" if args.dry_run else sbatch(s)

            final_ids.append(jid4)

        dep = ":".join(final_ids)

        # FINALIZE
        s = sbatch_dir / "final.sh"
        write(s, f"""#!/bin/bash
#SBATCH --mem={args.mem}
#SBATCH --dependency=afterok:{dep}
python3 {pipeline} finalize {common}
""")
        jid = "DRY" if args.dry_run else sbatch(s)

        print("final:", jid)


if __name__ == "__main__":
    main()