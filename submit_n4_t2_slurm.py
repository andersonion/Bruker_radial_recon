#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
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
        text=True,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def load_helper_source(helper_path: Path):
    if not helper_path.exists():
        raise FileNotFoundError(f"Helper script not found: {helper_path}")
    text = helper_path.read_text()
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return text, sha


def write_job_bundle(
    bundle_dir: Path,
    helper_text: str,
    helper_name: str,
    metadata: dict,
):
    bundle_dir.mkdir(parents=True, exist_ok=True)

    helper_snapshot = bundle_dir / helper_name
    helper_snapshot.write_text(helper_text)
    helper_snapshot.chmod(0o755)

    metadata_path = bundle_dir / "job_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    return helper_snapshot, metadata_path


def build_job_script(
    runno: str,
    in_nii: Path,
    out_nii: Path,
    in_method: Path,
    out_method: Path,
    support_nii: Path,
    bias_nii: Path,
    brainmask_nii: Path,
    brain_t2_nii: Path,
    diff_nii: Path | None,
    tmp_otsu_pre_nii: Path,
    helper_snapshot: Path,
    metadata_path: Path,
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
    brain_seed_min_distance: float,
    brain_seed_score_pct: float,
    brain_seed_erode_iters: int,
    brain_center_weight: float,
    brain_grad_penalty: float,
    brain_candidate_intensity_min: float,
    brain_candidate_gradient_max: float,
    brain_max_grow_iters: int,
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
support_nii=""" + shell_quote(str(support_nii)) + r"""
bias_nii=""" + shell_quote(str(bias_nii)) + r"""
brainmask_nii=""" + shell_quote(str(brainmask_nii)) + r"""
brain_t2_nii=""" + shell_quote(str(brain_t2_nii)) + r"""
tmp_otsu_pre_nii=""" + shell_quote(str(tmp_otsu_pre_nii)) + r"""
helper_snapshot=""" + shell_quote(str(helper_snapshot)) + r"""
metadata_path=""" + shell_quote(str(metadata_path)) + r"""
python_exe=""" + shell_quote(str(python_path)) + r"""
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
echo "Metadata snapshot:"
cat "$metadata_path"

mkdir -p "$(dirname "$out_nii")"

if [[ ! -f "$in_nii" ]]; then
    echo "ERROR: Input not found: $in_nii" >&2
    exit 1
fi

if [[ ! -f "$helper_snapshot" ]]; then
    echo "ERROR: Helper snapshot not found: $helper_snapshot" >&2
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
elif [[ ! -f "$out_nii" || ! -f "$bias_nii" || ! -f "$support_nii" ]]; then
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
    make_support_mask "$in_nii" "$tmp_otsu_pre_nii" "$support_nii"

    echo
    echo "Running N4BiasFieldCorrection..."
    cmd=(
        "$n4_exe"
        -d """ + str(dimension) + r"""
        -i "$in_nii"
        -x "$support_nii"
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
    if [[ ! -f "$support_nii" ]]; then
        echo "ERROR: Missing support mask for brain mask: $support_nii" >&2
        exit 1
    fi

    echo
    echo "Running immutable helper snapshot:"
    ls -l "$helper_snapshot"

    brain_cmd=(
        "$python_exe" "$helper_snapshot"
        --bfc "$out_nii"
        --support_mask "$support_nii"
        --out_mask "$brainmask_nii"
        --out_masked_bfc "$brain_t2_nii"
        --smooth_sigma """ + str(brain_smooth_sigma) + r"""
        --grad_sigma """ + str(brain_grad_sigma) + r"""
        --seed_min_distance """ + str(brain_seed_min_distance) + r"""
        --seed_score_pct """ + str(brain_seed_score_pct) + r"""
        --seed_erode_iters """ + str(brain_seed_erode_iters) + r"""
        --center_weight """ + str(brain_center_weight) + r"""
        --grad_penalty """ + str(brain_grad_penalty) + r"""
        --candidate_intensity_min """ + str(brain_candidate_intensity_min) + r"""
        --candidate_gradient_max """ + str(brain_candidate_gradient_max) + r"""
        --max_grow_iters """ + str(brain_max_grow_iters) + r"""
        --close_iters """ + str(brain_close_iters) + r"""
        --dilate_iters """ + str(brain_dilate_iters) + r"""
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
    p = argparse.ArgumentParser(
        description="Submit one Slurm job per runno for N4 + brain masking, with per-job immutable helper snapshots."
    )
    p.add_argument("--base_dir", required=True)

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--runnos", nargs="+")
    group.add_argument("--runno_file")
    group.add_argument("--discover", action="store_true")

    p.add_argument("--sbatch_dir", default=None)
    p.add_argument(
        "--brainmask_helper",
        default=None,
        help="Path to the actual make_brain_mask_from_bfc.py you want snapshotted per job. Defaults to sibling file next to this submit script.",
    )

    p.add_argument("--python_path", default="python3")
    p.add_argument("--n4_path", default="N4BiasFieldCorrection")
    p.add_argument("--thresholdimage_path", default="ThresholdImage")
    p.add_argument("--imagemath_path", default="ImageMath")
    p.add_argument("--printheader_path", default="PrintHeader")

    p.add_argument("--dimension", type=int, default=3, choices=[2, 3, 4])

    # N4 defaults
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

    # Edge-aware helper args: these now actually match the helper
    p.add_argument("--brain_smooth_sigma", type=float, default=1.0)
    p.add_argument("--brain_grad_sigma", type=float, default=1.0)
    p.add_argument("--brain_seed_min_distance", type=float, default=4.0)
    p.add_argument("--brain_seed_score_pct", type=float, default=92.0)
    p.add_argument("--brain_seed_erode_iters", type=int, default=1)
    p.add_argument("--brain_center_weight", type=float, default=1.5)
    p.add_argument("--brain_grad_penalty", type=float, default=2.0)
    p.add_argument("--brain_candidate_intensity_min", type=float, default=0.08)
    p.add_argument("--brain_candidate_gradient_max", type=float, default=0.45)
    p.add_argument("--brain_max_grow_iters", type=int, default=200)
    p.add_argument("--brain_close_iters", type=int, default=2)
    p.add_argument("--brain_dilate_iters", type=int, default=1)

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

    script_dir = Path(__file__).resolve().parent
    helper_path = Path(args.brainmask_helper).expanduser().resolve() if args.brainmask_helper else (script_dir / "make_brain_mask_from_bfc.py")
    helper_text, helper_sha = load_helper_source(helper_path)

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
    jobs_dir = sbatch_dir / "jobs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] base_dir      : {base_dir}")
    print(f"[INFO] sbatch_dir    : {sbatch_dir}")
    print(f"[INFO] helper_path   : {helper_path}")
    print(f"[INFO] helper_sha    : {helper_sha}")
    print(f"[INFO] runno count   : {len(runnos)}")

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
        support_nii = run_dir / f"{runno}_bfc_support_mask.nii.gz"
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
            have_core = out_nii.exists() and bias_nii.exists() and support_nii.exists()
            have_brain = brainmask_nii.exists() and brain_t2_nii.exists()
            have_diff = True if diff_nii is None else diff_nii.exists()
            if have_core and have_brain and have_diff:
                print(f"[SKIP] All requested outputs already exist for runno {runno}")
                n_skipped += 1
                continue

        job_name = sanitize_job_name(f"n4_t2_{runno}")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        bundle_name = f"{job_name}.{stamp}.{helper_sha}"
        bundle_dir = jobs_dir / bundle_name

        metadata = {
            "job_name": job_name,
            "runno": runno,
            "generated_at": stamp,
            "helper_source_path": str(helper_path),
            "helper_sha256_short": helper_sha,
            "paths": {
                "input": str(in_nii),
                "output_bfc": str(out_nii),
                "output_bias": str(bias_nii),
                "output_support_mask": str(support_nii),
                "output_brain_mask": str(brainmask_nii),
                "output_brain_bfc": str(brain_t2_nii),
                "output_method": str(out_method),
            },
            "args": vars(args),
        }

        helper_snapshot, metadata_path = write_job_bundle(
            bundle_dir=bundle_dir,
            helper_text=helper_text,
            helper_name="make_brain_mask_from_bfc.py",
            metadata=metadata,
        )

        log_out = logs_dir / f"{bundle_name}.%j.out"
        log_err = logs_dir / f"{bundle_name}.%j.err"
        script_path = scripts_dir / f"{bundle_name}.sbatch"

        script_text = build_job_script(
            runno=runno,
            in_nii=in_nii,
            out_nii=out_nii,
            in_method=in_method,
            out_method=out_method,
            support_nii=support_nii,
            bias_nii=bias_nii,
            brainmask_nii=brainmask_nii,
            brain_t2_nii=brain_t2_nii,
            diff_nii=diff_nii,
            tmp_otsu_pre_nii=tmp_otsu_pre_nii,
            helper_snapshot=helper_snapshot,
            metadata_path=metadata_path,
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
            brain_seed_min_distance=args.brain_seed_min_distance,
            brain_seed_score_pct=args.brain_seed_score_pct,
            brain_seed_erode_iters=args.brain_seed_erode_iters,
            brain_center_weight=args.brain_center_weight,
            brain_grad_penalty=args.brain_grad_penalty,
            brain_candidate_intensity_min=args.brain_candidate_intensity_min,
            brain_candidate_gradient_max=args.brain_candidate_gradient_max,
            brain_max_grow_iters=args.brain_max_grow_iters,
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
        print(f"[BUNDLE]   {bundle_dir}")

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