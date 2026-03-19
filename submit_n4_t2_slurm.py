#!/usr/bin/env python3

import argparse
import json
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


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text()


def submit_job(script_path: Path):
    result = subprocess.run(
        ["sbatch", "--parsable", str(script_path)],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def parse_job_id(sbatch_stdout: str) -> str:
    m = re.match(r"^(\d+)", sbatch_stdout.strip())
    if not m:
        raise RuntimeError(f"Could not parse job id from sbatch output: {sbatch_stdout!r}")
    return m.group(1)


def build_job_script(
    *,
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
    sbatch_dir: Path,
    helper_source_text: str,
    metadata_json_text: str,
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
    brain_grad_thresholds: str,
    brain_outer_rim_mm: float,
    brain_shell_close_iters: int,
    brain_shell_dilate_iters: int,
    brain_shell_open_iters: int,
    brain_shell_min_volume_mm3: float,
    brain_cavity_volume_min_mm3: float,
    brain_cavity_volume_max_mm3: float,
    brain_extent_x_min_mm: float,
    brain_extent_x_max_mm: float,
    brain_extent_y_min_mm: float,
    brain_extent_y_max_mm: float,
    brain_extent_z_min_mm: float,
    brain_extent_z_max_mm: float,
    brain_cavity_bbox_fill_frac_min: float,
    brain_balloon_close_iters: int,
    brain_balloon_fill_holes: bool,
    brain_balloon_max_iters: int,
    brain_moat_thresholds: str,
    brain_shell_inner_band_mm: float,
    brain_moat_min_volume_mm3: float,
    brain_moat_close_iters: int,
    brain_barrier_close_iters: int,
    brain_close_iters: int,
    brain_open_iters: int,
    brain_dilate_iters: int,
    brain_volume_hard_min_mm3: float,
    brain_volume_hard_max_mm3: float,
    brain_volume_preferred_min_mm3: float,
    brain_volume_preferred_max_mm3: float,
    brain_shell_gate_grad_max: float,
    tight_mask: bool,
    tight_mask_erode_iters: int,
    save_brain_debug: bool,
    threads: int,
    job_name: str,
    log_path_pattern: Path,
    partition: str | None,
    time_str: str,
    mem_gb: int,
    cpus: int,
    overwrite: bool,
):
    sbatch_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_path_pattern}",
        f"#SBATCH --error={log_path_pattern}",
        f"#SBATCH --time={time_str}",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH --cpus-per-task={cpus}",
    ]
    if partition:
        sbatch_lines.append(f"#SBATCH --partition={partition}")

    lines = []
    lines.extend(sbatch_lines)
    lines.append("")
    lines.append("set -euo pipefail")
    lines.append("")
    lines.append('echo "===== JOB START ====="')
    lines.append("date")
    lines.append("hostname")
    lines.append('echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"')
    lines.append(f'echo "RUNNO={runno}"')
    lines.append("")
    lines.append(f"export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS={threads}")
    lines.append("")
    lines.append(f"SBATCH_DIR={shell_quote(str(sbatch_dir))}")
    lines.append('mkdir -p "$SBATCH_DIR"')
    lines.append("")
    lines.append('HELPER_SNAPSHOT="${SBATCH_DIR}/${SLURM_JOB_ID}_make_brain_mask_from_bfc.py"')
    lines.append('METADATA_JSON="${SBATCH_DIR}/${SLURM_JOB_ID}_job_metadata.json"')
    lines.append("")
    lines.append("cat > \"$HELPER_SNAPSHOT\" <<'PYTHON_HELPER_EOF'")
    lines.append(helper_source_text.rstrip("\n"))
    lines.append("PYTHON_HELPER_EOF")
    lines.append('chmod 755 "$HELPER_SNAPSHOT"')
    lines.append("")
    lines.append("cat > \"$METADATA_JSON\" <<'JSON_METADATA_EOF'")
    lines.append(metadata_json_text.rstrip("\n"))
    lines.append("JSON_METADATA_EOF")
    lines.append("")
    lines.append(f"in_nii={shell_quote(str(in_nii))}")
    lines.append(f"out_nii={shell_quote(str(out_nii))}")
    lines.append(f"in_method={shell_quote(str(in_method))}")
    lines.append(f"out_method={shell_quote(str(out_method))}")
    lines.append(f"support_nii={shell_quote(str(support_nii))}")
    lines.append(f"bias_nii={shell_quote(str(bias_nii))}")
    lines.append(f"brainmask_nii={shell_quote(str(brainmask_nii))}")
    lines.append(f"brain_t2_nii={shell_quote(str(brain_t2_nii))}")
    lines.append(f"tmp_otsu_pre_nii={shell_quote(str(tmp_otsu_pre_nii))}")
    lines.append(f"python_exe={shell_quote(str(python_path))}")
    lines.append(f"overwrite_flag={'1' if overwrite else '0'}")
    lines.append(f"tight_mask_flag={'1' if tight_mask else '0'}")
    lines.append(f"save_brain_debug_flag={'1' if save_brain_debug else '0'}")
    lines.append(f"brain_balloon_fill_holes_flag={'1' if brain_balloon_fill_holes else '0'}")
    lines.append(f"diff_nii={shell_quote(str(diff_nii)) if diff_nii is not None else shell_quote('')}")
    lines.append(f"n4_exe={shell_quote(str(n4_path))}")
    lines.append(f"threshold_exe={shell_quote(str(thresholdimage_path))}")
    lines.append(f"imagemath_exe={shell_quote(str(imagemath_path))}")
    lines.append(f"printheader_exe={shell_quote(str(printheader_path)) if printheader_path else shell_quote('')}")
    lines.append("")
    lines.append('echo ""')
    lines.append('echo "Metadata snapshot:"')
    lines.append('cat "$METADATA_JSON"')
    lines.append("")
    lines.append('mkdir -p "$(dirname "$out_nii")"')
    lines.append("")
    lines.append('if [[ ! -f "$in_nii" ]]; then')
    lines.append('    echo "ERROR: Input not found: $in_nii" >&2')
    lines.append("    exit 1")
    lines.append("fi")
    lines.append("")
    lines.append('if [[ ! -f "$HELPER_SNAPSHOT" ]]; then')
    lines.append('    echo "ERROR: Helper snapshot missing: $HELPER_SNAPSHOT" >&2')
    lines.append("    exit 1")
    lines.append("fi")
    lines.append("")
    lines.append('if [[ -n "$printheader_exe" ]]; then')
    lines.append('    echo ""')
    lines.append('    echo "Header / spacing check:"')
    lines.append('    "$printheader_exe" "$in_nii" 1 || true')
    lines.append("fi")
    lines.append("")
    lines.append("need_n4=0")
    lines.append('if [[ "$overwrite_flag" == "1" ]]; then')
    lines.append("    need_n4=1")
    lines.append('elif [[ ! -f "$out_nii" || ! -f "$bias_nii" || ! -f "$support_nii" ]]; then')
    lines.append("    need_n4=1")
    lines.append("fi")
    lines.append("")
    lines.append("need_brainmask=0")
    lines.append('if [[ "$overwrite_flag" == "1" ]]; then')
    lines.append("    need_brainmask=1")
    lines.append('elif [[ ! -f "$brainmask_nii" || ! -f "$brain_t2_nii" ]]; then')
    lines.append("    need_brainmask=1")
    lines.append("fi")
    lines.append("")
    lines.append('if [[ -n "$diff_nii" ]]; then')
    lines.append("    need_diff=0")
    lines.append('    if [[ "$overwrite_flag" == "1" ]]; then')
    lines.append("        need_diff=1")
    lines.append('    elif [[ ! -f "$diff_nii" ]]; then')
    lines.append("        need_diff=1")
    lines.append("    fi")
    lines.append("else")
    lines.append("    need_diff=0")
    lines.append("fi")
    lines.append("")
    lines.append("make_support_mask() {")
    lines.append('    local src_nii="$1"')
    lines.append('    local tmp_otsu="$2"')
    lines.append('    local dst_mask="$3"')
    lines.append("")
    lines.append('    rm -f "$tmp_otsu" "$dst_mask"')
    lines.append("")
    lines.append(f'    "$threshold_exe" {dimension} "$src_nii" "$tmp_otsu" Otsu 4')
    lines.append(
        f'    "$threshold_exe" {dimension} "$tmp_otsu" "$dst_mask" '
        f"{pre_otsu_keep_low} {pre_otsu_keep_high} 1 0"
    )
    lines.append(f'    "$imagemath_exe" {dimension} "$dst_mask" MC "$dst_mask" {pre_close_radius}')
    lines.append(f'    "$imagemath_exe" {dimension} "$dst_mask" MD "$dst_mask" {pre_dilate_radius_pre_glc}')
    lines.append(f'    "$imagemath_exe" {dimension} "$dst_mask" FillHoles "$dst_mask" {pre_fill_holes_radius}')
    lines.append(f'    "$imagemath_exe" {dimension} "$dst_mask" GetLargestComponent "$dst_mask"')
    lines.append(f'    "$imagemath_exe" {dimension} "$dst_mask" MD "$dst_mask" {pre_dilate_radius_final}')
    lines.append("}")
    lines.append("")
    lines.append('if [[ "$need_n4" == "1" ]]; then')
    lines.append('    echo ""')
    lines.append('    echo "Creating pre-N4 support mask..."')
    lines.append('    make_support_mask "$in_nii" "$tmp_otsu_pre_nii" "$support_nii"')
    lines.append("")
    lines.append('    echo ""')
    lines.append('    echo "Running N4BiasFieldCorrection..."')
    lines.append("    cmd=(")
    lines.append('        "$n4_exe"')
    lines.append(f"        -d {dimension}")
    lines.append('        -i "$in_nii"')
    lines.append('        -x "$support_nii"')
    lines.append(f"        -s {shrink_factor}")
    lines.append(f"        -c {shell_quote(convergence)}")
    lines.append(f"        -b {shell_quote(bspline)}")
    lines.append(f"        -t {shell_quote(histogram_sharpening)}")
    lines.append("        -r 1")
    lines.append('        -o "[""$out_nii"",""$bias_nii""]"')
    lines.append("    )")
    lines.append("    printf '  %q' \"${cmd[@]}\"")
    lines.append('    echo ""')
    lines.append('    "${cmd[@]}"')
    lines.append("")
    lines.append('    if [[ ! -f "$out_nii" || ! -f "$bias_nii" ]]; then')
    lines.append('        echo "ERROR: N4 outputs missing." >&2')
    lines.append("        exit 1")
    lines.append("    fi")
    lines.append("")
    lines.append('    if [[ -f "$in_method" ]]; then')
    lines.append('        cp -f "$in_method" "$out_method"')
    lines.append("    fi")
    lines.append("else")
    lines.append('    echo ""')
    lines.append('    echo "Skipping N4 stage; existing outputs found and overwrite is off."')
    lines.append("fi")
    lines.append("")
    lines.append('if [[ "$need_brainmask" == "1" ]]; then')
    lines.append('    if [[ ! -f "$out_nii" ]]; then')
    lines.append('        echo "ERROR: Missing corrected image for brain mask: $out_nii" >&2')
    lines.append("        exit 1")
    lines.append("    fi")
    lines.append('    if [[ ! -f "$support_nii" ]]; then')
    lines.append('        echo "ERROR: Missing support mask for brain mask: $support_nii" >&2')
    lines.append("        exit 1")
    lines.append("    fi")
    lines.append("")
    lines.append('    echo ""')
    lines.append('    echo "Running immutable helper snapshot:"')
    lines.append('    ls -l "$HELPER_SNAPSHOT"')
    lines.append("")
    lines.append("    brain_cmd=(")
    lines.append('        "$python_exe" "$HELPER_SNAPSHOT"')
    lines.append('        --bfc "$out_nii"')
    lines.append('        --support_mask "$support_nii"')
    lines.append('        --out_mask "$brainmask_nii"')
    lines.append('        --out_masked_bfc "$brain_t2_nii"')
    lines.append(f"        --smooth_sigma {brain_smooth_sigma}")
    lines.append(f"        --grad_sigma {brain_grad_sigma}")
    lines.append(f"        --grad_thresholds {shell_quote(brain_grad_thresholds)}")
    lines.append(f"        --outer_rim_mm {brain_outer_rim_mm}")
    lines.append(f"        --shell_close_iters {brain_shell_close_iters}")
    lines.append(f"        --shell_dilate_iters {brain_shell_dilate_iters}")
    lines.append(f"        --shell_open_iters {brain_shell_open_iters}")
    lines.append(f"        --shell_min_volume_mm3 {brain_shell_min_volume_mm3}")
    lines.append(f"        --cavity_volume_min_mm3 {brain_cavity_volume_min_mm3}")
    lines.append(f"        --cavity_volume_max_mm3 {brain_cavity_volume_max_mm3}")
    lines.append(f"        --extent_x_min_mm {brain_extent_x_min_mm}")
    lines.append(f"        --extent_x_max_mm {brain_extent_x_max_mm}")
    lines.append(f"        --extent_y_min_mm {brain_extent_y_min_mm}")
    lines.append(f"        --extent_y_max_mm {brain_extent_y_max_mm}")
    lines.append(f"        --extent_z_min_mm {brain_extent_z_min_mm}")
    lines.append(f"        --extent_z_max_mm {brain_extent_z_max_mm}")
    lines.append(f"        --cavity_bbox_fill_frac_min {brain_cavity_bbox_fill_frac_min}")
    lines.append(f"        --balloon_close_iters {brain_balloon_close_iters}")
    lines.append(f"        --balloon_max_iters {brain_balloon_max_iters}")
    lines.append(f"        --moat_thresholds {shell_quote(brain_moat_thresholds)}")
    lines.append(f"        --shell_inner_band_mm {brain_shell_inner_band_mm}")
    lines.append(f"        --moat_min_volume_mm3 {brain_moat_min_volume_mm3}")
    lines.append(f"        --moat_close_iters {brain_moat_close_iters}")
    lines.append(f"        --barrier_close_iters {brain_barrier_close_iters}")
    lines.append(f"        --brain_close_iters {brain_close_iters}")
    lines.append(f"        --brain_open_iters {brain_open_iters}")
    lines.append(f"        --brain_dilate_iters {brain_dilate_iters}")
    lines.append(f"        --brain_volume_hard_min_mm3 {brain_volume_hard_min_mm3}")
    lines.append(f"        --brain_volume_hard_max_mm3 {brain_volume_hard_max_mm3}")
    lines.append(f"        --brain_volume_preferred_min_mm3 {brain_volume_preferred_min_mm3}")
    lines.append(f"        --brain_volume_preferred_max_mm3 {brain_volume_preferred_max_mm3}")
    lines.append(f"        --shell_gate_grad_max {brain_shell_gate_grad_max}")
    lines.append(f"        --tight_erode_iters {tight_mask_erode_iters}")
    lines.append("    )")
    lines.append("")
    lines.append('    if [[ "$brain_balloon_fill_holes_flag" == "1" ]]; then')
    lines.append("        brain_cmd+=( --balloon_fill_holes )")
    lines.append("    fi")
    lines.append("")
    lines.append('    if [[ "$tight_mask_flag" == "1" ]]; then')
    lines.append("        brain_cmd+=( --tight_mask )")
    lines.append("    fi")
    lines.append("")
    lines.append('    if [[ "$save_brain_debug_flag" == "1" ]]; then')
    lines.append('        brain_cmd+=( --debug_prefix "${SBATCH_DIR}/${SLURM_JOB_ID}_brain_dbg" )')
    lines.append("    fi")
    lines.append("")
    lines.append("    printf '  %q' \"${brain_cmd[@]}\"")
    lines.append('    echo ""')
    lines.append('    "${brain_cmd[@]}"')
    lines.append("")
    lines.append('    if [[ ! -f "$brainmask_nii" || ! -f "$brain_t2_nii" ]]; then')
    lines.append('        echo "ERROR: Brain-mask outputs missing." >&2')
    lines.append("        exit 1")
    lines.append("    fi")
    lines.append("else")
    lines.append('    echo ""')
    lines.append('    echo "Skipping brain-mask stage; outputs exist and overwrite is off."')
    lines.append("fi")
    lines.append("")
    lines.append('if [[ "$need_diff" == "1" ]]; then')
    lines.append(f'    "$imagemath_exe" {dimension} "$diff_nii" - "$out_nii" "$in_nii"')
    lines.append("fi")
    lines.append("")
    lines.append('rm -f "$tmp_otsu_pre_nii"')
    lines.append("")
    lines.append('echo "===== JOB END ====="')
    lines.append("date")

    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(
        description="Submit one Slurm job per runno for N4 + brain masking, with per-job immutable helper snapshots in run-local sbatch folders."
    )
    p.add_argument("--base_dir", required=True)

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--runnos", nargs="+")
    group.add_argument("--runno_file")
    group.add_argument("--discover", action="store_true")

    p.add_argument(
        "--brainmask_helper",
        default=None,
        help="Path to the real make_brain_mask_from_bfc.py. Defaults to sibling file next to this submit script.",
    )

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

    p.add_argument("--brain_smooth_sigma", type=float, default=1.0)
    p.add_argument("--brain_grad_sigma", type=float, default=1.0)

    p.add_argument("--brain_grad_thresholds", default="0.10,0.12,0.14,0.16,0.18,0.20")
    p.add_argument("--brain_outer_rim_mm", type=float, default=1.25)
    p.add_argument("--brain_shell_close_iters", type=int, default=2)
    p.add_argument("--brain_shell_dilate_iters", type=int, default=1)
    p.add_argument("--brain_shell_open_iters", type=int, default=0)
    p.add_argument("--brain_shell_min_volume_mm3", type=float, default=2.0)

    p.add_argument("--brain_cavity_volume_min_mm3", type=float, default=250.0)
    p.add_argument("--brain_cavity_volume_max_mm3", type=float, default=800.0)
    p.add_argument("--brain_extent_x_min_mm", type=float, default=7.0)
    p.add_argument("--brain_extent_x_max_mm", type=float, default=22.0)
    p.add_argument("--brain_extent_y_min_mm", type=float, default=7.0)
    p.add_argument("--brain_extent_y_max_mm", type=float, default=22.0)
    p.add_argument("--brain_extent_z_min_mm", type=float, default=4.0)
    p.add_argument("--brain_extent_z_max_mm", type=float, default=30.0)
    p.add_argument("--brain_cavity_bbox_fill_frac_min", type=float, default=0.35)

    p.add_argument("--brain_balloon_close_iters", type=int, default=2)
    p.add_argument("--brain_balloon_fill_holes", action="store_true")
    p.add_argument("--brain_balloon_max_iters", type=int, default=10000)

    p.add_argument("--brain_moat_thresholds", default="0.04,0.06,0.08,0.10")
    p.add_argument("--brain_shell_inner_band_mm", type=float, default=1.25)
    p.add_argument("--brain_moat_min_volume_mm3", type=float, default=1.0)
    p.add_argument("--brain_moat_close_iters", type=int, default=1)
    p.add_argument("--brain_barrier_close_iters", type=int, default=1)

    p.add_argument("--brain_close_iters", type=int, default=2)
    p.add_argument("--brain_open_iters", type=int, default=0)
    p.add_argument("--brain_dilate_iters", type=int, default=0)

    p.add_argument("--brain_volume_hard_min_mm3", type=float, default=300.0)
    p.add_argument("--brain_volume_hard_max_mm3", type=float, default=650.0)
    p.add_argument("--brain_volume_preferred_min_mm3", type=float, default=380.0)
    p.add_argument("--brain_volume_preferred_max_mm3", type=float, default=550.0)

    p.add_argument("--brain_shell_gate_grad_max", type=float, default=0.35)

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
    helper_path = (
        Path(args.brainmask_helper).expanduser().resolve()
        if args.brainmask_helper
        else (script_dir / "make_brain_mask_from_bfc.py")
    )
    helper_source_text = read_text(helper_path)

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

    print(f"[INFO] base_dir      : {base_dir}")
    print(f"[INFO] helper_path   : {helper_path}")
    print(f"[INFO] runno count   : {len(runnos)}")

    n_prepared = 0
    n_submitted = 0
    n_skipped = 0
    n_missing = 0
    n_failed_submit = 0

    for runno in runnos:
        run_dir = base_dir / f"z{runno}"
        sbatch_dir = run_dir / "sbatch"
        sbatch_dir.mkdir(parents=True, exist_ok=True)

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
        tmp_script = sbatch_dir / f"TMP_{job_name}.sbatch"
        log_path_pattern = sbatch_dir / "slurm-%j.out"

        metadata = {
            "runno": runno,
            "job_name": job_name,
            "input": str(in_nii),
            "output_bfc": str(out_nii),
            "output_bias": str(bias_nii),
            "output_support_mask": str(support_nii),
            "output_brain_mask": str(brainmask_nii),
            "output_brain_bfc": str(brain_t2_nii),
            "helper_source_path": str(helper_path),
            "submit_args": vars(args),
        }
        metadata_json_text = json.dumps(metadata, indent=2, sort_keys=True)

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
            sbatch_dir=sbatch_dir,
            helper_source_text=helper_source_text,
            metadata_json_text=metadata_json_text,
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
            brain_grad_thresholds=args.brain_grad_thresholds,
            brain_outer_rim_mm=args.brain_outer_rim_mm,
            brain_shell_close_iters=args.brain_shell_close_iters,
            brain_shell_dilate_iters=args.brain_shell_dilate_iters,
            brain_shell_open_iters=args.brain_shell_open_iters,
            brain_shell_min_volume_mm3=args.brain_shell_min_volume_mm3,
            brain_cavity_volume_min_mm3=args.brain_cavity_volume_min_mm3,
            brain_cavity_volume_max_mm3=args.brain_cavity_volume_max_mm3,
            brain_extent_x_min_mm=args.brain_extent_x_min_mm,
            brain_extent_x_max_mm=args.brain_extent_x_max_mm,
            brain_extent_y_min_mm=args.brain_extent_y_min_mm,
            brain_extent_y_max_mm=args.brain_extent_y_max_mm,
            brain_extent_z_min_mm=args.brain_extent_z_min_mm,
            brain_extent_z_max_mm=args.brain_extent_z_max_mm,
            brain_cavity_bbox_fill_frac_min=args.brain_cavity_bbox_fill_frac_min,
            brain_balloon_close_iters=args.brain_balloon_close_iters,
            brain_balloon_fill_holes=args.brain_balloon_fill_holes,
            brain_balloon_max_iters=args.brain_balloon_max_iters,
            brain_moat_thresholds=args.brain_moat_thresholds,
            brain_shell_inner_band_mm=args.brain_shell_inner_band_mm,
            brain_moat_min_volume_mm3=args.brain_moat_min_volume_mm3,
            brain_moat_close_iters=args.brain_moat_close_iters,
            brain_barrier_close_iters=args.brain_barrier_close_iters,
            brain_close_iters=args.brain_close_iters,
            brain_open_iters=args.brain_open_iters,
            brain_dilate_iters=args.brain_dilate_iters,
            brain_volume_hard_min_mm3=args.brain_volume_hard_min_mm3,
            brain_volume_hard_max_mm3=args.brain_volume_hard_max_mm3,
            brain_volume_preferred_min_mm3=args.brain_volume_preferred_min_mm3,
            brain_volume_preferred_max_mm3=args.brain_volume_preferred_max_mm3,
            brain_shell_gate_grad_max=args.brain_shell_gate_grad_max,
            tight_mask=args.tight_mask,
            tight_mask_erode_iters=args.tight_mask_erode_iters,
            save_brain_debug=args.save_brain_debug,
            threads=args.threads,
            job_name=job_name,
            log_path_pattern=log_path_pattern,
            partition=args.partition,
            time_str=args.time,
            mem_gb=args.mem_gb,
            cpus=args.cpus,
            overwrite=args.overwrite,
        )

        tmp_script.write_text(script_text)
        n_prepared += 1
        print(f"[PREPARED] {tmp_script}")

        if args.dry_run:
            continue

        rc, stdout, stderr = submit_job(tmp_script)
        if rc != 0:
            eprint(f"[SUBMIT FAIL] runno {runno}")
            if stdout:
                eprint(stdout)
            if stderr:
                eprint(stderr)
            n_failed_submit += 1
            continue

        job_id = parse_job_id(stdout)
        final_script = sbatch_dir / f"{job_id}_{job_name}.sbatch"
        tmp_script.rename(final_script)

        print(f"[SUBMITTED] runno {runno}: job_id={job_id}")
        print(f"[SCRIPT]    {final_script}")
        print(f"[LOG]       {sbatch_dir / f'slurm-{job_id}.out'}")
        n_submitted += 1

    print("\n[SUMMARY]")
    print(f"  Prepared     : {n_prepared}")
    print(f"  Submitted    : {n_submitted}")
    print(f"  Skipped      : {n_skipped}")
    print(f"  Missing input: {n_missing}")
    print(f"  Submit failed: {n_failed_submit}")

    return 0 if n_failed_submit == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())