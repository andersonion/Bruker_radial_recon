#!/usr/bin/env python3
"""
Hierarchical DCE coregistration pipeline with STRICT T2 lookup:
    only ${runno}_T2.nii.gz is accepted as the T2 input.

This script assumes:
    all_niis/z<runno>/
        <runno>_DCE_baseline.nii.gz
        <runno>_DCE_block1.nii.gz
        <runno>_DCE_block2.nii.gz
        <runno>_T2.nii.gz

Outputs go to:
    all_niis/<runno>_coregistered/
        sbatch/
        work/
        <same-basename final DCE files>
        <same-basename final T2 file>
        <runno>_meanDCE_initial_allvols.nii.gz
        <runno>_meanDCE_coregistered_allvols.nii.gz
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


def run_cmd(cmd, verbose=False):
    if verbose:
        print("RUN:", " ".join(map(str, cmd)))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n  {' '.join(map(str, cmd))}\n\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def require_exe(exe_name):
    if shutil.which(exe_name) is None:
        raise SystemExit(f"Missing dependency in PATH: {exe_name}")


def parse_runno(run_dir: Path) -> str:
    bn = run_dir.name
    if bn.startswith("z") and len(bn) > 1:
        return bn[1:]
    return bn


def strip_nii_gz(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else Path(name).stem


def load_3d(path: Path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI: {path} but got {data.shape}")
    return img, data


def load_4d(path: Path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D NIfTI: {path} but got {data.shape}")
    return img, data


def save_3d_like(data3d, ref_img, out_path):
    out = nib.Nifti1Image(data3d.astype(np.float32), ref_img.affine, ref_img.header)
    out.header.set_data_dtype(np.float32)
    nib.save(out, str(out_path))


def save_4d_like(data4d, ref_img, out_path):
    hdr = ref_img.header.copy()
    out = nib.Nifti1Image(data4d.astype(np.float32), ref_img.affine, hdr)
    out.header.set_data_dtype(np.float32)
    nib.save(out, str(out_path))


def ants_affine_reg(
    fixed: Path,
    moving: Path,
    out_prefix: Path,
    affine_step: float,
    conv_iters: str,
    conv_thresh: float,
    conv_window: int,
    shrink_factors: str,
    smoothing_sigmas: str,
    verbose=False,
):
    warped = Path(str(out_prefix) + "Warped.nii.gz")
    affine = Path(str(out_prefix) + "0GenericAffine.mat")

    cmd = [
        "antsRegistration",
        "--float",
        "-d", "3",
        "-v", "1",
        "-m", f"Mattes[{fixed},{moving},1,32,regular,0.3]",
        "-t", f"Affine[{affine_step}]",
        "-c", f"[{conv_iters},{conv_thresh},{conv_window}]",
        "-s", smoothing_sigmas,
        "-f", shrink_factors,
        "-u", "1",
        "-z", "1",
        "-o", f"[{out_prefix},{warped}]",
    ]
    run_cmd(cmd, verbose=verbose)

    if not warped.exists():
        raise RuntimeError(f"Expected warped output not found: {warped}")
    if not affine.exists():
        raise RuntimeError(f"Expected affine transform not found: {affine}")

    return {"warped": warped, "affine": affine}


def ants_apply_transforms(fixed: Path, moving: Path, out_path: Path, transforms, verbose=False):
    cmd = [
        "antsApplyTransforms",
        "-d", "3",
        "-i", str(moving),
        "-r", str(fixed),
        "-o", str(out_path),
        "-n", "Linear",
    ]
    for t in transforms:
        cmd += ["-t", str(t)]
    run_cmd(cmd, verbose=verbose)

    if not out_path.exists():
        raise RuntimeError(f"antsApplyTransforms failed to write: {out_path}")


def compute_mean_from_single_4d(nifti_4d: Path, out_mean_3d: Path):
    img, data = load_4d(nifti_4d)
    mean3d = np.mean(data, axis=3, dtype=np.float64).astype(np.float32)
    save_3d_like(mean3d, img, out_mean_3d)
    return out_mean_3d


def compute_mean_across_all_3d_vols(dce_4d_paths, out_mean_3d: Path):
    imgs_data = []
    vol_counts = []

    for p in dce_4d_paths:
        img, data = load_4d(p)
        imgs_data.append((img, data))
        vol_counts.append(data.shape[3])

    if not imgs_data:
        raise ValueError("No DCE 4D paths provided.")

    base_shape = imgs_data[0][1].shape[:3]
    for p, (_, d) in zip(dce_4d_paths, imgs_data):
        if d.shape[:3] != base_shape:
            raise ValueError(f"Spatial mismatch: {p} has {d.shape[:3]} vs {base_shape}")

    total_vols = int(np.sum(vol_counts))
    if total_vols <= 0:
        raise ValueError("No volumes found to average.")

    acc = np.zeros(base_shape, dtype=np.float64)
    for _, d in imgs_data:
        for t in range(d.shape[3]):
            acc += d[..., t]

    mean3d = (acc / float(total_vols)).astype(np.float32)
    ref_img = imgs_data[0][0]
    save_3d_like(mean3d, ref_img, out_mean_3d)
    return out_mean_3d


def compute_mean_from_3d_stack(vol_paths, ref_img, out_mean_3d: Path):
    if not vol_paths:
        raise ValueError("No 3D paths provided for mean.")

    acc = None
    for p in vol_paths:
        _, dat = load_3d(p)
        if acc is None:
            acc = np.zeros(dat.shape, dtype=np.float64)
        acc += dat
    mean3d = (acc / float(len(vol_paths))).astype(np.float32)
    save_3d_like(mean3d, ref_img, out_mean_3d)
    return out_mean_3d


def strict_t2_input(run_dir: Path, runno: str) -> Path:
    """
    STRICT lookup:
        only ${runno}_T2.nii.gz is accepted.
    """
    p = run_dir / f"{runno}_T2.nii.gz"
    if not p.is_file():
        raise RuntimeError(
            f"Strict T2 not found:\n  expected {p}\n"
            "Everything else is ignored by design."
        )
    return p


def strict_dce_inputs(run_dir: Path, runno: str):
    out = []
    for suffix in ("DCE_baseline", "DCE_block1", "DCE_block2"):
        p = run_dir / f"{runno}_{suffix}.nii.gz"
        if p.is_file():
            out.append(p)
    return out


def get_paths(run_dir: Path):
    runno = parse_runno(run_dir)
    all_niis_dir = run_dir.parent
    out_dir = all_niis_dir / f"{runno}_coregistered"
    work_dir = out_dir / "work"
    sbatch_dir = out_dir / "sbatch"

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    sbatch_dir.mkdir(parents=True, exist_ok=True)

    dce_inputs = strict_dce_inputs(run_dir, runno)
    t2_input = strict_t2_input(run_dir, runno)

    return runno, out_dir, work_dir, sbatch_dir, dce_inputs, t2_input


def manifest_dir_for(work_dir: Path, dce_path: Path):
    return work_dir / "per_nifti" / dce_path.name


def local_mean_initial_path(out_dir: Path, dce_path: Path):
    return out_dir / f"{strip_nii_gz(dce_path.name)}_mean_initial.nii.gz"


def local_mean_coreg_path(out_dir: Path, dce_path: Path):
    return out_dir / f"{strip_nii_gz(dce_path.name)}_mean_localcoreg.nii.gz"


def local_mean_to_global_path(out_dir: Path, dce_path: Path):
    return out_dir / f"{strip_nii_gz(dce_path.name)}_mean_localcoreg_to_global.nii.gz"


def local_reg_prefix(work_dir: Path, dce_path: Path, t_idx: int):
    return manifest_dir_for(work_dir, dce_path) / "local_reg_xfm" / f"vol{t_idx:04d}_"


def local_warped_vol_path(work_dir: Path, dce_path: Path, t_idx: int):
    return manifest_dir_for(work_dir, dce_path) / "local_warped_vols" / f"vol{t_idx:04d}.nii.gz"


def final_apply_vol_path(work_dir: Path, dce_path: Path, t_idx: int):
    return manifest_dir_for(work_dir, dce_path) / "final_warped_vols" / f"vol{t_idx:04d}.nii.gz"


def mean_to_global_prefix(work_dir: Path, dce_path: Path):
    return manifest_dir_for(work_dir, dce_path) / "mean_to_global_xfm" / "mean_"


def cmd_prep(args):
    run_dir = Path(args.run_dir).resolve()
    runno, out_dir, work_dir, _, dce_inputs, _ = get_paths(run_dir)

    if not dce_inputs:
        raise SystemExit(f"No strict DCE inputs found in {run_dir}")

    print(f"=== PREP {run_dir.name} (runno={runno}) ===")

    for dce in dce_inputs:
        _, data = load_4d(dce)
        nT = data.shape[3]

        md = manifest_dir_for(work_dir, dce)
        md.mkdir(parents=True, exist_ok=True)
        (md / "local_reg_xfm").mkdir(exist_ok=True)
        (md / "local_warped_vols").mkdir(exist_ok=True)
        (md / "final_warped_vols").mkdir(exist_ok=True)
        (md / "mean_to_global_xfm").mkdir(exist_ok=True)

        (md / "n_tasks.txt").write_text(f"{nT}\n")

        mean_initial = local_mean_initial_path(out_dir, dce)
        compute_mean_from_single_4d(dce, mean_initial)
        print(f"Prepared {dce.name}: {nT} tasks")

    global_initial = out_dir / f"{runno}_meanDCE_initial_allvols.nii.gz"
    compute_mean_across_all_3d_vols(dce_inputs, global_initial)
    print(f"Global original mean: {global_initial}")
    return 0


def cmd_local_reg(args):
    require_exe("antsRegistration")

    run_dir = Path(args.run_dir).resolve()
    dce_path = Path(args.dce_nifti).resolve()
    _, out_dir, work_dir, _, _, _ = get_paths(run_dir)

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"]) if args.task_id is None else int(args.task_id)

    mean_initial = local_mean_initial_path(out_dir, dce_path)
    if not mean_initial.exists():
        raise SystemExit(f"Missing local initial mean: {mean_initial}")

    img, data = load_4d(dce_path)
    if task_id < 0 or task_id >= data.shape[3]:
        raise SystemExit(f"task_id {task_id} out of range for {dce_path.name}")

    vol = data[..., task_id].astype(np.float32)
    moving_path = manifest_dir_for(work_dir, dce_path) / f"orig_vol{task_id:04d}.nii.gz"
    nib.save(nib.Nifti1Image(vol, img.affine, img.header), str(moving_path))

    prefix = local_reg_prefix(work_dir, dce_path, task_id)
    out = ants_affine_reg(
        fixed=mean_initial,
        moving=moving_path,
        out_prefix=prefix,
        affine_step=args.affine_step,
        conv_iters=args.conv_iters,
        conv_thresh=args.conv_thresh,
        conv_window=args.conv_window,
        shrink_factors=args.shrink_factors,
        smoothing_sigmas=args.smoothing_sigmas,
        verbose=args.verbose,
    )

    warped_path = local_warped_vol_path(work_dir, dce_path, task_id)
    warped_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out["warped"], warped_path)

    print(f"LOCAL_REG {dce_path.name} vol={task_id} -> {warped_path}")
    return 0


def cmd_local_mean(args):
    run_dir = Path(args.run_dir).resolve()
    dce_path = Path(args.dce_nifti).resolve()
    _, out_dir, work_dir, _, _, _ = get_paths(run_dir)

    _, data = load_4d(dce_path)
    nT = data.shape[3]

    vol_paths = [local_warped_vol_path(work_dir, dce_path, t) for t in range(nT)]
    missing = [p for p in vol_paths if not p.exists()]
    if missing:
        raise SystemExit(f"Missing locally warped vols for {dce_path.name}; example: {missing[0]}")

    ref_img, _ = load_3d(vol_paths[0])
    out_mean = local_mean_coreg_path(out_dir, dce_path)
    compute_mean_from_3d_stack(vol_paths, ref_img, out_mean)

    print(f"LOCAL_MEAN {dce_path.name} -> {out_mean}")
    return 0


def cmd_mean_to_global(args):
    require_exe("antsRegistration")

    run_dir = Path(args.run_dir).resolve()
    dce_path = Path(args.dce_nifti).resolve()
    runno, out_dir, work_dir, _, _, _ = get_paths(run_dir)

    moving = local_mean_coreg_path(out_dir, dce_path)
    fixed = out_dir / f"{runno}_meanDCE_initial_allvols.nii.gz"

    if not moving.exists():
        raise SystemExit(f"Missing local coreg mean: {moving}")
    if not fixed.exists():
        raise SystemExit(f"Missing global initial mean: {fixed}")

    prefix = mean_to_global_prefix(work_dir, dce_path)
    out = ants_affine_reg(
        fixed=fixed,
        moving=moving,
        out_prefix=prefix,
        affine_step=args.affine_step,
        conv_iters=args.conv_iters,
        conv_thresh=args.conv_thresh,
        conv_window=args.conv_window,
        shrink_factors=args.shrink_factors,
        smoothing_sigmas=args.smoothing_sigmas,
        verbose=args.verbose,
    )

    warped_mean = local_mean_to_global_path(out_dir, dce_path)
    shutil.copy2(out["warped"], warped_mean)

    print(f"MEAN_TO_GLOBAL {dce_path.name} -> {warped_mean}")
    return 0


def cmd_final_apply(args):
    require_exe("antsApplyTransforms")

    run_dir = Path(args.run_dir).resolve()
    dce_path = Path(args.dce_nifti).resolve()
    runno, out_dir, work_dir, _, _, _ = get_paths(run_dir)

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"]) if args.task_id is None else int(args.task_id)

    dce_img, dce_data = load_4d(dce_path)
    if task_id < 0 or task_id >= dce_data.shape[3]:
        raise SystemExit(f"task_id {task_id} out of range for {dce_path.name}")

    fixed = out_dir / f"{runno}_meanDCE_initial_allvols.nii.gz"
    if not fixed.exists():
        raise SystemExit(f"Missing global fixed target: {fixed}")

    moving_orig = manifest_dir_for(work_dir, dce_path) / f"orig_vol{task_id:04d}.nii.gz"
    if not moving_orig.exists():
        vol = dce_data[..., task_id].astype(np.float32)
        nib.save(nib.Nifti1Image(vol, dce_img.affine, dce_img.header), str(moving_orig))

    local_aff = Path(str(local_reg_prefix(work_dir, dce_path, task_id)) + "0GenericAffine.mat")
    global_aff = Path(str(mean_to_global_prefix(work_dir, dce_path)) + "0GenericAffine.mat")

    if not local_aff.exists():
        raise SystemExit(f"Missing local affine: {local_aff}")
    if not global_aff.exists():
        raise SystemExit(f"Missing global affine: {global_aff}")

    out3d = final_apply_vol_path(work_dir, dce_path, task_id)
    out3d.parent.mkdir(parents=True, exist_ok=True)
    ants_apply_transforms(fixed, moving_orig, out3d, [global_aff, local_aff], verbose=args.verbose)

    print(f"FINAL_APPLY {dce_path.name} vol={task_id} -> {out3d}")
    return 0


def cmd_finalize(args):
    require_exe("antsRegistration")

    run_dir = Path(args.run_dir).resolve()
    runno, out_dir, work_dir, _, dce_inputs, t2_input = get_paths(run_dir)

    if not dce_inputs:
        raise SystemExit(f"No DCE inputs found in {run_dir}")

    global_fixed = out_dir / f"{runno}_meanDCE_initial_allvols.nii.gz"
    if not global_fixed.exists():
        raise SystemExit(f"Missing global original mean: {global_fixed}")

    fixed_img, fixed_dat = load_3d(global_fixed)

    final_4d_paths = []
    for dce in dce_inputs:
        _, dat4 = load_4d(dce)
        nT = dat4.shape[3]
        vol_paths = [final_apply_vol_path(work_dir, dce, t) for t in range(nT)]
        missing = [p for p in vol_paths if not p.exists()]
        if missing:
            raise SystemExit(f"Missing final warped vols for {dce.name}; example: {missing[0]}")

        reg_4d = np.zeros((fixed_dat.shape[0], fixed_dat.shape[1], fixed_dat.shape[2], nT), dtype=np.float32)
        for t, p in enumerate(vol_paths):
            _, vd = load_3d(p)
            if vd.shape != fixed_dat.shape:
                raise SystemExit(f"Shape mismatch for {p}: {vd.shape} vs {fixed_dat.shape}")
            reg_4d[..., t] = vd

        out4d = out_dir / dce.name
        save_4d_like(reg_4d, fixed_img, out4d)
        final_4d_paths.append(out4d)
        print(f"FINAL_4D {out4d}")

    final_mean = out_dir / f"{runno}_meanDCE_coregistered_allvols.nii.gz"
    compute_mean_across_all_3d_vols(final_4d_paths, final_mean)

    t2_out = out_dir / t2_input.name
    prefix = work_dir / "T2_to_finalMean_"
    out = ants_affine_reg(
        fixed=final_mean,
        moving=t2_input,
        out_prefix=prefix,
        affine_step=args.affine_step,
        conv_iters=args.conv_iters,
        conv_thresh=args.conv_thresh,
        conv_window=args.conv_window,
        shrink_factors=args.shrink_factors,
        smoothing_sigmas=args.smoothing_sigmas,
        verbose=args.verbose,
    )
    shutil.copy2(out["warped"], t2_out)
    print(f"T2_FINAL {t2_out}")
    print(f"FINAL_MEAN {final_mean}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Hierarchical DCE coregistration pipeline")

    def add_common(p):
        p.add_argument("--run_dir", required=True)
        p.add_argument("--affine_step", type=float, default=0.05)
        p.add_argument("--conv_iters", default="100x100x100x20")
        p.add_argument("--conv_thresh", type=float, default=1e-7)
        p.add_argument("--conv_window", type=int, default=15)
        p.add_argument("--shrink_factors", default="8x4x2x1")
        p.add_argument("--smoothing_sigmas", default="3x2x1x0vox")
        p.add_argument("--verbose", action="store_true")

    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep")
    add_common(p)

    p = sub.add_parser("local_reg")
    add_common(p)
    p.add_argument("--dce_nifti", required=True)
    p.add_argument("--task_id", default=None)

    p = sub.add_parser("local_mean")
    add_common(p)
    p.add_argument("--dce_nifti", required=True)

    p = sub.add_parser("mean_to_global")
    add_common(p)
    p.add_argument("--dce_nifti", required=True)

    p = sub.add_parser("final_apply")
    add_common(p)
    p.add_argument("--dce_nifti", required=True)
    p.add_argument("--task_id", default=None)

    p = sub.add_parser("finalize")
    add_common(p)

    args = ap.parse_args()

    if args.cmd == "prep":
        return cmd_prep(args)
    if args.cmd == "local_reg":
        return cmd_local_reg(args)
    if args.cmd == "local_mean":
        return cmd_local_mean(args)
    if args.cmd == "mean_to_global":
        return cmd_mean_to_global(args)
    if args.cmd == "final_apply":
        return cmd_final_apply(args)
    if args.cmd == "finalize":
        return cmd_finalize(args)

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())