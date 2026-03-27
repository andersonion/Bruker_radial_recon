#!/usr/bin/env python3
"""
Hierarchical DCE coregistration pipeline with robust resume behavior.

Features
- strict T2 lookup: only ${runno}_T2.nii.gz
- strict DCE lookup:
    ${runno}_DCE_baseline.nii.gz
    ${runno}_DCE_block1.nii.gz
    ${runno}_DCE_block2.nii.gz
- optional first-round local target:
    * mean of 4D  (default)
    * first volume of 4D
- optional light smoothing before registration (registration inputs only)
- T2 registration split into rigid and affine stages, then gradient rigid+affine refinement
- fixed work-directory naming: no '.nii.gz' directory names
- skip-if-done on every stage/task
- hard fail on missing warped volumes when building time-series products

Outputs
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


def smooth1d_axis(arr, kernel, axis):
    pad = len(kernel) // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(arr, pad_width, mode="edge")

    def conv_func(v):
        return np.convolve(v, kernel, mode="valid")

    return np.apply_along_axis(conv_func, axis, padded)


def smooth3d_binomial(data, passes=1):
    if passes <= 0:
        return data.astype(np.float32)

    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    kernel /= np.sum(kernel)

    out = data.astype(np.float32, copy=True)
    for _ in range(passes):
        out = smooth1d_axis(out, kernel, axis=0)
        out = smooth1d_axis(out, kernel, axis=1)
        out = smooth1d_axis(out, kernel, axis=2)
    return out.astype(np.float32)


def gradient_magnitude_3d(data, zooms=None):
    if zooms is None:
        gx, gy, gz = np.gradient(data.astype(np.float32))
    else:
        gx, gy, gz = np.gradient(
            data.astype(np.float32),
            float(zooms[0]),
            float(zooms[1]),
            float(zooms[2]),
        )
    gm = np.sqrt(gx * gx + gy * gy + gz * gz, dtype=np.float32)
    return gm.astype(np.float32)


def write_gradient_mag_image(in_img: nib.Nifti1Image, in_data: np.ndarray, out_path: Path, smooth_passes: int):
    sm = smooth3d_binomial(in_data, passes=smooth_passes)
    zooms = in_img.header.get_zooms()[:3]
    gm = gradient_magnitude_3d(sm, zooms=zooms)
    save_3d_like(gm, in_img, out_path)
    return out_path


def maybe_write_smoothed_image(in_img: nib.Nifti1Image, in_data: np.ndarray, out_path: Path, smooth_passes: int):
    sm = smooth3d_binomial(in_data, passes=smooth_passes)
    save_3d_like(sm, in_img, out_path)
    return out_path


def ants_linear_reg(
    *,
    fixed: Path,
    moving: Path,
    out_prefix: Path,
    transform_type: str,
    step_size: float,
    conv_iters: str,
    conv_thresh: float,
    conv_window: int,
    shrink_factors: str,
    smoothing_sigmas: str,
    initial_transform: Path = None,
    verbose=False,
):
    if transform_type not in ("Rigid", "Affine"):
        raise ValueError(f"Unsupported transform_type: {transform_type}")

    warped = Path(str(out_prefix) + "Warped.nii.gz")
    affine = Path(str(out_prefix) + "0GenericAffine.mat")

    cmd = ["antsRegistration", "--float", "-d", "3", "-v", "1"]
    if initial_transform is not None:
        cmd += ["-r", str(initial_transform)]

    cmd += [
        "-m", f"Mattes[{fixed},{moving},1,32,regular,0.3]",
        "-t", f"{transform_type}[{step_size}]",
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


def extract_first_vol_from_4d(nifti_4d: Path, out_3d: Path):
    img, data = load_4d(nifti_4d)
    first = data[..., 0].astype(np.float32)
    save_3d_like(first, img, out_3d)
    return out_3d


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
    return work_dir / "per_nifti" / strip_nii_gz(dce_path.name)


def local_round1_target_path(out_dir: Path, dce_path: Path, target_mode: str):
    stem = strip_nii_gz(dce_path.name)
    if target_mode == "mean":
        return out_dir / f"{stem}_round1_target_mean.nii.gz"
    if target_mode == "firstvol":
        return out_dir / f"{stem}_round1_target_firstvol.nii.gz"
    raise ValueError(f"Unknown target_mode: {target_mode}")


def local_mean_coreg_path(out_dir: Path, dce_path: Path):
    return out_dir / f"{strip_nii_gz(dce_path.name)}_mean_localcoreg.nii.gz"


def local_mean_to_global_path(out_dir: Path, dce_path: Path):
    return out_dir / f"{strip_nii_gz(dce_path.name)}_mean_localcoreg_to_global.nii.gz"


def local_reg_prefix(work_dir: Path, dce_path: Path, t_idx: int):
    return manifest_dir_for(work_dir, dce_path) / "local_reg_xfm" / f"vol{t_idx:04d}_"


def local_affine_path(work_dir: Path, dce_path: Path, t_idx: int):
    return Path(str(local_reg_prefix(work_dir, dce_path, t_idx)) + "0GenericAffine.mat")


def local_warped_vol_path(work_dir: Path, dce_path: Path, t_idx: int):
    return manifest_dir_for(work_dir, dce_path) / "local_warped_vols" / f"vol{t_idx:04d}.nii.gz"


def final_apply_vol_path(work_dir: Path, dce_path: Path, t_idx: int):
    return manifest_dir_for(work_dir, dce_path) / "final_warped_vols" / f"vol{t_idx:04d}.nii.gz"


def mean_to_global_prefix(work_dir: Path, dce_path: Path):
    return manifest_dir_for(work_dir, dce_path) / "mean_to_global_xfm" / "mean_"


def mean_to_global_affine_path(work_dir: Path, dce_path: Path):
    return Path(str(mean_to_global_prefix(work_dir, dce_path)) + "0GenericAffine.mat")


def get_local_round1_fixed_for_registration(out_dir: Path, work_dir: Path, dce_path: Path, target_mode: str, reg_smooth_passes: int):
    base_target = local_round1_target_path(out_dir, dce_path, target_mode)
    fixed_img, fixed_dat = load_3d(base_target)

    if reg_smooth_passes > 0:
        reg_fixed = manifest_dir_for(work_dir, dce_path) / "round1_fixed_smoothed.nii.gz"
        if not reg_fixed.exists():
            reg_fixed.parent.mkdir(parents=True, exist_ok=True)
            maybe_write_smoothed_image(fixed_img, fixed_dat, reg_fixed, reg_smooth_passes)
        return reg_fixed

    return base_target


def cmd_prep(args):
    run_dir = Path(args.run_dir).resolve()
    runno, out_dir, work_dir, _, dce_inputs, _ = get_paths(run_dir)

    if not dce_inputs:
        raise SystemExit(f"No strict DCE inputs found in {run_dir}")

    print(f"=== PREP {run_dir.name} (runno={runno}) ===")

    global_initial = out_dir / f"{runno}_meanDCE_initial_allvols.nii.gz"

    all_local_targets_exist = True
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

        target_path = local_round1_target_path(out_dir, dce, args.first_round_target)
        if not target_path.exists():
            all_local_targets_exist = False
            if args.first_round_target == "mean":
                compute_mean_from_single_4d(dce, target_path)
            elif args.first_round_target == "firstvol":
                extract_first_vol_from_4d(dce, target_path)
            else:
                raise SystemExit(f"Unknown --first-round-target: {args.first_round_target}")

        print(f"Prepared {dce.name}: {nT} tasks, round1 target={args.first_round_target}")

    if not global_initial.exists():
        compute_mean_across_all_3d_vols(dce_inputs, global_initial)

    print(f"Global original mean: {global_initial}")
    return 0


def cmd_local_reg(args):
    require_exe("antsRegistration")
    require_exe("antsApplyTransforms")

    run_dir = Path(args.run_dir).resolve()
    dce_path = Path(args.dce_nifti).resolve()
    _, out_dir, work_dir, _, _, _ = get_paths(run_dir)

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"]) if args.task_id is None else int(args.task_id)

    warped_path = local_warped_vol_path(work_dir, dce_path, task_id)
    affine_path = local_affine_path(work_dir, dce_path, task_id)
    if warped_path.exists() and affine_path.exists():
        print(f"SKIP LOCAL_REG {dce_path.name} vol={task_id}: outputs already exist")
        return 0

    fixed_reg = get_local_round1_fixed_for_registration(
        out_dir=out_dir,
        work_dir=work_dir,
        dce_path=dce_path,
        target_mode=args.first_round_target,
        reg_smooth_passes=args.reg_smooth_passes,
    )

    img, data = load_4d(dce_path)
    if task_id < 0 or task_id >= data.shape[3]:
        raise SystemExit(f"task_id {task_id} out of range for {dce_path.name}")

    vol = data[..., task_id].astype(np.float32)
    moving_orig = manifest_dir_for(work_dir, dce_path) / f"orig_vol{task_id:04d}.nii.gz"
    if not moving_orig.exists():
        nib.save(nib.Nifti1Image(vol, img.affine, img.header), str(moving_orig))

    moving_for_reg = moving_orig
    if args.reg_smooth_passes > 0:
        moving_reg = manifest_dir_for(work_dir, dce_path) / f"orig_vol{task_id:04d}_smoothed.nii.gz"
        if not moving_reg.exists():
            maybe_write_smoothed_image(img, vol, moving_reg, args.reg_smooth_passes)
        moving_for_reg = moving_reg

    prefix = local_reg_prefix(work_dir, dce_path, task_id)
    out = ants_linear_reg(
        fixed=fixed_reg,
        moving=moving_for_reg,
        out_prefix=prefix,
        transform_type="Affine",
        step_size=args.affine_step,
        conv_iters=args.conv_iters,
        conv_thresh=args.conv_thresh,
        conv_window=args.conv_window,
        shrink_factors=args.shrink_factors,
        smoothing_sigmas=args.smoothing_sigmas,
        verbose=args.verbose,
    )

    warped_path.parent.mkdir(parents=True, exist_ok=True)
    ants_apply_transforms(
        fixed=local_round1_target_path(out_dir, dce_path, args.first_round_target),
        moving=moving_orig,
        out_path=warped_path,
        transforms=[out["affine"]],
        verbose=args.verbose,
    )

    print(f"LOCAL_REG {dce_path.name} vol={task_id} -> {warped_path}")
    return 0


def cmd_local_mean(args):
    run_dir = Path(args.run_dir).resolve()
    dce_path = Path(args.dce_nifti).resolve()
    _, out_dir, work_dir, _, _, _ = get_paths(run_dir)

    out_mean = local_mean_coreg_path(out_dir, dce_path)
    if out_mean.exists():
        print(f"SKIP LOCAL_MEAN {dce_path.name}: output already exists")
        return 0

    _, data = load_4d(dce_path)
    nT = data.shape[3]

    vol_paths = [local_warped_vol_path(work_dir, dce_path, t) for t in range(nT)]
    missing = [p for p in vol_paths if not p.exists()]
    if missing:
        raise SystemExit(
            f"Missing {len(missing)} locally warped vols for {dce_path.name}; "
            f"example: {missing[0]}"
        )

    ref_img, _ = load_3d(vol_paths[0])
    compute_mean_from_3d_stack(vol_paths, ref_img, out_mean)

    print(f"LOCAL_MEAN {dce_path.name} -> {out_mean}")
    return 0


def cmd_mean_to_global(args):
    require_exe("antsRegistration")
    require_exe("antsApplyTransforms")

    run_dir = Path(args.run_dir).resolve()
    dce_path = Path(args.dce_nifti).resolve()
    runno, out_dir, work_dir, _, _, _ = get_paths(run_dir)

    moving_orig = local_mean_coreg_path(out_dir, dce_path)
    fixed_orig = out_dir / f"{runno}_meanDCE_initial_allvols.nii.gz"
    warped_mean = local_mean_to_global_path(out_dir, dce_path)
    affine_path = mean_to_global_affine_path(work_dir, dce_path)

    if warped_mean.exists() and affine_path.exists():
        print(f"SKIP MEAN_TO_GLOBAL {dce_path.name}: outputs already exist")
        return 0

    if not moving_orig.exists():
        raise SystemExit(f"Missing local coreg mean: {moving_orig}")
    if not fixed_orig.exists():
        raise SystemExit(f"Missing global initial mean: {fixed_orig}")

    moving_for_reg = moving_orig
    fixed_for_reg = fixed_orig

    if args.reg_smooth_passes > 0:
        moving_img, moving_dat = load_3d(moving_orig)
        fixed_img, fixed_dat = load_3d(fixed_orig)

        moving_for_reg = manifest_dir_for(work_dir, dce_path) / "mean_to_global_moving_smoothed.nii.gz"
        fixed_for_reg = manifest_dir_for(work_dir, dce_path) / "mean_to_global_fixed_smoothed.nii.gz"

        if not moving_for_reg.exists():
            maybe_write_smoothed_image(moving_img, moving_dat, moving_for_reg, args.reg_smooth_passes)
        if not fixed_for_reg.exists():
            maybe_write_smoothed_image(fixed_img, fixed_dat, fixed_for_reg, args.reg_smooth_passes)

    prefix = mean_to_global_prefix(work_dir, dce_path)
    out = ants_linear_reg(
        fixed=fixed_for_reg,
        moving=moving_for_reg,
        out_prefix=prefix,
        transform_type="Affine",
        step_size=args.affine_step,
        conv_iters=args.conv_iters,
        conv_thresh=args.conv_thresh,
        conv_window=args.conv_window,
        shrink_factors=args.shrink_factors,
        smoothing_sigmas=args.smoothing_sigmas,
        verbose=args.verbose,
    )

    ants_apply_transforms(
        fixed=fixed_orig,
        moving=moving_orig,
        out_path=warped_mean,
        transforms=[out["affine"]],
        verbose=args.verbose,
    )

    print(f"MEAN_TO_GLOBAL {dce_path.name} -> {warped_mean}")
    return 0


def cmd_final_apply(args):
    require_exe("antsApplyTransforms")

    run_dir = Path(args.run_dir).resolve()
    dce_path = Path(args.dce_nifti).resolve()
    runno, out_dir, work_dir, _, _, _ = get_paths(run_dir)

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"]) if args.task_id is None else int(args.task_id)

    out3d = final_apply_vol_path(work_dir, dce_path, task_id)
    if out3d.exists():
        print(f"SKIP FINAL_APPLY {dce_path.name} vol={task_id}: output already exists")
        return 0

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

    local_aff = local_affine_path(work_dir, dce_path, task_id)
    global_aff = mean_to_global_affine_path(work_dir, dce_path)

    if not local_aff.exists():
        raise SystemExit(f"Missing local affine: {local_aff}")
    if not global_aff.exists():
        raise SystemExit(f"Missing global affine: {global_aff}")

    out3d.parent.mkdir(parents=True, exist_ok=True)
    ants_apply_transforms(fixed, moving_orig, out3d, [global_aff, local_aff], verbose=args.verbose)

    print(f"FINAL_APPLY {dce_path.name} vol={task_id} -> {out3d}")
    return 0


def cmd_finalize(args):
    require_exe("antsRegistration")
    require_exe("antsApplyTransforms")

    run_dir = Path(args.run_dir).resolve()
    runno, out_dir, work_dir, _, dce_inputs, t2_input = get_paths(run_dir)

    if not dce_inputs:
        raise SystemExit(f"No DCE inputs found in {run_dir}")

    global_fixed = out_dir / f"{runno}_meanDCE_initial_allvols.nii.gz"
    if not global_fixed.exists():
        raise SystemExit(f"Missing global original mean: {global_fixed}")

    final_mean = out_dir / f"{runno}_meanDCE_coregistered_allvols.nii.gz"
    t2_out = out_dir / t2_input.name

    # if truly final outputs all exist, skip finalize
    final_dce_outputs = [out_dir / dce.name for dce in dce_inputs]
    if final_mean.exists() and t2_out.exists() and all(p.exists() for p in final_dce_outputs):
        print(f"SKIP FINALIZE {runno}: final outputs already exist")
        return 0

    fixed_img, fixed_dat = load_3d(global_fixed)

    final_4d_paths = []
    for dce in dce_inputs:
        _, dat4 = load_4d(dce)
        nT = dat4.shape[3]
        vol_paths = [final_apply_vol_path(work_dir, dce, t) for t in range(nT)]
        missing = [p for p in vol_paths if not p.exists()]
        if missing:
            raise SystemExit(
                f"Missing {len(missing)} final warped vols for {dce.name}; "
                f"example: {missing[0]}"
            )

        out4d = out_dir / dce.name
        if not out4d.exists():
            reg_4d = np.zeros((fixed_dat.shape[0], fixed_dat.shape[1], fixed_dat.shape[2], nT), dtype=np.float32)
            for t, p in enumerate(vol_paths):
                _, vd = load_3d(p)
                if vd.shape != fixed_dat.shape:
                    raise SystemExit(f"Shape mismatch for {p}: {vd.shape} vs {fixed_dat.shape}")
                reg_4d[..., t] = vd
            save_4d_like(reg_4d, fixed_img, out4d)
            print(f"FINAL_4D {out4d}")

        final_4d_paths.append(out4d)

    if not final_mean.exists():
        compute_mean_across_all_3d_vols(final_4d_paths, final_mean)
        print(f"FINAL_MEAN {final_mean}")

    final_mean_for_reg = final_mean
    t2_for_reg = t2_input

    if args.reg_smooth_passes > 0:
        fm_img, fm_dat = load_3d(final_mean)
        t2_img, t2_dat = load_3d(t2_input)

        final_mean_for_reg = work_dir / "T2_fixed_smoothed.nii.gz"
        t2_for_reg = work_dir / "T2_moving_smoothed.nii.gz"

        if not final_mean_for_reg.exists():
            maybe_write_smoothed_image(fm_img, fm_dat, final_mean_for_reg, args.reg_smooth_passes)
        if not t2_for_reg.exists():
            maybe_write_smoothed_image(t2_img, t2_dat, t2_for_reg, args.reg_smooth_passes)

    rigid1_aff = work_dir / "T2_to_finalMean_rigid_0GenericAffine.mat"
    affine1_aff = work_dir / "T2_to_finalMean_affine_0GenericAffine.mat"
    affine2_aff = work_dir / "T2_to_finalMean_grad_affine_0GenericAffine.mat"

    # if T2 output already exists and all transforms exist, skip T2 stage
    if t2_out.exists() and rigid1_aff.exists() and affine1_aff.exists() and affine2_aff.exists():
        print(f"SKIP T2 FINAL {t2_out}: output already exists")
        return 0

    # Pass 1a: rigid on intensity
    t2_rigid_prefix = work_dir / "T2_to_finalMean_rigid_"
    rigid1 = ants_linear_reg(
        fixed=final_mean_for_reg,
        moving=t2_for_reg,
        out_prefix=t2_rigid_prefix,
        transform_type="Rigid",
        step_size=args.t2_rigid_step,
        conv_iters=args.t2_rigid_conv_iters,
        conv_thresh=args.t2_rigid_conv_thresh,
        conv_window=args.t2_rigid_conv_window,
        shrink_factors=args.t2_rigid_shrink_factors,
        smoothing_sigmas=args.t2_rigid_smoothing_sigmas,
        verbose=args.verbose,
    )

    # Pass 1b: affine on intensity
    t2_affine_prefix = work_dir / "T2_to_finalMean_affine_"
    affine1 = ants_linear_reg(
        fixed=final_mean_for_reg,
        moving=t2_for_reg,
        out_prefix=t2_affine_prefix,
        transform_type="Affine",
        step_size=args.t2_affine_step,
        conv_iters=args.t2_affine_conv_iters,
        conv_thresh=args.t2_affine_conv_thresh,
        conv_window=args.t2_affine_conv_window,
        shrink_factors=args.t2_affine_shrink_factors,
        smoothing_sigmas=args.t2_affine_smoothing_sigmas,
        initial_transform=rigid1["affine"],
        verbose=args.verbose,
    )

    final_mean_img, final_mean_dat = load_3d(final_mean)
    affine1_img, affine1_dat = load_3d(affine1["warped"])

    fixed_grad = work_dir / "T2_to_finalMean_fixed_grad.nii.gz"
    moving_grad = work_dir / "T2_to_finalMean_affineWarped_grad.nii.gz"

    if not fixed_grad.exists():
        write_gradient_mag_image(
            in_img=final_mean_img,
            in_data=final_mean_dat,
            out_path=fixed_grad,
            smooth_passes=args.t2_grad_smooth_passes,
        )
    if not moving_grad.exists():
        write_gradient_mag_image(
            in_img=affine1_img,
            in_data=affine1_dat,
            out_path=moving_grad,
            smooth_passes=args.t2_grad_smooth_passes,
        )

    # Pass 2a: rigid on gradients
    t2_grad_rigid_prefix = work_dir / "T2_to_finalMean_grad_rigid_"
    rigid2 = ants_linear_reg(
        fixed=fixed_grad,
        moving=moving_grad,
        out_prefix=t2_grad_rigid_prefix,
        transform_type="Rigid",
        step_size=args.t2_grad_rigid_step,
        conv_iters=args.t2_grad_rigid_conv_iters,
        conv_thresh=args.t2_grad_rigid_conv_thresh,
        conv_window=args.t2_grad_rigid_conv_window,
        shrink_factors=args.t2_grad_rigid_shrink_factors,
        smoothing_sigmas=args.t2_grad_rigid_smoothing_sigmas,
        verbose=args.verbose,
    )

    # Pass 2b: affine on gradients
    t2_grad_affine_prefix = work_dir / "T2_to_finalMean_grad_affine_"
    affine2 = ants_linear_reg(
        fixed=fixed_grad,
        moving=moving_grad,
        out_prefix=t2_grad_affine_prefix,
        transform_type="Affine",
        step_size=args.t2_grad_affine_step,
        conv_iters=args.t2_grad_affine_conv_iters,
        conv_thresh=args.t2_grad_affine_conv_thresh,
        conv_window=args.t2_grad_affine_conv_window,
        shrink_factors=args.t2_grad_affine_shrink_factors,
        smoothing_sigmas=args.t2_grad_affine_smoothing_sigmas,
        initial_transform=rigid2["affine"],
        verbose=args.verbose,
    )

    ants_apply_transforms(
        fixed=final_mean,
        moving=t2_input,
        out_path=t2_out,
        transforms=[affine2["affine"], affine1["affine"], rigid1["affine"]],
        verbose=args.verbose,
    )

    print(f"T2_FINAL {t2_out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Hierarchical DCE coregistration pipeline")

    def add_common(p):
        p.add_argument("--run_dir", required=True)

        p.add_argument("--first_round_target", choices=["mean", "firstvol"], default="mean")
        p.add_argument("--reg_smooth_passes", type=int, default=0)

        p.add_argument("--affine_step", type=float, default=0.05)
        p.add_argument("--conv_iters", default="100x100x100x20")
        p.add_argument("--conv_thresh", type=float, default=1e-7)
        p.add_argument("--conv_window", type=int, default=15)
        p.add_argument("--shrink_factors", default="8x4x2x1")
        p.add_argument("--smoothing_sigmas", default="3x2x1x0vox")

        p.add_argument("--t2_rigid_step", type=float, default=0.1)
        p.add_argument("--t2_rigid_conv_iters", default="100x100x100x20")
        p.add_argument("--t2_rigid_conv_thresh", type=float, default=1e-7)
        p.add_argument("--t2_rigid_conv_window", type=int, default=15)
        p.add_argument("--t2_rigid_shrink_factors", default="8x4x2x1")
        p.add_argument("--t2_rigid_smoothing_sigmas", default="3x2x1x0vox")

        p.add_argument("--t2_affine_step", type=float, default=0.025)
        p.add_argument("--t2_affine_conv_iters", default="100x100x100x20")
        p.add_argument("--t2_affine_conv_thresh", type=float, default=1e-7)
        p.add_argument("--t2_affine_conv_window", type=int, default=15)
        p.add_argument("--t2_affine_shrink_factors", default="8x4x2x1")
        p.add_argument("--t2_affine_smoothing_sigmas", default="3x2x1x0vox")

        p.add_argument("--t2_grad_rigid_step", type=float, default=0.05)
        p.add_argument("--t2_grad_rigid_conv_iters", default="100x100x100x20")
        p.add_argument("--t2_grad_rigid_conv_thresh", type=float, default=1e-7)
        p.add_argument("--t2_grad_rigid_conv_window", type=int, default=15)
        p.add_argument("--t2_grad_rigid_shrink_factors", default="8x4x2x1")
        p.add_argument("--t2_grad_rigid_smoothing_sigmas", default="3x2x1x0vox")

        p.add_argument("--t2_grad_affine_step", type=float, default=0.02)
        p.add_argument("--t2_grad_affine_conv_iters", default="100x100x100x20")
        p.add_argument("--t2_grad_affine_conv_thresh", type=float, default=1e-7)
        p.add_argument("--t2_grad_affine_conv_window", type=int, default=15)
        p.add_argument("--t2_grad_affine_shrink_factors", default="8x4x2x1")
        p.add_argument("--t2_grad_affine_smoothing_sigmas", default="3x2x1x0vox")
        p.add_argument("--t2_grad_smooth_passes", type=int, default=1)

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