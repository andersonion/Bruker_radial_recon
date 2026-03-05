#!/usr/bin/env python3
"""
Per-run DCE co-registration pipeline.

Input structure:
  all_niis/
    z<runno>/                         (input run folder; contains renamed NIfTIs)
      <runno>_DCE_baseline.nii.gz      (4D)
      <runno>_DCE_block1.nii.gz        (4D)
      <runno>_DCE_block2.nii.gz        (4D)
      <runno>_T2.nii.gz                (3D)

Output structure (created under all_niis/):
  <runno>_coregistered/
    sbatch/                           (slurm scripts + slurm-%j.out)
    work/                             (all temp files go here; NOT /tmp)
    <runno>_DCE_baseline.nii.gz        (coregistered; SAME BASENAME AS INPUT)
    <runno>_DCE_block1.nii.gz          (coregistered; SAME BASENAME AS INPUT)
    <runno>_DCE_block2.nii.gz          (coregistered; SAME BASENAME AS INPUT)
    <runno>_T2.nii.gz                  (T2 warped to meanDCE; SAME BASENAME AS INPUT)
    <runno>_meanDCE_initial_allvols.nii.gz
    <runno>_meanDCE_coregistered_allvols.nii.gz

Algorithm:
1) Build one initial mean DCE volume by averaging ACROSS ALL 3D volumes from all DCE 4D files,
   giving each 3D volume equal weight (DO NOT average 4D NIfTIs individually then average those).
2) Register T2 -> meanDCE (rigid/affine/SyN), write warped T2 into output folder with SAME filename as input T2.
3) Use warped T2 as fixed target. For EACH individual DCE volume:
   register vol -> warped_T2 and apply
4) Re-collate per original 4D file and write to output folder with SAME filename as input.
5) Build mean DCE from the COREGISTERED DCE 4D outputs (again equal weight across all 3D vols).

Dependencies:
- Python: numpy, nibabel
- ANTs in PATH: antsRegistrationSyNQuick.sh
"""

import argparse
import shutil
import subprocess
from glob import glob
from pathlib import Path

import numpy as np
import nibabel as nib


def run_cmd(cmd, verbose=False):
    if verbose:
        print("RUN:", " ".join(map(str, cmd)))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {p.returncode}): {' '.join(map(str, cmd))}\n\n"
            f"STDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}\n"
        )
    return p.stdout


def find_unique(folder: Path, pattern: str):
    hits = sorted(folder.glob(pattern))
    if not hits:
        return None
    if len(hits) > 1:
        raise RuntimeError(f"Ambiguous matches in {folder} for '{pattern}':\n  " + "\n  ".join(map(str, hits)))
    return hits[0]


def load_3d(path: Path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI: {path} but got shape {data.shape}")
    return img, data


def load_4d(path: Path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D NIfTI: {path} but got shape {data.shape}")
    return img, data


def save_3d_like(data3d: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path):
    out = nib.Nifti1Image(data3d.astype(np.float32), ref_img.affine, ref_img.header)
    out.header.set_data_dtype(np.float32)
    nib.save(out, str(out_path))


def save_4d_like(data4d: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path):
    hdr = ref_img.header.copy()
    out = nib.Nifti1Image(data4d.astype(np.float32), ref_img.affine, hdr)
    out.header.set_data_dtype(np.float32)
    nib.save(out, str(out_path))


def compute_mean_across_all_3d_vols(dce_4d_paths, out_mean_3d: Path, verbose=False):
    """
    Compute mean across ALL 3D volumes pooled from multiple 4D inputs,
    equal weight per 3D volume.
    """
    imgs_data = []
    vol_counts = []

    for p in dce_4d_paths:
        img, data = load_4d(p)
        imgs_data.append((img, data))
        vol_counts.append(data.shape[3])

    base_shape = imgs_data[0][1].shape[:3]
    for p, (_, d) in zip(dce_4d_paths, imgs_data):
        if d.shape[:3] != base_shape:
            raise ValueError(f"Spatial mismatch: {p} has {d.shape[:3]} vs {base_shape}")

    total_vols = int(np.sum(vol_counts))
    if total_vols <= 0:
        raise ValueError("No volumes found to average.")

    if verbose:
        print(f"  Mean: pooling {len(dce_4d_paths)} DCE files, total 3D vols = {total_vols}")
        for p, n in zip(dce_4d_paths, vol_counts):
            print(f"    {p.name}: {n} vols")

    acc = np.zeros(base_shape, dtype=np.float64)
    for (_, d) in imgs_data:
        # accumulate each timepoint equally
        for t in range(d.shape[3]):
            acc += d[..., t]

    mean3d = (acc / float(total_vols)).astype(np.float32)

    ref_img = imgs_data[0][0]
    save_3d_like(mean3d, ref_img, out_mean_3d)
    return out_mean_3d


def ants_reg_quick(fixed: Path, moving: Path, out_prefix: Path, transform: str, verbose=False):
    """
    transform: r (rigid), a (affine), s (SyN)
    """
    cmd = [
        "antsRegistrationSyNQuick.sh",
        "-d", "3",
        "-f", str(fixed),
        "-m", str(moving),
        "-t", transform,
        "-o", str(out_prefix),
    ]
    run_cmd(cmd, verbose=verbose)

    warped = Path(str(out_prefix) + "Warped.nii.gz")
    if not warped.exists():
        raise RuntimeError(f"Expected ANTs output not found: {warped}")
    return warped


def parse_runno(run_dir: Path) -> str:
    bn = run_dir.name
    if bn.startswith("z") and len(bn) > 1:
        return bn[1:]
    return bn


def parse_args():
    ap = argparse.ArgumentParser(description="Co-register DCE volumes per run using ANTs; output to <runno>_coregistered.")
    ap.add_argument("--run_dir", required=True, help="Path to a single run folder (e.g., all_niis/z123).")
    ap.add_argument(
        "--dce_suffixes",
        default="DCE_baseline,DCE_block1,DCE_block2",
        help="Comma-separated stems to locate DCE 4D NIfTIs (default: %(default)s).",
    )
    ap.add_argument("--t2_suffix", default="T2", help="Stem used to locate T2 NIfTI (default: %(default)s).")
    ap.add_argument(
        "--ants_transform_t2_to_mean",
        default="r",
        choices=["r", "a", "s"],
        help="ANTs quick transform for T2->meanDCE: r|a|s (default: %(default)s).",
    )
    ap.add_argument(
        "--ants_transform_dce_to_t2",
        default="r",
        choices=["r", "a", "s"],
        help="ANTs quick transform for each DCE vol -> warped_T2: r|a|s (default: %(default)s).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they exist.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return ap.parse_args()


def main():
    args = parse_args()

    # Basic deps
    for exe in ["antsRegistrationSyNQuick.sh"]:
        if shutil.which(exe) is None:
            raise SystemExit(f"Missing dependency in PATH: {exe}")

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    runno = parse_runno(run_dir)
    all_niis_dir = run_dir.parent

    out_dir = all_niis_dir / f"{runno}_coregistered"
    sbatch_dir = out_dir / "sbatch"
    work_dir = out_dir / "work"
    out_dir.mkdir(parents=True, exist_ok=True)
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    dce_suffixes = [s.strip() for s in args.dce_suffixes.split(",") if s.strip()]
    if not dce_suffixes:
        raise SystemExit("No DCE suffixes provided.")

    # Locate inputs (in run_dir)
    t2_in = find_unique(run_dir, f"*_{args.t2_suffix}.nii.gz")
    if t2_in is None:
        print(f"SKIP {run_dir.name}: no T2 matching '*_{args.t2_suffix}.nii.gz'")
        return 0

    dce_ins = []
    for suf in dce_suffixes:
        p = find_unique(run_dir, f"*_{suf}.nii.gz")
        if p is not None:
            dce_ins.append(p)

    if not dce_ins:
        print(f"SKIP {run_dir.name}: no DCE found for suffixes {dce_suffixes}")
        return 0

    # Outputs (same basenames as inputs)
    t2_out = out_dir / t2_in.name
    dce_outs = {p: (out_dir / p.name) for p in dce_ins}

    mean_initial = out_dir / f"{runno}_meanDCE_initial_allvols.nii.gz"
    mean_coreg = out_dir / f"{runno}_meanDCE_coregistered_allvols.nii.gz"

    # Skip logic
    if not args.overwrite:
        all_done = (
            mean_initial.exists()
            and mean_coreg.exists()
            and t2_out.exists()
            and all(dst.exists() for dst in dce_outs.values())
        )
        if all_done:
            print(f"SKIP {run_dir.name}: outputs already exist (use --overwrite to redo)")
            return 0

    print(f"\n=== {run_dir.name}  (runno={runno}) ===")
    print(f"  Input:  {run_dir}")
    print(f"  Output: {out_dir}")
    if args.verbose:
        print(f"  Work:   {work_dir}")

    # Step 1: initial mean DCE from ORIGINAL inputs (equal weight per 3D volume)
    if args.verbose:
        print("  Step 1: mean DCE (initial) across ALL original DCE volumes (equal-weight)")
    compute_mean_across_all_3d_vols(dce_ins, mean_initial, verbose=args.verbose)

    # Step 2: register T2 -> mean_initial; save warped into out_dir with SAME filename as input
    if args.verbose:
        print("  Step 2: T2 -> meanDCE (initial) registration")
    t2_prefix = work_dir / "T2_to_meanDCE_"
    t2_warped = ants_reg_quick(
        fixed=mean_initial,
        moving=t2_in,
        out_prefix=t2_prefix,
        transform=args.ants_transform_t2_to_mean,
        verbose=args.verbose,
    )
    shutil.copy2(t2_warped, t2_out)

    # Step 3: register each DCE volume -> warped T2; collate and write to out_dir with SAME basenames
    fixed_img, fixed_data = load_3d(t2_out)

    for dce_in in dce_ins:
        if args.verbose:
            print(f"  Step 3: coregister DCE 4D -> warped T2: {dce_in.name}")

        dce_img, dce_data = load_4d(dce_in)
        nT = dce_data.shape[3]

        reg_4d = np.zeros((fixed_data.shape[0], fixed_data.shape[1], fixed_data.shape[2], nT), dtype=np.float32)

        # per-group work folder to keep things tidy
        grp_work = work_dir / dce_in.stem
        grp_work.mkdir(parents=True, exist_ok=True)

        for t in range(nT):
            moving_vol = dce_data[..., t].astype(np.float32)
            moving_path = grp_work / f"vol{t:04d}.nii.gz"
            nib.save(nib.Nifti1Image(moving_vol, dce_img.affine, dce_img.header), str(moving_path))

            out_prefix = grp_work / f"vol{t:04d}_to_T2_"
            warped = ants_reg_quick(
                fixed=t2_out,
                moving=moving_path,
                out_prefix=out_prefix,
                transform=args.ants_transform_dce_to_t2,
                verbose=args.verbose,
            )

            _, wdat = load_3d(warped)
            if wdat.shape != fixed_data.shape:
                raise RuntimeError(f"Warped shape mismatch at t={t} for {dce_in.name}: {wdat.shape} vs {fixed_data.shape}")

            reg_4d[..., t] = wdat

        save_4d_like(reg_4d, fixed_img, dce_outs[dce_in])

    # Step 4: mean DCE from COREGISTERED outputs (equal weight across all 3D vols)
    if args.verbose:
        print("  Step 4: mean DCE (coregistered) across ALL coregistered DCE volumes (equal-weight)")
    coreg_dce_paths = [dce_outs[p] for p in dce_ins]
    compute_mean_across_all_3d_vols(coreg_dce_paths, mean_coreg, verbose=args.verbose)

    print("  DONE")
    print(f"    {mean_initial.name}")
    print(f"    {t2_out.name}")
    for dce_in in dce_ins:
        print(f"    {dce_outs[dce_in].name}")
    print(f"    {mean_coreg.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())