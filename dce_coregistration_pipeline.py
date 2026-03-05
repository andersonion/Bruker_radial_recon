#!/usr/bin/env python3
"""
DCE workflow per run folder (e.g., all_niis/z123/):

1) Compute a single mean DCE volume by averaging ACROSS ALL 3D volumes from all DCE 4D files,
   giving each 3D volume equal weight (DO NOT mean each 4D then average those means).

2) Register T2 -> meanDCE, apply transform to create T2_in_meanDCE.nii.gz.

3) Use T2_in_meanDCE as the fixed target for registering EACH individual DCE volume,
   apply transforms, and then re-collate back into original 4D groupings.

Requires:
- Python: nibabel, numpy
- ANTs in PATH: antsRegistrationSyNQuick.sh, antsApplyTransforms
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from glob import glob
from pathlib import Path

import numpy as np
import nibabel as nib


def run(cmd, verbose=False):
    if verbose:
        print("RUN:", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {p.returncode}): {' '.join(cmd)}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}\n"
        )
    return p.stdout


def load_4d(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D NIfTI for {path}, got shape {data.shape}")
    return img, data


def load_3d(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI for {path}, got shape {data.shape}")
    return img, data


def save_3d_like_fixed(data3d, fixed_img, out_path):
    out_img = nib.Nifti1Image(data3d.astype(np.float32), fixed_img.affine, fixed_img.header)
    out_img.header.set_data_dtype(np.float32)
    nib.save(out_img, str(out_path))


def save_4d_like_fixed(data4d, fixed_img, out_path):
    # Make a fresh header to avoid weird dim carryover
    hdr = fixed_img.header.copy()
    out_img = nib.Nifti1Image(data4d.astype(np.float32), fixed_img.affine, hdr)
    out_img.header.set_data_dtype(np.float32)
    nib.save(out_img, str(out_path))


def compute_mean_dce_across_all_volumes(dce_paths, out_mean_path, verbose=False):
    # Load first to set expected spatial shape and affine
    imgs = []
    vol_counts = []
    for p in dce_paths:
        img, data = load_4d(p)
        imgs.append((img, data))
        vol_counts.append(data.shape[3])

    # Consistency check (spatial dims must match)
    base_shape = imgs[0][1].shape[:3]
    for p, (img, data) in zip(dce_paths, imgs):
        if data.shape[:3] != base_shape:
            raise ValueError(f"Spatial shape mismatch: {p} has {data.shape[:3]} vs {base_shape}")

    total_vols = int(np.sum(vol_counts))
    if total_vols == 0:
        raise ValueError("No DCE volumes found to average.")

    if verbose:
        print(f"  DCE files: {len(dce_paths)}")
        for p, n in zip(dce_paths, vol_counts):
            print(f"    {Path(p).name}: {n} vols")
        print(f"  Total 3D volumes across all DCE files: {total_vols}")

    acc = np.zeros(base_shape, dtype=np.float64)  # reduce numerical error
    for (img, data) in imgs:
        # Accumulate each 3D volume equally
        for t in range(data.shape[3]):
            acc += data[..., t]

    mean3d = (acc / float(total_vols)).astype(np.float32)

    fixed_img = imgs[0][0]  # use first DCE's affine/header as meanDCE reference
    save_3d_like_fixed(mean3d, fixed_img, out_mean_path)
    return fixed_img, out_mean_path


def ants_reg_quick(fixed, moving, out_prefix, transform="r", verbose=False):
    """
    transform:
      r = rigid
      a = affine
      s = SyN (nonlinear)  [much slower]
    """
    cmd = [
        "antsRegistrationSyNQuick.sh",
        "-d", "3",
        "-f", str(fixed),
        "-m", str(moving),
        "-t", transform,
        "-o", str(out_prefix),
    ]
    run(cmd, verbose=verbose)


def ants_apply(fixed, moving, out_path, transforms, interp="Linear", verbose=False):
    cmd = [
        "antsApplyTransforms",
        "-d", "3",
        "-i", str(moving),
        "-r", str(fixed),
        "-o", str(out_path),
        "-n", interp,
    ]
    # ANTs applies in listed order (last is applied first internally); we provide the normal ANTs CLI order.
    for t in transforms:
        cmd += ["-t", str(t)]
    run(cmd, verbose=verbose)


def find_single(patterns, folder):
    hits = []
    for pat in patterns:
        hits.extend(glob(str(Path(folder) / pat)))
    hits = sorted(set(hits))
    if len(hits) == 0:
        return None
    if len(hits) > 1:
        raise RuntimeError(f"Ambiguous matches in {folder} for {patterns}:\n  " + "\n  ".join(hits))
    return hits[0]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Mean DCE across all volumes, register T2->mean, then register each DCE volume to T2_in_mean, collate back."
    )
    ap.add_argument(
        "--all_niis_dir",
        required=True,
        help="Path to all_niis (contains z*/ run folders).",
    )
    ap.add_argument(
        "--run_glob",
        default="z*",
        help="Glob under all_niis_dir for run folders (default: z*).",
    )
    ap.add_argument(
        "--dce_suffixes",
        default="DCE_baseline,DCE_block1,DCE_block2",
        help="Comma-separated stems to locate DCE 4D NIfTIs (default: %(default)s).",
    )
    ap.add_argument(
        "--t2_suffix",
        default="T2",
        help="Stem used to locate T2 NIfTI (default: %(default)s).",
    )
    ap.add_argument(
        "--ants_transform_t2_to_mean",
        default="r",
        choices=["r", "a", "s"],
        help="ANTs quick transform for T2->meanDCE: r=rigid, a=affine, s=SyN (default: %(default)s).",
    )
    ap.add_argument(
        "--ants_transform_dce_to_t2",
        default="r",
        choices=["r", "a", "s"],
        help="ANTs quick transform for each DCE vol -> T2_in_mean: r=rigid, a=affine, s=SyN (default: %(default)s).",
    )
    ap.add_argument(
        "--keep_work",
        action="store_true",
        help="Keep per-run temporary working directory (for debugging).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs if they exist.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    all_niis_dir = Path(args.all_niis_dir).resolve()
    if not all_niis_dir.is_dir():
        raise SystemExit(f"Not a directory: {all_niis_dir}")

    run_dirs = sorted([p for p in all_niis_dir.glob(args.run_glob) if p.is_dir()])
    if not run_dirs:
        raise SystemExit(f"No run folders found under {all_niis_dir} with glob '{args.run_glob}'")

    dce_suffixes = [s.strip() for s in args.dce_suffixes.split(",") if s.strip()]
    if not dce_suffixes:
        raise SystemExit("No DCE suffixes provided.")

    # Basic dependency check
    for exe in ["antsRegistrationSyNQuick.sh", "antsApplyTransforms"]:
        if shutil.which(exe) is None:
            raise SystemExit(f"Missing dependency in PATH: {exe}")

    for run_dir in run_dirs:
        print(f"\n=== Processing {run_dir.name} ===")

        # Locate T2
        t2 = find_single([f"*_{args.t2_suffix}.nii.gz"], run_dir)
        if t2 is None:
            print(f"  SKIP: no T2 found matching '*_{args.t2_suffix}.nii.gz'")
            continue

        # Locate DCE 4D files (any subset is allowed, but need at least one)
        dce_paths = []
        for suf in dce_suffixes:
            hit = find_single([f"*_{suf}.nii.gz"], run_dir)
            if hit is not None:
                dce_paths.append(hit)

        if not dce_paths:
            print(f"  SKIP: no DCE files found for suffixes {dce_suffixes}")
            continue

        # Outputs
        mean_dce_path = run_dir / "meanDCE_allvols.nii.gz"
        t2_in_mean_path = run_dir / "T2_in_meanDCE.nii.gz"

        out_dce_4d = {
            suf: run_dir / f"{suf}_regTo_T2inMeanDCE.nii.gz"
            for suf in dce_suffixes
        }

        # Overwrite logic
        if not args.overwrite:
            if mean_dce_path.exists() and t2_in_mean_path.exists() and any(p.exists() for p in out_dce_4d.values()):
                print("  Outputs exist (and --overwrite not set). Skipping.")
                continue

        # Work dir
        if args.keep_work:
            workdir = run_dir / "_work_reg"
            workdir.mkdir(exist_ok=True)
            tmp_cm = None
        else:
            tmp_cm = tempfile.TemporaryDirectory(prefix=f"{run_dir.name}_work_")
            workdir = Path(tmp_cm.name)

        try:
            # 1) Mean DCE across all volumes (equal weight per volume)
            if args.verbose:
                print("  Step 1: computing mean DCE across all 3D volumes (equal-weight)")
            fixed_img, _ = compute_mean_dce_across_all_volumes(
                dce_paths=dce_paths,
                out_mean_path=mean_dce_path,
                verbose=args.verbose,
            )

            # 2) Register T2 -> meanDCE, apply to T2
            if args.verbose:
                print("  Step 2: registering T2 -> meanDCE")
            t2_to_mean_prefix = workdir / "T2_to_meanDCE_"
            ants_reg_quick(
                fixed=mean_dce_path,
                moving=t2,
                out_prefix=t2_to_mean_prefix,
                transform=args.ants_transform_t2_to_mean,
                verbose=args.verbose,
            )

            # antsRegistrationSyNQuick.sh outputs:
            #   <prefix>Warped.nii.gz (moving warped into fixed)
            #   <prefix>0GenericAffine.mat (always)
            #   <prefix>1Warp.nii.gz (+ inverse) if nonlinear
            t2_warped = str(t2_to_mean_prefix) + "Warped.nii.gz"
            if not Path(t2_warped).exists():
                raise RuntimeError(f"Expected ANTs output not found: {t2_warped}")
            shutil.copy2(t2_warped, t2_in_mean_path)

            # Determine transforms (for applying to other images if needed)
            t2_aff = str(t2_to_mean_prefix) + "0GenericAffine.mat"
            t2_warp = str(t2_to_mean_prefix) + "1Warp.nii.gz"
            t2_transforms = []
            if Path(t2_warp).exists():
                # nonlinear: warp then affine (ANTs CLI order)
                t2_transforms = [t2_warp, t2_aff]
            else:
                # linear-only
                t2_transforms = [t2_aff]

            # 3) Register each DCE volume -> T2_in_meanDCE, apply, collate
            if args.verbose:
                print("  Step 3: registering each DCE volume -> T2_in_meanDCE and re-collating 4D")

            # Load fixed target (T2_in_mean) so we can adopt its affine/header for outputs
            fixed_t2_img, fixed_t2_data = load_3d(t2_in_mean_path)

            for suf in dce_suffixes:
                src = find_single([f"*_{suf}.nii.gz"], run_dir)
                if src is None:
                    continue  # allow missing blocks

                if args.verbose:
                    print(f"    DCE group: {suf}  ({Path(src).name})")

                dce_img, dce_data = load_4d(src)
                nT = dce_data.shape[3]

                # Make per-volume temp files, register, apply, store results
                reg_vols = np.zeros((fixed_t2_data.shape[0], fixed_t2_data.shape[1], fixed_t2_data.shape[2], nT), dtype=np.float32)

                # Sanity: if DCE spatial shape differs from fixed target, ANTs will resample fine, but warn.
                if dce_data.shape[:3] != fixed_t2_data.shape:
                    if args.verbose:
                        print(f"      NOTE: DCE shape {dce_data.shape[:3]} differs from fixed target {fixed_t2_data.shape}")

                for t in range(nT):
                    vol = dce_data[..., t].astype(np.float32)

                    moving_vol_path = workdir / f"{suf}_vol{t:04d}.nii.gz"
                    # Save moving volume using its own affine; ANTs will handle physical space via headers
                    nib.save(nib.Nifti1Image(vol, dce_img.affine, dce_img.header), str(moving_vol_path))

                    vol_prefix = workdir / f"{suf}_vol{t:04d}_to_T2inMean_"
                    ants_reg_quick(
                        fixed=t2_in_mean_path,
                        moving=moving_vol_path,
                        out_prefix=vol_prefix,
                        transform=args.ants_transform_dce_to_t2,
                        verbose=args.verbose,
                    )

                    warped = str(vol_prefix) + "Warped.nii.gz"
                    if not Path(warped).exists():
                        raise RuntimeError(f"Expected ANTs output not found: {warped}")

                    wimg, wdat = load_3d(warped)
                    # Ensure it matches fixed grid (it should, because fixed was the reference)
                    if wdat.shape != fixed_t2_data.shape:
                        raise RuntimeError(f"Warped vol shape {wdat.shape} != fixed shape {fixed_t2_data.shape} for {suf} t={t}")
                    reg_vols[..., t] = wdat.astype(np.float32)

                # Save collated 4D with fixed target geometry
                out_path = out_dce_4d[suf]
                save_4d_like_fixed(reg_vols, fixed_t2_img, out_path)

            print("  DONE")
            print(f"    meanDCE: {mean_dce_path.name}")
            print(f"    T2_in_meanDCE: {t2_in_mean_path.name}")
            for suf in dce_suffixes:
                p = out_dce_4d[suf]
                if p.exists():
                    print(f"    {p.name}")

        finally:
            if tmp_cm is not None:
                tmp_cm.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())