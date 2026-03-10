#!/usr/bin/env python3

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    binary_opening,
    binary_propagation,
    distance_transform_edt,
    gaussian_filter,
    gaussian_gradient_magnitude,
    generate_binary_structure,
    label,
)


def largest_component(mask: np.ndarray, structure=None) -> np.ndarray:
    if structure is None:
        structure = generate_binary_structure(3, 2)
    lab, nlab = label(mask, structure=structure)
    if nlab == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    best = counts.argmax()
    return lab == best


def robust_normalize(vol: np.ndarray, mask: np.ndarray, low_pct=2.0, high_pct=98.0) -> np.ndarray:
    vals = vol[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise RuntimeError("No finite values inside mask for normalization.")
    lo = np.percentile(vals, low_pct)
    hi = np.percentile(vals, high_pct)
    if hi <= lo:
        raise RuntimeError(f"Bad normalization range: lo={lo}, hi={hi}")
    out = (vol - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def remove_small_components(mask: np.ndarray, min_voxels: int, structure=None) -> np.ndarray:
    if structure is None:
        structure = generate_binary_structure(3, 2)
    if min_voxels <= 1:
        return mask.copy()
    lab, nlab = label(mask, structure=structure)
    if nlab == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(lab.ravel())
    keep = np.zeros(nlab + 1, dtype=bool)
    for k in range(1, nlab + 1):
        if counts[k] >= min_voxels:
            keep[k] = True
    return keep[lab]


def boundary_seed(mask: np.ndarray) -> np.ndarray:
    seed = np.zeros_like(mask, dtype=bool)
    seed[0, :, :] |= mask[0, :, :]
    seed[-1, :, :] |= mask[-1, :, :]
    seed[:, 0, :] |= mask[:, 0, :]
    seed[:, -1, :] |= mask[:, -1, :]
    seed[:, :, 0] |= mask[:, :, 0]
    seed[:, :, -1] |= mask[:, :, -1]
    return seed


def constrained_region_grow(seed: np.ndarray, allowed: np.ndarray, structure, max_iters: int) -> np.ndarray:
    grown = seed.copy()
    for _ in range(max_iters):
        nxt = binary_dilation(grown, structure=structure) & allowed
        if np.array_equal(nxt, grown):
            break
        grown = nxt
    return grown


def save_like(path: Path, data: np.ndarray, ref_img: nib.Nifti1Image):
    img = nib.Nifti1Image(data, ref_img.affine, ref_img.header)
    nib.save(img, str(path))


def main():
    ap = argparse.ArgumentParser(
        description="Build a brain mask from BFC T2 using shell/barrier logic plus cavity filling, shell inclusion, and rescue growth."
    )
    ap.add_argument("--bfc", required=True)
    ap.add_argument("--support_mask", required=True)
    ap.add_argument("--out_mask", required=True)
    ap.add_argument("--out_masked_bfc", required=True)
    ap.add_argument("--debug_prefix", default=None)

    ap.add_argument("--smooth_sigma", type=float, default=1.0)
    ap.add_argument("--grad_sigma", type=float, default=1.0)
    ap.add_argument("--grad_threshold", type=float, default=0.10)

    # physical cleanup thresholds, in mm^3
    ap.add_argument("--min_barrier_volume_mm3", type=float, default=0.05)
    ap.add_argument("--min_enclosed_volume_mm3", type=float, default=1.0)

    # barrier morphology
    ap.add_argument("--close_barrier_iters", type=int, default=1)

    # shell inclusion
    ap.add_argument("--shell_thickness_mm", type=float, default=0.35)
    ap.add_argument("--shell_grad_min", type=float, default=0.03)
    ap.add_argument("--shell_intensity_min", type=float, default=0.02)
    ap.add_argument("--shell_use_gate", action="store_true")

    # rescue growth
    ap.add_argument("--rescue_max_distance_mm", type=float, default=0.60,
                    help="Maximum outward rescue distance from current mask.")
    ap.add_argument("--rescue_intensity_min", type=float, default=0.05,
                    help="Minimum normalized intensity for rescue voxels.")
    ap.add_argument("--rescue_gradient_max", type=float, default=0.60,
                    help="Maximum normalized gradient for rescue voxels.")
    ap.add_argument("--rescue_iters", type=int, default=50,
                    help="Max constrained dilation iterations for rescue growth.")

    # final mask morphology
    ap.add_argument("--brain_close_iters", type=int, default=2)
    ap.add_argument("--brain_open_iters", type=int, default=0)
    ap.add_argument("--brain_dilate_iters", type=int, default=1)
    ap.add_argument("--brain_fill_holes", action="store_true")

    ap.add_argument("--tight_mask", action="store_true")
    ap.add_argument("--tight_erode_iters", type=int, default=1)

    args = ap.parse_args()

    bfc_img = nib.load(args.bfc)
    sup_img = nib.load(args.support_mask)

    bfc = bfc_img.get_fdata(dtype=np.float32)
    support = sup_img.get_fdata() > 0

    if bfc.shape != support.shape:
        raise RuntimeError("BFC image and support mask shapes do not match.")
    if not np.any(support):
        raise RuntimeError("Support mask is empty.")

    zooms = bfc_img.header.get_zooms()[:3]
    voxel_volume_mm3 = float(zooms[0] * zooms[1] * zooms[2])
    if voxel_volume_mm3 <= 0:
        raise RuntimeError(f"Bad voxel volume from header: {voxel_volume_mm3}")

    mean_spacing = float(np.mean(zooms))
    shell_thickness_vox = max(1.0, args.shell_thickness_mm / max(mean_spacing, 1e-6))
    rescue_max_distance_vox = max(1.0, args.rescue_max_distance_mm / max(mean_spacing, 1e-6))

    min_barrier_vox = max(1, int(round(args.min_barrier_volume_mm3 / voxel_volume_mm3)))
    min_enclosed_vox = max(1, int(round(args.min_enclosed_volume_mm3 / voxel_volume_mm3)))

    structure = generate_binary_structure(3, 2)

    bfc_smooth = gaussian_filter(bfc, sigma=args.smooth_sigma)
    inten_norm = robust_normalize(bfc_smooth, support, low_pct=2.0, high_pct=98.0)

    grad = gaussian_gradient_magnitude(bfc_smooth, sigma=args.grad_sigma)
    grad_norm = robust_normalize(grad, support, low_pct=2.0, high_pct=98.0)

    # Step 1: barrier from gradient shell
    barrier_raw = (grad_norm >= args.grad_threshold) & support
    barrier = remove_small_components(barrier_raw, min_barrier_vox, structure=structure)

    if args.close_barrier_iters > 0 and np.any(barrier):
        barrier = binary_closing(barrier, structure=structure, iterations=args.close_barrier_iters)

    open_space = support & (~barrier)

    seeds = boundary_seed(open_space)
    outside = binary_propagation(seeds, structure=structure, mask=open_space)

    enclosed = open_space & (~outside)
    enclosed = remove_small_components(enclosed, min_enclosed_vox, structure=structure)
    enclosed = largest_component(enclosed, structure=structure)

    if not np.any(enclosed):
        raise RuntimeError("No enclosed cavity found. Try lowering grad threshold or barrier cleanup.")

    # Step 2: fill holes so ventricles/internal dark regions are INCLUDED
    cavity_filled = binary_fill_holes(enclosed)
    cavity_filled = largest_component(cavity_filled, structure=structure)

    # Step 3: add shell band around cavity
    dist_to_cavity = distance_transform_edt(~cavity_filled)
    shell_zone = support & (dist_to_cavity <= shell_thickness_vox)

    if args.shell_use_gate:
        shell_gate = (grad_norm >= args.shell_grad_min) | (inten_norm >= args.shell_intensity_min)
        shell_zone &= shell_gate

    brain = cavity_filled | shell_zone
    brain &= support
    brain = largest_component(brain, structure=structure)

    # Step 4: rescue growth for missed tissue / brainstem
    dist_to_brain = distance_transform_edt(~brain)
    rescue_zone = support & (dist_to_brain <= rescue_max_distance_vox)

    rescue_gate = (inten_norm >= args.rescue_intensity_min) | (grad_norm <= args.rescue_gradient_max)
    rescue_allowed = rescue_zone & rescue_gate

    rescue_allowed |= brain
    brain_rescued = constrained_region_grow(brain, rescue_allowed, structure, args.rescue_iters)
    brain = largest_component(brain_rescued, structure=structure)

    # Step 5: cleanup
    if args.brain_fill_holes:
        brain = binary_fill_holes(brain)

    if args.brain_close_iters > 0:
        brain = binary_closing(brain, structure=structure, iterations=args.brain_close_iters)

    if args.brain_open_iters > 0:
        brain = binary_opening(brain, structure=structure, iterations=args.brain_open_iters)

    if args.brain_dilate_iters > 0:
        brain = binary_dilation(brain, structure=structure, iterations=args.brain_dilate_iters)

    brain &= support
    brain = largest_component(brain, structure=structure)

    # Always fill holes at end so ventricles/internal dark structures remain included
    brain = binary_fill_holes(brain)
    brain = largest_component(brain, structure=structure)

    if args.tight_mask:
        brain = binary_erosion(brain, structure=structure, iterations=args.tight_erode_iters)
        brain = largest_component(brain, structure=structure)
        brain = binary_fill_holes(brain)

    out_mask = brain.astype(np.uint8)
    out_masked_bfc = np.where(out_mask > 0, bfc, 0.0).astype(np.float32)

    save_like(Path(args.out_mask), out_mask, bfc_img)
    save_like(Path(args.out_masked_bfc), out_masked_bfc, bfc_img)

    if args.debug_prefix:
        prefix = Path(args.debug_prefix)
        save_like(prefix.with_name(prefix.name + "_grad.nii.gz"), grad.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_grad_norm.nii.gz"), grad_norm.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_inten_norm.nii.gz"), inten_norm.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_barrier_raw.nii.gz"), barrier_raw.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_barrier.nii.gz"), barrier.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_open_space.nii.gz"), open_space.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_outside.nii.gz"), outside.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_enclosed.nii.gz"), enclosed.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_cavity_filled.nii.gz"), cavity_filled.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_dist_to_cavity.nii.gz"), dist_to_cavity.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_shell_zone.nii.gz"), shell_zone.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_dist_to_brain.nii.gz"), dist_to_brain.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_rescue_allowed.nii.gz"), rescue_allowed.astype(np.uint8), bfc_img)


if __name__ == "__main__":
    main()