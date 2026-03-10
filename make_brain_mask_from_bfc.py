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


def save_like(path: Path, data: np.ndarray, ref_img: nib.Nifti1Image):
    img = nib.Nifti1Image(data, ref_img.affine, ref_img.header)
    nib.save(img, str(path))


def main():
    ap = argparse.ArgumentParser(
        description="Build an edge-aware brain mask from a BFC T2 and a generous support mask."
    )
    ap.add_argument("--bfc", required=True, help="Bias-corrected T2")
    ap.add_argument("--support_mask", required=True, help="Generous support/head mask")
    ap.add_argument("--out_mask", required=True, help="Output brain mask")
    ap.add_argument("--out_masked_bfc", required=True, help="BFC image masked by final brain mask")
    ap.add_argument("--debug_prefix", default=None, help="Optional prefix for debug outputs")

    # smoothing / gradients
    ap.add_argument("--smooth_sigma", type=float, default=1.0, help="Gaussian smoothing sigma for intensity image")
    ap.add_argument("--grad_sigma", type=float, default=1.0, help="Sigma for gradient magnitude")

    # seed generation
    ap.add_argument("--seed_min_distance", type=float, default=4.0,
                    help="Minimum distance (voxels) from support boundary for seed search")
    ap.add_argument("--seed_score_pct", type=float, default=92.0,
                    help="Percentile threshold on seed score within interior band")
    ap.add_argument("--seed_erode_iters", type=int, default=1,
                    help="Optional erosion of seed after thresholding")

    # score weights
    ap.add_argument("--center_weight", type=float, default=1.5,
                    help="Exponent on normalized distance-from-boundary term")
    ap.add_argument("--grad_penalty", type=float, default=2.0,
                    help="Penalty weight for gradient in seed score")

    # grow candidate / stopping
    ap.add_argument("--candidate_intensity_min", type=float, default=0.08,
                    help="Minimum normalized intensity allowed during growth")
    ap.add_argument("--candidate_gradient_max", type=float, default=0.45,
                    help="Maximum normalized gradient allowed during growth")
    ap.add_argument("--max_grow_iters", type=int, default=200,
                    help="Maximum dilation-grow iterations")

    # cleanup
    ap.add_argument("--close_iters", type=int, default=2)
    ap.add_argument("--dilate_iters", type=int, default=1)
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

    structure = generate_binary_structure(3, 2)

    # Smooth intensity and normalize inside support
    bfc_smooth = gaussian_filter(bfc, sigma=args.smooth_sigma)
    inten_norm = robust_normalize(bfc_smooth, support, low_pct=2.0, high_pct=98.0)

    # Gradient magnitude, normalized inside support
    grad = gaussian_gradient_magnitude(bfc_smooth, sigma=args.grad_sigma)
    grad_norm = robust_normalize(grad, support, low_pct=2.0, high_pct=98.0)

    # Distance from support boundary: favors central/intracranial stuff over boundary junk
    dist = distance_transform_edt(support)
    dist_norm = dist / max(float(dist.max()), 1.0)

    # Interior band for seed search
    interior = support & (dist >= args.seed_min_distance)
    if not np.any(interior):
        interior = support.copy()

    # Seed score:
    # bright + central + low gradient
    seed_score = inten_norm * (dist_norm ** args.center_weight) / (1.0 + args.grad_penalty * grad_norm)

    seed_vals = seed_score[interior]
    thresh = np.percentile(seed_vals, args.seed_score_pct)
    seed = interior & (seed_score >= thresh)

    seed = largest_component(seed, structure=structure)
    if args.seed_erode_iters > 0 and np.any(seed):
        seed = binary_erosion(seed, structure=structure, iterations=args.seed_erode_iters)

    if not np.any(seed):
        raise RuntimeError("Seed generation failed; got empty seed.")

    # Candidate region for growth:
    # stay inside support, avoid low intensity and strong edges
    candidate = support & (inten_norm >= args.candidate_intensity_min) & (grad_norm <= args.candidate_gradient_max)

    # Ensure seed is always included
    candidate |= seed

    # Seeded binary region growing by constrained dilation
    grown = seed.copy()
    for _ in range(args.max_grow_iters):
        nxt = binary_dilation(grown, structure=structure) & candidate
        if np.array_equal(nxt, grown):
            break
        grown = nxt

    # Cleanup
    grown = largest_component(grown, structure=structure)
    grown = binary_closing(grown, structure=structure, iterations=args.close_iters)
    grown = binary_fill_holes(grown)
    grown = binary_dilation(grown, structure=structure, iterations=args.dilate_iters)

    # Never allow outside support
    grown &= support
    grown = largest_component(grown, structure=structure)

    if args.tight_mask:
        grown = binary_erosion(grown, structure=structure, iterations=args.tight_erode_iters)
        grown = largest_component(grown, structure=structure)

    out_mask = grown.astype(np.uint8)
    out_masked_bfc = np.where(out_mask > 0, bfc, 0.0).astype(np.float32)

    save_like(Path(args.out_mask), out_mask, bfc_img)
    save_like(Path(args.out_masked_bfc), out_masked_bfc, bfc_img)

    if args.debug_prefix:
        prefix = Path(args.debug_prefix)
        save_like(prefix.with_name(prefix.name + "_inten_norm.nii.gz"), inten_norm.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_grad_norm.nii.gz"), grad_norm.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_dist_norm.nii.gz"), dist_norm.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_seed_score.nii.gz"), seed_score.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_seed.nii.gz"), seed.astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_candidate.nii.gz"), candidate.astype(np.uint8), bfc_img)


if __name__ == "__main__":
    main()