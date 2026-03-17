#!/usr/bin/env python3

import argparse
import json
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


def boundary_seed(mask: np.ndarray) -> np.ndarray:
    seed = np.zeros_like(mask, dtype=bool)
    seed[0, :, :] |= mask[0, :, :]
    seed[-1, :, :] |= mask[-1, :, :]
    seed[:, 0, :] |= mask[:, 0, :]
    seed[:, -1, :] |= mask[:, -1, :]
    seed[:, :, 0] |= mask[:, :, 0]
    seed[:, :, -1] |= mask[:, :, -1]
    return seed


def bbox_extents_mm(mask: np.ndarray, zooms_xyz):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ext_vox = maxs - mins + 1
    ext_mm = ext_vox.astype(float) * np.asarray(zooms_xyz, dtype=float)
    return ext_mm


def mask_volume_mm3(mask: np.ndarray, voxel_volume_mm3: float) -> float:
    return float(np.count_nonzero(mask) * voxel_volume_mm3)


def parse_float_list(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def flood_outside(open_space: np.ndarray, structure):
    seeds = boundary_seed(open_space)
    return binary_propagation(seeds, structure=structure, mask=open_space)


def build_shell_candidate(
    grad_norm: np.ndarray,
    support: np.ndarray,
    structure,
    grad_thr: float,
    min_shell_vox: int,
    shell_close_iters: int,
    shell_dilate_iters: int,
):
    shell_raw = (grad_norm >= grad_thr) & support
    shell = remove_small_components(shell_raw, min_shell_vox, structure=structure)

    if np.any(shell) and shell_close_iters > 0:
        shell = binary_closing(shell, structure=structure, iterations=shell_close_iters)
    if np.any(shell) and shell_dilate_iters > 0:
        shell = binary_dilation(shell, structure=structure, iterations=shell_dilate_iters)

    shell = remove_small_components(shell, min_shell_vox, structure=structure)
    shell = largest_component(shell, structure=structure)
    return shell_raw, shell


def shell_is_plausible(
    shell: np.ndarray,
    voxel_volume_mm3: float,
    zooms_xyz,
    shell_volume_min_mm3: float,
    shell_volume_max_mm3: float,
    extent_x_min_mm: float,
    extent_x_max_mm: float,
    extent_y_min_mm: float,
    extent_y_max_mm: float,
    extent_z_min_mm: float,
    extent_z_max_mm: float,
):
    if not np.any(shell):
        return False, {"reason": "empty_shell"}

    vol_mm3 = mask_volume_mm3(shell, voxel_volume_mm3)
    ext_mm = bbox_extents_mm(shell, zooms_xyz)
    if ext_mm is None:
        return False, {"reason": "no_extents"}

    ok = (
        shell_volume_min_mm3 <= vol_mm3 <= shell_volume_max_mm3
        and extent_x_min_mm <= ext_mm[0] <= extent_x_max_mm
        and extent_y_min_mm <= ext_mm[1] <= extent_y_max_mm
        and extent_z_min_mm <= ext_mm[2] <= extent_z_max_mm
    )

    info = {
        "shell_volume_mm3": vol_mm3,
        "shell_extent_x_mm": float(ext_mm[0]),
        "shell_extent_y_mm": float(ext_mm[1]),
        "shell_extent_z_mm": float(ext_mm[2]),
    }
    return ok, info


def build_brain_from_shell_and_moat(
    inten_norm: np.ndarray,
    grad_norm: np.ndarray,
    support: np.ndarray,
    shell: np.ndarray,
    structure,
    moat_thr: float,
    shell_inner_band_mm: float,
    shell_fill_close_iters: int,
    shell_fill_open_iters: int,
    final_dilate_iters: int,
    final_erode_iters: int,
    shell_gate_grad_max: float,
):
    zoom_guess = 1.0  # shell_inner_band_mm already converted before call in practice if needed
    dist_to_shell = distance_transform_edt(~shell)

    # Narrow band just inside/around the shell where the moat is expected
    shell_band = support & (dist_to_shell <= shell_inner_band_mm / max(zoom_guess, 1e-6))

    moat_raw = shell_band & (inten_norm <= moat_thr)
    moat = largest_component(remove_small_components(moat_raw, 5, structure=structure), structure=structure)

    # Remove dark moat from shell neighborhood, keep the shell itself as the enclosing cue
    barrier = shell | moat

    open_space = support & (~barrier)
    outside = flood_outside(open_space, structure=structure)
    enclosed = open_space & (~outside)
    enclosed = largest_component(enclosed, structure=structure)

    if not np.any(enclosed):
        return None

    brain = binary_fill_holes(enclosed)
    brain = largest_component(brain, structure=structure)

    if shell_fill_close_iters > 0:
        brain = binary_closing(brain, structure=structure, iterations=shell_fill_close_iters)
    if shell_fill_open_iters > 0:
        brain = binary_opening(brain, structure=structure, iterations=shell_fill_open_iters)

    if final_dilate_iters > 0:
        brain = binary_dilation(brain, structure=structure, iterations=final_dilate_iters)
    if final_erode_iters > 0:
        brain = binary_erosion(brain, structure=structure, iterations=final_erode_iters)

    brain &= support
    brain = largest_component(brain, structure=structure)
    brain = binary_fill_holes(brain)
    brain = largest_component(brain, structure=structure)

    # soft shell-contact metric: brain should sit just inside shell, not way outside
    shell_contact = int(np.count_nonzero(binary_dilation(brain, structure=structure, iterations=1) & shell))
    moat_overlap = int(np.count_nonzero(brain & moat))
    shell_grad_good = int(np.count_nonzero(shell & (grad_norm <= shell_gate_grad_max)))

    return {
        "brain": brain,
        "moat_raw": moat_raw,
        "moat": moat,
        "barrier": barrier,
        "open_space": open_space,
        "outside": outside,
        "enclosed": enclosed,
        "shell_contact": shell_contact,
        "moat_overlap": moat_overlap,
        "shell_grad_good": shell_grad_good,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Shell-first mouse brain masking from BFC T2 using gradient shell sweep, dark moat sweep, and inward fill."
    )
    ap.add_argument("--bfc", required=True)
    ap.add_argument("--support_mask", required=True)
    ap.add_argument("--out_mask", required=True)
    ap.add_argument("--out_masked_bfc", required=True)
    ap.add_argument("--debug_prefix", default=None)

    ap.add_argument("--smooth_sigma", type=float, default=1.0)
    ap.add_argument("--grad_sigma", type=float, default=1.0)

    # shell sweep
    ap.add_argument("--grad_thresholds", default="0.08,0.10,0.12,0.14,0.16")
    ap.add_argument("--shell_close_iters", type=int, default=1)
    ap.add_argument("--shell_dilate_iters", type=int, default=0)

    # shell plausibility by physical size / extent
    ap.add_argument("--shell_volume_min_mm3", type=float, default=15.0)
    ap.add_argument("--shell_volume_max_mm3", type=float, default=220.0)
    ap.add_argument("--extent_x_min_mm", type=float, default=7.0)
    ap.add_argument("--extent_x_max_mm", type=float, default=22.0)
    ap.add_argument("--extent_y_min_mm", type=float, default=7.0)
    ap.add_argument("--extent_y_max_mm", type=float, default=22.0)
    ap.add_argument("--extent_z_min_mm", type=float, default=7.0)
    ap.add_argument("--extent_z_max_mm", type=float, default=30.0)

    # dark moat sweep
    ap.add_argument("--moat_thresholds", default="0.06,0.08,0.10,0.12,0.14,0.16")

    # inward fill / cleanup
    ap.add_argument("--shell_inner_band_mm", type=float, default=1.5)
    ap.add_argument("--shell_fill_close_iters", type=int, default=2)
    ap.add_argument("--shell_fill_open_iters", type=int, default=0)
    ap.add_argument("--final_dilate_iters", type=int, default=0)
    ap.add_argument("--final_erode_iters", type=int, default=0)

    # candidate plausibility
    ap.add_argument("--brain_volume_hard_min_mm3", type=float, default=300.0)
    ap.add_argument("--brain_volume_hard_max_mm3", type=float, default=650.0)
    ap.add_argument("--brain_volume_preferred_min_mm3", type=float, default=380.0)
    ap.add_argument("--brain_volume_preferred_max_mm3", type=float, default=550.0)

    # extra soft scoring
    ap.add_argument("--shell_gate_grad_max", type=float, default=0.35)

    # optional tight mask
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

    structure = generate_binary_structure(3, 2)

    bfc_smooth = gaussian_filter(bfc, sigma=args.smooth_sigma)
    inten_norm = robust_normalize(bfc_smooth, support, low_pct=2.0, high_pct=98.0)

    grad = gaussian_gradient_magnitude(bfc_smooth, sigma=args.grad_sigma)
    grad_norm = robust_normalize(grad, support, low_pct=2.0, high_pct=98.0)

    min_shell_vox = max(1, int(round(args.shell_volume_min_mm3 / voxel_volume_mm3)))

    grad_thresholds = parse_float_list(args.grad_thresholds)
    moat_thresholds = parse_float_list(args.moat_thresholds)

    candidates = []
    shell_candidates = []

    for grad_thr in grad_thresholds:
        shell_raw, shell = build_shell_candidate(
            grad_norm=grad_norm,
            support=support,
            structure=structure,
            grad_thr=grad_thr,
            min_shell_vox=min_shell_vox,
            shell_close_iters=args.shell_close_iters,
            shell_dilate_iters=args.shell_dilate_iters,
        )

        ok_shell, shell_info = shell_is_plausible(
            shell=shell,
            voxel_volume_mm3=voxel_volume_mm3,
            zooms_xyz=zooms,
            shell_volume_min_mm3=args.shell_volume_min_mm3,
            shell_volume_max_mm3=args.shell_volume_max_mm3,
            extent_x_min_mm=args.extent_x_min_mm,
            extent_x_max_mm=args.extent_x_max_mm,
            extent_y_min_mm=args.extent_y_min_mm,
            extent_y_max_mm=args.extent_y_max_mm,
            extent_z_min_mm=args.extent_z_min_mm,
            extent_z_max_mm=args.extent_z_max_mm,
        )

        shell_candidates.append({
            "grad_threshold": grad_thr,
            "ok_shell": ok_shell,
            **shell_info,
        })

        if not ok_shell:
            continue

        for moat_thr in moat_thresholds:
            built = build_brain_from_shell_and_moat(
                inten_norm=inten_norm,
                grad_norm=grad_norm,
                support=support,
                shell=shell,
                structure=structure,
                moat_thr=moat_thr,
                shell_inner_band_mm=args.shell_inner_band_mm,
                shell_fill_close_iters=args.shell_fill_close_iters,
                shell_fill_open_iters=args.shell_fill_open_iters,
                final_dilate_iters=args.final_dilate_iters,
                final_erode_iters=args.final_erode_iters,
                shell_gate_grad_max=args.shell_gate_grad_max,
            )
            if built is None:
                continue

            brain = built["brain"]
            brain_vol = mask_volume_mm3(brain, voxel_volume_mm3)

            if not (args.brain_volume_hard_min_mm3 <= brain_vol <= args.brain_volume_hard_max_mm3):
                continue

            preferred_mid = 0.5 * (args.brain_volume_preferred_min_mm3 + args.brain_volume_preferred_max_mm3)
            volume_penalty = abs(brain_vol - preferred_mid)

            score = (
                4.0 * built["shell_contact"]
                - 10.0 * built["moat_overlap"]
                - 1.0 * volume_penalty
                + 0.25 * built["shell_grad_good"]
            )

            candidates.append({
                "grad_threshold": grad_thr,
                "moat_threshold": moat_thr,
                "brain_volume_mm3": brain_vol,
                "score": float(score),
                "shell_info": shell_info,
                "brain": brain,
                "shell_raw": shell_raw,
                "shell": shell,
                **built,
            })

    if not candidates:
        if args.debug_prefix:
            prefix = Path(args.debug_prefix)
            save_like(prefix.with_name(prefix.name + "_inten_norm.nii.gz"), inten_norm.astype(np.float32), bfc_img)
            save_like(prefix.with_name(prefix.name + "_grad_norm.nii.gz"), grad_norm.astype(np.float32), bfc_img)
            with open(prefix.with_name(prefix.name + "_selection.json"), "w") as f:
                json.dump(
                    {
                        "status": "failed",
                        "reason": "No plausible shell+moat+brain candidate satisfied hard constraints.",
                        "shell_candidates": shell_candidates,
                        "brain_volume_hard_min_mm3": args.brain_volume_hard_min_mm3,
                        "brain_volume_hard_max_mm3": args.brain_volume_hard_max_mm3,
                        "brain_volume_preferred_min_mm3": args.brain_volume_preferred_min_mm3,
                        "brain_volume_preferred_max_mm3": args.brain_volume_preferred_max_mm3,
                    },
                    f,
                    indent=2,
                )
        raise RuntimeError("No plausible shell+moat+brain candidate satisfied hard constraints.")

    preferred = [
        c for c in candidates
        if args.brain_volume_preferred_min_mm3 <= c["brain_volume_mm3"] <= args.brain_volume_preferred_max_mm3
    ]
    chosen = max(preferred if preferred else candidates, key=lambda c: c["score"])

    brain = chosen["brain"]

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

        save_like(prefix.with_name(prefix.name + "_inten_norm.nii.gz"), inten_norm.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_grad_norm.nii.gz"), grad_norm.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_shell_raw.nii.gz"), chosen["shell_raw"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_shell.nii.gz"), chosen["shell"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_moat_raw.nii.gz"), chosen["moat_raw"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_moat.nii.gz"), chosen["moat"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_barrier.nii.gz"), chosen["barrier"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_open_space.nii.gz"), chosen["open_space"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_outside.nii.gz"), chosen["outside"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_enclosed.nii.gz"), chosen["enclosed"].astype(np.uint8), bfc_img)

        with open(prefix.with_name(prefix.name + "_selection.json"), "w") as f:
            json.dump(
                {
                    "status": "ok",
                    "selected_grad_threshold": chosen["grad_threshold"],
                    "selected_moat_threshold": chosen["moat_threshold"],
                    "selected_brain_volume_mm3": chosen["brain_volume_mm3"],
                    "selected_score": chosen["score"],
                    "selected_shell_info": chosen["shell_info"],
                    "shell_candidates": shell_candidates,
                    "all_candidates": [
                        {
                            "grad_threshold": c["grad_threshold"],
                            "moat_threshold": c["moat_threshold"],
                            "brain_volume_mm3": c["brain_volume_mm3"],
                            "score": c["score"],
                            "shell_contact": c["shell_contact"],
                            "moat_overlap": c["moat_overlap"],
                            "shell_info": c["shell_info"],
                        }
                        for c in candidates
                    ],
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()