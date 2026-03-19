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
    return lab == counts.argmax()


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


def parse_float_list(text: str):
    return [float(x) for x in text.split(",") if x.strip()]


def mm3_to_vox(mm3: float, voxel_volume_mm3: float) -> int:
    return max(1, int(round(mm3 / max(voxel_volume_mm3, 1e-8))))


def mm_to_vox(mm: float, spacing_mm: float) -> float:
    return mm / max(spacing_mm, 1e-8)


def mask_volume_mm3(mask: np.ndarray, voxel_volume_mm3: float) -> float:
    return float(np.count_nonzero(mask) * voxel_volume_mm3)


def bbox_extents_mm(mask: np.ndarray, zooms_xyz) -> np.ndarray | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ext_vox = maxs - mins + 1
    return ext_vox.astype(float) * np.asarray(zooms_xyz, dtype=float)


def bbox_fill_fraction(mask: np.ndarray) -> float:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return 0.0
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ext_vox = maxs - mins + 1
    bbox_vox = int(np.prod(ext_vox))
    if bbox_vox <= 0:
        return 0.0
    return float(np.count_nonzero(mask) / bbox_vox)


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    return obj


def cavity_is_plausible(
    cavity: np.ndarray,
    voxel_volume_mm3: float,
    zooms_xyz,
    cavity_volume_min_mm3: float,
    cavity_volume_max_mm3: float,
    extent_x_min_mm: float,
    extent_x_max_mm: float,
    extent_y_min_mm: float,
    extent_y_max_mm: float,
    extent_z_min_mm: float,
    extent_z_max_mm: float,
    cavity_bbox_fill_frac_min: float,
):
    if not np.any(cavity):
        return False, {"reason": "empty_cavity"}

    vol_mm3 = mask_volume_mm3(cavity, voxel_volume_mm3)
    ext_mm = bbox_extents_mm(cavity, zooms_xyz)
    fill_frac = bbox_fill_fraction(cavity)

    if ext_mm is None:
        return False, {"reason": "no_bbox"}

    ok = (
        cavity_volume_min_mm3 <= vol_mm3 <= cavity_volume_max_mm3
        and extent_x_min_mm <= ext_mm[0] <= extent_x_max_mm
        and extent_y_min_mm <= ext_mm[1] <= extent_y_max_mm
        and extent_z_min_mm <= ext_mm[2] <= extent_z_max_mm
        and fill_frac >= cavity_bbox_fill_frac_min
    )

    info = {
        "cavity_volume_mm3": float(vol_mm3),
        "cavity_extent_x_mm": float(ext_mm[0]),
        "cavity_extent_y_mm": float(ext_mm[1]),
        "cavity_extent_z_mm": float(ext_mm[2]),
        "cavity_bbox_fill_fraction": float(fill_frac),
    }
    return bool(ok), info


def make_outer_rim_mask(support: np.ndarray, dist_to_support_edge_vox: np.ndarray, outer_rim_vox: float) -> np.ndarray:
    return support & (dist_to_support_edge_vox <= outer_rim_vox)


def build_shell_candidate(
    grad_norm: np.ndarray,
    support: np.ndarray,
    outer_rim: np.ndarray,
    structure,
    grad_thr: float,
    shell_min_vox: int,
    shell_close_iters: int,
    shell_dilate_iters: int,
    shell_open_iters: int,
):
    shell_raw = (grad_norm >= grad_thr) & outer_rim
    shell = remove_small_components(shell_raw, shell_min_vox, structure=structure)

    if np.any(shell) and shell_close_iters > 0:
        shell = binary_closing(shell, structure=structure, iterations=shell_close_iters)

    if np.any(shell) and shell_dilate_iters > 0:
        shell = binary_dilation(shell, structure=structure, iterations=shell_dilate_iters)

    if np.any(shell) and shell_open_iters > 0:
        shell = binary_opening(shell, structure=structure, iterations=shell_open_iters)

    shell = remove_small_components(shell, shell_min_vox, structure=structure)

    # Intentionally do not largest-component the shell.
    return shell_raw, shell


def choose_balloon_seed(
    support: np.ndarray,
    shell: np.ndarray,
    structure,
):
    allowed = support & (~shell)
    if not np.any(allowed):
        return np.zeros_like(allowed, dtype=bool), np.zeros_like(allowed, dtype=np.float32)

    dist = distance_transform_edt(allowed)
    maxd = float(dist.max())
    if maxd <= 0:
        return np.zeros_like(allowed, dtype=bool), dist.astype(np.float32)

    seed = allowed & (dist >= 0.90 * maxd)
    seed = largest_component(seed, structure=structure)
    return seed, dist.astype(np.float32)


def balloon_grow(
    seed: np.ndarray,
    allowed: np.ndarray,
    structure,
    max_iters: int,
):
    grown = seed.copy()
    for _ in range(max_iters):
        nxt = binary_dilation(grown, structure=structure) & allowed
        if np.array_equal(nxt, grown):
            break
        grown = nxt
    return grown


def build_cavity_from_shell_balloon(
    support: np.ndarray,
    shell: np.ndarray,
    structure,
    balloon_close_iters: int,
    balloon_fill_holes: bool,
    balloon_max_iters: int,
):
    seed, seed_dist = choose_balloon_seed(support, shell, structure)

    if not np.any(seed):
        return {
            "seed": seed,
            "seed_dist": seed_dist,
            "allowed": support & (~shell),
            "cavity_raw": np.zeros_like(support, dtype=bool),
            "cavity": np.zeros_like(support, dtype=bool),
        }

    allowed = support & (~shell)
    cavity_raw = balloon_grow(seed, allowed, structure, balloon_max_iters)
    cavity = cavity_raw.copy()

    if balloon_close_iters > 0:
        cavity = binary_closing(cavity, structure=structure, iterations=balloon_close_iters)

    if balloon_fill_holes:
        cavity = binary_fill_holes(cavity)

    cavity &= support
    cavity = largest_component(cavity, structure=structure)

    return {
        "seed": seed,
        "seed_dist": seed_dist,
        "allowed": allowed,
        "cavity_raw": cavity_raw,
        "cavity": cavity,
    }


def refine_with_moat(
    inten_norm: np.ndarray,
    grad_norm: np.ndarray,
    support: np.ndarray,
    shell: np.ndarray,
    cavity: np.ndarray,
    zooms_xyz,
    voxel_volume_mm3: float,
    structure,
    moat_thr: float,
    shell_inner_band_mm: float,
    moat_min_volume_mm3: float,
    moat_close_iters: int,
    barrier_close_iters: int,
    brain_close_iters: int,
    brain_open_iters: int,
    brain_dilate_iters: int,
    shell_gate_grad_max: float,
):
    mean_spacing = float(np.mean(zooms_xyz))
    shell_inner_band_vox = mm_to_vox(shell_inner_band_mm, mean_spacing)
    moat_min_vox = mm3_to_vox(moat_min_volume_mm3, voxel_volume_mm3)

    dist_to_shell_vox = distance_transform_edt(~shell)
    shell_inner_band = support & (dist_to_shell_vox <= shell_inner_band_vox)

    moat_raw = shell_inner_band & (inten_norm <= moat_thr)
    moat = remove_small_components(moat_raw, moat_min_vox, structure=structure)

    if np.any(moat) and moat_close_iters > 0:
        moat = binary_closing(moat, structure=structure, iterations=moat_close_iters)

    barrier = shell | moat

    if np.any(barrier) and barrier_close_iters > 0:
        barrier = binary_closing(barrier, structure=structure, iterations=barrier_close_iters)

    allowed = support & (~barrier)

    seed, seed_dist = choose_balloon_seed(allowed, np.zeros_like(allowed, dtype=bool), structure)
    if not np.any(seed):
        return None

    brain = balloon_grow(seed, allowed, structure, max_iters=10000)
    brain = largest_component(brain, structure=structure)

    if brain_close_iters > 0:
        brain = binary_closing(brain, structure=structure, iterations=brain_close_iters)
    if brain_open_iters > 0:
        brain = binary_opening(brain, structure=structure, iterations=brain_open_iters)
    if brain_dilate_iters > 0:
        brain = binary_dilation(brain, structure=structure, iterations=brain_dilate_iters)

    brain &= support
    brain = largest_component(brain, structure=structure)
    brain = binary_fill_holes(brain)
    brain = largest_component(brain, structure=structure)

    shell_contact = int(np.count_nonzero(binary_dilation(brain, structure=structure, iterations=1) & shell))
    moat_overlap = int(np.count_nonzero(brain & moat))
    shell_grad_good = int(np.count_nonzero(shell & (grad_norm <= shell_gate_grad_max)))

    return {
        "moat_raw": moat_raw,
        "moat": moat,
        "barrier": barrier,
        "allowed_post_moat": allowed,
        "refill_seed": seed,
        "refill_seed_dist": seed_dist,
        "brain": brain,
        "shell_contact": shell_contact,
        "moat_overlap": moat_overlap,
        "shell_grad_good": shell_grad_good,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Gradient-shell + balloon-fill mouse brain masking."
    )
    ap.add_argument("--bfc", required=True)
    ap.add_argument("--support_mask", required=True)
    ap.add_argument("--out_mask", required=True)
    ap.add_argument("--out_masked_bfc", required=True)
    ap.add_argument("--debug_prefix", default=None)

    ap.add_argument("--smooth_sigma", type=float, default=1.0)
    ap.add_argument("--grad_sigma", type=float, default=1.0)

    # shell extraction
    ap.add_argument("--grad_thresholds", default="0.10,0.12,0.14,0.16,0.18,0.20")
    ap.add_argument("--outer_rim_mm", type=float, default=1.25)
    ap.add_argument("--shell_close_iters", type=int, default=2)
    ap.add_argument("--shell_dilate_iters", type=int, default=1)
    ap.add_argument("--shell_open_iters", type=int, default=0)
    ap.add_argument("--shell_min_volume_mm3", type=float, default=2.0)

    # cavity plausibility
    ap.add_argument("--cavity_volume_min_mm3", type=float, default=250.0)
    ap.add_argument("--cavity_volume_max_mm3", type=float, default=800.0)
    ap.add_argument("--extent_x_min_mm", type=float, default=7.0)
    ap.add_argument("--extent_x_max_mm", type=float, default=22.0)
    ap.add_argument("--extent_y_min_mm", type=float, default=7.0)
    ap.add_argument("--extent_y_max_mm", type=float, default=22.0)
    ap.add_argument("--extent_z_min_mm", type=float, default=4.0)
    ap.add_argument("--extent_z_max_mm", type=float, default=30.0)
    ap.add_argument("--cavity_bbox_fill_frac_min", type=float, default=0.35)

    # balloon fill
    ap.add_argument("--balloon_close_iters", type=int, default=2)
    ap.add_argument("--balloon_fill_holes", action="store_true")
    ap.add_argument("--balloon_max_iters", type=int, default=10000)

    # moat refinement
    ap.add_argument("--moat_thresholds", default="0.04,0.06,0.08,0.10")
    ap.add_argument("--shell_inner_band_mm", type=float, default=1.25)
    ap.add_argument("--moat_min_volume_mm3", type=float, default=1.0)
    ap.add_argument("--moat_close_iters", type=int, default=1)
    ap.add_argument("--barrier_close_iters", type=int, default=1)

    # final cleanup and selection
    ap.add_argument("--brain_close_iters", type=int, default=2)
    ap.add_argument("--brain_open_iters", type=int, default=0)
    ap.add_argument("--brain_dilate_iters", type=int, default=0)

    ap.add_argument("--brain_volume_hard_min_mm3", type=float, default=300.0)
    ap.add_argument("--brain_volume_hard_max_mm3", type=float, default=650.0)
    ap.add_argument("--brain_volume_preferred_min_mm3", type=float, default=380.0)
    ap.add_argument("--brain_volume_preferred_max_mm3", type=float, default=550.0)

    ap.add_argument("--shell_gate_grad_max", type=float, default=0.35)

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
    mean_spacing = float(np.mean(zooms))

    structure = generate_binary_structure(3, 2)

    bfc_smooth = gaussian_filter(bfc, sigma=args.smooth_sigma)
    inten_norm = robust_normalize(bfc_smooth, support, low_pct=2.0, high_pct=98.0)

    grad = gaussian_gradient_magnitude(bfc_smooth, sigma=args.grad_sigma)
    grad_norm = robust_normalize(grad, support, low_pct=2.0, high_pct=98.0)

    dist_to_support_edge_vox = distance_transform_edt(support)
    outer_rim_vox = mm_to_vox(args.outer_rim_mm, mean_spacing)
    shell_min_vox = mm3_to_vox(args.shell_min_volume_mm3, voxel_volume_mm3)

    grad_thresholds = parse_float_list(args.grad_thresholds)
    moat_thresholds = parse_float_list(args.moat_thresholds)

    shell_candidates = []
    candidates = []

    for grad_thr in grad_thresholds:
        outer_rim = make_outer_rim_mask(support, dist_to_support_edge_vox, outer_rim_vox)

        shell_raw, shell = build_shell_candidate(
            grad_norm=grad_norm,
            support=support,
            outer_rim=outer_rim,
            structure=structure,
            grad_thr=grad_thr,
            shell_min_vox=shell_min_vox,
            shell_close_iters=args.shell_close_iters,
            shell_dilate_iters=args.shell_dilate_iters,
            shell_open_iters=args.shell_open_iters,
        )

        cav_built = build_cavity_from_shell_balloon(
            support=support,
            shell=shell,
            structure=structure,
            balloon_close_iters=args.balloon_close_iters,
            balloon_fill_holes=args.balloon_fill_holes,
            balloon_max_iters=args.balloon_max_iters,
        )

        cavity = cav_built["cavity"]

        ok_cavity, cavity_info = cavity_is_plausible(
            cavity=cavity,
            voxel_volume_mm3=voxel_volume_mm3,
            zooms_xyz=zooms,
            cavity_volume_min_mm3=args.cavity_volume_min_mm3,
            cavity_volume_max_mm3=args.cavity_volume_max_mm3,
            extent_x_min_mm=args.extent_x_min_mm,
            extent_x_max_mm=args.extent_x_max_mm,
            extent_y_min_mm=args.extent_y_min_mm,
            extent_y_max_mm=args.extent_y_max_mm,
            extent_z_min_mm=args.extent_z_min_mm,
            extent_z_max_mm=args.extent_z_max_mm,
            cavity_bbox_fill_frac_min=args.cavity_bbox_fill_frac_min,
        )

        shell_candidates.append({
            "grad_threshold": float(grad_thr),
            "ok_cavity": bool(ok_cavity),
            **cavity_info,
        })

        if not ok_cavity:
            continue

        for moat_thr in moat_thresholds:
            refined = refine_with_moat(
                inten_norm=inten_norm,
                grad_norm=grad_norm,
                support=support,
                shell=shell,
                cavity=cavity,
                zooms_xyz=zooms,
                voxel_volume_mm3=voxel_volume_mm3,
                structure=structure,
                moat_thr=moat_thr,
                shell_inner_band_mm=args.shell_inner_band_mm,
                moat_min_volume_mm3=args.moat_min_volume_mm3,
                moat_close_iters=args.moat_close_iters,
                barrier_close_iters=args.barrier_close_iters,
                brain_close_iters=args.brain_close_iters,
                brain_open_iters=args.brain_open_iters,
                brain_dilate_iters=args.brain_dilate_iters,
                shell_gate_grad_max=args.shell_gate_grad_max,
            )
            if refined is None:
                continue

            brain = refined["brain"]
            brain_vol = mask_volume_mm3(brain, voxel_volume_mm3)

            if not (args.brain_volume_hard_min_mm3 <= brain_vol <= args.brain_volume_hard_max_mm3):
                continue

            preferred_mid = 0.5 * (
                args.brain_volume_preferred_min_mm3 + args.brain_volume_preferred_max_mm3
            )
            volume_penalty = abs(brain_vol - preferred_mid)

            score = (
                4.0 * refined["shell_contact"]
                - 8.0 * refined["moat_overlap"]
                - 1.0 * volume_penalty
                + 0.15 * refined["shell_grad_good"]
            )

            candidates.append({
                "grad_threshold": float(grad_thr),
                "moat_threshold": float(moat_thr),
                "brain_volume_mm3": float(brain_vol),
                "score": float(score),
                "cavity_info": cavity_info,
                "outer_rim": outer_rim,
                "shell_raw": shell_raw,
                "shell": shell,
                **cav_built,
                **refined,
            })

    if not candidates:
        if args.debug_prefix:
            prefix = Path(args.debug_prefix)
            save_like(prefix.with_name(prefix.name + "_inten_norm.nii.gz"), inten_norm.astype(np.float32), bfc_img)
            save_like(prefix.with_name(prefix.name + "_grad_norm.nii.gz"), grad_norm.astype(np.float32), bfc_img)
            save_like(prefix.with_name(prefix.name + "_dist_to_support_edge_vox.nii.gz"), dist_to_support_edge_vox.astype(np.float32), bfc_img)
            with open(prefix.with_name(prefix.name + "_selection.json"), "w") as f:
                json.dump(
                    json_safe(
                        {
                            "status": "failed",
                            "reason": "No plausible cavity+moat+brain candidate satisfied hard constraints.",
                            "shell_candidates": shell_candidates,
                            "brain_volume_hard_min_mm3": args.brain_volume_hard_min_mm3,
                            "brain_volume_hard_max_mm3": args.brain_volume_hard_max_mm3,
                            "brain_volume_preferred_min_mm3": args.brain_volume_preferred_min_mm3,
                            "brain_volume_preferred_max_mm3": args.brain_volume_preferred_max_mm3,
                        }
                    ),
                    f,
                    indent=2,
                )
        raise RuntimeError("No plausible cavity+moat+brain candidate satisfied hard constraints.")

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
        save_like(prefix.with_name(prefix.name + "_dist_to_support_edge_vox.nii.gz"), dist_to_support_edge_vox.astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_outer_rim.nii.gz"), chosen["outer_rim"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_shell_raw.nii.gz"), chosen["shell_raw"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_shell.nii.gz"), chosen["shell"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_balloon_seed.nii.gz"), chosen["seed"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_balloon_seed_dist.nii.gz"), chosen["seed_dist"].astype(np.float32), bfc_img)
        save_like(prefix.with_name(prefix.name + "_balloon_allowed.nii.gz"), chosen["allowed"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_cavity_raw.nii.gz"), chosen["cavity_raw"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_cavity.nii.gz"), chosen["cavity"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_moat_raw.nii.gz"), chosen["moat_raw"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_moat.nii.gz"), chosen["moat"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_barrier.nii.gz"), chosen["barrier"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_allowed_post_moat.nii.gz"), chosen["allowed_post_moat"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_refill_seed.nii.gz"), chosen["refill_seed"].astype(np.uint8), bfc_img)
        save_like(prefix.with_name(prefix.name + "_refill_seed_dist.nii.gz"), chosen["refill_seed_dist"].astype(np.float32), bfc_img)

        with open(prefix.with_name(prefix.name + "_selection.json"), "w") as f:
            json.dump(
                json_safe(
                    {
                        "status": "ok",
                        "selected_grad_threshold": chosen["grad_threshold"],
                        "selected_moat_threshold": chosen["moat_threshold"],
                        "selected_brain_volume_mm3": chosen["brain_volume_mm3"],
                        "selected_score": chosen["score"],
                        "selected_cavity_info": chosen["cavity_info"],
                        "shell_candidates": shell_candidates,
                        "all_candidates": [
                            {
                                "grad_threshold": c["grad_threshold"],
                                "moat_threshold": c["moat_threshold"],
                                "brain_volume_mm3": c["brain_volume_mm3"],
                                "score": c["score"],
                                "shell_contact": c["shell_contact"],
                                "moat_overlap": c["moat_overlap"],
                                "cavity_info": c["cavity_info"],
                            }
                            for c in candidates
                        ],
                    }
                ),
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()