#!/usr/bin/env python3
"""
radial_grid_recommender.py

Given a Bruker series folder, parse acqp (and method if available) to:
- Extract FOV (mm), readout points Nsamp, and #projections Nproj (if possible).
- Estimate kmax ~ Nsamp / (2 * FOV_iso)   [units: 1/mm]
- Compute delta_x_min = 1 / (2*kmax) = FOV_iso / Nsamp
- Recommend an isotropic cubic grid size N for 3D radial MRI.
- Check angular sampling sufficiency: Nproj_min ≈ (π/2) * (kmax * FOV_iso)^2 = (π/8) * Nsamp^2
- Print memory estimates.

Notes:
- For Bruker ParaVision, FOV is usually in acqp as ACQ_fov (mm).
- Nsamp is typically ACQ_size[0] for radial (readout samples per spoke).
- Nproj is often in the method file as PVM_NPro (or similar). We'll look for several keys.
- If Nproj can't be found, we'll still compute the other quantities and warn.

Author: you :)
"""

import argparse
import math
import os
import re
import sys
from typing import Any, List, Optional, Tuple

# ---------- Bruker file parsing helpers ----------

BRUKER_KEY_RE = re.compile(r"##\$\s*([A-Za-z0-9_]+)\s*=\s*(.*)")

def _clean(s: str) -> str:
    return s.strip().replace('\r', '')

def _parse_bruker_multiline_value(first_line_val: str, lines_iter) -> str:
    """
    Handles values that may start on the next lines (e.g., arrays).
    Bruker arrays often look like:
      ##$ACQ_fov=( 3 )
      40 40 40
    or:
      ##$ACQ_size=( 3 ) 256 1 1
    We merge immediate following lines until we hit another '##$' or end.
    """
    val = first_line_val.rstrip()
    # If value looks complete on this line, return it
    if "##$" in val:
        return val.split("##$")[0].strip()

    collected = [val]
    # lines_iter is a generator; we need to peek—so we rely on caller feeding us lines in order
    return val

def read_bruker_file(path: str) -> dict:
    """
    Read a Bruker 'acqp' or 'method' file into a {key: raw_value_string} dict.
    Tries to capture multi-line arrays.
    """
    if not os.path.isfile(path):
        return {}
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()

    params = {}
    i = 0
    while i < len(lines):
        line = _clean(lines[i])
        m = BRUKER_KEY_RE.match(line)
        if m:
            key = m.group(1)
            rhs = m.group(2).strip()

            # If RHS starts with '(' it may be an array descriptor; the data might be same or next line(s).
            # We'll capture the immediate RHS and, if next line doesn't start with '##$', we append it.
            # Also handle values that wrap to next line.
            payload = rhs
            # accumulate following lines that don't start a new key and aren't empty comments
            j = i + 1
            tails = []
            while j < len(lines):
                nxt = _clean(lines[j])
                if nxt.startswith("##$"):
                    break
                # Stop at comments/owner lines or blanks only if we've already captured something
                # but safer to include numeric lines.
                if nxt != "":
                    tails.append(nxt)
                j += 1
            if tails:
                payload = " ".join([payload] + tails).strip()
            params[key] = payload
            i = j
        else:
            i += 1
    return params

def parse_numeric_list(val: str) -> List[float]:
    # Remove Bruker array descriptor like "( 3 )"
    val = re.sub(r"\(\s*\d+\s*\)", " ", val)
    # Collapse multiple spaces
    toks = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)
    return [float(t) for t in toks]

def parse_int_list(val: str) -> List[int]:
    val = re.sub(r"\(\s*\d+\s*\)", " ", val)
    toks = re.findall(r"[-+]?\d+", val)
    return [int(t) for t in toks]

def try_get_float(params: dict, key: str) -> Optional[float]:
    if key in params:
        nums = parse_numeric_list(params[key])
        if nums:
            return float(nums[0])
    return None

def try_get_float_list(params: dict, key: str) -> Optional[List[float]]:
    if key in params:
        nums = parse_numeric_list(params[key])
        return nums if nums else None
    return None

def try_get_int_list(params: dict, key: str) -> Optional[List[int]]:
    if key in params:
        nums = parse_int_list(params[key])
        return nums if nums else None
    return None

# ---------- Parameter extraction logic ----------

def extract_fov_mm(acqp: dict, method: dict) -> Optional[Tuple[float,float,float]]:
    """
    Prefer ACQ_fov from acqp; fallback to PVM_Fov from method (mm).
    Returns (fov_x, fov_y, fov_z) in mm if possible.
    """
    for key in ["ACQ_fov", "ACQ_fov_cm", "ACQ_fov_mm"]:
        if key in acqp:
            vals = try_get_float_list(acqp, key)
            if vals:
                # ACQ_fov is in mm in most PV versions. If 'ACQ_fov_cm', convert.
                if key == "ACQ_fov_cm":
                    vals = [v * 10.0 for v in vals]  # cm -> mm
                # Make sure it has 3 values for 3D; if 2, pad
                if len(vals) == 2: vals.append(vals[-1])
                return tuple(vals[:3])

    # fallback to method PVM_Fov
    if "PVM_Fov" in method:
        vals = try_get_float_list(method, "PVM_Fov")
        if vals:
            if len(vals) == 2: vals.append(vals[-1])
            return tuple(vals[:3])

    return None

def extract_nsamp(acqp: dict, method: dict) -> Optional[int]:
    """
    Try ACQ_size[0] first; fallback to PVM_EncMatrix or PVM_Matrix[0].
    """
    for key in ["ACQ_size", "ACQ_spatial_size"]:
        if key in acqp:
            arr = try_get_int_list(acqp, key)
            if arr and len(arr) >= 1:
                return int(arr[0])

    # fallbacks from method
    for key in ["PVM_EncMatrix", "PVM_Matrix"]:
        if key in method:
            arr = try_get_int_list(method, key)
            if arr and len(arr) >= 1:
                return int(arr[0])

    return None

def extract_nproj(acqp: dict, method: dict) -> Optional[int]:
    """
    Search common keys for number of projections/spokes.
    """
    candidate_keys = [
        "PVM_NPro", "PVM_EncNPro", "PVM_RadialSpokes", "PVM_Projections",
        "ACQ_n_pro", "ACQ_n_projections", "NPRO", "NProj", "NPro"
    ]
    for key in candidate_keys:
        if key in method:
            val = try_get_int_list(method, key)
            if val:
                return int(val[0])
        if key in acqp:
            val = try_get_int_list(acqp, key)
            if val:
                return int(val[0])
    # Some protocols encode projections as total repetitions * something:
    # Try ACQ_n_images as a very rough fallback (often wrong for multi-echo etc.)
    for key in ["ACQ_n_images", "NR", "NI"]:
        if key in acqp:
            val = try_get_int_list(acqp, key)
            if val:
                return int(val[0])

    return None

# ---------- Core math ----------

def recommend_grid(fov_mm_iso: float,
                   nsamp: int,
                   nproj: Optional[int],
                   voxel_mm: Optional[float],
                   oversamp: float,
                   round_to: int) -> dict:
    """
    Compute kmax, delta_x_min, recommended voxel size & N, angular check, memory estimates.
    """
    # kmax estimate from FOV + Nsamp:
    # Δk ~ 1/FOV, kmax ~ (Nsamp/2)*Δk = Nsamp / (2*FOV)
    kmax = nsamp / (2.0 * fov_mm_iso)  # 1/mm

    delta_x_min = 1.0 / (2.0 * kmax)  # mm  == fov/nsamp

    # choose voxel size
    if voxel_mm is None:
        voxel = delta_x_min  # no super-res beyond data
        voxel_src = "auto (Δx_min)"
    else:
        voxel = max(voxel_mm, delta_x_min)  # clamp
        voxel_src = f"user ({voxel_mm:.6g} mm) clamped to ≥ Δx_min"

    # target FOV with oversampling
    target_fov = fov_mm_iso * oversamp

    # N (pre-round)
    N_raw = math.ceil(target_fov / voxel)

    # round N to a friendly size
    if round_to > 1:
        N = int(math.ceil(N_raw / round_to) * round_to)
    else:
        N = int(N_raw)

    # Angular sufficiency
    # Nproj_min for alias-free to kmax at FOV_iso: ≈ (π/2)*(kmax*FOV)^2 = (π/8)*Nsamp^2
    nproj_min = (math.pi / 8.0) * (nsamp ** 2)
    angular_ok = None
    if nproj is not None:
        angular_ok = (nproj >= nproj_min)

    # memory estimates (complex64)
    bytes_per_complex64 = 8  # 2 * 4 bytes
    vol_bytes = N**3 * bytes_per_complex64
    # working set estimate for iterative recon (3–5x)
    work_lo = 3 * vol_bytes
    work_hi = 5 * vol_bytes

    def fmt_gb(x):
        return x / (1024**3)

    return dict(
        fov_mm_iso=target_fov,
        fov_source="ACQ_fov" if oversamp == 1.0 else f"ACQ_fov × oversamp ({oversamp}x)",
        nsamp=nsamp,
        nproj=nproj,
        kmax_per_mm=kmax,
        delta_x_min_mm=delta_x_min,
        voxel_mm=voxel,
        voxel_source=voxel_src,
        N_recommended=N,
        N_unrounded=N_raw,
        round_to=round_to,
        nproj_min_required=int(math.ceil(nproj_min)),
        angular_ok=angular_ok,
        mem_volume_gb=fmt_gb(vol_bytes),
        mem_working_set_gb=(fmt_gb(work_lo), fmt_gb(work_hi)),
    )

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Recommend 3D radial grid size (N) from Bruker acqp/method.")
    ap.add_argument("series_dir", help="Bruker series folder containing 'acqp' (and optionally 'method').")
    ap.add_argument("--voxel-mm", type=float, default=None,
                    help="Desired isotropic voxel size in mm. If smaller than Δx_min, it will be clamped.")
    ap.add_argument("--oversamp", type=float, default=1.0,
                    help="FOV oversampling factor (e.g., 1.1–1.5 for padding). Default=1.0.")
    ap.add_argument("--round", dest="round_to", type=int, default=16,
                    help="Round the recommended N up to a multiple of this. Use 1 to disable. Default=16.")
    ap.add_argument("--verbose", action="store_true", help="Print extra parsing details.")
    args = ap.parse_args()

    acqp_path = os.path.join(args.series_dir, "acqp")
    method_path = os.path.join(args.series_dir, "method")

    acqp = read_bruker_file(acqp_path)
    method = read_bruker_file(method_path) if os.path.isfile(method_path) else {}

    if args.verbose:
        print(f"[info] Parsed acqp keys: {len(acqp)} | method keys: {len(method)}", file=sys.stderr)

    fov_xyz = extract_fov_mm(acqp, method)
    nsamp = extract_nsamp(acqp, method)
    nproj = extract_nproj(acqp, method)

    if fov_xyz is None:
        print("ERROR: Could not find FOV in acqp/method (looked for ACQ_fov or PVM_Fov).", file=sys.stderr)
        sys.exit(2)
    if nsamp is None:
        print("ERROR: Could not determine readout samples (Nsamp) from ACQ_size/PVM_EncMatrix/PVM_Matrix.", file=sys.stderr)
        sys.exit(3)

    # For radial, assume isotropic FOV (use the minimum axis to be safe)
    fov_iso = float(min(fov_xyz))
    if args.verbose:
        print(f"[info] FOV_xyz (mm): {fov_xyz} → using isotropic FOV_iso={fov_iso:.6g} mm (min axis)", file=sys.stderr)
        print(f"[info] Nsamp (readout points per spoke): {nsamp}", file=sys.stderr)
        if nproj is not None:
            print(f"[info] Nproj (spokes): {nproj}", file=sys.stderr)
        else:
            print("[warn] Could not determine Nproj (spokes). Angular check will be limited.", file=sys.stderr)

    out = recommend_grid(
        fov_mm_iso=fov_iso,
        nsamp=nsamp,
        nproj=nproj,
        voxel_mm=args.voxel_mm,
        oversamp=args.oversamp,
        round_to=args.round_to,
    )

    print("\n=== 3D Radial Grid Recommendation ===")
    print(f"FOV_iso (effective):      {out['fov_mm_iso']:.6g} mm   [{out['fov_source']}]")
    print(f"Nsamp (per spoke):         {out['nsamp']}")
    print(f"Nproj (spokes):            {out['nproj'] if out['nproj'] is not None else 'unknown'}")
    print(f"kmax:                      {out['kmax_per_mm']:.6g} 1/mm")
    print(f"Δx_min (no super-res):     {out['delta_x_min_mm']:.6g} mm")
    print(f"Chosen voxel:              {out['voxel_mm']:.6g} mm   [{out['voxel_source']}]")
    print(f"Recommended N (rounded):   {out['N_recommended']}  (unrounded {out['N_unrounded']})  [round to multiple of {out['round_to']}]")

    print("\n--- Angular sampling check ---")
    print(f"Nproj required (min):      {out['nproj_min_required']}  (≈ π/8 · Nsamp^2)")
    if out['angular_ok'] is None:
        print("Angular sufficiency:       unknown (Nproj not found)")
    else:
        print(f"Angular sufficiency:       {'OK' if out['angular_ok'] else 'INSUFFICIENT'}")

    print("\n--- Memory estimates (complex64) ---")
    print(f"Volume only:               {out['mem_volume_gb']:.3f} GB")
    lo, hi = out['mem_working_set_gb']
    print(f"Working set (iterative):   {lo:.3f}–{hi:.3f} GB (≈3–5× volume)")

    if out['angular_ok'] is False:
        print("\n[Advice] Angular sampling appears insufficient for alias-free FOV at this Nsamp.")
        print("         Options: reduce FOV (crop), reduce N (larger voxels), or acquire more projections.")

if __name__ == "__main__":
    main()
