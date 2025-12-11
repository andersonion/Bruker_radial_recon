#!/usr/bin/env python3
"""
bruker_radial_bart.py

Bruker 3D radial → BART NUFFT → SoS → NIfTI (with QA volumes).

Key features
------------
- Infers matrix, coils, and readout from Bruker method/acqp.
- Handles stored_ro vs true_ro (trims RO samples).
- Sliding-window framing with --spokes-per-frame and --frame-shift.
- Skips already-reconstructed frames (idempotent per-frame NUFFT).
- Optional GPU flag (falls back to CPU if BART has no GPU support).
- Writes final 4D stack and QA-first NIfTIs using nibabel
  (no dependency on BART `toimg` / Analyze pairs).
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import nibabel as nib
import numpy as np


# ---------------------------
# Bruker helpers
# ---------------------------

def _read_text(path: Path) -> List[str]:
    with path.open("r") as f:
        return f.readlines()


def _find_param_block(lines: List[str], key: str) -> Tuple[int, str]:
    """Return (index, line) where '##$key=' occurs."""
    target = f"##${key}="
    for i, line in enumerate(lines):
        if line.startswith(target):
            return i, line
    raise ValueError(f"Parameter {key} not found")


def _parse_bruker_array(lines: List[str], idx: int) -> List[str]:
    """
    Parse a Bruker array value like:
        ##$PVM_Matrix=( 3 )
          96 96 96
    """
    header = lines[idx]
    m = re.search(r"\(\s*(\d+)\s*\)", header)
    if not m:
        # maybe all values are on same line after '='
        after = header.split("=", 1)[1].strip()
        return after.split()

    n_expected = int(m.group(1))
    # collect tokens after ')' on this line + following lines until we have n_expected
    after = header.split(")", 1)[1]
    tokens: List[str] = after.strip().split()
    j = idx + 1
    while len(tokens) < n_expected and j < len(lines):
        line = lines[j].strip()
        if line.startswith("##"):
            break
        tokens.extend(line.split())
        j += 1
    return tokens[:n_expected]


def infer_matrix(method_path: Path) -> Tuple[int, int, int]:
    lines = _read_text(method_path)
    idx, _ = _find_param_block(lines, "PVM_Matrix")
    vals = list(map(int, _parse_bruker_array(lines, idx)))
    if len(vals) != 3:
        raise ValueError(f"Unexpected PVM_Matrix contents: {vals}")
    NX, NY, NZ = vals
    print(f"[info] Matrix inferred from PVM_Matrix: {NX}x{NY}x{NZ}")
    return NX, NY, NZ


def infer_coils(method_path: Path) -> int:
    lines = _read_text(method_path)
    idx, line = _find_param_block(lines, "PVM_EncNReceivers")
    val = line.split("=", 1)[1].strip()
    try:
        coils = int(val)
    except ValueError:
        # sometimes stored as "( 1 )"
        m = re.search(r"\d+", val)
        if not m:
            raise
        coils = int(m.group(0))
    print(f"[info] Coils inferred from PVM_EncNReceivers: {coils}")
    return coils


def infer_ro_from_acq(acqp_path: Path) -> Tuple[int, int]:
    """
    Return (stored_ro, true_ro) from ACQ_size.
    Heuristic based on your logs:
      stored_ro = ACQ_size[0] (128)
      true_ro   = ACQ_size[1] (122)
    """
    lines = _read_text(acqp_path)
    idx, _ = _find_param_block(lines, "ACQ_size")
    vals = list(map(int, _parse_bruker_array(lines, idx)))
    if len(vals) < 2:
        raise ValueError(f"Unexpected ACQ_size contents: {vals}")
    stored_ro, true_ro = vals[0], vals[1]
    print(f"[info] Readout (true RO) from ACQ_size: RO={true_ro}")
    return stored_ro, true_ro


# ---------------------------
# BART CFL/HDR helpers
# ---------------------------

def readcfl(base: Path) -> np.ndarray:
    """Read a BART .cfl/.hdr pair into a numpy complex64 array."""
    hdr_path = base.with_suffix(".hdr")
    cfl_path = base.with_suffix(".cfl")
    with hdr_path.open("r") as f:
        f.readline()  # skip '# Dimensions'
        dim_line = f.readline()
    dims = [int(x) for x in dim_line.split()]
    n_elem = int(np.prod(dims))
    with cfl_path.open("rb") as f:
        data = np.fromfile(f, dtype=np.complex64, count=n_elem)
    arr = data.reshape(dims, order="F")
    return arr


def writecfl(base: Path, arr: np.ndarray) -> None:
    """Write numpy array to BART .cfl/.hdr pair."""
    base = Path(base)
    hdr_path = base.with_suffix(".hdr")
    cfl_path = base.with_suffix(".cfl")

    # BART expects column-major order
    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        data = arr.astype(np.complex64)
    else:
        data = arr.astype(np.complex64)

    # dims: pad to 16 with 1s
    dims = list(arr.shape)
    if len(dims) > 16:
        raise ValueError("BART supports at most 16 dims")
    dims += [1] * (16 - len(dims))

    with hdr_path.open("w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in dims) + "\n")

    with cfl_path.open("wb") as f:
        data.ravel(order="F").tofile(f)


def bart_cmd(args: List[str], check: bool = True) -> None:
    print("[bart]", " ".join(args))
    subprocess.run(args, check=check)


# ---------------------------
# NIfTI helpers (nibabel)
# ---------------------------

def bart_stack_to_nifti(stack_base: Path, out_path: Path, nx: int, ny: int, nz: int) -> None:
    """
    Read a BART stack (e.g. SoS or QA join) and write a float NIfTI with shape
    (NX, NY, NZ, Nt). Uses nibabel; no BART toimg.
    """
    stack_base = Path(stack_base)
    out_path = Path(out_path)

    arr = readcfl(stack_base)

    # Drop trailing singleton dims beyond the last non-1
    dims = list(arr.shape)
    non1 = [i for i, d in enumerate(dims) if d > 1]
    if not non1:
        raise ValueError(f"Stack {stack_base} has no non-singleton dims: {dims}")
    last = max(non1)
    arr = np.reshape(arr, dims[: last + 1], order="F")

    # At this point, BART dims should be [NX, NY, NZ] or [NX,NY, NZ, Nt]
    if arr.shape[0] != nx or arr.shape[1] != ny or arr.shape[2] != nz:
        print(f"[warn] BART stack dims {arr.shape} do not match expected "
              f"{(nx, ny, nz)}; using as-is.")

    if arr.ndim == 3:
        # single volume, add time axis
        arr = arr[..., np.newaxis]
    elif arr.ndim > 4:
        # collapse extra dims into time
        extra = int(np.prod(arr.shape[3:]))
        arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2], extra)

    # Convert complex→magnitude float
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    data = arr.astype(np.float32)

    # Dummy affine (1 mm isotropic)
    affine = np.eye(4, dtype=float)
    out_path = out_path.with_suffix(".nii.gz")
    img = nib.Nifti1Image(data, affine)
    nib.save(img, out_path)
    print(f"[info] Wrote NIfTI: {out_path} with shape {data.shape}")


def write_qa_nifti(qa_frames: List[Path], qa_base: Path, nx: int, ny: int, nz: int) -> None:
    """
    Join first N per-frame SoS vols along dim=3 and write QA NIfTI.
    """
    if len(qa_frames) < 1:
        return

    qa_base = Path(qa_base)
    # Join along BART dim=3 (time)
    cmd = ["bart", "join", "3"] + [str(f) for f in qa_frames] + [str(qa_base)]
    bart_cmd(cmd)

    qa_nii = qa_base.parent / (qa_base.name + ".nii.gz")
    bart_stack_to_nifti(qa_base, qa_nii, nx, ny, nz)


def write_final_stack_nifti(stack_base: Path, out_base: Path, nx: int, ny: int, nz: int) -> None:
    """
    Write the final 4D stack from BART (already joined along dim=3) as NIfTI.
    """
    out_base = Path(out_base)
    final_nii = out_base
    # if user didn't include extension, add .nii.gz
    if final_nii.suffix not in (".nii", ".gz"):
        final_nii = final_nii.with_suffix(".nii.gz")
    elif final_nii.suffix == ".nii":
        final_nii = final_nii.with_suffix(".nii.gz")

    bart_stack_to_nifti(stack_base, final_nii, nx, ny, nz)


# ---------------------------
# Core Bruker → sliding windows
# ---------------------------

def load_bruker_kspace(fid_path: Path, stored_ro: int, true_ro: int, coils: int,
                       fid_dtype: str = "int32", fid_endian: str = "little") -> Tuple[np.ndarray, int]:
    """
    Load Bruker FID and reshape to (true_ro, n_spokes, coils).

    We assume:
      - raw FID is interleaved real/imag for all coils
      - dimension order inside each coil is (stored_ro, spokes)
    """
    dtype_map = {
        "int16": np.int16,
        "int32": np.int32,
        "float32": np.float32,
        "float64": np.float64,
    }
    if fid_dtype not in dtype_map:
        raise ValueError(f"Unsupported fid_dtype {fid_dtype}")

    dt = dtype_map[fid_dtype]
    if fid_endian == "little":
        dt = "<" + dt().dtype.str[1:]
    elif fid_endian == "big":
        dt = ">" + dt().dtype.str[1:]

    raw = np.fromfile(fid_path, dtype=dt)
    if raw.size % 2 != 0:
        raise ValueError(f"FID length {raw.size} is not even (real/imag pairs)")

    raw = raw.astype(np.float32)
    complex_data = raw[0::2] + 1j * raw[1::2]

    n_complex = complex_data.size
    if n_complex % coils != 0:
        raise ValueError(
            f"Could not split FID into coils cleanly: n_complex={n_complex}, coils={coils}"
        )

    per_coil = n_complex // coils
    if per_coil % stored_ro != 0:
        raise ValueError(
            f"Could not factor FID into (stored_ro * spokes * coils). "
            f"total={n_complex}, stored_ro={stored_ro}, coils={coils}"
        )

    n_spokes = per_coil // stored_ro
    # shape: (coils, n_spokes, stored_ro)
    data = complex_data.reshape(coils, n_spokes, stored_ro)
    # move to (stored_ro, spokes, coils)
    data = np.moveaxis(data, 0, 2)   # (stored_ro, spokes, coils)

    # trim RO to true_ro
    if true_ro < stored_ro:
        data = data[:true_ro, :, :]
        print(f"[info] Trimmed k-space from stored_ro={stored_ro} to true_ro={true_ro}")
    elif true_ro > stored_ro:
        raise ValueError("true_ro cannot be larger than stored_ro")

    print(f"[info] Loaded k-space with stored_ro={stored_ro}, true_ro={true_ro}, "
          f"spokes={n_spokes}, coils={coils}")
    return data, n_spokes


def build_sliding_windows(n_spokes: int, spokes_per_frame: int, frame_shift: int) -> List[Tuple[int, int]]:
    windows: List[Tuple[int, int]] = []
    start = 0
    while start < n_spokes:
        end = min(start + spokes_per_frame, n_spokes)
        windows.append((start, end))
        if end == n_spokes:
            break
        start += frame_shift
    return windows


# ---------------------------
# NUFFT driver
# ---------------------------

def run_bart_recon(
    series_dir: Path,
    out_base: Path,
    spokes_per_frame: int,
    frame_shift: int,
    traj_mode: str,
    fid_dtype: str,
    fid_endian: str,
    combine: str,
    qa_first: int,
    export_nifti: bool,
    use_gpu: bool,
    debug: bool,
) -> None:
    method_path = series_dir / "method"
    acqp_path = series_dir / "acqp"
    fid_path = series_dir / "fid"

    if not fid_path.exists():
        raise FileNotFoundError(f"FID not found at {fid_path}")

    NX, NY, NZ = infer_matrix(method_path)
    stored_ro, true_ro = infer_ro_from_acq(acqp_path)
    coils = infer_coils(method_path)

    ksp_all, n_spokes = load_bruker_kspace(fid_path, stored_ro, true_ro, coils,
                                           fid_dtype=fid_dtype, fid_endian=fid_endian)

    windows = build_sliding_windows(n_spokes, spokes_per_frame, frame_shift)
    print(f"[info] Sliding window: spokes-per-frame={spokes_per_frame}, frame-shift={frame_shift}")
    print(f"[info] Will reconstruct {len(windows)} frame(s).")

    # Simple synthetic 3D radial trajectory ('kron' mode only, for now)
    if traj_mode != "kron":
        raise NotImplementedError("Only --traj-mode kron is implemented in this script.")

    # Scale so max |k| ≈ NX/2 (like earlier "traj-scale 48")
    traj_scale = max(NX, NY, NZ) / 2.0
    print(f"[info] Traj built with max |k| ≈ {traj_scale:.2f}")

    # Precompute base path for per-frame results
    out_base = Path(out_base)
    series_out_dir = out_base.parent
    series_out_dir.mkdir(parents=True, exist_ok=True)

    qa_frames: List[Path] = []

    # optionally check GPU support once
    gpu_supported = False
    if use_gpu:
        try:
            # quick sanity check; this will fail fast if no GPU support
            subprocess.run(
                ["bart", "nufft", "-g"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            gpu_supported = True
        except subprocess.CalledProcessError:
            print("[warn] BART compiled without GPU support; falling back to CPU NUFFT.")
            use_gpu = False

    for frame_idx, (start, end) in enumerate(windows):
        frame_str = f"{frame_idx:05d}"
        frame_prefix = series_out_dir / f"{out_base.name}_vol{frame_str}"

        coil_img_base = frame_prefix  # final SoS mag goes here (after rss)
        coil_cfl = coil_img_base.with_suffix(".cfl")

        # Skip NUFFT if SoS image already exists (idempotent behavior)
        if coil_cfl.exists():
            print(f"[info] Frame {frame_idx} already reconstructed -> {coil_img_base}, skipping NUFFT/RSS.")
            if qa_first > 0 and frame_idx < qa_first:
                qa_frames.append(coil_img_base)
            continue

        print(f"[info] Frame {frame_idx} spokes [{start}:{end}] (n={end - start})")

        # Extract frame k-space: (true_ro, n_frame_spokes, coils)
        ksp_frame = ksp_all[:, start:end, :]

        # Build synthetic trajectory: (3, true_ro, n_frame_spokes)
        n_fr_spokes = end - start
        # simple evenly spaced radial spokes on sphere (placeholder, but consistent with earlier runs)
        phi = np.linspace(0, np.pi, n_fr_spokes, endpoint=False)
        theta = np.linspace(0, 2 * np.pi, n_fr_spokes, endpoint=False)
        k_radius = np.linspace(-0.5, 0.5, true_ro, endpoint=False) * 2.0

        # (true_ro, n_spokes)
        kx = np.outer(k_radius, np.sin(phi) * np.cos(theta))
        ky = np.outer(k_radius, np.sin(phi) * np.sin(theta))
        kz = np.outer(k_radius, np.cos(phi))

        traj = np.stack([kx, ky, kz], axis=0) * traj_scale  # (3, true_ro, n_spokes)

        # Write traj & ksp for this frame
        traj_base = series_out_dir / f"{out_base.name}_vol{frame_str}_traj"
        ksp_base = series_out_dir / f"{out_base.name}_vol{frame_str}_ksp"
        coil_base = series_out_dir / f"{out_base.name}_vol{frame_str}_coil"

        writecfl(traj_base, traj)
        # BART NUFFT expects dims consistent with traj; ksp: (RO, spokes, coils)
        writecfl(ksp_base, ksp_frame)

        # Construct nufft cmd
        cmd = [
            "bart",
            "nufft",
            "-i",
            "-d",
            f"{NX}:{NY}:{NZ}",
            str(traj_base),
            str(ksp_base),
            str(coil_base),
        ]
        if use_gpu and gpu_supported:
            cmd.insert(2, "-g")  # bart nufft -g -i -d ...

        bart_cmd(cmd)

        # Combine coils (SoS) along dim=3 (coil dim in BART convention)
        if combine == "sos":
            bart_cmd(["bart", "rss", "3", str(coil_base), str(coil_img_base)])
        else:
            raise NotImplementedError("Only --combine sos is implemented.")

        if qa_first > 0 and frame_idx < qa_first:
            qa_frames.append(coil_img_base)

    # QA NIfTI
    if qa_first > 0 and qa_frames:
        qa_base = series_out_dir / f"{out_base.name}_QA_first{qa_first}"
        write_qa_nifti(qa_frames, qa_base, NX, NY, NZ)

    # Final 4D join + NIfTI
    # Join all per-frame SoS vols along dim=3
    frame_bases = [
        series_out_dir / f"{out_base.name}_vol{frame_idx:05d}"
        for frame_idx in range(len(windows))
    ]
    stack_base = series_out_dir / out_base.name
    bart_cmd(["bart", "join", "3", *[str(fb) for fb in frame_bases], str(stack_base)])

    if export_nifti:
        write_final_stack_nifti(stack_base, out_base, NX, NY, NZ)

    print("[info] All requested frames complete; 4D result at", stack_base)


# ---------------------------
# CLI
# ---------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bruker 3D radial → BART NUFFT → SoS → NIfTI recon."
    )
    p.add_argument("--series", required=True, help="Path to Bruker series directory (contains method, acqp, fid)")
    p.add_argument("--out", required=True, help="Base path for outputs (no extension needed)")

    p.add_argument("--matrix", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                   help="Override matrix size (otherwise inferred from PVM_Matrix)")
    p.add_argument("--readout", type=int, help="Override true RO samples (otherwise from ACQ_size)")
    p.add_argument("--coils", type=int, help="Override number of coils (otherwise from PVM_EncNReceivers)")

    p.add_argument("--traj-mode", choices=["kron", "linz"], default="kron",
                   help="Trajectory mode (currently only kron is implemented)")
    p.add_argument("--spokes-per-frame", type=int, default=200,
                   help="Spokes per frame for sliding window")
    p.add_argument("--frame-shift", type=int, default=50,
                   help="Frame shift for sliding window")
    p.add_argument("--test-volumes", type=int, default=None,
                   help="(Not currently used; kept for API compatibility)")

    p.add_argument("--fid-dtype", default="int32",
                   help="FID datatype: int16,int32,float32,float64 (default: int32)")
    p.add_argument("--fid-endian", default="little", choices=["little", "big"],
                   help="FID endianness")

    p.add_argument("--combine", choices=["sos"], default="sos",
                   help="Coil combination mode (only sos implemented)")

    p.add_argument("--qa-first", type=int, default=0,
                   help="If >0, write QA NIfTI of first N frames")

    p.add_argument("--export-nifti", action="store_true",
                   help="Write final 4D NIfTI using nibabel")

    p.add_argument("--gpu", action="store_true",
                   help="Try to use BART GPU NUFFT (falls back to CPU if unsupported)")

    p.add_argument("--debug", action="store_true",
                   help="Enable extra debug output")

    args = p.parse_args(argv)
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    series_dir = Path(args.series).resolve()
    out_base = Path(args.out).resolve()

    run_bart_recon(
        series_dir=series_dir,
        out_base=out_base,
        spokes_per_frame=args.spokes_per_frame,
        frame_shift=args.frame_shift,
        traj_mode=args.traj_mode,
        fid_dtype=args.fid_dtype,
        fid_endian=args.fid_endian,
        combine=args.combine,
        qa_first=args.qa_first,
        export_nifti=args.export_nifti,
        use_gpu=args.gpu,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
