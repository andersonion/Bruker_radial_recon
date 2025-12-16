#!/usr/bin/env python3
"""
Bruker 3D radial -> BART NUFFT recon driver.

- Reads Bruker method/acqp/fid
- Factors FID into (stored_ro, spokes, coils) using your working heuristic
- Builds synthetic 3D "kron" radial trajectory
- Runs BART NUFFT + SoS per-frame with sliding window
- Skips already-reconstructed frames (based on SoS .cfl presence)
- Writes:
    * QA NIfTI of first N frames  (via nibabel)
    * Final 4D NIfTI stack       (via nibabel)
- Prints BART image dims for first coil & SoS volumes for debugging

Assumes:
    bart is on PATH as "bart"
"""

import argparse
import os
import sys
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import nibabel as nib


# ---------------- Bruker helpers ---------------- #


def read_bruker_param(path: Path, key: str, default=None):
    """
    Tiny Bruker text parser for method/acqp.

    Returns:
      - scalar (int/float/str) if a single value
      - list if multiple
      - default if not found
    """
    if not path.exists():
        return default

    token = f"##${key}="

    try:
        with path.open("r") as f:
            lines = f.readlines()
    except Exception:
        return default

    for i, line in enumerate(lines):
        if line.startswith(token):
            parts = line.strip().split("=")
            if len(parts) < 2:
                continue

            rhs = parts[1].strip()

            # Multi-line / array style
            if rhs.startswith("("):
                vals = []
                j = i + 1
                while j < len(lines):
                    l2 = lines[j].strip()
                    if l2.startswith("##"):
                        break
                    if l2 == "" or l2.startswith("$$"):
                        j += 1
                        continue
                    vals.extend(l2.split())
                    j += 1

                parsed = []
                for v in vals:
                    try:
                        if "." in v or "e" in v.lower():
                            parsed.append(float(v))
                        else:
                            parsed.append(int(v))
                    except ValueError:
                        parsed.append(v)
                if len(parsed) == 1:
                    return parsed[0]
                return parsed

            # Single-line values or tuples on one line
            rhs = rhs.strip().strip("()")
            tokens = [t for t in rhs.split() if t]

            parsed = []
            for v in tokens:
                try:
                    if "." in v or "e" in v.lower():
                        parsed.append(float(v))
                    else:
                        parsed.append(int(v))
                except ValueError:
                    parsed.append(v)

            if len(parsed) == 1:
                return parsed[0]
            return parsed

    return default


def infer_matrix(method: Path):
    mat = read_bruker_param(method, "PVM_Matrix", default=None)
    if mat is None:
        raise ValueError("Could not infer PVM_Matrix")
    if isinstance(mat, (int, float)) or len(mat) != 3:
        raise ValueError(f"Unexpected PVM_Matrix: {mat}")
    NX, NY, NZ = map(int, mat)
    return NX, NY, NZ


def infer_true_ro(acqp: Path) -> int:
    acq_size = read_bruker_param(acqp, "ACQ_size", default=None)
    if acq_size is None:
        raise ValueError("Could not infer ACQ_size for true RO")

    if isinstance(acq_size, (int, float)):
        ro = int(acq_size)
    else:
        ro = int(acq_size[0])

    return ro


def infer_coils(method: Path) -> int:
    coils = read_bruker_param(method, "PVM_EncNReceivers", default=None)
    if coils is None:
        coils = 1
    if isinstance(coils, (list, tuple)):
        coils = int(coils[0])
    return int(coils)


def factor_fid(total_points: int, true_ro: int, coils_hint: int | None = None):
    """
    Factor total complex points into (stored_ro, spokes, coils).

    This is your previous "good" heuristic: we try several candidate stored_ro
    values and coil counts and pick a reasonable combination, preferring:
      - stored_ro >= true_ro
      - stored_ro close to true_ro
      - spokes not ridiculously small
      - matching the coil hint if possible.
    """
    block_candidates = [true_ro]
    for b in (128, 96, 64, 384, 512):
        if b != true_ro:
            block_candidates.append(b)

    best = None
    best_score = -1

    if coils_hint is not None:
        coil_candidates = [coils_hint] + [c for c in range(1, 33) if c != coils_hint]
    else:
        coil_candidates = list(range(1, 33))

    for stored_ro in block_candidates:
        for c in coil_candidates:
            denom = stored_ro * c
            if total_points % denom != 0:
                continue
            spokes = total_points // denom

            score = 0
            if stored_ro >= true_ro:
                score += 1
            if abs(stored_ro - true_ro) <= 10:
                score += 1
            if spokes > 100:
                score += 1
            if coils_hint is not None and c == coils_hint:
                score += 1

            if score > best_score:
                best_score = score
                best = (stored_ro, c, spokes)

    if best is None:
        raise ValueError(
            f"Could not factor FID into (stored_ro * spokes * coils). "
            f"total={total_points}, true_ro={true_ro}, coils_hint={coils_hint}"
        )

    return best


def load_bruker_kspace(
    fid_path: Path,
    true_ro: int,
    coils_hint: int | None = None,
    endian: str = "<",
    base_kind: str = "i4",
):
    """
    Load Bruker FID as complex data and reshape to (stored_ro, spokes, coils),
    then trim to (true_ro, spokes, coils).

    endian: '>' or '<'
    base_kind: 'i4' (int32) or 'f4' (float32)
    """
    if not fid_path.exists():
        raise FileNotFoundError(f"FID not found: {fid_path}")

    try:
        np_dtype = np.dtype(endian + base_kind)
    except TypeError as e:
        raise ValueError(
            f"Invalid dtype combination: endian={endian}, kind={base_kind}"
        ) from e

    raw = np.fromfile(fid_path, dtype=np_dtype).astype(np.float32)

    if raw.size % 2 != 0:
        raise ValueError(f"FID raw length {raw.size} not even (real/imag pairs).")

    complex_data = raw[0::2] + 1j * raw[1::2]
    total_points = complex_data.size

    stored_ro, coils, spokes = factor_fid(total_points, true_ro, coils_hint)

    if coils_hint is not None and coils != coils_hint:
        print(
            f"[warn] FID factorization suggests coils={coils}, "
            f"but header said {coils_hint}; using coils={coils}.",
            file=sys.stderr,
        )

    print(
        f"[info] Loaded k-space with stored_ro={stored_ro}, "
        f"true_ro={true_ro}, spokes={spokes}, coils={coils}"
    )

    ksp = complex_data.reshape(stored_ro, spokes, coils)

    if stored_ro != true_ro:
        if stored_ro < true_ro:
            raise ValueError(
                f"stored_ro={stored_ro} < true_ro={true_ro}, cannot trim."
            )
        ksp = ksp[:true_ro, :, :]
        print(f"[info] Trimmed k-space from stored_ro={stored_ro} to true_ro={true_ro}")

    return ksp, stored_ro, spokes, coils


# ---------------- BART CFL helpers ---------------- #


def writecfl(name: str, arr: np.ndarray):
    """
    Write BART .cfl/.hdr with dims following arr.shape in Fortran order.
    """
    base = Path(name)
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")

    arr_f = np.asfortranarray(arr.astype(np.complex64))

    with hdr.open("w") as f:
        f.write("# Dimensions\n")
        dims = list(arr_f.shape)
        dims += [1] * (16 - len(dims))
        f.write(" ".join(str(d) for d in dims) + "\n")

    with cfl.open("wb") as f:
        stacked = np.empty(arr_f.size * 2, dtype=np.float32)
        stacked[0::2] = arr_f.real.ravel(order="F")
        stacked[1::2] = arr_f.imag.ravel(order="F")
        stacked.tofile(f)


def readcfl(name: str) -> np.ndarray:
    """
    Read BART .cfl/.hdr into a numpy complex64 array (Fortran order-aware).
    """
    base = Path(name)
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")

    if not hdr.exists() or not cfl.exists():
        raise FileNotFoundError(f"CFL/HDR not found for base {base}")

    with hdr.open("r") as f:
        line = f.readline()  # "# Dimensions\n"
        if not line.startswith("#"):
            raise ValueError(f"Malformed BART hdr for {base}")
        dims_line = f.readline().strip()
        dims = [int(x) for x in dims_line.split()]

    ndim = 0
    for d in dims:
        if d > 1 or ndim == 0:
            ndim += 1
        else:
            break
    dims = dims[:ndim]

    data = np.fromfile(cfl, dtype=np.float32)
    if data.size % 2 != 0:
        raise ValueError(f"CFL data length {data.size} not even.")

    cplx = data[0::2] + 1j * data[1::2]
    arr = cplx.reshape(dims, order="F")
    return arr


def bart_image_dims(bart_bin: str, base: Path) -> list[int] | None:
    """
    Query the full 16-D BART shape for a CFL base using:
        bart show -d <dim> <file>

    Returns a 16-element list (dims 0..15), or None if it fails.

    This FIXES the previous bug where we called:
        bart show -d <file>
    which makes BART try to parse the filename as an integer dimension.
    """
    dims: list[int] = []
    for d in range(16):
        try:
            proc = subprocess.run(
                [bart_bin, "show", "-d", str(d), str(base)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            print(f"[warn] Failed to run 'bart show -d {d}': {e}", file=sys.stderr)
            return None

        if proc.returncode != 0:
            print(
                f"[warn] 'bart show -d {d}' nonzero exit ({proc.returncode}): "
                f"{proc.stderr.strip()}",
                file=sys.stderr,
            )
            return None

        # stdout is typically just an integer with optional whitespace/newline
        s = proc.stdout.strip()
        try:
            dims.append(int(s))
        except ValueError:
            print(
                f"[warn] Unexpected output from 'bart show -d {d}': {s!r}",
                file=sys.stderr,
            )
            return None

    return dims


def bart_supports_gpu(bart_bin: str = "bart") -> bool:
    """
    Quick probe: is 'bart nufft -g' a known option, and compiled with GPU support?
    """
    try:
        proc = subprocess.run(
            [bart_bin, "nufft", "-i", "-g"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return False

    if proc.returncode == 0:
        return True

    stderr = proc.stderr.lower()

    if "compiled without gpu support" in stderr:
        return False
    if "unknown option" in stderr:
        return False

    return False


# ---------------- Trajectory builder ---------------- #


def build_kron_traj(
    true_ro: int,
    spokes: int,
    NX: int,
    NY: int,
    NZ: int,
    traj_scale: float | None = None,
) -> np.ndarray:
    """
    Build a synthetic 3D "kron" golden-angle trajectory for BART.

    Output shape: (3, RO, spokes)
    """
    idx = np.arange(spokes, dtype=np.float64)

    if spokes > 1:
        z = 2.0 * (idx / (spokes - 1)) - 1.0
    else:
        z = np.zeros_like(idx)

    phi = np.pi * (1 + 5**0.5) * idx  # golden-ish

    r_xy = np.sqrt(np.clip(1.0 - z**2, 0.0, 1.0))
    dx = r_xy * np.cos(phi)
    dy = r_xy * np.sin(phi)
    dz = z

    base_s = np.linspace(-0.5, 0.5, true_ro, dtype=np.float64) * NX
    scale = float(traj_scale) if traj_scale is not None else 1.0
    s = base_s * scale

    traj = np.zeros((3, true_ro, spokes), dtype=np.complex64)
    for i in range(spokes):
        traj[0, :, i] = s * dx[i]
        traj[1, :, i] = s * dy[i]
        traj[2, :, i] = s * dz[i]

    max_rad = np.abs(traj).max()
    print(f"[info] Traj built with max |k| ≈ {max_rad:.2f}")

    return traj


# ---------------- NIfTI writers (via nibabel) ---------------- #


def bart_cfl_to_nifti(
    base: Path,
    out_nii_gz: Path,
    assume_abs: bool = True,
    extra_axes_to_end: bool = True,
):
    """
    Read a BART CFL and write a NIfTI via nibabel.
    """
    arr = readcfl(str(base))

    if assume_abs:
        arr = np.abs(arr)

    if extra_axes_to_end and arr.ndim > 1:
        leading_singletons = []
        other_axes = []
        for ax, size in enumerate(arr.shape):
            if ax == 0 and size == 1:
                leading_singletons.append(ax)
            else:
                other_axes.append(ax)
        if leading_singletons:
            order = other_axes + leading_singletons
            arr = np.transpose(arr, axes=order)

    img = nib.Nifti1Image(arr.astype(np.float32), np.eye(4))
    nib.save(img, str(out_nii_gz))


def write_qa_nifti(
    qa_frames: list[Path],
    qa_base: Path,
):
    """
    Stack first N SoS frames into a QA NIfTI using nibabel.
    """
    qa_base.parent.mkdir(parents=True, exist_ok=True)

    vols = []
    shape0 = None
    for p in qa_frames:
        arr = readcfl(str(p))
        mag = np.abs(arr)
        if shape0 is None:
            shape0 = mag.shape
        elif mag.shape != shape0:
            raise ValueError(
                f"QA frame {p} has shape {mag.shape}, expected {shape0}"
            )
        vols.append(mag)

    qa_stack = np.stack(vols, axis=-1)

    qa_img = nib.Nifti1Image(qa_stack.astype(np.float
