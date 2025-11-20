#!/usr/bin/env python3
"""
bruker_radial_bart.py

Bruker 3D radial reconstruction using BART (adjoint NUFFT) with sliding-window
temporal binning and synthetic golden-angle / Kronecker trajectories.

Behavior:

- Load Bruker FID, infer RO / Spokes / Coils, trim padded RO blocks.
- Treat acquisition as one continuous stream of spokes
  (reps * echoes * averages * slices collapsed).
- Temporal binning by spokes:
    --spokes-per-frame N
    --frame-shift M  (sliding window; default = N for non-overlap)
    --test-volumes K (limit to first K frames)
- Trajectory:
    * If $series/traj exists: currently NOT implemented (hard error).
    * Else: build synthetic 3D radial trajectory via --traj-mode kron|linear_z
      using GA/Kronecker math from the Bruker sequence code you provided.
- DCF:
    * Optional pipe-style DCF via NumPy: --dcf pipe:N (N iterations).
- Reconstruction per frame:
    * BART adjoint NUFFT (non-cart):
        traj dims: [3,1,1,1,1,1,1,1,1,1,RO,Sp,1,1,1,1]
        ksp  dims: [1,1,1,COILS,1,1,1,1,1,1,RO,Sp,1,1,1,1]
      then rss 8 for SoS.
- Output:
    * All frames stacked into a single 4D NIfTI: (NX, NY, NZ, T) at --out
    * One QC 3D NIfTI for a chosen frame: --qc-frame (1-based index)
"""

from __future__ import annotations
import argparse
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Iterable

import numpy as np

DEBUG = False
BART_GPU_AVAILABLE: Optional[bool] = None  # sticky cache


def dbg(*a):
    if DEBUG:
        print("[debug]", *a)


# ---------- Basic Bruker header parsing ----------

def _read_text_kv(path: Path) -> Dict[str, str]:
    """Very lightweight parser for Bruker-style method/acqp files."""
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    key = None
    vals: List[str] = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("##$"):
                if key is not None:
                    out[key] = " ".join(vals).strip()
                parts = line[3:].split("=", 1)
                key = parts[0].strip()
                vals = [parts[1].strip()] if len(parts) == 2 else [""]
            elif line.startswith("##"):
                if key is not None:
                    out[key] = " ".join(vals).strip()
                    key = None
                    vals = []
            elif line.startswith("$$"):
                continue
            else:
                if key is not None:
                    vals.append(line.strip())
    if key is not None:
        out[key] = " ".join(vals).strip()
    return out


def _get_int(hdr: Dict[str, str], key: str) -> Optional[int]:
    if key not in hdr:
        return None
    m = re.search(r"[-+]?\d+", hdr[key])
    return int(m.group(0)) if m else None


def _get_float(hdr: Dict[str, str], key: str) -> Optional[float]:
    if key not in hdr:
        return None
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", hdr[key])
    return float(m.group(0)) if m else None


# ---------- CFL I/O ----------

def _write_hdr(path: Path, dims: List[int]):
    """Write a valid BART .hdr file with 16 dims."""
    dims16 = [int(d) for d in dims] + [1] * (16 - len(dims))
    with open(path, "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in dims16) + "\n")


def _write_cfl(base: Path, arr: np.ndarray, dims: Optional[List[int]] = None):
    """
    Write BART .cfl/.hdr pair from a numpy array (complex64).

    Fixed to avoid the "last axis must be contiguous" error by explicitly
    flattening in Fortran order before viewing as float32.
    """
    base = Path(base)
    cfl = base.with_suffix(".cfl")
    hdr = base.with_suffix(".hdr")

    a = np.asarray(arr, dtype=np.complex64, order="F")
    flat = a.ravel(order="F").view(np.float32)

    if dims is None:
        shape = list(a.shape)
        dims = shape + [1] * (16 - len(shape))
    else:
        dims = [int(d) for d in dims] + [1] * (16 - len(dims))

    if np.prod(dims) != a.size:
        raise ValueError(f"dims product {np.prod(dims)} != array size {a.size}")

    _write_hdr(hdr, dims)
    flat.tofile(cfl)


def read_cfl(base: Path) -> np.ndarray:
    """Read a BART .cfl/.hdr pair into a numpy array."""
    base = Path(base)
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")
    if not hdr.exists() or not cfl.exists():
        raise FileNotFoundError(f"Missing cfl/hdr for {base}")
    with open(hdr, "r") as f:
        lines = f.readlines()
    if len(lines) < 2 or not lines[0].startswith("#"):
        raise ValueError("Invalid BART header")
    dims = [int(x) for x in lines[1].split()]
    n = int(np.prod(dims))
    data = np.fromfile(cfl, dtype=np.float32, count=2 * n)
    if data.size != 2 * n:
        raise ValueError("Unexpected cfl size")
    cpx = data.view(np.complex64)
    return cpx.reshape(dims, order="F")


# ---------- Bruker FID loader ----------

def _extras_factor(method: Dict[str, str], acqp: Dict[str, str]) -> int:
    """Product of repetition-like dims (reps * echoes * averages * slices)."""
    extras_keys = [
        ("NECHOES", acqp),
        ("ACQ_n_echo_images", acqp),
        ("PVM_NEchoImages", method),
        ("PVM_NRepetitions", method),
        ("NR", acqp),
        ("PVM_NAverages", method),
        ("NA", acqp),
        ("NSLICES", acqp),
        ("PVM_SPackArrNSlices", method),
    ]
    vals: List[int] = []
    for key, src in extras_keys:
        v = _get_int(src, key)
        if v is not None and v > 1:
            vals.append(v)
    factor = 1
    for v in vals:
        factor *= v
    return factor if factor > 0 else 1


def _pick_block_and_spokes(per_coil_total: int,
                           readout_hint: Optional[int],
                           spokes_hint: Optional[int]) -> Tuple[int, int]:
    """Heuristic factoring: returns (stored_ro, spokes_per_extras_collapsed)."""
    if spokes_hint and spokes_hint > 0 and per_coil_total % spokes_hint == 0:
        return per_coil_total // spokes_hint, spokes_hint

    BLOCKS = [
        128, 160, 192, 200, 224, 240, 256, 288, 320, 352, 384, 400, 416, 420,
        432, 448, 480, 496, 512, 544, 560, 576, 608, 640, 672, 704, 736, 768,
        800, 832, 896, 960, 992, 1024, 1152, 1280, 1536, 2048,
    ]

    if readout_hint and per_coil_total % readout_hint == 0:
        return readout_hint, per_coil_total // readout_hint

    for b in [x for x in BLOCKS if (not readout_hint or x >= readout_hint)]:
        if per_coil_total % b == 0:
            return b, per_coil_total // b

    # generic fallback: search factors around sqrt
    s = int(round(per_coil_total ** 0.5))
    for d in range(0, s + 1):
        for cand in (s + d, s - d):
            if cand > 0 and per_coil_total % cand == 0:
                return cand, per_coil_total // cand

    raise ValueError("Could not factor per_coil_total into (stored_ro, spokes).")


def load_bruker_kspace(
    series_dir: Path,
    matrix_ro_hint: Optional[int] = None,
    spokes: Optional[int] = None,
    readout: Optional[int] = None,
    coils: Optional[int] = None,
    fid_dtype: Optional[str] = None,
    fid_endian: Optional[str] = None,
) -> np.ndarray:
    """
    Load Bruker k-space as array (RO, Spokes, Coils).

    - Prefers existing ksp.cfl/hdr or ksp.npy if present.
    - Otherwise reads FID, infers RO/spokes/coils, trims padded RO blocks.
    """
    series_dir = Path(series_dir)
    dbg("series_dir:", series_dir)

    # 1) Existing k-space files
    ksp_base = series_dir / "ksp"
    if ksp_base.with_suffix(".cfl").exists() and ksp_base.with_suffix(".hdr").exists():
        arr = read_cfl(ksp_base)
        if arr.ndim != 3:
            raise ValueError("ksp.cfl must be 3D (RO,Spokes,Coils)")
        dbg("loaded ksp.cfl:", arr.shape)
        return arr
    npy = series_dir / "ksp.npy"
    if npy.exists():
        arr = np.load(npy)
        if arr.ndim != 3:
            raise ValueError("ksp.npy must be 3D (RO,Spokes,Coils)")
        dbg("loaded ksp.npy:", arr.shape)
        return arr

    # 2) FID
    fid_path = series_dir / "fid"
    if not fid_path.exists():
        raise FileNotFoundError("No k-space found (no fid, ksp.cfl, or ksp.npy)")

    method = _read_text_kv(series_dir / "method")
    acqp = _read_text_kv(series_dir / "acqp")

    # Endianness and word size
    if fid_endian is None:
        fid_endian = "little"
    if "BYTORDA" in acqp and "big" in acqp["BYTORDA"].lower():
        fid_endian = "big"

    if fid_dtype is None:
        fid_dtype = "int32"
    if "ACQ_word_size" in acqp:
        if "16" in acqp["ACQ_word_size"]:
            fid_dtype = "int16"
        elif "32" in acqp["ACQ_word_size"]:
            fid_dtype = "int32"

    dtype_map = {"int16": np.int16, "int32": np.int32, "float32": np.float32, "float64": np.float64}
    if fid_dtype not in dtype_map:
        raise ValueError("--fid-dtype must be one of int16,int32,float32,float64")
    dt = dtype_map[fid_dtype]

    raw = np.fromfile(fid_path, dtype=dt)
    if fid_endian == "big":
        raw = raw.byteswap().newbyteorder()
    if raw.size % 2 != 0:
        raw = raw[:-1]
    cpx = raw.astype(np.float32).view(np.complex64)
    total = cpx.size
    dbg("total complex samples:", total, "(dtype:", fid_dtype, ", endian:", fid_endian, ")")

    # RO hint: prefer explicit readout, else matrix hint
    if readout is None and matrix_ro_hint is not None:
        readout = matrix_ro_hint
    dbg("readout (matrix hint):", readout)

    # coils
    if coils is None:
        nrec = _get_int(method, "PVM_EncNReceivers")
        if nrec and nrec > 0:
            coils = nrec
    if coils is None or coils <= 0:
        coils = 1
    dbg("coils (initial):", coils)

    # extras (reps * echoes * averages * slices)
    other_dims = _extras_factor(method, acqp)
    dbg("other_dims factor:", other_dims)

    denom = coils * max(1, other_dims)
    if total % denom != 0:
        dbg("total not divisible by coils*other_dims; relaxing extras to 1")
        other_dims = 1
        denom = coils
        if total % denom != 0:
            dbg("still not divisible; relaxing coils->1")
            coils = 1
            denom = coils
            if total % denom != 0:
                raise ValueError("Cannot factor FID length with any (coils, other_dims) combo.")
    per_coil_total = total // denom
    dbg("per_coil_total (stored_ro*spokes_per_extras):", per_coil_total)

    stored_ro, spokes_per_extras = _pick_block_and_spokes(per_coil_total, readout, spokes)
    dbg("stored_ro (block):", stored_ro, " spokes_per_extras:", spokes_per_extras)

    spokes_final = spokes_per_extras * max(1, other_dims)
    if stored_ro * spokes_final * coils != total:
        raise ValueError("Internal factoring error: stored_ro*spokes_final*coils != total samples")

    ksp_blk = np.reshape(cpx, (stored_ro, spokes_final, coils), order="F")
    if readout is not None and stored_ro >= readout:
        ksp = ksp_blk[:readout, :, :]
        dbg("trimmed RO from", stored_ro, "to", readout)
    else:
        ksp = ksp_blk
        if readout is None:
            readout = stored_ro
    dbg("final k-space shape:", ksp.shape, "(RO, Spokes, Coils)")
    return ksp


# ---------- Trajectory builders (Kronecker & linear-z GA) ----------

def fib_closest_ge(n: int) -> int:
    if n <= 1:
        return 1
    a, b = 1, 1
    while b < n:
        a, b = b, a + b
    return b


def fib_prev(fk: int) -> int:
    if fk <= 1:
        return 1
    a, b = 1, 1
    while b < fk:
        a, b = b, a + b
    return a


def fib_prev2(fk: int) -> int:
    return fib_prev(fib_prev(fk))


def uv_to_dir(u: float, v: float) -> Tuple[float, float, float]:
    """Map (u,v) in [0,1)^2 to point on unit sphere."""
    z = 1.0 - 2.0 * u
    r2 = 1.0 - z * z
    if r2 < 0.0:
        r2 = 0.0
    r = math.sqrt(r2)
    az = 2.0 * math.pi * v
    dx = r * math.cos(az)
    dy = r * math.sin(az)
    dz = z
    return dx, dy, dz


def kronecker_dir(i: int, N: int) -> Tuple[float, float, float]:
    M = fib_closest_ge(N)
    q1 = fib_prev(M)
    q2 = fib_prev2(M)
    j = i % M
    u = ((j * q1) % M + 0.5) / float(M)
    v = ((j * q2) % M + 0.5) / float(M)
    return uv_to_dir(u, v)


def linZ_ga_dir(i: int, N: int) -> Tuple[float, float, float]:
    phi_inc = (math.sqrt(5.0) - 1.0) * math.pi
    z = 1.0 - 2.0 * ((i + 0.5) / float(N))
    r2 = 1.0 - z * z
    if r2 < 0.0:
        r2 = 0.0
    r = math.sqrt(r2)
    az = (i * phi_inc) % (2.0 * math.pi)
    dx = r * math.cos(az)
    dy = r * math.sin(az)
    dz = z
    return dx, dy, dz


def build_synthetic_traj(ro: int, spokes: int, mode: str) -> np.ndarray:
    """
    Build synthetic 3D radial trajectory of shape (3, RO, Spokes)
    using Bruker GA / Kronecker math. Coordinates are in units of
    1/FOV and span radius ~[-0.5,0.5].
    """
    mode = mode.lower()
    if mode not in ("kron", "linear_z"):
        raise ValueError("traj-mode must be 'kron' or 'linear_z'")
    N = spokes
    # radial coordinate from center-out, mapped to [-0.5,0.5]
    r_lin = (np.arange(ro, dtype=np.float32) - (ro - 1) / 2.0) / float(ro)
    traj = np.zeros((3, ro, spokes), dtype=np.float32)
    for i in range(spokes):
        if mode == "kron":
            dx, dy, dz = kronecker_dir(i, N)
        else:
            dx, dy, dz = linZ_ga_dir(i, N)
        traj[0, :, i] = dx * r_lin
        traj[1, :, i] = dy * r_lin
        traj[2, :, i] = dz * r_lin
    return traj.astype(np.complex64)


# ---------- DCF (Pipe-style, NumPy) ----------

def _normalize_traj_to_grid(traj: np.ndarray, grid_shape: Tuple[int, int, int]):
    ro, sp = traj.shape[1], traj.shape[2]
    kx, ky, kz = traj[0].real, traj[1].real, traj[2].real

    def robust_minmax(a):
        lo = np.nanpercentile(a, 0.5)
        hi = np.nanpercentile(a, 99.5)
        if not np.isfinite(hi - lo) or hi <= lo:
            hi = lo + 1e-3
        return float(lo), float(hi)

    xmin, xmax = robust_minmax(kx)
    ymin, ymax = robust_minmax(ky)
    zmin, zmax = robust_minmax(kz)

    def map_axis(a, lo, hi, n):
        t = (a - lo) / (hi - lo)
        t = np.clip(t, 0.0, 1.0)
        return np.clip(np.rint(t * (n - 1)).astype(np.int32), 0, n - 1)

    ix = map_axis(kx, xmin, xmax, grid_shape[0])
    iy = map_axis(ky, ymin, ymax, grid_shape[1])
    iz = map_axis(kz, zmin, zmax, grid_shape[2])
    return ix, iy, iz


def dcf_pipe_numpy(traj: np.ndarray, iters: int, grid_shape: Tuple[int, int, int]) -> np.ndarray:
    ro, sp = traj.shape[1], traj.shape[2]
    ix, iy, iz = _normalize_traj_to_grid(traj, grid_shape)
    w = np.ones((ro, sp), dtype=np.float32)
    eps = 1e-6
    for _ in range(max(1, iters)):
        grid = np.zeros(grid_shape, dtype=np.float32)
        np.add.at(grid, (ix, iy, iz), w)
        denom = grid[ix, iy, iz] + eps
        w = w / denom
        w *= (w.size / max(np.sum(w), eps))
    return w


# ---------- BART helpers for non-Cart layout ----------

def ksp_to_bart_noncart(ksp: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Map ksp (RO, Spokes, Coils) to BART noncart dims:
    [1,1,1,COILS,1,1,1,1,1,1,RO,Sp,1,1,1,1]
    """
    ro, sp, nc = ksp.shape
    arr = ksp.astype(np.complex64, order="F")
    arr = arr.reshape(1, 1, 1, nc, 1, 1, 1, 1, 1, 1, ro, sp, 1, 1, 1, 1, order="F")
    dims = [1, 1, 1, nc, 1, 1, 1, 1, 1, 1, ro, sp, 1, 1, 1, 1]
    return arr, dims


def traj_to_bart_noncart(traj: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Map traj (3, RO, Spokes) to BART noncart dims:
    [3,1,1,1,1,1,1,1,1,1,RO,Sp,1,1,1,1]
    """
    _, ro, sp = traj.shape
    arr = traj.astype(np.complex64, order="F")
    arr = arr.reshape(3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ro, sp, 1, 1, 1, 1, order="F")
    dims = [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ro, sp, 1, 1, 1, 1]
    return arr, dims


# ---------- BART wrappers ----------

def _bart_path() -> str:
    bart = shutil.which("bart")
    if not bart:
        raise RuntimeError("BART not found on PATH")
    return bart


def _bart_supports_gpu(tool: str) -> bool:
    global BART_GPU_AVAILABLE
    if BART_GPU_AVAILABLE is not None:
        return BART_GPU_AVAILABLE
    bart = _bart_path()
    try:
        subprocess.run(
            [bart, tool, "-g", "-h"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
        BART_GPU_AVAILABLE = True
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or b"").decode(errors="ignore")
        if ("compiled without GPU" in msg) or ("invalid option -- 'g'" in msg) or ("unknown option g" in msg):
            BART_GPU_AVAILABLE = False
        else:
            BART_GPU_AVAILABLE = False
    except Exception:
        BART_GPU_AVAILABLE = False
    return BART_GPU_AVAILABLE


def _run_bart(tool: str, args: List[str], gpu: bool):
    bart = _bart_path()
    use_gpu = gpu and _bart_supports_gpu(tool)
    cmd = [bart, tool]
    if use_gpu:
        cmd.append("-g")
    cmd += args
    print("[bart]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_bart(args: List[str], gpu: bool = False):
    if not args:
        raise ValueError("Empty BART cmd")
    _run_bart(args[0], args[1:], gpu=gpu)


# ---------- Recon helpers ----------

def bart_recon_frame(
    traj_f: np.ndarray,
    ksp_f: np.ndarray,
    out_cfl_base: Path,
    combine: str,
    gpu: bool,
):
    """
    traj_f: (3, RO, Spokes)
    ksp_f:  (RO, Spokes, Coils)
    Writes coil and combined image to BART .cfl/.hdr at out_cfl_base.
    """
    ro, sp, nc = ksp_f.shape

    traj_b, traj_dims = traj_to_bart_noncart(traj_f)
    ksp_b, ksp_dims = ksp_to_bart_noncart(ksp_f)

    traj_base = out_cfl_base.with_name(out_cfl_base.name + "_traj")
    ksp_base = out_cfl_base.with_name(out_cfl_base.name + "_ksp")
    coil_base = out_cfl_base.with_name(out_cfl_base.name + "_coil")

    _write_cfl(traj_base, traj_b, traj_dims)
    _write_cfl(ksp_base, ksp_b, ksp_dims)

    # adjoint NUFFT (image from noncart k-space)
    run_bart(["nufft", "-i", "-t", str(traj_base), str(ksp_base), str(coil_base)], gpu=gpu)

    combine = combine.lower()
    if combine == "sos":
        run_bart(["rss", "8", str(coil_base), str(out_cfl_base)], gpu=False)
    else:
        raise ValueError("Only sos coil combine is implemented right now.")


def cfl_to_3d_numpy(img_base: Path, nx: int, ny: int, nz: int) -> np.ndarray:
    """
    Load a BART image CFL and reshape to (NX,NY,NZ) if possible.
    Assumes image is single-volume, single-complex channel.
    """
    arr = read_cfl(img_base)  # dims16
    total = arr.size
    if total != nx * ny * nz:
        # Try to infer from dims >1 if exactly 3 such dims
        shape = list(arr.shape)
        non1 = [d for d in shape if d > 1]
        if len(non1) == 3 and np.prod(non1) == total:
            img = arr.reshape(non1, order="F")
            return img.astype(np.complex64)
        raise ValueError(
            f"BART image size {total} != NX*NY*NZ={nx*ny*nz}. "
            f"Shape from hdr: {arr.shape}"
        )
    img = arr.reshape((nx, ny, nz), order="F")
    return img.astype(np.complex64)


# ---------- Temporal binning ----------

def frame_starts(total_spokes: int, spokes_per_frame: int, frame_shift: Optional[int]) -> Iterable[int]:
    step = frame_shift if frame_shift and frame_shift > 0 else spokes_per_frame
    for s in range(0, total_spokes - spokes_per_frame + 1, step):
        yield s


# ---------- CLI ----------

def bart_exists() -> bool:
    return shutil.which("bart") is not None


def main():
    global DEBUG
    ap = argparse.ArgumentParser(
        description="Bruker 3D radial recon using BART (sliding window, GA/Kronecker traj)."
    )
    ap.add_argument("--series", type=Path, required=True,
                    help="Bruker series directory (contains fid, acqp, method).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output base path; 4D NIfTI written here.")
    ap.add_argument("--matrix", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))
    ap.add_argument("--combine", type=str, default="sos", help="Coil combine: sos (only).")
    ap.add_argument("--gpu", action="store_true", help="Use BART GPU NUFFT if available.")
    ap.add_argument("--debug", action="store_true")

    # Overrides for FID factoring
    ap.add_argument("--readout", type=int, default=None)
    ap.add_argument("--spokes", type=int, default=None)
    ap.add_argument("--coils", type=int, default=None)
    ap.add_argument("--fid-dtype", type=str, default=None)
    ap.add_argument("--fid-endian", type=str, default=None)

    # Temporal binning
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--spokes-per-frame", type=int, default=None)
    grp.add_argument("--time-per-frame-ms", type=float, default=None)
    ap.add_argument("--frame-shift", type=int, default=None,
                    help="Sliding window shift (spokes). Default = spokes-per-frame.")
    ap.add_argument("--tr-ms", type=float, default=None,
                    help="TR per spoke in ms (for time-per-frame-ms).")
    ap.add_argument("--test-volumes", type=int, default=None,
                    help="If set, reconstruct only first N frames.")
    ap.add_argument("--qc-frame", type=int, default=1,
                    help="1-based index of frame to write as separate 3D QC NIfTI.")

    # DCF
    ap.add_argument("--dcf", type=str, default="none", help="none | pipe:N (Pipe-style DCF iterations).")

    # Trajectory mode when no traj file is present
    ap.add_argument("--traj-mode", type=str, choices=["kron", "linear_z"],
                    help="Synthetic trajectory mode if no $series/traj file is present.")

    args = ap.parse_args()
    DEBUG = args.debug

    if not bart_exists():
        print("ERROR: BART not found in PATH.", file=sys.stderr)
        sys.exit(1)

    series_dir: Path = args.series
    out_base: Path = args.out
    NX, NY, NZ = args.matrix

    # Load k-space
    ksp = load_bruker_kspace(
        series_dir,
        matrix_ro_hint=NX,
        spokes=args.spokes,
        readout=args.readout,
        coils=args.coils,
        fid_dtype=args.fid_dtype,
        fid_endian=args.fid_endian,
    )
    ro, sp_total, nc = ksp.shape
    print(f"[info] Loaded k-space: RO={ro}, Spokes={sp_total}, Coils={nc}")

    # Trajectory handling
    traj_path = series_dir / "traj"
    if traj_path.exists():
        # Not yet wired: we currently rely on synthetic GA/Kronecker only.
        raise NotImplementedError("Reading existing $series/traj file not yet implemented.")
    else:
        if args.traj_mode is None:
            raise RuntimeError(
                "No traj file found; please specify --traj-mode kron|linear_z "
                "to build a synthetic GA/Kronecker trajectory."
            )
        print(f"[info] No traj file; building synthetic {args.traj_mode} trajectory from GA/Kronecker math.")
        traj_full = build_synthetic_traj(ro, sp_total, mode=args.traj_mode)

    # Temporal binning
    spokes_per_frame = args.spokes_per_frame
    if spokes_per_frame is None and args.time_per-frame_ms is not None:
        # This branch is currently unused in your tests; we can revisit later if needed.
        tr_ms = args.tr_ms
        if tr_ms is None or tr_ms <= 0.0:
            raise ValueError("--time-per-frame-ms provided but TR unknown; please pass --tr-ms explicitly.")
        spokes_per_frame = max(1, int(round(args.time_per_frame_ms / tr_ms)))
        print(
            f"[info] Using spokes_per_frame={spokes_per_frame} "
            f"from time_per_frame_ms={args.time_per_frame_ms} and TR={tr_ms} ms"
        )
    if spokes_per_frame is None:
        spokes_per_frame = min(sp_total, 1000)
        print(f"[warn] No binning specified; defaulting spokes_per_frame={spokes_per_frame}")

    frame_shift = args.frame_shift if args.frame_shift is not None else spokes_per_frame
    if frame_shift <= 0:
        frame_shift = spokes_per_frame

    starts = list(frame_starts(sp_total, spokes_per_frame, frame_shift))
    if args.test_volumes is not None:
        starts = starts[: max(0, int(args.test_volumes))]
    nframes = len(starts)
    if nframes == 0:
        raise ValueError("No frames to reconstruct with chosen (spokes-per-frame, frame-shift).")
    print(f"[info] Sliding-window frames: {nframes} (spf={spokes_per_frame}, shift={frame_shift})")

    # DCF config
    dcf_iters = 0
    dcf_mode = "none"
    dcf_arg = args.dcf.lower()
    if dcf_arg.startswith("pipe"):
        dcf_mode = "pipe"
        dcf_iters = 10
        if ":" in dcf_arg:
            try:
                dcf_iters = int(dcf_arg.split(":", 1)[1])
            except Exception:
                pass

    # per-frame recon -> collect into list of 3D images
    frames_imgs: List[np.ndarray] = []

    for fi, s0 in enumerate(starts, 1):
        s1 = s0 + spokes_per_frame
        if s1 > sp_total:
            print(f"[warn] Skipping last partial window at spokes {s0}:{s1}")
            break
        ksp_f = ksp[:, s0:s1, :]
        traj_f = traj_full[:, :, s0:s1]

        # DCF per frame (if requested)
        if dcf_mode == "pipe":
            dcf = dcf_pipe_numpy(traj_f, iters=dcf_iters, grid_shape=(NX, NY, NZ))
            ksp_w = ksp_f * dcf[..., None]
        else:
            ksp_w = ksp_f

        # Per-frame base name (for intermediate CFL)
        vol_base_stem = out_base.stem + f"_vol{fi:05d}"
        vol_cfl_base = out_base.with_name(vol_base_stem)

        # BART adjoint + rss
        bart_recon_frame(traj_f, ksp_w, vol_cfl_base, combine=args.combine, gpu=args.gpu)

        # Pull back as a 3D numpy array (NX,NY,NZ)
        img3 = cfl_to_3d_numpy(vol_cfl_base, NX, NY, NZ)
        frames_imgs.append(img3)

        print(f"[info] Frame {fi}/{nframes} done -> intermediate {vol_cfl_base}.cfl")

    if not frames_imgs:
        raise RuntimeError("No frames were reconstructed; nothing to write.")

    # Stack into 4D (NX, NY, NZ, T)
    img4d = np.stack(frames_imgs, axis=3).astype(np.complex64)
    nt = img4d.shape[3]

    # Main 4D output CFL
    main_cfl_base = out_base
    _write_cfl(main_cfl_base, img4d, [NX, NY, NZ, nt])

    # Choose QC frame (1-based index)
    qc_frame = args.qc_frame if args.qc_frame is not None else 1
    if qc_frame < 1:
        qc_frame = 1
    if qc_frame > nt:
        qc_frame = nt
    qc_idx = qc_frame - 1
    qc_img3 = img4d[:, :, :, qc_idx]
    qc_base = out_base.with_name(out_base.stem + f"_QC_vol{qc_frame:05d}")
    _write_cfl(qc_base, qc_img3, [NX, NY, NZ])

    # Convert CFL(s) to NIfTI via BART toimg
    main_nii = out_base.with_suffix(".nii.gz")
    run_bart(["toimg", str(main_cfl_base), str(main_nii)], gpu=False)

    qc_nii = qc_base.with_suffix(".nii.gz")
    run_bart(["toimg", str(qc_base), str(qc_nii)], gpu=False)

    print(f"[info] 4D NIfTI written to: {main_nii}")
    print(f"[info] QC frame {qc_frame} written to: {qc_nii}")
    print("[info] All requested frames complete.")


if __name__ == "__main__":
    main()
