#!/usr/bin/env python3
"""
bruker_radial_bart.py

Bruker 3D radial reconstruction using BART with robust fallbacks, now with
sliding‑window binning by time or by spokes, and a --test-volumes limiter.

Key features
- Bruker FID loader: infers RO/Spokes/Coils; trims padded readout blocks; respects traj-implied spokes.
- Trajectory: if $series/traj exists, it is ALWAYS used (no silent fallback).
  * Accepts binary float32/float64 or ASCII; reshapes to (3,RO,Spokes).
  * If size != 3*RO*Spokes, trims or zero‑pads with warnings.
- DCF: Pipe-style (NumPy) per-frame (or none).
- Recon per frame (sliding window):
  * Primary: BART adjoint NUFFT (+ SoS/SENSE) with sticky CPU fallback.
  * Fallback: Pure NumPy adjoint gridding (Kaiser–Bessel) via --force-python-adjoint.
- Temporal binning:
  * --spokes-per-frame N, optional --frame-shift M (default = N for non-overlap)
  * OR --time-per-frame-ms T plus --tr-ms (derived from headers if possible)
  * Treats acquisition as one continuous stream of spokes (no per-repetition reset)
- --test-volumes K reconstructs only the first K frames for quick testing.
- CFL I/O: correct 16D headers for non-Cartesian k-space/trajectory; single implementation.

Examples
--------
# by time (200 ms frames), sliding by 100 spokes, test first 3 volumes using Python adjoint
python bruker_radial_bart.py \
  --series "$path" \
  --matrix 256 256 256 \
  --traj file \
  --time-per-frame-ms 200 \
  --tr-ms 1.2 \
  --frame-shift 100 \
  --dcf pipe:10 \
  --combine sos \
  --force-python-adjoint \
  --test-volumes 3 \
  --export-nifti \
  --out "${out%.nii.gz}_SoS" \
  --debug

# by spokes (800 per frame, overlap 100), BART adjoint if available
python bruker_radial_bart.py \
  --series "$path" \
  --matrix 256 256 256 \
  --traj file \
  --spokes-per-frame 800 \
  --frame-shift 100 \
  --dcf pipe:10 \
  --combine sos \
  --export-nifti \
  --out "${out%.nii.gz}_SoS"
"""

from __future__ import annotations
import argparse, math, shutil, subprocess, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Iterable
import numpy as np

DEBUG = False

def dbg(*a):
    if DEBUG: print("[debug]", *a)

BART_GPU_AVAILABLE = None  # sticky cache

# ---------- CFL I/O (single source of truth) ----------

# Minimal golden-angle 3D fallback (only used if no traj file AND no method arrays)
@dataclass
class TrajSpec:
    readout: int
    spokes: int
    matrix: Tuple[int, int, int]

def golden_angle_3d(spec: TrajSpec) -> np.ndarray:
    RO, SP = int(spec.readout), int(spec.spokes)
    # low-discrepancy directions on sphere (Fibonacci lattice)
    phi = (1 + 5 ** 0.5) / 2
    dz = 2.0 / SP
    kmax = 0.5 * float(max(*spec.matrix))
    t = (np.arange(RO, dtype=np.float32) / (RO - 1) - 0.5) * (2.0 * kmax)
    kx = np.empty((RO, SP), dtype=np.float32)
    ky = np.empty((RO, SP), dtype=np.float32)
    kz = np.empty((RO, SP), dtype=np.float32)
    for s in range(SP):
        z = -1 + dz * (s + 0.5)
        r = max(1e-6, (1.0 - z * z) ** 0.5)
        theta = (2.0 * np.pi * s) / (phi)
        dirx, diry, dirz = r * np.cos(theta), r * np.sin(theta), z
        kx[:, s] = dirx * t
        ky[:, s] = diry * t
        kz[:, s] = dirz * t
    return np.stack([kx, ky, kz], axis=0)

# ---------- CFL I/O helpers ----------


def _write_hdr(path: Path, dims: List[int]):
    with open(path, "w") as f:
        f.write("# Dimensions
")
        f.write(" ".join(str(d) for d in dims) + "
")
    return np.reshape(data, dims, order="F")

# ---- helper: infer spokes from $series/traj length ----

def _probe_traj_spokes(series_dir: Path, ro_hint: Optional[int]) -> Optional[int]:
    tpath = Path(series_dir) / "traj"
    if ro_hint is None or ro_hint <= 0 or not tpath.exists():
        return None
    # Try binary float32 then float64
    for dt in (np.float32, np.float64):
        try:
            vals = np.fromfile(tpath, dtype=dt)
            if vals.size % 3 == 0:
                nsamp = vals.size // 3
                if nsamp % ro_hint == 0:
                    sp = nsamp // ro_hint
                    return int(sp) if sp > 0 else None
        except Exception:
            pass
    # Fallback: ASCII tokens
    try:
        toks = (tpath.read_text(errors="ignore").replace("{"," ").replace("}"," ")
                .replace(","," ").split())
        cnt = 0
        for t in toks:
            try:
                float(t)
                cnt += 1
            except Exception:
                continue
        if cnt % 3 == 0:
            nsamp = cnt // 3
            if nsamp % ro_hint == 0:
                sp = nsamp // ro_hint
                return int(sp) if sp > 0 else None
    except Exception:
        pass
    return None

# ---------- FID / k-space ----------

def load_bruker_kspace(
    series_dir: Path,
    matrix_ro_hint: Optional[int] = None,
    spokes: Optional[int] = None,
    readout: Optional[int] = None,
    coils: Optional[int] = None,
    fid_dtype: str = "int32",
    fid_endian: str = "little",
) -> np.ndarray:
    series_dir = Path(series_dir)
    dbg("series_dir:", series_dir)

    # existing k-space files take precedence
    cfl = series_dir / "ksp"
    if cfl.with_suffix(".cfl").exists() and cfl.with_suffix(".hdr").exists():
        arr = read_cfl(cfl)
        dbg("loaded ksp.cfl:", arr.shape)
        return arr
    npy = series_dir / "ksp.npy"
    if npy.exists():
        arr = np.load(npy)
        dbg("loaded ksp.npy:", arr.shape)
        if arr.ndim != 3:
            raise ValueError("ksp.npy must be 3D (RO,Spokes,Coils)")
        return arr

    fid_path = series_dir / "fid"
    if not fid_path.exists():
        raise FileNotFoundError("No k-space found (no fid, ksp.cfl, or ksp.npy)")

    method = _read_text_kv(series_dir / "method")
    acqp = _read_text_kv(series_dir / "acqp")

    if "BYTORDA" in acqp and "big" in acqp["BYTORDA"].lower():
        fid_endian = "big"
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

    if readout is None:
        acq_size = _parse_acq_size(method, acqp)
        if acq_size:
            readout = acq_size[0]
    if readout is None and matrix_ro_hint:
        readout = matrix_ro_hint
    dbg("readout (hdr/matrix hint):", readout)

    if coils is None:
        nrec = _get_int_from_headers(["PVM_EncNReceivers"], [method])
        if nrec and nrec > 0:
            coils = nrec
    if coils is None or coils <= 0:
        coils = 1
    dbg("coils (initial):", coils)

    extras = {
        "echoes": _get_int_from_headers(["NECHOES", "ACQ_n_echo_images", "PVM_NEchoImages"], [method, acqp])
        or 1,
        "reps": _get_int_from_headers(["PVM_NRepetitions", "NR"], [method, acqp]) or 1,
        "averages": _get_int_from_headers(["PVM_NAverages", "NA"], [method, acqp]) or 1,
        "slices": _get_int_from_headers(["NSLICES", "PVM_SPackArrNSlices"], [method, acqp]) or 1,
    }
    other_dims = 1
    for v in extras.values():
        if isinstance(v, int) and v > 1:
            other_dims *= v
    dbg("other_dims factor:", other_dims, extras)

    # If traj exists and implies spokes, prefer it so k-space matches traj
    if spokes is None:
        ro_hint_for_traj = readout if readout is not None else matrix_ro_hint
        sp_hint = _probe_traj_spokes(series_dir, ro_hint_for_traj)
        if sp_hint is not None:
            spokes = sp_hint

    denom = coils * max(1, other_dims)
    if total % denom != 0:
        dbg("total not divisible by coils*other_dims; relaxing extras")
        denom = coils
        if total % denom != 0:
            dbg("still not divisible; relaxing coils->1")
            coils = 1
            denom = coils
            if total % denom != 0:
                raise ValueError("Cannot factor FID length with any (coils, other_dims) combo.")
    per_coil_total = total // denom

    def pick_block_and_spokes(per_coil_total: int, readout_hint: Optional[int], spokes_hint: Optional[int]) -> Tuple[int, int]:
        # 1) honor an explicit spokes hint when it divides
        if spokes_hint and spokes_hint > 0 and per_coil_total % spokes_hint == 0:
            ro_cand = per_coil_total // spokes_hint
            # if we also have a readout hint and it's a divisor, prefer it over tiny ro
            if readout_hint and per_coil_total % readout_hint == 0:
                return readout_hint, per_coil_total // readout_hint
            return ro_cand, spokes_hint
        # 2) if readout hint divides, take it (prevents absurd RO like 3)
        if readout_hint and readout_hint > 0 and per_coil_total % readout_hint == 0:
            return readout_hint, per_coil_total // readout_hint
        # 3) common stored RO block sizes (Bruker pads to these)
        BLOCKS = [
            128, 160, 192, 200, 224, 240, 256, 288, 320, 352, 384, 400, 416, 420, 432, 448, 480, 496, 512,
            544, 560, 576, 608, 640, 672, 704, 736, 768, 800, 832, 896, 960, 992, 1024, 1152, 1280, 1536, 2048,
        ]
        for b in [x for x in BLOCKS if not readout_hint or x >= readout_hint]:
            if per_coil_total % b == 0:
                return b, per_coil_total // b
        # 4) fallback: nearest divisor to hint (or sqrt)
        s = int(round(per_coil_total ** 0.5))
        target = readout_hint if readout_hint else s
        best = None
        for d in range(1, s + 1):
            if per_coil_total % d == 0:
                ro1, sp1 = d, per_coil_total // d
                ro2, sp2 = sp1, ro1
                # choose factor closer to target
                cand = ro1 if abs(ro1 - target) <= abs(ro2 - target) else ro2
                pair = (ro1, sp1) if cand == ro1 else (ro2, sp2)
                if best is None or abs(pair[0] - target) < abs(best[0] - target):
                    best = pair
        if best is not None:
            ro_b, sp_b = best
            # if we still ended up with tiny RO while hint exists and divides, flip
            if readout_hint and per_coil_total % readout_hint == 0 and ro_b < max(32, readout_hint // 4):
                return readout_hint, per_coil_total // readout_hint
            return ro_b, sp_b
        raise ValueError("Could not factor per_coil_total into (stored_ro, spokes).")

    stored_ro, spokes_inf = pick_block_and_spokes(per_coil_total, readout, spokes)
    dbg("stored_ro (block):", stored_ro, " spokes (per extras-collapsed):", spokes_inf)
    spokes_final = spokes_inf * max(1, other_dims)
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

# ---------- Trajectory readers (Bruker) ----------

def _parse_bruker_braced_floats(txt: str) -> np.ndarray:
    """Parse Bruker braced lists like '{ 1 2 3 }' or nested '{{...}{...}}' into float32 array."""
    if not txt:
        return np.array([], dtype=np.float32)
    start = txt.find('{')
    end = txt.rfind('}')
    body = txt[start+1:end] if (start != -1 and end != -1 and end > start) else txt
    cleaned = body.replace('{', ' ').replace('}', ' ').replace('
', ' ').replace('', ' ').replace(',', ' ')
    vals = []
    for t in cleaned.split():
        try:
            vals.append(float(t))
        except Exception:
            continue
    return np.array(vals, dtype=np.float32)

def _read_traj_from_method(series_dir: Path, ro: int) -> Optional[Tuple[np.ndarray, int]]:
    method = _read_text_kv(Path(series_dir) / 'method')
    kx_txt = method.get('PVM_TrajKx') or method.get('##$PVM_TrajKx')
    ky_txt = method.get('PVM_TrajKy') or method.get('##$PVM_TrajKy')
    kz_txt = method.get('PVM_TrajKz') or method.get('##$PVM_TrajKz')
    if not (kx_txt and ky_txt and kz_txt):
        return None
    kx = _parse_bruker_braced_floats(kx_txt)
    ky = _parse_bruker_braced_floats(ky_txt)
    kz = _parse_bruker_braced_floats(kz_txt)
    if not (kx.size and ky.size and kz.size) or not (kx.size == ky.size == kz.size):
        print('[warn] PVM_TrajK* arrays not present or size mismatch; cannot derive trajectory from method.')
        return None
    nsamp = int(kx.size)
    if ro <= 0 or nsamp % ro != 0:
        print(f"[warn] PVM_TrajK* length {nsamp} not divisible by readout {ro}; cannot derive spokes.")
        return None
    sp = nsamp // ro
    traj = np.stack([
        kx.reshape(ro, sp, order='F'),
        ky.reshape(ro, sp, order='F'),
        kz.reshape(ro, sp, order='F')
    ], axis=0)
    return traj.astype(np.float32), int(sp)

def _read_bruker_traj_strict(series_dir: Path, ro: int, sp: int) -> np.ndarray:
    tpath = Path(series_dir) / 'traj'
    if not tpath.exists():
        raise FileNotFoundError(f"No trajectory file found at {tpath}")
    # Try binary float32/float64 then ASCII
    for dt in (np.float32, np.float64, None):
        try:
            if dt is not None:
                vals = np.fromfile(tpath, dtype=dt).astype(np.float32, copy=False)
            else:
                toks = tpath.read_text(errors='ignore').strip().split()
                vals = np.array([float(x) for x in toks], dtype=np.float32)
        except Exception:
            continue
        if vals.size % 3 != 0:
            try:
                data = np.loadtxt(tpath, dtype=np.float32)
                if data.ndim == 2 and data.shape[1] == 3:
                    vals = data.reshape(-1).astype(np.float32)
                else:
                    continue
            except Exception:
                continue
        nsamp = vals.size // 3
        need = ro * sp
        if nsamp % ro == 0 and nsamp != need:
            sp2 = nsamp // ro
            print(f"[warn] Trajectory implies spokes={sp2}, but k-space has {sp}. Using {sp2} to match traj.")
            sp = sp2
            need = ro * sp
        if nsamp > need:
            print(f"[warn] Trajectory has {nsamp} samples; expected {need}. Trimming extras.")
            vals = vals[: 3 * need]
        elif nsamp < need:
            print(f"[warn] Trajectory has only {nsamp} samples; expected {need}. Zero-padding the rest.")
            vals = np.concatenate([vals, np.zeros(3 * (need - nsamp), dtype=np.float32)], axis=0)
        return vals.reshape(3, ro, sp, order='F')
    raise ValueError(f"Could not parse trajectory file at {tpath} in any known format.")

# ---------- BART wrappers ----------

def _bart_path() -> str:
    bart = shutil.which("bart")
    if not bart:
        raise RuntimeError("BART not found in PATH")
    return bart

def _bart_supports_gpu(tool: str) -> bool:
    global BART_GPU_AVAILABLE
    if BART_GPU_AVAILABLE is not None:
        return BART_GPU_AVAILABLE
    bart = _bart_path()
    try:
        subprocess.run([bart, tool, "-g", "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        BART_GPU_AVAILABLE = True
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or b"").decode(errors="ignore")
        BART_GPU_AVAILABLE = not (
            ("compiled without GPU" in msg)
            or ("invalid option -- 'g'" in msg)
            or ("unknown option g" in msg)
        )
    except Exception:
        BART_GPU_AVAILABLE = False
    return BART_GPU_AVAILABLE

def _run_bart(tool: str, args: List[str], gpu: bool):
    global BART_GPU_AVAILABLE
    bart = _bart_path()
    if gpu and (BART_GPU_AVAILABLE is None or BART_GPU_AVAILABLE is True):
        if _bart_supports_gpu(tool):
            cmd = [bart, tool, "-g"] + args
            print("[bart]", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
                return
            except Exception:
                print("[warn] BART GPU attempt failed; using CPU for the rest of this run.")
                BART_GPU_AVAILABLE = False
    cmd = [bart, tool] + args
    print("[bart]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_bart(cmd: List[str], gpu: bool = False):
    if not cmd:
        raise ValueError("Empty BART command")
    _run_bart(cmd[0], cmd[1:], gpu=gpu)

# ---------- DCF (NumPy Pipe) ----------

def _normalize_traj_to_grid(traj: np.ndarray, grid_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ro, sp = traj.shape[1], traj.shape[2]
    kx, ky, kz = traj[0], traj[1], traj[2]
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

# ---------- Layout helpers for BART noncart ----------

def ksp_to_bart_noncart(ksp: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    # Expect ksp shape (RO, Spokes, Coils). Map to BART dims with RO at dim10, Spokes at dim11, Coils at dim3, leading dim 1
    ro, sp, nc = ksp.shape
    arr = ksp.astype(np.complex64, order="F")
    arr = arr.reshape(ro, sp, nc, *([1] * 13), order="F")
    # permute to match BART's noncart expectations
    # We'll produce an array with dims: [1,1,1, nc, 1,1,1,1,1,1, ro, sp, 1,1,1,1]
    arr = np.transpose(arr, (3, 4, 5, 2, 6, 7, 8, 9, 12, 13, 0, 1, 10, 11, 14, 15))
    dims = [1] * 16
    dims[3] = nc
    dims[10] = ro
    dims[11] = sp
    return arr, dims

def traj_to_bart_noncart(traj: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    # Expect traj shape (3, RO, Spokes). Map to dims: [3,1,1,1,1,1,1,1,1,1,RO,Sp,1,1,1,1]
    _, ro, sp = traj.shape
    arr = traj.astype(np.complex64, order="F")
    arr = arr.reshape(3, ro, sp, *([1] * 13), order="F")
    arr = np.transpose(arr, (0, 3, 4, 5, 6, 7, 8, 9, 12, 13, 1, 2, 10, 11, 14, 15))
    dims = [1] * 16
    dims[0] = 3
    dims[10] = ro
    dims[11] = sp
    return arr, dims

# ---------- NumPy adjoint (KB gridding) ----------

def _kaiser_bessel(u: np.ndarray, width: float, beta: float) -> np.ndarray:
    x = np.abs(u)
    out = np.zeros_like(x, dtype=np.float32)
    half = width / 2.0
    m = x <= half
    from numpy import i0
    t = np.sqrt(1.0 - (x[m] / half) ** 2)
    out[m] = (i0(beta * t) / i0(beta)).astype(np.float32)
    return out

def _deapod_1d(N: int, os: float) -> np.ndarray:
    k = (np.arange(int(N * os)) - (N * os) / 2) / (N * os)
    eps = 1e-6
    corr = np.maximum(eps, np.sinc(k))
    return (1.0 / corr).astype(np.float32)

def adjoint_grid_numpy(
    traj: np.ndarray,
    ksp: np.ndarray,
    dcf: Optional[np.ndarray],
    grid_shape: Tuple[int, int, int],
    oversamp: float = 1.5,
    kb_width: float = 3.0,
    kb_beta: float = 8.0,
) -> np.ndarray:
    RO, SP, NC = ksp.shape
    NX, NY, NZ = grid_shape
    NXg, NYg, NZg = int(round(NX * oversamp)), int(round(NY * oversamp)), int(round(NZ * oversamp))
    kx, ky, kz = traj[0], traj[1], traj[2]
    kmax = 0.5 * float(max(NX, NY, NZ))
    gx = (kx / (2 * kmax) + 0.5) * NXg
    gy = (ky / (2 * kmax) + 0.5) * NYg
    gz = (kz / (2 * kmax) + 0.5) * NZg
    deapx = _deapod_1d(NX, oversamp)
    deapy = _deapod_1d(NY, oversamp)
    deapz = _deapod_1d(NZ, oversamp)
    img_grid = np.zeros((NXg, NYg, NZg, NC), dtype=np.complex64)
    hw = int(math.ceil(kb_width / 2.0))
    w = np.ones((RO, SP), dtype=np.float32) if dcf is None else dcf.astype(np.float32)
    for s in range(SP):
        cx = np.floor(gx[:, s]).astype(int)
        cy = np.floor(gy[:, s]).astype(int)
        cz = np.floor(gz[:, s]).astype(int)
        for t in range(RO):
            x0, y0, z0 = cx[t], cy[t], cz[t]
            xr = np.arange(x0 - hw, x0 + hw + 1)
            yr = np.arange(y0 - hw, y0 + hw + 1)
            zr = np.arange(z0 - hw, z0 + hw + 1)
            wx = _kaiser_bessel(xr - gx[t, s], kb_width, kb_beta)
            wy = _kaiser_bessel(yr - gy[t, s], kb_width, kb_beta)
            wz = _kaiser_bessel(zr - gz[t, s], kb_width, kb_beta)
            wxyz = (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]).astype(np.float32) * w[t, s]
            xsel = (xr >= 0) & (xr < NXg)
            ysel = (yr >= 0) & (yr < NYg)
            zsel = (zr >= 0) & (zr < NZg)
            if not (np.any(xsel) and np.any(ysel) and np.any(zsel)):
                continue
            xr2, yr2, zr2 = xr[xsel], yr[ysel], zr[zsel]
            wxyz2 = wxyz[xsel][:, ysel][:, :, zsel]
            for c in range(NC):
                val = ksp[t, s, c]
                img_grid[np.ix_(xr2, yr2, zr2, [c])] += (wxyz2[..., None] * val).astype(np.complex64)
    img = np.fft.ifftn(np.fft.ifftshift(img_grid, axes=(0, 1, 2)), axes=(0, 1, 2))
    img = np.fft.fftshift(img, axes=(0, 1, 2))
    x0 = (NXg - NX) // 2
    y0 = (NYg - NY) // 2
    z0 = (NZg - NZ) // 2
    img = img[x0 : x0 + NX, y0 : y0 + NY, z0 : z0 + NZ, :]
    img *= deapx[:NX, None, None, None]
    img *= deapy[None, :NY, None, None]
    img *= deapz[None, None, :NZ, None]
    return img.astype(np.complex64)  # (NX,NY,NZ,NC)

# ---------- Recon flows ----------

def estimate_sens_maps(coil_imgs_base: Path, out_base: Path, calib: Optional[int] = None, gpu: bool = False):
    cmd = ["ecalib"]
    if calib is not None:
        cmd += ["-r", str(calib)]
    cmd += [str(coil_imgs_base), str(out_base)]
    run_bart(cmd, gpu=gpu)

def recon_adjoint_bart(traj_base: Path, ksp_base: Path, out_base: Path, combine: str, gpu: bool):
    coil_base = out_base.with_name(out_base.name + "_coil")
    run_bart(["nufft", "-a", "-t", str(traj_base), str(ksp_base), str(coil_base)], gpu=gpu)
    if combine.lower() == "sos":
        run_bart(["rss", "8", str(coil_base), str(out_base)], gpu=gpu)
    elif combine.lower() == "sens":
        maps = out_base.with_name(out_base.name + "_maps")
        estimate_sens_maps(coil_base, maps, gpu=gpu)
        run_bart(["pics", "-S", str(coil_base), str(maps), str(out_base)], gpu=gpu)
    else:
        raise ValueError("combine must be sos|sens")

def recon_adjoint_python(
    traj: np.ndarray,
    ksp: np.ndarray,
    dcf: Optional[np.ndarray],
    matrix: Tuple[int, int, int],
    out_base: Path,
    combine: str,
):
    NX, NY, NZ = matrix
    coil_img = adjoint_grid_numpy(traj, ksp, dcf, (NX, NY, NZ))  # (NX,NY,NZ,NC)
    coil_base = out_base.with_name(out_base.name + "_coil_py")
    _write_cfl(coil_base, coil_img, [NX, NY, NZ, coil_img.shape[3]] + [1] * 12)
    if combine.lower() == "sos":
        rss = np.sqrt(np.sum(np.abs(coil_img) ** 2, axis=3)).astype(np.complex64)
        _write_cfl(out_base, rss, [NX, NY, NZ] + [1] * 13)
    elif combine.lower() == "sens":
        maps = out_base.with_name(out_base.name + "_maps_py")
        estimate_sens_maps(coil_base, maps, gpu=False)
        run_bart(["pics", "-S", str(coil_base), str(maps), str(out_base)], gpu=False)
    else:
        raise ValueError("combine must be sos|sens")

# ---------- Frame binning ----------

def _derive_tr_ms(method: dict, acqp: dict) -> Optional[float]:
    for k in ("PVM_RepetitionTime", "ACQ_repetition_time"):
        if k in method:
            try:
                return float(method[k].split()[0])
            except:
                pass
        if k in acqp:
            try:
                return float(acqp[k].split()[0])
            except:
                pass
    return None

def frame_starts(total_spokes: int, spokes_per_frame: int, frame_shift: Optional[int]) -> Iterable[int]:
    step = frame_shift if frame_shift and frame_shift > 0 else spokes_per_frame
    for s in range(0, total_spokes - spokes_per_frame + 1, step):
        yield s

# ---------- CLI ----------

def bart_exists() -> bool:
    return shutil.which("bart") is not None

def main():
    global DEBUG
    ap = argparse.ArgumentParser(description="Bruker 3D radial reconstruction using BART (with sliding window)")
    ap.add_argument("--series", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--matrix", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))
    ap.add_argument("--traj", choices=["golden", "file"], default="file")
    ap.add_argument("--traj-file", type=Path, help="Optional external traj file (unused if $series/traj exists)")
    ap.add_argument("--dcf", type=str, default="none", help="none | pipe:N")
    ap.add_argument("--combine", type=str, default="sos", help="sos|sens")
    ap.add_argument("--iterative", action="store_true")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--wavelets", type=int, default=None)
    ap.add_argument("--export-nifti", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--debug", action="store_true")
    # overrides
    ap.add_argument("--readout", type=int, default=None)
    ap.add_argument("--spokes", type=int, default=None)
    ap.add_argument("--coils", type=int, default=None)
    ap.add_argument("--fid-dtype", type=str, default=None)
    ap.add_argument("--fid-endian", type=str, default=None)
    # python adjoint
    ap.add_argument("--force-python-adjoint", action="store_true")
    # temporal binning
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--spokes-per-frame", type=int, default=None)
    grp.add_argument("--time-per-frame-ms", type=float, default=None)
    ap.add_argument("--frame-shift", type=int, default=None, help="Sliding window shift in spokes (default = spokes-per-frame)")
    ap.add_argument("--tr-ms", type=float, default=None, help="Repetition time per spoke (ms); inferred from headers if possible")
    ap.add_argument("--test-volumes", type=int, default=None, help="If set, reconstruct only this many frames")

    args = ap.parse_args()
    DEBUG = args.debug

    if not bart_exists():
        print("ERROR: BART not found on PATH.", file=sys.stderr)
        sys.exit(1)

    series_dir: Path = args.series
    out_base: Path = args.out
    NX, NY, NZ = args.matrix

    # spokes hint from traj to keep k-space aligned with trajectory
    sp_hint = _probe_traj_spokes(series_dir, NX)
    if DEBUG:
        print(f"[debug] traj-derived spokes hint: {sp_hint}")

    # k-space
    ksp = load_bruker_kspace(
        series_dir,
        matrix_ro_hint=NX,
        spokes=(sp_hint if sp_hint is not None else args.spokes),
        readout=args.readout,
        coils=args.coils,
        fid_dtype=(args.fid_dtype or "int32"),
        fid_endian=(args.fid_endian or "little"),
    )
    ro, sp_total, nc = ksp.shape
    print(f"[info] Loaded k-space: RO={ro}, Spokes={sp_total}, Coils={nc}")

    method = _read_text_kv(series_dir / "method")
    acqp = _read_text_kv(series_dir / "acqp")

    # trajectory: STRICT use of $series/traj if present; else method(PVM_TrajKx/Ky/Kz); else golden/arg file
    traj_path = series_dir / "traj"
    if traj_path.exists():
        bruker_traj = _read_bruker_traj_strict(series_dir, ro, sp_total)
    else:
        # try from method keys
        mt = _read_traj_from_method(series_dir, ro)
        if mt is not None:
            bruker_traj, sp_from_method = mt
            if sp_from_method != sp_total:
                print(f"[warn] Method-derived spokes={sp_from_method} != k-space spokes={sp_total}; trimming to match k-space.")
                sp_use = min(sp_from_method, sp_total)
                bruker_traj = bruker_traj[:, :, :sp_use]
                ksp = ksp[:, :sp_use, :]
                sp_total = sp_use
        else:
            if args.traj == "file" and args.traj_file is not None:
                if args.traj_file.with_suffix(".cfl").exists() and args.traj_file.with_suffix(".hdr").exists():
                    bruker_traj = read_cfl(args.traj_file)
                elif args.traj_file.suffix == ".npy" and args.traj_file.exists():
                    bruker_traj = np.load(args.traj_file)
                else:
                    bruker_traj = golden_angle_3d(TrajSpec(readout=ro, spokes=sp_total, matrix=(NX, NY, NZ)))
            elif args.traj == "golden":
                bruker_traj = golden_angle_3d(TrajSpec(readout=ro, spokes=sp_total, matrix=(NX, NY, NZ)))
            else:
                raise FileNotFoundError("No trajectory found. Provide $series/traj (preferred), or --traj-file, or ensure method contains PVM_TrajKx/Ky/Kz.")))
    if bruker_traj.shape != (3, ro, sp_total):
        raise ValueError(f"trajectory must have shape (3,{ro},{sp_total}); got {bruker_traj.shape}")

    # temporal binning: compute spokes_per_frame
    spokes_per_frame = args.spokes_per_frame
    if spokes_per_frame is None and args.time_per_frame_ms is not None:
        tr_ms = args.tr_ms if args.tr_ms is not None else _derive_tr_ms(method, acqp)
        if tr_ms is None or tr_ms <= 0:
            raise ValueError("--time-per-frame-ms provided but TR unknown. Pass --tr-ms explicitly.")
        spokes_per_frame = max(1, int(round(args.time_per_frame_ms / tr_ms)))
        print(
            f"[info] Using spokes_per_frame={spokes_per_frame} from time_per_frame_ms={args.time_per_frame_ms} and TR={tr_ms} ms"
        )
    if spokes_per_frame is None:
        # default: single volume using all spokes (or modest default)
        spokes_per_frame = min(sp_total, 1000)
        print(f"[warn] No frame binning specified; defaulting spokes_per_frame={spokes_per_frame}")

    frame_shift = args.frame_shift if args.frame_shift is not None else spokes_per_frame
    if frame_shift <= 0:
        frame_shift = spokes_per_frame

    starts = list(frame_starts(sp_total, spokes_per_frame, frame_shift))
    if args.test_volumes is not None:
        starts = starts[: max(0, int(args.test_volumes))]
    nframes = len(starts)
    if nframes == 0:
        raise ValueError("No frames to reconstruct with the chosen (spokes_per_frame, frame_shift).")
    print(f"[info] Sliding-window frames: {nframes} (spf={spokes_per_frame}, shift={frame_shift})")

    # per-frame DCF+recon
    for fi, s0 in enumerate(starts, 1):
        s1 = s0 + spokes_per_frame
        ksp_f = ksp[:, s0:s1, :]
        traj_f = bruker_traj[:, :, s0:s1]
        if ksp_f.shape[1] < spokes_per_frame:
            print(f"[warn] Skipping last partial window at spokes {s0}:{s1}")
            break
        # DCF (per frame) if requested
        dcf = None
        if args.dcf.lower().startswith("pipe"):
            nit = 10
            if ":" in args.dcf:
                try:
                    nit = int(args.dcf.split(":", 1)[1])
                except:
                    pass
            dcf = dcf_pipe_numpy(traj_f, iters=nit, grid_shape=(NX, NY, NZ))
        # Recon: python or BART
        vol_base = out_base.with_name(out_base.name + f"_vol{fi:05d}")
        if args.force_python_adjoint:
            ksp_in = ksp_f if dcf is None else (ksp_f * dcf[..., None])
            recon_adjoint_python(traj_f, ksp_in, dcf, (NX, NY, NZ), vol_base, combine=args.combine)
        else:
            # Write per-frame inputs in BART layout
            ksp16, kspdims = ksp_to_bart_noncart(ksp_f if dcf is None else (ksp_f * dcf[..., None]))
            traj16, trajdims = traj_to_bart_noncart(traj_f)
            ksp_base = vol_base.with_name(vol_base.name + "_ksp")
            traj_base = vol_base.with_name(vol_base.name + "_traj")
            _write_cfl(ksp_base, ksp16, kspdims)
            _write_cfl(traj_base, traj16, trajdims)
            try:
                recon_adjoint_bart(traj_base, ksp_base, vol_base, combine=args.combine, gpu=args.gpu)
            except subprocess.CalledProcessError:
                print("[warn] BART adjoint failed; falling back to pure-NumPy adjoint gridding for this frame.")
                ksp_in = ksp_f if dcf is None else (ksp_f * dcf[..., None])
                recon_adjoint_python(traj_f, ksp_in, dcf, (NX, NY, NZ), vol_base, combine=args.combine)
        if args.export_nifti:
            run_bart(["toimg", str(vol_base), str(vol_base)], gpu=False)
        print(f"[info] Frame {fi}/{nframes} done -> {vol_base}")

    print("[info] All requested frames complete.")

if __name__ == "__main__":
    main()
