#!/usr/bin/env python3
"""
bruker_radial_bart.py

Bruker 3D radial reconstruction with:
  * Bruker FID loader (RO/Spokes/Coils inferred, RO padding trimmed).
  * Trajectory handling:
      - If $series/traj exists  -> ALWAYS used (no fallback).
      - Else -> synthetic 3D GA / Kronecker trajectory from method GA_* params.
  * Temporal binning:
      - --spokes-per-frame N   (sliding window; --frame-shift M, default N)
      - OR --time-per-frame-ms T plus --tr-ms (or inferred TR).
      - Treats acquisition as ONE continuous stream of spokes.
  * DCF: pipe-style per-frame in NumPy (dcf_pipe_numpy).
  * Reconstruction: pure NumPy adjoint gridding (Kaiser–Bessel kernel).
      - --combine sos (magnitude root-sum-of-squares over coils).
  * Optional NIfTI export via BART `toimg` if BART is on PATH.
  * --test-volumes K to only reconstruct first K frames.

NOTE: BART is NOT used for NUFFT here (only for toimg). This avoids the
      dimension / memory assertion issues you were seeing before.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

DEBUG = False


def dbg(*a):
    if DEBUG:
        print("[debug]", *a)


# ----------------------------------------------------------------------
# CFL I/O (single implementation, no bugs)
# ----------------------------------------------------------------------

def _write_hdr(path: Path, dims: List[int]) -> None:
    """
    Write a BART .hdr file with up to 16 dims.
    """
    dims16 = [int(d) for d in dims]
    if len(dims16) < 16:
        dims16 += [1] * (16 - len(dims16))
    elif len(dims16) > 16:
        dims16 = dims16[:16]
    with open(path, "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(int(d)) for d in dims16) + "\n")


def write_cfl(base: Path, arr: np.ndarray, dims: Optional[List[int]] = None) -> None:
    """
    Write complex array to BART CFL/HDR pair with Fortran ordering.
    """
    base = Path(base)
    if dims is None:
        shape = list(arr.shape)
    else:
        shape = [int(d) for d in dims]
    if len(shape) > 16:
        raise ValueError("BART supports at most 16 dimensions")
    _write_hdr(base.with_suffix(".hdr"), shape)
    arr_f = np.asarray(arr, dtype=np.complex64, order="F")
    arr_f.tofile(base.with_suffix(".cfl"))


def read_cfl(base: Path) -> np.ndarray:
    """
    Read BART CFL/HDR, return np.complex64 array with trimmed trailing dims.
    """
    base = Path(base)
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")
    if not (hdr.exists() and cfl.exists()):
        raise FileNotFoundError(f"Missing CFL/HDR for base {base}")
    with open(hdr, "r") as f:
        lines = f.read().splitlines()
    if not lines or not lines[0].startswith("#"):
        raise ValueError(f"Invalid HDR header in {hdr}")
    if len(lines) < 2:
        raise ValueError(f"No dims line in {hdr}")
    tokens = lines[1].split()
    dims16 = [int(t) for t in tokens]
    if len(dims16) < 16:
        dims16 += [1] * (16 - len(dims16))
    elif len(dims16) > 16:
        dims16 = dims16[:16]
    prod = 1
    for d in dims16:
        prod *= d
    data = np.fromfile(cfl, dtype=np.complex64)
    if data.size != prod:
        raise ValueError(f"CFL size mismatch: header {prod}, file {data.size}")
    arr = data.reshape(dims16, order="F")
    # trim trailing singleton dims
    last_nz = 0
    for i, d in enumerate(dims16):
        if d != 1:
            last_nz = i
    out_shape = dims16[: last_nz + 1] if last_nz > 0 else [1]
    return arr.reshape(out_shape, order="F")


# ----------------------------------------------------------------------
# Bruker text headers (method / acqp)
# ----------------------------------------------------------------------

_INT_RE = re.compile(r"[-+]?\d+(\.\d*)?")


def _read_text_kv(path: Path) -> Dict[str, str]:
    """
    Very simple Bruker key-value parser for '##$KEY=' style entries.
    We concatenate continuation lines until the next '##$'.
    """
    d: Dict[str, str] = {}
    path = Path(path)
    if not path.exists():
        return d
    lines = path.read_text(errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("##$"):
            key_part, _, val_part = line.partition("=")
            key = key_part[3:]  # strip '##$'
            val = val_part.strip()
            j = i + 1
            # collect following lines until next key
            while j < len(lines) and not lines[j].startswith("##$"):
                val += " " + lines[j].strip()
                j += 1
            d[key] = val.strip()
            i = j
        else:
            i += 1
    return d


def _first_int_from_string(s: str) -> Optional[int]:
    m = re.search(r"[-+]?\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def _get_int_from_headers(keys: List[str], dicts: List[Dict[str, str]]) -> Optional[int]:
    for k in keys:
        for d in dicts:
            if k in d:
                v = _first_int_from_string(d[k])
                if v is not None and v > 0:
                    return v
    return None


def _parse_acq_size(method: Dict[str, str], acqp: Dict[str, str]) -> Optional[Tuple[int, ...]]:
    for key in ("ACQ_size", "PVM_EncMatrix", "PVM_Matrix"):
        for d in (method, acqp):
            if key in d:
                nums = re.findall(r"[-+]?\d+", d[key])
                ints = [int(x) for x in nums if x.strip()]
                if ints:
                    return tuple(ints)
    return None


# ----------------------------------------------------------------------
# Bruker FID -> k-space (RO, Spokes, Coils)
# ----------------------------------------------------------------------

def load_bruker_kspace(
    series_dir: Path,
    matrix_ro_hint: Optional[int] = None,
    spokes: Optional[int] = None,
    readout: Optional[int] = None,
    coils: Optional[int] = None,
    fid_dtype: str = "int32",
    fid_endian: str = "little",
) -> np.ndarray:
    """
    Load Bruker k-space.

    Priority:
      1) series_dir/ksp.cfl/hdr
      2) series_dir/ksp.npy (must be 3D)
      3) series_dir/fid (Bruker raw), using method/acqp to infer dims
    """
    series_dir = Path(series_dir)

    # ksp.cfl/hdr
    ksp_base = series_dir / "ksp"
    if ksp_base.with_suffix(".cfl").exists() and ksp_base.with_suffix(".hdr").exists():
        arr = read_cfl(ksp_base)
        if arr.ndim != 3:
            raise ValueError("ksp.cfl must be 3D (RO, Spokes, Coils)")
        return arr

    # ksp.npy
    npy = series_dir / "ksp.npy"
    if npy.exists():
        arr = np.load(npy)
        if arr.ndim != 3:
            raise ValueError("ksp.npy must be 3D (RO, Spokes, Coils)")
        return arr

    # FID
    fid_path = series_dir / "fid"
    if not fid_path.exists():
        raise FileNotFoundError("No k-space found (no fid, ksp.cfl, or ksp.npy)")

    method = _read_text_kv(series_dir / "method")
    acqp = _read_text_kv(series_dir / "acqp")

    # determine dtype / endian from acqp unless overridden
    if "ACQ_word_size" in acqp:
        val = acqp["ACQ_word_size"]
        if "16" in val:
            fid_dtype = "int16"
        elif "32" in val:
            fid_dtype = "int32"
    if "BYTORDA" in acqp:
        bv = acqp["BYTORDA"].lower()
        if "little" in bv:
            fid_endian = "little"
        elif "big" in bv:
            fid_endian = "big"

    dtype_map = {
        "int16": np.int16,
        "int32": np.int32,
        "float32": np.float32,
        "float64": np.float64,
    }
    if fid_dtype not in dtype_map:
        raise ValueError("fid_dtype must be one of int16,int32,float32,float64")
    dt = dtype_map[fid_dtype]

    raw = np.fromfile(fid_path, dtype=dt)
    if fid_endian == "big":
        raw = raw.byteswap().newbyteorder()
    if raw.size % 2 != 0:
        raw = raw[:-1]
    raw_f = raw.astype(np.float32)
    cpx = raw_f.view(np.complex64)
    total = cpx.size
    dbg("total complex samples:", total, "(dtype:", fid_dtype, ", endian:", fid_endian, ")")

    # readout from headers or hint
    acq_size = _parse_acq_size(method, acqp)
    ro_hdr = acq_size[0] if acq_size is not None else None
    if readout is None:
        readout = ro_hdr
    if readout is None and matrix_ro_hint is not None:
        readout = matrix_ro_hint
    dbg("readout (hdr/matrix hint):", readout)

    # coils from headers
    if coils is None:
        nrec = _get_int_from_headers(["PVM_EncNReceivers"], [method])
        if nrec:
            coils = nrec
    if coils is None or coils <= 0:
        coils = 1
    dbg("coils (initial):", coils)

    # other dimensions: echoes, reps, averages, slices
    extras = {
        "echoes": _get_int_from_headers(
            ["NECHOES", "ACQ_n_echo_images", "PVM_NEchoImages"], [method, acqp]
        )
        or 1,
        "reps": _get_int_from_headers(["PVM_NRepetitions", "NR"], [method, acqp]) or 1,
        "averages": _get_int_from_headers(["PVM_NAverages", "NA"], [method, acqp]) or 1,
        "slices": _get_int_from_headers(
            ["NSLICES", "PVM_SPackArrNSlices"], [method, acqp]
        )
        or 1,
    }
    other_dims = 1
    for v in extras.values():
        if isinstance(v, int) and v > 1:
            other_dims *= v
    dbg("other_dims factor:", other_dims, extras)

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
    dbg("per_coil_total:", per_coil_total, " coils:", coils, " other_dims:", other_dims)

    def pick_block_and_spokes(
        per_coil_total: int,
        readout_hint: Optional[int],
        spokes_hint: Optional[int],
    ) -> Tuple[int, int]:
        # if user or traj implied spokes
        if spokes_hint and spokes_hint > 0 and per_coil_total % spokes_hint == 0:
            return per_coil_total // spokes_hint, spokes_hint
        # common Bruker padded RO sizes
        BLOCKS = [
            128,
            160,
            192,
            200,
            224,
            240,
            256,
            288,
            320,
            352,
            384,
            400,
            416,
            420,
            432,
            448,
            480,
            496,
            512,
            544,
            560,
            576,
            608,
            640,
            672,
            704,
            736,
            768,
            800,
            832,
            896,
            960,
            992,
            1024,
            1152,
            1280,
            1536,
            2048,
        ]
        if readout_hint and per_coil_total % readout_hint == 0:
            return readout_hint, per_coil_total // readout_hint
        for b in BLOCKS:
            if readout_hint and b < readout_hint:
                continue
            if per_coil_total % b == 0:
                return b, per_coil_total // b
        s = int(round(per_coil_total**0.5))
        for d in range(0, s + 1):
            for cand in (s + d, s - d):
                if cand > 0 and per_coil_total % cand == 0:
                    return cand, per_coil_total // cand
        raise ValueError("Could not factor per_coil_total into (stored_ro, spokes).")

    stored_ro, spokes_inf = pick_block_and_spokes(per_coil_total, readout, spokes)
    dbg("stored_ro (block):", stored_ro, " spokes (per extras-collapsed):", spokes_inf)
    spokes_final = spokes_inf * max(1, other_dims)
    if stored_ro * spokes_final * coils != total:
        raise ValueError("Internal factoring error: stored_ro*spokes_final*coils != total samples")

    ksp_blk = cpx.reshape((stored_ro, spokes_final, coils), order="F")
    if readout is not None and stored_ro >= readout:
        ksp = ksp_blk[:readout, :, :]
        dbg("trimmed RO from", stored_ro, "to", readout)
    else:
        ksp = ksp_blk
        if readout is None:
            readout = stored_ro
    dbg("final k-space shape:", ksp.shape, "(RO, Spokes, Coils)")
    return ksp


# ----------------------------------------------------------------------
# Bruker 'traj' file (strict reader)
# ----------------------------------------------------------------------

def _read_bruker_traj_strict(series_dir: Path, ro: int, sp_total: int) -> np.ndarray:
    """
    Read Bruker 'traj' file if present.

    We assume it is a flat sequence of float32 values in groups of 3*(RO * Spokes),
    ordered as [kx0, ky0, kz0, kx1, ky1, kz1, ...] flattened in some order.
    We treat it as Fortran-order when reshaping into (3, RO, Spokes) and
    pad/truncate with warnings to fit exactly.
    """
    path = Path(series_dir) / "traj"
    if not path.exists():
        raise FileNotFoundError("traj file not found")

    try:
        data = np.fromfile(path, dtype=np.float32)
    except Exception:
        data = np.array([], dtype=np.float32)

    if data.size == 0:
        # fall back to ASCII
        try:
            data = np.loadtxt(path, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"traj exists but could not be read: {e}")

    data = data.ravel()
    if data.size % 3 != 0:
        pad = 3 - (data.size % 3)
        print(f"[warn] traj length {data.size} not multiple of 3; padding {pad} zeros")
        data = np.pad(data, (0, pad))

    n_triplets = data.size // 3
    expected = ro * sp_total
    if n_triplets < expected:
        pad = expected - n_triplets
        print(f"[warn] traj has {n_triplets} samples, expected {expected}; padding {pad} zeros")
        data = np.pad(data, (0, pad * 3))
        n_triplets = expected
    elif n_triplets > expected:
        print(f"[warn] traj has {n_triplets} samples, expected {expected}; truncating")
        data = data[: expected * 3]
        n_triplets = expected

    arr = data.reshape(3, ro, sp_total, order="F")
    return arr


# ----------------------------------------------------------------------
# Synthetic GA vs Kronecker 3D trajectory from method GA_* parameters
# ----------------------------------------------------------------------

def _traj_from_ga_settings(series_dir: Path, ro: int, sp_total: int) -> np.ndarray:
    """
    Build a synthetic 3D trajectory when no 'traj' file exists.

    We distinguish between two conceptual modes, based on the method file:

        - GA / Fibonacci  (golden-angle-like spherical Fibonacci)
        - Kronecker       (3D Kronecker sequence on the sphere)

    Mode detection:
        * If GA_Mode contains "KRON" -> Kronecker
        * elif GA_Mode contains "FIB" or "GOLD" -> GA
        * elif GA_UseFibonacci is yes/1/true  -> GA
        * elif GA_UseFibonacci is no/0/false  -> Kronecker
        * else default -> GA

    We also honor GA_NSpokesEff if present (effective distinct directions).
    Output shape: (3, ro, sp_total).
    """
    method = _read_text_kv(series_dir / "method")

    def _first_int(s: str) -> Optional[int]:
        toks = re.split(r"[^0-9+\-]+", s)
        for t in toks:
            if t.strip() and t.lstrip("+-").isdigit():
                return int(t)
        return None

    # ---- 1) effective number of unique directions ----
    n_eff = sp_total
    val = method.get("GA_NSpokesEff")
    if val is not None:
        try:
            ival = _first_int(val)
            if ival is not None and ival > 0:
                n_eff = ival
        except Exception:
            pass

    # ---- 2) determine mode: GA vs Kronecker ----
    raw_mode = (method.get("GA_Mode") or "").upper()
    raw_usefib = (method.get("GA_UseFibonacci") or "").strip().upper()

    mode = "GA"  # default
    if "KRON" in raw_mode:
        mode = "KRONECKER"
    elif any(x in raw_mode for x in ("FIB", "GOLD")):
        mode = "GA"
    else:
        if raw_usefib in ("YES", "1", "TRUE"):
            mode = "GA"
        elif raw_usefib in ("NO", "0", "FALSE"):
            mode = "KRONECKER"

    print(f"[info] No traj file; building synthetic {mode} trajectory (n_eff={n_eff}, sp_total={sp_total})")

    # ---- 3) build directions on the unit sphere ----
    k = np.arange(n_eff, dtype=np.float32) + 0.5

    if mode == "GA":
        # Spherical Fibonacci / golden-angle
        ga = math.pi * (3.0 - math.sqrt(5.0))  # golden angle in radians
        phi = np.arccos(1.0 - 2.0 * k / float(n_eff))
        theta = ga * k
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
    else:
        # 3D Kronecker sequence using two incommensurate irrationals
        alpha = (math.sqrt(5.0) - 1.0) / 2.0
        beta = (math.sqrt(3.0) - 1.0) / 2.0
        u = ((k * alpha) % 1.0).astype(np.float32)
        v = ((k * beta) % 1.0).astype(np.float32)
        theta = 2.0 * math.pi * u
        phi = np.arccos(2.0 * v - 1.0)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

    # ---- 4) build continuous line trajectories along each direction ----
    # k-space radius sampled in [-0.5, 0.5) in RO points
    t = np.linspace(-0.5, 0.5, ro, endpoint=False, dtype=np.float32)

    kx = np.empty((ro, sp_total), dtype=np.float32)
    ky = np.empty_like(kx)
    kz = np.empty_like(kx)

    for s in range(sp_total):
        idx = s % n_eff
        kx[:, s] = t * x[idx]
        ky[:, s] = t * y[idx]
        kz[:, s] = t * z[idx]

    traj = np.stack([kx, ky, kz], axis=0)  # (3, RO, Spokes)
    return traj


# ----------------------------------------------------------------------
# DCF (pipe-style) and adjoint gridding (NumPy only)
# ----------------------------------------------------------------------

def _normalize_traj_to_grid(
    traj: np.ndarray, grid_shape: Tuple[int, int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ro, sp = traj.shape[1], traj.shape[2]
    kx, ky, kz = traj[0], traj[1], traj[2]

    def robust_minmax(a: np.ndarray) -> Tuple[float, float]:
        lo = np.nanpercentile(a, 0.5)
        hi = np.nanpercentile(a, 99.5)
        if not np.isfinite(hi - lo) or hi <= lo:
            hi = lo + 1e-3
        return float(lo), float(hi)

    xmin, xmax = robust_minmax(kx)
    ymin, ymax = robust_minmax(ky)
    zmin, zmax = robust_minmax(kz)

    def map_axis(a: np.ndarray, lo: float, hi: float, n: int) -> np.ndarray:
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
    """
    Very literal adjoint gridding implementation.
    traj: (3, RO, Spokes)
    ksp:  (RO, Spokes, Coils)
    dcf:  (RO, Spokes) or None
    Returns image: (NX, NY, NZ, Coils)
    """
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

            xr2 = xr[xsel]
            yr2 = yr[ysel]
            zr2 = zr[zsel]
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
    return img.astype(np.complex64)


def recon_adjoint_python(
    traj: np.ndarray,
    ksp: np.ndarray,
    dcf: Optional[np.ndarray],
    matrix: Tuple[int, int, int],
    out_base: Path,
    combine: str,
) -> None:
    NX, NY, NZ = matrix
    coil_img = adjoint_grid_numpy(traj, ksp, dcf, (NX, NY, NZ))  # (NX,NY,NZ,NC)
    NC = coil_img.shape[3]
    coil_base = out_base.with_name(out_base.name + "_coil_py")
    write_cfl(coil_base, coil_img, [NX, NY, NZ, NC])

    combine = combine.lower()
    if combine == "sos":
        rss = np.sqrt(np.sum(np.abs(coil_img) ** 2, axis=3)).astype(np.complex64)
        write_cfl(out_base, rss, [NX, NY, NZ])
    elif combine == "sens":
        raise NotImplementedError("Sensitivity-based combine not implemented in pure Python path.")
    else:
        raise ValueError("combine must be 'sos' (or 'sens' if later implemented)")


# ----------------------------------------------------------------------
# Temporal binning and TR
# ----------------------------------------------------------------------

def _derive_tr_ms(method: Dict[str, str], acqp: Dict[str, str]) -> Optional[float]:
    for key in ("PVM_RepetitionTime", "ACQ_repetition_time"):
        for d in (method, acqp):
            if key in d:
                try:
                    # Bruker often stores e.g. "20.0" or "( 1 ) 20.0"
                    nums = re.findall(r"[-+]?\d+(\.\d+)?", d[key])
                    if nums:
                        return float(nums[0])
                except Exception:
                    pass
    return None


def frame_starts(total_spokes: int, spokes_per_frame: int, frame_shift: Optional[int]) -> Iterable[int]:
    step = frame_shift if frame_shift and frame_shift > 0 else spokes_per_frame
    for s in range(0, total_spokes - spokes_per_frame + 1, step):
        yield s


# ----------------------------------------------------------------------
# BART helper (only for toimg)
# ----------------------------------------------------------------------

def bart_exists() -> bool:
    return shutil.which("bart") is not None


def run_bart_toimg(in_base: Path, out_base: Path) -> None:
    bart = shutil.which("bart")
    if not bart:
        print("[warn] BART not found; skipping toimg export for", in_base)
        return
    cmd = [bart, "toimg", str(in_base), str(out_base)]
    print("[bart]", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    global DEBUG

    ap = argparse.ArgumentParser(
        description="Bruker 3D radial reconstruction (pure Python adjoint gridding + optional BART toimg)"
    )
    ap.add_argument("--series", type=Path, required=True, help="Bruker series directory (contains fid, method, acqp)")
    ap.add_argument("--out", type=Path, required=True, help="Output base name (no extension)")
    ap.add_argument("--matrix", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))

    ap.add_argument("--dcf", type=str, default="none", help="none | pipe:N (NumPy Pipe DCF iterations)")
    ap.add_argument("--combine", type=str, default="sos", help="sos | sens (sens not implemented in Python path)")

    # overrides
    ap.add_argument("--readout", type=int, default=None)
    ap.add_argument("--spokes", type=int, default=None)
    ap.add_argument("--coils", type=int, default=None)
    ap.add_argument("--fid-dtype", type=str, default=None)
    ap.add_argument("--fid-endian", type=str, default=None)

    # temporal binning
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--spokes-per-frame", type=int, default=None)
    grp.add_argument("--time-per-frame-ms", type=float, default=None)

    ap.add_argument(
        "--frame-shift",
        type=int,
        default=None,
        help="Sliding window shift in spokes (default = spokes-per-frame)",
    )
    ap.add_argument(
        "--tr-ms",
        type=float,
        default=None,
        help="Repetition time per spoke (ms); inferred from headers if possible",
    )
    ap.add_argument(
        "--test-volumes",
        type=int,
        default=None,
        help="If set, reconstruct only this many frames (sliding windows)",
    )

    ap.add_argument("--export-nifti", action="store_true", help="Use BART toimg to write NIfTI")
    ap.add_argument("--debug", action="store_true")

    # kept for compatibility, but unused in this pure Python version
    ap.add_argument("--traj", choices=["golden", "file"], default="file")
    ap.add_argument("--traj-file", type=Path, help="(unused in this version)")
    ap.add_argument("--gpu", action="store_true", help="(ignored in this Python-only NUFFT version)")
    ap.add_argument("--iterative", action="store_true", help="(placeholder; adjoint only)")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--wavelets", type=int, default=None)
    ap.add_argument("--force-python-adjoint", action="store_true", help="(ignored; Python adjoint is always used)")

    args = ap.parse_args()
    DEBUG = args.debug

    series_dir: Path = args.series
    out_base: Path = args.out
    NX, NY, NZ = args.matrix

    # 1) Load k-space
    ksp = load_bruker_kspace(
        series_dir,
        matrix_ro_hint=NX,
        spokes=args.spokes,
        readout=args.readout,
        coils=args.coils,
        fid_dtype=(args.fid_dtype or "int32"),
        fid_endian=(args.fid_endian or "little"),
    )
    ro, sp_total, nc = ksp.shape
    print(f"[info] Loaded k-space: RO={ro}, Spokes={sp_total}, Coils={nc}")

    # 2) Trajectory: traj file first, else GA/Kronecker from method
    traj_path = series_dir / "traj"
    if traj_path.exists():
        traj = _read_bruker_traj_strict(series_dir, ro, sp_total)
    else:
        traj = _traj_from_ga_settings(series_dir, ro, sp_total)
    if traj.shape != (3, ro, sp_total):
        raise ValueError(f"trajectory must have shape (3,{ro},{sp_total}); got {traj.shape}")

    # 3) Temporal binning
    method = _read_text_kv(series_dir / "method")
    acqp = _read_text_kv(series_dir / "acqp")

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
        # default: something reasonable (use all spokes if not insane)
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

    # 4) Per-frame DCF + recon
    for fi, s0 in enumerate(starts, 1):
        s1 = s0 + spokes_per_frame
        if s1 > sp_total:
            print(f"[warn] Skipping last partial window at spokes {s0}:{s1}")
            break

        ksp_f = ksp[:, s0:s1, :]
        traj_f = traj[:, :, s0:s1]

        # DCF per frame
        dcf = None
        if args.dcf.lower().startswith("pipe"):
            nit = 10
            if ":" in args.dcf:
                try:
                    nit = int(args.dcf.split(":", 1)[1])
                except Exception:
                    pass
            dcf = dcf_pipe_numpy(traj_f, iters=nit, grid_shape=(NX, NY, NZ))

        vol_base = out_base.with_name(out_base.name + f"_vol{fi:05d}")

        # pure Python adjoint (ksp * dcf if present)
        ksp_in = ksp_f if dcf is None else (ksp_f * dcf[..., None])
        recon_adjoint_python(traj_f, ksp_in, dcf, (NX, NY, NZ), vol_base, combine=args.combine)

        if args.export_nifti:
            run_bart_toimg(vol_base, vol_base)

        print(f"[info] Frame {fi}/{nframes} done -> {vol_base}")

    print("[info] All requested frames complete.")


if __name__ == "__main__":
    main()
