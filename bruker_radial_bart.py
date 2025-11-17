#!/usr/bin/env python3
"""
bruker_radial_bart.py

Bruker 3D radial reconstruction using BART (optional) with:
- Bruker FID loader (infers RO/Spokes/Coils and trims padded RO).
- Trajectory from $series/traj if present.
- Otherwise synthetic trajectory rebuilt from GA / Kronecker settings in method file.
- Sliding-window binning by spokes with optional overlap.
- Optional DCF (Pipe-style) computed in NumPy.
- Optional adjoint NUFFT via BART, with pure-Python gridding fallback.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Iterable

import numpy as np

DEBUG = False
BART_GPU_AVAILABLE: Optional[bool] = None


def dbg(*a):
    if DEBUG:
        print("[debug]", *a)


# ---------- Simple Bruker header helpers ----------

def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except FileNotFoundError:
        return ""


def _read_text_kv(path: Path) -> Dict[str, str]:
    """
    Very simple key/value Bruker parser: ##$KEY=VALUE (single line).
    Multi-line arrays are handled separately.
    """
    txt = _read_text(path)
    out: Dict[str, str] = {}
    for line in txt.splitlines():
        if line.startswith("##$"):
            try:
                key, val = line[3:].split("=", 1)
                out[key.strip()] = val.strip()
            except ValueError:
                continue
    return out


def _get_int_from_headers(keys: List[str], dicts: List[Dict[str, str]]) -> Optional[int]:
    for d in dicts:
        if d is None:
            continue
        for k in keys:
            if k in d:
                raw = d[k]
                # strip brackets/parentheses
                raw = raw.replace("(", " ").replace(")", " ").replace(",", " ")
                for tok in raw.split():
                    try:
                        return int(tok)
                    except ValueError:
                        continue
    return None


def _extract_bruker_array(path: Path, name: str) -> Optional[Tuple[np.ndarray, List[int]]]:
    """
    Extract a numeric array parameter from a Bruker header file.
    Returns (array, dims) or None if not found.
    """
    txt = _read_text(path)
    lines = txt.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"##${name}="):
            start = i
            header = line
            break
    if start is None:
        return None

    # parse dims from "(d1,d2,...)" if present
    dims: List[int] = []
    if "(" in header and ")" in header:
        inside = header.split("(", 1)[1].split(")", 1)[0]
        for tok in inside.replace(",", " ").split():
            try:
                dims.append(int(tok))
            except ValueError:
                pass

    vals: List[float] = []
    for line in lines[start + 1:]:
        if line.startswith("##$") or line.startswith("##@"):
            break
        line = line.strip()
        if not line:
            continue
        for tok in line.replace(",", " ").split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass

    arr = np.asarray(vals, dtype=np.float64)
    return arr, dims


# ---------- CFL I/O ----------

def _write_hdr(path: Path, dims: List[int]) -> None:
    with open(path, "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(int(d)) for d in dims) + "\n")


def write_cfl(base: Path, arr: np.ndarray, dims: Optional[List[int]] = None) -> None:
    """
    Write BART .cfl/.hdr pair for complex64 array.
    """
    base = Path(base)
    cfl = base.with_suffix(".cfl")
    hdr = base.with_suffix(".hdr")

    arr_c = np.asarray(arr, dtype=np.complex64, order="F")
    if dims is None:
        dims = list(arr_c.shape)
    prod = 1
    for d in dims:
        prod *= int(d)
    if prod != arr_c.size:
        raise ValueError(f"dims product {prod} does not match array size {arr_c.size}")

    _write_hdr(hdr, dims)
    # BART stores complex as interleaved float32
    arr_float = np.empty(arr_c.size * 2, dtype=np.float32)
    arr_float[0::2] = arr_c.real.ravel(order="F")
    arr_float[1::2] = arr_c.imag.ravel(order="F")
    arr_float.tofile(cfl)


def read_cfl(base: Path) -> np.ndarray:
    base = Path(base)
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")
    with open(hdr, "r") as f:
        first = f.readline()
        if not first.startswith("#"):
            raise ValueError("Invalid BART .hdr (missing # Dimensions)")
        dims_line = f.readline()
    dims = [int(x) for x in dims_line.split()]
    data = np.fromfile(cfl, dtype=np.float32)
    if data.size % 2 != 0:
        data = data[:-1]
    cpx = data.view(np.complex64)
    prod = 1
    for d in dims:
        prod *= d
    if cpx.size != prod:
        raise ValueError(f".cfl data size {cpx.size} does not match header product {prod}")
    return cpx.reshape(dims, order="F")


# ---------- Bruker FID -> k-space (RO, Spokes, Coils) ----------

def load_bruker_kspace(
    series_dir: Path,
    matrix_ro_hint: int,
    coils: Optional[int] = None,
    fid_dtype: str = "int32",
    fid_endian: str = "little",
) -> np.ndarray:
    """
    Load Bruker FID and reshape to (RO, Spokes, Coils).

    Strategy:
    - infer number of coils from PVM_EncNReceivers if not provided.
    - assume all extra dims (echo, rep, avg, slice) are flattened into spokes.
    - choose stored_ro as a divisor of per_coil_total, preferring >= matrix_ro_hint
      and from a list of typical padded block sizes.
    - trim stored_ro to matrix_ro_hint (RO) if stored_ro >= matrix_ro_hint.
    """
    series_dir = Path(series_dir)
    fid_path = series_dir / "fid"
    if not fid_path.exists():
        raise FileNotFoundError(f"No fid found in {series_dir}")

    method = _read_text_kv(series_dir / "method")
    acqp = _read_text_kv(series_dir / "acqp")

    if coils is None:
        coils = _get_int_from_headers(["PVM_EncNReceivers"], [method, acqp])
    if coils is None or coils <= 0:
        coils = 1

    # dtype & endian
    dtype_map = {"int16": np.int16, "int32": np.int32, "float32": np.float32, "float64": np.float64}
    if fid_dtype not in dtype_map:
        raise ValueError("fid_dtype must be one of int16,int32,float32,float64")
    dt = dtype_map[fid_dtype]

    raw = np.fromfile(fid_path, dtype=dt)
    if fid_endian == "big":
        raw = raw.byteswap().newbyteorder()
    if raw.size % 2 != 0:
        raw = raw[:-1]
    cpx = raw.astype(np.float32).view(np.complex64)
    total = cpx.size
    dbg("total complex samples:", total, "coils:", coils)

    if total % coils != 0:
        raise ValueError(f"Total complex samples {total} not divisible by coils={coils}")

    per_coil_total = total // coils

    # pick stored_ro and spokes
    def pick_block_and_spokes(per_coil_total: int, readout_hint: int) -> Tuple[int, int]:
        # candidate RO block sizes (typical Bruker padded lengths)
        BLOCKS = [
            128, 160, 192, 200, 224, 240, 256, 288, 320, 352, 384, 400, 416, 420, 432, 448, 480, 496, 512,
            544, 560, 576, 608, 640, 672, 704, 736, 768, 800, 832, 896, 960, 992, 1024, 1152, 1280, 1536, 2048,
        ]
        # perfect match with requested RO
        if readout_hint > 0 and per_coil_total % readout_hint == 0:
            return readout_hint, per_coil_total // readout_hint
        # otherwise try padded RO >= hint
        for b in BLOCKS:
            if b >= readout_hint and per_coil_total % b == 0:
                return b, per_coil_total // b
        # fall back to generic factor near sqrt
        s = int(round(per_coil_total ** 0.5))
        for d in range(0, s + 1):
            for cand in (s + d, s - d):
                if cand > 0 and per_coil_total % cand == 0:
                    return cand, per_coil_total // cand
        raise ValueError("Could not factor per_coil_total into (stored_ro, spokes).")

    stored_ro, spokes = pick_block_and_spokes(per_coil_total, matrix_ro_hint)
    dbg("stored_ro:", stored_ro, "spokes:", spokes)

    if stored_ro * spokes * coils != total:
        raise ValueError("Internal factoring error: stored_ro*spokes*coils != total samples")

    ksp_blk = cpx.reshape((stored_ro, spokes, coils), order="F")

    ro = matrix_ro_hint
    if stored_ro >= ro:
        ksp = ksp_blk[:ro, :, :]
        dbg(f"trimmed RO from {stored_ro} to {ro}")
    else:
        # unlikely but handle
        ksp = np.zeros((ro, spokes, coils), dtype=np.complex64)
        ksp[:stored_ro, :, :] = ksp_blk
        dbg(f"padded RO from {stored_ro} to {ro}")

    dbg("final k-space shape:", ksp.shape)
    return ksp


# ---------- Golden-angle / Kronecker trajectory synthesis ----------

def _fib_sequence_upto(n: int) -> List[int]:
    fibs = [1, 2]
    while fibs[-1] < n:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


def _kronecker_indices(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate of Bruker kronecker_dir / fibonacci lattice scheme.
    """
    fibs = _fib_sequence_upto(N)
    M = fibs[-1]
    if len(fibs) >= 3:
        q1 = fibs[-2]
        q2 = fibs[-3]
    elif len(fibs) == 2:
        q1, q2 = fibs[1], fibs[0]
    else:
        q1, q2 = 1, 1

    j = np.arange(N, dtype=np.int64) % M
    u = ((j * q1) % M + 0.5) / float(M)
    v = ((j * q2) % M + 0.5) / float(M)
    return u.astype(np.float64), v.astype(np.float64)


def _uv_to_dir(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map [0,1]^2 -> points on unit sphere (Bruker uv_to_dir).
    """
    z = 1.0 - 2.0 * u
    r2 = 1.0 - z * z
    r2 = np.clip(r2, 0.0, None)
    r = np.sqrt(r2)
    az = 2.0 * math.pi * v
    dx = r * np.cos(az)
    dy = r * np.sin(az)
    dz = z
    return dx, dy, dz


def _traj_kronecker(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v = _kronecker_indices(N)
    return _uv_to_dir(u, v)


def _traj_linz_ga(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bruker linZ_ga_dir: golden-angle increment in azimuth, linear in z.
    """
    phi_inc = (math.sqrt(5.0) - 1.0) * math.pi
    i = np.arange(N, dtype=np.float64)
    z = 1.0 - 2.0 * ((i + 0.5) / float(N))
    r2 = 1.0 - z * z
    r2 = np.clip(r2, 0.0, None)
    r = np.sqrt(r2)
    az = (i * phi_inc) % (2.0 * math.pi)
    dx = r * np.cos(az)
    dy = r * np.sin(az)
    dz = z
    return dx, dy, dz


def _traj_from_ga_settings(series_dir: Path, ro: int, sp_total: int, override_mode: Optional[str] = None) -> np.ndarray:
    """
    Build synthetic 3D trajectory (3,RO,Spokes) from GA / Kronecker settings in method file.

    We do NOT use PVM_TrajKx/Ky/Kz arrays; instead we replicate the logic of SetProj3D
    for GA_Traj_Kronecker and GA_Traj_LinZ_GA and simply treat the acquisition as a
    continuous run of N = sp_total spokes.
    """
    method_txt = _read_text(series_dir / "method")
    method_kv = _read_text_kv(series_dir / "method")

    mode = None
    raw_mode = method_kv.get("GA_Mode", "")
    if override_mode is not None:
        mode = override_mode.lower()
    else:
        # Try to guess from text of GA_Mode line (if it includes symbolic names)
        if "kronecker" in raw_mode.lower():
            mode = "kronecker"
        elif "linz" in raw_mode.lower() or "lin_z" in raw_mode.lower():
            mode = "linz"
        elif "ga_traj_kronecker" in method_txt.lower():
            mode = "kronecker"
        elif "ga_traj_linz_ga" in method_txt.lower():
            mode = "linz"

    if mode is None:
        raise RuntimeError(
            "Cannot infer GA trajectory mode from method (GA_Mode). "
            "You can hard-code override_mode in _traj_from_ga_settings if needed."
        )

    N = sp_total  # treat entire run as one continuous sequence
    if mode == "kronecker":
        dx, dy, dz = _traj_kronecker(N)
    elif mode == "linz":
        dx, dy, dz = _traj_linz_ga(N)
    else:
        raise RuntimeError(f"Unsupported GA_MODE '{mode}' for synthetic trajectory.")

    # Turn direction vectors into k-space coordinates per readout sample.
    # Here we simply scale to k-space radius spanning [-0.5,0.5] in each cartesian axis.
    # We replicate along the readout dimension: k(t) = dir * ( (t - RO/2) / RO )
    t = np.linspace(-0.5, 0.5, ro, dtype=np.float64)  # abstract k-radius
    kx = np.outer(t, dx)  # (RO,Spokes)
    ky = np.outer(t, dy)
    kz = np.outer(t, dz)
    traj = np.stack([kx, ky, kz], axis=0).astype(np.float32)
    return traj  # (3,RO,Spokes)


# ---------- DCF (Pipe-style NumPy) ----------

def _normalize_traj_to_grid(traj: np.ndarray, grid_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ro, sp = traj.shape[1], traj.shape[2]
    kx, ky, kz = traj[0], traj[1], traj[2]

    def robust_minmax(a: np.ndarray) -> Tuple[float, float]:
        lo = float(np.nanpercentile(a, 0.5))
        hi = float(np.nanpercentile(a, 99.5))
        if not np.isfinite(hi - lo) or hi <= lo:
            hi = lo + 1e-3
        return lo, hi

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
    """
    Simple Pipe-style DCF estimation on a Cartesian grid.
    """
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


# ---------- Python adjoint gridding (Kaiser–Bessel) ----------

def _kaiser_bessel(u: np.ndarray, width: float, beta: float) -> np.ndarray:
    x = np.abs(u)
    out = np.zeros_like(x, dtype=np.float32)
    half = width / 2.0
    m = x <= half
    if not np.any(m):
        return out
    from numpy import i0
    t = np.sqrt(1.0 - (x[m] / half) ** 2)
    out[m] = (i0(beta * t) / i0(beta)).astype(np.float32)
    return out


def _deapod_1d(N: int, os: float) -> np.ndarray:
    k = (np.arange(int(N * os)) - (N * os) / 2.0) / (N * os)
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
    ro, sp, nc = ksp.shape
    NX, NY, NZ = grid_shape
    NXg = int(round(NX * oversamp))
    NYg = int(round(NY * oversamp))
    NZg = int(round(NZ * oversamp))

    kx, ky, kz = traj[0], traj[1], traj[2]
    kmax = 0.5 * float(max(NX, NY, NZ))
    gx = (kx / (2.0 * kmax) + 0.5) * NXg
    gy = (ky / (2.0 * kmax) + 0.5) * NYg
    gz = (kz / (2.0 * kmax) + 0.5) * NZg

    deapx = _deapod_1d(NX, oversamp)
    deapy = _deapod_1d(NY, oversamp)
    deapz = _deapod_1d(NZ, oversamp)

    img_grid = np.zeros((NXg, NYg, NZg, nc), dtype=np.complex64)
    hw = int(math.ceil(kb_width / 2.0))
    w = np.ones((ro, sp), dtype=np.float32) if dcf is None else dcf.astype(np.float32)

    for s in range(sp):
        cx = np.floor(gx[:, s]).astype(int)
        cy = np.floor(gy[:, s]).astype(int)
        cz = np.floor(gz[:, s]).astype(int)
        for t in range(ro):
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
            for c in range(nc):
                val = ksp[t, s, c]
                img_grid[np.ix_(xr2, yr2, zr2, [c])] += (wxyz2[..., None] * val).astype(np.complex64)

    img = np.fft.ifftn(np.fft.ifftshift(img_grid, axes=(0, 1, 2)), axes=(0, 1, 2))
    img = np.fft.fftshift(img, axes=(0, 1, 2))
    x0 = (NXg - NX) // 2
    y0 = (NYg - NY) // 2
    z0 = (NZg - NZ) // 2
    img = img[x0:x0 + NX, y0:y0 + NY, z0:z0 + NZ, :]
    img *= deapx[:NX, None, None, None]
    img *= deapy[None, :NY, None, None]
    img *= deapz[None, None, :NZ, None]
    return img.astype(np.complex64)


# ---------- BART helpers ----------

def bart_exists() -> bool:
    return shutil.which("bart") is not None


def _bart_path() -> str:
    bart = shutil.which("bart")
    if bart is None:
        raise RuntimeError("BART not found in PATH")
    return bart


def _bart_supports_gpu() -> bool:
    global BART_GPU_AVAILABLE
    if BART_GPU_AVAILABLE is not None:
        return BART_GPU_AVAILABLE
    bart = _bart_path()
    try:
        subprocess.run([bart, "nufft", "-g", "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        BART_GPU_AVAILABLE = True
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or b"").decode(errors="ignore")
        if "compiled without GPU" in msg or "invalid option" in msg or "unknown option" in msg:
            BART_GPU_AVAILABLE = False
        else:
            BART_GPU_AVAILABLE = False
    except Exception:
        BART_GPU_AVAILABLE = False
    return BART_GPU_AVAILABLE


def _run_bart_tool(tool: str, args: List[str], gpu: bool) -> None:
    bart = _bart_path()
    global BART_GPU_AVAILABLE
    cmd: List[str]
    if gpu and _bart_supports_gpu():
        cmd = [bart, tool, "-g"] + args
        print("[bart]", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
            return
        except Exception:
            print("[warn] BART GPU attempt failed; retrying on CPU for this run.")
            BART_GPU_AVAILABLE = False
    cmd = [bart, tool] + args
    print("[bart]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_bart(cmd: List[str], gpu: bool = False) -> None:
    if not cmd:
        raise ValueError("Empty BART command")
    _run_bart_tool(cmd[0], cmd[1:], gpu=gpu)


def ksp_to_bart_noncart(ksp: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Map ksp (RO,Spokes,Coils) -> BART dims [Coils,1,1,1,1,1,1,1,1,1,RO,Spokes,1,1,1,1]
    """
    ro, sp, nc = ksp.shape
    arr = np.asfortranarray(ksp.astype(np.complex64))
    arr = np.transpose(arr, (2, 0, 1))  # (Coils,RO,Spokes)
    dims = [1] * 16
    dims[0] = nc
    dims[10] = ro
    dims[11] = sp
    return arr.reshape(dims, order="F"), dims


def traj_to_bart_noncart(traj: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Map traj (3,RO,Spokes) -> BART dims [3,1,1,1,1,1,1,1,1,1,RO,Spokes,1,1,1,1]
    """
    _, ro, sp = traj.shape
    arr = np.asfortranarray(traj.astype(np.complex64))
    dims = [1] * 16
    dims[0] = 3
    dims[10] = ro
    dims[11] = sp
    return arr.reshape(dims, order="F"), dims


def recon_adjoint_bart(
    traj: np.ndarray,
    ksp: np.ndarray,
    dcf: Optional[np.ndarray],
    matrix: Tuple[int, int, int],
    out_base: Path,
    combine: str,
    gpu: bool,
) -> None:
    NX, NY, NZ = matrix
    ro, sp, nc = ksp.shape
    if dcf is not None:
        ksp = ksp * dcf[..., None]

    ksp16, kspdims = ksp_to_bart_noncart(ksp)
    traj16, trajdims = traj_to_bart_noncart(traj)

    ksp_base = out_base.with_name(out_base.name + "_ksp")
    traj_base = out_base.with_name(out_base.name + "_traj")
    write_cfl(ksp_base, ksp16, kspdims)
    write_cfl(traj_base, traj16, trajdims)

    coil_base = out_base.with_name(out_base.name + "_coil")
    run_bart(["nufft", "-a", "-t", str(traj_base), str(ksp_base), str(coil_base)], gpu=gpu)

    if combine.lower() == "sos":
        run_bart(["rss", "8", str(coil_base), str(out_base)], gpu=gpu)
    elif combine.lower() == "sens":
        maps = out_base.with_name(out_base.name + "_maps")
        run_bart(["ecalib", str(coil_base), str(maps)], gpu=gpu)
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
) -> None:
    NX, NY, NZ = matrix
    if dcf is not None:
        ksp = ksp * dcf[..., None]
    coil_img = adjoint_grid_numpy(traj, ksp, dcf=None, grid_shape=(NX, NY, NZ))
    coil_base = out_base.with_name(out_base.name + "_coil_py")
    # dims: [NX,NY,NZ,NC] -> BART dims [NX,NY,NZ,NC,1,...]
    NXg, NYg, NZg, nc = coil_img.shape
    dims = [NXg, NYg, NZg, nc] + [1] * 12
    write_cfl(coil_base, coil_img, dims)
    if combine.lower() == "sos":
        rss = np.sqrt(np.sum(np.abs(coil_img) ** 2, axis=3)).astype(np.complex64)
        out_dims = [NXg, NYg, NZg] + [1] * 13
        write_cfl(out_base, rss, out_dims)
    elif combine.lower() == "sens":
        maps = out_base.with_name(out_base.name + "_maps_py")
        run_bart(["ecalib", str(coil_base), str(maps)], gpu=False)
        run_bart(["pics", "-S", str(coil_base), str(maps), str(out_base)], gpu=False)
    else:
        raise ValueError("combine must be sos|sens")


# ---------- Frame binning ----------

def frame_starts(total_spokes: int, spokes_per_frame: int, frame_shift: Optional[int]) -> Iterable[int]:
    step = frame_shift if frame_shift and frame_shift > 0 else spokes_per_frame
    for s in range(0, total_spokes - spokes_per_frame + 1, step):
        yield s


# ---------- CLI ----------

def main():
    global DEBUG

    ap = argparse.ArgumentParser(description="Bruker 3D radial recon using BART/NumPy with GA/Kronecker fallback trajectory.")
    ap.add_argument("--series", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--matrix", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))
    ap.add_argument("--combine", type=str, default="sos", help="sos|sens")
    ap.add_argument("--dcf", type=str, default="none", help="none | pipe:N (NumPy Pipe-style)")
    ap.add_argument("--gpu", action="store_true", help="Use BART GPU if available")
    ap.add_argument("--force-python-adjoint", action="store_true", help="Force pure-Python adjoint gridding (no BART nufft)")
    ap.add_argument("--debug", action="store_true")
    # overrides for FID layout
    ap.add_argument("--readout", type=int, default=None, help="Optional override for RO (otherwise use matrix NX)")
    ap.add_argument("--coils", type=int, default=None)
    ap.add_argument("--fid-dtype", type=str, default="int32")
    ap.add_argument("--fid-endian", type=str, default="little")
    # temporal binning
    ap.add_argument("--spokes-per-frame", type=int, default=None, help="Spokes per frame (sliding window)")
    ap.add_argument("--frame-shift", type=int, default=None, help="Spoke shift between frames (default = spokes-per-frame)")
    ap.add_argument("--test-volumes", type=int, default=None, help="Only reconstruct first N frames")
    ap.add_argument("--export-nifti", action="store_true", help="Run bart toimg on each frame")
    args = ap.parse_args()

    DEBUG = args.debug

    if not bart_exists():
        print("ERROR: BART not found on PATH.", file=sys.stderr)
        # Python adjoint path still uses BART for toimg/sens maps; but we allow continuing if user doesn't ask for export/sens.
        if not args.force_python_adjoint and args.export_nifti:
            sys.exit(1)

    series_dir: Path = args.series
    out_base: Path = args.out
    NX, NY, NZ = args.matrix
    ro = args.readout or NX

    # k-space
    ksp = load_bruker_kspace(
        series_dir,
        matrix_ro_hint=ro,
        coils=args.coils,
        fid_dtype=args.fid_dtype,
        fid_endian=args.fid_endian,
    )
    ro2, sp_total, nc = ksp.shape
    if ro2 != ro:
        print(f"[warn] k-space RO={ro2} but requested matrix NX={ro}")
        ro = ro2
    print(f"[info] Loaded k-space: RO={ro2}, Spokes={sp_total}, Coils={nc}")

    # trajectory: 1) traj file; 2) GA/Kronecker synthetic; else error
    traj_path = series_dir / "traj"
    if traj_path.exists():
        # We accept binary float32/float64 or ASCII, flattened or shaped.
        raw = np.fromfile(traj_path, dtype=np.float32)
        if raw.size == 0:
            # maybe ASCII
            try:
                raw_txt = traj_path.read_text()
                vals = [float(t) for t in raw_txt.split()]
                raw = np.asarray(vals, dtype=np.float32)
            except Exception:
                raise RuntimeError("traj file exists but could not be read as binary or ASCII float list.")
        expected = 3 * ro * sp_total
        if raw.size < expected:
            print(f"[warn] traj has only {raw.size} floats; expected {expected}; zero-padding.")
            raw = np.pad(raw, (0, expected - raw.size))
        elif raw.size > expected:
            print(f"[warn] traj has {raw.size} floats; expected {expected}; trimming.")
            raw = raw[:expected]
        traj = raw.reshape((3, ro, sp_total), order="F")
        print(f"[info] Loaded traj from file: shape={traj.shape}")
    else:
        print("[info] No traj file; building synthetic GA/Kronecker trajectory from method GA_* parameters.")
        traj = _traj_from_ga_settings(series_dir, ro, sp_total)
        print(f"[info] Synthetic GA/Kronecker traj built: shape={traj.shape}")

    # spokes-per-frame & sliding window
    spokes_per_frame = args.spokes_per_frame
    if spokes_per_frame is None:
        spokes_per_frame = min(sp_total, 1000)
        print(f"[warn] --spokes-per-frame not set; defaulting to {spokes_per_frame}")
    frame_shift = args.frame_shift or spokes_per_frame
    if frame_shift <= 0:
        frame_shift = spokes_per_frame

    starts = list(frame_starts(sp_total, spokes_per_frame, frame_shift))
    if args.test_volumes is not None:
        starts = starts[: max(0, int(args.test_volumes))]
    nframes = len(starts)
    if nframes == 0:
        raise ValueError("No frames to reconstruct with chosen (spokes-per-frame, frame-shift).")
    print(f"[info] Sliding-window frames: {nframes} (spf={spokes_per_frame}, shift={frame_shift})")

    for fi, s0 in enumerate(starts, 1):
        s1 = s0 + spokes_per_frame
        if s1 > sp_total:
            print(f"[warn] Skipping partial frame at spokes [{s0}:{s1}]")
            break
        ksp_f = ksp[:, s0:s1, :]
        traj_f = traj[:, :, s0:s1]
        # DCF
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
        if args.force_python_adjoint:
            recon_adjoint_python(traj_f, ksp_f, dcf, (NX, NY, NZ), vol_base, combine=args.combine)
        else:
            try:
                recon_adjoint_bart(traj_f, ksp_f, dcf, (NX, NY, NZ), vol_base, combine=args.combine, gpu=args.gpu)
            except Exception as e:
                print(f"[warn] BART adjoint failed for frame {fi} ({e}); falling back to NumPy adjoint.")
                recon_adjoint_python(traj_f, ksp_f, dcf, (NX, NY, NZ), vol_base, combine=args.combine)
        if args.export_nifti:
            run_bart(["toimg", str(vol_base), str(vol_base)], gpu=False)
        print(f"[info] Frame {fi}/{nframes} done -> {vol_base}")

    print("[info] All requested frames complete.")


if __name__ == "__main__":
    main()
