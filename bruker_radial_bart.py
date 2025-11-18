#!/usr/bin/env python3
"""
bruker_radial_bart.py

Mode B design:
- Primary recon engine: BART adjoint NUFFT (nufft -a).
- Fallback: pure NumPy adjoint gridding if BART fails or --force-python-adjoint.

Trajectory logic:
- If $series/traj exists, read and use it (shape 3 x RO x Spokes).
- Else require --traj-mode {linear_z,kron} and synthesize trajectory using GA math
  equivalent to the Bruker golden_angle_UTE3D sequence snippet provided.

Temporal binning:
- --spokes-per-frame and optional --frame-shift.
- --test-volumes K to reconstruct only first K frames (handy for QC).

Quick QC:
- Use --test-volumes 1 to get a single NIfTI from the first frame.
NIfTI export is *always* performed via BART toimg (no --export-nifti flag).
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


def dbg(*a):
    if DEBUG:
        print("[debug]", *a)


# ---------- CFL I/O ----------


def _write_hdr(path: Path, dims: List[int]):
    path = Path(path)
    with open(path, "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in dims) + "\n")


def _read_hdr(path: Path) -> List[int]:
    path = Path(path)
    with open(path, "r") as f:
        first = f.readline()
        if not first.startswith("#"):
            raise ValueError(f"{path} is not a valid BART hdr (missing '# Dimensions').")
        line = f.readline().strip()
    dims = [int(x) for x in line.split()]
    if len(dims) < 16:
        dims = dims + [1] * (16 - len(dims))
    return dims[:16]


def write_cfl(base: Path, arr: np.ndarray, dims: Optional[List[int]] = None):
    """
    Write complex array in BART .cfl/.hdr format.

    arr will be converted to complex64 and Fortran order before writing.
    If dims is None, uses arr.shape padded/truncated to 16 dims.
    """
    base = Path(base)
    arr = np.asarray(arr, dtype=np.complex64, order="F")
    if dims is None:
        shape = list(arr.shape)
        if len(shape) < 16:
            shape = shape + [1] * (16 - len(shape))
        dims = shape[:16]
    prod_dims = int(np.prod(dims))
    if prod_dims != arr.size:
        raise ValueError(f"dims product {prod_dims} != array size {arr.size}")
    _write_hdr(base.with_suffix(".hdr"), dims)
    # BART expects interleaved float32 real/imag
    arr.view(np.float32).tofile(base.with_suffix(".cfl"))


def read_cfl(base: Path) -> np.ndarray:
    base = Path(base)
    dims = _read_hdr(base.with_suffix(".hdr"))
    prod_dims = int(np.prod(dims))
    data = np.fromfile(base.with_suffix(".cfl"), dtype=np.float32)
    if data.size != 2 * prod_dims:
        raise ValueError(f"cfl file size mismatch: got {data.size} floats, expected {2 * prod_dims}")
    cpx = data.view(np.complex64)
    arr = cpx.reshape(dims, order="F")
    return arr


# ---------- Bruker text helpers ----------


def _read_text_kv(path: Path) -> Dict[str, str]:
    """Very simple Bruker key/value reader for acqp/method."""
    d: Dict[str, str] = {}
    path = Path(path)
    if not path.exists():
        return d
    key = None
    buf: List[str] = []
    for line in path.read_text().splitlines():
        if line.startswith("##$"):
            if key is not None:
                d[key] = " ".join(buf).strip()
            parts = line[3:].split("=", 1)
            key = parts[0].strip()
            buf = [parts[1].strip()] if len(parts) > 1 else []
        elif line.startswith("#") or line.startswith("$$"):
            continue
        else:
            if key is not None:
                buf.append(line.strip())
    if key is not None:
        d[key] = " ".join(buf).strip()
    return d


def _get_int_from_headers(keys: List[str], dicts: List[Dict[str, str]]) -> Optional[int]:
    for k in keys:
        for d in dicts:
            if k in d:
                try:
                    tok = d[k].split()[0].replace("<", "").replace(">", "")
                    return int(tok)
                except Exception:
                    continue
    return None


def _parse_acq_size(method: Dict[str, str], acqp: Dict[str, str]) -> Optional[Tuple[int, int, int]]:
    cand_keys = ["PVM_EncMatrix", "PVM_Matrix", "ACQ_size"]
    for k in cand_keys:
        src = method if k in method else acqp if k in acqp else None
        if src is None:
            continue
        try:
            toks = src[k].split()
            vals = [int(t) for t in toks[:3]]
            return tuple(vals)
        except Exception:
            continue
    return None


# ---------- Bruker FID loader ----------


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
    Load Bruker FID and reshape to (RO, Spokes, Coils).

    Tries to infer RO/Spokes/Coils from acqp/method, trimming padded RO if needed.
    """
    series_dir = Path(series_dir)
    fid_path = series_dir / "fid"
    if not fid_path.exists():
        raise FileNotFoundError(f"No fid in {series_dir}")

    method = _read_text_kv(series_dir / "method")
    acqp = _read_text_kv(series_dir / "acqp")

    # endian / dtype from headers unless overridden
    if "BYTORDA" in acqp and "big" in acqp["BYTORDA"].lower():
        fid_endian = "big"
    if "ACQ_word_size" in acqp:
        if "16" in acqp["ACQ_word_size"]:
            fid_dtype = "int16"
        elif "32" in acqp["ACQ_word_size"]:
            fid_dtype = "int32"

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
    dbg("total complex samples:", total, "(dtype:", fid_dtype, ", endian:", fid_endian, ")")

    if readout is None:
        acq_size = _parse_acq_size(method, acqp)
        if acq_size:
            readout = acq_size[0]
    if readout is None and matrix_ro_hint is not None:
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
        "echoes": _get_int_from_headers(["NECHOES", "ACQ_n_echo_images", "PVM_NEchoImages"], [method, acqp]) or 1,
        "reps": _get_int_from_headers(["PVM_NRepetitions", "NR"], [method, acqp]) or 1,
        "averages": _get_int_from_headers(["PVM_NAverages", "NA"], [method, acqp]) or 1,
        "slices": _get_int_from_headers(["NSLICES", "PVM_SPackArrNSlices"], [method, acqp]) or 1,
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

    def pick_block_and_spokes(per_coil_total: int, readout_hint: Optional[int], spokes_hint: Optional[int]) -> Tuple[int, int]:
        if spokes_hint and spokes_hint > 0 and per_coil_total % spokes_hint == 0:
            return per_coil_total // spokes_hint, spokes_hint
        BLOCKS = [
            128, 160, 192, 200, 224, 240, 256, 288, 320, 352, 384, 400, 416, 420, 432, 448, 480, 496, 512,
            544, 560, 576, 608, 640, 672, 704, 736, 768, 800, 832, 896, 960, 992, 1024, 1152, 1280, 1536, 2048,
        ]
        if readout_hint and per_coil_total % readout_hint == 0:
            return readout_hint, per_coil_total // readout_hint
        for b in [x for x in BLOCKS if not readout_hint or x >= readout_hint]:
            if per_coil_total % b == 0:
                return b, per_coil_total // b
        s = int(round(per_coil_total ** 0.5))
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


# ---------- Trajectory handling ----------


def _read_bruker_traj_file(series_dir: Path, ro: int, spokes: int) -> Optional[np.ndarray]:
    """
    Read $series/traj if present.

    Format: assume binary float32 or float64 or ASCII with 3*RO*Spokes entries.
    """
    tpath = Path(series_dir) / "traj"
    if not tpath.exists():
        return None
    data = tpath.read_bytes()
    # try float32 / float64
    for dt in (np.float32, np.float64):
        try:
            arr = np.frombuffer(data, dtype=dt)
            if arr.size == 3 * ro * spokes:
                arr = arr.astype(np.float32)
                return arr.reshape(3, ro, spokes)
        except Exception:
            pass
    # fallback: ASCII
    try:
        txt = data.decode("ascii", errors="ignore").split()
        vals = np.array([float(x) for x in txt], dtype=np.float32)
        if vals.size >= 3 * ro * spokes:
            vals = vals[: 3 * ro * spokes]
            return vals.reshape(3, ro, spokes)
    except Exception:
        pass
    raise RuntimeError(f"Could not interpret {tpath} as a 3 x RO x Spokes trajectory")


# ---- GA/Kronecker helpers from Bruker C math ----


def _fib_closest_ge(n: int) -> int:
    if n <= 1:
        return 1
    a, b = 1, 1
    while b < n:
        a, b = b, a + b
    return b


def _fib_prev(fk: int) -> int:
    if fk <= 1:
        return 1
    a, b = 1, 1
    while b < fk:
        a, b = b, a + b
    return a


def _fib_prev2(fk: int) -> int:
    p = _fib_prev(fk)
    return _fib_prev(p)


def _uv_to_dir(u: float, v: float) -> Tuple[float, float, float]:
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


def _kronecker_dir(i: int, N: int) -> Tuple[float, float, float]:
    M = _fib_closest_ge(N)
    q1 = _fib_prev(M)
    q2 = _fib_prev2(M)
    j = i % M
    u = ((j * q1) % M + 0.5) / float(M)
    v = ((j * q2) % M + 0.5) / float(M)
    return _uv_to_dir(u, v)


def _linZ_ga_dir(i: int, N: int) -> Tuple[float, float, float]:
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


def build_synthetic_traj(
    series_dir: Path,
    ro: int,
    spokes: int,
    mode: str,
) -> np.ndarray:
    """
    Build synthetic trajectory from Bruker-style GA parameters.

    We use the exact direction math from the provided C snippet:
      - kronecker_dir
      - linZ_ga_dir  (LinZ_GA)
    Then we lay readout samples along that direction.

    We do not currently scale by gr/gp/gs; those are effectively 1.0 here.
    """
    N = spokes
    kx = np.zeros((ro, spokes), dtype=np.float32)
    ky = np.zeros_like(kx)
    kz = np.zeros_like(kx)

    # radial samples along the readout, symmetric in k-space
    kmax = 0.5  # relative units
    rline = np.linspace(-kmax, kmax, ro, dtype=np.float32)

    for i in range(N):
        if mode == "kron":
            dx, dy, dz = _kronecker_dir(i, N)
        elif mode == "linear_z":
            dx, dy, dz = _linZ_ga_dir(i, N)
        else:
            raise ValueError("traj_mode must be 'linear_z' or 'kron'")
        kx[:, i] = rline * float(dx)
        ky[:, i] = rline * float(dy)
        kz[:, i] = rline * float(dz)

    traj = np.stack([kx, ky, kz], axis=0)  # (3,ro,spokes)
    return traj


# ---------- DCF (pipe-style, NumPy) ----------


def _normalize_traj_to_grid(traj: np.ndarray, grid_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ro, sp = traj.shape[1], traj.shape[2]
    kx, ky, kz = traj[0], traj[1], traj[2]

    def robust_minmax(a):
        lo = float(np.nanpercentile(a, 0.5))
        hi = float(np.nanpercentile(a, 99.5))
        if not np.isfinite(hi - lo) or hi <= lo:
            hi = lo + 1e-3
        return lo, hi

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


# ---------- BART wrappers ----------


def bart_exists() -> bool:
    return shutil.which("bart") is not None


def _bart_path() -> str:
    bart = shutil.which("bart")
    if not bart:
        raise RuntimeError("BART not found in PATH")
    return bart


BART_GPU_AVAILABLE: Optional[bool] = None


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
        if ("compiled without GPU" in msg) or ("invalid option -- 'g'" in msg) or ("unknown option g" in msg):
            BART_GPU_AVAILABLE = False
        else:
            BART_GPU_AVAILABLE = True
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
                print("[warn] BART GPU attempt failed; using CPU for remainder of run.")
                BART_GPU_AVAILABLE = False
    cmd = [bart, tool] + args
    print("[bart]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_bart(cmd: List[str], gpu: bool = False):
    if not cmd:
        raise ValueError("Empty BART command")
    _run_bart(cmd[0], cmd[1:], gpu=gpu)


# ---------- Layout helpers: (RO,Sp,Coils)/(3,RO,Sp) -> BART dims ----------


def ksp_to_bart_noncart(ksp: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Map ksp (RO,Sp,Coils) to BART noncart layout.

    We use dims: [1,1,1,Coils,1,1,1,1,1,1,RO,Sp,1,1,1,1].
    """
    ro, sp, nc = ksp.shape
    ksp_t = np.asarray(ksp, dtype=np.complex64).transpose(2, 0, 1)  # (nc,ro,sp)
    dims = [1] * 16
    dims[3] = nc
    dims[10] = ro
    dims[11] = sp
    arr16 = ksp_t.reshape(dims, order="F")
    return arr16, dims


def traj_to_bart_noncart(traj: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Map traj (3,RO,Sp) to BART layout dims: [3,1,1,1,1,1,1,1,1,1,RO,Sp,1,1,1,1].
    """
    _, ro, sp = traj.shape
    t = np.asarray(traj, dtype=np.complex64)  # (3,ro,sp)
    dims = [1] * 16
    dims[0] = 3
    dims[10] = ro
    dims[11] = sp
    arr16 = t.reshape(dims, order="F")
    return arr16, dims


# ---------- Pure NumPy adjoint (slow but robust) ----------


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
    ro, sp, nc = ksp.shape
    NX, NY, NZ = grid_shape
    NXg, NYg, NZg = int(round(NX * oversamp)), int(round(NY * oversamp)), int(round(NZ * oversamp))
    kx, ky, kz = traj[0], traj[1], traj[2]
    kmax = 0.5 * max(NX, NY, NZ)
    gx = (kx / (2 * kmax) + 0.5) * NXg
    gy = (ky / (2 * kmax) + 0.5) * NYg
    gz = (kz / (2 * kmax) + 0.5) * NZg
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
            xr2, yr2, zr2 = xr[xsel], yr[ysel], zr[zsel]
            wxyz2 = wxyz[xsel][:, ysel][:, :, zsel]
            for c in range(nc):
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
    dims = [NX, NY, NZ, coil_img.shape[3]] + [1] * 12
    write_cfl(coil_base, coil_img, dims)
    if combine.lower() == "sos":
        rss = np.sqrt(np.sum(np.abs(coil_img) ** 2, axis=3)).astype(np.complex64)
        img_dims = [NX, NY, NZ] + [1] * 13
        write_cfl(out_base, rss, img_dims)
    elif combine.lower() == "sens":
        maps = out_base.with_name(out_base.name + "_maps_py")
        estimate_sens_maps(coil_base, maps, gpu=False)
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
    ap = argparse.ArgumentParser(
        description="Bruker 3D radial reconstruction using BART (Mode B: BART primary, Python fallback)"
    )
    ap.add_argument("--series", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--matrix", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))
    ap.add_argument("--dcf", type=str, default="none", help="none | pipe:N")
    ap.add_argument("--combine", type=str, default="sos", help="sos|sens")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--debug", action="store_true")
    # k-space overrides
    ap.add_argument("--readout", type=int, default=None)
    ap.add_argument("--spokes", type=int, default=None)
    ap.add_argument("--coils", type=int, default=None)
    ap.add_argument("--fid-dtype", type=str, default=None)
    ap.add_argument("--fid-endian", type=str, default=None)
    # temporal binning
    ap.add_argument("--spokes-per-frame", type=int, default=None)
    ap.add_argument("--frame-shift", type=int, default=None)
    ap.add_argument("--test-volumes", type=int, default=None)
    # recon engine
    ap.add_argument("--force-python-adjoint", action="store_true")
    # synthetic trajectory mode when no traj file is present
    ap.add_argument(
        "--traj-mode",
        choices=["linear_z", "kron"],
        default=None,
        help="Only used when $series/traj is absent. Required then.",
    )
    args = ap.parse_args()
    DEBUG = args.debug

    if not bart_exists():
        print("ERROR: BART not found on PATH.", file=sys.stderr)
        sys.exit(1)

    series_dir = args.series
    out_base = args.out
    NX, NY, NZ = args.matrix

    # k-space
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

    # trajectory: prefer $series/traj; else synthesize from GA settings using traj_mode
    traj = _read_bruker_traj_file(series_dir, ro, sp_total)
    if traj is not None:
        print("[info] Using trajectory from $series/traj")
    else:
        if args.traj_mode is None:
            raise RuntimeError(
                "No $series/traj present. Please specify --traj-mode linear_z|kron to synthesize trajectory."
            )
        print(f"[info] No traj file; building synthetic {args.traj_mode} trajectory from GA/Kronecker math.")
        traj = build_synthetic_traj(series_dir, ro, sp_total, args.traj_mode)

    if traj.shape != (3, ro, sp_total):
        raise ValueError(f"trajectory must have shape (3,{ro},{sp_total}); got {traj.shape}")

    # temporal binning
    spokes_per_frame = args.spokes_per_frame
    if spokes_per_frame is None:
        spokes_per_frame = sp_total  # single-frame default
        print(f"[warn] No spokes-per-frame specified; using all spokes ({sp_total}) in one frame.")
    frame_shift = args.frame_shift if args.frame_shift is not None else spokes_per_frame
    if frame_shift <= 0:
        frame_shift = spokes_per_frame
    starts = list(frame_starts(sp_total, spokes_per_frame, frame_shift))
    if args.test_volumes is not None:
        starts = starts[: max(0, int(args.test_volumes))]
    nframes = len(starts)
    if nframes == 0:
        raise ValueError("No frames to reconstruct with chosen (spokes_per_frame, frame_shift).")
    print(f"[info] Sliding-window frames: {nframes} (spf={spokes_per_frame}, shift={frame_shift})")

    # per-frame recon
    for fi, s0 in enumerate(starts, 1):
        s1 = s0 + spokes_per_frame
        if s1 > sp_total:
            print(f"[warn] Skipping partial window {s0}:{s1} beyond total spokes {sp_total}")
            break
        ksp_f = ksp[:, s0:s1, :]
        traj_f = traj[:, :, s0:s1]
        vol_base = out_base.with_name(out_base.name + f"_vol{fi:05d}")

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

        if args.force_python_adjoint:
            ksp_in = ksp_f if dcf is None else (ksp_f * dcf[..., None])
            recon_adjoint_python(traj_f, ksp_in, dcf, (NX, NY, NZ), vol_base, combine=args.combine)
        else:
            # write BART inputs and recon with nufft -a; Python fallback on failure
            ksp16, kspdims = ksp_to_bart_noncart(ksp_f if dcf is None else (ksp_f * dcf[..., None]))
            traj16, trajdims = traj_to_bart_noncart(traj_f)
            ksp_base = vol_base.with_name(vol_base.name + "_ksp")
            traj_base = vol_base.with_name(vol_base.name + "_traj")
            write_cfl(ksp_base, ksp16, kspdims)
            write_cfl(traj_base, traj16, trajdims)
            try:
                recon_adjoint_bart(traj_base, ksp_base, vol_base, combine=args.combine, gpu=args.gpu)
            except subprocess.CalledProcessError:
                print(f"[warn] BART adjoint failed for frame {fi}; falling back to pure-NumPy adjoint.")
                ksp_in = ksp_f if dcf is None else (ksp_f * dcf[..., None])
                recon_adjoint_python(traj_f, ksp_in, dcf, (NX, NY, NZ), vol_base, combine=args.combine)

        # Always export NIfTI via BART toimg; one NIfTI per frame
        run_bart(["toimg", str(vol_base), str(vol_base)], gpu=False)

        print(f"[info] Frame {fi}/{nframes} done -> {vol_base}")

        if args.test_volumes is not None and fi >= args.test_volumes:
            break

    print("[info] Reconstruction complete.")


if __name__ == "__main__":
    main()
