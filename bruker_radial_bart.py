#!/usr/bin/env python3
"""
bruker_radial_bart.py

Bruker 3D radial reconstruction using BART with robust fallbacks.

What's inside
- Bruker loader:
    * Auto-infers RO / Spokes / Coils from FID + headers (handles blocked-RO padding).
- Trajectory:
    * Prefers $series/traj (binary/ASCII). Else --traj-file. Else golden-angle.
- DCF:
    * Pipe-style (iterative occupancy) implemented in NumPy (no BART).
- Recon paths:
    * Primary: BART adjoint NUFFT (+ SoS or SENSE).
    * NEW Fallback: **Pure-NumPy adjoint gridding** (Kaiser–Bessel) if BART NUFFT OOM’s,
      or force via `--force-python-adjoint`.
- Output:
    * Writes .cfl/.hdr for intermediates and final; optional NIfTI via `bart toimg`.
- GPU:
    * Sticky fallback: if `-g` fails once, we stop trying it.

Notes
- Iterative PICS still requires BART NUFFT; if you hit OOM there, use the Python adjoint path
  for a gridded SoS image (or try a smaller matrix).

Example
-------
python bruker_radial_bart.py \\
  --series "$path" \\
  --matrix 256 256 256 \\
  --traj file \\
  --dcf pipe:10 \\
  --combine sos \\
  --gpu \\
  --export-nifti \\
  --out "${out%.nii.gz}_SoS"
"""

from __future__ import annotations
import argparse
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np

# ---------------- Debug ----------------
DEBUG = False
def dbg(*args):
    if DEBUG:
        print("[debug]", *args)

# Sticky cache for GPU availability this run
BART_GPU_AVAILABLE = None  # None unknown, True/False known

# --------------- CFL I/O ---------------
def _write_hdr(path: Path, dims: List[int]):
    with open(path, "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in dims) + "\n")

def _write_cfl(name: Path, array: np.ndarray, dims16: Optional[List[int]] = None):
    """
    Write array to BART .cfl/.hdr in column-major order.
    If dims16 is provided, it's used verbatim; otherwise array.shape is expanded to 16 dims.
    """
    name = Path(name)
    base = name.with_suffix("")
    if dims16 is None:
        dims16 = list(array.shape) + [1] * (16 - array.ndim)
    _write_hdr(base.with_suffix(".hdr"), dims16)
    arrF = np.asarray(array, dtype=np.complex64, order="F")
    arrF.ravel(order="F").view(np.float32).tofile(base.with_suffix(".cfl"))

def read_cfl(name: Path) -> np.ndarray:
    name = Path(name)
    base = name.with_suffix("")
    with open(base.with_suffix(".hdr"), "r") as f:
        lines = f.read().strip().splitlines()
    dims = tuple(int(x) for x in lines[1].split())
    dims = tuple(d for d in dims if d > 0)
    data = np.fromfile(base.with_suffix(".cfl"), dtype=np.complex64)
    return np.reshape(data, dims, order="F")

# -------- Bruker header helpers --------
def _read_text_kv(path: Path) -> Dict[str, str]:
    d: Dict[str, str] = {}
    if not path.exists():
        return d
    for line in path.read_text(errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip().strip("#$")] = v.strip()
    return d

def _parse_acq_size(method_txt: dict, acqp_txt: dict) -> Optional[Tuple[int, int, int]]:
    for key in ("ACQ_size", "PVM_Matrix"):
        if key in method_txt:
            try:
                toks = method_txt[key].replace("{"," ").replace("}"," ").replace("("," ").replace(")"," ").split()
                nums = [int(x) for x in toks if x.lstrip("+-").isdigit()]
                if len(nums) >= 3:
                    return nums[0], nums[1], nums[2]
            except Exception:
                pass
    return None

def _get_int_from_headers(keys: List[str], srcs: List[dict]) -> Optional[int]:
    for key in keys:
        for src in srcs:
            if key in src:
                try:
                    txt = src[key]
                    num = ""
                    tmp = ""
                    for ch in txt:
                        if ch.isdigit() or (ch in "+-" and not tmp):
                            tmp += ch
                        elif tmp:
                            num = tmp
                            break
                    if not num and tmp:
                        num = tmp
                    if num:
                        return int(num)
                except Exception:
                    pass
    return None

# --------------- Trajectory ---------------
@dataclass
class TrajSpec:
    readout: int
    spokes: int
    matrix: Tuple[int, int, int]

def golden_angle_3d(spec: TrajSpec) -> np.ndarray:
    ro, sp = spec.readout, spec.spokes
    nx, ny, nz = spec.matrix
    kmax = 0.5 * max(nx, ny, nz)
    i = np.arange(sp) + 0.5
    phi = 2.0 * math.pi * i / ((1 + math.sqrt(5)) / 2.0)
    cos_theta = 1 - 2 * i / sp
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    dirs = np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], axis=1)  # (sp,3)
    t = np.linspace(-1.0, 1.0, ro, endpoint=True)
    radii = kmax * t
    xyz = np.einsum("sr,sd->drs", radii[None, :], dirs)  # (3, ro, sp)
    return xyz.astype(np.float32)

def _read_bruker_traj(series_dir: Path, ro: int, sp: int) -> Optional[np.ndarray]:
    """
    Accepts Bruker 'traj' in several flavors:
    - binary float32 or float64 of length 3*ro*sp
    - ASCII flat list of 3*ro*sp numbers
    - ASCII 2D (ro*sp,3) or (3, ro*sp)
    Returns traj shaped (3, ro, sp) float32 if recognized.
    """
    tpath = Path(series_dir) / "traj"
    if not tpath.exists():
        return None
    # try binary float32
    try:
        arr = np.fromfile(tpath, dtype=np.float32)
        if arr.size == 3 * ro * sp:
            return arr.reshape(3, ro, sp, order="F").astype(np.float32)
    except Exception:
        pass
    # try binary float64
    try:
        arr = np.fromfile(tpath, dtype=np.float64)
        if arr.size == 3 * ro * sp:
            return arr.reshape(3, ro, sp, order="F").astype(np.float32)
    except Exception:
        pass
    # try ASCII
    try:
        toks = tpath.read_text(errors="ignore").strip().split()
        vals = np.array([float(x) for x in toks], dtype=np.float32)
        if vals.size == 3 * ro * sp:
            return vals.reshape(3, ro, sp, order="F")
        data = np.loadtxt(tpath, dtype=np.float32)
        if data.ndim == 2:
            if data.shape == (ro * sp, 3):
                return data.T.reshape(3, ro, sp, order="F")
            if data.shape == (3, ro * sp):
                return data.reshape(3, ro, sp, order="F")
    except Exception:
        pass
    return None

# --------------- FID / k-space loader ---------------
def load_bruker_kspace(series_dir: Path,
                       matrix_ro_hint: Optional[int] = None,
                       spokes: Optional[int] = None,
                       readout: Optional[int] = None,
                       coils: Optional[int] = None,
                       fid_dtype: str = "int32",
                       fid_endian: str = "little") -> np.ndarray:
    """
    Returns k-space as (RO, Spokes, Coils), trimming blocked RO if needed.
    Accepts:
      - series/ksp.cfl|.hdr
      - series/ksp.npy (RO,Spokes,Coils)
      - series/fid (+ headers)
    """
    series_dir = Path(series_dir)
    dbg("series_dir:", series_dir)

    # pre-made ksp?
    cfl = series_dir / "ksp"
    if cfl.with_suffix(".cfl").exists() and cfl.with_suffix(".hdr").exists():
        arr = read_cfl(cfl); dbg("loaded ksp.cfl:", arr.shape); return arr
    npy = series_dir / "ksp.npy"
    if npy.exists():
        arr = np.load(npy); dbg("loaded ksp.npy:", arr.shape)
        if arr.ndim != 3: raise ValueError("ksp.npy must be 3D (RO,Spokes,Coils)")
        return arr

    # raw FID
    fid_path = series_dir / "fid"
    if not fid_path.exists():
        raise FileNotFoundError("No k-space found (no fid, ksp.cfl, or ksp.npy)")

    method = _read_text_kv(series_dir / "method")
    acqp   = _read_text_kv(series_dir / "acqp")

    # endian/dtype from headers
    if "BYTORDA" in acqp and "big" in acqp["BYTORDA"].lower():
        fid_endian = "big"
    if "ACQ_word_size" in acqp:
        if "16" in acqp["ACQ_word_size"]:
            fid_dtype = "int16"
        elif "32" in acqp["ACQ_word_size"]:
            fid_dtype = "int32"

    dtype_map = {"int16": np.int16, "int32": np.int32, "float32": np.float32, "float64": np.float64}
    if fid_dtype not in dtype_map: raise ValueError("--fid-dtype must be one of int16,int32,float32,float64")
    dt = dtype_map[fid_dtype]

    raw = np.fromfile(fid_path, dtype=dt)
    if fid_endian == "big": raw = raw.byteswap().newbyteorder()
    if raw.size % 2 != 0: raw = raw[:-1]
    cpx = raw.astype(np.float32).view(np.complex64)
    total_cplx = cpx.size
    dbg("total complex samples:", total_cplx, "(dtype:", fid_dtype, ", endian:", fid_endian, ")")

    # hints
    if readout is None:
        acq_size = _parse_acq_size(method, acqp)
        if acq_size: readout = acq_size[0]
    if readout is None and matrix_ro_hint: readout = matrix_ro_hint
    dbg("readout (hdr/matrix hint):", readout)

    if coils is None:
        nrec = _get_int_from_headers(["PVM_EncNReceivers"], [method])
        if nrec and nrec > 0: coils = nrec
    if coils is None or coils <= 0: coils = 1
    dbg("coils (initial):", coils)

    extras = {
        "echoes":   _get_int_from_headers(["NECHOES", "ACQ_n_echo_images", "PVM_NEchoImages"], [method, acqp]) or 1,
        "reps":     _get_int_from_headers(["PVM_NRepetitions", "NR"], [method, acqp]) or 1,
        "averages": _get_int_from_headers(["PVM_NAverages", "NA"], [method, acqp]) or 1,
        "slices":   _get_int_from_headers(["NSLICES", "PVM_SPackArrNSlices"], [method, acqp]) or 1,
    }
    other_dims = 1
    for v in extras.values():
        if isinstance(v, int) and v > 1: other_dims *= v
    dbg("other_dims factor:", other_dims, extras)

    denom = coils * max(1, other_dims)
    if total_cplx % denom != 0:
        dbg("total not divisible by coils*other_dims; relaxing extras")
        denom = coils
        if total_cplx % denom != 0:
            dbg("still not divisible; relaxing coils->1")
            coils = 1; denom = coils
            if total_cplx % denom != 0:
                raise ValueError("Cannot factor FID length with any (coils, other_dims) combo.")
    per_coil_total = total_cplx // denom  # stored_ro * spokes
    dbg("per_coil_total (stored_ro*spokes):", per_coil_total, "  coils:", coils, " other_dims:", other_dims)

    def pick_block_and_spokes(per_coil_total: int, readout_hint: Optional[int], spokes_hint: Optional[int]) -> Tuple[int,int]:
        if spokes_hint and spokes_hint > 0 and per_coil_total % spokes_hint == 0:
            return per_coil_total // spokes_hint, spokes_hint
        BLOCKS = [128,160,192,200,224,240,256,288,320,352,384,400,416,420,432,448,480,496,512,
                  544,560,576,608,640,672,704,736,768,800,832,896,960,992,1024,1152,1280,1536,2048]
        cands = []
        if readout_hint:
            cands += [b for b in BLOCKS if b >= readout_hint]
            if per_coil_total % readout_hint == 0:
                return readout_hint, per_coil_total // readout_hint
        cands += [per_coil_total]
        for b in cands:
            if b > 0 and per_coil_total % b == 0:
                return b, per_coil_total // b
        s = int(round(per_coil_total ** 0.5))
        for d in range(0, s+1):
            for cand in (s+d, s-d):
                if cand > 0 and per_coil_total % cand == 0:
                    return cand, per_coil_total // cand
        raise ValueError("Could not factor per_coil_total into (stored_ro, spokes).")

    stored_ro, spokes_inf = pick_block_and_spokes(per_coil_total, readout, spokes)
    dbg("stored_ro (block):", stored_ro, " spokes (per extras-collapsed):", spokes_inf)
    spokes_final = spokes_inf * max(1, other_dims)
    if stored_ro * spokes_final * coils != total_cplx:
        raise ValueError("Internal factoring error: stored_ro*spokes_final*coils != total samples")

    ksp_blk = np.reshape(cpx, (stored_ro, spokes_final, coils), order="F")
    if readout is not None and stored_ro >= readout:
        ksp = ksp_blk[:readout, :, :]; dbg("trimmed RO from", stored_ro, "to", readout)
    else:
        ksp = ksp_blk
        if readout is None: readout = stored_ro

    dbg("final k-space shape:", ksp.shape, "(RO, Spokes, Coils)")
    return ksp  # (ro, sp, coils)

# --------- BART wrappers (sticky GPU fallback) ---------
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
        subprocess.run([bart, tool, "-g", "-h"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, check=True)
        BART_GPU_AVAILABLE = True
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or b"").decode(errors="ignore")
        if "compiled without GPU" in msg or "invalid option -- 'g'" in msg or "unknown option g" in msg:
            BART_GPU_AVAILABLE = False
        else:
            BART_GPU_AVAILABLE = False
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

# --------------- DCF (NumPy Pipe) ---------------
def _normalize_traj_to_grid(traj: np.ndarray, grid_shape: Tuple[int,int,int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map traj k-space coords to integer grid indices in [0..N-1] per axis.
    Uses robust min/max to map to grid. Returns (ix,iy,iz) int arrays of shape (ro, sp).
    """
    ro, sp = traj.shape[1], traj.shape[2]
    kx, ky, kz = traj[0], traj[1], traj[2]
    def robust_minmax(a):
        lo = np.percentile(a, 0.5); hi = np.percentile(a, 99.5)
        if hi <= lo: hi = lo + 1e-3
        return float(lo), float(hi)
    xmin,xmax = robust_minmax(kx)
    ymin,ymax = robust_minmax(ky)
    zmin,zmax = robust_minmax(kz)
    def map_axis(a, lo, hi, n):
        t = (a - lo) / (hi - lo)
        t = np.clip(t, 0.0, 1.0)
        return np.clip(np.rint(t * (n-1)).astype(np.int32), 0, n-1)
    ix = map_axis(kx, xmin, xmax, grid_shape[0])
    iy = map_axis(ky, ymin, ymax, grid_shape[1])
    iz = map_axis(kz, zmin, zmax, grid_shape[2])
    return ix, iy, iz

def dcf_pipe_numpy(traj: np.ndarray, iters: int, grid_shape: Tuple[int,int,int]) -> np.ndarray:
    """
    Pure NumPy Pipe-like DCF:
      initialize w=1
      repeat iters times:
        grid(x) = sum_j w_j at nearest grid cell
        denom_j = grid(x_j) + eps
        w <- w / denom
      renormalize mean(w) to 1.
    Returns w with shape (ro, sp) float32.
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

# ---------- Layout helpers ----------
def ksp_to_bart_noncart(ksp_ro_sp_coils: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Return array with dims [1,1,1,COIL,1,1,1,1,1,1,RO,SP,1,1,1,1]
    which is friendly to many BART non-Cart builds.
    """
    ro, sp, nc = ksp_ro_sp_coils.shape
    arr = ksp_ro_sp_coils.astype(np.complex64, order="F")
    arr16 = arr.reshape(ro, sp, nc, *([1]*13))
    # Place at indices: coils->3, ro->10, sp->11
    perm = [3,4,5, 2, 6,7,8,9,12,13, 0,1, 10,11,14,15]
    arr16 = np.transpose(arr16, perm)
    dims16 = [1]*16
    dims16[3]  = nc
    dims16[10] = ro
    dims16[11] = sp
    return arr16, dims16

def traj_to_bart_noncart(traj_3_ro_sp: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Return traj dims [3,1,1,1,1,1,1,1,1,1,RO,SP,1,1,1,1]
    """
    _, ro, sp = traj_3_ro_sp.shape
    arr = traj_3_ro_sp.astype(np.complex64, order="F")
    arr16 = arr.reshape(3, ro, sp, *([1]*13))
    perm = [0, 3,4,5,6,7,8,9,12,13, 1,2, 10,11,14,15]
    arr16 = np.transpose(arr16, perm)
    dims16 = [1]*16
    dims16[0]  = 3
    dims16[10] = ro
    dims16[11] = sp
    return arr16, dims16

def _make_weight_like(target: np.ndarray, w2d: np.ndarray) -> np.ndarray:
    """Broadcast w2d (ro,sp) onto target's axes (unknown order)."""
    ro, sp = w2d.shape
    tshape = target.shape
    axes = list(range(len(tshape)))
    pos_ro = [i for i in axes if tshape[i] == ro]
    pos_sp = [i for i in axes if tshape[i] == sp]
    for i in pos_ro:
        for j in pos_sp:
            if i != j:
                shape = [1]*len(tshape); shape[i] = ro; shape[j] = sp
                return w2d.reshape(shape).astype(np.complex64)
    if pos_ro and len(pos_ro) >= 2 and ro == sp:
        i, j = pos_ro[0], pos_ro[1]
        shape = [1]*len(tshape); shape[i] = ro; shape[j] = sp
        return w2d.reshape(shape).astype(np.complex64)
    shape = [1]*len(tshape)
    if len(tshape)>0: shape[0]=ro
    if len(tshape)>1: shape[1]=sp
    return w2d.reshape(shape).astype(np.complex64)

# ---------- Pure NumPy adjoint gridding ----------
def _kaiser_bessel(u: np.ndarray, width: float, beta: float) -> np.ndarray:
    """
    Kaiser-Bessel kernel (0 <= |u| <= width/2), normalized so u is in grid units.
    u can be any real array; outside support returns 0.
    """
    x = np.abs(u)
    out = np.zeros_like(x, dtype=np.float32)
    half = width / 2.0
    m = x <= half
    t = np.sqrt(1.0 - (x[m] / half)**2)
    # I0(beta * t) / I0(beta)
    from numpy import i0
    out[m] = (i0(beta * t) / i0(beta)).astype(np.float32)
    return out

def _deapod_1d(N: int, os: float, width: float, beta: float) -> np.ndarray:
    """
    1D deapodization for Kaiser-Bessel gridding using approximate frequency response.
    We compute correction on oversampled grid and decimate later.
    """
    # frequency grid normalized to [-0.5, 0.5)
    k = (np.arange(N*os) - (N*os)/2) / (N*os)
    # Approximate deapodization as reciprocal of kernel FT at these frequencies.
    # For Kaiser-Bessel, a common approximation is sinc-like; we use a safe floor.
    # (Good enough for radial with modest oversampling & width.)
    eps = 1e-6
    # Use window in image domain -> smooth rolloff in k-space; reciprocal here
    corr = np.maximum(eps, np.sinc(k))  # crude but stable
    return (1.0 / corr).astype(np.float32)

def adjoint_grid_numpy(traj_3_ro_sp: np.ndarray,
                       ksp_ro_sp_coils: np.ndarray,
                       dcf_ro_sp: Optional[np.ndarray],
                       grid_shape: Tuple[int,int,int],
                       oversamp: float = 1.5,
                       kb_width: float = 3.0,
                       kb_beta: float = 8.0) -> np.ndarray:
    """
    Adjoint NUFFT via separable Kaiser–Bessel gridding + iFFT.
    Returns complex coil images with shape (NX, NY, NZ, COILS).
    """
    RO, SP, NC = ksp_ro_sp_coils.shape
    NX, NY, NZ = grid_shape
    NXg = int(round(NX * oversamp))
    NYg = int(round(NY * oversamp))
    NZg = int(round(NZ * oversamp))

    # Map traj (in "pixel units" of k-space) to oversampled grid index space centered
    # Assume traj is in [-kmax, kmax] where kmax ~ 0.5*max(NX,NY,NZ)
    kx, ky, kz = traj_3_ro_sp[0], traj_3_ro_sp[1], traj_3_ro_sp[2]  # (RO,SP)
    kmax = 0.5 * float(max(NX, NY, NZ))
    gx = (kx / (2*kmax) + 0.5) * NXg
    gy = (ky / (2*kmax) + 0.5) * NYg
    gz = (kz / (2*kmax) + 0.5) * NZg

    # Precompute deapodization (approx) and window
    deapx = _deapod_1d(NX, oversamp, kb_width, kb_beta)
    deapy = _deapod_1d(NY, oversamp, kb_width, kb_beta)
    deapz = _deapod_1d(NZ, oversamp, kb_width, kb_beta)

    # Initialize oversampled grid for each coil
    img_grid = np.zeros((NXg, NYg, NZg, NC), dtype=np.complex64)

    # Support half-width in grid cells
    hw = int(math.ceil(kb_width / 2.0))

    # Apply DCF if provided
    if dcf_ro_sp is None:
        w = np.ones((RO, SP), dtype=np.float32)
    else:
        w = dcf_ro_sp.astype(np.float32)

    # Scatter-add for each sample across coils
    for s in range(SP):
        # integer center
        cx = np.floor(gx[:, s]).astype(int)
        cy = np.floor(gy[:, s]).astype(int)
        cz = np.floor(gz[:, s]).astype(int)

        for t in range(RO):
            x0 = cx[t]; y0 = cy[t]; z0 = cz[t]
            # local neighborhood
            xr = np.arange(x0 - hw, x0 + hw + 1)
            yr = np.arange(y0 - hw, y0 + hw + 1)
            zr = np.arange(z0 - hw, z0 + hw + 1)

            # fractional offsets
            wx = _kaiser_bessel(xr - gx[t, s], kb_width, kb_beta)  # (len(xr),)
            wy = _kaiser_bessel(yr - gy[t, s], kb_width, kb_beta)
            wz = _kaiser_bessel(zr - gz[t, s], kb_width, kb_beta)

            # outer product weights
            wxyz = (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]).astype(np.float32) * w[t, s]

            # bounds check / clip
            xsel = (xr >= 0) & (xr < NXg)
            ysel = (yr >= 0) & (yr < NYg)
            zsel = (zr >= 0) & (zr < NZg)
            if not (np.any(xsel) and np.any(ysel) and np.any(zsel)):
                continue
            xr2 = xr[xsel]; yr2 = yr[ysel]; zr2 = zr[zsel]
            wxyz2 = wxyz[xsel][:, ysel][:, :, zsel]

            # add to all coils at this sample
            for c in range(NC):
                val = ksp_ro_sp_coils[t, s, c]
                img_grid[np.ix_(xr2, yr2, zr2, [c])] += (wxyz2[..., None] * val).astype(np.complex64)

    # Inverse FFT to image domain
    img = np.fft.ifftn(np.fft.ifftshift(img_grid, axes=(0,1,2)), axes=(0,1,2))
    img = np.fft.fftshift(img, axes=(0,1,2))

    # Crop from oversampled grid to target matrix
    x0 = (NXg - NX) // 2; y0 = (NYg - NY) // 2; z0 = (NZg - NZ) // 2
    img = img[x0:x0+NX, y0:y0+NY, z0:z0+NZ, :]

    # Deapodize (approx)
    # Broadcast deapodization along axes
    img *= deapx[:NX, None, None, None]
    img *= deapy[None, :NY, None, None]
    img *= deapz[None, None, :NZ, None]

    return img.astype(np.complex64)  # (NX,NY,NZ,Coils)

# --------------- Recon flows ---------------
def estimate_sens_maps(coil_imgs_base: Path, out_base: Path, calib: Optional[int]=None, gpu: bool=False):
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

def recon_adjoint_python(traj_3_ro_sp: np.ndarray, ksp_ro_sp_coils: np.ndarray,
                         dcf_ro_sp: Optional[np.ndarray],
                         matrix: Tuple[int,int,int], out_base: Path, combine: str):
    NX, NY, NZ = matrix
    # Gridding
    coil_img = adjoint_grid_numpy(traj_3_ro_sp, ksp_ro_sp_coils, dcf_ro_sp, (NX, NY, NZ))  # (NX,NY,NZ,Coils)
    coil_base = out_base.with_name(out_base.name + "_coil_py")
    # Save as BART cfl with dims [X,Y,Z,COIL] (standard image layout)
    img = np.moveaxis(coil_img, (0,1,2,3), (0,1,2,3))  # already in X,Y,Z,Coil
    _write_cfl(coil_base, img, [NX, NY, NZ, coil_img.shape[3]] + [1]*12)

    if combine.lower() == "sos":
        # RSS over coil dim=3
        rss = np.sqrt(np.sum(np.abs(img)**2, axis=3)).astype(np.complex64)
        out = out_base
        _write_cfl(out, rss, [NX, NY, NZ] + [1]*13)
    elif combine.lower() == "sens":
        # For SENSE you'd want a sensitivity estimation; we can still call BART ecalib/pics on the CFL
        maps = out_base.with_name(out_base.name + "_maps_py")
        estimate_sens_maps(coil_base, maps, gpu=False)
        run_bart(["pics", "-S", str(coil_base), str(maps), str(out_base)], gpu=False)
    else:
        raise ValueError("combine must be sos|sens")

def recon_iterative(traj_base: Path, ksp_base: Path, out_base: Path,
                    lam: float, iters: int, wavelets: Optional[int], gpu: bool):
    tmpcoil = out_base.with_name(out_base.name + "_coil")
    run_bart(["nufft", "-a", "-t", str(traj_base), str(ksp_base), str(tmpcoil)], gpu=gpu)
    maps = out_base.with_name(out_base.name + "_maps")
    estimate_sens_maps(tmpcoil, maps, gpu=gpu)
    cmd = ["pics", "-S", "-i", str(iters)]
    if wavelets is not None:
        cmd += ["-R", f"W:7:{wavelets}:{lam}"]
    elif lam > 0:
        cmd += ["-R", f"W:7:0:{lam}"]
    cmd += ["-t", str(traj_base), str(ksp_base), str(maps), str(out_base)]
    run_bart(cmd, gpu=gpu)

# ---------------- CLI / Main ----------------
def bart_exists() -> bool:
    return shutil.which("bart") is not None

def main():
    global DEBUG
    ap = argparse.ArgumentParser(description="Bruker 3D radial reconstruction using BART")
    ap.add_argument("--series", type=Path, required=True, help="Bruker experiment dir (contains fid/method/acqp or ksp.*)")
    ap.add_argument("--out", type=Path, required=True, help="Output basename (no extension) for BART/NIfTI")
    ap.add_argument("--matrix", type=int, nargs=3, required=True, metavar=("NX","NY","NZ"))
    ap.add_argument("--traj", choices=["golden", "file"], default="file")
    ap.add_argument("--traj-file", type=Path, help="If --traj file, optional path to traj.cfl/.hdr or traj.npy")
    ap.add_argument("--dcf", type=str, default="none", help="DCF mode: none | pipe:Niters")
    ap.add_argument("--combine", type=str, default="sos", help="Coil combine: sos|sens")
    ap.add_argument("--iterative", action="store_true", help="Use iterative PICS reconstruction (BART)")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.0, help="Regularization for iterative recon")
    ap.add_argument("--iters", type=int, default=40, help="Iterations for iterative recon")
    ap.add_argument("--wavelets", type=int, default=None, help="Wavelet scale (optional)")
    ap.add_argument("--export-nifti", action="store_true", help="Export NIfTI via bart toimg")
    ap.add_argument("--gpu", action="store_true", help="Attempt GPU; sticky CPU fallback if unsupported")
    ap.add_argument("--debug", action="store_true", help="Print header inference and file checks")
    # optional overrides
    ap.add_argument("--readout", type=int, default=None)
    ap.add_argument("--spokes", type=int, default=None)
    ap.add_argument("--coils", type=int, default=None)
    ap.add_argument("--fid-dtype", type=str, default=None)
    ap.add_argument("--fid-endian", type=str, default=None)
    # force python adjoint
    ap.add_argument("--force-python-adjoint", action="store_true",
                    help="Skip BART NUFFT and use pure-NumPy adjoint gridding")

    args = ap.parse_args()
    DEBUG = args.debug

    if not bart_exists():
        print("ERROR: BART not found on PATH.", file=sys.stderr); sys.exit(1)

    series_dir: Path = args.series
    out_base: Path = args.out
    NX, NY, NZ = args.matrix

    # ---- k-space (RO, Spokes, Coils)
    ksp = load_bruker_kspace(series_dir,
                             matrix_ro_hint=NX,
                             spokes=args.spokes,
                             readout=args.readout,
                             coils=args.coils,
                             fid_dtype=(args.fid_dtype or "int32"),
                             fid_endian=(args.fid_endian or "little"))
    ro, sp, nc = ksp.shape
    print(f"[info] Loaded k-space: RO={ro}, Spokes={sp}, Coils={nc}")

    # Save KSP in non-Cart layout for BART path (and to have a CFL copy on disk)
    ksp_arr16, ksp_dims16 = ksp_to_bart_noncart(ksp)
    ksp_base = out_base.with_name(out_base.name + "_ksp")
    _write_cfl(ksp_base, ksp_arr16, ksp_dims16)

    # ---- trajectory (3, ro, sp)
    traj = _read_bruker_traj(series_dir, ro, sp)
    if traj is None:
        if args.traj == "golden":
            traj = golden_angle_3d(TrajSpec(readout=ro, spokes=sp, matrix=(NX, NY, NZ)))
        else:
            base = args.traj_file
            if base is not None:
                if base.with_suffix(".cfl").exists() and base.with_suffix(".hdr").exists():
                    traj = read_cfl(base)  # assume correct (3,ro,sp)
                elif base.with_suffix(".npy").exists():
                    traj = np.load(base.with_suffix(".npy"))
                else:
                    t2 = _read_bruker_traj(base.parent, ro, sp)
                    traj = t2 if t2 is not None else golden_angle_3d(TrajSpec(readout=ro, spokes=sp, matrix=(NX, NY, NZ)))
            else:
                print("[warn] No trajectory file found in series dir; generating golden-angle trajectory to match k-space.")
                traj = golden_angle_3d(TrajSpec(readout=ro, spokes=sp, matrix=(NX, NY, NZ)))
    if traj.shape != (3, ro, sp):
        raise ValueError(f"trajectory must have shape (3,{ro},{sp}); got {traj.shape}")

    traj_arr16, traj_dims16 = traj_to_bart_noncart(traj)
    traj_base = out_base.with_name(out_base.name + "_traj")
    _write_cfl(traj_base, traj_arr16, traj_dims16)

    # ---- DCF
    dcf_mode = args.dcf.lower()
    if dcf_mode.startswith("pipe"):
        nit = 10
        if ":" in dcf_mode:
            try: nit = int(dcf_mode.split(":", 1)[1])
            except Exception: pass
        dcf = dcf_pipe_numpy(traj, iters=nit, grid_shape=(NX, NY, NZ))
        dcf_base = out_base.with_name(out_base.name + "_dcf")
        # store DCF in non-Cart layout too
        dcf16 = dcf.astype(np.complex64)
        dcf16 = dcf16.reshape(ro, sp, *([1]*14))
        perm = [2,3,4,5,6,7,8,9,12,13, 0,1, 10,11,14,15]
        dcf16 = np.transpose(dcf16, perm)
        dcf_dims16 = [1]*16; dcf_dims16[10]=ro; dcf_dims16[11]=sp
        _write_cfl(dcf_base, dcf16, dcf_dims16)

        # apply DCF to ksp (NumPy)
        ksp_arr = read_cfl(ksp_base)     # 16D layout
        dcf_b = _make_weight_like(ksp_arr, dcf)  # broadcast (ro,sp) to ksp layout
        kspw = ksp_arr * dcf_b
        kspw_base = out_base.with_name(out_base.name + "_kspw")
        _write_cfl(kspw_base, kspw, list(ksp_arr.shape) + [1]*(16-len(ksp_arr.shape)))
        ksp_in = kspw_base
        ksp_for_python = ksp * dcf  # for Python gridder path (shape ro,sp,coils)
    else:
        ksp_in = ksp_base
        ksp_for_python = ksp

    # ---- reconstruction
    if args.iterative:
        out_img = out_base.with_name(out_base.name + "_recon")
        recon_iterative(traj_base, ksp_in, out_img, lam=args.lam, iters=args.iters, wavelets=args.wavelets, gpu=args.gpu)
    else:
        out_img = out_base.with_name(out_base.name + "_adj")

        if args.force_python_adjoint:
            # Pure NumPy adjoint gridding (always)
            recon_adjoint_python(traj, ksp_for_python, dcf if dcf_mode.startswith("pipe") else None,
                                 (NX,NY,NZ), out_img, combine=args.combine)
        else:
            # Try BART once; on failure, fall back to Python adjoint
            try:
                recon_adjoint_bart(traj_base, ksp_in, out_img, combine=args.combine, gpu=args.gpu)
            except subprocess.CalledProcessError:
                print("[warn] BART adjoint failed; falling back to pure-NumPy adjoint gridding.")
                recon_adjoint_python(traj, ksp_for_python, dcf if dcf_mode.startswith("pipe") else None,
                                     (NX,NY,NZ), out_img, combine=args.combine)

    # ---- export
    if args.export_nifti:
        run_bart(["toimg", str(out_img), str(out_base)], gpu=False)
        print(f"Wrote NIfTI: {out_base}.nii")
    else:
        print(f"Recon complete. BART base: {out_img} (.cfl/.hdr)")

if __name__ == "__main__":
    main()
