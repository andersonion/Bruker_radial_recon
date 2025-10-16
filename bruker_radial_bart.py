#!/usr/bin/env python3
"""
bruker_radial_bart.py

Bruker 3D radial reconstruction using BART with robust low-memory adjoint fallback.

- Auto-infers RO / Spokes / Coils from Bruker headers + raw FID (handles blocked-RO padding).
- Trajectory: prefers $series/traj (bin/ASCII); else --traj-file; else golden-angle.
- DCF (Pipe-style) implemented purely in NumPy (no BART) to avoid NUFFT asserts.
- Writes non-Cartesian-friendly CFLs:
    ksp  dims: [1,1,1,COIL,1,1,1,1,1,1,RO,SP,1,1,1,1]
    traj dims: [3,1,1,1,  1,1,1,1,1,1,RO,SP,1,1,1,1]
- GPU toggle with sticky CPU fallback.
- NEW: Adjoint NUFFT **chunked fallback** on spokes to avoid OOM. Force with --lowmem-sp-chunk.
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

DEBUG = False
def dbg(*args):
    if DEBUG:
        print("[debug]", *args)

BART_GPU_AVAILABLE = None  # sticky cache

# ---------- CFL I/O ----------
def _write_hdr(path: Path, dims: List[int]):
    with open(path, "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in dims) + "\n")

def _write_cfl(name: Path, array: np.ndarray, dims16: Optional[List[int]] = None):
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

# ---------- Bruker helpers ----------
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

# ---------- Trajectory ----------
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

# ---------- FID / k-space ----------
def load_bruker_kspace(series_dir: Path,
                       matrix_ro_hint: Optional[int] = None,
                       spokes: Optional[int] = None,
                       readout: Optional[int] = None,
                       coils: Optional[int] = None,
                       fid_dtype: str = "int32",
                       fid_endian: str = "little") -> np.ndarray:
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
    if "BYTORDA" in acqp and "big" in acqp["BYTORDA"].lower(): fid_endian = "big"
    if "ACQ_word_size" in acqp:
        if "16" in acqp["ACQ_word_size"]: fid_dtype = "int16"
        elif "32" in acqp["ACQ_word_size"]: fid_dtype = "int32"

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
                    return cand, per_coil_total // b
        # fallback brute force
        for b in range(1, min(4096, per_coil_total)+1):
            if per_coil_total % b == 0:
                return b, per_coil_total // b
        raise ValueError("Could not factor per_coil_total into (stored_ro, spokes).")

    stored_ro, spokes_inf = pick_block_and_spokes(per_coil_total, readout, None)
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

# ---------- BART wrappers (sticky GPU fallback) ----------
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

# ---------- DCF (NumPy Pipe) ----------
def _normalize_traj_to_grid(traj: np.ndarray, grid_shape: Tuple[int,int,int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    ro, sp, nc = ksp_ro_sp_coils.shape
    arr = ksp_ro_sp_coils.astype(np.complex64, order="F")
    arr16 = arr.reshape(ro, sp, nc, *([1]*13))
    # place: coils->3, ro->10, sp->11
    perm = [3,4,5, 2, 6,7,8,9,12,13, 0,1, 10,11,14,15]
    arr16 = np.transpose(arr16, perm)
    dims16 = [1]*16; dims16[3]=nc; dims16[10]=ro; dims16[11]=sp
    return arr16, dims16

def traj_to_bart_noncart(traj_3_ro_sp: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    _, ro, sp = traj_3_ro_sp.shape
    arr = traj_3_ro_sp.astype(np.complex64, order="F")
    arr16 = arr.reshape(3, ro, sp, *([1]*13))
    # place: axis0=3 at dim 0, ro->10, sp->11
    perm = [0, 3,4,5,6,7,8,9,12,13, 1,2, 10,11,14,15]
    arr16 = np.transpose(arr16, perm)
    dims16 = [1]*16; dims16[0]=3; dims16[10]=ro; dims16[11]=sp
    return arr16, dims16

def _make_weight_like(target: np.ndarray, w2d: np.ndarray) -> np.ndarray:
    ro, sp = w2d.shape
    tshape = target.shape
    axes = list(range(len(tshape)))
    pos_ro = [i for i in axes if tshape[i] == ro]
    pos_sp = [i for i in axes if tshape[i] == sp]
    for i in pos_ro:
        for j in pos_sp:
            if i != j:
                shape = [1]*len(tshape); shape[i]=ro; shape[j]=sp
                return w2d.reshape(shape).astype(np.complex64)
    if pos_ro and len(pos_ro)>=2 and ro==sp:
        i,j = pos_ro[0], pos_ro[1]
        shape=[1]*len(tshape); shape[i]=ro; shape[j]=sp
        return w2d.reshape(shape).astype(np.complex64)
    shape=[1]*len(tshape); 
    if len(tshape)>0: shape[0]=ro
    if len(tshape)>1: shape[1]=sp
    return w2d.reshape(shape).astype(np.complex64)

# ---------- Recon flows ----------
def estimate_sens_maps(coil_imgs_base: Path, out_base: Path, calib: Optional[int]=None, gpu: bool=False):
    cmd = ["ecalib"]
    if calib is not None:
        cmd += ["-r", str(calib)]
    cmd += [str(coil_imgs_base), str(out_base)]
    run_bart(cmd, gpu=gpu)

def _read_traj_3_ro_sp_from_cfl(traj_base: Path) -> np.ndarray:
    arr = read_cfl(traj_base)
    # find axes: 3, ro, sp
    shape = arr.shape
    ax3 = next((i for i,L in enumerate(shape) if L==3), None)
    if ax3 is None:
        raise ValueError("trajectory CFL does not contain leading dim=3")
    axes_other = [i for i in range(len(shape)) if i!=ax3 and shape[i]>1]
    if len(axes_other) < 2:
        raise ValueError("trajectory CFL missing RO/SP dims")
    i1,i2 = axes_other[0], axes_other[1]
    arr = np.moveaxis(arr, (ax3,i1,i2), (0,1,2))
    # squeeze trailing ones
    arr = arr.reshape(3, shape[i1], shape[i2])
    return arr.astype(np.complex64).real.astype(np.float32)  # ensure float32 (kx,ky,kz)

def _read_ksp_ro_sp_coils_from_cfl(ksp_base: Path) -> np.ndarray:
    arr = read_cfl(ksp_base)
    shape = arr.shape
    # find RO, SP, COIL axes
    # coil length is the only >1 that is NOT ro/sp when those are known (we’ll infer by process of elimination)
    # Prefer dims that match typical placements: 3->none here, ro/sp usually unique.
    axes_nz = [i for i,L in enumerate(shape) if L>1]
    # guess ro=largest axis, sp=second largest among >1 (works for most radial)
    sizes = [(i,shape[i]) for i in axes_nz]
    sizes_sorted = sorted(sizes, key=lambda x: x[1], reverse=True)
    if len(sizes_sorted) < 3:
        raise ValueError("ksp CFL missing dims (need at least RO, SP, COIL)")
    ro_i = sizes_sorted[0][0]
    sp_i = sizes_sorted[1][0]
    coil_i = sizes_sorted[2][0]
    arr = np.moveaxis(arr, (ro_i, sp_i, coil_i), (0,1,2))
    arr = arr.reshape(shape[ro_i], shape[sp_i], shape[coil_i])
    return arr

def _write_traj_chunk(traj3: np.ndarray, sp_lo: int, sp_hi: int, base: Path):
    chunk = traj3[:, :, sp_lo:sp_hi]  # (3, ro, spc)
    arr16, dims16 = traj_to_bart_noncart(chunk)
    _write_cfl(base, arr16, dims16)

def _write_ksp_chunk(ksp: np.ndarray, sp_lo: int, sp_hi: int, base: Path):
    # ksp is (ro, sp, coils)
    chunk = ksp[:, sp_lo:sp_hi, :]
    arr16, dims16 = ksp_to_bart_noncart(chunk)
    _write_cfl(base, arr16, dims16)

def recon_adjoint_chunked(traj_base: Path, ksp_base: Path, out_base: Path,
                          combine: str, gpu: bool, sp_total: int, chunk: int,
                          tmpdir: Path):
    """Adjoint NUFFT in chunks along spokes; accumulates complex coil image."""
    # Load numpy views
    traj3 = _read_traj_3_ro_sp_from_cfl(traj_base)
    ksp = _read_ksp_ro_sp_coils_from_cfl(ksp_base)  # (ro, sp, coils)

    # We’ll probe first chunk to get image dims, then allocate accumulator
    acc = None
    coil_base = out_base.with_name(out_base.name + "_coil")

    for lo in range(0, sp_total, chunk):
        hi = min(sp_total, lo + chunk)
        tb = tmpdir / f"traj_{lo}_{hi}"
        kb = tmpdir / f"ksp_{lo}_{hi}"
        ib = tmpdir / f"img_{lo}_{hi}"

        _write_traj_chunk(traj3, lo, hi, tb)
        _write_ksp_chunk(ksp, lo, hi, kb)
        run_bart(["nufft", "-a", "-t", str(tb), str(kb), str(ib)], gpu=gpu)

        img = read_cfl(ib)  # complex coil image (dims depend on BART plan)
        if acc is None:
            acc = np.array(img, dtype=np.complex64, copy=True)
        else:
            # broadcast-safe sum
            # make shapes equal by expanding singleton dims if needed
            if acc.shape != img.shape:
                # try to pad img to acc
                # move axes to match by aligning non-1 dims
                if img.size == acc.size:
                    img = img.reshape(acc.shape, order="F")
                else:
                    raise ValueError(f"Chunk image shape mismatch: {img.shape} vs {acc.shape}")
            acc += img

    # write accumulated coil image
    _write_cfl(coil_base, acc, list(acc.shape) + [1]*(16-len(acc.shape)))

    if combine.lower() == "sos":
        run_bart(["rss", "8", str(coil_base), str(out_base)], gpu=gpu)
    elif combine.lower() == "sens":
        maps = out_base.with_name(out_base.name + "_maps")
        estimate_sens_maps(coil_base, maps, gpu=gpu)
        run_bart(["pics", "-S", str(coil_base), str(maps), str(out_base)], gpu=gpu)
    else:
        raise ValueError("combine must be sos|sens")

def recon_adjoint(traj_base: Path, ksp_base: Path, combine: str, out_base: Path,
                  gpu: bool, sp_total: int, lowmem_sp_chunk: Optional[int] = None):
    coil_base = out_base.with_name(out_base.name + "_coil")
    tmpdir = Path("./.tmp_bart_adj"); tmpdir.mkdir(exist_ok=True)
    try:
        # Either force low-mem or try single-shot then fallback
        if lowmem_sp_chunk and lowmem_sp_chunk > 0:
            recon_adjoint_chunked(traj_base, ksp_base, out_base, combine, gpu, sp_total, lowmem_sp_chunk, tmpdir)
            return
        try:
            run_bart(["nufft", "-a", "-t", str(traj_base), str(ksp_base), str(coil_base)], gpu=gpu)
            if combine.lower() == "sos":
                run_bart(["rss", "8", str(coil_base), str(out_base)], gpu=gpu)
            elif combine.lower() == "sens":
                maps = out_base.with_name(out_base.name + "_maps")
                estimate_sens_maps(coil_base, maps, gpu=gpu)
                run_bart(["pics", "-S", str(coil_base), str(maps), str(out_base)], gpu=gpu)
            else:
                raise ValueError("combine must be sos|sens")
        except subprocess.CalledProcessError:
            print("[warn] Single-shot adjoint failed; falling back to spoke-chunked adjoint.")
            # choose a conservative chunk size
            chunk = min(4096, max(1, sp_total // 4))
            recon_adjoint_chunked(traj_base, ksp_base, out_base, combine, gpu, sp_total, chunk, tmpdir)
    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass

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

# ---------- CLI ----------
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
    ap.add_argument("--combine", type=str, default="sos", help="Coil combine for adjoint: sos|sens")
    ap.add_argument("--iterative", action="store_true", help="Use iterative PICS reconstruction")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.0, help="Regularization weight for iterative recon")
    ap.add_argument("--iters", type=int, default=40, help="Iterations for iterative recon")
    ap.add_argument("--wavelets", type=int, default=None, help="Wavelet scale parameter (optional)")
    ap.add_argument("--export-nifti", action="store_true", help="Export NIfTI via bart toimg")
    ap.add_argument("--gpu", action="store_true", help="Attempt GPU; sticky CPU fallback if unsupported")
    ap.add_argument("--debug", action="store_true", help="Print header inference and file checks")
    # optional overrides
    ap.add_argument("--readout", type=int, default=None)
    ap.add_argument("--spokes", type=int, default=None)
    ap.add_argument("--coils", type=int, default=None)
    ap.add_argument("--fid-dtype", type=str, default=None)
    ap.add_argument("--fid-endian", type=str, default=None)
    # low-memory control
    ap.add_argument("--lowmem-sp-chunk", type=int, default=0, help="Force spoke-chunked adjoint NUFFT with this chunk size (0=auto only on failure)")

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

    # Save KSP in non-Cart layout
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
    else:
        ksp_in = ksp_base

    # ---- reconstruction
    if args.iterative:
        out_img = out_base.with_name(out_base.name + "_recon")
        recon_iterative(traj_base, ksp_in, out_img, lam=args.lam, iters=args.iters, wavelets=args.wavelets, gpu=args.gpu)
    else:
        out_img = out_base.with_name(out_base.name + "_adj")
        recon_adjoint(traj_base, ksp_in, args.combine, out_img, gpu=args.gpu,
                      sp_total=sp, lowmem_sp_chunk=args.lowmem_sp_chunk)

    # ---- export
    if args.export_nifti:
        run_bart(["toimg", str(out_img), str(out_base)], gpu=args.gpu)
        print(f"Wrote NIfTI: {out_base}.nii")
    else:
        print(f"Recon complete. BART base: {out_img} (.cfl/.hdr)")

if __name__ == "__main__":
    main()
