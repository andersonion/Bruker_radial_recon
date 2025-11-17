#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bruker_radial_bart.py

Bruker 3D radial reconstruction (BART + NumPy fallback) with:
- Robust FID loader that infers RO/Spokes/Coils and trims padded RO blocks
- STRICT trajectory sourcing:
    (1) $series/traj  -> used as-is
    (2) method PVM_TrajKx/Ky/Kz arrays
    (3) method-driven rebuild (Kronecker or classic GA), no arbitrary fallback
- Sliding-window temporal binning by spokes or time, with --test-volumes limiter
- DCF (pipe-style) and adjoint recon (BART or pure-NumPy Kaiser–Bessel)

CLI examples:
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

  python bruker_radial_bart.py \
    --series "$path" \
    --matrix 256 256 256 \
    --traj file \
    --time-per-frame-ms 200 \
    --tr-ms 1.2 \
    --dcf pipe:8 \
    --combine sos \
    --force-python-adjoint \
    --test-volumes 3 \
    --export-nifti \
    --out "${out%.nii.gz}_SoS" \
    --debug
"""

from __future__ import annotations
import argparse, math, re, shutil, subprocess, sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Iterable
import numpy as np

DEBUG = False
BART_GPU_AVAILABLE = None  # sticky cache


# ------------------------ small utils ------------------------

def dbg(*a):
    if DEBUG:
        print("[debug]", *a)


def bart_exists() -> bool:
    return shutil.which("bart") is not None


def _bart_path() -> str:
    bart = shutil.which("bart")
    if not bart:
        raise RuntimeError("BART not found in PATH")
    return bart


def _bart_supports_gpu(tool: str) -> bool:
    """Detect -g support once and cache (handles 'compiled without GPU' and 'invalid option -g')."""
    global BART_GPU_AVAILABLE
    if BART_GPU_AVAILABLE is not None:
        return BART_GPU_AVAILABLE
    bart = _bart_path()
    try:
        subprocess.run([bart, tool, "-g", "-h"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, check=True)
        BART_GPU_AVAILABLE = True
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or b"").decode(errors="ignore").lower()
        if "compiled without gpu" in msg or "invalid option" in msg or "unknown option" in msg:
            BART_GPU_AVAILABLE = False
        else:
            BART_GPU_AVAILABLE = False
    except Exception:
        BART_GPU_AVAILABLE = False
    return BART_GPU_AVAILABLE


def _run_bart(tool: str, args: List[str], gpu: bool):
    """Run a single BART tool with optional -g; sticky CPU fallback if GPU fails."""
    global BART_GPU_AVAILABLE
    bart = _bart_path()
    if gpu and (BART_GPU_AVAILABLE is None or BART_GPU_AVAILABLE):
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


# ------------------------ CFL I/O (correct newlines!) ------------------------

def _write_hdr(path: Path, dims: List[int]):
    with open(path, "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in dims) + "\n")


def write_cfl(base: Path, arr: np.ndarray, dims16: List[int]) -> None:
    """Write complex array in BART CFL format with explicit 16 dims in .hdr."""
    base = Path(base)
    _write_hdr(base.with_suffix(".hdr"), dims16)
    a = np.asarray(arr, dtype=np.complex64, order="F")
    a.view(np.float32).tofile(base.with_suffix(".cfl"))


def read_cfl(base: Path) -> np.ndarray:
    """Load CFL as complex64."""
    base = Path(base)
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")
    if not (hdr.exists() and cfl.exists()):
        raise FileNotFoundError(f"Missing CFL pair: {base}")
    with open(hdr, "r") as f:
        lines = f.read().strip().split()
    if len(lines) < 2 or lines[0].lower().startswith("#"):
        # First token "#", second token "Dimensions", then dims…
        tokens = []
        with open(hdr, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                tokens += line.split()
        dims = list(map(int, tokens))
    else:
        dims = list(map(int, lines))
    if len(dims) < 1:
        raise ValueError("Bad CFL header")
    data = np.fromfile(cfl, dtype=np.float32)
    if data.size % 2 != 0:
        data = data[:-1]
    cpx = data.view(np.complex64)
    # Build 16-dim shape
    dims16 = dims + [1] * (16 - len(dims))
    sz = 1
    for d in dims16:
        sz *= d
    if cpx.size != sz:
        raise ValueError(f"CFL size mismatch: have {cpx.size}, want {sz}")
    return cpx.reshape(tuple(dims16), order="F")


# ------------------------ Bruker text parsing ------------------------

def _read_text_kv(path: Path) -> Dict[str, str]:
    d: Dict[str, str] = {}
    if not Path(path).exists():
        return d
    key = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("##$"):
                m = re.match(r"##\$(\S+)\s*=\s*(.*)", line)
                if m:
                    key = m.group(1)
                    d[key] = m.group(2).strip()
            elif key is not None:
                d[key] += " " + line.strip()
    return d


def _get_int(keys: Iterable[str], srcs: List[Dict[str, str]]) -> Optional[int]:
    for k in keys:
        for s in srcs:
            if k in s:
                try:
                    return int(re.findall(r"-?\d+", s[k])[0])
                except Exception:
                    pass
    return None


def _get_float(keys: Iterable[str], srcs: List[Dict[str, str]]) -> Optional[float]:
    for k in keys:
        for s in srcs:
            if k in s:
                try:
                    return float(s[k].split()[0])
                except Exception:
                    try:
                        return float(re.findall(r"[+-]?\d+(\.\d+)?([eE][+-]?\d+)?", s[k])[0][0])
                    except Exception:
                        pass
    return None


# ------------------------ FID loader ------------------------

def _parse_acq_size(method: Dict[str, str], acqp: Dict[str, str]) -> Optional[Tuple[int, int, int]]:
    # VisuCoreSize-like hints or PVM_Matrix if present
    for k in ("PVM_Matrix", "PVM_EncMatrix", "PVM_Matrix1"):
        if k in method:
            try:
                toks = list(map(int, re.findall(r"\d+", method[k])))
                if len(toks) >= 3:
                    return toks[0], toks[1], toks[2]
                if len(toks) == 2:
                    return toks[0], toks[1], toks[1]
            except Exception:
                pass
    return None


def _probe_traj_spokes(series_dir: Path, ro_hint: Optional[int]) -> Optional[int]:
    """If a $series/traj exists, infer its spoke count given RO hint; else None."""
    t = Path(series_dir) / "traj"
    if not t.exists():
        return None
    try:
        data = np.fromfile(t, dtype=np.float32)
        if data.size % 3 == 0 and ro_hint:
            sp = (data.size // 3) // ro_hint
            if sp > 0 and 3 * ro_hint * sp == data.size:
                return sp
        # try ASCII
        txt = t.read_text(errors="ignore").strip().split()
        vals = np.array(list(map(float, txt)), dtype=np.float64)
        if vals.size % 3 == 0 and ro_hint:
            sp = (vals.size // 3) // ro_hint
            if sp > 0 and 3 * ro_hint * sp == vals.size:
                return sp
    except Exception:
        return None
    return None


def load_bruker_kspace(
    series_dir: Path,
    matrix_ro_hint: Optional[int] = None,
    spokes: Optional[int] = None,
    readout: Optional[int] = None,
    coils: Optional[int] = None,
    fid_dtype: str = "int32",      # default Bruker int32 FID
    fid_endian: str = "little",    # default
) -> np.ndarray:
    """Returns k-space shaped (RO, Spokes, Coils)."""
    series_dir = Path(series_dir)
    # Pre-existing ksp?
    for base in ("ksp", ):
        cflb = series_dir / base
        if cflb.with_suffix(".cfl").exists() and cflb.with_suffix(".hdr").exists():
            arr = read_cfl(cflb)
            # Expect RO at dim10, Sp at dim11, Coils at dim3 (common BART noncart layout)
            # We'll try to squeeze it back to (RO,Sp,Coils)
            dims = arr.shape
            ro = dims[10] if len(dims) >= 12 else None
            sp = dims[11] if len(dims) >= 12 else None
            nc = dims[3] if len(dims) >= 4 else None
            if ro and sp and nc:
                a = np.asarray(arr, dtype=np.complex64, order="F")
                a = np.moveaxis(a, (10, 11, 3), (0, 1, 2))
                return a[:ro, :sp, :nc]
        npy = series_dir / f"{base}.npy"
        if npy.exists():
            arr = np.load(npy)
            if arr.ndim == 3:
                return arr.astype(np.complex64, copy=False)

    # FID route
    fid_path = series_dir / "fid"
    if not fid_path.exists():
        raise FileNotFoundError("No k-space found (no fid, ksp.cfl, or ksp.npy)")

    method = _read_text_kv(series_dir / "method")
    acqp = _read_text_kv(series_dir / "acqp")

    # endian/dtype from headers if present
    if "BYTORDA" in acqp:
        if "big" in acqp["BYTORDA"].lower():
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

    # hints
    if readout is None:
        acq_size = _parse_acq_size(method, acqp)
        if acq_size:
            readout = acq_size[0]
    if readout is None and matrix_ro_hint:
        readout = matrix_ro_hint
    dbg("readout (hdr/matrix hint):", readout)

    if coils is None:
        coils = _get_int(("PVM_EncNReceivers",), [method]) or 1
    dbg("coils (initial):", coils)

    # other dimension factors (echo/reps/avg/slices)
    extras = {
        "echoes": _get_int(("NECHOES", "ACQ_n_echo_images", "PVM_NEchoImages"), [method, acqp]) or 1,
        "reps": _get_int(("PVM_NRepetitions", "NR"), [method, acqp]) or 1,
        "averages": _get_int(("PVM_NAverages", "NA"), [method, acqp]) or 1,
        "slices": _get_int(("NSLICES", "PVM_SPackArrNSlices"), [method, acqp]) or 1,
    }
    other_dims = 1
    for v in extras.values():
        if isinstance(v, int) and v > 1:
            other_dims *= v
    dbg("other_dims factor:", other_dims, extras)

    # Let traj (if present) inform spokes
    if spokes is None:
        ro_hint_for_traj = readout if readout is not None else matrix_ro_hint
        sp_hint = _probe_traj_spokes(series_dir, ro_hint_for_traj)
        if sp_hint is not None and sp_hint > 0:
            spokes = sp_hint

    # factor total -> stored_ro * (spokes * other_dims) * coils
    def pick_block_and_spokes(per_coil_total: int, readout_hint: Optional[int], spokes_hint: Optional[int]) -> Tuple[int, int]:
        # 1) honor explicit spokes when divides
        if spokes_hint and spokes_hint > 0 and per_coil_total % spokes_hint == 0:
            return per_coil_total // spokes_hint, spokes_hint
        # 2) prefer readout hint when divides
        if readout_hint and readout_hint > 0 and per_coil_total % readout_hint == 0:
            return readout_hint, per_coil_total // readout_hint
        # 3) common block sizes
        BLOCKS = [128, 160, 192, 200, 224, 240, 256, 288, 320, 352, 384, 400, 416, 420, 432, 448,
                  480, 496, 512, 544, 560, 576, 608, 640, 672, 704, 736, 768, 800, 832, 896, 960,
                  992, 1024, 1152, 1280, 1536, 2048]
        for b in [x for x in BLOCKS if not readout_hint or x >= readout_hint]:
            if per_coil_total % b == 0:
                return b, per_coil_total // b
        # 4) fallback near-sqrt
        s = int(round(per_coil_total ** 0.5))
        for d in range(0, s + 1):
            for cand in (s + d, s - d):
                if cand > 0 and per_coil_total % cand == 0:
                    return cand, per_coil_total // cand
        raise ValueError("Could not factor per_coil_total into (stored_ro, spokes).")

    denom = coils * max(1, other_dims)
    if total % denom != 0:
        dbg("total not divisible by coils*other_dims; relaxing extras")
        denom = coils
        if total % denom != 0:
            dbg("still not divisible; relaxing coils->1")
            coils = 1
            denom = 1
            if total % denom != 0:
                raise ValueError("Cannot factor FID length with any (coils, other_dims) combo.")
    per_coil_total = total // denom
    stored_ro, spokes_inf = pick_block_and_spokes(per_coil_total, readout, spokes)
    dbg("stored_ro (block):", stored_ro, " spokes (per extras-collapsed):", spokes_inf)
    spokes_final = spokes_inf * max(1, other_dims)
    if stored_ro * spokes_final * coils != total:
        raise ValueError("Internal factoring error (stored_ro*spokes_final*coils != total)")

    ksp_blk = np.reshape(cpx, (stored_ro, spokes_final, coils), order="F")
    if readout is not None and stored_ro >= readout:
        ksp = ksp_blk[:readout, :, :]
        dbg("trimmed RO from", stored_ro, "to", readout)
    else:
        ksp = ksp_blk
        if readout is None:
            readout = stored_ro
    dbg("final k-space shape:", ksp.shape, "(RO, Spokes, Coils)")
    return ksp.astype(np.complex64, copy=False)


# ------------------------ Trajectory readers/builders ------------------------

def _read_traj_file(series_dir: Path, ro: int, spokes: int) -> Optional[np.ndarray]:
    """Read $series/traj as binary float32/64 or ASCII; expect flat xyz with length 3*ro*spokes."""
    t = Path(series_dir) / "traj"
    if not t.exists():
        return None
    # try binary float32
    for dt in (np.float32, np.float64):
        try:
            arr = np.fromfile(t, dtype=dt)
            if arr.size == 3 * ro * spokes:
                arr = arr.astype(np.float32, copy=False)
                return arr.reshape(3, ro, spokes, order="F")
        except Exception:
            pass
    # try ASCII
    try:
        vals = list(map(float, t.read_text(errors="ignore").strip().split()))
        if len(vals) == 3 * ro * spokes:
            arr = np.array(vals, dtype=np.float32)
            return arr.reshape(3, ro, spokes, order="F")
    except Exception:
        pass
    return None


def _read_traj_from_method_arrays(series_dir: Path, ro: int) -> Optional[Tuple[np.ndarray, int]]:
    """Read PVM_TrajKx/Ky/Kz arrays from method if present; return (traj, spokes)."""
    method = _read_text_kv(Path(series_dir) / "method")
    keys = ["PVM_TrajKx", "PVM_TrajKy", "PVM_TrajKz"]
    if not all(k in method for k in keys):
        return None
    try:
        kx = np.array(list(map(float, re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", method["PVM_TrajKx"]))), dtype=np.float32)
        ky = np.array(list(map(float, re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", method["PVM_TrajKy"]))), dtype=np.float32)
        kz = np.array(list(map(float, re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", method["PVM_TrajKz"]))), dtype=np.float32)
        if not (kx.size == ky.size == kz.size):
            return None
        if kx.size % (3) == 0 and ro > 0:
            spokes = kx.size // ro
            if spokes > 0 and ro * spokes == kx.size:
                traj = np.stack([kx.reshape(ro, spokes, order="F"),
                                 ky.reshape(ro, spokes, order="F"),
                                 kz.reshape(ro, spokes, order="F")], axis=0)
                return traj, spokes
    except Exception:
        return None
    return None


def _dirs_fibonacci(spokes: int) -> np.ndarray:
    phi_inc = 2.39996322972865332  # 2π/φ^2
    k = np.arange(spokes, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * k / spokes
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = k * phi_inc
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.stack([x, y, z], axis=1)  # (spokes,3)


def _dirs_kronecker(spokes: int) -> np.ndarray:
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    inc1, inc2 = invphi, invphi * invphi
    k = np.arange(spokes, dtype=np.float64)
    u = (k * inc1) % 1.0
    v = (k * inc2) % 1.0
    theta = np.arccos(1.0 - 2.0 * u)
    phi = 2.0 * np.pi * v
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=1)


def _derive_tr_ms(method: dict, acqp: dict) -> Optional[float]:
    for k in ("PVM_RepetitionTime", "ACQ_repetition_time"):
        if k in method:
            try: return float(method[k].split()[0])
            except: pass
        if k in acqp:
            try: return float(acqp[k].split()[0])
            except: pass
    return None


def probe_spokes_from_method(series_dir: Path) -> Optional[int]:
    m = _read_text_kv(Path(series_dir) / "method")
    a = _read_text_kv(Path(series_dir) / "acqp")
    for k in ("GA_NSpokesEff", "NPro"):
        v = _get_int((k,), [m, a])
        if v and v > 0:
            return v
    return None


def derive_traj_from_method(series_dir: Path, ro: int, spokes: int, matrix: Tuple[int, int, int]) -> np.ndarray:
    """Rebuild trajectory deterministically from GA/Kronecker flags in method. No arbitrary fallback."""
    m = _read_text_kv(Path(series_dir) / "method")
    a = _read_text_kv(Path(series_dir) / "acqp")
    # Spokes authority
    npro   = _get_int(("NPro",), [m, a])
    ga_eff = _get_int(("GA_NSpokesEff",), [m])
    sp_hdr = ga_eff if (ga_eff and ga_eff > 0) else npro
    if sp_hdr and sp_hdr > 0 and sp_hdr != spokes:
        spokes = min(sp_hdr, spokes)

    # GA/Kronecker decision
    ga_mode = _get_int(("GA_Mode",), [m])    # treat nonzero as Kronecker unless you provide enums
    use_fib = _get_int(("GA_UseFibonacci",), [m])
    use_kron = False
    if ga_mode is not None:
        use_kron = (ga_mode != 0)
    elif use_fib is not None:
        use_kron = (use_fib == 0)

    dirs = _dirs_kronecker(spokes) if use_kron else _dirs_fibonacci(spokes)

    NX, NY, NZ = matrix
    kmax = 0.5 * float(max(NX, NY, NZ))
    t = (np.arange(ro, dtype=np.float64) + 0.5) / ro
    r = t * kmax
    x = r[:, None] * dirs[:, 0][None, :]
    y = r[:, None] * dirs[:, 1][None, :]
    z = r[:, None] * dirs[:, 2][None, :]
    traj = np.stack([x, y, z], axis=0).astype(np.float32)
    return traj


def get_bruker_traj(series_dir: Path, ro: int, spokes: int, matrix: Tuple[int, int, int]) -> np.ndarray:
    """STRICT trajectory sourcing order."""
    # 1) raw file
    t = _read_traj_file(series_dir, ro, spokes)
    if t is not None:
        return t
    # 2) method arrays
    mt = _read_traj_from_method_arrays(series_dir, ro)
    if mt is not None:
        traj, sp_m = mt
        if sp_m != spokes:
            sp_use = min(sp_m, spokes)
            traj = traj[:, :, :sp_use]
        return traj
    # 3) rebuild from method flags (Kronecker/GA)
    sp_hdr = probe_spokes_from_method(series_dir)
    if sp_hdr is not None and sp_hdr > 0 and sp_hdr != spokes:
        spokes = min(sp_hdr, spokes)
    traj = derive_traj_from_method(series_dir, ro=ro, spokes=spokes, matrix=matrix)
    if traj.shape != (3, ro, spokes):
        raise ValueError(f"Derived trajectory has shape {traj.shape}, expected (3,{ro},{spokes})")
    return traj


# ------------------------ DCF (pipe, NumPy) ------------------------

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


# ------------------------ Layout helpers for BART ------------------------

def ksp_to_bart_noncart(ksp: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Map (RO,Sp,Coils) -> dims[16] with RO@10, Sp@11, Coils@3."""
    ro, sp, nc = ksp.shape
    arr = ksp.astype(np.complex64, order="F")
    arr = arr.reshape(ro, sp, nc, *([1] * 13), order="F")
    arr = np.transpose(arr, (3, 4, 5, 2, 6, 7, 8, 9, 12, 13, 0, 1, 10, 11, 14, 15))
    dims = [1] * 16
    dims[3] = nc
    dims[10] = ro
    dims[11] = sp
    return arr, dims


def traj_to_bart_noncart(traj: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Map (3,RO,Sp) -> dims[16] with 3@0, RO@10, Sp@11."""
    _, ro, sp = traj.shape
    arr = traj.astype(np.complex64, order="F")
    arr = arr.reshape(3, ro, sp, *([1] * 13), order="F")
    arr = np.transpose(arr, (0, 3, 4, 5, 6, 7, 8, 9, 12, 13, 1, 2, 10, 11, 14, 15))
    dims = [1] * 16
    dims[0] = 3
    dims[10] = ro
    dims[11] = sp
    return arr, dims


# ------------------------ NumPy adjoint (KB gridder) ------------------------

def _kb(u: np.ndarray, width: float, beta: float) -> np.ndarray:
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
            wx = _kb(xr - gx[t, s], kb_width, kb_beta)
            wy = _kb(yr - gy[t, s], kb_width, kb_beta)
            wz = _kb(zr - gz[t, s], kb_width, kb_beta)
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
    img = img[x0: x0 + NX, y0: y0 + NY, z0: z0 + NZ, :]
    img *= deapx[:NX, None, None, None]
    img *= deapy[None, :NY, None, None]
    img *= deapz[None, None, :NZ, None]
    return img.astype(np.complex64)


# ------------------------ Recon pipelines ------------------------

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
    write_cfl(coil_base, coil_img, [NX, NY, NZ, coil_img.shape[3]] + [1] * 12)
    if combine.lower() == "sos":
        rss = np.sqrt(np.sum(np.abs(coil_img) ** 2, axis=3)).astype(np.complex64)
        write_cfl(out_base, rss, [NX, NY, NZ] + [1] * 13)
    elif combine.lower() == "sens":
        maps = out_base.with_name(out_base.name + "_maps_py")
        estimate_sens_maps(coil_base, maps, gpu=False)
        run_bart(["pics", "-S", str(coil_base), str(maps), str(out_base)], gpu=False)
    else:
        raise ValueError("combine must be sos|sens")


# ------------------------ Frame binning ------------------------

def frame_starts(total_spokes: int, spokes_per_frame: int, frame_shift: Optional[int]) -> Iterable[int]:
    step = frame_shift if frame_shift and frame_shift > 0 else spokes_per_frame
    for s in range(0, total_spokes - spokes_per_frame + 1, step):
        yield s


# ------------------------ main ------------------------

def main():
    global DEBUG
    ap = argparse.ArgumentParser(description="Bruker 3D radial reconstruction using BART/NumPy with strict trajectory sourcing")
    ap.add_argument("--series", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--matrix", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))
    ap.add_argument("--traj", choices=["golden", "file"], default="file")
    ap.add_argument("--traj-file", type=Path, help="Explicit trajectory file (used only if $series/traj missing)")
    ap.add_argument("--dcf", type=str, default="none", help="none | pipe:N")
    ap.add_argument("--combine", type=str, default="sos", help="sos|sens")
    ap.add_argument("--iterative", action="store_true", help="(placeholder) iterative recon hook")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=40)
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
    ap.add_argument("--frame-shift", type=int, default=None)
    ap.add_argument("--tr-ms", type=float, default=None)
    ap.add_argument("--test-volumes", type=int, default=None)

    args = ap.parse_args()
    DEBUG = args.debug

    if not bart_exists() and not args.force_python_adjoint:
        print("ERROR: BART not found on PATH. Use --force-python-adjoint or install BART.", file=sys.stderr)
        sys.exit(1)

    series_dir: Path = args.series
    out_base: Path = args.out
    NX, NY, NZ = args.matrix

    # spokes hint from traj file (if present) to keep k-space alignment
    sp_hint = _probe_traj_spokes(series_dir, NX)
    if DEBUG:
        print(f"[debug] traj-derived spokes hint: {sp_hint}")

    # load k-space
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

    # trajectory: STRICT order
    # 1) $series/traj
    bruker_traj = _read_traj_file(series_dir, ro, sp_total)
    if bruker_traj is None:
        # 2) method arrays
        mt = _read_traj_from_method_arrays(series_dir, ro)
        if mt is not None:
            bruker_traj, sp_m = mt
            if sp_m != sp_total:
                print(f"[warn] method-derived spokes={sp_m} != k-space spokes={sp_total}; trimming.")
                sp_use = min(sp_m, sp_total)
                bruker_traj = bruker_traj[:, :, :sp_use]
                ksp = ksp[:, :sp_use, :]
                sp_total = sp_use
        else:
            # 3) rebuild from method GA/Kronecker
            sp_hdr = probe_spokes_from_method(series_dir)
            if sp_hdr is not None and sp_hdr > 0 and sp_hdr != sp_total:
                ksp = ksp[:, :min(sp_hdr, sp_total), :]
                sp_total = ksp.shape[1]
            bruker_traj = derive_traj_from_method(series_dir, ro=ro, spokes=sp_total, matrix=(NX, NY, NZ))
    if bruker_traj.shape != (3, ro, sp_total):
        raise ValueError(f"trajectory must have shape (3,{ro},{sp_total}); got {bruker_traj.shape}")

    # temporal binning
    spokes_per_frame = args.spokes_per_frame
    if spokes_per_frame is None and args.time_per_frame_ms is not None:
        tr_ms = args.tr_ms if args.tr_ms is not None else _derive_tr_ms(method, acqp)
        if tr_ms is None or tr_ms <= 0:
            raise ValueError("--time-per-frame-ms provided but TR unknown. Pass --tr-ms explicitly.")
        spokes_per_frame = max(1, int(round(args.time_per_frame_ms / tr_ms)))
        print(f"[info] Using spokes_per_frame={spokes_per_frame} (time_per_frame_ms={args.time_per_frame_ms}, TR={tr_ms} ms)")
    if spokes_per_frame is None:
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

    # per-frame loop
    for fi, s0 in enumerate(starts, 1):
        s1 = s0 + spokes_per_frame
        if s1 > sp_total:
            print(f"[warn] Skipping last partial window {s0}:{s1}")
            break
        ksp_f = ksp[:, s0:s1, :]
        traj_f = bruker_traj[:, :, s0:s1]

        # DCF
        dcf = None
        if args.dcf.lower().startswith("pipe"):
            nit = 10
            if ":" in args.dcf:
                try: nit = int(args.dcf.split(":", 1)[1])
                except: pass
            dcf = dcf_pipe_numpy(traj_f, iters=nit, grid_shape=(NX, NY, NZ))

        vol_base = out_base.with_name(out_base.name + f"_vol{fi:05d}")
        if args.force_python_adjoint:
            ksp_in = ksp_f if dcf is None else (ksp_f * dcf[..., None])
            recon_adjoint_python(traj_f, ksp_in, dcf, (NX, NY, NZ), vol_base, combine=args.combine)
        else:
            # BART path
            ksp16, kspdims = ksp_to_bart_noncart(ksp_f if dcf is None else (ksp_f * dcf[..., None]))
            traj16, trajdims = traj_to_bart_noncart(traj_f)
            ksp_base = vol_base.with_name(vol_base.name + "_ksp")
            traj_base = vol_base.with_name(vol_base.name + "_traj")
            write_cfl(ksp_base, ksp16, kspdims)
            write_cfl(traj_base, traj16, trajdims)
            try:
                recon_adjoint_bart(traj_base, ksp_base, vol_base, combine=args.combine, gpu=args.gpu)
            except subprocess.CalledProcessError:
                print("[warn] BART adjoint failed; falling back to pure-NumPy adjoint for this frame.")
                ksp_in = ksp_f if dcf is None else (ksp_f * dcf[..., None])
                recon_adjoint_python(traj_f, ksp_in, dcf, (NX, NY, NZ), vol_base, combine=args.combine)

        if args.export_nifti:
            run_bart(["toimg", str(vol_base), str(vol_base)], gpu=False)
        print(f"[info] Frame {fi}/{nframes} done -> {vol_base}")

    print("[info] All requested frames complete.")


if __name__ == "__main__":
    main()
