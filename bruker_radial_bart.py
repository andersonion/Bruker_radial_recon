#!/usr/bin/env python3
"""
bruker_radial_bart.py

Bruker 3D radial DCE recon using BART + sliding-window binning and synthetic
Kronecker / LinZ golden-angle trajectories when no traj file is present.

Workflow
- Load Bruker FID, infer (RO, Spokes, Coils), trim padded RO blocks.
- Trajectory:
  * If $series/traj.{cfl,hdr} or $series/traj exists, use it.
  * Else, build synthetic traj using Kronecker or LinZ_GA math:
      --traj-mode kron | linz | linear_z
- DCF:
  * "pipe:N" → simple iterative density compensation on a 3D grid (NumPy).
- Temporal binning:
  * --spokes-per-frame N
  * --frame-shift M (default = N)
- Recon:
  * Per-frame BART adjoint NUFFT:
    bart nufft -i -d NX:NY:NZ -t traj ksp coil
    + rss 8 coil img
  * If BART nufft fails, the script raises.
- Output:
  * Per-frame images are handled in CFL only.
  * At the end, stack all frames into a single 4D volume and write:
      OUT_4D.nii.gz              (t axis last)
      OUT_QC_vol00001.nii.gz     (QC frame, abs-magnitude)

Example
-------
python bruker_radial_bart.py \
  --series "$path" \
  --matrix 256 256 256 \
  --spokes-per-frame 800 \
  --frame-shift 100 \
  --dcf pipe:10 \
  --combine sos \
  --traj-mode kron \
  --out "${out%.nii.gz}_SoS"
"""

from __future__ import annotations
import argparse, math, shutil, subprocess, sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Iterable

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None

DEBUG = False


def dbg(*a):
    if DEBUG:
        print("[debug]", *a)


# ---------- CFL I/O ----------

def _write_hdr(path: Path, dims: List[int]):
    """Write a BART .hdr file with up to 16 dims."""
    with open(path, "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in dims) + "\n")


def _write_cfl(base: Path, a: np.ndarray, dims: Optional[List[int]] = None):
    """
    Write a complex64 CFL with a 16-dim header.
    a: complex ndarray, any shape compatible with dims.
    dims: if provided, will be padded to 16; otherwise uses a.shape.
    """
    base = Path(base)

    if dims is None:
        shape = list(a.shape)
    else:
        shape = list(dims)

    if len(shape) < 16:
        shape = shape + [1] * (16 - len(shape))

    _write_hdr(base.with_suffix(".hdr"), shape)

    arr = np.ascontiguousarray(a.astype(np.complex64))
    arr.view(np.float32).tofile(base.with_suffix(".cfl"))


def read_cfl(base: Path) -> np.ndarray:
    """Read a BART CFL into a complex64 ndarray with Fortran ordering."""
    base = Path(base)
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")

    with open(hdr, "r") as f:
        tokens = f.read().split()

    dims: List[int] = []
    for tok in tokens:
        try:
            dims.append(int(tok))
        except ValueError:
            continue

    if not dims:
        raise ValueError(f"Could not parse dims from {hdr}")

    n = int(np.prod(dims))
    data = np.fromfile(cfl, dtype=np.complex64, count=n)
    if data.size != n:
        raise ValueError(f"CFL size mismatch: got {data.size}, expected {n}")

    return data.reshape(dims, order="F")


# ---------- Bruker header helpers ----------

def _read_text_kv(path: Path) -> Dict[str, str]:
    """Parse simple Bruker text header into key->string."""
    d: Dict[str, str] = {}
    if not path.exists():
        return d

    key = None
    val_lines: List[str] = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("##$"):
            if key is not None:
                d[key] = " ".join(val_lines).strip()
            parts = line[3:].split("=", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                val_lines = [parts[1].strip()]
            else:
                key = parts[0].strip()
                val_lines = []
        elif line.startswith("$$"):
            continue
        else:
            if key is not None:
                val_lines.append(line.strip())
    if key is not None and key not in d:
        d[key] = " ".join(val_lines).strip()
    return d


def _get_int_from_headers(keys: List[str], dicts: List[Dict[str, str]]) -> Optional[int]:
    for k in keys:
        for d in dicts:
            if k in d:
                txt = d[k].strip()
                txt = txt.replace("(", " ").replace(")", " ").replace(",", " ")
                for tok in txt.split():
                    try:
                        return int(tok)
                    except ValueError:
                        continue
    return None


def _parse_acq_size(method: Dict[str, str], acqp: Dict[str, str]) -> Optional[Tuple[int, int, int]]:
    txt = method.get("PVM_EncMatrix", "") or method.get("ACQ_size", "")
    if not txt:
        return None
    nums: List[int] = []
    for tok in txt.replace("(", " ").replace(")", " ").replace(",", " ").split():
        try:
            nums.append(int(tok))
        except ValueError:
            continue
    if len(nums) >= 3:
        return nums[0], nums[1], nums[2]
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
    Load Bruker FID and reshape to (RO, Spokes, Coils) with:
    - readout inferred from PVM_EncMatrix or ACQ_size (or matrix_ro_hint)
    - coils from PVM_EncNReceivers (or 1)
    - extras (echoes,reps,averages,slices) collapsed into Spokes
    - handles padded RO blocks (e.g. 420 -> 512) and trims to readout
    """
    series_dir = Path(series_dir)
    fid_path = series_dir / "fid"
    if not fid_path.exists():
        raise FileNotFoundError(f"No fid in {series_dir}")

    method = _read_text_kv(series_dir / "method")
    acqp = _read_text_kv(series_dir / "acqp")

    # Endian and dtype from acqp
    if "BYTORDA" in acqp and "big" in acqp["BYTORDA"].lower():
        fid_endian = "big"
    if "ACQ_word_size" in acqp:
        if "16" in acqp["ACQ_word_size"]:
            fid_dtype = "int16"
        elif "32" in acqp["ACQ_word_size"]:
            fid_dtype = "int32"

    dtype_map = {
        "int16": np.int16,
        "int32": np.int32,
        "float32": np.float32,
        "float64": np.float64,
    }
    if fid_dtype not in dtype_map:
        raise ValueError("fid_dtype must be int16/int32/float32/float64")

    dt = dtype_map[fid_dtype]
    raw = np.fromfile(fid_path, dtype=dt)
    if fid_endian == "big":
        raw = raw.byteswap().newbyteorder()
    if raw.size % 2 != 0:
        raw = raw[:-1]

    cpx = raw.astype(np.float32).view(np.complex64)
    total = cpx.size
    dbg("total complex samples:", total, "(dtype:", fid_dtype, ", endian:", fid_endian, ")")

    # readout from headers / matrix
    if readout is None:
        acq_size = _parse_acq_size(method, acqp)
        if acq_size:
            dbg("PVM_EncMatrix/ACQ_size:", acq_size)
            readout = acq_size[0]
    if matrix_ro_hint is not None:
        if readout is None or readout < matrix_ro_hint:
            dbg("overriding header readout", readout, "to matrix_ro_hint", matrix_ro_hint)
            readout = matrix_ro_hint
    dbg("readout (final hint):", readout)

    # coils from PVM_EncNReceivers (fallback to 1)
    if coils is None:
        nrec = _get_int_from_headers(["PVM_EncNReceivers"], [method])
        if nrec and nrec > 0:
            coils = nrec
    if coils is None or coils <= 0:
        coils = 1
    dbg("coils initial:", coils)

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

    # factor total as stored_ro * spokes_total * coils * other_dims
    denom = coils * max(1, other_dims)
    if total % denom != 0:
        dbg("total not divisible by coils*other_dims; relaxing extras")
        denom = coils
        if total % denom != 0:
            dbg("still not divisible; relaxing coils->1")
            coils = 1
            denom = coils
            if total % denom != 0:
                raise ValueError("Cannot factor FID length (coils / extras).")

    per_coil_total = total // denom

    def pick_block_and_spokes(
        per_coil_total: int,
        readout_hint: Optional[int],
        spokes_hint: Optional[int],
    ) -> Tuple[int, int]:
        # If caller gave a spokes hint and it divides, use it
        if spokes_hint and spokes_hint > 0 and per_coil_total % spokes_hint == 0:
            return per_coil_total // spokes_hint, spokes_hint

        # Common stored RO block sizes Bruker likes to pad to
        BLOCKS = [
            128, 160, 192, 200, 224, 240, 256, 288, 320, 352, 384, 400, 416, 420, 432,
            448, 480, 496, 512, 544, 560, 576, 608, 640, 672, 704, 736, 768, 800, 832,
            896, 960, 992, 1024, 1152, 1280, 1536, 2048,
        ]

        if readout_hint and readout_hint > 0:
            if per_coil_total % readout_hint == 0:
                return readout_hint, per_coil_total // readout_hint
            candidates = [b for b in BLOCKS if b >= readout_hint]
        else:
            candidates = BLOCKS

        for b in candidates:
            if per_coil_total % b == 0:
                return b, per_coil_total // b

        # last-resort brute factor around sqrt
        s = int(round(per_coil_total**0.5))
        for d in range(0, s + 1):
            for cand in (s + d, s - d):
                if cand > 0 and per_coil_total % cand == 0:
                    return cand, per_coil_total // cand

        raise ValueError("Could not factor per_coil_total into (stored_ro, spokes).")

    stored_ro, spokes_inf = pick_block_and_spokes(per_coil_total, readout, spokes)
    dbg("stored_ro:", stored_ro, "spokes_per_extras:", spokes_inf)

    spokes_final = spokes_inf * max(1, other_dims)
    if stored_ro * spokes_final * coils != total:
        raise ValueError("Internal factoring error stored_ro*spokes_final*coils != total")

    ksp_blk = np.reshape(cpx, (stored_ro, spokes_final, coils), order="F")

    # Trim padded readout
    if readout is not None and stored_ro >= readout:
        ksp = ksp_blk[:readout, :, :]
        dbg("trimmed RO from", stored_ro, "to", readout)
    else:
        ksp = ksp_blk
        if readout is None:
            readout = stored_ro

    dbg("final k-space shape:", ksp.shape, "(RO,Spokes,Coils)")
    return ksp


# ---------- GA / Kronecker math (from your Bruker sequence) ----------

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
    p = fib_prev(fk)
    return fib_prev(p)


def uv_to_dir(u: float, v: float) -> Tuple[float, float, float]:
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


def kron_dir(i: int, N: int) -> Tuple[float, float, float]:
    M = fib_closest_ge(N)
    q1 = fib_prev(M)
    q2 = fib_prev2(M)
    j = i % M
    u = ((j * q1) % M + 0.5) / float(M)
    v = ((j * q2) % M + 0.5) / float(M)
    return uv_to_dir(u, v)


def linz_ga_dir(i: int, N: int) -> Tuple[float, float, float]:
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


def build_synthetic_traj(mode: str, ro: int, sp_total: int) -> np.ndarray:
    """
    Build a synthetic 3D radial trajectory with either Kronecker or LinZ_GA.

    mode: "kron" or "linz"
    returns traj with shape (3, RO, Spokes), scaled in pixel_size/FOV units
            approx in [-0.5, 0.5] radial range.
    """
    mode = mode.lower()
    N = sp_total

    kx = np.zeros((ro, N), dtype=np.float32)
    ky = np.zeros((ro, N), dtype=np.float32)
    kz = np.zeros((ro, N), dtype=np.float32)

    half = ro / 2.0
    r = (np.arange(ro, dtype=np.float32) - half) / ro  # ~[-0.5, 0.5]

    for i in range(N):
        if mode == "kron":
            dx, dy, dz = kron_dir(i, N)
        else:
            dx, dy, dz = linz_ga_dir(i, N)
        kx[:, i] = r * dx
        ky[:, i] = r * dy
        kz[:, i] = r * dz

    traj = np.stack([kx, ky, kz], axis=0)
    dbg("synthetic traj shape:", traj.shape, "mode:", mode)
    return traj


# ---------- DCF (Pipe-style, NumPy) ----------

def _normalize_traj_to_grid(traj: np.ndarray, grid_shape: Tuple[int, int, int]):
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
    A simple iterative "pipe" density compensation:
      w_{n+1} = w_n / (A^H A w_n)
    implemented via gridding counts on a Cartesian grid.
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


# ---------- BART wrappers ----------

def bart_exists() -> bool:
    return shutil.which("bart") is not None


def _bart_path() -> str:
    bart = shutil.which("bart")
    if not bart:
        raise RuntimeError("BART not found in PATH")
    return bart


def _run_bart(tool: str, args: List[str]):
    bart = _bart_path()
    cmd = [bart, tool] + args
    print("[bart]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_bart(args: List[str]):
    if not args:
        raise ValueError("Empty BART command")
    _run_bart(args[0], args[1:])


# ---------- BART non-cartesian layouts ----------

def ksp_to_bart_noncart(ksp: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Map ksp (RO, Spokes, Coils) -> BART dims:
      dims[3]  = Coils
      dims[10] = RO
      dims[11] = Spokes
    Data stays in shape (RO, Spokes, Coils) but header tells BART how to interpret.
    """
    ro, sp, nc = ksp.shape
    arr = np.asfortranarray(ksp.astype(np.complex64))
    dims = [1] * 16
    dims[3] = nc
    dims[10] = ro
    dims[11] = sp
    return arr.reshape(ro, sp, nc, order="F"), dims


def traj_to_bart_noncart(traj: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Map traj (3, RO, Spokes) -> BART dims:
      dims[0]  = 3
      dims[10] = RO
      dims[11] = Spokes
    """
    _, ro, sp = traj.shape
    arr = np.asfortranarray(traj.astype(np.complex64))
    dims = [1] * 16
    dims[0] = 3
    dims[10] = ro
    dims[11] = sp
    return arr.reshape(3, ro, sp, order="F"), dims


def bart_recon_frame(
    traj: np.ndarray,
    ksp: np.ndarray,
    out_base: Path,
    combine: str,
    matrix: Tuple[int, int, int],
):
    """
    One-frame BART adjoint NUFFT:
      nufft -i -d NX:NY:NZ -t traj ksp coil
      rss 8 coil img
    """
    NX, NY, NZ = matrix
    ro, sp, nc = ksp.shape
    dbg("bart_recon_frame: ro,sp,nc =", ro, sp, nc)

    ksp_arr, ksp_dims = ksp_to_bart_noncart(ksp)
    traj_arr, traj_dims = traj_to_bart_noncart(traj)

    ksp_base = out_base.with_name(out_base.name + "_ksp")
    traj_base = out_base.with_name(out_base.name + "_traj")

    _write_cfl(ksp_base, ksp_arr, ksp_dims)
    _write_cfl(traj_base, traj_arr, traj_dims)

    coil_base = out_base.with_name(out_base.name + "_coil")

    # Make sure stale coil.cfl/hdr from previous runs don't confuse BART
    for ext in (".cfl", ".hdr"):
        p = coil_base.with_suffix(ext)
        if p.exists():
            p.unlink()

    # BART adjoint NUFFT (image size forced by -d)
    run_bart(
        [
            "nufft",
            "-i",
            "-d",
            f"{NX}:{NY}:{NZ}",
            "-t",
            str(traj_base),
            str(ksp_base),
            str(coil_base),
        ]
    )

    if combine.lower() == "sos":
        run_bart(["rss", "8", str(coil_base), str(out_base)])
    else:
        raise ValueError("Only combine=sos currently supported.")


def cfl_to_3d_numpy(base: Path, NX: int, NY: int, NZ: int) -> np.ndarray:
    """
    Load a BART CFL image and crop to (NX,NY,NZ).
    Assumes spatial dims are leading after squeeze.
    """
    arr = read_cfl(base)
    arr = np.squeeze(arr)

    if arr.ndim < 3:
        raise ValueError(f"CFL image ndim={arr.ndim} < 3")

    if arr.shape[0] < NX or arr.shape[1] < NY or arr.shape[2] < NZ:
        raise ValueError(
            f"BART image size {arr.shape} too small for requested {NX}x{NY}x{NZ}"
        )

    img = arr[0:NX, 0:NY, 0:NZ]
    while img.ndim > 3:
        img = img[..., 0]
    return np.asarray(img, dtype=np.float32)


# ---------- Frame binning ----------

def frame_starts(total_spokes: int, spokes_per_frame: int, frame_shift: Optional[int]) -> Iterable[int]:
    step = frame_shift if (frame_shift is not None and frame_shift > 0) else spokes_per_frame
    for s in range(0, total_spokes - spokes_per_frame + 1, step):
        yield s


# ---------- CLI / main ----------

def main():
    global DEBUG
    ap = argparse.ArgumentParser(description="Bruker 3D radial DCE recon with BART (sliding window)")

    ap.add_argument("--series", type=Path, required=True, help="Bruker series directory (contains fid, acqp, method, etc.)")
    ap.add_argument("--out", type=Path, required=True, help="Base output name (without .nii.gz)")
    ap.add_argument("--matrix", type=int, nargs=3, required=True, metavar=("NX", "NY", "NZ"))

    ap.add_argument("--dcf", type=str, default="none", help="none | pipe:N   (N iterations of Pipe-style DCF)")
    ap.add_argument("--combine", type=str, default="sos", help="sos (currently only option)")

    ap.add_argument(
        "--traj-mode",
        type=str,
        choices=["kron", "linz", "linear_z"],
        default="kron",
        help="Synthetic trajectory mode if no traj file is present: kron (Kronecker) or linz (LinZ GA).",
    )

    ap.add_argument("--spokes-per-frame", type=int, default=None, help="Spokes per volume for sliding window.")
    ap.add_argument(
        "--frame-shift",
        type=int,
        default=None,
        help="Spoke shift between frames (default = spokes-per-frame, i.e., no overlap).",
    )

    ap.add_argument(
        "--qc-frame",
        type=int,
        default=1,
        help="Which frame index (1-based) to write as separate QC NIfTI.",
    )
    ap.add_argument("--test-volumes", type=int, default=None, help="If set, reconstruct only this many frames.")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()
    DEBUG = args.debug

    if not bart_exists():
        print("ERROR: BART not found on PATH.", file=sys.stderr)
        sys.exit(1)

    if nib is None:
        print("WARNING: nibabel not available; NIfTI export will be skipped.", file=sys.stderr)

    series_dir: Path = args.series
    out_base: Path = args.out
    NX, NY, NZ = args.matrix

    # Load k-space
    ksp = load_bruker_kspace(series_dir, matrix_ro_hint=NX)
    ro, sp_total, nc = ksp.shape
    print(f"[info] Loaded k-space: RO={ro}, Spokes={sp_total}, Coils={nc}")

    # Trajectory: prefer $series/traj.cfl/.hdr, then binary traj, else synthetic GA/Kronecker
    traj_path = series_dir / "traj"

    if traj_path.with_suffix(".cfl").exists() and traj_path.with_suffix(".hdr").exists():
        traj = read_cfl(traj_path)
        if traj.shape[0] != 3:
            raise ValueError(f"traj.cfl dim0 must be 3, got {traj.shape}")
        if traj.shape[1] != ro or traj.shape[2] != sp_total:
            print(f"[warn] traj shape {traj.shape} != (3,{ro},{sp_total}); trimming to match k-space.")
            ro_use = min(ro, traj.shape[1])
            sp_use = min(sp_total, traj.shape[2])
            traj = traj[:, :ro_use, :sp_use]
            ksp = ksp[:ro_use, :sp_use, :]
            ro, sp_total, nc = ksp.shape
    elif traj_path.exists():
        # Raw Bruker-like traj blob (float32 or float64) with 3*RO*Spokes entries
        data = np.fromfile(traj_path, dtype=np.float32)
        if data.size != 3 * ro * sp_total:
            data = np.fromfile(traj_path, dtype=np.float64)
        if data.size != 3 * ro * sp_total:
            raise ValueError(
                f"traj file size mismatch: got {data.size} elements, expected {3*ro*sp_total}."
            )
        traj = data.astype(np.float32).reshape(3, ro, sp_total, order="F")
    else:
        mode = "kron" if args.traj_mode == "kron" else "linz"
        print(f"[info] No traj file; building synthetic {mode} trajectory from GA/Kronecker math.")
        traj = build_synthetic_traj(mode, ro, sp_total)

    if traj.shape != (3, ro, sp_total):
        raise ValueError(f"trajectory must have shape (3,{ro},{sp_total}); got {traj.shape}")

    # DCF (optional)
    dcf = None
    if args.dcf.lower().startswith("pipe"):
        nit = 10
        if ":" in args.dcf:
            try:
                nit = int(args.dcf.split(":", 1)[1])
            except Exception:
                pass
        print(f"[info] Computing Pipe-style DCF (iters={nit}) ...")
        dcf = dcf_pipe_numpy(traj, iters=nit, grid_shape=(NX, NY, NZ))

    # Sliding-window setup
    spf = args.spokes_per_frame or min(sp_total, 1000)
    shift = args.frame_shift if args.frame_shift not in (None, 0) else spf

    starts = list(frame_starts(sp_total, spf, shift))
    if args.test_volumes is not None:
        starts = starts[: max(0, int(args.test_volumes))]

    if not starts:
        raise ValueError("No frames to reconstruct with the chosen (spokes_per_frame, frame_shift).")

    print(f"[info] Sliding-window frames: {len(starts)} (spf={spf}, shift={shift})")

    # Recon all frames, gather into 4D
    frames: List[np.ndarray] = []

    for fi, s0 in enumerate(starts, 1):
        s1 = s0 + spf
        if s1 > sp_total:
            print(f"[warn] Skipping partial frame {s0}:{s1} (> total spokes={sp_total}).")
            break

        ksp_f = ksp[:, s0:s1, :]
        traj_f = traj[:, :, s0:s1]

        if dcf is not None:
            dcf_f = dcf[:, s0:s1]
            ksp_w = ksp_f * dcf_f[..., None]
        else:
            ksp_w = ksp_f

        vol_cfl_base = out_base.with_name(out_base.name + f"_vol{fi:05d}")

        # BART NUFFT for this frame
        bart_recon_frame(traj_f, ksp_w, vol_cfl_base, combine=args.combine, matrix=(NX, NY, NZ))

        # Pull magnitude image back into NumPy for stacking
        img3 = cfl_to_3d_numpy(vol_cfl_base, NX, NY, NZ)
        frames.append(img3)
        print(f"[info] Frame {fi}/{len(starts)} done -> {vol_cfl_base}")

    if not frames:
        print("ERROR: no frames reconstructed.", file=sys.stderr)
        sys.exit(1)

    vol4d = np.stack(frames, axis=3)  # (NX,NY,NZ,Nt)
    Nt = vol4d.shape[3]
    print(f"[info] Assembled 4D volume: {vol4d.shape}")

    if nib is not None:
        # QC volume (one frame)
        qc_idx = max(1, min(args.qc_frame, Nt)) - 1
        qc_img = vol4d[..., qc_idx]
        qc_out = out_base.with_name(out_base.name + f"_QC_vol{qc_idx+1:05d}.nii.gz")
        qc_nii = nib.Nifti1Image(np.abs(qc_img).astype(np.float32), np.eye(4))
        nib.save(qc_nii, str(qc_out))
        print(f"[info] QC NIfTI written:", qc_out)

        # Full 4D NIfTI
        out_nii = out_base.with_suffix(".nii.gz")
        nii = nib.Nifti1Image(np.abs(vol4d).astype(np.float32), np.eye(4))
        nib.save(nii, str(out_nii))
        print(f"[info] 4D NIfTI written:", out_nii)
    else:
        print("[warn] nibabel not installed; skipping NIfTI export.", file=sys.stderr)

    print("[info] All requested frames complete.")


if __name__ == "__main__":
    main()
