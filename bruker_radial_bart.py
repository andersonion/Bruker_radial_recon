#!/usr/bin/env python3
"""
bruker_radial_bart.py

Bruker 3D radial -> BART NUFFT recon driver, with MATLAB-faithful parsing for:
  - fid (block-padded rays, trim-to-ray, reshape to complex)
  - traj (float64 ieee-le, reshape(3, RO, NPro), Fortran order)

Design goals (per your constraints):
  - NEVER require user to specify N_coils, N_read/RO, spokes, etc.
  - dtype and endianness inferred from acqp (ACQ_word_size, BYTORDA)
  - trajectory scaling is inferred from matrix size (NX), not hardcoded.

Supports:
  - --traj-source trajfile   (MATLAB-faithful)
  - --traj-source gradoutput (simple direction-based traj, optional)

Outputs:
  - per-frame BART CFLs
  - optional QA NIfTI
  - final 4D stack via bart join
  - optional final NIfTI
"""

from __future__ import annotations

import argparse
import sys
import subprocess
import textwrap
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import nibabel as nib


# ---------------- Bruker helpers ---------------- #

def read_bruker_param(path: Path, key: str, default=None):
    """
    Minimal Bruker JCAMP-ish parser for lines like:
      ##$KEY=VALUE
      ##$KEY=( ... )
        <multiline array>
    Returns int/float/str or list thereof.
    """
    if not path.exists():
        return default

    token = f"##${key}="
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except Exception:
        return default

    for i, line in enumerate(lines):
        if not line.startswith(token):
            continue

        rhs = line.split("=", 1)[1].strip()

        # multiline array
        if rhs.startswith("("):
            vals: list[str] = []
            j = i + 1
            while j < len(lines):
                l2 = lines[j].strip()
                if l2.startswith("##"):
                    break
                if l2 and not l2.startswith("$$"):
                    vals.extend(l2.split())
                j += 1

            out = []
            for v in vals:
                try:
                    out.append(float(v) if ("." in v or "e" in v.lower()) else int(v))
                except ValueError:
                    out.append(v)
            return out[0] if len(out) == 1 else out

        rhs = rhs.strip("()")
        toks = rhs.split()
        out = []
        for v in toks:
            try:
                out.append(float(v) if ("." in v or "e" in v.lower()) else int(v))
            except ValueError:
                out.append(v)
        return out[0] if len(out) == 1 else out

    return default


def infer_matrix(method: Path) -> Tuple[int, int, int]:
    mat = read_bruker_param(method, "PVM_Matrix", None)
    if mat is None or isinstance(mat, (int, float)) or len(mat) != 3:
        raise ValueError(f"Could not infer PVM_Matrix (got {mat})")
    return tuple(map(int, mat))


def infer_acq_size_words(acqp: Path) -> int:
    """
    ACQ_size is commonly given as a list; first element is readout length.
    IMPORTANT: In many Bruker datasets, this is *word count* along readout,
               which may be 2x the complex-sample count (re+im interleaved).
    We'll resolve complex RO later.
    """
    v = read_bruker_param(acqp, "ACQ_size", None)
    if v is None:
        raise ValueError("Could not infer ACQ_size")
    return int(v if isinstance(v, (int, float)) else v[0])


def infer_coils(method: Path) -> int:
    v = read_bruker_param(method, "PVM_EncNReceivers", 1)
    return int(v if not isinstance(v, (list, tuple)) else v[0])


def infer_npro(method: Path, acqp: Path) -> Optional[int]:
    v = read_bruker_param(method, "NPro", None)
    if v is None:
        v = read_bruker_param(acqp, "NPro", None)
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return int(v[0])
    try:
        return int(v)
    except Exception:
        return None


def infer_bytorda_endian(acqp: Path) -> str:
    """
    Bruker: ##$BYTORDA=big or little
    Return numpy endian prefix: '>' or '<'
    """
    v = read_bruker_param(acqp, "BYTORDA", None)
    if v is None:
        # Most modern Bruker is little-endian; but be explicit if missing.
        print("[warn] acqp missing BYTORDA; assuming little-endian.", file=sys.stderr)
        return "<"
    if isinstance(v, (list, tuple)):
        v = v[0]
    s = str(v).strip().lower()
    if "big" in s:
        return ">"
    if "little" in s:
        return "<"
    print(f"[warn] Unrecognized BYTORDA={v}; assuming little-endian.", file=sys.stderr)
    return "<"


def infer_acq_word_size_dtype(acqp: Path) -> np.dtype:
    """
    Bruker: ##$ACQ_word_size=_32_BIT or _16_BIT etc.
    We'll map to signed int types (as in your MATLAB int32 read).
    """
    v = read_bruker_param(acqp, "ACQ_word_size", None)
    if v is None:
        print("[warn] acqp missing ACQ_word_size; assuming int32.", file=sys.stderr)
        return np.dtype(np.int32)
    if isinstance(v, (list, tuple)):
        v = v[0]
    s = str(v).strip().upper()
    if "32" in s:
        return np.dtype(np.int32)
    if "16" in s:
        return np.dtype(np.int16)
    if "8" in s:
        return np.dtype(np.int8)
    print(f"[warn] Unrecognized ACQ_word_size={v}; assuming int32.", file=sys.stderr)
    return np.dtype(np.int32)


# ---------------- BART CFL helpers ---------------- #

def writecfl(name: str, arr: np.ndarray):
    base = Path(name)
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")

    arr_f = np.asfortranarray(arr.astype(np.complex64))
    dims = list(arr_f.shape) + [1] * (16 - arr_f.ndim)

    with hdr.open("w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in dims) + "\n")

    stacked = np.empty(arr_f.size * 2, dtype=np.float32)
    stacked[0::2] = arr_f.real.ravel(order="F")
    stacked[1::2] = arr_f.imag.ravel(order="F")
    stacked.tofile(cfl)


def readcfl(name: str) -> np.ndarray:
    base = Path(name)
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")
    if not hdr.exists() or not cfl.exists():
        raise FileNotFoundError(f"CFL/HDR not found: {base}")

    lines = hdr.read_text(errors="ignore").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Malformed hdr: {hdr}")

    dims16 = [int(x) for x in lines[1].split()]
    last_non1 = 0
    for i, d in enumerate(dims16):
        if d > 1:
            last_non1 = i
    ndim = max(1, last_non1 + 1)
    dims = dims16[:ndim]

    data = np.fromfile(cfl, dtype=np.float32)
    if data.size % 2 != 0:
        raise ValueError(f"CFL length {data.size} not even: {cfl}")

    cplx = data[0::2] + 1j * data[1::2]
    expected = int(np.prod(dims))
    if cplx.size != expected:
        raise ValueError(
            f"CFL size mismatch: have {cplx.size} complex, expected {expected} from dims {dims} for {base}"
        )

    return cplx.reshape(dims, order="F")


def bart_supports_gpu(bart_bin: str = "bart") -> bool:
    proc = subprocess.run([bart_bin, "nufft", "-i", "-g"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode == 0:
        return True
    s = (proc.stderr or "").lower()
    if "compiled without gpu support" in s or "unknown option" in s:
        return False
    return False


def bart_image_dims(bart_bin: str, base: Path) -> list[int] | None:
    dims: list[int] = []
    for d in range(16):
        proc = subprocess.run(
            [bart_bin, "show", "-d", str(d), str(base)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            print(f"[warn] bart show -d {d} failed: {proc.stderr.strip()}", file=sys.stderr)
            return None
        dims.append(int(proc.stdout.strip()))
    return dims


# ---------------- Diagnostics ---------------- #

def traj_radial_profile_debug(traj: np.ndarray, label: str = "traj") -> None:
    k_mag = np.sqrt(
        np.real(traj[0]) ** 2 +
        np.real(traj[1]) ** 2 +
        np.real(traj[2]) ** 2
    ).astype(np.float64, copy=False)
    k_ro_med = np.median(k_mag, axis=1)
    imin = int(np.argmin(k_ro_med))
    mid = k_ro_med.shape[0] // 2
    print(
        f"[debug] {label} |k| median at RO[0],RO[mid],RO[-1]: "
        f"{k_ro_med[0]:.6g}, {k_ro_med[mid]:.6g}, {k_ro_med[-1]:.6g}"
    )
    print(f"[debug] {label} |k| median min at RO={imin} (|k|≈{k_ro_med[imin]:.6g})")


def recenter_traj_by_kmin_centered_only(traj: np.ndarray, label: str) -> np.ndarray:
    """
    Robust RO recentering for centered readouts:
      roll so argmin(median_spokes |k|) lands at RO[mid].
    If min is near endpoints, treat as center-out and do nothing.
    """
    k_mag = np.sqrt(
        np.real(traj[0]) ** 2 +
        np.real(traj[1]) ** 2 +
        np.real(traj[2]) ** 2
    ).astype(np.float64, copy=False)
    k_ro_med = np.median(k_mag, axis=1)
    imin = int(np.argmin(k_ro_med))
    mid = int(k_ro_med.shape[0] // 2)

    # Heuristic for centered vs center-out
    if not (0.1 * k_ro_med.shape[0] < imin < 0.9 * k_ro_med.shape[0]):
        print(f"[info] {label}: center-out readout detected (|k| min at RO={imin}); no RO recentering applied")
        return traj

    shift = mid - imin
    if shift != 0:
        traj2 = np.roll(traj, shift=shift, axis=1)
        print(f"[info] {label}: centered readout; shifted RO axis by {shift} to place |k| med-min at RO[mid]={mid} (imin={imin})")
        return traj2
    return traj


# ---------------- MATLAB-faithful trajfile parsing ---------------- #

def find_traj_candidates(series_path: Path) -> List[Path]:
    cands: List[Path] = []

    p0 = series_path / "traj"
    if p0.exists() and p0.is_file():
        cands.append(p0)

    pdata = series_path / "pdata"
    if pdata.exists():
        for p in pdata.rglob("traj"):
            if p.exists() and p.is_file():
                cands.append(p)

    seen = set()
    out: List[Path] = []
    for p in cands:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def load_trajfile_matlabfaithful(traj_path: Path, *, ro: int, npro: int, endian: str, traj_scaling: float) -> np.ndarray:
    """
    MATLAB:
      traj = fread(f,Inf,"float64","ieee-le");
      traj = reshape(traj,3,n_read,[]);
      traj = traj * traj_scaling;

    We emulate MATLAB reshape semantics (column-major): use order='F'.
    """
    if not traj_path.exists():
        raise FileNotFoundError(f"traj not found: {traj_path}")

    dt = np.dtype(np.float64).newbyteorder(endian)  # '<f8' or '>f8'
    v = np.fromfile(traj_path, dtype=dt)

    expected = 3 * ro * npro
    if v.size != expected:
        raise ValueError(
            f"traj element count mismatch: got {v.size}, expected {expected} (=3*RO*NPro) "
            f"with RO={ro}, NPro={npro}"
        )

    arr = v.reshape((3, ro, npro), order="F").astype(np.float32, copy=False)
    arr = (arr * float(traj_scaling)).astype(np.float32, copy=False)

    traj = np.asfortranarray(arr).astype(np.complex64)  # imag=0
    return traj


# ---------------- MATLAB-faithful FID parsing ---------------- #

def load_fid_matlabfaithful(
    fid_path: Path,
    *,
    coils: int,
    ro_complex: int,
    word_dtype: np.dtype,
    endian: str,
    block_bytes: int = 1024
) -> np.ndarray:
    """
    MATLAB logic (generalized, without n_vols):
      x = fread(f,Inf,"int32","ieee-le");
      n_ray = n_coils * n_read * 2;           # words per ray (re+im)
      n_blocks = ceil(n_ray*4 / 1024);
      n_block_samples = n_blocks*1024 / 4;
      x = reshape(x,n_block_samples,[]);
      x = x(1:n_ray,:);
      x = x(:);
      x = reshape(x,2,n_read,n_coils,[]);
      x = squeeze(complex(x(1,:,:,:),x(2,:,:,:)));

    Output:
      ksp (RO, spokes, coils) complex64
    """
    if not fid_path.exists():
        raise FileNotFoundError(f"FID not found: {fid_path}")

    wd = np.dtype(word_dtype).newbyteorder(endian)

    x = np.fromfile(fid_path, dtype=wd)
    if x.size == 0:
        raise ValueError(f"Empty FID: {fid_path}")

    word_bytes = wd.itemsize

    # words per ray = coils * ro_complex * 2(re/im)
    n_ray_words = int(coils) * int(ro_complex) * 2
    n_ray_bytes = n_ray_words * word_bytes

    n_blocks = int(np.ceil(n_ray_bytes / float(block_bytes)))
    n_block_words = int(n_blocks * block_bytes // word_bytes)

    if x.size % n_block_words != 0:
        # Bruker should be block-aligned, but don't die—trim trailing partial if needed.
        n_rays = x.size // n_block_words
        x = x[: n_rays * n_block_words]

    n_rays = x.size // n_block_words
    if n_rays <= 0:
        raise ValueError("Could not infer any rays from FID (block factoring failed).")

    # MATLAB reshape uses column-major; emulate with order='F'
    xb = x.reshape((n_block_words, n_rays), order="F")

    # trim per ray to n_ray_words
    xb = xb[:n_ray_words, :]

    # flatten column-major (x = x(:))
    xf = xb.reshape((-1,), order="F")

    # reshape to (2, ro_complex, coils, rays) column-major
    xr = xf.reshape((2, ro_complex, coils, n_rays), order="F")

    # complex combine
    xc = xr[0, :, :, :] + 1j * xr[1, :, :, :]

    # Now: (RO, coils, rays) -> (RO, rays, coils)
    ksp = np.transpose(xc, (0, 2, 1)).astype(np.complex64, copy=False)
    return ksp


def expand_traj_spokes(traj: np.ndarray, target_spokes: int, order: str) -> np.ndarray:
    """
    traj: (3, ro, nsp)
    Expand along spoke axis to (3, ro, target_spokes) if divisible.
    """
    nsp = traj.shape[2]
    if target_spokes == nsp:
        return traj
    if target_spokes % nsp != 0:
        raise ValueError(f"Cannot expand traj spokes {nsp} to {target_spokes} (not divisible)")
    reps = target_spokes // nsp
    print(f"[info] Expanding traj spokes {nsp} -> {target_spokes} with reps={reps} using order='{order}'")
    if order == "tile":
        return np.tile(traj, (1, 1, reps))
    if order == "repeat":
        return np.repeat(traj, reps, axis=2)
    raise ValueError(f"Unknown spoke-order: {order}")


def bart_cfl_to_nifti(base: Path, out_nii_gz: Path):
    arr = np.abs(readcfl(str(base)))
    nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), str(out_nii_gz))


def write_qa_nifti(qa_frames: list[Path], qa_base: Path):
    vols = []
    shape0 = None
    for p in qa_frames:
        mag = np.abs(readcfl(str(p)))
        if shape0 is None:
            shape0 = mag.shape
        elif mag.shape != shape0:
            raise ValueError(f"QA frame {p} has shape {mag.shape}, expected {shape0}")
        vols.append(mag)
    qa_stack = np.stack(vols, axis=-1)
    out = qa_base.with_suffix(".nii.gz")
    nib.save(nib.Nifti1Image(qa_stack.astype(np.float32), np.eye(4)), str(out))
    print(f"[info] Wrote QA NIfTI {out} with shape {qa_stack.shape}")


# ---------------- grad.output trajectory (optional fallback) ---------------- #

def _extract_grad_block(lines: list[str], name: str) -> np.ndarray:
    hdr_pat = re.compile(rf"^\s*\d+:{re.escape(name)}:\s+index\s*=\s*\d+,\s+size\s*=\s*(\d+)\s*$")

    start = None
    size = None
    for i, line in enumerate(lines):
        m = hdr_pat.match(line.strip())
        if m:
            start = i + 1
            size = int(m.group(1))
            break

    if start is None or size is None:
        raise ValueError(f"Could not find {name} in grad.output")

    vals: list[int] = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+:\w+:", line) or line.startswith("Ramp Shape"):
            break
        parts = line.split()
        if len(parts) < 2:
            continue
        vals.append(int(parts[1]))
        if len(vals) >= size:
            break

    if len(vals) != size:
        raise ValueError(f"{name}: expected {size} values, got {len(vals)}")
    return np.array(vals, dtype=np.float64)


def load_grad_output_dirs(grad_output_path: Path, normalize: bool = True) -> np.ndarray:
    lines = grad_output_path.read_text(errors="ignore").splitlines()
    r = _extract_grad_block(lines, "ProjR")
    p = _extract_grad_block(lines, "ProjP")
    s = _extract_grad_block(lines, "ProjS")

    scale = float(1 << 30)
    dirs = np.stack([r, p, s], axis=1) / scale

    norms = np.linalg.norm(dirs, axis=1)
    print(f"[info] Parsed {dirs.shape[0]} spoke directions from {grad_output_path}")
    print(f"[info] Direction norms BEFORE normalize: min={norms.min():.4f} median={np.median(norms):.4f} max={norms.max():.4f}")

    if normalize:
        bad = norms == 0
        if np.any(bad):
            raise ValueError(f"Found {bad.sum()} zero-norm direction(s) in {grad_output_path}")
        dirs = dirs / norms[:, None]
        norms2 = np.linalg.norm(dirs, axis=1)
        print(f"[info] Direction norms AFTER  normalize: min={norms2.min():.4f} median={np.median(norms2):.4f} max={norms2.max():.4f}")

    return dirs


def expand_spoke_dirs(dirs: np.ndarray, target_spokes: int, order: str) -> np.ndarray:
    n = dirs.shape[0]
    if target_spokes == n:
        return dirs
    if target_spokes % n != 0:
        raise ValueError(f"Cannot expand dirs length {n} to {target_spokes} (not divisible)")
    reps = target_spokes // n
    print(f"[info] Expanding {n} dirs to {target_spokes} spokes with reps={reps} using order='{order}'")
    if order == "tile":
        return np.tile(dirs, (reps, 1))
    if order == "repeat":
        return np.repeat(dirs, reps, axis=0)
    raise ValueError(f"Unknown spoke-order: {order}")


def build_traj_from_dirs(ro: int, dirs_xyz: np.ndarray, NX: int, centered: bool) -> np.ndarray:
    spokes = dirs_xyz.shape[0]
    kmax = 0.5 * float(NX)
    if centered:
        s = np.linspace(-kmax, kmax, ro, dtype=np.float64)
    else:
        s = np.linspace(0.0, kmax, ro, dtype=np.float64)

    traj = np.zeros((3, ro, spokes), dtype=np.complex64)
    dx, dy, dz = dirs_xyz[:, 0], dirs_xyz[:, 1], dirs_xyz[:, 2]
    for i in range(spokes):
        traj[0, :, i] = s * dx[i]
        traj[1, :, i] = s * dy[i]
        traj[2, :, i] = s * dz[i]

    print(f"[info] Traj built from grad.output with max |k| ≈ {np.abs(traj).max():.2f}")
    return traj


# ---------------- Core recon ---------------- #

def run_bart(
    series_path: Path,
    method: Path,
    acqp: Path,
    out_base: Path,
    NX: int, NY: int, NZ: int,
    traj_source: str,
    spoke_order: str,
    spokes_per_frame: int,
    frame_shift: int,
    qa_first: Optional[int],
    export_nifti: bool,
    use_gpu: bool,
    reverse_readout: bool,
):
    bart_bin = "bart"

    coils = infer_coils(method)
    endian = infer_bytorda_endian(acqp)
    word_dtype = infer_acq_word_size_dtype(acqp)
    acq_size_words = infer_acq_size_words(acqp)

    # ---- decide traj path (if needed) ----
    traj_path: Optional[Path] = None
    if traj_source in ("trajfile", "auto"):
        cands = find_traj_candidates(series_path)
        if cands:
            traj_path = cands[0]

    # ---- infer NPro ----
    npro = infer_npro(method, acqp)
    if npro is None and traj_source == "trajfile":
        raise RuntimeError("trajfile parsing requires NPro, but could not read ##$NPro from method/acqp.")

    # ---- infer RO_complex ----
    # If traj exists and ACQ_size looks like 2*RO_complex, use traj RO_complex.
    ro_complex: Optional[int] = None

    if traj_source in ("trajfile", "auto") and traj_path is not None and npro is not None:
        # We don't know RO yet; try the common case: ACQ_size_words is 2*RO_complex.
        if acq_size_words % 2 == 0:
            guess_ro = acq_size_words // 2
            # See if traj size matches 3*guess_ro*npro as float64
            try:
                n_elem = traj_path.stat().st_size // 8
                if n_elem == 3 * guess_ro * int(npro):
                    ro_complex = guess_ro
            except Exception:
                pass

    if ro_complex is None:
        # Default heuristic: if ACQ_size_words is even, assume it's word-count (re+im) -> complex RO = ACQ_size/2
        # If that ever breaks, you can flip this to ro_complex=acq_size_words.
        if acq_size_words % 2 == 0:
            ro_complex = acq_size_words // 2
        else:
            ro_complex = acq_size_words

    print(f"[info] Matrix inferred from PVM_Matrix: {NX}x{NY}x{NZ}")
    print(f"[info] Coils inferred from PVM_EncNReceivers: {coils}")
    print(f"[info] ACQ_size (words) from acqp: {acq_size_words} -> using complex RO={ro_complex}")
    print(f"[info] FID dtype from ACQ_word_size: {word_dtype.name}, endian from BYTORDA: {'LE' if endian=='<' else 'BE'}")

    # ---- load k-space MATLAB-faithful ----
    fid = series_path / "fid"
    ksp = load_fid_matlabfaithful(
        fid_path=fid,
        coils=coils,
        ro_complex=ro_complex,
        word_dtype=word_dtype,
        endian=endian,
    )
    spokes_all = ksp.shape[1]
    print(f"[info] Loaded k-space (MATLAB-faithful): RO={ksp.shape[0]}, spokes={spokes_all}, coils={ksp.shape[2]}")

    # ---- build/load trajectory ----
    traj_used = ""
    traj_full: np.ndarray

    if traj_source == "gradoutput" or (traj_source == "auto" and traj_path is None):
        grad_path = series_path / "grad.output"
        if not grad_path.exists():
            raise RuntimeError(f"grad.output not found in {series_path}")
        dirs = load_grad_output_dirs(grad_path, normalize=True)
        dirs_full = expand_spoke_dirs(dirs, spokes_all, spoke_order)
        traj_full = build_traj_from_dirs(ro=ro_complex, dirs_xyz=dirs_full, NX=NX, centered=True)
        traj_used = "gradoutput"
    else:
        if traj_path is None:
            raise RuntimeError(f"--traj-source {traj_source} requested, but no traj file found under {series_path} or pdata/*")

        assert npro is not None
        traj_scaling = float(NX)  # MATLAB used 96 for a 96^3 recon; generalize to NX
        traj_full = load_trajfile_matlabfaithful(
            traj_path=traj_path,
            ro=ro_complex,
            npro=int(npro),
            endian=endian,
            traj_scaling=traj_scaling,
        )
        traj_used = f"trajfile:f8_{'le' if endian=='<' else 'be'}_F:(NPro={npro},RO={ro_complex},3)"

        if reverse_readout:
            traj_full = traj_full[:, ::-1, :].copy()
            print("[info] trajfile: applied --reverse-readout")

        # Centered recentering (only if it looks centered)
        traj_radial_profile_debug(traj_full, label="trajfile raw")
        traj_full = recenter_traj_by_kmin_centered_only(traj_full, label="trajfile")
        traj_radial_profile_debug(traj_full, label="trajfile after_recenter_kmin")

        # expand spokes if k-space has repetitions / multiple vols concatenated
        traj_full = expand_traj_spokes(traj_full, target_spokes=spokes_all, order=spoke_order)
        traj_radial_profile_debug(traj_full, label="trajfile final")

    print(f"[info] Trajectory source used: {traj_used}")

    # ---- final sanity check ----
    if traj_full.shape[1] != ksp.shape[0]:
        raise ValueError(f"traj RO={traj_full.shape[1]} does not match ksp RO={ksp.shape[0]}")

    # ---- sliding window params ----
    if spokes_per_frame <= 0:
        spokes_per_frame = spokes_all
    if frame_shift <= 0:
        frame_shift = spokes_per_frame

    frame_starts = list(range(0, max(1, spokes_all - spokes_per_frame + 1), frame_shift))
    print(f"[info] Sliding window: spokes-per-frame={spokes_per_frame}, frame-shift={frame_shift}")
    print(f"[info] Will reconstruct {len(frame_starts)} frame(s).")

    # ---- gpu support ----
    have_gpu = False
    if use_gpu:
        have_gpu = bart_supports_gpu(bart_bin)
        if not have_gpu:
            print("[warn] BART has no GPU support; falling back to CPU.", file=sys.stderr)

    out_dir = out_base.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    qa_written = False
    first_dims_reported = False
    rss_mask = str(1 << 3)  # coil dim=3 => 8

    for i, start in enumerate(frame_starts):
        stop = min(spokes_all, start + spokes_per_frame)
        nsp = stop - start

        tag = f"vol{i:05d}"
        traj_base = out_dir / f"{out_base.name}_{tag}_traj"
        ksp_base = out_dir / f"{out_base.name}_{tag}_ksp"
        coil_base = out_dir / f"{out_base.name}_{tag}_coil"
        sos_base = out_dir / f"{out_base.name}_{tag}"

        frame_paths.append(sos_base)

        if sos_base.with_suffix(".cfl").exists():
            print(f"[info] Frame {i} already reconstructed -> {sos_base}, skipping.")
            continue

        print(f"[info] Frame {i} spokes [{start}:{stop}] (n={nsp})")

        ksp_frame = ksp[:, start:stop, :]         # (ro, spokes, coils)
        traj_frame = traj_full[:, :, start:stop]  # (3, ro, spokes)

        # BART NUFFT expects k-space dims[0] == 1 => (1, RO, spokes, coils)
        ksp_bart = ksp_frame[np.newaxis, :, :, :]

        writecfl(str(traj_base), traj_frame)
        writecfl(str(ksp_base), ksp_bart)

        cmd = [bart_bin, "nufft", "-i", "-d", f"{NX}:{NY}:{NZ}"]
        if use_gpu and have_gpu:
            cmd.insert(2, "-g")
        cmd += [str(traj_base), str(ksp_base), str(coil_base)]
        print("[bart]", " ".join(cmd))
        subprocess.run(cmd, check=True)

        cmd2 = [bart_bin, "rss", rss_mask, str(coil_base), str(sos_base)]
        print("[bart]", " ".join(cmd2))
        subprocess.run(cmd2, check=True)

        if not first_dims_reported:
            dims_traj = bart_image_dims(bart_bin, traj_base)
            dims_ksp = bart_image_dims(bart_bin, ksp_base)
            dims_coil = bart_image_dims(bart_bin, coil_base)
            dims_sos = bart_image_dims(bart_bin, sos_base)
            if dims_traj is not None:
                print(f"[debug] BART dims traj vol0: {dims_traj}")
            if dims_ksp is not None:
                print(f"[debug] BART dims ksp  vol0: {dims_ksp}")
            if dims_coil is not None:
                print(f"[debug] BART dims coil vol0: {dims_coil}")
            if dims_sos is not None:
                print(f"[debug] BART dims SoS  vol0: {dims_sos}")
            first_dims_reported = True

        if qa_first is not None and not qa_written and len(frame_paths) >= qa_first:
            qa_base = out_dir / f"{out_base.name}_QA_first{qa_first}"
            write_qa_nifti(frame_paths[:qa_first], qa_base)
            qa_written = True

    sos_existing = [p for p in frame_paths if p.with_suffix(".cfl").exists()]
    if not sos_existing:
        print("[warn] No SoS frames exist; skipping 4D stack.", file=sys.stderr)
        return

    stack_base = out_dir / out_base.name
    join_cmd = ["bart", "join", "3"] + [str(p) for p in sos_existing] + [str(stack_base)]
    print("[bart]", " ".join(join_cmd))
    subprocess.run(join_cmd, check=True)

    if export_nifti:
        out_nii = stack_base.with_suffix(".nii.gz")
        bart_cfl_to_nifti(stack_base, out_nii)
        print(f"[info] Wrote final 4D NIfTI {out_nii}")

    print(f"[info] Done. 4D result base: {stack_base}")


# ---------------- CLI ---------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Bruker 3D radial → BART NUFFT recon driver (MATLAB-faithful fid+traj parsing).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:

              Use MATLAB-faithful trajectory file (Bruker writes <series>/traj):
                python bruker_radial_bart.py --series /path/to/21 --traj-source trajfile --spoke-order repeat --export-nifti --out /tmp/out

              Use grad.output (fallback):
                python bruker_radial_bart.py --series /path/to/21 --traj-source gradoutput --export-nifti --out /tmp/out
            """
        ),
    )

    ap.add_argument("--series", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--matrix", nargs=3, type=int, metavar=("NX", "NY", "NZ"), help="Override PVM_Matrix if needed")

    ap.add_argument("--spokes-per-frame", type=int, default=0)
    ap.add_argument("--frame-shift", type=int, default=0)

    ap.add_argument("--spoke-order", choices=["tile", "repeat"], default="tile")
    ap.add_argument("--traj-source", choices=["auto", "trajfile", "gradoutput"], default="auto")

    ap.add_argument("--qa-first", type=int, default=0)
    ap.add_argument("--export-nifti", action="store_true")

    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--reverse-readout", action="store_true")

    args = ap.parse_args()

    series_path = Path(args.series).resolve()
    out_base = Path(args.out).resolve()
    out_base.parent.mkdir(parents=True, exist_ok=True)

    method = series_path / "method"
    acqp = series_path / "acqp"
    fid = series_path / "fid"
    if not (method.exists() and acqp.exists() and fid.exists()):
        ap.error(f"Could not find method/acqp/fid under {series_path}")

    if args.matrix is not None:
        NX, NY, NZ = map(int, args.matrix)
        print(f"[info] Matrix overridden from CLI: {NX}x{NY}x{NZ}")
    else:
        NX, NY, NZ = infer_matrix(method)
        # printed later in run_bart for consistent ordering

    spf = args.spokes_per_frame
    shift = args.frame_shift
    qa_first = args.qa_first if args.qa_first > 0 else None

    run_bart(
        series_path=series_path,
        method=method,
        acqp=acqp,
        out_base=out_base,
        NX=NX, NY=NY, NZ=NZ,
        traj_source=args.traj_source,
        spoke_order=args.spoke_order,
        spokes_per_frame=spf,
        frame_shift=shift,
        qa_first=qa_first,
        export_nifti=args.export_nifti,
        use_gpu=args.gpu,
        reverse_readout=args.reverse_readout,
    )


if __name__ == "__main__":
    main()
