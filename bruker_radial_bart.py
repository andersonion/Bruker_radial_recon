#!/usr/bin/env python3
"""
bruker_radial_bart.py

Bruker 3D radial reconstruction using BART, with:
- MATLAB-faithful fid parsing (block padding removal)
- Trajectory file parsing (float64, LE/BE)
- Optional gradient delay correction via opposed-spoke RO shift fitting
  and per-spoke fractional sample shifting (Fourier shift along RO).

No hardcoded n_read/n_coils/n_vols. Everything inferred from Bruker params and file sizes.

Typical usage:
  python bruker_radial_bart.py --series /path/to/bruker/21 --traj-source trajfile --out outbase --qa-first 3 --export-nifti

Gradient delay correction:
  python bruker_radial_bart.py --series ... --traj-source trajfile \
      --gdelay-fit xyz --gdelay-max-shift 6 --gdelay-dot-thresh 0.995 \
      --gdelay-apply --out outbase

Requirements:
  - numpy
  - nibabel (only if --export-nifti)
  - BART in PATH (bart), or set BART_BIN
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------
# Small utilities
# ----------------------------

def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


def die(msg: str, code: int = 2) -> None:
    eprint(f"[error] {msg}")
    raise SystemExit(code)


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run(cmd: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), check=check)


# ----------------------------
# Bruker parameter parsing
# ----------------------------

_BRUKER_KV_RE = re.compile(r"^\s*##\$(?P<key>[^=]+)=(?P<val>.*)\s*$")


def read_bruker_params(path: Path) -> Dict[str, str]:
    """
    Read a Bruker parameter file (method/acqp/reco) into a dict of raw strings.

    Rules:
    - Lines begin with ##$KEY=VALUE
    - Multi-line values continue until the next ##$KEY=... line
    - Stop continuation if we hit a '$$' line (Bruker metadata / @vis etc)
    """
    if not path.exists():
        return {}

    lines = path.read_text(errors="ignore").splitlines()

    out: Dict[str, str] = {}
    i = 0
    while i < len(lines):
        m = _BRUKER_KV_RE.match(lines[i])
        if not m:
            i += 1
            continue

        key = m.group("key").strip()
        val = m.group("val").strip()

        j = i + 1
        chunks = [val]

        while j < len(lines):
            nxt = lines[j].strip()

            # new parameter begins
            if nxt.startswith("##$"):
                break

            # Bruker "comment/metadata" region begins (don't append these)
            if nxt.startswith("$$"):
                break

            # keep continuation lines
            if nxt != "":
                chunks.append(nxt)

            j += 1

        out[key] = " ".join(chunks).strip()
        i = j

    return out

def bruker_array_ints(raw: str) -> List[int]:
    """
    Parse Bruker array-like values, e.g.:

      '( 3 ) 126 31324 1'
      or with newlines already flattened by read_bruker_params.

    Returns the actual array values (length N), NOT the leading N.
    If format doesn't match, falls back to parsing all ints.
    """
    s = raw.strip()
    ints = parse_ints(s)
    if not ints:
        return []

    # Detect "( N ) ..." pattern by checking if string starts with '('
    if s.startswith("(") and len(ints) >= 1:
        n = ints[0]
        vals = ints[1:1 + n]
        # If file is malformed and we don't have enough, just return what we have
        return vals

    return ints


def get_array_ints(params: Dict[str, str], key: str) -> List[int]:
    if key not in params:
        return []
    return bruker_array_ints(params[key])


def parse_ints(s: str) -> List[int]:
    return [int(x) for x in re.findall(r"[-+]?\d+", s)]


def parse_floats(s: str) -> List[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)]


def get_first_int(params: Dict[str, str], key: str) -> Optional[int]:
    if key not in params:
        return None
    vals = parse_ints(params[key])
    return vals[0] if vals else None


def get_first_float(params: Dict[str, str], key: str) -> Optional[float]:
    if key not in params:
        return None
    vals = parse_floats(params[key])
    return vals[0] if vals else None

def infer_matrix_xyz(method: Dict[str, str]) -> Optional[Tuple[int, int, int]]:
    vals = get_array_ints(method, "PVM_Matrix")
    if len(vals) >= 3:
        return (vals[0], vals[1], vals[2])
    return None

def infer_ncoils(method: Dict[str, str], acqp: Dict[str, str]) -> Optional[int]:
    # Bruker varies: method has PVM_EncNReceivers, acqp has ACQ_ReceiverSelect/ACQ_nReceivers, etc.
    for k in ("PVM_EncNReceivers", "PVM_EncNReceivers1", "PVM_EncNReceivers2"):
        v = get_first_int(method, k)
        if v is not None and v > 0:
            return v

    for k in ("ACQ_nReceivers", "ACQ_NReceivers"):
        v = get_first_int(acqp, k)
        if v is not None and v > 0:
            return v

    # ReceiverSelect often looks like: ( 4 ) yes yes yes yes
    if "ACQ_ReceiverSelect" in acqp:
        ints = parse_ints(acqp["ACQ_ReceiverSelect"])
        if ints:
            n = ints[0]
            if n > 0:
                return n

    return None


def infer_acq_size_words(acqp: Dict[str, str]) -> Optional[int]:
    vals = get_array_ints(acqp, "ACQ_size")
    if len(vals) >= 1:
        return int(vals[0])  # first value is the readout length in "words"
    return None

def infer_npro(method: Dict[str, str], acqp: Dict[str, str]) -> Optional[int]:
    # You’ve been using ##$NPro from method/acqp in your earlier code.
    for k in ("NPro", "PVM_NPro", "PVM_RadialNPro", "RadialNPro"):
        v = get_first_int(method, k)
        if v is not None and v > 0:
            return v
        v = get_first_int(acqp, k)
        if v is not None and v > 0:
            return v
    return None


def infer_dwell_time(acqp: Dict[str, str]) -> Optional[float]:
    # Bruker naming varies; you might have ACQ_dw or DWELLTIME
    for k in ("ACQ_dw", "DWELLTIME", "ACQ_DwellTime"):
        v = get_first_float(acqp, k)
        if v is not None and v > 0:
            return v
    return None


def fid_dtype_and_endian(acqp: Dict[str, str]) -> Tuple[np.dtype, str]:
    # ACQ_word_size is like "_32_BIT"
    ws = acqp.get("ACQ_word_size", "").strip()
    by = acqp.get("BYTORDA", "").strip().upper()

    if "32" in ws:
        base = np.int32
    elif "16" in ws:
        base = np.int16
    else:
        # Default to 32-bit; better than guessing float
        base = np.int32

    endian = "LE"
    if "BE" in by:
        endian = "BE"
    elif "LE" in by:
        endian = "LE"

    if endian == "LE":
        dt = np.dtype(base).newbyteorder("<")
    else:
        dt = np.dtype(base).newbyteorder(">")

    return dt, endian


# ----------------------------
# MATLAB-faithful FID parsing
# ----------------------------

def read_fid_matlab_faithful(fid_path: Path, *, ncoils: int, ro_complex: int, endian_dt: np.dtype) -> np.ndarray:
    """
    MATLAB code logic, generalized:
      - read all int32 (or int16) into x
      - n_ray = n_coils * n_read * 2      (2 = re/im)
      - n_block_bytes = 1024
      - n_blocks = ceil(n_ray*bytes_per_word / n_block_bytes)
      - n_block_words = n_blocks * (n_block_bytes/bytes_per_word)
      - reshape words into (n_block_words, -1)
      - trim to first n_ray
      - flatten, then reshape to (2, n_read, n_coils, n_pro, n_vols)
      - make complex and return as (ro, coils, spokes, vols)
    """
    if not fid_path.exists():
        die(f"FID not found: {fid_path}")

    raw = np.fromfile(str(fid_path), dtype=endian_dt)
    if raw.size == 0:
        die(f"FID file is empty: {fid_path}")

    bytes_per_word = endian_dt.itemsize
    n_read = int(ro_complex)
    n_ray = int(ncoils) * int(n_read) * 2  # 2 for re/im
    n_block_bytes = 1024
    n_blocks = int(math.ceil((n_ray * bytes_per_word) / n_block_bytes))
    n_block_words = int(n_blocks * (n_block_bytes // bytes_per_word))

    if raw.size % n_block_words != 0:
        # It's still often reshape-able; but warn.
        eprint(f"[warn] FID size {raw.size} words not divisible by block_words={n_block_words}. "
               f"Proceeding by truncation to full blocks.")
        n_full = (raw.size // n_block_words) * n_block_words
        raw = raw[:n_full]

    blk = raw.reshape(n_block_words, -1, order="F")  # MATLAB reshape default is column-major; keep that behavior
    blk = blk[:n_ray, :]
    flat = blk.reshape(-1, order="F")

    # Now interpret as (2, n_read, n_coils, n_spokes_total)
    if flat.size % (2 * n_read * ncoils) != 0:
        die(f"After block-trim, data size {flat.size} not divisible by 2*n_read*ncoils={2*n_read*ncoils}")

    n_spokes_total = flat.size // (2 * n_read * ncoils)

    x = flat.reshape(2, n_read, ncoils, n_spokes_total, order="F")
    xc = x[0, ...].astype(np.float32) + 1j * x[1, ...].astype(np.float32)  # complex64-ish

    # Return shape: (RO, coils, spokes_total)
    ksp = np.asarray(xc, dtype=np.complex64)
    # current shape: (n_read, n_coils, n_spokes_total)
    return ksp


# ----------------------------
# Trajectory parsing
# ----------------------------

def read_trajfile(
    traj_path: Path,
    *,
    ro: int,
    npro: int,
    endian: str,
    matrix_xyz: tuple[int, int, int],
) -> np.ndarray:
    """
    Read Bruker traj file as float64 and reshape to (3, RO, NPro).
    MATLAB does:
        traj = fread(f, Inf, "float64", "ieee-le");
        traj = reshape(traj, 3, n_read, []);
        traj = traj * traj_scaling;

    Here, traj_scaling is taken from PVM_Matrix = (NX,NY,NZ) (per-axis scaling).
    """
    if not traj_path.exists():
        die(f"traj file not found: {traj_path}")

    dt = np.dtype(np.float64).newbyteorder("<" if endian == "LE" else ">")
    raw = np.fromfile(str(traj_path), dtype=dt)
    if raw.size == 0:
        die(f"traj file empty: {traj_path}")

    expected = 3 * int(ro) * int(npro)
    if raw.size < expected:
        die(
            f"traj too small: got {raw.size} float64, expected at least {expected} (=3*RO*NPro)"
        )
    if raw.size != expected:
        eprint(f"[warn] traj size {raw.size} != expected {expected}; trimming to expected.")
        raw = raw[:expected]

    # MATLAB-faithful reshape (column-major)
    traj = raw.reshape(3, int(ro), int(npro), order="F")

    # MATLAB-faithful scaling to BART pixel units
    NX, NY, NZ = matrix_xyz
    traj = traj.astype(np.float64, copy=False)
    #traj[0, :, :] /= float(NX)
    #traj[1, :, :] /= float(NY)
   # traj[2, :, :] /= float(NZ)

    traj = np.asarray(traj, dtype=np.complex64)
    return traj



# ----------------------------
# BART CFL I/O
# ----------------------------

def writecfl(base: Path, arr: np.ndarray) -> None:
    """
    Write BART .cfl/.hdr.
    BART expects complex64 interleaved float32 in .cfl, dims in .hdr.
    """
    base = Path(base)
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")

    a = np.asarray(arr)

    if not np.iscomplexobj(a):
        raise ValueError("writecfl expects complex array (complex64)")

    a = np.asarray(a, dtype=np.complex64, order="F")

    # BART uses up to 16 dims in header; pad with 1s
    dims = list(a.shape)
    if len(dims) > 16:
        die(f"Too many dims for CFL: shape={a.shape}")
    dims16 = dims + [1] * (16 - len(dims))

    hdr.write_text("# Dimensions\n" + " ".join(map(str, dims16)) + "\n")

    inter = np.empty(a.size * 2, dtype=np.float32)
    inter[0::2] = a.real.ravel(order="F")
    inter[1::2] = a.imag.ravel(order="F")
    inter.tofile(cfl)

# ----------------------------
# Debug helpers
# ----------------------------

def traj_debug_stats(traj: np.ndarray, label: str) -> None:
    """
    traj: (3, RO, spokes)
    prints |k| median at RO endpoints + argmin index.
    """
    # Use float64 for stable median
    kx = np.real(traj[0]).astype(np.float64, copy=False)
    ky = np.real(traj[1]).astype(np.float64, copy=False)
    kz = np.real(traj[2]).astype(np.float64, copy=False)
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)  # (RO, spokes)

    k_med = np.median(kmag, axis=1)  # (RO,)
    mid = k_med.size // 2
    imin = int(np.argmin(k_med))

    def p(i: int) -> float:
        return float(k_med[i])

    print(f"[debug] {label} |k| median at RO[0],RO[mid],RO[-1]: {p(0):.6g}, {p(mid):.6g}, {p(-1):.6g}")
    print(f"[debug] {label} |k| median min at RO={imin} (|k|≈{float(k_med[imin]):.6g})")


def infer_spoke_unit_vectors(traj: np.ndarray) -> np.ndarray:
    """
    Infer unit direction per spoke using the last sample (largest radius).
    traj: (3, RO, spokes)
    returns u: (spokes, 3)
    """
    v = np.real(traj[:, -1, :]).T.astype(np.float64, copy=False)  # (spokes, 3)
    n = np.linalg.norm(v, axis=1)
    n[n == 0] = 1.0
    u = v / n[:, None]
    return u


# ----------------------------
# Gradient delay correction via opposed spoke shifts
# ----------------------------

@dataclass
class GDelayFitResult:
    mode: str  # "iso" or "xyz"
    a: np.ndarray  # (1,) for iso, (3,) for xyz
    used_pairs: int
    rms_resid: float


def _best_shift_corr(a: np.ndarray, b: np.ndarray, max_shift: int) -> Tuple[float, float]:
    """
    Find integer shift s (in samples) that maximizes |corr| between:
      a[ro] and shifted(b)[ro]

    Returns:
      best_s, best_score

    a and b are 1D complex arrays of same length.
    """
    ro = a.size
    best_s = 0
    best = -1.0

    # Normalize to reduce magnitude bias
    aa = a - np.mean(a)
    bb = b - np.mean(b)

    na = np.linalg.norm(aa)
    nb = np.linalg.norm(bb)
    if na == 0 or nb == 0:
        return 0.0, 0.0

    aa = aa / na
    bb = bb / nb

    # brute force since RO is small (e.g., 63)
    for s in range(-max_shift, max_shift + 1):
        if s < 0:
            x = aa[:s]
            y = bb[-s:]
        elif s > 0:
            x = aa[s:]
            y = bb[:-s]
        else:
            x = aa
            y = bb

        if x.size < max(8, ro // 4):
            continue

        c = np.vdot(x, y)  # complex correlation
        score = float(np.abs(c))
        if score > best:
            best = score
            best_s = s

    return float(best_s), float(best)


def fit_gradient_delay(
    traj: np.ndarray,
    ksp: np.ndarray,
    *,
    dot_thresh: float,
    max_pairs: int,
    max_shift: int,
    mode: str,
    rng: np.random.Generator,
) -> GDelayFitResult:
    """
    Build opposed pairs using traj directions, estimate per-pair RO shift via correlation,
    then fit:
      s_i ≈ ax*uix + ay*uiy + az*uiz     (mode="xyz")
    or
      s_i ≈ a0                           (mode="iso")  (a constant shift)

    ksp: (RO, coils, spokes_total) complex64
    traj: (3, RO, spokes_total) float32 (only used for directions)
    """
    ro, coils, spokes = ksp.shape
    u = infer_spoke_unit_vectors(traj)  # (spokes,3)

    # Candidate opposed pairs: i<j with dot(u_i, u_j) <= -dot_thresh.
    # O(N^2) is too big. Instead: for each spoke, find nearest neighbor to -u_i via random subsample.
    # Good enough for a robust fit.
    idx_all = np.arange(spokes)
    rng.shuffle(idx_all)

    pairs: List[Tuple[int, int, float]] = []
    # Build a smaller search set for speed
    search_set = idx_all[: min(spokes, 20000)]
    search_u = u[search_set]

    for i in idx_all[: min(spokes, 20000)]:
        ui = u[i]
        target = -ui
        dots = search_u @ target
        jpos = int(np.argmax(dots))
        j = int(search_set[jpos])
        d = float(u[i] @ u[j])
        if d <= -dot_thresh and i != j:
            a, b = (i, j) if i < j else (j, i)
            pairs.append((a, b, d))
        if len(pairs) >= max_pairs:
            break

    if len(pairs) < 16:
        die(f"Not enough opposed pairs found (got {len(pairs)}). "
            f"Try lowering --gdelay-dot-thresh (currently {dot_thresh}).")

    # Estimate shifts
    S: List[float] = []
    W: List[float] = []
    U: List[np.ndarray] = []

    # Coil combine for shift estimation: use SoS magnitude weighting across coils
    # but keep complex phase by summing coils normalized by magnitude.
    # This tends to be stable enough for RO shift.
    eps = 1e-6

    for (i, j, d) in pairs:
        a = ksp[:, :, i]  # (RO, coils)
        b = ksp[:, :, j]

        wa = np.sqrt(np.sum(np.abs(a) ** 2, axis=1) + eps)  # (RO,)
        wb = np.sqrt(np.sum(np.abs(b) ** 2, axis=1) + eps)

        # complex coil-combined lines
        la = np.sum(a, axis=1) / (np.sum(np.abs(a), axis=1) + eps)
        lb = np.sum(b, axis=1) / (np.sum(np.abs(b), axis=1) + eps)

        # opposed spokes: compare la vs conj(reverse(lb))
        lb2 = np.conj(lb[::-1])

        s, score = _best_shift_corr(la, lb2, max_shift=max_shift)
        if score <= 0:
            continue

        S.append(float(s))
        W.append(float(score))
        U.append(u[i].astype(np.float64))

    if len(S) < 16:
        die(f"Opposed shift estimation produced too few usable pairs ({len(S)}). "
            f"Try increasing --gdelay-max-pairs or --gdelay-max-shift.")

    svec = np.asarray(S, dtype=np.float64)
    w = np.asarray(W, dtype=np.float64)
    umat = np.vstack(U)  # (n,3)

    # Weighted least squares
    if mode == "iso":
        # s ≈ a0
        a0 = np.sum(w * svec) / (np.sum(w) + 1e-12)
        resid = svec - a0
        rms = float(np.sqrt(np.sum(w * resid * resid) / (np.sum(w) + 1e-12)))
        return GDelayFitResult(mode="iso", a=np.asarray([a0], dtype=np.float64), used_pairs=len(svec), rms_resid=rms)

    if mode == "xyz":
        # s ≈ u @ a
        Wsqrt = np.sqrt(w)[:, None]
        A = umat * Wsqrt  # (n,3)
        b = svec[:, None] * Wsqrt  # (n,1)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        coef = coef[:, 0]  # (3,)
        resid = svec - (umat @ coef)
        rms = float(np.sqrt(np.sum(w * resid * resid) / (np.sum(w) + 1e-12)))
        return GDelayFitResult(mode="xyz", a=coef, used_pairs=len(svec), rms_resid=rms)

    die(f"Unknown gdelay fit mode: {mode}")
    raise AssertionError("unreachable")


def apply_gradient_delay_shift(
    traj: np.ndarray,
    ksp: np.ndarray,
    fit: GDelayFitResult,
) -> np.ndarray:
    """
    Apply per-spoke fractional RO shift to k-space using Fourier shift theorem.

    shift (in samples) per spoke:
      iso: s = a0
      xyz: s = u @ a

    Returns corrected ksp with same shape.
    """
    ro, coils, spokes = ksp.shape
    u = infer_spoke_unit_vectors(traj)  # (spokes,3)

    if fit.mode == "iso":
        shifts = np.full((spokes,), float(fit.a[0]), dtype=np.float64)
    else:
        shifts = (u @ fit.a.reshape(3, 1)).reshape(-1).astype(np.float64)

    # FFT along RO for each coil+spoke
    # We'll do it in chunks for memory sanity.
    out = np.empty_like(ksp, dtype=np.complex64)

    n = np.arange(ro, dtype=np.float64)  # frequency index in Fourier domain after FFT
    # Use FFT shift formula: shift in sample domain corresponds to phase ramp in FFT domain.
    # For discrete FFT, the phase ramp uses k = 0..RO-1 with wrap; np.fft uses that ordering.
    # Multiply by exp(-2πi * shift * k / RO)
    for s0 in range(spokes):
        sh = shifts[s0]
        phase = np.exp(-2j * np.pi * sh * n / float(ro)).astype(np.complex64)  # (RO,)
        for c in range(coils):
            x = ksp[:, c, s0]
            X = np.fft.fft(x, axis=0)
            y = np.fft.ifft(X * phase, axis=0)
            out[:, c, s0] = y.astype(np.complex64, copy=False)

    return out


# ----------------------------
# Trajectory formatting for BART
# ----------------------------

def traj_to_bart_complex(traj: np.ndarray) -> np.ndarray:
    """
    BART wants traj as complex array (imag part 0) with dims [3, RO, spokes].
    We'll store as complex64.
    """
    t = np.asarray(traj, dtype=np.float32)
    return (t.astype(np.complex64)).copy(order="F")


def ksp_to_bart(ksp: np.ndarray) -> np.ndarray:
    """
    BART wants ksp dims [1, RO, spokes, coils]
    """
    ro, coils, spokes = ksp.shape
    y = np.transpose(ksp, (0, 2, 1))  # (RO, spokes, coils)
    y = y.reshape(1, ro, spokes, coils)
    return np.asarray(y, dtype=np.complex64, order="F")


# ----------------------------
# Sliding window frames
# ----------------------------

def compute_frames(nspokes: int, spokes_per_frame: int, frame_shift: int) -> List[Tuple[int, int]]:
    if spokes_per_frame <= 0 or frame_shift <= 0:
        die("spokes_per_frame and frame_shift must be > 0")
    if spokes_per_frame > nspokes:
        return [(0, nspokes)]
    frames = []
    start = 0
    while start + spokes_per_frame <= nspokes:
        frames.append((start, start + spokes_per_frame))
        start += frame_shift
        if start == frames[-1][0]:
            break
    if not frames:
        frames = [(0, nspokes)]
    return frames


# ----------------------------
# NIfTI export
# ----------------------------

def write_nifti(path: Path, vol: np.ndarray, voxel_size: Optional[Tuple[float, float, float]] = None) -> None:
    try:
        import nibabel as nib
    except Exception as ex:
        die(f"nibabel required for --export-nifti but failed to import: {ex}")

    v = np.asarray(vol, dtype=np.float32)
    if v.ndim == 3:
        affine = np.eye(4, dtype=np.float32)
        if voxel_size is not None:
            affine[0, 0] = float(voxel_size[0])
            affine[1, 1] = float(voxel_size[1])
            affine[2, 2] = float(voxel_size[2])
        img = nib.Nifti1Image(v, affine)
        nib.save(img, str(path))
        return

    if v.ndim == 4:
        affine = np.eye(4, dtype=np.float32)
        if voxel_size is not None:
            affine[0, 0] = float(voxel_size[0])
            affine[1, 1] = float(voxel_size[1])
            affine[2, 2] = float(voxel_size[2])
        img = nib.Nifti1Image(v, affine)
        nib.save(img, str(path))
        return

    die(f"write_nifti expects 3D or 4D, got shape={v.shape}")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Bruker radial reconstruction with BART + optional gradient delay correction."
    )
    ap.add_argument("--series", type=Path, required=True, help="Path to Bruker series directory (contains fid, traj, method, acqp).")
    ap.add_argument("--traj-source", choices=["trajfile"], default="trajfile", help="Trajectory source.")
    ap.add_argument("--out", type=str, required=True, help="Output base path (no extension).")

    ap.add_argument("--spokes-per-frame", type=int, default=0, help="Sliding window spokes per frame (0 => all).")
    ap.add_argument("--frame-shift", type=int, default=0, help="Sliding window frame shift (0 => all).")
    ap.add_argument("--spoke-order", choices=["repeat"], default="repeat",
                    help="How to expand per-volume traj to total spokes. 'repeat' => repeat same NPro for each volume block.")

    ap.add_argument("--qa-first", type=int, default=1, help="Write QA NIfTI of first N reconstructed frames.")
    ap.add_argument("--export-nifti", action="store_true", help="Write final 4D NIfTI as <out>.nii.gz")

    # Gradient delay correction options
    ap.add_argument("--gdelay-fit", choices=["none", "iso", "xyz"], default="none",
                    help="Fit gradient delay as RO shift using opposed spoke pairs.")
    ap.add_argument("--gdelay-apply", action="store_true",
                    help="Apply fitted per-spoke RO shifts to k-space before recon.")
    ap.add_argument("--gdelay-dot-thresh", type=float, default=0.995,
                    help="Opposed spoke threshold: require dot(u_i,u_j) <= -thresh.")
    ap.add_argument("--gdelay-max-pairs", type=int, default=2048,
                    help="Max opposed pairs to use for fitting.")
    ap.add_argument("--gdelay-max-shift", type=int, default=8,
                    help="Max RO shift (samples) searched per pair.")
    ap.add_argument("--gdelay-seed", type=int, default=0, help="RNG seed for pairing search.")

    # BART path
    ap.add_argument("--bart-bin", type=str, default=os.environ.get("BART_BIN", "bart"),
                    help="Path to BART 'bart' executable (or set BART_BIN env var).")

    args = ap.parse_args()

    series = args.series
    if not series.exists():
        die(f"--series not found: {series}")

    fid_path = series / "fid"
    traj_path = series / "traj"
    method_path = series / "method"
    acqp_path = series / "acqp"

    method = read_bruker_params(method_path)
    acqp = read_bruker_params(acqp_path)

    # Infer matrix size
    mat = infer_matrix_xyz(method)
    if mat is None:
        die("Could not infer PVM_Matrix from method.")
    NX, NY, NZ = mat
    print(f"[info] Matrix inferred from PVM_Matrix: {NX}x{NY}x{NZ}")

    # Infer coils
    ncoils = infer_ncoils(method, acqp)
    if ncoils is None:
        die("Could not infer number of coils (PVM_EncNReceivers / ACQ_nReceivers).")
    print(f"[info] Coils inferred from PVM_EncNReceivers: {ncoils}")

    # Infer ACQ_size (words) and RO complex
    acq_size_words = infer_acq_size_words(acqp)
    if acq_size_words is None:
        die("Could not read ACQ_size from acqp.")
    # MATLAB-faithful convention: ACQ_size is words; for complex data, words = 2*n_read
    # But your earlier logs sometimes treated ACQ_size as stored_ro. Here we do faithful:
    if acq_size_words % 2 != 0:
        die(f"ACQ_size={acq_size_words} is odd; expected even for complex re/im words.")
    ro_complex = acq_size_words // 2
    print(f"[info] ACQ_size (words) from acqp: {acq_size_words} -> using complex RO={ro_complex}")

    # Infer NPro (spokes per volume)
    npro = infer_npro(method, acqp)
    if npro is None:
        die("Could not infer NPro (spokes per volume). Ensure method/acqp has ##$NPro.")
    print(f"[info] NPro inferred from method/acqp: {npro}")

    # FID dtype + endian
    fid_dt, endian = fid_dtype_and_endian(acqp)
    print(f"[info] FID dtype from ACQ_word_size: {fid_dt.name}, endian from BYTORDA: {endian}")

    # Read k-space (RO, coils, spokes_total)
    ksp = read_fid_matlab_faithful(fid_path, ncoils=ncoils, ro_complex=ro_complex, endian_dt=fid_dt)
    ro_ksp, coils_ksp, spokes_total = ksp.shape
    if ro_ksp != ro_complex or coils_ksp != ncoils:
        die(f"Internal shape mismatch after fid parse: ksp={ksp.shape} expected RO={ro_complex},coils={ncoils}")
    print(f"[info] Loaded k-space (MATLAB-faithful): RO={ro_ksp}, spokes={spokes_total}, coils={coils_ksp}")

    # Infer n_vols from total spokes / NPro
    if spokes_total % npro != 0:
        eprint(f"[warn] spokes_total={spokes_total} not divisible by NPro={npro}. "
               f"Proceeding, but spoke grouping into volumes may be wrong.")
        nvols = max(1, spokes_total // npro)
    else:
        nvols = spokes_total // npro

    # Read traj (3, RO, NPro) and expand to total spokes via repeat
    traj_vol = read_trajfile(
    traj_path,
    ro=ro_complex,
    npro=npro,
    endian=endian,
    matrix_xyz=(NX, NY, NZ),
)

    print(f"[info] trajfile parsed (MATLAB-faithful): f8_{endian.lower()} (NPro={npro}, RO={ro_complex})")
	
    traj_debug_stats(traj_vol, "trajfile raw")

    if args.spoke_order != "repeat":
        die("Only --spoke-order repeat is implemented in this script.")

    reps = int(math.ceil(spokes_total / npro))
    traj_full = np.tile(traj_vol, reps=(1, 1, reps))[:, :, :spokes_total]  # (3, RO, spokes_total)

    traj_debug_stats(traj_full, "trajfile expanded")

    # Optional gradient delay fit + apply
    if args.gdelay_fit != "none":
        rng = np.random.default_rng(int(args.gdelay_seed))
        fit = fit_gradient_delay(
            traj_full,
            ksp,
            dot_thresh=float(args.gdelay_dot_thresh),
            max_pairs=int(args.gdelay_max_pairs),
            max_shift=int(args.gdelay_max_shift),
            mode=str(args.gdelay_fit),
            rng=rng,
        )
        if fit.mode == "iso":
            print(f"[info] gdelay fit iso: shift={fit.a[0]:.4f} samples  (pairs={fit.used_pairs}, rms={fit.rms_resid:.4f})")
        else:
            ax, ay, az = fit.a.tolist()
            print(f"[info] gdelay fit xyz: ax={ax:.4f}, ay={ay:.4f}, az={az:.4f} samples  (pairs={fit.used_pairs}, rms={fit.rms_resid:.4f})")

        if args.gdelay_apply:
            print("[info] Applying gradient delay correction (per-spoke fractional RO shift) to k-space...")
            ksp = apply_gradient_delay_shift(traj_full, ksp, fit)
        else:
            print("[info] gdelay fit computed but not applied (use --gdelay-apply).")

    # Frame plan
    if args.spokes_per_frame <= 0 or args.frame_shift <= 0:
        spokes_per_frame = spokes_total
        frame_shift = spokes_total
    else:
        spokes_per_frame = int(args.spokes_per_frame)
        frame_shift = int(args.frame_shift)

    frames = compute_frames(spokes_total, spokes_per_frame, frame_shift)
    print(f"[info] Sliding window: spokes-per-frame={spokes_per_frame}, frame-shift={frame_shift}")
    print(f"[info] Will reconstruct {len(frames)} frame(s).")

    # BART presence
    bart = args.bart_bin
    if not (os.path.isabs(bart) or which(bart)):
        die(f"Could not find BART executable '{bart}'. Set --bart-bin or BART_BIN.")

    outbase = Path(args.out)
    outdir = outbase.parent if outbase.parent != Path("") else Path(".")
    outdir.mkdir(parents=True, exist_ok=True)

    # Reconstruct each frame
    recon_vols: List[np.ndarray] = []
    qa_n = max(0, int(args.qa_first))

    with tempfile.TemporaryDirectory(prefix="bruker_radial_bart_") as td:
        #td = Path(td)
        td = outdir
        for fi, (a, b) in enumerate(frames):
            print(f"[info] Frame {fi} spokes [{a}:{b}] (n={b-a})")

            traj_f = traj_to_bart_complex(traj_full[:, :, a:b])  # (3, RO, spokes)
            ksp_f = ksp_to_bart(ksp[:, :, a:b])                  # (1, RO, spokes, coils)

            traj_cfl = td / f"vol{fi:05d}_traj"
            ksp_cfl = td / f"vol{fi:05d}_ksp"
            coil_cfl = td / f"vol{fi:05d}_coil"
            sos_cfl = td / f"vol{fi:05d}_sos"

            writecfl(traj_cfl, traj_f)
            writecfl(ksp_cfl, ksp_f)

            # bart nufft -i -d NX:NY:NZ traj ksp coilimg
            cmd_nufft = [
                bart, "nufft", "-i", f"-d{NX}:{NY}:{NZ}",
                str(traj_cfl), str(ksp_cfl), str(coil_cfl)
            ]
            print(f"[bart] {' '.join(cmd_nufft)}")
            run(cmd_nufft)

            # bart rss 8 coilimg sosimg   (dim=8 is coil dim for BART convention; matches your logs)
            cmd_rss = [bart, "rss", "8", str(coil_cfl), str(sos_cfl)]
            print(f"[bart] {' '.join(cmd_rss)}")
            run(cmd_rss)

            # Read back sos for QA / final; easiest via nibabel? We'll read CFL directly.
            sos = read_cfl_simple(sos_cfl)
            # sos is complex (should be real-ish). Use abs.
            sos_mag = np.abs(sos).astype(np.float32)

            recon_vols.append(sos_mag)

            if qa_n > 0 and fi < qa_n:
                # write per-frame nifti for quick eyeballing
                qa_path = outdir / f"{outbase.name}_QA_frame{fi:03d}.nii.gz"
                write_nifti(qa_path, sos_mag)
                print(f"[info] Wrote QA NIfTI {qa_path} with shape {sos_mag.shape}")

    # Stack to 4D
    out4d = np.stack(recon_vols, axis=-1).astype(np.float32)
    if args.export_nifti:
        nii_path = outdir / f"{outbase.name}.nii.gz"
        write_nifti(nii_path, out4d)
        print(f"[info] Wrote final 4D NIfTI {nii_path}")

    print(f"[info] Done. 4D result base: {outbase}")


def read_cfl_simple(base: Path) -> np.ndarray:
    """
    Minimal CFL reader for internal use.
    """
    base = Path(base)
    hdr = base.with_suffix(".hdr").read_text(errors="ignore").splitlines()
    dims_line = None
    for i, line in enumerate(hdr):
        if line.strip().startswith("#"):
            continue
        if line.strip() == "":
            continue
        dims_line = line.strip()
        break
    if dims_line is None:
        # Try second line (BART style header often: "# Dimensions" then dims line)
        for line in hdr:
            if re.match(r"^\s*\d", line):
                dims_line = line.strip()
                break
    if dims_line is None:
        die(f"Could not parse dims from {base.with_suffix('.hdr')}")

    dims = [int(x) for x in dims_line.split()]
    # drop trailing ones
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()

    raw = np.fromfile(str(base.with_suffix(".cfl")), dtype=np.float32)
    if raw.size % 2 != 0:
        die(f"Odd float count in {base.with_suffix('.cfl')}")
    c = raw[0::2] + 1j * raw[1::2]
    n = int(np.prod(dims))
    if c.size != n:
        die(f"CFL size mismatch for {base}: expected {n} complex, got {c.size}")
    arr = c.reshape(dims, order="F")
    return arr


if __name__ == "__main__":
    main()
	