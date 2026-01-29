#!/usr/bin/env python3
"""
Bruker 3D radial -> BART NUFFT recon driver.

Supports trajectory from:
  - traj file (Bruker writes <series>/traj)  --traj-source trajfile
  - grad.output directions                   --traj-source gradoutput
  - auto: prefer trajfile if usable else gradoutput

Key points:
  - BART NUFFT expects k-space dims[0] == 1 => write ksp as (1, RO, spokes, coils)
  - Traj written as (3, RO, spokes)

This version:
  - Keeps prior argument surface; NEVER requires specifying N_coils/N_read/N_vols manually.
  - Uses MATLAB-faithful traj parsing by default:
      * reads float64, tries LE/BE, reshapes with order='F'
      * infers traj RO from file length and NPro (##$NPro)
      * scales by NX (PVM_Matrix[0]) so [-0.5,0.5] -> [-NX/2, NX/2]
  - Adds MATLAB-faithful FID parsing path for int32 with 1024-byte block trimming.

If you want your older autoshape logic back later, we can add it as a fallback mode,
but right now we're prioritizing “match MATLAB that you verified”.
"""

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

        if rhs.startswith("("):  # multiline array
            vals = []
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
                    out.append(float(v) if "." in v or "e" in v.lower() else int(v))
                except ValueError:
                    out.append(v)
            return out[0] if len(out) == 1 else out

        rhs = rhs.strip("()")
        toks = rhs.split()
        out = []
        for v in toks:
            try:
                out.append(float(v) if "." in v or "e" in v.lower() else int(v))
            except ValueError:
                out.append(v)
        return out[0] if len(out) == 1 else out

    return default


def infer_matrix(method: Path) -> Tuple[int, int, int]:
    mat = read_bruker_param(method, "PVM_Matrix", None)
    if mat is None or isinstance(mat, (int, float)) or len(mat) != 3:
        raise ValueError(f"Could not infer PVM_Matrix (got {mat})")
    return tuple(map(int, mat))


def infer_true_ro(acqp: Path) -> int:
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


def bart_supports_gpu(bart_bin: str = "bart") -> bool:
    proc = subprocess.run([bart_bin, "nufft", "-i", "-g"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode == 0:
        return True
    s = (proc.stderr or "").lower()
    if "compiled without gpu support" in s or "unknown option" in s:
        return False
    return False


# ---------------- NIfTI writers ---------------- #

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


# ---------------- Trajectory: MATLAB-faithful trajfile ---------------- #

def _traj_sanity_score(v: np.ndarray) -> float:
    """
    Very lightweight sanity:
      - finite
      - not all zeros
      - magnitude percentiles not absurd
    Returns higher=better.
    """
    if not np.all(np.isfinite(v)):
        return -1e9
    vmax = float(np.max(np.abs(v)))
    if vmax == 0.0:
        return -1e9
    p = np.percentile(np.abs(v), [50, 95, 99])
    # prefer values that look like “fractions” before scaling (often <= ~1)
    score = 0.0
    score += -abs(np.log10(max(p[0], 1e-12)) - np.log10(0.5))
    score += -abs(np.log10(max(p[1], 1e-12)) - np.log10(1.0))
    score += -abs(np.log10(max(p[2], 1e-12)) - np.log10(2.0))
    return score


def parse_trajfile_matlab_faithful(
    traj_path: Path,
    *,
    npro: int,
    nx: int,
) -> Tuple[np.ndarray, int, str]:
    """
    MATLAB-faithful:
      traj = fread(..., 'float64', 'ieee-le');
      traj = reshape(traj, 3, n_read, []);
      traj = traj * traj_scaling;   % where traj_scaling should be NX (not hardcoded)

    We infer n_read from file length and NPro:
      len = 3 * n_read * npro  (float64 elements)

    Returns:
      traj: complex64 shaped (3, n_read, npro)
      n_read
      tag: 'f8_le_F' or 'f8_be_F'
    """
    b = traj_path.read_bytes()
    if len(b) % 8 != 0:
        raise ValueError(f"traj bytes {len(b)} not multiple of 8; cannot be float64")

    ne = len(b) // 8
    if ne % (3 * npro) != 0:
        raise ValueError(
            f"traj element count {ne} not divisible by (3*NPro)={3*npro}. "
            f"Cannot infer n_read."
        )

    n_read = ne // (3 * npro)

    v_le = np.frombuffer(b, dtype="<f8")
    v_be = np.frombuffer(b, dtype=">f8")

    s_le = _traj_sanity_score(v_le)
    s_be = _traj_sanity_score(v_be)

    if s_be > s_le:
        v = v_be
        tag = "f8_be_F"
    else:
        v = v_le
        tag = "f8_le_F"

    traj = v.reshape(3, n_read, npro, order="F").astype(np.float32, copy=False)

    # Scaling: MATLAB used traj_scaling=96, but the general rule is scale by NX
    # so [-0.5,0.5] -> [-NX/2, NX/2].
    traj *= float(nx)

    traj_c = np.asfortranarray(traj).astype(np.complex64, copy=False)
    return traj_c, int(n_read), tag


def traj_radial_profile_debug(traj: np.ndarray, label: str = "traj") -> None:
    k_mag = np.sqrt(
        np.real(traj[0]) ** 2 +
        np.real(traj[1]) ** 2 +
        np.real(traj[2]) ** 2
    ).astype(np.float64, copy=False)
    k_ro_med = np.median(k_mag, axis=1)
    imin = int(np.argmin(k_ro_med))
    mid = k_ro_med.shape[0] // 2
    print(f"[debug] {label} |k| median at RO[0],RO[mid],RO[-1]: {k_ro_med[0]:.6g}, {k_ro_med[mid]:.6g}, {k_ro_med[-1]:.6g}")
    print(f"[debug] {label} |k| median min at RO={imin} (|k|≈{k_ro_med[imin]:.6g})")


def recenter_ro_by_kmin_if_centered(traj: np.ndarray, *, label: str) -> Tuple[np.ndarray, int]:
    """
    If centered readout, roll RO so median(|k|) minimum is at RO[mid].
    """
    k_mag = np.sqrt(
        np.real(traj[0]) ** 2 +
        np.real(traj[1]) ** 2 +
        np.real(traj[2]) ** 2
    ).astype(np.float64, copy=False)
    k_ro_med = np.median(k_mag, axis=1)
    imin = int(np.argmin(k_ro_med))
    mid = int(k_ro_med.shape[0] // 2)

    # only shift if min isn't near ends
    if 0.1 * k_ro_med.shape[0] < imin < 0.9 * k_ro_med.shape[0]:
        shift = mid - imin
        if shift != 0:
            traj2 = np.roll(traj, shift=shift, axis=1)
            print(f"[info] {label}: centered readout; shifted RO axis by {shift} to place |k| med-min at RO[mid]={mid} (imin={imin})")
            return traj2, shift
        return traj, 0

    print(f"[info] {label}: center-out readout detected (|k| min at RO={imin}); no RO recentering applied")
    return traj, 0


def expand_traj_spokes(traj: np.ndarray, target_spokes: int, order: str) -> np.ndarray:
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


# ---------------- grad.output trajectory ---------------- #

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


def build_traj_from_dirs(true_ro: int, dirs_xyz: np.ndarray, NX: int) -> np.ndarray:
    spokes = dirs_xyz.shape[0]
    # Build centered readout from -NX/2..NX/2 in "pixel" units
    kmax = 0.5 * float(NX)
    s = np.linspace(-kmax, kmax, true_ro, dtype=np.float64)

    traj = np.zeros((3, true_ro, spokes), dtype=np.complex64)
    dx, dy, dz = dirs_xyz[:, 0], dirs_xyz[:, 1], dirs_xyz[:, 2]
    for i in range(spokes):
        traj[0, :, i] = s * dx[i]
        traj[1, :, i] = s * dy[i]
        traj[2, :, i] = s * dz[i]

    print(f"[info] Traj built from grad.output with max |k| ≈ {np.abs(traj).max():.2f}")
    return traj


# ---------------- FID / k-space loading ---------------- #

def factor_fid(total_points: int, true_ro: int, coils_hint: Optional[int]):
    block_candidates = [true_ro]
    for b in (128, 96, 64, 256, 192, 384, 512):
        if b != true_ro:
            block_candidates.append(b)

    if coils_hint is not None:
        coil_candidates = [coils_hint] + [c for c in range(1, 33) if c != coils_hint]
    else:
        coil_candidates = list(range(1, 33))

    best = None
    best_score = -1

    for stored_ro in block_candidates:
        for c in coil_candidates:
            denom = stored_ro * c
            if denom <= 0:
                continue
            if total_points % denom != 0:
                continue
            spokes = total_points // denom

            score = 0
            if stored_ro >= true_ro:
                score += 2
            if abs(stored_ro - true_ro) <= 10:
                score += 2
            if spokes > 100:
                score += 1
            if coils_hint is not None and c == coils_hint:
                score += 2

            if score > best_score:
                best_score = score
                best = (stored_ro, c, spokes)

    if best is None:
        raise ValueError(
            f"Could not factor FID: total={total_points}, true_ro={true_ro}, coils_hint={coils_hint}"
        )
    return best  # (stored_ro, coils, spokes)


def load_bruker_kspace_simple(
    fid_path: Path,
    true_ro: int,
    coils_hint: Optional[int],
    endian: str,
    base_kind: str,
    fid_layout: str,
):
    """
    Your prior “simple” loader: read raw interleaved re/im, factor into (stored_ro, spokes, coils),
    trim to true_ro, return ksp (true_ro, spokes, coils).
    """
    if not fid_path.exists():
        raise FileNotFoundError(f"FID not found: {fid_path}")

    np_dtype = np.dtype(endian + base_kind)
    raw = np.fromfile(fid_path, dtype=np_dtype).astype(np.float32)
    if raw.size % 2 != 0:
        raise ValueError(f"FID length {raw.size} not even (real/imag pairs).")

    cplx = raw[0::2] + 1j * raw[1::2]
    total_points = cplx.size

    stored_ro, coils, spokes = factor_fid(total_points, true_ro, coils_hint)

    if coils_hint is not None and coils != coils_hint:
        print(f"[warn] FID suggests coils={coils}, header said {coils_hint}; using {coils}.", file=sys.stderr)

    print(f"[info] Loaded k-space with stored_ro={stored_ro}, true_ro={true_ro}, spokes={spokes}, coils={coils}")

    if fid_layout == "ro_spokes_coils":
        ksp = cplx.reshape(stored_ro, spokes, coils)
    elif fid_layout == "ro_coils_spokes":
        ksp = cplx.reshape(stored_ro, coils, spokes).transpose(0, 2, 1)
    else:
        raise ValueError(f"Unknown fid_layout: {fid_layout}")

    if stored_ro != true_ro:
        if stored_ro < true_ro:
            raise ValueError(f"stored_ro={stored_ro} < true_ro={true_ro}")
        ksp = ksp[:true_ro]
        print(f"[info] Trimmed k-space from stored_ro={stored_ro} to true_ro={true_ro}")

    return ksp.astype(np.complex64, copy=False), stored_ro, spokes, coils


def load_bruker_kspace_matlab_blocktrim_i4(
    fid_path: Path,
    *,
    n_read: int,
    n_coils: int,
    npro: Optional[int],
    endian: str = "<",
    block_bytes: int = 1024,
):
    """
    MATLAB-faithful block-trim path for int32 LE/BE.

    Mirrors:
      x = fread(..., int32, ieee-le)
      n_ray = n_coils * n_read * 2
      n_blocks = ceil(n_ray*4 / 1024)
      n_block_samples = n_blocks*1024 / 4
      x = reshape(x, n_block_samples, [])
      x = x(1:n_ray,:)
      x = x(:)
      x = reshape(x, 2, n_read, n_coils, [], n_vols)

    We infer:
      total_rays = raw.size / n_block_samples
      spokes_total = total_rays
      if npro provided and divisible: n_vols = spokes_total / npro, spokes_per_vol = npro
      else: n_vols = 1, spokes_per_vol = spokes_total
    """
    if not fid_path.exists():
        raise FileNotFoundError(f"FID not found: {fid_path}")

    raw = np.fromfile(fid_path, dtype=np.dtype(endian + "i4"))

    n_ray = int(n_coils * n_read * 2)
    bytes_per_sample = 4
    n_blocks = int(np.ceil((n_ray * bytes_per_sample) / float(block_bytes)))
    n_block_samples = int((n_blocks * block_bytes) // bytes_per_sample)

    if n_block_samples <= 0:
        raise ValueError("Computed n_block_samples <= 0; check inputs.")

    if raw.size % n_block_samples != 0:
        raise ValueError(
            f"FID size {raw.size} not divisible by n_block_samples={n_block_samples} "
            f"(n_blocks={n_blocks}, block_bytes={block_bytes})."
        )

    total_rays = raw.size // n_block_samples

    # column-major reshape like MATLAB
    x = raw.reshape(n_block_samples, total_rays, order="F")
    x = x[:n_ray, :]
    x = x.reshape(-1, order="F")

    denom = 2 * n_read * n_coils
    if x.size % denom != 0:
        raise ValueError(
            f"After block trim, sample count {x.size} not divisible by (2*n_read*n_coils)={denom}."
        )
    spokes_total = x.size // denom

    if npro is not None and npro > 0 and (spokes_total % npro == 0):
        n_vols = spokes_total // npro
        spokes_per_vol = npro
    else:
        n_vols = 1
        spokes_per_vol = spokes_total

    # shape: (2, n_read, n_coils, spokes_per_vol, n_vols)
    x2 = x.reshape(2, n_read, n_coils, spokes_per_vol, n_vols, order="F").astype(np.float32, copy=False)
    cplx = x2[0, ...] + 1j * x2[1, ...]  # (n_read, n_coils, spokes_per_vol, n_vols)

    # convert to (n_read, spokes_total, n_coils) by stacking vols along spokes
    if n_vols == 1:
        ksp = cplx[:, :, :, 0].transpose(0, 2, 1)  # (n_read, spokes, n_coils)
        spokes_all = spokes_per_vol
    else:
        # (n_read, n_coils, spokes, vols) -> (n_read, n_coils, spokes*vols) -> (n_read, spokes_all, n_coils)
        cplx2 = cplx.reshape(n_read, n_coils, spokes_per_vol * n_vols, order="F")
        ksp = cplx2.transpose(0, 2, 1)
        spokes_all = spokes_per_vol * n_vols

    print(f"[info] MATLAB-blocktrim FID: n_read={n_read}, coils={n_coils}, spokes_all={spokes_all}, n_vols={n_vols}, n_block_samples={n_block_samples}")

    return ksp.astype(np.complex64, copy=False), spokes_all, n_vols


# ---------------- traj source selection ---------------- #

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
    out = []
    for p in cands:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def load_traj_auto(
    series_path: Path,
    method: Path,
    acqp: Path,
    spokes_all: int,
    NX: int,
    true_ro: int,
    spoke_order: str,
    traj_source: str,
    traj_file: Optional[Path],
) -> Tuple[np.ndarray, str]:
    """
    Returns traj_full (3, RO, spokes_all), description.
    trajfile path uses MATLAB-faithful parsing and will infer RO from the file itself.
    """

    def _from_gradoutput() -> np.ndarray:
        grad_path = series_path / "grad.output"
        if not grad_path.exists():
            raise RuntimeError(f"grad.output not found in {series_path}")
        dirs = load_grad_output_dirs(grad_path, normalize=True)
        dirs_full = expand_spoke_dirs(dirs, spokes_all, spoke_order)
        traj = build_traj_from_dirs(true_ro, dirs_full, NX)
        return traj

    def _from_trajfile(p: Path) -> Tuple[np.ndarray, str]:
        npro = infer_npro(method, acqp)
        if npro is None:
            raise RuntimeError("trajfile parsing requires NPro, but could not read ##$NPro from method/acqp.")

        traj, n_read, tag = parse_trajfile_matlab_faithful(p, npro=int(npro), nx=int(NX))

        # Sanity logs
        print(f"[info] trajfile parsed (MATLAB-faithful): {tag} (NPro={npro}, RO={n_read})")
        traj_radial_profile_debug(traj, label="trajfile raw")

        # Recenter if centered (kmin to mid)
        traj2, _shift = recenter_ro_by_kmin_if_centered(traj, label="trajfile")
        if _shift != 0:
            traj_radial_profile_debug(traj2, label="trajfile after_recenter_kmin")

        # Expand spokes if needed
        traj2 = expand_traj_spokes(traj2, target_spokes=spokes_all, order=spoke_order)

        traj_radial_profile_debug(traj2, label="trajfile final")
        return traj2, f"trajfile:{tag}:(NPro={npro},RO={n_read},3)"

    if traj_source == "gradoutput":
        return _from_gradoutput(), "gradoutput"

    if traj_file is not None:
        if not traj_file.exists():
            raise RuntimeError(f"--traj-file does not exist: {traj_file}")
        traj, mode = _from_trajfile(traj_file)
        return traj, mode

    if traj_source in ("trajfile", "auto"):
        cands = find_traj_candidates(series_path)
        if traj_source == "trajfile" and not cands:
            raise RuntimeError(f"--traj-source trajfile requested, but no traj candidate found under {series_path} or pdata/*")

        last_err = None
        for p in cands:
            try:
                traj, mode = _from_trajfile(p)
                return traj, mode
            except Exception as e:
                last_err = e

        if traj_source == "trajfile":
            raise RuntimeError(
                f"--traj-source trajfile requested, but no candidate trajectory could be parsed.\nLast error: {last_err}"
            )

        return _from_gradoutput(), "gradoutput(fallback)"

    raise ValueError(f"Unknown traj-source: {traj_source}")


# ---------------- Core recon ---------------- #

def run_bart(
    series_path: Path,
    method: Path,
    acqp: Path,
    fid: Path,
    out_base: Path,
    NX: int, NY: int, NZ: int,
    true_ro: int,
    coils_hint: int,
    spokes_per_frame: int,
    frame_shift: int,
    qa_first: Optional[int],
    export_nifti: bool,
    use_gpu: bool,
    spoke_order: str,
    traj_source: str,
    traj_file: Optional[Path],
    fid_dtype: str,
    fid_endian: str,
    fid_layout: str,
):
    bart_bin = "bart"

    # --------------------------
    # Load trajectory first (trajfile mode infers its own RO)
    # --------------------------
    # We still need spokes_all from kspace; load kspace first in a way that doesn't require manual params.

    npro = infer_npro(method, acqp)

    if fid_dtype == "i4" and traj_source in ("trajfile", "auto"):
        # Use trajfile to infer n_read, then use MATLAB block-trim on FID
        # (This is the most faithful path when your MATLAB code is verified.)
        # We need n_read from trajfile parse; do that with spokes_all placeholder first? No:
        # parse trajfile without spokes_all (only needs NPro + NX).
        if npro is None:
            raise RuntimeError("For fid-dtype i4 + trajfile, need NPro in method/acqp.")

        traj_tmp, n_read_traj, tag = parse_trajfile_matlab_faithful((traj_file or (series_path / "traj")), npro=int(npro), nx=int(NX))
        # Now parse FID using n_read_traj + coils_hint
        ksp, spokes_all, n_vols = load_bruker_kspace_matlab_blocktrim_i4(
            fid,
            n_read=int(n_read_traj),
            n_coils=int(coils_hint),
            npro=int(npro) if npro is not None else None,
            endian=fid_endian,
        )

        # Trajectory for reconstruction must match ksp RO (= n_read_traj) and spokes_all
        traj_full, traj_used = load_traj_auto(
            series_path=series_path,
            method=method,
            acqp=acqp,
            spokes_all=spokes_all,
            NX=NX,
            true_ro=true_ro,        # only used for gradoutput fallback
            spoke_order=spoke_order,
            traj_source=traj_source,
            traj_file=traj_file,
        )

        # But trajfile RO may differ from true_ro; ensure RO matches ksp RO
        if traj_full.shape[1] != ksp.shape[0]:
            raise ValueError(f"traj RO={traj_full.shape[1]} does not match ksp RO={ksp.shape[0]} (MATLAB-faithful path)")
        print(f"[info] Trajectory source used: {traj_used}")

    else:
        # Fallback: old simple parser for fid (i2/i4/f4) based on true_ro
        ksp, stored_ro, spokes_all, coils = load_bruker_kspace_simple(
            fid_path=fid,
            true_ro=true_ro,
            coils_hint=coils_hint,
            endian=fid_endian,
            base_kind=fid_dtype,
            fid_layout=fid_layout,
        )
        traj_full, traj_used = load_traj_auto(
            series_path=series_path,
            method=method,
            acqp=acqp,
            spokes_all=spokes_all,
            NX=NX,
            true_ro=true_ro,
            spoke_order=spoke_order,
            traj_source=traj_source,
            traj_file=traj_file,
        )
        print(f"[info] Trajectory source used: {traj_used}")

        if traj_full.shape[1] != ksp.shape[0]:
            # If trajfile inferred a different RO than ACQ_size path, that's a big mismatch.
            # In that case, user should run with fid-dtype i4 + trajfile for MATLAB-faithful.
            raise ValueError(f"traj RO={traj_full.shape[1]} does not match ksp RO={ksp.shape[0]}")

    have_gpu = False
    if use_gpu:
        have_gpu = bart_supports_gpu(bart_bin)
        if not have_gpu:
            print("[warn] BART has no GPU support; falling back to CPU.", file=sys.stderr)

    if spokes_per_frame <= 0:
        spokes_per_frame = spokes_all
    if frame_shift <= 0:
        frame_shift = spokes_per_frame

    frame_starts = list(range(0, max(1, spokes_all - spokes_per_frame + 1), frame_shift))
    print(f"[info] Sliding window: spokes-per-frame={spokes_per_frame}, frame-shift={frame_shift}")
    print(f"[info] Will reconstruct {len(frame_starts)} frame(s).")

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

        ksp_bart = ksp_frame[np.newaxis, :, :, :]  # (1, ro, spokes, coils)

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
        description="Bruker 3D radial → BART NUFFT recon driver (trajfile MATLAB-faithful or grad.output).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:

              Use trajectory file (Bruker writes <series>/traj):
                python bruker_radial_bart.py --series /path/to/21 --traj-source trajfile --fid-dtype i4 --export-nifti --out /tmp/out

              Use grad.output:
                python bruker_radial_bart.py --series /path/to/29 --traj-source gradoutput --fid-dtype i2 --export-nifti --out /tmp/out
            """
        ),
    )

    ap.add_argument("--series", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--matrix", nargs=3, type=int, metavar=("NX", "NY", "NZ"))
    ap.add_argument("--readout", type=int)
    ap.add_argument("--coils", type=int)

    ap.add_argument("--spokes-per-frame", type=int, default=0)
    ap.add_argument("--frame-shift", type=int, default=0)

    ap.add_argument("--fid-dtype", choices=["i2", "i4", "f4"], default="i2")
    ap.add_argument("--fid-endian", choices=[">", "<"], default="<")
    ap.add_argument("--fid-layout", choices=["ro_spokes_coils", "ro_coils_spokes"], default="ro_spokes_coils")

    ap.add_argument("--spoke-order", choices=["tile", "repeat"], default="tile")

    ap.add_argument("--traj-source", choices=["auto", "trajfile", "gradoutput"], default="auto")
    ap.add_argument("--traj-file", default=None, help="Explicit traj file path (defaults to <series>/traj or pdata/**/traj)")

    ap.add_argument("--qa-first", type=int, default=0)
    ap.add_argument("--export-nifti", action="store_true")

    ap.add_argument("--gpu", action="store_true")

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
        print(f"[info] Matrix inferred from PVM_Matrix: {NX}x{NY}x{NZ}")

    if args.readout is not None:
        true_ro = int(args.readout)
        print(f"[info] Readout overridden from CLI: RO={true_ro}")
    else:
        true_ro = infer_true_ro(acqp)
        print(f"[info] Readout (true RO) from ACQ_size: RO={true_ro}")

    if args.coils is not None:
        coils_hint = int(args.coils)
        print(f"[info] Coils overridden from CLI: {coils_hint}")
    else:
        coils_hint = infer_coils(method)
        print(f"[info] Coils inferred from PVM_EncNReceivers: {coils_hint}")

    spf = args.spokes_per_frame
    shift = args.frame_shift
    qa_first = args.qa_first if args.qa_first > 0 else None
    traj_file = Path(args.traj_file).resolve() if args.traj_file else None

    run_bart(
        series_path=series_path,
        method=method,
        acqp=acqp,
        fid=fid,
        out_base=out_base,
        NX=NX, NY=NY, NZ=NZ,
        true_ro=true_ro,
        coils_hint=coils_hint,
        spokes_per_frame=spf,
        frame_shift=shift,
        qa_first=qa_first,
        export_nifti=args.export_nifti,
        use_gpu=args.gpu,
        spoke_order=args.spoke_order,
        traj_source=args.traj_source,
        traj_file=traj_file,
        fid_dtype=args.fid_dtype,
        fid_endian=args.fid_endian,
        fid_layout=args.fid_layout,
    )


if __name__ == "__main__":
    main()
