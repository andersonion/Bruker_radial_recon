#!/usr/bin/env python3
"""
Bruker 3D radial -> BART NUFFT recon driver.

This version:
  - k-space passed to BART NUFFT is written as (RO, spokes, 1, coils),
    i.e. RO is dim0 and spokes is dim1 (BART convention),
    NOT (1, RO, spokes, coils).

  - Automatically uses <series>/grad.output if present (ProjR/ProjP/ProjS)
  - Normalizes direction vectors to unit length
  - Supports spoke expansion tile/repeat when spokes = reps * N_dirs
  - Supports FID layouts:
        ro_spokes_coils (default)
        ro_coils_spokes (test2)
  - Supports readout origin centered vs zero, and reverse readout
  - Uses correct BART rss bitmask for coil dim=3 => 8

NEW (detour support):
  - Can use a *known-good trajectory file* when present in the dataset, as an option:
        --traj-source auto        (default): prefer traj file if found+parsable; else grad.output
        --traj-source trajfile    : require a traj file (error if missing)
        --traj-source gradoutput  : require grad.output (error if missing)
    You can also force a specific traj file via:
        --traj-file /path/to/traj_or_traj.cfl (base path ok)

  - Robust traj parsing:
        * If traj is a BART .cfl/.hdr pair => readcfl() and adapt dims to (3, RO, spokes)
        * Else try raw binary float32/float64 with length 3*RO*spokes (interleaved or blocked)
        * Else if file encodes per-spoke directions (spokes,3) or (3,spokes), expand along RO
          using kmax = 0.5*NX and your readout-origin/reverse settings
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


def infer_matrix(method: Path):
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


# ---------------- FID / k-space loading ---------------- #

def factor_fid(total_points: int, true_ro: int, coils_hint: int | None):
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


def load_bruker_kspace(
    fid_path: Path,
    true_ro: int,
    coils_hint: int | None,
    endian: str,
    base_kind: str,
    fid_layout: str,
):
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
        print(
            f"[warn] FID suggests coils={coils}, header said {coils_hint}; using {coils}.",
            file=sys.stderr,
        )

    print(
        f"[info] Loaded k-space with stored_ro={stored_ro}, "
        f"true_ro={true_ro}, spokes={spokes}, coils={coils}"
    )

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

    return ksp, stored_ro, spokes, coils


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
            f"CFL size mismatch: have {cplx.size} complex, expected {expected} "
            f"from dims {dims} for {base}"
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
    proc = subprocess.run(
        [bart_bin, "nufft", "-i", "-g"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode == 0:
        return True
    s = (proc.stderr or "").lower()
    if "compiled without gpu support" in s or "unknown option" in s:
        return False
    return False


# ---------------- grad.output trajectory ---------------- #

def _extract_grad_block(lines: list[str], name: str) -> np.ndarray:
    hdr_pat = re.compile(
        rf"^\s*\d+:{re.escape(name)}:\s+index\s*=\s*\d+,\s+size\s*=\s*(\d+)\s*$"
    )

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
    print(
        f"[info] Direction norms BEFORE normalize: "
        f"min={norms.min():.4f} median={np.median(norms):.4f} max={norms.max():.4f}"
    )

    if normalize:
        # Avoid divide-by-zero if any weird entry appears
        bad = norms == 0
        if np.any(bad):
            raise ValueError(f"Found {bad.sum()} zero-norm direction(s) in {grad_output_path}")
        dirs = dirs / norms[:, None]
        norms2 = np.linalg.norm(dirs, axis=1)
        print(
            f"[info] Direction norms AFTER  normalize: "
            f"min={norms2.min():.4f} median={np.median(norms2):.4f} max={norms2.max():.4f}"
        )

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


def _build_readout_s(
    true_ro: int,
    NX: int,
    traj_scale: float | None,
    readout_origin: str,
    reverse_readout: bool,
) -> np.ndarray:
    kmax = 0.5 * NX
    if traj_scale is not None:
        kmax *= float(traj_scale)

    if readout_origin == "zero":
        s = np.linspace(0.0, kmax, true_ro, dtype=np.float64)
    else:
        s = np.linspace(-kmax, kmax, true_ro, dtype=np.float64)

    if reverse_readout:
        s = s[::-1].copy()

    return s


def build_traj_from_dirs(
    true_ro: int,
    dirs_xyz: np.ndarray,   # (spokes,3)
    NX: int,
    traj_scale: float | None,
    readout_origin: str,
    reverse_readout: bool,
) -> np.ndarray:
    spokes = dirs_xyz.shape[0]
    s = _build_readout_s(true_ro, NX, traj_scale, readout_origin, reverse_readout)

    traj = np.zeros((3, true_ro, spokes), dtype=np.complex64)
    dx, dy, dz = dirs_xyz[:, 0], dirs_xyz[:, 1], dirs_xyz[:, 2]
    for i in range(spokes):
        traj[0, :, i] = s * dx[i]
        traj[1, :, i] = s * dy[i]
        traj[2, :, i] = s * dz[i]

    print(
        f"[info] Traj built with max |k| ≈ {np.abs(traj).max():.2f} "
        f"(origin={readout_origin}, reverse={reverse_readout})"
    )
    return traj


# ---------------- traj file support (detour) ---------------- #

def _normalize_traj_shape_to_3_ro_spokes(
    arr: np.ndarray,
    true_ro: int,
    spokes_all: int,
) -> np.ndarray:
    """
    Accept many plausible traj shapes and normalize to (3, RO, spokes) complex64.
    """
    a = np.asarray(arr)

    # If complex, keep; if real, convert to complex (BART wants complex for CFL)
    if not np.iscomplexobj(a):
        a = a.astype(np.float32)

    # Drop singleton axes
    a = np.squeeze(a)

    # Common cases:
    #  - (3, RO, spokes)
    #  - (RO, spokes, 3)
    #  - (3, spokes, RO)
    #  - (spokes, 3) directions only
    #  - (3, spokes) directions only

    if a.ndim == 3:
        if a.shape == (3, true_ro, spokes_all):
            out = a
        elif a.shape == (true_ro, spokes_all, 3):
            out = np.transpose(a, (2, 0, 1))
        elif a.shape == (3, spokes_all, true_ro):
            out = np.transpose(a, (0, 2, 1))
        else:
            raise ValueError(f"Unrecognized traj 3D shape {a.shape}, expected variants of (3,{true_ro},{spokes_all})")
        return np.asfortranarray(out.astype(np.complex64))

    if a.ndim == 2:
        if a.shape == (spokes_all, 3):
            # directions only (spokes,3) => expand later
            return a
        if a.shape == (3, spokes_all):
            return a.T
        raise ValueError(f"Unrecognized traj 2D shape {a.shape}, expected ({spokes_all},3) directions")

    raise ValueError(f"Unrecognized traj ndim={a.ndim} shape={a.shape}")


def find_traj_candidates(series_path: Path) -> List[Path]:
    """
    Heuristic search for trajectory candidates under series/ and series/pdata/.
    We include both BART-style base names and raw files.
    """
    patterns = [
        "*traj*.hdr",
        "*traj*.cfl",
        "*trajectory*.hdr",
        "*trajectory*.cfl",
        "*traj*",
        "*trajectory*",
    ]

    roots = [series_path]
    pdata = series_path / "pdata"
    if pdata.exists():
        roots.append(pdata)

    cands: List[Path] = []
    for root in roots:
        for pat in patterns:
            for p in root.rglob(pat):
                if not p.is_file():
                    continue
                # exclude obvious non-traj heavy hitters
                if p.name in ("fid", "2dseq", "rawdata.job0"):
                    continue
                cands.append(p)

    # de-duplicate and sort by size asc (small first, but we'll validate)
    uniq = sorted(set(cands), key=lambda x: (x.stat().st_size, str(x)))
    return uniq


def read_traj_from_file_or_base(path_or_base: Path) -> np.ndarray:
    """
    If given a BART base (or .cfl/.hdr), read via readcfl.
    Else return raw bytes decoded as float arrays by the caller.
    """
    p = path_or_base
    if p.suffix in (".cfl", ".hdr"):
        base = p.with_suffix("")
        return readcfl(str(base))
    # base given directly?
    if p.with_suffix(".cfl").exists() and p.with_suffix(".hdr").exists():
        return readcfl(str(p))
    raise FileNotFoundError(f"Not a BART CFL base or .cfl/.hdr: {p}")


def try_parse_raw_binary_traj(
    traj_path: Path,
    true_ro: int,
    spokes_all: int,
) -> Optional[np.ndarray]:
    """
    Attempt raw binary parsing as float32/float64:
      - interleaved xyz per sample (RO*spokes,3)
      - blocked x then y then z (each RO*spokes)
    Returns traj (3,RO,spokes) complex64 or None.
    """
    raw = traj_path.read_bytes()
    nbytes = len(raw)
    target32 = 3 * true_ro * spokes_all * 4
    target64 = 3 * true_ro * spokes_all * 8

    def _build_from_arr(arr: np.ndarray, mode: str) -> np.ndarray:
        if mode == "interleaved_xyz":
            xyz = arr.reshape(true_ro * spokes_all, 3)
            x = xyz[:, 0].reshape(true_ro, spokes_all)
            y = xyz[:, 1].reshape(true_ro, spokes_all)
            z = xyz[:, 2].reshape(true_ro, spokes_all)
        elif mode == "blocked_x_y_z":
            n = true_ro * spokes_all
            x = arr[0:n].reshape(true_ro, spokes_all)
            y = arr[n:2*n].reshape(true_ro, spokes_all)
            z = arr[2*n:3*n].reshape(true_ro, spokes_all)
        else:
            raise ValueError("bad mode")

        traj = np.zeros((3, true_ro, spokes_all), dtype=np.complex64)
        traj[0] = x
        traj[1] = y
        traj[2] = z
        return np.asfortranarray(traj)

    for dt, tgt in ((np.float32, target32), (np.float64, target64)):
        if nbytes != tgt:
            continue
        arr = np.frombuffer(raw, dtype=dt)
        for mode in ("interleaved_xyz", "blocked_x_y_z"):
            try:
                traj = _build_from_arr(arr, mode)
                print(f"[info] Parsed raw binary traj {traj_path} as {dt} ({mode}), shape={traj.shape}")
                return traj
            except Exception:
                pass

    return None


def load_traj_auto(
    series_path: Path,
    true_ro: int,
    spokes_all: int,
    NX: int,
    traj_scale: float | None,
    spoke_order: str,
    readout_origin: str,
    reverse_readout: bool,
    traj_source: str,
    traj_file: Optional[Path],
) -> Tuple[np.ndarray, str]:
    """
    Return traj_full (3,RO,spokes) and a short string describing the source used.
    """
    # Helper for gradoutput
    def _from_gradoutput() -> np.ndarray:
        grad_path = series_path / "grad.output"
        if not grad_path.exists():
            raise RuntimeError(f"grad.output not found in {series_path}")
        dirs = load_grad_output_dirs(grad_path, normalize=True)
        dirs_full = expand_spoke_dirs(dirs, spokes_all, spoke_order)
        return build_traj_from_dirs(true_ro, dirs_full, NX, traj_scale, readout_origin, reverse_readout)

    # Helper for trajfile
    def _from_trajfile(p: Path) -> np.ndarray:
        # 1) If BART CFL base: read and normalize dims
        try:
            arr = read_traj_from_file_or_base(p)
            normed = _normalize_traj_shape_to_3_ro_spokes(arr, true_ro, spokes_all)
            if normed.ndim == 2 and normed.shape == (spokes_all, 3):
                # directions only -> expand
                s = _build_readout_s(true_ro, NX, traj_scale, readout_origin, reverse_readout)
                traj = np.zeros((3, true_ro, spokes_all), dtype=np.complex64)
                for i in range(spokes_all):
                    traj[0, :, i] = s * normed[i, 0]
                    traj[1, :, i] = s * normed[i, 1]
                    traj[2, :, i] = s * normed[i, 2]
                print(f"[info] Expanded direction-only traj to full RO using kmax=0.5*NX (NX={NX})")
                return traj
            return normed
        except FileNotFoundError:
            pass
        except Exception as e:
            # if it's a CFL but wrong dims, we want to report and fall through to raw attempt
            print(f"[warn] Failed to parse as BART CFL traj: {p} ({e})", file=sys.stderr)

        # 2) Try raw binary float32/64
        t = try_parse_raw_binary_traj(p, true_ro, spokes_all)
        if t is not None:
            return t

        raise RuntimeError(
            f"Could not parse trajectory file {p}.\n"
            f"  Tried: BART .cfl/.hdr, raw float32/float64 binary (3*RO*spokes).\n"
            f"  You may need to point --traj-file at the correct file/base."
        )

    # Resolve explicit override first
    if traj_file is not None:
        traj = _from_trajfile(traj_file)
        return traj, f"trajfile:{traj_file}"

    if traj_source == "gradoutput":
        return _from_gradoutput(), "gradoutput"

    if traj_source in ("trajfile", "auto"):
        # search for candidates
        cands = find_traj_candidates(series_path)
        if traj_source == "trajfile" and not cands:
            raise RuntimeError(f"--traj-source trajfile requested, but no traj candidate found under {series_path} or pdata/*")

        # Try candidates until one parses
        for p in cands:
            # Prefer bases (hdr/cfl) when seen
            base = p.with_suffix("") if p.suffix in (".cfl", ".hdr") else p
            try:
                traj = _from_trajfile(base)
                return traj, f"trajfile:auto:{base}"
            except Exception:
                continue

        if traj_source == "trajfile":
            raise RuntimeError(f"--traj-source trajfile requested, but no candidate trajectory could be parsed under {series_path} or pdata/*")

        # auto fallback
        return _from_gradoutput(), "gradoutput(fallback)"

    raise ValueError(f"Unknown traj-source: {traj_source}")


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


# ---------------- Core recon ---------------- #

def run_bart(
    series_path: Path,
    out_base: Path,
    NX: int, NY: int, NZ: int,
    true_ro: int,
    ksp: np.ndarray,                 # (ro, spokes, coils)
    spokes_all: int,
    spokes_per_frame: int,
    frame_shift: int,
    qa_first: int | None,
    export_nifti: bool,
    traj_scale: float | None,
    use_gpu: bool,
    spoke_order: str,
    readout_origin: str,
    reverse_readout: bool,
    traj_source: str,
    traj_file: Optional[Path],
):
    bart_bin = "bart"

    traj_full, traj_used = load_traj_auto(
        series_path=series_path,
        true_ro=true_ro,
        spokes_all=spokes_all,
        NX=NX,
        traj_scale=traj_scale,
        spoke_order=spoke_order,
        readout_origin=readout_origin,
        reverse_readout=reverse_readout,
        traj_source=traj_source,
        traj_file=traj_file,
    )
    print(f"[info] Trajectory source used: {traj_used}")

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
        ksp_base  = out_dir / f"{out_base.name}_{tag}_ksp"
        coil_base = out_dir / f"{out_base.name}_{tag}_coil"
        sos_base  = out_dir / f"{out_base.name}_{tag}"

        frame_paths.append(sos_base)

        if sos_base.with_suffix(".cfl").exists():
            print(f"[info] Frame {i} already reconstructed -> {sos_base}, skipping.")
            continue

        print(f"[info] Frame {i} spokes [{start}:{stop}] (n={nsp})")

        ksp_frame = ksp[:, start:stop, :]         # (ro, spokes, coils)
        traj_frame = traj_full[:, :, start:stop]  # (3, ro, spokes)

        # BART NUFFT expects k-space with RO in dim0 and spokes in dim1.
        # Put a singleton dim2 (slice) and coils in dim3:
        #   (RO, spokes, 1, coils)
        ksp_bart = ksp_frame[:, :, np.newaxis, :]  # (ro, spokes, 1, coils)

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
            dims_ksp  = bart_image_dims(bart_bin, ksp_base)
            dims_traj = bart_image_dims(bart_bin, traj_base)
            dims_coil = bart_image_dims(bart_bin, coil_base)
            dims_sos  = bart_image_dims(bart_bin, sos_base)
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

def _validate_out_path(out_base: Path):
    s = str(out_base)
    if " --" in s or "\n" in s or "\t" in s:
        raise SystemExit(
            "[fatal] Your --out contains what looks like an embedded flag (e.g. ' --reverse-readout').\n"
            "        You likely did: --out \"... --reverse-readout\" by accident.\n"
            "        Fix: move --reverse-readout outside the --out quotes.\n"
            f"        Got --out: {s}"
        )
    if " " in s:
        raise SystemExit(
            "[fatal] Your --out contains spaces. That will break path tokenization and BART commands.\n"
            "        Use underscores instead, or quote carefully. Recommended: no spaces.\n"
            f"        Got --out: {s}"
        )


def main():
    ap = argparse.ArgumentParser(
        description="Bruker 3D radial → BART NUFFT recon driver (grad.output or traj file).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:

              # Default (auto): prefer a traj file if found; else use grad.output
              python bruker_radial_bart.py --series /path/to/29 --export-nifti --out /tmp/outprefix

              # Force using dataset traj file (error if none parsable)
              python bruker_radial_bart.py --series /path/to/29 --traj-source trajfile --export-nifti --out /tmp/outprefix

              # Force grad.output
              python bruker_radial_bart.py --series /path/to/29 --traj-source gradoutput --export-nifti --out /tmp/outprefix

              # Force a specific traj file/base
              python bruker_radial_bart.py --series /path/to/29 --traj-source trajfile --traj-file /path/to/traj_base --export-nifti --out /tmp/outprefix

            Notes:
              - grad.output is still automatic when used; no --grad-output option.
              - If your traj file is direction-only, we expand along RO using kmax=0.5*NX.
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

    ap.add_argument("--fid-dtype", choices=["i4", "f4"], default="i4")
    ap.add_argument("--fid-endian", choices=[">", "<"], default="<")
    ap.add_argument("--fid-layout", choices=["ro_spokes_coils", "ro_coils_spokes"], default="ro_spokes_coils")

    ap.add_argument("--spoke-order", choices=["tile", "repeat"], default="tile")
    ap.add_argument("--readout-origin", choices=["centered", "zero"], default="centered")
    ap.add_argument("--reverse-readout", action="store_true")
    ap.add_argument("--traj-scale", type=float, default=None)

    ap.add_argument(
        "--traj-source",
        choices=["auto", "trajfile", "gradoutput"],
        default="auto",
        help="Trajectory source: auto (prefer traj file), trajfile (require traj file), gradoutput (require grad.output).",
    )
    ap.add_argument(
        "--traj-file",
        default=None,
        help="Optional explicit trajectory file/base to use (BART base or raw binary).",
    )

    ap.add_argument("--qa-first", type=int, default=0)
    ap.add_argument("--export-nifti", action="store_true")

    ap.add_argument("--gpu", action="store_true")

    args = ap.parse_args()

    series_path = Path(args.series).resolve()
    out_base = Path(args.out).resolve()
    _validate_out_path(out_base)

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

    ksp, stored_ro, spokes_all, coils = load_bruker_kspace(
        fid_path=fid,
        true_ro=true_ro,
        coils_hint=coils_hint,
        endian=args.fid_endian,
        base_kind=args.fid_dtype,
        fid_layout=args.fid_layout,
    )

    spf = spokes_all if args.spokes_per_frame <= 0 else args.spokes_per_frame
    shift = spf if args.frame_shift <= 0 else args.frame_shift
    qa_first = args.qa_first if args.qa_first > 0 else None

    traj_file = Path(args.traj_file).resolve() if args.traj_file else None

    run_bart(
        series_path=series_path,
        out_base=out_base,
        NX=NX, NY=NY, NZ=NZ,
        true_ro=true_ro,
        ksp=ksp,
        spokes_all=spokes_all,
        spokes_per_frame=spf,
        frame_shift=shift,
        qa_first=qa_first,
        export_nifti=args.export_nifti,
        traj_scale=args.traj_scale,
        use_gpu=args.gpu,
        spoke_order=args.spoke_order,
        readout_origin=args.readout_origin,
        reverse_readout=args.reverse_readout,
        traj_source=args.traj_source,
        traj_file=traj_file,
    )


if __name__ == "__main__":
    main()
