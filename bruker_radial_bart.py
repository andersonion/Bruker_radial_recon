#!/usr/bin/env python3
"""
Bruker 3D radial -> BART NUFFT recon driver.

Fix in this version:
  - k-space passed to BART NUFFT is written as (RO, spokes, 1, coils),
    i.e. RO is dim0 and spokes is dim1 (BART convention),
    NOT (1, RO, spokes, coils).

Also:
  - Automatically uses <series>/grad.output if present (ProjR/ProjP/ProjS)
  - Normalizes direction vectors to unit length
  - Supports spoke expansion tile/repeat when spokes = reps * N_dirs
  - Supports FID layouts:
        ro_spokes_coils (default)
        ro_coils_spokes (test2)
  - Supports readout origin centered vs zero, and reverse readout
  - Uses correct BART rss bitmask for coil dim=3 => 8

NEW:
  - Optional traj-file mode for "known-good" datasets:
        --traj-source auto|trajfile|gradoutput
        --traj-file <path> (optional; default tries <series>/traj then pdata/**/traj)
        --traj-dtype f4|f8|i4|i2 (default f4)
        --traj-endian <|> (default <)
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
        f"[info] Loaded k-space with stored_ro={stored_ro}, true_ro={true_ro}, spokes={spokes}, coils={coils}"
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


def _build_readout_s(true_ro: int, NX: int, traj_scale: float | None, readout_origin: str, reverse_readout: bool) -> np.ndarray:
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

    print(f"[info] Traj built with max |k| ≈ {np.abs(traj).max():.2f} (origin={readout_origin}, reverse={reverse_readout})")
    return traj


# ---------------- traj file support (known-good detour datasets) ---------------- #

def find_traj_candidates(series_path: Path) -> List[Path]:
    """
    Prefer explicit Bruker traj locations first:
      - <series>/traj
      - <series>/pdata/*/traj
    Then fall back to globbing anything with traj in the name.
    """
    cands: List[Path] = []

    p0 = series_path / "traj"
    if p0.exists() and p0.is_file():
        cands.append(p0)

    pdata = series_path / "pdata"
    if pdata.exists():
        for p in pdata.rglob("traj"):
            if p.exists() and p.is_file():
                cands.append(p)

    # fallback search
    for p in series_path.rglob("*traj*"):
        if not p.is_file():
            continue
        if p.name in ("fid", "2dseq", "rawdata.job0"):
            continue
        if p not in cands:
            cands.append(p)

    # unique preserve order
    seen = set()
    out = []
    for p in cands:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)

    return out


def _dtype_from_flag(flag: str) -> np.dtype:
    m = {"f4": np.float32, "f8": np.float64, "i4": np.int32, "i2": np.int16}
    if flag not in m:
        raise ValueError(f"Unknown traj dtype flag {flag}")
    return np.dtype(m[flag])


def try_parse_bruker_traj_file(
    traj_path: Path,
    true_ro: int,
    spokes_all: int,
    dtype: np.dtype,
    endian: str,
) -> Optional[np.ndarray]:
    """
    Try common Bruker-style raw traj encodings with user-controllable dtype/endian.

    Successful output must be (3, RO, spokes) complex64.
    """
    dt = np.dtype(endian + dtype.str[1:])  # e.g. '<f4'
    data = np.fromfile(traj_path, dtype=dt)
    n = data.size
    nbytes = traj_path.stat().st_size

    # Hypotheses and their expected element counts:
    # full per-sample (3,RO,spokes) => 3*RO*spokes
    # direction-only (spokes,3) => 3*spokes
    expect_full = 3 * true_ro * spokes_all
    expect_dirs = 3 * spokes_all

    def as_traj_from_full(arr: np.ndarray, mode: str) -> np.ndarray:
        # Build (3,RO,spokes)
        if mode == "blocked_3_ro_spokes":
            t = arr.reshape(3, true_ro, spokes_all)
        elif mode == "ro_spokes_3":
            t = arr.reshape(true_ro, spokes_all, 3).transpose(2, 0, 1)
        elif mode == "3_spokes_ro":
            t = arr.reshape(3, spokes_all, true_ro).transpose(0, 2, 1)
        elif mode == "spokes_ro_3":
            t = arr.reshape(spokes_all, true_ro, 3).transpose(2, 1, 0)
        else:
            raise ValueError("bad mode")
        return np.asfortranarray(t.astype(np.complex64))

    def as_dirs(arr: np.ndarray, mode: str) -> np.ndarray:
        if mode == "spokes_3":
            d = arr.reshape(spokes_all, 3)
        elif mode == "3_spokes":
            d = arr.reshape(3, spokes_all).T
        else:
            raise ValueError("bad dirs mode")
        return d.astype(np.float64)

    # Try full traj shapes
    if n == expect_full:
        for mode in ("blocked_3_ro_spokes", "ro_spokes_3", "3_spokes_ro", "spokes_ro_3"):
            try:
                traj = as_traj_from_full(data, mode)
                print(f"[info] Parsed traj {traj_path} as raw {dt} mode={mode}, shape={traj.shape}")
                return traj
            except Exception:
                pass

    # Try direction-only
    if n == expect_dirs:
        for mode in ("spokes_3", "3_spokes"):
            try:
                d = as_dirs(data, mode)
                # normalize direction cosines just in case
                norms = np.linalg.norm(d, axis=1)
                bad = norms == 0
                if np.any(bad):
                    raise ValueError("zero-norm direction found")
                d = d / norms[:, None]
                print(f"[info] Parsed traj {traj_path} as direction-only raw {dt} mode={mode}, shape={d.shape}")
                # Caller will expand along RO using kmax model
                return d  # (spokes,3)
            except Exception:
                pass

    # If failed, print a quick diagnostic once (caller decides)
    exp_bytes_full = expect_full * dt.itemsize
    exp_bytes_dirs = expect_dirs * dt.itemsize
    print(
        f"[warn] Could not parse {traj_path} as raw traj with dt={dt}. "
        f"bytes={nbytes}. Expected bytes: full={exp_bytes_full}, dirs={exp_bytes_dirs}",
        file=sys.stderr,
    )
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
    traj_dtype_flag: str,
    traj_endian: str,
) -> Tuple[np.ndarray, str]:
    """
    Return traj_full (3,RO,spokes) and a short string describing the source used.
    """

    def _from_gradoutput() -> np.ndarray:
        grad_path = series_path / "grad.output"
        if not grad_path.exists():
            raise RuntimeError(f"grad.output not found in {series_path}")
        dirs = load_grad_output_dirs(grad_path, normalize=True)
        dirs_full = expand_spoke_dirs(dirs, spokes_all, spoke_order)
        return build_traj_from_dirs(true_ro, dirs_full, NX, traj_scale, readout_origin, reverse_readout)

    def _expand_dirs_to_traj(dirs_sp3: np.ndarray) -> np.ndarray:
        s = _build_readout_s(true_ro, NX, traj_scale, readout_origin, reverse_readout)
        traj = np.zeros((3, true_ro, spokes_all), dtype=np.complex64)
        for i in range(spokes_all):
            traj[0, :, i] = s * dirs_sp3[i, 0]
            traj[1, :, i] = s * dirs_sp3[i, 1]
            traj[2, :, i] = s * dirs_sp3[i, 2]
        return traj

    def _from_trajfile(p: Path) -> np.ndarray:
        dt = _dtype_from_flag(traj_dtype_flag)
        parsed = try_parse_bruker_traj_file(p, true_ro, spokes_all, dt, traj_endian)
        if parsed is None:
            raise RuntimeError(f"Failed to parse traj file: {p}")
        # parsed may be full traj or direction-only
        if parsed.ndim == 2 and parsed.shape == (spokes_all, 3):
            return _expand_dirs_to_traj(parsed)
        if parsed.ndim == 3 and parsed.shape == (3, true_ro, spokes_all):
            return parsed
        raise RuntimeError(f"Parsed traj has unexpected shape {parsed.shape} from {p}")

    # Explicit override first
    if traj_file is not None:
        if not traj_file.exists():
            raise RuntimeError(f"--traj-file does not exist: {traj_file}")
        return _from_trajfile(traj_file), f"trajfile:{traj_file}"

    if traj_source == "gradoutput":
        return _from_gradoutput(), "gradoutput"

    if traj_source in ("trajfile", "auto"):
        cands = find_traj_candidates(series_path)
        if traj_source == "trajfile" and not cands:
            raise RuntimeError(f"--traj-source trajfile requested, but no traj candidate found under {series_path} or pdata/*")

        last_err = None
        for p in cands:
            try:
                return _from_trajfile(p), f"trajfile:auto:{p}"
            except Exception as e:
                last_err = e
                continue

        if traj_source == "trajfile":
            raise RuntimeError(
                f"--traj-source trajfile requested, but no candidate trajectory could be parsed under {series_path} or pdata/*.\n"
                f"Last error: {last_err}"
            )

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
    traj_dtype_flag: str,
    traj_endian: str,
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
        traj_dtype_flag=traj_dtype_flag,
        traj_endian=traj_endian,
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

def main():
    ap = argparse.ArgumentParser(
        description="Bruker 3D radial → BART NUFFT recon driver (grad.output or traj).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Example (known-good traj dataset):
              python bruker_radial_bart.py \
                --series /path/to/8 \
                --traj-source trajfile \
                --traj-dtype f4 --traj-endian '<' \
                --export-nifti \
                --out /path/to/outprefix

            If parsing fails, try:
              --traj-dtype f8
              --traj-endian '>'
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

    ap.add_argument("--traj-source", choices=["auto", "trajfile", "gradoutput"], default="auto")
    ap.add_argument("--traj-file", default=None, help="Explicit traj file path (defaults to <series>/traj or pdata/**/traj)")
    ap.add_argument("--traj-dtype", choices=["f4", "f8", "i4", "i2"], default="f4", help="Raw traj dtype for Bruker traj file")
    ap.add_argument("--traj-endian", choices=["<", ">"], default="<", help="Raw traj endianness for Bruker traj file")

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
        traj_dtype_flag=args.traj_dtype,
        traj_endian=args.traj_endian,
    )


if __name__ == "__main__":
    main()
