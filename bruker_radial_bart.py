#!/usr/bin/env python3
"""
Bruker 3D radial -> BART NUFFT recon driver.

Key features:
- Reads Bruker method/acqp/fid
- Factors FID into (stored_ro, spokes, coils) using heuristic
- AUTO trajectory:
    * If <series>/grad.output exists, uses ProjR/ProjP/ProjS as spoke directions
      (and NORMALIZES to unit length)
    * Otherwise falls back to synthetic "kron" directions
- Sliding-window recon (BART nufft) + coil combine (BART rss) per frame
- Skips already-reconstructed frames
- Writes:
    * QA NIfTI of first N frames (nibabel)
    * Final 4D NIfTI stack (nibabel) if requested
- Prints BART dims (via bart show -d <dim>) for vol0 coil and SoS

Notes:
- BART rss takes a BITMASK of dims. Coil dim is 3 => mask = (1<<3) = 8.
- Two likely FID layouts supported:
    --fid-layout ro_spokes_coils  (default)
    --fid-layout ro_coils_spokes  ("test 2")
- For grad.output: many systems store ProjR/P/S as fixed-point scaled by 2^30.

Assumes:
    bart is on PATH as "bart"
"""

import argparse
import sys
import subprocess
import textwrap
import re
from pathlib import Path

import numpy as np
import nibabel as nib


# ---------------- Bruker helpers ---------------- #

def read_bruker_param(path: Path, key: str, default=None):
    """
    Tiny Bruker text parser for method/acqp.

    Returns:
      - scalar (int/float/str) if a single value
      - list if multiple
      - default if not found
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

        # Multi-line / array style
        if rhs.startswith("("):
            vals = []
            j = i + 1
            while j < len(lines):
                l2 = lines[j].strip()
                if l2.startswith("##"):
                    break
                if l2 == "" or l2.startswith("$$"):
                    j += 1
                    continue
                vals.extend(l2.split())
                j += 1

            parsed = []
            for v in vals:
                try:
                    if "." in v or "e" in v.lower():
                        parsed.append(float(v))
                    else:
                        parsed.append(int(v))
                except ValueError:
                    parsed.append(v)

            return parsed[0] if len(parsed) == 1 else parsed

        # Single-line values or tuples on one line
        rhs = rhs.strip().strip("()")
        tokens = [t for t in rhs.split() if t]

        parsed = []
        for v in tokens:
            try:
                if "." in v or "e" in v.lower():
                    parsed.append(float(v))
                else:
                    parsed.append(int(v))
            except ValueError:
                parsed.append(v)

        return parsed[0] if len(parsed) == 1 else parsed

    return default


def infer_matrix(method: Path):
    mat = read_bruker_param(method, "PVM_Matrix", default=None)
    if mat is None:
        raise ValueError("Could not infer PVM_Matrix")
    if isinstance(mat, (int, float)) or len(mat) != 3:
        raise ValueError(f"Unexpected PVM_Matrix: {mat}")
    NX, NY, NZ = map(int, mat)
    return NX, NY, NZ


def infer_true_ro(acqp: Path) -> int:
    acq_size = read_bruker_param(acqp, "ACQ_size", default=None)
    if acq_size is None:
        raise ValueError("Could not infer ACQ_size for true RO")

    if isinstance(acq_size, (int, float)):
        ro = int(acq_size)
    else:
        ro = int(acq_size[0])
    return ro


def infer_coils(method: Path) -> int:
    coils = read_bruker_param(method, "PVM_EncNReceivers", default=None)
    if coils is None:
        return 1
    if isinstance(coils, (list, tuple)):
        coils = int(coils[0])
    return int(coils)


# ---------------- FID / k-space loading ---------------- #

def factor_fid(total_points: int, true_ro: int, coils_hint: int | None = None):
    """
    Factor total complex points into (stored_ro, coils, spokes).

    Heuristic: try several stored_ro candidates and coil counts.
    """
    block_candidates = [true_ro]
    for b in (128, 96, 64, 256, 192, 384, 512):
        if b != true_ro:
            block_candidates.append(b)

    best = None
    best_score = -1

    if coils_hint is not None:
        coil_candidates = [coils_hint] + [c for c in range(1, 33) if c != coils_hint]
    else:
        coil_candidates = list(range(1, 33))

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
    coils_hint: int | None = None,
    endian: str = "<",
    base_kind: str = "i4",
    fid_layout: str = "ro_spokes_coils",
):
    """
    Load Bruker FID as complex data and reshape to (true_ro, spokes, coils).

    fid_layout:
      - ro_spokes_coils: complex reshape (stored_ro, spokes, coils)
      - ro_coils_spokes: complex reshape (stored_ro, coils, spokes) then transpose -> (stored_ro, spokes, coils)
    """
    if not fid_path.exists():
        raise FileNotFoundError(f"FID not found: {fid_path}")

    np_dtype = np.dtype(endian + base_kind)
    raw = np.fromfile(fid_path, dtype=np_dtype).astype(np.float32)

    if raw.size % 2 != 0:
        raise ValueError(f"FID raw length {raw.size} not even (real/imag pairs).")

    cplx = raw[0::2] + 1j * raw[1::2]
    total_points = cplx.size

    stored_ro, coils, spokes = factor_fid(total_points, true_ro, coils_hint)

    if coils_hint is not None and coils != coils_hint:
        print(
            f"[warn] FID factorization suggests coils={coils}, header said {coils_hint}; using coils={coils}.",
            file=sys.stderr,
        )

    print(f"[info] Loaded k-space with stored_ro={stored_ro}, true_ro={true_ro}, spokes={spokes}, coils={coils}")

    if fid_layout == "ro_spokes_coils":
        ksp = cplx.reshape(stored_ro, spokes, coils)
    elif fid_layout == "ro_coils_spokes":
        ksp = cplx.reshape(stored_ro, coils, spokes).transpose(0, 2, 1)
    else:
        raise ValueError(f"Unknown fid_layout: {fid_layout}")

    if stored_ro != true_ro:
        if stored_ro < true_ro:
            raise ValueError(f"stored_ro={stored_ro} < true_ro={true_ro}, cannot trim")
        ksp = ksp[:true_ro, :, :]
        print(f"[info] Trimmed k-space from stored_ro={stored_ro} to true_ro={true_ro}")

    return ksp, stored_ro, spokes, coils


# ---------------- BART CFL helpers ---------------- #

def writecfl(name: str, arr: np.ndarray):
    base = Path(name)
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")

    arr_f = np.asfortranarray(arr.astype(np.complex64))

    with hdr.open("w") as f:
        f.write("# Dimensions\n")
        dims = list(arr_f.shape)
        dims += [1] * (16 - len(dims))
        f.write(" ".join(str(d) for d in dims) + "\n")

    with cfl.open("wb") as f:
        stacked = np.empty(arr_f.size * 2, dtype=np.float32)
        stacked[0::2] = arr_f.real.ravel(order="F")
        stacked[1::2] = arr_f.imag.ravel(order="F")
        stacked.tofile(f)


def readcfl(name: str) -> np.ndarray:
    base = Path(name)
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")

    if not hdr.exists() or not cfl.exists():
        raise FileNotFoundError(f"CFL/HDR not found for base {base}")

    lines = hdr.read_text(errors="ignore").splitlines()
    if len(lines) < 2 or not lines[0].startswith("#"):
        raise ValueError(f"Malformed BART hdr for {base}")

    dims16 = [int(x) for x in lines[1].split()]
    # keep dims up to last non-1 (but at least 1 dim)
    last_non1 = 0
    for i, d in enumerate(dims16):
        if d > 1:
            last_non1 = i
    ndim = max(1, last_non1 + 1)
    dims = dims16[:ndim]

    data = np.fromfile(cfl, dtype=np.float32)
    if data.size % 2 != 0:
        raise ValueError(f"CFL data length {data.size} not even for {base}")

    cplx = data[0::2] + 1j * data[1::2]
    expected = int(np.prod(dims))
    if cplx.size != expected:
        raise ValueError(
            f"CFL size mismatch for {base}: have {cplx.size} complex, expected {expected} from dims {dims} (hdr {dims16})"
        )

    return cplx.reshape(dims, order="F")


def bart_image_dims(bart_bin: str, base: Path) -> list[int] | None:
    """
    Query full 16-D BART shape for a CFL base using:
        bart show -d <dim> <file>
    """
    dims: list[int] = []
    for d in range(16):
        proc = subprocess.run(
            [bart_bin, "show", "-d", str(d), str(base)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            print(
                f"[warn] 'bart show -d {d}' failed ({proc.returncode}): {proc.stderr.strip()}",
                file=sys.stderr,
            )
            return None
        try:
            dims.append(int(proc.stdout.strip()))
        except ValueError:
            print(f"[warn] Unexpected output from 'bart show -d {d}': {proc.stdout!r}", file=sys.stderr)
            return None
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
    if "compiled without gpu support" in s:
        return False
    if "unknown option" in s:
        return False
    return False


# ---------------- Trajectory from grad.output ---------------- #

def _extract_grad_block(lines: list[str], name: str) -> np.ndarray:
    """
    Extract a block like ProjR/ProjP/ProjS from Bruker grad.output.

    Expected header line format somewhere:
      "<idx>:ProjR: index = <n>, size = <size>"
    Followed by <size> lines typically like:
      "  0   1073741824"
      "  1   -915356728"
    """
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
        raise ValueError(f"Could not find block header for {name} in grad.output")

    vals: list[int] = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        # stop at next block header / ramp sections
        if re.match(r"^\d+:\w+:", line) or line.startswith("Ramp Shape"):
            break
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            vals.append(int(parts[1]))
        except ValueError:
            continue
        if len(vals) >= size:
            break

    if len(vals) != size:
        raise ValueError(f"{name}: expected {size} values, got {len(vals)}")

    return np.array(vals, dtype=np.float64)


def load_grad_output_dirs(grad_output_path: Path, normalize: bool = True) -> np.ndarray:
    """
    Load spoke directions from grad.output ProjR/ProjP/ProjS, scaled by 2^30.

    Returns:
      dirs: (N,3) float64
    """
    lines = grad_output_path.read_text(errors="ignore").splitlines()

    r = _extract_grad_block(lines, "ProjR")
    p = _extract_grad_block(lines, "ProjP")
    s = _extract_grad_block(lines, "ProjS")

    scale = float(1 << 30)  # 2^30
    dirs = np.stack([r, p, s], axis=1) / scale

    norms = np.linalg.norm(dirs, axis=1)
    print(f"[info] Parsed {dirs.shape[0]} spoke directions from {grad_output_path}")
    print(f"[info] Direction norms BEFORE normalize: min={norms.min():.4f} median={np.median(norms):.4f} max={norms.max():.4f}")

    if normalize:
        dirs = dirs / norms[:, None]
        norms2 = np.linalg.norm(dirs, axis=1)
        print(f"[info] Direction norms AFTER  normalize: min={norms2.min():.4f} median={np.median(norms2):.4f} max={norms2.max():.4f}")

    return dirs


def expand_spoke_dirs(dirs: np.ndarray, target_spokes: int, order: str) -> np.ndarray:
    """
    Expand (N,3) dirs to (target_spokes,3) when target_spokes is an integer multiple of N.

    order:
      - tile:   [dirs, dirs, dirs, ...] (block repetition)
      - repeat: [d0,d0,d0, d1,d1,d1, ...] (each dir repeated)
    """
    n = dirs.shape[0]
    if target_spokes == n:
        return dirs
    if target_spokes % n != 0:
        raise ValueError(f"Cannot expand dirs length {n} to target_spokes {target_spokes} (not divisible).")
    reps = target_spokes // n
    print(f"[info] Expanding {n} dirs to {target_spokes} spokes with reps={reps} using order='{order}'")
    if order == "tile":
        return np.tile(dirs, (reps, 1))
    if order == "repeat":
        return np.repeat(dirs, reps, axis=0)
    raise ValueError(f"Unknown spoke expansion order: {order}")


def build_traj_from_dirs(
    true_ro: int,
    dirs_xyz: np.ndarray,   # (spokes, 3)
    NX: int,
    traj_scale: float | None,
    readout_origin: str,
    reverse_readout: bool,
) -> np.ndarray:
    """
    Build BART trajectory using provided per-spoke direction cosines.
    Output: (3, RO, spokes) complex64
    """
    spokes = dirs_xyz.shape[0]

    kmax = 0.5 * NX
    if traj_scale is not None:
        kmax *= float(traj_scale)

    if readout_origin == "zero":
        s = np.linspace(0.0, kmax, true_ro, dtype=np.float64)
    else:
        s = np.linspace(-kmax, kmax, true_ro, dtype=np.float64)

    if reverse_readout:
        s = s[::-1].copy()

    traj = np.zeros((3, true_ro, spokes), dtype=np.complex64)
    dx = dirs_xyz[:, 0]
    dy = dirs_xyz[:, 1]
    dz = dirs_xyz[:, 2]

    for i in range(spokes):
        traj[0, :, i] = s * dx[i]
        traj[1, :, i] = s * dy[i]
        traj[2, :, i] = s * dz[i]

    max_rad = np.abs(traj).max()
    print(f"[info] Traj built with max |k| ≈ {max_rad:.2f} (origin={readout_origin}, reverse={reverse_readout})")
    return traj


# ---------------- Synthetic trajectory fallback ---------------- #

def build_kron_traj(
    true_ro: int,
    spokes: int,
    NX: int,
    traj_scale: float | None,
    readout_origin: str,
    reverse_readout: bool,
) -> np.ndarray:
    """
    Synthetic "kron-ish" fallback. Use only if grad.output is missing.
    """
    idx = np.arange(spokes, dtype=np.float64)

    if spokes > 1:
        z = 2.0 * (idx / (spokes - 1)) - 1.0
    else:
        z = np.zeros_like(idx)

    phi = np.pi * (1 + 5**0.5) * idx
    r_xy = np.sqrt(np.clip(1.0 - z**2, 0.0, 1.0))
    dx = r_xy * np.cos(phi)
    dy = r_xy * np.sin(phi)
    dz = z

    kmax = 0.5 * NX
    if traj_scale is not None:
        kmax *= float(traj_scale)

    if readout_origin == "zero":
        s = np.linspace(0.0, kmax, true_ro, dtype=np.float64)
    else:
        s = np.linspace(-kmax, kmax, true_ro, dtype=np.float64)

    if reverse_readout:
        s = s[::-1].copy()

    traj = np.zeros((3, true_ro, spokes), dtype=np.complex64)
    for i in range(spokes):
        traj[0, :, i] = s * dx[i]
        traj[1, :, i] = s * dy[i]
        traj[2, :, i] = s * dz[i]

    max_rad = np.abs(traj).max()
    print(f"[info] Traj built with max |k| ≈ {max_rad:.2f} (synthetic fallback; origin={readout_origin}, reverse={reverse_readout})")
    return traj


# ---------------- NIfTI writers ---------------- #

def bart_cfl_to_nifti(base: Path, out_nii_gz: Path, assume_abs: bool = True):
    arr = readcfl(str(base))
    if assume_abs:
        arr = np.abs(arr)
    img = nib.Nifti1Image(arr.astype(np.float32), np.eye(4))
    nib.save(img, str(out_nii_gz))


def write_qa_nifti(qa_frames: list[Path], qa_base: Path):
    qa_base.parent.mkdir(parents=True, exist_ok=True)

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
    qa_img = nib.Nifti1Image(qa_stack.astype(np.float32), np.eye(4))
    qa_nii_gz = qa_base.with_suffix(".nii.gz")
    nib.save(qa_img, str(qa_nii_gz))
    print(f"[info] Wrote QA NIfTI {qa_nii_gz} with shape {qa_stack.shape}")


# ---------------- Core recon ---------------- #

def run_bart(
    series_path: Path,
    out_base: Path,
    NX: int,
    NY: int,
    NZ: int,
    true_ro: int,
    ksp: np.ndarray,
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
):
    bart_bin = "bart"

    # Trajectory selection: grad.output if present, else synthetic fallback
    grad_path = series_path / "grad.output"
    if grad_path.exists():
        dirs = load_grad_output_dirs(grad_path, normalize=True)
        dirs_full = expand_spoke_dirs(dirs, spokes_all, spoke_order)
        traj_full = build_traj_from_dirs(
            true_ro=true_ro,
            dirs_xyz=dirs_full,
            NX=NX,
            traj_scale=traj_scale,
            readout_origin=readout_origin,
            reverse_readout=reverse_readout,
        )
    else:
        print(f"[warn] grad.output not found under {series_path}; using synthetic fallback.", file=sys.stderr)
        traj_full = build_kron_traj(
            true_ro=true_ro,
            spokes=spokes_all,
            NX=NX,
            traj_scale=traj_scale,
            readout_origin=readout_origin,
            reverse_readout=reverse_readout,
        )

    # GPU probe
    have_gpu = False
    if use_gpu:
        have_gpu = bart_supports_gpu(bart_bin)
        if not have_gpu:
            print("[warn] BART has no GPU support; falling back to CPU.", file=sys.stderr)

    # Sliding window
    if spokes_per_frame <= 0:
        spokes_per_frame = spokes_all
    if frame_shift <= 0:
        frame_shift = spokes_per_frame

    frame_starts = list(range(0, max(1, spokes_all - spokes_per_frame + 1), frame_shift))
    n_frames = len(frame_starts)

    print(f"[info] Sliding window: spokes-per-frame={spokes_per_frame}, frame-shift={frame_shift}")
    print(f"[info] Will reconstruct {n_frames} frame(s).")

    per_series_dir = out_base.parent
    per_series_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    qa_written = False
    first_dims_reported = False

    for i, start in enumerate(frame_starts):
        stop = min(spokes_all, start + spokes_per_frame)
        frame_spokes = stop - start

        tag = f"vol{i:05d}"
        traj_base = per_series_dir / f"{out_base.name}_{tag}_traj"
        ksp_base = per_series_dir / f"{out_base.name}_{tag}_ksp"
        coil_base = per_series_dir / f"{out_base.name}_{tag}_coil"
        sos_base = per_series_dir / f"{out_base.name}_{tag}"

        frame_paths.append(sos_base)

        if sos_base.with_suffix(".cfl").exists():
            print(f"[info] Frame {i} already reconstructed -> {sos_base}, skipping.")
        else:
            print(f"[info] Frame {i} spokes [{start}:{stop}] (n={frame_spokes})")

            ksp_frame = ksp[:, start:stop, :]          # (ro, spokes_frame, coils)
            traj_frame = traj_full[:, :, start:stop]   # (3, ro, spokes_frame)

            # BART expects k-space shaped like (1, ro, spokes, coils) in our writecfl
            ksp_bart = ksp_frame[np.newaxis, ...]

            writecfl(str(traj_base), traj_frame)
            writecfl(str(ksp_base), ksp_bart)

            cmd = [bart_bin, "nufft", "-i", "-d", f"{NX}:{NY}:{NZ}"]
            if use_gpu and have_gpu:
                cmd.insert(2, "-g")
            cmd += [str(traj_base), str(ksp_base), str(coil_base)]
            print("[bart]", " ".join(cmd))
            subprocess.run(cmd, check=True)

            # Coil combine: rss bitmask for dim=3 => 8
            rss_mask = str(1 << 3)
            cmd2 = [bart_bin, "rss", rss_mask, str(coil_base), str(sos_base)]
            print("[bart]", " ".join(cmd2))
            subprocess.run(cmd2, check=True)

        if not first_dims_reported:
            dims_coil = bart_image_dims(bart_bin, coil_base)
            dims_sos = bart_image_dims(bart_bin, sos_base)
            if dims_coil is not None:
                print(f"[debug] BART dims coil vol0: {dims_coil}")
            if dims_sos is not None:
                print(f"[debug] BART dims SoS  vol0: {dims_sos}")
            first_dims_reported = True

        if qa_first is not None and not qa_written and len(frame_paths) >= qa_first:
            qa_frames = frame_paths[:qa_first]
            qa_base = per_series_dir / f"{out_base.name}_QA_first{qa_first}"
            write_qa_nifti(qa_frames, qa_base)
            qa_written = True

    # Final 4D stack via bart join
    sos_existing = [p for p in frame_paths if p.with_suffix(".cfl").exists()]
    if not sos_existing:
        print("[warn] No per-frame SoS CFLs exist; skipping 4D stack.", file=sys.stderr)
        return

    stack_base = per_series_dir / out_base.name
    join_cmd = ["bart", "join", "3"] + [str(p) for p in sos_existing] + [str(stack_base)]
    print("[bart]", " ".join(join_cmd))
    subprocess.run(join_cmd, check=True)

    if export_nifti:
        stack_nii_gz = stack_base.with_suffix(".nii.gz")
        bart_cfl_to_nifti(stack_base, stack_nii_gz)
        print(f"[info] Wrote final 4D NIfTI {stack_nii_gz}")

    print(f"[info] All requested frames complete; 4D result at {stack_base}")


# ---------------- CLI ---------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Bruker 3D radial → BART NUFFT recon driver (auto grad.output).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Example:

              python bruker_radial_bart.py \
                --series /path/to/Bruker/29 \
                --spokes-per-frame 800 \
                --frame-shift 200 \
                --spoke-order repeat \
                --fid-layout ro_coils_spokes \
                --readout-origin zero \
                --qa-first 2 \
                --export-nifti \
                --out /path/to/outprefix

            grad.output:
              If /path/to/Bruker/29/grad.output exists, it will be used automatically.
            """
        ),
    )

    ap.add_argument("--series", required=True, help="Bruker 3D radial series directory")
    ap.add_argument("--out", required=True, help="Output base path (no extension)")

    ap.add_argument("--matrix", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                    help="Override PVM_Matrix")
    ap.add_argument("--readout", type=int, help="Override true RO length (ACQ_size[0])")
    ap.add_argument("--coils", type=int, help="Override number of coils (PVM_EncNReceivers)")

    ap.add_argument("--spokes-per-frame", type=int, default=0,
                    help="Spokes per frame (0 => all spokes)")
    ap.add_argument("--frame-shift", type=int, default=0,
                    help="Frame shift (0 => spokes-per-frame)")

    ap.add_argument("--fid-dtype", choices=["i4", "f4"], default="i4",
                    help="Base numeric type of FID data; combined with --fid-endian")
    ap.add_argument("--fid-endian", choices=[">", "<"], default="<",
                    help="FID endianness: '>' big-endian, '<' little-endian (default)")

    ap.add_argument("--fid-layout", choices=["ro_spokes_coils", "ro_coils_spokes"],
                    default="ro_spokes_coils",
                    help="Raw FID reshape layout (test2 = ro_coils_spokes)")

    ap.add_argument("--spoke-order", choices=["tile", "repeat"], default="tile",
                    help="If spokes == reps * N_dirs, how to expand dirs: tile or repeat")

    ap.add_argument("--readout-origin", choices=["centered", "zero"], default="centered",
                    help="Radial coordinate: centered (-k..+k) or zero (0..k)")
    ap.add_argument("--reverse-readout", action="store_true",
                    help="Reverse sample order along readout")

    ap.add_argument("--traj-scale", type=float, default=None,
                    help="Extra scale factor for k-space coordinates (try 0.5, 1.0, 2.0)")

    ap.add_argument("--qa-first", type=int, default=0,
                    help="Write QA NIfTI of first N frames (0 => disable)")
    ap.add_argument("--export-nifti", action="store_true",
                    help="Export final 4D stack as NIfTI (.nii.gz)")

    ap.add_argument("--gpu", action="store_true",
                    help="Try BART GPU NUFFT (-g) if supported")

    args = ap.parse_args()

    series_path = Path(args.series).resolve()
    if not series_path.exists():
        ap.error(f"Series path does not exist: {series_path}")

    out_base = Path(args.out).resolve()
    out_base.parent.mkdir(parents=True, exist_ok=True)

    method = series_path / "method"
    acqp = series_path / "acqp"
    fid = series_path / "fid"
    if not method.exists() or not acqp.exists() or not fid.exists():
        ap.error(f"Could not find method/acqp/fid under {series_path}")

    # Matrix
    if args.matrix is not None:
        NX, NY, NZ = map(int, args.matrix)
        print(f"[info] Matrix overridden from CLI: {NX}x{NY}x{NZ}")
    else:
        NX, NY, NZ = infer_matrix(method)
        print(f"[info] Matrix inferred from PVM_Matrix: {NX}x{NY}x{NZ}")

    # True RO
    if args.readout is not None:
        true_ro = int(args.readout)
        print(f"[info] Readout overridden from CLI: RO={true_ro}")
    else:
        true_ro = infer_true_ro(acqp)
        print(f"[info] Readout (true RO) from ACQ_size: RO={true_ro}")

    # Coils
    if args.coils is not None:
        coils_hint = int(args.coils)
        print(f"[info] Coils overridden from CLI: {coils_hint}")
    else:
        coils_hint = infer_coils(method)
        print(f"[info] Coils inferred from PVM_EncNReceivers: {coils_hint}")

    # Load k-space
    ksp, stored_ro, spokes_all, coils = load_bruker_kspace(
        fid_path=fid,
        true_ro=true_ro,
        coils_hint=coils_hint,
        endian=args.fid_endian,
        base_kind=args.fid_dtype,
        fid_layout=args.fid_layout,
    )

    spokes_per_frame = spokes_all if args.spokes_per_frame <= 0 else args.spokes_per_frame
    frame_shift = spokes_per_frame if args.frame_shift <= 0 else args.frame_shift
    qa_first = args.qa_first if args.qa_first > 0 else None

    run_bart(
        series_path=series_path,
        out_base=out_base,
        NX=NX,
        NY=NY,
        NZ=NZ,
        true_ro=true_ro,
        ksp=ksp,
        spokes_all=spokes_all,
        spokes_per_frame=spokes_per_frame,
        frame_shift=frame_shift,
        qa_first=qa_first,
        export_nifti=args.export_nifti,
        traj_scale=args.traj_scale,
        use_gpu=args.gpu,
        spoke_order=args.spoke_order,
        readout_origin=args.readout_origin,
        reverse_readout=args.reverse_readout,
    )


if __name__ == "__main__":
    main()
