#!/usr/bin/env python3
"""
Bruker 3D radial -> BART NUFFT recon driver.

- Reads Bruker method/acqp/fid
- Factors FID into (stored_ro, spokes, coils) using heuristic
- Builds 3D radial trajectory:
    * either synthetic kron (fallback)
    * or from Bruker grad.output ProjR/ProjP/ProjS (recommended if available)
- Runs BART NUFFT + SoS per-frame with sliding window
- Skips already-reconstructed frames (based on SoS .cfl presence)
- Writes:
    * QA NIfTI of first N frames  (via nibabel)
    * Final 4D NIfTI stack       (via nibabel)
- Prints BART dims for first coil & SoS volumes for debugging

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
        if line.startswith(token):
            parts = line.strip().split("=")
            if len(parts) < 2:
                continue

            rhs = parts[1].strip()

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

                if len(parsed) == 1:
                    return parsed[0]
                return parsed

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

            if len(parsed) == 1:
                return parsed[0]
            return parsed

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
        coils = 1
    if isinstance(coils, (list, tuple)):
        coils = int(coils[0])
    return int(coils)


def factor_fid(total_points: int, true_ro: int, coils_hint: int | None = None):
    """
    Factor total complex points into (stored_ro, spokes, coils).

    Heuristic: try candidates for stored_ro and coil counts, prefer:
      - stored_ro >= true_ro
      - stored_ro close to true_ro
      - spokes not tiny
      - match coil hint if possible
    """
    block_candidates = [true_ro]
    for b in (128, 96, 64, 384, 512, 256, 192):
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
            f"Could not factor FID into (stored_ro * spokes * coils). "
            f"total={total_points}, true_ro={true_ro}, coils_hint={coils_hint}"
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
    Load Bruker FID as complex data and reshape to (stored_ro, spokes, coils),
    then trim to (true_ro, spokes, coils).

    fid_layout:
      - ro_spokes_coils: raw reshape = (stored_ro, spokes, coils)
      - ro_coils_spokes: raw reshape = (stored_ro, coils, spokes) then transpose to (ro, spokes, coils)

    endian: '>' or '<'
    base_kind: 'i4' (int32) or 'f4' (float32)
    """
    if not fid_path.exists():
        raise FileNotFoundError(f"FID not found: {fid_path}")

    try:
        np_dtype = np.dtype(endian + base_kind)
    except TypeError as e:
        raise ValueError(
            f"Invalid dtype combination: endian={endian}, kind={base_kind}"
        ) from e

    raw = np.fromfile(fid_path, dtype=np_dtype).astype(np.float32)

    if raw.size % 2 != 0:
        raise ValueError(f"FID raw length {raw.size} not even (real/imag pairs).")

    complex_data = raw[0::2] + 1j * raw[1::2]
    total_points = complex_data.size

    stored_ro, coils, spokes = factor_fid(total_points, true_ro, coils_hint)

    if coils_hint is not None and coils != coils_hint:
        print(
            f"[warn] FID factorization suggests coils={coils}, "
            f"but header said {coils_hint}; using coils={coils}.",
            file=sys.stderr,
        )

    print(
        f"[info] Loaded k-space with stored_ro={stored_ro}, "
        f"true_ro={true_ro}, spokes={spokes}, coils={coils}"
    )

    if fid_layout == "ro_spokes_coils":
        ksp = complex_data.reshape(stored_ro, spokes, coils)
    elif fid_layout == "ro_coils_spokes":
        ksp = complex_data.reshape(stored_ro, coils, spokes).transpose(0, 2, 1)
    else:
        raise ValueError(f"Unknown fid_layout: {fid_layout}")

    if stored_ro != true_ro:
        if stored_ro < true_ro:
            raise ValueError(
                f"stored_ro={stored_ro} < true_ro={true_ro}, cannot trim."
            )
        ksp = ksp[:true_ro, :, :]
        print(f"[info] Trimmed k-space from stored_ro={stored_ro} to true_ro={true_ro}")

    return ksp, stored_ro, spokes, coils


# ---------------- BART CFL helpers ---------------- #


def writecfl(name: str, arr: np.ndarray):
    """
    Write BART .cfl/.hdr with dims following arr.shape in Fortran order.
    """
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
    """
    Read BART .cfl/.hdr into a numpy complex64 array (Fortran order-aware).

    Robust trimming: keep all dims up to the last non-1 dim (but at least 1 dim).
    This fixes failures when dim0==1 (leading singleton dims).
    """
    base = Path(name)
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")

    if not hdr.exists() or not cfl.exists():
        raise FileNotFoundError(f"CFL/HDR not found for base {base}")

    with hdr.open("r") as f:
        first = f.readline()
        if not first.startswith("#"):
            raise ValueError(f"Malformed BART hdr for {base}")
        dims_line = f.readline().strip()
        dims16 = [int(x) for x in dims_line.split()]

    last_non1 = 0
    for i, d in enumerate(dims16):
        if d > 1:
            last_non1 = i
    ndim = max(1, last_non1 + 1)
    dims = dims16[:ndim]

    data = np.fromfile(cfl, dtype=np.float32)
    if data.size % 2 != 0:
        raise ValueError(f"CFL data length {data.size} not even.")

    cplx = data[0::2] + 1j * data[1::2]
    expected = int(np.prod(dims))
    if cplx.size != expected:
        raise ValueError(
            f"CFL size mismatch for {base}: have {cplx.size} complex, "
            f"expected {expected} from dims {dims} (hdr line: {dims16})."
        )

    return cplx.reshape(dims, order="F")


def bart_image_dims(bart_bin: str, base: Path) -> list[int] | None:
    """
    Query full 16-D BART shape for a CFL base using:
        bart show -d <dim> <file>
    """
    dims: list[int] = []
    for d in range(16):
        try:
            proc = subprocess.run(
                [bart_bin, "show", "-d", str(d), str(base)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            print(f"[warn] Failed to run 'bart show -d {d}': {e}", file=sys.stderr)
            return None

        if proc.returncode != 0:
            print(
                f"[warn] 'bart show -d {d}' nonzero exit ({proc.returncode}): "
                f"{proc.stderr.strip()}",
                file=sys.stderr,
            )
            return None

        s = proc.stdout.strip()
        try:
            dims.append(int(s))
        except ValueError:
            print(
                f"[warn] Unexpected output from 'bart show -d {d}': {s!r}",
                file=sys.stderr,
            )
            return None

    return dims


def bart_supports_gpu(bart_bin: str = "bart") -> bool:
    """
    Quick probe: is 'bart nufft -g' a known option, and compiled with GPU support?
    """
    try:
        proc = subprocess.run(
            [bart_bin, "nufft", "-i", "-g"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return False

    if proc.returncode == 0:
        return True

    stderr = proc.stderr.lower()
    if "compiled without gpu support" in stderr:
        return False
    if "unknown option" in stderr:
        return False

    return False


# ---------------- Trajectory from grad.output ---------------- #


def parse_grad_output_proj_vectors(grad_output_path: Path) -> np.ndarray:
    """
    Parse Bruker-style grad.output and extract ProjR/ProjP/ProjS arrays.

    Returns:
        dirs: (N,3) float64 direction cosines, where:
              dirs[i] = [ProjR[i], ProjP[i], ProjS[i]] / 2^30
    """
    txt = grad_output_path.read_text(errors="ignore").splitlines()

    def _extract_block(name: str) -> list[int]:
        hdr_pat = re.compile(
            rf"^\s*\d+:{re.escape(name)}:\s+index\s*=\s*\d+,\s+size\s*=\s*(\d+)\s*$"
        )

        start = None
        size = None
        for i, line in enumerate(txt):
            m = hdr_pat.match(line)
            if m:
                start = i + 1
                size = int(m.group(1))
                break

        if start is None or size is None:
            raise ValueError(f"Could not find block header for {name} in {grad_output_path}")

        vals: list[int] = []
        for line in txt[start:]:
            line = line.strip()
            if not line:
                continue
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
            raise ValueError(f"{name}: expected {size} values, got {len(vals)} from {grad_output_path}")

        return vals

    r = np.array(_extract_block("ProjR"), dtype=np.float64)
    p = np.array(_extract_block("ProjP"), dtype=np.float64)
    s = np.array(_extract_block("ProjS"), dtype=np.float64)

    scale = float(1 << 30)  # 2^30 = 1073741824
    dirs = np.stack([r, p, s], axis=1) / scale

    norms = np.linalg.norm(dirs, axis=1)
    print(f"[info] Parsed {dirs.shape[0]} spoke directions from {grad_output_path}")
    print(f"[info] Direction norms: min={norms.min():.4f} median={np.median(norms):.4f} max={norms.max():.4f}")

    return dirs


def expand_spoke_dirs(dirs: np.ndarray, target_spokes: int, order: str) -> np.ndarray:
    """
    Expand an (N,3) direction list to (target_spokes,3) by either:
      - tile:   [dirs, dirs, dirs, ...] (block repetition)
      - repeat: [d0,d0,d0, d1,d1,d1, ...] (each dir repeated)
    """
    n = dirs.shape[0]
    if target_spokes == n:
        return dirs
    if target_spokes % n != 0:
        raise ValueError(
            f"Cannot expand dirs length {n} to target_spokes {target_spokes} (not divisible)."
        )

    reps = target_spokes // n
    print(f"[info] Expanding {n} dirs to {target_spokes} spokes with reps={reps} using order='{order}'")

    if order == "tile":
        return np.tile(dirs, (reps, 1))
    elif order == "repeat":
        return np.repeat(dirs, reps, axis=0)
    else:
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
    scale = float(traj_scale) if traj_scale is not None else 1.0
    kmax *= scale

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


# ---------------- Synthetic trajectory (fallback) ---------------- #


def build_kron_traj(
    true_ro: int,
    spokes: int,
    NX: int,
    NY: int,
    NZ: int,
    traj_scale: float | None = None,
    readout_origin: str = "centered",
    reverse_readout: bool = False,
) -> np.ndarray:
    """
    Synthetic 3D "kron" golden-angle trajectory for BART (fallback).
    Output shape: (3, RO, spokes)
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
    scale = float(traj_scale) if traj_scale is not None else 1.0
    kmax *= scale

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
    print(f"[info] Traj built with max |k| ≈ {max_rad:.2f} (synthetic; origin={readout_origin}, reverse={reverse_readout})")
    return traj


# ---------------- NIfTI writers (via nibabel) ---------------- #


def bart_cfl_to_nifti(
    base: Path,
    out_nii_gz: Path,
    assume_abs: bool = True,
):
    """
    Read a BART CFL and write a NIfTI via nibabel.

    NOTE: this does NOT attempt to interpret BART dimension semantics.
    It simply writes whatever array readcfl() returns.
    """
    arr = readcfl(str(base))
    if assume_abs:
        arr = np.abs(arr)
    img = nib.Nifti1Image(arr.astype(np.float32), np.eye(4))
    nib.save(img, str(out_nii_gz))


def write_qa_nifti(
    qa_frames: list[Path],
    qa_base: Path,
):
    """
    Stack first N SoS frames into a QA NIfTI using nibabel.
    """
    qa_base.parent.mkdir(parents=True, exist_ok=True)

    vols = []
    shape0 = None
    for p in qa_frames:
        arr = readcfl(str(p))
        mag = np.abs(arr)
        if shape0 is None:
            shape0 = mag.shape
        elif mag.shape != shape0:
            raise ValueError(
                f"QA frame {p} has shape {mag.shape}, expected {shape0}"
            )
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
    traj_mode: str,
    spokes_per_frame: int,
    frame_shift: int,
    combine: str,
    qa_first: int | None,
    export_nifti: bool,
    traj_scale: float | None,
    use_gpu: bool,
    debug: bool,
    grad_output_path: Path | None,
    spoke_order: str,
    readout_origin: str,
    reverse_readout: bool,
):
    bart_bin = "bart"

    ro, spokes_all, coils = ksp.shape

    if spokes_per_frame <= 0:
        raise ValueError("spokes-per-frame must be > 0")

    if frame_shift <= 0:
        frame_shift = spokes_per_frame

    frame_starts = list(
        range(0, max(1, spokes_all - spokes_per_frame + 1), frame_shift)
    )
    n_frames = len(frame_starts)

    print(
        f"[info] Sliding window: spokes-per-frame={spokes_per_frame}, "
        f"frame-shift={frame_shift}"
    )
    print(f"[info] Will reconstruct {n_frames} frame(s).")

    # ---- Build trajectory ---- #
    if grad_output_path is not None:
        dirs = parse_grad_output_proj_vectors(grad_output_path)
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
        if traj_mode != "kron":
            raise ValueError("Only traj-mode 'kron' is currently implemented.")
        traj_full = build_kron_traj(
            true_ro=true_ro,
            spokes=spokes_all,
            NX=NX,
            NY=NY,
            NZ=NZ,
            traj_scale=traj_scale,
            readout_origin=readout_origin,
            reverse_readout=reverse_readout,
        )

    # ---- GPU probe ---- #
    have_gpu = False
    if use_gpu:
        have_gpu = bart_supports_gpu(bart_bin)
        if not have_gpu:
            print(
                "[warn] BART compiled without GPU support; "
                "falling back to CPU NUFFT and disabling --gpu.",
                file=sys.stderr,
            )

    per_series_dir = out_base.parent
    per_series_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []

    qa_written = False
    first_dims_reported = False

    # ---- Per-frame NUFFT/RSS ---- #
    for i, start in enumerate(frame_starts):
        stop = start + spokes_per_frame
        if stop > spokes_all:
            stop = spokes_all
        frame_spokes = stop - start

        tag = f"vol{i:05d}"
        traj_base = per_series_dir / f"{out_base.name}_{tag}_traj"
        ksp_base = per_series_dir / f"{out_base.name}_{tag}_ksp"
        coil_base = per_series_dir / f"{out_base.name}_{tag}_coil"
        sos_base = per_series_dir / f"{out_base.name}_{tag}"

        frame_paths.append(sos_base)

        if sos_base.with_suffix(".cfl").exists():
            print(
                f"[info] Frame {i} already reconstructed -> "
                f"{sos_base}, skipping NUFFT/RSS."
            )
        else:
            print(f"[info] Frame {i} spokes [{start}:{stop}] (n={frame_spokes})")

            ksp_frame = ksp[:, start:stop, :]          # (ro, spokes_frame, coils)
            traj_frame = traj_full[:, :, start:stop]   # (3, ro, spokes_frame)

            # BART expects ksp dims: (1, ro, spokes, coils) for our usage here
            ksp_bart = ksp_frame[np.newaxis, ...]

            writecfl(str(traj_base), traj_frame)
            writecfl(str(ksp_base), ksp_bart)

            cmd = [bart_bin, "nufft", "-i", "-d", f"{NX}:{NY}:{NZ}"]
            if use_gpu and have_gpu:
                cmd.insert(2, "-g")

            print("[bart]", " ".join(cmd + [str(traj_base), str(ksp_base), str(coil_base)]))
            cmd += [str(traj_base), str(ksp_base), str(coil_base)]
            subprocess.run(cmd, check=True)

            if combine == "sos":
                # IMPORTANT: rss argument is a BITMASK of dims.
                # Coil dim for our NUFFT output is dim=3 (x,y,z,coil,...),
                # so mask = 1<<3 = 8.
                rss_mask = str(1 << 3)
                cmd2 = [bart_bin, "rss", rss_mask, str(coil_base), str(sos_base)]
                print("[bart]", " ".join(cmd2))
                subprocess.run(cmd2, check=True)
            else:
                raise ValueError(f"Unsupported combine mode: {combine}")

        # After first frame: print dims for debugging
        if not first_dims_reported:
            dims_coil = bart_image_dims(bart_bin, coil_base)
            dims_sos = bart_image_dims(bart_bin, sos_base)
            if dims_coil is not None:
                print(f"[debug] BART dims coil vol0: {dims_coil}")
            if dims_sos is not None:
                print(f"[debug] BART dims SoS vol0: {dims_sos}")
            first_dims_reported = True

        # Write QA as soon as first qa_first frames exist
        if qa_first is not None and not qa_written and len(frame_paths) >= qa_first:
            qa_frames = frame_paths[:qa_first]
            qa_base = per_series_dir / f"{out_base.name}_QA_first{qa_first}"
            write_qa_nifti(qa_frames, qa_base)
            qa_written = True

    # ---- Final 4D stack ---- #
    sos_existing = [p for p in frame_paths if p.with_suffix(".cfl").exists()]
    if not sos_existing:
        print("[warn] No per-frame SoS CFLs exist; skipping 4D stack.", file=sys.stderr)
        return

    stack_base = per_series_dir / out_base.name
    # join along dim=3 (4th dim), consistent with typical (x,y,z,t) usage
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
        description="Bruker 3D radial → BART NUFFT recon driver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Typical usage:

              python bruker_radial_bart.py \\
                --series /path/to/Bruker/29 \\
                --spokes-per-frame 200 \\
                --frame-shift 50 \\
                --traj-mode kron \\
                --combine sos \\
                --qa-first 2 \\
                --export-nifti \\
                --out /some/output/prefix

            If you have a grad.output containing ProjR/ProjP/ProjS (recommended):
              --grad-output /path/to/grad.output --spoke-order tile

            """
        ),
    )

    ap.add_argument("--series", required=True, help="Bruker 3D radial series directory")
    ap.add_argument("--out", required=True, help="Output base path (no extension)")

    ap.add_argument(
        "--matrix",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        help="Override PVM_Matrix",
    )
    ap.add_argument("--readout", type=int, help="Override true RO length (ACQ_size[0])")
    ap.add_argument("--coils", type=int, help="Override number of coils (PVM_EncNReceivers)")

    ap.add_argument(
        "--traj-mode",
        choices=["kron"],
        default="kron",
        help="Trajectory mode (fallback synthetic; ignored if --grad-output provided)",
    )

    ap.add_argument("--spokes-per-frame", type=int, default=0,
                    help="Spokes per frame for sliding window (0 => all spokes in one frame)")
    ap.add_argument("--frame-shift", type=int, default=0,
                    help="Frame shift (0 => spokes-per-frame)")

    ap.add_argument("--fid-dtype", choices=["i4", "f4"], default="i4",
                    help="Base numeric type of FID data; combined with --fid-endian")
    ap.add_argument("--fid-endian", choices=[">", "<"], default="<",
                    help="FID endianness: '>' big-endian, '<' little-endian (default).")

    ap.add_argument(
        "--fid-layout",
        choices=["ro_spokes_coils", "ro_coils_spokes"],
        default="ro_spokes_coils",
        help="How to interpret raw FID layout before trimming. (test2 = ro_coils_spokes)",
    )

    ap.add_argument("--combine", choices=["sos"], default="sos",
                    help="Coil combine mode (only 'sos' implemented)")
    ap.add_argument("--qa-first", type=int, default=0,
                    help="Write QA NIfTI of first N volumes (0 => disable)")
    ap.add_argument("--export-nifti", action="store_true",
                    help="Export final 4D stack as NIfTI (.nii.gz) via nibabel")

    ap.add_argument("--traj-scale", type=float, default=None,
                    help="Extra scale factor for k-space coordinates (try 0.5, 1.0, 2.0)")

    ap.add_argument(
        "--readout-origin",
        choices=["centered", "zero"],
        default="centered",
        help="Radial readout coordinate: centered (-k..+k) or zero (0..k).",
    )
    ap.add_argument(
        "--reverse-readout",
        action="store_true",
        help="Reverse sample order along readout before trajectory multiplication.",
    )

    ap.add_argument(
        "--grad-output",
        default=None,
        help="Path to Bruker grad.output to extract ProjR/ProjP/ProjS directions.",
    )
    ap.add_argument(
        "--spoke-order",
        choices=["tile", "repeat"],
        default="tile",
        help="How to expand grad-derived directions if spokes > N_dirs (e.g. 2584->7752).",
    )

    ap.add_argument("--gpu", action="store_true",
                    help="Try to use BART GPU NUFFT (falls back to CPU if not supported)")
    ap.add_argument("--debug", action="store_true", help="Reserved debug flag")

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
        NX, NY, NZ = args.matrix
        print(f"[info] Matrix overridden from CLI: {NX}x{NY}x{NZ}")
    else:
        NX, NY, NZ = infer_matrix(method)
        print(f"[info] Matrix inferred from PVM_Matrix: {NX}x{NY}x{NZ}")

    # True RO
    if args.readout is not None:
        true_ro = int(args.readout)
        print(f"[info] Readout (true RO) overridden from CLI: RO={true_ro}")
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

    # grad.output path
    grad_output_path = None
    if args.grad_output is not None:
        grad_output_path = Path(args.grad_output).expanduser().resolve()
        if not grad_output_path.exists():
            ap.error(f"--grad-output does not exist: {grad_output_path}")

    # Load k-space
    ksp, stored_ro, spokes_all, coils = load_bruker_kspace(
        fid_path=fid,
        true_ro=true_ro,
        coils_hint=coils_hint,
        endian=args.fid_endian,
        base_kind=args.fid_dtype,
        fid_layout=args.fid_layout,
    )

    # Frame config
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
        traj_mode=args.traj_mode,
        spokes_per_frame=spokes_per_frame,
        frame_shift=frame_shift,
        combine=args.combine,
        qa_first=qa_first,
        export_nifti=args.export_nifti,
        traj_scale=args.traj_scale,
        use_gpu=args.gpu,
        debug=args.debug,
        grad_output_path=grad_output_path,
        spoke_order=args.spoke_order,
        readout_origin=args.readout_origin,
        reverse_readout=args.reverse_readout,
    )


if __name__ == "__main__":
    main()
