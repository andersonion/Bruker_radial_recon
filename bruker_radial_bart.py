#!/usr/bin/env python3
"""
bruker_radial_bart.py

Simple Bruker 3D radial → BART NUFFT recon driver.

Features:
- Infers matrix from PVM_Matrix.
- Infers true RO from ACQ_size.
- Robust factoring of FID into (stored_ro * spokes * coils), allows stored_ro=128 etc.
- Uses true RO by trimming along readout dimension.
- Sliding-window framing (spokes-per-frame, frame-shift).
- Per-frame caching: skips frames whose coil-combined CFL already exists.
- Optional QA: exports first N frames as NIfTI with correct dims (X,Y,Z,T).
- Final 4D stack export as NIfTI.
- Optional GPU flag with graceful fallback if BART has no GPU support.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import textwrap

import numpy as np
import nibabel as nib


# ---------------------------------------------------------------------------
# Bruker helpers
# ---------------------------------------------------------------------------

def read_bruker_param(path: Path, key: str, default=None):
    """
    Minimal Bruker parameter reader.

    Looks for lines like:
      ##$PVM_Matrix=( 3 ) 96 96 96

    Returns:
      - list of values if multiple
      - single scalar if only one
      - default if not found
    """
    if not path.exists():
        return default

    key_token = f"##${key}="
    try:
        with path.open("r") as f:
            lines = f.readlines()
    except Exception:
        return default

    for idx, line in enumerate(lines):
        if line.startswith(key_token):
            parts = line.strip().split("=")
            if len(parts) < 2:
                continue
            rhs = parts[1].strip()

            if rhs.startswith("("):
                vals = []
                j = idx + 1
                while j < len(lines):
                    l2 = lines[j].strip()
                    if l2.startswith("##"):
                        break
                    if l2 == "" or l2.startswith("$$"):
                        j += 1
                        continue
                    vals.extend(l2.split())
                    j += 1
                vals_parsed = []
                for v in vals:
                    try:
                        if "." in v or "e" in v.lower():
                            vals_parsed.append(float(v))
                        else:
                            vals_parsed.append(int(v))
                    except ValueError:
                        vals_parsed.append(v)
                if len(vals_parsed) == 1:
                    return vals_parsed[0]
                return vals_parsed
            else:
                rhs = rhs.strip()
                rhs = rhs.strip("()")
                tokens = [t for t in rhs.split() if t]
                vals_parsed = []
                for v in tokens:
                    try:
                        if "." in v or "e" in v.lower():
                            vals_parsed.append(float(v))
                        else:
                            vals_parsed.append(int(v))
                    except ValueError:
                        vals_parsed.append(v)
                if len(vals_parsed) == 1:
                    return vals_parsed[0]
                return vals_parsed

    return default


def infer_matrix(method: Path):
    mat = read_bruker_param(method, "PVM_Matrix", default=None)
    if mat is None:
        raise ValueError("Could not infer PVM_Matrix")
    if isinstance(mat, (int, float)) or len(mat) != 3:
        raise ValueError(f"Unexpected PVM_Matrix: {mat}")
    NX, NY, NZ = map(int, mat)
    return NX, NY, NZ


def infer_true_ro(acqp: Path):
    acq_size = read_bruker_param(acqp, "ACQ_size", default=None)
    if acq_size is None:
        raise ValueError("Could not infer ACQ_size for true RO")
    if isinstance(acq_size, (int, float)):
        ro = int(acq_size)
    else:
        ro = int(acq_size[0])
    return ro


def infer_coils(method: Path):
    coils = read_bruker_param(method, "PVM_EncNReceivers", default=None)
    if coils is None:
        coils = 1
    if isinstance(coils, (list, tuple)):
        coils = int(coils[0])
    return int(coils)


# ---------------------------------------------------------------------------
# FID loading and factoring
# ---------------------------------------------------------------------------

def factor_fid(total_points: int, true_ro: int, coils_hint: int | None = None):
    """
    Factor total complex samples into (stored_ro * spokes * coils).

    We allow stored_ro != true_ro (e.g. stored_ro=128, true_ro=122).
    """
    block_candidates = [true_ro]
    for b in (128, 96, 64, 384, 512):
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
                score += 1
            if abs(stored_ro - true_ro) <= 10:
                score += 1
            if spokes > 100:
                score += 1
            if c == coils_hint:
                score += 1

            if score > best_score:
                best_score = score
                best = (stored_ro, c, spokes)

    if best is None:
        raise ValueError(
            f"Could not factor FID into (stored_ro * spokes * coils). "
            f"total={total_points}, true_ro={true_ro}, coils_hint={coils_hint}"
        )

    return best  # (stored_ro, coils, spokes)


def load_bruker_kspace(fid_path: Path, true_ro: int, coils_hint: int | None = None, endian: str = ">"):
    """
    Load Bruker FID as complex64.

    Returns:
        ksp_true: complex array, shape (true_ro, spokes, coils)
        stored_ro: int
        spokes: int
        coils: int
    """
    if not fid_path.exists():
        raise FileNotFoundError(f"FID not found: {fid_path}")

    dtype = np.dtype(f"{endian}f4")
    raw = np.fromfile(fid_path, dtype=dtype)
    if raw.size % 2 != 0:
        raise ValueError(f"FID raw length {raw.size} not even (real/imag pairs).")

    complex_data = raw[0::2] + 1j * raw[1::2]
    total_points = complex_data.size

    stored_ro, coils, spokes = factor_fid(total_points, true_ro, coils_hint=coils_hint)

    if coils_hint is not None and coils != coils_hint:
        print(
            f"[warn] FID factorization suggests coils={coils}, "
            f"but header said {coils_hint}; using coils={coils}.",
            file=sys.stderr,
        )

    print(
        f"[info] Loaded k-space with stored_ro={stored_ro}, true_ro={true_ro}, "
        f"spokes={spokes}, coils={coils}"
    )

    ksp = complex_data.reshape(stored_ro, spokes, coils)

    if stored_ro != true_ro:
        if stored_ro < true_ro:
            raise ValueError(
                f"stored_ro={stored_ro} < true_ro={true_ro}, cannot trim."
            )
        ksp = ksp[:true_ro, :, :]
        print(f"[info] Trimmed k-space from stored_ro={stored_ro} to true_ro={true_ro}")

    return ksp, stored_ro, spokes, coils


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

def build_kron_traj(true_ro: int, spokes: int, NX: int, NY: int, NZ: int, traj_scale: float | None = None):
    """
    Build a synthetic 3D 'kron' radial trajectory.

    Output shape: (3, true_ro, spokes)

    BART expects traj in units of pixel_size/FOV, roughly [-0.5, 0.5] at edge.
    We treat `traj_scale` as "k_max" (in pixels), and scale to max radius 0.5.

    If traj_scale is None, we assume k_max ~ NX/2.
    """
    if traj_scale is None:
        kmax = NX / 2.0
    else:
        kmax = float(traj_scale)

    idx = np.arange(spokes, dtype=np.float64)
    z = 2.0 * (idx / max(1, spokes - 1)) - 1.0
    phi = np.pi * (1 + 5**0.5) * idx
    r_xy = np.sqrt(1.0 - z**2)
    dx = r_xy * np.cos(phi)
    dy = r_xy * np.sin(phi)
    dz = z

    s = np.linspace(-0.5, 0.5, true_ro, dtype=np.float64)
    scale = 0.5 / max(kmax, 1e-6)
    s_scaled = s * (kmax * scale)

    traj = np.zeros((3, true_ro, spokes), dtype=np.float32)
    for i in range(spokes):
        traj[0, :, i] = s_scaled * dx[i]
        traj[1, :, i] = s_scaled * dy[i]
        traj[2, :, i] = s_scaled * dz[i]

    return traj


# ---------------------------------------------------------------------------
# BART CFL/HDR helpers
# ---------------------------------------------------------------------------

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


def readcfl(name: str):
    base = Path(name)
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")

    with hdr.open("r") as f:
        _ = f.readline()
        dims_line = f.readline()
    dims = [int(x) for x in dims_line.strip().split()]
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()

    with cfl.open("rb") as f:
        data = np.fromfile(f, dtype=np.float32)
    if data.size % 2 != 0:
        raise ValueError(f"CFL data size {data.size} not even.")

    cplx = data[0::2] + 1j * data[1::2]
    arr = cplx.view(np.complex64)
    arr = arr.reshape(dims, order="F")
    return arr


# ---------------------------------------------------------------------------
# NIfTI helpers
# ---------------------------------------------------------------------------

def bartstack_to_nifti(bart_base: Path, out_nifti: Path):
    """
    Load a coil-combined 4D CFL (BART ordering) and write a 4D NIfTI.

    We assume images are stored with dims (X, Y, Z, T) or a transposed variant.
    """
    img = readcfl(str(bart_base))
    mag = np.abs(img)
    shape = mag.shape
    non1 = [d for d in shape if d > 1]

    if len(non1) < 3:
        raise ValueError(f"Not enough non-singleton dims in {shape} for a 3D+T image.")

    if len(shape) == 4:
        X, Y, Z, T = shape
        vol = mag
    else:
        n_spatial = np.prod(shape[:-1])
        X = int(round(n_spatial ** (1.0 / 3.0)))
        if X ** 3 != n_spatial:
            raise ValueError(f"Cannot infer cubic spatial shape from {shape}.")
        T = shape[-1]
        vol = mag.reshape((X, X, X, T))

    affine = np.eye(4, dtype=float)
    nif = nib.Nifti1Image(vol.astype(np.float32), affine)
    nib.save(nif, str(out_nifti))
    print(f"[info] Wrote NIfTI -> {out_nifti}")


# ---------------------------------------------------------------------------
# BART GPU probe
# ---------------------------------------------------------------------------

def bart_supports_gpu(bart_bin: str = "bart") -> bool:
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

    return "unknown option" not in stderr


# ---------------------------------------------------------------------------
# Main recon
# ---------------------------------------------------------------------------

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
):
    bart_bin = "bart"

    ro, spokes_all, coils = ksp.shape

    if spokes_per_frame <= 0:
        raise ValueError("spokes-per-frame must be > 0")
    if frame_shift <= 0:
        frame_shift = spokes_per_frame

    frame_start_indices = list(range(0, max(1, spokes_all - spokes_per_frame + 1), frame_shift))
    n_frames = len(frame_start_indices)
    print(f"[info] Sliding window: spokes-per-frame={spokes_per_frame}, frame-shift={frame_shift}")
    print(f"[info] Will reconstruct {n_frames} frame(s).")

    traj_full = build_kron_traj(true_ro, spokes_all, NX, NY, NZ, traj_scale=traj_scale)

    have_gpu = False
    if use_gpu:
        have_gpu = bart_supports_gpu(bart_bin)
        if not have_gpu:
            print(
                "[warn] BART compiled without GPU support; falling back to CPU NUFFT and disabling --gpu.",
                file=sys.stderr,
            )

    per_series_dir = out_base.parent
    per_series_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    for i, start in enumerate(frame_start_indices):
        stop = start + spokes_per_frame
        if stop > spokes_all:
            stop = spokes_all
        frame_spokes = stop - start

        frame_tag = f"vol{i:05d}"
        traj_base = per_series_dir / f"{out_base.name}_{frame_tag}_traj"
        ksp_base = per_series_dir / f"{out_base.name}_{frame_tag}_ksp"
        coil_base = per_series_dir / f"{out_base.name}_{frame_tag}_coil"
        sos_base = per_series_dir / f"{out_base.name}_{frame_tag}"

        frame_paths.append(sos_base)

        if sos_base.with_suffix(".cfl").exists():
            print(f"[info] Frame {i} already reconstructed -> {sos_base}, skipping NUFFT/RSS.")
            continue

        print(f"[info] Frame {i} spokes [{start}:{stop}] (n={frame_spokes})")

        ksp_frame = ksp[:, start:stop, :]
        traj_frame = traj_full[:, :, start:stop]

        ksp_bart = ksp_frame[np.newaxis, ...]  # (1, ro, spokes_frame, coils)

        writecfl(str(traj_base), traj_frame)
        writecfl(str(ksp_base), ksp_bart)

        cmd = [bart_bin, "nufft", "-i", "-d", f"{NX}:{NY}:{NZ}"]
        if use_gpu and have_gpu:
            cmd.insert(2, "-g")

        cmd += [str(traj_base), str(ksp_base), str(coil_base)]
        print("[bart]", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[error] BART NUFFT/RSS failed for frame {i} with code {e.returncode}", file=sys.stderr)
            raise

        if combine == "sos":
            cmd2 = [bart_bin, "rss", "3", str(coil_base), str(sos_base)]
            print("[bart]", " ".join(cmd2))
            subprocess.run(cmd2, check=True)
        else:
            raise ValueError(f"Unsupported combine mode: {combine}")

    if qa_first is not None and qa_first > 0:
        qa_frames = frame_paths[:qa_first]
        if len(qa_frames) > 0:
            qa_join_base = per_series_dir / f"{out_base.name}_QA_first{qa_first}"
            join_cmd = ["bart", "join", "3"] + [str(base) for base in qa_frames] + [str(qa_join_base)]
            print("[bart]", " ".join(join_cmd))
            subprocess.run(join_cmd, check=True)

            qa_nii = qa_join_base.with_suffix(".nii.gz")
            try:
                bartstack_to_nifti(qa_join_base, qa_nii)
            except Exception as e:
                print(f"[warn] Failed to convert QA CFL to NIfTI: {e}", file=sys.stderr)

    sos_existing = [p for p in frame_paths if p.with_suffix(".cfl").exists()]
    if not sos_existing:
        print("[warn] No per-frame SoS CFLs exist; skipping 4D stack.", file=sys.stderr)
        return

    stack_base = per_series_dir / out_base.name
    join_cmd = ["bart", "join", "3"] + [str(p) for p in sos_existing] + [str(stack_base)]
    print("[bart]", " ".join(join_cmd))
    subprocess.run(join_cmd, check=True)

    if export_nifti:
        stack_nii = stack_base.with_suffix(".nii.gz")
        try:
            bartstack_to_nifti(stack_base, stack_nii)
        except Exception as e:
            print(f"[warn] Failed to convert final 4D CFL to NIfTI: {e}", file=sys.stderr)
    print(f"[info] All requested frames complete; 4D result at {stack_base}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Bruker 3D radial → BART NUFFT recon driver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Example:
              bruker_radial_bart.py \\
                --series /path/to/Bruker/29 \\
                --spokes-per-frame 200 \\
                --frame-shift 50 \\
                --traj-mode kron \\
                --traj-scale 48 \\
                --combine sos \\
                --export-nifti \\
                --qa-first 2 \\
                --out /some/output/dir/29_bart_recon_SoS
            """
        ),
    )
    ap.add_argument("--series", required=True, help="Bruker series directory (e.g. .../29)")
    ap.add_argument("--out", required=True, help="Base output path (no extension)")

    ap.add_argument("--matrix", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                    help="Override PVM_Matrix")
    ap.add_argument("--readout", type=int, help="Override true RO (from ACQ_size)")
    ap.add_argument("--coils", type=int, help="Override number of coils")

    ap.add_argument("--traj-mode", choices=["kron"], default="kron",
                    help="Trajectory mode (currently only 'kron' supported)")
    ap.add_argument("--spokes-per-frame", type=int, default=0, help="Spokes per frame (sliding window)")
    ap.add_argument("--frame-shift", type=int, default=0, help="Frame shift (default = spokes-per-frame)")

    ap.add_argument("--test-volumes", type=int, nargs="+",
                    help="(Unused placeholder) specific volumes to test")

    ap.add_argument("--fid-dtype", default=">f4", help="FID dtype (ignored, fixed to big-endian float32)")
    ap.add_argument("--fid-endian", choices=[">", "<"], default=">",
                    help="Endianness for FID floats (default big-endian '>')")

    ap.add_argument("--combine", choices=["sos"], default="sos", help="Coil combination method")
    ap.add_argument("--qa-first", type=int, default=0,
                    help="Write QA NIfTI with first N frames (0 disables)")
    ap.add_argument("--export-nifti", action="store_true", help="Export final 4D stack as NIfTI")
    ap.add_argument("--traj-scale", type=float, default=None,
                    help="Physical k_max in pixels (e.g. NX/2). Internally scaled to BART units (~0.5 at edge).")

    ap.add_argument("--gpu", action="store_true", help="Use BART GPU NUFFT if available")
    ap.add_argument("--debug", action="store_true", help="Verbose debug output")

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
        ap.error(
            f"Could not find Bruker files under {series_path}. "
            f"Expected method, acqp, fid."
        )

    if args.matrix is not None:
        NX, NY, NZ = args.matrix
        print(f"[info] Matrix overridden from CLI: {NX}x{NY}x{NZ}")
    else:
        NX, NY, NZ = infer_matrix(method)
        print(f"[info] Matrix inferred from PVM_Matrix: {NX}x{NY}x{NZ}")

    if args.readout is not None:
        true_ro = int(args.readout)
        print(f"[info] Readout (true RO) overridden from CLI: RO={true_ro}")
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
        fid, true_ro=true_ro, coils_hint=coils_hint, endian=args.fid_endian
    )

    if args.spokes_per_frame <= 0:
        spokes_per_frame = spokes_all
    else:
        spokes_per_frame = args.spokes_per_frame

    frame_shift = args.frame_shift if args.frame_shift > 0 else spokes_per_frame

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
        qa_first=args.qa_first if args.qa_first > 0 else None,
        export_nifti=args.export_nifti,
        traj_scale=args.traj_scale,
        use_gpu=args.gpu,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
