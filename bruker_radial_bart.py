#!/usr/bin/env python3

import argparse
import os
import sys
import subprocess
import textwrap
from pathlib import Path

import numpy as np


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
        with path.open("r") as f:
            lines = f.readlines()
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

    We try several candidate stored_ro values and coil counts and pick a
    "reasonable" combination, preferring:
      - stored_ro >= true_ro
      - stored_ro close to true_ro
      - spokes not ridiculously small
      - matching the coil hint if possible.
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
            if coils_hint is not None and c == coils_hint:
                score += 1

            if score > best_score:
                best_score = score
                best = (stored_ro, c, spokes)

    if best is None:
        raise ValueError(
            f"Could not factor FID into (stored_ro * spokes * coils). "
            f"total={total_points}, true_ro={true_ro}, coils_hint={coils_hint}"
        )

    return best


def load_bruker_kspace(
    fid_path: Path,
    true_ro: int,
    coils_hint: int | None = None,
    endian: str = "<",
    base_kind: str = "i4",
):
    """
    Load Bruker FID as complex data and reshape to (stored_ro, spokes, coils),
    then trim to (true_ro, spokes, coils).

    endian: '>' or '<'
    base_kind: 'i4' (int32) or 'f4' (float32)
    """
    if not fid_path.exists():
        raise FileNotFoundError(f"FID not found: {fid_path}")

    try:
        np_dtype = np.dtype(endian + base_kind)
    except TypeError as e:
        raise ValueError(f"Invalid dtype combination: endian={endian}, kind={base_kind}") from e

    raw = np.fromfile(fid_path, dtype=np_dtype)
    # Work in float32 for the complex pairing
    raw = raw.astype(np.float32)

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

    ksp = complex_data.reshape(stored_ro, spokes, coils)

    # Trim down to true RO if needed
    if stored_ro != true_ro:
        if stored_ro < true_ro:
            raise ValueError(
                f"stored_ro={stored_ro} < true_ro={true_ro}, cannot trim."
            )
        ksp = ksp[:true_ro, :, :]
        print(f"[info] Trimmed k-space from stored_ro={stored_ro} to true_ro={true_ro}")

    return ksp, stored_ro, spokes, coils


# ---------------- Trajectory + BART helpers ---------------- #


def build_kron_traj(
    true_ro: int,
    spokes: int,
    NX: int,
    NY: int,
    NZ: int,
    traj_scale: float | None = None,
) -> np.ndarray:
    """
    Build a synthetic 3D "kron" golden-angle trajectory for BART.

    Output shape: (3, RO, spokes)

    Coordinates are in units of pixel_size / FOV as BART expects.
    We set the maximum k-space radius to ~NX/2, so BART's internal
    "estimated image size" becomes ~NX instead of 2.
    """
    idx = np.arange(spokes, dtype=np.float64)
    # Map index to z in [-1, 1]
    if spokes > 1:
        z = 2.0 * (idx / (spokes - 1)) - 1.0
    else:
        z = np.zeros_like(idx)

    phi = np.pi * (1 + 5**0.5) * idx  # golden angle-ish

    r_xy = np.sqrt(np.clip(1.0 - z**2, 0.0, 1.0))
    dx = r_xy * np.cos(phi)
    dy = r_xy * np.sin(phi)
    dz = z

    # Radial coordinate: base in [-0.5, 0.5], then scale by NX and traj_scale.
    base_s = np.linspace(-0.5, 0.5, true_ro, dtype=np.float64) * NX
    scale = float(traj_scale) if traj_scale is not None else 1.0
    s = base_s * scale

    traj = np.zeros((3, true_ro, spokes), dtype=np.complex64)
    for i in range(spokes):
        traj[0, :, i] = s * dx[i]
        traj[1, :, i] = s * dy[i]
        traj[2, :, i] = s * dz[i]

    # Tiny debug print so we can sanity-check magnitude if needed
    max_rad = np.abs(traj).max()
    print(f"[info] Traj built with max |k| ≈ {max_rad:.2f}")

    return traj


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

    # Build full trajectory once, then slice per frame.
    if traj_mode != "kron":
        raise ValueError(f"Only traj-mode 'kron' is currently implemented.")
    traj_full = build_kron_traj(true_ro, spokes_all, NX, NY, NZ, traj_scale)

    # GPU probe
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

        # If SoS exists for this frame, skip redo.
        if sos_base.with_suffix(".cfl").exists():
            print(
                f"[info] Frame {i} already reconstructed -> "
                f"{sos_base}, skipping NUFFT/RSS."
            )
            continue

        print(f"[info] Frame {i} spokes [{start}:{stop}] (n={frame_spokes})")

        ksp_frame = ksp[:, start:stop, :]  # (ro, spokes_frame, coils)
        traj_frame = traj_full[:, :, start:stop]  # (3, ro, spokes_frame)

        # BART expects first dim == 1 for k-space -> (1, ro, spokes_frame, coils)
        ksp_bart = ksp_frame[np.newaxis, ...]

        writecfl(str(traj_base), traj_frame)
        writecfl(str(ksp_base), ksp_bart)

        # NUFFT: bart nufft [-g] -i -d NX:NY:NZ traj ksp coil
        cmd = [bart_bin, "nufft", "-i", "-d", f"{NX}:{NY}:{NZ}"]
        if use_gpu and have_gpu:
            cmd.insert(2, "-g")
        print("[bart]", " ".join(cmd + [str(traj_base), str(ksp_base), str(coil_base)]))
        cmd += [str(traj_base), str(ksp_base), str(coil_base)]

        subprocess.run(cmd, check=True)

        # Coil combine
        if combine == "sos":
            cmd2 = [bart_bin, "rss", "3", str(coil_base), str(sos_base)]
            print("[bart]", " ".join(cmd2))
            subprocess.run(cmd2, check=True)
        else:
            raise ValueError(f"Unsupported combine mode: {combine}")

    # ---- QA volumes (first N) ---- #
    if qa_first is not None and qa_first > 0 and frame_paths:
        qa_frames = frame_paths[:qa_first]
        qa_base = per_series_dir / f"{out_base.name}_QA_first{qa_first}"

        join_cmd = ["bart", "join", "3"] + [str(p) for p in qa_frames] + [str(qa_base)]
        print("[bart]", " ".join(join_cmd))
        subprocess.run(join_cmd, check=True)

        qa_nii = qa_base.with_suffix(".nii")
        print("[bart]", f"toimg {qa_base} {qa_nii}")
        subprocess.run([bart_bin, "toimg", str(qa_base), str(qa_nii)], check=True)
        subprocess.run(["gzip", "-f", str(qa_nii)], check=True)

    # ---- Final 4D stack ---- #
    sos_existing = [p for p in frame_paths if p.with_suffix(".cfl").exists()]
    if not sos_existing:
        print(
            "[warn] No per-frame SoS CFLs exist; skipping 4D stack.",
            file=sys.stderr,
        )
        return

    stack_base = per_series_dir / out_base.name
    join_cmd = ["bart", "join", "3"] + [str(p) for p in sos_existing] + [str(stack_base)]
    print("[bart]", " ".join(join_cmd))
    subprocess.run(join_cmd, check=True)

    if export_nifti:
        stack_nii = stack_base.with_suffix(".nii")
        print("[bart]", f"toimg {stack_base} {stack_nii}")
        subprocess.run([bart_bin, "toimg", str(stack_nii)], check=True)
        subprocess.run(["gzip", "-f", str(stack_nii)], check=True)

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
    ap.add_argument(
        "--readout",
        type=int,
        help="Override true RO length (ACQ_size[0])",
    )
    ap.add_argument(
        "--coils",
        type=int,
        help="Override number of coils (PVM_EncNReceivers)",
    )

    ap.add_argument(
        "--traj-mode",
        choices=["kron"],
        default="kron",
        help="Trajectory mode (currently only 'kron')",
    )
    ap.add_argument(
        "--spokes-per-frame",
        type=int,
        default=0,
        help="Spokes per frame for sliding window (0 => all spokes in one frame)",
    )
    ap.add_argument(
        "--frame-shift",
        type=int,
        default=0,
        help="Frame shift (0 => spokes-per-frame)",
    )

    ap.add_argument(
        "--test-volumes",
        type=int,
        nargs="+",
        help="Reserved for future use (subset of volumes). Currently ignored.",
    )

    ap.add_argument(
        "--fid-dtype",
        choices=["i4", "f4"],
        default="i4",
        help="Base numeric type of FID data (32-bit int or float); "
             "combined with --fid-endian.",
    )
    ap.add_argument(
        "--fid-endian",
        choices=[">", "<"],
        default="<",
        help="FID endianness: '>' big-endian, '<' little-endian (default).",
    )

    ap.add_argument(
        "--combine",
        choices=["sos"],
        default="sos",
        help="Coil combine mode (only 'sos' implemented)",
    )
    ap.add_argument(
        "--qa-first",
        type=int,
        default=0,
        help="Write QA NIfTI of first N volumes (0 => disable)",
    )
    ap.add_argument(
        "--export-nifti",
        action="store_true",
        help="Export final 4D stack as NIfTI (.nii.gz)",
    )

    ap.add_argument(
        "--traj-scale",
        type=float,
        default=None,
        help="Extra scale factor for k-space coordinates. "
             "Leave unset unless you know you need it.",
    )

    ap.add_argument(
        "--gpu",
        action="store_true",
        help="Try to use BART GPU NUFFT (falls back to CPU if not supported)",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Reserved debug flag",
    )

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

    # Load k-space
    ksp, stored_ro, spokes_all, coils = load_bruker_kspace(
        fid,
        true_ro,
        coils_hint,
        endian=args.fid_endian,
        base_kind=args.fid_dtype,
    )

    # Frame config
    if args.spokes_per_frame <= 0:
        spokes_per_frame = spokes_all
    else:
        spokes_per_frame = args.spokes_per_frame

    frame_shift = args.frame_shift if args.frame_shift > 0 else spokes_per_frame
    qa_first = args.qa_first if args.qa_first > 0 else None

    # Run recon
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
    )


if __name__ == "__main__":
    main()
