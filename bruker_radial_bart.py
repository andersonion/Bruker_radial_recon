#!/usr/bin/env python3
"""
bruker_radial_bart.py

Bruker 3D radial -> BART NUFFT recon, with MATLAB-faithful parsing of:
  - fid: int32 ieee-le, block-trimmed to n_ray per "ray", then reshaped with column-major semantics
  - traj: float64 ieee-le, reshaped (3, n_read, n_spokes) with column-major semantics, scaled by traj_scaling

Also includes an OPTIONAL gradient-delay / readout-shift correction:
  - Find opposed spoke pairs from traj directions
  - Estimate per-pair 1D shift (samples) by FFT cross-correlation against conjugate-reversed opposite spoke
  - Fit axis-dependent model: s_i ≈ ax*uix + ay*uiy + az*uiz
  - Predict per-spoke shifts and apply fractional shift along RO via Fourier shift theorem

Notes:
  - BART NUFFT expects k-space dims[0] == 1 => ksp written as (1, RO, spokes, coils)
  - traj written as (3, RO, spokes)
  - This script aims to be boring and faithful. No autoshape/heuristics for traj parsing.
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib


# ----------------------------
# Bruker param helpers
# ----------------------------

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


# ----------------------------
# BART CFL I/O
# ----------------------------

def writecfl(name: str, arr: np.ndarray) -> None:
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


def bart_cfl_to_nifti(base: Path, out_nii_gz: Path, voxel_mm: Optional[float] = None) -> None:
    arr = np.abs(readcfl(str(base)))
    aff = np.eye(4, dtype=np.float64)
    if voxel_mm is not None and voxel_mm > 0:
        aff[0, 0] = voxel_mm
        aff[1, 1] = voxel_mm
        aff[2, 2] = voxel_mm
    nib.save(nib.Nifti1Image(arr.astype(np.float32), aff), str(out_nii_gz))


# ----------------------------
# MATLAB-faithful readers
# ----------------------------

def read_fid_matlab_like(
    fid_file: Path,
    *,
    n_vols: int,
    n_coils: int,
    n_read: int,
    endian: str = "<",              # MATLAB: ieee-le
    dtype: str = "i4",              # MATLAB: int32
    block_bytes: int = 1024,
) -> np.ndarray:
    """
    Returns complex64 array shaped (n_read, n_coils, n_spokes, n_vols),
    mirroring your MATLAB semantics, including column-major reshapes.
    """
    fid_file = Path(fid_file)
    raw = np.fromfile(fid_file, dtype=np.dtype(endian + dtype))

    n_ray = int(n_coils * n_read * 2)  # 2 for re/im
    bytes_per_sample = np.dtype(endian + dtype).itemsize
    n_blocks = int(np.ceil((n_ray * bytes_per_sample) / float(block_bytes)))
    n_block_samples = int((n_blocks * block_bytes) // bytes_per_sample)

    if n_block_samples <= 0:
        raise ValueError("Computed n_block_samples <= 0; check inputs.")

    if raw.size % n_block_samples != 0:
        raise ValueError(
            f"FID size {raw.size} not divisible by n_block_samples={n_block_samples} "
            f"(n_blocks={n_blocks}, block_bytes={block_bytes}, bytes/sample={bytes_per_sample})."
        )

    # MATLAB reshape is column-major; mirror with order='F'
    raw2 = raw.reshape(n_block_samples, -1, order="F")
    raw2 = raw2[:n_ray, :]
    raw2 = raw2.reshape(-1, order="F")  # x(:)

    denom = 2 * n_read * n_coils * n_vols
    if raw2.size % denom != 0:
        raise ValueError(
            "FID size after trim does not factor into "
            f"(2, n_read={n_read}, n_coils={n_coils}, n_vols={n_vols}). "
            f"Have {raw2.size} samples, denom={denom}."
        )

    n_spokes = raw2.size // denom

    x = raw2.reshape(2, n_read, n_coils, n_spokes, n_vols, order="F")
    cplx = x[0, ...].astype(np.float32) + 1j * x[1, ...].astype(np.float32)
    return cplx.astype(np.complex64, copy=False)


def read_traj_matlab_like(
    traj_file: Path,
    *,
    n_read: int,
    endian: str = "<",              # MATLAB: ieee-le
    dtype: str = "f8",              # MATLAB: float64
    traj_scaling: float = 96.0,
) -> np.ndarray:
    """
    Returns float32 array shaped (3, n_read, n_spokes), mirroring MATLAB reshape scaling.
    """
    traj_file = Path(traj_file)
    v = np.fromfile(traj_file, dtype=np.dtype(endian + dtype))

    if v.size % (3 * n_read) != 0:
        raise ValueError(
            f"traj length {v.size} not divisible by (3*n_read)={3*n_read}. "
            f"n_read={n_read}"
        )

    n_spokes = v.size // (3 * n_read)
    traj = v.reshape(3, n_read, n_spokes, order="F")
    traj = (traj * float(traj_scaling)).astype(np.float32, copy=False)
    return traj


# ----------------------------
# Gradient delay / readout shift correction
# ----------------------------

def spoke_directions_from_traj(traj: np.ndarray) -> np.ndarray:
    """
    traj: (3, RO, spokes) float/complex
    returns u: (spokes, 3) unit directions from endpoints
    """
    tx = np.real(traj[0]).astype(np.float64, copy=False)
    ty = np.real(traj[1]).astype(np.float64, copy=False)
    tz = np.real(traj[2]).astype(np.float64, copy=False)

    dx = tx[-1, :] - tx[0, :]
    dy = ty[-1, :] - ty[0, :]
    dz = tz[-1, :] - tz[0, :]

    dn = np.sqrt(dx * dx + dy * dy + dz * dz)
    dn[dn == 0] = 1.0
    dx /= dn
    dy /= dn
    dz /= dn

    return np.stack([dx, dy, dz], axis=1)


def find_opposed_pairs(u: np.ndarray, dot_threshold: float = 0.98) -> list[tuple[int, int]]:
    """
    Approx fast pairing: for each i, choose best match to -u[i].
    Accept if dot >= dot_threshold, then dedup.
    """
    n = u.shape[0]
    pairs = []

    UT = u.T  # (3, n)
    for i in range(n):
        target = -u[i]
        dots = target @ UT
        j = int(np.argmax(dots))
        if j != i and float(dots[j]) >= dot_threshold:
            a, b = (i, j) if i < j else (j, i)
            pairs.append((a, b))

    return sorted(set(pairs))


def estimate_shift_samples(a: np.ndarray, b: np.ndarray, *, max_shift: Optional[int] = None) -> float:
    """
    Cross-correlation peak shift estimate (integer shift).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"a.shape {a.shape} != b.shape {b.shape}")

    n = a.size
    A = np.fft.fft(a)
    B = np.fft.fft(b)
    cc = np.fft.ifft(A * np.conj(B))
    cc = np.fft.fftshift(cc)
    mag = np.abs(cc)

    if max_shift is not None and max_shift < (n // 2):
        mid = n // 2
        lo = mid - max_shift
        hi = mid + max_shift + 1
        mag_win = mag[lo:hi]
        k = int(np.argmax(mag_win)) + lo
    else:
        k = int(np.argmax(mag))

    return float(k - (n // 2))


def estimate_gradient_delay_from_opposed_pairs(
    ksp: np.ndarray,
    traj: np.ndarray,
    *,
    dot_threshold: float = 0.98,
    use_coils_sum: bool = True,
    max_shift: Optional[int] = None,
    max_pairs: Optional[int] = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    ksp: (RO, spokes, coils)
    traj: (3, RO, spokes)
    Returns:
      us: (n_pairs, 3) unit directions for spoke i
      shifts: (n_pairs,) shift samples
      coeffs: (3,) least squares coeffs [ax, ay, az]
    """
    RO, spokes, coils = ksp.shape
    u = spoke_directions_from_traj(traj)
    pairs = find_opposed_pairs(u, dot_threshold=dot_threshold)

    if len(pairs) == 0:
        raise RuntimeError("No opposed pairs found (try lowering --dot-threshold).")

    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    shifts = []
    us = []

    for i, j in pairs:
        if use_coils_sum:
            a = np.sum(ksp[:, i, :], axis=1)
            b = np.sum(ksp[:, j, :], axis=1)
        else:
            a = ksp[:, i, 0]
            b = ksp[:, j, 0]

        b_cr = np.conj(b[::-1])
        s = estimate_shift_samples(a, b_cr, max_shift=max_shift)

        shifts.append(s)
        us.append(u[i])

    shifts = np.asarray(shifts, dtype=np.float64)
    us = np.asarray(us, dtype=np.float64)

    coeffs, *_ = np.linalg.lstsq(us, shifts, rcond=None)
    return us, shifts, coeffs


def predict_shifts_from_coeffs(traj: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    u = spoke_directions_from_traj(traj)
    return (u @ coeffs.reshape(3, 1)).ravel()


def apply_ro_shift_per_spoke(ksp: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """
    Fractional RO shift per spoke using Fourier shift theorem along RO.
    ksp: (RO, spokes, coils)
    shifts: (spokes,)
    """
    RO, spokes, coils = ksp.shape
    if shifts.shape[0] != spokes:
        raise ValueError(f"shifts len {shifts.shape[0]} != spokes {spokes}")

    f = np.fft.fftfreq(RO)  # cycles/sample
    out = np.empty_like(ksp)

    for s in range(spokes):
        phase = np.exp(-1j * 2.0 * np.pi * f * float(shifts[s])).astype(np.complex64)
        for c in range(coils):
            X = np.fft.fft(ksp[:, s, c])
            X *= phase
            out[:, s, c] = np.fft.ifft(X)

    return out


# ----------------------------
# Recon
# ----------------------------

def run_bart_nufft(
    bart_bin: str,
    *,
    NX: int,
    NY: int,
    NZ: int,
    traj: np.ndarray,               # (3, RO, spokes) complex64
    ksp: np.ndarray,                # (RO, spokes, coils) complex64
    out_base: Path,
    gpu: bool,
    rss: bool,
) -> Path:
    """
    Writes traj/ksp CFLs and runs bart nufft. Returns base path of output image (CFL/HDR).
    """
    out_dir = out_base.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_base = out_dir / f"{out_base.name}_traj"
    ksp_base = out_dir / f"{out_base.name}_ksp"
    img_base = out_dir / f"{out_base.name}_img"

    # BART expects ksp dims[0]==1
    ksp_bart = ksp[np.newaxis, :, :, :]  # (1, RO, spokes, coils)

    writecfl(str(traj_base), traj.astype(np.complex64, copy=False))
    writecfl(str(ksp_base), ksp_bart.astype(np.complex64, copy=False))

    cmd = [bart_bin, "nufft", "-i", "-d", f"{NX}:{NY}:{NZ}"]
    if gpu:
        cmd.insert(2, "-g")
    cmd += [str(traj_base), str(ksp_base), str(img_base)]
    print("[bart]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if rss:
        # If you run nufft directly into img, BART will keep coil dim if present.
        # Here we assume coil dim is dim=3 (mask 1<<3 = 8) IF it exists.
        # For safety, try rss and if it fails, just leave img as-is.
        rss_base = out_dir / f"{out_base.name}_rss"
        rss_mask = str(1 << 3)
        cmd2 = [bart_bin, "rss", rss_mask, str(img_base), str(rss_base)]
        print("[bart]", " ".join(cmd2))
        p = subprocess.run(cmd2)
        if p.returncode == 0:
            return rss_base
        print("[warn] bart rss failed; leaving coil image.", file=sys.stderr)

    return img_base


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bruker 3D radial -> BART NUFFT using MATLAB-faithful fid/traj parsing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Example (match MATLAB):
              python bruker_radial_bart_matlab.py \\
                --series /path/to/21 \\
                --out /tmp/recon \\
                --n-read 61 --n-coils 4 --n-vols 15 \\
                --traj-scaling 96 \\
                --vol-idx 15 \\
                --export-nifti --voxel-mm 0.20833

            With gradient-delay fit + correction:
              python bruker_radial_bart_matlab.py \\
                --series /path/to/21 \\
                --out /tmp/recon \\
                --n-read 61 --n-coils 4 --n-vols 15 \\
                --traj-scaling 96 \\
                --vol-idx 15 \\
                --delay-fit --dot-threshold 0.98 --max-shift 20
            """
        ),
    )

    ap.add_argument("--series", required=True, help="Bruker series directory containing fid/method/acqp/traj")
    ap.add_argument("--out", required=True, help="Output base path (no extension).")

    ap.add_argument("--matrix", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                    help="Override matrix (default from PVM_Matrix).")

    # MATLAB-faithful inputs
    ap.add_argument("--n-read", type=int, default=0, help="Readout samples used by MATLAB (n_read).")
    ap.add_argument("--n-coils", type=int, default=0, help="Number of coils used by MATLAB (n_coils).")
    ap.add_argument("--n-vols", type=int, default=1, help="Number of volumes (n_vols).")
    ap.add_argument("--traj-scaling", type=float, default=96.0, help="Trajectory scaling multiplier (MATLAB traj_scaling).")

    ap.add_argument("--fid-endian", choices=["<", ">"], default="<")
    ap.add_argument("--traj-endian", choices=["<", ">"], default="<")

    ap.add_argument("--vol-idx", type=int, default=1,
                    help="1-based volume index to reconstruct (MATLAB example uses vol_idx=15). Use 0 to reconstruct ALL vols.")

    # Gradient delay / readout shift correction
    ap.add_argument("--delay-fit", action="store_true",
                    help="Estimate and apply readout shift correction using opposed spoke pairs.")
    ap.add_argument("--dot-threshold", type=float, default=0.98,
                    help="Opposed-pair threshold for dot(-u_i, u_j) >= dot_threshold.")
    ap.add_argument("--max-shift", type=int, default=0,
                    help="Max shift (samples) search window for correlation peak (0 means full).")
    ap.add_argument("--max-pairs", type=int, default=5000,
                    help="Cap number of pairs used in fit (speed control).")
    ap.add_argument("--no-coils-sum", action="store_true",
                    help="Use only coil 0 for shift estimation (default sums coils).")

    # BART / outputs
    ap.add_argument("--bart-bin", default="bart")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--rss", action="store_true", help="Run bart rss after nufft if possible.")
    ap.add_argument("--export-nifti", action="store_true")
    ap.add_argument("--voxel-mm", type=float, default=0.0, help="Voxel size (mm) for NIfTI affine diag.")
    args = ap.parse_args()

    series = Path(args.series).resolve()
    out_base = Path(args.out).resolve()

    method = series / "method"
    acqp = series / "acqp"
    fid = series / "fid"
    traj_path = series / "traj"

    if not (method.exists() and acqp.exists() and fid.exists()):
        ap.error(f"Could not find method/acqp/fid under {series}")
    if not traj_path.exists():
        ap.error(f"Could not find traj under {series} (expected {traj_path})")

    if args.matrix is not None:
        NX, NY, NZ = map(int, args.matrix)
        print(f"[info] Matrix overridden: {NX}x{NY}x{NZ}")
    else:
        NX, NY, NZ = infer_matrix(method)
        print(f"[info] Matrix from PVM_Matrix: {NX}x{NY}x{NZ}")

    header_coils = infer_coils(method)
    header_ro = infer_true_ro(acqp)

    # If user doesn't supply n_read/n_coils, we pick header values and warn loudly.
    n_read = int(args.n_read) if args.n_read and args.n_read > 0 else int(header_ro)
    n_coils = int(args.n_coils) if args.n_coils and args.n_coils > 0 else int(header_coils)
    n_vols = int(args.n_vols) if args.n_vols and args.n_vols > 0 else 1

    if args.n_read <= 0:
        print(f"[warn] --n-read not provided; defaulting to ACQ_size RO={n_read}. "
              f"If MATLAB used a different n_read (e.g. 61 vs 126), pass --n-read explicitly.",
              file=sys.stderr)
    if args.n_coils <= 0:
        print(f"[warn] --n-coils not provided; defaulting to PVM_EncNReceivers={n_coils}. "
              f"If MATLAB used a different n_coils, pass --n-coils explicitly.",
              file=sys.stderr)

    print(f"[info] MATLAB-faithful parse config: n_read={n_read}, n_coils={n_coils}, n_vols={n_vols}, traj_scaling={args.traj_scaling}")

    # Read data
    x = read_fid_matlab_like(
        fid,
        n_vols=n_vols,
        n_coils=n_coils,
        n_read=n_read,
        endian=args.fid_endian,
        dtype="i4",
        block_bytes=1024,
    )  # (RO, coils, spokes, vols)
    traj_f = read_traj_matlab_like(
        traj_path,
        n_read=n_read,
        endian=args.traj_endian,
        dtype="f8",
        traj_scaling=args.traj_scaling,
    )  # (3, RO, spokes)

    # Consistency
    RO, coils, spokes, vols = x.shape
    if traj_f.shape[1] != RO:
        raise ValueError(f"traj RO={traj_f.shape[1]} != fid RO={RO}")
    if traj_f.shape[2] != spokes:
        # In your MATLAB snippet, traj was reshaped (3, n_read, []) and used directly for that ksp.
        # So spokes must match for a clean nufft call.
        raise ValueError(f"traj spokes={traj_f.shape[2]} != fid spokes={spokes}")

    print(f"[info] Parsed fid: shape (RO,coils,spokes,vols)=({RO},{coils},{spokes},{vols})")
    print(f"[info] Parsed traj: shape (3,RO,spokes)=({traj_f.shape[0]},{traj_f.shape[1]},{traj_f.shape[2]})")

    # Make traj complex for BART CFL writer (imag=0)
    traj = np.asfortranarray(traj_f.astype(np.float32)).astype(np.complex64)

    # Which volumes?
    if args.vol_idx == 0:
        vol_indices = list(range(vols))  # 0-based all
    else:
        vi = int(args.vol_idx) - 1
        if vi < 0 or vi >= vols:
            raise ValueError(f"--vol-idx {args.vol_idx} out of range [1..{vols}]")
        vol_indices = [vi]

    # For each volume: build ksp (RO, spokes, coils) and optionally correct
    out_dir = out_base.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    img_bases = []

    for vi in vol_indices:
        # x is (RO, coils, spokes, vols) => rearrange to (RO, spokes, coils)
        ksp = x[:, :, :, vi].transpose(0, 2, 1).astype(np.complex64, copy=False)

        print(f"[info] Volume {vi+1}/{vols}: ksp shape (RO,spokes,coils)={ksp.shape}")

        if args.delay_fit:
            print("[info] Delay fit enabled: estimating opposed-pair shifts...")
            max_shift = int(args.max_shift) if args.max_shift and args.max_shift > 0 else None
            us, shifts_pairs, coeffs = estimate_gradient_delay_from_opposed_pairs(
                ksp=ksp,
                traj=traj,
                dot_threshold=float(args.dot_threshold),
                use_coils_sum=(not args.no_coils_sum),
                max_shift=max_shift,
                max_pairs=int(args.max_pairs) if args.max_pairs and args.max_pairs > 0 else None,
            )
            print(f"[info] Fit coeffs (samples): ax={coeffs[0]:.6g}, ay={coeffs[1]:.6g}, az={coeffs[2]:.6g}")
            print(f"[info] Pair shifts (samples): median={np.median(shifts_pairs):.6g}, "
                  f"p5={np.percentile(shifts_pairs,5):.6g}, p95={np.percentile(shifts_pairs,95):.6g}, n_pairs={shifts_pairs.size}")

            shifts_all = predict_shifts_from_coeffs(traj, coeffs)
            print(f"[info] Pred shifts (samples): median={np.median(shifts_all):.6g}, "
                  f"p5={np.percentile(shifts_all,5):.6g}, p95={np.percentile(shifts_all,95):.6g}")

            ksp = apply_ro_shift_per_spoke(ksp, shifts_all.astype(np.float64))
            print("[info] Applied fractional RO shift correction to k-space.")

        vol_tag = f"v{vi+1:03d}"
        this_out = out_dir / f"{out_base.name}_{vol_tag}"

        img_base = run_bart_nufft(
            args.bart_bin,
            NX=NX, NY=NY, NZ=NZ,
            traj=traj,
            ksp=ksp,
            out_base=this_out,
            gpu=args.gpu,
            rss=args.rss,
        )
        img_bases.append(img_base)

        if args.export_nifti:
            nii = img_base.with_suffix(".nii.gz")
            voxel = float(args.voxel_mm) if args.voxel_mm and args.voxel_mm > 0 else None
            bart_cfl_to_nifti(img_base, nii, voxel_mm=voxel)
            print(f"[info] Wrote NIfTI: {nii}")

    # If multiple vols, join to 4D stack (dim=3 in BART is the 4th axis in many conventions)
    if len(img_bases) > 1:
        stack_base = out_dir / f"{out_base.name}_4d"
        join_cmd = [args.bart_bin, "join", "3"] + [str(b) for b in img_bases] + [str(stack_base)]
        print("[bart]", " ".join(join_cmd))
        subprocess.run(join_cmd, check=True)

        if args.export_nifti:
            nii = stack_base.with_suffix(".nii.gz")
            voxel = float(args.voxel_mm) if args.voxel_mm and args.voxel_mm > 0 else None
            bart_cfl_to_nifti(stack_base, nii, voxel_mm=voxel)
            print(f"[info] Wrote 4D NIfTI: {nii}")

        print(f"[info] Done. 4D base: {stack_base}")
    else:
        print("[info] Done.")


if __name__ == "__main__":
    main()
