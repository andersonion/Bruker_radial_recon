#!/usr/bin/env python3
import argparse
import os
import re
import sys
import gzip
import shutil
import subprocess
from pathlib import Path

import numpy as np


# ----------------------------
# Bruker header helpers
# ----------------------------

def read_bruker_param_ints(path, key):
    """
    Robust-ish Bruker param reader.
    Returns list of ints for a key like PVM_Matrix, ACQ_size, PVM_EncNReceivers.
    """
    with open(path, "r", errors="ignore") as f:
        txt = f.read()

    # Capture the block of numeric lines following "##$KEY="
    # until the next "##" or end of file.
    pattern = rf"##\${re.escape(key)}=.*?\n((?:[ \t0-9+\-]+\n)+)"
    m = re.search(pattern, txt, flags=re.MULTILINE | re.DOTALL)
    if not m:
        return None

    block = m.group(1)
    nums = re.findall(r"[-+]?\d+", block)
    if not nums:
        return None
    return [int(x) for x in nums]


def infer_matrix(method_path):
    vals = read_bruker_param_ints(method_path, "PVM_Matrix")
    if not vals or len(vals) < 3:
        # Fall back to your current dataset default instead of dying
        print("[warn] Could not read PVM_Matrix; defaulting to 96x96x96", file=sys.stderr)
        return 96, 96, 96
    NX, NY, NZ = vals[:3]
    print(f"[info] Matrix inferred from PVM_Matrix: {NX}x{NY}x{NZ}")
    return NX, NY, NZ


def infer_true_ro(acqp_path):
    vals = read_bruker_param_ints(acqp_path, "ACQ_size")
    if not vals:
        raise ValueError("Could not read ACQ_size from acqp.")
    ro = vals[0]
    print(f"[info] Readout (true RO) from ACQ_size: RO={ro}")
    return ro


def infer_coils(method_path):
    vals = read_bruker_param_ints(method_path, "PVM_EncNReceivers")
    if not vals:
        print("[warn] Could not read PVM_EncNReceivers; assuming 1 coil.", file=sys.stderr)
        return 1
    coils = vals[0]
    print(f"[info] Coils inferred from PVM_EncNReceivers: {coils}")
    return coils


# ----------------------------
# BART CFL writer
# ----------------------------

def writecfl(base: str, arr: np.ndarray):
    """
    Write a numpy array to BART cfl/hdr pair with name 'base'.
    BART expects column-major dims, but we can stick to the same
    convention as typical Python<->BART wrappers: dims in arr.shape order.
    """
    arr = np.asarray(arr)
    # Convert to complex64; for real arrays, imag part is zero.
    carr = np.asarray(arr, dtype=np.complex64)

    # BART always stores 16 dims in the header
    dims = list(carr.shape)
    if len(dims) > 16:
        raise ValueError("BART supports at most 16 dims")
    dims += [1] * (16 - len(dims))

    hdr_path = base + ".hdr"
    cfl_path = base + ".cfl"

    with open(hdr_path, "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in dims) + "\n")

    # BART expects interleaved real/imag as float32
    # Flatten in C-order (row-major).
    stacked = np.empty(carr.shape + (2,), dtype=np.float32)
    stacked[..., 0] = carr.real
    stacked[..., 1] = carr.imag
    stacked.tofile(cfl_path)


# ----------------------------
# Load Bruker k-space
# ----------------------------

def load_bruker_kspace(fid_path, true_ro, coils):
    """
    Load FID as complex64 and reshape into (RO, spokes, coils).
    Assumes Bruker-style FID (complex pairs).
    """
    raw = np.fromfile(fid_path, dtype=np.int32)  # or int32 depending on scanner
    # FID is typically stored real/imag interleaved
    if raw.size % 2 != 0:
        raise ValueError(f"FID size {raw.size} not divisible by 2 (real/imag).")
    cplx = raw.astype(np.float32).view(np.complex64)

    total = cplx.size
    if total % coils != 0:
        raise ValueError(f"Total complex points {total} not divisible by coils={coils}.")
    per_coil = total // coils

    # stored_ro is padded up to a block size (e.g. 128)
    if per_coil % true_ro != 0:
        raise ValueError(
            f"Could not factor FID into (stored_ro * spokes * coils). "
            f"total={total}, true_ro={true_ro}, coils={coils}"
        )

    spokes = per_coil // true_ro
    stored_ro = true_ro
    print(f"[info] Using RO={true_ro} samples per spoke for recon")
    print(f"[info] FID factorization: RO={true_ro}, Spokes={spokes}, Coils={coils}")

    # Shape into (coils, spokes, RO)
    data = cplx.reshape(coils, spokes, true_ro)
    # Reorder to (RO, spokes, coils)
    data = np.transpose(data, (2, 1, 0))
    # data shape: (RO, spokes, coils)
    return data


# ----------------------------
# Trajectory builders
# ----------------------------

def build_kron_traj(ro, spokes, scale=0.5):
    """
    Build a synthetic 3D 'kron' trajectory with shape (3, ro, spokes).
    Scale is in units of (FOV/2) basically; default 0.5.
    """
    # 2 * pi * normalized radius in [-0.5, 0.5]
    r = np.linspace(-0.5, 0.5, ro, endpoint=False) * (2 * scale)
    # Create spokes on the sphere using golden-angle-ish scheme
    phi = (np.sqrt(5) - 1.0) / 2.0  # golden ratio conjugate
    angles = np.arange(spokes) * 2 * np.pi * phi

    # Simple 3D "stack of rotated spokes" model:
    # one angular coordinate along angle, one along r;
    # we just rotate in 3D using two angles derived from 'angles'.
    theta = angles % np.pi
    psi = (angles * 0.5) % (2 * np.pi)

    # Preallocate
    traj = np.zeros((3, ro, spokes), dtype=np.float32)
    for i in range(spokes):
        ct = np.cos(theta[i])
        st = np.sin(theta[i])
        cp = np.cos(psi[i])
        sp = np.sin(psi[i])
        # direction vector
        dx = ct * cp
        dy = ct * sp
        dz = st
        traj[0, :, i] = r * dx
        traj[1, :, i] = r * dy
        traj[2, :, i] = r * dz

    print(f"[info] Built synthetic kron trajectory (scale={scale}): shape={traj.shape}")
    return traj


def build_linz_traj(ro, spokes, scale=0.5):
    """
    Placeholder for 'linz' style radial 3D traj. For now just calls kron.
    """
    print("[warn] 'linz' trajectory not implemented; using 'kron' instead.")
    return build_kron_traj(ro, spokes, scale=scale)


# ----------------------------
# BART NUFFT / RSS runner
# ----------------------------

def check_bart_has_gpu(bart_cmd):
    """Return True if this BART has GPU support."""
    try:
        # BART doesn't have a clean flag; we just try 'bart nufft -h' and
        # look for '-g' in the help text as a crude test.
        out = subprocess.check_output([bart_cmd, "nufft", "-h"], stderr=subprocess.STDOUT, text=True)
    except Exception:
        return False
    return "-g" in out


def run_bart(
    bart_cmd,
    out_prefix,
    ksp,
    traj,
    matrix,
    spokes_per_frame,
    frame_shift,
    combine_mode="sos",
    qa_first=None,
    export_nifti=False,
    use_gpu=False,
):
    """
    ksp: (RO, spokes, coils)
    traj: (3, RO, spokes)
    """
    NX, NY, NZ = matrix
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    ro, total_spokes, coils = ksp.shape
    assert traj.shape == (3, ro, total_spokes)

    # Sliding window indexing
    frames = []
    start = 0
    while start < total_spokes:
        stop = start + spokes_per_frame
        if stop > total_spokes:
            break
        frames.append((start, stop))
        start += frame_shift

    n_frames = len(frames)
    print(f"[info] Sliding window: spokes-per-frame={spokes_per_frame}, frame-shift={frame_shift}")
    print(f"[info] Will reconstruct {n_frames} frame(s).")

    # GPU sanity check
    gpu_available = check_bart_has_gpu(bart_cmd)
    if use_gpu and not gpu_available:
        print("[warn] BART compiled without GPU support; falling back to CPU and ignoring --gpu.", file=sys.stderr)
        use_gpu = False

    frame_vol_paths = []

    for fi, (s0, s1) in enumerate(frames):
        frame_idx = fi
        # zero-padded index for filenames
        idx_str = f"{frame_idx:05d}"

        frame_base = out_prefix.with_name(out_prefix.name + f"_vol{idx_str}")
        traj_base = out_prefix.with_name(out_prefix.name + f"_vol{idx_str}_traj")
        ksp_base = out_prefix.with_name(out_prefix.name + f"_vol{idx_str}_ksp")
        coil_base = out_prefix.with_name(out_prefix.name + f"_vol{idx_str}_coil")

        coil_cfl = Path(str(coil_base) + ".cfl")

        if coil_cfl.exists():
            print(f"[info] Frame {frame_idx} already reconstructed -> {coil_cfl}, skipping NUFFT/RSS.")
        else:
            # Extract this frame's window
            ksp_win = ksp[:, s0:s1, :]        # (RO, spokes_frame, coils)
            traj_win = traj[:, :, s0:s1]      # (3, RO, spokes_frame)

            # BART wants ksp_dims[0] == 1 (sample dimension is dim1).
            # We'll use dims (1, RO, spokes_frame, coils).
            ksp_bart = ksp_win[np.newaxis, ...]  # (1, RO, spokes, coils)
            writecfl(str(ksp_base), ksp_bart)
            writecfl(str(traj_base), traj_win)

            # NUFFT -> coil image
            cmd = [bart_cmd, "nufft", "-i", "-d", f"{NX}:{NY}:{NZ}", "-t", str(traj_base), str(ksp_base), str(coil_base)]
            if use_gpu:
                # For older BARTs GPU is '-g' directly after 'nufft'
                cmd = [bart_cmd, "nufft", "-i", "-g", "-d", f"{NX}:{NY}:{NZ}", "-t", str(traj_base), str(ksp_base), str(coil_base)]

            print("[bart]", " ".join(cmd))
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[error] BART NUFFT failed for frame {frame_idx} with code {e.returncode}", file=sys.stderr)
                raise

            # Combine coils
            if combine_mode == "sos":
                vol_base = out_prefix.with_name(out_prefix.name + f"_vol{idx_str}")
                cmd = [bart_cmd, "rss", "3", str(coil_base), str(vol_base)]
                print("[bart]", " ".join(cmd))
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"[error] BART RSS failed for frame {frame_idx} with code {e.returncode}", file=sys.stderr)
                    raise
                frame_vol_paths.append(str(vol_base))
                print(f"[info] Frame {frame_idx}/{n_frames - 1} done -> {vol_base}")
            else:
                raise ValueError(f"Unsupported combine mode: {combine_mode}")

    # If nothing reconstructed, nothing more to do
    if not frame_vol_paths:
        print("[warn] No frames reconstructed; nothing to join/export.")
        return

    # Join all into 4D
    stacked_base = out_prefix
    join_cmd = [bart_cmd, "join", "3"] + frame_vol_paths + [str(stacked_base)]
    print("[bart]", " ".join(join_cmd))
    try:
        subprocess.run(join_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[error] BART join failed with code {e.returncode}", file=sys.stderr)
        raise

    if export_nifti:
        # Write .nii and gzip it
        nii_path = str(stacked_base) + ".nii"
        toimg_cmd = [bart_cmd, "toimg", str(stacked_base), nii_path]
        print("[bart]", " ".join(toimg_cmd))
        try:
            subprocess.run(toimg_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[warn] BART toimg failed on 4D volume: {e}", file=sys.stderr)
        else:
            # gzip
            gz_path = nii_path + ".gz"
            with open(nii_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(nii_path)
            print(f"[info] Gzipped 4D NIfTI -> {gz_path}")

    # QA export (first N frames)
    if qa_first is not None and qa_first > 0:
        qa_count = min(qa_first, len(frame_vol_paths))
        qa_base = out_prefix.with_name(out_prefix.name + f"_QA_first{qa_count}")
        qa_join_cmd = [bart_cmd, "join", "3"] + frame_vol_paths[:qa_count] + [str(qa_base)]
        print("[bart]", " ".join(qa_join_cmd))
        try:
            subprocess.run(qa_join_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[error] BART join for QA failed with code {e.returncode}", file=sys.stderr)
        else:
            if export_nifti:
                qa_nii = str(qa_base) + ".nii"
                qa_toimg_cmd = [bart_cmd, "toimg", str(qa_base), qa_nii]
                print("[bart]", " ".join(qa_toimg_cmd))
                try:
                    subprocess.run(qa_toimg_cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"[warn] BART toimg failed on QA volume: {e}", file=sys.stderr)
                else:
                    qa_gz = qa_nii + ".gz"
                    with open(qa_nii, "rb") as f_in, gzip.open(qa_gz, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(qa_nii)
                    print(f"[info] QA NIfTI (first {qa_count} frame(s)) -> {qa_gz}")

    print(f"[info] All requested frames complete; 4D result at {stacked_base}")


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", required=True, help="Bruker series path (contains fid, acqp, method)")
    parser.add_argument("--out", required=True, help="Output prefix for BART recon")
    parser.add_argument("--matrix", nargs=3, type=int, help="Override matrix NX NY NZ")
    parser.add_argument("--readout", type=int, help="Override readout points (true RO)")
    parser.add_argument("--coils", type=int, help="Override number of coils")
    parser.add_argument("--traj-mode", choices=["kron", "linz"], default="kron")
    parser.add_argument("--spokes-per-frame", type=int, default=200)
    parser.add_argument("--frame-shift", type=int, default=50)
    parser.add_argument("--test-volumes", type=int, help="(Unused placeholder)")
    parser.add_argument("--fid-dtype", default="int32", help="(Unused placeholder, kept for compatibility)")
    parser.add_argument("--fid-endian", default="little", help="(Unused placeholder, kept for compatibility)")
    parser.add_argument("--combine", choices=["sos"], default="sos")
    parser.add_argument("--qa-first", type=int, default=None, help="Export QA NIfTI of first N frames")
    parser.add_argument("--export-nifti", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="Try to use BART GPU NUFFT")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    series = Path(args.series)
    fid_path = series / "fid"
    acqp_path = series / "acqp"
    method_path = series / "method"

    if not fid_path.exists():
        raise FileNotFoundError(f"Missing fid: {fid_path}")
    if not acqp_path.exists():
        raise FileNotFoundError(f"Missing acqp: {acqp_path}")
    if not method_path.exists():
        raise FileNotFoundError(f"Missing method: {method_path}")

    # Matrix
    if args.matrix:
        NX, NY, NZ = args.matrix
        print(f"[info] Matrix overridden from CLI: {NX}x{NY}x{NZ}")
    else:
        NX, NY, NZ = infer_matrix(method_path)

    # Coils
    if args.coils:
        coils = args.coils
        print(f"[info] Coils overridden from CLI: {coils}")
    else:
        coils = infer_coils(method_path)

    # True RO
    if args.readout:
        true_ro = args.readout
        print(f"[info] Readout overridden from CLI: RO={true_ro}")
    else:
        true_ro = infer_true_ro(acqp_path)

    # Load k-space
    ksp = load_bruker_kspace(str(fid_path), true_ro=true_ro, coils=coils)
    ro, spokes, coils2 = ksp.shape
    if coils2 != coils:
        print(f"[warn] Coil count mismatch after reshape: expected {coils}, got {coils2}", file=sys.stderr)

    # Build trajectory
    if args.traj_mode == "kron":
        traj = build_kron_traj(ro, spokes, scale=0.5)
    else:
        traj = build_linz_traj(ro, spokes, scale=0.5)

    bart_cmd = "/home/apps/bart/bart"

    run_bart(
        bart_cmd=bart_cmd,
        out_prefix=args.out,
        ksp=ksp,
        traj=traj,
        matrix=(NX, NY, NZ),
        spokes_per_frame=args.spokes_per_frame,
        frame_shift=args.frame_shift,
        combine_mode=args.combine,
        qa_first=args.qa_first,
        export_nifti=args.export_nifti,
        use_gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
