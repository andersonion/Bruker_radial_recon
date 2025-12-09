#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import numpy as np
import nibabel as nib
import re
from pathlib import Path

BART = "/home/apps/bart/bart"

# -------------------------------
# Bruker parsing
# -------------------------------

def read_param(path, key):
    txt = open(path).read()
    m = re.search(rf"##\\${key}=\\( ?\\d+ ?\\)\\n([\\d\\s]+)", txt)
    if not m:
        return None
    return list(map(int, m.group(1).split()))

def infer_matrix(method):
    m = read_param(method, "PVM_Matrix")
    if not m:
        raise ValueError("Could not infer PVM_Matrix")
    return tuple(m)

def infer_coils(method):
    c = read_param(method, "PVM_EncNReceivers")
    if not c:
        raise ValueError("Could not infer PVM_EncNReceivers")
    return int(c[0])

def infer_true_ro(acqp):
    a = read_param(acqp, "ACQ_size")
    if not a:
        raise ValueError("Could not infer ACQ_size")
    return int(a[0])

# -------------------------------
# FID loader with Bruker padding
# -------------------------------

def load_bruker_kspace(fid_path, true_ro, coils, dtype=np.complex64):
    raw = np.fromfile(fid_path, dtype=np.complex64)
    stored_ro = int(np.ceil(true_ro / 128) * 128)

    spokes = raw.size // (stored_ro * coils)
    if stored_ro * spokes * coils != raw.size:
        raise ValueError("FID size does not match padded RO blocks")

    ksp = raw.reshape(coils, spokes, stored_ro)
    ksp = ksp[..., :true_ro]         # trim padded RO
    ksp = np.transpose(ksp, (2, 1, 0))  # [RO, spokes, coils]

    # BART requires [1, RO, spokes, coils]
    ksp = ksp[np.newaxis, ...]
    return ksp

# -------------------------------
# Trajectory
# -------------------------------

def build_kron_traj(ro, spokes, scale):
    i = np.arange(spokes)
    phi = (np.sqrt(5) - 1) / 2
    ang = 2 * np.pi * i * phi
    z = 2 * i / spokes - 1
    r = np.sqrt(1 - z**2)
    traj = np.zeros((3, ro, spokes), np.float32)
    traj[0] = r * np.cos(ang)
    traj[1] = r * np.sin(ang)
    traj[2] = z
    return traj * scale

# -------------------------------
# BART runner (with GPU fallback)
# -------------------------------

def run_bart(cmd):
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        if "-g" in cmd:
            print("[warn] BART has no GPU support; retrying CPU")
            cmd.remove("-g")
            subprocess.run(cmd, check=True)
        else:
            raise

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spokes-per-frame", type=int, required=True)
    ap.add_argument("--frame-shift", type=int, required=True)
    ap.add_argument("--traj-mode", choices=["kron", "linz"], default="kron")
    ap.add_argument("--traj-scale", type=float, default=48.0)
    ap.add_argument("--combine", choices=["sos"], default="sos")
    ap.add_argument("--qa-first", type=int, default=0)
    ap.add_argument("--export-nifti", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    series = Path(args.series)
    method = series / "method"
    acqp = series / "acqp"
    fid = series / "fid"

    NX, NY, NZ = infer_matrix(method)
    coils = infer_coils(method)
    true_ro = infer_true_ro(acqp)

    print(f"[info] Matrix: {NX}x{NY}x{NZ}")
    print(f"[info] Coils: {coils}")
    print(f"[info] True RO: {true_ro}")

    ksp = load_bruker_kspace(fid, true_ro, coils)
    _, RO, spokes, _ = ksp.shape

    print(f"[info] Loaded k-space: RO={RO}, Spokes={spokes}")

    traj = build_kron_traj(RO, spokes, args.traj_scale)

    spf = args.spokes_per_frame
    shift = args.frame_shift

    frames = list(range(0, spokes - spf + 1, shift))
    n_frames = len(frames)
    print(f"[info] Will reconstruct {n_frames} frames")

    base = args.out

    for i, s in enumerate(frames):
        tag = f"{base}_vol{i:05d}"
        coil_file = Path(tag + "_coil.cfl")
        if coil_file.exists():
            print(f"[info] Frame {i} exists, skipping")
            continue

        sub_ksp = ksp[:, :, s:s+spf, :]
        sub_traj = traj[:, :, s:s+spf]

        sub_ksp.astype(np.complex64).tofile(tag + "_ksp")
        sub_traj.astype(np.float32).tofile(tag + "_traj")

        cmd = [BART]
        if args.gpu:
            cmd.append("-g")

        cmd += [
            "nufft", "-i", "-d", f"{NX}:{NY}:{NZ}",
            "-t", tag + "_traj", tag + "_ksp", tag + "_coil"
        ]

        run_bart(cmd)

        if args.combine == "sos":
            run_bart([BART, "rss", "3", tag + "_coil", tag])

        print(f"[info] Frame {i}/{n_frames-1} done")

    # ---- JOIN ----
    vol_list = [f"{base}_vol{i:05d}" for i in range(n_frames)]
    joined = base + "_4d"
    run_bart([BART, "join", "3", *vol_list, joined])

    # ---- NIFTI ----
    if args.export_nifti:
        nii = joined + ".nii"
        run_bart([BART, "toimg", joined, nii])
        subprocess.run(["gzip", "-f", nii])

    # ---- QA ----
    if args.qa_first > 0:
        qa_list = [f"{base}_vol{i:05d}" for i in range(args.qa_first)]
        qa_join = base + f"_QA_first{args.qa_first}"
        run_bart([BART, "join", "3", *qa_list, qa_join])
        qa_nii = qa_join + ".nii"
        run_bart([BART, "toimg", qa_join, qa_nii])
        subprocess.run(["gzip", "-f", qa_nii])

    print("[info] DONE")

if __name__ == "__main__":
    main()
