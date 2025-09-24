#!/usr/bin/env python3
"""
bruker_radial_recon_3d.py

Minimal but production-leaning 3D radial (UTE/FLASH) reconstruction for Bruker PV6+ series.
- Reads raw FID + headers via `brkraw`
- Parses dims from acqp (ACQ_dim=3, ACQ_size -> nread, nspokes)
- Builds 3D radial trajectory (spiral phyllotaxis / golden-section)
- Iterative Pipe–Menon 3D DCF
- ESPIRiT sensitivity estimation (from low-iter L2 recon)
- L2-SENSE CG reconstruction
- Writes NIfTI (float32 magnitude) and optional PNGs of orthogonal slices for QA

Requirements:
    pip install sigpy brkraw nibabel numpy pillow
    # Optional GPU: pip install cupy-cuda12x

Usage:
    python bruker_radial_recon_3d.py \
        --series /path/to/Bruker/4 \
        --out recon3d_radial.nii.gz \
        --matrix 192 192 192 \
        --traj ga

Notes:
- Prefer the TRUE trajectory if your method/sequence exports it; hook provided in `build_traj_from_method()`.
- For very large spoke counts, you can subset spokes (temporal or angle decimation) via `--spoke-step`.
- Gradient-delay correction is sequence/scanner-specific; placeholder included.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import nibabel as nib

try:
    import brkraw
except ImportError as e:
    raise SystemExit("Missing dependency: pip install brkraw")

# ---------------- Bruker loader ---------------- #

def load_bruker_series(series_dir: str):
    ser = brkraw.RawBruker(series_dir)
    acqp = ser.get_dict("acqp")
    method = ser.get_dict("method") if ser.has_method() else {}
    visu = ser.get_dict("visu_pars") if ser.has_visu_pars() else {}

    dim = int(acqp.get("ACQ_dim", 3))
    if dim != 3:
        raise RuntimeError(f"ACQ_dim={dim} (expected 3 for 3D)")

    # ACQ_size often stores [nread, nprojections, 1]
    acq_size = acqp.get("ACQ_size")
    if isinstance(acq_size, (list, tuple)) and len(acq_size) >= 2:
        nread, nspokes = int(acq_size[0]), int(acq_size[1])
    else:
        raise RuntimeError("Could not parse ACQ_size from acqp.")

    # Coils
    rec_sel = acqp.get("ACQ_ReceiverSelect")
    if isinstance(rec_sel, (list, tuple)):
        ncoils = sum(str(x).strip().lower() == 'yes' for x in rec_sel)
    elif isinstance(rec_sel, str):
        ncoils = sum(t.strip().lower() == 'yes' for t in rec_sel.split())
    else:
        ncoils = int(acqp.get("RG", 1))  # fallback

    SW_h = float(acqp.get("SW_h", 0.0))
    dwell = 1.0 / SW_h if SW_h > 0 else None
    TE_s = float((acqp.get("ACQ_echo_time", [0.0])[0] if isinstance(acqp.get("ACQ_echo_time"), list) else acqp.get("ACQ_echo_time", 0.0)))
    TR_s = float((acqp.get("ACQ_repetition_time", [0.0])[0] if isinstance(acqp.get("ACQ_repetition_time"), list) else acqp.get("ACQ_repetition_time", 0.0)))

    # Raw complex data: typically (nspokes*ncoils, nread)
    raw = np.asarray(ser.get_fid())
    if raw.ndim != 2:
        raise RuntimeError(f"Unexpected raw ndim={raw.ndim}")

    # Arrange to (nread, nspokes, ncoils)
    if raw.shape[1] == nread:
        vecs = raw
    elif raw.shape[0] == nread:
        vecs = raw.T
    else:
        raise RuntimeError(f"Cannot match readout length: raw shape {raw.shape}, nread={nread}")

    total_vecs = vecs.shape[0]
    if total_vecs % ncoils != 0:
        raise RuntimeError(f"Vector count {total_vecs} not divisible by ncoils={ncoils}")
    per_coil = total_vecs // ncoils

    # Assume spoke-major ordering per coil
    if per_coil != nspokes:
        # Some sequences oversample/spread; try to infer
        nspokes = per_coil
        print(f"[warn] Adjusting nspokes to {nspokes} based on raw vectors.")
    arr = vecs.reshape(ncoils, nspokes, nread)
    kdata = np.transpose(arr, (2, 1, 0)).astype(np.complex64)  # (nread, nspokes, ncoils)

    meta = dict(nread=nread, nspokes=nspokes, ncoils=ncoils, SW_h=SW_h, dwell=dwell,
                TE_s=TE_s, TR_s=TR_s, fov=np.array(visu.get("VisuCoreExtent", [0,0,0]), float),
                vox=np.array(visu.get("VisuCoreVoxelSize", [1,1,1]), float))
    return kdata, meta, dict(acqp=acqp, method=method, visu=visu)

# ---------------- Trajectory ---------------- #

def build_traj_ga_3d(nread: int, nspokes: int) -> np.ndarray:
    kr = np.linspace(-np.pi, np.pi, nread, endpoint=False, dtype=np.float32)
    i = np.arange(nspokes, dtype=np.float32) + 0.5
    phi = np.arccos(1 - 2*i / nspokes)
    theta = np.pi * (1 + 5**0.5) * i
    dx = np.sin(phi) * np.cos(theta)
    dy = np.sin(phi) * np.sin(theta)
    dz = np.cos(phi)
    kx = (kr[:, None] * dx[None, :]).ravel()
    ky = (kr[:, None] * dy[None, :]).ravel()
    kz = (kr[:, None] * dz[None, :]).ravel()
    return np.stack([kx, ky, kz], axis=-1).astype(np.float32)


def build_traj_from_method(method: Dict, nread: int, nspokes: int) -> Optional[np.ndarray]:
    # Placeholder: if method contains direction cosines or spherical angles per projection
    return None

# ---------------- DCF ---------------- #

def make_dcf(coord: np.ndarray, niter: int = 30) -> np.ndarray:
    return mr.dcf.pipe_menon(coord, niter=niter).astype(np.float32)

# ---------------- Sensitivity + Recon ---------------- #

def estimate_sens_3d(kdata: np.ndarray, coord: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    A = mr.linop.NUFFT(img_shape, coord)
    dcf = make_dcf(coord, niter=10)
    y = (dcf**0.5)[:, None] * kdata.reshape(-1, kdata.shape[-1])
    coil_imgs = []
    for c in range(kdata.shape[-1]):
        x = sp.alg.cg(A.H @ A, A.H @ y[:, c], max_iter=10, tol=1e-4)
        coil_imgs.append(x.reshape(img_shape))
    coil_stack = np.stack(coil_imgs, axis=-1)
    smap = mr.app.EspiritCalib(coil_stack, calib_width=12, thresh=0.02, max_iter=30).run()
    return smap[..., 0]


def recon_l2_sense_3d(kdata: np.ndarray, coord: np.ndarray, smap: np.ndarray, img_shape: Tuple[int,int,int], lam: float=0.0) -> np.ndarray:
    S = sp.linop.Multiply(img_shape + (kdata.shape[-1],), smap)
    F = mr.linop.NUFFT(img_shape, coord)
    A = F @ S

    dcf = make_dcf(coord, niter=30)
    y = kdata.reshape(-1, kdata.shape[-1])
    W = sp.linop.Diag(np.repeat(dcf[:, None], y.shape[1], axis=1).ravel()**0.5)
    AH = A.H
    lhs = AH @ W.H @ W @ A + lam * sp.linop.Identity(img_shape)
    rhs = (AH @ W.H @ (W @ y.ravel())).reshape(img_shape)
    x = sp.alg.cg(lhs, rhs, max_iter=40, tol=1e-5)
    return x

# ---------------- NIfTI I/O ---------------- #

def save_nifti(vol: np.ndarray, vox_mm: np.ndarray, out_path: str):
    vol = np.asarray(vol, dtype=np.float32)
    affine = np.diag([
        vox_mm[0] if len(vox_mm)>0 else 1.0,
        vox_mm[1] if len(vox_mm)>1 else 1.0,
        vox_mm[2] if len(vox_mm)>2 else 1.0,
        1.0
    ]).astype(np.float32)
    nib.save(nib.Nifti1Image(vol, affine), str(out_path))

# ---------------- Main ---------------- #

def main():
    ap = argparse.ArgumentParser(description="3D radial Bruker reconstruction (SigPy)")
    ap.add_argument("--series", required=True, help="Path to Bruker series dir (contains fid, acqp)")
    ap.add_argument("--out", required=True, help="Output NIfTI path (e.g., recon3d.nii.gz)")
    ap.add_argument("--matrix", type=int, nargs=3, default=[192,192,192], help="Reconstruction matrix (Nx Ny Nz)")
    ap.add_argument("--traj", choices=["ga","method"], default="ga")
    ap.add_argument("--spoke-step", type=int, default=1, help="Use every k-th spoke to reduce load")
    ap.add_argument("--lam", type=float, default=0.0, help="L2 reg (Tikhonov)")
    args = ap.parse_args()

    kdata, meta, hdr = load_bruker_series(args.series)
    nread, nspokes, ncoils = meta["nread"], meta["nspokes"], meta["ncoils"]

    # Spoke decimation if desired
    take = slice(0, nspokes, max(1, args.spoke_step))
    kdata = kdata[:, take, :]
    nspokes = kdata.shape[1]

    if args.traj == "method":
        coord = build_traj_from_method(hdr.get("method", {}), nread, nspokes)
        if coord is None:
            print("[warn] No method trajectory found; falling back to golden-section.")
            coord = build_traj_ga_3d(nread, nspokes)
    else:
        coord = build_traj_ga_3d(nread, nspokes)

    img_shape = tuple(args.matrix)
    smap = estimate_sens_3d(kdata, coord, img_shape)
    vol = recon_l2_sense_3d(kdata, coord, smap, img_shape, lam=args.lam)

    vol = np.abs(vol)
    vox = meta.get("vox", np.array([1,1,1], dtype=float))
    save_nifti(vol, vox, args.out)
    print(f"Saved NIfTI → {args.out}")

    # Optional quicklook PNGs
    try:
        from PIL import Image
        outbase = Path(args.out).with_suffix("").with_suffix("")
        outdir = Path(str(outbase) + "_png")
        outdir.mkdir(parents=True, exist_ok=True)
        zc, yc, xc = [s//2 for s in vol.shape]
        for plane, sl in ("ax", vol[zc,:,:]), ("cor", vol[:,yc,:]), ("sag", vol[:,:,xc]):
            im = sl / (sl.max() + 1e-8)
            im = (im * 255.0).astype(np.uint8)
            Image.fromarray(im).save(outdir / f"{plane}.png")
        print(f"Saved PNGs → {outdir}")
    except Exception as e:
        print(f"[warn] PNG export failed: {e}")

if __name__ == "__main__":
    main()
