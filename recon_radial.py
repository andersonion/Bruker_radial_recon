# recon_radial.py
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import nibabel as nib
from pathlib import Path

# ---------- 1) BRUKER LOADER ----------
def load_bruker_raw(series_dir: str):
    """
    Returns:
      kdata: complex64, shape (nread, nspokes, ncoils)
      meta: dict with keys {fov, vox, dwell, base_res, nspokes, ncoils, ndim, is3d, TE, TR}
      hdr:  raw header dicts if you want deeper stuff
    """
    try:
        import brkraw  # pip install brkraw
    except ImportError as e:
        raise RuntimeError("pip install brkraw to read Bruker ParaVision raw") from e

    ser = brkraw.RawBruker(series_dir)
    acqp = ser.get_dict("acqp")
    method = ser.get_dict("method")
    visu = ser.get_dict("visu_pars")

    # Bruker raw data (FID)
    raw = ser.get_fid()  # shape typically (nspokes*ncoils, nread) or similar; varies by sequence
    raw = np.asarray(raw)

    # Heuristic reshaping — adapt if your sequence packages differently:
    ncoils = int(acqp.get("ACQ_spatial_size_2", [1])[0]) if "ACQ_spatial_size_2" in acqp else int(acqp.get("NR", 1))
    nread  = int(acqp.get("ACQ_size", [0, 0])[0])
    nspokes = raw.shape[0] // ncoils
    kdata = raw.reshape(nspokes, ncoils, nread).transpose(2, 0, 1).astype(np.complex64)  # (nread, nspokes, ncoils)

    # FOV & voxel sizes (mm)
    fov = np.array(visu.get("VisuCoreExtent", [1, 1, 1]), dtype=float)  # mm
    vox = np.array(visu.get("VisuCoreVoxelSize", [1, 1, 1]), dtype=float)  # mm
    dwell = 1.0 / float(acqp.get("SW_h", 1.0))  # seconds per sample (≈ 1/BW)
    base_res = int(method.get("PVM_Matrix", [nread, 1, 1])[0])
    TE = float(acqp.get("ACQ_echo_time", 0)) * 1e3  # ms
    TR = float(acqp.get("ACQ_repetition_time", 0)) * 1e3  # ms

    # Determine 2D vs 3D
    dim = int(visu.get("VisuCoreDim", 2))
    is3d = True if dim == 3 else False

    meta = dict(fov=fov, vox=vox, dwell=dwell, base_res=base_res,
                nspokes=nspokes, ncoils=ncoils, ndim=dim, is3d=is3d, TE=TE, TR=TR)
    return kdata, meta, dict(acqp=acqp, method=method, visu=visu)

# ---------- 2) TRAJECTORY ----------
def make_radial_traj(nread, nspokes, is3d, golden=True, fov_mm=(220,220,220)):
    """
    Returns coords with shape (..., dim) in radians per FOV for SigPy NUFFT.
    2D: shape (nread*nspokes, 2)
    3D: shape (nread*nspokes, 3)
    """
    # sample positions along readout: normalized from -pi..pi
    kr = np.linspace(-np.pi, np.pi, nread, endpoint=False, dtype=np.float32)

    if not is3d:
        # angles for spokes
        if golden:
            ga = np.pi * (3 - np.sqrt(5))  # ~111.246°
            th = (np.arange(nspokes) * ga) % np.pi
        else:
            th = np.linspace(0, np.pi, nspokes, endpoint=False, dtype=np.float32)
        kx = (kr[:, None] * np.cos(th)[None, :]).ravel()
        ky = (kr[:, None] * np.sin(th)[None, :]).ravel()
        coord = np.stack([kx, ky], axis=-1)
    else:
        # 3D: spiral phyllotaxis (Koay; golden section) for directions
        # unit directions on sphere:
        i = np.arange(nspokes, dtype=np.float32) + 0.5
        phi = np.arccos(1 - 2*i / nspokes)    # polar
        theta = np.pi * (1 + 5**0.5) * i      # azimuth
        dx = np.sin(phi) * np.cos(theta)
        dy = np.sin(phi) * np.sin(theta)
        dz = np.cos(phi)
        # scale by readout
        kx = (kr[:, None] * dx[None, :]).ravel()
        ky = (kr[:, None] * dy[None, :]).ravel()
        kz = (kr[:, None] * dz[None, :]).ravel()
        coord = np.stack([kx, ky, kz], axis=-1).astype(np.float32)

    return coord.astype(np.float32)

# ---------- 3) DENSITY COMPENSATION ----------
def make_dcf(coord, niter=30):
    # Pipe–Menon iterative dcf in SigPy
    return mr.dcf.pipe_menon(coord, niter=niter).astype(np.float32)

# ---------- 4) SENSITIVITY MAPS ----------
def estimate_sens(kdata, coord, img_shape, dcf=None):
    # Low-res recon for calibration (heavier apodization)
    A = mr.linop.NUFFT(img_shape, coord)
    if dcf is None:
        dcf = np.ones(coord.shape[0], np.float32)
    w = sp.linop.Diag(dcf**0.5)
    # Solve coil images with simple CG (per coil)
    coil_imgs = []
    for c in range(kdata.shape[-1]):
        y = (dcf**0.5)[:, None] * kdata.reshape(-1, kdata.shape[-1])[:, c]
        x = sp.alg.cg(A.H @ w.H @ w @ A, A.H @ w.H @ y, max_iter=20, tol=1e-4)
        coil_imgs.append(x.reshape(img_shape))
    coil_imgs = np.stack(coil_imgs, axis=-1)  # (..., coils)

    # ESPIRiT via SigPy
    smap = mr.app.EspiritCalib(coil_imgs, calib_width=24, thresh=0.02, max_iter=30).run()
    # choose first map set
    return smap[..., 0]

# ---------- 5) L2-SENSE RECON ----------
def recon_l2_sense(kdata, coord, smap, img_shape, dcf=None, lam=0.0, max_iter=40):
    # NUFFT with coil sensitivities
    S = sp.linop.Multiply(img_shape + (kdata.shape[-1],), smap)
    F = mr.linop.NUFFT(img_shape, coord)
    A = F @ S

    y = kdata.reshape(-1, kdata.shape[-1])
    if dcf is None:
        dcf = np.ones(coord.shape[0], np.float32)
    W = sp.linop.Diag(np.repeat(dcf[:, None], y.shape[1], axis=1).ravel()**0.5)

    # Normal equations: (A^H W^H W A + lam*I) x = A^H W^H W y
    AH = A.H
    lhs = AH @ W.H @ W @ A + lam * sp.linop.Identity(img_shape)
    rhs = (AH @ W.H @ (W @ y.ravel())).reshape(img_shape)

    x = sp.alg.cg(lhs, rhs, max_iter=max_iter, tol=1e-5)
    return x

# ---------- 6) SAVE NIFTI ----------
def save_nifti(img, vox_mm, out_path):
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:  # make it 3D for NIfTI
        img = img[None, ...]
    affine = np.diag([vox_mm[0], vox_mm[1], vox_mm[2] if len(vox_mm) > 2 else 1.0, 1.0]).astype(np.float32)
    nib.save(nib.Nifti1Image(img, affine), str(out_path))

# ---------- MAIN ----------
def run(series_dir, out_nii, target_matrix=(256,256,256), is3d=True):
    kdata, meta, _hdr = load_bruker_raw(series_dir)
    nread, nspokes, ncoils = kdata.shape
    if not is3d:
        img_shape = (target_matrix[0], target_matrix[1])
    else:
        # pick reasonable z based on FOV ratio
        fov = meta["fov"]
        zdim = int(round(target_matrix[0] * (fov[2]/max(fov[0], fov[1]))))
        img_shape = (target_matrix[0], target_matrix[1], max(8, zdim))

    coord = make_radial_traj(nread, nspokes, is3d=is3d)
    dcf = make_dcf(coord, niter=30)

    smap = estimate_sens(kdata, coord, img_shape, dcf=dcf)
    img = recon_l2_sense(kdata, coord, smap, img_shape, dcf=dcf, lam=0.0, max_iter=40)

    # intensity normalize (optional)
    img = np.abs(img)
    vox = meta["vox"] if meta["is3d"] else np.r_[meta["vox"][:2], [1.0]]
    save_nifti(img, vox, out_nii)
    print(f"Saved: {out_nii}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True, help="Path to Bruker series dir containing fid/acqp/method/visu_pars")
    ap.add_argument("--out", required=True, help="Output NIfTI path")
    ap.add_argument("--matrix", type=int, nargs="+", default=[256,256,256])
    ap.add_argument("--is3d", action="store_true")
    args = ap.parse_args()
    run(args.series, args.out, tuple(args.matrix + [0]*(3-len(args.matrix))), is3d=args.is3d)
