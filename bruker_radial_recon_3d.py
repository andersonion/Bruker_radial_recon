#!/usr/bin/env python3
# Bruker 3D radial recon — v4.3 (complete)
# See header for usage examples.

from __future__ import annotations

import argparse, os, re
from pathlib import Path
import numpy as np
import nibabel as nib
import sigpy as sp
import sigpy.mri as mr
from PIL import Image, ImageDraw, ImageFont

try:
    import cupy as cp  # optional
except Exception:
    cp = None

# ---------- JCAMP parsing ----------
def _parse_jcamp(path: Path) -> dict:
    d, key = {}, None
    if not path.exists():
        return d
    with path.open("r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("$$"):
                continue
            if s.startswith("##$") and "=" in s:
                key, val = s[3:].split("=", 1)
                d[key.strip()] = val.strip()
            elif key:
                d[key] += " " + s
    return d

_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _nums(s: str) -> list[float]:
    if not s: return []
    out = []
    for v in _num_re.findall(s):
        if "." in v or "e" in v.lower():
            out.append(float(v))
        else:
            try: out.append(int(v))
            except ValueError: out.append(float(v))
    return out

def _get_int(d: dict, key: str, default=0) -> int:
    try:
        v = _nums(d.get(key, ""))
        return int(v[0]) if v else default
    except Exception:
        return default

def _get_fov_mm(acqp: dict, method: dict):
    if "PVM_Fov" in method:
        v = np.array(_nums(method["PVM_Fov"]), float)
        if v.size >= 2:
            if v.size == 2: v = np.array([v[0], v[1], v[1]], float)
            return v[:3]
    if "ACQ_fov_cm" in acqp:
        v = np.array(_nums(acqp["ACQ_fov_cm"]), float)
        if v.size >= 2:
            if v.size == 2: v = np.array([v[0], v[1], v[1]], float)
            return v[:3] * 10.0
    if "ACQ_fov_mm" in acqp:
        v = np.array(_nums(acqp["ACQ_fov_mm"]), float)
        if v.size >= 2:
            if v.size == 2: v = np.array([v[0], v[1], v[1]], float)
            return v[:3]
    if "ACQ_fov" in acqp:
        v = np.array(_nums(acqp["ACQ_fov"]), float)
        if v.size >= 2:
            if v.size == 2: v = np.array([v[0], v[1], v[1]], float)
            if np.nanmax(v[:3]) <= 10.0:
                v = v * 10.0
            return v[:3]
    return None

# ---------- Bruker loader ----------
def load_bruker_series(series_dir: str):
    series = Path(series_dir)
    acqp   = _parse_jcamp(series / "acqp")
    method = _parse_jcamp(series / "method")
    visu   = _parse_jcamp(series / "visu_pars")

    dim = int((_nums(acqp.get("ACQ_dim", "")) or [3])[0])
    if dim != 3:
        raise RuntimeError(f"ACQ_dim={dim} (expected 3 for 3D)")

    acq_size = _nums(acqp.get("ACQ_size", ""))
    nspokes_hint = int(acq_size[1]) if len(acq_size) > 1 else None

    rec_sel = (acqp.get("ACQ_ReceiverSelect", "") or "").lower().split()
    ncoils = rec_sel.count("yes") or _get_int(method, "PVM_EncNReceivers", 1) or 1

    fid_path = series / "fid"
    if not fid_path.exists():
        raise RuntimeError(f"Missing FID: {fid_path}")

    fmt  = (acqp.get("GO_raw_data_format") or method.get("GO_raw_data_format") or "GO_32BIT_SGN_INT").strip()
    byto = (acqp.get("BYTORDA") or method.get("BYTORDA") or "little").strip().lower()
    endian = "<" if ("little" in byto or "lsb" in byto) else ">"

    dt_map = {"GO_16BIT_SGN_INT": np.int16, "GO_32BIT_SGN_INT": np.int32, "GO_32BIT_FLOAT": np.float32}
    base = dt_map.get(fmt, np.int32)
    raw  = np.fromfile(fid_path, dtype=endian + np.dtype(base).str[1:])
    if raw.size % 2 != 0:
        raise RuntimeError("Raw FID length is odd; expected interleaved real/imag pairs.")
    ri = raw.reshape(-1, 2).astype(np.float32)
    data = ri[:, 0] + 1j * ri[:, 1]

    complex_samples = data.size
    nread = None
    td = _get_int(acqp, "TD", 0)
    for cand in [td, td // 2]:
        if cand and cand > 8 and (complex_samples % cand == 0):
            nread = cand; break
    if nread is None and nspokes_hint and ncoils:
        NR, NECHOES, NA, NI = (_get_int(acqp, k, 1) or 1 for k in ("NR","NECHOES","NA","NI"))
        denom = nspokes_hint * ncoils * max(1, NR) * max(1, NECHOES) * max(1, NA) * max(1, NI)
        if denom > 0 and complex_samples % denom == 0:
            cand = complex_samples // denom
            if 8 < cand < 8192: nread = int(cand)
    if nread is None:
        for cand in (128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024):
            if complex_samples % cand == 0:
                nread = cand; break
    if nread is None:
        raise RuntimeError("Failed to infer nread.")

    if os.environ.get("RADIAL_FLIP_READ","0") == "1":
        data = data.reshape(-1, nread)[:, ::-1].reshape(-1)

    if data.size % nread != 0:
        raise RuntimeError(f"Readout mismatch: nread={nread}, complex={data.size}")
    vecs = data.reshape(-1, nread)
    total_vecs = vecs.shape[0]
    if total_vecs % ncoils != 0:
        raise RuntimeError(f"Vector count {total_vecs} not divisible by ncoils={ncoils}")
    per_coil = total_vecs // ncoils
    nspokes  = per_coil
    if nspokes_hint and nspokes_hint != nspokes:
        print(f"[warn] ACQ_size hinted nspokes={nspokes_hint}, inferred from FID={nspokes}.")

    arr   = vecs.reshape(ncoils, nspokes, nread)
    kdata = np.transpose(arr, (2, 1, 0)).astype(np.complex64)  # (nread, nspokes, ncoils)

    SW_h  = float((_nums(acqp.get("SW_h", "0")) or [0])[0])
    dwell = (1.0 / SW_h) if SW_h > 0 else None
    TE_s  = float((_nums(acqp.get("ACQ_echo_time", "0")) or [0])[0])
    TR_s  = float((_nums(acqp.get("ACQ_repetition_time", "0")) or [0])[0])
    vox   = np.array(_nums(visu.get("VisuCoreVoxelSize", "")) or [1,1,1], dtype=float)
    fovmm = _get_fov_mm(acqp, method)

    meta = dict(nread=nread, nspokes=nspokes, ncoils=ncoils, SW_h=SW_h, dwell=dwell,
                TE_s=TE_s, TR_s=TR_s, vox=vox, fovmm=fovmm)
    return kdata, meta, dict(acqp=acqp, method=method, visu=visu)

# ---------- Trajectory helpers ----------
def make_ga_traj_3d(nspokes: int, nread: int, center_out: bool = False) -> np.ndarray:
    i = np.arange(nspokes, dtype=np.float64) + 0.5
    phi = (1 + np.sqrt(5)) / 2
    z = 1.0 - 2.0 * i / (nspokes + 1.0)
    r = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    theta = 2.0 * np.pi * i / (phi ** 2)
    dirs = np.stack([r*np.cos(theta), r*np.sin(theta), z], axis=1)  # (nspokes,3)
    kmax = np.pi
    if center_out:
        t = np.linspace(0.0, 1.0, nread, endpoint=False, dtype=np.float64)  # center→edge
    else:
        t = np.linspace(-1.0, 1.0, nread, endpoint=False, dtype=np.float64) # symmetric
    radii = (kmax * t)[:, None, None]
    ktraj = (radii * dirs[None, :, :]).reshape(-1, 3)
    return ktraj.astype(np.float32)

def _read_txt_array(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("//"): continue
            parts = re.split(r"[,\s]+", s)
            try:
                row = [float(p) for p in parts if p != ""]
                if row: rows.append(row)
            except ValueError:
                continue
    return np.array(rows, dtype=np.float64) if rows else np.empty((0,))

def _scale_to_radians(k: np.ndarray, img_shape_xyz: tuple[int,int,int],
                      fov_mm_xyz, vox_mm_xyz):
    if k.size == 0: return k
    Nx, Ny, Nz = img_shape_xyz
    if fov_mm_xyz is not None and len(fov_mm_xyz) >= 3:
        dxi = np.array([fov_mm_xyz[0]/Nx, fov_mm_xyz[1]/Ny, fov_mm_xyz[2]/Nz], float)
    else:
        dxi = np.array(vox_mm_xyz if vox_mm_xyz is not None and len(vox_mm_xyz) >= 3 else [1,1,1], float)
    k = np.asarray(k, float)
    kabs = np.percentile(np.linalg.norm(k, axis=1), 99.0)
    if kabs <= 1.2:
        return k * (2*np.pi)
    elif kabs <= 4.2:
        return k
    else:
        scale_unit = 1.0 if kabs < 1000 else (1/1000.0)
        return (k * scale_unit) * (2*np.pi) * dxi

def load_series_traj(series_dir: str, nread: int, nspokes: int,
                     img_shape_xyz: tuple[int,int,int], fov_mm_xyz, vox_mm_xyz,
                     traj_order: str):
    p = Path(series_dir) / "traj"
    if not p.exists():
        return None
    M = nread * nspokes
    arr = _read_txt_array(p)
    def maybe_reorder(arrM3: np.ndarray) -> np.ndarray:
        if traj_order == "sample":
            return arrM3
        try:
            return arrM3.reshape(nspokes, nread, 3).transpose(1,0,2).reshape(M,3)
        except Exception:
            return arrM3
    if arr.ndim == 2 and arr.size > 0:
        if arr.shape[0] == M and arr.shape[1] >= 3:
            k = maybe_reorder(arr[:, :3])
            return _scale_to_radians(k, img_shape_xyz, fov_mm_xyz, vox_mm_xyz).astype(np.float32)
        if arr.shape[0] == nspokes and arr.shape[1] >= 3:
            dirs = arr[:, :3]
            kmax = np.pi
            t = np.linspace(-1.0, 1.0, nread, endpoint=False)[:, None]
            k = (kmax * t)[:, None] * dirs[None, :, :]
            return k.reshape(-1, 3).astype(np.float32)
        if arr.shape[0] == nspokes and arr.shape[1] == 2:
            th, ph = arr[:,0], arr[:,1]
            dirs = np.stack([np.sin(ph)*np.cos(th), np.sin(ph)*np.sin(th), np.cos(ph)], axis=1)
            kmax = np.pi
            t = np.linspace(-1.0, 1.0, nread, endpoint=False)[:, None]
            k = (kmax * t)[:, None] * dirs[None, :, :]
            return k.reshape(-1, 3).astype(np.float32)
    for dtype in (np.float32, np.float64):
        try:
            flat = np.fromfile(p, dtype=dtype)
        except Exception:
            continue
        if flat.size == M * 3:
            k = maybe_reorder(flat.reshape(M, 3).astype(np.float64))
            return _scale_to_radians(k, img_shape_xyz, fov_mm_xyz, vox_mm_xyz).astype(np.float32)
        if flat.size == nspokes * 3:
            dirs = flat.reshape(nspokes, 3).astype(np.float64)
            kmax = np.pi
            t = np.linspace(-1.0, 1.0, nread, endpoint=False)[:, None]
            k = (kmax * t)[:, None] * dirs[None, :, :]
            return k.reshape(-1, 3).astype(np.float32)
        if flat.size == nspokes * 2:
            ang = flat.reshape(nspokes, 2).astype(np.float64)
            th, ph = ang[:,0], ang[:,1]
            dirs = np.stack([np.sin(ph)*np.cos(th), np.sin(ph)*np.sin(th), np.cos(ph)], axis=1)
            kmax = np.pi
            t = np.linspace(-1.0, 1.0, nread, endpoint=False)[:, None]
            k = (kmax * t)[:, None] * dirs[None, :, :]
            return k.reshape(-1, 3).astype(np.float32)
    print(f"[warn] Could not parse 'traj'; using golden-angle fallback.")
    return None

# ---------- DCF ----------
def compute_dcf(coords: np.ndarray, mode: str = "pipe", os: float = 1.75, width: int = 4):
    if mode == "none":
        return None
    dcf = None
    try:
        if hasattr(mr.dcf, "pipe_menon"):
            dcf = mr.dcf.pipe_menon(coords, niter=30, os=os, width=width)
        elif hasattr(mr.dcf, "pipe_menon_dcf"):
            dcf = mr.dcf.pipe_menon_dcf(coords, max_iter=30, os=os, width=width)
    except Exception:
        dcf = None
    if dcf is None:
        r = np.linalg.norm(coords, axis=1)
        dcf = (r**2 + 1e-6)
    dcf = dcf.astype(np.float32, copy=False)
    dcf *= (dcf.size / (dcf.sum() + 1e-8))
    return dcf

# ---------- Helpers: axes & delay ----------
def apply_axes(coords: np.ndarray, order: str, flips: tuple[bool,bool,bool]):
    assert coords.shape[1] == 3
    m = {'x':0, 'y':1, 'z':2}
    if order not in ("xyz","xzy","yxz","yzx","zxy","zyx"):
        return coords
    idx = [m[c] for c in order]
    out = coords[:, idx].copy()
    for i,do_flip in enumerate(flips):
        if do_flip:
            out[:, i] *= -1.0
    return out

def apply_read_shift(coords: np.ndarray, nread: int, nspokes: int, shift_samples: float):
    if abs(shift_samples) < 1e-9:
        return coords
    delta = float(shift_samples) * (2*np.pi / float(nread))
    uvw = coords.reshape(nread, nspokes, 3)
    dirs = uvw[-1,:,:] - uvw[0,:,:]
    norm = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    dirs = dirs / norm
    uvw = uvw + delta * dirs[None, :, :]
    return uvw.reshape(-1, 3)

def psf_volume(coords: np.ndarray, img_shape_zyx: tuple[int,int,int], os_fac: float = 1.75, width: int = 4):
    ones = np.ones(coords.shape[0], np.complex64)
    psf = sp.nufft_adjoint(ones, coords, oshape=img_shape_zyx, oversamp=os_fac, width=width)
    psf_mag = np.abs(psf); psf_mag /= (psf_mag.max() + 1e-12)
    return psf_mag

def anisotropy_metrics(vol: np.ndarray):
    Z, Y, X = vol.shape
    zc, yc, xc = Z//2, Y//2, X//2
    r = min(Z, Y, X)//6
    sub = vol[zc-r:zc+r+1, yc-r:yc+r+1, xc-r:xc+r+1]
    def second_moment(a, axis):
        idx = np.arange(a.shape[axis]) - a.shape[axis]/2.0
        shape = [1,1,1]; shape[axis] = -1
        w = idx.reshape(shape)
        num = np.sum((w**2) * a); den = np.sum(a) + 1e-12
        return float(num/den)
    vz = second_moment(sub, 0); vy = second_moment(sub, 1); vx = second_moment(sub, 2)
    return vz, vy, vx

def psf_aniso_cost(coords: np.ndarray, img_shape_zyx_small: tuple[int,int,int], os_fac: float, width: int):
    psf = psf_volume(coords, img_shape_zyx_small, os_fac=os_fac, width=width)
    vz, vy, vx = anisotropy_metrics(psf)
    vmean = (vx+vy+vz)/3.0
    cost = abs(vz/vmean-1) + abs(vy/vmean-1) + abs(vx/vmean-1)
    return cost, (vz, vy, vx)

# ---------- Tripanel ----------
def _tripanel_from_volume(vol: np.ndarray, labels=("Axial","Coronal","Sagittal")):
    Z, Y, X = vol.shape
    ax  = vol[Z//2,:,:]
    cor = vol[:,Y//2,:]
    sag = vol[:,:,X//2]
    def to_u8(a):
        vmin = np.percentile(a, 1.0); vmax = np.percentile(a, 99.0)
        a = np.clip((a - vmin) / (vmax - vmin + 1e-12), 0, 1)
        return (a * 255).astype(np.uint8)
    imgs = [Image.fromarray(to_u8(x)) for x in (ax, cor, sag)]
    target_h = max(im.size[1] for im in imgs)
    rs = []
    for im in imgs:
        w, h = im.size
        if h != target_h:
            im = im.resize((int(round(w * (target_h / h))), target_h), Image.BICUBIC)
        rs.append(im)
    margin=20; gap=16; label_h=28
    total_w = margin*2 + sum(im.size[0] for im in rs) + gap*2
    total_h = margin*2 + label_h + target_h
    canvas = Image.new("L", (total_w, total_h), color=16)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        def _text_w(t): return draw.textlength(t, font=font)
    except Exception:
        font = ImageFont.load_default()
        def _text_w(t): 
            bbox = draw.textbbox((0,0), t, font=font)
            return bbox[2]-bbox[0]
    x = margin; y_img = margin + label_h
    for im, lab in zip(rs, labels):
        w, h = im.size
        tw = _text_w(lab); th = font.size
        tx = x + (w - int(tw))//2; ty = margin + (label_h - th)//2
        draw.text((tx+1, ty+1), lab, fill=0, font=font)
        draw.text((tx,   ty),   lab, fill=240, font=font)
        canvas.paste(im, (x, y_img))
        x += w + gap
    return canvas

# ---------- Recon ----------
def recon_adj_sos(kdata: np.ndarray, img_shape_zyx: tuple[int,int,int],
                  coords: np.ndarray, dcf_mode: str = "pipe",
                  os_fac: float = 1.75, width: int = 4,
                  gpu: int = -1):
    nread, nspokes, ncoils = kdata.shape
    assert coords.shape == (nread * nspokes, 3)
    if gpu < 0 or cp is None:
        dcf = compute_dcf(coords, mode=dcf_mode, os=os_fac, width=width)
        coils = []
        for c in range(ncoils):
            y = kdata[:, :, c].reshape(-1)
            if dcf is not None:
                y = y * dcf
            x = sp.nufft_adjoint(y, coords, oshape=img_shape_zyx, oversamp=os_fac, width=width)
            coils.append(x)
        coils = np.stack(coils, axis=0)
        return np.sqrt((np.abs(coils) ** 2).sum(axis=0))
    with sp.Device(gpu):
        coords_g = cp.asarray(coords, dtype=cp.float32)
        dcf = compute_dcf(coords, mode=dcf_mode, os=os_fac, width=width)
        dcf_g = cp.asarray(dcf, dtype=cp.float32) if dcf is not None else None
        coils_g = []
        for c in range(ncoils):
            y_g = cp.asarray(kdata[:, :, c].reshape(-1))
            if dcf_g is not None:
                y_g = y_g * dcf_g
            x_g = sp.nufft_adjoint(y_g, coords_g, oshape=img_shape_zyx, oversamp=os_fac, width=width)
            coils_g.append(x_g)
        coils_g = cp.stack(coils_g, axis=0)
        sos_g = cp.sqrt((cp.abs(coils_g) ** 2).sum(axis=0))
        return cp.asnumpy(sos_g)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Bruker 3D radial recon — v4.3")
    ap.add_argument("--series", required=True, help="Path with acqp/method/visu_pars/fid[/traj]")
    ap.add_argument("--out", required=True, help="Output NIfTI filename")
    ap.add_argument("--matrix", nargs=3, type=int, metavar=("NX","NY","NZ"), required=True)
    ap.add_argument("--spoke-step", dest="spoke_step", type=int, default=1, help="Take every Nth spoke for speed")
    ap.add_argument("--dcf", choices=["none","pipe"], default="pipe", help="Density compensation (pipe=Pipe–Menon)")
    ap.add_argument("--os", dest="os_fac", type=float, default=1.75, help="NUFFT oversampling (default 1.75)")
    ap.add_argument("--width", type=int, default=4, help="NUFFT kernel width (default 4)")
    ap.add_argument("--png", action="store_true", help="Write a single tripanel PNG")
    ap.add_argument("--psf", action="store_true", help="Also save PSF tripanel")
    ap.add_argument("--gpu", type=int, default=-1, help="GPU id to use (e.g., 0). -1 = CPU")
    ap.add_argument("--fov-scale", type=float, default=1.0, help="Scale the effective FOV (e.g., 0.85 to crop by 15%)")
    ap.add_argument("--traj-order", choices=["sample","spoke"], default="sample",
                    help="If 'traj' is per-sample Mx3, specify its layout. 'spoke' = spoke-major on disk.")
    ap.add_argument("--alt-rev", action="store_true", help="Reverse every other spoke's readout (odd spokes)")
    ap.add_argument("--rev-all", action="store_true", help="Reverse all spokes' readouts")
    ap.add_argument("--auto-pi", action="store_true", help="Uniformly scale coords so |k|_p99 = π")
    ap.add_argument("--center-out", action="store_true", help="Use center→edge radii for fallback GA traj")
    ap.add_argument("--axes", default="xyz", help="Permutation of k-axes: one of xyz,xzy,yxz,yzx,zxy,zyx")
    ap.add_argument("--flip-x", action="store_true", help="Flip sign of kx")
    ap.add_argument("--flip-y", action="store_true", help="Flip sign of ky")
    ap.add_argument("--flip-z", action="store_true", help="Flip sign of kz")
    ap.add_argument("--rd-shift", type=float, default=0.0, help="Read-direction shift (in samples)")
    ap.add_argument("--auto-delay", action="store_true", help="Grid-search rd-shift to minimize PSF anisotropy")
    ap.add_argument("--auto-delay-range", nargs=2, type=float, metavar=("MIN","MAX"), default=(-0.6, 0.6),
                    help="Range (samples) for auto-delay search (default ±0.6)")
    ap.add_argument("--auto-delay-steps", type=int, default=25, help="Number of steps for auto-delay (default 25)")
    ap.add_argument("--scale-x", type=float, default=1.0, help="Per-axis k-scale for x")
    ap.add_argument("--scale-y", type=float, default=1.0, help="Per-axis k-scale for y")
    ap.add_argument("--scale-z", type=float, default=1.0, help="Per-axis k-scale for z")
    ap.add_argument("--auto-scale", action="store_true", help="Fit per-axis scales to minimize PSF anisotropy")
    ap.add_argument("--auto-scale-range", nargs=2, type=float, metavar=("MIN","MAX"), default=(0.7, 1.3),
                    help="Range for per-axis scale search (default 0.7..1.3)")
    ap.add_argument("--auto-scale-steps", type=int, default=5, help="Grid steps per axis (default 5)")
    args = ap.parse_args()

    # Device
    use_gpu = (args.gpu >= 0 and cp is not None)
    if use_gpu:
        try:
            ndev = cp.cuda.runtime.getDeviceCount()
            if ndev > args.gpu:
                with sp.Device(args.gpu):
                    props = cp.cuda.runtime.getDeviceProperties(args.gpu)
                    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
                print(f"Using GPU {args.gpu}: {name}")
            else:
                print(f"[warn] Requested GPU {args.gpu} but only {ndev} device(s) found — falling back to CPU.")
                args.gpu = -1
        except Exception as e:
            print(f"[warn] CuPy GPU init failed ({e}) — falling back to CPU.")
            args.gpu = -1
    else:
        if args.gpu >= 0:
            print("[warn] --gpu set but CuPy/CUDA not available — falling back to CPU.")
        print("Using CPU.")

    # Load series
    kdata, meta, hdr = load_bruker_series(args.series)
    nread, nspokes = meta["nread"], meta["nspokes"]
    vox = meta.get("vox", np.array([1,1,1], float))
    fovmm = meta.get("fovmm", None)
    print(f"Loaded: nread={nread}, nspokes={nspokes}, ncoils={meta['ncoils']}, SW_h={meta['SW_h']} Hz")
    if fovmm is not None:
        print(f"Header FOV (mm): {fovmm}")

    # Shapes
    NX, NY, NZ = [int(x) for x in args.matrix]
    img_shape_xyz = (NX, NY, NZ)
    img_shape_zyx = (NZ, NY, NX)

    # Coords
    coords = load_series_traj(args.series, nread, nspokes, img_shape_xyz, fovmm, vox, args.traj_order)
    if coords is None:
        coords = make_ga_traj_3d(nspokes, nread, center_out=args.center_out)

    # Readout direction fixes
    if args.rev_all:
        kdata = kdata[::-1, :, :]
        print("[readout] Reversed all spokes.")
    elif args.alt_rev:
        kdata[:, 1::2, :] = kdata[::-1, 1::2, :]
        print("[readout] Reversed every other (odd) spoke.")

    # Spoke decimation
    if args.spoke_step > 1:
        take = slice(0, nspokes, max(1, args.spoke_step))
        kdata   = kdata[:, take, :]
        coords  = coords.reshape(nread, nspokes, 3)[:, take, :].reshape(-1, 3)
        nspokes = kdata.shape[1]
        print(f"Decimated spokes by {args.spoke_step}: nspokes={nspokes}")

    # FOV crop
    if args.fov_scale != 1.0:
        if args.fov_scale <= 0:
            print("[warn] --fov-scale must be >0. Ignoring.")
        else:
            coords = coords / float(args.fov_scale)
            print(f"[fov] Applying effective FOV scale = {args.fov_scale:.3f} (smaller = more crop)")

    # Axes & flips
    coords = apply_axes(coords, args.axes, (args.flip_x, args.flip_y, args.flip_z))
    if args.axes != "xyz" or args.flip_x or args.flip_y or args.flip_z:
        print(f"[axes] order={args.axes} flips=({args.flip_x},{args.flip_y},{args.flip_z})")

    # Auto-π scaling
    p99 = np.percentile(np.linalg.norm(coords, axis=1), 99.0)
    if args.auto_pi and p99 > 0:
        s = float(np.pi / p99)
        coords = coords * s
        print(f"[auto-pi] Scaled coords by {s:.4f} so |k|_p99 ≈ π.")

    # Delay correction
    if abs(args.rd_shift) > 1e-9:
        coords = apply_read_shift(coords, nread, nspokes, args.rd_shift)
        print(f"[delay] Applied read shift of {args.rd_shift:.3f} samples.")

    # Auto delay grid search
    if args.auto_delay:
        img_shape_zyx_small = tuple(max(64, s//2) for s in img_shape_zyx)
        lo, hi = args.auto_delay_range
        steps = max(3, int(args.auto_delay_steps))
        best_s = 0.0; best_cost = 1e9; best_v = (0,0,0)
        for s in np.linspace(lo, hi, steps):
            c = apply_read_shift(coords, nread, nspokes, s)
            cost, v = psf_aniso_cost(c, img_shape_zyx_small, os_fac=args.os_fac, width=args.width)
            if cost < best_cost:
                best_cost, best_s, best_v = cost, s, v
        coords = apply_read_shift(coords, nread, nspokes, best_s)
        print(f"[auto-delay] Best read shift ≈ {best_s:.3f} samples (cost={best_cost:.4f}).  PSF vars(Z,Y,X)≈{best_v}")

    # Per-axis scaling (manual)
    if any(abs(s-1.0) > 1e-9 for s in (args.scale_x, args.scale_y, args.scale_z)):
        scales = np.array([args.scale_x, args.scale_y, args.scale_z], float)
        coords = coords * scales[None, :]
        print(f"[scale] Applied manual per-axis scales (x,y,z) = {tuple(scales)}")

    # Auto per-axis scaling
    if args.auto_scale:
        img_shape_zyx_small = tuple(max(64, s//2) for s in img_shape_zyx)
        lo, hi = args.auto_scale_range
        steps = max(3, int(args.auto_scale_steps))
        def search_axis(scales, axis):
            candidates = np.linspace(lo, hi, steps)
            best = (1e9, scales[axis])
            for val in candidates:
                test = scales.copy(); test[axis] = val
                c = coords * test[None, :]
                cost, _ = psf_aniso_cost(c, img_shape_zyx_small, os_fac=args.os_fac, width=args.width)
                if cost < best[0]:
                    best = (cost, val)
            scales[axis] = best[1]
            return scales, best[0]
        scales = np.array([1.0, 1.0, 1.0], float)
        for _ in range(2):
            for ax in (0,1,2):
                scales, _ = search_axis(scales, ax)
            span = (hi - lo) * 0.5
            lo, hi = (max(0.3, scales.min()-span/4), scales.max()+span/4)
        coords = coords * scales[None, :]
        print(f"[auto-scale] Best per-axis scales (x,y,z) ≈ ({scales[0]:.3f}, {scales[1]:.3f}, {scales[2]:.3f})")
        print(f"            Re-run tip: --scale-x {scales[0]:.3f} --scale-y {scales[1]:.3f} --scale-z {scales[2]:.3f}")

    # |k| stats
    norm = np.linalg.norm(coords, axis=1)
    p = np.percentile(norm, [0, 50, 95, 99, 100])
    print(f"|k| percentiles (radians): {p} (expect max ≈ π={np.pi:.3f})")

    # Recon
    img = recon_adj_sos(kdata, img_shape_zyx, coords, dcf_mode=args.dcf, os_fac=args.os_fac, width=args.width, gpu=args.gpu)

    # Save NIfTI
    affine = np.diag([vox[0], vox[1], vox[2], 1.0])
    nib.save(nib.Nifti1Image(np.asarray(img, np.float32), affine), args.out)
    print(f"Wrote {args.out}")

    # Tripanel
    if args.png:
        trip = _tripanel_from_volume(img, labels=("Axial","Coronal","Sagittal"))
        trip_path = str(Path(args.out).with_suffix("")) + "_tripanel.png"
        trip.save(trip_path)
        print(f"Saved tripanel PNG → {trip_path}")

    # PSF
    if args.psf:
        psf_mag = psf_volume(coords, img_shape_zyx, os_fac=args.os_fac, width=args.width)
        vz, vy, vx = anisotropy_metrics(psf_mag)
        trip_psf = _tripanel_from_volume(psf_mag, labels=("PSF Axial","PSF Coronal","PSF Sagittal"))
        psf_path = str(Path(args.out).with_suffix("")) + "_psf.png"
        trip_psf.save(psf_path)
        print(f"Saved PSF PNG → {psf_path}")
        vmean = (vx+vy+vz)/3.0
        print(f"[psf] second-moment variances (Z,Y,X) = ({vz:.4f}, {vy:.4f}, {vx:.4f}); anisotropy ratios vs mean = ({vz/vmean:.3f}, {vy/vmean:.3f}, {vx/vmean:.3f})")

if __name__ == "__main__":
    main()
