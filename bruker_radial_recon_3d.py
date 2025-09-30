#!/usr/bin/env python3
# Bruker 3D radial recon — v4.7 (production auto-cal + ring control)
# Deps: pip install sigpy nibabel numpy pillow
# New flags:
#   --prod (coarse 64³ auto-delay+auto-scale, then final recon at --matrix)
#   --gpu-psf (use GPU for PSF during auto-* searches)
#   --demean-spoke (subtract per-spoke DC)
#   --clip-pi (clip |k| to <= π)
#   --apod {none,hann,tukey,kaiser,gauss}, with --tukey-alpha/--kaiser-beta/--gauss-sigma

from __future__ import annotations
import argparse, os, re, json
from pathlib import Path
import numpy as np
import nibabel as nib
import sigpy as sp
import sigpy.mri as mr
from PIL import Image, ImageDraw, ImageFont

try:
    import cupy as cp
except Exception:
    cp = None

# ---------------- JCAMP parsing ----------------
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
    for k,scale in (("PVM_Fov",1.0),("ACQ_fov_mm",1.0),("ACQ_fov_cm",10.0),("ACQ_fov",10.0)):
        if k in method or k in acqp:
            src = method if k in method else acqp
            v = np.array(_nums(src[k]), float)
            if v.size == 2: v = np.array([v[0], v[1], v[1]], float)
            if v.size >= 3:
                v = v[:3] * scale
                return v
    return None

# ---------------- Bruker loader ----------------
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

    # Infer nread
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

    # Shape (nread, nspokes, ncoils)
    vecs = data.reshape(-1, nread)
    total_vecs = vecs.shape[0]
    if total_vecs % ncoils != 0:
        raise RuntimeError(f"Vector count {total_vecs} not divisible by ncoils={ncoils}")
    per_coil = total_vecs // ncoils
    nspokes  = per_coil
    if nspokes_hint and nspokes_hint != nspokes:
        print(f"[warn] ACQ_size hinted nspokes={nspokes_hint}, inferred from FID={nspokes}.")

    arr   = vecs.reshape(ncoils, nspokes, nread)
    kdata = np.transpose(arr, (2, 1, 0)).astype(np.complex64)

    SW_h  = float((_nums(acqp.get("SW_h", "0")) or [0])[0])
    vox   = np.array(_nums(visu.get("VisuCoreVoxelSize", "")) or [1,1,1], dtype=float)
    fovmm = _get_fov_mm(acqp, method)

    meta = dict(nread=nread, nspokes=nspokes, ncoils=ncoils, SW_h=SW_h, vox=vox, fovmm=fovmm)
    return kdata, meta, dict(acqp=acqp, method=method, visu=visu)

# -------------- Trajectory (GA fallback) --------------
def make_ga_traj_3d(nspokes: int, nread: int) -> np.ndarray:
    i = np.arange(nspokes, dtype=np.float64) + 0.5
    phi = (1 + np.sqrt(5)) / 2
    z = 1.0 - 2.0 * i / (nspokes + 1.0)
    r = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    theta = 2.0 * np.pi * i / (phi ** 2)
    dirs = np.stack([r*np.cos(theta), r*np.sin(theta), z], axis=1)
    kmax = np.pi
    t = np.linspace(-1.0, 1.0, nread, endpoint=False, dtype=np.float64)
    ktraj = (kmax * t)[:, None, None] * dirs[None, :, :]
    return ktraj.reshape(-1, 3).astype(np.float32)

# ---- External 'traj' loader ----
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

def _scale_to_radians(k: np.ndarray, img_shape_xyz, fov_mm_xyz, vox_xyz):
    if k.size == 0: return k
    Nx, Ny, Nz = img_shape_xyz
    if fov_mm_xyz is not None:
        dxi = np.array([fov_mm_xyz[0]/Nx, fov_mm_xyz[1]/Ny, fov_mm_xyz[2]/Nz], float)
    else:
        dxi = np.array(vox_xyz if vox_xyz is not None else [1,1,1], float)
    k = np.asarray(k, float)
    kabs = np.percentile(np.linalg.norm(k, axis=1), 99.0)
    if kabs <= 1.2:      # cycles/FOV
        return k * (2*np.pi)
    elif kabs <= 4.2:    # already radians/pixel
        return k
    else:                # 1/mm or 1/m
        scale_unit = 1.0 if kabs < 1000 else (1/1000.0)
        return (k * scale_unit) * (2*np.pi) * dxi

def load_series_traj(series_dir: str, nread: int, nspokes: int,
                     img_shape_xyz, fov_mm, vox_xyz, traj_order: str):
    p = Path(series_dir) / "traj"
    if not p.exists():
        return None
    M = nread * nspokes
    arr = _read_txt_array(p)
    def maybe_reorder(A):
        if traj_order == "sample": return A
        try:
            return A.reshape(nspokes, nread, 3).transpose(1,0,2).reshape(M,3)
        except Exception:
            return A
    if arr.ndim == 2 and arr.size > 0:
        if arr.shape[0] == M and arr.shape[1] >= 3:
            k = maybe_reorder(arr[:, :3])
            return _scale_to_radians(k, img_shape_xyz, fov_mm, vox_xyz).astype(np.float32)
        if arr.shape[0] == nspokes and arr.shape[1] >= 3:
            dirs = arr[:, :3]
            kmax = np.pi; t = np.linspace(-1.0, 1.0, nread, endpoint=False)[:, None]
            k = (kmax * t)[:, None] * dirs[None, :, :]
            return k.reshape(-1, 3).astype(np.float32)
        if arr.shape[0] == nspokes and arr.shape[1] == 2:
            th, ph = arr[:,0], arr[:,1]
            dirs = np.stack([np.sin(ph)*np.cos(th), np.sin(ph)*np.sin(th), np.cos(ph)], axis=1)
            kmax = np.pi; t = np.linspace(-1.0, 1.0, nread, endpoint=False)[:, None]
            k = (kmax * t)[:, None] * dirs[None, :, :]
            return k.reshape(-1, 3).astype(np.float32)
    for dtype in (np.float32, np.float64):
        try:
            flat = np.fromfile(p, dtype=dtype)
        except Exception:
            continue
        if flat.size == M * 3:
            k = maybe_reorder(flat.reshape(M, 3).astype(np.float64))
            return _scale_to_radians(k, img_shape_xyz, fov_mm, vox_xyz).astype(np.float32)
        if flat.size == nspokes * 3:
            dirs = flat.reshape(nspokes, 3).astype(np.float64)
            kmax = np.pi; t = np.linspace(-1.0, 1.0, nread, endpoint=False)[:, None]
            k = (kmax * t)[:, None] * dirs[None, :, :]
            return k.reshape(-1, 3).astype(np.float32)
        if flat.size == nspokes * 2:
            ang = flat.reshape(nspokes, 2).astype(np.float64)
            th, ph = ang[:,0], ang[:,1]
            dirs = np.stack([np.sin(ph)*np.cos(th), np.sin(ph)*np.sin(th), np.cos(ph)], axis=1)
            kmax = np.pi; t = np.linspace(-1.0, 1.0, nread, endpoint=False)[:, None]
            k = (kmax * t)[:, None] * dirs[None, :, :]
            return k.reshape(-1, 3).astype(np.float32)
    print("[warn] Could not parse 'traj'; using golden-angle fallback.")
    return None

# ---------------- NUFFT helpers ----------------
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

def apply_read_shift(coords: np.ndarray, nread: int, nspokes: int, shift_samples: float):
    if abs(shift_samples) < 1e-12: return coords
    delta = float(shift_samples) * (2*np.pi / float(nread))
    uvw = coords.reshape(nread, nspokes, 3)
    dirs = uvw[-1,:,:] - uvw[0,:,:]
    norm = np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    dirs = dirs / norm
    uvw = uvw + delta * dirs[None, :, :]
    return uvw.reshape(-1, 3)

def psf_volume(coords: np.ndarray, img_shape_zyx: tuple[int,int,int], os_fac=1.75, width=4, gpu=-1):
    if gpu >= 0 and cp is not None:
        with sp.Device(gpu):
            ones = cp.ones(coords.shape[0], dtype=cp.complex64)
            coords_g = cp.asarray(coords, dtype=cp.float32)
            psf = sp.nufft_adjoint(ones, coords_g, oshape=img_shape_zyx, oversamp=os_fac, width=width)
            psf = cp.abs(psf); psf /= cp.max(psf) + 1e-12
            return cp.asnumpy(psf)
    ones = np.ones(coords.shape[0], np.complex64)
    psf = sp.nufft_adjoint(ones, coords, oshape=img_shape_zyx, oversamp=os_fac, width=width)
    psf = np.abs(psf); psf /= psf.max() + 1e-12
    return psf

def anisotropy_metrics(vol: np.ndarray):
    Z, Y, X = vol.shape
    zc, yc, xc = Z//2, Y//2, X//2
    r = max(4, min(Z, Y, X)//6)
    sub = vol[zc-r:zc+r+1, yc-r:yc+r+1, xc-r:xc+r+1]
    def second_moment(a, axis):
        idx = np.arange(a.shape[axis]) - a.shape[axis]/2.0
        shape = [1,1,1]; shape[axis] = -1
        w = idx.reshape(shape)
        num = float(np.sum((w**2) * a)); den = float(np.sum(a) + 1e-12)
        return num/den
    vz = second_moment(sub, 0); vy = second_moment(sub, 1); vx = second_moment(sub, 2)
    return vz, vy, vx

def psf_aniso_cost(coords: np.ndarray, img_shape_zyx_small, os_fac, width, gpu):
    psf = psf_volume(coords, img_shape_zyx_small, os_fac=os_fac, width=width, gpu=gpu)
    vz, vy, vx = anisotropy_metrics(psf)
    vmean = (vx+vy+vz)/3.0
    cost = abs(vz/vmean-1) + abs(vy/vmean-1) + abs(vx/vmean-1)
    return cost, (vz, vy, vx)

def radial_apod_window(nread: int, kind: str, tukey_alpha=0.30, kaiser_beta=6.0, gauss_sigma=0.18):
    r = np.linspace(-1, 1, nread, endpoint=False)
    if kind == "hann":
        w = 0.5 * (1 + np.cos(np.pi * r))
    elif kind == "tukey":
        a = float(tukey_alpha); at = np.abs(r); w = np.ones_like(r)
        if a <= 0: pass
        elif a >= 1: w = 0.5 * (1 + np.cos(np.pi * r))
        else:
            m = (at > (1-a)) & (a > 0)
            w[m] = 0.5 * (1 + np.cos(np.pi/a * (at[m] - (1-a))))
            w[at >= 1] = 0.0
    elif kind == "kaiser":
        from numpy import i0
        b = float(kaiser_beta); w = i0(b * np.sqrt(1 - r**2)) / i0(b)
    elif kind == "gauss":
        s = float(gauss_sigma); w = np.exp(-0.5 * (r/s)**2)
    else:
        w = np.ones_like(r)
    return (w / (w.mean() + 1e-12)).astype(np.float32)

def recon_adj_sos(kdata: np.ndarray, img_shape_zyx, coords, dcf_mode="pipe",
                  os_fac=1.75, width=4, gpu=-1, extra_w_flat=None):
    nread, nspokes, ncoils = kdata.shape
    if gpu < 0 or cp is None:
        dcf = compute_dcf(coords, mode=dcf_mode, os=os_fac, width=width)
        coils = []
        for c in range(ncoils):
            y = kdata[:, :, c].reshape(-1)
            if dcf is not None: y *= dcf
            if extra_w_flat is not None: y *= extra_w_flat
            x = sp.nufft_adjoint(y, coords, oshape=img_shape_zyx, oversamp=os_fac, width=width)
            coils.append(x)
        coils = np.stack(coils, axis=0)
        return np.sqrt((np.abs(coils) ** 2).sum(axis=0))
    with sp.Device(gpu):
        coords_g = cp.asarray(coords, dtype=cp.float32)
        dcf = compute_dcf(coords, mode=dcf_mode, os=os_fac, width=width)
        dcf_g = cp.asarray(dcf, dtype=cp.float32) if dcf is not None else None
        w_g = cp.asarray(extra_w_flat, dtype=cp.float32) if extra_w_flat is not None else None
        coils_g = []
        for c in range(ncoils):
            y_g = cp.asarray(kdata[:, :, c].reshape(-1))
            if dcf_g is not None: y_g *= dcf_g
            if w_g   is not None: y_g *= w_g
            x_g = sp.nufft_adjoint(y_g, coords_g, oshape=img_shape_zyx, oversamp=os_fac, width=width)
            coils_g.append(x_g)
        coils_g = cp.stack(coils_g, axis=0)
        sos_g = cp.sqrt((cp.abs(coils_g) ** 2).sum(axis=0))
        return cp.asnumpy(sos_g)

# ---------------- PNG helpers ----------------
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
        textw = lambda t: draw.textlength(t, font=font)
    except Exception:
        font = ImageFont.load_default()
        textw = lambda t: draw.textbbox((0,0), t, font=font)[2]
    x = margin; y_img = margin + label_h
    for im, lab in zip(rs, labels):
        w, h = im.size; tw = textw(lab); th = font.size
        tx = x + (w - int(tw))//2; ty = margin + (label_h - th)//2
        draw.text((tx+1, ty+1), lab, fill=0, font=font)
        draw.text((tx,   ty),   lab, fill=240, font=font)
        canvas.paste(im, (x, y_img))
        x += w + gap
    return canvas

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Bruker 3D radial recon (production)")
    ap.add_argument("--series", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--matrix", nargs=3, type=int, metavar=("NX","NY","NZ"), required=True)
    ap.add_argument("--spoke-step", type=int, default=1)
    ap.add_argument("--dcf", choices=["none","pipe"], default="pipe")
    ap.add_argument("--os", dest="os_fac", type=float, default=1.75)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--psf", action="store_true")
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--gpu-psf", action="store_true")
    ap.add_argument("--fov-scale", type=float, default=1.0)
    ap.add_argument("--traj-order", choices=["sample","spoke"], default="sample")
    ap.add_argument("--alt-rev", action="store_true")
    ap.add_argument("--rev-all", action="store_true")
    ap.add_argument("--auto-pi", action="store_true")
    ap.add_argument("--rd-shift", type=float, default=0.0)
    ap.add_argument("--auto-delay", action="store_true")
    ap.add_argument("--auto-delay-range", nargs=2, type=float, default=(-0.6,0.6))
    ap.add_argument("--auto-delay-steps", type=int, default=21)
    ap.add_argument("--scale-x", type=float, default=1.0)
    ap.add_argument("--scale-y", type=float, default=1.0)
    ap.add_argument("--scale-z", type=float, default=1.0)
    ap.add_argument("--auto-scale", action="store_true")
    ap.add_argument("--auto-scale-range", nargs=2, type=float, default=(0.7,1.3))
    ap.add_argument("--auto-scale-steps", type=int, default=5)
    ap.add_argument("--apod", choices=["none","hann","tukey","kaiser","gauss"], default="none")
    ap.add_argument("--tukey-alpha", type=float, default=0.30)
    ap.add_argument("--kaiser-beta", type=float, default=8.0)
    ap.add_argument("--gauss-sigma", type=float, default=0.22)
    ap.add_argument("--demean-spoke", action="store_true")
    ap.add_argument("--clip-pi", action="store_true")
    ap.add_argument("--prod", action="store_true", help="coarse 64³ auto-cal, then final recon")
    args = ap.parse_args()

    # Device banner
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

    # Build coords
    NX, NY, NZ = [int(x) for x in args.matrix]
    img_shape_xyz = (NX, NY, NZ); img_shape_zyx = (NZ, NY, NX)
    coords = load_series_traj(args.series, nread, nspokes, img_shape_xyz, fovmm, vox, args.traj_order)
    if coords is None:
        coords = make_ga_traj_3d(nspokes, nread)

    # Spoke reversal (on data)
    if args.rev_all:
        kdata = kdata[::-1, :, :]
        print("[readout] Reversed all spokes.")
    elif args.alt_rev:
        kdata[:, 1::2, :] = kdata[::-1, 1::2, :]
        print("[readout] Reversed every other (odd) spoke.")

    # Optional per-spoke DC remove
    if args.demean_spoke:
        kdata = kdata - kdata.mean(axis=0, keepdims=True)
        print("[pre] Subtracted per-spoke DC (demean).")

    # Decimate spokes if requested
    if args.spoke_step > 1:
        take = slice(0, nspokes, max(1, args.spoke_step))
        kdata   = kdata[:, take, :]
        coords  = coords.reshape(nread, nspokes, 3)[:, take, :].reshape(-1, 3)
        nspokes = kdata.shape[1]
        print(f"Decimated spokes by {args.spoke_step}: nspokes={nspokes}")

    # FOV scale
    if args.fov_scale != 1.0 and args.fov_scale > 0:
        coords = coords / float(args.fov_scale)
        print(f"[fov] Applying effective FOV scale = {args.fov_scale:.3f} (smaller = more crop)")

    # Auto-pi normalization
    p99 = np.percentile(np.linalg.norm(coords, axis=1), 99.0)
    if args.auto_pi and p99 > 0:
        s = float(np.pi / p99)
        coords = coords * s
        print(f"[auto-pi] Scaled coords by {s:.4f} so |k|_p99 ≈ π.")


    # Optional clip (broadcast-safe)
    if args.clip_pi:
        r = np.linalg.norm(coords, axis=1, keepdims=True)       # (M,1)
        s = np.minimum(1.0, np.pi / (r + 1e-12))                # (M,1) scale ≤ 1
        nclip = int((r > np.pi).sum())
        coords *= s                                             # (M,3) ← (M,1) broadcast
        print(f"[pre] Clipped |k| to ≤ π for {nclip}/{coords.shape[0]} samples.")


    # ----- PROD mode: fast tuner then final -----
    if args.prod:
        small = tuple(max(64, s//2) for s in img_shape_zyx)
        gpu_psf = args.gpu if args.gpu_psf else -1

        # Auto-delay coarse
        lo_d, hi_d = args.auto_delay_range; steps_d = max(5, args.auto_delay_steps)
        best_d, best_cost = 0.0, 1e9
        for sft in np.linspace(lo_d, hi_d, steps_d):
            c = apply_read_shift(coords, nread, nspokes, sft)
            cost,_ = psf_aniso_cost(c, small, os_fac=1.6, width=4, gpu=gpu_psf)
            if cost < best_cost:
                best_cost, best_d = cost, sft
        coords = apply_read_shift(coords, nread, nspokes, best_d)
        print(f"[prod] auto-delay ≈ {best_d:.3f} (cost={best_cost:.4f})")

        # Auto-scale coarse (coordinate-wise)
        lo_s, hi_s = args.auto_scale_range; steps_s = max(3, args.auto_scale_steps)
        scales = np.array([1.0,1.0,1.0], float)
        for _ in range(2):  # two passes
            for ax in (0,1,2):
                best_v, best_cost = scales[ax], 1e9
                for v in np.linspace(lo_s, hi_s, steps_s):
                    test = scales.copy(); test[ax] = v
                    c = coords * test[None,:]
                    cost,_ = psf_aniso_cost(c, small, os_fac=1.6, width=4, gpu=gpu_psf)
                    if cost < best_cost: best_cost, best_v = cost, v
                scales[ax] = best_v
        coords = coords * scales[None,:]
        print(f"[prod] auto-scale ≈ (x,y,z)=({scales[0]:.3f},{scales[1]:.3f},{scales[2]:.3f})")

        # Final recon uses full settings; also export a JSON sidecar
        args.rd_shift = float(best_d)
        args.scale_x, args.scale_y, args.scale_z = float(scales[0]), float(scales[1]), float(scales[2])

    # Manual delay/scale (if provided)
    if abs(args.rd_shift) > 1e-12:
        coords = apply_read_shift(coords, nread, nspokes, args.rd_shift)
        print(f"[delay] Applied read shift of {args.rd_shift:.3f} samples.")
    if any(abs(s-1.0)>1e-12 for s in (args.scale_x,args.scale_y,args.scale_z)):
        s = np.array([args.scale_x,args.scale_y,args.scale_z], float)
        coords = coords * s[None,:]
        print(f"[scale] Applied per-axis scales (x,y,z) = {tuple(s)}")

    # Diagnostics
    norm = np.linalg.norm(coords, axis=1)
    p = np.percentile(norm, [0,50,95,99,100])
    print(f"|k| percentiles (radians): {p}  (expect max ≈ π={np.pi:.3f})")

    # Apod weights
    w_flat = None
    if args.apod != "none":
        w_read = radial_apod_window(nread, args.apod, args.tukey_alpha, args.kaiser_beta, args.gauss_sigma)
        w_flat = np.repeat(w_read[:, None], nspokes, axis=1).reshape(-1)
        print(f"[apod] {args.apod} (mean={float(w_flat.mean()):.3f})")

    # Recon
    img = recon_adj_sos(kdata, img_shape_zyx, coords,
                        dcf_mode=args.dcf, os_fac=args.os_fac, width=args.width,
                        gpu=args.gpu, extra_w_flat=w_flat)

    affine = np.diag([vox[0], vox[1], vox[2], 1.0])
    nib.save(nib.Nifti1Image(np.asarray(img, np.float32), affine), args.out)
    print(f"Wrote {args.out}")

    # PNGs
    if args.png:
        trip = _tripanel_from_volume(img, labels=("Axial","Coronal","Sagittal"))
        trip_path = str(Path(args.out).with_suffix("")) + "_tripanel.png"
        trip.save(trip_path); print(f"Saved tripanel PNG → {trip_path}")
    if args.psf:
        psf_mag = psf_volume(coords, img_shape_zyx, os_fac=args.os_fac, width=args.width,
                             gpu=(args.gpu if args.gpu_psf else -1))
        vz, vy, vx = anisotropy_metrics(psf_mag)
        vmean = (vx+vy+vz)/3.0
        trip_psf = _tripanel_from_volume(psf_mag, labels=("PSF Axial","PSF Coronal","PSF Sagittal"))
        psf_path = str(Path(args.out).with_suffix("")) + "_psf.png"
        trip_psf.save(psf_path); print(f"Saved PSF PNG → {psf_path}")
        print(f"[psf] second-moment variances (Z,Y,X) = ({vz:.4f}, {vy:.4f}, {vx:.4f}); "
              f"anisotropy ratios vs mean = ({vz/vmean:.3f}, {vy/vmean:.3f}, {vx/vmean:.3f})")

    # Sidecar for reproducibility
    sidecar = {
        "series": str(args.series), "out": str(args.out),
        "matrix": [int(NX),int(NY),int(NZ)],
        "traj_order": args.traj_order, "alt_rev": args.alt_rev, "rev_all": args.rev_all,
        "auto_pi": args.auto_pi, "rd_shift": args.rd_shift,
        "scale": {"x": args.scale_x, "y": args.scale_y, "z": args.scale_z},
        "apod": {"kind": args.apod, "tukey_alpha": args.tukey_alpha,
                 "kaiser_beta": args.kaiser_beta, "gauss_sigma": args.gauss_sigma},
        "os": args.os_fac, "width": args.width, "fov_scale": args.fov_scale,
        "demean_spoke": args.demean_spoke, "clip_pi": args.clip_pi
    }
    with open(str(Path(args.out).with_suffix("")) + "_recon.json", "w") as f:
        json.dump(sidecar, f, indent=2)
        print("Saved params →", f.name)

if __name__ == "__main__":
    main()
