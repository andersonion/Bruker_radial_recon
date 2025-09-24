#!/usr/bin/env python3
# Bruker 3D radial recon (auto-uses "traj" file if present)
# Deps: pip install sigpy nibabel numpy pillow

from __future__ import annotations
import argparse, os, re
from pathlib import Path
import numpy as np
import nibabel as nib
import sigpy as sp
import sigpy.mri as mr  # only for DCF if available

try:
    import cupy as cp
except Exception:
    cp = None  # CPU fallback


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

# ---------------- Bruker loader ----------------
def load_bruker_series(series_dir: str):
    series = Path(series_dir)
    acqp   = _parse_jcamp(series / "acqp")
    method = _parse_jcamp(series / "method")
    visu   = _parse_jcamp(series / "visu_pars")

    dim = int((_nums(acqp.get("ACQ_dim", "")) or [3])[0])
    if dim != 3:
        raise RuntimeError(f"ACQ_dim={dim} (expected 3 for 3D)")

    acq_size = _nums(acqp.get("ACQ_size", ""))  # often [3, NPROJ, 1]
    nspokes_hint = int(acq_size[1]) if len(acq_size) > 1 else None

    rec_sel = (acqp.get("ACQ_ReceiverSelect", "") or "").lower().split()
    ncoils = rec_sel.count("yes") or _get_int(method, "PVM_EncNReceivers", 1) or 1

    # FID (interleaved real/imag)
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

    # Infer nread robustly
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

    # Optional readout flip (env knob)
    if os.environ.get("RADIAL_FLIP_READ","0") == "1":
        data = data.reshape(-1, nread)[:, ::-1].reshape(-1)

    # Shape
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

    meta = dict(nread=nread, nspokes=nspokes, ncoils=ncoils, SW_h=SW_h, dwell=dwell,
                TE_s=TE_s, TR_s=TR_s, vox=vox)
    return kdata, meta, dict(acqp=acqp, method=method, visu=visu)

# -------------- Trajectory (GA fallback) --------------
def make_ga_traj_3d(nspokes: int, nread: int) -> np.ndarray:
    i = np.arange(nspokes, dtype=np.float64) + 0.5
    phi = (1 + np.sqrt(5)) / 2
    z = 1.0 - 2.0 * i / (nspokes + 1.0)
    r = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    theta = 2.0 * np.pi * i / (phi ** 2)
    dirs = np.stack([r*np.cos(theta), r*np.sin(theta), z], axis=1)  # (nspokes,3)
    kmax = np.pi
    t = np.linspace(-1.0, 1.0, nread, endpoint=False, dtype=np.float64)
    radii = (kmax * t)[:, None, None]  # (nread,1,1)  <-- broadcasting-safe
    ktraj = (radii * dirs[None, :, :]).reshape(-1, 3)  # (M,3)
    return ktraj.astype(np.float32)

# ---- External 'traj' loader (auto text/binary; auto units) ----
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

def _scale_to_radians(k: np.ndarray, img_shape: tuple[int,int,int], vox: np.ndarray) -> np.ndarray:
    """Auto-infer units: cycles/FOV (~<=1), radians (~<=π), or 1/mm (larger)."""
    if k.size == 0:
        return k
    Nx, Ny, Nz = img_shape
    dxi = np.array(vox if vox is not None and len(vox) >= 3 else [1,1,1], float)
    k = np.asarray(k, float)
    kabs = np.percentile(np.linalg.norm(k, axis=1), 99.0)
    if kabs <= 1.2:  # cycles/FOV
        scale = np.array([2*np.pi/Nx, 2*np.pi/Ny, 2*np.pi/Nz], float)
        return (k * scale[None, :])
    elif kabs <= 4.2:  # radians
        return k
    else:  # 1/mm
        scale = 2*np.pi * dxi  # per-dim Δx
        return (k * scale[None, :])

def load_series_traj(series_dir: str, nread: int, nspokes: int,
                     img_shape: tuple[int,int,int], vox: np.ndarray) -> np.ndarray | None:
    p = Path(series_dir) / "traj"
    if not p.exists():
        return None
    M = nread * nspokes

    # Try text first
    arr = _read_txt_array(p)
    if arr.ndim == 2 and arr.size > 0:
        if arr.shape[0] == M and arr.shape[1] >= 3:      # per-sample (M,3)
            k = arr[:, :3]
            return _scale_to_radians(k, img_shape, vox).astype(np.float32)
        if arr.shape[0] == nspokes and arr.shape[1] >= 3:  # directions (nspokes,3)
            dirs = arr[:, :3]
            kmax = np.pi
            t = np.linspace(-1.0, 1.0, nread, endpoint=False)[:, None]
            k = (kmax * t)[:, None] * dirs[None, :, :]
            return k.reshape(-1, 3).astype(np.float32)
        if arr.shape[0] == nspokes and arr.shape[1] == 2:  # angles (theta,phi)
            th, ph = arr[:,0], arr[:,1]
            dirs = np.stack([np.sin(ph)*np.cos(th), np.sin(ph)*np.sin(th), np.cos(ph)], axis=1)
            kmax = np.pi
            t = np.linspace(-1.0, 1.0, nread, endpoint=False)[:, None]
            k = (kmax * t)[:, None] * dirs[None, :, :]
            return k.reshape(-1, 3).astype(np.float32)

    # Try binary float32 / float64
    for dtype in (np.float32, np.float64):
        try:
            flat = np.fromfile(p, dtype=dtype)
        except Exception:
            continue
        if flat.size == M * 3:
            k = flat.reshape(M, 3).astype(np.float64)
            return _scale_to_radians(k, img_shape, vox).astype(np.float32)
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

# ---------------- DCF (version-safe) ----------------
def compute_dcf(coords: np.ndarray, mode: str = "pipe") -> np.ndarray | None:
    if mode == "none":
        return None
    dcf = None
    try:
        if hasattr(mr.dcf, "pipe_menon"):
            dcf = mr.dcf.pipe_menon(coords, niter=30)
        elif hasattr(mr.dcf, "pipe_menon_dcf"):
            dcf = mr.dcf.pipe_menon_dcf(coords, max_iter=30)
    except Exception:
        dcf = None
    if dcf is None:
        r = np.linalg.norm(coords, axis=1)  # fallback ~ r^2 for 3D
        dcf = (r**2 + 1e-6)
    dcf = dcf.astype(np.float32, copy=False)
    dcf *= (dcf.size / (dcf.sum() + 1e-8))
    return dcf

# ---------------- Quicklooks ----------------
def save_quicklook_pngs(img: np.ndarray, out_path: str, pct: float = 99.5) -> None:
    from PIL import Image
    outdir = Path(out_path).with_suffix("").with_suffix("")
    outdir = Path(str(outdir) + "_png")
    outdir.mkdir(parents=True, exist_ok=True)
    vmax = float(np.percentile(img, pct)); vmin = 0.0; rng = max(vmax - vmin, 1e-8)
    zc, yc, xc = [s // 2 for s in img.shape]
    planes = {"ax": img[zc,:,:], "cor": img[:,yc,:], "sag": img[:,:,xc]}
    for name, sl in planes.items():
        sln = np.clip((sl - vmin) / rng, 0, 1)
        Image.fromarray((sln * 255.0).astype(np.uint8)).save(outdir / f"{name}.png")

# ---------------- Recon (adjoint + SoS) ----------------
def recon_adj_sos(kdata: np.ndarray, img_shape: tuple[int,int,int],
                  coords: np.ndarray, dcf_mode: str = "pipe",
                  gpu: int = -1) -> np.ndarray:
    """Adjoint NUFFT per coil + SoS. Returns numpy array (nz, ny, nx)."""
    nread, nspokes, ncoils = kdata.shape
    assert coords.shape == (nread * nspokes, 3)

    # CPU path
    if gpu < 0 or cp is None:
        dcf = compute_dcf(coords, mode=dcf_mode)  # numpy
        coils = []
        for c in range(ncoils):
            y = kdata[:, :, c].reshape(-1)
            if dcf is not None:
                y = y * dcf
            x = sp.nufft_adjoint(y, coords, oshape=img_shape)
            coils.append(x)
        coils = np.stack(coils, axis=0)
        return np.sqrt((np.abs(coils) ** 2).sum(axis=0))

    # GPU path
    with sp.Device(gpu):
        coords_g = cp.asarray(coords, dtype=cp.float32)
        dcf = compute_dcf(coords, mode=dcf_mode)            # compute on CPU
        dcf_g = cp.asarray(dcf, dtype=cp.float32) if dcf is not None else None

        coils_g = []
        for c in range(ncoils):
            y_g = cp.asarray(kdata[:, :, c].reshape(-1))    # complex64
            if dcf_g is not None:
                y_g = y_g * dcf_g
            x_g = sp.nufft_adjoint(y_g, coords_g, oshape=img_shape)
            coils_g.append(x_g)
        coils_g = cp.stack(coils_g, axis=0)
        sos_g = cp.sqrt((cp.abs(coils_g) ** 2).sum(axis=0))
        return cp.asnumpy(sos_g)

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Bruker 3D radial recon (auto 'traj' or GA)")
    ap.add_argument("--series", required=True, help="Path with acqp/method/visu_pars/fid[/traj]")
    ap.add_argument("--out", required=True, help="Output NIfTI filename")
    ap.add_argument("--matrix", nargs=3, type=int, metavar=("NX","NY","NZ"), required=True)
    ap.add_argument("--spoke-step", dest="spoke_step", type=int, default=1,
                    help="Take every Nth spoke for speed (e.g., 4)")
    ap.add_argument("--dcf", choices=["none","pipe"], default="pipe",
                    help="Density compensation (pipe=Pipe–Menon, default)")
    ap.add_argument("--png", action="store_true", help="Write axial/cor/sag PNGs")
    ap.add_argument("--gpu", type=int, default=-1,
                help="GPU id to use (e.g., 0). -1 = CPU (default)")
    args = ap.parse_args()

    kdata, meta, hdr = load_bruker_series(args.series)
    nread, nspokes = meta["nread"], meta["nspokes"]
    print(f"Loaded: nread={nread}, nspokes={nspokes}, ncoils={meta['ncoils']}, SW_h={meta['SW_h']} Hz")

    # Build/load coords, then apply SAME decimation to kdata & coords
    vox = meta.get("vox", np.array([1,1,1], float))
    full_coords = load_series_traj(args.series, nread, nspokes, tuple(args.matrix), vox)
    if full_coords is None:
        full_coords = make_ga_traj_3d(nspokes, nread)

    if args.spoke_step > 1:
        take = slice(0, nspokes, max(1, args.spoke_step))
        kdata   = kdata[:, take, :]
        coords  = full_coords.reshape(nread, nspokes, 3)[:, take, :].reshape(-1, 3)
        nspokes = kdata.shape[1]
        print(f"Decimated spokes by {args.spoke_step}: nspokes={nspokes}")
    else:
        coords = full_coords

    img = recon_adj_sos(kdata, tuple(args.matrix), coords,
                    dcf_mode=args.dcf, gpu=args.gpu)


    affine = np.diag([vox[0], vox[1], vox[2], 1.0])
    nib.save(nib.Nifti1Image(np.asarray(img, np.float32), affine), args.out)
    print(f"Wrote {args.out}")

    if args.png:
        save_quicklook_pngs(img, args.out)
        print(f"Saved PNGs → {Path(args.out).with_suffix('').with_suffix('')}_png")

if __name__ == "__main__":
    main()
