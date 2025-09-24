#!/usr/bin/env python3
"""
Bruker 3D Radial Recon (CPU-first, no brkraw)

- Parses JCAMP headers (acqp/method/visu_pars) directly
- Reads interleaved complex 'fid' with correct endianness/format
- Robustly infers nread (uses TD, TD//2, and divisibility checks)
- Shapes to (nread, nspokes, ncoils)
- Builds 3D golden-angle-ish trajectory
- NUFFT adjoint per coil (sigpy.nufft_adjoint) + Sum-of-Squares
- Saves NIfTI; optional PNGs

Deps:  pip install sigpy nibabel numpy pillow
Example:
  python bruker_radial_recon_3d.py \
    --series /path/to/Bruker/series \
    --out recon3d_radial.nii.gz \
    --matrix 192 192 192 \
    --traj ga \
    --spoke-step 4 --png
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import nibabel as nib
import sigpy as sp  # <-- use sp.nufft_adjoint

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

def _get_int(d: dict, key: str, default=1) -> int:
    try:
        v = _nums(d.get(key, ""))
        return int(v[0]) if v else default
    except Exception:
        return default

# ---------- Data loading ----------
def load_bruker_series(series_dir: str):
    series = Path(series_dir)
    acqp   = _parse_jcamp(series / "acqp")
    method = _parse_jcamp(series / "method")
    visu   = _parse_jcamp(series / "visu_pars")

    dim = int((_nums(acqp.get("ACQ_dim", "")) or [3])[0])
    if dim != 3:
        raise RuntimeError(f"ACQ_dim={dim} (expected 3 for 3D)")

    acq_size = _nums(acqp.get("ACQ_size", ""))  # often [3, NPROJ, 1] for 3D radial
    nspokes_hint = int(acq_size[1]) if len(acq_size) > 1 else None

    # Coils
    rec_sel = (acqp.get("ACQ_ReceiverSelect", "") or "").lower().split()
    ncoils = rec_sel.count("yes")
    if ncoils == 0:
        ncoils = _get_int(method, "PVM_EncNReceivers", 1)

    # Read FID (interleaved real/imag)
    fid_path = series / "fid"
    if not fid_path.exists():
        raise RuntimeError(f"Missing FID: {fid_path}")

    fmt  = (acqp.get("GO_raw_data_format") or method.get("GO_raw_data_format") or "GO_32BIT_SGN_INT").strip()
    byto = (acqp.get("BYTORDA") or method.get("BYTORDA") or "little").strip().lower()
    endian = "<" if ("little" in byto or "lsb" in byto) else ">"

    dt_map = {"GO_16BIT_SGN_INT": np.int16, "GO_32BIT_SGN_INT": np.int32, "GO_32BIT_FLOAT": np.float32}
    base = dt_map.get(fmt, np.int32)
    basecode = np.dtype(base).str[1:]  # 'i2', 'i4', 'f4'
    raw = np.fromfile(fid_path, dtype=endian + basecode)
    if raw.size % 2 != 0:
        raise RuntimeError("Raw FID length is odd; expected interleaved real/imag pairs.")
    complex_samples = raw.size // 2
    ri = raw.reshape(-1, 2).astype(np.float32)
    data = ri[:, 0] + 1j * ri[:, 1]  # (complex_samples,)

    # ---- Infer nread robustly ----
    nread = None
    td = _get_int(acqp, "TD", 0)  # Bruker 'TD' can be real-sample count or complex count depending on sequence
    for cand in [td, td // 2]:
        if cand and cand > 8 and (complex_samples % cand == 0):
            nread = cand
            break
    # If TD didn't help, try to use hints (nspokes_hint, ncoils, reps)
    if nread is None:
        NR       = _get_int(acqp, "NR", 1)
        NECHOES  = _get_int(acqp, "NECHOES", 1)
        NA       = _get_int(acqp, "NA", 1) or _get_int(acqp, "ACQ_averages", 1)
        NI       = _get_int(acqp, "NI", 1)
        mult = max(1, NR) * max(1, NECHOES) * max(1, NA) * max(1, NI)
        if nspokes_hint and ncoils and mult:
            denom = nspokes_hint * ncoils * mult
            if denom > 0 and complex_samples % denom == 0:
                cand = complex_samples // denom
                if 8 < cand < 8192:
                    nread = int(cand)
    # Last resort: sweep divisors in a sane range
    if nread is None:
        for cand in (128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024):
            if complex_samples % cand == 0:
                nread = cand
                break
    if nread is None:
        raise RuntimeError("Failed to infer nread (readout points).")

    # Now shape to readout vectors
    if data.size % nread != 0:
        raise RuntimeError(f"Readout length mismatch: nread={nread}, total complex samples={data.size}")
    vecs = data.reshape(-1, nread)  # (nvecs, nread)
    total_vecs = vecs.shape[0]
    if total_vecs % ncoils != 0:
        raise RuntimeError(f"Vector count {total_vecs} not divisible by ncoils={ncoils}")
    per_coil = total_vecs // ncoils
    nspokes = per_coil  # assume spoke-major ordering per coil
    if nspokes_hint and nspokes_hint != nspokes:
        print(f"[warn] ACQ_size hinted nspokes={nspokes_hint}, inferred from FID={nspokes}.")

    arr   = vecs.reshape(ncoils, nspokes, nread)
    kdata = np.transpose(arr, (2, 1, 0)).astype(np.complex64)  # (nread, nspokes, ncoils)

    # Meta
    SW_h  = float((_nums(acqp.get("SW_h", "0")) or [0])[0])
    dwell = (1.0 / SW_h) if SW_h > 0 else None
    TE_s  = float((_nums(acqp.get("ACQ_echo_time", "0")) or [0])[0])
    TR_s  = float((_nums(acqp.get("ACQ_repetition_time", "0")) or [0])[0])
    vox   = np.array(_nums(visu.get("VisuCoreVoxelSize", "")) or [1, 1, 1], dtype=float)

    meta = dict(nread=nread, nspokes=nspokes, ncoils=ncoils, SW_h=SW_h, dwell=dwell,
                TE_s=TE_s, TR_s=TR_s, vox=vox)
    return kdata, meta, dict(acqp=acqp, method=method, visu=visu)

# ---------- Trajectory ----------
def make_ga_traj_3d(nspokes: int, nread: int) -> np.ndarray:
    """Golden-angle-ish 3D radial coords (M, 3) in ~[-0.5, 0.5)."""
    i = np.arange(nspokes, dtype=np.float64) + 0.5
    phi = (1 + np.sqrt(5)) / 2
    z = 1.0 - 2.0 * i / (nspokes + 1.0)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = 2.0 * np.pi * i / (phi ** 2)
    dirs = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)  # (nspokes, 3)
    kmax = 0.5
    t = np.linspace(-1.0, 1.0, nread, endpoint=False, dtype=np.float64)
    radii = (kmax * t).astype(np.float64)[:, None]                         # (nread, 1)
    ktraj = (radii * dirs[None, :, :]).reshape(-1, 3)                      # (M, 3)
    return ktraj.astype(np.float32)

# ---------- Recon (adjoint + SoS) ----------
def recon_radial_3d_adjoint(kdata: np.ndarray, matrix: tuple[int,int,int],
                            spoke_step: int = 1, traj_kind: str = "ga") -> np.ndarray:
    """kdata: (nread, nspokes, ncoils) -> |img| as (nz, ny, nx)"""
    nread, nspokes, ncoils = kdata.shape
    if spoke_step > 1:
        kdata = kdata[:, ::spoke_step, :]
        nspokes = kdata.shape[1]
    if traj_kind != "ga":
        raise NotImplementedError("Only 'ga' trajectory is implemented.")
    coords = make_ga_traj_3d(nspokes, nread)  # (M, 3)
    img_shape = tuple(int(x) for x in matrix)

    coils_img = []
    for c in range(ncoils):
        y = kdata[:, :, c].reshape(-1)       # (M,)
        x = sp.nufft_adjoint(y, coords, oshape=img_shape)  # complex (nz, ny, nx)
        coils_img.append(x)
    coils_img = np.stack(coils_img, axis=0)  # (ncoils, nz, ny, nx)
    sos = np.sqrt((np.abs(coils_img) ** 2).sum(axis=0))
    return sos

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Bruker 3D radial reconstruction (NUFFT adjoint)")
    ap.add_argument("--series", required=True, help="Path to Bruker scan folder (acqp, method, fid)")
    ap.add_argument("--out", required=True, help="Output NIfTI filename")
    ap.add_argument("--matrix", nargs=3, type=int, metavar=("NX","NY","NZ"), required=True)
    ap.add_argument("--traj", default="ga", choices=["ga"])
    ap.add_argument("--spoke-step", dest="spoke_step", type=int, default=1,
                    help="Take every Nth spoke (e.g., 4) for quick tests")
    ap.add_argument("--png", action="store_true", help="Write axial/coronal/sagittal PNGs")
    args = ap.parse_args()

    kdata, meta, hdr = load_bruker_series(args.series)
    print(f"Loaded: nread={meta['nread']}, nspokes={meta['nspokes']}, ncoils={meta['ncoils']}, SW_h={meta['SW_h']} Hz")
    if args.spoke_step > 1:
        print(f"Decimating spokes by {args.spoke_step}...")

    img = recon_radial_3d_adjoint(kdata, tuple(args.matrix),
                                  spoke_step=args.spoke_step, traj_kind=args.traj)

    vox = meta.get("vox", np.array([1,1,1], float))
    affine = np.diag([vox[0], vox[1], vox[2], 1.0])
    nib.save(nib.Nifti1Image(np.asarray(img, np.float32), affine), args.out)
    print(f"Wrote {args.out}")

    if args.png:
        try:
            from PIL import Image
            outdir = Path(args.out).with_suffix("").with_suffix("")
            outdir = Path(str(outdir) + "_png")
            outdir.mkdir(parents=True, exist_ok=True)
            zc, yc, xc = [s // 2 for s in img.shape]
            planes = {"ax": img[zc,:,:], "cor": img[:,yc,:], "sag": img[:,:,xc]}
            for name, sl in planes.items():
                im = (sl / (sl.max() + 1e-8) * 255.0).astype(np.uint8)
                Image.fromarray(im).save(outdir / f"{name}.png")
            print(f"Saved PNGs → {outdir}")
        except Exception as e:
            print(f"[warn] PNG export failed: {e}")

if __name__ == "__main__":
    main()
