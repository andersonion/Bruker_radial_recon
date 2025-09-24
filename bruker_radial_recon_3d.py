#!/usr/bin/env python3
"""
Bruker 3D Radial Recon (CPU-first, no brkraw)

- Parses Bruker JCAMP headers (acqp/method/visu_pars) directly
- Reads interleaved complex 'fid' with correct endianness/format
- Shapes to (nread, nspokes, ncoils)
- Builds 3D golden-angle-ish trajectory (uniform on the sphere)
- NUFFT adjoint per coil + sum-of-squares combine
- Saves NIfTI; optional PNGs of axial/coronal/sagittal quicklooks

Dependencies:
    pip install sigpy nibabel numpy pillow

Example:
    python bruker_radial_recon_3d.py \
        --series /path/to/Bruker/series_dir \
        --out recon3d_radial.nii.gz \
        --matrix 192 192 192 \
        --traj ga \
        --spoke-step 4
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
import nibabel as nib
import sigpy.mri as mr


# ----------------------------- JCAMP parsing ----------------------------- #

def _parse_jcamp(path: Path) -> dict:
    """Parse Bruker JCAMP-DX text file into a dict."""
    d = {}
    if not path.exists():
        return d
    key = None
    with path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("$$"):
                continue
            if line.startswith("##$") and "=" in line:
                key, val = line[3:].split("=", 1)
                key = key.strip()
                d[key] = val.strip()
            elif key:
                d[key] += " " + line
    return d


_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
def _nums(s: str) -> list[float]:
    if not s:
        return []
    vals = _num_re.findall(s)
    out = []
    for v in vals:
        if "." in v or "e" in v.lower():
            out.append(float(v))
        else:
            try:
                out.append(int(v))
            except ValueError:
                out.append(float(v))
    return out


# ----------------------------- Data loading ------------------------------ #

def load_bruker_series(series_dir: str):
    """
    Load a Bruker 3D radial series.

    Returns:
        kdata: complex64 array of shape (nread, nspokes, ncoils)
        meta:  dict with nread, nspokes, ncoils, SW_h, dwell, TE_s, TR_s, vox
        hdr:   dict(acqp=..., method=..., visu=...)
    """
    series = Path(series_dir)
    acqp   = _parse_jcamp(series / "acqp")
    method = _parse_jcamp(series / "method")
    visu   = _parse_jcamp(series / "visu_pars")

    # Expect 3D dataset
    dim = int((_nums(acqp.get("ACQ_dim", "")) or [3])[0])
    if dim != 3:
        raise RuntimeError(f"ACQ_dim={dim} (expected 3 for 3D)")

    # Read dimensions
    size = _nums(acqp.get("ACQ_size", ""))
    if len(size) < 2:
        raise RuntimeError("Could not parse ACQ_size from acqp.")
    nread, nspokes_hint = int(size[0]), int(size[1])

    # Coil count
    rec_sel_tokens = (acqp.get("ACQ_ReceiverSelect", "") or "").lower().split()
    ncoils = rec_sel_tokens.count("yes")
    if ncoils == 0:
        ncoils = int((_nums(method.get("PVM_EncNReceivers", "1")) or [1])[0])

    # Raw FID path
    fid_path = series / "fid"
    if not fid_path.exists():
        raise RuntimeError(f"Missing FID: {fid_path}")

    # Endianness and base data type
    fmt  = (acqp.get("GO_raw_data_format") or method.get("GO_raw_data_format") or "GO_32BIT_SGN_INT").strip()
    byto = (acqp.get("BYTORDA") or method.get("BYTORDA") or "little").strip().lower()
    endian = "<" if ("little" in byto or "lsb" in byto) else ">"

    dt_map = {
        "GO_16BIT_SGN_INT": np.int16,
        "GO_32BIT_SGN_INT": np.int32,
        "GO_32BIT_FLOAT":   np.float32,
    }
    base = dt_map.get(fmt, np.int32)
    basecode = np.dtype(base).str[1:]  # 'i2', 'i4', 'f4', etc.

    raw = np.fromfile(fid_path, dtype=endian + basecode)
    if raw.size % 2 != 0:
        raise RuntimeError("Raw FID length is odd; expected interleaved real/imag pairs.")
    ri = raw.reshape(-1, 2).astype(np.float32)
    data = ri[:, 0] + 1j * ri[:, 1]

    if data.size % nread != 0:
        raise RuntimeError(f"Readout length mismatch: nread={nread}, total complex samples={data.size}")
    vecs = data.reshape(-1, nread)  # each row is a readout vector

    total_vecs = vecs.shape[0]
    if total_vecs % ncoils != 0:
        raise RuntimeError(f"Vector count {total_vecs} not divisible by ncoils={ncoils}")
    per_coil = total_vecs // ncoils
    nspokes  = per_coil  # assume spoke-major ordering per coil
    if nspokes_hint and nspokes_hint != nspokes:
        print(f"[warn] ACQ_size hinted nspokes={nspokes_hint}, inferred from FID={nspokes}.")

    arr   = vecs.reshape(ncoils, nspokes, nread)
    kdata = np.transpose(arr, (2, 1, 0)).astype(np.complex64)  # (nread, nspokes, ncoils)

    # Timing & voxel info
    SW_h  = float((_nums(acqp.get("SW_h", "0")) or [0])[0])
    dwell = (1.0 / SW_h) if SW_h > 0 else None
    TE_s  = float((_nums(acqp.get("ACQ_echo_time", "0")) or [0])[0])
    TR_s  = float((_nums(acqp.get("ACQ_repetition_time", "0")) or [0])[0])
    vox   = np.array(_nums(visu.get("VisuCoreVoxelSize", "")) or [1, 1, 1], dtype=float)

    meta = dict(nread=nread, nspokes=nspokes, ncoils=ncoils, SW_h=SW_h, dwell=dwell,
                TE_s=TE_s, TR_s=TR_s, vox=vox)
    return kdata, meta, dict(acqp=acqp, method=method, visu=visu)


# ------------------------------ Trajectory ------------------------------- #

def make_ga_traj_3d(nspokes: int, nread: int) -> np.ndarray:
    """
    Golden-angle-ish 3D radial trajectory.
    Returns k-space coords with shape (Nd=3, M=nread*nspokes) in normalized units ~[-0.5, 0.5).
    """
    # Directions on the sphere via Fibonacci / phyllotaxis
    i = np.arange(nspokes, dtype=np.float64) + 0.5
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    z = 1.0 - 2.0 * i / (nspokes + 1.0)
    r = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    theta = 2.0 * np.pi * i / (phi ** 2)
    dirs = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=0)  # (3, nspokes)

    # Readout radii from -kmax..kmax (SigPy normalized k-space uses approx [-0.5, 0.5))
    kmax = 0.5
    t = np.linspace(-1.0, 1.0, nread, endpoint=False, dtype=np.float64)
    radii = (kmax * t).astype(np.float64)  # (nread,)

    ktraj = (dirs[None, :, :] * radii[:, None, None]).reshape(3, -1)  # (3, nread*nspokes)
    return ktraj.astype(np.float32)


# ----------------------------- Reconstruction ---------------------------- #

def recon_radial_3d_adjoint(kdata: np.ndarray,
                            matrix: tuple[int, int, int],
                            traj_kind: str = "ga",
                            spoke_step: int = 1) -> np.ndarray:
    """
    NUFFT adjoint per coil + sum-of-squares combine.

    Args:
        kdata: (nread, nspokes, ncoils)
        matrix: (nx, ny, nz)
        traj_kind: currently only 'ga'
        spoke_step: take every Nth spoke to decimate

    Returns:
        img: complex image volume of shape (nz, ny, nx) (we'll save magnitude)
    """
    nread, nspokes, ncoils = kdata.shape
    if spoke_step > 1:
        kdata = kdata[:, ::spoke_step, :]
        nspokes = kdata.shape[1]

    if traj_kind != "ga":
        raise NotImplementedError("Only 'ga' (golden-angle-ish) trajectory is implemented.")

    # Build coords once (Nd, M)
    ktraj = make_ga_traj_3d(nspokes, nread)  # shape (3, M)
    img_shape = tuple(int(x) for x in matrix)

    # NUFFT adjoint per coil
    coils_img = []
    for c in range(ncoils):
        y = kdata[:, :, c].reshape(-1)  # (M,)
        x = mr.nufft_adjoint(y, ktraj, img_shape)
        coils_img.append(x)
    coils_img = np.stack(coils_img, axis=0)  # (ncoils, nz, ny, nx)

    # Sum-of-squares combine
    sos = np.sqrt((np.abs(coils_img) ** 2).sum(axis=0))
    return sos


# --------------------------------- CLI ---------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Bruker 3D radial reconstruction (NUFFT adjoint)")
    ap.add_argument("--series", required=True, help="Path to Bruker scan folder (contains acqp, method, fid)")
    ap.add_argument("--out", required=True, help="Output NIfTI filename")
    ap.add_argument("--matrix", nargs=3, type=int, metavar=("NX", "NY", "NZ"), required=True,
                    help="Reconstruction matrix, e.g. 192 192 192")
    ap.add_argument("--traj", default="ga", choices=["ga"], help="Trajectory type")
    ap.add_argument("--spoke-step", dest="spoke_step", type=int, default=1,
                    help="Subsample spokes: take every Nth spoke (e.g., 4 for quick test)")
    ap.add_argument("--png", action="store_true", help="Write axial/coronal/sagittal PNG quicklooks")
    args = ap.parse_args()

    # Load data
    kdata, meta, hdr = load_bruker_series(args.series)
    print(f"Loaded: nread={meta['nread']}, nspokes={meta['nspokes']}, ncoils={meta['ncoils']}, SW_h={meta['SW_h']} Hz")
    if args.spoke_step > 1:
        print(f"Decimating spokes by {args.spoke_step}...")

    # Recon
    img = recon_radial_3d_adjoint(kdata, tuple(args.matrix), traj_kind=args.traj, spoke_step=args.spoke_step)

    # Save NIfTI
    vox = meta.get("vox", np.array([1, 1, 1], float))
    affine = np.diag([vox[0], vox[1], vox[2], 1.0])
    nii = nib.Nifti1Image(np.asarray(img, np.float32), affine)
    nib.save(nii, args.out)
    print(f"Wrote {args.out}")

    # Optional PNGs
    if args.png:
        try:
            from PIL import Image
            outdir = Path(args.out).with_suffix("").with_suffix("")
            outdir = Path(str(outdir) + "_png")
            outdir.mkdir(parents=True, exist_ok=True)
            zc, yc, xc = [s // 2 for s in img.shape]
            quicks = {
                "ax":  img[zc, :, :],
                "cor": img[:, yc, :],
                "sag": img[:, :, xc],
            }
            for name, sl in quicks.items():
                sln = sl / (sl.max() + 1e-8)
                im = (sln * 255.0).astype(np.uint8)
                Image.fromarray(im).save(outdir / f"{name}.png")
            print(f"Saved PNGs → {outdir}")
        except Exception as e:
            print(f"[warn] PNG export failed: {e}")


if __name__ == "__main__":
    main()
