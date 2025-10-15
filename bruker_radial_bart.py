#!/usr/bin/env python3
"""
bruker_radial_bart.py

Reset, clean Python wrapper for 3D radial reconstruction using BART.

Features
- Bruker-centric CLI: accepts a study/scan directory (PV 6.0.1-style)
- Trajectory: built-in 3D golden-angle (kooshball) or load from file
- DCF: pipe-style iterative estimator or none (placeholder for Voronoi)
- Recon paths:
  * Gridding (adjoint NUFFT) + SoS combine
  * Iterative PICS (CG-SENSE / L1-wavelet optional) with NUFFT operator
- **GPU toggle**: `--gpu` adds BART global `-g` to enable CUDA when available
- Exports NIfTI via `bart toimg`

Notes
- This wrapper assumes BART is installed and `bart` is on PATH.
- Reading raw Bruker FID is instrument/sequence-specific. A minimal hook is
  provided (`load_bruker_kspace(...)`). Replace that with your lab's loader,
  or use the new `--from-fid` pathway with dtype/endianness hints.
- All interchange with BART uses .cfl/.hdr files via helpers here.

Example
-------
# Adjoint gridding + SoS
python bruker_radial_bart.py \
  --series /path/to/Bruker/2 \
  --matrix 256 256 256 \
  --spokes 200000 \
  --readout 256 \
  --traj golden \
  --dcf pipe:10 \
  --combine sos \
  --gpu \
  --out img_sos --export-nifti

# Iterative PICS (CG-SENSE with L1-wavelets)
python bruker_radial_bart.py \
  --series /path/to/Bruker/2 \
  --matrix 256 256 256 \
  --spokes 200000 \
  --readout 256 \
  --traj golden \
  --dcf pipe:10 \
  --iterative \
  --lambda 0.002 \
  --iters 40 \
  --wavelets 3 \
  --gpu \
  --out img_iter --export-nifti

"""

from __future__ import annotations
import argparse
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# -----------------------------
# Utilities: BART .cfl IO
# -----------------------------

def _write_hdr(path: Path, dims: Tuple[int, ...]):
    with open(path, 'w') as f:
        f.write('# Dimensions\n')
        f.write(' '.join(str(d) for d in dims) + '\n')


def write_cfl(name: Path, array: np.ndarray):
    """Write complex array to BART .cfl/.hdr with column-major layout.
    Array expected as complex64/complex128, any shape up to 16 dims.
    """
    name = Path(name)
    base = name.with_suffix('')
    dims = list(array.shape) + [1] * (16 - array.ndim)
    _write_hdr(base.with_suffix('.hdr'), tuple(dims))
    # BART uses column-major (Fortran) order
    arr = np.asarray(array, dtype=np.complex64, order='F')
    arr.view(np.float32).tofile(base.with_suffix('.cfl'))


def read_cfl(name: Path) -> np.ndarray:
    name = Path(name)
    base = name.with_suffix('')
    with open(base.with_suffix('.hdr'), 'r') as f:
        lines = f.read().strip().splitlines()
    dims = tuple(int(x) for x in lines[1].split())
    dims = tuple(d for d in dims if d > 0)
    data = np.fromfile(base.with_suffix('.cfl'), dtype=np.complex64)
    return np.reshape(data, dims, order='F')


# -----------------------------
# Bruker loader hooks
# -----------------------------

def _read_text_kv(path: Path) -> dict:
    d = {}
    if not path.exists():
        return d
    for line in path.read_text(errors='ignore').splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            d[k].strip() if k in d else None
            d[k.strip()] = v.strip()
    return d


def _parse_acq_size(method_txt: dict, acqp_txt: dict) -> Optional[Tuple[int, int, int]]:
    # Try to parse ACQ_size or PVM_Matrix
    for key in ('ACQ_size', 'PVM_Matrix'):  # {RO,PE1,PE2}
        if key in method_txt:
            try:
                nums = [int(x) for x in method_txt[key].replace('{', ' ').replace('}', ' ').split() if x.isdigit()]
                if len(nums) >= 3:
                    return (nums[0], nums[1], nums[2])
            except Exception:
                pass
    return None


def load_bruker_kspace(series_dir: Path,
                        spokes: Optional[int] = None,
                        readout: Optional[int] = None,
                        coils: Optional[int] = None,
                        from_fid: bool = False,
                        fid_dtype: str = 'int32',
                        fid_endian: str = 'little') -> np.ndarray:
    """Return k-space as complex array with shape (readout, spokes, coils).

    Priority:
      1) series_dir/ksp.cfl (BART format) -> return loaded
      2) series_dir/ksp.npy  (numpy complex64) with shape (RO, Spokes, Coils)
      3) If `from_fid` **or fid exists with no ksp**, attempt raw read using header hints
      4) NotImplementedError with clear message
    """
    series_dir = Path(series_dir)
    cfl = series_dir / 'ksp'
    npy = series_dir / 'ksp.npy'
    if cfl.with_suffix('.cfl').exists() and cfl.with_suffix('.hdr').exists():
        return read_cfl(cfl)
    if npy.exists():
        arr = np.load(npy)
        if arr.ndim != 3:
            raise ValueError(f"ksp.npy must be 3D (readout, spokes, coils); got {arr.shape}")
        return arr

    fid_path = series_dir / 'fid'
    auto_from_fid = fid_path.exists()
    if from_fid or auto_from_fid:
        method_txt = _read_text_kv(series_dir / 'method')
        acqp_txt = _read_text_kv(series_dir / 'acqp')
        # Infer defaults from headers when possible
        if coils is None:
            for key in ('PVM_EncNReceivers', 'ACQ_ReceiverSelect', 'PVM_EncChanScaling'):  # primary -> fallback heuristics
                if key in method_txt:
                    try:
                        nums = [int(x) for x in method_txt[key].replace('{',' ').replace('}',' ').replace('(',' ').replace(')',' ').split() if x.isdigit()]
                        if nums:
                            coils = nums[0]
                            break
                    except Exception:
                        pass
        if readout is None or spokes is None:
            acq = _parse_acq_size(method_txt, acqp_txt)
            if acq and readout is None:
                readout = acq[0]
        # Endianness from BYTORDA
        if 'BYTORDA' in acqp_txt:
            val = acqp_txt['BYTORDA'].lower()
            fid_endian = 'little' if 'little' in val else ('big' if 'big' in val else fid_endian)
        # Word size -> dtype
        if 'ACQ_word_size' in acqp_txt:
            try:
                ws = int(''.join([c for c in acqp_txt['ACQ_word_size'] if c.isdigit()]))
                if ws == 16:
                    fid_dtype = 'int16'
                elif ws == 32:
                    fid_dtype = 'int32'
            except Exception:
                pass
        # Read raw
        dtype_map = {
            'int16': np.int16,
            'int32': np.int32,
            'float32': np.float32,
            'float64': np.float64,
        }
        if fid_dtype not in dtype_map:
            raise ValueError('--fid-dtype must be one of int16,int32,float32,float64')
        dt = dtype_map[fid_dtype]
        if not fid_path.exists():
            raise FileNotFoundError(f"No 'fid' file found in {series_dir}")
        raw = np.fromfile(fid_path, dtype=dt)
        if fid_endian == 'big':
            raw = raw.byteswap().newbyteorder()
        if raw.size % 2 != 0:
            raise ValueError('FID length not even: cannot form complex pairs')
        raw = raw.astype(np.float32)
        cpx = raw.view(np.complex64)
        if readout is None or spokes is None or coils is None:
            raise ValueError('Cannot infer (readout, spokes, coils) from headers; please pass --readout/--spokes/--coils')
        expected = readout * spokes * coils
        if cpx.size != expected:
            # Try infer coils if missing/mismatched
            if coils and cpx.size % (readout * spokes) == 0:
                coils = cpx.size // (readout * spokes)
                expected = readout * spokes * coils
            if cpx.size != expected:
                raise ValueError(f"FID size mismatch: have {cpx.size}, expected {expected} (ro*sp*coils); check --coils/--readout/--spokes")
        arr = np.reshape(cpx, (readout, spokes, coils), order='F')
        return arr

    raise NotImplementedError(
        "No k-space found. Provide ksp.cfl/.hdr or ksp.npy in the series directory, "
        "or include a 'fid' file (auto-detected) or use --from-fid with --readout/--spokes/--coils and dtype/endian hints.")


# -----------------------------
# Trajectory generators
# -----------------------------

@dataclass
class TrajSpec:
    readout: int
    spokes: int
    matrix: Tuple[int, int, int]
    fov: Optional[Tuple[float, float, float]] = None  # optional meters


def golden_angle_3d(spec: TrajSpec) -> np.ndarray:
    """3D kooshball golden-angle trajectory in units of k-space pixels.

    Returns array shape (3, readout, spokes).
    * Direction set via spherical Fibonacci ordering (quasi-uniform)
    * Readout samples linearly from -kmax..+kmax (Nyquist-normalized)
    """
    ro, sp = spec.readout, spec.spokes
    nx, ny, nz = spec.matrix
    kmax = 0.5 * max(nx, ny, nz)
    i = np.arange(sp) + 0.5
    phi = 2.0 * math.pi * i / ((1 + math.sqrt(5)) / 2.0)  # golden ratio spacing
    cos_theta = 1 - 2 * i / sp
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    dirs = np.stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ], axis=1)  # (spokes, 3)
    t = np.linspace(-1.0, 1.0, ro, endpoint=True)
    radii = kmax * t
    xyz = np.einsum('sr,sd->drs', radii[None, :], dirs)  # (3, ro, spokes)
    return xyz.astype(np.float32)


def save_traj_for_bart(traj: np.ndarray, out_base: Path):(traj: np.ndarray, out_base: Path):
    """Save trajectory as BART .cfl/.hdr with dims (3, RO, Spokes)."""
    assert traj.shape[0] == 3, "Trajectory must have leading dim 3 (kx,ky,kz)."
    write_cfl(out_base, traj)


# -----------------------------
# Density Compensation (pipe-style iterative)
# -----------------------------

def dcf_pipe(traj: np.ndarray, iters: int = 10, grid_shape: Tuple[int, int, int] = (256, 256, 256), gpu: bool = False) -> np.ndarray:
    """Simple iterative DCF (pipe): w_{n+1} = w_n / G^H G w_n on ones.

    Returns DCF with shape (readout, spokes).
    """
    ro, sp = traj.shape[1], traj.shape[2]
    w = np.ones((ro, sp), dtype=np.float32)

    # attempt BART-based iteration by shuttling through .cfl files
    bart = shutil.which('bart')
    if bart is None:
        # crude fallback: bin samples to nearest voxel and use inverse of hit-counts
        kx = np.rint(traj[0] - traj[0].min()).astype(int)
        ky = np.rint(traj[1] - traj[1].min()).astype(int)
        kz = np.rint(traj[2] - traj[2].min()).astype(int)
        kx = np.clip(kx, 0, grid_shape[0]-1)
        ky = np.clip(ky, 0, grid_shape[1]-1)
        kz = np.clip(kz, 0, grid_shape[2]-1)
        hist = np.zeros(grid_shape, dtype=np.float32)
        for j in range(sp):
            np.add.at(hist, (kx[:, j], ky[:, j], kz[:, j]), 1)
        sample_counts = hist[kx, ky, kz] + 1e-6
        return (1.0 / sample_counts).astype(np.float32)

    tmp = Path('./.tmp_bart_dcf')
    tmp.mkdir(exist_ok=True)

    def bart_cmd(*args):
        base = [bart]
        if gpu:
            base.append('-g')
        return base + list(args)

    try:
        for _ in range(iters):
            # ones -> forward NUFFT -> weighted adjoint -> update
            ones = np.ones((grid_shape[0], grid_shape[1], grid_shape[2]), dtype=np.complex64)
            write_cfl(tmp / 'ones', ones)
            save_traj_for_bart(traj, tmp / 'traj')
            # forward NUFFT: kspace of ones
            subprocess.run(bart_cmd('nufft', '-t', str(tmp / 'traj'), str(tmp / 'ones'), str(tmp / 'kones')), check=True)
            # apply current weights
            write_cfl(tmp / 'w', w.astype(np.complex64))
            subprocess.run(bart_cmd('fmac', str(tmp / 'kones'), str(tmp / 'w'), str(tmp / 'kw')), check=True)
            # adjoint NUFFT
            subprocess.run(bart_cmd('nufft', '-a', '-t', str(tmp / 'traj'), str(tmp / 'kw'), str(tmp / 'backproj')), check=True)
            back = read_cfl(tmp / 'backproj')
            # sample backprojected values at sample locations via second forward
            write_cfl(tmp / 'back', back)
            subprocess.run(bart_cmd('nufft', '-t', str(tmp / 'traj'), str(tmp / 'back'), str(tmp / 'kback')), check=True)
            kback = read_cfl(tmp / 'kback')
            denom = np.abs(kback).astype(np.float32) + 1e-6
            w = w / denom
        return w.astype(np.float32)
    finally:
        try:
            shutil.rmtree(tmp)
        except Exception:
            pass


# -----------------------------
# BART ops (GPU-aware)
# -----------------------------

def run_bart(cmd: list[str], gpu: bool = False):
    bart = shutil.which('bart')
    if not bart:
        raise RuntimeError('BART not found in PATH')
    full = [bart]
    if gpu:
        full.append('-g')
    full += cmd
    print('[bart]', ' '.join(full))
    subprocess.run(full, check=True)


def bart_exists() -> bool:
    return shutil.which('bart') is not None


# -----------------------------
# Coil maps (ESPiRIT) helper
# -----------------------------

def estimate_sens_maps(coil_imgs_base: Path, out_base: Path, calib: Optional[int] = None, gpu: bool = False):
    cmd = ['ecalib']
    if calib is not None:
        cmd += ['-r', str(calib)]
    cmd += [str(coil_imgs_base), str(out_base)]
    run_bart(cmd, gpu=gpu)


# -----------------------------
# Main recon flows
# -----------------------------

def recon_adjoint(traj_base: Path, ksp_base: Path, combine: str, out_base: Path, gpu: bool):
    # NUFFT adjoint to coil images
    coil_base = out_base.with_name(out_base.name + '_coil')
    run_bart(['nufft', '-a', '-t', str(traj_base), str(ksp_base), str(coil_base)], gpu=gpu)

    if combine.lower() == 'sos':
        run_bart(['rss', '8', str(coil_base), str(out_base)], gpu=gpu)
    elif combine.lower() == 'sens':
        maps = out_base.with_name(out_base.name + '_maps')
        estimate_sens_maps(coil_base, maps, gpu=gpu)
        run_bart(['pics', '-S', str(coil_base), str(maps), str(out_base)], gpu=gpu)
    else:
        raise ValueError('combine must be sos|sens')


def recon_iterative(traj_base: Path, ksp_base: Path, out_base: Path,
                    lam: float, iters: int, wavelets: Optional[int], gpu: bool):
    # quick coil maps: adjoint per coil -> ecalib
    tmpcoil = out_base.with_name(out_base.name + '_coil')
    run_bart(['nufft', '-a', '-t', str(traj_base), str(ksp_base), str(tmpcoil)], gpu=gpu)
    maps = out_base.with_name(out_base.name + '_maps')
    estimate_sens_maps(tmpcoil, maps, gpu=gpu)

    # PICS with NUFFT operator
    cmd = ['pics', '-S', '-i', str(iters)]
    if wavelets is not None:
        cmd += ['-R', f'W:7:{wavelets}:{lam}']
    elif lam > 0:
        cmd += ['-R', f'W:7:0:{lam}']
    cmd += ['-t', str(traj_base), str(ksp_base), str(maps), str(out_base)]
    run_bart(cmd, gpu=gpu)


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description='Bruker 3D radial reconstruction using BART')
    ap.add_argument('--series', type=Path, required=True, help='Path to Bruker scan directory (PV 6.x) or folder containing ksp.*')
    ap.add_argument('--out', type=Path, required=True, help='Output basename (no extension) for BART/NIfTI')
    ap.add_argument('--matrix', type=int, nargs=3, required=True, metavar=('NX','NY','NZ'))
    ap.add_argument('--spokes', type=int, required=True)
    ap.add_argument('--readout', type=int, required=True)
    ap.add_argument('--traj', choices=['golden', 'file'], default='file')
    ap.add_argument('--traj-file', type=Path, help='If --traj file, path to traj.cfl/.hdr or .npy with shape (3,RO,Spokes)')
    ap.add_argument('--dcf', type=str, default='none', help='DCF mode: none | pipe:Niters')
    ap.add_argument('--combine', type=str, default='sos', help='Coil combine for adjoint: sos|sens')
    ap.add_argument('--iterative', action='store_true', help='Use iterative PICS reconstruction')
    ap.add_argument('--lambda', dest='lam', type=float, default=0.0, help='Regularization weight for iterative recon')
    ap.add_argument('--iters', type=int, default=40, help='Iterations for iterative recon')
    ap.add_argument('--wavelets', type=int, default=None, help='Wavelet scale parameter (optional)')
    ap.add_argument('--export-nifti', action='store_true', help='Export NIfTI via bart toimg')
    ap.add_argument('--gpu', action='store_true', help='Enable BART GPU (adds global -g)')
    # optional raw FID path
    ap.add_argument('--from-fid', action='store_true', help="Attempt to read raw Bruker 'fid' when ksp.* not present")
    ap.add_argument('--coils', type=int, default=None, help='Number of coils (required with --from-fid if not inferrable)')
    ap.add_argument('--fid-dtype', type=str, default='int32', choices=['int16','int32','float32','float64'], help='Scalar type of fid samples before complex pairing')
    ap.add_argument('--fid-endian', type=str, default='little', choices=['little','big'], help='Endianness of fid file')

    args = ap.parse_args()

    if not bart_exists():
        print('ERROR: BART not found on PATH. Install BART and try again.', file=sys.stderr)
        sys.exit(1)

    series_dir: Path = args.series
    out_base: Path = args.out
    nx, ny, nz = args.matrix
    ro = args.readout
    sp = args.spokes

    # ---- k-space (readout, spokes, coils)
    ksp = load_bruker_kspace(series_dir, spokes=sp, readout=ro, coils=args.coils,
                             from_fid=args.from_fid, fid_dtype=args.fid_dtype, fid_endian=args.fid_endian)
    if ksp.shape[0] != ro or ksp.shape[1] != sp:
        raise ValueError(f"k-space shape mismatch: got {ksp.shape}, expected (readout={ro}, spokes={sp}, coils)")

    # Save k-space for BART (dims: RO, Spokes, Coils)
    write_cfl(out_base.with_name(out_base.name + '_ksp'), ksp)

    # ---- trajectory (3, ro, sp)
    if args.traj == 'golden':
        traj = golden_angle_3d(TrajSpec(readout=ro, spokes=sp, matrix=(nx, ny, nz)))
    else:
        # default: try to load from series_dir/traj.{cfl,hdr} or traj.npy when --traj-file not given
        if args.traj_file is None:
            default_traj_base = series_dir / 'traj'
            if default_traj_base.with_suffix('.cfl').exists() and default_traj_base.with_suffix('.hdr').exists():
                traj = read_cfl(default_traj_base)
            elif (series_dir / 'traj.npy').exists():
                traj = np.load(series_dir / 'traj.npy')
            else:
                raise ValueError("--traj file requires --traj-file path or a default 'traj.cfl/.hdr' or 'traj.npy' in the series directory")
        else:
            if args.traj_file.suffix == '.npy':
                traj = np.load(args.traj_file)
            else:
                traj = read_cfl(args.traj_file)
    if traj.shape != (3, ro, sp):
        raise ValueError(f"trajectory must have shape (3,{ro},{sp}), got {traj.shape}")

    save_traj_for_bart(traj, out_base.with_name(out_base.name + '_traj'))

    # ---- optional DCF
    dcf_mode = args.dcf.lower()
    if dcf_mode.startswith('pipe'):
        nit = 10
        if ':' in dcf_mode:
            try:
                nit = int(dcf_mode.split(':', 1)[1])
            except Exception:
                pass
        dcf = dcf_pipe(traj, iters=nit, grid_shape=(nx, ny, nz), gpu=args.gpu)
        write_cfl(out_base.with_name(out_base.name + '_dcf'), dcf.astype(np.complex64))
        # weight kspace in-place for adjoint path; for PICS we can pass unweighted
        run_bart(['fmac', str(out_base.with_name(out_base.name + '_ksp')), str(out_base.with_name(out_base.name + '_dcf')), str(out_base.with_name(out_base.name + '_kspw'))], gpu=args.gpu)
        ksp_base = out_base.with_name(out_base.name + '_kspw')
    else:
        ksp_base = out_base.with_name(out_base.name + '_ksp')

    # ---- reconstruction
    if args.iterative:
        recon_iterative(out_base.with_name(out_base.name + '_traj'), ksp_base, out_base.with_name(out_base.name + '_recon'),
                        lam=args.lam, iters=args.iters, wavelets=args.wavelets, gpu=args.gpu)
        img_base = out_base.with_name(out_base.name + '_recon')
    else:
        recon_adjoint(out_base.with_name(out_base.name + '_traj'), ksp_base, args.combine, out_base.with_name(out_base.name + '_adj'), gpu=args.gpu)
        img_base = out_base.with_name(out_base.name + '_adj')

    # ---- export
    if args.export_nifti:
        run_bart(['toimg', str(img_base), str(out_base)], gpu=args.gpu)
        print(f'Wrote NIfTI: {out_base}.nii')
    else:
        print(f'Recon complete. BART base: {img_base} (.cfl/.hdr)')


if __name__ == '__main__':
    main()
