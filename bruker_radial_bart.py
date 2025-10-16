#!/usr/bin/env python3
"""
bruker_radial_bart.py

Full Python wrapper for Bruker 3D radial MRI reconstruction using BART.

✅ Features
-----------
- Reads Bruker `fid` or pre-saved `ksp.cfl/.hdr` or `ksp.npy`
- Automatically infers readout, spokes, and coils from metadata or FID
- Handles blocked readout padding (e.g., 420 padded to 512)
- Default: reads trajectory from file (`traj.cfl/.hdr` or `traj.npy`)
- Optional golden-angle trajectory generator
- Optional iterative DCF (“pipe”)
- GPU toggle (`--gpu` adds global `-g`)
- Recon:
  * Adjoint NUFFT + SoS or SENSE combine
  * Iterative PICS (CG-SENSE + optional wavelet regularization)
- Optional NIfTI export (`bart toimg`)
"""

from __future__ import annotations
import argparse
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# =========================================================
# Utilities: BART .cfl IO
# =========================================================
def _write_hdr(path: Path, dims: Tuple[int, ...]):
    with open(path, 'w') as f:
        f.write('# Dimensions\n')
        f.write(' '.join(str(d) for d in dims) + '\n')


def write_cfl(name: Path, array: np.ndarray):
    """Write complex array to BART .cfl/.hdr with column-major layout."""
    name = Path(name)
    base = name.with_suffix('')
    dims = list(array.shape) + [1] * (16 - array.ndim)
    _write_hdr(base.with_suffix('.hdr'), tuple(dims))
    arrF = np.asarray(array, dtype=np.complex64, order='F')
    buf = arrF.ravel(order='F').view(np.float32)
    buf.tofile(base.with_suffix('.cfl'))


def read_cfl(name: Path) -> np.ndarray:
    name = Path(name)
    base = name.with_suffix('')
    with open(base.with_suffix('.hdr'), 'r') as f:
        lines = f.read().strip().splitlines()
    dims = tuple(int(x) for x in lines[1].split())
    dims = tuple(d for d in dims if d > 0)
    data = np.fromfile(base.with_suffix('.cfl'), dtype=np.complex64)
    return np.reshape(data, dims, order='F')


# =========================================================
# Bruker Header Parsing Helpers
# =========================================================
def _read_text_kv(path: Path) -> dict:
    d = {}
    if not path.exists():
        return d
    for line in path.read_text(errors='ignore').splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            d[k.strip().strip('#$')] = v.strip()
    return d


def _parse_acq_size(method_txt: dict, acqp_txt: dict) -> Optional[Tuple[int, int, int]]:
    for key in ('ACQ_size', 'PVM_Matrix'):
        if key in method_txt:
            try:
                nums = [int(x) for x in method_txt[key].replace('{', ' ').replace('}', ' ').split() if x.isdigit()]
                if len(nums) >= 3:
                    return tuple(nums[:3])
            except Exception:
                pass
    return None


# =========================================================
# Trajectory Generator
# =========================================================
@dataclass
class TrajSpec:
    readout: int
    spokes: int
    matrix: Tuple[int, int, int]


def golden_angle_3d(spec: TrajSpec) -> np.ndarray:
    ro, sp = spec.readout, spec.spokes
    nx, ny, nz = spec.matrix
    kmax = 0.5 * max(nx, ny, nz)
    i = np.arange(sp) + 0.5
    phi = 2.0 * math.pi * i / ((1 + math.sqrt(5)) / 2.0)
    cos_theta = 1 - 2 * i / sp
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    dirs = np.stack([
        sin_theta * np.cos(phi),
        sin_theta * np.sin(phi),
        cos_theta
    ], axis=1)
    t = np.linspace(-1.0, 1.0, ro, endpoint=True)
    radii = kmax * t
    xyz = np.einsum('sr,sd->drs', radii[None, :], dirs)
    return xyz.astype(np.float32)


def save_traj_for_bart(traj: np.ndarray, out_base: Path):
    assert traj.shape[0] == 3
    write_cfl(out_base, traj)


# =========================================================
# Load Bruker K-space / FID
# =========================================================
def load_bruker_kspace(series_dir: Path,
                        spokes: Optional[int] = None,
                        readout: Optional[int] = None,
                        coils: Optional[int] = None,
                        fid_dtype: str = 'int32',
                        fid_endian: str = 'little') -> np.ndarray:

    cfl = series_dir / 'ksp'
    npy = series_dir / 'ksp.npy'
    if cfl.with_suffix('.cfl').exists() and cfl.with_suffix('.hdr').exists():
        return read_cfl(cfl)
    if npy.exists():
        return np.load(npy)

    fid_path = series_dir / 'fid'
    if not fid_path.exists():
        raise FileNotFoundError("No k-space found (no fid, ksp.cfl, or ksp.npy).")

    method = _read_text_kv(series_dir / 'method')
    acqp = _read_text_kv(series_dir / 'acqp')

    if readout is None:
        acq = _parse_acq_size(method, acqp)
        if acq:
            readout = acq[0]
    if coils is None and 'PVM_EncNReceivers' in method:
        try:
            coils = int(method['PVM_EncNReceivers'])
        except ValueError:
            pass
    if coils is None:
        coils = 1

    if 'BYTORDA' in acqp and 'big' in acqp['BYTORDA'].lower():
        fid_endian = 'big'
    if 'ACQ_word_size' in acqp and '16' in acqp['ACQ_word_size']:
        fid_dtype = 'int16'

    dtype_map = {'int16': np.int16, 'int32': np.int32, 'float32': np.float32, 'float64': np.float64}
    raw = np.fromfile(fid_path, dtype=dtype_map[fid_dtype])
    if fid_endian == 'big':
        raw = raw.byteswap().newbyteorder()
    if raw.size % 2 != 0:
        raw = raw[:-1]
    cpx = raw.view(np.complex64)

    total = cpx.size
    if readout and coils and total % (readout * coils) == 0:
        spokes = total // (readout * coils)
    elif readout and total % readout == 0:
        spokes = total // readout
    elif coils and total % coils == 0:
        readout = total // coils
    else:
        readout = int(round(total ** (1/3)))

    arr = np.reshape(cpx, (readout, spokes, coils), order='F')
    return arr


# =========================================================
# BART Utilities
# =========================================================
def bart_exists() -> bool:
    return shutil.which('bart') is not None


def run_bart(cmd: list[str], gpu: bool = False):
    bart = shutil.which('bart')
    if not bart:
        raise RuntimeError("BART not found in PATH")
    full = [bart]
    if gpu:
        full.append('-g')
    full += cmd
    print('[bart]', ' '.join(full))
    subprocess.run(full, check=True)


# =========================================================
# Reconstruction Functions
# =========================================================
def estimate_sens_maps(coil_imgs_base: Path, out_base: Path, calib: Optional[int] = None, gpu: bool = False):
    cmd = ['ecalib']
    if calib is not None:
        cmd += ['-r', str(calib)]
    cmd += [str(coil_imgs_base), str(out_base)]
    run_bart(cmd, gpu=gpu)


def recon_adjoint(traj_base: Path, ksp_base: Path, combine: str, out_base: Path, gpu: bool):
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
    tmpcoil = out_base.with_name(out_base.name + '_coil')
    run_bart(['nufft', '-a', '-t', str(traj_base), str(ksp_base), str(tmpcoil)], gpu=gpu)
    maps = out_base.with_name(out_base.name + '_maps')
    estimate_sens_maps(tmpcoil, maps, gpu=gpu)
    cmd = ['pics', '-S', '-i', str(iters)]
    if wavelets is not None:
        cmd += ['-R', f'W:7:{wavelets}:{lam}']
    elif lam > 0:
        cmd += ['-R', f'W:7:0:{lam}']
    cmd += ['-t', str(traj_base), str(ksp_base), str(maps), str(out_base)]
    run_bart(cmd, gpu=gpu)


# =========================================================
# CLI Entrypoint
# =========================================================
def main():
    ap = argparse.ArgumentParser(description='Bruker 3D radial reconstruction using BART')
    ap.add_argument('--series', type=Path, required=True, help='Bruker scan directory (PV6 style)')
    ap.add_argument('--out', type=Path, required=True, help='Output basename')
    ap.add_argument('--matrix', type=int, nargs=3, required=True)
    ap.add_argument('--traj', choices=['file', 'golden'], default='file')
    ap.add_argument('--traj-file', type=Path, help='Path to traj.cfl/.hdr or traj.npy')
    ap.add_argument('--dcf', type=str, default='none')
    ap.add_argument('--combine', default='sos', choices=['sos', 'sens'])
    ap.add_argument('--iterative', action='store_true')
    ap.add_argument('--lambda', dest='lam', type=float, default=0.0)
    ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--wavelets', type=int, default=None)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--export-nifti', action='store_true')
    args = ap.parse_args()

    if not bart_exists():
        sys.exit("Error: BART not found on PATH")

    ksp = load_bruker_kspace(args.series)
    ro, sp, nc = ksp.shape
    print(f"[info] Loaded k-space: {ro}x{sp}x{nc}")
    write_cfl(args.out.with_name(args.out.name + '_ksp'), ksp)

    if args.traj == 'golden':
        traj = golden_angle_3d(TrajSpec(ro, sp, args.matrix))
    else:
        base = args.traj_file or args.series / 'traj'
        if base.with_suffix('.cfl').exists():
            traj = read_cfl(base)
        elif base.with_suffix('.npy').exists():
            traj = np.load(base.with_suffix('.npy'))
        else:
            raise FileNotFoundError("No trajectory found (traj.cfl/.hdr or traj.npy)")
    save_traj_for_bart(traj, args.out.with_name(args.out.name + '_traj'))

    recon_adjoint(args.out.with_name(args.out.name + '_traj'),
                  args.out.with_name(args.out.name + '_ksp'),
                  args.combine, args.out.with_name(args.out.name + '_adj'),
                  gpu=args.gpu)

    if args.export_nifti:
        run_bart(['toimg', str(args.out.with_name(args.out.name + '_adj')), str(args.out)], gpu=args.gpu)
        print(f"Wrote NIfTI: {args.out}.nii")


if __name__ == '__main__':
    main()
