#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib

DEBUG = False
USE_GPU = False  # will be set from args
GPU_SUPPORTED = True  # we will probe this lazily


def dbg(*a):
    if DEBUG:
        print("[debug]", *a)


# ---------- Bruker helpers ----------

def parse_param_file(path: Path) -> Dict[str, str]:
    """Parse Bruker-style parameter file into KEY -> value string.

    Handles array-valued params like:

        ##$PVM_Matrix=( 3 )
        96 96 96

    so that d["PVM_Matrix"] == "96 96 96".
    """
    d: Dict[str, str] = {}
    if not path.exists():
        return d

    current_key: Optional[str] = None
    collecting_array = False

    with open(path, "r") as f:
        for line in f:
            raw = line.rstrip("\n")
            line = raw.strip()

            # New parameter line
            if line.startswith("##$"):
                collecting_array = False
                current_key = None

                body = line[3:]  # drop "##$"
                if "=" not in body:
                    continue
                key, val = body.split("=", 1)
                key = key.strip()
                val = val.strip()

                # Array-style: values on subsequent line(s)
                if val.startswith("("):
                    d[key] = ""
                    current_key = key
                    collecting_array = True
                else:
                    d[key] = val
                continue

            # comment/vis lines mark end of array collect
            if line.startswith("##") or line.startswith("$$"):
                collecting_array = False
                current_key = None
                continue

            # Collect numeric lines for current array param
            if collecting_array and current_key is not None:
                if line:
                    if d[current_key]:
                        d[current_key] += " " + line
                    else:
                        d[current_key] = line

    return d


def _parse_int_list(val: str) -> List[int]:
    out: List[int] = []
    if not val:
        return out
    cleaned = val.replace("(", " ").replace(")", " ").replace(",", " ")
    for tok in cleaned.split():
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return out


def infer_matrix(method: Dict[str, str]) -> Tuple[int, int, int]:
    val = method.get("PVM_Matrix", "")
    ints = _parse_int_list(val)
    if len(ints) >= 3:
        return ints[0], ints[1], ints[2]
    raise ValueError("Could not infer matrix from PVM_Matrix; please pass --matrix NX NY NZ.")


def infer_coils(method: Dict[str, str], default: int = 1) -> int:
    val = method.get("PVM_EncNReceivers", "")
    ints = _parse_int_list(val)
    if ints:
        return ints[0]
    cleaned = val.replace("(", " ").replace(")", " ")
    for tok in cleaned.split():
        if tok.isdigit():
            return int(tok)
    return default


def parse_param_file_acqp(path: Path) -> Dict[str, str]:
    return parse_param_file(path)


def infer_fid_dtype(acqp: Dict[str, str]) -> str:
    val = acqp.get("ACQ_word_size", "")
    if "16" in val:
        return "int16"
    if "32" in val:
        return "int32"
    return "int32"


def infer_fid_endian(acqp: Dict[str, str]) -> str:
    val = acqp.get("BYTORDA", "").lower()
    if "big" in val:
        return "big"
    return "little"


def infer_true_ro_and_acq_spokes(acqp: Dict[str, str]) -> Tuple[int, Optional[int]]:
    """From ACQ_size=(3) true_ro acq_spokes ?"""
    val = acqp.get("ACQ_size", "")
    ints = _parse_int_list(val)
    if len(ints) >= 1:
        true_ro = ints[0]
    else:
        raise ValueError("Could not infer true RO from ACQ_size.")

    acq_spokes = ints[1] if len(ints) >= 2 else None
    return true_ro, acq_spokes


def load_bruker_kspace(
    series_dir: Path,
    coils: int,
    fid_dtype: Optional[str] = None,
    fid_endian: Optional[str] = None,
) -> Tuple[np.ndarray, int, int]:
    """Load Bruker FID and return (kspace, true_ro, spokes).

    Handles Bruker padding to a power-of-two block along RO. We infer:
      - true_ro from ACQ_size[0]
      - ACQ spokes from ACQ_size[1] (if present)
      - stored_ro (e.g. 128) and total spokes from FID length.
    """
    fid_path = series_dir / "fid"
    if not fid_path.exists():
        raise FileNotFoundError(f"No fid file in series dir: {fid_path}")

    method = parse_param_file(series_dir / "method")
    acqp = parse_param_file_acqp(series_dir / "acqp")

    if fid_dtype is None:
        fid_dtype = infer_fid_dtype(acqp)
    if fid_endian is None:
        fid_endian = infer_fid_endian(acqp)
    if coils is None or coils <= 0:
        coils = infer_coils(method, default=1)

    true_ro, acq_spokes = infer_true_ro_and_acq_spokes(acqp)

    dbg("fid_dtype:", fid_dtype, "fid_endian:", fid_endian, "coils:", coils)
    dbg("true_ro (ACQ_size[0]):", true_ro, "acq_spokes (ACQ_size[1]):", acq_spokes)

    dt_map = {
        "int16": np.int16,
        "int32": np.int32,
        "float32": np.float32,
        "float64": np.float64,
    }
    if fid_dtype not in dt_map:
        raise ValueError(f"Unsupported fid_dtype={fid_dtype}")
    dt = dt_map[fid_dtype]

    raw = np.fromfile(fid_path, dtype=dt)
    if fid_endian == "big":
        raw = raw.byteswap().newbyteorder()

    if raw.size % 2 != 0:
        raw = raw[:-1]
    cpx = raw.astype(np.float32).view(np.complex64)
    total = cpx.size
    dbg("total complex samples:", total)

    # Infer stored_ro (padded block) and spokes from total length.
    # Use a small set of plausible block sizes including true_ro.
    candidate_blocks = sorted(set([true_ro, 128, 256, 512, 1024]))
    stored_ro = None
    spokes = None
    repeats = None

    for blk in candidate_blocks:
        denom = blk * coils
        if total % denom != 0:
            continue
        sp = total // denom
        rep = None
        if acq_spokes and acq_spokes > 0 and sp % acq_spokes == 0:
            rep = sp // acq_spokes
        elif not acq_spokes:
            rep = 1
        if rep is not None:
            stored_ro = blk
            spokes = sp
            repeats = rep
            break

    if stored_ro is None or spokes is None:
        raise ValueError(
            f"Could not factor FID into (stored_ro * spokes * coils). "
            f"total={total}, true_ro={true_ro}, coils={coils}"
        )

    dbg("stored_ro:", stored_ro, "spokes:", spokes, "repeats_per_traj:", repeats)

    # Reshape to (stored_ro, spokes, coils)
    if total != stored_ro * spokes * coils:
        raise ValueError(
            f"Inferred stored_ro={stored_ro}, spokes={spokes}, coils={coils} "
            f"but stored_ro*spokes*coils={stored_ro*spokes*coils} != total={total}"
        )

    ksp_full = np.reshape(cpx, (stored_ro, spokes, coils), order="F")

    # Trim padded samples down to true_ro along RO
    if stored_ro < true_ro:
        raise ValueError(f"stored_ro={stored_ro} < true_ro={true_ro} ??")
    if stored_ro > true_ro:
        ksp = ksp_full[:true_ro, :, :]
        print(
            f"[info] Loaded k-space with stored_ro={stored_ro}, trimmed to true RO={true_ro}; "
            f"Spokes={spokes}, Coils={coils}"
        )
    else:
        ksp = ksp_full
        print(f"[info] Loaded k-space: RO={true_ro}, Spokes={spokes}, Coils={coils}")

    return ksp, true_ro, spokes


# ---------- Synthetic trajectory: Kronecker / LinZ-GA ----------

def fib_closest_ge(n: int) -> int:
    if n <= 1:
        return 1
    a, b = 1, 1
    while b < n:
        a, b = b, a + b
    return b


def fib_prev(fk: int) -> int:
    if fk <= 1:
        return 1
    a, b = 1, 1
    while b < fk:
        a, b = b, a + b
    return a


def fib_prev2(fk: int) -> int:
    return fib_prev(fib_prev(fk))


def uv_to_dir(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = 1.0 - 2.0 * u
    r2 = 1.0 - z * z
    r2 = np.maximum(r2, 0.0)
    r = np.sqrt(r2)
    az = 2.0 * np.pi * v
    dx = r * np.cos(az)
    dy = r * np.sin(az)
    return dx, dy, z


def kronecker_dirs(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = fib_closest_ge(N)
    q1 = fib_prev(M)
    q2 = fib_prev2(M)
    j = np.arange(N, dtype=np.int64) % M
    u = ((j * int(q1)) % M + 0.5) / float(M)
    v = ((j * int(q2)) % M + 0.5) / float(M)
    dx, dy, z = uv_to_dir(u, v)
    return dx, dy, z


def linz_ga_dirs(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi_inc = (math.sqrt(5.0) - 1.0) * math.pi
    i = np.arange(N, dtype=np.float64)
    z = 1.0 - 2.0 * ((i + 0.5) / float(N))
    r2 = 1.0 - z * z
    r2 = np.maximum(r2, 0.0)
    r = np.sqrt(r2)
    az = np.mod(i * phi_inc, 2.0 * math.pi)
    dx = r * np.cos(az)
    dy = r * np.sin(az)
    return dx, dy, z


def build_traj(mode: str, ro: int, spokes: int) -> np.ndarray:
    mode = mode.lower()
    if mode not in ("kron", "linz"):
        raise ValueError("traj-mode must be 'kron' or 'linz'")

    if mode == "kron":
        dx, dy, dz = kronecker_dirs(spokes)
    else:
        dx, dy, dz = linz_ga_dirs(spokes)

    # radius goes from -0.5..0.5 in k-space units (pixel_size/FOV)
    r = np.linspace(-0.5, 0.5, ro, endpoint=False, dtype=np.float32)
    traj = np.zeros((3, ro, spokes), dtype=np.float32)
    for s in range(spokes):
        traj[0, :, s] = r * dx[s]
        traj[1, :, s] = r * dy[s]
        traj[2, :, s] = r * dz[s]

    print(f"[info] Built synthetic {mode} trajectory: shape={traj.shape}")
    return traj


# ---------- CFL I/O ----------

def _write_hdr(path: Path, dims: List[int]):
    with open(path, "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in dims) + "\n")


def write_cfl(base: Path, arr: np.ndarray, dims: List[int]):
    arr = np.asarray(arr, dtype=np.complex64, order="F")
    if len(dims) < 16:
        dims = dims + [1] * (16 - len(dims))
    if int(np.prod(dims)) != arr.size:
        raise ValueError(f"dims product {np.prod(dims)} != array size {arr.size}")
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")
    _write_hdr(hdr, dims)
    flat = arr.flatten(order="F")
    out = np.empty((2, flat.size), dtype=np.float32)
    out[0, :] = flat.real.astype(np.float32)
    out[1, :] = flat.imag.astype(np.float32)
    out.tofile(cfl)


def load_cfl(base: Path) -> np.ndarray:
    """Load a BART-style CFL into a complex64 numpy array."""
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")
    with open(hdr, "r") as f:
        first = f.readline()
        if not first.startswith("#"):
            raise ValueError(f"Unexpected HDR header: {first}")
        dims_line = f.readline()
    dims = [int(x) for x in dims_line.split()]
    dims = [d for d in dims if d > 0]
    data = np.fromfile(cfl, dtype=np.float32)
    if data.size % 2 != 0:
        raise ValueError("CFL data length is not even (real/imag).")
    data = data.reshape(2, -1, order="F")
    cpx = data[0] + 1j * data[1]
    arr = cpx.reshape(dims, order="F").astype(np.complex64)
    return arr


# ---------- BART wrappers ----------

def bart_path() -> str:
    p = shutil.which("bart")
    if p is None:
        raise RuntimeError("BART executable 'bart' not found in PATH")
    return p


def run_bart(args: List[str], allow_gpu_fallback: bool = False):
    """Run BART with optional GPU support.

    If USE_GPU is True and allow_gpu_fallback is True, we will:
      - try adding '-g' to the nufft command
      - if BART says 'compiled without GPU support', re-run without '-g'
    """
    global GPU_SUPPORTED

    cmd_base = [bart_path()]
    full_args = args[:]

    use_gpu_here = False
    if USE_GPU and allow_gpu_fallback and GPU_SUPPORTED and args and args[0] == "nufft":
        # insert '-g' right after 'nufft'
        full_args = [args[0], "-g"] + args[1:]
        use_gpu_here = True

    cmd = cmd_base + full_args
    print("[bart]", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        if use_gpu_here and e.returncode != 0 and GPU_SUPPORTED:
            # try to detect 'compiled without GPU support'
            # run once without -g
            print(
                "[warn] BART compiled without GPU support; "
                "falling back to CPU NUFFT and disabling --gpu."
            )
            GPU_SUPPORTED = False
            # re-run without '-g'
            cpu_args = [a for a in args if a != "-g"]
            cmd = cmd_base + cpu_args
            print("[bart]", " ".join(cmd))
            subprocess.run(cmd, check=True)
        else:
            raise


# ---------- Frame binning ----------

def frame_starts(total_spokes: int, spf: int, shift: int) -> List[int]:
    if spf <= 0:
        raise ValueError("spokes-per-frame must be > 0")
    if shift <= 0:
        shift = spf
    starts: List[int] = []
    s = 0
    while s + spf <= total_spokes:
        starts.append(s)
        s += shift
    return starts


# ---------- Main recon driver ----------

def main():
    global DEBUG, USE_GPU

    ap = argparse.ArgumentParser(
        description="Bruker 3D radial recon using BART NUFFT (Kronecker/LinZ-GA + sliding window)."
    )
    ap.add_argument("--series", type=Path, required=True, help="Bruker series dir (contains fid, method, acqp)")
    ap.add_argument("--out", type=Path, required=True, help="Output prefix (basename only, not extension)")

    ap.add_argument("--matrix", type=int, nargs=3, help="Override matrix size NX NY NZ (default from PVM_Matrix)")
    ap.add_argument("--readout", type=int, help="Override readout samples per spoke (normally from ACQ_size)")
    ap.add_argument("--coils", type=int, help="Override number of receiver coils (default from PVM_EncNReceivers)")

    ap.add_argument(
        "--traj-mode",
        choices=["kron", "linz"],
        default="kron",
        help="Synthetic trajectory type (Kronecker or LinZ-GA).",
    )

    ap.add_argument(
        "--spokes-per-frame",
        type=int,
        default=None,
        help="Spokes per frame (sliding window). If omitted, uses all spokes as one frame.",
    )
    ap.add_argument(
        "--frame-shift",
        type=int,
        default=None,
        help="Spoke shift between successive frames (default = spokes-per-frame).",
    )

    ap.add_argument(
        "--test-volumes",
        type=int,
        default=None,
        help="If set, reconstruct only this many frames for quick testing.",
    )

    ap.add_argument("--fid-dtype", type=str, default=None, help="Override FID dtype (int16,int32,float32,float64)")
    ap.add_argument("--fid-endian", type=str, default=None, help="Override FID endian (little,big)")

    ap.add_argument("--combine", choices=["sos"], default="sos", help="Coil combine mode (currently only sos).")

    ap.add_argument(
        "--qa-first",
        type=int,
        default=0,
        help="If >0, write a QA NIfTI with the first N frames stacked (after recon).",
    )

    ap.add_argument("--export-nifti", action="store_true", help="Write final 4D magnitude NIfTI (.nii.gz)")
    ap.add_argument("--gpu", action="store_true", help="Try to use GPU NUFFT (requires BART compiled with CUDA)")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()
    DEBUG = args.debug
    USE_GPU = args.gpu

    series_dir: Path = args.series
    out_base: Path = args.out

    method = parse_param_file(series_dir / "method")
    acqp = parse_param_file_acqp(series_dir / "acqp")

    # matrix from PVM_Matrix, unless overridden
    if args.matrix is not None:
        NX, NY, NZ = args.matrix
    else:
        NX, NY, NZ = infer_matrix(method)
        print(f"[info] Matrix inferred from PVM_Matrix: {NX}x{NY}x{NZ}")

    # coils from PVM_EncNReceivers, unless overridden
    if args.coils is not None:
        coils = args.coils
    else:
        coils = infer_coils(method, default=1)
        print(f"[info] Coils inferred from PVM_EncNReceivers: {coils}")

    # load k-space (handles padded RO)
    ksp, true_ro, spokes_total = load_bruker_kspace(
        series_dir,
        coils=coils,
        fid_dtype=args.fid_dtype,
        fid_endian=args.fid_endian,
    )

    # readout (RO) either from user or from ACQ_size (true_ro)
    if args.readout is not None:
        RO = args.readout
        if RO > ksp.shape[0]:
            raise ValueError(f"Requested RO={RO} > loaded RO={ksp.shape[0]}")
        if RO < ksp.shape[0]:
            ksp = ksp[:RO, :, :]
    else:
        RO = true_ro
    print(f"[info] Using RO={RO} samples per spoke for recon")

    ro, sp_total, nc = ksp.shape

    # synthetic Kronecker / LinZ-GA trajectory with correct dims: 3 x RO x spokes
    traj = build_traj(args.traj_mode, ro=RO, spokes=sp_total)

    # sliding-window setup
    if args.spokes_per_frame is None:
        spf = sp_total
        shift = sp_total
        print(f"[info] No sliding-window args; using single frame with all {sp_total} spokes.")
    else:
        spf = args.spokes_per_frame
        shift = args.frame_shift if args.frame_shift is not None else spf
        print(f"[info] Sliding window: spokes-per-frame={spf}, frame-shift={shift}")

    starts = frame_starts(sp_total, spf, shift)
    if args.test_volumes is not None:
        starts = starts[: max(0, int(args.test_volumes))]
    if not starts:
        raise ValueError("No frames to reconstruct with chosen spokes-per-frame / frame-shift.")
    nframes = len(starts)
    print(f"[info] Will reconstruct {nframes} frame(s).")

    vol_bases: List[Path] = []

    # per-frame BART NUFFT + RSS
    for fi, s0 in enumerate(starts):
        s1 = s0 + spf
        if s1 > sp_total:
            print(f"[warn] Skipping partial last frame at spokes {s0}:{s1}")
            break

        # Per-frame file prefixes
        vol_prefix = out_base.with_name(f"{out_base.name}_vol{fi:05d}")
        traj_base = vol_prefix.with_name(vol_prefix.name + "_traj")
        ksp_base = vol_prefix.with_name(vol_prefix.name + "_ksp")
        coil_base = vol_prefix.with_name(vol_prefix.name + "_coil")
        img_base = vol_prefix

        # If coil image already exists, assume this frame is done
        coil_hdr = coil_base.with_suffix(".hdr")
        coil_cfl = coil_base.with_suffix(".cfl")
        if coil_hdr.exists() and coil_cfl.exists():
            print(f"[info] Frame {fi} already reconstructed -> {coil_base}, skipping NUFFT/RSS.")
            vol_bases.append(img_base)
            continue

        # Slice this frame: ksp_f (RO, spf, coils), traj_f (3, RO, spf)
        ksp_f = ksp[:, s0:s1, :]
        traj_f = traj[:, :, s0:s1]

        # BART expects:
        #   traj: 3 x RO x spokes
        #   ksp:  RO x spokes x 1 x coils
        traj_arr = traj_f.astype(np.complex64)
        ksp_arr = ksp_f.reshape(RO, spf, 1, nc, order="F").astype(np.complex64)

        write_cfl(traj_base, traj_arr, [3, RO, spf])
        write_cfl(ksp_base, ksp_arr, [RO, spf, 1, nc])

        try:
            run_bart(
                [
                    "nufft",
                    "-i",
                    "-d",
                    f"{NX}:{NY}:{NZ}",
                    "-t",
                    str(traj_base),
                    str(ksp_base),
                    str(coil_base),
                ],
                allow_gpu_fallback=True,
            )
            run_bart(["rss", "3", str(coil_base), str(img_base)])
        except subprocess.CalledProcessError as e:
            print(f"[error] BART NUFFT/RSS failed for frame {fi} with code {e.returncode}", file=sys.stderr)
            raise

        vol_bases.append(img_base)
        print(f"[info] Frame {fi}/{nframes - 1} done -> {img_base}")

    if not vol_bases:
        print("[error] No frames successfully reconstructed.", file=sys.stderr)
        sys.exit(1)

    # Optional QA: first N frames as NIfTI
    if args.qa_first and args.qa_first > 0:
        qa_n = min(args.qa_first, len(vol_bases))
        qa_base = out_base.with_name(out_base.name + f"_QA_first{qa_n}")
        qa_inputs = vol_bases[:qa_n]

        # join along dim=3 (time) -> 4D CFL
        run_bart(["join", "3"] + [str(vb) for vb in qa_inputs] + [str(qa_base)])

        # convert QA CFL -> magnitude NIfTI
        try:
            qa_arr = load_cfl(qa_base)
            mag = np.abs(qa_arr).astype(np.float32)
            affine = np.eye(4, dtype=np.float32)
            qa_nii = qa_base.with_suffix(".nii.gz")
            nib.save(nib.Nifti1Image(mag, affine), str(qa_nii))
            print(f"[info] Wrote QA NIfTI with first {qa_n} frame(s) to {qa_nii}")
        except Exception as e:
            print(f"[warn] Failed to write QA NIfTI: {e}", file=sys.stderr)

    # Build 4D volume at out_base: stack/join along dim=3
    if len(vol_bases) == 1:
        run_bart(["copy", str(vol_bases[0]), str(out_base)])
    else:
        run_bart(["join", "3"] + [str(vb) for vb in vol_bases] + [str(out_base)])

    # Export final 4D NIfTI using nibabel (magnitude)
    if args.export_nifti:
        try:
            arr4d = load_cfl(out_base)
            mag4d = np.abs(arr4d).astype(np.float32)
            affine = np.eye(4, dtype=np.float32)
            out_nii = out_base.with_suffix(".nii.gz")
            nib.save(nib.Nifti1Image(mag4d, affine), str(out_nii))
            print(f"[info] Wrote 4D NIfTI to {out_nii}")
        except Exception as e:
            print(f"[warn] Failed to write 4D NIfTI: {e}", file=sys.stderr)

    print("[info] All requested frames complete; 4D CFL base at", out_base)


if __name__ == "__main__":
    main()
