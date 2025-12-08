#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None

DEBUG = False


def dbg(*a):
    if DEBUG:
        print("[debug]", *a)


# ---------- Bruker helpers ----------

def parse_param_file(path: Path) -> Dict[str, str]:
    """Parse Bruker-style parameter file into KEY -> value string."""
    d: Dict[str, str] = {}
    if not path.exists():
        return d

    current_key: Optional[str] = None
    collecting_array = False

    with open(path, "r") as f:
        for line in f:
            raw = line.rstrip("\n")
            line = raw.strip()

            if line.startswith("##$"):
                collecting_array = False
                current_key = None

                body = line[3:]  # drop "##$"
                if "=" not in body:
                    continue
                key, val = body.split("=", 1)
                key = key.strip()
                val = val.strip()

                # Array-style, values on subsequent lines
                if val.startswith("("):
                    d[key] = ""
                    current_key = key
                    collecting_array = True
                else:
                    d[key] = val
                continue

            if line.startswith("##") or line.startswith("$$"):
                collecting_array = False
                current_key = None
                continue

            if collecting_array and current_key is not None and line:
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


def infer_true_ro_and_block(acqp: Dict[str, str]) -> Tuple[int, Optional[int]]:
    """Infer true readout (RO) and possible stored block length.

    ACQ_size is usually (indirectDim, RO, somethingElse).
    For 3D radial UTE, ACQ_size[1] is the block length on disk.
    True RO is often the number of nonzero samples before zero padding
    and may be encoded elsewhere; in your case it's 122, with blocks of 128.
    For now: infer true RO as ACQ_size[0] when ACQ_size has length>=2 and
    the second entry is an integer multiple of the first.
    """
    val = acqp.get("ACQ_size", "")
    ints = _parse_int_list(val)
    if len(ints) >= 2:
        cand_ro = ints[0]
        block = ints[1]
        if block % cand_ro == 0:
            return cand_ro, block
    # Fallback: just use first entry as RO, no block info
    if ints:
        return ints[0], None
    raise ValueError("Could not infer RO from ACQ_size; please pass --readout.")


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


def load_bruker_kspace(
    series_dir: Path,
    true_ro: int,
    coils: int,
    fid_dtype: Optional[str] = None,
    fid_endian: Optional[str] = None,
) -> np.ndarray:
    """Load Bruker FID as k-space [RO, spokes, coils], trimming padded blocks.

    Handles the 'stored in blocks of 128, true RO=122' situation by treating
    stored_ro as the next multiple of true_ro and trimming the last 6 samples
    on each spoke.
    """
    fid_path = series_dir / "fid"
    if not fid_path.exists():
        raise FileNotFoundError(f"No fid file in series dir: {fid_path}")

    acqp = parse_param_file_acqp(series_dir / "acqp")

    if fid_dtype is None:
        fid_dtype = infer_fid_dtype(acqp)
    if fid_endian is None:
        fid_endian = infer_fid_endian(acqp)

    dbg("fid_dtype:", fid_dtype, "fid_endian:", fid_endian, "coils:", coils)

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

    # Guess stored_ro as smallest multiple of true_ro that divides total/(coils * spokes)
    # but we don't know spokes yet. We know Bruker likes power-of-two-ish chunk sizes.
    # For your data, stored_ro = 128, true_ro = 122.
    # We'll try common block sizes and pick the one that yields integer spokes.
    candidate_blocks = [true_ro]
    for blk in (64, 96, 112, 120, 128, 160, 192, 256, 384, 512):
        if blk % true_ro == 0:
            candidate_blocks.append(blk)
    candidate_blocks = sorted(set(candidate_blocks))

    spokes = None
    stored_ro = None
    denom_base = coils
    for blk in candidate_blocks:
        if total % (blk * denom_base) == 0:
            spokes_candidate = total // (blk * denom_base)
            if spokes_candidate > 0:
                stored_ro = blk
                spokes = spokes_candidate
                break

    if stored_ro is None or spokes is None:
        raise ValueError(
            f"Could not factor FID into (stored_ro * spokes * coils). "
            f"total={total}, true_ro={true_ro}, coils={coils}"
        )

    dbg("stored_ro:", stored_ro, "true_ro:", true_ro, "spokes:", spokes)

    # Reshape to [stored_ro, spokes, coils] then trim to [true_ro, spokes, coils]
    ksp_full = np.reshape(cpx, (stored_ro, spokes, coils), order="F")
    if stored_ro != true_ro:
        print(
            f"[info] Loaded k-space with stored_ro={stored_ro}, trimmed to true RO={true_ro}; "
            f"Spokes={spokes}, Coils={coils}"
        )
        ksp = ksp_full[:true_ro, :, :]
    else:
        print(f"[info] Loaded k-space: RO={true_ro}, Spokes={spokes}, Coils={coils}")
        ksp = ksp_full

    return ksp


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

    # radius -0.5..0.5 in k-space units (pixel_size / FOV)
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


def read_cfl(base: Path) -> np.ndarray:
    """Read a BART .cfl/.hdr pair into a complex64 numpy array."""
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")
    if not hdr.exists() or not cfl.exists():
        raise FileNotFoundError(f"Missing CFL pair for base {base}")
    with open(hdr, "r") as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError(f"Bad hdr file: {hdr}")
    dims = [int(x) for x in lines[1].split()]
    dims = [d for d in dims if d > 0]
    data = np.fromfile(cfl, dtype=np.float32)
    if data.size % 2 != 0:
        raise ValueError(f"CFL data length not even for {cfl}")
    data = data.reshape(2, -1, order="F")
    cpx = data[0] + 1j * data[1]
    arr = cpx.reshape(dims, order="F").astype(np.complex64)
    return arr


def export_nifti_from_cfl(base: Path, out_nii: Path):
    if nib is None:
        print(
            f"[warn] nibabel not available; skipping NIfTI export for {base}",
            file=sys.stderr,
        )
        return
    vol_c = read_cfl(base)
    # magnitude; for coil-combined images this is already real
    mag = np.abs(vol_c).astype(np.float32)
    # Dummy affine (identity) – user can reheader later
    aff = np.eye(4, dtype=np.float32)
    img = nib.Nifti1Image(mag, aff)
    out_nii_str = str(out_nii)
    if not out_nii_str.endswith(".nii") and not out_nii_str.endswith(".nii.gz"):
        out_nii_str += ".nii.gz"
    nib.save(img, out_nii_str)
    print(f"[info] Wrote NIfTI: {out_nii_str}")


# ---------- BART wrappers ----------

def bart_path() -> str:
    p = shutil.which("bart")
    if p is None:
        raise RuntimeError("BART executable 'bart' not found in PATH")
    return p


def run_bart(args: List[str], use_gpu: bool = False):
    """Run BART command.

    NOTE: Your BART build does NOT support a '-g' GPU flag. If BART was
    compiled with CUDA, it will use the GPU internally; otherwise this
    flag is effectively a no-op here (we just change the log tag).
    """
    cmd = [bart_path()] + args
    tag = "bart-gpu" if use_gpu else "bart"
    print(f"[{tag}]", " ".join(cmd))
    subprocess.run(cmd, check=True)


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
    global DEBUG

    ap = argparse.ArgumentParser(
        description="Bruker 3D radial recon using BART NUFFT (Kronecker/LinZ-GA + sliding window)."
    )
    ap.add_argument("--series", type=Path, required=True, help="Bruker series dir (contains fid, method, acqp)")
    ap.add_argument("--out", type=Path, required=True, help="Output prefix (basename only, not extension)")

    ap.add_argument("--matrix", type=int, nargs=3, help="Override matrix size NX NY NZ (default from PVM_Matrix)")
    ap.add_argument("--readout", type=int, help="Override true RO samples per spoke (default from ACQ_size)")
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
        help="If >0 and --export-nifti, also write standalone NIfTI for the first N frames.",
    )
    ap.add_argument("--export-nifti", action="store_true", help="Export 4D NIfTI (and QA volumes if requested).")
    ap.add_argument("--gpu", action="store_true", help="Hint: use GPU-capable BART (no-op if unsupported).")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()
    DEBUG = args.debug

    series_dir: Path = args.series
    out_base: Path = args.out

    method = parse_param_file(series_dir / "method")
    acqp = parse_param_file_acqp(series_dir / "acqp")

    # Matrix
    if args.matrix is not None:
        NX, NY, NZ = args.matrix
    else:
        NX, NY, NZ = infer_matrix(method)
        print(f"[info] Matrix inferred from PVM_Matrix: {NX}x{NY}x{NZ}")

    # True RO
    if args.readout is not None:
        true_ro = args.readout
        print(f"[info] Readout (true RO) overridden on CLI: RO={true_ro}")
    else:
        true_ro, block = infer_true_ro_and_block(acqp)
        if block is not None and block != true_ro:
            print(f"[info] Readout (true RO) from ACQ_size: RO={true_ro}, stored block = {block}")
        else:
            print(f"[info] Readout (true RO) from ACQ_size: RO={true_ro}")

    # Coils
    if args.coils is not None:
        coils = args.coils
    else:
        coils = infer_coils(method, default=1)
        print(f"[info] Coils inferred from PVM_EncNReceivers: {coils}")

    # k-space
    ksp = load_bruker_kspace(
        series_dir,
        true_ro=true_ro,
        coils=coils,
        fid_dtype=args.fid_dtype,
        fid_endian=args.fid_endian,
    )
    ro, sp_total, nc = ksp.shape

    # Traj
    traj = build_traj(args.traj_mode, ro=ro, spokes=sp_total)

    # Sliding-window
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

    qa_first = max(0, int(args.qa_first or 0))

    vol_bases: List[Path] = []

    # per-frame BART NUFFT + RSS
    for fi, s0 in enumerate(starts):
        s1 = s0 + spf
        ksp_f = ksp[:, s0:s1, :]
        if ksp_f.shape[1] < spf:
            print(f"[warn] Skipping partial last frame at spokes {s0}:{s1}")
            break
        traj_f = traj[:, :, s0:s1]

        nsamp = ro * spf
        traj_s = traj_f.reshape(3, nsamp, order="F").astype(np.complex64)
        ksp_s = ksp_f.reshape(ro * spf, nc, order="F").astype(np.complex64)
        ksp_s = ksp_s.reshape(1, nsamp, 1, nc, order="F")

        vol_prefix = out_base.with_name(f"{out_base.name}_vol{fi:05d}")
        traj_base = vol_prefix.with_name(vol_prefix.name + "_traj")
        ksp_base = vol_prefix.with_name(vol_prefix.name + "_ksp")
        coil_base = vol_prefix.with_name(vol_prefix.name + "_coil")
        img_base = vol_prefix

        coil_cfl = coil_base.with_suffix(".cfl")
        img_cfl = img_base.with_suffix(".cfl")
        if coil_cfl.exists() and img_cfl.exists():
            print(f"[info] Frame {fi} already reconstructed -> {img_base}, skipping NUFFT/RSS.")
            vol_bases.append(img_base)
            continue

        write_cfl(traj_base, traj_s, [3, nsamp, 1, 1])
        write_cfl(ksp_base, ksp_s, [1, nsamp, 1, nc])

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
                use_gpu=args.gpu,
            )
            run_bart(["rss", "3", str(coil_base), str(img_base)], use_gpu=args.gpu)
        except subprocess.CalledProcessError as e:
            print(f"[error] BART NUFFT/RSS failed for frame {fi} with code {e.returncode}", file=sys.stderr)
            raise

        vol_bases.append(img_base)
        print(f"[info] Frame {fi}/{nframes - 1} done -> {img_base}")

        if qa_first and args.export_nifti and fi < qa_first:
            qa_nii = img_base.with_suffix(".nii.gz")
            export_nifti_from_cfl(img_base, qa_nii)

    if not vol_bases:
        print("[error] No frames successfully reconstructed.", file=sys.stderr)
        sys.exit(1)

    # 4D volume: stack along dim=3 using 'join'
    if len(vol_bases) == 1:
        run_bart(["copy", str(vol_bases[0]), str(out_base)], use_gpu=args.gpu)
    else:
        join_args = ["join", "3"] + [str(vb) for vb in vol_bases] + [str(out_base)]
        run_bart(join_args, use_gpu=args.gpu)

    if args.export_nifti:
        out_nii = out_base.with_suffix(".nii.gz")
        export_nifti_from_cfl(out_base, out_nii)

    print("[info] All requested frames complete; 4D result at", out_base)


if __name__ == "__main__":
    main()
