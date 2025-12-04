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

DEBUG = False


def dbg(*a):
    if DEBUG:
        print("[debug]", *a)


# ---------- Bruker helpers ----------

def parse_param_file(path: Path) -> Dict[str, str]:
    """Very simple Bruker param parser: maps 'KEY' -> 'value string'."""
    d: Dict[str, str] = {}
    if not path.exists():
        return d
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("##$"):
                try:
                    key, val = line[3:].split("=", 1)
                    d[key.strip()] = val.strip()
                except ValueError:
                    continue
    return d


def _parse_int_list(val: str) -> List[int]:
    out: List[int] = []
    if not val:
        return out
    # strip parentheses and commas etc
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
    # fallback: older heuristic
    cleaned = val.replace("(", " ").replace(")", " ")
    for tok in cleaned.split():
        if tok.isdigit():
            return int(tok)
    return default


def infer_readout(method: Dict[str, str]) -> int:
    # Prefer PVM_TrajIntAll, then NPro
    for key in ("PVM_TrajIntAll", "NPro"):
        val = method.get(key, "")
        ints = _parse_int_list(val)
        if ints:
            return ints[0]
    raise ValueError("Could not infer readout (RO) from method (PVM_TrajIntAll or NPro). Please pass --readout.")


def parse_param_file_acqp(path: Path) -> Dict[str, str]:
    # same as parse_param_file, but separate name for clarity
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


def load_bruker_kspace(
    series_dir: Path,
    readout: int,
    coils: int,
    fid_dtype: Optional[str] = None,
    fid_endian: Optional[str] = None,
) -> np.ndarray:
    """
    Load Bruker FID and reshape to (RO, Spokes, Coils).

    This version assumes a single echo / slice / rep collapsed series and
    relies on correct readout and coils.
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

    denom = readout * coils
    if total % denom != 0:
        raise ValueError(
            f"FID length {total} not divisible by readout*coils={denom}. "
            "This simple loader assumes a single echo/slice/rep 3D radial series. "
            "Adjust --readout/--coils or extend loader to factor other dims."
        )

    spokes = total // denom
    dbg("computed spokes:", spokes)

    ksp = np.reshape(cpx, (readout, spokes, coils), order="F")
    print(f"[info] Loaded k-space: RO={readout}, Spokes={spokes}, Coils={coils}")
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


def uv_to_dir(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    """
    Build synthetic 3D radial trajectory of shape (3, RO, Spokes),
    coordinates roughly in [-0.5, 0.5] along each axis.

    mode: 'kron' or 'linz'
    """
    mode = mode.lower()
    if mode not in ("kron", "linz"):
        raise ValueError("traj-mode must be 'kron' or 'linz'")

    if mode == "kron":
        dx, dy, dz = kronecker_dirs(spokes)
    else:
        dx, dy, dz = linz_ga_dirs(spokes)

    # sample along readout from -0.5..0.5
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
    """
    Write BART-compliant CFL/HDR for a complex64 ndarray.

    dims: BART dimensions list (length <= 16; we pad with 1s).
    """
    arr = np.asarray(arr, dtype=np.complex64, order="F")
    # pad dims to 16
    if len(dims) < 16:
        dims = dims + [1] * (16 - len(dims))
    # sanity check
    if int(np.prod(dims)) != arr.size:
        raise ValueError(f"dims product {np.prod(dims)} != array size {arr.size}")
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")
    _write_hdr(hdr, dims)
    # BART stores complex as [2, prod(dims)] float32
    flat = arr.flatten(order="F")
    out = np.empty((2, flat.size), dtype=np.float32)
    out[0, :] = flat.real.astype(np.float32)
    out[1, :] = flat.imag.astype(np.float32)
    out.tofile(cfl)


# ---------- BART wrappers ----------

def bart_path() -> str:
    p = shutil.which("bart")
    if p is None:
        raise RuntimeError("BART executable 'bart' not found in PATH")
    return p


def run_bart(args: List[str]):
    """
    Run BART with the given arguments list (tool + args).
    """
    cmd = [bart_path()] + args
    print("[bart]", " ".join(cmd))
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
    ap.add_argument("--readout", type=int, help="Override readout samples per spoke (default from PVM_TrajIntAll/NPro)")
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
    ap.add_argument(
        "--qc-frame",
        type=int,
        default=0,
        help="Frame index to treat as QC for NIfTI export (only this frame gets toimg).",
    )

    ap.add_argument("--fid-dtype", type=str, default=None, help="Override FID dtype (int16,int32,float32,float64)")
    ap.add_argument("--fid-endian", type=str, default=None, help="Override FID endian (little,big)")

    ap.add_argument("--combine", choices=["sos"], default="sos", help="Coil combine mode (currently only sos).")
    ap.add_argument("--export-nifti", action="store_true", help="Call 'bart toimg' on QC frame only")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()
    DEBUG = args.debug

    series_dir: Path = args.series
    out_base: Path = args.out

    # Parse method/acqp once
    method = parse_param_file(series_dir / "method")
    acqp = parse_param_file_acqp(series_dir / "acqp")

    # Infer matrix if not overridden
    if args.matrix is not None:
        NX, NY, NZ = args.matrix
    else:
        NX, NY, NZ = infer_matrix(method)
        print(f"[info] Matrix inferred from PVM_Matrix: {NX}x{NY}x{NZ}")

    # Infer readout if not overridden
    if args.readout is not None:
        RO = args.readout
    else:
        RO = infer_readout(method)
        print(f"[info] Readout inferred from PVM_TrajIntAll/NPro: RO={RO}")

    # Infer coils if not overridden
    if args.coils is not None:
        coils = args.coils
    else:
        coils = infer_coils(method, default=1)
        print(f"[info] Coils inferred from PVM_EncNReceivers: {coils}")

    # Load k-space: (RO, Spokes, Coils)
    ksp = load_bruker_kspace(
        series_dir,
        readout=RO,
        coils=coils,
        fid_dtype=args.fid_dtype,
        fid_endian=args.fid_endian,
    )
    ro, sp_total, nc = ksp.shape
    if ro != RO:
        print(f"[warn] Loaded RO={ro} differs from requested/inferred RO={RO}; using RO={ro} from data.")
        RO = ro

    # Trajectory: (3, RO, Spokes), synthetic GA/Kronecker
    traj = build_traj(args.traj_mode, ro=RO, spokes=sp_total)

    # Frame binning
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

    # Loop over frames
    for fi, s0 in enumerate(starts):
        s1 = s0 + spf
        ksp_f = ksp[:, s0:s1, :]  # (RO, spf, coils)
        if ksp_f.shape[1] < spf:
            print(f"[warn] Skipping partial last frame at spokes {s0}:{s1}")
            break
        traj_f = traj[:, :, s0:s1]  # (3, RO, spf)

        # Flatten RO*spf -> samples dimension for BART
        nsamp = RO * spf
        # traj dims: [3, nsamp, 1, 1, 1, ...]
        traj_s = traj_f.reshape(3, nsamp, order="F").astype(np.complex64)
        # ksp dims: [1, nsamp, 1, coils, 1, ...], coil dim=3
        ksp_s = ksp_f.reshape(RO * spf, nc, order="F").astype(np.complex64)
        ksp_s = ksp_s.reshape(1, nsamp, 1, nc, order="F")

        vol_prefix = out_base.with_name(f"{out_base.name}_vol{fi:05d}")
        traj_base = vol_prefix.with_name(vol_prefix.name + "_traj")
        ksp_base = vol_prefix.with_name(vol_prefix.name + "_ksp")
        coil_base = vol_prefix.with_name(vol_prefix.name + "_coil")
        img_base = vol_prefix

        # Write traj and ksp to CFL
        write_cfl(traj_base, traj_s, [3, nsamp, 1, 1])
        write_cfl(ksp_base, ksp_s, [1, nsamp, 1, nc])

        # BART NUFFT (adjoint) and RSS (coil combine)
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
                ]
            )
            # coil dimension is 3 (0-based indexing: x,y,z,coil)
            run_bart(["rss", "3", str(coil_base), str(img_base)])
        except subprocess.CalledProcessError as e:
            print(f"[error] BART NUFFT/RSS failed for frame {fi} with code {e.returncode}", file=sys.stderr)
            raise

        # Export only QC frame as NIfTI if requested
        if args.export_nifti and fi == args.qc_frame:
            try:
                run_bart(["toimg", str(img_base), str(img_base)])
            except subprocess.CalledProcessError as e:
                print(f"[warn] BART toimg failed for QC frame {fi}: {e}", file=sys.stderr)

        qc_tag = " (QC frame)" if fi == args.qc_frame else ""
        print(f"[info] Frame {fi}/{nframes - 1} done -> {img_base}{qc_tag}")

    print("[info] All requested frames complete.")


if __name__ == "__main__":
    main()
