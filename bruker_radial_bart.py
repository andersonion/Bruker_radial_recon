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


def infer_readout_from_method(method: Dict[str, str]) -> Optional[int]:
    """Fallback: try method params if ACQ_size isn't usable."""
    for key in ("PVM_TrajIntAll", "NPro"):
        val = method.get(key, "")
        ints = _parse_int_list(val)
        if ints:
            return ints[0]
    return None


def infer_true_ro_from_acqp(acqp: Dict[str, str]) -> Optional[int]:
    """True readout samples (non-padded) from ACQ_size[0] if available."""
    val = acqp.get("ACQ_size", "")
    ints = _parse_int_list(val)
    if len(ints) >= 1:
        return ints[0]
    return None


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


def find_stored_ro(total_complex: int, coils: int, true_ro: int) -> int:
    """Given total complex samples, coils, and true RO, infer padded stored RO.

    We search for the smallest stored_ro >= true_ro such that
    total_complex is divisible by stored_ro * coils.
    """
    denom_base = coils
    for stored_ro in range(true_ro, true_ro + 2048):
        if (total_complex % (stored_ro * denom_base)) == 0:
            return stored_ro
    # Fallback: no padding detected; just use true_ro
    return true_ro


def load_bruker_kspace(
    series_dir: Path,
    readout_true: int,
    coils: int,
    fid_dtype: Optional[str] = None,
    fid_endian: Optional[str] = None,
) -> np.ndarray:
    """Load Bruker FID as k-space array [RO, Spokes, Coils].

    Handles padded storage blocks in RO (e.g., true RO=122, stored as 128).
    """
    fid_path = series_dir / "fid"
    if not fid_path.exists():
        raise FileNotFoundError(f"No fid file in series dir: {fid_path}")

    acqp = parse_param_file_acqp(series_dir / "acqp")

    if fid_dtype is None:
        fid_dtype = infer_fid_dtype(acqp)
    if fid_endian is None:
        fid_endian = infer_fid_endian(acqp)
    if coils is None or coils <= 0:
        raise ValueError("Number of coils must be >0 in load_bruker_kspace.")

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

    # Interpret as complex
    cpx = raw.astype(np.float32).view(np.complex64)
    total_complex = cpx.size
    dbg("total complex samples:", total_complex)

    # Infer stored_ro (padded block size) from total size and coils
    stored_ro = find_stored_ro(total_complex, coils, readout_true)
    spokes = total_complex // (stored_ro * coils)
    if spokes <= 0:
        raise ValueError(
            f"Unable to factor FID into (stored_ro * spokes * coils). "
            f"total_complex={total_complex}, stored_ro={stored_ro}, coils={coils}"
        )

    dbg("stored_ro:", stored_ro, "spokes:", spokes)

    # Reshape using stored_ro, then trim back to true RO along readout
    ksp_full = np.reshape(cpx, (stored_ro, spokes, coils), order="F")
    if readout_true > stored_ro:
        raise ValueError(
            f"True RO={readout_true} > stored_ro={stored_ro}; "
            f"padding model assumption violated."
        )

    ksp = ksp_full[:readout_true, :, :]
    print(
        f"[info] Loaded k-space with stored_ro={stored_ro}, trimmed to true RO={readout_true}; "
        f"Spokes={spokes}, Coils={coils}"
    )
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

    # radius goes from -0.5..0.5 in k-space units (BART expects traj in pixel/FOV units)
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


# ---------- BART wrappers ----------


def bart_path() -> str:
    p = shutil.which("bart")
    if p is None:
        raise RuntimeError("BART executable 'bart' not found in PATH")
    return p


def run_bart(args: List[str], gpu: bool = False):
    cmd = [bart_path()]
    if gpu:
        cmd.append("-g")
    cmd += args
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
    ap.add_argument("--readout", type=int, help="Override true readout samples per spoke (default from ACQ_size[0])")
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
        help="If >0 and --export-nifti set, export first N frames as per-frame NIfTI QA volumes.",
    )
    ap.add_argument("--export-nifti", action="store_true", help="Export final 4D volume (and QA frames) as NIfTI.")

    ap.add_argument("--gpu", action="store_true", help="Use BART GPU mode (bart -g ...).")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()
    DEBUG = args.debug

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

    # true readout (RO) from ACQ_size[0], unless overridden
    if args.readout is not None:
        true_ro = args.readout
    else:
        true_ro = infer_true_ro_from_acqp(acqp)
        if true_ro is not None:
            print(f"[info] Readout (true RO) from ACQ_size: RO={true_ro}")
        else:
            fallback_ro = infer_readout_from_method(method)
            if fallback_ro is None:
                raise ValueError(
                    "Could not infer readout (RO). Tried ACQ_size[0] and PVM_TrajIntAll/NPro. "
                    "Please pass --readout."
                )
            true_ro = fallback_ro
            print(f"[info] Readout (true RO) from method (PVM_TrajIntAll/NPro): RO={true_ro}")

    # coils from PVM_EncNReceivers, unless overridden
    if args.coils is not None:
        coils = args.coils
    else:
        coils = infer_coils(method, default=1)
        print(f"[info] Coils inferred from PVM_EncNReceivers: {coils}")

    # k-space
    ksp = load_bruker_kspace(
        series_dir,
        readout_true=true_ro,
        coils=coils,
        fid_dtype=args.fid_dtype,
        fid_endian=args.fid_endian,
    )
    ro, sp_total, nc = ksp.shape

    # synthetic Kronecker / LinZ-GA trajectory
    traj = build_traj(args.traj_mode, ro=ro, spokes=sp_total)

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
        ksp_f = ksp[:, s0:s1, :]
        if ksp_f.shape[1] < spf:
            print(f"[warn] Skipping partial last frame at spokes {s0}:{s1}")
            break
        traj_f = traj[:, :, s0:s1]

        nsamp = ro * spf
        traj_s = traj_f.reshape(3, nsamp, order="F").astype(np.complex64)
        # BART expects dims: [1, nsamp, 1, nc] for ksp; [3, nsamp, 1, 1] for traj
        ksp_s = ksp_f.reshape(ro * spf, nc, order="F").astype(np.complex64)
        ksp_s = ksp_s.reshape(1, nsamp, 1, nc, order="F")

        vol_prefix = out_base.with_name(f"{out_base.name}_vol{fi:05d}")
        traj_base = vol_prefix.with_name(vol_prefix.name + "_traj")
        ksp_base = vol_prefix.with_name(vol_prefix.name + "_ksp")
        coil_base = vol_prefix.with_name(vol_prefix.name + "_coil")
        img_base = vol_prefix

        # Check if this frame already fully exists (coil + SoS)
        coil_cfl = coil_base.with_suffix(".cfl")
        img_cfl = img_base.with_suffix(".cfl")
        if coil_cfl.exists() and img_cfl.exists():
            print(
                f"[info] Frame {fi} already reconstructed -> {coil_base}, skipping NUFFT/RSS."
            )
            vol_bases.append(img_base)
            continue

        # Write frame-wise traj/ksp
        write_cfl(traj_base, traj_s, [3, nsamp, 1, 1])
        write_cfl(ksp_base, ksp_s, [1, nsamp, 1, nc])

        try:
            # NUFFT -> coil images
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
                gpu=args.gpu,
            )
            # RSS along coil dimension
            run_bart(["rss", "3", str(coil_base), str(img_base)], gpu=args.gpu)
        except subprocess.CalledProcessError as e:
            print(f"[error] BART NUFFT/RSS failed for frame {fi} with code {e.returncode}", file=sys.stderr)
            raise

        vol_bases.append(img_base)
        print(f"[info] Frame {fi}/{nframes - 1} done -> {img_base}")

        # Optional QA export: per-frame NIfTI for early frames
        if args.export_nifti and args.qa_first > 0 and fi < args.qa_first:
            qa_nifti = img_base.with_suffix(".nii")
            try:
                run_bart(["toimg", str(img_base), str(qa_nifti)], gpu=False)
            except subprocess.CalledProcessError as e:
                print(f"[warn] BART toimg failed for QA frame {fi}: {e}", file=sys.stderr)

    if not vol_bases:
        print("[error] No frames successfully reconstructed.", file=sys.stderr)
        sys.exit(1)

    # Build 4D volume at out_base: join along dim=3
    join_args = ["join", "3"] + [str(vb) for vb in vol_bases] + [str(out_base)]
    try:
        run_bart(join_args, gpu=args.gpu)
    except subprocess.CalledProcessError as e:
        print(f"[error] BART join failed: {e}", file=sys.stderr)
        raise

    # Final 4D NIfTI export
    if args.export_nifti:
        nifti_out = out_base.with_suffix(".nii")
        try:
            run_bart(["toimg", str(out_base), str(nifti_out)], gpu=False)
        except subprocess.CalledProcessError as e:
            print(f"[warn] BART toimg failed on 4D volume: {e}", file=sys.stderr)

    print("[info] All requested frames complete; 4D result at", out_base)


if __name__ == "__main__":
    main()
