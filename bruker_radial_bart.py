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


def infer_readout(method: Dict[str, str], acqp: Dict[str, str]) -> int:
    """
    Infer *true* readout samples per spoke (RO) from headers.

    For your UTE3D case:
      ACQ_size = 122 2584 1
      -> RO = 122 (2nd entry)

    Fallbacks if needed:
      - PVM_TrajIntAll
      - NPro
    """
    # ACQ_size: e.g. "122 2584 1" where the second entry is RO
    acq_val = acqp.get("ACQ_size", "")
    acq_ints = _parse_int_list(acq_val)
    if len(acq_ints) >= 2:
        ro = acq_ints[0] if False else acq_ints[1]  # 122 in your example
        print(f"[info] Readout (true RO) inferred from ACQ_size: RO={ro}")
        return ro

    # Fallback: PVM_TrajIntAll
    val = method.get("PVM_TrajIntAll", "")
    ints = _parse_int_list(val)
    if ints:
        ro = ints[0]
        print(f"[info] Readout inferred from PVM_TrajIntAll: RO={ro}")
        return ro

    # Fallback: NPro
    val = method.get("NPro", "")
    ints = _parse_int_list(val)
    if ints:
        ro = ints[0]
        print(f"[info] Readout inferred from NPro: RO={ro}")
        return ro

    raise ValueError("Could not infer readout (RO) from ACQ_size / PVM_TrajIntAll / NPro. Please pass --readout.")


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


def infer_extras(method: Dict[str, str], acqp: Dict[str, str]) -> Dict[str, int]:
    """Infer extra dimensions: echoes, reps, averages, slices."""
    def get_int(keys, srcs, default=1):
        for k in keys:
            for s in srcs:
                if k in s:
                    ints = _parse_int_list(s[k])
                    if ints:
                        return ints[0]
        return default

    echoes = get_int(
        ["NECHOES", "ACQ_n_echo_images", "PVM_NEchoImages"],
        [method, acqp],
        default=1,
    )
    reps = get_int(["PVM_NRepetitions", "NR"], [method, acqp], default=1)
    averages = get_int(["PVM_NAverages", "NA"], [method, acqp], default=1)
    slices = get_int(["NSLICES", "PVM_SPackArrNSlices"], [method, acqp], default=1)

    extras = {
        "echoes": echoes,
        "reps": reps,
        "averages": averages,
        "slices": slices,
    }
    return extras


def load_bruker_kspace(
    series_dir: Path,
    readout: Optional[int],
    coils: Optional[int],
    fid_dtype: Optional[str] = None,
    fid_endian: Optional[str] = None,
) -> np.ndarray:
    """
    Bruker FID loader that understands padded readout blocks and extra dims.

    - Factors FID into (stored_ro, spokes_total, coils) using a BLOCKS table.
    - Uses `readout` as a *hint* for the true RO (e.g., 122).
    - Trims from stored_ro down to `readout` if stored_ro >= readout.
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

    readout_hint = readout  # e.g., 122 for your case

    dbg("fid_dtype:", fid_dtype, "fid_endian:", fid_endian, "coils:", coils, "readout_hint:", readout_hint)

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

    # complex packing
    if raw.size % 2 != 0:
        raw = raw[:-1]
    cpx = raw.astype(np.float32).view(np.complex64)
    total = cpx.size
    dbg("total complex samples:", total)

    # extras (echoes, reps, averages, slices)
    extras = infer_extras(method, acqp)
    other_factor = 1
    for key, v in extras.items():
        if isinstance(v, int) and v > 1:
            other_factor *= v
    if other_factor < 1:
        other_factor = 1
    dbg("extras:", extras, "other_factor:", other_factor)

    # First attempt: assume extras are correct
    def factor_with_extras(total, coils, other_factor, readout_hint):
        denom = coils * max(1, other_factor)
        if total % denom != 0:
            return None
        per_coil_total = total // denom

        def pick_block_and_spokes(per_coil_total: int, readout_hint: Optional[int]) -> Tuple[int, int]:
            # Common Bruker stored RO block sizes
            BLOCKS = [
                64, 96, 128, 160, 192, 200, 224, 240, 256, 288, 320, 352, 384, 400,
                416, 420, 432, 448, 480, 496, 512, 544, 560, 576, 608, 640, 672, 704,
                736, 768, 800, 832, 896, 960, 992, 1024, 1152, 1280, 1536, 2048
            ]

            # If readout_hint works exactly, use it as stored_ro
            if readout_hint and readout_hint > 0 and per_coil_total % readout_hint == 0:
                return readout_hint, per_coil_total // readout_hint

            # Otherwise, search common block sizes >= readout_hint (if present)
            for b in BLOCKS:
                if readout_hint is not None and b < readout_hint:
                    continue
                if per_coil_total % b == 0:
                    return b, per_coil_total // b

            # Last resort: generic factor close to sqrt
            s = int(round(per_coil_total ** 0.5))
            for d in range(0, s + 1):
                for cand in (s + d, s - d):
                    if cand > 0 and per_coil_total % cand == 0:
                        return cand, per_coil_total // cand

            raise ValueError("Could not factor per_coil_total into (stored_ro, spokes_per_extra).")

        stored_ro, spokes_per_extra = pick_block_and_spokes(per_coil_total, readout_hint)
        return stored_ro, spokes_per_extra, per_coil_total

    # Try factoring with extras; if that fails, relax extras and/or coils like a degenerate case
    fact = factor_with_extras(total, coils, other_factor, readout_hint)
    if fact is None:
        dbg("total not divisible by coils*other_factor; relaxing extras to 1")
        other_factor = 1
        fact = factor_with_extras(total, coils, other_factor, readout_hint)
        if fact is None:
            dbg("still not divisible; relaxing coils->1")
            coils = 1
            fact = factor_with_extras(total, coils, other_factor, readout_hint)
            if fact is None:
                raise ValueError("Cannot factor FID length with any (coils, other_factor) combo.")

    stored_ro, spokes_per_extra, per_coil_total = fact
    spokes_total = spokes_per_extra * max(1, other_factor)
    dbg("stored_ro (block):", stored_ro, "spokes_per_extra:", spokes_per_extra, "spokes_total:", spokes_total)
    if stored_ro * spokes_total * coils != total:
        raise ValueError("Internal factoring error: stored_ro*spokes_total*coils != total samples")

    # reshape to (stored_ro, spokes_total, coils)
    ksp_blk = np.reshape(cpx, (stored_ro, spokes_total, coils), order="F")

    # finally trim to the true RO if we have a hint and stored_ro >= readout_hint
    if readout_hint is not None and stored_ro >= readout_hint:
        ro = readout_hint
        ksp = ksp_blk[:ro, :, :]
        dbg(f"trimmed RO from stored_ro={stored_ro} to ro={ro}")
    else:
        ro = stored_ro
        ksp = ksp_blk

    print(f"[info] Loaded k-space: RO={ro}, Spokes={spokes_total}, Coils={coils}")
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


def run_bart(args: List[str]):
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
    ap.add_argument("--readout", type=int, help="Override true RO (default from ACQ_size / PVM_TrajIntAll / NPro)")
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
    ap.add_argument("--export-nifti", action="store_true", help="Call 'bart toimg' on final 4D volume")
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

    # readout (true RO) from ACQ_size / PVM_TrajIntAll / NPro, unless overridden
    if args.readout is not None:
        RO_hint = args.readout
        print(f"[info] Readout overridden from CLI: RO={RO_hint}")
    else:
        RO_hint = infer_readout(method, acqp)

    # coils from PVM_EncNReceivers, unless overridden
    if args.coils is not None:
        coils = args.coils
        print(f"[info] Coils overridden from CLI: {coils}")
    else:
        coils = infer_coils(method, default=1)
        print(f"[info] Coils inferred from PVM_EncNReceivers: {coils}")

    # k-space (handles padded blocks & extras, trims to RO_hint when possible)
    ksp = load_bruker_kspace(
        series_dir,
        readout=RO_hint,
        coils=coils,
        fid_dtype=args.fid_dtype,
        fid_endian=args.fid_endian,
    )
    ro, sp_total, nc = ksp.shape

    # synthetic Kronecker / LinZ-GA trajectory, matched to trimmed k-space
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
        ksp_s = ksp_f.reshape(ro * spf, nc, order="F").astype(np.complex64)
        ksp_s = ksp_s.reshape(1, nsamp, 1, nc, order="F")

        vol_prefix = out_base.with_name(f"{out_base.name}_vol{fi:05d}")
        traj_base = vol_prefix.with_name(vol_prefix.name + "_traj")
        ksp_base = vol_prefix.with_name(vol_prefix.name + "_ksp")
        coil_base = vol_prefix.with_name(vol_prefix.name + "_coil")
        img_base = vol_prefix

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
                ]
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

    # Build 4D volume at out_base: stack along dim=3
    if len(vol_bases) == 1:
        run_bart(["copy", str(vol_bases[0]), str(out_base)])
    else:
        stack_args = ["stack", "3"] + [str(vb) for vb in vol_bases] + [str(out_base)]
        run_bart(stack_args)

    if args.export_nifti:
        try:
            run_bart(["toimg", str(out_base), str(out_base)])
        except subprocess.CalledProcessError as e:
            print(f"[warn] BART toimg failed on 4D volume: {e}", file=sys.stderr)

    print("[info] All requested frames complete; 4D result at", out_base)


if __name__ == "__main__":
    main()
