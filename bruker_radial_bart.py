#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Try nibabel for NIfTI export
try:
    import nibabel as nib  # type: ignore

    HAS_NIB = True
except Exception:
    HAS_NIB = False

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
        for raw in f:
            line = raw.rstrip("\n").strip()

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


def infer_readout_from_acqp(acqp: Dict[str, str]) -> Optional[int]:
    """True RO from ACQ_size[0] if available."""
    val = acqp.get("ACQ_size", "")
    ints = _parse_int_list(val)
    if len(ints) >= 1:
        return ints[0]
    return None


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


def factor_stored_ro(total_cpx: int, true_ro: int, coils: int) -> Tuple[int, int]:
    """
    Given total complex samples, the true (unpadded) RO and coil count,
    guess the stored RO (with padding) and total spokes.

    We search over a small set of plausible stored_ro values.
    """
    if coils <= 0:
        raise ValueError("coils must be > 0 for factorization")

    # plausible stored block lengths (Bruker-ish)
    candidates = sorted(
        set(
            [
                true_ro,
                64,
                96,
                128,
                160,
                192,
                256,
                384,
                512,
                768,
                1024,
            ]
        )
    )

    for stored_ro in candidates:
        if stored_ro < true_ro:
            continue
        denom = stored_ro * coils
        if total_cpx % denom == 0:
            spokes = total_cpx // denom
            return stored_ro, spokes

    raise ValueError(
        f"Could not factor FID into (stored_ro * spokes * coils). "
        f"total={total_cpx}, true_ro={true_ro}, coils={coils}"
    )


def load_bruker_kspace(
    series_dir: Path,
    true_ro: int,
    coils: int,
    fid_dtype: Optional[str] = None,
    fid_endian: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load Bruker FID as k-space [RO, Spokes, Coils].

    Bruker stores data in padded blocks (stored_ro >= true_ro).
    We:
      1) read raw FID,
      2) factor into (stored_ro, spokes, coils),
      3) reshape using stored_ro,
      4) trim the RO dimension down to true_ro.
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

    # complex
    if raw.size % 2 != 0:
        raw = raw[:-1]
    cpx = raw.astype(np.float32).view(np.complex64)
    total = cpx.size
    dbg("total complex samples:", total)

    stored_ro, spokes = factor_stored_ro(total, true_ro, coils)
    dbg("stored_ro:", stored_ro, "spokes:", spokes)

    # reshape with stored_ro padding, then trim
    ksp_full = np.reshape(cpx, (stored_ro, spokes, coils), order="F")
    if stored_ro > true_ro:
        ksp = ksp_full[:true_ro, :, :]
        print(
            f"[info] Loaded k-space with stored_ro={stored_ro}, "
            f"trimmed to true RO={true_ro}; Spokes={spokes}, Coils={coils}"
        )
    else:
        ksp = ksp_full
        print(f"[info] Loaded k-space: RO={true_ro}, Spokes={spokes}, Coils={coils}")

    return ksp.astype(np.complex64), spokes


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

    # radius -0.5..0.5 in k-space units
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


def read_cfl(base: Path) -> Tuple[np.ndarray, List[int]]:
    """Minimal CFL reader: returns complex64 array and dims."""
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")

    with open(hdr, "r") as f:
        lines = f.readlines()
    if len(lines) < 2:
        raise ValueError(f"Bad CFL header: {hdr}")
    dims = [int(x) for x in lines[1].split()]
    dims = [d for d in dims if d > 0]
    nprod = int(np.prod(dims))

    raw = np.fromfile(cfl, dtype=np.float32)
    if raw.size != 2 * nprod:
        raise ValueError(f"Unexpected CFL size: expected {2*nprod}, got {raw.size}")
    raw = raw.reshape(2, nprod, order="F")
    cpx = raw[0, :] + 1j * raw[1, :]
    arr = cpx.reshape(dims, order="F").astype(np.complex64)
    return arr, dims


# ---------- BART wrappers ----------

GPU_NUFFT = False


def bart_path() -> str:
    p = shutil.which("bart")
    if p is None:
        raise RuntimeError("BART executable 'bart' not found in PATH")
    return p


def run_bart(args: List[str]):
    cmd = [bart_path()] + args
    print("[bart]", " ".join(str(a) for a in cmd))
    subprocess.run(cmd, check=True)


def run_bart_nufft(NX: int, NY: int, NZ: int, traj_base: Path, ksp_base: Path, coil_base: Path):
    """
    Run BART NUFFT, optionally trying GPU. If GPU is requested but BART
    is compiled without GPU support, fall back to CPU and disable GPU_NUFFT.
    """
    global GPU_NUFFT

    base_args = [
        "nufft",
        "-i",
        "-d",
        f"{NX}:{NY}:{NZ}",
        "-t",
        str(traj_base),
        str(ksp_base),
        str(coil_base),
    ]

    if GPU_NUFFT:
        gpu_args = ["nufft", "-i", "-g", "-d", f"{NX}:{NY}:{NZ}", "-t", str(traj_base), str(ksp_base), str(coil_base)]
        cmd = [bart_path()] + gpu_args
        print("[bart]", " ".join(str(a) for a in cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            stdout = e.stdout or ""
            if "compiled without GPU support" in stderr or "compiled without GPU support" in stdout:
                print(
                    "[warn] BART compiled without GPU support; "
                    "falling back to CPU NUFFT and disabling --gpu."
                )
                GPU_NUFFT = False
            else:
                # Some other NUFFT error; re-raise
                print(stderr, file=sys.stderr)
                raise

    # CPU path
    cmd = [bart_path()] + base_args
    print("[bart]", " ".join(str(a) for a in cmd))
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


# ---------- NIfTI helpers ----------

def export_nifti_from_cfl(base: Path):
    """Export magnitude NIfTI (.nii.gz) from a BART CFL volume."""
    if not HAS_NIB:
        print("[warn] nibabel not available; skipping NIfTI export for", base)
        return

    arr_cpx, dims = read_cfl(base)
    # magnitude, drop trailing singleton dims
    mag = np.abs(arr_cpx).astype(np.float32)
    while mag.ndim > 0 and mag.shape[-1] == 1:
        mag = mag[..., 0]

    affine = np.eye(4, dtype=np.float32)
    img = nib.Nifti1Image(mag, affine)

    out_nii = base.with_suffix(".nii.gz")
    nib.save(img, str(out_nii))
    print("[info] Wrote NIfTI:", out_nii)


# ---------- Main recon driver ----------

def main():
    global DEBUG, GPU_NUFFT

    ap = argparse.ArgumentParser(
        description="Bruker 3D radial recon using BART NUFFT (Kronecker/LinZ-GA + sliding window)."
    )
    ap.add_argument("--series", type=Path, required=True, help="Bruker series dir (contains fid, method, acqp)")
    ap.add_argument("--out", type=Path, required=True, help="Output prefix (basename only, not extension)")

    ap.add_argument("--matrix", type=int, nargs=3, help="Override matrix size NX NY NZ (default from PVM_Matrix)")
    ap.add_argument("--readout", type=int, help="Override readout samples per spoke (default from ACQ_size[0])")
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
        default=None,
        help="After these many frames are reconstructed, write a QA 4D NIfTI of the first frames.",
    )
    ap.add_argument(
        "--export-nifti",
        action="store_true",
        help="Export final 4D result (and QA, if requested) as NIfTI using nibabel (magnitude).",
    )
    ap.add_argument(
        "--gpu",
        action="store_true",
        help="Use BART NUFFT GPU flag (-g) if compiled with GPU support.",
    )
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()
    DEBUG = args.debug
    GPU_NUFFT = args.gpu

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

    # readout (true RO) from ACQ_size[0], unless overridden
    if args.readout is not None:
        RO = args.readout
    else:
        ro_from_acqp = infer_readout_from_acqp(acqp)
        if ro_from_acqp is None:
            raise ValueError("Could not infer readout (RO) from ACQ_size in acqp; please pass --readout.")
        RO = ro_from_acqp
        print(f"[info] Readout (true RO) from ACQ_size: RO={RO}")

    # coils from PVM_EncNReceivers, unless overridden
    if args.coils is not None:
        coils = args.coils
    else:
        coils = infer_coils(method, default=1)
        print(f"[info] Coils inferred from PVM_EncNReceivers: {coils}")

    # k-space
    ksp, spokes_total = load_bruker_kspace(
        series_dir,
        true_ro=RO,
        coils=coils,
        fid_dtype=args.fid_dtype,
        fid_endian=args.fid_endian,
    )
    ro, sp_total, nc = ksp.shape
    assert sp_total == spokes_total

    # synthetic Kronecker / LinZ-GA trajectory
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
        ksp_f = ksp[:, s0:s1, :]
        if ksp_f.shape[1] < spf:
            print(f"[warn] Skipping partial last frame at spokes {s0}:{s1}")
            break
        traj_f = traj[:, :, s0:s1]

        nsamp = RO * spf
        traj_s = traj_f.reshape(3, nsamp, order="F").astype(np.complex64)
        ksp_s = ksp_f.reshape(RO * spf, nc, order="F").astype(np.complex64)
        ksp_s = ksp_s.reshape(1, nsamp, 1, nc, order="F")

        vol_prefix = out_base.with_name(f"{out_base.name}_vol{fi:05d}")
        traj_base = vol_prefix.with_name(vol_prefix.name + "_traj")
        ksp_base = vol_prefix.with_name(vol_prefix.name + "_ksp")
        coil_base = vol_prefix.with_name(vol_prefix.name + "_coil")
        img_base = vol_prefix

        # If coil image already exists, skip NUFFT/RSS
        coil_cfl = coil_base.with_suffix(".cfl")
        if coil_cfl.exists():
            print(f"[info] Frame {fi} already reconstructed -> {coil_base}, skipping NUFFT/RSS.")
            vol_bases.append(img_base)
        else:
            write_cfl(traj_base, traj_s, [3, nsamp, 1, 1])
            write_cfl(ksp_base, ksp_s, [1, nsamp, 1, nc])

            try:
                run_bart_nufft(NX, NY, NZ, traj_base, ksp_base, coil_base)
                run_bart(["rss", "3", str(coil_base), str(img_base)])
            except subprocess.CalledProcessError as e:
                print(f"[error] BART NUFFT/RSS failed for frame {fi} with code {e.returncode}", file=sys.stderr)
                raise

            vol_bases.append(img_base)
            print(f"[info] Frame {fi}/{nframes - 1} done -> {img_base}")

        # QA export once we’ve completed qa_first frames
        if args.qa_first is not None and (fi + 1) == args.qa_first:
            qa_out = out_base.with_name(f"{out_base.name}_QA_first{args.qa_first}")
            try:
                run_bart(["join", "3"] + [str(vb) for vb in vol_bases[: args.qa_first]] + [str(qa_out)])
                if args.export_nifti:
                    export_nifti_from_cfl(qa_out)
                print(f"[info] QA 4D volume written at {qa_out}")
            except subprocess.CalledProcessError as e:
                print(f"[warn] BART join failed for QA volume: {e}", file=sys.stderr)

    if not vol_bases:
        print("[error] No frames successfully reconstructed.", file=sys.stderr)
        sys.exit(1)

    # Build 4D volume at out_base: stack / join along dim=3
    try:
        run_bart(["join", "3"] + [str(vb) for vb in vol_bases] + [str(out_base)])
    except subprocess.CalledProcessError as e:
        print(f"[error] BART join failed on final 4D volume: {e}", file=sys.stderr)
        raise

    if args.export_nifti:
        export_nifti_from_cfl(out_base)

    print("[info] All requested frames complete; 4D result at", out_base)


if __name__ == "__main__":
    main()
