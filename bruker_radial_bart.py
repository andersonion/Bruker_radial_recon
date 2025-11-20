#!/usr/bin/env python3
"""
bruker_radial_bart.py

Bruker 3D radial prep script for BART.

This version focuses on the pieces that are hard to do in BART:
- Read Bruker FID and infer (RO, Spokes, Coils), trimming padded readout.
- Build synthetic 3D radial trajectory using your GA/Kronecker math
  (Kronecker or linear-z golden-angle).
- Write BART CFL/HDR pairs for:
    * full k-space:  ksp_all.[cfl,hdr]  (RO, Spokes, Coils)
    * full traj:     traj_all.[cfl,hdr] (3,  RO, Spokes)
- Optionally write a sliding-window frame index table for later use.

This script intentionally does NOT call BART's NUFFT. Use the outputs
with `bart nufft` from the shell, where you can inspect dims and tune
the recon commands.

Example:

  python bruker_radial_bart.py \\
    --series "$path" \\
    --matrix 256 256 256 \\
    --traj-mode kron \\
    --spokes-per-frame 800 \\
    --frame-shift 100 \\
    --test-volumes 5 \\
    --out "${out%.nii.gz}_SoS" \\
    --debug

Then, from shell:

  # Example: reconstruct first frame once dims are understood
  bart extract 1 0 799  ${out}_SoS_ksp_all  ${out}_SoS_ksp_f0
  bart extract 2 0 799  ${out}_SoS_traj_all ${out}_SoS_traj_f0
  bart nufft -i -d 256:256:256 -t ${out}_SoS_traj_f0 ${out}_SoS_ksp_f0 ${out}_SoS_coil_f0
  bart rss 8 ${out}_SoS_coil_f0 ${out}_SoS_img_f0
  bart toimg ${out}_SoS_img_f0 ${out}_SoS_img_f0
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# -------------- debug helper --------------

def dbg(enabled: bool, *a):
    if enabled:
        print("[debug]", *a)


# -------------- BART CFL I/O (clean) --------------

def _write_hdr(path: Path, dims: List[int]) -> None:
    """Write a minimal BART .hdr file with the given dimensions."""
    with open(path, "w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in dims) + "\n")


def write_cfl(base: Path, arr: np.ndarray, dims: Optional[List[int]] = None) -> None:
    """
    Write BART CFL/HDR pair.

    - base: path without extension
    - arr:  numpy array (complex or real)
    - dims: optional list of dimension sizes. If provided, product(dims) must
            equal arr.size. If omitted, we use [arr.size] and flatten.
    """
    base = Path(base)
    cfl = base.with_suffix(".cfl")
    hdr = base.with_suffix(".hdr")

    a = np.asarray(arr)
    if dims is None:
        dims = [int(a.size)]
        a_flat = a.ravel()
    else:
        total = 1
        for d in dims:
            total *= int(d)
        if total != a.size:
            raise ValueError(f"dims product {total} != array size {a.size}")
        a_flat = a.reshape(-1)

    _write_hdr(hdr, dims)

    # BART CFL stores complex64 as interleaved float32 [re0, im0, re1, im1, ...]
    if not np.iscomplexobj(a_flat):
        a_flat = a_flat.astype(np.complex64)
    else:
        a_flat = a_flat.astype(np.complex64)

    a_view = a_flat.view(np.float32)
    a_view.tofile(cfl)


# -------------- Bruker header helpers --------------

def _read_text_kv(path: Path) -> Dict[str, str]:
    d: Dict[str, str] = {}
    if not path.exists():
        return d
    key = None
    val_lines: List[str] = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("##$"):
                if key is not None:
                    d[key] = " ".join(val_lines).strip()
                key = line[3:].split("=", 1)[0]
                val_lines = [line.split("=", 1)[1].strip()] if "=" in line else []
            elif line.startswith("$$"):
                continue
            else:
                if key is not None:
                    val_lines.append(line.strip())
    if key is not None and key not in d:
        d[key] = " ".join(val_lines).strip()
    return d


def _get_int_from_headers(keys: List[str], dicts: List[Dict[str, str]]) -> Optional[int]:
    for k in keys:
        for d in dicts:
            if k in d:
                txt = d[k].strip().split()[0]
                try:
                    return int(txt)
                except Exception:
                    continue
    return None


def _parse_acq_size(method: Dict[str, str],
                    acqp: Dict[str, str]) -> Optional[Tuple[int, int, int]]:
    for k in ("PVM_EncMatrix", "PVM_Matrix"):
        if k in method:
            txt = method[k].strip().strip("(){}")
            parts = [p for p in txt.replace(",", " ").split() if p]
            if len(parts) >= 3:
                try:
                    return int(parts[0]), int(parts[1]), int(parts[2])
                except Exception:
                    pass
    return None


# -------------- FID → (RO, Spokes, Coils) loader --------------

def load_bruker_kspace(
    series_dir: Path,
    matrix_ro_hint: Optional[int] = None,
    spokes_hint: Optional[int] = None,
    readout: Optional[int] = None,
    coils: Optional[int] = None,
    fid_dtype: str = "int32",
    fid_endian: str = "little",
    debug: bool = False,
) -> np.ndarray:
    """
    Load Bruker FID and reshape to (RO, Spokes, Coils), trimming padded readout.

    We factor the total complex samples into stored_ro * spokes * coils * other_dims.
    When possible we use header hints to constrain RO and spokes.
    """
    series_dir = Path(series_dir)
    fid_path = series_dir / "fid"
    if not fid_path.exists():
        raise FileNotFoundError(f"No fid at {fid_path}")

    method = _read_text_kv(series_dir / "method")
    acqp = _read_text_kv(series_dir / "acqp")

    # Infer endian/word size if possible
    if "BYTORDA" in acqp and "big" in acqp["BYTORDA"].lower():
        fid_endian = "big"
    if "ACQ_word_size" in acqp:
        if "16" in acqp["ACQ_word_size"]:
            fid_dtype = "int16"
        elif "32" in acqp["ACQ_word_size"]:
            fid_dtype = "int32"

    dtype_map = {
        "int16": np.int16,
        "int32": np.int32,
        "float32": np.float32,
        "float64": np.float64,
    }
    if fid_dtype not in dtype_map:
        raise ValueError("fid_dtype must be one of int16,int32,float32,float64")
    dt = dtype_map[fid_dtype]

    raw = np.fromfile(fid_path, dtype=dt)
    if fid_endian == "big":
        raw = raw.byteswap().newbyteorder()
    if raw.size % 2 != 0:
        raw = raw[:-1]
    cpx = raw.astype(np.float32).view(np.complex64)
    total = cpx.size
    dbg(debug, "total complex samples:", total,
        "(dtype:", fid_dtype, ", endian:", fid_endian, ")")

    # readout hint
    if readout is None:
        acq_size = _parse_acq_size(method, acqp)
        if acq_size:
            readout = acq_size[0]
    if readout is None and matrix_ro_hint:
        readout = matrix_ro_hint
    dbg(debug, "readout (hdr/matrix hint):", readout)

    # coils
    if coils is None:
        nrec = _get_int_from_headers(["PVM_EncNReceivers"], [method])
        if nrec and nrec > 0:
            coils = nrec
    if coils is None or coils <= 0:
        coils = 1
    dbg(debug, "coils (initial):", coils)

    # extras (echoes, reps, averages, slices)
    extras = {
        "echoes": _get_int_from_headers(
            ["NECHOES", "ACQ_n_echo_images", "PVM_NEchoImages"], [method, acqp]
        ) or 1,
        "reps": _get_int_from_headers(["PVM_NRepetitions", "NR"], [method, acqp]) or 1,
        "averages": _get_int_from_headers(["PVM_NAverages", "NA"], [method, acqp]) or 1,
        "slices": _get_int_from_headers(["NSLICES", "PVM_SPackArrNSlices"], [method, acqp]) or 1,
    }
    other_dims = 1
    for v in extras.values():
        if isinstance(v, int) and v > 1:
            other_dims *= v
    dbg(debug, "other_dims factor:", other_dims, extras)

    denom = coils * max(1, other_dims)
    if total % denom != 0:
        dbg(debug, "total not divisible by coils*other_dims; relaxing extras")
        denom = coils
        if total % denom != 0:
            dbg(debug, "still not divisible; relaxing coils->1")
            coils = 1
            denom = coils
            if total % denom != 0:
                raise ValueError("Cannot factor FID length with any (coils, other_dims) combo.")
    per_coil_total = total // denom

    def pick_block_and_spokes(per_coil_total: int,
                              readout_hint: Optional[int],
                              spokes_hint_local: Optional[int]) -> Tuple[int, int]:
        # If we have a spokes hint and it divides the total, prefer that
        if spokes_hint_local and spokes_hint_local > 0 and per_coil_total % spokes_hint_local == 0:
            return per_coil_total // spokes_hint_local, spokes_hint_local

        # Common stored RO block sizes Bruker tends to use
        BLOCKS = [
            128, 160, 192, 200, 224, 240, 256, 288, 320, 352, 384, 400, 416, 420, 432,
            448, 480, 496, 512, 544, 560, 576, 608, 640, 672, 704, 736, 768, 800, 832,
            896, 960, 992, 1024, 1152, 1280, 1536, 2048,
        ]
        if readout_hint and per_coil_total % readout_hint == 0:
            return readout_hint, per_coil_total // readout_hint
        for b in [x for x in BLOCKS if (not readout_hint) or x >= readout_hint]:
            if per_coil_total % b == 0:
                return b, per_coil_total // b

        # fallback: factor near sqrt
        s = int(round(per_coil_total ** 0.5))
        for d in range(0, s + 1):
            for cand in (s + d, s - d):
                if cand > 0 and per_coil_total % cand == 0:
                    return cand, per_coil_total // cand
        raise ValueError("Could not factor per_coil_total into (stored_ro, spokes).")

    stored_ro, spokes_inf = pick_block_and_spokes(per_coil_total, readout, spokes_hint)
    dbg(debug, "stored_ro (block):", stored_ro,
        " spokes (per extras-collapsed):", spokes_inf)

    spokes_final = spokes_inf * max(1, other_dims)
    if stored_ro * spokes_final * coils != total:
        raise ValueError("Internal factoring error: stored_ro*spokes_final*coils != total samples")

    ksp_blk = np.reshape(cpx, (stored_ro, spokes_final, coils), order="F")

    if readout is not None and stored_ro >= readout:
        ksp = ksp_blk[:readout, :, :]
        dbg(debug, "trimmed RO from", stored_ro, "to", readout)
    else:
        ksp = ksp_blk
        if readout is None:
            readout = stored_ro

    dbg(debug, "final k-space shape:", ksp.shape, "(RO, Spokes, Coils)")
    return ksp


# -------------- GA / Kronecker math (Python port of your C) --------------

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
    p = fib_prev(fk)
    return fib_prev(p)


def uv_to_dir(u: float, v: float) -> Tuple[float, float, float]:
    z = 1.0 - 2.0 * u
    r2 = 1.0 - z * z
    if r2 < 0.0:
        r2 = 0.0
    r = math.sqrt(r2)
    az = 2.0 * math.pi * v
    dx = r * math.cos(az)
    dy = r * math.sin(az)
    dz = z
    return dx, dy, dz


def kronecker_dir(i: int, N: int) -> Tuple[float, float, float]:
    M = fib_closest_ge(N)
    q1 = fib_prev(M)
    q2 = fib_prev2(M)
    j = i % M
    u = ((j * q1) % M + 0.5) / float(M)
    v = ((j * q2) % M + 0.5) / float(M)
    return uv_to_dir(u, v)


def linz_ga_dir(i: int, N: int) -> Tuple[float, float, float]:
    phi_inc = (math.sqrt(5.0) - 1.0) * math.pi
    z = 1.0 - 2.0 * ((i + 0.5) / float(N))
    r2 = 1.0 - z * z
    if r2 < 0.0:
        r2 = 0.0
    r = math.sqrt(r2)
    az = (i * phi_inc) % (2.0 * math.pi)
    dx = r * math.cos(az)
    dy = r * math.sin(az)
    dz = z
    return dx, dy, dz


def build_ga_traj(
    mode: str,
    ro: int,
    spokes: int,
    gr: float = 1.0,
    gp: float = 1.0,
    gs: float = 1.0,
) -> np.ndarray:
    """
    mode: "kron" or "linz"

    Port of your Bruker snippet:

        const long N = GA_NSpokesEff;
        for (long i = 0; i < N; ++i) {
          double dx, dy, dz;
          if (GA_Mode == GA_Traj_Kronecker)
            kronecker_dir(i, N, &dx, &dy, &dz);
          else  /* GA_Traj_LinZ_GA */
            linZ_ga_dir(i, N, &dx, &dy, &dz);

          r[i] = dx * gr;
          p[i] = dy * gp;
          s[i] = dz * gs;
        }

    Here we build (3, RO, Spokes): for each spoke we sample along
    radius r in [-0.5, 0.5) scaled by the chosen direction (dx,dy,dz).
    """
    N = spokes
    dirs = np.zeros((3, N), dtype=np.float32)
    for i in range(N):
        if mode == "kron":
            dx, dy, dz = kronecker_dir(i, N)
        else:
            dx, dy, dz = linz_ga_dir(i, N)
        dirs[0, i] = dx * gr
        dirs[1, i] = dy * gp
        dirs[2, i] = dz * gs

    # per-readout sample positions along each spoke
    r = np.linspace(-0.5, 0.5, ro, endpoint=False, dtype=np.float32)
    traj = np.zeros((3, ro, spokes), dtype=np.float32)
    for s in range(spokes):
        dx, dy, dz = dirs[:, s]
        traj[0, :, s] = r * dx
        traj[1, :, s] = r * dy
        traj[2, :, s] = r * dz
    return traj


# -------------- frame binning indices --------------

def frame_starts(total_spokes: int,
                 spokes_per_frame: int,
                 frame_shift: Optional[int]) -> List[int]:
    step = frame_shift if frame_shift and frame_shift > 0 else spokes_per_frame
    if spokes_per_frame <= 0:
        return []
    return list(range(0, max(0, total_spokes - spokes_per_frame + 1), step))


# -------------- main CLI --------------

def main():
    ap = argparse.ArgumentParser(
        description="Bruker 3D radial prep for BART (FID + GA/Kronecker trajectory)")
    ap.add_argument("--series", type=Path, required=True,
                    help="Bruker series directory (contains fid, method, acqp)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output base name (without extension)")
    ap.add_argument("--matrix", type=int, nargs=3, required=True,
                    metavar=("NX", "NY", "NZ"))
    ap.add_argument("--traj-mode", choices=["kron", "linz"], default="kron",
                    help="Synthetic trajectory mode: kron or linz")
    ap.add_argument("--spokes-per-frame", type=int, default=None,
                    help="If set, define sliding-window frame size in spokes")
    ap.add_argument("--frame-shift", type=int, default=None,
                    help="Sliding-window shift in spokes (default = spokes-per-frame)")
    ap.add_argument("--test-volumes", type=int, default=None,
                    help="If set, only keep this many frame entries in frames.txt")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    NX, NY, NZ = args.matrix
    series_dir = args.series
    out_base = args.out

    # 1) Load k-space (RO, Spokes, Coils)
    ksp = load_bruker_kspace(
        series_dir,
        matrix_ro_hint=NX,
        spokes_hint=None,
        debug=args.debug,
    )
    ro, sp_total, nc = ksp.shape
    print(f"[info] Loaded k-space: RO={ro}, Spokes={sp_total}, Coils={nc}")

    # 2) Build synthetic GA/Kronecker trajectory
    traj_path = series_dir / "traj"
    if traj_path.exists():
        print("[info] NOTE: traj file exists but this script currently uses synthetic traj.")
        print("       (You can wire in a reader later if you want raw Bruker traj.)")

    mode = args.traj_mode
    print(f"[info] Building synthetic {mode} trajectory (RO={ro}, Spokes={sp_total})")
    traj = build_ga_traj(mode, ro, sp_total)

    # 3) Write full ksp / traj for BART
    ksp_all_base = out_base.with_name(out_base.name + "_ksp_all")
    traj_all_base = out_base.with_name(out_base.name + "_traj_all")

    # Simple 3D layout: (RO, Spokes, Coils) and (3, RO, Spokes)
    write_cfl(ksp_all_base, ksp, [ro, sp_total, nc])
    write_cfl(traj_all_base, traj.astype(np.complex64), [3, ro, sp_total])

    print(f"[info] Wrote full k-space to {ksp_all_base}.cfl/.hdr")
    print(f"[info] Wrote full traj   to {traj_all_base}.cfl/.hdr")

    # 4) Sliding-window frame indices (no per-frame images, just index metadata)
    if args.spokes_per_frame is not None:
        spf = args.spokes_per_frame
        shift = args.frame_shift if args.frame_shift is not None else spf
        starts = frame_starts(sp_total, spf, shift)
        if args.test_volumes is not None:
            starts = starts[: max(0, int(args.test_volumes))]
        if not starts:
            print("[warn] No frames formed with these parameters; skipping frames.txt.")
        else:
            print(f"[info] Sliding-window frames: {len(starts)} "
                  f"(spf={spf}, shift={shift})")
            meta_path = out_base.with_suffix(".frames.txt")
            with open(meta_path, "w") as f:
                f.write("# start_spoke  end_spoke_exclusive\n")
                for s0 in starts:
                    s1 = s0 + spf
                    if s1 <= sp_total:
                        f.write(f"{s0} {s1}\n")
            print(f"[info] Wrote frame index table to {meta_path}")
            print("       Use BART 'extract' along the spokes dimension to build frames.")
    else:
        print("[info] No frame binning requested (no --spokes-per-frame).")

    print("[info] Done. Next step is to drive BART nufft/rss/toimg at the shell "
          "using *_ksp_all and *_traj_all.")


if __name__ == "__main__":
    main()
