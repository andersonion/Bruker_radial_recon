#!/usr/bin/env python3
"""
Bruker 3D radial -> BART NUFFT recon driver.

Key points:
  - For BART NUFFT, k-space must be written as (1, RO, spokes, coils)
    (BART asserts ksp_dims[0] == 1).
  - Supports trajectory from:
        * traj file (Bruker writes <series>/traj)  --traj-source trajfile
        * grad.output directions                   --traj-source gradoutput
        * auto (prefer traj if parsable else grad) --traj-source auto
  - Supports FID dtypes i2/i4/f4 and endianness
  - Supports FID layouts:
        ro_spokes_coils (default)
        ro_coils_spokes
  - Supports spoke expansion tile/repeat when spokes_all is a multiple of NPro or N_dirs
"""

import argparse
import sys
import subprocess
import textwrap
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import nibabel as nib


# ---------------- Bruker helpers ---------------- #

def read_bruker_param(path: Path, key: str, default=None):
    if not path.exists():
        return default

    token = f"##${key}="
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except Exception:
        return default

    for i, line in enumerate(lines):
        if not line.startswith(token):
            continue

        rhs = line.split("=", 1)[1].strip()

        if rhs.startswith("("):  # multiline array
            vals = []
            j = i + 1
            while j < len(lines):
                l2 = lines[j].strip()
                if l2.startswith("##"):
                    break
                if l2 and not l2.startswith("$$"):
                    vals.extend(l2.split())
                j += 1

            out = []
            for v in vals:
                try:
                    out.append(float(v) if "." in v or "e" in v.lower() else int(v))
                except ValueError:
                    out.append(v)
            return out[0] if len(out) == 1 else out

        rhs = rhs.strip("()")
        toks = rhs.split()
        out = []
        for v in toks:
            try:
                out.append(float(v) if "." in v or "e" in v.lower() else int(v))
            except ValueError:
                out.append(v)
        return out[0] if len(out) == 1 else out

    return default


def infer_matrix(method: Path):
    mat = read_bruker_param(method, "PVM_Matrix", None)
    if mat is None or isinstance(mat, (int, float)) or len(mat) != 3:
        raise ValueError(f"Could not infer PVM_Matrix (got {mat})")
    return tuple(map(int, mat))


def infer_true_ro(acqp: Path) -> int:
    v = read_bruker_param(acqp, "ACQ_size", None)
    if v is None:
        raise ValueError("Could not infer ACQ_size")
    return int(v if isinstance(v, (int, float)) else v[0])


def infer_coils(method: Path) -> int:
    v = read_bruker_param(method, "PVM_EncNReceivers", 1)
    return int(v if not isinstance(v, (list, tuple)) else v[0])


def infer_npro(method: Path, acqp: Path) -> Optional[int]:
    v = read_bruker_param(method, "NPro", None)
    if v is None:
        v = read_bruker_param(acqp, "NPro", None)
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return int(v[0])
    try:
        return int(v)
    except Exception:
        return None


# ---------------- FID / k-space loading ---------------- #

def factor_fid(total_points: int, true_ro: int, coils_hint: int | None):
    block_candidates = [true_ro]
    for b in (128, 96, 64, 256, 192, 384, 512):
        if b != true_ro:
            block_candidates.append(b)

    if coils_hint is not None:
        coil_candidates = [coils_hint] + [c for c in range(1, 33) if c != coils_hint]
    else:
        coil_candidates = list(range(1, 33))

    best = None
    best_score = -1

    for stored_ro in block_candidates:
        for c in coil_candidates:
            denom = stored_ro * c
            if denom <= 0:
                continue
            if total_points % denom != 0:
                continue
            spokes = total_points // denom

            score = 0
            if stored_ro >= true_ro:
                score += 2
            if abs(stored_ro - true_ro) <= 10:
                score += 2
            if spokes > 100:
                score += 1
            if coils_hint is not None and c == coils_hint:
                score += 2

            if score > best_score:
                best_score = score
                best = (stored_ro, c, spokes)

    if best is None:
        raise ValueError(
            f"Could not factor FID: total={total_points}, true_ro={true_ro}, coils_hint={coils_hint}"
        )
    return best  # (stored_ro, coils, spokes)


def load_bruker_kspace(
    fid_path: Path,
    true_ro: int,
    coils_hint: int | None,
    endian: str,
    base_kind: str,
    fid_layout: str,
):
    if not fid_path.exists():
        raise FileNotFoundError(f"FID not found: {fid_path}")

    np_dtype = np.dtype(endian + base_kind)
    raw = np.fromfile(fid_path, dtype=np_dtype).astype(np.float32)
    if raw.size % 2 != 0:
        raise ValueError(f"FID length {raw.size} not even (real/imag pairs).")

    cplx = raw[0::2] + 1j * raw[1::2]
    total_points = cplx.size

    stored_ro, coils, spokes = factor_fid(total_points, true_ro, coils_hint)

    if coils_hint is not None and coils != coils_hint:
        print(
            f"[warn] FID suggests coils={coils}, header said {coils_hint}; using {coils}.",
            file=sys.stderr,
        )

    print(
        f"[info] Loaded k-space with stored_ro={stored_ro}, true_ro={true_ro}, spokes={spokes}, coils={coils}"
    )

    if fid_layout == "ro_spokes_coils":
        ksp = cplx.reshape(stored_ro, spokes, coils)
    elif fid_layout == "ro_coils_spokes":
        ksp = cplx.reshape(stored_ro, coils, spokes).transpose(0, 2, 1)
    else:
        raise ValueError(f"Unknown fid_layout: {fid_layout}")

    if stored_ro != true_ro:
        if stored_ro < true_ro:
            raise ValueError(f"stored_ro={stored_ro} < true_ro={true_ro}")
        ksp = ksp[:true_ro]
        print(f"[info] Trimmed k-space from stored_ro={stored_ro} to true_ro={true_ro}")

    return ksp, stored_ro, spokes, coils


# ---------------- BART CFL helpers ---------------- #

def writecfl(name: str, arr: np.ndarray):
    base = Path(name)
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")

    arr_f = np.asfortranarray(arr.astype(np.complex64))
    dims = list(arr_f.shape) + [1] * (16 - arr_f.ndim)

    with hdr.open("w") as f:
        f.write("# Dimensions\n")
        f.write(" ".join(str(d) for d in dims) + "\n")

    stacked = np.empty(arr_f.size * 2, dtype=np.float32)
    stacked[0::2] = arr_f.real.ravel(order="F")
    stacked[1::2] = arr_f.imag.ravel(order="F")
    stacked.tofile(cfl)


def readcfl(name: str) -> np.ndarray:
    base = Path(name)
    hdr = base.with_suffix(".hdr")
    cfl = base.with_suffix(".cfl")
    if not hdr.exists() or not cfl.exists():
        raise FileNotFoundError(f"CFL/HDR not found: {base}")

    lines = hdr.read_text(errors="ignore").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Malformed hdr: {hdr}")

    dims16 = [int(x) for x in lines[1].split()]
    last_non1 = 0
    for i, d in enumerate(dims16):
        if d > 1:
            last_non1 = i
    ndim = max(1, last_non1 + 1)
    dims = dims16[:ndim]

    data = np.fromfile(cfl, dtype=np.float32)
    if data.size % 2 != 0:
        raise ValueError(f"CFL length {data.size} not even: {cfl}")

    cplx = data[0::2] + 1j * data[1::2]
    expected = int(np.prod(dims))
    if cplx.size != expected:
        raise ValueError(
            f"CFL size mismatch: have {cplx.size} complex, expected {expected} from dims {dims} for {base}"
        )

    return cplx.reshape(dims, order="F")


def bart_image_dims(bart_bin: str, base: Path) -> list[int] | None:
    dims: list[int] = []
    for d in range(16):
        proc = subprocess.run(
            [bart_bin, "show", "-d", str(d), str(base)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            print(f"[warn] bart show -d {d} failed: {proc.stderr.strip()}", file=sys.stderr)
            return None
        dims.append(int(proc.stdout.strip()))
    return dims


def bart_supports_gpu(bart_bin: str = "bart") -> bool:
    proc = subprocess.run([bart_bin, "nufft", "-i", "-g"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode == 0:
        return True
    s = (proc.stderr or "").lower()
    if "compiled without gpu support" in s or "unknown option" in s:
        return False
    return False

def _traj_monotonic_score(traj_3_ro_sp: np.ndarray) -> float:
    """
    traj_3_ro_sp: (3, RO, spokes) float32/float64
    Score higher when |k| starts small and increases with RO (center-out).
    """
    k_mag = np.sqrt(
        traj_3_ro_sp[0] ** 2 +
        traj_3_ro_sp[1] ** 2 +
        traj_3_ro_sp[2] ** 2
    )  # (RO, spokes)

    ro0 = float(np.median(k_mag[0, :]))
    rom = float(np.median(k_mag[k_mag.shape[0] // 2, :]))
    rol = float(np.median(k_mag[-1, :]))

    slope = rol - ro0
    mid_bonus = rom - ro0

    start_frac = (ro0 / rol) if rol > 0 else 1.0

    score = 0.0
    score += 5.0 * slope
    score += 2.0 * mid_bonus
    score += -10.0 * start_frac

    return score


def parse_trajfile_autoshape(traj_path: Path, *, npro: int, ro: int) -> tuple[np.ndarray, str]:
    """
    Returns:
      traj (3, RO, NPro) float32
      tag describing chosen interpretation

    Tries:
      - float32 LE/BE
      - int32 LE/BE scaled by Q30
    and multiple reshape/permutation/order hypotheses, selecting the one
    that best matches a center-out readout (|k| increasing with RO).
    """
    b = traj_path.read_bytes()
    if len(b) % 4 != 0:
        raise ValueError(f"traj bytes not multiple of 4: {len(b)}")

    nelem = len(b) // 4
    expected = 3 * ro * npro
    if nelem != expected:
        raise ValueError(
            f"traj element count mismatch: have {nelem}, expected {expected} (=3*RO*NPro)"
        )

    candidates: list[tuple[str, np.ndarray]] = []

    # float32
    candidates.append(("f4_le", np.frombuffer(b, dtype="<f4").astype(np.float32)))
    candidates.append(("f4_be", np.frombuffer(b, dtype=">f4").astype(np.float32)))

    # int32 Q30
    candidates.append(("i4_le_q30", np.frombuffer(b, dtype="<i4").astype(np.float32) / float(1 << 30)))
    candidates.append(("i4_be_q30", np.frombuffer(b, dtype=">i4").astype(np.float32) / float(1 << 30)))

    shapes = [
        ("3_ro_sp", (3, ro, npro)),
        ("ro_3_sp", (ro, 3, npro)),
        ("sp_ro_3", (npro, ro, 3)),
        ("sp_3_ro", (npro, 3, ro)),
        ("ro_sp_3", (ro, npro, 3)),
        ("3_sp_ro", (3, npro, ro)),
    ]
    orders = ["C", "F"]

    best_score: float | None = None
    best_tag: str | None = None
    best_traj: np.ndarray | None = None

    for dtype_tag, v in candidates:
        if not np.all(np.isfinite(v)):
            continue

        vmax = float(np.max(np.abs(v)))
        if dtype_tag.startswith("f4") and vmax > 1e6:
            continue

        for shape_tag, shp in shapes:
            for ord_tag in orders:
                try:
                    arr = v.reshape(shp, order=ord_tag)
                except Exception:
                    continue

                if shape_tag == "3_ro_sp":
                    traj = arr
                elif shape_tag == "ro_3_sp":
                    traj = np.transpose(arr, (1, 0, 2))
                elif shape_tag == "sp_ro_3":
                    traj = np.transpose(arr, (2, 1, 0))
                elif shape_tag == "sp_3_ro":
                    traj = np.transpose(arr, (1, 2, 0))
                elif shape_tag == "ro_sp_3":
                    traj = np.transpose(arr, (2, 0, 1))
                elif shape_tag == "3_sp_ro":
                    traj = np.transpose(arr, (0, 2, 1))
                else:
                    continue

                score = _traj_monotonic_score(traj)

                # Penalize if RO[0] is not near zero relative to RO[-1]
                k_mag0 = np.sqrt(traj[0, 0, :] ** 2 + traj[1, 0, :] ** 2 + traj[2, 0, :] ** 2)
                k_magl = np.sqrt(traj[0, -1, :] ** 2 + traj[1, -1, :] ** 2 + traj[2, -1, :] ** 2)
                ro0_med = float(np.median(k_mag0))
                rol_med = float(np.median(k_magl))
                if rol_med > 0 and (ro0_med / rol_med) > 0.25:
                    score -= 100.0

                tag = f"{dtype_tag}:{shape_tag}:{ord_tag}"

                if best_score is None or score > best_score:
                    best_score = score
                    best_tag = tag
                    best_traj = traj

    if best_traj is None or best_tag is None:
        raise RuntimeError(f"Could not find any plausible traj reshape for {traj_path}")

    k_mag = np.sqrt(best_traj[0] ** 2 + best_traj[1] ** 2 + best_traj[2] ** 2)
    ro0 = float(np.median(k_mag[0, :]))
    rom = float(np.median(k_mag[k_mag.shape[0] // 2, :]))
    rol = float(np.median(k_mag[-1, :]))
    end_spread = np.percentile(k_mag[-1, :], [5, 50, 95])

    print(f"[info] traj autoshape chose {best_tag}")
    print(f"[debug] traj sanity |k| median at RO[0],RO[mid],RO[-1]: {ro0:.6g}, {rom:.6g}, {rol:.6g}")
    print(f"[debug] traj sanity |k| at RO[-1] p5/p50/p95: {end_spread[0]:.6g}, {end_spread[1]:.6g}, {end_spread[2]:.6g}")

    return best_traj.astype(np.float32), best_tag


# ---------------- grad.output trajectory ---------------- #

def _extract_grad_block(lines: list[str], name: str) -> np.ndarray:
    hdr_pat = re.compile(rf"^\s*\d+:{re.escape(name)}:\s+index\s*=\s*\d+,\s+size\s*=\s*(\d+)\s*$")

    start = None
    size = None
    for i, line in enumerate(lines):
        m = hdr_pat.match(line.strip())
        if m:
            start = i + 1
            size = int(m.group(1))
            break

    if start is None or size is None:
        raise ValueError(f"Could not find {name} in grad.output")

    vals: list[int] = []
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+:\w+:", line) or line.startswith("Ramp Shape"):
            break
        parts = line.split()
        if len(parts) < 2:
            continue
        vals.append(int(parts[1]))
        if len(vals) >= size:
            break

    if len(vals) != size:
        raise ValueError(f"{name}: expected {size} values, got {len(vals)}")
    return np.array(vals, dtype=np.float64)


def load_grad_output_dirs(grad_output_path: Path, normalize: bool = True) -> np.ndarray:
    lines = grad_output_path.read_text(errors="ignore").splitlines()
    r = _extract_grad_block(lines, "ProjR")
    p = _extract_grad_block(lines, "ProjP")
    s = _extract_grad_block(lines, "ProjS")

    scale = float(1 << 30)
    dirs = np.stack([r, p, s], axis=1) / scale

    norms = np.linalg.norm(dirs, axis=1)
    print(f"[info] Parsed {dirs.shape[0]} spoke directions from {grad_output_path}")
    print(f"[info] Direction norms BEFORE normalize: min={norms.min():.4f} median={np.median(norms):.4f} max={norms.max():.4f}")

    if normalize:
        bad = norms == 0
        if np.any(bad):
            raise ValueError(f"Found {bad.sum()} zero-norm direction(s) in {grad_output_path}")
        dirs = dirs / norms[:, None]
        norms2 = np.linalg.norm(dirs, axis=1)
        print(f"[info] Direction norms AFTER  normalize: min={norms2.min():.4f} median={np.median(norms2):.4f} max={norms2.max():.4f}")

    return dirs


def expand_spoke_dirs(dirs: np.ndarray, target_spokes: int, order: str) -> np.ndarray:
    n = dirs.shape[0]
    if target_spokes == n:
        return dirs
    if target_spokes % n != 0:
        raise ValueError(f"Cannot expand dirs length {n} to {target_spokes} (not divisible)")
    reps = target_spokes // n
    print(f"[info] Expanding {n} dirs to {target_spokes} spokes with reps={reps} using order='{order}'")
    if order == "tile":
        return np.tile(dirs, (reps, 1))
    if order == "repeat":
        return np.repeat(dirs, reps, axis=0)
    raise ValueError(f"Unknown spoke-order: {order}")


def build_traj_from_dirs(
    true_ro: int,
    dirs_xyz: np.ndarray,   # (spokes,3)
    NX: int,
    traj_scale: float | None,
    readout_origin: str,
    reverse_readout: bool,
) -> np.ndarray:
    spokes = dirs_xyz.shape[0]
    # BART NUFFT expects k in cycles/FOV units ~ [-0.5, 0.5]
    kmax = 0.5
	    if traj_scale is not None:
        kmax *= float(traj_scale)

    if readout_origin == "zero":
        s = np.linspace(0.0, kmax, true_ro, dtype=np.float64)
    else:
        s = np.linspace(-kmax, kmax, true_ro, dtype=np.float64)

    if reverse_readout:
        s = s[::-1].copy()

    traj = np.zeros((3, true_ro, spokes), dtype=np.complex64)
    dx, dy, dz = dirs_xyz[:, 0], dirs_xyz[:, 1], dirs_xyz[:, 2]
    for i in range(spokes):
        traj[0, :, i] = s * dx[i]
        traj[1, :, i] = s * dy[i]
        traj[2, :, i] = s * dz[i]

    print(f"[info] Traj built with max |k| ≈ {np.abs(traj).max():.2f} (origin={readout_origin}, reverse={reverse_readout})")
    return traj


# ---------------- traj file support ---------------- #

def find_traj_candidates(series_path: Path) -> List[Path]:
    cands: List[Path] = []

    p0 = series_path / "traj"
    if p0.exists() and p0.is_file():
        cands.append(p0)

    pdata = series_path / "pdata"
    if pdata.exists():
        for p in pdata.rglob("traj"):
            if p.exists() and p.is_file():
                cands.append(p)

    seen = set()
    out = []
    for p in cands:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _trajfile_stats(arr: np.ndarray) -> Tuple[float, float, float]:
    # returns (finite_frac, max_abs, median_abs)
    finite = np.isfinite(arr)
    finite_frac = float(finite.mean())
    if finite_frac == 0.0:
        return 0.0, float("inf"), float("inf")
    a = np.abs(arr[finite])
    return finite_frac, float(a.max()), float(np.median(a))


def parse_trajfile_bruker_traj_autofmt(
    traj_path: Path,
    npro: int,
    true_ro: int,
) -> Tuple[np.ndarray, str]:
    """
    Parse Bruker <series>/traj for 3D radial as shape (NPro, RO, 3) and return
    BART-style (3, RO, NPro) complex64 trajectory.

    We auto-try common encodings:
      - float32 (LE/BE)
      - int32 fixed-point (LE/BE), scaled by 2^30
      - int16 fixed-point (LE/BE), scaled by 2^15

    We pick the candidate with:
      - 100% finite values
      - a reasonable max(|k|) (not tiny, not astronomical)
      - and nontrivial median magnitude

    Returns (traj, description_string).
    """
    nbytes = traj_path.stat().st_size
    expect = npro * true_ro * 3

    # Helpers
    def _as_traj_from_float(arr_f: np.ndarray) -> np.ndarray:
        arr = arr_f.reshape(npro, true_ro, 3).transpose(2, 1, 0)  # (3, ro, npro)
        traj = np.asfortranarray(arr.astype(np.float32)).astype(np.complex64)
        return traj

    def _stats_real(x: np.ndarray) -> Tuple[float, float, float]:
        finite = np.isfinite(x)
        finite_frac = float(finite.mean())
        if finite_frac == 0.0:
            return 0.0, float("inf"), float("inf")
        a = np.abs(x[finite])
        return finite_frac, float(a.max()), float(np.median(a))

    def _score(finite_frac: float, max_abs: float, med_abs: float) -> float:
        # Must be fully finite to be trusted
        if finite_frac < 1.0:
            return -1e12
        # Reject obviously wrong scales
        if max_abs <= 1e-6:
            return -1e12
        if max_abs >= 1e10:
            return -1e12
        # Prefer max_abs in a "BART-ish" range; keep this loose
        target = 50.0
        return -abs(np.log((max_abs + 1e-12) / target)) + 0.05 * np.log(med_abs + 1e-12)

    # Build candidates: (name, float_array)
    candidates: List[Tuple[str, np.ndarray]] = []

    # float32
    if nbytes == expect * 4:
        raw_le = np.fromfile(traj_path, dtype=np.dtype("<f4"))
        raw_be = raw_le.byteswap().newbyteorder()
        candidates.append(("f4_le", raw_le))
        candidates.append(("f4_be", raw_be))

    # int32 fixed-point (common Bruker style)
    if nbytes == expect * 4:
        raw_le_i4 = np.fromfile(traj_path, dtype=np.dtype("<i4")).astype(np.float64) / float(1 << 30)
        raw_be_i4 = raw_le_i4.byteswap().newbyteorder()
        candidates.append(("i4_le_q30", raw_le_i4.astype(np.float32)))
        candidates.append(("i4_be_q30", raw_be_i4.astype(np.float32)))

    # int16 fixed-point
    if nbytes == expect * 2:
        raw_le_i2 = np.fromfile(traj_path, dtype=np.dtype("<i2")).astype(np.float64) / float(1 << 15)
        raw_be_i2 = raw_le_i2.byteswap().newbyteorder()
        candidates.append(("i2_le_q15", raw_le_i2.astype(np.float32)))
        candidates.append(("i2_be_q15", raw_be_i2.astype(np.float32)))

    if not candidates:
        raise RuntimeError(
            f"traj size {nbytes} bytes does not match expected encodings for "
            f"(NPro={npro}, RO={true_ro}, 3): "
            f"expected {expect*4} (f4/i4) or {expect*2} (i2)."
        )

    best_name = None
    best_score = -1e18
    best_traj = None

    for name, raw in candidates:
        fin, mx, med = _stats_real(raw)
        print(f"[info] traj probe {name}: finite={fin:.3f} max|k|={mx:.6g} med|k|={med:.6g}")
        sc = _score(fin, mx, med)
        if sc > best_score:
            best_score = sc
            best_name = name
            best_traj = _as_traj_from_float(raw)

    if best_traj is None or best_name is None:
        raise RuntimeError("Failed to select a valid traj interpretation (all candidates rejected).")

    # Final sanity
    if not np.isfinite(best_traj.real).all():
        raise RuntimeError("Parsed traj contains non-finite values after candidate selection.")

    print(f"[info] Parsed trajfile {traj_path} using {best_name} with max |k|={np.abs(best_traj).max():.6g}")
    return best_traj, best_name


def expand_traj_spokes(traj: np.ndarray, target_spokes: int, order: str) -> np.ndarray:
    """
    traj: (3, ro, nsp)
    Expand along spoke axis to (3, ro, target_spokes) if divisible.
    """
    nsp = traj.shape[2]
    if target_spokes == nsp:
        return traj
    if target_spokes % nsp != 0:
        raise ValueError(f"Cannot expand traj spokes {nsp} to {target_spokes} (not divisible)")
    reps = target_spokes // nsp
    print(f"[info] Expanding traj spokes {nsp} -> {target_spokes} with reps={reps} using order='{order}'")
    if order == "tile":
        return np.tile(traj, (1, 1, reps))
    if order == "repeat":
        return np.repeat(traj, reps, axis=2)
    raise ValueError(f"Unknown spoke-order: {order}")


def load_traj_auto(
    series_path: Path,
    method: Path,
    acqp: Path,
    true_ro: int,
    spokes_all: int,
    NX: int,
    traj_scale: float | None,
    spoke_order: str,
    readout_origin: str,
    reverse_readout: bool,
    traj_source: str,
    traj_file: Optional[Path],
) -> Tuple[np.ndarray, str]:
    """
    Return traj_full (3,RO,spokes_all) and description string.
    """

    def _from_gradoutput() -> np.ndarray:
        grad_path = series_path / "grad.output"
        if not grad_path.exists():
            raise RuntimeError(f"grad.output not found in {series_path}")
        dirs = load_grad_output_dirs(grad_path, normalize=True)
        dirs_full = expand_spoke_dirs(dirs, spokes_all, spoke_order)
        return build_traj_from_dirs(true_ro, dirs_full, NX, traj_scale, readout_origin, reverse_readout)

    def _from_trajfile(p: Path) -> Tuple[np.ndarray, str]:
        npro = infer_npro(method, acqp)
        if npro is None:
            raise RuntimeError("trajfile parsing requires NPro, but could not read ##$NPro from method/acqp.")

        # Parse traj as (3, RO, NPro)
        traj, fmt = parse_trajfile_bruker_traj_autofmt(p, npro=int(npro), true_ro=true_ro)

        # ---- trajfile-specific sanity / fixing ----
        # NOTE: traj is complex64 (imag=0) because we store CFL that way.
        # Always operate on the real part to avoid ComplexWarning.
        tx = np.real(traj[0])
        ty = np.real(traj[1])
        tz = np.real(traj[2])

        k_mag = np.sqrt(tx * tx + ty * ty + tz * tz)  # (RO, NPro)
        ro0 = float(np.median(k_mag[0, :]))
        rol = float(np.median(k_mag[-1, :]))
        print(f"[debug] trajfile pre-fix |k| median RO[0]={ro0:.6g} RO[-1]={rol:.6g}")

        # For center-out, |k| should generally increase with RO.
        # If it decreases, flip the readout axis.
        if rol < ro0:
            traj = traj[:, ::-1, :].copy()
            tx = np.real(traj[0])
            ty = np.real(traj[1])
            tz = np.real(traj[2])
            k_mag = np.sqrt(tx * tx + ty * ty + tz * tz)
            ro0 = float(np.median(k_mag[0, :]))
            rol = float(np.median(k_mag[-1, :]))
            print(f"[info] trajfile: flipped readout axis so center is at RO[0]. Now RO[0]={ro0:.6g} RO[-1]={rol:.6g}")

        # Apply explicit reverse-readout flag for trajfile too
        if reverse_readout:
            traj = traj[:, ::-1, :].copy()
            tx = np.real(traj[0])
            ty = np.real(traj[1])
            tz = np.real(traj[2])
            k_mag = np.sqrt(tx * tx + ty * ty + tz * tz)
            ro0 = float(np.median(k_mag[0, :]))
            rol = float(np.median(k_mag[-1, :]))
            print(f"[info] trajfile: applied --reverse-readout. Now RO[0]={ro0:.6g} RO[-1]={rol:.6g}")

        # Expand spokes if acquisition has multiple volumes and traj is per-volume
        traj = expand_traj_spokes(traj, target_spokes=spokes_all, order=spoke_order)

        # ---- IMPORTANT: BART NUFFT trajectory scaling ----
        # BART expects k in cycles/FOV (about [-0.5, 0.5], or [0, 0.5] for center-out).
        # So the correct target kmax is 0.5, NOT NX/2.
        #
        # Auto-scale based on the median radius at the last RO sample (robust for center-out).
        tx = np.real(traj[0])
        ty = np.real(traj[1])
        tz = np.real(traj[2])
        k_end = np.sqrt(tx[-1, :] * tx[-1, :] + ty[-1, :] * ty[-1, :] + tz[-1, :] * tz[-1, :])
        kmax_curr = float(np.median(k_end))
        kmax_tgt = 0.5

        if kmax_curr > 0:
            scale = kmax_tgt / kmax_curr
            traj = (traj * scale).astype(np.complex64, copy=False)
            print(f"[info] trajfile auto-scale (BART units): kmax_curr≈{kmax_curr:.6g} -> kmax_tgt={kmax_tgt} (scale={scale:.6g})")
        else:
            print("[warn] trajfile auto-scale skipped: kmax_curr≈0", file=sys.stderr)

        return traj, f"trajfile:{fmt}:(NPro={npro},RO={true_ro},3)"


    if traj_source == "gradoutput":
        return _from_gradoutput(), "gradoutput"

    if traj_file is not None:
        if not traj_file.exists():
            raise RuntimeError(f"--traj-file does not exist: {traj_file}")
        traj, mode = _from_trajfile(traj_file)
        return traj, mode

    if traj_source in ("trajfile", "auto"):
        cands = find_traj_candidates(series_path)
        if traj_source == "trajfile" and not cands:
            raise RuntimeError(f"--traj-source trajfile requested, but no traj candidate found under {series_path} or pdata/*")

        last_err = None
        for p in cands:
            try:
                traj, mode = _from_trajfile(p)
                return traj, mode
            except Exception as e:
                last_err = e

        if traj_source == "trajfile":
            raise RuntimeError(
                f"--traj-source trajfile requested, but no candidate trajectory could be parsed under {series_path} or pdata/*.\n"
                f"Last error: {last_err}"
            )

        return _from_gradoutput(), "gradoutput(fallback)"

    raise ValueError(f"Unknown traj-source: {traj_source}")


# ---------------- NIfTI writers ---------------- #

def bart_cfl_to_nifti(base: Path, out_nii_gz: Path):
    arr = np.abs(readcfl(str(base)))
    nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), str(out_nii_gz))


def write_qa_nifti(qa_frames: list[Path], qa_base: Path):
    vols = []
    shape0 = None
    for p in qa_frames:
        mag = np.abs(readcfl(str(p)))
        if shape0 is None:
            shape0 = mag.shape
        elif mag.shape != shape0:
            raise ValueError(f"QA frame {p} has shape {mag.shape}, expected {shape0}")
        vols.append(mag)
    qa_stack = np.stack(vols, axis=-1)
    out = qa_base.with_suffix(".nii.gz")
    nib.save(nib.Nifti1Image(qa_stack.astype(np.float32), np.eye(4)), str(out))
    print(f"[info] Wrote QA NIfTI {out} with shape {qa_stack.shape}")


# ---------------- Core recon ---------------- #

def run_bart(
    series_path: Path,
    method: Path,
    acqp: Path,
    out_base: Path,
    NX: int, NY: int, NZ: int,
    true_ro: int,
    ksp: np.ndarray,                 # (ro, spokes, coils)
    spokes_all: int,
    spokes_per_frame: int,
    frame_shift: int,
    qa_first: int | None,
    export_nifti: bool,
    traj_scale: float | None,
    use_gpu: bool,
    spoke_order: str,
    readout_origin: str,
    reverse_readout: bool,
    traj_source: str,
    traj_file: Optional[Path],
):
    bart_bin = "bart"

    traj_full, traj_used = load_traj_auto(
        series_path=series_path,
        method=method,
        acqp=acqp,
        true_ro=true_ro,
        spokes_all=spokes_all,
        NX=NX,
        traj_scale=traj_scale,
        spoke_order=spoke_order,
        readout_origin=readout_origin,
        reverse_readout=reverse_readout,
        traj_source=traj_source,
        traj_file=traj_file,
    )
    print(f"[info] Trajectory source used: {traj_used}")

    # ---- TRAJ SANITY CHECKS ----
    # traj_full shape: (3, RO, spokes)
    k_mag = np.sqrt(
        traj_full[0].real**2 +
        traj_full[1].real**2 +
        traj_full[2].real**2
    )  # (RO, spokes)

    ro0 = np.median(k_mag[0, :])
    rom = np.median(k_mag[k_mag.shape[0] // 2, :])
    rol = np.median(k_mag[-1, :])

    # variation across spokes at the END of readout
    end_spread = np.percentile(k_mag[-1, :], [5, 50, 95])

    print(f"[debug] traj sanity |k| median at RO[0],RO[mid],RO[-1]: {ro0:.6g}, {rom:.6g}, {rol:.6g}")
    print(f"[debug] traj sanity |k| at RO[-1] p5/p50/p95: {end_spread[0]:.6g}, {end_spread[1]:.6g}, {end_spread[2]:.6g}")


    # ---- IMPORTANT: BART NUFFT trajectory scaling (global) ----
    # BART expects k in cycles/FOV units ~ [-0.5, 0.5] (or [0, 0.5] for center-out).
    # So the correct target kmax is 0.5, NOT NX/2.
    #
    # Also: do NOT double-scale. If already ~0.5, skip.
    if traj_used.startswith("trajfile"):
        tx = np.real(traj_full[0])
        ty = np.real(traj_full[1])
        tz = np.real(traj_full[2])

        k_end = np.sqrt(tx[-1, :] * tx[-1, :] + ty[-1, :] * ty[-1, :] + tz[-1, :] * tz[-1, :])
        kmax_curr = float(np.median(k_end))
        kmax_tgt = 0.5

        if kmax_curr > 0:
            # If we're already basically at the BART target, don't rescale again.
            if abs(kmax_curr - kmax_tgt) < 0.05:
                print(f"[info] trajfile auto-scale skipped (already BART-scaled): kmax_curr≈{kmax_curr:.6g}")
            else:
                scale = kmax_tgt / kmax_curr
                traj_full = (traj_full * scale).astype(np.complex64, copy=False)
                print(f"[info] trajfile auto-scale (BART units): kmax_curr≈{kmax_curr:.6g} -> kmax_tgt={kmax_tgt} (scale={scale:.6g})")
        else:
            print("[warn] trajfile auto-scale skipped: kmax_curr≈0", file=sys.stderr)

    have_gpu = False
    if use_gpu:
        have_gpu = bart_supports_gpu(bart_bin)
        if not have_gpu:
            print("[warn] BART has no GPU support; falling back to CPU.", file=sys.stderr)

    if spokes_per_frame <= 0:
        spokes_per_frame = spokes_all
    if frame_shift <= 0:
        frame_shift = spokes_per_frame

    frame_starts = list(range(0, max(1, spokes_all - spokes_per_frame + 1), frame_shift))
    print(f"[info] Sliding window: spokes-per-frame={spokes_per_frame}, frame-shift={frame_shift}")
    print(f"[info] Will reconstruct {len(frame_starts)} frame(s).")

    out_dir = out_base.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    qa_written = False
    first_dims_reported = False

    rss_mask = str(1 << 3)  # coil dim=3 => 8

    for i, start in enumerate(frame_starts):
        stop = min(spokes_all, start + spokes_per_frame)
        nsp = stop - start

        tag = f"vol{i:05d}"
        traj_base = out_dir / f"{out_base.name}_{tag}_traj"
        ksp_base  = out_dir / f"{out_base.name}_{tag}_ksp"
        coil_base = out_dir / f"{out_base.name}_{tag}_coil"
        sos_base  = out_dir / f"{out_base.name}_{tag}"

        frame_paths.append(sos_base)

        if sos_base.with_suffix(".cfl").exists():
            print(f"[info] Frame {i} already reconstructed -> {sos_base}, skipping.")
            continue

        print(f"[info] Frame {i} spokes [{start}:{stop}] (n={nsp})")

        ksp_frame = ksp[:, start:stop, :]          # (ro, spokes, coils)
        traj_frame = traj_full[:, :, start:stop]   # (3, ro, spokes)

        # IMPORTANT: BART NUFFT expects k-space dims[0] == 1
        # => k-space should be (1, RO, spokes, coils)
        ksp_bart = ksp_frame[np.newaxis, :, :, :]  # (1, ro, spokes, coils)

        writecfl(str(traj_base), traj_frame)
        writecfl(str(ksp_base), ksp_bart)

        cmd = [bart_bin, "nufft", "-i", "-d", f"{NX}:{NY}:{NZ}"]
        if use_gpu and have_gpu:
            cmd.insert(2, "-g")
        cmd += [str(traj_base), str(ksp_base), str(coil_base)]
        print("[bart]", " ".join(cmd))
        subprocess.run(cmd, check=True)

        cmd2 = [bart_bin, "rss", rss_mask, str(coil_base), str(sos_base)]
        print("[bart]", " ".join(cmd2))
        subprocess.run(cmd2, check=True)

        if not first_dims_reported:
            dims_traj = bart_image_dims(bart_bin, traj_base)
            dims_ksp  = bart_image_dims(bart_bin, ksp_base)
            dims_coil = bart_image_dims(bart_bin, coil_base)
            dims_sos  = bart_image_dims(bart_bin, sos_base)
            if dims_traj is not None:
                print(f"[debug] BART dims traj vol0: {dims_traj}")
            if dims_ksp is not None:
                print(f"[debug] BART dims ksp  vol0: {dims_ksp}")
            if dims_coil is not None:
                print(f"[debug] BART dims coil vol0: {dims_coil}")
            if dims_sos is not None:
                print(f"[debug] BART dims SoS  vol0: {dims_sos}")
            first_dims_reported = True

        if qa_first is not None and not qa_written and len(frame_paths) >= qa_first:
            qa_base = out_dir / f"{out_base.name}_QA_first{qa_first}"
            write_qa_nifti(frame_paths[:qa_first], qa_base)
            qa_written = True

    sos_existing = [p for p in frame_paths if p.with_suffix(".cfl").exists()]
    if not sos_existing:
        print("[warn] No SoS frames exist; skipping 4D stack.", file=sys.stderr)
        return

    stack_base = out_dir / out_base.name
    join_cmd = ["bart", "join", "3"] + [str(p) for p in sos_existing] + [str(stack_base)]
    print("[bart]", " ".join(join_cmd))
    subprocess.run(join_cmd, check=True)

    if export_nifti:
        out_nii = stack_base.with_suffix(".nii.gz")
        bart_cfl_to_nifti(stack_base, out_nii)
        print(f"[info] Wrote final 4D NIfTI {out_nii}")

    print(f"[info] Done. 4D result base: {stack_base}")


# ---------------- CLI ---------------- #

def main():
    ap = argparse.ArgumentParser(
        description="Bruker 3D radial → BART NUFFT recon driver (grad.output or trajfile).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:

              Use trajectory file (Bruker writes <series>/traj):
                python bruker_radial_bart.py --series /path/to/8 --traj-source trajfile --fid-dtype i2 --export-nifti --out /tmp/out

              Use grad.output:
                python bruker_radial_bart.py --series /path/to/29 --traj-source gradoutput --fid-dtype i4 --export-nifti --out /tmp/out
            """
        ),
    )

    ap.add_argument("--series", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--matrix", nargs=3, type=int, metavar=("NX", "NY", "NZ"))
    ap.add_argument("--readout", type=int)
    ap.add_argument("--coils", type=int)

    ap.add_argument("--spokes-per-frame", type=int, default=0)
    ap.add_argument("--frame-shift", type=int, default=0)

    ap.add_argument("--fid-dtype", choices=["i2", "i4", "f4"], default="i2")
    ap.add_argument("--fid-endian", choices=[">", "<"], default="<")
    ap.add_argument("--fid-layout", choices=["ro_spokes_coils", "ro_coils_spokes"], default="ro_spokes_coils")

    ap.add_argument("--spoke-order", choices=["tile", "repeat"], default="tile")
    ap.add_argument("--readout-origin", choices=["centered", "zero"], default="centered")
    ap.add_argument("--reverse-readout", action="store_true")
    ap.add_argument("--traj-scale", type=float, default=None)

    ap.add_argument("--traj-source", choices=["auto", "trajfile", "gradoutput"], default="auto")
    ap.add_argument("--traj-file", default=None, help="Explicit traj file path (defaults to <series>/traj or pdata/**/traj)")

    ap.add_argument("--qa-first", type=int, default=0)
    ap.add_argument("--export-nifti", action="store_true")

    ap.add_argument("--gpu", action="store_true")

    args = ap.parse_args()

    series_path = Path(args.series).resolve()
    out_base = Path(args.out).resolve()
    out_base.parent.mkdir(parents=True, exist_ok=True)

    method = series_path / "method"
    acqp = series_path / "acqp"
    fid = series_path / "fid"
    if not (method.exists() and acqp.exists() and fid.exists()):
        ap.error(f"Could not find method/acqp/fid under {series_path}")

    if args.matrix is not None:
        NX, NY, NZ = map(int, args.matrix)
        print(f"[info] Matrix overridden from CLI: {NX}x{NY}x{NZ}")
    else:
        NX, NY, NZ = infer_matrix(method)
        print(f"[info] Matrix inferred from PVM_Matrix: {NX}x{NY}x{NZ}")

    if args.readout is not None:
        true_ro = int(args.readout)
        print(f"[info] Readout overridden from CLI: RO={true_ro}")
    else:
        true_ro = infer_true_ro(acqp)
        print(f"[info] Readout (true RO) from ACQ_size: RO={true_ro}")

    if args.coils is not None:
        coils_hint = int(args.coils)
        print(f"[info] Coils overridden from CLI: {coils_hint}")
    else:
        coils_hint = infer_coils(method)
        print(f"[info] Coils inferred from PVM_EncNReceivers: {coils_hint}")

    ksp, stored_ro, spokes_all, coils = load_bruker_kspace(
        fid_path=fid,
        true_ro=true_ro,
        coils_hint=coils_hint,
        endian=args.fid_endian,
        base_kind=args.fid_dtype,
        fid_layout=args.fid_layout,
    )

    spf = spokes_all if args.spokes_per_frame <= 0 else args.spokes_per_frame
    shift = spf if args.frame_shift <= 0 else args.frame_shift
    qa_first = args.qa_first if args.qa_first > 0 else None

    traj_file = Path(args.traj_file).resolve() if args.traj_file else None

    run_bart(
        series_path=series_path,
        method=method,
        acqp=acqp,
        out_base=out_base,
        NX=NX, NY=NY, NZ=NZ,
        true_ro=true_ro,
        ksp=ksp,
        spokes_all=spokes_all,
        spokes_per_frame=spf,
        frame_shift=shift,
        qa_first=qa_first,
        export_nifti=args.export_nifti,
        traj_scale=args.traj_scale,
        use_gpu=args.gpu,
        spoke_order=args.spoke_order,
        readout_origin=args.readout_origin,
        reverse_readout=args.reverse_readout,
        traj_source=args.traj_source,
        traj_file=traj_file,
    )


if __name__ == "__main__":
    main()
