#!/usr/bin/env python3
"""
Collect selected Bruker-derived NIfTI+method files into a single folder and
emit a CSV with the discovered scan numbers (scannos).

Assumptions / behaviors:
- Excel row 1 is headers.
- Required columns:
    - 'Bruker_folder'   (read into bfolder)
    - 'Scan Date'       (converted to date=MMDDYY)
    - 'Arunno_or_Crunno' (rows with a value are processed)
- Data directory pattern:
    data_dir = {study_dir}/{date}/{nii_subdir}/*{bfolder}*1_1
  (where {nii_subdir} defaults to "nii")
- For each file type, we search within data_dir for candidates matching patterns below,
  but we IGNORE any files whose basename does NOT start with 1–2 digits + '_' (e.g. '7_...', '12_...').
- If multiple candidates match, we choose the one with the LOWER leading number.
- For each chosen candidate, we copy BOTH:
    - *.nii.gz
    - *.method
  into {study_dir}/all_niis (or override via CLI), renaming to:
    - {Arunno_or_Crunno}_DCE_baseline.*
    - {Arunno_or_Crunno}_DCE_block1.*
    - {Arunno_or_Crunno}_DCE_block2.*
    - {Arunno_or_Crunno}_T2.*

Outputs:
- Copies into all_niis (created if missing)
- A CSV suitable to paste back into Excel, INCLUDING skipped rows, with scannos for:
    Baseline, Block1, Block2, T2

Example:
  python collect_niis.py /path/to/sheet.xlsx --out_csv scan_lookup.csv --dry_run
"""

import argparse
import csv
import os
import re
import shutil
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Tuple

import pandas as pd


LEADING_SCANNO_RE = re.compile(r"^(?P<scanno>\d{1,2})_")  # must be 1–2 digits then underscore


@dataclass
class PatternSpec:
    key: str            # used for CSV column and status
    src_glob: str       # glob pattern within data_dir
    dest_stem: str      # destination base name (without extension)


PATTERNS: List[PatternSpec] = [
    PatternSpec(
        key="Baseline",
        src_glob="*_1_UTE3D_DT_Test_UTE3D_DT_block2_baseline.*",
        dest_stem="DCE_baseline",
    ),
    PatternSpec(
        key="Block1",
        src_glob="*_1_UTE3D_DT_Test_UTE3D_DT_block1.*",
        dest_stem="DCE_block1",
    ),
    PatternSpec(
        key="Block2",
        src_glob="*_1_UTE3D_DT_Test_UTE3D_DT_block2.*",
        dest_stem="DCE_block2",
    ),
    PatternSpec(
        key="T2",
        src_glob="*1_T2_weighted_3D_TurboRare.*",
        dest_stem="T2",
    ),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Collect Bruker-derived NIfTI/method files and emit scanno CSV."
    )
    ap.add_argument("excel_path", help="Input Excel sheet (e.g., .xlsx)")
    ap.add_argument(
        "--study_dir",
        default="/mnt/newStor/paros/paros_MRI/DennisTurner",
        help="Study directory base (default: %(default)s)",
    )
    ap.add_argument(
        "--nii_subdir",
        default="nii",
        help="Subdirectory name under each date folder (default: %(default)s)",
    )
    ap.add_argument(
        "--all_niis_dir",
        default=None,
        help="Output directory for collected files (default: {study_dir}/all_niis)",
    )
    ap.add_argument(
        "--out_csv",
        default="scanno_lookup.csv",
        help="Output CSV path (default: %(default)s)",
    )
    ap.add_argument(
        "--sheet_name",
        default=None,
        help="Excel sheet name (default: first sheet)",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="Do not copy; only report what would happen.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-row details.",
    )
    return ap.parse_args()


def scan_date_to_mmddyy(val) -> Optional[str]:
    """
    Convert an Excel date-ish value to MMDDYY.
    Returns None if parsing fails.
    """
    if pd.isna(val):
        return None
    try:
        dt = pd.to_datetime(val, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.strftime("%m%d%y")
    except Exception:
        return None


def list_data_dirs(study_dir: str, date_mmddyy: str, nii_subdir: str, bfolder: str) -> List[str]:
    """
    Return matching directories for:
      {study_dir}/{date}/{nii_subdir}/*{bfolder}*1_1
    """
    base = os.path.join(study_dir, date_mmddyy, nii_subdir)
    pattern = os.path.join(base, f"*{bfolder}*1_1")
    matches = [p for p in glob(pattern) if os.path.isdir(p)]
    return sorted(matches)


def pick_best_candidate(files: List[str]) -> Tuple[Optional[int], Optional[str]]:
    """
    From a list of file paths, keep only those with basename starting with 1–2 digits + '_'.
    Return (lowest_scanno, representative_file_path) where representative is the file
    (any extension) belonging to the lowest scanno group.
    """
    by_scanno: Dict[int, List[str]] = {}
    for f in files:
        bn = os.path.basename(f)
        m = LEADING_SCANNO_RE.match(bn)
        if not m:
            continue
        sc = int(m.group("scanno"))
        by_scanno.setdefault(sc, []).append(f)

    if not by_scanno:
        return None, None

    best_sc = min(by_scanno.keys())
    # pick a stable representative (sorted) for that scanno
    rep = sorted(by_scanno[best_sc])[0]
    return best_sc, rep


def find_pair_for_rep(rep_any_ext: str) -> Dict[str, str]:
    """
    Given a representative file path (with some extension), attempt to locate both:
      - .nii.gz
      - .method
    by using a shared stem prefix up to the first '.' in basename.

    Returns dict ext_key -> path for ext_key in {"nii.gz","method"} if found.
    """
    d = os.path.dirname(rep_any_ext)
    bn = os.path.basename(rep_any_ext)

    # Shared prefix: everything before first dot
    # e.g. "12_1_UTE3D...baseline" from "12_1_UTE3D...baseline.nii.gz"
    stem = bn.split(".", 1)[0]

    out: Dict[str, str] = {}
    nii = os.path.join(d, stem + ".nii.gz")
    method = os.path.join(d, stem + ".method")
    if os.path.isfile(nii):
        out["nii.gz"] = nii
    if os.path.isfile(method):
        out["method"] = method
    return out


def copy_with_rename(src: str, dst: str, dry_run: bool) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if dry_run:
        return
    shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()

    all_niis_dir = args.all_niis_dir or os.path.join(args.study_dir, "all_niis")
    if not args.dry_run:
        os.makedirs(all_niis_dir, exist_ok=True)

    # Read Excel
    df = pd.read_excel(args.excel_path, sheet_name=args.sheet_name)

    required_cols = ["Bruker_folder", "Scan Date", "Arunno_or_Crunno"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required column(s) in Excel: {missing}")

    results_rows = []

    for idx, row in df.iterrows():
        runno = row.get("Arunno_or_Crunno")
        bfolder = row.get("Bruker_folder")
        scan_date_raw = row.get("Scan Date")
        date_mmddyy = scan_date_to_mmddyy(scan_date_raw)

        out_row = {
            "row_index_0based": idx,
            "Arunno_or_Crunno": "" if pd.isna(runno) else str(runno),
            "Bruker_folder": "" if pd.isna(bfolder) else str(bfolder),
            "Scan Date": "" if pd.isna(scan_date_raw) else str(scan_date_raw),
            "date_MMDDYY": "" if date_mmddyy is None else date_mmddyy,
            "Baseline_scanno": "",
            "Block1_scanno": "",
            "Block2_scanno": "",
            "T2_scanno": "",
            "status": "",
        }

        # Always include skipped rows for clean reinsertion into Excel
        if pd.isna(runno) or str(runno).strip() == "":
            out_row["status"] = "SKIP: Arunno_or_Crunno empty"
            results_rows.append(out_row)
            continue

        if pd.isna(bfolder) or str(bfolder).strip() == "":
            out_row["status"] = "SKIP: Bruker_folder empty"
            results_rows.append(out_row)
            continue

        if date_mmddyy is None:
            out_row["status"] = "SKIP: Scan Date parse failed"
            results_rows.append(out_row)
            continue

        runno = str(runno).strip()
        bfolder = str(bfolder).strip()

        # Find data_dir (may match multiple; we try them in order until we find hits)
        data_dirs = list_data_dirs(args.study_dir, date_mmddyy, args.nii_subdir, bfolder)
        if not data_dirs:
            out_row["status"] = "MISS: no matching data_dir"
            results_rows.append(out_row)
            continue

        if args.verbose:
            print(f"[row {idx}] runno={runno} date={date_mmddyy} bfolder={bfolder}")
            for dd in data_dirs:
                print(f"  data_dir: {dd}")

        status_bits = []

        # For each pattern, search across candidate data_dirs; take the first data_dir that yields a match.
        for spec in PATTERNS:
            chosen_scanno: Optional[int] = None
            chosen_rep: Optional[str] = None
            chosen_dir: Optional[str] = None

            for dd in data_dirs:
                cand = glob(os.path.join(dd, spec.src_glob))
                sc, rep = pick_best_candidate(cand)
                if sc is not None and rep is not None:
                    chosen_scanno, chosen_rep, chosen_dir = sc, rep, dd
                    break

            col_name = f"{spec.key}_scanno"
            if chosen_scanno is None or chosen_rep is None:
                out_row[col_name] = ""
                status_bits.append(f"{spec.key}:MISS")
                continue

            out_row[col_name] = str(chosen_scanno)

            # Find both .nii.gz and .method for the chosen scan
            pair = find_pair_for_rep(chosen_rep)
            missing_exts = [e for e in ("nii.gz", "method") if e not in pair]

            # Copy any that exist; still record missing ext(s) in status
            for ext_key, src_path in pair.items():
                # destination naming
                if ext_key == "nii.gz":
                    dst_name = f"{runno}_{spec.dest_stem}.nii.gz"
                elif ext_key == "method":
                    dst_name = f"{runno}_{spec.dest_stem}.method"
                else:
                    # shouldn't happen
                    continue
                dst_path = os.path.join(all_niis_dir, dst_name)
                copy_with_rename(src_path, dst_path, args.dry_run)

            if missing_exts:
                status_bits.append(f"{spec.key}:OK_missing_{'+'.join(missing_exts)}")
            else:
                status_bits.append(f"{spec.key}:OK")

            if args.verbose:
                print(f"  {spec.key}: scanno={chosen_scanno} dir={chosen_dir}")
                for ext_key in ("nii.gz", "method"):
                    print(f"    {ext_key}: {pair.get(ext_key, 'MISSING')}")

        out_row["status"] = "; ".join(status_bits)
        results_rows.append(out_row)

    # Write CSV
    fieldnames = list(results_rows[0].keys()) if results_rows else [
        "row_index_0based",
        "Arunno_or_Crunno",
        "Bruker_folder",
        "Scan Date",
        "date_MMDDYY",
        "Baseline_scanno",
        "Block1_scanno",
        "Block2_scanno",
        "T2_scanno",
        "status",
    ]

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results_rows)

    if args.verbose or True:
        print(f"\nWrote CSV: {args.out_csv}")
        if args.dry_run:
            print("Dry run: no files copied.")
        else:
            print(f"Files copied into: {all_niis_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())