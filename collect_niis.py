#!/usr/bin/env python3
"""
Collect selected Bruker-derived NIfTI+method files into a single folder and
emit a CSV with discovered scan numbers (scannos).

Key features:
- Reads an inventory file with headers in row 1.
- Accepts CSV directly.
- If given Excel-like input (.xlsx/.xls/.ods), auto-converts to CSV using LibreOffice
  (headless) and reads that CSV, avoiding pandas/openpyxl dependency issues.

Required columns:
    - 'Bruker_folder'
    - 'Scan Date'
    - 'Arunno_or_Crunno'

Data dir pattern:
    data_dir = {study_dir}/{date}/{nii_subdir}/*{bfolder}*1_1

For each row where 'Arunno_or_Crunno' has a value, we search data_dir and copy:
    *_1_UTE3D_DT_Test_UTE3D_DT_block2_baseline.* -> {runno}_DCE_baseline.{nii.gz|method}
    *_1_UTE3D_DT_Test_UTE3D_DT_block1.*          -> {runno}_DCE_block1.{nii.gz|method}
    *_1_UTE3D_DT_Test_UTE3D_DT_block2.*          -> {runno}_DCE_block2.{nii.gz|method}
    *1_T2_weighted_3D_TurboRare.*                -> {runno}_T2.{nii.gz|method}

Rules:
- Ignore any candidate files whose basename does not begin with 1–2 digits + '_' (e.g. '7_...')
- If multiple matches exist, choose the one with the LOWER leading number (scanno).
- Copy both .nii.gz and .method (if present); note missing in status.

Output:
- Copies into {study_dir}/all_niis (or override)
- Writes an output CSV with scannos for Baseline/Block1/Block2/T2, including skipped rows
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional, Tuple

import pandas as pd


LEADING_SCANNO_RE = re.compile(r"^(?P<scanno>\d{1,2})_")


@dataclass
class PatternSpec:
    key: str
    src_glob: str
    dest_stem: str


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


EXCEL_LIKE_EXTS = {".xlsx", ".xls", ".ods"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Collect Bruker-derived NIfTI/method files and emit scanno CSV."
    )
    ap.add_argument("inventory_path", help="Input inventory (.csv, .xlsx, .xls, .ods)")
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
        help="Excel sheet name (default: first sheet). Ignored if CSV.",
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
    ap.add_argument(
        "--libreoffice_cmd",
        default="libreoffice",
        help="LibreOffice command (default: %(default)s). Use 'soffice' if needed.",
    )
    return ap.parse_args()


def scan_date_to_mmddyy(val) -> Optional[str]:
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
    base = os.path.join(study_dir, date_mmddyy, nii_subdir)
    pattern = os.path.join(base, f"*{bfolder}*1_1")
    matches = [p for p in glob(pattern) if os.path.isdir(p)]
    return sorted(matches)


def pick_best_candidate(files: List[str]) -> Tuple[Optional[int], Optional[str]]:
    by_scanno: Dict[int, List[str]] = {}
    for fpath in files:
        bn = os.path.basename(fpath)
        m = LEADING_SCANNO_RE.match(bn)
        if not m:
            continue
        sc = int(m.group("scanno"))
        by_scanno.setdefault(sc, []).append(fpath)

    if not by_scanno:
        return None, None

    best_sc = min(by_scanno.keys())
    rep = sorted(by_scanno[best_sc])[0]
    return best_sc, rep


def find_pair_for_rep(rep_any_ext: str) -> Dict[str, str]:
    d = os.path.dirname(rep_any_ext)
    bn = os.path.basename(rep_any_ext)
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


def convert_to_csv_with_libreoffice(
    input_path: str,
    output_dir: str,
    libreoffice_cmd: str,
    verbose: bool = False,
) -> str:
    """
    Convert an Excel-like file to CSV using LibreOffice headless mode.
    Returns the path to the generated CSV.

    Note: LibreOffice will place the output CSV in output_dir. The output file name
    is generally the input basename with .csv extension.
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        libreoffice_cmd,
        "--headless",
        "--convert-to",
        "csv",
        "--outdir",
        output_dir,
        input_path,
    ]

    if verbose:
        print("Running:", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"LibreOffice command not found: '{libreoffice_cmd}'. "
            f"Try --libreoffice_cmd soffice, or ensure libreoffice is in PATH."
        ) from e

    if proc.returncode != 0:
        raise RuntimeError(
            "LibreOffice conversion failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_csv = os.path.join(output_dir, base + ".csv")

    if not os.path.isfile(out_csv):
        # LibreOffice sometimes emits slightly different naming; search output_dir for newest csv
        candidates = sorted(
            glob(os.path.join(output_dir, "*.csv")),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(
                f"LibreOffice reported success but no CSV found in: {output_dir}\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}\n"
            )
        out_csv = candidates[0]

    return out_csv


def read_inventory_to_dataframe(
    inventory_path: str,
    sheet_name: Optional[str],
    libreoffice_cmd: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    If input is CSV: read directly.
    If input is Excel-like: auto-convert to CSV (LibreOffice) then read CSV.

    Returns pandas DataFrame.
    """
    inv_path = os.path.abspath(inventory_path)
    ext = os.path.splitext(inv_path)[1].lower()

    if ext == ".csv":
        if verbose:
            print(f"Reading CSV inventory: {inv_path}")
        return pd.read_csv(inv_path)

    if ext in EXCEL_LIKE_EXTS:
        with tempfile.TemporaryDirectory(prefix="collect_niis_lo_") as tmpdir:
            out_csv = convert_to_csv_with_libreoffice(
                input_path=inv_path,
                output_dir=tmpdir,
                libreoffice_cmd=libreoffice_cmd,
                verbose=verbose,
            )
            if verbose:
                print(f"Converted inventory to CSV: {out_csv}")
            df = pd.read_csv(out_csv)
            return df

    # If you want to support TSV or other delimited text, add here.
    raise ValueError(
        f"Unsupported inventory extension '{ext}'. Use .csv, .xlsx, .xls, or .ods."
    )


def main() -> int:
    args = parse_args()

    all_niis_dir = args.all_niis_dir or os.path.join(args.study_dir, "all_niis")
    if not args.dry_run:
        os.makedirs(all_niis_dir, exist_ok=True)

    df = read_inventory_to_dataframe(
        inventory_path=args.inventory_path,
        sheet_name=args.sheet_name,
        libreoffice_cmd=args.libreoffice_cmd,
        verbose=args.verbose,
    )

    required_cols = ["Bruker_folder", "Scan Date", "Arunno_or_Crunno"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required column(s) in inventory: {missing}")

    results_rows: List[Dict[str, str]] = []

    for idx, row in df.iterrows():
        runno = row.get("Arunno_or_Crunno")
        bfolder = row.get("Bruker_folder")
        scan_date_raw = row.get("Scan Date")
        date_mmddyy = scan_date_to_mmddyy(scan_date_raw)

        out_row: Dict[str, str] = {
            "row_index_0based": str(idx),
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

        data_dirs = list_data_dirs(args.study_dir, date_mmddyy, args.nii_subdir, bfolder)
        if not data_dirs:
            out_row["status"] = "MISS: no matching data_dir"
            results_rows.append(out_row)
            continue

        if args.verbose:
            print(f"[row {idx}] runno={runno} date={date_mmddyy} bfolder={bfolder}")
            for dd in data_dirs:
                print(f"    data_dir: {dd}")

        status_bits: List[str] = []

        for spec in PATTERNS:
            chosen_scanno: Optional[int] = None
            chosen_rep: Optional[str] = None

            for dd in data_dirs:
                candidates = glob(os.path.join(dd, spec.src_glob))
                sc, rep = pick_best_candidate(candidates)
                if sc is not None and rep is not None:
                    chosen_scanno, chosen_rep = sc, rep
                    break

            col_name = f"{spec.key}_scanno"
            if chosen_scanno is None or chosen_rep is None:
                out_row[col_name] = ""
                status_bits.append(f"{spec.key}:MISS")
                continue

            out_row[col_name] = str(chosen_scanno)

            pair = find_pair_for_rep(chosen_rep)
            missing_exts = [e for e in ("nii.gz", "method") if e not in pair]

            for ext_key, src_path in pair.items():
                if ext_key == "nii.gz":
                    dst_name = f"{runno}_{spec.dest_stem}.nii.gz"
                elif ext_key == "method":
                    dst_name = f"{runno}_{spec.dest_stem}.method"
                else:
                    continue

                dst_path = os.path.join(all_niis_dir, dst_name)
                copy_with_rename(src_path, dst_path, args.dry_run)

            if missing_exts:
                status_bits.append(f"{spec.key}:OK_missing_{'+'.join(missing_exts)}")
            else:
                status_bits.append(f"{spec.key}:OK")

            if args.verbose:
                print(f"    {spec.key}: scanno={chosen_scanno}")
                print(f"        nii.gz: {pair.get('nii.gz', 'MISSING')}")
                print(f"        method: {pair.get('method', 'MISSING')}")

        out_row["status"] = "; ".join(status_bits)
        results_rows.append(out_row)

    fieldnames = [
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

    print(f"\nWrote CSV: {args.out_csv}")
    if args.dry_run:
        print("Dry run: no files copied.")
    else:
        print(f"Files copied into: {all_niis_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())