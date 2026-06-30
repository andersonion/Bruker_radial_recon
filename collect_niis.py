#!/usr/bin/env python3
"""
Collect selected Bruker-derived NIfTI+method files into standardized locations
and emit a minimal CSV aligned 1:1 with input rows.

Supports:
- Baseline
- Block1
- Block2
- T2
- CEST
- T1_FISP_stack

Special behavior:
- CEST NIfTI/method goes into all_niis/
- CEST raw data goes into raw_CEST/{runno}/
- T1_FISP_stack goes directly into all_niis/z{runno}/
- If all_niis/z{runno} exists, only process CEST and T1_FISP_stack.
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
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


LEADING_SCANNO_RE = re.compile(r"^(?P<scanno>\d{1,2})_")

EXCEL_LIKE_EXTS = {".xlsx", ".xls", ".ods"}

CSV_HEADERS = [
    "Baseline",
    "Block1",
    "Block2",
    "T2",
    "CEST",
    "T1_FISP_stack",
]


@dataclass
class PatternSpec:
    key: str
    src_glob: str
    dest_stem: str
    is_cest: bool = False
    goes_to_zrunno: bool = False


PATTERNS: List[PatternSpec] = [
    PatternSpec(
        key="Baseline",
        src_glob="*_1_UTE3D_DT_Test_UTE3D_DT_block2_baseline.nii.gz",
        dest_stem="DCE_baseline",
    ),
    PatternSpec(
        key="Block1",
        src_glob="*_1_UTE3D_DT_Test_UTE3D_DT_block1.nii.gz",
        dest_stem="DCE_block1",
    ),
    PatternSpec(
        key="Block2",
        src_glob="*_1_UTE3D_DT_Test_UTE3D_DT_block2.nii.gz",
        dest_stem="DCE_block2",
    ),
    PatternSpec(
        key="T2",
        src_glob="*1_T2_weighted_3D_TurboRare.nii.gz",
        dest_stem="T2",
    ),
    PatternSpec(
        key="CEST",
        src_glob="*_1_Jonah_CEST_48offsets.nii.gz",
        dest_stem="CEST",
        is_cest=True,
    ),
    PatternSpec(
        key="T1_FISP_stack",
        src_glob="*_*_T1_TrueFISP.nii.gz",
        dest_stem="T1_FISP_stack",
        goes_to_zrunno=True,
    ),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Collect Bruker-derived NIfTI/method files."
    )

    ap.add_argument("inventory_path", help="Input inventory (.csv, .xlsx, .xls, .ods)")

    ap.add_argument(
        "--study_dir",
        default="/mnt/newStor/paros/paros_MRI/DennisTurner",
    )

    ap.add_argument(
        "--nii_subdir",
        default="nii",
    )

    ap.add_argument(
        "--out_csv",
        default="scanno_lookup_minimal.csv",
    )

    ap.add_argument(
        "--sheet_name",
        default=None,
    )

    ap.add_argument(
        "--dry_run",
        action="store_true",
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files/directories.",
    )

    ap.add_argument(
        "--verbose",
        action="store_true",
    )

    ap.add_argument(
        "--libreoffice_cmd",
        default="libreoffice",
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


def convert_to_csv_with_libreoffice(
    input_path: str,
    output_dir: str,
    libreoffice_cmd: str,
    verbose: bool = False,
) -> str:
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
            f"LibreOffice command not found: {libreoffice_cmd}"
        ) from e

    if proc.returncode != 0:
        raise RuntimeError(
            "LibreOffice conversion failed.\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}\n"
        )

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_csv = os.path.join(output_dir, base + ".csv")

    if not os.path.isfile(out_csv):
        candidates = sorted(
            glob(os.path.join(output_dir, "*.csv")),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        if not candidates:
            raise RuntimeError(f"No CSV created in {output_dir}")
        out_csv = candidates[0]

    return out_csv


def read_inventory_to_dataframe(
    inventory_path: str,
    sheet_name: Optional[str],
    libreoffice_cmd: str,
    verbose: bool = False,
) -> pd.DataFrame:
    inv_path = os.path.abspath(inventory_path)
    ext = os.path.splitext(inv_path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(inv_path)

    if ext in EXCEL_LIKE_EXTS:
        with tempfile.TemporaryDirectory(prefix="collect_niis_lo_") as tmpdir:
            out_csv = convert_to_csv_with_libreoffice(
                input_path=inv_path,
                output_dir=tmpdir,
                libreoffice_cmd=libreoffice_cmd,
                verbose=verbose,
            )
            return pd.read_csv(out_csv)

    raise ValueError(f"Unsupported inventory extension: {ext}")


def list_data_dirs(
    study_dir: str,
    date_mmddyy: str,
    nii_subdir: str,
    bfolder: str,
) -> List[str]:
    base = os.path.join(study_dir, date_mmddyy, nii_subdir)
    pattern = os.path.join(base, f"*{bfolder}*1_1")

    return sorted(
        p for p in glob(pattern)
        if os.path.isdir(p)
    )


def extract_scanno(path: str) -> Optional[int]:
    bn = os.path.basename(path)
    m = LEADING_SCANNO_RE.match(bn)

    if not m:
        return None

    return int(m.group("scanno"))


def pick_best_candidate(files: List[str]) -> Tuple[Optional[int], Optional[str], List[int]]:
    valid = []

    for fpath in files:
        scanno = extract_scanno(fpath)
        if scanno is None:
            continue
        valid.append((scanno, fpath))

    if not valid:
        return None, None, []

    valid.sort(key=lambda x: x[0])
    all_scannos = sorted(set(v[0] for v in valid))

    return valid[0][0], valid[0][1], all_scannos


def find_method_pair(nii_path: str) -> Optional[str]:
    if not nii_path.endswith(".nii.gz"):
        return None

    method_path = nii_path[:-7] + ".method"

    if os.path.isfile(method_path):
        return method_path

    return None


def ensure_no_collision(
    dst: str,
    seen_destinations: Set[str],
    overwrite: bool,
) -> bool:
    if dst in seen_destinations:
        print(f"DUPLICATE DESTINATION DETECTED, SKIPPING: {dst}")
        return False

    seen_destinations.add(dst)

    if os.path.exists(dst) and not overwrite:
        print(f"EXISTS, SKIPPING (use --overwrite): {dst}")
        return False

    return True


def safe_copy_file(
    src: str,
    dst: str,
    dry_run: bool,
) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if dry_run:
        return

    shutil.copy2(src, dst)


def safe_copytree(
    src: str,
    dst: str,
    overwrite: bool,
    dry_run: bool,
) -> bool:
    if os.path.exists(dst):
        if not overwrite:
            print(f"RAW DEST EXISTS, SKIPPING: {dst}")
            return False

        if not dry_run:
            shutil.rmtree(dst)

    if not dry_run:
        shutil.copytree(src, dst)

    return True


def process_raw_cest(
    nii_dir: str,
    scanno: int,
    runno: str,
    study_dir: str,
    overwrite: bool,
    dry_run: bool,
    verbose: bool,
) -> None:
    raw_dir = nii_dir.replace("/nii/", "/raw/")
    scanno_dir = os.path.join(raw_dir, str(scanno))

    if not os.path.isdir(scanno_dir):
        print(f"WARNING: raw CEST dir missing: {scanno_dir}")
        return

    raw_cest_root = os.path.join(study_dir, "raw_CEST")
    runno_root = os.path.join(raw_cest_root, runno)

    if os.path.exists(runno_root) and not overwrite:
        print(f"RAW CEST RUNNO DEST EXISTS, SKIPPING RAW COPY: {runno_root}")
        return

    if os.path.exists(runno_root) and overwrite and not dry_run:
        shutil.rmtree(runno_root)

    if not dry_run:
        os.makedirs(runno_root, exist_ok=True)

    dst_scanno_dir = os.path.join(runno_root, str(scanno))

    safe_copytree(
        src=scanno_dir,
        dst=dst_scanno_dir,
        overwrite=overwrite,
        dry_run=dry_run,
    )

    for item in sorted(os.listdir(raw_dir)):
        src_item = os.path.join(raw_dir, item)

        if os.path.isdir(src_item):
            continue

        dst_item = os.path.join(runno_root, item)

        if os.path.exists(dst_item) and not overwrite:
            print(f"RAW SIBLING EXISTS, SKIPPING: {dst_item}")
            continue

        if verbose:
            print(f"        raw sibling file: {src_item}")

        safe_copy_file(
            src=src_item,
            dst=dst_item,
            dry_run=dry_run,
        )


def mark_all_missing(out_row: Dict[str, str]) -> None:
    for key in CSV_HEADERS:
        out_row[key] = "MISSING"


def destination_paths_for_spec(
    spec: PatternSpec,
    runno: str,
    all_niis_dir: str,
    dry_run: bool,
) -> Tuple[str, str]:
    if spec.goes_to_zrunno:
        dest_dir = os.path.join(all_niis_dir, f"z{runno}")

        if not dry_run:
            os.makedirs(dest_dir, exist_ok=True)

        nii_dst = os.path.join(dest_dir, f"{runno}_{spec.dest_stem}.nii.gz")
        method_dst = os.path.join(dest_dir, f"{runno}_{spec.dest_stem}.method")

        return nii_dst, method_dst

    nii_dst = os.path.join(all_niis_dir, f"{runno}_{spec.dest_stem}.nii.gz")
    method_dst = os.path.join(all_niis_dir, f"{runno}_{spec.dest_stem}.method")

    return nii_dst, method_dst


def main() -> int:
    args = parse_args()

    all_niis_dir = os.path.join(args.study_dir, "all_niis")

    if not args.dry_run:
        os.makedirs(all_niis_dir, exist_ok=True)

    df = read_inventory_to_dataframe(
        inventory_path=args.inventory_path,
        sheet_name=args.sheet_name,
        libreoffice_cmd=args.libreoffice_cmd,
        verbose=args.verbose,
    )

    required_cols = [
        "Bruker_folder",
        "Scan Date",
        "Arunno_or_Crunno",
    ]

    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    out_rows = []
    seen_destinations: Set[str] = set()

    for idx, row in df.iterrows():
        out_row = {key: "" for key in CSV_HEADERS}

        runno = row.get("Arunno_or_Crunno")

        if pd.isna(runno) or str(runno).strip() == "":
            out_rows.append(out_row)
            continue

        runno = str(runno).strip()

        bfolder = row.get("Bruker_folder")
        scan_date_raw = row.get("Scan Date")

        if pd.isna(bfolder) or str(bfolder).strip() == "":
            mark_all_missing(out_row)
            out_rows.append(out_row)
            continue

        date_mmddyy = scan_date_to_mmddyy(scan_date_raw)

        if date_mmddyy is None:
            mark_all_missing(out_row)
            out_rows.append(out_row)
            continue

        bfolder = str(bfolder).strip()

        data_dirs = list_data_dirs(
            study_dir=args.study_dir,
            date_mmddyy=date_mmddyy,
            nii_subdir=args.nii_subdir,
            bfolder=bfolder,
        )

        if not data_dirs:
            mark_all_missing(out_row)
            out_rows.append(out_row)
            continue

        zrunno_dir = os.path.join(all_niis_dir, f"z{runno}")
        zrunno_exists = os.path.isdir(zrunno_dir)

        if args.verbose:
            print(f"[row {idx}] runno={runno} date={date_mmddyy} bfolder={bfolder}")
            if zrunno_exists:
                print(f"    z{runno} exists -> only processing CEST and T1_FISP_stack")
            for dd in data_dirs:
                print(f"    data_dir: {dd}")

        for spec in PATTERNS:
            if zrunno_exists and spec.key not in {"CEST", "T1_FISP_stack"}:
                continue

            chosen_scanno = None
            chosen_rep = None
            candidate_scannos = []

            for dd in data_dirs:
                candidates = glob(os.path.join(dd, spec.src_glob))
                scanno, rep, all_scannos = pick_best_candidate(candidates)

                if scanno is not None and rep is not None:
                    chosen_scanno = scanno
                    chosen_rep = rep
                    candidate_scannos = all_scannos
                    break

            if chosen_scanno is None or chosen_rep is None:
                out_row[spec.key] = "MISSING"
                continue

            if args.verbose and len(candidate_scannos) > 1:
                print(
                    f"    {spec.key}: selected scanno "
                    f"{chosen_scanno} over candidates {candidate_scannos}"
                )

            method_path = find_method_pair(chosen_rep)

            if method_path is None:
                out_row[spec.key] = "MISSING"
                continue

            nii_dst, method_dst = destination_paths_for_spec(
                spec=spec,
                runno=runno,
                all_niis_dir=all_niis_dir,
                dry_run=args.dry_run,
            )

            nii_ok = ensure_no_collision(
                dst=nii_dst,
                seen_destinations=seen_destinations,
                overwrite=args.overwrite,
            )

            method_ok = ensure_no_collision(
                dst=method_dst,
                seen_destinations=seen_destinations,
                overwrite=args.overwrite,
            )

            if not (nii_ok and method_ok):
                out_row[spec.key] = "MISSING"
                continue

            safe_copy_file(
                src=chosen_rep,
                dst=nii_dst,
                dry_run=args.dry_run,
            )

            safe_copy_file(
                src=method_path,
                dst=method_dst,
                dry_run=args.dry_run,
            )

            out_row[spec.key] = str(chosen_scanno)

            if spec.is_cest:
                process_raw_cest(
                    nii_dir=os.path.dirname(chosen_rep),
                    scanno=chosen_scanno,
                    runno=runno,
                    study_dir=args.study_dir,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    verbose=args.verbose,
                )

            if args.verbose:
                print(f"    {spec.key}: scanno={chosen_scanno}")
                print(f"        nii.gz: {chosen_rep}")
                print(f"        method: {method_path}")

        out_rows.append(out_row)

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nWrote CSV: {args.out_csv}")

    if args.dry_run:
        print("Dry run only.")
    else:
        print(f"Files copied into: {all_niis_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())