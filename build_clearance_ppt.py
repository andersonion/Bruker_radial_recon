#!/usr/bin/env python3
"""
Build a PowerPoint where each slide corresponds to a subfolder named: z<runno>

For each folder that contains ANY .png files, we create a slide and (when present)
place up to 6 images + 6 text boxes according to the template you specified.

Usage:
  python build_clearance_ppt.py \
    --root "/Volumes/newJetStor/newJetStor/paros/paros_MRI/DennisTurner/all_niis" \
    --excel "/Users/rja20/Documents/MATLAB/Bruker_radial_recon/Animals_Clearance_Turner_complete_list.xlsx" \
    --out "clearance_report.pptx"

Notes:
- Requires: python-pptx, pandas, openpyxl
- Paths are local to your machine (Mac paths in your example).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor


# ----------------------------
# Template constants (inches)
# ----------------------------

SLIDE_W_IN = 13.333  # widescreen
SLIDE_H_IN = 7.5

IMAGES = [
    # (glob_pattern, left, top, width, height)
    ("{runno}_CM_roi_intensity_xyz_*_r2.5.png", 0.1, 0.0, 4.25, 3.19),
    ("{runno}_CSF_roi_intensity_xyz_*_r2.5.png", 4.55, 0.0, 4.25, 3.19),
    ("{runno}_Hc_roi_intensity_xyz_*_r2.5.png", 9.0, 0.0, 4.25, 3.19),
    ("{runno}_CM.png", 0.1, 3.16, 4.25, 4.25),
    ("{runno}_CSF.png", 4.55, 3.16, 4.25, 4.25),
    ("{runno}_Hc.png", 9.0, 3.16, 4.25, 4.25),
]

# Text boxes set 1 (white)
TXT1_FONT = ("Aptos (Body)", 12, RGBColor(255, 255, 255))
TXT1_BOX_H = 0.3
TXT1_BOX_W = 1.41

# Text labels (set 1)
# (text, left, top, width_override_or_None)
LABELS_SET1 = [
    ("Cisterna Magna", 2.6, 7.0, None),
    ("Cerebral Spinal Fluid", 6.95, 7.0, 1.97),  # width exception
    ("Cisterna Magna", 11.7, 7.0, None),
]

# Text boxes set 2 (black)
TXT2_FONT = ("Aptos Display (Headings)", 16, RGBColor(0, 0, 0))
TXT2_BOX_H = 0.37
TXT2_BOX_W = 1.83

# (format_string, left, top)
LABELS_SET2 = [
    ("Genotype: {genotype}", 2.55, 2.4),
    ("Method: {method}", 7.5, 2.4),
    ("Usable: {status}", 11.65, 2.4),
]


def find_first(pattern: str, folder: Path) -> Optional[Path]:
    """Return first matching file (sorted) or None."""
    matches = sorted(folder.glob(pattern))
    return matches[0] if matches else None


def genotype_method_from_runno(runno: str) -> Tuple[str, str]:
    prefix = runno[:1].upper()
    if prefix == "A":
        return "APOE", "IV"
    if prefix == "P":
        return "APOE", "CM"
    if prefix == "C":
        return "CVN", "IV"
    if prefix == "V":
        return "CVN", "CM"
    return "Unknown", "Unknown"


def build_status_map(excel_path: Path) -> Dict[str, str]:
    """
    Map Arunno_or_Crunno -> first character of Image_QA (uppercased).
    """
    df = pd.read_excel(excel_path, engine="openpyxl")
    # normalize columns
    cols = {c.strip(): c for c in df.columns}
    run_col = cols.get("Arunno_or_Crunno")
    qa_col = cols.get("Image_QA")
    if run_col is None or qa_col is None:
        raise ValueError(
            "Excel is missing required columns. "
            "Expected headers: 'Arunno_or_Crunno' and 'Image_QA'. "
            f"Found: {list(df.columns)}"
        )

    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        rn = row[run_col]
        qa = row[qa_col]
        if pd.isna(rn):
            continue
        rn_str = str(rn).strip()
        qa_str = "" if pd.isna(qa) else str(qa).strip()
        mapping[rn_str] = qa_str[:1].upper() if qa_str else ""
    return mapping


def status_from_map(runno: str, status_map: Dict[str, str]) -> str:
    ch = status_map.get(runno, "")
    ch = (ch or "").strip()[:1].upper()
    if ch == "U":
        return "Yes"
    if ch == "N":
        return "No"
    return "Maybe"


def add_textbox(slide, text: str, left: float, top: float, width: float, height: float,
                font_name: str, font_size_pt: int, font_rgb: RGBColor):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    font = run.font
    font.name = font_name
    font.size = Pt(font_size_pt)
    font.color.rgb = font_rgb
    return box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing subfolders z<runno>")
    ap.add_argument("--excel", required=True, help="Path to Animals_Clearance_Turner_complete_list.xlsx")
    ap.add_argument("--out", default="clearance_report.pptx", help="Output PPTX path")
    ap.add_argument("--template", default=None, help="Optional .pptx to use as a starting template")
    ap.add_argument("--include-missing", action="store_true",
                    help="Still make a slide even if some of the 6 images are missing (default: yes).")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    excel = Path(args.excel).expanduser()
    out = Path(args.out).expanduser()

    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")
    if not excel.exists():
        raise FileNotFoundError(f"Excel file not found: {excel}")

    status_map = build_status_map(excel)

    prs = Presentation(args.template) if args.template else Presentation()
    if not args.template:
        prs.slide_width = Inches(SLIDE_W_IN)
        prs.slide_height = Inches(SLIDE_H_IN)

    blank_layout = prs.slide_layouts[6]

    # Collect subfolders z<runno> that contain ANY png
    subfolders = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("z")])

    made = 0
    warnings: List[str] = []

    for folder in subfolders:
        if not any(folder.glob("*.png")):
            continue

        runno = folder.name[1:]  # strip leading 'z'
        genotype, method = genotype_method_from_runno(runno)
        status = status_from_map(runno, status_map)

        slide = prs.slides.add_slide(blank_layout)
        # black background (so white labels are visible)
        if not args.template:
            bg = slide.background
            bg.fill.solid()
            bg.fill.fore_color.rgb = RGBColor(0, 0, 0)

        # Place images
        for pattern_tmpl, left, top, w, h in IMAGES:
            pattern = pattern_tmpl.format(runno=runno)
            img_path = find_first(pattern, folder)
            if img_path is None:
                warnings.append(f"[{runno}] Missing image: {pattern} (in {folder})")
                continue
            slide.shapes.add_picture(str(img_path), Inches(left), Inches(top), width=Inches(w), height=Inches(h))

        # Text labels set 1
        for txt, left, top, w_override in LABELS_SET1:
            w = w_override if w_override is not None else TXT1_BOX_W
            add_textbox(
                slide, txt, left, top, w, TXT1_BOX_H,
                font_name=TXT1_FONT[0], font_size_pt=TXT1_FONT[1], font_rgb=TXT1_FONT[2]
            )

        # Text labels set 2 (genotype/method/status)
        for fmt, left, top in LABELS_SET2:
            txt = fmt.format(genotype=genotype, method=method, status=status)
            add_textbox(
                slide, txt, left, top, TXT2_BOX_W, TXT2_BOX_H,
                font_name=TXT2_FONT[0], font_size_pt=TXT2_FONT[1], font_rgb=TXT2_FONT[2]
            )

        made += 1

    prs.save(str(out))

    print(f"✅ Wrote {out} with {made} slide(s).")
    if warnings:
        print("\nWarnings (missing images):")
        for w in warnings:
            print(" - " + w)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
