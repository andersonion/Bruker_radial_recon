#!/usr/bin/env python3

import os
import glob
import argparse
import pandas as pd

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing z<runno> subfolders")
    ap.add_argument("--excel", required=True, help="Excel file path")
    ap.add_argument("--out", required=True, help="Output PPTX filename")
    ap.add_argument("--template", default=None, help="Optional PPTX template to inherit theme/layout/size from")
    ap.add_argument("--slide_width", type=float, default=13.333, help='Slide width in inches (default: 13.333 for 16:9)')
    ap.add_argument("--slide_height", type=float, default=7.5, help='Slide height in inches (default: 7.5 for 16:9)')
    return ap.parse_args()


def load_lookup_table(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)

    required = ["Arunno_or_Crunno", "Image_QA"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in Excel: {missing}")

    # normalize to string for exact matching
    df["Arunno_or_Crunno"] = df["Arunno_or_Crunno"].astype(str)
    df["Image_QA"] = df["Image_QA"].astype(str)

    return df


def lookup_status(df: pd.DataFrame, runno: str) -> str:
    matches = df[df["Arunno_or_Crunno"] == runno]
    if len(matches) == 0:
        return "Maybe"

    qa = str(matches.iloc[0]["Image_QA"]).strip().upper()
    if qa.startswith("U"):
        return "Yes"
    if qa.startswith("N"):
        return "No"
    return "Maybe"


def parse_genotype_method(runno: str):
    prefix = runno[0].upper()
    if prefix == "A":
        return "APOE", "IV"
    if prefix == "P":
        return "APOE", "CM"
    if prefix == "C":
        return "CVN", "IV"
    if prefix == "V":
        return "CVN", "CM"
    return "Unknown", "Unknown"


def force_white_background(slide):
    # Force solid white background regardless of template theme
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(255, 255, 255)


def first_match(pattern: str):
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def add_image(slide, pattern, left, top, width, height):
    path = first_match(pattern)
    if not path:
        print(f"Missing image: {pattern}")
        return False
    slide.shapes.add_picture(
        path,
        Inches(left),
        Inches(top),
        width=Inches(width),
        height=Inches(height),
    )
    return True


def add_textbox(slide, text, left, top, width, height, font_name, font_size, color_rgb, bold=False):
    box = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    p.font.name = font_name
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color_rgb
    return box


def main():
    args = parse_args()

    df = load_lookup_table(args.excel)

    if args.template:
        prs = Presentation(args.template)
        # If user provides template, we assume its size is correct and DO NOT override.
        # (But if they still want override, they can omit --template or explicitly set below.)
    else:
        prs = Presentation()
        # Force slide size (python-pptx default is often 4:3, which breaks your coordinates)
        prs.slide_width = Inches(args.slide_width)
        prs.slide_height = Inches(args.slide_height)

    blank_layout = prs.slide_layouts[6]  # blank

    # Iterate z<runno> folders
    root = args.root
    for folder in sorted(os.listdir(root)):
        if not folder.startswith("z"):
            continue

        runno = folder[1:]
        fullpath = os.path.join(root, folder)
        if not os.path.isdir(fullpath):
            continue

        # include any folder with any png
        if not glob.glob(os.path.join(fullpath, "*.png")):
            continue

        genotype, method = parse_genotype_method(runno)
        status = lookup_status(df, runno)

        print(f"Adding slide for {runno}")

        slide = prs.slides.add_slide(blank_layout)
        force_white_background(slide)

        # -------- Images --------
        add_image(slide, os.path.join(fullpath, f"{runno}_CM_roi_intensity_xyz_*_r2.5.png"), 0.1, 0.0, 4.25, 3.19)
        add_image(slide, os.path.join(fullpath, f"{runno}_CSF_roi_intensity_xyz_*_r2.5.png"), 4.55, 0.0, 4.25, 3.19)
        add_image(slide, os.path.join(fullpath, f"{runno}_Hc_roi_intensity_xyz_*_r2.5.png"), 9.0, 0.0, 4.25, 3.19)

        add_image(slide, os.path.join(fullpath, f"{runno}_CM.png"), 0.1, 3.16, 4.25, 4.25)
        add_image(slide, os.path.join(fullpath, f"{runno}_CSF.png"), 4.55, 3.16, 4.25, 4.25)
        add_image(slide, os.path.join(fullpath, f"{runno}_Hc.png"), 9.0, 3.16, 4.25, 4.25)

        # -------- Bottom Labels (Set 1) --------
        add_textbox(slide, "Cisterna Magna", 2.6, 7.0, 1.41, 0.3, "Aptos (Body)", 12, RGBColor(255, 255, 255))
        add_textbox(slide, "Cerebral Spinal Fluid", 6.95, 7.0, 1.97, 0.3, "Aptos (Body)", 12, RGBColor(255, 255, 255))
        add_textbox(slide, "Cisterna Magna", 11.7, 7.0, 1.41, 0.3, "Aptos (Body)", 12, RGBColor(255, 255, 255))

        # -------- Genotype/Method/Usable (Set 2) --------
        add_textbox(slide, f"Genotype: {genotype}", 2.55, 2.4, 1.83, 0.37, "Aptos Display", 16, RGBColor(0, 0, 0))
        add_textbox(slide, f"Method: {method}", 7.5, 2.4, 1.83, 0.37, "Aptos Display", 16, RGBColor(0, 0, 0))
        add_textbox(slide, f"Usable: {status}", 11.65, 2.4, 1.83, 0.37, "Aptos Display", 16, RGBColor(0, 0, 0))

        # -------- Title box ($runno) --------
        add_textbox(slide, runno, 2.55, 0.17, 1.5, 0.4, "Aptos Display", 16, RGBColor(0, 0, 0), bold=True)

    prs.save(args.out)
    print(f"\nSaved presentation to {args.out}")


if __name__ == "__main__":
    main()