#!/usr/bin/env python3

import os
import re
import glob
import argparse
import pandas as pd

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN


# -----------------------------
# Argument parsing
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--excel", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--template", default=None)
args = parser.parse_args()


# -----------------------------
# Load Excel lookup table
# -----------------------------
df = pd.read_excel(args.excel)

if "Arunno_or_Crunno" not in df.columns:
    raise RuntimeError("Column 'Arunno_or_Crunno' not found in Excel file.")
if "Image_QA" not in df.columns:
    raise RuntimeError("Column 'Image_QA' not found in Excel file.")

df["Arunno_or_Crunno"] = df["Arunno_or_Crunno"].astype(str)


def lookup_status(runno):
    matches = df[df["Arunno_or_Crunno"] == runno]
    if len(matches) == 0:
        return "Maybe"

    qa = str(matches.iloc[0]["Image_QA"]).strip().upper()
    if qa.startswith("U"):
        return "Yes"
    elif qa.startswith("N"):
        return "No"
    else:
        return "Maybe"


def parse_genotype_method(runno):
    prefix = runno[0].upper()
    if prefix == "A":
        return "APOE", "IV"
    elif prefix == "P":
        return "APOE", "CM"
    elif prefix == "C":
        return "CVN", "IV"
    elif prefix == "V":
        return "CVN", "CM"
    else:
        return "Unknown", "Unknown"


# -----------------------------
# Presentation setup
# -----------------------------
if args.template:
    prs = Presentation(args.template)
else:
    prs = Presentation()

blank_layout = prs.slide_layouts[6]  # blank


# -----------------------------
# Helper functions
# -----------------------------
def add_image(slide, pattern, left, top, width, height):
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"Missing image: {pattern}")
        return
    slide.shapes.add_picture(
        matches[0],
        Inches(left),
        Inches(top),
        width=Inches(width),
        height=Inches(height),
    )


def add_textbox(
    slide,
    text,
    left,
    top,
    width,
    height,
    font_name,
    font_size,
    color,
    bold=False,
):
    box = slide.shapes.add_textbox(
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    p.font.name = font_name
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color


# -----------------------------
# Iterate over folders
# -----------------------------
for folder in sorted(os.listdir(args.root)):
    if not folder.startswith("z"):
        continue

    runno = folder[1:]
    fullpath = os.path.join(args.root, folder)

    pngs = glob.glob(os.path.join(fullpath, "*.png"))
    if not pngs:
        continue

    print(f"Adding slide for {runno}")

    genotype, method = parse_genotype_method(runno)
    status = lookup_status(runno)

    slide = prs.slides.add_slide(blank_layout)

    # -------- Images --------
    add_image(
        slide,
        os.path.join(fullpath, f"{runno}_CM_roi_intensity_xyz_*_r2.5.png"),
        0.1,
        0,
        4.25,
        3.19,
    )

    add_image(
        slide,
        os.path.join(fullpath, f"{runno}_CSF_roi_intensity_xyz_*_r2.5.png"),
        4.55,
        0,
        4.25,
        3.19,
    )

    add_image(
        slide,
        os.path.join(fullpath, f"{runno}_Hc_roi_intensity_xyz_*_r2.5.png"),
        9,
        0,
        4.25,
        3.19,
    )

    add_image(
        slide,
        os.path.join(fullpath, f"{runno}_CM.png"),
        0.1,
        3.16,
        4.25,
        4.25,
    )

    add_image(
        slide,
        os.path.join(fullpath, f"{runno}_CSF.png"),
        4.55,
        3.16,
        4.25,
        4.25,
    )

    add_image(
        slide,
        os.path.join(fullpath, f"{runno}_Hc.png"),
        9,
        3.16,
        4.25,
        4.25,
    )

    # -------- Bottom Labels (White) --------
    add_textbox(
        slide,
        "Cisterna Magna",
        2.6,
        7,
        1.41,
        0.3,
        "Aptos (Body)",
        12,
        RGBColor(255, 255, 255),
    )

    add_textbox(
        slide,
        "Cerebral Spinal Fluid",
        6.95,
        7,
        1.97,
        0.3,
        "Aptos (Body)",
        12,
        RGBColor(255, 255, 255),
    )

    add_textbox(
        slide,
        "Hippocampus",
        11.7,
        7,
        1.41,
        0.3,
        "Aptos (Body)",
        12,
        RGBColor(255, 255, 255),
    )

    # -------- Genotype / Method / Status --------
    add_textbox(
        slide,
        f"Genotype: {genotype}",
        2.55,
        2.4,
        1.83,
        0.37,
        "Aptos Display",
        16,
        RGBColor(0, 0, 0),
    )

    add_textbox(
        slide,
        f"Method: {method}",
        7.5,
        2.4,
        1.83,
        0.37,
        "Aptos Display",
        16,
        RGBColor(0, 0, 0),
    )

    add_textbox(
        slide,
        f"Usable: {status}",
        11.65,
        2.4,
        1.83,
        0.37,
        "Aptos Display",
        16,
        RGBColor(0, 0, 0),
    )

    # -------- NEW TITLE BOX ($runno) --------
    add_textbox(
        slide,
        runno,
        2.55,
        0.17,
        1.5,
        0.4,
        "Aptos Display",
        16,
        RGBColor(0, 0, 0),
        bold=True,
    )

# -----------------------------
# Save
# -----------------------------
prs.save(args.out)
print(f"\nSaved presentation to {args.out}")