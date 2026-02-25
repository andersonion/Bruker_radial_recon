#!/usr/bin/env python3

import os
import glob
import argparse
import pandas as pd
from datetime import datetime

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing z<runno> subfolders")
    ap.add_argument("--excel", required=True, help="Excel file path")
    ap.add_argument("--out", required=True, help="Output PPTX filename")
    ap.add_argument("--template", default=None, help="Optional PPTX template to inherit theme/layout/size from")
    ap.add_argument("--slide_width", type=float, default=13.333, help='Slide width in inches (default: 13.333 for 16:9)')
    ap.add_argument("--slide_height", type=float, default=7.5, help='Slide height in inches (default: 7.5 for 16:9)')
    return ap.parse_args()


# -----------------------------
# Excel lookups
# -----------------------------
def load_lookup_table(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path)

    required = ["Arunno_or_Crunno", "Image_QA"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in Excel: {missing}")

    # Optional (for summary)
    # We will look for Sex column if present.
    # Common variants: Sex, sex
    # We'll normalize access later.

    df["Arunno_or_Crunno"] = df["Arunno_or_Crunno"].astype(str)
    df["Image_QA"] = df["Image_QA"].astype(str)
    return df


def lookup_row(df: pd.DataFrame, runno: str):
    matches = df[df["Arunno_or_Crunno"] == runno]
    if len(matches) == 0:
        return None
    return matches.iloc[0]


def lookup_status(df: pd.DataFrame, runno: str) -> str:
    row = lookup_row(df, runno)
    if row is None:
        return "Maybe"

    qa = str(row["Image_QA"]).strip().upper()
    if qa.startswith("U"):
        return "Yes"
    if qa.startswith("N"):
        return "No"
    return "Maybe"


def lookup_sex(df: pd.DataFrame, runno: str) -> str:
    """
    Returns 'm', 'f', or '?'.
    Looks for column 'Sex' (case-insensitive).
    Accepts values like M/F, Male/Female, m/f.
    """
    row = lookup_row(df, runno)
    if row is None:
        return "?"

    sex_col = None
    for c in df.columns:
        if str(c).strip().lower() == "sex":
            sex_col = c
            break

    if sex_col is None:
        return "?"

    raw = str(row[sex_col]).strip().lower()
    if raw in ("m", "male"):
        return "m"
    if raw in ("f", "female"):
        return "f"
    return "?"


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


# -----------------------------
# PPT helpers
# -----------------------------
def force_white_background(slide):
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


# -----------------------------
# Slides: Title + Summary
# -----------------------------
def add_title_slide(prs, blank_layout):
    slide = prs.slides.add_slide(blank_layout)
    force_white_background(slide)

    # Title
    add_textbox(
        slide,
        "Brain Clearance in Mice",
        left=0.9,
        top=1.6,
        width=11.6,
        height=1.0,
        font_name="Aptos Display",
        font_size=44,
        color_rgb=RGBColor(0, 0, 0),
        bold=True,
    )

    # Subtitle
    add_textbox(
        slide,
        "Initial Quality Assurance Report",
        left=0.95,
        top=2.6,
        width=11.6,
        height=0.6,
        font_name="Aptos Display",
        font_size=24,
        color_rgb=RGBColor(0, 0, 0),
        bold=False,
    )

    # Date (month spelled out)
    date_str = datetime.now().strftime("%B %d, %Y")
    add_textbox(
        slide,
        date_str,
        left=0.95,
        top=3.35,
        width=6.5,
        height=0.45,
        font_name="Aptos (Body)",
        font_size=16,
        color_rgb=RGBColor(0, 0, 0),
        bold=False,
    )

    # Author
    add_textbox(
        slide,
        "Dr. B.J. Anderson, Ph.D., QIAL, Duke University Medical Center",
        left=0.95,
        top=4.0,
        width=12.0,
        height=0.6,
        font_name="Aptos (Body)",
        font_size=16,
        color_rgb=RGBColor(0, 0, 0),
        bold=False,
    )


def key_label_from_prefix(prefix: str):
    prefix = prefix.upper()
    if prefix == "A":
        return "APOE with IV", "A"
    if prefix == "P":
        return "APOE with CM", "P"
    if prefix == "C":
        return "CVN with IV", "C"
    if prefix == "V":
        return "CVN with CM", "V"
    return f"Unknown ({prefix})", prefix


def format_bucket_lines(bucket_counts):
    """
    bucket_counts: dict(prefix -> dict(sex->count, total->count))
    returns list[str] lines in A,P,C,V order.
    """
    order = ["A", "P", "C", "V"]
    lines = []
    for pref in order:
        label, _ = key_label_from_prefix(pref)
        info = bucket_counts.get(pref, {"m": 0, "f": 0, "?": 0, "total": 0})
        total = info.get("total", 0)
        m = info.get("m", 0)
        f = info.get("f", 0)
        q = info.get("?", 0)

        sex_bits = [f"{m} m", f"{f} f"]
        if q:
            sex_bits.append(f"{q} ?")
        sex_str = ", ".join(sex_bits)

        lines.append(f"{label}: {total} ({sex_str})")
    return lines


def add_summary_slide(prs, blank_layout, summary_counts):
    slide = prs.slides.add_slide(blank_layout)
    force_white_background(slide)

    # Title
    add_textbox(
        slide,
        "Summary",
        left=0.9,
        top=0.55,
        width=11.6,
        height=0.7,
        font_name="Aptos Display",
        font_size=40,
        color_rgb=RGBColor(0, 0, 0),
        bold=True,
    )

    # Layout regions:
    # Left half: big "Usable Data" + lines
    # Right half top: "Possibly Usable Data" + lines
    # Right half mid/bot: "Unusable Data" + lines
    left_x = 0.9
    left_w = 6.2
    right_x = 7.4
    right_w = 5.6

    # Usable (Yes) - center of attention
    add_textbox(
        slide,
        "Usable Data:",
        left=left_x,
        top=1.55,
        width=left_w,
        height=0.5,
        font_name="Aptos Display",
        font_size=26,
        color_rgb=RGBColor(0, 0, 0),
        bold=True,
    )

    yes_lines = format_bucket_lines(summary_counts["Yes"])
    add_textbox(
        slide,
        "\n".join(yes_lines),
        left=left_x,
        top=2.15,
        width=left_w,
        height=4.6,
        font_name="Aptos (Body)",
        font_size=22,
        color_rgb=RGBColor(0, 0, 0),
        bold=False,
    )

    # Maybe (Possibly usable) - right top
    add_textbox(
        slide,
        "Possibly Usable Data:",
        left=right_x,
        top=1.55,
        width=right_w,
        height=0.45,
        font_name="Aptos Display",
        font_size=20,
        color_rgb=RGBColor(0, 0, 0),
        bold=True,
    )

    maybe_lines = format_bucket_lines(summary_counts["Maybe"])
    add_textbox(
        slide,
        "\n".join(maybe_lines),
        left=right_x,
        top=2.05,
        width=right_w,
        height=2.1,
        font_name="Aptos (Body)",
        font_size=16,
        color_rgb=RGBColor(0, 0, 0),
        bold=False,
    )

    # No (Unusable) - right lower
    add_textbox(
        slide,
        "Unusable Data:",
        left=right_x,
        top=4.35,
        width=right_w,
        height=0.45,
        font_name="Aptos Display",
        font_size=20,
        color_rgb=RGBColor(0, 0, 0),
        bold=True,
    )

    no_lines = format_bucket_lines(summary_counts["No"])
    add_textbox(
        slide,
        "\n".join(no_lines),
        left=right_x,
        top=4.85,
        width=right_w,
        height=1.7,
        font_name="Aptos (Body)",
        font_size=16,
        color_rgb=RGBColor(0, 0, 0),
        bold=False,
    )

    # Footnote
    add_textbox(
        slide,
        "*Data may be improved/made usable after motion corrections (coregistration and volume pruning).",
        left=0.9,
        top=7.05,
        width=12.4,
        height=0.35,
        font_name="Aptos (Body)",
        font_size=12,
        color_rgb=RGBColor(0, 0, 0),
        bold=False,
    )


def init_summary_counts():
    # structure: counts[status][prefix]["m"/"f"/"?"/"total"]
    counts = {}
    for status in ("Yes", "Maybe", "No"):
        counts[status] = {}
        for pref in ("A", "P", "C", "V"):
            counts[status][pref] = {"m": 0, "f": 0, "?": 0, "total": 0}
    return counts


def bump_counts(counts, status, prefix, sex):
    if status not in counts:
        return
    if prefix not in counts[status]:
        return
    if sex not in ("m", "f", "?"):
        sex = "?"
    counts[status][prefix]["total"] += 1
    counts[status][prefix][sex] += 1


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    df = load_lookup_table(args.excel)

    if args.template:
        prs = Presentation(args.template)
        # If you use a template, we assume its slide size is already correct and DO NOT override.
    else:
        prs = Presentation()
        # Force slide size so your absolute inch coordinates match what you expect
        prs.slide_width = Inches(args.slide_width)
        prs.slide_height = Inches(args.slide_height)

    blank_layout = prs.slide_layouts[6]  # blank

    # Title slide first
    add_title_slide(prs, blank_layout)

    # Summary accumulator
    summary_counts = init_summary_counts()

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
        sex = lookup_sex(df, runno)
        prefix = runno[0].upper()

        # Update summary counts
        bump_counts(summary_counts, status, prefix, sex)

        print(f"Adding slide for {runno} (status={status}, sex={sex})")

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
        add_textbox(slide, f"Genotype: {genotype}", 3.5, 2.85, 1.83, 0.37, "Aptos Display", 16, RGBColor(0, 0, 0))
        add_textbox(slide, f"Method: {method}", 8.1, 2.85, 1.83, 0.37, "Aptos Display", 16, RGBColor(0, 0, 0))
        add_textbox(slide, f"Usable: {status}", 11.7, 2.85, 1.83, 0.37, "Aptos Display", 16, RGBColor(0, 0, 0))

        # -------- Title box ($runno) --------
        add_textbox(slide, runno, 2.55, 0.17, 1.5, 0.4, "Aptos Display", 16, RGBColor(0, 0, 0), bold=True)

    # Summary slide last
    add_summary_slide(prs, blank_layout, summary_counts)

    prs.save(args.out)
    print(f"\nSaved presentation to {args.out}")


if __name__ == "__main__":
    main()