#!/usr/bin/env python3

import os
import glob
import argparse
import pandas as pd
from datetime import datetime

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR


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
    Accepts M/F, Male/Female, etc.
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

def add_textbox(
    slide,
    text,
    left,
    top,
    width,
    height,
    font_name,
    font_size,
    color_rgb,
    bold=False,
    align="left",      # left|center|right
    valign="top",      # top|middle|bottom
):
    box = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = box.text_frame
    tf.clear()

    if valign == "middle":
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    elif valign == "bottom":
        tf.vertical_anchor = MSO_ANCHOR.BOTTOM
    else:
        tf.vertical_anchor = MSO_ANCHOR.TOP

    p = tf.paragraphs[0]
    p.text = text

    if align == "center":
        p.alignment = PP_ALIGN.CENTER
    elif align == "right":
        p.alignment = PP_ALIGN.RIGHT
    else:
        p.alignment = PP_ALIGN.LEFT

    p.font.name = font_name
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color_rgb
    return box

# -----------------------------
# Title slide
# -----------------------------
def add_title_slide(prs, blank_layout):
    slide = prs.slides.add_slide(blank_layout)
    force_white_background(slide)

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


# -----------------------------
# Summary counts + rubric helpers
# -----------------------------
def init_summary_counts():
    # counts[status][genotype][method] -> dict(sex->count, total->count)
    counts = {}
    for status in ("Yes", "Maybe", "No"):
        counts[status] = {}
        for genotype in ("APOE", "CVN"):
            counts[status][genotype] = {}
            for method in ("IV", "CM"):
                counts[status][genotype][method] = {"m": 0, "f": 0, "?": 0, "total": 0}
    return counts


def bump_counts(counts, status, genotype, method, sex):
    if status not in counts:
        return
    if genotype not in counts[status]:
        return
    if method not in counts[status][genotype]:
        return
    if sex not in ("m", "f", "?"):
        sex = "?"
    cell = counts[status][genotype][method]
    cell["total"] += 1
    cell[sex] += 1

def add_cell(slide, left, top, width, height, border_width_pt=1.5):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(left), Inches(top), Inches(width), Inches(height),
    )
    shape.fill.background()  # no fill
    shape.line.color.rgb = RGBColor(0, 0, 0)
    shape.line.width = Pt(border_width_pt)
    return shape


def fmt_cell(cell: dict) -> str:
    total = cell.get("total", 0)
    m = cell.get("m", 0)
    f = cell.get("f", 0)
    q = cell.get("?", 0)
    bits = [f"{m} m", f"{f} f"]
    if q:
        bits.append(f"{q} ?")
    return f"{total} ({', '.join(bits)})"

def add_rubric(slide, title, counts_block, left, top, width, height, big=False):
    # ---- Layout knobs ----
    width_scale = 0.75  # 75% width
    v_shift_frac = 1.0 / 3.0  # shift block down by 1/3 of remaining height

    # Font sizes
    count_fs = 20 if big else 14
    label_fs = count_fs + 2
    title_fs = 26 if big else 18

    # Compact geometry
    title_h = 0.38
    gap = 0.08
    header_h = 0.40 if big else 0.32
    row_h = 0.85 if big else 0.55

    grid_h = header_h + 2 * row_h
    block_h = title_h + gap + grid_h  # title + grid treated as one "block"

    # ---- Apply 1/3-down vertical placement (NOT centered) ----
    extra_h = max(0.0, height - block_h)
    block_top = top + v_shift_frac * extra_h

    # ---- Apply 75% width, centered within allotted width ----
    rub_w = width * width_scale
    rub_left = left + (width - rub_w) / 2.0

    # --- Title (moves with rubric) ---
    add_textbox(
        slide, title,
        rub_left, block_top, rub_w, title_h,
        "Aptos Display", title_fs, RGBColor(0, 0, 0),
        bold=True, align="left", valign="middle",
    )

    grid_top = block_top + title_h + gap

    label_col_w = 0.85 if big else 0.70
    data_col_w = (rub_w - label_col_w) / 2.0
    bw = 2.0 if big else 1.25

    # --- Borders (3 cols x 3 rows) ---
    # Header
    add_cell(slide, rub_left, grid_top, label_col_w, header_h, bw)
    add_cell(slide, rub_left + label_col_w, grid_top, data_col_w, header_h, bw)
    add_cell(slide, rub_left + label_col_w + data_col_w, grid_top, data_col_w, header_h, bw)

    # IV row
    add_cell(slide, rub_left, grid_top + header_h, label_col_w, row_h, bw)
    add_cell(slide, rub_left + label_col_w, grid_top + header_h, data_col_w, row_h, bw)
    add_cell(slide, rub_left + label_col_w + data_col_w, grid_top + header_h, data_col_w, row_h, bw)

    # CM row
    add_cell(slide, rub_left, grid_top + header_h + row_h, label_col_w, row_h, bw)
    add_cell(slide, rub_left + label_col_w, grid_top + header_h + row_h, data_col_w, row_h, bw)
    add_cell(slide, rub_left + label_col_w + data_col_w, grid_top + header_h + row_h, data_col_w, row_h, bw)

    # --- Labels ---
    add_textbox(slide, "", rub_left, grid_top, label_col_w, header_h,
                "Aptos Display", label_fs, RGBColor(0, 0, 0),
                True, align="center", valign="middle")
    add_textbox(slide, "APOE", rub_left + label_col_w, grid_top, data_col_w, header_h,
                "Aptos Display", label_fs, RGBColor(0, 0, 0),
                True, align="center", valign="middle")
    add_textbox(slide, "CVN", rub_left + label_col_w + data_col_w, grid_top, data_col_w, header_h,
                "Aptos Display", label_fs, RGBColor(0, 0, 0),
                True, align="center", valign="middle")

    add_textbox(slide, "IV", rub_left, grid_top + header_h, label_col_w, row_h,
                "Aptos Display", label_fs, RGBColor(0, 0, 0),
                True, align="center", valign="middle")
    add_textbox(slide, "CM", rub_left, grid_top + header_h + row_h, label_col_w, row_h,
                "Aptos Display", label_fs, RGBColor(0, 0, 0),
                True, align="center", valign="middle")

    # --- Data ---
    add_textbox(slide, fmt_cell(counts_block["APOE"]["IV"]),
                rub_left + label_col_w, grid_top + header_h, data_col_w, row_h,
                "Aptos (Body)", count_fs, RGBColor(0, 0, 0),
                False, align="center", valign="middle")

    add_textbox(slide, fmt_cell(counts_block["CVN"]["IV"]),
                rub_left + label_col_w + data_col_w, grid_top + header_h, data_col_w, row_h,
                "Aptos (Body)", count_fs, RGBColor(0, 0, 0),
                False, align="center", valign="middle")

    add_textbox(slide, fmt_cell(counts_block["APOE"]["CM"]),
                rub_left + label_col_w, grid_top + header_h + row_h, data_col_w, row_h,
                "Aptos (Body)", count_fs, RGBColor(0, 0, 0),
                False, align="center", valign="middle")

    add_textbox(slide, fmt_cell(counts_block["CVN"]["CM"]),
                rub_left + label_col_w + data_col_w, grid_top + header_h + row_h, data_col_w, row_h,
                "Aptos (Body)", count_fs, RGBColor(0, 0, 0),
                False, align="center", valign="middle")


def add_summary_slide(prs, blank_layout, summary_counts):
    slide = prs.slides.add_slide(blank_layout)
    force_white_background(slide)

    add_textbox(slide, "QA Summary", 0.9, 0.5, 11.6, 0.7,
                "Aptos Display", 40, RGBColor(0,0,0), True)

    left_x, left_w = 0.9, 6.3
    right_x, right_w = 7.4, 5.5

    add_rubric(slide, "Usable Data", summary_counts["Yes"],
               left=left_x, top=1.45, width=left_w, height=5.8, big=True)

    add_rubric(slide, "Possibly Usable Data", summary_counts["Maybe"],
               left=right_x, top=1.45, width=right_w, height=2.8, big=False)

    add_rubric(slide, "Unusable Data", summary_counts["No"],
               left=right_x, top=4.35, width=right_w, height=2.9, big=False)

    add_textbox(
        slide,
        "*Data may be improved/made usable after motion corrections (coregistration and volume pruning).",
        0.9, 7.18, 12.4, 0.35,
        "Aptos (Body)", 12, RGBColor(0,0,0), False
    )

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    df = load_lookup_table(args.excel)

    # Pass 1: scan folders, collect run list + summary counts
    run_folders = []
    summary_counts = init_summary_counts()

    for folder in sorted(os.listdir(args.root)):
        if not folder.startswith("z"):
            continue
        runno = folder[1:]
        fullpath = os.path.join(args.root, folder)
        if not os.path.isdir(fullpath):
            continue
        if not glob.glob(os.path.join(fullpath, "*.png")):
            continue

        status = lookup_status(df, runno)
        sex = lookup_sex(df, runno)
        genotype, method = parse_genotype_method(runno)

        bump_counts(summary_counts, status, genotype, method, sex)
        run_folders.append((runno, fullpath))

    # Pass 2: build PPT in the required order
    if args.template:
        prs = Presentation(args.template)
        # template controls size/theme
    else:
        prs = Presentation()
        prs.slide_width = Inches(args.slide_width)
        prs.slide_height = Inches(args.slide_height)

    blank_layout = prs.slide_layouts[6]

    # Title then Summary immediately after
    add_title_slide(prs, blank_layout)
    add_summary_slide(prs, blank_layout, summary_counts)

    # Then the per-run slides
    for runno, fullpath in run_folders:
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

        # -------- Genotype/Method/Usable (Set 2) [YOUR UPDATED POSITIONS] --------
        add_textbox(slide, f"Genotype: {genotype}", 3.5, 2.85, 1.83, 0.37, "Aptos Display", 16, RGBColor(0, 0, 0))
        add_textbox(slide, f"Method: {method}", 8.1, 2.85, 1.83, 0.37, "Aptos Display", 16, RGBColor(0, 0, 0))
        add_textbox(slide, f"Usable: {status}", 11.7, 2.85, 1.83, 0.37, "Aptos Display", 16, RGBColor(0, 0, 0))

        # -------- Title box ($runno) --------
        add_textbox(slide, runno, 0.1, 2.85, 1.5, 0.4, "Aptos Display", 16, RGBColor(0, 0, 0), bold=True)

    prs.save(args.out)
    print(f"\nSaved presentation to {args.out}")


if __name__ == "__main__":
    main()