# -*- coding: utf-8 -*-
"""
Create APA-7 style patient-features table from patient_data.csv.

Requirements:
  pip install pandas numpy python-docx

Run:
  python make_patient_table_apa7.py
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


# -----------------------------
# Helpers: formatting + borders
# -----------------------------
def fmt_mean_sd(x: pd.Series) -> str:
    x = pd.to_numeric(x, errors="coerce")
    m = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=1)
    if pd.isna(m):
        return ""
    return f"{m:.2f} ({sd:.2f})"


def fmt_n_pct(n: int, denom: int) -> str:
    if denom <= 0:
        return f"{n} (0.0%)"
    pct = (n / denom) * 100.0
    return f"{n} ({pct:.1f}%)"


def missing_n_pct(x: pd.Series, N: int) -> str:
    miss = int(pd.isna(x).sum())
    return fmt_n_pct(miss, N)


def _set_cell_text(cell, text: str, bold: bool = False, italic: bool = False,
                   align: str = "left", left_indent_in: float | None = None):
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic

    if align == "right":
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    elif align == "center":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT

    if left_indent_in is not None:
        p.paragraph_format.left_indent = Inches(left_indent_in)


def _set_cell_border(cell, **kwargs):
    """
    Set cell borders. kwargs: top, bottom, left, right each a dict:
      {"sz": 8, "val": "single", "color": "000000"}
    """
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn("w:tcBorders"))
    if tcBorders is None:
        tcBorders = OxmlElement("w:tcBorders")
        tcPr.append(tcBorders)

    for edge in ("top", "left", "bottom", "right"):
        edge_data = kwargs.get(edge)
        tag = qn(f"w:{edge}")
        element = tcBorders.find(tag)
        if edge_data is None:
            # remove border
            if element is not None:
                tcBorders.remove(element)
            continue
        if element is None:
            element = OxmlElement(f"w:{edge}")
            tcBorders.append(element)
        for key, val in edge_data.items():
            element.set(qn(f"w:{key}"), str(val))


def apply_apa_table_rules(table):
    """
    APA-style rules:
      - No vertical lines
      - Only: top border of table, bottom border of header row, bottom border of table
    """
    n_rows = len(table.rows)
    n_cols = len(table.columns)

    # clear all borders
    for r in range(n_rows):
        for c in range(n_cols):
            _set_cell_border(table.cell(r, c), top=None, bottom=None, left=None, right=None)

    rule = {"val": "single", "sz": "8", "color": "000000"}  # ~1pt

    # top rule (row 0)
    for c in range(n_cols):
        _set_cell_border(table.cell(0, c), top=rule)

    # header bottom rule (row 0 bottom)
    for c in range(n_cols):
        _set_cell_border(table.cell(0, c), top=rule, bottom=rule)

    # bottom rule (last row bottom)
    last = n_rows - 1
    for c in range(n_cols):
        _set_cell_border(table.cell(last, c), bottom=rule)


# -----------------------------
# Codebooks (subcategories EN)
# -----------------------------
GENDER = {
    1: "Female",
    2: "Non-binary",
    3: "Genderfluid",
    4: "Male",
    5: "Other",
    6: "Prefer not to say",
}

SCHOOL = {
    0: "No school-leaving certificate",
    1: "Lower secondary school or equivalent",
    2: "Intermediate secondary school or equivalent",
    3: "Higher education entrance qualification or equivalent",
    4: "Completed vocational training",
    5: "Other school-leaving certificate",
}

EMPLOYMENT = {
    1: "Employed",
    2: "Self-employed",
    3: "Unemployed",
    4: "Retired",
    5: "Student / in training",
    6: "Other",
}

LOCATION = {
    1: "Rural",
    2: "Medium-sized city/town",
    3: "Large city",
}

YESNO = {0: "No", 1: "Yes"}  # used only if you later decide to show both rows


# Variable labels (EN)
LABEL = {
    "t0_age_p": "Age (years)",
    "t0_gender_p": "Gender identity",
    "t0_distress_sum": "Baseline distress (sum score)",
    "t1_distress_sum": "Week 6 distress (sum score)",
    "t0_school_p": "Highest educational attainment",
    "t0_employment_p": "Employment status",
    "t0_location_p": "Residential location",
    "t0_relationship_p": "Relationship status",
    "t0_migration_p_1": "Migration background",
    "t0_ses_p": "Socioeconomic status",

}


CODEBOOK = {
    "t0_gender_p": GENDER,
    "t0_school_p": SCHOOL,
    "t0_employment_p": EMPLOYMENT,
    "t0_location_p": LOCATION,
}


# -----------------------------
# Build rows for the table
# -----------------------------
def add_continuous_rows(rows, df, col, N):
    label = LABEL.get(col, col)
    rows.append({
        "Variable": label,
        "Value": fmt_mean_sd(df[col]),
        "Missing": missing_n_pct(df[col], N),
        "is_header": False,
        "indent": 0.0,
    })


def add_categorical_rows(rows, df, col, N):
    label = LABEL.get(col, col)
    rows.append({
        "Variable": label,
        "Value": "",
        "Missing": missing_n_pct(df[col], N),
        "is_header": True,
        "indent": 0.0,
    })

    ser = pd.to_numeric(df[col], errors="coerce")
    denom = int(ser.notna().sum())
    mapper = CODEBOOK.get(col)

    # Use codebook order if available; otherwise sort unique values
    if mapper is not None:
        cat_values = [k for k in mapper.keys() if k in set(ser.dropna().astype(int).unique())]
        # also include any unexpected codes at the end
        unexpected = sorted(set(ser.dropna().astype(int).unique()) - set(mapper.keys()))
        cat_values += unexpected
    else:
        cat_values = sorted(ser.dropna().unique())

    for v in cat_values:
        n = int((ser == v).sum())
        if mapper is not None:
            cat_label = mapper.get(int(v), f"Code {int(v)}")
        else:
            cat_label = str(v)
        rows.append({
            "Variable": cat_label,
            "Value": fmt_n_pct(n, denom),
            "Missing": "",
            "is_header": False,
            "indent": 0.25,  # indent categories
        })


def add_binary_heading_yes(rows, df, col, N):
    """
    APA explicit style:
      Row 1: variable label (Missing shown here)
      Row 2: indented 'Yes' with n (%), denom = non-missing
    """
    label = LABEL.get(col, col)
    rows.append({
        "Variable": label,
        "Value": "",
        "Missing": missing_n_pct(df[col], N),
        "is_header": True,
        "indent": 0.0,
    })

    ser = pd.to_numeric(df[col], errors="coerce")
    denom = int(ser.notna().sum())
    yes_n = int((ser == 1).sum())
    rows.append({
        "Variable": "Yes",
        "Value": fmt_n_pct(yes_n, denom),
        "Missing": "",
        "is_header": False,
        "indent": 0.25,
    })


def detect_binary(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce")
    uniq = set(s.unique().tolist())
    return uniq.issubset({0, 1}) and len(uniq) > 0


def make_rows(df: pd.DataFrame) -> list[dict]:
    N = len(df)
    rows: list[dict] = []

    # Preferred order (include only those present)
    ordered = [
        "t0_age_p",
        "t0_gender_p",
        "t0_distress_sum",
        "t1_distress_sum",
        "t0_school_p",
        "t0_employment_p",
        "t0_location_p",
        "t0_relationship_p",
        "t0_migration_p_1",
        "t0_ses_p",
    ]
    cols = [c for c in ordered if c in df.columns]

    continuous = {
        "t0_age_p", "t0_distress_sum", "t1_distress_sum", "t0_ses_p",
    }
    categorical = {"t0_gender_p", "t0_school_p", "t0_employment_p", "t0_location_p"}

    for col in cols:
        if col in continuous:
            add_continuous_rows(rows, df, col, N)
        elif col in categorical:
            add_categorical_rows(rows, df, col, N)
        else:
            # default: binary if 0/1, else treat as continuous
            if detect_binary(df[col]):
                add_binary_heading_yes(rows, df, col, N)
            else:
                add_continuous_rows(rows, df, col, N)

    return rows


# -----------------------------
# Create DOCX
# -----------------------------
def create_docx_table(rows: list[dict], N: int, out_path: str):
    doc = Document()

    # Base font
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    # Table number + title (APA: Table number italic; title italic in sentence case is common)
    p_num = doc.add_paragraph()
    r1 = p_num.add_run("Table 1")
    r1.italic = True

    p_title = doc.add_paragraph()
    r2 = p_title.add_run("Patient characteristics")
    r2.italic = True

    doc.add_paragraph()  # small spacer

    # Table with 3 columns: Variable | Value | Missing
    table = doc.add_table(rows=len(rows) + 1, cols=3)
    table.autofit = True

    # Header
    _set_cell_text(table.cell(0, 0), "Variable", bold=True)
    _set_cell_text(table.cell(0, 1), "M (SD) / n (%)", bold=True, align="right")
    _set_cell_text(table.cell(0, 2), "Missing n (%)", bold=True, align="right")

    # Body
    for i, row in enumerate(rows, start=1):
        var = row["Variable"]
        val = row["Value"]
        miss = row["Missing"]
        indent = row.get("indent", 0.0)

        _set_cell_text(table.cell(i, 0), var, bold=row.get("is_header", False),
                       align="left", left_indent_in=indent)
        _set_cell_text(table.cell(i, 1), val, align="right")
        _set_cell_text(table.cell(i, 2), miss, align="right")

    # Apply APA horizontal rules
    apply_apa_table_rules(table)

    # Note
    doc.add_paragraph()
    note = doc.add_paragraph()
    note_run = note.add_run(
        f"Note. N = {N}. Continuous variables are presented as M (SD). "
        f"Categorical variables are presented as n (%). Percentages for binary variables "
        f"refer to the proportion of 'Yes' among non-missing responses."
    )
    note_run.italic = False

    doc.save(out_path)


def main():
    in_path = "patient_data.csv"  # adjust if needed
    out_path = "Table_Patient_Features_APA7_updated.docx"

    df = pd.read_csv(in_path)

    N = len(df)
    rows = make_rows(df)
    create_docx_table(rows, N, out_path)

    print(f"Wrote: {out_path} (N={N}, columns={len(df.columns)})")


if __name__ == "__main__":
    main()
