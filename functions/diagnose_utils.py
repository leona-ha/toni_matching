import re
import ast
import numpy as np
import pandas as pd

# ICD-10-ish pattern: letter + 2 digits + optional ".<1-3 digits>"
ICD_RE = re.compile(r"[A-Z][0-9]{2}(?:\.[0-9]{1,3})?")

# -------------------------
# 1) Parsing + Normalizing
# -------------------------
def parse_icd_codes(x) -> list[str]:
    """Return a list of ICD-like codes from x, robust to malformed strings."""
    if x is None or (isinstance(x, float) and pd.isna(x)) or x is pd.NA:
        return []

    if isinstance(x, list):
        out = []
        for c in x:
            s = str(c).strip().upper()
            if s and s not in {"NONE", "NAN", "NULL", "<NA>"}:
                out.append(s)
        return out

    if isinstance(x, str):
        s = x.strip()
        if s in ("", "[]", ","):
            return []
        if s.upper() in {"NONE", "NAN", "NULL", "<NA>"}:
            return []

        # Try literal_eval first (works for valid "['F32.2', 'F41.0']" strings)
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list):
                out = []
                for c in obj:
                    cc = str(c).strip().upper()
                    if cc and cc not in {"NONE", "NAN", "NULL", "<NA>"}:
                        out.append(cc)
                return out
        except (ValueError, SyntaxError):
            pass

        # Fallback: regex-extract codes from the string (handles broken quotes)
        return ICD_RE.findall(s.upper())

    return []


def norm_icd(code, level=3) -> str | None:
    """
    level=3 -> 'F32.2' -> 'F32'
    level='full' -> keep full 'F32.2'
    Returns None for invalid/missing.
    """
    if code is None or (isinstance(code, float) and pd.isna(code)) or code is pd.NA:
        return None

    c = str(code).strip().upper()
    if c in {"", "NONE", "NAN", "NULL", "<NA>"}:
        return None

    m = re.match(r"^([A-Z])([0-9]{2})(?:\.(\d+))?$", c)
    if not m:
        return None

    letter, d2, dec = m.group(1), m.group(2), m.group(3)
    return f"{letter}{d2}" if level == 3 else c


# ------------------------------------------
# 2) Mapping to Table 2 diagnostic categories
# ------------------------------------------
TABLE2_CATS = {
    "substance":  "Mental and behavioral disorders due to substance use",
    "psychotic":  "Schizophrenia, schizotypal and delusional disorders",
    "affective":  "Affective disorders",
    "anxiety":    "Anxiety disorders",
    "ocd":        "Obsessive–compulsive disorder",
    "asr_ptsd":   "Acute stress reaction and posttraumatic stress disorder",
    "adjustment": "Adjustment disorder",
    "somatoform": "Somatoform disorders",
    "eating":     "Eating disorders",
    "personality":"Personality disorders",
    "adhd":       "Hyperkinetic and attention-deficit disorders",
    "other":      "Other disorders",
    "missing":    "Missing diagnosis",
}


def map_primary_to_table2(dx_full: str, *, f43_fallback: str = "other", missing_label: str = "Missing") -> str:
    """
    Map a (full) primary ICD code (e.g., 'F43.1') to the Table 2 categories.
    Returns a STRING category (CatBoost-safe; never np.nan/pd.NA).
    """
    if dx_full is None:
        return TABLE2_CATS["missing"]

    s = str(dx_full).strip().upper()
    if s in {"", "<NA>", "NAN", "NONE", "NULL", missing_label.upper()}:
        return TABLE2_CATS["missing"]

    dx3 = norm_icd(s, level=3)  # e.g., F43
    if dx3 is None:
        return TABLE2_CATS["other"]

    # Substance use: F10–F19
    if dx3.startswith("F"):
        try:
            fnum = int(dx3[1:3])
        except ValueError:
            fnum = None

        if fnum is not None and 10 <= fnum <= 19:
            return TABLE2_CATS["substance"]
        if fnum is not None and 20 <= fnum <= 29:
            return TABLE2_CATS["psychotic"]
        if fnum is not None and 30 <= fnum <= 39:
            return TABLE2_CATS["affective"]

    # Anxiety + OCD
    if dx3 in {"F40", "F41"}:
        return TABLE2_CATS["anxiety"]
    if dx3 == "F42":
        return TABLE2_CATS["ocd"]

    # Stress/adjustment (needs subcode)
    if dx3 == "F43":
        if re.match(r"^F43\.0(\D|$)|^F43\.1(\D|$)", s):
            return TABLE2_CATS["asr_ptsd"]
        if re.match(r"^F43\.2(\D|$)", s):
            return TABLE2_CATS["adjustment"]
        return TABLE2_CATS.get(f43_fallback, TABLE2_CATS["other"])

    # Somatoform, eating, personality, ADHD
    if dx3 == "F45":
        return TABLE2_CATS["somatoform"]
    if dx3 == "F50":
        return TABLE2_CATS["eating"]
    if dx3 in {"F60", "F61", "F62"}:
        return TABLE2_CATS["personality"]
    if dx3 == "F90":
        return TABLE2_CATS["adhd"]

    return TABLE2_CATS["other"]


def first_code_with_letter(lst, *, letter: str = "F", missing_label: str = "Missing") -> str:
    """
    First ICD code in lst that starts with `letter` (keeps full code).
    CatBoost-safe: returns `missing_label` (string) instead of np.nan.
    """
    if not isinstance(lst, list) or len(lst) == 0:
        return missing_label
    for c in lst:
        c_full = norm_icd(c, level="full")
        if isinstance(c_full, str) and c_full.startswith(letter):
            return c_full
    return missing_label


# -------------------------
# 3) Multi-hot helper
# -------------------------
def _multihot_from_lists(
    df: pd.DataFrame,
    list_col: str,
    universe: list[str] | None,
    prefix: str,
    dtype=np.int8,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create multi-hot columns from a column of lists of codes.
    Guaranteed numeric 0/1 with NO NaNs.
    """
    exploded = df[list_col].explode()
    exploded = exploded[exploded.notna()]

    if universe is None:
        universe = sorted({c for c in exploded.unique().tolist() if isinstance(c, str)})

    # no codes at all: still create columns if universe provided
    if len(exploded) == 0:
        if universe:
            zero = pd.DataFrame(
                0, index=df.index,
                columns=[f"{prefix}_{c}" for c in universe],
                dtype=dtype
            )
            df = df.join(zero)
        return df, universe

    mat = pd.crosstab(exploded.index, exploded)
    mat = mat.reindex(columns=universe, fill_value=0)
    mat = (mat > 0).astype(dtype)

    mat.columns = [f"{prefix}_{c}" for c in mat.columns]

    # align to df rows (missing rows => 0)
    mat = mat.reindex(df.index, fill_value=0).astype(dtype)

    df = df.join(mat)

    # final safety
    df[mat.columns] = df[mat.columns].fillna(0).astype(dtype)
    return df, universe


# -------------------------
# 4) Main feature builder
# -------------------------
def add_dx_features(
    df: pd.DataFrame,
    *,
    primary_col: str = "primarydiagnosis_tp",
    comorbid_col: str = "comorbiddiagnosis_tp",
    level_multihot: int | str = 3,                 # usually 3-digit (F32)
    allowed_letters: tuple[str, ...] | None = ("F",),
    primary_prefix: str = "pdx3",
    comorbid_prefix: str = "cdx3",
    f43_fallback: str = "other",
    universe_primary: list[str] | None = None,     # pass for CV hygiene
    universe_comorbid: list[str] | None = None,
    keep_lists: bool = True,
    missing_label: str = "Missing",
    multihot_dtype=np.int8,                         # numeric 0/1
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Outputs types:
      - primary_dx_icd10_3, primary_dx_table2: categorical STRINGS (CatBoost-safe, no NaN)
      - multi-hot pdx*/cdx*: numeric 0/1 (dtype multihot_dtype; no NaN)
      - counts: Python ints (no NaN)
    """
    df = df.copy()

    # parse lists
    df["primary_list"] = df[primary_col].apply(parse_icd_codes)
    if comorbid_col in df.columns:
        df["comorbid_list"] = df[comorbid_col].apply(parse_icd_codes)
    else:
        df["comorbid_list"] = [[] for _ in range(len(df))]

    # ---- Raw ICD code variables as categorical STRINGS (CatBoost-safe) ----
    df["primary_dx_icd10"] = df["primary_list"].apply(
        lambda lst: first_code_with_letter(lst, letter="F", missing_label=missing_label)
    ).astype(str)

    df["primary_dx_icd10_3"] = df["primary_dx_icd10"].apply(
        lambda x: norm_icd(x, level=3) if isinstance(x, str) and x != missing_label else missing_label
    ).astype(str)

    df["primary_dx_table2"] = df["primary_dx_icd10"].apply(
        lambda x: map_primary_to_table2(x, f43_fallback=f43_fallback, missing_label=missing_label)
    ).astype(str)

    # ---- normalize + filter lists (for counts + multi-hot universes) ----
    def _norm_filter(lst):
        out = []
        for c in (lst or []):
            cc = norm_icd(c, level=level_multihot)
            if cc is None:
                continue
            if allowed_letters is not None and cc[0] not in allowed_letters:
                continue
            out.append(cc)
        return out

    primary_norm = df["primary_list"].apply(_norm_filter)
    comorbid_norm = df["comorbid_list"].apply(_norm_filter)

    # counts as INTEGERS
    df["primary_count"] = primary_norm.apply(len).astype(int)
    df["primary_count_unique"] = primary_norm.apply(lambda lst: len(set(lst))).astype(int)

    df["n_comorbid"] = comorbid_norm.apply(len).astype(int)
    df["n_comorbid_unique"] = comorbid_norm.apply(lambda lst: len(set(lst))).astype(int)

    # ---- multi-hot (numeric 0/1) ----
    df["_primary_norm_set"] = primary_norm.apply(lambda lst: sorted(set(lst)))
    df["_comorbid_norm_set"] = comorbid_norm.apply(lambda lst: sorted(set(lst)))

    df, universe_primary = _multihot_from_lists(
        df, "_primary_norm_set", universe_primary, primary_prefix, dtype=multihot_dtype
    )
    df, universe_comorbid = _multihot_from_lists(
        df, "_comorbid_norm_set", universe_comorbid, comorbid_prefix, dtype=multihot_dtype
    )

    # cleanup
    df.drop(columns=["_primary_norm_set", "_comorbid_norm_set"], inplace=True)

    if not keep_lists:
        df.drop(columns=["primary_list", "comorbid_list"], inplace=True)

    # Final safety: enforce 0/1 dtype on multihot columns (no NaN)
    mh_cols = [c for c in df.columns if c.startswith((primary_prefix + "_", comorbid_prefix + "_"))]
    if mh_cols:
        df[mh_cols] = df[mh_cols].fillna(0).astype(multihot_dtype)

    return df, universe_primary, universe_comorbid
