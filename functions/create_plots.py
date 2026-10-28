# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

from scipy.stats import gaussian_kde
from PIL import Image


# ============================================================
# Distribution plot (unchanged)
# ============================================================
def _freedman_diaconis_bins(x: np.ndarray, max_bins: int = 80) -> int:
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 10
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    if iqr <= 1e-12:
        return min(30, max(5, int(np.sqrt(n))))
    bin_width = 2 * iqr * (n ** (-1 / 3))
    if bin_width <= 1e-12:
        return min(30, max(5, int(np.sqrt(n))))
    bins = int(np.ceil((x.max() - x.min()) / bin_width))
    return int(np.clip(bins, 5, max_bins))


def _kde(y: np.ndarray, xs: np.ndarray) -> np.ndarray | None:
    y = y[np.isfinite(y)]
    if y.size < 2 or np.nanstd(y) <= 1e-12:
        return None
    try:
        return gaussian_kde(y)(xs)
    except Exception:
        return None


def plot_density_apa7_green_gold(
    df: pd.DataFrame,
    y_col: str,
    *,
    x_label: str | None = None,
    y_label: str | None = None,
    double_col_width_in: float = 7.09,
    height_in: float = 5.0,
    max_bins: int = 90,
    show_baseline: bool = False,
    baseline_col: str = "t0_distress_sum",
    stat: Literal["count", "density"] = "count",
    alpha_hist: float = 0.45,
    show_kde: bool = True,
    legend_items: Literal["kde_only", "hist_only", "all", "none"] = "kde_only",
):
    y = pd.to_numeric(df[y_col], errors="coerce").dropna().to_numpy(dtype=float)
    if y.size == 0:
        raise ValueError(f"No non-missing values in {y_col}")

    b = None
    if show_baseline:
        if baseline_col not in df.columns:
            raise KeyError(f"baseline_col='{baseline_col}' not found in df.")
        b = pd.to_numeric(df[baseline_col], errors="coerce").dropna().to_numpy(dtype=float)
        if b.size == 0:
            raise ValueError(f"show_baseline=True but no non-missing values in {baseline_col}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "figure.dpi": 100,
    })

    fig, ax = plt.subplots(figsize=(double_col_width_in, height_in))

    base_color = "#56B4E9"
    out_color  = "#009E73"

    all_x = np.concatenate([y, b]) if show_baseline else y
    xmin = float(np.nanmin(all_x))
    xmax = float(np.nanmax(all_x))
    if np.isclose(xmin, xmax):
        xmin -= 1.0
        xmax += 1.0

    n_bins = _freedman_diaconis_bins(all_x, max_bins=max_bins)
    edges = np.linspace(xmin, xmax, n_bins + 1)
    bin_width = edges[1] - edges[0]
    xs = np.linspace(xmin, xmax, 400)

    density_flag = (stat == "density")
    if y_label is None:
        y_label = "Density" if density_flag else "Count"

    if show_baseline:
        ax.hist(
            b, bins=edges, density=density_flag, alpha=alpha_hist,
            color=base_color, edgecolor=base_color, linewidth=1.0,
            label=("Baseline" if legend_items in ("all", "hist_only") else None),
            zorder=2,
        )

    ax.hist(
        y, bins=edges, density=density_flag, alpha=alpha_hist,
        color=out_color, edgecolor=out_color, linewidth=1.0,
        label=("Week 6" if legend_items in ("all", "hist_only") else None),
        zorder=3,
    )

    if show_kde:
        y_kde = _kde(y, xs)
        if y_kde is not None:
            if not density_flag:
                y_kde = y_kde * y.size * bin_width
            ax.plot(xs, y_kde, color=out_color, linewidth=2.2,
                    label=("Outcome KDE" if legend_items in ("all", "kde_only") else None),
                    zorder=4)
        if show_baseline:
            b_kde = _kde(b, xs)
            if b_kde is not None:
                if not density_flag:
                    b_kde = b_kde * b.size * bin_width
                ax.plot(xs, b_kde, color=base_color, linewidth=2.2,
                        label=("Baseline KDE" if legend_items in ("all", "kde_only") else None),
                        zorder=4)

    cutoff_x = 20
    ax.axvline(cutoff_x, linestyle="--", linewidth=1.5, color="black", alpha=0.9, zorder=5)

    ylim = ax.get_ylim()
    xmin_ax, xmax_ax = ax.get_xlim()
    xspan = xmax_ax - xmin_ax
    x_text = max(cutoff_x - 0.15 * xspan, xmin_ax + 0.04 * xspan)

    ax.annotate(
        "Clinical cutoff",
        xy=(cutoff_x, ylim[1] * 0.86),
        xytext=(x_text, ylim[1] * 0.93),
        ha="right", va="center", fontsize=10,
        arrowprops=dict(arrowstyle="->", lw=1.0, color="black"),
        zorder=6,
    )

    ax.set_xlabel(x_label if x_label is not None else y_col)
    ax.set_ylabel(y_label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    if legend_items != "none":
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > 0:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                      ncol=2, frameon=False, handlelength=2.5)

    fig.tight_layout()
    return fig, ax


def export_figure_apa(
    fig,
    out_base: str,
    *,
    dpi: int = 300,
    jpg_quality: int = 95,
    save_pdf: bool = True,
    save_eps: bool = True,
    save_svg: bool = True,
    save_tiff: bool = True,
    save_png: bool = True,
    save_jpg: bool = True,
) -> None:
    if save_pdf:
        fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
    if save_eps:
        fig.savefig(f"{out_base}.eps", bbox_inches="tight")
    if save_svg:
        fig.savefig(f"{out_base}.svg", bbox_inches="tight")

    if save_tiff:
        fig.savefig(f"{out_base}.tiff", dpi=dpi, bbox_inches="tight", facecolor="white")
    if save_png:
        fig.savefig(f"{out_base}.png", dpi=dpi, bbox_inches="tight", facecolor="white")

    if save_jpg:
        tmp_png = f"{out_base}__tmp.png"
        out_jpg = f"{out_base}.jpg"
        fig.savefig(tmp_png, dpi=dpi, bbox_inches="tight")

        im = Image.open(tmp_png)
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")

        im.save(out_jpg, quality=jpg_quality, dpi=(dpi, dpi))

    exts = []
    if save_pdf: exts.append("pdf")
    if save_eps: exts.append("eps")
    if save_svg: exts.append("svg")
    if save_tiff: exts.append("tiff")
    if save_png: exts.append("png")
    if save_jpg: exts.append("jpg")
    print(f"Saved: {out_base}." + "/.".join(exts) + f" (dpi={dpi}, jpg_quality={jpg_quality})")


# ============================================================
# Importance plotting utilities (FI or SHAP)
# ============================================================
def resolve_tables_dir(
    output_root: str | Path = "model_outputs",
    run_name: str | None = None,
    tables_subdir: str = "tables",
) -> Path:
    root = Path(output_root)
    if not root.exists():
        raise FileNotFoundError(f"output_root not found: {root.resolve()}")

    if run_name is None:
        run_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")]
        if not run_dirs:
            raise FileNotFoundError(f"No run_* directories found in: {root.resolve()}")
        run_dir = max(run_dirs, key=lambda p: p.stat().st_mtime)
    else:
        run_dir = root / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir not found: {run_dir.resolve()}")

    tables_dir = run_dir / tables_subdir
    if not tables_dir.exists():
        raise FileNotFoundError(f"tables directory not found: {tables_dir.resolve()}")
    return tables_dir


def _has_therapist_token(name: str) -> bool:
    return "t" in str(name).split("_")


def _is_patient_therapist_pair(i: str, j: str) -> bool:
    return _has_therapist_token(i) ^ _has_therapist_token(j)


def _strip_role_prefix(label: str) -> str:
    s = str(label)
    s = re.sub(r"^(Therapist|Patient)\s+", "", s, flags=re.IGNORECASE)
    return s


def _read_importances(path: str | Path) -> pd.DataFrame:
    """
    Accepts either:
      - CatBoost FI summary: feature, mean_importance, std_importance, runs_present, mean_importance_norm
      - SHAP main summary:   feature, mean_abs_shap, std_abs_shap, runs_present, mean_abs_shap_norm
    Normalizes to: feature, mean_importance, std_importance, runs_present, mean_importance_norm (where available)
    """
    df = pd.read_csv(path)

    value_candidates = ["mean_importance", "mean_abs_shap", "importance", "mean_value", "value"]
    std_candidates   = ["std_importance", "std_abs_shap", "std_value"]
    norm_candidates  = ["mean_importance_norm", "mean_abs_shap_norm", "mean_value_norm"]

    val_col = next((c for c in value_candidates if c in df.columns), None)
    if val_col is None:
        raise KeyError(f"Feature CSV must contain one of: {value_candidates}. Found: {list(df.columns)}")
    if val_col != "mean_importance":
        df = df.rename(columns={val_col: "mean_importance"})

    std_col = next((c for c in std_candidates if c in df.columns), None)
    if std_col and std_col != "std_importance":
        df = df.rename(columns={std_col: "std_importance"})

    norm_col = next((c for c in norm_candidates if c in df.columns), None)
    if norm_col and norm_col != "mean_importance_norm":
        df = df.rename(columns={norm_col: "mean_importance_norm"})

    for c in ["mean_importance", "std_importance", "runs_present", "mean_importance_norm"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "feature" not in df.columns or "mean_importance" not in df.columns:
        raise KeyError("Feature CSV must contain column 'feature' and a recognized importance column.")
    return df


def _read_interactions(path: str | Path) -> pd.DataFrame:
    """
    Accepts either:
      - CatBoost interaction summary: orig_i, orig_j, mean_score, std_score, runs_present
      - SHAP interaction summary:     orig_i, orig_j, mean_abs_interaction, std_abs_interaction, runs_present
    Normalizes to: orig_i, orig_j, mean_score, std_score, runs_present (where available)
    """
    df = pd.read_csv(path)

    value_candidates = ["mean_score", "mean_abs_interaction", "value", "mean_value"]
    std_candidates   = ["std_score", "std_abs_interaction", "std_value"]

    val_col = next((c for c in value_candidates if c in df.columns), None)
    if val_col is None:
        raise KeyError(f"Interaction CSV must contain one of: {value_candidates}. Found: {list(df.columns)}")
    if val_col != "mean_score":
        df = df.rename(columns={val_col: "mean_score"})

    std_col = next((c for c in std_candidates if c in df.columns), None)
    if std_col and std_col != "std_score":
        df = df.rename(columns={std_col: "std_score"})

    for c in ["mean_score", "std_score", "runs_present"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    needed = {"orig_i", "orig_j", "mean_score"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Interaction CSV missing columns: {sorted(missing)}")
    return df


def humanize_feature(raw: str, *, custom_map: Optional[Dict[str, str]] = None) -> str:
    s = str(raw)

    if custom_map and s in custom_map:
        return custom_map[s]

    if "surveys." in s:
        s = s.split(".")[-1]
    if s.startswith("t0_"):
        s = s[3:]

    sum_map = {
        "phq8_p_sum": "PHQ-8 sum",
        "gad7_p_sum": "GAD-7 sum",
        "mhse_p_sum": "MHSE sum",
        "msfq_p_sum": "MSFQ sum",
        "swls_p_sum": "SWLS sum",
        "audit_p_sum": "AUDIT sum",
        "dudit_p_sum": "DUDIT sum",
        "ace_p_sum": "ACE sum",
        "change_p_sum": "Change sum",
    }
    if s in sum_map:
        return sum_map[s]

    item_patterns = [
        (r"^phq_p_(\d+)$", "PHQ-Item {}"),
        (r"^gad_p_(\d+)$", "GAD-Item {}"),
        (r"^mhse_p_(\d+)$", "MHSE-Item {}"),
        (r"^msfq_p_(\d+)$", "MSFQ-Item {}"),
        (r"^swls_p_(\d+)$", "SWLS-Item {}"),
        (r"^audit_p_(\d+)$", "AUDIT-Item {}"),
        (r"^dudit_p_(\d+)$", "DUDIT-Item {}"),
        (r"^aqol_p_(\d+)$", "AQoL-Item {}"),
        (r"^pid_p_(\d+)$", "PID-Item {}"),
        (r"^opd_p_(\d+)$", "OPD-Item {}"),
        (r"^bfi_t_(\d+)$", "BFI-Item {}"),
    ]
    for pat, fmt in item_patterns:
        m = re.match(pat, s)
        if m:
            return fmt.format(m.group(1))

    aqol_named = {
        "aqol_p_mental_health": "AQoL MH",
        "aqol_p_coping": "AQoL coping",
        "aqol_p_pain": "AQoL pain",
        "aqol_p_happiness": "AQoL happiness",
        "aqol_p_relationship": "AQoL relationships",
        "aqol_p_independent_living": "AQoL independent living",
    }
    if s in aqol_named:
        return aqol_named[s]

    pid_named = {
        "pid_p_negative_affect": "PID negative affect",
        "pid_p_detachment": "PID detachment",
        "pid_p_antagonism": "PID antagonism",
        "pid_p_disinhibition": "PID disinhibition",
        "pid_p_anankastia": "PID anankastia",
        "pid_p_psychoticism": "PID psychoticism",
    }
    if s in pid_named:
        return pid_named[s]

    opd_named = {
        "opd_p_contact": "OPD contact",
        "opd_p_self": "OPD self",
        "opd_p_relationship": "OPD relationships",
    }
    if s in opd_named:
        return opd_named[s]

    demo_map = {
        "age_p": "Age",
        "age_t": "Age",
        "gender_p": "Gender",
        "gender_t": "Gender",
        "location_p": "Location",
        "location_t": "Location",
        "sexualorientation_t": "Sexual orientation",
        "relationship_t": "Relationship status",
        "employment_p": "Employment",
        "school_p": "Education",
        "experience_t": "Experience",
    }
    if s in demo_map:
        return demo_map[s]

    bfi_traits = {
        "bfi_t_openness": "BFI openness",
        "bfi_t_extraversion": "BFI extraversion",
        "bfi_t_agreeableness": "BFI agreeableness",
        "bfi_t_neuroticism": "BFI neuroticism",
    }
    if s in bfi_traits:
        return bfi_traits[s]

    s2 = s.replace("_", " ")
    s2 = re.sub(r"\bphq\b", "PHQ", s2, flags=re.I)
    s2 = re.sub(r"\bgad\b", "GAD", s2, flags=re.I)
    s2 = re.sub(r"\baqol\b", "AQoL", s2, flags=re.I)
    s2 = re.sub(r"\bpid\b", "PID", s2, flags=re.I)
    s2 = re.sub(r"\bopd\b", "OPD", s2, flags=re.I)
    s2 = re.sub(r"\bbfi\b", "BFI", s2, flags=re.I)
    s2 = " ".join(w.capitalize() if w.islower() else w for w in s2.split())
    return s2


def label_with_role_suffix(raw: str, *, custom_map: Optional[Dict[str, str]] = None) -> str:
    base = humanize_feature(raw, custom_map=custom_map)
    base = _strip_role_prefix(base)
    suffix = "(T)" if _has_therapist_token(raw) else "(P)"
    return f"{base} {suffix}"


def humanize_interaction(i: str, j: str, *, custom_map: Optional[Dict[str, str]] = None) -> str:
    if _has_therapist_token(i) and not _has_therapist_token(j):
        i, j = j, i
    return f"{label_with_role_suffix(i, custom_map=custom_map)} × {label_with_role_suffix(j, custom_map=custom_map)}"


# ============================================================
# UPDATED plot_catboost_importance:
#   - shows P×P interactions first (striped in patient color)
#   - then shows remaining interactions (typically P×T, optionally T×T)
# ============================================================
def plot_catboost_importance(
    *,
    output_root: str | Path = "model_outputs",
    run_name: str | None = None,
    tables_subdir: str = "tables",
    feature_filename: str = "cv_cat_importance_mean.csv",
    interaction_filename: str = "cv_cat_interactions_mean.csv",
    top_pre_therapist: int = 9,
    top_therapist_features: int = 6,
    top_interactions: int = 8,                 # after the P×P block
    top_pp_interactions: int = 0,              # how many P×P interactions shown first
    include_tt_interactions: bool = True,
    interactions_only_involving_therapist: bool = True,
    min_runs_present: int = 1,
    bar_step: float = 1.12,
    gap_size: float = 0.25,
    gap_int_blocks: float = 0.25,              # gap between P×P and rest
    show_legend: bool = False,
    left_pad: float = 0.6,
    x_label_rotation: int = 30,
    tick_fontsize: int = 10,
    custom_label_map: Optional[Dict[str, str]] = None,
    width_in: float = 11.0,
    height_in: float = 4.6,
    bar_width: float = 0.72,
    y_label_main: str = "Mean feature importance",
    y_label_int: str = "Mean interaction importance",
    title_main: str = "Main effects",
    title_int: str = "Interactions",
) -> Tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:

    tables_dir = resolve_tables_dir(output_root=output_root, run_name=run_name, tables_subdir=tables_subdir)
    feat_path = tables_dir / feature_filename
    int_path = tables_dir / interaction_filename

    df_feat = _read_importances(feat_path).copy()
    df_int = _read_interactions(int_path).copy()

    # --- main effects filtering
    df_feat["is_t"] = df_feat["feature"].apply(_has_therapist_token)
    if "runs_present" in df_feat.columns:
        df_feat = df_feat[df_feat["runs_present"].fillna(0) >= min_runs_present]

    # --- interactions filtering
    if "runs_present" in df_int.columns:
        df_int = df_int[df_int["runs_present"].fillna(0) >= min_runs_present]

    # --- tag interaction types
    df_int["has_t"] = df_int["orig_i"].apply(_has_therapist_token) | df_int["orig_j"].apply(_has_therapist_token)
    df_int["pt_pair"] = df_int.apply(lambda r: _is_patient_therapist_pair(r["orig_i"], r["orig_j"]), axis=1)
    df_int["pp_pair"] = (~df_int["has_t"])                # neither involves therapist
    df_int["tt_pair"] = df_int["has_t"] & (~df_int["pt_pair"])  # not P×T but involves therapist

    # --- main effects selection
    df_feat_sorted = df_feat.sort_values("mean_importance", ascending=False)
    pre = df_feat_sorted.head(int(top_pre_therapist)).copy()
    therapist_feats = (
        df_feat_sorted[df_feat_sorted["is_t"]]
        .loc[~df_feat_sorted["feature"].isin(pre["feature"])]
        .head(int(top_therapist_features))
        .copy()
    )
    main_rows = pd.concat([pre, therapist_feats], ignore_index=True)

    # --- interactions selection: P×P first
    df_pp = (
        df_int[df_int["pp_pair"]]
        .sort_values("mean_score", ascending=False)
        .head(int(top_pp_interactions))
        .copy()
    )

    # --- remaining interactions block
    df_rest = df_int.copy()
    if interactions_only_involving_therapist:
        df_rest = df_rest[df_rest["has_t"]].copy()

    if not include_tt_interactions:
        df_rest = df_rest[df_rest["pt_pair"]].copy()

    df_rest = df_rest[~df_rest["pp_pair"]].copy()
    df_rest = df_rest.sort_values("mean_score", ascending=False).head(int(top_interactions)).copy()

    df_int_sel = pd.concat([df_pp, df_rest], ignore_index=True)

    # --- styling
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.linewidth": 1.0,
        "figure.dpi": 100,
        "hatch.linewidth": 1.2,
    })

    base_color = "#56B4E9"  # patient
    out_color  = "#009E73"  # therapist

    fig, (ax_main, ax_int) = plt.subplots(ncols=2, figsize=(width_in, height_in))


    # =========================
    # Main effects panel
    # =========================
    x_pre = np.arange(len(pre), dtype=float) * bar_step
    start_th = (len(pre) + gap_size) * bar_step
    x_th = start_th + np.arange(len(therapist_feats), dtype=float) * bar_step
    x_main = np.concatenate([x_pre, x_th]) if len(therapist_feats) > 0 else x_pre

    y_main = main_rows["mean_importance"].to_numpy(dtype=float)
    is_t_main = main_rows["is_t"].to_numpy(dtype=bool)

    for xi, yi, is_t in zip(x_main, y_main, is_t_main):
        c = out_color if is_t else base_color
        ax_main.bar(xi, yi, width=bar_width, color=c, edgecolor=c, alpha=0.85, linewidth=1.0, zorder=3)

    labels_main = [label_with_role_suffix(f, custom_map=custom_label_map) for f in main_rows["feature"].tolist()]
    ax_main.set_xticks(x_main)
    ax_main.set_xticklabels(labels_main, rotation=x_label_rotation, ha="right", fontsize=tick_fontsize)
    ax_main.set_ylabel(y_label_main)
    ax_main.set_title(title_main)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.grid(False)
    ax_main.set_ylim(0, np.nanmax(y_main) * 1.15 if np.isfinite(np.nanmax(y_main)) else 1.0)

    # =========================
    # Interactions panel (P×P first, then rest)
    # =========================
    n_pp = len(df_pp)
    n_rest = len(df_rest)

    x_pp = np.arange(n_pp, dtype=float) * bar_step
    x_rest_start = (n_pp + gap_int_blocks) * bar_step if (n_pp > 0 and n_rest > 0) else n_pp * bar_step
    x_rest = x_rest_start + np.arange(n_rest, dtype=float) * bar_step

    x_int = np.concatenate([x_pp, x_rest]) if n_rest > 0 else x_pp
    y_int = df_int_sel["mean_score"].to_numpy(dtype=float) if len(df_int_sel) else np.array([], float)

    for idx, r in df_int_sel.reset_index(drop=True).iterrows():
        xi = x_int[idx]
        v = float(r["mean_score"])
        pp_pair = bool(r["pp_pair"])
        pt_pair = bool(r["pt_pair"])
        tt_pair = bool(r["tt_pair"])

        if pt_pair:
            # P×T: dual-striped (therapist + patient)
            ax_int.bar(xi, v, width=bar_width, color="white", edgecolor="black", linewidth=1.0, zorder=2)
            ax_int.bar(xi, v, width=bar_width, color="none", edgecolor=out_color, hatch="///", linewidth=0.0, zorder=4)
            ax_int.bar(xi, v, width=bar_width, color="none", edgecolor=base_color, hatch="\\\\\\", linewidth=0.0, zorder=5)

        elif pp_pair:
            # P×P: striped in patient colour (REQUESTED CHANGE)
            ax_int.bar(xi, v, width=bar_width, color="white", edgecolor="black", linewidth=1.0, zorder=2)
            ax_int.bar(xi, v, width=bar_width, color="none", edgecolor=base_color, hatch="\\\\\\", linewidth=0.0, zorder=4)

        elif tt_pair:
            # T×T: solid therapist colour (subtle)
            ax_int.bar(xi, v, width=bar_width, color=out_color, edgecolor=out_color, alpha=0.45, linewidth=1.0, zorder=3)

        else:
            ax_int.bar(xi, v, width=bar_width, color="grey", edgecolor="grey", alpha=0.35, linewidth=1.0, zorder=3)

    labels_int = [
        humanize_interaction(i, j, custom_map=custom_label_map)
        for i, j in zip(df_int_sel["orig_i"].tolist(), df_int_sel["orig_j"].tolist())
    ]
    ax_int.set_xticks(x_int)
    ax_int.set_xticklabels(labels_int, rotation=x_label_rotation, ha="right", fontsize=tick_fontsize)
    ax_int.set_ylabel(y_label_int)
    ax_int.set_title(title_int)
    ax_int.spines["top"].set_visible(False)
    ax_int.spines["right"].set_visible(False)
    ax_int.grid(False)
    ax_int.set_ylim(0, np.nanmax(y_int) * 1.20 if (len(y_int) and np.isfinite(np.nanmax(y_int))) else 1.0)

    # =========================
    # Align x-lims across panels
    # =========================
    xmax_main = float(np.max(x_main)) if len(x_main) else 0.0
    xmax_int  = float(np.max(x_int))  if len(x_int)  else 0.0

    common_left  = -bar_step * (0.5 + left_pad)
    common_right = max(xmax_main, xmax_int) + bar_step * 0.5

    for ax in (ax_main, ax_int):
        ax.set_xlim(common_left, common_right)
        ax.margins(x=0)

    # =========================
    # Legend (updated: P×P is striped patient colour)
    # =========================
    if show_legend:
        h_main_p = Patch(facecolor=base_color, edgecolor=base_color)
        h_main_t = Patch(facecolor=out_color,  edgecolor=out_color)

        h_pp = Patch(facecolor="white", edgecolor=base_color, hatch="\\\\\\")
        h_tt = Patch(facecolor=out_color,  edgecolor=out_color,  alpha=0.45)
        h_pt_g = Patch(facecolor="white", edgecolor=out_color, hatch="///")
        h_pt_b = Patch(facecolor="white", edgecolor=base_color, hatch="\\\\\\")

        fig.legend(
            handles=[h_main_p, h_main_t, h_pp, (h_pt_g, h_pt_b), h_tt],
            labels=[
                "Main effect: patient feature",
                "Main effect: therapist feature",
                "Interaction: patient×patient (striped)",
                "Interaction: patient×therapist (striped)",
                "Interaction: therapist×therapist (solid)",
            ],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            frameon=False,
            handlelength=2.5,
            columnspacing=1.6,
        )
        fig.subplots_adjust(bottom=0.30, wspace=0.32)
    else:
        fig.subplots_adjust(bottom=0.22, wspace=0.32)

    print(f"Loaded from: {tables_dir.resolve()}")
    return fig, (ax_main, ax_int)


# ============================================================
# NEW: One-axis combined SHAP plot (main + interactions)
# ============================================================
def plot_shap_combined_importance(
    *,
    output_root: str | Path = "model_outputs",
    run_name: str | None = None,
    tables_subdir: str = "tables",
    shap_main_filename: str = "cv_cat_shap_main_mean.csv",
    shap_interactions_filename: str = "cv_cat_shap_interactions_mean.csv",
    top_patient_features: int = 10,
    top_therapist_features: int = 10,
    top_interactions: int = 12,
    min_runs_present: int = 2,
    interactions_only_involving_therapist: bool = True,
    bar_step: float = 1.25,
    gap_main_pt: float = 0.25,
    gap_sections: float = 0.55,
    left_pad: float = 0.6,
    x_label_rotation: int = 30,
    tick_fontsize: int = 8,
    custom_label_map: Optional[Dict[str, str]] = None,
    width_in: float = 13.0,
    height_in: float = 4.8,
    bar_width: float = 0.72,
    show_legend: bool = True,
    title: str = "SHAP importance (main effects + interactions)",
    y_label: str = "Mean absolute SHAP contribution",
) -> Tuple[plt.Figure, plt.Axes]:

    tables_dir = resolve_tables_dir(output_root=output_root, run_name=run_name, tables_subdir=tables_subdir)
    feat_path = tables_dir / shap_main_filename
    int_path  = tables_dir / shap_interactions_filename

    df_feat = _read_importances(feat_path).copy()
    df_int  = _read_interactions(int_path).copy()

    if "runs_present" in df_feat.columns:
        df_feat = df_feat[df_feat["runs_present"].fillna(0) >= min_runs_present]
    if "runs_present" in df_int.columns:
        df_int = df_int[df_int["runs_present"].fillna(0) >= min_runs_present]

    df_feat["is_t"] = df_feat["feature"].apply(_has_therapist_token)
    df_feat_sorted = df_feat.sort_values("mean_importance", ascending=False)

    patient_main = df_feat_sorted[~df_feat_sorted["is_t"]].head(int(top_patient_features)).copy()
    therapist_main = (
        df_feat_sorted[df_feat_sorted["is_t"]]
        .loc[~df_feat_sorted["feature"].isin(patient_main["feature"])]
        .head(int(top_therapist_features))
        .copy()
    )
    main_rows = pd.concat([patient_main, therapist_main], ignore_index=True)

    df_int["has_t"] = df_int["orig_i"].apply(_has_therapist_token) | df_int["orig_j"].apply(_has_therapist_token)
    df_int["pt_pair"] = df_int.apply(lambda r: _is_patient_therapist_pair(r["orig_i"], r["orig_j"]), axis=1)
    df_int["pp_pair"] = (~df_int["has_t"])
    df_int["tt_pair"] = df_int["has_t"] & (~df_int["pt_pair"])

    df_int_sel = df_int.copy()
    if interactions_only_involving_therapist:
        df_int_sel = df_int_sel[df_int_sel["has_t"]]
    df_int_sel = df_int_sel.sort_values("mean_score", ascending=False).head(int(top_interactions)).copy()

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.linewidth": 1.0,
        "figure.dpi": 100,
        "hatch.linewidth": 1.2,
    })

    base_color = "#56B4E9"
    out_color  = "#009E73"

    fig, ax = plt.subplots(figsize=(width_in, height_in))

    n_pat = len(patient_main)
    n_th  = len(therapist_main)
    n_int = len(df_int_sel)

    x_pat = np.arange(n_pat, dtype=float) * bar_step
    x_th_start = (n_pat + gap_main_pt) * bar_step
    x_th = x_th_start + np.arange(n_th, dtype=float) * bar_step
    x_main = np.concatenate([x_pat, x_th]) if n_th > 0 else x_pat

    y_main = main_rows["mean_importance"].to_numpy(dtype=float)
    main_is_t = main_rows["is_t"].to_numpy(dtype=bool)

    x_int_start = ((n_pat + (gap_main_pt if n_th > 0 else 0) + n_th) + gap_sections) * bar_step
    x_int = x_int_start + np.arange(n_int, dtype=float) * bar_step
    y_int = df_int_sel["mean_score"].to_numpy(dtype=float)

    for xi, yi, is_t in zip(x_main, y_main, main_is_t):
        c = out_color if is_t else base_color
        ax.bar(xi, yi, width=bar_width, color=c, edgecolor=c, alpha=0.85, linewidth=1.0, zorder=3)

    for idx, r in df_int_sel.reset_index(drop=True).iterrows():
        xi = x_int[idx]
        v = float(r["mean_score"])
        pp_pair = bool(r["pp_pair"])
        pt_pair = bool(r["pt_pair"])
        tt_pair = bool(r["tt_pair"])

        if pt_pair:
            ax.bar(xi, v, width=bar_width, color="white", edgecolor="black", linewidth=1.0, zorder=2)
            ax.bar(xi, v, width=bar_width, color="none", edgecolor=out_color, hatch="///", linewidth=0.0, zorder=4)
            ax.bar(xi, v, width=bar_width, color="none", edgecolor=base_color, hatch="\\\\\\", linewidth=0.0, zorder=5)
        elif pp_pair:
            # P×P: striped patient colour
            ax.bar(xi, v, width=bar_width, color="white", edgecolor="black", linewidth=1.0, zorder=2)
            ax.bar(xi, v, width=bar_width, color="none", edgecolor=base_color, hatch="\\\\\\", linewidth=0.0, zorder=4)
        elif tt_pair:
            ax.bar(xi, v, width=bar_width, color=out_color, edgecolor=out_color, alpha=0.45, linewidth=1.0, zorder=3)
        else:
            ax.bar(xi, v, width=bar_width, color="grey", edgecolor="grey", alpha=0.35, linewidth=1.0, zorder=3)

    labels_main = [label_with_role_suffix(f, custom_map=custom_label_map) for f in main_rows["feature"].tolist()]
    labels_int = [
        humanize_interaction(i, j, custom_map=custom_label_map)
        for i, j in zip(df_int_sel["orig_i"].tolist(), df_int_sel["orig_j"].tolist())
    ]

    x_all = np.concatenate([x_main, x_int]) if n_int > 0 else x_main
    labels_all = labels_main + labels_int

    ax.set_xticks(x_all)
    ax.set_xticklabels(labels_all, rotation=x_label_rotation, ha="right", fontsize=tick_fontsize)

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    y_max = np.nanmax(np.concatenate([y_main, y_int]) if n_int > 0 else y_main)
    ax.set_ylim(0, y_max * 1.20 if np.isfinite(y_max) else 1.0)

    xmax = float(np.max(x_all)) if len(x_all) else 0.0
    common_left  = -bar_step * (0.5 + left_pad)
    common_right = xmax + bar_step * 0.5
    ax.set_xlim(common_left, common_right)
    ax.margins(x=0)

    if n_int > 0:
        sep_x = (x_int_start - (gap_sections * bar_step) / 2.0)
        ax.axvline(sep_x, linestyle=":", linewidth=1.0, color="black", alpha=0.5, zorder=1)

    if show_legend:
        h_patient = Patch(facecolor=base_color, edgecolor=base_color)
        h_therap  = Patch(facecolor=out_color,  edgecolor=out_color)
        h_pp = Patch(facecolor="white", edgecolor=base_color, hatch="\\\\\\")
        h_pt_g = Patch(facecolor="white", edgecolor=out_color, hatch="///")
        h_pt_b = Patch(facecolor="white", edgecolor=base_color, hatch="\\\\\\")
        h_tt = Patch(facecolor=out_color, edgecolor=out_color, alpha=0.45)

        fig.legend(
            handles=[h_patient, h_therap, h_pp, (h_pt_g, h_pt_b), h_tt],
            labels=[
                "Main effect: patient feature",
                "Main effect: therapist feature",
                "Interaction: patient×patient (striped)",
                "Interaction: patient×therapist (striped)",
                "Interaction: therapist×therapist (solid)",
            ],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
            frameon=False,
            handlelength=2.5,
            columnspacing=1.6,
        )
        fig.subplots_adjust(bottom=0.30)
    else:
        fig.subplots_adjust(bottom=0.20)

    print(f"Loaded from: {tables_dir.resolve()}")
    return fig, ax


# ============================================================
# Example calls
# ============================================================
# Two-panel (FI or SHAP):
# fig, _ = plot_catboost_importance(
#     feature_filename="cv_cat_shap_main_mean.csv",
#     interaction_filename="cv_cat_shap_interactions_mean.csv",
#     min_runs_present=2,
#     top_pp_interactions=4,
#     top_interactions=8,
#     y_label_main="Mean |SHAP|",
#     y_label_int="Mean |SHAP interaction|",
#     title_main="SHAP (main effects)",
#     title_int="SHAP (interactions)",
#     show_legend=True,
# )
#
# Combined SHAP (single axis):
# fig, ax = plot_shap_combined_importance()