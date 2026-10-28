# therapist_icc_t1_distress_boot.py

import warnings
import statsmodels.formula.api as smf
import pandas as pd

# -----------------------------
# Helpers: variance + ICC
# -----------------------------
def _therapist_var(fit) -> float:
    try:
        return float(np.asarray(getattr(fit, "cov_re", np.array([[np.nan]])))[0, 0])
    except Exception:
        return float(fit.cov_re.iloc[0, 0])

def _extract_parts_simple(fit):
    v_t = _therapist_var(fit)
    v_e = float(fit.scale)
    denom = v_t + v_e
    icc = v_t / denom if np.isfinite(denom) and denom > 0 else np.nan
    return icc, v_t, v_e

def _fit_mixedlm(formula: str, d: pd.DataFrame, group_col: str, reml: bool = True):
    methods = ["lbfgs", "bfgs", "cg", "powell", "nm"]
    fit = None
    used_method = None
    last_err = None

    for m in methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(formula, d, groups=d[group_col], re_formula="1")
                fit = model.fit(method=m, reml=reml, maxiter=2000, disp=0)
            if getattr(fit, "converged", False):
                used_method = m
                break
        except Exception as e:
            last_err = str(e)
            fit = None

    return fit, used_method, last_err


# -----------------------------
# Cluster bootstrap for ICC
# -----------------------------
def bootstrap_icc_cluster(
    df: pd.DataFrame,
    formula: str,
    therapist_col: str,
    B: int = 2000,
    reml: bool = True,
    seed: int = 123,
    min_patients_per_therapist: int = 2,
):
    """
    Nonparametric cluster bootstrap by therapist.
    Important: when sampling therapists with replacement, we RELABEL therapist IDs
    so duplicate draws become independent clusters.
    Returns: dict with mean, CI, n_ok, icc_samples (np.array)
    """
    rng = np.random.default_rng(seed)

    d = df.copy()
    d[therapist_col] = d[therapist_col].astype(str)

    # filter very small caseload therapists (stability; matches thesis logic)
    if min_patients_per_therapist is not None and min_patients_per_therapist > 1:
        counts = d[therapist_col].value_counts()
        keep = counts[counts >= min_patients_per_therapist].index
        d = d[d[therapist_col].isin(keep)].copy()

    therapists = d[therapist_col].unique()
    if len(therapists) < 2:
        return {
            "boot_mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_ok": 0,
            "B": B,
            "icc_samples": np.array([]),
            "message": "Insufficient therapists after filtering",
        }

    iccs = []
    for b in range(B):
        # sample therapist clusters with replacement
        sampled = rng.choice(therapists, size=len(therapists), replace=True)

        # build bootstrap sample, relabeling therapist ids to avoid “same group twice”
        chunks = []
        for j, t in enumerate(sampled):
            chunk = d[d[therapist_col] == t].copy()
            chunk[therapist_col] = f"{t}__boot{b}_{j}"
            chunks.append(chunk)
        db = pd.concat(chunks, ignore_index=True)

        fit, _, _ = _fit_mixedlm(formula, db, therapist_col, reml=reml)
        if fit is None or not getattr(fit, "converged", False):
            continue

        icc, _, _ = _extract_parts_simple(fit)
        if np.isfinite(icc):
            iccs.append(float(icc))

    iccs = np.array(iccs, dtype=float)
    if iccs.size == 0:
        return {
            "boot_mean": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_ok": 0,
            "B": B,
            "icc_samples": iccs,
            "message": "All bootstrap fits failed or returned NaN",
        }

    return {
        "boot_mean": float(np.mean(iccs)),
        "ci_low": float(np.quantile(iccs, 0.025)),
        "ci_high": float(np.quantile(iccs, 0.975)),
        "n_ok": int(iccs.size),
        "B": int(B),
        "icc_samples": iccs,
        "message": "ok",
    }


# -----------------------------
# Your original fits (point estimate ICC)
# -----------------------------
def fit_icc_point(df, outcome_col="t1_distress_sum", baseline_col=None,
                  therapist_col="therapist_id", reml=True, min_patients_per_therapist=2):

    cols = [outcome_col, therapist_col] + ([] if baseline_col is None else [baseline_col])
    d = df[cols].dropna().copy()
    d[therapist_col] = d[therapist_col].astype(str)

    if min_patients_per_therapist is not None and min_patients_per_therapist > 1:
        counts = d[therapist_col].value_counts()
        keep = counts[counts >= min_patients_per_therapist].index
        d = d[d[therapist_col].isin(keep)].copy()

    formula = f"{outcome_col} ~ 1" if baseline_col is None else f"{outcome_col} ~ {baseline_col}"
    fit, used_method, err = _fit_mixedlm(formula, d, therapist_col, reml=reml)

    if fit is None or not getattr(fit, "converged", False):
        return {"icc": np.nan, "v_t": np.nan, "v_e": np.nan, "n": len(d),
                "n_therapists": d[therapist_col].nunique(),
                "converged": False, "method": None, "message": err or "no converge", "formula": formula}

    icc, v_t, v_e = _extract_parts_simple(fit)
    return {"icc": float(icc), "v_t": float(v_t), "v_e": float(v_e), "n": len(d),
            "n_therapists": d[therapist_col].nunique(),
            "converged": True, "method": used_method, "message": "ok", "formula": formula}