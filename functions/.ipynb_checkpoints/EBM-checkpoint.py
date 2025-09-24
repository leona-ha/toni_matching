# -*- coding: utf-8 -*-
"""
EBM (GA2M) attribute-only therapist recommender (no IPS/DR).
- No therapist_id as a feature; learns patient×therapist-attribute interactions.
- Per-therapist time-based split (leave last / penultimate).
- Numeric imputation: 'iterative' (IterativeImputer) or 'median'; categorical: most_frequent.
- Outputs: factual test RMSE, ordinal diagnostic (rank ~ outcome), top-k tallies,
           and EBM explanations (global importances, term shapes, local contributions).
- Optional HPO for EBM via GridSearchCV + PredefinedSplit (time-based).
"""

import numpy as np
import pandas as pd
import logging, sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import Counter

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV, PredefinedSplit

from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm  # noqa: F401
from pandas.api import types as pdt

from interpret.glassbox import ExplainableBoostingRegressor

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Match_EBM")

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
@dataclass
class MatchConfig:
    # IDs / columns
    PATIENT_ID_COL: str = "patient_id"
    THERAPIST_ID_COL: str = "therapist_id"
    DATE_COL: str = "enrollment_date"
    DATE_DAYFIRST: bool = True
    DATE_FORMAT: Optional[str] = None

    # Outcome
    LABEL_COL: str = "delta_6w"      # higher = better (set TARGET_MODE="t1" if lower is better)
    TARGET_MODE: str = "delta"       # "delta" | "t1"
    OUTCOME_HIGHER_IS_BETTER: Optional[bool] = None
    STANDARDIZE_Y: bool = False

    # Features (NO therapist_id here)
    patient_numeric: List[str] = field(default_factory=list)
    patient_categorical: List[str] = field(default_factory=list)
    therapist_numeric: List[str] = field(default_factory=list)
    therapist_categorical: List[str] = field(default_factory=list)

    # Imputation
    impute_strategy: str = "iterative"   # "iterative" | "median"
    iterative_imputer_params: Dict[str, object] = field(default_factory=lambda: {
        "max_iter": 10, "initial_strategy": "median", "random_state": 0, "sample_posterior": True,
    })

    # EBM hyperparams (used when USE_HPO=False; during HPO we override via grid)
    ebm_params: Dict[str, object] = field(default_factory=lambda: dict(
        interactions=50, max_bins=128, learning_rate=0.02, validation_size=0.2,
        random_state=42
    ))

    # Optional HPO for EBM
    USE_HPO: bool = True
    ebm_param_grid: Dict[str, List] = field(default_factory=lambda: {
        "ebm__interactions": [20, 50, 100],
        "ebm__learning_rate": [0.01, 0.02],
        "ebm__max_bins": [64, 128],
    })

    # Reporting / examples
    TOP_KS: List[int] = field(default_factory=lambda: [1, 3, 5])
    SAVE_TOP_TERM_SHAPES: int = 5  # how many top terms' shapes to return

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _dedupe(seq):
    return list(dict.fromkeys(seq))

def _ohe_dense():
    # EBM expects dense; keep OneHot dense across sklearn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def _y_eff(y: np.ndarray, cfg: MatchConfig) -> np.ndarray:
    hib = cfg.OUTCOME_HIGHER_IS_BETTER if cfg.OUTCOME_HIGHER_IS_BETTER is not None else (cfg.TARGET_MODE != "t1")
    y = np.asarray(y, float)
    return y if hib else -y

def _s_eff(a: np.ndarray, cfg: MatchConfig) -> np.ndarray:
    return _y_eff(a, cfg)

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _rmse_scorer():
    # Works even if "neg_root_mean_squared_error" is unavailable
    return make_scorer(lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False)

def _filter_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    # Keep only columns that actually exist
    return [c for c in _dedupe(cols) if c in df.columns]

# ------------------------------------------------------------------
# Split (per-therapist, time-based)
# ------------------------------------------------------------------
def split_leave_last(df: pd.DataFrame, therapist_col: str, date_col: str
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr, va, te = [], [], []
    # Sort per therapist by date; NaT goes to end (okay for "leave last")
    for _, g in df.sort_values(date_col).groupby(therapist_col, sort=False):
        n = len(g)
        if n >= 3:
            te.append(g.iloc[[-1]]); va.append(g.iloc[[-2]]); tr.append(g.iloc[:-2])
        elif n == 2:
            te.append(g.iloc[[-1]]); tr.append(g.iloc[[0]])
        else:
            tr.append(g)
    df_train = pd.concat(tr).reset_index(drop=True)
    df_val   = pd.concat(va).reset_index(drop=True) if va else df.iloc[0:0].copy()
    df_test  = pd.concat(te).reset_index(drop=True) if te else df.iloc[0:0].copy()
    return df_train, df_val, df_test

# ------------------------------------------------------------------
# Tables & dyads
# ------------------------------------------------------------------
def therapists_table_from_train(cfg: MatchConfig, df_train: pd.DataFrame) -> pd.DataFrame:
    keep = [cfg.THERAPIST_ID_COL] + cfg.therapist_numeric + cfg.therapist_categorical
    keep = _filter_cols(df_train, keep)  # ensures id + features present
    if cfg.THERAPIST_ID_COL not in keep:
        raise KeyError(f"'{cfg.THERAPIST_ID_COL}' not found in training data.")
    return df_train[keep].drop_duplicates(cfg.THERAPIST_ID_COL).reset_index(drop=True)

def build_dyads_for_patient(cfg: MatchConfig, patient_row: pd.Series, therapists_table: pd.DataFrame) -> pd.DataFrame:
    p_cols = [cfg.PATIENT_ID_COL] + cfg.patient_numeric + cfg.patient_categorical
    p_cols = [c for c in p_cols if c in patient_row.index]
    p_rep = pd.DataFrame([patient_row[p_cols].values] * len(therapists_table), columns=p_cols)
    p_rep.index = therapists_table.index
    return pd.concat([p_rep, therapists_table], axis=1)

# ------------------------------------------------------------------
# Preprocessor
# ------------------------------------------------------------------
def make_feature_preprocessor(cfg: MatchConfig) -> ColumnTransformer:
    # Numeric imputation per cfg; categorical most_frequent + dense OHE (EBM)
    if cfg.impute_strategy == "iterative":
        num_imputer = IterativeImputer(**cfg.iterative_imputer_params)
    elif cfg.impute_strategy == "median":
        num_imputer = SimpleImputer(strategy="median")
    else:
        raise ValueError(f"Unknown impute_strategy: {cfg.impute_strategy}")

    to_numeric = FunctionTransformer(lambda X: pd.DataFrame(X).apply(pd.to_numeric, errors="coerce"))
    cat_imputer = SimpleImputer(strategy="most_frequent")
    transformers = []
    num_cols = cfg.patient_numeric + cfg.therapist_numeric
    cat_cols = cfg.patient_categorical + cfg.therapist_categorical
    if num_cols:
        transformers.append(("num", Pipeline([
            ("to_num", to_numeric), ("impute", num_imputer), ("scale", StandardScaler(with_mean=False)),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("impute", cat_imputer), ("ohe", _ohe_dense()),
        ]), cat_cols))
    # Always return dense ndarray; plus ask for pandas DataFrame with names if supported
    ct = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
    try:
        ct.set_output(transform="pandas")  # sklearn >=1.2
    except Exception:
        pass
    return ct

# ------------------------------------------------------------------
# EBM explanation utilities (robust across interpret versions)
# ------------------------------------------------------------------
def ebm_global_importance_df(ebm: ExplainableBoostingRegressor) -> pd.DataFrame:
    # Try standard API first
    try:
        gl = ebm.explain_global()
        d = gl.data()
        names = None
        scores = None
        if isinstance(d, dict):
            if "overall" in d:
                names = d["overall"].get("names")
                scores = d["overall"].get("scores")
            if (not names) and ("names" in d) and ("scores" in d):
                names, scores = d["names"], d["scores"]
        if names and scores and len(names) == len(scores) and len(names) > 0:
            return (pd.DataFrame({"term": names, "importance": scores})
                    .sort_values("importance", ascending=False).reset_index(drop=True))
    except Exception:
        pass
    # Fallback to model attributes
    term_names = getattr(ebm, "term_names_", None)
    term_imps  = getattr(ebm, "term_importances_", None)
    if term_names is not None and term_imps is not None and len(term_names) > 0:
        return (pd.DataFrame({"term": list(term_names), "importance": list(term_imps)})
                .sort_values("importance", ascending=False).reset_index(drop=True))
    return pd.DataFrame(columns=["term", "importance"])

def ebm_global_term_shape(ebm: ExplainableBoostingRegressor, term: str | int, warn_if_mismatch: bool = False):
    """Return a tidy shape for a univariate term; robust to interpret version quirks."""
    log = logging.getLogger("Match_EBM")
    gl = ebm.explain_global()

    # Resolve index
    if isinstance(term, int):
        idx = term
        term_name = f"#{term}"
    else:
        d_all = gl.data()
        names = []
        if isinstance(d_all, dict) and "overall" in d_all:
            names = d_all["overall"].get("names", []) or []
        if not names:
            names = list(getattr(ebm, "term_names_", []))
        if term not in names:
            raise ValueError(f"EBM term '{term}' not found.")
        idx = names.index(term)
        term_name = term

    td = gl.data(idx)
    ttype = td.get("type", None)

    if ttype != "univariate":
        # For interactions, return the raw payload (caller can inspect)
        return {
            "left_names":  td.get("left_names", td.get("feature_names_left", [])),
            "right_names": td.get("right_names", td.get("feature_names_right", [])),
            "scores":      td.get("scores", []),
        }

    # --- Univariate shape ---
    bins = td.get("bin_labels") or td.get("names")
    scores = td.get("scores", [])

    # If bins are missing but edges exist, synthesize labels
    if not bins and ("bin_edges_left" in td and "bin_edges_right" in td):
        left = td.get("bin_edges_left", [])
        right = td.get("bin_edges_right", [])
        try:
            bins = [(float(l) + float(r)) / 2.0 for l, r in zip(left, right)]
        except Exception:
            bins = [f"{l}-{r}" for l, r in zip(left, right)]

    # Final fallback: index labels
    if not bins:
        bins = list(range(len(scores)))

    # Flatten nested scores if present
    if len(scores) > 0 and isinstance(scores[0], (list, tuple, np.ndarray)):
        scores = [float(s[0]) if isinstance(s, (list, tuple, np.ndarray)) and len(s) else float(s) for s in scores]

    # Length guard
    n = min(len(bins), len(scores))
    if len(bins) != len(scores) and warn_if_mismatch:
        log.warning("Term shape length mismatch for %s (bins=%d, scores=%d) -> truncating to %d",
                    term_name, len(bins), len(scores), n)

    return pd.DataFrame({"bin": list(bins)[:n], "score": list(scores)[:n]})

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def ebm_local_explanation_df_from_pipeline(pipe: Pipeline, X_row_df: pd.DataFrame, y_row: Optional[pd.Series]=None) -> pd.DataFrame:
    pre: ColumnTransformer = pipe.named_steps["pre"]
    ebm: ExplainableBoostingRegressor = pipe.named_steps["ebm"]

    # Transform raw row -> model space used by EBM
    X_tx = pre.transform(X_row_df)
    if hasattr(X_tx, "toarray"):  # handle sparse just in case
        X_tx = X_tx.toarray()

    # Use exact feature names if available
    feat_names = list(getattr(ebm, "feature_names_in_", []))
    if feat_names and len(feat_names) == X_tx.shape[1]:
        X_tx_df = pd.DataFrame(X_tx, columns=feat_names)
    else:
        X_tx_df = pd.DataFrame(X_tx)

    loc = ebm.explain_local(X_tx_df, y_row) if y_row is not None else ebm.explain_local(X_tx_df)
    d = loc.data(0)

    # Tolerate key name differences across versions
    names = d.get("names") or d.get("feature_names") or []
    values = d.get("values") or d.get("features") or []
    contrib = d.get("scores") or d.get("contributions") or []

    return (pd.DataFrame({"term": names, "value": values, "contribution": contrib})
            .sort_values("contribution", ascending=False).reset_index(drop=True))

# ----------------- Feature/term naming & importance mapping -----------------
def _extract_base_from_transformed(name: str, cfg: MatchConfig) -> str:
    """Map transformed name ('cat__col=value' | 'cat__col_value' | 'num__col') to base column name."""
    s = str(name)
    if "__" in s:
        s = s.split("__", 1)[1]
    s = s.split("=", 1)[0]  # 'col=value' -> 'col'
    raw_cols = (cfg.patient_numeric + cfg.therapist_numeric +
                cfg.patient_categorical + cfg.therapist_categorical)
    if s in raw_cols:
        return s
    for col in sorted(raw_cols, key=len, reverse=True):
        if s.startswith(col + "_"):
            return col
    return s

def ebm_feature_and_term_tables(pipe: Pipeline, cfg: MatchConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return:
      - feature_name_mapping: idx -> transformed_name, base_feature, source
      - term_table: each EBM term with importance, type, transformed & base features
      - top_raw_features: aggregated importances per base raw feature (univariates)
      - top_interactions: most important interaction terms
    """
    pre: ColumnTransformer = pipe.named_steps["pre"]
    ebm: ExplainableBoostingRegressor = pipe.named_steps["ebm"]

    # Transformed feature names that EBM saw
    feat_names = list(getattr(ebm, "feature_names_in_", []))
    if not feat_names:
        try:
            feat_names = list(pre.get_feature_names_out())
        except Exception:
            n = int(getattr(ebm, "n_features_in_", 0))
            feat_names = [f"feature_{i:03d}" for i in range(n)]

    # Feature mapping
    fmap_rows = []
    for i, n in enumerate(feat_names):
        n = str(n)
        base = _extract_base_from_transformed(n, cfg)
        source = "cat" if n.startswith("cat__") else ("num" if n.startswith("num__") else "")
        fmap_rows.append({
            "feature_index": i,
            "transformed_name": n,
            "base_feature": base,
            "source": source,
        })
    feature_name_mapping = pd.DataFrame(fmap_rows)

    # Term structure from model
    term_feats = list(getattr(ebm, "term_features_", []))

    # --- Build *robust* term importances from explain_global() ---
    imp_df = ebm_global_importance_df(ebm)  # uses explain_global(); works across versions
    imp_map = {str(k): float(v) for k, v in zip(imp_df.get("term", []), imp_df.get("importance", []))}

    # Term names from model if available; else synthesize from transformed feature names
    model_term_names = list(getattr(ebm, "term_names_", []))
    if not model_term_names or len(model_term_names) != len(term_feats):
        model_term_names = []
        for idxs in term_feats:
            idxs = list(idxs)
            model_term_names.append(" x ".join(str(feat_names[j]) for j in idxs))

    # Build the term table
    term_rows = []
    for t_idx, idxs in enumerate(term_feats):
        idxs = list(idxs)
        ttype = "interaction" if len(idxs) > 1 else "univariate"
        trans = [feat_names[j] for j in idxs]
        base  = [_extract_base_from_transformed(n, cfg) for n in trans]
        term_name = str(model_term_names[t_idx]) if t_idx < len(model_term_names) else " x ".join(trans)

        # importance from imp_map (robust), fallback to NaN if not found
        importance = imp_map.get(term_name, np.nan)

        term_rows.append({
            "term_index": t_idx,
            "term_name": term_name,
            "importance": importance,
            "type": ttype,
            "features_transformed": trans,
            "features_base": base,
            "pretty": " x ".join(base),
        })

    term_table = (pd.DataFrame(term_rows)
                    .sort_values("importance", ascending=False, na_position="last")
                    .reset_index(drop=True))

    # Aggregations
    if not term_table.empty:
        # Raw features ranked by *univariate* term importances
        tmp = term_table[term_table["type"] == "univariate"].copy()
        tmp["base_feature"] = tmp["features_base"].apply(lambda lst: lst[0] if isinstance(lst, list) and lst else None)
        top_raw_features = (tmp.dropna(subset=["base_feature"])
                              .groupby("base_feature", as_index=False)["importance"].sum()
                              .sort_values("importance", ascending=False, na_position="last")
                              .reset_index(drop=True))
        # Most important interactions
        top_interactions = (term_table[term_table["type"] == "interaction"]
                              .loc[:, ["term_index", "term_name", "pretty", "importance", "features_transformed", "features_base"]]
                              .sort_values("importance", ascending=False, na_position="last")
                              .reset_index(drop=True))
    else:
        top_raw_features = pd.DataFrame(columns=["base_feature", "importance"])
        top_interactions = pd.DataFrame(columns=["term_index", "term_name", "pretty", "importance", "features_transformed", "features_base"])

    return feature_name_mapping, term_table, top_raw_features, top_interactions

# ------------------------------------------------------------------
# Model wrapper (EBM inside a sklearn Pipeline)
# ------------------------------------------------------------------
class EBMMatcher:
    def __init__(self, cfg: MatchConfig):
        self.cfg = cfg
        self.pipe_: Optional[Pipeline] = None
        self.therapists_: Optional[pd.DataFrame] = None
        self.best_params_: Dict[str, object] = {}

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame):
        # Build therapist candidate table from TRAIN
        self.therapists_ = therapists_table_from_train(self.cfg, df_train)

        # Feature pipe + EBM
        pre = make_feature_preprocessor(self.cfg)

        # --- Build EBM (version-safe + no outer_bags warning in HPO) ---
        hpo_mode = bool(getattr(self.cfg, "USE_HPO", False))
        params = dict(self.cfg.ebm_params)
        if hpo_mode:
            params["validation_size"] = 0.0     # avoid nested validation
            params["outer_bags"] = 1            # avoid warning & speed up
            try:
                ebm = ExplainableBoostingRegressor(early_stopping=False, **params)
            except TypeError:
                ebm = ExplainableBoostingRegressor(**params)
        else:
            ebm = ExplainableBoostingRegressor(**params)

        pipe = Pipeline([("pre", pre), ("ebm", ebm)])

        # Train on observed dyads (train+val), with PredefinedSplit for HPO
        base_all = _dedupe(
            self.cfg.patient_numeric + self.cfg.patient_categorical +
            self.cfg.therapist_numeric + self.cfg.therapist_categorical
        )
        tr_cols = _filter_cols(df_train, base_all)
        va_cols = _filter_cols(df_val, base_all) if len(df_val) else tr_cols

        tr = df_train[tr_cols].copy(); ytr = df_train[self.cfg.LABEL_COL].to_numpy()
        if len(df_val):
            va = df_val[va_cols].copy();   yva = df_val[self.cfg.LABEL_COL].to_numpy()
        else:
            va = df_val.copy(); yva = np.array([])

        X_trval = pd.concat([tr, va], ignore_index=True) if len(va) else tr
        y_trval = np.concatenate([ytr, yva]) if len(yva) else ytr

        if hpo_mode and len(df_val) > 0:
            test_fold = np.r_[np.full(len(tr), -1), np.zeros(len(va), dtype=int)]
            cv = PredefinedSplit(test_fold)
            gs = GridSearchCV(
                pipe,
                param_grid=self.cfg.ebm_param_grid,
                scoring=_rmse_scorer(),
                refit=True,
                cv=cv,
                n_jobs=1,
                verbose=0,
            )
            gs.fit(X_trval, y_trval)
            self.pipe_ = gs.best_estimator_
            self.best_params_ = dict(gs.best_params_)
            logger.info("EBM HPO best params: %s", self.best_params_)
        else:
            pipe.fit(X_trval, y_trval)
            self.pipe_ = pipe
            ebm_fitted: ExplainableBoostingRegressor = self.pipe_.named_steps["ebm"]
            self.best_params_ = {
                "ebm__interactions": ebm_fitted.interactions,
                "ebm__learning_rate": ebm_fitted.learning_rate,
                "ebm__max_bins": ebm_fitted.max_bins,
            }

    # Predict mu(x, t) for all candidate therapists for a given patient row
    def _predict_all(self, patient_row: pd.Series) -> pd.DataFrame:
        if self.therapists_ is None:
            raise RuntimeError("Call fit(...) before predicting.")
        dyads = build_dyads_for_patient(self.cfg, patient_row, self.therapists_)
        base_cols = _dedupe(
            self.cfg.patient_numeric + self.cfg.patient_categorical +
            self.cfg.therapist_numeric + self.cfg.therapist_categorical
        )
        base_cols = _filter_cols(dyads, base_cols)
        preds = self.pipe_.predict(dyads[base_cols]).astype(float)
        out = self.therapists_[[self.cfg.THERAPIST_ID_COL]].copy()
        out["mu_hat"] = preds
        return out

    # Top-k set S_k(x) (therapists and mu values)
    def recommend_topk(self, patient_row: pd.Series, k: int) -> pd.DataFrame:
        sc = self._predict_all(patient_row)
        hib = self.cfg.OUTCOME_HIGHER_IS_BETTER if self.cfg.OUTCOME_HIGHER_IS_BETTER is not None else (self.cfg.TARGET_MODE != "t1")
        sc = sc.sort_values("mu_hat", ascending=not hib).head(k).reset_index(drop=True)
        return sc

# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------
class MatchPipelineEBM:
    def __init__(self, cfg: MatchConfig):
        self.cfg = cfg
        self.df: Optional[pd.DataFrame] = None
        self.matcher_: Optional[EBMMatcher] = None

    def set_data(self, df: pd.DataFrame) -> None:
        df = df.copy()
        if self.cfg.OUTCOME_HIGHER_IS_BETTER is None:
            self.cfg.OUTCOME_HIGHER_IS_BETTER = (self.cfg.TARGET_MODE != "t1")
        # Parse date
        if self.cfg.DATE_COL in df.columns and not pdt.is_datetime64_any_dtype(df[self.cfg.DATE_COL]):
            df[self.cfg.DATE_COL] = pd.to_datetime(
                df[self.cfg.DATE_COL], dayfirst=self.cfg.DATE_DAYFIRST, format=self.cfg.DATE_FORMAT, errors="coerce"
            )
        self.df = df

    def _ordinal_diag(self, df_test: pd.DataFrame, ther_table: pd.DataFrame, y_test: np.ndarray):
        # Per-patient factual rank among all therapists, then ordinal regression: rank ~ outcome
        ranks, top1_tids, top5_tids_all = [], [], []
        tid_col = self.cfg.THERAPIST_ID_COL

        for _, row in df_test.iterrows():
            sc = self.matcher_._predict_all(row)
            scores_eff = _s_eff(sc["mu_hat"].to_numpy(), self.cfg)
            tids = sc[tid_col].astype("string").to_numpy()
            order = np.argsort(-scores_eff)
            tids_ordered = tids[order]

            top1_tids.append(tids_ordered[0])
            top5_tids_all.extend(list(tids_ordered[:5]))

            factual = str(row[tid_col])
            pos = np.where(tids_ordered == factual)[0]
            ranks.append(int(pos[0]) + 1 if len(pos) else len(tids_ordered) + 1)

        ranks = np.asarray(ranks, int)
        yy_test = _y_eff(y_test, self.cfg)

        # Ordinal regression (rank ~ outcome)
        llf = aic = bic = ll_per_case = pseudo_r2 = np.nan
        coef_change_std = or_change_std = np.nan
        used_link = used_opt = None; sd_used = np.nan

        if len(df_test) > 1:
            y_for_fit = yy_test.copy()
            sd = float(np.nanstd(y_for_fit))
            if sd > 0:
                y_for_fit = (y_for_fit - float(np.nanmean(y_for_fit))) / sd
                sd_used = sd
            exog_name = self.cfg.LABEL_COL
            for link in ["logit"]:
                for method in ["lbfgs", "bfgs", "newton"]:
                    try:
                        endog = pd.Series(ranks, name="rank")
                        exog  = pd.DataFrame({exog_name: y_for_fit})
                        mdl_t = OrderedModel(endog, exog, distr=link)
                        res_t = mdl_t.fit(method=method, disp=False, maxiter=400)
                        llf = float(res_t.llf); ll_per_case = llf / len(df_test)
                        c = float(res_t.params.get(exog_name, np.nan))
                        coef_change_std = c
                        or_change_std = float(np.exp(c)) if np.isfinite(c) else np.nan
                        try:
                            null = OrderedModel(endog, pd.DataFrame({"const": np.ones_like(ranks)}), distr=link)
                            res_null = null.fit(method=method, disp=False, maxiter=400)
                            pseudo_r2 = 1.0 - (llf / float(res_null.llf))
                        except Exception:
                            pseudo_r2 = np.nan
                        used_link, used_opt = link, method
                        break
                    except Exception:
                        continue
                if used_link is not None:
                    break

        # Tallies
        ther_ids = ther_table[tid_col].astype("string").tolist()
        n_pat = len(df_test)
        c_top1 = Counter(top1_tids); c_top5 = Counter(top5_tids_all)
        ther_reco_counts = pd.DataFrame({tid_col: ther_ids})
        ther_reco_counts["top1_count"] = ther_reco_counts[tid_col].map(c_top1).fillna(0).astype(int)
        ther_reco_counts["top5_count"] = ther_reco_counts[tid_col].map(c_top5).fillna(0).astype(int)
        ther_reco_counts["n_test_patients"] = n_pat
        ther_reco_counts["top1_share"] = ther_reco_counts["top1_count"] / max(n_pat, 1)
        ther_reco_counts["top5_share"] = ther_reco_counts["top5_count"] / max(n_pat, 1)
        ther_reco_counts = ther_reco_counts.sort_values(
            ["top1_count","top5_count",tid_col], ascending=[False, False, True]
        ).reset_index(drop=True)

        row = {
            "test_rmse": None,  # filled outside
            "ord_ll": llf,
            "ord_ll_per_case": ll_per_case,
            "ord_pseudo_r2": pseudo_r2,
            "ord_coef_change_std": coef_change_std,
            "ord_or_change_std": or_change_std,
            "ord_link": used_link,
            "ord_optimizer": used_opt,
            "ord_sd_used": sd_used,
        }
        return row, ther_reco_counts, ranks

    def run(self) -> Dict[str, object]:
        if self.df is None:
            raise RuntimeError("Call set_data(df) first.")
        df_train, df_val, df_test = split_leave_last(self.df, self.cfg.THERAPIST_ID_COL, self.cfg.DATE_COL)
        logger.info("Therapists: %d | Train=%d | Val=%d | Test=%d",
                    self.df[self.cfg.THERAPIST_ID_COL].nunique(), len(df_train), len(df_val), len(df_test))

        self.matcher_ = EBMMatcher(self.cfg)
        self.matcher_.fit(df_train, df_val)

        # Factual RMSE on test (observed dyads)
        base_all = _dedupe(self.cfg.patient_numeric + self.cfg.patient_categorical +
                           self.cfg.therapist_numeric + self.cfg.therapist_categorical)
        base_cols_test = _filter_cols(df_test, base_all)
        y_test = df_test[self.cfg.LABEL_COL].to_numpy()
        y_pred_factual = self.matcher_.pipe_.predict(df_test[base_cols_test]).astype(float) if len(df_test) else np.array([])
        rmse_test = rmse(y_test, y_pred_factual) if len(df_test) else np.nan

        # Ordinal diagnostic & tallies
        ther_table = self.matcher_.therapists_
        ord_row, ther_reco_counts, ranks = self._ordinal_diag(df_test, ther_table, y_test)
        ord_row["test_rmse"] = rmse_test

        # Example: top-k recommendations for first test patient
        example_scores = pd.DataFrame()
        if len(df_test) > 0:
            row = df_test.iloc[0]
            sc = self.matcher_.recommend_topk(row, max(self.cfg.TOP_KS))
            example_scores = sc.rename(columns={"mu_hat": "predicted_outcome"})

        # --- EBM explanations ---
        ebm: ExplainableBoostingRegressor = self.matcher_.pipe_.named_steps["ebm"]
        ebm_imp = ebm_global_importance_df(ebm)

        # Term shapes (no spammy warnings by default)
        top_terms = ebm_imp["term"].head(self.cfg.SAVE_TOP_TERM_SHAPES).tolist()
        term_shapes: Dict[str, object] = {}
        for t in top_terms:
            try:
                term_shapes[t] = ebm_global_term_shape(ebm, t, warn_if_mismatch=False)  # set True to log mismatches
            except Exception as ex:
                logging.getLogger("Match_EBM").warning("Skipping term shape for %s: %s", t, ex)

        # Local explanation for 1st test row
        ebm_loc = pd.DataFrame()
        if len(df_test) > 0:
            X_one_raw = df_test[base_cols_test].iloc[[0]].copy()
            y_one = df_test[self.cfg.LABEL_COL].iloc[[0]].copy()
            ebm_loc = ebm_local_explanation_df_from_pipeline(self.matcher_.pipe_, X_one_raw, y_one)

        # Feature/term introspection
        feature_name_mapping, term_table, top_raw_features, top_interactions = ebm_feature_and_term_tables(self.matcher_.pipe_, self.cfg)

        summary = pd.DataFrame([ord_row])

        return {
            "summary": summary,                                  # test_rmse + ordinal metrics
            "therapists": ther_table,                            # candidate therapists (train)
            "therapist_recommendations": ther_reco_counts,       # top1/top5 tallies on test
            "example_scores_first_test_patient": example_scores, # ranked list for one patient
            "ebm_global_importance": ebm_imp,                    # term importances
            "ebm_global_term_shapes": term_shapes,               # shapes for top terms (robust)
            "ebm_local_example": ebm_loc,                        # per-dyad contributions
            "test_ranks": pd.Series(ranks, name="rank"),         # factual rank per test case
            "ebm_best_params": getattr(self.matcher_, "best_params_", {}),
            # NEW: readable feature/interaction reporting
            "feature_name_mapping": feature_name_mapping,        # idx -> transformed & base names
            "ebm_terms": term_table,                             # every term (uni + interactions) with importance
            "top_raw_features": top_raw_features,                # raw features ranked by summed univariate importance
            "top_interactions": top_interactions,                # most important interactions (pretty "A x B")
        }

# ---- Runner
def run_pipeline(df: pd.DataFrame, cfg: MatchConfig):
    pipe = MatchPipelineEBM(cfg)
    pipe.set_data(df)
    return pipe.run()
