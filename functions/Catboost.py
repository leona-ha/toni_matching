# -*- coding: utf-8 -*-
"""
CatBoost dual-model with hierarchical shrinkage (M1 with therapist_id, M0 without) and IPS/DR evaluation.
- Per-therapist time-based split preserved.
- Numeric imputation: 'iterative' or 'median' (categorical filled/imputed).
- Off-policy metrics only: IPS/DR policy value & uplift vs. status quo.
- Also returns:
  * long-table of top-k recommendations per test patient (k>=5 ensured)
  * per-therapist aggregate counts: top1_count / top5_count (+ shares)
"""

import numpy as np
import pandas as pd
import logging, sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

from catboost import CatBoostRegressor, Pool

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Match_CB_Hier_DR")

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
    LABEL_COL: str = "delta_6w"
    TARGET_MODE: str = "delta"       # "delta" | "t1"
    OUTCOME_HIGHER_IS_BETTER: Optional[bool] = None

    # Features
    patient_numeric: List[str] = field(default_factory=list)
    patient_categorical: List[str] = field(default_factory=list)
    therapist_numeric: List[str] = field(default_factory=list)
    therapist_categorical: List[str] = field(default_factory=list)
    include_therapist_id_feature: bool = True  # in M1 only

    # Propensity-specific patient features (fallback to patient_* if None)
    propensity_patient_numeric:  Optional[List[str]] = None
    propensity_patient_categorical: Optional[List[str]] = None

    # Imputation
    impute_strategy: str = "iterative"  # "iterative" | "median"
    iterative_imputer_params: Dict[str, object] = field(default_factory=lambda: {
        "max_iter": 10, "initial_strategy": "median", "random_state": 0, "sample_posterior": True,
    })

    # Hierarchical shrinkage
    shrink_m: float = 50.0
    min_seen_for_id: int = 1

    # CatBoost params (you can tune; here are sane defaults)
    params_m1: Dict[str, object] = field(default_factory=lambda: dict(
        loss_function="RMSE", depth=8, l2_leaf_reg=10.0, learning_rate=0.05, iterations=1000,
        od_type="Iter", od_wait=100, random_seed=42, verbose=False
    ))
    params_m0: Dict[str, object] = field(default_factory=lambda: dict(
        loss_function="RMSE", depth=8, l2_leaf_reg=10.0, learning_rate=0.05, iterations=1000,
        od_type="Iter", od_wait=100, random_seed=42, verbose=False
    ))

    # Policy evaluation
    TOP_KS: List[int] = field(default_factory=lambda: [1, 3, 5])
    MIN_PROPENSITY: float = 1e-6

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _dedupe(seq): return list(dict.fromkeys(seq))
def rmse(y_true, y_pred): return float(np.sqrt(mean_squared_error(y_true, y_pred)))
def _ohe_dense():
    try:
        # sklearn >=1.2
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)
    except TypeError:
        # sklearn <1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def _y_eff(y: np.ndarray, cfg: MatchConfig) -> np.ndarray:
    hib = cfg.OUTCOME_HIGHER_IS_BETTER if cfg.OUTCOME_HIGHER_IS_BETTER is not None else (cfg.TARGET_MODE != "t1")
    y = np.asarray(y, float)
    return y if hib else -y

# ------------------------------------------------------------------
# Split
# ------------------------------------------------------------------
def split_leave_last(df: pd.DataFrame, therapist_col: str, date_col: str
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr, va, te = [], [], []
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
    keep = [c for c in keep if c in df_train.columns] + [cfg.THERAPIST_ID_COL]
    return df_train[_dedupe(keep)].drop_duplicates(cfg.THERAPIST_ID_COL).reset_index(drop=True)

def build_dyads_for_patient(cfg: MatchConfig, patient_row: pd.Series, therapists_table: pd.DataFrame) -> pd.DataFrame:
    p_cols = [cfg.PATIENT_ID_COL] + cfg.patient_numeric + cfg.patient_categorical
    p_cols = [c for c in p_cols if c in patient_row.index]
    p_rep = pd.DataFrame([patient_row[p_cols].values] * len(therapists_table), columns=p_cols)
    p_rep.index = therapists_table.index
    return pd.concat([p_rep, therapists_table], axis=1)

# ------------------------------------------------------------------
# Propensity (patient features only)
# ------------------------------------------------------------------
def make_propensity_pipeline(cfg: MatchConfig) -> Pipeline:
    # use the dedicated propensity lists, fall back to patient_* if None
    p_num = cfg.propensity_patient_numeric or cfg.patient_numeric
    p_cat = cfg.propensity_patient_categorical or cfg.patient_categorical

    if cfg.impute_strategy == "iterative":
        num_imputer = IterativeImputer(**cfg.iterative_imputer_params)
    else:
        num_imputer = SimpleImputer(strategy="median")

    to_numeric = FunctionTransformer(lambda X: pd.DataFrame(X).apply(pd.to_numeric, errors="coerce"))
    cat_imputer = SimpleImputer(strategy="most_frequent")

    pre = ColumnTransformer([
        ("num", Pipeline([("to_num", to_numeric), ("impute", num_imputer), ("scale", StandardScaler(with_mean=False))]), p_num),
        ("cat", Pipeline([("impute", cat_imputer), ("ohe", _ohe_dense())]), p_cat),
    ], remainder="drop", sparse_threshold=0.0)  # force dense

    # auto-multinomial with lbfgs and >2 classes; no deprecated 'multi_class'
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0)
    return Pipeline([("pre", pre), ("clf", clf)])

# ------------------------------------------------------------------
# CatBoost blended predictor
# ------------------------------------------------------------------
class CatBoostBlendedRegressor:
    def __init__(self, cfg: MatchConfig):
        self.cfg = cfg
        self.m1_: Optional[CatBoostRegressor] = None
        self.m0_: Optional[CatBoostRegressor] = None
        self.num_imputer_: Optional[object] = None
        self.cat_fill_: str = "NA"
        self.counts_: Dict[int, int] = {}
        self.cols_m1_: List[str] = []
        self.cols_m0_: List[str] = []
        self.cat_idx_m1_: List[int] = []
        self.cat_idx_m0_: List[int] = []

    def _impute_numeric_fit(self, X):
        if self.cfg.impute_strategy == "iterative":
            self.num_imputer_ = IterativeImputer(**self.cfg.iterative_imputer_params)
        else:
            self.num_imputer_ = SimpleImputer(strategy="median")
        num_cols = [c for c in self.cfg.patient_numeric + self.cfg.therapist_numeric if c in X.columns]
        if num_cols:
            self.num_imputer_.fit(X[num_cols].apply(pd.to_numeric, errors="coerce"))

    def _impute_numeric_apply(self, X):
        num_cols = [c for c in self.cfg.patient_numeric + self.cfg.therapist_numeric if c in X.columns]
        if num_cols:
            X[num_cols] = self.num_imputer_.transform(X[num_cols].apply(pd.to_numeric, errors="coerce"))
        return X

    def _prep_cats(self, X: pd.DataFrame, include_id: bool) -> Tuple[pd.DataFrame, List[int]]:
        cats = self.cfg.patient_categorical + self.cfg.therapist_categorical
        if include_id and self.cfg.include_therapist_id_feature:
            cats = cats + [self.cfg.THERAPIST_ID_COL]
        cats = [c for c in cats if c in X.columns]
        for c in cats:
            X[c] = X[c].astype("string").fillna(self.cat_fill_).astype(str)
        idx = [list(X.columns).index(c) for c in cats]
        return X, idx

    def _pool(self, X: pd.DataFrame, y=None, include_id=True) -> Tuple[Pool, List[str], List[int]]:
        cols = _dedupe(self.cfg.patient_numeric + self.cfg.patient_categorical +
                       self.cfg.therapist_numeric + self.cfg.therapist_categorical +
                       ([self.cfg.THERAPIST_ID_COL] if include_id and self.cfg.include_therapist_id_feature else []))
        cols = [c for c in cols if c in X.columns]
        Xf = X[cols].copy()
        if y is not None:
            self._impute_numeric_fit(Xf)
        Xf = self._impute_numeric_apply(Xf)
        Xf, cat_idx = self._prep_cats(Xf, include_id)
        return (Pool(Xf, label=y, cat_features=cat_idx) if y is not None
                else Pool(Xf, cat_features=cat_idx)), cols, cat_idx

    def fit(self, X: pd.DataFrame, y: np.ndarray, params_m1: Dict, params_m0: Dict):
        self.counts_ = X[self.cfg.THERAPIST_ID_COL].value_counts().to_dict()
        pool_m1, self.cols_m1_, self.cat_idx_m1_ = self._pool(X, y, include_id=True)
        pool_m0, self.cols_m0_, self.cat_idx_m0_ = self._pool(X, y, include_id=False)
        self.m1_ = CatBoostRegressor(**params_m1).fit(pool_m1)
        self.m0_ = CatBoostRegressor(**params_m0).fit(pool_m0)
        return self

    def _w(self, tid: int) -> float:
        n = float(self.counts_.get(int(tid), 0))
        if n < self.cfg.min_seen_for_id:
            return 0.0
        m = float(self.cfg.shrink_m)
        return n / (n + m)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xm1 = X[self.cols_m1_].copy() if self.cols_m1_ else X.copy()
        Xm0 = X[self.cols_m0_].copy() if self.cols_m0_ else X.copy()
        Xm1 = self._impute_numeric_apply(Xm1); Xm0 = self._impute_numeric_apply(Xm0)
        for c in [col for i, col in enumerate(self.cols_m1_) if i in self.cat_idx_m1_]:
            Xm1[c] = Xm1[c].astype("string").fillna(self.cat_fill_).astype(str)
        for c in [col for i, col in enumerate(self.cols_m0_) if i in self.cat_idx_m0_]:
            Xm0[c] = Xm0[c].astype("string").fillna(self.cat_fill_).astype(str)
        y1 = self.m1_.predict(Pool(Xm1, cat_features=self.cat_idx_m1_))
        y0 = self.m0_.predict(Pool(Xm0, cat_features=self.cat_idx_m0_))
        tids = X[self.cfg.THERAPIST_ID_COL].astype(int).to_numpy()
        w = np.array([self._w(t) for t in tids], float)
        return w * y1 + (1.0 - w) * y0

# ------------------------------------------------------------------
# Policy evaluation (IPS/DR)
# ------------------------------------------------------------------
def evaluate_policy_ips_dr(df_test: pd.DataFrame,
                           cfg: MatchConfig,
                           therapists: pd.DataFrame,
                           outcome_predict_fn,         # fn(patient_row, therapist_id_array) -> mu_hat array
                           propensity_model: Pipeline, # fitted on train+val
                           label_map: Dict[int, int],  # tid -> class index
                           ) -> pd.DataFrame:
    hib = cfg.OUTCOME_HIGHER_IS_BETTER if cfg.OUTCOME_HIGHER_IS_BETTER is not None else (cfg.TARGET_MODE != "t1")
    y_eff = _y_eff(df_test[cfg.LABEL_COL].to_numpy(), cfg)
    observed_mean = float(np.mean(y_eff))

    # use dedicated propensity lists, fallback if None
    p_num = cfg.propensity_patient_numeric or cfg.patient_numeric
    p_cat = cfg.propensity_patient_categorical or cfg.patient_categorical
    prop_cols = p_num + p_cat

    res = []
    for k in cfg.TOP_KS:
        ips_vals, dr_vals, mu_bars, w_list = [], [], [], []
        for _, row in df_test.iterrows():
            # Predict mu for all therapists (vectorized over therapist table)
            tids = therapists[cfg.THERAPIST_ID_COL].astype(int).to_numpy()
            mu_all = outcome_predict_fn(row, tids)  # shape (T,)

            order = np.argsort(mu_all) if not hib else np.argsort(-mu_all)
            top_idx = order[:k]
            S_tids = tids[top_idx]
            mu_S = mu_all[top_idx]
            mu_bar = float(mu_S.mean()) if mu_S.size else 0.0

            # Propensity mass for the set: build a single-row DF with all expected columns
            Xp = pd.DataFrame([{c: row.get(c, np.nan) for c in prop_cols}], columns=prop_cols)
            probs = propensity_model.predict_proba(Xp)[0]

            pi_S = 0.0
            for tid in S_tids.tolist():
                cls = label_map.get(int(tid), None)
                if cls is not None and cls < len(probs):
                    pi_S += float(probs[cls])
            pi_S = max(pi_S, cfg.MIN_PROPENSITY)

            factual_tid = int(row[cfg.THERAPIST_ID_COL])
            yi = float(_y_eff(np.array([row[cfg.LABEL_COL]]), cfg)[0])

            # predicted for the factual dyad; if not present, fall back to mu_bar
            where_f = np.where(tids == factual_tid)[0]
            mu_t_f = float(mu_all[where_f[0]]) if where_f.size else mu_bar

            in_S = int(factual_tid in set(S_tids.tolist()))
            w = float(in_S / pi_S)
            ips_vals.append(w * yi)
            dr_vals.append(mu_bar + w * (yi - mu_t_f))
            mu_bars.append(mu_bar); w_list.append(w)

        value_ips = float(np.mean(ips_vals)) if ips_vals else np.nan
        value_dr  = float(np.mean(dr_vals))  if dr_vals  else np.nan
        uplift_ips = value_ips - observed_mean
        uplift_dr  = value_dr  - observed_mean
        w = np.asarray(w_list, float)
        ess = float((w.sum() ** 2) / np.maximum((w ** 2).sum(), 1e-12)) if w.size else np.nan

        res.append({
            "k": int(k),
            "value_ips": value_ips,
            "value_dr": value_dr,
            "uplift_ips": uplift_ips,
            "uplift_dr": uplift_dr,
            "value_model_mean": float(np.mean(mu_bars)) if mu_bars else np.nan,
            "observed_mean": observed_mean,
            "ess_ips": ess,
        })
    return pd.DataFrame(res)

# ------------------------------------------------------------------
# Make top-k recommendations (long table; ensure k>=5)
# ------------------------------------------------------------------
def make_topk_recommendations(df_test: pd.DataFrame,
                              cfg: MatchConfig,
                              therapists: pd.DataFrame,
                              outcome_predict_fn) -> pd.DataFrame:
    """Return long DataFrame with one row per (patient, rank<=k_max).
       k_max is at least 5 so we can compute top5 counts."""
    hib = cfg.OUTCOME_HIGHER_IS_BETTER if cfg.OUTCOME_HIGHER_IS_BETTER is not None else (cfg.TARGET_MODE != "t1")
    k_cfg = int(max(cfg.TOP_KS)) if cfg.TOP_KS else 1
    k_max = max(5, k_cfg)  # ensure we have ranks up to 5
    rows = []
    tid_col = cfg.THERAPIST_ID_COL
    for _, row in df_test.iterrows():
        tids = therapists[tid_col].astype(int).to_numpy()
        mu_all = outcome_predict_fn(row, tids)
        order = np.argsort(mu_all) if not hib else np.argsort(-mu_all)
        top_idx = order[:k_max]
        factual_tid = int(row[tid_col])
        in_topk = bool(factual_tid in tids[top_idx])
        for rnk, j in enumerate(top_idx, start=1):
            rows.append({
                cfg.PATIENT_ID_COL: row[cfg.PATIENT_ID_COL],
                "factual_tid": factual_tid,
                "rank": int(rnk),
                tid_col: int(tids[j]),
                "mu_hat": float(mu_all[j]),
                "is_factual_in_topk": in_topk,
            })
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# Aggregate per-therapist counts (top1 / top5)
# ------------------------------------------------------------------
def aggregate_recommendation_counts(recs_long: pd.DataFrame, cfg: MatchConfig) -> pd.DataFrame:
    tid = cfg.THERAPIST_ID_COL
    if recs_long.empty:
        return pd.DataFrame(columns=[tid, "top1_count", "top5_count", "n_test_patients", "top1_share", "top5_share"])
    n_pat = recs_long[cfg.PATIENT_ID_COL].nunique()
    agg = (
        recs_long
        .groupby(tid, as_index=False)
        .agg(
            top1_count=("rank", lambda s: int((s == 1).sum())),
            top5_count=("rank", lambda s: int((s <= 5).sum())),
        )
        .sort_values(["top1_count", "top5_count", tid], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    agg["n_test_patients"] = n_pat
    agg["top1_share"] = agg["top1_count"] / n_pat
    agg["top5_share"] = agg["top5_count"] / n_pat
    return agg

# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------
class MatchPipelineCB_Hier_DR:
    def __init__(self, cfg: MatchConfig):
        self.cfg = cfg
        self.df: Optional[pd.DataFrame] = None
        self.therapists_: Optional[pd.DataFrame] = None
        self.blend_: Optional[CatBoostBlendedRegressor] = None
        self.propensity_: Optional[Pipeline] = None
        self.label_map_: Dict[int, int] = {}
        self.class_to_tid_: Dict[int, int] = {}

    def set_data(self, df: pd.DataFrame) -> None:
        df = df.copy()
        # outcome orientation default
        if self.cfg.OUTCOME_HIGHER_IS_BETTER is None:
            self.cfg.OUTCOME_HIGHER_IS_BETTER = (self.cfg.TARGET_MODE != "t1")

        # parse date & create intake_* if missing
        if self.cfg.DATE_COL in df.columns and not pd.api.types.is_datetime64_any_dtype(df[self.cfg.DATE_COL]):
            df[self.cfg.DATE_COL] = pd.to_datetime(
                df[self.cfg.DATE_COL], dayfirst=self.cfg.DATE_DAYFIRST,
                format=self.cfg.DATE_FORMAT, errors="coerce"
            )
        if self.cfg.DATE_COL in df.columns:
            iso = df[self.cfg.DATE_COL].dt.isocalendar()
            if "intake_week" not in df.columns:
                df["intake_week"]  = iso["week"].astype("int")
            if "intake_month" not in df.columns:
                df["intake_month"] = df[self.cfg.DATE_COL].dt.month.astype("int")
            if "intake_year" not in df.columns:
                df["intake_year"]  = iso["year"].astype(str)

        self.df = df

    def _predict_mu_for_row(self, patient_row: pd.Series, tids: np.ndarray) -> np.ndarray:
        # Build dyads for these tids and predict with blended CatBoost
        ther_df = self.therapists_.set_index(self.cfg.THERAPIST_ID_COL).loc[tids].reset_index()
        p_cols = [self.cfg.PATIENT_ID_COL] + self.cfg.patient_numeric + self.cfg.patient_categorical
        p_cols = [c for c in p_cols if c in patient_row.index]
        p_rep = pd.DataFrame([patient_row[p_cols].values] * len(ther_df), columns=p_cols)
        p_rep.index = ther_df.index
        dyads = pd.concat([p_rep, ther_df], axis=1)
        return self.blend_.predict(dyads).astype(float)

    def run(self) -> Dict[str, pd.DataFrame]:
        if self.df is None:
            raise RuntimeError("Call set_data(df) first.")

        df_train, df_val, df_test = split_leave_last(self.df, self.cfg.THERAPIST_ID_COL, self.cfg.DATE_COL)
        logger.info("Therapists: %d | Train=%d | Val=%d | Test=%d",
                    self.df[self.cfg.THERAPIST_ID_COL].nunique(), len(df_train), len(df_val), len(df_test))

        # Therapist table
        self.therapists_ = therapists_table_from_train(self.cfg, df_train)

        # Fit blended CatBoost on train+val
        base_cols = _dedupe(self.cfg.patient_numeric + self.cfg.patient_categorical +
                            self.cfg.therapist_numeric + self.cfg.therapist_categorical +
                            [self.cfg.THERAPIST_ID_COL])
        X_trval = pd.concat([df_train[base_cols], df_val[base_cols]], ignore_index=True)
        y_trval = np.concatenate([df_train[self.cfg.LABEL_COL].to_numpy(),
                                  df_val[self.cfg.LABEL_COL].to_numpy()])
        self.blend_ = CatBoostBlendedRegressor(self.cfg).fit(
            X_trval, y_trval,
            params_m1=self.cfg.params_m1,
            params_m0=self.cfg.params_m0
        )

        # Propensity on patient features predicting therapist_id (train+val)
        p_num = self.cfg.propensity_patient_numeric or self.cfg.patient_numeric
        p_cat = self.cfg.propensity_patient_categorical or self.cfg.patient_categorical

        X_prop = pd.concat([df_train, df_val], ignore_index=True)
        y_tid  = X_prop[self.cfg.THERAPIST_ID_COL].astype(int).to_numpy()

        tids = np.sort(pd.unique(y_tid))
        self.label_map_    = {int(t): i for i, t in enumerate(tids)}
        self.class_to_tid_ = {i: int(t) for i, t in enumerate(tids)}
        y_cls = np.array([self.label_map_.get(int(t), -1) for t in y_tid], dtype=int)
        keep  = y_cls >= 0

        # guard against missing propensity columns
        missing = [c for c in p_num + p_cat if c not in X_prop.columns]
        if missing:
            logger.warning("Dropping missing propensity columns: %s", missing)
        p_num = [c for c in p_num if c in X_prop.columns]
        p_cat = [c for c in p_cat if c in X_prop.columns]

        prop = make_propensity_pipeline(self.cfg)
        prop.fit(X_prop.loc[keep, p_num + p_cat], y_cls[keep])
        self.propensity_ = prop

        # Sanity: factual RMSE on test
        X_test = df_test[base_cols].copy()
        y_test = df_test[self.cfg.LABEL_COL].to_numpy()
        yhat_factual = self.blend_.predict(X_test).astype(float)
        rmse_test = rmse(y_test, yhat_factual)

        # Off-policy evaluation (IPS/DR)
        eval_df = evaluate_policy_ips_dr(
            df_test=df_test, cfg=self.cfg, therapists=self.therapists_,
            outcome_predict_fn=self._predict_mu_for_row,
            propensity_model=self.propensity_,
            label_map=self.label_map_,
        )
        eval_df["rmse_factual"] = rmse_test

        # Top-k recommendations (long table) and aggregated counts
        recs_long = make_topk_recommendations(
            df_test=df_test, cfg=self.cfg, therapists=self.therapists_,
            outcome_predict_fn=self._predict_mu_for_row
        )
        reco_counts = aggregate_recommendation_counts(recs_long, self.cfg)

        # Example: first test patient (optional convenience)
        example = {}
        if len(df_test) > 0:
            row = df_test.iloc[0]
            tids_all = self.therapists_[self.cfg.THERAPIST_ID_COL].astype(int).to_numpy()
            mu_all = self._predict_mu_for_row(row, tids_all)
            hib = self.cfg.OUTCOME_HIGHER_IS_BETTER if self.cfg.OUTCOME_HIGHER_IS_BETTER is not None else (self.cfg.TARGET_MODE != "t1")
            order = np.argsort(mu_all) if not hib else np.argsort(-mu_all)
            for k in self.cfg.TOP_KS:
                idx = order[:k]
                example[f"top{k}"] = pd.DataFrame({
                    self.cfg.THERAPIST_ID_COL: tids_all[idx],
                    "mu_hat": mu_all[idx]
                }).reset_index(drop=True)

        return {
            "summary": eval_df,                               # IPS/DR policy values & uplift
            "therapists": self.therapists_,
            "topk_recommendations": recs_long,                # long table (rank<=max(5, max(TOP_KS)))
            "therapist_recommendation_counts": reco_counts,   # aggregated per-therapist counts/shares
            "example_first_test_patient": example,
        }

# ---- Runner
def run_pipeline(df: pd.DataFrame, cfg: MatchConfig):
    pipe = MatchPipelineCB_Hier_DR(cfg)
    pipe.set_data(df)
    return pipe.run()
