# -*- coding: utf-8 -*-
"""
EBM (GA2M) attribute-only therapist recommender with IPS/DR policy evaluation.
- No therapist_id as a feature; learns patient×therapist-attribute interactions.
- Per-therapist time-based split (leave last / penultimate).
- Numeric imputation: 'iterative' (IterativeImputer) or 'median'; categorical: most_frequent.
- Off-policy metrics only: IPS and Doubly-Robust (DR) policy value and uplift vs. status quo.
"""

import numpy as np
import pandas as pd
import logging, sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state

from interpret.glassbox import ExplainableBoostingRegressor

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Match_EBM_DR")

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

    # EBM hyperparams
    ebm_params: Dict[str, object] = field(default_factory=lambda: dict(
        interactions=50, max_bins=128, learning_rate=0.02, validation_size=0.2,
        early_stopping=True, random_state=42
    ))

    # Policy evaluation
    TOP_KS: List[int] = field(default_factory=lambda: [1, 3, 5])
    MIN_PROPENSITY: float = 1e-6       # floor to stabilize IPS/DR
    BOOTSTRAP_REPS: int = 0            # set >0 for CIs (e.g., 200). 0 = no bootstrap

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _dedupe(seq): return list(dict.fromkeys(seq))

def _ohe_dense():
    # EBM expects dense; keep OneHot dense
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def _y_eff(y: np.ndarray, cfg: MatchConfig) -> np.ndarray:
    hib = cfg.OUTCOME_HIGHER_IS_BETTER if cfg.OUTCOME_HIGHER_IS_BETTER is not None else (cfg.TARGET_MODE != "t1")
    y = np.asarray(y, float)
    return y if hib else -y

def rmse(y_true, y_pred): return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ------------------------------------------------------------------
# Split (per-therapist, time-based)
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
# Preprocessors
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
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.1)

def make_propensity_pipeline(cfg: MatchConfig) -> Pipeline:
    # Multinomial logistic on patient features ONLY (no therapist attrs to avoid class-conditional leakage).
    if cfg.impute_strategy == "iterative":
        num_imputer = IterativeImputer(**cfg.iterative_imputer_params)
    else:
        num_imputer = SimpleImputer(strategy="median")

    to_numeric = FunctionTransformer(lambda X: pd.DataFrame(X).apply(pd.to_numeric, errors="coerce"))
    cat_imputer = SimpleImputer(strategy="most_frequent")

    num_cols = cfg.patient_numeric
    cat_cols = cfg.patient_categorical

    pre = ColumnTransformer([
        ("num", Pipeline([("to_num", to_numeric), ("impute", num_imputer), ("scale", StandardScaler(with_mean=False))]), num_cols),
        ("cat", Pipeline([("impute", cat_imputer), ("ohe", _ohe_dense())]), cat_cols),
    ], remainder="drop", sparse_threshold=1.1)

    clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", max_iter=1000, C=1.0, n_jobs=None
    )
    return Pipeline([("pre", pre), ("clf", clf)])

# ------------------------------------------------------------------
# Model wrapper (EBM inside a sklearn Pipeline)
# ------------------------------------------------------------------
class EBMMatcher:
    def __init__(self, cfg: MatchConfig):
        self.cfg = cfg
        self.pipe_: Optional[Pipeline] = None
        self.therapists_: Optional[pd.DataFrame] = None
        self.propensity_: Optional[Pipeline] = None
        self.label_map_: Dict[int, int] = {}   # therapist_id -> class index
        self.class_to_tid_: Dict[int, int] = {}# class index -> therapist_id

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame):
        # Build therapist candidate table from TRAIN
        self.therapists_ = therapists_table_from_train(self.cfg, df_train)

        # Feature pipe + EBM
        pre = make_feature_preprocessor(self.cfg)
        ebm = ExplainableBoostingRegressor(**self.cfg.ebm_params)
        self.pipe_ = Pipeline([("pre", pre), ("ebm", ebm)])

        # Train on observed dyads (train+val)
        base_cols = _dedupe(
            self.cfg.patient_numeric + self.cfg.patient_categorical +
            self.cfg.therapist_numeric + self.cfg.therapist_categorical
        )
        tr = df_train[base_cols].copy(); ytr = df_train[self.cfg.LABEL_COL].to_numpy()
        va = df_val[base_cols].copy();   yva = df_val[self.cfg.LABEL_COL].to_numpy()
        X_trval = pd.concat([tr, va], ignore_index=True)
        y_trval = np.concatenate([ytr, yva])

        self.pipe_.fit(X_trval, y_trval)

        # Propensity: multinomial logit on patient features predicting therapist_id
        prop = make_propensity_pipeline(self.cfg)
        y_trval_tid = pd.concat([df_train, df_val], ignore_index=True)[self.cfg.THERAPIST_ID_COL].astype(int).to_numpy()

        # Map therapist_id -> class index (0..C-1)
        tids = np.sort(pd.unique(y_trval_tid))
        self.label_map_ = {int(t): i for i, t in enumerate(tids)}
        self.class_to_tid_ = {i: int(t) for i, t in enumerate(tids)}
        y_classes = np.array([self.label_map_.get(int(t), -1) for t in y_trval_tid], dtype=int)
        keep = y_classes >= 0
        prop.fit(pd.concat([df_train, df_val], ignore_index=True).loc[keep, self.cfg.patient_numeric + self.cfg.patient_categorical],
                 y_classes[keep])
        self.propensity_ = prop

    # Predict mu(x, t) for all candidate therapists for a given patient row
    def _predict_all(self, patient_row: pd.Series) -> pd.DataFrame:
        dyads = build_dyads_for_patient(self.cfg, patient_row, self.therapists_)
        base_cols = _dedupe(
            self.cfg.patient_numeric + self.cfg.patient_categorical +
            self.cfg.therapist_numeric + self.cfg.therapist_categorical
        )
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

    # Propensity π_hat(t | x)
    def _propensity_probs(self, patient_row: pd.Series) -> Tuple[np.ndarray, List[int]]:
        Xp = pd.DataFrame([patient_row[self.cfg.patient_numeric + self.cfg.patient_categorical].to_dict()])
        prob = self.propensity_.predict_proba(Xp)[0]  # shape (C,)
        classes = list(range(len(prob)))
        return prob, classes

# ------------------------------------------------------------------
# DR / IPS policy evaluation
# ------------------------------------------------------------------
def evaluate_policy_ips_dr(matcher: EBMMatcher,
                           df_test: pd.DataFrame,
                           cfg: MatchConfig) -> pd.DataFrame:
    """
    Compute IPS and DR estimates for top-k policies (uniform random among top-k).
    Returns a tidy DataFrame with columns:
      ['k', 'value_ips', 'value_dr', 'uplift_dr', 'uplift_ips',
       'value_model_mean', 'observed_mean', 'ess_ips']
    """
    rng = check_random_state(42)
    y = df_test[cfg.LABEL_COL].to_numpy()
    y_eff = _y_eff(y, cfg)
    observed_mean = float(np.mean(y_eff))  # status-quo value

    results = []

    # Pre-extract patient features into rows for speed
    for k in cfg.TOP_KS:
        v_ips = []; v_dr = []
        mu_bar_list = []; w_list = []

        for _, row in df_test.iterrows():
            # Top-k set and its mean mu (uniform mixture policy)
            topk = matcher.recommend_topk(row, k)
            mu_topk = topk["mu_hat"].to_numpy()
            mu_bar = float(np.mean(mu_topk)) if len(mu_topk) else 0.0

            # Patient propensities over *seen* therapist classes
            probs, classes = matcher._propensity_probs(row)
            # Map top-k tids to class indices; sum propensities over the set
            pi_S = 0.0
            for tid in topk[cfg.THERAPIST_ID_COL].astype(int).tolist():
                cls = matcher.label_map_.get(int(tid), None)
                if cls is not None and cls < len(probs):
                    pi_S += float(probs[cls])

            pi_S = max(pi_S, cfg.MIN_PROPENSITY)

            # Factual therapist & mu(x, factual)
            factual_tid = int(row[cfg.THERAPIST_ID_COL])
            factual_in_S = int(factual_tid in set(topk[cfg.THERAPIST_ID_COL].astype(int).tolist()))
            # mu(x, t_factual): get via matcher._predict_all once (cheap to recompute here)
            mu_all = matcher._predict_all(row)
            mu_t_f = float(mu_all.loc[mu_all[cfg.THERAPIST_ID_COL] == factual_tid, "mu_hat"].values[0])

            w = float(factual_in_S / pi_S)  # IPS weight for set-valued policy
            w_list.append(w)
            # Effective outcome orientation
            yi = float(_y_eff(np.array([row[cfg.LABEL_COL]]), cfg)[0])

            # IPS and DR contributions
            v_ips.append(w * yi)
            v_dr.append(mu_bar + w * (yi - mu_t_f))
            mu_bar_list.append(mu_bar)

        # Aggregate
        value_ips = float(np.mean(v_ips)) if v_ips else np.nan
        value_dr  = float(np.mean(v_dr))  if v_dr  else np.nan
        value_model_mean = float(np.mean(mu_bar_list)) if mu_bar_list else np.nan

        # ESS for IPS weights
        w = np.asarray(w_list, float)
        ess = float((w.sum() ** 2) / np.maximum((w ** 2).sum(), 1e-12)) if w.size else np.nan

        results.append({
            "k": int(k),
            "value_ips": value_ips,
            "value_dr": value_dr,
            "uplift_ips": value_ips - observed_mean,
            "uplift_dr": value_dr - observed_mean,
            "value_model_mean": value_model_mean,
            "observed_mean": observed_mean,
            "ess_ips": ess,
        })

    return pd.DataFrame(results)

# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------
class MatchPipelineEBM_DR:
    def __init__(self, cfg: MatchConfig):
        self.cfg = cfg
        self.df: Optional[pd.DataFrame] = None
        self.matcher_: Optional[EBMMatcher] = None
        self.ther_table_: Optional[pd.DataFrame] = None

    def set_data(self, df: pd.DataFrame) -> None:
        df = df.copy()
        if self.cfg.OUTCOME_HIGHER_IS_BETTER is None:
            self.cfg.OUTCOME_HIGHER_IS_BETTER = (self.cfg.TARGET_MODE != "t1")
        # Parse date
        if self.cfg.DATE_COL in df.columns and not np.issubdtype(df[self.cfg.DATE_COL].dtype, np.datetime64):
            df[self.cfg.DATE_COL] = pd.to_datetime(
                df[self.cfg.DATE_COL], dayfirst=self.cfg.DATE_DAYFIRST, format=self.cfg.DATE_FORMAT, errors="coerce"
            )
        self.df = df

    def run(self) -> Dict[str, pd.DataFrame]:
        if self.df is None: raise RuntimeError("Call set_data(df) first.")
        df_train, df_val, df_test = split_leave_last(self.df, self.cfg.THERAPIST_ID_COL, self.cfg.DATE_COL)
        logger.info("Therapists: %d | Train=%d | Val=%d | Test=%d",
                    self.df[self.cfg.THERAPIST_ID_COL].nunique(), len(df_train), len(df_val), len(df_test))

        self.matcher_ = EBMMatcher(self.cfg)
        self.matcher_.fit(df_train, df_val)

        # Factual RMSE on test (optional sanity check)
        base_cols = _dedupe(self.cfg.patient_numeric + self.cfg.patient_categorical +
                            self.cfg.therapist_numeric + self.cfg.therapist_categorical)
        y_test = df_test[self.cfg.LABEL_COL].to_numpy()
        y_pred_factual = self.matcher_.pipe_.predict(df_test[base_cols]).astype(float)
        rmse_test = rmse(y_test, y_pred_factual)

        # Off-policy evaluation (IPS/DR)
        eval_df = evaluate_policy_ips_dr(self.matcher_, df_test, self.cfg)
        eval_df["rmse_factual"] = rmse_test

        # Example: top-k recommendations for first test patient (for inspection)
        example = {}
        if len(df_test) > 0:
            row = df_test.iloc[0]
            for k in self.cfg.TOP_KS:
                ex = self.matcher_.recommend_topk(row, k)
                example[f"top{k}"] = ex

        return {
            "summary": eval_df,                 # ONLY IPS/DR policy values & uplift
            "therapists": self.matcher_.therapists_,
            "example_first_patient": example,
        }

# ---- Runner
def run_pipeline(df: pd.DataFrame, cfg: MatchConfig):
    pipe = MatchPipelineEBM_DR(cfg); pipe.set_data(df); return pipe.run()
