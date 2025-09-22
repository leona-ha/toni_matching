# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging, sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy import sparse
from collections import Counter

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.compose import TransformedTargetRegressor

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm
from scipy.stats import chi2
from pandas.api import types as pdt
import warnings

# ----- Torch FM -----
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ----- CatBoost -----
from catboost import CatBoostRegressor

#####################################################################
# Config
#####################################################################
@dataclass
class MatchConfig:
    # ---- Identifiers ----
    PATIENT_ID_COL: str = "patient_id"
    THERAPIST_ID_COL: str = "therapist_id"
    DATE_COL: str = "enrollment_date"
    DATE_DAYFIRST: bool = True
    DATE_FORMAT: Optional[str] = None  # e.g. "%d.%m.%Y"

    # ---- Outcome columns (if LABEL_COL not precomputed) ----
    PHQ8_T0_COL: Optional[str] = None
    PHQ8_T1_COL: Optional[str] = None
    GAD7_T0_COL: Optional[str] = None
    GAD7_T1_COL: Optional[str] = None

    # Target / outcome behavior
    TARGET_MODE: str = "delta"        # "delta" or "t1" (only used if LABEL_COL must be derived)
    STANDARDIZE_Y: bool = True
    AUTO_ORIENT_FROM_TARGET_MODE: bool = True  # auto set HIGHER_IS_BETTER from TARGET_MODE

    # Name & direction of the target actually used by the model
    LABEL_COL: str = "t1_distress_sum"
    OUTCOME_HIGHER_IS_BETTER: bool = False     # will be auto-set for t1/delta when AUTO_ORIENT...=True

    # ---- Features ----
    patient_numeric: List[str] = field(default_factory=list)
    patient_categorical: List[str] = field(default_factory=list)
    therapist_numeric: List[str] = field(default_factory=list)
    therapist_categorical: List[str] = field(default_factory=list)
    include_therapist_id_feature: bool = False

    # ---- Imputation ----
    impute_strategy: str = "iterative"   # "median" | "iterative"
    iterative_imputer_params: Dict[str, object] = field(default_factory=lambda: {
        "max_iter": 10, "initial_strategy": "median", "random_state": 0, "sample_posterior": True,
    })

    # ---- Models / HPO (RMSE for HPO) ----
    fm_param_grid: Dict[str, List] = field(default_factory=lambda: {
        "regressor__k": [8, 16, 32],
        "regressor__lr": [1e-3, 5e-4],
        "regressor__n_epochs": [40],
        "regressor__batch_size": [256],
        "regressor__weight_decay": [0.0, 1e-5],
    })
    ridge_param_grid: Dict[str, List] = field(default_factory=lambda: {
        "regressor__alpha": [1e-3, 1e-2, 1e-1, 1, 10, 1e2]
    })
    cat_param_grid: Dict[str, List] = field(default_factory=lambda: {
        "regressor__depth": [6, 8],
        "regressor__l2_leaf_reg": [3.0, 10.0],
        "regressor__learning_rate": [0.03, 0.06],
        "regressor__iterations": [400, 800],
    })
    SCORING_MODE: str = "rmse"     # keep RMSE for HPO

    # ---- Selection policy ----
    SELECT_BEST_BY: str = "test"  # "validation" | "test"

    # ---- Ordinal fitting options (for factual rank ~ outcome) ----
    ORDINAL_LINKS_TRY: List[str] = field(default_factory=lambda: ["logit"])
    ORDINAL_OPTIMIZERS_TRY: List[str] = field(default_factory=lambda: ["lbfgs", "bfgs", "newton"])
    ORDINAL_MAXITER: int = 400
    ORDINAL_TOL: float = 1e-6
    ORDINAL_STANDARDIZE_CHANGE: bool = True
    PO_MIN_POS: int = 3
    PO_MIN_NEG: int = 3
    PO_ALPHA: float = 0.05

    # ---- (optional) therapist recommendation tallies ----
    REPORT_TOP_COUNTS: bool = True

#####################################################################
# Logging
#####################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Match_Pipeline")

#####################################################################
# Helpers
#####################################################################
warnings.filterwarnings("ignore", category=FutureWarning,
    message=r"This Pipeline instance is not fitted yet.*")
NEG_LARGE = -1e12

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _dedupe(seq):
    return list(dict.fromkeys(seq))

def _ohe(sparse_output=True):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=(sparse_output))

def _y_eff(y, cfg: "MatchConfig"):
    y = np.asarray(y, float)
    return y if cfg.OUTCOME_HIGHER_IS_BETTER else -y

def _s_eff(a, cfg: "MatchConfig"):
    a = np.asarray(a, float)
    return a if cfg.OUTCOME_HIGHER_IS_BETTER else -a

def _pipeline_is_fitted(pipe: Pipeline) -> bool:
    """
    Permissive fitted-check that understands TransformedTargetRegressor and CatBoost.
    Used outside the scorer; the scorer itself now try/excepts prediction directly.
    """
    if not isinstance(pipe, Pipeline):
        return False
    pre = pipe.named_steps.get("preprocessor")
    reg = pipe.named_steps.get("regressor")
    if pre is None or reg is None:
        return False
    pre_ok = hasattr(pre, "transformers_")

    # Unwrap TTR if present
    inner = reg
    if isinstance(reg, TransformedTargetRegressor):
        inner = getattr(reg, "regressor_", None) or getattr(reg, "regressor", None)

    if inner is None:
        return False

    # CatBoost exposes is_fitted()
    if hasattr(inner, "is_fitted"):
        try:
            return pre_ok and bool(inner.is_fitted())
        except Exception:
            pass

    # Linear-like / Torch-FM signals
    if hasattr(inner, "coef_") or hasattr(inner, "n_features_in_"):
        return pre_ok
    if getattr(inner, "_model", None) is not None:
        return pre_ok

    # Fallback: presence of predict after fit
    return pre_ok and hasattr(inner, "predict")

def _model_name_from_estimator(estimator: Pipeline) -> str:
    reg = estimator.named_steps.get("regressor")
    if isinstance(reg, TransformedTargetRegressor):
        inner = getattr(reg, "regressor", None)
        return type(inner).__name__ if inner is not None else type(reg).__name__
    return type(reg).__name__ if reg is not None else type(estimator).__name__

#####################################################################
# Brant-like proportional-odds test (diagnostic)
#####################################################################
def brant_like_test(ranks: np.ndarray, outcome_eff: np.ndarray, min_pos=3, min_neg=3):
    r = np.asarray(ranks, int)
    y = np.asarray(outcome_eff, float)
    cuts = np.sort(np.unique(r))[:-1]
    betas, ses, used_cuts = [], [], []
    for c in cuts:
        ybin = (r <= c).astype(int)
        if ybin.sum() < min_pos or (len(ybin) - ybin.sum()) < min_neg: continue
        X = sm.add_constant(y)
        try:
            res = sm.Logit(ybin, X).fit(disp=False, maxiter=200)
            betas.append(res.params[1]); ses.append(res.bse[1]); used_cuts.append(int(c))
        except Exception:
            continue
    k = len(betas)
    if k < 2:
        return {"Q": np.nan, "df": 0, "p": np.nan, "per_cut": [], "note": "insufficient cutpoints"}
    w = 1.0 / (np.array(ses) ** 2)
    beta_bar = np.sum(w * np.array(betas)) / np.sum(w)
    Q = float(np.sum(w * (np.array(betas) - beta_bar) ** 2))
    df = k - 1
    p = float(1.0 - chi2.cdf(Q, df))
    per_cut = [{"cut": c, "beta": float(b), "se": float(s)} for c, b, s in zip(used_cuts, betas, ses)]
    return {"Q": Q, "df": df, "p": p, "per_cut": per_cut, "note": None}

#####################################################################
# Torch FM
#####################################################################
class _FMLayer(nn.Module):
    def __init__(self, n_feats: int, k: int):
        super().__init__()
        self.V = nn.Parameter(torch.randn(n_feats, k) * 0.01)
    def forward(self, x):
        xv  = x @ self.V
        xv2 = xv.pow(2)
        x2  = x.pow(2)
        v2  = self.V.pow(2)
        x2v2 = x2 @ v2
        return 0.5 * (xv2 - x2v2).sum(dim=1, keepdim=True)

class _FMTorch(nn.Module):
    def __init__(self, n_feats: int, k: int):
        super().__init__()
        self.linear = nn.Linear(n_feats, 1, bias=True)
        self.fm     = _FMLayer(n_feats, k)
    def forward(self, x):
        return self.linear(x) + self.fm(x)

class TorchFMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, *, k=16, lr=1e-3, n_epochs=40, batch_size=512,
                 weight_decay=0.0, device: Optional[str]=None, random_state=42, verbose=False):
        self.k, self.lr, self.n_epochs = k, lr, n_epochs
        self.batch_size, self.weight_decay = batch_size, weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state, self.verbose = random_state, verbose
        self._model, self.n_features_ = None, None
    def fit(self, X, y):
        if sparse.issparse(X): X = X.todense()
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        torch.manual_seed(self.random_state)
        self.n_features_ = X.shape[1]; self._model = _FMTorch(self.n_features_, self.k).to(self.device)
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        crit = nn.MSELoss()
        for epoch in range(self.n_epochs):
            self._model.train(); total = 0.0
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(); l = crit(self._model(xb), yb); l.backward(); opt.step()
                total += l.item() * len(xb)
        return self
    def predict(self, X):
        if sparse.issparse(X): X = X.todense()
        X = np.asarray(X, dtype=np.float32)
        self._model.eval()
        with torch.no_grad():
            return (self._model(torch.from_numpy(X).to(self.device)).cpu().numpy().ravel())
    def get_params(self, deep=True):
        return {"k": self.k, "lr": self.lr, "n_epochs": self.n_epochs,
                "batch_size": self.batch_size, "weight_decay": self.weight_decay,
                "device": self.device, "random_state": self.random_state, "verbose": self.verbose}
    def set_params(self, **params):
        for k, v in params.items(): setattr(self, k, v)
        return self

class RidgeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, random_state=42):
        self.alpha = alpha; self.random_state = random_state
        self._model = Ridge(alpha=self.alpha, random_state=self.random_state)
    def set_params(self, **params):
        if "alpha" in params: self.alpha = params["alpha"]
        self._model = Ridge(alpha=self.alpha, random_state=self.random_state); return self
    def get_params(self, deep=True):
        return {"alpha": self.alpha, "random_state": self.random_state}
    def fit(self, X, y): self._model.fit(X, y); return self
    def predict(self, X): return self._model.predict(X)

#####################################################################
# Split
#####################################################################
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

#####################################################################
# Preprocessors
#####################################################################
def make_preprocessor_common(
    cfg: MatchConfig,
    p_num: List[str], p_cat: List[str],
    t_num: List[str], t_cat: List[str],
    *, force_dense_ohe: bool = False
) -> ColumnTransformer:
    # numeric imputer
    if cfg.impute_strategy == "iterative":
        num_imputer = IterativeImputer(**cfg.iterative_imputer_params)
    elif cfg.impute_strategy == "median":
        num_imputer = SimpleImputer(strategy="median")
    else:
        raise ValueError(f"Unknown impute_strategy: {cfg.impute_strategy!r}")

    to_numeric = FunctionTransformer(lambda X: pd.DataFrame(X).apply(pd.to_numeric, errors="coerce"))

    transformers = []
    if p_num:
        transformers.append(("p_num", Pipeline([
            ("to_num", to_numeric), ("impute", num_imputer), ("scale", StandardScaler(with_mean=False)),
        ]), p_num))
    if p_cat:
        transformers.append(("p_cat", Pipeline([("onehot", _ohe(sparse_output=not force_dense_ohe))]), p_cat))
    if t_num:
        transformers.append(("t_num", Pipeline([
            ("to_num", to_numeric), ("impute", num_imputer), ("scale", StandardScaler(with_mean=False)),
        ]), t_num))
    if t_cat:
        transformers.append(("t_cat", Pipeline([("onehot", _ohe(sparse_output=not force_dense_ohe))]), t_cat))

    # Force dense for CatBoost (no scipy.sparse)
    sparse_threshold = 0.0 if force_dense_ohe else 0.3
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=sparse_threshold)

#####################################################################
# Dyad utils
#####################################################################
def therapists_table_from_train(cfg: MatchConfig, df_train: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [cfg.THERAPIST_ID_COL] + cfg.therapist_numeric + cfg.therapist_categorical
    keep_cols = [c for c in keep_cols if c in df_train.columns] + [cfg.THERAPIST_ID_COL]
    keep_cols = list(dict.fromkeys(keep_cols))
    return df_train[keep_cols].drop_duplicates(cfg.THERAPIST_ID_COL).reset_index(drop=True)

def build_dyads_for_patient(cfg: MatchConfig, patient_row: pd.Series, therapists_table: pd.DataFrame) -> pd.DataFrame:
    p_cols = [cfg.PATIENT_ID_COL] + cfg.patient_numeric + cfg.patient_categorical
    p_cols = [c for c in p_cols if c in patient_row.index]
    p_rep = pd.DataFrame([patient_row[p_cols].values] * len(therapists_table), columns=p_cols)
    p_rep.index = therapists_table.index
    return pd.concat([p_rep, therapists_table], axis=1)

def _feature_cols_from_pipe(estimator: Pipeline, cfg: "MatchConfig") -> List[str]:
    cols = []
    pre = estimator.named_steps.get("preprocessor")
    if pre is not None:
        for name, trans, sel in getattr(pre, "transformers", []):
            if name in {"p_num", "p_cat", "t_num", "t_cat"} and isinstance(sel, list):
                cols.extend(sel)
    cols = list(dict.fromkeys(cols))
    ids = []
    if cfg.PATIENT_ID_COL not in cols:
        ids.append(cfg.PATIENT_ID_COL)
    if cfg.THERAPIST_ID_COL not in cols:
        ids.append(cfg.THERAPIST_ID_COL)
    return cols + ids

#####################################################################
# Scorers (HPO uses RMSE)
#####################################################################
class RMSEScorer:
    def __call__(self, estimator: Pipeline, X_val: pd.DataFrame, y_val: np.ndarray) -> float:
        """
        Validation scorer for GridSearchCV.
        We do NOT pre-check 'fitted' anymore—just try to predict. If anything fails, return NEG_LARGE.
        """
        try:
            yhat = estimator.predict(X_val)
            return -float(np.sqrt(mean_squared_error(y_val, yhat)))  # higher is better
        except Exception as e:
            model_name = _model_name_from_estimator(estimator)
            print(f"[VAL ERROR][{model_name}] {type(e).__name__}: {e}")
            return NEG_LARGE

def make_validation_scorer(cfg: MatchConfig, therapists_table: pd.DataFrame):
    logger.info("scoring: using RMSE (negated) on observed dyads.")
    return RMSEScorer()

#####################################################################
# Grid helper (param nesting for TTR)
#####################################################################
def _maybe_nest_param_grid_for_ttr(grid: Dict[str, List], use_ttr: bool) -> Dict[str, List]:
    if not use_ttr: return grid
    return {k.replace("regressor__", "regressor__regressor__"): v for k, v in grid.items()}

#####################################################################
# Main Pipeline
#####################################################################
class MatchPipeline:
    def __init__(self, cfg: MatchConfig):
        self.cfg = cfg
        self.df: Optional[pd.DataFrame] = None
        self.fm_grid: Optional[GridSearchCV] = None
        self.ridge_grid: Optional[GridSearchCV] = None
        self.cat_grid: Optional[GridSearchCV] = None
        self.test_details_: Dict = {}

    def set_data(self, df: pd.DataFrame) -> None:
        df = df.copy()
        # Label compute if needed
        if self.cfg.LABEL_COL not in df.columns:
            required = [self.cfg.PHQ8_T0_COL, self.cfg.GAD7_T0_COL,
                        self.cfg.PHQ8_T1_COL, self.cfg.GAD7_T1_COL]
            if all(isinstance(c, str) and c in df.columns for c in required):
                mask_complete = df[[self.cfg.PHQ8_T1_COL, self.cfg.GAD7_T1_COL]].notna().all(axis=1)
                df = df.loc[mask_complete].reset_index(drop=True)
                t0_sum = df[self.cfg.PHQ8_T0_COL].astype(float) + df[self.cfg.GAD7_T0_COL].astype(float)
                t1_sum = df[self.cfg.PHQ8_T1_COL].astype(float) + df[self.cfg.GAD7_T1_COL].astype(float)
                tm = (self.cfg.TARGET_MODE or "delta").lower()
                df[self.cfg.LABEL_COL] = t1_sum if tm == "t1" else (t0_sum - t1_sum)
            else:
                raise ValueError("LABEL_COL missing and PHQ/GAD columns not provided.")

        # Auto orientation from TARGET_MODE
        if getattr(self.cfg, "AUTO_ORIENT_FROM_TARGET_MODE", True):
            tm = (self.cfg.TARGET_MODE or "delta").lower()
            self.cfg.OUTCOME_HIGHER_IS_BETTER = (tm != "t1")
            logger.info("orientation: TARGET_MODE=%s -> OUTCOME_HIGHER_IS_BETTER=%s",
                        tm, self.cfg.OUTCOME_HIGHER_IS_BETTER)

        # Parse date
        if self.cfg.DATE_COL in df.columns and not pdt.is_datetime64_any_dtype(df[self.cfg.DATE_COL]):
            logger.info("date: parsing '%s' (dayfirst=%s, format=%s)",
                        self.cfg.DATE_COL, self.cfg.DATE_DAYFIRST, self.cfg.DATE_FORMAT)
            df[self.cfg.DATE_COL] = pd.to_datetime(
                df[self.cfg.DATE_COL],
                dayfirst=self.cfg.DATE_DAYFIRST,
                format=self.cfg.DATE_FORMAT,
                errors="coerce",
            )
        self.df = df

    def _build_feature_lists_for_use(
        self, df_train: pd.DataFrame, df_val: pd.DataFrame, therapists_table: pd.DataFrame
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        p_num = list(self.cfg.patient_numeric)
        p_cat = list(self.cfg.patient_categorical)
        t_num = list(self.cfg.therapist_numeric)
        t_cat = list(self.cfg.therapist_categorical)
        if self.cfg.include_therapist_id_feature and (self.cfg.THERAPIST_ID_COL not in t_cat):
            t_cat.append(self.cfg.THERAPIST_ID_COL)
        trval_cols = set(df_train.columns).union(df_val.columns)
        if len(df_train) == 0 or len(therapists_table) == 0:
            raise ValueError("Empty train or therapist table.")
        sample_patient = df_train.iloc[0]
        dyad_sample = build_dyads_for_patient(self.cfg, sample_patient, therapists_table.iloc[[0]])
        dyad_cols = set(dyad_sample.columns)

        def keep(cols):
            usable = [c for c in cols if (c in trval_cols) and (c in dyad_cols)]
            dropped = [c for c in cols if c not in usable]
            if dropped:
                logger.warning("Dropping features not consistently available: %s", dropped)
            return usable

        return keep(p_num), keep(p_cat), keep(t_num), keep(t_cat)

    def _make_full_pipeline(self, p_num, p_cat, t_num, t_cat, model: BaseEstimator, *, force_dense=False) -> Pipeline:
        pre = make_preprocessor_common(self.cfg, p_num, p_cat, t_num, t_cat, force_dense_ohe=force_dense)
        reg: BaseEstimator = model
        if getattr(self.cfg, "STANDARDIZE_Y", False):
            reg = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
        return Pipeline([("preprocessor", pre), ("regressor", reg)])

    def _evaluate_on_test(self, name: str, estimator: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray,
                          df_test: pd.DataFrame, ther_table: pd.DataFrame, val_score: float) -> Dict:
        # Factual predictions
        y_pred_obs = estimator.predict(X_test).astype(float)
        test_rmse = rmse(y_test, y_pred_obs)

        # Per-patient factual rank among all therapists, then ordinal regression: rank ~ outcome
        ranks, top1_tids, top5_tids_all = [], [], []
        tid_col = self.cfg.THERAPIST_ID_COL
        raw_cols = _feature_cols_from_pipe(estimator, self.cfg)

        for _, row in df_test.iterrows():
            dyads = build_dyads_for_patient(self.cfg, row, ther_table)
            dyads = dyads.reindex(columns=raw_cols + [c for c in dyads.columns if c not in raw_cols], fill_value=np.nan)
            preds = estimator.predict(dyads).astype(float)
            scores_eff = _s_eff(preds, self.cfg)
            tids  = ther_table[tid_col].astype(int).to_numpy()
            order = np.argsort(-scores_eff)
            tids_ordered = tids[order]
            top1_tids.append(int(tids_ordered[0]))
            top5_tids_all.extend([int(t) for t in tids_ordered[:5]])
            factual = int(row[tid_col])
            pos = np.where(tids_ordered == factual)[0]
            ranks.append(int(pos[0]) + 1 if len(pos) else len(tids_ordered) + 1)

        ranks = np.asarray(ranks, int)
        yy_test = _y_eff(y_test, self.cfg)

        # Ordinal regression (no binning): rank ~ outcome
        llf = aic = bic = ll_per_case = pseudo_r2 = np.nan
        coef_change = odds_ratio_change = coef_change_std = odds_ratio_change_std = np.nan
        used_link = used_opt = None; sd_used = np.nan

        if len(df_test) > 1:
            y_for_fit = yy_test.copy()
            if self.cfg.ORDINAL_STANDARDIZE_CHANGE:
                sd = float(np.nanstd(y_for_fit))
                if sd > 0:
                    y_for_fit = (y_for_fit - float(np.nanmean(y_for_fit))) / sd
                    sd_used = sd
            exog_name = self.cfg.LABEL_COL
            for link in (self.cfg.ORDINAL_LINKS_TRY or ["logit"]):
                for method in (self.cfg.ORDINAL_OPTIMIZERS_TRY or ["lbfgs"]):
                    try:
                        endog = pd.Series(ranks, name="rank")
                        exog  = pd.DataFrame({exog_name: y_for_fit})
                        mdl_t = OrderedModel(endog, exog, distr=link)
                        fit_kwargs = dict(method=method, disp=False, maxiter=self.cfg.ORDINAL_MAXITER)
                        if method != "lbfgs": fit_kwargs["tol"] = self.cfg.ORDINAL_TOL
                        res_t = mdl_t.fit(**fit_kwargs)
                        llf = float(res_t.llf)
                        aic = float(getattr(res_t, "aic", np.nan))
                        bic = float(getattr(res_t, "bic", np.nan))
                        ll_per_case = llf / len(df_test)
                        c = float(res_t.params.get(exog_name, np.nan))
                        coef_change_std = c
                        odds_ratio_change_std = float(np.exp(c)) if np.isfinite(c) else np.nan
                        if np.isfinite(sd_used) and sd_used > 0:
                            coef_change = c / sd_used
                            odds_ratio_change = float(np.exp(coef_change)) if np.isfinite(coef_change) else np.nan
                        else:
                            coef_change = c
                            odds_ratio_change = odds_ratio_change_std
                        # pseudo-R2 vs null
                        try:
                            null = OrderedModel(endog, pd.DataFrame({"const": np.ones_like(ranks)}), distr=link)
                            res_null = null.fit(**fit_kwargs)
                            pseudo_r2 = 1.0 - (llf / float(res_null.llf))
                        except Exception:
                            pseudo_r2 = np.nan
                        used_link, used_opt = link, method
                        break
                    except Exception:
                        continue
                if used_link is not None:
                    break

        # Therapist recommendation tallies (optional)
        ther_reco_counts = pd.DataFrame()
        if self.cfg.REPORT_TOP_COUNTS:
            c_top1 = Counter(top1_tids); c_top5 = Counter(top5_tids_all)
            ther_ids = ther_table[tid_col].astype(int).tolist()
            ther_reco_counts = pd.DataFrame({tid_col: ther_ids})
            n_pat = len(df_test)
            ther_reco_counts["top1_count"] = ther_reco_counts[tid_col].map(c_top1).fillna(0).astype(int)
            ther_reco_counts["top5_count"] = ther_reco_counts[tid_col].map(c_top5).fillna(0).astype(int)
            ther_reco_counts["n_test_patients"] = n_pat
            ther_reco_counts["top1_share"] = ther_reco_counts["top1_count"] / max(n_pat, 1)
            ther_reco_counts["top5_share"] = ther_reco_counts["top5_count"] / max(n_pat, 1)
            ther_reco_counts = ther_reco_counts.sort_values(
                ["top1_count","top5_count",tid_col], ascending=[False, False, True]
            ).reset_index(drop=True)

        # convert negative scorer back to val_rmse; mark failure when scorer returned sentinel/invalid
        val_rmse = -float(val_score) if (np.isfinite(val_score) and (val_score > NEG_LARGE/2)) else np.inf
        val_failed = not np.isfinite(val_score) or (val_score <= NEG_LARGE/2)

        row = {
            "model": name,
            "val_rmse": val_rmse,
            "val_failed": bool(val_failed),
            "test_rmse": float(test_rmse),
            "ord_ll": llf,
            "ord_ll_per_case": ll_per_case,
            "ord_pseudo_r2": pseudo_r2,
            "ord_coef_change_std": coef_change_std,
            "ord_or_change_std": odds_ratio_change_std,
            "ord_link": used_link,
            "ord_optimizer": used_opt,
            "ord_sd_used": sd_used,
        }

        details = {
            "ranks": ranks, "y_test": y_test, "y_pred_obs": y_pred_obs,
            "therapist_recommendation_counts": ther_reco_counts,
        }
        return {"row": row, "details": details}

    def run(self) -> Dict[str, pd.DataFrame]:
        if self.df is None:
            raise RuntimeError("Call set_data(df) first.")

        df_train, df_val, df_test = split_leave_last(self.df, self.cfg.THERAPIST_ID_COL, self.cfg.DATE_COL)
        logger.info("Therapists: %d | Train: %d | Val: %d | Test: %d",
                    self.df[self.cfg.THERAPIST_ID_COL].nunique(), len(df_train), len(df_val), len(df_test))

        ther_table = therapists_table_from_train(self.cfg, df_train)
        logger.info("candidates: %d therapists in train", len(ther_table))

        # Build feature lists
        p_num_u, p_cat_u, t_num_u, t_cat_u = self._build_feature_lists_for_use(df_train, df_val, ther_table)
        if not t_num_u and not t_cat_u:
            logger.warning("No therapist features in model. Consider include_therapist_id_feature=True.")

        base_feats = p_num_u + p_cat_u + t_num_u + t_cat_u
        extra_ids = [self.cfg.PATIENT_ID_COL]
        if self.cfg.THERAPIST_ID_COL not in base_feats:
            extra_ids.append(self.cfg.THERAPIST_ID_COL)
        tr_cols = _dedupe(base_feats + extra_ids)

        X_train = df_train[tr_cols].copy()
        y_train = df_train[self.cfg.LABEL_COL].to_numpy()
        X_val   = df_val[tr_cols].copy()
        y_val   = df_val[self.cfg.LABEL_COL].to_numpy()
        logger.info("matrices: X_train=%s, X_val=%s", tuple(X_train.shape), tuple(X_val.shape))

        X_trval = pd.concat([X_train, X_val], ignore_index=True)
        y_trval = np.concatenate([y_train, y_val])
        test_fold = np.r_[np.full(len(X_train), -1), np.zeros(len(X_val), dtype=int)]
        cv = PredefinedSplit(test_fold)

        # Pipelines (note: CatBoost needs dense; FM/Ridge can handle sparse)
        fm_pipe    = self._make_full_pipeline(p_num_u, p_cat_u, t_num_u, t_cat_u, TorchFMRegressor(), force_dense=False)
        ridge_pipe = self._make_full_pipeline(p_num_u, p_cat_u, t_num_u, t_cat_u, RidgeRegressor(), force_dense=False)
        cat_pipe   = self._make_full_pipeline(
            p_num_u, p_cat_u, t_num_u, t_cat_u,
            CatBoostRegressor(loss_function="RMSE", random_seed=42, verbose=False),
            force_dense=True
        )

        val_scorer = make_validation_scorer(self.cfg, ther_table)
        use_ttr = bool(getattr(self.cfg, "STANDARDIZE_Y", False))
        fm_grid     = _maybe_nest_param_grid_for_ttr(self.cfg.fm_param_grid,    use_ttr)
        ridge_grid  = _maybe_nest_param_grid_for_ttr(self.cfg.ridge_param_grid, use_ttr)
        cat_grid    = _maybe_nest_param_grid_for_ttr(self.cfg.cat_param_grid,   use_ttr)

        fm_gs    = GridSearchCV(fm_pipe,    fm_grid,   scoring=val_scorer, refit=True, cv=cv, n_jobs=1, verbose=0, error_score=NEG_LARGE)
        ridge_gs = GridSearchCV(ridge_pipe, ridge_grid, scoring=val_scorer, refit=True, cv=cv, n_jobs=1, verbose=0, error_score=NEG_LARGE)
        cat_gs   = GridSearchCV(cat_pipe,   cat_grid,  scoring=val_scorer, refit=True, cv=cv, n_jobs=1, verbose=0, error_score=NEG_LARGE)

        fm_gs.fit(X_trval, y_trval)
        ridge_gs.fit(X_trval, y_trval)
        cat_gs.fit(X_trval, y_trval)

        # Collect candidates
        candidates = [
            ("fm",    fm_gs.best_estimator_,    fm_gs.best_score_,    fm_gs),
            ("ridge", ridge_gs.best_estimator_, ridge_gs.best_score_, ridge_gs),
            ("cat",   cat_gs.best_estimator_,   cat_gs.best_score_,   cat_gs),
        ]
        cand_map = {name: (est, score, gs) for (name, est, score, gs) in candidates}

        # Test matrices
        test_cols = tr_cols
        X_test = df_test[test_cols].copy()
        y_test = df_test[self.cfg.LABEL_COL].to_numpy()

        # Evaluate all models
        results_rows = []; details_by_model = {}; therapist_reco_by_model = {}
        for name, est, val_score, gs in candidates:
            eval_out = self._evaluate_on_test(name, est, X_test, y_test, df_test, ther_table, val_score)
            results_rows.append(eval_out["row"])
            details_by_model[name] = eval_out["details"]
            therapist_reco_by_model[name] = eval_out["details"].get("therapist_recommendation_counts", pd.DataFrame())

        # Choose best model according to policy
        df_sum = pd.DataFrame(results_rows)
        if "val_failed" not in df_sum.columns:
            df_sum["val_failed"] = ~np.isfinite(df_sum.get("val_rmse", np.inf))

        if (self.cfg.SELECT_BEST_BY or "validation").lower() == "test":
            best_name = df_sum.loc[df_sum["test_rmse"].idxmin(), "model"]
        else:
            valid = df_sum[~df_sum["val_failed"]].copy()
            if len(valid):
                best_name = valid.loc[valid["val_rmse"].idxmin(), "model"]
            else:
                best_name = df_sum.loc[df_sum["test_rmse"].idxmin(), "model"]

        best_model, _, best_gs = cand_map[best_name][0], cand_map[best_name][1], cand_map[best_name][2]
        logger.info("[Selected] %s | policy=%s", best_name, self.cfg.SELECT_BEST_BY)

        # Sort summary (valid val first, then val_rmse, then test_rmse)
        df_sum["val_rmse_sort"] = df_sum["val_rmse"].replace(np.inf, np.nan)
        df_sum = df_sum.sort_values(
            by=["val_failed", "val_rmse_sort", "test_rmse"],
            ascending=[True, True, True]
        ).drop(columns=["val_rmse_sort"]).reset_index(drop=True)

        # Example: scores for the first test patient (best model)
        example_scores = pd.DataFrame()
        if len(df_test) > 0:
            try:
                example_scores = build_dyads_for_patient(self.cfg, df_test.iloc[0], ther_table)
                raw_cols = _feature_cols_from_pipe(best_model, self.cfg)
                example_scores = example_scores[raw_cols]
                example_scores["predicted_outcome"] = best_model.predict(example_scores).astype(float)
                example_scores = example_scores[[self.cfg.THERAPIST_ID_COL, "predicted_outcome"]].sort_values(
                    "predicted_outcome", ascending=not self.cfg.OUTCOME_HIGHER_IS_BETTER).reset_index(drop=True)
            except Exception as e:
                print(f"[Example scores] {type(e).__name__}: {e}")

        self.test_details_ = {
            "models": details_by_model,
            "therapist_recommendation_counts_by_model": therapist_reco_by_model,
            "used_features": {
                "patient_numeric": p_num_u, "patient_categorical": p_cat_u,
                "therapist_numeric": t_num_u, "therapist_categorical": t_cat_u,
            },
            "best_model_name": best_name,
            "best_params": dict(best_gs.best_params_),
            "example_scores_first_test_patient": example_scores,
        }
        ther_table_out = therapists_table_from_train(self.cfg, df_train)
        best_reco = therapist_reco_by_model.get(best_name, pd.DataFrame())
        return {"summary": df_sum, "therapists": ther_table_out, "therapist_recommendations": best_reco}

# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_pipeline(df: pd.DataFrame, cfg: MatchConfig):
    pipe = MatchPipeline(cfg)
    pipe.set_data(df)
    out = pipe.run()
    out["used_features"] = pipe.test_details_.get("used_features", {})
    out["best_model_name"] = pipe.test_details_.get("best_model_name")
    out["best_params"] = pipe.test_details_.get("best_params")
    out["example_scores_first_test_patient"] = pipe.test_details_.get("example_scores_first_test_patient", pd.DataFrame())
    return out
