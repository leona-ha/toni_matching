# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy import sparse
from scipy.stats import spearmanr, kendalltau
from collections import Counter

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

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
    TARGET_MODE: str = "t1"          # "delta" or "t1" (used only if LABEL_COL missing and we need to derive)
    STANDARDIZE_Y: bool = True         # if True, scale y during fit and inverse at predict

    # Name & direction of the target actually used by the model
    LABEL_COL: str = "t1_distress_sum"
    OUTCOME_HIGHER_IS_BETTER: bool = False

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

    # ---- Models / HPO ----
    fm_param_grid: Dict[str, List] = field(default_factory=lambda: {
        "regressor__k": [4, 8, 16, 32],
        "regressor__lr": [1e-3, 5e-4],
        "regressor__n_epochs": [40],
        "regressor__batch_size": [256],
        "regressor__weight_decay": [0.0, 1e-5],
    })
    ridge_param_grid: Dict[str, List] = field(default_factory=lambda: {
        "regressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3]
    })
    SCORING_MODE: str = "rmse"     # "rmse" | "ordinal" | "spearman"

    # ---- Ordinal fitting options (no binning for primary) ----
    ORDINAL_LINKS_TRY: List[str] = field(default_factory=lambda: ["logit"])
    ORDINAL_OPTIMIZERS_TRY: List[str] = field(default_factory=lambda: ["lbfgs", "bfgs", "newton"])
    ORDINAL_MAXITER: int = 400
    ORDINAL_TOL: float = 1e-6
    ORDINAL_STANDARDIZE_CHANGE: bool = True
    ORDINAL_REQUIRE_NON_NAN_PRIMARY: bool = False

    # ---- Brant-like proportional-odds diagnostic ----
    PO_MIN_POS: int = 3
    PO_MIN_NEG: int = 3
    PO_ALPHA: float = 0.05

    # ---- PPO sensitivity (disabled by default) ----
    PPO_SENSITIVITY: bool = False
    PPO_SENS_MAX_CATS: int = 12

    # ---- Top-k metrics ----
    HIT_KS: List[int] = field(default_factory=lambda: [1, 5])
    NDCG_KS: List[int] = field(default_factory=lambda: [5])

    # ---- Uplift re-ranking (NEW) ----
    UPLIFT_RERANK: bool = False
    UPLIFT_ALPHA_GRID: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75])
    UPLIFT_TOPM: Optional[int] = None   # e.g., 20 => re-rank only top-20 by raw preds
    USE_UPLIFT_FOR_PRIMARY: bool = False


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
# Helpers & Metrics
#####################################################################
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def ndcg_at_k_from_rank(rank: int, k: Optional[int] = None) -> float:
    if k is not None and rank > k:
        return 0.0
    return 1.0 / np.log2(rank + 1.0)

def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, float); s = np.asarray(y_pred, float)
    mask = np.isfinite(y) & np.isfinite(s)
    y, s = y[mask], s[mask]
    n = len(y); num = den = 0
    for i in range(n):
        yi, si = y[i], s[i]
        for j in range(i + 1, n):
            yj, sj = y[j], s[j]
            if yi == yj:
                continue
            den += 1
            if (si - sj) * (yi - yj) > 0:
                num += 1
    return (num / den) if den else np.nan

def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        for c in missing:
            df[c] = 0.0
    return df

def _ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)

def _dedupe(seq):
    return list(dict.fromkeys(seq))

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

def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 2 or b.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    r = spearmanr(a, b, nan_policy="omit").correlation
    return float(r) if np.isfinite(r) else 0.0

def _safe_kendall(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return np.nan
    try:
        t = kendalltau(a, b).correlation
        return float(t) if np.isfinite(t) else np.nan
    except Exception:
        return np.nan

def _y_eff(y, cfg: "MatchConfig"):
    """Transform the outcome so that 'higher is better' within ranking/ordinal diagnostics."""
    y = np.asarray(y, float)
    return y if cfg.OUTCOME_HIGHER_IS_BETTER else -y

warnings.filterwarnings(
    "ignore", category=FutureWarning,
    message=r"This Pipeline instance is not fitted yet.*"
)

NEG_LARGE = -1e12

def _pipeline_is_fitted(pipe: Pipeline) -> bool:
    if not isinstance(pipe, Pipeline):
        return False
    pre = pipe.named_steps.get("preprocessor")
    reg = pipe.named_steps.get("regressor")
    if pre is None or reg is None:
        return False
    pre_ok = hasattr(pre, "transformers_")

    # handle plain estimators, TorchFM, and TransformedTargetRegressor
    if isinstance(reg, TransformedTargetRegressor):
        inner = getattr(reg, "regressor_", None) or getattr(reg, "regressor", None)
        if inner is None:
            reg_ok = False
        else:
            reg_ok = hasattr(inner, "coef_") or getattr(inner, "_model", None) is not None
    else:
        reg_ok = hasattr(reg, "coef_") or getattr(reg, "_model", None) is not None

    return pre_ok and reg_ok


# -------- Uplift helpers --------
def compute_therapist_pred_baselines(estimator: Pipeline, cfg: MatchConfig,
                                     df_ref: pd.DataFrame, ther_table: pd.DataFrame) -> Dict[int, float]:
    """Mean predicted score per therapist over a reference patient set (e.g., validation)."""
    if len(df_ref) == 0 or len(ther_table) == 0:
        return {}
    raw_cols = _feature_cols_from_pipe(estimator, cfg)
    tid_col = cfg.THERAPIST_ID_COL
    agg = np.zeros(len(ther_table), dtype=float)
    for _, prow in df_ref.iterrows():
        dy = build_dyads_for_patient(cfg, prow, ther_table)
        dy = _ensure_columns(dy, raw_cols)
        agg += estimator.predict(dy).astype(float)
    baseline = agg / max(len(df_ref), 1)
    return {int(t): float(b) for t, b in zip(ther_table[tid_col].astype(int).to_numpy(), baseline)}

def rerank_with_uplift(preds: np.ndarray, tids: np.ndarray,
                       baseline_map: Dict[int, float], alpha: float,
                       topM: Optional[int]) -> np.ndarray:
    """Return indices ordering therapists by uplift-adjusted score."""
    base = np.array([baseline_map.get(int(t), 0.0) for t in tids], dtype=float)
    adj = preds - alpha * base
    if topM is not None and topM < len(adj):
        coarse = np.argsort(-preds)[:topM]
        order = coarse[np.argsort(-adj[coarse])]
    else:
        order = np.argsort(-adj)
    return order

def choose_alpha_for_model(estimator: Pipeline, cfg: MatchConfig,
                           df_val: pd.DataFrame, y_val: np.ndarray,
                           ther_table: pd.DataFrame,
                           baseline_map: Dict[int, float]) -> float:
    """Pick alpha that maximizes Spearman(-rank, outcome) on validation."""
    if (not cfg.UPLIFT_RERANK) or len(df_val) == 0 or len(ther_table) == 0 or not baseline_map:
        return 0.0
    raw_cols = _feature_cols_from_pipe(estimator, cfg)
    tid_col = cfg.THERAPIST_ID_COL

    def val_score_for_alpha(alpha: float) -> float:
        ranks = []
        yy = _y_eff(y_val, cfg)
        for _, row in df_val.iterrows():
            dy = build_dyads_for_patient(cfg, row, ther_table)
            dy = _ensure_columns(dy, raw_cols)
            preds = estimator.predict(dy).astype(float)
            tids  = ther_table[tid_col].astype(int).to_numpy()
            order = rerank_with_uplift(preds, tids, baseline_map, alpha, cfg.UPLIFT_TOPM)
            ranked_tids = tids[order]
            factual = int(row[tid_col])
            pos = np.where(ranked_tids == factual)[0]
            ranks.append(int(pos[0]) + 1 if len(pos) else len(ranked_tids) + 1)
        return _safe_spearman(-np.asarray(ranks, float), yy)

    best_alpha, best_s = 0.0, -np.inf
    for a in cfg.UPLIFT_ALPHA_GRID:
        s = val_score_for_alpha(a)
        if s > best_s:
            best_alpha, best_s = a, s
    logger.info("[uplift] selected alpha=%.2f (val Spearman=%.4f)", best_alpha, best_s)
    return best_alpha


#####################################################################
# Brant-like proportional-odds test
#####################################################################
def brant_like_test(ranks: np.ndarray, outcome_eff: np.ndarray, min_pos=3, min_neg=3):
    """
    Approximates Brant test by fitting cumulative logits:
    logit Pr(rank <= c) = α_c + β*y, and testing if β is equal across cutpoints.
    """
    r = np.asarray(ranks, int)
    y = np.asarray(outcome_eff, float)
    cuts = np.sort(np.unique(r))[:-1]  # all but top-most category

    betas, ses, used_cuts = [], [], []
    for c in cuts:
        ybin = (r <= c).astype(int)
        if ybin.sum() < min_pos or (len(ybin) - ybin.sum()) < min_neg:
            continue
        X = sm.add_constant(y)
        try:
            res = sm.Logit(ybin, X).fit(disp=False, maxiter=200)
            betas.append(res.params[1])
            ses.append(res.bse[1])
            used_cuts.append(int(c))
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
        if sparse.issparse(X):
            X = X.todense()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        torch.manual_seed(self.random_state)
        self.n_features_ = X.shape[1]
        self._model = _FMTorch(self.n_features_, self.k).to(self.device)
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
            if self.verbose and (epoch+1) % 5 == 0:
                print(f"[TorchFM] epoch {epoch+1}/{self.n_epochs}  MSE={total/len(ds):.6f}")
        return self
    def predict(self, X):
        if sparse.issparse(X):
            X = X.todense()
        X = np.asarray(X, dtype=np.float32)
        self._model.eval()
        with torch.no_grad():
            return (self._model(torch.from_numpy(X).to(self.device))
                    .cpu().numpy().ravel())
    def get_params(self, deep=True):
        return {"k": self.k, "lr": self.lr, "n_epochs": self.n_epochs,
                "batch_size": self.batch_size, "weight_decay": self.weight_decay,
                "device": self.device, "random_state": self.random_state, "verbose": self.verbose}
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

class RidgeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha=1.0, random_state=42):
        self.alpha = alpha
        self.random_state = random_state
        self._model = Ridge(alpha=self.alpha, random_state=self.random_state)
    def set_params(self, **params):
        if "alpha" in params:
            self.alpha = params["alpha"]
        self._model = Ridge(alpha=self.alpha, random_state=self.random_state)
        return self
    def get_params(self, deep=True):
        return {"alpha": self.alpha, "random_state": self.random_state}
    def fit(self, X, y):
        self._model.fit(X, y); return self
    def predict(self, X):
        return self._model.predict(X)


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
# Preprocessor
#####################################################################
def make_preprocessor_for_feature_list(
    cfg: MatchConfig,
    patient_numeric: List[str],
    patient_categorical: List[str],
    therapist_numeric: List[str],
    therapist_categorical: List[str],
) -> ColumnTransformer:
    if cfg.impute_strategy == "iterative":
        num_imputer = IterativeImputer(**cfg.iterative_imputer_params)
    elif cfg.impute_strategy == "median":
        num_imputer = SimpleImputer(strategy="median")
    else:
        raise ValueError(f"Unknown impute_strategy: {cfg.impute_strategy!r}")

    to_numeric = FunctionTransformer(lambda X: pd.DataFrame(X).apply(pd.to_numeric, errors="coerce"))

    transformers = []
    if patient_numeric:
        transformers.append(("p_num", Pipeline([
            ("to_num", to_numeric), ("impute", num_imputer), ("scale", StandardScaler(with_mean=False)),
        ]), patient_numeric))
    if patient_categorical:
        transformers.append(("p_cat", Pipeline([("onehot", _ohe())]), patient_categorical))
    if therapist_numeric:
        transformers.append(("t_num", Pipeline([
            ("to_num", to_numeric), ("impute", num_imputer), ("scale", StandardScaler(with_mean=False)),
        ]), therapist_numeric))
    if therapist_categorical:
        transformers.append(("t_cat", Pipeline([("onehot", _ohe())]), therapist_categorical))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


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

def predict_rank_for_patient(estimator: Pipeline, cfg: MatchConfig,
                             patient_row: pd.Series, therapists_table: pd.DataFrame) -> int:
    dyads = build_dyads_for_patient(cfg, patient_row, therapists_table)
    raw_cols = _feature_cols_from_pipe(estimator, cfg)
    dyads = _ensure_columns(dyads, raw_cols)
    preds = estimator.predict(dyads)
    order = np.argsort(-preds)
    ranked_tids = therapists_table[cfg.THERAPIST_ID_COL].values[order]
    hist_tid = patient_row[cfg.THERAPIST_ID_COL]
    match = np.where(ranked_tids == hist_tid)[0]
    return (int(match[0]) + 1) if len(match) else int(len(ranked_tids) + 1)


#####################################################################
# Configurable scorers (HPO)
#####################################################################
class RMSEScorer:
    def __call__(self, estimator: Pipeline, X_val: pd.DataFrame, y_val: np.ndarray) -> float:
        try:
            if not _pipeline_is_fitted(estimator):
                return NEG_LARGE
            yhat = estimator.predict(X_val)
            return -float(np.sqrt(mean_squared_error(y_val, yhat)))  # higher is better
        except Exception as e:
            logger.debug(f"RMSEScorer error: {e}")
            return NEG_LARGE

class OrdinalLLScorer:
    def __init__(self, cfg: MatchConfig, therapists_table: pd.DataFrame):
        self.cfg = cfg; self.ther = therapists_table
    def __call__(self, estimator: Pipeline, X_val: pd.DataFrame, y_val: np.ndarray) -> float:
        try:
            if not _pipeline_is_fitted(estimator) or len(X_val) == 0 or len(self.ther) == 0:
                return NEG_LARGE
            ranks = [predict_rank_for_patient(estimator, self.cfg, row, self.ther)
                     for _, row in X_val.iterrows()]
            ranks = np.asarray(ranks, int)
            yy = _y_eff(y_val, self.cfg)
            if ranks.size > 1:
                for link in (self.cfg.ORDINAL_LINKS_TRY or ["logit"]):
                    for method in (self.cfg.ORDINAL_OPTIMIZERS_TRY or ["lbfgs"]):
                        try:
                            mdl = OrderedModel(pd.Series(ranks, name="rank"),
                                               pd.DataFrame({self.cfg.LABEL_COL: yy}),
                                               distr=link)
                            fit_kwargs = dict(method=method, disp=False, maxiter=self.cfg.ORDINAL_MAXITER)
                            if method != "lbfgs":  # avoid tol warning for lbfgs
                                fit_kwargs["tol"] = self.cfg.ORDINAL_TOL
                            res = mdl.fit(**fit_kwargs)
                            return float(res.llf)
                        except Exception:
                            continue
            return _safe_spearman(-ranks.astype(float), yy)
        except Exception as e:
            logger.debug(f"OrdinalLLScorer error: {e}")
            return NEG_LARGE

class SpearmanRankScorer:
    def __init__(self, cfg: MatchConfig, therapists_table: pd.DataFrame):
        self.cfg = cfg; self.ther = therapists_table
    def __call__(self, estimator: Pipeline, X_val: pd.DataFrame, y_val: np.ndarray) -> float:
        try:
            if not _pipeline_is_fitted(estimator) or len(X_val) == 0 or len(self.ther) == 0:
                return NEG_LARGE
            ranks = [predict_rank_for_patient(estimator, self.cfg, row, self.ther)
                     for _, row in X_val.iterrows()]
            yy = _y_eff(y_val, self.cfg)
            return _safe_spearman(-np.asarray(ranks, float), yy)
        except Exception as e:
            logger.debug(f"SpearmanRankScorer error: {e}")
            return NEG_LARGE

def make_validation_scorer(cfg: MatchConfig, therapists_table: pd.DataFrame):
    mode = (cfg.SCORING_MODE or "ordinal").lower()
    if mode == "rmse":
        logger.info("scoring: using RMSE (negated) on observed dyads.")
        return RMSEScorer()
    if mode == "spearman":
        logger.info("scoring: using Spearman(-rank, outcome).")
        return SpearmanRankScorer(cfg, therapists_table)
    if mode == "ordinal":
        logger.info("scoring: using Ordinal Log-Likelihood(rank ~ outcome).")
        return OrdinalLLScorer(cfg, therapists_table)
    raise ValueError(f"Unknown SCORING_MODE: {cfg.SCORING_MODE!r}")


#####################################################################
# Top-k & retrieval helpers
#####################################################################
def compute_topk_metrics(ranks: np.ndarray, ks_hit: List[int], ks_ndcg: List[int]) -> Dict[str, float]:
    ranks = np.asarray(ranks, int)
    out = {}
    out["mrr"] = float(np.mean(1.0 / ranks)) if len(ranks) else np.nan
    for k in ks_hit:
        out[f"hit@{k}"] = float(np.mean(ranks <= k))
    for k in ks_ndcg:
        out[f"ndcg@{k}"] = float(np.mean([ndcg_at_k_from_rank(r, k=k) for r in ranks]))
    return out

def therapist_topk_table(df_test: pd.DataFrame, ranks: np.ndarray, cfg: MatchConfig) -> pd.DataFrame:
    r = np.asarray(ranks, int)
    tids = df_test[cfg.THERAPIST_ID_COL].to_numpy()
    df = pd.DataFrame({
        cfg.THERAPIST_ID_COL: tids,
        "is_top1": (r == 1).astype(int),
        "is_top5": (r <= 5).astype(int),
    })
    agg = df.groupby(cfg.THERAPIST_ID_COL).agg(
        n_cases=("is_top1", "size"),
        top1=("is_top1", "sum"),
        top5=("is_top5", "sum"),
    ).reset_index()
    agg["top1_rate"] = agg["top1"] / agg["n_cases"]
    agg["top5_rate"] = agg["top5"] / agg["n_cases"]
    return agg.sort_values(["top1", "top5", "n_cases"], ascending=[False, False, False]).reset_index(drop=True)


#####################################################################
# Grid helper (param nesting for TTR)
#####################################################################
def _maybe_nest_param_grid_for_ttr(grid: Dict[str, List], use_ttr: bool) -> Dict[str, List]:
    if not use_ttr:
        return grid
    # "regressor__*" → "regressor__regressor__*" (because TTR.regressor)
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
                if (self.cfg.TARGET_MODE or "delta").lower() == "t1":
                    # Predict T1 distress directly
                    df[self.cfg.LABEL_COL] = t1_sum
                else:
                    # Default: delta = T0 - T1 (improvement positive)
                    df[self.cfg.LABEL_COL] = t0_sum - t1_sum
            else:
                raise ValueError("LABEL_COL missing and PHQ/GAD columns not provided.")
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

    def _make_full_pipeline(self, p_num, p_cat, t_num, t_cat, model: BaseEstimator) -> Pipeline:
        logger.info("preprocessor: building (imputer=%s)", getattr(self.cfg, "impute_strategy", "N/A"))
        pre = make_preprocessor_for_feature_list(self.cfg, p_num, p_cat, t_num, t_cat)

        reg: BaseEstimator = model
        if getattr(self.cfg, "STANDARDIZE_Y", False):
            # Standardize y during fit, auto-inverse on predict ⇒ metrics remain in raw units
            reg = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())

        return Pipeline([("preprocessor", pre), ("regressor", reg)])

    def _evaluate_on_test(self, name: str, estimator: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray,
                          df_test: pd.DataFrame, ther_table: pd.DataFrame, val_score: float,
                          *, baseline_map: Optional[Dict[int, float]] = None, alpha: float = 0.0) -> Dict:
        yy_test = _y_eff(y_test, self.cfg)

        # observed dyad predictions for factual pairs (point metrics; unaffected by uplift)
        y_pred_obs = estimator.predict(X_test).astype(float)
        test_rmse = rmse(y_test, y_pred_obs)
        test_cindex = concordance_index(yy_test, y_pred_obs)

        # --------- Build rankings per test patient + therapist-level recommendation tallies
        ranks_uplift = []
        ranks_raw = []
        top1_tids = []
        top5_tids_all = []

        tid_col = self.cfg.THERAPIST_ID_COL
        raw_cols = _feature_cols_from_pipe(estimator, self.cfg)

        for _, row in df_test.iterrows():
            dyads = build_dyads_for_patient(self.cfg, row, ther_table)
            dyads = _ensure_columns(dyads, raw_cols)
            preds = estimator.predict(dyads).astype(float)
            tids  = ther_table[tid_col].astype(int).to_numpy()

            order_raw = np.argsort(-preds)
            if self.cfg.UPLIFT_RERANK and alpha > 0.0 and baseline_map:
                order = rerank_with_uplift(preds, tids, baseline_map, alpha, self.cfg.UPLIFT_TOPM)
            else:
                order = order_raw

            tids_ordered = tids[order]
            tids_ordered_raw = tids[order_raw]

            # recommendation tallies (uplift order)
            top1_tids.append(int(tids_ordered[0]))
            top5_tids_all.extend([int(t) for t in tids_ordered[:5]])

            # ranks for factual therapist
            factual = int(row[tid_col])
            pos_u = np.where(tids_ordered == factual)[0]
            pos_r = np.where(tids_ordered_raw == factual)[0]
            ranks_uplift.append(int(pos_u[0]) + 1 if len(pos_u) else len(tids_ordered) + 1)
            ranks_raw.append(int(pos_r[0]) + 1 if len(pos_r) else len(tids_ordered_raw) + 1)

        ranks_uplift = np.asarray(ranks_uplift, int)
        ranks_raw = np.asarray(ranks_raw, int)

        # therapist-level recommendation counts (uplift order)
        c_top1 = Counter(top1_tids)
        c_top5 = Counter(top5_tids_all)
        ther_ids = ther_table[tid_col].astype(int).tolist()
        ther_reco_counts = pd.DataFrame({tid_col: ther_ids})
        ther_reco_counts["top1_count"] = ther_reco_counts[tid_col].map(c_top1).fillna(0).astype(int)
        ther_reco_counts["top5_count"] = ther_reco_counts[tid_col].map(c_top5).fillna(0).astype(int)
        n_pat = len(df_test)
        ther_reco_counts["n_test_patients"] = n_pat
        ther_reco_counts["top1_share"] = ther_reco_counts["top1_count"] / n_pat
        ther_reco_counts["top5_share"] = ther_reco_counts["top5_count"] / n_pat
        ther_reco_counts = ther_reco_counts.sort_values(
            ["top1_count", "top5_count", tid_col], ascending=[False, False, True]
        ).reset_index(drop=True)

        # retrieval metrics (patient-level) use uplift order
        topk = compute_topk_metrics(ranks_uplift, self.cfg.HIT_KS, self.cfg.NDCG_KS)

        # secondary: Spearman (-rank vs outcome) on uplift order
        test_spearman = _safe_spearman(-ranks_uplift, yy_test)

        # ---- Proportional-odds diagnostic (choose which ranks to test)
        ranks_for_po = ranks_uplift if self.cfg.USE_UPLIFT_FOR_PRIMARY else ranks_raw
        po_test = brant_like_test(ranks_for_po, yy_test, min_pos=self.cfg.PO_MIN_POS, min_neg=self.cfg.PO_MIN_NEG)
        po_p = po_test.get("p", np.nan)
        po_violation = (np.isfinite(po_p) and (po_p < self.cfg.PO_ALPHA))
        if po_violation:
            msg = (f"[PO] Parallel-odds violated (p={po_p:.4g} < α={self.cfg.PO_ALPHA}). "
                   "Proceeding with ranking metrics (hit@k, ndcg, Spearman) as primary; "
                   "ordinal LL reported as reference. "
                   "Set cfg.PPO_SENSITIVITY=True to add MNLogit sensitivity.")
            logger.warning(msg)
            try:
                print(msg)
            except Exception:
                pass

        # ---- Primary: Ordinal model (no binning), try multiple links/optimizers
        coef_change = np.nan; odds_ratio_change = np.nan
        coef_change_std = np.nan; odds_ratio_change_std = np.nan
        coef_change_std_flip = np.nan; odds_ratio_change_std_flip = np.nan
        llf = np.nan; aic = np.nan; bic = np.nan; ll_per_case = np.nan; pseudo_r2 = np.nan
        used_link = None; used_opt = None
        primary_detail = {}; sd_used = np.nan

        if len(df_test) > 1:
            # optionally standardize outcome for interpretability within the ordinal diagnostic
            y_for_fit = yy_test.copy()
            if self.cfg.ORDINAL_STANDARDIZE_CHANGE:
                sd = float(np.nanstd(y_for_fit))
                if sd > 0:
                    y_for_fit = (y_for_fit - float(np.nanmean(y_for_fit))) / sd
                    sd_used = sd

            # choose ranks for primary LL
            use_ranks = ranks_uplift if self.cfg.USE_UPLIFT_FOR_PRIMARY else ranks_raw

            exog_name = self.cfg.LABEL_COL  # show real target name in the ordinal model
            for link in (self.cfg.ORDINAL_LINKS_TRY or ["logit"]):
                for method in (self.cfg.ORDINAL_OPTIMIZERS_TRY or ["lbfgs"]):
                    try:
                        endog = pd.Series(use_ranks, name="rank")
                        exog  = pd.DataFrame({exog_name: y_for_fit})
                        mdl_t = OrderedModel(endog, exog, distr=link)
                        fit_kwargs = dict(method=method, disp=False, maxiter=self.cfg.ORDINAL_MAXITER)
                        if method != "lbfgs":  # avoid tol warning for lbfgs
                            fit_kwargs["tol"] = self.cfg.ORDINAL_TOL
                        res_t = mdl_t.fit(**fit_kwargs)

                        llf = float(res_t.llf)
                        aic = float(getattr(res_t, "aic", np.nan))
                        bic = float(getattr(res_t, "bic", np.nan))
                        ll_per_case = llf / len(df_test)
                        c = float(res_t.params.get(exog_name, np.nan))
                        coef_change_std = c
                        odds_ratio_change_std = float(np.exp(c)) if np.isfinite(c) else np.nan

                        # Sign-flipped (“pro-good”) view:
                        # With P(rank<=k) = F(cut_k − β·x) and x set so higher=better (via _y_eff),
                        # β < 0 means better outcome → better (lower) rank.
                        coef_change_std_flip = -c
                        odds_ratio_change_std_flip = float(np.exp(-c)) if np.isfinite(c) else np.nan

                        # de-standardize back to raw units if we z-scaled y_for_fit
                        if np.isfinite(sd_used) and sd_used > 0:
                            coef_change = c / sd_used
                            odds_ratio_change = float(np.exp(coef_change)) if np.isfinite(coef_change) else np.nan
                        else:
                            coef_change = c
                            odds_ratio_change = odds_ratio_change_std

                        # pseudo-R2 vs null
                        try:
                            null = OrderedModel(endog, pd.DataFrame({"const": np.ones_like(use_ranks)}), distr=link)
                            res_null = null.fit(**fit_kwargs)
                            pseudo_r2 = 1.0 - (llf / float(res_null.llf))
                        except Exception:
                            pseudo_r2 = np.nan

                        used_link, used_opt = link, method
                        primary_detail = {"params": res_t.params.to_dict() if hasattr(res_t, "params") else {}}
                        break
                    except Exception as e:
                        primary_detail = {"error": f"{type(e).__name__}: {e}"}
                if used_link is not None:
                    break

        # PPO sensitivity (optional)
        mnlogit_info = {}
        if self.cfg.PPO_SENSITIVITY:
            try:
                yX = sm.add_constant(yy_test)
                mn = sm.MNLogit(ranks_for_po, yX).fit(disp=False, maxiter=300)
                mnlogit_info = {
                    "mnlogit_llf": float(mn.llf),
                    "mnlogit_df_model": int(mn.df_model),
                    "mnlogit_params_shape": tuple(mn.params.shape),
                }
                logger.info("[PPO] MNLogit sensitivity fitted.")
            except Exception as e:
                mnlogit_info = {"error": f"{type(e).__name__}: {e}"}

        row = {
            "model": name,
            "val_score": val_score,

            # PRIMARY (LL or NaN)
            "test_primary_ordinal_ll": llf,
            "test_ordinal_ll_per_case": ll_per_case,
            "test_ordinal_aic": aic,
            "test_ordinal_bic": bic,
            "test_ordinal_pseudo_r2": pseudo_r2,
            "test_ordinal_link": used_link,
            "test_ordinal_optimizer": used_opt,

            # Coefficients / OR (raw & std; plus flipped sign view)
            "test_coef_change": coef_change,
            "test_odds_ratio_change": odds_ratio_change,
            "test_coef_change_std": coef_change_std,
            "test_odds_ratio_change_std": odds_ratio_change_std,
            "test_coef_change_std_flip": coef_change_std_flip,
            "test_odds_ratio_change_std_flip": odds_ratio_change_std_flip,
            "test_ordinal_sd_change_used": sd_used,

            # Diagnostics
            "test_ordinal_success": bool(np.isfinite(llf)),
            "test_ordinal_error": primary_detail.get("error"),

            # Brant-like PO check
            "po_brant_Q": po_test.get("Q"),
            "po_brant_df": po_test.get("df"),
            "po_brant_p": po_p,
            "po_violation": bool(po_violation),

            # Retrieval (patient-level; uplift order)
            "test_mrr": topk.get("mrr"),
            **{f"test_hit@{k}": topk[f"hit@{k}"] for k in self.cfg.HIT_KS},
            **{f"test_ndcg@{k}": topk[f"ndcg@{k}"] for k in self.cfg.NDCG_KS},

            # Legacy metrics
            "test_spearman": test_spearman,
            "test_rmse": test_rmse,
            "test_cindex": float(test_cindex) if np.isfinite(test_cindex) else np.nan,
            "n_test": int(len(df_test)),

            # Uplift info
            "uplift_alpha_used": float(alpha),
        }

        details = {
            "ranks_uplift": ranks_uplift,
            "ranks_raw": ranks_raw,
            "y_test": y_test,
            "y_pred_obs": y_pred_obs,
            "c_index": test_cindex,
            "spearman": test_spearman,
            "primary_detail": primary_detail,
            "po_test": po_test,
            "ppo_sensitivity": mnlogit_info,
            "therapist_recommendation_counts": ther_reco_counts,
            "uplift_alpha_used": float(alpha),
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

        p_num_u, p_cat_u, t_num_u, t_cat_u = self._build_feature_lists_for_use(df_train, df_val, ther_table)
        base_feats = p_num_u + p_cat_u + t_num_u + t_cat_u

        extra_ids = [self.cfg.PATIENT_ID_COL]
        if self.cfg.THERAPIST_ID_COL not in base_feats:
            extra_ids.append(self.cfg.THERAPIST_ID_COL)
        tr_cols = _dedupe(base_feats + extra_ids)

        X_train = _ensure_columns(df_train[tr_cols].copy(), tr_cols)
        y_train = df_train[self.cfg.LABEL_COL].to_numpy()
        X_val   = _ensure_columns(df_val[tr_cols].copy(), tr_cols)
        y_val   = df_val[self.cfg.LABEL_COL].to_numpy()
        logger.info("matrices: X_train=%s, X_val=%s", tuple(X_train.shape), tuple(X_val.shape))

        X_trval = pd.concat([X_train, X_val], ignore_index=True)
        y_trval = np.concatenate([y_train, y_val])
        test_fold = np.r_[np.full(len(X_train), -1), np.zeros(len(X_val), dtype=int)]
        cv = PredefinedSplit(test_fold)

        fm_pipe    = self._make_full_pipeline(p_num_u, p_cat_u, t_num_u, t_cat_u, TorchFMRegressor())
        ridge_pipe = self._make_full_pipeline(p_num_u, p_cat_u, t_num_u, t_cat_u, RidgeRegressor())
        val_scorer = make_validation_scorer(self.cfg, ther_table)

        use_ttr = bool(getattr(self.cfg, "STANDARDIZE_Y", False))
        fm_grid    = _maybe_nest_param_grid_for_ttr(self.cfg.fm_param_grid,    use_ttr)
        ridge_grid = _maybe_nest_param_grid_for_ttr(self.cfg.ridge_param_grid, use_ttr)

        fm_gs = GridSearchCV(fm_pipe, fm_grid, scoring=val_scorer, refit=True, cv=cv, n_jobs=1, verbose=0)
        fm_gs.fit(X_trval, y_trval)
        ridge_gs = GridSearchCV(ridge_pipe, ridge_grid, scoring=val_scorer, refit=True, cv=cv, n_jobs=1, verbose=0)
        ridge_gs.fit(X_trval, y_trval)

        candidates = [
            ("fm", fm_gs.best_estimator_, fm_gs.best_score_, fm_gs),
            ("ridge", ridge_gs.best_estimator_, ridge_gs.best_score_, ridge_gs),
        ]
        best_name, best_model, best_valscore, _ = max(candidates, key=lambda t: t[2])
        logger.info("[Selected] %s | Val Score: %.6f", best_name, best_valscore)

        # Test matrices
        tr_cols_no_ids = base_feats
        test_extra_ids = [self.cfg.PATIENT_ID_COL]
        if self.cfg.THERAPIST_ID_COL not in tr_cols_no_ids:
            test_extra_ids.append(self.cfg.THERAPIST_ID_COL)
        test_cols = _dedupe(tr_cols_no_ids + test_extra_ids)

        X_test = _ensure_columns(df_test[test_cols].copy(), test_cols)
        y_test = df_test[self.cfg.LABEL_COL].to_numpy()

        # --- Baselines per therapist from validation (for uplift) ---
        ther_baseline_pred = {
            name: compute_therapist_pred_baselines(est, self.cfg, df_val, ther_table)
            for name, est, _, _ in candidates
        }

        # --- Choose alpha per model on validation ---
        alpha_by_model = {
            name: choose_alpha_for_model(est, self.cfg, df_val, y_val, ther_table, ther_baseline_pred.get(name, {}))
            for name, est, _, _ in candidates
        }

        # Evaluate both models (winner + benchmark)
        results_rows = []
        details_by_model = {}
        therapist_reco_by_model = {}

        for name, est, val_score, gs in candidates:
            eval_out = self._evaluate_on_test(
                name, est, X_test, y_test, df_test, ther_table, val_score,
                baseline_map=ther_baseline_pred.get(name, {}), alpha=alpha_by_model.get(name, 0.0)
            )
            results_rows.append(eval_out["row"])
            details_by_model[name] = eval_out["details"]
            therapist_reco_by_model[name] = eval_out["details"]["therapist_recommendation_counts"]

        # mark best
        for row in results_rows:
            row["is_best"] = (row["model"] == best_name)

        summary = pd.DataFrame(results_rows).sort_values(["is_best", "val_score"], ascending=[False, False]).reset_index(drop=True)

        self.test_details_ = {
            "models": details_by_model,
            "therapist_recommendation_counts_by_model": therapist_reco_by_model,
            "therapist_baselines_by_model": ther_baseline_pred,
            "uplift_alpha_by_model": alpha_by_model,
            "used_features": {
                "patient_numeric": p_num_u,
                "patient_categorical": p_cat_u,
                "therapist_numeric": t_num_u,
                "therapist_categorical": t_cat_u,
            },
            "best_model_name": best_name,
            "best_params": dict(candidates[0][3].best_params_) if candidates[0][0]==best_name else dict(candidates[1][3].best_params_)
        }
        ther_table_out = therapists_table_from_train(self.cfg, df_train)
        best_reco = therapist_reco_by_model.get(best_name, pd.DataFrame())
        return {"summary": summary, "therapists": ther_table_out, "therapist_recommendations": best_reco}

# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_pipeline(df: pd.DataFrame, cfg: MatchConfig):
    pipe = MatchPipeline(cfg)
    pipe.set_data(df)
    out = pipe.run()
    # surface internals
    out["used_features"] = pipe.test_details_.get("used_features", {})
    out["best_model_name"] = pipe.test_details_.get("best_model_name")
    out["best_params"] = pipe.test_details_.get("best_params")
    out["therapist_baselines_by_model"] = pipe.test_details_.get("therapist_baselines_by_model", {})
    out["uplift_alpha_by_model"] = pipe.test_details_.get("uplift_alpha_by_model", {})
    return out
