# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import logging
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy import sparse

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from pandas.api import types as pdt
import warnings

# Optional: statsmodels for proportional-odds etc.
try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    _HAS_SM_ORD = True
    _SM_ORD_ERR = None
except Exception as e:
    _HAS_SM_ORD = False
    _SM_ORD_ERR = e

# Optional: correlations
try:
    from scipy.stats import spearmanr, kendalltau
    _HAS_SPEARMAN = True
    _HAS_KENDALL = True
except Exception:
    try:
        from scipy.stats import spearmanr
        _HAS_SPEARMAN = True
    except Exception:
        _HAS_SPEARMAN = False
    _HAS_KENDALL = False

# Torch FM
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

    # Precomputed or computed label
    LABEL_COL: str = "delta_distress"

    # Direction: True if larger values = better outcome (improvement)
    OUTCOME_HIGHER_IS_BETTER: bool = True

    # ---- Features ----
    patient_numeric: List[str] = field(default_factory=list)
    patient_categorical: List[str] = field(default_factory=list)
    therapist_numeric: List[str] = field(default_factory=list)
    therapist_categorical: List[str] = field(default_factory=list)

    # ---- Optional: include therapist_id as categorical (one-hot)
    include_therapist_id_feature: bool = False

    # ---- FM hyperparameter grid (Torch FM)
    fm_param_grid: Dict[str, List] = field(default_factory=lambda: {
        "regressor__k": [8, 16, 32, 64],
        "regressor__lr": [1e-3, 5e-4],
        "regressor__n_epochs": [40],
        "regressor__batch_size": [256, 128],
        "regressor__weight_decay": [0.0, 1e-5],
    })

    # ---- Imputation (uniform for all numeric features) ----
    impute_strategy: str = "median"              # "median" or "iterative"
    iterative_imputer_params: Dict[str, object] = field(default_factory=lambda: {
        "max_iter": 10,
        "initial_strategy": "median",
        "random_state": 0,
        "sample_posterior": False,
    })

    # ---- Ridge baseline grid
    ridge_param_grid: Dict[str, List] = field(default_factory=lambda: {
        "regressor__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3]
    })

    # ---- Validation scorer for GridSearchCV: "rmse" | "ordinal" | "spearman"
    SCORING_MODE: str = "ordinal"

    # ---- Ordinal fitting robustness (no binning/coalescing)
    ORDINAL_LINKS_TRY: List[str] = field(default_factory=lambda: ["logit", "probit", "cloglog"])
    ORDINAL_OPTIMIZERS_TRY: List[str] = field(default_factory=lambda: ["lbfgs", "bfgs", "newton"])
    ORDINAL_MAXITER: int = 500
    ORDINAL_TOL: float = 1e-6
    ORDINAL_STANDARDIZE_CHANGE: bool = True   # z-score change before fit?

    # ---- Diagnostics thresholds
    ORDINAL_MAX_RAW_CATS_WARN: int = 30


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
    y = np.asarray(y_true, float)
    s = np.asarray(y_pred, float)
    mask = np.isfinite(y) & np.isfinite(s)
    y, s = y[mask], s[mask]
    n = len(y)
    num = den = 0
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
    return list(dict.fromkeys(seq))  # preserve order

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
    if not _HAS_SPEARMAN:
        return 0.0
    r = spearmanr(a, b, nan_policy="omit").correlation
    return float(r) if np.isfinite(r) else 0.0

def _safe_kendall(a: np.ndarray, b: np.ndarray) -> float:
    if not _HAS_KENDALL:
        return np.nan
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return np.nan
    try:
        t = kendalltau(a, b).correlation
        return float(t) if np.isfinite(t) else np.nan
    except Exception:
        return np.nan

def _y_eff(y, cfg: "MatchConfig"):
    y = np.asarray(y, float)
    return y if cfg.OUTCOME_HIGHER_IS_BETTER else -y

# Silence the specific sklearn FutureWarning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"This Pipeline instance is not fitted yet.*"
)

NEG_LARGE = -1e12  # safe finite fallback score

def _pipeline_is_fitted(pipe: Pipeline) -> bool:
    """Check fit status without triggering predict()."""
    if not isinstance(pipe, Pipeline):
        return False
    pre = pipe.named_steps.get("preprocessor")
    reg = pipe.named_steps.get("regressor")
    if pre is None or reg is None:
        return False
    pre_ok = hasattr(pre, "transformers_")
    reg_ok = hasattr(reg, "coef_") or getattr(reg, "_model", None) is not None
    return pre_ok and reg_ok


#####################################################################
# Robust ordinal LL fit (no binning)
#####################################################################
def _fit_ordered_ll_robust(ranks: np.ndarray, y_eff: np.ndarray, cfg: "MatchConfig") -> Dict[str, object]:
    """
    Try multiple links & optimizers; optionally z-score y; no binning/coalescing.
    Returns dict with success flag and details; primary metric is llf if success else NaN.
    """
    out = {
        "success": False, "error": None,
        "llf": np.nan, "aic": np.nan, "bic": np.nan, "ll_per_case": np.nan,
        "coef_change": np.nan, "odds_ratio_change": np.nan, "pseudo_r2": np.nan,
        "link": None, "optimizer": None,
        "n_cats": int(np.unique(ranks).size)
    }
    if not _HAS_SM_ORD:
        out["error"] = f"statsmodels OrderedModel unavailable: {_SM_ORD_ERR}"
        return out

    r = np.asarray(ranks, int)
    y = np.asarray(y_eff, float)
    mask = np.isfinite(r) & np.isfinite(y)
    r, y = r[mask], y[mask]

    if r.size < 2 or np.unique(r).size < 2:
        out["error"] = "not enough categories to fit"
        return out

    if cfg.ORDINAL_STANDARDIZE_CHANGE:
        std = np.nanstd(y)
        if np.isfinite(std) and std > 0:
            y = (y - np.nanmean(y)) / std

    err_msgs = []
    for link in (cfg.ORDINAL_LINKS_TRY or ["logit"]):
        for method in (cfg.ORDINAL_OPTIMIZERS_TRY or ["lbfgs"]):
            try:
                mdl = OrderedModel(pd.Series(r, name="rank"),
                                   pd.DataFrame({"change": y}),
                                   distr=link)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = mdl.fit(method=method, maxiter=cfg.ORDINAL_MAXITER, tol=cfg.ORDINAL_TOL, disp=False)
                llf = float(res.llf)
                aic = float(getattr(res, "aic", np.nan))
                bic = float(getattr(res, "bic", np.nan))
                coef = float(res.params.get("change", np.nan))
                orat = float(np.exp(coef)) if np.isfinite(coef) else np.nan
                llpc = llf / len(r)
                # null model for pseudo-R2
                try:
                    null = OrderedModel(pd.Series(r, name="rank"),
                                        pd.DataFrame({"const": np.ones_like(r)}),
                                        distr=link).fit(method=method, maxiter=cfg.ORDINAL_MAXITER, tol=cfg.ORDINAL_TOL, disp=False)
                    pr2 = 1.0 - (llf / float(null.llf))
                except Exception:
                    pr2 = np.nan

                out.update({
                    "success": True, "llf": llf, "aic": aic, "bic": bic,
                    "ll_per_case": llpc, "coef_change": coef, "odds_ratio_change": orat,
                    "pseudo_r2": pr2, "link": link, "optimizer": method
                })
                return out
            except Exception as e:
                err_msgs.append(f"{link}/{method}: {e}")

    out["error"] = " | ".join(err_msgs[:3]) + (" ..." if len(err_msgs) > 3 else "")
    return out


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
                opt.zero_grad()
                l = crit(self._model(xb), yb); l.backward(); opt.step()
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
        return {
            "k": self.k, "lr": self.lr, "n_epochs": self.n_epochs,
            "batch_size": self.batch_size, "weight_decay": self.weight_decay,
            "device": self.device, "random_state": self.random_state, "verbose": self.verbose,
        }
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
        self._model.fit(X, y)
        return self
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
            te.append(g.iloc[[-1]])
            va.append(g.iloc[[-2]])
            tr.append(g.iloc[:-2])
        elif n == 2:
            te.append(g.iloc[[-1]])
            tr.append(g.iloc[[0]])
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
            ("to_num", to_numeric),
            ("impute", num_imputer),
            ("scale", StandardScaler(with_mean=False)),
        ]), patient_numeric))
    if patient_categorical:
        transformers.append(("p_cat", Pipeline([
            ("onehot", _ohe()),
        ]), patient_categorical))
    if therapist_numeric:
        transformers.append(("t_num", Pipeline([
            ("to_num", to_numeric),
            ("impute", num_imputer),
            ("scale", StandardScaler(with_mean=False)),
        ]), therapist_numeric))
    if therapist_categorical:
        transformers.append(("t_cat", Pipeline([
            ("onehot", _ohe()),
        ]), therapist_categorical))

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
            res = _fit_ordered_ll_robust(np.asarray(ranks, int), _y_eff(y_val, self.cfg), self.cfg)
            if res["success"]:
                return float(res["llf"])
            # keep GridSearch stable if ORD fails completely
            return _safe_spearman(-np.asarray(ranks, float), _y_eff(y_val, self.cfg))
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
            return _safe_spearman(-np.asarray(ranks, float), _y_eff(y_val, self.cfg))
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
# Diagnostics (no binning trials)
#####################################################################
def ordinal_fit_diagnostics(ranks: np.ndarray, outcome: np.ndarray, *, cfg: MatchConfig) -> Dict[str, object]:
    ranks = np.asarray(ranks, int)
    y = _y_eff(outcome, cfg)

    di = {}
    di["n_test"] = int(len(ranks))
    di["finite_ranks"] = bool(np.isfinite(ranks).all())
    di["finite_outcome"] = bool(np.isfinite(y).all())
    cats, counts = np.unique(ranks, return_counts=True)
    di["n_categories"] = int(len(cats))
    di["min_cat_count"] = int(counts.min()) if len(counts) else None
    di["median_cat_count"] = float(np.median(counts)) if len(counts) else None
    di["mean_cat_count"] = float(np.mean(counts)) if len(counts) else None
    di["all_ranks_equal"] = bool(len(cats) <= 1)
    di["outcome_std"] = float(np.nanstd(y))
    di["spearman_neg_rank_vs_outcome"] = _safe_spearman(-ranks.astype(float), y)
    di["kendall_neg_rank_vs_outcome"] = _safe_kendall(-ranks.astype(float), y)
    di["cindex_neg_rank_vs_outcome"] = float(concordance_index(y, -ranks))

    di["warn_many_categories"] = bool(len(cats) > cfg.ORDINAL_MAX_RAW_CATS_WARN)
    di["warn_outcome_near_const"] = bool(di["outcome_std"] < 1e-8)
    di["warn_small_test"] = bool(len(ranks) < 30)
    di["warn_quasi_separation"] = bool(
        (abs(di["spearman_neg_rank_vs_outcome"]) > 0.95 if np.isfinite(di["spearman_neg_rank_vs_outcome"]) else False)
        or (abs(di["kendall_neg_rank_vs_outcome"]) > 0.9 if np.isfinite(di["kendall_neg_rank_vs_outcome"]) else False)
        or (di["cindex_neg_rank_vs_outcome"] > 0.98)
    )

    if di["warn_many_categories"]:
        logger.info("[diagnostics] Many rank categories (%d). ORD may struggle.", di["n_categories"])
    if di["warn_quasi_separation"]:
        logger.info("[diagnostics] Near-separation signals (high monotonicity).")
    if di["warn_outcome_near_const"]:
        logger.info("[diagnostics] Outcome nearly constant on test fold.")
    if di["all_ranks_equal"]:
        logger.info("[diagnostics] All ranks identical — ordinal model not identifiable.")

    return di


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
        if self.cfg.LABEL_COL not in df.columns:
            required = [self.cfg.PHQ8_T0_COL, self.cfg.GAD7_T0_COL,
                        self.cfg.PHQ8_T1_COL, self.cfg.GAD7_T1_COL]
            if all(isinstance(c, str) and c in df.columns for c in required):
                mask_complete = df[[self.cfg.PHQ8_T1_COL, self.cfg.GAD7_T1_COL]].notna().all(axis=1)
                df = df.loc[mask_complete].reset_index(drop=True)
                t0_sum = df[self.cfg.PHQ8_T0_COL].astype(float) + df[self.cfg.GAD7_T0_COL].astype(float)
                t1_sum = df[self.cfg.PHQ8_T1_COL].astype(float) + df[self.cfg.GAD7_T1_COL].astype(float)
                df[self.cfg.LABEL_COL] = t0_sum - t1_sum
            else:
                raise ValueError("LABEL_COL missing and PHQ/GAD columns not provided.")
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
        return Pipeline([("preprocessor", pre), ("regressor", model)])

    def _evaluate_on_test(self, name: str, estimator: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray,
                          df_test: pd.DataFrame, ther_table: pd.DataFrame, val_score: float) -> Dict:
        yy_test = _y_eff(y_test, self.cfg)

        # observed dyad predictions
        y_pred_obs = estimator.predict(X_test)
        test_rmse = rmse(y_test, y_pred_obs)
        test_cindex = concordance_index(yy_test, y_pred_obs)

        # ranks & ndcg
        ranks, ndcgs = [], []
        for _, row in df_test.iterrows():
            r = predict_rank_for_patient(estimator, self.cfg, row, ther_table)
            ranks.append(r); ndcgs.append(ndcg_at_k_from_rank(r))
        ranks = np.asarray(ranks, int)
        ndcgs = np.asarray(ndcgs, float)

        # secondary: Spearman between -rank and outcome (always reported)
        test_spearman = _safe_spearman(-ranks, yy_test)

        # diagnostics (no binning)
        diagnostics = ordinal_fit_diagnostics(ranks, y_test, cfg=self.cfg)

        # primary: robust ordinal fit (no binning). if it fails, primary is NaN.
        ord_res = _fit_ordered_ll_robust(ranks, yy_test, self.cfg)

        row = {
            "model": name,
            "val_score": val_score,
            "test_primary_ordinal_ll": float(ord_res["llf"]) if ord_res["success"] else np.nan,
            "test_coef_change": float(ord_res["coef_change"]) if ord_res["success"] else np.nan,
            "test_odds_ratio_change": float(ord_res["odds_ratio_change"]) if ord_res["success"] else np.nan,
            "test_ordinal_ll_per_case": float(ord_res["ll_per_case"]) if ord_res["success"] else np.nan,
            "test_ordinal_aic": float(ord_res["aic"]) if ord_res["success"] else np.nan,
            "test_ordinal_bic": float(ord_res["bic"]) if ord_res["success"] else np.nan,
            "test_ordinal_pseudo_r2": float(ord_res["pseudo_r2"]) if ord_res["success"] else np.nan,
            "test_ordinal_link": ord_res["link"],
            "test_ordinal_optimizer": ord_res["optimizer"],
            "test_ordinal_success": bool(ord_res["success"]),
            "test_ordinal_error": ord_res["error"] if not ord_res["success"] else None,
            "test_spearman": test_spearman,
            "test_ndcg_mean": float(np.mean(ndcgs)),
            "test_ndcg_std": float(np.std(ndcgs)),
            "test_rmse": test_rmse,
            "test_cindex": float(test_cindex) if np.isfinite(test_cindex) else np.nan,
            "n_test": int(len(df_test)),
        }
        details = {
            "ranks": ranks, "ndcgs": ndcgs, "y_test": y_test, "y_pred_obs": y_pred_obs,
            "c_index": test_cindex, "spearman": test_spearman,
            "ordinal_result": ord_res,
            "diagnostics": diagnostics,
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

        fm_gs = GridSearchCV(fm_pipe, self.cfg.fm_param_grid, scoring=val_scorer, refit=True, cv=cv, n_jobs=1, verbose=0)
        fm_gs.fit(X_trval, y_trval)
        ridge_gs = GridSearchCV(ridge_pipe, self.cfg.ridge_param_grid, scoring=val_scorer, refit=True, cv=cv, n_jobs=1, verbose=0)
        ridge_gs.fit(X_trval, y_trval)

        # Select best by validation score
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

        # Evaluate BOTH models (best + benchmark)
        results_rows = []
        details_by_model = {}
        for name, est, val_score, gs in candidates:
            eval_out = self._evaluate_on_test(name, est, X_test, y_test, df_test, ther_table, val_score)
            results_rows.append(eval_out["row"])
            details_by_model[name] = eval_out["details"]

        # Mark which is best
        for row in results_rows:
            row["is_best"] = (row["model"] == best_name)

        summary = pd.DataFrame(results_rows).sort_values(["is_best", "val_score"], ascending=[False, False]).reset_index(drop=True)

        self.test_details_ = {
            "models": details_by_model,
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
        return {"summary": summary, "therapists": ther_table_out}


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_pipeline(df: pd.DataFrame, cfg: MatchConfig):
    pipe = MatchPipeline(cfg)
    pipe.set_data(df)
    return pipe.run()
