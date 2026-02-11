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

from joblib import Memory
import os

# ----- CatBoost -----
from catboost import CatBoostRegressor, Pool, EFstrType

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
    TARGET_MODE: str = "t1"        # "delta" or "t1"
    STANDARDIZE_Y: bool = True
    AUTO_ORIENT_FROM_TARGET_MODE: bool = True  # auto set HIGHER_IS_BETTER from TARGET_MODE

    # Name & direction of the target actually used by the model
    LABEL_COL: str = "t1_distress_sum"
    OUTCOME_HIGHER_IS_BETTER: bool = False     # auto-set for t1/delta when AUTO_ORIENT...=True
    ORDINAL_USE_ORIENTED: bool = True  # if False, use raw LABEL_COL in ordinal regression

    # ---- Features ----
    patient_numeric: List[str] = field(default_factory=list)
    patient_categorical: List[str] = field(default_factory=list)
    therapist_numeric: List[str] = field(default_factory=list)
    therapist_categorical: List[str] = field(default_factory=list)
    include_therapist_id_feature: bool = False

    # ---- Imputation ----
    impute_strategy: str = "iterative"   # "median" | "iterative"
    iterative_imputer_params: Dict[str, object] = field(default_factory=lambda: {
        "max_iter": 5, "verbose": 0, "initial_strategy": "median", "random_state": 0, "sample_posterior": False,
    })

    # --- Caching of preprocessing steps (speeds up GridSearch) ---
    CACHE_STEPS: bool = True
    CACHE_DIR: str = field(default_factory=lambda: os.environ.get("SKCACHE_DIR", "./.sk_cache"))

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
    # EBM grid (!! use integers for interactions !!)
    ebm_param_grid: Dict[str, List] = field(default_factory=lambda: {
        "regressor__interactions": [20, 50, 100],
        "regressor__validation_size": [0.15],
        "regressor__learning_rate": [0.04],
        "regressor__max_bins": [128],
        "regressor__max_leaves": [2, 3],
        "regressor__smoothing_rounds": [750, 1000],
        "regressor__min_samples_leaf": [2, 5],
        "regressor__n_jobs": [-1],
        "regressor__random_state": [42],
    })
    SCORING_MODE: str = "rank_slope"  # "rmse" or "rank_slope"

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

    # ---- CatBoost categorical handling ----
    CAT_USE_NATIVE_CATEGORICAL: bool = False   # True => use native cats; False => OHE
    CAT_MIN_FREQ_FOR_OHE: int = 1              # min_frequency if using OHE

    # ---- Explanations (configurable) ----
    EXPLAIN_COMPUTE_FI: bool = True                      # split-based FI (Cat) / term FI (EBM)
    EXPLAIN_COMPUTE_SHAP_GLOBAL: bool = True             # CatBoost SHAP (global)
    EXPLAIN_COMPUTE_INTERACTIONS: bool = True            # CatBoost interaction / EBM pair terms
    EXPLAIN_COMPUTE_SHAP_INTERACTIONS: bool = True       # SHAP interaction values (TreeExplainer; Cat only)
    EXPLAIN_SAMPLE_ROWS: Optional[int] = None            # sample rows for explanations; None=use all
    EXPLAIN_TOPK: int = 50                               # top-K rows for *_ohe / *_agg; also used for printing

    # ---- RMSE CI options ----
    RMSE_CI_ENABLE: bool = True
    RMSE_CI_METHOD: str = "bootstrap"   # "bootstrap" or "delta"
    RMSE_CI_LEVEL: float = 0.95
    RMSE_CI_BOOTSTRAP_B: int = 2000
    RMSE_CI_RANDOM_STATE: int = 0

    # ---- Cross-validation / repeat strategy ----
    CV_STRATEGY: str = "per_therapist_random"     # "leave_last" | "per_therapist_random"
    CV_N_REPEATS: int = 10               # e.g., 10 to repeat the random split 10x
    CV_RANDOM_STATE: int = 42            # base seed for repeatable random splits

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

def _rmse_ci_bootstrap(y_true: np.ndarray, y_pred: np.ndarray, level=0.95, B=2000, rng=0) -> Tuple[float, float]:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    n = len(y_true); rs = np.random.RandomState(rng)
    stats = np.empty(B, float)
    for b in range(B):
        idx = rs.randint(0, n, size=n)
        stats[b] = np.sqrt(np.mean((y_true[idx] - y_pred[idx])**2))
    lo = np.percentile(stats, (1 - level) / 2 * 100)
    hi = np.percentile(stats, (1 + level) / 2 * 100)
    return float(lo), float(hi)

def _rmse_ci_delta(y_true: np.ndarray, y_pred: np.ndarray, level=0.95) -> Tuple[float, float]:
    from scipy.stats import norm
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    res2 = (y_true - y_pred)**2
    mse = float(np.mean(res2))
    rmse_v = float(np.sqrt(mse))
    if rmse_v == 0.0:
        return rmse_v, rmse_v
    var_z = float(np.var(res2, ddof=1)) if len(res2) > 1 else 0.0
    n = len(res2)
    var_rmse = (var_z / n) / (4.0 * mse) if mse > 0 and n > 1 else 0.0
    se = float(np.sqrt(max(var_rmse, 0.0)))
    z = norm.ppf(0.5 + level/2.0)
    return rmse_v - z*se, rmse_v + z*se

def _dedupe(seq):
    return list(dict.fromkeys(seq))

def _ohe(sparse_output=True, min_frequency=1):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output, min_frequency=min_frequency)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=(sparse_output))

def _y_eff(y, cfg: "MatchConfig"):
    y = np.asarray(y, float)
    return y if cfg.OUTCOME_HIGHER_IS_BETTER else -y

def _s_eff(a, cfg: "MatchConfig"):
    a = np.asarray(a, float)
    return a if cfg.OUTCOME_HIGHER_IS_BETTER else -a

def _pipeline_is_fitted(pipe: Pipeline) -> bool:
    if not isinstance(pipe, Pipeline): return False
    pre = pipe.named_steps.get("preprocessor"); reg = pipe.named_steps.get("regressor")
    if pre is None or reg is None: return False
    pre_ok = hasattr(pre, "transformers_")
    inner = reg
    if isinstance(reg, TransformedTargetRegressor):
        inner = getattr(reg, "regressor_", None) or getattr(reg, "regressor", None)
    if inner is None: return False
    if hasattr(inner, "is_fitted"):
        try: return pre_ok and bool(inner.is_fitted())
        except Exception: pass
    if hasattr(inner, "coef_") or hasattr(inner, "n_features_in_"): return pre_ok
    if getattr(inner, "_model", None) is not None: return pre_ok
    return pre_ok and hasattr(inner, "predict")

def _model_name_from_estimator(estimator: Pipeline) -> str:
    reg = estimator.named_steps.get("regressor")
    if isinstance(reg, TransformedTargetRegressor):
        inner = getattr(reg, "regressor", None)
        return type(inner).__name__ if inner is not None else type(reg).__name__
    return type(reg).__name__ if reg is not None else type(estimator).__name__

#####################################################################
# Picklable FunctionTransformer helpers
#####################################################################
def _to_numeric_df(X):
    from scipy import sparse as _sparse
    if _sparse.issparse(X):
        X = X.toarray()
    if isinstance(X, pd.DataFrame):
        return X.apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")

def _to_str_df(X):
    from scipy import sparse as _sparse
    if _sparse.issparse(X):
        X = X.toarray()
    if isinstance(X, pd.DataFrame):
        return X.astype(str)
    return pd.DataFrame(X).astype(str)

def _FT(func):
    try:
        return FunctionTransformer(func=func, feature_names_out="one-to-one", validate=False)
    except TypeError:
        return FunctionTransformer(func=func, validate=False)

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
            self._model.train()
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(); l = crit(self._model(xb), yb); l.backward(); opt.step()
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

# --- NEW: reproducible per-therapist random split + dispatcher ---

def _seed_for_tid(tid, base_seed: int, repeat_index: int) -> int:
    """Stable per-therapist seed from (therapist_id, base_seed, repeat_index)."""
    h = hash((str(tid), int(base_seed), int(repeat_index)))
    return int(np.abs(h) % (2**32 - 1) or 1)

def split_per_therapist_random(
    df: pd.DataFrame,
    therapist_col: str,
    *,
    base_seed: int = 0,
    repeat_index: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each therapist:
      - n >= 3: pick 1 random row for TEST and 1 random row for VAL (distinct); rest TRAIN
      - n == 2: randomly pick 1 for TEST, the other goes to TRAIN (no VAL for that therapist)
      - n == 1: all TRAIN
    Randomness is reproducible given (base_seed, repeat_index).
    """
    tr, va, te = [], [], []
    for tid, g in df.groupby(therapist_col, sort=False):
        g = g.sample(frac=1.0, random_state=_seed_for_tid(tid, base_seed, repeat_index)).reset_index(drop=True)
        n = len(g)
        if n >= 3:
            idx = np.arange(n)
            rs = np.random.RandomState(_seed_for_tid(f"{tid}-test", base_seed, repeat_index))
            i_test = int(rs.choice(idx))
            remaining = np.delete(idx, i_test)
            rs2 = np.random.RandomState(_seed_for_tid(f"{tid}-val", base_seed, repeat_index))
            i_val = int(rs2.choice(remaining))
            te.append(g.iloc[[i_test]])
            va.append(g.iloc[[i_val]])
            keep = [i for i in idx if i not in (i_test, i_val)]
            if keep:
                tr.append(g.iloc[keep])
        elif n == 2:
            rs = np.random.RandomState(_seed_for_tid(f"{tid}-2", base_seed, repeat_index))
            i_test = int(rs.randint(0, 2))
            te.append(g.iloc[[i_test]])
            tr.append(g.drop(g.index[i_test]))
        else:  # n == 1 or 0
            tr.append(g)

    df_train = pd.concat(tr).reset_index(drop=True) if tr else df.iloc[0:0].copy()
    df_val   = pd.concat(va).reset_index(drop=True) if va else df.iloc[0:0].copy()
    df_test  = pd.concat(te).reset_index(drop=True) if te else df.iloc[0:0].copy()
    return df_train, df_val, df_test

def split_dataset(
    df: pd.DataFrame,
    cfg: "MatchConfig",
    *,
    repeat_index: int = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Dispatch to the configured CV strategy (no validation fallback)."""
    strat = (cfg.CV_STRATEGY or "leave_last").lower()
    if strat in {"per_therapist_random", "per-therapist-random", "ptr"}:
        return split_per_therapist_random(
            df, cfg.THERAPIST_ID_COL,
            base_seed=int(getattr(cfg, "CV_RANDOM_STATE", 0)),
            repeat_index=int(repeat_index)
        )
    # default
    return split_leave_last(df, cfg.THERAPIST_ID_COL, cfg.DATE_COL)

#####################################################################
# Preprocessors
#####################################################################
def make_preprocessor_common(
    cfg: MatchConfig,
    p_num: List[str], p_cat: List[str],
    t_num: List[str], t_cat: List[str],
    *, force_dense_ohe: bool = False
) -> ColumnTransformer:
    if cfg.impute_strategy == "iterative":
        num_imputer = IterativeImputer(**cfg.iterative_imputer_params)
    elif cfg.impute_strategy == "median":
        num_imputer = SimpleImputer(strategy="median")
    else:
        raise ValueError(f"Unknown impute_strategy: {cfg.impute_strategy!r}")

    # picklable FunctionTransformer (no lambdas)
    to_numeric = _FT(_to_numeric_df)

    transformers = []
    if p_num:
        transformers.append(("p_num", Pipeline([
            ("to_num", to_numeric), ("impute", num_imputer), ("scale", StandardScaler(with_mean=False)),
        ]), p_num))
    if p_cat:
        transformers.append(("p_cat", Pipeline([
            ("onehot", _ohe(sparse_output=not force_dense_ohe, min_frequency=cfg.CAT_MIN_FREQ_FOR_OHE))
        ]), p_cat))
    if t_num:
        transformers.append(("t_num", Pipeline([
            ("to_num", to_numeric), ("impute", num_imputer), ("scale", StandardScaler(with_mean=False)),
        ]), t_num))
    if t_cat:
        transformers.append(("t_cat", Pipeline([
            ("onehot", _ohe(sparse_output=not force_dense_ohe, min_frequency=cfg.CAT_MIN_FREQ_FOR_OHE))
        ]), t_cat))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=(0.0 if force_dense_ohe else 0.3),
        verbose_feature_names_out=True,
    )
    try:
        if force_dense_ohe: ct.set_output(transform="pandas")
        else: ct.set_output(transform="default")
    except Exception:
        pass
    return ct


def make_preprocessor_cat_native(
    cfg: MatchConfig,
    p_num: List[str], p_cat: List[str],
    t_num: List[str], t_cat: List[str]
) -> ColumnTransformer:
    """Preprocessor for CatBoost/EBM native categoricals (no OHE)."""
    if cfg.impute_strategy == "iterative":
        num_imputer = IterativeImputer(**cfg.iterative_imputer_params)
    elif cfg.impute_strategy == "median":
        num_imputer = SimpleImputer(strategy="median")
    else:
        raise ValueError(f"Unknown impute_strategy: {cfg.impute_strategy!r}")

    # picklable transformers
    to_numeric = _FT(_to_numeric_df)
    to_str     = _FT(_to_str_df)
    cat_imputer = SimpleImputer(strategy="most_frequent")

    transformers = []
    num_cols = (p_num or []) + (t_num or [])
    if num_cols:
        transformers.append(("num", Pipeline([
            ("to_num", to_numeric), ("impute", num_imputer), ("scale", StandardScaler(with_mean=False)),
        ]), num_cols))
    cat_cols = (p_cat or []) + (t_cat or [])
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("impute", cat_imputer),
            ("to_str", to_str),  # ensure strings for CatBoost/EBM
        ]), cat_cols))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )
    try:
        ct.set_output(transform="pandas")
    except Exception:
        pass
    return ct

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
        try:
            for _name, _trans, sel in getattr(pre, "transformers", []):
                if isinstance(sel, list):
                    cols.extend(sel)
            if cols:
                cols = list(dict.fromkeys(cols))
        except Exception:
            cols = []
    if not cols:
        cols = list(dict.fromkeys(
            cfg.patient_numeric + cfg.patient_categorical + cfg.therapist_numeric + cfg.therapist_categorical
        ))
    ids = []
    if cfg.PATIENT_ID_COL not in cols: ids.append(cfg.PATIENT_ID_COL)
    if cfg.THERAPIST_ID_COL not in cols: ids.append(cfg.THERAPIST_ID_COL)
    return cols + ids

#####################################################################
# Scorers (HPO)
#####################################################################
class RMSEScorer:
    def __call__(self, estimator: Pipeline, X_val: pd.DataFrame, y_val: np.ndarray) -> float:
        try:
            yhat = estimator.predict(X_val)
            return -float(np.sqrt(mean_squared_error(y_val, yhat)))  # higher is better
        except Exception as e:
            model_name = _model_name_from_estimator(estimator)
            print(f"[VAL ERROR][{model_name}] {type(e).__name__}: {e}")
            return NEG_LARGE

class RankSlopeScorer:
    """
    Score = -coef from proportional-odds model of rank ~ y_reg.
    (More negative coef = stronger relation 'better outcome -> lower rank' => higher score.)
    Falls back to Spearman corr between (-rank) and y_reg if ordinal fit fails.
    """
    def __init__(self, cfg: MatchConfig, therapists_table: pd.DataFrame):
        self.cfg = cfg
        self.ther_table = therapists_table

    def __call__(self, estimator: Pipeline, X_val: pd.DataFrame, y_val: np.ndarray) -> float:
        try:
            # --- build factual ranks on the validation set
            tid_col = self.cfg.THERAPIST_ID_COL
            raw_cols = _feature_cols_from_pipe(estimator, self.cfg)
            ranks = []
            for _, row in X_val.iterrows():
                dyads = build_dyads_for_patient(self.cfg, row, self.ther_table)
                dyads = dyads.reindex(columns=raw_cols + [c for c in dyads.columns if c not in raw_cols],
                                      fill_value=np.nan)
                preds = estimator.predict(dyads).astype(float)
                scores_eff = _s_eff(preds, self.cfg)
                tids = self.ther_table[tid_col].astype(int).to_numpy()
                order = np.argsort(-scores_eff)
                tids_ordered = tids[order]
                factual = int(row[tid_col])
                pos = np.where(tids_ordered == factual)[0]
                ranks.append(int(pos[0]) + 1 if len(pos) else len(tids_ordered) + 1)

            ranks = np.asarray(ranks, int)
            y = np.asarray(y_val, float)
            y_reg = _y_eff(y, self.cfg) if self.cfg.ORDINAL_USE_ORIENTED else y
            if self.cfg.ORDINAL_STANDARDIZE_CHANGE:
                sd = float(np.nanstd(y_reg))
                if sd > 0:
                    y_reg = (y_reg - float(np.nanmean(y_reg))) / sd

            # --- ordinal logit: rank ~ y_reg
            endog = pd.Series(ranks, name="rank")
            exog = pd.DataFrame({self.cfg.LABEL_COL: y_reg})
            link  = (self.cfg.ORDINAL_LINKS_TRY or ["logit"])[0]
            opt   = (self.cfg.ORDINAL_OPTIMIZERS_TRY or ["lbfgs"])[0]
            try:
                mdl = OrderedModel(endog, exog, distr=link)
                res = mdl.fit(method=opt, disp=False, maxiter=self.cfg.ORDINAL_MAXITER)
                coef = float(res.params.get(self.cfg.LABEL_COL, np.nan))
                if not np.isfinite(coef):
                    raise RuntimeError("coef not finite")
                return float(-coef)  # more negative coef ⇒ better ⇒ higher score
            except Exception:
                # --- fallback: Spearman correlation between (-rank) and y_reg
                from scipy.stats import spearmanr
                rho = spearmanr(-ranks, y_reg, nan_policy="omit").correlation
                return float(rho) if np.isfinite(rho) else NEG_LARGE
        except Exception as e:
            model_name = _model_name_from_estimator(estimator)
            print(f"[VAL ERROR][{model_name}][rank_slope] {type(e).__name__}: {e}")
            return NEG_LARGE


def make_validation_scorer(cfg: MatchConfig, therapists_table: pd.DataFrame):
    mode = (cfg.SCORING_MODE or "rmse").lower()
    if mode == "rmse":
        logger.info("scoring: using RMSE (negated) on observed dyads.")
        return RMSEScorer()
    elif mode in {"rank_slope", "rankslope"}:
        logger.info("scoring: using rank_slope (−ordinal coef; higher is better).")
        return RankSlopeScorer(cfg, therapists_table)
    else:
        logger.warning("Unknown SCORING_MODE=%s; defaulting to RMSE.", cfg.SCORING_MODE)
        return RMSEScorer()

#####################################################################
# Feature name & parent mapping helpers (robust)
#####################################################################
def _ct_get_feature_names(pre: ColumnTransformer) -> List[str]:
    try:
        return list(pre.get_feature_names_out())
    except Exception:
        return []

def _find_ohe_in_transformer(trans):
    if isinstance(trans, Pipeline):
        for _, step in trans.named_steps.items():
            if isinstance(step, OneHotEncoder):
                return step
    if isinstance(trans, OneHotEncoder):
        return trans
    return None

def _ohe_out_sizes(ohe: OneHotEncoder) -> List[int]:
    sizes = []
    drop_idx = getattr(ohe, "drop_idx_", None)
    for i, cats in enumerate(ohe.categories_):
        drop_one = 1 if (ohe.drop is not None and drop_idx is not None and drop_idx[i] is not None) else 0
        sizes.append(len(cats) - drop_one)
    return sizes

def _input_feature_candidates_from_pre(pre: ColumnTransformer) -> List[str]:
    cands = []
    try:
        for _name, _trans, cols in pre.transformers_:
            if isinstance(cols, (list, tuple)):
                cands.extend(list(cols))
    except Exception:
        pass
    return list(dict.fromkeys(cands))

def _parent_mapping_from_pre(pre: ColumnTransformer,
                             out_feature_names: Optional[List[str]] = None,
                             orig_candidates: Optional[List[str]] = None) -> Dict[str, str]:
    names = list(out_feature_names) if out_feature_names is not None else _ct_get_feature_names(pre)
    if not names:
        return {}
    mapping: Dict[str, str] = {}
    out_idx = getattr(pre, "output_indices_", None)
    if out_idx is not None:
        for name, trans, cols in pre.transformers_:
            if name not in out_idx: continue
            sl = out_idx[name]
            block_names = names[sl] if not isinstance(sl, slice) else names[sl]
            if isinstance(block_names, np.ndarray):
                block_names = block_names.tolist()
            cols = list(cols) if isinstance(cols, (list, tuple)) else list(cols)

            ohe = _find_ohe_in_transformer(trans)
            if ohe is not None:
                sizes = _ohe_out_sizes(ohe)
                pos = 0
                for i, col in enumerate(cols):
                    cnt = sizes[i]
                    for _ in range(cnt):
                        if pos < len(block_names):
                            mapping[block_names[pos]] = col
                        pos += 1
            else:
                for i, col in enumerate(cols):
                    if i < len(block_names):
                        mapping[block_names[i]] = col
        return mapping

    cand_set = set(orig_candidates or _input_feature_candidates_from_pre(pre))
    cand_sorted = sorted(cand_set, key=len, reverse=True)
    def guess_parent(feat_name: str) -> str:
        for c in cand_sorted:
            if c in feat_name:
                return c
        s = str(feat_name)
        if "__" in s:
            s = s.split("__")[-1]
        if "=" in s:
            s = s.split("=")[0]
        for c in cand_sorted:
            if s.startswith(c):
                return c
        return s
    for n in names:
        mapping[n] = guess_parent(n)
    return mapping

def _unwrap_cat_model_from_pipeline(pipeline: Pipeline):
    reg = pipeline.named_steps.get("regressor")
    inner = reg
    if isinstance(reg, TransformedTargetRegressor):
        inner = getattr(reg, "regressor_", None) or getattr(reg, "regressor", None)
    if isinstance(inner, CatBoostNativeWrapper):
        return inner.cb_
    return inner  # CatBoostRegressor

def _transformed_df_and_names(fitted_pipeline: Pipeline, X_df: pd.DataFrame):
    pre = fitted_pipeline.named_steps["preprocessor"]
    X_tx = pre.transform(X_df)
    names = _ct_get_feature_names(pre)
    if sparse.issparse(X_tx):
        X_tx = pd.DataFrame(X_tx.toarray(), columns=names if names else None)
    elif isinstance(X_tx, np.ndarray):
        X_tx = pd.DataFrame(X_tx, columns=names if names else None)
    elif isinstance(X_tx, pd.DataFrame):
        if names and len(names) == X_tx.shape[1] and list(X_tx.columns) != list(names):
            X_tx = X_tx.copy(); X_tx.columns = names
    else:
        X_tx = pd.DataFrame(np.asarray(X_tx), columns=names if names else None)
    if not names:
        names = list(X_tx.columns)
    return X_tx, names

def _get_cat_features_idx_for_pool(fitted_pipeline: Pipeline, X_tx: pd.DataFrame) -> List[int]:
    reg = fitted_pipeline.named_steps.get("regressor")
    inner = reg
    if isinstance(reg, TransformedTargetRegressor):
        inner = getattr(reg, "regressor_", None) or getattr(reg, "regressor", None)

    if isinstance(inner, CatBoostNativeWrapper) and getattr(inner, "cat_idx_", None):
        return list(inner.cat_idx_)

    idxs = []
    for i, dt in enumerate(X_tx.dtypes):
        if pdt.is_object_dtype(dt) or pdt.is_string_dtype(dt) or pdt.is_categorical_dtype(dt):
            idxs.append(i)
    return idxs

#####################################################################
# CatBoost explanations
#####################################################################
def cat_global_importance_df(fitted_pipeline: Pipeline, X_df: pd.DataFrame, y=None, kind="PredictionValuesChange"):
    model = _unwrap_cat_model_from_pipeline(fitted_pipeline)
    X_tx, names = _transformed_df_and_names(fitted_pipeline, X_df)
    pre = fitted_pipeline.named_steps["preprocessor"]
    candidates = _input_feature_candidates_from_pre(pre)
    parent = _parent_mapping_from_pre(pre, out_feature_names=list(X_tx.columns), orig_candidates=candidates)

    cat_idx = _get_cat_features_idx_for_pool(fitted_pipeline, X_tx)
    if len(cat_idx):
        X_tx.iloc[:, cat_idx] = X_tx.iloc[:, cat_idx].astype(str)
        logger.info("[Explain] Passing %d categorical cols to Pool (FI).", len(cat_idx))
    pool = Pool(X_tx, label=y, cat_features=cat_idx if len(cat_idx) else None)

    ftype = getattr(EFstrType, kind) if isinstance(kind, str) else kind
    vals = model.get_feature_importance(pool, type=ftype)
    df = pd.DataFrame({"feature": list(X_tx.columns), "importance": np.asarray(vals, float)})
    df["orig_feature"] = df["feature"].map(parent).fillna(df["feature"])
    df_sorted = df.sort_values("importance", ascending=False).reset_index(drop=True)
    agg = (df_sorted
           .groupby("orig_feature", as_index=False)["importance"].sum()
           .sort_values("importance", ascending=False)
           .reset_index(drop=True))
    return df_sorted, agg

def cat_shap_global_df(fitted_pipeline: Pipeline, X_df: pd.DataFrame, y=None):
    model = _unwrap_cat_model_from_pipeline(fitted_pipeline)
    X_tx, names = _transformed_df_and_names(fitted_pipeline, X_df)
    pre = fitted_pipeline.named_steps["preprocessor"]
    candidates = _input_feature_candidates_from_pre(pre)
    parent = _parent_mapping_from_pre(pre, out_feature_names=list(X_tx.columns), orig_candidates=candidates)

    cat_idx = _get_cat_features_idx_for_pool(fitted_pipeline, X_tx)
    if len(cat_idx):
        X_tx.iloc[:, cat_idx] = X_tx.iloc[:, cat_idx].astype(str)
        logger.info("[Explain] Passing %d categorical cols to Pool (SHAP).", len(cat_idx))
    pool = Pool(X_tx, label=y, cat_features=cat_idx if len(cat_idx) else None)

    shap_mat = model.get_feature_importance(pool, type=EFstrType.ShapValues)
    shap_vals = np.asarray(shap_mat)[:, :-1]  # drop expected value col

    per_feat = pd.DataFrame({
        "feature": list(X_tx.columns),
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0)
    })
    per_feat["orig_feature"] = per_feat["feature"].map(parent).fillna(per_feat["feature"])
    per_feat_sorted = per_feat.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    agg = (per_feat_sorted
           .groupby("orig_feature", as_index=False)["mean_abs_shap"].sum()
           .sort_values("mean_abs_shap", ascending=False)
           .reset_index(drop=True))
    return per_feat_sorted, agg, shap_vals

def cat_interactions_df(fitted_pipeline: Pipeline, X_df: pd.DataFrame, y=None, top_k=50):
    model = _unwrap_cat_model_from_pipeline(fitted_pipeline)
    X_tx, names = _transformed_df_and_names(fitted_pipeline, X_df)
    pre = fitted_pipeline.named_steps["preprocessor"]
    candidates = _input_feature_candidates_from_pre(pre)
    parent = _parent_mapping_from_pre(pre, out_feature_names=list(X_tx.columns), orig_candidates=candidates)
    cols = list(X_tx.columns)

    cat_idx = _get_cat_features_idx_for_pool(fitted_pipeline, X_tx)
    if len(cat_idx):
        X_tx.iloc[:, cat_idx] = X_tx.iloc[:, cat_idx].astype(str)
        logger.info("[Explain] Passing %d categorical cols to Pool (Interactions).", len(cat_idx))
    pool = Pool(X_tx, label=y, cat_features=cat_idx if len(cat_idx) else None)

    pairs = model.get_feature_importance(pool, type=EFstrType.Interaction)
    arr = np.asarray(pairs, dtype=object)

    if arr.ndim == 2 and arr.shape[1] == 3:
        df_raw = pd.DataFrame(arr, columns=["a", "b", "score"])
        ai = pd.to_numeric(df_raw["a"], errors="coerce")
        aj = pd.to_numeric(df_raw["b"], errors="coerce")
        if ai.notna().all() and aj.notna().all():
            df_pairs = pd.DataFrame({
                "i": ai.astype(int),
                "j": aj.astype(int),
                "score": pd.to_numeric(df_raw["score"], errors="coerce"),
            })
            m = len(cols)
            valid = (df_pairs["i"] >= 0) & (df_pairs["i"] < m) & (df_pairs["j"] >= 0) & (df_pairs["j"] < m)
            df_pairs = df_pairs.loc[valid].copy()
            df_pairs["feat_i"] = df_pairs["i"].map(lambda k: cols[int(k)])
            df_pairs["feat_j"] = df_pairs["j"].map(lambda k: cols[int(k)])
        else:
            df_pairs = pd.DataFrame({
                "feat_i": df_raw["a"].astype(str),
                "feat_j": df_raw["b"].astype(str),
                "score": pd.to_numeric(df_raw["score"], errors="coerce"),
            })
            present = set(cols)
            df_pairs = df_pairs[df_pairs["feat_i"].isin(present) & df_pairs["feat_j"].isin(present)].copy()
    else:
        arr = np.asarray(arr)
        m = min(arr.shape[0], len(cols))
        arr = arr[:m, :m]
        idx_i, idx_j = np.triu_indices(m, k=1)
        df_pairs = pd.DataFrame({
            "feat_i": [cols[i] for i in idx_i],
            "feat_j": [cols[j] for j in idx_j],
            "score": arr[idx_i, idx_j],
        })

    df_pairs["orig_i"] = df_pairs["feat_i"].map(parent).fillna(df_pairs["feat_i"])
    df_pairs["orig_j"] = df_pairs["feat_j"].map(parent).fillna(df_pairs["feat_j"])

    ohe_top = df_pairs.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    df_pairs_agg = df_pairs.copy()
    agg = (df_pairs_agg.assign(pair=list(zip(df_pairs_agg["orig_i"], df_pairs_agg["orig_j"])))
           .groupby("pair", as_index=False)["score"].sum()
           .sort_values("score", ascending=False)
           .head(top_k))
    if len(agg):
        agg[["orig_i", "orig_j"]] = pd.DataFrame(agg["pair"].tolist(), index=agg.index)
        agg = agg.drop(columns=["pair"]).reset_index(drop=True)
    else:
        agg = pd.DataFrame(columns=["orig_i","orig_j","score"])
    return ohe_top, agg

def shap_interactions_df(fitted_pipeline: Pipeline, X_df: pd.DataFrame, y=None, top_k=50):
    try:
        import shap  # type: ignore
    except Exception:
        logger.warning("shap not available; skipping SHAP interaction computation.")
        return (pd.DataFrame(columns=["feat_i","feat_j","score","orig_i","orig_j"]),
                pd.DataFrame(columns=["orig_i","orig_j","score"]))

    model = _unwrap_cat_model_from_pipeline(fitted_pipeline)
    X_tx, names = _transformed_df_and_names(fitted_pipeline, X_df)
    pre = fitted_pipeline.named_steps["preprocessor"]
    candidates = _input_feature_candidates_from_pre(pre)
    parent = _parent_mapping_from_pre(pre, out_feature_names=list(X_tx.columns), orig_candidates=candidates)

    explainer = shap.TreeExplainer(model)
    inter = explainer.shap_interaction_values(X_tx)  # [n, m, m]
    inter = np.asarray(inter)
    mean_abs = np.mean(np.abs(inter), axis=0)  # [m, m]

    cols = list(X_tx.columns)
    m = len(cols)
    idx_i, idx_j = np.triu_indices(m, k=1)
    scores = mean_abs[idx_i, idx_j]
    df_pairs = pd.DataFrame({
        "feat_i": [cols[i] for i in idx_i],
        "feat_j": [cols[j] for j in idx_j],
        "score": scores,
    }).sort_values("score", ascending=False).reset_index(drop=True)

    df_pairs["orig_i"] = df_pairs["feat_i"].map(parent).fillna(df_pairs["feat_i"])
    df_pairs["orig_j"] = df_pairs["feat_j"].map(parent).fillna(df_pairs["feat_j"])

    ohe_top = df_pairs.head(top_k).reset_index(drop=True)
    df_pairs_agg = df_pairs.copy()
    agg = (df_pairs_agg.assign(pair=list(zip(df_pairs_agg["orig_i"], df_pairs_agg["orig_j"])))
           .groupby("pair", as_index=False)["score"].sum()
           .sort_values("score", ascending=False)
           .head(top_k))
    if len(agg):
        agg[["orig_i","orig_j"]] = pd.DataFrame(agg["pair"].tolist(), index=agg.index)
        agg = agg.drop(columns=["pair"]).reset_index(drop=True)
    else:
        agg = pd.DataFrame(columns=["orig_i","orig_j","score"])
    return ohe_top, agg

#####################################################################
# EBM (Explainable Boosting Machine)
#####################################################################
def _interpret_available() -> bool:
    try:
        import importlib
        importlib.import_module("interpret.glassbox")
        return True
    except Exception:
        return False

class EBMWrapper(BaseEstimator, RegressorMixin):
    """Lazy wrapper around interpret.glassbox.ExplainableBoostingRegressor."""
    def __init__(self, **params):
        self.params = params
        self._model = None  # real EBM will be here

    def get_params(self, deep=True):
        return dict(self.params)

    def set_params(self, **params):
        self.params.update(params or {})
        return self

    def fit(self, X, y):
        try:
            from interpret.glassbox import ExplainableBoostingRegressor
        except Exception as e:
            raise ImportError(
                "EBM requested but 'interpret' is not installed. "
                "Install via: conda install -c conda-forge interpret  (or) pip install interpret-core interpret"
            ) from e
        self._model = ExplainableBoostingRegressor(**self.params)
        self._model.fit(X, y)
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("EBMWrapper not fitted")
        return self._model.predict(X)

def _unwrap_ebm_from_pipeline(pipeline: Pipeline):
    reg = pipeline.named_steps.get("regressor")
    inner = reg
    if isinstance(reg, TransformedTargetRegressor):
        inner = getattr(reg, "regressor_", None) or getattr(reg, "regressor", None)
    if isinstance(inner, EBMWrapper):
        return inner._model
    return None

# --------- EBM introspection helpers (robust across interpret versions) ---------
def _extract_base_from_transformed(name: str, cfg: MatchConfig) -> str:
    s = str(name)
    if "__" in s:
        s = s.split("__", 1)[1]
    s = s.split("=", 1)[0]
    raw_cols = (cfg.patient_numeric + cfg.therapist_numeric +
                cfg.patient_categorical + cfg.therapist_categorical)
    if s in raw_cols:
        return s
    for col in sorted(raw_cols, key=len, reverse=True):
        if s.startswith(col + "_"):
            return col
    return s

def ebm_global_importance_df(ebm_model) -> pd.DataFrame:
    # Try explain_global().data() (newer interpret), fallback to attributes
    try:
        gl = ebm_model.explain_global()
        d = gl.data()
        names = None; scores = None
        if isinstance(d, dict):
            if "overall" in d:
                names = d["overall"].get("names"); scores = d["overall"].get("scores")
            if (not names) and ("names" in d) and ("scores" in d):
                names, scores = d["names"], d["scores"]
        if names and scores and len(names) == len(scores) and len(names) > 0:
            return (pd.DataFrame({"term": names, "importance": scores})
                    .sort_values("importance", ascending=False).reset_index(drop=True))
    except Exception:
        pass
    term_names = getattr(ebm_model, "term_names_", None)
    term_imps  = getattr(ebm_model, "term_importances_", None)
    if term_names is not None and term_imps is not None and len(term_names) > 0:
        return (pd.DataFrame({"term": list(term_names), "importance": list(term_imps)})
                .sort_values("importance", ascending=False).reset_index(drop=True))
    return pd.DataFrame(columns=["term","importance"])

def ebm_feature_and_term_tables_from_full(pipe: Pipeline, cfg: MatchConfig
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pre: ColumnTransformer = pipe.named_steps["preprocessor"]
    ebm = _unwrap_ebm_from_pipeline(pipe)
    if ebm is None:
        return (pd.DataFrame(columns=["feature_index","transformed_name","base_feature","source"]),
                pd.DataFrame(columns=["term_index","term_name","importance","type","features_transformed","features_base","pretty"]),
                pd.DataFrame(columns=["base_feature","importance"]),
                pd.DataFrame(columns=["term_index","term_name","pretty","importance","features_transformed","features_base"]))

    # Transformed feature names EBM saw
    feat_names = list(getattr(ebm, "feature_names_in_", []))
    if not feat_names:
        try:
            feat_names = list(pre.get_feature_names_out())
        except Exception:
            n = int(getattr(ebm, "n_features_in_", 0))
            feat_names = [f"feature_{i:03d}" for i in range(n)]

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

    term_feats = list(getattr(ebm, "term_features_", []))
    imp_df = ebm_global_importance_df(ebm)
    imp_map = {str(k): float(v) for k, v in zip(imp_df.get("term", []), imp_df.get("importance", []))}

    model_term_names = list(getattr(ebm, "term_names_", []))
    if not model_term_names or len(model_term_names) != len(term_feats:
        ):
        model_term_names = []
        for idxs in term_feats:
            idxs = list(idxs)
            model_term_names.append(" x ".join(str(feat_names[j]) for j in idxs))

    term_rows = []
    for t_idx, idxs in enumerate(term_feats):
        idxs = list(idxs)
        ttype = "interaction" if len(idxs) > 1 else "univariate"
        trans = [feat_names[j] for j in idxs]
        base  = [_extract_base_from_transformed(n, cfg) for n in trans]
        term_name = str(model_term_names[t_idx]) if t_idx < len(model_term_names) else " x ".join(trans)
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

    if not term_table.empty:
        tmp = term_table[term_table["type"] == "univariate"].copy()
        tmp["base_feature"] = tmp["features_base"].apply(lambda lst: lst[0] if isinstance(lst, list) and lst else None)
        top_raw_features = (tmp.dropna(subset=["base_feature"])
                              .groupby("base_feature", as_index=False)["importance"].sum()
                              .sort_values("importance", ascending=False, na_position="last")
                              .reset_index(drop=True))
        top_interactions = (term_table[term_table["type"] == "interaction"]
                              .loc[:, ["term_index","term_name","pretty","importance","features_transformed","features_base"]]
                              .sort_values("importance", ascending=False, na_position="last")
                              .reset_index(drop=True))
    else:
        top_raw_features = pd.DataFrame(columns=["base_feature","importance"])
        top_interactions = pd.DataFrame(columns=["term_index","term_name","pretty","importance","features_transformed","features_base"])

    return feature_name_mapping, term_table, top_raw_features, top_interactions

def ebm_aggregate_base_feature_importance(term_table: pd.DataFrame) -> pd.DataFrame:
    if term_table is None or term_table.empty:
        return pd.DataFrame(columns=["base_feature","importance"])
    acc: Dict[str, float] = {}
    for _, row in term_table.iterrows():
        imp = float(row.get("importance", np.nan))
        if not np.isfinite(imp): continue
        bases = row.get("features_base", []) or []
        if row.get("type") == "univariate":
            if bases:
                acc[bases[0]] = acc.get(bases[0], 0.0) + imp
        else:
            uniq = list(dict.fromkeys(bases))
            if len(uniq) > 0:
                share = imp / float(len(uniq))
                for b in uniq:
                    acc[b] = acc.get(b, 0.0) + share
    if not acc:
        return pd.DataFrame(columns=["base_feature","importance"])
    return (pd.DataFrame({"base_feature": list(acc.keys()), "importance": list(acc.values())})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))

def _pairs_from_interactions_table(top_interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in top_interactions.iterrows():
        imp = float(r.get("importance", np.nan))
        bases = r.get("features_base", []) or []
        uniq = list(dict.fromkeys(bases))
        if len(uniq) >= 2 and np.isfinite(imp):
            rows.append({"orig_i": uniq[0], "orig_j": uniq[1], "score": imp})
    if not rows:
        return pd.DataFrame(columns=["orig_i","orig_j","score"])
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

#####################################################################
# CatBoost native-wrapper (for native categorical handling)
#####################################################################
class CatBoostNativeWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **cb_params):
        params = {"loss_function": "RMSE", "random_seed": 42, "verbose": False}
        params.update(cb_params or {})
        self.cb_params = params
        self._model = CatBoostRegressor(**self.cb_params)
        self.cat_idx_ = None

    def get_params(self, deep=True):
        return dict(self.cb_params)

    def set_params(self, **params):
        self.cb_params.update(params or {})
        self._model = CatBoostRegressor(**self.cb_params)
        return self

    def _infer_cat_idx(self, X):
        if isinstance(X, pd.DataFrame):
            idxs = []
            for i, dt in enumerate(X.dtypes):
                if pdt.is_object_dtype(dt) or pdt.is_string_dtype(dt) or pdt.is_categorical_dtype(dt):
                    idxs.append(i)
            return idxs
        return []

    def _coerce_cats_to_str(self, X):
        if not isinstance(X, pd.DataFrame) or not self.cat_idx_:
            return X
        X = X.copy()
        cols = list(X.columns)
        for i in self.cat_idx_:
            if 0 <= i < len(cols):
                c = cols[i]
                if not (pdt.is_object_dtype(X[c]) or pdt.is_integer_dtype(X[c])):
                    X[c] = X[c].astype(str)
        return X

    def fit(self, X, y):
        self.cat_idx_ = self._infer_cat_idx(X)
        X2 = self._coerce_cats_to_str(X)
        self._model.fit(X2, y, cat_features=self.cat_idx_ if self.cat_idx_ else None)
        return self

    def predict(self, X):
        X2 = self._coerce_cats_to_str(X)
        return self._model.predict(X2)

    @property
    def cb_(self):
        return self._model

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
        self.ebm_grid: Optional[GridSearchCV] = None
        self.test_details_: Dict = {}
        self._repeat_index: int = 0

    def set_repeat_index(self, i: int = 0):
        self._repeat_index = int(i)

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
        mem = Memory(location=self.cfg.CACHE_DIR, verbose=0) if getattr(self.cfg, "CACHE_STEPS", False) else None
        return Pipeline([("preprocessor", pre), ("regressor", reg)], memory=mem)

    def _evaluate_on_test(self, name: str, estimator: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray,
                          df_test: pd.DataFrame, ther_table: pd.DataFrame, val_score: float) -> Dict:
        y_pred_obs = estimator.predict(X_test).astype(float)
        test_rmse = rmse(y_test, y_pred_obs)

        test_rmse_lo = test_rmse_hi = np.nan
        if getattr(self.cfg, "RMSE_CI_ENABLE", True) and len(y_test) > 1:
            method = (self.cfg.RMSE_CI_METHOD or "bootstrap").lower()
            level  = float(getattr(self.cfg, "RMSE_CI_LEVEL", 0.95))
            if method == "bootstrap":
                test_rmse_lo, test_rmse_hi = _rmse_ci_bootstrap(
                    y_test, y_pred_obs, level=level,
                    B=int(getattr(self.cfg, "RMSE_CI_BOOTSTRAP_B", 2000)),
                    rng=int(getattr(self.cfg, "RMSE_CI_RANDOM_STATE", 0))
                )
            else:
                test_rmse_lo, test_rmse_hi = _rmse_ci_delta(y_test, y_pred_obs, level=level)

        ranks, top1_tids, top5_tids_all = [], [], []
        per_pat_rows = []
        tid_col = self.cfg.THERAPIST_ID_COL
        pid_col = self.cfg.PATIENT_ID_COL
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
            factual_rank = int(pos[0]) + 1 if len(pos) else len(tids_ordered) + 1
            ranks.append(factual_rank)

            y_raw = float(row[self.cfg.LABEL_COL])
            y_oriented = float(_y_eff(np.array([y_raw]), self.cfg)[0])

            idx_f = np.where(tids == factual)[0]
            mu_f = float(preds[int(idx_f[0])]) if len(idx_f) else np.nan

            per_pat_rows.append({
                pid_col: row.get(pid_col, np.nan),
                "factual_therapist_id": factual,
                "factual_rank": factual_rank,
                self.cfg.LABEL_COL: y_raw,
                "y_eff": y_oriented,
                "mu_hat_factual": mu_f,
            })

        per_patient_df = pd.DataFrame(per_pat_rows)
        ranks = np.asarray(ranks, int)

        n_pat = len(df_test)
        factual_top1_count = int(np.sum(ranks == 1))
        factual_top5_count = int(np.sum(ranks <= 5))
        factual_top1_share = (factual_top1_count / n_pat) if n_pat else np.nan
        factual_top5_share = (factual_top5_count / n_pat) if n_pat else np.nan

        y_for_reg = _y_eff(y_test, self.cfg) if self.cfg.ORDINAL_USE_ORIENTED else np.asarray(y_test, float)
        llf = aic = bic = ll_per_case = pseudo_r2 = np.nan
        coef_change = odds_ratio_change = coef_change_std = odds_ratio_change_std = np.nan
        used_link = used_opt = None; sd_used = np.nan

        if len(df_test) > 1:
            y_for_fit = y_for_reg.copy()
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
                        res_t = mdl_t.fit(**{"method": method, "disp": False, "maxiter": self.cfg.ORDINAL_MAXITER})
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
                        try:
                            null = OrderedModel(endog, pd.DataFrame({"const": np.ones_like(ranks)}), distr=link)
                            res_null = null.fit(**{"method": method, "disp": False, "maxiter": self.cfg.ORDINAL_MAXITER})
                            pseudo_r2 = 1.0 - (llf / float(res_null.llf))
                        except Exception:
                            pseudo_r2 = np.nan
                        used_link, used_opt = link, method
                        break
                    except Exception:
                        continue
                if used_link is not None:
                    break

        ther_reco_counts = pd.DataFrame()
        if self.cfg.REPORT_TOP_COUNTS:
            c_top1 = Counter(top1_tids); c_top5 = Counter(top5_tids_all)
            ther_ids = ther_table[tid_col].astype(int).tolist()
            ther_reco_counts = pd.DataFrame({tid_col: ther_ids})
            ther_reco_counts["top1_count"] = ther_reco_counts[tid_col].map(c_top1).fillna(0).astype(int)
            ther_reco_counts["top5_count"] = ther_reco_counts[tid_col].map(c_top5).fillna(0).astype(int)
            ther_reco_counts["n_test_patients"] = n_pat
            ther_reco_counts["top1_share"] = ther_reco_counts["top1_count"] / max(n_pat, 1)
            ther_reco_counts["top5_share"] = ther_reco_counts["top5_count"] / max(n_pat, 1)
            ther_reco_counts = ther_reco_counts.sort_values(
                ["top1_count","top5_count",tid_col], ascending=[False, False, True]
            ).reset_index(drop=True)

        val_failed = not (np.isfinite(val_score) and (val_score > NEG_LARGE/2))
        
        row = {
            "model": name,
            "val_failed": bool(val_failed),
            "test_rmse": float(test_rmse),
            "test_rmse_lo": float(test_rmse_lo),
            "test_rmse_hi": float(test_rmse_hi),
            "factual_top1_count": factual_top1_count,
            "factual_top1_share": factual_top1_share,
            "factual_top5_count": factual_top5_count,
            "factual_top5_share": factual_top5_share,
            "ord_ll": llf,
            "ord_ll_per_case": ll_per_case,
            "ord_pseudo_r2": pseudo_r2,
            "ord_coef_change_std": coef_change_std,
            "ord_or_change_std": odds_ratio_change_std,
            "ord_link": used_link,
            "ord_optimizer": used_opt,
            "ord_sd_used": sd_used,
        }
        
        mode = (self.cfg.SCORING_MODE or "rmse").lower()
        if mode == "rmse":
            row["val_rmse"] = -float(val_score) if not val_failed else np.inf
        else:
            row["val_rank_slope"] = float(val_score) if not val_failed else NEG_LARGE
            row["val_rmse"] = np.nan

        details = {
            "ranks": ranks,
            "y_test": y_test,
            "y_pred_obs": y_pred_obs,
            "therapist_recommendation_counts": ther_reco_counts,
            "rank_outcome_table": per_patient_df,
        }
        return {"row": row, "details": details}

    def run(self) -> Dict[str, pd.DataFrame]:
        if self.df is None:
            raise RuntimeError("Call set_data(df) first.")

        # use configured split strategy (per-therapist random / leave_last) with repeat index
        df_train, df_val, df_test = split_dataset(self.df, self.cfg, repeat_index=int(getattr(self, "_repeat_index", 0)))
        logger.info(
            "Therapists: %d | Train: %d | Val: %d | Test: %d | repeat=%d | CV=%s",
            self.df[self.cfg.THERAPIST_ID_COL].nunique(), len(df_train), len(df_val), len(df_test),
            int(getattr(self, "_repeat_index", 0)), (self.cfg.CV_STRATEGY or "leave_last")
        )

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

        # Pipelines
        fm_pipe    = self._make_full_pipeline(p_num_u, p_cat_u, t_num_u, t_cat_u, TorchFMRegressor(), force_dense=False)
        ridge_pipe = self._make_full_pipeline(p_num_u, p_cat_u, t_num_u, t_cat_u, RidgeRegressor(), force_dense=False)

        if getattr(self.cfg, "CAT_USE_NATIVE_CATEGORICAL", False):
            pre_cat = make_preprocessor_cat_native(self.cfg, p_num_u, p_cat_u, t_num_u, t_cat_u)
            cat_model = CatBoostNativeWrapper(loss_function="RMSE", random_seed=42, verbose=False)
            reg: BaseEstimator = cat_model
            if getattr(self.cfg, "STANDARDIZE_Y", False):
                reg = TransformedTargetRegressor(regressor=cat_model, transformer=StandardScaler())
            mem = Memory(location=self.cfg.CACHE_DIR, verbose=0) if getattr(self.cfg, "CACHE_STEPS", False) else None
            cat_pipe = Pipeline([("preprocessor", pre_cat), ("regressor", reg)], memory=mem)
        else:
            cat_pipe  = self._make_full_pipeline(
                p_num_u, p_cat_u, t_num_u, t_cat_u,
                CatBoostRegressor(loss_function="RMSE", random_seed=42, verbose=False),
                force_dense=True
            )

        # EBM (native cats)
        ebm_pipe = None
        if _interpret_available():
            pre_ebm = make_preprocessor_cat_native(self.cfg, p_num_u, p_cat_u, t_num_u, t_cat_u)
            ebm_model = EBMWrapper()  # params via grid
            reg_ebm: BaseEstimator = ebm_model
            if getattr(self.cfg, "STANDARDIZE_Y", False):
                reg_ebm = TransformedTargetRegressor(regressor=ebm_model, transformer=StandardScaler())
            mem = Memory(location=self.cfg.CACHE_DIR, verbose=0) if getattr(self.cfg, "CACHE_STEPS", False) else None
            ebm_pipe = Pipeline([("preprocessor", pre_ebm), ("regressor", reg_ebm)], memory=mem)
        else:
            logger.warning("interpret not found; skipping EBM model.")

        val_scorer = make_validation_scorer(self.cfg, ther_table)
        use_ttr = bool(getattr(self.cfg, "STANDARDIZE_Y", False))
        fm_grid     = {k.replace("regressor__", "regressor__regressor__"): v for k, v in self.cfg.fm_param_grid.items()} if use_ttr else self.cfg.fm_param_grid
        ridge_grid  = {k.replace("regressor__", "regressor__regressor__"): v for k, v in self.cfg.ridge_param_grid.items()} if use_ttr else self.cfg.ridge_param_grid
        cat_grid    = {k.replace("regressor__", "regressor__regressor__"): v for k, v in self.cfg.cat_param_grid.items()} if use_ttr else self.cfg.cat_param_grid
        ebm_grid    = None
        if ebm_pipe is not None:
            ebm_grid = {k.replace("regressor__", "regressor__regressor__"): v for k, v in self.cfg.ebm_param_grid.items()} if use_ttr else self.cfg.ebm_param_grid

        fm_gs    = GridSearchCV(fm_pipe,    fm_grid,   scoring=val_scorer, refit=True, cv=cv, n_jobs=1, verbose=0, error_score=NEG_LARGE)
        ridge_gs = GridSearchCV(ridge_pipe, ridge_grid, scoring=val_scorer, refit=True, cv=cv, n_jobs=1, verbose=0, error_score=NEG_LARGE)
        cat_gs   = GridSearchCV(cat_pipe,   cat_grid,  scoring=val_scorer, refit=True, cv=cv, n_jobs=1, verbose=0, error_score=NEG_LARGE)
        ebm_gs   = None
        if ebm_pipe is not None and ebm_grid is not None:
            ebm_gs = GridSearchCV(ebm_pipe, ebm_grid, scoring=val_scorer, refit=True, cv=cv, n_jobs=1, verbose=0, error_score=NEG_LARGE)

        print("[PROGRESS] Fitting FM grid...", flush=True)
        fm_gs.fit(X_trval, y_trval)
        print(f"[PROGRESS] FM done. best={fm_gs.best_params_}", flush=True)

        print("[PROGRESS] Fitting Ridge grid...", flush=True)
        ridge_gs.fit(X_trval, y_trval)
        print(f"[PROGRESS] Ridge done. best={ridge_gs.best_params_}", flush=True)

        print("[PROGRESS] Fitting CatBoost grid...", flush=True)
        cat_gs.fit(X_trval, y_trval)
        print(f"[PROGRESS] CatBoost done. best={cat_gs.best_params_}", flush=True)

        if ebm_gs is not None:
            print("[PROGRESS] Fitting EBM grid...", flush=True)
            ebm_gs.fit(X_trval, y_trval)
            print(f"[PROGRESS] EBM done. best={ebm_gs.best_params_}", flush=True)

        # initialize explanation slots
        self.test_details_.update({
            # generic (best-model) mirrors
            "cat_importance_ohe": pd.DataFrame(columns=["feature","importance","orig_feature"]),
            "cat_importance_agg": pd.DataFrame(columns=["orig_feature","importance"]),
            "cat_shap_ohe": pd.DataFrame(columns=["feature","mean_abs_shap","orig_feature"]),
            "cat_shap_agg": pd.DataFrame(columns=["orig_feature","mean_abs_shap"]),
            "cat_interactions_ohe": pd.DataFrame(columns=["feat_i","feat_j","score","orig_i","orig_j"]),
            "cat_interactions_agg": pd.DataFrame(columns=["orig_i","orig_j","score"]),
            "shap_interactions_ohe": pd.DataFrame(columns=["feat_i","feat_j","score","orig_i","orig_j"]),
            "shap_interactions_agg": pd.DataFrame(columns=["orig_i","orig_j","score"]),
            # explicit per-model tables (NEW)
            "cat_model_importance_agg": pd.DataFrame(columns=["orig_feature","importance"]),
            "cat_model_interactions_agg": pd.DataFrame(columns=["orig_i","orig_j","score"]),
            "cat_model_shap_agg": pd.DataFrame(columns=["orig_feature","mean_abs_shap"]),
            "cat_model_shap_interactions_agg": pd.DataFrame(columns=["orig_i","orig_j","score"]),
            "ebm_model_importance_agg": pd.DataFrame(columns=["base_feature","importance"]),
            "ebm_model_interactions_agg": pd.DataFrame(columns=["orig_i","orig_j","score"]),
            # EBM raw tables
            "ebm_feature_importance_agg": pd.DataFrame(columns=["base_feature","importance"]),
            "ebm_interactions_table": pd.DataFrame(columns=["term_index","term_name","pretty","importance","features_transformed","features_base"]),
        })

        # Collect candidates
        candidates = [
            ("fm",    fm_gs.best_estimator_,    fm_gs.best_score_,    fm_gs),
            ("ridge", ridge_gs.best_estimator_, ridge_gs.best_score_, ridge_gs),
            ("cat",   cat_gs.best_estimator_,   cat_gs.best_score_,   cat_gs),
        ]
        if ebm_gs is not None:
            candidates.append(("ebm", ebm_gs.best_estimator_, ebm_gs.best_score_, ebm_gs))
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
            mode = (self.cfg.SCORING_MODE or "rmse").lower()
            if mode == "rmse":
                valid = df_sum[~df_sum["val_failed"]].copy()
                if len(valid):
                    best_name = valid.loc[valid["val_rmse"].idxmin(), "model"]
                else:
                    best_name = df_sum.loc[df_sum["test_rmse"].idxmin(), "model"]
            else:
                best_name = df_sum.loc[df_sum["val_rank_slope"].idxmax(), "model"]

        best_model, _, best_gs = cand_map[best_name][0], cand_map[best_name][1], cand_map[best_name][2]
        logger.info("[Selected] %s | policy=%s", best_name, self.cfg.SELECT_BEST_BY)

        # Sort summary table
        mode = (self.cfg.SCORING_MODE or "rmse").lower()
        if mode == "rmse":
            df_sum["val_key"] = df_sum["val_rmse"].replace(np.inf, np.nan)
            asc_val = True
        else:
            df_sum["val_key"] = -df_sum["val_rank_slope"]
            asc_val = True
        df_sum = df_sum.sort_values(
            by=["val_failed", "val_key", "test_rmse"],
            ascending=[True, asc_val, True]
        ).drop(columns=["val_key"]).reset_index(drop=True)
        
        # ---------- Explanations: compute for BOTH CatBoost and EBM every run ----------
        try:
            X_imp_full = X_trval[_dedupe(p_num_u + p_cat_u + t_num_u + t_cat_u)].copy()
            y_imp_full = y_trval
            if isinstance(self.cfg.EXPLAIN_SAMPLE_ROWS, int) and 0 < self.cfg.EXPLAIN_SAMPLE_ROWS < len(X_imp_full):
                X_imp = X_imp_full.sample(self.cfg.EXPLAIN_SAMPLE_ROWS, random_state=0)
                y_imp = y_imp_full[X_imp.index]
                logger.info("Explanations on sampled rows: %d / %d", len(X_imp), len(X_imp_full))
            else:
                X_imp, y_imp = X_imp_full, y_imp_full

            # --- CatBoost per-run explanations (always) ---
            try:
                cat_est = cand_map.get("cat", (None,))[0]
                if cat_est is not None:
                    if self.cfg.EXPLAIN_COMPUTE_FI:
                        fi_ohe, fi_agg = cat_global_importance_df(cat_est, X_imp, y_imp, kind="PredictionValuesChange")
                        self.test_details_["cat_model_importance_agg"] = fi_agg.head(self.cfg.EXPLAIN_TOPK)
                    if self.cfg.EXPLAIN_COMPUTE_SHAP_GLOBAL:
                        shap_ohe, shap_agg, _ = cat_shap_global_df(cat_est, X_imp, y_imp)
                        self.test_details_["cat_model_shap_agg"] = shap_agg.head(self.cfg.EXPLAIN_TOPK)
                    if self.cfg.EXPLAIN_COMPUTE_INTERACTIONS:
                        ints_ohe, ints_agg = cat_interactions_df(cat_est, X_imp, y_imp, top_k=self.cfg.EXPLAIN_TOPK)
                        self.test_details_["cat_model_interactions_agg"] = ints_agg
                    if self.cfg.EXPLAIN_COMPUTE_SHAP_INTERACTIONS:
                        sints_ohe, sints_agg = shap_interactions_df(cat_est, X_imp, y_imp, top_k=self.cfg.EXPLAIN_TOPK)
                        self.test_details_["cat_model_shap_interactions_agg"] = sints_agg
            except Exception as e:
                logger.warning("CatBoost explanations failed: %s: %s", type(e).__name__, e)

            # --- EBM per-run explanations (always if available) ---
            try:
                ebm_est = cand_map.get("ebm", (None,))[0]
                if ebm_est is not None:
                    fmap, term_table, top_raw_features, top_interactions = ebm_feature_and_term_tables_from_full(ebm_est, self.cfg)
                    feat_agg_all = ebm_aggregate_base_feature_importance(term_table)
                    self.test_details_["ebm_feature_importance_agg"] = feat_agg_all
                    self.test_details_["ebm_interactions_table"] = top_interactions
                    self.test_details_["ebm_model_importance_agg"] = feat_agg_all
                    pair_agg = _pairs_from_interactions_table(top_interactions)
                    self.test_details_["ebm_model_interactions_agg"] = pair_agg
            except Exception as e:
                logger.warning("EBM explanations failed: %s: %s", type(e).__name__, e)

            # --- Generic mirrors for the chosen best model (back-compat) ---
            if best_name == "cat":
                self.test_details_["cat_importance_agg"] = self.test_details_["cat_model_importance_agg"].copy()
                self.test_details_["cat_shap_agg"] = self.test_details_["cat_model_shap_agg"].copy()
                self.test_details_["cat_interactions_agg"] = self.test_details_["cat_model_interactions_agg"].copy()
                self.test_details_["shap_interactions_agg"] = self.test_details_["cat_model_shap_interactions_agg"].copy()
            elif best_name == "ebm":
                # mirror EBM into generic keys for downstream saving
                feat_agg_all = self.test_details_["ebm_model_importance_agg"]
                pair_agg = self.test_details_["ebm_model_interactions_agg"]
                self.test_details_["cat_importance_agg"] = feat_agg_all.rename(columns={"base_feature":"orig_feature"})
                self.test_details_["cat_importance_ohe"] = self.test_details_["cat_importance_agg"].rename(
                    columns={"orig_feature":"feature","importance":"importance"}).assign(orig_feature=lambda d: d["feature"])
                self.test_details_["cat_interactions_agg"] = pair_agg
                self.test_details_["cat_interactions_ohe"] = pair_agg.rename(
                    columns={"orig_i":"feat_i","orig_j":"feat_j","score":"score"}).assign(orig_i=lambda d: d["feat_i"], orig_j=lambda d: d["feat_j"])
                # SHAP not available for EBM; leave generic SHAP empty
        except Exception as e:
            print(f"[Explanations] {type(e).__name__}: {e}")
        # ---------------------------------------------------------------------

        # Example scores for first test patient
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

        # stash
        self.test_details_ = {
            **self.test_details_,
            "models": details_by_model,
            "therapist_recommendation_counts_by_model": therapist_reco_by_model,
            "used_features": {
                "patient_numeric": p_num_u, "patient_categorical": p_cat_u,
                "therapist_numeric": t_num_u, "therapist_categorical": t_cat_u,
            },
            "best_model_name": best_name,
            "best_params": dict(best_gs.best_params_),
            "example_scores_first_test_patient": example_scores,
            "rank_outcome_table_by_model": {k: v.get("rank_outcome_table") for k, v in details_by_model.items()},
        }
        ther_table_out = therapists_table_from_train(self.cfg, df_train)
        best_reco = therapist_reco_by_model.get(best_name, pd.DataFrame())
        return {"summary": df_sum, "therapists": ther_table_out, "therapist_recommendations": best_reco}

# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_pipeline(df: pd.DataFrame, cfg: MatchConfig):
    """
    If cfg.CV_N_REPEATS > 1 and CV_STRATEGY='per_therapist_random', we:
      - repeat the full split+fit+test 'CV_N_REPEATS' times
      - aggregate RMSE across runs (per model + 'best-by-policy')
      - aggregate importances & interactions for CatBoost and EBM separately (across ALL runs)
      - aggregate CatBoost SHAP main & interactions across ALL runs (even if CatBoost wasn't best)
      - aggregate outcomes (rank_outcome_table) across runs for best-by-policy
    Otherwise, perform a single run (backward compatible).
    """
    n_rep = int(getattr(cfg, "CV_N_REPEATS", 1) or 1)
    strat = (cfg.CV_STRATEGY or "leave_last").lower()

    # Single run (legacy path)
    if not (n_rep > 1 and strat in {"per_therapist_random", "per-therapist-random", "ptr"}):
        pipe = MatchPipeline(cfg)
        pipe.set_data(df)
        pipe.set_repeat_index(0)
        out = pipe.run()
        out["used_features"] = pipe.test_details_.get("used_features", {})
        out["best_model_name"] = pipe.test_details_.get("best_model_name")
        out["best_params"] = pipe.test_details_.get("best_params")
        out["example_scores_first_test_patient"] = pipe.test_details_.get("example_scores_first_test_patient", pd.DataFrame())

        # keep outputs as DataFrames
        def _df_or_empty(df_obj, cols):
            return df_obj if isinstance(df_obj, pd.DataFrame) else pd.DataFrame(columns=cols)

        # Generic keys reused for whichever model won (CatBoost or EBM)
        out["cat_importance_ohe"] = _df_or_empty(pipe.test_details_.get("cat_importance_ohe"),
                                                 ["feature","importance","orig_feature"])
        out["cat_importance_agg"] = _df_or_empty(pipe.test_details_.get("cat_importance_agg"),
                                                 ["orig_feature","importance"])
        out["cat_shap_ohe"] = _df_or_empty(pipe.test_details_.get("cat_shap_ohe"),
                                           ["feature","mean_abs_shap","orig_feature"])
        out["cat_shap_agg"] = _df_or_empty(pipe.test_details_.get("cat_shap_agg"),
                                           ["orig_feature","mean_abs_shap"])
        out["cat_interactions_ohe"] = _df_or_empty(pipe.test_details_.get("cat_interactions_ohe"),
                                                   ["feat_i","feat_j","score","orig_i","orig_j"])
        out["cat_interactions_agg"] = _df_or_empty(pipe.test_details_.get("cat_interactions_agg"),
                                                   ["orig_i","orig_j","score"])
        out["shap_interactions_ohe"] = _df_or_empty(pipe.test_details_.get("shap_interactions_ohe"),
                                                   ["feat_i","feat_j","score","orig_i","orig_j"])
        out["shap_interactions_agg"] = _df_or_empty(pipe.test_details_.get("shap_interactions_agg"),
                                                   ["orig_i","orig_j","score"])

        # Explicit per-model (NEW)
        out["cat_model_importance_agg"] = _df_or_empty(pipe.test_details_.get("cat_model_importance_agg"),
                                                       ["orig_feature","importance"])
        out["cat_model_interactions_agg"] = _df_or_empty(pipe.test_details_.get("cat_model_interactions_agg"),
                                                         ["orig_i","orig_j","score"])
        out["cat_model_shap_agg"] = _df_or_empty(pipe.test_details_.get("cat_model_shap_agg"),
                                                 ["orig_feature","mean_abs_shap"])
        out["cat_model_shap_interactions_agg"] = _df_or_empty(pipe.test_details_.get("cat_model_shap_interactions_agg"),
                                                              ["orig_i","orig_j","score"])

        out["ebm_feature_importance_agg"] = _df_or_empty(pipe.test_details_.get("ebm_feature_importance_agg"),
                                                         ["base_feature","importance"])
        out["ebm_interactions_table"] = _df_or_empty(pipe.test_details_.get("ebm_interactions_table"),
                                                     ["term_index","term_name","pretty","importance","features_transformed","features_base"])
        out["ebm_model_importance_agg"] = _df_or_empty(pipe.test_details_.get("ebm_model_importance_agg"),
                                                       ["base_feature","importance"])
        out["ebm_model_interactions_agg"] = _df_or_empty(pipe.test_details_.get("ebm_model_interactions_agg"),
                                                         ["orig_i","orig_j","score"])

        # per-patient rank/outcome table for the selected best model
        best_name = out.get("best_model_name")
        rtab_map = pipe.test_details_.get("rank_outcome_table_by_model", {}) or {}
        out["rank_outcome_table"] = _df_or_empty(rtab_map.get(best_name),
                                                 [cfg.PATIENT_ID_COL,"factual_therapist_id","factual_rank",
                                                  cfg.LABEL_COL,"y_eff","mu_hat_factual"])
        return out

    # ----- Repeated runs with aggregation -----
    all_run_summaries = []
    all_best_rows = []
    all_fi_bestmirror = []  # (legacy) mirror of best only
    all_rank_tabs = []
    outs_last = None

    # NEW accumulators across ALL runs per model
    cat_fi_runs, ebm_fi_runs = [], []
    cat_inter_runs, ebm_inter_runs = [], []
    shap_main_runs, shap_inter_runs = [], []  # CatBoost SHAP

    for r in range(n_rep):
        pipe = MatchPipeline(cfg)
        pipe.set_data(df)
        pipe.set_repeat_index(r)
        out_r = pipe.run()
        outs_last = out_r  # keep last run's main tables for convenience

        # summary per model
        df_sum = out_r["summary"].copy()
        df_sum["run"] = r
        all_run_summaries.append(df_sum)

        # best-of-run row (by your SELECT_BEST_BY policy inside .run())
        best_model_name = pipe.test_details_.get("best_model_name")
        best_row = df_sum.loc[df_sum["model"] == best_model_name].copy()
        if len(best_row):
            best_row["run"] = r
            all_best_rows.append(best_row.iloc[[0]])

        # (legacy) aggregate FI for whichever model was mirrored into generic key
        fi_mirror = pipe.test_details_.get("cat_importance_agg")
        if isinstance(fi_mirror, pd.DataFrame) and len(fi_mirror):
            tmp = fi_mirror.copy()
            tmp["run"] = r
            all_fi_bestmirror.append(tmp)

        # outcomes per test patient for the best model in this run
        rtab_map = pipe.test_details_.get("rank_outcome_table_by_model", {}) or {}
        rtab = rtab_map.get(best_model_name)
        if isinstance(rtab, pd.DataFrame) and len(rtab):
            rt = rtab.copy()
            rt["run"] = r
            all_rank_tabs.append(rt)

        # -------- NEW: collect per-model tables (ALL runs, regardless of best) --------
        # CatBoost
        cfi = pipe.test_details_.get("cat_model_importance_agg")
        if isinstance(cfi, pd.DataFrame) and len(cfi):
            t = cfi.rename(columns={"orig_feature": "feature", "importance": "importance"}).copy()
            t["run"] = r
            cat_fi_runs.append(t[["feature", "importance", "run"]])
        cint = pipe.test_details_.get("cat_model_interactions_agg")
        if isinstance(cint, pd.DataFrame) and len(cint):
            t = cint.copy()
            t["i"] = t["orig_i"].astype(str); t["j"] = t["orig_j"].astype(str)
            t["run"] = r
            cat_inter_runs.append(t[["i", "j", "score", "run"]])
        cshap = pipe.test_details_.get("cat_model_shap_agg")
        if isinstance(cshap, pd.DataFrame) and len(cshap):
            t = cshap.rename(columns={"orig_feature":"feature","mean_abs_shap":"value"}).copy()
            t["run"] = r
            shap_main_runs.append(t[["feature","value","run"]])
        cshapint = pipe.test_details_.get("cat_model_shap_interactions_agg")
        if isinstance(cshapint, pd.DataFrame) and len(cshapint):
            t = cshapint.copy()
            t["i"] = t["orig_i"].astype(str); t["j"] = t["orig_j"].astype(str)
            t["run"] = r
            shap_inter_runs.append(t[["i","j","value" if "value" in t.columns else "score","run"]].rename(
                columns={"score":"value"}))

        # EBM
        efi = pipe.test_details_.get("ebm_model_importance_agg")
        if isinstance(efi, pd.DataFrame) and len(efi):
            t = efi.rename(columns={"base_feature":"feature","importance":"importance"}).copy()
            t["run"] = r
            ebm_fi_runs.append(t[["feature","importance","run"]])
        eint = pipe.test_details_.get("ebm_model_interactions_agg")
        if isinstance(eint, pd.DataFrame) and len(eint):
            t = eint.copy()
            t["i"] = t["orig_i"].astype(str); t["j"] = t["orig_j"].astype(str)
            t["run"] = r
            ebm_inter_runs.append(t[["i","j","score","run"]])
        # ---------------------------------------------------------------------------

    # Stack per-run summaries
    df_runs = pd.concat(all_run_summaries, ignore_index=True) if len(all_run_summaries) else pd.DataFrame()
    df_best = pd.concat(all_best_rows, ignore_index=True) if len(all_best_rows) else pd.DataFrame()

    # RMSE stats per model
    rmse_stats_per_model = pd.DataFrame()
    if len(df_runs):
        rmse_stats_per_model = (df_runs.groupby("model", as_index=False)["test_rmse"]
                                .agg(mean="mean", std="std", min="min", max="max", n="count")
                                .sort_values("mean").reset_index(drop=True))

    # RMSE stats for best-by-policy across runs
    rmse_stats_best = pd.DataFrame()
    if len(df_best):
        rmse_stats_best = (df_best["test_rmse"]
                           .agg(["mean","std","min","max"])
                           .to_frame(name="best_policy_test_rmse")
                           .T.reset_index(drop=True))
        rmse_stats_best["n_runs"] = len(df_best)

    # (legacy) Aggregate FI across runs of the selected model (best each time)
    fi_mean_bestmirror = pd.DataFrame()
    if len(all_fi_bestmirror):
        df_fi = pd.concat(all_fi_bestmirror, ignore_index=True)
        fi_mean_bestmirror = (df_fi.groupby("orig_feature", as_index=False)["importance"]
                              .agg(mean_importance="mean", std_importance="std", runs_present="count")
                              .sort_values("mean_importance", ascending=False).reset_index(drop=True))
        s = fi_mean_bestmirror["mean_importance"].sum()
        if s > 0:
            fi_mean_bestmirror["mean_importance_norm"] = fi_mean_bestmirror["mean_importance"] / s

    # ---- NEW aggregation helpers (for ALL runs per model) ----
    def _agg_feat_mean_std(df: pd.DataFrame, val_col: str):
        if df is None or not len(df): 
            return pd.DataFrame(columns=["feature", f"mean_{val_col}", f"std_{val_col}", "runs_present", f"mean_{val_col}_norm"])
        g = (df.groupby("feature", as_index=False)[val_col]
               .agg(**{f"mean_{val_col}": "mean", f"std_{val_col}": "std", "runs_present": "count"})
               .sort_values(f"mean_{val_col}", ascending=False).reset_index(drop=True))
        s = g[f"mean_{val_col}"].sum()
        g[f"mean_{val_col}_norm"] = g[f"mean_{val_col}"] / s if s > 0 else 0.0
        return g

    def _agg_pair_mean_std(df: pd.DataFrame, val_col: str, out_mean_name: str, out_std_name: str):
        if df is None or not len(df):
            return pd.DataFrame(columns=["orig_i", "orig_j", out_mean_name, out_std_name, "runs_present"])
        d = df.copy()
        a = d["i"].astype(str); b = d["j"].astype(str)
        left = np.where(a <= b, a, b)
        right = np.where(a <= b, b, a)
        d["orig_i"] = left; d["orig_j"] = right
        g = (d.groupby(["orig_i", "orig_j"], as_index=False)[val_col]
               .agg(**{out_mean_name: "mean", out_std_name: "std", "runs_present": "count"})
               .sort_values(out_mean_name, ascending=False).reset_index(drop=True))
        return g

    # Build concatenated per-model frames
    cat_fi_df  = pd.concat(cat_fi_runs,  ignore_index=True) if len(cat_fi_runs)  else pd.DataFrame(columns=["feature","importance","run"])
    ebm_fi_df  = pd.concat(ebm_fi_runs,  ignore_index=True) if len(ebm_fi_runs)  else pd.DataFrame(columns=["feature","importance","run"])
    cat_int_df = pd.concat(cat_inter_runs, ignore_index=True) if len(cat_inter_runs) else pd.DataFrame(columns=["i","j","score","run"])
    ebm_int_df = pd.concat(ebm_inter_runs, ignore_index=True) if len(ebm_inter_runs) else pd.DataFrame(columns=["i","j","score","run"])
    shap_main_df = pd.concat(shap_main_runs, ignore_index=True) if len(shap_main_runs) else pd.DataFrame(columns=["feature","value","run"])
    shap_int_df  = pd.concat(shap_inter_runs, ignore_index=True) if len(shap_inter_runs) else pd.DataFrame(columns=["i","j","value","run"])

    # Cross-run means (ALL runs per model)
    cv_cat_importance_mean = _agg_feat_mean_std(cat_fi_df,  "importance")
    cv_ebm_importance_mean = _agg_feat_mean_std(ebm_fi_df,  "importance")
    cv_cat_interactions_mean = _agg_pair_mean_std(cat_int_df, "score", "mean_score", "std_score")
    cv_ebm_interactions_mean = _agg_pair_mean_std(ebm_int_df, "score", "mean_score", "std_score")

    # CatBoost SHAP (ALL runs)
    cv_cat_shap_main_mean         = _agg_feat_mean_std(shap_main_df, "value").rename(columns={
        "mean_value": "mean_abs_shap", "std_value": "std_abs_shap", "mean_value_norm": "mean_abs_shap_norm"
    })
    cv_cat_shap_interactions_mean = _agg_pair_mean_std(shap_int_df, "value", "mean_abs_interaction", "std_abs_interaction")

    # Aggregate outcomes across runs (best model each run)
    rank_outcome_all = pd.concat(all_rank_tabs, ignore_index=True) if len(all_rank_tabs) else pd.DataFrame()

    # Compose final output: keep last run's regular outputs + add cv aggregates
    out = dict(outs_last or {})
    out["cv_n_repeats"] = n_rep
    out["cv_runs_summary"] = df_runs
    out["cv_best_policy_summary"] = df_best
    out["cv_rmse_stats_per_model"] = rmse_stats_per_model
    out["cv_rmse_stats_best"] = rmse_stats_best

    # (legacy best-by-policy FI)
    out["cv_importance_agg_mean"] = fi_mean_bestmirror

    # NEW per-model / all-runs aggregates
    out["cv_cat_importance_mean"] = cv_cat_importance_mean
    out["cv_ebm_importance_mean"] = cv_ebm_importance_mean
    out["cv_cat_interactions_mean"] = cv_cat_interactions_mean
    out["cv_ebm_interactions_mean"] = cv_ebm_interactions_mean

    # NEW CatBoost SHAP all-runs aggregates
    out["cv_cat_shap_main_mean"] = cv_cat_shap_main_mean
    out["cv_cat_shap_interactions_mean"] = cv_cat_shap_interactions_mean

    out["cv_rank_outcome_all_runs"] = rank_outcome_all  # aggregated outcomes
    out["cv_outcomes_all_runs"] = rank_outcome_all      # alias for clarity

    # also pass through details from the last run for convenience
    return out
