# -*- coding: utf-8 -*-

import os
import sys
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.model_selection import GridSearchCV, PredefinedSplit, BaseCrossValidator, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from typing import Any, Dict, List, Union

import statsmodels.api as sm
from pandas.api import types as pdt

# ----- Torch FM -----
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from joblib import Memory

# ----- CatBoost -----
from catboost import CatBoostRegressor, Pool, EFstrType

import json
from pathlib import Path
from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # fallback

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

    # ---- Target ----
    LABEL_COL: str = "t1_distress_sum"
    OUTCOME_HIGHER_IS_BETTER: bool = False  # distress: lower is better
    STANDARDIZE_Y: bool = True

    # ---- Features ----
    patient_numeric: List[str] = field(default_factory=list)
    patient_categorical: List[str] = field(default_factory=list)
    therapist_numeric: List[str] = field(default_factory=list)
    therapist_categorical: List[str] = field(default_factory=list)
    include_therapist_id_feature: bool = False

    # ---- Model selection ----
    # Choose which models to run. Valid: ["fm", "ridge", "cat", "ebm"]
    RUN_MODELS: List[str] = field(default_factory=lambda: ["fm", "ridge", "cat", "ebm"])

    # ---- Benchmarks ----
    INCLUDE_NAIVE_MEAN_BASELINE: bool = True
    INCLUDE_PATIENT_ONLY_BENCHMARK: bool = True

    # ---- Imputation ----
    impute_strategy: str = "median"   # "median" | "iterative" | "mean"
    iterative_imputer_params: Dict[str, object] = field(default_factory=lambda: {
        "max_iter": 5,
        "tol": 1e-3,
        "verbose": 0,
        "initial_strategy": "median",
        "random_state": 0,
        "sample_posterior": False,
    })

    # ---- Caching ----
    # Not necessary by default; can be turned on for experimentation.
    CACHE_STEPS: bool = False
    CACHE_DIR: str = field(default_factory=lambda: os.environ.get("SKCACHE_DIR", "./.sk_cache"))

    # ---- Output persistence ----
    OUTPUT_SAVE: bool = True
    OUTPUT_BASE_DIR: str = "./model_outputs"
    OUTPUT_TIMESTAMP_FMT: str = "%Y%m%d_%H"   # day+hour
    OUTPUT_RUN_DIR: Optional[str] = None      # filled automatically at runtime
    OUTPUT_SAVE_MODELS: bool = True
    OUTPUT_FORMAT: str = "csv"                # "csv" (default)

    # ---- CatBoost file writing ----
    CATBOOST_ALLOW_WRITING_FILES: bool = True

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
    ebm_param_grid: Dict[str, List] = field(default_factory=lambda: {
        "regressor__interactions": [100, 200],
        "regressor__validation_size": [0.15],
        "regressor__learning_rate": [0.04],
        "regressor__max_bins": [128],
        "regressor__max_leaves": [2, 3],
        "regressor__smoothing_rounds": [750, 1000],
        "regressor__min_samples_leaf": [2, 5],
        "regressor__n_jobs": [-1],
        "regressor__random_state": [42],
    })

    # ---- Selection policy ----
    SELECT_BEST_BY: str = "test"  # fixed as requested

    # ---- (optional) therapist recommendation tallies ----
    REPORT_TOP_COUNTS: bool = True

    # ---- CatBoost categorical handling ----
    CAT_USE_NATIVE_CATEGORICAL: bool = True   # True => native cats; False => OHE
    CAT_MIN_FREQ_FOR_OHE: int = 1              # min_frequency if using OHE
    CAT_NATIVE_HANDLE_MISSING: bool = False

    # ---- Explanations ----
    EXPLAIN_COMPUTE_FI: bool = True
    EXPLAIN_COMPUTE_SHAP_GLOBAL: bool = True
    EXPLAIN_COMPUTE_INTERACTIONS: bool = True
    EXPLAIN_COMPUTE_SHAP_INTERACTIONS: bool = True
    EXPLAIN_SAMPLE_ROWS: Optional[int] = None
    EXPLAIN_TOPK: int = 50

    # ---- Feature selection (all features) ----
    DROP_ZERO_VARIANCE_FEATURES: bool = True
    ZERO_VAR_COUNT_NAN_AS_VALUE: bool = False



    # ---- RMSE CI options (bootstrap only; delta removed) ----
    RMSE_CI_ENABLE: bool = True
    RMSE_CI_LEVEL: float = 0.95
    RMSE_CI_BOOTSTRAP_B: int = 2000
    RMSE_CI_RANDOM_STATE: int = 0

    # ---- Cross-validation / repeat strategy ----
    # "leave_last"                 : time-ordered per therapist (train/val/test as implemented)
    # "per_therapist_random"       : random per-therapist train/val/test; supports outer repeats via CV_N_REPEATS
    # "outer_leave_last_inner_repeated": outer leave-last TEST (time-ordered), inner repeated per-therapist random holdout for HPO
    # "kfold"
    CV_STRATEGY: str = "per_therapist_random"
    CV_N_REPEATS: int = 5          # only used when CV_STRATEGY == "per_therapist_random" (outer repeats)
    CV_RANDOM_STATE: int = 42

    # ---- Inner tuning CV (only used when CV_STRATEGY == "outer_leave_last_inner_repeated") ----
    INNER_N_REPEATS: int = 5
    INNER_RANDOM_STATE: int = 123
    INNER_VAL_PER_THERAPIST: int = 1          # how many cases per therapist go to inner validation in each repeat
    INNER_MIN_N_FOR_VAL: int = 1              # therapists with < this many cases contribute no inner-val cases
    INNER_FALLBACK_GLOBAL_VAL_FRAC: float = 0.10  # used only if an inner split ends up with empty val

    # ---- KFold (row-wise) outer evaluation ----
    KFOLD_N_SPLITS: int = 5
    KFOLD_SHUFFLE: bool = True
    KFOLD_VAL_FRAC: float = 0.10   # fraction of the fold-train set used as "val" for PredefinedSplit HPO



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
warnings.filterwarnings("ignore", category=FutureWarning, message=r"This Pipeline instance is not fitted yet.*")
NEG_LARGE = -1e12

def _drop_zero_variance_feature_lists(
    df_ref: pd.DataFrame,
    *,
    p_num: List[str], p_cat: List[str],
    t_num: List[str], t_cat: List[str],
    count_nan_as_value: bool = False,
) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, str]]:
    """
    Drops features that have <=1 unique value in df_ref (training only).
    Applies to patient + therapist features.
    - If count_nan_as_value=False: ignore NaNs for uniqueness; all-NaN => drop.
    Returns filtered lists and a dict {feature: reason}.
    """
    dropped: Dict[str, str] = {}

    def nunique_series(s: pd.Series) -> int:
        if count_nan_as_value:
            return int(s.nunique(dropna=False))
        return int(s.dropna().nunique())  # all-NaN => 0

    def filter_list(cols: List[str]) -> List[str]:
        out = []
        for c in cols:
            if c not in df_ref.columns:
                out.append(c)
                continue
            nuniq = nunique_series(df_ref[c])
            if nuniq <= 1:
                dropped[c] = f"zero_variance(n_unique={nuniq})"
            else:
                out.append(c)
        return out

    p_num_out = filter_list(list(p_num or []))
    p_cat_out = filter_list(list(p_cat or []))
    t_num_out = filter_list(list(t_num or []))
    t_cat_out = filter_list(list(t_cat or []))

    return p_num_out, p_cat_out, t_num_out, t_cat_out, dropped



def _now_berlin(cfg: MatchConfig) -> datetime:
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo("Europe/Berlin"))
        except Exception:
            pass
    return datetime.now()

def _make_run_dir(cfg: MatchConfig) -> Path:
    base = Path(cfg.OUTPUT_BASE_DIR)
    base.mkdir(parents=True, exist_ok=True)

    stamp = _now_berlin(cfg).strftime(cfg.OUTPUT_TIMESTAMP_FMT)
    run_dir = base / f"run_{stamp}"

    # avoid collisions (hour-level timestamp can collide)
    if run_dir.exists():
        k = 2
        while True:
            cand = base / f"run_{stamp}_v{k}"
            if not cand.exists():
                run_dir = cand
                break
            k += 1

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _init_run_context(cfg: MatchConfig, *, repeat_index: Optional[int] = None) -> Path:
    """
    Ensures cfg.OUTPUT_RUN_DIR is set and returns:
      - base run_dir if repeat_index is None
      - run_dir/repeat_XX if repeat_index is provided
    Also scopes cache under the same folder when CACHE_STEPS=True.
    """
    base = Path(cfg.OUTPUT_RUN_DIR) if cfg.OUTPUT_RUN_DIR else _make_run_dir(cfg)
    cfg.OUTPUT_RUN_DIR = str(base)

    if repeat_index is None:
        out_dir = base
    else:
        out_dir = base / f"repeat_{int(repeat_index):02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

    if getattr(cfg, "CACHE_STEPS", False):
        # isolate cache per (repeat) run to avoid cross-run contamination
        cfg.CACHE_DIR = str(out_dir / "sk_cache")
        Path(cfg.CACHE_DIR).mkdir(parents=True, exist_ok=True)

    return out_dir


def _ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def _json_sanitize(obj):
    # convert numpy types to python scalars for json
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def _save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=_json_sanitize)

def _save_df(df: pd.DataFrame, path: str) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        return
    if not len(df):
        # still write an empty file with headers for reproducibility
        df.to_csv(path, index=False)
        return
    df.to_csv(path, index=False)

def _save_npz(path: str, **arrays) -> None:
    # arrays can include lists; numpy will store them (object) if needed
    np.savez_compressed(path, **arrays)

def _save_outputs_bundle(out_dir: str, out: Dict[str, object]) -> None:
    """
    Saves the user-facing outputs returned by run_pipeline(...) plus extras if present.
    """
    _ensure_dir(out_dir)
    tables_dir = _ensure_dir(os.path.join(out_dir, "tables"))
    explain_dir = _ensure_dir(os.path.join(out_dir, "explanations"))
    models_dir = _ensure_dir(os.path.join(out_dir, "models"))
    meta_dir = _ensure_dir(os.path.join(out_dir, "meta"))

    # core tables
    if isinstance(out.get("summary"), pd.DataFrame):
        _save_df(out["summary"], os.path.join(tables_dir, "summary.csv"))
    if isinstance(out.get("therapists"), pd.DataFrame):
        _save_df(out["therapists"], os.path.join(tables_dir, "therapists.csv"))
    if isinstance(out.get("therapist_recommendations"), pd.DataFrame):
        _save_df(out["therapist_recommendations"], os.path.join(tables_dir, "therapist_recommendations.csv"))

    # single-run extras (from pipe.test_details_)
    for key, fname in [
        ("example_scores_first_test_patient", "example_scores_first_test_patient.csv"),
        ("cat_model_importance_agg", "cat_fi_agg.csv"),
        ("cat_model_interactions_agg", "cat_interactions_agg.csv"),
        ("cat_model_shap_agg", "cat_shap_feature_importance_agg.csv"),
        ("cat_model_shap_interactions_agg", "cat_shap_interactions_agg.csv"),
        ("ebm_model_importance_agg", "ebm_fi_agg.csv"),
        ("ebm_model_interactions_agg", "ebm_interactions_agg.csv"),
        ("rank_outcome_table", "rank_outcome_table_best_model.csv"),
    ]:
        if isinstance(out.get(key), pd.DataFrame):
            _save_df(out[key], os.path.join(explain_dir, fname))

    # repeated-run aggregate tables (if present)
    for key, fname in [
        ("cv_runs_summary", "cv_runs_summary.csv"),
        ("cv_best_policy_summary", "cv_best_policy_summary.csv"),
        ("cv_rmse_stats_per_model", "cv_rmse_stats_per_model.csv"),
        ("cv_rmse_stats_best", "cv_rmse_stats_best.csv"),
        ("cv_cat_importance_mean", "cv_cat_importance_mean.csv"),
        ("cv_ebm_importance_mean", "cv_ebm_importance_mean.csv"),
        ("cv_cat_interactions_mean", "cv_cat_interactions_mean.csv"),
        ("cv_ebm_interactions_mean", "cv_ebm_interactions_mean.csv"),
        ("cv_cat_shap_main_mean", "cv_cat_shap_main_mean.csv"),
        ("cv_cat_shap_interactions_mean", "cv_cat_shap_interactions_mean.csv"),
        ("cv_rank_outcome_all_runs", "cv_rank_outcome_all_runs.csv"),
    ]:
        if isinstance(out.get(key), pd.DataFrame):
            _save_df(out[key], os.path.join(tables_dir, fname))

    # metadata
    meta = {
        "output_dir": out.get("output_dir"),
        "best_model_name": out.get("best_model_name"),
        "best_params": out.get("best_params"),
        "used_features": out.get("used_features"),
        "cv_n_repeats": out.get("cv_n_repeats"),
        "run_models": out.get("run_models"),
    }
    _save_json(os.path.join(meta_dir, "run_meta.json"), meta)

    # models (optional)
    if out.get("_models_to_save") and isinstance(out["_models_to_save"], dict):
        for name, est in out["_models_to_save"].items():
            try:
                import joblib
                joblib.dump(est, os.path.join(models_dir, f"{name}.joblib"))
            except Exception as e:
                logger.warning("Could not save model %s: %s", name, e)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _rmse_ci_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    level: float = 0.95,
    B: int = 2000,
    rng: int = 0
) -> Tuple[float, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    rs = np.random.RandomState(rng)
    stats = np.empty(B, float)
    for b in range(B):
        idx = rs.randint(0, n, size=n)
        stats[b] = np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))
    lo = np.percentile(stats, (1 - level) / 2 * 100)
    hi = np.percentile(stats, (1 + level) / 2 * 100)
    return float(lo), float(hi)


def _dedupe(seq):
    return list(dict.fromkeys(seq))


def _ohe(sparse_output=True, min_frequency=1):
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output, min_frequency=min_frequency)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=(sparse_output))


def _s_eff(scores: np.ndarray, cfg: MatchConfig) -> np.ndarray:
    """Convert predictions to 'higher is better' scores for sorting."""
    scores = np.asarray(scores, float)
    return scores if cfg.OUTCOME_HIGHER_IS_BETTER else -scores


def _model_name_from_estimator(estimator: Pipeline) -> str:
    reg = estimator.named_steps.get("regressor")
    if isinstance(reg, TransformedTargetRegressor):
        inner = getattr(reg, "regressor", None)
        return type(inner).__name__ if inner is not None else type(reg).__name__
    return type(reg).__name__ if reg is not None else type(estimator).__name__


# Treat already-binary dummy columns as numeric in the native-cat pipeline
_DUMMY_PREFIXES = (
    "t0_method_t_", "t0_appliedmethod_t_", "t0_disease_p_",
    "t0_therapy_p_", "t0_migration_p_",
)
_DUMMY_EXACT = {"t0_familyhistory_p", "t0_meds_p"}

def _is_preencoded_dummy(col: str) -> bool:
    return (col in _DUMMY_EXACT) or any(str(col).startswith(p) for p in _DUMMY_PREFIXES)


#####################################################################
# Picklable FunctionTransformer helpers
#####################################################################
def _to_numeric_df(X):
    if sparse.issparse(X):
        X = X.toarray()
    if isinstance(X, pd.DataFrame):
        return X.apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")


def _to_str_df(X):
    if sparse.issparse(X):
        X = X.toarray()
    if isinstance(X, pd.DataFrame):
        return X.astype(str)
    return pd.DataFrame(X).astype(str)

def _to_str_preserve_nan_df(X):
    """
    Convert values to strings but KEEP missing values as NaN (do NOT turn NaN into 'nan').
    CatBoost then handles missing categoricals natively.
    """
    if sparse.issparse(X):
        X = X.toarray()

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    X = X.copy()
    for c in X.columns:
        s = X[c]
        mask = s.isna()
        s2 = s.astype("object")

        # convert only non-missing to str
        s2.loc[~mask] = s2.loc[~mask].map(lambda v: str(v))

        # enforce actual NaN for missing
        s2.loc[mask] = np.nan
        X[c] = s2

    return X


def _to_float_preserve_nan_df(X):
    """
    Coerce to float, preserve NaN, and replace +/-inf with NaN.
    """
    if sparse.issparse(X):
        X = X.toarray()
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    Xn = X.apply(pd.to_numeric, errors="coerce").astype(float)
    Xn = Xn.replace([np.inf, -np.inf], np.nan)
    return Xn



def _FT(func):
    try:
        return FunctionTransformer(func=func, feature_names_out="one-to-one", validate=False)
    except TypeError:
        return FunctionTransformer(func=func, validate=False)


#####################################################################
# Torch FM
#####################################################################
class _FMLayer(nn.Module):
    def __init__(self, n_feats: int, k: int):
        super().__init__()
        self.V = nn.Parameter(torch.randn(n_feats, k) * 0.01)

    def forward(self, x):
        xv = x @ self.V
        xv2 = xv.pow(2)
        x2 = x.pow(2)
        v2 = self.V.pow(2)
        x2v2 = x2 @ v2
        return 0.5 * (xv2 - x2v2).sum(dim=1, keepdim=True)


class _FMTorch(nn.Module):
    def __init__(self, n_feats: int, k: int):
        super().__init__()
        self.linear = nn.Linear(n_feats, 1, bias=True)
        self.fm = _FMLayer(n_feats, k)

    def forward(self, x):
        return self.linear(x) + self.fm(x)


class TorchFMRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        *,
        k=16,
        lr=1e-3,
        n_epochs=40,
        batch_size=512,
        weight_decay=0.0,
        device: Optional[str] = None,
        random_state=42,
        verbose=False
    ):
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

        for _epoch in range(self.n_epochs):
            self._model.train()
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = crit(self._model(xb), yb)
                loss.backward()
                opt.step()
        return self

    def predict(self, X):
        if sparse.issparse(X):
            X = X.todense()
        X = np.asarray(X, dtype=np.float32)
        self._model.eval()
        with torch.no_grad():
            return self._model(torch.from_numpy(X).to(self.device)).cpu().numpy().ravel()

    def get_params(self, deep=True):
        return {
            "k": self.k,
            "lr": self.lr,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "device": self.device,
            "random_state": self.random_state,
            "verbose": self.verbose,
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

def split_kfold(
    df: pd.DataFrame,
    *,
    n_splits: int,
    shuffle: bool,
    random_state: int,
    repeat_index: int,
    val_frac: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Row-wise KFold outer split (like standard ML papers):
      - TEST: one fold
      - TRAIN/VAL: remaining folds, with VAL sampled globally from TRAIN according to val_frac
    """
    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError("KFOLD_N_SPLITS must be >= 2.")

    kf = KFold(n_splits=n_splits, shuffle=bool(shuffle), random_state=int(random_state))
    splits = list(kf.split(df))
    fold = int(repeat_index) % n_splits
    train_idx, test_idx = splits[fold]

    df_test = df.iloc[test_idx].copy().reset_index(drop=True)
    df_trainval = df.iloc[train_idx].copy().reset_index(drop=True)

    # Global val sampled from fold-train
    df_val = df.iloc[0:0].copy()
    df_train = df_trainval

    n_tv = len(df_trainval)
    if val_frac and n_tv >= 3:
        n_val = int(round(float(val_frac) * n_tv))
        n_val = max(1, n_val)
        n_val = min(n_val, n_tv - 1)  # keep at least 1 in train

        rs = np.random.RandomState(int(random_state) + 10_000 + fold)
        val_idx = rs.choice(np.arange(n_tv), size=n_val, replace=False)

        df_val = df_trainval.iloc[val_idx].copy().reset_index(drop=True)
        df_train = df_trainval.drop(df_trainval.index[val_idx]).reset_index(drop=True)

    return df_train, df_val, df_test


def split_leave_last(df: pd.DataFrame, therapist_col: str, date_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    df_train = pd.concat(tr).reset_index(drop=True) if tr else df.iloc[0:0].copy()
    df_val = pd.concat(va).reset_index(drop=True) if va else df.iloc[0:0].copy()
    df_test = pd.concat(te).reset_index(drop=True) if te else df.iloc[0:0].copy()
    return df_train, df_val, df_test

def split_leave_last_test_only(
    df: pd.DataFrame,
    therapist_col: str,
    date_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Outer split for prospective evaluation:
      - TEST: last patient per therapist (time-ordered)
      - TRAIN: all earlier patients
      - VAL: empty (inner CV handles tuning)
    """
    tr, te = [], []
    df_sorted = df.sort_values(date_col)
    for _, g in df_sorted.groupby(therapist_col, sort=False):
        n = len(g)
        if n >= 2:
            te.append(g.iloc[[-1]])
            tr.append(g.iloc[:-1])
        else:
            tr.append(g)

    df_train = pd.concat(tr).reset_index(drop=True) if tr else df.iloc[0:0].copy()
    df_val = df.iloc[0:0].copy()  # intentionally empty
    df_test = pd.concat(te).reset_index(drop=True) if te else df.iloc[0:0].copy()
    return df_train, df_val, df_test


def _seed_for_tid(tid, base_seed: int, repeat_index: int) -> int:
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
      - n == 2: pick 1 for TEST; other goes TRAIN (no VAL for that therapist)
      - n == 1: all TRAIN
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
        else:
            tr.append(g)

    df_train = pd.concat(tr).reset_index(drop=True) if tr else df.iloc[0:0].copy()
    df_val = pd.concat(va).reset_index(drop=True) if va else df.iloc[0:0].copy()
    df_test = pd.concat(te).reset_index(drop=True) if te else df.iloc[0:0].copy()
    return df_train, df_val, df_test


def split_dataset(df: pd.DataFrame, cfg: MatchConfig, *, repeat_index: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strat = (cfg.CV_STRATEGY or "leave_last").lower()

    if strat in {"kfold", "k_fold", "kf"}:
        return split_kfold(
            df,
            n_splits=int(getattr(cfg, "KFOLD_N_SPLITS", 5)),
            shuffle=bool(getattr(cfg, "KFOLD_SHUFFLE", True)),
            random_state=int(getattr(cfg, "CV_RANDOM_STATE", 42)),
            repeat_index=int(repeat_index),
            val_frac=float(getattr(cfg, "KFOLD_VAL_FRAC", 0.10)),
        )

    if strat in {"per_therapist_random", "per-therapist-random", "ptr"}:
        return split_per_therapist_random(
            df, cfg.THERAPIST_ID_COL,
            base_seed=int(getattr(cfg, "CV_RANDOM_STATE", 0)),
            repeat_index=int(repeat_index),
        )

    return split_leave_last(df, cfg.THERAPIST_ID_COL, cfg.DATE_COL)



class RepeatedPerTherapistHoldoutCV(BaseCrossValidator):
    """
    Inner tuning CV:
      - Repeats R times
      - In each repeat, for each therapist with >= min_n_for_val cases:
            randomly assign val_per_therapist cases to validation, rest to training
        Therapists with fewer cases contribute only to training.
      - If validation ends up empty, fall back to a small global validation split.
    """
    def __init__(
        self,
        *,
        n_repeats: int,
        therapist_col: str,
        base_seed: int = 0,
        val_per_therapist: int = 1,
        min_n_for_val: int = 2,
        fallback_global_val_frac: float = 0.10,
    ):
        self.n_repeats = int(n_repeats)
        self.therapist_col = str(therapist_col)
        self.base_seed = int(base_seed)
        self.val_per_therapist = int(val_per_therapist)
        self.min_n_for_val = int(min_n_for_val)
        self.fallback_global_val_frac = float(fallback_global_val_frac)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_repeats

    def split(self, X, y=None, groups=None):
        # Prefer explicit groups; else fall back to X[therapist_col] if available
        if groups is None:
            if isinstance(X, pd.DataFrame) and (self.therapist_col in X.columns):
                groups = X[self.therapist_col].to_numpy()
            else:
                raise ValueError("RepeatedPerTherapistHoldoutCV requires 'groups' or X containing therapist_col.")

        groups = np.asarray(groups)
        n = len(groups)
        all_idx = np.arange(n)

        # Pre-index by therapist for speed and determinism
        by_tid: Dict[object, np.ndarray] = {}
        for i, tid in enumerate(groups):
            by_tid.setdefault(tid, []).append(i)
        by_tid = {tid: np.asarray(ixs, dtype=int) for tid, ixs in by_tid.items()}

        for r in range(self.n_repeats):
            rs = np.random.RandomState(self.base_seed + r)

            train_idx_parts = []
            val_idx_parts = []

            for tid, idxs in by_tid.items():
                m = len(idxs)
                if m >= self.min_n_for_val:
                    k = min(self.val_per_therapist, m - 1)  # keep at least 1 in train
                    chosen = rs.choice(idxs, size=k, replace=False)
                    val_idx_parts.append(chosen)
                    keep = np.setdiff1d(idxs, chosen, assume_unique=False)
                    if len(keep):
                        train_idx_parts.append(keep)
                else:
                    train_idx_parts.append(idxs)

            train_idx = np.concatenate(train_idx_parts) if train_idx_parts else np.array([], dtype=int)
            val_idx = np.concatenate(val_idx_parts) if val_idx_parts else np.array([], dtype=int)

            # Safety fallback: if val is empty, do a small global split
            if len(val_idx) == 0:
                k = max(1, int(round(self.fallback_global_val_frac * n)))
                k = min(k, n - 1) if n > 1 else 0
                if k > 0:
                    val_idx = rs.choice(all_idx, size=k, replace=False)
                    train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)

            # Final sanity
            if len(val_idx) == 0 or len(train_idx) == 0:
                # If still degenerate, skip by making a minimal valid split (best-effort)
                if n >= 2:
                    val_idx = np.array([0], dtype=int)
                    train_idx = np.arange(1, n, dtype=int)
                else:
                    val_idx = np.array([], dtype=int)
                    train_idx = np.array([], dtype=int)

            try:
                logger.info(
                    "[INNER CV] repeat=%02d | train_n=%d | val_n=%d | train_therapists=%d | val_therapists=%d",
                    r,
                    len(train_idx), len(val_idx),
                    len(np.unique(groups[train_idx])) if len(train_idx) else 0,
                    len(np.unique(groups[val_idx])) if len(val_idx) else 0,
                )
            except Exception:
                logger.info("[INNER CV] repeat=%02d | train_n=%d | val_n=%d", r, len(train_idx), len(val_idx))

            yield train_idx, val_idx

#####################################################################
# Preprocessors
#####################################################################
def make_preprocessor_common(
    cfg: MatchConfig,
    p_num: List[str], p_cat: List[str],
    t_num: List[str], t_cat: List[str],
    *,
    force_dense_ohe: bool = False
) -> ColumnTransformer:
    if cfg.impute_strategy == "iterative":
        num_imputer = IterativeImputer(**cfg.iterative_imputer_params)
    elif cfg.impute_strategy == "median":
        num_imputer = SimpleImputer(strategy="median")
    elif cfg.impute_strategy == "mean":
        num_imputer =SimpleImputer(strategy="mean", add_indicator=True)
    else:
        raise ValueError(f"Unknown impute_strategy: {cfg.impute_strategy!r}")

    to_numeric = _FT(_to_numeric_df)

    transformers = []
    if p_num:
        transformers.append(("p_num", Pipeline([
            ("to_num", to_numeric),
            ("impute", num_imputer),
            ("scale", StandardScaler(with_mean=False)),
        ]), p_num))
    if p_cat:
        transformers.append(("p_cat", Pipeline([
            ("onehot", _ohe(sparse_output=not force_dense_ohe, min_frequency=cfg.CAT_MIN_FREQ_FOR_OHE))
        ]), p_cat))
    if t_num:
        transformers.append(("t_num", Pipeline([
            ("to_num", to_numeric),
            ("impute", num_imputer),
            ("scale", StandardScaler(with_mean=False)),
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
        if force_dense_ohe:
            ct.set_output(transform="pandas")
        else:
            ct.set_output(transform="default")
    except Exception:
        pass
    return ct


def make_preprocessor_cat_native(
    cfg: MatchConfig,
    p_num: List[str], p_cat: List[str],
    t_num: List[str], t_cat: List[str]
) -> ColumnTransformer:
    """
    Preprocessor for CatBoost native categoricals (no OHE).
    If cfg.CAT_NATIVE_HANDLE_MISSING=True:
      - NO numeric imputation
      - NO scaling (sklearn scalers can't handle NaN)
      - categoricals converted to string while preserving NaN
    """
    num_cols = (p_num or []) + (t_num or [])
    cat_cols = (p_cat or []) + (t_cat or [])

    transformers = []

    if getattr(cfg, "CAT_NATIVE_HANDLE_MISSING", True):
        # --- native missing: pass NaNs through to CatBoost ---
        to_float = _FT(_to_float_preserve_nan_df)
        to_str_nan = _FT(_to_str_preserve_nan_df)

        if num_cols:
            transformers.append(("num", Pipeline([
                ("to_float", to_float),
                # no imputer, no scaler
            ]), num_cols))

        if cat_cols:
            transformers.append(("cat", Pipeline([
                ("to_str_nan", to_str_nan),
                # no imputer
            ]), cat_cols))

    else:
        # --- legacy behavior (your current approach) ---
        if cfg.impute_strategy == "iterative":
            num_imputer = IterativeImputer(**cfg.iterative_imputer_params)
        elif cfg.impute_strategy == "median":
            num_imputer = SimpleImputer(strategy="median")
        elif cfg.impute_strategy == "mean":
            num_imputer = SimpleImputer(strategy="mean", add_indicator=True)
        else:
            raise ValueError(f"Unknown impute_strategy: {cfg.impute_strategy!r}")

        to_numeric = _FT(_to_numeric_df)
        to_str = _FT(_to_str_df)
        cat_imputer = SimpleImputer(strategy="most_frequent")

        if num_cols:
            transformers.append(("num", Pipeline([
                ("to_num", to_numeric),
                ("impute", num_imputer),
                ("scale", StandardScaler(with_mean=False)),
            ]), num_cols))

        if cat_cols:
            transformers.append(("cat", Pipeline([
                ("impute", cat_imputer),
                ("to_str", to_str),
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


def _feature_cols_from_pipe(estimator: Pipeline, cfg: MatchConfig) -> List[str]:
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
    if cfg.PATIENT_ID_COL not in cols:
        ids.append(cfg.PATIENT_ID_COL)
    if cfg.THERAPIST_ID_COL not in cols:
        ids.append(cfg.THERAPIST_ID_COL)
    return cols + ids


#####################################################################
# Scorer (HPO): RMSE only
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


def _parent_mapping_from_pre(
    pre: ColumnTransformer,
    out_feature_names: Optional[List[str]] = None,
    orig_candidates: Optional[List[str]] = None
) -> Dict[str, str]:
    names = list(out_feature_names) if out_feature_names is not None else _ct_get_feature_names(pre)
    if not names:
        return {}

    mapping: Dict[str, str] = {}
    out_idx = getattr(pre, "output_indices_", None)

    if out_idx is not None:
        for name, trans, cols in pre.transformers_:
            if name not in out_idx:
                continue
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
            X_tx = X_tx.copy()
            X_tx.columns = names
    else:
        X_tx = pd.DataFrame(np.asarray(X_tx), columns=names if names else None)

    if not names:
        names = list(X_tx.columns)
    return X_tx, names


def _get_cat_features_idx_for_pool(fitted_pipeline: Pipeline, X_tx: pd.DataFrame) -> List[int]:
    idxs = []
    for i, dt in enumerate(X_tx.dtypes):
        if pdt.is_object_dtype(dt) or pdt.is_string_dtype(dt) or pdt.is_categorical_dtype(dt):
            idxs.append(i)
    return idxs


#####################################################################
# CatBoost explanations
#####################################################################
def _unwrap_cat_model_from_pipeline(pipeline: Pipeline):
    reg = pipeline.named_steps.get("regressor")
    inner = reg
    if isinstance(reg, TransformedTargetRegressor):
        inner = getattr(reg, "regressor_", None) or getattr(reg, "regressor", None)
    if isinstance(inner, CatBoostNativeWrapper):
        return inner.cb_
    return inner  # CatBoostRegressor


def cat_global_importance_df(fitted_pipeline: Pipeline, X_df: pd.DataFrame, y=None, kind="PredictionValuesChange"):
    model = _unwrap_cat_model_from_pipeline(fitted_pipeline)
    X_tx, _ = _transformed_df_and_names(fitted_pipeline, X_df)

    pre = fitted_pipeline.named_steps.get("preprocessor")
    candidates = _input_feature_candidates_from_pre(pre)
    parent = _parent_mapping_from_pre(pre, out_feature_names=list(X_tx.columns), orig_candidates=candidates)

    cat_idx = _get_cat_features_idx_for_pool(fitted_pipeline, X_tx)
    if len(cat_idx):
        X_tx.iloc[:, cat_idx] = _to_str_preserve_nan_df(X_tx.iloc[:, cat_idx])

    pool = Pool(X_tx, label=y, cat_features=cat_idx if len(cat_idx) else None)
    ftype = getattr(EFstrType, kind) if isinstance(kind, str) else kind
    vals = model.get_feature_importance(pool, type=ftype)

    df = pd.DataFrame({"feature": list(X_tx.columns), "importance": np.asarray(vals, float)})
    df["orig_feature"] = df["feature"].map(parent).fillna(df["feature"])
    df_sorted = df.sort_values("importance", ascending=False).reset_index(drop=True)
    agg = (df_sorted.groupby("orig_feature", as_index=False)["importance"].sum()
           .sort_values("importance", ascending=False).reset_index(drop=True))
    return df_sorted, agg


def cat_shap_global_df(fitted_pipeline: Pipeline, X_df: pd.DataFrame, y=None):
    model = _unwrap_cat_model_from_pipeline(fitted_pipeline)
    X_tx, _ = _transformed_df_and_names(fitted_pipeline, X_df)

    pre = fitted_pipeline.named_steps.get("preprocessor")
    candidates = _input_feature_candidates_from_pre(pre)
    parent = _parent_mapping_from_pre(pre, out_feature_names=list(X_tx.columns), orig_candidates=candidates)

    cat_idx = _get_cat_features_idx_for_pool(fitted_pipeline, X_tx)
    if len(cat_idx):
        X_tx.iloc[:, cat_idx] = _to_str_preserve_nan_df(X_tx.iloc[:, cat_idx])

    pool = Pool(X_tx, label=y, cat_features=cat_idx if len(cat_idx) else None)
    shap_mat = model.get_feature_importance(pool, type=EFstrType.ShapValues)
    shap_vals = np.asarray(shap_mat)[:, :-1]  # drop expected value col

    per_feat = pd.DataFrame({
        "feature": list(X_tx.columns),
        "mean_abs_shap": np.abs(shap_vals).mean(axis=0)
    })
    per_feat["orig_feature"] = per_feat["feature"].map(parent).fillna(per_feat["feature"])
    per_feat_sorted = per_feat.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    agg = (per_feat_sorted.groupby("orig_feature", as_index=False)["mean_abs_shap"].sum()
           .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    return per_feat_sorted, agg


def cat_interactions_df(fitted_pipeline: Pipeline, X_df: pd.DataFrame, y=None, top_k=50):
    model = _unwrap_cat_model_from_pipeline(fitted_pipeline)
    X_tx, _ = _transformed_df_and_names(fitted_pipeline, X_df)

    pre = fitted_pipeline.named_steps.get("preprocessor")
    candidates = _input_feature_candidates_from_pre(pre)
    parent = _parent_mapping_from_pre(pre, out_feature_names=list(X_tx.columns), orig_candidates=candidates)
    cols = list(X_tx.columns)

    cat_idx = _get_cat_features_idx_for_pool(fitted_pipeline, X_tx)
    if len(cat_idx):
        X_tx.iloc[:, cat_idx] = _to_str_preserve_nan_df(X_tx.iloc[:, cat_idx])

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

    agg = (df_pairs.assign(pair=list(zip(df_pairs["orig_i"], df_pairs["orig_j"])))
           .groupby("pair", as_index=False)["score"].sum()
           .sort_values("score", ascending=False)
           .head(top_k))
    if len(agg):
        agg[["orig_i", "orig_j"]] = pd.DataFrame(agg["pair"].tolist(), index=agg.index)
        agg = agg.drop(columns=["pair"]).reset_index(drop=True)
    else:
        agg = pd.DataFrame(columns=["orig_i", "orig_j", "score"])
    return ohe_top, agg


def shap_interactions_df(fitted_pipeline: Pipeline, X_df: pd.DataFrame, y=None, top_k=50):
    try:
        import shap  # type: ignore
    except Exception:
        logger.warning("shap not available; skipping SHAP interaction computation.")
        return (
            pd.DataFrame(columns=["feat_i", "feat_j", "score", "orig_i", "orig_j"]),
            pd.DataFrame(columns=["orig_i", "orig_j", "score"]),
        )

    model = _unwrap_cat_model_from_pipeline(fitted_pipeline)
    X_tx, _ = _transformed_df_and_names(fitted_pipeline, X_df)

    pre = fitted_pipeline.named_steps.get("preprocessor")
    candidates = _input_feature_candidates_from_pre(pre)
    parent = _parent_mapping_from_pre(pre, out_feature_names=list(X_tx.columns), orig_candidates=candidates)

    explainer = shap.TreeExplainer(model)
    inter = np.asarray(explainer.shap_interaction_values(X_tx))  # [n, m, m]
    mean_abs = np.mean(np.abs(inter), axis=0)

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
    agg = (df_pairs.assign(pair=list(zip(df_pairs["orig_i"], df_pairs["orig_j"])))
           .groupby("pair", as_index=False)["score"].sum()
           .sort_values("score", ascending=False)
           .head(top_k))
    if len(agg):
        agg[["orig_i", "orig_j"]] = pd.DataFrame(agg["pair"].tolist(), index=agg.index)
        agg = agg.drop(columns=["pair"]).reset_index(drop=True)
    else:
        agg = pd.DataFrame(columns=["orig_i", "orig_j", "score"])
    return ohe_top, agg


#####################################################################
# EBM (optional; interpret)
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
        self._model = None

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


def ebm_global_importance_df(ebm_model) -> pd.DataFrame:
    try:
        gl = ebm_model.explain_global()
        d = gl.data()
        names = scores = None
        if isinstance(d, dict):
            if "overall" in d:
                names = d["overall"].get("names")
                scores = d["overall"].get("scores")
            if (not names) and ("names" in d) and ("scores" in d):
                names, scores = d["names"], d["scores"]
        if names and scores and len(names) == len(scores):
            return (pd.DataFrame({"term": names, "importance": scores})
                    .sort_values("importance", ascending=False).reset_index(drop=True))
    except Exception:
        pass

    term_names = getattr(ebm_model, "term_names_", None)
    term_imps = getattr(ebm_model, "term_importances_", None)
    if term_names is not None and term_imps is not None and len(term_names) > 0:
        return (pd.DataFrame({"term": list(term_names), "importance": list(term_imps)})
                .sort_values("importance", ascending=False).reset_index(drop=True))
    return pd.DataFrame(columns=["term", "importance"])


def ebm_aggregate_base_feature_importance(term_table: pd.DataFrame) -> pd.DataFrame:
    if term_table is None or term_table.empty:
        return pd.DataFrame(columns=["base_feature", "importance"])

    acc: Dict[str, float] = {}
    for _, row in term_table.iterrows():
        imp = float(row.get("importance", np.nan))
        if not np.isfinite(imp):
            continue
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
        return pd.DataFrame(columns=["base_feature", "importance"])
    return (pd.DataFrame({"base_feature": list(acc.keys()), "importance": list(acc.values())})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def _pairs_from_interactions_table(top_interactions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in top_interactions.iterrows():
        imp = float(r.get("importance", np.nan))
        bases = r.get("features_base", []) or []
        uniq = list(dict.fromkeys(bases))
        if len(uniq) >= 2 and np.isfinite(imp):
            rows.append({"orig_i": uniq[0], "orig_j": uniq[1], "score": imp})
    if not rows:
        return pd.DataFrame(columns=["orig_i", "orig_j", "score"])
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


#####################################################################
# CatBoost native-wrapper (for native categorical handling)
#####################################################################
class CatBoostNativeWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **cb_params):
        params = {
            "loss_function": "RMSE",
            "random_seed": 42,
            "verbose": False,
            "task_type": "CPU",
        }
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
                s = X[c]

                # If it's already object/string/categorical, keep it (but still ensure NaN stays NaN)
                mask = s.isna()
                s2 = s.astype("object")
                s2.loc[~mask] = s2.loc[~mask].map(lambda v: str(v))
                s2.loc[mask] = np.nan
                X[c] = s2

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
# Post-hoc: outcome ~ rank regression (TEST set)
#####################################################################
def rank_on_outcome_ols(y: np.ndarray, ranks: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, float)
    r = np.asarray(ranks, float)

    mask = np.isfinite(y) & np.isfinite(r)
    y = y[mask]
    r = r[mask]

    if len(y) < 3:
        return {"n": float(len(y)), "beta": np.nan, "se": np.nan, "p": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "r2": np.nan}

    X = sm.add_constant(r)
    res = sm.OLS(y, X).fit(cov_type="HC3")

    params = np.asarray(res.params)
    bse = np.asarray(res.bse)
    pvals = np.asarray(res.pvalues)

    beta = float(params[1]) if params.size > 1 else np.nan
    se   = float(bse[1])    if bse.size > 1 else np.nan
    p    = float(pvals[1])  if pvals.size > 1 else np.nan

    ci = res.conf_int()
    ci_arr = ci.to_numpy() if isinstance(ci, (pd.DataFrame, pd.Series)) else np.asarray(ci)
    ci_low  = float(ci_arr[1, 0]) if (ci_arr.ndim == 2 and ci_arr.shape[0] > 1 and ci_arr.shape[1] >= 2) else np.nan
    ci_high = float(ci_arr[1, 1]) if (ci_arr.ndim == 2 and ci_arr.shape[0] > 1 and ci_arr.shape[1] >= 2) else np.nan

    r2 = float(getattr(res, "rsquared", np.nan))
    return {"n": float(len(y)), "beta": beta, "se": se, "p": p,
            "ci_low": ci_low, "ci_high": ci_high, "r2": r2}


#####################################################################
# Main Pipeline
#####################################################################
class MatchPipeline:
    def __init__(self, cfg: MatchConfig, *, run_dir=None):
        self.cfg = cfg
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.df: Optional[pd.DataFrame] = None
        self.test_details_: Dict = {}
        self._repeat_index: int = 0

    def set_repeat_index(self, i: int = 0):
        self._repeat_index = int(i)

    def set_data(self, df: pd.DataFrame) -> None:
        df = df.copy()

        # Ensure LABEL_COL exists (fallback optional)
        if self.cfg.LABEL_COL not in df.columns:
            phq = getattr(self.cfg, "PHQ8_T1_COL", None)
            gad = getattr(self.cfg, "GAD7_T1_COL", None)
            if phq and gad and (phq in df.columns) and (gad in df.columns):
                mask_complete = df[[phq, gad]].notna().all(axis=1)
                df = df.loc[mask_complete].reset_index(drop=True)
                df[self.cfg.LABEL_COL] = df[phq].astype(float) + df[gad].astype(float)
            else:
                raise ValueError(
                    "LABEL_COL missing and PHQ8_T1_COL/GAD7_T1_COL not provided (or not in df)."
                )

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
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        therapists_table: pd.DataFrame
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

        # sanity: columns must exist in both (train/val) AND the dyad template
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

    def _make_full_pipeline(
        self,
        p_num, p_cat, t_num, t_cat,
        model: BaseEstimator,
        *,
        force_dense: bool = False
    ) -> Pipeline:
        pre = make_preprocessor_common(self.cfg, p_num, p_cat, t_num, t_cat, force_dense_ohe=force_dense)
        reg: BaseEstimator = model
        if getattr(self.cfg, "STANDARDIZE_Y", False):
            reg = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
        mem = Memory(location=self.cfg.CACHE_DIR, verbose=0) if getattr(self.cfg, "CACHE_STEPS", False) else None
        return Pipeline([("preprocessor", pre), ("regressor", reg)], memory=mem)

    def _evaluate_on_test(
        self,
        name: str,
        estimator: Pipeline,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        df_test: pd.DataFrame,
        ther_table: pd.DataFrame,
        val_score: float,
        *,
        compute_ranking: bool = True
    ) -> Dict:
        # val_rmse from val_score (score = -RMSE)
        if np.isfinite(val_score) and (val_score > NEG_LARGE / 2):
            val_rmse = -float(val_score)
        else:
            val_rmse = np.inf

        # If no test data, return minimal row
        if y_test is None or len(y_test) == 0 or len(df_test) == 0:
            row = {
                "model": name,
                "val_rmse": float(val_rmse),
                "test_rmse": np.nan,
                "test_rmse_lo": np.nan,
                "test_rmse_hi": np.nan,
                "factual_top1_count": 0,
                "factual_top1_share": np.nan,
                "factual_top5_count": 0,
                "factual_top5_share": np.nan,
                "rank_to_outcome_beta": np.nan,
                "rank_to_outcome_se": np.nan,
                "rank_to_outcome_p": np.nan,
                "rank_to_outcome_ci_low": np.nan,
                "rank_to_outcome_ci_high": np.nan,
                "rank_to_outcome_r2": np.nan,
                "rank_to_outcome_n": 0.0,
            }
            details = {
                "ranks": np.array([], int),
                "y_test": np.array([], float),
                "y_pred_obs": np.array([], float),
                "therapist_recommendation_counts": pd.DataFrame(),
                "rank_outcome_table": pd.DataFrame(),
                "posthoc_rank_on_outcome": {},
            }
            return {"row": row, "details": details}

        # observed-dyad test RMSE
        y_pred_obs = estimator.predict(X_test).astype(float)
        test_rmse = rmse(y_test, y_pred_obs)

        # RMSE CI (bootstrap only)
        test_rmse_lo = test_rmse_hi = np.nan
        if getattr(self.cfg, "RMSE_CI_ENABLE", True) and len(y_test) > 1:
            test_rmse_lo, test_rmse_hi = _rmse_ci_bootstrap(
                y_test, y_pred_obs,
                level=float(getattr(self.cfg, "RMSE_CI_LEVEL", 0.95)),
                B=int(getattr(self.cfg, "RMSE_CI_BOOTSTRAP_B", 2000)),
                rng=int(getattr(self.cfg, "RMSE_CI_RANDOM_STATE", 0)),
            )

        # defaults when ranking is skipped
        ranks_arr = np.array([], dtype=int)
        factual_top1_count = 0
        factual_top5_count = 0
        factual_top1_share = np.nan
        factual_top5_share = np.nan
        posthoc = {}
        ther_reco_counts = pd.DataFrame()
        per_patient_df = pd.DataFrame()

        if compute_ranking:
            # ranking factual therapist among all candidate therapists (per patient)
            ranks, top1_tids, top5_tids_all = [], [], []
            per_pat_rows = []

            tid_col = self.cfg.THERAPIST_ID_COL
            pid_col = self.cfg.PATIENT_ID_COL
            raw_cols = _feature_cols_from_pipe(estimator, self.cfg)

            for _, row0 in df_test.iterrows():
                dyads = build_dyads_for_patient(self.cfg, row0, ther_table)
                dyads = dyads.reindex(
                    columns=raw_cols + [c for c in dyads.columns if c not in raw_cols],
                    fill_value=np.nan
                )
                preds = estimator.predict(dyads).astype(float)
                scores_eff = _s_eff(preds, self.cfg)

                tids = ther_table[tid_col].astype(str).to_numpy()
                order = np.argsort(-scores_eff)  # higher score = better
                tids_ordered = tids[order]

                top1_tids.append(str(tids_ordered[0]))
                top5_tids_all.extend([str(t) for t in tids_ordered[:5]])

                factual = str(row0[tid_col])
                pos = np.where(tids_ordered == factual)[0]
                factual_rank = int(pos[0]) + 1 if len(pos) else len(tids_ordered) + 1
                ranks.append(factual_rank)

                idx_f = np.where(tids == factual)[0]
                mu_f = float(preds[int(idx_f[0])]) if len(idx_f) else np.nan

                y_raw = float(row0[self.cfg.LABEL_COL])
                per_pat_rows.append({
                    pid_col: row0.get(pid_col, np.nan),
                    "factual_therapist_id": factual,
                    "factual_rank": factual_rank,
                    self.cfg.LABEL_COL: y_raw,
                    "mu_hat_factual": mu_f,
                })

            per_patient_df = pd.DataFrame(per_pat_rows)
            ranks_arr = np.asarray(ranks, int)

            # top-k factual hit rates
            n_pat = len(df_test)
            factual_top1_count = int(np.sum(ranks_arr == 1))
            factual_top5_count = int(np.sum(ranks_arr <= 5))
            factual_top1_share = (factual_top1_count / n_pat) if n_pat else np.nan
            factual_top5_share = (factual_top5_count / n_pat) if n_pat else np.nan

            # post-hoc: outcome ~ rank
            posthoc = rank_on_outcome_ols(y_test, ranks_arr)

            # therapist recommendation tallies
            if self.cfg.REPORT_TOP_COUNTS:
                c_top1 = Counter(top1_tids)
                c_top5 = Counter(top5_tids_all)
                ther_ids = ther_table[tid_col].astype(str).tolist()
                ther_reco_counts = pd.DataFrame({tid_col: ther_ids})
                ther_reco_counts["top1_count"] = ther_reco_counts[tid_col].map(c_top1).fillna(0).astype(int)
                ther_reco_counts["top5_count"] = ther_reco_counts[tid_col].map(c_top5).fillna(0).astype(int)
                ther_reco_counts["n_test_patients"] = n_pat
                ther_reco_counts["top1_share"] = ther_reco_counts["top1_count"] / max(n_pat, 1)
                ther_reco_counts["top5_share"] = ther_reco_counts["top5_count"] / max(n_pat, 1)
                ther_reco_counts = ther_reco_counts.sort_values(
                    ["top1_count", "top5_count", tid_col],
                    ascending=[False, False, True]
                ).reset_index(drop=True)

        row = {
            "model": name,
            "val_rmse": float(val_rmse),
            "test_rmse": float(test_rmse),
            "test_rmse_lo": float(test_rmse_lo),
            "test_rmse_hi": float(test_rmse_hi),
            "factual_top1_count": factual_top1_count,
            "factual_top1_share": factual_top1_share,
            "factual_top5_count": factual_top5_count,
            "factual_top5_share": factual_top5_share,
            # post-hoc association: distress ~ rank
            "rank_to_outcome_beta": float(posthoc.get("beta", np.nan)),
            "rank_to_outcome_se": float(posthoc.get("se", np.nan)),
            "rank_to_outcome_p": float(posthoc.get("p", np.nan)),
            "rank_to_outcome_ci_low": float(posthoc.get("ci_low", np.nan)),
            "rank_to_outcome_ci_high": float(posthoc.get("ci_high", np.nan)),
            "rank_to_outcome_r2": float(posthoc.get("r2", np.nan)),
            "rank_to_outcome_n": float(posthoc.get("n", np.nan)),
        }

        details = {
            "ranks": ranks_arr,
            "y_test": y_test,
            "y_pred_obs": y_pred_obs,
            "therapist_recommendation_counts": ther_reco_counts,
            "rank_outcome_table": per_patient_df,
            "posthoc_rank_on_outcome": posthoc,
        }
        return {"row": row, "details": details}

    # -----------------------------
    # Benchmarks helpers (NEW)
    # -----------------------------
    def _eval_from_predictions(
        self,
        *,
        name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        val_rmse: float
    ) -> Dict:
        """Build a summary-compatible row from precomputed predictions (ranking intentionally omitted)."""
        y_true = np.asarray(y_true, float)
        y_pred = np.asarray(y_pred, float)

        if len(y_true) == 0:
            test_rmse = np.nan
            lo = hi = np.nan
        else:
            test_rmse = rmse(y_true, y_pred)
            lo = hi = np.nan
            if getattr(self.cfg, "RMSE_CI_ENABLE", True) and len(y_true) > 1:
                lo, hi = _rmse_ci_bootstrap(
                    y_true, y_pred,
                    level=float(getattr(self.cfg, "RMSE_CI_LEVEL", 0.95)),
                    B=int(getattr(self.cfg, "RMSE_CI_BOOTSTRAP_B", 2000)),
                    rng=int(getattr(self.cfg, "RMSE_CI_RANDOM_STATE", 0)),
                )

        row = {
            "model": name,
            "val_rmse": float(val_rmse) if np.isfinite(val_rmse) else np.inf,
            "test_rmse": float(test_rmse) if np.isfinite(test_rmse) else np.nan,
            "test_rmse_lo": float(lo) if np.isfinite(lo) else np.nan,
            "test_rmse_hi": float(hi) if np.isfinite(hi) else np.nan,
            "factual_top1_count": 0,
            "factual_top1_share": np.nan,
            "factual_top5_count": 0,
            "factual_top5_share": np.nan,
            "rank_to_outcome_beta": np.nan,
            "rank_to_outcome_se": np.nan,
            "rank_to_outcome_p": np.nan,
            "rank_to_outcome_ci_low": np.nan,
            "rank_to_outcome_ci_high": np.nan,
            "rank_to_outcome_r2": np.nan,
            "rank_to_outcome_n": 0.0,
        }
        details = {
            "ranks": np.array([], int),
            "y_test": y_true,
            "y_pred_obs": y_pred,
            "therapist_recommendation_counts": pd.DataFrame(),
            "rank_outcome_table": pd.DataFrame(),
            "posthoc_rank_on_outcome": {},
        }
        return {"row": row, "details": details}

    def _naive_mean_baseline(
        self,
        *,
        name: str,
        y_fit: np.ndarray,
        y_test: np.ndarray,
        cv_hpo,
        X_fit: pd.DataFrame,
        groups: Optional[np.ndarray],
        strat: str,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame
    ) -> Dict:
        """
        Naïve mean baseline:
          - val_rmse: computed per CV fold (mean on training portion in each fold)
          - test_rmse: mean computed on full y_fit applied to test
        """
        y_fit = np.asarray(y_fit, float)
        y_test = np.asarray(y_test, float)

        val_rmse = np.inf

        # nested strategy: fold-wise mean baseline using cv_hpo splits
        if strat in {"outer_leave_last_inner_repeated", "outer-leave-last-inner-repeated", "nested"}:
            ss = 0.0
            nn = 0
            try:
                split_iter = cv_hpo.split(X_fit, y_fit, groups=groups) if groups is not None else cv_hpo.split(X_fit, y_fit)
                for tr_idx, va_idx in split_iter:
                    tr_idx = np.asarray(tr_idx, int)
                    va_idx = np.asarray(va_idx, int)
                    if len(tr_idx) == 0 or len(va_idx) == 0:
                        continue
                    mu = float(np.mean(y_fit[tr_idx]))
                    err = y_fit[va_idx] - mu
                    ss += float(np.sum(err ** 2))
                    nn += int(len(va_idx))
                if nn > 0:
                    val_rmse = float(np.sqrt(ss / nn))
            except Exception:
                val_rmse = np.inf

        # non-nested strategy: use explicit train/val split
        else:
            if len(df_train) > 0 and len(df_val) > 0:
                mu = float(np.mean(df_train[self.cfg.LABEL_COL].to_numpy(dtype=float)))
                y_val = df_val[self.cfg.LABEL_COL].to_numpy(dtype=float)
                val_rmse = rmse(y_val, np.full_like(y_val, mu, dtype=float))

        mu_fit = float(np.mean(y_fit)) if len(y_fit) else np.nan
        y_pred_test = np.full_like(y_test, mu_fit, dtype=float) if len(y_test) else np.array([], float)

        return self._eval_from_predictions(
            name=name,
            y_true=y_test,
            y_pred=y_pred_test,
            val_rmse=val_rmse
        )

    def run(self) -> Dict[str, pd.DataFrame]:
        if self.df is None:
            raise RuntimeError("Call set_data(df) first.")

        # --------------------------
        # Which models to run?
        # --------------------------
        allowed = {"fm", "ridge", "cat", "ebm"}
        requested = _dedupe([str(m).strip().lower() for m in (self.cfg.RUN_MODELS or []) if str(m).strip()])

        if not requested:
            requested = ["fm", "ridge", "cat", "ebm"]

        unknown = [m for m in requested if m not in allowed]
        if unknown:
            raise ValueError(f"Unknown model(s) in RUN_MODELS: {unknown}. Allowed: {sorted(allowed)}")

        strat = (self.cfg.CV_STRATEGY or "leave_last").lower()

        # --- Outer split ---
        if strat in {"outer_leave_last_inner_repeated", "outer-leave-last-inner-repeated", "nested"}:
            # Outer evaluation = "next incoming patient per therapist" (time-ordered)
            df_train, df_val, df_test = split_leave_last_test_only(
                self.df, self.cfg.THERAPIST_ID_COL, self.cfg.DATE_COL
            )
            # Inner CV handles tuning => df_val intentionally empty
        else:
            df_train, df_val, df_test = split_dataset(self.df, self.cfg, repeat_index=int(self._repeat_index))

            # guard: if VAL is empty, create a small global validation split so PredefinedSplit works
            if len(df_val) == 0 and len(df_train) >= 4:
                rs = np.random.RandomState(int(getattr(self.cfg, "CV_RANDOM_STATE", 0)) + int(self._repeat_index))
                n_val = max(1, int(0.1 * len(df_train)))
                idx = rs.choice(np.arange(len(df_train)), size=n_val, replace=False)
                df_val = df_train.iloc[idx].copy().reset_index(drop=True)
                df_train = df_train.drop(df_train.index[idx]).reset_index(drop=True)
                logger.warning("Validation was empty; created a global fallback validation set of size %d.", len(df_val))

        logger.info(
            "Therapists: %d | Train: %d | Val: %d | Test: %d | repeat=%d | CV=%s | RUN_MODELS=%s",
            self.df[self.cfg.THERAPIST_ID_COL].nunique(),
            len(df_train), len(df_val), len(df_test),
            int(self._repeat_index), (self.cfg.CV_STRATEGY or "leave_last"),
            requested,
        )

        ther_table = therapists_table_from_train(self.cfg, df_train)
        logger.info("candidates: %d therapists in train", len(ther_table))

        p_num_u, p_cat_u, t_num_u, t_cat_u = self._build_feature_lists_for_use(df_train, df_val, ther_table)

        # --------------------------
        # Zero-variance filter (train-only): patient + therapist
        # --------------------------
        if getattr(self.cfg, "DROP_ZERO_VARIANCE_FEATURES", True):
            p_num_u2, p_cat_u2, t_num_u2, t_cat_u2, dropped = _drop_zero_variance_feature_lists(
                df_train,
                p_num=p_num_u, p_cat=p_cat_u,
                t_num=t_num_u, t_cat=t_cat_u,
                count_nan_as_value=bool(getattr(self.cfg, "ZERO_VAR_COUNT_NAN_AS_VALUE", False)),
            )
            if dropped:
                logger.info(
                    "[Feature selection] Dropped %d zero-variance features (train-only): %s",
                    len(dropped),
                    sorted(list(dropped.keys()))[:50] + (["..."] if len(dropped) > 50 else []),
                )
            p_num_u, p_cat_u, t_num_u, t_cat_u = p_num_u2, p_cat_u2, t_num_u2, t_cat_u2



        if getattr(self.cfg, "CAT_USE_NATIVE_CATEGORICAL", False):
            p_cat_native = [c for c in p_cat_u if not _is_preencoded_dummy(c)]
            t_cat_native = [c for c in t_cat_u if not _is_preencoded_dummy(c)]

            p_dummy_as_num = [c for c in p_cat_u if _is_preencoded_dummy(c)]
            t_dummy_as_num = [c for c in t_cat_u if _is_preencoded_dummy(c)]

            p_num_cat = _dedupe(p_num_u + p_dummy_as_num)
            t_num_cat = _dedupe(t_num_u + t_dummy_as_num)
        else:
            p_cat_native, t_cat_native = p_cat_u, t_cat_u
            p_num_cat, t_num_cat = p_num_u, t_num_u

        if not t_num_u and not t_cat_u:
            logger.warning("No therapist features in model. Consider include_therapist_id_feature=True.")

        base_feats = p_num_u + p_cat_u + t_num_u + t_cat_u
        extra_ids = [self.cfg.PATIENT_ID_COL]
        if self.cfg.THERAPIST_ID_COL not in base_feats:
            extra_ids.append(self.cfg.THERAPIST_ID_COL)
        tr_cols = _dedupe(base_feats + extra_ids)

        # --- Data for HPO fit ---
        # Two modes:
        #   (A) nested: X_fit = df_train only; cv_hpo = repeated per-therapist holdout; pass groups
        #   (B) legacy: X_fit = train+val; cv_hpo = PredefinedSplit; no groups
        if strat in {"outer_leave_last_inner_repeated", "outer-leave-last-inner-repeated", "nested"}:
            X_fit = df_train[tr_cols].copy()
            y_fit = df_train[self.cfg.LABEL_COL].to_numpy()

            cv_hpo = RepeatedPerTherapistHoldoutCV(
                n_repeats=int(getattr(self.cfg, "INNER_N_REPEATS", 10)),
                therapist_col=self.cfg.THERAPIST_ID_COL,
                base_seed=int(getattr(self.cfg, "INNER_RANDOM_STATE", 123)),
                val_per_therapist=int(getattr(self.cfg, "INNER_VAL_PER_THERAPIST", 1)),
                min_n_for_val=int(getattr(self.cfg, "INNER_MIN_N_FOR_VAL", 2)),
                fallback_global_val_frac=float(getattr(self.cfg, "INNER_FALLBACK_GLOBAL_VAL_FRAC", 0.10)),
            )
            fit_kwargs = {"groups": df_train[self.cfg.THERAPIST_ID_COL].to_numpy()}

            logger.info("=" * 80)
            logger.info("[INNER CV] Using RepeatedPerTherapistHoldoutCV for HPO (nested strategy).")
            logger.info("[INNER CV] outer_train_n=%d | therapists=%d",
                        len(df_train), df_train[self.cfg.THERAPIST_ID_COL].nunique())
            logger.info("[INNER CV] repeats=%d | val_per_therapist=%d | min_n_for_val=%d | fallback_global_val_frac=%.3f | base_seed=%d",
                        cv_hpo.n_repeats, cv_hpo.val_per_therapist, cv_hpo.min_n_for_val,
                        cv_hpo.fallback_global_val_frac, cv_hpo.base_seed)
            logger.info("=" * 80)

        else:
            X_train = df_train[tr_cols].copy()
            y_train = df_train[self.cfg.LABEL_COL].to_numpy()
            X_val = df_val[tr_cols].copy()
            y_val = df_val[self.cfg.LABEL_COL].to_numpy()

            X_fit = pd.concat([X_train, X_val], ignore_index=True)
            y_fit = np.concatenate([y_train, y_val])

            test_fold = np.r_[np.full(len(X_train), -1), np.zeros(len(X_val), dtype=int)]
            cv_hpo = PredefinedSplit(test_fold)
            fit_kwargs = {}

        # -------------------------------------------------
        # Build (selected) models + GridSearchCV
        # -------------------------------------------------
        val_scorer = RMSEScorer()
        use_ttr = bool(getattr(self.cfg, "STANDARDIZE_Y", False))

        def _grid_for(cfg_grid: Union[Dict[str, List[Any]], List[Dict[str, List[Any]]]]):
            """Map pipeline param names when using TransformedTargetRegressor (TTR).
            Supports both dict grids and list-of-dict grids (paired schedules)."""
            if not use_ttr:
                return cfg_grid

            def _map_keys(d: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
                return {k.replace("regressor__", "regressor__regressor__"): v for k, v in d.items()}

            if isinstance(cfg_grid, dict):
                return _map_keys(cfg_grid)

            if isinstance(cfg_grid, list):
                return [_map_keys(d) for d in cfg_grid]

            raise TypeError(f"param_grid must be dict or list[dict], got {type(cfg_grid)}")

        gs_by_name: Dict[str, GridSearchCV] = {}

        # FM
        if "fm" in requested:
            fm_pipe = self._make_full_pipeline(p_num_u, p_cat_u, t_num_u, t_cat_u, TorchFMRegressor(), force_dense=False)
            fm_gs = GridSearchCV(
                fm_pipe, _grid_for(self.cfg.fm_param_grid),
                scoring=val_scorer, refit=True, cv=cv_hpo, n_jobs=1, verbose=2,
                error_score=NEG_LARGE
            )
            gs_by_name["fm"] = fm_gs

        # Ridge
        if "ridge" in requested:
            ridge_pipe = self._make_full_pipeline(p_num_u, p_cat_u, t_num_u, t_cat_u, RidgeRegressor(), force_dense=False)
            ridge_gs = GridSearchCV(
                ridge_pipe, _grid_for(self.cfg.ridge_param_grid),
                scoring=val_scorer, refit=True, cv=cv_hpo, n_jobs=1, verbose=2,
                error_score=NEG_LARGE
            )
            gs_by_name["ridge"] = ridge_gs

        # CatBoost
        if "cat" in requested:
            # CatBoost (CPU)
            train_dir = None
            if self.run_dir is not None:
                cb_info = (self.run_dir / "catboost_info")
                cb_info.mkdir(parents=True, exist_ok=True)
                train_dir = str(cb_info.resolve())

            if getattr(self.cfg, "CAT_USE_NATIVE_CATEGORICAL", False):
                pre_cat = make_preprocessor_cat_native(self.cfg, p_num_cat, p_cat_native, t_num_cat, t_cat_native)

                cat_model = CatBoostNativeWrapper(
                    loss_function="RMSE",
                    random_seed=42,
                    verbose=False,
                    task_type="CPU",
                    train_dir=train_dir,
                    allow_writing_files=bool(getattr(self.cfg, "CATBOOST_ALLOW_WRITING_FILES", True)),
                )
                reg_cat: BaseEstimator = cat_model
                if getattr(self.cfg, "STANDARDIZE_Y", False):
                    reg_cat = TransformedTargetRegressor(regressor=cat_model, transformer=StandardScaler())
                mem = Memory(location=self.cfg.CACHE_DIR, verbose=0) if getattr(self.cfg, "CACHE_STEPS", False) else None
                cat_pipe = Pipeline([("preprocessor", pre_cat), ("regressor", reg_cat)], memory=mem)
            else:
                cat_pipe = self._make_full_pipeline(
                    p_num_u, p_cat_u, t_num_u, t_cat_u,
                    CatBoostRegressor(
                        loss_function="RMSE",
                        random_seed=42,
                        verbose=False,
                        task_type="CPU",
                        train_dir=train_dir,
                        allow_writing_files=bool(getattr(self.cfg, "CATBOOST_ALLOW_WRITING_FILES", True)),
                    ),
                    force_dense=True
                )

            cat_gs = GridSearchCV(
                cat_pipe, _grid_for(self.cfg.cat_param_grid),
                scoring=val_scorer, refit=True, cv=cv_hpo, n_jobs=1, verbose=2,
                error_score=NEG_LARGE
            )
            gs_by_name["cat"] = cat_gs

        # EBM (optional)
        if "ebm" in requested:
            if not _interpret_available():
                logger.warning("interpret not found; skipping EBM model.")
            else:
                pre_ebm = make_preprocessor_common(self.cfg, p_num_u, p_cat_u, t_num_u, t_cat_u, force_dense_ohe=False)
                ebm_model = EBMWrapper()
                reg_ebm: BaseEstimator = ebm_model
                if getattr(self.cfg, "STANDARDIZE_Y", False):
                    reg_ebm = TransformedTargetRegressor(regressor=ebm_model, transformer=StandardScaler())
                mem = Memory(location=self.cfg.CACHE_DIR, verbose=0) if getattr(self.cfg, "CACHE_STEPS", False) else None
                ebm_pipe = Pipeline([("preprocessor", pre_ebm), ("regressor", reg_ebm)], memory=mem)

                ebm_gs = GridSearchCV(
                    ebm_pipe, _grid_for(self.cfg.ebm_param_grid),
                    scoring=val_scorer, refit=True, cv=cv_hpo, n_jobs=1, verbose=2,
                    error_score=NEG_LARGE
                )
                gs_by_name["ebm"] = ebm_gs

        if not gs_by_name:
            raise RuntimeError("No models selected to run (after availability checks).")

        # Fit selected models
        for mname, gs in gs_by_name.items():
            logger.info("[Fitting] %s", mname)
            gs.fit(X_fit, y_fit, **fit_kwargs)

        # test data
        X_test = df_test[tr_cols].copy() if len(df_test) else df_test.iloc[0:0][tr_cols].copy()
        y_test = df_test[self.cfg.LABEL_COL].to_numpy() if len(df_test) else np.array([], float)

        # evaluate all fitted models
        candidates = []
        for mname, gs in gs_by_name.items():
            candidates.append((mname, gs.best_estimator_, gs.best_score_, gs))

        cand_map = {name: (est, score, gs) for (name, est, score, gs) in candidates}

        results_rows = []
        details_by_model = {}
        therapist_reco_by_model = {}

        for name, est, val_score, _gs in candidates:
            eval_out = self._evaluate_on_test(name, est, X_test, y_test, df_test, ther_table, val_score, compute_ranking=True)
            results_rows.append(eval_out["row"])
            details_by_model[name] = eval_out["details"]
            therapist_reco_by_model[name] = eval_out["details"].get("therapist_recommendation_counts", pd.DataFrame())

        df_sum = pd.DataFrame(results_rows)

        # pick best model by val_rmse (finite), else by test_rmse  (NOTE: benchmarks are added later and do NOT affect selection)
        if len(df_sum):
            valid = df_sum[np.isfinite(df_sum["val_rmse"]) & (df_sum["val_rmse"] < np.inf)].copy()
            if len(valid):
                best_name = valid.loc[valid["val_rmse"].idxmin(), "model"]
            else:
                best_name = df_sum.loc[df_sum["test_rmse"].idxmin(), "model"]
        else:
            best_name = candidates[0][0]

        best_model, _, best_gs = cand_map[best_name]
        logger.info("[Selected] %s | policy=validation", best_name)

        # -------------------------------------------------
        # Benchmarks (NEW): naive mean + patient-only
        # These are appended to df_sum but do NOT affect best_name selection above.
        # -------------------------------------------------
        benchmark_rows = []
        benchmark_details = {}
        benchmark_reco = {}

        # Naïve mean baseline (fold-wise mean in validation)
        if getattr(self.cfg, "INCLUDE_NAIVE_MEAN_BASELINE", True):
            groups_arr = fit_kwargs.get("groups") if isinstance(fit_kwargs, dict) else None
            base_out = self._naive_mean_baseline(
                name="naive_mean",
                y_fit=y_fit,
                y_test=y_test,
                cv_hpo=cv_hpo,
                X_fit=X_fit,
                groups=groups_arr,
                strat=strat,
                df_train=df_train,
                df_val=df_val
            )
            benchmark_rows.append(base_out["row"])
            benchmark_details["naive_mean"] = base_out["details"]
            benchmark_reco["naive_mean"] = pd.DataFrame()

        # Patient-only benchmark: refit best architecture without therapist features
        if getattr(self.cfg, "INCLUDE_PATIENT_ONLY_BENCHMARK", True):
            if (len(p_num_u) + len(p_cat_u)) == 0:
                logger.warning("Patient-only benchmark skipped: no patient features configured.")
            else:
                best_arch = best_name  # one of {"fm","ridge","cat","ebm"}

                # patient-only matrices (keep patient_id as harmless column)
                po_cols = _dedupe(p_num_u + p_cat_u + [self.cfg.PATIENT_ID_COL])
                X_fit_po = X_fit.reindex(columns=po_cols, fill_value=np.nan)
                X_test_po = X_test.reindex(columns=po_cols, fill_value=np.nan)

                val_scorer_po = RMSEScorer()
                use_ttr_po = bool(getattr(self.cfg, "STANDARDIZE_Y", False))

                _grid_for_po = _grid_for

                gs_po = None

                if best_arch == "fm":
                    po_pipe = self._make_full_pipeline(p_num_u, p_cat_u, [], [], TorchFMRegressor(), force_dense=False)
                    gs_po = GridSearchCV(
                        po_pipe, _grid_for_po(self.cfg.fm_param_grid),
                        scoring=val_scorer_po, refit=True, cv=cv_hpo, n_jobs=1, verbose=2,
                        error_score=NEG_LARGE
                    )

                elif best_arch == "ridge":
                    po_pipe = self._make_full_pipeline(p_num_u, p_cat_u, [], [], RidgeRegressor(), force_dense=False)
                    gs_po = GridSearchCV(
                        po_pipe, _grid_for_po(self.cfg.ridge_param_grid),
                        scoring=val_scorer_po, refit=True, cv=cv_hpo, n_jobs=1, verbose=2,
                        error_score=NEG_LARGE
                    )

                elif best_arch == "cat":
                    train_dir_po = None
                    if self.run_dir is not None:
                        cb_info_po = (self.run_dir / "catboost_info_patient_only")
                        cb_info_po.mkdir(parents=True, exist_ok=True)
                        train_dir_po = str(cb_info_po.resolve())

                    if getattr(self.cfg, "CAT_USE_NATIVE_CATEGORICAL", False):
                        p_cat_native_po = [c for c in p_cat_u if not _is_preencoded_dummy(c)]
                        p_dummy_as_num_po = [c for c in p_cat_u if _is_preencoded_dummy(c)]
                        p_num_cat_po = _dedupe(p_num_u + p_dummy_as_num_po)

                        pre_po = make_preprocessor_cat_native(self.cfg, p_num_cat_po, p_cat_native_po, [], [])
                        cat_model_po = CatBoostNativeWrapper(
                            loss_function="RMSE",
                            random_seed=42,
                            verbose=False,
                            task_type="CPU",
                            train_dir=train_dir_po,
                            allow_writing_files=bool(getattr(self.cfg, "CATBOOST_ALLOW_WRITING_FILES", True)),
                        )
                        reg_po = cat_model_po
                        if getattr(self.cfg, "STANDARDIZE_Y", False):
                            reg_po = TransformedTargetRegressor(regressor=cat_model_po, transformer=StandardScaler())
                        mem = Memory(location=self.cfg.CACHE_DIR, verbose=0) if getattr(self.cfg, "CACHE_STEPS", False) else None
                        po_pipe = Pipeline([("preprocessor", pre_po), ("regressor", reg_po)], memory=mem)
                    else:
                        po_pipe = self._make_full_pipeline(
                            p_num_u, p_cat_u, [], [],
                            CatBoostRegressor(
                                loss_function="RMSE",
                                random_seed=42,
                                verbose=False,
                                task_type="CPU",
                                train_dir=train_dir_po,
                                allow_writing_files=bool(getattr(self.cfg, "CATBOOST_ALLOW_WRITING_FILES", True)),
                            ),
                            force_dense=True
                        )

                    gs_po = GridSearchCV(
                        po_pipe, _grid_for_po(self.cfg.cat_param_grid),
                        scoring=val_scorer_po, refit=True, cv=cv_hpo, n_jobs=1, verbose=2,
                        error_score=NEG_LARGE
                    )

                elif best_arch == "ebm":
                    if not _interpret_available():
                        logger.warning("Patient-only benchmark for EBM skipped: interpret not installed.")
                        gs_po = None
                    else:
                        pre_po = make_preprocessor_common(self.cfg, p_num_u, p_cat_u, [], [], force_dense_ohe=False)
                        ebm_model_po = EBMWrapper()
                        reg_po = ebm_model_po
                        if getattr(self.cfg, "STANDARDIZE_Y", False):
                            reg_po = TransformedTargetRegressor(regressor=ebm_model_po, transformer=StandardScaler())
                        mem = Memory(location=self.cfg.CACHE_DIR, verbose=0) if getattr(self.cfg, "CACHE_STEPS", False) else None
                        po_pipe = Pipeline([("preprocessor", pre_po), ("regressor", reg_po)], memory=mem)

                        gs_po = GridSearchCV(
                            po_pipe, _grid_for_po(self.cfg.ebm_param_grid),
                            scoring=val_scorer_po, refit=True, cv=cv_hpo, n_jobs=1, verbose=2,
                            error_score=NEG_LARGE
                        )

                if gs_po is not None:
                    logger.info("[Patient-only benchmark] refitting best architecture without therapist features: %s", best_arch)
                    gs_po.fit(X_fit_po, y_fit, **fit_kwargs)

                    est_po = gs_po.best_estimator_
                    val_score_po = gs_po.best_score_
                    eval_po = self._evaluate_on_test(
                        name=f"patient_only_{best_arch}",
                        estimator=est_po,
                        X_test=X_test_po,
                        y_test=y_test,
                        df_test=df_test,
                        ther_table=ther_table,
                        val_score=val_score_po,
                        compute_ranking=False
                    )
                    benchmark_rows.append(eval_po["row"])
                    benchmark_details[f"patient_only_{best_arch}"] = eval_po["details"]
                    benchmark_reco[f"patient_only_{best_arch}"] = pd.DataFrame()

        # append benchmarks to summary + details maps (but keep best_name unchanged)
        if benchmark_rows:
            df_bench = pd.DataFrame(benchmark_rows)
            df_sum = pd.concat([df_sum, df_bench], ignore_index=True, sort=False)

            details_by_model.update(benchmark_details)
            therapist_reco_by_model.update(benchmark_reco)

        # sort summary (by val_rmse then test_rmse)
        df_sum = df_sum.sort_values(
            by=["val_rmse", "test_rmse"],
            ascending=[True, True]
        ).reset_index(drop=True)

        # ---------- Explanations (Cat if present; EBM if present) ----------
        self.test_details_.update({
            "cat_model_importance_agg": pd.DataFrame(columns=["orig_feature", "importance"]),
            "cat_model_interactions_agg": pd.DataFrame(columns=["orig_i", "orig_j", "score"]),
            "cat_model_shap_agg": pd.DataFrame(columns=["orig_feature", "mean_abs_shap"]),
            "cat_model_shap_interactions_agg": pd.DataFrame(columns=["orig_i", "orig_j", "score"]),
            "ebm_model_importance_agg": pd.DataFrame(columns=["base_feature", "importance"]),
            "ebm_model_interactions_agg": pd.DataFrame(columns=["orig_i", "orig_j", "score"]),
        })

        try:
            X_imp_full = X_fit[_dedupe(p_num_u + p_cat_u + t_num_u + t_cat_u)].copy()
            y_imp_full = y_fit
            if isinstance(self.cfg.EXPLAIN_SAMPLE_ROWS, int) and 0 < self.cfg.EXPLAIN_SAMPLE_ROWS < len(X_imp_full):
                X_imp = X_imp_full.sample(self.cfg.EXPLAIN_SAMPLE_ROWS, random_state=0)
                y_imp = y_imp_full[X_imp.index]
                logger.info("Explanations on sampled rows: %d / %d", len(X_imp), len(X_imp_full))
            else:
                X_imp, y_imp = X_imp_full, y_imp_full

            # CatBoost explanations (only if cat was run)
            cat_est = cand_map.get("cat", (None,))[0]
            if cat_est is not None:
                if self.cfg.EXPLAIN_COMPUTE_FI:
                    _fi_ohe, fi_agg = cat_global_importance_df(cat_est, X_imp, y_imp, kind="PredictionValuesChange")
                    self.test_details_["cat_model_importance_agg"] = fi_agg.head(self.cfg.EXPLAIN_TOPK)
                if self.cfg.EXPLAIN_COMPUTE_SHAP_GLOBAL:
                    _shap_ohe, shap_agg = cat_shap_global_df(cat_est, X_imp, y_imp)
                    self.test_details_["cat_model_shap_agg"] = shap_agg.head(self.cfg.EXPLAIN_TOPK)
                if self.cfg.EXPLAIN_COMPUTE_INTERACTIONS:
                    _ints_ohe, ints_agg = cat_interactions_df(cat_est, X_imp, y_imp, top_k=self.cfg.EXPLAIN_TOPK)
                    self.test_details_["cat_model_interactions_agg"] = ints_agg
                if self.cfg.EXPLAIN_COMPUTE_SHAP_INTERACTIONS:
                    _sints_ohe, sints_agg = shap_interactions_df(cat_est, X_imp, y_imp, top_k=self.cfg.EXPLAIN_TOPK)
                    self.test_details_["cat_model_shap_interactions_agg"] = sints_agg

            # EBM explanations (only if ebm was run)
            ebm_est = cand_map.get("ebm", (None,))[0]
            if ebm_est is not None:
                ebm_model = _unwrap_ebm_from_pipeline(ebm_est)
                if ebm_model is not None:
                    imp_df = ebm_global_importance_df(ebm_model)
                    # crude but stable aggregation: split "a x b" into base features
                    rows = []
                    term_names = imp_df.get("term", [])
                    term_imps = imp_df.get("importance", [])
                    for t, imp in zip(term_names, term_imps):
                        t = str(t)
                        parts = [p.strip() for p in t.split(" x ")]
                        if len(parts) == 1:
                            rows.append({"type": "univariate", "features_base": [parts[0]], "importance": float(imp)})
                        else:
                            rows.append({"type": "interaction", "features_base": parts, "importance": float(imp)})
                    term_table = pd.DataFrame(rows)
                    feat_agg = ebm_aggregate_base_feature_importance(term_table)
                    self.test_details_["ebm_model_importance_agg"] = feat_agg.head(self.cfg.EXPLAIN_TOPK)
                    # interactions (top terms that look like interactions)
                    inter_terms = term_table[term_table["type"] == "interaction"].copy()
                    inter_terms = inter_terms.sort_values("importance", ascending=False).head(self.cfg.EXPLAIN_TOPK)
                    top_pairs = _pairs_from_interactions_table(inter_terms.rename(columns={"importance": "importance"}))
                    self.test_details_["ebm_model_interactions_agg"] = top_pairs
        except Exception as e:
            logger.warning("Explanations failed: %s: %s", type(e).__name__, e)

        # Example scores for first test patient (best model)
        example_scores = pd.DataFrame()
        if len(df_test) > 0:
            try:
                example_scores = build_dyads_for_patient(self.cfg, df_test.iloc[0], ther_table)
                raw_cols = _feature_cols_from_pipe(best_model, self.cfg)
                example_scores = example_scores[raw_cols]
                example_scores["predicted_outcome"] = best_model.predict(example_scores).astype(float)
                example_scores = example_scores[[self.cfg.THERAPIST_ID_COL, "predicted_outcome"]].sort_values(
                    "predicted_outcome",
                    ascending=(not self.cfg.OUTCOME_HIGHER_IS_BETTER)  # for distress: ascending = better
                ).reset_index(drop=True)
            except Exception as e:
                logger.warning("[Example scores] %s: %s", type(e).__name__, e)

        # stash details
        self.test_details_ = {
            **self.test_details_,
            "models": details_by_model,
            "therapist_recommendation_counts_by_model": therapist_reco_by_model,
            "used_features": {
                "patient_numeric": p_num_u,
                "patient_categorical": p_cat_u,
                "therapist_numeric": t_num_u,
                "therapist_categorical": t_cat_u,
            },
            "best_model_name": best_name,
            "best_params": dict(best_gs.best_params_),
            "example_scores_first_test_patient": example_scores,
            "rank_outcome_table_by_model": {k: v.get("rank_outcome_table") for k, v in details_by_model.items()},
            "posthoc_rank_on_outcome_by_model": {k: v.get("posthoc_rank_on_outcome") for k, v in details_by_model.items()},
            "run_models": requested,
        }

        ther_table_out = therapists_table_from_train(self.cfg, df_train)
        best_reco = therapist_reco_by_model.get(best_name, pd.DataFrame())

        return {
            "summary": df_sum,  # NOTE: no "val_failed" column anymore
            "therapists": ther_table_out,
            "therapist_recommendations": best_reco,
        }


#####################################################################
# Runner
#####################################################################
def run_pipeline(df: pd.DataFrame, cfg: MatchConfig) -> Dict[str, object]:
    """
    Implements:
      - deterministic run folder: ./model_outputs/run_<YYYYMMDD_HH>[_vK]
      - per-repeat subfolders: .../repeat_00, repeat_01, ...
      - CatBoost writes into .../catboost_info
      - caching (optional) lives inside the same folder
      - saves outputs (tables/explanations/meta) automatically when OUTPUT_SAVE=True
    """
    n_rep = int(getattr(cfg, "CV_N_REPEATS", 1) or 1)
    strat = (cfg.CV_STRATEGY or "leave_last").lower()

    if strat in {"kfold", "k_fold", "kf"}:
        n_rep = int(getattr(cfg, "KFOLD_N_SPLITS", 5) or 5)
    else:
        n_rep = int(getattr(cfg, "CV_N_REPEATS", 1) or 1)


    def _df_or_empty(obj, cols):
        return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(columns=cols)

    def _attach_details(out: Dict[str, object], pipe: MatchPipeline) -> Dict[str, object]:
        """Add convenience fields + explanation tables to the out dict (single run)."""
        out = dict(out or {})
        out["used_features"] = pipe.test_details_.get("used_features", {})
        out["best_model_name"] = pipe.test_details_.get("best_model_name")
        out["best_params"] = pipe.test_details_.get("best_params")  # single-run OK
        out["example_scores_first_test_patient"] = pipe.test_details_.get(
            "example_scores_first_test_patient", pd.DataFrame()
        )
        out["run_models"] = pipe.test_details_.get("run_models", getattr(cfg, "RUN_MODELS", None))

        out["cat_model_importance_agg"] = _df_or_empty(
            pipe.test_details_.get("cat_model_importance_agg"), ["orig_feature", "importance"]
        )
        out["cat_model_interactions_agg"] = _df_or_empty(
            pipe.test_details_.get("cat_model_interactions_agg"), ["orig_i", "orig_j", "score"]
        )
        out["cat_model_shap_agg"] = _df_or_empty(
            pipe.test_details_.get("cat_model_shap_agg"), ["orig_feature", "mean_abs_shap"]
        )
        out["cat_model_shap_interactions_agg"] = _df_or_empty(
            pipe.test_details_.get("cat_model_shap_interactions_agg"), ["orig_i", "orig_j", "score"]
        )
        out["ebm_model_importance_agg"] = _df_or_empty(
            pipe.test_details_.get("ebm_model_importance_agg"), ["base_feature", "importance"]
        )
        out["ebm_model_interactions_agg"] = _df_or_empty(
            pipe.test_details_.get("ebm_model_interactions_agg"), ["orig_i", "orig_j", "score"]
        )

        best_name = out.get("best_model_name")
        rtab_map = pipe.test_details_.get("rank_outcome_table_by_model", {}) or {}
        out["rank_outcome_table"] = _df_or_empty(
            rtab_map.get(best_name),
            [cfg.PATIENT_ID_COL, "factual_therapist_id", "factual_rank", cfg.LABEL_COL, "mu_hat_factual"],
        )
        out["posthoc_rank_on_outcome"] = (pipe.test_details_.get("posthoc_rank_on_outcome_by_model", {}) or {}).get(
            best_name, {}
        )

        out["rank_outcome_table_by_model"] = rtab_map
        out["posthoc_rank_on_outcome_by_model"] = pipe.test_details_.get("posthoc_rank_on_outcome_by_model", {}) or {}
        out["therapist_recommendation_counts_by_model"] = (
            pipe.test_details_.get("therapist_recommendation_counts_by_model", {}) or {}
        )
        return out

    def _run_once(*, repeat_index: int, out_dir: Path) -> Tuple[Dict[str, object], MatchPipeline]:
        pipe = MatchPipeline(cfg, run_dir=out_dir)
        pipe.set_data(df)
        pipe.set_repeat_index(repeat_index)
        out = pipe.run()

        out = _attach_details(out, pipe)
        out["output_dir"] = str(out_dir)

        if getattr(cfg, "OUTPUT_SAVE", True):
            _save_outputs_bundle(str(out_dir), out)

        return out, pipe

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------
    if not (n_rep > 1 and strat in {"per_therapist_random", "per-therapist-random", "ptr", "kfold", "k_fold", "kf"}):

        out_dir = _init_run_context(cfg, repeat_index=None)
        out, _pipe = _run_once(repeat_index=0, out_dir=out_dir)
        return out

    # ------------------------------------------------------------------
    # Repeated runs
    # ------------------------------------------------------------------
    base_dir = _init_run_context(cfg, repeat_index=None)  # creates base + sets cfg.OUTPUT_RUN_DIR

    run_summaries = []
    best_rows = []
    rank_tabs = []
    posthoc_best_rows = []

    cat_fi_runs, cat_int_runs = [], []
    shap_main_runs, shap_int_runs = [], []
    ebm_fi_runs, ebm_int_runs = [], []

    ther_best_runs = []

    outs_last = None
    used_features_last = {}
    best_model_last = None
    example_scores_last = pd.DataFrame()
    run_models_last = getattr(cfg, "RUN_MODELS", None)

    for r in range(n_rep):
        out_dir_r = _init_run_context(cfg, repeat_index=r)
        out_r, pipe = _run_once(repeat_index=r, out_dir=out_dir_r)
        outs_last = out_r

        used_features_last = out_r.get("used_features", {}) or used_features_last
        best_model_last = out_r.get("best_model_name") or best_model_last
        example_scores_last = out_r.get("example_scores_first_test_patient", example_scores_last)
        run_models_last = out_r.get("run_models", run_models_last)

        df_sum = out_r.get("summary", pd.DataFrame()).copy()
        if isinstance(df_sum, pd.DataFrame) and len(df_sum):
            df_sum["run"] = r
            run_summaries.append(df_sum)

            best_name = out_r.get("best_model_name")
            if best_name is None:
                best_name = df_sum.loc[df_sum["test_rmse"].idxmin(), "model"]
            best_row = df_sum.loc[df_sum["model"] == best_name].copy()
            if len(best_row):
                best_row["run"] = r
                best_rows.append(best_row.iloc[[0]])

        # per-patient rank/outcome for best model
        rtab_map = out_r.get("rank_outcome_table_by_model", {}) or {}
        best_name = out_r.get("best_model_name")
        rtab = rtab_map.get(best_name)
        if isinstance(rtab, pd.DataFrame) and len(rtab):
            t = rtab.copy()
            t["run"] = r
            t["best_model"] = best_name
            rank_tabs.append(t)

        # posthoc rank~outcome (best model)
        posthoc_map = out_r.get("posthoc_rank_on_outcome_by_model", {}) or {}
        posthoc = posthoc_map.get(best_name, {}) or {}
        if posthoc:
            posthoc_best_rows.append({
                "run": r,
                "best_model": best_name,
                "n": float(posthoc.get("n", np.nan)),
                "beta": float(posthoc.get("beta", np.nan)),
                "se": float(posthoc.get("se", np.nan)),
                "p": float(posthoc.get("p", np.nan)),
                "ci_low": float(posthoc.get("ci_low", np.nan)),
                "ci_high": float(posthoc.get("ci_high", np.nan)),
                "r2": float(posthoc.get("r2", np.nan)),
            })

        # therapist reco counts for best model
        reco_map = out_r.get("therapist_recommendation_counts_by_model", {}) or {}
        reco_best = reco_map.get(best_name)
        if isinstance(reco_best, pd.DataFrame) and len(reco_best):
            tb = reco_best.copy()
            tb["run"] = r
            tb["best_model"] = best_name
            ther_best_runs.append(tb)

        # collect Cat / EBM explanation tables (per run)
        cfi = out_r.get("cat_model_importance_agg")
        if isinstance(cfi, pd.DataFrame) and len(cfi):
            t = cfi.rename(columns={"orig_feature": "feature", "importance": "importance"}).copy()
            t["run"] = r
            cat_fi_runs.append(t[["feature", "importance", "run"]])

        cint = out_r.get("cat_model_interactions_agg")
        if isinstance(cint, pd.DataFrame) and len(cint):
            t = cint.copy()
            t["i"] = t["orig_i"].astype(str)
            t["j"] = t["orig_j"].astype(str)
            t["run"] = r
            cat_int_runs.append(t[["i", "j", "score", "run"]])

        cshap = out_r.get("cat_model_shap_agg")
        if isinstance(cshap, pd.DataFrame) and len(cshap):
            t = cshap.rename(columns={"orig_feature": "feature", "mean_abs_shap": "value"}).copy()
            t["run"] = r
            shap_main_runs.append(t[["feature", "value", "run"]])

        cshapint = out_r.get("cat_model_shap_interactions_agg")
        if isinstance(cshapint, pd.DataFrame) and len(cshapint):
            t = cshapint.copy()
            t["i"] = t["orig_i"].astype(str)
            t["j"] = t["orig_j"].astype(str)
            t["run"] = r
            shap_int_runs.append(t[["i", "j", "score", "run"]].rename(columns={"score": "value"}))

        efi = out_r.get("ebm_model_importance_agg")
        if isinstance(efi, pd.DataFrame) and len(efi):
            t = efi.rename(columns={"base_feature": "feature", "importance": "importance"}).copy()
            t["run"] = r
            ebm_fi_runs.append(t[["feature", "importance", "run"]])

        eint = out_r.get("ebm_model_interactions_agg")
        if isinstance(eint, pd.DataFrame) and len(eint):
            t = eint.copy()
            t["i"] = t["orig_i"].astype(str)
            t["j"] = t["orig_j"].astype(str)
            t["run"] = r
            ebm_int_runs.append(t[["i", "j", "score", "run"]])

    df_runs = pd.concat(run_summaries, ignore_index=True) if run_summaries else pd.DataFrame()
    df_best = pd.concat(best_rows, ignore_index=True) if best_rows else pd.DataFrame()
    df_posthoc_best = pd.DataFrame(posthoc_best_rows)

    cv_selected_model_counts = pd.DataFrame(columns=["model", "count", "share"])
    if len(df_best) and "model" in df_best.columns:
        vc = df_best["model"].value_counts(dropna=False)
        cv_selected_model_counts = pd.DataFrame({
            "model": vc.index.astype(str),
            "count": vc.values.astype(int),
            "share": (vc.values / float(vc.values.sum())) if vc.values.sum() else 0.0,
        })

    rmse_stats_per_model = pd.DataFrame()
    if len(df_runs) and "test_rmse" in df_runs.columns:
        rmse_stats_per_model = (
            df_runs.groupby("model", as_index=False)["test_rmse"]
            .agg(mean="mean", std="std", min="min", max="max", n="count")
            .sort_values("mean")
            .reset_index(drop=True)
        )

    rmse_stats_best = pd.DataFrame()
    if len(df_best) and "test_rmse" in df_best.columns:
        rmse_stats_best = (
            df_best["test_rmse"].agg(["mean", "std", "min", "max"])
            .to_frame(name="best_policy_test_rmse")
            .T.reset_index(drop=True)
        )
        rmse_stats_best["n_runs"] = len(df_best)

    def _agg_feat_mean_std(df_in: pd.DataFrame, val_col: str):
        if df_in is None or not len(df_in):
            return pd.DataFrame(columns=["feature", f"mean_{val_col}", f"std_{val_col}", "runs_present", f"mean_{val_col}_norm"])
        g = (
            df_in.groupby("feature", as_index=False)[val_col]
            .agg(**{f"mean_{val_col}": "mean", f"std_{val_col}": "std", "runs_present": "count"})
            .sort_values(f"mean_{val_col}", ascending=False)
            .reset_index(drop=True)
        )
        s = g[f"mean_{val_col}"].sum()
        g[f"mean_{val_col}_norm"] = g[f"mean_{val_col}"] / s if s > 0 else 0.0
        return g

    def _agg_pair_mean_std(df_in: pd.DataFrame, val_col: str, out_mean_name: str, out_std_name: str):
        if df_in is None or not len(df_in):
            return pd.DataFrame(columns=["orig_i", "orig_j", out_mean_name, out_std_name, "runs_present"])
        d = df_in.copy()
        a = d["i"].astype(str)
        b = d["j"].astype(str)
        left = np.where(a <= b, a, b)
        right = np.where(a <= b, b, a)
        d["orig_i"] = left
        d["orig_j"] = right
        g = (
            d.groupby(["orig_i", "orig_j"], as_index=False)[val_col]
            .agg(**{out_mean_name: "mean", out_std_name: "std", "runs_present": "count"})
            .sort_values(out_mean_name, ascending=False)
            .reset_index(drop=True)
        )
        return g

    cat_fi_df = pd.concat(cat_fi_runs, ignore_index=True) if cat_fi_runs else pd.DataFrame(columns=["feature", "importance", "run"])
    ebm_fi_df = pd.concat(ebm_fi_runs, ignore_index=True) if ebm_fi_runs else pd.DataFrame(columns=["feature", "importance", "run"])
    cat_int_df = pd.concat(cat_int_runs, ignore_index=True) if cat_int_runs else pd.DataFrame(columns=["i", "j", "score", "run"])
    ebm_int_df = pd.concat(ebm_int_runs, ignore_index=True) if ebm_int_runs else pd.DataFrame(columns=["i", "j", "score", "run"])
    shap_main_df = pd.concat(shap_main_runs, ignore_index=True) if shap_main_runs else pd.DataFrame(columns=["feature", "value", "run"])
    shap_int_df = pd.concat(shap_int_runs, ignore_index=True) if shap_int_runs else pd.DataFrame(columns=["i", "j", "value", "run"])

    cv_cat_importance_mean = _agg_feat_mean_std(cat_fi_df, "importance")
    cv_ebm_importance_mean = _agg_feat_mean_std(ebm_fi_df, "importance")
    cv_cat_interactions_mean = _agg_pair_mean_std(cat_int_df, "score", "mean_score", "std_score")
    cv_ebm_interactions_mean = _agg_pair_mean_std(ebm_int_df, "score", "mean_score", "std_score")

    cv_cat_shap_main_mean = _agg_feat_mean_std(shap_main_df, "value").rename(columns={
        "mean_value": "mean_abs_shap",
        "std_value": "std_abs_shap",
        "mean_value_norm": "mean_abs_shap_norm",
    })
    cv_cat_shap_interactions_mean = _agg_pair_mean_std(shap_int_df, "value", "mean_abs_interaction", "std_abs_interaction")

    rank_outcome_all = pd.concat(rank_tabs, ignore_index=True) if rank_tabs else pd.DataFrame()

    cv_ther_best_all = pd.concat(ther_best_runs, ignore_index=True) if ther_best_runs else pd.DataFrame()
    cv_ther_best_mean = pd.DataFrame()
    if isinstance(cv_ther_best_all, pd.DataFrame) and len(cv_ther_best_all):
        tid = cfg.THERAPIST_ID_COL
        agg_cols = [c for c in ["top1_count", "top5_count", "top1_share", "top5_share", "n_test_patients"] if c in cv_ther_best_all.columns]
        if agg_cols and tid in cv_ther_best_all.columns:
            cv_ther_best_mean = (
                cv_ther_best_all.groupby(tid, as_index=False)[agg_cols]
                .agg(["mean", "std", "sum", "count"])
            )
            cv_ther_best_mean.columns = [f"{a}_{b}" if b else str(a) for (a, b) in cv_ther_best_mean.columns.to_list()]
            cv_ther_best_mean = cv_ther_best_mean.reset_index(drop=True)
            order_key = "top1_share_mean" if "top1_share_mean" in cv_ther_best_mean.columns else (
                "top1_count_sum" if "top1_count_sum" in cv_ther_best_mean.columns else None
            )
            if order_key is not None:
                cv_ther_best_mean = cv_ther_best_mean.sort_values(order_key, ascending=False).reset_index(drop=True)

    out = dict(outs_last or {})
    out["output_dir"] = str(base_dir)
    out["cv_n_repeats"] = n_rep

    out["used_features"] = used_features_last
    out["best_model_name"] = best_model_last
    out["example_scores_first_test_patient"] = example_scores_last
    out["run_models"] = run_models_last

    out["cv_runs_summary"] = df_runs
    out["cv_best_policy_summary"] = df_best
    out["cv_best_model_by_run"] = df_best
    out["cv_selected_model_counts"] = cv_selected_model_counts

    out["cv_rmse_stats_per_model"] = rmse_stats_per_model
    out["cv_rmse_stats_best"] = rmse_stats_best

    out["cv_cat_importance_mean"] = cv_cat_importance_mean
    out["cv_ebm_importance_mean"] = cv_ebm_importance_mean
    out["cv_cat_interactions_mean"] = cv_cat_interactions_mean
    out["cv_ebm_interactions_mean"] = cv_ebm_interactions_mean
    out["cv_cat_shap_main_mean"] = cv_cat_shap_main_mean
    out["cv_cat_shap_interactions_mean"] = cv_cat_shap_interactions_mean

    out["cv_rank_outcome_all_runs"] = rank_outcome_all
    out["cv_outcomes_all_runs"] = rank_outcome_all
    out["cv_posthoc_best_by_run"] = df_posthoc_best

    out["cv_therapist_recommendations_best_all_runs"] = cv_ther_best_all
    out["cv_therapist_recommendations_best_mean"] = cv_ther_best_mean

    if getattr(cfg, "OUTPUT_SAVE", True):
        _save_outputs_bundle(str(base_dir), out)

    return out
