"""Model zoo: the market itself, six classic learners, and TabPFN-3.5.

Every model exposes ``fit(X, y)`` and ``predict_proba(X)[:, 1]`` and runs on CPU.
Classic learners get a fair shot: library defaults *and* a 20-config random
search scored on a held-out validation slice. TabPFN is never tuned.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_THREADS = int(os.environ.get("TT_THREADS", os.cpu_count() or 4))
EPS = 1e-4


class MarketPrice(BaseEstimator, ClassifierMixin):
    """The contract's own mid price, read as a probability. Nothing is fitted."""

    def __init__(self, col: int = 0):
        self.col = col

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.clip(np.asarray(X)[:, self.col], EPS, 1 - EPS)
        return np.column_stack([1 - p, p])


def _lr(**kw):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, **kw))


def _mlp(**kw):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         MLPClassifier(early_stopping=True, max_iter=500, random_state=0, **kw))


def _rf(**kw):
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_jobs=N_THREADS, random_state=0, **kw))


def _xgb(**kw):
    from xgboost import XGBClassifier

    return XGBClassifier(n_jobs=N_THREADS, random_state=0, tree_method="hist",
                         eval_metric="logloss", **kw)


def _lgbm(**kw):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(n_jobs=N_THREADS, random_state=0, verbose=-1, **kw)


def _cat(**kw):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(thread_count=N_THREADS, random_seed=0, verbose=0,
                              allow_writing_files=False, **kw)


def _loguni(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


# Random-search spaces. Each sampler returns constructor kwargs.
SPACES: dict[str, Callable[[np.random.Generator], dict]] = {
    "Logistic regression": lambda r: {"C": _loguni(r, 1e-3, 1e2)},
    "Random forest": lambda r: {
        "n_estimators": int(r.choice([200, 400, 600])),
        "min_samples_leaf": int(r.choice([1, 5, 10, 25, 50, 100])),
        "max_features": r.choice(["sqrt", 0.3, 0.5, 0.8]),
        "max_depth": r.choice([None, 6, 10, 16])},
    "XGBoost": lambda r: {
        "n_estimators": int(r.choice([100, 200, 400, 800])),
        "max_depth": int(r.choice([2, 3, 4, 6, 8])),
        "learning_rate": _loguni(r, 0.01, 0.3),
        "subsample": float(r.uniform(0.5, 1.0)),
        "colsample_bytree": float(r.uniform(0.4, 1.0)),
        "min_child_weight": _loguni(r, 1, 50),
        "reg_lambda": _loguni(r, 0.1, 20)},
    "LightGBM": lambda r: {
        "n_estimators": int(r.choice([100, 200, 400, 800])),
        "num_leaves": int(r.choice([4, 8, 15, 31, 63])),
        "learning_rate": _loguni(r, 0.01, 0.3),
        "min_child_samples": int(r.choice([5, 20, 50, 100, 200])),
        "subsample": float(r.uniform(0.5, 1.0)), "subsample_freq": 1,
        "colsample_bytree": float(r.uniform(0.4, 1.0)),
        "reg_lambda": _loguni(r, 0.01, 20)},
    "CatBoost": lambda r: {
        "iterations": int(r.choice([200, 500, 1000])),
        "depth": int(r.choice([3, 4, 6, 8])),
        "learning_rate": _loguni(r, 0.01, 0.3),
        "l2_leaf_reg": _loguni(r, 1, 30)},
    "MLP": lambda r: {
        "hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)][int(r.integers(4))],
        "alpha": _loguni(r, 1e-5, 1e-1),
        "learning_rate_init": _loguni(r, 1e-4, 1e-2)},
}
BUILDERS: dict[str, Callable[..., Any]] = {
    "Logistic regression": _lr, "Random forest": _rf, "XGBoost": _xgb,
    "LightGBM": _lgbm, "CatBoost": _cat, "MLP": _mlp,
}
CLASSIC = list(BUILDERS)


# ------------------------------------------------------------------- TabPFN
def tabpfn_token() -> str | None:
    """TABPFN_TOKEN from the environment (get one at https://ux.priorlabs.ai)."""
    return os.environ.get("TABPFN_TOKEN") or os.environ.get("TABFN_TOKEN")


def make_tabpfn(variant: str = "3.5", **kw):
    """Local TabPFN-3.5 (open weights) on CPU. ``variant`` is "3.5" or "3.5-fast"."""
    tok = tabpfn_token()
    if tok:
        os.environ.setdefault("TABPFN_TOKEN", tok)
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    version = {"3.5": ModelVersion.V3_5, "3.5-fast": ModelVersion.V3_5_FAST}[variant]
    opts = {"device": "cpu", "random_state": 0, "fit_mode": "fit_with_cache"}
    opts.update(kw)
    return TabPFNClassifier.create_default_for_version(version, **opts)


def make_tabpfn_thinking(**kw):
    """TabPFN-3.5 Thinking via the Prior Labs API (tabpfn-client)."""
    import tabpfn_client
    from tabpfn_client import TabPFNClassifier

    tok = tabpfn_token()
    if tok:
        tabpfn_client.set_access_token(tok)
    opts = {"thinking_mode": True}
    opts.update(kw)
    return TabPFNClassifier(**opts)


# ------------------------------------------------------------------ fitting
@dataclass
class FitResult:
    model: Any
    params: dict = field(default_factory=dict)
    fit_s: float = 0.0
    predict_s: float = 0.0
    search_s: float = 0.0
    val_logloss: float = np.nan


def _ll(y, p) -> float:
    return float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1]))


def fit_classic(name: str, X_tr, y_tr, X_val=None, y_val=None, tune: bool = False,
                n_iter: int = 20, seed: int = 0) -> FitResult:
    """Fit a classic learner; with ``tune`` pick the best of default + n_iter random configs on val."""
    build = BUILDERS[name]
    t0 = time.perf_counter()
    if not tune:
        m = build()
        m.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0
        vl = _ll(y_val, m.predict_proba(X_val)[:, 1]) if X_val is not None else np.nan
        return FitResult(m, {}, fit_s, val_logloss=vl)
    rng = np.random.default_rng(seed)
    configs = [{}] + [SPACES[name](rng) for _ in range(n_iter)]
    best = None
    for cfg in configs:
        try:
            m = build(**cfg)
            m.fit(X_tr, y_tr)
            vl = _ll(y_val, m.predict_proba(X_val)[:, 1])
        except Exception:  # an invalid sampled config just loses
            continue
        if best is None or vl < best[0]:
            best = (vl, cfg, m)
    search_s = time.perf_counter() - t0
    vl, cfg, m = best
    # Time one clean refit of the winner so fit_s is comparable across models.
    t1 = time.perf_counter()
    m = build(**cfg)
    m.fit(X_tr, y_tr)
    return FitResult(m, {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in cfg.items()},
                     time.perf_counter() - t1, search_s=search_s, val_logloss=vl)


def timed_predict(res: FitResult, X) -> np.ndarray:
    t0 = time.perf_counter()
    p = np.asarray(res.model.predict_proba(X))[:, 1]
    res.predict_s += time.perf_counter() - t0
    return np.clip(p, EPS, 1 - EPS)


def fit_timed(model, X_tr, y_tr) -> FitResult:
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    return FitResult(model, {}, time.perf_counter() - t0)
