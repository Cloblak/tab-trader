"""Metrics, day-block bootstrap, Benjamini–Hochberg, and the calibrated-edge trade rule."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from .data import MARKETS, ROOT
from .run import slug

EPS = 1e-4
B = 2000
MARGINS_C = (0, 1, 2, 3, 5, 8)


# --------------------------------------------------------------- predictions
def load_preds(market: str, n_tag: str, model: str, sub: str = "preds") -> pd.DataFrame | None:
    d = ROOT / "results" / sub / market / n_tag / slug(model)
    files = sorted(d.glob("b*.parquet"))
    if not files:
        return None
    parts = []
    for f in files:
        p = pd.read_parquet(f)
        p["block"] = int(f.stem[1:])
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


def load_meta(market: str, n_tag: str, model: str, sub: str = "preds") -> list[dict]:
    d = ROOT / "results" / sub / market / n_tag / slug(model)
    return [json.loads(f.read_text()) for f in sorted(d.glob("b*.json"))]


# ------------------------------------------------------------------- metrics
def row_logloss(y, p) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def ece(y, p, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(y[m].mean() - p[m].mean())
    return float(tot / len(y))


def metrics(y, p) -> dict:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return {"n": int(len(y)), "logloss": float(row_logloss(y, p).mean()),
            "brier": float(((p - y) ** 2).mean()), "auc": float(roc_auc_score(y, p)),
            "ece": ece(y, p)}


def reliability(y, p, bins: int = 10) -> list[dict]:
    idx = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 10:
            out.append({"bin": b, "p_mean": float(np.mean(p[m])), "y_mean": float(np.mean(y[m])),
                        "n": int(m.sum())})
    return out


# ---------------------------------------------------------------- bootstrap
def day_weights(days: pd.Series, reps: int = B, seed: int = 0):
    """Return (codes, W): W[r, d] = how often day d is drawn in resample r."""
    codes, uniq = pd.factorize(days, sort=True)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=reps)
    return codes, W


def boot_mean(values: np.ndarray, days: pd.Series, reps: int = B, seed: int = 0) -> dict:
    """Day-block bootstrap of a row-weighted mean: point, 95% CI, two-sided p vs 0."""
    codes, W = day_weights(days, reps, seed)
    nd = W.shape[1]
    s = np.bincount(codes, weights=values, minlength=nd)
    c = np.bincount(codes, minlength=nd).astype(float)
    bs = (W @ s) / np.maximum(W @ c, 1)
    point = float(values.mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    return {"mean": point, "lo": float(lo), "hi": float(hi), "p": max(p, 1 / reps)}


def bh(pvals: list[float], q: float = 0.10) -> list[bool]:
    p = np.asarray(pvals)
    order = np.argsort(p)
    m = len(p)
    thresh = q * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(m, bool)
    out[order[:k]] = True
    return out.tolist()


# ------------------------------------------------------------------ trading
def fee_c(price_c):
    """Kalshi taker fee in cents per contract, unrounded: 0.07 * C * (1 - C), C in dollars."""
    c = np.asarray(price_c, float) / 100
    return 7.0 * c * (1 - c)


def _trades(df: pd.DataFrame, q: np.ndarray, margin: float) -> pd.DataFrame:
    """Calibrated-edge rule; at most one trade per market (earliest decision)."""
    ask, bid, y = df["ask"].to_numpy(), df["bid"].to_numpy(), df["label"].to_numpy()
    no_ask = 100 - bid
    e_yes = 100 * q - ask - fee_c(ask)
    e_no = 100 * (1 - q) - no_ask - fee_c(no_ask)
    side = np.where((e_yes >= e_no) & (e_yes > margin), 1, np.where(e_no > margin, -1, 0))
    t = df.assign(side=side, edge=np.maximum(e_yes, e_no))
    t = t[t.side != 0].sort_values("decision_ts").drop_duplicates("market", keep="first")
    px = np.where(t.side == 1, t.ask, 100 - t.bid)
    win = np.where(t.side == 1, t.label, 1 - t.label)
    t = t.assign(price_c=px, net_c=100 * win - px - fee_c(px))
    return t


def trade_block(val: pd.DataFrame, test: pd.DataFrame, p_val, p_test) -> tuple[pd.DataFrame, dict]:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
    iso.fit(p_val, val["label"].to_numpy())
    q_val, q_te = iso.predict(p_val), iso.predict(p_test)
    best = None
    for m in MARGINS_C:
        tv = _trades(val, q_val, m)
        tot = tv["net_c"].sum() if len(tv) >= 20 else -np.inf
        if best is None or tot > best[0]:
            best = (tot, m)
    m = best[1]
    return _trades(test, q_te, m), {"margin_c": m, "val_total_c": float(best[0])}


def max_drawdown(cum: np.ndarray) -> float:
    if len(cum) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))
    return float((np.concatenate([[0.0], cum]) - peak).min())
