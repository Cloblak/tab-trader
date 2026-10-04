"""Bottom line up front: every model as the trade filter of the author's live strategy.

Reads the private signal bundle (the strategy's own features, which are never published), fits each
model on a fixed Train / Test / Holdout split, and writes only scores, trade flags and outcomes:

* ``data/strategy/bluf_predictions.parquet`` — one row per signal: day, split, entry price,
  win flag, net cents per contract, ``p__<model>`` and ``take__<model>``.
* ``data/strategy/live_trades.parquet`` — the real live trades since 2026-09-12 (Kalshi fills, cents per
  contract) with a flag for whether TabPFN would have kept each one.

Splits (by signal time, UTC): Train < 2026-07-14 ≤ Test < 2026-09-12 ≤ Holdout. The Holdout is the period
the strategy traded live.

* Train scores are out-of-fold (5 contiguous folds), so no model is scored on rows it was fitted on.
* Test is used once, to calibrate each model and pick its trading margin.
* Holdout is scored with everything frozen.

    python scripts/package_bluf.py --src <private artifacts dir> --live-threshold <private>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tabtrader.evaluate import fee_c  # noqa: E402
from tabtrader.models import CLASSIC, BUILDERS, fit_classic, make_tabpfn  # noqa: E402

TEST_START = pd.Timestamp("2026-07-14", tz="UTC")
HOLD_START = pd.Timestamp("2026-09-12", tz="UTC")
MARGINS = (0, 1, 2, 3, 5, 8)
K = 5
EPS = 1e-4


def edge_take(q: np.ndarray, px: np.ndarray, m: float) -> np.ndarray:
    return 100 * q - px - fee_c(px) > m


def decision_layer(p_test, y_test, px_test, net_test):
    """Isotonic calibration + margin, both chosen on Test only."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p_test, y_test)
    best = None
    for m in MARGINS:
        t = edge_take(iso.predict(p_test), px_test, m)
        tot = net_test[t].sum() if t.sum() >= 20 else -np.inf
        if best is None or tot > best[0]:
            best = (tot, m)
    return iso, best[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--live-threshold", type=float, required=True)
    args = ap.parse_args()
    src = Path(args.src)
    raw = pd.read_parquet(src / "dl_bundle" / "signals.parquet")
    d = raw[raw["clean"] & raw["live_hours"]].sort_values("entry_ts").reset_index(drop=True)
    feats = [c for c in d.columns if c.startswith("raw__")]

    def design(frame):
        x = frame[feats].to_numpy(float)
        return np.column_stack([x, np.log(np.clip(frame["px"], 1, 99) / (100 - np.clip(frame["px"], 1, 99)))])
    X = design(d)
    # Every live-hours signal in the holdout period (clean quote or not), so each real live trade gets a verdict.
    ex_rows = raw[raw["live_hours"] & (raw["entry_ts"] >= HOLD_START)].reset_index(drop=True)
    X_ex = design(ex_rows)
    y = d["won"].to_numpy(int)
    px = d["px"].to_numpy(float)
    net = d["ex_net_c"].to_numpy(float)
    split = np.where(d["entry_ts"] < TEST_START, "train", np.where(d["entry_ts"] < HOLD_START, "test", "holdout"))
    tr, te, ho = (split == "train"), (split == "test"), (split == "holdout")
    print({s: int((split == s).sum()) for s in ("train", "test", "holdout")}, flush=True)

    out = pd.DataFrame({"day": d["entry_ts"].dt.strftime("%Y-%m-%d"), "split": split, "px": px.round(2),
                        "won": y, "net_c_exits": net.round(3), "net_c_hold": d["hold_net_c"].round(3)})
    meta = {}
    extra = {}
    folds = np.array_split(np.where(tr)[0], K)

    def run(name, fit_predict):
        t0 = time.perf_counter()
        p = np.full(len(d), np.nan)
        for f in folds:  # out-of-fold scores on Train
            fit_idx = np.setdiff1d(np.where(tr)[0], f)
            p[f] = fit_predict(X[fit_idx], y[fit_idx], X[f])
        n_main = int((te | ho).sum())
        both = fit_predict(X[tr], y[tr], np.vstack([X[te | ho], X_ex]))
        p[te | ho] = both[:n_main]
        p = np.clip(p, EPS, 1 - EPS)
        iso, m = decision_layer(p[te], y[te], px[te], net[te])
        p_ex = np.clip(both[n_main:], EPS, 1 - EPS)
        extra[name] = edge_take(iso.predict(p_ex), ex_rows["px"].to_numpy(float), m)
        out[f"p__{name}"] = p.round(5)
        out[f"take__{name}"] = edge_take(iso.predict(p), px, m)
        meta[name] = {"margin_c": m, "seconds": round(time.perf_counter() - t0, 1)}
        print(name, meta[name], flush=True)

    # Classic learners: hyper-parameters picked once (fit on Train, scored on Test log loss), then reused.
    for c in CLASSIC:
        res = fit_classic(c, X[tr], y[tr], X[te], y[te], tune=True)
        cfg = res.params

        def fp(Xa, ya, Xb, c=c, cfg=cfg):
            m = BUILDERS[c](**cfg)
            m.fit(Xa, ya)
            return m.predict_proba(Xb)[:, 1]
        run(c, fp)

    def tabpfn(Xa, ya, Xb):
        m = make_tabpfn("3.5")
        m.fit(Xa, ya)
        return m.predict_proba(Xb)[:, 1]
    run("TabPFN-3.5", tabpfn)

    # Baselines.
    run("Market price", lambda Xa, ya, Xb: np.clip(1 / (1 + np.exp(-Xb[:, -1])), EPS, 1 - EPS))
    pdep = d["p_deployed"].to_numpy(float)
    out["p__Live gate"] = pd.Series(pdep).rank(pct=True).round(5).to_numpy()  # rank: hides the threshold
    out["take__Live gate"] = pdep >= args.live_threshold
    out["take__Every signal"] = True

    # Rows within a day are shuffled so row order carries no intraday timing. Equity math in
    # tabtrader.bankroll is per day and order-independent (fractional stakes).
    rng = np.random.default_rng(0)
    order = np.lexsort((rng.random(len(out)), out["day"].to_numpy()))
    dst = ROOT / "data" / "strategy"
    out.iloc[order].reset_index(drop=True).to_parquet(dst / "bluf_predictions.parquet", index=False)

    # Live trades since the holdout start, with real Kalshi fills, and TabPFN's verdict on each.
    live = pd.read_parquet(src / "live.parquet")
    live = live[(live["entry_ts"] >= HOLD_START) & live["real_net_per_ct"].notna()].copy()
    tk = dict(zip(ex_rows["market"], extra["TabPFN-3.5"]))
    lg = dict(zip(ex_rows["market"], ex_rows["p_deployed"].to_numpy(float) >= args.live_threshold))
    live = live[live["market"].isin(tk.keys())]  # the few live trades with no recorded signal are dropped
    lt = pd.DataFrame({
        "day": live["entry_ts"].dt.strftime("%Y-%m-%d"),
        "px_fill": live["fill_entry_c"].round(2),
        "real_net_c": live["real_net_per_ct"].round(3),
        "tabpfn_keeps": live["market"].map(tk).astype(bool),
        "backtest_live_gate_takes": live["market"].map(lg).astype(bool),
    })
    order = np.lexsort((rng.random(len(lt)), lt["day"].to_numpy()))
    lt.iloc[order].reset_index(drop=True).to_parquet(dst / "live_trades.parquet", index=False)
    (dst / "bluf_meta.json").write_text(json.dumps(
        {"splits": {"train": ["2026-04-15", "2026-07-13"], "test": ["2026-07-14", "2026-09-11"],
                    "holdout": ["2026-09-12", str(d["entry_ts"].max().date())]},
         "n": {s: int((split == s).sum()) for s in ("train", "test", "holdout")},
         "n_features_private": len(feats) + 1, "models": meta, "live_trades": int(len(lt))}, indent=1))
    print("wrote", dst / "bluf_predictions.parquet", out.shape, "live", lt.shape)


if __name__ == "__main__":
    main()
