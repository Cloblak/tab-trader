"""Rolling weekly walk-forward of TabPFN-3.5 as the live strategy's trade filter (PREREG amendment 3).

Each test week (Mon-Sun, from 2026-05-11) every model is refit on the previous 12 weeks of signals and
decides each new signal with one rule: trade iff 100*p - entry - fee > 0. Nothing is tuned.

Reads the private signal bundle (features never published) and writes only scores, decisions and outcomes:

* ``data/strategy/rolling_predictions.parquet`` — one row per test-period signal: day, week, entry price,
  win flag, net ¢ per contract, and for each model ``p__``, ``take__`` and ``frac__`` (half-Kelly stake).
* ``data/strategy/rolling_live_trades.parquet`` — the real live trades since 2026-09-12 (Kalshi fills) with
  the rolling TabPFN verdict and stake for each.

    python scripts/package_rolling.py --src <private artifacts dir> --live-threshold <private>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tabtrader.evaluate import fee_c  # noqa: E402
from tabtrader.models import BUILDERS, make_tabpfn  # noqa: E402

warnings.filterwarnings("ignore")
FIRST_WEEK = pd.Timestamp("2026-05-11", tz="UTC")
CONTEXT_WEEKS = 12
GAP = pd.Timedelta(minutes=15)
LIVE_START = pd.Timestamp("2026-09-12", tz="UTC")
EPS = 1e-4
CAP, KELLY_MULT = 0.15, 0.5


def kelly_stake(p, px_c):
    cost = (np.asarray(px_c, float) + fee_c(px_c)) / 100
    return np.minimum(CAP, KELLY_MULT * np.clip((np.asarray(p, float) - cost) / (1 - cost), 0, None))


def edge_take(p, px_c):
    return 100 * np.asarray(p, float) - np.asarray(px_c, float) - fee_c(px_c) > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--live-threshold", type=float, required=True)
    args = ap.parse_args()
    src = Path(args.src)
    raw = pd.read_parquet(src / "dl_bundle" / "signals.parquet")
    feats = [c for c in raw.columns if c.startswith("raw__")]

    def design(f: pd.DataFrame) -> np.ndarray:
        x = f[feats].astype(float).copy()
        x["raw__bb_pct"] = x["raw__bb_pct"].clip(-5, 5)  # a handful of near-zero-width bands blow up
        px = f["px"].astype(float).clip(1, 99)
        return np.column_stack([x.to_numpy(), np.log(px / (100 - px))])

    d = raw[raw["clean"] & raw["live_hours"]].sort_values("entry_ts").reset_index(drop=True)
    X, y = design(d), d["won"].to_numpy(int)
    # Every in-hours signal in the live period (clean quote or not) gets a verdict, for the real-fill check.
    ex = raw[raw["live_hours"] & (raw["entry_ts"] >= LIVE_START)].reset_index(drop=True)
    X_ex = design(ex)

    end = d["entry_ts"].max()
    weeks = pd.date_range(FIRST_WEEK, end, freq="7D")
    models = ["TabPFN-3.5", "Logistic regression", "LightGBM"]
    P = {m: np.full(len(d), np.nan) for m in models}
    P_ex = {m: np.full(len(ex), np.nan) for m in models}
    week_of = np.full(len(d), -1)
    log = []
    for i, a in enumerate(weeks):
        b = a + pd.Timedelta(days=7)
        ctx = (d["entry_ts"] < a - GAP) & (d["entry_ts"] >= a - pd.Timedelta(weeks=CONTEXT_WEEKS))
        tst = (d["entry_ts"] >= a) & (d["entry_ts"] < b)
        tex = (ex["entry_ts"] >= a) & (ex["entry_ts"] < b)
        if tst.sum() == 0 and tex.sum() == 0:
            continue
        ci = np.where(ctx)[0][-5000:]
        week_of[tst.to_numpy()] = i
        Xq = np.vstack([X[tst.to_numpy()], X_ex[tex.to_numpy()]])
        nt = int(tst.sum())
        for m in models:
            t0 = time.perf_counter()
            if m == "TabPFN-3.5":
                est = make_tabpfn("3.5")
            else:
                est = BUILDERS[m]()
            est.fit(X[ci], y[ci])
            p = np.clip(est.predict_proba(Xq)[:, 1], EPS, 1 - EPS) if len(Xq) else np.array([])
            P[m][tst.to_numpy()] = p[:nt]
            P_ex[m][tex.to_numpy()] = p[nt:]
            log.append({"week": str(a.date()), "model": m, "n_context": int(len(ci)), "n_test": nt,
                        "seconds": round(time.perf_counter() - t0, 1)})
        print(f"week {a.date()} context {len(ci)} test {nt} live-period extra {int(tex.sum())}", flush=True)

    keep = week_of >= 0
    out = pd.DataFrame({"day": d["entry_ts"].dt.strftime("%Y-%m-%d"), "week": week_of,
                        "px": d["px"].round(2), "won": y, "net_c_exits": d["ex_net_c"].round(3)})[keep].copy()
    out["split"] = "rolling"
    px = out["px"].to_numpy(float)
    for m in models:
        p = P[m][keep]
        out[f"p__{m}"] = p.round(5)
        out[f"take__{m}"] = edge_take(p, px)
        out[f"frac__{m} + Kelly"] = kelly_stake(p, px).round(4)
        out[f"take__{m} + Kelly"] = out[f"take__{m}"]
        # Secondary rule: isotonic map from the model's own out-of-sample predictions in the previous 4 weeks.
        q = p.copy()
        wk = out["week"].to_numpy()
        for w in np.unique(wk):
            prev = (wk < w) & (wk >= w - 4)
            if prev.sum() >= 100 and len(np.unique(out["won"].to_numpy()[prev])) == 2:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
                iso.fit(p[prev], out["won"].to_numpy()[prev])
                q[wk == w] = iso.predict(p[wk == w])
        out[f"take__{m} · calibrated"] = edge_take(q, px)
    pdep = d["p_deployed"].to_numpy(float)[keep]
    out["take__Live gate"] = pdep >= args.live_threshold
    out["take__Every signal"] = True
    # Rows keep their order within each day as ``seq`` (no timestamps): the capped bankroll simulation
    # needs the sequence, and the sequence alone reveals nothing about time of day.
    out["seq"] = out.groupby("day").cumcount()
    order = np.arange(len(out))
    dst = ROOT / "data" / "strategy"
    out.iloc[order].reset_index(drop=True).to_parquet(dst / "rolling_predictions.parquet", index=False)

    live = pd.read_parquet(src / "live.parquet")
    live = live[(live["entry_ts"] >= LIVE_START) & live["real_net_per_ct"].notna()]
    pk = dict(zip(ex["market"], P_ex["TabPFN-3.5"]))
    live = live[live["market"].isin(pk) & live["market"].map(pk).notna()]
    p_live = live["market"].map(pk).to_numpy(float)
    lt = pd.DataFrame({"day": live["entry_ts"].dt.strftime("%Y-%m-%d"), "px_fill": live["fill_entry_c"].round(2),
                       "real_net_c": live["real_net_per_ct"].round(3),
                       "tabpfn_keeps": edge_take(p_live, live["fill_entry_c"].to_numpy(float)),
                       "frac_kelly": kelly_stake(p_live, live["fill_entry_c"].to_numpy(float)).round(4)})
    lt = lt.assign(_ts=live["entry_ts"].to_numpy()).sort_values("_ts").drop(columns="_ts").reset_index(drop=True)
    lt["seq"] = lt.groupby("day").cumcount()
    order = np.arange(len(lt))
    lt.iloc[order].reset_index(drop=True).to_parquet(dst / "rolling_live_trades.parquet", index=False)
    (dst / "rolling_meta.json").write_text(json.dumps(
        {"first_week": str(FIRST_WEEK.date()), "last_signal": str(end.date()), "context_weeks": CONTEXT_WEEKS,
         "weeks": int(len(np.unique(week_of[keep]))), "signals": int(keep.sum()), "live_trades": int(len(lt)),
         "rule": "trade iff 100*p - entry - fee > 0; flat 15% or half-Kelly capped at 15%", "fits": log}, indent=1))
    print("wrote", dst / "rolling_predictions.parquet", out.shape, "live", lt.shape)


if __name__ == "__main__":
    main()
