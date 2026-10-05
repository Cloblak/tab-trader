"""Data preparation for the report (everything is read from data/ and results/)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import bankroll, eda
from .data import DATA, MARKETS, ROOT
from .features import FEATURE_GROUPS

BLUF_MODELS = ["TabPFN-3.5 + Kelly", "TabPFN-3.5", "Live gate", "Every signal", "Market price", "Logistic regression",
               "Random forest", "XGBoost", "LightGBM", "CatBoost", "MLP"]
SPLITS = ["train", "test", "holdout"]


def r(x, k=2):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


# ----------------------------------------------------------------- data blocks
def explorer_data() -> dict:
    raw = DATA / "raw_week"
    q = pd.read_parquet(raw / "kxbtc15m_quotes_1s.parquet")
    s = pd.read_parquet(raw / "settlements_15m.parquet")
    q = q[q.quote_valid == 1].copy()
    q["t"] = q.sec_end.dt.floor("15s")
    g = q.groupby(["market", "t"]).agg(bid=("bid", "last"), ask=("ask", "last"), btc=("btc", "last")).reset_index()
    out15 = []
    for _, m in s.sort_values("close_time").iterrows():
        x = g[g.market == m.market]
        if len(x) < 20:
            continue
        t0 = m.open_time
        out15.append({
            "m": m.market, "open": m.open_time.strftime("%Y-%m-%d %H:%M"), "k": r(m.strike, 2), "y": int(m.label),
            "s": [int((t - t0).total_seconds()) for t in x.t],
            "b": [r(v, 1) for v in x.bid], "a": [r(v, 1) for v in x.ask], "x": [r(v, 0) for v in x.btc],
        })
    qh = pd.read_parquet(raw / "kxbtcd_quotes_10s.parquet")
    sh = pd.read_parquet(raw / "settlements_1h.parquet")
    qh = qh[qh.quote_valid == 1].copy()
    qh["t"] = qh.sec_end.dt.floor("1min")
    gh = qh.groupby(["market", "t"]).agg(bid=("bid", "last"), ask=("ask", "last")).reset_index()
    outh = []
    for ct, legs in sh.groupby("close_time"):
        lines = []
        for _, m in legs.sort_values("strike").iterrows():
            x = gh[gh.market == m.market]
            if len(x) < 10:
                continue
            lines.append({"k": r(m.strike, 2), "y": int(m.label),
                          "s": [int((t - m.open_time).total_seconds() // 60) for t in x.t],
                          "p": [r((b + a) / 2, 1) for b, a in zip(x.bid, x.ask)]})
        if len(lines) >= 3:
            outh.append({"close": ct.strftime("%Y-%m-%d %H:%M"), "legs": lines})
    return {"m15": out15, "m1h": outh}


def eda_data() -> dict:
    t = eda.eda_tables()
    v = eda.volume_totals(t)
    d15, d1h = t["tape_15m_daily"], t["tape_1h_daily"]
    w = eda.weekly_quality(t)
    out = {
        "totals": v, "milestones": eda.MILESTONES,
        "daily": {"15m": [[str(a.date()), int(b)] for a, b in zip(d15.day, d15.rows)],
                  "1h": [[str(a.date()), int(b)] for a, b in zip(d1h.day, d1h.rows)]},
        "weekly": {k: [[str(a.date()), r(b, 2), r(c, 2)] for a, b, c in zip(g.week, g.valid_pct, g.crossed_pct)]
                   for k, g in w.groupby("market")},
        "candle": {k: eda.candle_agreement(k).round(2).to_dict("records") for k in MARKETS},
        "funnel": {k: eda.cleaning_funnel(k) for k in MARKETS},
        "calib": {k: eda.market_calibration(k).round(4).to_dict("records") for k in MARKETS},
    }
    lab = t["labels_daily"].groupby("series").agg(m=("markets", "sum"), y=("yes", "sum"))
    out["yes_rate"] = {s: r(row.y / row.m, 3) for s, row in lab.iterrows()}
    return out


def feature_info() -> list[dict]:
    from sklearn.metrics import roc_auc_score

    from .data import load_market

    df = load_market("15m")
    tr = df[df.decision_ts < "2026-08-03"]
    resid = tr.label - tr.p_mid
    rows = []
    for g, fs in FEATURE_GROUPS.items():
        for f in fs:
            x = tr[f].fillna(tr[f].median())
            rows.append({"f": f, "g": g, "auc": r(roc_auc_score(tr.label, x), 3),
                         "rc": r(np.corrcoef(x, resid)[0, 1], 3)})
    return rows


def bluf_data() -> dict:
    d = pd.read_parquet(DATA / "strategy" / "bluf_predictions.parquet")
    meta = json.loads((DATA / "strategy" / "bluf_meta.json").read_text())
    out = {"meta": meta, "splits": {}, "auc": {}}
    for sp in SPLITS:
        out["splits"][sp] = bankroll.run_models(d, BLUF_MODELS, sp)
        g = d[d.split == sp]
        out["auc"][sp] = {m: r(roc_auc_score(g.won, g[f"p__{m}"]), 4) for m in BLUF_MODELS if f"p__{m}" in g}
        out["auc"][sp]["TabPFN-3.5 + Kelly"] = out["auc"][sp]["TabPFN-3.5"]
    # Live period: real fills vs the TabPFN filter, both at $100 and 15% per trade.
    lt = pd.read_parquet(DATA / "strategy" / "live_trades.parquet")
    ho = d[d.split == "holdout"]
    days = pd.date_range(pd.to_datetime(min(lt.day.min(), ho.day.min())),
                         pd.to_datetime(max(lt.day.max(), ho.day.max())), freq="D")
    series = {}
    for name, frame, net, px, take, frac in [
        ("Live, as traded (real fills)", lt, "real_net_c", "px_fill", np.ones(len(lt), bool), bankroll.RISK),
        ("TabPFN veto + Kelly sizing (real fills)", lt, "real_net_c", "px_fill", lt.tabpfn_keeps.to_numpy(),
         lt["frac_kelly"].to_numpy()),
        ("TabPFN veto, flat 15% (real fills)", lt, "real_net_c", "px_fill", lt.tabpfn_keeps.to_numpy(), bankroll.RISK),
        ("TabPFN + Kelly (backtest)", ho, "net_c_exits", "px", ho["take__TabPFN-3.5 + Kelly"].to_numpy(),
         ho["frac__TabPFN-3.5 + Kelly"].to_numpy()),
        ("Live gate (backtest)", ho, "net_c_exits", "px", ho["take__Live gate"].to_numpy(), bankroll.RISK),
    ]:
        eq = bankroll.daily_equity(frame.day, frame[net].to_numpy(), frame[px].to_numpy(), take, days, risk=frac)
        ts = bankroll.tear_sheet(eq, int(take.sum()), int((frame[net].to_numpy()[take] > 0).sum()))
        ts["curve"] = [[str(k.date()), round(float(v), 2)] for k, v in eq.items()]
        ts["c_per_ct"] = r(frame[net].to_numpy()[take].mean(), 2)
        series[name] = ts
    out["live"] = {"series": series, "n_live": int(len(lt)), "n_kept": int(lt.tabpfn_keeps.sum()),
                   "kept_c": r(lt.real_net_c[lt.tabpfn_keeps].mean(), 2),
                   "skipped_c": r(lt.real_net_c[~lt.tabpfn_keeps].mean(), 2),
                   "first": str(lt.day.min()), "last": str(lt.day.max())}
    return out


def bench_data() -> dict:
    S = json.loads((ROOT / "results" / "summary.json").read_text())
    ctrl_p = ROOT / "results" / "controls.json"
    S["_controls"] = json.loads(ctrl_p.read_text()) if ctrl_p.exists() else None
    return S


# ------------------------------------------------------------------- narrative
def fmt_ci(v, k=4, unit=""):
    return f"{v['mean']:+.{k}f}{unit} [{v['lo']:+.{k}f}, {v['hi']:+.{k}f}]"


def headline(S: dict) -> dict:
    """Plain-language answers, computed from the results (never typed by hand)."""
    out = {}
    for mk in MARKETS:
        if not S[mk].get("policy_A") or "TabPFN-3.5" not in S[mk]["policy_A"]["models"]:
            out[mk] = {"rank": None}
            continue
        L = S[mk]["policy_A"]["models"]
        learners = {m: v for m, v in L.items() if m != "Market price" and m != "TabPFN-3.5-Thinking"}
        ranked = sorted(learners, key=lambda m: learners[m]["logloss"])
        tab = L.get("TabPFN-3.5")
        classic = [m for m in ranked if not m.startswith("TabPFN")]
        best_c = classic[0] if classic else None
        beats = [m for m in classic if L[m]["vs_ref"]["mean"] > 0]
        sig = [m for m in classic if L[m]["vs_ref"].get("bh_pass") and L[m]["vs_ref"]["mean"] > 0]
        out[mk] = {
            "rank": ranked.index("TabPFN-3.5") + 1 if tab else None, "n_learners": len(ranked),
            "best_classic": best_c, "tab_ll": tab["logloss"] if tab else None,
            "best_classic_ll": L[best_c]["logloss"] if best_c else None,
            "vs_best_classic": L[best_c]["vs_ref"] if best_c else None,
            "beats": len(beats), "sig": len(sig), "n_classic": len(classic),
            "vs_market": tab["vs_market"] if tab else None, "market_ll": L["Market price"]["logloss"],
            "rows": S[mk]["policy_A"]["n_rows"], "days": S[mk]["policy_A"]["n_days"],
        }
    return out


