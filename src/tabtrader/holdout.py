"""Train / test / holdout with the TabPFN API, the way the Prior Labs playground does it.

    python -m tabtrader.holdout export   # data/tabpfn_ready/kxbtc15m_features.csv (playground-ready)
    python -m tabtrader.holdout run      # results/holdout_api.json (needs TABPFN_TOKEN; public data only)

The split is by time, never random: on market data a shuffled split trains on the future.
  train   2026-06-07 .. 2026-08-02   fit
  test    2026-08-03 .. 2026-09-06   pick the trading margin
  holdout 2026-09-07 .. 2026-10-03   scored once, nothing changes after
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from .data import DATA, ROOT, load_market
from .evaluate import MARGINS_C, _trades, boot_mean, metrics
from .features import FEATURES

CSV = DATA / "tabpfn_ready" / "kxbtc15m_features.csv"
SPLITS = {"train": ("2026-06-01", "2026-08-03"), "test": ("2026-08-03", "2026-09-07"),
          "holdout": ("2026-09-07", "2026-10-04")}
OUT = ROOT / "results" / "holdout_api.json"


def export() -> None:
    df = load_market("15m")
    df["split"] = None
    for k, (a, b) in SPLITS.items():
        df.loc[(df.decision_ts >= a) & (df.decision_ts < b), "split"] = k
    out = df[["decision_ts", "split", "market", "bid", "ask"] + FEATURES + ["label"]].rename(columns={"label": "target"})
    out["decision_ts"] = out["decision_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CSV, index=False, float_format="%.6g")
    print("wrote", CSV, out.shape, out.split.value_counts().to_dict())


def _trade_frame(df: pd.DataFrame) -> pd.DataFrame:
    t = df.rename(columns={"target": "label"}).copy()
    t["decision_ts"] = pd.to_datetime(t["decision_ts"], utc=True)
    t["day"] = t["decision_ts"].dt.floor("D")
    return t


def trade(test: pd.DataFrame, hold: pd.DataFrame, p_test, p_hold) -> dict:
    """Pick the margin on test (raw probabilities), then trade the holdout once."""
    best = None
    for m in MARGINS_C:
        tv = _trades(test, np.asarray(p_test), m)
        tot = tv["net_c"].sum() if len(tv) >= 20 else -np.inf
        if best is None or tot > best[0]:
            best = (tot, m)
    th = _trades(hold, np.asarray(p_hold), best[1])
    if th.empty:
        return {"margin_c": best[1], "trades": 0}
    ci = boot_mean(th["net_c"].to_numpy(float), th["day"])
    return {"margin_c": best[1], "trades": int(len(th)), "c_per_ct": round(ci["mean"], 2),
            "ci95": [round(ci["lo"], 2), round(ci["hi"], 2)], "win_rate": round(float((th.net_c > 0).mean()), 3)}


def run() -> None:
    import tabpfn_client
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from tabpfn_client import TabPFNClassifier

    token = os.environ.get("TABPFN_TOKEN")
    if not token:
        sys.exit("set TABPFN_TOKEN (https://ux.priorlabs.ai)")
    tabpfn_client.set_access_token(token)

    df = pd.read_csv(CSV)
    tr, te, ho = (df[df.split == s] for s in ("train", "test", "holdout"))
    X = lambda d: d[FEATURES]  # noqa: E731
    out = {"splits": {s: {"rows": int((df.split == s).sum()), "from": SPLITS[s][0], "to": SPLITS[s][1]}
                      for s in SPLITS}, "models": {}}
    models = {
        "TabPFN-3.5 (API)": lambda: TabPFNClassifier(model_path="v3.5_default", n_estimators=8),
        "Logistic regression": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                     LogisticRegression(max_iter=2000)),
    }
    preds = {}
    for name, make in models.items():
        t0 = time.perf_counter()
        m = make()
        m.fit(X(tr), tr["target"])
        p_te = np.asarray(m.predict_proba(X(te)))[:, 1]
        p_ho = np.asarray(m.predict_proba(X(ho)))[:, 1]
        preds[name] = (p_te, p_ho)
        print(f"{name}: {time.perf_counter() - t0:.1f}s", flush=True)
    preds["Market price"] = (te["p_mid"].to_numpy(), ho["p_mid"].to_numpy())
    tt, th = _trade_frame(te), _trade_frame(ho)
    for name, (p_te, p_ho) in preds.items():
        out["models"][name] = {"test": metrics(te["target"], p_te), "holdout": metrics(ho["target"], p_ho),
                               "trading_holdout": trade(tt, th, p_te, p_ho)}
    OUT.write_text(json.dumps(out, indent=1))
    pd.DataFrame({"split": ["test"] * len(te) + ["holdout"] * len(ho),
                  **{f"p__{k}": np.concatenate(v) for k, v in preds.items()},
                  "target": np.concatenate([te.target, ho.target])}).to_parquet(
        ROOT / "results" / "holdout_api_preds.parquet", index=False)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"export": export, "run": run}[sys.argv[1]]()
