"""Turn cached predictions into ``results/summary.json`` (everything the report shows).

    python -m tabtrader.analyze
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import MARKETS, ROOT, blocks, load_market, split_block
from .evaluate import (bh, boot_mean, load_meta, load_preds, max_drawdown, metrics, reliability,
                       row_logloss, trade_block)
from .models import CLASSIC

REF = "TabPFN-3.5"
MAIN = (["Market price"] + [f"{c} · tuned" for c in CLASSIC] + ["TabPFN-3.5", "TabPFN-3.5-Fast"])
LC_SIZES = ["250", "500", "1000", "2000", "5000"]
LC_MODELS = ["Market price"] + [f"{c} · default" for c in CLASSIC] + ["TabPFN-3.5", "TabPFN-3.5-Fast"]


def _r(x, k=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def joined(df: pd.DataFrame, preds: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    p = preds[preds.split == split][["row_id", "p", "block"]]
    return df.merge(p, on="row_id", how="inner", validate="one_to_one")


def cpu(meta: list[dict]) -> dict:
    if not meta:
        return {}
    f = [m.get("fit_s", 0) for m in meta]
    s = [m.get("search_s", 0) for m in meta]
    p = [m.get("predict_s", 0) / max(m.get("predict_rows", 1), 1) * 1000 for m in meta]
    lat = [m["single_row_s"] for m in meta if "single_row_s" in m]
    return {"fit_s_mean": _r(np.mean(f), 2), "search_s_mean": _r(np.mean(s), 2),
            "predict_ms_per_row": _r(np.mean(p), 3), "single_market_s": _r(lat[0], 3) if lat else None,
            "n_fit_mean": _r(np.mean([m["n_fit"] for m in meta]), 0)}


def leaderboard(df, n_tag: str, models: list[str], market: str, ref: str = REF,
                fixed: dict[str, str] | None = None) -> dict:
    """``fixed`` maps a model to the n_tag it is always read from (TabPFN/market stay at n5000)."""
    fixed = fixed or {}
    tags = {m: fixed.get(m, n_tag) for m in models}
    rows, base = {}, {}
    for m in models:
        pr = load_preds(market, tags[m], m)
        if pr is not None:
            base[m] = joined(df, pr)
    if ref not in base:
        ref = "Market price"
    common = set.intersection(*(set(v.row_id) for v in base.values()))
    ref_df = base[ref].set_index("row_id").loc[sorted(common)]
    y = ref_df["label"].to_numpy()
    ll_ref = row_logloss(y, ref_df["p"].to_numpy())
    mkt = base["Market price"].set_index("row_id").loc[sorted(common)]
    ll_mkt = row_logloss(y, mkt["p"].to_numpy())
    days = ref_df["day"]
    pvals, keys = [], []
    for m, d in base.items():
        d = d.set_index("row_id").loc[sorted(common)]
        p = d["p"].to_numpy()
        ll = row_logloss(y, p)
        r = metrics(y, p)
        r["vs_ref"] = boot_mean(ll - ll_ref, days) if m != ref else None
        r["vs_market"] = boot_mean(ll - ll_mkt, days) if m != "Market price" else None
        r["cpu"] = cpu(load_meta(market, tags[m], m))
        r["per_block_logloss"] = {int(b): _r(row_logloss(g.label.to_numpy(), g.p.to_numpy()).mean())
                                  for b, g in d.groupby("block")}
        r["reliability"] = reliability(y, p)
        rows[m] = r
        if m != ref:
            pvals.append(r["vs_ref"]["p"])
            keys.append(m)
    for k, passed in zip(keys, bh(pvals, 0.10)):
        rows[k]["vs_ref"]["bh_pass"] = passed
    return {"ref": ref, "n_rows": len(common), "n_days": int(days.nunique()), "models": rows}


def learning_curve(df, market: str) -> dict:
    out = {}
    for n in LC_SIZES:
        tag = f"n{n}"
        out[n] = {}
        for m in LC_MODELS:
            pr = load_preds(market, tag, m)
            if pr is None:
                continue
            d = joined(df, pr)
            out[n][m] = {"logloss": _r(row_logloss(d.label.to_numpy(), d.p.to_numpy()).mean()),
                         "n_test": int(len(d))}
    return out


def trading(df, market: str, n_tag: str, models: list[str]) -> dict:
    spec = MARKETS[market]
    out = {}
    for m in models:
        pr = load_preds(market, n_tag, m)
        if pr is None:
            continue
        allt, margins = [], []
        for i, (a, b) in enumerate(blocks(spec)):
            _, val, test = split_block(df, spec, a, b, 5000)
            pv = pr[(pr.block == i) & (pr.split == "val")].set_index("row_id").p
            pt = pr[(pr.block == i) & (pr.split == "test")].set_index("row_id").p
            if pt.empty:
                continue
            val = val[val.row_id.isin(pv.index)]
            test = test[test.row_id.isin(pt.index)]
            t, info = trade_block(val, test, pv.loc[val.row_id].to_numpy(), pt.loc[test.row_id].to_numpy())
            margins.append(info["margin_c"])
            allt.append(t)
        t = pd.concat(allt, ignore_index=True) if allt else pd.DataFrame()
        if t.empty:
            out[m] = {"trades": 0}
            continue
        t = t.sort_values("decision_ts")
        daily = t.groupby("day")["net_c"].sum()
        cum = daily.cumsum()
        ci = boot_mean(t["net_c"].to_numpy(float), t["day"])
        out[m] = {"trades": int(len(t)), "c_per_ct": _r(ci["mean"], 2), "ci95": [_r(ci["lo"], 2), _r(ci["hi"], 2)],
                  "p": _r(ci["p"], 4), "win_rate": _r((t.net_c > 0).mean(), 3),
                  "avg_price_c": _r(t.price_c.mean(), 1), "total_c_1ct": _r(t.net_c.sum(), 1),
                  "max_dd_c_1ct": _r(max_drawdown(cum.to_numpy()), 1), "margins_c": margins,
                  "cum": [[str(k.date()), _r(v, 1)] for k, v in cum.items()]}
    return out


def thinking(df, market: str) -> dict | None:
    pr = load_preds(market, "n5000", "TabPFN-3.5-Thinking")
    if pr is None:
        return None
    blocks_done = sorted(pr.block.unique().tolist())
    res = {"blocks": blocks_done}
    sub = {}
    for m in ["Market price", "TabPFN-3.5", "TabPFN-3.5-Fast", "TabPFN-3.5-Thinking", "LightGBM · tuned",
              "CatBoost · tuned", "Logistic regression · tuned"]:
        p = load_preds(market, "n5000", m)
        if p is None:
            continue
        p = p[p.block.isin(blocks_done)]
        sub[m] = joined(df, p).set_index("row_id")
    common = sorted(set.intersection(*(set(v.index) for v in sub.values())))
    y = sub["Market price"].loc[common].label.to_numpy()
    days = sub["Market price"].loc[common].day
    ll_t = row_logloss(y, sub["TabPFN-3.5-Thinking"].loc[common].p.to_numpy())
    for m, d in sub.items():
        p = d.loc[common].p.to_numpy()
        r = metrics(y, p)
        if m != "TabPFN-3.5-Thinking":
            r["thinking_minus_model"] = boot_mean(ll_t - row_logloss(y, p), days)
        res[m] = r
    res["cpu"] = cpu(load_meta(market, "n5000", "TabPFN-3.5-Thinking"))
    return res


def data_summary(market: str) -> dict:
    spec = MARKETS[market]
    raw = load_market(market, clean=False)
    from .data import clean_mask

    cm = clean_mask(raw, spec)
    return {"snapshots": int(len(raw)), "clean": int(cm.sum()),
            "status_counts": raw.candle_status.value_counts().to_dict(),
            "first": str(raw.decision_ts.min()), "last": str(raw.decision_ts.max()),
            "label_rate_clean": _r(raw[cm].label.mean(), 4)}


def main() -> None:
    out = {}
    for market in MARKETS:
        df = load_market(market)
        print("analyze", market, flush=True)
        out[market] = {
            "data": data_summary(market),
            "policy_A": leaderboard(df, "n5000", MAIN, market),
            "policy_B": leaderboard(df, "nall", ["Market price"] + [f"{c} · tuned" for c in CLASSIC] + ["TabPFN-3.5"],
                                    market, fixed={"Market price": "n5000", "TabPFN-3.5": "n5000"})
            if load_preds(market, "nall", "LightGBM · tuned") is not None else None,
            "learning_curve": learning_curve(df, market),
            "trading": trading(df, market, "n5000", MAIN),
            "thinking": thinking(df, market),
        }
    # policy B references TabPFN-3.5 / market predictions from n5000
    path = ROOT / "results" / "summary.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print("wrote", path)


if __name__ == "__main__":
    main()
