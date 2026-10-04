"""Walk-forward runner. Caches every prediction so runs are resumable.

    python -m tabtrader.run --market 15m --n 5000 --models market,classic_tuned
    python -m tabtrader.run --market 15m --n 5000 --models tabpfn
    python -m tabtrader.run --market 1h  --n all  --models classic_tuned
    python -m tabtrader.run --market 15m --n 1000 --models classic_default,tabpfn

Output: ``results/preds/<market>/n<N>/<model>/b<block>.parquet`` (row_id, split, p)
plus a ``.json`` with timings and chosen hyper-parameters.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .data import MARKETS, ROOT, blocks, load_market, split_block, xy
from .features import FEATURES
from .models import (CLASSIC, MarketPrice, fit_classic, fit_timed, make_tabpfn,
                     make_tabpfn_thinking, timed_predict)

warnings.filterwarnings("ignore")
PRED_DIR = ROOT / "results" / "preds"
P_MID = FEATURES.index("p_mid")

TABPFN_MODELS = {"TabPFN-3.5": "3.5", "TabPFN-3.5-Fast": "3.5-fast"}


def slug(name: str) -> str:
    return name.replace(" ", "_").replace("·", "-").replace("/", "-")


def out_path(market: str, n_tag: str, model: str, block: int, sub: str = "preds") -> Path:
    return ROOT / "results" / sub / market / n_tag / slug(model) / f"b{block}.parquet"


def expand(models: str) -> list[str]:
    out = []
    for m in models.split(","):
        if m == "market":
            out.append("Market price")
        elif m == "classic_tuned":
            out += [f"{c} · tuned" for c in CLASSIC]
        elif m == "classic_default":
            out += [f"{c} · default" for c in CLASSIC]
        elif m == "tabpfn":
            out += list(TABPFN_MODELS)
        elif m == "tabpfn35":
            out.append("TabPFN-3.5")
        elif m == "tabpfn_fast":
            out.append("TabPFN-3.5-Fast")
        elif m == "thinking":
            out.append("TabPFN-3.5-Thinking")
        else:
            out.append(m)
    return out


def _with_time(df: pd.DataFrame) -> pd.DataFrame:
    """Feature frame plus the decision time, for Thinking mode's time-aware fitting."""
    X = df[FEATURES].astype(float).copy()
    X["decision_epoch_s"] = (
        df["decision_ts"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    return X.reset_index(drop=True)


def run_one(model: str, fit, val, test, latency: bool = False, transform=None) -> tuple[pd.DataFrame, dict]:
    X_fit, y_fit = xy(fit)
    X_val, y_val = xy(val)
    X_te, _ = xy(test)
    if model == "TabPFN-3.5-Thinking":  # API model: pass named columns + the time column
        X_fit, X_val, X_te = _with_time(fit), _with_time(val), _with_time(test)
    if transform is not None:
        X_fit, y_fit, X_val, y_val, X_te = transform(X_fit, y_fit, X_val, y_val, X_te)
    meta: dict = {"model": model, "n_fit": int(len(fit)), "n_val": int(len(val)), "n_test": int(len(test))}

    if model == "Market price":
        res = fit_timed(MarketPrice(P_MID), X_fit, y_fit)
    elif model.endswith(" · tuned") or model.endswith(" · default"):
        name, mode = model.split(" · ")
        res = fit_classic(name, X_fit, y_fit, X_val, y_val, tune=(mode == "tuned"))
        meta["params"] = res.params
        meta["search_s"] = res.search_s
    elif model in TABPFN_MODELS:
        res = fit_timed(make_tabpfn(TABPFN_MODELS[model]), X_fit, y_fit)
    elif model == "TabPFN-3.5-Thinking":
        res = fit_timed(make_tabpfn_thinking(thinking_effort="high", time_col="decision_epoch_s"),
                        X_fit, y_fit)
    else:
        raise ValueError(model)

    p_val = timed_predict(res, X_val)
    p_te = timed_predict(res, X_te)
    meta.update(fit_s=res.fit_s, predict_s=res.predict_s,
                predict_rows=int(len(X_val) + len(X_te)))
    if latency and model != "TabPFN-3.5-Thinking":  # one market, one decision: the live question
        t0 = time.perf_counter()
        for i in range(5):
            res.model.predict_proba(X_te[i:i + 1])
        meta["single_row_s"] = (time.perf_counter() - t0) / 5
    preds = pd.concat([
        pd.DataFrame({"row_id": val["row_id"].values, "split": "val", "p": p_val}),
        pd.DataFrame({"row_id": test["row_id"].values, "split": "test", "p": p_te}),
    ], ignore_index=True)
    return preds, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, choices=list(MARKETS))
    ap.add_argument("--n", default="5000", help="fit-set size, or 'all'")
    ap.add_argument("--models", default="market,classic_tuned,tabpfn")
    ap.add_argument("--blocks", default="all", help="'all' or comma list of block indices")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sub", default="preds", help="results sub-folder (e.g. 'quick')")
    args = ap.parse_args()

    spec = MARKETS[args.market]
    df = load_market(args.market)
    n_fit = None if args.n == "all" else int(args.n)
    n_tag = f"n{args.n}"
    bl = blocks(spec)
    idx = range(len(bl)) if args.blocks == "all" else [int(i) for i in args.blocks.split(",")]
    for model in expand(args.models):
        for i in idx:
            a, b = bl[i]
            path = out_path(args.market, n_tag, model, i, sub=args.sub)
            if path.exists() and not args.force:
                continue
            fit, val, test = split_block(df, spec, a, b, n_fit)
            if len(test) == 0:
                continue
            t0 = time.perf_counter()
            preds, meta = run_one(model, fit, val, test, latency=(i == idx[0]))
            meta.update(block=i, block_start=str(a), block_end=str(b),
                        fit_from=str(fit["decision_ts"].min()), fit_to=str(fit["decision_ts"].max()),
                        val_to=str(val["decision_ts"].max()), wall_s=time.perf_counter() - t0)
            path.parent.mkdir(parents=True, exist_ok=True)
            preds.to_parquet(path, index=False)
            path.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
            print(f"{args.market} {n_tag} {model:28s} b{i} fit={meta['fit_s']:.1f}s "
                  f"pred={meta['predict_s']:.1f}s n_fit={meta['n_fit']} n_test={meta['n_test']}", flush=True)


if __name__ == "__main__":
    main()
