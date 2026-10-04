"""Summarise ``make quick``: one test week per market, every model, on CPU."""

from __future__ import annotations

import pandas as pd

from .data import MARKETS, load_market
from .evaluate import load_meta, load_preds, metrics
from .run import expand


def main() -> None:
    for market in MARKETS:
        df = load_market(market)
        rows = []
        for m in expand("market,classic_default,tabpfn"):
            pr = load_preds(market, "n5000", m, sub="quick")
            if pr is None:
                continue
            d = df.merge(pr[pr.split == "test"][["row_id", "p"]], on="row_id")
            meta = load_meta(market, "n5000", m, sub="quick")[0]
            rows.append({"model": m, **metrics(d.label, d.p), "fit_s": round(meta["fit_s"], 1)})
        if rows:
            print(f"\n{MARKETS[market].title} — last test week")
            print(pd.DataFrame(rows).set_index("model").sort_values("logloss").round(4).to_string())


if __name__ == "__main__":
    main()
