"""Harness controls (15-minute market, test blocks 0, 4, 8).

* planted : the weakest feature (hour_cos) is overwritten with a noisy copy of the true residual
            (label - market price), correlation ~0.3. Every model should now beat the market.
* null    : labels are redrawn as Bernoulli(market mid), so the market is calibrated by
            construction and nothing can beat it. A harness that "finds" skill here is broken.

    python -m tabtrader.controls
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import MARKETS, ROOT, blocks, load_market, split_block
from .evaluate import boot_mean, row_logloss
from .run import run_one, slug

MODELS = ["Market price", "Logistic regression · default", "LightGBM · default", "TabPFN-3.5"]
BLOCKS = [0, 4, 8]


def make(df: pd.DataFrame, kind: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    if kind == "null":
        df["label"] = rng.binomial(1, df["p_mid"].clip(0.001, 0.999))
    elif kind == "planted":
        resid = df["label"] - df["p_mid"]
        noise = rng.normal(0, resid.std() * 3.2, len(df))
        df["hour_cos"] = resid + noise
    return df


def main() -> None:
    spec = MARKETS["15m"]
    base = load_market("15m")
    out = {}
    for kind in ["planted", "null"]:
        df = make(base, kind)
        if kind == "planted":
            out["planted_corr_with_residual"] = round(float(np.corrcoef(df.hour_cos, df.label - df.p_mid)[0, 1]), 3)
        preds = {}
        for m in MODELS:
            parts = []
            for i in BLOCKS:
                path = ROOT / "results" / "controls" / kind / slug(m) / f"b{i}.parquet"
                if path.exists():
                    parts.append(pd.read_parquet(path))
                    continue
                a, b = blocks(spec)[i]
                fit, val, test = split_block(df, spec, a, b, 5000)
                p, meta = run_one(m, fit, val, test)
                p["block"] = i
                path.parent.mkdir(parents=True, exist_ok=True)
                p.to_parquet(path, index=False)
                parts.append(p)
                print(kind, m, i, flush=True)
            pr = pd.concat(parts)
            preds[m] = df.merge(pr[pr.split == "test"][["row_id", "p"]], on="row_id")
        mk = preds["Market price"]
        ll_m = row_logloss(mk.label.to_numpy(), mk.p.to_numpy())
        res = {}
        for m, d in preds.items():
            d = d.set_index("row_id").loc[mk.row_id]
            ll = row_logloss(d.label.to_numpy(), d.p.to_numpy())
            res[m] = {"logloss": round(float(ll.mean()), 5),
                      "minus_market": None if m == "Market price" else boot_mean(ll - ll_m, mk.day)}
        out[kind] = res
    (ROOT / "results" / "controls.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
