"""Package section 6 (TabPFN inside the author's live strategy) for publication.

Input: the private research artifacts (per-signal out-of-sample predictions from the
Oct 2026 gate study). Output: ``data/strategy/oos_predictions.parquet`` with ONLY
outcomes and model scores — no features, no timestamps finer than a day, no market
tickers, no trade direction, no thresholds or exit rules.

    python scripts/package_strategy.py --src <private artifacts dir> --live-threshold <private>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Private model keys -> published names.
MODELS = {
    "PRICE": "price", "LR": "logistic", "ENS": "ensemble_lr_lgbm_xgb", "A3_MLP": "mlp",
    "A1_ARTICLE": "lstm_btc_bars", "A2_ARTICLE_STATE": "lstm_bars_plus_state", "STACK": "stack",
    "TABPFN_GATE": "tabpfn", "TABPFN_PLUS": "tabpfn_plus_momentum",
}
LAYERS = {"L0_threshold": "threshold", "L1_calibrated_edge": "edge"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--live-threshold", type=float, required=True,
                    help="the live gate's private threshold; used to mark live trades, never written out")
    args = ap.parse_args()
    src = Path(args.src)
    s8 = pd.read_parquet(src / "s8_test_preds.parquet")
    s9 = pd.read_parquet(src / "s9_test_preds.parquet")
    bundle = pd.read_parquet(src / "dl_bundle" / "signals.parquet",
                             columns=["market", "ex_net_c", "p_deployed", "day"])
    assert (s8["market"].to_numpy() == s9["market"].to_numpy()).all()
    s8 = s8.merge(bundle, on="market", how="left", validate="one_to_one")
    live_threshold = args.live_threshold  # applied here, never published

    out = pd.DataFrame({
        "day": pd.to_datetime(s8["day"]).dt.strftime("%Y-%m-%d"),
        "block": s8["block"].astype(int),
        "won": s8["won"].astype(int),
        "net_c_hold": s8["hold_net_c"].round(3),
        "net_c_exits": s8["ex_net_c"].round(3),
        "take__live": s8["p_deployed"].to_numpy() >= live_threshold,
        # Rank-transformed so the published score cannot reveal the live threshold (AUC is unchanged).
        "p__live_gate": s8["p_deployed"].rank(pct=True).round(5),
    })
    for k, name in MODELS.items():
        src_df = s9 if k.startswith("TABPFN") else s8
        out[f"p__{name}"] = src_df[f"p__{k}"].round(5).to_numpy()
        for lk, lname in LAYERS.items():
            col = f"take__{k}__{lk}"
            if col in src_df:
                out[f"take__{name}__{lname}"] = src_df[col].astype(bool).to_numpy()
    # Shuffle rows within day so row order carries no intraday timing.
    rng = np.random.default_rng(0)
    out = out.assign(_r=rng.random(len(out))).sort_values(["day", "_r"]).drop(columns="_r")
    dst = ROOT / "data" / "strategy" / "oos_predictions.parquet"
    out.reset_index(drop=True).to_parquet(dst, index=False)
    print(dst, out.shape, "days", out.day.nunique())


if __name__ == "__main__":
    main()
