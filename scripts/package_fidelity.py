"""Section 4: how closely the author's backtest matches live trading (aggregates only).

Inputs are private (live journal, paper-twin journal, Kalshi fills, backtest signals).
Output ``data/fidelity/fidelity.json`` holds rates and cents-per-contract only:
no dollar amounts, no position sizes, no strategy parameters.

    python scripts/package_fidelity.py --src <private artifacts dir> --twin <private instance name>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def boot_ci(x: np.ndarray, days: np.ndarray, reps: int = 4000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    u = np.unique(days)
    pos = {d: np.where(days == d)[0] for d in u}
    bs = [x[np.concatenate([pos[d] for d in rng.choice(u, len(u))])].mean() for _ in range(reps)]
    return [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--twin", required=True, help="private instance name of the paper twin")
    args = ap.parse_args()
    src = Path(args.src)
    live = pd.read_parquet(src / "live.parquet")
    sig = pd.read_parquet(src / "signals.parquet")
    bt = pd.read_parquet(src / "dl_bundle" / "signals.parquet",
                         columns=["market", "entry_ts", "side", "px", "won", "ex_net_c", "hold_net_c"])
    out: dict = {"period": [str(live.entry_ts.min().date()), str(live.entry_ts.max().date())],
                 "live_trades": int(len(live)), "live_days": int(live.day.nunique())}

    # 1. live vs paper twin (same code path, 1 contract, paper fills at the quote)
    twin = sig[(sig.instance == args.twin) & (sig.taken == 1)].drop_duplicates("market")
    m = live.merge(twin[["market", "side", "won", "net_per_ct", "exit_reason"]],
                   on="market", how="left", suffixes=("", "_twin"))
    paired = m.side_twin.notna()
    out["vs_paper_twin"] = {
        "live_trades_also_taken_by_twin": round(float(paired.mean()), 4),
        "same_side": round(float((m.side == m.side_twin)[paired].mean()), 4),
        "same_outcome": round(float((m.won_settle == m.won)[paired & m.won.notna()].mean()), 4),
        "same_exit_reason": round(float((m.exit_reason == m.exit_reason_twin)[paired].mean()), 4),
        "n_paired": int(paired.sum()),
    }

    # 2. live vs backtest replay of the same rule on recorded data
    b = live.merge(bt.rename(columns={"side": "side_bt", "entry_ts": "entry_ts_bt"}),
                   on="market", how="left")
    has = b.side_bt.notna()
    same = has & (b.side == b.side_bt)
    dt = (b.entry_ts - b.entry_ts_bt).dt.total_seconds().abs()
    out["vs_backtest"] = {
        "live_trades_with_backtest_signal": round(float(has.mean()), 4),
        "same_side": round(float(same[has].mean()), 4),
        "entry_time_diff_s_median": round(float(dt[same].median()), 2),
        "entry_time_within_5s": round(float((dt[same] <= 5).mean()), 4),
        "price_diff_c_mean": round(float((b.entry_price_c - b.px)[same].mean()), 2),
        "price_diff_c_median": round(float((b.entry_price_c - b.px)[same].median()), 2),
        "price_within_1c": round(float(((b.entry_price_c - b.px).abs() <= 1)[same].mean()), 4),
        "same_outcome_settled": round(float((b.won_settle == b.won)[same & b.won.notna() & b.won_settle.notna()].mean()), 4),
        "n_settled": int((same & b.won.notna() & b.won_settle.notna()).sum()),
        "n_matched": int(same.sum()),
    }

    # 3. execution: real Kalshi fills vs the quote the strategy saw
    f = live[live.fill_entry_c.notna()]
    slip = f.fill_entry_c - f.entry_price_c
    out["execution"] = {
        "n_with_fills": int(len(f)),
        "entry_slippage_c_mean": round(float(slip.mean()), 2),
        "entry_slippage_c_median": round(float(slip.median()), 2),
        "entry_slippage_within_1c": round(float((slip.abs() <= 1).mean()), 4),
        "journal_fee_c_per_ct": round(float((f.fees_c / f.contracts).mean()), 2),
        "kalshi_fee_c_per_ct": round(float((f.fill_fee_c / f.fill_ct_total).mean()), 2)
        if "fill_ct_total" in f else round(float((f.fill_fee_c / f.fill_entry_ct).mean()), 2),
    }

    # 4. cents per contract on the SAME trades, four ways
    same_rows = b[same].drop_duplicates("market")
    days = same_rows.day.astype(str).to_numpy()
    rows = {
        "backtest (recorded quotes, live exits)": same_rows.ex_net_c.to_numpy(float),
        "paper twin (live code, paper fills)": m[paired].net_per_ct.to_numpy(float),
        "live journal (quoted prices)": same_rows.net_per_ct.to_numpy(float),
        "live real (Kalshi fills and fees)": same_rows.real_net_per_ct.to_numpy(float),
    }
    cpc = {}
    for k, x in rows.items():
        d = days if len(x) == len(days) else m[paired].day.astype(str).to_numpy()
        ok = ~np.isnan(x)
        cpc[k] = {"n": int(ok.sum()), "c_per_ct": round(float(x[ok].mean()), 2), "ci95": boot_ci(x[ok], d[ok])}
    out["cents_per_contract_same_trades"] = cpc
    dst = ROOT / "data" / "fidelity" / "fidelity.json"
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
