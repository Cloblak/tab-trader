"""EDA helpers shared by notebook 02 and the report (all inputs are in data/)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DATA, MARKETS, load_market

MILESTONES = [
    ("2026-03-21", "15-minute tape + BTC spot (≈4 rows/s)"),
    ("2026-05-01", "Settlement labels retained (earlier ones were purged by the exchange API)"),
    ("2026-06-07", "Exchange 1-minute candles available to cross-check every quote"),
    ("2026-06-30", "Full L2 order book + per-venue spot feeds"),
    ("2026-07-05", "Hourly strike-ladder tape"),
    ("2026-07-30", "Collector v2: exact deci-cent quotes, crossed books eliminated"),
    ("2026-08-02", "CF Benchmarks BRTI index (the settlement source) recorded at 1 s"),
]


def eda_tables() -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_parquet(p) for p in sorted((DATA / "eda").glob("*.parquet"))}


def volume_totals(t: dict[str, pd.DataFrame]) -> dict:
    o = t["other_tables_daily"].groupby("table")["rows"].sum().to_dict()
    lab = t["labels_daily"].groupby("series")["markets"].sum().to_dict()
    return {
        "tape_15m_rows": int(t["tape_15m_daily"]["rows"].sum()),
        "tape_15m_markets": int(t["tape_15m_daily"]["markets"].sum()),
        "tape_1h_rows": int(t["tape_1h_daily"]["rows"].sum()),
        "order_book_rows": int(o.get("order_book_l2", 0)),
        "spot_venue_rows": int(o.get("spot_venue_events", 0)),
        "indicator_rows": int(o.get("indicators", 0)),
        "brti_rows": int(o.get("brti_index", 0)),
        "labels_15m": int(lab.get("KXBTC15M", 0)),
        "labels_1h_all_legs": int(lab.get("KXBTCD", 0)),
        "first_day": str(t["tape_15m_daily"]["day"].min().date()),
        "last_day": str(t["tape_15m_daily"]["day"].max().date()),
    }


def weekly_quality(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = []
    for key, name in [("tape_15m_daily", "15-minute"), ("tape_1h_daily", "hourly")]:
        d = t[key].copy()
        d["week"] = d["day"].dt.to_period("W-SUN").dt.start_time
        g = d.groupby("week")[["rows", "valid", "crossed"]].sum()
        g["valid_pct"] = 100 * g["valid"] / g["rows"]
        g["crossed_pct"] = 100 * g["crossed"] / g["rows"]
        g["market"] = name
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def candle_agreement(key: str) -> pd.DataFrame:
    """Share of decision quotes confirmed by the exchange's own candles, by month."""
    raw = load_market(key, clean=False)
    raw["month"] = raw["decision_ts"].dt.strftime("%Y-%m")
    raw["confirmed"] = (raw["candle_status"] == "ok") & (raw["candle_close_diff_c"] <= 2)
    has = raw["candle_status"].isin(["ok", "mismatch"])
    g = raw[has].groupby("month").agg(checked=("confirmed", "size"), confirmed=("confirmed", "sum"),
                                      median_diff_c=("candle_close_diff_c", "median"))
    g["confirmed_pct"] = 100 * g["confirmed"] / g["checked"]
    return g.reset_index()


def cleaning_funnel(key: str) -> list[tuple[str, int]]:
    spec = MARKETS[key]
    raw = load_market(key, clean=False)
    steps = [("settled markets × decision times", len(raw))]
    m = raw["bid"].notna()
    steps.append(("valid two-sided quote ≤ 5 s old", int(m.sum())))
    m &= raw["candle_status"] == "ok"
    steps.append(("inside the exchange candle's range", int(m.sum())))
    m &= raw["candle_close_diff_c"] <= 2
    steps.append(("within 2¢ of the exchange close", int(m.sum())))
    m &= raw["spot_ok"]
    steps.append(("BTC bar present", int(m.sum())))
    if spec.mid_range:
        mid = (raw.bid + raw.ask) / 2
        m &= mid.between(*spec.mid_range)
        steps.append((f"mid between {spec.mid_range[0]:.0f}¢ and {spec.mid_range[1]:.0f}¢", int(m.sum())))
    return steps


def market_calibration(key: str, bins: int = 10) -> pd.DataFrame:
    """How well does the market's own mid price predict settlement? (clean rows)"""
    df = load_market(key)
    df["bin"] = np.minimum((df["p_mid"] * bins).astype(int), bins - 1)
    g = df.groupby("bin").agg(p=("p_mid", "mean"), y=("label", "mean"), n=("label", "size"))
    se = np.sqrt(g["y"] * (1 - g["y"]) / g["n"])
    g["lo"], g["hi"] = g["y"] - 1.96 * se, g["y"] + 1.96 * se
    return g.reset_index()
