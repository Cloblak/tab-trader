"""Build the published datasets in ``data/`` from the private ClickHouse store.

This script documents provenance; it cannot run without access to the
author's database. Everything it writes is committed to the repo, so nothing
downstream needs it.

Connection comes from the environment only (no defaults are baked in)::

    CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER, CLICKHOUSE_PASSWORD, CLICKHOUSE_DB

or ``--env-file path/to/.env`` with the same keys (``LOCAL_CLICKHOUSE_*`` also accepted).

Data rules applied here (each one fixes a contaminant found in earlier research):

* Quotes are taken at a single instant: the last tape row at or before the
  decision time, at most 5 s old. Never a window mean.
* Only rows the collector marked ``quote_valid = 1`` with ``ask > bid``.
* Every decision quote is cross-checked against the exchange's own 1-minute
  candlestick for that minute; the result is stored in ``candle_status``.
* ``SETTINGS optimize_move_to_prewhere = 0`` on every quote query
  (ClickHouse 26.6 silently returned wrong rows for bid/ask predicates without it).

Usage::

    python scripts/export_from_clickhouse.py --env-file path/to/private/.env all
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UTC = timezone.utc
NO_PREWHERE = " SETTINGS optimize_move_to_prewhere = 0"

# Collection milestones (UTC).
TAPE_START = datetime(2026, 3, 21, tzinfo=UTC)
LABELS_15M_START = datetime(2026, 5, 1, tzinfo=UTC)
HOURLY_START = datetime(2026, 7, 5, tzinfo=UTC)
END = datetime(2026, 10, 4, tzinfo=UTC)  # exclusive
RAW_WEEK = (datetime(2026, 9, 26, tzinfo=UTC), datetime(2026, 10, 3, tzinfo=UTC))

# Known collector outages (quotes 100% NULL while spot stayed healthy).
OUTAGES = [
    (datetime(2026, 7, 9, 9, 0, tzinfo=UTC), datetime(2026, 7, 10, 1, 45, tzinfo=UTC)),
    (datetime(2026, 7, 11, 12, 0, tzinfo=UTC), datetime(2026, 7, 11, 23, 0, tzinfo=UTC)),
]

# Decision times, in seconds before close, and the look-back offsets used for
# contract-momentum features (quote 60 s and 180 s before the decision).
TAUS_15M = (600, 300)
TAUS_1H = (1800, 900)
LAGS = (0, 60, 180)
MAX_AGE_S = 5


# ---------------------------------------------------------------- connection
def _load_env_file(path: str) -> None:
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _env(*names: str, default: str | None = None) -> str:
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    if default is None:
        sys.exit(f"missing environment variable: {names[0]}")
    return default


def client():
    from clickhouse_driver import Client

    return Client(
        host=_env("CLICKHOUSE_HOST", "LOCAL_CLICKHOUSE_HOST"),
        port=int(_env("CLICKHOUSE_PORT", "LOCAL_CLICKHOUSE_PORT", default="9000")),
        user=_env("CLICKHOUSE_USER", "LOCAL_CLICKHOUSE_USER"),
        password=_env("CLICKHOUSE_PASSWORD", "LOCAL_CLICKHOUSE_PASSWORD", default=""),
        database=_env("CLICKHOUSE_DB", "LOCAL_CLICKHOUSE_DB", default="market_data"),
        settings={"max_execution_time": 1800},
    )


def q(sql: str, params: dict | None = None, cols: list[str] | None = None) -> pd.DataFrame:
    rows = client().execute(sql, params or {})
    return pd.DataFrame(rows, columns=cols)


def _fmt(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _utc(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s)
    return out.dt.tz_localize("UTC") if out.dt.tz is None else out.dt.tz_convert("UTC")


def _chunks(start: datetime, end: datetime, days: int = 7):
    t = start
    while t < end:
        yield t, min(t + timedelta(days=days), end)
        t += timedelta(days=days)


# --------------------------------------------------------------- settlements
def settlements(series: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = q(
        "SELECT market, open_time, close_time, result, floor_strike "
        "FROM market_settlements FINAL "
        "WHERE series = %(s)s AND result IN ('yes','no') AND floor_strike IS NOT NULL "
        "  AND close_time >= %(a)s AND close_time < %(b)s ORDER BY close_time, market",
        {"s": series, "a": _fmt(start), "b": _fmt(end)},
        ["market", "open_time", "close_time", "result", "strike"],
    )
    for c in ("open_time", "close_time"):
        df[c] = _utc(df[c])
    if series == "KXBTCD":  # keep the 1-hour legs; the series also lists daily/weekly ones
        dur = (df.close_time - df.open_time).dt.total_seconds()
        df = df[(dur > 3000) & (dur < 4200)]
    df["label"] = (df.result == "yes").astype(np.int8)
    return df.drop(columns="result").reset_index(drop=True)


# ----------------------------------------------------------------- snapshots
def _offsets(taus) -> list[int]:
    return sorted({t + lag for t in taus for lag in LAGS})


def _tape_quotes(table: str, where: str, series: str, start: datetime, end: datetime,
                 offsets: list[int]) -> pd.DataFrame:
    """Last valid quote at or before each (close - offset), at most MAX_AGE_S old."""
    sql = f"""
    SELECT market, off,
           argMax(t.yes_best_bid, t.ts) AS bid,
           argMax(t.yes_best_ask, t.ts) AS ask,
           argMax(t.quote_src_ver, t.ts) AS src_ver,
           max(t.ts) AS quote_ts
    FROM (
        SELECT t.ts, t.market, t.yes_best_bid, t.yes_best_ask, t.quote_src_ver,
               dateDiff('millisecond', t.ts, s.close_time) AS d,
               arrayFirst(x -> d >= x * 1000 AND d < (x + {MAX_AGE_S}) * 1000,
                          {offsets}) AS off
        FROM {table} AS t
        INNER JOIN (
            SELECT market, close_time FROM market_settlements FINAL
            WHERE series = %(s)s AND close_time > %(a)s AND close_time <= %(b2)s
        ) AS s USING (market)
        WHERE {where} AND t.ts >= %(a)s AND t.ts < %(b)s
          AND t.quote_valid = 1 AND t.yes_best_ask > t.yes_best_bid
    ) AS t
    WHERE off > 0
    GROUP BY market, off
    {NO_PREWHERE}
    """
    df = q(sql, {"s": series, "a": _fmt(start), "b": _fmt(end),
                 "b2": _fmt(end + timedelta(hours=2))},
           ["market", "off", "bid", "ask", "src_ver", "quote_ts"])
    df["quote_ts"] = _utc(df["quote_ts"])
    return df


def _candles(series: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = q(
        """
        SELECT market, ts,
               argMax(yes_bid_low, fetched_at), argMax(yes_bid_high, fetched_at),
               argMax(yes_bid_close, fetched_at),
               argMax(yes_ask_low, fetched_at), argMax(yes_ask_high, fetched_at),
               argMax(yes_ask_close, fetched_at)
        FROM kalshi_candles_1m
        WHERE series = %(s)s AND ts >= %(a)s AND ts < %(b)s
        GROUP BY market, ts
        """,
        {"s": series, "a": _fmt(start), "b": _fmt(end + timedelta(hours=2))},
        ["market", "ts", "c_bid_lo", "c_bid_hi", "c_bid", "c_ask_lo", "c_ask_hi", "c_ask"],
    )
    df["ts"] = _utc(df["ts"])
    return df


def snapshots(series: str, table: str, where: str, taus, start: datetime, end: datetime) -> pd.DataFrame:
    """One row per (market, tau): decision quote, two lagged quotes, candle check, label."""
    st = settlements(series, start, end)
    offs = _offsets(taus)
    parts, cparts = [], []
    for a, b in _chunks(start, end):
        print(f"  {series} quotes {a:%m-%d}..{b:%m-%d}", flush=True)
        parts.append(_tape_quotes(table, where, series, a, b, offs))
        cparts.append(_candles(series, a, b))
    tape = pd.concat(parts, ignore_index=True)
    candles = pd.concat(cparts, ignore_index=True).drop_duplicates(["market", "ts"])

    rows = []
    for tau in taus:
        snap = st.copy()
        snap["tau_s"] = tau
        snap["decision_ts"] = snap.close_time - pd.to_timedelta(tau, unit="s")
        for lag in LAGS:
            sub = tape[tape.off == tau + lag][["market", "bid", "ask", "src_ver", "quote_ts"]]
            sfx = "" if lag == 0 else f"_lag{lag}"
            sub = sub.rename(columns={"bid": f"bid{sfx}", "ask": f"ask{sfx}",
                                      "src_ver": f"src_ver{sfx}", "quote_ts": f"quote_ts{sfx}"})
            snap = snap.merge(sub, on="market", how="left")
        rows.append(snap)
    snap = pd.concat(rows, ignore_index=True)
    snap = snap.drop(columns=[c for c in snap.columns if c.startswith(("src_ver_lag", "quote_ts_lag"))])
    snap["quote_age_s"] = (snap.decision_ts - snap.quote_ts).dt.total_seconds()

    # Exchange cross-check: candle whose bar END equals the decision instant.
    snap = snap.merge(candles.rename(columns={"ts": "decision_ts"}), on=["market", "decision_ts"], how="left")
    has_c = snap.c_bid.notna() & snap.c_ask.notna()
    in_rng = (snap.bid.between(snap.c_bid_lo - 0.5, snap.c_bid_hi + 0.5)
              & snap.ask.between(snap.c_ask_lo - 0.5, snap.c_ask_hi + 0.5))
    snap["candle_status"] = np.select(
        [snap.bid.isna(), ~has_c, in_rng], ["no_quote", "no_candle", "ok"], "mismatch")
    snap["candle_close_diff_c"] = ((snap.bid - snap.c_bid).abs() + (snap.ask - snap.c_ask).abs()) / 2
    out = snap.decision_ts.apply(lambda t: any(lo <= t < hi for lo, hi in OUTAGES))
    snap.loc[out & snap.bid.isna(), "candle_status"] = "outage"
    keep = ["market", "open_time", "close_time", "decision_ts", "tau_s", "strike", "label",
            "bid", "ask", "bid_lag60", "ask_lag60", "bid_lag180", "ask_lag180",
            "src_ver", "quote_age_s", "candle_status", "candle_close_diff_c"]
    return snap[keep].sort_values(["decision_ts", "market"]).reset_index(drop=True)


# ------------------------------------------------------------------ BTC bars
def btc_1m(start: datetime, end: datetime) -> pd.DataFrame:
    """1-minute BTC closes from the collector's composite spot (bar END timestamps)."""
    parts = []
    for a, b in _chunks(start, end, days=14):
        print(f"  btc_1m {a:%m-%d}..{b:%m-%d}", flush=True)
        parts.append(q(
            f"""
            SELECT toStartOfMinute(ts) + INTERVAL 1 MINUTE AS bar_end,
                   argMax(spot_close, ts) AS close, count() AS n_ticks
            FROM tick_15min
            WHERE asset = 'BTC' AND market LIKE 'KXBTC15M-%%'
              AND ts >= %(a)s AND ts < %(b)s AND spot_close > 0
            GROUP BY bar_end ORDER BY bar_end {NO_PREWHERE}
            """,
            {"a": _fmt(a), "b": _fmt(b)}, ["bar_end", "close", "n_ticks"]))
    df = pd.concat(parts, ignore_index=True)
    df["bar_end"] = _utc(df["bar_end"])
    return df.drop_duplicates("bar_end").sort_values("bar_end").reset_index(drop=True)


# ------------------------------------------------------------------ raw week
def raw_week() -> dict[str, pd.DataFrame]:
    a, b = RAW_WEEK
    out = {}
    out["kxbtc15m_quotes_1s"] = q(
        f"""
        SELECT toStartOfSecond(ts) + INTERVAL 1 SECOND AS sec_end, market,
               argMax(yes_best_bid, ts) AS bid, argMax(yes_best_ask, ts) AS ask,
               argMax(quote_valid, ts) AS quote_valid, argMax(spot_close, ts) AS btc,
               count() AS n_rows
        FROM tick_15min
        WHERE asset = 'BTC' AND market LIKE 'KXBTC15M-%%' AND ts >= %(a)s AND ts < %(b)s
        GROUP BY sec_end, market ORDER BY sec_end, market {NO_PREWHERE}
        """,
        {"a": _fmt(a), "b": _fmt(b)}, ["sec_end", "market", "bid", "ask", "quote_valid", "btc", "n_rows"])
    out["kxbtcd_quotes_10s"] = q(
        f"""
        SELECT toStartOfInterval(ts, INTERVAL 10 SECOND) + INTERVAL 10 SECOND AS sec_end, market,
               argMax(yes_best_bid, ts) AS bid, argMax(yes_best_ask, ts) AS ask,
               argMax(quote_valid, ts) AS quote_valid, count() AS n_rows
        FROM tick_btc_hd
        WHERE horizon = '1h' AND ts >= %(a)s AND ts < %(b)s
        GROUP BY sec_end, market ORDER BY sec_end, market {NO_PREWHERE}
        """,
        {"a": _fmt(a), "b": _fmt(b)}, ["sec_end", "market", "bid", "ask", "quote_valid", "n_rows"])
    for k in out:
        out[k]["sec_end"] = _utc(out[k]["sec_end"])
    out["settlements_15m"] = settlements("KXBTC15M", a, b + timedelta(minutes=15))
    out["settlements_1h"] = settlements("KXBTCD", a, b + timedelta(hours=1))
    return out


# ----------------------------------------------------------------------- EDA
def eda() -> dict[str, pd.DataFrame]:
    out = {}
    print("  eda: tick_15min daily", flush=True)
    out["tape_15m_daily"] = q(
        f"""
        SELECT toDate(ts) AS day, count() AS rows, uniq(market) AS markets,
               countIf(quote_valid = 1) AS valid,
               countIf(yes_best_ask < yes_best_bid) AS crossed,
               countIf(yes_best_bid IS NULL OR yes_best_ask IS NULL) AS null_quote,
               max(quote_src_ver) AS src_ver,
               countIf(spot_close > 0) AS spot_rows
        FROM tick_15min WHERE asset = 'BTC' AND market LIKE 'KXBTC15M-%%'
        GROUP BY day ORDER BY day {NO_PREWHERE}
        """, cols=["day", "rows", "markets", "valid", "crossed", "null_quote", "src_ver", "spot_rows"])
    print("  eda: tick_btc_hd daily", flush=True)
    out["tape_1h_daily"] = q(
        f"""
        SELECT toDate(ts) AS day, count() AS rows, uniq(market) AS markets,
               countIf(quote_valid = 1) AS valid,
               countIf(yes_best_ask < yes_best_bid) AS crossed,
               max(quote_src_ver) AS src_ver
        FROM tick_btc_hd WHERE horizon = '1h'
        GROUP BY day ORDER BY day {NO_PREWHERE}
        """, cols=["day", "rows", "markets", "valid", "crossed", "src_ver"])
    print("  eda: other tables daily", flush=True)
    other = []
    for name, sql in [
        ("order_book_l2", "SELECT toDate(ts) d, count() FROM order_book WHERE asset='BTC' GROUP BY d"),
        ("spot_venue_events", "SELECT toDate(ts) d, count() FROM spot_prices WHERE asset='BTC' GROUP BY d"),
        ("indicators", "SELECT toDate(ts) d, count() FROM indicator_15min WHERE asset='BTC' GROUP BY d"),
        ("brti_index", "SELECT toDate(ts) d, count() FROM kalshi_index_value WHERE asset='BTC' GROUP BY d"),
    ]:
        d = q(sql + NO_PREWHERE, cols=["day", "rows"])
        d["table"] = name
        other.append(d)
    out["other_tables_daily"] = pd.concat(other, ignore_index=True)
    print("  eda: settlements daily", flush=True)
    out["labels_daily"] = q(
        """
        SELECT series, toDate(close_time) AS day, count() AS markets, countIf(result = 'yes') AS yes
        FROM market_settlements FINAL
        WHERE series IN ('KXBTC15M', 'KXBTCD') AND result IN ('yes', 'no')
        GROUP BY series, day ORDER BY series, day
        """, cols=["series", "day", "markets", "yes"])
    for k in out:
        out[k]["day"] = pd.to_datetime(out[k]["day"])
    return out


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["all", "snap15", "snap1h", "btc", "raw", "eda"])
    ap.add_argument("--env-file")
    args = ap.parse_args()
    if args.env_file:
        _load_env_file(args.env_file)
    w = args.what

    if w in ("all", "btc"):
        print("btc_1m", flush=True)
        btc_1m(TAPE_START, END).to_parquet(DATA / "btc_1m.parquet", index=False)
    if w in ("all", "snap15"):
        print("snapshots KXBTC15M", flush=True)
        s = snapshots("KXBTC15M", "tick_15min", "t.asset = 'BTC' AND t.market LIKE 'KXBTC15M-%%'",
                      TAUS_15M, LABELS_15M_START, END)
        s.to_parquet(DATA / "snapshots" / "kxbtc15m.parquet", index=False)
        print(s.candle_status.value_counts().to_string())
    if w in ("all", "snap1h"):
        print("snapshots KXBTCD", flush=True)
        s = snapshots("KXBTCD", "tick_btc_hd", "t.horizon = '1h'", TAUS_1H, HOURLY_START, END)
        s.to_parquet(DATA / "snapshots" / "kxbtcd.parquet", index=False)
        print(s.candle_status.value_counts().to_string())
    if w in ("all", "raw"):
        print("raw week", flush=True)
        for k, df in raw_week().items():
            df.to_parquet(DATA / "raw_week" / f"{k}.parquet", index=False)
    if w in ("all", "eda"):
        print("eda", flush=True)
        for k, df in eda().items():
            df.to_parquet(DATA / "eda" / f"{k}.parquet", index=False)


if __name__ == "__main__":
    main()
