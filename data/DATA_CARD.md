# Data card

All market data here was recorded by the author's own collectors from Kalshi's public market-data feeds
and public BTC spot feeds, then stored in a private ClickHouse database. The files in this folder were
exported with [`scripts/export_from_clickhouse.py`](../scripts/export_from_clickhouse.py), and that script
documents every query. Timestamps are UTC. Prices are in cents (1–99). `label = 1` means the contract settled YES.

## Collection timeline

| Since (UTC) | What the collectors record |
|---|---|
| 2026-03-21 | KXBTC15M tape at ~4 rows/s (top of book, spot composite), derived indicators |
| 2026-05-01 | Settlement labels retained (earlier labels were purged by the exchange API) |
| 2026-06-07 | Exchange 1-minute candlesticks backfilled daily (used to cross-check quotes) |
| 2026-06-30 | Full L2 order book, per-venue spot (Coinbase, Kraken, Binance, Alpaca) |
| 2026-07-05 | KXBTCD hourly strike-ladder tape at ~1 row/s per strike |
| 2026-07-30 | Collector v2: exact deci-cent quotes; crossed books eliminated |
| 2026-08-02 | CF Benchmarks BRTI index (the settlement source) at 1 s |

## Files

| File | Rows | What it is |
|---|---|---|
| `snapshots/kxbtc15m.parquet` | 29,110 | One row per settled 15-minute market × decision time (600 s and 300 s before close), 2026-05-01 → 2026-10-03 |
| `snapshots/kxbtcd.parquet` | 26,056 | One row per settled hourly strike × decision time (1,800 s and 900 s before close), 2026-07-05 → 2026-10-03 |
| `btc_1m.parquet` | 275,220 | BTC 1-minute closes from the collector's spot composite, indexed by bar **end** |
| `raw_week/kxbtc15m_quotes_1s.parquet` | 563,298 | Sample week (2026-09-26 → 10-02): last 15-minute-market quote and BTC price in each second |
| `raw_week/kxbtcd_quotes_10s.parquet` | 279,491 | Same week, hourly ladder, last quote in each 10 s |
| `raw_week/settlements_{15m,1h}.parquet` | 665 / 1,083 | Strikes, open/close times and labels for the sample week |
| `eda/*.parquet` | — | Daily aggregates only: rows, markets, valid/crossed counts, labels |
| `strategy/bluf_predictions.parquet` | 3,552 | Bottom line: every signal of the author's live strategy (Apr 15 – Oct 2) with the split (train/test/holdout), entry price, outcome, net ¢ per contract, and each model's score and take/skip decision. No features, no timestamps finer than a day, no tickers or trade direction |
| `strategy/live_trades.parquet` | — | The real live trades since 2026-09-12: day, fill price, net ¢ per contract after Kalshi fees, and whether TabPFN would have kept each one |
| `strategy/bluf_meta.json` | — | Split dates, counts and each model's chosen trading margin |
| `fidelity/fidelity.json` | — | "Backtest vs live" section: live-vs-backtest agreement rates and cents per contract (no dollar amounts) |

### Snapshot columns

| Column | Meaning |
|---|---|
| `decision_ts` | the decision instant (`close_time − tau_s`) |
| `bid`, `ask` | YES best bid/ask: the last collector-validated, uncrossed quote at or before `decision_ts`, at most 5 s old |
| `bid_lag60`, `ask_lag60`, `bid_lag180`, `ask_lag180` | the same quote 60 s / 180 s before the decision |
| `strike` | the market's reference price (15-minute) or strike (hourly) |
| `src_ver` | collector version of the quote (1 = legacy, 2 = from 2026-07-30) |
| `quote_age_s` | age of the quote at the decision instant |
| `candle_status` | `ok`: inside the exchange's own 1-minute candle; `mismatch`; `no_candle`; `no_quote`; `outage` |
| `candle_close_diff_c` | mean absolute difference (¢) between this quote and the exchange candle's closing bid/ask |

**Modelling rows** are those with `candle_status == "ok"`, `candle_close_diff_c <= 2`, and a BTC bar at the
decision minute. Hourly rows must also have a mid between 10¢ and 90¢ (see `src/tabtrader/data.py`).

## Known limitations

* No exchange candles exist before 2026-06-07, so May 15-minute snapshots cannot be cross-checked and are not used for modelling.
* The hourly tape recorded by collector v1 (July) disagrees with the exchange most of the time and is excluded by the candle check.
* Volume columns are not published: the collector's per-tick volume definition changed over time.
* BTC spot is the collector's multi-venue composite, not the BRTI settlement index (which is only recorded from 2026-08-02).
