# Pre-registration: TabPFN-3.5 vs classic ML on Kalshi BTC binaries

Written 2026-10-04 ~13:20 UTC, **before** any walk-forward test metric was computed. (A code smoke test at 13:24 produced
block-0 predictions for three models to check the pipeline; no test metric was computed from them.)
One development slice was looked at during setup (15-minute market, rows 5,000 to 6,300 of the clean
table, all in July and before the first test block): TabPFN-3.5 log loss 0.4925 vs the market's 0.4888.
That slice is training data in every test fold below.

## Question

Given the contract's own price plus ~20 off-the-shelf momentum features, which model gives the best
out-of-sample probability that a Kalshi BTC contract settles YES, on a laptop-class CPU?

## Data

| | 15-minute up/down (KXBTC15M) | Hourly strike ladder (KXBTCD) |
|---|---|---|
| Rows | one per market × decision time | one per strike × decision time |
| Decision times | 600 s and 300 s before close | 1,800 s and 900 s before close |
| Cleaning | `quote_valid`, ask > bid, quote ≤ 5 s old, exchange-candle check passed and within 2 ¢ of the candle close, spot bar present | same, plus mid between 10 ¢ and 90 ¢ |
| Test blocks | 9 weekly blocks, Mon 2026-08-03 → Sat 2026-10-03 | 6 weekly blocks, Mon 2026-08-24 → Sat 2026-10-03 |

Hourly blocks start later because clean hourly quotes only exist from the 2026-07-30 collector upgrade;
the rule is "first Monday whose training pool has ≥ 2,000 rows after all cleaning".

*Amendment 13:25 UTC, before any test metric:* the rule was first evaluated before the 10–90 ¢ filter,
which gave Aug 17 (pool 1,749 after filtering). Re-applied correctly it gives Aug 24 (pool 2,431),
and the hourly validation slice is set to 750 rows so the first fit set is not smaller than validation.

## Protocol (identical for every model)

* **Training pool** for a block: every clean row whose market **closed before** the block starts.
* **Validation slice**: the most recent 1,250 rows of the pool (15-minute) / 750 rows (hourly).
  Used for classic-model tuning, isotonic calibration and the trading margin. Never for TabPFN tuning.
* **Fit set**:
  * **Policy A (equal context, headline):** the 5,000 rows before the validation slice. 5,000 is
    TabPFN-3.5's default CPU limit.
  * **Policy B (classic models get everything):** all rows before the validation slice (up to ~16k).
    TabPFN stays at 5,000.
* **Models**: market mid price (benchmark, nothing fitted); logistic regression; random forest; XGBoost;
  LightGBM; CatBoost; MLP; TabPFN-3.5; TabPFN-3.5-Fast. Classic models: best of library defaults +
  20 random configs by validation log loss. TabPFN: defaults, never tuned. API Thinking mode: extra
  row on as many blocks as credits allow, compared only on those blocks.
* **Learning curve**: fit-set sizes 250 / 500 / 1,000 / 2,000 / 5,000, library defaults for classic models.

## Metrics

* **Primary**: out-of-sample log loss pooled over all test rows, per market. Pairwise differences
  vs TabPFN-3.5 with 95% CIs from a day-block bootstrap (2,000 resamples of calendar days).
  Benjamini–Hochberg at q = 0.10 across the pairwise comparisons.
* **Secondary**: Brier score, AUC, expected calibration error (10 bins), CPU seconds (fit, predict,
  single-market latency).
* **Trading** (descriptive, not the primary claim): isotonic-calibrate on validation; buy YES if
  q − ask − fee > m, buy NO if (1 − q) − no_ask − fee > m; margin m ∈ {0,1,2,3,5,8} ¢ picked on
  validation; at most one trade per market (earliest decision time); Kalshi taker fee 0.07·C·(1−C);
  net ¢ per contract, trade-weighted, day-block CI.

## Controls

* **Positive control**: a planted feature correlated with the true residual (label − price).
  Every model must beat the market with it.
* **Calibrated-market null**: labels replaced by Bernoulli(market mid). No model may beat the
  market significantly; the harness must not manufacture skill.

## What would count as "TabPFN is better"

TabPFN-3.5 (Policy A) has lower pooled log loss than every classic model, with BH-adjusted CIs
excluding zero for at least half of them, in at least one market. Beating the **market price**
is a separate, harder bar, reported either way.

## Amendment 2 — feature set changed (2026-10-04 ~14:20 UTC)

After the first 15-minute Policy A results were computed, the author asked to replace the five
multi-horizon BTC return features (1/3/5/15/60-minute returns) and the MACD line. The reason was privacy,
not performance: multi-horizon trend measures sit too close to the author's private live strategy.
They were replaced with standard non-slope indicators: Stochastic %K(14), CCI(20), EMA 9/21 gap,
Bollinger width(20), and a 60-minute z-score. The 10-minute slope and its R² stay as the only
trend-line features. Everything else in this protocol is unchanged, and the benchmark was rerun in full.

The superseded run's summary is kept in `results/archive_v1_features/summary.json`. On the 15-minute market,
TabPFN-3.5 had the lowest log loss of the eight learners there, and its gap to five of the six tuned classic
models was significant after Benjamini–Hochberg. No model beat the market price. Both results are reported,
so the reader can check that the conclusion does not depend on the feature swap.
