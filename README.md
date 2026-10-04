# tab-trader

TabPFN-3.5 against six tuned machine-learning models and the market itself, pricing Kalshi's Bitcoin contracts.
It runs on one CPU and uses six and a half months of market data I recorded myself.

**Interactive report:** https://cloblak.github.io/tab-trader/

<!-- RESULTS:START -->
## Bottom line

**My real strategy, holdout 2026-09-12 to 2026-10-02 (the weeks I traded live).** $100 to start, 15% of the account staked per trade. Each model only picks which of my strategy's signals to take.

| Trade filter | $100 becomes | Sharpe | Sortino | Max drawdown | Trades | Win rate |
|---|---|---|---|---|---|---|
| My current filter | $1,735.09 | 10.87 | 38.41 | -41.5% | 295 | 78.6% |
| **TabPFN-3.5** | $1,467.37 | 9.53 | 41.88 | -27.4% | 314 | 72.0% |
| Random forest | $769.25 | 8.18 | 20.35 | -53.2% | 392 | 70.9% |
| Logistic regression | $757.36 | 7.39 | 21.67 | -53.2% | 396 | 69.9% |
| XGBoost | $548.47 | 7.36 | 16.78 | -71.4% | 358 | 68.7% |
| Take every signal | $352.76 | 6.06 | 18.34 | -85.2% | 556 | 68.3% |
| LightGBM | $316.53 | 5.66 | 13.92 | -53.3% | 256 | 67.6% |
| CatBoost | $231.48 | 5.19 | 11.61 | -80.2% | 308 | 68.2% |
| Market price | $195.38 | 4.52 | 9.79 | -52.0% | 292 | 71.2% |
| MLP | $146.12 | 4.63 | 11.44 | -82.3% | 380 | 67.4% |

Same setup on the earlier periods: TabPFN train $0.10, test $115,014.07; my current filter train $49.96, test $1,315.99.

With real Kalshi fills: of my 345 live trades in the holdout, TabPFN would have kept 232. Those earned +4.08¢ per contract, the ones it would have skipped -0.15¢.

Staking 15% per trade compounds very fast. Read the dollar figures as a comparison between filters, not as achievable profit: Kalshi's books are too thin for those sizes, and the backtest overstates live results by a few cents per contract.

**Public benchmark (generic momentum indicators).** Lower log loss is better. Every model gets the same 5,000 rows; classic models are tuned, TabPFN is not.

*KXBTC15M · 15-minute up/down: still running.*

*KXBTCD · hourly strike ladder: still running.*

<!-- RESULTS:END -->

## What this project is

Kalshi lists a Bitcoin contract every 15 minutes ("will BTC finish this window higher?") and an hourly ladder
("will BTC be above $K at the top of the hour?"). Each contract pays $1, so its price is the crowd's probability.
A model only makes money when its probability beats the price by more than the fee.

I have recorded these markets about four times a second since 21 March 2026: about 245 million rows of quotes, order book,
exchange spot prices and settlement-index prints. Kalshi keeps only about two months of history, so most of this data
cannot be downloaded today.

The report has two parts:

1. **My real strategy** (top of the report). Each model acts as the trade filter for my live strategy's signals, with
   $100 risking 15% per trade, across train, test and a holdout that matches the weeks I traded live. These models use
   my strategy's private features. Only their predictions and outcomes are published.
2. **A public benchmark** (rest of the report). The same comparison on generic, textbook momentum indicators that anyone
   can rebuild from this repo.

## How TabPFN-3.5 is used

| Use | Variant | Setup |
|---|---|---|
| Benchmark and strategy filter | TabPFN-3.5, open weights, CPU | `fit_mode="fit_with_cache"`, refit weekly, never tuned |
| Benchmark | TabPFN-3.5-Fast, CPU | same API, `ModelVersion.V3_5_FAST` |
| Extra comparison | TabPFN-3.5 Thinking, Prior Labs API | `thinking_mode=True`, `time_col` set to the decision time |

## Run it

```bash
git clone https://github.com/Cloblak/tab-trader && cd tab-trader
uv sync
export TABPFN_TOKEN=...     # free at https://ux.priorlabs.ai
make quick                  # one test week per market, all models, about 10 minutes on CPU
make benchmark              # full benchmark, a few hours, resumable
make analyze report         # rebuild results/summary.json and docs/index.html
make notebooks verify       # run the notebooks and the repository checks
```

## What is in the repo

| Path | Contents |
|---|---|
| `notebooks/` | 00 bottom line · 01 Kalshi primer · 02 data · 03 cleaning and features · 04 benchmark |
| `data/` | everything needed to reproduce the results (see `data/DATA_CARD.md`) |
| `src/tabtrader/` | features, models, walk-forward runner, metrics, bankroll simulation, report builder |
| `results/` | cached predictions and `summary.json` |
| `PREREG.md` | the test plan, written before the benchmark ran, with dated amendments |
| `docs/index.html` | the interactive report |

## How the comparison is kept fair

- Walk-forward weekly tests. Every model trains only on markets that closed before the test week.
- Every model gets the same 5,000 training rows. Classic models are tuned; TabPFN is not.
- The market price is always one of the competitors.
- Prices are taken at a single instant, and each one is checked against Kalshi's own 1-minute candles.
- A planted-signal check and a no-signal check confirm that the test setup tells signal from noise.

## Limits

- The test and holdout weeks in the strategy section were also seen in my earlier research. A forward test is next.
- Generic momentum indicators carry little information beyond the price. Read the public benchmark's trading results as descriptive.
- TabPFN-3.5's open weights are licensed for evaluation. Live trading with them needs a commercial licence or the Prior Labs API.

## Licence

Code and data in this repository: Apache License 2.0. TabPFN-3.5 weights are not included and are governed by Prior Labs' licence.
