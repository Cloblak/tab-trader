# tab-trader

**TabPFN-3.5 deciding which Kalshi 15-minute Bitcoin trades to take, on real, messy, self-recorded market data.**
One CPU, no training loop, no hyper-parameter tuning.

Interactive report: https://cloblak.github.io/tab-trader/

<!-- RESULTS:START -->
## What TabPFN does well here

- **It ranks my live strategy's signals best.** In a weekly walk-forward (21 weeks, 2,831 signals, each week refit on the previous 12), TabPFN-3.5 separates winners from losers with AUC 0.731, against 0.726 for logistic regression and 0.689 for LightGBM. No tuning.
- **Its veto works on real fills.** On my 345 actual live trades since 2026-09-12, the trades TabPFN would have kept (149) earned +5.91¢ per contract; the 196 it would have skipped earned +0.25¢. At 15% per trade, $100 ends at $455.40 with its veto against $336.57 as I traded.

- **It is at or near the top on public data.** With generic indicators that anyone can rerun from this repo (15-minute up/down: TabPFN-3.5 ranks #2 of 8 on prediction error and is significantly better than 5 of 6 tuned classic models; hourly strike ladder: TabPFN-3.5 ranks #1 of 8 on prediction error and is significantly better than 1 of 6 tuned classic models). No model beats the market price itself on prediction error.

## The full walk-forward

Weekly walk-forward on my live strategy's signals, 2026-05-11 to 2026-10-02. Every Monday each model is refit on the previous 12 weeks and takes a signal only if its probability beats the price plus the fee. The rules were fixed before the run ([`PREREG.md`](PREREG.md), amendment 3). $100 to start.

| Trade filter | $100 becomes | Sharpe | Max drawdown | Trades | ¢ per contract |
|---|---|---|---|---|---|
| Logistic regression, half-Kelly | $54,357.33 | 6.11 | -84.5% | 1642 | +1.76 |
| My current filter (frozen), flat 15% | $5,853.62 | 4.26 | -88.5% | 1570 | +1.83 |
| **TabPFN-3.5**, half-Kelly | $5,209.30 | 4.38 | -92.4% | 1555 | +1.45 |
| Logistic regression, flat 15% | $240.97 | 3.0 | -99.7% | 1642 | +1.76 |
| LightGBM, half-Kelly | $75.93 | 2.39 | -99.8% | 2223 | +0.89 |
| **TabPFN-3.5**, flat 15% | $15.13 | 1.87 | -99.9% | 1555 | +1.45 |
| LightGBM, flat 15% | $3.71 | 1.53 | -100.0% | 2223 | +0.89 |
| Take every signal, flat 15% | $0.27 | 1.81 | -100.0% | 2831 | +0.73 |

July decided this table. My signal lost 4.65¢ per contract that month, and TabPFN, still learning from April to June, took 173 of 382 signals at -5.60¢ each. At a flat 15% stake no filter survived it intact; half-Kelly sizing, which bets less when the edge is thin, is what kept accounts alive.

Staking 15% per trade compounds fast. Read the dollar figures as a comparison between filters, not as achievable profit: Kalshi's order books are too thin for those sizes, and backtests overstate live results by a few cents per contract.

<!-- RESULTS:END -->

## How TabPFN is used

Kalshi lists a new Bitcoin contract every 15 minutes: *will BTC finish this window higher than it started?* It pays $1 or
$0, so its price is the crowd's probability. Making money means finding the moments when that price is wrong by more than the fee.

1. **Record.** My collectors have logged these markets about four times a second since March 2026: quotes, order book,
   exchange spot prices and the settlement index. The raw tape is messy. It has crossed books, stale quotes, outages and a
   collector rewrite halfway through. Every price used here is cleaned and checked against Kalshi's own records.
2. **Signal.** A momentum signal from my live strategy proposes a trade a few times an hour, described by 32 features of
   the BTC trend and the contract. The features stay private.
3. **Predict.** TabPFN-3.5 reads the last 12 weeks of past signals and their outcomes as context, and returns the
   probability that the new signal wins. Nothing is trained or tuned. The pretrained model is used as is.
4. **Decide.** Take the trade only if that probability beats the entry price plus Kalshi's fee. Stake 15% of the account, or
   half the Kelly stake implied by TabPFN's probability, capped at 15%.
5. **Repeat weekly.** Refit on the newest 12 weeks every Monday. A refit takes under a minute on a CPU, and scoring a live
   signal about a second.

## Why TabPFN suits this problem

- **Little data per regime.** A few hundred to a few thousand signals is all that exists before market conditions change.
  Tree models need far more rows to stop overfitting. TabPFN was pretrained for exactly this size of table.
- **Probabilities are compared to a price.** A trade is only good if the probability is right, not just well ranked.
  TabPFN's probabilities are well calibrated out of the box (about one percentage point of average error on the
  15-minute benchmark), so they can be compared with the price directly.
- **The world drifts.** Weekly refits with no tuning are what make a rolling strategy cheap to run.

## Public benchmark (fully reproducible)

The same question with only generic, textbook momentum indicators (RSI, Stochastic, CCI, MACD, Bollinger, one 10-minute
slope) on data published in this repo. It covers 9 weekly walk-forward tests on the 15-minute market and 6 on the hourly
strike ladder. TabPFN-3.5, untuned, runs against six tuned classic models with the same 5,000 training rows each. Results
are in the report and `notebooks/04_model_benchmark.ipynb`. The test plan was fixed in advance in [`PREREG.md`](PREREG.md).

## About the data

This repo publishes everything needed to rerun the public benchmark: decision snapshots for every settled market,
1-minute BTC prices, one raw week at 1-second resolution, and daily data-quality aggregates (see `data/DATA_CARD.md`).
For my live strategy it publishes only model scores, decisions and outcomes, not the features.

The full dataset (about 245 million rows since March 2026: tape, L2 order book, multi-venue spot, settlement index) is
my own and is the basis of my trading strategies, so I have not made it public. If you would like to work with it, I am
happy to talk. Please open an issue on this repo or reach me through my GitHub profile
([@Cloblak](https://github.com/Cloblak)).

## Run it

```bash
git clone https://github.com/Cloblak/tab-trader && cd tab-trader
uv sync
export TABPFN_TOKEN=...      # free at https://ux.priorlabs.ai
make quick                   # one test week per market, all models, about 10 minutes on CPU
make benchmark               # the full public benchmark (a few hours, resumable)
make analyze report          # rebuild results/summary.json and docs/index.html
```

`notebooks/` walks through the same steps: 00 the strategy result, 01 the markets, 02 the data and its quality,
03 cleaning and features, 04 the benchmark.

## Limits

- The strategy section uses my own signals; its features are private, so it can be checked but not rerun from this repo.
- Earlier research of mine looked at some of these weeks, so the next step is a live forward test.
- TabPFN-3.5's open weights are licensed for evaluation. Live trading with them needs a commercial licence or the Prior Labs API.

## Licence

Code and published data: Apache License 2.0. Built with [TabPFN](https://github.com/PriorLabs/TabPFN); the
TabPFN-3.5 weights are not included and are governed by Prior Labs' licence.
