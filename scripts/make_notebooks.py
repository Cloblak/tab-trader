"""Generate the five notebooks in notebooks/ (then execute them with `make notebooks`)."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = """\
import sys, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from tabtrader import plots
plots.style()
pd.set_option("display.width", 140, "display.max_columns", 30)"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB / name)
    print("wrote", NB / name)


# --------------------------------------------------------------------------- 01
write("01_kalshi_primer.ipynb", [
    md("""
# 01 · Kalshi BTC markets in five minutes

**Kalshi** is a US exchange (CFTC-regulated) for event contracts. Each contract pays **$1 if the event
happens and $0 if not**, so its price in cents reads directly as the crowd's probability.

This project looks at two recurring Bitcoin markets:

| | **KXBTC15M** — 15-minute up/down | **KXBTCD** — hourly strike ladder |
|---|---|---|
| Question | "Will BTC be higher at the end of this 15-minute window than at its start?" | "Will BTC be above $K at the top of the hour?" for a ladder of strikes K |
| New market | every 15 minutes, 96 per day | every hour, many strikes per hour |
| Settles on | the 60-second average of the CF Benchmarks BRTI index before close | same |
| Price | 1–99 ¢ = implied probability | same |

**Buying and fees.** You buy YES at the *ask* or NO at 100 − *bid*. A taker pays
`0.07 × C × (1 − C)` dollars per contract (C = price in dollars). That is 1.75 ¢ at 50 ¢, the most
expensive point, so any edge has to clear the fee first.

**The trading idea in one line:** the contract price is already a forecast. A model is only useful if
its probability is *better than the price*, by more than the fee, on contracts you can actually buy.
"""),
    code(SETUP),
    code("""
raw = ROOT / "data" / "raw_week"
q15 = pd.read_parquet(raw / "kxbtc15m_quotes_1s.parquet")
s15 = pd.read_parquet(raw / "settlements_15m.parquet")
print(f"one week of 15-minute markets: {s15.market.nunique()} markets, {len(q15):,} one-second quote rows")
s15.head()"""),
    md("### One 15-minute market, second by second"),
    code("""
m = s15.iloc[len(s15) // 2]
g = q15[q15.market == m.market].sort_values("sec_end")
g = g[g.quote_valid == 1]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
a1.fill_between(g.sec_end, g.bid, g.ask, color=plots.TABPFN, alpha=.25, lw=0, label="bid–ask")
a1.plot(g.sec_end, (g.bid + g.ask) / 2, color=plots.TABPFN, lw=1.5, label="mid (¢)")
a1.set_ylabel("YES price (¢)"); a1.set_ylim(0, 100); a1.legend(loc="upper left")
a1.set_title(f"{m.market} — settled {'YES' if m.label else 'NO'}")
a2.plot(g.sec_end, g.btc, color=plots.INK, lw=1.2, label="BTC spot")
a2.axhline(m.strike, color=plots.MUTED, ls="--", lw=1, label="reference (strike)")
a2.set_ylabel("BTC (USD)"); a2.legend(loc="upper left")
plt.tight_layout()"""),
    md("### The hourly ladder at one instant (30 minutes before close)"),
    code("""
qh = pd.read_parquet(raw / "kxbtcd_quotes_10s.parquet")
sh = pd.read_parquet(raw / "settlements_1h.parquet")
ev = sh.close_time.value_counts().index[len(sh.close_time.unique()) // 2]
legs = sh[sh.close_time == ev].sort_values("strike")
t = ev - pd.Timedelta(minutes=30)
snap = (qh[qh.market.isin(legs.market) & (qh.sec_end <= t) & (qh.quote_valid == 1)]
        .sort_values("sec_end").groupby("market").tail(1).merge(legs, on="market").sort_values("strike"))
snap["mid"] = (snap.bid + snap.ask) / 2
ax = snap.plot.bar(x="strike", y="mid", color=plots.TABPFN, legend=False, figsize=(7, 3))
ax.set_ylabel("P(BTC above strike), ¢"); ax.set_xlabel("strike (USD)")
ax.set_title(f"Hourly ladder, close {ev:%Y-%m-%d %H:%M} UTC")
snap[["market", "strike", "bid", "ask", "mid", "label"]]"""),
    md("### The fee curve"),
    code("""
from tabtrader.evaluate import fee_c
c = np.arange(1, 100)
plt.figure(figsize=(6, 3))
plt.plot(c, fee_c(c), color=plots.INK)
plt.xlabel("contract price (¢)"); plt.ylabel("taker fee (¢ per contract)")
plt.title("Kalshi taker fee peaks at 50¢"); plt.tight_layout()"""),
])

# --------------------------------------------------------------------------- 02
write("02_eda.ipynb", [
    md("""
# 02 · The dataset: what was collected, how clean it is, and why it is enough

Everything here was recorded by my own collectors, which have run continuously since **21 March 2026**.
None of it can be downloaded after the fact. The exchange's API only keeps about two months of candles
and fills, and it does not keep sub-second order-book history at all.
"""),
    code(SETUP),
    code("""
from tabtrader import eda
t = eda.eda_tables()
pd.DataFrame(eda.MILESTONES, columns=["since", "what is recorded"])"""),
    code("""
v = eda.volume_totals(t)
pd.Series({
  "15-minute market tape (rows)": v["tape_15m_rows"],
  "hourly ladder tape (rows)": v["tape_1h_rows"],
  "L2 order-book snapshots": v["order_book_rows"],
  "per-venue BTC spot events": v["spot_venue_rows"],
  "derived indicator rows": v["indicator_rows"],
  "BRTI settlement-index prints": v["brti_rows"],
  "settled 15-minute markets (labels)": v["labels_15m"],
  "settled hourly ladder legs (labels)": v["labels_1h_all_legs"],
}).map("{:,}".format).to_frame("count")"""),
    md("### Coverage: rows per day"),
    code("""
d15, d1h = t["tape_15m_daily"], t["tape_1h_daily"]
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(d15.day, d15.rows / 1e3, color=plots.TABPFN, label="15-minute tape")
ax.plot(d1h.day, d1h.rows / 1e3, color=plots.FAST, label="hourly tape")
ax.set_ylabel("thousand rows / day"); ax.legend(); ax.set_title("Collection never stopped")
plt.tight_layout()"""),
    md("""
### Quality: the collector upgrade on 30 July

`quote_valid` marks a two-sided, uncrossed, non-sentinel quote. The v1 collector stored about 60%
valid rows, and 2–4% of its rows were *crossed* (ask < bid, an impossible book). The v2 collector
fixed both. Earlier research on this data found that crossed rows produce phantom "edges" in
backtests, so they are excluded everywhere here.
"""),
    code("""
w = eda.weekly_quality(t)
fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.2))
for mk, col in [("15-minute", plots.TABPFN), ("hourly", plots.FAST)]:
    g = w[w.market == mk]
    a.plot(g.week, g.valid_pct, color=col, label=mk); b.plot(g.week, g.crossed_pct, color=col, label=mk)
for ax in (a, b): ax.axvline(pd.Timestamp("2026-07-30"), color=plots.MUTED, ls="--", lw=1)
a.set_title("valid quotes, % of rows"); b.set_title("crossed books, % of rows"); a.legend()
plt.tight_layout()"""),
    md("""
### Every decision quote checked against the exchange

For every decision snapshot used in modelling, my recorded quote is compared with Kalshi's own
1-minute candle for that minute. A snapshot is **confirmed** if it lies inside the candle's range and
within 2 ¢ of the exchange's closing quote. The hourly tape before the v2 upgrade fails this check
most of the time, so it is excluded.
"""),
    code("""
pd.concat({k: eda.candle_agreement(k).set_index("month") for k in ["15m", "1h"]}, axis=1).round(1)"""),
    md("### Is the market's own price already a good forecast?"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, k, name in [(axes[0], "15m", "15-minute"), (axes[1], "1h", "hourly")]:
    c = eda.market_calibration(k)
    c = c[c.n >= 50]
    ax.plot([0, 1], [0, 1], color=plots.MUTED, ls="--", lw=1)
    ax.errorbar(c.p, c.y, yerr=[c.y - c.lo, c.hi - c.y], fmt="o", color=plots.TABPFN, ms=5, capsize=2)
    ax.set_xlabel("market mid price"); ax.set_ylabel("settled YES rate"); ax.set_title(name)
plt.tight_layout()"""),
    md("""
Very nearly. Each 10¢ price bucket settles close to its price, with a slight lean at the extremes
(favourites win a little more often than their price says). So the market is a **strong baseline**, and
any model has to beat the price, not a coin flip. That is why every comparison in notebook 04 includes
the market itself.
"""),
    code("""
lab = t["labels_daily"]
lab.groupby("series").agg(markets=("markets", "sum"), yes=("yes", "sum")).assign(
    yes_rate=lambda d: (d.yes / d.markets).round(3))"""),
])

# --------------------------------------------------------------------------- 03
write("03_cleaning_and_features.ipynb", [
    md("""
# 03 · Cleaning and off-the-shelf momentum features

**Unit of analysis:** one row per *market × decision time*.
- 15-minute markets: decide **10 min and 5 min** before close.
- Hourly ladder: decide **30 min and 15 min** before close.

Each row carries the quote at that instant, the quotes 60 s and 180 s earlier, the strike, and the
settlement label.

**Rules learned the hard way (each one once produced a fake edge):**
1. Use the quote at a single instant, never an average over a window.
2. Use only collector-validated, uncrossed quotes at most 5 s old.
3. Confirm every quote against the exchange's own candle.
4. Use only BTC bars that had closed by the decision time.
"""),
    code(SETUP),
    code("""
from tabtrader import eda
from tabtrader.data import load_market, MARKETS
for k in ["15m", "1h"]:
    print(MARKETS[k].title)
    for step, n in eda.cleaning_funnel(k):
        print(f"   {n:>7,}  {step}")"""),
    md("### Reproducing the published table from the raw week"),
    code("""
raw = ROOT / "data" / "raw_week"
q = pd.read_parquet(raw / "kxbtc15m_quotes_1s.parquet")
btc_pub = pd.read_parquet(ROOT / "data" / "btc_1m.parquet")
# 1-minute BTC bars from the 1-second tape: the close is the last price in the minute.
sec = q.groupby("sec_end").btc.last().dropna()
rebuilt = sec[sec.index.second == 0]
rebuilt.index.name = "bar_end"
cmp = btc_pub.set_index("bar_end").close.reindex(rebuilt.index)
print(f"BTC 1-minute closes rebuilt from raw seconds: {np.isclose(cmp, rebuilt).mean():.2%} identical "
      f"({len(rebuilt):,} minutes)")"""),
    code("""
snap = pd.read_parquet(ROOT / "data" / "snapshots" / "kxbtc15m.parquet")
wk = snap[(snap.decision_ts >= "2026-09-26") & (snap.decision_ts < "2026-10-03") & snap.bid.notna()]
qq = q[q.quote_valid == 1].rename(columns={"sec_end": "decision_ts"})
m = wk.merge(qq[["market", "decision_ts", "bid", "ask"]], on=["market", "decision_ts"], suffixes=("", "_raw"))
same = (m.bid == m.bid_raw) & (m.ask == m.ask_raw)
print(f"decision quotes rebuilt from the 1-second raw tape: {same.mean():.1%} identical "
      f"of {len(m):,} (the rest moved in the final tick before the decision instant)")"""),
    md("""
### The features: about 20 standard indicators, all at textbook settings

| group | features |
|---|---|
| contract | mid price (as probability), its logit, spread, minutes to close, change in mid over the last 60 s and 180 s |
| moneyness | log(BTC / strike); the same in volatility units, z = log(S/K)/(σ√τ); Φ(z), a textbook digital price |
| BTC momentum (1-min bars) | RSI-14, Stochastic %K(14), CCI(20), MACD(12,26,9) histogram, EMA 9/21 gap, Bollinger %B(20,2), 60-min z-score, position in the 60-min range, one 10-minute OLS slope and its R² |
| volatility | Bollinger width(20), realised volatility over 15 and 60 min |
| calendar | hour of day (sine, cosine) |

None of them is tuned, and none of them is the feature set my live strategy uses.
"""),
    code("""
from tabtrader.features import FEATURES, FEATURE_GROUPS
df = load_market("15m")
df[FEATURES].describe().T[["mean", "std", "min", "max"]].round(3)"""),
    md("### How much do these features know beyond the price? (training period only, before 3 August)"),
    code("""
from sklearn.metrics import roc_auc_score
tr = df[df.decision_ts < "2026-08-03"]
resid = tr.label - tr.p_mid
info = pd.DataFrame({
    "AUC vs label": {f: roc_auc_score(tr.label, tr[f].fillna(tr[f].median())) for f in FEATURES},
    "corr with (label − price)": {f: np.corrcoef(tr[f].fillna(tr[f].median()), resid)[0, 1] for f in FEATURES},
}).round(3)
info.sort_values("corr with (label − price)", key=abs, ascending=False)"""),
    md("""
Alone, almost every feature looks predictive (AUC well above 0.5), because almost every feature also
tracks the price. Measured against **what the price misses**, each one is a weak, noisy signal.
Combining many weak signals without overfitting a few thousand rows is exactly the problem a
pretrained tabular prior is built for. Notebook 04 tests whether it succeeds.
"""),
])

# --------------------------------------------------------------------------- 04
write("04_model_benchmark.ipynb", [
    md("""
# 04 · TabPFN-3.5 vs six classic learners and the market itself (CPU only)

The protocol is fixed in [`PREREG.md`](../PREREG.md), which was written before any test fold was run.
- Weekly walk-forward test blocks.
- Every model trains only on markets that closed before the block starts.
- **Policy A:** every model sees the same 5,000 most recent rows.
- Classic learners are tuned on a validation slice (defaults + 20 random configurations). TabPFN is not tuned.

`make benchmark` reruns everything (a few hours on an 8-core CPU). This notebook reads the cached
results in `results/`.
"""),
    code(SETUP),
    code("""
S = json.loads((ROOT / "results" / "summary.json").read_text())
def board(market, policy="policy_A"):
    L = S[market][policy]
    rows = []
    for m, r in L["models"].items():
        v, mk = r.get("vs_ref"), r.get("vs_market")
        rows.append({"model": m, "log loss": r["logloss"], "Brier": r["brier"], "AUC": r["auc"], "ECE": r["ece"],
                     "Δ vs TabPFN-3.5": None if not v else v["mean"],
                     "95% CI": None if not v else f"[{v['lo']:+.4f}, {v['hi']:+.4f}]",
                     "BH": None if not v else v.get("bh_pass"),
                     "Δ vs market": None if not mk else mk["mean"],
                     "fit s": r["cpu"].get("fit_s_mean"), "tune s": r["cpu"].get("search_s_mean"),
                     "1 market s": r["cpu"].get("single_market_s")})
    return pd.DataFrame(rows).sort_values("log loss").set_index("model")
print(S["15m"]["policy_A"]["n_rows"], "test rows over", S["15m"]["policy_A"]["n_days"], "days (15-minute)")
board("15m").round(4)"""),
    code("""
print(S["1h"]["policy_A"]["n_rows"], "test rows over", S["1h"]["policy_A"]["n_days"], "days (hourly)")
board("1h").round(4)"""),
    md("### Paired difference in log loss vs TabPFN-3.5 (right of zero = the model is worse than TabPFN)"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
for ax, mk in zip(axes, ["15m", "1h"]):
    L = S[mk]["policy_A"]["models"]
    items = sorted([(m, r["vs_ref"]) for m, r in L.items() if r.get("vs_ref")], key=lambda x: x[1]["mean"])
    for i, (m, v) in enumerate(items):
        ax.plot([v["lo"], v["hi"]], [i, i], color=plots.color(m), lw=2)
        ax.plot(v["mean"], i, "o", color=plots.color(m), ms=7)
    ax.set_yticks(range(len(items)), [m for m, _ in items]); ax.axvline(0, color=plots.INK, lw=1)
    ax.set_title({"15m": "15-minute", "1h": "hourly"}[mk]); ax.set_xlabel("Δ log loss (model − TabPFN-3.5)")
plt.tight_layout()"""),
    md("### Learning curves: log loss vs the number of training rows (library defaults)"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, mk in zip(axes, ["15m", "1h"]):
    lc = S[mk]["learning_curve"]
    ns = [int(n) for n in lc]
    models = sorted({m for n in lc for m in lc[n]})
    for m in models:
        ys = [lc[n].get(m, {}).get("logloss") for n in lc]
        if m == "Market price":
            ax.axhline(ys[-1], color=plots.INK, ls="--", lw=1.2, label="market price")
            continue
        hi = m.startswith("TabPFN")
        ax.plot(ns, ys, marker="o", ms=4, color=plots.color(m), lw=2.2 if hi else 1, alpha=1 if hi else .7,
                label=m.replace(" · default", ""))
    ax.set_xscale("log"); ax.set_xlabel("training rows"); ax.set_ylabel("test log loss")
    ax.set_title({"15m": "15-minute", "1h": "hourly"}[mk])
axes[1].legend(fontsize=7.5, loc="upper right"); plt.tight_layout()"""),
    md("### Calibration (reliability) on the test blocks"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, mk in zip(axes, ["15m", "1h"]):
    ax.plot([0, 1], [0, 1], color=plots.MUTED, ls="--", lw=1)
    for m in ["Market price", "TabPFN-3.5", "LightGBM · tuned", "Random forest · tuned"]:
        r = S[mk]["policy_A"]["models"].get(m)
        if not r: continue
        rel = pd.DataFrame(r["reliability"])
        ax.plot(rel.p_mean, rel.y_mean, marker="o", ms=4, color=plots.color(m),
                lw=2 if m.startswith("TabPFN") else 1.2, label=f"{m} (ECE {r['ece']:.3f})")
    ax.set_title({"15m": "15-minute", "1h": "hourly"}[mk]); ax.legend(fontsize=7.5)
plt.tight_layout()"""),
    md("### Policy B: the classic learners get *all* history; TabPFN stays at 5,000 rows"),
    code("""
for mk in ["15m", "1h"]:
    if S[mk].get("policy_B"):
        print(mk); display(board(mk, "policy_B")[["log loss", "AUC", "Δ vs TabPFN-3.5", "95% CI", "BH"]].round(4))"""),
    md("### TabPFN-3.5 Thinking (API, time-aware) on the same blocks"),
    code("""
for mk in ["15m", "1h"]:
    th = S[mk].get("thinking")
    if not th: continue
    rows = {m: {"log loss": r["logloss"], "AUC": r["auc"],
                "Thinking − model": (r.get("thinking_minus_model") or {}).get("mean")}
            for m, r in th.items() if isinstance(r, dict) and "logloss" in r}
    print(mk, "blocks", th["blocks"]); display(pd.DataFrame(rows).T.round(4))"""),
    md("""
### From probabilities to trades

The calibrated-edge rule buys only when the model's calibrated probability beats the ask plus the fee
by a margin chosen on validation. Results are in cents per contract after taker fees,
trade-weighted, with day-block 95% CIs.
"""),
    code("""
for mk in ["15m", "1h"]:
    T = S[mk]["trading"]
    print(mk)
    display(pd.DataFrame({m: {k: v for k, v in r.items() if k not in ("cum", "margins_c")} for m, r in T.items()}).T)"""),
    md("### Live demo: fit TabPFN-3.5-Fast on CPU and score one market"),
    code("""
import os, time
from tabtrader.data import load_market, MARKETS, blocks, split_block, xy
from tabtrader.models import make_tabpfn
if os.environ.get("TABPFN_TOKEN"):
    df = load_market("15m"); spec = MARKETS["15m"]; a, b = blocks(spec)[-1]
    fit, val, test = split_block(df, spec, a, b, 5000)
    X, y = xy(fit); Xt, yt = xy(test)
    m = make_tabpfn("3.5-fast"); t0 = time.perf_counter(); m.fit(X, y); t_fit = time.perf_counter() - t0
    t0 = time.perf_counter(); p1 = m.predict_proba(Xt[:1])[0, 1]; t_one = time.perf_counter() - t0
    print(f"fit on {len(X):,} rows: {t_fit:.1f}s · one live market scored in {t_one*1000:.0f} ms → P(YES)={p1:.3f}"
          f" vs market {test.p_mid.iloc[0]:.3f}")
else:
    print("set TABPFN_TOKEN (https://ux.priorlabs.ai) to run the live demo")"""),
])

# --------------------------------------------------------------------------- 00
write("00_bottom_line.ipynb", [
    md("""
# 00 · TabPFN-3.5 as a weekly-refit trade filter for my live strategy

**This notebook uses my live strategy's real signals.** Their 32 features are private and not in this repo; the file
`data/strategy/rolling_predictions.parquet` holds each model's probability, its decision and the outcome for every signal.

The rule (PREREG.md, amendment 3), fixed before the run:
- every Monday, refit on the previous 12 weeks of signals (TabPFN-3.5 untuned; logistic regression and LightGBM at defaults);
- take a signal only if 100 × p − entry price − fee > 0;
- stake 15% of the account, or half the Kelly stake from p capped at 15%.

Every other notebook uses public data and generic indicators.
"""),
    code(SETUP),
    code("""
from tabtrader import bankroll
from tabtrader.report_data import ROLL_MODELS
d = pd.read_parquet(ROOT / "data" / "strategy" / "rolling_predictions.parquet")
meta = json.loads((ROOT / "data" / "strategy" / "rolling_meta.json").read_text())
print(meta["first_week"], "to", meta["last_signal"], "·", meta["weeks"], "weeks ·", meta["signals"], "signals")
d.head()"""),
    md("### Account value from $100"),
    code("""
res = bankroll.run_models(d, ROLL_MODELS)
fig, ax = plt.subplots(figsize=(10, 4))
for m in ROLL_MODELS[::-1]:
    c = pd.DataFrame(res[m]["curve"], columns=["day", "v"]); c["day"] = pd.to_datetime(c["day"])
    style = {"TabPFN-3.5": (plots.TABPFN, 2.8, "-"), "TabPFN-3.5 + Kelly": (plots.FAST, 2, "-"),
             "Live gate": ("#7a8594", 1.8, "--"), "Every signal": (plots.CLASSIC, 1.4, ":")}.get(m, (plots.CLASSIC, 1, "-"))
    ax.plot(c.day, c.v, color=style[0], lw=style[1], ls=style[2], label=m)
ax.set_yscale("log"); ax.set_ylabel("account value ($, log scale)"); ax.legend(fontsize=8); plt.tight_layout()"""),
    md("### Tear sheet"),
    code("""
cols = ["final", "sharpe", "sortino", "max_dd_pct", "trades", "win_rate", "c_per_ct", "weeks_up"]
pd.DataFrame({m: {k: r[k] for k in cols} for m, r in res.items()}).T.sort_values("final", ascending=False)"""),
    md("### Month by month (¢ per contract)"),
    code("""
d["month"] = d.day.str[:7]
rows = {}
for mo, g in d.groupby("month"):
    rows[mo] = {m: (round(g.net_c_exits[g[f"take__{m}"]].mean(), 2), int(g[f"take__{m}"].sum()))
                for m in ["Every signal", "TabPFN-3.5", "Live gate", "Logistic regression", "LightGBM"]}
pd.DataFrame(rows).T"""),
    md("### Real fills: my live trades since 2026-09-12, with and without TabPFN's veto"),
    code("""
lt = pd.read_parquet(ROOT / "data" / "strategy" / "rolling_live_trades.parquet")
days = pd.date_range(lt.day.min(), lt.day.max(), freq="D")
fig, ax = plt.subplots(figsize=(9, 3.6))
for lab, take, frac, col in [("as traded", np.ones(len(lt), bool), 0.15, "#7a8594"),
                             ("TabPFN veto, flat 15%", lt.tabpfn_keeps.values, 0.15, plots.TABPFN),
                             ("TabPFN veto, half-Kelly", lt.tabpfn_keeps.values, lt.frac_kelly.values, plots.FAST)]:
    eq = bankroll.daily_equity(lt.day, lt.real_net_c.values, lt.px_fill.values, take, days, risk=frac)
    ax.plot(eq.index, eq.values, color=col, lw=2.4, label=f"{lab}: ${eq.iloc[-1]:,.2f}")
ax.set_ylabel("account value ($)"); ax.legend(); plt.tight_layout()
k = lt.tabpfn_keeps
print(f"{len(lt)} live trades; TabPFN keeps {k.sum()}: kept {lt.real_net_c[k].mean():+.2f} ¢/ct, skipped {lt.real_net_c[~k].mean():+.2f} ¢/ct")"""),
])

# --------------------------------------------------------------------------- 05
write("05_train_test_holdout.ipynb", [
    md("""
# 05 · Train, test, holdout with the TabPFN API (the playground recipe)

The same steps as the Prior Labs playground, on the public, playground-ready table
`data/tabpfn_ready/kxbtc15m_features.csv`. One change matters: the split is **by time**. A random
`train_test_split` would let the model learn from the future.

| split | period | used for |
|---|---|---|
| train | Jun 7 – Aug 2 | fitting |
| test | Aug 3 – Sep 6 | picking the trading margin, once |
| holdout | Sep 7 – Oct 3 | scored once, at the end |

Needs `TABPFN_TOKEN` (free at https://ux.priorlabs.ai). Only public data is sent to the API.
"""),
    code(SETUP),
    code("""
import os, tabpfn_client
from tabpfn_client import TabPFNClassifier
tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])

df = pd.read_csv(ROOT / "data" / "tabpfn_ready" / "kxbtc15m_features.csv")
features = [c for c in df.columns if c not in ("decision_ts", "split", "market", "bid", "ask", "target")]
train, test, holdout = (df[df.split == s] for s in ("train", "test", "holdout"))
print({s: len(x) for s, x in zip(("train", "test", "holdout"), (train, test, holdout))}, len(features), "features")"""),
    code("""
model = TabPFNClassifier(model_path="v3.5_default", n_estimators=8)
model.fit(train[features], train["target"])
p_test = model.predict_proba(test[features])[:, 1]
p_hold = model.predict_proba(holdout[features])[:, 1]"""),
    code("""
from tabtrader.evaluate import metrics
from tabtrader.holdout import trade, _trade_frame
rows = {}
for name, pt, ph in [("TabPFN-3.5 (API)", p_test, p_hold), ("Market price", test.p_mid.values, holdout.p_mid.values)]:
    rows[name] = {**{f"holdout {k}": v for k, v in metrics(holdout.target, ph).items() if k != "n"},
                  **trade(_trade_frame(test), _trade_frame(holdout), pt, ph)}
pd.DataFrame(rows).T"""),
    md("""
### The same split on my strategy's signals (local TabPFN-3.5, private features)

Train scores are out-of-fold; the test period sets each filter's trade rule; the holdout is the weeks I traded live.
"""),
    code("""
from tabtrader import bankroll
d = pd.read_parquet(ROOT / "data" / "strategy" / "bluf_predictions.parquet")
rows = ["TabPFN-3.5 + Kelly", "TabPFN-3.5", "Live gate", "Random forest", "Logistic regression", "Every signal"]
pd.DataFrame({s: {m: r["final"] for m, r in bankroll.run_models(d, rows, s).items()}
              for s in ("train", "test", "holdout")}).round(2)"""),
])
