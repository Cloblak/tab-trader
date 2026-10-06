"""Write the results block of README.md from data/ and results/ (numbers are never typed by hand)."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import bankroll
from .data import DATA, MARKETS, ROOT

ROLL_MODELS = ["TabPFN-3.5", "TabPFN-3.5 + Kelly", "Logistic regression", "Logistic regression + Kelly",
               "LightGBM", "LightGBM + Kelly", "Live gate", "Every signal"]
NAMES = {"TabPFN-3.5": "**TabPFN-3.5**, flat 15%", "TabPFN-3.5 + Kelly": "**TabPFN-3.5**, half-Kelly",
         "Logistic regression": "Logistic regression, flat 15%", "Logistic regression + Kelly": "Logistic regression, half-Kelly",
         "LightGBM": "LightGBM, flat 15%", "LightGBM + Kelly": "LightGBM, half-Kelly",
         "Live gate": "My current filter (frozen), flat 15%", "Every signal": "Take every signal, flat 15%"}


def rolling_block() -> list[str]:
    from sklearn.metrics import roc_auc_score

    d = pd.read_parquet(DATA / "strategy" / "rolling_predictions.parquet")
    meta = json.loads((DATA / "strategy" / "rolling_meta.json").read_text())
    res = bankroll.run_models(d, ROLL_MODELS)
    auc = {m: roc_auc_score(d.won, d[f"p__{m}"]) for m in ("TabPFN-3.5", "Logistic regression", "LightGBM")}
    lt = pd.read_parquet(DATA / "strategy" / "rolling_live_trades.parquet")
    k = lt["tabpfn_keeps"].to_numpy()
    days = pd.date_range(lt.day.min(), lt.day.max())
    eq = lambda take, frac=bankroll.RISK: float(bankroll.daily_equity(  # noqa: E731
        lt.day, lt.real_net_c.values, lt.px_fill.values, take, days, risk=frac).iloc[-1])
    d["month"] = d["day"].str[:7]
    jul = d[d.month == "2026-07"]
    tj = jul["take__TabPFN-3.5"].to_numpy()
    out = ["## What TabPFN does well here", "",
           f"- **It ranks my live strategy's signals best.** In a weekly walk-forward ({meta['weeks']} weeks, "
           f"{meta['signals']:,} signals, each week refit on the previous {meta['context_weeks']}), TabPFN-3.5 separates "
           f"winners from losers with AUC {auc['TabPFN-3.5']:.3f}, against {auc['Logistic regression']:.3f} for logistic "
           f"regression and {auc['LightGBM']:.3f} for LightGBM. No tuning.",
           f"- **Its veto works on real fills.** On my {len(lt)} actual live trades since {lt.day.min()}, the trades "
           f"TabPFN would have kept ({int(k.sum())}) earned {lt.real_net_c[k].mean():+.2f}¢ per contract; the "
           f"{int((~k).sum())} it would have skipped earned {lt.real_net_c[~k].mean():+.2f}¢. At 15% per trade, $100 ends at "
           f"${eq(k):,.2f} with its veto against ${eq(np.ones(len(lt), bool)):,.2f} as I traded.",
           "", "![Real fills with and without TabPFN's veto](docs/img/live_fills.png)", ""]
    out += bench_line_items()
    out += ["", "## The full walk-forward", "",
            f"Weekly walk-forward on my live strategy's signals, {meta['first_week']} to {meta['last_signal']}. Every "
            "Monday each model is refit on the previous 12 weeks and takes a signal only if its probability beats the "
            "price plus the fee. The rules were fixed before the run ([`PREREG.md`](PREREG.md), amendment 3). $100 to start.", "",
            "| Trade filter | $100 becomes | Sharpe | Max drawdown | Trades | ¢ per contract |",
            "|---|---|---|---|---|---|"]
    for m, r in sorted(res.items(), key=lambda kv: -kv[1]["final"]):
        out.append(f"| {NAMES.get(m, m)} | ${r['final']:,.2f} | {r['sharpe']} | {r['max_dd_pct']}% | {r['trades']} | "
                   f"{r['c_per_ct']:+.2f} |")
    out += ["", "![Weekly walk-forward equity curves and monthly results](docs/img/walkforward.png)", "",
            f"July decided this table. My signal lost {abs(jul.net_c_exits.mean()):.2f}¢ per contract that month, and "
            f"TabPFN, still learning from April to June, took {int(tj.sum())} of {len(jul)} signals at "
            f"{jul.net_c_exits[tj].mean():+.2f}¢ each. At a flat 15% stake no filter survived it intact; half-Kelly sizing, "
            "which bets less when the edge is thin, is what kept accounts alive.", "",
            "Staking 15% per trade compounds fast. Read the dollar figures as a comparison between filters, not as "
            "achievable profit: Kalshi's order books are too thin for those sizes, and backtests overstate live results "
            "by a few cents per contract.", ""]
    return out


def bench_line_items() -> list[str]:
    S = json.loads((ROOT / "results" / "summary.json").read_text())
    bits = []
    for mk in MARKETS:
        L = S[mk]["policy_A"]
        if not L or "TabPFN-3.5" not in L["models"]:
            continue
        learners = {m: v["logloss"] for m, v in L["models"].items() if m != "Market price"}
        rank = sorted(learners, key=learners.get).index("TabPFN-3.5") + 1
        sig = sum(1 for m, v in L["models"].items() if v.get("vs_ref") and v["vs_ref"].get("bh_pass")
                  and v["vs_ref"]["mean"] > 0 and m != "Market price" and not m.startswith("TabPFN"))
        bits.append(f"{MARKETS[mk].title.split(' · ')[1]}: TabPFN-3.5 ranks #{rank} of {len(learners)} on prediction "
                    f"error and is significantly better than {sig} of 6 tuned classic models")
    return [f"- **It is at or near the top on public data.** With generic indicators that anyone can rerun from this repo "
            f"({'; '.join(bits)}). No model beats the market price itself on prediction error."]


def holdout_block() -> list[str]:
    r = json.loads((ROOT / "results" / "holdout_api.json").read_text())
    sp = r["splits"]
    out = ["## Train, test, holdout with the TabPFN API", "",
           "The Prior Labs playground recipe, applied to market data. One change matters: the split is by time. A random "
           "`train_test_split` would let the model learn from the future.", "",
           "```python",
           "import os, pandas as pd, tabpfn_client",
           "from tabpfn_client import TabPFNClassifier",
           "",
           'tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])',
           'df = pd.read_csv("data/tabpfn_ready/kxbtc15m_features.csv")   # public, playground-ready',
           'features = [c for c in df.columns if c not in ("decision_ts", "split", "market", "bid", "ask", "target")]',
           'train, test, holdout = (df[df.split == s] for s in ("train", "test", "holdout"))',
           "",
           'model = TabPFNClassifier(model_path="v3.5_default", n_estimators=8)',
           'model.fit(train[features], train["target"])',
           "p_test = model.predict_proba(test[features])[:, 1]        # used once, to pick the trading margin",
           "p_hold = model.predict_proba(holdout[features])[:, 1]     # scored once, at the end",
           "```", "",
           f"Train {sp['train']['rows']:,} rows (to {pd.Timestamp(sp['train']['to']) - pd.Timedelta(days=1):%b %d}), "
           f"test {sp['test']['rows']:,} (to {pd.Timestamp(sp['test']['to']) - pd.Timedelta(days=1):%b %d}), "
           f"holdout {sp['holdout']['rows']:,} (to {pd.Timestamp(sp['holdout']['to']) - pd.Timedelta(days=1):%b %d}). "
           "Full code: `python -m tabtrader.holdout run` and `notebooks/05_train_test_holdout.ipynb`.", "",
           "**Public data, generic indicators (holdout):**", "",
           "| Model | Log loss | AUC | Trades | ¢ per contract [95% CI] |", "|---|---|---|---|---|"]
    for m, v in sorted(r["models"].items(), key=lambda kv: kv[1]["holdout"]["logloss"]):
        t = v["trading_holdout"]
        tr = (f"{t['trades']} | {t['c_per_ct']:+.2f} [{t['ci95'][0]:+.1f}, {t['ci95'][1]:+.1f}]" if t.get("trades")
              else "0 | no trade cleared price + fee")
        name = f"**{m}**" if m.startswith("TabPFN") else m
        out.append(f"| {name} | {v['holdout']['logloss']:.4f} | {v['holdout']['auc']:.4f} | {tr} |")
    out += ["", "With only generic indicators the market price stays ahead, and neither trading result is distinguishable from zero. The same "
            "split on my strategy's signals, where the features carry real information (local TabPFN-3.5, features "
            "private; train scores are out-of-fold, the test period sets each filter's trade rule):", ""]
    d = pd.read_parquet(DATA / "strategy" / "bluf_predictions.parquet")
    meta = json.loads((DATA / "strategy" / "bluf_meta.json").read_text())
    rows = ["TabPFN-3.5 + Kelly", "TabPFN-3.5", "Live gate", "Random forest", "Logistic regression", "Every signal"]
    res = {s_: bankroll.run_models(d, rows, s_) for s_ in ("train", "test", "holdout")}
    names = {"TabPFN-3.5 + Kelly": "**TabPFN-3.5**, half-Kelly", "TabPFN-3.5": "**TabPFN-3.5**, flat 15%",
             "Live gate": "My current filter", "Random forest": "Random forest", "Logistic regression": "Logistic regression",
             "Every signal": "Take every signal"}
    out += [f"| $100 becomes | Train ({meta['splits']['train'][0][5:]} – {meta['splits']['train'][1][5:]}) | "
            f"Test ({meta['splits']['test'][0][5:]} – {meta['splits']['test'][1][5:]}) | "
            f"Holdout ({meta['splits']['holdout'][0][5:]} – {meta['splits']['holdout'][1][5:]}) |", "|---|---|---|---|"]
    for m in rows:
        out.append(f"| {names[m]} | " + " | ".join(f"${res[s_][m]['final']:,.2f}" for s_ in ("train", "test", "holdout"))
                   + " |")
    out += ["", "In spring my signal itself lost money, so every filter lost in the train period. The half-Kelly rule was "
            "picked among five TabPFN variants by test-period Sharpe, with the holdout visible at the time, so treat that row "
            "as indicative. The weekly walk-forward above is the stricter test.", ""]
    return out


def _put(s: str, tag: str, block: str) -> str:
    return re.sub(rf"<!-- {tag}:START -->.*<!-- {tag}:END -->",
                  lambda _: f"<!-- {tag}:START -->\n{block}\n<!-- {tag}:END -->", s, flags=re.S)


def main() -> None:
    figures()
    p = ROOT / "README.md"
    s = p.read_text()
    s = _put(s, "DATA", "\n".join(data_block()))
    s = _put(s, "RESULTS", "\n".join(rolling_block()))
    if "<!-- HOLDOUT:START -->" not in s:
        s = s.replace("<!-- RESULTS:END -->", "<!-- RESULTS:END -->\n\n<!-- HOLDOUT:START -->\n<!-- HOLDOUT:END -->", 1)
    s = _put(s, "HOLDOUT", "\n".join(holdout_block()))
    p.write_text(s)
    print("README data + results blocks and docs/img figures updated")



# ---------------------------------------------------------------- data section
def data_block() -> list[str]:
    from sklearn.metrics import roc_auc_score

    from . import eda

    t = eda.eda_tables()
    v = eda.volume_totals(t)
    w = eda.weekly_quality(t)
    q15 = w[w.market == "15-minute"]
    pre, post = q15[q15.week < "2026-07-27"], q15[q15.week >= "2026-08-03"]
    hc = eda.candle_agreement("1h").set_index("month")
    total = (v["tape_15m_rows"] + v["tape_1h_rows"] + v["order_book_rows"] + v["spot_venue_rows"]
             + v["indicator_rows"] + v["brti_rows"])
    d = pd.read_parquet(DATA / "strategy" / "rolling_predictions.parquet")
    lt = pd.read_parquet(DATA / "strategy" / "rolling_live_trades.parquet")
    auc_px = roc_auc_score(d.won, d.px)
    return [
        "## Why this is a real test",
        "",
        "Nothing here is a benchmark download. Every row was recorded by my own collectors while the markets traded, "
        "and my strategy trades on it with real money.",
        "",
        f"- **Real.** {total / 1e6:,.0f} million rows since {v['first_day']}: quotes four times a second, the full order book, "
        f"spot prices from four exchanges and the settlement index. Kalshi keeps only about two months of history, so most "
        f"of this exists nowhere else. Since {lt.day.min()} the strategy has traded it with real money: {len(lt)} trades "
        "with real fills and fees.",
        f"- **Messy.** Before a collector rewrite on 30 July only {100 * pre.valid.sum() / pre.rows.sum():.0f}% of quotes were "
        f"clean, and {100 * pre.crossed.sum() / pre.rows.sum():.0f}% of rows showed impossible crossed books (after it: "
        f"{100 * post.valid.sum() / post.rows.sum():.0f}% clean, none crossed). In July only "
        f"{hc.loc['2026-07', 'confirmed_pct']:.0f}% of the hourly ladder's recorded quotes matched the exchange's own records. "
        "There are outages, stale quotes and a regime change mid-sample. Every price used here is checked against Kalshi's "
        "candles.",
        f"- **Small and noisy.** My strategy produced {len(d):,} usable signals in five months, and only a few hundred per "
        f"regime. Signals win about {100 * d.won.mean():.0f}% of the time and the market price alone already ranks them at "
        f"AUC {auc_px:.3f}, so there is little left for any model to find.",
        "- **Honest labels, fast.** Every contract settles at $1 or $0 within 15 minutes, so every prediction is scored "
        "against reality, and the market price is a strong baseline to beat.",
        "",
        "Small, noisy, drifting tables with a hard baseline are the setting TabPFN was built for. "
        "No synthetic data, no cleaned-up competition set.",
        "",
    ]


# --------------------------------------------------------------------- figures
def figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    NAVY, STEEL, GRAY, MID, LIGHT, INK = "#1d3a6e", "#5a8fd4", "#4b5563", "#7a8594", "#b4bcc7", "#1b2533"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": MID, "xtick.color": GRAY, "ytick.color": GRAY,
                         "axes.grid": True, "grid.color": "#e6e9ee", "grid.linewidth": 0.8,
                         "axes.axisbelow": True})
    out = ROOT / "docs" / "img"
    out.mkdir(parents=True, exist_ok=True)
    money = FuncFormatter(lambda x, _: f"${x:,.0f}" if x >= 1 else f"${x:.2f}")

    # 1. walk-forward equity + monthly cents per contract
    d = pd.read_parquet(DATA / "strategy" / "rolling_predictions.parquet")
    res = bankroll.run_models(d, ROLL_MODELS)
    lines = [("TabPFN-3.5 + Kelly", "TabPFN-3.5, half-Kelly", NAVY, 2.6, "-"),
             ("TabPFN-3.5", "TabPFN-3.5, flat 15%", STEEL, 1.8, "-"),
             ("Logistic regression + Kelly", "Logistic regression, half-Kelly", GRAY, 1.6, "-"),
             ("Live gate", "My current filter", MID, 1.6, "--"),
             ("Every signal", "Take every signal", LIGHT, 1.4, ":")]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.3, 1]})
    for lo, hi, lab, col in [("2026-07-01", "2026-08-01", "July: the signal breaks down", "#a83232"),
                             ("2026-09-12", "2026-10-03", "live trading", NAVY)]:
        for a_ in (ax, bx):
            a_.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color=col, alpha=0.07, lw=0)
        ax.text(pd.Timestamp(lo) + pd.Timedelta(days=1), 2.2e5, lab, color=col, fontsize=8.5, va="top")
    handles = {}
    for key, lab, col, lw, ls in lines[::-1]:  # draw TabPFN last so it sits on top
        c = pd.DataFrame(res[key]["curve"], columns=["day", "v"])
        c["day"] = pd.to_datetime(c["day"])
        end = c.v.iloc[-1]
        handles[key], = ax.plot(c.day, c.v, color=col, lw=lw, ls=ls,
                                label=f"{lab}  (${end:,.0f})" if end >= 1 else f"{lab}  (${end:.2f})")
    ax.set_yscale("log")
    ax.set_ylim(0.005, 3e5)
    ax.yaxis.set_major_formatter(money)
    ax.axhline(100, color=INK, lw=0.8, alpha=0.5)
    ax.set_ylabel("account value, $100 start (log)")
    ax.legend(handles=[handles[k] for k, *_ in lines], loc="lower left", fontsize=8.3, frameon=False)
    ax.set_xlim(pd.Timestamp("2026-05-08"), pd.Timestamp("2026-10-18"))
    bx.xaxis.set_major_locator(mdates.MonthLocator())
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_title("Weekly walk-forward on my live strategy's signals (each model refit every Monday)",
                 loc="left", fontsize=10.5, color=INK)
    d["month"] = pd.to_datetime(d["day"]).dt.to_period("M").dt.to_timestamp()
    mo = d.groupby("month").apply(lambda g: pd.Series({
        "all": g.net_c_exits.mean(), "tab": g.net_c_exits[g["take__TabPFN-3.5"]].mean()}), include_groups=False)
    x = mo.index + pd.Timedelta(days=15)
    bx.bar(x - pd.Timedelta(days=4), mo["all"], width=8, color=LIGHT, label="every signal")
    bx.bar(x + pd.Timedelta(days=4), mo["tab"], width=8, color=NAVY, label="signals TabPFN took")
    bx.axhline(0, color=INK, lw=0.8)
    bx.set_ylabel("¢ per contract")
    bx.legend(loc="upper left", fontsize=8.3, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "walkforward.png", dpi=150)
    plt.close(fig)

    # 2. live period, real fills
    lt = pd.read_parquet(DATA / "strategy" / "rolling_live_trades.parquet")
    days = pd.date_range(lt.day.min(), lt.day.max())
    k = lt.tabpfn_keeps.to_numpy()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    curves = {}
    for lab, take, frac, col, lw in [("As I traded (flat 15%)", np.ones(len(lt), bool), bankroll.RISK, MID, 2.0),
                                     ("With TabPFN's veto, flat 15%", k, bankroll.RISK, NAVY, 2.6),
                                     ("With TabPFN's veto, half-Kelly", k, lt.frac_kelly.to_numpy(), STEEL, 1.8)]:
        eq = bankroll.daily_equity(lt.day, lt.real_net_c.to_numpy(), lt.px_fill.to_numpy(), take, days, risk=frac)
        curves[lab] = eq
        ax.plot(eq.index, eq.values, color=col, lw=lw, label=f"{lab}  (${eq.iloc[-1]:,.0f})")
    ax.axhline(100, color=INK, lw=0.8, alpha=0.5)
    ax.yaxis.set_major_formatter(money)
    ax.set_ylabel("account value")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    bad = pd.Timestamp("2026-10-02")
    if bad in curves["As I traded (flat 15%)"].index:
        yb = float(curves["As I traded (flat 15%)"].loc[bad])
        ax.annotate("one bad day (Oct 2)", xy=(bad, yb), xytext=(bad - pd.Timedelta(days=5), yb * 0.45),
                    fontsize=8.3, color=GRAY, arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.8})
    ax.set_title(f"Real Kalshi fills: my {len(lt)} live trades since {lt.day.min()}", loc="left", fontsize=10.5, color=INK)
    ax.legend(loc="upper left", fontsize=8.3, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "live_fills.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
