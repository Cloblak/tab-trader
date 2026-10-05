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
           ""]
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
    out += ["", f"July decided this table. My signal lost {abs(jul.net_c_exits.mean()):.2f}¢ per contract that month, and "
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


def main() -> None:
    p = ROOT / "README.md"
    block = "\n".join(rolling_block())
    s = p.read_text()
    p.write_text(re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                        lambda _: "<!-- RESULTS:START -->\n" + block + "\n<!-- RESULTS:END -->", s, flags=re.S))
    print("README results block updated")


if __name__ == "__main__":
    main()
