"""Write the results block of README.md from data/ and results/ (numbers are never typed by hand)."""

from __future__ import annotations

import json
import re

import pandas as pd

from . import bankroll
from .data import DATA, MARKETS, ROOT
from .report_data import BLUF_MODELS

NAMES = {"Live gate": "My current filter", "Every signal": "Take every signal"}


def bluf_block() -> list[str]:
    d = pd.read_parquet(DATA / "strategy" / "bluf_predictions.parquet")
    meta = json.loads((DATA / "strategy" / "bluf_meta.json").read_text())
    out = ["## Bottom line", "",
           f"**My real strategy, holdout {meta['splits']['holdout'][0]} to {meta['splits']['holdout'][1]} "
           "(the weeks I traded live).** $100 to start, 15% of the account staked per trade. "
           "Each model only picks which of my strategy's signals to take.", "",
           "| Trade filter | $100 becomes | Sharpe | Sortino | Max drawdown | Trades | Win rate |",
           "|---|---|---|---|---|---|---|"]
    res = bankroll.run_models(d, BLUF_MODELS, "holdout")
    for m, r in sorted(res.items(), key=lambda kv: -kv[1]["final"]):
        name = NAMES.get(m, m)
        name = f"**{name}**" if m == "TabPFN-3.5" else name
        out.append(f"| {name} | ${r['final']:,.2f} | {r['sharpe']} | {r['sortino']} | {r['max_dd_pct']}% | "
                   f"{r['trades']} | {100 * r['win_rate']:.1f}% |")
    tr = {sp: bankroll.run_models(d, ["TabPFN-3.5", "Live gate"], sp) for sp in ("train", "test")}
    lt = pd.read_parquet(DATA / "strategy" / "live_trades.parquet")
    k = lt.tabpfn_keeps
    out += ["", "Same setup on the earlier periods: TabPFN "
            + ", ".join(f"{sp} ${tr[sp]['TabPFN-3.5']['final']:,.2f}" for sp in tr)
            + "; my current filter " + ", ".join(f"{sp} ${tr[sp]['Live gate']['final']:,.2f}" for sp in tr) + ".", "",
            f"With real Kalshi fills: of my {len(lt)} live trades in the holdout, TabPFN would have kept {int(k.sum())}. "
            f"Those earned {lt.real_net_c[k].mean():+.2f}¢ per contract, the ones it would have skipped "
            f"{lt.real_net_c[~k].mean():+.2f}¢.", "",
            "Staking 15% per trade compounds very fast. Read the dollar figures as a comparison between filters, not as "
            "achievable profit: Kalshi's books are too thin for those sizes, and the backtest overstates live results by "
            "a few cents per contract.", ""]
    return out


def bench_block() -> list[str]:
    S = json.loads((ROOT / "results" / "summary.json").read_text())
    out = ["**Public benchmark (generic momentum indicators).** Lower log loss is better. Every model gets the same "
           "5,000 rows; classic models are tuned, TabPFN is not.", ""]
    for mk in MARKETS:
        L = S[mk]["policy_A"]
        if not L or "TabPFN-3.5" not in L["models"]:
            out += [f"*{MARKETS[mk].title}: still running.*", ""]
            continue
        out += [f"*{MARKETS[mk].title}: {L['n_rows']:,} test predictions over {L['n_days']} days*", "",
                "| Model | Log loss | vs TabPFN-3.5 | Significant |", "|---|---|---|---|"]
        for m, r in sorted(L["models"].items(), key=lambda kv: kv[1]["logloss"]):
            v = r.get("vs_ref")
            name = f"**{m}**" if m.startswith("TabPFN") else m.replace(" · tuned", "")
            out.append(f"| {name} | {r['logloss']:.4f} | {'reference' if not v else f'{v['mean']:+.4f}'} | "
                       f"{'' if not v else ('yes' if v.get('bh_pass') else 'no')} |")
        out.append("")
    return out


def main() -> None:
    p = ROOT / "README.md"
    block = "\n".join(bluf_block() + bench_block())
    s = p.read_text()
    p.write_text(re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                        lambda _: "<!-- RESULTS:START -->\n" + block + "\n<!-- RESULTS:END -->", s, flags=re.S))
    print("README results block updated")


if __name__ == "__main__":
    main()
