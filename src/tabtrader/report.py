"""Build the write-up ``docs/index.html`` from data/ and results/.

    python -m tabtrader.report                      # docs/index.html (full page, GitHub Pages)
    python -m tabtrader.report --fragment out.html  # same content without the <html>/<head> wrapper
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .data import ROOT
from .evaluate import fee_c
from .features import FEATURE_GROUPS
from .report_assets import CSS, JS
from .report_data import (ROLL_MODELS, bench_data, eda_data, explorer_data, feature_info, headline,
                          rolling_data)

E = html.escape


def usd(v: float) -> str:
    return f"${v:,.0f}" if abs(v) >= 1000 else f"${v:,.2f}"


def page(fragment: bool = False) -> str:
    S = bench_data()
    H = headline(S)
    ro = rolling_data()
    ed = eda_data()
    fid = json.loads((ROOT / "data" / "fidelity" / "fidelity.json").read_text())
    feat = feature_info()
    ex = explorer_data()
    tot = ed["totals"]
    total_rows = (tot["tape_15m_rows"] + tot["tape_1h_rows"] + tot["order_book_rows"] + tot["spot_venue_rows"]
                  + tot["indicator_rows"] + tot["brti_rows"])
    data = {"ex": ex, "eda": ed, "feat": feat, "bench": S, "roll": ro, "rollModels": ROLL_MODELS}
    meta, res, live = ro["meta"], ro["res"], ro["live"]

    # ---------- computed sentences: the wording follows the numbers
    lv = live["series"]
    real, veto = lv["Live, as traded (real fills)"], lv["TabPFN veto, flat 15% (real fills)"]
    auc = ro["auc"]
    best_auc = max(auc, key=auc.get)
    top = max(res, key=lambda m: res[m]["final"])
    jul = next((m for m in ro["monthly"] if m["month"] == "2026-07"), None)
    names = {"TabPFN-3.5": "TabPFN-3.5 at a flat 15%", "TabPFN-3.5 + Kelly": "TabPFN-3.5 with half-Kelly stakes",
             "Logistic regression + Kelly": "logistic regression with half-Kelly stakes", "Live gate": "my current filter",
             "Logistic regression": "logistic regression at a flat 15%", "LightGBM": "LightGBM at a flat 15%",
             "LightGBM + Kelly": "LightGBM with half-Kelly stakes", "Every signal": "taking every signal"}
    lead = (f"<li><b>It ranks my live strategy's signals best.</b> In a weekly walk-forward ({meta['weeks']} weeks, "
            f"{meta['signals']:,} signals), TabPFN-3.5 scores AUC {auc['TabPFN-3.5']:.3f}, against "
            f"{auc['Logistic regression']:.3f} for logistic regression and {auc['LightGBM']:.3f} for LightGBM, with no tuning."
            "</li>" if best_auc == "TabPFN-3.5" else "")
    lead += (f"<li><b>Its veto works on real fills.</b> Of my {live['n']} actual live trades since {live['first']}, the "
             f"{live['kept']} TabPFN would have kept earned {live['kept_c']:+.2f}¢ per contract; the rest earned "
             f"{live['skipped_c']:+.2f}¢. At 15% per trade, $100 ends at {usd(veto['final'])} with its veto, against "
             f"{usd(real['final'])} as I traded.</li>")
    tab_flat, tab_k = res["TabPFN-3.5"], res["TabPFN-3.5 + Kelly"]
    lead_tail = (f"<li><b>The long walk-forward is a tough test.</b> From {meta['first_week']} to {meta['last_signal']}, "
                 f"{names.get(top, top)} finished highest ({usd(res[top]['final'])}); TabPFN ends at {usd(tab_k['final'])} with "
                 f"half-Kelly stakes and {usd(tab_flat['final'])} at a flat 15%."
                 + (f" July decided it: my signal lost {abs(jul['Every signal']['c']):.2f}¢ per contract that month, and "
                    f"TabPFN, still learning from April to June, did not step aside." if jul else "") + "</li>")

    def gen_line(mk: str, label: str) -> str:
        h = H[mk]
        if h["rank"] is None:
            return ""
        lead_ = (f"TabPFN-3.5 has the lowest prediction error of {h['n_learners']} models" if h["rank"] == 1
                 else f"TabPFN-3.5 ranks #{h['rank']} of {h['n_learners']} models on prediction error")
        return f"{label}: {lead_}, significantly ahead of {h['sig']} of {h['n_classic']} tuned classic models"
    bench_bits = [b for b in (gen_line("15m", "15-minute market"), gen_line("1h", "hourly ladder")) if b]
    lead += ("<li><b>Public benchmark</b> (generic indicators, rerunnable from this repo): " + "; ".join(bench_bits)
             + ". Nothing beats the market price itself on prediction error.</li>")
    lead += lead_tail
    lead += "<li><b>One CPU, no tuning.</b> A weekly refit takes under a minute; scoring a live signal about a second.</li>"
    concl = [
        "TabPFN-3.5 works as a practical model on messy, real market data: refit weekly on a few thousand rows, no tuning, one CPU.",
        f"It ranked my strategy's signals best of the models tested, and its veto on my real live trades kept the trades "
        f"that made money ({live['kept_c']:+.2f}¢ vs {live['skipped_c']:+.2f}¢ per contract).",
        "It did not escape a regime change on its own: in July it kept trading a signal that had stopped working. Sizing "
        "matters as much as the model, and half-Kelly stakes protected every filter.",
        "On generic indicators it is at or near the top of seven learners, and nothing beats the market price on prediction error.",
    ]

    vb, vt, ex_ = fid["vs_backtest"], fid["vs_paper_twin"], fid["execution"]
    cpc = fid["cents_per_contract_same_trades"]
    gap = cpc["backtest (recorded quotes, live exits)"]["c_per_ct"] - cpc["live real (Kalshi fills and fees)"]["c_per_ct"]
    fid_rows = "".join(
        f"<tr><td>{E(k)}</td><td class='num'>{v['n']}</td><td class='num'>{v['c_per_ct']:+.2f}¢</td>"
        f"<td class='num'>[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]</td></tr>" for k, v in cpc.items())
    feat_rows = "".join(f"<tr><td>{E(g.replace('_', ' '))}</td><td>{', '.join(E(f) for f in fs)}</td></tr>"
                        for g, fs in FEATURE_GROUPS.items())
    milestones = "".join(f"<tr><td class='num'>{E(d)}</td><td>{E(t)}</td></tr>" for d, t in ed["milestones"])
    th_rows = ""
    for mk, mname in [("15m", "15-minute"), ("1h", "hourly")]:
        t = S[mk].get("thinking") or {}
        for m in ["TabPFN-3.5-Thinking", "TabPFN-3.5", "Market price"]:
            if m in t and isinstance(t[m], dict) and "logloss" in t[m]:
                th_rows += (f"<tr class='{'tab' if m == 'TabPFN-3.5' else ''}'><td>{mname}</td><td>{E(m)}</td>"
                            f"<td class='num'>{t[m]['logloss']:.4f}</td><td class='num'>{t[m]['auc']:.4f}</td></tr>")
    ctrl = S.get("_controls")
    ctrl_txt = ""
    if ctrl:
        found = [m.replace(" · default", "") for m, v in ctrl["planted"].items() if v["minus_market"] and v["minus_market"]["hi"] < 0]
        missed = [m.replace(" · default", "") for m, v in ctrl["planted"].items() if v["minus_market"] and v["minus_market"]["hi"] >= 0]
        fake = [m.replace(" · default", "") for m, v in ctrl["null"].items() if v["minus_market"] and v["minus_market"]["hi"] < 0]
        ctrl_txt = ("<h3>Sanity checks</h3><p>A planted signal was found by " + " and ".join(E(m) for m in found)
                    + (f" ({', '.join(E(m) for m in missed)} missed it with 5,000 rows)" if missed else "")
                    + ". With labels drawn from the market price itself, "
                    + ("no model found fake skill." if not fake else f"{', '.join(fake)} showed fake skill.") + "</p>")

    hr = json.loads((ROOT / "results" / "holdout_api.json").read_text())
    ho_rows = ""
    for m, v in sorted(hr["models"].items(), key=lambda kv: kv[1]["holdout"]["logloss"]):
        t = v["trading_holdout"]
        tr = (f"<td class='num'>{t['trades']}</td><td class='num'>{t['c_per_ct']:+.2f} [{t['ci95'][0]:+.1f}, {t['ci95'][1]:+.1f}]</td>"
              if t.get("trades") else "<td class='num'>0</td><td>no trade cleared price + fee</td>")
        ho_rows += (f"<tr class='{'tab' if m.startswith('TabPFN') else ''}'><td>{E(m)}</td><td class='num'>"
                    f"{v['holdout']['logloss']:.4f}</td><td class='num'>{v['holdout']['auc']:.4f}</td>{tr}</tr>")
    import pandas as _pd
    from . import bankroll as _bk
    bd = _pd.read_parquet(ROOT / "data" / "strategy" / "bluf_predictions.parquet")
    srows = ["TabPFN-3.5 + Kelly", "TabPFN-3.5", "Live gate", "Random forest", "Logistic regression", "Every signal"]
    sres = {sp_: _bk.run_models(bd, srows, sp_) for sp_ in ("train", "test", "holdout")}
    snames = {"TabPFN-3.5 + Kelly": "TabPFN-3.5, half-Kelly", "TabPFN-3.5": "TabPFN-3.5, flat 15%",
              "Live gate": "My current filter", "Every signal": "Take every signal"}
    split_rows = "".join(
        f"<tr class='{'tab' if m.startswith('TabPFN') else ''}'><td>{E(snames.get(m, m))}</td>"
        + "".join(f"<td class='num'>{usd(sres[sp_][m]['final'])}</td>" for sp_ in ("train", "test", "holdout")) + "</tr>"
        for m in srows)
    bt = ro.get("boot")
    boot_txt = (f" {E(names.get(bt['top'], bt['top']))[0].upper() + E(names.get(bt['top'], bt['top']))[1:]} ends "
                f"{res[bt['top']]['final'] / res['TabPFN-3.5 + Kelly']['final']:.1f}× higher than TabPFN-3.5 at half-Kelly, but the "
                f"95% range of that ratio runs from {bt['lo']:.2f}× to {bt['hi']:.1f}×, so the two are not distinguishable."
                if bt else "")
    head = "<title>tab-trader</title><style>" + CSS + "</style>"
    body = f"""
<div class="top"><div class="wrap">
<div class="kicker">TabPFN-3.5 Hackathon · tab-trader</div>
<h1>TabPFN-3.5 on Kalshi's 15-minute Bitcoin markets</h1>
<p class="sub">Real, messy, self-recorded market data ({total_rows/1e6:,.0f} million rows since {tot['first_day']}).
TabPFN decides which trades to take. One CPU, no training loop, no tuning.</p>

<div class="bluf"><h2>Bottom line</h2><ul>{lead}</ul></div>

<h2>1. The strategy: TabPFN as a weekly-refit trade filter</h2>
<p>The table and chart below show every filter, including the ones that beat TabPFN.</p>
<div class="note"><b>Private features.</b> This section uses my live strategy's real signals and its own 32 features, which are not
published. The repo contains each model's scores, decisions and outcomes, so the results can be checked. The public-data
results (sections 2, 3 and 6) use only generic indicators.</div>
<ol>
<li>A momentum signal from my live strategy proposes a trade a few times an hour.</li>
<li>Every Monday, TabPFN-3.5 reads the previous {meta['context_weeks']} weeks of signals and their outcomes as context. It is not
trained or tuned.</li>
<li>For each new signal it returns the probability of winning. The trade is taken only if that probability beats the entry price
plus Kalshi's fee.</li>
<li>Stake: a flat 15% of the account, or half the Kelly stake implied by TabPFN's probability, capped at 15%.</li>
</ol>
<p>The same weekly refit and the same rule are applied to logistic regression and LightGBM. My current filter is a model trained once
in spring and then frozen. The rules were written down before the test ran
(<a href="https://github.com/Cloblak/tab-trader/blob/main/PREREG.md">PREREG.md</a>, amendment 3).</p>
<figure><div class="legend" id="eq-legend"></div><div id="eq-roll"></div>
<p class="what"><b>What it shows:</b> account value from $100, on a log scale, over {meta['weeks']} test weeks. Higher is better.</p></figure>
<figure><div class="tw" id="ts-table"></div>
<p class="what"><b>What it shows:</b> the same runs as numbers. "Calibrated" rows add an isotonic correction learned from each model's
own predictions in the previous four weeks. Sharpe and Sortino use daily returns annualised over 365 days. The 95% ranges come from
resampling whole days.{boot_txt}</p></figure>
<figure><div id="mo-bars"></div><div class="legend"><span><i style="border-top-color:var(--classic)"></i>Take every signal</span><span><i style="border-top-color:var(--c1)"></i>TabPFN-3.5</span></div>
<p class="what"><b>What it shows:</b> profit per contract by month, for every signal and for the signals TabPFN took. TabPFN added
value in August and September; in July, when the signal broke down, it did not.</p></figure>
<h3>Zoom: real fills since {live['first']}</h3>
<figure><div class="legend" id="live-legend"></div><div id="live-chart"></div>
<p class="what"><b>What it shows:</b> my actual live trades with real Kalshi fills. Gray is what I traded; navy is the same trades with
TabPFN's veto ({usd(veto['final'])} vs {usd(real['final'])} at the end).</p></figure>
<div class="note"><b>How to read the dollar figures.</b> Each trade stakes 15% of the balance (or half-Kelly) but never more than
500 contracts, a stand-in for the depth of Kalshi's order book, so balances grow roughly linearly past a few thousand dollars. The
dollar amounts compare the filters. The backtest also overstates live results by about {gap:.1f}¢ per contract (section 5); the ¢ per
contract column is the size-free measure.</div>

<h2>2. Train, test, holdout: the playground recipe</h2>
<p>The same steps as the Prior Labs playground, with one change that matters for markets: the split is by time. A random
split would let the model learn from the future. The table it runs on is public and playground-ready
(<code>data/tabpfn_ready/kxbtc15m_features.csv</code>).</p>
<pre>model = TabPFNClassifier(model_path="v3.5_default", n_estimators=8)
model.fit(train[features], train["target"])          # Jun 7 – Aug 2
p_test = model.predict_proba(test[features])[:, 1]   # Aug 3 – Sep 6: pick the trading margin, once
p_hold = model.predict_proba(holdout[features])[:, 1]  # Sep 7 – Oct 3: scored once</pre>
<div class="tw"><table><thead><tr><th>Public data, holdout</th><th>Log loss</th><th>AUC</th><th>Trades</th><th>¢ per contract [95% CI]</th></tr></thead><tbody>{ho_rows}</tbody></table></div>
<p>With generic indicators the market price stays ahead. On my strategy's signals, where the features carry real information,
the same three-way split looks like this (local TabPFN-3.5; train scores are out-of-fold):</p>
<div class="tw"><table><thead><tr><th>$100 becomes</th><th>Train (Apr 15 – Jul 13)</th><th>Test (Jul 14 – Sep 11)</th><th>Holdout (Sep 12 – Oct 2)</th></tr></thead><tbody>{split_rows}</tbody></table></div>
<p class="small">In spring my signal itself lost money, so every filter lost in the train period. The half-Kelly rule was picked among
five TabPFN variants by test-period Sharpe with the holdout visible, so treat that row as indicative; the weekly walk-forward in
section 1 is the stricter test.</p>

<h2>3. The data: real and messy</h2>
<div class="tiles">
<div class="tile"><div class="v">{tot['tape_15m_rows']/1e6:.1f}M</div><div class="k">15-minute market rows, about 4 per second</div></div>
<div class="tile"><div class="v">{tot['order_book_rows']/1e6:.0f}M</div><div class="k">full order-book snapshots</div></div>
<div class="tile"><div class="v">{tot['labels_15m']:,}</div><div class="k">settled 15-minute markets</div></div>
<div class="tile"><div class="v">{tot['first_day']}</div><div class="k">recording started</div></div>
</div>
<p>Kalshi keeps only about two months of history, so this record exists only because it was captured live. It is also messy:
crossed books, stale quotes, outages and a collector rewrite on 30 July. Every price used for modelling is taken at a single instant and
checked against Kalshi's own 1-minute candles.</p>
<figure><div id="eda-q"></div><p class="what"><b>What it shows:</b> the share of clean quotes each week. The collector upgrade lifted it
from about 60% to over 95%.</p></figure>
<figure><div class="controls"><div id="eda-cal-ctl"></div></div><div id="eda-cal"></div>
<p class="what"><b>What it shows:</b> when the market says 70%, the contract settles YES about 70% of the time. The crowd is accurate, so
beating its price is hard.</p></figure>
<details><summary>Collection timeline and daily volume</summary>
<div class="tw"><table><thead><tr><th>Since</th><th>Recorded</th></tr></thead><tbody>{milestones}</tbody></table></div>
<figure><div id="eda-daily"></div><p class="what"><b>What it shows:</b> rows recorded per day.</p></figure></details>

<h2>4. The markets</h2>
<p>A KXBTC15M contract asks whether BTC will finish a 15-minute window higher than it started; KXBTCD asks whether BTC will be above a
strike at the top of the hour. Both pay $1 and settle on the CF Benchmarks BRTI index. Buying costs the ask plus a fee of
0.07 × price × (1 − price), {fee_c(50):.2f}¢ at 50¢.</p>
<figure><div class="controls"><div id="ex-mode"></div><label for="ex-day">day</label><select id="ex-day"></select><label for="ex-market">market</label><select id="ex-market"></select></div>
<div id="ex-chart"></div><div id="ex-chart2"></div><p class="what" id="ex-cap"></p></figure>

<h2>5. Backtest vs live</h2>
<div class="tiles">
<div class="tile"><div class="v">{100*vt['live_trades_also_taken_by_twin']:.0f}%</div><div class="k">of live trades match the paper twin exactly</div></div>
<div class="tile"><div class="v">{100*vb['live_trades_with_backtest_signal']:.1f}%</div><div class="k">of live trades also appear in the backtest</div></div>
<div class="tile"><div class="v">{100*vb['same_outcome_settled']:.1f}%</div><div class="k">same outcome, backtest vs live</div></div>
<div class="tile"><div class="v">{ex_['entry_slippage_c_median']:+.1f}¢</div><div class="k">median fill vs quoted price</div></div>
</div>
<div class="tw"><table><thead><tr><th>The same trades, measured four ways</th><th>n</th><th>¢ per contract</th><th>95% range</th></tr></thead><tbody>{fid_rows}</tbody></table></div>
<p>The backtest picks the same trades as live. The gap is price: live orders arrive about {vb['entry_time_diff_s_median']:.0f} s
later and pay a little more, so the backtest overstates profit by about {gap:.1f}¢ per contract.</p>

<h2>6. Public benchmark: generic indicators, anyone can rerun it</h2>
<p>Every model gets the same {sum(len(v) for v in FEATURE_GROUPS.values())} textbook indicators and the same 5,000 most recent rows,
in weekly walk-forward tests (9 weeks on the 15-minute market, 6 on the hourly ladder). The six classic models are tuned; TabPFN is
not. Lower log loss means better probabilities.</p>
<details><summary>The indicators</summary><div class="tw"><table><thead><tr><th>Group</th><th>Features</th></tr></thead><tbody>{feat_rows}</tbody></table></div>
<figure><div id="feat-rc"></div><p class="what"><b>What it shows:</b> how much each indicator knows that the price does not. All of them
are weak.</p></figure></details>
<figure><div class="controls"><div id="lb-mk"></div><div id="lb-pol"></div></div><div id="lb-dots"></div>
<p class="what" id="lb-cap"></p><div class="tw" id="lb-table"></div></figure>
<h3>How much data each model needs</h3>
<figure><div class="controls"><div id="lc-mk"></div></div><div id="lc-chart"></div>
<p class="what"><b>What it shows:</b> prediction error as the training set grows. Lower is better.</p></figure>
<h3>Are the probabilities honest?</h3>
<figure><div class="controls"><div id="rel-mk"></div></div><div class="legend" id="rel-legend"></div><div id="rel-chart"></div>
<p class="what"><b>What it shows:</b> predicted probability vs what actually happened. On the diagonal means honest.</p></figure>
<h3>Cost</h3>
<figure><div class="controls"><div id="cpu-mk"></div></div><div id="cpu-chart"></div>
<p class="what"><b>What it shows:</b> CPU time per weekly refit, including tuning, vs prediction error. Bottom-left is best.</p></figure>
<h3>TabPFN-3.5 Thinking (API)</h3>
<div class="tw"><table><thead><tr><th>Market</th><th>Model</th><th>Log loss</th><th>AUC</th></tr></thead><tbody>{th_rows}</tbody></table></div>
{ctrl_txt}

<h2>7. Conclusion</h2>
<ul>{''.join(f'<li>{c}</li>' for c in concl)}</ul>
<h3>About the data</h3>
<p>The repo publishes everything needed to rerun the public benchmark, and the scores, decisions and outcomes behind section 1. The full
dataset (about {total_rows/1e6:,.0f} million rows of tape, order book, spot and settlement index since {tot['first_day']}) is mine and is the
basis of my trading strategies, so it is not public. If you would like to work with it, open an issue on the repo or reach me through
<a href="https://github.com/Cloblak">my GitHub profile</a>.</p>
<h3>Reproduce</h3>
<pre>git clone https://github.com/Cloblak/tab-trader && cd tab-trader
uv sync
export TABPFN_TOKEN=...      # free at https://ux.priorlabs.ai
make quick                   # one test week per market, about 10 minutes on CPU
make benchmark               # everything (a few hours)
make analyze report          # rebuild this page</pre>
<p class="small">Earlier research of mine looked at some of section 1's weeks, so a live forward test is the next step. TabPFN-3.5's open
weights are licensed for evaluation; live trading with them needs a commercial licence or the Prior Labs API.</p>
</div></div>
<script type="application/json" id="data">{json.dumps(data, separators=(',', ':'), default=str)}</script>
<script>{JS}</script>
"""
    if fragment:
        return head + body
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">"
            + head + "</head><body>" + body + "</body></html>")


# Conclusion bullets are set after reading the results (kept in one place so they are easy to check).
CONCLUSION: list[str] = []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fragment")
    args = ap.parse_args()
    out = ROOT / "docs" / "index.html"
    out.write_text(page())
    print("wrote", out, f"{out.stat().st_size/1e6:.2f} MB")
    if args.fragment:
        Path(args.fragment).write_text(page(fragment=True))
        print("wrote", args.fragment)


if __name__ == "__main__":
    main()
