"""Build the write-up ``docs/index.html`` from data/ and results/.

    python -m tabtrader.report                      # docs/index.html (full page, GitHub Pages)
    python -m tabtrader.report --fragment out.html  # same content without the <html>/<head> wrapper
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .data import MARKETS, ROOT
from .evaluate import fee_c
from .features import FEATURE_GROUPS
from .report_data import BLUF_MODELS, bench_data, bluf_data, eda_data, explorer_data, feature_info, headline
from .report_assets import CSS, JS

E = html.escape
SPLIT_NAME = {"train": "Train", "test": "Test", "holdout": "Holdout"}
LABEL = {"Live gate": "my current filter", "Every signal": "taking every signal"}


def lab(m: str) -> str:
    return LABEL.get(m, m)


def usd(v: float) -> str:
    return f"${v:,.0f}" if abs(v) >= 1000 else f"${v:,.2f}"


def page(fragment: bool = False) -> str:
    S = bench_data()
    H = headline(S)
    bl = bluf_data()
    ed = eda_data()
    fid = json.loads((ROOT / "data" / "fidelity" / "fidelity.json").read_text())
    feat = feature_info()
    ex = explorer_data()
    tot = ed["totals"]
    total_rows = (tot["tape_15m_rows"] + tot["tape_1h_rows"] + tot["order_book_rows"] + tot["spot_venue_rows"]
                  + tot["indicator_rows"] + tot["brti_rows"])
    data = {"ex": ex, "eda": ed, "feat": feat, "bench": S, "bluf": bl, "blufModels": BLUF_MODELS}
    meta = bl["meta"]

    # ---------- computed sentences: the wording follows the numbers
    ho = bl["splits"]["holdout"]
    tab, live_g = ho["TabPFN-3.5"], ho["Live gate"]
    ml_others = {m: v for m, v in ho.items() if m not in ("TabPFN-3.5", "Live gate", "Every signal", "Market price")}
    best_ml = max(ml_others, key=lambda m: ml_others[m]["final"])
    n_mod = len(BLUF_MODELS)
    ranks = {sp: sorted(bl["splits"][sp], key=lambda m: -bl["splits"][sp][m]["final"]).index("TabPFN-3.5") + 1
             for sp in ("train", "test", "holdout")}
    dd_rank = sorted(ho, key=lambda m: -ho[m]["max_dd_pct"]).index("TabPFN-3.5") + 1
    sortino_rank = sorted(ho, key=lambda m: -(ho[m]["sortino"] or -1e9)).index("TabPFN-3.5") + 1
    auc_ho = bl["auc"]["holdout"]
    auc_rank = sorted(auc_ho, key=lambda m: -auc_ho[m]).index("TabPFN-3.5") + 1
    raw_train = bl["splits"]["train"]["Every signal"]["c_per_ct"]
    lvd = bl["live"]
    lv = lvd["series"]
    live_real = lv["Live, as traded (real fills)"]
    live_kept = lv["Live trades TabPFN keeps (real fills)"]
    ordinal = {1: "best", 2: "second best", 3: "third best"}

    def strengths() -> str:
        bits = []
        if dd_rank == 1:
            bits.append(f"the smallest drawdown ({tab['max_dd_pct']:.0f}% vs {live_g['max_dd_pct']:.0f}% for my current filter)")
        if sortino_rank == 1:
            bits.append(f"the best Sortino ratio ({tab['sortino']} vs {live_g['sortino']})")
        if auc_rank == 1:
            bits.append(f"the best ranking of winners (AUC {auc_ho['TabPFN-3.5']:.3f} vs {auc_ho['Live gate']:.3f})")
        return ("TabPFN has " + ", ".join(bits[:-1]) + (" and " if len(bits) > 1 else "") + bits[-1] + ".") if bits else ""

    if tab["final"] >= live_g["final"]:
        vs_live = f"That beats my current filter ({usd(live_g['final'])})"
    else:
        vs_live = f"My current filter makes more ({usd(live_g['final'])})"
    first_ml = tab["final"] >= ml_others[best_ml]["final"]
    bullets = (
        f"<li><b>My real strategy, on the weeks I traded live:</b> $100, staking 15% per trade, ends at "
        f"<b>{usd(tab['final'])}</b> with TabPFN as the trade filter. {vs_live}. "
        + (f"TabPFN beats every other machine-learning filter (the best of them, {E(best_ml)}, makes {usd(ml_others[best_ml]['final'])}). "
           if first_ml else f"{E(best_ml)} makes {usd(ml_others[best_ml]['final'])}. ")
        + strengths() + "</li>"
        f"<li><b>Real fills agree:</b> of my {lvd['n_live']} live trades in that period, TabPFN would have kept {lvd['n_kept']}. "
        f"Those earned {lvd['kept_c']:+.2f}¢ per contract, the ones it would have skipped {lvd['skipped_c']:+.2f}¢.</li>"
        f"<li><b>Across periods:</b> TabPFN's rank by final balance, of {n_mod} filters: "
        + ", ".join(f"{sp} #{ranks[sp]}" for sp in ranks) + "."
        + (f" In the train period the raw signal itself lost money ({raw_train:+.2f}¢ per contract), so every filter lost."
           if raw_train < 0 else "") + "</li>"
    )

    def gen_line(mk: str, name: str) -> tuple[str, bool]:
        h = H[mk]
        if h["rank"] is None:
            return f"<li><b>{name}, generic indicators:</b> still running.</li>", False
        lead = (f"TabPFN-3.5 has the lowest prediction error of {h['n_learners']} models"
                if h["rank"] == 1 else f"TabPFN-3.5 ranks #{h['rank']} of {h['n_learners']} models on prediction error")
        vm = h["vs_market"]
        mkt = ("It also beats the market price." if vm["hi"] < 0 else
               "No model beats the market price itself." if vm["mean"] >= 0 else
               "It edges the market price, within noise.")
        return (f"<li><b>{name}, generic indicators:</b> {lead}. It beats {h['beats']} of {h['n_classic']} tuned "
                f"classic models, {h['sig']} of them by a statistically significant margin. {mkt}</li>",
                h["beats"] == h["n_classic"])

    g15, all15 = gen_line("15m", "15-minute market")
    g1h, all1h = gen_line("1h", "Hourly ladder")
    bullets += g15 + g1h + "<li><b>Everything runs on one CPU.</b> TabPFN needs no tuning, and a weekly refit takes minutes.</li>"
    concl = [
        ("On my real strategy, TabPFN is the strongest machine-learning trade filter I tested" if first_ml else
         "On my real strategy, TabPFN is competitive with the best machine-learning filters")
        + (f", with the {('smallest drawdown' if dd_rank == 1 else 'lowest risk')} on the weeks I traded live. "
           if dd_rank == 1 else ". ")
        + ("My current filter still made more money on those weeks." if tab["final"] < live_g["final"] else
           "It also out-earned my current filter on those weeks."),
        ("On generic momentum indicators, untuned TabPFN-3.5 beats all six tuned classic models, on a CPU."
         if all15 and all1h else
         "On generic momentum indicators, untuned TabPFN-3.5 is at or near the top of seven learners, on a CPU."),
        "The market price is a tough opponent. Most of the value comes from choosing which trades to skip.",
        "Clean, point-in-time data and a measured backtest-to-live gap are what make these numbers trustworthy.",
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
    for mk, name in [("15m", "15-minute"), ("1h", "hourly")]:
        t = S[mk].get("thinking")
        if not t:
            continue
        for m in ["TabPFN-3.5-Thinking", "TabPFN-3.5", "Market price"]:
            if m in t and isinstance(t[m], dict) and "logloss" in t[m]:
                th_rows += (f"<tr class='{'tab' if m == 'TabPFN-3.5' else 'mkt' if m == 'Market price' else ''}'>"
                            f"<td>{name}</td><td>{E(m)}</td><td class='num'>{t[m]['logloss']:.4f}</td>"
                            f"<td class='num'>{t[m]['auc']:.4f}</td></tr>")
    ctrl = S.get("_controls")
    ctrl_txt = ""
    if ctrl:
        planted_ok = all(v["minus_market"]["hi"] < 0 for v in ctrl["planted"].values() if v["minus_market"])
        null_ok = all(v["minus_market"]["lo"] > -0.002 or v["minus_market"]["hi"] >= 0
                      for v in ctrl["null"].values() if v["minus_market"])
        ctrl_txt = (f"<h3>Sanity checks</h3><p>With a planted signal, every model "
                    f"{'finds it' if planted_ok else 'does not always find it'}. With labels drawn from the market price "
                    f"itself, {'no model finds fake skill' if null_ok else 'a model shows fake skill'}. "
                    f"{'So the test setup can tell signal from noise.' if planted_ok and null_ok else ''}</p>")

    sp = meta["splits"]
    head = "<title>tab-trader</title><style>" + CSS + "</style>"
    body = f"""
<div class="top"><div class="wrap">
<div class="kicker">TabPFN-3.5 Hackathon · tab-trader</div>
<h1>TabPFN-3.5 on Kalshi Bitcoin contracts</h1>
<p class="sub">{total_rows/1e6:,.0f} million rows of my own market data, recorded since {tot['first_day']}. One CPU.
TabPFN-3.5 against six tuned machine-learning models, my current live filter, and the market price.</p>

<div class="bluf"><h2>Bottom line</h2><ul>{bullets}</ul></div>

<h2>1. My real strategy: $100, 15% of the account per trade</h2>
<div class="note"><b>Read this first.</b> This section uses my live strategy's real signals and its own private features.
Each model only decides which signals to take. The features and the exact approach are not published. Every section
after this one uses generic, off-the-shelf momentum indicators instead.</div>
<p>Each model is trained on <b>Train</b> ({sp['train'][0]} to {sp['train'][1]}). <b>Test</b> ({sp['test'][0]} to {sp['test'][1]})
is used once, to set each model's trade rule. <b>Holdout</b> ({sp['holdout'][0]} to {sp['holdout'][1]}) is scored with everything
frozen, and it is exactly the period I traded live. Train results are out-of-fold, so no model is scored on data it was fit on.</p>
<figure><div class="legend" id="eq-legend"></div>
<div class="panels"><div><h4>Train · {meta['n']['train']:,} signals</h4><div id="eq-train"></div></div>
<div><h4>Test · {meta['n']['test']:,} signals</h4><div id="eq-test"></div></div>
<div><h4>Holdout · {meta['n']['holdout']:,} signals</h4><div id="eq-holdout"></div></div></div>
<p class="what"><b>What it shows:</b> the account value on a log scale, starting from $100 in each period. Higher is better.
TabPFN is the navy line.</p></figure>
<div class="note"><b>How to read the dollar figures.</b> Staking 15% of the account on every trade compounds very fast, in both
directions. The dollar amounts compare the filters; they are not profits anyone could collect. Kalshi's order books are far
too thin for those sizes, and the backtest overstates live results by about {gap:.1f}¢ per contract (see "Backtest vs live").
The ¢ per contract column is the size-free measure.</div>

<h3>Tear sheet</h3>
<figure><div class="controls"><div id="ts-split"></div></div><div class="tw" id="ts-table"></div>
<p class="what"><b>What it shows:</b> the same runs as numbers. Sharpe and Sortino use daily returns annualised over 365 days.
Results are per contract after Kalshi fees, using the strategy's normal exits.</p></figure>
<figure><div id="wk-bars"></div><div class="legend"><span><i style="border-top-color:var(--c1)"></i>TabPFN-3.5</span><span><i style="border-top-color:var(--classic)"></i>My current filter</span></div>
<p class="what"><b>What it shows:</b> each week's return, TabPFN next to my current filter. Each period restarts at $100.</p></figure>

<h3>Zoom: the live period since {bl['live']['first']}</h3>
<p>I took {bl['live']['n_live']} real trades in this period. TabPFN would have kept {bl['live']['n_kept']} of them. The kept trades
earned {bl['live']['kept_c']:+.2f}¢ per contract with real fills. The ones TabPFN would have skipped earned {bl['live']['skipped_c']:+.2f}¢.</p>
<figure><div class="legend" id="live-legend"></div><div id="live-chart"></div>
<p class="what"><b>What it shows:</b> $100 at 15% per trade over the live weeks. Solid lines use real Kalshi fills: gray is what I
actually traded ({usd(live_real['final'])} at the end), navy is the same trades with TabPFN's veto ({usd(live_kept['final'])}).
Dashed lines are each filter's backtest on every signal.</p></figure>

<h2>2. The idea</h2>
<p>A Kalshi contract pays $1 if something happens, so its price is the crowd's probability. A model makes money only when its
probability beats the price by more than the fee. That makes the market price one of the competitors on every chart.</p>

<h2>3. The markets</h2>
<div class="two"><div><h3>15-minute up/down (KXBTC15M)</h3><p>Will BTC be higher at the end of this 15-minute window than at the start?
96 markets a day.</p></div><div><h3>Hourly strike ladder (KXBTCD)</h3><p>Will BTC be above $K at the top of the hour? One contract per strike.</p></div></div>
<p>Both settle on the CF Benchmarks BRTI index. Buying costs the ask plus a fee of 0.07 × price × (1 − price), which is
{fee_c(50):.2f}¢ for a 50¢ contract.</p>
<figure><div class="controls"><div id="ex-mode"></div><label for="ex-day">day</label><select id="ex-day"></select><label for="ex-market">market</label><select id="ex-market"></select></div>
<div id="ex-chart"></div><div id="ex-chart2"></div><p class="what" id="ex-cap"></p></figure>

<h2>4. The data</h2>
<div class="tiles">
<div class="tile"><div class="v">{tot['tape_15m_rows']/1e6:.1f}M</div><div class="k">15-minute market rows (about 4 per second)</div></div>
<div class="tile"><div class="v">{tot['tape_1h_rows']/1e6:.1f}M</div><div class="k">hourly ladder rows</div></div>
<div class="tile"><div class="v">{tot['order_book_rows']/1e6:.0f}M</div><div class="k">full order-book snapshots</div></div>
<div class="tile"><div class="v">{tot['labels_15m']:,}</div><div class="k">settled 15-minute markets</div></div>
</div>
<p>Kalshi keeps only about two months of history, so most of this exists only because I recorded it live.</p>
<div class="tw"><table><thead><tr><th>Since</th><th>Recorded</th></tr></thead><tbody>{milestones}</tbody></table></div>
<figure><div id="eda-daily"></div><p class="what"><b>What it shows:</b> rows recorded per day. Collection has run since March.</p></figure>
<figure><div id="eda-q"></div><p class="what"><b>What it shows:</b> the share of clean quotes each week. A collector upgrade on
30 July lifted it from about 60% to over 95%. Only clean, exchange-confirmed quotes are used for modelling.</p></figure>
<figure><div class="controls"><div id="eda-cal-ctl"></div></div><div id="eda-cal"></div>
<p class="what"><b>What it shows:</b> when the market says 70%, the contract settles YES about 70% of the time. The crowd is already
accurate, so beating it is hard.</p></figure>

<h2>5. Backtest vs live</h2>
<div class="tiles">
<div class="tile"><div class="v">{100*vt['live_trades_also_taken_by_twin']:.0f}%</div><div class="k">of live trades match the paper twin exactly</div></div>
<div class="tile"><div class="v">{100*vb['live_trades_with_backtest_signal']:.1f}%</div><div class="k">of live trades also appear in the backtest</div></div>
<div class="tile"><div class="v">{100*vb['same_outcome_settled']:.1f}%</div><div class="k">same outcome, backtest vs live</div></div>
<div class="tile"><div class="v">{ex_['entry_slippage_c_median']:+.1f}¢</div><div class="k">median fill vs quoted price</div></div>
</div>
<div class="tw"><table><thead><tr><th>The same trades, measured four ways</th><th>n</th><th>¢ per contract</th><th>95% range</th></tr></thead><tbody>{fid_rows}</tbody></table></div>
<p>The backtest picks the same trades as live. The gap is price: live orders arrive about {vb['entry_time_diff_s_median']:.0f} s
later and pay a little more, so the backtest overstates profit by about {gap:.1f}¢ per contract. That gap is measured and planned for.</p>

<h2>6. Generic features</h2>
<p>From here on, every model sees the same {sum(len(v) for v in FEATURE_GROUPS.values())} standard indicators at textbook settings.
None is tuned, and none is one of my strategy's features.</p>
<div class="tw"><table><thead><tr><th>Group</th><th>Features</th></tr></thead><tbody>{feat_rows}</tbody></table></div>
<p class="small">Each row is one market at one decision time: 10 and 5 minutes before close for the 15-minute market, 30 and 15 minutes
for the hourly ladder. Every price is taken at a single instant and checked against Kalshi's own records, and only BTC bars that had
already closed are used.</p>
<figure><div id="feat-rc"></div><p class="what"><b>What it shows:</b> how much each indicator knows that the price does not. Every bar is
tiny, so the signal is weak and noisy. That is where a pretrained model like TabPFN should help most.</p></figure>

<h2>7. The benchmark</h2>
<p>Weekly walk-forward tests: 9 weeks for the 15-minute market and 6 for the hourly ladder. Each week, every model trains only on earlier
markets and gets the same 5,000 most recent rows. The six classic models are tuned (best of their defaults and 20 random settings);
TabPFN is not. Lower log loss means better probabilities. The test plan was written down before it ran
(<a href="https://github.com/Cloblak/tab-trader/blob/main/PREREG.md">PREREG.md</a>).</p>
<figure><div class="controls"><div id="lb-mk"></div><div id="lb-pol"></div></div><div id="lb-dots"></div>
<p class="what" id="lb-cap"></p><div class="tw" id="lb-table"></div></figure>
<h3>How much data each model needs</h3>
<figure><div class="controls"><div id="lc-mk"></div></div><div id="lc-chart"></div>
<p class="what"><b>What it shows:</b> prediction error as the training set grows. Lower is better. The dashed line is the market.</p></figure>
<h3>Are the probabilities honest?</h3>
<figure><div class="controls"><div id="rel-mk"></div></div><div class="legend" id="rel-legend"></div><div id="rel-chart"></div>
<p class="what"><b>What it shows:</b> predicted probability vs what actually happened. On the diagonal means honest.</p></figure>
<h3>Cost</h3>
<figure><div class="controls"><div id="cpu-mk"></div></div><div id="cpu-chart"></div>
<p class="what"><b>What it shows:</b> CPU time per weekly refit, including tuning, vs prediction error. Bottom-left is best.</p></figure>
<h3>TabPFN-3.5 Thinking (API)</h3>
<p>Thinking mode runs on Prior Labs' servers and spends extra compute on each fit. It ran on the same test weeks.</p>
<div class="tw"><table><thead><tr><th>Market</th><th>Model</th><th>Log loss</th><th>AUC</th></tr></thead><tbody>{th_rows}</tbody></table></div>
{ctrl_txt}

<h2>8. Conclusion</h2>
<ul>{''.join(f'<li>{c}</li>' for c in concl)}</ul>
<h3>Reproduce</h3>
<pre>git clone https://github.com/Cloblak/tab-trader && cd tab-trader
uv sync
export TABPFN_TOKEN=...      # free at https://ux.priorlabs.ai
make quick                   # one test week per market, about 10 minutes on CPU
make benchmark               # everything (a few hours)
make analyze report          # rebuild this page</pre>
<p class="small">My earlier research also looked at section 1's test and holdout weeks, so a forward test is the next step.
TabPFN-3.5's open weights are licensed for evaluation. Live trading with them needs a commercial licence or the Prior Labs API.</p>
</div></div>
<script type="application/json" id="data">{json.dumps(data, separators=(',', ':'), default=str)}</script>
<script>{JS}</script>
"""
    if fragment:
        return head + body
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1,viewport-fit=cover\">"
            + head + "</head><body>" + body + "</body></html>")


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
