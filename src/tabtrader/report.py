"""Build the interactive write-up ``docs/index.html`` from data/ and results/.

    python -m tabtrader.report            # docs/index.html (full page for GitHub Pages)
    python -m tabtrader.report --fragment out.html   # same content without <html>/<head> wrapper
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import eda
from .data import DATA, MARKETS, ROOT
from .evaluate import fee_c
from .features import FEATURE_GROUPS, FEATURES

E = html.escape
TABPFN_NAMES = ("TabPFN-3.5", "TabPFN-3.5-Fast", "TabPFN-3.5-Thinking")


def r(x, k=2):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)


# ----------------------------------------------------------------- data blocks
def explorer_data() -> dict:
    raw = DATA / "raw_week"
    q = pd.read_parquet(raw / "kxbtc15m_quotes_1s.parquet")
    s = pd.read_parquet(raw / "settlements_15m.parquet")
    q = q[q.quote_valid == 1].copy()
    q["t"] = q.sec_end.dt.floor("15s")
    g = q.groupby(["market", "t"]).agg(bid=("bid", "last"), ask=("ask", "last"), btc=("btc", "last")).reset_index()
    out15 = []
    for _, m in s.sort_values("close_time").iterrows():
        x = g[g.market == m.market]
        if len(x) < 20:
            continue
        t0 = m.open_time
        out15.append({
            "m": m.market, "open": m.open_time.strftime("%Y-%m-%d %H:%M"), "k": r(m.strike, 2), "y": int(m.label),
            "s": [int((t - t0).total_seconds()) for t in x.t],
            "b": [r(v, 1) for v in x.bid], "a": [r(v, 1) for v in x.ask], "x": [r(v, 0) for v in x.btc],
        })
    qh = pd.read_parquet(raw / "kxbtcd_quotes_10s.parquet")
    sh = pd.read_parquet(raw / "settlements_1h.parquet")
    qh = qh[qh.quote_valid == 1].copy()
    qh["t"] = qh.sec_end.dt.floor("1min")
    gh = qh.groupby(["market", "t"]).agg(bid=("bid", "last"), ask=("ask", "last")).reset_index()
    outh = []
    for ct, legs in sh.groupby("close_time"):
        lines = []
        for _, m in legs.sort_values("strike").iterrows():
            x = gh[gh.market == m.market]
            if len(x) < 10:
                continue
            lines.append({"k": r(m.strike, 2), "y": int(m.label),
                          "s": [int((t - m.open_time).total_seconds() // 60) for t in x.t],
                          "p": [r((b + a) / 2, 1) for b, a in zip(x.bid, x.ask)]})
        if len(lines) >= 3:
            outh.append({"close": ct.strftime("%Y-%m-%d %H:%M"), "legs": lines})
    return {"m15": out15, "m1h": outh}


def eda_data() -> dict:
    t = eda.eda_tables()
    v = eda.volume_totals(t)
    d15, d1h = t["tape_15m_daily"], t["tape_1h_daily"]
    w = eda.weekly_quality(t)
    out = {
        "totals": v, "milestones": eda.MILESTONES,
        "daily": {"15m": [[str(a.date()), int(b)] for a, b in zip(d15.day, d15.rows)],
                  "1h": [[str(a.date()), int(b)] for a, b in zip(d1h.day, d1h.rows)]},
        "weekly": {k: [[str(a.date()), r(b, 2), r(c, 2)] for a, b, c in zip(g.week, g.valid_pct, g.crossed_pct)]
                   for k, g in w.groupby("market")},
        "candle": {k: eda.candle_agreement(k).round(2).to_dict("records") for k in MARKETS},
        "funnel": {k: eda.cleaning_funnel(k) for k in MARKETS},
        "calib": {k: eda.market_calibration(k).round(4).to_dict("records") for k in MARKETS},
    }
    lab = t["labels_daily"].groupby("series").agg(m=("markets", "sum"), y=("yes", "sum"))
    out["yes_rate"] = {s: r(row.y / row.m, 3) for s, row in lab.iterrows()}
    return out


def feature_info() -> list[dict]:
    from sklearn.metrics import roc_auc_score

    from .data import load_market

    df = load_market("15m")
    tr = df[df.decision_ts < "2026-08-03"]
    resid = tr.label - tr.p_mid
    rows = []
    for g, fs in FEATURE_GROUPS.items():
        for f in fs:
            x = tr[f].fillna(tr[f].median())
            rows.append({"f": f, "g": g, "auc": r(roc_auc_score(tr.label, x), 3),
                         "rc": r(np.corrcoef(x, resid)[0, 1], 3)})
    return rows


def strategy_data() -> dict:
    from sklearn.metrics import roc_auc_score

    d = pd.read_parquet(DATA / "strategy" / "oos_predictions.parquet")
    names = {"tabpfn_plus_momentum": "TabPFN-3.5 + momentum summary", "tabpfn": "TabPFN-3.5",
             "logistic": "Logistic regression", "ensemble_lr_lgbm_xgb": "LR + LightGBM + XGBoost", "mlp": "MLP",
             "price": "Market price", "stack": "Stack of models", "lstm_bars_plus_state": "LSTM + contract state",
             "live_gate": "Live gate (current)", "lstm_btc_bars": "LSTM on 1-min BTC bars"}
    rng = np.random.default_rng(0)
    days = d.day.unique()
    idx = {k: np.where(d.day.values == k)[0] for k in days}
    boots = [np.concatenate([idx[k] for k in rng.choice(days, len(days))]) for _ in range(1000)]
    y, pp = d.won.values, d.p__price.values
    aucs = []
    for k, lab in names.items():
        p = d[f"p__{k}"].values
        diffs = np.array([roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], pp[b]) for b in boots])
        aucs.append({"label": lab, "auc": r(roc_auc_score(y, p), 4), "d": r(diffs.mean(), 4),
                     "lo": r(np.percentile(diffs, 2.5), 4), "hi": r(np.percentile(diffs, 97.5), 4),
                     "tab": k.startswith("tabpfn"), "base": k in ("price", "live_gate")})
    ex, live = d.net_c_exits.values, d.take__live.values
    lb = np.array([ex[b][live[b]].mean() for b in boots])

    def stats(take, lab, key):
        diffs = np.array([ex[b][take[b]].mean() for b in boots]) - lb
        daily = pd.Series(ex * take, index=d.day).groupby(level=0).sum()
        cum = daily.cumsum()
        return {"label": lab, "key": key, "trades": int(take.sum()), "c": r(ex[take].mean(), 2),
                "d": r(diffs.mean(), 2), "lo": r(np.percentile(diffs, 2.5), 2), "hi": r(np.percentile(diffs, 97.5), 2),
                "total": r(ex[take].sum(), 0), "dd": r((cum - cum.cummax()).min(), 0),
                "cum": [[k, r(v, 0)] for k, v in cum.items()]}

    rules = [stats(live, "Live gate (current)", "live"), stats(np.ones(len(d), bool), "Take every signal", "all")]
    for k, lab in names.items():
        if k in ("live_gate",):
            continue
        col = f"take__{k}__edge"
        if col in d:
            rules.append(stats(d[col].values, f"{lab} · calibrated edge", k))
    rules.sort(key=lambda x: -x["c"])
    return {"n": int(len(d)), "days": int(d.day.nunique()), "first": d.day.min(), "last": d.day.max(),
            "auc": sorted(aucs, key=lambda x: -x["auc"]), "rules": rules}


def bench_data() -> dict:
    S = json.loads((ROOT / "results" / "summary.json").read_text())
    ctrl_p = ROOT / "results" / "controls.json"
    S["_controls"] = json.loads(ctrl_p.read_text()) if ctrl_p.exists() else None
    return S


# ------------------------------------------------------------------- narrative
def fmt_ci(v, k=4, unit=""):
    return f"{v['mean']:+.{k}f}{unit} [{v['lo']:+.{k}f}, {v['hi']:+.{k}f}]"


def headline(S: dict) -> dict:
    """Plain-language answers, computed from the results (never typed by hand)."""
    out = {}
    for mk in MARKETS:
        L = S[mk]["policy_A"]["models"]
        learners = {m: v for m, v in L.items() if m != "Market price" and m != "TabPFN-3.5-Thinking"}
        ranked = sorted(learners, key=lambda m: learners[m]["logloss"])
        tab = L.get("TabPFN-3.5")
        classic = [m for m in ranked if not m.startswith("TabPFN")]
        best_c = classic[0] if classic else None
        beats = [m for m in classic if L[m]["vs_ref"]["mean"] > 0]
        sig = [m for m in classic if L[m]["vs_ref"].get("bh_pass") and L[m]["vs_ref"]["mean"] > 0]
        out[mk] = {
            "rank": ranked.index("TabPFN-3.5") + 1 if tab else None, "n_learners": len(ranked),
            "best_classic": best_c, "tab_ll": tab["logloss"] if tab else None,
            "best_classic_ll": L[best_c]["logloss"] if best_c else None,
            "vs_best_classic": L[best_c]["vs_ref"] if best_c else None,
            "beats": len(beats), "sig": len(sig), "n_classic": len(classic),
            "vs_market": tab["vs_market"] if tab else None, "market_ll": L["Market price"]["logloss"],
            "rows": S[mk]["policy_A"]["n_rows"], "days": S[mk]["policy_A"]["n_days"],
        }
    return out


# ------------------------------------------------------------------------ html
CSS = r"""
:root{/* layout: one reading column, figures span the column, tables scroll inside their own box */
--bg:#f7f8f6;--surface:#ffffff;--fg:#14161a;--muted:#5a5f66;--faint:#e4e6e2;--rule:#cfd3cc;
--c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--classic:#9aa0a6;--neg:#e34948;--band:rgba(42,120,214,.12);
--display:"Fraunces",Georgia,serif;--body:"Source Sans 3","Segoe UI",system-ui,sans-serif;--mono:"JetBrains Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141513;--surface:#1a1a19;--fg:#eceae4;--muted:#a9a8a0;--faint:#2a2b28;--rule:#3a3b37;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--classic:#7d8288;--neg:#e66767;--band:rgba(57,135,229,.18);color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141513;--surface:#1a1a19;--fg:#eceae4;--muted:#a9a8a0;--faint:#2a2b28;--rule:#3a3b37;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--classic:#7d8288;--neg:#e66767;--band:rgba(57,135,229,.18);color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--body);font-size:16.5px;line-height:1.58}
.wrap{max-width:900px;margin:0 auto;padding-inline:20px;padding-block:32px 80px}
h1{font-family:var(--display);font-weight:600;font-size:clamp(2rem,5vw,2.9rem);line-height:1.08;margin:.15em 0 .35em;text-wrap:balance;letter-spacing:-.01em}
h2{font-family:var(--display);font-weight:600;font-size:1.55rem;margin:2.6em 0 .45em;text-wrap:balance;border-top:1px solid var(--rule);padding-top:1.1em}
h2 .n{font-family:var(--mono);font-size:.8rem;color:var(--muted);font-weight:400;display:block;letter-spacing:.06em;margin-bottom:.3em}
h3{font-size:1.04rem;margin:1.7em 0 .35em}
p,li{max-width:68ch}
a{color:var(--c1)}
code{font-family:var(--mono);font-size:.84em;background:var(--faint);padding:1px 5px;border-radius:3px}
.eyebrow{font-family:var(--mono);font-size:.74rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.lede{font-size:1.12rem;color:var(--fg);max-width:64ch}
.meta{color:var(--muted);font-size:.92rem}
.answers{list-style:none;padding:0;margin:1.6em 0;display:grid;gap:10px}
.answers li{max-width:none;background:var(--surface);border:1px solid var(--faint);border-radius:6px;padding:12px 16px}
.answers b{display:block;margin-bottom:2px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:1.2em 0}
.tile{background:var(--surface);border:1px solid var(--faint);border-radius:6px;padding:12px 14px}
.tile .v{font-family:var(--mono);font-size:1.35rem;font-variant-numeric:tabular-nums}
.tile .k{color:var(--muted);font-size:.84rem;line-height:1.3}
figure{margin:1.3em 0 1.8em;background:var(--surface);border:1px solid var(--faint);border-radius:6px;padding:14px 14px 10px;position:relative}
figcaption{font-size:.89rem;color:var(--muted);margin-top:6px;max-width:72ch}
figcaption b{color:var(--fg)}
.controls{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin:0 0 10px;font-size:.88rem}
.controls label{color:var(--muted)}
.seg{display:inline-flex;border:1px solid var(--rule);border-radius:5px;overflow:hidden}
.seg button{font:inherit;font-size:.85rem;background:transparent;color:var(--fg);border:0;padding:4px 11px;cursor:pointer}
.seg button+button{border-left:1px solid var(--rule)}
.seg button[aria-pressed="true"]{background:var(--fg);color:var(--bg)}
.seg button:focus-visible,select:focus-visible,.chip:focus-visible{outline:2px solid var(--c1);outline-offset:2px}
select{font:inherit;font-size:.86rem;background:var(--surface);color:var(--fg);border:1px solid var(--rule);border-radius:5px;padding:3px 6px;max-width:100%}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font:inherit;font-size:.8rem;border:1px solid var(--rule);border-radius:999px;padding:2px 10px;background:transparent;color:var(--muted);cursor:pointer;display:inline-flex;align-items:center;gap:6px}
.chip i{width:10px;height:3px;border-radius:2px;display:inline-block}
.chip[aria-pressed="true"]{color:var(--fg);border-color:var(--fg)}
svg.chart{width:100%;height:auto;display:block;font-family:var(--mono);overflow:visible}
.chart .grid{stroke:var(--faint);stroke-width:1}
.chart .axis{stroke:var(--muted);stroke-width:1}
.chart .tick{fill:var(--muted);font-size:10.5px}
.chart .lab{fill:var(--fg);font-size:11.5px;font-family:var(--body)}
.chart .val{fill:var(--muted);font-size:10.5px}
.chart .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart .hit{fill:transparent;cursor:crosshair}
.chart .xhair{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}
.tip{position:absolute;pointer-events:none;background:var(--surface);color:var(--fg);border:1px solid var(--rule);border-radius:5px;padding:6px 9px;font-size:.8rem;line-height:1.35;box-shadow:0 4px 14px rgba(0,0,0,.12);white-space:nowrap;z-index:5;font-family:var(--body)}
.tip b{font-weight:600}
.tw{overflow-x:auto;margin:.8em 0 1.4em}
table{border-collapse:collapse;font-size:.88rem;font-variant-numeric:tabular-nums;min-width:100%}
th{text-align:left;font-weight:600;color:var(--muted);font-size:.77rem;letter-spacing:.03em;border-bottom:1px solid var(--rule);padding:6px 12px 6px 0;white-space:nowrap}
td{border-bottom:1px solid var(--faint);padding:6px 12px 6px 0;vertical-align:top}
td.num{font-family:var(--mono);font-size:.82rem;white-space:nowrap}
tr.tab td:first-child{font-weight:600;color:var(--c1)}
tr.mkt td:first-child{font-style:italic}
.call{border-left:3px solid var(--c1);padding:2px 0 2px 14px;margin:1.3em 0}
.small{font-size:.86rem;color:var(--muted)}
pre{background:var(--surface);border:1px solid var(--faint);border-radius:6px;padding:12px 14px;overflow-x:auto;font-family:var(--mono);font-size:.8rem;line-height:1.5}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:.82rem;color:var(--muted);margin:2px 0 8px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:3px}
.legend i.dash{background:repeating-linear-gradient(90deg,var(--fg) 0 4px,transparent 4px 7px)!important}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr));gap:14px}
.two>*{min-width:0}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
@media (max-width:520px){body{font-size:15.5px}.tile .v{font-size:1.15rem}}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const NS = 'http://www.w3.org/2000/svg';
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const COLOR = m => m==='TabPFN-3.5'?'var(--c1)':m==='TabPFN-3.5-Fast'?'var(--c2)':m==='TabPFN-3.5-Thinking'?'var(--c3)':m==='Market price'?'var(--fg)':'var(--classic)';
function el(tag, attrs={}, parent){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e;}
function txt(parent,x,y,s,cls,anchor='start'){const t=el('text',{x,y,class:cls,'text-anchor':anchor},parent);t.textContent=s;return t;}
function niceTicks(lo,hi,n=5){const span=hi-lo||1;const step0=span/n;const mag=Math.pow(10,Math.floor(Math.log10(step0)));const err=step0/mag;const step=(err>=7.5?10:err>=3.5?5:err>=1.5?2:1)*mag;const out=[];for(let v=Math.ceil(lo/step)*step;v<=hi+1e-9;v+=step)out.push(+v.toFixed(10));return out;}
function fmt(v,d=2){return v==null?'–':(+v).toFixed(d);}
function sgn(v,d=3){return v==null?'–':(v>=0?'+':'')+(+v).toFixed(d);}
function tipFor(fig){let t=fig.querySelector('.tip');if(!t){t=document.createElement('div');t.className='tip';t.hidden=true;fig.appendChild(t);}return t;}
function showTip(fig,html,evt){const t=tipFor(fig);t.innerHTML=html;t.hidden=false;const r=fig.getBoundingClientRect();let x=evt.clientX-r.left+12,y=evt.clientY-r.top+12;const w=t.offsetWidth;if(x+w>r.width-4)x=evt.clientX-r.left-w-12;if(x<4)x=4;t.style.left=x+'px';t.style.top=y+'px';}
function hideTip(fig){const t=fig.querySelector('.tip');if(t)t.hidden=true;}

/* line chart: series [{name,color,dash,width,pts:[[x,y]]}], opts {x:'time'|'lin'|'log', yfmt, xfmt, h, y0, y1, ref:[{y,label}]} */
function lineChart(host, series, o={}){
  host.innerHTML=''; const W=720,H=o.h||300,L=o.left||52,R=o.right||16,T=12,B=34;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,class:'chart',role:'img','aria-label':o.label||'chart'},host);
  const xs=series.flatMap(s=>s.pts.map(p=>p[0])), ys=series.flatMap(s=>s.pts.map(p=>p[1])).filter(v=>v!=null);
  const tx=o.x==='time'?(v=>new Date(v).getTime()):(o.x==='log'?(v=>Math.log10(v)):(v=>v));
  let x0=Math.min(...xs.map(tx)),x1=Math.max(...xs.map(tx)); if(x0===x1){x0-=1;x1+=1;}
  let y0=o.y0!=null?o.y0:Math.min(...ys),y1=o.y1!=null?o.y1:Math.max(...ys);
  if(o.ref)o.ref.forEach(r=>{y0=Math.min(y0,r.y);y1=Math.max(y1,r.y);});
  const pad=(y1-y0)*0.06||1; if(o.y0==null)y0-=pad; if(o.y1==null)y1+=pad;
  const X=v=>L+(tx(v)-x0)/(x1-x0)*(W-L-R), Y=v=>T+(1-(v-y0)/(y1-y0))*(H-T-B);
  niceTicks(y0,y1,5).forEach(v=>{el('line',{x1:L,x2:W-R,y1:Y(v),y2:Y(v),class:'grid'},svg);txt(svg,L-6,Y(v)+3.5,(o.yfmt||(a=>a))(v),'tick','end');});
  let xt; if(o.x==='time'){const d0=new Date(x0),d1=new Date(x1);const days=(x1-x0)/864e5;const step=days>120?30:days>40?14:days>10?7:1;xt=[];for(let d=new Date(Date.UTC(d0.getUTCFullYear(),d0.getUTCMonth(),d0.getUTCDate()));d<=d1;d=new Date(d.getTime()+step*864e5))xt.push(d.getTime());}
  else if(o.x==='log'){xt=(o.xticks||[]).map(v=>Math.log10(v));} else xt=o.xticks||niceTicks(x0,x1,6);
  xt.forEach(v=>{const xx=L+(v-x0)/(x1-x0)*(W-L-R);if(xx<L-1||xx>W-R+1)return;el('line',{x1:xx,x2:xx,y1:H-B,y2:H-B+4,class:'axis'},svg);txt(svg,xx,H-B+16,o.x==='time'?new Date(v).toISOString().slice(5,10):o.x==='log'?String(Math.round(Math.pow(10,v))):(o.xfmt||(a=>a))(v),'tick','middle');});
  el('line',{x1:L,x2:W-R,y1:H-B,y2:H-B,class:'axis'},svg);
  if(o.xlabel)txt(svg,(L+W-R)/2,H-2,o.xlabel,'tick','middle');
  if(o.ylabel){const t=txt(svg,12,T+(H-T-B)/2,o.ylabel,'tick','middle');t.setAttribute('transform',`rotate(-90 12 ${T+(H-T-B)/2})`);}
  (o.ref||[]).forEach(r=>{el('line',{x1:L,x2:W-R,y1:Y(r.y),y2:Y(r.y),stroke:r.color||'var(--fg)','stroke-dasharray':'5 4','stroke-width':1.3},svg);if(r.label)txt(svg,W-R-4,Y(r.y)-5,r.label,'val','end');});
  if(o.band)el('rect',{x:L,width:W-L-R,y:Y(o.band[1]),height:Y(o.band[0])-Y(o.band[1]),fill:'var(--band)'},svg);
  series.forEach(s=>{if(s.area){const top=s.pts.map(p=>`${X(p[0])},${Y(p[2])}`).join(' ');const bot=s.pts.slice().reverse().map(p=>`${X(p[0])},${Y(p[1])}`).join(' ');el('polygon',{points:top+' '+bot,fill:s.color,opacity:.18},svg);return;}
    const segs=[];let cur=[];s.pts.forEach(p=>{if(p[1]==null){if(cur.length)segs.push(cur);cur=[];}else cur.push(`${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`);});if(cur.length)segs.push(cur);
    segs.forEach(sg=>el('polyline',{points:sg.join(' '),class:'ln',stroke:s.color,'stroke-width':s.width||2,'stroke-dasharray':s.dash||'',opacity:s.opacity||1},svg));
    if(s.marker)s.pts.forEach(p=>{if(p[1]!=null)el('circle',{cx:X(p[0]),cy:Y(p[1]),r:3.5,fill:s.color,stroke:'var(--surface)','stroke-width':1.5},svg);});
    if(s.endLabel){const p=s.pts[s.pts.length-1];txt(svg,X(p[0])+6,Y(p[1])+4,s.endLabel,'lab');}});
  const fig=host.closest('figure'); const xh=el('line',{y1:T,y2:H-B,class:'xhair',visibility:'hidden'},svg);
  const hit=el('rect',{x:L,y:T,width:W-L-R,height:H-T-B,class:'hit'},svg);
  const allx=[...new Set(series.filter(s=>!s.area&&!s.nohover).flatMap(s=>s.pts.map(p=>p[0])))].sort((a,b)=>tx(a)-tx(b));
  function move(evt){const pt=svg.createSVGPoint();pt.x=evt.clientX;pt.y=evt.clientY;const sp=pt.matrixTransform(svg.getScreenCTM().inverse());let best=allx[0],bd=1e18;allx.forEach(v=>{const d=Math.abs(X(v)-sp.x);if(d<bd){bd=d;best=v;}});
    xh.setAttribute('x1',X(best));xh.setAttribute('x2',X(best));xh.setAttribute('visibility','visible');
    const rows=series.filter(s=>!s.area&&!s.nohover).map(s=>{const p=s.pts.find(q=>q[0]===best);return p&&p[1]!=null?`<div><span style="color:${s.color}">●</span> ${s.name}: <b>${(o.tipfmt||(a=>fmt(a,3)))(p[1])}</b></div>`:'';}).join('');
    showTip(fig,`<div class="meta">${o.x==='time'?new Date(best).toISOString().slice(0,10):(o.tipx||(a=>a))(best)}</div>${rows}`,evt);}
  hit.addEventListener('pointermove',move);hit.addEventListener('pointerleave',()=>{xh.setAttribute('visibility','hidden');hideTip(fig);});
  return svg;
}

/* dot + CI rows: [{label,mean,lo,hi,color,tip}] */
function dotCI(host, rows, o={}){
  host.innerHTML=''; const W=720,rowH=o.rowH||28,L=o.left||210,R=70,T=8,B=30,H=T+B+rows.length*rowH;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,class:'chart',role:'img','aria-label':o.label||'chart'},host);
  let lo=Math.min(0,...rows.map(r=>r.lo??r.mean)),hi=Math.max(0,...rows.map(r=>r.hi??r.mean));const pad=(hi-lo)*.08||.01;lo-=pad;hi+=pad;
  const X=v=>L+(v-lo)/(hi-lo)*(W-L-R);
  niceTicks(lo,hi,5).forEach(v=>{el('line',{x1:X(v),x2:X(v),y1:T,y2:H-B,class:'grid'},svg);txt(svg,X(v),H-B+15,(o.xfmt||(a=>a))(v),'tick','middle');});
  el('line',{x1:X(0),x2:X(0),y1:T,y2:H-B,stroke:'var(--fg)','stroke-width':1.2},svg);
  if(o.xlabel)txt(svg,(L+W-R)/2,H-3,o.xlabel,'tick','middle');
  const fig=host.closest('figure');
  rows.forEach((r,i)=>{const y=T+i*rowH+rowH/2;const g=el('g',{},svg);
    if(r.lo!=null)el('line',{x1:X(r.lo),x2:X(r.hi),y1:y,y2:y,stroke:r.color,'stroke-width':2.2,'stroke-linecap':'round'},g);
    el('circle',{cx:X(r.mean),cy:y,r:5.5,fill:r.color,stroke:'var(--surface)','stroke-width':2},g);
    const t=txt(svg,L-10,y+4,r.label,'lab','end');if(r.bold)t.setAttribute('font-weight','600');
    txt(svg,W-R+8,y+4,(o.vfmt||(a=>sgn(a,3)))(r.mean),'val');
    const h=el('rect',{x:0,y:y-rowH/2,width:W,height:rowH,class:'hit'},g);
    h.addEventListener('pointermove',e=>showTip(fig,r.tip||`<b>${r.label}</b><br>${sgn(r.mean,4)} [${sgn(r.lo,4)}, ${sgn(r.hi,4)}]`,e));h.addEventListener('pointerleave',()=>hideTip(fig));});
}

/* scatter: [{label,x,y,color}] */
function scatter(host, pts, o={}){
  host.innerHTML='';const W=720,H=o.h||300,L=60,R=20,T=14,B=38;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,class:'chart',role:'img','aria-label':o.label||'chart'},host);
  const lx=v=>Math.log10(Math.max(v,1e-3));const xs=pts.map(p=>lx(p.x)),ys=pts.map(p=>p.y);
  let x0=Math.floor(Math.min(...xs)),x1=Math.ceil(Math.max(...xs));let y0=Math.min(...ys),y1=Math.max(...ys);const pad=(y1-y0)*.12||.001;y0-=pad;y1+=pad;
  const X=v=>L+(lx(v)-x0)/(x1-x0)*(W-L-R),Y=v=>T+(1-(v-y0)/(y1-y0))*(H-T-B);
  for(let e=x0;e<=x1;e++){const xx=L+(e-x0)/(x1-x0)*(W-L-R);el('line',{x1:xx,x2:xx,y1:T,y2:H-B,class:'grid'},svg);txt(svg,xx,H-B+15,e<0?(Math.pow(10,e)).toFixed(-e):String(Math.pow(10,e)),'tick','middle');}
  niceTicks(y0,y1,5).forEach(v=>{el('line',{x1:L,x2:W-R,y1:Y(v),y2:Y(v),class:'grid'},svg);txt(svg,L-6,Y(v)+3.5,v.toFixed(3),'tick','end');});
  if(o.ref!=null){el('line',{x1:L,x2:W-R,y1:Y(o.ref),y2:Y(o.ref),stroke:'var(--fg)','stroke-dasharray':'5 4','stroke-width':1.2},svg);txt(svg,W-R-4,Y(o.ref)-5,'market price','val','end');}
  txt(svg,(L+W-R)/2,H-3,o.xlabel||'','tick','middle');const yl=txt(svg,14,(T+H-B)/2,o.ylabel||'','tick','middle');yl.setAttribute('transform',`rotate(-90 14 ${(T+H-B)/2})`);
  const fig=host.closest('figure');
  pts.forEach(p=>{const g=el('g',{},svg);el('circle',{cx:X(p.x),cy:Y(p.y),r:p.big?7:5.5,fill:p.color,stroke:'var(--surface)','stroke-width':2},g);
    if(p.big||o.labelAll)txt(svg,X(p.x)+9,Y(p.y)+4,p.label,'lab');
    const h=el('circle',{cx:X(p.x),cy:Y(p.y),r:14,class:'hit'},g);h.addEventListener('pointermove',e=>showTip(fig,`<b>${p.label}</b><br>${o.xname}: ${fmt(p.x,2)} s<br>${o.yname}: ${fmt(p.y,4)}`,e));h.addEventListener('pointerleave',()=>hideTip(fig));});
}

function seg(host, options, value, onChange){host.innerHTML='';host.className='seg';options.forEach(([v,l])=>{const b=document.createElement('button');b.type='button';b.textContent=l;b.setAttribute('aria-pressed',String(v===value));b.onclick=()=>{[...host.children].forEach(c=>c.setAttribute('aria-pressed','false'));b.setAttribute('aria-pressed','true');onChange(v);};host.appendChild(b);});}

/* ---------- explorer ---------- */
(function(){
  const sel=document.getElementById('ex-market'),dsel=document.getElementById('ex-day'),mode=document.getElementById('ex-mode');
  const host=document.getElementById('ex-chart'),host2=document.getElementById('ex-chart2'),cap=document.getElementById('ex-cap');
  let M='15';
  const days15=[...new Set(D.ex.m15.map(m=>m.open.slice(0,10)))];
  const days1h=[...new Set(D.ex.m1h.map(m=>m.close.slice(0,10)))];
  function fillDays(){const ds=M==='15'?days15:days1h;dsel.innerHTML=ds.map(d=>`<option value="${d}">${d}</option>`).join('');dsel.value=ds[Math.min(3,ds.length-1)];fillMarkets();}
  function fillMarkets(){const d=dsel.value;if(M==='15'){const ms=D.ex.m15.filter(m=>m.open.startsWith(d));sel.innerHTML=ms.map((m,i)=>`<option value="${D.ex.m15.indexOf(m)}">${m.open.slice(11)} UTC · ${m.y?'settled YES':'settled NO'}</option>`).join('');sel.value=sel.options[Math.floor(sel.options.length*0.55)]?.value;}
    else{const ms=D.ex.m1h.filter(m=>m.close.startsWith(d));sel.innerHTML=ms.map(m=>`<option value="${D.ex.m1h.indexOf(m)}">closes ${m.close.slice(11)} UTC · ${m.legs.length} strikes</option>`).join('');sel.value=sel.options[Math.floor(sel.options.length*0.55)]?.value;}draw();}
  function draw(){const i=+sel.value;if(isNaN(i))return;
    if(M==='15'){const m=D.ex.m15[i];host2.hidden=false;
      lineChart(host,[{name:'ask',color:'var(--c1)',pts:m.s.map((s,j)=>[s/60,m.a[j]]),width:1.2,opacity:.55},{name:'mid',color:'var(--c1)',pts:m.s.map((s,j)=>[s/60,(m.a[j]+m.b[j])/2]),width:2.2},{name:'bid',color:'var(--c1)',pts:m.s.map((s,j)=>[s/60,m.b[j]]),width:1.2,opacity:.55}],
        {x:'lin',xticks:[0,3,5,10,12,15],y0:0,y1:100,ylabel:'YES price (¢)',xlabel:'minutes since open',tipx:a=>`minute ${fmt(a,2)}`,tipfmt:a=>fmt(a,1)+'¢',h:250,label:'contract price'});
      lineChart(host2,[{name:'BTC',color:'var(--fg)',pts:m.s.map((s,j)=>[s/60,m.x[j]]),width:1.6}],{x:'lin',xticks:[0,3,5,10,12,15],ref:[{y:m.k,label:'strike '+m.k.toLocaleString()}],ylabel:'BTC (USD)',xlabel:'minutes since open',tipx:a=>`minute ${fmt(a,2)}`,tipfmt:a=>'$'+Math.round(a).toLocaleString(),h:200,left:64,label:'BTC'});
      cap.innerHTML=`<b>${m.m}</b>: opened ${m.open} UTC, reference price $${m.k.toLocaleString()}, settled <b>${m.y?'YES (BTC finished higher)':'NO (BTC finished lower)'}</b>. Shaded markers at minutes 5 and 10 are the two decision times used in the benchmark.`;
      [5,10].forEach(mn=>{const svg=host.querySelector('svg');const L=52,W=720,R=16;const xs=m.s.map(s=>s/60);const x0=Math.min(...xs),x1=Math.max(...xs);const xx=L+(mn-x0)/(x1-x0)*(W-L-R);el('line',{x1:xx,x2:xx,y1:12,y2:216,stroke:'var(--c2)','stroke-width':1.5,'stroke-dasharray':'2 3'},svg);txt(svg,xx+4,24,`decide (τ=${15-mn} min)`,'val');});
    } else {const ev=D.ex.m1h[i];host2.hidden=true;
      const ser=ev.legs.map((l,j)=>({name:`$${Math.round(l.k).toLocaleString()} ${l.y?'✓':'✗'}`,color:l.y?'var(--c1)':'var(--c2)',pts:l.s.map((s,q)=>[s,l.p[q]]),width:1.8,endLabel:`$${Math.round(l.k).toLocaleString()}`}));
      lineChart(host,ser,{x:'lin',xticks:[0,15,30,45,60],y0:0,y1:100,ylabel:'P(BTC above strike), ¢',xlabel:'minutes since open',tipx:a=>`minute ${a}`,tipfmt:a=>fmt(a,1)+'¢',h:300,right:74,label:'hourly ladder'});
      cap.innerHTML=`Hourly ladder closing <b>${ev.close} UTC</b>: one line per strike. Blue strikes settled YES (BTC finished above), orange settled NO. Decisions are made at minute 30 and minute 45.`;}
  }
  seg(mode,[['15','15-minute'],['1h','hourly ladder']],'15',v=>{M=v;fillDays();});
  dsel.onchange=fillMarkets; sel.onchange=draw; fillDays();
})();

/* ---------- EDA charts ---------- */
lineChart(document.getElementById('eda-daily'),[{name:'15-minute tape',color:'var(--c1)',pts:D.eda.daily['15m'].map(r=>[r[0],r[1]/1e3])},{name:'hourly tape',color:'var(--c2)',pts:D.eda.daily['1h'].map(r=>[r[0],r[1]/1e3])}],{x:'time',ylabel:'thousand rows / day',tipfmt:a=>fmt(a,0)+'k',label:'rows per day'});
(function(){const host=document.getElementById('eda-q');const ctl=document.getElementById('eda-q-ctl');
  function draw(k){const W=D.eda.weekly;lineChart(host,[{name:'15-minute',color:'var(--c1)',pts:W['15-minute'].map(r=>[r[0],r[k]]),marker:true},{name:'hourly',color:'var(--c2)',pts:W['hourly'].map(r=>[r[0],r[k]]),marker:true}],{x:'time',y0:0,y1:k===1?100:null,ylabel:k===1?'valid quotes, % of rows':'crossed books, % of rows',tipfmt:a=>fmt(a,2)+'%',label:'weekly quality'});
    const svg=host.querySelector('svg');}
  seg(ctl,[[1,'valid quotes'],[2,'crossed books']],1,draw);draw(1);})();
(function(){const host=document.getElementById('eda-cal');const ctl=document.getElementById('eda-cal-ctl');
  function draw(k){const c=D.eda.calib[k].filter(r=>r.n>=50);lineChart(host,[{name:'perfect calibration',color:'var(--muted)',pts:[[0,0],[1,1]],dash:'4 4',width:1,nohover:true},{name:'settled YES rate',color:'var(--c1)',pts:c.map(r=>[r.p,r.y]),marker:true},{name:'95% band',color:'var(--c1)',area:true,pts:c.map(r=>[r.p,r.lo,r.hi])}],{x:'lin',xticks:[0,.2,.4,.6,.8,1],xfmt:a=>a.toFixed(1),y0:0,y1:1,yfmt:a=>a.toFixed(1),xlabel:'market mid price (as probability)',ylabel:'settled YES rate',tipx:a=>'price '+fmt(a,3),tipfmt:a=>fmt(a,3),h:320,label:'market calibration'});}
  seg(ctl,[['15m','15-minute'],['1h','hourly']],'15m',draw);draw('15m');})();

/* ---------- features ---------- */
(function(){const host=document.getElementById('feat-rc');const rows=D.feat.slice().sort((a,b)=>Math.abs(b.rc)-Math.abs(a.rc)).map(f=>({label:f.f,mean:f.rc,color:f.g==='contract'||f.g==='moneyness'?'var(--c1)':'var(--classic)',tip:`<b>${f.f}</b> (${f.g})<br>AUC alone: ${fmt(f.auc,3)}<br>corr with label − price: ${sgn(f.rc,3)}`}));
  dotCI(host,rows,{rowH:20,left:170,xlabel:'correlation with what the price misses (label − price)',vfmt:a=>sgn(a,3),label:'feature information beyond price'});})();

/* ---------- benchmark ---------- */
const B=D.bench;
function boardRows(mk,pol){const L=B[mk][pol];if(!L)return[];return Object.entries(L.models).map(([m,r])=>({m,...r})).sort((a,b)=>a.logloss-b.logloss);}
(function(){const tbl=document.getElementById('lb-table'),dots=document.getElementById('lb-dots'),cm=document.getElementById('lb-mk'),cp=document.getElementById('lb-pol'),cap=document.getElementById('lb-cap');let mk='15m',pol='policy_A';
  function draw(){const rows=boardRows(mk,pol);if(!rows.length){tbl.innerHTML='<p class="small">Not run.</p>';return;}const L=B[mk][pol];
    tbl.innerHTML=`<table><thead><tr><th>Model</th><th>Log loss</th><th>Δ vs ${L.ref} [95% CI]</th><th>BH</th><th>Δ vs market</th><th>Brier</th><th>AUC</th><th>ECE</th><th>Fit s</th><th>Tuning s</th></tr></thead><tbody>${rows.map(r=>`<tr class="${r.m.startsWith('TabPFN')?'tab':r.m==='Market price'?'mkt':''}"><td>${r.m}</td><td class="num">${fmt(r.logloss,4)}</td><td class="num">${r.vs_ref?`${sgn(r.vs_ref.mean,4)} [${sgn(r.vs_ref.lo,4)}, ${sgn(r.vs_ref.hi,4)}]`:'reference'}</td><td class="num">${r.vs_ref?(r.vs_ref.bh_pass?'yes':'no'):''}</td><td class="num">${r.vs_market?sgn(r.vs_market.mean,4):'–'}</td><td class="num">${fmt(r.brier,4)}</td><td class="num">${fmt(r.auc,4)}</td><td class="num">${fmt(r.ece,4)}</td><td class="num">${fmt(r.cpu.fit_s_mean,1)}</td><td class="num">${r.cpu.search_s_mean?fmt(r.cpu.search_s_mean,1):'–'}</td></tr>`).join('')}</tbody></table>`;
    const dr=rows.filter(r=>r.vs_ref).sort((a,b)=>a.vs_ref.mean-b.vs_ref.mean).map(r=>({label:r.m,mean:r.vs_ref.mean,lo:r.vs_ref.lo,hi:r.vs_ref.hi,color:COLOR(r.m),bold:r.m.startsWith('TabPFN')||r.m==='Market price'}));
    dotCI(dots,dr,{xlabel:`Δ log loss vs ${L.ref} (right = worse than ${L.ref})`,vfmt:a=>sgn(a,4),label:'paired differences'});
    cap.innerHTML=`<b>${L.n_rows.toLocaleString()} test rows over ${L.n_days} days.</b> Each line is a paired difference in log loss with a day-block bootstrap 95% CI. BH = significant after Benjamini–Hochberg (q = 0.10) across the comparisons.`;}
  seg(cm,[['15m','15-minute'],['1h','hourly']],mk,v=>{mk=v;draw();});
  seg(cp,[['policy_A','equal 5,000 rows'],['policy_B','classic models get all history']],pol,v=>{pol=v;draw();});draw();})();

(function(){const host=document.getElementById('lc-chart'),cm=document.getElementById('lc-mk'),chips=document.getElementById('lc-chips');let mk='15m';const on={};
  function draw(){const lc=B[mk].learning_curve;const ns=Object.keys(lc).map(Number);const models=[...new Set(Object.values(lc).flatMap(o=>Object.keys(o)))];
    models.forEach(m=>{if(on[m]===undefined)on[m]=true;});
    chips.innerHTML='';models.filter(m=>m!=='Market price').forEach(m=>{const b=document.createElement('button');b.type='button';b.className='chip';b.setAttribute('aria-pressed',String(on[m]));b.innerHTML=`<i style="background:${COLOR(m)}"></i>${m.replace(' · default','')}`;b.onclick=()=>{on[m]=!on[m];draw();};chips.appendChild(b);});
    const mkt=lc[String(ns[ns.length-1])]['Market price'];
    const ser=models.filter(m=>m!=='Market price'&&on[m]).map(m=>({name:m.replace(' · default',''),color:COLOR(m),width:m.startsWith('TabPFN')?2.6:1.3,marker:true,pts:ns.map(n=>[n,(lc[String(n)][m]||{}).logloss??null])}));
    lineChart(host,ser,{x:'log',xticks:ns,ref:mkt?[{y:mkt.logloss,label:'market price'}]:[],xlabel:'training rows (log scale)',ylabel:'test log loss',tipx:a=>`${a.toLocaleString()} training rows`,tipfmt:a=>fmt(a,4),h:330,yfmt:a=>a.toFixed(3),label:'learning curves'});}
  seg(cm,[['15m','15-minute'],['1h','hourly']],mk,v=>{mk=v;draw();});draw();})();

(function(){const host=document.getElementById('rel-chart'),cm=document.getElementById('rel-mk'),ms=document.getElementById('rel-model');let mk='15m';
  function opts(){const L=B[mk].policy_A.models;ms.innerHTML=Object.keys(L).filter(m=>m!=='Market price'&&m!=='TabPFN-3.5').map(m=>`<option>${m}</option>`).join('');ms.value=Object.keys(L).find(m=>m.startsWith('LightGBM'))||ms.options[0].value;}
  function draw(){const L=B[mk].policy_A.models;const pick=['Market price','TabPFN-3.5',ms.value];
    const ser=[{name:'perfect',color:'var(--muted)',pts:[[0,0],[1,1]],dash:'4 4',width:1,nohover:true}].concat(pick.filter(m=>L[m]).map(m=>({name:`${m} (ECE ${fmt(L[m].ece,3)})`,color:m===pick[2]?'var(--c2)':COLOR(m),width:m==='TabPFN-3.5'?2.6:1.6,marker:true,pts:L[m].reliability.map(b=>[+b.p_mean.toFixed(3),b.y_mean])})));
    lineChart(host,ser,{x:'lin',xticks:[0,.2,.4,.6,.8,1],xfmt:a=>a.toFixed(1),y0:0,y1:1,yfmt:a=>a.toFixed(1),xlabel:'predicted probability (bin mean)',ylabel:'observed YES rate',tipfmt:a=>fmt(a,3),tipx:a=>'p ≈ '+fmt(a,3),h:330,label:'reliability'});
    document.getElementById('rel-legend').innerHTML=pick.filter(m=>L[m]).map(m=>`<span><i style="background:${m===pick[2]?'var(--c2)':COLOR(m)}"></i>${m}: ECE ${fmt(L[m].ece,4)}</span>`).join('');}
  seg(cm,[['15m','15-minute'],['1h','hourly']],mk,v=>{mk=v;opts();draw();});ms.onchange=draw;opts();draw();})();

(function(){const host=document.getElementById('cpu-chart'),cm=document.getElementById('cpu-mk');let mk='15m';
  function draw(){const L=B[mk].policy_A.models;const pts=Object.entries(L).filter(([m,r])=>m!=='Market price'&&r.cpu&&r.cpu.fit_s_mean!=null).map(([m,r])=>({label:m.replace(' · tuned',''),x:(r.cpu.fit_s_mean||0)+(r.cpu.search_s_mean||0)+0.01,y:r.logloss,color:COLOR(m),big:m.startsWith('TabPFN')}));
    scatter(host,pts,{xlabel:'CPU seconds per weekly refit, including hyper-parameter search (log scale)',ylabel:'test log loss',xname:'seconds',yname:'log loss',ref:L['Market price'].logloss,labelAll:true,label:'cost vs accuracy'});}
  seg(cm,[['15m','15-minute'],['1h','hourly']],mk,v=>{mk=v;draw();});draw();})();

(function(){const host=document.getElementById('tr-chart'),tbl=document.getElementById('tr-table'),cm=document.getElementById('tr-mk');let mk='15m';
  function draw(){const T=B[mk].trading;const rows=Object.entries(T).filter(([m,r])=>r.trades>0).sort((a,b)=>b[1].c_per_ct-a[1].c_per_ct);
    tbl.innerHTML=`<table><thead><tr><th>Model</th><th>Trades</th><th>¢ / contract [95% CI]</th><th>Win rate</th><th>Avg price</th><th>Total ¢ (1 ct)</th><th>Max drawdown ¢</th></tr></thead><tbody>${rows.map(([m,r])=>`<tr class="${m.startsWith('TabPFN')?'tab':m==='Market price'?'mkt':''}"><td>${m}</td><td class="num">${r.trades}</td><td class="num">${sgn(r.c_per_ct,2)} [${sgn(r.ci95[0],2)}, ${sgn(r.ci95[1],2)}]</td><td class="num">${fmt(100*r.win_rate,1)}%</td><td class="num">${fmt(r.avg_price_c,1)}</td><td class="num">${fmt(r.total_c_1ct,0)}</td><td class="num">${fmt(r.max_dd_c_1ct,0)}</td></tr>`).join('')}</tbody></table>`;
    lineChart(host,rows.map(([m,r])=>({name:m.replace(' · tuned',''),color:COLOR(m),width:m.startsWith('TabPFN')?2.6:1.2,pts:r.cum.map(c=>[c[0],c[1]])})),{x:'time',ylabel:'cumulative ¢ at 1 contract',tipfmt:a=>fmt(a,0)+'¢',h:300,label:'cumulative trading result'});}
  seg(cm,[['15m','15-minute'],['1h','hourly']],mk,v=>{mk=v;draw();});draw();})();

/* ---------- strategy ---------- */
(function(){const S=D.strat;dotCI(document.getElementById('st-auc'),S.auc.filter(a=>a.label!=='Market price').map(a=>({label:a.label,mean:a.d,lo:a.lo,hi:a.hi,color:a.tab?'var(--c1)':a.label.startsWith('Live')?'var(--fg)':'var(--classic)',bold:a.tab,tip:`<b>${a.label}</b><br>AUC ${fmt(a.auc,4)}<br>vs price ${sgn(a.d,4)} [${sgn(a.lo,4)}, ${sgn(a.hi,4)}]`})),{xlabel:'AUC minus the market price\'s AUC (right = ranks winners better than the price)',vfmt:a=>sgn(a,3),label:'AUC vs price'});
  dotCI(document.getElementById('st-rules'),S.rules.filter(r=>r.key!=='live').map(r=>({label:r.label,mean:r.d,lo:r.lo,hi:r.hi,color:r.key.startsWith('tabpfn')?'var(--c1)':r.key==='all'||r.key==='price'?'var(--fg)':'var(--classic)',bold:r.key.startsWith('tabpfn'),tip:`<b>${r.label}</b><br>${r.trades} trades · ${sgn(r.c,2)} ¢/ct<br>vs live gate ${sgn(r.d,2)} [${sgn(r.lo,2)}, ${sgn(r.hi,2)}]<br>max drawdown ${fmt(r.dd,0)}¢`})),{left:280,xlabel:'¢ per contract minus the live gate\'s, paired by day (right = better)',vfmt:a=>sgn(a,2),label:'rules vs live'});
  const keep=['live','all','tabpfn_plus_momentum','tabpfn','logistic'];const col={live:'var(--fg)',all:'var(--classic)',tabpfn_plus_momentum:'var(--c2)',tabpfn:'var(--c1)',logistic:'var(--c3)'};
  lineChart(document.getElementById('st-cum'),S.rules.filter(r=>keep.includes(r.key)).map(r=>({name:r.label,color:col[r.key],width:r.key.startsWith('tabpfn')?2.6:1.5,dash:r.key==='all'?'4 3':'',pts:r.cum})),{x:'time',ylabel:'cumulative ¢ at 1 contract',tipfmt:a=>fmt(a,0)+'¢',h:300,label:'strategy cumulative'});})();
"""


def page(fragment: bool = False) -> str:
    S = bench_data()
    H = headline(S)
    ed = eda_data()
    fid = json.loads((DATA / "fidelity" / "fidelity.json").read_text())
    st = strategy_data()
    feat = feature_info()
    ex = explorer_data()
    tot = ed["totals"]
    total_rows = (tot["tape_15m_rows"] + tot["tape_1h_rows"] + tot["order_book_rows"] + tot["spot_venue_rows"]
                  + tot["indicator_rows"] + tot["brti_rows"])
    data = {"ex": ex, "eda": ed, "feat": feat, "bench": S, "strat": st}

    def mk_answer(mk: str, title: str) -> str:
        h = H[mk]
        if h["rank"] is None:
            return ""
        vb, vm = h["vs_best_classic"], h["vs_market"]
        if h["rank"] == 1:
            lead = (f"TabPFN-3.5 has the lowest log loss of all {h['n_learners']} learners "
                    f"({h['tab_ll']:.4f} vs {h['best_classic_ll']:.4f} for the best tuned classic model, "
                    f"{h['best_classic'].replace(' · tuned', '')}).")
        else:
            lead = (f"TabPFN-3.5 ranks {h['rank']} of {h['n_learners']} learners on log loss "
                    f"({h['tab_ll']:.4f}; best classic: {h['best_classic'].replace(' · tuned', '')} {h['best_classic_ll']:.4f}).")
        sig = (f" It beats {h['beats']} of {h['n_classic']} tuned classic models; {h['sig']} of those gaps survive "
               f"Benjamini–Hochberg.")
        mkt = ("It is <b>better than the market price</b>" if vm["hi"] < 0 else
               "Like every model, it is <b>not better than the market price</b>" if vm["mean"] >= 0 else
               "It edges the market price, but the interval includes zero")
        mkt += f" (Δ log loss {vm['mean']:+.4f}, 95% CI [{vm['lo']:+.4f}, {vm['hi']:+.4f}])."
        return f"<li><b>{title}</b>{lead}{sig} {mkt}</li>"

    def lc_answer() -> str:
        bits = []
        for mk, name in [("15m", "15-minute"), ("1h", "hourly")]:
            lc = S[mk]["learning_curve"]
            if "250" not in lc or "TabPFN-3.5" not in lc["250"]:
                continue
            small = lc["250"]
            classic = {m: v["logloss"] for m, v in small.items() if "· default" in m}
            if not classic:
                continue
            best = min(classic, key=classic.get)
            bits.append(f"{name}: {small['TabPFN-3.5']['logloss']:.4f} vs {classic[best]:.4f} for the best classic model "
                        f"({best.replace(' · default', '')})")
        return ("<li><b>Small data is where the gap is largest.</b>With only 250 training rows, TabPFN-3.5 "
                "scores " + "; ".join(bits) + ". Classic learners need thousands of rows to catch up.</li>") if bits else ""

    # ---- computed captions for section 7
    pos = [a for a in st["auc"] if a["label"] not in ("Market price",) and a["d"] > 0]
    top = max(st["auc"], key=lambda a: a["d"])
    sig_pos = [a for a in st["auc"] if a["lo"] > 0]
    live_auc = next(a for a in st["auc"] if a["label"].startswith("Live"))
    st_auc_cap = (f"The price alone is a strong ranker. {len(pos)} of {len(st['auc']) - 1} models rank winners better "
                  f"than it; {E(top['label'])} improves on it the most ({top['d']:+.4f}, CI [{top['lo']:+.4f}, {top['hi']:+.4f}])"
                  + (", and none of these gains is statistically significant." if not sig_pos else ".")
                  + (f" My current live gate is significantly <i>worse</i> than the price ({live_auc['d']:+.4f})."
                     if live_auc["hi"] < 0 else ""))
    tr_ = {r["key"]: r for r in st["rules"]}
    lv = tr_["live"]
    tb = [tr_[k] for k in ("tabpfn_plus_momentum", "tabpfn") if k in tr_]
    st_cum_cap = ("The TabPFN rules take " + " and ".join(f"{r['trades']}" for r in tb)
                  + f" trades against the live gate's {lv['trades']}, for " + " and ".join(f"{r['total']:.0f}¢" for r in tb)
                  + f" in total against {lv['total']:.0f}¢, with worst drawdowns of "
                  + " and ".join(f"{r['dd']:.0f}¢" for r in tb) + f" against {lv['dd']:.0f}¢.")

    tab_cpu = S["15m"]["policy_A"]["models"].get("TabPFN-3.5", {}).get("cpu", {})
    fast_cpu = S["15m"]["policy_A"]["models"].get("TabPFN-3.5-Fast", {}).get("cpu", {})
    best_rule = next(r for r in st["rules"] if r["key"].startswith("tabpfn"))
    live_rule = next(r for r in st["rules"] if r["key"] == "live")
    top_auc = st["auc"][0]
    cpc = fid["cents_per_contract_same_trades"]
    vb, vt, ex_ = fid["vs_backtest"], fid["vs_paper_twin"], fid["execution"]

    answers = "".join([
        mk_answer("15m", "15-minute up/down markets."),
        mk_answer("1h", "Hourly strike ladder."),
        lc_answer(),
        f"<li><b>All of it runs on an 8-core CPU.</b>A weekly TabPFN-3.5 refit on 5,000 rows takes "
        f"{tab_cpu.get('fit_s_mean', 0):.0f} s, and scoring one live market takes {tab_cpu.get('single_market_s', 0):.2f} s. "
        f"TabPFN-3.5-Fast does the same in {fast_cpu.get('fit_s_mean', 0):.0f} s and {fast_cpu.get('single_market_s', 0):.2f} s. "
        f"Nothing is tuned, so there is no search to rerun.</li>",
        f"<li><b>Inside my live strategy, TabPFN is the best filter I have tested.</b>It ranks winners best "
        f"(AUC {top_auc['auc']:.3f} vs {next(a for a in st['auc'] if a['label'] == 'Market price')['auc']:.3f} for the price "
        f"and {next(a for a in st['auc'] if a['label'].startswith('Live'))['auc']:.3f} for my current gate). "
        f"As a calibrated-edge filter it earns {best_rule['c']:+.2f} ¢/contract against the live gate's "
        f"{live_rule['c']:+.2f} on the same signals; the paired interval [{best_rule['lo']:+.2f}, {best_rule['hi']:+.2f}] "
        f"still includes zero.</li>",
    ])

    def fid_row(k, v):
        return (f"<tr><td>{E(k)}</td><td class='num'>{v['n']}</td><td class='num'>{v['c_per_ct']:+.2f}</td>"
                f"<td class='num'>[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]</td></tr>")

    feat_rows = "".join(
        f"<tr><td>{E(g)}</td><td>{', '.join(f'<code>{E(f)}</code>' for f in fs)}</td></tr>"
        for g, fs in FEATURE_GROUPS.items())
    milestones = "".join(f"<tr><td class='num'>{E(d)}</td><td>{E(t)}</td></tr>" for d, t in ed["milestones"])
    candle_rows = ""
    for k, name in [("15m", "15-minute"), ("1h", "hourly")]:
        for c in ed["candle"][k]:
            candle_rows += (f"<tr><td>{name}</td><td class='num'>{c['month']}</td><td class='num'>{c['checked']:,}</td>"
                            f"<td class='num'>{c['confirmed_pct']:.1f}%</td><td class='num'>{c['median_diff_c']:.2f}¢</td></tr>")
    funnel = ""
    for k, name in [("15m", "15-minute"), ("1h", "hourly")]:
        steps = ed["funnel"][k]
        funnel += f"<tr><th colspan='2'>{name}</th></tr>" + "".join(
            f"<tr><td>{E(s)}</td><td class='num'>{n:,}</td></tr>" for s, n in steps)
    th = S["15m"].get("thinking")
    thinking_html = ""
    if th:
        rows = []
        for mk, name in [("15m", "15-minute"), ("1h", "hourly")]:
            t = S[mk].get("thinking")
            if not t:
                continue
            for m in ["Market price", "TabPFN-3.5", "TabPFN-3.5-Fast", "TabPFN-3.5-Thinking", "LightGBM · tuned",
                      "CatBoost · tuned", "Logistic regression · tuned"]:
                if m in t and isinstance(t[m], dict) and "logloss" in t[m]:
                    d = t[m].get("thinking_minus_model")
                    rows.append(f"<tr class='{'tab' if m.startswith('TabPFN') else 'mkt' if m == 'Market price' else ''}'>"
                                f"<td>{name}</td><td>{E(m)}</td><td class='num'>{t[m]['logloss']:.4f}</td>"
                                f"<td class='num'>{t[m]['auc']:.4f}</td><td class='num'>"
                                f"{'' if not d else f'{d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]'}</td></tr>")
        thinking_html = ("<div class='tw'><table><thead><tr><th>Market</th><th>Model</th><th>Log loss</th><th>AUC</th>"
                         "<th>Thinking − model [95% CI]</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
    ctrl = S.get("_controls")
    ctrl_html = ""
    if ctrl:
        def cr(kind):
            return "".join(
                f"<tr><td>{E(m)}</td><td class='num'>{v['logloss']:.4f}</td><td class='num'>"
                f"{'' if not v['minus_market'] else fmt_ci(v['minus_market'])}</td></tr>"
                for m, v in ctrl[kind].items())
        ctrl_html = (f"<div class='two'><div class='tw'><table><thead><tr><th colspan='3'>Planted signal "
                     f"(corr {ctrl['planted_corr_with_residual']:.2f} with label − price): every model must beat the market</th></tr>"
                     f"<tr><th>Model</th><th>Log loss</th><th>vs market [95% CI]</th></tr></thead><tbody>{cr('planted')}</tbody></table></div>"
                     f"<div class='tw'><table><thead><tr><th colspan='3'>Calibrated-market null (labels drawn from the price): "
                     f"nothing may beat the market</th></tr><tr><th>Model</th><th>Log loss</th><th>vs market [95% CI]</th></tr></thead>"
                     f"<tbody>{cr('null')}</tbody></table></div></div>")

    head = """<title>tab-trader</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=JetBrains+Mono:wght@400;500&family=Source+Sans+3:wght@400;600&display=swap">
<style>""" + CSS + "</style>"

    body = f"""
<div class="wrap">
<div class="eyebrow">TabPFN-3.5 Hackathon · Kalshi KXBTC15M + KXBTCD · data {tot['first_day']} → {tot['last_day']}</div>
<h1>Can a pretrained tabular model price a Bitcoin binary better than the crowd?</h1>
<p class="lede">I have been recording Kalshi's Bitcoin markets four times a second since March:
{total_rows/1e6:,.0f} million rows that cannot be downloaded after the fact. This page tests TabPFN-3.5, running on an
ordinary CPU, against six tuned classic learners and against the market's own price. All of them use only off-the-shelf
momentum features.</p>
<p class="meta">Code, data and notebooks: <a href="https://github.com/Cloblak/tab-trader">github.com/Cloblak/tab-trader</a>
· protocol fixed in advance in <a href="https://github.com/Cloblak/tab-trader/blob/main/PREREG.md">PREREG.md</a>
· every number on this page is read from <code>results/</code></p>
<ol class="answers">{answers}</ol>

<h2><span class="n">1 · THE IDEA</span>Edge means beating the contract's own price</h2>
<p>A Kalshi contract pays $1 if its event happens, so a 63¢ price <i>is</i> the market's 63% forecast. Being right about BTC's
direction is not enough. A model makes money only when its probability is better than the price by more than the
fee, on contracts it can actually buy. So every chart below includes the market price as a competitor, and usually it is the
one to beat.</p>
<p>My live strategy takes this approach: a directional trend signal, then a model that decides which signals are worth paying for.
This page uses a deliberately generic version (about 20 textbook momentum indicators) to compare learners fairly.
Section 7 shows TabPFN inside the real strategy with its features withheld.</p>

<h2><span class="n">2 · THE MARKETS</span>Two Bitcoin markets, 96 + 24 new contracts a day</h2>
<div class="two"><div>
<h3>KXBTC15M · 15-minute up/down</h3>
<p>"Will BTC be higher at the end of this 15-minute window than at its start?" A new market opens every 15 minutes.</p>
</div><div>
<h3>KXBTCD · hourly strike ladder</h3>
<p>"Will BTC be above $K at the top of the hour?" for a ladder of strikes $100 apart. Each strike is its own contract.</p>
</div></div>
<p>Both settle on the 60-second average of the CF Benchmarks BRTI index before close. Buying YES costs the ask; buying NO costs
100 − bid. A taker also pays <code>0.07 × C × (1 − C)</code> dollars per contract, which peaks at {fee_c(50):.2f}¢ for a 50¢ contract.</p>
<figure><div class="controls"><div id="ex-mode"></div><label for="ex-day">day</label><select id="ex-day"></select>
<label for="ex-market">market</label><select id="ex-market"></select></div>
<div id="ex-chart"></div><div id="ex-chart2"></div>
<figcaption id="ex-cap"></figcaption></figure>
<p class="small">The explorer shows the published sample week ({E(str(ex['m15'][0]['open'][:10]))} to
{E(str(ex['m15'][-1]['open'][:10]))}) from <code>data/raw_week/</code>, resampled to 15 s (15-minute) and 1 min (hourly) for
this page. The files themselves are at 1 s and 10 s.</p>

<h2><span class="n">3 · THE DATA</span>Six and a half months of tape nobody else has</h2>
<div class="tiles">
<div class="tile"><div class="v">{tot['tape_15m_rows']/1e6:.1f}M</div><div class="k">15-minute market tape rows (≈4 per second)</div></div>
<div class="tile"><div class="v">{tot['tape_1h_rows']/1e6:.1f}M</div><div class="k">hourly ladder tape rows</div></div>
<div class="tile"><div class="v">{tot['order_book_rows']/1e6:.0f}M</div><div class="k">full L2 order-book snapshots</div></div>
<div class="tile"><div class="v">{tot['labels_15m']:,}</div><div class="k">settled 15-minute markets with labels</div></div>
<div class="tile"><div class="v">{tot['labels_1h_all_legs']:,}</div><div class="k">settled hourly ladder contracts</div></div>
</div>
<p>The collectors write the order book, per-exchange BTC spot, the BRTI settlement index and derived indicators to ClickHouse.
Kalshi's API keeps only about two months of candles and fills, and no sub-second book history. Most of this history therefore
exists only because it was recorded live.</p>
<div class="tw"><table><thead><tr><th>Since</th><th>What is recorded</th></tr></thead><tbody>{milestones}</tbody></table></div>
<figure><div id="eda-daily"></div><figcaption><b>Rows recorded per day.</b> The collectors have run continuously since
{tot['first_day']}. Dips are logged outages, which are excluded from modelling.</figcaption></figure>
<h3>Quality, and the collector upgrade of 30 July</h3>
<figure><div class="controls"><div id="eda-q-ctl"></div></div><div id="eda-q"></div>
<figcaption><b>Weekly share of valid quotes and of crossed books (ask below bid, which is impossible).</b>
Collector v1 stored about 60% valid rows and 2–4% crossed rows. Earlier research on this data showed that crossed rows manufacture
fake backtest profit, because a strategy that hunts cheap prices finds the corrupt ones first. Collector v2 (30 July) fixed both.</figcaption></figure>
<p>Every decision quote used for modelling is then checked against Kalshi's own 1-minute candle for that minute. It is kept only
if it lies inside the candle's range and within 2¢ of the exchange's closing quote:</p>
<div class="tw"><table><thead><tr><th>Market</th><th>Month</th><th>Snapshots checked</th><th>Confirmed</th><th>Median |Δ| vs exchange</th></tr></thead><tbody>{candle_rows}</tbody></table></div>
<p>The hourly tape from before the upgrade fails this check most of the time, so July hourly data is excluded. The check
caught a real collector defect that row-level validity flags could not see.</p>
<h3>The baseline every model must beat</h3>
<figure><div class="controls"><div id="eda-cal-ctl"></div></div><div id="eda-cal"></div>
<figcaption><b>Market mid price vs how often the contract settled YES</b> (clean decision snapshots, 10¢ buckets, 95% band).
The crowd is well calibrated, with a slight lean at the extremes. Any edge is a few points of probability at most, which is
why fees and calibration decide everything.</figcaption></figure>

<h2><span class="n">4 · LIVE TRADING</span>The backtest is checked against real fills</h2>
<p>My strategy has traded real money on this market since {E(fid['period'][0])}, alongside a paper twin that runs the same code
at 1 contract. Over {fid['live_trades']} live trades on {fid['live_days']} days, I compared the backtest replay, the paper twin,
the live journal and Kalshi's own fills, trade by trade:</p>
<div class="tiles">
<div class="tile"><div class="v">{100*vt['live_trades_also_taken_by_twin']:.0f}%</div><div class="k">live trades also taken by the paper twin, same side, same outcome</div></div>
<div class="tile"><div class="v">{100*vb['live_trades_with_backtest_signal']:.1f}%</div><div class="k">live trades the backtest also signals ({100*vb['same_side']:.0f}% same side)</div></div>
<div class="tile"><div class="v">{100*vb['same_outcome_settled']:.1f}%</div><div class="k">same settled outcome, backtest vs live</div></div>
<div class="tile"><div class="v">{ex_['entry_slippage_c_median']:+.1f}¢</div><div class="k">median fill vs quoted price ({ex_['entry_slippage_c_mean']:+.2f}¢ mean)</div></div>
</div>
<div class="tw"><table><thead><tr><th>Same trades, four ways</th><th>n</th><th>¢ / contract</th><th>95% CI (day blocks)</th></tr></thead>
<tbody>{''.join(fid_row(k, v) for k, v in cpc.items())}</tbody></table></div>
<p>The signal logic reproduces almost exactly. The remaining gap is price. Live orders land a median
{vb['entry_time_diff_s_median']:.0f} s after the recorded signal and pay a median {vb['price_diff_c_median']:+.0f}¢ more
({vb['price_diff_c_mean']:+.1f}¢ mean). So the replay overstates profit by about
{cpc['backtest (recorded quotes, live exits)']['c_per_ct'] - cpc['live real (Kalshi fills and fees)']['c_per_ct']:.1f}¢ per contract,
and I budget for that. In the other direction, my journal charges {ex_['journal_fee_c_per_ct']:.2f}¢ in fees per contract
while Kalshi actually bills {ex_['kalshi_fee_c_per_ct']:.2f}¢, so the backtest's fees are conservative. Knowing exactly where and
by how much the backtest diverges is what makes it usable.</p>

<h2><span class="n">5 · CLEANING AND FEATURES</span>One row per market per decision, priced at an instant</h2>
<p>Each row is a settled market at a decision time: 10 and 5 minutes before close for the 15-minute market, 30 and 15
minutes before close for the hourly ladder. A row holds the quote at that instant, the quotes 60 s and 180 s earlier, the
strike, the 1-minute BTC bars that had already closed, and the settlement label.</p>
<div class="two"><div class="tw"><table><thead><tr><th>Cleaning step</th><th>Rows</th></tr></thead><tbody>{funnel}</tbody></table></div>
<div><p class="small">Rules learned the hard way, because each one once produced a fake edge in my research:</p><ul class="small">
<li>Price every decision at a single instant, never with a window average.</li>
<li>Use only collector-validated, uncrossed quotes at most 5 s old.</li>
<li>Confirm every quote against the exchange's own candle.</li>
<li>Use only BTC bars that had closed by the decision time.</li></ul>
<p class="small">Notebook 03 rebuilds the published tables from the raw sample week: 1-minute BTC bars match 100%, and
decision quotes match tick for tick except where the book moved in the final instant.</p></div></div>
<div class="tw"><table><thead><tr><th>Group</th><th>Features (textbook settings, nothing tuned)</th></tr></thead><tbody>{feat_rows}</tbody></table></div>
<figure><div id="feat-rc"></div><figcaption><b>How much each feature knows beyond the price</b>: correlation with
(label − market price) on the 15-minute training period before the first test week. Contract and moneyness features are in blue.
Alone, most features look predictive (hover to see each one's stand-alone AUC), but almost all of that is the price again.
What is left is a set of weak, noisy signals: few rows, little information each. This is the setting where a pretrained prior
should help most.</figcaption></figure>

<h2><span class="n">6 · THE BENCHMARK</span>TabPFN-3.5 vs six tuned classic learners and the market</h2>
<p>The protocol was written down before any test fold ran. Weekly walk-forward blocks are used: 9 blocks for the 15-minute market
(3 Aug – 3 Oct) and 6 for the hourly ladder (24 Aug – 3 Oct). Every model trains only on markets that closed before the test week.
<b>Policy A</b> gives every model the same 5,000 most recent rows, which is TabPFN-3.5's default CPU limit. Classic learners get
their library defaults <i>plus</i> a 20-configuration random search on a validation slice, and keep whichever is better.
TabPFN gets neither. The primary metric is log loss. All results come from one 8-core CPU.</p>
<figure><div class="controls"><div id="lb-mk"></div><div id="lb-pol"></div></div>
<div id="lb-dots"></div><div class="tw" id="lb-table"></div><figcaption id="lb-cap"></figcaption></figure>
<h3>Learning curves: how much data does each model need?</h3>
<figure><div class="controls"><div id="lc-mk"></div><div class="chips" id="lc-chips"></div></div><div id="lc-chart"></div>
<figcaption><b>Test log loss vs training rows</b> (library defaults for classic models, the same test weeks for every point).
The dashed line is the market price. Toggle models with the chips.</figcaption></figure>
<h3>Calibration</h3>
<figure><div class="controls"><div id="rel-mk"></div><label for="rel-model">compare with</label><select id="rel-model"></select></div>
<div class="legend" id="rel-legend"></div><div id="rel-chart"></div>
<figcaption><b>Reliability on the test weeks.</b> A calibrated model follows the diagonal. TabPFN's probabilities arrive
calibrated without any post-processing. Trees are often over-confident at the edges.</figcaption></figure>
<h3>Accuracy per CPU-second</h3>
<figure><div class="controls"><div id="cpu-mk"></div></div><div id="cpu-chart"></div>
<figcaption><b>CPU cost of one weekly refit (including the hyper-parameter search a classic model needs) vs test log loss.</b>
TabPFN's cost is a single forward pass over the training rows: no search, no early stopping, no feature scaling.</figcaption></figure>
<h3>TabPFN-3.5 Thinking (API)</h3>
<p>Thinking mode spends extra inference compute and is told which column is time (<code>time_col</code>). It ran on the same
test weeks using the hackathon API credits, and saw only the published generic features.</p>
{thinking_html}
<h3>From probabilities to trades</h3>
<p>Each model's probabilities are isotonic-calibrated on its validation slice. The rule then buys YES (or NO) only when the
calibrated probability beats the ask plus the taker fee by a margin, also chosen on validation, with at most one trade per market.
This is descriptive: with generic features, nobody should expect a large edge.</p>
<figure><div class="controls"><div id="tr-mk"></div></div><div id="tr-chart"></div><div class="tw" id="tr-table"></div>
<figcaption><b>Out-of-sample trading at 1 contract</b>, net of Kalshi taker fees, trade-weighted, with day-block 95% CIs.</figcaption></figure>
<h3>Does the harness itself work?</h3>
{ctrl_html or '<p class="small">Controls are pending.</p>'}

<h2><span class="n">7 · MY STRATEGY</span>TabPFN as the filter inside the live strategy</h2>
<p>My live 15-minute strategy emits a directional trend signal, and a <b>gate</b> model decides whether the signal is worth
paying for. The current gate is a LightGBM model trained once in spring. Here every candidate gate scores the same
<b>{st['n']:,} out-of-sample signals</b> ({E(st['first'])} → {E(st['last'])}, three 30-day blocks, each trained on the
previous 90 days). The 32 features, the thresholds, the entry window and the exit rules are private. The repo ships only
each model's score, which trades each rule took, and the outcome per contract.</p>
<figure><div id="st-auc"></div><figcaption><b>Ranking skill relative to the market price</b> (AUC difference, day-block 95% CI).
{st_auc_cap}</figcaption></figure>
<figure><div id="st-rules"></div><figcaption><b>Each gate turned into a calibrated-edge rule, against the live gate</b>
(¢ per contract with the live exits, paired by day).</figcaption></figure>
<figure><div id="st-cum"></div><figcaption><b>Cumulative result at 1 contract.</b> {st_cum_cap}</figcaption></figure>
<p class="small">These test blocks were reused across my earlier experiments, so they are not an untouched holdout. The next step
is a forward test. Live deployment of the open TabPFN-3.5 weights would also need a commercial licence or the Prior Labs API.</p>

<h2><span class="n">8 · CONCLUSION</span>What this shows</h2>
<ul>
<li><b>TabPFN-3.5 is a strong default for small, noisy, drifting tables.</b> Untuned, on a CPU, it matches or beats six tuned classic
learners on a problem where most of them do worse than the market price.</li>
<li><b>The hard part is the data, not the model.</b> Point-in-time pricing, exchange-confirmed quotes and a measured live-vs-backtest
gap are what make any of these numbers trustworthy.</li>
<li><b>Generic momentum is nearly priced in.</b> No model, TabPFN included, finds a large edge from off-the-shelf indicators. The
value of TabPFN here is that it extracts what little there is, with calibrated probabilities, cheaply enough to refit every week.</li>
</ul>
<h3>Reproduce it</h3>
<pre>git clone https://github.com/Cloblak/tab-trader && cd tab-trader
uv sync                        # or: pip install -e .
export TABPFN_TOKEN=...        # free at https://ux.priorlabs.ai
make quick                     # one test week, both markets, about 10 min on CPU
make benchmark                 # everything (a few hours on 8 cores; resumable)
make analyze report            # results/summary.json → docs/index.html</pre>
</div>
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
