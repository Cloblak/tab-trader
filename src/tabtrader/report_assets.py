"""CSS and JavaScript for docs/index.html (plain SVG charts, no chart library)."""

CSS = r"""
:root{/* layout: one 920px column; navy marks TabPFN, grays mark everything else; white page */
--bg:#ffffff;--fg:#1b2533;--muted:#5f6b7a;--faint:#e6e9ee;--rule:#d3d9e0;--surface:#ffffff;--panel:#f5f7fa;
--c1:#1d3a6e;--c2:#5a8fd4;--c3:#a9c4ea;--gray:#7a8594;--classic:#b4bcc7;--neg:#a83232;--band:rgba(29,58,110,.06);
--sans:Arial,"Helvetica Neue",Helvetica,sans-serif;--mono:Consolas,"SF Mono",Menlo,monospace;color-scheme:light}
*{box-sizing:border-box}
html,body{background:var(--bg)}
body{margin:0;color:var(--fg);font-family:var(--sans);font-size:16px;line-height:1.55}
.top{border-top:6px solid var(--c1)}
.wrap{max-width:920px;margin:0 auto;padding-inline:20px;padding-block:28px 72px}
h1{font-size:clamp(1.7rem,4.2vw,2.35rem);line-height:1.15;margin:.1em 0 .3em;color:var(--c1);text-wrap:balance;letter-spacing:-.01em}
h2{font-size:1.35rem;color:var(--c1);margin:2.4em 0 .4em;padding-top:.9em;border-top:1px solid var(--rule);text-wrap:balance}
h3{font-size:1.02rem;margin:1.6em 0 .3em;color:var(--fg)}
p,li{max-width:70ch}
a{color:var(--c1)}
code{font-family:var(--mono);font-size:.86em;background:var(--panel);padding:1px 4px;border-radius:3px}
.kicker{font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);font-weight:bold}
.sub{font-size:1.06rem;color:var(--muted);max-width:66ch;margin:0 0 1.2em}
.meta{color:var(--muted);font-size:.9rem}
.bluf{border:1px solid var(--rule);border-radius:4px;padding:16px 20px;margin:1.2em 0 1.6em;background:var(--panel)}
.bluf h2{margin:0 0 .5em;padding:0;border:0;font-size:1rem;letter-spacing:.08em;text-transform:uppercase}
.bluf ul{margin:0;padding-left:1.2em;display:grid;gap:.45em}
.bluf li{max-width:none}
.note{border:1px solid var(--rule);border-radius:4px;padding:10px 14px;font-size:.92rem;color:var(--fg);margin:.8em 0 1.2em}
.note b{color:var(--c1)}
.what{font-size:.92rem;color:var(--muted);margin:.35em 0 0}
.what b{color:var(--fg)}
figure{margin:1em 0 1.6em;position:relative}
.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
.panels>div{min-width:0;border:1px solid var(--faint);border-radius:4px;padding:8px 8px 2px;position:relative}
.panels h4{margin:0 0 2px;font-size:.82rem;color:var(--muted);font-weight:bold}
@media (max-width:760px){.panels{grid-template-columns:1fr}}
.controls{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin:0 0 8px;font-size:.88rem}
.controls label{color:var(--muted)}
.seg{display:inline-flex;border:1px solid var(--rule);border-radius:4px;overflow:hidden}
.seg button{font:inherit;font-size:.85rem;background:#fff;color:var(--fg);border:0;padding:4px 11px;cursor:pointer}
.seg button+button{border-left:1px solid var(--rule)}
.seg button[aria-pressed="true"]{background:var(--c1);color:#fff}
.seg button:focus-visible,select:focus-visible{outline:2px solid var(--c2);outline-offset:2px}
select{font:inherit;font-size:.86rem;background:#fff;color:var(--fg);border:1px solid var(--rule);border-radius:4px;padding:3px 6px;max-width:100%}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:.84rem;color:var(--muted);margin:2px 0 6px}
.legend i{display:inline-block;width:18px;height:0;border-top:3px solid;margin-right:6px;vertical-align:4px}
svg.chart{width:100%;height:auto;display:block;font-family:var(--sans);overflow:visible}
.chart .grid{stroke:var(--faint);stroke-width:1}
.chart .axis{stroke:var(--rule);stroke-width:1}
.chart .tick{fill:var(--muted);font-size:11px}
.chart .lab{fill:var(--fg);font-size:11.5px}
.chart .val{fill:var(--muted);font-size:11px}
.chart .ln{fill:none;stroke-linejoin:round;stroke-linecap:round}
.chart .hit{fill:transparent;cursor:crosshair}
.chart .xhair{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}
.tip{position:absolute;pointer-events:none;background:#fff;color:var(--fg);border:1px solid var(--rule);border-radius:4px;padding:6px 9px;font-size:.8rem;line-height:1.35;box-shadow:0 3px 10px rgba(27,37,51,.12);white-space:nowrap;z-index:5}
.tw{overflow-x:auto;margin:.6em 0 1.2em}
table{border-collapse:collapse;font-size:.88rem;font-variant-numeric:tabular-nums;min-width:100%}
th{text-align:left;font-weight:bold;color:var(--c1);font-size:.78rem;border-bottom:2px solid var(--c1);padding:6px 12px 6px 0;white-space:nowrap}
td{border-bottom:1px solid var(--faint);padding:6px 12px 6px 0;vertical-align:top}
td.num{white-space:nowrap}
tr.tab td{font-weight:bold;color:var(--c1)}
tr.mkt td:first-child{font-style:italic}
td.neg{color:var(--neg)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:1em 0}
.tile{border:1px solid var(--faint);border-radius:4px;padding:10px 12px}
.tile .v{font-size:1.4rem;font-weight:bold;color:var(--c1);font-variant-numeric:tabular-nums}
.tile .k{color:var(--muted);font-size:.84rem;line-height:1.3}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,380px),1fr));gap:14px}
.two>*{min-width:0}
pre{background:var(--panel);border:1px solid var(--faint);border-radius:4px;padding:12px 14px;overflow-x:auto;font-family:var(--mono);font-size:.82rem;line-height:1.5}
.small{font-size:.86rem;color:var(--muted)}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
@media (max-width:520px){body{font-size:15px}.tile .v{font-size:1.2rem}}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const NS = 'http://www.w3.org/2000/svg';
const STYLE = {'TabPFN-3.5 + Kelly':{c:'var(--c1)',w:3},'TabPFN-3.5':{c:'var(--c2)',w:2.2},'TabPFN-3.5-Fast':{c:'var(--c2)',w:2.2},'TabPFN-3.5-Thinking':{c:'var(--c3)',w:2.2},
  'Market price':{c:'var(--fg)',w:1.3,d:'2 3'},'Live gate':{c:'var(--gray)',w:2,d:'7 4'},'Every signal':{c:'var(--classic)',w:1.6,d:'3 3'}};
const NAME = m => ({'Live gate':'My current filter','Every signal':'Take every signal','TabPFN-3.5 + Kelly':'TabPFN-3.5 + Kelly sizing','TabPFN-3.5':'TabPFN-3.5, flat 15%'})[m]||m.replace(' · tuned','').replace(' · default','');
const BENCH = {'TabPFN-3.5':'var(--c1)','TabPFN-3.5-Fast':'var(--c2)','TabPFN-3.5-Thinking':'var(--c3)','Market price':'var(--fg)'};
const COLOR = m => BENCH[m]||(STYLE[m]||{c:'var(--classic)'}).c;
const BNAME = m => m.replace(' · tuned','').replace(' · default','');
function el(tag,attrs={},parent){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e;}
function txt(p,x,y,s,cls,anchor='start'){const t=el('text',{x,y,class:cls,'text-anchor':anchor},p);t.textContent=s;return t;}
function niceTicks(lo,hi,n=5){const span=hi-lo||1;const s0=span/n;const mag=Math.pow(10,Math.floor(Math.log10(s0)));const e=s0/mag;const st=(e>=7.5?10:e>=3.5?5:e>=1.5?2:1)*mag;const out=[];for(let v=Math.ceil(lo/st)*st;v<=hi+1e-9;v+=st)out.push(+v.toFixed(10));return out;}
function fmt(v,d=2){return v==null?'–':(+v).toFixed(d);}
function sgn(v,d=3){return v==null?'–':(v>=0?'+':'')+(+v).toFixed(d);}
function money(v){return v==null?'–':'$'+(+v).toLocaleString(undefined,{maximumFractionDigits:v>=1000?0:2,minimumFractionDigits:v>=1000?0:2});}
function host2fig(h){return h.closest('figure')||h.parentElement;}
function tipFor(fig){let t=null;for(const c of fig.children)if(c.classList&&c.classList.contains('tip'))t=c;if(!t){t=document.createElement('div');t.className='tip';t.hidden=true;fig.appendChild(t);}return t;}
function showTip(fig,h,evt){const t=tipFor(fig);t.innerHTML=h;t.hidden=false;const r=fig.getBoundingClientRect();let x=evt.clientX-r.left+12,y=evt.clientY-r.top+12;const w=t.offsetWidth;if(x+w>r.width-4)x=evt.clientX-r.left-w-12;if(x<4)x=4;t.style.left=x+'px';t.style.top=y+'px';}
function hideTip(fig){const t=tipFor(fig);t.hidden=true;}

function lineChart(host,series,o={}){
  host.innerHTML='';const W=o.w||720,H=o.h||300,L=o.left||52,R=o.right||16,T=12,B=34;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,class:'chart',role:'img','aria-label':o.label||'chart'},host);
  const xs=series.flatMap(s=>s.pts.map(p=>p[0])),ys=series.flatMap(s=>s.pts.flatMap(p=>p.slice(1))).filter(v=>v!=null);
  const tx=o.x==='time'?(v=>new Date(v).getTime()):(o.x==='log'?(v=>Math.log10(v)):(v=>v));
  let x0=Math.min(...xs.map(tx)),x1=Math.max(...xs.map(tx));if(x0===x1){x0-=1;x1+=1;}
  const ty=o.ylog?(v=>Math.log10(Math.max(v,1e-3))):(v=>v);
  let y0=o.y0!=null?o.y0:Math.min(...ys.map(ty)),y1=o.y1!=null?o.y1:Math.max(...ys.map(ty));
  (o.ref||[]).forEach(r=>{y0=Math.min(y0,ty(r.y));y1=Math.max(y1,ty(r.y));});
  if(o.ylog){y0=Math.floor(y0);y1=Math.ceil(y1);if(y0===y1)y1+=1;}
  else{const pad=(y1-y0)*0.06||1;if(o.y0==null)y0-=pad;if(o.y1==null)y1+=pad;}
  const X=v=>L+(tx(v)-x0)/(x1-x0)*(W-L-R),Y=v=>T+(1-(ty(v)-y0)/(y1-y0))*(H-T-B);
  const yt=o.ylog?Array.from({length:y1-y0+1},(_,i)=>Math.pow(10,y0+i)):niceTicks(y0,y1,o.ny||5);
  yt.forEach(v=>{el('line',{x1:L,x2:W-R,y1:Y(v),y2:Y(v),class:'grid'},svg);txt(svg,L-6,Y(v)+3.5,(o.yfmt||(a=>a))(v),'tick','end');});
  let xt;if(o.x==='time'){const days=(x1-x0)/864e5;const step=o.xstep||(days>120?30:days>40?14:days>10?7:1);xt=[];const d0=new Date(x0);for(let d=new Date(Date.UTC(d0.getUTCFullYear(),d0.getUTCMonth(),d0.getUTCDate()));d.getTime()<=x1;d=new Date(d.getTime()+step*864e5))xt.push(d.getTime());}
  else if(o.x==='log')xt=(o.xticks||[]).map(v=>Math.log10(v));else xt=o.xticks||niceTicks(x0,x1,6);
  xt.forEach(v=>{const xx=L+(v-x0)/(x1-x0)*(W-L-R);if(xx<L-1||xx>W-R+1)return;el('line',{x1:xx,x2:xx,y1:H-B,y2:H-B+4,class:'axis'},svg);txt(svg,xx,H-B+16,o.x==='time'?new Date(v).toISOString().slice(5,10):o.x==='log'?String(Math.round(Math.pow(10,v))):(o.xfmt||(a=>a))(v),'tick','middle');});
  el('line',{x1:L,x2:W-R,y1:H-B,y2:H-B,class:'axis'},svg);
  if(o.xlabel)txt(svg,(L+W-R)/2,H-2,o.xlabel,'tick','middle');
  if(o.ylabel){const t=txt(svg,12,T+(H-T-B)/2,o.ylabel,'tick','middle');t.setAttribute('transform',`rotate(-90 12 ${T+(H-T-B)/2})`);}
  (o.ref||[]).forEach(r=>{el('line',{x1:L,x2:W-R,y1:Y(r.y),y2:Y(r.y),stroke:r.color||'var(--fg)','stroke-dasharray':'5 4','stroke-width':1.2},svg);if(r.label)txt(svg,W-R-4,Y(r.y)-5,r.label,'val','end');});
  series.forEach(s=>{if(s.area){const top=s.pts.map(p=>`${X(p[0])},${Y(p[2])}`).join(' ');const bot=s.pts.slice().reverse().map(p=>`${X(p[0])},${Y(p[1])}`).join(' ');el('polygon',{points:top+' '+bot,fill:s.color,opacity:.15},svg);return;}
    const segs=[];let cur=[];s.pts.forEach(p=>{if(p[1]==null){if(cur.length)segs.push(cur);cur=[];}else cur.push(`${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`);});if(cur.length)segs.push(cur);
    segs.forEach(sg=>el('polyline',{points:sg.join(' '),class:'ln',stroke:s.color,'stroke-width':s.width||2,'stroke-dasharray':s.dash||'',opacity:s.opacity||1},svg));
    if(s.marker)s.pts.forEach(p=>{if(p[1]!=null)el('circle',{cx:X(p[0]),cy:Y(p[1]),r:3.5,fill:s.color,stroke:'#fff','stroke-width':1.5},svg);});
    if(s.endLabel){const p=s.pts[s.pts.length-1];txt(svg,X(p[0])+6,Y(p[1])+4,s.endLabel,'lab');}});
  const fig=host2fig(host);const xh=el('line',{y1:T,y2:H-B,class:'xhair',visibility:'hidden'},svg);
  const hit=el('rect',{x:L,y:T,width:W-L-R,height:H-T-B,class:'hit'},svg);
  const allx=[...new Set(series.filter(s=>!s.area&&!s.nohover).flatMap(s=>s.pts.map(p=>p[0])))].sort((a,b)=>tx(a)-tx(b));
  hit.addEventListener('pointermove',evt=>{const pt=svg.createSVGPoint();pt.x=evt.clientX;pt.y=evt.clientY;const sp=pt.matrixTransform(svg.getScreenCTM().inverse());let best=allx[0],bd=1e18;allx.forEach(v=>{const d=Math.abs(X(v)-sp.x);if(d<bd){bd=d;best=v;}});
    xh.setAttribute('x1',X(best));xh.setAttribute('x2',X(best));xh.setAttribute('visibility','visible');
    const rows=series.filter(s=>!s.area&&!s.nohover).map(s=>{const p=s.pts.find(q=>q[0]===best);return p&&p[1]!=null?{s,v:p[1]}:null;}).filter(Boolean).sort((a,b)=>b.v-a.v).slice(0,o.tipmax||12)
      .map(({s,v})=>`<div><span style="color:${s.color}">■</span> ${s.name}: <b>${(o.tipfmt||(a=>fmt(a,3)))(v)}</b></div>`).join('');
    showTip(fig,`<div class="meta">${o.x==='time'?new Date(best).toISOString().slice(0,10):(o.tipx||(a=>a))(best)}</div>${rows}`,evt);});
  hit.addEventListener('pointerleave',()=>{xh.setAttribute('visibility','hidden');hideTip(fig);});
  return svg;
}
function dotCI(host,rows,o={}){
  host.innerHTML='';const W=720,rowH=o.rowH||28,L=o.left||210,R=72,T=8,B=30,H=T+B+rows.length*rowH;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,class:'chart',role:'img','aria-label':o.label||'chart'},host);
  let lo=Math.min(0,...rows.map(r=>r.lo??r.mean)),hi=Math.max(0,...rows.map(r=>r.hi??r.mean));const pad=(hi-lo)*.08||.01;lo-=pad;hi+=pad;
  const X=v=>L+(v-lo)/(hi-lo)*(W-L-R);
  niceTicks(lo,hi,5).forEach(v=>{el('line',{x1:X(v),x2:X(v),y1:T,y2:H-B,class:'grid'},svg);txt(svg,X(v),H-B+15,(o.xfmt||(a=>a))(v),'tick','middle');});
  el('line',{x1:X(0),x2:X(0),y1:T,y2:H-B,stroke:'var(--fg)','stroke-width':1.2},svg);
  if(o.xlabel)txt(svg,(L+W-R)/2,H-3,o.xlabel,'tick','middle');
  const fig=host2fig(host);
  rows.forEach((r,i)=>{const y=T+i*rowH+rowH/2;const g=el('g',{},svg);
    if(r.lo!=null)el('line',{x1:X(r.lo),x2:X(r.hi),y1:y,y2:y,stroke:r.color,'stroke-width':2.4,'stroke-linecap':'round'},g);
    el('circle',{cx:X(r.mean),cy:y,r:5.5,fill:r.color,stroke:'#fff','stroke-width':2},g);
    const t=txt(svg,L-10,y+4,r.label,'lab','end');if(r.bold)t.setAttribute('font-weight','bold');
    txt(svg,W-R+8,y+4,(o.vfmt||(a=>sgn(a,3)))(r.mean),'val');
    const h=el('rect',{x:0,y:y-rowH/2,width:W,height:rowH,class:'hit'},g);
    h.addEventListener('pointermove',e=>showTip(fig,r.tip||`<b>${r.label}</b><br>${sgn(r.mean,4)} [${sgn(r.lo,4)}, ${sgn(r.hi,4)}]`,e));h.addEventListener('pointerleave',()=>hideTip(fig));});
}
function bars(host,cats,series,o={}){
  host.innerHTML='';const W=720,H=o.h||240,L=48,R=10,T=18,B=40;
  const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,class:'chart',role:'img','aria-label':o.label||'chart'},host);
  const vals=series.flatMap(s=>s.v.filter(v=>v!=null));let y0=Math.min(0,...vals),y1=Math.max(0,...vals);const pad=(y1-y0)*.08||1;y0-=pad;y1+=pad;
  const Y=v=>T+(1-(v-y0)/(y1-y0))*(H-T-B);const bw=(W-L-R)/cats.length;const gw=bw*0.78/series.length;
  (o.bands||[]).forEach(b=>{const x0=L+b.from*bw,x1=L+b.to*bw;el('rect',{x:x0,y:T-14,width:x1-x0,height:H-T-B+14,fill:b.fill},svg);txt(svg,(x0+x1)/2,T-3,b.label,'val','middle');});
  niceTicks(y0,y1,5).forEach(v=>{el('line',{x1:L,x2:W-R,y1:Y(v),y2:Y(v),class:'grid'},svg);txt(svg,L-6,Y(v)+3.5,(o.yfmt||(a=>a))(v),'tick','end');});
  el('line',{x1:L,x2:W-R,y1:Y(0),y2:Y(0),stroke:'var(--fg)','stroke-width':1},svg);
  const fig=host2fig(host);
  cats.forEach((c,i)=>{const gx=L+i*bw+bw*0.11;series.forEach((s,j)=>{const v=s.v[i];if(v==null)return;const x=gx+j*gw;const y=Y(Math.max(v,0)),h=Math.abs(Y(v)-Y(0));
      el('rect',{x:x+0.5,y,width:Math.max(gw-1,1),height:Math.max(h,0.5),fill:s.color},svg);});
    if(i%(o.labelEvery||1)===0)txt(svg,L+i*bw+bw/2,H-B+14,c.slice(5),'tick','middle');
    const hb=el('rect',{x:L+i*bw,y:T,width:bw,height:H-T-B,class:'hit'},svg);
    hb.addEventListener('pointermove',e=>showTip(fig,`<div class="meta">week ending ${c}</div>`+series.map(s=>`<div><span style="color:${s.color}">■</span> ${s.name}: <b>${s.v[i]==null?'–':sgn(s.v[i],1)+'%'}</b></div>`).join(''),e));hb.addEventListener('pointerleave',()=>hideTip(fig));});
}
function seg(host,options,value,onChange){host.innerHTML='';host.className='seg';options.forEach(([v,l])=>{const b=document.createElement('button');b.type='button';b.textContent=l;b.setAttribute('aria-pressed',String(v===value));b.onclick=()=>{[...host.children].forEach(c=>c.setAttribute('aria-pressed','false'));b.setAttribute('aria-pressed','true');onChange(v);};host.appendChild(b);});}
function legendHTML(models){return models.map(m=>{const s=STYLE[m]||{c:'var(--classic)'};return `<span><i style="border-top-color:${s.c};border-top-style:${s.d?'dashed':'solid'}"></i>${NAME(m)}</span>`;}).join('');}

/* ===== bottom line: real strategy ===== */
const BL=D.bluf, MODELS=D.blufModels;
function eqSeries(sp){return MODELS.map(m=>{const s=STYLE[m]||{c:'var(--classic)',w:1.1};return {name:NAME(m),color:s.c,width:s.w,dash:s.d||'',pts:BL.splits[sp][m].curve,opacity:STYLE[m]?1:.9};}).reverse();}
const short=v=>v>=1e6?'$'+v/1e6+'M':v>=1e3?'$'+v/1e3+'k':v>=1?'$'+v:'$'+v;
['train','test','holdout'].forEach(sp=>lineChart(document.getElementById('eq-'+sp),eqSeries(sp),{x:'time',w:400,h:270,left:52,ylog:true,yfmt:short,tipfmt:money,label:'account value, '+sp}));
document.getElementById('eq-legend').innerHTML=legendHTML(['TabPFN-3.5 + Kelly','TabPFN-3.5','Live gate','Every signal','Market price'])+'<span><i style="border-top-color:var(--classic)"></i>six other ML models (hover for names)</span>';
(function(){const host=document.getElementById('ts-table'),ctl=document.getElementById('ts-split');
  function draw(sp){const S=BL.splits[sp];const rows=MODELS.map(m=>({m,...S[m]})).sort((a,b)=>b.final-a.final);
    host.innerHTML=`<table><thead><tr><th>Trade filter</th><th>$100 becomes</th><th>Return</th><th>Sharpe</th><th>Sortino</th><th>Max drawdown</th><th>Trades</th><th>Win rate</th><th>¢ / contract</th><th>Weeks up</th></tr></thead><tbody>${rows.map(r=>`<tr class="${r.m.startsWith('TabPFN')?'tab':r.m==='Market price'?'mkt':''}"><td>${NAME(r.m)}</td><td class="num">${money(r.final)}</td><td class="num ${r.return_pct<0?'neg':''}">${sgn(r.return_pct,1)}%</td><td class="num">${fmt(r.sharpe,2)}</td><td class="num">${fmt(r.sortino,2)}</td><td class="num">${fmt(r.max_dd_pct,1)}%</td><td class="num">${r.trades}</td><td class="num">${r.win_rate==null?'–':fmt(100*r.win_rate,1)+'%'}</td><td class="num">${sgn(r.c_per_ct,2)}</td><td class="num">${r.weeks_up}</td></tr>`).join('')}</tbody></table>`;}
  seg(ctl,[['train','Train'],['test','Test'],['holdout','Holdout (live period)']],'holdout',draw);draw('holdout');})();
(function(){const pick=['TabPFN-3.5 + Kelly','Live gate'];const weeks=[];const bands=[];
  ['train','test','holdout'].forEach(sp=>{const w=BL.splits[sp]['TabPFN-3.5 + Kelly'].weekly.map(x=>x[0]);bands.push({from:weeks.length,to:weeks.length+w.length,label:{train:'Train',test:'Test',holdout:'Holdout'}[sp],fill:sp==='holdout'?'rgba(29,58,110,.09)':sp==='test'?'rgba(29,58,110,.045)':'rgba(29,58,110,0)'});weeks.push(...w.map(x=>[sp,x]));});
  const ser=pick.map(m=>({name:NAME(m),color:m.startsWith('TabPFN')?'var(--c1)':'var(--classic)',v:weeks.map(([sp,wk])=>{const r=BL.splits[sp][m].weekly.find(x=>x[0]===wk);return r?r[1]:null;})}));
  bars(document.getElementById('wk-bars'),weeks.map(x=>x[1]),ser,{yfmt:a=>a+'%',labelEvery:3,bands,label:'weekly returns'});})();
(function(){const L=BL.live.series;const order=['TabPFN + Kelly (backtest)','Live gate (backtest)','TabPFN veto, flat 15% (real fills)','Live, as traded (real fills)','TabPFN veto + Kelly sizing (real fills)'];
  const sty={'TabPFN + Kelly (backtest)':{c:'var(--c2)',w:1.8,d:'6 3'},'Live gate (backtest)':{c:'var(--classic)',w:1.6,d:'6 3'},'TabPFN veto, flat 15% (real fills)':{c:'var(--c2)',w:2.2},'Live, as traded (real fills)':{c:'var(--gray)',w:2.4},'TabPFN veto + Kelly sizing (real fills)':{c:'var(--c1)',w:3}};
  const ks=order.filter(k=>L[k]);
  lineChart(document.getElementById('live-chart'),ks.map(k=>({name:k,color:sty[k].c,width:sty[k].w,dash:sty[k].d||'',pts:L[k].curve})),{x:'time',xstep:3,ylog:true,yfmt:short,tipfmt:money,h:300,ylabel:'account value (log scale)',label:'live period'});
  document.getElementById('live-legend').innerHTML=ks.map(k=>`<span><i style="border-top-color:${sty[k].c};border-top-style:${sty[k].d?'dashed':'solid'}"></i>${k}</span>`).join('');})();

/* ===== explorer ===== */
(function(){
  const sel=document.getElementById('ex-market'),dsel=document.getElementById('ex-day'),mode=document.getElementById('ex-mode');
  const host=document.getElementById('ex-chart'),host2=document.getElementById('ex-chart2'),cap=document.getElementById('ex-cap');let M='15';
  const days15=[...new Set(D.ex.m15.map(m=>m.open.slice(0,10)))],days1h=[...new Set(D.ex.m1h.map(m=>m.close.slice(0,10)))];
  function fillDays(){const ds=M==='15'?days15:days1h;dsel.innerHTML=ds.map(d=>`<option value="${d}">${d}</option>`).join('');dsel.value=ds[Math.min(3,ds.length-1)];fillMarkets();}
  function fillMarkets(){const d=dsel.value;if(M==='15'){const ms=D.ex.m15.filter(m=>m.open.startsWith(d));sel.innerHTML=ms.map(m=>`<option value="${D.ex.m15.indexOf(m)}">${m.open.slice(11)} UTC · ${m.y?'settled YES':'settled NO'}</option>`).join('');}
    else{const ms=D.ex.m1h.filter(m=>m.close.startsWith(d));sel.innerHTML=ms.map(m=>`<option value="${D.ex.m1h.indexOf(m)}">closes ${m.close.slice(11)} UTC · ${m.legs.length} strikes</option>`).join('');}
    const o=sel.options[Math.floor(sel.options.length*0.55)];if(o)sel.value=o.value;draw();}
  function draw(){const i=+sel.value;if(isNaN(i))return;
    if(M==='15'){const m=D.ex.m15[i];host2.hidden=false;
      lineChart(host,[{name:'ask',color:'var(--c2)',pts:m.s.map((s,j)=>[s/60,m.a[j]]),width:1.1},{name:'mid',color:'var(--c1)',pts:m.s.map((s,j)=>[s/60,(m.a[j]+m.b[j])/2]),width:2.4},{name:'bid',color:'var(--c2)',pts:m.s.map((s,j)=>[s/60,m.b[j]]),width:1.1}],
        {x:'lin',xticks:[0,3,5,10,12,15],y0:0,y1:100,ylabel:'YES price (¢)',xlabel:'minutes since open',tipx:a=>`minute ${fmt(a,2)}`,tipfmt:a=>fmt(a,1)+'¢',h:240,label:'contract price'});
      lineChart(host2,[{name:'BTC',color:'var(--fg)',pts:m.s.map((s,j)=>[s/60,m.x[j]]),width:1.6}],{x:'lin',xticks:[0,3,5,10,12,15],ref:[{y:m.k,label:'start price $'+m.k.toLocaleString()}],ylabel:'BTC (USD)',xlabel:'minutes since open',tipx:a=>`minute ${fmt(a,2)}`,tipfmt:a=>'$'+Math.round(a).toLocaleString(),h:190,left:64,label:'BTC'});
      cap.innerHTML=`<b>What it shows:</b> one 15-minute market. The contract price (top) follows whether BTC (bottom) is above its start price. This one settled <b>${m.y?'YES':'NO'}</b>.`;}
    else{const ev=D.ex.m1h[i];host2.hidden=true;
      lineChart(host,ev.legs.map(l=>({name:`$${Math.round(l.k).toLocaleString()} ${l.y?'(YES)':'(NO)'}`,color:l.y?'var(--c1)':'var(--classic)',pts:l.s.map((s,q)=>[s,l.p[q]]),width:2,endLabel:`$${Math.round(l.k).toLocaleString()}`})),
        {x:'lin',xticks:[0,15,30,45,60],y0:0,y1:100,ylabel:'P(above strike), ¢',xlabel:'minutes since open',tipx:a=>`minute ${a}`,tipfmt:a=>fmt(a,1)+'¢',h:300,right:74,label:'hourly ladder'});
      cap.innerHTML=`<b>What it shows:</b> one hour of the strike ladder, closing ${ev.close} UTC. Each line is one strike. Navy strikes settled YES, gray ones NO.`;}}
  seg(mode,[['15','15-minute'],['1h','hourly ladder']],'15',v=>{M=v;fillDays();});dsel.onchange=fillMarkets;sel.onchange=draw;fillDays();})();

/* ===== data ===== */
lineChart(document.getElementById('eda-daily'),[{name:'15-minute tape',color:'var(--c1)',pts:D.eda.daily['15m'].map(r=>[r[0],r[1]/1e3])},{name:'hourly tape',color:'var(--c2)',pts:D.eda.daily['1h'].map(r=>[r[0],r[1]/1e3])}],{x:'time',ylabel:'thousand rows / day',tipfmt:a=>fmt(a,0)+'k',h:230,label:'rows per day'});
lineChart(document.getElementById('eda-q'),[{name:'15-minute',color:'var(--c1)',pts:D.eda.weekly['15-minute'].map(r=>[r[0],r[1]]),marker:true},{name:'hourly',color:'var(--c2)',pts:D.eda.weekly['hourly'].map(r=>[r[0],r[1]]),marker:true}],{x:'time',y0:0,y1:100,ylabel:'clean quotes, % of rows',tipfmt:a=>fmt(a,1)+'%',h:230,label:'weekly quality'});
(function(){const host=document.getElementById('eda-cal'),ctl=document.getElementById('eda-cal-ctl');
  function draw(k){const c=D.eda.calib[k].filter(r=>r.n>=50);lineChart(host,[{name:'perfect',color:'var(--muted)',pts:[[0,0],[1,1]],dash:'4 4',width:1,nohover:true},{name:'95% band',color:'var(--c1)',area:true,pts:c.map(r=>[r.p,r.lo,r.hi])},{name:'settled YES rate',color:'var(--c1)',pts:c.map(r=>[r.p,r.y]),marker:true}],{x:'lin',xticks:[0,.2,.4,.6,.8,1],xfmt:a=>a.toFixed(1),y0:0,y1:1,yfmt:a=>a.toFixed(1),xlabel:'market price (as probability)',ylabel:'how often it settled YES',tipx:a=>'price '+fmt(a,2),tipfmt:a=>fmt(a,3),h:300,label:'market calibration'});}
  seg(ctl,[['15m','15-minute'],['1h','hourly']],'15m',draw);draw('15m');})();

/* ===== features ===== */
dotCI(document.getElementById('feat-rc'),D.feat.slice().sort((a,b)=>Math.abs(b.rc)-Math.abs(a.rc)).map(f=>({label:f.f,mean:f.rc,color:f.g==='contract'||f.g==='moneyness'?'var(--c1)':'var(--classic)',tip:`<b>${f.f}</b> (${f.g})<br>alone, AUC ${fmt(f.auc,3)}<br>beyond the price: ${sgn(f.rc,3)}`})),{rowH:19,left:170,xlabel:'correlation with what the price misses',vfmt:a=>sgn(a,3),label:'feature information'});

/* ===== benchmark ===== */
const B=D.bench;
(function(){const dots=document.getElementById('lb-dots'),tbl=document.getElementById('lb-table'),cm=document.getElementById('lb-mk'),cp=document.getElementById('lb-pol'),cap=document.getElementById('lb-cap');let mk='15m',pol='policy_A';
  function draw(){const L=B[mk][pol];if(!L){tbl.innerHTML='<p class="small">Not run yet.</p>';dots.innerHTML='';cap.innerHTML='';return;}
    const rows=Object.entries(L.models).map(([m,r])=>({m,...r})).sort((a,b)=>a.logloss-b.logloss);
    dotCI(dots,rows.filter(r=>r.vs_ref).sort((a,b)=>a.vs_ref.mean-b.vs_ref.mean).map(r=>({label:BNAME(r.m),mean:r.vs_ref.mean,lo:r.vs_ref.lo,hi:r.vs_ref.hi,color:COLOR(r.m),bold:r.m.startsWith('TabPFN')||r.m==='Market price'})),{xlabel:`extra prediction error vs ${L.ref} (right = worse)`,vfmt:a=>sgn(a,4),label:'paired differences'});
    tbl.innerHTML=`<table><thead><tr><th>Model</th><th>Log loss</th><th>vs ${L.ref}</th><th>Significant?</th><th>AUC</th><th>Calibration error</th></tr></thead><tbody>${rows.map(r=>`<tr class="${r.m.startsWith('TabPFN')?'tab':r.m==='Market price'?'mkt':''}"><td>${BNAME(r.m)}</td><td class="num">${fmt(r.logloss,4)}</td><td class="num">${r.vs_ref?sgn(r.vs_ref.mean,4):'reference'}</td><td class="num">${r.vs_ref?(r.vs_ref.bh_pass?'yes':'no'):''}</td><td class="num">${fmt(r.auc,4)}</td><td class="num">${fmt(r.ece,4)}</td></tr>`).join('')}</tbody></table>`;
    cap.innerHTML=`<b>What it shows:</b> how much worse each model predicts than ${L.ref}, with a 95% range. Right of zero means worse. ${L.n_rows.toLocaleString()} test predictions over ${L.n_days} days.`;}
  seg(cm,[['15m','15-minute'],['1h','hourly']],mk,v=>{mk=v;draw();});seg(cp,[['policy_A','same 5,000 rows'],['policy_B','others get all history']],pol,v=>{pol=v;draw();});draw();})();
(function(){const host=document.getElementById('lc-chart'),cm=document.getElementById('lc-mk');let mk='15m';
  function draw(){const lc=B[mk].learning_curve||{};const ns=Object.keys(lc).filter(n=>Object.keys(lc[n]).length).map(Number);if(!ns.length){host.innerHTML='<p class="small">Not run yet.</p>';return;}const models=[...new Set(ns.flatMap(n=>Object.keys(lc[String(n)])))];
    const mkt=(lc[String(ns[ns.length-1])]||{})['Market price'];
    lineChart(host,models.filter(m=>m!=='Market price').map(m=>({name:BNAME(m),color:COLOR(m),width:m.startsWith('TabPFN')?2.8:1.2,marker:true,pts:ns.map(n=>[n,(lc[String(n)][m]||{}).logloss??null])})),{x:'log',xticks:ns,ref:mkt?[{y:mkt.logloss,label:'market price'}]:[],xlabel:'training rows',ylabel:'prediction error (log loss)',tipx:a=>`${a.toLocaleString()} rows`,tipfmt:a=>fmt(a,4),h:300,yfmt:a=>a.toFixed(3),label:'learning curves'});}
  seg(cm,[['15m','15-minute'],['1h','hourly']],mk,v=>{mk=v;draw();});draw();})();
(function(){const host=document.getElementById('cpu-chart'),cm=document.getElementById('cpu-mk');let mk='15m';
  function draw(){if(!B[mk].policy_A){host.innerHTML='<p class="small">Not run yet.</p>';return;}const L=B[mk].policy_A.models;const pts=Object.entries(L).filter(([m,r])=>m!=='Market price'&&r.cpu&&r.cpu.fit_s_mean!=null);if(!pts.length){host.innerHTML='';return;}
    const W=720,H=280,Lp=60,R=20,T=14,Bt=38;host.innerHTML='';const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,class:'chart',role:'img','aria-label':'cpu vs error'},host);
    const tot=r=>(r.cpu.fit_s_mean||0)+(r.cpu.search_s_mean||0);const lx=v=>Math.log10(Math.max(v,1e-2));const xs=pts.map(([m,r])=>lx(tot(r)));const ys=pts.map(([m,r])=>r.logloss).concat([L['Market price'].logloss]);
    const x0=Math.floor(Math.min(...xs)),x1=Math.max(Math.ceil(Math.max(...xs)),x0+1);let y0=Math.min(...ys),y1=Math.max(...ys);const pd=(y1-y0)*.12||.001;y0-=pd;y1+=pd;
    const X=v=>Lp+(lx(v)-x0)/(x1-x0)*(W-Lp-R-90),Y=v=>T+(1-(v-y0)/(y1-y0))*(H-T-Bt);
    for(let e=x0;e<=x1;e++){const xx=Lp+(e-x0)/(x1-x0)*(W-Lp-R-90);el('line',{x1:xx,x2:xx,y1:T,y2:H-Bt,class:'grid'},svg);txt(svg,xx,H-Bt+15,(Math.pow(10,e))+' s','tick','middle');}
    niceTicks(y0,y1,5).forEach(v=>{el('line',{x1:Lp,x2:W-R,y1:Y(v),y2:Y(v),class:'grid'},svg);txt(svg,Lp-6,Y(v)+3.5,v.toFixed(3),'tick','end');});
    el('line',{x1:Lp,x2:W-R,y1:Y(L['Market price'].logloss),y2:Y(L['Market price'].logloss),stroke:'var(--fg)','stroke-dasharray':'5 4'},svg);txt(svg,W-R-4,Y(L['Market price'].logloss)-5,'market price','val','end');
    txt(svg,(Lp+W-R)/2,H-3,'CPU seconds per weekly refit, including tuning (log scale)','tick','middle');
    const fig=host2fig(host);pts.forEach(([m,r])=>{const x=tot(r);el('circle',{cx:X(x),cy:Y(r.logloss),r:m.startsWith('TabPFN')?7:5.5,fill:COLOR(m),stroke:'#fff','stroke-width':2},svg);txt(svg,X(x)+9,Y(r.logloss)+4,BNAME(m),'lab');
      const h=el('circle',{cx:X(x),cy:Y(r.logloss),r:14,class:'hit'},svg);h.addEventListener('pointermove',e=>showTip(fig,`<b>${BNAME(m)}</b><br>${fmt(x,1)} s per refit<br>log loss ${fmt(r.logloss,4)}`,e));h.addEventListener('pointerleave',()=>hideTip(fig));});}
  seg(cm,[['15m','15-minute'],['1h','hourly']],mk,v=>{mk=v;draw();});draw();})();
(function(){const host=document.getElementById('rel-chart'),cm=document.getElementById('rel-mk');let mk='15m';
  function draw(){if(!B[mk].policy_A){host.innerHTML='<p class="small">Not run yet.</p>';document.getElementById('rel-legend').innerHTML='';return;}const L=B[mk].policy_A.models;const other=Object.keys(L).find(m=>m.startsWith('LightGBM'));const pick=['Market price','TabPFN-3.5',other].filter(m=>m&&L[m]);
    lineChart(host,[{name:'perfect',color:'var(--muted)',pts:[[0,0],[1,1]],dash:'4 4',width:1,nohover:true}].concat(pick.map(m=>({name:BNAME(m),color:m===other?'var(--c2)':COLOR(m),width:m==='TabPFN-3.5'?2.8:1.6,dash:m==='Market price'?'2 3':'',marker:true,pts:L[m].reliability.map(b=>[+b.p_mean.toFixed(3),b.y_mean])}))),{x:'lin',xticks:[0,.2,.4,.6,.8,1],xfmt:a=>a.toFixed(1),y0:0,y1:1,yfmt:a=>a.toFixed(1),xlabel:'predicted probability',ylabel:'actual YES rate',tipfmt:a=>fmt(a,3),tipx:a=>'p ≈ '+fmt(a,2),h:300,label:'reliability'});
    document.getElementById('rel-legend').innerHTML=pick.map(m=>`<span><i style="border-top-color:${m===other?'var(--c2)':COLOR(m)};border-top-style:${m==='Market price'?'dashed':'solid'}"></i>${BNAME(m)}: calibration error ${fmt(L[m].ece,4)}</span>`).join('');}
  seg(cm,[['15m','15-minute'],['1h','hourly']],mk,v=>{mk=v;draw();});draw();})();
"""
