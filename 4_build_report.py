"""
4_build_report.py
=================
Builds ONE self-contained interactive HTML report from the pipeline outputs —
tabs per step, sortable tables, interactive equity curves (hover for values,
click legend entries to show/hide series). No dependencies, works offline.

INPUT : OUTPUTS/results_outputs/*.csv   (steps 1, 1b, 2, 3 must have been run)
OUTPUT: OUTPUTS/report.html
"""

import json
import os

import numpy as np
import pandas as pd

RES = "OUTPUTS/results_outputs"
OUT_HTML = "OUTPUTS/report.html"
MAX_PTS = 2500          # decimate long series for a snappy page

PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948",
           "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#86bcb6", "#d37295"]


def read_csv(name, **kw):
    p = os.path.join(RES, name)
    return pd.read_csv(p, **kw) if os.path.exists(p) else None


def decimate(xs, ys):
    n = len(xs)
    if n <= MAX_PTS:
        return xs, ys
    stride = int(np.ceil(n / MAX_PTS))
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [xs[i] for i in idx], [ys[i] for i in idx]


def series(name, times, values, color, bold=False, fill=False):
    xs = [int(t.timestamp() * 1000) for t in times]
    ys = [round(float(v)) for v in values]
    xs, ys = decimate(xs, ys)
    return {"name": name, "x": xs, "y": ys, "color": color,
            "bold": bold, "fill": fill}


def table_payload(df, drop=()):
    if df is None:
        return None
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df = df.where(pd.notna(df), None)
    return {"cols": list(df.columns), "rows": df.values.tolist()}


# ── LOAD ─────────────────────────────────────────────────────────────────────
trades = {}
for s in ["RR", "GG"]:
    t = read_csv(f"{s}_maemfe_combined_trades.csv",
                 parse_dates=["entry_time", "exit_time"])
    if t is not None:
        trades[s] = t.sort_values("exit_time").reset_index(drop=True)
if not trades:
    raise SystemExit(f"No *_maemfe_combined_trades.csv in {RES} — run step 2 first.")

recs = {s: read_csv(f"{s}_recommendations.csv") for s in ["RR", "GG"]}
summaries = {s: read_csv(f"{s}_maemfe_window_summary.csv") for s in ["RR", "GG"]}
alloc = read_csv("multi_strategy_allocation.csv")
pertrade = read_csv("rr_pertrade_recommendations.csv")

# ── STEP-2 SERIES ────────────────────────────────────────────────────────────
step2_charts = {}
for s, t in trades.items():
    ss = [series(f"{s} combined", t["exit_time"], t["net"].cumsum(), "#222", bold=True)]
    wins = sorted(t["window"].unique(), key=lambda w: int(w.split("-")[0]))
    for i, w in enumerate(wins):
        sub = t[t["window"] == w]
        rr = sub["RR"].iloc[0]
        ss.append(series(f"{w}@{rr:g}", sub["exit_time"], sub["net"].cumsum(),
                         PALETTE[i % len(PALETTE)]))
    step2_charts[s] = ss

both = (pd.concat(trades.values(), ignore_index=True)
        .sort_values("exit_time").reset_index(drop=True))
eq_all = both["net"].cumsum()
cross = [series("TOTAL (RR+GG)", both["exit_time"], eq_all, "#222", bold=True)]
for i, (s, t) in enumerate(trades.items()):
    cross.append(series(f"{s} only", t["exit_time"], t["net"].cumsum(), PALETTE[i]))
under = [series("drawdown", both["exit_time"], eq_all - eq_all.cummax(),
                "#e15759", fill=True)]

# ── STEP-3 PER-ACCOUNT REPLAY ────────────────────────────────────────────────
acct_series, alloc_note = [], ""
if alloc is not None:
    for i, row in alloc.iterrows():
        strat = row["strategy"]
        if strat not in trades:
            continue
        wins = [w.strip().split("@")[0] for w in str(row["windows"]).split(",")]
        sub = (trades[strat][trades[strat]["window"].isin(wins)]
               .sort_values(["entry_time", "exit_time"]))
        keep, open_until = [], pd.Timestamp.min
        for idx, r in sub.iterrows():          # same one-position replay as step 3
            if r["entry_time"] >= open_until:
                keep.append(idx)
                open_until = r["exit_time"]
        sub = sub.loc[keep].sort_values("exit_time")
        label = f'{row["account"]} [{strat}: {row["windows"]}]'
        acct_series.append(series(label, sub["exit_time"], sub["net"].cumsum(),
                                  PALETTE[i % len(PALETTE)]))
    alloc_note = ("Curves use the same one-position replay as the allocator "
                  "(entries arriving while a trade is open are skipped).")

# ── OVERVIEW CARDS ───────────────────────────────────────────────────────────
def maxdd(x):
    e = np.cumsum(np.asarray(x, float))
    return float((np.maximum.accumulate(e) - e).max())

cards = [
    {"t": "Portfolio net (all windows)", "v": f"${both['net'].sum():,.0f}",
     "s": f"{len(both):,} trades, {both['exit_time'].min().date()} → {both['exit_time'].max().date()}"},
    {"t": "Portfolio max drawdown", "v": f"${maxdd(both['net']):,.0f}",
     "s": "closed-equity, all windows together"},
]
for s, t in trades.items():
    cards.append({"t": f"{s} strategy", "v": f"${t['net'].sum():,.0f}",
                  "s": f"{t['window'].nunique()} windows, maxDD ${maxdd(t['net']):,.0f}"})
if alloc is not None:
    cards.append({"t": "Allocation net (7 accounts)", "v": f"${alloc['net_profit'].sum():,.0f}",
                  "s": f"worst account at {alloc['used_%_of_available'].max():.1f}% of its DD"})
cards.append({"t": "Costs", "v": "$1 / round-turn",
              "s": "all figures net of commission"})

# ── PAYLOAD ──────────────────────────────────────────────────────────────────
data = {
    "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    "cards": cards,
    "tables": {
        "rr_recs": table_payload(recs["RR"], drop=("rr_lo", "rr_hi")),
        "gg_recs": table_payload(recs["GG"], drop=("rr_lo", "rr_hi")),
        "pertrade": table_payload(pertrade),
        "rr_sum": table_payload(summaries["RR"]),
        "gg_sum": table_payload(summaries["GG"]),
        "alloc": table_payload(alloc),
    },
    "charts": {
        "rr_eq": step2_charts.get("RR", []),
        "gg_eq": step2_charts.get("GG", []),
        "cross": cross,
        "under": under,
        "accounts": acct_series,
    },
    "alloc_note": alloc_note,
}

TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>RR/GG Window Study — Report</title>
<style>
 body{font:14px/1.45 system-ui,Segoe UI,Arial;margin:0;background:#f4f5f7;color:#1a1a1a}
 header{background:#1f2937;color:#fff;padding:14px 22px}
 header h1{font-size:17px;margin:0} header small{color:#9ca3af}
 nav{display:flex;gap:4px;background:#1f2937;padding:0 16px}
 nav button{border:0;padding:10px 16px;background:transparent;color:#cbd5e1;cursor:pointer;
   font-size:13.5px;border-bottom:3px solid transparent}
 nav button.on{color:#fff;border-color:#60a5fa}
 main{max-width:1200px;margin:18px auto;padding:0 16px}
 section{display:none} section.on{display:block}
 .cards{display:flex;flex-wrap:wrap;gap:12px;margin:6px 0 18px}
 .card{background:#fff;border-radius:10px;padding:14px 18px;box-shadow:0 1px 3px rgba(0,0,0,.08);min-width:200px}
 .card .t{font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:.4px}
 .card .v{font-size:22px;font-weight:600;margin:2px 0}
 .card .s{font-size:12px;color:#6b7280}
 h2{font-size:15px;margin:22px 0 8px}
 .panel{background:#fff;border-radius:10px;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:16px}
 table{border-collapse:collapse;width:100%;font-size:12.6px}
 th{cursor:pointer;user-select:none;text-align:right;padding:6px 8px;background:#f1f5f9;
    position:sticky;top:0;white-space:nowrap}
 th:first-child,td:first-child{text-align:left}
 td{padding:5px 8px;text-align:right;border-top:1px solid #eef0f3;white-space:nowrap}
 tr:hover td{background:#f8fafc}
 .twrap{max-height:420px;overflow:auto;border:1px solid #e5e7eb;border-radius:6px}
 .legend{display:flex;flex-wrap:wrap;gap:4px 14px;margin:4px 0 6px;font-size:12.5px}
 .legend span{cursor:pointer;user-select:none;display:inline-flex;align-items:center;gap:5px}
 .legend .off{opacity:.35;text-decoration:line-through}
 .sw{width:14px;height:3px;display:inline-block;border-radius:2px}
 .chartbox{position:relative}
 .tip{position:absolute;pointer-events:none;background:rgba(17,24,39,.94);color:#f9fafb;
   font-size:11.5px;padding:6px 9px;border-radius:6px;display:none;z-index:5;white-space:nowrap}
 .note{font-size:12.5px;color:#6b7280;margin:4px 0 10px}
 .verd-OK{color:#15803d;font-weight:600}.verd-WEAK{color:#b45309;font-weight:600}
 .verd-UNLOCK_ONLY{color:#6d28d9;font-weight:600}.verd-LOSING{color:#b91c1c;font-weight:600}
</style></head><body>
<header><h1>RR / GG Time-Window Study <small>— generated __GEN__</small></h1></header>
<nav id="nav"></nav>
<main>
 <section id="tab0"><div class="cards" id="cards"></div>
  <div class="panel"><h2 style="margin-top:2px">How to read this report</h2>
  <b>Step 1 — RR picks:</b> DD-aware RR per window from MT5 optimizations (+ per-trade re-checks).
  <b>Step 2 — Portfolio:</b> real per-trade equity curves for the chosen windows.
  <b>Step 3 — Allocation:</b> which windows each prop account trades.
  Click legend entries to show/hide curves; hover charts for values; click table headers to sort.</div>
 </section>
 <section id="tab1">
  <h2>RR strategy — window recommendations (step 1)</h2><div class="panel twrap" id="t_rr_recs"></div>
  <h2>GG strategy — window recommendations (step 1)</h2><div class="panel twrap" id="t_gg_recs"></div>
  <h2>Per-trade RR sweeps (step 1b — ground truth)</h2><div class="panel twrap" id="t_pertrade"></div>
 </section>
 <section id="tab2">
  <h2>Cross-strategy equity</h2><div class="panel" id="c_cross"></div>
  <h2>Portfolio drawdown (underwater)</h2><div class="panel" id="c_under"></div>
  <h2>RR — combined & per-window equity</h2><div class="panel" id="c_rr_eq"></div>
  <div class="panel twrap" id="t_rr_sum"></div>
  <h2>GG — combined & per-window equity</h2><div class="panel" id="c_gg_eq"></div>
  <div class="panel twrap" id="t_gg_sum"></div>
 </section>
 <section id="tab3">
  <h2>Account allocation (step 3)</h2><div class="panel twrap" id="t_alloc"></div>
  <h2>Per-account equity</h2><div class="note" id="alloc_note"></div>
  <div class="panel" id="c_accounts"></div>
 </section>
</main>
<script>
const DATA=__DATA__;
const TABS=["Overview","Step 1 — RR picks","Step 2 — Portfolio","Step 3 — Allocation"];
const nav=document.getElementById('nav');
TABS.forEach((t,i)=>{const b=document.createElement('button');b.textContent=t;
 b.onclick=()=>{document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('section').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');document.getElementById('tab'+i).classList.add('on');};
 nav.appendChild(b);});
nav.children[0].click();

const fmt$=v=>(v<0?'-$':'$')+Math.abs(Math.round(v)).toLocaleString('en-US');
const fmtD=ms=>{const d=new Date(ms);return d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');};

// cards
document.getElementById('cards').innerHTML=DATA.cards.map(c=>
 `<div class="card"><div class="t">${c.t}</div><div class="v">${c.v}</div><div class="s">${c.s}</div></div>`).join('');
document.getElementById('alloc_note').textContent=DATA.alloc_note||'';

// sortable tables
function renderTable(id,payload){
 const el=document.getElementById(id);
 if(!payload){el.innerHTML='<div class="note">not generated</div>';return;}
 let rows=payload.rows.slice(),dir=1,sortk=-1;
 const draw=()=>{
  let h='<table><thead><tr>'+payload.cols.map((c,i)=>`<th data-i="${i}">${c}</th>`).join('')+'</tr></thead><tbody>';
  h+=rows.map(r=>'<tr>'+r.map((v,i)=>{
    let cls='';
    if(String(payload.cols[i]).toLowerCase()==='verdict'&&v)cls=' class="verd-'+v+'"';
    if(v===null||v===undefined||v!==v)v='';
    if(typeof v==='number'&&!Number.isInteger(v))v=v.toLocaleString('en-US',{maximumFractionDigits:2});
    else if(typeof v==='number')v=v.toLocaleString('en-US');
    return `<td${cls}>${v}</td>`;}).join('')+'</tr>').join('');
  el.innerHTML=h+'</tbody></table>';
  el.querySelectorAll('th').forEach(th=>th.onclick=()=>{const i=+th.dataset.i;
   dir=(sortk===i)?-dir:1;sortk=i;
   rows.sort((a,b)=>{const x=a[i],y=b[i];
    if(x==null)return 1;if(y==null)return -1;
    return (typeof x==='number'&&typeof y==='number')?dir*(x-y):dir*String(x).localeCompare(String(y));});
   draw();});
 };draw();
}
renderTable('t_rr_recs',DATA.tables.rr_recs);
renderTable('t_gg_recs',DATA.tables.gg_recs);
renderTable('t_pertrade',DATA.tables.pertrade);
renderTable('t_rr_sum',DATA.tables.rr_sum);
renderTable('t_gg_sum',DATA.tables.gg_sum);
renderTable('t_alloc',DATA.tables.alloc);

// interactive line chart
function niceTicks(a,b,n){const r=b-a||1,s0=Math.pow(10,Math.floor(Math.log10(r/n)));
 let best=s0;[1,2,5,10].forEach(m=>{if(Math.abs(r/(s0*m)-n)<Math.abs(r/best-n))best=s0*m;});
 const t=[];for(let v=Math.ceil(a/best)*best;v<=b;v+=best)t.push(v);return t;}
function chart(id,seriesList,yFmt){
 const host=document.getElementById(id);
 if(!seriesList||!seriesList.length){host.innerHTML='<div class="note">no data</div>';return;}
 const vis=seriesList.map(()=>true);
 const box=document.createElement('div');box.className='chartbox';
 const leg=document.createElement('div');leg.className='legend';
 host.appendChild(leg);host.appendChild(box);
 const tip=document.createElement('div');tip.className='tip';box.appendChild(tip);
 const W=1120,H=360,m={l:74,r:16,t:12,b:30};
 function draw(){
  const act=seriesList.filter((s,i)=>vis[i]);
  let x0=1/0,x1=-1/0,y0=1/0,y1=-1/0;
  act.forEach(s=>{x0=Math.min(x0,s.x[0]);x1=Math.max(x1,s.x[s.x.length-1]);
   s.y.forEach(v=>{y0=Math.min(y0,v);y1=Math.max(y1,v);});});
  if(!act.length){x0=0;x1=1;y0=0;y1=1;}
  if(y0===y1){y0-=1;y1+=1;}
  const pad=(y1-y0)*.05;y0-=pad;y1+=pad;
  const sx=t=>m.l+(t-x0)/(x1-x0)*(W-m.l-m.r), sy=v=>H-m.b-(v-y0)/(y1-y0)*(H-m.t-m.b);
  let g='';
  niceTicks(y0,y1,5).forEach(v=>{const y=sy(v);
   g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y}" y2="${y}" stroke="#e5e7eb"/>`+
      `<text x="${m.l-8}" y="${y+4}" text-anchor="end" font-size="11" fill="#6b7280">${yFmt(v)}</text>`;});
  niceTicks(x0,x1,6).forEach(v=>{const x=sx(v);
   g+=`<text x="${x}" y="${H-8}" text-anchor="middle" font-size="11" fill="#6b7280">${fmtD(v).slice(0,7)}</text>`;});
  act.forEach(s=>{
   let d='M'+s.x.map((t,i)=>sx(t).toFixed(1)+','+sy(s.y[i]).toFixed(1)).join('L');
   if(s.fill){d+=`L${sx(s.x[s.x.length-1]).toFixed(1)},${sy(0).toFixed(1)}L${sx(s.x[0]).toFixed(1)},${sy(0).toFixed(1)}Z`;
    g+=`<path d="${d}" fill="${s.color}" fill-opacity=".25" stroke="${s.color}" stroke-width="1"/>`;}
   else g+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.bold?2.4:1.4}"/>`;});
  g+=`<line id="${id}_cur" x1="0" x2="0" y1="${m.t}" y2="${H-m.b}" stroke="#9ca3af" stroke-dasharray="3,3" visibility="hidden"/>`;
  box.insertAdjacentHTML('afterbegin',
   `<svg viewBox="0 0 ${W} ${H}" style="width:100%;display:block" id="${id}_svg">${g}</svg>`);
  const old=box.querySelectorAll('svg');if(old.length>1)old[old.length-1].remove();
  const svg=document.getElementById(id+'_svg'),cur=document.getElementById(id+'_cur');
  svg.onmousemove=e=>{const r=svg.getBoundingClientRect();
   const xp=(e.clientX-r.left)*W/r.width;
   if(xp<m.l||xp>W-m.r){tip.style.display='none';cur.setAttribute('visibility','hidden');return;}
   const t=x0+(xp-m.l)/(W-m.l-m.r)*(x1-x0);
   cur.setAttribute('x1',xp);cur.setAttribute('x2',xp);cur.setAttribute('visibility','visible');
   let html='<b>'+fmtD(t)+'</b>';
   seriesList.forEach((s,i)=>{if(!vis[i])return;
    let lo=0,hi=s.x.length-1;
    while(hi-lo>1){const md=(lo+hi)>>1;(s.x[md]<t)?lo=md:hi=md;}
    const j=(t-s.x[lo]<s.x[hi]-t)?lo:hi;
    html+=`<br><span style="color:${s.color}">●</span> ${s.name}: ${yFmt(s.y[j])}`;});
   tip.innerHTML=html;tip.style.display='block';
   const bx=box.getBoundingClientRect();
   let tx=e.clientX-bx.left+14;if(tx>bx.width-230)tx=e.clientX-bx.left-235;
   tip.style.left=tx+'px';tip.style.top=(e.clientY-bx.top+10)+'px';};
  svg.onmouseleave=()=>{tip.style.display='none';cur.setAttribute('visibility','hidden');};
 }
 seriesList.forEach((s,i)=>{const sp=document.createElement('span');
  sp.innerHTML=`<i class="sw" style="background:${s.color}"></i>${s.name}`;
  sp.onclick=()=>{vis[i]=!vis[i];sp.classList.toggle('off',!vis[i]);draw();};
  leg.appendChild(sp);});
 draw();
}
chart('c_cross',DATA.charts.cross,fmt$);
chart('c_under',DATA.charts.under,fmt$);
chart('c_rr_eq',DATA.charts.rr_eq,fmt$);
chart('c_gg_eq',DATA.charts.gg_eq,fmt$);
chart('c_accounts',DATA.charts.accounts,fmt$);
</script></body></html>"""

html = (TEMPLATE
        .replace("__GEN__", data["generated"])
        .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Report written: {OUT_HTML}  ({os.path.getsize(OUT_HTML)/1e6:.1f} MB)")
print("Open it in any browser — fully self-contained, no internet needed.")
