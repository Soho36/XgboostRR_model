"""
4_build_report.py
=================
Builds ONE self-contained interactive HTML report from the pipeline outputs.

Design: the per-trade data is shipped to the browser and every chart is
recomputed client-side from the CURRENT window selection. So ticking a window
on/off updates the equity curve, the drawdown subplot, and all the analytics
charts (monthly / yearly / hour / weekday / seasonality / histogram) together —
the drawdown you see is always the drawdown of exactly the windows you selected.

Charts use Plotly (inlined, offline): drag to zoom, double-click to reset,
modebar for pan/box-zoom/save-png. Equity and drawdown share an x-axis, so
zooming one zooms the other.

INPUT : data/3_results/*.csv   (steps 1, 2, 3)
OUTPUT: reports/report.html
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.offline as po

import provenance as prov

RES = "data/3_results"
OUT_HTML = "reports/report.html"

PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948",
           "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#86bcb6", "#d37295",
           "#8cd17d", "#b6992d", "#499894", "#d4a6c8", "#79706e", "#fabfd2",
           "#a0cbe8"]


def read_csv(name, **kw):
    p = os.path.join(RES, name)
    return pd.read_csv(p, **kw) if os.path.exists(p) else None


def table_payload(df, drop=()):
    if df is None:
        return None
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df = df.where(pd.notna(df), None)
    return {"cols": list(df.columns), "rows": df.values.tolist()}


# ── LOAD PER-TRADE DATA ──────────────────────────────────────────────────────
trades = {}
for s in ["RR", "GG"]:
    t = read_csv(f"{s}_maemfe_combined_trades.csv",
                 parse_dates=["entry_time", "exit_time"])
    if t is not None:
        trades[s] = t
if not trades:
    raise SystemExit(f"No *_maemfe_combined_trades.csv in {RES} — run step 2 first.")

ALL = pd.concat(trades.values(), ignore_index=True)
ALL = ALL.sort_values("exit_time").reset_index(drop=True)

# window catalogue (stable index used by the client)
wmeta, wkey = [], {}
for s in ["RR", "GG"]:
    if s not in trades:
        continue
    for w in sorted(trades[s]["window"].unique(), key=lambda x: int(x.split("-")[0])):
        rr = trades[s].loc[trades[s]["window"] == w, "RR"].iloc[0]
        wkey[(s, w)] = len(wmeta)
        wmeta.append({"s": s, "w": w, "rr": float(rr),
                      "color": PALETTE[len(wmeta) % len(PALETTE)]})

widx = [wkey[(r.strategy, r.window)] for r in ALL.itertuples()]

# compact arrays: minutes since epoch keeps the JSON ~40% smaller than ms
tr = {
    "tx": (ALL["exit_time"].astype("int64") // 60_000_000_000).tolist(),
    "te": (ALL["entry_time"].astype("int64") // 60_000_000_000).tolist(),
    "w": widx,
    "n": [round(float(v), 1) for v in ALL["net"]],
}

# ── STEP-3 ACCOUNT SERIES (one-position replay, same as the allocator) ───────
alloc = read_csv("multi_strategy_allocation.csv")
accounts, alloc_note = [], ""
if alloc is not None:
    for i, row in alloc.iterrows():
        strat = row["strategy"]
        if strat not in trades:
            continue
        wins = [w.strip().split("@")[0] for w in str(row["windows"]).split(",")]
        sub = (trades[strat][trades[strat]["window"].isin(wins)]
               .sort_values(["entry_time", "exit_time"]))
        keep, open_until = [], pd.Timestamp.min
        for idx, r in sub.iterrows():
            if r["entry_time"] >= open_until:
                keep.append(idx)
                open_until = r["exit_time"]
        sub = sub.loc[keep].sort_values("exit_time")
        accounts.append({
            "name": str(row["account"]),
            "label": f'{row["account"]} · {strat} · {row["windows"]}',
            "color": PALETTE[i % len(PALETTE)],
            "tx": (sub["exit_time"].astype("int64") // 60_000_000_000).tolist(),
            "n": [round(float(v), 1) for v in sub["net"]],
            "limit": float(row.get("DD_available", row.get("DD_limit", 0)) or 0),
        })
    alloc_note = ("Curves use the same one-position replay as the allocator "
                  "(entries arriving while a trade is open are skipped).")


# ── OVERVIEW CARDS ───────────────────────────────────────────────────────────
def maxdd(x):
    e = np.cumsum(np.asarray(x, float))
    return float((np.maximum.accumulate(e) - e).max())


cards = [
    {"t": "Portfolio net (all windows)", "v": f"${ALL['net'].sum():,.0f}",
     "s": f"{len(ALL):,} trades · {ALL['exit_time'].min().date()} → {ALL['exit_time'].max().date()}"},
    {"t": "Portfolio max drawdown", "v": f"${maxdd(ALL['net']):,.0f}",
     "s": "closed-equity, all windows together"},
]
for s, t in trades.items():
    cards.append({"t": f"{s} strategy", "v": f"${t['net'].sum():,.0f}",
                  "s": f"{t['window'].nunique()} windows · maxDD ${maxdd(t['net']):,.0f}"})
if alloc is not None:
    cards.append({"t": "Allocation net (7 accounts)",
                  "v": f"${alloc['net_profit'].sum():,.0f}",
                  "s": f"worst account at {alloc['used_%_of_available'].max():.1f}% of its DD"})
cards.append({"t": "Costs", "v": "$1 / round-turn", "s": "all figures net of commission"})

PROV4 = prov.base("4_build_report",
                  upstream=prov.load("data/3_results/_provenance_step3.json"))
prov.write("data/3_results/_provenance_step4.json", PROV4)

# flatten the chain so the page can show the whole lineage in one block
_chain, _n = [], PROV4
while _n:
    _chain.append({k: _n.get(k) for k in
                   ("step", "run_id", "generated", "git", "data_cutoff",
                    "data_cutoffs_seen", "validated_passes",
                    "validation_mismatches", "passes_missing_stats",
                    "passes_account_blown", "overrides", "ea",
                    "ea_unknown_windows", "unsafe_windows",
                    "calibration_entries", "calibration_max", "report_fraction")})
    _n = _n.get("upstream")
_warn = []
for _link in _chain:
    _warn += prov.warnings_for(_link)

data = {
    "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    "prov_chain": _chain,
    "prov_warnings": sorted(set(_warn)),
    "cards": cards,
    "windows": wmeta,
    "trades": tr,
    "accounts": accounts,
    "alloc_note": alloc_note,
    "tables": {
        "rr_recs": table_payload(read_csv("RR_recommendations.csv"), drop=("rr_lo", "rr_hi")),
        "gg_recs": table_payload(read_csv("GG_recommendations.csv"), drop=("rr_lo", "rr_hi")),
        "pertrade": table_payload(read_csv("rr_pertrade_recommendations.csv")),
        "rr_sum": table_payload(read_csv("RR_maemfe_window_summary.csv")),
        "gg_sum": table_payload(read_csv("GG_maemfe_window_summary.csv")),
        "alloc": table_payload(alloc),
    },
}

TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>RR/GG Window Study — Report</title>
<script>__PLOTLYJS__</script>
<style>
 body{font:14px/1.45 system-ui,Segoe UI,Arial;margin:0;background:#f4f5f7;color:#1a1a1a}
 header{background:#1f2937;color:#fff;padding:14px 22px}
 header h1{font-size:17px;margin:0} header small{color:#9ca3af}
 nav{display:flex;gap:4px;background:#1f2937;padding:0 16px}
 nav button{border:0;padding:10px 16px;background:transparent;color:#cbd5e1;cursor:pointer;
   font-size:13.5px;border-bottom:3px solid transparent}
 nav button.on{color:#fff;border-color:#60a5fa}
 main{max-width:1280px;margin:18px auto;padding:0 16px}
 section{display:none} section.on{display:block}
 .cards{display:flex;flex-wrap:wrap;gap:12px;margin:6px 0 18px}
 .card{background:#fff;border-radius:10px;padding:14px 18px;box-shadow:0 1px 3px rgba(0,0,0,.08);min-width:200px}
 .card .t{font-size:12px;color:#6b7280;text-transform:uppercase;letter-spacing:.4px}
 .card .v{font-size:22px;font-weight:600;margin:2px 0}
 .card .s{font-size:12px;color:#6b7280}
 h2{font-size:15px;margin:22px 0 8px}
 .panel{background:#fff;border-radius:10px;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,.08);margin-bottom:16px}
 table{border-collapse:collapse;width:100%;font-size:12.6px}
 th{cursor:pointer;user-select:none;text-align:right;padding:6px 8px;background:#f1f5f9;position:sticky;top:0;white-space:nowrap}
 th:first-child,td:first-child{text-align:left}
 td{padding:5px 8px;text-align:right;border-top:1px solid #eef0f3;white-space:nowrap}
 tr:hover td{background:#f8fafc}
 .twrap{max-height:420px;overflow:auto;border:1px solid #e5e7eb;border-radius:6px}
 .note{font-size:12.5px;color:#6b7280;margin:4px 0 10px}
 #provbox{font-size:12.2px;color:#4b5563}
 #provbox h2{margin:0 0 8px}
 #provbox code{background:#f1f5f9;padding:1px 5px;border-radius:4px}
 #provbox .pw{color:#b45309;font-weight:600}
 #provbox table{font-size:12px;margin-top:6px;width:auto}
 .picker{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center;margin:2px 0 10px}
 .chip{display:inline-flex;align-items:center;gap:6px;cursor:pointer;user-select:none;
   border:1px solid #d7dbe0;border-radius:14px;padding:3px 10px;font-size:12.5px;background:#fff}
 .chip.off{opacity:.4;background:#f1f5f9}
 .chip .sw{width:10px;height:10px;border-radius:50%}
 .btn{border:1px solid #d7dbe0;background:#fff;border-radius:6px;padding:4px 10px;cursor:pointer;font-size:12.5px}
 .btn:hover{background:#f1f5f9}
 .stat{display:flex;flex-wrap:wrap;gap:18px;font-size:13px;margin:2px 0 8px;color:#374151}
 .stat b{font-size:15px}
 .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 @media(max-width:900px){.grid2{grid-template-columns:1fr}}
 .verd-OK{color:#15803d;font-weight:600}.verd-WEAK{color:#b45309;font-weight:600}
 .verd-UNLOCK_ONLY{color:#6d28d9;font-weight:600}.verd-LOSING{color:#b91c1c;font-weight:600}
</style></head><body>
<header><h1>RR / GG Time-Window Study <small>— generated __GEN__</small></h1></header>
<nav id="nav"></nav>
<main>
 <section id="tab0"><div class="cards" id="cards"></div>
  <div class="panel"><h2 style="margin-top:2px">How to read this report</h2>
   <b>Step 1 — RR picks:</b> DD-aware RR per window from MT5 optimizations, plus the
   per-trade (step 1b) re-checks.<br>
   <b>Step 2 — Portfolio:</b> real per-trade equity for the chosen windows. Everything on
   that tab recomputes from the windows you tick, including the drawdown subplot.<br>
   <b>Step 3 — Allocation:</b> which windows each prop account trades.<br>
   <span class="note">Charts: drag to zoom · double-click to reset · modebar (top-right)
   for pan / box-zoom / save-png. Equity and drawdown share an x-axis, so zooming one
   zooms the other.</span></div>
  <div class="panel" id="provbox"></div>
 </section>

 <section id="tab1">
  <h2>RR strategy — window recommendations (step 1)</h2><div class="panel twrap" id="t_rr_recs"></div>
  <h2>GG strategy — window recommendations (step 1)</h2><div class="panel twrap" id="t_gg_recs"></div>
  <h2>Per-trade RR sweeps (step 1b — ground truth)</h2><div class="panel twrap" id="t_pertrade"></div>
 </section>

 <section id="tab2">
  <div class="panel">
   <div class="picker" id="picker"></div>
   <div class="stat" id="selstat"></div>
  </div>
  <div class="panel"><div id="c_equity" style="height:520px"></div></div>
  <div class="grid2">
   <div class="panel"><div id="c_monthly" style="height:300px"></div></div>
   <div class="panel"><div id="c_yearly" style="height:300px"></div></div>
  </div>
  <div class="panel"><div id="c_heat" style="height:320px"></div></div>
  <div class="grid2">
   <div class="panel"><div id="c_hour" style="height:300px"></div></div>
   <div class="panel"><div id="c_dow" style="height:300px"></div></div>
  </div>
  <div class="grid2">
   <div class="panel"><div id="c_seas" style="height:300px"></div></div>
   <div class="panel"><div id="c_hist" style="height:300px"></div></div>
  </div>
  <h2>RR — per-window summary</h2><div class="panel twrap" id="t_rr_sum"></div>
  <h2>GG — per-window summary</h2><div class="panel twrap" id="t_gg_sum"></div>
 </section>

 <section id="tab3">
  <h2>Account allocation (step 3)</h2><div class="panel twrap" id="t_alloc"></div>
  <h2>Per-account equity</h2><div class="note" id="alloc_note"></div>
  <div class="panel"><div id="c_accounts" style="height:460px"></div></div>
 </section>
</main>
<script>
const DATA=__DATA__;
const TR=DATA.trades, WM=DATA.windows, NT=TR.n.length;
const MIN=60000;                       // stored timestamps are minutes since epoch
const CFG={displaylogo:false,responsive:true,
  modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']};
const FONT={family:'system-ui,Segoe UI,Arial',size:11.5};
const fmt$=v=>(v<0?'-$':'$')+Math.abs(Math.round(v)).toLocaleString('en-US');

// ---- tabs -------------------------------------------------------------------
const TABS=["Overview","Step 1 — RR picks","Step 2 — Portfolio","Step 3 — Allocation"];
const nav=document.getElementById('nav');
TABS.forEach((t,i)=>{const b=document.createElement('button');b.textContent=t;
 b.onclick=()=>{document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('section').forEach(x=>x.classList.remove('on'));
  b.classList.add('on');document.getElementById('tab'+i).classList.add('on');
  window.dispatchEvent(new Event('resize'));      // let plotly size itself
  if(i===2&&!window._t2){window._t2=1;redraw();}
  if(i===3&&!window._t3){window._t3=1;drawAccounts();}};
 nav.appendChild(b);});

document.getElementById('cards').innerHTML=DATA.cards.map(c=>
 `<div class="card"><div class="t">${c.t}</div><div class="v">${c.v}</div><div class="s">${c.s}</div></div>`).join('');
document.getElementById('alloc_note').textContent=DATA.alloc_note||'';

// ---- sortable tables --------------------------------------------------------
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
    else if(typeof v==='number')v=Number.isInteger(v)?v.toLocaleString('en-US')
        :v.toLocaleString('en-US',{maximumFractionDigits:2});
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

// ---- window picker ----------------------------------------------------------
const sel=new Set(WM.map((_,i)=>i));
const picker=document.getElementById('picker');
function chip(html,fn){const b=document.createElement('button');b.className='btn';
 b.innerHTML=html;b.onclick=fn;picker.appendChild(b);}
chip('All',()=>{WM.forEach((_,i)=>sel.add(i));syncChips();redraw();});
chip('None',()=>{sel.clear();syncChips();redraw();});
['RR','GG'].forEach(s=>{ if(!WM.some(m=>m.s===s))return;
 chip(s+' only',()=>{sel.clear();WM.forEach((m,i)=>{if(m.s===s)sel.add(i);});syncChips();redraw();});});
WM.forEach((m,i)=>{
 const el=document.createElement('span');el.className='chip';el.dataset.i=i;
 el.innerHTML=`<i class="sw" style="background:${m.color}"></i>${m.s} ${m.w}@${m.rr}`;
 el.onclick=()=>{sel.has(i)?sel.delete(i):sel.add(i);syncChips();redraw();};
 picker.appendChild(el);});
function syncChips(){picker.querySelectorAll('.chip').forEach(c=>
 c.classList.toggle('off',!sel.has(+c.dataset.i)));}

// ---- client-side recompute --------------------------------------------------
function compute(){
 const tx=[],eq=[],dd=[],net=[],ms=[];
 let e=0,pk=0,mdd=0,wins=0;
 const by={};                       // aggregation buckets
 const mo={},yr={},hr=new Array(24).fill(0),dw=new Array(7).fill(0),sea=new Array(12).fill(0);
 const cnt={hr:new Array(24).fill(0),dw:new Array(7).fill(0)};
 for(let i=0;i<NT;i++){
  if(!sel.has(TR.w[i]))continue;
  const n=TR.n[i], t=TR.tx[i]*MIN, d=new Date(TR.te[i]*MIN);
  e+=n; pk=Math.max(pk,e); mdd=Math.max(mdd,pk-e);
  ms.push(t); eq.push(Math.round(e*10)/10); dd.push(Math.round((e-pk)*10)/10);
  net.push(n); if(n>0)wins++;
  // UTC getters ONLY. MT5 timestamps are naive broker time and were encoded as
  // epoch-treated-as-UTC, so getUTC* round-trips the original clock exactly.
  // getHours()/getDay() would re-interpret them in the VIEWER's timezone: on a
  // GMT+2/+3 machine window "2-3" showed up as hour 4-5, and because the shift
  // is DST-dependent it also moved trades across day/month boundaries.
  const y=d.getUTCFullYear(),m=d.getUTCMonth();
  const mk=y+'-'+String(m+1).padStart(2,'0');
  mo[mk]=(mo[mk]||0)+n; yr[y]=(yr[y]||0)+n; sea[m]+=n;
  hr[d.getUTCHours()]+=n; cnt.hr[d.getUTCHours()]++;
  dw[d.getUTCDay()]+=n;   cnt.dw[d.getUTCDay()]++;
 }
 return {ms,eq,dd,net,mo,yr,hr,dw,sea,cnt,mdd,wins,n:net.length,
         total:e,peak:pk};
}
function perWindowSeries(){
 const out=[];
 WM.forEach((m,wi)=>{ if(!sel.has(wi))return;
  const x=[],y=[];let e=0;
  for(let i=0;i<NT;i++){ if(TR.w[i]!==wi)continue; e+=TR.n[i];
   x.push(TR.tx[i]*MIN); y.push(Math.round(e*10)/10);}
  out.push({x,y,name:`${m.s} ${m.w}@${m.rr}`,color:m.color});});
 return out;
}

const BAR=v=>v.map(x=>x>=0?'#59a14f':'#e15759');
const MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const DAYS=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];

function redraw(){
 const c=compute();
 document.getElementById('selstat').innerHTML=
  `<span>windows <b>${sel.size}/${WM.length}</b></span>`+
  `<span>trades <b>${c.n.toLocaleString()}</b></span>`+
  `<span>net <b>${fmt$(c.total)}</b></span>`+
  `<span>max DD <b>${fmt$(c.mdd)}</b></span>`+
  `<span>recovery <b>${c.mdd>0?(c.total/c.mdd).toFixed(2):'—'}</b></span>`+
  `<span>win% <b>${c.n?(100*c.wins/c.n).toFixed(1):'—'}</b></span>`;

 // equity + drawdown, shared x-axis (zoom one -> zooms both)
 const traces=[];
 perWindowSeries().forEach(s=>traces.push({x:s.x,y:s.y,name:s.name,type:'scatter',
   mode:'lines',line:{width:1.1,color:s.color},opacity:.75,
   hovertemplate:'%{x|%Y-%m-%d}<br>'+s.name+': $%{y:,.0f}<extra></extra>'}));
 traces.push({x:c.ms,y:c.eq,name:'SELECTION',type:'scatter',mode:'lines',
   line:{width:2.4,color:'#111'},
   hovertemplate:'%{x|%Y-%m-%d}<br>equity: $%{y:,.0f}<extra></extra>'});
 traces.push({x:c.ms,y:c.dd,name:'drawdown',type:'scatter',mode:'lines',fill:'tozeroy',
   line:{width:1,color:'#e15759'},fillcolor:'rgba(225,87,89,.28)',yaxis:'y2',
   hovertemplate:'%{x|%Y-%m-%d}<br>DD: $%{y:,.0f}<extra></extra>'});
 Plotly.react('c_equity',traces,{
   margin:{l:66,r:16,t:34,b:34},font:FONT,hovermode:'x unified',dragmode:'zoom',
   title:{text:'Equity & drawdown of the selected windows',x:0,font:{size:13}},
   xaxis:{domain:[0,1],anchor:'y2',showgrid:true,gridcolor:'#eef0f3',type:'date',
         tickformatstops:[],hoverformat:'%Y-%m-%d'},
   yaxis:{domain:[.34,1],title:'equity $',gridcolor:'#eef0f3',zeroline:false},
   yaxis2:{domain:[0,.26],title:'DD $',gridcolor:'#eef0f3'},
   legend:{orientation:'h',y:-.12,font:{size:10}},
   plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);

 // monthly
 const mk=Object.keys(c.mo).sort(),mv=mk.map(k=>c.mo[k]);
 Plotly.react('c_monthly',[{x:mk,y:mv,type:'bar',marker:{color:BAR(mv)},
   hovertemplate:'%{x}<br>$%{y:,.0f}<extra></extra>'}],
  {margin:{l:60,r:10,t:34,b:40},font:FONT,title:{text:'Monthly net PnL',x:0,font:{size:13}},
   yaxis:{gridcolor:'#eef0f3'},xaxis:{type:'category',nticks:14},
   plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);

 // yearly
 const yk=Object.keys(c.yr).sort(),yv=yk.map(k=>c.yr[k]);
 Plotly.react('c_yearly',[{x:yk,y:yv,type:'bar',marker:{color:BAR(yv)},
   text:yv.map(v=>fmt$(v)),textposition:'outside',
   hovertemplate:'%{x}<br>$%{y:,.0f}<extra></extra>'}],
  {margin:{l:60,r:10,t:34,b:40},font:FONT,title:{text:'Yearly net PnL',x:0,font:{size:13}},
   yaxis:{gridcolor:'#eef0f3'},xaxis:{type:'category'},
   plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);

 // heatmap year x month
 const years=[...new Set(mk.map(k=>k.slice(0,4)))].sort();
 const z=years.map(y=>MONTHS.map((_,m)=>{
   const k=y+'-'+String(m+1).padStart(2,'0');return (k in c.mo)?c.mo[k]:null;}));
 const amax=Math.max(1,...z.flat().filter(v=>v!=null).map(Math.abs));
 Plotly.react('c_heat',[{z,x:MONTHS,y:years,type:'heatmap',zmid:0,zmin:-amax,zmax:amax,
   colorscale:[[0,'#b2182b'],[.5,'#f7f7f7'],[1,'#1a7f37']],
   hovertemplate:'%{y} %{x}<br>$%{z:,.0f}<extra></extra>',
   colorbar:{thickness:12,title:{text:'$',side:'right'}}}],
  {margin:{l:56,r:10,t:34,b:34},font:FONT,
   title:{text:'Monthly PnL heatmap (year × month)',x:0,font:{size:13}},
   plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);

 // by entry hour
 Plotly.react('c_hour',[{x:c.hr.map((_,i)=>i),y:c.hr,type:'bar',marker:{color:BAR(c.hr)},
   customdata:c.cnt.hr,
   hovertemplate:'%{x}:00<br>$%{y:,.0f}<br>%{customdata} trades<extra></extra>'}],
  {margin:{l:60,r:10,t:34,b:40},font:FONT,
   title:{text:'PnL by hour of day (entry)',x:0,font:{size:13}},
   xaxis:{dtick:1,title:'hour'},yaxis:{gridcolor:'#eef0f3'},
   plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);

 // by weekday (Mon-Fri)
 const dOrd=[1,2,3,4,5],dV=dOrd.map(i=>c.dw[i]);
 Plotly.react('c_dow',[{x:dOrd.map(i=>DAYS[i]),y:dV,type:'bar',marker:{color:BAR(dV)},
   customdata:dOrd.map(i=>c.cnt.dw[i]),
   hovertemplate:'%{x}<br>$%{y:,.0f}<br>%{customdata} trades<extra></extra>'}],
  {margin:{l:60,r:10,t:34,b:40},font:FONT,
   title:{text:'PnL by weekday (entry)',x:0,font:{size:13}},
   yaxis:{gridcolor:'#eef0f3'},plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);

 // seasonality
 Plotly.react('c_seas',[{x:MONTHS,y:c.sea,type:'bar',marker:{color:BAR(c.sea)},
   hovertemplate:'%{x}<br>$%{y:,.0f}<extra></extra>'}],
  {margin:{l:60,r:10,t:34,b:40},font:FONT,
   title:{text:'PnL by month of year (all years combined)',x:0,font:{size:13}},
   yaxis:{gridcolor:'#eef0f3'},plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);

 // trade distribution
 Plotly.react('c_hist',[{x:c.net,type:'histogram',nbinsx:60,marker:{color:'#4e79a7'},
   hovertemplate:'PnL %{x}<br>%{y} trades<extra></extra>'}],
  {margin:{l:60,r:10,t:34,b:40},font:FONT,
   title:{text:'Per-trade PnL distribution',x:0,font:{size:13}},
   xaxis:{title:'net $ per trade'},yaxis:{title:'trades',gridcolor:'#eef0f3'},
   shapes:[{type:'line',x0:0,x1:0,yref:'paper',y0:0,y1:1,
            line:{color:'#6b7280',width:1,dash:'dot'}}],
   plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);
}

// ---- provenance: the lineage of THIS report --------------------------------
(function(){
 const box=document.getElementById('provbox');
 const ch=DATA.prov_chain||[];
 if(!box)return;
 if(!ch.length){box.style.display='none';return;}
 const head=ch[0],g=head.git||{};
 const ov=[];
 ch.forEach(l=>Object.entries(l.overrides||{}).forEach(([k,v])=>{
   if(v&&(!Array.isArray(v)||v.length))ov.push(`${l.step}: ${k}=${Array.isArray(v)?v.join(' '):v}`);}));
 const cut=ch.map(l=>l.data_cutoff).find(Boolean);
 const val=ch.find(l=>l.validated_passes!=null)||{};
 const cal=ch.find(l=>l.calibration_entries!=null)||{};
 const ea=(ch.find(l=>l.ea&&Object.keys(l.ea).length)||{}).ea||{};
 const rows=ch.map(l=>`<tr><td>${l.step||''}</td><td><code>${l.run_id||''}</code></td><td>${l.generated||''}</td></tr>`).join('');
 box.innerHTML='<h2>Provenance</h2>'+
  `<div>report run <code>${head.run_id}</code> · analysis code <code>${g.commit||'n/a'}${g.dirty?' +dirty':''}</code> · data cutoff <code>${cut||'n/a'}</code></div>`+
  `<div>validated ${val.validated_passes||0} pass(es) · ${val.validation_mismatches||0} mismatch · ${val.passes_missing_stats||0} without MT5 stats</div>`+
  `<div>DD calibration: ${cal.calibration_entries||0} pass(es) scaled, max &times;${Number(cal.calibration_max||1).toFixed(3)}</div>`+
  `<div>EA ex5: ${Object.entries(ea).map(([k,v])=>`${k} <code>${(v.ex5_sha256_16||[]).join(', ')}</code>`).join(' · ')||'n/a'}</div>`+
  (ov.length?`<div class="pw">overrides: ${ov.join(' · ')}</div>`:'<div>no overrides used</div>')+
  ((DATA.prov_warnings||[]).length?`<div class="pw">${DATA.prov_warnings.map(w=>'&#9888; '+w).join('<br>')}</div>`:'')+
  `<table><thead><tr><th>step</th><th>run id</th><th>generated</th></tr></thead><tbody>${rows}</tbody></table>`;
})();

// ---- step 3 accounts --------------------------------------------------------
function drawAccounts(){
 const tr=DATA.accounts.map(a=>{
  const x=[],y=[];let e=0;
  for(let i=0;i<a.n.length;i++){e+=a.n[i];x.push(a.tx[i]*MIN);y.push(Math.round(e*10)/10);}
  return {x,y,name:a.label,type:'scatter',mode:'lines',line:{width:1.6,color:a.color},
   hovertemplate:'%{x|%Y-%m-%d}<br>'+a.name+': $%{y:,.0f}<extra></extra>'};});
 Plotly.react('c_accounts',tr,{margin:{l:66,r:16,t:34,b:34},font:FONT,
   title:{text:'Per-account equity (one-position replay)',x:0,font:{size:13}},
   hovermode:'x unified',dragmode:'zoom',
   xaxis:{gridcolor:'#eef0f3',type:'date',hoverformat:'%Y-%m-%d'},
   yaxis:{title:'equity $',gridcolor:'#eef0f3'},
   legend:{orientation:'h',y:-.14,font:{size:10}},
   plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);
}

nav.children[0].click();
</script></body></html>"""

html = (TEMPLATE
        .replace("__PLOTLYJS__", po.get_plotlyjs())
        .replace("__GEN__", data["generated"])
        .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Report written: {OUT_HTML}  ({os.path.getsize(OUT_HTML)/1e6:.1f} MB)")
print(f"  {len(ALL):,} trades · {len(wmeta)} windows · {len(accounts)} accounts")
print("Open in any browser — self-contained (Plotly inlined), works offline.")
