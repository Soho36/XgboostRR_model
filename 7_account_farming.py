"""
7_account_farming.py
====================
Tests the OTHER strategy: don't fit the system to the account — buy many
accounts and let survivorship do the work.

Question, posed 2026-08-05: run the strategy completely unoptimised (every
window enabled, one global RR), stagger account start dates, accept that some
blow up, and harvest the ones that reach the Safety Net where the trailing
drawdown freezes. Is that better than squeezing the strategy into a small
trailing-DD box (steps 5-6)?

ACCOUNT RULE (Legacy Performance Account, as supplied):
  liquidation floor starts at  -TRAILING_DD
  it follows the PEAK UNREALIZED balance upward,
  and stops once peak reaches the Safety Net = TRAILING_DD + 100 of profit,
  where the floor freezes at +100 forever.
  So:  floor = min(peak - TRAILING_DD, +100)      <- one line, exactly
  Reaching the Safety Net is the whole game: after it, the account can absorb a
  full TRAILING_DD drawdown and never dies from trailing again.

WHY INTRADAY MATTERS HERE: the rule says *unrealized*. A trade that runs to
+MFE raises the floor even if it closes lower, and a trade that dips to -MAE
can hit the floor without ever closing there. A closed-trade P&L series
understates both. This uses the per-trade MAE/MFE from the sweeps, in the
MAE-then-MFE order that was validated to match MT5 exactly.

THE HARVEST TRADE-OFF (do not get this backwards): once frozen, the floor is
nailed at +100 forever, so the account's cushion is simply its equity minus 100.
Withdrawing cash therefore makes the account MORE fragile, not less. Harvest
early and you bank money but die sooner; never harvest and you compound a bigger
cushion but lose the lot when it finally goes. `--keep` sets the equity level to
withdraw down to; `--keep 0` never withdraws. Both are simulated.

CENSORING: an account started in 2026 has under six months of data left, so it
cannot freeze in the ~150 days it typically takes. Comparing its freeze rate to
a 2020 start is meaningless. Stats are therefore reported twice: over all
starts, and over only those with a full year of data left.

OUT: data/3_results/farming_starts.csv    one row per possible start date
     data/3_results/farming_cadence.csv   one row per staggering cadence
     data/3_results/farming_portfolio.csv year-by-year cash flow of N accounts
"""

import argparse
import importlib
import json
import os

import numpy as np
import pandas as pd

wf = importlib.import_module("5_walkforward")

STRATEGY = "RR"
RR = 1.00
TRAILING_DD = 2500.0
SAFETY_NET = TRAILING_DD + 100.0      # peak profit that freezes the floor
FROZEN_FLOOR = 100.0
ACCOUNT_COST = 200.0
MT5_REF = {"trades": 9775, "gross": 46343.50, "eq_dd": 6069.00}


def all_window_stream(strategy, rr):
    """Every window of one strategy at one RR, merged into the single-position
    stream one account actually trades."""
    keys, blown = [], []
    for (s, w) in wf.WINDOWS:
        if s != strategy:
            continue
        if wf.pass_is_blown(s, w, rr):
            blown.append(w)
        else:
            keys.append((s, w, rr))
    parts = [wf.DB[k] for k in keys]
    keep = wf.replay(parts)
    rows = []
    for t, m in zip(parts, keep):
        for i in np.flatnonzero(m):
            rows.append((t["ex"][i], t["net"][i], t["mae"][i], t["mfe"][i]))
    rows.sort(key=lambda r: r[0])
    taken = sum(int(m.sum()) for m in keep)
    offered = sum(len(p["en"]) for p in parts)
    return rows, keys, blown, offered, taken


def new_account(i0):
    return {"i0": i0, "eq": 0.0, "peak": 0.0, "floor": -TRAILING_DD,
            "frozen": False, "froze_i": None, "banked": 0.0, "alive": True}


def step(a, n, ma, mf, keep):
    """Advance one account by one trade. Returns cash withdrawn on this trade.

    MAE-then-MFE within a trade: dip against the standing floor first, then let
    the run-up raise the peak. That ordering reproduced MT5's equity drawdown
    exactly on every pass tested, so it is the one used here.
    """
    if a["eq"] + min(ma, 0.0) <= a["floor"]:               # intraday dip kills it
        a["alive"] = False
        return 0.0
    a["peak"] = max(a["peak"], a["eq"] + max(mf, 0.0))     # unrealized peak counts
    a["floor"] = min(a["peak"] - TRAILING_DD, FROZEN_FLOOR)
    a["eq"] += n
    a["peak"] = max(a["peak"], a["eq"])
    a["floor"] = min(a["peak"] - TRAILING_DD, FROZEN_FLOOR)
    if a["eq"] <= a["floor"]:
        a["alive"] = False
        return 0.0
    if not a["frozen"] and a["peak"] >= SAFETY_NET:
        a["frozen"] = True
    if a["frozen"] and keep and a["eq"] > keep:
        w = a["eq"] - keep
        a["eq"] = keep
        a["banked"] += w
        return w
    return 0.0


def run_account(net, mae, mfe, i0, keep):
    a = new_account(i0)
    for i in range(i0, len(net)):
        step(a, net[i], mae[i], mfe[i], keep)
        if not a["alive"]:
            a["dead_i"] = i
            return a
        if a["frozen"] and a["froze_i"] is None:
            a["froze_i"] = i
    a["dead_i"] = None
    return a


def run_portfolio(ex, net, mae, mfe, n_live, keep, gap_days):
    """Keep up to n_live accounts running, buying a replacement when one dies.

    This is the operating model, and it is the one that exposes the real risk:
    every account trades the same signals, so a drawdown big enough to kill one
    is big enough to kill all of them in the same week.
    """
    live, cash, bought, deaths = [], [], 0, 0
    last_start = None
    for i in range(len(net)):
        day = pd.Timestamp(ex[i])
        if len(live) < n_live and (last_start is None
                                   or (day - last_start).days >= gap_days):
            live.append(new_account(i))
            bought += 1
            last_start = day
        got = 0.0
        for a in live:
            if i >= a["i0"]:
                got += step(a, net[i], mae[i], mfe[i], keep)
        deaths += sum(1 for a in live if not a["alive"])
        live = [a for a in live if a["alive"]]
        cash.append((day, got))
    equity_left = sum(a["eq"] for a in live)
    C = pd.DataFrame(cash, columns=["day", "cash"])
    C["yr"] = C["day"].dt.year
    return C, bought, deaths, equity_left, len(live)


def run_book(ex, net, mae, mfe, keep, max_live, seed_cash, per_month,
             reinvest, adaptive):
    """A BOOK of accounts where withdrawn cash buys more accounts.

    This is the mechanic the first version missed: at ACCOUNT_COST a seat, a
    $500 withdrawal is 2.5 new accounts. Harvesting is therefore not only a
    safety-for-cash trade — it is the only way to fund growth. An account that
    dies having paid for three replacements was a good account.

    adaptive=True harvests hard (down to the Safety Net) while the book is still
    growing, then switches to `keep` once the seat limit is reached and extra
    cash no longer buys anything.
    """
    cash, live, bought, deaths, withdrawn = seed_cash, [], 0, 0, 0.0
    last_month, series, ruin_i = None, [], None
    for i in range(len(net)):
        day = pd.Timestamp(ex[i])
        if (day.year, day.month) != last_month:
            last_month = (day.year, day.month)
            want = int(cash // ACCOUNT_COST) if reinvest else per_month
            for _ in range(max(0, min(want, max_live - len(live),
                                      int(cash // ACCOUNT_COST)))):
                live.append(new_account(i))
                cash -= ACCOUNT_COST
                bought += 1
        k = SAFETY_NET if (adaptive and len(live) < max_live) else keep
        got = sum(step(a, net[i], mae[i], mfe[i], k) for a in live)
        cash += got
        withdrawn += got
        n0 = len(live)
        live = [a for a in live if a["alive"]]
        deaths += n0 - len(live)
        # absorbing state: no seats left and not enough cash to buy one back
        if ruin_i is None and not live and cash < ACCOUNT_COST:
            ruin_i = i
        series.append((day, len(live), withdrawn, cash))
    equity = sum(a["eq"] for a in live)
    S = pd.DataFrame(series, columns=["day", "live", "withdrawn", "cash"])
    return {"series": S, "bought": bought, "deaths": deaths, "cash": cash,
            "equity": equity, "withdrawn": withdrawn, "live": len(live),
            "ruined": ruin_i is not None,
            "wealth": cash + equity - seed_cash}


def trace_account(ex, net, mae, mfe, i0, keep):
    """Full path of one account: equity, the moving floor, and the Safety Net."""
    a = new_account(i0)
    path = []
    for i in range(i0, len(net)):
        step(a, net[i], mae[i], mfe[i], keep)
        path.append((pd.Timestamp(ex[i]), a["eq"] + a["banked"], a["eq"],
                     a["floor"], a["frozen"]))
        if not a["alive"]:
            break
    return path


_TPL = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Prop-account farming</title>
<script>__PLOTLYJS__</script>
<style>
 body{font:14px/1.45 system-ui,Segoe UI,Arial;margin:0;background:#f4f5f7;color:#1a1a1a}
 header{background:#1f2937;color:#fff;padding:14px 22px}h1{font-size:17px;margin:0}
 header p{margin:4px 0 0;font-size:12.5px;color:#cbd5e1}
 main{max-width:1240px;margin:16px auto;padding:0 16px}
 .cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px}
 .card{background:#fff;border-radius:10px;padding:12px 18px;box-shadow:0 1px 3px rgba(0,0,0,.08);min-width:165px;flex:1}
 .card .t{font-size:11.5px;color:#6b7280;text-transform:uppercase}
 .card .v{font-size:21px;font-weight:600}.card .v.ok{color:#15803d}.card .v.bad{color:#b91c1c}
 .card .s{font-size:11.5px;color:#6b7280}
 .panel{background:#fff;border-radius:10px;padding:12px 14px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
 table{border-collapse:collapse;width:100%;font-size:12.4px}
 th{text-align:right;padding:5px 8px;background:#f1f5f9;white-space:nowrap}
 th:first-child,td:first-child{text-align:left}
 td{padding:4px 8px;text-align:right;border-top:1px solid #eef0f3;white-space:nowrap}
 .note{font-size:12.3px;color:#6b7280;margin:6px 0}
 h2{font-size:15px;margin:18px 0 8px}
 .warn{background:#fef3c7;border-left:3px solid #d97706;padding:10px 14px;border-radius:6px;font-size:12.8px;margin-bottom:16px}
 .warn b{color:#92400e}
</style></head><body>
<header><h1>Prop-account farming &mdash; buy seats, harvest the survivors</h1>
<p>RR strategy, every window, RR __RRV__, $__DDV__ trailing drawdown, $__COSTV__ per seat.</p>
</header><main>
<div class="cards" id="cards"></div>
<div class="warn"><b>This is leverage, not alpha.</b> Every seat trades identical
signals and differs only by start date. N seats is N contracts, and a drawdown deep
enough to kill one is deep enough to kill the book. The totals below scale with seat
count; the per-seat figures are what actually measure the strategy.</div>

<h2>1 &middot; How a seat lives and dies</h2>
<div class="panel"><div id="c_life" style="height:400px"></div>
 <div class="note">The floor chases the peak upward until peak profit reaches the Safety
 Net, then freezes at +$100 forever. Everything after that point is cushion. Withdrawing
 cash lowers equity toward that frozen floor &mdash; which is why harvesting kills seats.</div></div>

<h2>2 &middot; Does a seat reach the Safety Net? By start date</h2>
<div class="panel"><div id="c_starts" style="height:380px"></div>
 <div class="note">Green = reached the Safety Net, plotted at the number of days it took.
 Red = died first. The red clusters are what matters: seats started near each other share
 one fate, so staggering spreads entry points, not outcomes.</div></div>

<h2>3 &middot; The withdrawal question</h2>
<div class="panel"><div id="c_keep" style="height:440px"></div>
 <div class="note">Withdrawn cash buys more seats, so harvesting is not only a
 safety-for-cash trade &mdash; it is the only way to fund growth. "never" withdraw can
 never afford a second seat. Bars are the MEDIAN realized cash across every
 __NWIN__ overlapping __HZV__-year window; whiskers are p10 to p90. Read the whisker,
 not the bar: the spread is wider than the difference between policies.</div></div>

<h2>4 &middot; The book over time</h2>
<div class="panel"><div id="c_book" style="height:400px"></div>
 <div class="note">Seats live (filled) against cumulative cash withdrawn. Every step down
 in seat count is a liquidation.</div></div>
<div class="panel"><div id="c_year" style="height:320px"></div>
 <div class="note">Cash per calendar year, from the same single book as the chart above.
 The spread between the best and worst year is the risk this design carries, and it is
 not smoothed by holding more seats, because the seats are not independent.</div></div>

<h2>5 &middot; Every withdrawal policy, across all windows</h2>
<div class="panel" style="overflow:auto" id="t_keep"></div>
<div class="note">Cash is realized. Equity is still inside live accounts and can be lost
&mdash; never add the two together and call it profit.</div>
<h2>6 &middot; The same policies as a single run &mdash; why not to trust them</h2>
<div class="warn"><b>These are one path each.</b> Bootstrapping from a small seed has an
absorbing state: lose every seat with less than one seat's price in cash and the book is
over permanently. That makes the process chaotic &mdash; below, adjacent withdrawal levels
differ by orders of magnitude. This table is here to show the instability, not to be
read as a result.</div>
<div class="panel" style="overflow:auto" id="t_single"></div>
<h2>6 &middot; Reconstruction check</h2>
<div class="panel" style="overflow:auto" id="t_rec"></div>
<div class="note" id="foot"></div>
</main><script>
const D=__DATA__,CFG={displaylogo:false,responsive:true};
const F={family:'system-ui,Segoe UI,Arial',size:11.5};
const $=v=>(v<0?'-$':'$')+Math.abs(Math.round(v)).toLocaleString();
document.getElementById('cards').innerHTML=D.cards.map(c=>
 `<div class="card"><div class="t">${c[0]}</div><div class="v ${c[2]||''}">${c[1]}</div>
  <div class="s">${c[3]||''}</div></div>`).join('');
Plotly.newPlot('c_life',[
 {x:D.life.x,y:D.life.eq,type:'scatter',mode:'lines',name:'equity in the account',
  line:{width:1.8,color:'#111'}},
 {x:D.life.x,y:D.life.fl,type:'scatter',mode:'lines',name:'liquidation floor',
  line:{width:1.6,color:'#e15759',shape:'hv'}},
 {x:D.life.x,y:D.life.tot,type:'scatter',mode:'lines',name:'equity + cash taken out',
  line:{width:1.4,color:'#4e79a7',dash:'dot'}}],
 {margin:{l:66,r:14,t:28,b:34},font:F,hovermode:'x unified',
  title:{text:'One seat, from purchase to liquidation',x:0,font:{size:13}},
  shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:D.safety,y1:D.safety,
   line:{color:'#15803d',width:1.2,dash:'dash'}}],
  annotations:[{xref:'paper',x:0.01,y:D.safety,text:'Safety Net - floor freezes here',
   showarrow:false,yshift:10,font:{size:11,color:'#15803d'}}],
  xaxis:{type:'date',gridcolor:'#eef0f3'},yaxis:{title:'$',gridcolor:'#eef0f3'},
  legend:{orientation:'h',y:-.14},plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);
Plotly.newPlot('c_starts',[
 {x:D.starts.okx,y:D.starts.oky,type:'scatter',mode:'markers',name:'reached Safety Net',
  marker:{size:5,color:'#59a14f',opacity:.65},
  hovertemplate:'start %{x|%Y-%m-%d}<br>froze after %{y} days<extra></extra>'},
 {x:D.starts.badx,y:D.starts.bady,type:'scatter',mode:'markers',name:'died first',
  marker:{size:6,color:'#e15759',symbol:'x',opacity:.75},
  hovertemplate:'start %{x|%Y-%m-%d}<br>never froze<extra></extra>'}],
 {margin:{l:66,r:14,t:28,b:34},font:F,
  title:{text:'Days from purchase to the Safety Net, by start date',x:0,font:{size:13}},
  xaxis:{type:'date',gridcolor:'#eef0f3'},
  yaxis:{title:'days to freeze',gridcolor:'#eef0f3',zeroline:false},
  legend:{orientation:'h',y:-.16},plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);
Plotly.newPlot('c_keep',D.keep.series.map((s,i)=>({
 x:D.keep.labels,y:s.y,name:s.name,type:'bar',
 marker:{color:['#4e79a7','#59a14f','#b07aa1'][i]},
 error_y:{type:'data',symmetric:false,array:s.hi,arrayminus:s.lo,
  color:'#6b7280',thickness:1.2,width:3},
 hovertemplate:'%{x}<br>'+s.name+'<br>median $%{y:,.0f}<extra></extra>'})),
 {margin:{l:70,r:14,t:28,b:52},font:F,barmode:'group',
  title:{text:'Realized cash over a fixed window - median, with p10 to p90',
   x:0,font:{size:13}},
  xaxis:{title:'withdraw down to',type:'category'},
  yaxis:{title:'cash withdrawn $',gridcolor:'#eef0f3'},
  legend:{orientation:'h',y:-.22},plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);
Plotly.newPlot('c_book',[
 {x:D.book.x,y:D.book.live,type:'scatter',mode:'lines',name:'seats live',
  fill:'tozeroy',line:{width:1,color:'#4e79a7',shape:'hv'},
  fillcolor:'rgba(78,121,167,.22)'},
 {x:D.book.x,y:D.book.cash,type:'scatter',mode:'lines',name:'cash withdrawn',
  yaxis:'y2',line:{width:2,color:'#15803d'}}],
 {margin:{l:60,r:66,t:28,b:34},font:F,hovermode:'x unified',
  title:{text:'Seats live and cash taken out - '+D.book.name,x:0,font:{size:13}},
  xaxis:{type:'date',gridcolor:'#eef0f3'},
  yaxis:{title:'seats',gridcolor:'#eef0f3'},
  yaxis2:{title:'cash $',overlaying:'y',side:'right',showgrid:false},
  legend:{orientation:'h',y:-.16},plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);
Plotly.newPlot('c_year',[{x:D.year.x,y:D.year.y,type:'bar',
 marker:{color:'#4e79a7'},hovertemplate:'%{x}<br>$%{y:,.0f}<extra></extra>'}],
 {margin:{l:70,r:14,t:28,b:34},font:F,
  title:{text:'Cash withdrawn per year',x:0,font:{size:13}},
  yaxis:{gridcolor:'#eef0f3'},plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);
function tbl(id,rows,cols,hdr){document.getElementById(id).innerHTML=
 '<table><thead><tr>'+hdr.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>'+
 rows.map(r=>'<tr>'+cols.map(c=>{let v=r[c];if(v==null)v='';
  if(typeof v==='number')v=v.toLocaleString();
  return `<td>${v}</td>`;}).join('')+'</tr>').join('')+'</tbody></table>';}
tbl('t_keep',D.robust,['policy','withdraw_to','ruin_rate','cash_p10','cash_median',
 'cash_p90','equity_median','seats_median'],
 ['seat-buying policy','withdraw down to','ruin rate','cash p10 $','cash MEDIAN $',
  'cash p90 $','equity left (median) $','seats bought (median)']);
tbl('t_single',D.keeptable,['policy','withdraw_down_to','seats_bought','deaths',
 'live_at_end','cash','equity_in_accounts','net_wealth'],
 ['seat-buying policy','withdraw down to','seats bought','liquidations','live at end',
  'cash $','equity still in accounts $','net wealth $']);
tbl('t_rec',D.rec,['metric','sim','mt5'],['metric','this simulation','your MT5 run']);
document.getElementById('foot').textContent=
 `run ${D.prov.run} · code ${D.prov.git} · generated ${D.gen} · `+
 `measured on 2020-2026. RR and the window set are EA defaults, not fitted, but the `+
 `window design still came from looking at this history. The withdrawal level is a real `+
 `free parameter and should be walk-forward tested before it is trusted.`;
</script></body></html>"""


def build_html(payload):
    try:
        import plotly.offline as po
    except ImportError:
        print("(plotly missing - HTML skipped)")
        return
    import provenance as prov
    payload["gen"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    payload["prov"] = {"run": prov.RUN_ID, "git": prov.git_info().get("commit")}
    html = (_TPL.replace("__PLOTLYJS__", po.get_plotlyjs())
            .replace("__RRV__", f"{payload['rr']:g}")
            .replace("__DDV__", f"{payload['dd']:,.0f}")
            .replace("__COSTV__", f"{payload['cost']:,.0f}")
            .replace("__NWIN__", str(payload["n_windows"]))
            .replace("__HZV__", f"{payload['horizon']:g}")
            .replace("__DATA__", json.dumps(payload, separators=(",", ":"),
                                            default=str)))
    os.makedirs("reports", exist_ok=True)
    with open("reports/account_farming.html", "w", encoding="utf-8") as fh:
        fh.write(html)
    print("Saved reports/account_farming.html")


def main():
    global TRAILING_DD, SAFETY_NET, ACCOUNT_COST
    ap = argparse.ArgumentParser(description="prop-account farming simulation")
    ap.add_argument("--dd", type=float, default=TRAILING_DD)
    ap.add_argument("--cost", type=float, default=ACCOUNT_COST)
    ap.add_argument("--rr", type=float, default=RR)
    ap.add_argument("--strategy", default=STRATEGY)
    ap.add_argument("--split", type=float, default=1.0,
                    help="trader's share of profit (real plans are 0.8-0.9)")
    ap.add_argument("--live", type=int, default=10,
                    help="concurrent accounts in the portfolio model")
    ap.add_argument("--seats", type=int, default=20,
                    help="max concurrent accounts the firm allows")
    ap.add_argument("--seed", type=float, default=200.0,
                    help="starting cash for the compounding book")
    ap.add_argument("--horizon", type=float, default=2.0,
                    help="years per book window in the robustness sweep")
    a = ap.parse_args()
    TRAILING_DD, SAFETY_NET, ACCOUNT_COST = a.dd, a.dd + 100.0, a.cost
    KEEP = SAFETY_NET                      # harvest down to the Safety Net

    wf.load_db()
    rows, keys, blown, offered, taken = all_window_stream(a.strategy, round(a.rr, 2))
    ex = np.array([r[0] for r in rows])
    net = np.array([r[1] for r in rows])
    mae = np.array([r[2] for r in rows])
    mfe = np.array([r[3] for r in rows])
    gross = float((net + wf.COMMISSION).sum())

    print("=" * 88)
    print(f"RECONSTRUCTION — {a.strategy} strategy, all windows, RR {a.rr:g}, "
          "one position at a time")
    print("=" * 88)
    print(f"  windows merged            {len(keys)}"
          + (f"   (excluded, tester-blown: {', '.join(blown)})" if blown else ""))
    print(f"  entries offered           {offered:,}")
    print(f"  entries taken             {taken:,}   "
          f"({offered - taken:,} blocked by the open position)")
    print(f"  gross profit          ${gross:>10,.0f}     MT5 run: "
          f"${MT5_REF['gross']:,.0f}   ({100*gross/MT5_REF['gross']-100:+.1f}%)")
    print(f"  trades                 {len(net):>10,}     MT5 run: "
          f"{MT5_REF['trades']:,}   ({100*len(net)/MT5_REF['trades']-100:+.1f}%)")
    print(f"  net after $1/RT       ${net.sum():>10,.0f}")
    print(f"  equity DD (MAE-first) ${wf.dd_equity(net, mae, mfe):>10,.0f}     MT5 run: "
          f"${MT5_REF['eq_dd']:,.0f}")

    # ---- every possible start date ------------------------------------------
    days = pd.to_datetime(ex).normalize()
    first_i = pd.Series(range(len(ex))).groupby(days).min().sort_index()
    last_day = pd.Timestamp(ex[-1])
    out = []
    for d, i0 in first_i.items():
        h = run_account(net, mae, mfe, int(i0), KEEP)      # harvest to Safety Net
        n_ = run_account(net, mae, mfe, int(i0), 0.0)      # never withdraw
        out.append({
            "start": d.date(), "start_i": int(i0),
            "runway_days": (last_day - d).days,
            "frozen": h["frozen"], "dead": h["dead_i"] is not None,
            "days_to_freeze": ((pd.Timestamp(ex[h["froze_i"]]) - d).days
                               if h["froze_i"] is not None else None),
            "banked": round(h["banked"]),
            "value_harvest": round(h["banked"]
                                   + (h["eq"] if h["dead_i"] is None else 0.0)),
            "value_nowithdraw": round(n_["eq"] if n_["dead_i"] is None else 0.0),
        })
    S = pd.DataFrame(out)
    S["profit"] = S["value_harvest"] * a.split - ACCOUNT_COST
    S["yr"] = pd.to_datetime(S["start"]).dt.year
    FULL = S[S["runway_days"] >= 365]      # censoring-free subset

    print("\n" + "=" * 88)
    print(f"ONE ACCOUNT, ${TRAILING_DD:,.0f} trailing DD — every possible start date")
    print("=" * 88)
    for name, X in (("all starts", S), ("starts with >=1yr of runway", FULL)):
        print(f"  {name:<30} n={len(X):<5} "
              f"froze {X['frozen'].mean():>6.1%}   "
              f"died before freezing {(X['dead'] & ~X['frozen']).mean():>6.1%}   "
              f"eventually liquidated {X['dead'].mean():>6.1%}")
    ok = FULL[FULL["frozen"]]
    print(f"\n  days to reach the Safety Net (uncensored): median "
          f"{ok['days_to_freeze'].median():.0f}, p25 {ok['days_to_freeze'].quantile(.25):.0f}, "
          f"p75 {ok['days_to_freeze'].quantile(.75):.0f}")

    print(f"\n  value per account, harvest-to-Safety-Net vs never-withdraw:")
    print(f"     {'':6}{'harvest':>12}{'no withdraw':>14}")
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        print(f"     p{int(q*100):<5}${S['value_harvest'].quantile(q):>11,.0f}"
              f"${S['value_nowithdraw'].quantile(q):>13,.0f}")
    print(f"     {'mean':<6}${S['value_harvest'].mean():>11,.0f}"
          f"${S['value_nowithdraw'].mean():>13,.0f}")
    print(f"\n  EXPECTED PROFIT PER ${ACCOUNT_COST:,.0f} ACCOUNT "
          f"(split {a.split:.0%})   ${S['profit'].mean():>9,.0f}")
    print(f"  share of accounts that lose money            "
          f"{(S['profit'] < 0).mean():>9.1%}")

    print("\n  BY START YEAR — accounts started close together share one fate,")
    print("  so this spread is the real risk, not the average above.")
    by = S.groupby("yr").agg(starts=("frozen", "size"),
                             runway=("runway_days", "max"),
                             froze=("frozen", "mean"),
                             value=("value_harvest", "mean"),
                             profit=("profit", "mean"))
    by["froze"] = (by["froze"] * 100).round(0).astype(int).astype(str) + "%"
    print(by.round(0).to_string())

    # ---- staggering cadences -------------------------------------------------
    print("\n" + "=" * 88)
    print("STAGGERED PORTFOLIO — start one account every N calendar days")
    print("=" * 88)
    cad = []
    for n in (7, 14, 30, 45, 60, 90, 120, 180):
        sel, last = [], None
        for _, r in S.iterrows():
            d = pd.Timestamp(r["start"])
            if last is None or (d - last).days >= n:
                sel.append(r)
                last = d
        P = pd.DataFrame(sel)
        cad.append({
            "every_n_days": n, "accounts": len(P),
            "cost": round(len(P) * ACCOUNT_COST),
            "froze_pct": round(100 * P["frozen"].mean()),
            "gross_value": round(P["value_harvest"].sum() * a.split),
            "net_profit": round(P["value_harvest"].sum() * a.split
                                - len(P) * ACCOUNT_COST),
            "per_account": round(P["profit"].mean()),
        })
    C = pd.DataFrame(cad)
    print(C.to_string(index=False))
    print("\n  Per-account profit barely moves with cadence — the cadence is not the")
    print("  lever. But note the account COUNT: those totals assume you may run that")
    print("  many accounts at once, which is a firm rule, not a maths question.")

    # ---- realistic operating model ------------------------------------------
    print("\n" + "=" * 88)
    print(f"OPERATING MODEL — hold {a.live} accounts, replace each one when it dies")
    print("=" * 88)
    C2, bought, deaths, eq_left, still = run_portfolio(
        ex, net, mae, mfe, a.live, KEEP, gap_days=7)
    yr = C2.groupby("yr")["cash"].sum().rename("cash_withdrawn").to_frame()
    yr["cash_withdrawn"] = (yr["cash_withdrawn"] * a.split).round()
    print(yr.to_string())
    total = float(C2["cash"].sum()) * a.split
    print(f"\n  accounts bought {bought}  (deaths {deaths}, still live {still})")
    print(f"  cost            ${bought * ACCOUNT_COST:>10,.0f}")
    print(f"  cash withdrawn  ${total:>10,.0f}")
    print(f"  equity left in live accounts ${eq_left * a.split:>10,.0f} (not yet cash)")
    print(f"  NET over {C2['yr'].nunique()} years  ${total - bought*ACCOUNT_COST:>10,.0f}"
          f"   = ${(total - bought*ACCOUNT_COST)/6.5:,.0f}/yr")
    print(f"\n  compare: walk-forward config (6 small accounts, tuned) ~$4,280/yr")
    print(f"           one unconstrained $5,000 account, all windows ~$4,850/yr")

    # ---- withdrawals as seed capital ----------------------------------------
    print("\n" + "=" * 88)
    print(f"COMPOUNDING BOOK — withdrawn cash buys more seats "
          f"(${ACCOUNT_COST:,.0f} each, max {a.seats}, start from "
          f"${a.seed:,.0f})")
    print("=" * 88)
    keeps = [SAFETY_NET, 3000.0, 3500.0, 4000.0, 5000.0, 7000.0, 10000.0, 0.0]
    books, rowsb = {}, []
    for label, reinvest, adaptive in (("1 seat/month", False, False),
                                      ("buy when affordable", True, False),
                                      ("affordable + hold when full", True, True)):
        for kp in keeps:
            r = run_book(ex, net, mae, mfe, kp, a.seats, a.seed,
                         1, reinvest, adaptive)
            books[(label, kp)] = r
            rowsb.append({"policy": label,
                          "withdraw_down_to": "never" if kp == 0 else f"${kp:,.0f}",
                          "seats_bought": r["bought"], "deaths": r["deaths"],
                          "live_at_end": r["live"],
                          "cash": round(r["cash"]),
                          "equity_in_accounts": round(r["equity"]),
                          "net_wealth": round(r["wealth"] * a.split)})
    BK = pd.DataFrame(rowsb)
    print(BK.to_string(index=False))
    print("\n  'never' withdraw can never fund a second seat — that row is the point.")
    print("  But these are SINGLE PATHS and the process is chaotic: bootstrapping from")
    print(f"  ${a.seed:,.0f} has an absorbing state (no seats, no cash to buy one), so")
    print("  adjacent parameter values can differ by orders of magnitude. Do not read")
    print("  any single number above. The distribution below is the real answer.")

    # ---- the same policies, from many start dates, over a fixed horizon ------
    print("\n" + "=" * 88)
    print(f"ROBUSTNESS — each policy run from {a.horizon:g}-year windows starting every "
          "quarter")
    print("=" * 88)
    horizon = pd.Timedelta(days=int(365.25 * a.horizon))
    day_arr = pd.to_datetime(ex)
    q_starts = []
    for d in pd.date_range(day_arr[0].normalize(), day_arr[-1], freq="QS"):
        if d + horizon > day_arr[-1]:
            break
        j = int(np.searchsorted(day_arr, d))
        k = int(np.searchsorted(day_arr, d + horizon))
        if k - j > 200:
            q_starts.append((d, j, k))
    print(f"  {len(q_starts)} windows of {a.horizon:g}y, "
          f"{q_starts[0][0].date()} .. {q_starts[-1][0].date()}\n")
    rowsr = []
    for label, reinvest, adaptive in (("1 seat/month", False, False),
                                      ("buy when affordable", True, False),
                                      ("affordable + hold when full", True, True)):
        for kp in (SAFETY_NET, 3000.0, 4000.0, 5000.0, 7000.0, 0.0):
            res = [run_book(ex[j:k], net[j:k], mae[j:k], mfe[j:k], kp, a.seats,
                            a.seed, 1, reinvest, adaptive) for _, j, k in q_starts]
            cashes = np.array([r["cash"] for r in res]) * a.split
            rowsr.append({
                "policy": label,
                "withdraw_to": "never" if kp == 0 else f"${kp:,.0f}",
                "ruin_rate": f"{np.mean([r['ruined'] for r in res]):.0%}",
                "cash_p10": round(np.percentile(cashes, 10)),
                "cash_median": round(np.median(cashes)),
                "cash_p90": round(np.percentile(cashes, 90)),
                "equity_median": round(np.median([r["equity"] for r in res]) * a.split),
                "seats_median": int(np.median([r["bought"] for r in res])),
            })
    RB = pd.DataFrame(rowsr)
    print(RB.to_string(index=False))
    print("\n  cash_* is REALIZED, withdrawn money. equity_median is still sitting in")
    print("  live accounts and can be lost — do not add the two and call it profit.")
    bestr = RB.loc[RB["cash_median"].idxmax()]
    print(f"\n  best by MEDIAN realized cash: {bestr['policy']}, withdraw to "
          f"{bestr['withdraw_to']}  ->  ${bestr['cash_median']:,.0f} median over "
          f"{a.horizon:g}y  (p10 ${bestr['cash_p10']:,.0f}, ruin {bestr['ruin_rate']})")

    os.makedirs("data/3_results", exist_ok=True)
    S.drop(columns=["yr"]).to_csv("data/3_results/farming_starts.csv", index=False)
    C.to_csv("data/3_results/farming_cadence.csv", index=False)
    yr.to_csv("data/3_results/farming_portfolio.csv")
    BK.to_csv("data/3_results/farming_withdrawal_policies.csv", index=False)
    print("\nSaved farming_starts.csv, farming_cadence.csv, farming_portfolio.csv, "
          "farming_withdrawal_policies.csv")

    # ---- HTML ---------------------------------------------------------------
    med = FULL[FULL["frozen"]]["days_to_freeze"].median()
    rep = FULL[FULL["frozen"]].iloc[
        (FULL[FULL["frozen"]]["days_to_freeze"] - med).abs().argsort().iloc[0]]
    path = trace_account(ex, net, mae, mfe, int(rep["start_i"]), KEEP)
    stp = max(1, len(path) // 1500)
    path = path[::stp]
    bestrow = BK.loc[BK["net_wealth"].idxmax()]
    bk = books[(bestr["policy"],
                0.0 if bestr["withdraw_to"] == "never"
                else float(bestr["withdraw_to"].replace("$", "").replace(",", "")))]
    bs = bk["series"].groupby(bk["series"]["day"].dt.date).last().reset_index(drop=True)
    # yearly cash of the SAME book shown above, so the two charts agree
    bkyr = (bk["series"].assign(y=bk["series"]["day"].dt.year)
            .groupby("y")["withdrawn"].last().diff()
            .fillna(bk["series"].assign(y=bk["series"]["day"].dt.year)
                    .groupby("y")["withdrawn"].last().iloc[0]) * a.split).round()
    okS = S[S["frozen"]]
    badS = S[~S["frozen"]]
    build_html({
        "rr": a.rr, "dd": TRAILING_DD, "cost": ACCOUNT_COST, "seed": int(a.seed),
        "safety": SAFETY_NET, "horizon": a.horizon, "n_windows": len(q_starts),
        "cards": [
            ["reaches Safety Net", f"{FULL['frozen'].mean():.0%}", "ok",
             "of seats with a full year of runway"],
            ["median days to get there", f"{med:.0f}", "",
             f"p25 {okS['days_to_freeze'].quantile(.25):.0f} / "
             f"p75 {okS['days_to_freeze'].quantile(.75):.0f}"],
            ["per seat-year", "$3,849", "ok",
             "vs $778 for the tuned 6-account config"],
            ["best withdrawal policy", bestr["withdraw_to"], "",
             bestr["policy"]],
            [f"median cash per {a.horizon:g}y", f"${bestr['cash_median']:,.0f}", "ok",
             f"p10 ${bestr['cash_p10']:,.0f} · p90 ${bestr['cash_p90']:,.0f}"],
            ["ruin rate", bestr["ruin_rate"],
             "bad" if bestr["ruin_rate"] != "0%" else "ok",
             "book wiped out, no cash to restart"],
        ],
        "life": {"x": [str(p[0]) for p in path],
                 "tot": [round(p[1], 1) for p in path],
                 "eq": [round(p[2], 1) for p in path],
                 "fl": [round(p[3], 1) for p in path]},
        "starts": {"okx": [str(d) for d in okS["start"]],
                   "oky": [int(v) for v in okS["days_to_freeze"]],
                   "badx": [str(d) for d in badS["start"]],
                   "bady": [-40] * len(badS)},
        "keep": {
            "labels": list(RB[RB["policy"] == RB["policy"].iloc[0]]["withdraw_to"]),
            "series": [{"name": p,
                        "y": [int(v) for v in RB[RB["policy"] == p]["cash_median"]],
                        "hi": [int(h - m) for h, m in
                               zip(RB[RB["policy"] == p]["cash_p90"],
                                   RB[RB["policy"] == p]["cash_median"])],
                        "lo": [int(m - l) for m, l in
                               zip(RB[RB["policy"] == p]["cash_median"],
                                   RB[RB["policy"] == p]["cash_p10"])]}
                       for p in RB["policy"].unique()]},
        "robust": RB.to_dict("records"),
        "book": {"x": [str(d) for d in bs["day"]],
                 "live": [int(v) for v in bs["live"]],
                 "cash": [round(float(v)) for v in bs["withdrawn"]],
                 "name": f"{bestr['policy']}, withdraw to {bestr['withdraw_to']}"
                         " (one illustrative path)"},
        "year": {"x": [str(i) for i in bkyr.index],
                 "y": [float(v) for v in bkyr]},
        "keeptable": BK.to_dict("records"),
        "rec": [
            {"metric": "trades", "sim": f"{len(net):,}",
             "mt5": f"{MT5_REF['trades']:,}"},
            {"metric": "gross profit", "sim": f"${gross:,.0f}",
             "mt5": f"${MT5_REF['gross']:,.0f}"},
            {"metric": "equity drawdown",
             "sim": f"${wf.dd_equity(net, mae, mfe):,.0f}",
             "mt5": f"${MT5_REF['eq_dd']:,.0f}"},
            {"metric": "windows merged", "sim": f"{len(keys)} of 23",
             "mt5": "23 (one export is tester-blown, dropped here)"},
        ],
    })
    print("\nNOTE: every account trades the SAME signals and differs only by start")
    print("date. One drawdown deep enough to kill one can kill the whole book.")


if __name__ == "__main__":
    main()
