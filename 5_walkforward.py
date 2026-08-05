"""
5_walkforward.py
================
Anchored walk-forward validation of the WHOLE selection process.
Design + pass criteria: FORWARD_TESTING_PLAN.md  (read that first).

Every number the pipeline reports is in-sample. This script answers the real
question: if we had frozen the pipeline's decision at the end of year N, what
would year N+1 have paid?

For each fold:
  1. slice every sweep file to the FIT period only
  2. re-run the step-1 selection on that slice (tiers, +/-SMOOTH_RR
     neighbourhood, DD cap, verdict) -> recommended RR per OK window
  3. re-run the step-3 ILP on the fit slice -> account allocation
  4. score that FROZEN decision on the untouched TEST period
Then stitch the test periods into one continuous out-of-sample equity curve.

Also runs the critical control: every window at one fixed RR, no selection.
If the machinery can't beat that out-of-sample, it isn't earning its keep.

Faithful to the production selection/allocation constraints (same commission,
neighbourhood robustness, strategy-pure groups of 1-2 windows, one-position
replay, and one group per account). The risk calculation deliberately uses the
MFE-first equity-DD upper bound for the cap and OOS survival checks. This is a
non-leaking conservative substitute for the per-pass MT5 calibration used by
step 1: those stats cover the full 2020-2026 run and therefore cannot be used
inside an earlier fit fold without looking into its test period.

OUT: data/3_results/walkforward_folds.csv / walkforward_summary.csv
     data/3_results/_provenance_step5.json
"""

import glob
import itertools
import json
import os
import re
import time

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

import provenance as prov

# ---- CONFIG (mirrors steps 1/3 — keep in sync) ------------------------------
SWEEPS_DIR = "data/1_sweeps"
COMMISSION = 1.0
MAX_DD_USD = 2000.0
DEPOSIT = 5000.0
MIN_RECOVERY = 2.0
SMOOTH_RR = 0.10
CAP_FRACTION = 0.85
MAX_PER_ACCOUNT = 2
BASELINE_RR = 1.50
ALLOW_MIXED_STRATEGIES = False
FORBID_ADJACENT_WINDOWS = False
# mirror 3_allocate_accounts.py
ACCOUNTS = {"PA-09-1500": 1500.0, "PA-10-1500": 1500.0, "PA-12-2000": 2000.0,
            "PA-13-2000": 2000.0, "PA-14-2000": 2000.0, "PA-15-2500": 2500.0}

FIT_START = pd.Timestamp("2020-01-01")
FOLDS = [  # (fit_end_exclusive, planned_test_end_exclusive)
    (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")),
    (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
    (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")),
    (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")),
]

COLS = ["ticket", "entry_time", "exit_time", "mae", "mfe", "profit", "candle_range"]


def dd_equity(net, mae, mfe):
    """MAE-then-MFE equity-DD estimate used by step 1."""
    eq = peak = mdd = 0.0
    for n, a, f in zip(net, mae, mfe):
        mdd = max(mdd, peak - (eq + min(a, 0.0)))
        peak = max(peak, eq + max(f, 0.0))
        eq += n
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return float(mdd)


def dd_equity_upper(net, mae, mfe):
    """MFE-then-MAE upper bound from the same per-trade excursions.

    Tick order inside a trade is unavailable in the export. MT5's tester DD is
    known to lie between this path and ``dd_equity``; use the upper path for
    WFA risk limits so a fold never relies on a future-period calibration.
    """
    eq = peak = mdd = 0.0
    for n, a, f in zip(net, mae, mfe):
        peak = max(peak, eq + max(f, 0.0))
        mdd = max(mdd, peak - (eq + min(a, 0.0)))
        eq += n
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return float(mdd)


# ---- LOAD EVERYTHING ONCE, WITH INPUT INTEGRITY GATES ----------------------
def fail_input(message):
    raise SystemExit(f"Walk-forward input integrity failure: {message}")


def read_manifest(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as e:
        fail_input(f"unreadable manifest {path}: {e}")


def read_pass(path):
    try:
        d = pd.read_csv(path, sep="\t", header=None, names=COLS, encoding="utf-16")
    except Exception as e:
        fail_input(f"unreadable sweep {path}: {e}")
    d = d[pd.to_numeric(d["ticket"], errors="coerce").notna()].copy()
    for c in ("mae", "mfe", "profit"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("entry_time", "exit_time"):
        d[c] = pd.to_datetime(d[c].astype(str).str.strip(),
                              format="%Y.%m.%d %H:%M:%S", errors="coerce")
    required = ["entry_time", "exit_time", "mae", "mfe", "profit"]
    if d.empty or d[required].isna().any().any():
        fail_input(f"empty or malformed trade data in {path}")
    return d.sort_values("exit_time")


DB = {}   # (strat, window, rr) -> dict of numpy arrays sorted by exit_time
MANIFESTS = {}
rr_grid = None
legacy_manifest_count = 0
DATA_START = DATA_END_EXCLUSIVE = None
WINDOWS = []


def load_db():
    """Read and gate every sweep pass into memory. Idempotent; call once.

    Kept as a function (rather than import-time work) so a driver script can
    reuse this exact loader and the exact selection/allocation code below
    instead of reimplementing it and drifting.
    """
    global rr_grid, legacy_manifest_count, DATA_START, DATA_END_EXCLUSIVE, WINDOWS
    if DB:
        return
    print("Loading verified sweeps (once; folds are in-memory slices) ...")
    t0 = time.time()
    cutoffs, starts = set(), set()
    sweep_dirs = sorted(glob.glob(os.path.join(SWEEPS_DIR, "*", "*")))
    for wdir in sweep_dirs:
        if not os.path.isdir(wdir):
            continue
        strat = os.path.basename(os.path.dirname(wdir))
        if strat.endswith("_stats"):
            continue
        window = os.path.basename(wdir)
        mf = os.path.join(wdir, "_manifest.json")
        if not os.path.isfile(mf):
            fail_input(f"missing manifest for {strat} {window}")
        manifest = read_manifest(mf)
        if manifest.get("complete") is False:
            fail_input(f"incomplete manifest for {strat} {window}")
        if manifest.get("complete") is None:
            legacy_manifest_count += 1
        if not manifest.get("from") or not manifest.get("to"):
            fail_input(f"manifest for {strat} {window} has no data range")
        starts.add(str(manifest["from"]))
        cutoffs.add(str(manifest["to"]))

        found = []
        for path in sorted(glob.glob(os.path.join(wdir, "*.csv"))):
            m = re.match(rf"^{re.escape(window)}_([0-9.]+)\.csv$", os.path.basename(path))
            if not m:
                fail_input(f"unexpected CSV name in {strat} {window}: {os.path.basename(path)}")
            rr = round(float(m.group(1)), 2)
            key = (strat, window, rr)
            if key in DB:
                fail_input(f"duplicate RR {rr:g} in {strat} {window}")
            d = read_pass(path)
            DB[key] = {
                "en": d["entry_time"].values.astype("datetime64[s]"),
                "ex": d["exit_time"].values.astype("datetime64[s]"),
                "net": (d["profit"].values - COMMISSION).astype(np.float64),
                "mae": d["mae"].values.astype(np.float64),
                "mfe": d["mfe"].values.astype(np.float64),
            }
            found.append(rr)
        found = tuple(sorted(found))
        declared = manifest.get("expected", manifest.get("files_expected",
                                                          manifest.get("files_collected")))
        if declared is not None and int(declared) != len(found):
            fail_input(f"{strat} {window} has {len(found)} passes, manifest says {declared}")
        if rr_grid is None:
            rr_grid = found
        elif found != rr_grid:
            fail_input(f"RR grid differs in {strat} {window}")
        MANIFESTS[(strat, window)] = manifest

    if not DB:
        fail_input(f"no sweep passes under {SWEEPS_DIR}")
    if len(starts) != 1 or len(cutoffs) != 1:
        fail_input(f"mixed source ranges: from={sorted(starts)}, to={sorted(cutoffs)}")
    try:
        DATA_START = pd.to_datetime(next(iter(starts)), format="%Y.%m.%d")
        DATA_END_EXCLUSIVE = (pd.to_datetime(next(iter(cutoffs)), format="%Y.%m.%d")
                              + pd.Timedelta(days=1))
    except (TypeError, ValueError) as e:
        fail_input(f"invalid manifest date: {e}")
    WINDOWS = sorted({(s, w) for (s, w, _r) in DB},
                     key=lambda k: (k[0], int(k[1].split("-")[0])))
    print(f"  {len(DB)} passes, {len(WINDOWS)} windows, {len(rr_grid)} RRs | "
          f"{DATA_START.date()} -> {(DATA_END_EXCLUSIVE - pd.Timedelta(days=1)).date()} "
          f"in {time.time()-t0:,.0f}s"
          + (f"  ({legacy_manifest_count} legacy manifests verified by file grid)"
             if legacy_manifest_count else ""))


def sl(key, a, b):
    """Trades of one pass with a <= exit_time < b."""
    d = DB[key]
    m = (d["ex"] >= np.datetime64(a)) & (d["ex"] < np.datetime64(b))
    return {k: v[m] for k, v in d.items()}


# ---- STEP-1 SELECTION, PER FOLD --------------------------------------------
def select(fit_a, fit_b):
    """recommended RR per OK window, computed only from the fit slice."""
    picks = {}
    for (s, w) in WINDOWS:
        rrs = sorted(r for (s2, w2, r) in DB if (s2, w2) == (s, w))
        prof = np.empty(len(rrs)); dd = np.empty(len(rrs))
        for i, r in enumerate(rrs):
            t = sl((s, w, r), fit_a, fit_b)
            prof[i] = t["net"].sum()
            # No full-period MT5 calibration here: that would leak test-period
            # information into the fit. The upper MAE/MFE ordering is a safe,
            # fit-only cap for selection and allocation.
            dd[i] = dd_equity_upper(t["net"], t["mae"], t["mfe"])
        arr = np.array(rrs)
        best_rec, best = -np.inf, None
        for i, r in enumerate(rrs):
            nb = np.abs(arr - r) <= SMOOTH_RR + 1e-9
            p_nb, d_nb = prof[nb].mean(), dd[nb].max()
            if prof[i] <= 0 or p_nb <= 0 or d_nb > MAX_DD_USD:
                continue
            rec = p_nb / d_nb if d_nb > 0 else 0.0
            if rec > best_rec:
                best_rec, best = rec, r
        if best is not None and best_rec >= MIN_RECOVERY:      # verdict OK
            picks[(s, w)] = best
    return picks


# ---- STEP-3 ALLOCATION, PER FOLD -------------------------------------------
def replay(parts):
    """One-position replay across windows sharing an account; returns index
    masks per part. parts: list of trade dicts (fit or test slice)."""
    order = []
    for pi, t in enumerate(parts):
        for i in range(len(t["en"])):
            order.append((t["en"][i], t["ex"][i], pi, i))
    order.sort(key=lambda x: (x[0], x[1]))
    keep = [np.zeros(len(t["en"]), dtype=bool) for t in parts]
    open_until = np.datetime64("1970-01-01")
    for en, ex, pi, i in order:
        if en >= open_until:
            keep[pi][i] = True
            open_until = ex
    return keep


def replayed_rows(keys, a, b):
    """Accepted trade rows of a group on [a,b), after one-position replay."""
    parts = [sl(k, a, b) for k in keys]
    keep = replay(parts)
    rows = []
    for t, m in zip(parts, keep):
        for i in np.flatnonzero(m):
            rows.append((t["ex"][i], t["net"][i], t["mae"][i], t["mfe"][i]))
    rows.sort(key=lambda x: x[0])
    return rows


def group_eval(keys, a, b):
    """Profit plus raw and conservative DD of a replayed window group."""
    rows = replayed_rows(keys, a, b)
    net = np.array([r[1] for r in rows]) if rows else np.zeros(0)
    mae = np.array([r[2] for r in rows]) if rows else np.zeros(0)
    mfe = np.array([r[3] for r in rows]) if rows else np.zeros(0)
    return (float(net.sum()), dd_equity(net, mae, mfe),
            dd_equity_upper(net, mae, mfe), rows)


def has_adjacent_hours(combo):
    hours = sorted(int(k[1].split("-")[0]) for k in combo)
    return any(b - a == 1 for a, b in zip(hours, hours[1:]))


def build_groups(picks, fit_a, fit_b):
    """Candidate account groups (strategy-pure, 1..MAX_PER_ACCOUNT windows) with
    their fit-period profit and drawdowns. Independent of the account cap, so a
    cap sweep can reuse one build."""
    keys = [(s, w, picks[(s, w)]) for (s, w) in picks]
    by_s = {}
    for k in keys:
        by_s.setdefault(k[0], []).append(k)
    groups = []
    pools = [list(picks)] if ALLOW_MIXED_STRATEGIES else by_s.values()
    for pool in pools:
        for size in range(1, min(MAX_PER_ACCOUNT, len(pool)) + 1):
            for combo in itertools.combinations(pool, size):
                if FORBID_ADJACENT_WINDOWS and has_adjacent_hours(combo):
                    continue
                p, d_raw, d_upper, _ = group_eval(combo, fit_a, fit_b)
                groups.append({"keys": combo, "profit": p, "dd_raw": d_raw,
                               "dd": d_upper})
    return keys, groups


def solve_groups(keys, groups, cap_fraction, require_full=True):
    """Exact ILP: each account at most one group (exactly one when require_full),
    each window at most once, group DD <= cap_fraction * account limit."""
    names = list(ACCOUNTS)
    var = [(gi, ai) for gi, g in enumerate(groups) for ai, n in enumerate(names)
           if g["dd"] <= cap_fraction * ACCOUNTS[n]]
    if not var:
        return []
    c = np.array([-groups[gi]["profit"] for gi, _ in var])
    rows, lb, ub = [], [], []
    for ai in range(len(names)):                     # same as step 3: exactly one group
        rows.append([1.0 if a == ai else 0.0 for _, a in var])
        lb.append(1 if require_full else 0); ub.append(1)
    for k in keys:                                   # each window <= once
        rows.append([1.0 if k in groups[gi]["keys"] else 0.0 for gi, _ in var])
        lb.append(0); ub.append(1)
    res = milp(c=c, constraints=LinearConstraint(np.array(rows), lb, ub),
               integrality=np.ones(len(var)), bounds=Bounds(0, 1))
    if not res.success:
        return []
    return [(groups[gi], names[ai]) for (gi, ai), x in zip(var, res.x) if x > 0.5]


def allocate(picks, fit_a, fit_b):
    """Exact ILP on the fit slice: strategy-pure groups of 1-2, each account at
    most one group, each window once, group DD <= CAP_FRACTION * account."""
    keys, groups = build_groups(picks, fit_a, fit_b)
    return solve_groups(keys, groups, CAP_FRACTION)


def score_allocation(alloc, a, b):
    """Score frozen groups on a test slice using the same replayed trades.

    The returned rows are exactly the rows used for profit and DD; this keeps
    the stitched OOS curve consistent with every account-level statistic.
    """
    total, worst, accepted = 0.0, ("", 0.0, 0.0, 0.0), []
    for g, acct in alloc:
        p, d_raw, d_upper, rows = group_eval(g["keys"], a, b)
        total += p
        if d_upper / ACCOUNTS[acct] > worst[1]:
            worst = (acct, d_upper / ACCOUNTS[acct], d_raw, d_upper)
        accepted.extend((ex, net, mae, mfe, acct) for ex, net, mae, mfe in rows)
    return total, worst, accepted


def fixed_rr_picks():
    """Risk-matched control: all windows use one pre-declared fixed RR.

    It retains the same fit-period allocator, account cap, and replay rule as
    the selected model, isolating the value of per-window RR selection.
    """
    picks = {}
    for s, w in WINDOWS:
        rrs = [r for s2, w2, r in DB if (s2, w2) == (s, w)]
        picks[(s, w)] = min(rrs, key=lambda r: abs(r - BASELINE_RR))
    return picks


_BLOWN = {}


def pass_is_blown(strat, window, rr):
    """True if MT5 liquidated this pass, so its per-trade export is truncated.

    The full-period stats are intentionally *not* used to calibrate a fold or
    alter its choice. They are used only to detect a pass whose tester
    liquidation omitted its fatal trade from the per-trade export; scoring such
    a pass would otherwise manufacture a false result. Cached per pass.
    """
    key = (strat, window, rr)
    if key in _BLOWN:
        return _BLOWN[key]
    path = os.path.join(SWEEPS_DIR, f"{strat}_stats", window,
                        f"{window}_{rr:.2f}_stats.csv")
    if not os.path.isfile(path):
        fail_input(f"missing MT5 stats for selected pass {strat} {window}@{rr:g}")
    try:
        try:                                       # MT5 writes UTF-16; tolerate UTF-8 too
            stats = pd.read_csv(path, sep="\t", encoding="utf-16")
        except UnicodeError:
            stats = pd.read_csv(path, sep="\t", encoding="utf-8")
        row = stats.iloc[0]
        blown = (float(row["equity_dd"]) >= DEPOSIT * 0.95 or
                 float(row["net_profit"]) <= -DEPOSIT * 0.95)
    except (OSError, ValueError, KeyError, IndexError) as e:
        fail_input(f"unreadable MT5 stats for selected pass {strat} {window}@{rr:g}: {e}")
    _BLOWN[key] = blown
    return blown


def reject_known_blown_picks(picks):
    """Fail closed if a selected export is known to be tester-truncated.

    Time-bounded MT5 stats are required if this gate fires.
    """
    for (strat, window), rr in picks.items():
        if pass_is_blown(strat, window, rr):
            raise SystemExit(
                "Walk-forward cannot score a selected tester-blown pass from a "
                f"truncated export: {strat} {window}@{rr:g}. Generate "
                "time-bounded MT5 stats for this fold before continuing."
            )


_WF_TPL = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Walk-forward — out-of-sample</title>
<script>__PLOTLYJS__</script>
<style>
 body{font:14px/1.45 system-ui,Segoe UI,Arial;margin:0;background:#f4f5f7;color:#1a1a1a}
 header{background:#1f2937;color:#fff;padding:14px 22px}h1{font-size:17px;margin:0}
 main{max-width:1240px;margin:16px auto;padding:0 16px}
 .cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:16px}
 .card{background:#fff;border-radius:10px;padding:12px 18px;box-shadow:0 1px 3px rgba(0,0,0,.08);min-width:170px}
 .card .t{font-size:11.5px;color:#6b7280;text-transform:uppercase}
 .card .v{font-size:21px;font-weight:600}.card .v.bad{color:#b91c1c}.card .v.ok{color:#15803d}
 .panel{background:#fff;border-radius:10px;padding:12px 14px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
 table{border-collapse:collapse;width:100%;font-size:12.4px}
 th{text-align:right;padding:5px 8px;background:#f1f5f9;white-space:nowrap}
 th:first-child,td:first-child{text-align:left}
 td{padding:4px 8px;text-align:right;border-top:1px solid #eef0f3;white-space:nowrap}
 .bad{color:#b91c1c;font-weight:600}.ok{color:#15803d;font-weight:600}
 .note{font-size:12.3px;color:#6b7280;margin:6px 0}
 h2{font-size:15px;margin:16px 0 8px}
</style></head><body>
<header><h1>Walk-forward — frozen decisions on unseen data</h1></header><main>
<div class="cards" id="cards"></div>
<div class="panel"><div id="c_eq" style="height:430px"></div>
 <div class="note">Stitched out-of-sample equity: each segment is a decision frozen at the
 fold boundary (dashed lines) and scored on data the selection never saw. Only
 replay-accepted trades of the allocated accounts are counted.</div></div>
<div class="panel"><div id="c_bars" style="height:300px"></div></div>
<div class="panel"><div id="c_surv" style="height:280px"></div>
 <div class="note">Survival: worst account's OOS drawdown as % of its limit
 (MFE-first upper bound). Above the red line, a real prop account is terminated.</div></div>
<h2>Folds</h2><div class="panel" style="overflow:auto" id="t_folds"></div>
<h2>Prop-reality: termination at the limit</h2>
<div class="panel" style="overflow:auto" id="t_breach"></div>
<div class="note" id="foot"></div>
</main><script>
const D=__DATA__,S=D.sum,CFG={displaylogo:false,responsive:true};
const F={family:'system-ui,Segoe UI,Arial',size:11.5};
const $=v=>(v<0?'-$':'$')+Math.abs(Math.round(v)).toLocaleString();
document.getElementById('cards').innerHTML=[
 ['WFE',S.WFE,S.WFE>=0.5?'ok':'bad'],
 ['folds OOS-positive',S.folds_OOS_positive+' / '+S.folds,S.folds_OOS_positive>=3?'ok':'bad'],
 ['OOS total (as scored)',$(S.OOS_total_net),''],
 ['termination-adjusted',$(D.adj.adjusted),D.adj.n?'bad':'ok'],
 ['fixed-RR control',$(S.fixed_RR_total_net),S.beats_fixed_RR_risk_matched?'ok':'bad'],
 ['breached acct-folds',D.adj.n,D.adj.n?'bad':'ok'],
].map(c=>`<div class="card"><div class="t">${c[0]}</div><div class="v ${c[2]}">${c[1]}</div></div>`).join('');
const shp=D.bounds.map(b=>({type:'line',x0:b,x1:b,yref:'paper',y0:0,y1:1,
 line:{color:'#9ca3af',width:1,dash:'dash'}}));
Plotly.newPlot('c_eq',[
 {x:D.eq.x,y:D.eq.y,type:'scatter',mode:'lines',name:'OOS equity',line:{width:2,color:'#111'},
  hovertemplate:'%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>'},
 {x:D.eq.x,y:D.eq.dd,type:'scatter',mode:'lines',name:'drawdown',yaxis:'y2',
  fill:'tozeroy',line:{width:1,color:'#e15759'},fillcolor:'rgba(225,87,89,.25)',
  hovertemplate:'%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>'}],
 {margin:{l:64,r:14,t:26,b:30},font:F,hovermode:'x unified',shapes:shp,
  title:{text:'Stitched out-of-sample equity (as scored, no termination)',x:0,font:{size:13}},
  xaxis:{type:'date',anchor:'y2',gridcolor:'#eef0f3'},
  yaxis:{domain:[.32,1],title:'equity $',gridcolor:'#eef0f3'},
  yaxis2:{domain:[0,.24],title:'DD $',gridcolor:'#eef0f3'},
  legend:{orientation:'h',y:-.12},plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);
const fl=D.folds.map(f=>'F'+f.fold+' ('+String(f.test_start).slice(0,4)+')');
Plotly.newPlot('c_bars',[
 {x:fl,y:D.folds.map(f=>f.OOS_net),name:'selection OOS',type:'bar',marker:{color:'#4e79a7'}},
 {x:fl,y:D.folds.map(f=>f.fixed_RR_OOS_net),name:'fixed-RR control',type:'bar',marker:{color:'#bab0ac'}}],
 {margin:{l:60,r:10,t:26,b:30},font:F,barmode:'group',
  title:{text:'Per fold: selection vs risk-matched fixed-RR control',x:0,font:{size:13}},
  yaxis:{gridcolor:'#eef0f3'},plot_bgcolor:'#fff',paper_bgcolor:'#fff',
  legend:{orientation:'h',y:-.14}},CFG);
Plotly.newPlot('c_surv',[
 {x:fl,y:D.folds.map(f=>f.OOS_worst_pct_of_limit),name:'selection',type:'bar',
  marker:{color:D.folds.map(f=>f.OOS_worst_pct_of_limit>100?'#e15759':'#59a14f')}},
 {x:fl,y:D.folds.map(f=>f.fixed_RR_worst_pct_of_limit),name:'fixed-RR control',type:'bar',
  marker:{color:'#bab0ac'}}],
 {margin:{l:60,r:10,t:26,b:30},font:F,barmode:'group',
  title:{text:'Worst account OOS drawdown, % of its limit',x:0,font:{size:13}},
  shapes:[{type:'line',x0:-.5,x1:fl.length-.5,y0:100,y1:100,line:{color:'#b91c1c',width:1.5,dash:'dot'}}],
  yaxis:{gridcolor:'#eef0f3',ticksuffix:'%'},plot_bgcolor:'#fff',paper_bgcolor:'#fff',
  legend:{orientation:'h',y:-.14}},CFG);
function tbl(id,rows,cols){document.getElementById(id).innerHTML=
 '<table><thead><tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>'+
 rows.map(r=>'<tr>'+cols.map(c=>{let v=r[c];if(v==null)v='';
  const cls=(c.includes('pct')&&v>100)||(c==='breached'&&v)?' class="bad"':'';
  if(typeof v==='number')v=v.toLocaleString();
  return `<td${cls}>${v}</td>`;}).join('')+'</tr>').join('')+'</tbody></table>';}
tbl('t_folds',D.folds,['fold','test_start','test_end_inclusive','windows_picked',
 'IS_net','OOS_net','OOS_worst_acct','OOS_worst_pct_of_limit','fixed_RR_OOS_net',
 'fixed_RR_worst_pct_of_limit','picks']);
tbl('t_breach',D.breach,['fold','acct','limit','OOS_net_full','breached',
 'breach_date','net_at_breach','post_breach_net']);
document.getElementById('foot').textContent=
 `run ${D.prov.run} · code ${D.prov.git} · generated ${D.gen} · `+
 `termination-adjusted = as-scored OOS minus post-breach profit (${$(D.adj.post)}) that a `+
 `real terminated account would never have earned; replacement eval fees not included.`;
</script></body></html>"""

# ---- WALK THE FOLDS ---------------------------------------------------------
def years(a, b):
    return max((b - a).days / 365.25, 1e-9)


# ---- PROP-REALITY BREACH ANALYSIS -------------------------------------------
# The scorer never terminates a breached account: it keeps counting that
# account's trades for the whole test period. A real prop account STOPS at its
# limit. This finds, per fold x account, the first trade where the running
# (upper-bound) equity DD exceeds the limit, and splits the account's OOS net
# into "up to breach" vs "after breach" — the latter is profit a real account
# would never have realised. termination_adjusted_OOS is the honest prop number
# (before eval/reset fees for replacing the blown account).
def breach_table(acct_rows):
    """acct_rows: (fold, acct, exit_time, net, mae, mfe) of scored OOS trades."""
    breach_rows = []
    for (fn, acct), grp in pd.DataFrame(
            acct_rows, columns=["fold", "acct", "ex", "net", "mae", "mfe"]
            ).groupby(["fold", "acct"]):
        grp = grp.sort_values("ex")
        limit = ACCOUNTS[acct]
        eq = peak = mdd = 0.0
        b_i, b_eq = None, None
        for i, (n, a_, f_) in enumerate(zip(grp["net"], grp["mae"], grp["mfe"])):
            peak = max(peak, eq + max(f_, 0.0))
            mdd = max(mdd, peak - (eq + min(a_, 0.0)))
            eq += n
            peak = max(peak, eq)
            mdd = max(mdd, peak - eq)
            if b_i is None and mdd > limit:
                b_i, b_eq = i, eq                   # breach happens ON this trade
        total = float(grp["net"].sum())
        breach_rows.append({
            "fold": fn, "acct": acct, "limit": limit, "OOS_net_full": round(total),
            "breached": b_i is not None,
            "breach_date": str(grp["ex"].iloc[b_i])[:10] if b_i is not None else "",
            "net_at_breach": round(b_eq) if b_i is not None else None,
            "post_breach_net": round(total - b_eq) if b_i is not None else 0,
        })
    return pd.DataFrame(breach_rows)


def main():
    load_db()
    fold_rows, oos_trades, acct_rows = [], [], []
    for fn, (fit_end, planned_test_end) in enumerate(FOLDS, 1):
        test_end = min(planned_test_end, DATA_END_EXCLUSIVE)
        if test_end <= fit_end:
            raise SystemExit(f"Fold {fn} has no test data after the source cutoff.")
        t0 = time.time()
        picks = select(FIT_START, fit_end)
        reject_known_blown_picks(picks)
        alloc = allocate(picks, FIT_START, fit_end)
        if len(alloc) != len(ACCOUNTS):
            raise SystemExit(f"Fold {fn}: no production-equivalent allocation at "
                             f"{CAP_FRACTION:.0%} cap ({len(alloc)}/{len(ACCOUNTS)} accounts).")
        is_p = sum(g["profit"] for g, _ in alloc)
        oos_p, worst, accepted = score_allocation(alloc, fit_end, test_end)
        oos_trades.extend(accepted)
        acct_rows.extend((fn, acct, ex, net, mae, mfe)
                         for ex, net, mae, mfe, acct in accepted)

        # The control uses exactly the same fit-period allocator, DD cap and
        # replay constraint. Only its RR rule is simpler: one fixed RR.
        base_alloc = allocate(fixed_rr_picks(), FIT_START, fit_end)
        if len(base_alloc) != len(ACCOUNTS):
            raise SystemExit(f"Fold {fn}: fixed-RR control has no risk-matched allocation at "
                             f"{CAP_FRACTION:.0%} cap ({len(base_alloc)}/{len(ACCOUNTS)} accounts).")
        base, base_worst, _ = score_allocation(base_alloc, fit_end, test_end)
        is_y, oos_y = years(FIT_START, fit_end), years(fit_end, test_end)
        fold_rows.append({
            "fold": fn, "fit_end_exclusive": fit_end.date(),
            "test_start": fit_end.date(),
            "test_end_inclusive": (test_end - pd.Timedelta(days=1)).date(),
            "windows_picked": len(picks), "accounts_used": len(alloc),
            "IS_net": round(is_p), "IS_net_per_yr": round(is_p / is_y),
            "OOS_net": round(oos_p), "OOS_net_per_yr": round(oos_p / oos_y),
            "OOS_worst_acct": worst[0], "OOS_worst_dd_raw": round(worst[2]),
            "OOS_worst_dd_upper": round(worst[3]),
            "OOS_worst_pct_of_limit": round(worst[1] * 100, 1),
            "fixed_RR_accounts_used": len(base_alloc),
            "fixed_RR_OOS_net": round(base),
            "fixed_RR_worst_acct": base_worst[0],
            "fixed_RR_worst_pct_of_limit": round(base_worst[1] * 100, 1),
            "picks": "; ".join(f"{s} {w}@{picks[(s,w)]:g}" for (s, w) in sorted(picks)),
        })
        r = fold_rows[-1]
        print(f"\nFold {fn}  fit->{fit_end.date()}  "
              f"test->{(test_end - pd.Timedelta(days=1)).date()}  ({time.time()-t0:,.0f}s)\n"
              f"  picked {r['windows_picked']} windows on {r['accounts_used']} accounts | "
              f"IS ${r['IS_net']:,}/{is_y:.1f}y  OOS ${r['OOS_net']:,}/{oos_y:.1f}y | "
              f"worst acct {r['OOS_worst_acct']} {r['OOS_worst_pct_of_limit']}% | "
              f"fixed-RR control ${r['fixed_RR_OOS_net']:,}")

    F = pd.DataFrame(fold_rows)
    oos_trades.sort(key=lambda x: x[0])
    eq = np.cumsum([n for _, n, *_ in oos_trades])
    peak = np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:]
    stitched_closed_dd = float((peak - eq).max()) if len(eq) else 0.0

    wfe = F["OOS_net_per_yr"].sum() / max(F["IS_net_per_yr"].sum(), 1e-9)
    summary = {
        "folds": len(F),
        "folds_OOS_positive": int((F["OOS_net"] > 0).sum()),
        "WFE": round(wfe, 3),
        "OOS_total_net": int(F["OOS_net"].sum()),
        "fixed_RR_total_net": int(F["fixed_RR_OOS_net"].sum()),
        "beats_fixed_RR_risk_matched": bool(F["OOS_net"].sum() > F["fixed_RR_OOS_net"].sum()),
        "stitched_OOS_closedDD": round(stitched_closed_dd),
        "any_acct_over_limit": bool((F["OOS_worst_pct_of_limit"] > 100).any()),
    }
    print("\n" + "=" * 78)
    print("WALK-FORWARD SUMMARY   (pass bar: WFE>=0.5, >=3/4 folds +, no acct >100%, "
          "beats risk-matched fixed-RR control)")
    print("=" * 78)
    for k, v in summary.items():
        print(f"  {k:<24} {v}")

    os.makedirs("data/3_results", exist_ok=True)
    F.to_csv("data/3_results/walkforward_folds.csv", index=False)
    pd.DataFrame([summary]).to_csv("data/3_results/walkforward_summary.csv", index=False)
    prov.write("data/3_results/_provenance_step5.json", prov.base(
        "5_walkforward",
        folds=[str(f[0].date()) for f in FOLDS],
        source_data={
            "from": str(DATA_START.date()),
            "to_inclusive": str((DATA_END_EXCLUSIVE - pd.Timedelta(days=1)).date()),
            "manifests": len(MANIFESTS),
            "windows": len(WINDOWS),
            "rr_count": len(rr_grid),
            "legacy_manifests_verified_by_file_grid": legacy_manifest_count,
        },
        summary=summary,
        settings={"CAP_FRACTION": CAP_FRACTION, "BASELINE_RR": BASELINE_RR,
                  "accounts": ACCOUNTS, "MAX_PER_ACCOUNT": MAX_PER_ACCOUNT,
                  "ALLOW_MIXED_STRATEGIES": ALLOW_MIXED_STRATEGIES,
                  "FORBID_ADJACENT_WINDOWS": FORBID_ADJACENT_WINDOWS,
                  "DD_FOR_CAP_AND_OOS": "MFE-first upper bound"},
        limitations=["no exact per-fold MT5 calibration; conservative MFE-first DD bound used",
                     "selected globally blown passes fail closed; time-bounded stats required",
                     "thresholds chosen with full-history knowledge",
                     "meta-parameters/window design remain historically fitted"]))
    print("\nSaved data/3_results/walkforward_folds.csv, walkforward_summary.csv")

    B = breach_table(acct_rows)
    post = int(B["post_breach_net"].sum())
    n_breach = int(B["breached"].sum())
    adj = int(F["OOS_net"].sum()) - post
    print("\nPROP-REALITY: account termination at limit")
    print(f"  breached account-folds        {n_breach} of {len(B)}")
    print(f"  post-breach net (uncounted by a real account) ${post:,}")
    print(f"  OOS as scored (no termination) ${int(F['OOS_net'].sum()):,}")
    print(f"  OOS termination-adjusted       ${adj:,}  (+ eval/reset fees x {n_breach})")
    B.to_csv("data/3_results/walkforward_breaches.csv", index=False)

    # ---- HTML REPORT --------------------------------------------------------
    try:
        import plotly.offline as po
        ms = [pd.Timestamp(t[0]).value // 10**6 for t in oos_trades]
        eqs = np.round(eq_ := np.cumsum([t[1] for t in oos_trades]), 1).tolist()
        dds = np.round(eq_ - np.maximum.accumulate(
            np.concatenate(([0.0], eq_)))[1:], 1).tolist()
        payload = {
            "sum": summary, "folds": F.to_dict("records"),
            "breach": B.to_dict("records"),
            "eq": {"x": ms, "y": eqs, "dd": dds},
            "bounds": [str(f[0].date()) for f in FOLDS],
            "adj": {"post": post, "n": n_breach, "adjusted": adj},
            "gen": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "prov": {"run": prov.RUN_ID, "git": prov.git_info().get("commit")},
        }
        html = (_WF_TPL.replace("__PLOTLYJS__", po.get_plotlyjs())
                .replace("__DATA__", json.dumps(payload, separators=(",", ":"), default=str)))
        with open("reports/walkforward.html", "w", encoding="utf-8") as fh:
            fh.write(html)
        print("Saved reports/walkforward.html")
    except ImportError:
        print("(plotly missing - HTML skipped)")


if __name__ == "__main__":
    main()
