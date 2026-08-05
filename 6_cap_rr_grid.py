"""
6_cap_rr_grid.py
================
Walk-forward 2D grid over the two parameters that actually matter after the
per-window RR layer failed (FORWARD_TESTING_PLAN.md §11):

    CAP_FRACTION  0.40 -> 0.85     how much of an account's DD limit we allow
    fixed RR      1.00 -> 2.50     one global take-profit RR for every window

Every cell is scored the same way `5_walkforward.py` scores its folds: fit on
2020->N, freeze the decision, score the untouched next year, repeat. Nothing is
chosen in-sample.

Two arms are evaluated on the identical rig:
  * `fixed`     — every window gets ONE global RR; the fit-period ILP still
                  chooses WHICH windows to trade on which account. This is the
                  candidate strategy.
  * `selected`  — the per-window RR selection of step 1 (`5_walkforward.select`).
                  Kept as the reference arm; its picks do not depend on the cap,
                  so it costs one extra build per fold and shows whether the
                  earlier "selection loses" result holds at every cap.

WHY THIS IS CHEAP: the expensive part is evaluating candidate account groups on
the fit slice, and that depends on (arm, RR, fold) but NOT on the cap. So groups
are built once and every cap is a re-solve of the ILP over the same groups.

PRE-REGISTERED DECISION RULE (fixed before looking at any output):
  1. the cell must allocate in all folds;
  2. ZERO breached account-folds out of 24 (survival first — a breached prop
     account is terminated, no later profit exists for it);
  3. among survivors, highest termination-adjusted OOS total;
  4. prefer a cell in the MIDDLE of a contiguous surviving plateau over an
     isolated argmax — the same "plateau, not pixel peak" rule step 1 uses for
     RR, for the same reason (neighbouring cells differ by sample noise).

HONEST LIMITATION, stated up front: choosing a cell by out-of-sample score makes
that score optimistic — it is second-order fitting on 4 folds. The plateau rule
in (4) is the mitigation, not a cure. The winning cell's number is a selection
estimate; only the demo forward test is unconditioned evidence.

DEVIATION from `5_walkforward.py`, deliberate: the ILP here allows an account to
sit EMPTY (`require_full=False`) instead of forcing exactly one group per
account. At a 40% cap, forcing six accounts to be filled is simply infeasible,
which would blank out the whole low-cap half of the grid. Not trading an account
is always allowed in reality, so this makes the cap axis comparable end to end.
`accounts_used` is reported per cell so any thinning is visible.

OUT: data/3_results/wfa_grid_cells.csv    one row per (arm, RR, cap)
     data/3_results/wfa_grid_folds.csv    one row per (arm, RR, cap, fold)
     data/3_results/wfa_grid_breaches.csv breached account-folds only
     data/3_results/_provenance_step6.json
"""

import argparse
import importlib
import json
import os
import time

import numpy as np
import pandas as pd

import provenance as prov

wf = importlib.import_module("5_walkforward")   # module name starts with a digit

CAPS = [round(0.40 + 0.05 * i, 2) for i in range(10)]        # 0.40 .. 0.85
RRS = [round(1.00 + 0.10 * i, 2) for i in range(16)]         # 1.00 .. 2.50


_AVAIL = {}


def nearest_rr(strat, window, rr):
    """Snap a requested RR onto the swept grid for that window."""
    if not _AVAIL:
        for (s, w, r) in wf.DB:
            _AVAIL.setdefault((s, w), []).append(r)
    return min(_AVAIL[(strat, window)], key=lambda r: (abs(r - rr), r))


def fixed_picks(rr):
    """Every window at one global RR, minus passes MT5 liquidated.

    A tester-blown pass has a truncated export (its fatal trade never reached
    the file), so scoring it would invent a result. `5_walkforward` fails closed
    on those; here the window is simply unavailable at that RR, which is what a
    trader would face. Reported per cell as `blown_excluded`.
    """
    picks, excluded = {}, []
    for (s, w) in wf.WINDOWS:
        r = nearest_rr(s, w, rr)
        if wf.pass_is_blown(s, w, r):
            excluded.append(f"{s} {w}@{r:g}")
        else:
            picks[(s, w)] = r
    return picks, excluded


def ck(arm, rr, cap):
    """Hashable cell key. The `selected` arm has no RR, and NaN != NaN, so it
    can never be used as a dict key directly."""
    return (arm, None if rr is None or pd.isna(rr) else round(float(rr), 2),
            round(float(cap), 2))


def run_grid(arms, caps, rrs):
    """Score every (arm, RR, cap) cell across all folds."""
    fold_rows, cell_acct_rows, blown_note = [], {}, {}

    for fn, (fit_end, planned_test_end) in enumerate(wf.FOLDS, 1):
        test_end = min(planned_test_end, wf.DATA_END_EXCLUSIVE)
        if test_end <= fit_end:
            raise SystemExit(f"Fold {fn} has no test data after the source cutoff.")
        is_y, oos_y = (wf.years(wf.FIT_START, fit_end), wf.years(fit_end, test_end))

        builds = []                                  # (arm, rr_label, keys, groups)
        if "selected" in arms:
            t0 = time.time()
            picks = wf.select(wf.FIT_START, fit_end)
            wf.reject_known_blown_picks(picks)        # same fail-closed gate as step 5
            keys, groups = wf.build_groups(picks, wf.FIT_START, fit_end)
            builds.append(("selected", np.nan, keys, groups, len(picks)))
            print(f"  fold {fn} selected: {len(picks)} windows, {len(groups)} groups "
                  f"({time.time()-t0:,.0f}s)")
        if "fixed" in arms:
            for rr in rrs:
                t0 = time.time()
                picks, excluded = fixed_picks(rr)
                blown_note[round(float(rr), 2)] = excluded
                keys, groups = wf.build_groups(picks, wf.FIT_START, fit_end)
                builds.append(("fixed", rr, keys, groups, len(picks)))
                print(f"  fold {fn} fixed RR {rr:.2f}: {len(picks)} windows, "
                      f"{len(groups)} groups ({time.time()-t0:,.0f}s)")

        for arm, rr, keys, groups, n_picks in builds:
            for cap in caps:
                alloc = wf.solve_groups(keys, groups, cap, require_full=False)
                is_p = sum(g["profit"] for g, _ in alloc)
                oos_p, worst, accepted = wf.score_allocation(alloc, fit_end, test_end)
                cell_acct_rows.setdefault(ck(arm, rr, cap), []).extend(
                    (fn, acct, ex, net, mae, mfe) for ex, net, mae, mfe, acct in accepted)
                fold_rows.append({
                    "arm": arm, "RR": rr, "cap": cap, "fold": fn,
                    "test_start": fit_end.date(),
                    "test_end_inclusive": (test_end - pd.Timedelta(days=1)).date(),
                    "windows_available": n_picks,
                    "accounts_used": len(alloc),
                    "windows_traded": sum(len(g["keys"]) for g, _ in alloc),
                    "IS_net": round(is_p), "IS_net_per_yr": round(is_p / is_y),
                    "OOS_net": round(oos_p), "OOS_net_per_yr": round(oos_p / oos_y),
                    "OOS_worst_acct": worst[0],
                    "OOS_worst_pct_of_limit": round(worst[1] * 100, 1),
                    "alloc": "; ".join(
                        f"{a}:" + "+".join(f"{s} {w}@{r:g}" for s, w, r in g["keys"])
                        for g, a in sorted(alloc, key=lambda x: x[1])),
                })
        print(f"fold {fn} done ({fit_end.date()} -> "
              f"{(test_end - pd.Timedelta(days=1)).date()})\n")

    return pd.DataFrame(fold_rows), cell_acct_rows, blown_note


def summarise(FR, cell_acct_rows, blown_note):
    """Collapse folds into one row per cell, with prop-reality termination."""
    cells, breaches = [], []
    for (arm, rr, cap), grp in FR.groupby(["arm", "RR", "cap"], dropna=False):
        rows = cell_acct_rows[ck(arm, rr, cap)]
        B = wf.breach_table(rows) if rows else pd.DataFrame(
            columns=["fold", "acct", "breached", "post_breach_net"])
        post = int(B["post_breach_net"].sum()) if len(B) else 0
        n_breach = int(B["breached"].sum()) if len(B) else 0
        oos = int(grp["OOS_net"].sum())
        if n_breach:
            b = B[B["breached"]].copy()
            b.insert(0, "cap", cap); b.insert(0, "RR", rr); b.insert(0, "arm", arm)
            breaches.append(b)
        cells.append({
            "arm": arm, "RR": rr, "cap": cap,
            "folds_allocated": int((grp["accounts_used"] > 0).sum()),
            "accounts_used_min": int(grp["accounts_used"].min()),
            "accounts_used_max": int(grp["accounts_used"].max()),
            "windows_traded_med": float(grp["windows_traded"].median()),
            "IS_net": int(grp["IS_net"].sum()),
            "OOS_net": oos,
            "OOS_adj": oos - post,                   # termination-adjusted
            "post_breach_net": post,
            "breached_acct_folds": n_breach,
            "acct_folds": int(len(B)),
            "folds_OOS_positive": int((grp["OOS_net"] > 0).sum()),
            "WFE": round(grp["OOS_net_per_yr"].sum()
                         / max(grp["IS_net_per_yr"].sum(), 1e-9), 3),
            "worst_pct_of_limit": float(grp["OOS_worst_pct_of_limit"].max()),
            "worst_pct_median": float(grp["OOS_worst_pct_of_limit"].median()),
            "blown_excluded": len(blown_note.get(ck(arm, rr, 0)[1], [])),
        })
    C = pd.DataFrame(cells).sort_values(["arm", "RR", "cap"]).reset_index(drop=True)
    B = (pd.concat(breaches, ignore_index=True) if breaches
         else pd.DataFrame(columns=["arm", "RR", "cap", "fold", "acct", "breached"]))
    return C, B


def plateau_pick(C, n_folds):
    """Apply the pre-registered rule to the `fixed` arm.

    Survivors = allocated in every fold and zero breached account-folds. The
    reported winner is the highest-OOS_adj survivor; `plateau_size` counts its
    4-neighbourhood (one RR step, one cap step) that also survives, so an
    isolated peak is visible as such.
    """
    F = C[(C["arm"] == "fixed") & (C["folds_allocated"] == n_folds)].copy()
    surv = F[F["breached_acct_folds"] == 0].copy()
    if surv.empty:
        return None, surv, F
    ok = {(round(r, 2), round(c, 2)) for r, c in zip(surv["RR"], surv["cap"])}
    rr_step = round(RRS[1] - RRS[0], 2) if len(RRS) > 1 else 0.1
    cap_step = round(CAPS[1] - CAPS[0], 2) if len(CAPS) > 1 else 0.05
    surv["plateau_size"] = [
        sum(((round(r + dr, 2), round(c + dc, 2)) in ok)
            for dr, dc in ((rr_step, 0), (-rr_step, 0), (0, cap_step), (0, -cap_step)))
        for r, c in zip(surv["RR"], surv["cap"])]
    surv = surv.sort_values(["OOS_adj", "plateau_size"], ascending=False)
    return surv.iloc[0], surv, F


def grid_text(C, arm, value, fmt="{:.0f}"):
    """Small fixed-width RR x cap table for the console."""
    G = C[C["arm"] == arm].pivot(index="RR", columns="cap", values=value)
    head = "  RR \\ cap " + "".join(f"{c:>9.2f}" for c in G.columns)
    lines = [head, "  " + "-" * (len(head) - 2)]
    for rr, row in G.iterrows():
        lines.append(f"  {rr:>9.2f}" + "".join(
            ("{:>9}".format("-") if pd.isna(v) else "{:>9}".format(fmt.format(v)))
            for v in row))
    return "\n".join(lines)


_TPL = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Cap x RR walk-forward grid</title>
<script>__PLOTLYJS__</script>
<style>
 body{font:14px/1.45 system-ui,Segoe UI,Arial;margin:0;background:#f4f5f7;color:#1a1a1a}
 header{background:#1f2937;color:#fff;padding:14px 22px}h1{font-size:17px;margin:0}
 header p{margin:4px 0 0;font-size:12.5px;color:#cbd5e1}
 main{max-width:1240px;margin:16px auto;padding:0 16px}
 .panel{background:#fff;border-radius:10px;padding:12px 14px;margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
 table{border-collapse:collapse;width:100%;font-size:12.4px}
 th{text-align:right;padding:5px 8px;background:#f1f5f9;white-space:nowrap}
 th:first-child,td:first-child{text-align:left}
 td{padding:4px 8px;text-align:right;border-top:1px solid #eef0f3;white-space:nowrap}
 .bad{color:#b91c1c;font-weight:600}.ok{color:#15803d;font-weight:600}
 .note{font-size:12.3px;color:#6b7280;margin:6px 0}
 h2{font-size:15px;margin:16px 0 8px}
</style></head><body>
<header><h1>Cap &times; RR grid — every cell scored walk-forward</h1>
<p>Fit 2020&rarr;N, freeze, score the untouched next year, repeat. Nothing chosen in-sample.</p>
</header><main>
<div class="panel"><div id="h_breach" style="height:520px"></div>
 <div class="note">Breached account-folds: a real prop account is terminated the moment its
 trailing drawdown exceeds the limit. Green = survived every fold. This is the survival
 question and it is answered by the CAP axis, not the RR axis.</div></div>
<div class="panel"><div id="h_oos" style="height:520px"></div>
 <div class="note">Termination-adjusted out-of-sample profit: post-breach profit removed,
 because a terminated account never earns it. Read together with the panel above — a large
 number in a breaching cell is not money you could have kept.</div></div>
<div class="panel"><div id="h_worst" style="height:520px"></div>
 <div class="note">Worst account's OOS drawdown as % of its limit, max over the four folds.
 Above 100 the account is gone.</div></div>
<h2>Cap marginal — robustness across the whole RR axis</h2>
<div class="panel" style="overflow:auto" id="t_cap"></div>
<div class="note">"safe RRs" counts how many of the swept RR values survive every fold at
that cap. It measures how much you can be wrong about RR and still keep the accounts.</div>
<h2>Reference arm — per-window RR selection on the identical rig</h2>
<div class="panel" style="overflow:auto" id="t_sel"></div>
<div class="note" id="foot"></div>
</main><script>
const D=__DATA__,CFG={displaylogo:false,responsive:true};
const F={family:'system-ui,Segoe UI,Arial',size:11.5};
function heat(id,z,title,scale,rev,fmt){Plotly.newPlot(id,[{z:z,x:D.caps,y:D.rrs,
 type:'heatmap',colorscale:scale,reversescale:rev,
 hovertemplate:'cap %{x:.0%} · RR %{y:.2f}<br>'+fmt+'<extra></extra>',
 xgap:1,ygap:1}],
 {margin:{l:60,r:14,t:30,b:44},font:F,
  title:{text:title,x:0,font:{size:13}},
  xaxis:{title:'CAP_FRACTION',tickformat:'.0%',dtick:0.05},
  yaxis:{title:'fixed RR',dtick:0.1},
  plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);}
// zero breaches gets its own unmistakable colour: it is a different KIND of
// result, not one step better than one breach.
heat('h_breach',D.z_breach,'Breached account-folds (of 24) — green survived every fold',
     [[0,'#15803d'],[.001,'#bbf7d0'],[.25,'#fde68a'],[.6,'#f97316'],[1,'#7f1d1d']],
     false,'%{z} breached account-folds');
heat('h_oos',D.z_oos,'Termination-adjusted OOS profit $','Viridis',false,'$%{z:,.0f}');
heat('h_worst',D.z_worst,'Worst account OOS drawdown, % of limit','RdYlGn',false,'%{z:.0f}% of limit');
function tbl(id,rows,cols,hdr){document.getElementById(id).innerHTML=
 '<table><thead><tr>'+hdr.map(c=>'<th>'+c+'</th>').join('')+'</tr></thead><tbody>'+
 rows.map(r=>'<tr>'+cols.map(c=>{let v=r[c];if(v==null)v='';
  let cls='';
  if(c==='worst_pct_of_limit'||c==='worst_pct_med')cls=v>100?' class="bad"':'';
  if(c==='breached_acct_folds')cls=v>0?' class="bad"':' class="ok"';
  if(typeof v==='number')v=(Math.abs(v)>=1000?Math.round(v).toLocaleString():v);
  return `<td${cls}>${v}</td>`;}).join('')+'</tr>').join('')+'</tbody></table>';}
tbl('t_cap',D.capmarg,['cap','safe_rrs','breach_rate','OOS_adj_med','OOS_adj_max',
 'worst_pct_med','accounts'],
 ['cap','safe RRs (of '+D.rrs.length+')','breach rate','adj OOS median $',
  'adj OOS best $','worst % median','accounts used']);
tbl('t_sel',D.sel,['cap','OOS_net','OOS_adj','WFE','breached_acct_folds',
 'worst_pct_of_limit','accounts_used_min'],
 ['cap','OOS $','adjusted OOS $','WFE','breached acct-folds','worst % of limit','min accounts']);
document.getElementById('foot').textContent=
 `run ${D.prov.run} · code ${D.prov.git} · generated ${D.gen} · `+
 `${D.cells} cells · caution: picking a cell by its out-of-sample score is itself fitting; `+
 `prefer the middle of a wide surviving plateau over the single best cell.`;
</script></body></html>"""


def build_report(C):
    """Heatmap report from the saved cells table (no grid re-run needed)."""
    try:
        import plotly.offline as po
    except ImportError:
        print("(plotly missing - HTML skipped)")
        return
    FIX = C[C["arm"] == "fixed"]
    rrs = sorted(FIX["RR"].unique())
    caps = sorted(FIX["cap"].unique())

    def z(col):
        p = FIX.pivot(index="RR", columns="cap", values=col).reindex(
            index=rrs, columns=caps)
        return [[None if pd.isna(v) else float(v) for v in row] for row in p.values]

    capmarg = []
    for cap, g in FIX.groupby("cap"):
        rate = (g["breached_acct_folds"].sum()
                / max(g["acct_folds"].sum(), 1))
        capmarg.append({
            "cap": f"{cap:.0%}",
            "safe_rrs": int((g["breached_acct_folds"] == 0).sum()),
            "breach_rate": f"{rate:.1%}",
            "OOS_adj_med": int(g["OOS_adj"].median()),
            "OOS_adj_max": int(g["OOS_adj"].max()),
            "worst_pct_med": round(float(g["worst_pct_of_limit"].median()), 1),
            "accounts": f"{int(g['accounts_used_min'].min())}-"
                        f"{int(g['accounts_used_max'].max())}",
        })
    S = C[C["arm"] == "selected"].copy()
    S["cap"] = S["cap"].map(lambda c: f"{c:.0%}")
    payload = {
        "rrs": rrs, "caps": caps,
        "z_breach": z("breached_acct_folds"), "z_oos": z("OOS_adj"),
        "z_worst": z("worst_pct_of_limit"),
        "capmarg": capmarg, "sel": S.to_dict("records"), "cells": int(len(C)),
        "gen": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "prov": {"run": prov.RUN_ID, "git": prov.git_info().get("commit")},
    }
    os.makedirs("reports", exist_ok=True)
    html = (_TPL.replace("__PLOTLYJS__", po.get_plotlyjs())
            .replace("__DATA__", json.dumps(payload, separators=(",", ":"), default=str)))
    with open("reports/cap_rr_grid.html", "w", encoding="utf-8") as fh:
        fh.write(html)
    print("Saved reports/cap_rr_grid.html")


def deploy(rr, cap):
    """Fit the chosen cell on ALL available data and emit the live configuration.

    The walk-forward above validates the PROCEDURE; once it has, the allocation
    you actually trade should use every bar of history there is. Nothing here is
    scored — by construction there is no unseen data left to score it on.
    """
    wf.load_db()
    picks, excluded = fixed_picks(rr)
    keys, groups = wf.build_groups(picks, wf.FIT_START, wf.DATA_END_EXCLUSIVE)
    alloc = wf.solve_groups(keys, groups, cap, require_full=False)
    rows = []
    for g, acct in sorted(alloc, key=lambda x: x[1]):
        rows.append({
            "account": acct, "limit": wf.ACCOUNTS[acct],
            "cap_usd": round(cap * wf.ACCOUNTS[acct]),
            "windows": " + ".join(f"{s} {w}" for s, w, _ in g["keys"]),
            "RR": rr,
            "fit_net": round(g["profit"]),
            "fit_dd_upper": round(g["dd"]),
            "fit_dd_pct_of_limit": round(100 * g["dd"] / wf.ACCOUNTS[acct], 1),
        })
    D = pd.DataFrame(rows)
    os.makedirs("data/3_results", exist_ok=True)
    D.to_csv("data/3_results/forward_test_config.csv", index=False)
    print("\n" + "=" * 78)
    print(f"FORWARD-TEST CONFIGURATION — fixed RR {rr:g}, cap {cap:.0%}, "
          f"fit {wf.DATA_START.date()} -> "
          f"{(wf.DATA_END_EXCLUSIVE - pd.Timedelta(days=1)).date()} (ALL data)")
    print("=" * 78)
    if D.empty:
        print("  no group fits this cap — loosen it or accept fewer accounts")
    else:
        n_windows = int(D["windows"].str.count(r"\+").sum() + len(D))
        print(D.to_string(index=False))
        print(f"\n  {len(D)} accounts, {n_windows} windows, "
              f"fit-period net ${D['fit_net'].sum():,}")
    if excluded:
        print(f"  ({len(excluded)} window(s) unavailable at RR {rr:g} — tester-blown "
              f"export: {', '.join(excluded)})")
    print("\n  These are IN-SAMPLE fit figures: they size the risk, they do not predict.")
    print("  Saved data/3_results/forward_test_config.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--rr-step", type=float, default=0.10)
    ap.add_argument("--rr-range", type=float, nargs=2, default=(1.00, 2.50))
    ap.add_argument("--cap-step", type=float, default=0.05)
    ap.add_argument("--cap-range", type=float, nargs=2, default=(0.40, 0.85))
    ap.add_argument("--arms", nargs="+", default=["fixed", "selected"],
                    choices=["fixed", "selected"])
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild reports/cap_rr_grid.html from the saved cells CSV")
    ap.add_argument("--deploy", nargs=2, type=float, metavar=("RR", "CAP"),
                    help="skip the grid; fit this cell on ALL data and write the "
                         "live forward-test allocation")
    a = ap.parse_args()

    if a.report_only:
        build_report(pd.read_csv("data/3_results/wfa_grid_cells.csv"))
        return
    if a.deploy:
        deploy(round(a.deploy[0], 2), round(a.deploy[1], 2))
        return

    global RRS, CAPS
    RRS = [round(a.rr_range[0] + a.rr_step * i, 2)
           for i in range(int(round((a.rr_range[1] - a.rr_range[0]) / a.rr_step)) + 1)]
    CAPS = [round(a.cap_range[0] + a.cap_step * i, 2)
            for i in range(int(round((a.cap_range[1] - a.cap_range[0]) / a.cap_step)) + 1)]

    wf.load_db()
    print(f"\nGrid: arms={a.arms} | {len(RRS)} RRs {RRS[0]}..{RRS[-1]} | "
          f"{len(CAPS)} caps {CAPS[0]}..{CAPS[-1]} | {len(wf.FOLDS)} folds")
    print("Scoring is walk-forward throughout: fit -> freeze -> score unseen.\n")

    t0 = time.time()
    FR, cell_acct_rows, blown_note = run_grid(a.arms, CAPS, RRS)
    C, B = summarise(FR, cell_acct_rows, blown_note)
    print(f"grid complete in {time.time()-t0:,.0f}s\n")

    os.makedirs("data/3_results", exist_ok=True)
    FR.to_csv("data/3_results/wfa_grid_folds.csv", index=False)
    C.to_csv("data/3_results/wfa_grid_cells.csv", index=False)
    B.to_csv("data/3_results/wfa_grid_breaches.csv", index=False)

    if "fixed" in a.arms:
        print("=" * 78)
        print("BREACHED ACCOUNT-FOLDS (of "
              f"{int(C[C['arm'] == 'fixed']['acct_folds'].max())}) — survival first")
        print("=" * 78)
        print(grid_text(C, "fixed", "breached_acct_folds"))
        print("\n" + "=" * 78)
        print("TERMINATION-ADJUSTED OOS TOTAL $ (post-breach profit removed)")
        print("=" * 78)
        print(grid_text(C, "fixed", "OOS_adj"))
        print("\n" + "=" * 78)
        print("WORST ACCOUNT OOS DD, % OF ITS LIMIT (max over folds; >100 = blown)")
        print("=" * 78)
        print(grid_text(C, "fixed", "worst_pct_of_limit", "{:.0f}"))

        win, surv, F = plateau_pick(C, len(wf.FOLDS))
        print("\n" + "=" * 78)
        print("PRE-REGISTERED PICK")
        print("=" * 78)
        if win is None:
            print("  NO surviving cell: every (cap, RR) breached at least one "
                  "account-fold.\n  Safest cells by breach count:")
            print(F.nsmallest(5, ["breached_acct_folds"])[
                ["RR", "cap", "breached_acct_folds", "OOS_net", "OOS_adj",
                 "worst_pct_of_limit", "accounts_used_min"]].to_string(index=False))
        else:
            print(f"  cap {win['cap']:.2f}  RR {win['RR']:.2f}  |  "
                  f"OOS ${win['OOS_net']:,} (adj ${win['OOS_adj']:,})  "
                  f"WFE {win['WFE']}  worst {win['worst_pct_of_limit']:.0f}% of limit  "
                  f"accounts {win['accounts_used_min']}-{win['accounts_used_max']}  "
                  f"plateau {int(win['plateau_size'])}/4 neighbours")
            print(f"\n  {len(surv)} surviving cells; top 10 by adjusted OOS:")
            print(surv.head(10)[["RR", "cap", "OOS_net", "OOS_adj", "WFE",
                                 "worst_pct_of_limit", "accounts_used_min",
                                 "plateau_size"]].to_string(index=False))

    if "selected" in a.arms:
        S = C[C["arm"] == "selected"]
        print("\n" + "=" * 78)
        print("REFERENCE ARM — per-window RR selection, same rig, by cap")
        print("=" * 78)
        print(S[["cap", "OOS_net", "OOS_adj", "WFE", "breached_acct_folds",
                 "worst_pct_of_limit", "accounts_used_min"]].to_string(index=False))

    prov.write("data/3_results/_provenance_step6.json", prov.base(
        "6_cap_rr_grid",
        grid={"arms": a.arms, "caps": CAPS, "rrs": RRS,
              "folds": [str(f[0].date()) for f in wf.FOLDS],
              "cells": int(len(C))},
        source_data={"from": str(wf.DATA_START.date()),
                     "to_inclusive": str((wf.DATA_END_EXCLUSIVE
                                          - pd.Timedelta(days=1)).date()),
                     "windows": len(wf.WINDOWS), "passes": len(wf.DB)},
        settings={"MAX_PER_ACCOUNT": wf.MAX_PER_ACCOUNT, "accounts": wf.ACCOUNTS,
                  "require_full_allocation": False,
                  "DD_FOR_CAP_AND_OOS": "MFE-first upper bound"},
        limitations=["cell chosen on OOS score: second-order fitting on 4 folds",
                     "accounts may sit empty (require_full=False) unlike step 3",
                     "tester-blown passes excluded from the fixed arm's window pool",
                     "no per-fold MT5 calibration; conservative MFE-first DD bound",
                     "thresholds and window design remain historically fitted"]))
    build_report(C)
    print("\nSaved data/3_results/wfa_grid_cells.csv, wfa_grid_folds.csv, "
          "wfa_grid_breaches.csv")


if __name__ == "__main__":
    main()
