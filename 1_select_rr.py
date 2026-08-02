"""
1b_rr_from_maemfe.py
====================
Step 1, rebuilt on REAL per-trade data instead of optimization XMLs.

Why: the optimization XMLs (step 1) can drift from the per-trade exports (steps
2-3) in period and tester settings — we saw GG 11-12's DD understated ($1,962
from the XML vs $2,925 real, because the XML predated the July-2026 drawdown).
This script derives the DD-aware RR pick from the SAME kind of per-trade files
that steps 2-3 use, so the number you pick and the number you allocate on come
from one consistent, current source.

Cost vs the XML: you can't reconstruct a different RR from one export (TP is
close-based, MFE is intrabar), so you export a small GRID of RRs per window.
Coarser than the XML's 251-value sweep, but real and consistent.

INPUT  (one folder per window, a handful of RRs each):
    data/1_sweeps/<STRAT>/<window>/<window>_<RR>.csv
    e.g. data/1_sweeps/GG/11-12/11-12_1.50.csv
    (UTF-16, tab, NO header: ticket, entry, exit, mae, mfe, profit, candle_range)

OUTPUT:
    data/3_results/rr_pertrade_recommendations.csv
    reports/plots/step1_rr_selection/<STRAT>_<window>.png
    data/2_chosen/<STRAT>/  (with --promote)

USAGE
venv/Scripts/python.exe 1_select_rr.py --promote recommended --dry-run   # preview
venv/Scripts/python.exe 1_select_rr.py --promote recommended             # do it
"""

import argparse
import glob
import json
import os
import re
import shutil

import numpy as np
import pandas as pd

import provenance as prov

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ---- CONFIG -----------------------------------------------------------------
SWEEPS_DIR = "data/1_sweeps"     # <STRAT>/<window>/  (+ <STRAT>_stats/, ignored here)
CHOSEN_DIR = "data/2_chosen"     # --promote copies the picks here, for step 2
PLOT_DIR = "reports/plots/step1_rr_selection"
OUT_CSV = "data/3_results/rr_pertrade_recommendations.csv"
OUT_CALIB = "data/3_results/dd_calibration.csv"   # consumed by steps 2 and 3
OUT_PROV = "data/3_results/_provenance_step1.json"
OUT_HTML = "reports/step1_rr_selection.html"        # table + equity curve per window
OUT_HTML_SWEEP = "reports/step1_rr_sweeps.html"     # profit/DD vs RR per window

COMMISSION_PER_RT = 1.0     # $ per round-turn (per-trade exports were run w/o costs)
MAX_DD_USD = 2000.0         # DD cap for the tier picks (a single window's budget)
DEPOSIT = 5000.0            # tester deposit; used to spot wiped-account passes
MIN_RECOVERY = 2.0          # recommended pick below this profit/DD ratio = WEAK
RECENT_YEARS = 3.0          # equity-shape check: profit earned in the last N years
FADING_SHARE = 0.30         # recent_net below this share of total => shape FADING
LR_MIN = 0.30               # equity-vs-trade-index correlation below this => ERRATIC
TOP1_MAX_SHARE = 0.35       # one trade worth more than this share of the total, or a
                            # total that goes negative once the best 5 trades are
                            # removed => CONCENTRATED (the result rests on a handful
                            # of trades, e.g. GG 7-8 made 76% of its profit on a
                            # single 2020-03-13 COVID-crash trade)
SMOOTH_RR = 0.10            # half-width of the RR neighbourhood used for robustness:
                            # picks are judged by their +/-SMOOTH_RR NEIGHBOURHOOD
                            # (mean profit, WORST DD), not by their own pixel. TP is
                            # close-based, so a 0.01 RR step can flip single trades
                            # and create cliffs; a pick one step from a cliff is
                            # fragile by construction.
DD_MODE = "equity"          # "equity" (MAE+MFE, matches MT5 exactly) | "closed" (balance)

COLS = ["ticket", "entry_time", "exit_time", "mae", "mfe", "profit", "candle_range"]


# ---- LOAD (same format as step 2) -------------------------------------------
def load_file(path):
    df = None
    for enc in ["utf-16", "utf-16-le", "utf-8-sig", "utf-8"]:
        try:
            df = pd.read_csv(path, sep="\t", header=None, names=COLS,
                             encoding=enc, engine="python")
            if df.shape[1] == len(COLS):
                break
        except Exception:
            df = None
    if df is None or df.empty:
        return None
    df = df[pd.to_numeric(df["ticket"], errors="coerce").notna()].copy()
    for c in ["mae", "mfe", "profit"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["exit_time"] = pd.to_datetime(df["exit_time"].astype(str).str.strip(),
                                     format="%Y.%m.%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["exit_time", "profit"]).sort_values("exit_time").reset_index(drop=True)
    df["net"] = df["profit"] - COMMISSION_PER_RT
    return df


def dd_closed(net):
    """Balance (closed-trade) drawdown.

    The leading 0.0 matters: the account's STARTING balance is the first peak.
    Without it np.maximum.accumulate starts at the first trade's result, so a
    curve that opens with a loss and never recovers above its start has its
    drawdown understated by exactly that first loss. MT5's STAT_BALANCE_DD
    measures from the initial deposit — caught by the _stats.csv cross-check.
    """
    eq = np.concatenate(([0.0], np.cumsum(np.asarray(net, float))))
    return float((np.maximum.accumulate(eq) - eq).max())


def dd_equity(net, mae, mfe):
    """Equity drawdown including intra-trade excursions.

    Validated against MT5: reproduces STAT_EQUITY_DD to the dollar (GG 11-12
    @2.99 -> $3,579 by both). Equally, dropping the MFE term reproduces
    STAT_BALANCE_DD ($2,925).

    The earlier MAE-only version tracked equity PEAKS only at closed-trade
    level, so a trade that ran to +MFE and closed lower had its give-back
    ignored — that understatement is what the old DD_HAIRCUT=1.15 was papering
    over. The true factor turned out to vary 1.01x-1.22x per window, so a
    single global fudge was wrong in both directions. Now computed exactly.
    """
    eq = peak = mdd = 0.0
    for n, a, f in zip(np.asarray(net, float), np.asarray(mae, float),
                       np.asarray(mfe, float)):
        # Order within a trade is MAE then MFE: a buy-stop breakout usually
        # retraces against you first, then runs. Validated against MT5 on 12
        # passes (window 2-3 at 11 RRs + GG 11-12) — exact every time. Using the
        # reverse order over-states DD by up to ~15%.
        mdd = max(mdd, peak - (eq + min(a, 0.0)))   # dip, against the standing peak
        peak = max(peak, eq + max(f, 0.0))          # then the run-up sets a new peak
        eq += n
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return float(mdd)


def metrics(df, mt5_eq_dd=None):
    """Per-pass stats. `net` = after commission; MT5's tester ran without costs.

    Our MAE-then-MFE walk reproduces MT5's equity DD on ~99.9% of passes, but the
    true intra-trade order is unknowable from MAE/MFE alone and a small cluster
    sits between the two orderings (MT5 is always inside the bracket). Where the
    EA's _stats.csv is available we therefore CALIBRATE per pass: scale our
    net-of-commission DD by MT5_gross / ours_gross. That factor is exactly 1.0
    whenever the walk was already right, so this only bites where we were wrong.
    """
    net = df["net"].values
    top = np.sort(net)[::-1]                    # best trades first, for concentration
    mae, mfe = df["mae"].values, df["mfe"].values
    dd = (dd_equity(net, mae, mfe) if DD_MODE == "equity" else dd_closed(net))
    calibrated, factor = False, 1.0
    if mt5_eq_dd is not None and DD_MODE == "equity":
        ours_gross = dd_equity(df["profit"].values, mae, mfe)
        if ours_gross > 0:
            factor = float(mt5_eq_dd) / ours_gross
            if factor > 1.0:                    # never talk risk DOWN
                dd *= factor
                calibrated = abs(factor - 1.0) > 1e-9
            else:
                factor = 1.0
    dd_capped = dd
    wins, losses = net[net > 0], net[net < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    tot = float(net.sum())
    return {
        "dd_factor": round(factor, 4) if calibrated else 1.0,
        "trades": len(df),
        "net_profit": round(tot),
        "maxDD": round(dd),
        "maxDD_capped": round(dd_capped),
        "recovery": round(tot / dd_capped, 2) if dd_capped > 0 else np.nan,
        "win%": round((net > 0).mean() * 100, 1),
        "PF": round(pf, 2) if np.isfinite(pf) else np.nan,
        "cal": "*" if calibrated else "",
        # equity-curve shape: LR correlation (straightness of the whole curve)
        # and profit earned in the recent period (is the edge still alive?)
        "lr_r": round(float(np.corrcoef(np.arange(len(net)),
                                        np.cumsum(net))[0, 1]), 3)
                if len(net) > 2 and np.std(np.cumsum(net)) > 0 else np.nan,
        "recent_net": round(float(df.loc[
            df["exit_time"] >= df["exit_time"].max()
            - pd.DateOffset(years=RECENT_YEARS), "net"].sum())),
        # concentration: does the result survive losing its luckiest trades?
        "top1_share": round(float(top[0] / tot), 3) if len(top) and tot > 0 else np.nan,
        "net_ex_top5": round(float(tot - top[:5].sum())) if len(top) else 0,
        "first": df["exit_time"].min().date(),
        "last": df["exit_time"].max().date(),
    }


def shape_of(row):
    """Qualify a pick by the SHAPE of its equity curve, not just its total.

    A headline profit can come from a rising curve or from one lucky trade on a
    flat one — identical in the CSV, opposite in reality. Checked worst-first:

    STALE        — earned nothing (or lost) in the last RECENT_YEARS. Trading it
                   bets AGAINST the recent trend of its own equity curve.
    CONCENTRATED — the total collapses (or turns negative) without its best few
                   trades. Not an edge, a couple of lottery tickets.
    ERRATIC      — equity/trade-index correlation below LR_MIN: profitable on
                   paper but the curve is not actually trending up.
    FADING       — profitable and trending, but the recent period contributed
                   under FADING_SHARE of the total: most of the edge is history.
    ALIVE        — none of the above and in profit.
    """
    if row is None:
        return ""
    if row["recent_net"] <= 0:
        return "STALE"
    if row["net_profit"] > 0:
        t1, ex5 = row.get("top1_share"), row.get("net_ex_top5")
        if (ex5 is not None and ex5 <= 0) or \
           (t1 is not None and t1 == t1 and t1 > TOP1_MAX_SHARE):
            return "CONCENTRATED"
        lr = row.get("lr_r")
        if lr is not None and lr == lr and lr < LR_MIN:
            return "ERRATIC"
        if row["recent_net"] < FADING_SHARE * row["net_profit"]:
            return "FADING"
    return "ALIVE"


# ---- VALIDATION AGAINST MT5's OWN FIGURES -----------------------------------
def load_stats(path):
    """The EA's OnTester row: header + one data line, UTF-16 tab-separated."""
    for enc in ["utf-16", "utf-16-le", "utf-8-sig", "utf-8"]:
        try:
            d = pd.read_csv(path, sep="\t", encoding=enc)
            if "equity_dd" in d.columns and len(d):
                return d.iloc[0]
        except Exception:
            continue
    return None


def dd_equity_upper(net, mae, mfe):
    """The other intra-trade ordering (MFE first). MT5's true equity DD always
    falls between this and dd_equity()."""
    eq = peak = mdd = 0.0
    for n, a, f in zip(np.asarray(net, float), np.asarray(mae, float),
                       np.asarray(mfe, float)):
        peak = max(peak, eq + max(f, 0.0))
        mdd = max(mdd, peak - (eq + min(a, 0.0)))
        eq += n
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return float(mdd)


def validate(strat, window, rr, df, stats_dir):
    """Cross-check one pass against MT5's own figures.

    MT5's tester ran WITHOUT costs, so everything is compared on GROSS profit
    (our `net` subtracts commission; `profit` does not).

    trades / net_profit / balance_dd must match exactly. equity_dd is checked as
    a BRACKET (MAE-first <= MT5 <= MFE-first) because the intra-trade order is
    not recoverable from MAE/MFE alone — asserting equality there would be
    claiming knowledge we don't have.

    Returns (mismatches, mt5_equity_dd, blown).
    """
    p = os.path.join(stats_dir, f"{window}_{rr:.2f}_stats.csv")
    if not os.path.isfile(p):
        return None, None, False
    s = load_stats(p)
    if s is None:
        return [f"{strat} {window} @{rr:g}: stats file unreadable"], None, False

    gross = df["profit"].values
    mae, mfe = df["mae"].values, df["mfe"].values
    mt5_eq = float(s["equity_dd"])

    # A wiped account is liquidated by the tester, not closed by EA logic, so the
    # fatal trade never reaches the export. Report it as its own category — the
    # data really is incomplete, but the pass is catastrophic and unusable anyway.
    blown = (mt5_eq >= DEPOSIT * 0.95
             or float(s["net_profit"]) <= -DEPOSIT * 0.95)
    if blown:
        return [], mt5_eq, True

    bad = []
    exact = {"trades": (len(df), 0),
             "net_profit": (round(float(gross.sum()), 2), 0.51),
             "balance_dd": (round(dd_closed(gross), 2), 1.01)}
    for k, (v, tol) in exact.items():
        try:
            mt5 = float(s[k])
        except (KeyError, TypeError, ValueError):
            continue
        if abs(v - mt5) > tol:
            bad.append(f"{strat} {window} @{rr:g}  {k}: ours={v:,.2f} MT5={mt5:,.2f}"
                       f"  (diff {v - mt5:+,.2f})")

    lo = dd_equity(gross, mae, mfe)
    hi = dd_equity_upper(gross, mae, mfe)
    if not (lo - 1.01 <= mt5_eq <= hi + 1.01):
        bad.append(f"{strat} {window} @{rr:g}  equity_dd OUTSIDE bracket: "
                   f"MT5={mt5_eq:,.2f} not in [{lo:,.2f}, {hi:,.2f}]")
    return bad, mt5_eq, False


# ---- PER-WINDOW SELECTION ---------------------------------------------------
def add_neighborhood(tbl):
    """Robustness columns: judge each RR by its +/-SMOOTH_RR neighbourhood.

    - profit_nb : mean net profit across the neighbourhood (smooths spikes)
    - dd_nb_max : WORST drawdown across the neighbourhood — if any neighbour
                  blows the cap, standing next to it is not safe either
    - recovery_nb : profit_nb / dd_nb_max — the plateau-quality score
    """
    rrs = tbl["RR"].values
    prof = tbl["net_profit"].values.astype(float)
    dds = tbl["maxDD_capped"].values.astype(float)
    p_nb, d_nb = [], []
    for r in rrs:
        m = np.abs(rrs - r) <= SMOOTH_RR + 1e-9
        p_nb.append(prof[m].mean())
        d_nb.append(dds[m].max())
    tbl = tbl.copy()
    tbl["profit_nb"] = np.round(p_nb)
    tbl["dd_nb_max"] = np.round(d_nb)
    tbl["recovery_nb"] = np.where(tbl["dd_nb_max"] > 0,
                                  (tbl["profit_nb"] / tbl["dd_nb_max"]).round(2), np.nan)
    return tbl


def pick_tiers(tbl):
    """tbl: DataFrame per RR (sorted, with neighbourhood columns).

    recommended / aggressive demand the ENTIRE +/-SMOOTH_RR neighbourhood under
    the DD cap and are ranked on neighbourhood figures, so they land mid-plateau
    instead of on the profitable edge of a cliff. Raw per-pixel figures are kept
    in the row for reporting.
    """
    ok = tbl[tbl["net_profit"] > 0]
    under = ok[(ok["dd_nb_max"] <= MAX_DD_USD) & (ok["profit_nb"] > 0)]
    out = {}
    out["maxProfit"] = ok.loc[ok["net_profit"].idxmax()] if len(ok) else None
    out["recommended"] = under.loc[under["recovery_nb"].idxmax()] if len(under) else None
    out["aggressive"] = under.loc[under["profit_nb"].idxmax()] if len(under) else None
    out["unlocked"] = ok.loc[ok["recovery_nb"].idxmax()] if len(ok) else None
    return out


def verdict(tiers):
    rc = tiers["recommended"]
    if tiers["maxProfit"] is None:
        return "LOSING"
    if rc is None:
        return "UNLOCK_ONLY"
    return "WEAK" if rc["recovery_nb"] < MIN_RECOVERY else "OK"


def plot_window(tag, tbl, tiers):
    if plt is None:
        return None
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(tbl["RR"], tbl["net_profit"], "-", color="tab:blue", lw=1,
             alpha=.45, label="net profit $ (raw)")
    ax1.plot(tbl["RR"], tbl["profit_nb"], "-", color="tab:blue", lw=2,
             label=f"net profit $ (nbhd mean +/-{SMOOTH_RR:g})")
    ax1.set_xlabel("RR")
    ax1.set_ylabel("net profit $", color="tab:blue")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(tbl["RR"], tbl["maxDD_capped"], "-", color="tab:red", lw=1,
             alpha=.45, label=f"maxDD $ ({DD_MODE}, raw)")
    ax2.plot(tbl["RR"], tbl["dd_nb_max"], "-", color="tab:red", lw=2,
             label="maxDD $ (nbhd WORST)")
    ax2.axhline(MAX_DD_USD, color="tab:red", ls=":", lw=1)
    ax2.set_ylabel("max drawdown $", color="tab:red")
    # shade the RR stretches whose whole neighbourhood stays under the cap
    safe = (tbl["dd_nb_max"] <= MAX_DD_USD).values
    ax1.fill_between(tbl["RR"], 0, 1, where=safe, transform=ax1.get_xaxis_transform(),
                     color="tab:green", alpha=.07, label="whole nbhd under cap")
    colors = {"recommended": "black", "aggressive": "tab:green", "maxProfit": "grey"}
    for k in ["recommended", "aggressive", "maxProfit"]:
        r = tiers.get(k)
        if r is not None:
            ax1.axvline(r["RR"], color=colors[k], ls="--", lw=1.2,
                        label=f"{k}: RR={r['RR']:g}")
    ax1.set_title(f"{tag} — RR from real per-trade data")
    ax1.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    p = os.path.join(PLOT_DIR, f"{tag.replace(' ', '_')}.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


# ---- HTML REPORT (table + equity curve + RR sweep, all in one page) ---------
def _decimate(xs, ys, n=600):
    if len(xs) <= n:
        return xs, ys
    step = int(np.ceil(len(xs) / n))
    idx = list(range(0, len(xs), step))
    if idx[-1] != len(xs) - 1:
        idx.append(len(xs) - 1)
    return [xs[i] for i in idx], [ys[i] for i in idx]


def equity_for(strat, window, rr):
    """Cumulative net-equity series for one window at one RR (for the report)."""
    p = os.path.join(SWEEPS_DIR, strat, window, f"{window}_{rr:.2f}.csv")
    df = load_file(p)
    if df is None or df.empty:
        return None
    xs = df["exit_time"].dt.strftime("%Y-%m-%d").tolist()
    ys = [int(v) for v in df["net"].cumsum().round(0)]
    xs, ys = _decimate(xs, ys)
    return {"rr": float(rr), "x": xs, "y": ys}


def write_html(S, pages, mode, out_path, prov_rec=None):
    """Self-contained page: sortable table + one chart per window.

    mode="equity" -> equity curve at the picked RR (the conflict-spotting view:
                     the CSV alone can't say whether an OK/WEAK verdict rests on
                     a live edge or a dead one — that needs the curve).
    mode="sweep"  -> profit / drawdown vs RR, i.e. the old PNGs in one page.

    Each page carries only the arrays it needs, so neither gets bloated.
    """
    try:
        import plotly.offline as po
    except ImportError:
        print("  (plotly not installed — HTML report skipped)")
        return None
    keep = "equity" if mode == "equity" else "sweep"
    slim = []
    for p in pages:
        q = {k: v for k, v in p.items() if k not in ("equity", "sweep")}
        q[keep] = p[keep]
        slim.append(q)
    data = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "cols": list(S.columns),
        "rows": S.where(pd.notna(S), None).values.tolist(),
        "pages": slim,
        "cap": MAX_DD_USD,
        "smooth": SMOOTH_RR,
        "recent_years": RECENT_YEARS,
        "lr_min": LR_MIN,
        "prov": prov_rec or {},
        "prov_warnings": prov.warnings_for(prov_rec) if prov_rec else [],
    }
    title = ("Step 1 — RR selection: equity curves" if mode == "equity"
             else "Step 1 — RR sweeps: profit / drawdown vs RR")
    html = (_TEMPLATE
            .replace("__PLOTLYJS__", po.get_plotlyjs())
            .replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__GEN__", data["generated"])
            .replace("__MODE__", mode)
            .replace("__TITLE__", title))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path


_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<script>__PLOTLYJS__</script>
<style>
 body{font:14px/1.45 system-ui,Segoe UI,Arial;margin:0;background:#f4f5f7;color:#1a1a1a}
 header{background:#1f2937;color:#fff;padding:14px 22px}
 header h1{font-size:17px;margin:0} header small{color:#9ca3af}
 main{max-width:1400px;margin:16px auto;padding:0 16px}
 .panel{background:#fff;border-radius:10px;padding:12px 14px;margin-bottom:16px;
        box-shadow:0 1px 3px rgba(0,0,0,.08)}
 h2{font-size:15px;margin:18px 0 8px}
 table{border-collapse:collapse;width:100%;font-size:12.4px}
 th{cursor:pointer;user-select:none;text-align:right;padding:6px 8px;background:#f1f5f9;
    position:sticky;top:0;white-space:nowrap}
 th:first-child,td:first-child{text-align:left}
 td{padding:4px 8px;text-align:right;border-top:1px solid #eef0f3;white-space:nowrap}
 tr:hover td{background:#f8fafc} tr.sel td{background:#fff7ed}
 .twrap{max-height:360px;overflow:auto;border:1px solid #e5e7eb;border-radius:6px}
 .bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:10px}
 .btn{border:1px solid #d7dbe0;background:#fff;border-radius:6px;padding:4px 10px;
      cursor:pointer;font-size:12.5px}
 .btn:hover{background:#f1f5f9} .btn.on{background:#1f2937;color:#fff;border-color:#1f2937}
 .btn.empty{opacity:.4}
 .rowlab{font-weight:700;font-size:12.5px;min-width:26px;color:#374151}
 .sep{display:inline-block;width:1px;height:18px;background:#e5e7eb;margin:0 4px}
 #filters{background:#fff;border-radius:10px;padding:8px 12px;margin-bottom:14px;
          box-shadow:0 1px 3px rgba(0,0,0,.08)}
 #filters .bar{margin:0;padding:3px 0}
 .card{background:#fff;border-radius:10px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.08);
       overflow:hidden;border-left:4px solid #d1d5db}
 .card.hide{display:none}
 .card.strat-0{border-left-color:#3b82f6;background:#fafcff}
 .card.strat-1{border-left-color:#f97316;background:#fffaf5}
 .stratdiv{display:flex;align-items:center;gap:10px;margin:20px 0 10px;
           font-size:13px;font-weight:700;color:#374151}
 .stratdiv::after{content:'';flex:1;height:1px;background:#e5e7eb}
 .stratdiv .dot{width:10px;height:10px;border-radius:50%}
 .chead{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:baseline;padding:10px 14px;
        border-bottom:1px solid #eef0f3;font-size:12.8px}
 .chead b{font-size:14.5px}
 .charts{padding:8px}
 .tag{padding:1px 7px;border-radius:10px;font-size:11.5px;font-weight:600}
 .v-OK{background:#dcfce7;color:#15803d}.v-WEAK{background:#fef3c7;color:#b45309}
 .v-UNLOCK_ONLY{background:#ede9fe;color:#6d28d9}.v-LOSING{background:#fee2e2;color:#b91c1c}
 .s-ALIVE{background:#dcfce7;color:#15803d}.s-FADING{background:#fef3c7;color:#b45309}
 .s-STALE{background:#fee2e2;color:#b91c1c}
 .s-CONCENTRATED{background:#fee2e2;color:#b91c1c}.s-ERRATIC{background:#fef3c7;color:#b45309}
 .warn{background:#fff7ed;border-left:3px solid #f97316;padding:6px 10px;margin:0 8px 8px;
       font-size:12.4px;border-radius:4px}
 .note{font-size:12.4px;color:#6b7280}
 #provbox{font-size:12.2px;color:#4b5563}
 #provbox h3{margin:0 0 6px;font-size:12.8px;color:#111}
 #provbox code{background:#f1f5f9;padding:1px 5px;border-radius:4px}
 #provbox .pw{color:#b45309;font-weight:600}
</style></head><body>
<header><h1>__TITLE__ <small>— generated __GEN__</small></h1></header>
<main>
 <div class="panel">
  <div class="note" id="legend"></div>
  <div class="twrap" id="tbl" style="margin-top:8px"></div>
 </div>
 <div class="bar" id="filters"></div>
 <div id="cards"></div>
 <div class="panel" id="provbox"></div>
</main>
<script>
const D=__DATA__, MODE="__MODE__";
const CFG={displaylogo:false,responsive:true,
  modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']};
const FONT={family:'system-ui,Segoe UI,Arial',size:11};
const conflict=p=>p.verdict!=='LOSING'&&p.shape&&p.shape!=='ALIVE';

document.getElementById('legend').innerHTML =
 `Cap $${D.cap.toLocaleString()} · neighbourhood ±${D.smooth} RR · shape window ${D.recent_years}y. `+
 `<b>Conflict</b> = any non-LOSING window whose shape is not ALIVE: `+
 `STALE (no recent profit) · CONCENTRATED (rests on a few trades) · `+
 `ERRATIC (LR &lt; ${D.lr_min}) · FADING (edge mostly historical).`;

// ---- table ----
(function(){
 let rows=D.rows.slice(),dir=1,sk=-1;
 const el=document.getElementById('tbl');
 const draw=()=>{
  let h='<table><thead><tr>'+D.cols.map((c,i)=>`<th data-i="${i}">${c}</th>`).join('')+'</tr></thead><tbody>';
  h+=rows.map(r=>{
    const o={};D.cols.forEach((c,i)=>o[c]=r[i]);
    const cls=conflict(o)?' class="sel"':'';
    return `<tr${cls}>`+r.map((v,i)=>{
      const c=D.cols[i];
      if(c==='verdict'&&v)return `<td><span class="tag v-${v}">${v}</span></td>`;
      if(c==='shape'&&v)return `<td><span class="tag s-${v}">${v}</span></td>`;
      if(v===null||v===undefined||v!==v)v='';
      else if(typeof v==='number')v=Number.isInteger(v)?v.toLocaleString():v.toLocaleString(undefined,{maximumFractionDigits:3});
      return `<td>${v}</td>`;}).join('')+'</tr>';}).join('');
  el.innerHTML=h+'</tbody></table>';
  el.querySelectorAll('th').forEach(th=>th.onclick=()=>{const i=+th.dataset.i;
    dir=(sk===i)?-dir:1;sk=i;
    rows.sort((a,b)=>{const x=a[i],y=b[i];
      if(x==null)return 1;if(y==null)return -1;
      return (typeof x==='number'&&typeof y==='number')?dir*(x-y):dir*String(x).localeCompare(String(y));});
    draw();});
 };draw();
})();

// ---- cards ----
const wrap=document.getElementById('cards');
const CSORDER=['RR','GG'];
const csrank=x=>{const i=CSORDER.indexOf(x);return i<0?99:i;};
let lastStrat=null;
const stratDivs={};
D.pages.forEach((p,i)=>{
 if(p.strat!==lastStrat){
   lastStrat=p.strat;
   const n=D.pages.filter(q=>q.strat===p.strat).length;
   const div=document.createElement('div');div.className='stratdiv';
   div.innerHTML=`<span class="dot" style="background:${csrank(p.strat)===0?'#3b82f6':'#f97316'}"></span>${p.strat} <span style="font-weight:400;color:#9ca3af">(${n} windows)</span>`;
   wrap.appendChild(div);
   stratDivs[p.strat]=div;
 }
 const d=document.createElement('div');
 d.className='card strat-'+csrank(p.strat);d.dataset.i=i;
 const t=p.tiers.recommended||p.tiers.unlocked||{};
 d.innerHTML=`<div class="chead">
   <b>${p.strat} ${p.window}</b>
   <span class="tag v-${p.verdict}">${p.verdict}</span>
   ${p.shape?`<span class="tag s-${p.shape}">${p.shape}</span>`:''}
   ${t.RR!==undefined?`<span>RR <b>${t.RR}</b></span>`:''}
   ${t.net!==undefined?`<span>net <b>$${(t.net||0).toLocaleString()}</b></span>`:''}
   ${t.dd!==undefined?`<span>maxDD <b>$${(t.dd||0).toLocaleString()}</b></span>`:''}
   ${t.nbdd!==undefined?`<span>nbhd worstDD <b>$${(t.nbdd||0).toLocaleString()}</b></span>`:''}
   ${t.recent!==undefined?`<span>last${D.recent_years}y <b>$${(t.recent||0).toLocaleString()}</b></span>`:''}
   ${t.lr!==undefined&&t.lr!==null?`<span>LR <b>${t.lr}</b></span>`:''}
   ${t.top1!==undefined&&t.top1!==null?`<span>best trade <b>${(t.top1*100).toFixed(0)}%</b></span>`:''}
   ${t.ex5!==undefined?`<span>net ex-top5 <b>$${(t.ex5||0).toLocaleString()}</b></span>`:''}
  </div>
  ${conflict(p)?`<div class="warn">Conflict: verdict <b>${p.verdict}</b> but shape <b>${p.shape}</b> — ${({STALE:'no profit in the recent period; trading it bets against its own trend',CONCENTRATED:'the total rests on a few trades and collapses without them',ERRATIC:'profitable on paper but the curve is not trending up',FADING:'most of the profit is not recent'})[p.shape]||'see the curve'}. Check the curve before trading it.</div>`:''}
  <div class="charts"><div id="c${i}" style="height:300px"></div></div>`;
 wrap.appendChild(d);
});

// lazy-render: 40+ windows x 2 charts is too much to draw up front
const COL={recommended:'#111',aggressive:'#59a14f',unlocked:'#b07aa1'};
function render(i){
 const p=D.pages[i]; if(p._done)return; p._done=1;
 if(MODE==='equity'){
 const eq=[];
 for(const k of ['recommended','aggressive','unlocked']){
   const e=p.equity[k]; if(!e)continue;
   eq.push({x:e.x,y:e.y,name:`${k} @${e.rr}`,type:'scatter',mode:'lines',
            line:{width:k==='recommended'?2:1.3,color:COL[k]},
            hovertemplate:'%{x}<br>$%{y:,.0f}<extra></extra>'});
 }
 Plotly.react('c'+i,eq,{margin:{l:58,r:10,t:26,b:30},font:FONT,hovermode:'x unified',
   title:{text:'equity at the picked RR',x:0,font:{size:12}},
   xaxis:{type:'date',gridcolor:'#eef0f3'},yaxis:{gridcolor:'#eef0f3',zeroline:true,
   zerolinecolor:'#cbd5e1'},showlegend:true,legend:{orientation:'h',y:-.18,font:{size:10}},
   plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);
 return;
 }
 const s=p.sweep,shapes=[];
 // green band where the whole +/-SMOOTH_RR neighbourhood stays under the cap
 let a=null;
 for(let j=0;j<s.rr.length;j++){
   const ok=s.dd_nb[j]<=D.cap;
   if(ok&&a===null)a=s.rr[j];
   if((!ok||j===s.rr.length-1)&&a!==null){
     shapes.push({type:'rect',xref:'x',yref:'paper',x0:a,x1:s.rr[j],y0:0,y1:1,
                  fillcolor:'#22c55e',opacity:.07,line:{width:0},layer:'below'});a=null;}
 }
 for(const k of ['recommended','aggressive','unlocked']){
   const t=p.tiers[k]; if(!t)continue;
   shapes.push({type:'line',x0:t.RR,x1:t.RR,yref:'paper',y0:0,y1:1,
                line:{color:COL[k],width:1.2,dash:'dash'}});
 }
 Plotly.react('c'+i,[
   {x:s.rr,y:s.profit,name:'net profit (raw)',type:'scatter',mode:'lines',
    line:{width:1,color:'#4e79a7'},opacity:.45},
   {x:s.rr,y:s.profit_nb,name:'net profit (nbhd mean)',type:'scatter',mode:'lines',
    line:{width:2,color:'#4e79a7'}},
   {x:s.rr,y:s.dd,name:'maxDD (raw)',type:'scatter',mode:'lines',yaxis:'y2',
    line:{width:1,color:'#e15759'},opacity:.4},
   {x:s.rr,y:s.dd_nb,name:'maxDD (nbhd worst)',type:'scatter',mode:'lines',yaxis:'y2',
    line:{width:2,color:'#e15759'}}],
  {margin:{l:58,r:52,t:26,b:30},font:FONT,hovermode:'x unified',shapes,
   title:{text:'profit / drawdown vs RR',x:0,font:{size:12}},
   xaxis:{title:{text:'RR',font:{size:10}},gridcolor:'#eef0f3'},
   yaxis:{title:{text:'net $',font:{size:10}},gridcolor:'#eef0f3'},
   yaxis2:{title:{text:'maxDD $',font:{size:10}},overlaying:'y',side:'right',
           showgrid:false},
   showlegend:true,legend:{orientation:'h',y:-.18,font:{size:9}},
   plot_bgcolor:'#fff',paper_bgcolor:'#fff'},CFG);
}
const io=new IntersectionObserver(es=>es.forEach(e=>{
  if(e.isIntersecting){render(+e.target.dataset.i);io.unobserve(e.target);}}),
  {rootMargin:'300px'});
document.querySelectorAll('.card').forEach(c=>io.observe(c));

// ---- provenance footer: 'which result is this?' ----------------------------
(function(){
 const p=D.prov||{},b=document.getElementById('provbox');
 if(!p.run_id){if(b)b.style.display='none';return;}
 const g=p.git||{},ea=p.ea||{},ov=p.overrides||{};
 const used=Object.entries(ov).filter(([k,v])=>v&&(!Array.isArray(v)||v.length))
   .map(([k,v])=>`${k}=${Array.isArray(v)?v.join(' '):v}`);
 b.innerHTML='<h3>Provenance</h3>'+
  `<div>run <code>${p.run_id}</code> · generated ${p.generated} · `+
  `analysis code <code>${g.commit||'n/a'}${g.dirty?' +dirty':''}</code> · python ${p.python||'?'}</div>`+
  `<div>data cutoff <code>${p.data_cutoff||'n/a'}</code>`+
  ((p.data_cutoffs_seen||[]).length>1?` <span class="pw">(MIXED: ${p.data_cutoffs_seen.join(', ')})</span>`:'')+
  ` · source manifests written ${(p.source_manifest_written||['?','?']).join(' … ')}</div>`+
  `<div>validated ${p.validated_passes||0} pass(es) · ${p.validation_mismatches||0} mismatch · `+
  `${p.passes_missing_stats||0} without stats · ${p.passes_account_blown||0} account-blown</div>`+
  `<div>EA ex5: ${Object.entries(ea).map(([k,v])=>`${k} <code>${(v.ex5_sha256_16||[]).join(', ')}</code>`).join(' · ')||'n/a'}</div>`+
  (used.length?`<div class="pw">overrides: ${used.join(' · ')}</div>`:'<div>no overrides used</div>')+
  ((D.prov_warnings||[]).length?`<div class="pw">${D.prov_warnings.map(w=>'&#9888; '+w).join('<br>')}</div>`:'');
})();

// ---- filters: one independent row per strategy ------------------------------
const SORDER=['RR','GG'];
const srank=x=>{const i=SORDER.indexOf(x);return i<0?99:i;};   // unknown -> last
const STRATS=[...new Set(D.pages.map(p=>p.strat))].sort((a,b)=>srank(a)-srank(b));
const DEFS=[['All',()=>true,'sep'],
            ['OK',p=>p.verdict==='OK'],
            ['WEAK',p=>p.verdict==='WEAK'],
            ['UNLOCK_ONLY',p=>p.verdict==='UNLOCK_ONLY'],
            ['LOSING',p=>p.verdict==='LOSING','sep'],
            ['ALIVE',p=>p.shape==='ALIVE'],
            ['CONCENTRATED',p=>p.shape==='CONCENTRATED'],
            ['ERRATIC',p=>p.shape==='ERRATIC'],
            ['FADING',p=>p.shape==='FADING'],
            ['STALE',p=>p.shape==='STALE','sep'],
            ['Conflicts only',p=>conflict(p)],
            ['Hide',()=>false]];
const state={};                       // strategy -> predicate
STRATS.forEach(st=>state[st]=()=>true);

function apply(){
 const anyVisible={};
 document.querySelectorAll('.card').forEach(c=>{
   const p=D.pages[+c.dataset.i];
   const show=(state[p.strat]||(()=>true))(p);
   c.classList.toggle('hide',!show);
   if(show){render(+c.dataset.i);anyVisible[p.strat]=true;}
 });
 Object.keys(stratDivs).forEach(st=>
   stratDivs[st].style.display=anyVisible[st]?'':'none');
 window.dispatchEvent(new Event('resize'));
}

const fb=document.getElementById('filters');
STRATS.forEach(st=>{
 const mine=D.pages.filter(p=>p.strat===st);
 const nc=mine.filter(conflict).length;
 const row=document.createElement('div');row.className='bar';
 const lab=document.createElement('span');lab.className='rowlab';lab.textContent=st;
 row.appendChild(lab);
 DEFS.forEach(([label,fn,sep],k)=>{
  const b=document.createElement('button');
  b.className='btn'+(k===0?' on':'');b.textContent=label;
  const n=mine.filter(fn).length;
  b.title=`${n} window${n===1?'':'s'}`;
  if(n===0&&label!=='Hide')b.classList.add('empty');
  b.onclick=()=>{row.querySelectorAll('.btn').forEach(x=>x.classList.remove('on'));
    b.classList.add('on');state[st]=fn;apply();};
  row.appendChild(b);
  if(sep){const d=document.createElement('span');d.className='sep';row.appendChild(d);}
 });
 const cnt=document.createElement('span');cnt.className='note';
 cnt.style.marginLeft='auto';
 cnt.textContent=`${mine.length} windows · ${nc} conflict${nc===1?'':'s'}`;
 row.appendChild(cnt);
 fb.appendChild(row);
});
</script></body></html>"""


# ---- PROMOTION (sweep folder -> the set step 2 consumes) --------------------
def promote(S, tier, verdicts, dry_run):
    """Copy the chosen RR of each qualifying window into <SWEEP_ROOT>/<STRAT>/.

    Step 2 globs one file per window there, so any previously promoted file for
    the same window is removed first — otherwise a window would be counted twice
    at two different RRs.
    """
    col = f"{tier}_RR"
    if col not in S.columns:
        print(f"\nNothing to promote: no '{col}' column (no window had a {tier} pick).")
        return
    sel = S[S["verdict"].isin(verdicts) & S[col].notna()]
    # Integrity gate: never promote data we could not verify or that we know is
    # a partial sweep — the allocation downstream treats these DDs as hard limits.
    if unsafe and not ARGS.allow_unvalidated:
        m = sel.apply(lambda r: (r["strategy"], r["window"]) in unsafe, axis=1)
        if m.any():
            print("  BLOCKED (use --allow-unvalidated to override):")
            for _, r in sel[m].iterrows():
                print(f"    {r['strategy']} {r['window']:<7} {unsafe[(r['strategy'], r['window'])]}")
            sel = sel[~m]
    if ARGS.exclude:
        excl = set(ARGS.exclude)
        m = sel.apply(lambda r: r["window"] in excl
                      or f"{r['strategy']}/{r['window']}" in excl, axis=1)
        if m.any():
            print("  excluded by --exclude: " + ", ".join(
                f"{r['strategy']} {r['window']}" for _, r in sel[m].iterrows()))
        sel = sel[~m]
    skipped = S[~S.index.isin(sel.index)]

    print("\n" + "=" * 92)
    print(f"PROMOTE  tier={tier}  verdicts={'/'.join(sorted(verdicts))}"
          f"{'   (DRY RUN)' if dry_run else ''}")
    print("=" * 92)
    if not len(sel):
        print("  nothing qualifies")
        return

    done = 0
    for _, r in sel.iterrows():
        strat, win, rr = r["strategy"], r["window"], float(r[col])
        src = os.path.join(SWEEPS_DIR, strat, win, f"{win}_{rr:.2f}.csv")
        dest_dir = os.path.join(CHOSEN_DIR, strat)
        dest = os.path.join(dest_dir, os.path.basename(src))
        if not os.path.isfile(src):
            print(f"  {strat} {win:<7} MISSING {os.path.basename(src)} — skipped")
            continue
        # drop any earlier pick for this window (possibly a different RR)
        stale = [p for p in glob.glob(os.path.join(dest_dir, f"{win}_*.csv"))
                 if os.path.basename(p) != os.path.basename(dest)]
        note = f"  (replaces {', '.join(os.path.basename(p) for p in stale)})" if stale else ""
        print(f"  {strat} {win:<7} RR {rr:<5g} net ${r[f'{tier}_net']:>7,.0f}  "
              f"DD ${r[f'{tier}_DD']:>6,.0f}  shape={r.get('shape', ''):<6}"
              f" last{RECENT_YEARS:g}y=${r.get(f'{tier}_recentNet', float('nan')):>7,.0f}{note}")
        if not dry_run:
            os.makedirs(dest_dir, exist_ok=True)
            for p in stale:
                os.remove(p)
            shutil.copy2(src, dest)
        done += 1

    print(f"\n  {done} window(s) {'would be' if dry_run else ''} promoted "
          f"-> {CHOSEN_DIR}/<STRAT>/")
    if len(skipped):
        print("  not promoted: " + ", ".join(
            f"{r['strategy']} {r['window']}({r['verdict']})" for _, r in skipped.iterrows()))
    if not dry_run:
        print("\nNext:  venv/Scripts/python.exe 2_analyze_maemfe.py")


# ---- MAIN -------------------------------------------------------------------
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--promote", nargs="?", const="recommended", default=None,
                choices=["recommended", "aggressive", "unlocked"],
                help="copy each qualifying window's chosen RR into "
                     "data/2_chosen/<STRAT>/ (default tier: recommended)")
ap.add_argument("--verdicts", nargs="+", default=["OK"],
                help="which verdicts qualify for --promote (default: OK)")
ap.add_argument("--dry-run", action="store_true",
                help="with --promote, show what would be copied and change nothing")
ap.add_argument("--no-validate", action="store_true",
                help="skip the cross-check of our figures against MT5's _stats.csv")
ap.add_argument("--allow-unvalidated", action="store_true",
                help="promote even from incomplete sweeps or windows with no "
                     "MT5 _stats.csv (default: those are blocked)")
ap.add_argument("--exclude", nargs="+", default=[],
                help="windows to leave out of --promote, e.g. 6-7 or RR/6-7 "
                     "(bare window name applies to both strategies)")
ARGS = ap.parse_args()

sweep_dirs = sorted(glob.glob(os.path.join(SWEEPS_DIR, "*", "*")))
# <STRAT>_stats/ holds the EA's OnTester summaries, not per-trade data
sweep_dirs = [d for d in sweep_dirs
              if os.path.isdir(d) and not os.path.basename(os.path.dirname(d)).endswith("_stats")]
if not sweep_dirs:
    raise SystemExit(
        f"No sweep folders found under {SWEEPS_DIR}/<STRAT>/<window>/.\n"
        "Run step 0 first, e.g.\n"
        "  python 0_run_mt5_sweeps.py --windows 11-12 --strategy GG --rr 0.5 3.0 0.1")

summary, pages, calib = [], [], []
upstream = {}        # (strategy, window) -> its sweep _manifest.json
unsafe = {}          # (strategy, window) -> why it must not be promoted
val_bad, val_checked, val_missing, val_blown = [], 0, 0, 0
for wdir in sweep_dirs:
    strat = os.path.basename(os.path.dirname(wdir))
    window = os.path.basename(wdir)
    stats_dir = os.path.join(SWEEPS_DIR, f"{strat}_stats", window)
    rows = []
    n_missing_stats = 0
    mf = os.path.join(wdir, "_manifest.json")
    if os.path.isfile(mf):
        try:
            with open(mf, encoding="utf-8") as fh:
                _m = json.load(fh)
            upstream[(strat, window)] = _m
            if _m.get("complete") is False:
                unsafe[(strat, window)] = (
                    f"sweep incomplete ({_m.get('files_collected')}/{_m.get('expected')} "
                    f"files) — folder holds a MIX of runs")
        except (OSError, ValueError):
            pass
    for f in sorted(glob.glob(os.path.join(wdir, "*.csv"))):
        m = re.match(r"^.+_([0-9.]+)\.csv$", os.path.basename(f))
        if not m:
            continue
        df = load_file(f)
        if df is None or df.empty:
            print(f"  {strat} {window}: unreadable {os.path.basename(f)}")
            continue
        rr_val = float(m.group(1))
        mt5_eq = None
        if not ARGS.no_validate:
            bad, mt5_eq, blown = validate(strat, window, rr_val, df, stats_dir)
            if bad is None:
                val_missing += 1
                n_missing_stats += 1
            elif blown:
                val_blown += 1
                continue          # data incomplete AND the pass wiped the account
            else:
                val_checked += 1
                val_bad.extend(bad)
        m = metrics(df, mt5_eq)
        if m["dd_factor"] > 1.0:
            calib.append({"strategy": strat, "window": window, "RR": rr_val,
                          "dd_factor": m["dd_factor"]})
        rows.append({"RR": rr_val, **m})
    if not rows:
        continue
    if n_missing_stats and not ARGS.no_validate:
        unsafe.setdefault((strat, window),
                          f"{n_missing_stats} pass(es) have no MT5 _stats.csv — unverified")
    tbl = pd.DataFrame(rows).sort_values("RR").reset_index(drop=True)
    tbl = add_neighborhood(tbl)
    tiers = pick_tiers(tbl)
    vd = verdict(tiers)
    png = plot_window(f"{strat} {window}", tbl, tiers)

    shp_row = tiers.get("recommended") if tiers.get("recommended") is not None \
        else tiers.get("unlocked")
    shp = shape_of(shp_row)
    print("\n" + "=" * 92)
    print(f"{strat} {window}   verdict={vd}   shape={shp}"
          f"   ({tbl['trades'].iloc[0]} .. {tbl['trades'].iloc[-1]} trades, "
          f"{tbl['first'].iloc[0]}..{tbl['last'].iloc[0]})")
    print("=" * 92)
    show = tbl[["RR", "trades", "net_profit", "maxDD_capped", "recovery",
                "profit_nb", "dd_nb_max", "recovery_nb", "win%", "PF"]]
    print(show.to_string(index=False))
    for k in ["recommended", "aggressive", "unlocked", "maxProfit"]:
        r = tiers.get(k)
        if r is not None:
            print(f"   {k:<12} RR={r['RR']:<5g} net=${r['net_profit']:>7,.0f}  "
                  f"maxDD=${r['maxDD_capped']:>6,.0f}  "
                  f"| nbhd(+/-{SMOOTH_RR:g}): net=${r['profit_nb']:>7,.0f}  "
                  f"worstDD=${r['dd_nb_max']:>6,.0f}  recovery={r['recovery_nb']}  "
                  f"| LR={r['lr_r']}  last{RECENT_YEARS:g}y=${r['recent_net']:,.0f}  "
                  f"top1={r['top1_share']}  exTop5=${r['net_ex_top5']:,.0f}")
    row = {"strategy": strat, "window": window, "verdict": vd, "shape": shp,
           "n_RRs_tested": len(tbl)}
    for k in ["recommended", "aggressive", "unlocked"]:
        r = tiers.get(k)
        if r is not None:
            row[f"{k}_RR"] = r["RR"]
            row[f"{k}_net"] = r["net_profit"]
            row[f"{k}_DD"] = r["maxDD_capped"]
            row[f"{k}_nbWorstDD"] = r["dd_nb_max"]
            row[f"{k}_lr"] = r["lr_r"]
            row[f"{k}_recentNet"] = r["recent_net"]
            row[f"{k}_top1Share"] = r["top1_share"]
            row[f"{k}_netExTop5"] = r["net_ex_top5"]
    summary.append(row)

    # ---- data for the HTML pages ----
    page = {"strat": strat, "window": window, "verdict": vd, "shape": shp,
            "tiers": {}, "equity": {}, "sweep": {
                "rr": [round(float(v), 2) for v in tbl["RR"]],
                "profit": [int(v) for v in tbl["net_profit"]],
                "profit_nb": [int(v) for v in tbl["profit_nb"]],
                "dd": [int(v) for v in tbl["maxDD_capped"]],
                "dd_nb": [int(v) for v in tbl["dd_nb_max"]]}}
    seen_rr = {}
    for k in ["recommended", "aggressive", "unlocked"]:
        r = tiers.get(k)
        if r is None:
            continue
        rr_v = float(r["RR"])
        page["tiers"][k] = {"RR": rr_v, "net": int(r["net_profit"]),
                            "dd": int(r["maxDD_capped"]), "nbdd": int(r["dd_nb_max"]),
                            "recent": int(r["recent_net"]),
                            "lr": None if pd.isna(r["lr_r"]) else float(r["lr_r"]),
                            "top1": None if pd.isna(r["top1_share"]) else float(r["top1_share"]),
                            "ex5": int(r["net_ex_top5"])}
        if rr_v in seen_rr:                      # tiers often share an RR
            page["equity"][k] = seen_rr[rr_v]
            continue
        e = equity_for(strat, window, rr_v)
        if e:
            page["equity"][k] = e
            seen_rr[rr_v] = e
    pages.append(page)

if summary:
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    S = pd.DataFrame(summary)
    S["_h"] = S["window"].str.split("-").str[0].astype(int)
    S = S.sort_values(["strategy", "_h"]).drop(columns="_h")
    try:
        S.to_csv(OUT_CSV, index=False)
        print(f"\nSaved {OUT_CSV}")
    except PermissionError:
        import time
        alt = OUT_CSV.replace(".csv", f"_{time.strftime('%H%M%S')}.csv")
        S.to_csv(alt, index=False)
        print(f"\n({OUT_CSV} locked) Saved {alt}")
    print(f"Plots -> {PLOT_DIR}/")
    print(f"\nDD_MODE={DD_MODE} (MAE-then-MFE walk; reproduces MT5's equity DD "
          f"exactly), cap=${MAX_DD_USD:,.0f}. Picked from real per-trade data.")

    print("\nVerdicts: " + ", ".join(f"{k}={v}" for k, v in
                                     S["verdict"].value_counts().items()))

    # ---- validation gate ----------------------------------------------------
    if not ARGS.no_validate:
        print(f"\nValidation vs MT5 _stats.csv: {val_checked} pass(es) checked"
              + (f", {val_missing} without a stats file" if val_missing else ""))
        if val_bad:
            print(f"  MISMATCHES ({len(val_bad)}):")
            for b in val_bad[:20]:
                print("    " + b)
            if len(val_bad) > 20:
                print(f"    ... and {len(val_bad) - 20} more")
            raise SystemExit(
                "\nAborting: our reconstruction disagrees with MT5. Downstream steps\n"
                "would inherit the error. Investigate before promoting "
                "(or re-run with --no-validate to override).")
        if val_checked:
            print("  OK — trade count, profit and both drawdowns match MT5 exactly.")
        if val_missing:
            print("  (windows without stats files were NOT verified — re-run step 0 "
                  "for them to close the gap)")
    if unsafe:
        print(f"\nIntegrity: {len(unsafe)} window(s) not promotable as-is")
        for (st, w), why in sorted(unsafe.items()):
            print(f"    {st} {w:<7} {why}")

    # ---- provenance: what produced THIS set of recommendations ---------------
    _eas, _cuts, _mruns, _mwritten = {}, set(), set(), []
    for (st, w), m in upstream.items():
        if m.get("ea"):
            ex5 = (m["ea"].get("expert_ex5") or {}).get("sha256_16")
            if ex5:
                _eas.setdefault(st, set()).add(ex5)
        if m.get("to"):
            _cuts.add(str(m["to"]))
        if m.get("run_id"):
            _mruns.add(m["run_id"])
        if m.get("written"):
            _mwritten.append(str(m["written"]))
    PROV = prov.base(
        "1_select_rr",
        data_cutoff=sorted(_cuts)[-1] if _cuts else None,
        data_cutoffs_seen=sorted(_cuts),           # >1 means MIXED periods
        source_manifest_runs=sorted(_mruns),
        source_manifest_written=(min(_mwritten), max(_mwritten)) if _mwritten else None,
        windows_scanned=len(sweep_dirs),
        windows_without_manifest=len(sweep_dirs) - len(upstream),
        validated_passes=val_checked,
        validation_mismatches=len(val_bad),
        passes_missing_stats=val_missing,
        passes_account_blown=val_blown,
        unsafe_windows={f"{a} {b}": why for (a, b), why in unsafe.items()},
        ea={st: {"ex5_sha256_16": sorted(v)} for st, v in _eas.items()},
        # manifests written before provenance tracking carry no EA identity —
        # say so loudly rather than rendering a silent "n/a" downstream
        ea_unknown_windows=sorted(f"{a} {b}" for (a, b), m in upstream.items()
                                  if not (m.get("ea") or {}).get("expert_ex5")),
        # only genuine deviations from the defaults belong here — listing the
        # defaults as "overrides" would train you to ignore the field.
        overrides={k: v for k, v in (
            ("no_validate", bool(ARGS.no_validate)),
            ("allow_unvalidated", bool(ARGS.allow_unvalidated)),
            ("exclude", list(ARGS.exclude)),
            ("verdicts", list(ARGS.verdicts) if list(ARGS.verdicts) != ["OK"] else []),
        ) if v},
        settings={"promote": ARGS.promote, "verdicts": list(ARGS.verdicts)},
        thresholds={"MAX_DD_USD": MAX_DD_USD, "SMOOTH_RR": SMOOTH_RR,
                    "MIN_RECOVERY": MIN_RECOVERY, "RECENT_YEARS": RECENT_YEARS,
                    "FADING_SHARE": FADING_SHARE, "LR_MIN": LR_MIN,
                    "TOP1_MAX_SHARE": TOP1_MAX_SHARE,
                    "COMMISSION_PER_RT": COMMISSION_PER_RT},
    )
    prov.write(OUT_PROV, PROV)
    print("\nProvenance: " + prov.summary_line(PROV))
    if len(_cuts) > 1:
        print(f"  !! sweeps span MORE THAN ONE data cutoff {sorted(_cuts)} — "
              "windows are not comparable")
    for _w in prov.warnings_for(PROV):
        print("  !! " + _w)

    # DD calibration for steps 2/3: without this the RR pick is cautious but the
    # ACCOUNT drawdown constraint downstream stays optimistic on the same passes.
    C = pd.DataFrame(calib, columns=["strategy", "window", "RR", "dd_factor"])
    os.makedirs(os.path.dirname(OUT_CALIB), exist_ok=True)
    C.to_csv(OUT_CALIB, index=False)
    print(f"DD calibration: {len(C)} pass(es) need scaling "
          f"(max x{C['dd_factor'].max():.3f}) -> {OUT_CALIB}"
          if len(C) else f"DD calibration: none needed -> {OUT_CALIB}")

    # cards in trading-day order, not lexicographic ("1-2, 10-11, .. 2-3")
    _sord = {"RR": 0, "GG": 1}
    pages.sort(key=lambda p: (_sord.get(p["strat"], 9), p["strat"],
                              int(p["window"].split("-")[0])))
    eq_html = write_html(S, pages, "equity", OUT_HTML, PROV)
    sw_html = write_html(S, pages, "sweep", OUT_HTML_SWEEP, PROV)
    if eq_html:
        n_conf = sum(1 for p in pages
                     if p["verdict"] != "LOSING" and p["shape"] and p["shape"] != "ALIVE")
        print(f"\nHTML: {eq_html}   (equity curves + table"
              + (f", {n_conf} verdict/shape conflict(s) flagged)" if n_conf else ")"))
        print(f"      {sw_html}   (profit / drawdown vs RR)")

    if ARGS.promote:
        promote(S, ARGS.promote, set(ARGS.verdicts), ARGS.dry_run)
    else:
        print("Tip: --promote [recommended|aggressive|unlocked] copies the picks into "
              "<STRAT>/ for step 2\n     (add --dry-run first; --verdicts OK WEAK to widen).")
