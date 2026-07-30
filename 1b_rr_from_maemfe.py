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
    INPUTS/data_2_maemfe_input/<STRAT>_sweeps/<window>/<window>_<RR>.csv
    e.g. INPUTS/data_2_maemfe_input/GG_sweeps/11-12/11-12_1.5.csv
    (UTF-16, tab, NO header: ticket, entry, exit, mae, mfe, profit, candle_range)

OUTPUT:
    OUTPUTS/results_outputs/rr_pertrade_recommendations.csv
    OUTPUTS/plots_outputs/step1b_rr_pertrade/<STRAT>_<window>.png
"""

import glob
import os
import re

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ---- CONFIG -----------------------------------------------------------------
SWEEP_ROOT = "INPUTS/data_2_maemfe_input"          # holds <STRAT>_sweeps/<window>/
PLOT_DIR = "OUTPUTS/plots_outputs/step1b_rr_pertrade"
OUT_CSV = "OUTPUTS/results_outputs/rr_pertrade_recommendations.csv"

COMMISSION_PER_RT = 1.0     # $ per round-turn (per-trade exports were run w/o costs)
MAX_DD_USD = 2000.0         # DD cap for the tier picks (a single window's budget)
MIN_RECOVERY = 2.0          # recommended pick below this profit/DD ratio = WEAK
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
    eq = np.cumsum(np.asarray(net, float))
    return float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0


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


def metrics(df):
    net = df["net"].values
    dd = (dd_equity(net, df["mae"].values, df["mfe"].values)
          if DD_MODE == "equity" else dd_closed(net))
    dd_capped = dd
    wins, losses = net[net > 0], net[net < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    tot = float(net.sum())
    return {
        "trades": len(df),
        "net_profit": round(tot),
        "maxDD": round(dd),
        "maxDD_capped": round(dd_capped),
        "recovery": round(tot / dd_capped, 2) if dd_capped > 0 else np.nan,
        "win%": round((net > 0).mean() * 100, 1),
        "PF": round(pf, 2) if np.isfinite(pf) else np.nan,
        "first": df["exit_time"].min().date(),
        "last": df["exit_time"].max().date(),
    }


# ---- PER-WINDOW SELECTION ---------------------------------------------------
def pick_tiers(tbl):
    """tbl: DataFrame per RR (sorted). Returns dict of tier -> row (or None)."""
    ok = tbl[tbl["net_profit"] > 0]
    under = ok[ok["maxDD_capped"] <= MAX_DD_USD]
    out = {}
    out["maxProfit"] = ok.loc[ok["net_profit"].idxmax()] if len(ok) else None
    out["recommended"] = under.loc[under["recovery"].idxmax()] if len(under) else None
    out["aggressive"] = under.loc[under["net_profit"].idxmax()] if len(under) else None
    out["unlocked"] = ok.loc[ok["recovery"].idxmax()] if len(ok) else None
    return out


def verdict(tiers):
    rc = tiers["recommended"]
    if tiers["maxProfit"] is None:
        return "LOSING"
    if rc is None:
        return "UNLOCK_ONLY"
    return "WEAK" if rc["recovery"] < MIN_RECOVERY else "OK"


def plot_window(tag, tbl, tiers):
    if plt is None:
        return None
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(tbl["RR"], tbl["net_profit"], "o-", color="tab:blue", label="net profit $")
    ax1.set_xlabel("RR")
    ax1.set_ylabel("net profit $", color="tab:blue")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(tbl["RR"], tbl["maxDD_capped"], "s--", color="tab:red",
             label=f"maxDD $ ({DD_MODE}%s)" % (f" x{DD_HAIRCUT}" if DD_HAIRCUT != 1 else ""))
    ax2.axhline(MAX_DD_USD, color="tab:red", ls=":", lw=1)
    ax2.set_ylabel("max drawdown $", color="tab:red")
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


# ---- MAIN -------------------------------------------------------------------
sweep_dirs = sorted(glob.glob(os.path.join(SWEEP_ROOT, "*_sweeps", "*")))
sweep_dirs = [d for d in sweep_dirs if os.path.isdir(d)]
if not sweep_dirs:
    raise SystemExit(
        f"No sweep folders found under {SWEEP_ROOT}/<STRAT>_sweeps/<window>/.\n"
        "Export per-trade files at a few RRs, e.g.\n"
        f"  {SWEEP_ROOT}/GG_sweeps/11-12/11-12_1.0.csv , _1.25.csv , _1.5.csv ...")

summary = []
for wdir in sweep_dirs:
    strat = os.path.basename(os.path.dirname(wdir)).replace("_sweeps", "")
    window = os.path.basename(wdir)
    rows = []
    for f in sorted(glob.glob(os.path.join(wdir, "*.csv"))):
        m = re.match(r"^.+_([0-9.]+)\.csv$", os.path.basename(f))
        if not m:
            continue
        df = load_file(f)
        if df is None or df.empty:
            print(f"  {strat} {window}: unreadable {os.path.basename(f)}")
            continue
        rows.append({"RR": float(m.group(1)), **metrics(df)})
    if not rows:
        continue
    tbl = pd.DataFrame(rows).sort_values("RR").reset_index(drop=True)
    tiers = pick_tiers(tbl)
    vd = verdict(tiers)
    png = plot_window(f"{strat} {window}", tbl, tiers)

    print("\n" + "=" * 92)
    print(f"{strat} {window}   verdict={vd}   ({tbl['trades'].iloc[0]} .. "
          f"{tbl['trades'].iloc[-1]} trades, {tbl['first'].iloc[0]}..{tbl['last'].iloc[0]})")
    print("=" * 92)
    show = tbl[["RR", "trades", "net_profit", "maxDD", "maxDD_capped", "recovery", "win%", "PF"]]
    print(show.to_string(index=False))
    for k in ["recommended", "aggressive", "unlocked", "maxProfit"]:
        r = tiers.get(k)
        if r is not None:
            print(f"   {k:<12} RR={r['RR']:<5g} net=${r['net_profit']:>7,.0f}  "
                  f"maxDD(capped)=${r['maxDD_capped']:>6,.0f}  recovery={r['recovery']}")
    row = {"strategy": strat, "window": window, "verdict": vd,
           "n_RRs_tested": len(tbl)}
    for k in ["recommended", "aggressive", "unlocked"]:
        r = tiers.get(k)
        if r is not None:
            row[f"{k}_RR"] = r["RR"]
            row[f"{k}_net"] = r["net_profit"]
            row[f"{k}_DD"] = r["maxDD_capped"]
    summary.append(row)

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
    print(f"\nDD_MODE={DD_MODE}, DD_HAIRCUT={DD_HAIRCUT} (inflates floating DD toward true "
          "equity DD), cap=${:,.0f}. Pick from real, current data — no XML.".format(MAX_DD_USD))
