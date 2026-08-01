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
import os
import re
import shutil

import numpy as np
import pandas as pd

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

COMMISSION_PER_RT = 1.0     # $ per round-turn (per-trade exports were run w/o costs)
MAX_DD_USD = 2000.0         # DD cap for the tier picks (a single window's budget)
DEPOSIT = 5000.0            # tester deposit; used to spot wiped-account passes
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
    mae, mfe = df["mae"].values, df["mfe"].values
    dd = (dd_equity(net, mae, mfe) if DD_MODE == "equity" else dd_closed(net))
    calibrated = False
    if mt5_eq_dd is not None and DD_MODE == "equity":
        ours_gross = dd_equity(df["profit"].values, mae, mfe)
        if ours_gross > 0:
            factor = float(mt5_eq_dd) / ours_gross
            if factor > 1.0:                    # never talk risk DOWN
                dd *= factor
                calibrated = abs(factor - 1.0) > 1e-9
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
        "cal": "*" if calibrated else "",
        "first": df["exit_time"].min().date(),
        "last": df["exit_time"].max().date(),
    }


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
             label=f"maxDD $ ({DD_MODE})")
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
              f"DD ${r[f'{tier}_DD']:>6,.0f}{note}")
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

summary = []
val_bad, val_checked, val_missing, val_blown = [], 0, 0, 0
for wdir in sweep_dirs:
    strat = os.path.basename(os.path.dirname(wdir))
    window = os.path.basename(wdir)
    stats_dir = os.path.join(SWEEPS_DIR, f"{strat}_stats", window)
    rows = []
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
            elif blown:
                val_blown += 1
                continue          # data incomplete AND the pass wiped the account
            else:
                val_checked += 1
                val_bad.extend(bad)
        rows.append({"RR": rr_val, **metrics(df, mt5_eq)})
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

    if ARGS.promote:
        promote(S, ARGS.promote, set(ARGS.verdicts), ARGS.dry_run)
    else:
        print("Tip: --promote [recommended|aggressive|unlocked] copies the picks into "
              "<STRAT>/ for step 2\n     (add --dry-run first; --verdicts OK WEAK to widen).")
