"""
analyze_maemfe.py
=================
Takes the per-window MT5 trade exports in MAEMFE/ (one file per chosen
window+RR, e.g. "2-3_1.14.csv") and answers the question the optimization XMLs
never could:

    "If I traded ALL these windows, what do I actually get — and what does the
     equity curve look like?"

The optimization tables only had summary stats per RR. These files have every
trade with timestamps, so we can build REAL equity curves, real drawdowns, and
a real combined portfolio result.

INPUT : MAEMFE/<from>-<till>_<RR>.csv
        UTF-16, tab-separated, NO header, columns in EA write order:
        ticket, entry_time, exit_time, mae_money, mfe_money, trade_profit, candle_range

OUTPUT: output_files/maemfe_window_summary.csv   per-window stats (net of commission)
        output_files/maemfe_combined_trades.csv  all trades + running equity
        plots/maemfe/combined_equity.png         portfolio equity + drawdown
        plots/maemfe/per_window_equity.png       small multiples per window

NOTE ON COMBINING: each file was backtested with only that window enabled, so
summing them models your MULTI-ACCOUNT setup (windows run independently, no
competition for a position slot). A SINGLE account running all windows would
see blocking (trades routinely run past their own hour) and do worse.
"""

import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# ── CONFIG ────────────────────────────────────────────────────────────────────
MAEMFE_DIR = "MAEMFE"
PLOT_DIR = "plots/maemfe"
OUT_SUMMARY = "output_files/maemfe_window_summary.csv"
OUT_TRADES = "output_files/maemfe_combined_trades.csv"
COMMISSION_PER_RT = 1.0   # $ per round-turn
COLS = ["ticket", "entry_time", "exit_time", "mae", "mfe", "profit", "candle_range"]


def load_file(path):
    """UTF-16 tab-separated, header-less MT5 export."""
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
    # drop a header row if one happens to be present
    df = df[pd.to_numeric(df["ticket"], errors="coerce").notna()].copy()
    for c in ["mae", "mfe", "profit", "candle_range"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["entry_time", "exit_time"]:
        df[c] = pd.to_datetime(df[c].astype(str).str.strip(),
                               format="%Y.%m.%d %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["exit_time", "profit"]).reset_index(drop=True)
    df["net"] = df["profit"] - COMMISSION_PER_RT
    return df


def dd_stats(net_series):
    """Max drawdown ($) and its recovery factor from a series of trade PnLs."""
    eq = np.cumsum(np.asarray(net_series, dtype=float))
    if not len(eq):
        return 0.0, 0.0, np.array([])
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    maxdd = float(dd.max())
    total = float(eq[-1])
    rec = total / maxdd if maxdd > 0 else np.inf
    return maxdd, rec, dd


def dd_floating(net_series, mae_series):
    """Max drawdown including OPEN floating P/L, using each trade's MAE.

    Closed-trade DD understates what a prop firm measures: during a trade the
    equity dips to (equity_before + MAE) before the trade closes. This walks the
    trades and tracks the worst peak-to-trough including those intra-trade dips.
    Assumes one position at a time (true per window; approximate when combined).
    """
    eq = peak = maxdd = 0.0
    for n, m in zip(np.asarray(net_series, float), np.asarray(mae_series, float)):
        trough = eq + min(m, 0.0)              # worst point while the trade is open
        maxdd = max(maxdd, peak - trough)
        eq += n
        peak = max(peak, eq)
        maxdd = max(maxdd, peak - eq)
    return float(maxdd)


def summarise(df, label, rr=None):
    net = df["net"].values
    maxdd, rec, _ = dd_stats(net)
    maxdd_f = dd_floating(net, df["mae"].values)
    wins, losses = net[net > 0], net[net < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    # max consecutive losses
    mcl = cur = 0
    for x in net:
        cur = cur + 1 if x < 0 else 0
        mcl = max(mcl, cur)
    return {
        "window": label, "RR": rr, "trades": len(df),
        "gross_profit": round(df["profit"].sum()),
        "commission": round(len(df) * COMMISSION_PER_RT),
        "net_profit": round(net.sum()),
        "maxDD$": round(maxdd),
        "maxDD_float$": round(maxdd_f),
        "recovery": round(rec, 2) if np.isfinite(rec) else None,
        "recovery_float": round(net.sum() / maxdd_f, 2) if maxdd_f > 0 else None,
        "win%": round((net > 0).mean() * 100, 1),
        "PF": round(pf, 2) if np.isfinite(pf) else None,
        "avg_trade": round(net.mean(), 2),
        "maxLossStreak": mcl,
        "first": df["exit_time"].min().date(),
        "last": df["exit_time"].max().date(),
    }


# ── LOAD ──────────────────────────────────────────────────────────────────────
paths = sorted(glob.glob(os.path.join(MAEMFE_DIR, "*.csv")))
if not paths:
    raise SystemExit(f"No CSVs in {MAEMFE_DIR}/")

rows, frames = [], []
print(f"Loading {len(paths)} window files from {MAEMFE_DIR}/ ...")
for p in paths:
    m = re.match(r"^(\d+-\d+)_([\d.]+)\.csv$", os.path.basename(p))
    if not m:
        print(f"  skip (name pattern): {os.path.basename(p)}")
        continue
    win, rr = m.group(1), float(m.group(2))
    df = load_file(p)
    if df is None or df.empty:
        print(f"  skip (unreadable): {os.path.basename(p)}")
        continue
    df["window"], df["RR"] = win, rr
    frames.append(df)
    rows.append(summarise(df, win, rr))

W = pd.DataFrame(rows)
W["_h"] = W["window"].str.split("-").str[0].astype(int)
W = W.sort_values("_h").drop(columns="_h").reset_index(drop=True)

# ── COMBINED PORTFOLIO (trades realised in exit-time order) ───────────────────
ALL = pd.concat(frames, ignore_index=True).sort_values("exit_time").reset_index(drop=True)
ALL["equity"] = ALL["net"].cumsum()
peak = ALL["equity"].cummax()
ALL["drawdown"] = ALL["equity"] - peak
comb = summarise(ALL, "== COMBINED ==", None)

print("\n" + "=" * 118)
print("PER-WINDOW RESULTS (net of $%.0f/round-turn commission)" % COMMISSION_PER_RT)
print("=" * 118)
show = ["window", "RR", "trades", "net_profit", "maxDD$", "maxDD_float$",
        "recovery_float", "win%", "PF", "avg_trade", "maxLossStreak"]
print(W[show].to_string(index=False))

print("\n" + "=" * 118)
print("COMBINED PORTFOLIO  (all windows traded together, as in your multi-account setup)")
print("=" * 118)
for k in ["trades", "gross_profit", "commission", "net_profit", "maxDD$",
          "maxDD_float$", "recovery", "recovery_float", "win%", "PF", "avg_trade",
          "maxLossStreak", "first", "last"]:
    print(f"  {k:<16} {comb[k]}")
print(f"  {'sum of window':<16} {W['net_profit'].sum():.0f}  (equals combined net profit)")
print(f"  {'sum of window DD':<16} {W['maxDD$'].sum():.0f}  <- if DDs all hit at once (worst case);"
      f" actual combined DD is ${comb['maxDD$']:,} thanks to diversification")

# ── YEARLY BREAKDOWN ──────────────────────────────────────────────────────────
ALL["year"] = ALL["exit_time"].dt.year
yearly = ALL.groupby("year").agg(trades=("net", "size"), net=("net", "sum")).round(0)
yearly["maxDD$"] = ALL.groupby("year")["net"].apply(lambda s: round(dd_stats(s)[0]))
print("\n== Combined by year ==")
print(yearly.to_string())

# ── SAVE ──────────────────────────────────────────────────────────────────────
os.makedirs("output_files", exist_ok=True)


def save_csv(df, path):
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        import time
        alt = path.replace(".csv", f"_{time.strftime('%H%M%S')}.csv")
        df.to_csv(alt, index=False)
        print(f"  ({path} locked — open in Excel? wrote {alt})")
        return alt


out_all = pd.concat([W, pd.DataFrame([comb])], ignore_index=True)
p1 = save_csv(out_all, OUT_SUMMARY)
p2 = save_csv(ALL[["window", "RR", "entry_time", "exit_time", "mae", "mfe",
                   "profit", "net", "equity", "drawdown"]], OUT_TRADES)

# ── PLOTS ─────────────────────────────────────────────────────────────────────
os.makedirs(PLOT_DIR, exist_ok=True)

fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
a1.plot(ALL["exit_time"], ALL["equity"], color="tab:blue", lw=1.5)
a1.set_ylabel("Equity $ (net)")
a1.set_title(f"Combined portfolio — {len(frames)} windows, {comb['trades']} trades   "
             f"net \${comb['net_profit']:,.0f}   maxDD \${comb['maxDD$']:,.0f}   "
             f"recovery {comb['recovery']}")
a1.grid(alpha=0.25)
i_dd = ALL["drawdown"].idxmin()
a1.annotate(f"max DD \${-ALL['drawdown'].min():,.0f}",
            xy=(ALL["exit_time"][i_dd], ALL["equity"][i_dd]),
            xytext=(10, -30), textcoords="offset points", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="tab:red"), color="tab:red")
a2.fill_between(ALL["exit_time"], ALL["drawdown"], 0, color="tab:red", alpha=0.4)
a2.set_ylabel("Drawdown $")
a2.set_xlabel("Date")
a2.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "combined_equity.png"), dpi=110)
plt.close(fig)

n = len(frames)
ncol, nrow = 3, int(np.ceil(n / 3))
fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.0 * nrow), sharex=True)
for ax, fr in zip(np.ravel(axes), sorted(frames, key=lambda d: int(d["window"].iloc[0].split("-")[0]))):
    s = fr.sort_values("exit_time")
    eq = s["net"].cumsum()
    w, rr = s["window"].iloc[0], s["RR"].iloc[0]
    mdd = dd_stats(s["net"])[0]
    ax.plot(s["exit_time"], eq, lw=1.3,
            color="tab:green" if eq.iloc[-1] > 0 else "tab:red")
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_title(f"{w} @ RR {rr}   net \${eq.iloc[-1]:,.0f}  DD \${mdd:,.0f}", fontsize=9)
    ax.grid(alpha=0.2)
for ax in np.ravel(axes)[n:]:
    ax.axis("off")
fig.suptitle("Per-window equity curves (net of commission)", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "per_window_equity.png"), dpi=110)
plt.close(fig)

print(f"\nSaved: {p1}")
print(f"       {p2}")
print(f"Plots: {PLOT_DIR}/combined_equity.png , {PLOT_DIR}/per_window_equity.png")
print("\nNOTE: combined = windows run independently (your multi-account setup). A SINGLE")
print("account running all windows would hit position-slot blocking and do worse.")
