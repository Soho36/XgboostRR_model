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

import provenance as prov

matplotlib.use("Agg")

# ── CONFIG ────────────────────────────────────────────────────────────────────
# One entry per strategy: name -> folder of per-window trade exports.
STRATEGIES = {"RR": "data/2_chosen/RR", "GG": "data/2_chosen/GG"}
PLOT_DIR = "reports/plots/step2_portfolio"
OUT_SUMMARY = "data/3_results/{s}_maemfe_window_summary.csv"
OUT_TRADES = "data/3_results/{s}_maemfe_combined_trades.csv"
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


def load_calibration(path="data/3_results/dd_calibration.csv"):
    """Per-(strategy, window, RR) DD scale factors written by step 1.

    Step 1 calibrates its equity-DD reconstruction against MT5's own
    STAT_EQUITY_DD. Without importing those factors here, the RR *pick* would be
    cautious while the portfolio/account DD stayed optimistic on exactly the
    passes where our intra-trade ordering assumption is wrong.
    """
    out = {}
    if os.path.exists(path):
        try:
            c = pd.read_csv(path)
            for r in c.itertuples(index=False):
                out[(r.strategy, r.window, round(float(r.RR), 2))] = float(r.dd_factor)
        except Exception as e:
            print(f"  (could not read {path}: {e})")
    return out


CALIB = load_calibration()


def dd_stats(net_series):
    """Max balance drawdown ($) and recovery factor from a series of trade PnLs.

    The starting balance counts as the first peak (hence the leading 0.0),
    matching MT5's STAT_BALANCE_DD. Without it, a curve that opens with a loss
    and never recovers above its start understates DD by that first loss.
    """
    raw = np.asarray(net_series, dtype=float)
    if not len(raw):
        return 0.0, 0.0, np.array([])
    eq = np.cumsum(raw)
    peak = np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:]
    dd = peak - eq
    maxdd = float(dd.max())
    total = float(eq[-1])
    rec = total / maxdd if maxdd > 0 else np.inf
    return maxdd, rec, dd


def dd_floating(net_series, mae_series, mfe_series):
    """Max EQUITY drawdown, including intra-trade excursions in both directions.

    Closed-trade DD understates what a prop firm measures: while a trade is open
    the equity swings out to +MFE and down to +MAE. Tracking BOTH reproduces
    MT5's STAT_EQUITY_DD exactly (verified on GG 11-12 @2.99: $3,579 by this
    and by MT5); using MAE alone gave $3,081, and dropping both gives the
    balance DD ($2,925 — also an exact match).

    Assumes one position at a time (true per window; approximate when combined).
    """
    eq = peak = maxdd = 0.0
    for n, m, f in zip(np.asarray(net_series, float), np.asarray(mae_series, float),
                       np.asarray(mfe_series, float)):
        # MAE happens before MFE: a buy-stop breakout typically retraces first,
        # then runs. Validated exact against MT5 on 12 passes.
        maxdd = max(maxdd, peak - (eq + min(m, 0.0)))   # dip vs the standing peak
        peak = max(peak, eq + max(f, 0.0))              # then the run-up
        eq += n
        peak = max(peak, eq)
        maxdd = max(maxdd, peak - eq)
    return float(maxdd)


def summarise(df, label, rr=None, dd_factor=1.0):
    net = df["net"].values
    maxdd, rec, _ = dd_stats(net)
    maxdd_f = dd_floating(net, df["mae"].values, df["mfe"].values) * dd_factor
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


# ── LOAD ──────────────────────────────────────────────────────────────────────
def run_strategy(STRAT, MAEMFE_DIR):
  plot_dir = os.path.join(PLOT_DIR, STRAT)
  paths = sorted(glob.glob(os.path.join(MAEMFE_DIR, "*.csv")))
  if not paths:
    print(f"\n### {STRAT}: no CSVs in {MAEMFE_DIR}/ — skipped.")
    return None

  print(f"\n{'#' * 60}\n### STRATEGY {STRAT}  ({MAEMFE_DIR}/)\n{'#' * 60}")
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
    rows.append(summarise(df, win, rr, CALIB.get((STRAT, win, round(rr, 2)), 1.0)))

  W = pd.DataFrame(rows)
  W["_h"] = W["window"].str.split("-").str[0].astype(int)
  W = W.sort_values("_h").drop(columns="_h").reset_index(drop=True)

  # ── COMBINED PORTFOLIO (trades realised in exit-time order) ─────────────────
  ALL = pd.concat(frames, ignore_index=True).sort_values("exit_time").reset_index(drop=True)
  ALL["strategy"] = STRAT
  ALL["equity"] = ALL["net"].cumsum()
  peak = ALL["equity"].cummax()
  ALL["drawdown"] = ALL["equity"] - peak
  # conservative for the combined view: worst factor among the windows in it
  comb_f = max([CALIB.get((STRAT, w, round(r, 2)), 1.0)
                for w, r in zip(W["window"], W["RR"])] or [1.0])
  comb = summarise(ALL, "== COMBINED ==", None, comb_f)

  print("\n" + "=" * 118)
  print(f"[{STRAT}] PER-WINDOW RESULTS (net of ${COMMISSION_PER_RT:.0f}/round-turn commission)")
  print("=" * 118)
  show = ["window", "RR", "trades", "net_profit", "maxDD$", "maxDD_float$",
          "recovery_float", "win%", "PF", "avg_trade", "maxLossStreak"]
  print(W[show].to_string(index=False))

  print("\n" + "=" * 118)
  print(f"[{STRAT}] COMBINED PORTFOLIO (all its windows together)")
  print("=" * 118)
  for k in ["trades", "gross_profit", "commission", "net_profit", "maxDD$",
            "maxDD_float$", "recovery", "recovery_float", "win%", "PF", "avg_trade",
            "maxLossStreak", "first", "last"]:
      print(f"  {k:<16} {comb[k]}")
  print(f"  {'sum of window DD':<16} {W['maxDD$'].sum():.0f}  <- if all DDs hit at once;"
        f" actual combined DD ${comb['maxDD$']:,} (diversification)")

  # ── YEARLY BREAKDOWN ────────────────────────────────────────────────────────
  ALL["year"] = ALL["exit_time"].dt.year
  yearly = ALL.groupby("year").agg(trades=("net", "size"), net=("net", "sum")).round(0)
  yearly["maxDD$"] = ALL.groupby("year")["net"].apply(lambda s: round(dd_stats(s)[0]))
  print(f"\n== [{STRAT}] combined by year ==")
  print(yearly.to_string())

  # ── SAVE ────────────────────────────────────────────────────────────────────
  os.makedirs("data/3_results", exist_ok=True)
  out_all = pd.concat([W, pd.DataFrame([comb])], ignore_index=True)
  p1 = save_csv(out_all, OUT_SUMMARY.format(s=STRAT))
  p2 = save_csv(ALL[["strategy", "window", "RR", "entry_time", "exit_time", "mae",
                     "mfe", "profit", "net", "equity", "drawdown"]],
                OUT_TRADES.format(s=STRAT))

  # ── PLOTS ───────────────────────────────────────────────────────────────────
  os.makedirs(plot_dir, exist_ok=True)
  fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
  a1.plot(ALL["exit_time"], ALL["equity"], color="tab:blue", lw=1.5)
  a1.set_ylabel("Equity $ (net)")
  a1.set_title(f"[{STRAT}] Combined — {len(frames)} windows, {comb['trades']} trades   "
               f"net \\${comb['net_profit']:,.0f}   maxDD \\${comb['maxDD$']:,.0f}   "
               f"recovery {comb['recovery']}")
  a1.grid(alpha=0.25)
  i_dd = ALL["drawdown"].idxmin()
  a1.annotate(f"max DD \\${-ALL['drawdown'].min():,.0f}",
              xy=(ALL["exit_time"][i_dd], ALL["equity"][i_dd]),
              xytext=(10, -30), textcoords="offset points", fontsize=9,
              arrowprops=dict(arrowstyle="->", color="tab:red"), color="tab:red")
  a2.fill_between(ALL["exit_time"], ALL["drawdown"], 0, color="tab:red", alpha=0.4)
  a2.set_ylabel("Drawdown $")
  a2.set_xlabel("Date")
  a2.grid(alpha=0.25)
  fig.tight_layout()
  fig.savefig(os.path.join(plot_dir, "combined_equity.png"), dpi=110)
  plt.close(fig)

  n = len(frames)
  ncol, nrow = 3, int(np.ceil(n / 3))
  fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.0 * nrow), sharex=True,
                           squeeze=False)
  order = sorted(frames, key=lambda d: int(d["window"].iloc[0].split("-")[0]))
  for ax, fr in zip(np.ravel(axes), order):
      s = fr.sort_values("exit_time")
      eq = s["net"].cumsum()
      w, rr = s["window"].iloc[0], s["RR"].iloc[0]
      mdd = dd_stats(s["net"])[0]
      ax.plot(s["exit_time"], eq, lw=1.3,
              color="tab:green" if eq.iloc[-1] > 0 else "tab:red")
      ax.axhline(0, color="grey", lw=0.7)
      ax.set_title(f"{w} @ RR {rr}   net \\${eq.iloc[-1]:,.0f}  DD \\${mdd:,.0f}", fontsize=9)
      ax.grid(alpha=0.2)
  for ax in np.ravel(axes)[n:]:
      ax.axis("off")
  fig.suptitle(f"[{STRAT}] per-window equity curves (net of commission)", y=1.0)
  fig.tight_layout()
  fig.savefig(os.path.join(plot_dir, "per_window_equity.png"), dpi=110)
  plt.close(fig)

  print(f"\nSaved: {p1}\n       {p2}\nPlots: {plot_dir}/")
  return W, ALL, comb


# ── MAIN: run every strategy, then a cross-strategy view ──────────────────────
results = {}
for STRAT, DIRNAME in STRATEGIES.items():
    r = run_strategy(STRAT, DIRNAME)
    if r is not None:
        results[STRAT] = r

if len(results) > 1:
    print("\n" + "#" * 118)
    print("CROSS-STRATEGY VIEW (every window of every strategy traded together)")
    print("#" * 118)
    BOTH = pd.concat([r[1] for r in results.values()], ignore_index=True)
    BOTH = BOTH.sort_values("exit_time").reset_index(drop=True)
    BOTH["equity"] = BOTH["net"].cumsum()
    cb = summarise(BOTH, "== RR+GG ==", None)
    for k in ["trades", "net_profit", "maxDD$", "recovery", "win%", "PF", "avg_trade"]:
        print(f"  {k:<14} {cb[k]}")
    per = {s: r[2] for s, r in results.items()}
    print("\n  strategy      net_profit   maxDD$   recovery")
    for s, c in per.items():
        print(f"  {s:<12} {c['net_profit']:>10}  {c['maxDD$']:>7}   {c['recovery']}")
    print(f"  {'SUM of DDs':<12} {'':>10}  {sum(c['maxDD$'] for c in per.values()):>7}"
          f"   <- vs combined ${cb['maxDD$']:,}  (cross-strategy diversification)")

print("\nNOTE: combined = windows run independently (multi-account setup). One account")
print("running everything would hit position-slot blocking and do worse.")

# ── PROVENANCE ────────────────────────────────────────────────────────────────
PROV2 = prov.base(
    "2_analyze_maemfe",
    upstream=prov.load("data/3_results/_provenance_step1.json"),
    chosen_files={s: sorted(os.path.basename(p)
                            for p in glob.glob(os.path.join(d, "*.csv")))
                  for s, d in STRATEGIES.items()},
    calibration_entries=len(CALIB),
    calibration_max=max(CALIB.values()) if CALIB else 1.0,
    commission_per_rt=COMMISSION_PER_RT,
)
prov.write("data/3_results/_provenance_step2.json", PROV2)
print("\nProvenance: " + prov.summary_line(PROV2))
