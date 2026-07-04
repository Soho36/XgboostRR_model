"""
analyze_optimization.py
=======================
Reads MT5 optimization result XMLs (one per time window) — each is a sweep of
RiskReward for a SINGLE isolated window — and helps pick a DRAWDOWN-AWARE RR
for every window, not the profit-maximising one MT5 sorts to the top.

Why this data is clean: in an isolated window the ENTRY set doesn't depend on
RR (RR only changes the exit), so every RR row is the SAME trades. The RR sweep
is therefore a controlled experiment: profit vs drawdown, all else equal.

INPUT : Optimization_xlmls/<window>.xml   (MT5 "Optimization Results" export)
OUTPUT: - console table: recommended RR per window under several criteria
        - plots/optimization/<window>.png : Profit & DD vs RR, Sharpe/Recovery
        - output_files/window_rr_recommendations.csv

Pick your prop DD ceiling below (MAX_DD_PCT). The 'capped' recommendation is the
most profitable RR whose Equity-DD% stays under that ceiling.
"""

import glob, os
import xml.etree.ElementTree as ET
# import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# ── CONFIG ────────────────────────────────────────────────────────────────────
XML_DIR = "Optimization_xlmls"
PLOT_DIR = "plots/optimization"
OUT_CSV = "output_files/window_rr_recommendations.csv"
MAX_DD_USD = 2000.0  # per-account prop drawdown ceiling in $ (at tester lot size)
MIN_TRADES = 100     # ignore passes with fewer trades (safety)
MIN_RECOVERY = 2.0   # a window whose best pass earns < this x its own maxDD = WEAK

SS = "urn:schemas-microsoft-com:office:spreadsheet"


# ── ROBUST SpreadsheetML PARSER (respects ss:Index gaps) ─────────────────────
def parse_mt5_xml(path):
    root = ET.parse(path).getroot()
    rows = root.findall(f".//{{{SS}}}Worksheet/{{{SS}}}Table/{{{SS}}}Row")
    parsed = []
    for row in rows:
        cells, col = {}, 0
        for cell in row.findall(f"{{{SS}}}Cell"):
            idx = cell.get(f"{{{SS}}}Index")
            col = int(idx) if idx else col + 1
            data = cell.find(f"{{{SS}}}Data")
            cells[col] = data.text if data is not None else None
        parsed.append(cells)
    if not parsed:
        return None
    header = [parsed[0].get(i) for i in range(1, max(parsed[0]) + 1)]
    recs = []
    for cells in parsed[1:]:
        recs.append({header[i - 1]: cells.get(i) for i in range(1, len(header) + 1)})
    df = pd.DataFrame(recs)
    df.columns = [str(c).strip() for c in df.columns]
    ren = {"Profit": "profit", "Profit Factor": "pf", "Recovery Factor": "recovery",
           "Sharpe Ratio": "sharpe", "Equity DD %": "dd_pct", "Trades": "trades",
           "RiskReward": "rr", "Expected Payoff": "exp_payoff", "Result": "result"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    for c in ["profit", "pf", "recovery", "sharpe", "dd_pct", "trades", "rr", "exp_payoff", "result"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # absolute $ drawdown = net profit / recovery factor (MT5's definition)
    if "recovery" in df.columns:
        df["dd_usd"] = (df["profit"] / df["recovery"]).where(
            (df["recovery"] > 0) & (df["profit"] > 0))
    df = df.dropna(subset=["rr", "profit"]).sort_values("rr").reset_index(drop=True)
    return df


def pick(df, col, maximize=True):
    d = df[df["trades"] >= MIN_TRADES] if "trades" in df else df
    if d.empty:
        return None
    i = d[col].idxmax() if maximize else d[col].idxmin()
    return d.loc[i]


def recommend(df):
    out = {}
    # 1. what MT5 highlights: max profit (DD-blind)
    r = pick(df, "profit");
    out["maxProfit"] = r
    # 2. best risk-adjusted: recovery factor (= profit / maxDD) and sharpe
    r = pick(df, "recovery");
    out["maxRecovery"] = r
    r = pick(df, "sharpe");
    out["maxSharpe"] = r
    # 3. prop recommendation: best risk-adjusted (Recovery Factor) RR whose
    #    $ drawdown stays under the ceiling — the knee, not the edge of the cliff
    allowed = df[(df["dd_usd"] <= MAX_DD_USD) & df["dd_usd"].notna()
                 & (df["trades"] >= MIN_TRADES)]
    out["recommended"] = allowed.loc[allowed["recovery"].idxmax()] if not allowed.empty else None
    return out


# ── PLOT ──────────────────────────────────────────────────────────────────────
def plot_window(name, df, rec):
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # top: profit (left axis) + DD% (right axis)
    ax1.plot(df["rr"], df["profit"], color="tab:blue", lw=1.8, label="Profit $")
    ax1.set_ylabel("Profit $", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(df["rr"], df["dd_usd"], color="tab:red", lw=1.4, alpha=0.8, label="Max DD $")
    ax2.axhline(MAX_DD_USD, color="tab:red", ls=":", lw=1, alpha=0.7)
    ax2.set_ylabel("Max Drawdown $", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    colors = {"maxProfit": "grey", "maxRecovery": "tab:green",
              "maxSharpe": "tab:purple", "recommended": "black"}
    for key, r in rec.items():
        if r is not None:
            ax1.axvline(r["rr"], color=colors[key], ls="--", lw=1.3,
                        label=f"{key}: RR={r['rr']:.2f} (DD=${r['dd_usd']:,.0f})")
    ax1.set_title(f"Window {name} — RR sweep  (trades={int(df['trades'].median())})")
    ax1.legend(loc="upper left", fontsize=8)

    # bottom: risk-adjusted curves
    ax3.plot(df["rr"], df["recovery"], color="tab:green", lw=1.6, label="Recovery Factor")
    ax3.plot(df["rr"], df["sharpe"], color="tab:purple", lw=1.6, label="Sharpe Ratio")
    ax3.plot(df["rr"], df["pf"], color="tab:orange", lw=1.2, alpha=0.8, label="Profit Factor")
    ax3.set_xlabel("RiskReward")
    ax3.set_ylabel("risk-adjusted")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="best", fontsize=8)

    fig.tight_layout()
    p = os.path.join(PLOT_DIR, f"{name}.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


# ── MAIN ──────────────────────────────────────────────────────────────────────
paths = sorted(glob.glob(os.path.join(XML_DIR, "*.xml")))
if not paths:
    raise SystemExit(f"No XMLs in {XML_DIR}/. Export MT5 optimization results there.")

os.makedirs("output_files", exist_ok=True)
summary = []
for p in paths:
    name = os.path.splitext(os.path.basename(p))[0].replace("_opt", "")
    df = parse_mt5_xml(p)
    if df is None or df.empty:
        print(f"  {name}: could not parse — skipped")
        continue
    rec = recommend(df)
    png = plot_window(name, df, rec)
    row = {"window": name, "rr_lo": df["rr"].min(), "rr_hi": df["rr"].max(),
           "trades": int(df["trades"].median())}
    for key, r in rec.items():
        if r is not None:
            row[f"{key}_RR"] = round(float(r["rr"]), 2)
            row[f"{key}_profit"] = round(float(r["profit"]), 0)
            row[f"{key}_DD$"] = round(float(r["dd_usd"]), 0) if pd.notna(r.get("dd_usd")) else None
    # verdict under the $ ceiling
    mp, rc = rec["maxProfit"], rec["recommended"]
    if mp is None or mp["profit"] <= 0:
        row["verdict"] = "LOSING"
    elif rc is None:
        row["verdict"] = "EXCEEDS_DD"
    elif rc["recovery"] < MIN_RECOVERY:
        row["verdict"] = "WEAK"
    else:
        row["verdict"] = "OK"
    summary.append(row)
    print(f"\n-- Window {name}  (trades={row['trades']})  verdict={row['verdict']} --")
    for key, r in rec.items():
        if r is not None:
            ddv = r.get("dd_usd")
            dds = f"${ddv:>7,.0f}" if pd.notna(ddv) else "    n/a"
            print(f"   {key:<12} RR={r['rr']:.2f}  profit=${r['profit']:>8,.0f}  "
                  f"maxDD={dds}  recovery={r['recovery']:.2f}  sharpe={r['sharpe']:.2f}")
    print(f"   plot: {png}")

if summary:
    S = pd.DataFrame(summary)
    try:
        S.to_csv(OUT_CSV, index=False)
        out_path = OUT_CSV
    except PermissionError:
        import time
        out_path = OUT_CSV.replace(".csv", f"_{time.strftime('%H%M%S')}.csv")
        S.to_csv(out_path, index=False)
        print(f"  ({OUT_CSV} was locked — is it open in Excel? Wrote {out_path} instead.)")
    print(f"\nSaved recommendations table → {out_path}")
    print(f"Plots → {PLOT_DIR}/")
    n_ok = (S["verdict"] == "OK").sum()
    print(f"\nWindows OK under ${MAX_DD_USD:,.0f} ceiling: {n_ok} / {len(S)}")
    print("\nCAVEAT: this caps EACH window ALONE. Your $2k limit is PER ACCOUNT and each")
    print("account runs ~2 windows/day — two windows can stack drawdowns. The real")
    print("per-account check is the multi-account portfolio simulation (next step).")
