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

import glob
import os
import xml.etree.ElementTree as ET
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

# ── CONFIG ────────────────────────────────────────────────────────────────────
XML_DIR = "INPUTS/data_1_optimization_input"
# Each strategy is its own subfolder of XML_DIR with its own inputs and output CSV.
# RR = enter after last RED candle; GG = enter after last GREEN candle.
# Add more strategy names here and drop a matching subfolder of XMLs to include them.
STRATEGIES = ["RR", "GG"]
# Two-period layout (optional). If these subfolders exist inside XML_DIR, PRIMARY
# drives the live RR picks and REFERENCE is used only for a regime-stability
# comparison. If neither exists, flat XML_DIR is the single (primary) period.
PRIMARY_SUBDIR = "recent"   # e.g. 2020-2026 optimizations  (the market you trade)
REF_SUBDIR = "full"         # e.g. 2010-2026 optimizations  (calm-market reference)
REGIME_RR_TOL = 0.40        # |recommended_RR primary - reference| above this = regime-sensitive
PLOT_DIR = "OUTPUTS/plots_outputs/step1_optimization"
OUT_CSV = "results/window_rr_recommendations.csv"
MAX_DD_USD = 2000.0  # per-account prop drawdown ceiling in $ (at tester lot size)
COMMISSION_PER_RT = 1.0  # $ commission per round-turn; MT5 opt was run WITHOUT costs
MIN_TRADES = 100     # ignore passes with fewer trades (safety)
MIN_RECOVERY = 2.0   # a window whose best pass earns < this x its own maxDD = WEAK
SMOOTH_RR = 0.10     # half-width (in RR units) for plateau smoothing / neighborhood
ROBUST_MIN_FRAC = 0.70  # >= this share of the RR neighborhood must stay under the DD cap = 'solid'

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
    # absolute $ drawdown = GROSS profit / recovery factor (MT5's definition).
    # Compute from gross figures BEFORE we net out commission.
    if "recovery" in df.columns:
        df["dd_usd"] = (df["profit"] / df["recovery"]).where(
            (df["recovery"] > 0) & (df["profit"] > 0))
    # Apply commission (MT5 opt was run without costs). Trades is constant across
    # RR in an isolated window, so this is a fixed $ haircut. profit -> NET; recovery
    # recomputed on NET profit but keeping the tester's (gross) $ drawdown.
    # NOTE: dd_usd stays GROSS — commission drag would raise the true DD somewhat,
    # which we can't reconstruct from summary stats (re-run MT5 with commission for
    # exact DD). Net profit is the part that flips a window from winner to loser.
    if "trades" in df.columns and COMMISSION_PER_RT:
        df["commission"] = df["trades"] * COMMISSION_PER_RT
        df["profit_gross"] = df["profit"]
        df["profit"] = df["profit"] - df["commission"]
        df["recovery"] = (df["profit"] / df["dd_usd"]).where(df["dd_usd"] > 0)
    df = df.dropna(subset=["rr", "profit"]).sort_values("rr").reset_index(drop=True)
    # plateau smoothing: rolling mean over a +/- SMOOTH_RR neighbourhood, so an
    # isolated spike gets averaged down by its neighbours (robustness).
    step = df["rr"].diff().median()
    win = max(3, int(round(2 * SMOOTH_RR / step)) | 1) if step and step > 0 else 3
    for c in ["profit", "dd_usd", "recovery"]:
        if c in df.columns:
            df[c + "_s"] = df[c].rolling(win, center=True, min_periods=max(3, win // 3)).mean()
    return df


def robustness(df, rr0):
    """How solid is the pick? Share of the +/-SMOOTH_RR neighbourhood that stays
    under the DD cap, and profit variability there (coefficient of variation)."""
    nb = df[(df["rr"] - rr0).abs() <= SMOOTH_RR]
    if nb.empty:
        return float("nan"), float("nan")
    frac_under = float((nb["dd_usd"] <= MAX_DD_USD).mean())
    m = nb["profit"].mean()
    cv = float(nb["profit"].std() / m) if m else float("nan")
    return frac_under, cv


def pick(df, col, maximize=True):
    d = df[df["trades"] >= MIN_TRADES] if "trades" in df else df
    if d.empty:
        return None
    i = d[col].idxmax() if maximize else d[col].idxmin()
    return d.loc[i]


def recommend(df):
    """Tiered picks for a trailing-DD prop account.

    Phase 1 (new account, DD trailing active -> must stay under the $ cap):
        recommended = safest, best smoothed recovery under the cap
        aggressive  = most profit on a still-under-cap smoothed plateau
    Phase 2 (buffer banked, trailing DD frozen -> cap no longer binds):
        unlocked    = best smoothed recovery ignoring the cap
    """
    out = {}
    base = df[df["trades"] >= MIN_TRADES] if "trades" in df else df
    # under-cap = ACTUAL dd under the hard limit AND its neighbourhood (smoothed) too
    under = base[(base["dd_usd"] <= MAX_DD_USD) & (base["dd_usd_s"] <= MAX_DD_USD)
                 & base["dd_usd"].notna() & base["dd_usd_s"].notna()]

    def amax(d, col):
        d = d[d[col].notna()]
        return d.loc[d[col].idxmax()] if not d.empty else None

    # diagnostics (raw, DD-blind)
    out["maxProfit"] = amax(base, "profit")
    out["maxRecovery"] = amax(base, "recovery")
    # Phase-1 picks (under the $ cap)
    out["recommended"] = amax(under, "recovery_s")
    out["aggressive"] = amax(under, "profit_s")
    # Phase-2 pick (cap ignored — for seasoned accounts / 'saved for later')
    out["unlocked"] = amax(base, "recovery_s")
    return out


# ── PLOT ──────────────────────────────────────────────────────────────────────
def plot_window(name, df, rec, plot_dir=PLOT_DIR):
    os.makedirs(plot_dir, exist_ok=True)
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

    # shade contiguous RR ranges that sit UNDER the DD cap (the usable plateaus)
    mask = (df["dd_usd"] <= MAX_DD_USD).fillna(False).values
    rr = df["rr"].values
    i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            ax1.axvspan(rr[i], rr[j], color="green", alpha=0.07)
            i = j + 1
        else:
            i += 1

    colors = {"recommended": "black", "aggressive": "tab:blue",
              "unlocked": "tab:orange", "maxProfit": "grey"}
    for key in ["recommended", "aggressive", "unlocked", "maxProfit"]:
        r = rec.get(key)
        if r is not None:
            ax1.axvline(r["rr"], color=colors[key], ls="--", lw=1.3,
                        label=f"{key}: RR={r['rr']:.2f} (DD=${r['dd_usd']:,.0f})")
    ax1.set_title(f"Window {name} — RR sweep  (trades={int(df['trades'].median())})")
    ax1.legend(loc="upper left", fontsize=8)

    # bottom: risk-adjusted curves (raw + smoothed recovery to show the plateau)
    ax3.plot(df["rr"], df["recovery"], color="tab:green", lw=1.0, alpha=0.4, label="Recovery (raw)")
    ax3.plot(df["rr"], df["recovery_s"], color="tab:green", lw=2.0, label="Recovery (smoothed)")
    ax3.plot(df["rr"], df["sharpe"], color="tab:purple", lw=1.2, alpha=0.7, label="Sharpe Ratio")
    ax3.plot(df["rr"], df["pf"], color="tab:orange", lw=1.2, alpha=0.8, label="Profit Factor")
    ax3.set_xlabel("RiskReward")
    ax3.set_ylabel("risk-adjusted")
    ax3.grid(alpha=0.25)
    ax3.legend(loc="best", fontsize=8)

    fig.tight_layout()
    p = os.path.join(plot_dir, f"{name}.png")
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


# ── PROCESS ONE PERIOD (a folder of window XMLs) ──────────────────────────────
def process_dir(xml_dir, plot_dir, verbose=True):
    summary = []
    for p in sorted(glob.glob(os.path.join(xml_dir, "*.xml"))):
        name = os.path.splitext(os.path.basename(p))[0].replace("_opt", "")
        df = parse_mt5_xml(p)
        if df is None or df.empty:
            if verbose:
                print(f"  {name}: could not parse — skipped")
            continue
        rec = recommend(df)
        png = plot_window(name, df, rec, plot_dir)
        row = {"window": name, "rr_lo": df["rr"].min(), "rr_hi": df["rr"].max(),
               "trades": int(df["trades"].median()),
               "commission$": round(int(df["trades"].median()) * COMMISSION_PER_RT)}
        for key, r in rec.items():
            if r is not None:
                row[f"{key}_RR"] = round(float(r["rr"]), 2)
                row[f"{key}_profit"] = round(float(r["profit"]), 0)
                row[f"{key}_DD$"] = round(float(r["dd_usd"]), 0) if pd.notna(r.get("dd_usd")) else None
        mp, rc = rec["maxProfit"], rec["recommended"]
        robust = None
        if rc is not None:
            frac, cv = robustness(df, rc["rr"])
            row["plateau_under_cap%"] = round(frac * 100) if pd.notna(frac) else None
            row["nbhd_cv"] = round(cv, 2) if pd.notna(cv) else None
            robust = "solid" if (pd.notna(frac) and frac >= ROBUST_MIN_FRAC) else "fragile"
            row["robust"] = robust
        if mp is None or mp["profit"] <= 0:
            row["verdict"] = "LOSING"
        elif rc is None:
            row["verdict"] = "UNLOCK_ONLY"
        elif rc["recovery_s"] < MIN_RECOVERY:
            row["verdict"] = "WEAK"
        else:
            row["verdict"] = "OK"
        summary.append(row)
        if verbose:
            tag = f"{row['verdict']}" + (f"/{robust}" if robust else "")
            print(f"\n-- Window {name}  (trades={row['trades']})  {tag} --")
            for key, r in rec.items():
                if r is not None:
                    ddv = r.get("dd_usd")
                    dds = f"${ddv:>7,.0f}" if pd.notna(ddv) else "    n/a"
                    print(f"   {key:<12} RR={r['rr']:.2f}  profit=${r['profit']:>8,.0f}  "
                          f"maxDD={dds}  recovery={r['recovery']:.2f}  sharpe={r['sharpe']:.2f}")
    return pd.DataFrame(summary)


def save_csv(df, path):
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        import time
        alt = path.replace(".csv", f"_{time.strftime('%H%M%S')}.csv")
        df.to_csv(alt, index=False)
        print(f"  ({path} locked — open in Excel? Wrote {alt} instead.)")
        return alt


def sort_by_hour(S):
    """Order windows chronologically by start hour (2-3, 3-4, ... 22-23)."""
    if not len(S):
        return S
    S = S.copy()
    S["_h"] = S["window"].str.split("-").str[0].astype(int)
    return S.sort_values("_h").drop(columns="_h").reset_index(drop=True)


# ── RUN ONE STRATEGY (its own inputs -> its own recommendations CSV) ──────────
def run_strategy(name, base_dir, plot_base, out_csv):
    primary_dir = os.path.join(base_dir, PRIMARY_SUBDIR)
    ref_dir = os.path.join(base_dir, REF_SUBDIR)
    two_period = os.path.isdir(primary_dir) and glob.glob(os.path.join(primary_dir, "*.xml"))

    print(f"\n########################  STRATEGY: {name}  ########################")
    if two_period:
        print(f"Two-period mode: PRIMARY={primary_dir}  REFERENCE={ref_dir}")
        S = process_dir(primary_dir, os.path.join(plot_base, PRIMARY_SUBDIR))
        if os.path.isdir(ref_dir) and glob.glob(os.path.join(ref_dir, "*.xml")):
            R = process_dir(ref_dir, os.path.join(plot_base, REF_SUBDIR), verbose=False)
            ref = R[["window", "recommended_RR"]].rename(columns={"recommended_RR": "ref_RR"})
            S = S.merge(ref, on="window", how="left")
            drr = (S["recommended_RR"] - S["ref_RR"]).abs()
            S["regime"] = "n/a"
            S.loc[drr.notna() & (drr <= REGIME_RR_TOL), "regime"] = "stable"
            S.loc[drr.notna() & (drr > REGIME_RR_TOL), "regime"] = "SENSITIVE"
    else:
        if not glob.glob(os.path.join(base_dir, "*.xml")):
            print(f"  No XMLs in {base_dir}/ (or its {PRIMARY_SUBDIR}/ subfolder) — skipped.")
            return
        print(f"Single-period mode: {base_dir}  (add {PRIMARY_SUBDIR}/ + {REF_SUBDIR}/ "
              f"subfolders for regime comparison)")
        S = process_dir(base_dir, plot_base)

    if not len(S):
        return
    S = sort_by_hour(S)
    if "regime" in S.columns:
        print("\n== Regime stability (primary vs reference recommended_RR) ==")
        print(S[["window", "verdict", "recommended_RR", "ref_RR", "regime"]].to_string(index=False))
    out_path = save_csv(S, out_csv)
    print(f"\n[{name}] Saved recommendations → {out_path}   Plots → {plot_base}/")
    print(f"[{name}] Verdicts (net of ${COMMISSION_PER_RT:.0f}/RT commission): "
          f"{S['verdict'].value_counts().to_dict()}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
os.makedirs("OUTPUTS/results_outputs", exist_ok=True)
strat_dirs = [s for s in STRATEGIES
              if os.path.isdir(os.path.join(XML_DIR, s))
              and (glob.glob(os.path.join(XML_DIR, s, "*.xml"))
                   or os.path.isdir(os.path.join(XML_DIR, s, PRIMARY_SUBDIR)))]

if strat_dirs:
    for name in strat_dirs:
        run_strategy(name, os.path.join(XML_DIR, name),
                     os.path.join(PLOT_DIR, name),
                     os.path.join("OUTPUTS/results_outputs", f"{name}_recommendations.csv"))
else:
    # backward-compat: no strategy subfolders yet -> treat flat XML_DIR as one strategy
    print(f"No strategy subfolders {STRATEGIES} found under {XML_DIR}/ — "
          f"treating {XML_DIR}/ as a single strategy. (Move RR XMLs into {XML_DIR}/RR/ "
          f"and GG XMLs into {XML_DIR}/GG/ to separate them.)")
    run_strategy("RR", XML_DIR, PLOT_DIR, OUT_CSV)

print("\nTiers per window: recommended_RR (safe, Phase 1) | aggressive_RR (more profit,"
      " under cap) | unlocked_RR (Phase 2, cap ignored). UNLOCK_ONLY = seasoned-only.")
print("CAVEAT: caps EACH window alone; $2k is PER ACCOUNT and you stack ~2 windows/day"
      " -> multi-account portfolio sim is the real per-account check (next step).")
