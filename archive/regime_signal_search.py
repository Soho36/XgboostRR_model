"""
regime_signal_search.py
=======================
Follow-up to analyze_multi_rr.py. Two changes driven by what we learned:

  1. DRAWDOWN-AWARE metrics. For a prop account the objective is not "max total
     R" but survival: return per unit of max drawdown, and the longest losing
     streak. Every table below reports maxDD (in R) and R/DD, not just profit.

  2. A SENSITIVE, causally-available regime signal. The 30-min 200-EMA failed
     (best RR was 3.0 in BOTH up and down states). The by-year table, however,
     proved a real regime exists (2022 -> RR1 best, bull years -> RR3). So we
     test *daily* trend signals that turn over in weeks, and ask whether
     "high RR in up-regime, low RR in down-regime" beats the best FIXED RR
     out-of-sample on a drawdown-adjusted basis.

All numbers are real MT5 output. R = trade_profit / (candle_range*POINT_VALUE*LOTS).
"""

import glob, io, os, re
import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_DIR   = "input_files"
FILE_GLOB   = "trade_stats_rr_*.csv"
OHLCV_FILE  = "input_files/MT5_databento-ohlcv-1m.csv"
POINT_VALUE = 2.0
LOTS        = 1.0
TRAIN_RATIO = 0.80

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

COL_MAP = {"ticket": "ticket", "entry_time": "entry_time", "exit_time": "exit_time",
           "mae_money": "mae", "mae": "mae", "mfe_money": "mfe", "mfe": "mfe",
           "trade_profit": "pnl", "pnl": "pnl", "candle_range": "sl", "sl": "sl"}


def _read_any(path):
    for enc in ["utf-16-le", "utf-16", "utf-8-sig", "utf-8"]:
        try:
            raw = open(path, "r", encoding=enc).read()
            for sep in ["\t", ",", ";"]:
                df = pd.read_csv(io.StringIO(raw), sep=sep)
                if df.shape[1] >= 5:
                    return df
        except Exception:
            continue
    raise RuntimeError(f"Cannot parse {path}")


def load_rr(path):
    df = _read_any(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={c: COL_MAP[c] for c in df.columns if c in COL_MAP})
    df = df[df["entry_time"].astype(str).str.match(r"^\d{4}\.\d{2}\.\d{2}")].copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], format="%Y.%m.%d %H:%M:%S")
    for c in ["pnl", "sl"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["sl"].notna() & (df["sl"] > 0) & df["pnl"].notna()].copy()
    df["R"] = df["pnl"] / (df["sl"] * POINT_VALUE * LOTS)
    return df.sort_values("entry_time").reset_index(drop=True)


print("Loading RR exports ...")
rr_data = {}
for p in sorted(glob.glob(os.path.join(INPUT_DIR, FILE_GLOB))):
    m = re.search(r"rr_([0-9.]+)\.csv$", os.path.basename(p))
    if m:
        rr_data[float(m.group(1))] = load_rr(p)
RRS = sorted(rr_data)
print(f"  RR values: {RRS}")


# ── DAILY REGIME SIGNALS (causal: use only completed prior days) ──────────────
print("Building DAILY regime signals from OHLCV ...")
oh = pd.read_csv(OHLCV_FILE, sep="\t", usecols=["<DATE>", "<TIME>", "<CLOSE>"],
                 dtype={"<CLOSE>": "float32"})
ts = pd.to_datetime(oh["<DATE>"] + " " + oh["<TIME>"], format="%Y.%m.%d %H:%M:%S")
close_1m = pd.Series(oh["<CLOSE>"].values, index=ts).sort_index()

# Daily close series, stamped at day END so merge_asof(backward) is lookahead-free
daily = close_1m.resample("1D", label="right", closed="right").last().dropna()
D = pd.DataFrame({"t": daily.index, "close": daily.values})
D["ema20"]  = D["close"].ewm(span=20, adjust=False).mean()
D["ema50"]  = D["close"].ewm(span=50, adjust=False).mean()
# candidate regime signals (all boolean = "up-regime")
D["sig_ema20_slope"] = D["ema20"] > D["ema20"].shift(1)        # 20d EMA rising
D["sig_ema50_slope"] = D["ema50"] > D["ema50"].shift(1)        # 50d EMA rising
D["sig_px_gt_ema50"] = D["close"] > D["ema50"]                 # price above 50d EMA
D["sig_mom20"]       = D["close"] > D["close"].shift(20)       # up over last 20 days
SIGNALS = ["sig_ema20_slope", "sig_ema50_slope", "sig_px_gt_ema50", "sig_mom20"]
D = D.dropna().reset_index(drop=True)


def attach_signals(df):
    return pd.merge_asof(df.sort_values("entry_time"), D[["t"] + SIGNALS],
                         left_on="entry_time", right_on="t", direction="backward")


# ── MATCHED PANEL: R at every RR for the SAME trades + regime signals ─────────
panel = rr_data[RRS[0]][["entry_time"]].copy()
for rr in RRS:
    panel = panel.merge(rr_data[rr][["entry_time", "R"]].rename(columns={"R": f"R_{rr}"}),
                        on="entry_time", how="inner")
panel = panel.drop_duplicates("entry_time").reset_index(drop=True)
panel = attach_signals(panel).dropna(subset=SIGNALS).reset_index(drop=True)
for s in SIGNALS:
    panel[s] = panel[s].astype(bool)
rr_cols = [f"R_{rr}" for rr in RRS]
print(f"  Matched panel: {len(panel)} trades common to all {len(RRS)} runs\n")


# ── DD-AWARE METRICS ──────────────────────────────────────────────────────────
def dd_stats(r):
    r = np.asarray(r, float)
    eq = np.cumsum(r)
    maxdd = float((np.maximum.accumulate(eq) - eq).max()) if len(r) else 0.0
    # longest losing streak
    mcl = cur = 0
    for x in r:
        cur = cur + 1 if x < 0 else 0
        mcl = max(mcl, cur)
    tot = float(r.sum())
    return dict(totalR=tot, maxDD_R=maxdd,
                R_per_DD=(tot / maxdd if maxdd > 0 else np.inf),
                win=(r > 0).mean() * 100 if len(r) else 0, maxLossStreak=mcl, n=len(r))


def show(tag, r):
    m = dd_stats(r)
    print(f"  {tag:<34} totalR={m['totalR']:>7.1f}  maxDD_R={m['maxDD_R']:>6.1f}  "
          f"R/DD={m['R_per_DD']:>5.2f}  win%={m['win']:>4.1f}  maxLossStreak={m['maxLossStreak']:>3}")


# ── 1) FIXED RR ON THE SAME TRADES, NOW WITH DRAWDOWN ─────────────────────────
print("=" * 100)
print("1)  FIXED RR compared on identical trades — with DRAWDOWN (the prop metric)")
print("=" * 100)
for rr in RRS:
    show(f"Fixed RR={rr}", panel[f"R_{rr}"].values)
print("  ^ Read R/DD (return per unit of drawdown), not just totalR.")


# ── 2) DOES EACH DAILY SIGNAL SEPARATE THE BEST RR? ───────────────────────────
print("\n" + "=" * 100)
print("2)  Does a DAILY regime signal make the best RR differ up vs down?")
print("=" * 100)
for s in SIGNALS:
    up, dn = panel[panel[s]], panel[~panel[s]]
    up_best = max(RRS, key=lambda rr: up[f"R_{rr}"].sum())
    dn_best = max(RRS, key=lambda rr: dn[f"R_{rr}"].sum())
    flag = "  <-- differs!" if up_best != dn_best else ""
    print(f"  {s:<18} up-regime(n={len(up):>5}) best RR={up_best} | "
          f"down-regime(n={len(dn):>5}) best RR={dn_best}{flag}")


# ── 3) HONEST TRAIN→TEST, DD-ADJUSTED, FOR EACH SIGNAL ───────────────────────
print("\n" + "=" * 100)
print("3)  Learn RR-per-regime on TRAIN, judge on TEST (profit AND drawdown)")
print("=" * 100)
split = int(len(panel) * TRAIN_RATIO)
tr, te = panel.iloc[:split], panel.iloc[split:]
glob_best = max(RRS, key=lambda rr: tr[f"R_{rr}"].sum())

print(f"  Baselines on TEST:")
show("Fixed RR=1.0", te["R_1.0"].values)
show(f"Fixed RR={glob_best} (best global on train)", te[f"R_{glob_best}"].values)

for s in SIGNALS:
    up_best = max(RRS, key=lambda rr: tr[tr[s]][f"R_{rr}"].sum())
    dn_best = max(RRS, key=lambda rr: tr[~tr[s]][f"R_{rr}"].sum())
    picked = np.where(te[s].values,
                      te[f"R_{up_best}"].values, te[f"R_{dn_best}"].values)
    show(f"Adaptive[{s}] up={up_best},dn={dn_best}", picked)


# ── 4) THE 'CUT RR IN DOWN-REGIME' RULE (your prop intuition) ────────────────
print("\n" + "=" * 100)
print("4)  Prop rule: default high RR, CUT to RR=1 in down-regime (DD focus)")
print("=" * 100)
HIGH = glob_best
for s in SIGNALS:
    picked_full = np.where(panel[s].values, panel[f"R_{HIGH}"].values, panel["R_1.0"].values)
    show(f"default {HIGH}, cut->1 when !{s}", picked_full)
print(f"  Compare against full-sample fixed RR={HIGH} and fixed RR=1.0:")
show(f"  fixed RR={HIGH} (all trades)", panel[f"R_{HIGH}"].values)
show("  fixed RR=1.0 (all trades)", panel["R_1.0"].values)

print("\nWhat to look for:")
print("  * A signal in section 2 where up/down pick DIFFERENT RR is the prerequisite.")
print("  * In sections 3/4, a rule that keeps totalR close to the best fixed RR")
print("    while cutting maxDD_R / raising R/DD is the real prop win.")
print("\nDone.")
