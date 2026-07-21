"""
analyze_multi_rr.py
===================
Consumes several REAL MT5 backtest exports (one per RiskReward value) and asks
the only question that matters:

    "Is there an observable regime that tells me to use a bigger RR here
     and a smaller RR there — and does acting on it beat a single fixed RR?"

No ML, no payoff reconstruction. Every number below is real MT5 output.

INPUT  (drop these in input_files/ — produced by the EA, one run per RR):
    trade_stats_rr_1.0.csv
    trade_stats_rr_1.5.csv
    trade_stats_rr_2.0.csv
    trade_stats_rr_2.5.csv
    trade_stats_rr_3.0.csv

Column format is auto-detected. The EA writes:
    ticket, entry_time, exit_time, mae_money, mfe_money, trade_profit, candle_range
(older exports used Ticket/Entry_time/.../PNL/SL — both are handled.)

NOTE ON UNITS: trade_profit / mae / mfe are in ACCOUNT MONEY; candle_range is in
POINTS. We convert each trade to R = trade_profit / (candle_range * POINT_VALUE * LOTS).
"""

import glob
import io
import os
import re
import numpy as np
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────────────────
INPUT_DIR   = "input_files"
FILE_GLOB   = "trade_stats_rr_*.csv"
OHLCV_FILE  = "input_files/MT5_databento-ohlcv-1m.csv"

POINT_VALUE = 2.0     # MNQ = $2 per point per contract. Change if wrong.
LOTS        = 1.0     # must match the 'Lots' input used in the backtests

TREND_TF    = "30min"
EMA_PERIOD  = 200
TRAIN_RATIO = 0.80

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


# ── LOADING / COLUMN NORMALISATION ────────────────────────────────────────────
COL_MAP = {
    "ticket": "ticket", "entry_time": "entry_time", "exit_time": "exit_time",
    "mae_money": "mae", "mae": "mae", "mfe_money": "mfe", "mfe": "mfe",
    "trade_profit": "pnl", "pnl": "pnl",
    "candle_range": "sl", "sl": "sl",
}


def _read_any_encoding(path):
    for enc in ["utf-16-le", "utf-16", "utf-8-sig", "utf-8"]:
        try:
            with open(path, "r", encoding=enc) as f:
                raw = f.read()
            # EA writes CSV with ';' or ',' — sniff both plus tab
            for sep in ["\t", ",", ";"]:
                df = pd.read_csv(io.StringIO(raw), sep=sep)
                if df.shape[1] >= 5:
                    return df
        except Exception:
            continue
    raise RuntimeError(f"Cannot parse {path}")


def load_rr_file(path):
    df = _read_any_encoding(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={c: COL_MAP[c] for c in df.columns if c in COL_MAP})

    # keep real trade rows only (entry_time starts with a date)
    df = df[df["entry_time"].astype(str).str.match(r"^\d{4}\.\d{2}\.\d{2}")].copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], format="%Y.%m.%d %H:%M:%S")
    for c in ["mae", "mfe", "pnl", "sl"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["sl"].notna() & (df["sl"] > 0) & df["pnl"].notna()].copy()

    risk_money = df["sl"] * POINT_VALUE * LOTS      # 1R in account money
    df["R"] = df["pnl"] / risk_money
    return df.sort_values("entry_time").reset_index(drop=True)


print("Loading RR exports ...")
paths = sorted(glob.glob(os.path.join(INPUT_DIR, FILE_GLOB)))
if not paths:
    raise SystemExit(
        f"No files matching {INPUT_DIR}/{FILE_GLOB}.\n"
        "Run the EA once per RiskReward (1.0, 1.5, 2.0, 2.5, 3.0) and drop the\n"
        "resulting trade_stats_rr_*.csv files into input_files/."
    )

rr_data = {}
for p in paths:
    m = re.search(r"rr_([0-9.]+)\.csv$", os.path.basename(p))
    if not m:
        continue
    rr = float(m.group(1))
    try:
        rr_data[rr] = load_rr_file(p)
        print(f"  RR {rr:<4} : {len(rr_data[rr]):>6} trades   ({os.path.basename(p)})")
    except Exception as e:
        print(f"  RR {rr:<4} : FAILED to load ({os.path.basename(p)}): {e}")

RRS = sorted(rr_data)
if not RRS:
    raise SystemExit("Found files but could not parse RR value from names.")


# ── REGIME LABELS FROM OHLCV ──────────────────────────────────────────────────
print("Building regime labels from OHLCV ...")
ohlcv = pd.read_csv(OHLCV_FILE, sep="\t",
                    usecols=["<DATE>", "<TIME>", "<CLOSE>"],
                    dtype={"<CLOSE>": "float32"})
ts = pd.to_datetime(ohlcv["<DATE>"] + " " + ohlcv["<TIME>"], format="%Y.%m.%d %H:%M:%S")
close_1m = pd.Series(ohlcv["<CLOSE>"].values, index=ts).sort_index()
htf = close_1m.resample(TREND_TF, label="right", closed="right").last().dropna()
ema = htf.ewm(span=EMA_PERIOD, adjust=False).mean()
trend = pd.DataFrame({"bar_close_time": htf.index, "trend_up": (htf > ema).values})
trend = trend.iloc[EMA_PERIOD:]


def attach_regime(df):
    df = pd.merge_asof(df.sort_values("entry_time"), trend,
                       left_on="entry_time", right_on="bar_close_time",
                       direction="backward")
    df["trend_up"] = df["trend_up"].astype("boolean")
    df["year"] = df["entry_time"].dt.year
    h = df["entry_time"].dt.hour
    df["session"] = np.select(
        [h < 1, h < 10, h < 23],
        ["00 closed", "01-10 morning", "10-23 main"],
        default="23-24 evening",
    )
    return df


for rr in RRS:
    rr_data[rr] = attach_regime(rr_data[rr])


# ── METRICS ───────────────────────────────────────────────────────────────────
def metrics(df):
    R = df["R"].values
    money = df["pnl"].values
    wins, losses = R[R > 0], R[R < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    equity = np.cumsum(money)
    dd = (np.maximum.accumulate(equity) - equity).max() if len(equity) else 0.0
    return dict(n=len(R), money=money.sum(), R=R.sum(), avgR=R.mean() if len(R) else 0,
                win=(R > 0).mean() * 100 if len(R) else 0, pf=pf, maxDD=dd)


def print_row(tag, m):
    print(f"  {tag:<26} n={m['n']:>6}  ${m['money']:>11,.0f}  totalR={m['R']:>8.1f}  "
          f"avgR={m['avgR']:>6.3f}  win%={m['win']:>5.1f}  PF={m['pf']:>5.2f}  maxDD=${m['maxDD']:>10,.0f}")


# ── 1) OVERALL PER-RR PERFORMANCE (full / train / test) ───────────────────────
print("\n" + "=" * 100)
print("1)  OVERALL PERFORMANCE PER FIXED RR   (real MT5 results)")
print("=" * 100)
for rr in RRS:
    d = rr_data[rr]
    split = int(len(d) * TRAIN_RATIO)
    print(f"\n RR = {rr}")
    print_row("full sample", metrics(d))
    print_row("train (first 80%)", metrics(d.iloc[:split]))
    print_row("test  (last 20%)", metrics(d.iloc[split:]))


# ── 2) MATCHED PANEL: same trades, R at every RR ──────────────────────────────
# Join all RR runs on entry_time so we compare like-for-like per trade.
print("\n" + "=" * 100)
print("2)  MATCHED-TRADE PANEL (entries common to ALL RR runs)")
print("=" * 100)
panel = rr_data[RRS[0]][["entry_time", "trend_up", "year", "session"]].copy()
for rr in RRS:
    r_series = rr_data[rr][["entry_time", "R"]].rename(columns={"R": f"R_{rr}"})
    panel = panel.merge(r_series, on="entry_time", how="inner")
panel = panel.drop_duplicates("entry_time").reset_index(drop=True)
rr_cols = [f"R_{rr}" for rr in RRS]
print(f"  {len(panel)} trades present in all {len(RRS)} runs "
      f"(each individual run had ~{int(np.mean([len(rr_data[rr]) for rr in RRS]))}).")


def best_rr_table(group_col):
    """Total R by RR within each group; flag the winning RR per row."""
    g = panel.groupby(group_col)[rr_cols].sum()
    g.columns = [c.replace("R_", "RR ") for c in g.columns]
    g["BEST"] = g.idxmax(axis=1)
    g["n"] = panel.groupby(group_col).size()
    return g.round(1)


# ── 3) REGIME DIAGNOSTICS — does the optimal RR move with the regime? ─────────
print("\n" + "=" * 100)
print("3)  DOES OPTIMAL RR DRIFT WITH REGIME?   (total R by RR within each bucket)")
print("=" * 100)

print("\n [By calendar year]  <-- directly tests 'optimal RR changes by period'")
print(best_rr_table("year").to_string())

print("\n [By 200-EMA trend state]")
print(best_rr_table("trend_up").to_string())

print("\n [By session / time of day]")
print(best_rr_table("session").to_string())


# ── 4) HONEST TEST: learn a per-trend RR on train, judge on test ──────────────
print("\n" + "=" * 100)
print("4)  HONEST CHECK — pick RR-per-trend on TRAIN, apply to unseen TEST")
print("=" * 100)
split = int(len(panel) * TRAIN_RATIO)
tr, te = panel.iloc[:split], panel.iloc[split:]

# best single global RR on train
glob_best = max(RRS, key=lambda rr: tr[f"R_{rr}"].sum())
# best RR per trend state on train
by_trend = {}
for state in [True, False]:
    sub = tr[tr["trend_up"] == state]
    by_trend[state] = max(RRS, key=lambda rr: sub[f"R_{rr}"].sum()) if len(sub) else glob_best

print(f"  Learned on train:  global best RR = {glob_best}   |   "
      f"uptrend -> RR {by_trend[True]},  downtrend -> RR {by_trend[False]}")

def test_total(pick_fn):
    r = [te.iloc[i][f"R_{pick_fn(te.iloc[i])}"] for i in range(len(te))]
    return np.array(r)

fixed1   = te["R_1.0"].values if "R_1.0" in te else te[rr_cols[0]].values
fixedbst = te[f"R_{glob_best}"].values
regime   = test_total(lambda row: by_trend[bool(row["trend_up"])] if pd.notna(row["trend_up"]) else glob_best)

print("\n  TEST-set totals (real R):")
print(f"    Fixed RR=1.0 (baseline)      totalR={fixed1.sum():>8.1f}   avgR={fixed1.mean():>6.3f}")
print(f"    Fixed RR={glob_best} (best global)     totalR={fixedbst.sum():>8.1f}   avgR={fixedbst.mean():>6.3f}")
print(f"    Regime rule (RR by trend)    totalR={regime.sum():>8.1f}   avgR={regime.mean():>6.3f}")

print("\nInterpretation guide:")
print("  * If 'best global RR' >> 'RR=1' but 'regime rule' ~= 'best global', the")
print("    edge is just a better fixed RR — no regime switching needed.")
print("  * If 'regime rule' clearly beats 'best global', adaptive RR is real.")
print("  * If the BEST column in section 3 is the same RR for every bucket, there")
print("    is no regime to exploit — pick that RR and stop.")
print("\nDone.")
