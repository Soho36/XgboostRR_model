"""
build_training_dataset.py
=========================
Builds a training-ready CSV from:
  - trade_stats.csv    (MAE, MFE, SL distance per trade)
  - Trades.xlsx        (entry prices, exit reasons)
  - OHLCV 1-min CSV   (market context features)

Output: training_dataset.csv
  One row per completed trade.
  Columns: features computed at entry + rr_bucket label (0/1/2/3).

Usage:
  python build_training_dataset.py

Edit the CONFIGURATION section below to match your file paths.
"""

import pandas as pd
import numpy as np
import io
import os

# ── CONFIGURATION ────────────────────────────────────────────────────────────

TRADE_STATS_FILE = "trade_stats.csv"          # MAE/MFE/SL file from MT5
TRADES_FILE      = "Trades.xlsx"              # Deal log from MT5
OHLCV_FILE       = "MT5_databento-ohlcv-1m.csv"  # 1-min OHLCV

OUTPUT_FILE      = "training_dataset.csv"

# RR bucket boundaries  [0, low, mid, high, inf]
# Trades with achievable RR in each range get label 0/1/2/3
RR_BINS   = [0.0, 1.0, 2.0, 3.0, np.inf]
RR_LABELS = [0,   1,   2,   3]

# How many 1-min bars to load BEFORE each entry for feature computation
LOOKBACK_BARS = 300   # ~5 hours of 1-min data

# ── STEP 1: LOAD TRADE STATS ─────────────────────────────────────────────────

print("Loading trade_stats.csv ...")

# MT5 exports this file as UTF-16-LE with tab separation
with open(TRADE_STATS_FILE, "r", encoding="utf-16-le") as f:
    content = f.read()
stats = pd.read_csv(io.StringIO(content), sep="\t")

# Rename the unnamed SL-distance column
stats = stats.rename(columns={"Unnamed: 6": "SL_distance"})

# Parse timestamps
stats["Entry_time"] = pd.to_datetime(stats["Entry_time"], format="%Y.%m.%d %H:%M:%S")
stats["Exit_time"]  = pd.to_datetime(stats["Exit_time"],  format="%Y.%m.%d %H:%M:%S")

# Drop rows where SL_distance is zero or missing (avoids divide-by-zero)
stats = stats[stats["SL_distance"].notna() & (stats["SL_distance"] > 0)].copy()

print(f"  {len(stats)} completed trades loaded.")

# ── STEP 2: COMPUTE LABELS ───────────────────────────────────────────────────

print("Computing RR labels ...")

# MFE is in price points (same units as SL_distance)
stats["achievable_rr"] = stats["MFE"] / stats["SL_distance"]

# Clip negative MFE to 0 (trade never moved in our favour at all)
stats["achievable_rr"] = stats["achievable_rr"].clip(lower=0)

stats["rr_bucket"] = pd.cut(
    stats["achievable_rr"],
    bins=RR_BINS,
    labels=RR_LABELS,
    right=False          # [low, high)
).astype(int)

label_counts = stats["rr_bucket"].value_counts().sort_index()
print("  Label distribution:")
for bucket, count in label_counts.items():
    pct = count / len(stats) * 100
    print(f"    Bucket {bucket}: {count} trades ({pct:.1f}%)")

# ── STEP 3: LOAD TRADES.XLSX FOR ENTRY PRICES ────────────────────────────────

print("Loading Trades.xlsx ...")

trades_raw = pd.read_excel(TRADES_FILE, header=1)
trades_raw["Time"] = pd.to_datetime(trades_raw["Time"], format="%Y.%m.%d %H:%M:%S")

# Keep only entry rows (direction == 'in')
entries = trades_raw[trades_raw["Direction"] == "in"][["Time", "Price"]].copy()
entries = entries.rename(columns={"Time": "Entry_time", "Price": "entry_price"})
entries["Entry_time"] = pd.to_datetime(entries["Entry_time"])

# Merge entry price into stats
stats = stats.merge(entries, on="Entry_time", how="left")

missing_price = stats["entry_price"].isna().sum()
if missing_price > 0:
    print(f"  WARNING: {missing_price} trades missing entry price — they will have NaN entry_price feature.")

print(f"  Entry prices joined.")

# ── STEP 4: LOAD OHLCV ───────────────────────────────────────────────────────

print("Loading OHLCV data (this may take a moment for 15 years of 1-min data) ...")

ohlcv = pd.read_csv(OHLCV_FILE, sep="\t")

# Standardise column names
ohlcv = ohlcv.rename(columns={
    "<DATE>":    "date",
    "<TIME>":    "time",
    "<OPEN>":    "open",
    "<HIGH>":    "high",
    "<LOW>":     "low",
    "<CLOSE>":   "close",
    "<TICKVOL>": "volume",
    "<VOL>":     "vol",
    "<SPREAD>":  "spread",
})

ohlcv["timestamp"] = pd.to_datetime(
    ohlcv["date"].astype(str) + " " + ohlcv["time"].astype(str),
    format="%Y.%m.%d %H:%M:%S"
)
ohlcv = ohlcv.set_index("timestamp").sort_index()
ohlcv = ohlcv[["open", "high", "low", "close", "volume"]]

print(f"  OHLCV loaded: {len(ohlcv):,} bars from {ohlcv.index[0]} to {ohlcv.index[-1]}")

# ── STEP 5: FEATURE COMPUTATION ──────────────────────────────────────────────

def compute_atr(df, period=14):
    """Average True Range using Wilder's EMA."""
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def compute_features(entry_time, ohlcv_1m):
    """
    Given an entry timestamp, look back into 1-min OHLCV and return
    a dict of features computed strictly from pre-entry data.
    """
    feats = {}

    # Slice the lookback window (bars strictly before entry)
    window = ohlcv_1m[ohlcv_1m.index < entry_time].tail(LOOKBACK_BARS)

    if len(window) < 50:
        # Not enough history — return NaNs; these rows will be dropped later
        return None

    close  = window["close"]
    high   = window["high"]
    low    = window["low"]
    volume = window["volume"]

    # ── Volatility features ──────────────────────────────────────────────────

    atr_series = compute_atr(window, period=14)
    atr_now    = atr_series.iloc[-1]
    feats["atr_14"] = atr_now

    # ATR percentile rank: where does current ATR sit vs last 200 bars?
    atr_history = atr_series.tail(200)
    feats["atr_pctrank"] = float((atr_history < atr_now).mean())

    # Realised volatility: std of 1-min close-to-close returns
    rets = close.pct_change().dropna()
    feats["rvol_30"]  = rets.tail(30).std()
    feats["rvol_60"]  = rets.tail(60).std()
    feats["rvol_120"] = rets.tail(120).std()

    # ── Trend / momentum features ────────────────────────────────────────────

    ema20 = close.ewm(span=20,  adjust=False).mean()
    ema50 = close.ewm(span=50,  adjust=False).mean()
    ema200= close.ewm(span=200, adjust=False).mean()

    # EMA slopes (change over last 5 bars, normalised by price)
    price_now = close.iloc[-1]
    feats["ema20_slope"]  = (ema20.iloc[-1]  - ema20.iloc[-6])  / (price_now + 1e-9)
    feats["ema50_slope"]  = (ema50.iloc[-1]  - ema50.iloc[-6])  / (price_now + 1e-9)

    # Price position relative to EMAs
    feats["price_vs_ema20"]  = (price_now - ema20.iloc[-1])  / (atr_now + 1e-9)
    feats["price_vs_ema50"]  = (price_now - ema50.iloc[-1])  / (atr_now + 1e-9)
    feats["price_vs_ema200"] = (price_now - ema200.iloc[-1]) / (atr_now + 1e-9)

    # EMA alignment: 20 vs 50 (positive = uptrend stack)
    feats["ema20_vs_ema50"] = (ema20.iloc[-1] - ema50.iloc[-1]) / (atr_now + 1e-9)

    # ── Range / structure features ───────────────────────────────────────────

    # Where is price within the last 20-bar high/low range?  0=bottom, 1=top
    high20 = high.tail(20).max()
    low20  = low.tail(20).min()
    range20 = high20 - low20
    feats["price_in_range_20"] = (price_now - low20) / (range20 + 1e-9)

    # Same for 60-bar range
    high60 = high.tail(60).max()
    low60  = low.tail(60).min()
    range60 = high60 - low60
    feats["price_in_range_60"] = (price_now - low60) / (range60 + 1e-9)

    # Last bar's range relative to ATR (expansion/contraction signal)
    last_bar_range = high.iloc[-1] - low.iloc[-1]
    feats["bar_range_vs_atr"] = last_bar_range / (atr_now + 1e-9)

    # ── Volume features ──────────────────────────────────────────────────────

    vol_mean_20 = volume.tail(20).mean()
    vol_now     = volume.iloc[-1]
    feats["vol_ratio"] = vol_now / (vol_mean_20 + 1e-9)   # >1 = above-average volume

    # ── Time features (cyclical encoding) ───────────────────────────────────

    hour = entry_time.hour + entry_time.minute / 60.0
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feats["dow_sin"]  = np.sin(2 * np.pi * entry_time.dayofweek / 5)
    feats["dow_cos"]  = np.cos(2 * np.pi * entry_time.dayofweek / 5)

    return feats


# ── STEP 6: BUILD DATASET ROW BY ROW ─────────────────────────────────────────

print(f"Computing features for {len(stats)} trades ...")
print("  (This is the slow step — one lookback slice per trade)")

rows = []
skipped = 0

for i, trade in stats.iterrows():
    entry_time = trade["Entry_time"]

    feats = compute_features(entry_time, ohlcv)

    if feats is None:
        skipped += 1
        continue

    # Add label and metadata
    feats["entry_time"]    = entry_time
    feats["exit_time"]     = trade["Exit_time"]
    feats["entry_price"]   = trade.get("entry_price", np.nan)
    feats["mae"]           = trade["MAE"]
    feats["mfe"]           = trade["MFE"]
    feats["pnl"]           = trade["PNL"]
    feats["sl_distance"]   = trade["SL_distance"]
    feats["achievable_rr"] = trade["achievable_rr"]
    feats["rr_bucket"]     = trade["rr_bucket"]

    rows.append(feats)

    if (len(rows) % 500) == 0:
        print(f"  ... {len(rows)} trades processed")

print(f"  Done. {len(rows)} trades processed, {skipped} skipped (insufficient history).")

# ── STEP 7: SAVE ─────────────────────────────────────────────────────────────

dataset = pd.DataFrame(rows)

if dataset.empty:
    print("\nERROR: No trades were processed.")
    print("Most likely cause: the OHLCV file date range does not overlap with your trade dates.")
    print(f"  OHLCV range : {ohlcv.index[0]} → {ohlcv.index[-1]}")
    print(f"  Trade range : {stats['Entry_time'].min()} → {stats['Entry_time'].max()}")
    print("Make sure you are using your full OHLCV file, not the sample excerpt.")
    raise SystemExit(1)

dataset = dataset.sort_values("entry_time").reset_index(drop=True)

# Drop any rows with NaN features (shouldn't be many)
n_before = len(dataset)
dataset  = dataset.dropna(subset=[c for c in dataset.columns if c not in
                                  ["entry_time","exit_time","entry_price"]])
n_after  = len(dataset)
if n_before != n_after:
    print(f"  Dropped {n_before - n_after} rows with NaN features.")

dataset.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE}  ({len(dataset)} rows x {len(dataset.columns)} columns)")

# ── STEP 8: QUICK SUMMARY ────────────────────────────────────────────────────

print("\n── Dataset Summary ─────────────────────────────────────────────────────")
print(f"  Date range : {dataset['entry_time'].min()} → {dataset['entry_time'].max()}")
print(f"  Total trades: {len(dataset)}")
print(f"\n  Label distribution:")
for b in RR_LABELS:
    n   = (dataset["rr_bucket"] == b).sum()
    pct = n / len(dataset) * 100
    rr_lo = RR_BINS[b]
    rr_hi = RR_BINS[b+1] if RR_BINS[b+1] != np.inf else "∞"
    print(f"    Bucket {b}  (RR {rr_lo}–{rr_hi}): {n:4d} trades  ({pct:.1f}%)")

feature_cols = [c for c in dataset.columns if c not in
                ["entry_time","exit_time","entry_price","mae","mfe","pnl",
                 "sl_distance","achievable_rr","rr_bucket"]]
print(f"\n  Feature columns ({len(feature_cols)}):")
for fc in feature_cols:
    print(f"    {fc}")

print("\nDone! Next step: feed training_dataset.csv into XGBoost.")
print("See the train_model.py script for that.")
