"""
build_training_dataset_v2.py
============================
Candle-based feature pipeline for the Red Candle Breakout strategy.

Strategy recap:
  - A red candle forms (close < open)
  - Buy stop placed at candle HIGH
  - SL placed at candle LOW
  - SL_distance = candle HIGH - LOW  (= candle range)
  - achievable_rr = MFE / SL_distance

Features describe ONLY what a trader can see at the moment the entry triggers:
  - The triggering red candle's own shape/size
  - The N candles immediately before it (context/momentum)
  - Structure: where is price relative to recent swing high
  - All values normalized — no absolute prices

Output: training_dataset_v2.csv
"""

import pandas as pd
import numpy as np
import io

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

TRADE_STATS_FILE = "trade_stats.csv"
OHLCV_FILE = "MT5_databento-ohlcv-1m.csv"
OUTPUT_FILE = "training_dataset_v2.csv"

# RR bucket boundaries
RR_BINS = [0.0, 1.0, 2.0, 3.0, np.inf]
RR_LABELS = [0, 1, 2, 3]

# How many candles BEFORE the trigger candle to use as context
CONTEXT_BARS = 20  # gives enough for ATR and swing detection

# ── LOAD TRADE STATS ──────────────────────────────────────────────────────────

print("Loading trade_stats.csv ...")
with open(TRADE_STATS_FILE, "r", encoding="utf-16-le") as f:
    content = f.read()
stats = pd.read_csv(io.StringIO(content), sep="\t")
stats = stats.rename(columns={"Unnamed: 6": "SL_distance"})
stats["Entry_time"] = pd.to_datetime(stats["Entry_time"], format="%Y.%m.%d %H:%M:%S")
stats = stats[stats["SL_distance"].notna() & (stats["SL_distance"] > 0)].copy()
print(f"  {len(stats)} trades loaded.")

# ── LABELS ────────────────────────────────────────────────────────────────────

print("Computing RR labels ...")
stats["achievable_rr"] = (stats["MFE"] / stats["SL_distance"]).clip(lower=0)
stats["rr_bucket"] = pd.cut(
    stats["achievable_rr"], bins=RR_BINS, labels=RR_LABELS, right=False
).astype(int)

for b in RR_LABELS:
    n = (stats["rr_bucket"] == b).sum()
    print(f"  Bucket {b}: {n} trades ({n / len(stats) * 100:.1f}%)")

# ── LOAD OHLCV ────────────────────────────────────────────────────────────────

print("\nLoading OHLCV ...")
ohlcv = pd.read_csv(OHLCV_FILE, sep="\t")
ohlcv = ohlcv.rename(columns={
    "<DATE>": "date", "<TIME>": "time",
    "<OPEN>": "open", "<HIGH>": "high", "<LOW>": "low",
    "<CLOSE>": "close", "<TICKVOL>": "volume"
})
ohlcv["timestamp"] = pd.to_datetime(
    ohlcv["date"].astype(str) + " " + ohlcv["time"].astype(str),
    format="%Y.%m.%d %H:%M:%S"
)
ohlcv = ohlcv.set_index("timestamp").sort_index()
ohlcv = ohlcv[["open", "high", "low", "close", "volume"]]
print(f"  {len(ohlcv):,} bars: {ohlcv.index[0]} → {ohlcv.index[-1]}")


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def atr(bars, period=14):
    """ATR using Wilder EMA."""
    tr = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - bars["close"].shift(1)).abs(),
        (bars["low"] - bars["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().iloc[-1]


def swing_high(bars, lookback=10):
    """Highest high in the lookback window."""
    return bars["high"].tail(lookback).max()


def count_consecutive(series, condition_fn):
    """
    Count how many bars from the END of `series` satisfy condition_fn
    without interruption.
    Example: consecutive red candles = count_consecutive(bars, lambda r: r['close'] < r['open'])
    """
    count = 0
    for _, row in series.iloc[::-1].iterrows():
        if condition_fn(row):
            count += 1
        else:
            break
    return count


# ── FEATURE COMPUTATION ───────────────────────────────────────────────────────

def compute_features(entry_time, ohlcv):
    """
    All features computed from candles visible BEFORE entry fires.

    The trigger candle is the last fully closed 1-min bar before entry_time.
    Context candles are the CONTEXT_BARS bars before that.
    """

    # Slice: all bars up to (not including) the entry bar itself
    # The entry bar opens after the trigger candle closes
    pre_entry = ohlcv[ohlcv.index < entry_time]

    if len(pre_entry) < CONTEXT_BARS + 5:
        return None

    # ── Trigger candle (the red candle that set up the trade) ────────────────
    tc = pre_entry.iloc[-1]  # last closed bar before entry

    tc_open = tc["open"]
    tc_close = tc["close"]
    tc_high = tc["high"]
    tc_low = tc["low"]
    tc_range = tc_high - tc_low  # = SL_distance

    if tc_range == 0:
        return None

    tc_body = abs(tc_open - tc_close)
    tc_upper_wick = tc_high - max(tc_open, tc_close)
    tc_lower_wick = min(tc_open, tc_close) - tc_low

    # ── Context window (bars before the trigger candle) ──────────────────────
    ctx = pre_entry.iloc[-(CONTEXT_BARS + 1):-1]  # CONTEXT_BARS bars before tc

    # ATR of context window — our normalization unit
    atr_val = atr(ctx, period=min(14, len(ctx) - 1))
    if atr_val == 0:
        return None

    feats = {}

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 1: Trigger candle shape
    # These describe the exact candle the strategy is based on.
    # ════════════════════════════════════════════════════════════════════════

    # Body as fraction of total range: 1.0 = full marubozu, 0 = doji
    feats["tc_body_ratio"] = tc_body / tc_range

    # Wick ratios: where did price reject within the candle?
    feats["tc_upper_wick_ratio"] = tc_upper_wick / tc_range
    feats["tc_lower_wick_ratio"] = tc_lower_wick / tc_range

    # Is this candle large or small vs recent volatility?
    # >1 = bigger than average, <1 = smaller
    feats["tc_size_vs_atr"] = tc_range / atr_val

    # Is it bigger or smaller than the 3 candles just before it?
    ctx3_avg_range = (ctx["high"] - ctx["low"]).tail(3).mean()
    feats["tc_size_vs_prior3"] = tc_range / (ctx3_avg_range + 1e-9)

    # Volume on the trigger candle vs 20-bar average
    ctx_vol_avg = ctx["volume"].mean()
    feats["tc_vol_ratio"] = tc["volume"] / (ctx_vol_avg + 1e-9)

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 2: The candles immediately before — momentum/context
    # ════════════════════════════════════════════════════════════════════════

    # Direction of each of the 5 candles before trigger: +1 bull, -1 bear
    # These capture the "is this a deep pullback or a one-candle dip" question
    for i, n in enumerate([1, 2, 3, 4, 5]):
        bar = ctx.iloc[-n] if len(ctx) >= n else None
        if bar is not None:
            feats[f"prior_{n}_direction"] = 1.0 if bar["close"] >= bar["open"] else -1.0
            feats[f"prior_{n}_size_vs_atr"] = (bar["high"] - bar["low"]) / atr_val
        else:
            feats[f"prior_{n}_direction"] = 0.0
            feats[f"prior_{n}_size_vs_atr"] = 0.0

    # How many consecutive red candles immediately before the trigger?
    # (a long red run = deep momentum; 0 = isolated red candle after greens)
    feats["consecutive_red_before"] = count_consecutive(
        ctx, lambda r: r["close"] < r["open"]
    )

    # Average direction of last 5 context bars: +1 = all green, -1 = all red
    last5 = ctx.tail(5)
    feats["prior5_avg_direction"] = (
        (last5["close"] - last5["open"]).apply(np.sign).mean()
    )

    # Momentum: did the candles before trigger close progressively lower?
    # (slope of closes, normalised)
    if len(ctx) >= 5:
        closes5 = ctx["close"].tail(5).values
        slope = np.polyfit(range(5), closes5, 1)[0]
        feats["prior5_close_slope"] = slope / atr_val
    else:
        feats["prior5_close_slope"] = 0.0

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 3: Structure — where is price relative to recent key levels?
    # ════════════════════════════════════════════════════════════════════════

    # Distance from entry (trigger candle HIGH) to recent swing high
    # If the last swing high is just above entry, MFE is capped
    sh10 = swing_high(ctx, lookback=10)
    sh20 = swing_high(ctx, lookback=20)

    feats["dist_to_swing_high_10"] = (sh10 - tc_high) / atr_val
    feats["dist_to_swing_high_20"] = (sh20 - tc_high) / atr_val

    # How far has price pulled back to form this trigger candle?
    # Deep pullbacks from a swing high = more room to recover
    feats["pullback_depth"] = (sh20 - tc_close) / atr_val

    # Where is the trigger candle's HIGH within the recent 20-bar range?
    # 0 = at the bottom, 1 = at the top
    range_high = ctx["high"].tail(20).max()
    range_low = ctx["low"].tail(20).min()
    range_span = range_high - range_low
    feats["entry_in_range_20"] = (tc_high - range_low) / (range_span + 1e-9)

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 4: Volatility regime at time of entry
    # ════════════════════════════════════════════════════════════════════════

    # ATR percentile: is current volatility high or low vs recent history?
    # Use 60-bar ATR history
    if len(pre_entry) >= 74:
        atr_history = []
        for j in range(60):
            window = pre_entry.iloc[-(75 + j):-(14 + j)]
            if len(window) >= 15:
                atr_history.append(atr(window, period=14))
        if atr_history:
            feats["atr_pctrank"] = float((np.array(atr_history) < atr_val).mean())
        else:
            feats["atr_pctrank"] = 0.5
    else:
        feats["atr_pctrank"] = 0.5

    # ════════════════════════════════════════════════════════════════════════
    # GROUP 5: Time of day (cyclical)
    # Kept because session timing genuinely affects how far breakouts run
    # ════════════════════════════════════════════════════════════════════════

    hour = entry_time.hour + entry_time.minute / 60.0
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    feats["dow_sin"] = np.sin(2 * np.pi * entry_time.dayofweek / 5)
    feats["dow_cos"] = np.cos(2 * np.pi * entry_time.dayofweek / 5)

    return feats


# ── BUILD DATASET ─────────────────────────────────────────────────────────────

print(f"\nComputing candle features for {len(stats)} trades ...")

rows = []
skipped = 0

for i, trade in stats.iterrows():
    entry_time = trade["Entry_time"]
    feats = compute_features(entry_time, ohlcv)

    if feats is None:
        skipped += 1
        continue

    feats["entry_time"] = entry_time
    feats["mae"] = trade["MAE"]
    feats["mfe"] = trade["MFE"]
    feats["pnl"] = trade["PNL"]
    feats["sl_distance"] = trade["SL_distance"]
    feats["achievable_rr"] = trade["achievable_rr"]
    feats["rr_bucket"] = trade["rr_bucket"]
    rows.append(feats)

    if len(rows) % 1000 == 0:
        print(f"  ... {len(rows)} trades done")

print(f"  Done. {len(rows)} processed, {skipped} skipped.")

# ── SAVE ──────────────────────────────────────────────────────────────────────

dataset = pd.DataFrame(rows)

if dataset.empty:
    print("\nERROR: No trades processed — OHLCV date range likely doesn't overlap trades.")
    print(f"  OHLCV: {ohlcv.index[0]} → {ohlcv.index[-1]}")
    print(f"  Trades: {stats['Entry_time'].min()} → {stats['Entry_time'].max()}")
    raise SystemExit(1)

dataset = dataset.sort_values("entry_time").reset_index(drop=True)
dataset.to_csv(OUTPUT_FILE, index=False)

feature_cols = [c for c in dataset.columns if c not in
                ["entry_time", "mae", "mfe", "pnl", "sl_distance", "achievable_rr", "rr_bucket"]]

print(f"\nSaved: {OUTPUT_FILE}")
print(f"  {len(dataset)} rows  x  {len(dataset.columns)} columns")
print(f"  {len(feature_cols)} features:")

# Print features grouped
groups = {
    "Trigger candle shape": [f for f in feature_cols if f.startswith("tc_")],
    "Prior candle context": [f for f in feature_cols if f.startswith("prior") or f.startswith("consecutive")],
    "Structure / levels": [f for f in feature_cols if "swing" in f or "pullback" in f or "range" in f],
    "Volatility regime": [f for f in feature_cols if "atr" in f],
    "Time": [f for f in feature_cols if "hour" in f or "dow" in f],
}
for group, cols in groups.items():
    print(f"\n  [{group}]")
    for c in cols:
        print(f"    {c}")

print("\nDone! Now run train_model.py (no changes needed there).")
