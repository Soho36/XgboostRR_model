"""
regime_rr_backtest.py
=====================
No ML. Tests one plain idea:

    "In an uptrend the trade can run further, so a bigger RR target pays.
     In a downtrend it can't, so a smaller RR target is better."

We turn that into a rule and check whether it beats a single fixed RR.

How it works
------------
1. Load the trades (MAE / MFE / SL / PNL per trade).
2. Load 1-min OHLCV, resample to 30-min, compute a 200-EMA trend filter.
   For every trade we look ONLY at the last *completed* 30-min bar before
   entry (no peeking into the future) and label the trade uptrend / downtrend.
3. For any candidate RR we can reconstruct what a trade would have paid,
   in R units (multiples of the initial risk = SL distance):
        MFE >= RR*SL              -> +RR      (take-profit hit)
        else MAE <= -SL           -> -1       (stop hit)
        else                      -> PNL/SL   (closed at the candle close)
   TP-priority when both are touched (optimistic); we also print the
   pessimistic SL-priority number so you can see the honest range.
4. Choose the best RR per trend-state on the TRAIN part only, then apply
   it to the TEST part and compare against fixed RR=1 (your current rule).

Everything is expressed in R so it is instrument- and size-independent.
"""

import pandas as pd
import numpy as np
import io

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRADE_STATS_FILE = "input_files/trade_stats.csv"
OHLCV_FILE       = "input_files/MT5_databento-ohlcv-1m.csv"

TREND_TF   = "30min"   # timeframe for the trend filter
EMA_PERIOD = 200       # EMA length on that timeframe
TRAIN_RATIO = 0.80     # first 80% of trades (by time) to pick the rule

# Candidate RR targets to search over
RR_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]


# ── LOAD TRADES ───────────────────────────────────────────────────────────────
def load_trade_stats(filepath):
    for enc in ["utf-16-le", "utf-16", "utf-8-sig", "utf-8"]:
        try:
            with open(filepath, "r", encoding=enc) as f:
                raw = f.read()
            df = pd.read_csv(io.StringIO(raw), sep="\t")
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f"Cannot read {filepath}")

    date_pat = r"^\d{4}\.\d{2}\.\d{2}"
    df = df[df["Entry_time"].astype(str).str.match(date_pat)]
    df["Entry_time"] = pd.to_datetime(df["Entry_time"], format="%Y.%m.%d %H:%M:%S")
    df = df[df["SL"].notna() & (df["SL"] > 0)].copy()
    return df.sort_values("Entry_time").reset_index(drop=True)


print("Loading trades ...")
trades = load_trade_stats(TRADE_STATS_FILE)
print(f"  {len(trades)} trades  ({trades['Entry_time'].min().date()} -> {trades['Entry_time'].max().date()})")


# ── LOAD OHLCV -> TREND STATE ─────────────────────────────────────────────────
print("Loading OHLCV (1-min) and building trend filter ...")
ohlcv = pd.read_csv(
    OHLCV_FILE, sep="\t",
    usecols=["<DATE>", "<TIME>", "<CLOSE>"],
    dtype={"<CLOSE>": "float32"},
)
ts = pd.to_datetime(ohlcv["<DATE>"] + " " + ohlcv["<TIME>"], format="%Y.%m.%d %H:%M:%S")
close_1m = pd.Series(ohlcv["<CLOSE>"].values, index=ts).sort_index()

# Resample to the trend timeframe. label/closed='right' => the bar's timestamp
# is its COMPLETION time, so merge_asof(backward) can never use a future bar.
htf_close = close_1m.resample(TREND_TF, label="right", closed="right").last().dropna()
ema = htf_close.ewm(span=EMA_PERIOD, adjust=False).mean()

trend = pd.DataFrame({
    "bar_close_time": htf_close.index,
    "htf_close": htf_close.values,
    "ema": ema.values,
})
trend["trend_up"] = trend["htf_close"] > trend["ema"]
trend = trend.iloc[EMA_PERIOD:]  # drop EMA warm-up bars
print(f"  {len(trend):,} completed {TREND_TF} bars, EMA{EMA_PERIOD} ready from {trend['bar_close_time'].iloc[0].date()}")

# Attach trend state to each trade using the last completed HTF bar (no lookahead)
trades = pd.merge_asof(
    trades, trend[["bar_close_time", "trend_up"]],
    left_on="Entry_time", right_on="bar_close_time",
    direction="backward",
)
before = len(trades)
trades = trades.dropna(subset=["trend_up"]).reset_index(drop=True)
trades["trend_up"] = trades["trend_up"].astype(bool)
print(f"  {len(trades)} trades kept ({before - len(trades)} dropped: before EMA warm-up)")


# ── PAYOFF RECONSTRUCTION (in R units) ────────────────────────────────────────
def outcome_R(df, rr, sl_priority=False):
    """R (risk-multiple) outcome for every trade at target `rr`, vectorised."""
    sl  = df["SL"].values
    mfe = df["MFE"].values
    mae = df["MAE"].values
    pnl = df["PNL"].values

    hit_tp = mfe >= rr * sl
    hit_sl = mae <= -sl

    out = pnl / sl                      # default: closed at the close
    out = np.where(hit_sl, -1.0, out)   # stop hit
    out = np.where(hit_tp, rr,   out)   # take-profit hit (TP-priority)
    if sl_priority:                     # both touched -> stop wins (pessimistic)
        both = hit_tp & hit_sl
        out = np.where(both, -1.0, out)
    return out


def summarise(r):
    r = np.asarray(r, float)
    wins, losses = r[r > 0], r[r < 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    return {
        "trades": len(r),
        "total_R": r.sum(),
        "avg_R": r.mean(),
        "win%": (r > 0).mean() * 100,
        "PF": pf,
    }


def show(title, r):
    s = summarise(r)
    print(f"  {title:<34} trades={s['trades']:>5}  totalR={s['total_R']:>8.1f}  "
          f"avgR={s['avg_R']:>6.3f}  win%={s['win%']:>5.1f}  PF={s['PF']:.2f}")


# ── STEP 1: DOES THE HYPOTHESIS EVEN HOLD? ────────────────────────────────────
print("\n" + "=" * 78)
print("STEP 1  Is 'uptrend runs further' actually true in the data? (full sample)")
print("=" * 78)
trades["achievable_rr"] = (trades["MFE"] / trades["SL"]).clip(lower=0)
for name, mask in [("UPTREND  ", trades["trend_up"]), ("DOWNTREND", ~trades["trend_up"])]:
    sub = trades[mask]
    print(f"  {name}: n={len(sub):>5}  "
          f"mean achievable RR (MFE/SL) = {sub['achievable_rr'].mean():.2f}   "
          f"median = {sub['achievable_rr'].median():.2f}")


# ── STEP 2: BEST FIXED RR PER STATE (full sample, for intuition) ──────────────
print("\n" + "=" * 78)
print("STEP 2  Best single RR target for each state (full sample, TP-priority)")
print("=" * 78)
best_full = {}
for name, mask in [("UPTREND", trades["trend_up"]), ("DOWNTREND", ~trades["trend_up"])]:
    sub = trades[mask]
    scores = {rr: outcome_R(sub, rr).sum() for rr in RR_GRID}
    best_rr = max(scores, key=scores.get)
    best_full[name] = best_rr
    grid_str = "  ".join(f"{rr}:{scores[rr]:.0f}" for rr in RR_GRID)
    print(f"  {name:<10} best RR = {best_rr}   (totalR by RR:  {grid_str})")


# ── STEP 3: HONEST TRAIN/TEST EVALUATION ──────────────────────────────────────
print("\n" + "=" * 78)
print("STEP 3  Pick the rule on TRAIN, judge it on unseen TEST")
print("=" * 78)
split = int(len(trades) * TRAIN_RATIO)
train, test = trades.iloc[:split], trades.iloc[split:]
print(f"  Train: {len(train)} trades ({train['Entry_time'].iloc[0].date()} -> {train['Entry_time'].iloc[-1].date()})")
print(f"  Test : {len(test)} trades ({test['Entry_time'].iloc[0].date()} -> {test['Entry_time'].iloc[-1].date()})")

# Learn best RR per state on TRAIN only
learned = {}
for state, mask_fn in [(True, lambda d: d["trend_up"]), (False, lambda d: ~d["trend_up"])]:
    sub = train[mask_fn(train)]
    scores = {rr: outcome_R(sub, rr).sum() for rr in RR_GRID}
    learned[state] = max(scores, key=scores.get)
print(f"  Learned rule:  uptrend -> RR {learned[True]}   |   downtrend -> RR {learned[False]}")

# Also find the best SINGLE global RR on train (the honest thing to beat)
global_scores = {rr: outcome_R(train, rr).sum() for rr in RR_GRID}
best_global = max(global_scores, key=global_scores.get)

# Apply everything to TEST
def regime_R(df, sl_priority=False):
    up = outcome_R(df[df["trend_up"]],  learned[True],  sl_priority)
    dn = outcome_R(df[~df["trend_up"]], learned[False], sl_priority)
    return np.concatenate([up, dn])

print("\n  --- TEST-SET RESULTS (optimistic, TP wins ties) ---")
show(f"Fixed RR=1.0 (your baseline)", outcome_R(test, 1.0))
show(f"Fixed RR={best_global} (best single, from train)", outcome_R(test, best_global))
show(f"Regime rule (up={learned[True]}, dn={learned[False]})", regime_R(test))

print("\n  --- Same on TEST but PESSIMISTIC (stop wins ties) ---")
show(f"Fixed RR=1.0", outcome_R(test, 1.0, sl_priority=True))
show(f"Fixed RR={best_global}", outcome_R(test, best_global, sl_priority=True))
show(f"Regime rule", regime_R(test, sl_priority=True))

print("\nDone.")
