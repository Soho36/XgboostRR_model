"""
per_hour_rr.py
==============
Re-examine the 5 RR exports at ENTRY-HOUR granularity, on the FULL independent
runs (not the matched panel that biased my earlier conclusion). For each hour
and each RR we report profit AND drawdown, so we can see whether low RR really
is preferred per-hour (as the user's MT5 window optimisation found).

Caveat kept honest: each RR file is one full-period run with its own single-slot
blocking, so an hour's subset here is a *diagnostic*, not the same as isolating
that hour in its own backtest. Directional corroboration only. Grid is coarse
(1.0..3.0) so it can separate 'low vs high', not resolve 1.05 vs 1.2.
"""
import glob, io, os, re
import numpy as np
import pandas as pd

INPUT_DIR, FILE_GLOB = "input_files", "trade_stats_rr_*.csv"
POINT_VALUE, LOTS = 2.0, 1.0
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
COL_MAP = {"entry_time": "entry_time", "trade_profit": "pnl", "pnl": "pnl",
           "candle_range": "sl", "sl": "sl"}


def load_rr(path):
    for enc in ["utf-16-le", "utf-16", "utf-8-sig", "utf-8"]:
        try:
            raw = open(path, "r", encoding=enc).read()
            for sep in ["\t", ",", ";"]:
                df = pd.read_csv(io.StringIO(raw), sep=sep)
                if df.shape[1] >= 5:
                    break
            else:
                continue
            break
        except Exception:
            continue
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={c: COL_MAP[c] for c in df.columns if c in COL_MAP})
    df = df[df["entry_time"].astype(str).str.match(r"^\d{4}\.\d{2}\.\d{2}")].copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], format="%Y.%m.%d %H:%M:%S")
    for c in ["pnl", "sl"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["sl"].notna() & (df["sl"] > 0) & df["pnl"].notna()].copy()
    df["R"] = df["pnl"] / (df["sl"] * POINT_VALUE * LOTS)
    df["hour"] = df["entry_time"].dt.hour
    return df.sort_values("entry_time").reset_index(drop=True)


def maxdd_R(r):
    eq = np.cumsum(np.asarray(r, float))
    return float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0


rr_data = {}
for p in sorted(glob.glob(os.path.join(INPUT_DIR, FILE_GLOB))):
    m = re.search(r"rr_([0-9.]+)\.csv$", os.path.basename(p))
    if m:
        rr_data[float(m.group(1))] = load_rr(p)
RRS = sorted(rr_data)

hours = sorted(set(pd.concat([d["hour"] for d in rr_data.values()]).unique()))
prof = pd.DataFrame(index=hours, columns=[f"$RR{rr}" for rr in RRS], dtype=float)
rdd  = pd.DataFrame(index=hours, columns=[f"R/DD{rr}" for rr in RRS], dtype=float)
ntr  = pd.Series(index=hours, dtype=int)

for h in hours:
    ntr[h] = (rr_data[RRS[0]]["hour"] == h).sum()
    for rr in RRS:
        sub = rr_data[rr][rr_data[rr]["hour"] == h]
        prof.loc[h, f"$RR{rr}"] = sub["pnl"].sum()
        dd = maxdd_R(sub["R"].values)
        rdd.loc[h, f"R/DD{rr}"] = (sub["R"].sum() / dd) if dd > 0 else np.nan

prof["BEST_$"] = prof[[f"$RR{rr}" for rr in RRS]].idxmax(axis=1).str.replace("$RR", "")
rdd["BEST_R/DD"] = rdd[[f"R/DD{rr}" for rr in RRS]].idxmax(axis=1).str.replace("R/DD", "")
prof["n"] = ntr

print("=" * 110)
print("PROFIT ($) by entry hour x RR   (full independent runs)")
print("=" * 110)
print(prof.round(0).to_string())
print("\n" + "=" * 110)
print("RETURN / DRAWDOWN by entry hour x RR   (higher = better risk-adjusted; the prop metric)")
print("=" * 110)
print(rdd.round(2).to_string())

# how often is a LOW rr (<=1.5) the risk-adjusted winner?
low_wins = rdd["BEST_R/DD"].astype(float) <= 1.5
print(f"\nHours where best RISK-ADJUSTED RR is <=1.5:  {low_wins.sum()} / {len(rdd)}")
print(f"Hours where best PROFIT RR is 3.0:           {(prof['BEST_$']=='3.0').sum()} / {len(prof)}")
