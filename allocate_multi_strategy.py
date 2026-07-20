"""
allocate_multi_strategy.py
==========================
Pick WHICH windows to trade (from RR + GG) and WHICH account each goes to.

Why this differs from allocate_accounts.py: there we had 12 windows and 12+ slots,
so every window was traded and total profit was constant -> pure risk-shuffling.
Now we have 19 candidate windows (12 RR + 7 GG) but only 7 accounts x 2 = 14
slots, so we must SELECT. Profit is no longer constant, so the objective becomes:

    maximise total net profit
    subject to  every account's drawdown <= (cap fraction) x its own DD limit

CONSTRAINT — strategy-pure accounts: an account runs ONE script, and the broker
account is NETTING, so two scripts on one account would produce conflicting
orders. Therefore, every window on a given account must come from the same
strategy. (Day-alternating between strategies is possible in principle but each
window would then only trade ~half the days, losing profit, so it is not modelled.)

Solved EXACTLY with scipy.optimize.milp (integer program), not a heuristic.

INPUT : output_files/RR_maemfe_combined_trades.csv
        output_files/GG_maemfe_combined_trades.csv   (from analyze_maemfe.py)
OUTPUT: output_files/multi_strategy_allocation.csv
"""

import itertools
import os

import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

# ---- CONFIG -----------------------------------------------------------------
TRADE_FILES = {
    "RR": "output_files/RR_maemfe_combined_trades.csv",
    "GG": "output_files/GG_maemfe_combined_trades.csv",
}
OUT_CSV = "output_files/multi_strategy_allocation.csv"

ACCOUNT_LIMITS = [1500.0] * 4 + [2000.0] * 3
MAX_WINDOWS_PER_ACCOUNT = 2
# Fraction of each account's limit the historical DD may use. Lower = more
# future headroom but less profit. We solve the whole frontier.
CAP_FRACTIONS = [1.00, 0.90, 0.85, 0.80, 0.70]
REPORT_FRACTION = 0.85          # which one to print in full / save

NA = len(ACCOUNT_LIMITS)
LIMITS = np.array(ACCOUNT_LIMITS)


# ---- LOAD -------------------------------------------------------------------
frames = []
for strat, path in TRADE_FILES.items():
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path} — run analyze_maemfe.py first.")
    d = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    d["strategy"] = strat
    frames.append(d)
T = pd.concat(frames, ignore_index=True).sort_values("exit_time").reset_index(drop=True)

# candidate windows keyed by (strategy, window)
KEYS = sorted(
    T.groupby(["strategy", "window"]).groups.keys(),
    key=lambda k: (k[0], int(k[1].split("-")[0])),
)
NW = len(KEYS)
K_IDX = {k: i for i, k in enumerate(KEYS)}
T["kidx"] = list(zip(T["strategy"], T["window"]))
T["kidx"] = T["kidx"].map(K_IDX)
kidx = T["kidx"].to_numpy()
net = T["net"].to_numpy(float)
mae = T["mae"].to_numpy(float)
RR_OF = {K_IDX[k]: T.loc[T["kidx"] == K_IDX[k], "RR"].iloc[0] for k in KEYS}

print(f"{len(T)} trades | {NW} candidate windows "
      f"({sum(1 for k in KEYS if k[0]=='RR')} RR + {sum(1 for k in KEYS if k[0]=='GG')} GG)"
      f" | {NA} accounts, {NA*MAX_WINDOWS_PER_ACCOUNT} slots")


def dd_floating(order):
    """Max DD including open floating loss (MAE), trades in exit-time order."""
    equity = peak = maxdd = 0.0
    for n, m in zip(net[order], mae[order]):
        maxdd = max(maxdd, peak - (equity + min(m, 0.0)))
        equity += n
        peak = max(peak, equity)
        maxdd = max(maxdd, peak - equity)
    return maxdd


# ---- CANDIDATE GROUPS (1-2 windows, SAME strategy) --------------------------
groups = []          # (tuple_of_window_idx, dd, profit, strategy)
by_strategy = {}
for i, (s, w) in enumerate(KEYS):
    by_strategy.setdefault(s, []).append(i)

for s, idxs in by_strategy.items():
    for size in range(1, MAX_WINDOWS_PER_ACCOUNT + 1):
        for combo in itertools.combinations(idxs, size):
            sel = np.isin(kidx, combo)          # already in exit_time order
            order = np.flatnonzero(sel)
            groups.append((combo, dd_floating(order), float(net[order].sum()), s))

print(f"{len(groups)} candidate groups (1-2 windows, strategy-pure)")


def solve(cap_fraction):
    """Exact ILP: max profit s.t. each account <= cap_fraction * its limit."""
    caps = LIMITS * cap_fraction
    # variables x[g, a] only where the group fits that account
    var = [(gi, a) for gi, g in enumerate(groups) for a in range(NA) if g[1] <= caps[a]]
    if not var:
        return None
    n = len(var)
    c = np.array([-groups[gi][2] for gi, _ in var])          # maximise profit

    rows, lb, ub = [], [], []
    # each account gets exactly one group
    for a in range(NA):
        r = np.array([1.0 if aa == a else 0.0 for _, aa in var])
        rows.append(r); lb.append(1); ub.append(1)
    # each window used at most once
    for w in range(NW):
        r = np.array([1.0 if w in groups[gi][0] else 0.0 for gi, _ in var])
        rows.append(r); lb.append(0); ub.append(1)

    res = milp(c=c, constraints=LinearConstraint(np.array(rows), lb, ub),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    if not res.success:
        return None
    chosen = [var[i] for i in np.flatnonzero(np.round(res.x) == 1)]
    return chosen


def describe(chosen, cap_fraction):
    rows, used = [], set()
    for gi, a in sorted(chosen, key=lambda t: t[1]):
        combo, dd, profit, s = groups[gi]
        used.update(combo)
        wins = [f"{KEYS[i][1]}@{RR_OF[i]:g}" for i in combo]
        hours = sorted(int(KEYS[i][1].split("-")[0]) for i in combo)
        adjacent = len(hours) == 2 and hours[1] - hours[0] == 1
        rows.append({
            "account": f"A{a+1}", "DD_limit": int(LIMITS[a]), "strategy": s,
            "windows": ", ".join(wins), "net_profit": round(profit),
            "maxDD": round(dd), "used_%_of_limit": round(dd / LIMITS[a] * 100, 1),
            "headroom": round(LIMITS[a] - dd),
            "adjacent_hours": "yes" if adjacent else "",
        })
    return pd.DataFrame(rows), used


# ---- FRONTIER ---------------------------------------------------------------
print("\n" + "=" * 96)
print("PROFIT vs SAFETY FRONTIER  (exact optimum at each cap)")
print("=" * 96)
print(f"{'cap':>6}  {'$1500->':>8} {'$2000->':>8}  {'total net profit':>17}  {'windows used':>13}")
frontier = {}
for f in CAP_FRACTIONS:
    ch = solve(f)
    if ch is None:
        print(f"{f*100:5.0f}%  {'':>8} {'':>8}  {'INFEASIBLE':>17}")
        continue
    df, used = describe(ch, f)
    frontier[f] = (ch, df, used)
    print(f"{f*100:5.0f}%  {1500*f:>8.0f} {2000*f:>8.0f}  "
          f"{df['net_profit'].sum():>17,.0f}  {len(used):>10}/{NW}")

# ---- DETAIL AT THE REPORTING CAP --------------------------------------------
if REPORT_FRACTION not in frontier:
    REPORT_FRACTION = max(frontier)
chosen, A, used = frontier[REPORT_FRACTION]
print("\n" + "=" * 96)
print(f"ALLOCATION AT {REPORT_FRACTION*100:.0f}% CAP")
print("=" * 96)
print(A.to_string(index=False))
print(f"\nTotal net profit: ${A['net_profit'].sum():,.0f}   "
      f"worst account at {A['used_%_of_limit'].max():.1f}% of its limit")

dropped = [KEYS[i] for i in range(NW) if i not in used]
print(f"\nDropped windows ({len(dropped)}):")
for s, w in dropped:
    i = K_IDX[(s, w)]
    sel = np.flatnonzero(kidx == i)
    print(f"   {s} {w:<6} RR {RR_OF[i]:<5g} net ${net[sel].sum():>7,.0f}  "
          f"DD ${dd_floating(sel):>6,.0f}")

if (A["adjacent_hours"] == "yes").any():
    print("\nNOTE: accounts flagged 'adjacent_hours' pair back-to-back hours; one EA holds")
    print("one position, so a trade running long can block the next window's entry.")
    print("Our per-window data was recorded in isolation, so that effect is not modelled.")

os.makedirs("output_files", exist_ok=True)
try:
    A.to_csv(OUT_CSV, index=False)
except PermissionError:
    import time
    OUT_CSV = OUT_CSV.replace(".csv", f"_{time.strftime('%H%M%S')}.csv")
    A.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV}")
print("\nEach account is strategy-pure (one script, netting-safe). Add accounts by")
print("extending ACCOUNT_LIMITS — the optimiser will pull in the dropped windows.")
