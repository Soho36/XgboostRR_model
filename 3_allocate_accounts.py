"""
allocate_multi_strategy.py
==========================
Pick WHICH windows to trade (from RR + GG) and WHICH account each goes to.

Why this differs from allocate_accounts.py: there we had 12 windows and 12+
slots, so every window was traded and total profit was constant. Now we have
more candidate windows than account slots, so profit is no longer constant:

    maximize total net profit
    subject to every account's drawdown <= cap_fraction * available DD

Candidate account groups are scored with an account-level one-position replay
by default. That means if a 2-3 trade is still open when a 3-4 signal arrives,
the later entry is skipped before profit and drawdown are measured.

INPUT : output_files/RR_maemfe_combined_trades.csv
        output_files/GG_maemfe_combined_trades.csv   (from analyze_maemfe.py)
OUTPUT: output_files/multi_strategy_allocation.csv
"""

from functools import lru_cache
import itertools
import os

import numpy as np
import pandas as pd

import provenance as prov

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
except ImportError:
    Bounds = LinearConstraint = milp = None


# ---- CONFIG -----------------------------------------------------------------
TRADE_FILES = {
    "RR": "data/3_results/RR_maemfe_combined_trades.csv",
    "GG": "data/3_results/GG_maemfe_combined_trades.csv",
}
OUT_CSV = "data/3_results/multi_strategy_allocation.csv"

ACCOUNT_NAMES = [
    "PA-08-1500",
    "PA-09-1500",
    "PA-10-1500",
    "PA-11-1500",
    "PA-12-2000",
    "PA-13-2000",
    "PA-14-2000",
]
ACCOUNT_LIMITS = [1500.0] * 4 + [2000.0] * 3

# Override this when an account has already used some trailing DD. The cap
# frontier is applied to this available amount, not blindly to the nominal limit.
ACCOUNT_DD_AVAILABLE = ACCOUNT_LIMITS.copy()

MAX_WINDOWS_PER_ACCOUNT = 2

# Keep this False while one broker account can safely run only one strategy EA.
# If you later build one unified RR+GG controller, switch it to True and compare.
ALLOW_MIXED_STRATEGIES = False

# True = simulate one open position per account group, skipping entries that
# arrive before the previous accepted trade closes.
REPLAY_ONE_POSITION = True

# Optional hard operational rule. Leave False if blocked entries are acceptable
# and you want the optimizer to price them in; set True to disallow 2-3 + 3-4.
FORBID_ADJACENT_WINDOWS = False

# Fraction of each account's available DD the historical DD may use. Lower =
# more future headroom but less profit. We solve the whole frontier.
CAP_FRACTIONS = [1.00, 0.90, 0.85, 0.80, 0.70]
REPORT_FRACTION = 0.85


NA = len(ACCOUNT_LIMITS)
if len(ACCOUNT_NAMES) != NA:
    raise SystemExit(
        f"ACCOUNT_NAMES has {len(ACCOUNT_NAMES)} entries but ACCOUNT_LIMITS has {NA}."
    )
if len(ACCOUNT_DD_AVAILABLE) != NA:
    raise SystemExit(
        "ACCOUNT_DD_AVAILABLE must have one entry for each ACCOUNT_LIMITS entry."
    )

LIMITS = np.array(ACCOUNT_LIMITS, dtype=float)
AVAILABLE_DD = np.array(ACCOUNT_DD_AVAILABLE, dtype=float)


# ---- LOAD -------------------------------------------------------------------
frames = []
for strat, path in TRADE_FILES.items():
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path} - run analyze_maemfe.py first.")
    d = pd.read_csv(path, parse_dates=["entry_time", "exit_time"])
    d["strategy"] = strat
    frames.append(d)
T = pd.concat(frames, ignore_index=True).sort_values("exit_time").reset_index(drop=True)

# DD calibration from step 1 (see combo_factor). Without it the RR picks would be
# cautious while this script's hard DD constraint stayed optimistic.
CALIB = {}
_cal = "data/3_results/dd_calibration.csv"
if os.path.exists(_cal):
    try:
        _c = pd.read_csv(_cal)
        CALIB = {(r.strategy, r.window, round(float(r.RR), 2)): float(r.dd_factor)
                 for r in _c.itertuples(index=False)}
    except Exception as _e:
        print(f"  (could not read {_cal}: {_e})")
print(f"DD calibration: {len(CALIB)} pass(es) scaled up"
      + (f" (max x{max(CALIB.values()):.3f})" if CALIB else " — none needed"))

KEYS = sorted(
    T.groupby(["strategy", "window"]).groups.keys(),
    key=lambda k: (k[0], int(k[1].split("-")[0])),
)
NW = len(KEYS)
K_IDX = {k: i for i, k in enumerate(KEYS)}
T["kidx"] = list(zip(T["strategy"], T["window"]))
T["kidx"] = T["kidx"].map(K_IDX)
RR_OF = {K_IDX[k]: T.loc[T["kidx"] == K_IDX[k], "RR"].iloc[0] for k in KEYS}

print(
    f"{len(T)} trades | {NW} candidate windows "
    f"({sum(1 for k in KEYS if k[0] == 'RR')} RR + "
    f"{sum(1 for k in KEYS if k[0] == 'GG')} GG) "
    f"| {NA} accounts, {NA * MAX_WINDOWS_PER_ACCOUNT} slots"
)
print(
    f"Group scoring: {'one-position replay' if REPLAY_ONE_POSITION else 'isolated add-up'}"
)
print(
    f"Strategy mixing: {'allowed' if ALLOW_MIXED_STRATEGIES else 'disabled'}"
)
print(f"Adjacent-hour pairs: {'forbidden' if FORBID_ADJACENT_WINDOWS else 'allowed'}")
print(f"Solver: {'scipy.optimize.milp' if milp is not None else 'exact DP fallback'}")


# ---- GROUP METRICS ----------------------------------------------------------
def max_dd_floating(df):
    """Max EQUITY DD including open floating P/L in BOTH directions.

    Tracking the intra-trade peak (MFE) as well as the trough (MAE) reproduces
    MT5's STAT_EQUITY_DD exactly; MAE alone understates it (a trade that runs
    to +MFE then closes lower is a real equity give-back a prop firm counts).
    """
    equity = peak = maxdd = 0.0
    for row in df.sort_values("exit_time").itertuples(index=False):
        # MAE before MFE — validated exact against MT5 on 12 passes.
        maxdd = max(maxdd, peak - (equity + min(float(row.mae), 0.0)))
        peak = max(peak, equity + max(float(row.mfe), 0.0))
        equity += float(row.net)
        peak = max(peak, equity)
        maxdd = max(maxdd, peak - equity)
    return float(maxdd)


def replay_one_position(df):
    """Keep the first signal while flat; skip later signals until that trade exits."""
    accepted_idx = []
    open_until = pd.Timestamp.min
    ordered = df.sort_values(["entry_time", "exit_time", "strategy", "window"])
    for idx, row in ordered.iterrows():
        if row["entry_time"] >= open_until:
            accepted_idx.append(idx)
            open_until = row["exit_time"]
    return df.loc[accepted_idx].copy()


def combo_mask(combo):
    return sum(1 << i for i in combo)


def combo_strategy(combo):
    strategies = sorted({KEYS[i][0] for i in combo})
    return strategies[0] if len(strategies) == 1 else "MIXED"


def has_adjacent_hours(combo):
    hours = sorted(int(KEYS[i][1].split("-")[0]) for i in combo)
    return any(b - a == 1 for a, b in zip(hours, hours[1:]))


def combo_trade_frame(combo):
    return T[T["kidx"].isin(combo)].copy()


def combo_factor(combo):
    """Worst DD calibration factor among the windows on this account.

    Step 1 measured, per (strategy, window, RR), how much our equity-DD walk
    understated MT5's true STAT_EQUITY_DD. MT5 only ever tested single windows,
    so there is no ground truth for a COMBINATION — taking the worst
    constituent factor is the conservative choice for a hard DD limit.
    """
    return max([CALIB.get((KEYS[i][0], KEYS[i][1], round(float(RR_OF[i]), 2)), 1.0)
                for i in combo] or [1.0])


def combo_metrics(combo):
    isolated = combo_trade_frame(combo)
    effective = replay_one_position(isolated) if REPLAY_ONE_POSITION else isolated
    fac = combo_factor(combo)
    return {
        "combo": tuple(combo),
        "mask": combo_mask(combo),
        "strategy": combo_strategy(combo),
        "profit": float(effective["net"].sum()),
        "dd": max_dd_floating(effective) * fac,
        "trades": int(len(effective)),
        "blocked_trades": int(len(isolated) - len(effective)),
        "isolated_profit": float(isolated["net"].sum()),
        "isolated_dd": max_dd_floating(isolated) * fac,
        "dd_factor": fac,
    }


def candidate_pools():
    if ALLOW_MIXED_STRATEGIES:
        return [list(range(NW))]
    by_strategy = {}
    for i, (strategy, _window) in enumerate(KEYS):
        by_strategy.setdefault(strategy, []).append(i)
    return list(by_strategy.values())


groups = []
for pool in candidate_pools():
    max_size = min(MAX_WINDOWS_PER_ACCOUNT, len(pool))
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(pool, size):
            if FORBID_ADJACENT_WINDOWS and has_adjacent_hours(combo):
                continue
            groups.append(combo_metrics(combo))

print(
    f"{len(groups)} candidate groups "
    f"(1-{MAX_WINDOWS_PER_ACCOUNT} windows"
    f"{'' if ALLOW_MIXED_STRATEGIES else ', strategy-pure'})"
)


# ---- EXACT SOLVERS ----------------------------------------------------------
def solve_with_milp(cap_fraction):
    """Exact ILP: max profit s.t. each account <= cap_fraction * available DD."""
    caps = AVAILABLE_DD * cap_fraction
    var = [
        (gi, account_idx)
        for gi, group in enumerate(groups)
        for account_idx in range(NA)
        if group["dd"] <= caps[account_idx]
    ]
    if not var:
        return None

    n = len(var)
    c = np.array([-groups[gi]["profit"] for gi, _account_idx in var])

    rows, lb, ub = [], [], []
    for account_idx in range(NA):
        r = np.array([1.0 if a == account_idx else 0.0 for _gi, a in var])
        rows.append(r)
        lb.append(1)
        ub.append(1)

    for window_idx in range(NW):
        bit = 1 << window_idx
        r = np.array([1.0 if groups[gi]["mask"] & bit else 0.0 for gi, _a in var])
        rows.append(r)
        lb.append(0)
        ub.append(1)

    res = milp(
        c=c,
        constraints=LinearConstraint(np.array(rows), lb, ub),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
    )
    if not res.success:
        return None
    return [var[i] for i in np.flatnonzero(res.x > 0.5)]


def solve_with_dp(cap_fraction):
    """Exact fallback solver for environments without scipy."""
    caps = AVAILABLE_DD * cap_fraction

    @lru_cache(maxsize=None)
    def dp(account_idx, used_mask):
        if account_idx == NA:
            return 0.0, ()

        remaining_accounts = NA - account_idx
        unused_windows = NW - bin(used_mask).count("1")
        if unused_windows < remaining_accounts:
            return -np.inf, None

        best_profit, best_choice = -np.inf, None
        for gi, group in enumerate(groups):
            if group["mask"] & used_mask:
                continue
            if group["dd"] > caps[account_idx]:
                continue
            child_profit, child_choice = dp(account_idx + 1, used_mask | group["mask"])
            if child_choice is None:
                continue
            profit = group["profit"] + child_profit
            if profit > best_profit:
                best_profit = profit
                best_choice = ((gi, account_idx),) + child_choice
        return best_profit, best_choice

    _profit, choice = dp(0, 0)
    return list(choice) if choice is not None else None


def solve(cap_fraction):
    if milp is not None:
        return solve_with_milp(cap_fraction)
    return solve_with_dp(cap_fraction)


# ---- REPORTING --------------------------------------------------------------
def format_window(window_idx, include_strategy):
    strategy, window = KEYS[window_idx]
    prefix = f"{strategy} " if include_strategy else ""
    return f"{prefix}{window}@{RR_OF[window_idx]:g}"


def describe(chosen):
    rows, used = [], set()
    for gi, account_idx in sorted(chosen, key=lambda t: t[1]):
        group = groups[gi]
        combo = group["combo"]
        used.update(combo)
        include_strategy = group["strategy"] == "MIXED"
        wins = [format_window(i, include_strategy) for i in combo]
        adjacent = has_adjacent_hours(combo)
        rows.append(
            {
                "account": ACCOUNT_NAMES[account_idx],
                "DD_limit": int(LIMITS[account_idx]),
                "DD_available": int(AVAILABLE_DD[account_idx]),
                "strategy": group["strategy"],
                "windows": ", ".join(wins),
                "net_profit": round(group["profit"]),
                "maxDD": round(group["dd"]),
                "used_%_of_available": round(
                    group["dd"] / AVAILABLE_DD[account_idx] * 100, 1
                ),
                "used_%_of_limit": round(group["dd"] / LIMITS[account_idx] * 100, 1),
                "headroom_available": round(AVAILABLE_DD[account_idx] - group["dd"]),
                "trades": group["trades"],
                "blocked_trades": group["blocked_trades"],
                "adjacent_hours": "yes" if adjacent else "",
            }
        )
    return pd.DataFrame(rows), used


def single_window_metrics(window_idx):
    return combo_metrics((window_idx,))


# ---- FRONTIER ---------------------------------------------------------------
print("\n" + "=" * 110)
print("PROFIT vs SAFETY FRONTIER  (exact optimum at each cap)")
print("=" * 110)
print(
    f"{'cap':>6}  {'total net profit':>17}  {'windows used':>13}  "
    f"{'blocked':>8}  {'worst avail use':>15}"
)
frontier = {}
for fraction in CAP_FRACTIONS:
    chosen = solve(fraction)
    if chosen is None:
        print(f"{fraction * 100:5.0f}%  {'INFEASIBLE':>17}")
        continue
    df, used = describe(chosen)
    frontier[fraction] = (chosen, df, used)
    print(
        f"{fraction * 100:5.0f}%  {df['net_profit'].sum():>17,.0f}  "
        f"{len(used):>10}/{NW:<2}  {df['blocked_trades'].sum():>8,.0f}  "
        f"{df['used_%_of_available'].max():>14.1f}%"
    )


# ---- DETAIL AT THE REPORTING CAP -------------------------------------------
if not frontier:
    raise SystemExit("No feasible allocation found at any configured cap fraction.")
if REPORT_FRACTION not in frontier:
    REPORT_FRACTION = max(frontier)

chosen, A, used = frontier[REPORT_FRACTION]
print("\n" + "=" * 110)
print(f"ALLOCATION AT {REPORT_FRACTION * 100:.0f}% CAP")
print("=" * 110)
print(A.to_string(index=False))
print(
    f"\nTotal net profit: ${A['net_profit'].sum():,.0f}   "
    f"worst account at {A['used_%_of_available'].max():.1f}% of available DD"
)

dropped = [KEYS[i] for i in range(NW) if i not in used]
print(f"\nDropped windows ({len(dropped)}):")
for strategy, window in dropped:
    window_idx = K_IDX[(strategy, window)]
    m = single_window_metrics(window_idx)
    print(
        f"   {strategy} {window:<6} RR {RR_OF[window_idx]:<5g} "
        f"net ${m['profit']:>7,.0f}  DD ${m['dd']:>6,.0f}"
    )

if REPLAY_ONE_POSITION:
    blocked = int(A["blocked_trades"].sum())
    print(
        f"\nReplay note: candidate groups were scored after skipping blocked entries "
        f"({blocked:,} skipped trades in the reported allocation)."
    )
if (A["adjacent_hours"] == "yes").any():
    print(
        "Adjacent-hour pairs remain flagged for operations, but their blocking "
        "impact is now included in profit/DD scoring."
    )

os.makedirs("data/3_results", exist_ok=True)
try:
    A.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")
except PermissionError:
    import time

    OUT_CSV = OUT_CSV.replace(".csv", f"_{time.strftime('%H%M%S')}.csv")
    try:
        A.to_csv(OUT_CSV, index=False)
        print(f"\nSaved {OUT_CSV}")
    except PermissionError:
        print(f"\nCould not save {OUT_CSV}: permission denied or file locked.")
print("\nEach account is strategy-pure unless ALLOW_MIXED_STRATEGIES=True.")
print("Add accounts by extending ACCOUNT_NAMES, ACCOUNT_LIMITS, and ACCOUNT_DD_AVAILABLE.")

# ---- PROVENANCE -------------------------------------------------------------
PROV3 = prov.base(
    "3_allocate_accounts",
    upstream=prov.load("data/3_results/_provenance_step2.json"),
    accounts=dict(zip(ACCOUNT_NAMES, ACCOUNT_DD_AVAILABLE)),
    report_fraction=REPORT_FRACTION,
    cap_fractions=CAP_FRACTIONS,
    calibration_entries=len(CALIB),
    calibration_max=max(CALIB.values()) if CALIB else 1.0,
    settings={"MAX_WINDOWS_PER_ACCOUNT": MAX_WINDOWS_PER_ACCOUNT,
              "ALLOW_MIXED_STRATEGIES": ALLOW_MIXED_STRATEGIES,
              "REPLAY_ONE_POSITION": REPLAY_ONE_POSITION,
              "FORBID_ADJACENT_WINDOWS": FORBID_ADJACENT_WINDOWS},
)
prov.write("data/3_results/_provenance_step3.json", PROV3)
print("\nProvenance: " + prov.summary_line(PROV3))
