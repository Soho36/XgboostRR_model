"""
allocate_accounts.py
====================
Allocate selected time windows across prop accounts so risk is spread across
the actual account drawdown limits.

Key idea: every chosen window is traded regardless of who trades it, so total
profit is constant across assignments. The allocation problem is therefore:

    minimize max(account_drawdown / account_limit)

The account limits are unequal (4 x $1500, 3 x $2000), so the best assignment
should deliberately place the heavier windows on the roomier accounts. Because
there are only a handful of windows and each account can take at most two, the
fixed allocation is solved exactly with dynamic programming rather than by
random local search.

INPUT : output_files/maemfe_combined_trades.csv  (from analyze_maemfe.py)
OUTPUT: output_files/account_allocation.csv
        plots/allocation/account_equity.png      (if matplotlib is available)

=================
Mechanics:
1. Load every trade (from maemfe_combined_trades.csv): which window it belongs to, its net P/L, its mae.
2. GROUP_METRICS — for each possible group of 1 or 2 windows, pull just those windows' trades, put them in time order, and measure that group's max drawdown. This is precomputed for every candidate pairing.
3. solve_exact_assignment (the DP) — hands account 1 some windows, account 2 some of what's left, and so on, trying every legal way, and keeps the arrangement whose worst account (drawdown ÷ its own limit) is lowest. The dp(account_idx, remaining_mask) memoization just avoids re-solving the same "these accounts left, these windows left" situation twice.
4. Score three approaches on the same yardstick — worst account's % of its limit:
    Optimised fixed (the DP answer)
    3,000 random fixed assignments
    300 daily-reshuffle simulations
4. Output the assignment table + the equity/DD plot.
=================
"""

from functools import lru_cache
from itertools import combinations
import os

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ---- CONFIG -----------------------------------------------------------------
TRADES_CSV = "output_files/maemfe_combined_trades.csv"
PLOT_DIR = "plots/allocation"
OUT_CSV = "output_files/account_allocation.csv"

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
MIN_WINDOWS_PER_ACCOUNT = 1
MAX_WINDOWS_PER_ACCOUNT = 2

# "floating" matches prop-firm risk better because MAE can hit before a trade
# closes. Use "closed" if you want to reproduce the old closed-PnL results.
DD_MODE = "floating"
SAFETY_TARGET = 0.70

MC_RANDOM = 3000
MC_SHUFFLE = 300
SHUFFLE_ASSIGN_ON = "entry_time"  # a live schedule assigns the account at entry
RNG = np.random.default_rng(42)


# ---- LOAD -------------------------------------------------------------------
if not os.path.exists(TRADES_CSV):
    raise SystemExit(f"Missing {TRADES_CSV} - run analyze_maemfe.py first.")

T = pd.read_csv(TRADES_CSV, parse_dates=["entry_time", "exit_time"])
T = T.sort_values("exit_time").reset_index(drop=True)

WINDOWS = sorted(T["window"].unique(), key=lambda w: int(w.split("-")[0]))
W_IDX = {w: i for i, w in enumerate(WINDOWS)}
w_of_trade = T["window"].map(W_IDX).to_numpy()
net = T["net"].to_numpy(float)
mae = T["mae"].to_numpy(float)
date_code, DATES = pd.factorize(T[SHUFFLE_ASSIGN_ON].dt.date)

NW, NA = len(WINDOWS), len(ACCOUNT_LIMITS)
LIMITS = np.array(ACCOUNT_LIMITS, dtype=float)
if len(ACCOUNT_NAMES) != NA:
    raise SystemExit(
        f"ACCOUNT_NAMES has {len(ACCOUNT_NAMES)} entries but ACCOUNT_LIMITS has {NA}."
    )

if DD_MODE not in {"closed", "floating"}:
    raise SystemExit("DD_MODE must be 'closed' or 'floating'.")
if NW < NA * MIN_WINDOWS_PER_ACCOUNT:
    raise SystemExit(
        f"{NW} windows cannot give every account at least "
        f"{MIN_WINDOWS_PER_ACCOUNT} window(s)."
    )
if NW > NA * MAX_WINDOWS_PER_ACCOUNT:
    raise SystemExit(
        f"{NW} windows do not fit in {NA} accounts with "
        f"MAX_WINDOWS_PER_ACCOUNT={MAX_WINDOWS_PER_ACCOUNT}."
    )

print(
    f"{len(T)} trades | {NW} windows | {NA} accounts "
    f"({sum(x == 1500 for x in ACCOUNT_LIMITS)}x$1500, "
    f"{sum(x == 2000 for x in ACCOUNT_LIMITS)}x$2000)"
)
print(f"Total net profit (constant, whoever trades it): ${net.sum():,.0f}")
print(f"Drawdown mode: {DD_MODE}")
print(
    f"Windows/account: min {MIN_WINDOWS_PER_ACCOUNT}, "
    f"max {MAX_WINDOWS_PER_ACCOUNT}"
)


# ---- DRAWDOWN ---------------------------------------------------------------
def max_dd_closed(pnl):
    """Closed-trade max drawdown from a chronological PnL series."""
    pnl = np.asarray(pnl, dtype=float)
    if not len(pnl):
        return 0.0
    equity = np.cumsum(pnl)
    return float((np.maximum.accumulate(equity) - equity).max())


def max_dd_floating(pnl, adverse_excursion):
    """Approximate floating maxDD by applying each trade's MAE before its close.

    The MAE export does not include the timestamp of the worst open loss, so
    this is still an approximation when two windows overlap on the same account.
    It is more conservative than closed-PnL drawdown and matches the upstream
    MAEMFE analysis.
    """
    equity = peak = maxdd = 0.0
    for n, m in zip(np.asarray(pnl, float), np.asarray(adverse_excursion, float)):
        trough = equity + min(m, 0.0)
        maxdd = max(maxdd, peak - trough)
        equity += n
        peak = max(peak, equity)
        maxdd = max(maxdd, peak - equity)
    return float(maxdd)


def drawdown_for_trades(mask):
    if DD_MODE == "floating":
        return max_dd_floating(net[mask], mae[mask])
    return max_dd_closed(net[mask])


# ---- GROUP METRICS ----------------------------------------------------------
def mask_to_windows(mask):
    return [WINDOWS[i] for i in range(NW) if mask & (1 << i)]


def window_mask_size(mask):
    return bin(mask).count("1")  # int.bit_count() needs Python 3.10+


def account_group_metrics(mask):
    """Return drawdown and profit for a set of window indexes encoded as bits."""
    if mask == 0:
        return 0.0, 0.0
    windows = [i for i in range(NW) if mask & (1 << i)]
    trade_mask = np.isin(w_of_trade, windows)
    return drawdown_for_trades(trade_mask), float(net[trade_mask].sum())


allowed_group_masks = []
for size in range(MIN_WINDOWS_PER_ACCOUNT, MAX_WINDOWS_PER_ACCOUNT + 1):
    for combo in combinations(range(NW), size):
        allowed_group_masks.append(sum(1 << i for i in combo))

GROUP_METRICS = {mask: account_group_metrics(mask) for mask in allowed_group_masks}


# ---- EXACT FIXED ASSIGNMENT -------------------------------------------------
def submasks_of_size(remaining_mask, size):
    bits = [i for i in range(NW) if remaining_mask & (1 << i)]
    for combo in combinations(bits, size):
        yield sum(1 << i for i in combo)


def solve_exact_assignment():
    """Solve minimax allocation exactly.

    Primary objective: lowest worst account DD/limit.
    Tiebreaker: lower sum of squared usage ratios, which avoids needlessly
    lopsided assignments when the worst account is unchanged.
    """
    full_mask = (1 << NW) - 1

    @lru_cache(maxsize=None)
    def dp(account_idx, remaining_mask):
        if account_idx == NA:
            if remaining_mask == 0:
                return 0.0, 0.0, ()
            return np.inf, np.inf, None

        remaining_accounts = NA - account_idx
        remaining_windows = window_mask_size(remaining_mask)
        if remaining_windows < remaining_accounts * MIN_WINDOWS_PER_ACCOUNT:
            return np.inf, np.inf, None
        if remaining_windows > remaining_accounts * MAX_WINDOWS_PER_ACCOUNT:
            return np.inf, np.inf, None

        future_accounts = remaining_accounts - 1
        min_now = max(
            MIN_WINDOWS_PER_ACCOUNT,
            remaining_windows - future_accounts * MAX_WINDOWS_PER_ACCOUNT,
        )
        max_now = min(
            MAX_WINDOWS_PER_ACCOUNT,
            remaining_windows - future_accounts * MIN_WINDOWS_PER_ACCOUNT,
        )

        best = (np.inf, np.inf, None)
        for size in range(min_now, max_now + 1):
            for mask in submasks_of_size(remaining_mask, size):
                dd, _profit = GROUP_METRICS[mask]
                ratio = dd / LIMITS[account_idx]
                child_worst, child_sumsq, child_masks = dp(
                    account_idx + 1, remaining_mask & ~mask
                )
                if child_masks is None:
                    continue
                score = (max(float(ratio), child_worst), ratio * ratio + child_sumsq)
                if score < best[:2]:
                    best = (score[0], score[1], (mask,) + child_masks)
        return best

    worst_ratio, sumsq, assignment_masks = dp(0, full_mask)
    if assignment_masks is None:
        raise RuntimeError("No valid account allocation found.")
    return assignment_masks, worst_ratio, sumsq


def evaluate_masks(account_masks):
    dds = np.zeros(NA)
    profits = np.zeros(NA)
    for account_idx, mask in enumerate(account_masks):
        dds[account_idx], profits[account_idx] = GROUP_METRICS[mask]
    return float((dds / LIMITS).max()), dds, profits


def assignment_to_masks(acct_of_window):
    account_masks = [0] * NA
    for window_idx, account_idx in enumerate(acct_of_window):
        account_masks[account_idx] |= 1 << window_idx
    return tuple(account_masks)


def random_assignment(rng):
    """Random valid fixed assignment under the same min/max account rules."""
    counts = np.full(NA, MIN_WINDOWS_PER_ACCOUNT, dtype=int)
    extra_needed = NW - int(counts.sum())
    if extra_needed:
        extra_slots = np.repeat(np.arange(NA), MAX_WINDOWS_PER_ACCOUNT - MIN_WINDOWS_PER_ACCOUNT)
        chosen = rng.choice(len(extra_slots), size=extra_needed, replace=False)
        np.add.at(counts, extra_slots[chosen], 1)
    return rng.permutation(np.repeat(np.arange(NA), counts))


def random_daily_schedule(rng, days):
    """Return a (days, windows) table of random valid daily account assignments."""
    base = np.tile(np.repeat(np.arange(NA), MIN_WINDOWS_PER_ACCOUNT), (days, 1))
    extra_needed = NW - base.shape[1]
    if extra_needed:
        extra_slots = np.repeat(
            np.arange(NA), MAX_WINDOWS_PER_ACCOUNT - MIN_WINDOWS_PER_ACCOUNT
        )
        picks = np.argsort(rng.random((days, len(extra_slots))), axis=1)[:, :extra_needed]
        labels = np.concatenate([base, extra_slots[picks]], axis=1)
    else:
        labels = base
    order = np.argsort(rng.random(labels.shape), axis=1)
    return np.take_along_axis(labels, order, axis=1)


# ---- 1. OPTIMISED FIXED ASSIGNMENT -----------------------------------------
print("\nOptimising fixed assignment exactly ...")
best_masks, best_score, best_sumsq = solve_exact_assignment()
worst, dds, profits = evaluate_masks(best_masks)

print("\n" + "=" * 96)
print("OPTIMISED FIXED ASSIGNMENT")
print("=" * 96)
rows = []
for account_idx, mask in enumerate(best_masks):
    wins = mask_to_windows(mask)
    ratio = dds[account_idx] / LIMITS[account_idx]
    rows.append(
        {
            "account": ACCOUNT_NAMES[account_idx],
            "DD_limit": int(LIMITS[account_idx]),
            "windows": ", ".join(wins),
            "net_profit": round(profits[account_idx]),
            "maxDD": round(dds[account_idx]),
            "used_%_of_limit": round(ratio * 100, 1),
            "headroom": round(LIMITS[account_idx] - dds[account_idx]),
            "status": "OK"
            if ratio <= SAFETY_TARGET
            else ("TIGHT" if ratio < 1 else "BREACH"),
        }
    )
A = pd.DataFrame(rows)
print(A.to_string(index=False))
print(
    f"\nWorst account uses {worst * 100:.1f}% of its limit "
    f"(target <= {SAFETY_TARGET * 100:.0f}%)"
)
print(f"Risk-balance tiebreak score: {best_sumsq:.4f}")


# ---- 2. RANDOM FIXED ASSIGNMENTS -------------------------------------------
print("\nMonte-Carlo: random fixed assignments ...")
rnd = np.array(
    [
        evaluate_masks(assignment_to_masks(random_assignment(RNG)))[0]
        for _ in range(MC_RANDOM)
    ]
)
print(
    f"  worst-account limit usage: median {np.median(rnd) * 100:.1f}%, "
    f"p90 {np.percentile(rnd, 90) * 100:.1f}%, max {rnd.max() * 100:.1f}%"
)
print(
    f"  share of random assignments that BREACH some account: "
    f"{(rnd >= 1).mean() * 100:.1f}%"
)
print(
    f"  share exceeding the {SAFETY_TARGET * 100:.0f}% safety target: "
    f"{(rnd > SAFETY_TARGET).mean() * 100:.1f}%"
)


# ---- 3. DAILY SHUFFLE -------------------------------------------------------
print("\nMonte-Carlo: daily re-shuffle ...")
shuf = np.empty(MC_SHUFFLE)
for run_idx in range(MC_SHUFFLE):
    table = random_daily_schedule(RNG, len(DATES))
    acct = table[date_code, w_of_trade]
    dds_s = np.zeros(NA)
    for account_idx in range(NA):
        trade_mask = acct == account_idx
        dds_s[account_idx] = drawdown_for_trades(trade_mask)
    shuf[run_idx] = (dds_s / LIMITS).max()

print(
    f"  assigned by: {SHUFFLE_ASSIGN_ON} date; "
    f"same min/max account rules as fixed allocation"
)
print(
    f"  worst-account limit usage: median {np.median(shuf) * 100:.1f}%, "
    f"p90 {np.percentile(shuf, 90) * 100:.1f}%, max {shuf.max() * 100:.1f}%"
)
print(
    f"  share of shuffled runs that BREACH some account: "
    f"{(shuf >= 1).mean() * 100:.1f}%"
)

print("\n" + "=" * 96)
print(f"{'OPTIMISED fixed':<22} worst account at {worst * 100:5.1f}% of limit")
print(f"{'RANDOM fixed (median)':<22} worst account at {np.median(rnd) * 100:5.1f}% of limit")
print(f"{'DAILY SHUFFLE (median)':<22} worst account at {np.median(shuf) * 100:5.1f}% of limit")
print("=" * 96)


# ---- SAVE + PLOT ------------------------------------------------------------
os.makedirs("output_files", exist_ok=True)
try:
    A.to_csv(OUT_CSV, index=False)
except PermissionError:
    import time

    OUT_CSV = OUT_CSV.replace(".csv", f"_{time.strftime('%H%M%S')}.csv")
    A.to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV}")

if plt is None:
    print("Plot skipped: matplotlib is not installed in this Python environment.")
else:
    os.makedirs(PLOT_DIR, exist_ok=True)
    acct = np.empty(NW, dtype=int)
    for account_idx, mask in enumerate(best_masks):
        for window_idx in range(NW):
            if mask & (1 << window_idx):
                acct[window_idx] = account_idx
    acct_of_trade = acct[w_of_trade]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [2, 1]}
    )
    for account_idx in range(NA):
        trade_mask = acct_of_trade == account_idx
        if not trade_mask.any():
            continue
        ax1.plot(
            T["exit_time"][trade_mask],
            np.cumsum(net[trade_mask]),
            lw=1.3,
            label=(
                f"{ACCOUNT_NAMES[account_idx]} (${int(LIMITS[account_idx])}) "
                f"DD ${dds[account_idx]:,.0f}"
            ),
        )
    ax1.set_title(f"Per-account equity under the optimised assignment ({DD_MODE} DD)")
    ax1.set_ylabel("Equity $ (net)")
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    x = np.arange(NA)
    ax2.bar(
        x,
        dds,
        color=[
            "tab:green" if dd / limit <= SAFETY_TARGET else "tab:orange"
            for dd, limit in zip(dds, LIMITS)
        ],
    )
    ax2.plot(x, LIMITS, "r_", markersize=28, markeredgewidth=2.5, label="DD limit")
    ax2.set_xticks(x, ACCOUNT_NAMES, rotation=35, ha="right")
    ax2.set_title("Max drawdown vs limit")
    ax2.set_ylabel("$")
    ax2.legend()
    ax2.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "account_equity.png"), dpi=110)
    plt.close(fig)
    print(f"Plot  {PLOT_DIR}/account_equity.png")

print("\nNOTE: drawdowns are historical (2020-2026). Future DD can exceed them,")
print(f"which is why the target is <= {SAFETY_TARGET * 100:.0f}% of the limit, not 99%.")
if DD_MODE == "floating":
    print("Floating DD uses trade MAE as an approximation; tick-level open equity would be better.")


"""
Why the shuffle loses — and why fixed isn't "removing diversification"
This is the heart of it, and your intuition has one wrong assumption. Here's the corrected picture:

Diversification is already 100% baked into the 12 windows and it never changes. All 12 together always make $63,452 with a $3,348 combined drawdown — no matter how you hand them out. You cannot add or remove that by assignment. So the assignment question is not "how much diversification" — it's "where does the fixed pile of drawdown land, and can each account hold its share?"

Analogy — 12 rocks, 7 shelves with weight limits. You must place all 12 rocks (total weight fixed). Each shelf can break at its limit (1500 or 2000).

Fixed optimised = you deliberately place heavy rocks on strong shelves and balance the rest. Best possible: worst shelf at 83%.
Random fixed = you place them once, blindfolded. Worst shelf typically at 143% — something broke.
Daily shuffle = every single day you re-toss all rocks onto random shelves. Over years, each shelf eventually catches a bad pile-up. Worst shelf ends at 171% — worse than doing it randomly once, because you keep re-rolling and every shelf gets many chances to be overloaded.
Measured: 83% (optimised) < 143% (random once) < 171% (shuffle) — 100% of shuffles blew an account.

Why shuffling can't help: your limit binds on the worst of 7 accounts. Random placement gives you no control over the worst one — you're taking the max of 7 noisy outcomes, and the max of noise is bad. Deliberate placement controls all 7 at once.

And there's a bonus fixed gives you that shuffle can't: you can permanently pair windows whose bad patches happen at different times, so they cancel. Proof from your own data — account A3:

5-6 alone drops $956, 7-8 alone drops $716. Put them on the same account permanently → combined drawdown $698 — less than 7-8 by itself. Their drawdowns offset.

That's real diversification working at the account level — and you only capture it by keeping the pair together. Shuffle scatters them, so no account ever gets a clean offsetting pair. Fixed assignment doesn't remove diversification; it places it where it protects each account.
"""
