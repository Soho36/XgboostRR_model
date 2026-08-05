"""
7_account_farming.py
====================
Tests the OTHER strategy: don't fit the system to the account — buy many
accounts and let survivorship do the work.

Question, posed 2026-08-05: run the strategy completely unoptimised (every
window enabled, one global RR), stagger account start dates, accept that some
blow up, and harvest the ones that reach the Safety Net where the trailing
drawdown freezes. Is that better than squeezing the strategy into a small
trailing-DD box (steps 5-6)?

ACCOUNT RULE (Legacy Performance Account, as supplied):
  liquidation floor starts at  -TRAILING_DD
  it follows the PEAK UNREALIZED balance upward,
  and stops once peak reaches the Safety Net = TRAILING_DD + 100 of profit,
  where the floor freezes at +100 forever.
  So:  floor = min(peak - TRAILING_DD, +100)      <- one line, exactly
  Reaching the Safety Net is the whole game: after it, the account can absorb a
  full TRAILING_DD drawdown and never dies from trailing again.

WHY INTRADAY MATTERS HERE: the rule says *unrealized*. A trade that runs to
+MFE raises the floor even if it closes lower, and a trade that dips to -MAE
can hit the floor without ever closing there. A closed-trade P&L series
understates both. This uses the per-trade MAE/MFE from the sweeps, in the
MAE-then-MFE order that was validated to match MT5 exactly.

THE HARVEST TRADE-OFF (do not get this backwards): once frozen, the floor is
nailed at +100 forever, so the account's cushion is simply its equity minus 100.
Withdrawing cash therefore makes the account MORE fragile, not less. Harvest
early and you bank money but die sooner; never harvest and you compound a bigger
cushion but lose the lot when it finally goes. `--keep` sets the equity level to
withdraw down to; `--keep 0` never withdraws. Both are simulated.

CENSORING: an account started in 2026 has under six months of data left, so it
cannot freeze in the ~150 days it typically takes. Comparing its freeze rate to
a 2020 start is meaningless. Stats are therefore reported twice: over all
starts, and over only those with a full year of data left.

OUT: data/3_results/farming_starts.csv    one row per possible start date
     data/3_results/farming_cadence.csv   one row per staggering cadence
     data/3_results/farming_portfolio.csv year-by-year cash flow of N accounts
"""

import argparse
import importlib
import os

import numpy as np
import pandas as pd

wf = importlib.import_module("5_walkforward")

STRATEGY = "RR"
RR = 1.00
TRAILING_DD = 2500.0
SAFETY_NET = TRAILING_DD + 100.0      # peak profit that freezes the floor
FROZEN_FLOOR = 100.0
ACCOUNT_COST = 200.0
MT5_REF = {"trades": 9775, "gross": 46343.50, "eq_dd": 6069.00}


def all_window_stream(strategy, rr):
    """Every window of one strategy at one RR, merged into the single-position
    stream one account actually trades."""
    keys, blown = [], []
    for (s, w) in wf.WINDOWS:
        if s != strategy:
            continue
        if wf.pass_is_blown(s, w, rr):
            blown.append(w)
        else:
            keys.append((s, w, rr))
    parts = [wf.DB[k] for k in keys]
    keep = wf.replay(parts)
    rows = []
    for t, m in zip(parts, keep):
        for i in np.flatnonzero(m):
            rows.append((t["ex"][i], t["net"][i], t["mae"][i], t["mfe"][i]))
    rows.sort(key=lambda r: r[0])
    taken = sum(int(m.sum()) for m in keep)
    offered = sum(len(p["en"]) for p in parts)
    return rows, keys, blown, offered, taken


def new_account(i0):
    return {"i0": i0, "eq": 0.0, "peak": 0.0, "floor": -TRAILING_DD,
            "frozen": False, "froze_i": None, "banked": 0.0, "alive": True}


def step(a, n, ma, mf, keep):
    """Advance one account by one trade. Returns cash withdrawn on this trade.

    MAE-then-MFE within a trade: dip against the standing floor first, then let
    the run-up raise the peak. That ordering reproduced MT5's equity drawdown
    exactly on every pass tested, so it is the one used here.
    """
    if a["eq"] + min(ma, 0.0) <= a["floor"]:               # intraday dip kills it
        a["alive"] = False
        return 0.0
    a["peak"] = max(a["peak"], a["eq"] + max(mf, 0.0))     # unrealized peak counts
    a["floor"] = min(a["peak"] - TRAILING_DD, FROZEN_FLOOR)
    a["eq"] += n
    a["peak"] = max(a["peak"], a["eq"])
    a["floor"] = min(a["peak"] - TRAILING_DD, FROZEN_FLOOR)
    if a["eq"] <= a["floor"]:
        a["alive"] = False
        return 0.0
    if not a["frozen"] and a["peak"] >= SAFETY_NET:
        a["frozen"] = True
    if a["frozen"] and keep and a["eq"] > keep:
        w = a["eq"] - keep
        a["eq"] = keep
        a["banked"] += w
        return w
    return 0.0


def run_account(net, mae, mfe, i0, keep):
    a = new_account(i0)
    for i in range(i0, len(net)):
        step(a, net[i], mae[i], mfe[i], keep)
        if not a["alive"]:
            a["dead_i"] = i
            return a
        if a["frozen"] and a["froze_i"] is None:
            a["froze_i"] = i
    a["dead_i"] = None
    return a


def run_portfolio(ex, net, mae, mfe, n_live, keep, gap_days):
    """Keep up to n_live accounts running, buying a replacement when one dies.

    This is the operating model, and it is the one that exposes the real risk:
    every account trades the same signals, so a drawdown big enough to kill one
    is big enough to kill all of them in the same week.
    """
    live, cash, bought, deaths = [], [], 0, 0
    last_start = None
    for i in range(len(net)):
        day = pd.Timestamp(ex[i])
        if len(live) < n_live and (last_start is None
                                   or (day - last_start).days >= gap_days):
            live.append(new_account(i))
            bought += 1
            last_start = day
        got = 0.0
        for a in live:
            if i >= a["i0"]:
                got += step(a, net[i], mae[i], mfe[i], keep)
        deaths += sum(1 for a in live if not a["alive"])
        live = [a for a in live if a["alive"]]
        cash.append((day, got))
    equity_left = sum(a["eq"] for a in live)
    C = pd.DataFrame(cash, columns=["day", "cash"])
    C["yr"] = C["day"].dt.year
    return C, bought, deaths, equity_left, len(live)


def main():
    global TRAILING_DD, SAFETY_NET, ACCOUNT_COST
    ap = argparse.ArgumentParser(description="prop-account farming simulation")
    ap.add_argument("--dd", type=float, default=TRAILING_DD)
    ap.add_argument("--cost", type=float, default=ACCOUNT_COST)
    ap.add_argument("--rr", type=float, default=RR)
    ap.add_argument("--strategy", default=STRATEGY)
    ap.add_argument("--split", type=float, default=1.0,
                    help="trader's share of profit (real plans are 0.8-0.9)")
    ap.add_argument("--live", type=int, default=10,
                    help="concurrent accounts in the portfolio model")
    a = ap.parse_args()
    TRAILING_DD, SAFETY_NET, ACCOUNT_COST = a.dd, a.dd + 100.0, a.cost
    KEEP = SAFETY_NET                      # harvest down to the Safety Net

    wf.load_db()
    rows, keys, blown, offered, taken = all_window_stream(a.strategy, round(a.rr, 2))
    ex = np.array([r[0] for r in rows])
    net = np.array([r[1] for r in rows])
    mae = np.array([r[2] for r in rows])
    mfe = np.array([r[3] for r in rows])
    gross = float((net + wf.COMMISSION).sum())

    print("=" * 88)
    print(f"RECONSTRUCTION — {a.strategy} strategy, all windows, RR {a.rr:g}, "
          "one position at a time")
    print("=" * 88)
    print(f"  windows merged            {len(keys)}"
          + (f"   (excluded, tester-blown: {', '.join(blown)})" if blown else ""))
    print(f"  entries offered           {offered:,}")
    print(f"  entries taken             {taken:,}   "
          f"({offered - taken:,} blocked by the open position)")
    print(f"  gross profit          ${gross:>10,.0f}     MT5 run: "
          f"${MT5_REF['gross']:,.0f}   ({100*gross/MT5_REF['gross']-100:+.1f}%)")
    print(f"  trades                 {len(net):>10,}     MT5 run: "
          f"{MT5_REF['trades']:,}   ({100*len(net)/MT5_REF['trades']-100:+.1f}%)")
    print(f"  net after $1/RT       ${net.sum():>10,.0f}")
    print(f"  equity DD (MAE-first) ${wf.dd_equity(net, mae, mfe):>10,.0f}     MT5 run: "
          f"${MT5_REF['eq_dd']:,.0f}")

    # ---- every possible start date ------------------------------------------
    days = pd.to_datetime(ex).normalize()
    first_i = pd.Series(range(len(ex))).groupby(days).min().sort_index()
    last_day = pd.Timestamp(ex[-1])
    out = []
    for d, i0 in first_i.items():
        h = run_account(net, mae, mfe, int(i0), KEEP)      # harvest to Safety Net
        n_ = run_account(net, mae, mfe, int(i0), 0.0)      # never withdraw
        out.append({
            "start": d.date(), "start_i": int(i0),
            "runway_days": (last_day - d).days,
            "frozen": h["frozen"], "dead": h["dead_i"] is not None,
            "days_to_freeze": ((pd.Timestamp(ex[h["froze_i"]]) - d).days
                               if h["froze_i"] is not None else None),
            "banked": round(h["banked"]),
            "value_harvest": round(h["banked"]
                                   + (h["eq"] if h["dead_i"] is None else 0.0)),
            "value_nowithdraw": round(n_["eq"] if n_["dead_i"] is None else 0.0),
        })
    S = pd.DataFrame(out)
    S["profit"] = S["value_harvest"] * a.split - ACCOUNT_COST
    S["yr"] = pd.to_datetime(S["start"]).dt.year
    FULL = S[S["runway_days"] >= 365]      # censoring-free subset

    print("\n" + "=" * 88)
    print(f"ONE ACCOUNT, ${TRAILING_DD:,.0f} trailing DD — every possible start date")
    print("=" * 88)
    for name, X in (("all starts", S), ("starts with >=1yr of runway", FULL)):
        print(f"  {name:<30} n={len(X):<5} "
              f"froze {X['frozen'].mean():>6.1%}   "
              f"died before freezing {(X['dead'] & ~X['frozen']).mean():>6.1%}   "
              f"eventually liquidated {X['dead'].mean():>6.1%}")
    ok = FULL[FULL["frozen"]]
    print(f"\n  days to reach the Safety Net (uncensored): median "
          f"{ok['days_to_freeze'].median():.0f}, p25 {ok['days_to_freeze'].quantile(.25):.0f}, "
          f"p75 {ok['days_to_freeze'].quantile(.75):.0f}")

    print(f"\n  value per account, harvest-to-Safety-Net vs never-withdraw:")
    print(f"     {'':6}{'harvest':>12}{'no withdraw':>14}")
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        print(f"     p{int(q*100):<5}${S['value_harvest'].quantile(q):>11,.0f}"
              f"${S['value_nowithdraw'].quantile(q):>13,.0f}")
    print(f"     {'mean':<6}${S['value_harvest'].mean():>11,.0f}"
          f"${S['value_nowithdraw'].mean():>13,.0f}")
    print(f"\n  EXPECTED PROFIT PER ${ACCOUNT_COST:,.0f} ACCOUNT "
          f"(split {a.split:.0%})   ${S['profit'].mean():>9,.0f}")
    print(f"  share of accounts that lose money            "
          f"{(S['profit'] < 0).mean():>9.1%}")

    print("\n  BY START YEAR — accounts started close together share one fate,")
    print("  so this spread is the real risk, not the average above.")
    by = S.groupby("yr").agg(starts=("frozen", "size"),
                             runway=("runway_days", "max"),
                             froze=("frozen", "mean"),
                             value=("value_harvest", "mean"),
                             profit=("profit", "mean"))
    by["froze"] = (by["froze"] * 100).round(0).astype(int).astype(str) + "%"
    print(by.round(0).to_string())

    # ---- staggering cadences -------------------------------------------------
    print("\n" + "=" * 88)
    print("STAGGERED PORTFOLIO — start one account every N calendar days")
    print("=" * 88)
    cad = []
    for n in (7, 14, 30, 45, 60, 90, 120, 180):
        sel, last = [], None
        for _, r in S.iterrows():
            d = pd.Timestamp(r["start"])
            if last is None or (d - last).days >= n:
                sel.append(r)
                last = d
        P = pd.DataFrame(sel)
        cad.append({
            "every_n_days": n, "accounts": len(P),
            "cost": round(len(P) * ACCOUNT_COST),
            "froze_pct": round(100 * P["frozen"].mean()),
            "gross_value": round(P["value_harvest"].sum() * a.split),
            "net_profit": round(P["value_harvest"].sum() * a.split
                                - len(P) * ACCOUNT_COST),
            "per_account": round(P["profit"].mean()),
        })
    C = pd.DataFrame(cad)
    print(C.to_string(index=False))
    print("\n  Per-account profit barely moves with cadence — the cadence is not the")
    print("  lever. But note the account COUNT: those totals assume you may run that")
    print("  many accounts at once, which is a firm rule, not a maths question.")

    # ---- realistic operating model ------------------------------------------
    print("\n" + "=" * 88)
    print(f"OPERATING MODEL — hold {a.live} accounts, replace each one when it dies")
    print("=" * 88)
    C2, bought, deaths, eq_left, still = run_portfolio(
        ex, net, mae, mfe, a.live, KEEP, gap_days=7)
    yr = C2.groupby("yr")["cash"].sum().rename("cash_withdrawn").to_frame()
    yr["cash_withdrawn"] = (yr["cash_withdrawn"] * a.split).round()
    print(yr.to_string())
    total = float(C2["cash"].sum()) * a.split
    print(f"\n  accounts bought {bought}  (deaths {deaths}, still live {still})")
    print(f"  cost            ${bought * ACCOUNT_COST:>10,.0f}")
    print(f"  cash withdrawn  ${total:>10,.0f}")
    print(f"  equity left in live accounts ${eq_left * a.split:>10,.0f} (not yet cash)")
    print(f"  NET over {C2['yr'].nunique()} years  ${total - bought*ACCOUNT_COST:>10,.0f}"
          f"   = ${(total - bought*ACCOUNT_COST)/6.5:,.0f}/yr")
    print(f"\n  compare: walk-forward config (6 small accounts, tuned) ~$4,280/yr")
    print(f"           one unconstrained $5,000 account, all windows ~$4,850/yr")

    os.makedirs("data/3_results", exist_ok=True)
    S.drop(columns=["yr"]).to_csv("data/3_results/farming_starts.csv", index=False)
    C.to_csv("data/3_results/farming_cadence.csv", index=False)
    yr.to_csv("data/3_results/farming_portfolio.csv")
    print("\nSaved farming_starts.csv, farming_cadence.csv, farming_portfolio.csv")
    print("\nNOTE: every account trades the SAME signals and differs only by start")
    print("date. One drawdown deep enough to kill one can kill the whole book.")


if __name__ == "__main__":
    main()
