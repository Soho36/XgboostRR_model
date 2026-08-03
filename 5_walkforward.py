"""
5_walkforward.py
================
Anchored walk-forward validation of the WHOLE selection process.
Design + pass criteria: FORWARD_TESTING_PLAN.md  (read that first).

Every number the pipeline reports is in-sample. This script answers the real
question: if we had frozen the pipeline's decision at the end of year N, what
would year N+1 have paid?

For each fold:
  1. slice every sweep file to the FIT period only
  2. re-run the step-1 selection on that slice (tiers, +/-SMOOTH_RR
     neighbourhood, DD cap, verdict) -> recommended RR per OK window
  3. re-run the step-3 ILP on the fit slice -> account allocation
  4. score that FROZEN decision on the untouched TEST period
Then stitch the test periods into one continuous out-of-sample equity curve.

Also runs the critical control: every window at one fixed RR, no selection.
If the machinery can't beat that out-of-sample, it isn't earning its keep.

Faithful to steps 1/3 (same maths, reimplemented because those scripts execute
at import): equity-DD walk MAE-then-MFE, commission $1/RT, neighbourhood
robustness, strategy-pure groups of 1-2 windows, one-position replay, cap.
Differences, stated: no MT5 _stats calibration (full-period only; the walk
matches MT5 within ~0.1% on 99.9% of passes) and no blown-pass exclusion.

OUT: data/3_results/walkforward_folds.csv / walkforward_summary.csv
     data/3_results/_provenance_step5.json
"""

import glob
import itertools
import os
import re
import time

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

import provenance as prov

# ---- CONFIG (mirrors steps 1/3 — keep in sync) ------------------------------
SWEEPS_DIR = "data/1_sweeps"
COMMISSION = 1.0
MAX_DD_USD = 2000.0
MIN_RECOVERY = 2.0
SMOOTH_RR = 0.10
CAP_FRACTION = 0.85
MAX_PER_ACCOUNT = 2
BASELINE_RR = 1.50
# mirror 3_allocate_accounts.py
ACCOUNTS = {"PA-09-1500": 1500.0, "PA-10-1500": 1500.0, "PA-12-2000": 2000.0,
            "PA-13-2000": 2000.0, "PA-14-2000": 2000.0, "PA-15-2500": 2500.0}

FIT_START = pd.Timestamp("2020-01-01")
FOLDS = [  # (fit_end_exclusive, test_end_inclusive)
    (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")),
    (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
    (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")),
    (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")),
]

COLS = ["ticket", "entry_time", "exit_time", "mae", "mfe", "profit", "candle_range"]


def dd_equity(net, mae, mfe):
    """MAE-then-MFE walk; starting balance is the first peak. Same as step 1."""
    eq = peak = mdd = 0.0
    for n, a, f in zip(net, mae, mfe):
        mdd = max(mdd, peak - (eq + min(a, 0.0)))
        peak = max(peak, eq + max(f, 0.0))
        eq += n
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return float(mdd)


# ---- LOAD EVERYTHING ONCE ---------------------------------------------------
print("Loading sweeps (once; folds are in-memory slices) ...")
t0 = time.time()
DB = {}   # (strat, window, rr) -> dict of numpy arrays sorted by exit_time
for path in sorted(glob.glob(os.path.join(SWEEPS_DIR, "*", "*", "*.csv"))):
    strat = path.split(os.sep)[-3]
    if strat.endswith("_stats"):
        continue
    m = re.match(r"^(.+)_([0-9.]+)\.csv$", os.path.basename(path))
    if not m:
        continue
    win, rr = m.group(1), round(float(m.group(2)), 2)
    try:
        d = pd.read_csv(path, sep="\t", header=None, names=COLS, encoding="utf-16")
    except Exception:
        continue
    d = d[pd.to_numeric(d["ticket"], errors="coerce").notna()]
    for c in ("mae", "mfe", "profit"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("entry_time", "exit_time"):
        d[c] = pd.to_datetime(d[c].astype(str).str.strip(),
                              format="%Y.%m.%d %H:%M:%S", errors="coerce")
    d = d.dropna(subset=["exit_time", "profit"]).sort_values("exit_time")
    DB[(strat, win, rr)] = {
        "en": d["entry_time"].values.astype("datetime64[s]"),
        "ex": d["exit_time"].values.astype("datetime64[s]"),
        "net": (d["profit"].values - COMMISSION).astype(np.float64),
        "mae": d["mae"].values.astype(np.float64),
        "mfe": d["mfe"].values.astype(np.float64),
    }
WINDOWS = sorted({(s, w) for (s, w, _r) in DB},
                 key=lambda k: (k[0], int(k[1].split("-")[0])))
print(f"  {len(DB)} passes, {len(WINDOWS)} windows in {time.time()-t0:,.0f}s")


def sl(key, a, b):
    """Trades of one pass with a <= exit_time < b."""
    d = DB[key]
    m = (d["ex"] >= np.datetime64(a)) & (d["ex"] < np.datetime64(b))
    return {k: v[m] for k, v in d.items()}


# ---- STEP-1 SELECTION, PER FOLD --------------------------------------------
def select(fit_a, fit_b):
    """recommended RR per OK window, computed only from the fit slice."""
    picks = {}
    for (s, w) in WINDOWS:
        rrs = sorted(r for (s2, w2, r) in DB if (s2, w2) == (s, w))
        prof = np.empty(len(rrs)); dd = np.empty(len(rrs))
        for i, r in enumerate(rrs):
            t = sl((s, w, r), fit_a, fit_b)
            prof[i] = t["net"].sum()
            dd[i] = dd_equity(t["net"], t["mae"], t["mfe"])
        arr = np.array(rrs)
        best_rec, best = -np.inf, None
        for i, r in enumerate(rrs):
            nb = np.abs(arr - r) <= SMOOTH_RR + 1e-9
            p_nb, d_nb = prof[nb].mean(), dd[nb].max()
            if prof[i] <= 0 or p_nb <= 0 or d_nb > MAX_DD_USD:
                continue
            rec = p_nb / d_nb if d_nb > 0 else 0.0
            if rec > best_rec:
                best_rec, best = rec, r
        if best is not None and best_rec >= MIN_RECOVERY:      # verdict OK
            picks[(s, w)] = best
    return picks


# ---- STEP-3 ALLOCATION, PER FOLD -------------------------------------------
def replay(parts):
    """One-position replay across windows sharing an account; returns index
    masks per part. parts: list of trade dicts (fit or test slice)."""
    order = []
    for pi, t in enumerate(parts):
        for i in range(len(t["en"])):
            order.append((t["en"][i], t["ex"][i], pi, i))
    order.sort(key=lambda x: (x[0], x[1]))
    keep = [np.zeros(len(t["en"]), dtype=bool) for t in parts]
    open_until = np.datetime64("1970-01-01")
    for en, ex, pi, i in order:
        if en >= open_until:
            keep[pi][i] = True
            open_until = ex
    return keep


def group_eval(keys, a, b):
    """Net + equity DD of a window-group on [a,b) with one-position replay."""
    parts = [sl(k, a, b) for k in keys]
    keep = replay(parts)
    rows = []
    for t, m in zip(parts, keep):
        for i in np.flatnonzero(m):
            rows.append((t["ex"][i], t["net"][i], t["mae"][i], t["mfe"][i]))
    rows.sort(key=lambda x: x[0])
    net = np.array([r[1] for r in rows]) if rows else np.zeros(0)
    mae = np.array([r[2] for r in rows]) if rows else np.zeros(0)
    mfe = np.array([r[3] for r in rows]) if rows else np.zeros(0)
    return float(net.sum()), dd_equity(net, mae, mfe)


def allocate(picks, fit_a, fit_b):
    """Exact ILP on the fit slice: strategy-pure groups of 1-2, each account at
    most one group, each window once, group DD <= CAP_FRACTION * account."""
    keys = [(s, w, picks[(s, w)]) for (s, w) in picks]
    by_s = {}
    for k in keys:
        by_s.setdefault(k[0], []).append(k)
    groups = []
    for s, pool in by_s.items():
        for size in (1, 2):
            for combo in itertools.combinations(pool, size):
                p, d = group_eval(combo, fit_a, fit_b)
                groups.append({"keys": combo, "profit": p, "dd": d})
    names = list(ACCOUNTS)
    var = [(gi, ai) for gi, g in enumerate(groups) for ai, n in enumerate(names)
           if g["dd"] <= CAP_FRACTION * ACCOUNTS[n]]
    if not var:
        return []
    c = np.array([-groups[gi]["profit"] for gi, _ in var])
    rows, lb, ub = [], [], []
    for ai in range(len(names)):                     # each account <= 1 group
        rows.append([1.0 if a == ai else 0.0 for _, a in var]); lb.append(0); ub.append(1)
    for k in keys:                                   # each window <= once
        rows.append([1.0 if k in groups[gi]["keys"] else 0.0 for gi, _ in var])
        lb.append(0); ub.append(1)
    res = milp(c=c, constraints=LinearConstraint(np.array(rows), lb, ub),
               integrality=np.ones(len(var)), bounds=Bounds(0, 1))
    if not res.success:
        return []
    return [(groups[gi], names[ai]) for (gi, ai), x in zip(var, res.x) if x > 0.5]


# ---- WALK THE FOLDS ---------------------------------------------------------
years = lambda a, b: max((b - a).days / 365.25, 1e-9)
fold_rows, oos_trades = [], []
for fn, (fit_end, test_end) in enumerate(FOLDS, 1):
    t0 = time.time()
    picks = select(FIT_START, fit_end)
    alloc = allocate(picks, FIT_START, fit_end)
    is_p = sum(g["profit"] for g, _ in alloc)
    oos_p, worst = 0.0, ("", 0.0, 0.0)
    for g, acct in alloc:
        p, d = group_eval(g["keys"], fit_end, test_end)
        oos_p += p
        if d / ACCOUNTS[acct] > worst[1]:
            worst = (acct, d / ACCOUNTS[acct], d)
        for k in g["keys"]:
            t = sl(k, fit_end, test_end)
            for i in range(len(t["ex"])):
                oos_trades.append((t["ex"][i], t["net"][i]))
    # baseline: every window, fixed RR closest to BASELINE_RR, no selection
    base = 0.0
    for (s, w) in WINDOWS:
        rrs = [r for (s2, w2, r) in DB if (s2, w2) == (s, w)]
        r0 = min(rrs, key=lambda r: abs(r - BASELINE_RR))
        base += sl((s, w, r0), fit_end, test_end)["net"].sum()
    is_y, oos_y = years(FIT_START, fit_end), years(fit_end, min(test_end, pd.Timestamp("2026-07-14")))
    fold_rows.append({
        "fold": fn, "fit_end": fit_end.date(), "test_end": test_end.date(),
        "windows_picked": len(picks), "accounts_used": len(alloc),
        "IS_net": round(is_p), "IS_net_per_yr": round(is_p / is_y),
        "OOS_net": round(oos_p), "OOS_net_per_yr": round(oos_p / oos_y),
        "OOS_worst_acct": worst[0], "OOS_worst_dd": round(worst[2]),
        "OOS_worst_pct_of_limit": round(worst[1] * 100, 1),
        "baseline_OOS_net": round(base),
        "picks": "; ".join(f"{s} {w}@{picks[(s,w)]:g}" for (s, w) in sorted(picks)),
    })
    r = fold_rows[-1]
    print(f"\nFold {fn}  fit->{fit_end.date()}  test->{test_end.date()}  "
          f"({time.time()-t0:,.0f}s)\n"
          f"  picked {r['windows_picked']} windows on {r['accounts_used']} accounts | "
          f"IS ${r['IS_net']:,}/{is_y:.1f}y  OOS ${r['OOS_net']:,}/{oos_y:.1f}y | "
          f"worst acct {r['OOS_worst_acct']} {r['OOS_worst_pct_of_limit']}% | "
          f"baseline ${r['baseline_OOS_net']:,}")

F = pd.DataFrame(fold_rows)
oos_trades.sort(key=lambda x: x[0])
eq = np.cumsum([n for _, n in oos_trades])
peak = np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:]
stitched_dd = float((peak - eq).max()) if len(eq) else 0.0

wfe = F["OOS_net_per_yr"].sum() / max(F["IS_net_per_yr"].sum(), 1e-9)
summary = {
    "folds": len(F),
    "folds_OOS_positive": int((F["OOS_net"] > 0).sum()),
    "WFE": round(wfe, 3),
    "OOS_total_net": int(F["OOS_net"].sum()),
    "baseline_total_net": int(F["baseline_OOS_net"].sum()),
    "beats_baseline": bool(F["OOS_net"].sum() > F["baseline_OOS_net"].sum()),
    "stitched_OOS_maxDD": round(stitched_dd),
    "any_acct_over_limit": bool((F["OOS_worst_pct_of_limit"] > 100).any()),
}
print("\n" + "=" * 78)
print("WALK-FORWARD SUMMARY   (pass bar: WFE>=0.5, >=3/4 folds +, no acct >100%, "
      "beats baseline)")
print("=" * 78)
for k, v in summary.items():
    print(f"  {k:<24} {v}")

os.makedirs("data/3_results", exist_ok=True)
F.to_csv("data/3_results/walkforward_folds.csv", index=False)
pd.DataFrame([summary]).to_csv("data/3_results/walkforward_summary.csv", index=False)
prov.write("data/3_results/_provenance_step5.json", prov.base(
    "5_walkforward", folds=[str(f[0].date()) for f in FOLDS], summary=summary,
    settings={"CAP_FRACTION": CAP_FRACTION, "BASELINE_RR": BASELINE_RR,
              "accounts": ACCOUNTS},
    limitations=["no MT5 stats calibration per fold (full-period only)",
                 "thresholds chosen with full-history knowledge",
                 "baseline ignores account DD caps"]))
print("\nSaved data/3_results/walkforward_folds.csv, walkforward_summary.csv")
