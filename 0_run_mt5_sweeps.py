"""
0_run_mt5_sweeps.py
===================
Drives the MT5 Strategy Tester from Python so you don't have to click through
the GUI once per window.

How the automation splits up:
  * the RR sweep      -> done BY MT5 (one optimization pass per RR value)
  * the window loop   -> done HERE (one MT5 launch per window)

So `python 0_run_mt5_sweeps.py --windows 11-12 9-10` runs two optimizations and
leaves you ~500 per-trade CSVs, already named <window>_<RR>.csv, ready for
1_select_rr.py.

REQUIRES the EA patch (see below) — without it every pass overwrites the same
file and the exports scatter across per-agent sandbox folders:
  input string RunTag = "";
  g_csvName = RunTag + "_" + DoubleToString(RiskReward, 2) + ".csv";
  FileOpen(g_csvName, ...|FILE_COMMON);  FileIsExist(g_csvName, FILE_COMMON);
  FileDelete(g_csvName, FILE_COMMON);

USAGE
  python 0_run_mt5_sweeps.py --list                 # show window -> input mapping
  python 0_run_mt5_sweeps.py --windows 11-12 --dry-run   # print the .ini only
  python 0_run_mt5_sweeps.py --windows 11-12 --strategy GG
  python 0_run_mt5_sweeps.py --windows 2-3 3-4 4-5 --strategy GG
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

# ---- CONFIG: set these once -------------------------------------------------
# Each strategy has its OWN MT5 installation here, so terminal/expert/symbol are
# all per-strategy. "expert" is relative to MQL5\Experts\ in that terminal's DATA
# folder (File > Open Data Folder) and must be the compiled .ex5, INCLUDING any
# subfolder — a bare filename fails with "tester didn't start / EX5 not found".
STRATEGIES = {
    "RR": {
        "terminal": r"I:\Programs\1AMP Global (USA) MT5 Exchange-Traded Futures Only\terminal64.exe",
        "expert": r"444\RR_r_MFE_buy-stop-entry.ex5",
        "symbol": "MNQcontDATABENTOcurr6",
    },
    "GG": {
        "terminal": r"I:\Programs\1MetaTrader 5\terminal64.exe",
        "expert": r"555\GG_r_MFE_buy-stop-entry.ex5",
        "symbol": "MNQcontDATABENTOcurr6",
    },
}
PERIOD = "M30"
DEPOSIT = 5000
CURRENCY = "USD"
LEVERAGE = "1:100"

# Keep these IDENTICAL to whatever produced your existing exports, otherwise the
# new sweeps are not comparable with the old files (this bit us before).
FROM_DATE = "2020.01.02"
TO_DATE = "2026.07.14"
MODEL = 1            # 0=every tick, 1=1-min OHLC, 2=open prices, 4=real ticks

# RR sweep: start, step, stop
RR_START, RR_STEP, RR_STOP = 1, 0.1, 2.00

# 1 = SLOW COMPLETE algorithm. Do NOT use 2 (genetic) — it skips RR values.
OPTIMIZATION_MODE = 1

COMMON_FILES = os.path.join(os.environ.get("APPDATA", ""),
                            "MetaQuotes", "Terminal", "Common", "Files")
DEST_ROOT = "INPUTS/data_2_maemfe_input"     # <STRAT>_sweeps/<window>/ goes here
INI_DIR = "OUTPUTS/mt5_ini"

# Every window toggle the EA exposes (order irrelevant; all are forced false
# except the target one).
ALL_WINDOW_INPUTS = [
    "W0000W0100", "W0100W0130", "W0130W0200", "W0200W0300", "W0300W0400",
    "W0400W0500", "W0500W0600", "W0600W0700", "W0700W0800", "W0800W0900",
    "W0900W1000", "W1000W1100", "W1100W1200", "W1200W1300", "W1300W1400",
    "W1400W1500", "W1500W1600", "W1600W1700", "W1700W1800", "W1800W1900",
    "W1900W2000", "W2000W2100", "W2100W2200", "W2200W2300", "W2300W2330",
    "W2330W0000",
]


def inputs_for_window(win):
    """'11-12' -> ['W1100W1200'].  Handles the split/odd hours explicitly."""
    special = {
        "1-2": ["W0100W0130", "W0130W0200"],   # EA splits 01:00-02:00 in halves
        "23-0": ["W2300W2330", "W2330W0000"],
        "23-24": ["W2300W2330", "W2330W0000"],
    }
    if win in special:
        return special[win]
    a, b = win.split("-")
    name = f"W{int(a):02d}00W{int(b):02d}00"
    if name not in ALL_WINDOW_INPUTS:
        raise SystemExit(f"Window {win!r} -> {name}, which the EA does not expose.\n"
                         f"Run --list to see valid windows.")
    return [name]


def data_dir_for(terminal_exe):
    """Map terminal64.exe -> its %APPDATA%\\MetaQuotes\\Terminal\\<hash> data folder."""
    root = os.path.join(os.environ.get("APPDATA", ""), "MetaQuotes", "Terminal")
    want = os.path.dirname(os.path.abspath(terminal_exe)).lower()
    if not os.path.isdir(root):
        return None
    for name in os.listdir(root):
        origin = os.path.join(root, name, "origin.txt")
        if not os.path.isfile(origin):
            continue
        try:
            with open(origin, "rb") as fh:
                txt = fh.read().decode("utf-16", errors="ignore").strip("﻿\x00 \r\n")
        except OSError:
            continue
        if txt.lower() == want:
            return os.path.join(root, name)
    return None


def preflight(strategy):
    """Fail loudly BEFORE launching. A wrong Expert path makes MT5 exit in ~8s
    with no output at all, which is near-impossible to diagnose from outside."""
    cfg = STRATEGIES[strategy]
    problems = []
    if not os.path.isfile(cfg["terminal"]):
        problems.append(f"terminal not found: {cfg['terminal']}")
        return problems
    dd = data_dir_for(cfg["terminal"])
    if dd is None:
        print(f"  (note: could not resolve data folder for {strategy}; "
              f"skipping the Expert existence check)")
        return problems
    ex5 = os.path.join(dd, "MQL5", "Experts", cfg["expert"].replace("\\", os.sep))
    if not os.path.isfile(ex5):
        found = []
        base = os.path.basename(cfg["expert"])
        for r, _d, fs in os.walk(os.path.join(dd, "MQL5", "Experts")):
            for f in fs:
                if f.lower() == base.lower():
                    rel = os.path.relpath(os.path.join(r, f),
                                          os.path.join(dd, "MQL5", "Experts"))
                    found.append(rel)
        msg = f"Expert not found: MQL5\\Experts\\{cfg['expert']}"
        if found:
            msg += "\n      did you mean:  " + "\n                     ".join(found)
        problems.append(msg)
    return problems


def build_ini(win, tag, strategy, rr):
    on = set(inputs_for_window(win))
    rr_start, rr_stop, rr_step = rr
    cfg = STRATEGIES[strategy]
    lines = [
        "[Tester]",
        f"Expert={cfg['expert']}",
        f"Symbol={cfg['symbol']}",
        f"Period={PERIOD}",
        f"Model={MODEL}",
        f"FromDate={FROM_DATE}",
        f"ToDate={TO_DATE}",
        f"Deposit={DEPOSIT}",
        f"Currency={CURRENCY}",
        f"Leverage={LEVERAGE}",
        f"Optimization={OPTIMIZATION_MODE}",
        "OptimizationCriterion=0",
        "ForwardMode=0",          # forward split would halve the export period
        "Visual=0",
        "ShutdownTerminal=1",     # required so this script can loop
        "",
        "[TesterInputs]",
        f"RunTag={tag}",
        # value||start||step||stop||Y  — Y marks it as optimized
        f"RiskReward={rr_start}||{rr_start}||{rr_step}||{rr_stop}||Y",
        "UseTradeWindow=true",
    ]
    lines += [f"{w}={'true' if w in on else 'false'}" for w in ALL_WINDOW_INPUTS]
    return "\n".join(lines) + "\n"


def check_manifest(dest_dir, strategy, rr):
    """Record what produced this folder's data, and shout if a re-run changes it.

    Collection overwrites existing files silently, so re-running a window with a
    different date range / symbol / model would leave a folder holding a MIX of
    two periods — exactly the staleness class of bug that has bitten this project
    repeatedly. The manifest makes that impossible to miss.
    """
    cfg = STRATEGIES[strategy]
    now = {
        "strategy": strategy, "expert": cfg["expert"], "symbol": cfg["symbol"],
        "period": PERIOD, "from": FROM_DATE, "to": TO_DATE, "model": MODEL,
        "rr_start": rr[0], "rr_stop": rr[1], "rr_step": rr[2],
    }
    path = os.path.join(dest_dir, "_manifest.json")
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                old = json.load(fh)
        except (OSError, ValueError):
            old = {}
        # only the fields that change what the DATA means
        keys = ["expert", "symbol", "period", "from", "to", "model"]
        diff = [k for k in keys if str(old.get(k)) != str(now[k])]
        if diff:
            print("      !! SETTINGS CHANGED since this folder was last filled:")
            for k in diff:
                print(f"         {k}: {old.get(k)!r} -> {now[k]!r}")
            print("         Old files not covered by this sweep will remain and the"
                  " folder will hold MIXED data.")
            print("         Delete the folder first if you want a clean re-run.")
    os.makedirs(dest_dir, exist_ok=True)
    now["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(now, fh, indent=2)


def collect(tag, dest_dir, stats_dir):
    """Move <tag>_*.csv out of MT5's Common\\Files into the project.

    Per-trade files go to dest_dir; the EA's OnTester summaries (*_stats.csv)
    go to stats_dir so they don't confuse the per-trade globs downstream.
    """
    if not os.path.isdir(COMMON_FILES):
        print(f"  ! Common folder not found: {COMMON_FILES}")
        return 0, 0
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    n_tr = n_st = 0
    for f in os.listdir(COMMON_FILES):
        if not (f.startswith(tag + "_") and f.lower().endswith(".csv")):
            continue
        if f.lower().endswith("_stats.csv"):
            shutil.move(os.path.join(COMMON_FILES, f), os.path.join(stats_dir, f))
            n_st += 1
        else:
            shutil.move(os.path.join(COMMON_FILES, f), os.path.join(dest_dir, f))
            n_tr += 1
    return n_tr, n_st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", help="e.g. 11-12 9-10 2-3")
    ap.add_argument("--strategy", default="RR", choices=sorted(STRATEGIES),
                    help="which EA/terminal to run; also picks the <STRAT>_sweeps folder")
    ap.add_argument("--rr", nargs=3, type=float, metavar=("START", "STOP", "STEP"),
                    default=[RR_START, RR_STOP, RR_STEP],
                    help=f"RR sweep, default {RR_START} {RR_STOP} {RR_STEP}")
    ap.add_argument("--dry-run", action="store_true", help="write/print .ini, don't launch")
    ap.add_argument("--list", action="store_true", help="show window -> input mapping")
    a = ap.parse_args()
    rr_start, rr_stop, rr_step = a.rr
    rr = (rr_start, rr_stop, rr_step)

    if a.list:
        print("EA window inputs:")
        for w in ALL_WINDOW_INPUTS:
            print("  ", w)
        print("\nPass windows as H-H, e.g. 2-3 -> W0200W0300, 11-12 -> W1100W1200")
        return
    if not a.windows:
        ap.error("give --windows (or --list)")

    cfg = STRATEGIES[a.strategy]
    n_rr = int(round((rr_stop - rr_start) / rr_step)) + 1
    print(f"strategy {a.strategy}\n"
          f"  terminal {cfg['terminal']}\n"
          f"  expert   {cfg['expert']}   symbol {cfg['symbol']}\n"
          f"{len(a.windows)} window(s) x {n_rr} RR values "
          f"({rr_start}..{rr_stop} step {rr_step}) "
          f"= {len(a.windows) * n_rr} backtests\n"
          f"period {FROM_DATE}..{TO_DATE}  model={MODEL}  "
          f"{'COMPLETE sweep' if OPTIMIZATION_MODE == 1 else 'GENETIC (skips values!)'}")

    if not a.dry_run:
        problems = preflight(a.strategy)
        if problems:
            raise SystemExit("Preflight failed for strategy %s:\n    %s\n"
                             "Fix STRATEGIES[%r] at the top of this script."
                             % (a.strategy, "\n    ".join(problems), a.strategy))
        print("  preflight OK (terminal + expert found)")

    os.makedirs(INI_DIR, exist_ok=True)
    for win in a.windows:
        tag = win                                  # -> files named 2-3_1.00.csv
        ini = os.path.join(INI_DIR, f"{a.strategy}_{win}.ini")
        with open(ini, "w", encoding="utf-8") as fh:
            fh.write(build_ini(win, tag, a.strategy, rr))

        if a.dry_run:
            print(f"\n----- {ini} -----\n{build_ini(win, tag, a.strategy, rr)}")
            print(f"expected files: {tag}_{rr_start:.2f}.csv .. {tag}_{rr_stop:.2f}.csv "
                  f"({n_rr} of them, plus *_stats.csv) in\n  {COMMON_FILES}")
            continue

        # <STRAT>_sweeps/<window>/ is what 1_select_rr.py globs for.
        # (INPUTS/data_2_maemfe_input/<STRAT>/ is the separate, hand-picked set
        # of one-RR-per-window files that step 2 consumes.)
        dest = os.path.join(DEST_ROOT, f"{a.strategy}_sweeps", win)
        stats = os.path.join(DEST_ROOT, f"{a.strategy}_sweeps_stats", win)
        print(f"\n[{win}] launching MT5 ...")
        t0 = time.time()
        subprocess.run([cfg["terminal"], f"/config:{os.path.abspath(ini)}"], check=False)
        check_manifest(dest, a.strategy, rr)
        n_tr, n_st = collect(tag, dest, stats)
        print(f"[{win}] done in {time.time()-t0:,.0f}s — {n_tr} trade CSVs -> {dest}"
              f"   |   {n_st} stats -> {stats}")
        if n_tr == 0:
            dd = data_dir_for(cfg["terminal"])
            print("      ! nothing collected. Most likely causes:")
            print("        - symbol not available in that terminal")
            print("        - date range has no data")
            print("        - EA patch (RunTag + FILE_COMMON) not compiled in")
            if dd:
                print(f"      check the MT5 log: {os.path.join(dd, 'logs')}")

    if not a.dry_run:
        print("\nNext:  venv/Scripts/python.exe 1_select_rr.py")


if __name__ == "__main__":
    main()
