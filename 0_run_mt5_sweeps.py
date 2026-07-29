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
1b_rr_from_maemfe.py.

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
import os
import shutil
import subprocess
import sys
import time

# ---- CONFIG: set these once -------------------------------------------------
TERMINAL_EXE = r"I:\Programs\1AMP Global (USA) MT5 Exchange-Traded Futures Only\terminal64.exe"
# One compiled EA per strategy. Paths are relative to MQL5\Experts\ in the MT5
# DATA folder (File > Open Data Folder), and must be the .ex5, not the source.
EA_PATHS = {
    "RR": r"RR_r_MFE_buy-stop-entry.ex5",
    "GG": r"GG_r_MFE_buy-stop-entry(example).ex5",
}
SYMBOL = "MNQcontDATABENTOcurr6"
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


def build_ini(win, tag, strategy, rr):
    on = set(inputs_for_window(win))
    rr_start, rr_stop, rr_step = rr
    lines = [
        "[Tester]",
        f"Expert={EA_PATHS[strategy]}",
        f"Symbol={SYMBOL}",
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


def collect(tag, dest_dir):
    """Move <tag>_*.csv out of the MT5 Common\\Files folder into the project."""
    if not os.path.isdir(COMMON_FILES):
        print(f"  ! Common folder not found: {COMMON_FILES}")
        return 0
    os.makedirs(dest_dir, exist_ok=True)
    n = 0
    for f in os.listdir(COMMON_FILES):
        if f.startswith(tag + "_") and f.lower().endswith(".csv"):
            shutil.move(os.path.join(COMMON_FILES, f), os.path.join(dest_dir, f))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", help="e.g. 11-12 9-10 2-3")
    ap.add_argument("--strategy", default="RR", choices=sorted(EA_PATHS),
                    help="which EA to run; also picks the <STRAT>_sweeps folder")
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

    n_rr = int(round((rr_stop - rr_start) / rr_step)) + 1
    print(f"strategy {a.strategy} ({EA_PATHS[a.strategy]})\n"
          f"{len(a.windows)} window(s) x {n_rr} RR values "
          f"({rr_start}..{rr_stop} step {rr_step}) "
          f"= {len(a.windows) * n_rr} backtests\n"
          f"period {FROM_DATE}..{TO_DATE}  model={MODEL}  "
          f"{'COMPLETE sweep' if OPTIMIZATION_MODE == 1 else 'GENETIC (skips values!)'}")

    if not a.dry_run and not os.path.isfile(TERMINAL_EXE):
        raise SystemExit(f"terminal64.exe not found at {TERMINAL_EXE}\n"
                         "Edit TERMINAL_EXE at the top of this script.")

    os.makedirs(INI_DIR, exist_ok=True)
    for win in a.windows:
        tag = win                                  # -> files named 2-3_1.00.csv
        ini = os.path.join(INI_DIR, f"{a.strategy}_{win}.ini")
        with open(ini, "w", encoding="utf-8") as fh:
            fh.write(build_ini(win, tag, a.strategy, rr))

        if a.dry_run:
            print(f"\n----- {ini} -----\n{build_ini(win, tag, a.strategy, rr)}")
            print(f"expected files: {tag}_{rr_start:.2f}.csv .. {tag}_{rr_stop:.2f}.csv "
                  f"({n_rr} of them) in\n  {COMMON_FILES}")
            continue

        dest = os.path.join(DEST_ROOT, f"{a.strategy}_sweeps", win)
        print(f"\n[{win}] launching MT5 ...")
        t0 = time.time()
        subprocess.run([TERMINAL_EXE, f"/config:{os.path.abspath(ini)}"], check=False)
        got = collect(tag, dest)
        print(f"[{win}] done in {time.time()-t0:,.0f}s — collected {got} CSVs -> {dest}")
        if got == 0:
            print("      ! nothing collected. Check that the EA patch is applied "
                  "(RunTag + FILE_COMMON) and that RunTag matched the window label.")

    if not a.dry_run:
        print("\nNext:  venv/Scripts/python.exe 1b_rr_from_maemfe.py")


if __name__ == "__main__":
    main()
