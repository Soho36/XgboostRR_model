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
  python 0_run_mt5_sweeps.py --windows 11-12 --strategy RR --rr 1.0 2.0 0.1
  python 0_run_mt5_sweeps.py --windows all --strategy RR --rr 1.0 2.0 0.1
  python 0_run_mt5_sweeps.py --windows 1-2 2-3 3-4 4-5 5-6 6-7 7-8 8-9 9-10 10-11 11-12 12-13 13-14 14-15 15-16 16-17 17-18 18-19 19-20 20-21 21-22 22-23 23-24 --strategy RR --rr 0.5 3.0 0.1
  python 0_run_mt5_sweeps.py --windows 1-2 2-3 3-4 4-5 5-6 6-7 7-8 8-9 9-10 10-11 11-12 12-13 13-14 14-15 15-16 16-17 17-18 18-19 19-20 20-21 21-22 22-23 23-24 --strategy GG --rr 0.5 3.0 0.1
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
# import sys
import time

import provenance as prov

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
        # repo working copy, compared against the DEPLOYED .mq5 to catch drift
        "repo_source": "RR_r_MFE_buy-stop-entry(example).cs",
    },
    "GG": {
        "terminal": r"I:\Programs\1MetaTrader 5\terminal64.exe",
        "expert": r"555\GG_r_MFE_buy-stop-entry.ex5",
        "symbol": "MNQcontDATABENTOcurr6",
        "repo_source": "GG_r_MFE_buy-stop-entry(example).cs",
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
RR_START, RR_STEP, RR_STOP = 1.0, 0.1, 2.00

# 1 = SLOW COMPLETE algorithm. Do NOT use 2 (genetic) — it skips RR values.
OPTIMIZATION_MODE = 1

COMMON_FILES = os.path.join(os.environ.get("APPDATA", ""),
                            "MetaQuotes", "Terminal", "Common", "Files")
SWEEPS_DIR = "data/1_sweeps"      # -> <STRAT>/<window>/ and <STRAT>_stats/<window>/
INI_DIR = "run/mt5_ini"

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
        "all": ALL_WINDOW_INPUTS,
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


def manifest_now(strategy, rr):
    cfg = STRATEGIES[strategy]
    return {
        "strategy": strategy, "expert": cfg["expert"], "symbol": cfg["symbol"],
        "period": PERIOD, "from": FROM_DATE, "to": TO_DATE, "model": MODEL,
        "rr_start": rr[0], "rr_stop": rr[1], "rr_step": rr[2],
    }


def manifest_warn(dest_dir, strategy, rr):
    """Compare (do NOT write) the folder's provenance against this run.

    Collection overwrites same-named files silently, so re-running a window with
    a different date range / symbol / model leaves a folder holding a MIX of two
    periods — the staleness class of bug that has bitten this project repeatedly.
    """
    path = os.path.join(dest_dir, "_manifest.json")
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as fh:
            old = json.load(fh)
    except (OSError, ValueError):
        return
    now = manifest_now(strategy, rr)
    keys = ["expert", "symbol", "period", "from", "to", "model"]   # data-defining
    diff = [k for k in keys if str(old.get(k)) != str(now[k])]
    if diff:
        print("      !! SETTINGS CHANGED since this folder was last filled:")
        for k in diff:
            print(f"         {k}: {old.get(k)!r} -> {now[k]!r}")
        print("         Files not overwritten by this sweep will remain -> MIXED data.")
        print("         Delete the folder first if you want a clean re-run.")


def manifest_write(dest_dir, strategy, rr, n_files, expected):
    """Written only AFTER a successful collection, so a failed run can never
    leave old data stamped with new settings.

    Records completeness explicitly: a partial sweep still leaves usable files,
    but downstream must be able to see that the folder is a MIX of runs rather
    than one clean sweep. Step 1 refuses to promote from an incomplete folder.
    """
    os.makedirs(dest_dir, exist_ok=True)
    cfg = STRATEGIES[strategy]
    rec = manifest_now(strategy, rr)
    rec["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    rec["files_collected"] = n_files
    rec["expected"] = expected
    rec["complete"] = bool(n_files == expected)
    # WHO produced this data: run id, analysis-code revision, and the identity of
    # the strategy MT5 actually executed (not the repo copy of it).
    rec["run_id"] = prov.RUN_ID
    rec["git"] = prov.git_info()
    rec["ea"] = prov.ea_info(cfg["terminal"], cfg["expert"],
                             data_dir_for(cfg["terminal"]),
                             cfg.get("repo_source"))
    with open(os.path.join(dest_dir, "_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)


def mt5_log_tail(terminal_exe, nbytes=400_000):
    # Generous window: a *successful* optimization writes a line per pass (300+),
    # which can flush an earlier LiveUpdate handoff out of a small tail.
    """Tail of that terminal's newest log (UTF-8/UTF-16 tolerant)."""
    dd = data_dir_for(terminal_exe)
    if not dd:
        return ""
    # Terminal logs are named YYYYMMDD.log; the folder also holds metaeditor.log,
    # which sorts AFTER them alphabetically — so filter, then take newest by mtime.
    logs = [p for p in glob.glob(os.path.join(dd, "logs", "*.log"))
            if os.path.basename(p)[:8].isdigit()]
    if not logs:
        return ""
    logs.sort(key=os.path.getmtime)
    try:
        with open(logs[-1], "rb") as fh:
            fh.seek(0, os.SEEK_END)
            start = max(0, fh.tell() - nbytes)
            fh.seek(start - (start % 2))        # keep UTF-16 code units aligned
            raw = fh.read()
    except OSError:
        return ""
    # MT5 writes these logs as UTF-16LE. Decoding as UTF-8 with errors="ignore"
    # never raises — it silently mangles the text and substring searches fail —
    # so pick the encoding from NUL density rather than guessing.
    if raw.count(b"\x00") > len(raw) // 4:
        return raw.decode("utf-16-le", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def count_new(tag, since):
    """Per-trade files for `tag` written at/after `since` (stats excluded)."""
    if not os.path.isdir(COMMON_FILES):
        return 0
    n = 0
    for f in os.listdir(COMMON_FILES):
        low = f.lower()
        if not (f.startswith(tag + "_") and low.endswith(".csv")) or low.endswith("_stats.csv"):
            continue
        try:
            if os.path.getmtime(os.path.join(COMMON_FILES, f)) >= since - 2:
                n += 1
        except OSError:
            pass
    return n


def wait_for_sweep(tag, n_expected, since, poll=3.0, quiet_for=120.0, hard_timeout=7200):
    """subprocess.run() returning does NOT mean the test actually ran.

    MT5's LiveUpdate can hand our /config to a second process and exit at once
    (log: 'LiveUpdate start ... /update', then 'terminal process already
    started'). The real test then finishes minutes later — after we have moved
    on to the next window. So wait on the FILES, not on the process exiting.

    Returns (files_seen, status) where status is complete | stalled | nothing.
    """
    t_start = time.time()
    last_n, last_change = -1, time.time()
    while True:
        n = count_new(tag, since)
        if n >= n_expected:
            return n, "complete"
        if n != last_n:
            last_n, last_change = n, time.time()
        if time.time() - last_change > quiet_for:
            return n, ("stalled" if n else "nothing")
        if time.time() - t_start > hard_timeout:
            return n, "timeout"
        time.sleep(poll)


def liveupdate_hijack(terminal_exe):
    """True when the newest log shows MT5 relaunching itself for an update."""
    tail = mt5_log_tail(terminal_exe)
    return ("LiveUpdate" in tail and "/update" in tail) or \
           ("terminal process already started" in tail)


def purge_common(tag):
    """Remove stale <tag>_*.csv from MT5's SHARED Common\\Files before a run.

    Both strategies write there and filenames carry only the window (no strategy,
    no run id), so a leftover 2-3_1.50.csv from an interrupted RR run would be
    collected into the next GG run for 2-3. Anything matching is stale by
    definition here — we are about to regenerate it.
    """
    if not os.path.isdir(COMMON_FILES):
        return 0
    n = 0
    for f in os.listdir(COMMON_FILES):
        if f.startswith(tag + "_") and f.lower().endswith(".csv"):
            try:
                os.remove(os.path.join(COMMON_FILES, f))
                n += 1
            except OSError:
                pass
    if n:
        print(f"      (purged {n} stale {tag}_*.csv from Common\\Files before running)")
    return n


def collect(tag, dest_dir, stats_dir, since):
    """Move <tag>_*.csv out of MT5's Common\\Files into the project.

    `since` = the moment this run's MT5 was launched. Only files written at or
    after that are taken, so anything left behind by an earlier/other-strategy
    run cannot be absorbed into this one (filenames carry no strategy or run id).

    Per-trade files go to dest_dir; the EA's OnTester summaries (*_stats.csv)
    go to stats_dir so they don't confuse the per-trade globs downstream.
    """
    if not os.path.isdir(COMMON_FILES):
        print(f"  ! Common folder not found: {COMMON_FILES}")
        return 0, 0, 0
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    n_tr = n_st = n_skip = 0
    for f in os.listdir(COMMON_FILES):
        if not (f.startswith(tag + "_") and f.lower().endswith(".csv")):
            continue
        src = os.path.join(COMMON_FILES, f)
        try:
            if os.path.getmtime(src) < since - 2:      # 2s slack for clock/FS jitter
                n_skip += 1
                continue
        except OSError:
            continue
        if f.lower().endswith("_stats.csv"):
            shutil.move(src, os.path.join(stats_dir, f))
            n_st += 1
        else:
            shutil.move(src, os.path.join(dest_dir, f))
            n_tr += 1
    return n_tr, n_st, n_skip


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
        _ea = prov.ea_info(cfg["terminal"], cfg["expert"],
                           data_dir_for(cfg["terminal"]), cfg.get("repo_source"))
        _ex5 = _ea.get("expert_ex5") or {}
        print(f"  run {prov.RUN_ID} | EA {_ex5.get('sha256_16')} "
              f"(built {_ex5.get('mtime')})")
        if _ea.get("ex5_newer_than_mq5") is False:
            print("  !! the .ex5 is OLDER than its .mq5 — MT5 will run STALE logic. "
                  "Recompile before sweeping.")
        if _ea.get("repo_matches_deployed") is False:
            print("  !! the deployed .mq5 DIFFERS from the repo copy "
                  f"({cfg.get('repo_source')}) — analysis would be reproducible "
                  "but the strategy would not.")

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

        # data/1_sweeps/<STRAT>/<window>/ is what 1_select_rr.py globs for;
        # its --promote then copies the winners on to data/2_chosen/<STRAT>/.
        dest = os.path.join(SWEEPS_DIR, a.strategy, win)
        stats = os.path.join(SWEEPS_DIR, f"{a.strategy}_stats", win)
        print(f"\n[{win}] launching MT5 ...")
        manifest_warn(dest, a.strategy, rr)     # compare only; write after success
        purge_common(tag)                       # drop leftovers from any earlier run
        t0 = time.time()
        subprocess.run([cfg["terminal"], f"/config:{os.path.abspath(ini)}"], check=False)

        # The process exiting proves nothing — wait until the files are actually there.
        seen, status = wait_for_sweep(tag, n_rr, t0)
        if status != "complete":
            if liveupdate_hijack(cfg["terminal"]):
                raise SystemExit(
                    f"\nABORTING at [{win}] — MT5 LiveUpdate is hijacking the launch.\n"
                    f"  The terminal wants to update itself, fails to replace its own\n"
                    f"  exe, spawns an updater with our /config and exits immediately,\n"
                    f"  so the tester never runs (or runs minutes later, unattended).\n\n"
                    f"  FIX: open this terminal by hand and let the update finish:\n"
                    f"    {cfg['terminal']}\n"
                    f"  Restart it, confirm Help > About shows the new build, close it,\n"
                    f"  then re-run this script. Nothing here is salvageable until then.")
            print(f"      ! run did not complete ({status}): {seen}/{n_rr} files after "
                  f"{time.time()-t0:,.0f}s")

        n_tr, n_st, n_skip = collect(tag, dest, stats, since=t0)
        print(f"[{win}] done in {time.time()-t0:,.0f}s — {n_tr} trade CSVs -> {dest}"
              f"   |   {n_st} stats -> {stats}")
        if n_skip:
            print(f"      (ignored {n_skip} {tag}_*.csv older than this run — "
                  f"not produced by it)")
        if n_tr:
            manifest_write(dest, a.strategy, rr, n_tr, n_rr)
        if n_tr and n_tr != n_rr:
            print(f"      ! expected {n_rr} trade CSVs but got {n_tr} — the sweep "
                  f"may be incomplete; this folder now holds a MIX of runs.")
        if n_tr == 0:
            dd = data_dir_for(cfg["terminal"])
            print("      ! nothing collected. Most likely causes:")
            print("        - clear cache in MT5 terminal folder (remove optimizatons .opt)")
            print("        - symbol not available in that terminal")
            print("        - date range has no data")
            print("        - EA patch (RunTag + FILE_COMMON) not compiled in")
            if dd:
                print(f"      check the MT5 log: {os.path.join(dd, 'logs')}")

    if not a.dry_run:
        print("\nNext:  venv/Scripts/python.exe 1_select_rr.py")


if __name__ == "__main__":
    main()
