"""
run_pipeline.py
===============
Run the whole pipeline as ONE operation, so "which files are current?" always
has an answer.

The problem it solves: running steps by hand, stopping halfway, and coming back
later leaves `data/3_results/` and `reports/` holding a MIX of vintages — some
files from today's step 1, some from last week's step 3 — and nothing on disk
says which is which. Timestamps don't help either, because `promote()` copies
preserve the SOURCE file's mtime by design.

This runner:
  * DELETES every artefact of the steps it is about to run, before running them,
    so a stale file can never survive into a new run;
  * ABORTS if any owned artefact cannot be removed (for example because Excel
    has it open).  A pipeline run never falls back to timestamped filenames:
    its downstream steps must consume the canonical files from this run;
  * runs the steps in order and stops at the first failure — leaving the
    already-cleaned later stages EMPTY rather than stale, which is the honest
    state ("not produced yet") instead of a misleading one ("looks produced");
  * records the whole thing in `data/3_results/_pipeline.json`;
  * can tell you afterwards, via `--status`, whether every step on disk came
    from the same run — by checking each step's provenance chain against the
    step before it.

Step 0 (the MT5 sweeps) is EXCLUDED by default: it is hours of MT5 work, it is
data collection rather than analysis, and its output (`data/1_sweeps/`) is never
touched by any cleanup here.

USAGE
  python run_pipeline.py                       # steps 1-4, clean first
  python run_pipeline.py --status              # what is on disk, is it consistent?
  python run_pipeline.py --dry-run             # show what would be deleted/run
  python run_pipeline.py --from 2              # only steps 2-4
  python run_pipeline.py --promote aggressive  # pass a tier through to step 1
  python run_pipeline.py --with-sweeps --windows 2-3 3-4 --strategy RR
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import provenance as prov

RESULTS = "data/3_results"
REPORTS = "reports"
CHOSEN = "data/2_chosen"
PIPELINE_JSON = os.path.join(RESULTS, "_pipeline.json")


class CleanupError(RuntimeError):
    """An old artefact could not be removed, so a clean run is impossible."""

# What each step OWNS. Anything listed here is deleted before that step re-runs,
# so the folder can only ever contain output from the current run.
# `data/1_sweeps/` is deliberately absent — that is step 0's expensive output.
STEPS = {
    0: {
        "script": "0_run_mt5_sweeps.py",
        "name": "MT5 sweeps",
        "owns": [],                       # never auto-cleaned: hours of work
    },
    1: {
        "script": "1_select_rr.py",
        "name": "RR selection + promote",
        "owns": [
            f"{CHOSEN}/*/*.csv",
            # Also remove timestamped fallbacks from older manual runs.
            f"{RESULTS}/rr_pertrade_recommendations*.csv",
            f"{RESULTS}/dd_calibration.csv",
            f"{RESULTS}/_provenance_step1.json",
            f"{REPORTS}/step1_rr_selection.html",
            f"{REPORTS}/step1_rr_sweeps.html",
            f"{REPORTS}/plots/step1_rr_selection",
        ],
    },
    2: {
        "script": "2_analyze_maemfe.py",
        "name": "portfolio analysis",
        "owns": [
            f"{RESULTS}/*_maemfe_window_summary*.csv",
            f"{RESULTS}/*_maemfe_combined_trades*.csv",
            f"{RESULTS}/_provenance_step2.json",
            f"{REPORTS}/plots/step2_portfolio",
        ],
    },
    3: {
        "script": "3_allocate_accounts.py",
        "name": "account allocation",
        "owns": [
            f"{RESULTS}/multi_strategy_allocation*.csv",
            f"{RESULTS}/_provenance_step3.json",
            f"{REPORTS}/plots/step3_allocation",
        ],
    },
    4: {
        "script": "4_build_report.py",
        "name": "interactive report",
        "owns": [
            f"{REPORTS}/report.html",
            f"{RESULTS}/_provenance_step4.json",
        ],
    },
}


def expand(patterns):
    """Files and directories matching the step's ownership patterns."""
    out = []
    for pat in patterns:
        out.extend(sorted(glob.glob(pat)))
    return out


def clean(step, dry_run):
    targets = expand(STEPS[step]["owns"])
    if not targets:
        return 0
    label = "would delete" if dry_run else "deleted"
    failures = []
    for t in targets:
        if not dry_run:
            try:
                shutil.rmtree(t) if os.path.isdir(t) else os.remove(t)
            except OSError as e:
                print(f"      ! could not delete {t}: {e}")
                failures.append((t, e))
    if failures:
        removed = len(targets) - len(failures)
        print(f"    cleanup incomplete: {removed} item(s) {label}, "
              f"{len(failures)} could not be removed")
        names = ", ".join(t for t, _ in failures)
        raise CleanupError(
            "Cannot start a clean pipeline run: close any program holding "
            f"these artefacts, then rerun: {names}"
        )
    print(f"    {label} {len(targets)} item(s)")
    return len(targets)


def run(step, extra_args, dry_run):
    cfg = STEPS[step]
    cmd = [sys.executable, cfg["script"]] + extra_args
    print(f"\n{'=' * 78}\nSTEP {step} — {cfg['name']}\n  $ {' '.join(cmd)}\n{'=' * 78}")
    if dry_run:
        print("  (dry run — not executed)")
        return True, 0.0
    # The standalone scripts retain their convenient timestamped fallback for
    # ad-hoc/manual use.  In a pipeline that would make downstream consumers
    # read an old canonical file, so require canonical output instead.
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PIPELINE_RUN="1")
    t0 = time.time()
    rc = subprocess.run(cmd, env=env).returncode
    dt = time.time() - t0
    print(f"  -> exit {rc} in {dt:,.0f}s")
    return rc == 0, dt


def status():
    """Is everything on disk from the same pipeline run?

    Each step's provenance embeds the FULL record of the step before it, so a
    mismatch between step N's `upstream.run_id` and step N-1's own `run_id`
    means step N was produced against a different (older) upstream — i.e. the
    artefacts on disk are a mix of vintages.
    """
    print("=" * 78)
    print("PIPELINE STATUS")
    print("=" * 78)
    recs = {}
    for n in (1, 2, 3, 4):
        recs[n] = prov.load(os.path.join(RESULTS, f"_provenance_step{n}.json"))

    if not any(recs.values()):
        print("  nothing has been run yet (no provenance sidecars found)")
        return 1

    print(f"  {'step':<26}{'run id':<26}{'generated':<21}status")
    ok = True
    prev = None
    for n in (1, 2, 3, 4):
        r = recs[n]
        name = f"{n} {STEPS[n]['name']}"
        if r is None:
            print(f"  {name:<26}{'-':<26}{'-':<21}NOT RUN")
            ok = False
            prev = None
            continue
        note = "current"
        if prev is not None:
            up = (r.get("upstream") or {}).get("run_id")
            if up != prev:
                note = f"STALE (built on {up or 'unknown'}, not {prev})"
                ok = False
        print(f"  {name:<26}{r.get('run_id',''):<26}{str(r.get('generated','')):<21}{note}")
        prev = r.get("run_id")

    pl = prov.load(PIPELINE_JSON)
    if pl:
        print(f"\n  last pipeline run: {pl.get('run_id')} "
              f"({pl.get('generated')}) — steps {pl.get('steps_run')}, "
              f"{'COMPLETED' if pl.get('completed') else 'FAILED at step ' + str(pl.get('failed_at'))}")
    print("\n  " + ("All steps present and consistent — reports are current."
                    if ok else
                    "INCONSISTENT — re-run `python run_pipeline.py` to rebuild cleanly."))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true",
                    help="report what is on disk and whether it is consistent")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be cleaned and run, change nothing")
    ap.add_argument("--from", dest="start", type=int, default=1, choices=[0, 1, 2, 3, 4],
                    help="first step to run (default 1 — sweeps are skipped)")
    ap.add_argument("--to", dest="end", type=int, default=4, choices=[0, 1, 2, 3, 4],
                    help="last step to run (default 4)")
    ap.add_argument("--with-sweeps", action="store_true",
                    help="include step 0 (slow: hours of MT5 work)")
    ap.add_argument("--promote", default="recommended",
                    help="tier passed to step 1 (default: recommended)")
    ap.add_argument("--verdicts", nargs="+", default=None,
                    help="verdicts passed to step 1 (default: its own default)")
    ap.add_argument("--exclude", nargs="+", default=None,
                    help="windows excluded from promotion in step 1")
    ap.add_argument("--allow-unvalidated", action="store_true",
                    help="let step 1 promote unverified/incomplete windows")
    # step-0 passthrough
    ap.add_argument("--windows", nargs="+", default=None)
    ap.add_argument("--strategy", default=None)
    ap.add_argument("--rr", nargs=3, default=None, metavar=("START", "STOP", "STEP"))
    a = ap.parse_args()

    if a.status:
        sys.exit(status())

    start = 0 if a.with_sweeps else a.start
    steps = [n for n in (0, 1, 2, 3, 4) if start <= n <= a.end]
    if 0 in steps and not a.windows:
        ap.error("step 0 needs --windows (and usually --strategy/--rr)")

    print("=" * 78)
    print(f"PIPELINE RUN {prov.RUN_ID}   steps {steps}"
          f"{'   (DRY RUN)' if a.dry_run else ''}")
    print("=" * 78)
    if 0 not in steps:
        print("  step 0 (MT5 sweeps) SKIPPED — data/1_sweeps/ left untouched")

    # Clean everything we are about to rebuild, FIRST. Doing it all up front
    # means a mid-pipeline failure leaves later stages empty (honest) rather
    # than holding output from a previous run (misleading).
    print("\nCleaning artefacts of the steps about to run:")
    try:
        for n in steps:
            if STEPS[n]["owns"]:
                print(f"  step {n} ({STEPS[n]['name']}):")
                clean(n, a.dry_run)
            else:
                print(f"  step {n} ({STEPS[n]['name']}): nothing auto-cleaned (expensive)")
    except CleanupError as e:
        print(f"\n!! CLEANUP FAILED — no pipeline steps were started.\n   {e}")
        sys.exit(1)

    results, failed_at = {}, None
    for n in steps:
        extra = []
        if n == 0:
            extra += ["--windows"] + list(a.windows)
            if a.strategy:
                extra += ["--strategy", a.strategy]
            if a.rr:
                extra += ["--rr"] + list(a.rr)
        elif n == 1:
            extra += ["--promote", a.promote]
            if a.verdicts:
                extra += ["--verdicts"] + list(a.verdicts)
            if a.exclude:
                extra += ["--exclude"] + list(a.exclude)
            if a.allow_unvalidated:
                extra += ["--allow-unvalidated"]
        ok, dt = run(n, extra, a.dry_run)
        results[n] = {"ok": ok, "seconds": round(dt, 1)}
        if not ok:
            failed_at = n
            print(f"\n!! step {n} FAILED — stopping. Steps after it were cleaned and "
                  f"are intentionally EMPTY, not stale.")
            break

    if not a.dry_run:
        prov.write(PIPELINE_JSON, prov.base(
            "run_pipeline", steps_run=steps, per_step=results,
            completed=failed_at is None, failed_at=failed_at,
            skipped_sweeps=0 not in steps,
            args={"promote": a.promote, "verdicts": a.verdicts,
                  "exclude": a.exclude, "allow_unvalidated": a.allow_unvalidated}))

    print("\n" + "=" * 78)
    if failed_at is None:
        total = sum(v["seconds"] for v in results.values())
        print(f"PIPELINE COMPLETE in {total:,.0f}s — every artefact in "
              f"{RESULTS}/ and {REPORTS}/ is from run {prov.RUN_ID}")
        print(f"  open {REPORTS}/report.html")
    else:
        print(f"PIPELINE INCOMPLETE — failed at step {failed_at}")
        print("  fix the error, then re-run `python run_pipeline.py`")
    print("=" * 78)
    sys.exit(0 if failed_at is None else 1)


if __name__ == "__main__":
    main()
