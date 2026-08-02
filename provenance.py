"""
provenance.py
=============
Shared run-provenance helpers, so every artefact can answer "which result is
this, and what produced it?" without archaeology.

Two problems this exists to prevent:

1. "Which result is current?" — a CSV or HTML on disk looks identical whether it
   came from today's clean sweep or a half-finished one from last week. Every
   step now writes a `_provenance.json` sidecar and stamps the HTML footer with
   a run id, the data cutoff, validation counts and any override flags used.

2. "The analysis is reproducible but the STRATEGY isn't." — the repo holds
   `*(example).cs` working copies while MT5 actually runs a compiled `.ex5`
   somewhere else entirely. Hashing the deployed `.ex5` and its `.mq5`, and
   diffing that `.mq5` against the repo copy, makes a drifted EA visible instead
   of silently invalidating everything downstream.

Nothing here changes any number — it only records what produced them.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

# One id per interpreter run; every artefact written by that run carries it.
RUN_ID = time.strftime("%Y%m%dT%H%M%S") + "-" + hashlib.sha1(
    (str(time.time()) + str(os.getpid())).encode()).hexdigest()[:6]


def sha16(path):
    """First 16 hex chars of the file's SHA-256 — enough to spot a change."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


def file_info(path, label=None):
    """Identity of a file: hash, size, mtime. None-safe for missing files."""
    if not path or not os.path.isfile(path):
        return {"label": label, "path": path, "exists": False}
    st = os.stat(path)
    return {
        "label": label,
        "path": os.path.abspath(path),
        "exists": True,
        "sha256_16": sha16(path),
        "bytes": st.st_size,
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
    }


def git_info():
    """Repo revision of the ANALYSIS code (not the EA — see ea_info)."""
    def _run(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None
    commit = _run("git", "rev-parse", "--short", "HEAD")
    if commit is None:
        return {"available": False}
    status = _run("git", "status", "--porcelain")
    return {
        "available": True,
        "commit": commit,
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "dirty_files": len(status.splitlines()) if status else 0,
    }


def normalise_source(path):
    """Text of an MQL5 source, normalised so encoding/line-endings don't matter.

    The repo copies are UTF-8/CRLF working files while MetaEditor writes UTF-8
    with BOM — comparing raw bytes would report a difference that isn't one.
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    for enc in ("utf-8-sig", "utf-16", "utf-8", "latin-1"):
        try:
            txt = raw.decode(enc)
            break
        except Exception:
            continue
    else:
        return None
    return "\n".join(line.rstrip() for line in txt.splitlines()).strip()


def ea_info(terminal_exe, expert_rel, data_dir, repo_source=None):
    """Identity of the DEPLOYED strategy, plus drift vs the repo copy.

    `expert_rel` is what the tester .ini references, e.g. "444\\RR_...ex5".
    We hash that .ex5, the .mq5 beside it, and compare the .mq5 against the
    repo's example file — because "analysis reproducible, strategy unknown" is
    the failure mode worth catching.
    """
    out = {"terminal": file_info(terminal_exe, "terminal64.exe"),
           "expert_rel": expert_rel}
    ex5 = mq5 = None
    if data_dir and expert_rel:
        ex5 = os.path.join(data_dir, "MQL5", "Experts",
                           expert_rel.replace("\\", os.sep))
        mq5 = os.path.splitext(ex5)[0] + ".mq5"
    out["expert_ex5"] = file_info(ex5, "deployed .ex5")
    out["expert_mq5"] = file_info(mq5, "deployed .mq5")
    out["repo_source"] = file_info(repo_source, "repo copy")

    # Is the compiled binary newer than its source? (i.e. was it recompiled
    # after the last edit — a stale .ex5 silently runs old logic.)
    out["ex5_newer_than_mq5"] = None
    try:
        if ex5 and mq5 and os.path.isfile(ex5) and os.path.isfile(mq5):
            out["ex5_newer_than_mq5"] = os.path.getmtime(ex5) >= os.path.getmtime(mq5)
    except OSError:
        pass

    # Does the deployed source still match the repo copy?
    out["repo_matches_deployed"] = None
    if repo_source and mq5:
        a, b = normalise_source(repo_source), normalise_source(mq5)
        if a is not None and b is not None:
            out["repo_matches_deployed"] = (a == b)
    return out


def base(step, **extra):
    """Common header for any artefact this pipeline writes."""
    rec = {
        "run_id": RUN_ID,
        "step": step,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_info(),
        "python": sys.version.split()[0],
        "host": platform.node(),
        "cwd": os.getcwd(),
    }
    rec.update(extra)
    return rec


def write(path, rec):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, default=str)
    return path


def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def summary_line(rec):
    """One-line console banner: the 'which result is this' answer at a glance."""
    g = rec.get("git") or {}
    bits = [f"run {rec.get('run_id')}"]
    if g.get("available"):
        bits.append(f"code {g.get('commit')}{'+dirty' if g.get('dirty') else ''}")
    if rec.get("data_cutoff"):
        bits.append(f"data->{rec['data_cutoff']}")
    if rec.get("validated_passes") is not None:
        bits.append(f"validated {rec['validated_passes']}")
    ov = [k for k, v in (rec.get("overrides") or {}).items() if v]
    if ov:
        bits.append("OVERRIDES: " + ",".join(ov))
    return " | ".join(bits)


def warnings_for(rec):
    """Human-readable red flags worth printing/showing, if any."""
    out = []
    g = rec.get("git") or {}
    if g.get("available") and g.get("dirty"):
        out.append(f"analysis code has {g.get('dirty_files')} uncommitted change(s)")
    for st, ea in (rec.get("ea") or {}).items():
        if ea.get("repo_matches_deployed") is False:
            out.append(f"{st}: deployed .mq5 DIFFERS from the repo copy")
        if ea.get("ex5_newer_than_mq5") is False:
            out.append(f"{st}: .ex5 is OLDER than its .mq5 — recompile needed")
        if not (ea.get("expert_ex5") or {}).get("exists", True):
            out.append(f"{st}: deployed .ex5 not found")
    unknown = rec.get("ea_unknown_windows") or []
    if unknown:
        out.append(f"EA identity unknown for {len(unknown)} window(s) — their sweep "
                   f"manifests predate provenance tracking; re-run step 0 to capture "
                   f"the deployed .ex5 hash")
    for k, v in (rec.get("overrides") or {}).items():
        if v:
            out.append(f"override in effect: {k}={v}")
    return out
