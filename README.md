# MNQ Time-Window RR Study

Tooling to size and allocate a long-only MNQ 30-min breakout strategy across
prop accounts. Two variants are analysed:

- **RR** — enter after the last **red** candle (buy-stop above its high)
- **GG** — enter after the last **green** candle

> The folder is still named `Xgboost_RR_model` for legacy reasons (the project
> began as an XGBoost experiment, since abandoned). The name is cosmetic — don't
> rename the folder; tooling/config is keyed to it.

## How the folders work

Folders are named for **what stage the data is at**, not for which script owns
them. Nothing is an "input folder" or an "output folder" — each script reads one
stage and writes the next:

```
                  ┌─────────────────┐
 MT5  ──0──────►  │ data/1_sweeps   │  every window × every RR
                  └────────┬────────┘
                           │ 1 (--promote)
                  ┌────────▼────────┐
                  │ data/2_chosen   │  the picks: one file per window
                  └────────┬────────┘
                           │ 2
                  ┌────────▼────────┐
                  │ data/3_results  │  analysis CSVs  ◄── 3 reads+writes
                  └────────┬────────┘
                           │ 4
                  ┌────────▼────────┐
                  │ reports/        │  report.html + plots
                  └─────────────────┘
```

| Folder | Holds | Written by | Read by |
|---|---|---|---|
| `data/1_sweeps/<RR\|GG>/<window>/` | per-trade CSV per RR, + `_manifest.json` | 0 | 1 |
| `data/1_sweeps/<RR\|GG>_stats/<window>/` | MT5 `OnTester` summaries (cross-check) | 0 | — |
| `data/2_chosen/<RR\|GG>/` | the selected RR, one file per window | 1 `--promote` | 2 |
| `data/3_results/` | every analysis CSV | 1, 2, 3 | 3, 4 |
| `reports/` | `report.html`, `plots/step*/` | 1, 2, 3, 4 | you |
| `run/mt5_ini/` | generated tester configs (transient) | 0 | MT5 |
| `legacy/` | retired optimization-XML path | — | — |

## Running it

Use the runner — it cleans before it builds, so `data/3_results/` and
`reports/` can never hold a mix of vintages:

```bash
python run_pipeline.py            # steps 1-4 (sweeps skipped), clean first
python run_pipeline.py --status   # is what's on disk current and consistent?
python run_pipeline.py --dry-run  # show what would be deleted and run
python run_pipeline.py --from 2   # only steps 2-4 (e.g. after editing accounts)
```

**Step 0 is excluded by default** — hours of MT5 work, and it's data collection
rather than analysis. `data/1_sweeps/` is never auto-cleaned. Include it only
deliberately:

```bash
python run_pipeline.py --with-sweeps --windows 2-3 3-4 --strategy RR --rr 0.5 3.0 0.1
```

Why a runner rather than running scripts by hand: stopping halfway leaves
`data/3_results/` holding some files from today's step 1 and some from last
week's step 3, with nothing on disk saying which is which — and file
timestamps *can't* tell you, because `promote()` copies preserve the source
file's mtime by design. The runner deletes each step's artefacts before that
step re-runs, and on failure it stops, leaving the later stages **empty**
(honestly "not produced") rather than stale (misleadingly "looks produced").

`--status` answers "is this current?" by walking the provenance chain: each
step embeds the record of the step before it, so a run-id mismatch means that
artefact was built against a different upstream.

## Pipeline

```bash
# 0. sweep every window across an RR range (drives MT5 headlessly)
python 0_run_mt5_sweeps.py --windows 2-3 3-4 --strategy RR --rr 0.5 3.0 0.1

# 1. pick the RR per window, then copy the winners into data/2_chosen/
python 1_select_rr.py --promote recommended --dry-run
python 1_select_rr.py --promote recommended

# 2-4. portfolio, allocation, report
python 2_analyze_maemfe.py
python 3_allocate_accounts.py
python 4_build_report.py
```

### 0 — `0_run_mt5_sweeps.py`
Launches MT5 per window (config `.ini` + `ShutdownTerminal=1`). The RR sweep is
done *by* MT5's optimizer (one pass per RR); this script loops the **windows**.
Each strategy has its own MT5 install, so `STRATEGIES` holds a per-strategy
`terminal` / `expert` / `symbol`. `preflight()` verifies both exist before
launching — a wrong Expert path otherwise makes MT5 exit in ~8s with no output.

### 1 — `1_select_rr.py`
Per-window RR tiers (`recommended` / `aggressive` / `unlocked`) from the real
per-trade sweeps, with the DD cap applied to **equity** drawdown.

Picks are judged by their **±`SMOOTH_RR` neighbourhood** (mean profit, *worst*
neighbour DD), not by a single RR value — TP is close-based, so one 0.01 step can
flip individual trades and create cliffs, and a pick one step from a cliff is
fragile by construction.

Every pass is **cross-checked against MT5's own `_stats.csv`** and the script
aborts before promoting if the two disagree (`--no-validate` overrides).

Also scores **equity-curve shape**: `lr_r` (straightness) and `recent_net`
(profit in the last `RECENT_YEARS`), flagged `ALIVE` / `FADING` / `STALE`.
A **conflict** is a tradeable verdict (OK/WEAK) whose shape is STALE or FADING —
the totals say yes but the edge is historical. Shape is *reported, never
auto-filtered*; drop one by hand with `--exclude RR/5-6`.

`--promote <tier>` copies each qualifying window's chosen file into
`data/2_chosen/` (removing any earlier pick for that window, so step 2 can't
count a window twice). `--verdicts OK WEAK` widens the filter; `--dry-run`
previews.

- **in:**  `data/1_sweeps/<RR|GG>/<window>/` (+ `<STRAT>_stats/` for validation)
- **out:** `data/3_results/rr_pertrade_recommendations.csv`
- **out:** `reports/step1_rr_selection.html` — sortable table + equity curve per
  window, conflicts highlighted, filters incl. **Conflicts only**. This is the
  page that replaces "open MT5 and re-run this window by hand" when a row looks
  contradictory.
- **out:** `reports/step1_rr_sweeps.html` — profit / drawdown vs RR per window,
  with the under-cap band shaded and the tier picks marked (the old PNGs, in one
  page; PNGs still written to `reports/plots/step1_rr_selection/`).

### 2 — `2_analyze_maemfe.py`
Real equity curves, drawdowns, per-year breakdowns, combined portfolio and a
cross-strategy view, from `data/2_chosen/`.

### 3 — `3_allocate_accounts.py`
Exact integer program (`scipy.optimize.milp`): which windows to trade and which
account each goes to, maximising net profit subject to every account's DD limit.
Accounts are strategy-pure (netting-safe); a one-position replay prices in
blocked entries.

### 4 — `4_build_report.py`
One self-contained interactive HTML (Plotly inlined, offline). Per-trade data is
shipped into the page, so **every chart on the Portfolio tab recomputes from the
windows you tick** — including the drawdown subplot.

## EA requirements (both RR and GG)

1. `RunTag=""` → filename derived from the enabled window via
   `ActiveWindowLabel()`, so an export can't be mislabelled.
2. Exports named `<window>_<RR:2dp>.csv`, opened with **`FILE_COMMON`** so every
   tester agent writes to one shared folder.
3. `OnTester()` writes `<window>_<RR>_stats.csv` with MT5's own figures
   (`STAT_EQUITY_DD` etc.) for cross-checking. Note `STAT_LR_CORRELATION` does
   **not** exist in MQL5 — compute curve straightness in Python.

⚠ The `.cs` files in this repo are working copies. MT5 runs the compiled `.ex5`
built from the `.mq5` in its own data folder — patch **there** and recompile, or
the change has no effect. CLI compile:
`metaeditor64.exe /compile:"<abs .mq5>" /log:"<abs .log>"` (exit code is
unreliable; check the `.ex5` timestamp).

## Provenance — "which result is this?"

Every step stamps what produced it, so no artefact is anonymous:

- `provenance.py` supplies a per-interpreter **run id**, the analysis-code git
  revision (+ dirty flag), and file hashes.
- Step 0 records in each `_manifest.json`: run id, data cutoff, sweep
  completeness, and the identity of the **strategy MT5 actually executed** —
  SHA-256 of the deployed `.ex5` and its `.mq5`, whether the `.ex5` is newer
  than its source (stale-compile check), and whether that `.mq5` still matches
  the repo's `*(example).cs`.
- Steps 1–4 each write `data/3_results/_provenance_step<N>.json`, chaining the
  step before it, and print a one-line banner.
- Both step-1 HTMLs and `reports/report.html` carry a **Provenance** panel: run
  ids for the whole chain, data cutoff, validation counts, DD-calibration
  extent, EA hashes, any overrides used, and warnings.

Warnings surfaced automatically: uncommitted analysis code · deployed `.mq5`
differing from the repo copy · `.ex5` older than its `.mq5` · sweeps spanning
more than one data cutoff · unknown EA identity · any override in effect.

> The repo holds `*(example).cs` working copies while MT5 runs a compiled `.ex5`
> elsewhere. Hashing the deployed binary is what keeps "the analysis is
> reproducible" from quietly coexisting with "the strategy is not".

## Notes / gotchas

- MT5 exports: `mae/mfe/trade_profit` are **money**; `candle_range` is **points**.
  UTF-16, tab-separated, no header.
- Backtests run **without costs**; commission ($1/round-turn) is applied in Python.
- **Equity drawdown** walks each trade *MAE first, then MFE* (a buy-stop breakout
  usually retraces before it runs). This reproduces MT5's `STAT_EQUITY_DD`
  exactly — validated on 12 passes. Dropping the MFE term reproduces
  `STAT_BALANCE_DD`. No fudge factor.
- **MT5 timestamps are naive broker time.** They are encoded as
  epoch-treated-as-UTC and must be decoded with **UTC getters** (`getUTCHours`,
  `getUTCDay`, …). Using local-time getters re-interprets them in the viewer's
  zone — on a UTC+2/+3 machine window "2-3" showed up as hour 4-5, and the shift
  is DST-dependent so it also moved trades across day/month boundaries.
- **DD calibration flows step 1 → 2 → 3** via `data/3_results/dd_calibration.csv`.
  Step 1 measures how far our equity-DD walk sits below MT5's `STAT_EQUITY_DD`
  per pass; steps 2 and 3 scale by it (step 3 uses the *worst* factor among an
  account's windows, since MT5 never tested combinations). Without this the RR
  pick is cautious while the account DD constraint stays optimistic.
- **Unverified data cannot be promoted.** Step 0 records `complete` in each
  `_manifest.json`; step 1 blocks promotion of incomplete sweeps or windows with
  no `_stats.csv` unless you pass `--allow-unvalidated`.
- **Keep every export pinned to one common end date.** Every DD discrepancy this
  project hit traced back to data generated at different times. Step 0 writes a
  `_manifest.json` per window and warns if a re-run changes symbol/dates/model.
- File collection **overwrites** same-named files silently — intended, but delete
  a sweep folder first if you want a guaranteed-clean re-run.
- Drawdowns are historical; future DD can exceed them, hence the sub-100% cap
  (default 85%) in step 3.
- Prop DD limit is trailing and **freezes** once the account banks its buffer, so
  the static-limit view is conservative once an account is seasoned.

## Legacy

`legacy/optimization_xml/` and `archive/analyze_optimization_xml.py` are the
retired path that parsed MT5 optimization XMLs. Those are summary-only and had to
be regenerated in lockstep with the per-trade exports to stay comparable — the
failure mode that motivated moving step 1 onto per-trade data.
