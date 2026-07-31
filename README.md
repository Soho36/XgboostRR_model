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

## Pipeline

```bash
# 0. sweep every window across an RR range (drives MT5 headlessly)
venv/Scripts/python.exe 0_run_mt5_sweeps.py --windows 2-3 3-4 --strategy RR --rr 0.5 3.0 0.1

# 1. pick the RR per window, then copy the winners into data/2_chosen/
venv/Scripts/python.exe 1_select_rr.py --promote recommended --dry-run
venv/Scripts/python.exe 1_select_rr.py --promote recommended

# 2-4. portfolio, allocation, report
venv/Scripts/python.exe 2_analyze_maemfe.py
venv/Scripts/python.exe 3_allocate_accounts.py
venv/Scripts/python.exe 4_build_report.py
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
`--promote <tier>` copies each qualifying window's chosen file into
`data/2_chosen/` (removing any earlier pick for that window, so step 2 can't
count a window twice). `--verdicts OK WEAK` widens the filter; `--dry-run`
previews.

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

## Notes / gotchas

- MT5 exports: `mae/mfe/trade_profit` are **money**; `candle_range` is **points**.
  UTF-16, tab-separated, no header.
- Backtests run **without costs**; commission ($1/round-turn) is applied in Python.
- **Equity drawdown** walks each trade *MAE first, then MFE* (a buy-stop breakout
  usually retraces before it runs). This reproduces MT5's `STAT_EQUITY_DD`
  exactly — validated on 12 passes. Dropping the MFE term reproduces
  `STAT_BALANCE_DD`. No fudge factor.
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
