# MNQ Time-Window RR Study

Tooling to size and allocate a long-only MNQ 30-min breakout strategy across
prop accounts. Two variants are analysed:

- **RR** — enter after the last **red** candle (buy-stop above its high)
- **GG** — enter after the last **green** candle

> The folder is still named `Xgboost_RR_model` for legacy reasons (the project
> began as an XGBoost experiment, since abandoned). The name is cosmetic — don't
> rename the folder; tooling/config is keyed to it.

## Pipeline (run in order)

```
 0_run_mt5_sweeps.py   drives MT5: one optimization per window, sweeping RR
        │              -> per-trade CSVs + per-pass _stats.csv
        ▼
 1_select_rr.py        DD-aware RR pick per window, from REAL per-trade data
        │              -> recommended / aggressive / unlocked tiers
        ▼
 2_analyze_maemfe.py   equity curves, drawdowns, combined portfolio
        │
        ▼
 3_allocate_accounts.py  exact ILP: which windows on which prop account
        │
        ▼
 4_build_report.py     one interactive HTML report of everything
```

Everything downstream of step 0 reads **per-trade data only**. The old
optimization-XML path is retired (see *Legacy* below) — it was a summary-only
source that drifted from the per-trade exports in period and tester settings,
which caused repeated drawdown discrepancies.

### Step 0 — `0_run_mt5_sweeps.py`
Launches MT5 headlessly per window (config `.ini` + `ShutdownTerminal=1`) and
collects the exports. The RR sweep is done *by* MT5's optimizer (one pass per RR
value); this script loops the **windows**.

Each strategy has its own MT5 install, so `STRATEGIES` holds a per-strategy
`terminal` / `expert` / `symbol`. A `preflight()` verifies the terminal and the
compiled `.ex5` exist before launching — without it a wrong Expert path makes
MT5 exit in ~8s with no output at all.

- **out:** `INPUTS/data_2_maemfe_input/<STRAT>_sweeps/<window>/<window>_<RR>.csv`
           `INPUTS/data_2_maemfe_input/<STRAT>_sweeps_stats/<window>/..._stats.csv`
           plus a `_manifest.json` per window recording symbol/dates/model

```bash
venv/Scripts/python.exe 0_run_mt5_sweeps.py --windows 2-3 3-4 --strategy RR --rr 1.0 2.0 0.1
venv/Scripts/python.exe 0_run_mt5_sweeps.py --list        # window -> EA input mapping
```

### Step 1 — `1_select_rr.py`
Per-window RR tiers (`recommended` / `aggressive` / `unlocked`) computed from
the real per-trade sweeps, with the DD cap applied to the **equity** drawdown.

- **in:**  `INPUTS/data_2_maemfe_input/<STRAT>_sweeps/<window>/`
- **out:** `OUTPUTS/results_outputs/rr_pertrade_recommendations.csv`,
           `OUTPUTS/plots_outputs/step1_rr_selection/`

### Step 2 — `2_analyze_maemfe.py`
Takes the **chosen** window+RR files and builds real equity curves, drawdowns,
per-year breakdowns and the combined portfolio (plus a cross-strategy view).

- **in:**  `INPUTS/data_2_maemfe_input/<RR|GG>/<window>_<RR>.csv`
           (one file per window — copy the winners out of the sweep folders)
- **out:** `OUTPUTS/results_outputs/<RR|GG>_maemfe_window_summary.csv`,
           `OUTPUTS/results_outputs/<RR|GG>_maemfe_combined_trades.csv`,
           `OUTPUTS/plots_outputs/step2_portfolio/`

### Step 3 — `3_allocate_accounts.py`
Exact integer program (`scipy.optimize.milp`): pick which windows to trade and
which account each goes to, maximising net profit subject to every account's DD
limit. Accounts are strategy-pure (netting-safe); a one-position replay prices
in blocked entries.

- **in:**  `OUTPUTS/results_outputs/<RR|GG>_maemfe_combined_trades.csv`
- **out:** `OUTPUTS/results_outputs/multi_strategy_allocation.csv`

### Step 4 — `4_build_report.py`
One self-contained interactive HTML (Plotly inlined, offline). Per-trade data is
shipped into the page, so **every chart on the Portfolio tab recomputes from the
windows you tick** — including the drawdown subplot.

- **out:** `OUTPUTS/report.html` (~5 MB, open in any browser)

## Folder map

| Folder | Contents | In git? |
|--------|----------|---------|
| `INPUTS/data_2_maemfe_input/<RR\|GG>/` | the **chosen** one-file-per-window set (step 2) | no |
| `INPUTS/data_2_maemfe_input/<STRAT>_sweeps/` | full RR sweeps per window (step 1) | no |
| `INPUTS/data_2_maemfe_input/<STRAT>_sweeps_stats/` | MT5 `OnTester` summaries, for cross-checking | no |
| `INPUTS/legacy_optimization_xml/` | **legacy** — the retired optimization XMLs | no |
| `OUTPUTS/results_outputs/` | all CSV outputs | no |
| `OUTPUTS/plots_outputs/` | static PNGs per step (`legacy_*` = from the XML path) | no |
| `OUTPUTS/mt5_ini/` | generated tester configs (regenerated each run) | no |
| `archive/` | superseded scripts, incl. `analyze_optimization_xml.py` | no |
| `*_r_MFE_buy-stop-entry(example).cs` | the two MQL5 EAs (despite the `.cs` extension) | no |

## EA requirements (both RR and GG)

The pipeline depends on three EA behaviours:

1. `RunTag=""` → filename derived from the enabled window via
   `ActiveWindowLabel()`, so an export can't be mislabelled.
2. Exports named `<window>_<RR:2dp>.csv` and opened with **`FILE_COMMON`**, so
   every tester agent writes to one shared folder.
3. `OnTester()` writes `<window>_<RR>_stats.csv` with MT5's own figures
   (`STAT_EQUITY_DD` etc.) — used to cross-check Python's reconstruction.
   Note `STAT_LR_CORRELATION` does **not** exist in MQL5; compute curve
   straightness in Python instead.

## Notes / gotchas

- MT5 exports: `mae/mfe/trade_profit` are **money**; `candle_range` is **points**.
  UTF-16, tab-separated, no header.
- Backtests run **without costs**; commission ($1/round-turn) is applied in Python.
- **Equity drawdown** is reconstructed by walking each trade as *MAE first, then
  MFE* (a buy-stop breakout usually retraces before it runs). This reproduces
  MT5's `STAT_EQUITY_DD` exactly — validated on 12 passes. Dropping the MFE term
  reproduces `STAT_BALANCE_DD`. There is no fudge factor.
- **Keep every export pinned to one common end date.** Every DD discrepancy this
  project has hit traced back to data generated at different times. Step 0 writes
  a `_manifest.json` per window and warns if a re-run changes symbol/dates/model.
- Collection **overwrites** same-named files silently — intended (fresh data
  wins), but delete a sweep folder first if you want a guaranteed-clean re-run.
- Drawdowns are historical (2020–2026); future DD can exceed them, hence the
  sub-100% DD cap (default 85%) in step 3.
- Prop DD limit is trailing and **freezes** once the account banks its buffer,
  so the static-limit view is conservative once an account is seasoned.

## Legacy

`1_analyze_optimization.py` (now `archive/analyze_optimization_xml.py`) parsed
MT5 optimization XMLs. It still works, but the XMLs are a summary-only source
that must be regenerated in lockstep with the per-trade exports to stay
comparable — which is exactly the failure mode that motivated step 1. Kept for
reference; `INPUTS/legacy_optimization_xml/` holds its inputs.
