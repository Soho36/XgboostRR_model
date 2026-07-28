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
 MT5 optimization XMLs            MT5 per-trade exports           prop accounts
 (RR sweep per window)            (chosen window @ RR)            (unequal DD limits)
        │                                 │                             │
        ▼                                 ▼                             ▼
 1_analyze_optimization.py  ──►  2_analyze_maemfe.py  ──────►  3_allocate_accounts.py
   pick DD-aware RR per window     real equity curves,           choose which windows to
   (recommended/aggressive/        combined portfolio,           trade + which account each
    unlocked tiers)                per-window & per-year          goes to (exact ILP, max
        │                                                         profit under DD limits)
        ▼                                                               │
 1b_rr_from_maemfe.py  (ground truth)                                   ▼
   re-check an RR pick from real                              4_build_report.py
   per-trade exports at a small                                 one interactive HTML
   RR grid — no XML staleness                                   report of everything
```

### Step 1 — `1_analyze_optimization.py`
For each window, reads an MT5 RiskReward optimization sweep and picks a
drawdown-aware RR (best Recovery Factor whose $ drawdown stays under the cap),
plus a plateau/robustness check and a recent-vs-full regime-stability flag.

- **in:**  `INPUTS/data_1_optimization_input/<RR|GG>/<recent|full>/<from>-<till>.xml`
- **out:** `OUTPUTS/results_outputs/<RR|GG>_recommendations.csv`,
           `OUTPUTS/plots_outputs/step1_optimization/<RR|GG>/`

### Step 1b — `1b_rr_from_maemfe.py` (per-trade RR check)
Same tier logic as step 1 but computed from **real per-trade exports** at a
small RR grid — the ground-truth check when the XMLs may be stale (period or
tester-settings mismatch). Use it for any window whose DD recently changed.

- **in:**  `INPUTS/data_2_maemfe_input/<STRAT>_sweeps/<window>/<window>_<RR>.csv`
- **out:** `OUTPUTS/results_outputs/rr_pertrade_recommendations.csv`,
           `OUTPUTS/plots_outputs/step1b_rr_pertrade/`

### Step 2 — `2_analyze_maemfe.py`
Reads per-trade exports for the **chosen** window+RR set and builds real
equity curves, drawdowns, and the combined portfolio (+ a cross-strategy view).

- **in:**  `INPUTS/data_2_maemfe_input/<RR|GG>/<from>-<till>_<RR>.csv`  (UTF-16, tab, no header)
- **out:** `OUTPUTS/results_outputs/<RR|GG>_maemfe_window_summary.csv`,
           `OUTPUTS/results_outputs/<RR|GG>_maemfe_combined_trades.csv`,
           `OUTPUTS/plots_outputs/step2_portfolio/<RR|GG>/`

### Step 3 — `3_allocate_accounts.py`
Exact integer program: pick which windows to trade and which account each goes
to, **maximising net profit** subject to every account staying under its DD
limit. Accounts are strategy-pure (netting-safe); one-position replay prices in
blocked entries.

- **in:**  `OUTPUTS/results_outputs/<RR|GG>_maemfe_combined_trades.csv`
- **out:** `OUTPUTS/results_outputs/multi_strategy_allocation.csv`

### Step 4 — `4_build_report.py` (interactive report)
Aggregates every result into **one self-contained interactive HTML** — tabs per
step, sortable tables, hover/toggle equity curves. No dependencies, offline.

- **in:**  everything in `OUTPUTS/results_outputs/`
- **out:** `OUTPUTS/report.html`  (open in any browser)

## Folder map

| Folder | Contents | In git? |
|--------|----------|---------|
| `INPUTS/data_1_optimization_input/` | MT5 optimization XMLs (RR/GG × recent/full) | no |
| `INPUTS/data_2_maemfe_input/` | per-window MT5 trade exports (RR/, GG/, *_sweeps/) | no |
| `OUTPUTS/results_outputs/` | all CSV outputs | no |
| `OUTPUTS/plots_outputs/` | static PNG plots per step | no |
| `OUTPUTS/report.html` | the interactive report (step 4) | no |
| `archive/` | superseded scripts (early regime study, single-strategy allocator) | no |
| `legacy_regime_data/` | raw OHLCV / trade_stats from the abandoned regime study | no |
| `strategy_ea.cs` | the MT5 Expert Advisor | yes |

## Key config knobs (top of each script)

- **Step 1:** `MAX_DD_USD`, `COMMISSION_PER_RT`, `SMOOTH_RR`, `REGIME_RR_TOL`
- **Step 1b:** `MAX_DD_USD`, `DD_MODE`, `DD_HAIRCUT` (≈1.15 lifts MAE-based
  floating DD to true equity DD)
- **Step 2:** `COMMISSION_PER_RT`, `STRATEGIES` (add a strategy = add a folder)
- **Step 3:** `ACCOUNT_NAMES` / `ACCOUNT_LIMITS` / `ACCOUNT_DD_AVAILABLE`,
  `CAP_FRACTIONS`, `ALLOW_MIXED_STRATEGIES`, `REPLAY_ONE_POSITION`

## Run

```bash
venv/Scripts/python.exe 1_analyze_optimization.py
venv/Scripts/python.exe 2_analyze_maemfe.py
venv/Scripts/python.exe 3_allocate_accounts.py
venv/Scripts/python.exe 4_build_report.py
```

## Notes / gotchas

- MT5 exports: `mae/mfe/trade_profit` are **money**; `candle_range` is **points**.
- Optimizations were run **without costs** — commission is applied in Python.
- **Keep all exports pinned to one common end date.** Every period mismatch we
  hit (XML vs per-trade DD discrepancies) came from data generated at different
  times. XML DD ≈ true equity DD when periods match; big gaps mean stale data,
  not intrabar effects.
- Our MAE-based floating DD runs ~10–15% below MT5's true equity DD (MAE-timing
  approximation) — hence `DD_HAIRCUT` in step 1b.
- Drawdowns are historical (2020–2026); future DD can exceed them, hence the
  sub-100% DD cap (default 85%).
- Prop DD limit is trailing and **freezes** once the account banks its buffer,
  so the static-limit view is conservative once an account is seasoned.
