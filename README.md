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
                                                                  profit under DD limits)
```

### Step 1 — `1_analyze_optimization.py`
For each window, reads an MT5 RiskReward optimization sweep and picks a
drawdown-aware RR (best Recovery Factor whose $ drawdown stays under the cap),
plus a plateau/robustness check and a recent-vs-full regime-stability flag.

- **in:**  `data_1_optimization/<RR|GG>/<recent|full>/<from>-<till>.xml`
- **out:** `results/<RR|GG>_recommendations.csv`, `plots/step1_optimization/<RR|GG>/`

### Step 2 — `2_analyze_maemfe.py`
Reads per-trade exports for the **chosen** window+RR set and builds real
equity curves, drawdowns, and the combined portfolio (+ a cross-strategy view).

- **in:**  `data_2_maemfe/<RR|GG>/<from>-<till>_<RR>.csv`  (UTF-16, tab, no header)
- **out:** `results/<RR|GG>_maemfe_window_summary.csv`,
           `results/<RR|GG>_maemfe_combined_trades.csv`,
           `plots/step2_portfolio/<RR|GG>/`

### Step 3 — `3_allocate_accounts.py`
Exact integer program: pick which windows to trade and which account each goes
to, **maximising net profit** subject to every account staying under its DD
limit. Accounts are strategy-pure (netting-safe); one-position replay prices in
blocked entries.

- **in:**  `results/<RR|GG>_maemfe_combined_trades.csv`
- **out:** `results/multi_strategy_allocation.csv`

## Folder map

| Folder | Contents | In git? |
|--------|----------|---------|
| `data_1_optimization/` | MT5 optimization XMLs (RR/GG × recent/full) | no |
| `data_2_maemfe/` | per-window MT5 trade exports (RR/, GG/) | **yes** |
| `results/` | all CSV outputs | no |
| `plots/` | all generated plots (step1/step2/step3) | no |
| `archive/` | superseded scripts (early regime study, single-strategy allocator) | yes |
| `legacy_regime_data/` | raw OHLCV / trade_stats from the abandoned regime study | no |
| `strategy_ea.cs` | the MT5 Expert Advisor | yes |

## Key config knobs (top of each script)

- **Step 1:** `MAX_DD_USD`, `COMMISSION_PER_RT`, `SMOOTH_RR`, `REGIME_RR_TOL`
- **Step 2:** `COMMISSION_PER_RT`, `STRATEGIES` (add a strategy = add a folder)
- **Step 3:** `ACCOUNT_NAMES` / `ACCOUNT_LIMITS` / `ACCOUNT_DD_AVAILABLE`,
  `CAP_FRACTIONS`, `ALLOW_MIXED_STRATEGIES`, `REPLAY_ONE_POSITION`

## Run

```bash
venv/Scripts/python.exe 1_analyze_optimization.py
venv/Scripts/python.exe 2_analyze_maemfe.py
venv/Scripts/python.exe 3_allocate_accounts.py
```

## Notes / gotchas

- MT5 exports: `mae/mfe/trade_profit` are **money**; `candle_range` is **points**.
- Optimizations were run **without costs** — commission is applied in Python.
- Drawdowns are historical (2020–2026); future DD can exceed them, hence the
  sub-100% DD cap (default 85%).
- Prop DD limit is trailing and **freezes** once the account banks its buffer,
  so the static-limit view is conservative once an account is seasoned.
