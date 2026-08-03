# Forward / Out-of-Sample Testing — Plan

Status: **design agreed, not yet implemented.** Written 2026-08-03 so the plan
survives beyond the chat it was designed in.

---

## 1. Why this is the missing piece

Everything the pipeline currently reports is **in-sample**:

- ~12,400 sweep passes evaluated (46 window/strategy combos × ~300 RR values)
- one RR chosen per window from those passes (`recommended` tier)
- an exact ILP then picks 12–15 windows and assigns them to accounts
- **fitted on 2020-01-02 → 2026-07-14, and scored on the same period**

That is a lot of selection pressure on one finite history. The headline
(`$75,615` at the 85% cap, worst account 82.2%) is what the best-fitting
configuration achieved *on the data used to find it*. It is an upper bound on
what to expect, not an estimate of it.

Nothing about the current numbers is wrong — they are correctly computed and
now well validated against MT5. They just answer a different question than
"what will this do next year".

---

## 2. The three levels of validation (industry standard)

| Level | What it tests | Cost | Verdict quality |
|---|---|---|---|
| **A. Single OOS holdout** | Does the selection survive one unseen period? | ~free (data exists) | Weak but immediate |
| **B. Walk-forward analysis (WFA)** | Does the *selection process* work repeatedly? | ~free (data exists) | The real standard |
| **C. Live/demo forward test** | Does it work with real fills, spread, latency? | Time only | Gold standard |

**A** and **B** need no new MT5 runs — see §3. **C** cannot be shortcut and
should start in parallel, because it costs nothing but calendar time.

### Why WFA rather than a single holdout

A single 2020–2023 fit / 2024–2026 test gives one number, and one number from
one split is itself noisy — it mostly tells you whether 2024–2026 happened to
suit the picks. WFA repeats the *whole selection procedure* on several
consecutive folds and stitches the untouched test periods into one continuous
out-of-sample equity curve. That tests the **method**, not a single lucky split.

---

## 3. The key enabler: no new MT5 runs are needed

`data/1_sweeps/<STRAT>/<window>/<window>_<RR>.csv` already holds **every trade,
with timestamps, for every window at every RR, across the entire 2020–2026
period.**

So a fold is just a date filter:

1. slice every sweep file to the **fit** period
2. run the existing selection logic on that slice only
   (tiers → `±SMOOTH_RR` neighbourhood → DD cap → verdict → shape)
3. take the RR it picks per window, and the ILP allocation it produces
4. score **that frozen decision** on the **next, untouched** period
5. roll forward and repeat

This is why WFA here is hours of compute, not weeks of MT5.

---

## 4. Proposed fold structure

Data spans ~6.5 years (2020-01-02 → 2026-07-14).

**Primary: anchored (expanding) walk-forward**

| Fold | Fit (selection sees) | Test (frozen, unseen) |
|---|---|---|
| 1 | 2020-01 → 2022-12 | 2023 |
| 2 | 2020-01 → 2023-12 | 2024 |
| 3 | 2020-01 → 2024-12 | 2025 |
| 4 | 2020-01 → 2025-12 | 2026 (partial, ~6.5 mo) |

Anchored (rather than rolling) because more history should genuinely help RR
selection, and we have little enough of it. **Rolling 3-year fit** should be run
afterwards as a sensitivity check — if the two disagree sharply, the edge is
regime-dependent.

Sample size sanity: each window sees ~80–150 trades/year, so a 1-year test fold
is ~100 trades per window, ~1,500 across the portfolio. Thin per window,
adequate at portfolio level. **Per-window fold results will be noisy — read the
portfolio aggregate, not individual cells.**

---

## 5. Metrics that decide the question

**Walk-Forward Efficiency (WFE)** — the headline:

```
WFE = (annualised OOS net profit) / (annualised in-sample net profit)
```

- `WFE > 0.5–0.6` → the process generalises acceptably (common industry rule)
- `WFE ≈ 1.0` → suspicious; check for leakage
- `WFE < 0.3` → the selection is mostly fitting noise

**Supporting:**
- OOS max **equity** DD per account vs its limit — does the 85% cap still hold
  out-of-sample? This is the survival question, and matters more than profit.
- Fraction of folds with positive OOS profit (consistency, not just total)
- RR **stability**: how far the chosen RR moves per window between folds. A
  window whose RR swings 0.5→3.0 across folds is being fitted, not measured.
- Verdict churn: how often a window flips OK ↔ WEAK ↔ UNLOCK_ONLY across folds
- OOS shape flags (`CONCENTRATED` / `STALE`) computed on test data only

**A deliberately included baseline:** compare against "trade every window at one
fixed RR" (e.g. 1.5). If the elaborate per-window selection cannot beat a single
global RR out-of-sample, the selection machinery is not earning its complexity.
This is the single most important control in the whole exercise.

---

## 6. What to build — `5_walkforward.py`

Reuses existing logic rather than reimplementing it (same DD maths, same tier
selection, same ILP), so the thing being tested is the *actual* pipeline.

```
for each fold:
    fit_trades   = sweeps sliced to fit period
    picks        = select_rr(fit_trades)          # step 1 logic, unchanged
    allocation   = allocate(picks, fit_trades)    # step 3 ILP, unchanged
    oos_result   = score(allocation, test period) # frozen decision, unseen data
stitch OOS periods -> continuous equity curve
report WFE, per-fold table, per-account OOS DD vs limit
```

Outputs:
- `data/3_results/walkforward_folds.csv` — per fold: picks, IS vs OOS profit/DD
- `data/3_results/walkforward_summary.csv` — WFE and aggregate verdict
- `reports/walkforward.html` — stitched OOS equity vs the in-sample curve,
  per-fold table, RR-stability view
- provenance sidecar, same as every other step

---

## 7. Honest limitations to state in the output

These do **not** invalidate the exercise, but must be reported alongside it:

1. **Meta-parameters were chosen with full-history knowledge.**
   `MAX_DD_USD`, `SMOOTH_RR = 0.10`, `MIN_RECOVERY = 2.0`, `LR_MIN = 0.30`,
   `TOP1_MAX_SHARE = 0.35`, `FADING_SHARE = 0.30`, `RECENT_YEARS = 3` were all
   set while looking at all of 2020–2026. WFA re-runs *selection* per fold but
   inherits these thresholds. Residual optimism remains; freezing them before
   the run and never touching them afterwards is the discipline that matters.

2. **The window set and strategy design are themselves fitted.** Which 23 hourly
   windows exist, and the red/green + close-based-TP mechanic, came from
   examining this history. WFA cannot un-see that.

3. **~4 folds is few.** WFE from 4 folds is itself a noisy statistic. Treat it as
   directional, not precise.

4. **2020 is unusual.** The COVID period is in every anchored fit window and
   produced at least one single trade worth 76% of a window's lifetime profit
   (GG 7-8). Consider a sensitivity run excluding 2020.

5. **Backtest fills throughout.** No spread widening, slippage, partial fills or
   news-time latency. Only level C (live/demo) tests those.

---

## 8. Suggested order of work

1. **Start a demo/paper forward test now** — costs only calendar time, and every
   week of delay is a week of evidence not accumulating. Use the current
   allocation, log fills, compare against backtest expectation later.
2. Build `5_walkforward.py`, anchored folds, portfolio level.
3. Add the fixed-RR baseline comparison (§5) — the key control.
4. Rolling-window sensitivity + the exclude-2020 variant.
5. Only then consider re-tuning anything. If WFE is poor, the answer is
   *fewer parameters*, not different ones.

---

## 9. What "pass" looks like

Decide this **before** running, so the result cannot be rationalised afterwards:

- WFE ≥ 0.5 at portfolio level
- No account's OOS equity DD exceeds its real limit in any fold
- ≥ 3 of 4 folds OOS-profitable
- The selection beats the fixed-RR baseline out-of-sample

Falling short of these is informative, not a failure of the project — it would
mean the honest conclusion is "trade fewer windows with simpler rules", which is
a real and useful result.

---

## 10. FIRST RESULTS (2026-08-04, run via 5_walkforward.py)

| Fold | test | OOS net | worst acct % of limit | baseline (all windows @1.5) |
|---|---|---|---|---|
| 1 | 2023 | $2,511 | 97.4% | $21,903 |
| 2 | 2024 | $4,178 | 87.4% | $20,714 |
| 3 | 2025 | $7,200 | **165.2% — BLOWN** | $26,498 |
| 4 | 2026H1 | $13,750 | **138.7% — BLOWN** | $5,486 |

Summary: WFE 0.787 (pass), 4/4 folds OOS-positive (pass),
**beats_baseline FALSE** ($27,639 vs $74,601), **any_acct_over_limit TRUE**.
Verdict vs pre-registered bar (§9): **2 of 4 criteria FAILED.**

Nuances:
- Baseline is not risk-comparable (46 windows, no caps/blocking). Per-window:
  baseline ≈ $1.6k/window OOS vs selection ≈ $1.7k/window — parity. So the RR
  optimisation layer adds ~nothing per window over fixed RR 1.5; the baseline's
  edge is BREADTH. Diversification >> per-window RR tuning.
- The real failure is SURVIVAL: an in-sample 85% cap produced OOS account DD of
  165% and 139% of limit in 2025/2026. In-sample DD understates next-year DD by
  up to ~2x. The cap must assume that (e.g. cap ~50%, or size DD budgets on
  OOS-measured DD, not in-sample DD).

Implications to explore next (new chat):
1. Re-run allocation with CAP_FRACTION ~0.45-0.55 and check OOS survival.
2. Test "many windows at moderate fixed RR" as the actual strategy — it may
   dominate the tuned version at equal risk once caps/blocking are applied to
   the baseline too (build a risk-matched baseline).
3. RR-stability per window across folds (already in walkforward_folds.csv picks
   column) — drop windows whose RR swings wildly.
