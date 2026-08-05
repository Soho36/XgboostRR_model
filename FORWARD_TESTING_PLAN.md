# Forward / Out-of-Sample Testing — Plan

Status: **implemented; corrected re-run required.** Written 2026-08-03 so the
plan survives beyond the chat it was designed in.

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

**A deliberately included baseline:** give every candidate window one fixed RR
(e.g. 1.5), then run the **same fit-period allocator, account cap, and
one-position replay** as the selected model. If the elaborate per-window
selection cannot beat that risk-matched control out-of-sample, the selection
machinery is not earning its complexity. A raw "trade all windows" total is a
useful breadth illustration, but never a pass/fail comparison.

---

## 6. What to build — `5_walkforward.py`

Mirrors the production constraints rather than importing scripts that execute
at import time. The fold uses the conservative MFE-first MAE/MFE drawdown bound
for risk caps: full-period MT5 stats cannot calibrate an earlier fold without
leaking its test period. Every selected pass is checked fail-closed against its
MT5 stats so a tester-truncated export cannot manufacture an OOS result.

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

## 10. Pre-correction result — superseded

The result below was produced before the walk-forward implementation corrected
its replayed stitched curve, production allocation constraints, risk-matched
baseline, input-integrity gates, source cutoff handling, and non-leaking risk
bound. **Do not use these numeric values to make a trading decision.** It is
kept only as an audit record; the next successful corrected run is authoritative.

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

## 11. CORRECTED RESULTS (2026-08-04, risk-matched control + breach analysis)

| Fold | test | OOS | worst% | risk-matched fixed-RR |
|---|---|---|---|---|
| 1 | 2023 | $3,156 | 97.4% | $4,632 |
| 2 | 2024 | $4,162 | 115.0% | $5,920 |
| 3 | 2025 | $8,420 | 153.9% | $7,567 |
| 4 | 2026H1 | $13,336 | 84.5% | $13,948 |

WFE 0.81 · 4/4 positive · selection $29,074 vs control $32,067 -> **selection
still loses even risk-matched** (per-window RR tuning adds ~nothing; simpler is
better). Survival still fails (2 folds breach). PROP-REALITY: 4 of 24
account-folds breached; post-breach profit $3,834 means as-scored OOS is
FLATTERED, not dragged down, by blown accounts — termination-adjusted OOS =
$25,240 (+4 eval/reset fees). So the modest OOS number is performance+risk,
NOT an artefact of blown accounts sitting out. reports/walkforward.html built.

## 12. CAP × RR GRID (2026-08-05, `6_cap_rr_grid.py`)

210 cells — fixed RR 0.50→2.50 step 0.10 (the requested 1.0–2.5 band, extended
down so the answer could not sit on a grid edge) × CAP_FRACTION 0.40→0.85 step
0.05 — every cell walk-forward scored on the same four folds, plus the
per-window selection arm at each cap as reference. `reports/cap_rr_grid.html`.

The decision rule (survival first, then adjusted OOS, then prefer the middle of
a plateau) was written into the script docstring **before** the run.

### 12.1 The cap is the whole survival story

Across the entire RR axis, per cap: how many of the 21 RRs kept every account
alive in every fold, and the breach rate over all account-folds.

| cap | safe RRs | breach rate | adj OOS median | adj OOS best | worst % median | accounts |
|---|---|---|---|---|---|---|
| 40% | 21/21 | 0.0% | $4,229 | $7,627 | 72 | 1–6 |
| 45% | 20/21 | 0.3% | $4,422 | $9,218 | 81 | 1–6 |
| 50% | 19/21 | 0.5% | $9,314 | $14,998 | 94 | 2–6 |
| **55%** | **14/21** | **2.0%** | **$11,968** | **$16,390** | **99** | **2–6** |
| 60% | 7/21 | 5.8% | $13,545 | $21,609 | 108 | 3–6 |
| 65% | 3/21 | 7.8% | $16,554 | $24,878 | 118 | 3–6 |
| 70% | 0/21 | 15.0% | $20,577 | $31,484 | 132 | 4–6 |
| 75% | 0/21 | 16.1% | $19,694 | $31,232 | 132 | 6 |
| 80% | 0/21 | 19.6% | $20,269 | $34,289 | 138 | 6 |
| 85% | 0/21 | 20.6% | $24,742 | $34,381 | 157 | 6 |

**At 70% and above there is no RR that survives — not one of 21.** The 85% cap
that produced every headline number in this project blows about one account-fold
in five. The transition is monotone and covers the whole RR axis, so it is a
property of the cap, not a lucky cell: **cap ≤55%, and 50% if you want room to
be wrong about RR.**

Caveat on the two tightest caps: 40–45% is safe partly by *not trading* —
allocation thins to 1–3 accounts and OOS collapses to $4k. 50–55% still deploys
5–6 accounts in the early folds, so its safety is real, not abstention.

### 12.2 RR barely matters for survival, and its profit plateau is wide

Within the surviving band the breach map is essentially flat along RR — no RR
value rescues a loose cap, and none ruins a tight one. For profit at cap 55%,
RR 1.5–2.0 is one contiguous breach-free block paying $15.1k–$16.4k adjusted;
RR ≥2.2 decays; RR ≤0.7 is weak. Summed across caps ≤55%, RR 1.2–1.5 is the
plateau ($35.7k/$36.2k/$45.1k/$38.1k) with RR 1.4 the peak.

The raw argmax of the pre-registered rule was **cap 65% / RR 2.00** ($21,884, no
breaches). It is **rejected by clause 4**: only 1 of its 4 neighbours survives,
its column breaches at the next cap step, and only 3 of 21 RRs survive at 65%.
Surviving there is luck, and its worst account still reached 94% of its limit.

### 12.3 Chosen operating point: **fixed RR 1.5, cap 55%**

| fold | test | OOS net | accounts | windows | worst acct % of limit |
|---|---|---|---|---|---|
| 1 | 2023 | $1,437 | 6 | 12 | 63.0% |
| 2 | 2024 | $4,291 | 6 | 11 | 69.7% |
| 3 | 2025 | $4,164 | 6 | 11 | 88.1% |
| 4 | 2026H1 | $5,256 | 4 | 7 | 47.7% |

$15,148 total, **0 of 22 account-folds breached**, WFE 0.855, 4/4 folds positive.
Measured against §9's pre-registered bar this is the **first configuration in the
project to pass every criterion**: WFE ≥0.5 ✓, ≥3/4 folds positive ✓, no account
over its limit ✓, and the per-window selection arm does not beat it (§12.4) ✓.

Why RR 1.5 rather than 1.7–2.0, which pay $0.3–1.2k more: its worst account
reached 88% of limit against their 98–100%, it is 4/4 folds positive, and it was
the value **pre-declared as the control** before any of this was run — so it is
the single least hindsight-loaded choice available on the plateau. Documented
alternatives: cap 50% (more conservative, ~$9.5k, 19/21 RRs safe) and cap 60%
(~$21.6k, still breach-free at RR 1.5, but only 7/21 RRs safe — much less margin
for being wrong about RR).

### 12.4 Per-window RR selection loses again, at almost every cap

Same rig, same folds, same caps. "Fixed @1.5" is the pre-declared RR, not a
hindsight pick:

| cap | selection adj $ | breaches | fixed @1.5 adj $ | breaches |
|---|---|---|---|---|
| 50% | 9,402 | 1 | 9,484 | 0 |
| 55% | 14,382 | 1 | **15,148** | **0** |
| 60% | 12,015 | 4 | 21,609 | 0 |
| 65% | 15,169 | 3 | 21,725 | 1 |
| 70% | 21,558 | 1 | 28,784 | 2 |
| 75% | **33,370** | 1 | 27,751 | 3 |
| 80% | 24,615 | 4 | 31,178 | 2 |
| 85% | 25,240 | 4 | 32,457 | 2 |

Fixed RR wins 9 of 10 caps and breaches less at every cap ≤60%. The one
exception (75%) sits inside a non-monotone cap response — $21.6k → $33.4k →
$24.6k → $25.2k — which is a noise signature, not an edge. **The per-window RR
layer is finished: it costs complexity, loses money, and loses accounts.**

### 12.5 The horizon problem this exposed — read before deploying

`--deploy 1.5 0.55` fits the chosen cell on **all** 6.5 years and fills only
**3 accounts** (RR 5-6 + RR 7-8; RR 3-4 + RR 23-24; GG 4-5), fit net $20,626.
Fold 1 fitted on 3 years and filled 6. The cause: the cap is applied to the
**max drawdown over the whole fit period**, and that maximum only grows as
history accumulates. So the same 55% cap gets strictly stricter every year, and
the deployed configuration is tested against a 6.5-year DD while every validated
fold sized against a 3–6 year one and then ran for a single year.

This is conservative, not wrong — but it is not what the walk-forward validated.
Three ways out, in preference order:

1. **Rolling fixed-length fit** (e.g. last 3 years) so the DD statistic always
   has the horizon the folds used. `select()` and `build_groups()` already take
   `fit_a`, so this is a small change and the §4 plan already lists it as the
   sensitivity check to run.
2. Size the cap on a DD **percentile** rather than the single worst excursion.
3. Accept 3 accounts and less deployed capital.

Until that is settled, the 3-account config is the safe thing to forward-test —
it under-deploys, it does not under-protect.

### 12.6 Standing limitation

Choosing a cell by its out-of-sample score is second-order fitting: 210 cells
against 4 folds. The mitigations are that the cap conclusion is monotone across
the entire grid (not a cherry-picked cell), the RR plateau is wide and flat, and
RR 1.5 was pre-declared. It remains true that the demo forward test is the only
evidence not conditioned on this history.
