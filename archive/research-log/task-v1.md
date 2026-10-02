# ASPECT-D v1.1 — Extension Runbook (task-v1.md)

You are Claude Code, resuming work in the **already-executed** ASPECT-D repo.
v1.0 is DONE: 45/45 runs, outcome class **S1 declared** (Δτ = +0.1102,
CI [0.0921, 0.1312]), paper v1 compiled and submittable, 23 days of calendar and
~86% of the hardware budget unspent. Your mission now is different in kind:
**the submission is de-risked, so run a continuous background program that
(a) hardens the v1.0 claim against the predictable reviewer attacks and
(b) adds at most a small number of new, pre-registered, quotable results —
without ever endangering what already exists.**

You run unattended for days. Work through the priority queue below, keep
producing results, and keep the feed (§3) updated so a human can drop in at any
time and see where things stand.

---

## 0. PRIME DIRECTIVES (additions to task.md §0 — all v1.0 directives still bind)

1. **v1.0 is immutable.** The declared S1 outcome, `PREREGISTRATION.md`,
   `artifacts/fits.json`, `artifacts/runs.csv`, `LOG.md` history, and the 45
   v1.0 runs are frozen. Never retrain, re-synthesize, re-score, or overwrite
   any v1.0 artifact. All v1.1 outputs go to `artifacts-v1.1/`,
   `fits-v1.1.json` inside it, and `LOG-v1.1.md`. New results can ADD claims;
   they can never revise the v1.0 declaration.
2. **Git hygiene before anything.** Tag the current tree
   `v1.0-submission-candidate`; create and work on branch `v1.1-extensions`.
   Copy `paper/main.tex` → `paper/main-v1-frozen.tex` (never edited again).
3. **Pre-registration before data, still.** Phase E0 writes
   `PREREGISTRATION-v1.1.md` by copying §9 of this file VERBATIM, and logs it,
   before any new synthesis or training. §9 is frozen now, while no extension
   data exists. Anything not in §9 is exploratory and must be labeled so.
4. **Canonicality for v1.1:** for all NEW quantities (schedules, item subsets,
   thresholds, caps, dates) THIS FILE is canonical; for all v1.0 machinery
   (architecture, recipe, sampler, metrics, statistics code paths) the v1.0
   precedence chain `configs/grid.json` → `protocol.html` → `task.md` is
   unchanged and wins on conflict. A number found nowhere may not be invented
   (task.md §0.1 procedure applies, logged in LOG-v1.1.md).
5. **Scope is fixed.** No AR training, no causal-mask variants, no width-only sweeps, no CFG axis.
   These are explicitly out of scope. If an experiment seems to require one of
   them, it's the wrong experiment.
6. **The paper's 4-page cap is a law.** Paper v2 main text may gain at most:
   the iso-latency figure (E2), the extended-T update to the step-curves
   figure (E1), and ≤2 sentences of robustness. Everything else is appendix.
   If it doesn't fit, appendix — never squeeze the Results prose.
7. **Two stop-losses again.** v1.1 compute envelope: **600 B200-equivalent
   GPU-hours charged** (same charging convention as v1.0; occupancy also
   logged). Calendar: milestones in §8; **paper v2 freeze 2026-08-24**;
   submission wall 2026-08-29 AoE unchanged. On projected overrun apply the
   cut list in §8, one cut at a time, logged.
8. **Resume protocol.** `state-v1.json` at repo root mirrors state.json's role:
   queue position, completed jobs, GPU-h, gate history, feed pointer. On every
   startup read it first and resume. Assume you can be killed at any time.

---

## 1. GATE E-AUDIT — checkpoint audit (run FIRST, before the pre-registration)

Everything in Tier 1 is inference on the 45 v1.0 checkpoints.

1. Verify a loadable final checkpoint exists for every (config, seed) of the 15
   active configs × seeds {0,1,2}; record path + step + hash in
   `artifacts-v1.1/checkpoint_audit.json`.
2. Smoke test: load C3 seed 0, synthesize 5 eval items at T=16, scores must be
   within 3 SE of the v1.0 per-item values (`runs/C3_0/synth_T16/scores.json`).
3. **If any checkpoint is missing or fails to load:** log it, post to the feed,
   and re-scope — jobs touching missing runs are dropped from E1/E3/E5;
   training jobs (E4, E6) are unaffected. If >20% of checkpoints are gone,
   invert priority: run E4 (budget D) and E6 (undertraining) first and post a
   feed item asking the human whether to re-train the missing v1.0 runs
   (do NOT retrain them on your own authority — directive 1).

## 2. WORK QUEUE (priority order; fill idle GPUs opportunistically via src/gpufill.py)

| # | Job | Type | Est. GPU-h | Gate |
|---|-----|------|-----------|------|
| E0 | Pre-registration v1.1 + git tag/branch | none | 0 | after E-AUDIT |
| E1 | Extended-T synthesis {24,32,64} | inference | ≈55 | E-AUDIT |
| E2 | Iso-latency Pareto analysis | CPU only | 0 | none (uses v1.0 surface) |
| E3 | NFE allocation across levels | inference | ≈10 | E-AUDIT |
| E4 | Budget-D grid (5 cfg × 2 seeds) | training | ≈250 | G1-D sanity |
| E5 | Robustness panel (2nd ASR, 2nd SV, refits) | scoring/CPU | ≈20 | E-AUDIT |
| E6 | Undertraining control (C1/C3/C5 @ 90k) | training | ≈35 | none |
| E7 | Confidence-adaptive T (method) | inference | ≈30 | GATE-E7 (§8) |
| E8 | Paper v2 + results-v1.1.html + DECISION-v1.1.md | CPU | 0 | after E1–E6 verdicts |
| E9 | Post-freeze background queue | mixed | remainder | after paper v2 freeze |

Run E0→E3 + E5 immediately and in parallel with launching E4/E6 training
(training saturates GPUs; inference jobs backfill). E2 costs nothing — do it
first and post the finding to the feed on day one.

## 3. THE FEED — `RESULTS-FEED.md` (append-only, human-facing)

After every completed job and every gate: append a timestamped entry with
(what ran, headline numbers with CIs, one-sentence interpretation, whether it
is pre-registered [cite §9 id] or exploratory, and what it changes for the
paper). This file is how the human monitors you — keep it current and honest.
Never delete or edit past entries; corrections are new entries.

## 4. JOB SPECS

### E1 — Extended T (attacks "T*=16 is a lower bound")
- Runs: {A3, B3, C1, C2, C3, C4, C5} × seeds {0,1,2} = 21 runs.
- T_ext = {24, 32, 64} steps/level (NFE 192, 256, 512), frozen sampler,
  matched per-item RNG (same streams as v1.0).
- Items: deterministic first 200 of eval_zs (items it0000–it0199). For curve
  continuity, recompute T ∈ {1,2,4,8,16} summary stats on the SAME 200-item
  subset from the existing v1.0 per-item scores (no re-synthesis).
- Score with the v1.0 metric stack; write per-run
  `runs/<cfg>_<seed>/synth_T<k>/scores.json` (new T dirs are additive, allowed)
  and `artifacts-v1.1/runs_ext.csv`.
- Analysis (pre-registered H-E1, §9): report T*_WER and T*_SIM on the extended
  range; refit τ on the extended surface for the 21-run subset; compare to
  v1.0 τ.

### E2 — Iso-latency Pareto (resurrects the practical question H-D3 served)
- Zero GPU. Use the v1.0 75-point surface + measured c_layer (flat in width).
- Serial latency L = 8·T·d·c_layer. Sweep budgets L ∈ a log grid covering
  (T,d) ∈ grid; at each L, find the best (d, T) per metric by interpolating the
  fitted M_sep surface AND by nearest measured points (report both; they must
  agree qualitatively or the disagreement is the finding).
- Deliverable: `artifacts-v1.1/figures/iso_latency_pareto.{svg,pdf}` — per
  metric, optimal allocation path in the (log T, log d) plane with the
  interior-d* ridge marked. Pre-registered claim H-E2 (§9).

### E3 — NFE allocation across codebook levels
- Runs: C-budget 15 runs (5 cfg × 3 seeds). Full 400 items.
- Three frozen schedules at matched total NFE = 32:
  uniform = [4,4,4,4,4,4,4,4]; coarse = [25,1,1,1,1,1,1,1];
  fine = [1,1,1,1,1,1,1,25]. Same confidence sampler within each level.
- Pre-registered H-E3 (§9): paired item-level bootstrap over the 400 items.

### E4 — Budget D (285M) grid
- Configs D1–D5 exactly as in `configs/grid.json` (dims already frozen there);
  seeds {0,1} mandatory, seed 2 stretch under the v1.0 stretch rule applied to
  the v1.1 envelope. 30k steps, recipe and LR rule unchanged (μP, base LR
  0.004).
- **G1-D sanity gate first:** 3,000-step proxy on D3 at the transferred LR:
  pass iff no divergence and val loss at 3k is below B3's and C3's 3k values
  (monotone-in-N check). Fail → one 5-point LR sweep at D3 only; adopt its
  argmin for all D runs; log.
- Eval: full T grid {1,2,4,8,16} × 400 items + extended {24,32,64} × 200 items.
- Analysis (pre-registered H-E4, §9): 4-budget Part-B refit → Δτ(A–D);
  extrapolation A+B+C → D (replaces failed v1.0 H-D4 test, labeled as a NEW
  test, not a redo); exploratory: d*(N) trend across 4 budgets.
- G3-equivalent health rule and divergence/restart policy identical to v1.0.

### E5 — Robustness panel (sensitivity, NOT hypotheses)
- Re-score all v1.0 T=16 syntheses (45 runs × 400 items, audio already on
  disk) with: (a) whisper-medium.en, (b) the SIM fallback
  microsoft/wavlm-base-plus-sv. Recompute Δτ per §7.2 machinery on each
  variant metric.
- Refit Part A and Part B with log-amplitude parameterization (log A, log B,
  log C unbounded; exponent bounds unchanged) — the fix for the A ≤ 10 defect.
  Report whether α becomes identified and what ρ then is. **Labeled
  exploratory-sensitivity; cannot rescue H-D1's pre-registered status.**
- Feed + appendix table: Δτ under {ASR × SV × parameterization} variants.

### E6 — Undertraining control (attacks "steps just compensate undertraining")
- Fresh runs C1, C3, C5, seed 0, 90,000 steps, cosine to 10% over the full 90k,
  warmup 600, all else identical. NOT resumed from v1.0 (schedule differs).
- Eval: full T grid + extended subset, same items as E1.
- Pre-registered H-E5 (§9): τ-fit on the 3-config surface at 90k vs the same
  3 configs at 30k, item-level bootstrap (1 seed — the weaker instrument is
  pre-registered as such).

### E7 — Confidence-adaptive T (the method; gated, see §8)
- Sampler addition `--adaptive`: within each level, after step s ≥ 2, stop
  early iff the fraction of cells that changed token in step s is < 2% for 2
  consecutive steps; T_max = 16. Log per-item NFE.
- Runs: C-budget 15 runs, 400 items. Baselines: fixed T = {4, 8, 16}.
- Pre-registered H-E6 (§9). If supported → one appendix section + one main-text
  sentence; if not → feed entry + appendix note, no paper claim.

### E8 — Paper v2
- Update step-curves figure with extended T (E1); add iso-latency figure (E2);
  add ≤2 robustness sentences (E5); update limitations: remove "ran out of
  dial" caveat if E1 resolves it, add the speaking-rate/char-rate limitation
  sentence (from the human QC pass), state D-budget scope if E4 landed.
- New numbers flow only through `src/paper.py` → `paper/numbers.tex` (extend
  the generator; never hand-type).
- Rebuild appendix: allocation table (E3), robustness table (E5), 4-budget
  Δτ (E4), adaptive-T (E7) as applicable. Verify main text still closes on
  page 4. Recompile; update DECISION-v1.1.md with a submission-readiness
  re-statement.

### E9 — Post-freeze background queue (after 2026-08-24, does not touch the paper)
Priority order, run until told to stop or envelope exhausted; everything here
is exploratory:
1. Seed 2 for budget D (if not already run).
2. Extended-T completion: remaining 24 v1.0 runs × {24,32,64} × 200 items.
3. d*(N) characterization: fit interior-optimum depth vs N across 4 budgets;
   post the trend to the feed.
4. 90k-step undertraining for A3/B3 (completes the N × training-compute plane).

## 5. STATISTICS (machinery unchanged)

All fits/bootstraps reuse `src/fit.py` conventions: weighted NLS, 32
multi-starts, AICc, 2,000 bootstrap replicates, RNG 7331, run-level resampling
wherever ≥2 seeds exist, item-level (paired where applicable) otherwise — the
instrument used is always stated next to the number. Exponent bounds unchanged;
amplitude parameterization per job spec (E5 defines the log-amplitude variant).

## 6. DELIVERABLES (definition of done for v1.1)

- [ ] `PREREGISTRATION-v1.1.md` (verbatim §9), logged before any extension data
- [ ] `artifacts-v1.1/checkpoint_audit.json`
- [ ] `artifacts-v1.1/runs_ext.csv`, `runs_D.csv`, `runs_90k.csv`,
      `runs_alloc.csv`, `runs_adaptive.csv` (as applicable), `fits-v1.1.json`
- [ ] `artifacts-v1.1/figures/`: iso_latency_pareto, step_curves_extended,
      delta_tau_4budgets, allocation_bars (+ adaptive_frontier if E7 ran)
- [ ] `RESULTS-FEED.md` — continuous, timestamped
- [ ] `LOG-v1.1.md` — every gate with measured values, every cut
- [ ] `results-v1.1.html` (house style; v1.0 results.html untouched)
- [ ] Paper v2: `paper/main.tex` rebuilt, `paper/main-v1-frozen.tex` preserved,
      main text ≤ 4 pages, compiles
- [ ] `DECISION-v1.1.md` — which pre-registered extension hypotheses resolved
      which way; final claim inventory for the submission
- [ ] `state-v1.json` phase = DONE (or BACKGROUND if E9 still running)

## 7. WHAT NOT TO DO (v1.1 additions)

- Do not modify, re-run, or re-score anything under v1.0's declared analysis.
- Do not let any v1.1 result reword the v1.0 pre-registered claims section of
  the paper; extensions get their own clearly-scoped sentences.
- Do not run AR models, CFG, or width-only sweeps (directive 5).
- Do not present E5's log-amplitude ρ as a supported H-D1 — it is sensitivity.
- Do not start E7 before its gate; do not let E9 items delay E8.
- Do not exceed 600 GPU-h charged or breach the Aug 24 paper freeze.

## 8. CALENDAR, GATES, CUTS

Milestones (post a feed entry at each):
- **V1** 2026-08-08: E-AUDIT + E0 done; E2 posted; E1/E3/E5 launched; E4/E6 training.
- **V2** 2026-08-11: E1, E3, E5 verdicts posted.
- **V3** 2026-08-15: E4 (2 seeds) + E6 done, 4-budget Δτ posted.
- **GATE-E7** 2026-08-16: E7 launches ONLY if V1–V3 all met and envelope ≤ 60%
  spent; otherwise E7 is cut (first cut).
- **V4** 2026-08-20: E7 verdict (if it ran).
- **V5** 2026-08-24: **paper v2 frozen.** After this, E9 only.
- Wall: 2026-08-29 AoE (submission; human uploads).

Cut list on projected compute/calendar overrun (in order): drop E7 → drop
extended-T T=64 (keep 24, 32) → D seeds 2→1 +flag (single-seed D is
exploratory-only, H-E4 then reports without CI claim) → drop E6 A3/B3 ambitions
(C-only stands) → shrink E1 items 200→100. E8 is never cut.

## 9. PRE-REGISTRATION v1.1 ADDENDUM (copy VERBATIM to PREREGISTRATION-v1.1.md at E0)

> **ASPECT-D pre-registration addendum v1.1** — frozen before any extension
> run; amends nothing in v1.0; the v1.0 declared outcome (S1) is immutable.
>
> - **H-E1 (extended test-time scaling).** On T ∈ {1,…,64} (21-run subset, 200
>   items): WER's saturation step T*_WER (95% of the T=64 value) is > 16, and
>   the refitted τ_WER on the extended range lies within the v1.0 τ_WER 95% CI
>   [0.8248, 0.8513]. Either half failing is reported as measured; T* values
>   are reported regardless.
> - **H-E2 (iso-latency allocation).** At every latency budget L in the tested
>   grid, the WER-optimal allocation satisfies d ≥ the budget's interior
>   optimum d* before steps are increased beyond the minimum tested — i.e.,
>   owned depth dominates rented steps for WER up to d*, by both the fitted
>   surface and nearest-measured-point methods. Decision: agreement of both
>   methods at ≥ 80% of tested L values.
> - **H-E3 (coarse-level allocation).** At matched NFE = 32 on the C budget:
>   WER(coarse) < WER(fine), paired item-level bootstrap 95% CI on the
>   difference excluding 0; secondary: WER(coarse) ≤ WER(uniform).
> - **H-E4 (scale persistence).** On the 4-budget (A–D) Part-B surface with
>   seeds {0,1} per config minimum: Δτ > 0 with run-level bootstrap 95% CI
>   excluding 0. Secondary (new test, not a redo of v1.0 H-D4): M_sep fit on
>   A+B+C predicts budget D config means at T=16 with MAPE ≤ 15% and ≤ the
>   N-only model's MAPE. Exploratory (no decision rule): d*(N) trend.
> - **H-E5 (training-compute control).** For {C1, C3, C5} trained to 90k
>   steps (seed 0): the τ-surface refit gives Δτ_90k with item-level bootstrap
>   95% CI excluding 0. Reported alongside: whether the Δτ_90k CI overlaps the
>   matched 30k 3-config Δτ CI. Single-seed scope is acknowledged; this is a
>   control, not a headline.
> - **H-E6 (adaptive steps).** Confidence-adaptive stopping (rule frozen in
>   task-v1.md §4-E7) on the C budget achieves corpus WER ≤ fixed-T16 WER +
>   1.0 absolute point at mean NFE ≤ 0.6 × 128, per-run, in ≥ 12 of 15 runs.
> - Analysis machinery, RNGs, and bounds per task-v1.md §5. Everything not
>   listed above is exploratory and will be labeled so wherever it appears.

## 10. STARTUP CHECKLIST (every boot)

1. Read `state-v1.json` (create from template on first boot), then
   `RESULTS-FEED.md` tail, then this file's §0.
2. Verify branch = `v1.1-extensions` and tag `v1.0-submission-candidate` exists.
3. Re-project compute + calendar (directive 7); apply cuts if needed; log.
4. Resume the queue at the recorded position; backfill idle GPUs.
5. Post a feed heartbeat if none in the last 12 h of activity.
