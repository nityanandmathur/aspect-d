# ASPECT-D — results feed (v1.1 extensions)

Append-only. Newest entries at the bottom. Every completed job and every gate
gets an entry: what ran, headline numbers with CIs, one-sentence interpretation,
whether it is **pre-registered** (with its §9 id) or **exploratory**, and what it
changes for the paper. Corrections are new entries; nothing above is ever edited.

v1.0 is frozen and unaffected by everything here: declared outcome **S1**,
Δτ = +0.1102 CI [0.0921, 0.1312], 45 runs, paper v1 at
`paper/main-v1-frozen.tex`, tag `v1.0-submission-candidate`.

---

## 2026-08-06 22:18 UTC — GATE E-AUDIT — **PASS** (pre-registered gate, task-v1.md §1)

**Ran:** load-test of every v1.0 final checkpoint + a re-synthesis smoke test.

- **45 / 45** checkpoints present and loadable at step 30 000 (15 configs × seeds {0,1,2}); none missing, none corrupt.
- Smoke: C3 seed 0 re-synthesised at T=16 → **bit-identical tokens** to the frozen v1.0 grids.

**Interpretation:** the v1.0 checkpoints are intact and the sampler is exactly
reproducible, so every Tier-1 inference job (E1/E3/E5/E7) can run on them and
remain comparable with v1.0. The >20 %-missing inversion clause did not fire;
the queue runs in stated priority order.

**Trap worth knowing:** the first smoke attempt reported only 35.7 % token
agreement — because it synthesised a *5-item* batch against v1.0's *50-item*
batches. The sampler's Gumbel draws are shaped `[B, Fmax, V]` from a generator
keyed by `(batch, level, step)`, so **batch composition is part of the RNG
state** (LOG.md P0-3 guarantees reproducibility only at fixed composition).
E1 therefore synthesises the first 200 items as v1.0's first four 50-item
batches; any other chunking would silently break curve comparability.

**Paper impact:** none directly; unblocks everything else.

---

## 2026-08-06 22:20 UTC — E0 pre-registration frozen

`PREREGISTRATION-v1.1.md` written with task-v1.md §9 copied **verbatim**
(H-E1…H-E6) before any extension datum existed. Tag
`v1.0-submission-candidate` pushed; work continues on branch `v1.1-extensions`;
`paper/main-v1-frozen.tex` preserved.

---

## 2026-08-06 22:35 UTC — E2 iso-latency Pareto — **H-E2 NOT SUPPORTED** (pre-registered, §9 H-E2)

**Ran:** zero-GPU analysis over the frozen v1.0 75-point surface with measured
`c_layer`. Serial latency `L = 8·T·d·c_layer`; 40 log-spaced budgets from
16 ms to 2272 ms; at each budget the WER- and SIM-optimal allocation found two
independent ways — minimising the **fitted** `M_sep` surface, and minimising the
**measured** config×T means.

| | WER | SIM-o |
|---|---|---|
| exact agreement (same config *and* T) | **62 %** | 64 % |
| same-depth agreement | 72 % | 64 % |
| "depth-first to d\*" holds | **6 %** | 0 % |

**Decision:** H-E2 required ≥ 80 % agreement → **not supported** (62 %). The
structural half fails much harder: when the optimum uses more than the minimum
T, its depth is at or above the budget's interior optimum d\* in only **6 %** of
budgets.

**What the measured optimum actually does** (WER, selected budgets):

| L (ms) | 18 | 51 | 84 | 140 | 385 | 640 | 1062 |
|---|---|---|---|---|---|---|---|
| best shape | A1 d4 | B1 d6 | B2 d10 | C1 d8 | B2 d10 | B3 d16 | B3 d16 |
| best T | 1 | 2 | 2 | 4 | 8 | 8 | 16 |
| err_WER | 1.128 | 0.851 | 0.766 | 0.351 | 0.209 | 0.166 | 0.119 |

**Interpretation (the finding, and it is a better one than H-E2 predicted):**
at a fixed *serial-latency* budget the optimum **interleaves depth and steps**
rather than saturating depth first, and it **never selects the deepest shapes**
— it tops out at d = 16 (B3) and never reaches the d\* ridge (18 / 30 / 36) that
minimises WER at fixed *parameters*. The reason is structural: latency charges
`8·T·d`, so depth and steps cost the same per unit of latency, but depth is
charged **twice** — once in latency and again in parameters, since at fixed N a
deeper shape must be narrower — while steps are charged **once**. The d\* ridge
is the right answer to "best shape at fixed parameter count" and the wrong
answer to "best shape at fixed latency". Note that the entire C budget is
latency-inefficient: B3 (50 M, d16) dominates the high-latency end outright.

**Also:** the 38 % method disagreement is concentrated in the mid-latency range
and is expected to be partly an artefact of the fitted surface, whose width
amplitude A sits at its pre-registered bound (the v1.0 defect). E5's
log-amplitude refit tests exactly that, so the disagreement gets re-measured
there rather than being explained away here.

**Pre-registered?** Yes — H-E2, and it is refuted as stated.
**Paper impact:** this is the iso-latency figure the v1.1 plan allots main-text
space to (`artifacts-v1.1/figures/iso_latency_pareto.svg`). It replaces the
practical question H-D3 was meant to answer, with a measured answer rather than
a fitted exchange rate.

---

## 2026-08-06 23:40 UTC — **CORRECTION** to the E2 entry above (two errors, verdict unchanged)

An adversarial code review of the v1.1 extensions found two real defects in how
the 22:35 E2 entry was computed and described. H-E2 remains **NOT SUPPORTED**,
but both the number behind that verdict and the interpretation were wrong. The
entry above is left untouched (this feed is append-only); read this instead.

**Error 1 — the decision rule tested the wrong thing.** §9 H-E2 states the
depth-first property "*by both the fitted surface and nearest-measured-point
methods*", decided at "*agreement of both methods at ≥ 80 %*". I implemented
"agreement" as *concordance between the two methods* — do they pick the same
(config, T)? — which measures fit quality, not the hypothesis. The correct
reading is that the **depth-first property must hold under each method** at
≥ 80 % of budgets. Recomputed:

| WER | depth-first holds |
|---|---|
| measured method | **5.9 %** |
| fitted method | **8.8 %** |
| (secondary, descriptive) method concordance | 61.5 % |

**H-E2 is refuted more decisively under the correct rule than under the wrong
one** — 6 % / 9 % against a 80 % bar, and both methods agree it fails. SIM-o:
0 % under both.

**Error 2 — the "never reaches the d\* ridge" claim was false.** I wrote that
the measured optimum "tops out at d = 16 and never reaches the d\* ridge
(18/30/36)". It does: at the two highest budgets the optimum is **B5 (d = 30)**
and **C5 (d = 36)** — exactly the d\* of budgets B and C. My table stopped at
1062 ms and hid them.

**What the data actually shows — the finding is "steps-first", the exact
opposite of H-E2.** Along the WER-optimal path, `T` reaches its maximum tested
value (16) at **L = 726 ms**, while depth first reaches d\* only at
**L = 2002 ms**. There is **not one budget** out of 40 where the optimum spends
below-max steps while sitting at d ≥ d\*. So depth is not bought first and then
steps — steps are bought first, and depth beyond d = 16 is purchased last, once
steps are exhausted. H-E2 predicted the ordering backwards.

**Caveat now recorded in the artifact:** T is capped at 16 by the v1.0 grid, so
"steps exhausted" is partly a boundary of the tested range rather than a true
saturation. **E1 extends T to {24, 32, 64} and this analysis is re-run on the
extended grid** before anything goes in the paper — if the optimum keeps buying
steps past 16, the steps-first reading strengthens; if it switches to depth, the
boundary explanation wins. Either way it is measured, not assumed.

**Pre-registered?** Yes — H-E2, refuted as stated, now by the rule as stated.
**Paper impact:** the iso-latency figure and its caption change from an
agreement statistic to the depth-first fractions and the steps-first ordering.

---

## 2026-08-06 23:45 UTC — GATE G1-D — **FAIL** → pre-registered LR-sweep remedy triggered

**Ran:** 3 000-step D3 proxy (285 M non-embed, w1088 d20) at the μP-transferred
base LR 0.004, per task-v1.md §4-E4.

| | val @ 3k |
|---|---|
| D3 (budget D) | **5.6520** |
| B3 (budget B) | 5.6485 |
| C3 (budget C) | 5.6893 |

No divergence; run completed in 0.61 GPU-h. The gate requires val@3k **below
both** B3's and C3's (a monotone-in-N check). D3 beats C3 but sits 0.0035 above
B3 → **gate fails**, by a hair.

**Action (pre-registered, not improvised):** §4-E4 specifies the remedy —
"*fail → one 5-point LR sweep at D3 only; adopt its argmin for all D runs; log*".
The sweep is running now over base LR ∈ {0.001, 0.002, 0.004, 0.008, 0.016};
the 0.004 point is the proxy above and is reused rather than recomputed
(identical config/seed/steps, deterministic). E4 launches at the argmin.

**Interpretation:** at 3 k steps a 285 M model has seen ~0.7 B tokens and is
nowhere near its capacity, so a near-tie with a 50 M model is weak evidence of
a bad LR — but the gate is pre-registered and it failed, so the remedy runs.
**Paper impact:** none yet; gates E4, which carries H-E4.
