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

---

## 2026-08-07 00:35 UTC — E1 extended test-time scaling, T ∈ {1…64} — **H-E1 PARTIALLY SUPPORTED** (pre-registered, §9 H-E1)

**Ran:** 21 runs (A3, B3, C1–C5 × seeds {0,1,2}) × T ∈ {1,2,4,8,16,24,32,64},
200 eval items, 63 new syntheses + 63 new scorings. **Every T is computed over
the same first 200 items** — the T ≤ 16 points are re-aggregated read-only from
v1.0's per-item rows rather than reused at 400 items, so the curve measures the
T range and not a change of subset. No v1.0 artifact was written.

| T | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| **WER** | 1.146 | 0.760 | 0.321 | 0.191 | 0.1516 | 0.1376 | 0.1314 | **0.1294** |
| **SIM-o** | 0.216 | 0.282 | 0.334 | 0.360 | 0.3696 | 0.3710 | 0.3705 | **0.3716** |
| **UTMOS** | 1.52 | 1.94 | 2.49 | 2.80 | 2.97 | 3.02 | 3.05 | **3.08** |

**Pre-registered part (a) — T\*_WER > 16: SUPPORTED.** T\*_WER = **32**
(v1.0's own 95 %-of-reference definition, `fit.saturation_T`, reference moved to
T=64). WER was *not* saturated at T=16: going 16 → 64 still buys a **15 %
relative** WER reduction (0.1516 → 0.1294).

**Pre-registered part (b) — refitted τ_WER inside the v1.0 CI: FAILS.**
τ_WER on the extended range = **0.9527**, CI [0.9371, 0.9681]; the v1.0 CI is
[0.8248, 0.8513]. **Disjoint.** H-E1 required both halves, so H-E1 as a
conjunction is **not supported** — reported as measured, exactly as §9 provides
for.

**Why τ moved — decomposed with a pre-planned control.** Refitting on the *same
7 configs and same 200 items* but only the v1.0 T range gives τ_WER = **0.8773**
CI [0.8632, 0.8933]. So the shift splits into a **subset effect of ≈ +0.03**
(7 configs / 200 items vs 15 configs / 400 items) and a **range effect of
≈ +0.075** (T ≤ 16 vs T ≤ 64). The range effect dominates: the T-power law is
genuinely steeper once the tail is observed, so v1.0's τ was mildly
*under*-estimated by truncation at T=16, not biased by the subset.

**The headline finding, now non-parametric.** T\*_SIM = **8** while
T\*_WER = **32** — a **4× gap in saturation step**. From T=16 to T=64, WER
improves 15 % relative while SIM-o improves **0.5 %** (0.3696 → 0.3716). The
paper's S1 claim — *refinement steps buy intelligibility, not identity* — no
longer rests on the fitted exchange rate Δτ alone; it is now visible as two
directly measured saturation points on the same runs and the same items. This
is the strongest single piece of evidence in the project.

**Pre-registered?** Yes — H-E1. Part (a) supported, part (b) refuted,
conjunction not supported; T\* values reported regardless, as pre-registered.
**Paper impact:** promotes the extended-T curve to a main-text figure; the S1
claim gains a fit-free demonstration; τ is reported with the truncation caveat.
**Feeds back into E2:** the iso-latency analysis capped T at 16 and concluded
"steps-first". Since WER demonstrably keeps improving to T=32, that conclusion
must be re-tested on the extended grid — queued.

---

## 2026-08-07 01:05 UTC — E2 re-run on the extended T grid — **the "steps-first" reading was a boundary artefact** (exploratory sensitivity)

Promised in the 23:40 correction: the original E2 capped T at 16, so its
"steps-first" ordering could have been an artefact of the tested range rather
than a property of the allocation. E1 now supplies T up to 64, so the analysis
was re-run on the extended grid. **Only budget C sweeps depth in the extended
subset** (5 configs, d ∈ {8,12,18,26,36}; budgets A and B contribute a single
config each), so this re-run is restricted to budget C and is
**exploratory-sensitivity, not a re-decision of H-E2**.

| | v1.0 grid (T ≤ 16, 3 budgets) | extended (T ≤ 64, budget C) |
|---|---|---|
| depth-first, measured | 6 % | **37 %** |
| depth-first, fitted | 9 % | 9 % |
| L where T first hits its max | 726 ms | **9089 ms** |
| L where d first reaches d\* | 2002 ms | **1198 ms** |

**H-E2 is refuted on both grids** (37 % and 6 %, against an 80 % bar), so the
pre-registered verdict is unchanged and robust.

**But my published interpretation of *why* was wrong and is withdrawn.** On the
capped grid, steps hit their maximum (726 ms) long before depth reached d\*
(2002 ms), and I described the optimum as "steps-first". On the extended grid
the order **reverses**: depth reaches d\* at 1198 ms while steps do not exhaust
until 9089 ms. The apparent steps-first behaviour was T=16 being the edge of the
tested range, exactly the failure mode flagged in the caveat.

**What survives, stated at the strength the data supports:** the optimum
**interleaves** depth and steps and does not saturate depth first — 37 % of
budgets satisfy the depth-first property, so it fails as a *rule* while being far
from never true. The clean directional claim ("depth is bought last") does not
survive and is not going in the paper. The artifact now records
`budgets_included` and `n_configs` so this scope limit travels with the number.

**Pre-registered?** No — H-E2's decision stands on the pre-registered v1.0 grid
(6 %, refuted). This re-run is a labelled sensitivity check on the
interpretation.
**Paper impact:** the iso-latency figure keeps its refutation but loses the
directional story; caption changes to the interleaving statement with the
budget-C scope stated. Cost: zero GPU.

---

## 2026-08-07 02:20 UTC — E3 NFE allocation across codebook levels — **H-E3 SUPPORTED** (pre-registered, §9 H-E3)

**Ran:** 15 C-budget runs (C1–C5 × seeds {0,1,2}) × 3 frozen per-level step
schedules at **matched total NFE = 32**, full 400 items — 45 syntheses, 45
scorings, 18 000 item-rows. Paired item-level bootstrap, 2 000 replicates,
RNG 7331; pairing is on the eval item, so item difficulty cancels.

| schedule | steps per codebook level | WER | SIM-o | UTMOS | degenerate |
|---|---|---|---|---|---|
| **coarse** | [25,1,1,1,1,1,1,1] | **0.2191** | 0.3057 | 2.168 | 0.1 % |
| uniform | [4,4,4,4,4,4,4,4] | 0.3144 | **0.3477** | **2.514** | 0.5 % |
| fine | [1,1,1,1,1,1,1,25] | 1.1481 | 0.2246 | 1.561 | **39.1 %** |

**Primary (pre-registered): SUPPORTED.** WER(coarse) − WER(fine) =
**−0.9289**, CI **[−0.9595, −0.8999]**, excludes 0.
**Secondary (pre-registered): SUPPORTED.** WER(coarse) − WER(uniform) =
**−0.0953**, CI [−0.1044, −0.0870], excludes 0 — comfortably ≤ 0.
**H-E3 is supported on both counts** — the first v1.1 hypothesis to confirm.

**Magnitude:** at identical inference cost, moving the 32 forward passes from
the finest codebook level to the coarsest changes WER by **5.2×** (1.148 →
0.219) and cuts the degenerate rate from **39.1 % to 0.1 %**. Spending steps on
the fine level is close to not refining at all: level 7 carries residual detail
that no amount of iteration can use to fix a wrong coarse plan, whereas level 0
sets the phonetic content every later level is conditioned on.

**Practical calibration against E1's curve:** uniform at NFE=32 is T=4 per
level, and E1 measures WER 0.321 at T=4 vs 0.191 at T=8. Coarse's 0.219 sits
between them — so **reallocating the same NFE to the coarse level buys roughly
what 1.7× more uniformly-spent NFE would buy**, for free.

**The interesting nuance — allocation is metric-selective too, and coarse is
not uniformly better.** Against uniform, coarse *wins* WER by 0.095 but *loses*
identity and perceptual quality, both with CIs excluding 0:
SIM-o **−0.0420** CI [−0.0461, −0.0378] and UTMOS **−0.3456** CI [−0.3589,
−0.3311]. So the coarse schedule trades identity and naturalness for
intelligibility. This is the same selectivity the project's headline reports for
the *number* of steps, now appearing in *where* the steps are spent — and it is
a caveat against recommending coarse allocation as a free win. Labeled
exploratory (the SIM/UTMOS contrasts were not pre-registered).

**Pre-registered?** Yes — H-E3, supported. The SIM/UTMOS coarse-vs-uniform
contrasts are exploratory and labeled so.
**Paper impact:** a main-text result with a directly actionable recommendation
plus its honest cost; strengthens the metric-selectivity story by showing it on
a second, independent axis (allocation, not just amount). Cost: 45 syntheses +
45 scorings, no training.

---

## 2026-08-07 03:15 UTC — GATE G1-D remedy complete — **lr = 0.002 adopted**, E4 launched

The 5-point LR sweep at D3 pre-registered in §4-E4 as the remedy for the failed
gate, 3 000 steps each:

| base LR | 0.001 | **0.002** | 0.004 | 0.008 | 0.016 |
|---|---|---|---|---|---|
| val @ 3k | 5.6946 | **5.5900** | 5.6520 | 6.1998 | 6.6751 |

A clean U-shape with an interior minimum. **argmin = 0.002**, adopted for all
budget-D runs as the rule specifies. At 0.002 the proxy reaches 5.5900, **below
both** B3 (5.6485) and C3 (5.6893) — so the monotone-in-N check the gate
encodes **passes at the adopted LR**; only the μP-transferred 0.004 failed it.

**Interpretation:** μP transfer from base width 256 held across budgets A–C but
degraded mildly at 285 M, landing ~2× too high. The gate is exactly the check
that catches this, and the pre-registered remedy fixed it without any judgement
call — the argmin is a deterministic computation, logged in
`artifacts-v1.1/g1d_lr_sweep.json`.

**E4 launched:** D1–D5 × seeds {0,1}, 30k steps, lr 0.002. Cost so far for the
gate + remedy: 3.0 GPU-h of the 600-h envelope.

---

## 2026-08-07 07:30 UTC — E5 robustness panel — **Δτ is robust; the fallback SV model fails instrument validation and is excluded** (sensitivity, NOT a hypothesis test)

**Ran:** 450 re-scorings of the frozen v1.0 audio across the full T grid under two
metric swaps, then a Δτ refit per cell with the §7.2 machinery unchanged. The
metric each variant does *not* change reproduces v1.0 exactly (4/4 consistency
checks OK), confirming the swap is isolated.

| variant | Δτ | 95 % CI |
|---|---|---|
| v1.0 baseline (reproduction) | +0.1117 | [0.0924, 0.1304] |
| ASR = whisper-medium.en | **+0.1292** | [0.1081, 0.1487] |
| log-amplitude parameterisation | **+0.1262** | [0.1043, 0.1474] |
| SV = wavlm-base-plus-sv *(see below)* | −0.5837 | [−0.6216, −0.5487] |
| both swapped *(see below)* | −0.5660 | [−0.6078, −0.5281] |

**Δτ is robust to the two variations that are valid**: swapping the ASR moves it
+0.018, and the log-amplitude reparameterisation moves it +0.015. Both keep the
sign, keep the CI clear of 0, and overlap the v1.0 declared interval. **S1
stands.**

**The SV swap flips the sign — and carries no evidential weight, because the
fallback SV model fails the project's own pre-registered instrument-validation
gate G0(c).** Run on *ground-truth human audio*, before any model output is
involved:

| G0(c) on ground truth | wavlm-large (primary, seed-tts-eval) | wavlm-base-plus-sv (fallback) | bar |
|---|---|---|---|
| same-speaker median SIM-o | 0.7005 | 0.9488 | ≥ 0.50 |
| cross-speaker median SIM-o | **0.0338** | **0.6601** | ≤ 0.25 |
| discriminative gap | 0.6667 | 0.2887 | — |
| **verdict** | **PASS** | **FAIL** | |

The fallback rates two **different real speakers** at 0.6601. Our *worst*
generated audio — T=1, 115 % word error, barely intelligible — scores 0.8494
under it, i.e. **higher than two genuine recordings of different people**. The
model has no discriminative power in the band our outputs occupy, so its τ_SIM
(1.4231 vs the primary's 0.7260) is measuring compression of an uninformative
scale, not identity scaling. G0(c) is exactly the pre-registered check for "is
this instrument fit to measure with", and this one is not.

**Decision:** the SV-swap cells are reported for completeness and **excluded from
inference**. No pre-registered claim changes. E5 is labeled in the runbook as
"sensitivity, NOT hypotheses" (§4-E5), and a sensitivity probe run on an
instrument that fails validation cannot and does not overturn a pre-registered
result decided under the validated one.

**Correction to my own earlier framing.** On first seeing the flip I suggested
the paper should demote Δτ and lead with E1's saturation contrast instead. That
was wrong on two counts: Δτ is robust under every *valid* variation, and the
flip came from a broken instrument rather than a fragile statistic. E1's
T\*_SIM=8 vs T\*_WER=32 remains valuable as a **fit-free corroboration** of S1 —
it is not a replacement for Δτ, and it is not motivated by this flip.

**Paper impact:** appendix gains the {ASR × SV × parameterisation} table with the
G0(c) row that disqualifies the fallback column; main-text claims unchanged.

---

## 2026-08-07 08:40 UTC — **RETRACTION: the "4× saturation gap" (T\*_SIM=8 vs T\*_WER=32) is an artifact of the T\* definition and is withdrawn**

An adversarial verification of my own E5 write-up attacked the E1 headline
instead, and it was right. **I promoted a scale artifact to "the strongest
single piece of evidence in the project." It is not evidence at all.**

**What is wrong.** v1.0's `T*` rule is "within 5 % of the T_ref value" applied to
each metric's **own raw level**. WER and 1−SIM have very different offsets, so
the same nominal 5 % is a wildly different demand on each:

| | 5 % of the T=64 level | as a share of that metric's total range |
|---|---|---|
| WER | 0.00647 | **0.64 %** (very strict) |
| 1−SIM | 0.03142 | **20.22 %** (very lenient) |

A **32× difference in strictness**. That, and not the models' behaviour, is what
produced 8 vs 32.

**Normalise it away and the gap vanishes.** Fraction of each metric's own total
T=1→64 improvement (invariant under err → a + b·err, so immune to offset and
scale — no fit, no convention):

| T | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|
| WER | 38.01 % | 81.13 % | 93.92 % | **97.81 %** | 99.19 % | 99.80 % |
| 1−SIM | 42.66 % | 75.95 % | 92.63 % | **98.74 %** | 99.62 % | 99.30 % |

**T\*(WER) = 16 and T\*(SIM) = 16 — identical. Gap 1×, not 4×.** The two curves
are near-superimposed, and at T=8 SIM is marginally the *later* of the two
(92.63 % vs 93.92 %). On the T ≤ 16 grid the affine-invariant fraction realised
by T=8 is WER 94.88 % CI [94.29, 95.38] vs 1−SIM 93.79 % CI [92.86, 94.74] — a
paired difference of −1.09 pp, CI [−1.95, −0.19], which **excludes zero in the
direction opposite to my claim**.

**What still stands, unchanged:**
- **H-E1 part (a) is still SUPPORTED as pre-registered.** "T\*_WER > 16" is a
  statement about WER on its own scale, using v1.0's frozen definition; T\*_WER
  = 32 > 16. It involves no cross-metric comparison. Part (b) still fails.
- **WER genuinely keeps improving past T=16**: 0.1516 → 0.1294, a 15 % relative
  reduction. That is a real, useful, correctly-scoped fact.
- **S1 / H-D2 stands on its pre-registered statistic**: Δτ = **+0.1102**, CI
  [0.0921, 0.1312], scoped to the frozen seed-tts-eval metric stack and the
  pre-registered parameterisation, robust to the ASR swap (+0.1283) and to the
  log-amplitude reparameterisation (+0.1260).

**What is withdrawn:** the cross-metric saturation comparison and every sentence
built on it — "a 4× gap in saturation step", "makes the S1 claim fit-free", "the
strongest single piece of evidence in the project", and the earlier suggestion
that the paper should lead with it instead of Δτ. That comparison was **never
pre-registered** — I invented it during v1.1 — and swapping the pre-registered
primary statistic for it would have violated §0 directive 1 and §7 in any case.

**The one fit-free contrast that may survive**, reported at its true strength: a
significance-based saturation point (smallest T whose paired-bootstrap CI for
err(T) − err(64) covers zero) gives T\*_WER = 32 vs T\*_SIM = 16 — a **2×**
contrast, not 4×. It is partly a precision artifact: SIM's detection floor is
≈1.8× looser relative to its own dynamic range. It goes in the appendix with
that caveat, not in the headline.

**Recorded in code so it cannot recur:** `e1_extended.py` now emits
`saturation_affine_invariant` and a `T_star_scale_caveat` block next to the raw
`T*`, stating that the raw-scale gap must not be reported as evidence of
metric-selective test-time scaling.

**Paper impact:** main-text claims revert to exactly what v1.0 pre-registered.
`extensions.html` corrected. Net effect of v1.1 on the headline: **none** — Δτ
survived three independent robustness axes, which is a stronger position than
the one I briefly talked myself into.

---

## 2026-08-07 09:30 UTC — **CORRECTION to my own retraction, and a serious finding about what Δτ actually means**

The verification workflow completed. It confirmed the T\* retraction but **refuted
my explanation of the SV flip**, and then found something worse in the paper's
wording. Three items.

### 1. "τ_SIM = 1.4231 is measuring scale compression" — FALSE, struck

τ is **affine-invariant** under M_sep. For err → a + b·err the identical fit is
attained by E → a+bE and A,B,C → bA,bB,bC with α, β, τ **unchanged**; the
weighted objective is invariant because the SEs scale by b and fit.py's SE floor
is itself scale-relative. Verified numerically: mapping wavlm-large's `err_sim`
onto base-plus-sv's exact numeric range moves τ_SIM from 0.726019 to **0.724643**
— a shift of −0.0014, i.e. **0.2 % of the +0.697 that needs explaining**.
Compression explains essentially none of the flip. My "1.52× vs 1.25× relative
decline" diagnostic was also wrong: err(1)/err(16) is a quantity M_sep never
sees, because E absorbs the origin.

The **real** mechanism is resolution loss and non-monotonicity: base-plus-sv's
SIM curve is non-monotone (SIM(8) > SIM(16) in 26/45 runs), which a strictly
monotone C·T^−τ term cannot represent, so τ runs up to flatten the tail. τ_SIM =
1.42 is sharply identified (profiled weighted RSS/min = 1.00 at 1.423, 7.48 at
0.6), not an optimizer artifact.

**The exclusion of base-plus-sv stands on G0(c) alone** — it fails
instrument validation (cross-speaker median 0.6601 against a ≤ 0.25 bar), so
whatever shape it measures is not speaker identity. That argument never needed
the compression story, and the compression sentence is struck from the 07:30
entry and from LOG-v1.1.md.

### 2. My retraction's arithmetic was off by 1.8×

I reported the T\* tolerance asymmetry as "0.64 % vs 20.22 %, a 32× difference".
I applied the 5 % band to `1 − SIM`'s level, but `fit.saturation_T` applies it on
each metric's **own reported scale** (SIM-o for identity, WER for
intelligibility). Corrected: band = 0.00681 = **0.67 %** of WER's range vs
0.01858 = **11.96 %** of SIM's — a **17.9×** difference, not 32×. The
retraction's conclusion is unchanged and independently confirmed: a pure affine
remap of wavlm-large's SIM-o (which moves τ by 0.0014) takes T\*_SIM from 8 to 2,
exactly reproducing base-plus-sv's value. **T\* is the scale-dependent statistic;
τ is the invariant one** — the opposite of what I asserted.

### 3. **Δτ > 0 asserts the opposite normalised ordering from the paper's title**

Because the T-term is separable, τ is a convergence rate on each metric's own
normalised curve, and **larger τ means earlier saturation**. With
τ_WER = 0.8362 > τ_SIM = 0.7260:

| | reaches 90 % of its asymptotic T-gain at |
|---|---|
| WER | **T = 15.7** |
| 1 − SIM-o | **T = 23.8** |

So Δτ > 0 says **intelligibility converges *sooner***, not that steps keep
helping intelligibility after identity stops. Model-free agreement: the fraction
of the T=1→16 gain realised at T=8 is 94.6 % for WER and 93.7 % for 1−SIM
(paired difference −0.96 pp, CI [−1.88, −0.07]) — essentially identical, with
identity marginally *later*.

**The pre-registered result is untouched**: Δτ = +0.1102 CI [0.0921, 0.1312]
excludes 0, so H-D2 passes and outcome class S1 is correctly declared. What is
wrong is the **English gloss** — "refinement steps buy intelligibility, not
identity" — which reads as a claim about rates and is backwards as such.

**What is true, and is what actually carries the paper — an absolute-magnitude
claim, not a rate claim.** Over T = 1 → 16 (45 runs, 400 items):

| | T=1 | T=16 | absolute error gain |
|---|---|---|---|
| WER | 1.1383 | 0.1865 | **0.9518** (6.10×) |
| SIM-o | 0.2093 | 0.3699 | 0.1606 (1.25×) |
| degenerate rate | 40.4 % | 0.1 % | — |

Steps move **5.9× more absolute error** on intelligibility than on identity.
That is a comparison of magnitudes across incommensurable metrics — it must be
labeled as such and never as a convergence-rate statement.

**Paper impact: the title and abstract gloss need rewording, and that is a
judgement call I am not making unilaterally.** The pre-registered statistic,
its CI, and the declared outcome class all stand. Flagged for decision.

**Not repeated here:** one lens claim (a shared-τ bootstrap CI including zero)
failed to reproduce under the protocol's own weighting scheme and is excluded.

---

## 2026-08-07 17:00 UTC — E4 budget-D grid — **H-E4 primary SUPPORTED, secondary not** (pre-registered, §9 H-E4)

**Ran:** D1–D5 × seeds {0,1}, 30k steps at the G1-D-adopted lr 0.002, then the
full T grid {1,2,4,8,16} × 400 items plus extended {24,32,64} × 200. Combined
with the frozen v1.0 table into a **275-row, 4-budget (A–D) surface**, 20M → 276M
non-embedding parameters. `artifacts/runs.csv` was read-only throughout; the
combined table is `artifacts-v1.1/runs_4budget.csv`.

**Primary — Δτ > 0 on A–D with run-level bootstrap CI excluding 0: SUPPORTED.**

| | Δτ | 95 % CI |
|---|---|---|
| v1.0, 3 budgets (20–126 M) | +0.1102 | [0.0921, 0.1312] |
| **v1.1, 4 budgets (20–276 M)** | **+0.1095** | **[0.0951, 0.1241]** |

τ_WER = 0.8584, τ_SIM = 0.7489. Adding a budget **4× larger than any in v1.0
moved Δτ by 0.0007** and tightened the interval. The metric-selectivity of
test-time scaling is not a small-model artifact — this is the strongest addition
v1.1 makes to the paper.

**Secondary — extrapolation A+B+C → D: NOT supported, by 0.1 pp.** M_sep fitted
on A+B+C predicts budget-D config means at T=16 with **MAPE 15.1 %** against a
≤ 15 % bar. The rule required *both* MAPE ≤ 15 % *and* ≤ the N-only model's, and
the second clause passes overwhelmingly: **M_N's MAPE is 319.3 %**, so the
shape-aware model extrapolates **21× better** while still formally failing.
Per-config error: D1 +7.1 %, D2 +34.8 %, D3 +4.2 %, D4 −14.7 %, D5 +14.8 % —
D2 alone carries the failure. Reported as measured.

**Exploratory d\*(N): no trend is reportable, and the naive reading is wrong.**
Raw d\* by budget is A 18, B 30, C 36, D 14, which looks like a rise-then-collapse
story. It is not:

- **B and C are censored** — their d\* sits at the deepest shape those budgets
  tested (30 and 36), so they are **lower bounds**, not interior optima.
- **D's minimum is a near-tie** — d=14 gives WER 0.0978 against d=38's 0.0990, a
  gap of **0.0012**, with a bump between them.

Only A (18) and D (14) are genuine interior optima, and D's is a coin flip. The
artifact now records `censored_at_max_depth` and `margin_to_runner_up` per budget
so this cannot be read off naively. **No d\*(N) claim goes in the paper.**

**Pre-registered?** Yes — H-E4. Primary supported, secondary refuted, d\* was
explicitly exploratory with no decision rule.
**Cost:** 10 runs ≈ 60 GPU-h training + evaluation.

---

## 2026-08-07 17:10 UTC — E6 training-compute control at 90k steps — **H-E5 SUPPORTED** (pre-registered, §9 H-E5)

**Ran:** {C1, C3, C5} seed 0 retrained to **90k steps** (3× v1.0's 30k), then
scored on the v1.0 T grid × 400 items. Item-level bootstrap, 1 000 replicates —
item-level because one seed per config leaves nothing to resample at the run
level, exactly as pre-registered.

Longer training worked, and it is not a no-op:

| config | val loss @30k | @90k | cost |
|---|---|---|---|
| C1 | 4.4818 | **4.0907** | 7.0 GPU-h |
| C3 | 4.4702 | **4.1194** | 9.8 GPU-h |
| C5 | 4.4823 | **4.1791** | 13.0 GPU-h |

**Result: Δτ_90k = +0.1767, CI [0.0803, 0.2849] — excludes 0. SUPPORTED.**
So **undertraining does not explain away the effect**, which is precisely the
threat this control existed to test.

**The matched-30k comparison, read carefully.** On the same three configs, same
seed, same items at 30k: Δτ = +0.0445, CI [−0.0572, +0.1446] — which *includes*
0. The two CIs **overlap**, so the honest statement is that Δτ at 90k is
consistent with Δτ at 30k and cannot be claimed to have grown.

**What that 30k interval does *not* mean:** it is not evidence that 30k lacks the
effect. Three configs and one seed is simply underpowered — v1.0's full
15-config, 3-seed surface at the *same* 30k gives Δτ = +0.1102 with a tight CI
[0.0921, 0.1312]. The 3-config CI widening to include 0 is a sample-size
property, not a training-compute one, and it must not be reported as "the effect
only appears with more training".

**Also observed (exploratory, not the test):** at 30k the three shapes are nearly
indistinguishable in val loss (4.470–4.482, spread 0.012); at 90k they separate
(4.091 / 4.119 / 4.179, spread 0.088 — 7× wider) and order consistently. Shape
differences that 30k could not resolve become visible with more training. This
is a single seed and is labeled exploratory.

**Pre-registered?** Yes — H-E5, supported, with the single-seed scope
acknowledged in the pre-registration as a control rather than a headline.
**Cost:** 29.8 GPU-h.

---

## 2026-08-08 14:30 UTC — protocol-compliance rerun: E1 and E6 bootstraps at the frozen 2 000 replicates

**MEASURED.** `e1_extended` and `e6_analysis` had last been run at 100 and 1 000
bootstrap replicates against the **2 000 frozen in task-v1.md §5 / protocol §7.2**.
Both re-run at 2 000; no verdict changes.

| | at the reduced count | **at 2 000 (authoritative)** |
|---|---|---|
| H-E5 Δτ_90k | +0.1767, CI [0.0803, 0.2849] | **+0.1767, CI [0.0789, 0.2854]** |
| H-E5 matched 30k | +0.0445, CI [−0.0572, 0.1446] | **+0.0445, CI [−0.0570, 0.1484]** |
| H-E1 τ_WER extended | CI [0.9371, 0.9681] | **CI [0.9374, 0.9686]** |
| H-E1 τ_WER control (T ≤ 16) | CI [0.8632, 0.8933] | **CI [0.8629, 0.8943]** |

Subset effect **+0.0411**, range effect **+0.0754** (both from the 2 000-replicate
refit). H-E5 remains SUPPORTED; H-E1's conjunction remains not supported; the 90k
and matched-30k intervals still overlap.

**INTERPRETATION.** Nothing changes. This entry exists because the reduced counts
were a silent protocol deviation, and the third such deviation found in this
project by re-checking rather than by noticing — the same class as the E5 panel's
bootstrap-median estimator. The paper's macros are regenerated from the 2 000-rep
artifacts, so every published number now comes from the frozen count.

---

## 2026-08-08 15:25 UTC — **GATE M-DONE: v1.1 merged to `main`, tagged `v1.1`, paper v3 final**

**MEASURED.** PR #1 (`v1.1-extensions` → `main`), 976 files, +84 680/−94, merged
as `55a47d43` with no history rewrite and no force push. Tag `v1.1` pushed;
`v1.0-submission-candidate` intact.

Immutability attestation at merge time —
`git diff main -- artifacts/ PREREGISTRATION.md LOG.md` **empty**;
`paper/main-v1-frozen.tex` is a file *added* by v1.1, not an edit.

Paper v3 verified: compiles with no errors, References begin on page 5 so main
text is exactly **4 pages**, anonymised, 97 generated macros and no hand-typed
number, retired-phrase grep zero in `paper/` outside the frozen v1.0 paper.

**Both open decisions are now closed.**

1. **Title/gloss** — resolved to the magnitude framing:
   *Test-Time Refinement Moves Intelligibility Six Times More Than Identity in
   Masked-Diffusion TTS*. The abstract states, in order, the pre-registered
   detection statistic (Δτ = +0.1102, CI [0.0921, 0.1312]), the magnitude claim
   that carries the paper (0.952 absolute WER against 0.161 absolute identity
   error, 5.9×), the rate-honesty sentence (fraction of total gain at T=8:
   94.6 % WER vs 93.7 % identity, identity marginally *later*), and four-budget
   persistence (Δτ = +0.1095, CI [0.0951, 0.1241]).
2. **Merge** — done, as above.

**INTERPRETATION.** The v1.1 program's net effect on the paper is that one new
independent finding was added (E3 allocation) and the original claim was hardened
against the four most obvious objections — scale, training compute, ASR choice,
and parameterisation — while three of the project's own intermediate claims were
withdrawn. The title change is the part worth restating: the paper's statistic
detects *selectivity*, but Δτ > 0 means WER converges **sooner**, so the old gloss
read backwards as a rate claim. The surviving claim is about magnitudes, and the
paper now says exactly that.

**Gate M-DONE is satisfied.** Program S may begin. Next action per task-v2.md §6:
copy §9 verbatim to `PREREGISTRATION-v1.2.md` before any S synthesis, then S0
(identity ledger + codec ceiling, zero training).

---

## 2026-08-08 15:30 UTC — S0 identity ledger + codec ceiling — **H-S0: MODERATE HEADROOM** (pre-registered, v1.2 §9, both lenses agree)

First Program-S job. `PREREGISTRATION-v1.2.md` was frozen verbatim (60/60 quoted
lines verified identical) before this ran. Zero training; 400 items; scoring with
the v1.0 primary SV model only (wavlm-large, seed-tts-eval — passes G0(c);
base-plus-sv excluded everywhere per §8).

### MEASURED *(frozen once posted)*

| quantity | value |
|---|---|
| SIM_rt — Mimi roundtrip of the GT target, vs the original prompt | **0.5554** (median 0.5733) |
| GT target, no roundtrip, vs prompt | 0.6784 |
| → cost of the codec alone | **0.1230** |
| best measured system (any budget, T=16) | **0.4130** (D5, budget D) |
| **headroom h = SIM_rt − best system** | **+0.1423** |
| second lens: per-item paired median (roundtrip − best-system) | **+0.1422** |
| classification, mean | moderate headroom |
| classification, paired median | moderate headroom |
| **verdict** | **MODERATE HEADROOM — both lenses agree** |

The two lenses land within 0.0001 of each other, and the classification is
0.0077 below the "large headroom" boundary at 0.15 — close enough that it is
reported as a boundary case, not as a comfortable interior classification.

**The identity ledger** — absolute SIM-o gain per candidate axis, from existing
artifacts, no new synthesis:

| axis | gain | source |
|---|---|---|
| refinement steps, T=1→16 | **+0.1606** | v1.0 grid |
| NFE allocation at matched cost (fine→uniform) | **+0.1231** | E3 |
| training compute, 30k→90k | **+0.0788** | E6 |
| parameters N, 20M→276M (best config per budget) | **+0.0531** | E4 4-budget table |
| shape at fixed N (best − worst) | **+0.0325** | v1.0 grid |
| *remaining headroom to the codec ceiling* | *+0.1423* | this job |

Per-budget best SIM-o at T=16: A 0.3600, B 0.3883, C 0.3923, D 0.4130.

### Rival table *(filled — "addressed" means a number)*

| pre-listed rival | discriminating check | result |
|---|---|---|
| the roundtrip inherits the reference recording conditions, inflating the ceiling | roundtrip of a **different same-speaker utterance** scored against the same prompt (n = 323) | **0.5546** vs 0.5554 for the same utterance — Δ = **−0.0008**. **Excluded**: the ceiling is a speaker-level property, not an artifact of the specific target recording. |
| the scorer saturates near 1.0, compressing everything at the top | same-speaker GT baseline distance from 1.0, frozen G0(c) | same-speaker median **0.7005**, i.e. **0.2995** of range still above it; cross-speaker 0.0338. **Excluded**: the scorer is nowhere near its ceiling. |

### INTERPRETATION *(revisable, labeled — NOT yet STABLE)*

**Roughly half the identity gap is not the model's to close.** From real audio
(0.6784) to the best measured system (0.4130) is 0.2654 of SIM-o, and the codec
roundtrip accounts for **0.1230 — 46 % of it**. No amount of modelling on Mimi
tokens can recover that half; it is spent before the model sees anything.

**The ledger reframes the headline, and against it two of the biggest identity
axes are not what they look like.** Steps buy the *most* identity of any axis
measured (+0.1606) — more than a 14× increase in parameters (+0.0531). That does
not contradict v1.0's finding, which is about the *asymmetry* between metrics
(steps move 5.9× more WER error than identity error), but it does mean "steps
don't buy identity" would be the wrong gloss: they buy more of it than anything
else we can currently spend. The caveat is what the baseline is: T=1 has a 40.4 %
degenerate rate and NFE-fine has 39.1 %, so the top two ledger entries are
substantially *"stop producing broken audio"* rather than *"acquire the target
speaker"*. A ledger entry measured from a degenerate baseline is not comparable
to one measured between two healthy systems, and the closing Program-S entry must
rank on that basis rather than on raw gain.

**What this sets up.** With +0.1423 of headroom, there is real room for a
test-time intervention to matter — S1 (context), S2 (search), S3 (rate-matching),
S4 (guidance) are all now interpretable against a known ceiling, as §1.5
requires. Had this come back "near-ceiling", most of Program S would have been
answered before it started.

**Cooling status:** the codec-ceiling number is one of only two things §5 permits
into the paper, so it is **NOT STABLE** until an independent adversarial
recomputation passes (§1.4). It does not enter the paper until then.

---

## 2026-08-08 18:40 UTC — **S0 CORRECTION after the §1.4 cooling pass: the headroom halves and the codec share rises to 62 %** (verdict label unchanged)

The cooling rule required an independent adversarial recomputation before the
codec-ceiling number could enter the paper. It found a **real selection error**,
plus four reporting defects. The arithmetic of the original entry reproduced
bit-exactly — every number was right *for the selection made*, and the selection
was wrong.

### MEASURED *(supersedes the 15:30 MEASURED block; that block stays on the record)*

**The error.** "Best measured system SIM-o (any budget, T=16)" was computed by
scanning only `artifacts/runs.csv` and the budget-D rows of `runs_4budget.csv` —
both 30k checkpoints. The frozen rule imposes **no checkpoint restriction**, and
the E6 90k runs are measured systems on the same 400 items with the same SV
model. All three beat every 30k run, on WER as well, with zero degeneracy:

| system | SIM-o | WER | degen |
|---|---|---|---|
| **C3_0_90k** | **0.4807** | 0.0560 | 0.000 |
| C1_0_90k | 0.4741 | 0.0665 | 0.000 |
| C5_0_90k | 0.4638 | 0.0530 | 0.000 |
| D5_0 *(the previously reported best)* | 0.4162 | 0.1051 | 0.003 |

`ledger()` in the same script was already reading those 90k scores and posting
0.4729 as the training-compute endpoint — **the artifact contradicted its own
"best system" row**, and I did not notice.

| quantity | as posted 15:30 | **corrected** |
|---|---|---|
| best measured system | 0.4130 (D5 config mean) | **0.4807 (C3_0_90k)**, 58 systems scanned |
| headroom h | +0.1423 | **+0.0747**, 95 % CI [0.0622, 0.0873] |
| P(h > 0.15) | *not computed* | **0.000** |
| per-item paired median | +0.1422 | **+0.0730** |
| lens gap | "0.0001" | **0.0017** |
| codec share of the identity gap | 46 % | **62.2 %** |
| classification | moderate headroom | **moderate headroom** |

**Three further reporting defects, all fixed:**
1. *The "two lenses agree to 0.0001" claim was an artifact of inconsistent
   aggregation* — the mean averaged both D5 seeds while the median took the single
   better seed. Both lenses now read the best system's own per-item rows. The
   honest gap is 0.0017.
2. *Rival 1's delta was unpaired* — a difference of means over different item sets,
   reported as −0.00076. Paired on the same 323 items it is **−0.00792**, ten times
   larger. The exclusion holds (0.008 against a headroom of 0.075), the number did not.
3. *No CI on a classification sitting near its own boundary.* Under the old
   selection h = 0.1423 had CI [0.1293, 0.1551] — **it crossed 0.15**, with
   P(h > 0.15) = 0.128. Under the corrected selection P = 0.000.

**Censoring check** (the d\* failure mode, now a default): the argmax is
C3_0_90k, which is *not* at an extreme budget or extreme shape index, so the
headroom is not censored from that direction. It remains a lower bound in the
trivial sense that a longer-trained system would raise it further.

**The verdict label is unchanged and is now marked STABLE**: h = +0.0747 falls in
(0.05, 0.15] → **moderate headroom**, and it holds under every selection tested
(30k-only 0.1423, 90k arm mean 0.0825, best single run 0.0747, first-200 subset
0.1481) and under both lenses.

### INTERPRETATION *(revisable, labeled)*

**The correction cuts against my own earlier reading, in the direction that
matters.** I wrote at 15:30 that "there is real room for a test-time intervention
to matter". There is roughly **half** as much: 0.0747, not 0.1423. And the codec's
share of the gap between real audio and the best system rises from 46 % to
**62 %** — so *most* of the remaining identity gap is the representation, not the
model.

**This sharpens what Program S can possibly find.** With 0.0747 of headroom, an
intervention that buys, say, +0.01 SIM-o is claiming ~13 % of everything that is
left. S1–S4 verdicts must be read against that, per §1.5 — and the ledger's
largest entries (steps +0.1606, allocation +0.1231) are measured from degenerate
baselines (40.4 % and 39.1 % degenerate) and so are mostly "stop producing broken
audio", not "acquire the speaker".

**Method note.** This is the second time in this project that a number survived
its own author's checking and fell to an independent recomputation, and the third
time an argmax was reported without a censoring check. The cooling rule is now
doing exactly the work it was written for. The paper-bound sentence will quote
62 % with the selection rule stated inline, not 46 %.

---

## 2026-08-08 19:50 UTC — S1 test-time context (prompt length) — **H-S1: DISCORDANT** (pre-registered, v1.2 §9; first DISCORDANT verdict, diagnosis below)

15 C-budget runs × 4 prompt-length arms × 397 items, T=16, frozen sampler. Every
arm scored against the **same fixed reference** — the canonical v1.0 3 s prompt
waveform — so no arm is scored against its own conditioning audio (§8).

### MEASURED

| arm | SIM-o | WER | degen | UTMOS |
|---|---|---|---|---|
| 1.5 s | 0.3213 | 0.1578 | 0.004 | 2.98 |
| **3 s** (v1.0 default) | **0.3839** | **0.1474** | 0.001 | 3.00 |
| 6 s | 0.3724 | 0.7692 | 0.015 | 2.80 |
| 9 s | 0.3759 | 0.8642 | 0.013 | 2.77 |

| lens | result | verdict |
|---|---|---|
| **primary** — paired per-run SIM-o(9 s) − SIM-o(3 s), run-level bootstrap | **−0.0080**, CI **[−0.0114, −0.0048]**, 2/15 runs positive | **NOT supported** (excludes zero, wrong direction) |
| **second** — Spearman over {1.5, 3, 6, 9} positive in ≥ 12/15 runs | positive in **15/15** | **supported** |
| guardrail | WER(9 s) − WER(3 s) = **+71.7 points** | — |

**Verdict: DISCORDANT.** Not resolved in favour of either reading (§8).

### Diagnosis *(required by §1.1 — which lens measures what, and why they split)*

The arms are **non-monotone with a peak at the v1.0 default**: 0.3213 → **0.3839**
→ 0.3724 → 0.3759. SIM ranks are [1, 4, 2, 3] against second-ranks [1, 2, 3, 4],
giving Spearman ρ = **+0.400** in every run.

- The **primary lens** asks a specific question — *does more context than training
  saw help?* — and answers no, with a tight interval on the wrong side of zero.
- The **second lens** asks *is there a positive association across the tested
  range?* With four points it is dominated by the single worst point: the 1.5 s
  arm is starved, so any curve that rises off it scores positive regardless of
  what happens afterwards. It is **not** measuring monotonicity, which is what
  "trend" was intended to capture.

The lenses do not contradict each other about the data; they are answering
different questions, and the pre-registration's second lens is the weaker
instrument for this shape. That is recorded here rather than resolved: per §1.6
I may not now swap in a better trend statistic and call the result supported.

### Rival table *(filled)*

| rival | check | result |
|---|---|---|
| reference confound | all arms scored against the same fixed 3 s reference | **excluded by construction** — verified all four arms embed `items[n]['prompt_id']` audio |
| train/test prompt-length shift harming WER only | DegenRate by arm | degen stays ≤ **1.6 %** in every arm while WER rises 0.147 → 0.864. The collapse is **not** degenerate output under the §6.4 rule |
| duration-of-evidence artifact in the scorer | content-drift diagnostic | at 9 s the output matches the **prompt** text *worse* (WER 2.97) than the target text (0.83), so it is not copying context |

### INTERPRETATION *(revisable, labeled — NOT STABLE)*

**The mechanism is a decoupling, and it is the interesting part.** At 6–9 s the
model emits fluent, correctly-timed, speaker-consistent audio that has stopped
tracking the phoneme conditioning. Identity is nearly intact (0.3759 vs 0.3839)
while intelligibility collapses by 71.7 WER points. Prompt-length extrapolation
**separates identity from content** — the two capabilities fail independently,
which is the same dissociation the paper reports for refinement steps, appearing
here in a completely different regime.

**Against the ledger (§1.5), S1 buys nothing.** The best arm is 3 s — exactly what
v1.0 already uses. The in-distribution contrast 1.5 s → 3 s is large and robust
(**+0.0625**, CI [+0.0598, +0.0652], 15/15 runs; labeled **exploratory** per §1.6,
invented mid-flight) and it is ~84 % of the entire remaining headroom of 0.0747 —
but it is a gain *from a starved baseline back to the default*, not a gain
available at the default. **No prompt length tested beats the status quo.**

**What this does not license.** It does not show that context is irrelevant to
identity — it shows that context beyond the training distribution does not help
*this* model, which was trained on 3 s prompts only. A model trained with variable
prompt lengths could behave differently, and that is a training-time question, not
a test-time one.

**Cooling status:** NOT STABLE. Nothing here touches the paper; under §5 S1 could
only ever contribute to the single outlook sentence, and a DISCORDANT verdict
selects the "no sentence" branch.

---

## 2026-08-08 21:15 UTC — GATE §1.7 instrument validation: **ECAPA PASSES**; S2 proceeds

Required before any S2 number is used. Same 400 ground-truth items, same speaker
pairing and RNG as the frozen G0(c).

| SV model | same-speaker median | cross-speaker median | gap | verdict |
|---|---|---|---|---|
| wavlm-large (v1.0 primary) | 0.7005 | 0.0338 | 0.6667 | PASS |
| **ECAPA (speechbrain)** | **0.6606** | **0.0598** | **0.6008** | **PASS** |
| wavlm-base-plus-sv | 0.9488 | 0.6601 | 0.2887 | FAIL (excluded, §8) |

Bars: same ≥ 0.50, cross ≤ 0.25. ECAPA clears both with a discriminative gap
within 10 % of the primary model's. **S2 may proceed** with WavLM-SV selecting and
ECAPA scoring — §1.8, selection ≠ scoring, no exceptions.

---

## 2026-08-08 21:20 UTC — S4 canary gate — **PASS at γ = 0.5**, full C-budget released

C3 seed 0, 100 items, T=16, speaker-contrastive guidance
`logits + γ·(logits_cond − logits_wrong-speaker)` with the deterministic
wrong-speaker assignment (item *i* takes the prompt of item *(i+7) mod N*).

| | SIM-o | Δ vs baseline | WER | degen |
|---|---|---|---|---|
| baseline γ=0 | 0.3915 | — | 0.1395 | 0.000 |
| **γ = 0.5** | **0.3998** | **+0.0083** | 0.1566 | 0.000 |
| γ = 1.0 | 0.3861 | −0.0054 | 0.2031 | 0.000 |
| γ = 2.0 | 0.3464 | −0.0451 | 0.3231 | 0.000 |

Gate rule: proceed only if some γ gives DegenRate ≤ 2× baseline **and** SIM-o not
worse than baseline. γ = 0.5 satisfies both (degeneracy stays at zero throughout,
so guidance is not breaking the model — it is trading intelligibility for
identity, and only at low strength). Higher γ degrades both metrics monotonically.

**Implementation note.** `γ = 0` is a strict no-op: the guided path reproduces the
frozen v1.0 grids bit-for-bit, verified before any S4 datum was generated. The
best-of-K candidate index is likewise a disjoint RNG block, and `cand = 0`
reproduces v1.0 exactly — so neither new lever perturbs the frozen record.

**Pre-registered?** The canary is a gate, not a hypothesis. H-S4 is decided on the
full C-budget, now running.

---

## 2026-08-08 23:05 UTC — S3 rate-matched length conditioning — **H-S3: DISCORDANT** (pre-registered, v1.2 §9)

15 C-budget runs × 400 items, T=16. Arm A is v1.0's corpus-median
seconds-per-character (the frozen `synth_T16`); arm B measures the rate from each
item's own prompt clip. Same text, same prompt, same sampler, same RNG keying —
only target length moves. Per-item rates span 0.035–0.171 s/char against the
corpus median of 0.060, changing target length for 389/400 items.

### MEASURED

| lens | result | verdict |
|---|---|---|
| **primary** — paired per-run SIM-o(B) − SIM-o(A), run-level bootstrap | **+0.0044**, CI **[+0.0017, +0.0067]**, 13/15 runs positive | **supported** |
| **second** — per-item paired median, sign test p < 0.05 | median **+0.00415**, 220/400 positive, **p = 0.051** | **NOT supported** (misses by 0.001) |
| guardrail | WER +0.86 points (0.1471 → 0.1557), within ±2.0 | within guardrail |

SIM-o 0.3831 → 0.3875. Degeneracy essentially unchanged (0.0008 → 0.0007).
Mean generated duration 7.59 s → 8.04 s.

**Verdict: DISCORDANT.** Not resolved toward the supported lens (§8).

### Diagnosis *(§1.1)*

The lenses aggregate different things and the data sit exactly between them.
The primary averages **within run** first (15 units), so per-item noise cancels
and a small, consistent shift becomes detectable — 13/15 runs move the same way.
The sign test asks the much noisier per-**item** question, and a +0.004 mean shift
with wide per-item spread yields only **55 % of items positive**, which at n = 400
gives p = 0.051.

So the honest statement is: rate-matching produces a **small mean improvement that
is consistent across runs but not reliable item-by-item**. Both lenses are
measuring correctly; they disagree because the effect is real and tiny. Per §1.6 I
may not now switch to a paired t-test or a Wilcoxon and claim support.

### Rival table *(filled)*

| rival | check | result |
|---|---|---|
| duration change alters how much audio is scoreable | SIM delta by baseline-duration quartile | **Q1 +0.0083, Q2 +0.0081, Q3 −0.0013, Q4 +0.0025** — the gain is concentrated in the two *shortest* quartiles, exactly where the corpus median mis-sets length most (Q1 4.7 s → 6.2 s). Not a uniform shift, so it is not a scoring artifact of longer audio. |
| ASR length sensitivity | WER delta by the same quartile | **Q1 −0.0076, Q2 +0.0113, Q3 +0.0137, Q4 +0.0172** — WER *improves* where SIM improves most and degrades on long items, so the two metrics do not move together; the SIM gain is not a by-product of an ASR length effect. |

### INTERPRETATION *(revisable, labeled — NOT STABLE)*

**Ledger-relative (§1.5): +0.0044 is 5.9 % of the remaining 0.0747 headroom.**
Real, cheap (zero training, one number changed at inference), and small. It is the
first axis in Program S to move identity *at all* without a cost — S1's best arm
was the status quo, and this beats the status quo slightly.

**The quartile pattern is the informative part.** The gain lives almost entirely
in short utterances (Q1/Q2), where a corpus-median rate mis-sets duration worst.
That is consistent with the mechanism the paper's new limitation sentence names:
length conditioning from a corpus constant penalises speakers whose rate differs
from the corpus. Fixing it recovers a small amount of identity, mostly for the
items it was hurting most.

**Cooling status:** NOT STABLE, and under §5 nothing from S3 may enter the paper
beyond the existing limitation sentence, which already states the mechanism.

---

## 2026-08-09 02:30 UTC — S2 test-time search vs refinement — **H-S2: SUPPORTED** (pre-registered, v1.2 §9; both lenses, all three tiers)

15 C-budget runs × 200 items, 18 000 rows. **Selection ≠ scoring (§1.8):**
candidates selected by WavLM-SV cosine to the prompt, selected candidate scored by
**ECAPA**, which passed the §1.7 instrument gate. Refinement arms reuse the frozen
audio, re-scored with the same instrument so both sides of every tier are measured
identically.

### MEASURED

| matched NFE | refinement | search | **ΔECAPA** | 95 % CI | per-item win rate | ΔWER |
|---|---|---|---|---|---|---|
| 128 | T=16 | K=2 @ T=8 | **+0.0153** | [+0.0127, +0.0179] | 57.4 % [55.6, 59.2] | +0.0408 |
| 256 | T=32 | K=4 @ T=8 | **+0.0322** | [+0.0307, +0.0338] | 65.3 % [63.6, 67.1] | +0.0612 |
| 512 | T=64 | K=8 @ T=8 | **+0.0444** | [+0.0416, +0.0470] | 72.3 % [70.6, 73.9] | +0.0639 |

Primary supported in **3/3** tiers; second lens (WavLM-SV directionally consistent
— +0.0292 / +0.0560 / +0.0803 — plus per-item win rate > 50 % with CI) supported
in **3/3**. The gain is **monotone in K**.

**Pre-registered secondary (symmetry): also supported.** WER(refinement) <
WER(best-of-K) in **3/3** tiers — refinement 0.1530/0.1335/0.1316 against search
0.1938/0.1947/0.1955.

### Rival table *(filled)*

| rival | check | result |
|---|---|---|
| selection–scoring circularity | ECAPA scores what WavLM-SV selected; ECAPA gated on ground truth first | **excluded by design.** ECAPA: same 0.6606 ≥ 0.50, cross 0.0598 ≤ 0.25. The gain also holds under the selector's own metric, which is *reported* and not treated as evidence. |
| best-of-K covertly selecting non-degenerates | DegenRate and WER of the *selected* candidates | degeneracy is flat and negligible on both sides (**0.0003–0.0010**), so selection is not just avoiding broken outputs. WER *rises* under search, so it is not covertly selecting intelligibility either. |
| variance-only effect (tail, not median) | per-item **median** delta | **+0.0127 / +0.0301 / +0.0414** — the median item moves nearly as much as the mean, so this is a shift of the whole distribution, not a tail artifact. |

### INTERPRETATION *(revisable, labeled — NOT STABLE)*

**This is the largest identity gain in Program S by a wide margin.** At NFE 512,
+0.0444 is **59 % of the entire remaining headroom** (0.0747) — against S3's
+0.0044 (5.9 %) and S1's nothing.

**And the two axes trade in opposite directions at identical cost.** Spending the
same NFE on refinement buys intelligibility (WER 0.1955 → 0.1316) while spending
it on search buys identity (ECAPA +0.0444). Same compute, same models, same
items; the currency you get depends only on how you spend it. That symmetry is
the sharpest thing this project has produced.

**Per §5 and the no-self-scoop directive, it does not go in the workshop paper.**
The symmetric-currencies result is the ICLR spine; it is recorded in
`ICLR-NOTES-v2.md` and here, and the workshop paper's outlook sentence is
governed by the template below, not by this.

---

## 2026-08-09 02:45 UTC — S4 speaker-contrastive guidance — **H-S4: REFUTED** (pre-registered, v1.2 §9)

15 C-budget runs × 400 items, T=16, two forward passes per step.
γ = 0 verified a strict no-op against the frozen v1.0 grids before any datum.

### MEASURED

| γ | ΔSIM-o | 95 % CI | runs + | ΔWER (points) | ΔUTMOS | degen |
|---|---|---|---|---|---|---|
| **0.5** | **+0.0121** | **[+0.0094, +0.0150]** | **15/15** | **+2.73** | −0.192 | 0.0008 |
| 1.0 | +0.0025 | [−0.0015, +0.0063] | 10/15 | +7.22 | −0.343 | 0.0015 |
| 2.0 | −0.0278 | [−0.0327, −0.0231] | 0/15 | +17.26 | −0.583 | 0.0077 |

The primary requires **all three** of: SIM CI > 0, WER ≤ +2.0 points, DegenRate
≤ 2× baseline. **No γ satisfies all three.** γ = 0.5 clears the SIM condition
convincingly (15/15 runs, CI well clear of zero) and the degeneracy condition, and
**misses the WER guardrail by 0.73 points**. γ = 1.0 and 2.0 fail outright.

The second lens (ECAPA confirmation) is **conditional on a primary winner** and
was therefore not applicable — recorded as not-run rather than as failed.

### Rival table *(filled)*

| rival | check | result |
|---|---|---|
| guidance trades naturalness for scorer-specific features | UTMOS(γ) − UTMOS(0); a SIM gain with UTMOS collapse > 0.5 is flagged | **−0.192 at γ=0.5**, −0.343 at 1.0, −0.583 at 2.0. Below the 0.5 flag at the only γ that gained SIM, but monotone — naturalness is being spent throughout. |
| wrong-speaker branch produces degenerate negatives | DegenRate by γ against baseline | **0.0008 / 0.0015 / 0.0077** vs baseline 0.0008. Degeneracy only becomes visible at γ=2.0, so the negative branch is not producing garbage at usable strengths. |

### INTERPRETATION *(revisable, labeled — NOT STABLE)*

**Guidance is a real identity knob that is not free, and the pre-registered bound
is what it fails.** +0.0121 at γ=0.5 is 16 % of the remaining headroom and is the
second-largest gain in Program S — but it costs 2.73 WER points, and the
pre-registration set 2.0 as the line between a win and a trade. It is a trade.

**The canary was optimistic, and that is worth recording.** On C3 seed 0 / 100
items the same γ cost **+1.7** WER points, inside the guardrail; on the full
15-run, 400-item measurement it costs **+2.73**. A 100-item single-run gate
under-estimated the cost by 60 %. Gates sized for cheapness should be read as
"proceed", never as "this will pass".

**Everything degrades monotonically in γ.** SIM, WER, UTMOS and degeneracy all
worsen from 0.5 → 2.0, so there is no larger-γ regime worth exploring; the axis is
characterised.

---

## 2026-08-09 03:10 UTC — **PROGRAM S CLOSING ENTRY: what buys speaker similarity, ranked on the ledger** (required by v1.2 §9 program-level statement)

Five pre-registered hypotheses, one gate, one instrument validation, one
correction. No single SUPPORTED verdict is glossed as "the" answer; every axis is
stated as its measured gain **relative to the S0 ledger**, with its cost, and the
rivals that remain unexcluded are listed rather than omitted.

**Remaining headroom to the codec ceiling: 0.0747 SIM-o.** Of the total gap
between real audio and the best measured system, **62 % is the codec itself**.

| rank | axis | SIM gain | % of headroom | verdict | cost |
|---|---|---|---|---|---|
| 1 | **S2 test-time search** (best-of-8, NFE 512) | **+0.0444** | **59.5 %** | **SUPPORTED** (3/3 tiers, both lenses) | +6.4 WER pts, 8× inference |
| 2 | S4 contrastive guidance (γ=0.5) | +0.0121 | 16.2 % | **REFUTED** — a trade, missing the guardrail by 0.73 pts | +2.73 WER pts, 2× forward passes |
| 3 | S3 rate-matched length | +0.0044 | 5.9 % | **DISCORDANT** (p = 0.051) | +0.86 WER pts, free |
| 4 | S1 test-time context | 0.0000 | 0.0 % | **DISCORDANT** — best arm *is* the default | — |

For scale, the non-test-time axes from the S0 ledger: training compute 30k→90k
buys **+0.0788**, a 14× parameter increase buys **+0.0531**, shape at fixed N
**+0.0325**.

**The answer to the standing question.** *Search* buys speaker similarity —
nothing else tested does, at usable cost. It buys roughly **as much as tripling
training compute**, and more than a 14× parameter increase, but it charges
intelligibility and 8× inference for the privilege. Everything else is small
(rate-matching), a trade (guidance), or nothing (context).

**The symmetry is the real finding, and it is not the workshop paper's.** At
identical NFE, refinement buys intelligibility (WER 0.1955 → 0.1316) and search
buys identity (ECAPA +0.0444). Same compute, same models, same items — the
currency depends only on how it is spent. Per §5 and the no-self-scoop directive
this is the ICLR spine and is recorded in `ICLR-NOTES-v2.md`, not spent here.

**Rivals that remain unexcluded** *(stated, per the program-level requirement)*:

1. **S1's long arms confound length with multi-utterance context** — no eval
   prompt exceeds 3.5 s, so 6/9 s had to be built by concatenating *different*
   same-speaker clips. A single long utterance was never tested.
2. **Everything is C-budget only**, one corpus, one codec, one language. No axis
   was re-tested at budget D or at 90k steps, where the baseline is stronger and
   the headroom smaller.
3. **S2's candidates all come from one model at T=8.** Best-of-K across
   checkpoints, temperatures, or seeds is untested.
4. **S2's selector and scorer share a family.** ECAPA and WavLM-SV are both
   WavLM/x-vector-lineage encoders; both pass G0(c) independently, but a
   genuinely different scorer family was not available, so shared representational
   bias is **not** excluded.
5. **S4 tested three γ values and one negative** (the fixed *i+7* wrong-speaker
   assignment). Other guidance forms and negatives are unexplored.
6. **The ceiling itself is a bound, not a truth.** It is defined by a Mimi
   roundtrip of the ground-truth *target*; a perfect system might exceed it by
   matching the prompt's channel more closely than the target recording does.
   Checked as a rival and bounded, not excluded.

**Verdict tally:** 1 SUPPORTED (H-S2), 1 REFUTED (H-S4), 2 DISCORDANT (H-S1,
H-S3), 1 classification STABLE (H-S0). Two gates passed (ECAPA §1.7, S4 canary).
One posted result corrected after its cooling pass (S0's best-system selection).

**Paper impact under §5.** (a) The codec-ceiling sentence is permitted and S0 is
STABLE, so it enters limitations. (b) The outlook sentence is **not** written:
the template makes it conditional on S1/S2, and S1 is DISCORDANT while S2 is
SUPPORTED — a mixed outcome, which the template assigns to the *no sentence*
branch. Nothing else from Program S enters the paper.

---

## 2026-08-09 01:40 UTC — S2 confound check — **the search gain survives; one headline number corrected**

Prompted by the question of whether S2 belongs in the workshop paper. The
Program-S closing entry listed the selector/scorer family overlap as an
*unexcluded* rival; this tests it instead of stating it. All 8 candidates per item
were scored with ECAPA (15 runs × 200 items × 8 = 24 000 embeddings).

### MEASURED

Selector/scorer correlation: **r = 0.7118**.

| matched NFE | random pick | WavLM-selected | ECAPA oracle | selection lift | 95 % CI | oracle gap | share of oracle |
|---|---|---|---|---|---|---|---|
| 128 | 0.4262 | 0.4400 | 0.4573 | **+0.0138** | [+0.0127, +0.0149] | +0.0311 | 44.4 % |
| 256 | 0.4261 | 0.4517 | 0.4824 | **+0.0256** | [+0.0245, +0.0268] | +0.0563 | 45.5 % |
| 512 | 0.4259 | 0.4613 | 0.5029 | **+0.0354** | [+0.0337, +0.0372] | +0.0770 | 46.0 % |

**Decomposition against the practical default** (refinement at T=16, ECAPA 0.4247):

| NFE | search vs default | refinement vs default | headline gap |
|---|---|---|---|
| 128 | +0.0153 | +0.0000 | +0.0153 |
| 256 | +0.0270 | −0.0052 | +0.0322 |
| 512 | **+0.0365** | **−0.0079** | +0.0444 |

### What this settles

**1. Selection does real work — the gain is not a T=8 sampling artifact.**
Random-pick ECAPA is **flat at ≈ 0.426 for every K** (0.4262 / 0.4261 / 0.4259),
exactly as it must be if drawing more candidates without choosing between them
buys nothing. Every bit of the search arm's advantage comes from the *choice*,
with CIs far clear of zero at all three tiers.

**2. The shared-bias rival is bounded, and the evidence points against it.**
If WavLM-SV were simply a proxy for ECAPA's idiosyncrasies, its argmax would
approach ECAPA's own argmax and it would capture near **100 %** of the oracle gap.
It captures **44–46 %**, stably across tiers — the signature of a genuinely
*imperfect* correlated instrument selecting on partially-shared speaker
information, not of two encoders agreeing on the same errors. The rival is
**bounded, not eliminated**: a scorer from a different family is still required to
exclude it outright.

**3. One reported number was inflated and is corrected.** The +0.0444 headline is
measured against refinement *at the same NFE*, and refinement's own identity
**declines** with T (0.4247 → 0.4195 → 0.4168). Against the configuration anyone
would actually deploy — refinement at T=16 — search buys **+0.0365**, which is
**49 %** of the remaining 0.0747 headroom, not 59.5 %. Roughly **18 % of the
headline gap was refinement degrading rather than search improving.** The H-S2
verdict is unaffected (it is a matched-NFE contrast, and that contrast is real);
the *ledger* figure changes.

**4. A forward-looking number.** The oracle gap **grows with K** (+0.0311 →
+0.0563 → +0.0770), so the candidate pool contains substantially more identity
than any current selector extracts. Better selection, not more candidates, is the
open lever.

### INTERPRETATION *(revisable, labeled)*

The result is stronger than it was before this check, not weaker: the mechanism is
now identified (choice, not sampling), the main rival is quantified rather than
hypothesised, and the deployable claim is stated against the right baseline.
It is still **not STABLE** — the family confound remains bounded rather than
excluded, and everything is C-budget, 30k steps, one corpus and one codec.

**Paper status unchanged.** §5 admits nothing from Program S beyond the codec
sentence, and this is the ICLR spine under the no-self-scoop directive. What has
changed is that the ICLR case is now materially better documented, with the
selection/sampling decomposition and the oracle ceiling already measured.

---

## 2026-08-10 15:10 UTC — **H-T1: SUPPORTED — search survives a stronger baseline, and claims a larger share of what is left** (pre-registered, v1.3)

The biggest unexcluded rival from Program S was that everything was measured on
C-budget models at 30k steps. This closes it with **no new training**: the ten
budget-D runs (276 M) and the three 90k runs already had checkpoints, so the
matched-NFE contrast was simply re-run there. 130 syntheses + 26 scoring jobs,
zero failures.

### MEASURED

Codec ceiling SIM_rt = 0.5554. NFE 512 (refinement T=64 vs best-of-8 at T=8):

| group | mean ECAPA @T=16 | own headroom | ΔECAPA | 95 % CI | win rate | share of own headroom |
|---|---|---|---|---|---|---|
| C-budget 30k *(reference)* | 0.4247 | 0.1306 | **+0.0444** | [+0.0416, +0.0470] | 72.3 % | 28.0 % |
| **budget-D 276 M** | 0.4413 | 0.1141 | **+0.0422** | [+0.0393, +0.0448] | 70.7 % | 32.1 % |
| **training 90k** | 0.4827 | 0.0726 | **+0.0368** | [+0.0321, +0.0428] | 68.7 % | **43.2 %** |

**Primary supported in both out-of-scope groups; second lens (per-item win rate
with CI) supported in both. Verdict: SUPPORTED, lenses agree.**

### Rival table *(filled)*

| rival | check | result |
|---|---|---|
| headroom shrinkage — a fixed gain looks better as the ceiling nears | each group's gain expressed against its **own** headroom | absolute gain shrinks modestly (+0.0444 → +0.0422 → +0.0368) but headroom shrinks **faster** (0.1306 → 0.1141 → 0.0726), so the *share* of what remains **grows**: 28 % → 32 % → **43 %** |
| selector/scorer family overlap | random / WavLM-selected / ECAPA-oracle decomposition, repeated per group | r = 0.712 / 0.722 / 0.733; share of oracle captured **44–46 % / 45–47 % / 43–46 %**. Shared bias predicts ≈100 %; the value is flat across 3× parameters and 3× training. |
| candidate-pool degeneracy | DegenRate of selected candidates by group | negligible and flat in every group (≤ 0.001) |

**Basis note.** The frozen H-S0 rule defines headroom against the *best* measured
system, but the search gain is a paired contrast **averaged over runs**. Dividing a
mean-based gain by a best-based headroom mixes bases — the S0 selection error in
miniature — so the percentages above use the mean basis throughout and the
best-basis figures are carried alongside in `t1_scope.json`. This was caught
before publication rather than after.

### INTERPRETATION *(revisable, labeled — NOT STABLE)*

**Search is not a small-model artifact.** It holds at 276 M parameters and at 3×
training compute, with overlapping CIs and win rates within 4 points of each
other. That was the single largest threat to the S2 result and it is now closed.

**The more interesting reading is the trend.** Every axis that improves the model
also shrinks the headroom, and search shrinks with it *more slowly* — so the
better the system, the larger the fraction of the remaining gap that search
recovers: **43 % at the strongest baseline we have.** If that continues, search
becomes *more* valuable as models improve, not less. Stated as a trend across
three points, not a law.

**The confound behaves identically at every scale.** A selector exploiting shared
idiosyncrasy with the scorer would have no reason to capture a stable 43–47 % of
the oracle gap across 3× parameters and 3× training. This does not replace an
independent scorer family, but it is much harder to explain as shared bias than a
single-scale measurement was.

**Still not STABLE**, and the family confound remains bounded rather than
excluded. H-T2 (is the training axis saturating?) and H-T3 (is context a training
limitation?) are running.

---

## 2026-08-11 09:20 UTC — **H-T2 SUPPORTED** (training compute is the strongest identity axis and is still paying) and **H-T3 REFUTED** (context is not a training limitation)

Preceded by a defect of mine, recorded first because it corrupted the first pass.

### DEFECT — a synth/score race silently manufactured 60–99 % phantom crashes

I put synthesis and scoring in a **single dispatcher job list**, so they ran
concurrently and scoring read directories that were still being written. Files
absent at scoring time were charged under the §10 crash policy. Result:
crash rates of 0.625–0.993 and a *nan* SIM on the 180k run — while every one of
the 400 `.flac` files existed and was valid (`it0003`: score record
`gen_seconds=0.100`, the crash placeholder; the file itself held 5.2 s of audio).

This is the P0-v1.1-1 failure mode in a new dress: the crash policy cannot tell
"never written" from "written and broken". **Fixed two ways** — the corrupted
`scores.json` were deleted and regenerated after synthesis completed (all crash
rates now 0.000), and `sample.py` now writes `synth.json` to a temp file and
`os.replace`s it, so the completion marker `score_dir` keys off can never be
observed beside a half-written directory.

### MEASURED — H-T2 (both lenses agree → SUPPORTED)

| lens | result |
|---|---|
| **primary** 90k → 180k, C3 seed 0, paired item bootstrap | SIM-o **0.4807 → 0.4962**, Δ = **+0.0155**, CI [+0.0064, +0.0243]; WER +1.40 pts (inside the ±2.0 guardrail) → **supported** |
| **second** does 30k→90k replicate over 3 seeds × 3 configs? | **+0.0827**, CI [+0.0733, +0.0908], **9/9 runs positive** → **supported** |

The single-seed ledger figure of +0.0788 was **not** a seed artifact — nine runs
give +0.0827 with a tight interval.

**Exploratory shape** (no decision rule): SIM-o at 30k / 90k / 180k =
0.4081 / 0.4807 / 0.4962. Extrapolating the 30k→90k line in log-steps predicts
0.5264 at 180k; observed is 0.4962, a residual of **−0.0302**. The axis is
**still paying but decelerating** — a further doubling bought +0.0155, about a
fifth of what the first tripling bought.

### MEASURED — H-T3 (both lenses agree → REFUTED)

C3 seed 0 retrained with per-item prompt lengths from {1.5, 3, 6, 9} s, then the
identical S1 arm sweep:

| arm | SIM-o (variable-prompt) | WER (variable-prompt) | WER (baseline C3_0) |
|---|---|---|---|
| 1.5 s | 0.3189 | 0.1326 | 0.1151 |
| **3 s** | **0.3899** | 0.1185 | 0.1115 |
| 6 s | 0.3747 | 0.7630 | 0.7201 |
| 9 s | 0.3694 | 0.8491 | 0.8080 |

Primary: SIM-o(9 s) − SIM-o(3 s) = **−0.0206**, CI [−0.0312, −0.0098], with WER
**+73.05 points** → not supported on either clause. Second lens: Spearman
ρ = +0.200 but the curve is **non-monotone** (peak still at 3 s) → not supported.
Requiring monotonicity here was pre-registered precisely because non-monotonicity
is what made H-S1 DISCORDANT.

**Rivals.** "The retrained model is simply better": it is **worse** at the 3 s arm
(−0.0119), so no. "Content-only fix": it moved neither — WER at 9 s is *higher*
than the baseline's (0.8491 vs 0.8080).

**Scope limit, stated because it bounds the conclusion.** Only 36 % of training
clips are long enough for a 9 s prompt plus a 1 s target (62 % for 6 s), so each
draw was capped at (clip frames − 13) and the realised prompt distribution was
**skewed short, not uniform** as the pre-registration wording assumed. The model
therefore saw comparatively few genuinely long prompts, at 30k steps.

### INTERPRETATION *(revisable, labeled)*

**Training compute is the best identity lever measured, and it is not exhausted.**
+0.0827 for 3× compute, a further +0.0155 for 2× more, against the best test-time
intervention's +0.0365. Decelerating, but still the largest single axis.

**Context is not simply a training-data limitation — at least not one this
experiment can fix.** The natural story after S1 was "the model only saw 3 s
prompts, so teach it longer ones." Trained that way, it still collapses at 6–9 s
and is slightly *worse* everywhere. Either the capped, short-skewed distribution
was too weak to teach the behaviour, or long-prompt conditioning fails for a
reason that more of the same data does not address. The honest reading is that
this attempt failed, not that the question is closed — and the distinction is
recorded rather than glossed.

---

## 2026-08-11 12:05 UTC — Cooling pass on the promoted search claim: **numbers all correct, several wordings were not** (two blockers, both fixed)

§5 was overridden by human decision and the search result went into the paper.
The §1.4 cooling rule still applies to paper edits, so an independent pass ran
before anything is tagged. It regenerated `numbers.tex` **byte-identically** and
re-derived every contrast from the raw per-item CSVs (200 items × 15/10/3 runs;
24 000 candidate rows). **All 15 macros are correct.** The defects were all in
prose.

### Blocker 1 — the abstract quoted a matched-NFE contrast with no baseline named

+0.0444 is genuinely the NFE-512 contrast, but refinement's own identity *declines*
with T (0.4247 → 0.4195 → 0.4168), so **17.8 % of it is the baseline degrading**.
Against the deployable T=16 default search buys **+0.0365**. This project had
already found and published that correction on 2026-08-09, and `src/paper.py`
already *generates* `\NsearchVsDefault = +0.0365` — which main.tex then **never
used**. A correction that lives only in the feed is not a correction. Both the
abstract and the Results paragraph now quote both figures with their baselines
named.

### Blocker 2 — the headroom percentages were a cross-scorer subtraction

"28 %, 32 %, 43 % of remaining headroom" divided an **ECAPA** numerator by a
headroom whose ceiling came from `identity_ledger.json`'s round-trip, which is
**WavLM**-scored. The two encoders are not on a common scale — on the same
synthesised audio ECAPA reads 0.4247 where WavLM reads 0.3715. Recomputed
consistently in WavLM the same shares are **44.9 / 50.8 / 77.0 %**. The *ordering*
survives on both bases; the *levels* do not, and no ECAPA round-trip ceiling exists
anywhere, so a scorer-consistent level cannot be computed at all. **The three
percentages are removed**; the paper now claims only the ordering, and says why the
levels are omitted. This is the same class as the T\* scale artifact — a quantity
compared across incommensurable scales — and I introduced it.

### Other wording fixes, all from the same pass

| was | now | why |
|---|---|---|
| "at matched **cost**" | "at matched **NFE**" | NFE covers the diffusion transformer only; search also pays K codec decodes and K selector passes. Now stated in Limitations. |
| "the **entire** gain comes from the choice" | "drawing more samples without choosing buys nothing" | selection is 79.7 % of the headline at that basis (96.9 % against the T=16 default), not all of it |
| "scored by a **different**, independently validated one" | "a second, independently gated one (see Limitations)" | the selector is an ECAPA-TDNN head on WavLM-large features; the scorer is a standalone ECAPA-TDNN. They share the ECAPA-TDNN head, so Results was contradicting Limitations. |
| "both **WavLM/x-vector**-lineage encoders" | "both **ECAPA-TDNN-family** (selector on WavLM-large features)" | the scorer has no WavLM front-end; the shared component is the head |
| "$r = 0.71$" | "$r = 0.71$–$0.73$" | 0.71 is the *minimum* of the three measured groups — quoting the min inside a sentence whose job is to bound the confound is favourable selection |
| "at 3× training compute" | "on three runs, at 3× training compute" | n = 3; and that group fails the second lens at NFE 128 |
| "276 M parameters" | "276 M **non-embedding** parameters" | consistency with the rest of the paper |
| *(absent)* | "search buys +0.0365 for **+0.0425 WER**" | the adoption cost a reviewer will ask for |
| "not a limit on what inference can buy" | "**less** a limit … than" | absolute phrasing the numbers do not license |

**Verified after the edits:** compiles with no errors, main text still closes on
**page 4**, retired-phrase grep clean, anonymised, 118 generated macros and no
hand-typed number.

**Interpretation.** The pattern across this project holds: the arithmetic survives
checking, the *framing* is where things break. Two of these — an uncarried
correction and a cross-scale denominator — are repeats of errors already made and
already logged, which is the more useful lesson than either individual fix.

---

## v1.4 — the primary result was coordinate-bound (2026-08-11)

**MEASURED.** Refitting the identical 225-row v1.0 surface after a monotone change of
the error variable, with the same code, weights and multi-starts:

| coordinate | tau_WER | tau_SIM | Delta-tau |
|---|---|---|---|
| identity (pre-registered) | 0.8362 | 0.7260 | **+0.1102** |
| log(1+err) | 0.6665 | 0.6913 | −0.0248 |
| sqrt(err) | 0.4325 | 0.6842 | −0.2517 |
| err capped at 1 | 0.7093 | 0.7260 | −0.0167 |
| err squared | 1.5613 | 0.8122 | +0.7491 |

**INTERPRETATION.** tau is invariant to *affine* rescalings of the error, which v1.0
verified. It is not invariant to monotone non-affine ones, and Delta-tau's sign is not
either. Delta-tau describes the coordinate, not the systems. H-D2 stands as a
pre-registered test in its declared coordinate; it no longer carries the paper.

**MEASURED — the extended surface.** All 45 runs extended to T=64 at the full 400 items
(315 rows, balanced; `artifacts-v1.4/runs_extended.csv`). The previous extended-T fit used
7 of 15 configs at 200 of 400 items, so it confounded range with config and item subset.
Refitting on nested windows: Delta-tau = +0.2191 (T<=8), +0.1102 (T<=16), +0.0603 (T<=32),
+0.0660 (T<=64). It roughly halves each time the window doubles.

**MEASURED — the replacement.** Fraction of the *reachable* range closed, referenced to
measured floors (ASR word error on the real recordings 0.0345; codec round-trip similarity
0.5554), model-free:

| T | intelligibility | identity | ratio |
|---|---|---|---|
| 4 | 69.8% | 35.7% | 1.96x |
| 16 | 86.2% [82.8, 89.1] | 46.4% [45.0, 47.8] | 1.86x [1.82, 1.90] |
| 64 | 88.4% | 47.9% | 1.85x |

**INTERPRETATION.** The ratio is stable in T (1.85–1.96x over a 16x range of budget, where
Delta-tau halves) and stable in the coordinate (1.68–1.86x under the maps that flip
Delta-tau's sign), because it is a ratio of differences on one axis. Identity plateaus near
48% of its reachable range: quadrupling refinement past T=16 buys 1.5 points.

**MEASURED — the scorer-lineage rival, closed.** Five encoders, four families, 15 runs x 200
items. Per-item win rate for search over refinement at matched NFE: ECAPA 72.3% [70.6, 73.8],
x-vector 79.0% [77.5, 80.5], GE2E 69.1% [67.5, 70.8], WavLM-base+ 64.6% [62.9, 66.3],
selector WavLM-large 84.9% (circular). 4/4 non-selector families above one half.

**INTERPRETATION.** Gate G0(c) rejects three of these on absolute cosine *scale* despite
AUC >= 0.983, so the gate is scale-dependent. A win rate is invariant to monotone rescaling
of an encoder's cosine; a difference of means is not. Agreement is therefore read from win
rates, and raw deltas are reported only beside their own scale.

**MEASURED — three further scope conditions.** chi^2/dof = 14.3 (WER) and 8.8 (SIM-o), so
the declared functional form is rejected by the criterion it minimises. The declared
Delta-tau CI [0.0921, 0.1312] widens to [0.0349, 0.1640] when the 1/SE^2 weights are
resampled with the runs instead of pinned. The separable fit implies a local exchange rate
kappa = tau/beta = 0.41, so "no exchange rate exists" was an overreach; what fails is a
*global* rate.

**PROCESS.** `src/check_claims.py` now enforces retractions mechanically. On first run it
found nine occurrences, three of them live assertions — including index.html carrying a
withdrawn thesis as its page title.

---

## v1.4 addendum — X2 (3x training compute) and X7 (T=128) (2026-08-11)

**MEASURED — X2.** Nine runs (C1/C3/C5 x three seeds) retrained to 90k steps, swept over
the same T grid at the same 400 items, against the same floors (they are properties of the
recordings and of Mimi, not of the model). Fraction of the reachable range closed:

| T | intelligibility | identity | ratio | (30k ratio) |
|---|---|---|---|---|
| 4 | 87.8% | 61.4% | 1.43x | 1.96x |
| 16 | 97.1% [96.4, 97.7] | 71.5% [69.6, 73.2] | 1.36x | 1.86x |
| 64 | 98.2% | 73.6% | 1.33x | 1.85x |

**INTERPRETATION.** The direction is robust — refinement closes more of intelligibility's
reachable range than of identity's at every T and at both training budgets — but the
*magnitude* is not a constant of the architecture. Under-training inflates it. The claim
"identity is stuck near half of its reachable range" is true of the 30k grid and false of
the 90k one (71.5%). The paper now offers the ordering, not the factor, as the finding, and
says so in the abstract.

**MEASURED — Delta-tau at 90k.** +0.2060 (T<=16), +0.2493 (T<=32), +0.2689 (T<=64).

**INTERPRETATION.** On the 30k grid Delta-tau *falls* as the window widens (+0.1102 ->
+0.0660); at 90k it *rises*. Its range-dependence does not even keep direction across
training budgets. This is further reason not to carry a claim on it.

**MEASURED — X7.** C5 (largest budget, three seeds) at T=128 against T=64: WER +0.0018,
SIM-o -0.0052.

**INTERPRETATION.** Past T=64 the refinement axis has stopped paying and begun, slightly,
to cost. The plateau is a plateau, not a pause before another descent.

**PROCESS.** First attempt hardcoded width/depth per config from memory; the values were
wrong and would have silently changed N = 12*d*w^2 and every fitted exponent. Shapes are
now read from each run's own run.json.

---

## v1.5 CORRECTION — the guidance arm was credited a 2x compute advantage (2026-08-12)

Found while designing the CFG experiment, by agents reading the sampler rather than
the analysis. Three defects, all in code that produced numbers **currently in the
paper** (the H-S4 guidance row of the identity ledger, and Table 7).

**DEFECT 1 — NFE under-recorded for guided runs.** `sample.py` wrote
`"nfe": 8*T` unconditionally. Guidance runs a *second* forward pass per step, so a
guided T=16 arm executes 256 forwards while `runs/C1_0/synth_gam1p0/synth.json`
recorded `nfe: 128`. VERIFIED by reading the file.

**DEFECT 2 — the S4 baseline was iso-T, not iso-NFE.** `s4_analysis.py:63` used
`base = load(r, "T16")`, comparing a 256-forward guided arm against a 128-forward
baseline. Every published S4 number therefore gave guidance twice the compute of its
reference.

**MEASURED, recomputed against unguided T=32 (iso-NFE, exists for all 15 runs,
shares the RNG stream, zero new synthesis):**

| arm | as published (vs T16) | iso-NFE (vs T32) |
|---|---|---|
| γ=0.5 | ΔSIM +0.0121, ΔWER +0.0273 | ΔSIM **+0.0083**, ΔWER **+0.0441** |
| γ=1.0 | ΔSIM +0.0025 | ΔSIM **−0.0014** (4/15 runs positive) |

**INTERPRETATION.** H-S4's verdict is unchanged — guidance is a trade, not a gain —
but it is a worse trade than published: at honest compute accounting the γ=0.5 gain
falls 31% and its WER cost rises 61%, and the γ=1.0 gain reverses sign. The identity
ledger's ordering (training compute > search > guidance > rate-matching > context) is
unaffected.

**DEFECT 3 — attended PAD in the contrastive branch.** In `synth_batch`, when the
wrong-speaker partner's prompt is shorter than the item's own, frames [n, n_prompt)
kept `PAD_ID` while `fmask` stayed True, and `cur_w[~fmask] = PAD_ID` did not clean
them. The model attended to PAD, a state never present in training. VERIFIED
independently: **175/400 items (43.8%), mean 2.59 frames, max 6**. Every existing γ
arm is contaminated, so the numbers above are computed from buggy samples and the 45
arms are being re-synthesised.

**FIXES.** `sample.py` now records forwards actually executed
(`nfe`, `nfe_unguided`, `forward_passes_per_step`) and marks the leaked frames
unattended via a separate `fmask_w`; `s4_analysis.py` takes `BASE_TAG = "T32"` with
the iso-T contrast retained as a labelled secondary.

**PROCESS.** These survived because NFE matching was asserted in prose
("matched NFE, 8T per candidate") and never checked against what the sampler wrote.
An assertion in a caption is not a test.

**PREREGISTRATION-v1.5.md committed at `860a95818b2f408ec3ee7fac37c39728166317b9`,
2026-08-12 03:56:59 +0000, before any v1.5 CFG run was launched.** The CFG inclusion
gate, the primary arm, the gamma-selection split (first 100 items by sorted id, reported
on the disjoint 300) and the iso-NFE baseline are all fixed as of that commit. The gate
contains a REFUTATION branch that forces a CFG result into the paper even when it
damages the finding.

---

## v1.5 WITHDRAWAL — H-T3's verdict was vacuous: `--variable-prompt` was a no-op (2026-08-12)

**MEASURED (by reading, then confirmed by grep).** `train.py` computed the per-item
sampled prompt length `PF` at line 411 under `--variable-prompt`, and then passed the
module constant `PROMPT_FRAMES` to `build_inputs` at lines 419/423. `PF` appeared
nowhere else in the file. The flag therefore changed nothing about the training
inputs; it only consumed draws from the shared generator, so the run differs from a
standard run the way a different seed does. `run.json` did not record the flag either,
so there was no provenance to catch it.

**WITHDRAWN.** H-T3 was recorded as *"refuted — variable-prompt training does not
repair it"* (DECISION-v1.2.md, and this feed on 2026-08-11). The model never saw a
variable-length prompt, so the comparison was standard-training vs standard-training.
The correct status of "is context a training limitation?" is **UNTESTED**, not refuted.
An untested question and a refuted hypothesis are not the same claim, and the second is
the more useful one to have, which is exactly why it must not be kept by accident.

**NOT IN THE PAPER.** Checked: `paper/main.tex` never cites H-T3. Its context claims
come from the S1 inference-time arms (H-S1, recorded discordant) and are unaffected.
The withdrawal touches project records only.

**FIXED.** `build_inputs`/`build_inputs_flat` now receive `PF`; `run.json` records
`variable_prompt` and `cond_dropout`. Standard training is unaffected — with the flag
off, `PF is PROMPT_FRAMES`, so the call is identical. H-T3 is re-queued to run properly
once the 180k training frees the GPUs.

**PROCESS.** Three of the four defects found today (guided NFE, the S4 baseline, this
one) share a shape: a quantity was computed, and then the code went on to use something
else. None was caught by a test because nothing asserted the intended relationship. The
new `src/test_cond_dropout.py` asserts four such relationships for the dropout, including
that a dropped condition is both PAD *and* unattended — the precise failure that produced
defect 3.

---

## v1.5 DEFECT — the scorer silently substituted an excluded encoder (2026-08-12)

**What happened.** A batch re-scoring job produced an apparent speaker-similarity gain of
**+0.517** for speaker-contrastive guidance. That is not a plausible number: SIM-o against
the prompt sits near 0.39, and the measured codec ceiling is 0.5554, so +0.52 would place
generated audio above what a codec round-trip of genuine same-speaker audio achieves.

**Root cause.** `Scorer._load_sv` wrapped the primary model load in a bare
`except Exception` and fell back to `microsoft/wavlm-base-plus-sv`, printing one line.
An HF Hub rate-limit made the primary load fail, and the fallback engaged unnoticed.
base-plus-sv **fails gate G0(c)** — same-speaker median 0.9488, cross-speaker 0.6601 —
so its cosines sit high and compressed and every arm scores ~0.90. Contrasting arms scored
that way against a baseline scored correctly manufactures the +0.5.

**Diagnosis that settled it.** Re-embedding the *same audio files* with the correct
encoder gave gamma=0.5 → 0.3853 and unguided T=32 → 0.3855, a difference of **-0.0001**.
The audio was never wrong; only the scores were. Unguided arms were also confirmed
bit-identical to their pre-edit values, so the frozen sampler is intact.

**Blast radius.** 59 of 655 scored directories: all 45 gamma arms and 14 of the C1 180k
sweep, every one of them written today. The other 596 — the whole v1.0/v1.1/v1.2/v1.4
grid and every number currently in the paper — use the correct encoder and are unaffected.
All 59 are being re-scored.

**FIXED.** The fallback is now reachable only when explicitly requested. An unexpected
failure raises with the reason and names the gate the fallback fails. Verified by making
the primary model unloadable and confirming a RuntimeError rather than a substitution.

**PROCESS.** This is the fourth defect of the same family in two days: a value was computed
one way and consumed as though it had been computed another. It was caught only because
the number was physically impossible against a floor we had already measured. Without the
codec ceiling on record, +0.517 would have looked like the session's best result.

---

## v1.5 — external anchor: F5-TTS v1 Base on our 400 items (2026-08-13)

**MEASURED.** F5-TTS v1 Base, synthesised in an isolated venv and scored with our frozen
stack on the identical 400 cross-sentence items: **WER 0.0248, SIM-o 0.6549, UTMOS 3.535,
400/400 rendered.** Passes the pre-declared verification band (WER <= 0.15, SIM-o >= 0.40,
>= 396 rendered), so the port is sound and the metric stack is calibrated.

**Two of our assumed bounds are not bounds.**

| quantity | ours | F5-TTS |
|---|---|---|
| ASR floor (word error on the REAL recordings) | 0.0345 | **0.0248** |
| Mimi codec ceiling on SIM-o | 0.5554 | **0.6549** |

**INTERPRETATION.** F5 is *more* intelligible to Whisper than the original recordings, so
"word error on real audio" is a floor for real audio, not an achievable minimum — read-aloud
synthesis is cleaner than spontaneous speech. And F5 exceeds the Mimi round-trip ceiling by
+0.0996 because it does not use Mimi. Our ceiling is a property of **our codec**, not of the
task. That is the correct bound for our own models, which emit Mimi tokens and cannot exceed
what Mimi represents, but the paper must say *whose* ceiling it is rather than let "the
reachable range" read as task-level.

**Effect on the headline, computed both ways:**

| bound used | intelligibility | identity | ratio |
|---|---|---|---|
| real audio / Mimi round trip (as published) | 86.2% | 46.4% | 1.86x |
| F5 as the empirical achievable bound | 85.5% | 36.0% | **2.37x** |

The asymmetry gets **larger**, not smaller, under the architecture-independent bound, so the
finding is sharpened rather than threatened. It also converts "62% of the identity gap is the
codec's" from an inference into a demonstration: a system with a different codec clears our
ceiling by 0.10.

**Cost.** 9 minutes of synthesis against a 3 h fence. The isolation guard confirmed the
project venv unchanged (transformers 5.14.1, numpy 2.4.6) after installing F5's stack.
