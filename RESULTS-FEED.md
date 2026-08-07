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
