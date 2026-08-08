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
