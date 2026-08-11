# ASPECT-D — final decision record (v1.2 / v1.3)

Companion to `DECISION.md` (v1.0). Records what the paper claims, what it does
not, and which questions were resolved which way across v1.1, Program S and the
training extensions. Everything here is traceable to `RESULTS-FEED.md` (append-only,
corrections are new entries) and the artifacts under `artifacts/`, `artifacts-v1.1/`,
`artifacts-v1.2/`, `artifacts-v1.3/`.

---

## 1. What the paper claims

**Title.** *Test-Time Refinement Moves Intelligibility Six Times More Than
Identity in Masked-Diffusion TTS*

**Primary (pre-registered, v1.0 H-D2, outcome class S1).**
Test-time scaling is metric-selective: Δτ = τ_WER − τ_SIM = **+0.1102**, 95 %
run-level bootstrap CI **[0.0921, 0.1312]**, excluding zero.

**The magnitude claim that carries the paper.** Over T = 1 → 16, refinement moves
**0.952** absolute WER (6.10× on the error scale) against **0.161** absolute
speaker-similarity error (1.25×) — a **5.9×** asymmetry.

**Rate honesty, stated in the abstract.** On each metric's own normalised curve
convergence is nearly metric-agnostic (fraction of total gain at T=8: **94.6 %**
WER, **93.7 %** identity, identity marginally *later*). The asymmetry is in
magnitudes, not rates. This sentence exists because the original gloss — "steps
buy intelligibility, not identity" — reads as a rate claim and is backwards as
one.

**Scale persistence.** Four budgets, 20 M–276 M: Δτ = **+0.1095**, CI
[0.0951, 0.1241].

**Second currency (promoted by explicit human decision overriding task-v2.md §5).**
At matched NFE, best-of-K search improves speaker similarity by **+0.0444**
(CI [0.0416, 0.0470], 72 % of items) against refinement at T=64, or **+0.0365**
against the deployable T=16 default, for **+0.0425** WER. Refinement remains the
better spend for intelligibility (WER 0.1316 vs 0.1955). The effect is selection,
not sampling: an arbitrary candidate scores 0.4259 regardless of K. It persists at
276 M non-embedding parameters (+0.0422) and, on three runs, at 3× training
compute (+0.0368).

**Limitations stated in the paper.** The codec accounts for **62 %** of the gap
between real audio and the best measured system. Selector and scorer are both
ECAPA-TDNN-family encoders (r = 0.71–0.73); each passes the ground-truth gate
independently and the selector captures only ~46 % of the oracle gap, which
*bounds* representational overlap without excluding it. NFE matching covers the
diffusion transformer only. Target length comes from a corpus-median
seconds-per-character.

---

## 2. What the paper does not claim, and why

| withdrawn / excluded | reason |
|---|---|
| "steps rent depth, not width" | reads as a rate claim; Δτ > 0 means WER converges *sooner* (T90 15.7 vs 23.8) |
| "T\*_SIM = 8 vs T\*_WER = 32, a 4× saturation gap" | raw-scale artifact: the 5 % band is 0.67 % of WER's range but 11.96 % of SIM's (17.9×). Affine-invariant T\* = **16 for both** |
| a depth-first latency allocation rule | H-E2 refuted (5.9 % measured / 8.8 % fitted vs an 80 % bar) |
| a "steps-first" allocation rule | boundary artifact of T ≤ 16; the ordering reverses on the extended grid |
| any d\*(N) trend | B and C censored at their deepest tested shape; D's minimum is a 0.0012 near-tie |
| κ, the depth–step exchange rate | H-D3 not supported (ΔAICc +69.3 / +71.3 against a ≤ +4 bar) |
| "τ_SIM under base-plus-sv is a compression artifact" | τ is affine-invariant; the model is excluded because it **fails G0(c)** (cross-speaker median 0.6601 vs a ≤ 0.25 bar) |
| headroom-share percentages for search | cross-scorer subtraction (ECAPA numerator, WavLM ceiling). Ordering survives; levels do not |

---

## 3. Hypothesis ledger

| id | claim | verdict |
|---|---|---|
| H-D1 | fixed-T anisotropy Δρ > 0 | not supported (α unidentified) |
| **H-D2** | **Δτ > 0 — metric-selective test-time scaling** | **SUPPORTED → outcome class S1** |
| H-D3 | depth–step exchange rate κ | not supported |
| H-D4 | small-budget fits extrapolate | not supported |
| H-E1 | extended-T: T\*_WER > 16 **and** τ inside the v1.0 CI | partial (a ✓, b ✗) |
| H-E2 | iso-latency allocation is depth-first | refuted |
| **H-E3** | **coarse-level NFE allocation beats fine** | **SUPPORTED** (ΔWER −0.9289) |
| **H-E4** | **scale persistence across four budgets** | **SUPPORTED** (primary); secondary refuted at MAPE 15.1 % vs 15 % |
| **H-E5** | **Δτ survives 3× training compute** | **SUPPORTED** (+0.1767) |
| H-S0 | codec-ceiling classification | **moderate headroom** (h = +0.0747), STABLE |
| H-S1 | prompt context buys identity | DISCORDANT — best arm *is* the default |
| **H-S2** | **search buys identity where refinement cannot** | **SUPPORTED** (3/3 tiers, both lenses) |
| H-S3 | rate-matched length buys identity | DISCORDANT (p = 0.051) |
| H-S4 | training-free contrastive guidance | refuted — a trade (+0.0121 SIM for +2.73 WER pts) |
| **H-T1** | **search survives a stronger baseline** | **SUPPORTED** (276 M and 90k) |
| **H-T2** | **training-compute identity axis not saturating** | **SUPPORTED** (+0.0155 for 2× more; +0.0827 over 9 runs) |
| H-T3 | context is a training limitation | refuted — variable-prompt training does not repair it |

**Identity ledger**, absolute SIM-o gain per axis, against 0.0747 of remaining
headroom: training compute 30k→90k **+0.0827** (9 runs) > search **+0.0365** >
guidance +0.0121 (a trade) > rate-matching +0.0044 > context 0.0000.
**Training compute remains the largest identity lever measured.**

---

## 4. Corrections made during the work

Nine, all recorded as new append-only entries rather than edits. The three that
changed a published number:

1. **The "4× saturation gap"** was promoted to a headline, then retracted as a
   scale artifact. Affine-invariant T\* is 16 for both metrics.
2. **S0's best-system selection** excluded the 90k runs, halving the headroom
   (0.1423 → 0.0747) and moving the codec share from 46 % to **62 %**. The
   script's own ledger was already reading those scores.
3. **The search headroom percentages** were a cross-scorer subtraction; removed.

Two process defects worth carrying forward: a synth/score race manufactured
60–99 % phantom crashes (fixed by publishing `synth.json` atomically, so the
completion marker cannot be seen beside a half-written directory), and 37
retracted claims were found still live across seven files — including one
*rebuilt* inside the retraction page itself. **Retracting a claim in one file is
not retracting it.**

---

## 5. Submission readiness

| check | status |
|---|---|
| compiles, no errors | ✅ |
| main text ≤ 4 pages | ✅ closes on page 4 (References begin p. 5) |
| anonymised | ✅ |
| numbers are generated macros | ✅ 118, zero hand-typed |
| retired-phrase grep clean in `paper/` | ✅ (excluding the deliberately frozen v1.0 paper) |
| v1.0 immutables untouched | ✅ `git diff` vs `v1.0-submission-candidate` empty for `artifacts/`, `PREREGISTRATION.md`, `LOG.md` |
| GenAI disclosure present | ✅ |

**Ready to submit.** Target: DiffuLM @ NeurIPS 2026, 4-page non-archival,
Aug 29 AoE. Paper freeze Aug 24.

---

## 6. What remains for the full-length paper

The workshop paper now spends the symmetric-currencies result, so the ICLR case
must stand on more than it. In priority order:

1. **Break the scorer-family confound** — an encoder outside the ECAPA-TDNN
   lineage. This is the single open rival and a reviewer will name it.
2. **Quantify the exchange rate** between refinement and search rather than
   reporting two endpoints.
3. **Chase the oracle gap**, which *grows* with K (+0.0311 → +0.0770): the
   candidate pool holds far more identity than any current selector extracts, so
   better selection — not more candidates — is the lever.
4. **S5** (CFG with condition dropout), post-freeze only, never cited here.
5. **Re-test the axes at budget D and 90k**, where headroom is smaller and the
   trend in §1 predicts search matters *more*.
