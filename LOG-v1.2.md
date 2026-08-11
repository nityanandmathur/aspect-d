# ASPECT-D v1.2 — decision log

Append-only. Companion to `LOG.md` (v1.0) and `LOG-v1.1.md` (v1.1). Verdict
discipline per `task-v2.md` §1 applies to every entry from Program S onward.

## 2026-08-08 14:10 — PHASE M: paper finalized against task-v2.md §2/§3

**Four blocking edits (DECISION.md "Not ready to submit") executed.**

1. *Cross-metric T\* saturation claim removed and its macros neutralised.*
   `src/paper.py` no longer emits `Ntstar<METRIC>`; it emits `NtstarRaw<METRIC>`,
   whose name states the scale, and the abstract/results/caption uses are gone.
   `Nlatencysaving` (= 100·(1−T\*_SIM/16) = 50 %, pure artifact of the retracted
   T\*_SIM = 8) is no longer generated.
2. *Interior-optimum "deep enough" rule removed.* `Nbestdepths` now marks each
   budget `(interior)` or `(censored)` by testing whether the argmin equals the
   deepest shape tested; B and C are censored, so the paper states the optimum is
   interior only in the smallest budget.
3. *Depth-first allocation rule removed* — H-E2 refutes it (5.9 % measured /
   8.8 % fitted against an 80 % bar). Replaced with the honest statement that a
   latency budget constrains only the product $T\,d$ and these data do not
   resolve the split in either direction.
4. *`step_curves` regenerated without raw-scale per-metric T\* markers.*
   `src/figures.py` no longer draws them, with the reason inline.

**Immutability choice, logged as a deviation-avoidance.** §3.5 forbids editing
`artifacts/`. Rather than overwrite the v1.0 figure, the corrected figure is
written to `artifacts-v1.2/figures/step_curves.{svg,pdf}` and the paper points
there. `artifacts/figures/step_curves.svg` still carries its two dashed markers
and is untouched, as the v1.0 record should be.

**§2 approved wording applied**, with every number a generated macro (97 total;
hand-typed numbers remain forbidden). The abstract now carries, in order: the
pre-registered detection statistic Δτ = +0.1102 CI [0.0921, 0.1312]; the
magnitude claim (0.952 absolute WER vs 0.161 absolute identity error, 5.9×);
the rate-honesty sentence (fraction of total gain at T=8: 94.6 % WER vs 93.7 %
identity, identity marginally later); and the four-budget persistence
Δτ = +0.1095 CI [0.0951, 0.1241]. The fraction-at-T=8 macros are computed
**per run then averaged**, which is what reproduces §2's stated 94.6/93.7.

**v1.1 strengtheners integrated within the 4-page law:** E4 and E6 and E5 in one
compressed passage of the H-D2 result, E1's extended-T fact with the scale named,
E3 allocation as one main-text sentence plus an appendix table carrying its
honest SIM/UTMOS cost, and the speaking-rate limitation sentence. The
significance-based 2× T\* contrast is in the appendix with its precision-floor
caveat and an explicit note that the raw-scale T\* must never be compared across
metrics.

**Verified:** compiles with no errors; References begin on page 5, so main text
is exactly 4 pages; anonymised; the §3.6 retired-phrase grep returns zero in
`paper/` excluding `main-v1-frozen.tex`, which is the deliberately preserved v1.0
paper and necessarily still contains the old title.

**Deviation to flag (pre-existing, not introduced here).** Before `task-v2.md`
existed, a cross-document audit found retracted claims live in `results.html`,
`index.html` and `protocol.html`, and I corrected them (19 insertions,
13 deletions). §3.5 now lists those files as frozen record. The corrections are
retained rather than reverted, because reverting would reinstate withdrawn claims
in published pages; they are additive outcome annotations and errata, not
rewrites of any measured value. No further edits to those files were made in
Phase M, and none will be.

## 2026-08-08 16:10 — S0 complete; S1 data prep, with two documented deviations

**S0 (H-S0): MODERATE HEADROOM**, both lenses agreeing to 0.0001 (mean +0.1423,
per-item paired median +0.1422). Both pre-registered rivals excluded with numbers.
The classification sits 0.0077 under the "large headroom" boundary and is reported
as a boundary case. Codec-ceiling number marked **NOT STABLE** pending the §1.4
adversarial pass, since §5a makes it paper-bound.

## S1 deviations (both structural, both logged before any S1 datum was scored)

**1. Longer prompts cannot come from the source utterance.** §4-S1 says to rebuild
{1.5, 3, 6, 9} s "from each eval item's source utterance". In this corpus
`eval_audio/<prompt_id>.flac` already *is* the untrimmed source clip — 
`stage_eval_audio` writes whole clips — and selection picked clips of 3.02–3.50 s.
No eval prompt contains 6 s or 9 s to cut. The 6/9 s arms therefore concatenate
*other utterances by the same speaker*, which had to be built from scratch (eval
speakers are held out, so their non-eval clips were never tokenized): raw tar →
mono → 24 kHz → peak-normalise → Mimi encode → espeak phonemes, 294 clips. Every
eval prompt_id and target_id in the whole eval set is excluded from context, so no
item can leak another's prompt or target. Result: **397/400 items carry all four
arms**, comfortably above the ≥200 threshold, so the 9 s arm is kept.

**2. "Matched RNG across arms" is not attainable with the frozen sampler.** The
Gumbel draw is shaped `[B, Fmax, V]` with `Fmax = max(n_prompt + n_target)`, so a
longer prompt changes the noise tensor's shape and each arm consumes a different
stream — the same `[B, Fmax, V]` dependence that produced the E-AUDIT smoke-test
trap. Making the noise target-relative would match the arms but would break
bit-reproduction of the v1.0 grids. The sampler stays frozen: arms use independent
noise rather than common random numbers, which is unbiased for the paired contrast
and costs variance that the 15-run × 397-item paired design absorbs.

**Error caught in my own verification.** The first prep truncated the 3 s arm to
exactly 38 frames and then "verified" it against that same truncation — a
self-fulfilling check that reported 400/400. Against the *real* v1.0 prompt it was
88/397. The 3 s arm is now the canonical prompt used unchanged, and the check
compares against `store.get(prompt_id)`: **tokens 397/397, phonemes 397/397**.
A verification that constructs its own reference is not a verification.

**Scoring alignment.** S1 drops 3 items from the *middle* of the canonical list, so
`all_items[:397]` would have scored a different set than was synthesised.
`sample.py` now records `item_ids` in `synth.json` and `score_dir` aligns to that
list when present.

## 2026-08-09 03:30 — Program S complete

Verdicts: H-S0 MODERATE HEADROOM (STABLE), H-S1 DISCORDANT, H-S2 SUPPORTED,
H-S3 DISCORDANT, H-S4 REFUTED. Gates: ECAPA §1.7 PASS, S4 canary PASS at γ=0.5.

**The verdict discipline earned its keep three times.**
1. The §1.4 cooling pass caught S0's best-system selection excluding the 90k runs —
   headroom halved (0.1423 → 0.0747), codec share 46 % → 62 %, and the artifact had
   been contradicting its own ledger row. Label survived; numbers did not.
2. Two DISCORDANT verdicts (S1, S3) that a single-lens design would have posted as
   clean results. S1's second lens reads +15/15 purely because the shortest arm is
   starved; S3's lenses split at p = 0.051 because the effect is real and tiny.
   Both diagnosed, neither resolved toward the exciting reading.
3. S4's canary under-estimated the WER cost by 60 % (+1.7 pts on 100 items from one
   run vs +2.73 on the full 15×400). Cheap gates say "proceed", not "this will pass".

**Paper additions under §5.** (a) The codec-ceiling limitation sentence is in, as
generated macros (\NsimRoundtrip, \NsimRealAudio, \NcodecShare); main text still
closes on page 4 and compiles clean. (b) The outlook sentence is deliberately NOT
written: the §5 template makes it conditional on S1/S2, and S1 DISCORDANT with S2
SUPPORTED is a mixed outcome, which the template assigns to the no-sentence branch.

**The symmetric-currencies result stays out of the workshop paper** per §5 and the
no-self-scoop directive, and is written up in `ICLR-NOTES-v2.md` with the four
things that must be done before it is publishable — chief among them breaking the
scorer-family confound, since ECAPA and WavLM-SV share a lineage.

E7 remains gated to Aug 16, spec unchanged. S5 is post-freeze only.

## 2026-08-11 10:40 — §5 overridden by human decision: the search result enters the paper

task-v2.md §5 reserved the test-time-search result for a later full-length paper
("the symmetric-currencies thesis, if supported, is the ICLR spine — spending it as
a workshop appendix is forbidden"). **The human who wrote that rule has revoked it**
and asked for the result in the workshop paper on the grounds that it makes the
paper stronger. Recorded here as an explicit, authorised deviation, not a drift.

**What went in** (all numbers generated macros; hand-typed numbers remain
forbidden): a Results paragraph giving the matched-NFE contrast at NFE 512
(ΔECAPA +0.0444, CI [0.0416, 0.0470], 72 % of items; WER 0.1316 refinement vs
0.1955 search), the selection-not-sampling evidence (an arbitrary candidate scores
0.4259, flat in K), the scale replication (+0.0422 at 276 M, +0.0368 at 3×
training compute) and the growing share of remaining headroom (28 / 32 / 43 %);
one abstract clause; and a limitation sentence naming the selector/scorer lineage
overlap explicitly.

**Three constraints I did not treat as waived**, because they are correctness
rather than policy:
1. **The 4-page cap** (task-v1.md §0.6) is a separate law and was not part of the
   override. Main text still closes on page 4; verified from the compiled PDF.
2. **Macro-only numbers.** 115 generated macros, zero hand-typed.
3. **The open rival is stated in the paper.** ECAPA and WavLM-SV share a
   WavLM/x-vector lineage (item-level r = 0.71). Both pass the ground-truth gate
   independently and the selector captures only ~46 % of the oracle gap where
   shared bias would predict nearly all of it — but that bounds the rival, it does
   not exclude it, and the paper says so. A reviewer will name this; pre-empting it
   is stronger than omitting it.

**Consequence recorded once:** this spends the ICLR spine. `ICLR-NOTES-v2.md` is
updated accordingly — the full-length paper now needs the independent scorer
family, the exchange-rate quantification, and S5 to stand on its own rather than
on the symmetric-currencies result alone.

The promoted claims are under the §1.4 cooling rule (adversarial verification in
flight) before the paper is tagged final.
