# PREREGISTRATION v1.5 — classifier-free guidance as a robustness check

**Written 2026-08-12, before any v1.5 CFG datum exists.** Committed with its git hash
recorded in `RESULTS-FEED.md` on the same day. The gate is evaluated by
`src/v15_gate.py`, which is committed at the same time and prints its verdict before
any human inspects the per-arm table.

---

## 0. Why this exists

The human directive is "put CFG in the paper if it improves results." That phrasing,
taken literally and applied after seeing the data, is outcome-dependent reporting: with
enough arms (2 conditioning signals x 3 guidance weights x 3 configs) something will
look good. An adversarial review of the first draft of this gate found its proposed
threshold passed **27 out of 27** arms already on disk — a rule that cannot fail is not
a rule. This document fixes what "improves" means while the answer is still unknown.

Two further facts, recorded now so they are not discovered later as convenient:

- `paper/main.tex` currently argues that the models are trained *without* CFG
  "because it would otherwise confound the step axis." If CFG enters the paper, that
  sentence must be rewritten, not quietly deleted.
- The main text is exactly full at 8 pages. A CFG subsection **displaces** existing
  content; §5 names what it displaces, chosen now rather than after the fact.

## 1. What is being tested

The paper's finding is an asymmetry: refinement closes a larger share of the reachable
intelligibility range than of the reachable identity range. Every model in the grid is
trained without condition dropout. The open question is whether the asymmetry is a
property of the system or an artefact of having no guidance available.

Let, for a checkpoint and a decoding setting,

```
A  =  (share of reachable WER range closed from T=1)
      / (share of reachable SIM-o range closed from T=1)
```

with floors fixed at the already-published measured values, `wer_asr_floor = 0.0345`
and `sim_codec_ceiling = 0.5554`. Both shares are normalised by that checkpoint's own
T=1 anchors, so `A` for a CFG checkpoint and `A` for a base checkpoint are comparable
only if the anchors are comparable — which precondition P0 tests rather than assumes.

## 2. Design (fixed now)

- **Training.** C1, C3, C5 x seeds {0, 1, 2} = **9 runs**, 30k steps, identical to the
  base grid in every other respect. Not 3 runs: an adversarial review showed 1 seed per
  config yields a run-level bootstrap of zero width, making any gate automatic. 9 runs
  matches the 90k design already used for the training-compute trend.
- **Condition dropout.** One 3-way categorical draw per example: 80% keep, 10% drop the
  speaker prompt, 10% drop the phoneme text. One training run therefore yields both
  guidance directions. Dropping is implemented by clearing the key-side attention bit,
  which removes the conditioning exactly; no new vocabulary entry, so existing
  checkpoints still load.
- **Inference.** Guidance weight gamma in {0.5, 1.0, 2.0} on each arm, T=16.
- **Primary arm, fixed a priori: prompt-CFG.** It is the arm that can *refute* the
  paper: if guidance buys identity, the asymmetry is a no-guidance artefact. Text-CFG
  is always reported and never gated on.
- **gamma is not maximised over.** The primary gamma is chosen on a **selection split**
  — the first 100 eval items by sorted item id — and all reported numbers come from the
  disjoint **evaluation split** of the remaining 300. The split is fixed here.
- **Baseline is iso-NFE.** Guidance costs two forwards per step, so guided T=16 (256
  forwards) is compared against unguided T=32 (256 forwards), never unguided T=16.
  This is the same defect corrected in the S4 arm on 2026-08-12.

## 3. Preconditions — any failure means INCONCLUSIVE, no claim anywhere

| id | test | threshold |
|---|---|---|
| P1 | condition dropout actually fired | realised cumulative drop rate in [0.085, 0.115] for each of prompt and text, from the training log |
| P2 | the CFG checkpoint is not simply worse | at gamma=0, T=16, vs its matched base run: WER <= +15% relative **and** SIM-o >= -0.02 absolute |
| P3 | the anchors are comparable | \|A_cfg(gamma=0, T=16) - A_base(matched 3 configs)\| <= 0.15 |
| P4 | guidance is live at inference | \|delta SIM-o\| >= 0.005 absolute on the primary arm vs its iso-NFE reference, paired-item 95% CI excluding 0 |

## 4. THE GATE — does CFG enter the paper?

Evaluated on the evaluation split, at iso-NFE, with run-level bootstrap over the 9 runs.
**Exactly one of these three outcomes obtains.**

**(a) REFUTATION — MUST be reported in the paper, whatever it does to the story.**
Guided `A` has a 95% CI whose upper bound is <= 1.05, i.e. guidance abolishes or
reverses the asymmetry. This would mean the paper's central contrast is a property of
un-guided decoding, and it is reported as such. *This branch is what makes the gate
honest: a result that damages the paper is not permitted to stay out of it.*

**(b) PARETO IMPROVEMENT — enters the paper as a robustness subsection.**
At iso-NFE, guided vs unguided on the same checkpoint:
`delta SIM-o >= +0.010` with 95% CI excluding 0, **and** `delta WER <= +0.005` absolute.
That is a genuinely free identity gain, and it strengthens the paper's own argument that
identity responds to inference-time intervention where refinement does not.

**(c) NEITHER — does not enter the paper.**
Reported in `ICLR-NOTES-v2.md`, `RESULTS-FEED.md` and `cfg.html` as a measured negative,
with one sentence in the paper's limitations recording that guidance was tested and did
not change the finding. No selective silence.

A result that is merely "directionally nice" — a SIM gain below 0.010, or one bought
with a WER cost above 0.005 — is outcome (c). This is the threshold that the earlier
draft got wrong.

## 5. If the gate opens, what the CFG subsection displaces

Chosen now, not after seeing results, in this order until space is found:
1. Table 2 (fitted exponents) moves to the appendix; the two tau values it carries are
   already quoted in prose and the form is descriptive only.
2. The `Extrapolation and degenerate outputs` paragraph compresses to two sentences.
3. Figure 3 (step curves) drops to a single panel.

`paper/main.tex`'s "trained without classifier-free guidance, which would otherwise
confound the step axis" is rewritten in every branch where CFG enters, to state that
the *main grid* is CFG-free by design and that a separate matched arm tests whether
that choice determines the result.

## 6. What this does NOT claim

Guidance *scaling* — how guidance strength trades against steps and parameters — is the
sequel's axis and is not touched here. This arm asks one question: does the measured
asymmetry survive when guidance is available? Three gamma values at one T is not a
scaling study and will not be described as one.
