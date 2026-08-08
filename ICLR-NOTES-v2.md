# ICLR notes v2 — the symmetric-currencies thesis

**Status:** candidate spine for the full-length paper. Deliberately **not** spent
in the DiffuLM workshop submission (task-v2.md §5, no-self-scoop directive).
Every number below is measured; provenance in `RESULTS-FEED.md`.

## The thesis

At a **fixed inference budget**, a masked-diffusion TTS system can buy either
intelligibility or speaker identity, and which one it gets depends only on how the
budget is spent.

Measured at matched NFE on 15 C-budget models × 200 items:

| NFE | spend it on refinement | spend it on search (best-of-K, T=8) |
|---|---|---|
| 128 | WER **0.1530** / ECAPA baseline | WER 0.1938 / ECAPA **+0.0153** |
| 256 | WER **0.1335** | WER 0.1947 / ECAPA **+0.0322** |
| 512 | WER **0.1316** | WER 0.1955 / ECAPA **+0.0444** |

Refinement wins WER in 3/3 tiers; search wins identity in 3/3 tiers, monotonically
in K, with per-item win rates 57 % → 65 % → 72 % and per-item *medians* moving
nearly as much as the means. Same compute, same checkpoints, same items.

## Why it is more than a curiosity

1. **It completes the workshop paper's asymmetry.** v1.0 showed refinement moves
   intelligibility 5.9× more than identity in absolute error. The natural reading
   — "identity is simply not purchasable at test time" — is **wrong**. Identity is
   purchasable; refinement is just the wrong instrument for it.
2. **It is bounded by a measured ceiling.** The Mimi roundtrip ceiling leaves
   0.0747 SIM-o of headroom, and 62 % of the total gap to real audio is codec
   loss. Search recovers **59.5 %** of what is left — roughly what tripling
   training compute buys (+0.0788), and more than a 14× parameter increase
   (+0.0531).
3. **It is a serving-time control, not an architecture claim.** One checkpoint,
   two spending policies, two different products.

## What must be done before this is a paper

- **Break the scorer-family confound.** ECAPA and WavLM-SV are both
  WavLM/x-vector lineage. Both pass G0(c) independently, but a genuinely
  different family (d-vector, or a fine-tuned ASR-speaker hybrid) is needed
  before the identity side is safe.
- **Scale it.** Everything is C-budget, 30k steps, one corpus, one codec, one
  language. The interesting question is whether the exchange rate moves with N
  and with training compute — the headroom shrinks as the baseline improves.
- **Vary the candidate source.** All K candidates come from one model at T=8.
  Best-of-K across seeds, temperatures or checkpoints may behave differently and
  is the obvious ablation.
- **Add the guidance axis properly.** S4 showed training-free contrastive
  guidance is a real but non-free knob (+0.0121 for +2.73 WER points). S5 (CFG
  with condition dropout, post-freeze) is the training-time version and belongs
  here, not in the workshop paper.
- **Find the exchange rate.** The workshop paper reports that steps and depth are
  *not* interchangeable (M_sub decisively worse). The open question is whether
  refinement and search have a stable exchange rate for identity-vs-intelligibility
  — that is the quantitative spine.

## What this file must not do

Nothing here may be cited in, or leak into, the workshop paper beyond the single
codec-ceiling limitation sentence §5(a) permits. The workshop paper is
claim-complete on Δτ and the magnitude asymmetry.
