# ASPECT-D — DECISION

> **Status: IN PROGRESS.** The outcome class is declared mechanically by
> `src/fit.py` from the full surface once Phase 3 finishes and gate G4 is
> evaluated; this file is rewritten at that point. Everything below the
> "Declared outcome class" heading is final only when that heading names a class.
> Sections that are already final (gate history, compute, calendar, expansion
> notes) are written as they land, per the append-as-you-go discipline of
> `task.md` directive 3.

## Declared outcome class

**PENDING** — awaiting the complete surface and gate G4. No hypothesis verdict is
stated here before the run-level bootstrap has run; `fits.json` decides, and a
missing CI routes to `UNDETERMINED` (a loud analysis failure) rather than to the
F1 clean negative.

## Gate history (measured values)

| gate | verdict | key measured values |
|---|---|---|
| G0 (a) Mimi roundtrip | PASS | mel-L1 0.966 vs 4.046 cross-clip control |
| G0 (b) 200-step overfit | PASS | masked CE 7.640 → 0.169 (97.8 % reduction) |
| G0 (c) eval harness on ground truth | PASS (after repair R-1) | per-item WER 3.45 %, same-speaker SIM-o 0.701, cross-speaker 0.034 |
| G0 (d) sec_per_char | PASS | 0.06020 s/char over 10,000 training clips |
| G0 (e) sampler integrity, untrained | PASS | 99.0 % of generated cells differ T=1 vs T=16 |
| param-count test | PASS | 0.015–0.067 % from grid.json for A5/B3/C1 |
| G1 width transfer | PASS | argmin LR 0.004 (w=256) vs 0.002 (w=640) — within 2× |
| G1b depth transfer | PASS | argmin LR 0.002 (d=4) vs 0.002 (d=24) |
| coordinate check | PASS | block activation RMS 0.315 (w=256) vs 0.303 (w=640) |
| G2 pilot floors | PASS | WER(C3,T=16) 11.25 %, WER(A3,T=16) 16.13 %, SIM-o(C3,T=16) 0.408, DegenRate(C3,T=16) 0.00 % |
| §6.3 sampler integrity, trained | PASS | 98.5–98.6 % of generated cells differ T=1 vs T=16 |
| G3 grid health | pending | |
| G4 power | pending | |
| G5 compute | within cap | 31 GPU-h used, 118 projected of 500 |
| G6 calendar | on track | 23 days to the 2026-08-29 AoE wall at Phase 3 |

**Pivots fired: none.** P1-D was implemented and held ready when the 27 %-of-training
probe suggested a possible WER floor failure; the completed pilots cleared G2 by a
factor of 2.7 on the binding threshold, so the coarse-to-fine recipe of
`grid.json` stands. P2 was never applicable (budget A passed its floor).

**Cuts applied: none.** Neither the compute cut list nor the calendar cut list was
triggered: the projection is 24 % of the compute cap and the grid finished more
than two weeks inside the M3 milestone.

## Compute and calendar

- Hardware: 8 × NVIDIA B200 (183 GB, sm_100). Every run occupies exactly one GPU;
  no run is sharded (task.md §2). Charged GPU-hours are the sum of per-run wall
  time, which over-counts while several runs share a device; `state.json` also
  records the exact per-GPU occupancy from the scheduler intervals.
- Data preparation (download, selection, phonemisation, Mimi encoding of 2,000 h)
  and all gate checks ran inside Phase 0.

## What the full-length AR ASPECT paper should reuse (ICLR expansion notes)

1. **The harness, unchanged.** `src/model.py`, `src/train.py` and `src/data.py`
   are agnostic to the objective: the AR study needs only a causal mask and a
   next-token loss, and inherits the iso-N shape grid, the μP LR transfer with its
   G1/G1b certification, the config-independent batch/masking streams, and the
   effective-batch loss normalisation that makes gradients identical under any
   micro-batch split.
2. **The evaluation stack verbatim.** Whisper-large-v3 with the Whisper English
   normaliser, WavLM-large SV against the *original* prompt waveform,
   UTMOS22-strong, the frozen degenerate rule, and — importantly — the
   ground-truth-transcribability curation of the eval set (repair R-1). The AR
   paper should quote the same G0(c) sanity numbers so the two studies' WERs are
   comparable.
3. **The statistics module.** `src/fit.py` implements the model forms, the AICc
   comparison, the run-level bootstrap and the hypothesis routing. For AR the
   step axis disappears, so only Part A (M_full vs M_N, ρ = α/β) applies; Δρ and
   the G4 power gate transfer unchanged.
4. **What moved here, and what the AR paper must therefore not re-claim.**
   [to be completed once the outcome class is declared]
5. **The measurement that surprised us and should be re-run on AR hardware:**
   c_layer(width) is *flat* in width on a B200 at batch 1 (0.493–0.514 ms/layer
   across 256–1152), i.e. serial latency is set by depth × steps alone and width
   is free until batching makes the GEMMs compute-bound. Any serving corollary in
   the AR paper needs its own measurement of this, not an assumed constant.

## Submission readiness

[to be completed at Phase 7]
