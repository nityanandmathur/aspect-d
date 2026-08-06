# ASPECT-D — DECISION

## Declared outcome class: **S1**

**H-D2 is supported: test-time scaling in masked-diffusion TTS is metric-selective.**
Refinement steps reduce intelligibility error faster than speaker-identity error, and the
pre-registered run-level bootstrap interval excludes zero.

| statistic | value | 95 % percentile CI (run-level, 2,000 replicates, RNG 7331) |
|---|---|---|
| **Δτ = τ_WER − τ_SIM** | **+0.0650** | **[0.0528, 0.0770]** — excludes 0 |
| τ_WER | 0.8086 | [0.8029, 0.8144] |
| τ_SIM | 0.7437 | [0.7323, 0.7547] |
| Δρ = ρ_SIM − ρ_WER | −0.6152 | [−1.0046, +1.0110] — includes 0 |
| κ_WER (M_sub) | 3.0000 — **at the fitted bound, not identified** | [3.0000, 3.0000] |

Declared from the **largest balanced seed set** available at Phase 5 — seeds {0, 1} over
all 15 active configs × 5 T values = **150 surface points from 30 trained runs** — under
the composition rule fixed in `LOG.md` *before any fit was run*. The ragged fit that also
included the finished stretch runs is kept as
`artifacts/fits_ragged_3seed_robustness.json` and yields the same class with a larger Δτ
(0.139, CI [0.124, 0.153]), so the declaration does not depend on composition.

### Secondary hypotheses — all reported as they came out

- **H-D3 (PRIMARY, not supported).** The substitution form is *decisively worse* than the
  separable form: ΔAICc(M_sub − M_sep) = **+55.2** (WER) and **+89.8** (SIM-o) against the
  ≤ +4 the hypothesis required, and κ runs to its bound (+3) for both metrics. **No κ is
  quotable** — steps and depth do not trade off as `d·T^κ` in this regime. This is the one
  headline the pre-registration hoped for and the data refused; the paper says so, and the
  title was changed accordingly (see below).
- **H-D1 (secondary, not supported).** Δρ's CI spans zero. The point estimate is
  additionally uninformative: the width coefficient A collapses to its lower bound at
  T=16 for *both* metrics, so α — and hence ρ = α/β — is not identified. What *is*
  unambiguous is the other half of H-D1: iso-N shape matters, ΔAICc(M_full − M_N) =
  **−29.5** (WER) and **−16.3** (SIM-o), far past the −4 threshold. Error is not a
  function of parameter count alone.
- **H-D4 (mixed).** SIM-o extrapolates from the two smaller budgets to the largest with
  MAPE 2.2 % (vs 3.0 % for the N-only model), inside the 15 % bar; WER does not (24.7 %
  vs 23.2 %).
- **Descriptive.** T\* = 16 for WER — intelligibility had *not* saturated at the largest
  budget tested, so T\*_WER is a lower bound — and 8 for SIM-o. DegenRate falls
  41.0 % → 8.9 % → 0.78 % → 0.25 % → 0.11 % across T ∈ {1, 2, 4, 8, 16}.

### Title change forced by the result
`paper/OUTLINE.md` pre-specifies the S1 title "Denoising Steps Rent Depth, Not Width".
The "rent depth" half is exactly the H-D3 substitution claim, which the data reject, so
using it would overclaim. The paper ships as **"Refinement Steps Buy Intelligibility, Not
Identity: Metric-Selective Test-Time Scaling in Masked-Diffusion TTS"** — the S1 framing
("test-time scaling is metric-selective") without the clause H-D3 would have licensed.

## Gate history (measured values)

| gate | verdict | key measured values |
|---|---|---|
| G0 (a) Mimi roundtrip | PASS | mel-L1 0.966 vs 4.046 cross-clip control |
| G0 (b) 200-step overfit | PASS | masked CE 7.640 → 0.169 (97.8 % reduction) |
| G0 (c) eval harness on ground truth | PASS (after repair R-1) | per-item WER 3.45 %, same-speaker SIM-o 0.701, cross-speaker 0.034 |
| G0 (d) sec_per_char | PASS | 0.06020 s/char over 10,000 training clips |
| G0 (e) sampler integrity, untrained | PASS | 99.0 % of generated cells differ T=1 vs T=16 |
| param-count test | PASS | 0.015–0.067 % from grid.json (A5, B3, C1) |
| G1 width transfer | PASS | argmin LR 0.004 (w=256) vs 0.002 (w=640) — within 2× |
| G1b depth transfer | PASS | argmin LR 0.002 (d=4) vs 0.002 (d=24) |
| coordinate check | PASS | block activation RMS 0.315 (w=256) vs 0.303 (w=640) |
| G2 pilot floors | PASS | WER(C3,T=16) 11.25 %, WER(A3,T=16) 16.13 %, SIM-o(C3,T=16) 0.408, DegenRate 0.00 % |
| §6.3 sampler integrity, trained | PASS | 98.5–98.6 % of generated cells differ T=1 vs T=16, every run |
| G3 grid health | PASS | **15 of 15** configs completed both mandatory seeds (threshold 12); zero failed runs, zero restarts |
| G4 power | PASS | WER spread ≥ 2× pooled seed SD in **3 of 3** budgets (A 0.402 vs 0.149; B 0.178 vs 0.046; C 0.097 vs 0.064) |
| G5 compute | within cap | 143 GPU-h charged, 65 h true GPU occupancy, 165 h projected of **500** |
| G6 calendar | on track | mandatory grid finished 2026-08-06, 15 days inside M3; 23 days to the AoE wall |

**Pivots fired: none.** P1-D was implemented and held ready when a 27 %-of-training probe
suggested a possible WER-floor failure; the completed pilots then cleared G2 by a factor
of 2.7 on the binding threshold, so the coarse-to-fine recipe of `grid.json` stands. P2
was never applicable — budget A cleared its floor.

**Cuts applied: none.** Neither the compute nor the calendar cut list was triggered: the
projection is 33 % of the compute cap and the grid finished more than two weeks inside M3.
The stretch third seed was scheduled because `grid.json → seeds.note` permits it (M3 early
**and** G5 ≤ 70 % of cap).

## Two defects caught before they could reach the numbers

Recorded here because they are the reason to trust the rest. Both were found *before* any
grid run, and both would have silently destroyed the primary result:

1. An adversarial review of the harness against the frozen protocol found the sampler
   re-drawing already-committed tokens at every refinement step — the final step re-drew
   the entire level, which would have made T largely cosmetic and τ meaningless.
2. The first end-to-end sampler test then found the Gumbel confidence noise was NaN
   through an operator-precedence slip, so every sampled token was codebook entry 0 and
   T=1 and T=16 produced *identical* output (0.0 % differing cells — an immediate §6.3
   failure). Protocol §6.3 exists for exactly this, and it worked.

Eight further confirmed defects were fixed in the same pass (bootstrap weighting inside
the run-level resample, SIM-o embeddings on zero-padded batches, unseeded initialisation,
val-split concentration, the missing G3 restart path, and others). All are in `LOG.md`.

## Compute and calendar

- 8 × NVIDIA B200. Every run occupies exactly one GPU; no run is ever sharded
  (task.md §2). 143 GPU-h are *charged* as the sum of per-run wall time, which over-counts
  because several runs shared devices and some were paused to prioritise the gating
  pilots; the true machine time is the 65 h of per-GPU occupancy recorded in
  `state.json`. Both figures are reported rather than the flattering one.
- Phase 0 (download, 2,000 h selection, phonemisation, Mimi encoding, all G0 checks),
  Phase 1 (20 LR-sweep proxies), Phases 2–3 (30 mandatory runs + 15 stretch), Phase 4
  (T-sweep synthesis and scoring, overlapped with training), Phases 5–7 all completed
  inside 2026-08-05/06, 23 days before the wall.

## Submission readiness

`paper/main.tex` is populated and compiles to `paper/main.pdf` with the official
`neurips_2026.sty` fetched from the NeurIPS 2026 Call for Papers page (6 pages total;
**references begin on page 4**, so the main text is inside the 4-page cap). It is
anonymised: no author names, affiliations, company references, repo URLs or
acknowledgements, and the artefact is referred to only as a released harness. Every number
in the prose is a macro from `paper/numbers.tex`, generated by `src/paper.py` directly out
of `artifacts/fits.json`, `artifacts/runs.csv` and `state.json` — no number is typed by
hand. Figures are the four protocol §10 SVG+PDF pairs built from the same fits.

**Ready to submit.** The one judgement call a human should confirm before upload is the
title change above, which is forced by H-D3's refutation.

## What the full-length AR ASPECT paper should reuse (ICLR expansion notes)

1. **The harness, unchanged.** `src/model.py`, `src/train.py`, `src/data.py` are agnostic
   to the objective: the AR study needs only a causal mask and a next-token loss, and
   inherits the iso-N shape grid, μP LR transfer with its G1/G1b certification, the
   config-independent batch/masking streams, and the effective-batch loss normalisation
   that makes gradients identical under any micro-batch split.
2. **The evaluation stack verbatim.** Whisper-large-v3 with the Whisper English
   normaliser, WavLM-large SV against the *original* prompt waveform, UTMOS22-strong, the
   frozen degenerate rule, and the ground-truth-transcribability curation of the eval set
   (repair R-1). Quote the same G0(c) sanity numbers so the two studies' WERs are
   comparable.
3. **The statistics module.** `src/fit.py` implements the model forms, AICc comparison,
   run-level bootstrap and hypothesis routing. For AR the step axis disappears, so only
   Part A applies; Δρ and the G4 power gate transfer unchanged — but note that Part A on
   15 iso-N points left α unidentified here (A at its bound), so the AR paper should
   either widen the width range or expect the same problem.
4. **What moved here, and what the AR paper must therefore not re-claim.** The
   metric-selectivity result (Δτ > 0) is now published for the diffusion family and is
   *not* available to the AR paper, which has no step axis. Conversely the AR anisotropy
   headline remains untouched: this study could not identify ρ, so ASPECT-D makes no
   width-versus-depth ratio claim and leaves that ground clear. What ASPECT-D *does*
   establish and the AR paper must engage with is that iso-N shape matters at all
   (ΔAICc(M_full − M_N) ≤ −16 for both metrics) and that depth dominates intelligibility
   at fixed N (33.5, 16.5 and 5.6 WER points between the shallowest and deepest shape of
   the 20 M, 50 M and 125 M budgets).
5. **The measurement that surprised us and should be re-run on AR hardware:**
   c_layer(width) is *flat* in width on a B200 at batch 1 (0.493–0.514 ms/layer across
   256–1152), so serial latency is set by depth × steps alone and width is free until
   batching makes the GEMMs compute-bound. Any serving corollary needs its own
   measurement of this, never an assumed constant.
