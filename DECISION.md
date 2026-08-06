# ASPECT-D — DECISION

## Declared outcome class: **S1**

**H-D2 is supported: test-time scaling in masked-diffusion TTS is metric-selective.**
Refinement steps reduce intelligibility error faster than speaker-identity error, and the
pre-registered run-level bootstrap interval excludes zero in every replicate.

| statistic | value | 95 % percentile CI (run-level, 2,000 replicates, RNG 7331) |
|---|---|---|
| **Δτ = τ_WER − τ_SIM** | **+0.1102** | **[0.0921, 0.1312]** — excludes 0 (0 of 2,000 replicates ≤ 0) |
| τ_WER | 0.8362 | [0.8248, 0.8513] |
| τ_SIM | 0.7260 | [0.7088, 0.7440] |
| Δρ = ρ_SIM − ρ_WER | +0.1741 | [-0.7106, +0.3884] — includes 0 |
| κ_WER (M_sub) | 3.0000 — **at the fitted bound, not identified** | [3.0000, 3.0000] |

Declared from the **balanced three-seed grid** — seeds {0, 1, 2} over all 15 active
configs × 5 T values = **225 rows / 75 surface points from 45 trained runs** — under the
composition rule fixed in `LOG.md` *before any fit was run* (largest balanced seed set).
The two-seed analysis it supersedes is kept as
`artifacts/fits_twoseed_robustness.json`; it gave the same outcome class with a smaller
effect (Δτ = 0.0650, CI [0.0528, 0.0770]), so the declaration does not
depend on composition. The ragged intermediate fit is in
`artifacts/fits_ragged_3seed_robustness.json`.

### Secondary hypotheses — all reported as they came out

- **H-D3 (PRIMARY, not supported).** The substitution form is *decisively worse* than the
  separable form: ΔAICc(M_sub − M_sep) = **+69.3** (WER) and
  **+71.3** (SIM-o) against the ≤ +4 the hypothesis required, and κ runs to
  its bound (+3) in **every one of the 2,000 bootstrap replicates** for both metrics.
  **No κ is quotable** — steps and depth do not trade off as `d·T^κ` in this regime. This
  is the one headline the pre-registration hoped for and the data refused; the paper says
  so, and the title was changed accordingly (below).
- **H-D1 (secondary, not supported).** Δρ = +0.1741, CI [-0.7106, +0.3884], spans zero. The
  estimate is additionally uninformative, for a reason worth recording: the width amplitude
  A saturates at the **upper** bound the pre-registration set (A ≤ 10) in every fit — Part A
  A = 10.000 for both metrics, Part B 9.999 (WER) and 10.000 (SIM-o). Only the product
  A·w^−α is identified, so α, and any ratio built from it, is not. The bound itself is the
  binding constraint: the data want a larger width amplitude than protocol §7.1 permits.
  Evidence
  for that reading: Δρ's point estimate *flipped sign* between the two-seed
  (-0.615) and three-seed (+0.174) compositions while both CIs spanned zero.
  The other half of H-D1 does hold — iso-N shape matters: ΔAICc(M_full − M_N) =
  **-15.5** (WER) and **-5.5** (SIM-o) against a −4 threshold, though
  identity is now only marginally past it.
- **H-D4 (not supported).** Neither metric extrapolates from the two smaller budgets to the
  largest within 15 % *and* beating the N-only model: WER 20.2 % vs 20.4 %,
  SIM-o 2.4 % vs 2.2 %. (On two seeds SIM-o passed; with the third seed it
  no longer beats the N-only baseline. Reported as it fell.)
- **Exploratory (labelled, cannot change the declared class).** The *depth* exponent is
  identified — B is interior in both Part-B fits (4.613 for WER, 0.163 for SIM-o) — and it
  separates the two capabilities sharply: **β_WER = 2.040 against β_SIM = 0.553**, a factor
  of 3.7. That is the qualitative asymmetry H-D1 was reaching for, expressed in the
  parameter the data actually pin down rather than in the ratio ρ that the A-bound spoils.
  Part-B ρ values (0.771 WER, 1.662 SIM-o, Δρ = +0.890) do point the pre-registered way,
  but they inherit the same A-cap problem and the pre-registered test is Part A, so they
  are reported here only, as exploratory.
- **Descriptive.** T* = 16 for WER — intelligibility had *not* saturated at the
  largest budget tested, so T*_WER is a lower bound — and 8 for SIM-o. Grid-mean
  DegenRate falls 40.4 % → 8.8 % → 0.82 % → 0.23 % → 0.12 % across T ∈ {1,2,4,8,16};
  UTMOS rises 1.52 → 2.96 against 3.33 on ground-truth audio.
- **Depth saturates at an interior optimum.** At T=16, moving from the shallowest shape to
  the best depth is worth 40.2, 16.3 and 7.3 WER points in the 20, 50 and 125 M budgets,
  but the best depth is *interior* in the smaller budgets (d=18 of 4–24; d=30 of 6–30;
  d=36 of 8–36). The rule is "deep enough", not "as deep as possible" — the paper states
  this rather than a monotone depth claim.

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
| G3 grid health | PASS | **15 of 15** configs completed both mandatory seeds (threshold 12); zero failed runs, zero restarts; the stretch third seed also completed for all 15 |
| G4 power | PASS | spread ≥ 2× pooled seed SD in **3 of 3** budgets for **both** metrics on the three-seed grid |
| G5 compute | within cap | **172 GPU-h charged**, 65 h true GPU occupancy, of **500** (34 %) |
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
  (task.md §2). 172 GPU-h are *charged* as the sum of per-run wall time, which over-counts
  because several runs shared devices and some were paused to prioritise the gating
  pilots; the true machine time is the 65 h of per-GPU occupancy recorded in
  `state.json`. Both figures are reported rather than the flattering one.
- Phase 0 (download, 2,000 h selection, phonemisation, Mimi encoding, all G0 checks),
  Phase 1 (20 LR-sweep proxies), Phases 2–3 (30 mandatory runs + 15 stretch), Phase 4
  (T-sweep synthesis and scoring, overlapped with training), Phases 5–7 all completed
  inside 2026-08-05/06, 23 days before the wall.

## Submission readiness

`paper/main.tex` is populated and compiles to `paper/main.pdf` with the official
`neurips_2026.sty` fetched from the NeurIPS 2026 Call for Papers page (9 pages total;
**the main text closes on page 4 and references begin on page 5**, so the 4-page cap holds; the appendix and the disclosure follow it and do not count). It is
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
