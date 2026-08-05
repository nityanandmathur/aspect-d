# ASPECT-D — Pre-registration (frozen v1.0, prior to any training run)

Formal definitions: `protocol.html §7`. This file freezes the claims and their
priority ordering. Target venue: DiffuLM @ NeurIPS 2026 (non-archival, 4-page
extended abstract, deadline 2026-08-29 AoE).

## Hypotheses (in priority order)

- **H-D2 (PRIMARY — metric-selective test-time scaling).** Under
  err_m = E_m + A_m·w^(−α_m) + B_m·d^(−β_m) + C_m·T^(−τ_m) fit on the full
  (w, d, T) surface: Δτ = τ_WER − τ_SIM > 0.
  Decision: 95% percentile bootstrap CI (2,000 replicates, resampled at the RUN
  level — the T-points of one trained model are correlated) excludes 0.
- **H-D3 (PRIMARY — depth–step exchange rate).** Under the substitution form
  err_m = E_m + A_m·w^(−α_m) + B_m·(d·T^(κ_m))^(−β_m) for WER:
  κ_WER > 0 with CI excluding 0, AICc(M_sub) ≤ AICc(M_sep) + 4, and κ_WER > κ_SIM.
- **H-D1 (secondary — anisotropy at fixed T = 16).**
  Δρ = α_SIM/β_SIM − α_WER/β_WER > 0; CI excludes 0 and the two-axis model beats
  the N-only model (ΔAICc ≤ −4) for ≥1 of {WER, SIM}. (The AR-domain version of
  this claim is deliberately reserved for the separate full-length ASPECT paper.)
- **H-D4 (extrapolation).** Two-smaller-budget fits at T=16 predict the largest
  budget with MAPE ≤ 15% and ≤ the N-only model's MAPE.
- **Descriptive, no hypothesis:** per-metric saturation step T*_m (95% of the
  T=16 value); DegenRate(T) curves.

## Outcome classes (exactly one will be declared)

S1: H-D2 supported. S2: CI excludes 0 with Δτ < 0. F1: CI includes 0 with power
gate G4 passed. F2: G4 failed. Paper framings for each are frozen in
protocol §7.4; the T-FLAT pilot flag routes to F1 only if the sampler-integrity
check passed.

## Analysis lock

Model forms, error transforms, weighting, bounds, multi-start counts, bootstrap
scheme (run-level, 2,000 reps, RNG 7331), the frozen sampler (MaskGIT confidence
decode, cosine schedule, Gumbel noise annealed 1→0, temperature 1.0, no CFG,
matched per-item RNG across T), gate thresholds, pivot rules, the compute cap
(500 GPU-h), and the calendar milestones are frozen as written in
`protocol.html` v1.0 and `configs/grid.json`. Amendments require a version bump
plus a LOG.md entry demonstrating the change cannot bias H-D2/H-D3. Exploratory
analyses are permitted in a clearly labeled section and cannot alter the
declared outcome class.
