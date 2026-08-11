# paper/OUTLINE.md — the 4-page extended abstract, section by section

Venue: DiffuLM @ NeurIPS 2026. Limits: 4 pages MAIN TEXT (references and a short
appendix excluded), NeurIPS 2026 style, anonymized, double-blind. Non-archival.

Working title (adjust to outcome class, see protocol §7.4):
  S1: "Denoising Steps Rent Depth, Not Width: Metric-Selective Test-Time Scaling   [WITHDRAWN: reads as a rate claim; see DECISION-v1.2.md §2]
       in Masked-Diffusion TTS"
  S2: mirror title with the observed direction.
  F1: "Refinement Lifts All Boats: Test-Time Scaling in Masked-Diffusion TTS Is
       Not Metric-Selective"
  F2: "How Much Signal Is There in TTS Shape Scaling? A Controlled Measurement"

Budget: ~55 lines/page NeurIPS style. Numbers below are targets, not laws;
the 4-page main-text cap IS a law.

## §1 Introduction — 0.75 page
- Para 1: diffusion LMs decouple owned serial compute (depth) from rented serial
  compute (steps); AR cannot. One sentence on why TTS is the clean probe: two
  separable capabilities (intelligibility, identity) with standard metrics.
- Para 2: the two primary claims (H-D2, H-D3 in words), the pre-registered design,
  and the headline numbers (Δτ CI; κ_WER CI) from fits.json.
- Contributions list (3 bullets max): (i) first per-metric (w, d, T) scaling
  study for speech generation; (ii) the Δτ / κ findings with CIs; (iii) the
  serving decision rule + released suite.

## §2 Related work — 0.4 page
- Compress the gap table of index.html §2 into one paragraph: text MDM scaling;
  SoundStorm/MaskGCT (recipes, no laws); AR TTS N-scaling (Llasa); test-time
  scaling in reasoning (single-metric). One sentence each. End with the gap.

## §3 Method — 0.9 page
- Backbone + objective + frozen sampler in one tight paragraph each (cite
  grid.json values: 12·d·w², head_dim 64, Mimi 12.5 Hz × 8 books, SoundStorm-style
  training, MaskGIT decode, T per level, NFE = 8T, no CFG, length from char-rate).
- Design: iso-N budgets {20M, 50M, 125M} × 5 aspect ratios × seeds; T swept at
  inference only; matched RNG across T. State μP + depth-transfer gate in one
  sentence. State run-level bootstrap in one sentence (reviewers at this venue
  will check).
- The two model forms M_sep and M_sub as display equations; define Δτ, κ, Δρ.

## §4 Results — 1.4 pages (the paper lives here)
- Fig 1 (two panels, from artifacts/figures): step_curves.svg — err vs T per
  metric, normalized, with T* marked; substitution_plane.svg — iso-WER in
  (log T, log d) with fitted slope −κ and the iso-latency line.
- Table 1: fitted exponents (α, β, τ) per metric with CIs; Δτ row bolded;
  AICc comparison columns (M_sep vs M_sub vs N-only).
- Text: Part-B findings first (H-D2, H-D3 with CIs), then Part-A anisotropy at
  T=16 (H-D1, one paragraph), then H-D4 extrapolation (two sentences), then
  DegenRate(T) (two sentences).
- The serving corollary: one short paragraph + the populated latency table
  (budget → optimal d, T, w) with measured c_layer. If κ<1 vs κ>1, state the
  allocation rule explicitly.

## §5 Limitations & outlook — 0.35 page
- Scale ceiling (≤125M / ≤285M), single language, single codec, CFG excluded
  (first follow-up), any calendar cuts taken (be exact: which cuts, why),
  outcome-class caveats per protocol §7.4.
- One sentence: full-length AR/shape companion study in preparation.

## §6 Reproducibility statement — 2–3 lines
- Pre-registered protocol + frozen sampler + released runs/fits (anonymized
  placeholder for the artifact link during review).

## Appendix (beyond 4 pages, optional, keep short)
- Grid table (15–20 configs), gate history summary, extra figures
  (aniso_contours_T16.svg, extrapolation.svg), fit diagnostics.

## Hard rules for Phase 7
- Every number traces to artifacts/fits.json or artifacts/runs.csv.
- No claims outside the declared outcome class; exploratory material only in the
  appendix under an "Exploratory" heading.
- Anonymize fully: no author names, affiliations, company references, repo URLs,
  or acknowledgments.
- If figures must shrink to fit 4 pages, cut Fig 1 to the substitution panel
  only and move step_curves to the appendix — never cut Table 1.
