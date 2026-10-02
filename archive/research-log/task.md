# ASPECT-D — Autonomous Execution Runbook (task.md)

You are Claude Code, executing Project ASPECT-D end-to-end **without human
intervention**, against a hard external deadline: **DiffuLM @ NeurIPS 2026,
submission due 2026-08-29 AoE** (4-page extended abstract, NeurIPS 2026 style,
double-blind, OpenReview). The human may not see anything until you are done.
This file is *what to do*; `protocol.html` is *the binding thresholds and
procedures*; `configs/grid.json` is *every number* (architecture, recipe, frozen
sampler, calendar). `paper/OUTLINE.md` is the structure Phase 7 must fill.

---

## 0. PRIME DIRECTIVES (re-read before every gate decision)

1. **No invented numbers, ever.** The theory lives in `index.html`, procedures in
   `protocol.html`. Before ANY decision — threshold, pivot, hyperparameter, metric,
   scope cut — re-read the relevant section of `protocol.html` / `configs/grid.json`
   and use it verbatim. If a needed value exists nowhere in the repo: use §10
   (Defaults); if absent there too, take the most conservative option and record in
   `LOG.md` what was missing, what you chose, and why it cannot bias H-D2/H-D3.
2. **Canonical precedence:** `configs/grid.json` → `protocol.html` → `task.md` →
   `index.html` (motivation only; its figures contain placeholder values by design).
3. **Log everything** in append-only `LOG.md`: every gate evaluation with measured
   values, every pivot/cut (compute AND calendar), every fallback, every deviation.
4. **Never tune per-shape or per-T.** After Phase 1 fixes LR, nothing may differ
   between runs except (w, d, heads, seed), and nothing may differ between T values
   except T itself (matched generation RNG per item across T — grid.json sampler
   spec). Frozen sampler means frozen: no temperature/noise/schedule adjustments.
5. **Pre-registration is binding** (`PREREGISTRATION.md`, `protocol.html §7`). No
   hypothesis edits after data. Exploratory material goes in a labeled section.
6. **Two stop-losses, one wall.** Compute (G5, 500 GPU-h) and calendar (G6,
   milestones + Aug 29 AoE). When they conflict with completeness, they win.
   A submitted paper with a smaller honest grid strictly dominates an unsubmitted
   complete one. The writing floor (grid.json → calendar.writing_floor) is absolute.
7. **All four outcomes are wins** (S1/S2/F1/F2, protocol §7.4). Never bend a
   threshold toward S1. The step-flatness flag (T-FLAT) is a finding, not a bug —
   provided the sampler-integrity check passed.
8. **Checkpoint for resumability.** Save training state every 1,000 steps.
   `state.json` at repo root tracks: phase, completed runs, GPU-h consumed, active
   grid, active recipe (coarse-to-fine vs P1-D flat), T-grid semantics, gate/cut
   history, calendar projections. On startup ALWAYS read `state.json` and resume;
   never redo completed work.
9. Push all models to hugginface under "nityanandmathur" as private repo. The HF API key is in .env
10. Keep pushing all code changes to the github repo, nothing should be lost whatsoever, no matter now small or big a change is, it should be on github.

---

## 1. MISSION

Train the active shape grid from `configs/grid.json` (seeds: 2 mandatory, 1
stretch), sweep T at evaluation, fit the Part-A and Part-B laws of `protocol.html
§7`, test H-D1–H-D4, produce all §10 deliverables **including the populated 4-page
paper**, and declare exactly one outcome class in `DECISION.md` — all before
2026-08-29 AoE.

## 2. ENVIRONMENT ASSUMPTIONS

- 8x B200s. Multi-GPU: parallelize across runs, never shard one run.
- PyTorch ≥ 2.4, torchaudio, transformers (`kyutai/mimi` via MimiModel), datasets,
  phonemizer + espeak-ng, whisper-large-v3, scipy, numpy, pandas, matplotlib.
  LaTeX toolchain optional (see Phase 7).
- Install failures: 2 distinct attempts → listed fallback (grid.json) → directive 1.
- Data fallbacks per `grid.json → data.fallback_order`; a switch is a logged event.
- Code in `src/`, runs in `runs/<config>_<seed>/`, syntheses in
  `runs/<config>_<seed>/synth_T<T>/`, deliverables in `artifacts/` + `paper/`.

## 3. PHASE 0 — Environment, data, harness → gate **G0** (target M1: Aug 12)

1. Implement `src/model.py` (bidirectional masked-diffusion transformer exactly per
   `grid.json → backbone`), `src/data.py` (resample 24 kHz → Mimi encode → phonemize
   → cache token shards), `src/train.py` (recipe + SoundStorm-style masking per
   grid.json), `src/sample.py` (MaskGIT confidence decode, level-by-level, T steps
   per level, matched-RNG-across-T), `src/evaluate.py` (protocol §6),
   `src/fit.py` (protocol §7).
2. Build the data subset and splits (protocol §3); compute and log `sec_per_char`.
3. Param-count unit test: instantiated non-embedding count within 1% of grid.json
   for 3 sampled configs. Mismatch → fix model code, never the JSON.
4. Run G0 checks exactly as written (protocol §8), including the sampler-integrity
   check on an untrained model.
5. Pass → Phase 1. Fail → ≤3 distinct repair attempts → **STOP-ENV** (DECISION.md
   explains the blocker).

## 4. PHASE 1 — LR transfer → gates **G1, G1b** (within M1)

μP per grid.json (base width 256, residual scale 1/√(2d)); coordinate check across
widths {256, 640} (activation RMS stable over 50 steps — failure is a bug to fix,
not a gate outcome). Then the four 5-point sweeps of protocol §5, verbatim pass/fail
rules and fallbacks. Record chosen LR rule in state.json. Update compute + calendar
projections (G5, G6).

## 5. PHASE 2 — Pilots → gate **G2** (target M2: Aug 14)

1. Full-schedule A3, B3, C3 (seed 0), evaluated at T ∈ {1, 16} on eval_zs.
2. Apply G2 verbatim (note G2-alt under the LibriTTS-R fallback; note the T-FLAT
   flag rule). Route failures exactly as the G2 box says:
   only-A3 → **P2** (needs G5 AND G6 headroom); C3 → retry 0.5× LR → **P1-D**
   (recipe pivot; T semantics change per the P1-D box; re-run Phases 1–2);
   both pivots exhausted → **STOP-PILOT**, classify F2.
3. Pilots count as (config, seed 0) grid runs.
4. Measure GPU-h/run precisely; re-project G5 and G6; apply cut lists if needed.

## 6. PHASE 3 — Grid → gate **G3** (target M3: Aug 21)

1. Launch remaining mandatory runs (seeds {0,1}), interleaved across budgets and
   shapes so an early freeze still leaves balanced coverage. Schedule stretch
   seed-2 runs ONLY when grid.json's seeds.note condition holds (M3 early + ≤70%
   compute cap projected).
2. Failure handling per G3 (divergence definition, one 0.5× LR restart, then failed).
3. Every 10 runs: update state.json, re-project G5 and G6, apply cuts in order.
4. Grid validity per G3; backfill rule; else F2-candidate and continue.

## 7. PHASE 4 — T-sweep synthesis + evaluation (target M4: Aug 24)

1. For every completed (config, seed): synthesize eval_zs at every T in the active
   T grid with the frozen sampler; per-item generation RNG matched across T.
2. Run the sampler-integrity check (protocol §6.3) once per config before scoring.
3. Score all metrics at every T; apply the degenerate rule §6.4 mechanically;
   measure c_layer(width) per protocol §6.5.
4. Write `artifacts/runs.csv` (one row per config × seed × T, all §10 columns).
   Export `samples/` per §10.

## 8. PHASE 5 — Fits + statistics → gate **G4** (target M5: Aug 25)

1. `src/fit.py` implements protocol §7 exactly: Part A at T=16, Part B on the full
   surface, run-level bootstrap (2,000 reps, RNG 7331), AICc tables, H-D1–H-D4.
2. Evaluate G4 first. Fail → F2 path (fits + power analysis, no H-D claims).
3. Pass → classify S1/S2/F1 mechanically. The CI decides; do not narrate around it.

## 9. PHASE 6 + 7 — Report, then paper (M5 → M6: Aug 27; wall Aug 29 AoE)

1. Phase 6: `artifacts/figures/` (the four figures of protocol §10, real data,
   SVG+PDF) and `results.html` (house style; copy CSS from index.html; every number
   traceable to runs.csv/fits.json; populated design-rule table with measured
   c_layer; gate/cut/calendar history).
2. Phase 7 (**begins no later than Aug 26 regardless of experiment state** —
   grid.json writing_floor): populate `paper/main.tex` following `paper/OUTLINE.md`
   section by section, real numbers only, anonymized (double-blind: no names,
   no smallest.ai references, no repo URLs), 4 pages of main text maximum in
   NeurIPS 2026 style. The style file must be fetched from the NeurIPS 2026 CfP
   page; if unavailable in the environment, leave the \usepackage line with the
   TODO marker already present in main.tex and note it in DECISION.md — never
   substitute a different year's file silently.
3. Compile to `paper/main.pdf` if a LaTeX toolchain exists (try `latexmk -pdf`,
   else `pdflatex` twice); otherwise verify main.tex is complete and figures are
   staged, and note the missing toolchain in DECISION.md.
4. `DECISION.md` per protocol §10.
5. Final `state.json`: phase = DONE. Verify §11.

## 10. DEFAULTS (only for values specified nowhere else — log every use)

- Loudness: peak-normalize to −1 dBFS before Mimi encode, uniformly.
- Val-loss smoothing for LR selection: EMA over last 3 val points.
- Bootstrap RNG 7331; NLS multi-start RNG 42; data selection RNG 1234 (grid.json).
- Eval-item synthesis crash: retry once; then score as maximal error for WER and
  degenerate=true, exclude from SIM/UTMOS, log a crash-rate column.
- Phoneme vocab: built from the training subset, frozen after Phase 0.
- Dataloader plumbing (workers, prefetch): whatever saturates the GPU; not logged.
- Anything else: most conservative option + LOG.md entry (directive 1).

## 11. DEFINITION OF DONE

- [ ] `LOG.md` complete decision trail (gates, pivots, compute AND calendar cuts)
- [ ] `state.json` phase = DONE with full history
- [ ] `artifacts/runs.csv` — every (config, seed, T), all protocol §10 columns
- [ ] `artifacts/fits.json` — Part A + Part B, CIs, AICc, bootstrap distributions
- [ ] `artifacts/figures/` — 4 figures, SVG + PDF, real data
- [ ] `results.html` — house style, real numbers only
- [ ] `samples/` — per protocol §10
- [ ] `paper/main.tex` populated per OUTLINE.md (+ main.pdf if toolchain exists),
      anonymized, ≤4 pages main text
- [ ] `DECISION.md` — one outcome class, submission-readiness statement
- [ ] Nothing contradicts `configs/grid.json` or `protocol.html`

## 12. WHAT NOT TO DO

- Do not read numeric values out of `index.html`.
- Do not add hypotheses, metrics, or model forms after seeing data (labeled
  exploratory section excepted).
- Do not tune anything per-shape or per-T after Phase 1. Not the noise schedule,
  not temperature, not the unmasking schedule, not steps allocation across levels.
- Do not bootstrap at the surface-point level — run-level only (protocol §7.2).
- Do not exceed 500 GPU-h, miss the writing floor, fire P1-D or P2 twice, or train
  past a STOP condition.
- Do not exclude degenerate items from primary metrics.
- Do not de-anonymize the paper or cite this repo in it.
- Do not treat F1/F2 as failures to argue around; declare them and write the
  corresponding paper framing from protocol §7.4.
