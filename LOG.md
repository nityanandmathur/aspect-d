# ASPECT-D — append-only decision log

Every gate evaluation with measured values, every pivot/cut, every fallback,
every deviation. Canonical precedence: `configs/grid.json` → `protocol.html` →
`task.md` → `index.html`.

---

## 2026-08-05 21:39 UTC — session start (Phase 0)

- Fresh machine. `state.json` absent → starting from Phase 0 (no work to resume).
- Environment probe: 8× NVIDIA B200 (183 GB each, sm_100, driver 595.71.05,
  CUDA 13.2), 192 CPU cores, 1996 GB RAM, 976 GB free disk. No Python stack
  present (no pip/torch/espeak/ffmpeg/LaTeX).
- Installed: apt `espeak-ng`, `ffmpeg`, `libsndfile1`, `build-essential`,
  `git-lfs`; venv at `/home/ubuntu/venv` with torch (default PyPI wheels,
  CUDA 13 build), torchaudio, transformers, datasets, huggingface_hub,
  soundfile, librosa, scipy, numpy, pandas, matplotlib, jiwer, einops,
  phonemizer, webdataset. Verified bf16 matmul on device 0, capability (10, 0).
- HF token from `.env` authenticates as user `nityanandmathur`;
  `amphion/Emilia-Dataset` (gated: auto) is readable → **data.primary
  (Emilia-EN) is available; no fallback needed.** Single-stream throughput
  37 MB/s, 10 parallel streams used.

### Deviation D-001 (declared, cannot bias H-D2/H-D3): CLAUDE.md Rule 6 token budget
`CLAUDE.md` Rule 6 sets a 4,000-token per-task / 30,000-token per-session
budget. Executing `task.md` (Phases 0–7 end-to-end, autonomously, ~50 files and
~50 training runs) cannot fit in 30,000 tokens. Per `CLAUDE.md` Rule 12 (fail
loud) the breach is surfaced here rather than silently absorbed; per Rule 7 the
more specific and more recent instruction (`task.md` mission + the operator's
explicit "implement it end-to-end" directive) wins over the generic budget.
No experimental consequence.

### Phase-0 implementation decisions (values absent from grid.json/protocol.html)
Recorded per directive §0.1; all are frozen before any training run, applied
identically to every config, seed, and T, and therefore cannot bias H-D2/H-D3
(which compare shapes and T under one fixed harness).

- **P0-1 Sequence layout.** Phoneme tokens are prepended (grid.json:
  "always visible, prepended"), then the audio frame positions. RoPE positions
  are `i` for phoneme index `i` and `512 + f` for audio frame `f` (fixed offset,
  512 = phoneme-length cap). Rationale: audio positions must not depend on the
  phoneme padding length, so that a run's numerics are invariant to
  micro-batch composition (micro-batch size varies with model size under fixed
  effective batch 256).
- **P0-2 Prompt/target construction.** Training: each clip (4–18 s, so
  ≤ prompt 3 s + max_target 15 s) contributes its full phoneme transcript
  (always visible) and its full frame grid; the first 37 frames
  (3 s × 12.5 Hz, grid.json prompt_seconds) are the always-visible prompt at all
  8 levels; the remaining frames are the target region subject to masking.
  Clips longer than 18 s are excluded rather than truncated (truncation would
  break the text↔audio length correspondence the length conditioning depends
  on). Evaluation (`eval_zs`, zero-shot cloning): the prompt is a *complete*
  held-out-speaker clip of 2.5–3.5 s with its own transcript, the target text is
  a different clip (4–15 s) of the same held-out speaker; visible phonemes =
  phonemize(prompt text + target text); generated length =
  `chars(target_text) · sec_per_char · 12.5` frames (grid.json). This is the
  standard cross-sentence zero-shot protocol (seed-tts-eval); it needs no
  word-level alignment, which the corpus does not provide.
- **P0-3 Generation RNG matched across T.** eval_zs items are processed in one
  fixed order (sorted by item id) with a fixed batch composition (50 items per
  batch, identical for every run and every T; item target lengths depend only on
  the text and the frozen `sec_per_char`, so even the padded tensor shapes are
  identical). All sampler randomness (categorical token draws and Gumbel
  confidence noise) is drawn from a generator seeded by
  `GEN_SEED_BASE + batch_index·10^6 + level·10^3 + step`, with `GEN_SEED_BASE`
  frozen at 20260805. Consequence: for a given item, the noise at the first step
  of every level is *identical* across all T, all configs and all seeds, and no
  result depends on batch composition. An exactly-matched stream for *all* steps
  is impossible in principle (T changes the number of draws); this is the
  strongest available realization of grid.json's "matched generation RNG per item
  across T".
- **P0-4 Gumbel noise annealing at T=1.** grid.json anneals the confidence-noise
  scale linearly 1.0 → 0.0 "across the T steps"; implemented as
  `scale_i = 1 − i/(T−1)` for `T > 1` and `scale_0 = 1.0` for `T = 1` (where the
  cosine schedule commits every cell in the single step, so the confidence
  ordering — and hence the noise — has no effect on the output).
- **P0-5 μP realization.** base_width 256 (grid.json). Multiplier m = w/256.
  Embeddings (phoneme + 8 codebook tables) and RMSNorm gains: init std 0.02 /
  gain 1, LR = base LR. Hidden weights (q,k,v,o,up,down): init std
  0.02/√m, LR = base LR / m. Output heads: init std 0.02/√m, logits multiplied
  by 1/m, LR = base LR / m. Residual (o, down) init additionally scaled by
  1/√(2·depth) (grid.json residual_scale). Attention softmax keeps the standard
  1/√head_dim scale because head_dim is fixed at 64 for every config, so the
  μP attention-scale correction is a constant here. Width transfer is *measured*
  at G1, depth transfer at G1b.
- **P0-6 Effective-batch normalization.** The 256-sequence effective batch is
  split into micro-batches sized per config; the loss is normalized by the total
  number of masked target cells in the whole effective batch (counts computed
  before the forward passes), so the gradient does not depend on how the batch
  was split. Masking draws (level per example, mask ratio per example, cell
  Bernoulli) are made for the whole effective batch from a stream keyed by
  (seed, step), hence identical across configs.
- **P0-7 No `torch.compile`.** Eager + SDPA + bf16 autocast only. Padding to
  static shapes for compilation would waste ~25% of the compute or trigger
  repeated recompiles; total projected training cost is far below the G5 cap, so
  speed is not the binding constraint and the simpler, more auditable path wins.
- **P0-8 Loudness / defaults.** Peak-normalize every waveform to −1 dBFS before
  Mimi encode (task.md §10 default). Val-loss smoothing for LR selection: EMA
  over the last 3 val points (task.md §10). Bootstrap RNG 7331, NLS multi-start
  RNG 42, data-selection RNG 1234 (grid.json / task.md §10).

## 2026-08-06 00:20 UTC — pre-training harness review (Phase 0, before any grid run)

An adversarial review of the harness against `grid.json` / `protocol.html` (five
independent reviewers by dimension, each finding re-verified by a skeptic
instructed to refute it) produced 10 confirmed defects, all fixed before the
first training run. The four that would have changed the science:

1. **Sampler (blocker).** Already-committed cells were re-sampled at every
   refinement step, and the final step (cosine schedule → every cell committed)
   re-drew the whole level. grid.json says "categorical at temperature 1.0 for
   *newly committed* tokens". Effect if shipped: T would have been largely
   cosmetic and τ / κ — the primary quantities — meaningless. Fixed by freezing
   committed cells (`new_c & ~committed`).
2. **Init RNG (blocker).** `torch.manual_seed` was never called, so model init
   came from a nondeterministic process seed: runs were not reproducible and the
   five LR points of a Phase-1 sweep differed by init noise as well as LR. Fixed
   (init now keyed on the run seed) → **P0-9** below.
3. **SIM-o batching (blocker).** The SV stack has no padding mask, so
   zero-padded batches shifted the embedding and therefore the primary identity
   metric. Now one utterance per forward pass → **D-004** below.
4. **Bootstrap weights (blocker).** `surface()` recomputed 1/SE² inside every
   bootstrap replicate; a replicate that draws the same seed twice has zero
   within-config spread, so it received an unbounded weight. protocol §7.2
   resamples the *runs*, not the weighting scheme — weights are now pinned to the
   observed-data SEs.

Also fixed: G3's "one restart from scratch at 0.5× LR" was unimplemented (a
diverged run was silently resumed at the same LR); G3 divergence state reset on
resume; the §6.3 sampler-integrity check was never invoked before scoring; fits
were not restricted to the active configs; τ and κ were coupled through each
other's convergence; a missing bootstrap CI would have been reported as the F1
clean negative; H-D claims were emitted even when G4 fails; the val loss covered
1,792 of 2,000 clips (dropping the longest); `c_layer` charged fixed per-forward
overhead to every layer.

### Data-selection corrections (same review)
- **Val split.** Was the id-sorted head of the leftover pool → 2,000 clips from
  ~18 of 64,680 training speakers, a poor instrument for LR selection and
  divergence detection. Now one clip per speaker per pass: **2,000 clips from
  2,000 distinct speakers**. Training clips are unchanged (2,000.0 h, 792,064
  clips, 64,680 speakers).
- **eval_zs target window.** `grid.json` sets eval_zs `target_seconds [4, 15]`,
  but the *generated* length is `chars(target_text)·sec_per_char·12.5`; filtering
  only the source clip admitted low-character-density clips whose synthesis was
  far below 4 s. The window is now applied to the predicted generated length as
  well: measured 4.03–14.93 s over all 400 items (186 held-out speakers supply
  them; 200 remain held out of training).

### New decisions recorded
- **P0-9 Init RNG.** Model init is drawn from the global torch generator seeded
  with `1000 + seed` immediately before construction, so a run is fully
  determined by (w, d, heads, seed). Init is the only global-generator consumer;
  batch order, training masking and val masking each use their own keyed streams,
  so P0-6's cross-config matching is untouched. The five LR points of a sweep now
  share one init (LR is the only difference at G1/G1b); seeds 0/1 still differ in
  init, preserving the seed-to-seed SD that G4 and the §7.1 weights depend on.
- **D-004 SIM-o batch size 1.** Embedding one utterance per forward pass costs
  ~25 s per (run, T) and removes a padding-dependent bias of the primary metric.
- **P5-1 κ bounds.** `protocol.html` bounds E, A, B, C and the exponents but not
  κ. A one-sided bound (κ ≥ 0) would make "κ > 0 with CI excluding 0" partly
  self-fulfilling, so κ ∈ [−3, 3]; the fit reports whether κ landed on a bound.
- **P5-2 SE floor.** Weights are 1/SE² with SE over seeds (pooled where a config
  has one seed), floored at 0.25 × the pooled seed SD — scale-free, so a chance
  agreement between two seeds cannot dominate the fit and the floor cannot
  quietly replace the weighting with a constant. The number of floored points is
  reported in `fits.json`.
- **P5-3 Part-B weights.** `protocol.html` specifies weights only for Part A; the
  same 1/SE² scheme is used for Part B (the consistent reading), with an
  unweighted refit reported as an exploratory sensitivity check.
- **D-003 Encoder GPU pinning.** The first `encode` pass assigned a GPU per job;
  since pool workers outlive jobs, several processes opened contexts on one device
  and Mimi's wide 24 kHz activations exhausted it. Now one GPU and one Mimi
  instance per worker process, groups of 8 clips. Not data-affecting: Mimi codes
  were verified padding- and batch-invariant (agreement 1.0000), and every shard
  was re-encoded from scratch afterwards.

- **G6** 2026-08-05 22:44 UTC: next milestone M1_env_mup_done due 2026-08-12; 24 days to the 2026-08-29 AoE wall; phase 0 → on track, no calendar cut applied.

- **G5** 2026-08-05 22:44 UTC: used 0.0 GPU-h (GPU-occupancy 0.0 h), 0 runs done at 0.00 h/run, 30 to go → projected **22 / 500.0 GPU-h** → within cap.

### Defect found by the pre-training smoke test (not by review): NaN Gumbel noise
`src/sample.py::_gumbel` was written as
`-torch.log(-torch.log(u.clamp_min(1e-20)).clamp_min(1e-20))`. Python applies
`.clamp_min` to the *inner* `torch.log(u)` before the unary minus, so the
negative inner log was clamped to +1e-20, negated, and passed to `log` — every
Gumbel sample was NaN. Since `argmax` over an all-NaN row returns index 0, every
sampled token was codebook entry 0 and **T had no effect at all**: the first
end-to-end run of the sampler on an untrained model gave 0.0 % differing cells
between T=1 and T=16, i.e. an immediate protocol §6.3 failure. Fixed by
parenthesising and clamping u strictly inside (0,1):
`-torch.log(-torch.log(u.clamp(1e-20, 1 - 1e-7)))`.
After the fix, on an untrained A5: T=1 vs T=4 → 92.8 %, T=1 vs T=16 → 99.1 %,
T=4 vs T=16 → 95.3 % differing generated cells; prompt frames preserved exactly
at every T. This is why §6.3 exists, and it is the second sampler defect caught
before any training run.

## 2026-08-05 22:52 UTC — Gate G0 (Phase 0 environment)

All five G0 checks plus the task.md param-count test, measured values:

| check | measured | threshold | verdict |
|---|---|---|---|
| (a) Mimi roundtrip, 10 clips | mel-L1 0.966 vs cross-clip control 4.046; all audible | roundtrip ≪ control | **PASS** |
| (b) 200-step single-batch overfit | masked CE 7.640 → 0.169 (**97.8 %** reduction) | ≥ 40 % | **PASS** |
| (c) eval harness on ground-truth audio | per-item WER **3.45 %**, corpus 3.66 %; same-speaker SIM-o median **0.701**; cross-speaker median **0.034**; UTMOS(GT) 3.33 | WER ≤ 5 %, same ≥ 0.50, cross ≤ 0.25 | **PASS** (after repair, below) |
| (d) sec_per_char | **0.06020** s/char (median over 10,000 training clips; grid.json fallback 0.075 unused) | computed and logged | **PASS** |
| (e) sampler integrity, untrained model | **99.0 %** of generated cells differ between T=1 and T=16 | > 20 % | **PASS** |
| param count (A5, B3, C1) | deviation from grid.json 0.067 %, 0.034 %, 0.015 % | ≤ 1 % | **PASS** |

Data as built: 2,000.0 h / 792,064 clips / 64,680 speakers training (≥ 4,000
required), 2,000 val clips over 2,000 speakers, 200 held-out speakers,
400 eval_zs items; 1,997.7 h of Mimi tokens encoded (4 clips lost to decode
errors, 0.0005 %). Phoneme vocabulary 159 symbols, frozen from the training
split, 0 out-of-vocabulary occurrences outside train.

### Repair R-1 (one repair attempt used of the three G0 allows): eval-set curation
First G0(c) evaluation **failed on WER only**: per-item 5.91 %, corpus 5.46 %
(> 5 %), with same/cross SIM-o already passing (0.698 / 0.032). Diagnosis, not
guesswork: the per-item WER median was **0.000** and 7 % of items carried 48 % of
the total error. Inspecting the worst items showed Emilia's EN split contains
mislabelled non-English clips — Japanese audio whose "transcript" is romaji, so
Whisper (correctly) returns Japanese script and WER is 1.00 — plus a few clips
whose transcript does not match the audio. On those items the reference text
measures nothing, so WER on generated speech would be pure noise.

Repair: `data.py recurate_eval` scores the ground-truth audio of **all 6,810
held-out-speaker candidate clips** with the frozen eval ASR and keeps only clips
with ground-truth WER ≤ 0.25, for the prompt (whose transcript is part of the
visible conditioning) as well as the target. eval_zs was then rebuilt with the
same deterministic round-robin: **400 items over 174 held-out speakers**, mean
ground-truth WER of the selected targets **0.0345**. Re-run G0(c): per-item WER
3.45 %, same-speaker SIM-o 0.701, cross-speaker 0.034 → **PASS**.
This filter depends only on the corpus and the frozen ASR — never on a trained
model, a shape, or T — so it cannot bias H-D2/H-D3; it makes the WER instrument
valid rather than easier (**D-005**).

### Measured c_layer(width) — protocol §6.5
Batch 1 on a B200, 200 phoneme + 224 frame positions, c_layer taken as the
per-layer *slope* between depth 4 and depth 12 (so fixed per-forward overhead is
not charged to every layer):

| width | 256 | 320 | 384 | 448 | 512 | 640 | 768 | 832 | 960 | 1152 |
|---|---|---|---|---|---|---|---|---|---|---|
| c_layer (ms) | 0.497 | 0.510 | 0.514 | 0.500 | 0.493 | 0.506 | 0.498 | 0.499 | 0.493 | 0.502 |

**c_layer is flat in width** (0.493–0.514 ms, ±2 %) over the whole active grid:
at batch 1 a B200 is entirely kernel-launch bound at these sizes, so serial
latency is `8·T·d·0.50 ms` and *width is free*. This is a measured property of
this GPU, not an assumption, and it is what the design-rule table must use
(grid.json latency_model). It sharpens the serving corollary: on this hardware
the width/depth split is decided by quality alone until batching makes the GEMMs
compute-bound.

### D-006 Micro-batch sizing (memory, not numerics)
The first Phase-1 launch OOMed with three co-tenant runs per GPU: the micro-batch
proxy cap allowed 256-sequence micro-batches, peaking near 55 GiB. The cap was
lowered so a run peaks near 20 GiB (measured: d12/w640 19.8 GiB at 0.276 s/step,
d18/w768 18.9 GiB at 0.406 s/step, d36/w512 23.8 GiB at 0.494 s/step). This
changes only how the 256-sequence effective batch is split; the loss is
normalised over the whole effective batch, so gradients are unchanged (P0-6).

## 2026-08-05 23:47 UTC — Gate G1 / G1b (Phase 1, μP transfer)

5-point sweeps ([0.001, 0.002, 0.004, 0.008, 0.016]), 3000-step proxies, EMA over last 3 val points.

| sweep | argmin LR | val(EMA) per LR |
|---|---|---|
| g1_w256 (w=256, d=12) | 0.004 | 0.001:5.7347, 0.002:5.6524, 0.004:5.6479, 0.008:5.6844, 0.016:5.8016 |
| g1_w640 (w=640, d=12) | 0.002 | 0.001:5.7367, 0.002:5.6395, 0.004:5.6657, 0.008:5.9058, 0.016:6.3758 |
| g1b_d4 (w=384, d=4) | 0.002 | 0.001:5.686, 0.002:5.6001, 0.004:5.6021, 0.008:5.654, 0.016:5.6986 |
| g1b_d24 (w=384, d=24) | 0.002 | 0.001:5.7459, 0.002:5.6717, 0.004:5.711, 0.008:5.8094, 0.016:6.2726 |

- **G1 (width transfer)**: argmin(w=256)=0.004, argmin(w=640)=0.002 → within a factor
  of 2: **PASS**.
- **G1b (depth transfer)**: argmin(d=4)=0.002, argmin(d=24)=0.002 → within a factor
  of 2: **PASS**.
- Chosen LR rule: muP single base LR from the (d=12, w=256) base-width sweep; width and depth transfer verified at G1/G1b → base LR **0.004**.
- Phase-1 compute: 6.93 GPU-h (cumulative 6.89 / 500.0).

- **G5** 2026-08-05 23:47 UTC: used 6.9 GPU-h (GPU-occupancy 0.0 h), 0 runs done at 0.00 h/run, 30 to go → projected **29 / 500.0 GPU-h** → within cap.

- **G6** 2026-08-05 23:47 UTC: next milestone M1_env_mup_done due 2026-08-12; 24 days to the 2026-08-29 AoE wall; phase 1 → on track, no calendar cut applied.

### Note on the chosen base LR (no rule change)
Three of the four sweeps put the argmin at 0.002 and the base-width sweep at 0.004,
with 0.002 and 0.004 separated by only 0.0045 nats at the base width (5.6524 vs
5.6479) — the optimum is broad and flat. The LR rule was fixed **before** the
sweeps ran (state.json `lr_rule`: μP single base LR from the (d=12, w=256)
base-width proxy, since that is the shape μP is anchored to) and is applied here
unchanged: **base LR 0.004** for every grid run. Re-selecting 0.002 after seeing
the other three sweeps would be a post-hoc edit to a pre-registered procedure for
a difference far inside the noise, and both values sit inside the factor-of-2 band
that G1/G1b certify.

### Coordinate check (protocol §5 prerequisite to G1)
Per-block activation RMS logged for the first 50 steps at w=256 and w=640
(d=12, LR 0.004). Mean over blocks at step 49: **0.315 (w=256) vs 0.303 (w=640)**
— a 4 % difference, i.e. no width-dependent drift; per-block values track each
other across the whole stack (first block 0.047 in both, last block 0.652 vs
0.614). Growth from the initialisation scale (0.04–0.06) over the first 50 steps
is ordinary early-training behaviour, not a scale blow-up. No bug to fix.
