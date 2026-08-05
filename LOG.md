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
