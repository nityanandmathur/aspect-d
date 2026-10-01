# Reproducing ASPECT-D

*Refinement Buys Intelligibility, Search Buys Identity: What Test-Time Compute Buys in
Masked-Diffusion TTS* (DiffuLM @ NeurIPS 2026, Track 2).

This file covers three tiers of reproduction, the compute behind each, and which script
produces every table and figure in the paper. The recipes are in [`recipes/`](recipes/).
Each one is a `bash` script with `set -euo pipefail`. Machine-specific paths are
environment variables with defaults relative to the repo (see [`recipes/common.sh`](recipes/common.sh)).

| tier | what it reproduces | needs | time |
|---|---|---|---|
| **A** | every generated number, table and figure, re-derived from the released run records | CPU, Python 3.12, numpy/pandas/scipy/matplotlib | about 15 s, or about 10 min with `DEEP=1` (18-core laptop) |
| **B** | re-scores the released audio and checkpoints (WER, SIM-o, UTMOS, floors), then tier A | 1+ CUDA GPU, the eval models, `eval_zs.json` and the eval audio | ~1 GPU-h per 225 cells, plus model download |
| **C** | retrains from scratch: data, the 45-run grid, synthesis, scoring, fits | 8x B200-class GPUs, Emilia-EN access, ~1 TB disk | ~172 GPU-h for v1.0, ~350+ GPU-h with the extensions |

## Tier A: re-derive the paper on a CPU

```bash
TIER=cpu bash recipes/00_env.sh                      # venv at .venv (Python 3.12)
DATASET_JSON=<hf-repo>/dataset.json PAPER_DIR=paper bash recipes/reproduce_paper_cpu.sh
DEEP=1 DATASET_JSON=... bash recipes/reproduce_paper_cpu.sh   # also re-collect runs.csv and refit fits.json
```

`reproduce_paper_cpu.sh` works on a scratch copy. It never writes to the checkout.
It runs `recipes/05_fit_and_paper.sh` inside the copy, which calls `src/figures.py`,
`src/s0_figure.py`, `src/paper.py`, `src/paper_v14.py` and `src/paper_v15.py`. When they are
present it also runs the camera-ready generators `src/camera_ready_search.py` and
`src/camera_ready_scope.py`. It then diffs every
macro, table and figure against `PAPER_DIR`. Figures are compared by rendering both PDFs
and comparing pixels. Exit codes:

- `0`: everything matches.
- `1`: at least one MISMATCH. The report lists the macro, both values and the file.
- `2`: no mismatch, but some items cannot be regenerated from the release.

`DEEP=1` also runs `evaluate.py collect` to rebuild `artifacts/runs.csv` from
`runs/*/{run.json,synth_T*/synth.json,scores.json}`, and runs `fit.py` with the full
2,000-replicate bootstrap. The collect step needs `torch` and `soundfile` installed,
because `evaluate.py` imports them at module level.

**Status on the camera-ready sources (2026-10-01, ~22:10). Details are in "Known deviations".**
- 690 items verified identical: every macro of `numbers.tex`, `numbers_v14.tex` and the
  camera-ready `numbers_cr_*.tex` files, the 30k and 90k parts of `numbers_v15.tex`, every
  table, and all five figures, pixel for pixel. The count was 265 before the camera-ready
  `numbers_cr_*.tex` files existed. Every `\N...` macro used in `main.tex` and the
  `sections/*.tex` and `tab_*.tex` files it `\input`s is defined by a regenerated file.
- 0 mismatches.
- 9 items **UNVERIFIABLE**: the 180k-step row of Table `tab:trend` and the macros derived from
  it. The script therefore exits 2 (`ALLOW_UNVERIFIABLE=1` makes that 0).

## Tier B: re-score released audio and checkpoints

```bash
bash recipes/00_env.sh                      # full pinned stack (requirements.txt), CUDA
bash recipes/03_synthesize.sh               # only if you regenerate audio (checkpoints from HF)
RUNS_ROOT=runs bash recipes/04_score.sh     # scores.json per (run, T), floors, runs.csv
```

Run tiers B and C in a **scratch copy** of the repo. Several scripts write fixed artifact
paths in place, for example `s0_ledger.py` writes `artifacts-v1.2/identity_ledger.json`. The
committed artifacts must not be overwritten.

Released audio is not in git (`*.flac` and `tokens.npz` are gitignored). The checkpoints are on
the HF repo as `<cfg>_<seed>/model.safetensors`, with `config.json` and `run.json` beside them.
Scoring also needs `$ASPECTD_DATA/proc/eval_zs.json` (the 400 items) and
`$ASPECTD_DATA/proc/eval_audio/*.flac` (the original Emilia waveforms of each prompt and target).
**Neither is released yet** (see "Known deviations").

## Tier C: retrain from scratch

| step | recipe | produces | measured cost (B200) |
|---|---|---|---|
| 0 | `00_env.sh` | venv (requirements.txt), espeak-ng, eval models | |
| 1 | `01_data.sh` | 2,000 h Emilia-EN subset, phonemes, Mimi tokens, eval_zs | CPU + GPU Mimi encode (not recorded) |
| 2 | `02_train_grid.sh` | Phase-1 muP sweep (20 proxies x 3k steps) and the 45 runs | 6.9 + 162.5 GPU-h |
| 3 | `03_synthesize.sh` | T in {1,2,4,8,16} x 400 items for each run, integrity check | 2.4 GPU-h |
| 4 | `04_score.sh` | scores.json, floors, runs.csv | not recorded per cell |
| 5 | `05_fit_and_paper.sh` | fits.json, figures, numbers*.tex, tab_*.tex | CPU, about 10 min (refit) |
| 6 | `06_extensions.sh STAGE=...` | v1.1-v1.5 experiments | see below |

## Hardware and compute disclosure

The paper ran on one node with 8x NVIDIA B200 (183 GB each), 192 CPU cores, 2 TB RAM,
driver 595.71.05 and CUDA 13.2, using Python 3.12 and PyTorch 2.11 (LOG.md, 2026-08-05).
Every run record says `device: NVIDIA B200`. The GPU-hours below are sums of each job's
wall-clock time (`run.json gpu_hours`, `synth.json gpu_hours`). Several small runs often shared
one GPU, so these sums **over-count** exclusive GPU time. `state.json` records 65.0 h of true
GPU occupancy for the orchestrated jobs.

| block | source | GPU-h |
|---|---|---|
| Main grid, 45 runs x 30k steps (15 configs x 3 seeds) | HF `runs.csv` `train_gpu_hours` (one value per run) | **162.5** (A 27.8, B 57.3, C 77.4) |
| Main grid synthesis, 225 (run, T) cells | `runs.csv` `synth_gpu_hours` | 2.4 |
| Phase-1 muP LR sweep (20 proxies x 3k steps) | `runs/sweep_*/run.json` | 6.9 |
| **v1.0 total charged** | sum; matches LOG.md "171.8 GPU-h charged / 65 h occupancy" | **171.8** |
| v1.1: budget D (10 runs), 90k x 3, D3 LR sweep | `runs-v1.1/*/run.json` | 56.5 + 29.8 + 3.0 |
| v1.3: 90k x 6, C3_0 180k, variable prompt | `runs-v1.3/*/run.json` | 57.9 + 19.0 + 3.2 |
| extension synthesis (extended T, search/guidance/context arms) | `synth.json` in runs/, runs-v1.1/, runs-v1.3/ | 5.0 + 2.1 + 0.7 |
| v1.5: 8 further 180k runs, CFG arms, F5 anchor, XL run | `runs-v1.4/` and logs, **not in the release** | not derivable here |

**Paper discrepancy (needs an author decision).** The appendix says
"Total cost `\Ngpuhours` GPU-hours" and prints **143**. The macro comes from
`state.json gpu_hours.total = 142.88`, a snapshot taken at 2026-08-06 09:53 UTC, when 36 of
the 45 runs were done (LOG.md, G5 entry). The final records give 162.5 GPU-h of training for
the 45 grid runs, or 171.8 including the sweep and synthesis, and the extensions add at least
another 177 GPU-h. A brief may have said "143 GPU-hours for the main grid". The records do not
support that figure.

## Every table and figure in `main.tex`

Labels follow the camera-ready `main.tex`. "Verified" means `reproduce_paper_cpu.sh`
regenerated the item from committed records and it matched exactly.

| paper item | generated file / macros | script | input artifact(s) | upstream recipe | tier A |
|---|---|---|---|---|---|
| Fig. `fig:pipeline` | `fig1_pipeline.tex` (TikZ schematic, no data) | none | none | none | n/a |
| Table `tab:gapclosed` | `tab_gapclosed.tex`; `\NwerFloor`, `\NsimCeiling`, `\NgapRatioLo/Hi` (numbers_v14) | `src/paper_v14.py` | `artifacts-v1.4/analysis.json` (B_gap_closed) from `src/v14_analysis.py` over `artifacts-v1.4/runs_extended.csv`, `artifacts/g0c_groundtruth.json` (WER floor), `artifacts-v1.2/identity_ledger.json` (codec ceiling) | 03 (T 32/64 via `06 STAGE=v14`), 04 (floors) | verified |
| Table `tab:scope` | `tab_scope.tex` | `src/paper_v14.py` | `analysis.json` A_coordinate_sensitivity, B.coordinate_invariance | 06 v14 | verified |
| Table `tab:trend` | `tab_trend.tex`, `numbers_v15.tex` | `src/paper_v15.py` | `runs*/C{1,3,5}_{0,1,2}{,_90k,_180k}/synth_T{1,16}/{scores,asr2}.json`, `artifacts-v1.5/g0c_asr2.json` | 06 e6, t, v15 | 30k and 90k rows verified; **180k row unverifiable** |
| Fig. `fig:diag` (left, right) | `figures/aniso_contours_T16.pdf`, `figures/substitution_plane.pdf` | `src/figures.py` | `artifacts/runs.csv`, `artifacts/fits.json` | 02-05 | verified (pixel-identical) |
| Table `tab:ledger` | `\NtrainGainSim`, `\NtrainDouble`, `\NsearchVsDefault`, `\NsearchWERcost`, `\NparamGainSim`, `\NguideGainSim`, `\NguideWER`, `\NrateGainSim`, `\NctxWerCost`, `\NheadroomSim` | `src/paper.py` | `artifacts-v1.3/t23_training.json`, `artifacts-v1.2/{s2_confound,s2_search,identity_ledger,s4_guidance,s3_rate,s1_context}.json` | 06 t, s0-s4 | verified (the camera-ready prints the guidance row through `\NcrGuide*`, same values; see deviation 1) |
| Fig. `fig:ledger` | `figures/identity_ledger.pdf` | camera-ready: `src/camera_ready_search.py --paper-dir paper` (previously `src/s0_figure.py`, output in `artifacts-v1.2/figures/`) | `artifacts-v1.2/identity_ledger.json`, `artifacts-v1.3/multi_encoder.json`, search parts and `artifacts-camera/search_cost.json` | 06 s0, s2, menc | verified (both versions, against the figure current at the time) |
| camera-ready macros | `numbers_cr_search.tex`, `numbers_cr_scope.tex` | `src/camera_ready_search.py`, `src/camera_ready_scope.py --amp-boot 2000` | committed per-item scores, `artifacts-camera/{search_cost,scope_fit_boot}.json` | 05 (5d) | verified |
| Table `tab:search` | `\Nsrch{Lo,Mid,Hi}*`, `\NmaxNfour`, `\NscopeD*`, `\NscopeN*` | `src/paper.py` | `artifacts-v1.2/s2_search.json`, `artifacts-v1.1/e4_scale.json`, `artifacts-v1.3/t1_scope.json` | 06 s2, e4, t | verified |
| Table `tab:menc` | `tab_menc.tex`, `\Nmenc*` | `src/paper_v14.py` | `artifacts-v1.3/multi_encoder.json` | 06 menc | verified |
| Appendix grid (`app:grid`) | `appendix_grid.tex` | `src/paper.py` | `artifacts/runs.csv`, `configs/grid.json` | 02-04 | verified |
| Table `tab:prereg` | verdict text typed in `main.tex` (not generated) | none | see "Pre-registration ledger" | | not machine-checked |
| Table `tab:negative` | `\Ngam{Lo,Mid,Hi}*`, `\Nrate*` | `src/paper.py` | `artifacts-v1.2/s4_guidance.json`, `artifacts-v1.2/s3_rate.json` | 06 s4, s3 | verified (the camera-ready prints the guidance rows through `\NcrGam*`, same values; see deviation 1) |
| Table `tab:context` | `\Nctx*` | `src/paper.py` | `artifacts-v1.2/s1_context.json` | 06 s1 | verified |
| Table `tab:alloc` | `\Nalloc*` | `src/paper.py` | `artifacts-v1.1/e3_nfe.json` | 06 e3 | verified |
| Table `tab:exponents` | `\Nalpha*`, `\Nbeta*`, `\Ntau*`, `\Naicc*` | `src/paper.py` | `artifacts/fits.json` (from `src/fit.py`) | 05 (REFIT=1) | verified; a refit reproduces |
| Table `tab:robust` | `\Ndtaufour*`, `\Ndtaunineok*`, `\Ndtauasr`, `\NdtauAsrCI`, `\Ndtaulogamp`, `\NdtauLogCI` | `src/paper.py` | `artifacts-v1.1/{e4_scale,e6_undertraining,e5_robustness}.json` | 06 e4, e6, e5 | verified |
| Fig. `fig:main` | `figures/step_curves.pdf` (copy of `artifacts-v1.2/figures/step_curves.pdf`) | `src/figures.py` (current version, no raw-scale T* markers) | `artifacts/runs.csv`, `artifacts/fits.json` | 05 | verified |
| Extrapolation figure (H-D4; no `\label`) | `figures/extrapolation.pdf` | `src/figures.py` | `artifacts/fits.json` (hd4) | 05 | verified |
| prose macros (`\Ndtau`, `\Nruns`, `\Nhours`, ...) | `numbers.tex` | `src/paper.py` | as above, plus `state.json` and HF `dataset.json` | | verified (`\Ngpuhours` is stale, see above) |

`src/paper_sync.py` copies the five figures into `paper/figures/` (`--check-figures`,
`--refresh-figures`). The copies under `paper/figures/` are what the paper compiles.

## Seeds and determinism

- **Data selection.** RNG 1234 (`data.py SELECT_RNG`). The run uses 200 held-out speakers,
  a speaker-stratified round robin up to 2,000 h, 2,000 validation clips spread across speakers,
  and 400 eval items over 174 speakers, re-curated so that ground-truth Whisper WER is at most
  0.25 (LOG.md D-005). The same shard set gives the same subset. **The shard set itself is not
  recorded in the release.**
- **Training.** `BatchPlan(seed)` is a seed-keyed permutation per epoch, sorted by length within
  superbatches of 8x256. The batch order is identical across configs.
  - Initialisation: `torch.manual_seed(1000 + seed)`.
  - Masking: `Generator(seed*1000003 + step)`, identical across configs, and the gradient does not
    depend on how the batch is split into micro-batches (loss normalised over the 256-sequence
    effective batch, LOG.md P0-6).
  - Validation: `VAL_RNG = 999` with mask ratios {0.25, 0.5, 0.75}, so validation loss is comparable across runs.
  - Under DDP each rank takes a disjoint stride of the same batch (commit df5ee36).
  - bf16 with TF32 matmuls enabled, so results are **not bitwise reproducible** across hardware or kernels.
- **Sampler (frozen).** Fixed by `configs/grid.json -> backbone.sampler_frozen`, never tuned per shape or T:
  - MaskGIT confidence decoding, level by level over levels 1 to 8, with T steps per level
    (NFE = 8T), a cosine unmasking schedule, and Gumbel noise annealed from 1 to 0, temperature 1, no CFG.
  - Generation RNG `GEN_SEED_BASE = 20260805`, keyed by (batch, level, step) over fixed batches of 50.
    Every item therefore sees the same noise at the first step of each level for every T, config and seed.
  - The protocol §6.3 integrity check (`sample.py integrity`) passed for all 45 runs (`artifacts/integrity.json`).
- **Analysis.**
  - NLS uses 32 multi-starts (RNG 42). The run-level bootstrap has 2,000 replicates (RNG 7331) and
    clusters on seeds within configs.
  - `paper_v15.py` clusters its bootstrap on configuration and resamples items (RNG 7331, 1,000 replicates).
  - With numpy 2.4.6 and scipy 1.17.1, a CPU refit reproduces `fits.json` point estimates to about
    1e-5 relative. It also reproduces the Δτ CI to all printed digits and every verdict.
  - The unidentified Δρ CI (α sits at its bound) moves in the third decimal. It is not printed in the paper.

## Pre-registration ledger: where the verdicts live

| hypotheses | registered in | verdict recorded in |
|---|---|---|
| H-D1 to H-D4 | `PREREGISTRATION.md` | `DECISION.md`, `artifacts/fits.json -> decision` (outcome class S1) |
| H-E1 to H-E5 | `docs/preregistration/PREREGISTRATION-v1.1.md` | `artifacts-v1.1/e1_extended.json` (H_E1), `iso_latency.json` (H_E2), `e3_nfe.json` (H_E3), `e4_scale.json` (H_E4), `e6_undertraining.json` (H_E5); `LOG-v1.1.md` |
| H-S0 to H-S4 | `docs/preregistration/PREREGISTRATION-v1.2.md` | `docs/research-log/DECISION-v1.2.md`, `artifacts-v1.2/{identity_ledger,s1_context,s2_search,s3_rate,s4_guidance}.json` (H_S* fields); summary in `artifacts-v1.2/verdicts.json` |
| H-T1 to H-T3 | `docs/preregistration/PREREGISTRATION-v1.3.md` | `artifacts-v1.3/t1_scope.json` (H_T1), `t23_training.json` (H_T2, H_T3) |
| v1.5 CFG gate, compute trend | `docs/preregistration/PREREGISTRATION-v1.5.md` | `artifacts-v1.5/cfg_gate.json`, `RESULTS-FEED.md` |

Thresholds and gates G0 to G6 are in `protocol.html`. Each gate's measured values are logged in
`LOG*.md` and `state.json -> gates`.

## Known deviations and gaps (camera-ready, 2026-10-01)

1. **The S4 guidance macros in `numbers.tex` were regenerated for the camera-ready.**
   - `artifacts-v1.2/s4_guidance.json` was refit on 2026-08-13 (commit b6ec121). That was v1.5
     stage 1, which fixed an attended-PAD sampling bug and moved to an iso-NFE baseline.
   - The submitted version's `numbers.tex` (generated 2026-08-11) still printed the pre-refit
     values, for example γ=0.5: +0.0121 SIM / +2.73 WER points.
   - For the camera-ready, `numbers.tex` was regenerated with `src/paper.py` from the committed
     artifacts. Only the 14 lines `\NguideGainSim`, `\NguideWER` and `\Ngam{Lo,Mid,Hi}*`
     changed. They now match the artifact: γ=0.5 +0.0087 [0.0063, 0.0115] / +4.41,
     γ=1.0 −0.0021, γ=2.0 −0.0322, and the UTMOS columns.
   - The camera-ready text does not print those 14 macros. It uses `\NcrGuide*`
     (`numbers_cr_search.tex`) and `\NcrGam*` (`numbers_cr_extra.tex`), which carry the same
     values from the same artifact.
   - The verdict, H-S4 REFUTED, is the same before and after the refit.
     `artifacts-v1.2/verdicts.json` still carries the old γ=0.5 numbers.
2. **The 180k-step trend row cannot be reproduced from the release.** `paper_v15.py` reads
   `runs-v1.4/*_180k/synth_T{1,16}/{scores,asr2}.json`, but `runs-v1.4/` is in `.gitignore`.
   Only `runs-v1.3/C3_0_180k` is committed. The eight missing runs are C1_{0,1,2}, C3_{1,2} and
   C5_{0,1,2}. Committing those small JSON files would make Table `tab:trend` fully verifiable.
   The 30k and 90k rows reproduce exactly, CIs included (`recipes/lib/v15_partial.py`).
3. **`\Ngpuhours` = 143 is a mid-run snapshot** (see the compute disclosure above).
4. **The data needed for tiers B and C is not fully released.**
   - Missing: the list of the 71 Emilia/EN shards scanned in v1.0, `eval_zs.json` (400 item ids
     and texts), `heldout_speakers.json`, `index.parquet` and the token shards. Only
     `dataset.json` and `phone_vocab.json` are on the HF repo.
   - `01_data.sh` refuses to guess the shard list. Rebuilding the subset from different shards
     gives a different subset.
5. **The wav2vec2 floor on real speech has no producer.** `\NasrTwoFloor` = 0.1501 comes from
   `artifacts-v1.5/g0c_asr2.json`, and no committed script writes that file. `asr2.py` scores only
   synthesis directories.
6. **Some generator constants are typed in, not read from artifacts.** Each was checked against
   its artifact:
   - `paper_v15.py` hard-codes `WER_FLOOR = 0.0344830` and `SIM_CEIL = 0.5553695`. They match
     `g0c_groundtruth.json` and `identity_ledger.json` to 7 digits.
   - `v14_analysis.data_description()` hard-codes `train_hours 2000` and `train_speakers_min 4000`.
     So `\NtrainSpk` is the pre-registered minimum (4,000), not the measured 64,680 (`\Nspeakers`).
   - `paper.py` hard-codes `\NselCorrRange` (0.71--0.73), `\NsearchMaxN` (276) and `\Nbudgetc`.
   - `paper_v14.py` hard-codes `\NmencItems` (200) and `\NmencMinAuc` (0.98).
7. **`evaluate.py collect` scans `<repo>/runs/*` and now also returns T in {24,32,64}.** Keeping
   T ≤ 16 gives back the committed 225-row `artifacts/runs.csv` exactly.
8. **Python 3.12 or newer is required** (`paper.py` uses PEP 701 f-strings).
   `orchestrate.py`'s docstring lists `phase2`/`phase3` subcommands that do not exist. The real
   subcommands are `phase1`, `train --phase N`, `phase4`, `score`, `status` and `g2`/`g3`/`g5`/`g6`.

All GPU steps (tiers B and C) are documented from the code and the run records. None of them
were executed for this release. Only tier A was run end to end; see `recipes/` and the notes.
