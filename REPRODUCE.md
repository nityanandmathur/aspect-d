# Reproducing ASPECT-D

*Refinement Buys Intelligibility, Search Buys Identity: What Test-Time Compute Buys in
Masked-Diffusion TTS* (DiffuLM @ NeurIPS 2026, Track 2).

This file covers three tiers of reproduction, the compute behind each, and which script
produces every table and figure in the paper. The recipes are in [`recipes/`](recipes/).
Each one is a `bash` script with `set -euo pipefail`. Machine-specific paths are
environment variables with defaults relative to the repo (see [`recipes/common.sh`](recipes/common.sh)).
Paths below are relative to the repo root. Measured outputs are under `results/`; the
research log and orchestrator state (`LOG.md`, `state.json`, `DECISION.md`, ...) are under
`archive/research-log/`; the project pages are under `docs/`.

The paper's LaTeX sources are in their own repository,
[aspect-d-paper](https://github.com/nityanandmathur/aspect-d-paper). The scripts find a
checkout of it through `ASPECTD_PAPER_DIR`, which defaults to `aspect-d-paper` next to this
repo, so to rebuild the paper numbers clone both repositories side by side (or set
`ASPECTD_PAPER_DIR`). File names such as `main.tex`, `numbers.tex` or `figures/step_curves.pdf`
below are paths in the paper sources. The camera-ready PDF is
[`docs/assets/paper.pdf`](docs/assets/paper.pdf), also linked from the project page.

| tier | what it reproduces | needs | time |
|---|---|---|---|
| **A** | every generated number, table and figure, re-derived from the released run records | CPU, Python 3.12, numpy/pandas/scipy/matplotlib | about 15 s, or about 10 min with `DEEP=1` (18-core laptop) |
| **B** | re-scores the released audio and checkpoints (WER, SIM-o, UTMOS, floors), then tier A | 1+ CUDA GPU, the eval models, `eval_zs.json` and the eval audio | ~1 GPU-h per 225 cells, plus model download |
| **C** | retrains from scratch: data, the 45-run grid, synthesis, scoring, fits | 8x B200-class GPUs, Emilia-EN access, ~1 TB disk | ~172 GPU-h for v1.0, ~350+ GPU-h with the extensions |

## Tier A: re-derive the paper on a CPU

```bash
git clone https://github.com/nityanandmathur/aspect-d-paper.git ../aspect-d-paper   # paper sources, next to the repo
TIER=cpu bash recipes/00_env.sh                      # venv at .venv (Python 3.12)
DATASET_JSON=<hf-repo>/dataset.json bash recipes/reproduce_paper_cpu.sh
DEEP=1 DATASET_JSON=... bash recipes/reproduce_paper_cpu.sh   # also re-collect runs.csv and refit fits.json
```

`PAPER_DIR`, the paper sources the script checks against, defaults to `$ASPECTD_PAPER_DIR`,
else `../aspect-d-paper`. If you set neither and `../aspect-d-paper` does not exist, the
script clones the paper repository (depth 1) into its scratch directory and says so. A
`PAPER_DIR` or `ASPECTD_PAPER_DIR` that you set must exist.

`reproduce_paper_cpu.sh` works on scratch copies of this repo and of `PAPER_DIR`. It never
writes to either checkout. It runs `recipes/05_fit_and_paper.sh` inside the copies, which
calls `src/figures.py`, `src/s0_figure.py`, `src/paper.py`, `src/paper_v14.py` and
`src/paper_v15.py`. It then diffs exactly what `PAPER_DIR/main.tex` uses against
`PAPER_DIR`: every macro of the `numbers*.tex` files it `\input`s (`numbers.tex`,
`numbers_v14.tex`, `numbers_v15.tex`), every `tab_*.tex` and `appendix_grid.tex` it
`\input`s, and every figure it `\includegraphics`. Figures are compared by rendering both
PDFs and comparing pixels. Exit codes:

- `0`: everything matches.
- `1`: at least one MISMATCH. The report lists the macro, both values and the file.
- `2`: no mismatch, but some items cannot be regenerated from the release.

`DEEP=1` also runs `evaluate.py collect` to rebuild `results/artifacts/runs.csv` from
`results/runs/*/{run.json,synth_T*/synth.json,scores.json}`, and runs `fit.py` with the full
2,000-replicate bootstrap. The collect step needs `torch` and `soundfile` installed,
because `evaluate.py` imports them at module level.

**Status on the camera-ready sources (2026-10-01). Details are in "Known deviations".**
- 279 items verified identical: the 207 macros of `numbers.tex`, the 47 of `numbers_v14.tex`,
  14 of the 22 in `numbers_v15.tex` (the 30k and 90k parts), the four tables `tab_gapclosed`,
  `tab_scope`, `tab_menc` and `appendix_grid`, the 30k and 90k rows of `tab_trend`, and all
  five figures, pixel for pixel. Every `\N...` macro used in `main.tex` and the `tab_*.tex`
  files it `\input`s is defined by a regenerated file.
- 0 mismatches.
- 9 items **UNVERIFIABLE**: the 180k-step row of Table `tab:trend` and the macros derived from
  it. The script therefore exits 2 (`ALLOW_UNVERIFIABLE=1` makes that 0).

## Tier B: re-score released audio and checkpoints

```bash
bash recipes/00_env.sh                      # full pinned stack (requirements.txt), CUDA
bash recipes/03_synthesize.sh               # only if you regenerate audio (checkpoints from HF)
RUNS_ROOT=results/runs bash recipes/04_score.sh   # scores.json per (run, T), floors, runs.csv
```

Run tiers B and C in a **scratch copy** of the repo. Several scripts write fixed artifact
paths in place, for example `s0_ledger.py` writes `results/artifacts-v1.2/identity_ledger.json`. The
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
| 5 | `05_fit_and_paper.sh` | fits.json, figures, numbers*.tex and tab_*.tex (in `$ASPECTD_PAPER_DIR`) | CPU, about 10 min (refit) |
| 6 | `06_extensions.sh STAGE=...` | v1.1-v1.5 experiments | see below |

## Hardware and compute disclosure

The paper ran on one node with 8x NVIDIA B200 (183 GB each), 192 CPU cores, 2 TB RAM,
driver 595.71.05 and CUDA 13.2, using Python 3.12 and PyTorch 2.11 (`archive/research-log/LOG.md`, 2026-08-05).
Every run record says `device: NVIDIA B200`. The GPU-hours below are sums of each job's
wall-clock time (`run.json gpu_hours`, `synth.json gpu_hours`). Several small runs often shared
one GPU, so these sums **over-count** exclusive GPU time. `archive/research-log/state.json` records 65.0 h of true
GPU occupancy for the orchestrated jobs.

| block | source | GPU-h |
|---|---|---|
| Main grid, 45 runs x 30k steps (15 configs x 3 seeds) | HF `runs.csv` `train_gpu_hours` (one value per run); this is the paper's `\Ngpuhours` | **162.5** (A 27.8, B 57.3, C 77.4) |
| Main grid synthesis, 225 (run, T) cells | `runs.csv` `synth_gpu_hours` | 2.4 |
| Phase-1 muP LR sweep (20 proxies x 3k steps) | `results/runs/sweep_*/run.json` | 6.9 |
| **v1.0 total charged** | sum; matches `LOG.md` "171.8 GPU-h charged / 65 h occupancy" | **171.8** |
| v1.1: budget D (10 runs), 90k x 3, D3 LR sweep | `results/runs-v1.1/*/run.json` | 56.5 + 29.8 + 3.0 |
| v1.3: 90k x 6, C3_0 180k, variable prompt | `results/runs-v1.3/*/run.json` | 57.9 + 19.0 + 3.2 |
| extension synthesis (extended T, search/guidance/context arms) | `synth.json` in results/runs/, results/runs-v1.1/, results/runs-v1.3/ | 5.0 + 2.1 + 0.7 |
| v1.5: 8 further 180k runs, CFG arms, F5 anchor, XL run | `results/runs-v1.4/` and logs, **not in the release** | not derivable here |

The paper's appendix ("Total cost `\Ngpuhours` GPU-hours") prints **162.5**, the grid's
training compute: `src/paper.py` sums `train_gpu_hours` over the 45 runs of
`results/artifacts/runs.csv`. (The submitted version printed 143, from `state.json
gpu_hours.total = 142.88`, a snapshot taken at 2026-08-06 09:53 UTC when 36 of the 45 runs
were done.) The other rows of this table come from the run records, not from the paper.

## Every table and figure in `main.tex`

Labels follow the camera-ready `main.tex`. "Verified" means `reproduce_paper_cpu.sh`
regenerated the item from committed records and it matched exactly.

| paper item | generated file / macros | script | input artifact(s) | upstream recipe | tier A |
|---|---|---|---|---|---|
| Fig. `fig:pipeline` | `fig1_pipeline.tex` (TikZ schematic, no data) | none | none | none | n/a |
| Table `tab:gapclosed` | `tab_gapclosed.tex`; `\NwerFloor`, `\NsimCeiling`, `\NgapRatioLo/Hi` (numbers_v14) | `src/paper_v14.py` | `results/artifacts-v1.4/analysis.json` (B_gap_closed) from `src/v14_analysis.py` over `results/artifacts-v1.4/runs_extended.csv`, `results/artifacts/g0c_groundtruth.json` (WER floor), `results/artifacts-v1.2/identity_ledger.json` (codec ceiling) | 03 (T 32/64 via `06 STAGE=v14`), 04 (floors) | verified |
| Table `tab:scope` | `tab_scope.tex` | `src/paper_v14.py` | `analysis.json` A_coordinate_sensitivity, B.coordinate_invariance | 06 v14 | verified |
| Table `tab:trend` | `tab_trend.tex`, `numbers_v15.tex` | `src/paper_v15.py` | `results/runs*/C{1,3,5}_{0,1,2}{,_90k,_180k}/synth_T{1,16}/{scores,asr2}.json`, `results/artifacts-v1.5/g0c_asr2.json` | 06 e6, t, v15 | 30k and 90k rows verified; **180k row unverifiable** |
| Fig. `fig:diag` (left, right) | `figures/aniso_contours_T16.pdf`, `figures/substitution_plane.pdf` | `src/figures.py` | `results/artifacts/runs.csv`, `results/artifacts/fits.json` | 02-05 | verified (pixel-identical) |
| Table `tab:ledger` | `\NtrainGainSim`, `\NtrainDouble`, `\NsearchVsDefault`, `\NsearchWERcost`, `\NparamGainSim`, `\NguideGainSim`, `\NguideWER`, `\NrateGainSim`, `\NctxWerCost`, `\NheadroomSim` | `src/paper.py` | `results/artifacts-v1.3/t23_training.json`, `results/artifacts-v1.2/{s2_confound,s2_search,identity_ledger,s4_guidance,s3_rate,s1_context}.json` | 06 t, s0-s4 | verified |
| Fig. `fig:ledger` | `figures/identity_ledger.pdf` (copy of `results/artifacts-v1.2/figures/identity_ledger.pdf`) | `src/s0_figure.py` (bars read the same artifact fields as the `tab:ledger` macros) | `results/artifacts-v1.2/{identity_ledger,s2_confound}.json`, `results/artifacts-v1.3/t23_training.json` | 06 s0, s2, t | verified (pixel-identical) |
| Table `tab:search` | `\Nsrch{Lo,Mid,Hi}*`, `\NmaxNfour`, `\NscopeD*`, `\NscopeN*` | `src/paper.py` | `results/artifacts-v1.2/s2_search.json`, `results/artifacts-v1.1/e4_scale.json`, `results/artifacts-v1.3/t1_scope.json` | 06 s2, e4, t | verified |
| Table `tab:menc` | `tab_menc.tex`, `\Nmenc*` | `src/paper_v14.py` | `results/artifacts-v1.3/multi_encoder.json` | 06 menc | verified |
| Appendix grid (`app:grid`) | `appendix_grid.tex` | `src/paper.py` | `results/artifacts/runs.csv`, `configs/grid.json` | 02-04 | verified |
| Table `tab:prereg` | verdict text typed in `main.tex` (not generated) | none | see "Pre-registration ledger" | | not machine-checked |
| Table `tab:negative` | `\Ngam{Lo,Mid,Hi}*`, `\Nrate*` | `src/paper.py` | `results/artifacts-v1.2/s4_guidance.json`, `results/artifacts-v1.2/s3_rate.json` | 06 s4, s3 | verified |
| Table `tab:context` | `\Nctx*` | `src/paper.py` | `results/artifacts-v1.2/s1_context.json` | 06 s1 | verified |
| Table `tab:alloc` | `\Nalloc*` | `src/paper.py` | `results/artifacts-v1.1/e3_nfe.json` | 06 e3 | verified |
| Table `tab:exponents` | `\Nalpha*`, `\Nbeta*`, `\Ntau*`, `\Naicc*` | `src/paper.py` | `results/artifacts/fits.json` (from `src/fit.py`) | 05 (REFIT=1) | verified; a refit reproduces |
| Table `tab:robust` | `\Ndtaufour*`, `\Ndtaunineok*`, `\Ndtauasr`, `\NdtauAsrCI`, `\Ndtaulogamp`, `\NdtauLogCI` | `src/paper.py` | `results/artifacts-v1.1/{e4_scale,e6_undertraining,e5_robustness}.json` | 06 e4, e6, e5 | verified |
| Fig. `fig:main` | `figures/step_curves.pdf` (copy of `results/artifacts-v1.2/figures/step_curves.pdf`) | `src/figures.py` (current version, no raw-scale T* markers) | `results/artifacts/runs.csv`, `results/artifacts/fits.json` | 05 | verified |
| Extrapolation figure (H-D4; no `\label`) | `figures/extrapolation.pdf` | `src/figures.py` | `results/artifacts/fits.json` (hd4) | 05 | verified |
| prose macros (`\Ndtau`, `\Nruns`, `\Nhours`, `\Ngpuhours`, ...) | `numbers.tex` | `src/paper.py` | as above, plus `archive/research-log/state.json`, HF `dataset.json`, and `results/artifacts/runs.csv` `train_gpu_hours` for `\Ngpuhours` | | verified |

The paper compiles copies of the five figures, in `figures/` of the paper sources.
`src/paper_sync.py --check-figures` checks that each copy matches its generated file under
`results/`, and `--refresh-figures` updates the copies.

**Supplementary review analyses (not used by the paper).** `src/camera_ready_*.py` and their
outputs in `results/artifacts-camera/` answer reviewer questions from committed artifacts only:
matched-NFE search against refinement (`camera_ready_search.py`, with the selector's cost and
a cross-selector control), T=2 and non-degenerate-item baselines and the F5-TTS calibration
anchor (`camera_ready_scope.py`), and the full compute records (`camera_ready_release.py`,
`camera_ready_integrate.py`). The paper does not `\input` their `numbers_cr_*.tex` macros or
include their figure, so `05_fit_and_paper.sh` and `reproduce_paper_cpu.sh` do not run or
check them; run them by hand (commands in `05_fit_and_paper.sh`, step 5d). Every output goes
to `results/artifacts-camera/`.

## Seeds and determinism

- **Data selection.** RNG 1234 (`data.py SELECT_RNG`). The run uses 200 held-out speakers,
  a speaker-stratified round robin up to 2,000 h, 2,000 validation clips spread across speakers,
  and 400 eval items over 174 speakers, re-curated so that ground-truth Whisper WER is at most
  0.25 (`LOG.md` D-005). The same shard set gives the same subset. **The shard set itself is not
  recorded in the release.**
- **Training.** `BatchPlan(seed)` is a seed-keyed permutation per epoch, sorted by length within
  superbatches of 8x256. The batch order is identical across configs.
  - Initialisation: `torch.manual_seed(1000 + seed)`.
  - Masking: `Generator(seed*1000003 + step)`, identical across configs, and the gradient does not
    depend on how the batch is split into micro-batches (loss normalised over the 256-sequence
    effective batch, `LOG.md` P0-6).
  - Validation: `VAL_RNG = 999` with mask ratios {0.25, 0.5, 0.75}, so validation loss is comparable across runs.
  - Under DDP each rank takes a disjoint stride of the same batch (commit df5ee36).
  - bf16 with TF32 matmuls enabled, so results are **not bitwise reproducible** across hardware or kernels.
- **Sampler (frozen).** Fixed by `configs/grid.json -> backbone.sampler_frozen`, never tuned per shape or T:
  - MaskGIT confidence decoding, level by level over levels 1 to 8, with T steps per level
    (NFE = 8T), a cosine unmasking schedule, and Gumbel noise annealed from 1 to 0, temperature 1, no CFG.
  - Generation RNG `GEN_SEED_BASE = 20260805`, keyed by (batch, level, step) over fixed batches of 50.
    Every item therefore sees the same noise at the first step of each level for every T, config and seed.
  - The protocol §6.3 integrity check (`sample.py integrity`) passed for all 45 runs (`results/artifacts/integrity.json`).
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
| H-D1 to H-D4 | `docs/preregistration/PREREGISTRATION.md` | `archive/research-log/DECISION.md`, `results/artifacts/fits.json -> decision` (outcome class S1) |
| H-E1 to H-E5 | `docs/preregistration/PREREGISTRATION-v1.1.md` | `results/artifacts-v1.1/e1_extended.json` (H_E1), `iso_latency.json` (H_E2), `e3_nfe.json` (H_E3), `e4_scale.json` (H_E4), `e6_undertraining.json` (H_E5); `archive/research-log/LOG-v1.1.md` |
| H-S0 to H-S4 | `docs/preregistration/PREREGISTRATION-v1.2.md` | `archive/research-log/DECISION-v1.2.md`, `results/artifacts-v1.2/{identity_ledger,s1_context,s2_search,s3_rate,s4_guidance}.json` (H_S* fields); summary in `results/artifacts-v1.2/verdicts.json` |
| H-T1 to H-T3 | `docs/preregistration/PREREGISTRATION-v1.3.md` | `results/artifacts-v1.3/t1_scope.json` (H_T1), `t23_training.json` (H_T2, H_T3) |
| v1.5 CFG gate, compute trend | `docs/preregistration/PREREGISTRATION-v1.5.md` | `results/artifacts-v1.5/cfg_gate.json`, `archive/research-log/RESULTS-FEED.md` |

Thresholds and gates G0 to G6 are in `docs/protocol.html`. Each gate's measured values are
logged in `archive/research-log/LOG*.md` and `archive/research-log/state.json -> gates`.

## Known deviations and gaps (camera-ready, 2026-10-01)

1. **The 180k-step trend row cannot be reproduced from the release.** `paper_v15.py` reads
   `results/runs-v1.4/*_180k/synth_T{1,16}/{scores,asr2}.json`, but `results/runs-v1.4/` is in `.gitignore`.
   Only `results/runs-v1.3/C3_0_180k` is committed. The eight missing runs are C1_{0,1,2}, C3_{1,2} and
   C5_{0,1,2}. Committing those small JSON files would make Table `tab:trend` fully verifiable.
   The 30k and 90k rows reproduce exactly, CIs included (`recipes/lib/v15_partial.py`).
2. **The data needed for tiers B and C is not fully released.**
   - Missing: the list of the 71 Emilia/EN shards scanned in v1.0, `eval_zs.json` (400 item ids
     and texts), `heldout_speakers.json`, `index.parquet` and the token shards. Only
     `dataset.json` and `phone_vocab.json` are on the HF repo.
   - `01_data.sh` refuses to guess the shard list. Rebuilding the subset from different shards
     gives a different subset.
3. **The wav2vec2 floor on real speech has no producer.** `\NasrTwoFloor` = 0.1501 comes from
   `results/artifacts-v1.5/g0c_asr2.json`, and no committed script writes that file. `asr2.py` scores only
   synthesis directories.
4. **Some generator constants are typed in, not read from artifacts.** Each was checked against
   its artifact:
   - `paper_v15.py` hard-codes `WER_FLOOR = 0.0344830` and `SIM_CEIL = 0.5553695`. They match
     `g0c_groundtruth.json` and `identity_ledger.json` to 7 digits.
   - `v14_analysis.data_description()` hard-codes `train_hours 2000` and `train_speakers_min 4000`.
     So `\NtrainSpk` is the pre-registered minimum (4,000), not the measured 64,680 (`\Nspeakers`).
   - `paper.py` hard-codes `\NselCorrRange` (0.71--0.73), `\NsearchMaxN` (276) and `\Nbudgetc`.
   - `paper_v14.py` hard-codes `\NmencItems` (200) and `\NmencMinAuc` (0.98).
5. **`evaluate.py collect` scans `<repo>/results/runs/*` and now also returns T in {24,32,64}.** Keeping
   T ≤ 16 gives back the committed 225-row `results/artifacts/runs.csv` exactly.
6. **Python 3.12 or newer is required** (`paper.py` uses PEP 701 f-strings).
   `orchestrate.py`'s docstring lists `phase2`/`phase3` subcommands that do not exist. The real
   subcommands are `phase1`, `train --phase N`, `phase4`, `score`, `status` and `g2`/`g3`/`g5`/`g6`.

All GPU steps (tiers B and C) are documented from the code and the run records. None of them
were executed for this release. Only tier A was run end to end; see `recipes/` and the notes.
