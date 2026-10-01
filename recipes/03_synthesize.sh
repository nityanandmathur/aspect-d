#!/usr/bin/env bash
# 03 -- synthesis: the T sweep of the FROZEN sampler over the 400 eval_zs items.
#
#   RUNS_ROOT=runs-retrain bash recipes/03_synthesize.sh
#   RUNS_ROOT=runs-retrain T_GRID="32 64" bash recipes/03_synthesize.sh    # v1.4 extension
#
# Feeds: the 75-point surface (15 configs x T in {1,2,4,8,16}, 3 seeds) behind every
# headline table/figure; T in {32,64} extends it for Table "gap closed" (runs_extended.csv).
#
# Frozen sampler (configs/grid.json -> backbone.sampler_frozen, src/sample.py): MaskGIT
# confidence decoding level by level 1..8, exactly T steps per level (NFE = 8T), cosine
# unmasking, Gumbel noise on log-probs annealed 1.0 -> 0.0 within a level, categorical
# sampling at temperature 1.0, no CFG. Never tuned per shape or per T.
# Generation RNG: generators keyed by (batch index, level, step) over a FIXED item order
# and FIXED batches of 50 (GEN_SEED_BASE = 20260805), so every item sees the same noise at
# the first step of every level for every T, config and seed (LOG.md P0-3).
#
# Compute: synth_gpu_hours in runs.csv sums to 2.42 GPU-h for all 225 (run, T) cells
# (~30-130 s per cell on a B200; the cost is mostly model load).
#
# Outputs: $RUNS_ROOT/<run>/synth_T<T>/{it0000..it0399.flac, tokens.npz, synth.json}
# (synth.json: T, nfe, recipe, items, ckpt_step, wall_seconds, gpu_hours).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need_gpu
cd "$REPO/src"

RUNS_ROOT="${RUNS_ROOT:-runs-retrain}"
case "$RUNS_ROOT" in /*) ;; *) RUNS_ROOT="$REPO/$RUNS_ROOT" ;; esac
ITEMS="${ITEMS:-}"            # empty = all 400; the v1.1 E1 extension used --items 200
IFS=, read -r -a GPU_ARR <<< "$GPUS"
LOGDIR="$RUNS_ROOT/_logs"; mkdir -p "$LOGDIR"

# One command per (run, T), e.g.
#   python src/sample.py synth --run runs-retrain/C3_0 --T 16 --device cuda:0
J="$LOGDIR/synth_jobs.txt"; : > "$J"
for cfg in $CONFIGS; do
  for s in $SEEDS; do
    rd="$RUNS_ROOT/${cfg}_${s}"
    [ -f "$rd/ckpt.pt" ] || { echo "[skip] $rd has no ckpt.pt"; continue; }
    for T in $T_GRID; do
      [ -f "$rd/synth_T$T/synth.json" ] && continue
      echo "synth_${cfg}_${s}_T$T|$PY sample.py synth --run $rd --T $T --device cuda:0 ${ITEMS:+--items $ITEMS}" >> "$J"
    done
  done
done
say "$(wc -l < "$J" | tr -d ' ') synthesis jobs"
n=${#GPU_ARR[@]}
for ((g = 0; g < n; g++)); do
  ( j=0; while IFS='|' read -r label cmd; do
      if (( j % n == g )); then
        CUDA_VISIBLE_DEVICES="${GPU_ARR[$g]}" bash -c "$cmd" >> "$LOGDIR/$label.log" 2>&1 \
          || echo "[warn] $label failed"
      fi; j=$((j + 1)); done < "$J" ) &
done
wait

# Protocol §6.3 sampler-integrity check, BEFORE any metric: T must reach the sampler, i.e.
# T=1 and T=16 outputs must differ on >= 20 % of generated cells, for every run.
# Paper run: PASS for all 45 runs (artifacts/integrity.json).
"$PY" sample.py integrity --runs-glob "$RUNS_ROOT/*" --t-lo 1 --t-hi 16 \
  --out "$RUNS_ROOT/_integrity.json"

# Measured per-layer latency c_layer(w) used by runs.csv latency_ms and the iso-latency
# analysis (artifacts/c_layer.json in the paper; GPU-model-specific):
#   python sample.py clayer --out "$RUNS_ROOT/_c_layer.json" --device cuda:0
say "done; next: recipes/04_score.sh"
