#!/usr/bin/env bash
# 02 -- training: the Phase-1 muP LR sweep, then the 45-run grid (15 configs x 3 seeds).
#
#   RUNS_ROOT=results/runs-retrain bash recipes/02_train_grid.sh          # sweep + grid
#   SKIP_SWEEP=1 RUNS_ROOT=results/runs-retrain bash recipes/02_train_grid.sh
#   ONLY=C3_0 RUNS_ROOT=results/runs-retrain bash recipes/02_train_grid.sh   # one run
#
# Feeds: every paper table/figure (the 75-point surface = 15 configs x 5 T, 3 seeds each).
#
# Retrain into a NEW root. The committed results/runs/ already holds the paper's run records with
# status "completed"; `src/orchestrate.py train` writes to results/runs/<cfg>_<seed> and skips
# completed runs, so in a fresh clone it would do nothing. The loop below calls
# src/train.py directly, which is exactly the command orchestrate.py issues
# (train_job(): train.py --config C --seed S --lr LR --out DIR --device cuda:0).
#
# Recipe (configs/grid.json -> training, frozen): 30,000 steps, effective batch 256
# sequences (micro-batch + grad-accum chosen per shape, loss normalised over the whole
# effective batch so the gradient is split-invariant), AdamW(0.9, 0.95) wd 0.1, clip 1.0,
# 600 warmup, cosine to 10 %, bf16, val every 1,000 steps at mask ratios {0.25,0.5,0.75}
# with VAL_RNG = 999. muP base LR 0.004 at base width 256.
#
# Determinism (src/train.py): data order = BatchPlan(seed) -- a seed-keyed permutation per
# epoch, length-sorted within superbatches of 8x256, identical for every config; init =
# torch.manual_seed(1000 + seed); masking draws = torch.Generator(seed*1000003 + step), so
# the masking stream is identical across configs and invariant to the micro-batch split.
# Bitwise equality with the released checkpoints is NOT expected (bf16, cuDNN/cuBLAS
# kernel choice, TF32 matmuls are enabled).
#
# Compute, measured (HF runs.csv train_gpu_hours, mean over 3 seeds, B200; wall-clock of
# each run, which over-counts exclusive GPU time when runs share a GPU):
#   A1 1.32  A2 1.61  A3 1.77  A4 2.13  A5 2.43         budget A  27.8 GPU-h / 15 runs
#   B1 2.06  B2 4.76  B3 3.18  B4 3.89  B5 5.22         budget B  57.3
#   C1 4.21  C2 5.01  C3 4.10  C4 5.96  C5 6.52         budget C  77.4
#   45-run grid total 162.5 GPU-h; Phase-1 sweep (20 x 3k-step proxies) 6.9 GPU-h.
#
# Outputs per run ($RUNS_ROOT/<cfg>_<seed>/): run.json (config, seed, lr, steps, params,
# micro_batch, accum, device, status, gpu_hours, val_hist, final_val_loss), train_log.jsonl,
# val_log.jsonl, ckpt.pt (resumable; checkpoint every 1,000 steps). The HF repo holds the
# released weights as <cfg>_<seed>/model.safetensors + config.json + run.json.
#
# Exit codes of train.py: 0 completed, 2 NaN, 3 diverged (gate G3). The paper's grid had
# no restarts (state.json restarts = {}); orchestrate.py's G3 rule is ONE restart from
# scratch at 0.5x LR.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need_gpu
cd "$REPO/src"

RUNS_ROOT="${RUNS_ROOT:-results/runs-retrain}"
case "$RUNS_ROOT" in /*) ;; *) RUNS_ROOT="$REPO/$RUNS_ROOT" ;; esac
[ "$RUNS_ROOT" != "$REPO/results/runs" ] || die "refusing to train into the committed results/runs/ (set RUNS_ROOT)"
LOGDIR="$RUNS_ROOT/_logs"; mkdir -p "$LOGDIR"
IFS=, read -r -a GPU_ARR <<< "$GPUS"

# lane runner: job i goes to GPU i % NGPUS; each GPU runs its jobs one after another
run_lanes() {   # args: a file with one "label|cmd..." per line
  local jobs="$1" n=${#GPU_ARR[@]} i=0
  for ((g = 0; g < n; g++)); do
    (
      j=0
      while IFS='|' read -r label cmd; do
        if (( j % n == g )); then
          echo "[$(date -u +%H:%M:%S)] GPU${GPU_ARR[$g]} $label"
          CUDA_VISIBLE_DEVICES="${GPU_ARR[$g]}" bash -c "$cmd" >> "$LOGDIR/$label.log" 2>&1 \
            || echo "[warn] $label exited $? (see $LOGDIR/$label.log)"
        fi
        j=$((j + 1))
      done < "$jobs"
    ) &
  done
  wait
}

# ---------------------------------------------------------------- 2a. Phase-1 muP sweep
# Gate G1 (width transfer: d=12, w in {256, 640}) and G1b (depth transfer: w=384,
# d in {4, 24}); 5 LRs x 4 proxies x 3,000 steps. The argmin of the w=256 sweep is the base
# LR (0.004 in the paper; both gates passed -- state.json gates.G1/G1b). Equivalent to
# `python orchestrate.py phase1 --steps 3000 --gpus $GPUS` but into $RUNS_ROOT.
if [ -z "${SKIP_SWEEP:-}" ] && [ -z "${ONLY:-}" ]; then
  J="$LOGDIR/sweep_jobs.txt"; : > "$J"
  for spec in "g1_w256 256 12 --coord-check 50" "g1_w640 640 12 --coord-check 50" \
              "g1b_d4 384 4" "g1b_d24 384 24"; do
    set -- $spec; name=$1 w=$2 d=$3; shift 3
    for lr in 0.001 0.002 0.004 0.008 0.016; do
      echo "sweep_${name}_lr${lr}|$PY train.py --config A1 --seed 0 --lr $lr --steps 3000 --out $RUNS_ROOT/sweep_${name}_lr${lr} --device cuda:0 --proxy-width $w --proxy-depth $d $*" >> "$J"
    done
  done
  run_lanes "$J"
  # pick: orchestrate.ema (last 3 val points, weights 0.25/0.5/1) per LR, then argmin;
  # expect (state.json gates): g1_w256 0.004, g1_w640 0.002, g1b_d4 0.002, g1b_d24 0.002
  "$PY" - "$RUNS_ROOT" <<'EOF'
import glob, json, os, sys
sys.path.insert(0, ".")
from orchestrate import ema
root = sys.argv[1]
for name in ("g1_w256", "g1_w640", "g1b_d4", "g1b_d24"):
    row = {}
    for rj in sorted(glob.glob(os.path.join(root, f"sweep_{name}_lr*", "run.json"))):
        r = json.load(open(rj))
        v = [h["val_loss"] for h in r.get("val_hist", [])]
        row[r["lr"]] = round(ema(v), 4) if v else None
    ok = {k: v for k, v in row.items() if v is not None}
    print(name, row, "argmin", min(ok, key=ok.get) if ok else None)
EOF
fi

# ---------------------------------------------------------------- 2b. the 45-run grid
# One command per run (largest budget first so the lanes finish together), e.g.
#   python src/train.py --config C5 --seed 0 --lr 0.004 --out results/runs-retrain/C5_0 --device cuda:0
J="$LOGDIR/grid_jobs.txt"; : > "$J"
for cfg in $(echo $CONFIGS | tr ' ' '\n' | sort -r); do
  for s in $SEEDS; do
    [ -z "${ONLY:-}" ] || [ "$ONLY" = "${cfg}_${s}" ] || continue
    if [ -f "$RUNS_ROOT/${cfg}_${s}/run.json" ] && \
       grep -q '"status": "completed"' "$RUNS_ROOT/${cfg}_${s}/run.json"; then continue; fi
    echo "${cfg}_${s}|$PY train.py --config $cfg --seed $s --lr $BASE_LR --out $RUNS_ROOT/${cfg}_${s} --device cuda:0" >> "$J"
  done
done
say "$(wc -l < "$J" | tr -d ' ') grid runs to train on GPUs $GPUS"
run_lanes "$J"

# ---------------------------------------------------------------- optional: 8-GPU DDP
# train.py also runs ONE model across GPUs under torchrun (WORLD_SIZE > 1): each rank takes
# a disjoint stride of the same BatchPlan batch, the loss-cell denominator is all-reduced, so
# effective batch, data order and seed semantics are unchanged (commit df5ee36: step-0 loss
# 7.6343 under 8-way DDP vs 7.6315 single-GPU). It was added for the v1.5 scale-up run and
# was NOT used for the 45-run grid (one run per GPU there). run.json's gpu_hours under DDP is
# rank-0 wall time, not wall x world size.
#   torchrun --standalone --nproc_per_node 8 train.py --config C3 --seed 0 --lr 0.004 --out $RUNS_ROOT/C3_0_ddp

say "done: $RUNS_ROOT/<cfg>_<seed>/run.json; next: recipes/03_synthesize.sh with RUNS_ROOT=$RUNS_ROOT"
