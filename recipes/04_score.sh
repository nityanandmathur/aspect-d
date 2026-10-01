#!/usr/bin/env bash
# 04 -- scoring: WER (Whisper-large-v3), SIM-o (WavLM-large SV), UTMOS22-strong, degenerate
# rate; the two measured floors; and runs.csv.
#
#   RUNS_ROOT=runs bash recipes/04_score.sh      # TIER B: re-score the released audio/ckpts
#
# !! Run tiers B/C in a SCRATCH COPY of the repo, never in the release checkout: several
# !! src/ scripts write to fixed artifacts*/ paths (s0_ledger.py -> artifacts-v1.2/,
# !! evaluate.py collect reads <repo>/runs/* only). Committed artifacts must not be overwritten.
#
# Feeds: runs.csv (every surface number), Table "gap closed" floors (\NwerFloor from
# g0c_groundtruth.json, \NsimCeiling from identity_ledger.json), the identity ledger.
#
# Metric stack (configs/grid.json -> eval_models, protocol.html §6):
#   WER   openai/whisper-large-v3, greedy, Whisper EnglishTextNormalizer, per-item jiwer WER,
#         mean over ALL 400 items (degenerates included)
#   SIM-o WavLM-large SV (microsoft/UniSpeech, seed-tts-eval) cosine of generated audio vs the
#         ORIGINAL prompt waveform ($ASPECTD_DATA/proc/eval_audio). Loaded from
#         $ASPECTD_MODELS/wavlm_sv; the scorer refuses to fall back silently.
#   UTMOS tarepan/SpeechMOS:v1.2.0 utmos22_strong (torch.hub); tertiary
#   degenerate (§6.4): empty hypothesis, < 30 % of the reference length, or a 4-gram
#         repeated >= 6 times consecutively
# Compute: scoring is dominated by model load (~500 s cold); 8 sharded scorers ~1 h for the
# 225 cells on 8x B200 (not separately recorded in runs.csv).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
need_gpu
cd "$REPO/src"

RUNS_ROOT="${RUNS_ROOT:-runs}"
case "$RUNS_ROOT" in /*) ;; *) RUNS_ROOT="$REPO/$RUNS_ROOT" ;; esac
OUT="${OUT:-$REPO/artifacts-camera/rescore}"; mkdir -p "$OUT"
IFS=, read -r -a GPU_ARR <<< "$GPUS"
n=${#GPU_ARR[@]}

# ---------------------------------------------------------------- 4a. score every (run, T)
# Equivalent single command: python src/evaluate.py score --run runs/C3_0 --T 16 --device cuda:0
# Here: one job file per GPU so each scorer loads Whisper/WavLM/UTMOS once (what
# `orchestrate.py score` and run_v14.py's worker_score do). Existing scores.json are skipped
# unless FORCE=1. Writes $RUNS_ROOT/<run>/synth_T<T>/scores.json (summary + per-item rows).
"$PY" - "$RUNS_ROOT" "$OUT" "$n" "${FORCE:-}" <<'EOF'
import glob, json, os, sys
root, out, n, force = sys.argv[1], sys.argv[2], int(sys.argv[3]), bool(sys.argv[4])
jobs = []
for sd in sorted(glob.glob(os.path.join(root, "*", "synth_T*"))):
    if os.path.basename(os.path.dirname(sd)).startswith("sweep_"):
        continue
    if not os.path.exists(os.path.join(sd, "synth.json")):
        continue
    if os.path.exists(os.path.join(sd, "scores.json")) and not force:
        continue
    jobs.append({"run": os.path.dirname(sd), "T": int(os.path.basename(sd)[7:])})
for g in range(n):
    json.dump(jobs[g::n], open(os.path.join(out, f"score_jobs_{g}.json"), "w"))
print(f"[score] {len(jobs)} (run, T) cells over {n} GPUs")
EOF
for ((g = 0; g < n; g++)); do
  CUDA_VISIBLE_DEVICES="${GPU_ARR[$g]}" "$PY" evaluate.py score --jobs "$OUT/score_jobs_$g.json" \
    --device cuda:0 ${FORCE:+--force} > "$OUT/score_$g.log" 2>&1 &
done
wait

# ---------------------------------------------------------------- 4b. the two floors
# (i) ASR floor = Whisper WER on the REAL recordings of the 400 targets, plus the SIM-o
# instrument gate G0(c) (same-speaker median >= 0.50, cross-speaker median <= 0.25).
# Paper: wer_mean_item 0.0345 (\NwerFloor 0.0345), sim_same_median 0.7005, cross 0.0338.
CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" "$PY" evaluate.py gt --device cuda:0 --items 400 \
  --out "$OUT/g0c_groundtruth.json"

# (ii) codec ceiling = SIM-o of a Mimi encode->decode round trip of each real target, scored
# against the original prompt (\NsimCeiling 0.5554), plus the identity ledger.
# s0_ledger.py WRITES artifacts-v1.2/identity_ledger.json and s0_per_item_sim.npy in place
# (no --out flag) -- this is why tier B must run in a scratch copy.
if [ -n "${ALLOW_ARTIFACT_WRITE:-}" ]; then
  CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" "$PY" s0_ledger.py --device cuda:0 --items 400
else
  say "skipping s0_ledger.py (writes artifacts-v1.2/ in place); set ALLOW_ARTIFACT_WRITE=1 in a scratch copy"
fi

# ---------------------------------------------------------------- 4c. runs.csv
# collect reads <repo>/runs/*/{run.json, synth_T*/scores.json, synth.json} -- the root is
# fixed. It now also picks up the extended T=24/32/64 cells; artifacts/runs.csv is the
# T in {1,2,4,8,16} subset (225 rows). Verified on CPU: that subset of a fresh collect is
# identical to the committed artifacts/runs.csv in all 34 columns (NOTES/recipes.md).
"$PY" evaluate.py collect --out "$OUT/runs_all_T.csv"
"$PY" - "$OUT/runs_all_T.csv" "$OUT/runs.csv" <<'EOF'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1])
df = df[df["T"].isin([1, 2, 4, 8, 16]) & ~df.config.str.startswith("sweep")]
df.to_csv(sys.argv[2], index=False)
print(f"[collect] {len(df)} rows (expect 225 = 15 configs x 3 seeds x 5 T) -> {sys.argv[2]}")
EOF
say "done; compare $OUT/runs.csv with artifacts/runs.csv, then recipes/05_fit_and_paper.sh RUNS_CSV=$OUT/runs.csv"
