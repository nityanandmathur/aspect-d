#!/usr/bin/env bash
# 06 -- the v1.1-v1.5 extensions, each mapped to the paper table/figure it feeds.
#
#   STAGE=e3 bash recipes/06_extensions.sh        # one stage (see the list below)
#   STAGE=list bash recipes/06_extensions.sh      # print the stages
#
# GPU stages need a scratch copy of the repo: the analysis scripts write their committed
# artifact paths in place (results/artifacts-v1.1/ .. results/artifacts-v1.5/). Every analysis step marked
# (CPU) re-runs from the committed per-item records with no GPU; reproduce_paper_cpu.sh
# checks the paper-facing outputs of all of them.
#
#  stage   what                                           paper                      GPU-h (records)
#  e1      extended T in {24,32,64} on 21 runs, 200 items Table tab:prereg H-E1      in synth below
#  e2      iso-latency Pareto (CPU only)                  Table tab:prereg H-E2      0
#  e3      NFE allocation across levels at NFE 32         Table tab:alloc            ~0.1
#  e4      budget D (276 M, 5 configs x 2 seeds)          Tables tab:robust, search  56.5 train + 3.0 LR sweep
#  e5      metric-stack robustness (ASR / SV / log-amp)   Table tab:robust           scoring only
#  e6      3x training compute, C{1,3,5} seed 0 @ 90k     Table tab:robust           29.8 train
#  s0      identity ledger + Mimi round-trip ceiling      Table tab:ledger, Fig fig:ledger, \NsimCeiling
#  s1      prompt context 1.5/3/6/9 s                     Table tab:context
#  s2      best-of-K search vs refinement at matched NFE  Tables tab:search, tab:ledger
#  s3      rate-matched length                            Table tab:negative
#  s4      speaker-contrastive guidance                   Tables tab:negative, tab:ledger
#  t       v1.3: 90k x seeds 1,2; C3_0 180k; varprompt;   Tables tab:ledger (training rows), tab:search
#          search scope on budget D and 90k               footnote                     80.0 train
#  menc    multi-encoder agreement (5 SV encoders)        Table tab:menc
#  v14     X1 all 45 runs to T{32,64}; X2 90k sweep; X7   Tables tab:gapclosed, tab:scope
#  v15     180k x 9 (3x/6x compute trend), second ASR,    Table tab:trend
#          CFG gate, F5-TTS anchor
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO/src"
STAGE="${STAGE:-list}"
C_RUNS="$(for i in 1 2 3 4 5; do for s in 0 1 2; do printf 'C%s_%s ' $i $s; done; done)"
LOGS="$REPO/results/artifacts-camera/ext_logs"   # created on first use (score_tags)

synth() {   # synth <run dir> <sample.py synth flags...>   (one GPU; set CUDA_VISIBLE_DEVICES)
  local rd="$1"; shift
  "$PY" sample.py synth --run "$rd" --device cuda:0 "$@"
}
score_tags() {   # score_tags <T> <tag> <run dir>...  -> scores.json beside each synth_<tag>
  local T="$1" tag="$2"; shift 2
  mkdir -p "$LOGS"
  "$PY" - "$LOGS/score_${tag}.json" "$T" "$tag" "$@" <<'EOF'
import json, sys
out, T, tag, runs = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
json.dump([{"run": r, "T": T, "tag": tag} for r in runs], open(out, "w"))
EOF
  "$PY" evaluate.py score --jobs "$LOGS/score_${tag}.json" --device cuda:0
}

case "$STAGE" in
list) sed -n '2,32p' "$RECIPES_DIR/06_extensions.sh" ;;

e1) # H-E1: A3, B3, C1-C5 x seeds {0,1,2} (21 runs) at T in {24,32,64}, first 200 items;
    # scored into scores_ext200.json; T<=16 re-aggregated from v1.0 scores restricted to 200.
    need_gpu
    for r in A3 B3 C1 C2 C3 C4 C5; do for s in 0 1 2; do for T in 24 32 64; do
      synth "$REPO/results/runs/${r}_$s" --T $T --items 200
      "$PY" evaluate.py score --run "$REPO/results/runs/${r}_$s" --T $T --items 200 --suffix _ext200 --device cuda:0
    done; done; done
    "$PY" e1_extended.py                       # (CPU) -> results/artifacts-v1.1/e1_extended.json, runs_ext.csv
    ;;

e2) "$PY" iso_latency.py                       # (CPU) -> results/artifacts-v1.1/iso_latency*.json/csv, figures
    "$PY" iso_latency.py --fits-from-e1 --out-tag _ext
    ;;

e3) # H-E3: three per-level schedules at total NFE 32 on the 15 C-budget runs, 400 items.
    need_gpu
    for r in $C_RUNS; do
      synth "$REPO/results/runs/$r" --T 4 --schedule 4,4,4,4,4,4,4,4  --tag nfe32uniform
      synth "$REPO/results/runs/$r" --T 4 --schedule 25,1,1,1,1,1,1,1 --tag nfe32coarse
      synth "$REPO/results/runs/$r" --T 4 --schedule 1,1,1,1,1,1,1,25 --tag nfe32fine
    done
    for t in nfe32uniform nfe32coarse nfe32fine; do
      score_tags 4 $t $(for r in $C_RUNS; do printf '%s ' "$REPO/results/runs/$r"; done)
    done
    "$PY" e3_nfe.py                            # (CPU) -> results/artifacts-v1.1/e3_nfe.json (\Nalloc*, \Nnfe*)
    ;;

e4) # H-E4: G1-D LR sweep at D3 (adopt its argmin), then D1-D5 x seeds {0,1}, 30k steps,
    # into results/runs-v1.1/. Then T grid {1..16} x 400 items + {24,32,64} x 200 items.
    need_gpu
    mkdir -p "$REPO/results/logs-v1.1"                # build_eval_jobs.py / run_v11.py write here (gitignored)
    "$PY" launch_e4.py --gpus "$GPUS" --seeds 0,1 --steps 30000
    "$PY" build_eval_jobs.py --which e4       # -> results/logs-v1.1/e4_synth_jobs.json, e4_eval_score_jobs.json
    "$PY" run_v11.py --jobs "$REPO/results/logs-v1.1/e4_synth_jobs.json" --name e4synth --gpus "$GPUS"
    "$PY" run_v11.py --jobs "$REPO/results/logs-v1.1/e4_eval_score_jobs.json" --name e4score --gpus "$GPUS"
    "$PY" e4_analysis.py                       # (CPU) -> results/artifacts-v1.1/e4_scale.json (\Ndtaufour, \NmaxNfour)
    ;;

e5) # metric-stack panel: re-score every v1.0 synthesis with a second ASR and a second SV.
    need_gpu
    for r in $CONFIGS; do for s in $SEEDS; do for T in $T_GRID; do
      "$PY" evaluate.py score --run "$REPO/results/runs/${r}_$s" --T $T --asr openai/whisper-medium.en --suffix _asrmed --device cuda:0
      "$PY" evaluate.py score --run "$REPO/results/runs/${r}_$s" --T $T --sv-fallback --suffix _svbase --device cuda:0
    done; done; done
    "$PY" e5_robust.py                         # (CPU) -> results/artifacts-v1.1/e5_robustness.json (\Ndtauasr, \NdtauAsrCI)
    "$PY" logamp.py                            # (CPU) -> results/artifacts-v1.1/logamp_refit.json (\Ndtaulogamp)
    ;;

e6) # H-E5: C1, C3, C5 seed 0 retrained to 90k steps into results/runs-v1.1/, T grid x 400 items.
    need_gpu
    for c in C1 C3 C5; do
      "$PY" train.py --config $c --seed 0 --lr "$BASE_LR" --steps 90000 --out "$REPO/results/runs-v1.1/${c}_0_90k" --device cuda:0
    done
    mkdir -p "$REPO/results/logs-v1.1"
    "$PY" build_eval_jobs.py --which e6
    "$PY" run_v11.py --jobs "$REPO/results/logs-v1.1/e6_synth_jobs.json" --name e6synth --gpus "$GPUS"
    "$PY" run_v11.py --jobs "$REPO/results/logs-v1.1/e6_eval_score_jobs.json" --name e6score --gpus "$GPUS"
    "$PY" e6_analysis.py                       # (CPU) -> results/artifacts-v1.1/e6_undertraining.json (\Ndtaunineok)
    ;;

s0) need_gpu
    "$PY" s0_ledger.py --device cuda:0 --items 400   # -> results/artifacts-v1.2/identity_ledger.json (\NsimCeiling 0.5554)
    "$PY" s0_figure.py                               # (CPU) -> results/artifacts-v1.2/figures/identity_ledger.pdf
    ;;

s1) # H-S1: prompt context. s1_prepare.py builds the 1.5/3/6/9 s prompts (Mimi encode);
    # each arm is scored against the SAME fixed 3 s reference waveform.
    need_gpu
    "$PY" s1_prepare.py --device cuda:0 --items 400  # -> results/artifacts-v1.2/s1_prep.json
    for r in $C_RUNS; do for arm in 1.5 3.0 6.0 9.0; do
      synth "$REPO/results/runs/$r" --T 16 --s1-arm $arm --tag "s1ctx${arm/./p}"
    done; done
    for arm in 1p5 3p0 6p0 9p0; do score_tags 16 "s1ctx$arm" $(for r in $C_RUNS; do printf '%s ' "$REPO/results/runs/$r"; done); done
    "$PY" s1_analysis.py                       # (CPU) -> results/artifacts-v1.2/s1_context.json (\Nctx*)
    ;;

s2) # H-S2: best-of-K at T=8 (K up to 8, candidates bok0..bok7, 200 items) vs refinement at
    # T=16/32/64; WavLM-SV SELECTS, ECAPA (gated, ecapa_gate.py) SCORES.
    need_gpu
    "$PY" ecapa_gate.py --device cuda:0 --items 400  # -> results/artifacts-v1.2/ecapa_gate.json
    for r in $C_RUNS; do for k in 0 1 2 3 4 5 6 7; do
      synth "$REPO/results/runs/$r" --T 8 --items 200 --cand $k --tag bok$k
    done; done
    "$PY" s2_search.py --device cuda:0         # per-run parts -> results/artifacts-v1.2/s2_parts/
    "$PY" s2_search.py --aggregate             # (CPU) -> results/artifacts-v1.2/s2_search.json (\Nsrch*, \Nsearch*)
    "$PY" s2_confound.py --device cuda:0
    "$PY" s2_confound.py --aggregate           # (CPU) -> results/artifacts-v1.2/s2_confound.json (\NsearchVsDefault, \NselCorr)
    ;;

s3) need_gpu
    for r in $C_RUNS; do synth "$REPO/results/runs/$r" --T 16 --rate-matched --tag rate; done
    score_tags 16 rate $(for r in $C_RUNS; do printf '%s ' "$REPO/results/runs/$r"; done)
    "$PY" s3_analysis.py                       # (CPU) -> results/artifacts-v1.2/s3_rate.json (\Nrate*)
    ;;

s4) # H-S4: gamma in {0.5,1,2} at T=16, baseline = unguided T=32 (iso-NFE). The committed
    # s4_guidance.json is the v1.5 STAGE-1 REPAIR (attended-PAD bug fixed, iso-NFE baseline;
    # `python run_v15.py --stage 1` re-synthesises the 45 arms and refits).
    need_gpu
    for r in $C_RUNS; do for g in 0.5 1.0 2.0; do
      synth "$REPO/results/runs/$r" --T 16 --gamma $g --tag "gam${g/./p}"
    done; done
    for g in 0p5 1p0 2p0; do score_tags 16 "gam$g" $(for r in $C_RUNS; do printf '%s ' "$REPO/results/runs/$r"; done); done
    "$PY" s4_analysis.py --device cuda:0       # -> results/artifacts-v1.2/s4_guidance.json (\Ngam*, \Nguide*)
    ;;

t)  # v1.3 (PREREGISTRATION-v1.3): H-T2 training axis, H-T3 variable prompt, H-T1 search scope.
    need_gpu
    for c in C1 C3 C5; do for s in 1 2; do
      "$PY" train.py --config $c --seed $s --lr "$BASE_LR" --steps 90000 --out "$REPO/results/runs-v1.3/${c}_${s}_90k" --device cuda:0
    done; done
    "$PY" train.py --config C3 --seed 0 --lr "$BASE_LR" --steps 180000 --out "$REPO/results/runs-v1.3/C3_0_180k" --device cuda:0
    "$PY" train.py --config C3 --seed 0 --lr "$BASE_LR" --variable-prompt --out "$REPO/results/runs-v1.3/C3_0_varprompt" --device cuda:0
    # each new run: T grid synthesis + scoring as in 03/04. H-T1 (search scope) repeats S2 on
    # the budget-D and 90k runs (bok0..7 at T=8, 200 items, synthesised as in stage s2):
    "$PY" s2_search.py --root runs-v1.1 --runs D1_0,D1_1,D2_0,D2_1,D3_0,D3_1,D4_0,D4_1,D5_0,D5_1,C1_0_90k,C3_0_90k,C5_0_90k --device cuda:0
    "$PY" s2_confound.py --root runs-v1.1 --runs D1_0,D1_1,D2_0,D2_1,D3_0,D3_1,D4_0,D4_1,D5_0,D5_1,C1_0_90k,C3_0_90k,C5_0_90k --device cuda:0
    "$PY" t1_analysis.py                       # (CPU) -> results/artifacts-v1.3/t1_scope.json (\Nscope*, \NsearchD)
    "$PY" t23_analysis.py                      # (CPU) -> results/artifacts-v1.3/t23_training.json (\NtrainGainSim, \NtrainDouble)
    ;;

menc) need_gpu
    "$PY" ge2e_gate.py --device cuda:0 --encoder ge2e    # -> results/artifacts-v1.3/ge2e_gate.json
    "$PY" ge2e_gate.py --device cuda:0 --encoder xvect   # -> results/artifacts-v1.3/xvect_gate.json
    "$PY" multi_encoder.py --device cuda:0               # per-run parts -> results/artifacts-v1.3/multienc_parts/
    "$PY" multi_encoder.py --aggregate                   # (CPU) -> results/artifacts-v1.3/multi_encoder.json/.csv (tab_menc.tex)
    ;;

v14) # X1: all 45 v1.0 runs to T in {32,64} at 400 items; X2: 90k x 9 runs full sweep; X7: T=128.
    need_gpu
    "$PY" run_v14.py --plan
    "$PY" run_v14.py --launch                  # 8 workers, synth then score
    "$PY" v14_analysis.py                      # (CPU) -> results/artifacts-v1.4/analysis.json (tab_gapclosed/scope)
    "$PY" x2_analysis.py                       # (CPU) -> results/artifacts-v1.4/x2_x7.json
    ;;

v15) # 3x / 6x training-compute trend: C1/C3/C5 x seeds {0,1,2} to 180k steps into results/runs-v1.4/
    # (C3_0_180k already exists under results/runs-v1.3/). ~13-19 GPU-h per run on a B200 (C3_0_180k:
    # 18.98 GPU-h). results/runs-v1.4/ is .gitignored: the 180k records behind Table tab:trend are
    # NOT in the release (see REPRODUCE.md "Known deviations").
    need_gpu
    "$PY" run_v15_train.py --plan
    "$PY" run_v15_train.py --launch
    "$PY" scheduler.py --run                   # work-stealing: 180k sweeps, CFG arms, ...
    "$PY" run_v15.py --plan                    # stage list; `--stage N` / `--all`
    # second ASR (wav2vec2-large-960h-lv60-self, CTC) beside scores.json for the 54 trend dirs
    "$PY" asr2.py --dirs $(for c in C1 C3 C5; do for s in 0 1 2; do for T in 1 16; do
        printf 'results/runs/%s_%s/synth_T%s ' $c $s $T; done; done; done) --device cuda:0
    # NOTE: results/artifacts-v1.5/g0c_asr2.json (wav2vec2 WER on the REAL recordings, \NasrTwoFloor
    # 0.1501) has no committed producer: asr2.py scores synth dirs of <item>.flac files.
    "$PY" v15_gate.py                          # (CPU) CFG gate verdict -> results/artifacts-v1.5/cfg_gate.json
    "$PY" anchor_f5.py --smoke 5               # F5-TTS v1 Base on our 400 items, isolated venv
    "$PY" anchor_f5.py --items 400 --budget-seconds 10800   # -> results/runs-v1.5/f5tts_anchor, results/artifacts-v1.5/anchor.json
    "$PY" paper_v15.py                         # (CPU) -> paper/numbers_v15.tex, tab_trend.tex
    ;;

*) die "unknown STAGE=$STAGE (STAGE=list)" ;;
esac
