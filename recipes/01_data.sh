#!/usr/bin/env bash
# 01 -- data: Emilia-EN -> 2,000 h speaker-stratified subset -> phonemes -> Mimi tokens,
# plus the 400-item zero-shot eval set (eval_zs). Exactly the stages of src/data.py.
#
#   SHARD_LIST=shards.txt bash recipes/01_data.sh
#
# Needs: HF_TOKEN with access to the gated dataset amphion/Emilia-Dataset; espeak-ng;
# GPUs for `encode` (Mimi, one process per GPU) and `recurate_eval` (Whisper-large-v3 on GPU 0).
# Disk: ~1 GB per raw shard; the v1.0 subset was drawn from 71 shards.
#
# Produces ($ASPECTD_DATA/proc):
#   meta.parquet          scan of every complete shard (id, speaker, duration, text, ...)
#   index.parquet         train / val / eval_prompt / eval_target rows with token + phoneme offsets
#   eval_zs.json          400 cross-sentence zero-shot items (prompt clip 2.5-3.5 s, target 4-15 s)
#   heldout_speakers.json 200 held-out speakers (selection RNG 1234)
#   phone_vocab.json      espeak-ng en-us phoneme vocabulary, built from TRAIN rows only
#   phones.npy, tokens_<shard>.npy (int16 [frames, 8], Mimi 12.5 Hz, 8 of 32 codebooks)
#   eval_audio/*.flac     ORIGINAL (non-codec) 24 kHz waveforms of eval prompts/targets (SIM-o reference)
#   eval_gtwer.json       ground-truth Whisper WER of every eval candidate (gate G0(c) curation)
#   dataset.json          stats; the released copy is on the HF repo (dataset.json)
#
# Expected dataset.json (HF release): train_hours 2000.0, train_clips 792,064,
# train_speakers 64,680, val_clips 2,000, heldout_speakers 200, eval_items 400 over 174
# speakers, sec_per_char 0.0602, shards_used 71, eval_curation.gt_wer_max 0.25,
# candidates 6,810, transcribable 6,520, gt_wer_mean_selected 0.0345.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO/src"

RAW="$ASPECTD_DATA/emilia_raw"
WORKERS="${WORKERS:-64}"
mkdir -p "$RAW" "$ASPECTD_DATA/proc"

# ---------------------------------------------------------------- 1a. raw shards
# stage_scan reads every $RAW/<shard>.tar that has a sibling <shard>.tar.done marker, so the
# set of shards on disk DEFINES the selection. The v1.0 run used 71 Emilia/EN shards, but
# WHICH 71 is not recorded in any committed artifact (dataset.json only gives the count).
# Exact reproduction of the subset needs that list: on the original machine it is
#   python -c "import pandas as p; print('\n'.join(sorted(p.read_parquet('$ASPECTD_DATA/proc/meta.parquet').shard.unique())))"
# Put one hub path per line (e.g. Emilia/EN/<shard>.tar) in $SHARD_LIST.
if [ -z "${SHARD_LIST:-}" ]; then
  die "set SHARD_LIST to a file of Emilia hub paths (see comment above); the 71-shard list of the paper run is not in the release"
fi
need_file "$SHARD_LIST"
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN unset (amphion/Emilia-Dataset is gated)"
"$PY" - "$SHARD_LIST" "$RAW" <<'EOF'
import os, shutil, sys
from huggingface_hub import hf_hub_download
lst, raw = sys.argv[1], sys.argv[2]
for f in [l.strip() for l in open(lst) if l.strip()]:
    dst = os.path.join(raw, os.path.basename(f))
    if os.path.exists(dst + ".done"):
        continue
    p = hf_hub_download("amphion/Emilia-Dataset", f, repo_type="dataset",
                        token=os.environ["HF_TOKEN"], local_dir=os.path.join(raw, ".hf"))
    shutil.move(p, dst)
    open(dst + ".done", "w").close()          # the marker stage_scan requires
    print("[fetch]", f, flush=True)
EOF

# ---------------------------------------------------------------- 1b. data.py stages
# Order is the order of `python data.py all`. Every stage is idempotent / resumable.
"$PY" data.py scan      --workers "$WORKERS"   # -> meta.parquet          (CPU)
"$PY" data.py select                           # -> index/eval_zs/dataset (CPU, RNG 1234)
"$PY" data.py phonemize --workers "$WORKERS"   # -> phones.npy, vocab     (CPU, espeak-ng)
"$PY" data.py eval_audio --workers "$WORKERS"  # -> eval_audio/*.flac     (CPU)
need_gpu
"$PY" data.py encode    --gpus "$NGPUS"        # -> tokens_<shard>.npy    (GPU, kyutai/mimi)
# Gate G0(c) repair (LOG.md D-005): keep only items whose ground-truth prompt AND target are
# transcribable (Whisper GT-WER <= 0.25), re-select 400 items by the same round-robin, and
# phonemise/encode the late additions into a pseudo-shard "evalextra". Uses GPU 0.
"$PY" data.py recurate_eval --workers "$WORKERS"

# ---------------------------------------------------------------- 1c. check
# The released dataset.json is the reference (HF repo root). Fetch it with
#   huggingface-cli/hf download nityanandmathur/aspect-d-masked-diffusion-tts dataset.json
if [ -n "${RELEASED_DATASET_JSON:-}" ]; then
  "$PY" - "$ASPECTD_DATA/proc/dataset.json" "$RELEASED_DATASET_JSON" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
bad = {k: (a.get(k), v) for k, v in b.items() if a.get(k) != v}
print("dataset.json matches the release" if not bad else f"MISMATCH vs release: {bad}")
sys.exit(1 if bad else 0)
EOF
fi
say "data ready under $ASPECTD_DATA/proc"
