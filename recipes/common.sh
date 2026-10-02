# shellcheck shell=bash
# Shared defaults for every recipe. Sourced, never executed.
#
# Every machine-specific path is an environment variable with a repo-relative default, so
# nothing in recipes/ names a particular host. The variable names are the ones src/ reads:
#
#   ASPECTD_DATA    data root; src/data.py reads $ASPECTD_DATA/emilia_raw and writes
#                   $ASPECTD_DATA/proc (index.parquet, eval_zs.json, dataset.json, tokens_*.npy)
#   ASPECTD_MODELS  eval-model root; src/evaluate.py loads the primary SIM-o model from
#                   $ASPECTD_MODELS/wavlm_sv (UniSpeech models/ecapa_tdnn.py + wavlm_large_finetune.pth)
#   ASPECTD_PY      interpreter the job launchers (orchestrate.py, run_v11.py, ...) spawn
#   ASPECTD_VENV    venv that src/anchor_f5.py probes for version isolation
#   ASPECTD_XL      root of the v1.5 scale-up corpus (fetch_xl.py / prep_xl.py / index_xl.py)
#
# NOTE: ASPECTD_MODELS / ASPECTD_PY / ASPECTD_VENV and the repo-relative defaults were added
# to src/ for the camera-ready release (they replace hard-coded /home/ubuntu paths). With an
# older checkout of src/, set them anyway and also check `grep -n /home/ubuntu src/*.py`.

RECIPES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO="${REPO:-$(cd "$RECIPES_DIR/.." && pwd)}"
export ASPECTD_DATA="${ASPECTD_DATA:-$REPO/data}"
export ASPECTD_MODELS="${ASPECTD_MODELS:-$REPO/models}"
export ASPECTD_VENV="${ASPECTD_VENV:-$REPO/.venv}"
export ASPECTD_PY="${ASPECTD_PY:-$ASPECTD_VENV/bin/python}"
export ASPECTD_XL="${ASPECTD_XL:-$REPO/data-xl}"

# GPUs. The paper ran on one node of 8x NVIDIA B200 (183 GB each).
NGPUS="${NGPUS:-8}"
GPUS="${GPUS:-$(seq -s, 0 $((NGPUS - 1)))}"

# The 15 configurations and 3 seeds of the main grid (configs/grid.json,
# archive/research-log/state.json).
CONFIGS="${CONFIGS:-A1 A2 A3 A4 A5 B1 B2 B3 B4 B5 C1 C2 C3 C4 C5}"
SEEDS="${SEEDS:-0 1 2}"
# Base LR chosen by the Phase-1 muP sweep (state.json chosen_lr; \Nbaselr in the paper).
BASE_LR="${BASE_LR:-0.004}"
# The frozen T grid (steps per codebook level, NFE = 8T; grid.json sampler_frozen.T_eval_grid).
T_GRID="${T_GRID:-1 2 4 8 16}"

PY="$ASPECTD_PY"

say() { printf '\n[%s] %s\n' "$(basename "$0")" "$*"; }
die() { printf '\n[%s] ERROR: %s\n' "$(basename "$0")" "$*" >&2; exit 1; }
need_file() { [ -e "$1" ] || die "missing $1${2:+ -- $2}"; }
need_gpu() {
  "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || die "this step needs a CUDA GPU (src/ defaults to --device cuda:0)"
}
