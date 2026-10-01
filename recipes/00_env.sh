#!/usr/bin/env bash
# 00 -- environment.
#
#   TIER=cpu  bash recipes/00_env.sh   # tier A: numpy/pandas/scipy/matplotlib only (any OS, no GPU)
#   TIER=gpu  bash recipes/00_env.sh   # tiers B/C: the full pinned stack of the paper runs
#
# The paper ran on Linux x86_64, Python 3.12, PyTorch 2.11 (CUDA 13 wheels), 8x NVIDIA B200,
# driver 595.71.05, CUDA 13.2 (LOG.md, 2026-08-05 21:39 UTC). requirements.txt carries the
# pins and says which are recorded ([known]) and which are reconstructed.
#
# Python >= 3.12 is required: src/paper.py uses PEP 701 f-strings and does not parse on 3.11.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TIER="${TIER:-gpu}"
UV="${UV:-uv}"
command -v "$UV" >/dev/null || die "uv not found (https://docs.astral.sh/uv/); or set UV=/path/to/uv"

say "creating venv at $ASPECTD_VENV (Python 3.12)"
[ -x "$ASPECTD_VENV/bin/python" ] || "$UV" venv --python 3.12 "$ASPECTD_VENV"

pin() {  # print "name==version" for a package as pinned in requirements.txt
  grep -E "^$1==" "$REPO/requirements.txt" | awk '{print $1}'
}

if [ "$TIER" = cpu ]; then
  # Exactly the analysis subset of requirements.txt -- enough for reproduce_paper_cpu.sh.
  "$UV" pip install --python "$ASPECTD_VENV/bin/python" \
    "$(pin numpy)" "$(pin pandas)" "$(pin scipy)" "$(pin matplotlib)" pymupdf
else
  need_file "$REPO/requirements.txt"
  "$UV" pip install --python "$ASPECTD_VENV/bin/python" -r "$REPO/requirements.txt"

  # System dependency of phonemizer (the paper used espeak-ng 1.51).
  command -v espeak-ng >/dev/null || die "espeak-ng not installed (apt install espeak-ng / brew install espeak-ng)"

  # Gated dataset: amphion/Emilia-Dataset needs an HF token that has accepted its terms.
  [ -n "${HF_TOKEN:-}" ] || say "WARNING: HF_TOKEN unset -- 01_data.sh cannot download Emilia"

  # Primary SIM-o model (grid.json eval_models.speaker_sim.primary): the WavLM-large SV
  # checkpoint of microsoft/UniSpeech used by the seed-tts-eval protocol. It is not on PyPI.
  # src/evaluate.py does `sys.path.insert(0, $ASPECTD_MODELS/wavlm_sv)` and then
  # `from models.ecapa_tdnn import ECAPA_TDNN_SMALL`, and loads
  # $ASPECTD_MODELS/wavlm_sv/wavlm_large_finetune.pth. Place both there by hand:
  #   $ASPECTD_MODELS/wavlm_sv/models/ecapa_tdnn.py   (UniSpeech speaker-verification code)
  #   $ASPECTD_MODELS/wavlm_sv/wavlm_large_finetune.pth
  # The scorer refuses to silently fall back to microsoft/wavlm-base-plus-sv (it fails gate
  # G0(c) and once invented a +0.517 gain; commit 8ca5d12), so a missing file is fatal.
  if [ ! -f "$ASPECTD_MODELS/wavlm_sv/wavlm_large_finetune.pth" ] || \
     [ ! -f "$ASPECTD_MODELS/wavlm_sv/models/ecapa_tdnn.py" ]; then
    say "WARNING: primary SIM-o model not found under $ASPECTD_MODELS/wavlm_sv (see comments)"
  fi

  # Remaining eval models are fetched on first use and cached by HF / torch.hub:
  #   openai/whisper-large-v3 (WER), kyutai/mimi (codec), tarepan/SpeechMOS:v1.2.0
  #   utmos22_strong (UTMOS), s3prl/s3prl (WavLM-large front end of the SV model),
  #   and for the extensions openai/whisper-medium.en, microsoft/wavlm-base-plus-sv,
  #   facebook/wav2vec2-large-960h-lv60-self (src/asr2.py), speechbrain/resemblyzer encoders.
  "$PY" - <<'EOF'
import torch, transformers, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| transformers", transformers.__version__, "| numpy", numpy.__version__)
EOF
fi

"$ASPECTD_VENV/bin/python" -c 'import sys; assert sys.version_info >= (3, 12), sys.version'
say "done. Export ASPECTD_PY=$ASPECTD_VENV/bin/python (common.sh does this by default)."
