# ASPECT-D: what test-time compute buys in masked-diffusion TTS

This repository holds the code, run records and analysis scripts for the paper

> **Refinement Buys Intelligibility, Search Buys Identity: What Test-Time Compute Buys in
> Masked-Diffusion TTS**
> Nityanand Mathur, Hamees Sayed, Ayush Pratap Singh.
> NeurIPS 2026 Workshop on Diffusion Language Models (DiffuLM), poster.

- Paper: <https://openreview.net/forum?id=E659lrDKOx>
- Models: <https://huggingface.co/nityanandmathur/aspect-d-masked-diffusion-tts>

## What we found

We trained 15 width-depth shapes of a masked-diffusion text-to-speech model (3 seeds each,
19M to 133M non-embedding parameters) on 2,000 hours of English speech from Emilia. We then
changed only the number of refinement steps T used at inference. Synthesis was scored on 400
zero-shot items from 174 speakers held out of training.

Going from T = 1 to T = 16 closes 86.2% of the reachable word-error range but only 46.4% of
the reachable speaker-similarity range. The reachable range runs to measured floors: the
ASR's error on the real recordings, and the similarity of real audio passed through the
codec. More refinement steps mostly fix the words; they do much less for whether the voice
sounds like the prompt speaker.

Speaker similarity does respond to search. At the same 128 generator forward passes,
best-of-2 sampling, with a speaker-verification model (WavLM-L) picking the candidate, beats
T = 16 by +0.0153 SIM-o when a different encoder (ECAPA-TDNN) scores the result, and wins on
57.4% of items. With the two encoders' roles swapped the gain is +0.0089. Search costs
+0.0408 absolute WER and 7.1-7.9% extra compute for selection. Retraining for 3x and 6x
as long shrinks the refinement asymmetry (1.84x, 1.36x and 1.23x at 30k, 90k and 180k
steps) but it stays above one. Part of the identity gap is outside the model's reach: the
Mimi codec itself loses 62% of the similarity gap between our best 90k-step model and real
audio. These results come from small models (19-133M non-embedding parameters), one English
corpus, one codec and automatic metrics only.

The numbers in this section are the values of macros in `paper/numbers*.tex`, which
`src/paper*.py` and `src/camera_ready_*.py` generate from the committed artifacts. [REPRODUCE.md](REPRODUCE.md) gives the script and input behind each one.

## Install

Inference needs Python 3.11 or newer, PyTorch, and the `espeak-ng` system package. The
scripts that build the paper, and the `pip install -e .` route below, need Python 3.12.

```bash
git clone https://github.com/nityanandmathur/aspect-d.git
cd aspect-d
brew install espeak-ng                  # Linux: sudo apt-get install espeak-ng
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch transformers safetensors huggingface_hub \
    phonemizer soundfile librosa numpy pandas
source .venv/bin/activate
```

We tested these commands on macOS (Apple silicon) with Python 3.12, and the same packages on
Python 3.11, getting torch 2.14.1, transformers 5.18.0 and espeak-ng 1.52.0. Inference has
not been tested on Linux or CUDA.

Other ways to install:

- `pip install -e ".[eval,analysis,dev]"` installs the dependencies declared in
  [pyproject.toml](pyproject.toml) (core, scoring, fits and figures, tests). It does not
  install a Python package; the scripts still run as `python src/<script>.py`. The `eval`
  extra pins torch to 2.11 so that it matches torchaudio 2.11.0, torchaudio's last release.
- `pip install -r requirements.txt` installs the paper's pinned environment (Python 3.12,
  torch 2.11.0, transformers 5.14.1, numpy 2.4.6). Only those versions and espeak-ng 1.51
  are on record. The other pins are reconstructed, and the file says which is which.

On macOS, `src/synthesize.py` points phonemizer at Homebrew's `libespeak-ng.dylib` when
`PHONEMIZER_ESPEAK_LIBRARY` is unset.

## Quickstart: synthesize speech

```bash
python src/synthesize.py --run C3_0 \
    --text "Masked diffusion writes every frame at once, then refines it." \
    --prompt-wav samples/it0000_prompt.flac \
    --prompt-text "Yep. So, shouldn't you just get network plus so that you get that?" \
    --steps 16 --seed 0 --out out.wav
```

The first call downloads the run folder and the Mimi codec (about 0.7 GB) into the Hugging
Face cache. The script writes a 24 kHz mono WAV that holds only the new speech, not the
prompt, and prints a JSON record with the frame counts, the number of forward passes, any
notes, and wall time per stage. On an Apple M5 Max (MPS), C3_0 at T = 16 runs faster than
real time once the model is loaded.

From Python:

```python
import sys; sys.path.insert(0, "src")
import soundfile as sf
from synthesize import SR, load_tts, synthesize

tts = load_tts(run="C3_0", device="auto")        # or load_tts(checkpoint="path/to/C3_0")
wav, info = synthesize(tts, text="Hello there, this is a test.",
                       prompt_wav="samples/it0000_prompt.flac",
                       prompt_text="Yep. So, shouldn't you just get network plus so that you get that?",
                       steps=16, seed=0)
sf.write("out.wav", wav, SR)
```

Things to know:

- **The prompt transcript is required** and must match the prompt audio word for word. A
  prompt of up to 3.5 s is used whole. A longer one is cut to its first 3.5 s
  (`--max-prompt-seconds`), and the transcript must then cover only the part that is kept.
  If the transcript and audio disagree, the model tends to speak the extra words.
- **Output length is fixed before sampling**, from the character count of the text. There is
  no duration predictor. Text longer than about 15 s of speech is squeezed into 15 s, so
  synthesize long text one sentence at a time.
- `--steps` is T, the number of steps per codebook level. The model runs 8T forward passes.
  `--device auto` picks CUDA, then MPS, then CPU.
- The sampler is the one frozen for the paper: no classifier-free guidance, temperature 1.
  The same seed on the same device gives the same audio. Different devices give different
  audio.
- A single call does not reproduce the paper's evaluation WAVs bit for bit. The paper
  sampled 50 items per batch in bf16 on B200 GPUs, and the batch shape changes the noise.
- The paper's phonemes came from espeak-ng 1.51. Other versions can phonemize some words
  differently. `oov_phonemes` in the JSON output counts phonemes the model never saw in
  training; they become `<unk>`.
- English only.

**Use responsibly.** This is zero-shot voice cloning from about 3 s of audio. Only clone
voices whose owners have agreed to it. Do not use it to impersonate anyone, to deceive, or
to get past voice authentication, and say that the audio is synthetic. No watermark is
applied.

## Released models

Every folder on the [Hugging Face repo](https://huggingface.co/nityanandmathur/aspect-d-masked-diffusion-tts)
holds `config.json`, `model.safetensors` (bf16) and `run.json`. Pass the folder name to
`--run`.

| Folder | Count | What it is |
|---|---|---|
| `A1_0` ... `C5_2` | 45 | The main grid, named `<config>_<seed>`. Budgets A, B and C target 20M, 50M and 125M non-embedding parameters. Within a budget, shape 1 is the widest and shape 5 the deepest ([configs/grid.json](configs/grid.json) lists width and depth). Seeds 0 to 2, 30k training steps. |
| `sweep_g1_*`, `sweep_g1b_*` | 20 | 3k-step learning-rate sweep proxies: width transfer (`w256`, `w640`) and depth transfer (`d4`, `d24`). Not meant for synthesis. |
| `v1.1/D1_0` ... `v1.1/D5_1` | 10 | The 276M budget: five shapes, two seeds, 30k steps. |
| `v1.1/C1_0_90k`, `v1.1/C3_0_90k`, `v1.1/C5_0_90k` | 3 | C1, C3 and C5 (seed 0) trained for 90k steps, three times the grid schedule. |
| `v1.1/lrsweep_D3_*`, `v1.1/proxy_D3_g1d` | 5 | 3k-step learning-rate checks for the 276M budget. |

Files at the root of the model repo:

- `runs.csv`: one row per grid run and T in {1, 2, 4, 8, 16}, 225 rows.
- `fits.json`: the pre-registered fits.
- `grid.json`: the shape grid.
- `dataset.json` and `phone_vocab.json`: needed for inference.
- `protocol.html`: the experimental protocol.

**Which checkpoint to use.** Start with `C3_0` (width 768, depth 18). In `runs.csv`, C3 has
the highest mean speaker similarity at T = 16 over its three seeds, and seed 0 is the best
of the three on both speaker similarity and WER. If intelligibility matters more than voice
match, C5 (depth 36) has the lowest mean WER at T = 16. A1, the smallest and widest model,
is barely intelligible. `v1.1/C3_0_90k` has the same shape trained three times longer, and
`--run v1.1/C3_0_90k` works the same way. In the paper, longer training is the largest
identity lever.

The other extension runs have no weights on the Hub: the remaining six 90k runs, the 180k
runs and the scale-up run. Model weights and audio are licensed separately from the code;
see [License](#license).

## Reproducing the paper

[REPRODUCE.md](REPRODUCE.md) covers three tiers, and the shell recipes are in
[recipes/](recipes/):

| Tier | What it does | Needs |
|---|---|---|
| A | Regenerates every number, table and figure the scripts produce from the committed run records, and diffs them against the paper sources | CPU and Python 3.12; a few minutes on a laptop |
| B | Re-scores audio (WER, speaker similarity, UTMOS), then runs tier A | CUDA GPU, the evaluation models, and evaluation-set files that are not yet released |
| C | Retrains from scratch: data, the 45-run grid, synthesis, scoring, fits | 8x B200-class GPUs and access to Emilia |

Tier A:

```bash
TIER=cpu bash recipes/00_env.sh            # numpy, pandas, scipy, matplotlib, pymupdf into .venv
DATASET_JSON=/path/to/dataset.json bash recipes/reproduce_paper_cpu.sh
```

`dataset.json` is at the root of the model repo. The check works on a scratch copy and
never writes to the checkout. It exits 0 if everything matches, 1 if anything differs, and
2 if nothing differs but some items cannot be rebuilt from the release. REPRODUCE.md, under
"Known deviations and gaps", lists the items that currently differ or cannot be rebuilt,
the compute behind each tier, and which script produces each table and figure.

Tiers B and C were documented from the code and the run records. They were not re-run for
this release. Run them in a scratch copy, because several scripts write fixed artifact paths
in place.

For scoring, the primary speaker-similarity model is the WavLM-large speaker-verification
checkpoint from microsoft/UniSpeech, the one used by the seed-tts-eval protocol. It is not
on PyPI. Place `wavlm_large_finetune.pth` and UniSpeech's `models/ecapa_tdnn.py` under
`$ASPECTD_MODELS/wavlm_sv/` by hand (see [recipes/00_env.sh](recipes/00_env.sh)). The other
evaluation models download on first use.

### Environment variables

Every machine-specific path is an environment variable with a default inside the repo:

| Variable | Default | Used for |
|---|---|---|
| `ASPECTD_DATA` | `<repo>/data` | Training data root (`emilia_raw/`, `proc/`) |
| `ASPECTD_MODELS` | `<repo>/models` | Evaluation models; `src/evaluate.py` loads the speaker-similarity model from `$ASPECTD_MODELS/wavlm_sv` |
| `ASPECTD_XL` | `<repo>/data-xl` | Data root of the scale-up corpus only |
| `ASPECTD_PY` | the running interpreter | Interpreter the job launchers use for child processes |
| `ASPECTD_VENV` | the active venv | Project venv that `src/anchor_f5.py` must not install into |
| `ASPECT_SCRATCH` | `<repo>/.cache/anchor_f5` | Scratch space for the F5-TTS anchor's separate venv |
| `ASPECTD_NETWORK_TESTS` | unset | Set to `1` to run the tests that download from the Hub |
| `HF_TOKEN` | unset | Needed only to download the gated Emilia dataset or to upload |

The recipes set `ASPECTD_VENV` to `<repo>/.venv` and `ASPECTD_PY` to that venv's Python
unless you set them ([recipes/common.sh](recipes/common.sh)).

## Tests

```bash
uv pip install --python .venv/bin/python pytest
python -m pytest -m "not network" -q tests                    # offline, CPU only
ASPECTD_NETWORK_TESTS=1 python -m pytest -m network -q tests  # downloads A1_0 from the Hub
```

The offline tests check the parameter-count formula against every grid config and committed
run record, the model's forward pass and attention masking, the sampler (exactly 8T
forward passes, no masked cells left), and that the inference input matches the input the
evaluation code builds for the same item. The CI workflow in
[.github/workflows/ci.yml](.github/workflows/ci.yml) runs the offline tests on Ubuntu 24.04
(whose espeak-ng is 1.51) with Python 3.11 and 3.12.

## Repository map

| Path | What it is |
|---|---|
| `src/` | All code, run as `python src/<script>.py`. Main entry points: `synthesize.py` (inference), `data.py` (data pipeline), `model.py`, `train.py`, `sample.py` (the frozen sampler), `evaluate.py` (scoring and `runs.csv`), `fit.py` (fits and bootstrap), `figures.py`. `paper.py`, `paper_v14.py`, `paper_v15.py` and `camera_ready_*.py` write the paper's macro files and tables. Scripts named after a hypothesis group (`e1_extended.py`, `s2_search.py`, `t23_analysis.py`, `v14_analysis.py`, ...) are the extension analyses. |
| `tests/` | pytest suite |
| `recipes/` | Numbered reproduction recipes `00_env.sh` to `06_extensions.sh`, and `reproduce_paper_cpu.sh` |
| `configs/grid.json` | Architecture, training recipe, frozen sampler settings, T grid |
| `runs/`, `runs-v1.1/`, `runs-v1.3/`, `runs-v1.5/` | Per-run records: `run.json`, and `synth.json` and `scores.json` in each `synth_*/` folder. No model weights. The only audio is `runs-v1.5/f5tts_anchor/`, 400 clips from the third-party F5-TTS model used as an external reference. |
| `artifacts/` | Main-grid outputs: `runs.csv`, `fits.json`, figures |
| `artifacts-v1.1/` ... `artifacts-v1.5/`, `artifacts-camera/` | Outputs of the extension studies and the camera-ready analyses |
| `paper/` | LaTeX sources of the paper and the generated `numbers*.tex` and `tab_*.tex` files. The OpenReview PDF is the version of record. |
| `samples/` | Audio for four evaluation items: prompt, ground truth, and syntheses at T = 1 and 16 from nine configs. `samples/index.html` is a listening page. |
| `docs/` | Pre-registrations and the research log; see [docs/README.md](docs/README.md) |
| `PREREGISTRATION.md` | The v1.0 pre-registration, frozen before any training run |
| `REPRODUCE.md` | Reproduction guide, compute disclosure, known gaps |
| `requirements.txt`, `pyproject.toml` | Pinned paper environment; dependency declaration |
| `LICENSE`, `MODEL_LICENSE.md`, `CITATION.cff` | Code license, weights and audio license, citation metadata |
| `index.html`, `protocol.html`, `implementation.html`, `results.html`, `extensions.html`, `coordinate-audit.html` | Project pages written during the research. `protocol.html` holds the thresholds and gates. |
| `LOG.md`, `LOG-v1.1.md`, `DECISION.md`, `RESULTS-FEED.md`, `state.json`, `state-v1.json`, `logs-v1.5/`, `.claims-v1.5/` | Research records that scripts read or write at the repo root |
| `CLAUDE.md` | Guidelines for the coding agent used during the project |

`src/push_hf.py`, `src/paper_sync.py` and `src/finalize_v11.sh` upload to or push to the
authors' Hugging Face and GitHub repositories. You do not need them, and they will fail
without the authors' credentials.

## Pre-registration

Each hypothesis was written down and committed before the data that tests it existed.
[PREREGISTRATION.md](PREREGISTRATION.md) covers the main grid. The four later addenda are in
[docs/preregistration/](docs/preregistration/), with the commit that froze each one.
REPRODUCE.md, under "Pre-registration ledger", says where each verdict is recorded.

## Citation

```bibtex
@inproceedings{mathur2026refinement,
  title     = {Refinement Buys Intelligibility, Search Buys Identity: What Test-Time
               Compute Buys in Masked-Diffusion {TTS}},
  author    = {Mathur, Nityanand and Sayed, Hamees and Singh, Ayush Pratap},
  booktitle = {NeurIPS 2026 Workshop on Diffusion Language Models: Foundations,
               Efficiency, and Reasoning (DiffuLM)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=E659lrDKOx},
  note      = {Poster}
}
```

[CITATION.cff](CITATION.cff) has the same information in machine-readable form.

## License

- Code: MIT, see [LICENSE](LICENSE).
- Model weights, generated audio, the Emilia clips in `samples/`, and dataset-derived files:
  CC BY-NC 4.0, see [MODEL_LICENSE.md](MODEL_LICENSE.md). The training data (Emilia) is
  licensed for non-commercial use only, and so are the weights.

## Acknowledgments

This work builds on:

- **Emilia** (He et al., IEEE SLT 2024): the training data.
- **Mimi** (Kyutai): the audio codec the models predict tokens for.
- **Whisper-large-v3** (OpenAI): the ASR for word error rate.
- **WavLM** (Microsoft) through the UniSpeech speaker-verification checkpoint and the
  seed-tts-eval protocol: speaker similarity.
- **UTMOS22** (SaruLab), loaded through SpeechMOS: the MOS proxy.
- **espeak-ng** and **phonemizer**: the text front end.
