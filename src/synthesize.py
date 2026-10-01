"""ASPECT-D inference — zero-shot TTS from a released checkpoint.

Nothing here re-implements the paper's pipeline; every step calls the frozen code:

    text      data._phon_init / data._phon_chunk   espeak-ng en-us, stress, punctuation
              -> phone_vocab.json ids (OOV -> <unk>), as in data.stage_phonemize
    prompt    mono -> 24 kHz (librosa) -> data._peak_normalize (-1 dBFS)
              -> data._encode_batch (Mimi, 8 codebooks), as in data._encode_worker
    item      phonemes = phon(prompt_text) + <sp> + phon(text), capped at 512;
              n_target = round(chars(text) * sec_per_char * 12.5), clamped to [1, 187]
              -- exactly sample.load_items (LOG.md P0-2)
    sampler   sample.synth_batch: MaskGIT, level 1..8, T steps per level, NFE = 8T,
              cosine schedule, annealed Gumbel confidence noise, temperature 1.0, no CFG
    decode    Mimi decode of [prompt | target] then drop the prompt samples, as in
              sample.cmd_synth

    python src/synthesize.py --run C3_0 --text "Hello there." \
        --prompt-wav prompt.flac --prompt-text "Transcript of the prompt." --out out.wav

The prompt transcript is REQUIRED: the eval protocol conditions on it (the visible
phonemes are prompt text + target text) and training always shows the model the
transcript of the visible prompt audio.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
from data import FRAME_SAMPLES, PROMPT_DUR, SR  # noqa: E402
from model import FRAME_RATE_HZ, PHONEME_POS_CAP, AspectD  # noqa: E402
from sample import MAX_TARGET_FRAMES, synth_batch  # noqa: E402

HF_REPO = "nityanandmathur/aspect-d-masked-diffusion-tts"
# eval_zs prompts are COMPLETE clips of at most PROMPT_DUR[1] = 3.5 s, used whole with
# their own transcript (LOG.md P0-2). Longer prompts are cut to this many seconds.
MAX_PROMPT_SECONDS = PROMPT_DUR[1]
ASSETS = ("phone_vocab.json", "dataset.json")


# ------------------------------------------------------------------ loading
@dataclass
class TTS:
    model: AspectD
    mimi: object
    vocab: Dict[str, int]
    sec_per_char: float
    device: torch.device
    config: Dict
    source: str
    extra: Dict = field(default_factory=dict)


def pick_device(name: str = "auto") -> torch.device:
    """auto: cuda > mps > cpu. The sampler's RNG streams are per device, so one seed
    gives different (equally valid) audio on different devices."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fetch_run(run: str, repo: str = HF_REPO, revision: Optional[str] = None) -> Dict[str, str]:
    """Download one run folder (config.json, model.safetensors, run.json if present) and
    the shared root assets (phone_vocab.json, dataset.json). Returns local paths."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    run = run.strip("/")
    out = {"config": hf_hub_download(repo, f"{run}/config.json", revision=revision),
           "weights": hf_hub_download(repo, f"{run}/model.safetensors", revision=revision)}
    try:
        out["run"] = hf_hub_download(repo, f"{run}/run.json", revision=revision)
    except EntryNotFoundError:
        pass
    for a in ASSETS:
        out[a] = hf_hub_download(repo, a, revision=revision)
    return out


def local_run(ckpt_dir: str, repo: str = HF_REPO, revision: Optional[str] = None
              ) -> Dict[str, str]:
    """A local copy of one run folder. The shared assets are looked up in the folder and
    its ancestors (the HF layout keeps them at the repo root), else fetched from `repo`."""
    out = {"config": os.path.join(ckpt_dir, "config.json"),
           "weights": os.path.join(ckpt_dir, "model.safetensors")}
    for k in ("config", "weights"):
        if not os.path.exists(out[k]):
            raise FileNotFoundError(out[k])
    if os.path.exists(os.path.join(ckpt_dir, "run.json")):
        out["run"] = os.path.join(ckpt_dir, "run.json")
    for a in ASSETS:
        d = os.path.abspath(ckpt_dir)
        for _ in range(4):
            if os.path.exists(os.path.join(d, a)):
                out[a] = os.path.join(d, a)
                break
            d = os.path.dirname(d)
        if a not in out:
            from huggingface_hub import hf_hub_download
            out[a] = hf_hub_download(repo, a, revision=revision)
    return out


def load_model(paths: Dict[str, str], device: torch.device) -> Tuple[AspectD, Dict, Dict]:
    """Build AspectD from config.json and load the bf16 safetensors into fp32 params
    (strict: every key must match model.py)."""
    from safetensors.torch import load_file
    cfg = json.load(open(paths["config"]))
    run = json.load(open(paths["run"])) if "run" in paths else {}
    recipe = run.get("recipe") or "coarse"          # v1.0 records predate the field
    if recipe != "coarse":
        raise SystemExit(f"run recipe {recipe!r}: only the coarse-to-fine sampler "
                         f"(sample.synth_batch) is wired into synthesize.py")
    vocab = json.load(open(paths["phone_vocab.json"]))["vocab"]
    model = AspectD(cfg["width"], cfg["depth"], cfg["heads"], len(vocab))
    model.load_state_dict(load_file(paths["weights"]), strict=True)
    return model.to(device).eval(), cfg, vocab


def load_tts(run: Optional[str] = None, checkpoint: Optional[str] = None,
             device: str = "auto", repo: str = HF_REPO,
             revision: Optional[str] = None) -> TTS:
    """`run`: a folder of the HF repo, e.g. "C3_0" or "v1.1/D3_0". `checkpoint`: a local
    folder holding config.json + model.safetensors. Exactly one of the two."""
    if (run is None) == (checkpoint is None):
        raise ValueError("pass exactly one of run= or checkpoint=")
    dev = pick_device(device)
    paths = fetch_run(run, repo, revision) if run else local_run(checkpoint, repo, revision)
    model, cfg, vocab = load_model(paths, dev)
    spc = float(json.load(open(paths["dataset.json"]))["sec_per_char"])
    mimi = data._load_mimi(dev)
    return TTS(model, mimi, vocab, spc, dev, cfg, run or os.path.abspath(checkpoint),
               {"paths": paths})


# ------------------------------------------------------------------ conditioning
def _ensure_espeak() -> None:
    """phonemizer finds libespeak-ng via ctypes, which misses Homebrew's prefix on macOS."""
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    for p in ("/opt/homebrew/lib/libespeak-ng.dylib", "/usr/local/lib/libespeak-ng.dylib"):
        if os.path.exists(p):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = p
            return


def phonemize(texts: List[str]) -> List[List[str]]:
    """Phone strings per text, with the training pipeline's backend and tokenisation."""
    if data._BACKEND is None:
        _ensure_espeak()
        data._phon_init()
    return data._phon_chunk([t.strip() for t in texts])


def to_ids(phones: List[str], vocab: Dict[str, int]) -> Tuple[np.ndarray, int]:
    """Frozen-vocab ids; symbols unseen in training map to <unk> (=1) as in
    data.stage_phonemize. Returns (ids, number of OOV symbols)."""
    ids = [vocab.get(p, 1) for p in phones]
    return np.asarray(ids, dtype=np.int64), sum(p not in vocab for p in phones)


def target_frames(text: str, sec_per_char: float) -> int:
    """sample.load_items: chars(target_text) * sec_per_char * 12.5, clamped to [1, 187]."""
    n_chars = len(text.strip())
    return int(min(MAX_TARGET_FRAMES, max(1, round(n_chars * sec_per_char * FRAME_RATE_HZ))))


def make_item(prompt_ids: np.ndarray, target_ids: np.ndarray, prompt_tokens: np.ndarray,
              text: str, sec_per_char: float, sp_id: int) -> Dict:
    """The dict sample.synth_batch consumes, built exactly as sample.load_items does."""
    phon = np.concatenate([prompt_ids, [sp_id], target_ids])[:PHONEME_POS_CAP]
    tk = np.asarray(prompt_tokens, dtype=np.int64)
    return {"phonemes": phon, "prompt_tokens": tk, "n_prompt": len(tk),
            "n_target": target_frames(text, sec_per_char), "n_chars": len(text.strip())}


def load_prompt_audio(path: str, max_seconds: float = MAX_PROMPT_SECONDS
                      ) -> Tuple[np.ndarray, bool]:
    """Any rate / channel count -> mono 24 kHz, first `max_seconds`, peak -1 dBFS.
    Returns (waveform, was_cut)."""
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    n_max = int(round(max_seconds * SR))
    cut = len(x) > n_max
    return data._peak_normalize(x[:n_max]), cut


@torch.no_grad()
def encode_prompt(tts: TTS, wav24: np.ndarray) -> np.ndarray:
    """Mimi tokens [frames, 8] of a 24 kHz waveform (frames = floor(len / 1920))."""
    return data._encode_batch(tts.mimi, [wav24], tts.device)[0]


# ------------------------------------------------------------------ synthesis
@torch.no_grad()
def synthesize(tts: TTS, text: str, prompt_wav: str, prompt_text: str, steps: int = 16,
               seed: int = 0, max_prompt_seconds: float = MAX_PROMPT_SECONDS
               ) -> Tuple[np.ndarray, Dict]:
    """Returns (24 kHz float32 waveform of the generated target only, info dict)."""
    if not text.strip():
        raise ValueError("empty text")
    if not prompt_text.strip():
        raise ValueError("prompt_text is required (the transcript of the prompt audio)")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not 0 <= seed < 2 ** 31:
        raise ValueError("seed must be in [0, 2**31)")
    t0 = time.time()
    notes: List[str] = []
    wav, cut = load_prompt_audio(prompt_wav, max_prompt_seconds)
    if cut:
        notes.append(f"prompt cut to its first {max_prompt_seconds:g} s; prompt_text must "
                     f"transcribe only that part")
    tk_p = encode_prompt(tts, wav)
    ph_p, ph_t = phonemize([prompt_text, text])
    ids_p, oov_p = to_ids(ph_p, tts.vocab)
    ids_t, oov_t = to_ids(ph_t, tts.vocab)
    item = make_item(ids_p, ids_t, tk_p, text, tts.sec_per_char, tts.vocab.get("<sp>", 1))
    if len(ids_p) + 1 + len(ids_t) > PHONEME_POS_CAP:
        notes.append(f"phonemes truncated to {PHONEME_POS_CAP}")
    if round(item["n_chars"] * tts.sec_per_char * FRAME_RATE_HZ) > MAX_TARGET_FRAMES:
        notes.append(f"text implies > {MAX_TARGET_FRAMES / FRAME_RATE_HZ:g} s; length "
                     f"clamped to {MAX_TARGET_FRAMES} frames (speech will be rushed)")
    t1 = time.time()
    with warnings.catch_warnings():
        # synth_batch wraps forwards in autocast("cuda", bf16); off CUDA that is a no-op
        # that warns, and the model then runs in fp32
        warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
        warnings.filterwarnings("ignore", message=".*device_type of 'cuda'.*")
        grid, _ = synth_batch(tts.model, [item], steps, tts.device, 0, cand=seed)
    t2 = time.time()
    n_p, n_t = item["n_prompt"], item["n_target"]
    codes = grid[0, :n_p + n_t].T[None].to(tts.device)               # [1, 8, frames]
    full = tts.mimi.decode(codes).audio_values[0, 0].float().cpu().numpy()
    gen = full[n_p * FRAME_SAMPLES:]
    t3 = time.time()
    info = {"source": tts.source, "config_id": tts.config.get("config_id"),
            "width": tts.config["width"], "depth": tts.config["depth"],
            "T": steps, "nfe": 8 * steps, "seed": seed, "device": str(tts.device),
            "n_prompt_frames": n_p, "n_target_frames": n_t,
            "prompt_seconds": len(wav) / SR, "target_seconds": len(gen) / SR,
            "n_phonemes": int(len(item["phonemes"])), "oov_phonemes": oov_p + oov_t,
            "sec_per_char": tts.sec_per_char,
            "wall_seconds": {"prep": t1 - t0, "sampler": t2 - t1, "decode": t3 - t2,
                             "total": t3 - t0},
            "notes": notes}
    return gen, info


def main(argv: Optional[List[str]] = None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help=f"run folder in the HF repo, e.g. C3_0 (repo {HF_REPO})")
    src.add_argument("--checkpoint", help="local folder with config.json + model.safetensors")
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--revision", default=None, help="HF commit/tag to pin")
    ap.add_argument("--text", required=True, help="text to speak (English)")
    ap.add_argument("--prompt-wav", required=True, help="speaker prompt, any rate; "
                    f"a 2.5-{MAX_PROMPT_SECONDS:g} s clip is used whole")
    ap.add_argument("--prompt-text", required=True, help="exact transcript of the prompt")
    ap.add_argument("--steps", type=int, default=16, help="T, steps per level (NFE = 8T)")
    ap.add_argument("--seed", type=int, default=0, help="sampler stream (cand index)")
    ap.add_argument("--max-prompt-seconds", type=float, default=MAX_PROMPT_SECONDS)
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda[:i]")
    ap.add_argument("--out", default="out.wav")
    a = ap.parse_args(argv)
    tts = load_tts(a.run, a.checkpoint, a.device, a.repo, a.revision)
    wav, info = synthesize(tts, a.text, a.prompt_wav, a.prompt_text, a.steps, a.seed,
                           a.max_prompt_seconds)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sf.write(a.out, wav, SR)
    info["out"] = os.path.abspath(a.out)
    for n in info["notes"]:
        print(f"[synthesize] note: {n}", file=sys.stderr)
    print(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    main()
