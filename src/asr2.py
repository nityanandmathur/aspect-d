"""A second, different-family ASR, because the first one may be measuring the wrong thing.

Every intelligibility number in this project comes from Whisper-large-v3. The external
anchor made that a live problem rather than a theoretical one: F5-TTS scores WER 0.0248 on
our items against 0.0345 for the *real recordings*. A metric that a synthesiser beats human
speech on is measuring ASR-friendliness at least in part -- and that supplies a deflationary
reading of the whole training-compute trend, namely that intelligibility "saturates" because
the metric bottoms out rather than because refinement has finished.

Swapping Whisper-large for Whisper-medium would not answer it: same architecture, same
training data family, same failure modes. This uses wav2vec2 CTC, which shares neither the
encoder-decoder shape, the training objective, nor the LM-like decoder that lets Whisper
guess through unclear audio. If the saturation survives both, it is a property of the audio.

Writes `asr2.json` beside the existing `scores.json` and never touches it; the frozen
metric stack is unchanged.

    python src/asr2.py --dirs results/runs-v1.4/C1_0_180k/synth_T16 ...
    python src/asr2.py --manifest results/logs-v1.5/asr2_0.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import List

import numpy as np
import soundfile as sf
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "facebook/wav2vec2-large-960h-lv60-self"
SR_ASR = 16000


def normalise(s: str) -> str:
    """Match the project's WER convention: lowercase, strip punctuation, collapse space."""
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return " ".join(s.split())


class CTCScorer:
    def __init__(self, device: str = "cuda:0"):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.dev = torch.device(device)
        self.proc = Wav2Vec2Processor.from_pretrained(MODEL)
        self.m = Wav2Vec2ForCTC.from_pretrained(MODEL).to(self.dev).eval()

    @torch.no_grad()
    def transcribe(self, wavs: List[np.ndarray], sr: int, batch: int = 16) -> List[str]:
        import torchaudio.functional as AF
        out = []
        for i in range(0, len(wavs), batch):
            chunk = [AF.resample(torch.from_numpy(np.asarray(w, np.float32)), sr, SR_ASR)
                     for w in wavs[i:i + batch]]
            inp = self.proc([c.numpy() for c in chunk], sampling_rate=SR_ASR,
                            return_tensors="pt", padding=True)
            logits = self.m(inp.input_values.to(self.dev),
                            attention_mask=inp.attention_mask.to(self.dev)).logits
            ids = torch.argmax(logits, dim=-1)
            out.extend(self.proc.batch_decode(ids))
        return out


def score_dir(sc: CTCScorer, sdir: str, meta: dict) -> dict:
    import jiwer
    sj = json.load(open(os.path.join(sdir, "synth.json")))
    ids = sj.get("item_ids") or sorted(meta)[:sj["items"]]
    wavs, refs, keep = [], [], []
    for i in ids:
        p = os.path.join(sdir, f"{i}.flac")
        if not os.path.exists(p):
            continue
        w, sr = sf.read(p, dtype="float32")
        wavs.append(w)
        refs.append(normalise(meta[i]["target_text"]))
        keep.append(i)
    hyps = [normalise(h) for h in sc.transcribe(wavs, sr)]
    per = [jiwer.wer(r, h) if r.strip() else 1.0 for r, h in zip(refs, hyps)]
    return {"asr_model": MODEL, "n_items": len(keep),
            "wer_mean": float(np.mean(per)),
            "wer_se": float(np.std(per, ddof=1) / np.sqrt(len(per))),
            "wer_corpus": float(jiwer.wer(refs, hyps)),
            "items": [{"item": i, "wer": float(w), "hyp": h}
                      for i, w, h in zip(keep, per, hyps)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=[])
    ap.add_argument("--manifest")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    from data import PROC_DIR
    meta = {d["item"]: d for d in json.load(open(os.path.join(PROC_DIR, "eval_zs.json")))}
    dirs = a.dirs or json.load(open(a.manifest))
    sc = CTCScorer(a.device)
    for d in dirs:
        d = d if os.path.isabs(d) else os.path.join(REPO, d)
        out = os.path.join(d, "asr2.json")
        if os.path.exists(out):
            print(f"[asr2] skip {d}", flush=True)
            continue
        try:
            r = score_dir(sc, d, meta)
        except Exception as e:
            print(f"[asr2] FAIL {d}: {type(e).__name__} {e}", flush=True)
            continue
        json.dump(r, open(out, "w"))
        print(f"[asr2] {os.path.relpath(d, REPO)}  WER {r['wer_mean']:.4f} "
              f"(n={r['n_items']})", flush=True)


if __name__ == "__main__":
    main()
