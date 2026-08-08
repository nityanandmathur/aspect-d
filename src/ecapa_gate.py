"""Instrument gate for ECAPA (task-v2.md §1.7) — run BEFORE any S2 number is used.

§1.7: any new scoring or selection model must pass G0(c) validation on ground
truth — for an SV model, same-speaker median >= 0.50 and cross-speaker median
<= 0.25 — before its numbers are used for anything. base-plus-sv failed exactly
this check and is excluded everywhere; ECAPA gets the same treatment, on the same
400 items, with the same speaker pairing and the same RNG as the frozen G0(c).

If ECAPA fails, S2 halts and the failure is posted (§4-S2: "do not substitute
silently").

    python src/ecapa_gate.py [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

import numpy as np
import soundfile as sf
import torch

from data import PROC_DIR, SR

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "artifacts-v1.2")
MODEL = "speechbrain/spkrec-ecapa-voxceleb"


def _read(p: str) -> np.ndarray:
    w, sr = sf.read(p, dtype="float32")
    assert sr == SR, f"{p} sr={sr}"
    return w


class Ecapa:
    """Same interface as evaluate.Scorer.embed: one utterance per forward pass,
    L2-normalised embeddings, 16 kHz input (the harness runs at 24 kHz)."""

    def __init__(self, device: str = "cuda:0"):
        from speechbrain.inference.speaker import EncoderClassifier
        self.device = torch.device(device)
        self.m = EncoderClassifier.from_hparams(
            source=MODEL, savedir=os.path.join(REPO, ".cache", "ecapa"),
            run_opts={"device": device})
        self.name = MODEL

    @torch.no_grad()
    def embed(self, wavs: List[np.ndarray]) -> torch.Tensor:
        import torchaudio.functional as AF
        out = []
        for w in wavs:
            x = AF.resample(torch.from_numpy(np.asarray(w, np.float32)), SR, 16000)
            e = self.m.encode_batch(x[None].to(self.device)).squeeze(0).squeeze(0)
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
        return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--items", type=int, default=400)
    a = ap.parse_args()

    items = json.load(open(os.path.join(PROC_DIR, "eval_zs.json")))[:a.items]
    aud = os.path.join(PROC_DIR, "eval_audio")
    tgt = [_read(os.path.join(aud, it["target_id"] + ".flac")) for it in items]
    prm = [_read(os.path.join(aud, it["prompt_id"] + ".flac")) for it in items]

    ec = Ecapa(a.device)
    e_t, e_p = ec.embed(tgt), ec.embed(prm)
    same = torch.nn.functional.cosine_similarity(e_t, e_p).numpy()

    # cross-speaker pairing reproduces the frozen G0(c) construction exactly
    spk = np.array([it["speaker"] for it in items])
    rng = np.random.default_rng(0)
    cross = []
    for i in range(len(items)):
        cand = np.where(spk != spk[i])[0]
        j = cand[rng.integers(len(cand))]
        cross.append(float(torch.nn.functional.cosine_similarity(e_t[i:i + 1], e_p[j:j + 1])))
    cross = np.array(cross)

    g0c = json.load(open(os.path.join(REPO, "artifacts", "g0c_groundtruth.json")))
    res = {
        "model": MODEL, "n_items": len(items),
        "rule": "task-v2.md §1.7 / protocol G0(c): same-speaker median >= 0.50 AND "
                "cross-speaker median <= 0.25",
        "sim_same_median": float(np.median(same)),
        "sim_cross_median": float(np.median(cross)),
        "discriminative_gap": float(np.median(same) - np.median(cross)),
        "pass_same": bool(np.median(same) >= 0.50),
        "pass_cross": bool(np.median(cross) <= 0.25),
        "reference_wavlm_large": {
            "sim_same_median": g0c["sim_same_median"],
            "sim_cross_median": g0c["sim_cross_median"],
            "gap": g0c["sim_same_median"] - g0c["sim_cross_median"]},
    }
    res["passes"] = bool(res["pass_same"] and res["pass_cross"])
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ecapa_gate.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "reference_wavlm_large"}, indent=1),
          flush=True)
    print(f"\n[gate] ECAPA same {res['sim_same_median']:.4f} (bar >=0.50), "
          f"cross {res['sim_cross_median']:.4f} (bar <=0.25) -> "
          f"{'PASS' if res['passes'] else 'FAIL'}", flush=True)
    print(f"[gate] reference wavlm-large: same {g0c['sim_same_median']:.4f}, "
          f"cross {g0c['sim_cross_median']:.4f}", flush=True)
    if not res["passes"]:
        print("[gate] S2 HALTS — §4-S2 forbids substituting another scorer silently",
              flush=True)


if __name__ == "__main__":
    main()
