"""Third speaker encoder from an independent family, gated then applied.

The open limitation on the search result is that the selector (WavLM-large SV) and
the scorer (ECAPA-TDNN) share the ECAPA-TDNN head and VoxCeleb-family supervision,
correlating at item level r = 0.71--0.73. Their agreement therefore bounds
representational overlap without excluding it.

This adds a **GE2E d-vector** (Wan et al., 2018; `resemblyzer`): an LSTM speaker
encoder trained with generalized end-to-end loss on log-mel filterbanks. It shares
neither the TDNN head, nor the WavLM front-end, nor the AAM-softmax objective of the
other two, so agreement across all three is much harder to attribute to a common
representation.

Gated exactly like ECAPA (task-v2.md §1.7 / protocol G0(c)): same-speaker median
>= 0.50 and cross-speaker median <= 0.25 on the 400 ground-truth eval items, with
the same speaker pairing and RNG as the frozen G0(c). A model that fails the gate
does not get a vote.

    python src/ge2e_gate.py            # gate only
    python src/ge2e_gate.py --score    # gate, then rescore the S2 search arms
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch

from data import PROC_DIR, SR

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "artifacts-v1.3")
RUNS = [f"C{i}_{s}" for i in range(1, 6) for s in (0, 1, 2)]
TIERS = [(128, 16, 2), (256, 32, 4), (512, 64, 8)]
N_ITEMS = 200
BOOT_RNG = 7331


def _read(p: str) -> np.ndarray:
    return sf.read(p, dtype="float32")[0]


class XVect:
    """speechbrain x-vector: a TDNN speaker encoder trained with a different objective
    and topology from ECAPA, and with no WavLM front-end."""

    def __init__(self, device: str = "cuda:0"):
        from speechbrain.inference.speaker import EncoderClassifier
        self.device = torch.device(device)
        self.m = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir=os.path.join(REPO, ".cache", "xvect"), run_opts={"device": device})
        self.name = "x-vector (speechbrain, VoxCeleb)"

    @torch.no_grad()
    def embed(self, wavs: List[np.ndarray]) -> torch.Tensor:
        import torchaudio.functional as AF
        out = []
        for w in wavs:
            x = AF.resample(torch.from_numpy(np.asarray(w, np.float32)), SR, 16000)
            e = self.m.encode_batch(x[None].to(self.device)).squeeze(0).squeeze(0)
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
        return torch.stack(out)


class GE2E:
    """Same embed() contract as the other encoders: L2-normalised, one clip at a time."""

    def __init__(self, device: str = "cuda:0"):
        from resemblyzer import VoiceEncoder
        self.m = VoiceEncoder(device.split(":")[0] if "cuda" not in device else "cuda")
        self.name = "GE2E d-vector (resemblyzer)"

    def embed(self, wavs: List[np.ndarray]) -> torch.Tensor:
        import librosa
        out = []
        for w in wavs:
            x = librosa.resample(np.asarray(w, np.float32), orig_sr=SR, target_sr=16000)
            e = torch.from_numpy(self.m.embed_utterance(x).astype(np.float32))
            out.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.stack(out)


def gate(enc, items, aud) -> Dict:
    tgt = [_read(os.path.join(aud, it["target_id"] + ".flac")) for it in items]
    prm = [_read(os.path.join(aud, it["prompt_id"] + ".flac")) for it in items]
    e_t, e_p = enc.embed(tgt), enc.embed(prm)
    same = torch.nn.functional.cosine_similarity(e_t, e_p).numpy()
    spk = np.array([it["speaker"] for it in items])
    rng = np.random.default_rng(0)
    cross = []
    for i in range(len(items)):
        cand = np.where(spk != spk[i])[0]
        j = cand[rng.integers(len(cand))]
        cross.append(float(torch.nn.functional.cosine_similarity(e_t[i:i + 1], e_p[j:j + 1])))
    cross = np.array(cross)
    res = {"model": enc.name, "n_items": len(items),
           "sim_same_median": float(np.median(same)),
           "sim_cross_median": float(np.median(cross)),
           "discriminative_gap": float(np.median(same) - np.median(cross)),
           "pass_same": bool(np.median(same) >= 0.50),
           "pass_cross": bool(np.median(cross) <= 0.25)}
    res["passes"] = bool(res["pass_same"] and res["pass_cross"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--encoder", default="ge2e", choices=["ge2e", "xvect"])
    a = ap.parse_args()

    items = json.load(open(os.path.join(PROC_DIR, "eval_zs.json")))
    aud = os.path.join(PROC_DIR, "eval_audio")
    enc = GE2E(a.device) if a.encoder == "ge2e" else XVect(a.device)
    os.makedirs(OUT, exist_ok=True)

    g = gate(enc, items, aud)
    with open(os.path.join(OUT, f"{a.encoder}_gate.json"), "w") as fh:
        json.dump(g, fh, indent=1)
    print(f"[{a.encoder}] same {g['sim_same_median']:.4f} (bar >=0.50)  "
          f"cross {g['sim_cross_median']:.4f} (bar <=0.25)  gap {g['discriminative_gap']:.4f}"
          f"  -> {'PASS' if g['passes'] else 'FAIL'}", flush=True)
    if not g["passes"]:
        print("[ge2e] FAILS the instrument gate — its numbers are not used (§1.7)", flush=True)
        return
    if not a.score:
        return

    # rescore the frozen S2 arms with the third encoder
    meta = {d["item"]: d for d in items}
    ids = sorted(meta)[:N_ITEMS]
    e_p = {i: enc.embed([_read(os.path.join(aud, meta[i]["prompt_id"] + ".flac"))])[0]
           for i in ids}
    todo = a.runs.split(",") if a.runs else RUNS
    rows = []
    for r in todo:
        for nfe, T, K in TIERS:
            ref = [_read(os.path.join(REPO, "results", "runs", r, f"synth_T{T}", f"{i}.flac")) for i in ids]
            Er = enc.embed(ref)
            sr_ = torch.nn.functional.cosine_similarity(
                Er, torch.stack([e_p[i] for i in ids])).numpy()
            # best-of-K, selection unchanged (WavLM-SV picked these already in runs_s2.csv)
            import pandas as pd
            sel = pd.read_csv(os.path.join(REPO, "results", "artifacts-v1.2", "s2_parts",
                                           f"runs__{r}.csv"))
            picks = sel[(sel.nfe == nfe) & (sel.arm == "search")].set_index("item")["pick"]
            srch = [_read(os.path.join(REPO, "results", "runs", r, f"synth_bok{int(picks[i])}",
                                       f"{i}.flac")) for i in ids]
            Es = enc.embed(srch)
            ss = torch.nn.functional.cosine_similarity(
                Es, torch.stack([e_p[i] for i in ids])).numpy()
            for k, i in enumerate(ids):
                rows.append({"run": r, "nfe": nfe, "item": i,
                             "refine_ge2e": float(sr_[k]), "search_ge2e": float(ss[k])})
        print(f"[ge2e] {r} done", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    parts = os.path.join(OUT, "ge2e_parts")
    os.makedirs(parts, exist_ok=True)
    df.to_csv(os.path.join(parts, f"{'-'.join(todo)}.csv"), index=False)
    print(f"[ge2e] wrote {len(df)} rows", flush=True)


if __name__ == "__main__":
    main()
