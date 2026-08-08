"""S1 data prep — prompt-context variants at {1.5, 3, 6, 9} s (task-v2.md §4-S1).

**Documented deviation from the literal spec, and why.** §4-S1 says to rebuild the
variants "from each eval item's source utterance". In this corpus that is not
possible: `eval_audio/<prompt_id>.flac` already *is* the untrimmed source clip
(`data.stage_eval_audio` writes the whole clip), and the selection stage picked
clips of 3.0--3.5 s, so no eval prompt contains 6 s or 9 s of audio to cut.

Longer context therefore has to come from *other utterances by the same speaker*,
which are not in the token store either — eval speakers are held out of training,
so only their two eval clips were ever tokenized. This script builds the missing
data end to end, exactly as the main pipeline would have:

    raw tar -> mono -> resample to 24 kHz -> peak-normalise -> Mimi encode (8 levels)
    text -> espeak phonemes -> the frozen phone vocabulary

Arms:
  1.5 s  the canonical prompt truncated (tokens cut at 1.5 s; phonemes cut
         proportionally — there is no forced alignment, and that approximation is
         the arm's known weakness, recorded here rather than hidden)
  3 s    the canonical v1.0 prompt, byte-identical — asserted, not assumed
  6 s    canonical prompt + same-speaker continuation clips to >= 6 s
  9 s    the same, to >= 9 s
Continuation clips exclude every eval prompt_id and target_id in the whole eval
set, so no item's context can leak another item's prompt or target.

    python src/s1_prepare.py [--device cuda:0]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from typing import Dict, List

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from data import PROC_DIR, RAW_DIR, SR, TokenStore, _peak_normalize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "artifacts-v1.2")
S1_DIR = os.path.join(PROC_DIR, "s1_context")
ARMS = [1.5, 3.0, 6.0, 9.0]
FPS = 12.5                      # Mimi frame rate
N_LEVELS = 8


def pick_continuations(meta: pd.DataFrame, items: List[Dict]) -> Dict[str, List[str]]:
    """Per item, same-speaker clips (never any eval clip) ordered by id, enough for 9 s."""
    used = {i["prompt_id"] for i in items} | {i["target_id"] for i in items}
    by_spk: Dict[str, pd.DataFrame] = {s: g.sort_values("id")
                                       for s, g in meta.groupby("speaker")}
    need = 9.0
    out: Dict[str, List[str]] = {}
    for it in items:
        g = by_spk.get(it["speaker"])
        if g is None:
            out[it["item"]] = []
            continue
        cand = g[~g.id.isin(used)]
        take, tot = [], 0.0
        for r in cand.itertuples():
            take.append(r.id)
            tot += float(r.duration)
            if tot >= need:
                break
        out[it["item"]] = take
    return out


def extract(meta: pd.DataFrame, ids: List[str]) -> Dict[str, np.ndarray]:
    """Pull raw audio for `ids` out of the Emilia tars, matching the v1.0 recipe."""
    import librosa
    sub = meta[meta.id.isin(set(ids))]
    got: Dict[str, np.ndarray] = {}
    for shard, g in sub.groupby("shard"):
        want = dict(zip(g.member, g.id))
        p = os.path.join(RAW_DIR, shard + ".tar")
        if not os.path.exists(p):
            print(f"[s1] MISSING shard {shard}", flush=True)
            continue
        with tarfile.open(p) as tf:
            for m in tf:
                if not m.name.endswith(".mp3") or m.name[:-4] not in want:
                    continue
                x, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                if sr != SR:
                    x = librosa.resample(x, orig_sr=sr, target_sr=SR)
                got[want[m.name[:-4]]] = _peak_normalize(x)
    return got


def phonemize(texts: List[str]) -> List[List[str]]:
    import data as D
    D._phon_init()
    return D._phon_chunk(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--items", type=int, default=400)
    a = ap.parse_args()
    from transformers import MimiModel

    os.makedirs(S1_DIR, exist_ok=True)
    items = json.load(open(os.path.join(PROC_DIR, "eval_zs.json")))[:a.items]
    meta = pd.read_parquet(os.path.join(PROC_DIR, "meta.parquet"))
    store = TokenStore()
    vocab = store.vocab

    cont = pick_continuations(meta, items)
    need_ids = sorted({c for v in cont.values() for c in v})
    print(f"[s1] {len(need_ids)} continuation clips to encode "
          f"({len(items)} items, median {np.median([len(v) for v in cont.values()]):.0f} per item)",
          flush=True)

    wavs = extract(meta, need_ids)
    print(f"[s1] extracted {len(wavs)}/{len(need_ids)} from the raw tars", flush=True)
    txt = dict(zip(meta.id, meta.text))
    phon_list = phonemize([txt[i] for i in need_ids if i in wavs])
    ok_ids = [i for i in need_ids if i in wavs]
    phones = {i: p for i, p in zip(ok_ids, phon_list)}

    dev = torch.device(a.device)
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(dev).eval()
    toks: Dict[str, np.ndarray] = {}
    for k, cid in enumerate(ok_ids):
        w = wavs[cid]
        x = torch.from_numpy(np.asarray(w, np.float32))[None, None].to(dev)
        with torch.no_grad():
            c = mimi.encode(x, num_quantizers=N_LEVELS).audio_codes
        toks[cid] = c.to(torch.int16)[0].cpu().numpy().T        # [frames, 8]
        if (k + 1) % 200 == 0:
            print(f"[s1] encoded {k+1}/{len(ok_ids)}", flush=True)

    unk = vocab.get("<unk>", 0)
    built, dropped = {}, []
    for it in items:
        ph_p, tk_p = store.get(it["prompt_id"])
        arms: Dict[str, Dict] = {}
        for L in ARMS:
            nf = int(round(L * FPS))
            if L == 3.0:
                # the 3 s arm IS the canonical v1.0 prompt, used unchanged. v1.0 prompts
                # are whole clips of 3.02-3.50 s, so truncating them to exactly 38 frames
                # would NOT be bit-identical to v1.0 -- it would be a fifth condition.
                arms[str(L)] = {"tokens": tk_p.tolist(), "phonemes": ph_p.tolist(),
                                "seconds": len(tk_p) / FPS, "n_clips": 1}
            elif L < 3.0:
                # truncate the canonical prompt; phonemes cut in the same proportion
                keep = min(nf, len(tk_p))
                frac = keep / max(1, len(tk_p))
                nph = max(1, int(round(len(ph_p) * frac)))
                arms[str(L)] = {"tokens": tk_p[:keep].tolist(),
                                "phonemes": ph_p[:nph].tolist(),
                                "seconds": keep / FPS, "n_clips": 1}
            else:
                tk = [tk_p]
                ph = [ph_p]
                total = len(tk_p)
                nclip = 1
                for cid in cont[it["item"]]:
                    if total >= nf or cid not in toks:
                        break
                    tk.append(toks[cid])
                    ph.append(np.array([vocab.get(p, unk) for p in phones[cid]],
                                       dtype=ph_p.dtype))
                    total += len(toks[cid])
                    nclip += 1
                if total < nf:
                    arms[str(L)] = None
                    continue
                TK = np.concatenate(tk)[:nf]
                arms[str(L)] = {"tokens": TK.tolist(),
                                "phonemes": np.concatenate(ph).tolist(),
                                "seconds": len(TK) / FPS, "n_clips": nclip}
        if any(arms[str(L)] is None for L in ARMS):
            dropped.append(it["item"])
        built[it["item"]] = arms

    complete = [i for i in built if i not in dropped]
    print(f"[s1] items with all four arms: {len(complete)}/{len(items)} "
          f"(dropped {len(dropped)})", flush=True)

    # the 3 s arm must be byte-identical to the v1.0 prompt -- assert, do not assume
    nbad = 0
    for it in items:
        _, tk_p = store.get(it["prompt_id"])
        arm3 = built[it["item"]]["3.0"]
        if arm3 is None or not np.array_equal(
                np.asarray(arm3["tokens"], np.int64).reshape(-1, N_LEVELS),
                np.asarray(tk_p, np.int64).reshape(-1, N_LEVELS)):
            nbad += 1
    print(f"[s1] 3 s arm identical to the canonical v1.0 prompt tokens: "
          f"{len(items) - nbad}/{len(items)}", flush=True)

    with open(os.path.join(S1_DIR, "arms.json"), "w") as fh:
        json.dump({"arms_s": ARMS, "built": built, "dropped": dropped,
                   "n_complete": len(complete),
                   "three_s_identical": len(items) - nbad,
                   "note": "6/9 s arms concatenate same-speaker continuation clips; no "
                           "eval prompt or target is ever used as context"}, fh)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "s1_prep.json"), "w") as fh:
        json.dump({"n_items": len(items), "n_complete": len(complete),
                   "n_dropped": len(dropped), "dropped": dropped[:50],
                   "n_continuation_clips": len(ok_ids),
                   "three_s_identical": len(items) - nbad,
                   "mean_seconds_per_arm": {
                       str(L): float(np.mean([built[i][str(L)]["seconds"]
                                              for i in complete])) for L in ARMS}}, fh, indent=1)
    print(f"[s1] wrote {S1_DIR}/arms.json", flush=True)


if __name__ == "__main__":
    main()
