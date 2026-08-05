"""Gate G0 checks — protocol.html §8 (a)–(e) plus the task.md Phase-0 param-count test.

    python src/gates.py all --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch

from data import PROC_DIR, SR, PROMPT_FRAMES, TokenStore
from model import MASK_ID, N_LEVELS, AspectD, build_model, config_by_id, load_grid

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def _mel(x: np.ndarray, sr: int = SR, n_mels: int = 80) -> np.ndarray:
    import librosa
    m = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=1024, hop_length=256, n_mels=n_mels)
    return np.log(m + 1e-6)


# --------------------------------------------------------------------- G0 (a)
def g0a(device: str, n: int = 10) -> Dict:
    """Mimi roundtrip on 10 clips: audible output, mel distance far below a cross-clip
    control (no absolute threshold exists in the protocol → relative control, logged)."""
    from transformers import MimiModel
    store = TokenStore()
    ids = store.index[store.index.split == "train"].sort_values("id").id.tolist()[:n]
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    os.makedirs(os.path.join(ART, "g0_roundtrip"), exist_ok=True)
    rows = []
    mels_orig, mels_rt = [], []
    for cid in ids:
        _, tk = store.get(cid)
        codes = torch.from_numpy(tk.T[None]).to(device)
        with torch.no_grad():
            wav = mimi.decode(codes).audio_values[0, 0].float().cpu().numpy()
        # the "original" reference here is the codec input we stored tokens from: re-read
        # the eval-audio copy when present, else compare against the token-decoded RMS only
        sf.write(os.path.join(ART, "g0_roundtrip", f"{cid}_rt.flac"), wav, SR, format="FLAC")
        rows.append({"id": cid, "rt_rms": float(np.sqrt((wav ** 2).mean())),
                     "rt_seconds": len(wav) / SR, "n_frames": int(len(tk))})
        mels_rt.append(_mel(wav))
    # mel distance against the true waveform for the eval-set clips (originals on disk)
    ev = sorted(os.listdir(os.path.join(PROC_DIR, "eval_audio")))[:n] \
        if os.path.isdir(os.path.join(PROC_DIR, "eval_audio")) else []
    pair, ctrl = [], []
    for k, f in enumerate(ev):
        cid = f[:-5]
        if cid not in store.pos:
            continue
        x, _ = sf.read(os.path.join(PROC_DIR, "eval_audio", f), dtype="float32")
        _, tk = store.get(cid)
        with torch.no_grad():
            y = mimi.decode(torch.from_numpy(tk.T[None]).to(device)).audio_values[0, 0]
        y = y.float().cpu().numpy()
        L = min(len(x), len(y))
        mx, my = _mel(x[:L]), _mel(y[:L])
        pair.append(float(np.abs(mx - my).mean()))
        if k:
            xp, _ = sf.read(os.path.join(PROC_DIR, "eval_audio", ev[k - 1]), dtype="float32")
            L2 = min(len(x), len(xp))
            ctrl.append(float(np.abs(_mel(x[:L2]) - _mel(xp[:L2])).mean()))
    out = {"n_clips": len(rows), "clips": rows,
           "mel_L1_roundtrip_mean": float(np.mean(pair)) if pair else None,
           "mel_L1_crossclip_control_mean": float(np.mean(ctrl)) if ctrl else None,
           "audible": bool(all(r["rt_rms"] > 1e-3 for r in rows))}
    out["passes"] = bool(out["audible"] and (out["mel_L1_roundtrip_mean"] is None or
                        out["mel_L1_roundtrip_mean"] < 0.5 * out["mel_L1_crossclip_control_mean"]))
    return out


# --------------------------------------------------------------------- G0 (b)
def g0b(device: str, steps: int = 200, cfg_id: str = "A3", batch: int = 32,
        lr: float = 0.004) -> Dict:
    """Single-batch overfit must reduce masked CE by ≥ 40 %."""
    from train import assemble, build_inputs, draw_masks, masked_ce
    store = TokenStore()
    pos = np.where(store.index.split.values == "train")[0][:batch]
    ph, phm, tok, fm = assemble(store, pos)
    model = build_model(cfg_id, len(store.vocab)).to(device)
    opt = torch.optim.AdamW(model.param_groups(lr, 0.1), betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(4242)
    levels, ratios, cells = draw_masks(gen, len(pos), tok.shape[1])
    t = {k: torch.from_numpy(v).to(device) for k, v in
         {"ph": ph, "phm": phm, "tok": tok, "fm": fm}.items()}
    inp, loss_cells = build_inputs(t["tok"], t["fm"], levels, ratios, cells.to(device),
                                  PROMPT_FRAMES)
    n_cells = int(loss_cells.sum())
    hist = []
    for s in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = masked_ce(model, t["ph"], t["phm"], inp, t["fm"], t["tok"],
                          levels.to(device), loss_cells)
        opt.zero_grad(set_to_none=True)
        (l / n_cells).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append(float(l) / n_cells)
    red = 1.0 - hist[-1] / hist[0]
    return {"config": cfg_id, "batch": int(batch), "steps": steps, "lr": lr,
            "ce_first": hist[0], "ce_last": hist[-1], "reduction": float(red),
            "cells": n_cells, "curve": hist[::10], "passes": bool(red >= 0.40)}


# --------------------------------------------------------------------- G0 (e)
def g0e(device: str, n_items: int = 20, cfg_id: str = "A3") -> Dict:
    """Sampler-integrity on an UNTRAINED model: T=1 vs T=16 must differ on >20 % of cells."""
    from sample import batches, load_items, synth_batch
    store = TokenStore()
    model = build_model(cfg_id, len(store.vocab)).to(device).eval()
    items = load_items(store, n_items)
    grids = {}
    for T in (1, 16):
        outs = []
        for bi, b in enumerate(batches(items, 20)):
            g, fm = synth_batch(model, b, T, torch.device(device), bi)
            outs.append((g, fm, b))
        grids[T] = outs
    diff, tot = 0, 0
    for (g1, fm1, b1), (g16, fm16, _) in zip(grids[1], grids[16]):
        for i, it in enumerate(b1):
            s = it["n_prompt"]
            e = s + it["n_target"]
            a, c = g1[i, s:e].numpy(), g16[i, s:e].numpy()
            diff += int((a != c).sum())
            tot += a.size
    frac = diff / max(1, tot)
    return {"config": cfg_id, "n_items": len(items), "differing_cell_fraction": float(frac),
            "threshold": 0.20, "passes": bool(frac > 0.20)}


# ------------------------------------------------------------- param-count test
def param_test(sample_ids: List[str] = ("A5", "B3", "C1")) -> Dict:
    grid = load_grid()
    store = TokenStore()
    rows = []
    for cid in sample_ids:
        c = config_by_id(cid, grid)
        m = AspectD(c["width"], c["depth"], c["heads"], len(store.vocab))
        got, want = m.nonembed_params(), 12 * c["depth"] * c["width"] ** 2
        rows.append({"config": cid, "measured_nonembed": got, "formula_12dw2": want,
                     "grid_json_nonembed": c["nonembed_params"],
                     "rel_dev_vs_formula": abs(got - want) / want,
                     "rel_dev_vs_grid": abs(got - c["nonembed_params"]) / c["nonembed_params"],
                     "total_params": m.total_params(),
                     "passes": bool(abs(got - c["nonembed_params"]) / c["nonembed_params"] <= 0.01)}
                    )
        del m
    return {"configs": rows, "passes": all(r["passes"] for r in rows)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["g0a", "g0b", "g0e", "params", "all"])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    os.makedirs(ART, exist_ok=True)
    res = {}
    if a.check in ("params", "all"):
        res["param_count"] = param_test()
    if a.check in ("g0a", "all"):
        res["g0a_mimi_roundtrip"] = g0a(a.device)
    if a.check in ("g0b", "all"):
        res["g0b_overfit"] = g0b(a.device)
    if a.check in ("g0e", "all"):
        res["g0e_sampler_integrity_untrained"] = g0e(a.device)
    path = os.path.join(ART, f"g0_{a.check}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    for k, v in res.items():
        print(f"{k}: passes={v.get('passes')}", flush=True)
    print(json.dumps(res, indent=1)[:2500], flush=True)
    print(f"→ {path}", flush=True)
