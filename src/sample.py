"""ASPECT-D frozen sampler — MaskGIT confidence decoding over Mimi levels.

``grid.json -> backbone.sampler_frozen`` verbatim: level-by-level 1..8, exactly T
refinement steps per level (NFE = 8T), cosine unmasking schedule, Gumbel noise on
log-probs annealed linearly 1.0 → 0.0 across the T steps within a level,
categorical token sampling at temperature 1.0, no CFG. Frozen for every config,
seed and T — nothing here may be tuned per shape or per T (task.md directive 4).

Generation randomness is drawn from generators keyed by (batch index, level, step)
over a FIXED item order and FIXED batch composition, so every item sees the same
noise at the first step of every level for every T, config and seed (LOG.md P0-3).

Usage
    python src/sample.py synth --run runs/A3_0 --T 16 [--items N]
    python src/sample.py clayer --out artifacts/c_layer.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from data import PROC_DIR, SR, FRAME_SAMPLES, TokenStore
from model import CODEBOOK_SIZE, MASK_ID, N_LEVELS, PAD_ID, PHONEME_POS_CAP, AspectD, load_grid

GEN_SEED_BASE = 20260805        # fixed generation-RNG base, identical for all runs/T
EVAL_BATCH = 50                 # FIXED batch composition (LOG.md P0-3)
MAX_TARGET_FRAMES = 187         # 15 s * 12.5 Hz (grid.json max_target_seconds)


# ------------------------------------------------------------------- item build
def load_items(store: TokenStore, limit: Optional[int] = None) -> List[Dict]:
    with open(os.path.join(PROC_DIR, "eval_zs.json")) as fh:
        items = json.load(fh)
    items = sorted(items, key=lambda d: d["item"])
    if limit:
        items = items[:limit]
    spc = store.dataset["sec_per_char"]
    out = []
    for it in items:
        ph_p, tk_p = store.get(it["prompt_id"])
        ph_t, _ = store.get(it["target_id"])
        sp = store.vocab.get("<sp>", 1)
        phon = np.concatenate([ph_p, [sp], ph_t])[:PHONEME_POS_CAP]
        n_chars = len(it["target_text"].strip())
        n_tgt = int(min(MAX_TARGET_FRAMES, max(1, round(n_chars * spc * 12.5))))
        out.append({**it, "phonemes": phon, "prompt_tokens": tk_p,
                    "n_prompt": len(tk_p), "n_target": n_tgt, "n_chars": n_chars})
    return out


def batches(items: List[Dict], bs: int = EVAL_BATCH) -> List[List[Dict]]:
    return [items[i:i + bs] for i in range(0, len(items), bs)]


# ----------------------------------------------------------------- the sampler
def _gumbel(shape, gen: torch.Generator, device) -> torch.Tensor:
    u = torch.rand(shape, generator=gen, device=device, dtype=torch.float32)
    return -torch.log(-torch.log(u.clamp_min(1e-20)).clamp_min(1e-20))


@torch.no_grad()
def synth_batch(model: AspectD, batch: List[Dict], T: int, device, batch_idx: int
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (token grid [B, Fmax, 8], frame_mask [B, Fmax])."""
    B = len(batch)
    n_p = [b["n_prompt"] for b in batch]
    n_t = [b["n_target"] for b in batch]
    Fmax = max(p + t for p, t in zip(n_p, n_t))
    Pmax = max(len(b["phonemes"]) for b in batch)

    ph = torch.zeros((B, Pmax), dtype=torch.long)
    phm = torch.zeros((B, Pmax), dtype=torch.bool)
    grid = torch.full((B, Fmax, N_LEVELS), PAD_ID, dtype=torch.long)
    fmask = torch.zeros((B, Fmax), dtype=torch.bool)
    tgt_pos = torch.zeros((B, Fmax), dtype=torch.bool)
    for i, b in enumerate(batch):
        p = torch.from_numpy(np.asarray(b["phonemes"], dtype=np.int64))
        ph[i, :len(p)] = p
        phm[i, :len(p)] = True
        grid[i, :n_p[i]] = torch.from_numpy(b["prompt_tokens"])
        grid[i, n_p[i]:n_p[i] + n_t[i]] = MASK_ID
        fmask[i, :n_p[i] + n_t[i]] = True
        tgt_pos[i, n_p[i]:n_p[i] + n_t[i]] = True
    ph, phm, grid, fmask, tgt_pos = (x.to(device) for x in (ph, phm, grid, fmask, tgt_pos))
    n_t_dev = torch.tensor(n_t, device=device)
    ar = torch.arange(Fmax, device=device)[None, :].expand(B, Fmax)

    for level in range(N_LEVELS):
        committed = torch.zeros((B, Fmax), dtype=torch.bool, device=device)
        tokens_l = torch.full((B, Fmax), MASK_ID, dtype=torch.long, device=device)
        for step in range(T):
            gen = torch.Generator(device=device).manual_seed(
                GEN_SEED_BASE + batch_idx * 1_000_000 + level * 1_000 + step)
            cur = grid.clone()
            cur[..., level] = torch.where(tgt_pos, tokens_l, grid[..., level])
            for above in range(level + 1, N_LEVELS):
                cur[..., above] = torch.where(tgt_pos, torch.full_like(tokens_l, MASK_ID),
                                              grid[..., above])
            cur[~fmask] = PAD_ID
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model(ph, phm, cur, fmask)
            logp = torch.log_softmax(model.logits(hidden, level).float(), dim=-1)  # [B,F,V]
            g_tok = _gumbel(logp.shape, gen, device)
            sampled = (logp + g_tok).argmax(-1)                    # temperature 1.0
            conf = logp.gather(-1, sampled[..., None]).squeeze(-1)
            scale = 1.0 - step / (T - 1) if T > 1 else 1.0         # LOG.md P0-4
            conf = conf + scale * _gumbel(conf.shape, gen, device)
            conf = torch.where(committed, torch.full_like(conf, float("inf")), conf)
            conf = torch.where(tgt_pos, conf, torch.full_like(conf, float("-inf")))
            # cosine unmasking schedule: masked cells left after this step
            frac = math.cos(math.pi * (step + 1) / (2 * T))
            n_keep_masked = torch.floor(n_t_dev * frac)
            k = (n_t_dev - n_keep_masked).clamp(min=0).long()      # committed after step
            order = conf.argsort(dim=1, descending=True)
            rank = torch.empty_like(order)
            rank.scatter_(1, order, ar)
            new_c = (rank < k[:, None]) & tgt_pos
            tokens_l = torch.where(new_c, sampled, tokens_l)
            committed = new_c
        grid[..., level] = torch.where(tgt_pos, tokens_l, grid[..., level])
        grid[~fmask] = PAD_ID
    return grid.cpu(), fmask.cpu()


# ------------------------------------------------------------------- synthesise
def load_run(run_dir: str, n_phonemes: int, device):
    st = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
    cfg = st["cfg"]
    model = AspectD(cfg["width"], cfg["depth"], cfg["heads"], n_phonemes)
    model.load_state_dict(st["model"])
    return model.to(device).eval(), cfg, st["step"]


def cmd_synth(a):
    device = torch.device(a.device)
    store = TokenStore()
    model, cfg, step = load_run(a.run, len(store.vocab), device)
    items = load_items(store, a.items)
    out_dir = os.path.join(a.run, f"synth_T{a.T}")
    os.makedirs(out_dir, exist_ok=True)
    from transformers import MimiModel
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()

    t0 = time.time()
    all_tokens: Dict[str, np.ndarray] = {}
    meta = []
    for bi, batch in enumerate(batches(items)):
        grid, fmask = synth_batch(model, batch, a.T, device, bi)
        for i, b in enumerate(batch):
            n = b["n_prompt"] + b["n_target"]
            g = grid[i, :n]
            all_tokens[b["item"]] = g.numpy().astype(np.int16)
            codes = g.T[None].to(device)                          # [1,8,frames]
            with torch.no_grad():
                wav = mimi.decode(codes).audio_values[0, 0].float().cpu().numpy()
            gen = wav[b["n_prompt"] * FRAME_SAMPLES:]
            sf.write(os.path.join(out_dir, f"{b['item']}.flac"), gen, SR, format="FLAC")
            meta.append({"item": b["item"], "n_prompt": b["n_prompt"], "n_target": b["n_target"],
                         "gen_seconds": len(gen) / SR})
        if bi % 2 == 0:
            print(f"[synth {os.path.basename(a.run)} T={a.T}] batch {bi}/"
                  f"{len(batches(items))} {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(os.path.join(out_dir, "tokens.npz"), **all_tokens)
    with open(os.path.join(out_dir, "synth.json"), "w") as fh:
        json.dump({"run": a.run, "config": cfg["id"], "T": a.T, "nfe": 8 * a.T,
                   "items": len(items), "ckpt_step": step,
                   "wall_seconds": time.time() - t0, "gpu_hours": (time.time() - t0) / 3600,
                   "meta": meta}, fh)
    print(f"[synth] {a.run} T={a.T} done in {(time.time()-t0)/60:.1f} min", flush=True)


# ------------------------------------------------- protocol §6.3 integrity check
def integrity_check(run_dir: str, t_lo: int, t_hi: int, n_items: int = 20) -> Dict:
    lo = np.load(os.path.join(run_dir, f"synth_T{t_lo}", "tokens.npz"))
    hi = np.load(os.path.join(run_dir, f"synth_T{t_hi}", "tokens.npz"))
    keys = sorted(set(lo.files) & set(hi.files))[:n_items]
    diffs, cells = 0, 0
    for k in keys:
        a, b = lo[k], hi[k]
        n = min(len(a), len(b))
        diffs += int((a[:n] != b[:n]).sum())
        cells += int(a[:n].size)
    frac = diffs / max(1, cells)
    return {"items": len(keys), "T_lo": t_lo, "T_hi": t_hi, "differing_cell_fraction": frac,
            "passes": bool(frac > 0.20)}


# --------------------------------------------------------- §6.5 c_layer(width)
def cmd_clayer(a):
    device = torch.device(a.device)
    grid = load_grid()
    store = TokenStore()
    widths = sorted({c["width"] for c in grid["configs"] if not
                     grid["budgets"][c["budget"]].get("contingency", False)}
                    | set(a.extra_widths or []))
    res = {}
    for w in widths:
        depth = 4
        model = AspectD(w, depth, w // 64, len(store.vocab)).to(device).eval()
        ph = torch.zeros((1, 200), dtype=torch.long, device=device)
        phm = torch.ones((1, 200), dtype=torch.bool, device=device)
        au = torch.full((1, 224, N_LEVELS), MASK_ID, dtype=torch.long, device=device)
        fm = torch.ones((1, 224), dtype=torch.bool, device=device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(5):
                model(ph, phm, au, fm)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(50):
                model(ph, phm, au, fm)
            torch.cuda.synchronize()
        ms_per_forward = (time.time() - t0) / 50 * 1000
        res[str(w)] = {"ms_per_forward_batch1_depth4": ms_per_forward,
                       "c_layer_ms": ms_per_forward / depth}
        print(f"[clayer] w={w} {ms_per_forward:.2f} ms / {depth} layers → "
              f"{ms_per_forward/depth:.3f} ms/layer", flush=True)
        del model
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"gpu": torch.cuda.get_device_name(device), "measured": res,
                   "note": "batch 1, 200 phoneme + 224 frame positions, bf16, eager"}, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--run", required=True)
    s.add_argument("--T", type=int, required=True)
    s.add_argument("--items", type=int, default=None)
    s.add_argument("--device", default="cuda:0")
    s.set_defaults(fn=cmd_synth)
    c = sub.add_parser("clayer")
    c.add_argument("--out", default="artifacts/c_layer.json")
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--extra-widths", type=int, nargs="*")
    c.set_defaults(fn=cmd_clayer)
    args = ap.parse_args()
    args.fn(args)
