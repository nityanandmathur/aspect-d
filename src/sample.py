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


def load_items_s1(store: TokenStore, arm: str, limit: Optional[int] = None) -> List[Dict]:
    """S1 (task-v2.md §4-S1): the same eval items with a prompt-context arm swapped in.

    Only `prompt_tokens`, `phonemes` and `n_prompt` change; the target length, the
    text and the sampler are exactly as in v1.0. Items lacking any of the four arms
    are dropped, so every arm runs on the identical item set.

    **RNG note, recorded rather than glossed.** §4-S1 asks for "matched RNG across
    arms". That is not literally attainable with the frozen sampler: the Gumbel
    draw is shaped [B, Fmax, V] with Fmax = max(n_prompt + n_target), so a longer
    prompt changes the noise tensor's shape and every arm consumes a different
    stream. Changing the indexing to make noise target-relative would match the
    arms but would break bit-reproduction of the v1.0 grids, which is the stronger
    guarantee. We keep the sampler frozen: arms therefore use independent noise
    rather than common random numbers -- unbiased for the paired contrast, at the
    cost of variance that the 15-run x ~397-item paired design absorbs.
    """
    with open(os.path.join(PROC_DIR, "s1_context", "arms.json")) as fh:
        A = json.load(fh)
    keep = {k for k, v in A["built"].items() if all(v[str(L)] is not None for L in A["arms_s"])}
    base = load_items(store, None)
    out = []
    for b in base:
        if b["item"] not in keep:
            continue
        a = A["built"][b["item"]][arm]
        tk = np.asarray(a["tokens"], dtype=np.int16).reshape(-1, N_LEVELS)
        ph_t, _ = store.get(b["target_id"])
        sp = store.vocab.get("<sp>", 1)
        phon = np.concatenate([np.asarray(a["phonemes"], dtype=np.int64), [sp], ph_t])
        out.append({**b, "prompt_tokens": tk, "n_prompt": len(tk),
                    "phonemes": phon[:PHONEME_POS_CAP], "s1_arm": arm,
                    "s1_seconds": a["seconds"], "s1_n_clips": a["n_clips"]})
    if limit:
        out = out[:limit]
    return out


def load_items_rate(store: TokenStore, limit: Optional[int] = None) -> List[Dict]:
    """S3 (task-v2.md §4-S3): per-item speaking rate instead of the corpus median.

    v1.0 sets the target length from a single corpus-median seconds-per-character,
    so a speaker who talks faster or slower than the corpus is synthesised at the
    wrong duration -- the limitation the paper now states. Arm B measures the rate
    from the item's own PROMPT source clip (its duration over its character count)
    and uses that instead. Nothing else changes: same text, same prompt tokens,
    same sampler, same RNG keying.
    """
    import pandas as pd
    meta = pd.read_parquet(os.path.join(PROC_DIR, "meta.parquet"))
    dur = dict(zip(meta.id, meta.duration))
    nch = dict(zip(meta.id, meta.n_chars))
    base = load_items(store, limit)
    out = []
    for b in base:
        d, c = dur.get(b["prompt_id"]), nch.get(b["prompt_id"])
        if not d or not c:
            out.append({**b, "spc_item": None})
            continue
        spc = float(d) / max(1, int(c))
        n_t = int(min(MAX_TARGET_FRAMES, max(1, round(b["n_chars"] * spc * 12.5))))
        out.append({**b, "n_target": n_t, "spc_item": spc})
    return out


def batches(items: List[Dict], bs: int = EVAL_BATCH) -> List[List[Dict]]:
    return [items[i:i + bs] for i in range(0, len(items), bs)]


# ----------------------------------------------------------------- the sampler
def _gumbel(shape, gen: torch.Generator, device) -> torch.Tensor:
    """Gumbel(0,1) samples. The clamp keeps u strictly inside (0,1) so both logs are
    finite; note the parenthesisation — `-log(-log(u).clamp_min(e))` would clamp the
    *negative* inner log to +e and produce NaN for every element."""
    u = torch.rand(shape, generator=gen, device=device, dtype=torch.float32)
    return -torch.log(-torch.log(u.clamp(1e-20, 1.0 - 1e-7)))


@torch.no_grad()
def synth_batch(model: AspectD, batch: List[Dict], T: int, device, batch_idx: int,
                schedule: Optional[List[int]] = None, cand: int = 0,
                gamma: float = 0.0, wrong_tokens: Optional[List] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (token grid [B, Fmax, 8], frame_mask [B, Fmax]).

    `cand` (S2, task-v2.md §4-S2): best-of-K candidate index. It shifts the RNG
    stream by a disjoint 1e8 block, so candidates are independent draws from the
    same model and `cand=0` reproduces every v1.0/v1.1 grid bit-for-bit.

    `schedule` (E3, task-v1.md §4-E3): per-level step counts, e.g.
    [25,1,1,1,1,1,1,1]. When given it replaces the uniform T for every level;
    the RNG stream is keyed by (batch, level, step) and so is unaffected, which
    keeps schedules comparable to each other and to the v1.0 uniform runs."""
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

    grid_w = None
    if gamma:
        grid_w = grid.clone()
        for i, b in enumerate(batch):
            wt = torch.from_numpy(np.asarray(wrong_tokens[i]))
            n = min(n_p[i], len(wt))
            grid_w[i, :n_p[i]] = PAD_ID
            grid_w[i, :n] = wt[:n]

    for level in range(N_LEVELS):
        T_l = int(schedule[level]) if schedule is not None else T
        committed = torch.zeros((B, Fmax), dtype=torch.bool, device=device)
        tokens_l = torch.full((B, Fmax), MASK_ID, dtype=torch.long, device=device)
        for step in range(T_l):
            gen = torch.Generator(device=device).manual_seed(
                GEN_SEED_BASE + cand * 100_000_000 + batch_idx * 1_000_000
                + level * 1_000 + step)
            cur = grid.clone()
            cur[..., level] = torch.where(tgt_pos, tokens_l, grid[..., level])
            for above in range(level + 1, N_LEVELS):
                cur[..., above] = torch.where(tgt_pos, torch.full_like(tokens_l, MASK_ID),
                                              grid[..., above])
            cur[~fmask] = PAD_ID
            if gamma:
                cur_w = cur.clone()
                for i in range(B):
                    cur_w[i, :n_p[i]] = grid_w[i, :n_p[i]]
                cur_w[~fmask] = PAD_ID
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model(ph, phm, cur, fmask)
            logits_c = model.logits(hidden, level).float()
            if gamma:
                # S4 (task-v2.md §4-S4): the same text conditioned on a DELIBERATELY
                # WRONG speaker's prompt. Extrapolating away from it is the
                # training-free speaker-contrastive knob.
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden_w = model(ph, phm, cur_w, fmask)
                logits_w = model.logits(hidden_w, level).float()
                logits_c = logits_c + gamma * (logits_c - logits_w)
            logp = torch.log_softmax(logits_c, dim=-1)              # [B,F,V]
            g_tok = _gumbel(logp.shape, gen, device)
            sampled = (logp + g_tok).argmax(-1)                    # temperature 1.0
            conf = logp.gather(-1, sampled[..., None]).squeeze(-1)
            scale = 1.0 - step / (T_l - 1) if T_l > 1 else 1.0     # LOG.md P0-4
            conf = conf + scale * _gumbel(conf.shape, gen, device)
            conf = torch.where(committed, torch.full_like(conf, float("inf")), conf)
            conf = torch.where(tgt_pos, conf, torch.full_like(conf, float("-inf")))
            # cosine unmasking schedule: masked cells left after this step
            frac = math.cos(math.pi * (step + 1) / (2 * T_l))
            n_keep_masked = torch.floor(n_t_dev * frac)
            k = (n_t_dev - n_keep_masked).clamp(min=0).long()      # committed after step
            order = conf.argsort(dim=1, descending=True)
            rank = torch.empty_like(order)
            rank.scatter_(1, order, ar)
            new_c = (rank < k[:, None]) & tgt_pos
            # only NEWLY committed cells take the fresh draw; already-committed cells are
            # frozen (grid.json: "categorical ... for newly committed tokens"). Without the
            # ~committed guard every step would overwrite the whole committed prefix, and the
            # last step (k = n_target) would resample the entire level.
            tokens_l = torch.where(new_c & ~committed, sampled, tokens_l)
            committed = committed | new_c
        grid[..., level] = torch.where(tgt_pos, tokens_l, grid[..., level])
        grid[~fmask] = PAD_ID
    return grid.cpu(), fmask.cpu()


@torch.no_grad()
def synth_batch_flat(model: AspectD, batch: List[Dict], T: int, device, batch_idx: int,
                     schedule: Optional[List[int]] = None, cand: int = 0
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pivot P1-D sampler: whole-grid confidence decoding. Every target cell of all 8
    levels starts MASK; T is the TOTAL number of steps (NFE = T, not 8T); the cosine
    schedule and the annealed Gumbel confidence noise are otherwise identical to the
    coarse-to-fine sampler, and committed cells stay committed."""
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
    cellmask = tgt_pos[..., None].expand(B, Fmax, N_LEVELS)          # [B,F,8]
    n_cells = torch.tensor([n * N_LEVELS for n in n_t], device=device)
    committed = torch.zeros_like(cellmask)
    tokens = torch.where(cellmask, torch.full_like(grid, MASK_ID), grid)
    ar = torch.arange(Fmax * N_LEVELS, device=device)[None, :].expand(B, Fmax * N_LEVELS)

    for step in range(T):
        gen = torch.Generator(device=device).manual_seed(
            GEN_SEED_BASE + cand * 100_000_000 + batch_idx * 1_000_000 + step)
        cur = torch.where(cellmask, tokens, grid)
        cur[~fmask] = PAD_ID
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(ph, phm, cur, fmask)
        sampled = torch.empty_like(tokens)
        conf = torch.full(tokens.shape, float("-inf"), device=device)
        for lvl in range(N_LEVELS):
            logp = torch.log_softmax(model.logits(hidden, lvl).float(), dim=-1)
            g_tok = _gumbel(logp.shape, gen, device)
            s_l = (logp + g_tok).argmax(-1)
            sampled[..., lvl] = s_l
            conf[..., lvl] = logp.gather(-1, s_l[..., None]).squeeze(-1)
        scale = 1.0 - step / (T - 1) if T > 1 else 1.0
        conf = conf + scale * _gumbel(conf.shape, gen, device)
        conf = torch.where(committed, torch.full_like(conf, float("inf")), conf)
        conf = torch.where(cellmask, conf, torch.full_like(conf, float("-inf")))
        frac = math.cos(math.pi * (step + 1) / (2 * T))
        k = (n_cells - torch.floor(n_cells * frac)).clamp(min=0).long()
        flat_conf = conf.reshape(B, -1)
        order = flat_conf.argsort(dim=1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, ar)
        new_c = (rank < k[:, None]).reshape(conf.shape) & cellmask
        tokens = torch.where(new_c & ~committed, sampled, tokens)
        committed = committed | new_c
    grid = torch.where(cellmask, tokens, grid)
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
    items = (load_items_s1(store, a.s1_arm, a.items) if getattr(a, "s1_arm", None)
             else load_items_rate(store, a.items) if getattr(a, "rate_matched", False)
             else load_items(store, a.items))
    out_dir = os.path.join(a.run, f"synth_{a.tag}" if getattr(a, "tag", None)
                           else f"synth_T{a.T}")
    os.makedirs(out_dir, exist_ok=True)
    from transformers import MimiModel
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()

    t0 = time.time()
    all_tokens: Dict[str, np.ndarray] = {}
    meta = []
    sched = [int(x) for x in a.schedule.split(",")] if getattr(a, "schedule", None) else None
    if sched is not None:
        assert len(sched) == N_LEVELS, f"schedule needs {N_LEVELS} entries, got {len(sched)}"
    synth_fn = synth_batch_flat if getattr(a, "recipe", "coarse") == "flat" else synth_batch
    for bi, batch in enumerate(batches(items)):
        cand = int(getattr(a, "cand", 0))
        gamma = float(getattr(a, "gamma", 0.0))
        wrong = None
        if gamma:
            # §4-S4: deterministic wrong-speaker assignment, item i <- prompt of (i+7) mod N
            N = len(items)
            base = bi * EVAL_BATCH
            wrong = [items[(base + k + 7) % N]["prompt_tokens"] for k in range(len(batch))]
        grid, fmask = (synth_fn(model, batch, a.T, device, bi, sched, cand, gamma, wrong)
                       if sched is not None else
                       synth_fn(model, batch, a.T, device, bi, None, cand, gamma, wrong))
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
    # written LAST and atomically: `synth.json` is the completion marker that
    # `evaluate.score_dir` keys off, so a scorer racing a running synthesis must
    # never observe it beside a partially-written directory
    tmp = os.path.join(out_dir, "synth.json.tmp")
    with open(tmp, "w") as fh:
        json.dump({"run": a.run, "config": cfg["id"], "T": a.T,
                   "schedule": sched, "total_nfe": (sum(sched) if sched else 8 * a.T),
                   "item_ids": [b["item"] for b in items], "cand": int(getattr(a, "cand", 0)),
                   "gamma": float(getattr(a, "gamma", 0.0)),
                   "nfe": a.T if getattr(a, "recipe", "coarse") == "flat" else 8 * a.T,
                   "recipe": getattr(a, "recipe", "coarse"),
                   "items": len(items), "ckpt_step": step,
                   "wall_seconds": time.time() - t0, "gpu_hours": (time.time() - t0) / 3600,
                   "meta": meta}, fh)
    os.replace(tmp, os.path.join(out_dir, "synth.json"))      # atomic publish
    print(f"[synth] {a.run} T={a.T} done in {(time.time()-t0)/60:.1f} min", flush=True)


# ------------------------------------------------- protocol §6.3 integrity check
def integrity_check(run_dir: str, t_lo: int, t_hi: int, n_items: int = 20) -> Dict:
    """protocol §6.3: for 20 items, T_lo vs T_hi must differ on >20 % of token cells.
    Only GENERATED cells count — the prompt frames are copied verbatim at every T and
    would dilute the statistic toward failure."""
    lo = np.load(os.path.join(run_dir, f"synth_T{t_lo}", "tokens.npz"))
    hi = np.load(os.path.join(run_dir, f"synth_T{t_hi}", "tokens.npz"))
    meta = {m["item"]: m for m in
            json.load(open(os.path.join(run_dir, f"synth_T{t_hi}", "synth.json")))["meta"]}
    keys = sorted(set(lo.files) & set(hi.files))[:n_items]
    diffs, cells = 0, 0
    for k in keys:
        a, b = lo[k], hi[k]
        s = int(meta[k]["n_prompt"]) if k in meta else 0
        n = min(len(a), len(b))
        diffs += int((a[s:n] != b[s:n]).sum())
        cells += int(a[s:n].size)
    frac = diffs / max(1, cells)
    return {"items": len(keys), "T_lo": t_lo, "T_hi": t_hi, "generated_cells": cells,
            "differing_cell_fraction": frac, "threshold": 0.20, "passes": bool(frac > 0.20)}


def cmd_integrity(a):
    """Run the §6.3 check for every run that has both T endpoints synthesised."""
    import glob as _glob
    out = {}
    for run in sorted(_glob.glob(os.path.join(a.runs_glob))):
        lo = os.path.join(run, f"synth_T{a.t_lo}", "tokens.npz")
        hi = os.path.join(run, f"synth_T{a.t_hi}", "tokens.npz")
        if not (os.path.exists(lo) and os.path.exists(hi)):
            continue
        out[os.path.basename(run)] = integrity_check(run, a.t_lo, a.t_hi)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    n_fail = sum(1 for v in out.values() if not v["passes"])
    with open(a.out, "w") as fh:
        json.dump({"per_run": out, "n_runs": len(out), "n_failing": n_fail,
                   "all_pass": n_fail == 0}, fh, indent=1)
    for k, v in out.items():
        print(f"[integrity] {k}: {v['differing_cell_fraction']*100:.1f}% differing cells "
              f"→ {'PASS' if v['passes'] else 'FAIL'}", flush=True)
    if n_fail:
        raise SystemExit(f"§6.3 integrity FAILED for {n_fail} runs — fix before scoring")


# --------------------------------------------------------- §6.5 c_layer(width)
def cmd_clayer(a):
    device = torch.device(a.device)
    grid = load_grid()
    store = TokenStore()
    widths = sorted({c["width"] for c in grid["configs"] if not
                     grid["budgets"][c["budget"]].get("contingency", False)}
                    | set(a.extra_widths or []))
    res = {}
    ph = torch.zeros((1, 200), dtype=torch.long, device=device)
    phm = torch.ones((1, 200), dtype=torch.bool, device=device)
    au = torch.full((1, 224, N_LEVELS), MASK_ID, dtype=torch.long, device=device)
    fm = torch.ones((1, 224), dtype=torch.bool, device=device)

    def time_depth(w: int, depth: int) -> float:
        model = AspectD(w, depth, w // 64, len(store.vocab)).to(device).eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(5):
                model(ph, phm, au, fm)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(50):
                model(ph, phm, au, fm)
            torch.cuda.synchronize()
        ms = (time.time() - t0) / 50 * 1000
        del model
        torch.cuda.empty_cache()
        return ms

    # c_layer is the SLOPE in depth, so the depth-independent per-forward overhead
    # (embeddings, output head, launch latency) is not charged to every layer.
    d_lo, d_hi = 4, 12
    for w in widths:
        t_lo, t_hi = time_depth(w, d_lo), time_depth(w, d_hi)
        c = (t_hi - t_lo) / (d_hi - d_lo)
        res[str(w)] = {"ms_forward_depth4": t_lo, "ms_forward_depth12": t_hi,
                       "c_layer_ms": c, "fixed_overhead_ms": t_lo - d_lo * c}
        print(f"[clayer] w={w}: d4 {t_lo:.2f} ms, d12 {t_hi:.2f} ms → "
              f"c_layer {c:.3f} ms/layer (overhead {t_lo - d_lo*c:.2f} ms)", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"gpu": torch.cuda.get_device_name(device), "measured": res,
                   "note": "batch 1, 200 phoneme + 224 frame positions, bf16, eager; "
                           "c_layer = (t(d=12) - t(d=4)) / 8, i.e. the per-layer slope"},
                  fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--run", required=True)
    s.add_argument("--T", type=int, required=True)
    s.add_argument("--items", type=int, default=None)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--recipe", choices=["coarse", "flat"], default="coarse")
    s.add_argument("--schedule", default=None,
                   help="E3: 8 comma-separated per-level step counts, e.g. 25,1,1,1,1,1,1,1")
    s.add_argument("--tag", default=None, help="output dir suffix (default T<val>)")
    s.add_argument("--gamma", type=float, default=0.0,
                   help="S4 speaker-contrastive guidance strength (task-v2.md §4-S4)")
    s.add_argument("--rate-matched", action="store_true",
                   help="S3: per-item seconds-per-character from the prompt clip (§4-S3)")
    s.add_argument("--cand", type=int, default=0,
                   help="S2 best-of-K candidate index; shifts the RNG stream (task-v2.md §4-S2)")
    s.add_argument("--s1-arm", default=None,
                   help="S1 prompt-context arm in seconds, e.g. 9.0 (task-v2.md §4-S1)")
    s.set_defaults(fn=cmd_synth)
    c = sub.add_parser("clayer")
    c.add_argument("--out", default="artifacts/c_layer.json")
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--extra-widths", type=int, nargs="*")
    c.set_defaults(fn=cmd_clayer)
    ic = sub.add_parser("integrity")
    ic.add_argument("--runs-glob", default="runs/*")
    ic.add_argument("--t-lo", type=int, default=1)
    ic.add_argument("--t-hi", type=int, default=16)
    ic.add_argument("--out", default="artifacts/integrity.json")
    ic.set_defaults(fn=cmd_integrity)
    args = ap.parse_args()
    args.fn(args)
