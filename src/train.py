"""ASPECT-D training — SoundStorm-style coarse-to-fine masked diffusion.

Recipe frozen by ``configs/grid.json -> training`` and ``backbone.training_objective``:
30,000 steps, effective batch 256 sequences, AdamW(0.9, 0.95) wd 0.1, clip 1.0,
cosine→10 % of peak, 600 warmup, bf16, val every 1,000 steps at fixed mask ratios
{0.25, 0.5, 0.75} averaged over levels with a fixed val-masking RNG.

Per example: level l ~ U{1..8}; ratio t ~ U(0,1]; levels < l visible, level l
masked i.i.d. with prob t, levels > l all MASK; phonemes and the 3 s prompt are
always visible; cross-entropy on masked cells of level l only.

Nothing differs between runs except (w, d, heads, seed) — see LOG.md P0-6 for the
batch/masking RNG construction that makes this exact under grad accumulation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data import PROMPT_FRAMES, TokenStore
from model import (AUDIO_VOCAB, MASK_ID, N_LEVELS, PAD_ID, PHONEME_POS_CAP, build_model,
                   load_grid)

VAL_RNG = 999               # fixed val-masking RNG, identical for every run
VAL_RATIOS = (0.25, 0.5, 0.75)
MB_PROXY_CAP = 8e8          # micro-batch sizing proxy (depth*width*seq*mb); calibrated so a
                            # run peaks near 40 GB, i.e. several runs fit one 183 GB B200.
                            # Purely a memory/speed knob: the loss is normalised over the
                            # whole 256-sequence effective batch, so the gradient is identical
                            # for any micro-batch split (LOG.md P0-6).
SEQ_PROXY = 600


def micro_batch_for(depth: int, width: int, batch: int = 256) -> int:
    mb = batch
    while mb > 1 and depth * width * SEQ_PROXY * mb > MB_PROXY_CAP:
        mb //= 2
    return mb


# --------------------------------------------------------------------- batching
class BatchPlan:
    """Deterministic, config-independent batch order (grid.json identical_across_shapes)."""

    def __init__(self, pos: np.ndarray, n_frames: np.ndarray, seed: int, batch: int = 256,
                 superbatch: int = 8):
        self.pos = pos                      # absolute row positions in the index
        self.n_frames = n_frames            # aligned with pos
        self.n = len(pos)
        self.seed = seed
        self.batch = batch
        self.group = batch * superbatch
        self.bpe = self.n // batch
        self._epoch = -1
        self._batches: List[np.ndarray] = []

    def _build(self, epoch: int):
        rng = np.random.default_rng([self.seed, epoch])
        perm = rng.permutation(self.n)
        out = []
        for s in range(0, self.n - self.group + 1, self.group):
            blk = perm[s:s + self.group]
            blk = blk[np.argsort(self.n_frames[blk], kind="stable")]
            for b in range(0, len(blk), self.batch):
                out.append(self.pos[blk[b:b + self.batch]])
        tail = perm[(self.n // self.group) * self.group:]
        for b in range(0, len(tail) - self.batch + 1, self.batch):
            out.append(self.pos[tail[b:b + self.batch]])
        self._batches, self._epoch = out, epoch

    def batch_for_step(self, step: int) -> np.ndarray:
        if not self._batches:
            self._build(0)
        epoch, i = divmod(step, len(self._batches))
        if epoch != self._epoch:
            self._build(epoch)
        return self._batches[i % len(self._batches)]


def assemble(store: TokenStore, rows: np.ndarray, max_phon: int = PHONEME_POS_CAP
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """→ phonemes [B,P], phone_mask [B,P], clean tokens [B,F,8], frame_mask [B,F]."""
    phs, tks = [], []
    for r in rows:
        ph, tk = store.get_pos(int(r))
        phs.append(ph[:max_phon])
        tks.append(tk)
    P = max(1, max(len(p) for p in phs))
    Fr = max(len(t) for t in tks)
    ph_arr = np.zeros((len(rows), P), dtype=np.int64)
    ph_msk = np.zeros((len(rows), P), dtype=bool)
    tok = np.full((len(rows), Fr, N_LEVELS), PAD_ID, dtype=np.int64)
    fr_msk = np.zeros((len(rows), Fr), dtype=bool)
    for i, (p, t) in enumerate(zip(phs, tks)):
        ph_arr[i, :len(p)] = p
        ph_msk[i, :len(p)] = True
        tok[i, :len(t)] = t
        fr_msk[i, :len(t)] = True
    return ph_arr, ph_msk, tok, fr_msk


def draw_masks_flat(gen: torch.Generator, B: int, Fr: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pivot P1-D: flat joint masking — one ratio per example, i.i.d. Bernoulli over ALL
    8*Fr cells (protocol §8 P1-D: "t ~ U(0,1] i.i.d. over all cells, loss on all masked
    cells"). Same stream discipline as the coarse-to-fine draw: one draw per effective
    batch, keyed by (seed, step), so it is identical across configs."""
    ratios = 1.0 - torch.rand((B,), generator=gen)          # U(0,1]
    cells = torch.rand((B, Fr, N_LEVELS), generator=gen)
    return ratios, cells


def build_inputs_flat(tok: torch.Tensor, frame_mask: torch.Tensor, ratios: torch.Tensor,
                      cells: torch.Tensor, prompt_frames: int
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """→ (model input tokens, per-cell loss mask [B,F,8]) for the flat recipe."""
    B, Fr, _ = tok.shape
    dev = tok.device
    target = frame_mask & (torch.arange(Fr, device=dev)[None, :] >= prompt_frames)
    masked = (cells.to(dev) < ratios.to(dev)[:, None, None]) & target[..., None]
    inp = tok.clone()
    inp[masked] = MASK_ID
    inp[~frame_mask] = PAD_ID
    return inp, masked


def masked_ce_flat(model, ph, ph_msk, inp, fr_msk, tok, loss_cells) -> torch.Tensor:
    """Summed CE over every masked cell of every level (flat recipe)."""
    hidden = model(ph, ph_msk, inp, fr_msk)
    total = hidden.new_zeros(())
    for lvl in range(N_LEVELS):
        sel = loss_cells[..., lvl]
        if not bool(sel.any()):
            continue
        total = total + F.cross_entropy(model.logits(hidden[sel], lvl).float(),
                                        tok[..., lvl][sel], reduction="sum")
    return total


def draw_masks(gen: torch.Generator, B: int, Fr: int) -> Tuple[torch.Tensor, torch.Tensor,
                                                               torch.Tensor]:
    """Level per example, ratio per example, per-cell uniforms — one draw per effective
    batch so the stream is identical for every config (LOG.md P0-6)."""
    levels = torch.randint(0, N_LEVELS, (B,), generator=gen)
    ratios = 1.0 - torch.rand((B,), generator=gen)          # U(0,1]
    cells = torch.rand((B, Fr), generator=gen)
    return levels, ratios, cells


def build_inputs(tok: torch.Tensor, frame_mask: torch.Tensor, levels: torch.Tensor,
                 ratios: torch.Tensor, cells: torch.Tensor, prompt_frames: int
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the coarse-to-fine masking. Returns (model input tokens, loss-cell mask)."""
    B, Fr, _ = tok.shape
    dev = tok.device
    lvl_idx = torch.arange(N_LEVELS, device=dev)[None, None, :]          # [1,1,8]
    lv = levels.to(dev)[:, None, None]
    target = frame_mask & (torch.arange(Fr, device=dev)[None, :] >= prompt_frames)  # [B,F]
    masked_cell = (cells.to(dev) < ratios.to(dev)[:, None]) & target                # [B,F]
    inp = tok.clone()
    inp[(lvl_idx > lv) & target[..., None]] = MASK_ID            # levels above l: all MASK
    inp[(lvl_idx == lv) & masked_cell[..., None]] = MASK_ID      # level l: i.i.d. Bernoulli(t)
    inp[~frame_mask] = PAD_ID
    return inp, masked_cell                     # loss cells [B,F], at level `levels`


def masked_ce(model, ph, ph_msk, inp, fr_msk, tok, levels, loss_cells) -> torch.Tensor:
    """Summed cross-entropy over the level-l masked cells of this micro-batch."""
    hidden = model(ph, ph_msk, inp, fr_msk)
    total = hidden.new_zeros(())
    for lvl in levels.unique().tolist():
        sel = (levels == lvl)[:, None] & loss_cells
        if not bool(sel.any()):
            continue
        h = hidden[sel]
        tgt = tok[..., lvl][sel]
        total = total + F.cross_entropy(model.logits(h, lvl).float(), tgt, reduction="sum")
    return total


# ------------------------------------------------------------------- validation
def val_batches(store: TokenStore, batch: int = 256) -> List[np.ndarray]:
    val = store.index[store.index.split == "val"].sort_values("id")
    pos = np.array([store.pos[i] for i in val.id])
    pos = pos[np.argsort(store.a_n_frames[pos], kind="stable")]
    return [pos[i:i + batch] for i in range(0, len(pos), batch)] or [pos]


@torch.no_grad()
def validate_flat(model, store: TokenStore, vbatches: List[np.ndarray], device, mb: int) -> float:
    """Pivot P1-D validation: same fixed ratio grid and fixed val RNG, but the mask is
    joint over all 8 levels, so the per-level average is implicit in the cell average."""
    model.eval()
    tot_loss, tot_cells = 0.0, 0
    for bi, rows in enumerate(vbatches):
        ph, ph_msk, tok, fr_msk = assemble(store, rows)
        for ri, ratio in enumerate(VAL_RATIOS):
            gen = torch.Generator().manual_seed(VAL_RNG * 1000003 + bi * 1000 + ri)
            cells = torch.rand((len(rows), tok.shape[1], N_LEVELS), generator=gen)
            for s in range(0, len(rows), mb):
                sl = slice(s, s + mb)
                t_tok = torch.from_numpy(tok[sl]).to(device)
                t_fr = torch.from_numpy(fr_msk[sl]).to(device)
                rt = torch.full((t_tok.shape[0],), ratio)
                inp, loss_cells = build_inputs_flat(t_tok, t_fr, rt, cells[sl].to(device),
                                                    PROMPT_FRAMES)
                n = int(loss_cells.sum())
                if n == 0:
                    continue
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l = masked_ce_flat(model, torch.from_numpy(ph[sl]).to(device),
                                       torch.from_numpy(ph_msk[sl]).to(device), inp, t_fr,
                                       t_tok, loss_cells)
                tot_loss += float(l)
                tot_cells += n
    model.train()
    return tot_loss / max(1, tot_cells)


@torch.no_grad()
def validate(model, store: TokenStore, vbatches: List[np.ndarray], device, mb: int) -> float:
    model.eval()
    tot_loss, tot_cells = 0.0, 0
    for bi, rows in enumerate(vbatches):
        ph, ph_msk, tok, fr_msk = assemble(store, rows)
        for ri, ratio in enumerate(VAL_RATIOS):
            for lvl in range(N_LEVELS):
                gen = torch.Generator().manual_seed(VAL_RNG * 1000003 + bi * 1000 + lvl * 10 + ri)
                cells = torch.rand((len(rows), tok.shape[1]), generator=gen)
                for s in range(0, len(rows), mb):
                    sl = slice(s, s + mb)
                    t_tok = torch.from_numpy(tok[sl]).to(device)
                    t_fr = torch.from_numpy(fr_msk[sl]).to(device)
                    lv = torch.full((t_tok.shape[0],), lvl, dtype=torch.long)
                    rt = torch.full((t_tok.shape[0],), ratio)
                    inp, loss_cells = build_inputs(t_tok, t_fr, lv, rt, cells[sl].to(device),
                                                   PROMPT_FRAMES)
                    n = int(loss_cells.sum())
                    if n == 0:
                        continue
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        l = masked_ce(model, torch.from_numpy(ph[sl]).to(device),
                                      torch.from_numpy(ph_msk[sl]).to(device), inp, t_fr,
                                      t_tok, lv.to(device), loss_cells)
                    tot_loss += float(l)
                    tot_cells += n
    model.train()
    return tot_loss / max(1, tot_cells)


# ------------------------------------------------------------------------ train
def lr_factor(step: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine to 10 % of peak (grid.json training.schedule)."""
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, required=True, help="muP base LR (at base width 256)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--proxy-width", type=int, default=None, help="LR-sweep proxy shape")
    ap.add_argument("--proxy-depth", type=int, default=None)
    ap.add_argument("--coord-check", type=int, default=0)
    ap.add_argument("--no-val", action="store_true")
    ap.add_argument("--recipe", choices=["coarse", "flat"], default="coarse",
                    help="flat = pivot P1-D (joint masking over all 8 levels)")
    a = ap.parse_args()

    grid = load_grid()
    tr = grid["training"]
    steps = a.steps or tr["steps"]
    val_every = a.val_every or tr["val_cadence_steps"]
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)

    store = TokenStore()
    n_phon = len(store.vocab)
    train_pos = np.where(store.index.split.values == "train")[0]
    plan = BatchPlan(train_pos, store.a_n_frames[train_pos], seed=a.seed,
                     batch=tr["batch_sequences"])

    # the seed must control initialisation as well as data order and masking
    # (grid.json: nothing differs between runs except w, d, heads, seed)
    torch.manual_seed(1000 + a.seed)
    torch.cuda.manual_seed_all(1000 + a.seed)
    if a.proxy_width:                                    # Phase-1 proxy shape
        width, depth = a.proxy_width, a.proxy_depth
        heads = width // 64
        from model import AspectD
        model = AspectD(width, depth, heads, n_phon).to(device)
        cfg_meta = {"id": f"proxy_d{depth}_w{width}", "width": width, "depth": depth,
                    "heads": heads, "nonembed_params": 12 * depth * width ** 2}
    else:
        model = build_model(a.config, n_phon, grid).to(device)
        from model import config_by_id
        cfg_meta = dict(config_by_id(a.config, grid))
    mb = micro_batch_for(model.depth, model.width, tr["batch_sequences"])
    accum = tr["batch_sequences"] // mb

    groups = model.param_groups(a.lr, tr["weight_decay"])
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=tuple(tr["betas"]), eps=1e-8)

    run = {"config": cfg_meta["id"], "seed": a.seed, "lr": a.lr, "steps": steps,
           "width": model.width, "depth": model.depth, "heads": model.n_heads,
           "nonembed_params": model.nonembed_params(), "total_params": model.total_params(),
           "micro_batch": mb, "accum": accum, "device": torch.cuda.get_device_name(device),
           "started": time.time(), "status": "running"}
    with open(os.path.join(a.out, "run.json"), "w") as fh:
        json.dump(run, fh, indent=1)

    ckpt_path = os.path.join(a.out, "ckpt.pt")
    start_step = 0
    val_hist: List[Dict] = []
    prev_seconds = 0.0
    if os.path.exists(ckpt_path):                        # resume (task.md directive 8)
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_step = st["step"]
        val_hist = st.get("val_hist", [])
        prev_seconds = float(st.get("wall_seconds", 0.0))
        # load_state_dict restores the *saved* lr into every group: re-assert the requested
        # base LR so a 0.5x-LR restart / any LR change actually takes effect on resume
        for g, src in zip(opt.param_groups, groups):
            g["base_lr"] = src["base_lr"]
            g["weight_decay"] = src["weight_decay"]
        print(f"[train] resumed {a.out} at step {start_step} (base LR {a.lr})", flush=True)

    flat = a.recipe == "flat"
    run["recipe"] = a.recipe
    vbatches = [] if a.no_val else val_batches(store)
    log_f = open(os.path.join(a.out, "train_log.jsonl"), "a")
    coord_f = open(os.path.join(a.out, "coord_check.jsonl"), "a") if a.coord_check else None
    acts: Dict[int, float] = {}
    if coord_f:
        for bi, blk in enumerate(model.blocks):
            blk.register_forward_hook(
                lambda m, i, o, bi=bi: acts.__setitem__(bi, float(o.float().pow(2).mean().sqrt())))

    # background batch assembly (CPU) so the GPU never waits on numpy
    q: "queue.Queue" = queue.Queue(maxsize=4)

    def producer():
        for step in range(start_step, steps):
            rows = plan.batch_for_step(step)
            q.put((step, rows, assemble(store, rows)))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    t0 = time.time()
    # G3 divergence state survives a resume: rebuild it from the stored val history
    running_min = min([v["val_loss"] for v in val_hist], default=float("inf"))
    above_min = 0
    for v in val_hist:                                   # replay the consecutive-eval counter
        rm = float("inf")
        for w in val_hist:
            if w["step"] < v["step"]:
                rm = min(rm, w["val_loss"])
        above_min = above_min + 1 if v["val_loss"] > 1.2 * rm else 0
    status = "completed"
    tokens_seen = 0
    step = start_step - 1
    loss_val = float("nan")
    while True:
        item = q.get()
        if item is None:
            break
        step, rows, (ph, ph_msk, tok, fr_msk) = item
        B, Fr = tok.shape[0], tok.shape[1]
        gen = torch.Generator().manual_seed(a.seed * 1000003 + step)
        t_tok_all = torch.from_numpy(tok).to(device, non_blocking=True)
        t_fr_all = torch.from_numpy(fr_msk).to(device, non_blocking=True)
        # pre-count loss cells over the whole effective batch (LOG.md P0-6)
        if flat:
            levels = None
            ratios, cells = draw_masks_flat(gen, B, Fr)
            inp_all, cells_all = build_inputs_flat(t_tok_all, t_fr_all, ratios,
                                                   cells.to(device), PROMPT_FRAMES)
        else:
            levels, ratios, cells = draw_masks(gen, B, Fr)
            inp_all, cells_all = build_inputs(t_tok_all, t_fr_all, levels, ratios,
                                              cells.to(device), PROMPT_FRAMES)
        total_cells = int(cells_all.sum())
        if total_cells == 0:
            continue
        t_ph = torch.from_numpy(ph).to(device, non_blocking=True)
        t_phm = torch.from_numpy(ph_msk).to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        fac = lr_factor(step, steps, tr["warmup_steps"])
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * fac
        loss_val = 0.0
        for s in range(0, B, mb):
            sl = slice(s, s + mb)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if flat:
                    l = masked_ce_flat(model, t_ph[sl], t_phm[sl], inp_all[sl], t_fr_all[sl],
                                       t_tok_all[sl], cells_all[sl])
                else:
                    l = masked_ce(model, t_ph[sl], t_phm[sl], inp_all[sl], t_fr_all[sl],
                                  t_tok_all[sl], levels[sl].to(device), cells_all[sl])
            (l / total_cells).backward()
            loss_val += float(l.detach())
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
        opt.step()
        loss_val /= total_cells
        tokens_seen += int(fr_msk.sum()) * N_LEVELS

        if not math.isfinite(loss_val):
            status = "nan"
            print(f"[train] NaN at step {step}", flush=True)
            break
        if step % 20 == 0 or step == steps - 1:
            log_f.write(json.dumps({"step": step, "loss": loss_val, "lr_factor": fac,
                                    "grad_norm": float(gnorm), "cells": total_cells,
                                    "frames": int(fr_msk.sum()), "wall": time.time() - t0}) + "\n")
            log_f.flush()
        if coord_f and step < a.coord_check:
            coord_f.write(json.dumps({"step": step, "width": model.width,
                                      "depth": model.depth, "acts": acts.copy()}) + "\n")
            coord_f.flush()
        if step % 200 == 0:
            el = time.time() - t0
            print(f"[{run['config']}_s{a.seed}] step {step}/{steps} loss {loss_val:.4f} "
                  f"{el/max(1,step-start_step+1):.2f}s/step eta {(steps-step)*el/max(1,step-start_step+1)/3600:.2f}h",
                  flush=True)

        do_val = (not a.no_val) and ((step + 1) % val_every == 0 or step == steps - 1)
        if do_val:
            vl = (validate_flat if flat else validate)(model, store, vbatches, device, mb)
            val_hist.append({"step": step + 1, "val_loss": vl, "wall": time.time() - t0})
            with open(os.path.join(a.out, "val_log.jsonl"), "a") as vf:
                vf.write(json.dumps(val_hist[-1]) + "\n")
            print(f"[{run['config']}_s{a.seed}] val@{step+1} {vl:.4f}", flush=True)
            if vl > 1.2 * running_min:                   # gate G3 divergence definition
                above_min += 1
            else:
                above_min = 0
            running_min = min(running_min, vl)
            if above_min >= 3:
                status = "diverged"
                print("[train] diverged per G3", flush=True)
                break
        if (step + 1) % a.ckpt_every == 0 or step == steps - 1:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step + 1,
                        "val_hist": val_hist, "cfg": cfg_meta, "lr": a.lr, "seed": a.seed,
                        "wall_seconds": prev_seconds + time.time() - t0},
                       ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)

    wall = prev_seconds + time.time() - t0
    run.update({"status": status, "wall_seconds": wall, "gpu_hours": wall / 3600,
                "wall_seconds_this_segment": time.time() - t0,
                "final_step": step + 1, "val_hist": val_hist, "tokens_seen": tokens_seen,
                "final_train_loss": loss_val,
                "final_val_loss": val_hist[-1]["val_loss"] if val_hist else None,
                "finished": time.time()})
    with open(os.path.join(a.out, "run.json"), "w") as fh:
        json.dump(run, fh, indent=1)
    log_f.close()
    if coord_f:
        coord_f.close()
    print(f"[train] {run['config']}_s{a.seed} {status} in {wall/3600:.2f} GPU-h", flush=True)
    raise SystemExit(0 if status == "completed" else (2 if status == "nan" else 3))


if __name__ == "__main__":
    main()
