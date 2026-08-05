"""ASPECT-D backbone — bidirectional masked-diffusion transformer over Mimi tokens.

Architecture is frozen by ``configs/grid.json -> backbone``:
pre-LN RMSNorm, RoPE, full bidirectional attention, GELU MLP ratio 4,
head_dim 64, no weight tying, one embedding table per codebook (summed at the
input) and one output head per codebook.

    N_nonembed = 12 * depth * width**2   (attention 4w^2 + MLP 8w^2 per block)

Implementation decisions that grid.json/protocol.html do not fix are recorded in
LOG.md: P0-1 (sequence layout / RoPE offsets) and P0-5 (muP realization).
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- token space
N_LEVELS = 8            # grid.json backbone.codec.codebooks_used
CODEBOOK_SIZE = 2048    # grid.json backbone.codec.vocab_per_codebook
MASK_ID = 2048          # absorbing state, one per codebook table
PAD_ID = 2049
AUDIO_VOCAB = 2050
FRAME_RATE_HZ = 12.5

PHONEME_POS_CAP = 512   # RoPE position offset of audio frames (LOG.md P0-1)
BASE_WIDTH = 256        # grid.json training.mup.base_width
HEAD_DIM = 64           # grid.json backbone.head_dim
MLP_RATIO = 4
INIT_STD = 0.02

_GRID_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "configs", "grid.json")


def load_grid(path: str = _GRID_PATH) -> Dict:
    with open(path) as fh:
        return json.load(fh)


def config_by_id(cfg_id: str, grid: Optional[Dict] = None) -> Dict:
    grid = grid or load_grid()
    for c in grid["configs"]:
        if c["id"] == cfg_id:
            return c
    raise KeyError(f"config {cfg_id!r} not in grid.json")


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dtype)


def build_rope_table(max_pos: int, device, theta: float = 10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float32) / HEAD_DIM))
    freqs = torch.outer(torch.arange(max_pos, device=device, dtype=torch.float32), inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, S, D]; cos/sin: [S, D/2] (interleaved-pair convention)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None], sin[None, None]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack((o1, o2), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        assert heads * HEAD_DIM == width, f"heads*{HEAD_DIM} != width ({heads}, {width})"
        self.heads = heads
        self.norm1 = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        self.norm2 = RMSNorm(width)
        self.up = nn.Linear(width, MLP_RATIO * width, bias=False)
        self.down = nn.Linear(MLP_RATIO * width, width, bias=False)

    def forward(self, x, cos, sin, attn_mask):
        B, S, W = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, S, 3, self.heads, HEAD_DIM).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.o(a.transpose(1, 2).reshape(B, S, W))
        x = x + self.down(F.gelu(self.up(self.norm2(x))))
        return x


class AspectD(nn.Module):
    """Bidirectional masked-diffusion transformer. Sequence = [phonemes | frames]."""

    def __init__(self, width: int, depth: int, heads: int, n_phonemes: int):
        super().__init__()
        self.width, self.depth, self.n_heads = width, depth, heads
        self.mup_m = width / BASE_WIDTH
        self.out_mult = BASE_WIDTH / width          # muP output multiplier (LOG.md P0-5)
        self.phone_emb = nn.Embedding(n_phonemes, width)
        self.audio_emb = nn.ModuleList([nn.Embedding(AUDIO_VOCAB, width) for _ in range(N_LEVELS)])
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(depth)])
        self.norm_f = RMSNorm(width)
        self.out_heads = nn.ModuleList([nn.Linear(width, CODEBOOK_SIZE, bias=False)
                                        for _ in range(N_LEVELS)])
        self._rope_cache: Dict[tuple, tuple] = {}
        self._init_weights()

    # ------------------------------------------------------------------ init
    def _init_weights(self):
        m = self.mup_m
        nn.init.normal_(self.phone_emb.weight, std=INIT_STD)
        for emb in self.audio_emb:
            nn.init.normal_(emb.weight, std=INIT_STD)
        res = 1.0 / math.sqrt(2.0 * self.depth)     # grid.json residual_scale
        for blk in self.blocks:
            for lin, fan_in, scale in ((blk.qkv, self.width, 1.0),
                                       (blk.o, self.width, res),
                                       (blk.up, self.width, 1.0),
                                       (blk.down, MLP_RATIO * self.width, res)):
                nn.init.normal_(lin.weight, std=INIT_STD * math.sqrt(BASE_WIDTH / fan_in) * scale)
        for head in self.out_heads:
            nn.init.normal_(head.weight, std=INIT_STD / math.sqrt(m))

    def param_groups(self, base_lr: float, weight_decay: float) -> List[Dict]:
        """muP: hidden matrices and output heads get base_lr / m; vectors get base_lr."""
        vec, emb, rest = [], [], []
        for name, p in self.named_parameters():
            if p.ndim == 1:
                vec.append(p)
            elif name.startswith(("phone_emb", "audio_emb")):
                emb.append(p)
            else:
                rest.append(p)
        return [
            {"params": vec, "lr": base_lr, "weight_decay": 0.0, "name": "norm_gains"},
            {"params": emb, "lr": base_lr, "weight_decay": weight_decay, "name": "embeddings"},
            {"params": rest, "lr": base_lr / self.mup_m, "weight_decay": weight_decay,
             "name": "hidden_and_heads"},
        ]

    def nonembed_params(self) -> int:
        """Blocks + final norm; excludes embedding tables and output heads."""
        return (sum(p.numel() for p in self.blocks.parameters())
                + self.norm_f.weight.numel())

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # --------------------------------------------------------------- forward
    def _rope(self, n_phon: int, n_frames: int, device):
        key = (n_phon, n_frames, str(device))
        if key not in self._rope_cache:
            if len(self._rope_cache) > 64:
                self._rope_cache.clear()
            cos, sin = build_rope_table(PHONEME_POS_CAP + n_frames + 1, device)
            pos = torch.cat([torch.arange(n_phon, device=device),
                             PHONEME_POS_CAP + torch.arange(n_frames, device=device)])
            self._rope_cache[key] = (cos[pos], sin[pos])
        return self._rope_cache[key]

    def forward(self, phonemes: torch.Tensor, phone_mask: torch.Tensor,
                audio: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        """phonemes [B,P] long, phone_mask [B,P] bool (True = real token),
        audio [B,F,8] long (MASK_ID / PAD_ID allowed), frame_mask [B,F] bool.
        Returns hidden states at the audio positions: [B,F,width]."""
        B, P = phonemes.shape
        Fr = audio.shape[1]
        x_ph = self.phone_emb(phonemes)
        x_au = self.audio_emb[0](audio[..., 0])
        for lvl in range(1, N_LEVELS):
            x_au = x_au + self.audio_emb[lvl](audio[..., lvl])
        x = torch.cat([x_ph, x_au], dim=1)
        cos, sin = self._rope(P, Fr, x.device)
        keep = torch.cat([phone_mask, frame_mask], dim=1)           # [B,S]
        attn_mask = keep[:, None, None, :]      # broadcast over query axis
        for blk in self.blocks:
            x = blk(x, cos, sin, attn_mask)
        return self.norm_f(x[:, P:])

    def logits(self, hidden: torch.Tensor, level: int) -> torch.Tensor:
        """Codebook-``level`` logits from hidden states (any leading shape)."""
        return self.out_heads[level](hidden) * self.out_mult


def build_model(cfg_id: str, n_phonemes: int, grid: Optional[Dict] = None) -> AspectD:
    c = config_by_id(cfg_id, grid)
    return AspectD(width=c["width"], depth=c["depth"], heads=c["heads"], n_phonemes=n_phonemes)
