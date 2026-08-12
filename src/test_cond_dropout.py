"""Verify S5 condition dropout before spending 45 GPU-h on it.

The dropout has to satisfy four properties, and each test fails for a different real
mistake rather than restating the implementation:

1. p=0 is a strict no-op -- otherwise every existing checkpoint's training recipe has
   silently changed and the CFG arm is no longer comparable to the base grid.
2. A dropped prompt is genuinely gone: PAD in the inputs AND unattended. Feeding PAD
   while leaving the attention bit set is exactly the bug found in the sampler's
   contrastive branch on 2026-08-12, so it is tested here rather than assumed.
3. Dropout must not move the LOSS TARGET. The target is frame_mask & (index >= PF),
   and clearing prompt frames (index < PF) must leave it identical -- if it did not,
   the gradient normaliser would change and p=0 vs p>0 would not be comparable.
4. The three cells are disjoint and hit their nominal rates, so an example is never
   both text- and prompt-dropped (a joint null the model never has to handle).

    python src/test_cond_dropout.py
"""
from __future__ import annotations

import sys

import torch

from data import PROMPT_FRAMES
from model import MASK_ID, N_LEVELS, PAD_ID
from train import build_inputs, draw_masks

B, FR, P = 64, 96, PROMPT_FRAMES
FAIL = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        FAIL.append(name)


def fixture():
    g = torch.Generator().manual_seed(11)
    tok = torch.randint(0, 2048, (B, FR, N_LEVELS), generator=g)
    fmask = torch.zeros((B, FR), dtype=torch.bool)
    for i in range(B):
        fmask[i, :int(torch.randint(P + 20, FR, (1,), generator=g))] = True
    levels, ratios, cells = draw_masks(torch.Generator().manual_seed(3), B, FR)
    return tok, fmask, levels, ratios, cells


def apply_dropout(fmask, phm, p, seed=0, step=0):
    """Mirrors train.py exactly: one uniform, three disjoint cells."""
    dgen = torch.Generator().manual_seed(seed * 1000003 + step + 500_000_000)
    u = torch.rand((B,), generator=dgen)
    drop_p, drop_t = u < p, (u >= p) & (u < 2 * p)
    pre = torch.arange(FR)[None, :] < P
    fm = fmask.clone()
    fm[drop_p[:, None] & pre] = False
    ph = phm.clone()
    ph[drop_t] = False
    return fm, ph, drop_p, drop_t


def main() -> int:
    tok, fmask, levels, ratios, cells = fixture()
    phm = torch.ones((B, 40), dtype=torch.bool)

    print("1. p=0 is a strict no-op")
    fm0, ph0, dp0, dt0 = apply_dropout(fmask, phm, 0.0)
    base_inp, base_cells = build_inputs(tok, fmask, levels, ratios, cells, P)
    inp0, cells0 = build_inputs(tok, fm0, levels, ratios, cells, P)
    check("frame mask untouched", torch.equal(fm0, fmask))
    check("phoneme mask untouched", torch.equal(ph0, phm))
    check("inputs bit-identical", torch.equal(inp0, base_inp))
    check("loss cells bit-identical", torch.equal(cells0, base_cells))
    check("nothing dropped", int(dp0.sum()) == 0 and int(dt0.sum()) == 0)

    print("2. a dropped prompt is PAD and unattended")
    fm, ph, drop_p, drop_t = apply_dropout(fmask, phm, 0.10, seed=7)
    inp, _ = build_inputs(tok, fm, levels, ratios, cells, P)
    idx = drop_p.nonzero().flatten()
    if len(idx) == 0:
        check("at least one example dropped", False, "rate too low for this fixture")
    else:
        pre = torch.arange(FR)[None, :] < P
        sel = drop_p[:, None] & pre
        check("dropped prompt frames are PAD in the inputs",
              bool((inp[sel] == PAD_ID).all()))
        check("dropped prompt frames are unattended", bool((~fm[sel]).all()))
        keep = ~drop_p
        check("kept rows are byte-identical to no-dropout",
              torch.equal(inp[keep], base_inp[keep]))

    print("3. dropout does not move the loss target")
    _, cells_drop = build_inputs(tok, fm, levels, ratios, cells, P)
    check("loss cells identical to no-dropout", torch.equal(cells_drop, base_cells),
          f"{int(base_cells.sum())} cells")
    check("gradient normaliser unchanged",
          int(cells_drop.sum()) == int(base_cells.sum()))

    print("4. the three cells are disjoint and hit their rates")
    np_, nt_, n = 0, 0, 0
    for step in range(4000):
        _, _, dp, dt = apply_dropout(fmask, phm, 0.10, seed=1, step=step)
        check_overlap = bool((dp & dt).any())
        if check_overlap:
            check("prompt and text drops never overlap", False, f"step {step}")
            break
        np_ += int(dp.sum()); nt_ += int(dt.sum()); n += B
    else:
        check("prompt and text drops never overlap", True)
    rp, rt = np_ / n, nt_ / n
    check("realised prompt-drop rate in [0.085, 0.115]", 0.085 <= rp <= 0.115, f"{rp:.4f}")
    check("realised text-drop rate in [0.085, 0.115]", 0.085 <= rt <= 0.115, f"{rt:.4f}")

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED: {FAIL}")
        return 1
    print("all checks passed -- condition dropout is safe to train with")
    return 0


if __name__ == "__main__":
    sys.exit(main())
