"""The frozen MaskGIT sampler (sample.synth_batch) on a tiny random model."""
import numpy as np
import pytest
import torch

from model import CODEBOOK_SIZE, N_LEVELS, PAD_ID, AspectD
from sample import synth_batch

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return AspectD(width=64, depth=1, heads=1, n_phonemes=9).eval()


def _item(n_prompt=4, n_target=11, n_phon=7, seed=0):
    rng = np.random.default_rng(seed)
    return {"phonemes": rng.integers(2, 9, n_phon).astype(np.int64),
            "prompt_tokens": rng.integers(0, CODEBOOK_SIZE, (n_prompt, N_LEVELS)).astype(np.int64),
            "n_prompt": n_prompt, "n_target": n_target}


class Counter:
    def __init__(self, model):
        self.n = 0
        self.h = model.register_forward_hook(self)

    def __call__(self, *_):
        self.n += 1

    def close(self):
        self.h.remove()


@pytest.mark.parametrize("T", [1, 2, 3, 8])
def test_nfe_is_8T_and_everything_unmasked(tiny, T):
    item = _item()
    c = Counter(tiny)
    try:
        grid, fmask = synth_batch(tiny, [item], T, CPU, 0)
    finally:
        c.close()
    assert c.n == N_LEVELS * T, "NFE must be exactly 8T (T steps per level, one pass each)"
    n_p, n_t = item["n_prompt"], item["n_target"]
    assert grid.shape == (1, n_p + n_t, N_LEVELS)
    assert fmask.all()
    tgt = grid[0, n_p:]
    assert ((tgt >= 0) & (tgt < CODEBOOK_SIZE)).all(), "MASK/PAD left in the target"
    assert torch.equal(grid[0, :n_p], torch.from_numpy(item["prompt_tokens"])), \
        "prompt frames must be copied verbatim"


def test_batch_padding_and_lengths(tiny):
    a, b = _item(4, 11, 7, 0), _item(6, 5, 3, 1)
    grid, fmask = synth_batch(tiny, [a, b], 2, CPU, 0)
    assert grid.shape == (2, 15, N_LEVELS)
    assert fmask[1].sum() == 11 and not fmask[1, 11:].any()
    assert (grid[1, 11:] == PAD_ID).all()
    assert ((grid[1, 6:11] >= 0) & (grid[1, 6:11] < CODEBOOK_SIZE)).all()


def test_rng_is_keyed_and_cand_shifts_it(tiny):
    """Same (batch_idx, cand) -> identical tokens; another cand -> an independent draw."""
    item = _item(n_target=40)
    g0, _ = synth_batch(tiny, [item], 4, CPU, 0)
    g1, _ = synth_batch(tiny, [item], 4, CPU, 0)
    g2, _ = synth_batch(tiny, [item], 4, CPU, 0, cand=1)
    assert torch.equal(g0, g1)
    assert not torch.equal(g0[0, 4:], g2[0, 4:])
