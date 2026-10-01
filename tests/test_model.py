"""Forward-pass contract of the backbone on a tiny random model."""
import pytest
import torch

from model import CODEBOOK_SIZE, MASK_ID, N_LEVELS, PAD_ID, AspectD


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return AspectD(width=128, depth=2, heads=2, n_phonemes=11).eval()


def _inputs(B=2, P=6, Fr=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    ph = torch.randint(2, 11, (B, P), generator=g)
    phm = torch.ones(B, P, dtype=torch.bool)
    phm[1, 4:] = False
    au = torch.randint(0, CODEBOOK_SIZE, (B, Fr, N_LEVELS), generator=g)
    au[:, 5:, 3:] = MASK_ID
    fm = torch.ones(B, Fr, dtype=torch.bool)
    fm[1, 7:] = False
    au[1, 7:] = PAD_ID
    return ph, phm, au, fm


@torch.no_grad()
def test_shapes(tiny):
    ph, phm, au, fm = _inputs()
    h = tiny(ph, phm, au, fm)
    assert h.shape == (2, 9, 128)
    for lvl in range(N_LEVELS):
        assert tiny.logits(h, lvl).shape == (2, 9, CODEBOOK_SIZE)
    assert torch.isfinite(h).all()


@torch.no_grad()
def test_padding_is_invisible(tiny):
    """Keys with mask False (padded phonemes / frames) must not affect real positions."""
    ph, phm, au, fm = _inputs()
    h0 = tiny(ph, phm, au, fm)
    ph2, au2 = ph.clone(), au.clone()
    ph2[1, 4:] = 3
    au2[1, 7:] = 17
    h1 = tiny(ph2, phm, au2, fm)
    torch.testing.assert_close(h0[1, :7], h1[1, :7])
    torch.testing.assert_close(h0[0], h1[0])


@torch.no_grad()
def test_attention_is_bidirectional(tiny):
    """No causal mask: changing the LAST frame changes the FIRST frame's state, and
    changing a phoneme changes the audio states."""
    ph, phm, au, fm = _inputs()
    h0 = tiny(ph, phm, au, fm)
    au2 = au.clone()
    au2[0, 8, 0] = (au2[0, 8, 0] + 1) % CODEBOOK_SIZE
    assert not torch.allclose(h0[0, 0], tiny(ph, phm, au2, fm)[0, 0])
    ph2 = ph.clone()
    ph2[0, 0] = 2 if ph[0, 0] != 2 else 3
    assert not torch.allclose(h0[0, 0], tiny(ph2, phm, au, fm)[0, 0])
