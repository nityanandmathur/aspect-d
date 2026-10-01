"""Parameter accounting: the paper's x-axis is N_nonembed = 12 * depth * width^2."""
import glob
import json
import os

import pytest
import torch

from model import (AUDIO_VOCAB, CODEBOOK_SIZE, HEAD_DIM, N_LEVELS, AspectD, build_model,
                   load_grid)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = load_grid()
CONFIGS = GRID["configs"]


def _meta_model(width, depth, heads, n_phonemes=2):
    with torch.device("meta"):                       # shapes only: no allocation, no init
        return AspectD(width, depth, heads, n_phonemes)


def _matrix_params(model):
    return sum(p.numel() for p in model.blocks.parameters() if p.ndim == 2)


def _expected_total(width, depth, n_phonemes):
    nonembed = 12 * depth * width ** 2 + (2 * depth + 1) * width       # + RMSNorm gains
    return (nonembed + n_phonemes * width + N_LEVELS * AUDIO_VOCAB * width
            + N_LEVELS * CODEBOOK_SIZE * width)


@pytest.mark.parametrize("c", CONFIGS, ids=[c["id"] for c in CONFIGS])
def test_grid_formula(c):
    """grid.json's nonembed_params is exactly 12 d w^2 and every width is a whole number
    of 64-wide heads."""
    assert c["nonembed_params"] == 12 * c["depth"] * c["width"] ** 2
    assert c["heads"] * HEAD_DIM == c["width"]


@pytest.mark.parametrize("c", CONFIGS, ids=[c["id"] for c in CONFIGS])
def test_build_model_counts(c):
    """The weight matrices build_model actually creates are 12 d w^2; nonembed_params()
    adds only the 2d+1 RMSNorm gain vectors."""
    with torch.device("meta"):
        m = build_model(c["id"], n_phonemes=7)
    d, w = c["depth"], c["width"]
    assert len(m.blocks) == d and m.width == w
    assert _matrix_params(m) == 12 * d * w ** 2 == c["nonembed_params"]
    assert m.nonembed_params() == 12 * d * w ** 2 + (2 * d + 1) * w
    assert m.total_params() == _expected_total(w, d, 7)


def _committed_runs():
    out = []
    for f in sorted(glob.glob(os.path.join(REPO, "runs*", "*", "run.json"))):
        r = json.load(open(f))
        if all(k in r for k in ("width", "depth", "heads", "nonembed_params", "total_params")):
            out.append((os.path.relpath(f, REPO), r))
    return out


RUNS = _committed_runs()


@pytest.mark.skipif(not RUNS, reason="no committed run.json files")
@pytest.mark.parametrize("path,r", RUNS, ids=[p for p, _ in RUNS])
def test_committed_runs_match_model(path, r):
    """Every committed run record agrees with the model code: non-embedding count exactly,
    and the total implies a whole-number phoneme vocabulary."""
    w, d, h = r["width"], r["depth"], r["heads"]
    m = _meta_model(w, d, h)
    assert m.nonembed_params() == r["nonembed_params"]
    rest = r["total_params"] - r["nonembed_params"] - N_LEVELS * (AUDIO_VOCAB + CODEBOOK_SIZE) * w
    assert rest % w == 0 and rest // w >= 2, "total_params implies a fractional vocab"
    assert _meta_model(w, d, h, rest // w).total_params() == r["total_params"]


@pytest.mark.skipif(not RUNS, reason="no committed run.json files")
def test_grid_runs_share_one_vocab():
    """All 45 v1.0 grid runs were trained on one phoneme vocabulary."""
    ids = {c["id"] for c in CONFIGS}
    vs = set()
    for path, r in RUNS:
        name = os.path.basename(os.path.dirname(path))
        if path.startswith("runs" + os.sep) and name.rsplit("_", 1)[0] in ids:
            w = r["width"]
            vs.add((r["total_params"] - r["nonembed_params"]
                    - N_LEVELS * (AUDIO_VOCAB + CODEBOOK_SIZE) * w) // w)
    assert len(vs) == 1, vs
