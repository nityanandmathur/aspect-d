"""Released checkpoints load into model.py (network; ASPECTD_NETWORK_TESTS=1)."""
import json
import os

import numpy as np
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.network
def test_download_and_load_A1_0():
    import synthesize as S
    paths = S.fetch_run("A1_0")
    model, cfg, vocab = S.load_model(paths, torch.device("cpu"))      # strict key match
    assert model.nonembed_params() == cfg["nonembed_params"]
    assert model.total_params() == cfg["total_params"]
    assert (cfg["width"], cfg["depth"], cfg["heads"]) == (640, 4, 10)
    assert "<sp>" in vocab
    assert 0 < json.load(open(paths["dataset.json"]))["sec_per_char"] < 1


@pytest.mark.network
def test_end_to_end_A1_0_T1():
    """Full path incl. Mimi and espeak-ng on the shipped it0000 prompt, T=1 (8 forwards)."""
    import synthesize as S
    man = json.load(open(os.path.join(REPO, "samples", "manifest.json")))["items"][0]
    tts = S.load_tts(run="A1_0", device="cpu")
    wav, info = S.synthesize(tts, man["target_text"],
                             os.path.join(REPO, "samples", man["prompt_file"]),
                             man["prompt_text"], steps=1, seed=0)
    assert info["nfe"] == 8
    assert len(wav) == info["n_target_frames"] * 1920
    assert info["n_target_frames"] == S.target_frames(man["target_text"], tts.sec_per_char)
    assert np.isfinite(wav).all() and float(np.sqrt(np.mean(wav ** 2))) > 1e-3
