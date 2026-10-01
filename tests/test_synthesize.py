"""synthesize.py builds the same conditioning as the paper's eval harness."""
import json
import shutil
import types

import numpy as np
import pytest
import soundfile as sf
import torch

import sample
import synthesize as S
from data import FRAME_SAMPLES, SR
from model import CODEBOOK_SIZE, N_LEVELS, PHONEME_POS_CAP, AspectD


class FakeStore:
    """Duck-types data.TokenStore for sample.load_items."""

    def __init__(self, clips, vocab, spc):
        self.clips, self.vocab, self.dataset = clips, vocab, {"sec_per_char": spc}

    def get(self, cid):
        return self.clips[cid]


@pytest.mark.parametrize("n_tgt_phones,text", [
    (40, "A short sentence for the model."),
    (600, "x" * 400),                          # phoneme cap and 15 s length clamp
])
def test_make_item_matches_load_items(tmp_path, monkeypatch, n_tgt_phones, text):
    rng = np.random.default_rng(0)
    vocab = {"<pad>": 0, "<unk>": 1, "<sp>": 20}
    ph_p = rng.integers(2, 150, 30).astype(np.int64)
    ph_t = rng.integers(2, 150, n_tgt_phones).astype(np.int64)
    tk_p = rng.integers(0, CODEBOOK_SIZE, (38, N_LEVELS)).astype(np.int64)
    spc = 0.0601
    store = FakeStore({"p": (ph_p, tk_p), "t": (ph_t, None)}, vocab, spc)
    (tmp_path / "eval_zs.json").write_text(json.dumps(
        [{"item": "it0000", "prompt_id": "p", "target_id": "t", "target_text": text}]))
    monkeypatch.setattr(sample, "PROC_DIR", str(tmp_path))
    ref = sample.load_items(store)[0]
    got = S.make_item(ph_p, ph_t, tk_p, text, spc, vocab["<sp>"])
    assert np.array_equal(got["phonemes"], ref["phonemes"])
    assert len(got["phonemes"]) <= PHONEME_POS_CAP
    assert np.array_equal(got["prompt_tokens"], ref["prompt_tokens"])
    for k in ("n_prompt", "n_target", "n_chars"):
        assert got[k] == ref[k], k


def test_target_frames_formula():
    assert S.target_frames("  " + "a" * 100 + " ", 0.06) == round(100 * 0.06 * 12.5)
    assert S.target_frames("a", 0.001) == 1
    assert S.target_frames("a" * 10_000, 0.06) == sample.MAX_TARGET_FRAMES


def test_to_ids_maps_oov_to_unk():
    ids, oov = S.to_ids(["a", "<sp>", "zz"], {"<pad>": 0, "<unk>": 1, "a": 5, "<sp>": 6})
    assert ids.tolist() == [5, 6, 1] and oov == 1


def test_prompt_audio_resample_mono_cut_peak(tmp_path):
    sr = 48000
    t = np.arange(int(5.0 * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * 220 * t)
    path = tmp_path / "p.wav"
    sf.write(path, np.stack([x, 0.5 * x], 1), sr)
    w, cut = S.load_prompt_audio(str(path))
    assert cut and w.ndim == 1
    assert len(w) == int(round(S.MAX_PROMPT_SECONDS * SR))
    assert abs(np.abs(w).max() - 10 ** (-1 / 20)) < 1e-4          # data.PEAK_DBFS
    short = tmp_path / "s.wav"
    sf.write(short, x[: int(3.1 * sr)], sr)
    w2, cut2 = S.load_prompt_audio(str(short))
    assert not cut2 and abs(len(w2) - int(3.1 * SR)) <= 1


def _has_espeak():
    try:
        S.phonemize(["test"])
        return True
    except Exception:
        return False


@pytest.mark.skipif(shutil.which("espeak-ng") is None or not _has_espeak(),
                    reason="espeak-ng / phonemizer not available")
def test_phonemize_tokenisation():
    """Words are joined by <sp>; preserved punctuation is split off as its own token;
    stress marks are kept."""
    (toks,) = S.phonemize(["Hello, world."])
    assert toks.count("<sp>") == 1
    assert "," in toks and toks[-1] == "."
    assert any("ˈ" in t for t in toks)


class FakeMimi:
    """Stands in for transformers.MimiModel: 1920 samples per frame both ways."""

    def encode(self, x, num_quantizers):
        n = x.shape[-1] // FRAME_SAMPLES
        return types.SimpleNamespace(
            audio_codes=torch.zeros((x.shape[0], num_quantizers, n), dtype=torch.long))

    def decode(self, codes):
        n = codes.shape[-1]
        return types.SimpleNamespace(audio_values=torch.full((1, 1, n * FRAME_SAMPLES), 0.1))


def test_synthesize_glue(tmp_path, monkeypatch):
    """End-to-end on a tiny random model with a fake codec: output is exactly the target
    region (n_target frames of audio), the prompt is cut away, NFE is 8T."""
    torch.manual_seed(0)
    vocab = {"<pad>": 0, "<unk>": 1, "<sp>": 2, "a": 3, "b": 4}
    model = AspectD(64, 1, 1, len(vocab)).eval()
    tts = S.TTS(model, FakeMimi(), vocab, 0.06, torch.device("cpu"),
                {"width": 64, "depth": 1, "config_id": "tiny"}, "tiny")
    monkeypatch.setattr(S, "phonemize", lambda texts: [["a", "b", "zz"] for _ in texts])
    p = tmp_path / "p.wav"
    sf.write(p, 0.2 * np.random.default_rng(0).standard_normal(int(3.2 * 16000)), 16000)
    text = "Twenty-four characters."
    wav, info = S.synthesize(tts, text, str(p), "prompt words", steps=2, seed=3)
    n_t = S.target_frames(text, 0.06)
    assert info["n_target_frames"] == n_t and info["nfe"] == 16
    assert info["n_prompt_frames"] == int(3.2 * SR) // FRAME_SAMPLES
    assert len(wav) == n_t * FRAME_SAMPLES
    assert info["oov_phonemes"] == 2
    with pytest.raises(ValueError):
        S.synthesize(tts, text, str(p), "  ")
