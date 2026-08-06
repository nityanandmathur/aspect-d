"""ASPECT-D evaluation — protocol.html §6.

Metrics (grid.json eval_models): WER from Whisper-large-v3 (greedy, Whisper English
normalizer), SIM-o from the WavLM SV model against the ORIGINAL prompt waveform,
UTMOS22-strong, and the frozen degenerate-output rule §6.4. Primary metrics are
computed over ALL items (degenerates included, §1).

    python src/evaluate.py score  --jobs jobs.json --device cuda:0
    python src/evaluate.py gt     --device cuda:0        # gate G0(c)
    python src/evaluate.py collect --out artifacts/runs.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from data import PROC_DIR, SR

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIM_PRIMARY = "wavlm_large_sv_unispeech(seed-tts-eval)"
SIM_FALLBACK = "microsoft/wavlm-base-plus-sv"
SV_DIR = "/home/ubuntu/models/wavlm_sv"


# --------------------------------------------------------------- degenerate rule
def repeated_4gram(words: List[str], times: int = 6) -> bool:
    """True iff some 4-gram repeats >= `times` times consecutively (protocol §6.4)."""
    n = len(words)
    for i in range(n - 4 * times + 1):
        g = words[i:i + 4]
        k = 1
        while i + 4 * (k + 1) <= n and words[i + 4 * k:i + 4 * (k + 1)] == g:
            k += 1
        if k >= times:
            return True
    return False


def is_degenerate(hyp_norm: str, ref_norm: str) -> bool:
    if not hyp_norm.strip():
        return True
    if len(hyp_norm) < 0.30 * max(1, len(ref_norm)):
        return True
    return repeated_4gram(hyp_norm.split())


# ---------------------------------------------------------------------- scorer
class Scorer:
    def __init__(self, device: str = "cuda:0", use_utmos: bool = True,
                 asr_id: str = "openai/whisper-large-v3", force_sv_fallback: bool = False):
        """E5 robustness panel: `asr_id` and `force_sv_fallback` swap the metric stack.
        Defaults reproduce the v1.0 stack bit-for-bit."""
        import jiwer  # noqa: F401
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        self.device = torch.device(device)
        self.asr_id = asr_id
        self.proc = WhisperProcessor.from_pretrained(asr_id)
        self.asr = WhisperForConditionalGeneration.from_pretrained(
            asr_id, dtype=torch.float16).to(self.device).eval()
        self.norm = EnglishTextNormalizer(self.proc.tokenizer.english_spelling_normalizer)
        self.sim_model_name, self.sv, self.sv_kind = self._load_sv(force_sv_fallback)
        self.utmos = None
        if use_utmos:
            try:
                self.utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                                            trust_repo=True).to(self.device).eval()
            except Exception as e:                       # policy: tertiary, droppable
                print("[eval] UTMOS unavailable:", repr(e)[:120], flush=True)

    @staticmethod
    def _shim_torchaudio():
        """s3prl 0.4.18 imports two torchaudio APIs removed in torchaudio 2.x
        (`set_audio_backend`, `sox_effects`). Neither is reachable from the WavLM-large
        upstream path we use; stub them so the primary SIM model can load. LOG.md D-002."""
        import sys
        import types
        import torchaudio
        torchaudio.set_audio_backend = lambda *a, **k: None
        torchaudio.get_audio_backend = lambda *a, **k: "soundfile"
        if "torchaudio.sox_effects" not in sys.modules:
            mod = types.ModuleType("torchaudio.sox_effects")
            for f in ("apply_effects_tensor", "apply_effects_file", "init_sox_effects",
                      "effect_names"):
                setattr(mod, f, lambda *a, **k: None)
            sys.modules["torchaudio.sox_effects"] = mod
            torchaudio.sox_effects = mod

    def _load_sv(self, force_fallback: bool = False):
        try:
            if force_fallback:
                raise RuntimeError("E5: forced SIM fallback (microsoft/wavlm-base-plus-sv)")
            import sys
            self._shim_torchaudio()
            sys.path.insert(0, SV_DIR)
            from models.ecapa_tdnn import ECAPA_TDNN_SMALL
            m = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
            sd = torch.load(os.path.join(SV_DIR, "wavlm_large_finetune.pth"),
                            map_location="cpu", weights_only=False)
            m.load_state_dict(sd["model"], strict=False)
            return SIM_PRIMARY, m.to(self.device).eval(), "unispeech"
        except Exception as e:
            print("[eval] primary SIM model unavailable →", SIM_FALLBACK, repr(e)[:120], flush=True)
            from transformers import WavLMForXVector, AutoFeatureExtractor
            fe = AutoFeatureExtractor.from_pretrained(SIM_FALLBACK)
            m = WavLMForXVector.from_pretrained(SIM_FALLBACK).to(self.device).eval()
            return SIM_FALLBACK, (fe, m), "transformers"

    # ------------------------------------------------------------------ pieces
    @torch.no_grad()
    def transcribe(self, wavs: List[np.ndarray], batch: int = 24) -> List[str]:
        """Whisper expects 16 kHz; the harness works at 24 kHz (Mimi), so resample here."""
        import torchaudio.functional as AF
        out = []
        for s in range(0, len(wavs), batch):
            chunk = [w if len(w) else np.zeros(SR // 10, np.float32) for w in wavs[s:s + batch]]
            chunk = [AF.resample(torch.from_numpy(np.asarray(w, dtype=np.float32)),
                                 SR, 16000).numpy() for w in chunk]
            feats = self.proc(chunk, sampling_rate=16000, return_tensors="pt",
                              return_attention_mask=True)
            ids = self.asr.generate(
                feats.input_features.to(self.device, torch.float16),
                attention_mask=feats.attention_mask.to(self.device),
                do_sample=False, num_beams=1, language="en", task="transcribe",
                max_new_tokens=220)
            out.extend(self.proc.batch_decode(ids, skip_special_tokens=True))
        return out

    @torch.no_grad()
    def embed(self, wavs: List[np.ndarray]) -> torch.Tensor:
        """One utterance per forward pass. The SV stack has no padding mask, so a
        zero-padded batch changes the attentive-statistics pooling and shifts the
        embedding — batching here would silently perturb the primary SIM-o metric
        (LOG.md D-004)."""
        import torchaudio.functional as AF
        embs = []
        for w in wavs:
            x = AF.resample(torch.from_numpy(np.asarray(w, dtype=np.float32)), SR, 16000)
            x = x[None].to(self.device)
            if self.sv_kind == "unispeech":
                e = self.sv(x)
            else:
                fe, m = self.sv
                e = m(input_values=x).embeddings
            embs.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
        return torch.cat(embs)

    @torch.no_grad()
    def mos(self, wavs: List[np.ndarray]) -> Optional[List[float]]:
        if self.utmos is None:
            return None
        out = []
        for w in wavs:
            x = torch.from_numpy(np.asarray(w, dtype=np.float32))[None].to(self.device)
            out.append(float(self.utmos(x, sr=SR)))
        return out

    # ------------------------------------------------------------------- score
    def score_dir(self, run_dir: str, T: int, limit: Optional[int] = None,
                  suffix: str = "", tag: Optional[str] = None) -> Dict:
        import jiwer
        with open(os.path.join(PROC_DIR, "eval_zs.json")) as fh:
            all_items = sorted(json.load(fh), key=lambda d: d["item"])
        # `tag` mirrors `sample.py synth --tag` (E3's per-level NFE schedules write
        # synth_<tag>/ instead of synth_T<T>/); T is still recorded in the summary
        sdir = os.path.join(run_dir, f"synth_{tag}" if tag else f"synth_T{T}")
        # `limit` mirrors `sample.py synth --items N`. When not given, the denominator is
        # whatever this dir was SYNTHESISED with (synth.json records the requested count),
        # never a hardcoded 400 — E1's extended-T dirs hold 200 items, and assuming 400
        # charges 200 phantom crashes. Fail loud if synth.json is absent.
        if limit is None:
            with open(os.path.join(sdir, "synth.json")) as fh:
                limit = int(json.load(fh)["items"])
        items = {d["item"]: d for d in all_items[:limit]}
        # iterate the CANONICAL item list, not the files present: a missing item must be
        # scored under the crash policy (task.md §10), never dropped from the denominator
        names = sorted(items)
        wavs, crashed = [], []
        for n in names:
            try:
                w, sr = sf.read(os.path.join(sdir, f"{n}.flac"), dtype="float32")
                assert sr == SR and len(w) > 0
            except Exception:
                w = np.zeros(SR // 10, np.float32)
                crashed.append(n)
            wavs.append(w)
        hyps = self.transcribe(wavs)
        prompts = [_read(os.path.join(PROC_DIR, "eval_audio", items[n]["prompt_id"] + ".flac"))
                   for n in names]
        e_gen, e_pr = self.embed(wavs), self.embed(prompts)
        sims = torch.nn.functional.cosine_similarity(e_gen, e_pr).tolist()
        moss = self.mos(wavs)

        rows = []
        for i, n in enumerate(names):
            ref = self.norm(items[n]["target_text"])
            hyp = self.norm(hyps[i])
            degen = is_degenerate(hyp, ref) or (n in crashed)
            wer = 1.0 if n in crashed else float(jiwer.wer(ref, hyp)) if ref else float("nan")
            rows.append({"item": n, "wer": wer, "sim": float("nan") if n in crashed else sims[i],
                         "utmos": None if (moss is None or n in crashed) else moss[i],
                         "degenerate": bool(degen), "crashed": n in crashed,
                         "n_words_ref": len(ref.split()), "n_words_hyp": len(hyp.split()),
                         "hyp": hyps[i], "gen_seconds": len(wavs[i]) / SR})
        wer_all = np.array([r["wer"] for r in rows], float)
        sim_all = np.array([r["sim"] for r in rows], float)
        ok = ~np.array([r["degenerate"] for r in rows])
        res = {"run": run_dir, "T": T, "n_items": len(rows), "sim_model": self.sim_model_name,
               "asr_model": self.asr_id,
               "utmos_available": self.utmos is not None,
               "wer_mean": float(np.nanmean(wer_all)),
               "wer_se": float(np.nanstd(wer_all, ddof=1) / np.sqrt(np.isfinite(wer_all).sum())),
               "wer_corpus": float(jiwer.wer([self.norm(items[n]["target_text"]) for n in names],
                                             [self.norm(h) or " " for h in hyps])),
               "sim_mean": float(np.nanmean(sim_all)),
               "sim_se": float(np.nanstd(sim_all, ddof=1) / np.sqrt(np.isfinite(sim_all).sum())),
               "degen_rate": float(np.mean([r["degenerate"] for r in rows])),
               "crash_rate": float(np.mean([r["crashed"] for r in rows])),
               "wer_mean_nondegen": float(np.nanmean(wer_all[ok])) if ok.any() else None,
               "sim_mean_nondegen": float(np.nanmean(sim_all[ok])) if ok.any() else None}
        if moss is not None:
            m = np.array([r["utmos"] if r["utmos"] is not None else np.nan for r in rows], float)
            res["utmos_mean"] = float(np.nanmean(m))
            res["utmos_se"] = float(np.nanstd(m, ddof=1) / np.sqrt(np.isfinite(m).sum()))
        with open(os.path.join(sdir, f"scores{suffix}.json"), "w") as fh:
            json.dump({"summary": res, "items": rows}, fh)
        print(f"[score] {run_dir} T={T} WER {res['wer_mean']*100:.1f}% SIM {res['sim_mean']:.3f} "
              f"degen {res['degen_rate']*100:.1f}%", flush=True)
        return res


def _read(path: str) -> np.ndarray:
    w, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"{path} sr={sr}"
    return w


# ------------------------------------------------------------------ gate G0(c)
def cmd_gt(a):
    """Eval harness on ground-truth audio: WER ≤ 5 %, median same-speaker SIM-o ≥ 0.50,
    median cross-speaker ≤ 0.25 (protocol §8 G0c)."""
    import jiwer
    sc = Scorer(a.device)
    with open(os.path.join(PROC_DIR, "eval_zs.json")) as fh:
        items = json.load(fh)[:a.items]
    tgt = [_read(os.path.join(PROC_DIR, "eval_audio", it["target_id"] + ".flac")) for it in items]
    prm = [_read(os.path.join(PROC_DIR, "eval_audio", it["prompt_id"] + ".flac")) for it in items]
    hyps = sc.transcribe(tgt)
    refs = [sc.norm(it["target_text"]) for it in items]
    hn = [sc.norm(h) for h in hyps]
    per_item = [float(jiwer.wer(r, h or " ")) for r, h in zip(refs, hn)]
    e_t, e_p = sc.embed(tgt), sc.embed(prm)
    same = torch.nn.functional.cosine_similarity(e_t, e_p).numpy()
    spk = np.array([it["speaker"] for it in items])
    rng = np.random.default_rng(0)
    cross = []
    for i in range(len(items)):
        cand = np.where(spk != spk[i])[0]
        j = cand[rng.integers(len(cand))]
        cross.append(float(torch.nn.functional.cosine_similarity(e_t[i:i + 1], e_p[j:j + 1])))
    mos = sc.mos(tgt)
    out = {"n_items": len(items), "sim_model": sc.sim_model_name,
           "wer_corpus": float(jiwer.wer(refs, [h or " " for h in hn])),
           "wer_mean_item": float(np.mean(per_item)),
           "sim_same_median": float(np.median(same)), "sim_cross_median": float(np.median(cross)),
           "utmos_gt_mean": float(np.mean(mos)) if mos else None,
           "pass_wer": bool(np.mean(per_item) <= 0.05),
           "pass_sim_same": bool(np.median(same) >= 0.50),
           "pass_sim_cross": bool(np.median(cross) <= 0.25)}
    out["passes"] = bool(out["pass_wer"] and out["pass_sim_same"] and out["pass_sim_cross"])
    os.makedirs(os.path.join(REPO, "artifacts"), exist_ok=True)
    with open(os.path.join(REPO, "artifacts", "g0c_groundtruth.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1), flush=True)


def cmd_score(a):
    jobs = json.load(open(a.jobs)) if a.jobs else [{"run": a.run, "T": a.T, "items": a.items}]
    sc = Scorer(a.device, asr_id=a.asr, force_sv_fallback=a.sv_fallback)
    for j in jobs:
        tag = j.get("tag")
        sdir = os.path.join(j["run"], f"synth_{tag}" if tag else f"synth_T{j['T']}")
        if os.path.exists(os.path.join(sdir, f"scores{a.suffix}.json")) and not a.force:
            print(f"[score] skip {sdir} (done)", flush=True)
            continue
        sc.score_dir(j["run"], j["T"], j.get("items"), a.suffix, tag)


# ---------------------------------------------------------------- runs.csv (§10)
def cmd_collect(a):
    import pandas as pd
    from model import config_by_id, load_grid
    grid = load_grid()
    clayer = {}
    cl_path = os.path.join(REPO, "artifacts", "c_layer.json")
    if os.path.exists(cl_path):
        clayer = {int(k): v["c_layer_ms"]
                  for k, v in json.load(open(cl_path))["measured"].items()}
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(REPO, "runs", "*"))):
        rj = os.path.join(run_dir, "run.json")
        if not os.path.exists(rj):
            continue
        run = json.load(open(rj))
        if run.get("status") != "completed":
            continue
        base = os.path.basename(run_dir)
        cfg_id, seed = base.rsplit("_", 1)
        for sdir in sorted(glob.glob(os.path.join(run_dir, "synth_T*"))):
            T = int(os.path.basename(sdir).split("T")[1])
            sf_ = os.path.join(sdir, "scores.json")
            if not os.path.exists(sf_):
                continue
            s = json.load(open(sf_))["summary"]
            synth = json.load(open(os.path.join(sdir, "synth.json")))
            d, w = run["depth"], run["width"]
            rows.append({
                "config": cfg_id, "seed": int(seed), "T": T, "nfe": 8 * T,
                "budget": config_by_id(cfg_id, grid)["budget"], "width": w, "depth": d,
                "heads": run["heads"], "n_nonembed": run["nonembed_params"],
                "n_total": run["total_params"], "lr": run["lr"],
                "val_loss": run.get("final_val_loss"),
                "wer": s["wer_mean"], "wer_se": s["wer_se"], "wer_corpus": s["wer_corpus"],
                "sim": s["sim_mean"], "sim_se": s["sim_se"],
                "utmos": s.get("utmos_mean"), "utmos_se": s.get("utmos_se"),
                "degen_rate": s["degen_rate"], "crash_rate": s["crash_rate"],
                "wer_nondegen": s["wer_mean_nondegen"], "sim_nondegen": s["sim_mean_nondegen"],
                "err_wer": s["wer_mean"], "err_sim": 1.0 - s["sim_mean"],
                "err_ut": (5.0 - s["utmos_mean"]) / 4.0 if s.get("utmos_mean") else None,
                "c_layer_ms": clayer.get(w),
                "latency_ms": (8 * T * d * clayer[w]) if w in clayer else None,
                "train_gpu_hours": run["gpu_hours"], "train_wall_s": run["wall_seconds"],
                "synth_gpu_hours": synth["gpu_hours"], "n_items": s["n_items"],
                "sim_model": s["sim_model"], "ckpt_step": synth["ckpt_step"]})
    df = pd.DataFrame(rows).sort_values(["budget", "config", "seed", "T"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"[collect] {len(df)} rows → {a.out}", flush=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--jobs")
    s.add_argument("--run")
    s.add_argument("--T", type=int)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--force", action="store_true")
    s.add_argument("--items", type=int, default=None,
                   help="score only the first N canonical items (diagnostic probes)")
    s.add_argument("--asr", default="openai/whisper-large-v3")
    s.add_argument("--sv-fallback", action="store_true")
    s.add_argument("--suffix", default="", help="write scores<suffix>.json (E5 variants)")
    s.set_defaults(fn=cmd_score)
    g = sub.add_parser("gt")
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--items", type=int, default=400)
    g.set_defaults(fn=cmd_gt)
    c = sub.add_parser("collect")
    c.add_argument("--out", default="artifacts/runs.csv")
    c.set_defaults(fn=cmd_collect)
    args = ap.parse_args()
    args.fn(args)
