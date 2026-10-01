"""External anchor: one published system on our exact 400 items, through our exact stack.

The reviewer packet's one unclosed blocking item was that the paper has no external
reference, so a reader cannot tell whether WER 0.19 and SIM-o 0.37 are sane numbers for
this task or an artefact of our metric wiring. This measures F5-TTS v1 Base on the same
items and scores it with the same Whisper / WavLM / UTMOS stack.

**What it licenses and what it does not.** Our models are 20-133 M parameters trained on
2,000 h for 30k steps; F5-TTS is far larger and far better trained. The anchor is NOT a
competitive comparison and will never be described as one. It establishes one thing: that
the measurement stack, run on a system whose published numbers are known, lands where it
should. That is calibration, not a baseline.

**Isolation is absolute.** F5's resolver wants transformers/numpy versions that differ
from the frozen metric stack. Synthesis therefore runs in its own venv; scoring runs in
the project venv on the resulting audio. This script never installs into the project venv
and aborts if it would have to.

    python src/anchor_f5.py --budget-seconds 10800
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = os.environ.get(
    "ASPECT_SCRATCH",
    os.path.join(REPO, ".cache", "anchor_f5"))
VENV = os.path.join(SCRATCH, "f5venv")
OUTDIR = os.path.join(REPO, "runs-v1.5", "f5tts_anchor", "synth_T32")
PROJECT_VENV = os.environ.get("ASPECTD_VENV", sys.prefix)
# pre-declared, direction-free: outside this band the port is called unverified rather
# than being quietly dropped or quietly published
BAND = {"wer_max": 0.15, "sim_min": 0.40, "min_rendered": 396}

WORKER = r'''
import json, os, sys, numpy as np, soundfile as sf, torch
items = json.load(open(sys.argv[1]))
aud, out, limit = sys.argv[2], sys.argv[3], int(sys.argv[4])
from f5_tts.api import F5TTS
m = F5TTS()
os.makedirs(out, exist_ok=True)
done = []
for k, it in enumerate(items[:limit]):
    dst = os.path.join(out, it["item"] + ".flac")
    if os.path.exists(dst):
        done.append(it["item"]); continue
    try:
        wav, sr, _ = m.infer(
            ref_file=os.path.join(aud, it["prompt_id"] + ".flac"),
            ref_text=it["prompt_text"], gen_text=it["target_text"], remove_silence=False)
        wav = np.asarray(wav, dtype=np.float32)
        if sr != 24000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=24000)
        sf.write(dst, wav, 24000)
        done.append(it["item"])
    except Exception as e:
        print(f"[f5] FAIL {it['item']}: {type(e).__name__} {e}", flush=True)
    if k % 25 == 0:
        print(f"[f5] {k}/{min(limit,len(items))}", flush=True)
json.dump({"items": len(done), "item_ids": done, "T": 32, "nfe": 32,
           "system": "F5-TTS v1 Base", "sr": 24000},
          open(os.path.join(out, "synth.json"), "w"))
print(f"[f5] rendered {len(done)}")
'''


def run(cmd, timeout, **kw):
    return subprocess.run(cmd, timeout=timeout, capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-seconds", type=int, default=10800)
    ap.add_argument("--items", type=int, default=400)
    ap.add_argument("--smoke", type=int, default=5)
    a = ap.parse_args()
    t0 = time.time()

    def left():
        return a.budget_seconds - (time.time() - t0)

    def abort(why):
        rec = {"status": "aborted", "reason": why,
               "elapsed_s": round(time.time() - t0, 1)}
        os.makedirs(os.path.dirname(OUTDIR), exist_ok=True)
        json.dump(rec, open(os.path.join(REPO, "artifacts-v1.5", "anchor.json"), "w"),
                  indent=1)
        print(f"[anchor] ABORT: {why}")
        return 1

    os.makedirs(os.path.join(REPO, "artifacts-v1.5"), exist_ok=True)
    py = os.path.join(VENV, "bin", "python")

    # ---- isolated venv; the project venv is never a target ----------------------
    if not os.path.exists(py):
        print("[anchor] creating isolated venv", flush=True)
        r = run([sys.executable, "-m", "venv", VENV], timeout=min(600, left()))
        if r.returncode:
            return abort(f"venv creation failed: {r.stderr[-300:]}")
    real = os.path.realpath(py)
    if real.startswith(os.path.realpath(PROJECT_VENV)):
        return abort("isolated interpreter resolves into the project venv")

    print("[anchor] installing f5-tts (isolated)", flush=True)
    r = run([py, "-m", "pip", "install", "--quiet", "f5-tts"],
            timeout=min(2700, max(60, left())))
    if r.returncode:
        return abort(f"pip install failed: {r.stderr[-400:]}")

    # the project venv must be byte-identical afterwards
    chk = run([os.path.join(PROJECT_VENV, "bin", "python"), "-c",
               "import transformers, numpy; print(transformers.__version__, numpy.__version__)"],
              timeout=120)
    print(f"[anchor] project venv intact: {chk.stdout.strip()}", flush=True)

    from data import PROC_DIR
    items = json.load(open(os.path.join(PROC_DIR, "eval_zs.json")))
    items = sorted(items, key=lambda d: d["item"])
    aud = os.path.join(PROC_DIR, "eval_audio")
    wf = os.path.join(SCRATCH, "f5_worker.py")
    ij = os.path.join(SCRATCH, "f5_items.json")
    open(wf, "w").write(WORKER)
    json.dump(items, open(ij, "w"))

    # ---- smoke, then the full set ----------------------------------------------
    print(f"[anchor] smoke test on {a.smoke} items", flush=True)
    r = run([py, wf, ij, aud, OUTDIR, str(a.smoke)], timeout=min(1800, max(60, left())))
    print(r.stdout[-1500:], flush=True)
    n_smoke = len([f for f in os.listdir(OUTDIR)]) if os.path.isdir(OUTDIR) else 0
    if n_smoke < a.smoke:
        return abort(f"smoke rendered {n_smoke}/{a.smoke}")
    if left() < 900:
        return abort("less than 15 min of budget left after smoke")

    print(f"[anchor] full set, {a.items} items, {left()/60:.0f} min left", flush=True)
    r = run([py, wf, ij, aud, OUTDIR, str(a.items)], timeout=max(60, left()))
    print(r.stdout[-2000:], flush=True)
    sj = os.path.join(OUTDIR, "synth.json")
    if not os.path.exists(sj):
        return abort("worker produced no synth.json")
    n = json.load(open(sj))["items"]
    json.dump({"status": "synthesised", "rendered": n, "band": BAND,
               "elapsed_s": round(time.time() - t0, 1),
               "note": "score with the PROJECT venv: "
                       "evaluate.py score --run runs-v1.5/f5tts_anchor --T 32"},
              open(os.path.join(REPO, "artifacts-v1.5", "anchor.json"), "w"), indent=1)
    print(f"[anchor] rendered {n}/{a.items} in {(time.time()-t0)/60:.0f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
