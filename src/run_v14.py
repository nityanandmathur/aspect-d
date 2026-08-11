"""v1.4 GPU queue: rebuild the response surface without the subset confound.

The published extended-T fit (`artifacts-v1.1/e1_extended.json`) covers 7 of 15
configurations at 200 of 400 items. Any statement about how tau moves when the T
range grows is therefore confounded with *which configs* and *which items* were
extended. X1 removes both by extending all 45 v1.0 runs to T in {32, 64} at the
full 400 items; X2 rebuilds the 3x-training-compute row on 9 runs x 3 seeds with
a complete sweep rather than a single seed; X7 probes T = 128 for an upper end.

Workers claim jobs with O_CREAT|O_EXCL, so N workers over N GPUs never collide
and a killed worker leaves its claim behind rather than corrupting a directory.
Synthesis publishes `synth.json` atomically (src/sample.py), so a scorer can
never observe a completion marker beside a half-written directory.

    python src/run_v14.py --plan            # print the job list, run nothing
    python src/run_v14.py --gpu 0           # one worker
    python src/run_v14.py --launch          # 8 workers, one per GPU
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
CLAIMS = os.path.join(REPO, ".claims-v1.4")
LOGS = os.path.join(REPO, "logs-v1.4")
ITEMS = 400

CONFIGS = [f"{b}{i}" for b in "ABC" for i in range(1, 6)]
RUNS_V10 = [f"{c}_{s}" for c in CONFIGS for s in (0, 1, 2)]
RUNS_90K = [f"C{i}_{s}_90k" for i in (1, 3, 5) for s in (0, 1, 2)]


def jobs():
    """(root, run, T, why) — ordered longest-first so the tail does not straggle."""
    out = []
    for T in (64, 32):                                   # X1
        for r in RUNS_V10:
            out.append(("runs", r, T, "X1"))
    for T in (64, 32, 16, 8, 4, 2, 1):                   # X2
        for r in RUNS_90K:
            root = "runs-v1.1" if os.path.isdir(os.path.join(REPO, "runs-v1.1", r)) \
                else "runs-v1.3"
            out.append((root, r, T, "X2"))
    for r in ("C5_0", "C5_1", "C5_2"):                   # X7
        out.append(("runs", r, 128, "X7"))
    return out


def done(root: str, run: str, T: int) -> bool:
    p = os.path.join(REPO, root, run, f"synth_T{T}", "synth.json")
    s = os.path.join(REPO, root, run, f"synth_T{T}", "scores.json")
    if not (os.path.exists(p) and os.path.exists(s)):
        return False
    try:
        return json.load(open(p)).get("items", 0) >= ITEMS
    except (json.JSONDecodeError, OSError):
        return False


def claim(key: str) -> bool:
    os.makedirs(CLAIMS, exist_ok=True)
    try:
        fd = os.open(os.path.join(CLAIMS, key), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def synthed(root: str, run: str, T: int) -> bool:
    p = os.path.join(REPO, root, run, f"synth_T{T}", "synth.json")
    try:
        return json.load(open(p)).get("items", 0) >= ITEMS
    except (json.JSONDecodeError, OSError):
        return False


def worker_synth(gpu: str):
    """Synthesis only. Cheap per job (~60-90 s) and embarrassingly parallel."""
    os.makedirs(LOGS, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    with open(os.path.join(LOGS, f"synth{gpu}.log"), "a", buffering=1) as log:
        for root, run, T, why in jobs():
            key = f"synth__{root}__{run}__T{T}"
            if synthed(root, run, T) or not claim(key):
                continue
            rd = os.path.join(REPO, root, run)
            if not os.path.isdir(rd):
                log.write(f"[skip] {run} missing\n")
                continue
            t0 = time.time()
            p = subprocess.run(
                [sys.executable, os.path.join(SRC, "sample.py"), "synth",
                 "--run", rd, "--T", str(T), "--items", str(ITEMS),
                 "--device", "cuda:0"],
                env=env, cwd=SRC, capture_output=True, text=True)
            log.write(f"[synth] {why} {run} T={T} rc={p.returncode} "
                      f"{time.time()-t0:.0f}s\n")
            if p.returncode != 0:
                log.write(p.stdout[-2000:] + "\n" + p.stderr[-2000:] + "\n")
                try:
                    os.unlink(os.path.join(CLAIMS, key))
                except OSError:
                    pass
        log.write("[synth] drained\n")


def worker_score(gpu: str):
    """One Scorer per GPU for a whole shard: Whisper-large-v3, WavLM-SV and UTMOS
    load once instead of once per (run, T), which was the actual bottleneck --
    synthesis is ~70 s while a cold-start score was ~500 s, almost all of it load."""
    os.makedirs(LOGS, exist_ok=True)
    g = int(gpu)
    todo = [j for j in jobs() if synthed(j[0], j[1], j[2]) and not done(j[0], j[1], j[2])]
    shard = [j for i, j in enumerate(todo) if i % 8 == g]
    if not shard:
        return
    spec = [{"run": os.path.join(REPO, r, run), "T": T, "items": ITEMS}
            for r, run, T, _ in shard]
    jf = os.path.join(LOGS, f"score{gpu}.json")
    with open(jf, "w") as fh:
        json.dump(spec, fh)
    # Eight scorers each defaulting to one thread per core drove the load average to
    # ~700 on 192 cores and made forward progress stop: the work is CPU-side audio
    # handling, so the processes were thrashing rather than scoring. Give each an
    # equal, non-overlapping slice.
    per = max(2, (os.cpu_count() or 16) // 8)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
               OMP_NUM_THREADS=str(per), MKL_NUM_THREADS=str(per),
               OPENBLAS_NUM_THREADS=str(per), NUMEXPR_NUM_THREADS=str(per),
               TORCH_NUM_THREADS=str(per))
    with open(os.path.join(LOGS, f"score{gpu}.log"), "a", buffering=1) as log:
        log.write(f"=== shard of {len(spec)} jobs on gpu{gpu} ===\n")
        t0 = time.time()
        p = subprocess.run(
            [sys.executable, os.path.join(SRC, "evaluate.py"), "score",
             "--jobs", jf, "--device", "cuda:0"],
            env=env, cwd=SRC, capture_output=True, text=True)
        log.write(p.stdout[-8000:] + "\n" + p.stderr[-4000:] + "\n")
        log.write(f"[score] shard rc={p.returncode} {time.time()-t0:.0f}s\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu")
    ap.add_argument("--mode", choices=["synth", "score"], default="synth")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()

    if a.plan or a.status:
        js = jobs()
        sy = [j for j in js if synthed(j[0], j[1], j[2])]
        pend = [j for j in js if not done(j[0], j[1], j[2])]
        by = {}
        for _, _, _, w in pend:
            by[w] = by.get(w, 0) + 1
        print(f"{len(js)} jobs: {len(sy)} synthesised, {len(js)-len(pend)} scored, "
              f"{len(pend)} pending")
        for k in sorted(by):
            print(f"  {k}: {by[k]} pending")
        return
    if a.launch:
        os.makedirs(LOGS, exist_ok=True)
        for g in range(8):
            subprocess.Popen([sys.executable, os.path.abspath(__file__),
                              "--gpu", str(g), "--mode", a.mode],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        print(f"launched 8 {a.mode} workers")
        return
    (worker_synth if a.mode == "synth" else worker_score)(a.gpu)


if __name__ == "__main__":
    main()
