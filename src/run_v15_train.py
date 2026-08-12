"""v1.5 training: a third training-compute point, to answer the question X2 opened.

X2 measured the floor-referenced asymmetry at 30k and at 90k steps and found the
ratio falls from 1.86x to 1.36x. Two points cannot distinguish "converging to some
value above 1" from "converging to 1", and the second reading would make the paper's
finding an artefact of under-training. That is the first question a referee asks, so
we answer it before they do.

This trains C1/C3/C5 x seeds {0,1,2} to 180k steps -- the same three configurations
and the same three seeds as the 90k point, so the trend is measured on a matched
design rather than on whatever happened to be available. C3 seed 0 already exists at
180k and is skipped.

One run per GPU; each writes checkpoints every 1000 steps, so a killed worker resumes
rather than restarts.

    python src/run_v15_train.py --plan
    python src/run_v15_train.py --launch
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
OUTROOT = os.path.join(REPO, "runs-v1.4")
LOGS = os.path.join(REPO, "logs-v1.5")
CLAIMS = os.path.join(REPO, ".claims-v1.5")
STEPS, LR = 180_000, 0.004
CONFIGS, SEEDS = ("C1", "C3", "C5"), (0, 1, 2)


def existing(cfg: str, seed: int) -> str | None:
    """The 180k run may already live under an earlier root; do not retrain it."""
    for root in ("runs-v1.3", "runs-v1.1", "runs-v1.4"):
        p = os.path.join(REPO, root, f"{cfg}_{seed}_180k")
        rj = os.path.join(p, "run.json")
        if os.path.exists(rj):
            try:
                if json.load(open(rj)).get("final_step", 0) >= STEPS:
                    return p
            except (json.JSONDecodeError, OSError):
                pass
    return None


def jobs():
    return [(c, s) for c in CONFIGS for s in SEEDS if existing(c, s) is None]


def claim(key: str) -> bool:
    os.makedirs(CLAIMS, exist_ok=True)
    try:
        fd = os.open(os.path.join(CLAIMS, key), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def worker(gpu: str):
    os.makedirs(LOGS, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    with open(os.path.join(LOGS, f"train{gpu}.log"), "a", buffering=1) as log:
        for cfg, seed in jobs():
            key = f"{cfg}_{seed}_180k"
            if not claim(key):
                continue
            out = os.path.join(OUTROOT, key)
            log.write(f"=== {key} on gpu{gpu} ===\n")
            t0 = time.time()
            p = subprocess.run(
                [sys.executable, os.path.join(SRC, "train.py"),
                 "--config", cfg, "--seed", str(seed), "--lr", str(LR),
                 "--steps", str(STEPS), "--out", out, "--device", "cuda:0"],
                env=env, cwd=SRC, capture_output=True, text=True)
            log.write(f"[train] {key} rc={p.returncode} {(time.time()-t0)/3600:.1f} h\n")
            if p.returncode != 0:
                log.write(p.stdout[-3000:] + "\n" + p.stderr[-3000:] + "\n")
                try:
                    os.unlink(os.path.join(CLAIMS, key))   # let another worker resume
                except OSError:
                    pass
        log.write("[train] drained\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--launch", action="store_true")
    a = ap.parse_args()

    if a.plan:
        have = [(c, s) for c in CONFIGS for s in SEEDS if existing(c, s)]
        todo = jobs()
        print(f"180k target: {len(CONFIGS)*len(SEEDS)} runs "
              f"({len(have)} already trained, {len(todo)} to train)")
        for c, s in have:
            print(f"  have {c}_{s}_180k -> {existing(c, s)}")
        for c, s in todo:
            print(f"  train {c}_{s}_180k")
        return
    if a.launch:
        os.makedirs(LOGS, exist_ok=True)
        os.makedirs(OUTROOT, exist_ok=True)
        n = min(8, len(jobs()))
        for g in range(n):
            subprocess.Popen([sys.executable, os.path.abspath(__file__), "--gpu", str(g)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        print(f"launched {n} training workers")
        return
    worker(a.gpu)


if __name__ == "__main__":
    main()
