"""v1.1 evaluation supervisor — drain E4/E6 synth+score onto whatever GPU is idle.

Training (E4 budget-D, E6 90k) and evaluation overlap: as each training run
finishes it frees a GPU, and its own syntheses become runnable. This polls
`nvidia-smi` for genuinely idle GPUs, rebuilds the pending job lists (so runs
that finish mid-flight are picked up automatically), and launches one job per
idle GPU. Work is claimed with an atomic O_CREAT|O_EXCL lock, so this is safe to
run alongside the `run_v11.py` training dispatchers and safe to restart.

Synthesis is dispatched before scoring for a run, since scoring needs the audio.
Exits once no training remains and no jobs are pending.

    python src/supervise_v11.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from typing import Dict, List

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
LOGS = os.path.join(REPO, "logs-v1.1")
LOCKS = os.path.join(LOGS, "claims")
PY = "/home/ubuntu/venv/bin/python"


def claim(name: str) -> bool:
    os.makedirs(LOCKS, exist_ok=True)
    try:
        fd = os.open(os.path.join(LOCKS, name), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(time.time()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def gpu_busy() -> Dict[int, int]:
    """GPU index → number of compute processes on it (nvidia-smi is ground truth)."""
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout
    uuids = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout
    idx = {}
    for line in uuids.strip().split("\n"):
        if "," in line:
            i, u = line.split(",")
            idx[u.strip()] = int(i)
    busy = {i: 0 for i in idx.values()}
    for line in apps.strip().split("\n"):
        if "," in line:
            u = line.split(",")[0].strip()
            if u in idx:
                busy[idx[u]] += 1
    return busy


def training_active() -> bool:
    out = subprocess.run(["pgrep", "-f", "train.py --config"], capture_output=True,
                         text=True).stdout.strip()
    return bool(out)


def pending() -> List[Dict]:
    """Rebuild the job lists, then return synth jobs first, then score jobs."""
    jobs = []
    for which in ("e6", "e4"):        # E6 is small and unblocks H-E5 sooner
        subprocess.run([PY, os.path.join(SRC, "build_eval_jobs.py"), "--which", which],
                       capture_output=True, cwd=SRC)
        for kind in ("synth_jobs", "eval_score_jobs"):
            p = os.path.join(LOGS, f"{which}_{kind}.json")
            if os.path.exists(p):
                for j in json.load(open(p)):
                    j["key"] = f"{which}_{j['key']}"
                    jobs.append(j)
    order = {"synth": 0, "score": 1}
    return sorted(jobs, key=lambda j: order.get(j["key"].split("_")[1], 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--reserve", type=int, default=0,
                    help="leave this many idle GPUs untouched for training headroom")
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    live: List = []                     # (Popen, gpu_index)
    idle_rounds = 0
    while True:
        live = [(p, g) for p, g in live if p.poll() is None]
        jobs = pending()
        if not jobs and not training_active() and not live:
            idle_rounds += 1
            if idle_rounds >= 2:
                print("[sup] nothing pending, no training, no live jobs — done", flush=True)
                return
        else:
            idle_rounds = 0
        busy = gpu_busy()
        # a GPU we just launched onto is not yet visible to nvidia-smi (the child
        # spends ~30 s importing torch and loading weights before it allocates), so
        # exclude our own in-flight assignments or one idle GPU collects every job
        mine = {g for _, g in live}
        free = [g for g, n in sorted(busy.items()) if n == 0 and g not in mine]
        free = free[a.reserve:] if a.reserve else free
        for g in free:
            job = None
            while jobs:                       # take the first job nobody else has claimed
                cand = jobs.pop(0)
                if claim(cand["key"] + ".lock"):
                    job = cand
                    break
            if job is None:
                break
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(g)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            lf = open(os.path.join(LOGS, f"sup_{job['key']}.log"), "a")
            live.append((subprocess.Popen(job["cmd"], stdout=lf, stderr=subprocess.STDOUT,
                                          env=env, cwd=SRC), g))
            print(f"[sup] gpu{g} <- {job['key']} ({len(jobs)} still pending)", flush=True)
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
