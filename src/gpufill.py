"""Keep every GPU busy: watch for idle devices and dispatch the next pending job.

Work is claimed with an atomic lock file (O_CREAT|O_EXCL) keyed by the job identity, so
two dispatchers — or a dispatcher and a hand-launched job — can never write the same run
directory (the failure that produced two trainers for B4_2). Job priority:

    1. training runs with no run.json at all (never started)
    2. missing (run, T) synthesis for a completed run
    3. missing (run, T) scoring for a synthesised (run, T)

    python src/gpufill.py [--poll 60]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import subprocess
import time
from typing import Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.environ.get("ASPECTD_PY", sys.executable)
LOCKS = os.path.join(REPO, "logs", "locks")
JOBLOG = os.path.join(REPO, "logs", "jobs")
T_GRID = (16, 1, 8, 4, 2)


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
    """GPU index → number of compute processes on it (nvidia-smi is the ground truth)."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    uuids = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout
    idx = {}
    for line in uuids.strip().split("\n"):
        if "," in line:
            i, u = line.split(",")
            idx[u.strip()] = int(i)
    busy = {i: 0 for i in idx.values()}
    for line in out.strip().split("\n"):
        if "," in line:
            u = line.split(",")[0].strip()
            if u in idx:
                busy[idx[u]] += 1
    return busy


def all_runs(grid: Dict, seeds=(0, 1, 2)) -> List[str]:
    return [f"{c['id']}_{s}" for c in grid["configs"]
            if c["budget"] in ("A", "B", "C") for s in seeds]


def next_job(grid: Dict, lr: float) -> Optional[Tuple[str, List[str], str]]:
    # 1) never-started training runs
    for name in all_runs(grid):
        d = os.path.join(REPO, "runs", name)
        if os.path.exists(os.path.join(d, "run.json")):
            continue
        if not claim(f"train_{name}"):
            continue
        cfg, seed = name.rsplit("_", 1)
        return (f"train:{name}",
                [PY, "train.py", "--config", cfg, "--seed", seed, "--lr", str(lr),
                 "--out", d, "--device", "cuda:0"],
                os.path.join(JOBLOG, f"{name}.log"))
    # 2) missing synthesis for completed runs
    for d in sorted(glob.glob(os.path.join(REPO, "runs", "[ABC]*_[012]"))):
        rj = os.path.join(d, "run.json")
        if not os.path.exists(rj) or json.load(open(rj)).get("status") != "completed":
            continue
        for T in T_GRID:
            if os.path.exists(os.path.join(d, f"synth_T{T}", "synth.json")):
                continue
            if not claim(f"synth_{os.path.basename(d)}_T{T}"):
                continue
            return (f"synth:{os.path.basename(d)}_T{T}",
                    [PY, "sample.py", "synth", "--run", d, "--T", str(T)],
                    os.path.join(JOBLOG, f"synth_{os.path.basename(d)}_T{T}.log"))
    # 3) missing scoring for synthesised (run, T)
    for d in sorted(glob.glob(os.path.join(REPO, "runs", "[ABC]*_[012]"))):
        for s in sorted(glob.glob(os.path.join(d, "synth_T*"))):
            if not os.path.exists(os.path.join(s, "synth.json")):
                continue
            if os.path.exists(os.path.join(s, "scores.json")):
                continue
            T = int(os.path.basename(s).split("T")[1])
            key = f"score_{os.path.basename(d)}_T{T}"
            if not claim(key):
                continue
            jf = os.path.join(REPO, "logs", "jobs", key + ".json")
            json.dump([{"run": d, "T": T}], open(jf, "w"))
            return (f"score:{os.path.basename(d)}_T{T}",
                    [PY, "evaluate.py", "score", "--jobs", jf],
                    os.path.join(JOBLOG, key + ".log"))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poll", type=int, default=45)
    ap.add_argument("--max-per-gpu", type=int, default=1)
    a = ap.parse_args()
    grid = json.load(open(os.path.join(REPO, "configs", "grid.json")))
    lr = json.load(open(os.path.join(REPO, "state.json")))["chosen_lr"]
    os.makedirs(JOBLOG, exist_ok=True)
    live: Dict[int, subprocess.Popen] = {}
    idle_rounds = 0
    while True:
        for g in list(live):
            if live[g].poll() is not None:
                print(f"[gpufill] GPU{g} job finished rc={live[g].returncode}", flush=True)
                del live[g]
        busy = gpu_busy()
        placed = False
        for g in sorted(busy):
            if busy[g] >= a.max_per_gpu or g in live:
                continue
            job = next_job(grid, lr)
            if job is None:
                break
            label, cmd, logf = job
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(g)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            lf = open(logf, "a")
            lf.write(f"\n===== {label} on GPU{g} @ {time.strftime('%H:%M:%S')} (gpufill) =====\n")
            lf.flush()
            live[g] = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                                       cwd=os.path.join(REPO, "src"))
            print(f"[gpufill] GPU{g} ← {label}", flush=True)
            placed = True
        idle_rounds = 0 if placed else idle_rounds + 1
        if idle_rounds % 20 == 1:
            print(f"[gpufill] nothing pending; {len(live)} dispatched jobs live", flush=True)
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
