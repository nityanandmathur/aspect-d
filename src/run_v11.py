"""v1.1 parallel job dispatcher — pins one job per GPU, resumable, append-only progress.

Used by E1/E3/E5/E7. Jobs are shell argv lists with a stable `key`; a completed key
is recorded in `logs-v1.1/done_<name>.txt` so a killed run resumes without redoing
work (task-v1.md §0.8: assume you can be killed at any time).

    python src/run_v11.py --jobs jobs.json --name e1 --gpus 4,5,6,7
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from typing import Dict, List

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO, "logs-v1.1")


def load_done(name: str) -> set:
    p = os.path.join(LOGS, f"done_{name}.txt")
    return set(open(p).read().split()) if os.path.exists(p) else set()


def mark_done(name: str, key: str, lock: threading.Lock) -> None:
    with lock:
        with open(os.path.join(LOGS, f"done_{name}.txt"), "a") as fh:
            fh.write(key + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--per-gpu", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(LOGS, exist_ok=True)
    jobs: List[Dict] = json.load(open(a.jobs))
    done = load_done(a.name)
    pending = [j for j in jobs if j["key"] not in done]
    slots = [g for g in a.gpus.split(",") for _ in range(a.per_gpu)]
    print(f"[{a.name}] {len(pending)} pending of {len(jobs)} on slots {slots}", flush=True)

    lock = threading.Lock()
    free = list(slots)
    t0 = time.time()
    counter = {"ok": 0, "fail": 0}

    def worker(job: Dict, gpu: str):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        lf = open(os.path.join(LOGS, f"{a.name}_{job['key']}.log"), "a")
        rc = subprocess.call(job["cmd"], stdout=lf, stderr=subprocess.STDOUT, env=env,
                             cwd=os.path.join(REPO, "src"))
        lf.close()
        with lock:
            if rc == 0:
                counter["ok"] += 1
            else:
                counter["fail"] += 1
            free.append(gpu)
        if rc == 0:
            mark_done(a.name, job["key"], lock)
        print(f"[{a.name}] {job['key']} rc={rc} gpu={gpu} "
              f"({counter['ok']}ok/{counter['fail']}fail of {len(pending)}) "
              f"{(time.time()-t0)/60:.1f}min", flush=True)

    threads: List[threading.Thread] = []
    queue = list(pending)
    while queue or any(t.is_alive() for t in threads):
        threads = [t for t in threads if t.is_alive()]
        with lock:
            g = free.pop(0) if (free and queue) else None
        if g is None:
            time.sleep(2)
            continue
        job = queue.pop(0)
        t = threading.Thread(target=worker, args=(job, g), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.5)
    print(f"[{a.name}] DONE ok={counter['ok']} fail={counter['fail']} "
          f"wall={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
