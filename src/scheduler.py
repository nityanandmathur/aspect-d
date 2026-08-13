"""Keep every GPU busy with the next pending v1.5 job, without a human in the loop.

The v1.5 queue was written as a barrier: wait for all training, then run stages in order.
That serialised independent work, and every time a config finished its GPUs sat idle until
someone noticed and hand-filled them. This replaces the barrier with a work-stealing loop
-- a GPU that goes free takes the next pending job, whatever kind it is.

Jobs are declared, not scripted, so the loop can be restarted at any time and will simply
re-derive what is still outstanding. Dependencies are expressed as readiness predicates:
a 180k sweep is ready only once its training has reached 180000 steps, and a CFG sweep only
once its checkpoint exists.

    python src/scheduler.py --plan       # what is outstanding, and why
    python src/scheduler.py --run        # daemon: fill free GPUs until nothing is left
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
PY = sys.executable
LOGS = os.path.join(REPO, "logs-v1.5")
CLAIMS = os.path.join(REPO, ".claims-sched")
TS = (1, 2, 4, 8, 16, 32, 64)
CONFIGS, SEEDS = ("C1", "C3", "C5"), (0, 1, 2)


def run_dir(name):
    for root in ("runs-v1.4", "runs-v1.3", "runs-v1.1", "runs"):
        p = os.path.join(REPO, root, name)
        if os.path.isdir(p):
            return p
    return None


def trained(name, steps):
    d = run_dir(name)
    if not d:
        return False
    rj = os.path.join(d, "run.json")
    try:
        return json.load(open(rj)).get("final_step", 0) >= steps
    except (OSError, json.JSONDecodeError):
        return False


def swept(name):
    d = run_dir(name)
    return bool(d) and all(
        os.path.exists(os.path.join(d, f"synth_T{T}", "scores.json")) for T in TS)


def jobs():
    """(key, priority, ready, cmd_builder). Lower priority number runs first."""
    out = []
    for c in CONFIGS:
        for s in SEEDS:
            n180, ncfg = f"{c}_{s}_180k", f"{c}_{s}_cfg"
            # 1. sweep a finished 180k run -- this is the gating measurement
            if trained(n180, 180000) and not swept(n180):
                out.append((f"sweep:{n180}", 1, True, ("sweep", n180)))
            # 2. train the remaining CFG checkpoints
            if not trained(ncfg, 30000) and not os.path.exists(
                    os.path.join(REPO, "runs-v1.4", ncfg, "run.json")):
                out.append((f"train:{ncfg}", 2, True, ("cfg_train", c, s)))
            # 3. sweep a finished CFG checkpoint
            if trained(ncfg, 30000) and not swept(ncfg):
                out.append((f"sweep:{ncfg}", 3, True, ("sweep", ncfg)))
    return sorted(out, key=lambda j: j[1])


def free_gpus():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return [ln.split(",")[0].strip() for ln in r.stdout.strip().splitlines()
                if ln.strip() and int(ln.split(",")[1]) < 10 and int(ln.split(",")[2]) < 2000]
    except Exception:
        return []


def claim(key):
    os.makedirs(CLAIMS, exist_ok=True)
    p = os.path.join(CLAIMS, key.replace(":", "__"))
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


ACTIVE = {}          # gpu -> Popen of the child this loop started


def launch(spec, gpu):
    kind = spec[0]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS="24", MKL_NUM_THREADS="24")
    if kind == "cfg_train":
        _, c, s = spec
        out = os.path.join(REPO, "runs-v1.4", f"{c}_{s}_cfg")
        cmd = [PY, os.path.join(SRC, "train.py"), "--config", c, "--seed", str(s),
               "--lr", "0.004", "--steps", "30000", "--cond-dropout", "0.10",
               "--out", out, "--device", "cuda:0"]
        log = os.path.join(LOGS, f"cfg_{c}_{s}.log")
    else:
        name = spec[1]
        d = run_dir(name)
        # one process per T, then a single scorer for the whole run: a cold Scorer costs
        # ~500 s against ~70 s warm, so scoring seven dirs in one process matters
        inner = " && ".join(
            f'{PY} {SRC}/sample.py synth --run {d} --T {T} --items 400 --device cuda:0'
            for T in reversed(TS))
        jf = os.path.join(LOGS, f"sched_{name}.json")
        json.dump([{"run": d, "T": T, "items": 400} for T in TS], open(jf, "w"))
        inner += f' && {PY} {SRC}/evaluate.py score --jobs {jf} --device cuda:0'
        cmd = ["bash", "-c", inner]
        log = os.path.join(LOGS, f"sched_{name}.log")
    os.makedirs(LOGS, exist_ok=True)
    fh = open(log, "a")
    ACTIVE[str(gpu)] = subprocess.Popen(cmd, env=env, cwd=SRC, stdout=fh, stderr=fh,
                                        start_new_session=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--poll", type=int, default=120)
    a = ap.parse_args()

    if a.plan:
        js = jobs()
        print(f"{len(js)} job(s) outstanding, free GPUs: {free_gpus() or 'none'}")
        for k, p, _, _ in js:
            print(f"  [{p}] {k}{'  (claimed)' if os.path.exists(os.path.join(CLAIMS, k.replace(':','__'))) else ''}")
        return

    # A GPU counts as available only if this loop has no live child on it. Utilisation
    # alone is not enough: a sweep chains one sample.py per T value, so the GPU reads 0%
    # in the gaps between them and the loop handed the same GPU five jobs in a row.
    while True:
        for g, p in list(ACTIVE.items()):
            if p.poll() is not None:
                del ACTIVE[g]
        pending = jobs()
        claimed = {j[0] for j in pending
                   if os.path.exists(os.path.join(CLAIMS, j[0].replace(":", "__")))}
        ready = [j for j in pending if j[0] not in claimed]
        # Exit only when there is nothing left AND nothing of ours is still running.
        # Previously the loop exited whenever the ready list was momentarily empty, so a
        # job that became available later -- a sweep waiting on its training -- was never
        # picked up.
        if not ready and not ACTIVE and not _training_alive():
            print("[sched] all work complete", flush=True)
            return
        for gpu in free_gpus():
            if not ready:
                break
            if gpu in ACTIVE:
                continue
            key, _, _, spec = ready.pop(0)
            if claim(key):
                print(f"[sched] gpu{gpu} <- {key}", flush=True)
                launch(spec, gpu)
                time.sleep(25)
        time.sleep(a.poll)


def _training_alive() -> bool:
    """Work that is not ready yet but will be: a running training implies a future sweep."""
    r = subprocess.run(["pgrep", "-c", "-f", "train[.]py"], capture_output=True, text=True)
    return int(r.stdout.strip() or 0) > 0


if __name__ == "__main__":
    main()
