"""Report anything that is wasting GPU time, or about to.

A non-zero exit code catches a crash. It does not catch the three ways this project
actually loses compute: a run that diverges to NaN and exits cleanly, a run whose
process is alive but has stopped advancing, and eight idle GPUs with work still
queued because a waiter died. Each is checked here.

Prints one line per problem and exits non-zero if there are any, so it can drive a
monitor directly.

    python src/health.py            # human-readable
    python src/health.py --quiet    # only problems, for a watch loop
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STALL_S = 1800          # a live trainer that has not logged in 30 min is stalled
MIN_FREE_GB = 40        # ~2 GB per checkpoint, and the CFG stage writes nine
EXPECTED_180K = 9       # C1/C3/C5 x 3 seeds, one of which pre-exists elsewhere


def procs(pattern: str) -> int:
    return len(subprocess.run(["pgrep", "-c", "-f", pattern],
                              capture_output=True, text=True).stdout.split() or ["0"]) \
        and int(subprocess.run(["pgrep", "-c", "-f", pattern],
                               capture_output=True, text=True).stdout.strip() or 0)


def check() -> list:
    bad, now = [], time.time()

    free_gb = shutil.disk_usage(REPO).free / 2**30
    if free_gb < MIN_FREE_GB:
        bad.append(f"DISK  only {free_gb:.0f} GB free (< {MIN_FREE_GB}); "
                   f"a checkpoint is ~2 GB and nine more are queued")

    trainers = procs("train[.]py")
    live_runs = 0
    for d in sorted(glob.glob(os.path.join(REPO, "runs-v1.4", "*"))):
        rj, lg = os.path.join(d, "run.json"), os.path.join(d, "train_log.jsonl")
        if not os.path.exists(rj):
            continue
        try:
            r = json.load(open(rj))
        except json.JSONDecodeError:
            bad.append(f"CORRUPT  {os.path.basename(d)}/run.json is unreadable")
            continue
        st = r.get("status")
        name = os.path.basename(d)
        if st == "nan":
            bad.append(f"DIVERGED  {name} hit NaN at step {r.get('final_step')} "
                       f"-- its GPU-hours are spent and the run is unusable")
        elif st == "running":
            live_runs += 1
            if os.path.exists(lg):
                age = now - os.path.getmtime(lg)
                if age > STALL_S:
                    bad.append(f"STALLED  {name} is marked running but has not logged "
                               f"for {age/60:.0f} min -- holding a GPU and producing nothing")

    # a run marked running with no trainer process = died without updating run.json
    if live_runs and trainers == 0:
        bad.append(f"ORPHANED  {live_runs} run(s) marked running but no train.py process "
                   f"exists -- they died without recording it")

    # Per-GPU idleness. The earlier check only fired when *no* trainer existed at all,
    # so three GPUs freed by a finished config sat at 0% for hours beside five busy ones
    # and nothing reported it. Ask the GPUs directly instead of inferring from processes.
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        idle = [ln.split(",")[0].strip() for ln in smi.stdout.strip().splitlines()
                if ln.strip() and int(ln.split(",")[1]) < 5 and int(ln.split(",")[2]) < 2000]
    except Exception:
        idle = []
    if idle:
        q = os.path.join(REPO, "logs-v1.5", "v15.log")
        finished = os.path.exists(q) and "stage 7 rc=" in open(q).read()
        if not finished:
            bad.append(f"IDLE-GPU  {len(idle)} GPU(s) at 0% ({','.join(idle)}) while the "
                       f"v1.5 queue is unfinished -- capacity is being wasted")

    # idle GPUs with work outstanding is the expensive failure
    q = os.path.join(REPO, "logs-v1.5", "v15.log")
    waiter = procs("run_v15[.]py")
    if trainers == 0 and waiter == 0:
        done7 = os.path.exists(q) and "stage 7 rc=" in open(q).read()
        if not done7:
            bad.append("IDLE  no trainers and no v1.5 waiter, but the queue has not "
                       "reached stage 7 -- GPUs are idle with work outstanding")

    if os.path.exists(q):
        txt = open(q).read()
        for line in txt.splitlines():
            if "rc=" in line and not line.rstrip().endswith("rc=0") and "rc=0 " not in line:
                if any(f"rc={n}" in line for n in range(1, 10)):
                    bad.append(f"STAGE  {line.strip()}")

    # the 180k set is the critical path; report shortfall once it is finished
    if trainers == 0:
        have = len([d for d in glob.glob(os.path.join(REPO, "runs-v1.4", "*_180k"))
                    if json.load(open(os.path.join(d, "run.json"))).get("final_step", 0) >= 180000]
                   ) if glob.glob(os.path.join(REPO, "runs-v1.4", "*_180k/run.json")) else 0
        if 0 < have < EXPECTED_180K - 1:
            bad.append(f"SHORTFALL  only {have} of {EXPECTED_180K-1} new 180k runs "
                       f"reached 180000 steps")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    bad = check()
    if bad:
        for b in bad:
            print(b, flush=True)
        raise SystemExit(1)
    if not a.quiet:
        t = procs("train[.]py")
        print(f"OK  {t} trainer(s), "
              f"{shutil.disk_usage(REPO).free/2**30:.0f} GB free, nothing stalled")


if __name__ == "__main__":
    main()
