"""E4 — wait for the G1-D LR sweep, adopt its argmin, launch the budget-D grid.

task-v1.md §4-E4: G1-D failed (D3 val@3k 5.6520 vs B3 5.6485), and the
pre-registered remedy is "one 5-point LR sweep at D3 only; adopt its argmin for
all D runs; log". This script does exactly that — the choice is a deterministic
argmin over 3 000-step val loss, so it is code, not judgement.

    python src/launch_e4.py --gpus 0,4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import subprocess
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO, "logs-v1.1")
PY = os.environ.get("ASPECTD_PY", sys.executable)
PROXY_LR, PROXY_VAL = 0.004, 5.652029187286514   # the G1-D proxy itself


def sweep_results():
    out = {PROXY_LR: PROXY_VAL}
    for d in sorted(glob.glob(os.path.join(REPO, "runs-v1.1", "lrsweep_D3_*"))):
        rj = os.path.join(d, "run.json")
        if not os.path.exists(rj):
            continue
        r = json.load(open(rj))
        if r.get("status") != "completed":
            continue
        lr = float(os.path.basename(d).split("_")[-1].replace("p", "."))
        v = r.get("final_val_loss")
        if v is not None and v == v:          # NaN => diverged, excluded
            out[lr] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,4")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--expect", type=int, default=5, help="sweep points expected")
    ap.add_argument("--timeout-min", type=int, default=180)
    a = ap.parse_args()

    t0 = time.time()
    while True:
        res = sweep_results()
        if len(res) >= a.expect:
            break
        if (time.time() - t0) / 60 > a.timeout_min:
            print(f"[e4] timeout with {len(res)}/{a.expect} sweep points: {res}", flush=True)
            return
        time.sleep(60)

    best_lr = min(res, key=res.get)
    print(f"[e4] G1-D LR sweep at D3 (3k steps): "
          + "  ".join(f"lr={k:g}:{v:.4f}" for k, v in sorted(res.items())), flush=True)
    print(f"[e4] argmin lr={best_lr:g} (val {res[best_lr]:.4f}); adopted for all D runs",
          flush=True)

    b3 = [x["val_loss"] for x in
          json.load(open(os.path.join(REPO, "runs", "B3_0", "run.json")))["val_hist"]
          if x["step"] == 3000][0]
    c3 = [x["val_loss"] for x in
          json.load(open(os.path.join(REPO, "runs", "C3_0", "run.json")))["val_hist"]
          if x["step"] == 3000][0]
    with open(os.path.join(REPO, "artifacts-v1.1", "g1d_lr_sweep.json"), "w") as fh:
        json.dump({"gate": "G1-D", "verdict_at_transferred_lr": "FAIL",
                   "proxy_lr": PROXY_LR, "proxy_val_3k": PROXY_VAL,
                   "b3_val_3k": b3, "c3_val_3k": c3,
                   "sweep": {f"{k:g}": v for k, v in sorted(res.items())},
                   "adopted_lr": best_lr, "adopted_val_3k": res[best_lr],
                   "passes_gate_at_adopted_lr": bool(res[best_lr] < b3 and res[best_lr] < c3),
                   "rule": "task-v1.md §4-E4: fail -> one 5-point LR sweep at D3 only; "
                           "adopt its argmin for all D runs; log"}, fh, indent=1)

    jobs = []
    for cfg in ("D1", "D2", "D3", "D4", "D5"):
        for s in a.seeds.split(","):
            out = os.path.join(REPO, "runs-v1.1", f"{cfg}_{s}")
            jobs.append({"key": f"{cfg}_{s}",
                         "cmd": [PY, "train.py", "--config", cfg, "--seed", s,
                                 "--lr", f"{best_lr:g}", "--steps", str(a.steps),
                                 "--out", out, "--device", "cuda:0"]})
    jf = os.path.join(LOGS, "e4_train_jobs.json")
    json.dump(jobs, open(jf, "w"), indent=1)
    print(f"[e4] launching {len(jobs)} budget-D runs on GPUs {a.gpus}", flush=True)
    subprocess.Popen([PY, os.path.join(REPO, "src", "run_v11.py"), "--jobs", jf,
                      "--name", "e4", "--gpus", a.gpus],
                     stdout=open(os.path.join(LOGS, "dispatch_e4.log"), "a"),
                     stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
