"""v1.5 post-training queue, in priority order, fired when the 180k runs drain.

Stage 1 comes first because it repairs numbers that are in the paper *now*: the S4
guidance arms were sampled with an attended-PAD bug and scored against a baseline
running half their compute. Everything after it is new evidence, which matters less
than a published number being wrong.

    python src/run_v15.py --plan
    python src/run_v15.py --stage 1        # or --all, which waits for training first
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
GAMMAS = ("0p5", "1p0", "2p0")
CRUNS = [f"C{i}_{s}" for i in range(1, 6) for s in (0, 1, 2)]
CFG_RUNS = [(c, s) for c in ("C1", "C3", "C5") for s in (0, 1, 2)]


def gpus_busy() -> int:
    return len(subprocess.run(["pgrep", "-f", "train[.]py"],
                              capture_output=True, text=True).stdout.split())


def sh(cmd, gpu=None, log=None):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.update(OMP_NUM_THREADS="24", MKL_NUM_THREADS="24")
    p = subprocess.run(cmd, env=env, cwd=SRC, capture_output=True, text=True)
    if log:
        log.write(f"$ {' '.join(str(c) for c in cmd)}\n  rc={p.returncode}\n")
        if p.returncode:
            log.write(p.stdout[-2000:] + p.stderr[-2000:] + "\n")
        log.flush()
    return p.returncode


def stage1_resynth_gamma(log):
    """The 45 gamma arms, re-sampled with the attended-PAD fix and re-scored."""
    jobs = [(r, g) for r in CRUNS for g in GAMMAS]
    log.write(f"[stage1] re-synthesising {len(jobs)} gamma arms\n")
    procs = []
    for i, (r, g) in enumerate(jobs):
        rd = os.path.join(REPO, "runs", r)
        cmd = [PY, os.path.join(SRC, "sample.py"), "synth", "--run", rd, "--T", "16",
               "--items", "400", "--gamma", g.replace("p", "."), "--tag", f"gam{g}",
               "--device", "cuda:0"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % 8))
        procs.append(subprocess.Popen(cmd, env=env, cwd=SRC,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if len(procs) >= 8:
            for p in procs:
                p.wait()
            procs = []
    for p in procs:
        p.wait()
    # score in 8 shards with one Scorer each (a cold Scorer costs ~500 s, a warm job ~70)
    spec = [{"run": os.path.join(REPO, "runs", r), "T": 16, "tag": f"gam{g}",
             "items": 400} for r, g in jobs]
    procs = []
    for k in range(8):
        jf = os.path.join(REPO, "logs-v1.5", f"score_gam{k}.json")
        with open(jf, "w") as fh:
            json.dump(spec[k::8], fh)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k),
                   OMP_NUM_THREADS="24", MKL_NUM_THREADS="24")
        procs.append(subprocess.Popen(
            [PY, os.path.join(SRC, "evaluate.py"), "score", "--jobs", jf,
             "--force", "--device", "cuda:0"], env=env, cwd=SRC,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    return sh([PY, os.path.join(SRC, "s4_analysis.py")], gpu=0, log=log)


def stage2_integrity(log):
    """The frozen-sampler guarantee, re-checked after touching sample.py."""
    return sh([PY, os.path.join(SRC, "sample.py"), "integrity",
               "--runs-glob", os.path.join(REPO, "runs", "C1_0"),
               "--t-lo", "1", "--t-hi", "16",
               "--out", os.path.join(REPO, "artifacts-v1.4", "integrity_v15.json")],
              gpu=0, log=log)


def stage3_cfg_train(log):
    """9 runs with condition dropout, matching the 90k design (PREREGISTRATION-v1.5)."""
    procs = []
    for i, (c, s) in enumerate(CFG_RUNS):
        out = os.path.join(REPO, "runs-v1.4", f"{c}_{s}_cfg")
        if os.path.exists(os.path.join(out, "run.json")):
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % 8))
        procs.append(subprocess.Popen(
            [PY, os.path.join(SRC, "train.py"), "--config", c, "--seed", str(s),
             "--lr", "0.004", "--steps", "30000", "--cond-dropout", "0.10",
             "--out", out, "--device", "cuda:0"],
            env=env, cwd=SRC, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    log.write(f"[stage3] trained {len(procs)} CFG checkpoints\n")
    return 0


def stage4_ht3(log):
    """H-T3, properly this time: --variable-prompt now reaches build_inputs."""
    procs = []
    for i, (c, s) in enumerate([("C3", 0), ("C1", 0), ("C5", 0)]):
        out = os.path.join(REPO, "runs-v1.4", f"{c}_{s}_varprompt")
        if os.path.exists(os.path.join(out, "run.json")):
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % 8))
        procs.append(subprocess.Popen(
            [PY, os.path.join(SRC, "train.py"), "--config", c, "--seed", str(s),
             "--lr", "0.004", "--steps", "30000", "--variable-prompt",
             "--out", out, "--device", "cuda:0"],
            env=env, cwd=SRC, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs:
        p.wait()
    log.write(f"[stage4] trained {len(procs)} variable-prompt checkpoints\n")
    return 0


def _synth_many(jobs, log, tag_of):
    """jobs: list of (run_dir, T, extra_args, tag). 8-wide, one GPU each."""
    procs = []
    for i, (rd, T, extra, tag) in enumerate(jobs):
        out = os.path.join(rd, f"synth_{tag}")
        if os.path.exists(os.path.join(out, "synth.json")):
            continue
        cmd = [PY, os.path.join(SRC, "sample.py"), "synth", "--run", rd, "--T", str(T),
               "--items", "400", "--tag", tag, "--device", "cuda:0"] + extra
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % 8))
        procs.append(subprocess.Popen(cmd, env=env, cwd=SRC,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        if len(procs) >= 8:
            [p.wait() for p in procs]; procs = []
    [p.wait() for p in procs]
    spec = [{"run": rd, "T": T, "tag": tag, "items": 400} for rd, T, _, tag in jobs]
    procs = []
    for k in range(8):
        jf = os.path.join(REPO, "logs-v1.5", f"score_{tag_of}{k}.json")
        with open(jf, "w") as fh:
            json.dump(spec[k::8], fh)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k),
                   OMP_NUM_THREADS="24", MKL_NUM_THREADS="24")
        procs.append(subprocess.Popen(
            [PY, os.path.join(SRC, "evaluate.py"), "score", "--jobs", jf, "--device", "cuda:0"],
            env=env, cwd=SRC, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    [p.wait() for p in procs]
    log.write(f"[synth] {len(spec)} arms done ({tag_of})\n")


def stage5_cfg_sweep(log):
    """PREREGISTRATION-v1.5: T=1 anchor, T=16 point, T=32 iso-NFE baseline, and the
    two guidance arms at three weights each. Guidance costs two forwards per step, so
    guided T=16 is compared against unguided T=32, never unguided T=16."""
    jobs = []
    for c, s_ in CFG_RUNS:
        rd = os.path.join(REPO, "runs-v1.4", f"{c}_{s_}_cfg")
        if not os.path.isdir(rd):
            continue
        for T in (1, 16, 32):
            jobs.append((rd, T, [], f"T{T}"))
        for w in ("0.5", "1.0", "2.0"):
            wt = w.replace(".", "p")
            jobs.append((rd, 16, ["--cfg-prompt", w], f"cfgp{wt}"))
            jobs.append((rd, 16, ["--cfg-text", w], f"cfgt{wt}"))
    _synth_many(jobs, log, "cfg")
    return 0


def stage6_anchor(log):
    """External anchor, fenced. Runs in its own venv and never touches the frozen
    metric stack; synthesis there, scoring here."""
    scr = os.path.join(SRC, "anchor_f5.py")
    if not os.path.exists(scr):
        log.write("[stage6] anchor_f5.py absent -- skipped\n")
        return 0
    return sh([PY, scr, "--budget-seconds", "10800"], gpu=7, log=log)


def stage7_analyse(log):
    """Every analysis that the new data moves, in dependence order."""
    rc = 0
    rc |= sh([PY, os.path.join(SRC, "x2_analysis.py")], gpu=0, log=log)
    rc |= sh([PY, os.path.join(SRC, "v14_analysis.py")], gpu=0, log=log)
    gate = os.path.join(SRC, "v15_gate.py")
    if os.path.exists(gate):
        rc |= sh([PY, gate], gpu=0, log=log)
    return rc


STAGES = {1: ("re-synthesise the 45 gamma arms and refit S4", stage1_resynth_gamma),
          2: ("re-verify the frozen sampler reproduces v1.0", stage2_integrity),
          3: ("train 9 condition-dropout checkpoints", stage3_cfg_train),
          4: ("re-run H-T3 with a working --variable-prompt", stage4_ht3),
          5: ("sweep the CFG checkpoints, both arms x three weights", stage5_cfg_sweep),
          6: ("external anchor, fenced at 3 h in an isolated venv", stage6_anchor),
          7: ("re-run every analysis the new data moves", stage7_analyse)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(REPO, "logs-v1.5"), exist_ok=True)

    if a.plan:
        print(f"180k trainers still running: {gpus_busy()}")
        for k, (d, _) in sorted(STAGES.items()):
            print(f"  stage {k}: {d}")
        return
    with open(os.path.join(REPO, "logs-v1.5", "v15.log"), "a", buffering=1) as log:
        if a.all:
            while gpus_busy():
                time.sleep(120)
            log.write("[v15] training drained, starting queue\n")
        for k in ([a.stage] if a.stage else sorted(STAGES)):
            desc, fn = STAGES[k]
            log.write(f"=== stage {k}: {desc} ===\n")
            t0 = time.time()
            rc = fn(log)
            log.write(f"=== stage {k} rc={rc} in {(time.time()-t0)/60:.1f} min ===\n")


if __name__ == "__main__":
    main()
