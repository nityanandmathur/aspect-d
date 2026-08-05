"""ASPECT-D orchestration — job scheduling across 8 GPUs, state.json, gates G0–G6.

Multi-GPU policy (task.md §2): parallelise ACROSS runs, never shard one run.
`state.json` is the resumable source of truth; `LOG.md` is the append-only trail.

    python src/orchestrate.py phase1     # μP sweeps → G1, G1b
    python src/orchestrate.py phase2     # pilots A3,B3,C3 → G2
    python src/orchestrate.py phase3     # remaining mandatory grid → G3
    python src/orchestrate.py phase4     # T-sweep synthesis + scoring
    python src/orchestrate.py status
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(REPO, "state.json")
LOG = os.path.join(REPO, "LOG.md")
PY = "/home/ubuntu/venv/bin/python"
N_GPUS = 8
GPU_CAP = 500.0                 # grid.json compute.stop_loss_gpu_hours


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def load_state() -> Dict:
    if os.path.exists(STATE):
        with open(STATE) as fh:
            return json.load(fh)
    grid = json.load(open(os.path.join(REPO, "configs", "grid.json")))
    return {
        "phase": 0, "started": now(), "updated": now(),
        "active_recipe": "coarse-to-fine (grid.json backbone.training_objective)",
        "active_budgets": ["A", "B", "C"],
        "active_configs": [c["id"] for c in grid["configs"]
                           if c["budget"] in ("A", "B", "C")],
        "active_seeds": grid["seeds"]["mandatory"],
        "T_grid": grid["backbone"]["sampler_frozen"]["T_eval_grid"],
        "T_semantics": "steps per level, NFE = 8T (grid.json sampler_frozen.T_definition)",
        "gpu_hours": {"total": 0.0, "phase0": 0.0, "phase1": 0.0, "phase2": 0.0,
                      "phase3": 0.0, "phase4": 0.0},
        "chosen_lr": None, "lr_rule": None,
        "completed_runs": [], "failed_runs": [], "flags": [],
        "gates": {}, "cuts": [], "calendar": {}, "eval_items": 400,
        "utmos": None, "sim_model": None, "pivots_fired": [],
    }


def save_state(st: Dict) -> None:
    st["updated"] = now()
    with open(STATE, "w") as fh:
        json.dump(st, fh, indent=1)


def log(md: str) -> None:
    with open(LOG, "a") as fh:
        fh.write("\n" + md.rstrip() + "\n")
    print(md, flush=True)


# --------------------------------------------------------------------- scheduler
class Scheduler:
    """Runs shell jobs on free GPUs. `per_gpu` > 1 lets small runs share a device — each
    run still lives entirely on ONE GPU (task.md §2: never shard one run); co-tenancy only
    raises utilisation, which is very low for the narrow configs. GPU-hours are charged as
    the sum of per-run wall time, i.e. an over-count under co-tenancy (conservative
    against gate G5); `job_history` in state.json keeps the intervals for exact
    occupancy accounting."""

    def __init__(self, gpus: List[int], per_gpu: int = 1):
        self.free = [g for g in gpus for _ in range(per_gpu)]
        self.lock = threading.Lock()
        self.results: List[Dict] = []

    def run(self, jobs: List[Dict]) -> List[Dict]:
        pending = list(jobs)
        threads: List[threading.Thread] = []
        while pending or threads:
            threads = [t for t in threads if t.is_alive()]
            with self.lock:
                got = self.free.pop(0) if (self.free and pending) else None
            if got is None:
                time.sleep(2.0)
                continue
            job = pending.pop(0)
            t = threading.Thread(target=self._one, args=(job, got), daemon=True)
            t.start()
            threads.append(t)
            time.sleep(1.0)
        return self.results

    def _one(self, job: Dict, gpu: int) -> None:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.makedirs(os.path.dirname(job["log"]), exist_ok=True)
        t0 = time.time()
        with open(job["log"], "a") as lf:
            lf.write(f"\n===== {job['label']} on GPU{gpu} @ {now()} =====\n")
            lf.flush()
            rc = subprocess.call(job["cmd"], stdout=lf, stderr=subprocess.STDOUT, env=env,
                                 cwd=os.path.join(REPO, "src"))
        el = time.time() - t0
        print(f"[sched] {job['label']} rc={rc} {el/3600:.2f} GPU-h", flush=True)
        with self.lock:
            self.results.append({**{k: v for k, v in job.items() if k != "cmd"},
                                 "rc": rc, "gpu_hours": el / 3600, "gpu": gpu,
                                 "t_start": t0, "t_end": time.time()})
            self.free.append(gpu)


def train_job(cfg: str, seed: int, lr: float, out: str, steps: Optional[int] = None,
              proxy: Optional[tuple] = None, label: Optional[str] = None,
              extra: Optional[List[str]] = None) -> Dict:
    cmd = [PY, "train.py", "--config", cfg, "--seed", str(seed), "--lr", str(lr),
           "--out", os.path.join(REPO, out), "--device", "cuda:0"]
    if steps:
        cmd += ["--steps", str(steps)]
    if proxy:
        cmd += ["--proxy-width", str(proxy[0]), "--proxy-depth", str(proxy[1])]
    if extra:
        cmd += extra
    return {"label": label or f"train:{cfg}_s{seed}", "cmd": cmd, "out": out,
            "log": os.path.join(REPO, "logs", "jobs", (label or f"{cfg}_{seed}") + ".log"),
            "kind": "train"}


# ------------------------------------------------------------- gate helpers
def ema(vals: List[float], k: int = 3) -> float:
    v = vals[-k:]
    w = [0.5 ** i for i in range(len(v))][::-1]
    return sum(a * b for a, b in zip(v, w)) / sum(w)


def sweep_pick(paths: List[str], lrs: List[float]) -> Dict:
    """EMA over the last 3 val points (task.md §10) → argmin LR."""
    out = {}
    for lr, p in zip(lrs, paths):
        rj = os.path.join(REPO, p, "run.json")
        if not os.path.exists(rj):
            out[lr] = None
            continue
        r = json.load(open(rj))
        vh = [v["val_loss"] for v in r.get("val_hist", [])]
        out[lr] = ema(vh) if vh else None
    ok = {k: v for k, v in out.items() if v is not None}
    return {"per_lr": out, "argmin": min(ok, key=ok.get) if ok else None}


def _record_jobs(res: List[Dict]) -> None:
    """Append scheduler intervals to state.json for exact GPU-occupancy accounting."""
    st = load_state()
    st.setdefault("job_history", []).extend(
        [{k: r[k] for k in ("label", "gpu", "rc", "gpu_hours", "t_start", "t_end")} for r in res])
    st["gpu_occupancy_hours"] = occupancy_hours(st["job_history"])
    save_state(st)


def occupancy_hours(hist: List[Dict]) -> float:
    """Union of busy intervals per GPU (co-tenant runs on one GPU count once)."""
    tot = 0.0
    by_gpu: Dict[int, List] = {}
    for h in hist:
        by_gpu.setdefault(h["gpu"], []).append((h["t_start"], h["t_end"]))
    for iv in by_gpu.values():
        iv.sort()
        cs, ce = iv[0]
        for s, e in iv[1:]:
            if s > ce:
                tot += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        tot += ce - cs
    return tot / 3600


def gpu_hours_from_runs() -> float:
    tot = 0.0
    for rj in glob.glob(os.path.join(REPO, "runs", "**", "run.json"), recursive=True):
        try:
            tot += float(json.load(open(rj)).get("gpu_hours", 0.0))
        except Exception:
            pass
    for sj in glob.glob(os.path.join(REPO, "runs", "**", "synth_T*", "synth.json"),
                        recursive=True):
        try:
            tot += float(json.load(open(sj)).get("gpu_hours", 0.0))
        except Exception:
            pass
    return tot


# ------------------------------------------------------------------- phase 1
PROXIES = {"g1_w256": (256, 12), "g1_w640": (640, 12), "g1b_d4": (384, 4), "g1b_d24": (384, 24)}


def phase1(a):
    st = load_state()
    grid = json.load(open(os.path.join(REPO, "configs", "grid.json")))
    lrs = grid["training"]["mup"]["base_lr_sweep"]
    jobs, paths = [], {}
    for name, (w, d) in PROXIES.items():
        paths[name] = []
        for lr in lrs:
            out = f"runs/sweep_{name}_lr{lr}"
            paths[name].append(out)
            if os.path.exists(os.path.join(REPO, out, "run.json")) and \
                    json.load(open(os.path.join(REPO, out, "run.json")))["status"] == "completed":
                continue
            jobs.append(train_job("A1", 0, lr, out, steps=a.steps, proxy=(w, d),
                                  label=f"sweep_{name}_lr{lr}",
                                  extra=["--coord-check", "50"] if name.startswith("g1_") else None))
    print(f"[phase1] {len(jobs)} sweep jobs of {a.steps} steps", flush=True)
    res = Scheduler(list(range(N_GPUS)), per_gpu=getattr(a, "per_gpu", 1)).run(jobs)
    _record_jobs(res)
    picks = {name: sweep_pick(paths[name], lrs) for name in PROXIES}
    g1 = picks["g1_w256"]["argmin"], picks["g1_w640"]["argmin"]
    g1b = picks["g1b_d4"]["argmin"], picks["g1b_d24"]["argmin"]
    within2 = lambda x, y: bool(x and y and 0.5 - 1e-9 <= x / y <= 2.0 + 1e-9)
    g1_pass, g1b_pass = within2(*g1), within2(*g1b)
    st["gates"]["G1"] = {"argmin_w256": g1[0], "argmin_w640": g1[1], "passes": g1_pass,
                         "sweeps": {k: picks[k] for k in ("g1_w256", "g1_w640")}, "when": now()}
    st["gates"]["G1b"] = {"argmin_d4": g1b[0], "argmin_d24": g1b[1], "passes": g1b_pass,
                          "sweeps": {k: picks[k] for k in ("g1b_d4", "g1b_d24")}, "when": now()}
    st["chosen_lr"] = picks["g1_w256"]["argmin"]
    st["lr_rule"] = ("muP single base LR from the (d=12, w=256) base-width sweep; "
                     "width and depth transfer verified at G1/G1b"
                     if g1_pass and g1b_pass else "see LOG.md fallback entry")
    st["gpu_hours"]["phase1"] = sum(r["gpu_hours"] for r in res)
    st["gpu_hours"]["total"] = gpu_hours_from_runs()
    st["phase"] = 1
    save_state(st)
    log(f"""## {now()} — Gate G1 / G1b (Phase 1, μP transfer)

5-point sweeps ({lrs}), {a.steps}-step proxies, EMA over last 3 val points.

| sweep | argmin LR | val(EMA) per LR |
|---|---|---|
""" + "\n".join(
        f"| {k} (w={PROXIES[k][0]}, d={PROXIES[k][1]}) | {picks[k]['argmin']} | "
        + ", ".join(f"{lr}:{(v if v is None else round(v,4))}"
                    for lr, v in picks[k]["per_lr"].items()) + " |"
        for k in PROXIES) + f"""

- **G1 (width transfer)**: argmin(w=256)={g1[0]}, argmin(w=640)={g1[1]} → within a factor
  of 2: **{'PASS' if g1_pass else 'FAIL'}**.
- **G1b (depth transfer)**: argmin(d=4)={g1b[0]}, argmin(d=24)={g1b[1]} → within a factor
  of 2: **{'PASS' if g1b_pass else 'FAIL'}**.
- Chosen LR rule: {st['lr_rule']} → base LR **{st['chosen_lr']}**.
- Phase-1 compute: {st['gpu_hours']['phase1']:.2f} GPU-h (cumulative {st['gpu_hours']['total']:.2f} / {GPU_CAP}).
""")


# ------------------------------------------------------------------- phase 2/3
def grid_jobs(st: Dict, configs: List[str], seeds: List[int], steps: Optional[int] = None
              ) -> List[Dict]:
    jobs = []
    for cfg, seed in itertools.product(configs, seeds):
        out = f"runs/{cfg}_{seed}"
        rj = os.path.join(REPO, out, "run.json")
        if os.path.exists(rj) and json.load(open(rj)).get("status") == "completed":
            continue
        jobs.append(train_job(cfg, seed, st["chosen_lr"], out, steps=steps))
    return jobs


def _g3_restart(res: List[Dict], st: Dict, steps: Optional[int]) -> List[Dict]:
    """Gate G3: a run that NaNs or diverges gets ONE auto-restart FROM SCRATCH at 0.5× LR,
    then it is marked failed. The old run directory is moved aside (never resumed) and the
    restart is recorded in state.json so the one-restart limit is enforceable."""
    ledger = st.setdefault("restarts", {})
    jobs = []
    for r in res:
        name = os.path.basename(r["out"])
        rj = os.path.join(REPO, r["out"], "run.json")
        status = json.load(open(rj)).get("status") if os.path.exists(rj) else "missing"
        if status == "completed":
            continue
        if ledger.get(name, {}).get("count", 0) >= 1:
            log(f"- **G3**: `{name}` failed again after its single 0.5× LR restart "
                f"(status `{status}`) → marked **failed**, no further restart.")
            continue
        old = os.path.join(REPO, r["out"])
        dead = old + f".diverged{ledger.get(name, {}).get('count', 0)}"
        if os.path.exists(old):
            os.replace(old, dead)
        lr = st["chosen_lr"] * 0.5
        ledger[name] = {"count": ledger.get(name, {}).get("count", 0) + 1, "lr": lr,
                        "reason": status, "when": now(), "quarantined": os.path.basename(dead)}
        log(f"- **G3**: `{name}` status `{status}` → one restart FROM SCRATCH at 0.5× LR "
            f"({lr}); previous directory kept as `{os.path.basename(dead)}`.")
        cfg, seed = name.rsplit("_", 1)
        jobs.append(train_job(cfg, int(seed), lr, r["out"], steps=steps,
                              label=f"{name}_restart"))
    save_state(st)
    return jobs


def phase_train(a):
    st = load_state()
    assert st["chosen_lr"], "Phase 1 must set chosen_lr first"
    configs = a.configs.split(",") if a.configs else st["active_configs"]
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else st["active_seeds"]
    jobs = grid_jobs(st, configs, seeds, a.steps)
    print(f"[train] {len(jobs)} runs: {[j['label'] for j in jobs]}", flush=True)
    res = Scheduler(list(range(N_GPUS)), per_gpu=getattr(a, "per_gpu", 1)).run(jobs)
    _record_jobs(res)
    retry = _g3_restart(res, load_state(), a.steps)
    if retry:
        print(f"[train] G3 restarts: {[j['label'] for j in retry]}", flush=True)
        res2 = Scheduler(list(range(N_GPUS)), per_gpu=getattr(a, "per_gpu", 1)).run(retry)
        _record_jobs(res2)
        res = res + res2
    ph = f"phase{a.phase}"
    st = load_state()
    st["gpu_hours"][ph] = st["gpu_hours"].get(ph, 0.0) + sum(r["gpu_hours"] for r in res)
    st["gpu_hours"]["total"] = gpu_hours_from_runs()
    for r in res:
        name = os.path.basename(r["out"])
        rj = os.path.join(REPO, r["out"], "run.json")
        status = json.load(open(rj)).get("status") if os.path.exists(rj) else "missing"
        (st["completed_runs"] if status == "completed" else st["failed_runs"]).append(
            {"run": name, "status": status, "gpu_hours": r["gpu_hours"]})
    st["completed_runs"] = _dedup(st["completed_runs"])
    st["failed_runs"] = [f for f in _dedup(st["failed_runs"])
                         if f["run"] not in {c["run"] for c in st["completed_runs"]}]
    st["phase"] = a.phase
    save_state(st)
    print(f"[train] cumulative {st['gpu_hours']['total']:.2f} GPU-h", flush=True)


def _dedup(rows: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for r in rows:
        if r["run"] in seen:
            continue
        seen.add(r["run"])
        out.append(r)
    return out


# --------------------------------------------------------------------- phase 4
def phase4(a):
    st = load_state()
    runs = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(REPO, "runs", "*")))
            if os.path.exists(os.path.join(p, "ckpt.pt")) and not
            os.path.basename(p).startswith("sweep_")]
    if a.runs:
        runs = a.runs.split(",")
    Ts = [int(t) for t in a.T.split(",")] if a.T else st["T_grid"]
    jobs = []
    for run in runs:
        for T in Ts:
            out = os.path.join("runs", run, f"synth_T{T}")
            if os.path.exists(os.path.join(REPO, out, "synth.json")) and not a.force:
                continue
            jobs.append({"label": f"synth:{run}_T{T}",
                         "cmd": [PY, "sample.py", "synth", "--run", os.path.join(REPO, "runs", run),
                                 "--T", str(T), "--device", "cuda:0"],
                         "log": os.path.join(REPO, "logs", "jobs", f"synth_{run}_T{T}.log"),
                         "out": out, "kind": "synth"})
    print(f"[phase4] {len(jobs)} synthesis jobs over {len(runs)} runs × {Ts}", flush=True)
    res = Scheduler(list(range(N_GPUS)), per_gpu=getattr(a, "per_gpu", 1)).run(jobs)
    _record_jobs(res)
    failed = [r for r in res if r["rc"] != 0]
    if failed:                                   # retry once (task.md §10 crash default)
        log(f"- **Phase 4**: {len(failed)} synthesis jobs exited non-zero "
            f"({[r['label'] for r in failed]}) → retried once.")
        again = [j for j in jobs if j["label"] in {r["label"] for r in failed}]
        res2 = Scheduler(list(range(N_GPUS)), per_gpu=getattr(a, "per_gpu", 1)).run(again)
        _record_jobs(res2)
        res = res + res2
        still = [r for r in res2 if r["rc"] != 0]
        if still:
            log(f"- **Phase 4**: STILL failing after the retry: {[r['label'] for r in still]} "
                f"— these (config, T) points are absent from the surface and are reported as "
                f"missing, not silently dropped.")
    st = load_state()
    st["gpu_hours"]["phase4"] = st["gpu_hours"].get("phase4", 0.0) + sum(r["gpu_hours"] for r in res)
    st["gpu_hours"]["total"] = gpu_hours_from_runs()
    save_state(st)


def phase4_score(a):
    st = load_state()
    # protocol §6.3: the integrity check runs BEFORE any metric is computed, once per run
    if not a.skip_integrity:
        rc = subprocess.call([PY, "sample.py", "integrity", "--runs-glob",
                              os.path.join(REPO, "runs", "*"), "--t-lo", str(min(st["T_grid"])),
                              "--t-hi", str(max(st["T_grid"])), "--out",
                              os.path.join(REPO, "artifacts", "integrity.json")],
                             cwd=os.path.join(REPO, "src"))
        integ = json.load(open(os.path.join(REPO, "artifacts", "integrity.json")))
        st = load_state()
        st["gates"]["G_sampler_integrity"] = {"passes": integ["all_pass"],
                                              "n_runs": integ["n_runs"],
                                              "n_failing": integ["n_failing"], "when": now()}
        save_state(st)
        if rc != 0 or not integ["all_pass"]:
            log(f"- **protocol §6.3 sampler integrity FAILED** for {integ['n_failing']} of "
                f"{integ['n_runs']} runs → scoring aborted; T is not reaching the sampler.")
            raise SystemExit("sampler-integrity check failed — fix before scoring")
        log(f"- **protocol §6.3 sampler integrity**: PASS for all {integ['n_runs']} runs "
            f"(T=1 vs T={max(st['T_grid'])} differ on "
            f"{100*min(v['differing_cell_fraction'] for v in integ['per_run'].values()):.1f}–"
            f"{100*max(v['differing_cell_fraction'] for v in integ['per_run'].values()):.1f}% "
            f"of generated cells; threshold 20%).")
    jobs_all = []
    for sdir in sorted(glob.glob(os.path.join(REPO, "runs", "*", "synth_T*"))):
        if not os.path.exists(os.path.join(sdir, "synth.json")):
            continue
        if os.path.exists(os.path.join(sdir, "scores.json")) and not a.force:
            continue
        run = os.path.dirname(sdir)
        T = int(os.path.basename(sdir).split("T")[1])
        jobs_all.append({"run": run, "T": T})
    if not jobs_all:
        print("[score] nothing to do", flush=True)
        return
    chunks = [jobs_all[i::N_GPUS] for i in range(N_GPUS)]
    jobs = []
    os.makedirs(os.path.join(REPO, "logs", "jobs"), exist_ok=True)
    for i, ch in enumerate(chunks):
        if not ch:
            continue
        jf = os.path.join(REPO, "logs", "jobs", f"score_chunk{i}.json")
        json.dump(ch, open(jf, "w"))
        jobs.append({"label": f"score:chunk{i}({len(ch)})",
                     "cmd": [PY, "evaluate.py", "score", "--jobs", jf, "--device", "cuda:0"]
                             + (["--force"] if a.force else []),
                     "log": os.path.join(REPO, "logs", "jobs", f"score_chunk{i}.log"),
                     "out": jf, "kind": "score"})
    res = Scheduler(list(range(N_GPUS)), per_gpu=getattr(a, "per_gpu", 1)).run(jobs)
    _record_jobs(res)
    st = load_state()
    st["gpu_hours"]["phase4"] = st["gpu_hours"].get("phase4", 0.0) + sum(r["gpu_hours"] for r in res)
    st["gpu_hours"]["total"] = gpu_hours_from_runs() + st["gpu_hours"]["phase4"]
    save_state(st)


def status(a):
    st = load_state()
    print(json.dumps({k: v for k, v in st.items()
                      if k not in ("gates",)}, indent=1)[:4000], flush=True)
    print("gates:", json.dumps({k: v.get("passes") for k, v in st.get("gates", {}).items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("phase1")
    p1.add_argument("--steps", type=int, default=3000)
    p1.add_argument("--per-gpu", type=int, default=1)
    p1.set_defaults(fn=phase1)
    pt = sub.add_parser("train")
    pt.add_argument("--phase", type=int, default=3)
    pt.add_argument("--configs")
    pt.add_argument("--seeds")
    pt.add_argument("--steps", type=int, default=None)
    pt.add_argument("--per-gpu", type=int, default=1)
    pt.set_defaults(fn=phase_train)
    p4 = sub.add_parser("phase4")
    p4.add_argument("--runs")
    p4.add_argument("--T")
    p4.add_argument("--force", action="store_true")
    p4.add_argument("--per-gpu", type=int, default=1)
    p4.set_defaults(fn=phase4)
    ps = sub.add_parser("score")
    ps.add_argument("--force", action="store_true")
    ps.add_argument("--per-gpu", type=int, default=1)
    ps.add_argument("--skip-integrity", action="store_true")
    ps.set_defaults(fn=phase4_score)
    stt = sub.add_parser("status")
    stt.set_defaults(fn=status)
    args = ap.parse_args()
    args.fn(args)
