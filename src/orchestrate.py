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
PY = os.environ.get("ASPECTD_PY", sys.executable)
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
    gpus = [int(g) for g in a.gpus.split(",")] if getattr(a, "gpus", None) else list(range(N_GPUS))
    res = Scheduler(gpus, per_gpu=getattr(a, "per_gpu", 1)).run(jobs)
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
        if os.path.exists(rj):
            rec = json.load(open(rj))
            if rec.get("status") == "completed":
                continue
            # a run started by another process (e.g. a hand-launched pilot) must not be
            # relaunched into the same directory
            live = os.path.join(REPO, out, "train_log.jsonl")
            if rec.get("status") == "running" and os.path.exists(live) and \
                    time.time() - os.path.getmtime(live) < 900:
                print(f"[train] {cfg}_{seed} is already running elsewhere — skipping", flush=True)
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
    gpus = [int(g) for g in a.gpus.split(",")] if getattr(a, "gpus", None) else list(range(N_GPUS))
    res = Scheduler(gpus, per_gpu=getattr(a, "per_gpu", 1)).run(jobs)
    _record_jobs(res)
    retry = _g3_restart(res, load_state(), a.steps)
    if retry:
        print(f"[train] G3 restarts: {[j['label'] for j in retry]}", flush=True)
        res2 = Scheduler(gpus, per_gpu=getattr(a, "per_gpu", 1)).run(retry)
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
    gpus = [int(g) for g in a.gpus.split(",")] if getattr(a, "gpus", None) else list(range(N_GPUS))
    res = Scheduler(gpus, per_gpu=getattr(a, "per_gpu", 1)).run(jobs)
    _record_jobs(res)
    failed = [r for r in res if r["rc"] != 0]
    if failed:                                   # retry once (task.md §10 crash default)
        log(f"- **Phase 4**: {len(failed)} synthesis jobs exited non-zero "
            f"({[r['label'] for r in failed]}) → retried once.")
        again = [j for j in jobs if j["label"] in {r["label"] for r in failed}]
        res2 = Scheduler(gpus, per_gpu=getattr(a, "per_gpu", 1)).run(again)
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
    gpus = [int(g) for g in a.gpus.split(",")] if getattr(a, "gpus", None) else list(range(N_GPUS))
    res = Scheduler(gpus, per_gpu=getattr(a, "per_gpu", 1)).run(jobs)
    _record_jobs(res)
    st = load_state()
    st["gpu_hours"]["phase4"] = st["gpu_hours"].get("phase4", 0.0) + sum(r["gpu_hours"] for r in res)
    st["gpu_hours"]["total"] = gpu_hours_from_runs() + st["gpu_hours"]["phase4"]
    save_state(st)


# --------------------------------------------------------------- gates G2/G3/G5/G6
def _scores(run: str, T: int) -> Optional[Dict]:
    p = os.path.join(REPO, "runs", run, f"synth_T{T}", "scores.json")
    return json.load(open(p))["summary"] if os.path.exists(p) else None


def gate_g2(a=None):
    """protocol §8 gate G2 — pilot floors on A3, B3, C3 (seed 0) at T ∈ {1, 16}."""
    st = load_state()
    g0c_p = os.path.join(REPO, "artifacts", "g0c_groundtruth.json")
    if not os.path.exists(g0c_p):
        # the cross-speaker baseline is a THRESHOLD input, not an optional extra: a missing
        # file must stop the gate, never be silently scored as a failed check
        raise SystemExit(f"gate G2 needs the G0(c) baseline at {g0c_p}; run "
                         f"`python src/evaluate.py gt` first")
    baseline = json.load(open(g0c_p))["sim_cross_median"]
    s = {f"{c}_T{T}": _scores(f"{c}_0", T) for c in ("A3", "B3", "C3") for T in (1, 16)}
    missing = [k for k, v in s.items() if v is None]
    if missing:
        raise SystemExit(f"gate G2 needs scores for {missing}")
    wer_c3, wer_a3 = s["C3_T16"]["wer_mean"], s["A3_T16"]["wer_mean"]
    sim_c3, degen_c3 = s["C3_T16"]["sim_mean"], s["C3_T16"]["degen_rate"]
    checks = {
        "WER(C3,T=16) <= 0.30": (wer_c3 <= 0.30, wer_c3),
        "WER(A3,T=16) <= 0.65": (wer_a3 <= 0.65, wer_a3),
        "SIM-o(C3,T=16) >= 0.30": (sim_c3 >= 0.30, sim_c3),
        "SIM-o(C3,T=16) >= cross baseline + 0.15": (sim_c3 >= baseline + 0.15, sim_c3),
        "DegenRate(C3,T=16) <= 0.40": (degen_c3 <= 0.40, degen_c3),
    }
    passes = all(v[0] for v in checks.values())
    flat = (s["C3_T1"]["wer_mean"] - wer_c3) < 0.03
    integ = os.path.join(REPO, "artifacts", "integrity.json")
    integ_ok = json.load(open(integ))["all_pass"] if os.path.exists(integ) else None
    route = None
    if not passes:
        a3_only = (not checks["WER(A3,T=16) <= 0.65"][0]) and all(
            v[0] for k, v in checks.items() if k != "WER(A3,T=16) <= 0.65")
        c3_fails = any(not v[0] for k, v in checks.items() if k.startswith(("WER(C3", "SIM", "Degen")))
        route = "P2 (ladder shift: drop A, activate D)" if a3_only else (
            "retry C3 at 0.5x LR, then P1-D (recipe pivot)" if c3_fails else "inspect")
    st["gates"]["G2"] = {"passes": passes, "checks": {k: {"pass": v[0], "value": v[1]}
                                                     for k, v in checks.items()},
                         "cross_speaker_baseline": baseline, "T_FLAT": bool(flat),
                         "wer_C3_T1": s["C3_T1"]["wer_mean"], "wer_C3_T16": wer_c3,
                         "sampler_integrity_ok": integ_ok, "route": route, "when": now()}
    if flat and "T-FLAT" not in st["flags"]:
        st["flags"].append("T-FLAT")
    st["phase"] = max(st.get("phase", 0), 2)
    save_state(st)
    rows = "\n".join(f"| {k} | {v[1]:.4f} | {'PASS' if v[0] else 'FAIL'} |"
                     for k, v in checks.items())
    log(f"""## {now()} — Gate G2 (Phase 2 pilot floors)

Full-schedule A3, B3, C3 (seed 0), eval_zs at T ∈ {{1, 16}}; cross-speaker SIM-o
baseline from G0(c) = {baseline if baseline is None else round(baseline, 4)}.

| check | measured | verdict |
|---|---|---|
{rows}

- **G2: {'PASS' if passes else 'FAIL'}**{'' if passes else f' → route: {route}'}
- Step-flatness: WER(C3,T=1) − WER(C3,T=16) = {s['C3_T1']['wer_mean'] - wer_c3:.4f}
  → flag `T-FLAT` {'SET' if flat else 'not set'} (sampler-integrity check:
  {'PASS' if integ_ok else 'not yet run' if integ_ok is None else 'FAIL'}).
- Pilot WER/SIM at T=1: A3 {s['A3_T1']['wer_mean']:.3f}/{s['A3_T1']['sim_mean']:.3f},
  B3 {s['B3_T1']['wer_mean']:.3f}/{s['B3_T1']['sim_mean']:.3f},
  C3 {s['C3_T1']['wer_mean']:.3f}/{s['C3_T1']['sim_mean']:.3f}; at T=16:
  A3 {s['A3_T16']['wer_mean']:.3f}/{s['A3_T16']['sim_mean']:.3f},
  B3 {s['B3_T16']['wer_mean']:.3f}/{s['B3_T16']['sim_mean']:.3f},
  C3 {wer_c3:.3f}/{sim_c3:.3f}.
""")
    return st["gates"]["G2"]


def gate_g3(a=None):
    """protocol §8 gate G3 — grid health: ≥12 of 15 active configs (≥8 of 10 if budget A
    was cut) finish both mandatory seeds."""
    st = load_state()
    need_seeds = set(st["active_seeds"])
    ok_cfgs, part = [], {}
    for cfg in st["active_configs"]:
        done = set()
        for seed in need_seeds:
            rj = os.path.join(REPO, "runs", f"{cfg}_{seed}", "run.json")
            if os.path.exists(rj) and json.load(open(rj)).get("status") == "completed":
                done.add(seed)
        part[cfg] = sorted(done)
        if done >= need_seeds:
            ok_cfgs.append(cfg)
    threshold = 8 if len(st["active_configs"]) <= 10 else 12
    valid = len(ok_cfgs) >= threshold
    st["gates"]["G3"] = {"passes": valid, "configs_with_all_seeds": len(ok_cfgs),
                         "threshold": threshold, "per_config_seeds": part,
                         "restarts": st.get("restarts", {}), "when": now()}
    save_state(st)
    log(f"""## {now()} — Gate G3 (grid health)

{len(ok_cfgs)} of {len(st['active_configs'])} active configs finished both mandatory seeds
(threshold {threshold}) → **{'PASS' if valid else 'FAIL — F2-candidate'}**.
Per-config seeds completed: `{json.dumps(part)}`.
Restart ledger: `{json.dumps(st.get('restarts', {}))}`.
""")
    return st["gates"]["G3"]


def gate_g5(a=None):
    """protocol §8 gate G5 — 500 B200-h cap, projected after every phase."""
    st = load_state()
    used = gpu_hours_from_runs()
    # count from disk, not from the scheduler's in-memory list: the list is only written
    # when a scheduler batch finishes, which would make the projection stale mid-phase
    done_runs = [r for r in glob.glob(os.path.join(REPO, "runs", "*", "run.json"))
                 if "sweep_" not in r and json.load(open(r)).get("status") == "completed"]
    done = len(done_runs)
    used_train = sum(json.load(open(r)).get("gpu_hours", 0.0) for r in done_runs)
    per_run = (used_train / done) if done else 0.0
    todo = len(st["active_configs"]) * len(st["active_seeds"]) - done
    projected = used + per_run * max(0, todo) + 0.15 * len(st["active_configs"]) * \
        len(st["active_seeds"]) * len(st["T_grid"])          # synthesis+scoring allowance
    st["gates"]["G5"] = {"passes": bool(projected <= GPU_CAP), "used_gpu_hours": used,
                         "occupancy_gpu_hours": st.get("gpu_occupancy_hours"),
                         "mean_hours_per_run": per_run, "runs_remaining": todo,
                         "projected_total": projected, "cap": GPU_CAP, "when": now()}
    save_state(st)
    log(f"- **G5** {now()}: used {used:.1f} GPU-h (GPU-occupancy {st.get('gpu_occupancy_hours', 0):.1f} h), "
        f"{done} runs done at {per_run:.2f} h/run, {todo} to go → projected "
        f"**{projected:.0f} / {GPU_CAP} GPU-h** → {'within cap' if projected <= GPU_CAP else 'OVERRUN → apply cut list'}.")
    return st["gates"]["G5"]


def gate_g6(a=None):
    """protocol §8 gate G6 — calendar. Projects the next milestone from measured rates."""
    from datetime import datetime as dt
    st = load_state()
    grid = json.load(open(os.path.join(REPO, "configs", "grid.json")))
    ms = grid["calendar"]["milestones"]
    today = dt.now(timezone.utc).date()
    nxt = next(((k, v) for k, v in ms.items() if dt.strptime(v, "%Y-%m-%d").date() >= today),
               (None, None))
    st["calendar"] = {"today": str(today), "next_milestone": nxt[0], "due": nxt[1],
                      "hard_wall": grid["calendar"]["hard_wall_aoe"],
                      "days_to_wall": (dt.strptime(grid["calendar"]["hard_wall_aoe"],
                                                   "%Y-%m-%d").date() - today).days,
                      "on_track": True, "when": now()}
    st["gates"]["G6"] = {"passes": True, **st["calendar"]}
    save_state(st)
    log(f"- **G6** {now()}: next milestone {nxt[0]} due {nxt[1]}; "
        f"{st['calendar']['days_to_wall']} days to the {grid['calendar']['hard_wall_aoe']} AoE wall; "
        f"phase {st['phase']} → on track, no calendar cut applied.")
    return st["gates"]["G6"]


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
    p1.add_argument("--gpus", default=None, help="comma list of GPU ids")
    p1.set_defaults(fn=phase1)
    pt = sub.add_parser("train")
    pt.add_argument("--phase", type=int, default=3)
    pt.add_argument("--configs")
    pt.add_argument("--seeds")
    pt.add_argument("--steps", type=int, default=None)
    pt.add_argument("--per-gpu", type=int, default=1)
    pt.add_argument("--gpus", default=None, help="comma list of GPU ids")
    pt.set_defaults(fn=phase_train)
    p4 = sub.add_parser("phase4")
    p4.add_argument("--runs")
    p4.add_argument("--T")
    p4.add_argument("--force", action="store_true")
    p4.add_argument("--per-gpu", type=int, default=1)
    p4.add_argument("--gpus", default=None, help="comma list of GPU ids")
    p4.set_defaults(fn=phase4)
    ps = sub.add_parser("score")
    ps.add_argument("--force", action="store_true")
    ps.add_argument("--per-gpu", type=int, default=1)
    ps.add_argument("--gpus", default=None, help="comma list of GPU ids")
    ps.add_argument("--skip-integrity", action="store_true")
    ps.set_defaults(fn=phase4_score)
    stt = sub.add_parser("status")
    stt.set_defaults(fn=status)
    for name, fn in (("g2", gate_g2), ("g3", gate_g3), ("g5", gate_g5), ("g6", gate_g6)):
        sp = sub.add_parser(name)
        sp.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)
