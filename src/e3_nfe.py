"""E3 — NFE allocation across codebook levels (task-v1.md §4-E3, hypothesis H-E3).

Three frozen per-level step schedules at matched total NFE = 32, on the C budget
(5 configs × 3 seeds = 15 runs, full 400 items):

    uniform = [4,4,4,4,4,4,4,4]
    coarse  = [25,1,1,1,1,1,1,1]      all steps on codebook level 0
    fine    = [1,1,1,1,1,1,1,25]      all steps on codebook level 7

H-E3 (§9): WER(coarse) < WER(fine), paired item-level bootstrap 95 % CI on the
difference excluding 0. Secondary: WER(coarse) ≤ WER(uniform).

The bootstrap is **paired on items**: the same 400 eval items are synthesised by
every schedule on every run, so resampling items (not runs) and differencing
within item removes item difficulty, which is by far the largest variance
component. 2 000 replicates, RNG 7331 (§5).

    python src/e3_nfe.py
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "artifacts-v1.1")
SCHEDULES = {"uniform": [4] * 8, "coarse": [25] + [1] * 7, "fine": [1] * 7 + [25]}
RUNS = [f"C{i}_{s}" for i in range(1, 6) for s in (0, 1, 2)]
BOOT_RNG = 7331
N_BOOT = 2000


def load() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        for nm in SCHEDULES:
            f = os.path.join(REPO, "results", "runs", run, f"synth_nfe32{nm}", "scores.json")
            if not os.path.exists(f):
                print(f"[e3] MISSING {f}", flush=True)
                continue
            d = json.load(open(f))
            assert d["summary"]["n_items"] == 400, f"{f}: {d['summary']['n_items']} items"
            for r in d["items"]:
                rows.append({"run": run, "schedule": nm, "item": r["item"],
                             "wer": r["wer"], "sim": r["sim"],
                             "utmos": r["utmos"] if r["utmos"] is not None else np.nan,
                             "degenerate": r["degenerate"], "crashed": r["crashed"]})
    return pd.DataFrame(rows)


def paired_boot(df: pd.DataFrame, a: str, b: str, metric: str, n_boot: int) -> Dict:
    """CI on mean(metric[a]) - mean(metric[b]), resampling ITEMS with replacement.

    Each replicate averages over all 15 runs within the drawn items, so the unit of
    resampling is the eval item and the two schedules always see the same draw."""
    piv = df.pivot_table(index="item", columns="schedule", values=metric, aggfunc="mean")
    piv = piv[[a, b]].dropna()
    diff_by_item = (piv[a] - piv[b]).values
    items = np.arange(len(diff_by_item))
    rng = np.random.default_rng(BOOT_RNG)
    reps = np.array([diff_by_item[rng.choice(items, size=len(items), replace=True)].mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"a": a, "b": b, "metric": metric, "n_items": int(len(diff_by_item)),
            "mean_a": float(piv[a].mean()), "mean_b": float(piv[b].mean()),
            "diff": float(diff_by_item.mean()), "ci": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0), "n_boot": n_boot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    df = load()
    have = df.groupby("schedule").run.nunique().to_dict()
    print(f"[e3] loaded {len(df)} item-rows; runs per schedule: {have}", flush=True)
    if any(v != len(RUNS) for v in have.values()) or len(have) != 3:
        print(f"[e3] INCOMPLETE — refusing to test H-E3", flush=True)
        return

    per = df.groupby("schedule").agg(
        wer=("wer", "mean"), sim=("sim", "mean"), utmos=("utmos", "mean"),
        degen=("degenerate", "mean"), crash=("crashed", "mean")).to_dict("index")

    cf = paired_boot(df, "coarse", "fine", "wer", a.n_boot)
    cu = paired_boot(df, "coarse", "uniform", "wer", a.n_boot)
    sim_cf = paired_boot(df, "coarse", "fine", "sim", a.n_boot)
    # coarse beats uniform on WER but not obviously on identity/quality — the
    # metric-selectivity of *where* NFE is spent is the interesting part, so test it
    sim_cu = paired_boot(df, "coarse", "uniform", "sim", a.n_boot)
    ut_cu = paired_boot(df, "coarse", "uniform", "utmos", a.n_boot)
    res = {
        "schedules": SCHEDULES, "n_runs": len(RUNS), "total_nfe": 32,
        "per_schedule": per,
        "primary_coarse_minus_fine_wer": cf,
        "secondary_coarse_minus_uniform_wer": cu,
        "exploratory_coarse_minus_fine_sim": sim_cf,
        "exploratory_coarse_minus_uniform_sim": sim_cu,
        "exploratory_coarse_minus_uniform_utmos": ut_cu,
        "H_E3": {
            "rule": "WER(coarse) < WER(fine) with the paired item-level bootstrap 95% CI on "
                    "the difference excluding 0; secondary WER(coarse) <= WER(uniform) "
                    "(PREREGISTRATION-v1.1.md)",
            "primary_supported": bool(cf["diff"] < 0 and cf["excludes_zero"]),
            "secondary_supported": bool(cu["diff"] <= 0),
            "supported": bool(cf["diff"] < 0 and cf["excludes_zero"] and cu["diff"] <= 0),
        },
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "e3_nfe.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    df.to_csv(os.path.join(OUT, "e3_items.csv"), index=False)

    print(json.dumps(res["H_E3"], indent=1), flush=True)
    print(pd.DataFrame(per).T.to_string(), flush=True)
    for k, r in (("coarse-fine WER", cf), ("coarse-uniform WER", cu),
                 ("coarse-fine SIM", sim_cf), ("coarse-uniform SIM", sim_cu),
                 ("coarse-uniform UTMOS", ut_cu)):
        print(f"[e3] {k}: {r['diff']:+.4f} CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] "
              f"{'excludes 0' if r['excludes_zero'] else 'includes 0'}", flush=True)


if __name__ == "__main__":
    main()
