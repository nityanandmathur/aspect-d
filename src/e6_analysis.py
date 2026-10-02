"""E6 — training-compute control at 90k steps (task-v1.md §4-E6, hypothesis H-E5).

{C1, C3, C5} seed 0 retrained 3× longer (90k vs 30k steps). Refits the τ-surface
on those three configs and asks whether Δτ survives more training compute:

  §9 H-E5: Δτ_90k with **item-level** bootstrap 95 % CI excluding 0. Reported
  alongside: whether that CI overlaps the matched 30k 3-config Δτ CI.

The bootstrap is item-level, not run-level, because there is one seed per config
— there is nothing to resample at the run level. Resampling the 400 eval items
and recomputing every (config, T) cell from the per-item rows propagates the
uncertainty that actually exists here. The matched 30k comparison uses the *same
three configs, same seed, same items*, so the only difference is training steps.

Single-seed scope is acknowledged in the pre-registration: this is a control, not
a headline.

    python src/e6_analysis.py
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import fit as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "artifacts-v1.1")
CONFIGS = ["C1", "C3", "C5"]
T_GRID = [1, 2, 4, 8, 16]
N_ITEMS = 400
BOOT_RNG = 7331


def load_items(run_dir: str, T: int) -> Optional[List[Dict]]:
    f = os.path.join(run_dir, f"synth_T{T}", "scores.json")
    if not os.path.exists(f):
        return None
    return sorted(json.load(open(f))["items"], key=lambda r: r["item"])


def collect(which: str) -> Dict:
    """(config, T) -> per-item rows, for the 90k runs or the matched 30k runs."""
    out, missing = {}, []
    for cfg in CONFIGS:
        d = (os.path.join(REPO, "results", "runs-v1.1", f"{cfg}_0_90k") if which == "90k"
             else os.path.join(REPO, "results", "runs", f"{cfg}_0"))
        for T in T_GRID:
            rows = load_items(d, T)
            if rows is None or len(rows) != N_ITEMS:
                missing.append(f"{cfg}/T{T}")
                continue
            out[(cfg, T)] = rows
    out["__missing__"] = missing
    return out


def meta() -> Dict[str, Dict]:
    v1 = pd.read_csv(os.path.join(REPO, "results", "artifacts", "runs.csv"))
    return {c: v1[v1.config == c].iloc[0].to_dict() for c in CONFIGS}


def surface_from(per: Dict, m: Dict, idx: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Build the (config, T) fit table; `idx` selects a bootstrap item resample."""
    rows = []
    for cfg in CONFIGS:
        for T in T_GRID:
            r = per[(cfg, T)]
            wer = np.array([x["wer"] for x in r], float)
            sim = np.array([x["sim"] for x in r], float)
            if idx is not None:
                wer, sim = wer[idx], sim[idx]
            rows.append({"config": cfg, "seed": 0, "T": T,
                         "width": m[cfg]["width"], "depth": m[cfg]["depth"],
                         "n_nonembed": m[cfg]["n_nonembed"], "budget": m[cfg]["budget"],
                         "err_wer": float(np.nanmean(wer)),
                         "err_sim": float(1.0 - np.nanmean(sim))})
    return pd.DataFrame(rows)


def delta_tau(df: pd.DataFrame) -> Optional[float]:
    tw = F.part_b(df, "wer").get("tau")
    ts = F.part_b(df, "sim").get("tau")
    return None if tw is None or ts is None else tw - ts


def item_bootstrap(per: Dict, m: Dict, n_boot: int) -> Dict:
    rng = np.random.default_rng(BOOT_RNG)
    point = delta_tau(surface_from(per, m))
    reps = []
    for _ in range(n_boot):
        idx = rng.integers(0, N_ITEMS, N_ITEMS)
        try:
            v = delta_tau(surface_from(per, m, idx))
        except Exception:
            v = None
        if v is not None and np.isfinite(v):
            reps.append(v)
    return {"delta_tau": point, "delta_tau_ci": F.ci(reps),
            "n_reps_ok": len(reps),
            "excludes_zero": bool(F.ci(reps) and (F.ci(reps)[0] > 0 or F.ci(reps)[1] < 0))}


def overlap(a: List[float], b: List[float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    m = meta()
    p90, p30 = collect("90k"), collect("30k")
    miss = p90.pop("__missing__") + p30.pop("__missing__")
    if miss:
        print(f"[e6] INCOMPLETE — missing {miss}; refusing to test H-E5", flush=True)
        return

    print(f"[e6] item-level bootstrap, {a.n_boot} replicates, "
          f"{len(CONFIGS)} configs x {len(T_GRID)} T x {N_ITEMS} items", flush=True)
    r90 = item_bootstrap(p90, m, a.n_boot)
    r30 = item_bootstrap(p30, m, a.n_boot)
    val = {c: json.load(open(os.path.join(REPO, "results", "runs-v1.1", f"{c}_0_90k", "run.json")))
           for c in CONFIGS}
    v30 = {c: json.load(open(os.path.join(REPO, "results", "runs", f"{c}_0", "run.json")))
           for c in CONFIGS}

    res = {
        "scope": "single seed per config; a control, not a headline (PREREGISTRATION-v1.1 H-E5)",
        "configs": CONFIGS, "T_grid": T_GRID, "n_items": N_ITEMS,
        "bootstrap": "item-level (one seed per config leaves nothing to resample at run level)",
        "delta_tau_90k": r90, "delta_tau_30k_matched": r30,
        "val_loss": {c: {"90k": val[c].get("final_val_loss"),
                         "30k": v30[c].get("final_val_loss"),
                         "gpu_hours_90k": val[c].get("gpu_hours")} for c in CONFIGS},
        "curves": {},
    }
    for tag, per in (("90k", p90), ("30k", p30)):
        s = surface_from(per, m)
        res["curves"][tag] = s.groupby("T")[["err_wer", "err_sim"]].mean().to_dict()
    if r90["delta_tau_ci"] and r30["delta_tau_ci"]:
        res["ci_overlap_90k_vs_30k"] = overlap(r90["delta_tau_ci"], r30["delta_tau_ci"])
    res["H_E5"] = {
        "rule": "delta_tau_90k with item-level bootstrap 95% CI excluding 0; reported "
                "alongside whether it overlaps the matched 30k 3-config CI",
        "delta_tau_90k": r90["delta_tau"], "ci": r90["delta_tau_ci"],
        "supported": bool(r90["delta_tau"] is not None and r90["delta_tau"] > 0
                          and r90["excludes_zero"]),
        "overlaps_30k_ci": res.get("ci_overlap_90k_vs_30k"),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "e6_undertraining.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    print(json.dumps(res["H_E5"], indent=1), flush=True)
    for tag, r in (("90k", r90), ("30k matched", r30)):
        ci = r["delta_tau_ci"]
        print(f"[e6] {tag:<12} delta_tau {r['delta_tau']:+.4f} "
              f"CI [{ci[0]:+.4f}, {ci[1]:+.4f}]", flush=True)
    for c in CONFIGS:
        print(f"[e6] {c} val loss 30k {v30[c].get('final_val_loss'):.4f} -> "
              f"90k {val[c].get('final_val_loss'):.4f}", flush=True)


if __name__ == "__main__":
    main()
