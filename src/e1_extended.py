"""E1 — extended test-time scaling, T ∈ {1,…,64} (task-v1.md §4-E1, hypothesis H-E1).

Builds `artifacts-v1.1/runs_ext.csv` over the 21-run subset (A3, B3, C1–C5 ×
seeds {0,1,2}) at T ∈ {1,2,4,8,16,24,32,64}, then tests H-E1:

  (a) T*_WER (95 % of the T=64 value) > 16
  (b) the τ_WER refitted on the extended range lies inside the v1.0 95 % CI

**Every row is computed over the same first 200 eval items.** T ≤ 16 comes from
the frozen v1.0 `scores.json` (400 items) *restricted to the first 200*; T ≥ 24
comes from E1's own `scores_ext200.json`. Re-aggregating v1.0's per-item rows is
read-only — no v1.0 artifact is written (task-v1.md §0.1). Comparing a 200-item
extended curve against a 400-item v1.0 curve would confound subset with range,
which is exactly what H-E1 is trying to measure.

    python src/e1_extended.py
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

import fit as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "artifacts-v1.1")
CONFIGS = ["A3", "B3", "C1", "C2", "C3", "C4", "C5"]
SEEDS = [0, 1, 2]
T_ALL = [1, 2, 4, 8, 16, 24, 32, 64]
T_EXT = [24, 32, 64]
N_ITEMS = 200
V1_TAU_CI = [0.8248, 0.8513]        # PREREGISTRATION-v1.1.md, H-E1


def agg(rows: List[Dict]) -> Dict:
    """Re-aggregate per-item score rows exactly as evaluate.Scorer.score_dir does."""
    wer = np.array([r["wer"] for r in rows], float)
    sim = np.array([r["sim"] for r in rows], float)
    ut = np.array([r["utmos"] if r["utmos"] is not None else np.nan for r in rows], float)
    return {"wer": float(np.nanmean(wer)),
            "wer_se": float(np.nanstd(wer, ddof=1) / np.sqrt(np.isfinite(wer).sum())),
            "sim": float(np.nanmean(sim)),
            "sim_se": float(np.nanstd(sim, ddof=1) / np.sqrt(np.isfinite(sim).sum())),
            "utmos": float(np.nanmean(ut)),
            "degen_rate": float(np.mean([r["degenerate"] for r in rows])),
            "crash_rate": float(np.mean([r["crashed"] for r in rows])),
            "n_items": len(rows)}


def build_csv() -> pd.DataFrame:
    v1 = pd.read_csv(os.path.join(REPO, "artifacts", "runs.csv"))
    meta = {c: v1[v1.config == c].iloc[0] for c in CONFIGS}
    rows = []
    for cfg in CONFIGS:
        m = meta[cfg]
        for s in SEEDS:
            run = os.path.join(REPO, "runs", f"{cfg}_{s}")
            for T in T_ALL:
                f = os.path.join(run, f"synth_T{T}",
                                 "scores_ext200.json" if T in T_EXT else "scores.json")
                if not os.path.exists(f):
                    print(f"[e1] MISSING {f}", flush=True)
                    continue
                d = json.load(open(f))
                # restrict to the first 200 items so every T shares one item subset
                per = sorted(d["items"], key=lambda r: r["item"])[:N_ITEMS]
                assert len(per) == N_ITEMS, f"{f}: only {len(per)} items"
                a = agg(per)
                rows.append({"config": cfg, "seed": s, "T": T, "nfe": 8 * T,
                             "budget": m.budget, "width": m.width, "depth": m.depth,
                             "n_nonembed": m.n_nonembed, "c_layer_ms": m.c_layer_ms,
                             "latency_ms": 8 * T * m.depth * m.c_layer_ms,
                             **a,
                             # v1.0 error convention: WER as-is, SIM-o as 1-SIM
                             "err_wer": a["wer"], "err_sim": 1.0 - a["sim"],
                             "err_ut": -a["utmos"]})
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "runs_ext.csv"), index=False)
    print(f"[e1] runs_ext.csv: {len(df)} rows "
          f"({df.config.nunique()} configs x {df.seed.nunique()} seeds x {df['T'].nunique()} T)",
          flush=True)
    return df


def saturation(df: pd.DataFrame, t_ref: int) -> Dict:
    """v1.0's saturation_T definition (fit.py), with the reference T made explicit."""
    old = F.T_REF
    F.T_REF = t_ref
    try:
        out = F.saturation_T(df.rename(columns={"utmos": "ut"}))
    finally:
        F.T_REF = old
    return out


def refit_tau(df: pd.DataFrame, label: str, n_boot: int) -> Dict:
    """M_sep refit on `df`; run-level bootstrap CI on tau_wer. Machinery = v1.0 §5."""
    res = {"label": label, "n_rows": int(len(df)), "T_values": sorted(df["T"].unique().tolist())}
    for m in ("wer", "sim"):
        b = F.part_b(df, m)
        res[m] = {"M_sep": b["M_sep"]["params"], "aicc": b["M_sep"]["aicc"]}
    boot = F.bootstrap(df, n_boot=n_boot)
    for m in ("wer", "sim"):
        res[m]["tau"] = res[m]["M_sep"]["tau"]
        res[m]["tau_ci"] = F.ci(boot[f"{m}_tau"])
        res[m]["n_boot_ok"] = len(boot[f"{m}_tau"])
    res["delta_tau"] = F.ci(boot["delta_tau"])
    res["n_reps_ok"] = boot["n_reps_ok"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=F.N_BOOT)
    a = ap.parse_args()
    df = build_csv()
    expect = len(CONFIGS) * len(SEEDS) * len(T_ALL)
    if len(df) != expect:
        print(f"[e1] INCOMPLETE: {len(df)}/{expect} rows — refusing to test H-E1", flush=True)
        return

    sat64 = saturation(df, 64)
    # control: same 21 runs, same 200 items, but only the v1.0 T range
    sub = df[df["T"] <= 16]
    ext = refit_tau(df, "extended T<=64, 7 configs, 200 items", a.n_boot)
    ctl = refit_tau(sub, "control: T<=16, same 7 configs, same 200 items", a.n_boot)

    t_star = sat64["wer"]["pooled_T_star"]
    tau = ext["wer"]["tau"]
    inside = V1_TAU_CI[0] <= tau <= V1_TAU_CI[1]
    res = {
        "n_runs": int(df.groupby(["config", "seed"]).ngroups), "n_items": N_ITEMS,
        "T_values": T_ALL,
        "saturation_ref_T64": sat64,
        "refit_extended": ext, "refit_control_T16": ctl,
        "v1_0_tau_wer_ci": V1_TAU_CI,
        "H_E1": {
            "rule": "T*_WER (95% of the T=64 value) > 16 AND the refitted tau_WER lies "
                    "within the v1.0 95% CI [0.8248, 0.8513] (PREREGISTRATION-v1.1.md)",
            "T_star_wer": t_star, "part_a_T_star_gt_16": bool(t_star is not None and t_star > 16),
            "tau_wer_extended": tau, "tau_wer_extended_ci": ext["wer"]["tau_ci"],
            "part_b_tau_in_v1_ci": bool(inside),
            "supported": bool(t_star is not None and t_star > 16 and inside),
        },
        "T_star_sim": sat64["sim"]["pooled_T_star"],
    }
    with open(os.path.join(OUT, "e1_extended.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    print(json.dumps(res["H_E1"], indent=1), flush=True)
    print(f"[e1] T*_WER={t_star}  T*_SIM={res['T_star_sim']}  "
          f"tau_wer ext={tau:.4f} {ext['wer']['tau_ci']}  "
          f"ctl={ctl['wer']['tau']:.4f} {ctl['wer']['tau_ci']}", flush=True)
    piv = df.pivot_table(index="T", values=["wer", "sim", "utmos"], aggfunc="mean")
    print(piv.to_string(), flush=True)


if __name__ == "__main__":
    main()
