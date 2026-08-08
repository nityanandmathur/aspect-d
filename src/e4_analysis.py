"""E4 — budget-D grid and scale persistence (task-v1.md §4-E4, hypothesis H-E4).

Builds the 4-budget (A–D) Part-B surface by appending the newly trained budget-D
runs to the frozen v1.0 table, then:

  primary   (§9 H-E4)  Δτ > 0 on A–D with run-level bootstrap 95 % CI excluding 0
  secondary (§9 H-E4)  M_sep fit on A+B+C predicts budget-D config means at T=16
                       with MAPE ≤ 15 % and ≤ the N-only model's MAPE.
                       Explicitly a NEW test, not a redo of v1.0's failed H-D4.
  exploratory          d*(N) trend across the four budgets (no decision rule)

Only T ≤ 16 at 400 items enters the refit, so every row shares the v1.0 item
basis; the extended-T budget-D syntheses (200 items) are reported separately and
never mixed in. `artifacts/runs.csv` is read-only here — the 4-budget table is
written to `artifacts-v1.1/runs_4budget.csv` (task-v1.md §0.1).

    python src/e4_analysis.py
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
D_RUNS = [(f"D{i}", s) for i in range(1, 6) for s in (0, 1)]
T_FIT = [1, 2, 4, 8, 16]
N_ITEMS = 400


def d_rows() -> pd.DataFrame:
    """One row per (D config, seed, T) from the v1.1 run dirs, v1.0 column names."""
    clayer = {int(k): v["c_layer_ms"] for k, v in
              json.load(open(os.path.join(REPO, "artifacts", "c_layer.json")))["measured"].items()}
    rows, missing = [], []
    for cfg, seed in D_RUNS:
        d = os.path.join(REPO, "runs-v1.1", f"{cfg}_{seed}")
        rj = os.path.join(d, "run.json")
        if not os.path.exists(rj):
            missing.append(f"{cfg}_{seed}")
            continue
        run = json.load(open(rj))
        for T in T_FIT:
            sf = os.path.join(d, f"synth_T{T}", "scores.json")
            if not os.path.exists(sf):
                missing.append(f"{cfg}_{seed}/T{T}")
                continue
            s = json.load(open(sf))["summary"]
            if s["n_items"] != N_ITEMS:
                missing.append(f"{cfg}_{seed}/T{T}(n={s['n_items']})")
                continue
            w = run["width"]
            rows.append({"config": cfg, "seed": seed, "T": T, "nfe": 8 * T, "budget": "D",
                         "width": w, "depth": run["depth"], "heads": run["heads"],
                         "n_nonembed": run["nonembed_params"], "n_total": run["total_params"],
                         "lr": run["lr"], "val_loss": run.get("final_val_loss"),
                         "wer": s["wer_mean"], "sim": s["sim_mean"],
                         "utmos": s.get("utmos_mean"), "degen_rate": s["degen_rate"],
                         "crash_rate": s["crash_rate"],
                         "err_wer": s["wer_mean"], "err_sim": 1.0 - s["sim_mean"],
                         "err_ut": -(s.get("utmos_mean") or 0.0),
                         "c_layer_ms": clayer.get(w, np.nan),
                         "latency_ms": 8 * T * run["depth"] * clayer.get(w, np.nan),
                         "n_items": s["n_items"], "ckpt_step": 30000})
    df = pd.DataFrame(rows)
    df.attrs["missing"] = missing
    return df


def mape(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.mean(np.abs((pred - obs) / obs)))


def extrapolate(df4: pd.DataFrame, metric: str) -> Dict:
    """Fit on A+B+C, predict budget-D config means at T=16 (§9 H-E4 secondary)."""
    abc = df4[df4.budget != "D"]
    d16 = F.surface(df4[df4.budget == "D"], metric)
    d16 = d16[d16["T"] == 16]
    s_abc = F.surface(abc, metric)
    X, y, se = F.xy(s_abc)
    out = {}
    for form in ("M_sep", "M_N"):
        fitres = F.fit_form(form, X, y, se)
        if not fitres["ok"]:
            out[form] = {"ok": False}
            continue
        fn = F.FORMS[form][0]
        p = np.array([fitres["params"][k] for k in F.FORMS[form][1]])
        Xd = {"w": d16.w.values, "d": d16.d.values,
              "T": d16["T"].values.astype(float), "N": d16.N.values}
        pred = fn(p, Xd)
        out[form] = {"ok": True, "params": fitres["params"],
                     "mape": mape(pred, d16["mean"].values),
                     "per_config": {c: {"pred": float(pv), "obs": float(ov)}
                                    for c, pv, ov in zip(d16.config, pred, d16["mean"])}}
    if out["M_sep"].get("ok") and out["M_N"].get("ok"):
        out["supported"] = bool(out["M_sep"]["mape"] <= 0.15
                                and out["M_sep"]["mape"] <= out["M_N"]["mape"])
    return out


def d_star_by_budget(df4: pd.DataFrame) -> Dict:
    """Exploratory (§9 H-E4: no decision rule). Two things make a naive d*(N) trend
    misleading and are therefore recorded per budget:

    `censored` — d* sitting at the deepest shape the budget tested is a LOWER BOUND,
    not an interior optimum; the true optimum may be deeper than the grid goes.
    `margin_to_runner_up` — when the best and second-best depths are separated by
    less than seed noise, d* is a coin flip and must not be read as a trend.
    """
    t16 = df4[df4["T"] == 16].groupby(["budget", "depth"]).err_wer.mean().reset_index()
    out = {}
    for b, s in t16.groupby("budget"):
        s = s.sort_values("err_wer")
        best, runner = s.iloc[0], s.iloc[1]
        depths = sorted(t16[t16.budget == b].depth.unique())
        out[b] = {"d_star": int(best.depth),
                  "n_nonembed": float(df4[df4.budget == b].n_nonembed.mean()),
                  "depths_tested": [int(x) for x in depths],
                  "censored_at_max_depth": bool(int(best.depth) == max(depths)),
                  "runner_up_depth": int(runner.depth),
                  "margin_to_runner_up": float(runner.err_wer - best.err_wer),
                  "err_by_depth": {int(r.depth): float(r.err_wer) for _, r in s.iterrows()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=F.N_BOOT)
    a = ap.parse_args()
    v1 = pd.read_csv(os.path.join(REPO, "artifacts", "runs.csv"))
    d = d_rows()
    if d.attrs["missing"]:
        print(f"[e4] INCOMPLETE — missing: {d.attrs['missing'][:12]}"
              f"{' ...' if len(d.attrs['missing']) > 12 else ''}", flush=True)
        print(f"[e4] {len(d)} budget-D rows present; refusing to test H-E4", flush=True)
        return
    v1 = v1[v1["T"].isin(T_FIT)]
    df4 = pd.concat([v1[d.columns.intersection(v1.columns)], d], ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    df4.to_csv(os.path.join(OUT, "runs_4budget.csv"), index=False)
    print(f"[e4] 4-budget table: {len(df4)} rows, budgets {sorted(df4.budget.unique())}",
          flush=True)

    res = {"n_rows": int(len(df4)), "budgets": sorted(df4.budget.unique().tolist()),
           "T_grid": T_FIT, "n_items": N_ITEMS}
    for m in ("wer", "sim"):
        res[f"tau_{m}"] = F.part_b(df4, m).get("tau")
    res["delta_tau"] = res["tau_wer"] - res["tau_sim"]
    print(f"[e4] bootstrapping {a.n_boot} run-level replicates ...", flush=True)
    boot = F.bootstrap(df4, n_boot=a.n_boot)
    res["delta_tau_ci"] = F.ci(boot["delta_tau"])
    res["n_reps_ok"] = boot["n_reps_ok"]
    res["extrapolation_A_B_C_to_D"] = {m: extrapolate(df4, m) for m in ("wer", "sim")}
    res["d_star_by_budget"] = d_star_by_budget(df4)

    v1fits = json.load(open(os.path.join(REPO, "artifacts", "fits.json")))
    res["v1_0_three_budget"] = {"delta_tau": v1fits["decision"]["delta_tau"],
                                "delta_tau_ci": v1fits["decision"]["delta_tau_ci"]}
    ci = res["delta_tau_ci"]
    res["H_E4"] = {
        "rule": "primary: delta_tau > 0 on the 4-budget A-D surface with run-level "
                "bootstrap 95% CI excluding 0. secondary: M_sep fit on A+B+C predicts "
                "budget-D config means at T=16 with MAPE <= 15% and <= the N-only "
                "model's MAPE (NEW test, not a redo of v1.0 H-D4).",
        "primary_delta_tau": res["delta_tau"], "primary_ci": ci,
        "primary_supported": bool(res["delta_tau"] > 0 and ci and ci[0] > 0),
        "secondary_supported": res["extrapolation_A_B_C_to_D"]["wer"].get("supported"),
    }
    with open(os.path.join(OUT, "e4_scale.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    print(json.dumps(res["H_E4"], indent=1), flush=True)
    e = res["extrapolation_A_B_C_to_D"]["wer"]
    if e.get("M_sep", {}).get("ok"):
        print(f"[e4] extrapolation A+B+C -> D (WER @T=16): "
              f"M_sep MAPE {100*e['M_sep']['mape']:.1f}%  vs  "
              f"M_N MAPE {100*e['M_N']['mape']:.1f}%", flush=True)
    print(f"[e4] d* by budget: "
          f"{ {b: v['d_star'] for b, v in res['d_star_by_budget'].items()} }", flush=True)


if __name__ == "__main__":
    main()
