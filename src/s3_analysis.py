"""S3 — rate-matched length conditioning (task-v2.md §4-S3, hypothesis H-S3).

Arm A: v1.0's corpus-median seconds-per-character (the frozen `synth_T16`).
Arm B: per-item rate measured from the item's own prompt clip (`synth_rate`).
Same text, same prompt, same sampler, same RNG keying — only target length moves.

Two lenses (§1.1):
  primary     paired per-run SIM-o(B) − SIM-o(A), run-level bootstrap 95 % CI > 0
  second lens per-item paired median difference, sign test p < 0.05, same direction
Guardrail: WER change within ±2.0 points, else reported as a trade.

Rivals computed, not argued: duration change altering how much audio is scoreable
(SIM stratified by duration quartile) and ASR length sensitivity (WER likewise).

    python src/s3_analysis.py
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import binomtest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "artifacts-v1.2")
RUNS = [f"C{i}_{s}" for i in range(1, 6) for s in (0, 1, 2)]
BOOT_RNG = 7331
N_BOOT = 2000


def load(run: str, tag: str) -> Dict[str, Dict]:
    f = os.path.join(REPO, "runs", run, f"synth_{tag}", "scores.json")
    return {r["item"]: r for r in json.load(open(f))["items"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()

    rows = []
    for r in RUNS:
        A, B = load(r, "T16"), load(r, "rate")
        common = sorted(set(A) & set(B))
        for i in common:
            rows.append({"run": r, "item": i,
                         "sim_A": A[i]["sim"], "sim_B": B[i]["sim"],
                         "wer_A": A[i]["wer"], "wer_B": B[i]["wer"],
                         "dur_A": A[i]["gen_seconds"], "dur_B": B[i]["gen_seconds"],
                         "degen_A": A[i]["degenerate"], "degen_B": B[i]["degenerate"]})
    df = pd.DataFrame(rows)
    df["d_sim"] = df.sim_B - df.sim_A
    df["d_wer"] = df.wer_B - df.wer_A
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "runs_s3.csv"), index=False)

    # ---- primary: paired per-run, run-level bootstrap ----
    per_run = df.groupby("run").agg(d_sim=("d_sim", "mean"), d_wer=("d_wer", "mean"))
    d = per_run.d_sim.values
    rng = np.random.default_rng(BOOT_RNG)
    reps = np.array([d[rng.choice(len(d), len(d), True)].mean() for _ in range(a.n_boot)])
    ci = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
    primary = bool(d.mean() > 0 and ci[0] > 0)

    # ---- second lens: per-item paired sign test ----
    per_item = df.groupby("item").d_sim.mean()
    npos = int((per_item > 0).sum())
    n = int((per_item != 0).sum())
    p = float(binomtest(npos, n, 0.5).pvalue)
    med = float(per_item.median())
    second = bool(p < 0.05 and np.sign(med) == np.sign(d.mean()) and med != 0)

    d_wer_pts = float(100 * per_run.d_wer.mean())
    res = {
        "n_runs": len(RUNS), "n_items": int(df.item.nunique()),
        "MEASURED": {
            "sim_A_median_rate": float(df.sim_A.mean()),
            "sim_B_rate_matched": float(df.sim_B.mean()),
            "primary": {"d_sim": float(d.mean()), "ci": ci,
                        "runs_positive": int((d > 0).sum()), "supported": primary},
            "second_lens_sign_test": {"per_item_median": med, "n_positive": npos,
                                      "n": n, "p_value": p, "supported": second},
            "guardrail_wer": {"delta_points": d_wer_pts, "threshold": 2.0,
                              "verdict": ("within guardrail" if abs(d_wer_pts) <= 2.0
                                          else "trade")},
            "wer_A": float(df.wer_A.mean()), "wer_B": float(df.wer_B.mean()),
            "degen_A": float(df.degen_A.mean()), "degen_B": float(df.degen_B.mean()),
            "duration_A_mean_s": float(df.dur_A.mean()),
            "duration_B_mean_s": float(df.dur_B.mean()),
        },
    }

    # ---- rivals ----
    df = df.assign(dur_q=pd.qcut(df.dur_A, 4, labels=["Q1", "Q2", "Q3", "Q4"]))
    strat = df.groupby("dur_q", observed=True).agg(
        d_sim=("d_sim", "mean"), d_wer=("d_wer", "mean"),
        durA=("dur_A", "mean"), durB=("dur_B", "mean")).reset_index()
    res["rivals"] = {
        "duration_change_alters_scoreable_audio": {
            "check": "SIM delta stratified by duration quartile of the baseline arm",
            "by_quartile": {str(r.dur_q): {"d_sim": float(r.d_sim),
                                           "dur_A": float(r.durA),
                                           "dur_B": float(r.durB)}
                            for _, r in strat.iterrows()},
            "mean_duration_shift_s": float(df.dur_B.mean() - df.dur_A.mean())},
        "asr_length_sensitivity": {
            "check": "WER delta stratified the same way",
            "by_quartile": {str(r.dur_q): float(r.d_wer) for _, r in strat.iterrows()}},
    }
    res["H_S3"] = {
        "rule": "primary: paired per-run SIM-o(rate-matched) - SIM-o(median-rate) CI > 0. "
                "second lens: per-item paired median sign test p < 0.05 in the same "
                "direction. Guardrail: WER within +/-2.0 points else 'trade' "
                "(PREREGISTRATION-v1.2.md H-S3).",
        "primary_supported": primary, "second_lens_supported": second,
        "verdict": ("SUPPORTED" if primary and second else
                    "REFUTED" if not primary and not second else "DISCORDANT"),
        "lenses_agree": bool(primary == second),
    }
    with open(os.path.join(OUT, "s3_rate.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    m = res["MEASURED"]
    print(f"[s3] SIM  median-rate {m['sim_A_median_rate']:.4f} -> rate-matched "
          f"{m['sim_B_rate_matched']:.4f}   delta {d.mean():+.4f} "
          f"CI [{ci[0]:+.4f}, {ci[1]:+.4f}] ({int((d>0).sum())}/15 runs)")
    print(f"[s3] lens 2: per-item median {med:+.5f}, {npos}/{n} positive, p={p:.3g}")
    print(f"[s3] guardrail WER {d_wer_pts:+.2f} points -> {m['guardrail_wer']['verdict']}")
    print(f"[s3] duration {m['duration_A_mean_s']:.2f}s -> {m['duration_B_mean_s']:.2f}s")
    print(f"[s3] VERDICT: H-S3 {res['H_S3']['verdict']} "
          f"(lenses agree: {res['H_S3']['lenses_agree']})")


if __name__ == "__main__":
    main()
