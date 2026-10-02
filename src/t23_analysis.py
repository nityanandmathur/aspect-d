"""H-T2 and H-T3 (PREREGISTRATION-v1.3).

**H-T2 — is the training-compute identity axis saturating, and is +0.0788 a seed
artifact?** The S0 ledger's largest identity gain came from three configs at one
seed. Primary: SIM-o(180k) − SIM-o(90k) on C3 seed 0, paired item bootstrap CI > 0.
Second lens: with three seeds per config at 90k, the 30k→90k gain replicates with a
run-level bootstrap CI > 0. Guardrail: an identity gain with a WER regression
> 2.0 points is a trade. Exploratory: SIM-o against log-steps on {30k, 90k, 180k}.

**H-T3 — is context a training limitation rather than a test-time dead end?**
S1 found prompts longer than the 3 s seen in training decouple identity from
content. This re-runs the same arm sweep on a model trained with per-item prompt
lengths from the same set. Primary: paired per-item SIM-o(9 s) − SIM-o(3 s) CI > 0
**and** WER(9 s) − WER(3 s) ≤ +2.0 points. Second lens: positive Spearman trend
**and** a monotone arm curve — non-monotonicity is exactly what made H-S1
DISCORDANT, so it is required here rather than assumed.

    python src/t23_analysis.py
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "artifacts-v1.3")
BOOT_RNG = 7331
N_BOOT = 2000
ARMS = [("1p5", 1.5), ("3p0", 3.0), ("6p0", 6.0), ("9p0", 9.0)]


def summ(path: str) -> Dict:
    return json.load(open(path))["summary"]


def items(path: str) -> Dict[str, Dict]:
    return {r["item"]: r for r in json.load(open(path))["items"]}


def boot(x: np.ndarray, n_boot: int, rng) -> List[float]:
    reps = np.array([x[rng.choice(len(x), len(x), True)].mean() for _ in range(n_boot)])
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def ht2(rng, n_boot: int) -> Dict:
    P = lambda root, r: os.path.join(REPO, "results", root, r, "synth_T16", "scores.json")
    res: Dict = {"hypothesis": "H-T2"}

    # ---- primary: 90k -> 180k on C3 seed 0, paired on items ----
    a, b = P("runs-v1.1", "C3_0_90k"), P("runs-v1.3", "C3_0_180k")
    if not (os.path.exists(a) and os.path.exists(b)):
        return {"hypothesis": "H-T2", "status": "INCOMPLETE"}
    A, B = items(a), items(b)
    common = sorted(set(A) & set(B))
    d = np.array([B[i]["sim"] - A[i]["sim"] for i in common])
    dw = np.array([B[i]["wer"] - A[i]["wer"] for i in common])
    ci = boot(d, n_boot, rng)
    res["primary_90k_to_180k"] = {
        "sim_90k": float(np.mean([A[i]["sim"] for i in common])),
        "sim_180k": float(np.mean([B[i]["sim"] for i in common])),
        "delta": float(d.mean()), "ci": ci, "n_items": len(common),
        "d_wer_points": float(100 * dw.mean()),
        "supported": bool(d.mean() > 0 and ci[0] > 0),
        "guardrail": "trade" if (d.mean() > 0 and 100 * dw.mean() > 2.0) else "within"}

    # ---- second lens: does the 30k->90k gain replicate over three seeds? ----
    per_run = []
    for c in ("C1", "C3", "C5"):
        for s in (0, 1, 2):
            p30 = P("runs", f"{c}_{s}")
            p90 = (P("runs-v1.1", f"{c}_0_90k") if s == 0
                   else P("runs-v1.3", f"{c}_{s}_90k"))
            if os.path.exists(p30) and os.path.exists(p90):
                per_run.append(summ(p90)["sim_mean"] - summ(p30)["sim_mean"])
    g = np.array(per_run)
    gci = boot(g, n_boot, rng) if len(g) else [float("nan")] * 2
    res["second_lens_30k_to_90k_replication"] = {
        "n_runs": int(len(g)), "delta": float(g.mean()) if len(g) else None,
        "ci": gci, "runs_positive": int((g > 0).sum()) if len(g) else 0,
        "single_seed_value_from_S0_ledger": 0.0788,
        "supported": bool(len(g) >= 6 and g.mean() > 0 and gci[0] > 0)}

    # ---- exploratory: shape against log steps ----
    pts = {}
    for lbl, p in (("30k", P("runs", "C3_0")), ("90k", P("runs-v1.1", "C3_0_90k")),
                   ("180k", P("runs-v1.3", "C3_0_180k"))):
        if os.path.exists(p):
            pts[lbl] = summ(p)["sim_mean"]
    if len(pts) == 3:
        x = np.log([30000, 90000, 180000])
        y = np.array([pts["30k"], pts["90k"], pts["180k"]])
        sl, ic = np.polyfit(x[:2], y[:2], 1)          # line through the first two
        res["exploratory_log_shape"] = {
            "sim_by_steps": pts,
            "predicted_180k_from_30k_90k_line": float(sl * x[2] + ic),
            "observed_180k": float(y[2]),
            "residual": float(y[2] - (sl * x[2] + ic)),
            "reading": "negative residual = saturating; ~0 = log-linear",
            "label": "EXPLORATORY, no decision rule (PREREG v1.3)"}
    p = res["primary_90k_to_180k"]["supported"]
    s2 = res["second_lens_30k_to_90k_replication"]["supported"]
    res["verdict"] = ("SUPPORTED" if p and s2 else "REFUTED" if not p and not s2
                      else "DISCORDANT")
    res["lenses_agree"] = bool(p == s2)
    return res


def ht3(rng, n_boot: int) -> Dict:
    base = os.path.join(REPO, "results", "runs", "C3_0")
    var = os.path.join(REPO, "results", "runs-v1.3", "C3_0_varprompt")
    res: Dict = {"hypothesis": "H-T3"}
    got = {}
    for tag, sec in ARMS:
        p = os.path.join(var, f"synth_s1ctx{tag}", "scores.json")
        q = os.path.join(base, f"synth_s1ctx{tag}", "scores.json")
        if not (os.path.exists(p) and os.path.exists(q)):
            return {"hypothesis": "H-T3", "status": "INCOMPLETE"}
        got[sec] = (items(p), items(q))
    secs = [s for _, s in ARMS]
    res["per_arm_varprompt"] = {str(s): {
        "sim": float(np.mean([r["sim"] for r in got[s][0].values()])),
        "wer": float(np.mean([r["wer"] for r in got[s][0].values()])),
        "degen": float(np.mean([r["degenerate"] for r in got[s][0].values()]))}
        for s in secs}
    res["per_arm_baseline_C3_0"] = {str(s): {
        "sim": float(np.mean([r["sim"] for r in got[s][1].values()])),
        "wer": float(np.mean([r["wer"] for r in got[s][1].values()]))} for s in secs}

    A9, A3 = got[9.0][0], got[3.0][0]
    common = sorted(set(A9) & set(A3))
    d = np.array([A9[i]["sim"] - A3[i]["sim"] for i in common])
    dw = np.array([A9[i]["wer"] - A3[i]["wer"] for i in common])
    ci = boot(d, n_boot, rng)
    wer_pts = float(100 * dw.mean())
    primary = bool(d.mean() > 0 and ci[0] > 0 and wer_pts <= 2.0)
    res["primary_9s_vs_3s"] = {"delta": float(d.mean()), "ci": ci,
                               "d_wer_points": wer_pts, "n_items": len(common),
                               "supported": primary}
    y = [res["per_arm_varprompt"][str(s)]["sim"] for s in secs]
    rho = float(spearmanr(secs, y).statistic)
    monotone = bool(all(y[i] <= y[i + 1] for i in range(len(y) - 1)))
    res["second_lens"] = {"spearman_rho": rho, "monotone": monotone,
                          "sim_curve": y,
                          "rule": "positive trend AND monotone (H-S1 was DISCORDANT "
                                  "precisely because the curve was non-monotone)",
                          "supported": bool(rho > 0 and monotone)}
    # rival: is the retrained model simply better?
    b3 = res["per_arm_baseline_C3_0"]["3.0"]
    v3 = res["per_arm_varprompt"]["3.0"]
    res["rival_just_a_better_model"] = {
        "check": "compare the two models at the 3 s arm, where neither is extrapolating",
        "baseline_sim": b3["sim"], "varprompt_sim": v3["sim"],
        "delta_at_3s": float(v3["sim"] - b3["sim"]),
        "reading": "a uniform improvement at 3 s is not evidence about context"}
    # rival: content-only fix
    res["rival_content_only_fix"] = {
        "check": "report the SIM and WER deltas separately at 9 s",
        "d_sim_9s_vs_3s": float(d.mean()), "d_wer_points_9s_vs_3s": wer_pts,
        "baseline_d_wer_points": float(
            100 * (np.mean([r["wer"] for r in got[9.0][1].values()])
                   - np.mean([r["wer"] for r in got[3.0][1].values()]))),
        "classification": "content-only fix" if abs(d.mean()) < 0.005 else "moves both"}
    s2 = res["second_lens"]["supported"]
    res["verdict"] = ("SUPPORTED" if primary and s2 else
                      "REFUTED" if not primary and not s2 else "DISCORDANT")
    res["lenses_agree"] = bool(primary == s2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    rng = np.random.default_rng(BOOT_RNG)
    os.makedirs(OUT, exist_ok=True)
    r2, r3 = ht2(rng, a.n_boot), ht3(rng, a.n_boot)
    json.dump({"H_T2": r2, "H_T3": r3}, open(os.path.join(OUT, "t23_training.json"), "w"),
              indent=1)

    print("=== H-T2: training-compute identity axis ===")
    if r2.get("status") == "INCOMPLETE":
        print("  INCOMPLETE")
    else:
        p = r2["primary_90k_to_180k"]
        print(f"  90k->180k (C3 s0): SIM {p['sim_90k']:.4f} -> {p['sim_180k']:.4f}  "
              f"delta {p['delta']:+.4f} CI [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]  "
              f"WER {p['d_wer_points']:+.2f} pts -> {'supported' if p['supported'] else 'NOT supported'}")
        s = r2["second_lens_30k_to_90k_replication"]
        print(f"  30k->90k over {s['n_runs']} runs: {s['delta']:+.4f} "
              f"CI [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] ({s['runs_positive']}/{s['n_runs']} positive) "
              f"vs the single-seed 0.0788 -> {'supported' if s['supported'] else 'NOT supported'}")
        if "exploratory_log_shape" in r2:
            e = r2["exploratory_log_shape"]
            print(f"  shape: {e['sim_by_steps']}  residual at 180k {e['residual']:+.4f} "
                  f"({e['reading']})")
        print(f"  VERDICT: H-T2 {r2['verdict']} (lenses agree: {r2['lenses_agree']})")

    print("\n=== H-T3: is context a training limitation? ===")
    if r3.get("status") == "INCOMPLETE":
        print("  INCOMPLETE")
    else:
        print(f"  {'arm':>6} {'SIM (var)':>10} {'WER (var)':>10} {'WER (base)':>11}")
        for _, s in ARMS:
            v = r3["per_arm_varprompt"][str(s)]
            b = r3["per_arm_baseline_C3_0"][str(s)]
            print(f"  {s:>5}s {v['sim']:>10.4f} {v['wer']:>10.4f} {b['wer']:>11.4f}")
        p = r3["primary_9s_vs_3s"]
        print(f"  primary 9s-3s: SIM {p['delta']:+.4f} CI [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}], "
              f"WER {p['d_wer_points']:+.2f} pts -> {'supported' if p['supported'] else 'NOT supported'}")
        l = r3["second_lens"]
        print(f"  lens 2: rho {l['spearman_rho']:+.3f}, monotone {l['monotone']} -> "
              f"{'supported' if l['supported'] else 'NOT supported'}")
        print(f"  rival (just a better model?): 3s arm {r3['rival_just_a_better_model']['delta_at_3s']:+.4f}")
        print(f"  VERDICT: H-T3 {r3['verdict']} (lenses agree: {r3['lenses_agree']})")


if __name__ == "__main__":
    main()
