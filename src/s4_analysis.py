"""S4 — speaker-contrastive guidance, training-free (task-v2.md §4-S4, H-S4).

`logits ← logits_cond + γ·(logits_cond − logits_wrong-speaker)`, a second forward
pass per step on the same text with a deterministic wrong-speaker prompt
(item *i* takes the prompt of item *(i+7) mod N*). γ ∈ {0.5, 1.0, 2.0}, T=16,
15 C-budget runs × 400 items. γ = 0 is a strict no-op, verified against the
frozen v1.0 grids before any S4 datum existed.

Two lenses (§1.1):
  primary     ∃γ with paired per-run SIM-o(γ) − SIM-o(0) CI > 0
              AND WER(γ) ≤ WER(0) + 2.0 points AND DegenRate(γ) ≤ 2× DegenRate(0)
  second lens the SIM gain must also hold under ECAPA scoring (gated §1.7)

Rivals computed, not argued: guidance trading naturalness for scorer-specific
features (UTMOS reported; a SIM gain with UTMOS collapse > 0.5 is flagged, not
celebrated), and the wrong-speaker branch producing degenerate negatives.

    python src/s4_analysis.py
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "artifacts-v1.2")
RUNS = [f"C{i}_{s}" for i in range(1, 6) for s in (0, 1, 2)]
GAMMAS = [0.5, 1.0, 2.0]
BOOT_RNG = 7331
N_BOOT = 2000


def tag(g: float) -> str:
    return f"gam{str(g).replace('.', 'p')}"


def load(run: str, t: str) -> Dict[str, Dict]:
    f = os.path.join(REPO, "runs", run, f"synth_{t}", "scores.json")
    return {r["item"]: r for r in json.load(open(f))["items"]}


def summary(run: str, t: str) -> Dict:
    f = os.path.join(REPO, "runs", run, f"synth_{t}", "scores.json")
    return json.load(open(f))["summary"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ecapa-items", type=int, default=200,
                    help="items per run for the ECAPA second lens")
    a = ap.parse_args()

    rows = []
    for r in RUNS:
        base = load(r, "T16")
        for g in GAMMAS:
            arm = load(r, tag(g))
            for i in sorted(set(base) & set(arm)):
                rows.append({"run": r, "gamma": g, "item": i,
                             "sim0": base[i]["sim"], "sim": arm[i]["sim"],
                             "wer0": base[i]["wer"], "wer": arm[i]["wer"],
                             "ut0": base[i]["utmos"], "ut": arm[i]["utmos"],
                             "degen0": base[i]["degenerate"],
                             "degen": arm[i]["degenerate"]})
    df = pd.DataFrame(rows)
    df["d_sim"] = df.sim - df.sim0
    df["d_wer"] = df.wer - df.wer0
    df["d_ut"] = df.ut - df.ut0
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "runs_s4.csv"), index=False)

    rng = np.random.default_rng(BOOT_RNG)
    res = {"n_runs": len(RUNS), "gammas": GAMMAS, "per_gamma": {}}
    for g in GAMMAS:
        sub = df[df.gamma == g]
        pr = sub.groupby("run").agg(d_sim=("d_sim", "mean"), d_wer=("d_wer", "mean"),
                                    d_ut=("d_ut", "mean"), degen=("degen", "mean"),
                                    degen0=("degen0", "mean"))
        d = pr.d_sim.values
        reps = np.array([d[rng.choice(len(d), len(d), True)].mean()
                         for _ in range(a.n_boot)])
        ci = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
        wer_pts = float(100 * pr.d_wer.mean())
        degen, degen0 = float(pr.degen.mean()), float(pr.degen0.mean())
        res["per_gamma"][str(g)] = {
            "d_sim": float(d.mean()), "ci": ci,
            "runs_positive": int((d > 0).sum()),
            "d_wer_points": wer_pts,
            "d_utmos": float(pr.d_ut.mean()),
            "degen": degen, "degen_baseline": degen0,
            "sim_ci_positive": bool(d.mean() > 0 and ci[0] > 0),
            "wer_within_guardrail": bool(wer_pts <= 2.0),
            "degen_within_2x": bool(degen <= 2 * max(degen0, 1e-9)),
        }
        res["per_gamma"][str(g)]["primary_supported"] = bool(
            res["per_gamma"][str(g)]["sim_ci_positive"]
            and res["per_gamma"][str(g)]["wer_within_guardrail"]
            and res["per_gamma"][str(g)]["degen_within_2x"])

    winners = [g for g in GAMMAS if res["per_gamma"][str(g)]["primary_supported"]]
    res["primary_supported"] = bool(winners)
    res["winning_gamma"] = winners

    # ---- second lens: the SIM gain must hold under ECAPA (gated §1.7) ----
    second = None
    if winners:
        g = winners[0]
        from evaluate import Scorer
        from ecapa_gate import Ecapa
        import soundfile as sf
        from data import PROC_DIR
        meta = {d["item"]: d for d in json.load(open(os.path.join(PROC_DIR, "eval_zs.json")))}
        aud = os.path.join(PROC_DIR, "eval_audio")
        rd = lambda p: sf.read(p, dtype="float32")[0]
        sco = Ecapa(a.device)
        ids = sorted(meta)[:a.ecapa_items]
        e_p = {i: sco.embed([rd(os.path.join(aud, meta[i]["prompt_id"] + ".flac"))])[0]
               for i in ids}
        per_run = []
        for r in RUNS:
            vals = {}
            for t in ("T16", tag(g)):
                w = [rd(os.path.join(REPO, "runs", r, f"synth_{t}", f"{i}.flac")) for i in ids]
                E = sco.embed(w)
                vals[t] = torch.nn.functional.cosine_similarity(
                    E, torch.stack([e_p[i] for i in ids])).numpy().mean()
            per_run.append(float(vals[tag(g)] - vals["T16"]))
            print(f"[s4] ECAPA {r}: {per_run[-1]:+.4f}", flush=True)
        de = np.array(per_run)
        reps = np.array([de[rng.choice(len(de), len(de), True)].mean()
                         for _ in range(a.n_boot)])
        eci = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
        second = {"gamma": g, "d_ecapa": float(de.mean()), "ci": eci,
                  "runs_positive": int((de > 0).sum()),
                  "supported": bool(de.mean() > 0 and eci[0] > 0)}
    res["second_lens_ecapa"] = second

    # ---- rivals ----
    g0 = winners[0] if winners else GAMMAS[0]
    res["rivals"] = {
        "naturalness_traded_for_scorer_features": {
            "check": "UTMOS(γ) − UTMOS(0); a SIM gain with UTMOS collapse > 0.5 is flagged",
            "d_utmos_by_gamma": {str(g): res["per_gamma"][str(g)]["d_utmos"] for g in GAMMAS},
            "flagged": bool(res["per_gamma"][str(g0)]["d_utmos"] < -0.5)},
        "wrong_speaker_branch_degenerate": {
            "check": "DegenRate by γ against the γ=0 baseline",
            "degen_by_gamma": {str(g): res["per_gamma"][str(g)]["degen"] for g in GAMMAS},
            "degen_baseline": res["per_gamma"][str(GAMMAS[0])]["degen_baseline"]},
    }
    prim = res["primary_supported"]
    sec = bool(second and second["supported"])
    res["H_S4"] = {
        "rule": "primary: some γ with paired per-run SIM-o CI > 0, WER within +2.0 points, "
                "DegenRate <= 2x baseline. second lens: the SIM gain also holds under "
                "ECAPA scoring (PREREGISTRATION-v1.2.md H-S4).",
        "primary_supported": prim, "second_lens_supported": sec,
        "verdict": ("SUPPORTED" if prim and sec else
                    "REFUTED" if not prim and not sec else "DISCORDANT"),
        "lenses_agree": bool(prim == sec),
    }
    with open(os.path.join(OUT, "s4_guidance.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    print(f"\n{'γ':>5} {'ΔSIM':>9} {'95% CI':>22} {'runs+':>6} {'ΔWER pts':>9} "
          f"{'ΔUTMOS':>8} {'degen':>7}")
    for g in GAMMAS:
        p = res["per_gamma"][str(g)]
        print(f"{g:>5} {p['d_sim']:>+9.4f} [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}] "
              f"{p['runs_positive']:>4}/15 {p['d_wer_points']:>+9.2f} "
              f"{p['d_utmos']:>+8.3f} {p['degen']:>7.4f}", flush=True)
    if second:
        print(f"\n[s4] second lens ECAPA @γ={second['gamma']}: {second['d_ecapa']:+.4f} "
              f"CI [{second['ci'][0]:+.4f}, {second['ci'][1]:+.4f}] "
              f"({second['runs_positive']}/15) -> "
              f"{'supported' if second['supported'] else 'NOT supported'}")
    print(f"[s4] VERDICT: H-S4 {res['H_S4']['verdict']} "
          f"(lenses agree: {res['H_S4']['lenses_agree']})")


if __name__ == "__main__":
    main()
