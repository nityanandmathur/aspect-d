"""H-T1 — does the S2 search result survive a stronger baseline? (PREREG-v1.3)

Program S measured everything on C-budget models at 30k steps. This repeats the
matched-NFE search-vs-refinement contrast on the two groups outside that scope
that already have checkpoints: the ten **budget-D** runs (276 M non-embedding
parameters) and the three **90k** runs (3× training compute), whose best member is
the strongest system this project has measured (SIM-o 0.4807).

Pre-registered: primary is the paired per-run ECAPA contrast at NFE 512 with
CI > 0 in **both groups separately**; second lens is per-item win rate > 50 % with
CI, in both groups. Rivals, each with a number: headroom shrinkage (every gain is
also reported as a fraction of that group's OWN headroom), selector/scorer family
overlap (the random / WavLM-selected / ECAPA-oracle decomposition repeated per
group — shared bias predicts ≈100 % of the oracle gap captured, and the C-budget
value was 44–46 %), and candidate-pool degeneracy.

    python src/t1_analysis.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(REPO, "results", "artifacts-v1.2")
OUT = os.path.join(REPO, "results", "artifacts-v1.3")
GROUPS = {
    "C_budget_30k (Program S reference)": [f"C{i}_{s}" for i in range(1, 6) for s in (0, 1, 2)],
    "budget_D_276M": [f"D{i}_{s}" for i in range(1, 6) for s in (0, 1)],
    "training_90k": ["C1_0_90k", "C3_0_90k", "C5_0_90k"],
}
TIERS = [(128, 16, 2), (256, 32, 4), (512, 64, 8)]
BOOT_RNG = 7331
N_BOOT = 2000


def load(subdir: str) -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(IN, subdir, "*.csv")))
    if not fs:
        raise SystemExit(f"[t1] no parts in {subdir}")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def boot_ci(x: np.ndarray, n_boot: int, rng) -> List[float]:
    reps = np.array([x[rng.choice(len(x), len(x), True)].mean() for _ in range(n_boot)])
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    rng = np.random.default_rng(BOOT_RNG)
    S = load("s2_parts")             # per-item search/refine rows
    C = load("s2_confound_parts")    # per-candidate rows
    s0 = json.load(open(os.path.join(IN, "identity_ledger.json")))["MEASURED"]
    sim_rt = s0["sim_roundtrip_mean"]

    res: Dict = {"sim_roundtrip_ceiling": sim_rt, "groups": {}}
    print(f"codec ceiling SIM_rt = {sim_rt:.4f}\n")
    hdr = (f"{'group':<34} {'mean T16':>9} {'headroom':>9} {'ΔECAPA@512':>11} "
           f"{'95% CI':>21} {'win%':>6} {'% hdrm':>7}")
    print(hdr)
    print("-" * len(hdr))

    for gname, runs in GROUPS.items():
        s = S[S.run.isin(runs)]
        if s.empty:
            print(f"{gname:<34}  (no data yet)")
            continue
        g: Dict = {"runs": runs, "n_runs": int(s.run.nunique())}
        # each group's own baseline and headroom: best refinement T=16 in-group
        base_by_run = s[(s.arm == "refine") & (s.nfe == 128)].groupby("run").ecapa.mean()
        best_in_group = float(base_by_run.max())
        mean_in_group = float(base_by_run.mean())
        # Two headrooms, kept separate on purpose. The frozen H-S0 rule defines
        # headroom against the BEST measured system; the search gain, however, is a
        # paired contrast averaged over runs. Dividing a mean-based gain by a
        # best-based headroom mixes bases -- the S0 selection error in miniature --
        # so the reported percentage uses the mean basis and the best basis is
        # carried alongside for the frozen-rule comparison.
        headroom_best = sim_rt - best_in_group
        headroom_mean = sim_rt - mean_in_group
        g["best_in_group_ecapa_T16"] = best_in_group
        g["mean_in_group_ecapa_T16"] = mean_in_group
        g["own_headroom_from_best"] = headroom_best
        g["own_headroom_from_mean"] = headroom_mean
        headroom = headroom_mean

        for nfe, T, K in TIERS:
            per_run = []
            for r in sorted(s.run.unique()):
                sr = s[(s.run == r) & (s.nfe == nfe) & (s.arm == "search")]
                rf = s[(s.run == r) & (s.nfe == nfe) & (s.arm == "refine")]
                if sr.empty or rf.empty:
                    continue
                per_run.append(sr.ecapa.mean() - rf.ecapa.mean())
            d = np.array(per_run)
            ci = boot_ci(d, a.n_boot, rng)
            piv = s[s.nfe == nfe].pivot_table(index=["run", "item"], columns="arm",
                                              values="ecapa")
            win = float((piv["search"] > piv["refine"]).mean())
            wci = boot_ci((piv["search"] > piv["refine"]).values.astype(float),
                          a.n_boot, rng)
            # vs the group's own deployable default (refinement T=16)
            base = s[(s.arm == "refine") & (s.nfe == 128)].ecapa.mean()
            srch = s[(s.arm == "search") & (s.nfe == nfe)].ecapa.mean()
            g[str(nfe)] = {
                "d_ecapa": float(d.mean()), "ci": ci,
                "runs_positive": int((d > 0).sum()), "n_runs": int(len(d)),
                "primary_supported": bool(d.mean() > 0 and ci[0] > 0),
                "win_rate": win, "win_rate_ci": wci,
                "second_lens_supported": bool(win > 0.5 and wci[0] > 0.5),
                "vs_own_T16_default": float(srch - base),
                "pct_of_own_headroom_mean_basis": float(100 * (srch - base) / headroom_mean)
                if headroom_mean > 0 else float("nan"),
                "pct_of_own_headroom_best_basis": float(100 * (srch - base) / headroom_best)
                if headroom_best > 0 else float("nan"),
                "search_wer": float(s[(s.arm == "search") & (s.nfe == nfe)].wer.mean()),
                "refine_wer": float(s[(s.arm == "refine") & (s.nfe == nfe)].wer.mean()),
                "search_degen": float(s[(s.arm == "search") & (s.nfe == nfe)].degenerate.mean()),
            }

        # rival: selector/scorer family overlap, per group
        c = C[C.run.isin(runs)]
        if not c.empty:
            g["confound"] = {"selector_scorer_r": float(np.corrcoef(c.wavlm, c.ecapa)[0, 1])}
            for nfe, T, K in TIERS:
                rr, ww, oo = [], [], []
                for r, gr in c.groupby("run"):
                    for i, gi in gr.groupby("item"):
                        sub = gi[gi.cand < K]
                        rr.append(sub.ecapa.mean())
                        ww.append(float(sub.ecapa.values[int(np.argmax(sub.wavlm.values))]))
                        oo.append(float(sub.ecapa.max()))
                rand, wsel, orac = np.mean(rr), np.mean(ww), np.mean(oo)
                g["confound"][str(nfe)] = {
                    "random_pick": float(rand), "wavlm_selected": float(wsel),
                    "ecapa_oracle": float(orac),
                    "selection_lift": float(wsel - rand),
                    "oracle_gap": float(orac - rand),
                    "share_of_oracle": float((wsel - rand) / (orac - rand))
                    if orac > rand else float("nan")}
        res["groups"][gname] = g
        t = g["512"]
        print(f"{gname:<34} {mean_in_group:>9.4f} {headroom_mean:>9.4f} {t['d_ecapa']:>+11.4f} "
              f"[{t['ci'][0]:+.4f}, {t['ci'][1]:+.4f}] {100*t['win_rate']:>5.1f}% "
              f"{t['pct_of_own_headroom_mean_basis']:>6.1f}%")

    # pre-registered verdict: both out-of-scope groups must support, both lenses
    oos = ["budget_D_276M", "training_90k"]
    have = [k for k in oos if k in res["groups"]]
    prim = all(res["groups"][k]["512"]["primary_supported"] for k in have) and len(have) == 2
    sec = all(res["groups"][k]["512"]["second_lens_supported"] for k in have) and len(have) == 2
    res["H_T1"] = {
        "rule": "primary: paired per-run ECAPA contrast at NFE 512 with CI > 0 in BOTH "
                "out-of-scope groups. second lens: per-item win rate > 50% with CI, both "
                "groups (PREREGISTRATION-v1.3.md H-T1).",
        "groups_evaluated": have,
        "primary_supported": bool(prim), "second_lens_supported": bool(sec),
        "verdict": ("SUPPORTED" if prim and sec else
                    "REFUTED" if not prim and not sec else "DISCORDANT")
        if len(have) == 2 else "INCOMPLETE",
        "lenses_agree": bool(prim == sec),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "t1_scope.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    print("\nselector/scorer overlap rival, share of the ECAPA oracle gap captured")
    print("(shared bias would predict ~100%; the C-budget value was 44-46%):")
    for gname, g in res["groups"].items():
        if "confound" in g:
            sh = [g["confound"][str(n)]["share_of_oracle"] for n, _, _ in TIERS]
            print(f"  {gname:<34} r={g['confound']['selector_scorer_r']:.3f}  "
                  f"{100*sh[0]:.1f}% / {100*sh[1]:.1f}% / {100*sh[2]:.1f}%")
    print(f"\n[t1] VERDICT: H-T1 {res['H_T1']['verdict']} "
          f"(lenses agree: {res['H_T1']['lenses_agree']})")


if __name__ == "__main__":
    main()
