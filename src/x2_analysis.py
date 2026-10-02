"""Does the floor-referenced headline survive 3x training compute, and more refinement?

Two robustness questions the main grid cannot answer, because every model in it is
trained for the same 30k steps and swept to the same T.

X2 -- nine runs (three configs x three seeds) retrained to 90k steps and swept over the
same T grid at the same 400 items. If refinement's identity plateau were an artefact of
under-training, a 3x-better model should close more of the identity range. The floors do
not move: the ASR floor is a property of the recordings and the codec ceiling a property
of Mimi, so the same denominators apply.

X7 -- the largest budget pushed to T=128, twice the extended grid, to check that the
plateau is a plateau and not the start of another descent.

    python src/x2_analysis.py
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

import fit as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "artifacts-v1.4")
BOOT_RNG, N_BOOT = 7331, 2000
# width/depth come from each run's own run.json -- never hardcoded, since guessing
# them silently changes N = 12*d*w^2 and therefore the fit


def _collect(pattern: str, roots: List[str]) -> pd.DataFrame:
    rows = []
    for root in roots:
        for sdir in sorted(glob.glob(os.path.join(REPO, "results", root, pattern, "synth_T*"))):
            sf = os.path.join(sdir, "scores.json")
            if not os.path.exists(sf):
                continue
            rd = os.path.dirname(sdir)
            run = os.path.basename(rd)
            s = json.load(open(sf))["summary"]
            rj = json.load(open(os.path.join(rd, "run.json")))
            w, d = int(rj["width"]), int(rj["depth"])
            rows.append({"run": run, "config": rj["config"],
                         "seed": int(run.split("_")[1]),
                         "T": int(os.path.basename(sdir).split("T")[1]),
                         "width": w, "depth": d,
                         "n_nonembed": 12 * d * w * w,
                         "budget": rj["config"][0],
                         "wer": s["wer_mean"], "sim": s["sim_mean"],
                         "wer_se": s.get("wer_se", np.nan),
                         "sim_se": s.get("sim_se", np.nan),
                         "err_wer": s["wer_mean"],
                         "err_sim": 1.0 - s["sim_mean"]})
    return pd.DataFrame(rows)


def _floors() -> Dict:
    g0 = json.load(open(os.path.join(REPO, "results", "artifacts", "g0c_groundtruth.json")))
    led = json.load(open(os.path.join(REPO, "results", "artifacts-v1.2",
                                      "identity_ledger.json")))["MEASURED"]
    return {"wer": float(g0["wer_mean_item"]), "sim": float(led["sim_roundtrip_mean"])}


def gap_closed(df: pd.DataFrame, n_boot: int = N_BOOT) -> Dict:
    fl = _floors()
    rng = np.random.default_rng(BOOT_RNG)
    piv = {r: g.set_index("T") for r, g in df.groupby("run")}
    runs = sorted(piv)

    def frac(sub, T):
        w1 = np.mean([piv[r].wer[1] for r in sub])
        wT = np.mean([piv[r].wer[T] for r in sub])
        s1 = np.mean([piv[r].sim[1] for r in sub])
        sT = np.mean([piv[r].sim[T] for r in sub])
        a = (w1 - wT) / (w1 - fl["wer"])
        b = (sT - s1) / (fl["sim"] - s1)
        return {"wer": a, "sim": b, "ratio": a / b}

    out = {}
    for T in sorted({t for r in runs for t in piv[r].index if t > 1}):
        if not all(T in piv[r].index for r in runs):
            continue
        pt = frac(runs, T)
        reps = [frac([runs[i] for i in rng.integers(0, len(runs), len(runs))], T)
                for _ in range(n_boot)]
        out[int(T)] = {k: {"point": pt[k],
                           "ci": [float(np.percentile([r[k] for r in reps], 2.5)),
                                  float(np.percentile([r[k] for r in reps], 97.5))]}
                       for k in pt}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}

    # ---------------- X2: 3x training compute --------------------------------
    d90 = _collect("*_90k", ["runs-v1.1", "runs-v1.3"])
    d90.to_csv(os.path.join(OUT, "runs_90k.csv"), index=False)
    g90 = gap_closed(d90)
    base = json.load(open(os.path.join(OUT, "analysis.json")))["B_gap_closed"]["by_T"]
    res["X2_gap_closed_90k"] = {
        "n_runs": int(d90.run.nunique()), "configs": sorted(d90.config.unique()),
        "by_T": g90,
        "note": "same floors as the 30k grid: they are properties of the recordings "
                "and of the codec, not of the model"}

    # tau at 90k, on the windows the 30k grid also supports
    wins = {}
    for hi in (16, 32, 64):
        s = d90[d90["T"] <= hi]
        if s["T"].nunique() < 4:
            continue
        tw = F.part_b(s, "wer").get("tau")
        ts = F.part_b(s, "sim").get("tau")
        wins[f"T<={hi}"] = {"tau_wer": tw, "tau_sim": ts, "delta_tau": tw - ts}
    res["X2_range_90k"] = wins

    # ---------------- X7: T = 128 on the largest budget ----------------------
    d128 = _collect("C5_*", ["runs"])
    d128 = d128[d128.run.str.match(r"C5_\d$")]
    res["X7_T128"] = {
        "runs": sorted(d128.run.unique().tolist()),
        "curve": {int(T): {"wer": float(g.wer.mean()), "sim": float(g.sim.mean())}
                  for T, g in d128.groupby("T")},
    }
    if 128 in set(d128["T"]) and 64 in set(d128["T"]):
        c = res["X7_T128"]["curve"]
        res["X7_T128"]["delta_64_to_128"] = {
            "wer": c[128]["wer"] - c[64]["wer"], "sim": c[128]["sim"] - c[64]["sim"]}

    with open(os.path.join(OUT, "x2_x7.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    print(f"X2  {res['X2_gap_closed_90k']['n_runs']} runs at 90k "
          f"({', '.join(res['X2_gap_closed_90k']['configs'])}), 3x training compute")
    print(f"{'T':>5} {'intelligibility':>26} {'identity':>26} {'ratio':>8}   (30k ratio)")
    for T in sorted(g90):
        e = g90[T]
        b30 = base.get(str(T))
        r30 = f"{b30['ratio_wer_over_sim']['point']:.2f}x" if b30 else "--"
        print(f"{T:>5} {100*e['wer']['point']:>9.1f}% "
              f"[{100*e['wer']['ci'][0]:>5.1f},{100*e['wer']['ci'][1]:>6.1f}]"
              f"{100*e['sim']['point']:>11.1f}% "
              f"[{100*e['sim']['ci'][0]:>5.1f},{100*e['sim']['ci'][1]:>6.1f}]"
              f"{e['ratio']['point']:>7.2f}x   {r30:>8}")
    print("\nX2  delta-tau at 90k:",
          {k: round(v["delta_tau"], 4) for k, v in wins.items()})
    print("\nX7  C5 at T=128:", {k: {m: round(x, 4) for m, x in v.items()}
                                 for k, v in sorted(res["X7_T128"]["curve"].items())
                                 if k >= 16})
    if "delta_64_to_128" in res["X7_T128"]:
        dd = res["X7_T128"]["delta_64_to_128"]
        print(f"    T=64 -> 128: WER {dd['wer']:+.4f}, SIM {dd['sim']:+.4f}")


if __name__ == "__main__":
    main()
