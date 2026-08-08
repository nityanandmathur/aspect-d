"""S2 confound check — is the search gain real, or selection-on-correlated-noise?

The concern (listed as an unexcluded rival in the Program-S closing entry, now
tested rather than stated): best-of-K *maximises* WavLM-SV, and ECAPA correlates
with WavLM-SV at r = 0.71 on this data. Maximising one therefore partly maximises
the other for reasons that need not be speaker identity — the winner's-curse
version of the base-plus-sv lesson.

Discriminating check: score **all K candidates** with ECAPA, then compare

    random      mean ECAPA over the K candidates   (what an arbitrary pick gets)
    wavlm-sel   ECAPA of the WavLM-SV argmax       (what S2 reported)
    ecapa-oracle ECAPA of the ECAPA argmax         (upper bound for any selector)

If wavlm-sel ≈ random, selection does no work and the reported gap is a property
of T=8 sampling, not of search. If wavlm-sel sits well above random, selection is
doing real work, and (oracle − wavlm-sel) bounds how much a perfectly-aligned
selector could add — i.e. how much of the effect could be shared-bias inflation.

Also decomposes the headline gap against the practical default (T=16), since
refinement's own ECAPA declines with T.

    python src/s2_confound.py [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from data import PROC_DIR

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "artifacts-v1.2")
RUNS = [f"C{i}_{s}" for i in range(1, 6) for s in (0, 1, 2)]
TIERS = [(128, 16, 2), (256, 32, 4), (512, 64, 8)]
N_ITEMS = 200
BOOT_RNG = 7331


def _read(p: str) -> np.ndarray:
    return sf.read(p, dtype="float32")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    PARTS = os.path.join(OUT, "s2_confound_parts")
    os.makedirs(PARTS, exist_ok=True)

    if a.aggregate:
        import glob
        df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(PARTS, "*.csv")))],
                       ignore_index=True)
        df.to_csv(os.path.join(OUT, "s2_confound.csv"), index=False)
        analyse(df)
        return

    from evaluate import Scorer
    from ecapa_gate import Ecapa
    todo = a.runs.split(",") if a.runs else RUNS
    meta = {d["item"]: d for d in json.load(open(os.path.join(PROC_DIR, "eval_zs.json")))}
    ids = sorted(meta)[:N_ITEMS]
    aud = os.path.join(PROC_DIR, "eval_audio")
    sel = Scorer(a.device, use_utmos=False)
    sco = Ecapa(a.device)
    e_p_sel = {i: sel.embed([_read(os.path.join(aud, meta[i]["prompt_id"] + ".flac"))])[0]
               for i in ids}
    e_p_sco = {i: sco.embed([_read(os.path.join(aud, meta[i]["prompt_id"] + ".flac"))])[0]
               for i in ids}

    for r in todo:
        rows = []
        for i in ids:
            wavs = [_read(os.path.join(REPO, "runs", r, f"synth_bok{c}", f"{i}.flac"))
                    for c in range(8)]
            Ew = sel.embed(wavs)
            Ee = sco.embed(wavs)
            w = torch.nn.functional.cosine_similarity(Ew, e_p_sel[i][None]).numpy()
            e = torch.nn.functional.cosine_similarity(Ee, e_p_sco[i][None]).numpy()
            for c in range(8):
                rows.append({"run": r, "item": i, "cand": c,
                             "wavlm": float(w[c]), "ecapa": float(e[c])})
        pd.DataFrame(rows).to_csv(os.path.join(PARTS, f"{r}.csv"), index=False)
        print(f"[s2c] {r} done", flush=True)


def analyse(df: pd.DataFrame):
    rng = np.random.default_rng(BOOT_RNG)
    ref = pd.read_csv(os.path.join(OUT, "runs_s2.csv"))
    res = {"selector_scorer_correlation": float(np.corrcoef(df.wavlm, df.ecapa)[0, 1]),
           "tiers": {}}
    print(f"selector/scorer correlation r = {res['selector_scorer_correlation']:.4f}\n")
    print(f"{'NFE':>5} {'random':>9} {'wavlm-sel':>10} {'ecapa-oracle':>13} "
          f"{'sel lift':>9} {'oracle gap':>11} {'sel/oracle':>11}")
    for nfe, T, K in TIERS:
        per_run = {"rand": [], "wsel": [], "orac": []}
        for r, g in df.groupby("run"):
            rr, ww, oo = [], [], []
            for i, gi in g.groupby("item"):
                sub = gi[gi.cand < K]
                rr.append(sub.ecapa.mean())
                ww.append(float(sub.ecapa.values[int(np.argmax(sub.wavlm.values))]))
                oo.append(float(sub.ecapa.max()))
            per_run["rand"].append(np.mean(rr))
            per_run["wsel"].append(np.mean(ww))
            per_run["orac"].append(np.mean(oo))
        rand, wsel, orac = (np.array(per_run[k]) for k in ("rand", "wsel", "orac"))
        lift = wsel - rand
        reps = np.array([lift[rng.choice(len(lift), len(lift), True)].mean()
                         for _ in range(2000)])
        ci = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
        frac = float(lift.mean() / (orac.mean() - rand.mean())) if orac.mean() > rand.mean() else float("nan")
        res["tiers"][str(nfe)] = {
            "random_pick": float(rand.mean()), "wavlm_selected": float(wsel.mean()),
            "ecapa_oracle": float(orac.mean()),
            "selection_lift": float(lift.mean()), "selection_lift_ci": ci,
            "oracle_gap": float(orac.mean() - rand.mean()),
            "fraction_of_oracle_captured": frac,
            "lift_ci_excludes_zero": bool(ci[0] > 0)}
        print(f"{nfe:>5} {rand.mean():>9.4f} {wsel.mean():>10.4f} {orac.mean():>13.4f} "
              f"{lift.mean():>+9.4f} {orac.mean()-rand.mean():>+11.4f} {100*frac:>10.1f}%")

    # decomposition against the practical default (refinement T=16)
    base = ref[(ref.arm == "refine") & (ref.nfe == 128)].ecapa.mean()
    print(f"\ndecomposition against the practical default (refinement T=16, ECAPA {base:.4f}):")
    for nfe, T, K in TIERS:
        s = ref[(ref.arm == "search") & (ref.nfe == nfe)].ecapa.mean()
        f = ref[(ref.arm == "refine") & (ref.nfe == nfe)].ecapa.mean()
        res["tiers"][str(nfe)].update({
            "search_vs_T16_default": float(s - base),
            "refine_vs_T16_default": float(f - base),
            "headline_gap": float(s - f)})
        print(f"  NFE {nfe}: search {s-base:+.4f} vs default | refinement {f-base:+.4f} vs "
              f"default | headline gap {s-f:+.4f}")
    with open(os.path.join(OUT, "s2_confound.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
