"""Does the search result survive across independent speaker-encoder families?

The open limitation on the search contrast is that the selector (WavLM-large SV) and
the scorer (ECAPA-TDNN) share a lineage, correlating at item level r = 0.71--0.73.
Adding encoders is only useful if they are genuinely independent AND admissible.

**What the instrument gate actually rejects.** Gate G0(c) uses absolute cosine
thresholds (same-speaker median >= 0.50, cross-speaker <= 0.25). Measured on the 400
ground-truth eval items, all five encoders tested separate speakers almost perfectly,
but three of them fail the gate purely because their cosine scale is compressed:

    encoder            same    cross     gap     AUC    EER
    wavlm-large      0.7005   0.0338  0.6667  0.9947   1.25%   PASS
    ECAPA            0.6606   0.0598  0.6008  0.9958   2.00%   PASS
    base-plus-sv     0.9488   0.6601  0.2887  0.9834   5.75%   fail
    GE2E d-vector    0.8604   0.5911  0.2694  0.9901   4.50%   fail
    x-vector         0.9639   0.8953  0.0685  0.9906   5.25%   fail

An absolute-threshold gate is therefore scale-dependent: it rejects encoders for
where their cosines sit, not for whether they can tell speakers apart.

**Consequence for this test.** A *difference of mean cosines* is not comparable
across encoders with different scales, but a **per-item win rate** is: it asks only
whether an encoder ranks the search output above the refinement output for the same
item, which is invariant to any monotone rescaling of the similarity. We therefore
report the win rate for every encoder and treat the mean delta as scale-bound.

    python src/multi_encoder.py --runs C1_0            # shard
    python src/multi_encoder.py --aggregate
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from data import PROC_DIR

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "artifacts-v1.3")
PARTS = os.path.join(OUT, "multienc_parts")
RUNS = [f"C{i}_{s}" for i in range(1, 6) for s in (0, 1, 2)]
NFE, T_REF, K = 512, 64, 8
N_ITEMS = 200
BOOT_RNG = 7331


def _read(p: str) -> np.ndarray:
    return sf.read(p, dtype="float32")[0]


def encoders(device: str) -> Dict:
    from evaluate import Scorer
    from ecapa_gate import Ecapa
    from ge2e_gate import GE2E, XVect
    return {
        "wavlm_large": Scorer(device, use_utmos=False),          # the SELECTOR
        "ecapa": Ecapa(device),                                  # gated scorer
        "ge2e": GE2E(device),                                    # LSTM d-vector
        "xvect": XVect(device),                                  # TDNN x-vector
        "wavlm_base_plus": Scorer(device, use_utmos=False, force_sv_fallback=True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--runs", default=None)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(PARTS, exist_ok=True)

    if a.aggregate:
        fs = sorted(glob.glob(os.path.join(PARTS, "*.csv")))
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        df.to_csv(os.path.join(OUT, "multi_encoder.csv"), index=False)
        analyse(df, a.n_boot)
        return

    items = json.load(open(os.path.join(PROC_DIR, "eval_zs.json")))
    meta = {d["item"]: d for d in items}
    ids = sorted(meta)[:N_ITEMS]
    aud = os.path.join(PROC_DIR, "eval_audio")
    E = encoders(a.device)
    prompt_emb = {k: {i: e.embed([_read(os.path.join(aud, meta[i]["prompt_id"] + ".flac"))])[0]
                      for i in ids} for k, e in E.items()}

    todo = a.runs.split(",") if a.runs else RUNS
    for r in todo:
        sel = pd.read_csv(os.path.join(REPO, "results", "artifacts-v1.2", "s2_parts", f"runs__{r}.csv"))
        picks = sel[(sel.nfe == NFE) & (sel.arm == "search")].set_index("item")["pick"]
        ref = [_read(os.path.join(REPO, "results", "runs", r, f"synth_T{T_REF}", f"{i}.flac")) for i in ids]
        srch = [_read(os.path.join(REPO, "results", "runs", r, f"synth_bok{int(picks[i])}", f"{i}.flac"))
                for i in ids]
        rows = []
        for k, e in E.items():
            P = torch.stack([prompt_emb[k][i] for i in ids])
            dr = torch.nn.functional.cosine_similarity(e.embed(ref), P).numpy()
            ds = torch.nn.functional.cosine_similarity(e.embed(srch), P).numpy()
            for j, i in enumerate(ids):
                rows.append({"run": r, "encoder": k, "item": i,
                             "refine": float(dr[j]), "search": float(ds[j])})
        pd.DataFrame(rows).to_csv(os.path.join(PARTS, f"{r}.csv"), index=False)
        print(f"[menc] {r} done", flush=True)


def analyse(df: pd.DataFrame, n_boot: int):
    rng = np.random.default_rng(BOOT_RNG)
    gates = {}
    for nm, f in (("ecapa", "ecapa_gate.json"), ("ge2e", "ge2e_gate.json"),
                  ("xvect", "xvect_gate.json")):
        p = os.path.join(OUT, f) if nm != "ecapa" else os.path.join(
            REPO, "results", "artifacts-v1.2", "ecapa_gate.json")
        if os.path.exists(p):
            gates[nm] = json.load(open(p))
    res = {"nfe": NFE, "refinement_T": T_REF, "K": K, "encoders": {}}
    print(f"{'encoder':<18}{'win rate':>10}{'95% CI':>20}{'mean delta':>12}   role")
    for enc, g in df.groupby("encoder"):
        piv = g.pivot_table(index=["run", "item"], values=["refine", "search"])
        win = (piv["search"] > piv["refine"]).values.astype(float)
        reps = np.array([win[rng.choice(len(win), len(win), True)].mean()
                         for _ in range(n_boot)])
        ci = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
        per_run = g.groupby("run").apply(
            lambda x: x.search.mean() - x.refine.mean(), include_groups=False).values
        role = {"wavlm_large": "SELECTOR (circular)", "ecapa": "gated scorer",
                "ge2e": "independent family", "xvect": "independent family",
                "wavlm_base_plus": "independent scale"}[enc]
        res["encoders"][enc] = {
            "win_rate": float(win.mean()), "win_ci": ci,
            "mean_delta": float(per_run.mean()),
            "runs_positive": int((per_run > 0).sum()), "n_runs": int(len(per_run)),
            "win_ci_above_half": bool(ci[0] > 0.5), "role": role}
        print(f"{enc:<18}{100*win.mean():>9.1f}% [{100*ci[0]:>5.1f}, {100*ci[1]:>5.1f}]"
              f"{per_run.mean():>+12.4f}   {role}")
    agree = all(v["win_ci_above_half"] for v in res["encoders"].values())
    res["all_encoders_agree"] = bool(agree)
    res["n_independent_families_agreeing"] = int(sum(
        1 for k, v in res["encoders"].items()
        if v["win_ci_above_half"] and k not in ("wavlm_large",)))
    with open(os.path.join(OUT, "multi_encoder.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\n[menc] every encoder ranks search above refinement: {agree}")
    print(f"[menc] non-selector encoders agreeing: "
          f"{res['n_independent_families_agreeing']}/4")


if __name__ == "__main__":
    main()
