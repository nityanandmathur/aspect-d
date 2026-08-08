"""S0 — identity ledger + codec ceiling (task-v2.md §4-S0, hypothesis H-S0).

Two parts.

**Ceiling.** SIM_rt = mean SIM-o of a Mimi encode→decode roundtrip of each
ground-truth TARGET utterance, scored against the ORIGINAL prompt waveform —
i.e. the best speaker similarity any system built on this codec could reach if
its token prediction were perfect. Headroom h = SIM_rt − best measured system
SIM-o (any budget, T=16) classifies the ceiling by the frozen rule.

**Ledger.** Absolute SIM-o gain per competing axis, assembled from EXISTING
artifacts with no new synthesis, so every later "X buys identity" claim can be
read against what the other axes buy (§1.5).

Both pre-registered rivals are computed here, not argued:
  * roundtrip favouring the reference recording conditions → the roundtrip of a
    DIFFERENT same-speaker utterance is scored against the same prompt;
  * scorer saturation near 1.0 → the same-speaker GT baseline's distance from
    1.0 is reported from the frozen G0(c) artifact.

Scoring uses the v1.0 primary SV model only (wavlm-large, seed-tts-eval), which
passes gate G0(c); base-plus-sv is excluded everywhere (§8).

    python src/s0_ledger.py [--device cuda:0]
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

from data import PROC_DIR, SR

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "artifacts-v1.2")
N_LEVELS = 8


def _read(path: str) -> np.ndarray:
    w, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"{path} sr={sr}"
    return w


def roundtrip(mimi, wavs: List[np.ndarray], device, batch: int = 8) -> List[np.ndarray]:
    """Encode to 8 Mimi codebooks and decode back — the codec's own upper bound."""
    out = []
    for s in range(0, len(wavs), batch):
        chunk = wavs[s:s + batch]
        for w in chunk:
            x = torch.from_numpy(np.asarray(w, dtype=np.float32))[None, None].to(device)
            with torch.no_grad():
                codes = mimi.encode(x, num_quantizers=N_LEVELS).audio_codes
                y = mimi.decode(codes).audio_values[0, 0].float().cpu().numpy()
            out.append(y)
    return out


def sim(scorer, a: List[np.ndarray], b: List[np.ndarray]) -> np.ndarray:
    ea, eb = scorer.embed(a), scorer.embed(b)
    return torch.nn.functional.cosine_similarity(ea, eb).numpy()


def best_system_sim() -> Dict:
    """Best measured system SIM-o at T=16 over every budget, incl. the v1.1 D grid."""
    v1 = pd.read_csv(os.path.join(REPO, "artifacts", "runs.csv"))
    frames = [v1[v1["T"] == 16][["config", "budget", "sim"]]]
    p4 = os.path.join(REPO, "artifacts-v1.1", "runs_4budget.csv")
    if os.path.exists(p4):
        d4 = pd.read_csv(p4)
        frames.append(d4[(d4["T"] == 16) & (d4.budget == "D")][["config", "budget", "sim"]])
    allc = pd.concat(frames, ignore_index=True)
    g = allc.groupby(["budget", "config"]).sim.mean().reset_index()
    best = g.loc[g.sim.idxmax()]
    return {"config": best.config, "budget": best.budget, "sim": float(best.sim),
            "per_budget_best": {b: {"config": s.loc[s.sim.idxmax(), "config"],
                                    "sim": float(s.sim.max())}
                                for b, s in g.groupby("budget")}}


def ledger() -> Dict:
    """Absolute SIM-o gain per axis, from existing artifacts only (§4-S0)."""
    out: Dict = {}
    v1 = pd.read_csv(os.path.join(REPO, "artifacts", "runs.csv"))

    # steps: T=1 -> T=16, v1.0 grid
    t = v1.groupby("T").sim.mean()
    out["steps_T1_to_T16"] = {"from": float(t[1]), "to": float(t[16]),
                              "gain": float(t[16] - t[1]), "source": "artifacts/runs.csv"}

    # shape at fixed N: within-budget spread at T=16 (max - min over configs)
    s16 = v1[v1["T"] == 16].groupby(["budget", "config"]).sim.mean().reset_index()
    spread = {b: float(g.sim.max() - g.sim.min()) for b, g in s16.groupby("budget")}
    out["shape_at_fixed_N"] = {"per_budget_spread": spread,
                               "gain": float(max(spread.values())),
                               "source": "artifacts/runs.csv (best-minus-worst shape, T=16)"}

    # parameters N: best config of the smallest budget -> best of the largest
    p4 = os.path.join(REPO, "artifacts-v1.1", "runs_4budget.csv")
    if os.path.exists(p4):
        d4 = pd.read_csv(p4)
        g = d4[d4["T"] == 16].groupby(["budget", "config"]).sim.mean().reset_index()
        per = {b: float(x.sim.max()) for b, x in g.groupby("budget")}
        lo, hi = min(per), max(per)
        out["parameters_N"] = {"from_budget": lo, "to_budget": hi,
                               "from": per[lo], "to": per[hi],
                               "gain": float(per[hi] - per[lo]),
                               "per_budget_best": per,
                               "source": "artifacts-v1.1/runs_4budget.csv (best config per budget, T=16)"}

    # training compute: 30k -> 90k on C1/C3/C5 seed 0
    rows30, rows90 = [], []
    for c in ("C1", "C3", "C5"):
        f30 = os.path.join(REPO, "runs", f"{c}_0", "synth_T16", "scores.json")
        f90 = os.path.join(REPO, "runs-v1.1", f"{c}_0_90k", "synth_T16", "scores.json")
        if os.path.exists(f30) and os.path.exists(f90):
            rows30.append(json.load(open(f30))["summary"]["sim_mean"])
            rows90.append(json.load(open(f90))["summary"]["sim_mean"])
    if rows30:
        out["training_compute_30k_to_90k"] = {
            "from": float(np.mean(rows30)), "to": float(np.mean(rows90)),
            "gain": float(np.mean(rows90) - np.mean(rows30)),
            "source": "runs/C*_0 vs runs-v1.1/C*_0_90k, T=16"}

    # allocation at matched NFE=32: fine -> uniform -> coarse
    e3 = os.path.join(REPO, "artifacts-v1.1", "e3_nfe.json")
    if os.path.exists(e3):
        ps = json.load(open(e3))["per_schedule"]
        best = max(ps, key=lambda k: ps[k]["sim"])
        worst = min(ps, key=lambda k: ps[k]["sim"])
        out["allocation_at_matched_NFE"] = {
            "from": ps[worst]["sim"], "to": ps[best]["sim"],
            "from_schedule": worst, "to_schedule": best,
            "gain": float(ps[best]["sim"] - ps[worst]["sim"]),
            "per_schedule": {k: v["sim"] for k, v in ps.items()},
            "source": "artifacts-v1.1/e3_nfe.json"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--items", type=int, default=400)
    a = ap.parse_args()
    from transformers import MimiModel
    from evaluate import Scorer

    items = json.load(open(os.path.join(PROC_DIR, "eval_zs.json")))[:a.items]
    aud = os.path.join(PROC_DIR, "eval_audio")
    dev = torch.device(a.device)
    print(f"[s0] loading Mimi + the v1.0 primary SV model ({len(items)} items) ...", flush=True)
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(dev).eval()
    sc = Scorer(a.device, use_utmos=False)
    assert "wavlm_large" in sc.sim_model_name, f"wrong SV model: {sc.sim_model_name}"

    prompts = [_read(os.path.join(aud, it["prompt_id"] + ".flac")) for it in items]
    targets = [_read(os.path.join(aud, it["target_id"] + ".flac")) for it in items]

    print("[s0] roundtripping ground-truth targets through Mimi ...", flush=True)
    rt = roundtrip(mimi, targets, dev)
    sim_rt = sim(sc, rt, prompts)
    sim_gt = sim(sc, targets, prompts)          # un-roundtripped GT target vs prompt

    # RIVAL 1: does the roundtrip simply inherit the reference recording conditions?
    # score the roundtrip of a DIFFERENT same-speaker utterance against the same prompt.
    by_spk: Dict[str, List[int]] = {}
    for i, it in enumerate(items):
        by_spk.setdefault(it["speaker"], []).append(i)
    other_idx, have_other = [], []
    for i, it in enumerate(items):
        pool = [j for j in by_spk[it["speaker"]] if j != i]
        if pool:
            other_idx.append(pool[0])
            have_other.append(i)
    sim_other = np.full(len(items), np.nan)
    if have_other:
        print(f"[s0] rival 1: roundtrip of a different same-speaker utterance "
              f"({len(have_other)} items have one) ...", flush=True)
        rt_other = roundtrip(mimi, [targets[j] for j in other_idx], dev)
        vals = sim(sc, rt_other, [prompts[i] for i in have_other])
        for k, i in enumerate(have_other):
            sim_other[i] = vals[k]

    best = best_system_sim()
    h_mean = float(sim_rt.mean() - best["sim"])

    # SECOND LENS: per-item paired (roundtrip - best-system) distribution. The
    # per-item best-system SIM comes from that config's frozen per-item rows.
    bestf = None
    for root in ("runs", "runs-v1.1"):
        for seed in (0, 1, 2):
            p = os.path.join(REPO, root, f"{best['config']}_{seed}", "synth_T16", "scores.json")
            if os.path.exists(p):
                bestf = p
                break
        if bestf:
            break
    per_item_best = {}
    if bestf:
        for r in json.load(open(bestf))["items"]:
            per_item_best[r["item"]] = r["sim"]
    paired = np.array([sim_rt[i] - per_item_best.get(it["item"], np.nan)
                       for i, it in enumerate(items)], float)

    def classify(h: float) -> str:
        return ("near-ceiling" if h <= 0.05 else
                "moderate headroom" if h <= 0.15 else "large headroom")

    g0c = json.load(open(os.path.join(REPO, "artifacts", "g0c_groundtruth.json")))
    res = {
        "n_items": len(items),
        "sim_model": sc.sim_model_name,
        "MEASURED": {
            "sim_roundtrip_mean": float(sim_rt.mean()),
            "sim_roundtrip_median": float(np.median(sim_rt)),
            "sim_gt_no_roundtrip_mean": float(sim_gt.mean()),
            "codec_cost_gt_minus_roundtrip": float(sim_gt.mean() - sim_rt.mean()),
            "best_system": best,
            "headroom_mean": h_mean,
            "headroom_median_paired": float(np.nanmedian(paired)),
            "classification_mean": classify(h_mean),
            "classification_median": classify(float(np.nanmedian(paired))),
        },
        "rivals": {
            "roundtrip_inherits_reference_conditions": {
                "check": "roundtrip of a DIFFERENT same-speaker utterance vs the same prompt",
                "n": int(len(have_other)),
                "sim_mean": float(np.nanmean(sim_other)),
                "vs_same_utterance_roundtrip": float(np.nanmean(sim_other) - sim_rt.mean()),
            },
            "scorer_saturation_near_one": {
                "check": "same-speaker GT baseline distance from 1.0 (frozen G0(c))",
                "same_speaker_median": g0c["sim_same_median"],
                "distance_from_one": float(1.0 - g0c["sim_same_median"]),
                "cross_speaker_median": g0c["sim_cross_median"],
            },
        },
        "ledger": ledger(),
    }
    res["H_S0"] = {
        "rule": "h = SIM_rt - best measured system SIM-o (any budget, T=16); "
                "h<=0.05 near-ceiling; 0.05<h<=0.15 moderate headroom; h>0.15 large "
                "headroom. Second lens: the classification must hold for the per-item "
                "paired median as well as the mean (PREREGISTRATION-v1.2.md H-S0).",
        "headroom_mean": h_mean,
        "headroom_median": float(np.nanmedian(paired)),
        "classification": res["MEASURED"]["classification_mean"],
        "lenses_agree": bool(res["MEASURED"]["classification_mean"]
                             == res["MEASURED"]["classification_median"]),
        "verdict": (res["MEASURED"]["classification_mean"]
                    if res["MEASURED"]["classification_mean"]
                    == res["MEASURED"]["classification_median"] else "DISCORDANT"),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "identity_ledger.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    np.save(os.path.join(OUT, "s0_per_item_sim.npy"),
            np.vstack([sim_rt, sim_gt, sim_other, paired]))

    m = res["MEASURED"]
    print(f"\n[s0] SIM_rt (roundtrip GT target vs prompt) = {m['sim_roundtrip_mean']:.4f} "
          f"(median {m['sim_roundtrip_median']:.4f})", flush=True)
    print(f"[s0] GT target, no roundtrip               = {m['sim_gt_no_roundtrip_mean']:.4f} "
          f"-> codec costs {m['codec_cost_gt_minus_roundtrip']:.4f}", flush=True)
    print(f"[s0] best measured system                  = {best['sim']:.4f} "
          f"({best['config']}, budget {best['budget']})", flush=True)
    print(f"[s0] HEADROOM mean {h_mean:+.4f} -> {m['classification_mean']}; "
          f"paired median {res['H_S0']['headroom_median']:+.4f} -> "
          f"{m['classification_median']}", flush=True)
    print(f"[s0] VERDICT: {res['H_S0']['verdict']}", flush=True)
    print("\n[s0] identity ledger (absolute SIM-o gain per axis):", flush=True)
    for k, v in sorted(res["ledger"].items(), key=lambda kv: -kv[1]["gain"]):
        print(f"    {k:<32} {v['gain']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
