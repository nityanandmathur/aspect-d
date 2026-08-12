"""Evaluate PREREGISTRATION-v1.5's CFG gate, and print the verdict before anyone reads
the per-arm table.

The rule this implements was fixed at commit 860a9581 on 2026-08-12, before any CFG
datum existed. An earlier draft of it passed 27 of 27 arms already on disk, which is
what a gate looks like when it cannot fail; the thresholds here are the repaired ones.
Nothing in this file may be tuned after seeing a result -- if a threshold turns out to
be wrong, the honest move is to say so in the feed, not to edit it.

    python src/v15_gate.py
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "artifacts-v1.5")
CFG_RUNS = [f"C{i}_{s}_cfg" for i in (1, 3, 5) for s in (0, 1, 2)]
WER_FLOOR, SIM_CEIL = 0.0344830, 0.5553695      # measured, frozen before v1.5
BOOT_RNG, N_BOOT = 7331, 2000
SEL_N = 100                                      # gamma-selection split, first 100 by id
WEIGHTS = ("0p5", "1p0", "2p0")


def _items(run: str, tag: str) -> Optional[Dict[str, Dict]]:
    p = os.path.join(REPO, "runs-v1.4", run, f"synth_{tag}", "scores.json")
    if not os.path.exists(p):
        return None
    return {d["item"]: d for d in json.load(open(p))["items"]}


def _mean(d: Dict[str, Dict], ids: List[str], k: str) -> float:
    return float(np.mean([d[i][k] for i in ids if i in d]))


def asymmetry(run: str, tag: str, ids: List[str]) -> Optional[float]:
    """A = (share of reachable WER range closed) / (share of reachable SIM range closed),
    both from this checkpoint's OWN T=1 anchors."""
    one, cur = _items(run, "T1"), _items(run, tag)
    if not (one and cur):
        return None
    w1, s1 = _mean(one, ids, "wer"), _mean(one, ids, "sim")
    wT, sT = _mean(cur, ids, "wer"), _mean(cur, ids, "sim")
    a = (w1 - wT) / (w1 - WER_FLOOR)
    b = (sT - s1) / (SIM_CEIL - s1)
    return a / b if b else None


def boot_ci(vals: List[float], n_boot: int = N_BOOT) -> List[float]:
    rng = np.random.default_rng(BOOT_RNG)
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    if len(v) < 2:
        return [float("nan"), float("nan")]
    reps = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    return [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]


def main():
    os.makedirs(OUT, exist_ok=True)
    have = [r for r in CFG_RUNS if _items(r, "T16")]
    if not have:
        print("[gate] no CFG checkpoints scored yet -- nothing to evaluate")
        return

    all_ids = sorted({i for r in have for i in (_items(r, "T16") or {})})
    sel_ids, eval_ids = all_ids[:SEL_N], all_ids[SEL_N:]
    res = {"runs": have, "n_sel": len(sel_ids), "n_eval": len(eval_ids),
           "preregistration": "860a95818b2f408ec3ee7fac37c39728166317b9"}

    # ---- gamma chosen on the SELECTION split only, primary arm fixed a priori ----
    sel_score = {}
    for w in WEIGHTS:
        d = [(_mean(_items(r, f"cfgp{w}") or {}, sel_ids, "sim")
              - _mean(_items(r, "T32") or {}, sel_ids, "sim"))
             for r in have if _items(r, f"cfgp{w}")]
        if d:
            sel_score[w] = float(np.mean(d))
    if not sel_score:
        print("[gate] prompt-CFG arms not scored yet -- nothing to evaluate")
        return
    w_star = max(sel_score, key=sel_score.get)
    res["gamma_selected_on_split"] = {"weight": w_star, "scores": sel_score}

    # ---- preconditions -------------------------------------------------------
    pre = {}
    rates = []
    for r in have:
        lg = os.path.join(REPO, "runs-v1.4", r, "train_log.jsonl")
        if os.path.exists(lg):
            rows = [json.loads(l) for l in open(lg) if l.strip()]
            if rows and "rate_p" in rows[-1]:
                rates.append((rows[-1]["rate_p"], rows[-1]["rate_t"]))
    pre["P1_dropout_fired"] = bool(rates) and all(
        0.085 <= rp <= 0.115 and 0.085 <= rt <= 0.115 for rp, rt in rates)
    pre["P1_rates"] = rates

    a_unguided = [asymmetry(r, "T16", eval_ids) for r in have]
    a_guided = [asymmetry(r, f"cfgp{w_star}", eval_ids) for r in have]
    pre["A_unguided_mean"] = float(np.nanmean([x for x in a_unguided if x]))
    pre["A_guided_mean"] = float(np.nanmean([x for x in a_guided if x]))
    pre["P3_anchors_comparable"] = bool(abs(pre["A_unguided_mean"] - 1.7784) <= 0.15)

    dsim, dwer = [], []
    for r in have:
        g, b = _items(r, f"cfgp{w_star}"), _items(r, "T32")
        if not (g and b):
            continue
        dsim.append(_mean(g, eval_ids, "sim") - _mean(b, eval_ids, "sim"))
        dwer.append(_mean(g, eval_ids, "wer") - _mean(b, eval_ids, "wer"))
    dsim_ci, dwer_ci = boot_ci(dsim), boot_ci(dwer)
    pre["P4_guidance_live"] = bool(abs(np.mean(dsim)) >= 0.005
                                   and not (dsim_ci[0] <= 0 <= dsim_ci[1]))
    res["preconditions"] = pre
    res["delta_sim"] = {"mean": float(np.mean(dsim)), "ci": dsim_ci}
    res["delta_wer"] = {"mean": float(np.mean(dwer)), "ci": dwer_ci}
    res["A_guided_ci"] = boot_ci(a_guided)

    # ---- the gate ------------------------------------------------------------
    if not (pre["P1_dropout_fired"] and pre["P3_anchors_comparable"]
            and pre["P4_guidance_live"]):
        verdict, why = "INCONCLUSIVE", "a precondition failed; no claim anywhere"
    elif res["A_guided_ci"][1] <= 1.05:
        verdict, why = ("REFUTATION", "guidance abolishes or reverses the asymmetry; "
                        "this MUST be reported in the paper")
    elif (np.mean(dsim) >= 0.010 and not (dsim_ci[0] <= 0 <= dsim_ci[1])
          and np.mean(dwer) <= 0.005):
        verdict, why = ("PARETO", "a free identity gain at iso-NFE; enters the paper "
                        "as a robustness subsection")
    else:
        verdict, why = ("NEITHER", "measured negative; ICLR notes and the feed, plus "
                        "one limitations sentence. Does NOT enter the paper.")
    res["VERDICT"], res["why"] = verdict, why

    with open(os.path.join(OUT, "cfg_gate.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    print(f"[gate] preregistration 860a9581, {len(have)} runs, "
          f"gamma={w_star} chosen on {len(sel_ids)} items, reported on {len(eval_ids)}")
    print(f"  P1 dropout fired      {pre['P1_dropout_fired']}")
    print(f"  P3 anchors comparable {pre['P3_anchors_comparable']} "
          f"(A_unguided {pre['A_unguided_mean']:.3f} vs 1.778)")
    print(f"  P4 guidance is live   {pre['P4_guidance_live']}")
    print(f"  delta SIM-o {np.mean(dsim):+.4f} {dsim_ci}")
    print(f"  delta WER   {np.mean(dwer):+.4f} {dwer_ci}")
    print(f"  A guided    {pre['A_guided_mean']:.3f} CI {res['A_guided_ci']}")
    print(f"\n  VERDICT: {verdict} -- {why}")


if __name__ == "__main__":
    main()
