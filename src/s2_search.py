"""S2 — test-time search vs refinement at matched NFE (task-v2.md §4-S2, H-S2).

Matched-NFE tiers (NFE = 8·T per candidate):

    NFE 128   refinement T=16    vs   best-of-K  K=2 at T=8
    NFE 256   refinement T=32    vs   best-of-K  K=4 at T=8
    NFE 512   refinement T=64    vs   best-of-K  K=8 at T=8

**Selection ≠ scoring (§1.8, no exceptions).** Candidates are selected by
WavLM-SV cosine to the prompt; the selected candidate is then scored by ECAPA,
which passed the §1.7 instrument gate on ground truth (same 0.6606 ≥ 0.50,
cross 0.0598 ≤ 0.25). The selector never scores; the scorer never selects.

Refinement arms reuse the frozen audio already on disk (v1.0 T=16, E1 T=32/64),
re-scored with ECAPA on the same 200 items so both sides of every tier are
measured by the same instrument.

    python src/s2_search.py [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import soundfile as sf
import torch

from data import PROC_DIR, SR

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "artifacts-v1.2")
RUNS = [f"C{i}_{s}" for i in range(1, 6) for s in (0, 1, 2)]
TIERS = [(128, 16, 2), (256, 32, 4), (512, 64, 8)]
N_ITEMS = 200
BOOT_RNG = 7331
N_BOOT = 2000


def _read(p: str) -> np.ndarray:
    w, sr = sf.read(p, dtype="float32")
    return w if sr == SR else w


def item_ids() -> List[str]:
    items = json.load(open(os.path.join(PROC_DIR, "eval_zs.json")))
    return [d["item"] for d in sorted(items, key=lambda d: d["item"])][:N_ITEMS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--runs", default=None, help="comma list; shard the work across GPUs")
    ap.add_argument("--aggregate", action="store_true",
                    help="combine the per-run parts and run the H-S2 analysis")
    a = ap.parse_args()
    import pandas as pd
    PARTS = os.path.join(OUT, "s2_parts")
    os.makedirs(PARTS, exist_ok=True)
    if a.aggregate:
        import glob
        df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(PARTS, "*.csv")))],
                       ignore_index=True)
        df.to_csv(os.path.join(OUT, "runs_s2.csv"), index=False)
        print(f"[s2] aggregated {df.run.nunique()} runs, {len(df)} rows", flush=True)
        analyse(df, a.n_boot)
        return
    todo = a.runs.split(",") if a.runs else RUNS
    from evaluate import Scorer, is_degenerate
    from ecapa_gate import Ecapa
    import jiwer

    gate = json.load(open(os.path.join(OUT, "ecapa_gate.json")))
    assert gate["passes"], "ECAPA failed the §1.7 gate — S2 must halt (§4-S2)"

    ids = item_ids()
    meta = {d["item"]: d for d in json.load(open(os.path.join(PROC_DIR, "eval_zs.json")))}
    aud = os.path.join(PROC_DIR, "eval_audio")
    prompts = {i: _read(os.path.join(aud, meta[i]["prompt_id"] + ".flac")) for i in ids}

    sel = Scorer(a.device, use_utmos=False)          # SELECTOR: WavLM-SV
    assert "wavlm_large" in sel.sim_model_name
    sco = Ecapa(a.device)                            # SCORER: ECAPA (gated)

    e_prompt_sel = {i: sel.embed([prompts[i]])[0] for i in ids}
    e_prompt_sco = {i: sco.embed([prompts[i]])[0] for i in ids}

    rows = []
    for r in todo:
        # ---- best-of-K: select with WavLM-SV among the first K candidates ----
        cand_wav = {}
        for c in range(8):
            d = os.path.join(REPO, "runs", r, f"synth_bok{c}")
            for i in ids:
                cand_wav.setdefault(i, []).append(_read(os.path.join(d, f"{i}.flac")))
        # embed all 8 candidates ONCE per item; each tier then argmaxes over a prefix.
        # (the selector embeds one utterance per forward pass by design -- LOG.md D-004 --
        # so re-embedding per tier was tripling the cost for no new information)
        cand_sel = {}
        for i in ids:
            E = sel.embed(cand_wav[i])
            cand_sel[i] = torch.nn.functional.cosine_similarity(
                E, e_prompt_sel[i][None]).numpy()
        for nfe, T, K in TIERS:
            picks = {}
            for i in ids:
                j = int(np.argmax(cand_sel[i][:K]))
                picks[i] = (j, cand_wav[i][j])
            wavs = [picks[i][1] for i in ids]
            Es = sco.embed(wavs)
            ecapa = torch.nn.functional.cosine_similarity(
                Es, torch.stack([e_prompt_sco[i] for i in ids])).numpy()
            Ew = sel.embed(wavs)
            wavlm = torch.nn.functional.cosine_similarity(
                Ew, torch.stack([e_prompt_sel[i] for i in ids])).numpy()
            hyp = sel.transcribe(wavs)
            for k, i in enumerate(ids):
                ref = sel.norm(meta[i]["target_text"])
                h = sel.norm(hyp[k])
                rows.append({"run": r, "arm": "search", "nfe": nfe, "K": K, "T": 8,
                             "item": i, "ecapa": float(ecapa[k]), "wavlm": float(wavlm[k]),
                             "wer": float(jiwer.wer(ref, h)) if ref else np.nan,
                             "degenerate": bool(is_degenerate(h, ref)),
                             "pick": picks[i][0]})
        # ---- refinement: reuse the frozen audio, score with the same instrument ----
        for nfe, T, K in TIERS:
            d = os.path.join(REPO, "runs", r, f"synth_T{T}")
            wavs = [_read(os.path.join(d, f"{i}.flac")) for i in ids]
            Es = sco.embed(wavs)
            ecapa = torch.nn.functional.cosine_similarity(
                Es, torch.stack([e_prompt_sco[i] for i in ids])).numpy()
            Ew = sel.embed(wavs)
            wavlm = torch.nn.functional.cosine_similarity(
                Ew, torch.stack([e_prompt_sel[i] for i in ids])).numpy()
            hyp = sel.transcribe(wavs)
            for k, i in enumerate(ids):
                ref = sel.norm(meta[i]["target_text"])
                h = sel.norm(hyp[k])
                rows.append({"run": r, "arm": "refine", "nfe": nfe, "K": 1, "T": T,
                             "item": i, "ecapa": float(ecapa[k]), "wavlm": float(wavlm[k]),
                             "wer": float(jiwer.wer(ref, h)) if ref else np.nan,
                             "degenerate": bool(is_degenerate(h, ref)), "pick": 0})
        pd.DataFrame([x for x in rows if x["run"] == r]).to_csv(
            os.path.join(PARTS, f"{r}.csv"), index=False)
        print(f"[s2] {r} done -> s2_parts/{r}.csv", flush=True)


def analyse(df, n_boot: int):
    import pandas as pd
    rng = np.random.default_rng(BOOT_RNG)
    res = {"tiers": {}, "n_runs": len(RUNS), "n_items": N_ITEMS,
           "selector": "wavlm-large SV (seed-tts-eval)", "scorer": "ECAPA (gated §1.7)"}
    tiers_pos_primary, tiers_pos_second = 0, 0
    for nfe, T, K in TIERS:
        per_run = []
        for r in RUNS:
            s = df[(df.run == r) & (df.nfe == nfe) & (df.arm == "search")]
            f = df[(df.run == r) & (df.nfe == nfe) & (df.arm == "refine")]
            per_run.append({
                "run": r,
                "d_ecapa": float(s.ecapa.mean() - f.ecapa.mean()),
                "d_wavlm": float(s.wavlm.mean() - f.wavlm.mean()),
                "d_wer": float(s.wer.mean() - f.wer.mean()),
                "search_degen": float(s.degenerate.mean()),
                "refine_degen": float(f.degenerate.mean()),
                "search_wer": float(s.wer.mean()), "refine_wer": float(f.wer.mean())})
        d = np.array([x["d_ecapa"] for x in per_run])
        reps = np.array([d[rng.choice(len(d), len(d), True)].mean() for _ in range(n_boot)])
        ci = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
        dw = np.array([x["d_wavlm"] for x in per_run])
        # per-item win rate (second lens, model-free)
        m = df[df.nfe == nfe].pivot_table(index=["run", "item"], columns="arm",
                                          values="ecapa")
        win = float((m["search"] > m["refine"]).mean())
        wreps = np.array([np.mean((m["search"] > m["refine"]).values[
            rng.choice(len(m), len(m), True)]) for _ in range(n_boot)])
        wci = [float(np.percentile(wreps, 2.5)), float(np.percentile(wreps, 97.5))]
        # variance-only rival: does K help the median item, or only the tail?
        med = float(np.median((m["search"] - m["refine"]).values))
        prim = bool(d.mean() > 0 and ci[0] > 0)
        sec = bool(dw.mean() > 0 and win > 0.5 and wci[0] > 0.5)
        tiers_pos_primary += int(prim)
        tiers_pos_second += int(sec)
        res["tiers"][str(nfe)] = {
            "refinement_T": T, "K": K,
            "d_ecapa": float(d.mean()), "ci": ci, "primary_supported": prim,
            "d_wavlm": float(dw.mean()),
            "per_item_win_rate": win, "win_rate_ci": wci,
            "second_lens_supported": sec,
            "median_item_delta": med,
            "d_wer": float(np.mean([x["d_wer"] for x in per_run])),
            "search_wer": float(np.mean([x["search_wer"] for x in per_run])),
            "refine_wer": float(np.mean([x["refine_wer"] for x in per_run])),
            "search_degen": float(np.mean([x["search_degen"] for x in per_run])),
            "refine_degen": float(np.mean([x["refine_degen"] for x in per_run])),
        }
    primary = tiers_pos_primary >= 2
    second = tiers_pos_second >= 2
    res["H_S2"] = {
        "rule": "primary: ECAPA-SIM(best-of-K) - ECAPA-SIM(refinement) paired per-run "
                "CI > 0 in >= 2 of 3 NFE tiers. second lens: WavLM-SV contrast "
                "directionally consistent AND per-item win rate > 50% with CI "
                "(PREREGISTRATION-v1.2.md H-S2).",
        "primary_tiers_supported": tiers_pos_primary,
        "second_lens_tiers_supported": tiers_pos_second,
        "verdict": ("SUPPORTED" if primary and second else
                    "REFUTED" if not primary and not second else "DISCORDANT"),
        "lenses_agree": bool(primary == second),
    }
    with open(os.path.join(OUT, "s2_search.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"\n{'NFE':>5} {'refine':>8} {'search':>8} {'dECAPA':>9} {'95% CI':>22} "
          f"{'win%':>6} {'dWER':>8}", flush=True)
    for nfe, T, K in TIERS:
        t = res["tiers"][str(nfe)]
        print(f"{nfe:>5} {'T='+str(T):>8} {'K='+str(K):>8} {t['d_ecapa']:>+9.4f} "
              f"[{t['ci'][0]:+.4f}, {t['ci'][1]:+.4f}] {100*t['per_item_win_rate']:>5.1f}% "
              f"{t['d_wer']:>+8.4f}", flush=True)
    print(f"\n[s2] VERDICT: H-S2 {res['H_S2']['verdict']} "
          f"(primary {tiers_pos_primary}/3 tiers, second lens {tiers_pos_second}/3)",
          flush=True)


if __name__ == "__main__":
    main()
