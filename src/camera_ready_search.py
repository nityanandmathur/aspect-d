"""Camera-ready numbers for test-time search, and the identity-ledger figure.

Answers three reviewer points with committed artifacts only (no new GPU work):

(a) matched-NFE search vs refinement for every best-of-K / refinement-T pair the
    artifacts support, with the paper's own bootstrap (per-run paired means, 15 runs,
    2000 reps, rng 7331, as in s2_search.analyse), plus the cost of search that the
    generator-NFE ledger leaves out: one Mimi decode and one selector forward per
    candidate (and one selector forward on the prompt). `--cost` counts those FLOPs
    with torch's FlopCounterMode on randomly initialised copies of the three networks
    and times one item on this machine's CPU; without `--cost` the last saved
    artifacts-camera/search_cost.json is reused.
(b) a cross-selector control: select with encoder X, score with encoder Y != X. The
    only encoders with scores for all K=8 candidates are WavLM-large SV and ECAPA
    (artifacts-v1.2/s2_confound*.csv); the other three encoders in multi_encoder.csv
    only scored the WavLM-selected candidate, so they cannot select.
(c) the identity-ledger figure, redrawn so that every bar is read from the same
    artifact field as the table macro it stands for.

    python src/camera_ready_search.py [--cost]        # writes artifacts-camera/search*.json,
                                                      # artifacts-camera/figures/identity_ledger.*,
                                                      # artifacts-camera/numbers_cr_search.tex
    python src/camera_ready_search.py --paper-dir ../aspect-d-paper   # also copy to the paper repo
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "artifacts-camera")
A12 = os.path.join(REPO, "artifacts-v1.2")
A13 = os.path.join(REPO, "artifacts-v1.3")
BOOT_RNG, N_BOOT = 7331, 2000
T_SEARCH = 8                       # every best-of-K candidate is drawn at T=8 (s2_search.py)
LEVELS = 8                         # NFE = 8 * T per candidate
REFINE_T = (8, 16, 32, 64)         # T=8 refinement == candidate 0 (synth_bok0 reproduces v1.0 T=8)
GROUPS = {                          # run root -> runs, as in s2_search / t1_analysis
    "C": ("runs", [f"C{i}_{s}" for i in range(1, 6) for s in (0, 1, 2)]),
    "D": ("runs-v1.1", [f"D{i}_{s}" for i in range(1, 6) for s in (0, 1)]),
    "N": ("runs-v1.1", ["C1_0_90k", "C3_0_90k", "C5_0_90k"]),
}
ENC = {"wavlm": "WavLM-large SV", "ecapa": "ECAPA-TDNN"}


# ----------------------------------------------------------------- data
def load_group(g: str):
    """Per-candidate (8 per item) and per-refinement-T scores under both encoders."""
    root, runs = GROUPS[g]
    if g == "C":
        cand = pd.read_csv(os.path.join(A12, "s2_confound.csv"))
        s2 = pd.read_csv(os.path.join(A12, "runs_s2.csv"))
    else:
        cand = pd.concat([pd.read_csv(os.path.join(A12, "s2_confound_parts", f"{root}__{r}.csv"))
                          for r in runs], ignore_index=True)
        s2 = pd.concat([pd.read_csv(os.path.join(A12, "s2_parts", f"{root}__{r}.csv"))
                        for r in runs], ignore_index=True)
    cand = cand[cand.run.isin(runs)]
    s2 = s2[s2.run.isin(runs)]
    ref = s2[s2.arm == "refine"][["run", "item", "T", "ecapa", "wavlm", "wer"]].copy()
    # T=8 refinement is candidate 0 (same RNG stream, cand=0); it has no WER column here
    c0 = cand[cand.cand == 0][["run", "item", "wavlm", "ecapa"]].assign(T=8, wer=np.nan)
    ref = pd.concat([ref, c0], ignore_index=True)
    return runs, cand, s2, ref


def best_of_k(cand: pd.DataFrame, K: int, sel: str, sco: str) -> pd.DataFrame:
    """Score (under `sco`) of the candidate that `sel` ranks highest among the first K."""
    sub = cand[cand.cand < K]
    idx = sub.groupby(["run", "item"])[sel].idxmax()
    out = sub.loc[idx, ["run", "item", "cand", sco]].rename(columns={sco: "score",
                                                                       "cand": "pick"})
    return out.set_index(["run", "item"])


def random_pick(cand: pd.DataFrame, K: int, sco: str) -> pd.Series:
    return cand[cand.cand < K].groupby(["run", "item"])[sco].mean()


# ------------------------------------------------------------ statistics
def paired(a: pd.Series, b: pd.Series, runs, n_boot: int = N_BOOT) -> dict:
    """a - b, the paper's way: per-run difference of means, bootstrap over runs for the CI,
    per-(run, item) win rate with an item-level bootstrap (s2_search.analyse)."""
    rng = np.random.default_rng(BOOT_RNG)
    j = pd.concat({"a": a, "b": b}, axis=1).dropna()
    per_run = j.groupby(level="run").apply(lambda x: x.a.mean() - x.b.mean())
    d = per_run.reindex(runs).values
    reps = np.array([d[rng.choice(len(d), len(d), True)].mean() for _ in range(n_boot)])
    w = (j.a > j.b).values
    wreps = np.array([w[rng.choice(len(w), len(w), True)].mean() for _ in range(n_boot)])
    return {"delta": float(d.mean()),
            "ci": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))],
            "win": float(w.mean()),
            "win_ci": [float(np.percentile(wreps, 2.5)), float(np.percentile(wreps, 97.5))],
            "runs_positive": int((d > 0).sum()), "n_runs": int(len(d)), "n_pairs": int(len(w))}


# --------------------------------------------------------- (a) matched NFE
def matched_nfe(g: str = "C") -> dict:
    runs, cand, s2, ref = load_group(g)
    R = {T: ref[ref["T"] == T].set_index(["run", "item"]) for T in REFINE_T}
    res = {"group": g, "runs": runs, "selector": "wavlm", "scorer": "ecapa",
           "n_items": int(cand.item.nunique()), "pairs": {}, "matrix": {}}
    # sanity: best-of-K from the per-candidate table reproduces the selected rows of runs_s2
    for K, nfe in ((2, 128), (4, 256), (8, 512)):
        mine = best_of_k(cand, K, "wavlm", "ecapa").score
        theirs = s2[(s2.arm == "search") & (s2.nfe == nfe)].set_index(["run", "item"]).ecapa
        res.setdefault("reproduces_runs_s2", {})[str(K)] = float(
            np.abs(mine.reindex(theirs.index) - theirs).max())
    for K in range(1, 9):
        s = best_of_k(cand, K, "wavlm", "ecapa").score
        for T in REFINE_T:
            p = paired(s, R[T].ecapa, runs)
            p.update({"K": K, "T": T, "nfe_search": LEVELS * T_SEARCH * K,
                      "nfe_refine": LEVELS * T, "nfe_ratio": T_SEARCH * K / T})
            res["matrix"][f"K{K}_T{T}"] = p
            if T_SEARCH * K == T:
                res["pairs"][str(LEVELS * T)] = p
    # WER of each arm (search WER only exists for the three selected tiers in runs_s2)
    for nfe in (128, 256, 512):
        t = s2[s2.nfe == nfe]
        res["pairs"][str(nfe)].update({
            "search_wer": float(t[t.arm == "search"].groupby("run").wer.mean().mean()),
            "refine_wer": float(t[t.arm == "refine"].groupby("run").wer.mean().mean())})
    return res


# ------------------------------------------------------ (b) cross-selector
def cross_selector(g: str) -> dict:
    runs, cand, s2, ref = load_group(g)
    res = {"group": g, "n_runs": len(runs), "cells": {}, "pick_agreement": {}}
    for nfe, T, K in ((128, 16, 2), (256, 32, 4), (512, 64, 8)):
        base = ref[ref["T"] == T].set_index(["run", "item"])
        base16 = ref[ref["T"] == 16].set_index(["run", "item"])
        for sel in ENC:
            for sco in ENC:
                s = best_of_k(cand, K, sel, sco).score
                cell = paired(s, base[sco], runs)
                cell["vs_T16"] = paired(s, base16[sco], runs)
                cell["over_random_pick"] = paired(s, random_pick(cand, K, sco), runs)
                cell.update({"selector": sel, "scorer": sco, "circular": sel == sco,
                             "K": K, "T": T, "nfe": nfe})
                res["cells"][f"{nfe}_{sel}_to_{sco}"] = cell
        pw = best_of_k(cand, K, "wavlm", "wavlm").pick
        pe = best_of_k(cand, K, "ecapa", "ecapa").pick
        res["pick_agreement"][str(nfe)] = {"K": K, "agree": float((pw == pe).mean()),
                                           "chance": 1.0 / K}
    return res


# -------------------------------------------------------------- cost model
def generator_flops(width: int, depth: int, n_phon: int, n_frames: int) -> float:
    """FLOPs of one AspectD forward (model.py): 24 S w^2 + 4 S^2 w per block, plus one
    level's output head over the frame positions. Embedding lookups are not counted."""
    S = n_phon + n_frames
    return depth * (24 * S * width ** 2 + 4 * S * S * width) + 2 * n_frames * width * 2048


def phon_per_char() -> dict:
    """espeak-ng phones per character on the four items whose text ships in samples/,
    tokenised exactly as data._phon_chunk does (phones, punctuation runs, <sp>)."""
    import sys
    sys.path.insert(0, os.path.join(REPO, "src"))
    import data
    man = json.load(open(os.path.join(REPO, "samples", "manifest.json")))["items"]
    texts = [x[k] for x in man for k in ("prompt_text", "target_text")]
    data._phon_init()
    toks = data._phon_chunk(texts)
    n_ph, n_ch = sum(len(t) for t in toks), sum(len(t.strip()) for t in texts)
    return {"phones": n_ph, "chars": n_ch, "ratio": n_ph / n_ch, "n_texts": len(texts)}


def cost(dataset_json: str, device: str = "cpu") -> dict:
    """FLOPs and one-item wall time for generator forward, Mimi decode, WavLM-large forward."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from transformers import MimiConfig, MimiModel, WavLMConfig, WavLMModel
    import sys
    sys.path.insert(0, os.path.join(REPO, "src"))
    from model import AspectD

    grid = json.load(open(os.path.join(REPO, "configs", "grid.json")))
    spc = json.load(open(dataset_json))["sec_per_char"]   # proc/dataset.json (also on the HF repo)
    meta = json.load(open(os.path.join(REPO, "runs", "C1_0", "synth_bok0", "synth.json")))["meta"]
    n_p = np.array([m["n_prompt"] for m in meta])
    n_t = np.array([m["n_target"] for m in meta])
    try:
        ppc = phon_per_char()
    except Exception as e:                       # phonemizer/espeak missing: bound only
        ppc = {"ratio": None, "error": repr(e)}
    chars = (n_p + n_t) / (12.5 * spc)            # target chars exact; prompt chars at corpus rate
    n_ph = np.minimum(512, np.round(chars * ppc["ratio"]) + 1) if ppc["ratio"] else 0 * n_p

    cfgs = {c["id"]: c for c in grid["configs"]}
    gen = {}
    for cid in ("C1", "C2", "C3", "C4", "C5"):
        c = cfgs[cid]
        f = np.array([generator_flops(c["width"], c["depth"], int(a), int(b + p))
                      for a, b, p in zip(n_ph, n_t, n_p)])
        f0 = np.array([generator_flops(c["width"], c["depth"], 0, int(b + p))
                       for b, p in zip(n_t, n_p)])
        gen[cid] = {"flops_mean": float(f.mean()), "flops_no_phonemes_mean": float(f0.mean())}

    # verify the analytic count against FlopCounterMode on one mean-length item (C3)
    i0 = int(np.argsort(n_t)[len(n_t) // 2])
    c3 = cfgs["C3"]
    m = AspectD(c3["width"], c3["depth"], c3["heads"], 128).eval()
    P, Fr = int(n_ph[i0]), int(n_p[i0] + n_t[i0])
    ph = torch.zeros((1, P), dtype=torch.long); phm = torch.ones((1, P), dtype=torch.bool)
    au = torch.zeros((1, Fr, 8), dtype=torch.long); fm = torch.ones((1, Fr), dtype=torch.bool)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m.logits(m(ph, phm, au, fm), 0)
    gen_check = {"item": meta[i0]["item"], "n_phon": P, "n_frames": Fr,
                 "flop_counter": int(fc.get_total_flops()),
                 "analytic": generator_flops(c3["width"], c3["depth"], P, Fr)}
    with torch.no_grad():
        for _ in range(2):
            m.logits(m(ph, phm, au, fm), 0)
        t0 = time.perf_counter()
        for _ in range(5):
            m.logits(m(ph, phm, au, fm), 0)
        gen_check["cpu_seconds_per_forward"] = (time.perf_counter() - t0) / 5

    # Mimi decoder: codes [1, 8, n_target] -> 24 kHz audio (kyutai/mimi config, random weights)
    mc = MimiConfig.from_pretrained("kyutai/mimi")
    mimi = MimiModel(mc).eval()
    codes = torch.zeros((1, 8, int(n_t[i0])), dtype=torch.long)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        mimi.decode(codes)
    mimi_flops_item = int(fc.get_total_flops())
    with torch.no_grad():
        mimi.decode(codes)
        t0 = time.perf_counter()
        for _ in range(3):
            mimi.decode(codes)
        mimi_sec = (time.perf_counter() - t0) / 3

    # WavLM-large backbone on the 16 kHz candidate audio (microsoft/wavlm-large config)
    wc = WavLMConfig.from_pretrained("microsoft/wavlm-large")
    wav = WavLMModel(wc).eval()
    x = torch.randn((1, int(round(n_t[i0] / 12.5 * 16000))))
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        wav(x, output_hidden_states=True)
    wav_flops_item = int(fc.get_total_flops())
    with torch.no_grad():
        wav(x)
        t0 = time.perf_counter()
        for _ in range(3):
            wav(x, output_hidden_states=True)
        wav_sec = (time.perf_counter() - t0) / 3
    xp = torch.randn((1, int(round(n_p[i0] / 12.5 * 16000))))
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        wav(xp, output_hidden_states=True)
    wav_prompt_flops_item = int(fc.get_total_flops())

    # both decoders scale ~linearly in duration; carry the one-item count to the 200-item mean
    scale = float(n_t.mean() / n_t[i0])
    scale_p = float(n_p.mean() / n_p[i0])
    res = {
        "device": device, "torch": torch.__version__,
        "n_items": int(len(meta)), "mean_n_prompt": float(n_p.mean()),
        "mean_n_target": float(n_t.mean()), "mean_n_phon_est": float(np.mean(n_ph)),
        "phon_per_char": ppc, "sec_per_char": spc,
        "generator": gen, "generator_check": gen_check,
        "mimi_decode": {"flops_item": mimi_flops_item, "flops_mean": mimi_flops_item * scale,
                        "cpu_seconds_item": mimi_sec,
                        "params": int(sum(p.numel() for p in mimi.parameters()))},
        "wavlm_large": {"flops_item": wav_flops_item, "flops_mean": wav_flops_item * scale,
                        "prompt_flops_mean": wav_prompt_flops_item * scale_p,
                        "cpu_seconds_item": wav_sec,
                        "params": int(sum(p.numel() for p in wav.parameters()))},
        "excluded": ["ECAPA-TDNN-small speaker head on top of WavLM-large (seed-tts-eval) "
                     "is not counted", "24->16 kHz resampling is not counted",
                     "the ECAPA scorer and the ASR are evaluation-only and are never "
                     "part of deployed cost"],
    }
    return res


def cost_table(c: dict) -> dict:
    """Search overhead in generator-NFE equivalents, per C-budget config, for K=2,4,8."""
    out = {}
    mimi = c["mimi_decode"]["flops_mean"]
    sv = c["wavlm_large"]["flops_mean"]
    svp = c["wavlm_large"]["prompt_flops_mean"]
    for cid, g in c["generator"].items():
        for tag, gf in (("est", g["flops_mean"]), ("nophon", g["flops_no_phonemes_mean"])):
            for K in (2, 4, 8):
                nfe = LEVELS * T_SEARCH * K
                extra = (K - 1) * mimi + K * sv + svp     # refinement decodes once, never selects
                out[f"{cid}_{tag}_K{K}"] = {
                    "extra_nfe_equiv": extra / gf,
                    "overhead_frac": extra / (nfe * gf + mimi)}
    ks = {}
    for K in (2, 4, 8):
        e = [v["extra_nfe_equiv"] for k, v in out.items() if k.endswith(f"_est_K{K}")]
        f = [v["overhead_frac"] for k, v in out.items() if k.endswith(f"_est_K{K}")]
        f0 = [v["overhead_frac"] for k, v in out.items() if k.endswith(f"_nophon_K{K}")]
        ks[str(K)] = {"extra_nfe_lo": min(e), "extra_nfe_hi": max(e),
                      "overhead_lo": min(f), "overhead_hi": max(f),
                      "overhead_bound_nophon": max(f0)}
    # one-item wall-time ratio on CPU, C3, K=2 vs T=16 (both 128 generator forwards)
    gs = c["generator_check"]["cpu_seconds_per_forward"]
    ms, ws = c["mimi_decode"]["cpu_seconds_item"], c["wavlm_large"]["cpu_seconds_item"]
    wall = {}
    for K in (2, 4, 8):
        nfe = LEVELS * T_SEARCH * K
        ref = nfe * gs + ms
        srch = nfe * gs + K * ms + (K + 1) * ws        # prompt embedded once, same length bound
        wall[str(K)] = {"refine_s": ref, "search_s": srch, "ratio": srch / ref}
    return {"per_config": out, "by_K": ks, "wall_cpu_C3": wall}


def equal_total(mt: dict, ct: dict) -> dict:
    """Search vs the cheapest refinement arm whose total cost is at least search's
    (generator + decode + selector FLOPs); refinement's identity falls with T past 16,
    so this comparison can only favour refinement."""
    out = {}
    for K in (2, 4, 8):
        hi = ct["by_K"][str(K)]["extra_nfe_hi"]
        need = LEVELS * T_SEARCH * K + hi
        T = next((t for t in REFINE_T if LEVELS * t >= need), None)
        if T is None:
            out[str(K)] = {"T": None, "note": "no refinement arm on disk is that expensive"}
            continue
        out[str(K)] = {**mt["matrix"][f"K{K}_T{T}"], "search_total_nfe_equiv_hi": need}
    return out


# ------------------------------------------------------------- the figure
def ledger_values(cross: dict) -> list:
    """Every bar is read from the field its table macro is read from (paper.py)."""
    L = json.load(open(os.path.join(A12, "identity_ledger.json")))
    t23 = json.load(open(os.path.join(A13, "t23_training.json")))["H_T2"]
    s4 = json.load(open(os.path.join(A12, "s4_guidance.json")))["per_gamma"]["0.5"]
    s3 = json.load(open(os.path.join(A12, "s3_rate.json")))["MEASURED"]["primary"]
    srch = cross["cells"]["128_ecapa_to_wavlm"]
    srch8 = cross["cells"]["512_ecapa_to_wavlm"]       # best-of-8 vs T=64: matched NFE
    return [
        ("refinement\n$T{=}1\\to16$", L["ledger"]["steps_T1_to_T16"]["gain"], "steps",
         "identity_ledger.json ledger.steps_T1_to_T16 (= \\NabsSIMgain)"),
        ("training\n30k$\\to$90k", t23["second_lens_30k_to_90k_replication"]["delta"],
         "depth", "t23_training.json H_T2.second_lens_30k_to_90k_replication (= \\NtrainGainSim)"),
        ("parameters\n20M$\\to$276M", L["ledger"]["parameters_N"]["gain"], "depth",
         "identity_ledger.json ledger.parameters_N (= \\NparamGainSim)"),
        ("search\n$K{=}2$\nsame NFE", srch["delta"], "steps",
         "search_cross_selector.json C 128_ecapa_to_wavlm (= \\NcrLedgerSearch)"),
        ("search\n$K{=}8$\nsame NFE", srch8["delta"], "steps",
         "search_cross_selector.json C 512_ecapa_to_wavlm (= \\NcrXEWHid)"),
        ("guidance\n$\\gamma{=}0.5$", s4["d_sim"], "steps",
         "s4_guidance.json per_gamma.0.5.d_sim (= \\NcrGuideGainSim)"),
        ("rate\nmatching", s3["d_sim"], "steps",
         "s3_rate.json MEASURED.primary.d_sim (= \\NrateGainSim)"),
    ], L["MEASURED"]


def figure(cross: dict, outdir: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED = "#14181C", "#5B6470"
    C_STEPS, C_WIDTH, C_DEPTH, C_STOP = "#0F7B72", "#C4541D", "#23479C", "#B3261E"
    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF", "savefig.facecolor": "#FFFFFF",
        "font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 11,
        "axes.titlesize": 11.5, "axes.edgecolor": "#D9DEDE", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "axes.grid": True,
        "grid.color": "#ECF1F0", "grid.linewidth": 0.8, "legend.frameon": False,
        "legend.fontsize": 9, "figure.dpi": 130, "svg.hashsalt": "aspect-d",
        "pdf.fonttype": 42})
    bars, m = ledger_values(cross)
    names = [b[0] for b in bars]
    gains = [b[1] for b in bars]
    _c = {"steps": C_STEPS, "depth": C_DEPTH, "width": C_WIDTH}
    cols = [_c[b[2]] for b in bars]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.1),
                                  gridspec_kw={"width_ratios": [1.75, 1]},
                                  constrained_layout=True)
    ax.text(0.012, 0.985, "(A)", transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=INK)
    b = ax.bar(range(len(gains)), gains, color=cols, width=0.62, zorder=3)
    b[0].set_alpha(0.45)        # measured from one-step decoding: not an operating point
    head = m["headroom_mean"]
    ax.axhline(head, color=C_STOP, ls="--", lw=1.8, zorder=4,
               label=f"remaining headroom to the codec ceiling ({head:+.4f})")
    for r, g in zip(b, gains):
        ax.annotate(f"{g:+.4f}", (r.get_x() + r.get_width() / 2, g),
                    ha="center", va="bottom", fontsize=10, color=INK,
                    xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9.6)
    ax.set_ylabel("SIM-o gain (WavLM-L)")
    ax.set_title("SIM-o gain of each change", pad=6)
    ax.set_ylim(0, max(max(gains), head) * 1.22)
    ax.legend(loc="upper right")

    ax2.text(0.02, 0.985, "(B)", transform=ax2.transAxes, ha="left", va="top",
             fontsize=12, fontweight="bold", color=INK)
    gt, rt, sysbest = (m["sim_gt_no_roundtrip_mean"], m["sim_roundtrip_mean"],
                       m["best_system"]["sim"])
    codec, model = gt - rt, rt - sysbest
    ax2.barh([0], [codec], left=[sysbest + model], color=C_WIDTH, height=0.42, zorder=3,
             label=f"lost to the codec ({codec:.3f})")
    ax2.barh([0], [model], left=[sysbest], color=C_DEPTH, height=0.42, zorder=3,
             label=f"lost to the model ({model:.3f})")
    ax2.barh([0], [sysbest], color="#C9D3D2", height=0.42, zorder=3,
             label=f"best 90k-step model ({sysbest:.3f})")
    for x, lab, ha, y in ((sysbest, "best 90k model", "right", 0.28),
                          (rt, "codec ceiling", "center", 0.50), (gt, "real audio", "left", 0.28)):
        ax2.axvline(x, color=INK, lw=1.0, ls=":", zorder=4)
        ax2.annotate(f"{lab}\n{x:.3f}", (x, y), ha=ha, va="bottom", fontsize=9.5, color=INK,
                     xytext={"right": (-3, 0), "left": (3, 0), "center": (0, 0)}[ha],
                     textcoords="offset points")
    ax2.set_yticks([])
    ax2.set_xlim(0, gt * 1.16)
    ax2.set_ylim(-0.95, 0.80)
    ax2.set_xlabel("SIM-o against the prompt")
    ax2.set_title(f"{100 * codec / (codec + model):.0f}% of the gap to real audio is the codec", pad=6)
    ax2.legend(loc="lower right", fontsize=9.5, frameon=True, facecolor="#FFFFFF",
               edgecolor="none", framealpha=1.0).set_zorder(5)

    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(outdir, f"identity_ledger.{ext}"), bbox_inches="tight",
                    dpi=160 if ext == "png" else None,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)
    return {"bars": [{"label": n.replace("\n", " "), "value": v, "source": s}
                     for n, v, _, s in bars], "headroom": head}


# ---------------------------------------------------------------- macros
def fmt_ci(ci, pct=False):
    return ("[{:.1f}, {:.1f}]".format(100 * ci[0], 100 * ci[1]) if pct
            else "[{:.4f}, {:.4f}]".format(*ci))


def macros(mt, cross, ct, eq, fig, costj) -> str:
    L = []
    A = lambda k, v: L.append(f"\\newcommand{{\\Ncr{k}}}{{{v}}}\n")
    tag = {"128": "Lo", "256": "Mid", "512": "Hi"}
    # (a) matched NFE, ECAPA-scored, WavLM-selected (reproduces \Nsrch*)
    for nfe, t in tag.items():
        p = mt["pairs"][nfe]
        A(f"Match{t}d", f"{p['delta']:+.4f}")
        A(f"Match{t}ci", fmt_ci(p["ci"]))
        A(f"Match{t}win", f"{100 * p['win']:.1f}")
        A(f"Match{t}winci", fmt_ci(p["win_ci"], True))
        A(f"Match{t}K", str(p["K"]))
        A(f"Match{t}T", str(p["T"]))
    # unmatched comparisons against the deployable T=16 default, with their NFE ratio
    for K, w in ((4, "Four"), (8, "Eight")):
        p = mt["matrix"][f"K{K}_T16"]
        A(f"Default{w}d", f"{p['delta']:+.4f}")
        A(f"Default{w}ci", fmt_ci(p["ci"]))
        A(f"Default{w}ratio", f"{p['nfe_ratio']:.0f}$\\times$")
    # refinement's own ECAPA trend with T (why the 512 tier flatters search)
    for T, w in ((32, "ThirtyTwo"), (64, "SixtyFour")):
        r = mt["matrix"][f"K1_T{T}"]
        r16 = mt["matrix"]["K1_T16"]
        A(f"RefineDrop{w}", f"{r16['delta'] - r['delta']:+.4f}")
    # (a) cost of search in generator-NFE equivalents
    if ct:
        for K, w in (("2", "Two"), ("4", "Four"), ("8", "Eight")):
            k = ct["by_K"][K]
            A(f"Cost{w}nfe", f"{k['extra_nfe_lo']:.1f}--{k['extra_nfe_hi']:.1f}")
            A(f"Cost{w}pct", f"{100 * k['overhead_lo']:.1f}--{100 * k['overhead_hi']:.1f}")
            A(f"Cost{w}bound", f"{100 * k['overhead_bound_nophon']:.1f}")
            A(f"Wall{w}ratio", f"{ct['wall_cpu_C3'][K]['ratio']:.2f}$\\times$")
        A("MimiGflops", f"{costj['mimi_decode']['flops_mean'] / 1e9:.1f}")
        A("WavlmGflops", f"{costj['wavlm_large']['flops_mean'] / 1e9:.1f}")
        g = [v["flops_mean"] / 1e9 for v in costj["generator"].values()]
        A("GenGflops", f"{min(g):.1f}--{max(g):.1f}")
        A("PhonPerChar", f"{costj['phon_per_char']['ratio']:.2f}")
        for K, w in (("2", "Two"), ("4", "Four")):
            e = eq.get(K, {})
            if e.get("T"):
                A(f"Equal{w}T", str(e["T"]))
                A(f"Equal{w}d", f"{e['delta']:+.4f}")
                A(f"Equal{w}ci", fmt_ci(e["ci"]))
    # (b) cross-selector control, C budget
    C = cross["C"]["cells"]
    for nfe, t in tag.items():
        for sel, sco, nm in (("ecapa", "wavlm", "EW"), ("wavlm", "ecapa", "WE"),
                             ("wavlm", "wavlm", "WW"), ("ecapa", "ecapa", "EE")):
            c = C[f"{nfe}_{sel}_to_{sco}"]
            A(f"X{nm}{t}d", f"{c['delta']:+.4f}")
            A(f"X{nm}{t}ci", fmt_ci(c["ci"]))
            A(f"X{nm}{t}win", f"{100 * c['win']:.1f}")
            A(f"X{nm}{t}winci", fmt_ci(c["win_ci"], True))
        pa = cross["C"]["pick_agreement"][nfe]
        A(f"Agree{t}", f"{100 * pa['agree']:.1f}")
        A(f"Chance{t}", f"{100 * pa['chance']:.1f}")
    for g, w in (("D", "D"), ("N", "N")):
        c = cross[g]["cells"]["512_ecapa_to_wavlm"]
        A(f"XEW{w}d", f"{c['delta']:+.4f}")
        A(f"XEW{w}ci", fmt_ci(c["ci"]))
        A(f"XEW{w}win", f"{100 * c['win']:.1f}")
    # NFE-128 tier on the 3x-trained runs (App. E, "Replications"), both directions
    for sel, sco, nm in (("wavlm", "ecapa", "WE"), ("ecapa", "wavlm", "EW")):
        c = cross["N"]["cells"][f"128_{sel}_to_{sco}"]
        A(f"X{nm}NLod", f"{c['delta']:+.4f}")
        A(f"X{nm}NLoci", fmt_ci(c["ci"]))
        A(f"X{nm}NLowin", f"{100 * c['win']:.1f}")
    A("XNRuns", str(cross["N"]["n_runs"]))
    # (c) the ledger's search rows, in the headroom's own encoder, non-circular
    A("LedgerSearch", f"{C['128_ecapa_to_wavlm']['delta']:+.4f}")
    A("LedgerSearchCI", fmt_ci(C["128_ecapa_to_wavlm"]["ci"]))
    A("LedgerSearchEight", f"{C['512_ecapa_to_wavlm']['vs_T16']['delta']:+.4f}")
    A("LedgerSearchEightCI", fmt_ci(C["512_ecapa_to_wavlm"]["vs_T16"]["ci"]))
    A("LedgerSearchWER", f"{mt['pairs']['128']['search_wer'] - mt['pairs']['128']['refine_wer']:+.4f}")
    A("LedgerSearchEightWER", f"{mt['pairs']['512']['search_wer'] - mt['pairs']['128']['refine_wer']:+.4f}")
    # the same WER costs in WER points (1 point = 0.01 absolute WER), as Table tab:ledger
    # prints guidance and rate matching; the matched best-of-8 row is search vs T=64
    p128, p512 = mt["pairs"]["128"], mt["pairs"]["512"]
    A("LedgerSearchWERpts", f"{100 * (p128['search_wer'] - p128['refine_wer']):+.2f}")
    A("LedgerSearchMatchEightWERpts", f"{100 * (p512['search_wer'] - p512['refine_wer']):+.2f}")
    # guidance row as the committed artifact has it (numbers.tex \NguideGainSim predates the
    # v1.5 re-run of the guidance arm in commit b6ec121 and was not regenerated)
    g4 = json.load(open(os.path.join(A12, "s4_guidance.json")))["per_gamma"]["0.5"]
    A("GuideGainSim", f"{g4['d_sim']:+.4f}")
    A("GuideGainCI", fmt_ci(g4["ci"]))
    A("GuideWER", f"{g4['d_wer_points']:+.2f}")
    A("RefineEightMinusSixteen", f"{mt['matrix']['K1_T16']['delta']:+.4f}")
    A("RefineEightMinusSixteenCI", fmt_ci(mt["matrix"]["K1_T16"]["ci"]))
    hdr = ("% GENERATED by src/camera_ready_search.py from artifacts-v1.2/{s2_confound.csv,\n"
           "% runs_s2.csv, s2_parts/, s2_confound_parts/, identity_ledger.json, s3/s4},\n"
           "% artifacts-v1.3/t23_training.json and artifacts-camera/search_cost.json.\n"
           "% Do not edit; edit the source and rerun.\n")
    return hdr + "".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", action="store_true", help="recount FLOPs / wall time (torch)")
    ap.add_argument("--dataset-json", default=None,
                    help="proc/dataset.json (sec_per_char); defaults to data.PROC_DIR")
    ap.add_argument("--paper-dir", default=None, help="paper repo root to copy macros+figure into")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    mt = matched_nfe("C")
    cross = {g: cross_selector(g) for g in GROUPS}
    cp = os.path.join(OUT, "search_cost.json")
    if a.cost:
        dj = a.dataset_json
        if dj is None:
            import sys
            sys.path.insert(0, os.path.join(REPO, "src"))
            from data import PROC_DIR
            dj = os.path.join(PROC_DIR, "dataset.json")
        json.dump(cost(dj), open(cp, "w"), indent=1)
    costj = json.load(open(cp)) if os.path.exists(cp) else None
    ct = cost_table(costj) if costj else None
    eq = equal_total(mt, ct) if ct else {}
    fig = figure(cross["C"], os.path.join(OUT, "figures"))

    json.dump({"matched_nfe": mt, "cost_table": ct, "equal_total_compute": eq},
              open(os.path.join(OUT, "search_matched.json"), "w"), indent=1)
    json.dump(cross, open(os.path.join(OUT, "search_cross_selector.json"), "w"), indent=1)
    json.dump(fig, open(os.path.join(OUT, "search_ledger_figure.json"), "w"), indent=1)
    tex = macros(mt, cross, ct, eq, fig, costj)
    open(os.path.join(OUT, "numbers_cr_search.tex"), "w").write(tex)
    if a.paper_dir:
        open(os.path.join(a.paper_dir, "numbers_cr_search.tex"), "w").write(tex)
        shutil.copyfile(os.path.join(OUT, "figures", "identity_ledger.pdf"),
                        os.path.join(a.paper_dir, "figures", "identity_ledger.pdf"))

    # ---- console summary
    print("matched NFE (WavLM selects, ECAPA scores), C budget:")
    for nfe, p in mt["pairs"].items():
        print(f"  NFE {nfe:>3}: K={p['K']} vs T={p['T']}: {p['delta']:+.4f} {fmt_ci(p['ci'])} "
              f"win {100*p['win']:.1f}%  WER {p.get('refine_wer', float('nan')):.4f}/"
              f"{p.get('search_wer', float('nan')):.4f}")
    print("  reproduction vs runs_s2 (max abs diff):", mt["reproduces_runs_s2"])
    print("cross-selector, C budget (vs matched-NFE refinement):")
    for k, c in cross["C"]["cells"].items():
        print(f"  {k:<22} {c['delta']:+.4f} {fmt_ci(c['ci'])} win {100*c['win']:.1f}% "
              f"{fmt_ci(c['win_ci'], True)}  vs T16 {c['vs_T16']['delta']:+.4f}")
    print("pick agreement:", cross["C"]["pick_agreement"])
    if ct:
        print("cost by K:", json.dumps(ct["by_K"], indent=0))
        print("wall (CPU, C3):", ct["wall_cpu_C3"])
        print("equal total:", {k: (v.get("T"), v.get("delta")) for k, v in eq.items()})
    print("figure bars:", [(b["label"], round(b["value"], 4)) for b in fig["bars"]])


if __name__ == "__main__":
    main()
