"""Camera-ready: the scope and robustness numbers the reviewers asked for.

Four questions, each answered from committed artifacts only (per-item scores.json,
runs*.csv, fits*.json); nothing is re-synthesised and no metric model is run.

(a) Cx9t: does starting at T=1, where ~40% of outputs are degenerate, inflate the
    intelligibility share? Recompute the floor-referenced shares of
    src/v14_analysis.py::gap_closed (same estimator, same 45 runs, same run-level
    bootstrap, same RNG) from T=2 and T=4, and from T=1 on non-degenerate items only
    (unpaired, paired, and items non-degenerate at every T). The same variants are
    repeated on the 90k-step runs with the src/paper_v15.py bootstrap (cluster on config,
    resample items), because that is where the asymmetry is smallest.
(b) yHj7: an external system on the same items. F5-TTS v1 Base was synthesised by
    src/anchor_f5.py and scored with the frozen stack (results/runs-v1.5/f5tts_anchor). It is a
    CALIBRATION ANCHOR, not a competitive baseline (see that script's docstring).
(c) ws2F: which fitted quantities sit on a bound, in how many fits and replicates, and
    where the depth argmin sits per budget and T.
(d) scale: the numbers that bound the claim (training exposure, best SIM-o against the
    codec ceiling, the share by parameter budget).

    python src/camera_ready_scope.py [--amp-boot 2000]
    # -> results/artifacts-camera/scope.json, results/artifacts-camera/scope_fit_boot.json,
    #    results/artifacts-camera/numbers_cr_scope.tex (or --tex)

Supplementary analysis for the reviews; the paper does not use its outputs (the paper's
generated inputs come from src/paper*.py, src/figures.py and src/s0_figure.py only).
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

import fit as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "artifacts-camera")
BOOT_RNG, N_BOOT = 7331, 2000          # as v14_analysis.gap_closed
V15_BOOT = 1000                        # as paper_v15
T_ALL = (1, 2, 4, 8, 16, 32, 64)
T_GRID = (1, 2, 4, 8, 16)              # the declared grid
ROOTS = ("runs", "runs-v1.1", "runs-v1.3", "runs-v1.5")


# ------------------------------------------------------------------------ loading
def _scores(run, T, roots=ROOTS):
    for root in roots:
        p = os.path.join(REPO, "results", root, run, f"synth_T{T}", "scores.json")
        if os.path.exists(p):
            return json.load(open(p))
    return None


def _asr2(run, T):
    for root in ROOTS:
        p = os.path.join(REPO, "results", root, run, f"synth_T{T}", "asr2.json")
        if os.path.exists(p):
            return json.load(open(p))
    return None


def floors():
    g0 = json.load(open(os.path.join(REPO, "results", "artifacts", "g0c_groundtruth.json")))
    led = json.load(open(os.path.join(REPO, "results", "artifacts-v1.2", "identity_ledger.json")))
    per = np.load(os.path.join(REPO, "results", "artifacts-v1.2", "s0_per_item_sim.npy"))
    # row 0 = per-item Mimi round-trip SIM-o, row 3 = row 0 minus the best system
    # (C3_0_90k, T=16) per item; recovering that system's per-item scores from rows 0 and 3
    # checks that the row order is the sorted item order we index by
    best = _scores("C3_0_90k", 16)
    bs = np.array([d["sim"] for d in sorted(best["items"], key=lambda d: d["item"])])
    assert np.nanmax(np.abs((per[0] - per[3]) - bs)) < 1e-6, "s0 per-item order mismatch"
    assert abs(per[0].mean() - led["MEASURED"]["sim_roundtrip_mean"]) < 1e-6
    a2 = json.load(open(os.path.join(REPO, "results", "artifacts-v1.5", "g0c_asr2.json")))
    return {"wer_floor": float(g0["wer_mean_item"]),
            "sim_ceiling": float(led["MEASURED"]["sim_roundtrip_mean"]),
            "sim_ceiling_item": per[0],
            "sim_real_audio": float(led["MEASURED"]["sim_gt_no_roundtrip_mean"]),
            "utmos_real_audio": float(g0["utmos_gt_mean"]),
            "wer_floor_asr2": float(a2["wer_mean"])}


def cube(runs, Ts):
    """per-item arrays [run, T, item] for WER, SIM-o and the frozen degenerate flag"""
    W = np.full((len(runs), len(Ts), 400), np.nan)
    S, G = W.copy(), np.zeros(W.shape, bool)
    ids = None
    for a, r in enumerate(runs):
        for b, T in enumerate(Ts):
            it = sorted(_scores(r, T)["items"], key=lambda d: d["item"])
            ids = ids or [d["item"] for d in it]
            assert [d["item"] for d in it] == ids
            W[a, b] = [d["wer"] for d in it]
            S[a, b] = [d["sim"] for d in it]
            G[a, b] = [d["degenerate"] for d in it]
    return W, S, G, ids


# ------------------------------------------------------------- (a) gap-closed scope
def per_run(W, S, G, C, b0, b1, mode, Tidx_all=None, cfix=None):
    """Per-run means entering the v14 share: w0, w1, s0, s1, ceiling, n items.

    mode: all | nondeg_unpaired | nondeg_paired | nondeg_allT. The full-set modes use the
    published scalar ceiling `cfix` (as v14/v15 do); item subsets use the mean per-item
    round-trip ceiling of the items they keep."""
    R = W.shape[0]
    out = np.zeros((R, 6))
    for r in range(R):
        if mode == "all":
            m0 = m1 = np.ones(400, bool)
        elif mode == "nondeg_unpaired":
            m0, m1 = ~G[r, b0], ~G[r, b1]
        elif mode == "nondeg_paired":
            m0 = m1 = ~G[r, b0] & ~G[r, b1]
        elif mode == "nondeg_allT":
            m0 = m1 = ~G[r][Tidx_all].any(0)
        out[r] = [np.nanmean(W[r, b0, m0]), np.nanmean(W[r, b1, m1]),
                  np.nanmean(S[r, b0, m0]), np.nanmean(S[r, b1, m1]),
                  cfix if mode in ("all", "nondeg_unpaired") else C[m0 & m1].mean(),
                  (m0 & m1).sum()]
    return out


def shares(P, floor):
    """v14_analysis.gap_closed: means over runs first, then the floor-referenced share"""
    w0, w1, s0, s1, c = P[..., 0], P[..., 1], P[..., 2], P[..., 3], P[..., 4]
    w0, w1, s0, s1, c = (x.mean(-1) for x in (w0, w1, s0, s1, c))
    sw = (w0 - w1) / (w0 - floor)
    ss = (s1 - s0) / (c - s0)
    return sw, ss


def summarise(P, idx, floor):
    sw, ss = shares(P, floor)
    bw, bs = shares(P[idx], floor)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"wer": float(sw), "wer_ci": q(bw), "sim": float(ss), "sim_ci": q(bs),
            "ratio": float(sw / ss), "ratio_ci": q(bw / bs),
            "diff_pp": float(100 * (sw - ss)), "diff_pp_ci": q(100 * (bw - bs)),
            "p_boot_ratio_gt_1": float(np.mean(bw / bs > 1)),
            "ordering_holds": bool(sw > ss and np.percentile(bw - bs, 2.5) > 0),
            "mean_items_per_run": float(P[:, 5].mean()),
            "min_items_per_run": int(P[:, 5].min())}


def scope_degenerate(fl, n_boot):
    df = pd.read_csv(os.path.join(REPO, "results", "artifacts", "runs.csv"))
    runs = [f"{c}_{s}" for c, s in sorted(df.groupby(["config", "seed"]).groups)]
    W, S, G, ids = cube(runs, T_ALL)
    C, cf = fl["sim_ceiling_item"], fl["sim_ceiling"]
    ti = {T: k for k, T in enumerate(T_ALL)}
    grid_idx = [ti[T] for T in T_GRID]
    # the published headline CI draws one 2000x45 block per T in (2,4,8,16,32,64); the
    # T=16 block is the 4th. Reusing it gives every variant the published resamples.
    rng = np.random.default_rng(BOOT_RNG)
    blocks = [np.array([rng.integers(0, len(runs), len(runs)) for _ in range(n_boot)])
              for _ in (2, 4, 8, 16, 32, 64)]
    idx = blocks[3]
    res = {"n_runs": len(runs), "n_items": len(ids), "T_end": 16,
           "bootstrap": "run-level, 45 runs, the T=16 resamples of v14_analysis.gap_closed",
           "wer_floor_note": "per-item ASR floors are not stored; every variant uses the "
                             "400-item floor. 'floor0' re-runs a variant with floor 0, the "
                             "most conservative value for the intelligibility share.",
           "variants": {}}
    V = res["variants"]
    V["headline_T1"] = summarise(per_run(W, S, G, C, ti[1], ti[16], "all", cfix=cf), idx, fl["wer_floor"])
    for t0 in (2, 4):
        V[f"from_T{t0}"] = summarise(per_run(W, S, G, C, ti[t0], ti[16], "all", cfix=cf), idx,
                                     fl["wer_floor"])
    for mode in ("nondeg_unpaired", "nondeg_paired", "nondeg_allT"):
        V[f"T1_{mode}"] = summarise(per_run(W, S, G, C, ti[1], ti[16], mode, grid_idx, cf), idx,
                                    fl["wer_floor"])
    V["T1_nondeg_allT_floor0"] = summarise(
        per_run(W, S, G, C, ti[1], ti[16], "nondeg_allT", grid_idx), idx, 0.0)
    # items that are non-degenerate in EVERY run at EVERY grid T: one fixed item set
    keep = ~G[:, grid_idx].any((0, 1))
    res["n_items_nondeg_every_run_every_T"] = int(keep.sum())
    if keep.sum() >= 20:
        Gk = G.copy()
        Gk[:, :, ~keep] = True
        V["T1_fixed_itemset"] = summarise(
            per_run(W, S, Gk, C, ti[1], ti[16], "nondeg_allT", grid_idx), idx, fl["wer_floor"])
    # headline reproduction: must equal results/artifacts-v1.4/analysis.json
    pub = json.load(open(os.path.join(REPO, "results", "artifacts-v1.4", "analysis.json")))
    pb = pub["B_gap_closed"]["by_T"]["16"]
    h = V["headline_T1"]
    res["reproduces_published_headline"] = bool(
        abs(h["wer"] - pb["wer"]["point"]) < 1e-12 and abs(h["sim"] - pb["sim"]["point"]) < 1e-12
        and np.allclose(h["wer_ci"], pb["wer"]["ci"], rtol=0, atol=1e-12)
        and np.allclose(h["sim_ci"], pb["sim"]["ci"], rtol=0, atol=1e-12))

    # how much of the T=1 -> 16 WER drop comes from items degenerate at T=1
    d1 = G[:, ti[1]]
    drop = W[:, ti[1]] - W[:, ti[16]]
    res["T1_pooled"] = {
        "degen_share_items": float(d1.mean()),
        "wer_T1_degenerate_items": float(np.nanmean(W[:, ti[1]][d1])),
        "wer_T1_nondegenerate_items": float(np.nanmean(W[:, ti[1]][~d1])),
        "share_of_wer_drop_from_T1_degenerate_items": float(np.nansum(drop[d1]) / np.nansum(drop)),
        "sim_T1_degenerate_items": float(np.nanmean(S[:, ti[1]][d1])),
        "sim_T1_nondegenerate_items": float(np.nanmean(S[:, ti[1]][~d1])),
    }
    # degenerate rate by T and budget
    bud = {f"{c}_{s}": b for c, s, b in df[["config", "seed", "budget"]].drop_duplicates().values}
    rate = {}
    for b in sorted(set(bud.values())):
        rr = [k for k, r in enumerate(runs) if bud[r] == b]
        rate[b] = {T: float(G[rr, ti[T]].mean()) for T in T_ALL}
    rate["all_30k"] = {T: float(G[:, ti[T]].mean()) for T in T_ALL}
    d4 = pd.read_csv(os.path.join(REPO, "results", "artifacts-v1.1", "runs_4budget.csv"))
    rate["D"] = {int(T): float(v) for T, v in
                 d4[d4.budget == "D"].groupby("T").degen_rate.mean().items()}
    res["degen_rate"] = rate
    return res, (W, S, G, runs)


def scope_90k(fl):
    """The same variants on the 9 runs at 90k (and their 30k twins), with the v15 bootstrap."""
    cfg, seeds = ("C1", "C3", "C5"), (0, 1, 2)
    C = fl["sim_ceiling_item"]
    ti = {T: k for k, T in enumerate(T_ALL)}
    grid_idx = [ti[T] for T in T_GRID]
    out = {"bootstrap": "paper_v15: resample configs (all seeds of a drawn config) and "
                        "items, 1000 replicates, RNG 7331"}
    for bud, suf in (("30k", ""), ("90k", "_90k")):
        runs = [f"{c}_{s}{suf}" for c in cfg for s in seeds]
        W, S, G, _ = cube(runs, T_ALL)
        variants = {"headline_T1": (ti[1], "all"), "from_T2": (ti[2], "all"),
                    "from_T4": (ti[4], "all"), "T1_nondeg_paired": (ti[1], "nondeg_paired"),
                    "T1_nondeg_allT": (ti[1], "nondeg_allT")}
        rng = np.random.default_rng(BOOT_RNG)
        reps = [(rng.integers(0, 3, 3), rng.integers(0, 400, 400)) for _ in range(V15_BOOT)]
        res = {}
        for name, (b0, mode) in variants.items():
            def sh(ci, ii):
                rr = [3 * c + s for c in ci for s in range(3)]
                P = per_run(W[rr][:, :, ii], S[rr][:, :, ii], G[rr][:, :, ii], C[ii], b0,
                            ti[16], mode, grid_idx, fl["sim_ceiling"])
                return shares(P, fl["wer_floor"])
            sw, ss = sh(np.arange(3), np.arange(400))
            bw, bs = np.array([sh(c, i) for c, i in reps]).T
            q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            res[name] = {"wer": float(sw), "sim": float(ss), "ratio": float(sw / ss),
                         "ratio_ci": q(bw / bs), "diff_pp_ci": q(100 * (bw - bs)),
                         "ordering_holds": bool(sw > ss and np.percentile(bw - bs, 2.5) > 0)}
        res["degen_rate"] = {T: float(G[:, ti[T]].mean()) for T in T_ALL}
        out[bud] = res
    # 180k exists locally for one run only; report its degenerate rate, nothing pooled
    c180 = {T: _scores("C3_0_180k", T)["summary"]["degen_rate"] for T in T_ALL}
    out["180k_C3_0_degen_rate"] = c180
    return out


# --------------------------------------------------------------------- (b) anchor
def anchor(fl, n_boot):
    f5 = json.load(open(os.path.join(REPO, "results", "runs-v1.5", "f5tts_anchor", "synth_T32",
                                     "scores.json")))
    it = sorted(f5["items"], key=lambda d: d["item"])
    w = np.array([d["wer"] for d in it]); s = np.array([d["sim"] for d in it])
    u = np.array([d["utmos"] for d in it], float)
    rng = np.random.default_rng(BOOT_RNG)
    ii = rng.integers(0, len(it), (n_boot, len(it)))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    a2 = json.load(open(os.path.join(REPO, "results", "runs-v1.5", "f5tts_anchor", "synth_T32",
                                     "asr2.json")))
    out = {"label": "CALIBRATION ANCHOR, not a competitive baseline (src/anchor_f5.py)",
           "system": "F5-TTS v1 Base, NFE 32, 400/400 items, frozen metric stack",
           "sim_model": f5["summary"]["sim_model"], "asr_model": f5["summary"]["asr_model"],
           "f5": {"wer": float(w.mean()), "wer_ci": q(w[ii].mean(1)),
                  "sim": float(s.mean()), "sim_ci": q(s[ii].mean(1)),
                  "utmos": float(np.nanmean(u)), "utmos_ci": q(np.nanmean(u[ii], 1)),
                  "degen_rate": f5["summary"]["degen_rate"],
                  "wer_asr2": float(a2["wer_mean"])},
           "bounds": {"wer_floor_real_audio": fl["wer_floor"],
                      "sim_codec_ceiling": fl["sim_ceiling"],
                      "sim_real_audio": fl["sim_real_audio"],
                      "utmos_real_audio": fl["utmos_real_audio"],
                      "wer_floor_real_audio_asr2": fl["wer_floor_asr2"]}}
    b = out["f5"]
    out["f5_minus_bounds"] = {
        "wer_vs_floor": b["wer"] - fl["wer_floor"],
        "sim_vs_codec_ceiling": b["sim"] - fl["sim_ceiling"],
        "sim_vs_real_audio": b["sim"] - fl["sim_real_audio"],
        "wer_asr2_vs_floor_asr2": b["wer_asr2"] - fl["wer_floor_asr2"],
        "f5_beats_wer_floor": bool(b["wer_ci"][1] < fl["wer_floor"]),
        "f5_beats_sim_ceiling": bool(b["sim_ci"][0] > fl["sim_ceiling"])}
    # our systems, same items, same stack, T=16
    df = pd.read_csv(os.path.join(REPO, "results", "artifacts", "runs.csv"))
    t16 = df[df["T"] == 16]
    cm = t16.groupby("config")[["wer", "sim", "utmos"]].mean()
    d4 = pd.read_csv(os.path.join(REPO, "results", "artifacts-v1.1", "runs_4budget.csv"))
    dd = d4[(d4.budget == "D") & (d4["T"] == 16)]
    dm = dd.groupby("config")[["wer", "sim", "utmos"]].mean()
    r90 = [f"{c}_{s}_90k" for c in ("C1", "C3", "C5") for s in (0, 1, 2)]
    s90 = [_scores(r, 16)["summary"] for r in r90]
    s180 = _scores("C3_0_180k", 16)["summary"]
    ours = {
        "grid_30k_mean": {k: float(t16[k].mean()) for k in ("wer", "sim", "utmos")},
        "grid_30k_best_sim_config": {"config": cm.sim.idxmax(),
                                     **{k: float(cm.loc[cm.sim.idxmax(), k]) for k in cm}},
        "D_276M_best_sim_config": {"config": dm.sim.idxmax(),
                                   **{k: float(dm.loc[dm.sim.idxmax(), k]) for k in dm}},
        "c90k_mean": {"wer": float(np.mean([x["wer_mean"] for x in s90])),
                      "sim": float(np.mean([x["sim_mean"] for x in s90])),
                      "utmos": float(np.mean([x["utmos_mean"] for x in s90]))},
        "c90k_best_run": max(({"run": r, "wer": x["wer_mean"], "sim": x["sim_mean"],
                               "utmos": x["utmos_mean"]} for r, x in zip(r90, s90)),
                             key=lambda d: d["sim"]),
        "c180k_C3_0": {"wer": s180["wer_mean"], "sim": s180["sim_mean"],
                       "utmos": s180["utmos_mean"]},
    }
    # the second recogniser, on our 30k/90k twins and the one local 180k run
    a2w = {}
    for tag, rr in (("30k", [f"{c}_{s}" for c in ("C1", "C3", "C5") for s in (0, 1, 2)]),
                    ("90k", r90), ("180k_C3_0", ["C3_0_180k"])):
        a2w[tag] = float(np.mean([_asr2(r, 16)["wer_mean"] for r in rr]))
    ours["wer_asr2_T16"] = a2w
    out["ours_T16"] = ours
    # share of the remaining identity distance: best system to F5
    best = ours["c90k_best_run"]["sim"]
    out["identity_distance"] = {
        "best_ours_to_codec_ceiling": fl["sim_ceiling"] - best,
        "best_ours_to_f5": b["sim"] - best,
        "best_ours_fraction_of_f5_sim": best / b["sim"]}
    return out


def headline_against_f5(fl, W, S, G, runs, n_boot):
    """The v14 share with F5's measured WER and SIM-o as the empirical bound."""
    an = json.load(open(os.path.join(REPO, "results", "runs-v1.5", "f5tts_anchor", "synth_T32",
                                     "scores.json")))["summary"]
    ti = {T: k for k, T in enumerate(T_ALL)}
    rng = np.random.default_rng(BOOT_RNG)
    blocks = [np.array([rng.integers(0, len(runs), len(runs)) for _ in range(n_boot)])
              for _ in range(6)]
    C = np.full(400, an["sim_mean"])
    out = {}
    for t0 in (1, 2):
        P = per_run(W, S, G, C, ti[t0], ti[16], "all", cfix=an["sim_mean"])
        out[f"from_T{t0}"] = summarise(P, blocks[3], an["wer_mean"])
    out["note"] = ("F5 is not on our codec, so this bound is task-level for this item set "
                   "rather than reachable by a Mimi-token model; it is reported beside the "
                   "published floors, not instead of them")
    return out


# ------------------------------------------------------------ (c) identification
def _bootrep(args):
    rep, df, se_maps = args
    rng = np.random.default_rng([F.BOOT_RNG, rep])      # identical to fit._boot_one
    parts = []
    for cfg, sub in df.groupby("config"):
        seeds = sorted(sub.seed.unique())
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        for k, sd in enumerate(pick):
            take = sub[sub.seed == sd].copy()
            take["seed"] = 1000 + k
            parts.append(take)
    bdf = pd.concat(parts, ignore_index=True)
    res = {"rep": rep}
    for m in F.METRICS:
        a = F.part_a(bdf, m, F.N_STARTS, se_maps[m])
        b = F.part_b(bdf, m, F.N_STARTS, se_maps[m])
        res[m] = {"A_full": a["M_full"]["params"]["A"], "A_sep": b["M_sep"]["params"]["A"],
                  "alpha_full": a["M_full"]["params"]["alpha"],
                  "kappa": b["M_sub"]["params"]["kappa"]}
    return res


def amplitude_bootstrap(n_boot):
    """fits.json stores alpha/beta/kappa per replicate but not the amplitude A, so the
    'A at its cap in every replicate' statement needs the replicates refitted. Same data,
    same replicate RNG and weights as fit.bootstrap, so kappa must match the stored draws."""
    df = pd.read_csv(os.path.join(REPO, "results", "artifacts", "runs.csv"))
    df = df[df.seed.isin([0, 1, 2])]
    se_maps = {m: {(c, int(t)): e for c, t, e in
                   zip(*[F.surface(df, m)[k] for k in ("config", "T", "se")])} for m in F.METRICS}
    with mp.Pool(max(1, (os.cpu_count() or 4) - 2)) as pool:
        outs = pool.map(_bootrep, [(r, df, se_maps) for r in range(n_boot)], chunksize=4)
    return outs


def identification(boot):
    fits = json.load(open(os.path.join(REPO, "results", "artifacts", "fits.json")))
    lo_hi = {k: (v[2][0], v[2][1]) for k, v in F.FORMS.items()}
    AHI = lo_hi["M_full"][1][1]                          # amplitude cap (10)
    KHI = lo_hi["M_sub"][1][5]                           # kappa upper bound (3)
    EHI = F.EXP_HI                                       # exponent upper bound (3)
    near = lambda v, b, tol=1e-3: abs(v - b) <= tol * max(1.0, abs(b))

    # every M_sub / M_full / M_sep fit committed under results/artifacts*/
    tally = {"M_sub_fits": 0, "M_sub_kappa_at_upper": 0,
             "amp_fits": 0, "amp_at_cap": 0, "amp_at_zero": 0, "files": []}
    def walk(d, f):
        if isinstance(d, dict):
            for k, v in d.items():
                if k in ("dist", "pred", "obs"):
                    continue
                if isinstance(v, dict) and "params" in v and isinstance(v["params"], dict):
                    p = v["params"]
                    if k == "M_sub" and "kappa" in p:
                        tally["M_sub_fits"] += 1
                        tally["M_sub_kappa_at_upper"] += near(p["kappa"], KHI, 1e-6)
                    if k in ("M_full", "M_sep") and "A" in p and "logA" not in p:
                        tally["amp_fits"] += 1
                        tally["amp_at_cap"] += near(p["A"], AHI)
                        tally["amp_at_zero"] += abs(p["A"]) < 1e-6
                walk(v, f)
    for f in sorted(glob.glob(os.path.join(REPO, "results", "artifacts*", "fits*.json"))):
        walk(json.load(open(f)), f)
        tally["files"].append(os.path.relpath(f, REPO))
    lg = json.load(open(os.path.join(REPO, "results", "artifacts-v1.1", "logamp_refit.json")))
    for m in ("wer", "sim"):
        tally["M_sub_fits"] += 1
        tally["M_sub_kappa_at_upper"] += near(lg["part_b"][m]["kappa"], KHI, 1e-6)
    tally["files"].append("results/artifacts-v1.1/logamp_refit.json (kappa only)")

    dist = fits["bootstrap"]["dist"]
    out = {"bounds": {"amplitude": [0.0, AHI],
                      "exponent": [F.EXP_LO, EHI], "kappa": [-KHI, KHI], "E": [0.0, 1.0]},
           "kappa_point": {m: fits["part_b"][m]["kappa"] for m in ("wer", "sim")},
           "kappa_boot_at_upper": {m: int(sum(near(v, KHI, 1e-6) for v in dist[f"{m}_kappa"]))
                                   for m in ("wer", "sim")},
           "n_boot": int(fits["bootstrap"]["n_reps_ok"]),
           "A_point": {"part_a": {m: fits["part_a"][m]["M_full"]["params"]["A"]
                                  for m in ("wer", "sim")},
                       "part_b": {m: fits["part_b"][m]["M_sep"]["params"]["A"]
                                  for m in ("wer", "sim")}},
           "E_point_part_b": {m: fits["part_b"][m]["M_sep"]["params"]["E"] for m in ("wer", "sim")},
           "all_committed_fits": tally}
    # goodness of fit of BOTH forms, by the weighted criterion each minimises
    gof = {}
    for m in ("wer", "sim"):
        for form in ("M_sep", "M_sub"):
            r = fits["part_b"][m][form]
            gof[f"{m}_{form}"] = r["rss_weighted"] / (r["n"] - r["k"])
    out["chi2_per_dof"] = gof
    # unbounded amplitudes move the problem from A to alpha for WER
    out["logamp"] = {
        "amplitude_bounds_log": lg["amplitude_bounds_log"],
        "alpha_wer_part_a": lg["part_a"]["wer"]["M_full_params"]["alpha"],
        "alpha_wer_part_a_at_bound": lg["part_a"]["wer"]["M_full_at_bound"]["alpha"],
        "alpha_wer_part_b": lg["part_b"]["wer"]["M_sep_params"]["alpha"],
        "alpha_wer_part_b_at_bound": lg["part_b"]["wer"]["M_sep_at_bound"]["alpha"],
        "alpha_wer_boot_ci": lg["bootstrap"]["wer_alpha"],
        "alpha_sim_part_a": lg["part_a"]["sim"]["M_full_params"]["alpha"],
        "alpha_sim_boot_ci": lg["bootstrap"]["sim_alpha"],
        "A_wer_part_a": lg["part_a"]["wer"]["M_full_params"]["A"]}
    if boot:
        kw = {m: np.sort([b[m]["kappa"] for b in boot]) for m in ("wer", "sim")}
        out["amp_boot"] = {
            "n": len(boot),
            "kappa_matches_stored": {m: bool(np.allclose(kw[m], np.sort(dist[f"{m}_kappa"])))
                                     for m in ("wer", "sim")} if len(boot) == len(dist["wer_kappa"]) else None,
            "A_full_at_cap": {m: int(sum(near(b[m]["A_full"], AHI) for b in boot)) for m in ("wer", "sim")},
            "A_sep_at_cap": {m: int(sum(near(b[m]["A_sep"], AHI) for b in boot)) for m in ("wer", "sim")},
            "A_full_at_zero": {m: int(sum(abs(b[m]["A_full"]) < 1e-6 for b in boot)) for m in ("wer", "sim")},
            "kappa_at_upper": {m: int(sum(near(b[m]["kappa"], KHI, 1e-6) for b in boot)) for m in ("wer", "sim")}}
    return out


def depth_argmin():
    """argmin over the five shapes of each budget, per T and metric, from config means."""
    ext = pd.read_csv(os.path.join(REPO, "results", "artifacts-v1.4", "runs_extended.csv"))
    d4 = pd.read_csv(os.path.join(REPO, "results", "artifacts-v1.1", "runs_4budget.csv"))
    d4 = d4[d4.budget == "D"]
    out = {}
    for name, df in (("ABC", ext), ("D", d4)):
        for (b, T), g in df.groupby(["budget", "T"]):
            for m, sign in (("wer", 1), ("sim", -1)):
                cm = g.groupby(["config", "depth"])[m].agg(["mean", "std", "count"]).reset_index()
                cm = cm.sort_values("depth")
                order = cm.sort_values("mean", ascending=(sign == 1))
                best, second = order.iloc[0], order.iloc[1]
                se = float(np.sqrt((cm["std"] ** 2 / cm["count"]).mean()))
                out.setdefault(b, {}).setdefault(int(T), {})[m] = {
                    "argmin_depth": int(best.depth), "deepest": int(cm.depth.max()),
                    "shallowest": int(cm.depth.min()),
                    "at_deepest": bool(best.depth == cm.depth.max()),
                    "runner_up_depth": int(second.depth),
                    "margin": float(abs(best["mean"] - second["mean"])),
                    "se_config_mean": se, "n_seeds": int(cm["count"].min())}
    return out


# ------------------------------------------------------------------------ (d) scale
def scale(fl, dataset_json):
    df = pd.read_csv(os.path.join(REPO, "results", "artifacts", "runs.csv"))
    d4 = pd.read_csv(os.path.join(REPO, "results", "artifacts-v1.1", "runs_4budget.csv"))
    grid = json.load(open(os.path.join(REPO, "configs", "grid.json")))
    out = {"steps": grid["training"]["steps"], "batch": grid["training"]["batch_sequences"]}
    if dataset_json and os.path.exists(dataset_json):
        ds = json.load(open(dataset_json))
        seq = out["steps"] * out["batch"]
        out["train_clips"] = ds["train_clips"]
        out["epochs"] = {k: f * seq / ds["train_clips"] for k, f in
                         (("30k", 1), ("90k", 3), ("180k", 6))}
    # the share by parameter budget at 30k (A, B, C from the grid; D from the 4-budget run)
    rng = np.random.default_rng(BOOT_RNG)
    by = {}
    for b, g in pd.concat([df[["config", "seed", "budget", "T", "wer", "sim", "n_nonembed"]],
                           d4[d4.budget == "D"][["config", "seed", "budget", "T", "wer", "sim",
                                                 "n_nonembed"]]]).groupby("budget"):
        piv = g.pivot_table(index=["config", "seed"], columns="T", values=["wer", "sim"])
        P = np.stack([piv["wer"][1], piv["wer"][16], piv["sim"][1], piv["sim"][16],
                      np.full(len(piv), fl["sim_ceiling"]), np.full(len(piv), 400)], 1)
        idx = np.array([rng.integers(0, len(P), len(P)) for _ in range(N_BOOT)])
        by[b] = {"n_runs": len(P), "n_nonembed_M": float(g.n_nonembed.mean() / 1e6),
                 **summarise(P, idx, fl["wer_floor"])}
    out["share_by_param_budget"] = by
    out["n_nonembed_max_M"] = {"grid": float(df.n_nonembed.max() / 1e6),
                               "D": float(d4[d4.budget == "D"].n_nonembed.max() / 1e6)}
    return out


# --------------------------------------------------------------------------- tex
def emit(res, tex):
    M = []
    add = lambda k, v: M.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    pct = lambda x: f"{100 * x:.1f}"
    ci_pct = lambda c: f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"
    rx = lambda x: f"{x:.2f}$\\times$"
    ci_r = lambda c: f"[{c[0]:.2f}, {c[1]:.2f}]"
    W = {"from_T2": "FromTwo", "from_T4": "FromFour", "T1_nondeg_unpaired": "NdUnpaired",
         "T1_nondeg_paired": "NdPaired", "T1_nondeg_allT": "NdAllT",
         "T1_nondeg_allT_floor0": "NdAllTFloorZero", "T1_fixed_itemset": "NdFixed"}
    a = res["a_degenerate"]
    for k, nm in W.items():
        if k not in a["variants"]:
            continue
        v = a["variants"][k]
        add(f"Ncrs{nm}WER", pct(v["wer"])); add(f"Ncrs{nm}WERci", ci_pct(v["wer_ci"]))
        add(f"Ncrs{nm}SIM", pct(v["sim"])); add(f"Ncrs{nm}SIMci", ci_pct(v["sim_ci"]))
        add(f"Ncrs{nm}Ratio", rx(v["ratio"])); add(f"Ncrs{nm}Ratioci", ci_r(v["ratio_ci"]))
        add(f"Ncrs{nm}Items", f"{v['mean_items_per_run']:.0f}")
    vv = a["variants"]
    add("NcrsNdMinItems", min(vv[k]["min_items_per_run"] for k in W
                              if k in vv and k.startswith("T1_nondeg")))
    add("NcrsOrderingVariants", sum(vv[k]["ordering_holds"] for k in W if k in vv))
    add("NcrsNVariants", sum(1 for k in W if k in vv))
    add("NcrsFixedItems", a["n_items_nondeg_every_run_every_T"])
    p = a["T1_pooled"]
    add("NcrsDegenShareTone", pct(p["degen_share_items"]))
    add("NcrsWerToneDegen", f"{p['wer_T1_degenerate_items']:.3f}")
    add("NcrsWerToneNondegen", f"{p['wer_T1_nondegenerate_items']:.3f}")
    add("NcrsDropFromDegen", pct(p["share_of_wer_drop_from_T1_degenerate_items"]))
    WT = {1: "One", 2: "Two", 4: "Four", 8: "Eight", 16: "Sixteen", 32: "Thirtytwo",
          64: "Sixtyfour"}
    for b, row in a["degen_rate"].items():
        tag = {"all_30k": "All"}.get(b, b)
        for T, v in row.items():
            add(f"NcrsDegen{tag}T{WT[int(T)]}", pct(v))
    n9 = res["a_90k"]
    for bud, tag in (("30k", "Thirtyk"), ("90k", "Ninetyk")):
        for k, nm in (("headline_T1", "Head"), ("from_T2", "FromTwo"), ("from_T4", "FromFour"),
                      ("T1_nondeg_paired", "NdPaired"), ("T1_nondeg_allT", "NdAllT")):
            v = n9[bud][k]
            add(f"Ncrs{tag}{nm}WER", pct(v["wer"])); add(f"Ncrs{tag}{nm}SIM", pct(v["sim"]))
            add(f"Ncrs{tag}{nm}Ratio", rx(v["ratio"]))
            add(f"Ncrs{tag}{nm}Ratioci", ci_r(v["ratio_ci"]))
        add(f"NcrsDegen{tag}TOne", pct(n9[bud]["degen_rate"][1]))
        add(f"NcrsDegen{tag}TTwo", pct(n9[bud]["degen_rate"][2]))
        add(f"NcrsDegen{tag}TSixteen", pct(n9[bud]["degen_rate"][16]))
    add("NcrsNinetykOrdering", sum(n9["90k"][k]["ordering_holds"] for k in n9["90k"]
                                   if k != "degen_rate"))
    # (b)
    b = res["b_anchor"]; f5 = b["f5"]; o = b["ours_T16"]
    add("NcrsFfWER", f"{f5['wer']:.4f}"); add("NcrsFfWERci", f"[{f5['wer_ci'][0]:.4f}, {f5['wer_ci'][1]:.4f}]")
    add("NcrsFfSIM", f"{f5['sim']:.4f}"); add("NcrsFfSIMci", f"[{f5['sim_ci'][0]:.4f}, {f5['sim_ci'][1]:.4f}]")
    add("NcrsFfUTMOS", f"{f5['utmos']:.2f}"); add("NcrsFfWERasrTwo", f"{f5['wer_asr2']:.4f}")
    add("NcrsFfDegen", pct(f5["degen_rate"]))
    add("NcrsSimRealAudio", f"{b['bounds']['sim_real_audio']:.4f}")
    add("NcrsUtmosRealAudio", f"{b['bounds']['utmos_real_audio']:.2f}")
    add("NcrsFfOverCeiling", f"{b['f5_minus_bounds']['sim_vs_codec_ceiling']:+.4f}")
    add("NcrsFfUnderFloor", f"{b['f5_minus_bounds']['wer_vs_floor']:+.4f}")
    for k, nm in (("grid_30k_mean", "GridMean"), ("c90k_mean", "NinetykMean"),
                  ("c180k_C3_0", "EightykCthree")):
        add(f"Ncrs{nm}WER", f"{o[k]['wer']:.4f}"); add(f"Ncrs{nm}SIM", f"{o[k]['sim']:.4f}")
        add(f"Ncrs{nm}UTMOS", f"{o[k]['utmos']:.2f}")
    for k, nm in (("grid_30k_best_sim_config", "GridBest"), ("D_276M_best_sim_config", "DBest"),
                  ("c90k_best_run", "NinetykBest")):
        add(f"Ncrs{nm}Name", o[k].get("config", o[k].get("run", "")).replace("_", "\\_"))
        add(f"Ncrs{nm}WER", f"{o[k]['wer']:.4f}"); add(f"Ncrs{nm}SIM", f"{o[k]['sim']:.4f}")
        add(f"Ncrs{nm}UTMOS", f"{o[k]['utmos']:.2f}")
    for k, nm in (("30k", "Thirtyk"), ("90k", "Ninetyk"), ("180k_C3_0", "EightykCthree")):
        add(f"Ncrs{nm}WERasrTwo", f"{o['wer_asr2_T16'][k]:.4f}")
    add("NcrsBestToFf", f"{b['identity_distance']['best_ours_to_f5']:.4f}")
    hf = res["b_headline_vs_f5"]["from_T1"]
    add("NcrsFfBoundWER", pct(hf["wer"])); add("NcrsFfBoundSIM", pct(hf["sim"]))
    add("NcrsFfBoundRatio", rx(hf["ratio"])); add("NcrsFfBoundRatioci", ci_r(hf["ratio_ci"]))
    # (c)
    c = res["c_identification"]
    add("NcrsKappaBound", f"{c['bounds']['kappa'][1]:.0f}")
    add("NcrsAmpCap", f"{c['bounds']['amplitude'][1]:.0f}")
    add("NcrsExpCap", f"{c['bounds']['exponent'][1]:.0f}")
    add("NcrsKappaBootWER", c["kappa_boot_at_upper"]["wer"])
    add("NcrsKappaBootSIM", c["kappa_boot_at_upper"]["sim"])
    add("NcrsNboot", c["n_boot"])
    t = c["all_committed_fits"]
    add("NcrsSubFits", t["M_sub_fits"]); add("NcrsSubFitsAtBound", t["M_sub_kappa_at_upper"])
    add("NcrsAmpFits", t["amp_fits"]); add("NcrsAmpFitsAtCap", t["amp_at_cap"])
    add("NcrsAmpFitsAtZero", t["amp_at_zero"])
    add("NcrsApartAWER", f"{c['A_point']['part_a']['wer']:.3f}")
    add("NcrsApartASIM", f"{c['A_point']['part_a']['sim']:.3f}")
    add("NcrsApartBWER", f"{c['A_point']['part_b']['wer']:.3f}")
    add("NcrsApartBSIM", f"{c['A_point']['part_b']['sim']:.3f}")
    g = c["chi2_per_dof"]
    add("NcrsChiSubWER", f"{g['wer_M_sub']:.1f}"); add("NcrsChiSubSIM", f"{g['sim_M_sub']:.1f}")
    add("NcrsChiSepWER", f"{g['wer_M_sep']:.1f}"); add("NcrsChiSepSIM", f"{g['sim_M_sep']:.1f}")
    lg = c["logamp"]
    add("NcrsLogampAlphaWER", f"{lg['alpha_wer_part_b']:.2f}")
    add("NcrsLogampAlphaWERci", f"[{lg['alpha_wer_boot_ci'][0]:.2f}, {lg['alpha_wer_boot_ci'][1]:.2f}]")
    add("NcrsLogampAlphaSIMci", f"[{lg['alpha_sim_boot_ci'][0]:.2f}, {lg['alpha_sim_boot_ci'][1]:.2f}]")
    if "amp_boot" in c:
        ab = c["amp_boot"]
        add("NcrsAmpBootN", ab["n"])
        for m, nm in (("wer", "WER"), ("sim", "SIM")):
            add(f"NcrsAmpBootFull{nm}", ab["A_full_at_cap"][m])
            add(f"NcrsAmpBootSep{nm}", ab["A_sep_at_cap"][m])
    d = res["c_depth"]
    deep = [(b, T) for b in ("A", "B", "C") for T in T_GRID if d[b][T]["wer"]["at_deepest"]]
    add("NcrsDeepCellsWER", len(deep)); add("NcrsDepthCells", 3 * len(T_GRID))
    deeps = [(b, T) for b in ("A", "B", "C") for T in T_GRID if d[b][T]["sim"]["at_deepest"]]
    add("NcrsDeepCellsSIM", len(deeps))
    for b in ("A", "B", "C", "D"):
        for m, nm in (("wer", "WER"), ("sim", "SIM")):
            e = d[b][16][m]
            add(f"NcrsDstar{b}{nm}", e["argmin_depth"])
            add(f"NcrsDstarMargin{b}{nm}", f"{100 * e['margin'] if m == 'wer' else e['margin']:.{1 if m == 'wer' else 4}f}")
            add(f"NcrsDstarSE{b}{nm}", f"{100 * e['se_config_mean'] if m == 'wer' else e['se_config_mean']:.{1 if m == 'wer' else 4}f}")
            # SE of the difference between two configuration means (best minus runner-up),
            # sqrt(2) x the pooled SE of one mean: the scale on which the margin is judged
            sed = float(np.sqrt(2.0) * e["se_config_mean"])
            add(f"NcrsDstarSEdiff{b}{nm}", f"{100 * sed if m == 'wer' else sed:.{1 if m == 'wer' else 4}f}")
        add(f"NcrsDeepest{b}", d[b][16]["wer"]["deepest"])
    # (d)
    s = res["d_scale"]
    if "epochs" in s:
        for k, nm in (("30k", "Thirtyk"), ("90k", "Ninetyk"), ("180k", "Eightyk")):
            add(f"NcrsEpochs{nm}", f"{s['epochs'][k]:.1f}")
    for bb, v in s["share_by_param_budget"].items():
        add(f"NcrsBudget{bb}Ratio", rx(v["ratio"])); add(f"NcrsBudget{bb}Ratioci", ci_r(v["ratio_ci"]))
        add(f"NcrsBudget{bb}WER", pct(v["wer"])); add(f"NcrsBudget{bb}SIM", pct(v["sim"]))
    hdr = ("% GENERATED by src/camera_ready_scope.py from committed artifacts "
           "(results/artifacts-camera/scope.json). Do not edit; rerun the script.\n")
    open(tex, "w").write(hdr + "\n".join(M) + "\n")
    return len(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=os.path.join(OUT, "numbers_cr_scope.tex"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--amp-boot", type=int, default=0,
                    help="refit replicates to count A at its cap (0 skips; cached)")
    ap.add_argument("--dataset-json", default=os.path.join(
        os.path.dirname(REPO), "hf-repo", "dataset.json"),
        help="released dataset.json (train clip count) for the epochs figure")
    ap.add_argument("--from-json", action="store_true",
                    help="re-emit the macros from the saved results/artifacts-camera/scope.json only")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.from_json:
        def _intkeys(o):
            if isinstance(o, dict):
                return {(int(k) if isinstance(k, str) and k.isdigit() else k): _intkeys(v)
                        for k, v in o.items()}
            return [_intkeys(v) for v in o] if isinstance(o, list) else o
        res = _intkeys(json.load(open(os.path.join(OUT, "scope.json"))))
        n = emit(res, a.tex)
        print(f"[scope] re-emitted {n} macros from scope.json -> {a.tex}")
        return
    fl = floors()
    res = {}
    res["a_degenerate"], (W, S, G, runs) = scope_degenerate(fl, a.n_boot)
    print("[scope] (a) grid variants done; headline reproduced:",
          res["a_degenerate"]["reproduces_published_headline"], flush=True)
    res["a_90k"] = scope_90k(fl)
    print("[scope] (a) 30k/90k twins done", flush=True)
    res["b_anchor"] = anchor(fl, a.n_boot)
    res["b_headline_vs_f5"] = headline_against_f5(fl, W, S, G, runs, a.n_boot)
    cache = os.path.join(OUT, "scope_fit_boot.json")
    boot = None
    if a.amp_boot:
        if os.path.exists(cache) and json.load(open(cache))["n"] == a.amp_boot:
            boot = json.load(open(cache))["reps"]
        else:
            boot = amplitude_bootstrap(a.amp_boot)
            json.dump({"n": a.amp_boot, "reps": boot}, open(cache, "w"))
    res["c_identification"] = identification(boot)
    res["c_depth"] = depth_argmin()
    res["d_scale"] = scale(fl, a.dataset_json)
    fl_out = {k: v for k, v in fl.items() if k != "sim_ceiling_item"}
    res["floors"] = fl_out
    json.dump(res, open(os.path.join(OUT, "scope.json"), "w"), indent=1, default=float)
    n = emit(res, a.tex)
    print(f"[scope] wrote {n} macros -> {a.tex}")
    for k, v in res["a_degenerate"]["variants"].items():
        print(f"  {k:<24} WER {100*v['wer']:5.1f}%  SIM {100*v['sim']:5.1f}%  ratio "
              f"{v['ratio']:.2f}x {v['ratio_ci']}  items/run {v['mean_items_per_run']:.0f}  "
              f"ordering {v['ordering_holds']}")


if __name__ == "__main__":
    main()
