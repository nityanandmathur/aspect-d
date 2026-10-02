"""v1.4 analysis battery: replace a coordinate-bound headline with an invariant one.

Motivation. The v1.0 primary result is Delta-tau = tau_WER - tau_SIM > 0, fitted in
the pre-registered error coordinate (err_WER = WER, err_SIM = 1 - SIM). tau is
invariant to *affine* remappings of the error, which is what v1.0 checked. It is
not invariant to monotone but non-affine ones, and the sign of Delta-tau does not
survive them (A below). A quantity whose sign depends on whether you write WER or
sqrt(WER) cannot carry a claim about systems, so the paper must lead with something
that does.

The replacement is model-free and floor-referenced: the fraction of the *achievable*
range each metric closes between one-step decoding and T = 16, where "achievable"
means measured floors -- the ASR word error on real audio, and the speaker
similarity of a codec round trip -- rather than 0 and 1. Because it is a ratio of
differences on a common axis, any monotone reparameterisation that moves the
numerator moves the denominator with it (B below: the ratio holds at 1.68-1.86x
where Delta-tau flips sign).

    python src/v14_analysis.py            # writes results/artifacts-v1.4/analysis.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict

import numpy as np
import pandas as pd

import fit as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "artifacts-v1.4")
BOOT_RNG, N_BOOT = 7331, 2000


def _load() -> pd.DataFrame:
    return pd.read_csv(os.path.join(REPO, "results", "artifacts", "runs.csv"))


def _floors() -> Dict:
    g0 = json.load(open(os.path.join(REPO, "results", "artifacts", "g0c_groundtruth.json")))
    led = json.load(open(os.path.join(REPO, "results", "artifacts-v1.2",
                                      "identity_ledger.json")))["MEASURED"]
    return {"wer_asr_floor": float(g0["wer_mean_item"]),
            "sim_codec_ceiling": float(led["sim_roundtrip_mean"]),
            "utmos_ground_truth": float(g0.get("utmos_mean_item", np.nan)),
            "note": "floors are MEASURED on the same 400 items through the same stack"}


# ---------------------------------------------------------------- A. coordinate
def coordinate_sensitivity(df: pd.DataFrame) -> Dict:
    """Delta-tau refitted under monotone reparameterisations of the SAME numbers."""
    maps: Dict[str, Callable] = {
        "identity (pre-registered)": lambda x: x,
        "log(1+err)": np.log1p,
        "sqrt(err)": np.sqrt,
        "err capped at 1": lambda x: np.minimum(x, 1.0),
        "err squared": lambda x: x ** 2,
        "affine 3+2*err (v1.0 check)": lambda x: 3.0 + 2.0 * x,
    }
    rows = {}
    for name, t in maps.items():
        d = df.copy()
        d["err_wer"] = t(df.wer.values)
        d["err_sim"] = t(1.0 - df.sim.values)
        tw = F.part_b(d, "wer").get("tau")
        ts = F.part_b(d, "sim").get("tau")
        rows[name] = {"tau_wer": tw, "tau_sim": ts, "delta_tau": tw - ts}
    signs = {np.sign(v["delta_tau"]) for v in rows.values()}
    return {"by_coordinate": rows,
            "sign_is_stable": len(signs) == 1,
            "n_reparameterisations_flipping_sign": int(sum(
                1 for v in rows.values() if v["delta_tau"] < 0)),
            "INTERPRETATION": (
                "tau is affine-invariant, which v1.0 verified, but Delta-tau's sign is "
                "not invariant to monotone non-affine remappings of the error. "
                "Delta-tau is therefore a property of the chosen error coordinate, not "
                "of the systems, and cannot carry the paper's primary claim.")}


# ------------------------------------------------------------------- B. headline
def gap_closed(df: pd.DataFrame, n_boot: int = N_BOOT) -> Dict:
    """Fraction of the achievable range closed from T=1, referenced to measured floors.

    Bootstrap is run-level (resample the 45 runs with replacement) to match the
    declared uncertainty unit in protocol §7.2."""
    fl = _floors()
    rng = np.random.default_rng(BOOT_RNG)
    runs = sorted(df.groupby(["config", "seed"]).groups)
    piv = {r: g.set_index("T") for r, g in df.groupby(["config", "seed"])}

    def frac(sub: list, T: int) -> Dict[str, float]:
        w1 = np.mean([piv[r].wer[1] for r in sub])
        wT = np.mean([piv[r].wer[T] for r in sub])
        s1 = np.mean([piv[r].sim[1] for r in sub])
        sT = np.mean([piv[r].sim[T] for r in sub])
        u1 = np.mean([piv[r].utmos[1] for r in sub])
        uT = np.mean([piv[r].utmos[T] for r in sub])
        out = {"wer": (w1 - wT) / (w1 - fl["wer_asr_floor"]),
               "sim": (sT - s1) / (fl["sim_codec_ceiling"] - s1)}
        if np.isfinite(fl["utmos_ground_truth"]):
            out["utmos"] = (uT - u1) / (fl["utmos_ground_truth"] - u1)
        out["ratio_wer_over_sim"] = out["wer"] / out["sim"]
        return out

    res = {"floors": fl, "by_T": {}}
    Ts = [t for t in (2, 4, 8, 16, 32, 64)
          if all(t in piv[r].index for r in runs)]
    for T in Ts:
        pt = frac(runs, T)
        reps = [frac([runs[i] for i in rng.integers(0, len(runs), len(runs))], T)
                for _ in range(n_boot)]
        entry = {}
        for k in pt:
            v = np.array([r[k] for r in reps])
            entry[k] = {"point": pt[k],
                        "ci": [float(np.percentile(v, 2.5)),
                               float(np.percentile(v, 97.5))]}
        res["by_T"][T] = entry

    # invariance of the ratio under the same maps that flip Delta-tau's sign
    g = df.groupby("T").agg(wer=("wer", "mean"), sim=("sim", "mean"))
    inv = {}
    for nm, t in (("identity", lambda x: x), ("log1p", np.log1p),
                  ("sqrt", np.sqrt), ("square", lambda x: x ** 2)):
        a = ((t(g.wer[1]) - t(g.wer[16])) /
             (t(g.wer[1]) - t(fl["wer_asr_floor"])))
        b = ((t(1 - g.sim[1]) - t(1 - g.sim[16])) /
             (t(1 - g.sim[1]) - t(1 - fl["sim_codec_ceiling"])))
        inv[nm] = {"wer": a, "sim": b, "ratio": a / b}
    res["coordinate_invariance"] = inv
    res["ratio_range_across_coordinates"] = [
        float(min(v["ratio"] for v in inv.values())),
        float(max(v["ratio"] for v in inv.values()))]
    res["INTERPRETATION"] = (
        "Refinement closes most of the intelligibility range that is reachable at all, "
        "and about half of the identity range. Unlike Delta-tau the contrast keeps its "
        "sign and roughly its size under every reparameterisation tried, because "
        "numerator and denominator move together.")
    return res


# ------------------------------------------------------------- C. range / D. CI
def range_sensitivity(df: pd.DataFrame) -> Dict:
    """tau is a local slope: refit on nested T windows.

    The frozen v1.0 surface stops at T = 16, so widening the window needs the v1.4
    extension. `results/artifacts/runs.csv` is immutable, so the extended sweep is collected
    to its own file and only *this* analysis reads it; every declared v1.0 quantity
    still comes from the frozen table."""
    ext = os.path.join(OUT, "runs_extended.csv")
    src = "results/artifacts/runs.csv (frozen, T<=16 only)"
    if os.path.exists(ext):
        e = pd.read_csv(ext)
        if e["T"].max() > df["T"].max():
            df, src = e, f"results/artifacts-v1.4/runs_extended.csv (T<={int(e['T'].max())})"
    out = {"_source": src, "_T_available": sorted(df["T"].unique().tolist())}
    for hi in (8, 16, 32, 64, 128):
        d = df[df["T"] <= hi]
        if d["T"].nunique() < 4 or hi > df["T"].max():
            continue
        tw = F.part_b(d, "wer").get("tau")
        ts = F.part_b(d, "sim").get("tau")
        out[f"T<={hi}"] = {"tau_wer": tw, "tau_sim": ts, "delta_tau": tw - ts,
                           "n_T": int(d["T"].nunique()), "n_rows": int(len(d))}
    return {"windows": {k: v for k, v in out.items() if not k.startswith("_")},
            "source": out["_source"], "T_available": out["_T_available"],
            "INTERPRETATION": (
        "tau changes materially with the fitted window, so it summarises curvature "
        "over the measured range rather than identifying a scaling exponent.")}


def _honest_rep(args):
    df, rep, lo_w, lo_s = args
    rng = np.random.default_rng([BOOT_RNG, rep])
    runs = sorted(df.groupby(["config", "seed"]).groups)
    grp = {r: g for r, g in df.groupby(["config", "seed"])}
    sub = [runs[i] for i in rng.integers(0, len(runs), len(runs))]
    d = pd.concat([grp[r] for r in sub], ignore_index=True)
    d["wer_se"] = np.maximum(d.wer_se, lo_w)
    d["sim_se"] = np.maximum(d.sim_se, lo_s)
    try:
        return F.part_b(d, "wer").get("tau") - F.part_b(d, "sim").get("tau")
    except Exception:
        return None


def honest_ci(df: pd.DataFrame, n_boot: int = 400) -> Dict:
    """Declared CI pins 1/SE^2 weights (protocol §7.2). This also resamples them.

    Weights are floored at the observed minimum SE so a replicate that draws one run
    three times cannot acquire an unbounded weight -- the failure mode the declared
    estimator avoids by pinning. This is a sensitivity check on the declared interval,
    not a replacement estimator, so it runs at fewer replicates."""
    import multiprocessing as mp
    lo_w = float(df.wer_se[df.wer_se > 0].min())
    lo_s = float(df.sim_se[df.sim_se > 0].min())
    with mp.Pool(min(24, os.cpu_count() or 8)) as pool:
        vals = pool.map(_honest_rep, [(df, r, lo_w, lo_s) for r in range(n_boot)])
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    return {"n_reps_ok": int(len(v)),
            "ci_weights_resampled": [float(np.percentile(v, 2.5)),
                                     float(np.percentile(v, 97.5))],
            "sd": float(v.std()),
            "note": "wider than the declared interval; the declared one holds weights "
                    "fixed at the observed-data SEs and so omits weighting uncertainty."}


def goodness_of_fit(df: pd.DataFrame) -> Dict:
    """The pre-registered form, judged by its own weighted criterion."""
    out = {}
    for m in ("wer", "sim"):
        s = F.surface(df, m)
        X, y, se = F.xy(s)
        r = F.fit_form("M_sep", X, y, se)
        dof = int(r["n"] - r["k"])
        out[m] = {"chi2_weighted": float(r["rss_weighted"]), "dof": dof,
                  "chi2_per_dof": float(r["rss_weighted"] / dof)}
    out["INTERPRETATION"] = (
        "chi^2 per degree of freedom far exceeds 1, so the declared functional form is "
        "rejected as a generative model of these measurements. It is used here as a "
        "descriptive summary of curvature, and the paper says so.")
    return out


# ------------------------------------------------------------------- E. X5 scorer
def same_instrument_search(n_boot: int = N_BOOT) -> Dict:
    """Search gain measured with the SAME encoder as the ceiling it is compared to,
    plus a scale-free normalisation so encoders can be placed side by side."""
    p = os.path.join(REPO, "results", "artifacts-v1.3", "multi_encoder.csv")
    if not os.path.exists(p):
        return {"skipped": "multi_encoder.csv absent"}
    df = pd.read_csv(p)
    gates = {"wavlm_large": (0.7005, 0.0338), "ecapa": (0.6606, 0.0598),
             "wavlm_base_plus": (0.9488, 0.6601), "ge2e": (0.8604, 0.5911),
             "xvect": (0.9639, 0.8953)}
    rng = np.random.default_rng(BOOT_RNG)
    out = {}
    for enc, g in df.groupby("encoder"):
        d = (g.search - g.refine).values
        reps = np.array([d[rng.integers(0, len(d), len(d))].mean()
                         for _ in range(n_boot)])
        same, cross = gates[enc]
        out[enc] = {
            "raw_delta": float(d.mean()),
            "raw_ci": [float(np.percentile(reps, 2.5)),
                       float(np.percentile(reps, 97.5))],
            "win_rate": float((d > 0).mean()),
            "normalised_delta": float(d.mean() / (same - cross)),
            "note": "normalised by this encoder's own same-minus-cross-speaker gap, "
                    "which removes the cosine-scale difference between encoders"}
    return {"encoders": out, "INTERPRETATION": (
        "The ceiling in the identity ledger is a WavLM-large quantity, so the row that "
        "is compared against it must be the WavLM-large delta; the other encoders enter "
        "as agreement evidence through win rates and scale-normalised deltas, never as "
        "a numerator over a different encoder's denominator.")}


def data_description() -> Dict:
    from data import PROC_DIR
    ev = json.load(open(os.path.join(PROC_DIR, "eval_zs.json")))
    return {"corpus": "Emilia-EN", "train_hours": 2000,
            "selection_rng": 1234, "train_speakers_min": 4000,
            "heldout_speakers": 200,
            "eval_items": len(ev),
            "eval_speakers": len({e["speaker"] for e in ev}),
            "eval_target_seconds": [4, 15],
            "task": "cross-sentence zero-shot: prompt and target are different "
                    "utterances from the same held-out speaker",
            "codec": "Mimi at 12.5 Hz, 8 of 32 codebooks",
            "asr": "Whisper-large-v3", "sv": "WavLM-large SV (seed-tts-eval)",
            "mos": "UTMOS22-strong"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    df = _load()
    dest = os.path.join(OUT, "analysis.json")
    res = {"n_runs": int(df.groupby(["config", "seed"]).ngroups),
           "T_values": sorted(df["T"].unique().tolist()),
           "data": data_description()}

    def step(key, fn):
        """Write after every section: the slowest section must not be able to
        discard the ones that already succeeded."""
        res[key] = fn()
        with open(dest, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        print(f"[v14] {key} done", flush=True)

    # The headline is reported over the widest T we measured. The extension adds T
    # values on the *same* 45 runs and the same 400 items -- no config or item subset
    # -- so it widens the axis without changing the population being described.
    ext = os.path.join(OUT, "runs_extended.csv")
    df_gap = df
    if os.path.exists(ext):
        e = pd.read_csv(ext)
        if (e.groupby(["config", "seed"]).ngroups
                == df.groupby(["config", "seed"]).ngroups):
            df_gap = e
    step("B_gap_closed", lambda: gap_closed(df_gap, a.n_boot))
    step("A_coordinate_sensitivity", lambda: coordinate_sensitivity(df))
    step("C_range_sensitivity", lambda: range_sensitivity(df))
    step("E_goodness_of_fit", lambda: goodness_of_fit(df))
    step("F_same_instrument_search", lambda: same_instrument_search(a.n_boot))
    step("D_honest_ci", lambda: honest_ci(df))

    b = res["B_gap_closed"]["by_T"][16]
    print(f"HEADLINE  T=1->16, fraction of the achievable range closed")
    print(f"  intelligibility {100*b['wer']['point']:.1f}% "
          f"[{100*b['wer']['ci'][0]:.1f}, {100*b['wer']['ci'][1]:.1f}]")
    print(f"  identity        {100*b['sim']['point']:.1f}% "
          f"[{100*b['sim']['ci'][0]:.1f}, {100*b['sim']['ci'][1]:.1f}]")
    print(f"  ratio           {b['ratio_wer_over_sim']['point']:.2f}x "
          f"[{b['ratio_wer_over_sim']['ci'][0]:.2f}, "
          f"{b['ratio_wer_over_sim']['ci'][1]:.2f}]")
    print(f"  ratio across coordinates "
          f"{res['B_gap_closed']['ratio_range_across_coordinates']}")
    print(f"\nDelta-tau sign stable across coordinates: "
          f"{res['A_coordinate_sensitivity']['sign_is_stable']}")
    print(f"chi2/dof  wer {res['E_goodness_of_fit']['wer']['chi2_per_dof']:.1f}  "
          f"sim {res['E_goodness_of_fit']['sim']['chi2_per_dof']:.1f}")
    print(f"honest CI {res['D_honest_ci']['ci_weights_resampled']}")


if __name__ == "__main__":
    main()
