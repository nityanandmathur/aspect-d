"""E5 — robustness panel over the metric stack (task-v1.md §4-E5).

Re-scores are already on disk as `scores_<tag>.json` beside each v1.0 synthesis:

    asrmed   WER from openai/whisper-medium.en   (SIM unchanged, wavlm-large)
    svbase   SIM from microsoft/wavlm-base-plus-sv (WER unchanged, whisper-large-v3)

Since each pass recomputes both metrics, the *unchanged* metric in each file is a
free consistency check against v1.0 and is asserted here.

This builds the {ASR × SV} panel and refits Δτ under each cell with the §7.2
machinery unchanged. Combined with `logamp.py`'s parameterisation axis, this is
the {ASR × SV × parameterisation} appendix table the runbook asks for.

**Sensitivity, NOT hypotheses** (task-v1.md §4-E5) — nothing here decides a
pre-registered claim; it reports how far Δτ moves when the measuring instruments
change.

    python src/e5_robust.py
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

import fit as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "artifacts-v1.1")
T_GRID = [1, 2, 4, 8, 16]
# (name, tag supplying WER, tag supplying SIM); None = the frozen v1.0 score
PANEL = [
    ("v1.0 baseline", None, None),
    ("ASR=whisper-medium.en", "asrmed", None),
    ("SV=wavlm-base-plus-sv", None, "svbase"),
    ("both swapped", "asrmed", "svbase"),
]


def read(run: str, T: int, tag: Optional[str]) -> Optional[Dict]:
    f = os.path.join(REPO, "results", "runs", run, f"synth_T{T}",
                     "scores.json" if tag is None else f"scores_{tag}.json")
    if not os.path.exists(f):
        return None
    return json.load(open(f))["summary"]


def build(v1: pd.DataFrame, wer_tag: Optional[str], sim_tag: Optional[str]) -> pd.DataFrame:
    """v1.0's runs.csv with the requested metric columns swapped for the variant."""
    rows, missing = [], 0
    for _, r in v1.iterrows():
        if int(r["T"]) not in T_GRID:
            continue
        run = f"{r.config}_{int(r.seed)}"
        sw = read(run, int(r["T"]), wer_tag)
        ss = read(run, int(r["T"]), sim_tag)
        if sw is None or ss is None:
            missing += 1
            continue
        d = r.to_dict()
        d["wer"], d["sim"] = sw["wer_mean"], ss["sim_mean"]
        d["err_wer"], d["err_sim"] = sw["wer_mean"], 1.0 - ss["sim_mean"]
        rows.append(d)
    df = pd.DataFrame(rows)
    df.attrs["missing"] = missing
    return df


def refit(df: pd.DataFrame, n_boot: int) -> Dict:
    """Δτ is the POINT estimate τ_wer − τ_sim, matching how v1.0 declares it
    (fit.py:404 differences the point fits); the bootstrap supplies the CI only.
    Reporting the bootstrap median instead would be a different estimator and would
    not reproduce the declared 0.1102 — it is kept alongside as a diagnostic."""
    out = {"n_rows": int(len(df))}
    for m in ("wer", "sim"):
        out[f"tau_{m}"] = F.part_b(df, m).get("tau")
    out["delta_tau"] = out["tau_wer"] - out["tau_sim"]
    for m in ("wer", "sim"):
        out[f"rho_{m}"] = F.part_a(df, m).get("rho")
    out["delta_rho"] = out["rho_sim"] - out["rho_wer"]

    boot = F.bootstrap(df, n_boot=n_boot)
    out["n_reps_ok"] = boot["n_reps_ok"]
    out["delta_tau_ci"] = F.ci(boot["delta_tau"])
    out["delta_rho_ci"] = F.ci(boot["delta_rho"])
    out["delta_tau_boot_median"] = float(np.median(boot["delta_tau"]))
    out["delta_rho_boot_median"] = float(np.median(boot["delta_rho"]))
    out["delta_tau_excludes_zero"] = bool(
        out["delta_tau_ci"] and (out["delta_tau_ci"][0] > 0 or out["delta_tau_ci"][1] < 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=F.N_BOOT)
    a = ap.parse_args()
    v1 = pd.read_csv(os.path.join(REPO, "results", "artifacts", "runs.csv"))
    v1fits = json.load(open(os.path.join(REPO, "results", "artifacts", "fits.json")))

    # consistency: the metric a variant does NOT change must reproduce v1.0 exactly
    checks = []
    for run, T in (("C3_0", 16), ("A1_0", 4)):
        base, med, sv = read(run, T, None), read(run, T, "asrmed"), read(run, T, "svbase")
        if base and med:
            checks.append({"run": run, "T": T, "check": "asrmed leaves SIM unchanged",
                           "v1_0": base["sim_mean"], "variant": med["sim_mean"],
                           "match": abs(base["sim_mean"] - med["sim_mean"]) < 1e-9})
        if base and sv:
            checks.append({"run": run, "T": T, "check": "svbase leaves WER unchanged",
                           "v1_0": base["wer_mean"], "variant": sv["wer_mean"],
                           "match": abs(base["wer_mean"] - sv["wer_mean"]) < 1e-9})

    res = {"note": "SENSITIVITY, NOT HYPOTHESES (task-v1.md §4-E5).",
           "consistency_checks": checks, "panel": {}}
    for name, wt, st in PANEL:
        df = build(v1, wt, st)
        if df.attrs["missing"]:
            print(f"[e5] {name}: {df.attrs['missing']} missing score files — skipped",
                  flush=True)
            res["panel"][name] = {"incomplete_missing": df.attrs["missing"]}
            continue
        print(f"[e5] refitting {name} ({len(df)} rows) ...", flush=True)
        res["panel"][name] = refit(df, a.n_boot)
        res["panel"][name]["wer_source"] = wt or "whisper-large-v3 (v1.0)"
        res["panel"][name]["sim_source"] = st or "wavlm-large SV (v1.0)"

    lg = os.path.join(OUT, "logamp_refit.json")
    if os.path.exists(lg):
        b = json.load(open(lg))["bootstrap"]
        lgp = json.load(open(lg))
        res["panel"]["log-amplitude parameterisation"] = {
            "delta_tau": (lgp["part_b"]["wer"]["tau"] - lgp["part_b"]["sim"]["tau"]),
            "delta_tau_ci": b["delta_tau"],
            "delta_tau_boot_median": b["delta_tau_point"],
            "delta_rho": (lgp["part_a"]["sim"]["rho"] - lgp["part_a"]["wer"]["rho"]),
            "delta_rho_ci": b["delta_rho"],
            "wer_source": "whisper-large-v3 (v1.0)", "sim_source": "wavlm-large SV (v1.0)",
            "delta_tau_excludes_zero": bool(b["delta_tau"][0] > 0 or b["delta_tau"][1] < 0)}

    res["v1_0_declared"] = {"delta_tau": v1fits["decision"]["delta_tau"],
                            "delta_tau_ci": v1fits["decision"]["delta_tau_ci"]}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "e5_robustness.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    print("\nconsistency checks:", flush=True)
    for c in checks:
        verdict = "OK" if c["match"] else f"MISMATCH {c['v1_0']:.6f} vs {c['variant']:.6f}"
        print(f"  {c['run']} T={c['T']}: {c['check']} -> {verdict}", flush=True)
    print(f"\n{'variant':<34} {'delta_tau':>10}  95% CI", flush=True)
    for k, v in res["panel"].items():
        if "delta_tau" not in v:
            continue
        ci = v["delta_tau_ci"]
        print(f"{k:<34} {v['delta_tau']:>10.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]"
              f"{'' if v['delta_tau_excludes_zero'] else '   (includes 0)'}", flush=True)


if __name__ == "__main__":
    main()
