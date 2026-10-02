"""E5 (part 2) — log-amplitude refit of Part A and Part B (task-v1.md §4-E5).

The v1.0 defect: `M_full`'s width amplitude A hit its pre-registered upper bound
(A = 10.000, `at_bound.A = True`), so α — and therefore ρ = α/β and the H-D1
test — were not identified. This refit reparameterises the amplitudes as
log A, log B, log C (effectively unbounded) and leaves the exponent bounds
exactly as pre-registered, then asks whether α is identified once the amplitude
can move freely, and what ρ becomes.

**Exploratory-sensitivity. This cannot rescue H-D1's pre-registered status**
(task-v1.md §4-E5, §7) — H-D1 was decided under the pre-registered
parameterisation and stays decided there. Nothing here is a hypothesis test.

Implementation: the log forms are registered under the SAME names as v1.0's, so
`fit.part_a`, `fit.part_b` and the run-level bootstrap run byte-for-byte the
v1.0 code path (§5 requires the analysis machinery to be unchanged). Only the
parameterisation of the amplitudes differs.

    python src/logamp.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import fit as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "artifacts-v1.1")
LOG_LO, LOG_HI = -15.0, 15.0        # A ∈ [3e-7, 3e6]: unbounded for this problem's scale
EXP_LO, EXP_HI = F.EXP_LO, F.EXP_HI  # exponent bounds unchanged, as specified


def m_full_log(p, X):
    E, lA, a, lB, b = p
    return E + np.exp(lA) * X["w"] ** (-a) + np.exp(lB) * X["d"] ** (-b)


def m_n_log(p, X):
    E, lC, g = p
    return E + np.exp(lC) * X["N"] ** (-g)


def m_sep_log(p, X):
    E, lA, a, lB, b, lC, t = p
    return (E + np.exp(lA) * X["w"] ** (-a) + np.exp(lB) * X["d"] ** (-b)
            + np.exp(lC) * X["T"] ** (-t))


def m_sub_log(p, X):
    E, lA, a, lB, b, k = p
    return E + np.exp(lA) * X["w"] ** (-a) + np.exp(lB) * (X["d"] * X["T"] ** k) ** (-b)


LOG_FORMS = {
    "M_full": (m_full_log, ["E", "logA", "alpha", "logB", "beta"],
               ([0, LOG_LO, EXP_LO, LOG_LO, EXP_LO], [1, LOG_HI, EXP_HI, LOG_HI, EXP_HI])),
    "M_N": (m_n_log, ["E", "logC", "gamma"],
            ([0, LOG_LO, EXP_LO], [1, LOG_HI, EXP_HI])),
    "M_sep": (m_sep_log, ["E", "logA", "alpha", "logB", "beta", "logC", "tau"],
              ([0, LOG_LO, EXP_LO, LOG_LO, EXP_LO, LOG_LO, EXP_LO],
               [1, LOG_HI, EXP_HI, LOG_HI, EXP_HI, LOG_HI, EXP_HI])),
    "M_sub": (m_sub_log, ["E", "logA", "alpha", "logB", "beta", "kappa"],
              ([0, LOG_LO, EXP_LO, LOG_LO, EXP_LO, -3.0],
               [1, LOG_HI, EXP_HI, LOG_HI, EXP_HI, 3.0])),
}


def expify(params: dict) -> dict:
    """Report amplitudes on the natural scale alongside the fitted log scale."""
    out = dict(params)
    for k in ("logA", "logB", "logC"):
        if k in params:
            out[k[3:]] = float(np.exp(params[k]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(REPO, "results", "artifacts", "runs.csv"))
    ap.add_argument("--n-boot", type=int, default=F.N_BOOT)
    a = ap.parse_args()
    df = pd.read_csv(a.runs)
    v1 = json.load(open(os.path.join(REPO, "results", "artifacts", "fits.json")))

    # swap the parameterisation in place; every downstream fit.* call now uses it
    F.FORMS = LOG_FORMS

    res = {"note": "EXPLORATORY-SENSITIVITY (task-v1.md §4-E5). Log-amplitude "
                   "reparameterisation of the pre-registered forms; exponent bounds "
                   "unchanged. Does NOT alter H-D1's pre-registered status.",
           "amplitude_bounds_log": [LOG_LO, LOG_HI], "n_rows": int(len(df)),
           "part_a": {}, "part_b": {}}
    for m in ("wer", "sim"):
        pa = F.part_a(df, m)
        pb = F.part_b(df, m)
        res["part_a"][m] = {
            "M_full_params": expify(pa["M_full"]["params"]),
            "M_full_at_bound": pa["M_full"]["at_bound"],
            "M_N_params": expify(pa["M_N"]["params"]),
            "delta_aicc_full_minus_N": pa.get("delta_aicc_full_minus_N"),
            "rho": pa.get("rho"), "shape_matters": pa.get("shape_matters"),
            "v1_0_rho": v1["part_a"][m].get("rho"),
            "v1_0_A_at_bound": v1["part_a"][m]["M_full"]["at_bound"]["A"],
        }
        res["part_b"][m] = {
            "M_sep_params": expify(pb["M_sep"]["params"]),
            "M_sep_at_bound": pb["M_sep"]["at_bound"],
            "tau": pb.get("tau"), "kappa": pb.get("kappa"),
            "delta_aicc_sub_minus_sep": pb.get("delta_aicc_sub_minus_sep"),
            "v1_0_tau": v1["part_b"][m].get("tau"),
        }

    print("[logamp] point fits done; bootstrapping "
          f"{a.n_boot} run-level replicates ...", flush=True)
    boot = F.bootstrap(df, n_boot=a.n_boot)
    res["bootstrap"] = {
        "n_reps_ok": boot["n_reps_ok"],
        "delta_tau": F.ci(boot["delta_tau"]),
        "delta_rho": F.ci(boot["delta_rho"]),
        "wer_alpha": F.ci(boot["wer_alpha"]), "sim_alpha": F.ci(boot["sim_alpha"]),
        "wer_rho": F.ci(boot["wer_rho"]), "sim_rho": F.ci(boot["sim_rho"]),
        "wer_tau": F.ci(boot["wer_tau"]), "sim_tau": F.ci(boot["sim_tau"]),
        "delta_tau_point": float(np.median(boot["delta_tau"])),
        "delta_rho_point": float(np.median(boot["delta_rho"])),
    }
    dec = v1["decision"]
    res["comparison_to_v1_0"] = {
        "delta_tau_v1_0": dec["delta_tau"], "delta_tau_ci_v1_0": dec["delta_tau_ci"],
        "delta_rho_v1_0": dec["delta_rho"], "delta_rho_ci_v1_0": dec["delta_rho_ci"],
        "alpha_identified_now": {
            m: not res["part_a"][m]["M_full_at_bound"].get("logA", False)
            for m in ("wer", "sim")},
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "logamp_refit.json"), "w") as fh:
        json.dump(res, fh, indent=1)

    for m in ("wer", "sim"):
        p = res["part_a"][m]
        print(f"[logamp] {m}: A={p['M_full_params']['A']:.4g} "
              f"(at bound: {p['M_full_at_bound'].get('logA')})  "
              f"alpha={p['M_full_params']['alpha']:.4f}  beta={p['M_full_params']['beta']:.4f}  "
              f"rho={p['rho']:.4f}  (v1.0 rho={p['v1_0_rho']:.4f}, A was at bound: "
              f"{p['v1_0_A_at_bound']})", flush=True)
    b = res["bootstrap"]
    print(f"[logamp] delta_tau {b['delta_tau_point']:.4f} CI {b['delta_tau']}  "
          f"(v1.0 {dec['delta_tau']:.4f} CI {dec['delta_tau_ci']})", flush=True)
    print(f"[logamp] delta_rho {b['delta_rho_point']:.4f} CI {b['delta_rho']}  "
          f"(v1.0 {dec['delta_rho']:.4f} CI {dec['delta_rho_ci']})", flush=True)


if __name__ == "__main__":
    main()
