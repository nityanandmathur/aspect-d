"""ASPECT-D fits and pre-registered tests — protocol.html §7 / PREREGISTRATION.md.

Part A (T = 16):   M_full: err = E + A·w^−α + B·d^−β      vs  M_N: err = E + C·N^−γ
Part B (75 pts):   M_sep : err = E + A·w^−α + B·d^−β + C·T^−τ
                   M_sub : err = E + A·w^−α + B·(d·T^κ)^−β

Weighted NLS, 32 multi-starts (RNG 42), bounds E∈[0,1], A,B,C∈[0,10], exponents
∈(0,3]; κ∈[−3,3] (see LOG.md P5-1: protocol gives no κ bound, a symmetric bound
keeps the H-D3 test unbiased). Run-level bootstrap, 2,000 replicates, RNG 7331.

    python src/fit.py --runs artifacts/runs.csv --out artifacts/fits.json
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

BOOT_RNG = 7331
NLS_RNG = 42
N_STARTS = 32
N_BOOT = 2000
SE_FLOOR = 1e-6            # absolute guard only
SE_FLOOR_FRAC = 0.25       # floor = 0.25 x pooled seed SD (scale-free; LOG.md P5-2)
EXP_LO, EXP_HI = 1e-4, 3.0
METRICS = ("wer", "sim")
T_REF = 16


# ------------------------------------------------------------------ model forms
def m_full(p, X):                       # E + A w^-a + B d^-b
    E, A, a, B, b = p
    return E + A * X["w"] ** (-a) + B * X["d"] ** (-b)


def m_n(p, X):                          # E + C N^-g
    E, C, g = p
    return E + C * X["N"] ** (-g)


def m_sep(p, X):                        # E + A w^-a + B d^-b + C T^-t
    E, A, a, B, b, C, t = p
    return E + A * X["w"] ** (-a) + B * X["d"] ** (-b) + C * X["T"] ** (-t)


def m_sub(p, X):                        # E + A w^-a + B (d T^k)^-b
    E, A, a, B, b, k = p
    return E + A * X["w"] ** (-a) + B * (X["d"] * X["T"] ** k) ** (-b)


FORMS = {
    "M_full": (m_full, ["E", "A", "alpha", "B", "beta"],
               ([0, 0, EXP_LO, 0, EXP_LO], [1, 10, EXP_HI, 10, EXP_HI])),
    "M_N": (m_n, ["E", "C", "gamma"], ([0, 0, EXP_LO], [1, 10, EXP_HI])),
    "M_sep": (m_sep, ["E", "A", "alpha", "B", "beta", "C", "tau"],
              ([0, 0, EXP_LO, 0, EXP_LO, 0, EXP_LO], [1, 10, EXP_HI, 10, EXP_HI, 10, EXP_HI])),
    "M_sub": (m_sub, ["E", "A", "alpha", "B", "beta", "kappa"],
              ([0, 0, EXP_LO, 0, EXP_LO, -3.0], [1, 10, EXP_HI, 10, EXP_HI, 3.0])),
}


def fit_form(name: str, X: Dict[str, np.ndarray], y: np.ndarray, se: np.ndarray,
             n_starts: int = N_STARTS, seed: int = NLS_RNG) -> Dict:
    fn, pnames, (lo, hi) = FORMS[name]
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    wts = 1.0 / np.maximum(se, SE_FLOOR)

    def resid(p):
        return (fn(p, X) - y) * wts

    rng = np.random.default_rng(seed)
    best = None
    for s in range(n_starts):
        p0 = lo + rng.random(len(lo)) * (hi - lo)
        if s == 0:                       # one deterministic, sane start
            p0 = np.where(np.isfinite(hi), np.minimum(hi, np.maximum(lo, 0.5 * (lo + hi))), 1.0)
        try:
            r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=4000)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return {"ok": False}
    rss = float(2 * best.cost)           # cost = 0.5 * sum(resid^2)
    n, k = len(y), len(lo) + 1           # +1 for the variance parameter
    aicc = n * math.log(max(rss, 1e-300) / n) + 2 * k
    if n - k - 1 > 0:
        aicc += 2 * k * (k + 1) / (n - k - 1)
    else:
        aicc = float("nan")
    p = dict(zip(pnames, [float(v) for v in best.x]))
    at_bound = {kk: bool(abs(best.x[i] - lo[i]) < 1e-6 or abs(best.x[i] - hi[i]) < 1e-6)
                for i, kk in enumerate(pnames)}
    return {"ok": True, "params": p, "rss_weighted": rss, "aicc": aicc, "n": n, "k": k,
            "at_bound": at_bound,
            "pred": fn(best.x, X).tolist(), "obs": y.tolist()}


# ----------------------------------------------------------------- surface prep
def surface(df: pd.DataFrame, metric: str, se_map: Optional[Dict] = None) -> pd.DataFrame:
    """Per (config, T) means over seeds with SE over seeds (pooled where 1 seed).

    ``se_map`` pins the weights to the OBSERVED-data SEs: inside the run-level bootstrap a
    replicate that draws the same seed twice has zero within-config spread, and recomputing
    1/SE² from it would hand that replicate an unbounded weight (protocol §7.2 resamples the
    runs, not the weighting scheme)."""
    col = f"err_{metric}"
    g = df.groupby(["config", "T"])
    rows = []
    for (cfg, T), sub in g:
        vals = sub[col].astype(float).values
        rows.append({"config": cfg, "T": T, "mean": float(np.nanmean(vals)),
                     "n_seeds": int(np.isfinite(vals).sum()),
                     "sd": float(np.nanstd(vals, ddof=1)) if np.isfinite(vals).sum() > 1
                     else np.nan,
                     "w": float(sub.width.iloc[0]), "d": float(sub.depth.iloc[0]),
                     "N": float(sub.n_nonembed.iloc[0]), "budget": sub.budget.iloc[0]})
    s = pd.DataFrame(rows)
    with_sd = s[s.n_seeds > 1]
    pooled_sd = float(np.sqrt(np.nanmean(with_sd.sd.values ** 2))) if len(with_sd) else 0.05
    s["se"] = np.where(s.n_seeds > 1, s.sd / np.sqrt(s.n_seeds), pooled_sd / np.sqrt(s.n_seeds))
    s["se"] = s.se.fillna(pooled_sd)
    # scale-free floor: a chance agreement between two seeds must not dominate the fit,
    # but the floor must not silently replace 1/SE^2 weighting with a constant either
    floor = SE_FLOOR_FRAC * pooled_sd
    s["se_floored"] = s.se < floor
    s["se"] = np.maximum(s.se, floor)
    if se_map is not None:
        s["se"] = [se_map.get((c, int(t)), e) for c, t, e in zip(s.config, s["T"], s.se)]
    s.attrs["pooled_sd"] = pooled_sd
    s.attrs["n_se_floored"] = int(s.se_floored.sum())
    return s


def xy(s: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    X = {"w": s.w.values.astype(float), "d": s.d.values.astype(float),
         "T": s["T"].values.astype(float), "N": s.N.values.astype(float)}
    return X, s["mean"].values.astype(float), s.se.values.astype(float)


# --------------------------------------------------------------------- part A/B
def part_a(df: pd.DataFrame, metric: str, n_starts: int = N_STARTS,
           se_map: Optional[Dict] = None) -> Dict:
    s = surface(df, metric, se_map)
    s = s[s["T"] == T_REF]
    X, y, se = xy(s)
    full, nn = fit_form("M_full", X, y, se, n_starts), fit_form("M_N", X, y, se, n_starts)
    out = {"metric": metric, "n_points": len(y), "M_full": full, "M_N": nn,
           "configs": s.config.tolist(),
           "weights": {"pooled_seed_sd": s.attrs.get("pooled_sd"),
                       "n_points_at_se_floor": int(s.se_floored.sum()),
                       "se_min": float(s.se.min()), "se_max": float(s.se.max())}}
    if full["ok"] and nn["ok"]:
        out["delta_aicc_full_minus_N"] = full["aicc"] - nn["aicc"]
        out["rho"] = full["params"]["alpha"] / full["params"]["beta"]
        out["shape_matters"] = bool(out["delta_aicc_full_minus_N"] <= -4)
    return out


def part_b(df: pd.DataFrame, metric: str, n_starts: int = N_STARTS,
           se_map: Optional[Dict] = None) -> Dict:
    s = surface(df, metric, se_map)
    X, y, se = xy(s)
    sep, sub = fit_form("M_sep", X, y, se, n_starts), fit_form("M_sub", X, y, se, n_starts)
    out = {"metric": metric, "n_points": len(y), "M_sep": sep, "M_sub": sub,
           "weights": {"pooled_seed_sd": s.attrs.get("pooled_sd"),
                       "n_points_at_se_floor": int(s.se_floored.sum()),
                       "se_min": float(s.se.min()), "se_max": float(s.se.max())}}
    if sep["ok"]:
        out["tau"] = sep["params"]["tau"]
    if sub["ok"]:
        out["kappa"] = sub["params"]["kappa"]
        out["kappa_at_bound"] = sub["at_bound"]["kappa"]
    if sep["ok"] and sub["ok"]:
        out["delta_aicc_sub_minus_sep"] = sub["aicc"] - sep["aicc"]
    return out


# ------------------------------------------------------------------- bootstrap
def _boot_one(args) -> Optional[Dict]:
    rep, df_pkl, n_starts, se_maps = args
    df = df_pkl
    rng = np.random.default_rng([BOOT_RNG, rep])
    parts = []
    for cfg, sub in df.groupby("config"):
        seeds = sorted(sub.seed.unique())
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        for k, sd in enumerate(pick):
            take = sub[sub.seed == sd].copy()
            take["seed"] = 1000 + k          # distinct pseudo-seed per draw
            parts.append(take)
    bdf = pd.concat(parts, ignore_index=True)
    res = {}
    try:
        for m in METRICS:
            a = part_a(bdf, m, n_starts, se_maps[m])
            b = part_b(bdf, m, n_starts, se_maps[m])
            res[m] = {"rho": a.get("rho"), "tau": b.get("tau"), "kappa": b.get("kappa"),
                      "alpha": a["M_full"]["params"]["alpha"] if a["M_full"]["ok"] else None,
                      "beta": a["M_full"]["params"]["beta"] if a["M_full"]["ok"] else None,
                      "d_aicc_sub_sep": b.get("delta_aicc_sub_minus_sep")}
    except Exception:
        return None
    return res


def bootstrap(df: pd.DataFrame, n_boot: int = N_BOOT, n_starts: int = N_STARTS,
              workers: int = 0) -> Dict:
    """Run-level bootstrap (protocol §7.2): resample seeds within configs, recompute
    every T point, refit. Weights are the point weights of the observed data."""
    workers = workers or min(64, mp.cpu_count() - 2)
    se_maps = {m: {(c, int(t)): e for c, t, e in
                   zip(*[surface(df, m)[k] for k in ("config", "T", "se")])} for m in METRICS}
    jobs = [(r, df, n_starts, se_maps) for r in range(n_boot)]
    with mp.Pool(workers) as pool:
        outs = [o for o in pool.imap_unordered(_boot_one, jobs, chunksize=4) if o]
    def col(metric, key):
        return np.array([o[metric][key] for o in outs
                         if o.get(metric, {}).get(key) is not None], float)
    dist = {"n_reps_ok": len(outs)}
    for m in METRICS:
        for key in ("rho", "tau", "kappa", "alpha", "beta"):
            dist[f"{m}_{key}"] = col(m, key).tolist()
    d_tau = np.array([o["wer"]["tau"] - o["sim"]["tau"] for o in outs
                      if o["wer"].get("tau") is not None and o["sim"].get("tau") is not None])
    d_rho = np.array([o["sim"]["rho"] - o["wer"]["rho"] for o in outs
                      if o["wer"].get("rho") is not None and o["sim"].get("rho") is not None])
    d_kap = np.array([o["wer"]["kappa"] - o["sim"]["kappa"] for o in outs
                      if o["wer"].get("kappa") is not None and o["sim"].get("kappa") is not None])
    dist["delta_tau"] = d_tau.tolist()
    dist["delta_rho"] = d_rho.tolist()
    dist["kappa_wer_minus_sim"] = d_kap.tolist()
    return dist


def ci(v: List[float], lo=2.5, hi=97.5) -> Optional[List[float]]:
    a = np.asarray(v, float)
    a = a[np.isfinite(a)]
    if a.size < 10:
        return None
    return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]


# --------------------------------------------------------------------- gate G4
def gate_g4(df: pd.DataFrame) -> Dict:
    """≥1 of {WER, SIM} at T=16: within ≥2 of 3 active budgets,
    (max−min config mean across shapes) ≥ 2 × pooled seed SD."""
    out = {"per_metric": {}}
    ok_any = False
    for m in METRICS:
        s16 = surface(df, m)
        s16 = s16[s16["T"] == T_REF]
        per_budget = {}
        n_ok = 0
        for bud, sub in s16.groupby("budget"):
            spread = float(sub["mean"].max() - sub["mean"].min())
            sds = sub[sub.n_seeds > 1].sd.values
            pooled = float(np.sqrt(np.nanmean(sds ** 2))) if len(sds) else float("nan")
            passes = bool(np.isfinite(pooled) and spread >= 2 * pooled)
            per_budget[bud] = {"spread": spread, "pooled_seed_sd": pooled, "passes": passes,
                               "n_configs": int(len(sub))}
            n_ok += int(passes)
        out["per_metric"][m] = {"budgets": per_budget, "n_budgets_passing": n_ok,
                                "passes": bool(n_ok >= 2)}
        ok_any = ok_any or n_ok >= 2
    out["passes"] = bool(ok_any)
    return out


# --------------------------------------------------------------------- H-D4, T*
def hd4(df: pd.DataFrame) -> Dict:
    """M_full fit on the two smaller budgets at T=16 predicts the largest budget:
    MAPE ≤ 15 % and ≤ M_N's MAPE."""
    res = {}
    budgets = sorted(df.budget.unique())
    if len(budgets) < 3:
        return {"skipped": f"only {len(budgets)} budgets active"}
    small, large = budgets[:-1], budgets[-1]
    for m in METRICS:
        s = surface(df, m)
        s = s[s["T"] == T_REF]
        tr, te = s[s.budget.isin(small)], s[s.budget == large]
        Xtr, ytr, setr = xy(tr)
        Xte, yte, _ = xy(te)
        r = {}
        for name in ("M_full", "M_N"):
            f = fit_form(name, Xtr, ytr, setr)
            if not f["ok"]:
                continue
            fn = FORMS[name][0]
            p = np.array([f["params"][k] for k in FORMS[name][1]])
            pred = fn(p, Xte)
            r[name] = {"mape": float(np.mean(np.abs(pred - yte) / np.abs(yte))),
                       "pred": pred.tolist(), "obs": yte.tolist(),
                       "params": f["params"]}
        if "M_full" in r and "M_N" in r:
            r["supported"] = bool(r["M_full"]["mape"] <= 0.15
                                  and r["M_full"]["mape"] <= r["M_N"]["mape"])
        r["train_budgets"], r["test_budget"] = small, large
        res[m] = r
    return res


def saturation_T(df: pd.DataFrame) -> Dict:
    """T*_m = smallest T reaching 95 % of the T=16 value, PER METRIC and on the metric's own
    scale: SIM-o (higher better) needs SIM(T) >= 0.95·SIM(16); WER (lower better) needs
    WER(T) <= WER(16)/0.95. Both read "within 5 % of the T=16 value". Descriptive."""
    lower_better = {"wer": True, "sim": False, "ut": False}

    def reached(v, ref, lower):
        return v <= ref / 0.95 if lower else v >= 0.95 * ref

    out = {}
    for m in METRICS:
        low = lower_better[m]
        piv = df.pivot_table(index="config", columns="T", values=m, aggfunc="mean")
        per_cfg = {}
        for cfg, row in piv.iterrows():
            if T_REF not in row or not np.isfinite(row[T_REF]):
                continue
            hits = [int(t) for t in sorted(row.index)
                    if np.isfinite(row[t]) and reached(row[t], row[T_REF], low)]
            per_cfg[cfg] = min(hits) if hits else None
        pooled = df.groupby("T")[m].mean()
        hits = [int(t) for t in sorted(pooled.index)
                if reached(pooled[t], pooled.get(T_REF, np.nan), low)]
        out[m] = {"per_config": per_cfg, "pooled_T_star": min(hits) if hits else None,
                  "pooled_curve": {int(k): float(v) for k, v in pooled.items()},
                  "scale": "metric (lower-is-better)" if low else "metric (higher-is-better)"}
    return out


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="artifacts/runs.csv")
    ap.add_argument("--out", default="artifacts/fits.json")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--boot-starts", type=int, default=N_STARTS)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--configs", default=None, help="comma list; default = state.json active")
    ap.add_argument("--state", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state.json"))
    a = ap.parse_args()

    df = pd.read_csv(a.runs)
    df = df[np.isfinite(df.err_wer) & np.isfinite(df.err_sim)]
    # only the ACTIVE grid may enter the fits: after a pivot or a "drop budget" calendar cut,
    # stale rows would otherwise re-enter Part A / Part B and H-D4's budget split
    active = None
    if a.configs:
        active = a.configs.split(",")
    elif os.path.exists(a.state):
        active = json.load(open(a.state)).get("active_configs")
    if active:
        before = len(df)
        df = df[df.config.isin(active)]
        print(f"[fit] active-config filter: {before} → {len(df)} rows "
              f"({len(active)} configs)", flush=True)
    res = {"input": a.runs, "n_rows": int(len(df)),
           "configs": sorted(df.config.unique().tolist()),
           "seeds": sorted(int(s) for s in df.seed.unique()),
           "T_grid": sorted(int(t) for t in df["T"].unique()),
           "budgets": sorted(df.budget.unique().tolist()),
           "part_a": {m: part_a(df, m) for m in METRICS},
           "part_b": {m: part_b(df, m) for m in METRICS},
           "gate_g4": gate_g4(df), "hd4": hd4(df), "saturation": saturation_T(df),
           "degen_by_T": {int(t): float(v) for t, v in
                          df.groupby("T").degen_rate.mean().items()}}
    if "err_ut" in df.columns and df.err_ut.notna().all() and df.err_ut.notna().any():
        res["part_a_utmos"] = part_a(df.assign(err_ut=df.err_ut), "ut")
    res["exploratory_unweighted"] = {}
    for m in METRICS:
        sA = surface(df, m)
        one = {(c, int(t)): 1.0 for c, t in zip(sA.config, sA["T"])}
        res["exploratory_unweighted"][m] = {
            "note": "sensitivity check only (LOG.md P5-3): all points weight 1, cannot alter "
                    "the declared outcome class",
            "part_a": part_a(df, m, se_map=one), "part_b": part_b(df, m, se_map=one)}
    print("[fit] point fits done; bootstrapping...", flush=True)
    dist = bootstrap(df, a.n_boot, a.boot_starts, a.workers)
    res["bootstrap"] = {"n_reps": a.n_boot, "n_reps_ok": dist["n_reps_ok"], "rng": BOOT_RNG,
                        "level": "run (seeds within configs)",
                        "ci": {k: ci(v) for k, v in dist.items() if isinstance(v, list)},
                        "dist": {k: v for k, v in dist.items() if isinstance(v, list)}}

    pa, pb = res["part_a"], res["part_b"]
    cis = res["bootstrap"]["ci"]
    d_tau = pb["wer"].get("tau", float("nan")) - pb["sim"].get("tau", float("nan"))
    d_rho = pa["sim"].get("rho", float("nan")) - pa["wer"].get("rho", float("nan"))
    k_wer, k_sim = pb["wer"].get("kappa"), pb["sim"].get("kappa")
    ci_tau, ci_rho, ci_kap = cis.get("delta_tau"), cis.get("delta_rho"), cis.get("wer_kappa")
    excl = lambda c: bool(c and (c[0] > 0 or c[1] < 0))
    hd2 = bool(excl(ci_tau) and d_tau > 0)
    hd1 = bool(excl(ci_rho) and d_rho > 0
               and (pa["wer"].get("shape_matters") or pa["sim"].get("shape_matters")))
    hd3 = bool(k_wer is not None and k_wer > 0 and excl(ci_kap) and ci_kap[0] > 0
               and pb["wer"].get("delta_aicc_sub_minus_sep", 1e9) <= 4
               and k_sim is not None and k_wer > k_sim)
    g4 = res["gate_g4"]["passes"]
    if not g4:
        outcome = "F2"                       # protocol §7.4: no H-D claims in either direction
    elif ci_tau is None or not np.isfinite(d_tau):
        outcome = "UNDETERMINED"             # analysis failure, NOT the F1 clean negative
    elif excl(ci_tau):
        outcome = "S1" if d_tau > 0 else "S2"
    else:
        outcome = "F1"
    res["decision"] = {
        "delta_tau": d_tau, "delta_tau_ci": ci_tau,
        "delta_rho": d_rho, "delta_rho_ci": ci_rho,
        "kappa_wer": k_wer, "kappa_wer_ci": ci_kap, "kappa_sim": k_sim,
        "kappa_wer_minus_sim_ci": cis.get("kappa_wer_minus_sim"),
        "tau_wer": pb["wer"].get("tau"), "tau_sim": pb["sim"].get("tau"),
        "tau_wer_ci": cis.get("wer_tau"), "tau_sim_ci": cis.get("sim_tau"),
        "rho_wer": pa["wer"].get("rho"), "rho_sim": pa["sim"].get("rho"),
        "H-D1": hd1 if g4 else None, "H-D2": hd2 if g4 else None, "H-D3": hd3 if g4 else None,
        "H-D4": ({m: res["hd4"].get(m, {}).get("supported") for m in METRICS} if g4 else None),
        "hypotheses_evaluated": bool(g4),
        "note": ("gate G4 failed → outcome F2: fits, figures and a power analysis only, "
                 "no H-D claims in either direction (protocol §8 G4)" if not g4 else
                 "gate G4 passed → hypotheses evaluated mechanically from the CIs"),
        "gate_g4_passes": g4, "outcome_class": outcome}
    if outcome == "UNDETERMINED":
        print("[fit] ERROR: bootstrap produced no Δτ CI or Δτ is not finite — the analysis "
              "failed; this is NOT the F1 clean negative. Investigate before declaring.",
              flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res["decision"], indent=1), flush=True)
    print(f"[fit] → {a.out}", flush=True)


if __name__ == "__main__":
    main()
