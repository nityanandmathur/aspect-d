"""E2 — iso-latency Pareto analysis (task-v1.md §4-E2, hypothesis H-E2).

Zero GPU. Uses the frozen v1.0 75-point surface and the measured c_layer.
Serial latency L = 8·T·d·c_layer(w). For a log grid of budgets L we find the
WER- and SIM-optimal allocation two independent ways:

  fitted   — minimise the fitted M_sep prediction over (config, T)
  measured — minimise the measured config×T mean over (config, T)

H-E2 decision rule: the two methods agree at >= 80 % of tested L values.
Structural claim: whenever the optimum uses more than the minimum tested T,
its depth is already at least the budget's interior optimum d*.

    python src/iso_latency.py
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "artifacts-v1.1")
INK, MUTED = "#14181C", "#5B6470"
C_DEPTH, C_WIDTH, C_STEPS = "#23479C", "#C4541D", "#0F7B72"
C_MET = {"wer": C_STEPS, "sim": C_WIDTH}
LBL = {"wer": "WER (intelligibility)", "sim": "1 − SIM-o (identity)"}

plt.rcParams.update({
    "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF", "savefig.facecolor": "#FFFFFF",
    "font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 11,
    "axes.titlesize": 11.5, "axes.edgecolor": "#D9DEDE", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "axes.grid": True, "grid.color": "#ECF1F0", "grid.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 130,
})


def m_sep(p: Dict, w, d, T):
    return (p["E"] + p["A"] * w ** (-p["alpha"]) + p["B"] * d ** (-p["beta"])
            + p["C"] * T ** (-p["tau"]))


def build(df: pd.DataFrame, fits: Dict, clayer: Dict[int, float]) -> pd.DataFrame:
    """One row per (config, T): latency, measured error, fitted error, per metric."""
    g = df.groupby(["config", "budget", "width", "depth", "T"]).agg(
        err_wer=("err_wer", "mean"), err_sim=("err_sim", "mean")).reset_index()
    g["c_layer"] = g.width.map(clayer)
    g["L_ms"] = 8 * g["T"] * g.depth * g.c_layer
    for m in ("wer", "sim"):
        p = fits["part_b"][m]["M_sep"]["params"]
        g[f"fit_{m}"] = m_sep(p, g.width.values, g.depth.values, g["T"].values)
    return g


def d_star(df: pd.DataFrame) -> Dict[str, int]:
    """Interior optimum depth per budget at T=16 (the v1.1 descriptive result)."""
    t16 = df[df["T"] == 16].groupby(["budget", "depth"]).err_wer.mean().reset_index()
    return {b: int(s.loc[s.err_wer.idxmin(), "depth"])
            for b, s in t16.groupby("budget")}


def pareto(g: pd.DataFrame, metric: str, budgets: np.ndarray) -> pd.DataFrame:
    rows = []
    for L in budgets:
        sub = g[g.L_ms <= L]
        if sub.empty:
            continue
        mrow = sub.loc[sub[f"err_{metric}"].idxmin()]
        frow = sub.loc[sub[f"fit_{metric}"].idxmin()]
        rows.append({
            "L_ms": L,
            "meas_config": mrow.config, "meas_d": int(mrow.depth), "meas_w": int(mrow.width),
            "meas_T": int(mrow["T"]), "meas_err": float(mrow[f"err_{metric}"]),
            "meas_budget": mrow.budget,
            "fit_config": frow.config, "fit_d": int(frow.depth), "fit_w": int(frow.width),
            "fit_T": int(frow["T"]), "fit_err": float(frow[f"fit_{metric}"]),
            "fit_budget": frow.budget,
        })
    p = pd.DataFrame(rows)
    p["agree_exact"] = (p.meas_config == p.fit_config) & (p.meas_T == p.fit_T)
    p["agree_depth"] = p.meas_d == p.fit_d
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(REPO, "artifacts", "runs.csv"))
    ap.add_argument("--fits", default=os.path.join(REPO, "artifacts", "fits.json"))
    ap.add_argument("--n-budgets", type=int, default=40)
    a = ap.parse_args()
    df = pd.read_csv(a.runs)
    fits = json.load(open(a.fits))
    clayer = {int(k): v["c_layer_ms"] for k, v in
              json.load(open(os.path.join(REPO, "artifacts", "c_layer.json")))["measured"].items()}
    g = build(df, fits, clayer)
    ds = d_star(df)
    Tmin = int(g["T"].min())
    budgets = np.logspace(np.log10(g.L_ms.min()), np.log10(g.L_ms.max()), a.n_budgets)

    Tmax = int(g["T"].max())
    res: Dict = {"d_star_per_budget": ds, "T_min_tested": Tmin, "T_max_tested": Tmax,
                 "L_range_ms": [float(g.L_ms.min()), float(g.L_ms.max())],
                 "n_budgets": int(a.n_budgets), "metrics": {}}
    paretos = {}
    for m in ("wer", "sim"):
        p = pareto(g, m, budgets)
        paretos[m] = p
        # H-E2 as stated: depth reaches the budget's interior optimum d* BEFORE steps are
        # raised above the minimum tested. Evaluated per method: among budgets whose
        # optimum spends more than T_min, what fraction already sit at d >= d*?
        def depth_first(dcol: str, tcol: str, bcol: str) -> float:
            sub = p[p[tcol] > Tmin]
            return float((sub[dcol] >= sub[bcol].map(ds)).mean()) if len(sub) else float("nan")
        df_meas = depth_first("meas_d", "meas_T", "meas_budget")
        df_fit = depth_first("fit_d", "fit_T", "fit_budget")
        res["metrics"][m] = {
            "depth_first_frac_measured": df_meas,
            "depth_first_frac_fitted": df_fit,
            # secondary, descriptive: do the two methods pick the same allocation at all?
            "agree_exact_frac": float(p.agree_exact.mean()),
            "agree_depth_frac": float(p.agree_depth.mean()),
            # ordering diagnostic: the budget at which each resource is first exhausted
            "L_first_T_max": float(p[p.meas_T == Tmax].L_ms.min()) if (p.meas_T == Tmax).any()
                else float("nan"),
            "L_first_d_at_dstar": float(p[p.meas_d >= p.meas_budget.map(ds)].L_ms.min())
                if (p.meas_d >= p.meas_budget.map(ds)).any() else float("nan"),
            "n_L": int(len(p)),
            "path_measured": p[["L_ms", "meas_config", "meas_d", "meas_T", "meas_err"]]
                .to_dict("records"),
        }
    w = res["metrics"]["wer"]
    res["H_E2"] = {
        "rule": "the depth-first property (d >= d* before T > T_min) holds under BOTH the "
                "fitted and the measured method at >= 80% of tested L values "
                "(task-v1.md §9 H-E2)",
        "depth_first_wer_measured": w["depth_first_frac_measured"],
        "depth_first_wer_fitted": w["depth_first_frac_fitted"],
        "supported": bool(w["depth_first_frac_measured"] >= 0.80
                          and w["depth_first_frac_fitted"] >= 0.80),
        "secondary_method_concordance_wer": w["agree_exact_frac"],
        "caveat": f"T is capped at {Tmax} by the v1.0 grid, so 'steps exhausted' is partly a "
                  f"boundary of the tested range; E1 extends T to 64 and this analysis is "
                  f"re-run there.",
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "iso_latency.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for m, p in paretos.items():
        p.to_csv(os.path.join(OUT, f"iso_latency_path_{m}.csv"), index=False)

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for ax, m, letter in zip(axes, ("wer", "sim"), "AB"):
        p = paretos[m]
        ax.text(0.015, 0.985, f"({letter})", transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold", color=INK)
        ax.scatter(g["T"], g.depth, s=14, color=MUTED, alpha=0.30, zorder=1,
                   label="measured surface points")
        ax.plot(p.meas_T, p.meas_d, "-o", color=C_MET[m], lw=2.2, ms=6, zorder=3,
                label="optimal path (measured)")
        ax.plot(p.fit_T, p.fit_d, "--s", color=C_DEPTH, lw=1.6, ms=5, alpha=0.85, zorder=2,
                label="optimal path (fitted $M_{sep}$)")
        for b, dv in sorted(ds.items()):
            ax.axhline(dv, color=C_WIDTH, ls=":", lw=1.3, alpha=0.8)
            ax.annotate(f"$d^*_{{{b}}}$={dv}", (ax.get_xlim()[1], dv), color=C_WIDTH,
                        fontsize=8.5, va="bottom", ha="right",
                        xytext=(-2, 2), textcoords="offset points")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(sorted(g["T"].unique()))
        ax.set_xticklabels([str(int(t)) for t in sorted(g["T"].unique())])
        ax.set_yticks([4, 8, 12, 18, 26, 36])
        ax.set_yticklabels(["4", "8", "12", "18", "26", "36"])
        ax.set_xlabel("refinement steps per level $T$")
        ax.set_ylabel("depth $d$")
        ax.set_title(f"{LBL[m]}  —  depth-first {100*res['metrics'][m]['depth_first_frac_measured']:.0f}%",
                     color=C_MET[m], pad=6)
        ax.legend(loc="lower left", fontsize=8.5)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"iso_latency_pareto.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({k: v for k, v in res.items() if k != "metrics"}, indent=1), flush=True)
    for m in ("wer", "sim"):
        r = res["metrics"][m]
        print(f"[E2] {m}: depth-first measured {100*r['depth_first_frac_measured']:.0f}%  "
              f"fitted {100*r['depth_first_frac_fitted']:.0f}%  |  "
              f"method-concordance {100*r['agree_exact_frac']:.0f}%  |  "
              f"T hits max at L={r['L_first_T_max']:.0f}ms, "
              f"d hits d* at L={r['L_first_d_at_dstar']:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
