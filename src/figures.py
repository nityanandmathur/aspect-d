"""ASPECT-D figures — protocol.html §10, real data only, SVG + PDF.

    aniso_contours_T16   err contours in (width, depth) at T=16 from the M_full fit
    step_curves          normalised err vs T per metric, with T*
    substitution_plane   iso-WER contours in (log T, log d), slope −κ, iso-latency line
    extrapolation        H-D4: two-budget fit predicting the largest budget

House palette from index.html / protocol.html: depth #23479C, width #C4541D,
steps #0F7B72, ink #14181C, paper #F6F8F8, muted #5B6470.

    python src/figures.py --runs artifacts/runs.csv --fits artifacts/fits.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, PAPER, MUTED = "#14181C", "#FFFFFF", "#5B6470"
C_DEPTH, C_WIDTH, C_STEPS = "#23479C", "#C4541D", "#0F7B72"
C_MET = {"wer": C_STEPS, "sim": C_WIDTH, "ut": MUTED}
LBL = {"wer": "WER (intelligibility)", "sim": "1 − SIM-o (identity)", "ut": "UTMOS error"}

plt.rcParams.update({
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "axes.edgecolor": "#D9DEDE", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#ECF1F0", "grid.linewidth": 0.8,
    "legend.frameon": False, "figure.dpi": 130,
})


def _save(fig, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}.svg/.pdf", flush=True)


def fig_aniso(df: pd.DataFrame, fits: Dict, out_dir: str):
    ms = [m for m in ("wer", "sim") if fits["part_a"].get(m, {}).get("M_full", {}).get("ok")]
    fig, axes = plt.subplots(1, max(1, len(ms)), figsize=(3.4 * max(1, len(ms)), 3.0),
                             squeeze=False)
    for ax, m in zip(axes[0], ms):
        p = fits["part_a"][m]["M_full"]["params"]
        d16 = df[df["T"] == 16].groupby("config").agg(
            w=("width", "first"), d=("depth", "first"), e=(f"err_{m}", "mean")).reset_index()
        ws = np.linspace(d16.w.min() * 0.9, d16.w.max() * 1.05, 160)
        ds = np.linspace(d16.d.min() * 0.9, d16.d.max() * 1.05, 160)
        W, D = np.meshgrid(ws, ds)
        Z = p["E"] + p["A"] * W ** (-p["alpha"]) + p["B"] * D ** (-p["beta"])
        cs = ax.contourf(W, D, Z, levels=14, cmap="BuPu_r", alpha=0.9)
        ax.contour(W, D, Z, levels=8, colors=["#FFFFFF"], linewidths=0.6)
        sc = ax.scatter(d16.w, d16.d, c=d16.e, s=42, cmap="BuPu_r", edgecolor=INK, linewidth=0.7,
                        zorder=3, vmin=Z.min(), vmax=Z.max())
        for _, r in d16.iterrows():
            ax.annotate(r.config, (r.w, r.d), fontsize=6.5, color=INK,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("width $w$")
        ax.set_ylabel("depth $d$")
        ax.set_title(f"{LBL[m]} at $T{{=}}16$   "
                     r"$\rho=\alpha/\beta=$" + f"{fits['part_a'][m]['rho']:.2f}")
        fig.colorbar(cs, ax=ax, shrink=0.85, pad=0.02)
    fig.suptitle("Fitted anisotropy surface  err $= E + A\\,w^{-\\alpha} + B\\,d^{-\\beta}$  "
                 "(points: measured config means)", y=1.04, fontsize=9.5)
    _save(fig, out_dir, "aniso_contours_T16")


def fig_step_curves(df: pd.DataFrame, fits: Dict, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    Ts = sorted(df["T"].unique())
    for ax, m in zip(axes, ("wer", "sim")):
        piv = df.pivot_table(index="config", columns="T", values=f"err_{m}", aggfunc="mean")
        for cfg, row in piv.iterrows():
            v = row.values.astype(float)
            if not np.isfinite(v).all() or v[-1] == 0:
                continue
            ax.plot(row.index, v / v[0], color=C_MET[m], alpha=0.28, linewidth=0.9)
        mean = piv.mean(axis=0).values.astype(float)
        ax.plot(piv.columns, mean / mean[0], color=C_MET[m], linewidth=2.4,
                label="grid mean (normalised to $T{=}1$)")
        tstar = fits.get("saturation", {}).get(m, {}).get("pooled_T_star")
        if tstar:
            ax.axvline(tstar, color=INK, linestyle=":", linewidth=1.2)
            ax.annotate(f"$T^*={tstar}$", (tstar, 0.99), fontsize=8, color=INK,
                        xytext=(4, -2), textcoords="offset points")
        tau = fits["part_b"].get(m, {}).get("tau")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ts)
        ax.set_xticklabels([str(t) for t in Ts])
        ax.set_xlabel("refinement steps per level $T$   (NFE $=8T$)")
        ax.set_ylabel(f"{LBL[m]} / value at $T{{=}}1$")
        ax.set_title(LBL[m] + (f"   $\\tau={tau:.3f}$" if tau else ""))
        ax.legend(loc="best", fontsize=7.5)
    fig.suptitle("Test-time scaling per metric — every config, normalised", y=1.03, fontsize=9.5)
    _save(fig, out_dir, "step_curves")


def fig_substitution(df: pd.DataFrame, fits: Dict, out_dir: str):
    sub = fits["part_b"]["wer"]["M_sub"]
    if not sub.get("ok"):
        return
    p = sub["params"]
    k = p["kappa"]
    w0 = float(np.median(df.width.unique()))
    Ts = np.logspace(np.log2(1), np.log2(16), 120, base=2)
    ds = np.logspace(np.log2(4), np.log2(40), 120, base=2)
    T, D = np.meshgrid(Ts, ds)
    Z = p["E"] + p["A"] * w0 ** (-p["alpha"]) + p["B"] * (D * T ** k) ** (-p["beta"])
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    cs = ax.contour(T, D, Z, levels=9, colors=[C_STEPS], linewidths=1.1)
    ax.clabel(cs, inline=True, fontsize=6.5, fmt="%.3f")
    # iso-latency 8*T*d = const through the middle of the plane
    c = 8 * 4 * 16
    ax.plot(Ts, c / (8 * Ts), color=C_DEPTH, linestyle="--", linewidth=1.6,
            label=r"iso-latency $8Td=$const")
    if abs(k) > 1e-6:
        ax.plot(Ts, 16 * Ts ** (-k), color=INK, linewidth=1.4,
                label=fr"iso-WER slope $-\kappa={-k:.2f}$")
    pts = df[df["T"].isin(sorted(df["T"].unique()))]
    ax.scatter(pts["T"], pts.depth, s=8, color=MUTED, alpha=0.5, zorder=3,
               label="measured surface points")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("refinement steps per level $T$")
    ax.set_ylabel("depth $d$")
    ax.set_title(fr"Depth–step substitution for WER ($w={w0:.0f}$): "
                 fr"$B\,(d\,T^{{\kappa}})^{{-\beta}}$, $\kappa={k:.2f}$")
    ax.legend(fontsize=7)
    _save(fig, out_dir, "substitution_plane")


def fig_extrapolation(fits: Dict, out_dir: str):
    hd4 = fits.get("hd4", {})
    ms = [m for m in ("wer", "sim") if isinstance(hd4.get(m), dict) and "M_full" in hd4[m]]
    if not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(3.3 * len(ms), 3.0), squeeze=False)
    for ax, m in zip(axes[0], ms):
        h = hd4[m]
        obs = np.array(h["M_full"]["obs"], float)
        lo, hi = min(obs.min(), 0), obs.max() * 1.12
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=0.9, linestyle=":")
        for name, col, mk in (("M_full", C_WIDTH, "o"), ("M_N", C_DEPTH, "s")):
            pred = np.array(h[name]["pred"], float)
            ax.scatter(obs, pred, color=col, marker=mk, s=40, edgecolor=INK, linewidth=0.6,
                       label=f"{name.replace('_',chr(92)+'_')}  MAPE {h[name]['mape']*100:.1f}%")
        ax.set_xlabel(f"observed {LBL[m]} (largest budget, $T{{=}}16$)")
        ax.set_ylabel("predicted from the two smaller budgets")
        ax.set_title(f"H-D4 extrapolation — {LBL[m]}")
        ax.legend(fontsize=7)
    _save(fig, out_dir, "extrapolation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="artifacts/runs.csv")
    ap.add_argument("--fits", default="artifacts/fits.json")
    ap.add_argument("--out", default="artifacts/figures")
    a = ap.parse_args()
    df = pd.read_csv(a.runs)
    fits = json.load(open(a.fits))
    fig_aniso(df, fits, a.out)
    fig_step_curves(df, fits, a.out)
    fig_substitution(df, fits, a.out)
    fig_extrapolation(fits, a.out)


if __name__ == "__main__":
    main()
