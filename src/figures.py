"""ASPECT-D figures — protocol.html §10, real data only, SVG + PDF.

    aniso_contours_T16   err contours in (width, depth) at T=16 from the M_full fit
    step_curves          normalised err vs T per metric (no raw-scale T* markers)
    substitution_plane   iso-WER contours in (log T, log d), slope −κ, iso-latency line
    extrapolation        H-D4: two-budget fit predicting the largest budget

House palette from docs/motivation.html / docs/protocol.html: depth #23479C, width #C4541D,
steps #0F7B72, ink #14181C, paper #F6F8F8, muted #5B6470.

    python src/figures.py --runs results/artifacts/runs.csv --fits results/artifacts/fits.json
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
    "font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 11,
    "axes.titlesize": 11.5, "axes.edgecolor": "#D9DEDE", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": "#ECF1F0", "grid.linewidth": 0.8,
    "legend.frameon": False, "figure.dpi": 130, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 9,
})


def panel(ax, letter: str):
    """(A)/(B) panel letters, as in the house style."""
    ax.text(0.015, 0.985, f"({letter})", transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=INK)


def _save(fig, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}.svg/.pdf", flush=True)


def fig_aniso(df: pd.DataFrame, fits: Dict, out_dir: str):
    ms = [m for m in ("wer", "sim") if fits["part_a"].get(m, {}).get("M_full", {}).get("ok")]
    fig, axes = plt.subplots(1, max(1, len(ms)), figsize=(4.6 * max(1, len(ms)), 3.6),
                             squeeze=False, constrained_layout=True)
    for ax, m, letter in zip(axes[0], ms, "AB"):
        panel(ax, letter)
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
            ax.annotate(r.config, (r.w, r.d), fontsize=8, color=INK, fontweight="bold",
                        xytext=(6, 5), textcoords="offset points")
        ax.set_xlabel("width $w$")
        ax.set_ylabel("depth $d$")
        # ρ = α/β is deliberately NOT shown: the width amplitude A saturates its
        # pre-registered bound, so α — and any ratio built from it — is unidentified.
        # Putting it in the title would advertise a number the paper declines to claim.
        ax.set_title(f"{LBL[m]} at $T{{=}}16$", color=C_MET[m], pad=6)
        cb = fig.colorbar(cs, ax=ax, shrink=0.9, pad=0.02)
        cb.set_label("fitted error", fontsize=9)
    _save(fig, out_dir, "aniso_contours_T16")


def fig_step_curves(df: pd.DataFrame, fits: Dict, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    Ts = sorted(df["T"].unique())
    for ax, m, letter in zip(axes, ("wer", "sim"), "AB"):
        panel(ax, letter)
        piv = df.pivot_table(index="config", columns="T", values=f"err_{m}", aggfunc="mean")
        for cfg, row in piv.iterrows():
            v = row.values.astype(float)
            if not np.isfinite(v).all() or v[-1] == 0:
                continue
            ax.plot(row.index, v / v[0], color=C_MET[m], alpha=0.28, linewidth=0.9)
        mean = piv.mean(axis=0).values.astype(float)
        ax.plot(piv.columns, mean / mean[0], color=C_MET[m], linewidth=2.6,
                label="grid mean")
        # RETRACTED (v1.1): the per-metric raw-scale T* marker was drawn on each metric's
        # OWN scale, where the 5% band is 0.67% of WER's range but 11.96% of SIM's -- a
        # 17.9x difference in strictness. Readers compared the two markers across panels
        # and read a saturation gap that does not exist: on the affine-invariant statistic
        # T* = 16 for BOTH metrics. No per-metric T* marker is drawn.
        # See results/artifacts-v1.1/e1_extended.json -> T_star_scale_caveat.
        tstar = None
        if tstar:
            ax.axvline(tstar, color=INK, linestyle=":", linewidth=1.3)
            ax.annotate(f"$T^*\\!=\\!{tstar}$", (tstar, 0.06), fontsize=10, color=INK,
                        xytext=(-4 if tstar == max(Ts) else 4, 0),
                        textcoords="offset points",
                        ha="right" if tstar == max(Ts) else "left")
        tau = fits["part_b"].get(m, {}).get("tau")
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ts)
        ax.set_xticklabels([str(t) for t in Ts])
        ax.set_xlabel("refinement steps per level $T$   (NFE $=8T$)")
        ax.set_ylabel("normalised to $T{=}1$")
        ax.set_title(LBL[m] + (f"  ($\\tau={tau:.3f}$)" if tau else ""), color=C_MET[m])
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "step_curves")


def fig_substitution(df: pd.DataFrame, fits: Dict, out_dir: str):
    sub = fits["part_b"]["wer"]["M_sub"]
    if not sub.get("ok"):
        return
    p = sub["params"]
    k = p["kappa"]
    w0 = float(np.median(df.width.unique()))
    dlo, dhi = float(df.depth.min()), float(df.depth.max())
    Ts = np.logspace(np.log2(1), np.log2(16), 120, base=2)
    ds = np.logspace(np.log2(dlo * 0.85), np.log2(dhi * 1.15), 120, base=2)
    T, D = np.meshgrid(Ts, ds)
    Z = p["E"] + p["A"] * w0 ** (-p["alpha"]) + p["B"] * (D * T ** k) ** (-p["beta"])
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=True)
    cs = ax.contour(T, D, Z, levels=7, colors=[C_STEPS], linewidths=1.1)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
    # iso-latency 8*T*d = const, drawn only where it stays inside the measured depth range
    c = 8 * 16 * float(np.median(df.depth.unique()))
    lat = c / (8 * Ts)
    ok = (lat >= ds.min()) & (lat <= ds.max())
    ax.plot(Ts[ok], lat[ok], color=C_DEPTH, linestyle="--", linewidth=2.0,
            label=r"iso-latency $8Td=$const")
    ax.scatter(df["T"], df.depth, s=10, color=MUTED, alpha=0.45, zorder=3,
               label="measured surface points")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_ylim(ds.min(), ds.max())
    ax.set_xticks(sorted(df["T"].unique()))
    ax.set_xticklabels([str(int(t)) for t in sorted(df["T"].unique())])
    ax.set_yticks([4, 8, 12, 18, 26, 36])
    ax.set_yticklabels(["4", "8", "12", "18", "26", "36"])
    ax.set_xlabel("refinement steps per level $T$")
    ax.set_ylabel("depth $d$")
    ax.set_title(f"iso-WER contours of $M_{{sub}}$ at $w={w0:.0f}$ "
                 f"($\\kappa={k:.2f}$ at bound, model rejected)", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    _save(fig, out_dir, "substitution_plane")


def fig_extrapolation(fits: Dict, out_dir: str):
    hd4 = fits.get("hd4", {})
    ms = [m for m in ("wer", "sim") if isinstance(hd4.get(m), dict) and "M_full" in hd4[m]]
    if not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(4.0 * len(ms), 3.4), squeeze=False,
                             constrained_layout=True)
    for ax, m, letter in zip(axes[0], ms, "AB"):
        panel(ax, letter)
        h = hd4[m]
        obs = np.array(h["M_full"]["obs"], float)
        lo, hi = min(obs.min(), 0), obs.max() * 1.12
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=0.9, linestyle=":")
        for name, col, mk in (("M_full", C_WIDTH, "o"), ("M_N", C_DEPTH, "s")):
            pred = np.array(h[name]["pred"], float)
            nice = {"M_full": "$M_{full}$", "M_N": "$M_{N}$"}[name]
            ax.scatter(obs, pred, color=col, marker=mk, s=44, edgecolor=INK, linewidth=0.6,
                       label=f"{nice}  MAPE {h[name]['mape']*100:.1f}%")
        ax.set_xlabel("observed (largest budget)")
        ax.set_ylabel("predicted from smaller budgets")
        ax.set_title(LBL[m], color=C_MET[m], pad=6)
        ax.legend(fontsize=9, loc="upper left")
    _save(fig, out_dir, "extrapolation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/artifacts/runs.csv")
    ap.add_argument("--fits", default="results/artifacts/fits.json")
    ap.add_argument("--out", default="results/artifacts/figures")
    a = ap.parse_args()
    df = pd.read_csv(a.runs)
    fits = json.load(open(a.fits))
    fig_aniso(df, fits, a.out)
    fig_step_curves(df, fits, a.out)
    fig_substitution(df, fits, a.out)
    fig_extrapolation(fits, a.out)


if __name__ == "__main__":
    main()
