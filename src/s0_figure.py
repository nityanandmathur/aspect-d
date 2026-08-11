"""S0 figure — the identity ledger with the codec ceiling drawn on it.

One bar per candidate axis (absolute SIM-o gain), the measured ceiling as a line,
and the gap decomposition: how much of the distance from real audio to the best
system is the codec and how much is the model.

    python src/s0_figure.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "artifacts-v1.2")
INK, MUTED = "#14181C", "#5B6470"
C_STEPS, C_WIDTH, C_DEPTH, C_STOP = "#0F7B72", "#C4541D", "#23479C", "#B3261E"

plt.rcParams.update({
    "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF", "savefig.facecolor": "#FFFFFF",
    "font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 11,
    "axes.titlesize": 11.5, "axes.edgecolor": "#D9DEDE", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.grid": True,
    "grid.color": "#ECF1F0", "grid.linewidth": 0.8, "legend.frameon": False,
    "legend.fontsize": 9, "figure.dpi": 130,
})

LABEL = {
    "steps_T1_to_T16": "refinement steps\n$T{=}1\\to16$",
    "allocation_at_matched_NFE": "NFE allocation\n(fine $\\to$ uniform)",
    "training_compute_30k_to_90k": "training compute\n30k $\\to$ 90k",
    "parameters_N": "parameters $N$\n20M $\\to$ 276M",
    "shape_at_fixed_N": "shape at fixed $N$\n(best $-$ worst)",
    "search_best_of_K": "test-time search\nbest-of-$K$",
}

# colour by which lever the axis is, not by where it happens to rank
AXCOL = {"steps_T1_to_T16": "steps", "allocation_at_matched_NFE": "steps",
         "search_best_of_K": "steps", "training_compute_30k_to_90k": "depth",
         "parameters_N": "depth", "shape_at_fixed_N": "width"}


def main():
    r = json.load(open(os.path.join(OUT, "identity_ledger.json")))
    m, led = r["MEASURED"], dict(r["ledger"])

    # The paper's second claim is that test-time *search* buys identity, so the panel
    # that inventories identity axes must contain it. Measured with WavLM-large, the
    # same encoder the headroom line is computed in, against the deployable T=16
    # default rather than against one-step decoding.
    menc = os.path.join(os.path.dirname(OUT), "artifacts-v1.3", "multi_encoder.json")
    if os.path.exists(menc):
        led["search_best_of_K"] = {
            "gain": json.load(open(menc))["encoders"]["wavlm_large"]["mean_delta"],
            "source": "artifacts-v1.3/multi_encoder.json (WavLM-large, vs T=64)"}

    axes_sorted = sorted(led.items(), key=lambda kv: -kv[1]["gain"])
    names = [LABEL.get(k, k) for k, _ in axes_sorted]
    gains = [v["gain"] for _, v in axes_sorted]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1.55, 1]},
        constrained_layout=True)

    # ---- (A) ledger bars against the headroom line ----
    ax.text(0.012, 0.985, "(A)", transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold", color=INK)
    _c = {"steps": C_STEPS, "depth": C_DEPTH, "width": C_WIDTH}
    cols = [_c[AXCOL.get(k, "width")] for k, _ in axes_sorted]
    bars = ax.bar(range(len(gains)), gains, color=cols, width=0.62, zorder=3)
    head = m["headroom_mean"]
    ax.axhline(head, color=C_STOP, ls="--", lw=1.8, zorder=4,
               label=f"remaining headroom to the codec ceiling ({head:+.3f})")
    for b, g in zip(bars, gains):
        ax.annotate(f"{g:+.3f}", (b.get_x() + b.get_width() / 2, g),
                    ha="center", va="bottom", fontsize=9.5, color=INK,
                    xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("absolute SIM-o gain")
    ax.set_title("What each axis buys in speaker identity", pad=6)
    ax.set_ylim(0, max(max(gains), head) * 1.28)
    ax.legend(loc="upper right")

    # ---- (B) where the identity gap actually goes ----
    ax2.text(0.02, 0.985, "(B)", transform=ax2.transAxes, ha="left", va="top",
             fontsize=12, fontweight="bold", color=INK)
    gt = m["sim_gt_no_roundtrip_mean"]
    rt = m["sim_roundtrip_mean"]
    sysbest = m["best_system"]["sim"]
    codec = gt - rt
    model = rt - sysbest
    ax2.barh([0], [codec], left=[sysbest + model], color=C_WIDTH, height=0.42, zorder=3,
             label=f"lost to the codec ({codec:.3f})")
    ax2.barh([0], [model], left=[sysbest], color=C_DEPTH, height=0.42, zorder=3,
             label=f"lost to the model ({model:.3f})")
    ax2.barh([0], [sysbest], color="#C9D3D2", height=0.42, zorder=3,
             label=f"best measured system ({sysbest:.3f})")
    for x, lab in ((sysbest, "best system"), (rt, "codec ceiling"), (gt, "real audio")):
        ax2.axvline(x, color=INK, lw=1.0, ls=":", zorder=4)
        ax2.annotate(f"{lab}\n{x:.3f}", (x, 0.30), ha="center", va="bottom",
                     fontsize=8.5, color=INK)
    ax2.set_yticks([])
    ax2.set_xlim(0, gt * 1.16)
    ax2.set_ylim(-0.45, 0.72)
    ax2.set_xlabel("SIM-o against the prompt")
    ax2.set_title(f"{100 * codec / (codec + model):.0f}% of the identity gap is the codec",
                  pad=6)
    ax2.legend(loc="lower right", fontsize=8.5)

    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"identity_ledger.{ext}"),
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[s0-fig] identity_ledger.svg/.pdf  |  codec {codec:.4f} ({100*codec/(codec+model):.0f}%) "
          f"vs model {model:.4f} ({100*model/(codec+model):.0f}%)", flush=True)


if __name__ == "__main__":
    main()
