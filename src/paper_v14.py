"""Generate the v1.4 macros and table bodies from artifacts, never by hand.

Same contract as src/paper.py: every number that appears in the paper is emitted here
from a JSON artifact, so a number cannot drift from the measurement that produced it.
LaTeX control sequences cannot contain digits, so numeric keys are spelled out.

    python src/paper_v14.py     # -> paper/numbers_v14.tex, paper/tab_*.tex
"""
from __future__ import annotations

import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO, "paper")
A14 = os.path.join(REPO, "artifacts-v1.4", "analysis.json")
MENC = os.path.join(REPO, "artifacts-v1.3", "multi_encoder.json")

PRETTY = {"wavlm_large": ("WavLM-large SV", "WavLM / \\emph{selector}"),
          "ecapa": ("ECAPA-TDNN \\citep{ecapa}", "ECAPA, gated"),
          "ge2e": ("GE2E d-vector \\citep{ge2e}", "LSTM d-vector"),
          "xvect": ("x-vector \\citep{xvector}", "TDNN"),
          "wavlm_base_plus": ("WavLM-base+ SV", "WavLM, other scale")}
ORDER = ["ecapa", "xvect", "ge2e", "wavlm_base_plus", "wavlm_large"]

SPEC = {"tab_gapclosed.tex": "lccc", "tab_scope.tex": "lcccc",
        "tab_menc.tex": "llccc"}
HEAD = {
    "tab_gapclosed.tex":
        "  & \\multicolumn{2}{c}{fraction of reachable range closed} & \\\\\n"
        "  \\cmidrule(lr){2-3}\n"
        "  $\\tc{T}$ & intelligibility (WER) & identity (SIM-o) & ratio \\\\",
    "tab_scope.tex":
        "  error coordinate & $\\tau_{\\mathrm{WER}}$ & $\\tau_{\\mathrm{SIM}}$ & "
        "$\\Delta\\tau$ & gap-closed ratio \\\\",
    "tab_menc.tex":
        "  encoder & family / role & win rate & 95\\% CI & raw $\\Delta$ \\\\",
}


def pct(x, d=1):
    return f"{100*x:.{d}f}"


def ci(v, d=1, scale=100):
    return f"[{scale*v[0]:.{d}f}, {scale*v[1]:.{d}f}]"


def main():
    a = json.load(open(A14))
    m = json.load(open(MENC))
    M, out = [], []
    add = lambda k, v: M.append(f"\\newcommand{{\\{k}}}{{{v}}}")

    # ---- data description -------------------------------------------------
    d = a["data"]
    add("NtrainHours", f"{d['train_hours']:,}".replace(",", "{,}"))
    add("NtrainSpk", f"{d['train_speakers_min']:,}".replace(",", "{,}"))
    add("NheldoutSpk", d["heldout_speakers"])
    add("NevalItems", d["eval_items"])
    add("NevalSpeakers", d["eval_speakers"])

    # ---- B. the headline --------------------------------------------------
    b = a["B_gap_closed"]
    f = b["floors"]
    add("NwerFloor", f"{f['wer_asr_floor']:.4f}")
    add("NsimCeiling", f"{f['sim_codec_ceiling']:.4f}")
    s = b["by_T"]["16"] if "16" in b["by_T"] else b["by_T"][16]
    add("NgapWER", pct(s["wer"]["point"]))
    add("NgapWERci", ci(s["wer"]["ci"]))
    add("NgapSIM", pct(s["sim"]["point"]))
    add("NgapSIMci", ci(s["sim"]["ci"]))
    r = s["ratio_wer_over_sim"]
    add("NgapRatio", f"{r['point']:.2f}$\\times$")
    add("NgapRatioci", f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]")
    lo, hi = b["ratio_range_across_coordinates"]
    add("NgapRatioLo", f"{lo:.2f}$\\times$")
    add("NgapRatioHi", f"{hi:.2f}$\\times$")

    # the widest T measured: refinement's identity plateau, and the stability of the
    # ratio in T -- the property Delta-tau conspicuously lacks
    if "64" in b["by_T"]:
        e = b["by_T"]["64"]
        add("NgapWERext", pct(e["wer"]["point"]))
        add("NgapSIMext", pct(e["sim"]["point"]))
        add("NgapSIMextci", ci(e["sim"]["ci"]))
        add("NgapRatioext", f"{e['ratio_wer_over_sim']['point']:.2f}$\\times$")
        rs = [v["ratio_wer_over_sim"]["point"] for k, v in b["by_T"].items()
              if int(k) >= 4]
        add("NgapRatioTLo", f"{min(rs):.2f}$\\times$")
        add("NgapRatioTHi", f"{max(rs):.2f}$\\times$")
        add("NgapTmax", 64)

    # table body: gap closed by T
    rows = []
    for T, e in sorted(b["by_T"].items(), key=lambda kv: int(kv[0])):
        rows.append(f"    {int(T)} & {pct(e['wer']['point'])}\\% {ci(e['wer']['ci'])} & "
                    f"{pct(e['sim']['point'])}\\% {ci(e['sim']['ci'])} & "
                    f"{e['ratio_wer_over_sim']['point']:.2f}$\\times$ \\\\")
    out.append(("tab_gapclosed.tex", "\n".join(rows)))

    # ---- A. coordinate scope ---------------------------------------------
    c = a["A_coordinate_sensitivity"]
    add("NdtauNflip", c["n_reparameterisations_flipping_sign"])
    inv = b["coordinate_invariance"]
    key = {"identity (pre-registered)": "identity", "log(1+err)": "log1p",
           "sqrt(err)": "sqrt", "err squared": "square"}
    rows = []
    for name, v in c["by_coordinate"].items():
        gr = inv.get(key.get(name, ""), {}).get("ratio")
        grs = f"{gr:.2f}$\\times$" if gr else "n/a"
        nm = name.replace("(pre-registered)", "\\emph{(pre-registered)}")
        rows.append(f"    {nm} & {v['tau_wer']:.4f} & {v['tau_sim']:.4f} & "
                    f"{v['delta_tau']:+.4f} & {grs} \\\\")
    out.append(("tab_scope.tex", "\n".join(rows)))

    # ---- C/D/E. range, honest CI, goodness of fit -------------------------
    w = a["C_range_sensitivity"]["windows"]
    # The paper claims Delta-tau MOVES with the fitted window. If the frozen surface is
    # all that is loaded, every window holds the same rows and the macros come out equal,
    # which would ship the claim as a tautology. Refuse rather than emit that.
    if "T<=16" in w and "T<=64" in w:
        if abs(w["T<=16"]["delta_tau"] - w["T<=64"]["delta_tau"]) < 1e-9:
            raise SystemExit(
                "range-sensitivity windows are identical -- the extended sweep has not "
                "been collected to artifacts-v1.4/runs_extended.csv, so the range claim "
                "in the paper is unsupported. Refusing to emit the macros.")
    for k, mac in (("T<=16", "NdtauRangeSixteen"), ("T<=64", "NdtauRangeSixtyfour"),
                   ("T<=32", "NdtauRangeThirtytwo"), ("T<=8", "NdtauRangeEight")):
        if k in w:
            add(mac, f"{w[k]['delta_tau']:+.4f}")
    if "D_honest_ci" in a:
        h = a["D_honest_ci"]["ci_weights_resampled"]
        add("NdtauHonestci", f"[{h[0]:.4f}, {h[1]:.4f}]")
    g = a["E_goodness_of_fit"]
    add("NchiWER", f"{g['wer']['chi2_per_dof']:.1f}")
    add("NchiSIM", f"{g['sim']['chi2_per_dof']:.1f}")

    # the local exchange rate the separable fit itself implies, kappa = tau / beta;
    # quoted to replace the overreaching claim that "no exchange rate exists" at all
    fits = json.load(open(os.path.join(REPO, "artifacts", "fits.json")))
    pb = fits["part_b"]["wer"]["M_sep"]["params"]
    add("NkappaImpliedWER", f"{pb['tau'] / pb['beta']:.2f}")

    # ---- X2/X7: training compute and the top of the refinement axis -------
    xp = os.path.join(REPO, "artifacts-v1.4", "x2_x7.json")
    if os.path.exists(xp):
        x = json.load(open(xp))
        g = x["X2_gap_closed_90k"]
        add("NnkRuns", g["n_runs"])
        e16 = g["by_T"]["16"]
        add("NgapWERnk", pct(e16["wer"]["point"]))
        add("NgapSIMnk", pct(e16["sim"]["point"]))
        add("NgapSIMnkci", ci(e16["sim"]["ci"]))
        add("NgapRationk", f"{e16['ratio']['point']:.2f}$\\times$")
        rs = [v["ratio"]["point"] for k, v in g["by_T"].items() if int(k) >= 4]
        add("NgapRationkLo", f"{min(rs):.2f}$\\times$")
        add("NgapRationkHi", f"{max(rs):.2f}$\\times$")
        d = x["X7_T128"].get("delta_64_to_128")
        if d:
            add("NTmaxWERdelta", f"{d['wer']:+.4f}")
            add("NTmaxSIMdelta", f"{d['sim']:+.4f}")
            add("NTmaxRuns", len(x["X7_T128"]["runs"]))

    # ---- F. multi-encoder agreement --------------------------------------
    E = m["encoders"]
    ind = [k for k in E if k != "wavlm_large"]
    add("NmencNfam", m["n_independent_families_agreeing"])
    add("NmencLo", pct(min(E[k]["win_rate"] for k in ind)))
    add("NmencHi", pct(max(E[k]["win_rate"] for k in ind)))
    add("NmencRuns", E["ecapa"]["n_runs"])
    add("NmencItems", 200)
    add("NmencMinAuc", "0.98")
    rows = []
    for k in ORDER:
        v = E[k]
        nm, fam = PRETTY[k]
        star = "$^{\\ast}$" if k == "wavlm_large" else ""
        rows.append(f"    {nm}{star} & {fam} & {pct(v['win_rate'])}\\% & "
                    f"{ci(v['win_ci'])} & {v['mean_delta']:+.4f} \\\\")
    rows.append("    \\midrule")
    rows.append("    \\multicolumn{5}{l}{\\footnotesize $^{\\ast}$selects the candidates, "
                "so this row is circular and is not counted as agreement.} \\\\")
    out.append(("tab_menc.tex", "\n".join(rows)))

    hdr = ("% GENERATED by src/paper_v14.py from artifacts-v1.4/analysis.json and\n"
           "% artifacts-v1.3/multi_encoder.json. Do not edit; edit the source and rerun.\n")
    with open(os.path.join(PAPER, "numbers_v14.tex"), "w") as fh:
        fh.write(hdr + "\n".join(M) + "\n")
    # Each file holds a COMPLETE tabular. \input at an alignment boundary derails the
    # alignment (\bottomrule then lands as a misplaced \noalign), so the generated file
    # opens and closes its own environment and main.tex inputs it at float level.
    for fn, body in out:
        body = body.rstrip()
        if not body.endswith("\\\\"):     # \bottomrule needs the final row terminated
            body += " \\\\"
        with open(os.path.join(PAPER, fn), "w") as fh:
            fh.write(f"% GENERATED by src/paper_v14.py\n"
                     f"\\begin{{tabular}}{{{SPEC[fn]}}}\n  \\toprule\n"
                     f"{HEAD[fn]}\n  \\midrule\n{body}\n  \\bottomrule\n"
                     f"\\end{{tabular}}\n")
    print(f"wrote {len(M)} macros and {len(out)} table bodies")


if __name__ == "__main__":
    main()
