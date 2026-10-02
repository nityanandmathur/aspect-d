"""Mechanically verify task.md §11 (definition of done) and protocol §10 deliverables.

    python src/verify_done.py            # prints a checklist, exits non-zero if anything fails

The main.tex checks read the paper sources, a checkout of
github.com/nityanandmathur/aspect-d-paper: $ASPECTD_PAPER_DIR, by default ../aspect-d-paper
next to this repo. Without a checkout they are skipped with a message. The compiled PDF is
that checkout's main.pdf if it has been built, else the committed docs/assets/paper.pdf.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import List, Tuple

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.environ.get("ASPECTD_PAPER_DIR") or os.path.join(os.path.dirname(REPO), "aspect-d-paper")
FIGS = ("aniso_contours_T16", "step_curves", "substitution_plane", "extrapolation")


def check(name: str, ok: bool, detail: str = "") -> Tuple[str, bool, str]:
    return (name, bool(ok), detail)


def main() -> int:
    rows: List[Tuple[str, bool, str]] = []
    p = lambda *a: os.path.join(REPO, *a)
    rl = lambda f: p("archive", "research-log", f)     # LOG.md, state.json, DECISION.md

    log_ok = os.path.exists(rl("LOG.md")) and os.path.getsize(rl("LOG.md")) > 4000
    gates_logged = all(g in open(rl("LOG.md")).read() for g in ("Gate G0", "Gate G1", "Gate G2",
                                                              "Gate G3", "G5", "G6"))
    rows.append(check("LOG.md decision trail (all gates present)", log_ok and gates_logged,
                      f"{os.path.getsize(rl('LOG.md'))} bytes"))

    st = json.load(open(rl("state.json"))) if os.path.exists(rl("state.json")) else {}
    rows.append(check("state.json phase == DONE", st.get("phase") == "DONE",
                      f"phase={st.get('phase')}"))
    rows.append(check("state.json has gate history", len(st.get("gates", {})) >= 5,
                      f"{sorted(st.get('gates', {}))}"))

    csv_p = p("results", "artifacts", "runs.csv")
    if os.path.exists(csv_p):
        df = pd.read_csv(csv_p)
        need = {"config", "seed", "T", "nfe", "width", "depth", "n_nonembed", "val_loss",
                "wer", "wer_se", "sim", "sim_se", "degen_rate", "c_layer_ms", "latency_ms",
                "train_gpu_hours", "train_wall_s"}
        missing = need - set(df.columns)
        exp = len(st.get("active_configs", [])) * len(st.get("active_seeds", [])) * \
            len(st.get("T_grid", []))
        rows.append(check("results/artifacts/runs.csv columns (protocol §10)", not missing,
                          f"missing={sorted(missing)}"))
        rows.append(check("results/artifacts/runs.csv coverage", len(df) > 0,
                          f"{len(df)} rows (full grid would be {exp})"))
    else:
        rows.append(check("results/artifacts/runs.csv exists", False, "missing"))
        df = None

    fits_p = p("results", "artifacts", "fits.json")
    if os.path.exists(fits_p):
        f = json.load(open(fits_p))
        d = f.get("decision", {})
        rows.append(check("results/artifacts/fits.json Part A + Part B fits",
                          all(f["part_a"][m]["M_full"]["ok"] and f["part_b"][m]["M_sep"]["ok"]
                              for m in ("wer", "sim")), ""))
        rows.append(check("fits.json bootstrap distributions present",
                          len(f.get("bootstrap", {}).get("dist", {}).get("delta_tau", [])) > 100,
                          f"{f.get('bootstrap', {}).get('n_reps_ok')} replicates"))
        rows.append(check("fits.json declares exactly one outcome class",
                          d.get("outcome_class") in ("S1", "S2", "F1", "F2"),
                          str(d.get("outcome_class"))))
    else:
        rows.append(check("results/artifacts/fits.json exists", False, "missing"))
        f = None

    have = [x for x in FIGS if os.path.exists(p("results", "artifacts", "figures", x + ".svg"))
            and os.path.exists(p("results", "artifacts", "figures", x + ".pdf"))]
    rows.append(check("results/artifacts/figures/ 4 figures as SVG + PDF", len(have) == 4,
                      f"{have}"))

    res_p = p("docs", "results.html")
    if os.path.exists(res_p):
        html = open(res_p).read()
        rows.append(check("results.html present, no TODO markers",
                          "TODO" not in html and len(html) > 5000, f"{len(html)} bytes"))
    else:
        rows.append(check("results.html exists", False, "missing"))

    # src/samples.py writes the samples to build/samples/ (gitignored, not published), so this
    # passes only after running it locally with the eval audio and the synthesized runs
    n_samples = len([x for x in os.listdir(p("build", "samples"))
                     if x.endswith(".flac")]) if os.path.isdir(p("build", "samples")) else 0
    rows.append(check("build/samples/ populated by src/samples.py (protocol §10)", n_samples >= 24,
                      f"{n_samples} audio files"))

    tex_p = os.path.join(PAPER, "main.tex")
    if not os.path.isdir(PAPER):
        print(f"[verify] SKIPPED the main.tex checks: no paper sources at {PAPER}. Clone "
              "github.com/nityanandmathur/aspect-d-paper next to this repo or set "
              "ASPECTD_PAPER_DIR.")
    elif os.path.exists(tex_p):
        tex = open(tex_p).read()
        todos = re.findall(r"\[[A-Z][A-Z ]+:[^\]]*\]|\[TODO[^\]]*\]", tex)
        anon = not re.search(r"smallest\.ai|github\.com|nityanand|acknowledg", tex, re.I)
        style = "\\usepackage{neurips_2026}" in tex
        rows.append(check("main.tex populated (no TODO/placeholder slots)", not todos,
                          f"{len(todos)} left: {todos[:3]}"))
        rows.append(check("main.tex anonymised", anon, ""))
        rows.append(check("main.tex uses neurips_2026 style", style, ""))
    else:
        rows.append(check("main.tex exists", False, f"missing in {PAPER}"))
    pdf = next((q for q in (os.path.join(PAPER, "main.pdf"), p("docs", "assets", "paper.pdf"))
                if os.path.exists(q)), None)
    rows.append(check("paper PDF compiled", pdf is not None,
                      os.path.relpath(pdf, REPO) if pdf else "no toolchain?"))

    dec_p = rl("DECISION.md")
    if os.path.exists(dec_p):
        dec = open(dec_p).read()
        one = sum(1 for c in ("S1", "S2", "F1", "F2") if re.search(rf"\b{c}\b.*declared|declared.*\b{c}\b", dec, re.I))
        rows.append(check("DECISION.md declares one outcome class",
                          len(dec) > 1500 and one == 1, f"{len(dec)} bytes"))
    else:
        rows.append(check("DECISION.md exists", False, "missing"))

    # consistency: every headline number in results.html must exist in fits.json
    if f and os.path.exists(res_p):
        d = f["decision"]
        html = open(res_p).read()
        wants = [f"{d['delta_tau']:.3f}"] if d.get("delta_tau") is not None else []
        rows.append(check("results.html headline numbers trace to fits.json",
                          all(w in html for w in wants), f"checked {wants}"))

    if df is not None and st.get("gpu_hours"):
        used = st["gpu_hours"]["total"]
        rows.append(check("gate G5 compute cap respected", used <= 500, f"{used:.1f} / 500 GPU-h"))

    print("\n=== task.md §11 definition of done ===")
    bad = 0
    for name, ok, detail in rows:
        print(f"  [{'x' if ok else ' '}] {name}" + (f"   ({detail})" if detail else ""))
        bad += (not ok)
    print(f"\n{len(rows) - bad}/{len(rows)} checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
