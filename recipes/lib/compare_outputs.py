"""Compare regenerated paper inputs against the committed camera-ready ones.

    python recipes/lib/compare_outputs.py --ref <committed paper dir> --new <regenerated paper dir>
        [--v15 <v15_partial.json>] [--figs <name>=<regenerated.pdf> ...] [--report out.md]

Only what main.tex actually uses is checked: the numbers*.tex, tab_*.tex and appendix_grid.tex
it \\inputs (recursively, resolved relative to main.tex) and the figures it \\includegraphics.
Files in the paper dir that main.tex does not read are ignored. Checks, each reported per item:
  macros   every \\newcommand in each \\input numbers*.tex (numbers.tex, numbers_v14.tex,
           numbers_v15.tex): same name set, same value string (exact, the paper prints these)
  tables   each \\input tab_*.tex and appendix_grid.tex: exact text after stripping trailing
           whitespace
  usage    every \\N<name> macro used in main.tex, or in a file it \\inputs (tab_*.tex,
           fig1_pipeline.tex, ...), is defined by some regenerated file
  figures  each figures/<name>.pdf main.tex includes must be passed with --figs; PDFs rendered at
           100 dpi (PyMuPDF) and compared pixel by pixel; PDF bytes are not compared because
           matplotlib embeds a creation date
Exit status: 0 all verified; 1 at least one MISMATCH; 2 no mismatch but some items could not
be regenerated from the released records (UNVERIFIABLE).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

# Discovered from main.tex, so a generated input the paper starts to \input later is checked
# too -- and fails loudly if nothing regenerates it.
MACRO_FILES: tuple = ()
TABLE_FILES: tuple = ()
FIGURES: dict = {}   # basename -> path as \includegraphics names it (relative to main.tex)
LATEX_N = {"NeedsTeXFormat", "NewDocumentCommand", "NewDocumentEnvironment",
           "NewExpandableDocumentCommand", "NewCommandCopy", "NewEnvironmentCopy"}


def used_files(tex: str, root: str | None = None, seen: frozenset = frozenset()):
    """(\\input files, \\includegraphics files) of a .tex file, recursively, as paths relative
    to the directory of the top-level file (`root`), as LaTeX resolves them."""
    root = os.path.dirname(tex) if root is None else root
    s = re.sub(r"(?<!\\)%.*", "", open(tex).read())
    ins, figs = [], [g for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s)]
    for f in re.findall(r"\\input\{([^}]+)\}", s):
        f = f if f.endswith(".tex") else f + ".tex"
        q = os.path.join(root, f)
        if f in ins or q in seen:
            continue
        ins.append(f)
        if os.path.exists(q):
            i2, g2 = used_files(q, root, seen | {q})
            ins += [x for x in i2 if x not in ins]
            figs += g2
    return ins, figs


def discover(tex: str) -> None:
    global MACRO_FILES, TABLE_FILES, FIGURES
    ins, figs = used_files(tex)
    base = [os.path.basename(f) for f in ins]
    MACRO_FILES = tuple(f for f in base if f.startswith("numbers"))
    TABLE_FILES = tuple(f for f in base if f.startswith("tab_") or f == "appendix_grid.tex")
    FIGURES = {os.path.basename(g if g.endswith(".pdf") else g + ".pdf"):
               (g if g.endswith(".pdf") else g + ".pdf") for g in figs}


def macros(path: str) -> dict:
    """\\newcommand{\\Name}{value} with balanced braces in value."""
    s = open(path).read()
    out = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{", s):
        i, depth = m.end(), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(s[i], 0)
            i += 1
        out[m.group(1)] = s[m.end():i - 1]
    return out


def table_lines(path: str) -> list:
    return [l.rstrip() for l in open(path).read().rstrip().splitlines()]


def expand_inputs(path: str, skip: tuple = (), root: str | None = None,
                  seen: frozenset = frozenset()) -> str:
    """Text of a .tex file with comments stripped and every \\input{f} replaced by the
    (recursively expanded) text of f; files whose basename is in `skip` expand to nothing.
    Like LaTeX, every \\input path is resolved relative to the directory of the top-level
    file (`root`), also inside sections/*.tex."""
    root = os.path.dirname(path) if root is None else root
    s = re.sub(r"(?<!\\)%.*", "", open(path).read())

    def sub(m):
        q = os.path.join(root, m.group(1))
        q = q if q.endswith(".tex") else q + ".tex"
        if not os.path.exists(q) or q in seen or os.path.basename(q) in skip:
            return ""
        return expand_inputs(q, skip, root, seen | {q})

    return re.sub(r"\\input\{([^}]+)\}", sub, s)


def render(pdf: str, dpi: int = 100):
    import pymupdf as fitz  # PyMuPDF
    import numpy as np
    doc = fitz.open(pdf)
    pix = doc[0].get_pixmap(dpi=dpi)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    return a[..., :3].astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--v15", default=None)
    ap.add_argument("--figs", nargs="*", default=[])
    ap.add_argument("--tex", default=None, help="main.tex to audit macro usage (default <ref>/main.tex)")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    tex = a.tex or os.path.join(a.ref, "main.tex")
    discover(tex)
    ok, bad, unv = [], [], []
    v15 = json.load(open(a.v15)) if a.v15 else None

    # ---------------------------------------------------------------- macros
    defined = {}
    for fn in MACRO_FILES:
        if not os.path.exists(os.path.join(a.ref, fn)):
            bad.append((fn, "<file>", "<missing in committed>", "\\input by main.tex"))
            continue
        ref = macros(os.path.join(a.ref, fn))
        newp = os.path.join(a.new, fn)
        new = macros(newp) if os.path.exists(newp) else {}
        if fn == "numbers_v15.tex" and v15 and not new:
            new = dict(v15["macros"])
        defined.update(new)
        skip = set(v15["unverifiable_macros"]) if (fn == "numbers_v15.tex" and v15) else set()
        for k, v in ref.items():
            if k in skip:
                unv.append((fn, k, v, "180k records not in the release (runs-v1.4/ is gitignored)"))
            elif k not in new:
                bad.append((fn, k, v, "<not regenerated>"))
            elif new[k] != v:
                bad.append((fn, k, v, new[k]))
            else:
                ok.append((fn, k))
        for k in sorted(set(new) - set(ref)):
            bad.append((fn, k, "<absent in committed>", new[k]))

    # ---------------------------------------------------------------- tables
    for fn in TABLE_FILES:
        if not os.path.exists(os.path.join(a.ref, fn)):
            bad.append((fn, "<file>", "<missing in committed>", "\\input by main.tex"))
            continue
        ref = table_lines(os.path.join(a.ref, fn))
        newp = os.path.join(a.new, fn)
        if fn == "tab_trend.tex" and v15 and not os.path.exists(newp):
            rows = {r.split("&")[0].strip(): r.rstrip() for r in v15["table_rows"]}
            for l in ref:
                key = l.split("&")[0].strip()
                if key in ("30k", "90k", "180k"):
                    if key in rows:
                        (ok.append((fn, key)) if rows[key] == l else bad.append((fn, key, l, rows[key])))
                    else:
                        unv.append((fn, key, l, "180k records not in the release"))
            continue
        if not os.path.exists(newp):
            bad.append((fn, "<file>", "present", "<not regenerated>"))
            continue
        new = table_lines(newp)
        if new == ref:
            ok.append((fn, "<whole table>"))
        else:
            for i in range(max(len(ref), len(new))):
                r = ref[i] if i < len(ref) else "<none>"
                n = new[i] if i < len(new) else "<none>"
                if r != n:
                    bad.append((fn, f"line {i+1}", r, n))

    # ---------------------------------------------------------------- usage audit
    if os.path.exists(tex):
        body = expand_inputs(tex, skip=MACRO_FILES)  # the definitions are checked above
        # every paper macro is \N<letters> (\Ndtau, \NcrGuideSim, ...); skip the few LaTeX
        # kernel commands that share the prefix
        used = {k for k in re.findall(r"\\(N[A-Za-z]+)", body) if k not in LATEX_N}
        allref = {}
        for fn in MACRO_FILES:
            if os.path.exists(os.path.join(a.ref, fn)):
                allref.update(macros(os.path.join(a.ref, fn)))
        for k in sorted(used):
            if k not in allref:
                bad.append(("main.tex", k, "<used>", "<defined in no committed numbers*.tex>"))
            elif k not in defined and not any(k == u[1] for u in unv):
                bad.append(("main.tex", k, "<used>", "<not regenerated>"))

    # ---------------------------------------------------------------- figures
    for spec in a.figs:
        name, newpdf = spec.split("=", 1)
        if name not in FIGURES:     # regenerated, but main.tex does not include it
            continue
        refpdf = os.path.join(a.ref, FIGURES[name])
        if not os.path.exists(refpdf):
            bad.append(("figures", name, "<missing in committed>", "included by main.tex"))
            continue
        if not os.path.exists(newpdf):
            bad.append(("figures", name, "present", "<not regenerated>"))
            continue
        try:
            import numpy as np
            x, y = render(refpdf), render(newpdf)
        except ImportError:
            unv.append(("figures", name, "", "PyMuPDF not installed; figure not compared"))
            continue
        if x.shape != y.shape:
            bad.append(("figures", name, f"{x.shape[1]}x{x.shape[0]} px", f"{y.shape[1]}x{y.shape[0]} px"))
            continue
        diff = np.abs(x - y).max(axis=2)
        frac = float((diff > 0).mean())
        if frac == 0.0:
            ok.append(("figures", name))
        else:
            bad.append(("figures", name, "identical render",
                        f"{100*frac:.3f}% of pixels differ (max channel diff {int(diff.max())})"))

    passed = {spec.split("=", 1)[0] for spec in a.figs}
    for name in FIGURES:
        if name not in passed:
            bad.append(("figures", name, "included by main.tex", "<no generator in the recipe>"))

    # ---------------------------------------------------------------- report
    lines = [f"# Regeneration check\n", f"- verified: {len(ok)}", f"- MISMATCH: {len(bad)}",
             f"- UNVERIFIABLE from released records: {len(unv)}\n"]
    if bad:
        lines += ["## Mismatches\n", "| file | item | committed | regenerated |", "|---|---|---|---|"]
        lines += [f"| {f} | {k} | `{r}` | `{n}` |" for f, k, r, n in bad]
    if unv:
        lines += ["\n## Unverifiable\n", "| file | item | committed value | reason |", "|---|---|---|---|"]
        lines += [f"| {f} | {k} | `{v}` | {why} |" for f, k, v, why in unv]
    txt = "\n".join(lines) + "\n"
    print(txt)
    if a.report:
        open(a.report, "w").write(txt)
    sys.exit(1 if bad else (2 if unv else 0))


if __name__ == "__main__":
    main()
