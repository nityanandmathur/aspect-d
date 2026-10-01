"""Two-way sync between this repo's paper/ folder and the private Overleaf repo.

paper/ is self-contained: main.tex reads only paper-local paths, and the five figures it
needs live in paper/figures/ as copies of the generated artifacts. The mirror is therefore
flat -- main.tex sits at the repo root, which is what Overleaf expects with no
configuration.

Two consequences worth knowing:

- paper/figures/*.pdf are COPIES. Regenerating a figure (src/s0_figure.py, src/figures.py)
  updates artifacts-*/figures/ and leaves the copy stale, so `--check-figures` compares
  them and `--refresh-figures` updates them. Staleness is detected, not assumed absent.
- paper/main-v1-frozen.tex is excluded. It is the frozen v1.0 submission and still refers
  to ../artifacts/figures/, including a *different* step_curves.pdf than the current paper
  uses. Flattening it would silently overwrite one figure with the other, so it stays put
  and out of the mirror.

Build products (aux, log, fls, fdb_latexmk, blg, pdf) are not synced -- Overleaf makes its
own and syncing them conflicts on every compile.

    python src/paper_sync.py --status
    python src/paper_sync.py --check-figures
    python src/paper_sync.py --push
    python src/paper_sync.py --pull
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE = "https://github.com/nityanandmathur/aspect-d-paper.git"
WORK = os.path.join(REPO, ".paper-sync")
PAPER = os.path.join(REPO, "paper")
KEEP = (".tex", ".bib", ".sty", ".bbl", ".md")
EXCLUDE = ("main-v1-frozen.tex",)
# paper/figures/<name> <- <artifact source>; the source stays canonical
FIG_SRC = {
    "aniso_contours_T16.pdf": "artifacts/figures/aniso_contours_T16.pdf",
    "substitution_plane.pdf": "artifacts/figures/substitution_plane.pdf",
    "extrapolation.pdf": "artifacts/figures/extrapolation.pdf",
    "step_curves.pdf": "artifacts-v1.2/figures/step_curves.pdf",
    "identity_ledger.pdf": "artifacts-v1.2/figures/identity_ledger.pdf",  # src/s0_figure.py
}


def git(*args, cwd=WORK, check=True):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and p.returncode:
        raise SystemExit(f"git {' '.join(args)} failed:\n{p.stderr[-600:]}")
    return p.stdout.strip()


def ensure_clone():
    if not os.path.isdir(os.path.join(WORK, ".git")):
        subprocess.run(["git", "clone", REMOTE, WORK], check=True,
                       capture_output=True, text=True)
    else:
        git("fetch", "origin")
        git("checkout", "main", check=False)
        git("reset", "--hard", "origin/main", check=False)


def manifest():
    """(absolute source, path relative to the mirror root)"""
    out = []
    for f in sorted(os.listdir(PAPER)):
        if os.path.splitext(f)[1] in KEEP and f not in EXCLUDE:
            out.append((os.path.join(PAPER, f), f))
    secs = os.path.join(PAPER, "sections")      # camera-ready body text, one file per section
    if os.path.isdir(secs):
        for f in sorted(os.listdir(secs)):
            if f.endswith(".tex"):
                out.append((os.path.join(secs, f), os.path.join("sections", f)))
    figs = os.path.join(PAPER, "figures")
    if os.path.isdir(figs):
        for f in sorted(os.listdir(figs)):
            if f.endswith(".pdf"):
                out.append((os.path.join(figs, f), os.path.join("figures", f)))
    return out


def check_figures(fix: bool = False) -> int:
    """paper/figures/ holds copies; report or repair drift from the canonical artifacts."""
    stale, missing = [], []
    for name, src_rel in FIG_SRC.items():
        src, dst = os.path.join(REPO, src_rel), os.path.join(PAPER, "figures", name)
        if not os.path.exists(src):
            continue
        if not os.path.exists(dst):
            missing.append(name)
        elif not filecmp.cmp(src, dst, shallow=False):
            stale.append((name, src_rel))
    if fix:
        for name in missing + [n for n, _ in stale]:
            shutil.copy2(os.path.join(REPO, FIG_SRC[name]),
                         os.path.join(PAPER, "figures", name))
            print(f"[figures] refreshed {name}")
        if not (missing or stale):
            print("[figures] already current")
        return 0
    if not (stale or missing):
        print(f"[figures] all {len(FIG_SRC)} copies match their generated source")
        return 0
    for name, src_rel in stale:
        print(f"[figures] STALE  paper/figures/{name} differs from {src_rel}")
    for name in missing:
        print(f"[figures] MISSING paper/figures/{name}")
    print("  run --refresh-figures, then rebuild the paper")
    return 1


def differences():
    here, there, diff = [], [], []
    known = {rel for _, rel in manifest()}
    for src, rel in manifest():
        dst = os.path.join(WORK, rel)
        if not os.path.exists(dst):
            here.append(rel)
        elif not filecmp.cmp(src, dst, shallow=False):
            diff.append(rel)
    for root, dirs, files in os.walk(WORK):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), WORK)
            if rel not in known and rel != "README.md":
                there.append(rel)
    return here, there, diff


README = """# ASPECT-D — paper sources

Mirror of the `paper/` folder of the main (private) code repo. `main.tex` is at the root
and reads only paper-local paths, so Overleaf needs no configuration: import this repo and
compile.

`figures/` holds copies of figures generated by the analysis code in the main repo. Edit
the text here freely; regenerate figures there.

Edits made here flow back with `python src/paper_sync.py --pull` in the main repo. Build
products are intentionally untracked.
"""


def do_push(message: str):
    ensure_clone()
    if check_figures() != 0:
        raise SystemExit("[sync] refusing to push with stale figure copies "
                         "-- run --refresh-figures first")
    keep = {rel for _, rel in manifest()} | {"README.md"}
    for root, dirs, files in os.walk(WORK):        # drop anything no longer in the manifest
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), WORK)
            if rel not in keep:
                os.unlink(os.path.join(WORK, rel))
    for src, rel in manifest():
        dst = os.path.join(WORK, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    open(os.path.join(WORK, "README.md"), "w").write(README)
    git("add", "-A")
    if not git("status", "--porcelain"):
        print("[sync] nothing to push -- the mirror already matches")
        return 0
    git("-c", "user.email=noreply@anthropic.com", "-c", "user.name=aspect-d-sync",
        "commit", "-m", message)
    git("push", "origin", "HEAD:main")
    print(f"[sync] pushed {len(manifest())} files to {REMOTE}")
    return 0


def do_pull():
    ensure_clone()
    _, _, diff = differences()
    if not diff:
        print("[sync] nothing to pull -- paper/ already matches the mirror")
        return 0
    for src, rel in manifest():
        if rel in diff:
            shutil.copy2(os.path.join(WORK, rel), src)
            print(f"[sync] updated paper/{rel}")
    print(f"\n[sync] {len(diff)} file(s) pulled in. These are your Overleaf edits, so "
          f"re-record the baseline:\n    python src/paper_freeze.py --record")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--check-figures", action="store_true")
    ap.add_argument("--refresh-figures", action="store_true")
    ap.add_argument("-m", "--message", default="sync paper sources")
    a = ap.parse_args()

    if a.refresh_figures:
        return check_figures(fix=True)
    if a.check_figures:
        return check_figures()
    if a.push:
        return do_push(a.message)
    if a.pull:
        return do_pull()
    ensure_clone()
    here, there, diff = differences()
    print(f"[sync] {REMOTE}")
    print(f"  tracked files: {len(manifest())}")
    for label, items in (("only here (will be added by --push)", here),
                         ("DIFFERENT on the two sides", diff),
                         ("only in the mirror (will be removed by --push)", there)):
        if items:
            print(f"  {label} ({len(items)}): {', '.join(items[:8])}")
    if not (here or diff or there):
        print("  in sync")
    check_figures()
    return 0


if __name__ == "__main__":
    sys.exit(main())
