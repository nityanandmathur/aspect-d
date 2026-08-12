"""Two-way sync between this repo's paper/ folder and the private Overleaf repo.

Why the mirror is not flat. `paper/main.tex` pulls five figures from *outside* paper/
(`../artifacts/figures/…`, `../artifacts-v1.2/figures/…`). Putting main.tex at the root
of the sync repo would break those paths, and repairing them means editing main.tex.
So the sync repo reproduces the same directory shape instead:

    <sync repo>/paper/main.tex
    <sync repo>/artifacts/figures/*.pdf
    <sync repo>/artifacts-v1.2/figures/*.pdf

In Overleaf, set the main document to `paper/main.tex`; `../artifacts/…` then resolves
inside the project root and compiles unchanged.

Build products (aux, log, fls, fdb_latexmk, blg, pdf) are not synced -- Overleaf makes
its own, and syncing them produces conflicts on every compile.

    python src/paper_sync.py --status
    python src/paper_sync.py --push      # here -> Overleaf repo
    python src/paper_sync.py --pull      # Overleaf repo -> here
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
WORK = os.path.join(REPO, ".paper-sync")          # working clone, gitignored
KEEP = (".tex", ".bib", ".sty", ".bbl", ".md")
FIGS = ("artifacts/figures/aniso_contours_T16.pdf",
        "artifacts/figures/extrapolation.pdf",
        "artifacts/figures/substitution_plane.pdf",
        "artifacts-v1.2/figures/identity_ledger.pdf",
        "artifacts-v1.2/figures/step_curves.pdf")


def git(*args, cwd=WORK, check=True):
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and p.returncode:
        raise SystemExit(f"git {' '.join(args)} failed:\n{p.stderr[-600:]}")
    return p.stdout.strip()


def ensure_clone():
    if not os.path.isdir(os.path.join(WORK, ".git")):
        os.makedirs(os.path.dirname(WORK), exist_ok=True)
        subprocess.run(["git", "clone", REMOTE, WORK], check=True,
                       capture_output=True, text=True)
    else:
        git("fetch", "origin")
        if git("branch", "--list", "main"):
            git("checkout", "main", check=False)
            git("reset", "--hard", "origin/main", check=False)


def synced_files():
    """(source path in REPO, path inside the sync repo)"""
    out = []
    for f in sorted(os.listdir(os.path.join(REPO, "paper"))):
        if os.path.splitext(f)[1] in KEEP:
            out.append((os.path.join(REPO, "paper", f), os.path.join("paper", f)))
    for rel in FIGS:
        src = os.path.join(REPO, rel)
        if os.path.exists(src):
            out.append((src, rel))
    return out


def differences():
    """-> (only_here, only_there, differing)"""
    here, there, diff = [], [], []
    for src, rel in synced_files():
        dst = os.path.join(WORK, rel)
        if not os.path.exists(dst):
            here.append(rel)
        elif not filecmp.cmp(src, dst, shallow=False):
            diff.append(rel)
    known = {rel for _, rel in synced_files()}
    for root, dirs, files in os.walk(WORK):
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), WORK)
            if rel not in known and os.path.splitext(f)[1] in KEEP + (".pdf",):
                there.append(rel)
    return here, there, diff


def do_push(message: str):
    ensure_clone()
    for src, rel in synced_files():
        dst = os.path.join(WORK, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    readme = os.path.join(WORK, "README.md")
    if not os.path.exists(readme):
        open(readme, "w").write(
            "# ASPECT-D — paper sources\n\n"
            "Mirror of the `paper/` folder of the main (private) code repo, plus the five\n"
            "figures `main.tex` references from outside it.\n\n"
            "**Overleaf: set the main document to `paper/main.tex`.** The directory shape\n"
            "is deliberate — `main.tex` refers to `../artifacts/figures/…`, so it must sit\n"
            "one level down for those paths to resolve.\n\n"
            "Edits made here flow back with `python src/paper_sync.py --pull` in the main\n"
            "repo. Build products are intentionally not tracked.\n")
    git("add", "-A")
    if not git("status", "--porcelain"):
        print("[sync] nothing to push -- the Overleaf repo already matches")
        return 0
    git("-c", "user.email=noreply@anthropic.com", "-c", "user.name=aspect-d-sync",
        "commit", "-m", message)
    git("push", "origin", "HEAD:main")
    print(f"[sync] pushed to {REMOTE}")
    return 0


def do_pull():
    ensure_clone()
    _, _, diff = differences()
    if not diff:
        print("[sync] nothing to pull -- paper/ already matches the Overleaf repo")
        return 0
    for src, rel in synced_files():
        if rel in diff:
            shutil.copy2(os.path.join(WORK, rel), src)
            print(f"[sync] updated {rel}")
    print(f"\n[sync] {len(diff)} file(s) pulled in. These are YOUR edits from Overleaf, "
          f"so re-record the paper baseline:\n    python src/paper_freeze.py --record")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("-m", "--message", default="sync paper sources")
    a = ap.parse_args()

    if a.push:
        return do_push(a.message)
    if a.pull:
        return do_pull()
    ensure_clone()
    here, there, diff = differences()
    print(f"[sync] {REMOTE}")
    print(f"  tracked files: {len(synced_files())}")
    if here:
        print(f"  not yet in the Overleaf repo ({len(here)}): {', '.join(here[:6])}")
    if diff:
        print(f"  DIFFERENT on the two sides ({len(diff)}): {', '.join(diff)}")
        print("    --push sends this repo's version, --pull takes Overleaf's")
    if there:
        print(f"  present only in the Overleaf repo ({len(there)}): {', '.join(there[:6])}")
    if not (here or diff or there):
        print("  in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
