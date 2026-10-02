"""Keep the paper's figure copies in step with the figures this repo generates.

The paper sources are a checkout of github.com/nityanandmathur/aspect-d-paper:
$ASPECTD_PAPER_DIR, by default ../aspect-d-paper next to this repo. Its main.tex reads only
paths inside that checkout, so the five figures it includes live in its figures/ folder as
COPIES of the generated files under results/, which stay canonical. Regenerating a figure
(src/s0_figure.py, src/figures.py) updates results/artifacts-*/figures/ and leaves the copy
stale, so `--check-figures` compares them and `--refresh-figures` updates them. Staleness is
detected, not assumed absent. After a refresh, rebuild the paper and commit the new copies in
the paper repo.

    python src/paper_sync.py --check-figures     # non-zero exit if a copy is stale or missing
    python src/paper_sync.py --refresh-figures   # copy the generated figures into the paper
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.environ.get("ASPECTD_PAPER_DIR") or os.path.join(os.path.dirname(REPO), "aspect-d-paper")
# <paper sources>/figures/<name> <- <artifact source>; the source stays canonical
FIG_SRC = {
    "aniso_contours_T16.pdf": "results/artifacts/figures/aniso_contours_T16.pdf",
    "substitution_plane.pdf": "results/artifacts/figures/substitution_plane.pdf",
    "extrapolation.pdf": "results/artifacts/figures/extrapolation.pdf",
    "step_curves.pdf": "results/artifacts-v1.2/figures/step_curves.pdf",
    "identity_ledger.pdf": "results/artifacts-v1.2/figures/identity_ledger.pdf",  # src/s0_figure.py
}


def check_figures(fix: bool = False) -> int:
    """<paper sources>/figures/ holds copies; report or repair drift from the canonical artifacts."""
    if not os.path.isdir(PAPER):
        raise SystemExit(f"[figures] no paper sources at {PAPER}. Clone "
                         "github.com/nityanandmathur/aspect-d-paper next to this repo or set "
                         "ASPECTD_PAPER_DIR.")
    figs = os.path.join(PAPER, "figures")
    stale, missing = [], []
    for name, src_rel in FIG_SRC.items():
        src, dst = os.path.join(REPO, src_rel), os.path.join(figs, name)
        if not os.path.exists(src):
            continue
        if not os.path.exists(dst):
            missing.append(name)
        elif not filecmp.cmp(src, dst, shallow=False):
            stale.append((name, src_rel))
    if fix:
        if missing:
            os.makedirs(figs, exist_ok=True)
        for name in missing + [n for n, _ in stale]:
            shutil.copy2(os.path.join(REPO, FIG_SRC[name]), os.path.join(figs, name))
            print(f"[figures] refreshed {name}")
        if not (missing or stale):
            print("[figures] already current")
        return 0
    if not (stale or missing):
        print(f"[figures] all {len(FIG_SRC)} copies in {figs} match their generated source")
        return 0
    for name, src_rel in stale:
        print(f"[figures] STALE  {os.path.join(figs, name)} differs from {src_rel}")
    for name in missing:
        print(f"[figures] MISSING {os.path.join(figs, name)}")
    print("  run --refresh-figures, then rebuild the paper")
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check-figures", action="store_true", help="the default")
    g.add_argument("--refresh-figures", action="store_true")
    a = ap.parse_args()
    return check_figures(fix=a.refresh_figures)


if __name__ == "__main__":
    sys.exit(main())
