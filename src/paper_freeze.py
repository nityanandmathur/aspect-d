"""Tripwire: detect any change to the paper sources that was not explicitly agreed.

The user's instruction on 2026-08-12 is that the paper does not change without
discussing the change first. Intent is not a safeguard -- a generator run for a good
reason still rewrites numbers_v14.tex. This records a hash of every .tex and .bib file of
the paper sources and reports drift, so an unapproved edit is visible rather than assumed
absent.

The paper sources are a checkout of github.com/nityanandmathur/aspect-d-paper:
$ASPECTD_PAPER_DIR, by default ../aspect-d-paper next to this repo. If there is no checkout,
the check is skipped with a message.

    python src/paper_freeze.py --record     # after an agreed change
    python src/paper_freeze.py --check      # non-zero exit if the paper sources drifted
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.environ.get("ASPECTD_PAPER_DIR") or os.path.join(os.path.dirname(REPO), "aspect-d-paper")
STATE = os.path.join(REPO, "results", "artifacts-v1.5", "paper_freeze.json")
WATCH = ("*.tex", "*.bib")          # sources; the pdf follows from them
# The committed baseline was recorded when the sources were a folder of this repo, so its keys
# carry that folder's name as a prefix; it is dropped, and keys are paths in the paper sources.
OLD_PREFIX = "paper/"


def digest() -> dict:
    out = {}
    for pat in WATCH:
        for p in sorted(glob.glob(os.path.join(PAPER, pat))):
            with open(p, "rb") as fh:
                out[os.path.relpath(p, PAPER)] = hashlib.sha256(fh.read()).hexdigest()[:16]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not os.path.isdir(PAPER):
        print(f"[freeze] SKIPPED: no paper sources at {PAPER}. Clone "
              "github.com/nityanandmathur/aspect-d-paper next to this repo or set "
              "ASPECTD_PAPER_DIR.")
        return 0
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    now = digest()

    if a.record or not os.path.exists(STATE):
        json.dump(now, open(STATE, "w"), indent=1, sort_keys=True)
        print(f"[freeze] recorded {len(now)} paper source files as the agreed baseline")
        return 0

    # main-v1-frozen.tex was never part of aspect-d-paper (the frozen v1.0 text is
    # paper/main.tex at tag v1.0-submission-candidate), so it is not compared.
    was = {k[len(OLD_PREFIX):] if k.startswith(OLD_PREFIX) else k: v
           for k, v in json.load(open(STATE)).items()
           if k != OLD_PREFIX + "main-v1-frozen.tex"}
    changed = sorted(k for k in set(was) | set(now) if was.get(k) != now.get(k))
    if not changed:
        print(f"[freeze] paper sources unchanged ({len(now)} files match the baseline)")
        return 0
    print(f"[freeze] {len(changed)} PAPER FILE(S) CHANGED SINCE THE AGREED BASELINE:")
    for k in changed:
        kind = ("added" if k not in was else "deleted" if k not in now else "modified")
        print(f"    {kind:9s} {k}")
    print("  If this was agreed, re-record the baseline. If not, it needs discussing.")
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
