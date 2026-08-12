"""Tripwire: detect any change to paper/ that was not explicitly agreed.

The user's instruction on 2026-08-12 is that the paper does not change without
discussing the change first. Intent is not a safeguard -- a generator run for a good
reason still rewrites paper/numbers_v14.tex. This records a hash of every tracked file
under paper/ and reports drift, so an unapproved edit is visible rather than assumed
absent.

    python src/paper_freeze.py --record     # after an agreed change
    python src/paper_freeze.py --check      # non-zero exit if paper/ drifted
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(REPO, "paper")
STATE = os.path.join(REPO, "artifacts-v1.5", "paper_freeze.json")
WATCH = ("*.tex", "*.bib")          # sources; the pdf follows from them


def digest() -> dict:
    out = {}
    for pat in WATCH:
        for p in sorted(glob.glob(os.path.join(PAPER, pat))):
            with open(p, "rb") as fh:
                out[os.path.relpath(p, REPO)] = hashlib.sha256(fh.read()).hexdigest()[:16]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    now = digest()

    if a.record or not os.path.exists(STATE):
        json.dump(now, open(STATE, "w"), indent=1, sort_keys=True)
        print(f"[freeze] recorded {len(now)} paper source files as the agreed baseline")
        return 0

    was = json.load(open(STATE))
    changed = sorted(k for k in set(was) | set(now) if was.get(k) != now.get(k))
    if not changed:
        print(f"[freeze] paper/ unchanged ({len(now)} files match the baseline)")
        return 0
    print(f"[freeze] {len(changed)} PAPER FILE(S) CHANGED SINCE THE AGREED BASELINE:")
    for k in changed:
        kind = ("added" if k not in was else "deleted" if k not in now else "modified")
        print(f"    {kind:9s} {k}")
    print("  If this was agreed, re-record the baseline. If not, it needs discussing.")
    raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
