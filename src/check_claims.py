"""Fail loudly when a retracted claim is still live somewhere in the repo.

This exists because of a specific, repeated defect: a claim was retracted in one file
and left standing in six others, and on one occasion was rebuilt inside the very page
that retracted it. Grepping by hand does not catch that. Each entry below pairs a
pattern with the reason it was withdrawn and the files allowed to still contain it --
the append-only record and the frozen pre-registration must keep their history, so
they are exempt by design rather than by oversight.

    python src/check_claims.py          # non-zero exit if any live occurrence remains
"""
from __future__ import annotations

import os
import re
import sys
from typing import List, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# history files: retracted claims MUST survive here, that is what they are for
EXEMPT = ("RESULTS-FEED.md", "LOG.md", "LOG-v1.1.md", "LOG-v1.2.md",
          "PREREGISTRATION.md", "PREREGISTRATION-v1.2.md", "PREREGISTRATION-v1.3.md",
          "DECISION.md", "DECISION-v1.2.md", "protocol.html", "task-v1.md",
          "task-v2.md", "check_claims.py", "coordinate-audit.html", "ICLR-NOTES-v2.md")

# (pattern, why it was withdrawn, extra files allowed to contain it)
CLAIMS: List[Tuple[str, str, Tuple[str, ...]]] = [
    (r"4\s*[x×]\s*saturation gap",
     "raw-scale artifact; affine-invariant T* is 16 for both metrics", ()),
    (r"steps rent depth,? not width",
     "reads as a rate claim, and Delta-tau > 0 means WER converges sooner", ()),
    (r"compression artifact",
     "tau is affine-invariant; base-plus-sv is excluded for failing G0(c), not for scale",
     ()),
    (r"never reaches the .{0,12}ridge",
     "false: B5 at d=30 and C5 at d=36 do reach it", ()),
    (r"depth-first (latency )?allocation rule",
     "H-E2 refuted at 5.9% measured against an 80% bar", ()),
    (r"no (finite )?(depth[-–]step )?exchange rate (exists|describes)",
     "overreach: the separable fit implies a local kappa = tau/beta", ()),
    (r"base-plus-sv has no discriminative power",
     "wrong: AUC 0.9834, EER 5.75%; the gate failure was cosine scale", ()),
    (r"variable-prompt training does not repair",
     "the --variable-prompt flag was a no-op; H-T3 is UNTESTED, not refuted", ()),
    (r"context is not a training limitation",
     "rests on H-T3, whose flag was a no-op; the question is untested", ()),
    # v1.4: the exponent contrast may be quoted, but never as coordinate-free
    (r"\\Delta\\tau[^.]{0,80}\bproperty of the systems\b",
     "Delta-tau's sign reverses under monotone non-affine reparameterisation", ()),
]

SCAN_EXT = (".tex", ".md", ".html", ".py")
SKIP_DIRS = {".git", "runs", "runs-v1.1", "runs-v1.3", "runs-v1.4", "data",
             ".cache", "logs-v1.4", "node_modules", "paper-v1.0"}


def main() -> int:
    hits = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if not fn.endswith(SCAN_EXT) or fn in EXEMPT:
                continue
            p = os.path.join(root, fn)
            try:
                text = open(p, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for pat, why, extra in CLAIMS:
                if fn in extra:
                    continue
                for m in re.finditer(pat, text, re.I):
                    # A retraction has to be able to name what it retracts, and every
                    # one of ours does so inside quotation marks. Test whether the match
                    # falls *within* a quoted span rather than whether it is exactly
                    # delimited by one -- the withdrawn phrases are usually quoted as
                    # part of a longer title.
                    lo = text.rfind("\n", 0, m.start()) + 1
                    hi = text.find("\n", m.end())
                    ctx = text[lo:hi if hi > 0 else len(text)]
                    # an explicit marker on the line is the intentional way to keep a
                    # withdrawn phrase visible (e.g. a superseded title in a plan)
                    if re.search(r'\bWITHDRAWN\b|\bRETRACTED\b', ctx):
                        continue
                    a0, b0 = m.start() - lo, m.end() - lo
                    spans = [(q.start(), q.end()) for q in
                             re.finditer(r'"[^"]{0,300}"|\u201c[^\u201d]{0,300}\u201d', ctx)]
                    if any(x <= a0 and b0 <= y for x, y in spans):
                        continue
                    line = text[:m.start()].count("\n") + 1
                    hits.append((os.path.relpath(p, REPO), line, m.group(0), why))

    if not hits:
        print(f"[claims] clean: no retracted claim is live "
              f"({len(CLAIMS)} patterns checked)")
        return 0
    print(f"[claims] {len(hits)} LIVE RETRACTED CLAIM(S):\n")
    for f, ln, txt, why in hits:
        print(f"  {f}:{ln}\n      matched: {txt!r}\n      withdrawn because: {why}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
