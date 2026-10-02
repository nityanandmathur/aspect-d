#!/usr/bin/env bash
# v1.4 close-out: collect the extended sweep, refit, regenerate, rebuild, verify.
#
# results/artifacts/runs.csv is a v1.0 immutable, so the extension is collected to its own
# file and only src/v14_analysis.py reads it. Every declared v1.0 quantity still
# comes from the frozen table.
#
# Steps 3 to 5 write and build the paper sources, a checkout of
# github.com/nityanandmathur/aspect-d-paper: $ASPECTD_PAPER_DIR, by default ../aspect-d-paper
# next to this repo. Without a checkout those steps are skipped with a message.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${ASPECTD_PY:-python}"
export ASPECTD_PAPER_DIR="${ASPECTD_PAPER_DIR:-$(dirname "$PWD")/aspect-d-paper}"
HAVE_PAPER=1
if [ ! -d "$ASPECTD_PAPER_DIR" ]; then
  HAVE_PAPER=
  echo "   NOTE: no paper sources at $ASPECTD_PAPER_DIR; steps 3-5 skip the paper (clone"
  echo "   github.com/nityanandmathur/aspect-d-paper next to this repo or set ASPECTD_PAPER_DIR)"
fi

echo "== 1. collect the extended surface (NOT over results/artifacts/runs.csv) =="
(cd src && $PY evaluate.py collect --out ../results/artifacts-v1.4/runs_extended.csv)
# the frozen table is 225 rows (45 runs x 5 T); the "75-point surface" is that after
# seed aggregation. Check it against git rather than a row count, which is what a
# magic number would have let through.
# The immutables have moved since the tag, so each is compared at its tagged path against its
# current path (committed tree, then working tree).
for p in artifacts:results/artifacts PREREGISTRATION.md:docs/preregistration/PREREGISTRATION.md \
         LOG.md:archive/research-log/LOG.md; do
  if ! git diff --quiet "v1.0-submission-candidate:${p%%:*}" "HEAD:${p#*:}" \
     || ! git diff --quiet HEAD -- "${p#*:}"; then
    echo "   ABORT: a v1.0 immutable has been modified (${p#*:})"; exit 1
  fi
done
$PY - <<'EOF'
import pandas as pd
d = pd.read_csv("results/artifacts-v1.4/runs_extended.csv")
print(f"   extended: {len(d)} rows, T in {sorted(d['T'].unique())}, "
      f"{d.groupby(['config','seed']).ngroups} runs")
print("   per-T coverage:", d.groupby("T").size().to_dict())
EOF

echo "== 2. refit: range sensitivity is now measurable =="
(cd src && $PY v14_analysis.py)

echo "== 3. regenerate macros and tables (guard must pass) =="
if [ -n "$HAVE_PAPER" ]; then
  $PY src/paper_v14.py
else
  echo "   SKIPPED: no paper sources"
fi

echo "== 4. rebuild the paper =="
if [ -n "$HAVE_PAPER" ]; then
(cd "$ASPECTD_PAPER_DIR" && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 \
          && bibtex main >/dev/null 2>&1 || true
 cd . && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true)
(cd "$ASPECTD_PAPER_DIR" && pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true)
echo "   LaTeX errors: $(grep -c '^! ' "$ASPECTD_PAPER_DIR/main.log" || true)"
else
  echo "   SKIPPED: no paper sources"
fi

echo "== 5. verify =="
$PY src/check_claims.py
if [ -z "$HAVE_PAPER" ]; then
  echo "   SKIPPED the page count: no paper sources"
else
$PY - "$ASPECTD_PAPER_DIR/main.pdf" <<'EOF'
import subprocess, re, sys
t = subprocess.run(["pdftotext","-layout",sys.argv[1],"-"],
                   capture_output=True, text=True).stdout.split("\f")
i = next(k for k,p in enumerate(t) if re.search(r'^\s*\d*\s*References\s*$', p, re.M))
ls = t[i].split("\n"); j = next(k for k,l in enumerate(ls) if "References" in l)
spill = len([l for l in ls[:j] if l.strip()])
print(f"   {len(t)-1} pages; References on p{i+1}; main text ends "
      f"{'at top of' if spill==0 else f'{spill} lines into'} that page")
print("   MAIN TEXT WITHIN 8 PAGES:", i+1 <= 8 or (i+1 == 9 and spill == 0))
EOF
fi
echo "== done =="
