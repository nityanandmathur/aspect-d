#!/usr/bin/env bash
# Runs when the v1.1 evaluation supervisor drains: analyse, publish, push.
# Idempotent — safe to re-run; each analysis refuses if its inputs are incomplete.
set -u
cd "$(dirname "$0")/.."
PY="${ASPECTD_PY:-python}"

echo "[fin] waiting for supervisor to finish ..."
while pgrep -f "python src/supervise_v11" >/dev/null; do sleep 60; done
echo "[fin] supervisor done at $(date -u +%H:%M) UTC"

echo "[fin] === E4 / H-E4 ==="
(cd src && $PY e4_analysis.py 2>&1 | tail -20)

echo "[fin] === E6 / H-E5 ==="
(cd src && $PY e6_analysis.py --n-boot 1000 2>&1 | tail -20)

echo "[fin] === push models to HuggingFace ==="
$PY src/push_hf.py --runs-dir results/runs-v1.1 --prefix "v1.1/" --no-shared 2>&1 | tail -15

echo "[fin] === push to GitHub ==="
git add -A src results/artifacts-v1.1 archive/research-log/{RESULTS-FEED.md,LOG-v1.1.md,state-v1.json} \
  docs/extensions.html
git commit -q -m "E4/E6 complete: 4-budget scale persistence (H-E4) and 90k training-compute control (H-E5)

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" \
  && git push -q origin v1.1-extensions && echo "[fin] pushed $(git rev-parse --short HEAD)"
echo "[fin] COMPLETE $(date -u +%H:%M) UTC"
