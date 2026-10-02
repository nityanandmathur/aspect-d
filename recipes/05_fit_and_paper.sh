#!/usr/bin/env bash
# 05 -- fits, figures, and every generated LaTeX input of the paper. CPU only.
#
#   bash recipes/05_fit_and_paper.sh               # inside a SCRATCH COPY of the repo
#   REFIT=1 bash recipes/05_fit_and_paper.sh       # also refit fits.json (2,000-rep bootstrap)
#
# For "check every number against the camera-ready", use recipes/reproduce_paper_cpu.sh,
# which makes the scratch copy, runs this script inside it, and diffs the results.
#
# The generators write to FIXED paths inside the repo (paper.py -> paper/numbers.tex via
# --out but paper/appendix_grid.tex always; paper_v14.py / paper_v15.py -> paper/;
# s0_figure.py -> artifacts-v1.2/figures/; fit.py --out). This script therefore refuses to
# run in a git checkout unless ALLOW_ARTIFACT_WRITE=1.
#
# Inputs (all committed, except dataset.json which is on the HF repo):
#   artifacts/runs.csv, artifacts/fits.json, artifacts/c_layer.json, state.json,
#   configs/grid.json, artifacts-v1.1/*.json, artifacts-v1.2/*.json, artifacts-v1.3/*.json,
#   artifacts-v1.4/{analysis.json,x2_x7.json}, artifacts-v1.5/g0c_asr2.json,
#   runs*/<run>/synth_T{1,16}/{scores.json,asr2.json} (paper_v15.py),
#   $ASPECTD_DATA/proc/dataset.json (paper.py: \Nhours, \Nspeakers, \Nsecperchar)
#
# Outputs -> paper table / figure:
#   paper/numbers.tex        src/paper.py      every \N macro of v1.0-v1.3 (Tables ledger, search,
#                                              negative, context, alloc, exponents, robust; prose)
#   paper/appendix_grid.tex  src/paper.py      Appendix "The shape grid, as run"
#   paper/numbers_v14.tex    src/paper_v14.py  floors, gap-closed, scope, encoder macros
#   paper/tab_gapclosed.tex  src/paper_v14.py  Table tab:gapclosed
#   paper/tab_scope.tex      src/paper_v14.py  Table tab:scope
#   paper/tab_menc.tex       src/paper_v14.py  Table tab:menc
#   paper/numbers_v15.tex    src/paper_v15.py  compute-trend + second-ASR macros
#   paper/tab_trend.tex      src/paper_v15.py  Table tab:trend
#   $FIG_OUT/{aniso_contours_T16,substitution_plane,step_curves,extrapolation}.pdf
#                            src/figures.py    Figures fig:diag (left/right), fig:main, H-D4 extrapolation
#   artifacts-v1.2/figures/identity_ledger.pdf  src/s0_figure.py  Figure fig:ledger
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$REPO"

if [ -d "$REPO/.git" ] && [ -z "${ALLOW_ARTIFACT_WRITE:-}" ]; then
  die "this writes paper/ and artifacts-v1.2/figures/ in place; run it in a scratch copy (reproduce_paper_cpu.sh does) or set ALLOW_ARTIFACT_WRITE=1"
fi
"$PY" -c 'import sys; assert sys.version_info >= (3, 12), "src/paper.py needs Python >= 3.12 (PEP 701 f-strings)"'
need_file "$ASPECTD_DATA/proc/dataset.json" "copy dataset.json from the HF repo root"
FIG_OUT="${FIG_OUT:-$REPO/artifacts-camera/figures}"
export MPLBACKEND=Agg

# ---------------------------------------------------------------- 5a. (optional) refit
# Declared analysis: balanced 3-seed composition, weighted NLS with 32 multi-starts (RNG 42),
# run-level bootstrap 2,000 replicates (RNG 7331). Measured on an 18-core Apple M-series CPU
# with --workers 14: see NOTES/recipes.md for wall time. Point estimates reproduce the
# committed fits.json to ~1e-5 relative (scipy/numpy versions differ from the paper venv).
if [ -n "${REFIT:-}" ]; then
  "$PY" src/fit.py --runs artifacts/runs.csv --out artifacts/fits.json --seeds 0,1,2 \
    --workers "${WORKERS:-$(( $(getconf _NPROCESSORS_ONLN) - 2 ))}"
fi

# ---------------------------------------------------------------- 5b. figures
"$PY" src/figures.py --runs artifacts/runs.csv --fits artifacts/fits.json --out "$FIG_OUT"
"$PY" src/s0_figure.py

# ---------------------------------------------------------------- 5c. LaTeX inputs
"$PY" src/paper.py --out paper/numbers.tex --tex paper/main.tex
"$PY" src/paper_v14.py
# paper_v15.py needs the eight 180k runs that live only under the (gitignored) runs-v1.4/.
# On the released records it raises; recipes/lib/v15_partial.py then regenerates the 30k and
# 90k rows with the same functions and RNG and lists what cannot be regenerated.
if ! "$PY" src/paper_v15.py; then
  say "paper_v15.py failed (expected on the released records: runs-v1.4/*_180k absent)"
  rm -f paper/numbers_v15.tex paper/tab_trend.tex
  "$PY" "$RECIPES_DIR/lib/v15_partial.py" "$REPO" "${V15_OUT:-$REPO/artifacts-camera}"
fi
# ---------------------------------------------------------------- 5d. (not part of the paper)
# src/camera_ready_*.py are supplementary analyses written for the reviews (matched-NFE search,
# selector cost, cross-selector control, T=2 / non-degenerate baselines, F5-TTS anchor, compute
# records). The paper does not \input their macros, so they are not run here. To regenerate
# them (CPU only, committed artifacts only; every output goes to artifacts-camera/):
#   "$PY" src/camera_ready_search.py
#   "$PY" src/camera_ready_scope.py --amp-boot 2000 --dataset-json "$ASPECTD_DATA/proc/dataset.json"
#   "$PY" src/camera_ready_release.py
#   "$PY" src/camera_ready_integrate.py
say "done: paper/*.tex, $FIG_OUT/*.pdf, artifacts-v1.2/figures/identity_ledger.pdf"
