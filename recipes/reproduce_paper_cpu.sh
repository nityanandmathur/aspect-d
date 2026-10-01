#!/usr/bin/env bash
# Tier A: re-derive every generated number, table and figure of the paper from the released
# run records, on a CPU, in minutes -- and FAIL LOUDLY if anything differs.
#
#   TIER=cpu bash recipes/00_env.sh            # once: numpy/pandas/scipy/matplotlib + PyMuPDF
#   DATASET_JSON=/path/to/hf/dataset.json bash recipes/reproduce_paper_cpu.sh
#
# Environment:
#   PAPER_DIR      committed camera-ready sources to check against (default: <repo>/paper;
#                  the Overleaf mirror repo aspect-d-paper has the same files at its root)
#   DATASET_JSON   dataset.json from the HF model repo root (needed by src/paper.py for
#                  \Nhours, \Nspeakers, \Nsecperchar). If unset it is fetched from
#                  nityanandmathur/aspect-d-masked-diffusion-tts with huggingface_hub.
#   WORK           scratch directory (default: a fresh mktemp -d). Nothing outside it is written.
#   DEEP=1         also re-derive artifacts/runs.csv from runs/*/ (evaluate.py collect; needs
#                  torch + soundfile importable) and refit fits.json (2,000-rep bootstrap,
#                  tens of minutes), comparing both against the committed artifacts
#   ALLOW_UNVERIFIABLE=1  exit 0 when the only problems are items the released records
#                  cannot support (see below); default is to exit 2
#
# Exit status: 0 everything regenerated and identical; 1 MISMATCH (report lists every item);
# 2 no mismatch, but some items are UNVERIFIABLE from the release. On the camera-ready
# release this is expected to be 2: the 180k-step rows of Table tab:trend need eight runs
# whose records were never committed (runs-v1.4/ is gitignored).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PAPER_DIR="${PAPER_DIR:-$REPO/paper}"
WORK="${WORK:-$(mktemp -d "${TMPDIR:-/tmp}/aspectd-repro.XXXXXX")}"
mkdir -p "$WORK"
COPY="$WORK/aspect-d"
need_file "$PAPER_DIR/main.tex"
"$PY" -c 'import numpy, pandas, scipy, matplotlib' 2>/dev/null \
  || die "analysis stack missing in $PY (run: TIER=cpu bash recipes/00_env.sh)"

say "scratch copy -> $COPY"
rm -rf "$COPY"
if command -v rsync >/dev/null; then
  rsync -a --exclude .git --exclude .venv --exclude data --exclude models \
    --exclude 'ckpt.pt' --exclude '*.flac' "$REPO/" "$COPY/"
else
  cp -R "$REPO" "$COPY"; rm -rf "$COPY/.git"
fi
# audit macro usage against the camera-ready main.tex (and the tab_*.tex it \inputs,
# resolved next to it in PAPER_DIR), not whatever the mirror holds. compare_outputs.py checks
# exactly the generated files main.tex \inputs and the figures it \includegraphics.
cp "$PAPER_DIR/main.tex" "$COPY/paper/main.tex"
# remove every file the generators should produce, so a generator that silently writes
# nothing cannot "pass" by leaving the committed copy in place
rm -f "$COPY"/paper/{numbers,numbers_v14,numbers_v15,tab_gapclosed,tab_scope,tab_menc,tab_trend,appendix_grid}.tex \
      "$COPY"/artifacts-v1.2/figures/identity_ledger.{pdf,svg}

mkdir -p "$WORK/data/proc"
if [ -n "${DATASET_JSON:-}" ]; then
  cp "$DATASET_JSON" "$WORK/data/proc/dataset.json"
else
  "$PY" - "$WORK/data/proc" <<'EOF'
import shutil, sys
from huggingface_hub import hf_hub_download
p = hf_hub_download("nityanandmathur/aspect-d-masked-diffusion-tts", "dataset.json")
shutil.copy(p, sys.argv[1] + "/dataset.json")
EOF
fi

say "regenerating (recipes/05_fit_and_paper.sh inside the copy)"
rm -rf "$WORK/regen-figures" "$WORK/v15_partial.json"
if ! ( export REPO="$COPY" ASPECTD_DATA="$WORK/data" FIG_OUT="$WORK/regen-figures" V15_OUT="$WORK"
  [ -n "${DEEP:-}" ] && export REFIT=1
  if [ -n "${DEEP:-}" ]; then
    # runs.csv from the per-run records; artifacts/runs.csv is the T<=16 subset
    # (errexit is off inside an if-condition, so every step exits explicitly on failure)
    ( cd "$COPY/src" && "$PY" evaluate.py collect --out "$COPY/artifacts-camera/runs_all_T.csv" ) || exit 1
    "$PY" - "$COPY/artifacts-camera/runs_all_T.csv" "$COPY/artifacts/runs.csv" <<'EOF' || exit 1
import sys, numpy as np, pandas as pd
new, ref = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
new = new[new["T"].isin(sorted(ref["T"].unique()))]
k = ["config", "seed", "T"]
new, ref = (d.sort_values(k).reset_index(drop=True) for d in (new, ref[new.columns]))
bad = [c for c in ref.columns if not (
    np.allclose(new[c], ref[c], rtol=0, atol=1e-12, equal_nan=True) if ref[c].dtype.kind in "fi"
    else (new[c].astype(str) == ref[c].astype(str)).all())]
print(f"[deep] runs.csv: {len(new)} vs {len(ref)} rows; differing columns: {bad or 'none'}")
sys.exit(1 if bad or len(new) != len(ref) else 0)
EOF
  fi
  cp "$COPY/artifacts/fits.json" "$WORK/fits_committed.json" || exit 1
  bash "$COPY/recipes/05_fit_and_paper.sh" ) 2>&1 | tee "$WORK/regenerate.log"; then
  die "regeneration failed (see $WORK/regenerate.log)"
fi

if [ -n "${DEEP:-}" ]; then
  "$PY" - "$WORK/fits_committed.json" "$COPY/artifacts/fits.json" <<'EOF' | tee -a "$WORK/regenerate.log"
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
da, db = a["decision"], b["decision"]
bad = []
for k in ("delta_tau", "tau_wer", "tau_sim", "delta_rho", "rho_wer", "rho_sim", "kappa_wer", "kappa_sim"):
    if abs(da[k] - db[k]) > 5e-4:
        bad.append((k, da[k], db[k]))
for k in ("delta_tau_ci", "tau_wer_ci", "tau_sim_ci"):
    if max(abs(x - y) for x, y in zip(da[k], db[k])) > 2e-3:
        bad.append((k, da[k], db[k]))
for k in ("outcome_class", "H-D1", "H-D2", "H-D3", "gate_g4_passes"):
    if da[k] != db[k]:
        bad.append((k, da[k], db[k]))
print("[deep] refit fits.json vs committed:", "consistent" if not bad else f"MISMATCH {bad}")
sys.exit(1 if bad else 0)
EOF
fi

FIGS=()
for f in aniso_contours_T16 substitution_plane step_curves extrapolation; do
  FIGS+=("$f.pdf=$WORK/regen-figures/$f.pdf")
done
# Figure fig:ledger: src/s0_figure.py
FIGS+=("identity_ledger.pdf=$COPY/artifacts-v1.2/figures/identity_ledger.pdf")
V15=()
[ -f "$WORK/v15_partial.json" ] && V15=(--v15 "$WORK/v15_partial.json")

set +e
"$PY" "$RECIPES_DIR/lib/compare_outputs.py" --ref "$PAPER_DIR" --new "$COPY/paper" \
  ${V15[@]+"${V15[@]}"} --tex "$PAPER_DIR/main.tex" --figs "${FIGS[@]}" --report "$WORK/report.md"
rc=$?
set -e
say "report: $WORK/report.md (exit $rc)"
if [ "$rc" -eq 1 ]; then die "MISMATCH between regenerated and committed paper inputs"; fi
if [ "$rc" -eq 2 ] && [ -z "${ALLOW_UNVERIFIABLE:-}" ]; then
  printf '\n[reproduce_paper_cpu] UNVERIFIABLE items remain (see report); set ALLOW_UNVERIFIABLE=1 to accept\n' >&2
  exit 2
fi
say "all regenerated items identical to $PAPER_DIR"
