"""Regenerate the parts of numbers_v15.tex / tab_trend.tex that the released records support.

src/paper_v15.py needs per-item scores for C{1,3,5} x seeds {0,1,2} at 30k, 90k and 180k
training steps. The 30k and 90k records are committed (results/runs/, results/runs-v1.1/, results/runs-v1.3/); of
the nine 180k runs only C3_0_180k is (results/runs-v1.3/). The other eight lived under results/runs-v1.4/,
which is .gitignored, so paper_v15.py raises on a fresh clone.

This calls paper_v15's own functions (load / shares, unchanged) for the budgets whose records
are complete, in the same order and with the same RNG (BOOT_RNG, N_BOOT), so the 30k and 90k
rows -- including their bootstrap CIs -- are bit-comparable with the committed file. Every
macro it cannot regenerate is listed as UNVERIFIABLE rather than guessed.

    python recipes/lib/v15_partial.py <repo> <out_dir>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

repo, out_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(repo, "src"))
import paper_v15 as P  # noqa: E402

WORD = {"30k": "Thirtyk", "90k": "Ninetyk", "180k": "Eightyk"}


def complete(b: str) -> bool:
    return all(P._items(f"{c}_{s}{P.SUF[b]}", T) is not None
               for c in P.CFG for s in P.SEEDS for T in (1, 16))


avail = [b for b in P.SUF if complete(b)]
missing = [b for b in P.SUF if b not in avail]
# the bootstrap stream is consumed budget by budget in SUF order, so only a PREFIX of SUF
# can be regenerated with identical CIs
prefix = []
for b in P.SUF:
    if b not in avail:
        break
    prefix.append(b)
P.SUF = {b: P.SUF[b] for b in prefix}

D = P.load()
ids = sorted(D[("30k", "C1", 0, 1, "w")])
rng = np.random.default_rng(P.BOOT_RNG)
a2floor = json.load(open(os.path.join(P.OUT, "g0c_asr2.json")))["wer_mean"]
M, rows = {}, []
for b in prefix:                                   # verbatim from paper_v15.main()
    w, s = P.shares(D, b, ids, P.CFG)
    wa, _ = P.shares(D, b, ids, P.CFG, "a", a2floor)
    reps = []
    for _ in range(P.N_BOOT):
        cs = [P.CFG[i] for i in rng.integers(0, len(P.CFG), len(P.CFG))]
        ii = [ids[i] for i in rng.integers(0, len(ids), len(ids))]
        r = P.shares(D, b, ii, cs)
        reps.append(r[0] / r[1])
    ci = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
    M[f"Ngap{WORD[b]}WER"] = f"{100*w:.1f}"
    M[f"Ngap{WORD[b]}SIM"] = f"{100*s:.1f}"
    M[f"Ngap{WORD[b]}Ratio"] = f"{w/s:.2f}$\\times$"
    M[f"Ngap{WORD[b]}Ci"] = f"[{ci[0]:.2f}, {ci[1]:.2f}]"
    M[f"Ngap{WORD[b]}WERasrTwo"] = f"{100*wa:.1f}"
    rows.append(f"    {b} & {100*w:.1f}\\% & {100*wa:.1f}\\% & {100*s:.1f}\\% & "
                f"{w/s:.2f}$\\times$ & {ci[0]:.2f}, {ci[1]:.2f} \\\\")
M["NasrTwoFloor"] = f"{a2floor:.4f}"
M["NasrTwoName"] = "wav2vec2-large-960h-lv60-self"       # literal in paper_v15.py
M["NtrendRuns"] = str(len(P.CFG) * len(P.SEEDS))
M["NtrendBudgets"] = "3"                                  # literal in paper_v15.py

unverifiable = []
for b in missing:
    unverifiable += [f"Ngap{WORD[b]}{k}" for k in ("WER", "SIM", "Ratio", "Ci", "WERasrTwo")]
if "180k" in missing:
    unverifiable += ["NsatWER", "NsatWERasrTwo", "NsatSIM"]
lacking = sorted({f"{c}_{s}{P.SUF.get(b, '_' + b)}" for b in missing for c in P.CFG
                  for s in P.SEEDS if P._items(f"{c}_{s}_{b}", 16) is None})

os.makedirs(out_dir, exist_ok=True)
json.dump({"macros": M, "table_rows": rows, "regenerated_budgets": prefix,
           "missing_budgets": missing, "unverifiable_macros": unverifiable,
           "unverifiable_table_rows": missing, "missing_runs": lacking},
          open(os.path.join(out_dir, "v15_partial.json"), "w"), indent=1)
print(f"[v15] regenerated {prefix}; missing {missing}; records absent for {lacking}")
