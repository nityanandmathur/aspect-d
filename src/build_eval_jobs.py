"""Build synth + score job files for the v1.1 training jobs (E4 budget-D, E6 90k).

E4 (§4-E4): full T grid {1,2,4,8,16} × 400 items + extended {24,32,64} × 200 items.
E6 (§4-E6): v1.0 T grid {1,2,4,8,16} × 400 items, so the 90k τ-surface is directly
comparable with the matched 30k runs.

Only runs whose `run.json` reports `completed` are included, so this can be run
repeatedly as training finishes and it will simply pick up more work.

    python src/build_eval_jobs.py --which e4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(REPO, "logs-v1.1")
PY = os.environ.get("ASPECTD_PY", sys.executable)

SPEC = {
    "e4": {"runs": [f"D{i}_{s}" for i in range(1, 6) for s in (0, 1)],
           "grid": [(T, 400) for T in (1, 2, 4, 8, 16)] + [(T, 200) for T in (24, 32, 64)]},
    "e6": {"runs": ["C1_0_90k", "C3_0_90k", "C5_0_90k"],
           "grid": [(T, 400) for T in (1, 2, 4, 8, 16)]},
}


def completed(name: str) -> bool:
    rj = os.path.join(REPO, "runs-v1.1", name, "run.json")
    if not os.path.exists(rj):
        return False
    return json.load(open(rj)).get("status") == "completed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=sorted(SPEC))
    a = ap.parse_args()
    spec = SPEC[a.which]
    ready = [r for r in spec["runs"] if completed(r)]
    synth, score = [], []
    for r in ready:
        d = os.path.join(REPO, "runs-v1.1", r)
        for T, items in spec["grid"]:
            sdir = os.path.join(d, f"synth_T{T}")
            # scores.json is written by the scorer; synth.json by the sampler
            if not os.path.exists(os.path.join(sdir, "synth.json")):
                synth.append({"key": f"synth_{r}_T{T}",
                              "cmd": [PY, "sample.py", "synth", "--run", d,
                                      "--T", str(T), "--items", str(items)]})
            if not os.path.exists(os.path.join(sdir, "scores.json")):
                jf = os.path.join(LOGS, f"score_{a.which}_{r}_T{T}.json")
                json.dump([{"run": d, "T": T, "items": items}], open(jf, "w"))
                score.append({"key": f"score_{r}_T{T}",
                              "cmd": [PY, "evaluate.py", "score", "--jobs", jf]})
    json.dump(synth, open(os.path.join(LOGS, f"{a.which}_synth_jobs.json"), "w"), indent=1)
    json.dump(score, open(os.path.join(LOGS, f"{a.which}_eval_score_jobs.json"), "w"), indent=1)
    print(f"[{a.which}] {len(ready)}/{len(spec['runs'])} runs trained; "
          f"{len(synth)} synth + {len(score)} score jobs pending", flush=True)


if __name__ == "__main__":
    main()
