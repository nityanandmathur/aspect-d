"""Assemble the `docs/samples/` deliverable — docs/protocol.html §10.

"4 fixed items × {widest, squarest, deepest} of each active budget × T ∈ {1, 16}",
plus the original prompt (the SIM-o reference) and the ground-truth target audio,
a manifest, and a small listening page in the house style.

    python src/samples.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List

import pandas as pd

import site_bar

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_shapes(grid: Dict, budgets: List[str]) -> List[Dict]:
    out = []
    for b in budgets:
        cfgs = sorted([c for c in grid["configs"] if c["budget"] == b], key=lambda c: c["width"])
        if not cfgs:
            continue
        out += [{"role": "deepest", **cfgs[0]}, {"role": "squarest", **cfgs[len(cfgs) // 2]},
                {"role": "widest", **cfgs[-1]}]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "samples"))
    ap.add_argument("--n-items", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    grid = json.load(open(os.path.join(REPO, "configs", "grid.json")))
    st = json.load(open(os.path.join(REPO, "archive", "research-log", "state.json")))
    proc = os.path.join(os.environ.get("ASPECTD_DATA", os.path.join(REPO, "data")), "proc")
    items = sorted(json.load(open(os.path.join(proc, "eval_zs.json"))),
                   key=lambda d: d["item"])[:a.n_items]
    shapes = pick_shapes(grid, st["active_budgets"])
    os.makedirs(a.out, exist_ok=True)
    man, rows = [], []
    for it in items:                                     # references
        for kind, cid in (("prompt", it["prompt_id"]), ("target_gt", it["target_id"])):
            src = os.path.join(proc, "eval_audio", cid + ".flac")
            dst = os.path.join(a.out, f"{it['item']}_{kind}.flac")
            if os.path.exists(src):
                shutil.copy(src, dst)
        man.append({"item": it["item"], "speaker": it["speaker"],
                    "prompt_text": it["prompt_text"], "target_text": it["target_text"],
                    "prompt_file": f"{it['item']}_prompt.flac",
                    "target_gt_file": f"{it['item']}_target_gt.flac"})
    for s in shapes:
        for T in (1, 16):
            for it in items:
                src = os.path.join(REPO, "results", "runs", f"{s['id']}_{a.seed}", f"synth_T{T}",
                                   f"{it['item']}.flac")
                if not os.path.exists(src):
                    continue
                name = f"{it['item']}_{s['id']}_{s['role']}_T{T}.flac"
                shutil.copy(src, os.path.join(a.out, name))
                rows.append({"item": it["item"], "config": s["id"], "role": s["role"],
                             "budget": s["budget"], "width": s["width"], "depth": s["depth"],
                             "T": T, "seed": a.seed, "file": name})
    json.dump({"items": man, "syntheses": rows,
               "note": "protocol §10: 4 fixed items × {widest, squarest, deepest} per active "
                       "budget × T ∈ {1, 16}; prompt/target_gt are the original waveforms"},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=1)

    df = pd.DataFrame(rows)
    css = """body{margin:0;background:#F6F8F8;color:#14181C;font:16px/1.6 "Avenir Next","Segoe UI",
system-ui,sans-serif}main{max-width:1000px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:30px;margin:6px 0}h2{font-size:19px;margin:36px 0 8px}
.eyebrow{font-family:ui-monospace,Consolas,monospace;font-size:12px;letter-spacing:.14em;
text-transform:uppercase;color:#5B6470}table{border-collapse:collapse;width:100%;font-size:14px;
background:#fff;margin:12px 0}th,td{border:1px solid #D9DEDE;padding:6px 8px;vertical-align:top}
th{background:#ECF1F0}audio{width:230px;height:32px}code{background:#ECF1F0;padding:1px 5px;
border-radius:4px;font-family:ui-monospace,Consolas,monospace;font-size:13px}
.sub{color:#5B6470}"""
    H = [f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>ASPECT-D · samples</title>
{site_bar.style(wide_tables=True)}<style>{css}</style></head><body><main>
{site_bar.nav("../index.html")}
<div class="eyebrow">Project ASPECT-D · samples deliverable (protocol §10)</div>
<h1>Zero-shot cloning samples across shapes and refinement steps</h1>
<p class="sub">Seed {a.seed}. Each row is one fixed eval_zs item; the first two columns are the
original 3-second prompt (the SIM-o reference) and the ground-truth target recording, then the
synthesis of each shape at T=1 and T=16. Same frozen sampler everywhere; T is the only
inference-time difference.</p>"""]
    for it in man:
        H.append(f"""<h2>{it['item']} · speaker <code>{it['speaker']}</code></h2>
<p class="sub"><b>prompt text:</b> {it['prompt_text'][:160]}<br>
<b>target text:</b> {it['target_text'][:260]}</p>
<table><tr><th>reference</th><th>audio</th><th>shape</th><th>T=1</th><th>T=16</th></tr>
<tr><td>prompt (original)</td><td><audio controls src="{it['prompt_file']}"></audio></td>
<td rowspan="2" class="sub">ground truth</td>
<td rowspan="2" colspan="2"><audio controls src="{it['target_gt_file']}"></audio></td></tr>
<tr><td>target (ground truth)</td><td class="sub">—</td></tr>""")
        sub = df[df.item == it["item"]] if len(df) else df
        for cid in (sub.config.unique() if len(sub) else []):
            r = sub[sub.config == cid]
            role = r.role.iloc[0]
            w, d = int(r.width.iloc[0]), int(r.depth.iloc[0])
            f1 = r[r["T"] == 1].file.tolist()
            f16 = r[r["T"] == 16].file.tolist()
            H.append(f"""<tr><td colspan="2" class="sub">{cid} ({role})</td>
<td>w={w}, d={d}</td>
<td>{'<audio controls src="%s"></audio>' % f1[0] if f1 else '—'}</td>
<td>{'<audio controls src="%s"></audio>' % f16[0] if f16 else '—'}</td></tr>""")
        H.append("</table>")
    H.append("""<p class="sub" style="margin-top:40px">Generated by <a href="https://github.com/nityanandmathur/aspect-d/blob/main/src/samples.py"><code>src/samples.py</code></a>;
see <a href="../results.html">results.html</a> for the fitted laws and
<a href="../implementation.html">implementation.html</a> for the harness.</p>
</main></body></html>""")
    with open(os.path.join(a.out, "index.html"), "w") as fh:
        fh.write("\n".join(H))
    print(f"[samples] {len(rows)} syntheses + {2*len(man)} references → {a.out}", flush=True)


if __name__ == "__main__":
    main()
