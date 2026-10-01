"""Build the scale-up index: merge tokens, exclude the eval speakers, assign splits.

The single thing that must not go wrong here is leakage. Every number the paper reports is
measured on 400 cross-sentence items from 200 speakers held out of the v1.0 training set. If
any of those speakers appears in the scale-up corpus, the scale-up model has seen them and
no comparison to the paper means anything. Emilia speaker ids are stable across shards, so
the exclusion is exact rather than probabilistic, and this asserts it rather than assuming.

The evaluation set itself is untouched: it stays the frozen v1.0 `eval_zs.json`, scored by
the frozen stack, so a scale-up number sits on the same axis as every number already
reported.

    python src/index_xl.py --max-hours 0     # 0 = keep everything
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XL = os.environ.get("ASPECTD_XL", os.path.join(REPO, "data-xl"))
PROC = os.path.join(XL, "proc")
V10 = os.path.join(os.environ.get("ASPECTD_DATA", os.path.join(REPO, "data")), "proc")
N_VAL = 2000
RNG = 1234


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=float, default=0.0)
    a = ap.parse_args()

    idx = pd.read_parquet(os.path.join(PROC, "index_all.parquet"))
    tok = pd.concat([pd.read_parquet(f)
                     for f in sorted(glob.glob(os.path.join(PROC, "tokidx_*.parquet")))],
                    ignore_index=True)
    df = idx.merge(tok, on="id", how="inner")
    print(f"[index] {len(df):,} clips with tokens, {df.duration.sum()/3600:,.0f} h")

    # --- leakage control, asserted -------------------------------------------------
    held = set(json.load(open(os.path.join(V10, "heldout_speakers.json"))))
    ev = json.load(open(os.path.join(V10, "eval_zs.json")))
    ev_spk = {e["speaker"] for e in ev}
    ev_ids = {e["prompt_id"] for e in ev} | {e["target_id"] for e in ev}
    before = len(df)
    df = df[~df.speaker.isin(held | ev_spk)]
    df = df[~df.id.isin(ev_ids)]
    print(f"[index] removed {before-len(df):,} clips from {len(held|ev_spk):,} eval speakers")
    assert not (set(df.speaker) & (held | ev_spk)), "eval speaker leaked into training"
    assert not (set(df.id) & ev_ids), "eval clip leaked into training"

    if a.max_hours:
        # keep whole speakers, largest-first, until the budget is met: truncating mid
        # speaker would bias who the model has heard most of
        g = df.groupby("speaker").duration.sum().sort_values(ascending=False)
        keep, tot = [], 0.0
        for spk, d in g.items():
            if tot >= a.max_hours * 3600:
                break
            keep.append(spk); tot += d
        df = df[df.speaker.isin(set(keep))]
        print(f"[index] capped to {df.duration.sum()/3600:,.0f} h over {len(keep):,} speakers")

    rng = np.random.default_rng(RNG)
    val_spk = set(rng.choice(sorted(set(df.speaker)), size=min(N_VAL, df.speaker.nunique()),
                             replace=False))
    df["split"] = np.where(df.speaker.isin(val_spk), "val", "train")
    # one clip per val speaker keeps validation cheap and speaker-balanced
    val = df[df.split == "val"].groupby("speaker", as_index=False).head(1)
    df.loc[~df.index.isin(val.index) & (df.split == "val"), "split"] = "train"

    df = df.sort_values("id").reset_index(drop=True)
    df.to_parquet(os.path.join(PROC, "index.parquet"))
    tr = df[df.split == "train"]
    print(f"[index] train {len(tr):,} clips / {tr.duration.sum()/3600:,.0f} h / "
          f"{tr.speaker.nunique():,} speakers | val {int((df.split=='val').sum()):,}")
    json.dump({"train_clips": int(len(tr)), "train_hours": float(tr.duration.sum()/3600),
               "train_speakers": int(tr.speaker.nunique()),
               "val_clips": int((df.split == "val").sum()),
               "eval_speakers_excluded": int(len(held | ev_spk)),
               "source": "Emilia + Emilia-YODAS EN, 900 shards",
               "eval_set": "frozen v1.0 eval_zs.json (unchanged)"},
              open(os.path.join(PROC, "dataset.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
