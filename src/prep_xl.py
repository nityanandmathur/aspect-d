"""Build the scale-up training corpus: shard metadata -> index -> phonemes -> Mimi tokens.

Differs from the v1.0 pipeline in three ways, all deliberate:

- It reads per-shard metadata parquets written by `fetch_xl.py` rather than re-scanning
  tars, because the tars are deleted once encoded and would not be there to re-scan.
- It selects **everything** that passes the duration filter instead of a 2,000 h
  speaker-stratified subset. The v1.0 subset exists to hold data constant across the shape
  grid; here the point is the opposite.
- It retires each tar as soon as that shard's tokens are on disk, so raw audio never
  accumulates. Disk stays flat while the corpus grows.

The v1.0 root is never touched: everything lives under ASPECTD_XL.

    python src/prep_xl.py --loop      # run alongside fetch_xl.py until it drains
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XL = os.environ.get("ASPECTD_XL", "/home/ubuntu/data-xl")
RAW = os.path.join(XL, "emilia_raw")
PROC = os.path.join(XL, "proc")
META = os.path.join(XL, "shard_meta")
TRAIN_DUR = (4.0, 18.0)          # same filter as v1.0, so clips are comparable
MIN_DNSMOS = 3.0


def build_index() -> pd.DataFrame:
    """Concatenate the per-shard metadata and apply the v1.0 duration/quality filter."""
    fs = sorted(glob.glob(os.path.join(META, "*.parquet")))
    if not fs:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df = df[df.duration.between(*TRAIN_DUR) & (df.dnsmos >= MIN_DNSMOS)].copy()
    df["n_chars"] = df.text.str.len()
    df = df[df.n_chars > 0]
    os.makedirs(PROC, exist_ok=True)
    df.to_parquet(os.path.join(PROC, "index_all.parquet"))
    return df


def encode_shards(shards, gpus: int = 8) -> None:
    """Encode with the existing worker, one shard per task, ASPECTD_DATA pointed at XL."""
    env = dict(os.environ, ASPECTD_DATA=XL)
    idx = pd.read_parquet(os.path.join(PROC, "index_all.parquet"))
    todo = [s for s in shards
            if not os.path.exists(os.path.join(PROC, f"tokidx_{s}.parquet"))]
    if not todo:
        return
    sub = idx[idx.shard.isin(todo)]
    sub.to_parquet(os.path.join(PROC, "index.parquet"))
    print(f"[prep] encoding {len(todo)} shards, {len(sub)} clips", flush=True)
    subprocess.run([sys.executable, os.path.join(REPO, "src", "data.py"), "encode"],
                   env=env, cwd=os.path.join(REPO, "src"))


def retire(shards) -> int:
    """Delete tars whose tokens are on disk. This is what keeps disk flat."""
    n = 0
    for s in shards:
        if os.path.exists(os.path.join(PROC, f"tokidx_{s}.parquet")):
            for p in (os.path.join(RAW, s + ".tar"), os.path.join(RAW, s + ".tar.done")):
                if os.path.exists(p):
                    os.unlink(p)
                    n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--batch", type=int, default=24, help="shards per encode pass")
    a = ap.parse_args()
    while True:
        df = build_index()
        if df.empty:
            time.sleep(120)
            continue
        ready = sorted({os.path.basename(p)[:-9] for p in glob.glob(os.path.join(RAW, "*.tar.done"))})
        pending = [s for s in ready
                   if not os.path.exists(os.path.join(PROC, f"tokidx_{s}.parquet"))]
        if pending:
            encode_shards(pending[:a.batch])
            freed = retire(ready)
            done = len(glob.glob(os.path.join(PROC, "tokidx_*.parquet")))
            hrs = df[df.shard.isin(
                {os.path.basename(p)[7:-8] for p in glob.glob(os.path.join(PROC, "tokidx_*.parquet"))}
            )].duration.sum() / 3600
            import shutil
            print(f"[prep] {done} shards tokenised, {hrs:,.0f} h, retired {freed} files, "
                  f"{shutil.disk_usage(XL).free/2**30:.0f} GB free", flush=True)
        elif not a.loop:
            break
        else:
            fetching = subprocess.run(["pgrep", "-f", "fetch_xl"],
                                      capture_output=True, text=True).stdout.strip()
            if not fetching:
                print("[prep] nothing pending and the fetcher has stopped", flush=True)
                break
            time.sleep(120)


if __name__ == "__main__":
    main()
