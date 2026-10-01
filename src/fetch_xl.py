"""Stream Emilia-YODAS EN shards for the scale-up run, bounded by free disk.

The frozen 2,000 h subset under $ASPECTD_DATA is a v1.0 immutable, so everything here
goes to a separate root ($ASPECTD_XL, default <repo>/data-xl). Nothing in this file touches
the original.

Shards are ~1.02 GB each and there are 2,502 of them (~68,000 h, ~2.5 TB), which does not
fit. Tokens do: 1.3 GB bought 3,868 h, so the whole corpus tokenises to ~16 GB. The loop
therefore downloads, records the shard's metadata, and stops holding more raw audio than the
disk allows -- `--keep` shards at a time, oldest deleted once encoded.

Runs concurrently with encoding: both are idempotent per shard, keyed on files on disk.

    python src/fetch_xl.py --target-shards 600 --keep 120
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("ASPECTD_XL", os.path.join(REPO, "data-xl"))
RAW = os.path.join(ROOT, "emilia_raw")
META = os.path.join(ROOT, "shard_meta")
REPO_ID = "amphion/Emilia-Dataset"
PREFIXES = ("Emilia/EN/", "Emilia-YODAS/EN/")   # same-schema source first, then YODAS
MIN_FREE_GB = 80


def token() -> str:
    for l in open(os.path.join(REPO, ".env")):
        if l.strip().startswith("HF_TOKEN") and "=" in l:
            return l.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no HF_TOKEN in .env")


def free_gb() -> float:
    return shutil.disk_usage(ROOT).free / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-shards", type=int, default=600)
    ap.add_argument("--keep", type=int, default=120,
                    help="max raw tars held on disk at once")
    a = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(META, exist_ok=True)
    tok = token()

    from huggingface_hub import HfApi, hf_hub_download
    import sys
    sys.path.insert(0, os.path.join(REPO, "src"))
    import pandas as pd, tarfile

    def scan(path):
        """Emilia proper keys the clip 'id'; Emilia-YODAS keys it '_id'. Everything else
        matches, so normalise rather than maintain two readers."""
        out = []
        with tarfile.open(path) as tf:
            for m in tf:
                if not m.name.endswith(".json"):
                    continue
                r = json.load(tf.extractfile(m))
                cid = r.get("id") or r.get("_id")
                if cid is None:
                    continue
                out.append({"id": cid, "speaker": r["speaker"],
                            "duration": float(r["duration"]), "text": r["text"].strip(),
                            "dnsmos": float(r.get("dnsmos", 0.0)),
                            "language": r.get("language", "en"),
                            "shard": os.path.basename(path)[:-4], "member": m.name[:-5]})
        return out

    allf = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    files = []
    for pref in PREFIXES:                       # keep source order: Emilia, then YODAS
        files += sorted(f for f in allf if f.startswith(pref) and f.endswith(".tar"))
    print(f"[fetch] {len(files)} EN shards on the hub, target {a.target_shards}", flush=True)

    got = 0
    for f in files:
        name = ('Y_' if 'YODAS' in f else '') + os.path.basename(f)[:-4]
        mp_ = os.path.join(META, name + ".parquet")
        if os.path.exists(mp_):
            got += 1
            continue
        if got >= a.target_shards:
            break
        while free_gb() < MIN_FREE_GB:
            print(f"[fetch] only {free_gb():.0f} GB free, waiting for the encoder", flush=True)
            time.sleep(300)
        # too many raw tars held: wait for encoding to retire some
        held = [p for p in os.listdir(RAW) if p.endswith(".tar")]
        while len(held) >= a.keep:
            time.sleep(120)
            held = [p for p in os.listdir(RAW) if p.endswith(".tar")]
        t0 = time.time()
        try:
            p = hf_hub_download(REPO_ID, f, repo_type="dataset", token=tok,
                                local_dir=os.path.join(ROOT, "hf"))
        except Exception as e:
            print(f"[fetch] FAIL {name}: {type(e).__name__} {str(e)[:120]}", flush=True)
            continue
        dst = os.path.join(RAW, name + ".tar")
        shutil.move(p, dst)
        # record the shard's clip metadata now, while the tar is still here: the tar is
        # deleted after encoding and stage_scan would otherwise lose it
        try:
            recs = scan(dst)
            pd.DataFrame(recs).to_parquet(mp_)
        except Exception as e:
            print(f"[fetch] scan FAIL {name}: {type(e).__name__}", flush=True)
            os.unlink(dst)
            continue
        open(dst + ".done", "w").close()
        got += 1
        if got % 10 == 0 or got < 5:
            print(f"[fetch] {got}/{a.target_shards} shards, {len(recs)} clips in {name}, "
                  f"{time.time()-t0:.0f}s, {free_gb():.0f} GB free", flush=True)
    print(f"[fetch] done: {got} shards staged", flush=True)


if __name__ == "__main__":
    main()
