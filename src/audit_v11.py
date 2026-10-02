"""E-AUDIT — checkpoint audit over the 45 v1.0 runs (task-v1.md §1).

Verifies a loadable final checkpoint exists for every (config, seed) of the 15
active configs × seeds {0,1,2}, records path + step + hash, and smoke-tests C3
seed 0 against its frozen v1.0 per-item scores.

    python src/audit_v11.py [--smoke]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from typing import Dict, List

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "artifacts-v1.1")


def sha_head(path: str, n: int = 1 << 20) -> str:
    """Hash the first MiB + size — a full hash of a 2 GB ckpt costs minutes for no gain."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read(n))
    h.update(str(os.path.getsize(path)).encode())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    grid = json.load(open(os.path.join(REPO, "configs", "grid.json")))
    state = json.load(open(os.path.join(REPO, "archive", "research-log", "state.json")))
    active = state["active_configs"]
    seeds = [0, 1, 2]

    rows: List[Dict] = []
    missing: List[str] = []
    for cfg in active:
        for s in seeds:
            name = f"{cfg}_{s}"
            d = os.path.join(REPO, "results", "runs", name)
            ck = os.path.join(d, "ckpt.pt")
            rj = os.path.join(d, "run.json")
            rec = {"run": name, "config": cfg, "seed": s, "ckpt": ck,
                   "exists": os.path.exists(ck)}
            if not rec["exists"]:
                missing.append(name)
                rows.append(rec)
                continue
            try:
                st = torch.load(ck, map_location="cpu", weights_only=False)
                rec.update({"step": int(st["step"]), "loadable": True,
                            "n_tensors": len(st["model"]),
                            "size_bytes": os.path.getsize(ck),
                            "sha16": sha_head(ck),
                            "cfg_width": st["cfg"]["width"], "cfg_depth": st["cfg"]["depth"],
                            "lr": st.get("lr")})
                if os.path.exists(rj):
                    r = json.load(open(rj))
                    rec.update({"status": r.get("status"),
                                "final_val_loss": r.get("final_val_loss"),
                                "gpu_hours": r.get("gpu_hours")})
                del st
            except Exception as e:
                rec.update({"loadable": False, "error": repr(e)[:200]})
                missing.append(name)
            rows.append(rec)

    ok = [r for r in rows if r.get("loadable") and r.get("step") == 30000]
    audit = {"when": "2026-08-06", "n_expected": len(active) * len(seeds),
             "n_loadable_at_30k": len(ok), "n_missing_or_bad": len(missing),
             "missing": missing,
             "frac_missing": len(missing) / max(1, len(active) * len(seeds)),
             "runs": rows}

    if a.smoke:
        audit["smoke"] = smoke_c3(a.device)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "checkpoint_audit.json"), "w") as fh:
        json.dump(audit, fh, indent=1)
    print(f"[audit] {len(ok)}/{len(active)*len(seeds)} checkpoints loadable at step 30000; "
          f"missing/bad: {missing if missing else 'none'}", flush=True)
    if audit.get("smoke"):
        print(f"[audit] smoke: {audit['smoke']['verdict']}", flush=True)
    return audit


def smoke_c3(device: str) -> Dict:
    """Load C3 seed 0, synthesise 5 eval items at T=16, compare to the frozen v1.0
    per-item scores (task-v1.md §1.2)."""
    import numpy as np
    import sample as S
    from data import TokenStore
    store = TokenStore()
    run = os.path.join(REPO, "results", "runs", "C3_0")
    model, cfg, step = S.load_run(run, len(store.vocab), torch.device(device))
    # must reproduce v1.0's batch composition exactly: the sampler's RNG draws are
    # shaped [B, Fmax, V], so a 5-item batch and a 50-item batch see different streams
    # (LOG.md P0-3). Synthesise the full first batch of 50 and compare its first 5 items.
    items = S.load_items(store, S.EVAL_BATCH)
    grid_tok, _ = S.synth_batch(model, items, 16, torch.device(device), 0)
    items = items[:5]
    old = json.load(open(os.path.join(run, "synth_T16", "tokens.npz").replace(
        "tokens.npz", "scores.json")))
    per = {r["item"]: r for r in old["items"]}
    # token-level agreement against the stored v1.0 grids is the sharpest check
    import numpy as np
    npz = np.load(os.path.join(run, "synth_T16", "tokens.npz"))
    agree, tot = 0, 0
    for i, it in enumerate(items):
        a = npz[it["item"]]
        b = grid_tok[i, :len(a)].numpy()
        n = min(len(a), len(b))
        agree += int((a[:n] == b[:n]).sum())
        tot += int(a[:n].size)
    frac = agree / max(1, tot)
    return {"run": "C3_0", "ckpt_step": step, "n_items": len(items),
            "token_agreement_vs_v1.0": frac,
            "v1_0_item_wer": [per[it["item"]]["wer"] for it in items],
            "verdict": "PASS — regenerated tokens match the frozen v1.0 grids"
            if frac > 0.99 else f"CHECK — token agreement {frac:.4f}"}


if __name__ == "__main__":
    main()
