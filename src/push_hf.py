"""Push trained ASPECT-D models to the Hugging Face model repo (task.md directive 9).

One model repo holds every run as a folder `<config>_<seed>/`:
    model.safetensors   bf16 weights (no optimizer state)
    config.json         shape, param counts, LR, seed, recipe, status
    run.json            full training record (val history, GPU-h)
plus shared assets at the root: phone_vocab.json, dataset.json, grid.json, protocol.html,
runs.csv, fits.json. The model card (README.md) is uploaded with the shared assets only
when --card is given; by default the README on the Hub is left untouched.

    python src/push_hf.py --repo nityanandmathur/aspect-d-masked-diffusion-tts
    python src/push_hf.py --runs A1_0 --card /path/to/hf-repo/README.md   # also replace the card
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import tempfile

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.environ.get("ASPECTD_DATA", os.path.join(REPO_ROOT, "data")) + "/proc"


def token() -> str:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        for line in open(os.path.join(REPO_ROOT, ".env")):
            if line.startswith("HF_TOKEN="):
                tok = line.strip().split("=", 1)[1]
    assert tok, "HF_TOKEN not found"
    return tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="nityanandmathur/aspect-d-masked-diffusion-tts")
    ap.add_argument("--runs", default=None, help="comma list; default = all completed runs")
    ap.add_argument("--include-sweeps", action="store_true")
    ap.add_argument("--runs-dir", default="runs",
                    help="repo-relative run root; use runs-v1.1 for the extension models")
    ap.add_argument("--prefix", default="", help="path prefix inside the HF repo")
    ap.add_argument("--no-shared", action="store_true",
                    help="skip the shared root assets (already pushed by the v1.0 run)")
    ap.add_argument("--card", default=None,
                    help="model card uploaded as README.md; default: leave the Hub README untouched")
    a = ap.parse_args()
    if a.card and a.no_shared:
        ap.error("--card is uploaded with the shared assets; drop --no-shared")
    if a.card and not os.path.isfile(a.card):
        ap.error(f"--card {a.card}: no such file")
    from huggingface_hub import HfApi
    from safetensors.torch import save_file
    api = HfApi(token=token())
    api.create_repo(a.repo, repo_type="model", private=True, exist_ok=True)

    run_dirs = sorted(glob.glob(os.path.join(REPO_ROOT, a.runs_dir, "*")))
    if a.runs:
        want = set(a.runs.split(","))
        run_dirs = [d for d in run_dirs if os.path.basename(d) in want]
    elif not a.include_sweeps:
        run_dirs = [d for d in run_dirs if not os.path.basename(d).startswith("sweep_")]

    pushed = []
    for d in run_dirs:
        name = os.path.basename(d)
        ck = os.path.join(d, "ckpt.pt")
        rj = os.path.join(d, "run.json")
        if not (os.path.exists(ck) and os.path.exists(rj)):
            continue
        run = json.load(open(rj))
        st = torch.load(ck, map_location="cpu", weights_only=False)
        sd = {k: v.to(torch.bfloat16) if v.is_floating_point() else v
              for k, v in st["model"].items()}
        with tempfile.TemporaryDirectory() as tmp:
            save_file(sd, os.path.join(tmp, "model.safetensors"))
            cfg = {"arch": "aspect-d bidirectional masked-diffusion transformer over Mimi tokens",
                   "config_id": run["config"], "seed": run["seed"], "width": run["width"],
                   "depth": run["depth"], "heads": run["heads"],
                   "nonembed_params": run["nonembed_params"],
                   "total_params": run["total_params"], "mup_base_width": 256,
                   "lr_base": run["lr"], "steps": st["step"], "status": run["status"],
                   "codec": "kyutai/mimi (12.5 Hz, 8 codebooks, 2048+MASK+PAD)",
                   "phoneme_vocab_file": "phone_vocab.json",
                   "final_val_loss": run.get("final_val_loss"),
                   "gpu_hours": run.get("gpu_hours"), "dtype": "bfloat16"}
            json.dump(cfg, open(os.path.join(tmp, "config.json"), "w"), indent=1)
            json.dump(run, open(os.path.join(tmp, "run.json"), "w"), indent=1)
            api.upload_folder(folder_path=tmp, path_in_repo=a.prefix + name, repo_id=a.repo,
                              commit_message=f"add {name} ({run['status']}, step {st['step']})")
        pushed.append(name)
        print(f"[hf] pushed {name}", flush=True)

    if a.no_shared:
        # an extension-only push leaves the shared root assets (and the card) as they are
        print(f"[hf] {len(pushed)} runs pushed to https://huggingface.co/{a.repo} "
              f"(shared assets left untouched)", flush=True)
        return

    with tempfile.TemporaryDirectory() as tmp:
        for src, dst in ((os.path.join(PROC, "phone_vocab.json"), "phone_vocab.json"),
                         (os.path.join(PROC, "dataset.json"), "dataset.json"),
                         (os.path.join(REPO_ROOT, "configs", "grid.json"), "grid.json"),
                         (os.path.join(REPO_ROOT, "protocol.html"), "protocol.html"),
                         (os.path.join(REPO_ROOT, "artifacts", "runs.csv"), "runs.csv"),
                         (os.path.join(REPO_ROOT, "artifacts", "fits.json"), "fits.json")):
            if os.path.exists(src):
                open(os.path.join(tmp, dst), "wb").write(open(src, "rb").read())
        if a.card:
            shutil.copy(a.card, os.path.join(tmp, "README.md"))
        api.upload_folder(folder_path=tmp, repo_id=a.repo, commit_message="update shared assets")
    print(f"[hf] {len(pushed)} runs in https://huggingface.co/{a.repo}", flush=True)


if __name__ == "__main__":
    main()
