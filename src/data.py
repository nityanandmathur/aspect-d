"""ASPECT-D data pipeline — Emilia-EN → selection → phonemes → Mimi token shards.

Policy is fixed by ``configs/grid.json -> data`` / ``protocol.html §3``:
Emilia-EN 2,000 h speaker-stratified subset (selection RNG 1234), ≥4,000
speakers, 200 held-out speakers, val = 2,000 held-in clips, eval_zs = 400
held-out-speaker items with target 4–15 s. Construction of the prompt/target
pairs is LOG.md P0-2; loudness normalisation is task.md §10 (−1 dBFS peak).

Stages (idempotent, resumable):
    scan       tars → data/proc/meta.parquet
    select     meta → data/proc/index.parquet + eval_zs.json + dataset.json
    phonemize  index → phones.npy + phone_vocab.json (vocab from TRAIN only)
    encode     tars → tokens_<shard>.npy (int16 [frames, 8]) + offsets in index
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import tarfile
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DATA_ROOT = os.environ.get("ASPECTD_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
RAW_DIR = os.path.join(DATA_ROOT, "emilia_raw")
PROC_DIR = os.path.join(DATA_ROOT, "proc")
SR = 24000
FRAME_SAMPLES = 1920            # 24000 / 12.5 Hz
PROMPT_FRAMES = 37              # int(3 s * 12.5 Hz), grid.json prompt_seconds = 3
ENC_GROUP = 8                   # clips per Mimi encode call (24 kHz activations are wide)
N_LEVELS = 8
PEAK_DBFS = -1.0                # task.md §10 default
SELECT_RNG = 1234               # grid.json data.primary.selection
TRAIN_HOURS = 2000.0
N_VAL = 2000
N_EVAL = 400
N_HELDOUT_SPK = 200
MIN_SPEAKERS = 4000
TRAIN_DUR = (4.0, 18.0)         # prompt 3 s + max_target 15 s (grid.json training)
PROMPT_DUR = (2.5, 3.5)         # LOG.md P0-2
EVAL_TGT_DUR = (4.0, 15.0)      # grid.json eval_sets.eval_zs.target_seconds


# ------------------------------------------------------------------ stage: scan
def _scan_tar(path: str) -> List[Dict]:
    out = []
    with tarfile.open(path) as tf:
        for m in tf:
            if not m.name.endswith(".json"):
                continue
            r = json.load(tf.extractfile(m))
            out.append({"id": r["id"], "speaker": r["speaker"], "duration": float(r["duration"]),
                        "text": r["text"].strip(), "dnsmos": float(r.get("dnsmos", 0.0)),
                        "language": r.get("language", "en"),
                        "shard": os.path.basename(path)[:-4], "member": m.name[:-5]})
    return out


def stage_scan(workers: int = 48) -> pd.DataFrame:
    tars = sorted(p for p in os.listdir(RAW_DIR) if p.endswith(".tar")
                  and os.path.exists(os.path.join(RAW_DIR, p + ".done")))
    paths = [os.path.join(RAW_DIR, p) for p in tars]
    print(f"[scan] {len(paths)} complete shards", flush=True)
    with mp.Pool(min(workers, len(paths))) as pool:
        recs = pool.map(_scan_tar, paths)
    df = pd.DataFrame([r for chunk in recs for r in chunk])
    df["n_chars"] = df.text.str.len()
    os.makedirs(PROC_DIR, exist_ok=True)
    df.to_parquet(os.path.join(PROC_DIR, "meta.parquet"))
    print(f"[scan] {len(df)} clips, {df.duration.sum()/3600:.1f} h, "
          f"{df.speaker.nunique()} speakers", flush=True)
    return df


# ---------------------------------------------------------------- stage: select
def stage_select() -> Tuple[pd.DataFrame, List[Dict]]:
    df = pd.read_parquet(os.path.join(PROC_DIR, "meta.parquet"))
    df = df[(df.language == "en") & (df.n_chars >= 8)].copy()
    rng = np.random.default_rng(SELECT_RNG)

    sizes = df.groupby("speaker").size()
    prompt_spk = set(df[df.duration.between(*PROMPT_DUR)].speaker)
    tgt_spk = set(df[df.duration.between(*EVAL_TGT_DUR)].speaker)
    eligible = [s for s in sorted(prompt_spk & tgt_spk) if sizes[s] >= 3]
    order = rng.permutation(len(eligible))
    heldout = [eligible[i] for i in order[:N_HELDOUT_SPK]]

    # training pool: held-in speakers, 4–18 s, speaker-stratified round-robin to 2,000 h
    pool = df[(~df.speaker.isin(heldout)) & df.duration.between(*TRAIN_DUR)].copy()
    spk_list = sorted(pool.speaker.unique())
    spk_order = [spk_list[i] for i in rng.permutation(len(spk_list))]
    clips_by_spk = {s: g.sort_values("id").id.tolist() for s, g in pool.groupby("speaker")}
    dur_by_id = dict(zip(pool.id, pool.duration))
    chosen: List[str] = []
    total = 0.0
    round_i = 0
    while total < TRAIN_HOURS * 3600:
        added = 0
        for s in spk_order:
            cl = clips_by_spk[s]
            if round_i < len(cl):
                cid = cl[round_i]
                chosen.append(cid)
                total += dur_by_id[cid]
                added += 1
                if total >= TRAIN_HOURS * 3600:
                    break
        if added == 0:
            break
        round_i += 1
    train_ids = set(chosen)
    train_rows = pool[pool.id.isin(train_ids)]
    sec_per_char = float(np.median((train_rows.duration / train_rows.n_chars)
                                   .values[:10000]))

    # eval_zs: cross-sentence zero-shot items from held-out speakers (LOG.md P0-2).
    # The *generated* length is chars(target_text)·sec_per_char·12.5 frames, so the
    # grid.json eval window (target_seconds 4–15 s) is applied to that predicted length
    # as well as to the source clip's duration — filtering only the clip would admit
    # low-character-density clips whose synthesis is far shorter than 4 s.
    ho = df[df.speaker.isin(heldout)].sort_values("id").copy()
    ho["pred_seconds"] = ho.n_chars * sec_per_char
    cand = {s: (g[g.duration.between(*PROMPT_DUR)].id.tolist(),
                g[g.duration.between(*EVAL_TGT_DUR)
                  & g.pred_seconds.between(*EVAL_TGT_DUR)].id.tolist())
            for s, g in ho.groupby("speaker")}
    items: List[Dict] = []
    for k in range(0, 8):                          # k-th prompt of each speaker, in rounds
        for spk in heldout:
            if len(items) >= N_EVAL:
                break
            prompts, targets = cand[spk]
            if k >= len(prompts):
                continue
            p = prompts[k]
            others = [t for t in targets if t != p]
            if not others:
                continue
            items.append({"item": f"it{len(items):04d}", "speaker": spk,
                          "prompt_id": p, "target_id": others[k % len(others)]})
        if len(items) >= N_EVAL:
            break
    items = items[:N_EVAL]

    # val: 2,000 held-in clips NOT used for training, spread over as many training
    # speakers as possible (one clip per speaker per pass, speakers in id order) — the
    # id-sorted head of the leftover pool would concentrate val in a handful of speakers
    # and make it a poor instrument for LR selection and divergence detection.
    leftover = pool[~pool.id.isin(train_ids) & pool.speaker.isin(set(train_rows.speaker))]
    by_spk = {s: g.sort_values("id").id.tolist() for s, g in leftover.groupby("speaker")}
    val_ids: set = set()
    for k in range(0, 1 + max((len(v) for v in by_spk.values()), default=0)):
        for s in sorted(by_spk):
            if len(val_ids) >= N_VAL:
                break
            if k < len(by_spk[s]):
                val_ids.add(by_spk[s][k])
        if len(val_ids) >= N_VAL:
            break

    split = pd.Series("unused", index=df.id.values)
    split[list(train_ids)] = "train"
    split[list(val_ids)] = "val"
    split[[i["prompt_id"] for i in items]] = "eval_prompt"
    for i in items:                                # a target may coincide with a prompt slot
        if split[i["target_id"]] == "unused":
            split[i["target_id"]] = "eval_target"
    df["split"] = df.id.map(split)
    keep = df[df.split != "unused"].copy().reset_index(drop=True)

    train = keep[keep.split == "train"]
    chars_by_id = dict(zip(df.id, df.n_chars))
    _pred_s = [chars_by_id[i["target_id"]] * sec_per_char for i in items]
    stats = {"train_clips": int(len(train)), "train_hours": float(train.duration.sum() / 3600),
             "train_speakers": int(train.speaker.nunique()),
             "val_clips": int((keep.split == "val").sum()),
             "val_speakers": int(keep[keep.split == "val"].speaker.nunique()),
             "eval_items": len(items), "heldout_speakers": len(heldout),
             "eval_item_speakers": len({i["speaker"] for i in items}),
             "eval_pred_seconds_min": float(min(_pred_s)) if _pred_s else None,
             "eval_pred_seconds_max": float(max(_pred_s)) if _pred_s else None,
             "sec_per_char": sec_per_char,
             "sec_per_char_n": int(min(10000, len(train))),
             "total_speakers_available": int(df.speaker.nunique()),
             "shards_used": int(df.shard.nunique()),
             "min_speakers_ok": bool(train.speaker.nunique() >= MIN_SPEAKERS)}
    os.makedirs(PROC_DIR, exist_ok=True)
    keep.to_parquet(os.path.join(PROC_DIR, "index.parquet"))
    text_by_id = dict(zip(df.id, df.text))
    for it in items:
        it["prompt_text"] = text_by_id[it["prompt_id"]]
        it["target_text"] = text_by_id[it["target_id"]]
    with open(os.path.join(PROC_DIR, "eval_zs.json"), "w") as fh:
        json.dump(items, fh, indent=1)
    with open(os.path.join(PROC_DIR, "dataset.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    with open(os.path.join(PROC_DIR, "heldout_speakers.json"), "w") as fh:
        json.dump(heldout, fh, indent=1)
    print("[select]", json.dumps(stats, indent=1), flush=True)
    return keep, items


# ------------------------------------------------------------- stage: phonemize
_BACKEND = None


def _phon_init():
    global _BACKEND
    from phonemizer.backend import EspeakBackend
    _BACKEND = EspeakBackend("en-us", with_stress=True, preserve_punctuation=True,
                             punctuation_marks=";:,.!?¡¿—…\"«»“”()", language_switch="remove-flags")


def _phon_chunk(texts: List[str]) -> List[List[str]]:
    from phonemizer.separator import Separator
    sep = Separator(phone="|", word=" ", syllable="")
    out = _BACKEND.phonemize(texts, separator=sep, strip=True, njobs=1)
    if len(out) != len(texts):
        # espeak can split or drop an utterance (e.g. exotic unicode); redo 1:1 so the
        # phoneme rows stay aligned with the clip rows. Counted and logged, never silent.
        out = []
        for t in texts:
            o = _BACKEND.phonemize([t], separator=sep, strip=True, njobs=1)
            out.append(" ".join(o) if len(o) != 1 else o[0])
    import re
    # espeak glues preserved punctuation onto the adjacent phone ("t." , "...b"), which
    # would fragment 4 % of the token mass into hundreds of near-duplicate symbols.
    # Split each phone into punctuation runs and phone runs instead.
    splitter = re.compile(r"[;:,.!?¡¿—…\"«»“”()]+|[^;:,.!?¡¿—…\"«»“”()]+")
    res = []
    for s in out:
        toks: List[str] = []
        for wi, word in enumerate(s.split(" ")):
            if not word:
                continue
            if wi:
                toks.append("<sp>")
            for p in word.split("|"):
                if p:
                    toks.extend(splitter.findall(p))
        res.append(toks)
    return res


def stage_phonemize(workers: int = 64) -> None:
    idx = pd.read_parquet(os.path.join(PROC_DIR, "index.parquet"))
    texts = idx.text.tolist()
    chunks = [texts[i:i + 256] for i in range(0, len(texts), 256)]
    t0 = time.time()
    with mp.Pool(workers, initializer=_phon_init) as pool:
        outs = []
        for k, r in enumerate(pool.imap(_phon_chunk, chunks, chunksize=1)):
            if len(r) != len(chunks[k]):
                raise RuntimeError(f"chunk {k}: {len(r)} phonemizations for {len(chunks[k])} texts")
            outs.extend(r)
            if k % 200 == 0:
                print(f"[phonemize] {k}/{len(chunks)} chunks {time.time()-t0:.0f}s", flush=True)
    assert len(outs) == len(idx), f"{len(outs)} != {len(idx)}"

    train_mask = (idx.split == "train").values
    vocab_syms = sorted({p for toks, tr in zip(outs, train_mask) if tr for p in toks})
    vocab = {"<pad>": 0, "<unk>": 1}
    for s in vocab_syms:
        vocab[s] = len(vocab)
    oov = 0
    flat, offsets, lengths = [], [], []
    for toks in outs:
        ids = []
        for p in toks:
            if p not in vocab:
                oov += 1
            ids.append(vocab.get(p, 1))
        offsets.append(len(flat))
        lengths.append(len(ids))
        flat.extend(ids)
    arr = np.asarray(flat, dtype=np.int16)
    np.save(os.path.join(PROC_DIR, "phones.npy"), arr)
    idx["ph_offset"] = offsets
    idx["n_phones"] = lengths
    idx.to_parquet(os.path.join(PROC_DIR, "index.parquet"))
    with open(os.path.join(PROC_DIR, "phone_vocab.json"), "w") as fh:
        json.dump({"vocab": vocab, "oov_tokens_outside_train": oov}, fh, indent=1)
    print(f"[phonemize] vocab {len(vocab)} symbols, {oov} OOV occurrences outside train, "
          f"{len(arr)} phoneme tokens, max_len {max(lengths)}", flush=True)


# ---------------------------------------------------------------- stage: encode
def _peak_normalize(x: np.ndarray) -> np.ndarray:
    peak = float(np.abs(x).max())
    if peak < 1e-6:
        return x.astype(np.float32)
    return (x * (10.0 ** (PEAK_DBFS / 20.0) / peak)).astype(np.float32)


def _load_mimi(device):
    import torch
    from transformers import MimiModel
    m = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _encode_batch(mimi, wavs: List[np.ndarray], device) -> List[np.ndarray]:
    import torch
    lens = [len(w) for w in wavs]
    L = max(lens)
    L = ((L + FRAME_SAMPLES - 1) // FRAME_SAMPLES) * FRAME_SAMPLES
    buf = np.zeros((len(wavs), 1, L), dtype=np.float32)
    for i, w in enumerate(wavs):
        buf[i, 0, :len(w)] = w
    with torch.no_grad():
        codes = mimi.encode(torch.from_numpy(buf).to(device), num_quantizers=N_LEVELS).audio_codes
    codes = codes.to(torch.int16).cpu().numpy()          # [B, 8, frames]
    return [codes[i, :, :max(1, lens[i] // FRAME_SAMPLES)].T.copy() for i in range(len(wavs))]


_WORKER: Dict[str, object] = {}


def _encode_init(gpu_queue) -> None:
    """One GPU and one Mimi instance per worker PROCESS (not per job): otherwise a worker
    that outlives its first job opens a second context on another GPU and several
    processes pile onto the same device (observed OOM, LOG.md D-003)."""
    import torch
    gpu = gpu_queue.get()
    torch.cuda.set_device(gpu)
    _WORKER["gpu"] = gpu
    _WORKER["device"] = f"cuda:{gpu}"
    _WORKER["mimi"] = _load_mimi(f"cuda:{gpu}")


def _encode_worker(args) -> Tuple[str, int]:
    shard, want = args
    import torch
    import soundfile as sf
    from concurrent.futures import ThreadPoolExecutor
    gpu = _WORKER["gpu"]
    device = _WORKER["device"]
    mimi = _WORKER["mimi"]
    want = dict(want)                                    # member -> id
    tar_path = os.path.join(RAW_DIR, shard + ".tar")
    rows, chunks = [], []
    pool = ThreadPoolExecutor(16)

    def decode(payload):
        cid, raw = payload
        x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        return cid, _peak_normalize(x)

    buf: List[Tuple[str, np.ndarray]] = []
    n_frames_total = 0

    def flush():
        nonlocal buf, n_frames_total
        if not buf:
            return
        buf.sort(key=lambda t: len(t[1]))
        for i in range(0, len(buf), ENC_GROUP):
            grp = buf[i:i + ENC_GROUP]
            toks = _encode_batch(mimi, [w for _, w in grp], device)
            for (cid, _), tk in zip(grp, toks):
                rows.append({"id": cid, "tok_offset": n_frames_total, "n_frames": len(tk)})
                chunks.append(tk)
                n_frames_total += len(tk)
        buf = []

    with tarfile.open(tar_path) as tf:
        pending = []
        for m in tf:
            base = m.name[:-4] if m.name.endswith(".mp3") else None
            if base is None or base not in want:
                continue
            pending.append((want[base], tf.extractfile(m).read()))
            if len(pending) >= 256:
                buf.extend(pool.map(decode, pending))
                pending = []
                if len(buf) >= 512:
                    flush()
        if pending:
            buf.extend(pool.map(decode, pending))
        flush()
    pool.shutdown()
    tokens = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, N_LEVELS), np.int16)
    np.save(os.path.join(PROC_DIR, f"tokens_{shard}.npy"), tokens)
    pd.DataFrame(rows).to_parquet(os.path.join(PROC_DIR, f"tokidx_{shard}.parquet"))
    del chunks, tokens
    torch.cuda.empty_cache()
    return shard, len(rows)


def stage_encode(gpus: int = 8) -> None:
    idx = pd.read_parquet(os.path.join(PROC_DIR, "index.parquet"))
    jobs = []
    for shard, g in idx.groupby("shard"):
        if os.path.exists(os.path.join(PROC_DIR, f"tokidx_{shard}.parquet")):
            continue
        jobs.append((shard, list(zip(g.member, g.id))))
    print(f"[encode] {len(jobs)} shards to encode on {gpus} GPUs", flush=True)
    if jobs:
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        for i in range(gpus):
            q.put(i)
        with ctx.Pool(gpus, initializer=_encode_init, initargs=(q,)) as pool:
            for shard, n in pool.imap_unordered(_encode_worker, jobs):
                print(f"[encode] {shard}: {n} clips", flush=True)
    parts = [pd.read_parquet(os.path.join(PROC_DIR, f"tokidx_{s}.parquet"))
             for s in sorted(idx.shard.unique())
             if os.path.exists(os.path.join(PROC_DIR, f"tokidx_{s}.parquet"))]
    tok = pd.concat(parts, ignore_index=True)
    idx = idx.drop(columns=[c for c in ("tok_offset", "n_frames") if c in idx.columns])
    idx = idx.merge(tok, on="id", how="inner")
    idx.to_parquet(os.path.join(PROC_DIR, "index.parquet"))
    print(f"[encode] index rows with tokens: {len(idx)} "
          f"({idx.n_frames.sum()/12.5/3600:.1f} h of tokens)", flush=True)


# ------------------------------------------------------- stage: eval reference audio
def _eval_audio_shard(args) -> int:
    shard, want = args
    import soundfile as sf
    out_dir = os.path.join(PROC_DIR, "eval_audio")
    want = dict(want)
    n = 0
    with tarfile.open(os.path.join(RAW_DIR, shard + ".tar")) as tf:
        for m in tf:
            if not m.name.endswith(".mp3"):
                continue
            base = m.name[:-4]
            if base not in want:
                continue
            x, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
            if x.ndim > 1:
                x = x.mean(1)
            if sr != SR:
                import librosa
                x = librosa.resample(x, orig_sr=sr, target_sr=SR)
            sf.write(os.path.join(out_dir, want[base] + ".flac"), _peak_normalize(x), SR,
                     format="FLAC")
            n += 1
    return n


def stage_eval_audio(workers: int = 32) -> None:
    """ORIGINAL (non-codec) waveforms for the eval prompts and targets: SIM-o reference
    (grid.json eval_models.speaker_sim) and gate G0(c) ground-truth harness."""
    meta = pd.read_parquet(os.path.join(PROC_DIR, "meta.parquet"))
    with open(os.path.join(PROC_DIR, "eval_zs.json")) as fh:
        items = json.load(fh)
    need = {i["prompt_id"] for i in items} | {i["target_id"] for i in items}
    sub = meta[meta.id.isin(need)]
    os.makedirs(os.path.join(PROC_DIR, "eval_audio"), exist_ok=True)
    jobs = [(s, list(zip(g.member, g.id))) for s, g in sub.groupby("shard")]
    with mp.Pool(min(workers, len(jobs))) as pool:
        tot = sum(pool.map(_eval_audio_shard, jobs))
    print(f"[eval_audio] wrote {tot}/{len(need)} reference clips", flush=True)


# ------------------------------------------------- stage: eval curation (gate G0c)
GT_WER_MAX = 0.25       # ground-truth-transcribability filter, see LOG.md D-005


def _encode_ids(ids: List[str], meta: pd.DataFrame, shard_name: str, gpu: int = 0) -> pd.DataFrame:
    """Mimi-encode an arbitrary id list into one pseudo-shard (used for late eval additions)."""
    import soundfile as sf
    import torch
    sub = meta[meta.id.isin(ids)]
    mimi = _load_mimi(f"cuda:{gpu}")
    rows, chunks, off = [], [], 0
    for shard, g in sub.groupby("shard"):
        want = dict(zip(g.member, g.id))
        with tarfile.open(os.path.join(RAW_DIR, shard + ".tar")) as tf:
            buf = []
            for m in tf:
                if not m.name.endswith(".mp3") or m.name[:-4] not in want:
                    continue
                x, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                if sr != SR:
                    import librosa
                    x = librosa.resample(x, orig_sr=sr, target_sr=SR)
                buf.append((want[m.name[:-4]], _peak_normalize(x)))
                if len(buf) >= ENC_GROUP:
                    toks = _encode_batch(mimi, [w for _, w in buf], f"cuda:{gpu}")
                    for (cid, _), tk in zip(buf, toks):
                        rows.append({"id": cid, "tok_offset": off, "n_frames": len(tk)})
                        chunks.append(tk)
                        off += len(tk)
                    buf = []
            if buf:
                toks = _encode_batch(mimi, [w for _, w in buf], f"cuda:{gpu}")
                for (cid, _), tk in zip(buf, toks):
                    rows.append({"id": cid, "tok_offset": off, "n_frames": len(tk)})
                    chunks.append(tk)
                    off += len(tk)
    arr = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, N_LEVELS), np.int16)
    np.save(os.path.join(PROC_DIR, f"tokens_{shard_name}.npy"), arr)
    return pd.DataFrame(rows)


def stage_recurate_eval(gpu: int = 0, workers: int = 48) -> Dict:
    """Gate G0(c) repair: keep only eval items whose GROUND-TRUTH audio is transcribable.

    Emilia's EN split contains a small fraction of mislabelled non-English clips (Japanese
    audio with romaji transcripts) and clips whose transcript does not match the audio;
    on those the reference text measures nothing, so WER on generated speech is noise.
    Both the prompt clip (its transcript is part of the visible conditioning) and the
    target clip must reach ground-truth WER ≤ GT_WER_MAX under the frozen eval ASR.
    Model- and T-independent, so it cannot bias H-D2/H-D3."""
    import soundfile as sf
    meta = pd.read_parquet(os.path.join(PROC_DIR, "meta.parquet"))
    idx = pd.read_parquet(os.path.join(PROC_DIR, "index.parquet"))
    with open(os.path.join(PROC_DIR, "heldout_speakers.json")) as fh:
        heldout = json.load(fh)
    with open(os.path.join(PROC_DIR, "dataset.json")) as fh:
        stats = json.load(fh)
    spc = stats["sec_per_char"]

    ho = meta[meta.speaker.isin(heldout) & (meta.n_chars >= 8)].copy()
    ho["pred_seconds"] = ho.n_chars * spc
    is_p = ho.duration.between(*PROMPT_DUR)
    is_t = ho.duration.between(*EVAL_TGT_DUR) & ho.pred_seconds.between(*EVAL_TGT_DUR)
    cand = ho[is_p | is_t].sort_values("id")
    print(f"[recurate] {len(cand)} candidate clips from {cand.speaker.nunique()} held-out "
          f"speakers ({int(is_p.sum())} prompt-window, {int(is_t.sum())} target-window)",
          flush=True)

    os.makedirs(os.path.join(PROC_DIR, "eval_audio"), exist_ok=True)
    todo = [c for c in cand.id if not os.path.exists(
        os.path.join(PROC_DIR, "eval_audio", c + ".flac"))]
    if todo:
        sub = cand[cand.id.isin(todo)]
        jobs = [(s, list(zip(g.member, g.id))) for s, g in sub.groupby("shard")]
        with mp.Pool(min(workers, len(jobs))) as pool:
            print(f"[recurate] extracted {sum(pool.map(_eval_audio_shard, jobs))} clips",
                  flush=True)

    # ground-truth WER for every candidate under the frozen eval ASR
    gt_path = os.path.join(PROC_DIR, "eval_gtwer.json")
    gt = json.load(open(gt_path)) if os.path.exists(gt_path) else {}
    need = [c for c in cand.id if c not in gt]
    if need:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import jiwer
        from evaluate import Scorer
        sc = Scorer(f"cuda:{gpu}", use_utmos=False)
        text = dict(zip(cand.id, cand.text))
        for s in range(0, len(need), 200):
            batch = need[s:s + 200]
            wavs = [sf.read(os.path.join(PROC_DIR, "eval_audio", c + ".flac"),
                            dtype="float32")[0] for c in batch]
            hyps = sc.transcribe(wavs)
            for c, h in zip(batch, hyps):
                r, hn = sc.norm(text[c]), sc.norm(h)
                gt[c] = float(jiwer.wer(r, hn or " ")) if r else 1.0
            json.dump(gt, open(gt_path, "w"))      # incremental: resumable
            print(f"[recurate] GT-WER {s + len(batch)}/{len(need)}", flush=True)
    ok = {c for c in cand.id if gt.get(c, 1.0) <= GT_WER_MAX}
    print(f"[recurate] transcribable candidates: {len(ok)}/{len(cand)} "
          f"({100*len(ok)/len(cand):.1f}%) at GT-WER ≤ {GT_WER_MAX}", flush=True)

    good = cand[cand.id.isin(ok)]
    per = {s: (g[g.duration.between(*PROMPT_DUR)].id.tolist(),
               g[g.duration.between(*EVAL_TGT_DUR)
                 & g.pred_seconds.between(*EVAL_TGT_DUR)].id.tolist())
           for s, g in good.groupby("speaker")}
    items: List[Dict] = []
    for k in range(0, 12):
        for spk in heldout:                       # same deterministic round-robin as select
            if len(items) >= N_EVAL or spk not in per:
                continue
            prompts, targets = per[spk]
            if k >= len(prompts):
                continue
            p = prompts[k]
            others = [t for t in targets if t != p]
            if not others:
                continue
            items.append({"item": f"it{len(items):04d}", "speaker": spk,
                          "prompt_id": p, "target_id": others[k % len(others)]})
        if len(items) >= N_EVAL:
            break
    items = items[:N_EVAL]
    text_by_id = dict(zip(meta.id, meta.text))
    for it in items:
        it["prompt_text"] = text_by_id[it["prompt_id"]]
        it["target_text"] = text_by_id[it["target_id"]]
        it["gt_wer_prompt"] = gt[it["prompt_id"]]
        it["gt_wer_target"] = gt[it["target_id"]]
    need_ids = {i["prompt_id"] for i in items} | {i["target_id"] for i in items}
    print(f"[recurate] {len(items)} items over {len({i['speaker'] for i in items})} speakers, "
          f"{len(need_ids)} distinct clips", flush=True)

    # rows for eval clips that are not in the index yet: phonemise (frozen vocab) + encode
    missing = sorted(need_ids - set(idx.id))
    if missing:
        with open(os.path.join(PROC_DIR, "phone_vocab.json")) as fh:
            vocab = json.load(fh)["vocab"]
        _phon_init()
        rows = meta[meta.id.isin(missing)].copy()
        toks = _phon_chunk(rows.text.tolist())
        phones = np.load(os.path.join(PROC_DIR, "phones.npy"))
        flat, offs, lens, oov = [], [], [], 0
        for t in toks:
            ids = []
            for p in t:
                if p not in vocab:
                    oov += 1
                ids.append(vocab.get(p, 1))
            offs.append(len(phones) + len(flat))
            lens.append(len(ids))
            flat.extend(ids)
        np.save(os.path.join(PROC_DIR, "phones.npy"),
                np.concatenate([phones, np.asarray(flat, dtype=np.int16)]))
        rows["ph_offset"], rows["n_phones"] = offs, lens
        tok_rows = _encode_ids(missing, meta, "evalextra", gpu)
        rows = rows.merge(tok_rows, on="id", how="inner")
        rows["shard"] = "evalextra"
        rows["split"] = "eval_extra"
        idx = pd.concat([idx, rows[idx.columns.intersection(rows.columns)]], ignore_index=True)
        print(f"[recurate] added {len(rows)} eval rows ({oov} OOV phoneme occurrences)",
              flush=True)

    prompts = {i["prompt_id"] for i in items}
    targets = {i["target_id"] for i in items}
    split = idx.split.copy()
    split[idx.split.isin(("eval_prompt", "eval_target", "eval_extra"))] = "unused"
    split[idx.id.isin(prompts)] = "eval_prompt"
    split[idx.id.isin(targets) & ~idx.id.isin(prompts)] = "eval_target"
    idx["split"] = split
    idx.to_parquet(os.path.join(PROC_DIR, "index.parquet"))
    json.dump(items, open(os.path.join(PROC_DIR, "eval_zs.json"), "w"), indent=1)
    stats.update({"eval_items": len(items),
                  "eval_item_speakers": len({i["speaker"] for i in items}),
                  "eval_curation": {"gt_wer_max": GT_WER_MAX, "candidates": int(len(cand)),
                                    "transcribable": len(ok),
                                    "gt_wer_mean_selected": float(np.mean(
                                        [i["gt_wer_target"] for i in items]))}})
    json.dump(stats, open(os.path.join(PROC_DIR, "dataset.json"), "w"), indent=1)
    print(f"[recurate] mean GT-WER of selected targets: "
          f"{stats['eval_curation']['gt_wer_mean_selected']:.4f}", flush=True)
    return stats


# ------------------------------------------------------------------ loader side
class TokenStore:
    """Memory-mapped access to the per-shard token arrays and the phoneme array."""

    def __init__(self, proc_dir: str = PROC_DIR):
        self.proc_dir = proc_dir
        self.index = pd.read_parquet(os.path.join(proc_dir, "index.parquet"))
        self.phones = np.load(os.path.join(proc_dir, "phones.npy"), mmap_mode="r")
        with open(os.path.join(proc_dir, "phone_vocab.json")) as fh:
            self.vocab = json.load(fh)["vocab"]
        with open(os.path.join(proc_dir, "dataset.json")) as fh:
            self.dataset = json.load(fh)
        self._tok: Dict[str, np.ndarray] = {}
        self.pos = {cid: i for i, cid in enumerate(self.index.id.values)}
        self.a_shard = self.index.shard.values
        self.a_tok_off = self.index.tok_offset.values.astype(np.int64)
        self.a_n_frames = self.index.n_frames.values.astype(np.int64)
        self.a_ph_off = self.index.ph_offset.values.astype(np.int64)
        self.a_n_phones = self.index.n_phones.values.astype(np.int64)

    def tokens_of(self, shard: str) -> np.ndarray:
        if shard not in self._tok:
            self._tok[shard] = np.load(os.path.join(self.proc_dir, f"tokens_{shard}.npy"),
                                       mmap_mode="r")
        return self._tok[shard]

    def get_pos(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        """(phoneme ids, token grid [frames, 8]) for row ``i`` of the index."""
        o, n = self.a_tok_off[i], self.a_n_frames[i]
        tk = np.asarray(self.tokens_of(self.a_shard[i])[o:o + n])
        po, pn = self.a_ph_off[i], self.a_n_phones[i]
        ph = np.asarray(self.phones[po:po + pn])
        return ph.astype(np.int64), tk.astype(np.int64)

    def get(self, cid: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_pos(self.pos[cid])

    def split(self, name: str) -> pd.DataFrame:
        return self.index[self.index.split == name]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["scan", "select", "phonemize", "encode",
                                     "eval_audio", "recurate_eval", "all"])
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args()
    if a.stage in ("scan", "all"):
        stage_scan(a.workers)
    if a.stage in ("select", "all"):
        stage_select()
    if a.stage in ("phonemize", "all"):
        stage_phonemize(a.workers)
    if a.stage in ("eval_audio", "all"):
        stage_eval_audio(a.workers)
    if a.stage in ("encode", "all"):
        stage_encode(a.gpus)
    if a.stage in ("recurate_eval", "all"):
        stage_recurate_eval(gpu=0, workers=a.workers)
