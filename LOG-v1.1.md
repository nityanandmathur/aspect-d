# ASPECT-D v1.1 — append-only extension log

Companion to `LOG.md` (v1.0, frozen). Every v1.1 gate with measured values,
every cut, every fallback, every deviation. v1.0 artifacts are immutable
(task-v1.md §0.1).

---

## 2026-08-06 22:15 UTC — git hygiene (directive §0.2)

- Tag `v1.0-submission-candidate` created at `ed09b87` (the v1.0 tree with the
  declared S1 outcome, 45 runs, compiled paper) and pushed.
- Branch `v1.1-extensions` created; all v1.1 work happens here.
- `paper/main.tex` copied to `paper/main-v1-frozen.tex` — never edited again.
- New output roots: `artifacts-v1.1/`, `logs-v1.1/`, `runs-v1.1/`,
  `LOG-v1.1.md`, `state-v1.json`, `RESULTS-FEED.md`.

## 2026-08-06 22:18 UTC — GATE E-AUDIT — **PASS**

| check | measured | verdict |
|---|---|---|
| loadable final checkpoints, 15 configs × seeds {0,1,2} | **45 / 45** at step 30000 | PASS |
| missing or corrupt | **none** | PASS |
| smoke: C3 seed 0, T=16 re-synthesis vs frozen v1.0 grids | **token agreement 1.0000** (bit-identical) | PASS |

`frac_missing = 0`, so the >20% inversion clause of §1.3 does not fire and the
queue runs in its stated priority order. Full record:
`artifacts-v1.1/checkpoint_audit.json` (path, step, size, 16-hex head-hash,
final val loss and charged GPU-h per run).

### Smoke-test method note (a real trap, recorded)
The first smoke attempt compared a **5-item** batch against v1.0's **50-item**
batches and reported 35.7% token agreement, which looks like checkpoint drift
and is not. The sampler draws Gumbel noise shaped `[B, Fmax, V]` from a
generator keyed by `(batch_index, level, step)`, so batch composition is part of
the RNG state — exactly the invariance documented in LOG.md P0-3, which
guarantees reproducibility only for a *fixed* batch composition. Re-running with
the full 50-item first batch gives **bit-identical** tokens. Consequence for
E1: the extended-T runs must synthesise the first 200 items as the same four
50-item batches v1.0 used (batch indices 0–3), which is what the E1 job does;
any other item chunking would silently break comparability with the v1.0 curve.

## 2026-08-06 22:20 UTC — E0 pre-registration frozen

`PREREGISTRATION-v1.1.md` written with §9 of `task-v1.md` copied verbatim
(H-E1…H-E6), before any extension synthesis, training or scoring existed. The
G1-D sanity proxy launched at 22:16 is training-only and produces no extension
*datum* against any hypothesis; H-E4 concerns the D grid, which starts only
after G1-D is evaluated.

## 2026-08-06 23:40 — P0 defects found by adversarial review of v1.1 code

**P0-v1.1-1 — extended-T dirs scored against the canonical 400-item list.**
A *v1.0-era* incremental scoring loop (bash PID 190154, spawned during v1.0 and
never stopped) regenerates `logs/jobs/score_loop.json` every round as
`{"run": d, "T": T}` for any `synth_T*` dir lacking `scores.json` — with no
`items` key. `score_dir` then took `limit=None` and fell back to the canonical
400 items, charging the 200 items E1 never synthesised under the §10 crash
policy (wer 1.0, sim NaN, degenerate). Result: 16 `scores.json` files in
`runs/*/synth_T{24,32,64}/` recording `n_items 400, crash_rate 0.5,
wer_mean 0.5634` where the truth is `n_items 200, crash_rate 0.0,
wer_mean 0.1268`. Compounding it, `cmd_score` skips any dir that already has
the target file, so a later correct scoring would have been silently skipped.

*Impact contained:* E1's own dispatcher writes `scores_ext200.json`
(`--suffix _ext200`, explicit `"items": 200`) and those 19 files are correct —
verified `wer_mean 0.1268` == the poisoned file's `wer_mean_nondegen`, i.e. the
per-item rows were always right and only the denominator was wrong. **No v1.0
artifact was touched**: the loop only writes where `scores.json` is absent, and
the newest v1.0 `scores.json` mtime is 14:18, five hours before v1.1 began.

*Fixed:* (a) killed PIDs 190154 + 996648; (b) deleted all 16 poisoned files;
(c) `score_dir` now reads the denominator from the target dir's own
`synth.json` when `--items` is not given, and fails loud if it is missing.
Verified behaviour-identical for v1.0: all 225 v1.0 dirs record `items 400`,
all 63 E1 dirs record `items 200`. A missing `.flac` *within* the synthesised
set is still charged as a crash — the denominator is the requested count, not
the file count, so real crashes cannot be silently dropped.

**P0-v1.1-2 — H-E2 decision rule implemented as the wrong statistic.** §9 H-E2
asks whether the depth-first property holds *under each of* the fitted and
measured methods at ≥ 80 % of budgets; `iso_latency.py` instead measured
concordance *between* the methods (do they pick the same config and T). Fixed:
`depth_first_frac_{measured,fitted}` are now the decision statistic, concordance
is retained as a labelled secondary descriptive. Verdict unchanged (refuted),
margin much wider: 5.9 % / 8.8 % against an 80 % bar.

**P0-v1.1-3 — false claim in the E2 results entry.** "Never reaches the d\*
ridge" was contradicted by the analysis's own path CSV (B5 d30 and C5 d36 at
the top two budgets). Corrected in `RESULTS-FEED.md` as a new append-only
entry; the real ordering is steps-first (T maxes at L=726 ms, d reaches d\* only
at L=2002 ms, zero budgets with T<16 ∧ d≥d\*), with the T≤16 boundary caveat now
recorded in the artifact and E1's extended grid queued to test it.

## 2026-08-06 23:45 — GATE G1-D: FAIL → LR sweep

D3 val@3k 5.6520 vs B3 5.6485 / C3 5.6893. No divergence. Below C3, above B3 →
monotone-in-N check fails. Pre-registered remedy (§4-E4) running: 5-point base-LR
sweep at D3 {0.001, 0.002, 0.004, 0.008, 0.016}, 3k steps each; 0.004 reused
from the proxy. E4 adopts the argmin.

## 2026-08-07 07:30 — E5: fallback SV model fails G0(c); SV-swap cells excluded

The E5 panel showed Δτ flipping sign under `microsoft/wavlm-base-plus-sv`
(−0.5837 vs +0.1117). Before treating that as evidence about S1, the instrument
was validated against the project's own pre-registered gate **G0(c)**
(protocol §8: median same-speaker SIM-o ≥ 0.50, median cross-speaker ≤ 0.25),
run on ground-truth human audio:

| | wavlm-large (primary) | wavlm-base-plus-sv (fallback) |
|---|---|---|
| same-speaker median | 0.7005 | 0.9488 |
| cross-speaker median | 0.0338 | **0.6601** |
| G0(c) | PASS | **FAIL** |

The fallback rates two different real speakers at 0.6601 and rates our T=1 audio
(115 % WER) at 0.8494 — i.e. above its own cross-speaker floor. Its usable range
on real audio is [0.66, 0.95] and every generated output sits inside the band
where it cannot separate speakers, so `err = 1 − SIM` under it is a compressed,
uninformative scale and its τ_SIM = 1.4231 is an artifact of that compression.

**Ruling:** SV-swap cells are reported for completeness and excluded from
inference. Δτ is robust across every variation whose instrument passes G0(c):
baseline +0.1117, ASR swap +0.1292, log-amplitude +0.1262 — all same sign, all
CIs excluding 0, all overlapping the v1.0 declared interval. No pre-registered
claim changes.

`evaluate.py gt` gained `--sv-fallback` and `--out` so any candidate SIM model
can be held to the same bar before its numbers are used. This should be the
standing rule: **an instrument that fails G0(c) does not get a vote.**
