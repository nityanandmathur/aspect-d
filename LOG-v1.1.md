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
