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

## 2026-08-07 08:40 — P0-v1.1-4: T* is scale-dependent; the 8-vs-32 gap retracted

`fit.saturation_T` applies "within 5 % of the T_ref value" on each metric's raw
level. 5 % of WER's T=64 level is 0.64 % of WER's range; 5 % of (1−SIM)'s is
20.22 % of its range — 32× more lenient. The published "T*_SIM=8 vs T*_WER=32,
a 4× gap" is that asymmetry, not a property of the models.

Affine-invariant check (fraction of each metric's own total T=1→64 gain, immune
to err → a+b·err): T*(WER) = T*(SIM) = **16**. Gap 1×. At T=8, 1−SIM is
marginally *later* than WER (92.63 % vs 93.92 %); on the T ≤ 16 grid the paired
difference at T=8 is −1.09 pp CI [−1.95, −0.19], excluding zero **against** the
claim.

Retracted: the cross-metric saturation comparison and all derived wording
("fit-free demonstration of S1", "strongest single piece of evidence", and the
proposal to lead the paper with it). It was never pre-registered — invented
during v1.1 — and replacing the pre-registered primary (H-D2, Δτ) with it would
have breached §0 directive 1 and §7.

Unaffected: H-E1(a) (T*_WER = 32 > 16 — single-metric, own scale, v1.0's frozen
definition); the 15 % relative WER gain from T=16→64; and S1/H-D2 on Δτ =
+0.1102 CI [0.0921, 0.1312], robust to the ASR swap and the log-amplitude
reparameterisation.

Guard added: `e1_extended.py` emits `saturation_affine_invariant` and
`T_star_scale_caveat` alongside the raw T*.

**Standing lesson (with P0-v1.1-2/3): a statistic that is not invariant to a
metric's units may not be compared ACROSS metrics.** Δτ is exposed to the same
critique in principle; it survives only because it is the frozen pre-registered
statistic evaluated under the frozen instrument, and it is reported scoped that
way rather than as a scale-free truth.

## 2026-08-07 08:10 — P0-v1.1-5: E5 panel used the wrong estimator and 500 reps

`e5_robust.py` reported Δτ as the bootstrap MEDIAN while v1.0 declares it as the
POINT difference of the fits (fit.py:404), so the "v1.0 baseline" cell was not a
reproduction (0.1117 vs the declared 0.1102). It also ran 500 replicates against
the 2 000 frozen in §5 / protocol §7.2, unlogged. Both fixed: Δτ is now the point
estimate with the bootstrap supplying only the CI (median retained as a labelled
diagnostic), and the panel was rerun at 2 000. The baseline cell now reproduces
v1.0 **exactly** — 0.1102, CI [0.0921, 0.1312] — which is the real validation
that the variant pipeline is sound.

## 2026-08-07 09:30 — P0-v1.1-6: my SV-flip explanation was wrong; Δτ's gloss is reversed

**Struck:** "τ_SIM = 1.4231 under base-plus-sv is an artifact of compression"
(07:30 feed entry; LOG entry above). τ is **affine-invariant** under M_sep — for
err → a+b·err, E absorbs a, A/B/C absorb b, exponents unchanged, weighted
objective invariant (SE floor is scale-relative). Empirically an affine remap of
wavlm-large onto base-plus-sv's range moves τ_SIM by −0.0014, 0.2 % of the +0.697
flip. The real mechanism is resolution loss plus non-monotonicity (SIM(8) >
SIM(16) in 26/45 runs), which monotone C·T^−τ cannot fit. **The exclusion rests
on G0(c) alone and is unaffected.**

**Corrected:** the T\* tolerance asymmetry is **17.9×**, not the 32× I published —
the 5 % band applies on each metric's own reported scale (SIM-o, WER), not on
1−SIM. `saturation_affine` now derives the band per metric direction. Conclusion
unchanged; independently confirmed by the affine-remap test (T\*_SIM 8 → 2 under
a transform that moves τ by 0.0014).

**Robustness ordering was inverted in my write-up:** τ is the affine-invariant
statistic, T\* is the scale-dependent one.

**New, and material to the paper's wording:** larger τ ⇒ *earlier* saturation, so
Δτ = τ_WER − τ_SIM > 0 says **WER converges sooner** (T90 15.7 vs 23.8), not that
steps keep buying intelligibility after identity saturates. The pre-registered
test is unaffected (Δτ = +0.1102 CI [0.0921, 0.1312], H-D2 passes, S1 correctly
declared); the **title/abstract gloss** is what is backwards. The defensible
practical claim is absolute-magnitude: over T=1→16 steps buy 0.9518 of WER error
against 0.1606 of identity error (5.9×), degenerate rate 40.4 % → 0.1 %.
Escalated for a wording decision rather than changed unilaterally.

## 2026-08-08 12:40 — P0-v1.1-7: retracted claims were still live across 7 files

A cross-document audit found **37 live assertions** of claims this project had
already retracted, in `paper/main.tex` (4), `paper/numbers.tex` (3),
`src/paper.py` (the generator that emits them), `results.html` (4),
`protocol.html` (4), `DECISION.md` (5), `README.md` (3), `index.html` (4) and —
worst — `extensions.html`, the retraction page itself.

**The worst item was mine.** `extensions.html` had *rebuilt* the withdrawn
cross-metric saturation claim as a "significance-based T\*_WER=32 vs T\*_SIM=16,
a 2× gap" inside its own "What actually stands" box. That number exists on no
single scale: `e1_extended.json` gives raw {wer 32, sim 8} and affine-invariant
{wer 16, sim 16}. The same page also still argued the retraction from the
superseded 0.64 %/20.22 %/32× figures after those were corrected to
0.67 %/11.96 %/17.9× in the code and the feed. **Retracting a claim in one file
is not retracting it.**

Also fixed: the paper's abstract, its headline figure caption ("SIM-o has
flattened by T=8"), and its allocation rule ("spend depth until latency binds,
then steps") — the last being the depth-first rule H-E2 refutes at 5.9 %/8.8 %
against an 80 % bar. `src/paper.py` no longer emits `Nlatencysaving` (it was
100·(1−T\*_SIM/16) = 50 %, an artifact of the retracted T\*_SIM=8); the macros in
`numbers.tex` are annotated WITHDRAWN in place rather than deleted, because
`main-v1-frozen.tex` also `\input`s them and must keep compiling.

**Two protocol deviations found and corrected in the same pass:** `e1_extended`
and `e6_analysis` had last been run at 100 and 1 000 bootstrap replicates against
the 2 000 frozen in §5. Both re-run at 2 000; τ_WER CI [0.9374, 0.9686], control
[0.8629, 0.8943], subset effect +0.0411, range effect +0.0754.

**Standing rule added: a retraction is not complete until every file that
asserted the claim is fixed, and the generator that produced it is fixed too.**
