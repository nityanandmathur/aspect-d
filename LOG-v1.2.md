# ASPECT-D v1.2 — decision log

Append-only. Companion to `LOG.md` (v1.0) and `LOG-v1.1.md` (v1.1). Verdict
discipline per `task-v2.md` §1 applies to every entry from Program S onward.

## 2026-08-08 14:10 — PHASE M: paper finalized against task-v2.md §2/§3

**Four blocking edits (DECISION.md "Not ready to submit") executed.**

1. *Cross-metric T\* saturation claim removed and its macros neutralised.*
   `src/paper.py` no longer emits `Ntstar<METRIC>`; it emits `NtstarRaw<METRIC>`,
   whose name states the scale, and the abstract/results/caption uses are gone.
   `Nlatencysaving` (= 100·(1−T\*_SIM/16) = 50 %, pure artifact of the retracted
   T\*_SIM = 8) is no longer generated.
2. *Interior-optimum "deep enough" rule removed.* `Nbestdepths` now marks each
   budget `(interior)` or `(censored)` by testing whether the argmin equals the
   deepest shape tested; B and C are censored, so the paper states the optimum is
   interior only in the smallest budget.
3. *Depth-first allocation rule removed* — H-E2 refutes it (5.9 % measured /
   8.8 % fitted against an 80 % bar). Replaced with the honest statement that a
   latency budget constrains only the product $T\,d$ and these data do not
   resolve the split in either direction.
4. *`step_curves` regenerated without raw-scale per-metric T\* markers.*
   `src/figures.py` no longer draws them, with the reason inline.

**Immutability choice, logged as a deviation-avoidance.** §3.5 forbids editing
`artifacts/`. Rather than overwrite the v1.0 figure, the corrected figure is
written to `artifacts-v1.2/figures/step_curves.{svg,pdf}` and the paper points
there. `artifacts/figures/step_curves.svg` still carries its two dashed markers
and is untouched, as the v1.0 record should be.

**§2 approved wording applied**, with every number a generated macro (97 total;
hand-typed numbers remain forbidden). The abstract now carries, in order: the
pre-registered detection statistic Δτ = +0.1102 CI [0.0921, 0.1312]; the
magnitude claim (0.952 absolute WER vs 0.161 absolute identity error, 5.9×);
the rate-honesty sentence (fraction of total gain at T=8: 94.6 % WER vs 93.7 %
identity, identity marginally later); and the four-budget persistence
Δτ = +0.1095 CI [0.0951, 0.1241]. The fraction-at-T=8 macros are computed
**per run then averaged**, which is what reproduces §2's stated 94.6/93.7.

**v1.1 strengtheners integrated within the 4-page law:** E4 and E6 and E5 in one
compressed passage of the H-D2 result, E1's extended-T fact with the scale named,
E3 allocation as one main-text sentence plus an appendix table carrying its
honest SIM/UTMOS cost, and the speaking-rate limitation sentence. The
significance-based 2× T\* contrast is in the appendix with its precision-floor
caveat and an explicit note that the raw-scale T\* must never be compared across
metrics.

**Verified:** compiles with no errors; References begin on page 5, so main text
is exactly 4 pages; anonymised; the §3.6 retired-phrase grep returns zero in
`paper/` excluding `main-v1-frozen.tex`, which is the deliberately preserved v1.0
paper and necessarily still contains the old title.

**Deviation to flag (pre-existing, not introduced here).** Before `task-v2.md`
existed, a cross-document audit found retracted claims live in `results.html`,
`index.html` and `protocol.html`, and I corrected them (19 insertions,
13 deletions). §3.5 now lists those files as frozen record. The corrections are
retained rather than reverted, because reverting would reinstate withdrawn claims
in published pages; they are additive outcome annotations and errata, not
rewrites of any measured value. No further edits to those files were made in
Phase M, and none will be.
