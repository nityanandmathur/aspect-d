# ASPECT-D pre-registration addendum v1.2

Copied VERBATIM from `task-v2.md` §9, frozen **before any Program-S run, before
any Program-S synthesis, scoring or analysis**. Amends nothing in v1.0/v1.1; the
declared outcomes of both remain immutable.

Verdict discipline per `task-v2.md` §1 binds every hypothesis below: two
independent lenses (a verdict of SUPPORTED or REFUTED requires both to agree in
direction, otherwise the verdict is **DISCORDANT**), a filled rival table where
"addressed" means a number rather than a sentence, MEASURED/INTERPRETATION
separation in every feed entry, and the cooling rule before anything touches a
paper sentence.

**Frozen:** 2026-08-08, after the Gate M-DONE merge commit `55a47d43` and before
any Program-S datum existed.

---

> **ASPECT-D pre-registration addendum v1.2** — frozen before any Program-S
> run; amends nothing in v1.0/v1.1; declared outcomes remain immutable.
> Every hypothesis lists: primary lens, second lens, rivals with
> discriminating checks. Verdicts require both lenses per task-v2.md §1.
>
> - **H-S0 (codec ceiling).** Measurement: SIM_rt = mean SIM-o of
>   Mimi-roundtripped ground truth vs original prompt, 400 items.
>   Classification rule: headroom h = SIM_rt − best measured system SIM-o
>   (any budget, T=16). h ≤ 0.05 → "near-ceiling"; 0.05 < h ≤ 0.15 →
>   "moderate headroom"; h > 0.15 → "large headroom". Second lens:
>   per-item paired distribution of (roundtrip − best-system) SIM; the
>   classification must hold for the median as well as the mean. Rivals:
>   roundtrip favoring the reference recording conditions (check: roundtrip
>   of a DIFFERENT same-speaker utterance vs prompt, reported alongside);
>   scorer saturation at high similarity (check: same-speaker GT baseline
>   distance from 1.0).
> - **H-S1 (context buys identity).** Primary: paired per-run mean SIM-o
>   (fixed 3 s reference) at 9 s vs 3 s prompts; run-level bootstrap 95% CI
>   > 0 across the 15 C-budget runs. Second lens: Spearman trend of SIM-o
>   over {1.5, 3, 6, 9} within each run, ≥ 12/15 runs positive. Guardrail:
>   WER(9 s) − WER(3 s) reported; a SIM gain with > +2.0 WER points is
>   reported as a trade, not a win. Rivals: reference confound (excluded by
>   the fixed-reference protocol); duration-of-evidence artifact in the
>   scorer (check: secondary scoring vs full source utterance agrees in
>   direction); train/test prompt-length shift harming WER only (check:
>   DegenRate by arm).
> - **H-S2 (search buys identity where refinement cannot).** Primary: at
>   each matched NFE ∈ {128, 256, 512}, ECAPA-SIM(best-of-K) −
>   ECAPA-SIM(refinement) paired per-run CI > 0 in ≥ 2 of 3 NFE tiers.
>   Second lens: the same contrast under WavLM-SV *scoring on the
>   non-selected metric protocol* is directionally consistent (reported;
>   selection circularity acknowledged), plus per-item win-rate > 50% with
>   CI. Secondary (symmetry): WER(refinement) < WER(best-of-K) in ≥ 2 of 3
>   tiers. Rivals: selection-scoring circularity (excluded by ECAPA
>   scoring; ECAPA must pass the §1.7 instrument gate); best-of-K covertly
>   selecting non-degenerates (check: report DegenRate and WER of selected
>   candidates); variance-only effect (check: does K help the per-item
>   *median* or only the tail?).
> - **H-S3 (rate-matched length buys identity).** Primary: paired per-run
>   SIM-o(rate-matched) − SIM-o(median-rate) CI > 0. Second lens: per-item
>   paired median difference sign-test p < 0.05 same direction. Guardrail:
>   WER change within ±2.0 points, else "trade". Rivals: duration change
>   altering the amount of scoreable audio (check: report duration
>   distributions and SIM stratified by duration quartile); ASR length
>   sensitivity (check: WER stratified likewise).
> - **H-S4 (contrastive guidance buys identity, training-free).** Primary:
>   ∃ γ ∈ {0.5, 1.0, 2.0} with paired per-run SIM-o(γ) − SIM-o(0) CI > 0
>   and WER(γ) ≤ WER(0) + 2.0 and DegenRate(γ) ≤ 2× DegenRate(0), on the
>   full C-budget after the canary gate. Second lens: SIM gain must also
>   hold under ECAPA scoring. Rivals: guidance trading naturalness for
>   scorer-specific features (check: UTMOS(γ) reported; a SIM gain with
>   UTMOS collapse > 0.5 is flagged, not celebrated); wrong-speaker branch
>   producing degenerate negatives (check: DegenRate of the negative
>   branch alone on 20 items).
> - **Program-level statement.** These hypotheses are competing candidates
>   for one phenomenon. No single SUPPORTED verdict will be glossed as "the"
>   answer; the closing feed entry must rank all axes on the S0 ledger with
>   their measured gains and costs, and explicitly state which rivals remain
>   unexcluded. Analysis machinery per task-v1.md §5; RNGs unchanged;
>   everything not listed is exploratory and labeled so.

---

Anything not listed above is exploratory and is labeled so wherever it appears.
Statistics invented after this freeze are exploratory permanently (§1.6).
