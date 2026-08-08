# ASPECT-D v1.2 — Identity Program & Merge Runbook (task-v2.md)

You are Claude Code, resuming in the ASPECT-D repo. Current state: v1.0 declared
S1 (frozen); v1.1 extensions complete on branch `v1.1-extensions` with two open
decisions in `state-v1.json`. **Both decisions are now made by the human and
encoded here** (§2 — approved wording; §3 — merge). Your mission has two parts,
in strict order:

1. **PHASE M:** finish the paper edits and cleanly merge `v1.1-extensions`
   into `main` via a PR.
2. **PROGRAM S:** answer the reviewer's next question — *"steps buy WER; what
   buys speaker similarity?"* — as a pre-registered, multi-lens program
   (S0–S5), with the new verdict discipline of §1 so no conclusion ever again
   rests on a single statistic or a single explanation.

All v1.0 and v1.1 prime directives remain in force (task.md §0, task-v1.md §0),
including immutability of the v1.0 declared analysis, the 4-page law, the
no-self-scoop rule, compute charging conventions, and the feed. New state file:
`state-v2.json`. New log: `LOG-v1.2.md`. Same `RESULTS-FEED.md` (append-only).

---

## 1. VERDICT DISCIPLINE (new epistemic machinery — applies to every S-job)

The v1.1 postmortem: one finding was promoted then retracted (T* scale
artifact), one mechanism story was asserted then struck (compression), one
gloss was discovered backwards (rates vs magnitudes). Each error came from
committing to a single statistic or single explanation too early. The following
is therefore mandatory:

1. **Two-lens rule.** Every pre-registered hypothesis in §9 specifies TWO
   independent lenses: a primary statistic and a model-free corroboration
   (different code path, no shared fitting machinery). A verdict may be
   SUPPORTED or REFUTED only if both lenses agree in direction. If they
   disagree, the verdict is **DISCORDANT** — a first-class outcome that
   requires a diagnosis entry (which lens is measuring what, and why) and is
   never silently resolved in favor of the more exciting reading.
2. **Rival-explanations table.** Each hypothesis in §9 ships with pre-listed
   rivals (instrument artifact, scale/normalization artifact, selection or
   circularity, ceiling effect, confounded reference, distribution shift).
   A verdict entry must address every listed rival with the named
   discriminating check. "Addressed" means a number, not a sentence.
3. **Measurement/interpretation separation.** Every feed entry has a
   MEASURED block (numbers, CIs, code paths — frozen once posted) and an
   INTERPRETATION block (revisable, labeled). Paper and html edits may cite
   only MEASURED blocks that have survived rule 4.
4. **Cooling rule.** Any interpretation that would change a paper sentence,
   a title, or a headline figure sits for one full adversarial verification
   pass (independent recomputation from item-level rows through a separate
   script, plus the standard invariance checks: affine-invariance where
   applicable, censoring/boundary checks on any argmin/argmax, monotonicity
   checks on any "saturation" claim, instrument validity for any scorer).
   Only then may it be marked STABLE and acted on. The T*, d*, and
   base-plus-sv incidents are the reference cases; their checks are now
   defaults, not discoveries.
5. **Ledger-relative interpretation.** No single-axis claim about identity
   ("X buys SIM") may be posted before S0's ledger exists, and every such
   claim must state the axis's gain RELATIVE to the ledger (including the
   codec ceiling headroom). A +0.01 SIM gain is not "buys identity" when the
   ceiling headroom is 0.15 and N buys 0.06.
6. **No statistic promotion.** Statistics invented mid-flight are exploratory
   forever (v1.1 directive, restated because it was nearly violated once).
7. **Instrument gate.** Any new scoring or selection model must pass G0(c)
   validation on ground truth (same-speaker median ≥ 0.50, cross-speaker
   median ≤ 0.25 for SV models; WER ≤ 5% for ASR) before its numbers are used
   for anything. base-plus-sv remains excluded.
8. **Selection ≠ scoring.** Wherever a model selects a candidate (S2),
   a different, instrument-gated model scores it. No exceptions.

---

## 2. APPROVED WORDING (human decision — apply verbatim in Phase M)

**Title:** *Test-Time Refinement Moves Intelligibility Six Times More Than
Identity in Masked-Diffusion TTS*

**Abstract/intro gloss — required content, in this order:**
(a) the pre-registered detection statistic: Δτ = +0.1102, 95% CI
[0.0921, 0.1312], excludes zero — test-time scaling is metric-selective;
(b) the magnitude claim that carries the paper: over T = 1→16, refinement
moves 0.952 absolute WER (6.10× reduction) versus 0.161 absolute speaker-
similarity error (1.25×) — a 5.9× magnitude asymmetry;
(c) the rate honesty sentence: on each metric's own normalized curve,
convergence is nearly metric-agnostic (fraction of total gain at T=8:
WER 94.6%, identity 93.7%, identity marginally later) — the asymmetry is in
magnitudes, not rates;
(d) scale persistence: on 20M–276M (four budgets) Δτ = +0.1095,
CI [0.0951, 0.1241].
Any sentence elsewhere in the paper reading as "steps keep helping WER after
SIM stops" (a rate claim) is reworded to the magnitude form or deleted.

---

## 3. PHASE M — paper finalization + clean merge (FIRST task; nothing else runs before it)

1. **Working branch:** continue on `v1.1-extensions`. Confirm tag
   `v1.0-submission-candidate` exists and `paper/main-v1-frozen.tex` is
   untouched.
2. **Execute the four blocking edits** exactly as enumerated in DECISION.md
   ("Not ready to submit" list): remove the cross-metric T* saturation claim
   and neutralize its macros (\NtstarSIM, \Nlatencysaving; src/paper.py:103,
   140); remove the interior-optimum "deep enough" rule (\Nbestdepths;
   src/paper.py:133); remove the depth-first allocation rule (H-E2 refuted);
   regenerate step_curves without raw-scale per-metric T* markers.
3. **Apply §2 wording** to title, abstract, and all affected prose. Update
   `src/paper.py` so every §2 number is a generated macro; hand-typed numbers
   remain forbidden.
4. **Integrate v1.1 strengtheners within the 4-page law:** E4 scale
   persistence (≤2 sentences + Δτ table row), E6 undertraining control
   (1 sentence), E5 robustness (≤2 sentences), E1 extended-T update to the
   step-curves figure and the "T*_WER = 32 > 16 on its own scale" fact,
   E3 allocation (1 main-text sentence + appendix table with its honest
   SIM/UTMOS cost). Add the speaking-rate/char-rate limitation sentence.
   The significance-based 2× T* contrast goes to the appendix with its
   precision-floor caveat, exactly as the retraction entry specifies.
5. **Consistency sweep:** extensions.html and README taglines updated to the
   magnitude framing; one-line errata pointers where old glosses appeared.
   Do NOT edit frozen v1.0 artifacts (results.html, index.html, protocol.html,
   PREREGISTRATION.md, artifacts/, LOG.md) — they are the record, not the
   paper.
6. **Verify:** main text closes on page 4; compiles; anonymized; no retracted
   claim appears anywhere in paper/; `grep` for the retired phrases ("rent
   depth", "buy intelligibility, not identity", "4× saturation", "deep
   enough, not as deep as possible", "depth-first") returns nothing in
   paper/.
7. **PR and merge:** commit with a structured message; open a PR
   `v1.1-extensions → main` (use `gh pr create` if available; otherwise a
   local `git merge --no-ff` with the PR body as the merge-commit message —
   log which path was used). PR body must contain: v1.1 verdict summary
   (supported / refuted / retracted, one line each), the paper diff summary,
   the immutability attestation (`git diff main -- artifacts/
   PREREGISTRATION.md LOG.md results.html paper/main-v1-frozen.tex` is
   empty), and the two resolved decisions. Self-review checklist in the PR:
   page cap, macro-only numbers, anonymity, retraction absence, immutables
   clean. Merge (no history rewrite, no force push), tag `v1.1`, update
   `state.json`/`state-v1.json` open_decisions → resolved, post a feed entry.
   **Gate M-DONE:** Program S may not start until the merge commit exists on
   main and the feed entry is posted.

---

## 4. PROGRAM S — what buys speaker similarity?

Standing question, framed properly (directive §1.5): identity gains must be
read against a ledger. The competing candidate axes, all pre-listed so no
single one is chased blindly: **codec ceiling** (Mimi may cap SIM-o),
**parameters N**, **training compute**, **fine-level refinement allocation**,
**test-time context (prompt length)**, **test-time search (best-of-K)**,
**rate-matched length conditioning**, **guidance** (training-time lever,
gated). Shape-at-fixed-N is already measured (marginal) and is not re-run.

### S0 — Identity ledger + codec ceiling (zero training; run immediately after M)
- Compute SIM-o(Mimi encode-decode roundtrip of ground-truth target vs
  original prompt) on all 400 eval items; also same-speaker and cross-speaker
  GT baselines (exist in G0(c); reuse).
- Assemble the ledger from EXISTING artifacts (no new synthesis): absolute
  SIM-o gain per axis — N (A best → D best at T=16, from runs_4budget.csv),
  training compute (30k→90k, from e6 scores), steps (T1→T16, v1.0), allocation
  (fine→uniform→coarse, from e3_items.csv), shape (within-budget max−min at
  T=16). Headroom = SIM_roundtrip − best measured system SIM.
- Deliverables: `artifacts-v1.2/identity_ledger.json`,
  `figures/identity_ledger.{svg,pdf}` (one bar per axis + ceiling line).
- **H-S0 (pre-registered, §9):** ceiling classification per the frozen rule.
  The ledger is the reference for every later interpretation (§1.5).

### S1 — Test-time context: prompt-length sweep (inference; ~20 GPU-h)
- Rebuild prompt variants {1.5, 3, 6, 9 s} from each eval item's source
  utterance; keep items where all four lengths exist (target ≥200; if <200,
  drop the 9 s arm and log). 3 s arm must be bit-identical to v1.0 prompts.
- **Reference confound (pre-empted):** SIM-o is always scored against the
  SAME fixed reference — the v1.0 canonical 3 s prompt waveform — in every
  arm; secondary scoring vs the full source utterance is reported alongside.
  Never score arm k against its own conditioning audio.
- Runs: C-budget 15 runs, T=16, frozen sampler, matched RNG across arms.
- Note the distribution shift honestly: training used 3 s prompts only;
  6/9 s arms test context extrapolation — both directions are findings.
- **H-S1** per §9 (two lenses + rivals table).

### S2 — Test-time search vs refinement (inference; ~45 GPU-h)
- Instrument-gate ECAPA (speechbrain spkrec-ecapa-voxceleb) via §1.7 first;
  if it fails validation, halt S2 and post — do not substitute silently.
- Arms at matched NFE on 200 items, C-budget 15 runs:
  refinement {T=16, 32, 64} vs best-of-K {K=2, 4, 8 at T=8}, selection by
  WavLM-SV cosine to the prompt, scoring by ECAPA (§1.8). Report the WER of
  selected candidates (selection may covertly select intelligibility —
  a listed rival).
- **H-S2** per §9.

### S3 — Rate-matched length conditioning (inference; ~8 GPU-h)
- Arm A: v1.0 corpus-median sec_per_char (existing). Arm B: per-item
  sec_per_char measured from the prompt source utterance (chars/duration of
  the full source clip). C-budget 15 runs, 400 items, T=16.
- **H-S3** per §9, with the WER guardrail.

### S4 — Speaker-contrastive guidance, training-free (inference; canary-gated; ~15+25 GPU-h)
- Two forward passes per step: logits_cond − γ·logits_wrong-speaker (same
  text, deterministic wrong-speaker assignment: item i gets the prompt of
  item (i+7) mod 200), γ ∈ {0.5, 1.0, 2.0}, T=16.
- **Canary gate:** C3 seed 0, 100 items. Proceed to full C-budget only if
  some γ gives DegenRate ≤ 2× baseline AND SIM-o not worse than baseline
  (else post the negative and stop — a training-free identity knob failing
  is itself worth one feed entry, no more).
- **H-S4** per §9. High-risk, labeled.

### S5 — CFG with condition dropout (training; ~55 GPU-h; POST-FREEZE ONLY)
- Retrain C1/C3/C5 seed 0 with 10% prompt-dropout, guidance sweep at
  inference. Runs only after the Aug 24 paper freeze, results go to
  `ICLR-NOTES-v2.md` and the feed — never the workshop paper (no-self-scoop
  directive: guidance is the sequel's axis).

### E7 — adaptive-T (from task-v1.md) remains scheduled at its Aug 16 gate,
spec unchanged; it shares the queue at priority below S1–S3, above S4-full.

---

## 5. PAPER POLICY FOR PROGRAM S (strict)

The workshop paper is now claim-complete. From all of Program S it may gain
ONLY: (a) the codec-ceiling number, one sentence in limitations; (b) ONE
outlook sentence, chosen from pre-written templates conditional on stable
verdicts: if S1/S2 support — "preliminary evidence suggests identity responds
to test-time context and candidate search where refinement cannot reach it";
if they refute — "identity appears bounded by representation and codec rather
than by any test-time intervention we measured"; if discordant/mixed — no
sentence. Everything else lands in extensions.html, the feed, and
ICLR-NOTES-v2.md. The symmetric-currencies thesis, if supported, is the
ICLR spine — spending it as a workshop appendix is forbidden.

## 6. BUDGET, CALENDAR, CUTS

- Envelope unchanged: 600 GPU-h charged total for v1.1+v1.2 combined
  (~175 spent). Projected S0–S4 + E7 ≈ 130–160.
- Calendar: M-DONE by Aug 9. S0 same day. Pre-registration §9 copied to
  `PREREGISTRATION-v1.2.md` before any S synthesis. S1–S3 verdicts by
  Aug 13. S4 canary by Aug 13, full by Aug 16. E7 gate Aug 16 (unchanged).
  All STABLE verdicts by Aug 20. Paper v3 (the two permitted additions) by
  **freeze Aug 24**. Wall Aug 29 AoE. S5 and any backfill: post-freeze only.
- Cut order on overrun: S4-full → S2 K=8 arm → S1 9s arm → E7 → S2 entirely.
  M, S0, and the paper are never cut.

## 7. DELIVERABLES (v1.2 definition of done)

- [ ] Merge commit on main + tag `v1.1` + PR record + feed entry (Phase M)
- [ ] Paper v3: §2 wording applied, blocking edits done, ≤4 pages, compiles,
      retired-phrase grep clean
- [ ] `PREREGISTRATION-v1.2.md` (verbatim §9) logged before any S data
- [ ] `artifacts-v1.2/`: identity_ledger.json + figure, runs_s1.csv …
      runs_s4.csv, verdicts.json (hypothesis → lenses → rivals → verdict →
      STABLE flag)
- [ ] Feed entries per job with MEASURED/INTERPRETATION separation and filled
      rival tables
- [ ] `ICLR-NOTES-v2.md` updated with the symmetric-currencies assessment
- [ ] `state-v2.json` phase = DONE or BACKGROUND (S5/backfill running)

## 8. WHAT NOT TO DO (v1.2 additions)

- Do not start any S-job before the merge commit exists on main (Gate M-DONE).
- Do not post SUPPORTED/REFUTED with one lens, an unfilled rival table, or
  before S0's ledger exists.
- Do not score any S1 arm against its own conditioning audio.
- Do not let WavLM-SV both select and score in S2.
- Do not use base-plus-sv for anything.
- Do not put S-program results in the paper beyond §5's two permitted items.
- Do not run S5 before the paper freeze, and never cite it in the paper.
- Do not resolve a DISCORDANT verdict by picking a side without a diagnosis
  entry that has itself passed the cooling rule.

## 9. PRE-REGISTRATION v1.2 ADDENDUM (copy VERBATIM to PREREGISTRATION-v1.2.md before any S synthesis)

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

## 10. STARTUP CHECKLIST (every boot)

1. Read `state-v2.json` (create from template on first boot), feed tail,
   §0-directives of task.md and task-v1.md, then §1 of this file.
2. Verify Gate M-DONE status before touching Program S.
3. Re-project compute + calendar; apply §6 cuts if needed; log.
4. Resume queue; backfill idle GPUs; heartbeat the feed if > 12 h silent.
