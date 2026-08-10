# ASPECT-D pre-registration addendum v1.3 — training-inclusive identity program

Frozen **before any v1.3 datum exists**. Amends nothing in v1.0 / v1.1 / v1.2;
all declared outcomes remain immutable. Verdict discipline of `task-v2.md` §1
binds every hypothesis: two independent lenses, a filled rival table where
"addressed" means a number, MEASURED/INTERPRETATION separation, the cooling rule
before any paper edit, and ledger-relative interpretation against the S0 headroom
of **0.0747** SIM-o.

**Motivation.** Program S measured five identity axes at test time and found only
one that works (search, +0.0365 against the deployable default, 49 % of headroom).
But the S0 ledger already showed the largest single identity gain of any axis is
**training compute** (30k→90k, +0.0788), measured on three configs at one seed.
Program S's own closing entry listed two unexcluded rivals that training and scope
can address: everything was C-budget at 30k steps, and no axis was retested where
the baseline is stronger and the headroom smaller.

---

> ## Hypotheses
>
> - **H-T1 (search survives a stronger baseline).** Re-run the S2 matched-NFE
>   contrast on checkpoints outside the C-budget/30k scope: the ten budget-D runs
>   (276 M) and the three 90k runs. Primary: ECAPA-SIM(best-of-K) −
>   ECAPA-SIM(refinement) paired per-run CI > 0 at NFE 512 on **both** groups
>   separately. Second lens: per-item win rate > 50 % with CI, on both groups.
>   Rivals: headroom shrinkage (check: report each group's own headroom
>   SIM_rt − best-in-group, and the gain as a fraction of it); selection-scorer
>   family overlap (check: the random-pick / WavLM-selected / ECAPA-oracle
>   decomposition repeated per group — shared bias predicts ≈100 % of the oracle
>   gap captured, the C-budget value was 44–46 %); candidate-pool degeneracy
>   (check: DegenRate of candidates by group).
>
> - **H-T2 (the training-compute identity axis is not saturating).** Extend the
>   30k→90k measurement with seeds {1,2} at 90k for C1/C3/C5 and a 180k point for
>   C3 seed 0. Primary: SIM-o(180k) − SIM-o(90k) paired per-item bootstrap 95 % CI
>   > 0 on C3 seed 0. Second lens: across the three configs with three seeds each
>   at 90k, the 30k→90k gain replicates with a run-level bootstrap CI > 0 (i.e.
>   the +0.0788 single-seed figure is not a seed artifact). Rivals: WER/degeneracy
>   confound (check: both reported at every step count — an identity gain
>   accompanied by a WER regression > 2.0 points is a trade, not a win); log-linear
>   vs saturating shape (check: fit SIM-o against log-steps on {30k, 90k, 180k}
>   and report the residual at 180k, no decision rule attached).
>
> - **H-T3 (context is a training limitation, not a test-time dead end).** S1
>   found that prompts longer than the 3 s seen in training decouple identity from
>   content: at 9 s, identity is nearly intact (0.3759 vs 0.3839) while WER rises
>   71.7 points, with degeneracy under 1.6 %. Retrain C3 seed 0 with prompt lengths
>   sampled uniformly from {1.5, 3, 6, 9} s, then re-run the S1 arm sweep on it.
>   Primary: for the variable-length model, paired per-item SIM-o(9 s) − SIM-o(3 s)
>   CI > 0 **and** WER(9 s) − WER(3 s) ≤ +2.0 points. Second lens: the Spearman
>   trend over {1.5, 3, 6, 9} is positive **and** the arm curve is monotone
>   (the diagnosis that made H-S1 DISCORDANT was non-monotonicity, so monotonicity
>   is required here rather than assumed). Rivals: the retrained model is simply
>   better (check: compare it to C3 seed 0 at the 3 s arm — a uniform improvement
>   is not evidence about context); training-distribution match trivially fixing
>   WER without touching identity (check: report both deltas separately, and
>   classify as "content-only fix" if SIM is flat).
>
> - **Program-level statement.** H-T1 tests whether Program S's one positive
>   result generalises; H-T2 sharpens the largest measured identity lever; H-T3
>   converts a DISCORDANT verdict into a decidable question. No single SUPPORTED
>   verdict is glossed as "the" answer. The closing entry must re-rank every axis
>   on the S0 ledger including the new points, state which rivals remain
>   unexcluded, and report the cost of each axis in GPU-hours as well as in
>   metric. Analysis machinery per task-v1.md §5; RNGs unchanged; anything not
>   listed is exploratory and labeled so, permanently (§1.6).
>
> **Explicitly out of scope.** S5 (CFG with condition dropout) remains
> **post-freeze only** per task-v2.md §4-S5 and §8, is not run here, and is not
> cited in the paper. Nothing in v1.3 may enter the workshop paper beyond what
> task-v2.md §5 already permits.

---

**Frozen:** 2026-08-10, before any v1.3 run, synthesis, scoring or analysis.
Envelope: 89.2 of 600 GPU-h charged; v1.3 projected ≈ 130–170.
