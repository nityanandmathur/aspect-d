# ASPECT-D pre-registration addendum v1.1

Copied VERBATIM from `task-v1.md` §9 at Phase E0, frozen before any extension
run, before any extension synthesis, training or scoring. Amends nothing in
v1.0; the v1.0 declared outcome (S1) is immutable.

---

> **ASPECT-D pre-registration addendum v1.1** — frozen before any extension
> run; amends nothing in v1.0; the v1.0 declared outcome (S1) is immutable.
>
> - **H-E1 (extended test-time scaling).** On T ∈ {1,…,64} (21-run subset, 200
>   items): WER's saturation step T*_WER (95% of the T=64 value) is > 16, and
>   the refitted τ_WER on the extended range lies within the v1.0 τ_WER 95% CI
>   [0.8248, 0.8513]. Either half failing is reported as measured; T* values
>   are reported regardless.
> - **H-E2 (iso-latency allocation).** At every latency budget L in the tested
>   grid, the WER-optimal allocation satisfies d ≥ the budget's interior
>   optimum d* before steps are increased beyond the minimum tested — i.e.,
>   owned depth dominates rented steps for WER up to d*, by both the fitted
>   surface and nearest-measured-point methods. Decision: agreement of both
>   methods at ≥ 80% of tested L values.
> - **H-E3 (coarse-level allocation).** At matched NFE = 32 on the C budget:
>   WER(coarse) < WER(fine), paired item-level bootstrap 95% CI on the
>   difference excluding 0; secondary: WER(coarse) ≤ WER(uniform).
> - **H-E4 (scale persistence).** On the 4-budget (A–D) Part-B surface with
>   seeds {0,1} per config minimum: Δτ > 0 with run-level bootstrap 95% CI
>   excluding 0. Secondary (new test, not a redo of v1.0 H-D4): M_sep fit on
>   A+B+C predicts budget D config means at T=16 with MAPE ≤ 15% and ≤ the
>   N-only model's MAPE. Exploratory (no decision rule): d*(N) trend.
> - **H-E5 (training-compute control).** For {C1, C3, C5} trained to 90k
>   steps (seed 0): the τ-surface refit gives Δτ_90k with item-level bootstrap
>   95% CI excluding 0. Reported alongside: whether the Δτ_90k CI overlaps the
>   matched 30k 3-config Δτ CI. Single-seed scope is acknowledged; this is a
>   control, not a headline.
> - **H-E6 (adaptive steps).** Confidence-adaptive stopping (rule frozen in
>   task-v1.md §4-E7) on the C budget achieves corpus WER ≤ fixed-T16 WER +
>   1.0 absolute point at mean NFE ≤ 0.6 × 128, per-run, in ≥ 12 of 15 runs.
> - Analysis machinery, RNGs, and bounds per task-v1.md §5. Everything not
>   listed above is exploratory and will be labeled so wherever it appears.

---

**Frozen:** 2026-08-06, before any v1.1 extension datum existed. Anything not
listed above is exploratory and is labeled so wherever it appears.
