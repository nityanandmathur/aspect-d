
## 2026-08-06 02:05 UTC — first completed pilot (A3, seed 0) — recipe validated

A3 (21 M non-embedding, d=12, w=384) finished its full 30,000-step schedule and was
synthesised over the whole 400-item eval_zs at both T endpoints:

| T | WER | SIM-o | DegenRate |
|---|---|---|---|
| 1 | 118.8 % | 0.198 | 40.8 % |
| 16 | **16.1 %** | **0.363** | **0.2 %** |

Consequences:

1. **The coarse-to-fine recipe works.** The 01:30 probe (WER ≈ 1.0 at 27 % of the
   schedule) was simply early: text adherence emerges in the second half of
   training. WER 16.1 % on held-out speakers from the *smallest* pilot is far
   inside G2's A3 floor (≤ 65 %), and its SIM-o already exceeds the C3 floor
   (≥ 0.30) on its own. Pivot P1-D remains implemented but is **not** fired.
2. **The step axis bites, hard, and asymmetrically.** On the frozen error scale,
   T=1 → T=16 divides err_WER by 7.4 (1.188 → 0.161) but err_SIM by only 1.26
   (0.802 → 0.637). Under `err = E + C·T^−τ` that is a large τ_WER against a small
   τ_SIM, i.e. Δτ > 0 — the direction H-D2 pre-registered. This is one config of
   fifteen and carries no CI; the declared outcome will come from the full
   run-level bootstrap over the whole surface, not from here.
3. **The T-FLAT flag will not fire**: WER(C3,T=1) − WER(C3,T=16) is nowhere near
   the < 3-point flatness condition (A3 alone shows a 102.7-point gap).
4. DegenRate 40.8 % → 0.2 % across T is a clean descriptive curve for §7.3.
