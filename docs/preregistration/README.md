# Pre-registrations

Every hypothesis in this project was written down and committed before any of the data
that tests it existed. The commit that first added each file shows when it was frozen.
Each addendum amends nothing in the ones before it.

| File | Scope | First committed |
|---|---|---|
| [`../../PREREGISTRATION.md`](../../PREREGISTRATION.md) | v1.0, the main grid: H-D1 to H-D4, priority order, outcome classes, analysis lock. Frozen before any training run. | 2026-08-05 21:05 UTC, commit `fa36cc9` |
| [`PREREGISTRATION-v1.1.md`](PREREGISTRATION-v1.1.md) | v1.1 extensions (H-E1 to H-E5) | 2026-08-06 22:17 UTC, commit `b56397d` |
| [`PREREGISTRATION-v1.2.md`](PREREGISTRATION-v1.2.md) | v1.2 Program S | 2026-08-08 15:26 UTC, commit `267ae7c` |
| [`PREREGISTRATION-v1.3.md`](PREREGISTRATION-v1.3.md) | v1.3 training-inclusive identity program | 2026-08-10 09:58 UTC, commit `eeae92e` |
| [`PREREGISTRATION-v1.5.md`](PREREGISTRATION-v1.5.md) | v1.5 classifier-free guidance robustness check (evaluated by `src/v15_gate.py`) | 2026-08-12 03:56 UTC, commit `860a958` |

`PREREGISTRATION.md` (v1.0) stays at the repository root. `src/finish_v14.sh` checks it
against the `v1.0-submission-candidate` tag at that path. The addenda were moved here from
the root with `git mv`, so `git log --follow <file>` shows their full history.
`src/check_claims.py` matches these files by basename, so moving them did not change
which files it exempts.
