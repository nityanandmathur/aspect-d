# Research log (process documents)

These files are the working record of the project: the runbooks the experiments were
run from, decision logs, and execution state. They are kept for transparency. They are not
needed to use the code or the models, with one exception: several `src/` scripts read
`state.json` (the paper's `\Nbaselr` and `\Ncuts` macros come from it). The paper's other
numbers come from `src/` scripts that read the committed results in `results/`.

| File | What it is |
|---|---|
| `task.md` | v1.0 runbook: phases 0 to 7, decision routing, definition of done |
| `task-v1.md` | v1.1 extension runbook (pre-registration addendum v1.1 was copied from its §9) |
| `task-v2.md` | v1.2 identity-program runbook (addendum v1.2 was copied from its §9) |
| `LOG.md` | v1.0 append-only decision log; `src/orchestrate.py` appends to it |
| `LOG-v1.1.md` | v1.1 append-only extension log |
| `LOG-v1.2.md` | v1.2 append-only decision log |
| `DECISION.md` | v1.0 decision record (outcome class S1) |
| `DECISION-v1.2.md` | v1.2/v1.3 decision record |
| `RESULTS-FEED.md` | append-only results feed from v1.1 to v1.5 |
| `state.json` | v1.0 orchestrator state; `src/orchestrate.py` reads and writes it, and `src/paper.py`, `src/fit.py`, `src/report.py`, `src/samples.py` and other scripts read it (for example `chosen_lr`, the paper's `\Nbaselr`) |
| `state-v1.json` | v1.1 execution state |
| `state-v2.json` | v1.2 execution state |
| `ICLR-NOTES-v2.md` | notes toward a possible longer follow-up paper |
| `README-research.md` | the research-phase repository README (errata, status, canonicality), replaced by the release README |

Comments and docstrings in `src/` still cite these files by bare name (for example
"task-v2.md §4" or "LOG.md P0-3"). Those citations refer to the copies in this folder.
Until the 2026-10-02 reorganisation, `LOG.md`, `LOG-v1.1.md`, `DECISION.md`, `RESULTS-FEED.md`,
`state.json` and `state-v1.json` were at the repository root, and the rest of this folder
was `docs/research-log/`; `git log --follow <file>` shows the full history of each file.
