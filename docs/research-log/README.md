# Research log (process documents)

These files are the working record of the project: the runbooks the experiments were
run from, decision logs, and execution state. They are kept for transparency. They are
not needed to use the code or the models, and they are not a source of numbers. The paper's
numbers come from `src/` scripts that read the committed artifacts.

| File | What it is |
|---|---|
| `task.md` | v1.0 runbook: phases 0 to 7, decision routing, definition of done |
| `task-v1.md` | v1.1 extension runbook (pre-registration addendum v1.1 was copied from its §9) |
| `task-v2.md` | v1.2 identity-program runbook (addendum v1.2 was copied from its §9) |
| `LOG-v1.2.md` | v1.2 append-only decision log |
| `DECISION-v1.2.md` | v1.2/v1.3 decision record |
| `ICLR-NOTES-v2.md` | notes toward a possible longer follow-up paper |
| `state-v2.json` | v1.2 execution state |
| `README-research.md` | the research-phase repository README (errata, status, canonicality), replaced by the release README |

Comments and docstrings in `src/` still cite these files by bare name (for example
"task-v2.md §4"). Those citations refer to the copies in this folder.

These files are still at the repository root, because scripts read or write them at that path:
`LOG.md`, `LOG-v1.1.md`, `DECISION.md`, `RESULTS-FEED.md`, `state.json`, `state-v1.json`,
the `*.html` pages, `logs-v1.5/`, and `.claims-v1.5/`.
