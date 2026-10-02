# archive

Records of the research process, kept for transparency. You do not need them to run the
code or the models, with one exception: `src/` reads `research-log/state.json` (the paper's
`\Nbaselr` and `\Ncuts` macros come from it). Nothing else in the paper comes from here.

| Folder | Contents |
|---|---|
| [research-log/](research-log/) | Runbooks, decision logs, the results feed and the orchestrator state. `src/orchestrate.py` writes `state.json` and appends to `LOG.md`. See [research-log/README.md](research-log/README.md). |
| `claims-v1.5/` | Claim files, one per 180k-step training job, each holding the PID of the worker that took it. `src/run_v15_train.py` creates them so that two workers never train the same run. All eight jobs are claimed, so rerunning that script retrains nothing. |
