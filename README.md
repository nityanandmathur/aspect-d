# Project ASPECT-D

**Denoising steps rent depth, not width.** Per-metric (**w**idth, **d**epth,
**T** steps) scaling laws for masked-diffusion TTS.

Target: **DiffuLM @ NeurIPS 2026** — 4-page extended abstract, non-archival,
double-blind, **deadline 2026-08-29 AoE**.

One training grid (iso-N shapes: 20M/50M/125M × 5 aspect ratios; seeds 2
mandatory + 1 stretch), one inference-time sweep (T ∈ {1…16} steps per codebook
level, NFE = 8T), 75 surface points per metric. Primary pre-registered claims:
**H-D2** — refinement steps improve intelligibility far more than speaker
identity (test-time scaling is metric-selective); **H-D3** — for intelligibility,
steps and depth exchange at a measurable rate κ. ~240–420 GPU-h, capped at 500,
calendar-gated to the deadline.

## Relationship to Project ASPECT

Independent sibling. ASPECT (autoregressive) keeps the AR anisotropy headline for
a later full-length paper; ASPECT-D ships the diffusion-native step/κ headline to
the workshop. Shared: the validated iso-N shape dimensions and the eval-model
choices. Not shared: backbone family, objective, sampler, hypotheses priority,
reliability metric (degenerate rate, not runaways), statistics (run-level
bootstrap over a correlated T surface), and the calendar-gate machinery.

## Repo map

| File | Role |
|---|---|
| `index.html` | Theory, motivation, figures, benefits. **Never a source of numbers.** |
| `protocol.html` | **Canonical** thresholds, gates G0–G6 (G6 = calendar), pivots P1-D/P2, stats, deliverables. |
| `configs/grid.json` | **Canonical** architecture, recipe, frozen sampler, T grid, compute cap, calendar milestones. |
| `task.md` | Autonomous runbook: phases 0–7 (7 = paper), decision routing, defaults, definition of done. |
| `PREREGISTRATION.md` | Frozen H-D1–H-D4, priority order, outcome classes, analysis lock. |
| `paper/OUTLINE.md` | The 4-page structure Phase 7 fills, with per-section budgets and hard rules. |
| `paper/main.tex` | LaTeX skeleton with TODO slots traceable to `fits.json`/`runs.csv`. |
| `artifacts/` | Outputs land here; `results.html` at repo root. |

## Reading order

Humans: `index.html` → `protocol.html` → `paper/OUTLINE.md`.
Claude Code: `task.md` → `protocol.html` → `configs/grid.json` → `paper/OUTLINE.md`
(then `index.html` for context only).

## Canonicality

On any conflict: `configs/grid.json` → `protocol.html` → `task.md` → `index.html`.
A number found in none of them must not be invented (task.md §0, §10).

## Status

v1.0 — protocol and pre-registration frozen, no runs executed. Execution state
lives in `state.json`, the decision trail in `LOG.md` (created by the agent).
