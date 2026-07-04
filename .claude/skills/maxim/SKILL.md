---
name: maxim
description: Run Maxim, the multi-perspective research pipeline, to produce a verified, cited report on a technical topic. Use when the user wants to research approaches before starting a new feature, compare AI/ML/statistical options for a problem, map a technical field, or asks what the community is saying about a method. Costs real money per run ($2-25 by depth) — always confirm scope and depth with the user before launching.
---

# Maxim — multi-perspective research runs

Maxim plans a topic, fans out perspective researchers (AI/agentic, classical
ML, data science, statistics, community), mechanically verifies every quote
against its fetched source, and synthesizes one cited markdown report.

## Before running

1. **Refine the topic to one sentence.** Maxim's planner works best on a
   focused phrase ("hierarchical memory architectures for LLM agents"), not
   a paragraph. Ask the user to disambiguate only if genuinely ambiguous.
2. **Confirm depth and cost with the user** — this skill runs with `--yes`,
   which skips Maxim's own confirmation gate, so YOU are the gate:
   - `--depth quick` ≈ $2–3, a few minutes — orientation pass
   - `--depth standard` ≈ $5–8, ~10–20 min — the default
   - `--depth deep` ≈ $15–25, ~20–30 min — thorough, more repair loops
3. Optional narrowing: `--perspectives ai_agentic,classical_ml,...` (subset),
   `--budget-usd N` (hard cost ceiling), `--fresh` (bypass the page-fetch
   cache for fast-moving topics).

## Running

Launch as a background Bash task from the repo root — runs take minutes:

```bash
uv run maxim "<one-sentence topic>" --depth standard --json --yes
```

- stdout: exactly one line of `RunResult` JSON (nothing else).
- stderr: live per-agent progress — surface interesting lines (failures,
  retries, replans) to the user while waiting.

## Reading the result

Parse the JSON from stdout. Contract (`schema_version` is `"2"` — if it
differs, the schema may have drifted; prefer the report file over field
assumptions):

- `report_markdown` — the full report; also written to `maxim-reports/`
  (default-mode runs print that path on stdout instead).
- `dossiers[].findings[]` — each has `claim`, `confidence`
  (high/medium/low, stamped by a deterministic rubric), `verdict` from the
  grounding critic, and `evidence[]` with per-quote verification `status`
  and source `tier` (A–D).
- `pulse[]` — per-method community sentiment with mechanical sample sizes.
- `warnings[]` — degradations (failed researchers, salvages, quality-gate
  disclosures).

**Trust the pipeline's verification.** Findings marked verified/supported
were mechanically quote-checked against fetched sources and judged by a
fresh-context critic — do not re-research them. Treat `confidence` and the
Caveats section as the honesty signals they are.

## Presenting to the user

1. Give the TL;DR bullets and the Method Landscape table (both near the top
   of `report_markdown`), lightly reformatted if needed.
2. Link the report file path so the user can read the full document.
3. Surface `warnings[]` honestly — a partial run (exit 2) is still useful,
   but say which perspectives degraded.

## Exit codes

- `0` — clean run; present normally.
- `2` — partial (a researcher failed, a cap was hit, or synthesis
  degraded); present the report WITH its warnings.
- `3` — hard failure, no result; report stderr's last lines, don't retry
  more than once.
- `4` — plan declined (shouldn't occur with `--yes`).

## When not to use

Simple factual questions, single-source lookups, or anything a plain web
search answers — Maxim is for multi-perspective method research, and it
spends real money.
