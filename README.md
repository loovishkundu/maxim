# Maxim

**M**ulti-**A**gent e**X**ploration & **I**ntelligence **M**achine

A multi-agent research assistant. Give it a topic — a feature you're about to
build, a method you're evaluating, a field you need to map — and it plans the
research, fans out parallel researcher agents (AI/agentic, classical ML, data
science, statistics, community sentiment), verifies every claim against the
actual fetched sources, and synthesizes one cited markdown report.

Architecture and design decisions: see [PLAN.md](PLAN.md).

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python package/project manager)
- Python 3.12 (managed automatically by uv)
- An Anthropic API key (search/fetch run server-side through the API — no
  search-provider keys needed)

## Setup

```bash
uv sync
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

## Usage

```bash
uv run maxim "anomaly detection for vehicle telemetry"

# faster / cheaper or deeper runs
uv run maxim "..." --depth quick
uv run maxim "..." --depth deep

# subset of perspectives, machine-readable output, no confirmation prompt
uv run maxim "..." --perspectives classical_ml,statistics --json --yes
```

A run prints the research plan (domain, perspectives, assumptions, estimated
cost) and asks for confirmation before spending money. Reports land in
`./maxim-reports/<topic>-<date>.md`.

Rough cost per run (estimates): quick ≈ $2–3 · standard ≈ $5–8 · deep ≈ $15–25.
`--budget-usd` sets a cost ceiling (default scales with depth: $5 / $12 / $35);
when it's hit the run degrades to a raw findings dump instead of failing.

Exit codes: `0` ok · `2` partial (an agent failed, a cap was hit, or synthesis
degraded) · `3` hard failure · `4` plan declined at the confirmation prompt.
With `--json`, the full `RunResult` JSON is printed to stdout and all progress
goes to stderr — friendly to scripts and agent wrappers. `--quiet` implies
`--yes`.

## How it verifies claims

Every researcher must attach verbatim quotes to each finding. A deterministic
matcher checks each quote against the text of the actually-fetched page —
fabricated quotes are rejected before any LLM judgment. A fresh-context critic
then judges each surviving claim strictly against its evidence. Rejected claims
are listed in the report appendix, never silently kept.

## Development

```bash
uv run pytest        # tests (no network needed)
uv run ruff check .  # lint
```

## Project layout

```
src/maxim/       # application source (pipeline stages, one module each)
tests/           # test suite — FakeLLM end-to-end, no network
PLAN.md          # architecture plan
.env.example     # template for API keys (copy to .env, never commit .env)
```

## Secrets

All secrets/API keys live in a local `.env` file, which is gitignored. Only
`.env.example` (a template with empty values) is tracked in version control.
