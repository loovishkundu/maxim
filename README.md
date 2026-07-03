# Maxim

**M**ulti-**A**gent e**X**ploration & **I**ntelligence **M**achine

A multi-agent research assistant. Give it a topic — a feature you're about to
build, a method you're evaluating, a field you need to map — and it plans the
research, fans out parallel researcher agents (AI/agentic, classical ML, data
science, statistics), then runs a community-sentiment wave seeded with the
methods those agents actually found. Every claim is verified against the
actual fetched sources, weak dossiers are repaired or replanned in bounded
loops, and the result is one cited markdown report written for researchers,
software engineers, and AI engineers alike.

Architecture and design decisions: see [PLAN.md](PLAN.md).

## Architecture

```
maxim "topic"
   │
   ▼
┌─────────┐  domain, briefs, assumptions — confirmed
│ PLANNER │  before any money is spent (--yes skips)
└────┬────┘
     ▼
┌────────────────────────────────────────────────────────────────┐
│ WAVE 1 — technical researchers (parallel, Semaphore(3),        │
│ soft deadline + salvage, USD budget gate)                      │
│   AI/Agentic · Classical ML · Data Science · Statistics        │
│                                                                │
│   per researcher:                                              │
│   GATHER ──▶ DRAFT ──▶ MECHANICAL VERIFY ──▶ CRITIQUE          │
│     ▲        (web + paper    (quote-in-source   (fresh-context │
│     │         search tools)   match, 0 LLM)      haiku→opus)   │
│     │                                               │          │
│     └── RETRY / RE-VALIDATE / REPLAN ◀── decide()  ◀┘          │
│         (bounded, deterministic routing — never LLM judgment)  │
└────────────────────────────┬───────────────────────────────────┘
                             ▼
                ┌────────────────────────┐
                │ CANONICALIZE methods   │  "XGBoost" ≡ "gradient
                │ (one haiku call)       │   boosted trees"
                └────────────┬───────────┘
                             ▼
┌────────────────────────────────────────────────────────────────┐
│ WAVE 2 — community researcher, seeded with the methods wave 1  │
│ ACTUALLY found (HN/GitHub tools, engagement floors,            │
│ ≥2-thread corroboration, insufficient_data honesty)            │
└────────────────────────────┬───────────────────────────────────┘
                             ▼
┌─────────────┐  cites finding   ┌─────────────────────────────┐
│ SYNTHESIZER │  ids only        │ QUALITY GATE (0 LLM)        │
│ (streamed)  │─────────────────▶│ missing sections, uncited   │
│             │◀── one repair ───│ prose/rows, unknown ids     │
└─────────────┘    pass          └──────────────┬──────────────┘
                                                ▼
                              maxim-reports/<topic>-<date>.md
                              + sources by tier · rejected-claims
                                appendix · run-metadata footer
```

## How a run works

1. **Plan** — one call classifies the topic, scopes perspectives, and writes a
   research brief per agent. You confirm before money is spent.
2. **Wave 1** — the technical researchers run in parallel (bounded by a
   semaphore, a per-agent time budget, and a USD budget gate). Each is a
   bounded loop: gather → draft → mechanical verify → fresh-context critique,
   with deterministic routing into repair loops (below).
3. **Canonicalize** — method names from wave 1 are merged ("XGBoost" vs
   "gradient boosted trees") so the report speaks one vocabulary.
4. **Wave 2** — the community researcher investigates how practitioners are
   responding to *those* methods (HN, GitHub, benchmarks), with mechanical
   corroboration rules and engagement floors.
5. **Synthesize** — a streamed writer produces the report, allowed to cite
   only finding ids; a deterministic quality gate rejects uncited or
   template-breaking drafts and demands one repair pass.

### The repair loops (M2)

A pure-Python router — never LLM judgment — decides after each pass:

| Loop | Trigger | Action |
|---|---|---|
| RETRY | 20–50% of claims weak | search better evidence for flagged claims only |
| RE-VALIDATE | broken quotes on ≤30% of claims | re-fetch the exact URLs, no new searching |
| REPLAN | too few findings, >50% unsupported, ≥2 coverage gaps, tier collapse | fresh conversation on a revised brief seeded with what failed |

All loops are capped per depth preset; caps exhausted → the dossier ships
with its weaknesses disclosed and confidence capped, never silently.
Timeouts are graceful: researchers stop at a soft deadline and salvage
validated findings instead of losing the run.

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

Optional (never required — everything degrades gracefully without them):
`SEMANTIC_SCHOLAR_API_KEY` and `GITHUB_TOKEN` in `.env` raise rate limits for
the paper-search and community-search tools.

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

Citations are the product. The chain, in order:

1. **Mechanical quote matching** — every finding carries verbatim quotes; a
   deterministic matcher checks each against the text of the actually-fetched
   page (or tool result). Fabricated quotes are rejected before any LLM
   judgment.
2. **Source reputation** — every source is tiered A–D (peer-reviewed →
   forums) by a domain registry, with recency scored against per-perspective
   half-lives. A finding cannot reach *high* confidence without a verified
   quote from a tier A/B source; confidence is stamped by a deterministic
   rubric, never by the model.
3. **Fresh-context critique** — a critic that never sees the researcher's
   conversation judges each claim strictly against its quotes and the cached
   source context (haiku in batches; contradicted or split verdicts are
   re-arbitrated one-by-one on opus).
4. **Community corroboration** — sentiment claims need ≥2 independent
   qualifying threads (engagement floors from real metadata); below 3 threads
   a method's pulse renders as *insufficient data*, never a guess.
5. **Synthesis citation contract** — the writer may only cite finding ids; a
   post-processor resolves them to sources, flags unknown ids, and a
   deterministic quality gate rejects uncited paragraphs and uncited
   landscape rows (one repair pass, then disclosed as warnings + exit 2).

Rejected claims are listed in the report appendix with reasons, never
silently kept.

## Development

```bash
uv run pytest        # tests (no network needed)

# lint — all three must pass before any commit (see CLAUDE.md)
uv run isort .
uv run black .
uv run ruff check .
```

## Project layout

```
src/maxim/       # pipeline stages, one module each
  schemas.py     #   the inter-stage contract (pydantic, extra="forbid")
  llm.py         #   all Anthropic API knowledge: parse-retry, agentic loop,
                 #   pause_turn, client-tool execution, source harvesting
  planner.py     #   topic → plan; replan_task() for structural failures
  orchestrator.py#   two-wave fan-out, canonicalization, budget gate,
                 #   synthesis quality gate — the programmatic seam
  researcher.py  #   the bounded gather→draft→verify→critique loop
  loop.py        #   deterministic retry/re-validate/replan routing
  verification.py#   mechanical quote-in-source matching (zero LLM)
  reputation.py  #   source tiers A–D, recency half-lives, SEO blocklist
  critic.py      #   fresh-context critique: haiku batches + opus escalation
  sentiment.py   #   community floors, corroboration, per-method pulse
  methods.py     #   method-name canonicalization between waves
  quality.py     #   deterministic report-quality gate
  synthesizer.py #   streamed synthesis + repair pass
  report.py      #   citation resolution, sources, appendices, run metadata
  tools/         #   client-side tools: Semantic Scholar, arXiv, HN, GitHub
tests/           # FakeLLM + MockTransport suite — no network, no API key
PLAN.md          # architecture plan
CLAUDE.md        # project rules (lint gate, test gate, environment quirks)
.env.example     # template for API keys (copy to .env, never commit .env)
```

The pipeline is importable without the CLI (`orchestrator.run_pipeline`), and
the CLI keeps a strict wrapper contract (result on stdout, progress on
stderr, stable exit codes) — this is what a future Claude Code skill or MCP
server wraps.

## Secrets

All secrets/API keys live in a local `.env` file, which is gitignored. Only
`.env.example` (a template with empty values) is tracked in version control.
