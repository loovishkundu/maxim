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
│     │         search tools)   match, 0 LLM)      + arbitration)│
│     │                                               │          │
│     └── RETRY / RE-VALIDATE / REPLAN ◀── decide()  ◀┘          │
│         (bounded, deterministic routing — never LLM judgment)  │
└────────────────────────────┬───────────────────────────────────┘
                             ▼
                ┌────────────────────────┐
                │ CANONICALIZE methods   │  "XGBoost" ≡ "gradient
                │ (one low-effort call)  │   boosted trees"
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

### Failure handling

Every Anthropic API call runs through a
[pyresilience](https://github.com/AhsanSheraz/pyresilience) policy:
transient failures retry with exponential backoff and jitter — connection
drops around the request, drops *during* stream body iteration (which the
SDK does not wrap), 429s, 5xx/529s, and mid-stream `overloaded_error`
events; 4xx client errors never retry; a circuit breaker per call type
fails remaining calls fast once the API is clearly down. The
client tools' own HTTP calls (HN Algolia, GitHub, Semantic Scholar, arXiv)
retry the same way — transport errors, 429, 5xx — and a tool that stays
down degrades to capped is_error results the researcher routes around.
Above that, a researcher that dies outright gets one
fresh-conversation retry before its section is given up, one that dies
mid-run salvages its already-validated findings from a checkpoint, a
late-stage failure (synthesis, canonicalization, even writing the report
file) degrades to a fallback rather than discarding the paid-for run, and
every failure or retry is announced in the live progress stream the moment
it happens — not discovered in the report 20 minutes later.

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

# fast-moving topic: bypass the server-side page-fetch cache
uv run maxim "..." --fresh
```

Freshness: web search sees today's web (well-indexed sources surface
same-day) and the HN/GitHub tools are near-real-time; arXiv appears on the
next announcement cycle and Semantic Scholar lags days. Fetched pages are
served from a server-side cache by default — pass `--fresh` when today's
version of an already-published page matters.

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

## Use from Claude Code (skill)

The repo ships a Claude Code skill at
[.claude/skills/maxim/SKILL.md](.claude/skills/maxim/SKILL.md). Open this
repo in Claude Code and ask things like "research approaches for X before I
build it" or "what is the community saying about Y" — the skill confirms
depth/cost with you, runs Maxim as a background task (`--json --yes`),
streams progress, and presents the TL;DR + Method Landscape with a link to
the full report. It trusts the pipeline's verification (confidence, verdicts,
tiers) instead of re-researching. A test suite keeps the skill honest against
the real CLI: flags, exit codes, schema version, and field names are checked
in CI.

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
   source context (batched at low effort; contradicted or split verdicts are
   re-arbitrated one-by-one in a fresh call).
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
uv run pytest        # tests (no network needed; calibration excluded)

# lint — all three must pass before any commit (see CLAUDE.md)
uv run isort .
uv run black .
uv run ruff check .

# judge-drift calibration against the live API (costs a few cents)
uv run pytest -m calibration

# regenerate the golden report snapshot after an intentional template change
UPDATE_GOLDEN=1 uv run pytest tests/test_golden_report.py
```

CI (GitHub Actions) runs the same lint gate and test suite on every push
and PR; the calibration job runs only on manual dispatch.

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
  critic.py      #   fresh-context critique: batched verdicts + arbitration
  sentiment.py   #   community floors, corroboration, per-method pulse
  methods.py     #   method-name canonicalization between waves
  quality.py     #   deterministic report-quality gate
  synthesizer.py #   streamed synthesis + repair pass
  report.py      #   citation resolution, sources, appendices, run metadata
  tools/         #   client-side tools: Semantic Scholar, arXiv, HN, GitHub
tests/           # FakeLLM + MockTransport suite — no network, no API key
  golden/        #   snapshot for the golden-report regression test
.claude/skills/  # the Claude Code skill wrapping the CLI
.github/         # CI: lint gate + tests on push; calibration on dispatch
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
