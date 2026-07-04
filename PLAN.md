# Maxim — Architecture & Build Plan

> Status: M2 implemented — five-loop state machine with per-depth thresholds,
> two-wave fan-out with method canonicalization, community sentiment rigor,
> reputation tiers, custom client-side tools, a batched critic with
> single-finding arbitration, and a deterministic report quality gate. M3 (Claude skill,
> calibration suite, golden-report test) still to come.
> Maxim replaces manual internet-scrolling when you start a new feature: it scopes the topic,
> researches it from multiple perspectives in parallel, verifies every claim against its source,
> and synthesizes one structured, cited report — including how the community is actually
> responding to each method.

---

## 1. Big picture

A five-stage asyncio pipeline on the official `anthropic` SDK. No orchestration frameworks.

```
maxim "topic"
   │
   ▼
┌──────────┐   ┌─────────────────────────────────────────────┐   ┌────────────┐
│ PLANNER  │──▶│ WAVE 1: 4 perspective researchers (parallel) │──▶│ CANONICAL- │
│ 1 call:  │   │  AI/Agentic · Classical ML · Data Science ·  │   │ IZE method │
│ domain + │   │  Statistics                                  │   │ names      │
│ briefs   │   │  each: GATHER→DRAFT→VERIFY→CRITIQUE loop     │   └─────┬──────┘
└──────────┘   └─────────────────────────────────────────────┘         │
                                                                        ▼
┌────────────┐   ┌──────────────┐   ┌──────────────────────────────────────┐
│  REPORT    │◀──│ SYNTHESIZER  │◀──│ WAVE 2: Community-sentiment researcher│
│ markdown + │   │ streamed,    │   │ seeded with the methods wave 1       │
│ json       │   │ cites only   │   │ ACTUALLY found (not planner guesses) │
└────────────┘   │ finding ids  │   └──────────────────────────────────────┘
                 └──────────────┘
```

Key architectural commitments (each adjudicated by an adversarial design review):

- **Two-wave fan-out.** The community researcher runs *after* the perspective researchers,
  seeded with the union of methods they actually discovered. Sentiment about methods nobody
  found is noise; the user's core ask is "how is the community responding to *each method*".
- **Grounding is mechanical first, LLM second.** A deterministic quote-in-source matcher runs
  before any LLM critic. An LLM comparing two model-produced strings cannot catch a fabricated
  quote; a substring match against cached page text can.
- **Loop routing is pure Python.** The decision to retry / replan / re-validate is a
  deterministic function of critic verdicts and mechanical checks — testable, debuggable,
  immune to LLM-judge mood.
- **Synthesizer can only cite finding IDs.** A post-pass resolves IDs to URLs; sentences citing
  unknown IDs get repaired or flagged, never silently kept. Hallucination is blocked at the
  synthesis layer too, not just the research layer.
- **Bounded everything.** Semaphore(3) concurrency, per-agent wall-clock timeouts, tool-call
  caps, iteration caps, a USD budget gate. Exhaustion degrades to a partial report (exit 2),
  never a crash.

## 2. Requirements → mechanisms

| # | Requirement | Mechanism |
|---|---|---|
| 1 | Understand query, scope domains | `planner.py`: one `messages.parse` call → `ResearchPlan` (domain classification, in-scope perspectives with rationale, recency horizon, per-perspective briefs). Plan's assumptions printed before fan-out; `--yes` skips confirmation. |
| 2 | Spawn multiple research agents | `orchestrator.py`: `asyncio.gather` over researchers behind `Semaphore(3)` + `asyncio.wait_for` per agent (soft deadline + salvage checkpoint; hard `wait_for` as a hang backstop). |
| 3 | Reputable, recent sources | Server-side `web_search`/`web_fetch` + optional Semantic Scholar/arXiv tools; `reputation.py` A–D tier scoring (score, not filter; SEO-farm blocklist → `blocked_domains`); per-perspective recency half-lives modulated by planner's topic-level horizon. Tier mix shown in report. |
| 4 | Multi-perspective + reviews | One researcher per perspective, each brief carries an `avoid` field ("no LLM/GenAI methods" for ML/DS/stats) against perspective collapse. Wave-2 community researcher with corroboration rules + `how_people_test_it` per method. |
| 5 | Critique tool grounding each agent | Two gates: mechanical verify (quote-in-`SourceCache` fuzzy match ≥0.85; server `cited_text` blocks pre-verified) then fresh-context LLM critic (sees only claim + ±500-char cached excerpt — never the researcher's reasoning). |
| 6 | Retry / replan / re-validate loops | Five named loops (§5), deterministic routing, hard caps, defined exhaustion behavior. |
| 7 | Structured synthesizer output | Streamed prose constrained to a fixed template: TL;DR, Method Landscape table, four Takes, Community Pulse, Decision Guide, Caveats, tiered bibliography (§8). |
| 8 | Clone → `uv sync` → run | 4 runtime deps in M1, argparse, no infra, no resume machinery. Default run ≈ $5–8 / <10 min. |
| 9 | Claude integration | `.claude/skills/maxim/SKILL.md` wrapping the CLI (`--json`, stderr progress, stable exit codes). Optional MCP server later as an extra. |

## 3. Module layout

```
src/maxim/
├── __init__.py        # main() → asyncio.run(cli.run())      (entry point already wired)
├── cli.py             # argparse, rich progress, exit codes (0 ok, 2 partial/budget, 3 failed)
├── config.py          # Settings: model ids, effort per stage, ALL numeric budgets,
│                      # depth presets (quick/standard/deep), pricing table (estimate-labeled)
├── schemas.py         # every Pydantic model (the inter-stage contract; extra="forbid")
├── llm.py             # one shared AsyncAnthropic client; parse_with_retry() used by EVERY
│                      # stage; stream_agentic() handling pause_turn; byte-stable cached
│                      # system prompts; usage extraction
├── prompts/           # frozen system prompts per role (byte-stable for prompt caching)
├── planner.py         # Stage 1: plan(topic) → ResearchPlan; replan_task(task, critique)
├── orchestrator.py    # two-wave fan-out, semaphore, timeouts, method canonicalization,
│                      # UsageLedger budget gate
├── researcher.py      # ResearcherAgent state machine: GATHER→DRAFT→VERIFY→CRITIQUE
├── verification.py    # mechanical checks: quote match, recency, tier stamping (zero LLM)
├── critic.py          # fresh-context critique calls + deterministic decide() routing
├── sentiment.py       # community-researcher specialization (corroboration, aboutness,
│                      # engagement floors, insufficient_data rule)
├── reputation.py      # domain→tier registry, recency half-lives, tier-mix computation
├── synthesizer.py     # Stage 4: streamed synthesis, finding-id citation contract
├── report.py          # template, id→URL resolution + repair, markdown/JSON render,
│                      # writes ./maxim-reports/<slug>-<date>.md
├── usage.py           # UsageLedger: tokens+search-fees → USD, budget enforcement
└── tools/             # (M2) optional custom client-side tools, all graceful without keys
    ├── semantic_scholar.py · arxiv_api.py · hn_algolia.py · github_search.py
```

`pipeline.run(topic, settings) -> RunResult` is the single programmatic seam: no global
state, no printing inside the pipeline (progress via callback) — this is what makes the
Claude skill and a future MCP server nearly free.

## 4. Data models (the contract)

The load-bearing ones — all in `schemas.py`, Pydantic v2, used both as `messages.parse`
output formats and on-disk artifacts:

- **`ResearchPlan`** — domain, rationale, `perspectives_in_scope`, recency horizon (months),
  assumptions, one **`ResearchBrief`** per perspective: `sub_questions`, `seed_queries`,
  `must_cover_methods`, `avoid` (e.g. "no LLM-based methods" for the stats brief).
- **`EvidenceQuote`** — verbatim `quote`, `source_url`, `published_at`, `verified: bool`,
  `match_ratio`, `verification_skipped` (paywalls/talks), `evidence_kind`
  (`paper|blog|talk|docs|anecdote|benchmark|production_report`), `engagement` metadata.
- **`Finding`** — `id`, `perspective`, one falsifiable `claim`, `method_name`,
  `evidence: list[EvidenceQuote]` (min 1), `source_tier` (A–D), `confidence`
  (`high|medium|low|unverified` — set by deterministic rubric, not the model),
  `sentiment` + `sentiment_sample_size` (community only), `caveats`.
- **`ClaimVerdict`** — `supported | partially_supported | unsupported | contradicted |
  source_unreliable`, `fix_hint`, per finding. **`CritiqueReport`** — verdicts,
  `coverage_gaps`, `decision` + reasons.
- **`ResearchDossier`** — validated findings per perspective, `methods_identified`,
  `budget_exhausted`, iteration counters.
- **`CommunityPulse`** — per method: sentiment (`positive|mixed|negative|insufficient_data`),
  `notable_threads`, **`how_people_test_it: list[str]`** (the user's explicit ask).
- **`RunResult`** — plan + dossiers + pulse + report markdown + `UsageLedger`;
  carries `report_schema_version` for the Claude-skill contract.

## 5. The researcher agent loop (req 5 & 6)

Each researcher is a bounded state machine (manual loop — not `tool_runner` — because we need
`pause_turn` handling for server tools, mid-loop budget checks, and progress events).

**Tools:** server-side `web_search_20260209` + `web_fetch_20260209` (citations on,
`max_uses` per depth preset, `blocked_domains` from reputation policy) — zero-key baseline.
Perspective-specific custom tools in M2 (Semantic Scholar/arXiv for ML/stats; HN
Algolia/GitHub for community). **Every fetched page and tool result is captured into a
`SourceCache` (url → text) — the ground truth for all verification.**

**Phases per iteration:**

1. **GATHER** — agentic tool loop against the brief (streamed; re-send on `pause_turn`,
   cap ~6 continuations; per-iteration tool-call cap).
2. **DRAFT** — `messages.parse` → dossier draft; every finding must carry ≥1 verbatim quote.
3. **MECHANICAL VERIFY** (zero LLM cost) — locate each quote in `SourceCache[url]` via
   normalized-substring then fuzzy match (pass ≥0.85). Server `cited_text` blocks count as
   pre-verified. Paywalled/JS/video sources with no cached text → `verification_skipped` +
   confidence capped at `low` (never silently pass, never wrongly kill). Stamps tier + recency.
4. **LLM CRITIQUE** — a **fresh conversation**: the critic sees only claim + quote +
   ±500 chars of cached source context, never the researcher's reasoning or search trail.
   Batched 8 findings/call on `claude-opus-4-8` (effort=low); `contradicted`/
   `source_unreliable`/split verdicts are re-arbitrated one-by-one in a fresh call.
   Also reports `coverage_gaps` against the brief's sub-questions.

**Five loops, named precisely:**

| Loop | Trigger | Action | Cap |
|---|---|---|---|
| Transport retry | 429/5xx/timeouts | SDK `max_retries=4` (built-in backoff) | SDK |
| Parse retry | structured-output validation failure | shared `parse_with_retry()` in `llm.py`, error appended to prompt — **used by every stage, planner included** | 2 |
| Evidence RETRY | plan sound, 20–50% claims weak | same conversation + critic `fix_hints`, search better evidence for flagged findings only | 2 |
| RE-VALIDATE | mechanical-only failures (broken quotes/dead URLs on ≤30%) | targeted repair turn, re-fetch exact URLs, no new searching | 2 |
| REPLAN | structural failure (too few findings, >50% unsupported, tier collapse, ≥2 coverage gaps) | **fresh conversation** via `planner.replan_task()`, seeded with `queries_tried` (don't repeat), rejected findings (don't re-cite), frozen validated findings (keep, skip re-verify) | 1 |

Routing priority: REPLAN > RETRY > RE-VALIDATE. All pass-gate thresholds (min findings,
support ratio, tier mix) are **per-depth config values, not constants** — niche perspectives
can be legitimately thin, and hard gates would burn budget on futile retries.

**Budgets per researcher (standard depth):** ≤3 loop iterations, ~30 tool calls, 6–8 min
wall clock. On exhaustion: emit partial dossier of validated-only findings,
`budget_exhausted=True`, confidences capped, CLI exit code 2. `--deep` unlocks larger budgets.

## 6. Community sentiment (the reviews requirement)

Runs as wave 2, seeded with the **canonicalized union of methods wave 1 actually found**
(canonicalization = string normalization + one cheap low-effort call; without it "XGBoost" and
"gradient boosted trees" fragment the landscape table). Stricter gates than other researchers
because HN/Reddit/GitHub are noisy:

- **Corroboration:** any sentiment claim needs ≥2 quotes from distinct authors/threads,
  else it's demoted to a single-anecdote caveat.
- **Engagement floors** (mechanical, from fetched metadata): HN ≥10 points or ≥5 comments;
  GitHub issue ≥3 reactions or a maintainer reply; Reddit ≥20 upvotes.
- **Aboutness check:** the critic verifies the quote expresses the claimed sentiment about
  the claimed method — not a neighbor topic in the thread.
- **Evidence ranking:** `benchmark`/`production_report` outrank `anecdote` at synthesis.
- **Honesty:** <3 independent threads → `sentiment="insufficient_data"`, rendered as "–".
  `sentiment_sample_size` always reported so the synthesizer hedges.
- **`how_people_test_it`** captured per method: benchmarks, eval harnesses, A/B patterns
  seen in the wild.

## 7. Model strategy & cost

Opus 4.8 for everything that thinks and searches (planner, researchers, critic,
canonicalizer); Sonnet 5 writes the report; adaptive thinking everywhere (guarded per
model in code); **no `temperature`/`top_p`/`top_k` anywhere** (they 400 on Opus 4.8/
Sonnet 5). Byte-stable
system prompts with `cache_control: ephemeral`. (An earlier draft planned cache-warm
staggered fan-out — fire one researcher, await first token, then the rest. Dropped as
vacuous: prompt caching is a prefix match over tools+system, each perspective has a
different system prompt, and the shared tools block alone is below the cacheable
minimum, so agents share no warmable prefix. Caching pays off *within* each
researcher's own multi-turn loop instead.)

| Stage | Model | Effort | Notes |
|---|---|---|---|
| Planner | opus-4-8 | medium | `messages.parse`; scoping quality gates everything downstream |
| Researchers ×5 | opus-4-8 | medium (high at `--deep`) | streamed, manual loop; dominant cost (search results land as input tokens) |
| Mechanical verify | — | — | zero LLM |
| Critic | opus-4-8 batched, effort=low; single-finding arbitration | | low effort where volume is, focus where stakes are |
| Canonicalizer | opus-4-8 | low | one call between waves |
| Synthesizer | sonnet-5 | high | `messages.stream`, `max_tokens=32K` (streaming mandatory at this size) |

**Target: standard run ≈ $5–8, under ~10 min.** `--quick` ≈ $2–3 (fewer iterations/tool
uses); `--deep` ≈ $15–25 (larger budgets, effort=high researchers). Pricing lives in
`config.py` labeled as an estimate; the `UsageLedger` also meters **web-search per-use fees**
and enforces `--budget-usd` between loop iterations. Cost prints in the end-of-run table.

## 8. Report format

`./maxim-reports/<topic-slug>-<YYYY-MM-DD>.md` (+ `--json` → `RunResult`). Fixed template,
generated as streamed prose but constrained by the finding-id citation contract:

1. **Title + metadata** — topic, date, depth, cost, tier-mix badge (`A:41% B:32% C:18% D:9%`)
2. **TL;DR** — ≤6 bullets + one-line recommendation
3. **Method Landscape** — one table across all perspectives: Method | Perspective | Maturity |
   Effort to Adopt | Community Sentiment (▲/◆/▼/–) | Best When | Sources — *the screenshot artifact*
4. **AI / Agentic Take** · 5. **Classical ML Take** · 6. **Data Science Take** · 7. **Statistics Take**
   — per-method: what it is / why here / trade-offs / evidence, every claim footnoted
   `[F-id → URL, tier, verdict]`. Out-of-scope perspectives get one honest line, not filler.
8. **Community Pulse** — per method: sentiment verdict + sample size, 2–4 notable threads
   ("venue (date): takeaway [link]"), and **How people are testing it**
9. **Decision Guide** — constraints → recommended method; **must surface cross-perspective
   disagreements explicitly** (synthesizer prompt directive)
10. **Caveats & Unverified Claims** — partially-supported claims, demoted and flagged inline
11. **Appendix: Rejected claims** — what didn't survive grounding, with reasons (transparency)
12. **Bibliography** — grouped by tier; `[S7] Title — Publisher, date. <URL> (Tier B)`
13. **Run metadata footer** — models, efforts, tokens/$ per stage, loop counts per agent

## 9. CLI

```
maxim "vector search for multi-tenant SaaS"          # standard run
maxim "..." --depth quick|standard|deep              # budget presets
maxim "..." --perspectives ai,ml,stats               # subset
maxim "..." --json --quiet                           # machine mode: JSON on stdout,
                                                     # progress on stderr
maxim "..." --budget-usd 5 --yes --out PATH
```

- Plan assumptions/scope printed before fan-out (confirm unless `--yes`) — the cheapest
  insurance against a $6 run answering the wrong question.
- rich per-agent status lines (phase, sources, retries, running $) — simple lines, not a
  Live-dashboard subsystem.
- Exit codes: 0 ok · 2 partial/budget-exhausted · 3 plan/research failed. `NO_COLOR` honored.
- Per-run JSON artifacts (plan + dossiers) dumped to a scratch dir for debugging and golden
  tests. **No** `--resume`/manifest/HTTP-cache machinery — over-engineering for a <10-min run.

## 10. Claude Code integration (req 9)

**Skill over MCP** (unanimous across design reviews): an MCP server adds a running process,
holds a tool call open for a 5–15 min run, and duplicates orchestration; a skill keeps the
CLI the single source of truth and can run Maxim as a background Bash task.

Ship `.claude/skills/maxim/SKILL.md`: trigger description ("research approaches before
starting a new feature", "compare ML/stats options for X", "what is the community saying
about Y"); instructs Claude to refine the topic to one sentence, run
`uv run maxim "<topic>" --json --quiet` in the background, then present the TL;DR + Landscape
table and link the full report — trusting `Finding.confidence`/verdicts rather than
re-researching. The `--json` payload carries `report_schema_version` so the skill survives
releases. Optional M3: `maxim-mcp` as an `--extra mcp`, ~60 lines over `pipeline.run()`.

## 11. Dependencies

Runtime (M1 — exactly four): `anthropic`, `pydantic>=2`, `python-dotenv`, `rich`.
M2 adds `httpx` (custom tools) and `pyresilience` (API-layer retry/circuit-breaker). Dev: `pytest`, `ruff` (present) +
**`pytest-asyncio`**, `respx` (recorded fixtures). Deliberately excluded: LangChain/CrewAI,
typer/click (argparse suffices), tenacity (SDK retries + tiny helpers), tavily-python
(server-side web search covers it — note this in `.env.example`).

**Housekeeping:** prune `.env.example` — drop `OPENAI_API_KEY`/`SERPAPI_API_KEY`; keep
`ANTHROPIC_API_KEY` (required) + `SEMANTIC_SCHOLAR_API_KEY`/`GITHUB_TOKEN` (optional), with
a comment that Tavily/Brave are unnecessary because search runs server-side.

## 12. Testing

- Unit: schema round-trips, usage math, report rendering, quote-matcher edge cases
  (whitespace, unicode, truncation), loop-cap enforcement.
- **Critic calibration fixtures in CI**: known-good and known-bad finding/quote pairs the
  critic must classify correctly — guards against judge drift when prompts change.
- Pipeline tests with a `FakeLLM` injected at the `llm.py` seam (no network); recorded-fixture
  tests for the `pause_turn` re-send protocol (easy to get wrong).
- Golden-report snapshot test on one canned topic (M2+).

## 13. Milestones

**M1 — weekend vertical slice** (reqs 1, 2-lite, 3, 7, 8): schemas → `llm.py` (with
`parse_with_retry`) → planner → **one** researcher (server web tools, `pause_turn` handling,
**mechanical quote-verify from day one** — it's cheap, deterministic, and the requirement
most likely to be faked if deferred) → single-pass critic (drop unsupported) → streamed
synthesizer → markdown report → CLI with rich progress → FakeLLM tests → README + pruned
`.env.example`. Result: `uv sync && maxim "topic"` produces a real, cited, verified report.

**M2 — full breadth and rigor** (reqs 2, 4, 5, 6 complete): two-wave fan-out (4 perspectives
+ canonicalization + community wave) with semaphore and graceful timeouts; complete
five-loop state machine with per-depth thresholds; `reputation.py` tiers + recency half-lives
+ tier-mix badge; sentiment rigor (corroboration, floors, aboutness, `insufficient_data`,
`how_people_test_it`); custom tools with graceful no-key degradation; batched critic with
single-finding arbitration; full report template incl. Caveats + rejected appendix; `--quick`/`--deep`,
`--json`, budget gate; prompt-cache verification test.

**M3 — integration and hardening** (req 9): Claude Code skill authored and tested in-repo;
exit-code/JSON contract hardening + `report_schema_version`; critic calibration suite in CI;
golden-topic regression test; optional MCP extra; optional resume — only if runs prove long
enough to want it.

## 14. Top risks

1. **Critic rubber-stamping** → fresh-context critique, mechanical gate first, deterministic
   routing, CI calibration fixtures.
2. **Fabricated quotes** → quotes must match cached source text; server citations pre-verified;
   paywalls/talks → `verification_skipped` + capped confidence, disclosed in report.
3. **Cost/latency blowup** → hard caps everywhere, budget gate between iterations, live cost
   display, `--deep` opt-in rather than default.
4. **Perspective collapse into GenAI** → structural: per-perspective briefs with `avoid`
   fields, separate agents, independent report sections.
5. **Sentiment fabrication on niche topics** → corroboration floors + `insufficient_data`
   honesty rule + sample sizes surfaced.
6. **Rate limits with parallel Opus agents** → Semaphore(3), SDK retries, `--max-concurrency 1`
   escape hatch, stub-brief degradation so synthesis always runs.
