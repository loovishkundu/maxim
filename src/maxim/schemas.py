"""Pydantic models that form the contract between pipeline stages.

Two families live here:

- Draft* models are what the LLM emits during structured extraction. They carry
  no verification state — the model must never be able to mark its own work as
  verified.
- The enriched models (Evidence, Finding, ResearchDossier, ...) are produced by
  code, which stamps verification status, ids, and confidence deterministically.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1"

Perspective = Literal["ai_agentic", "classical_ml", "data_science", "statistics", "community"]

PERSPECTIVES: tuple[Perspective, ...] = (
    "ai_agentic",
    "classical_ml",
    "data_science",
    "statistics",
    "community",
)

PERSPECTIVE_LABELS: dict[str, str] = {
    "ai_agentic": "AI / Agentic",
    "classical_ml": "Classical ML",
    "data_science": "Data Science",
    "statistics": "Statistics",
    "community": "Community",
}

PERSPECTIVE_ID_PREFIX: dict[str, str] = {
    "ai_agentic": "ai",
    "classical_ml": "ml",
    "data_science": "ds",
    "statistics": "st",
    "community": "cm",
}

EvidenceKind = Literal[
    "paper", "blog", "article", "docs", "talk", "anecdote", "benchmark", "production_report"
]

VerificationStatus = Literal["verified", "failed", "skipped"]

Verdict = Literal[
    "supported", "partially_supported", "unsupported", "contradicted", "source_unreliable"
]

Confidence = Literal["high", "medium", "low"]

Sentiment = Literal["positive", "mixed", "negative", "insufficient_data"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- plan


class ResearchBrief(StrictModel):
    perspective: Perspective
    objective: str
    sub_questions: list[str]
    seed_queries: list[str]
    must_cover_methods: list[str]
    avoid: list[str]


class OutOfScope(StrictModel):
    perspective: Perspective
    reason: str


class ResearchPlan(StrictModel):
    topic: str
    domain: str
    rationale: str
    assumptions: list[str]
    recency_horizon_months: int
    briefs: list[ResearchBrief]
    out_of_scope: list[OutOfScope]


# --------------------------------------------------------------- model-emitted drafts


class DraftEvidence(StrictModel):
    quote: str
    source_url: str
    source_title: str
    published: str | None
    kind: EvidenceKind


class DraftFinding(StrictModel):
    claim: str
    method_name: str
    # min_length is validated client-side (the SDK strips unsupported schema
    # constraints); a violation triggers the parse retry loop.
    evidence: list[DraftEvidence] = Field(min_length=1)
    caveats: list[str]
    # Community researcher only; code demotes sentiment lacking corroboration.
    sentiment: Literal["positive", "mixed", "negative"] | None = None
    how_people_test_it: list[str] = Field(default_factory=list)


class DraftDossier(StrictModel):
    summary: str
    findings: list[DraftFinding]
    methods_identified: list[str]
    gaps: list[str]


# ------------------------------------------------------------------ enriched findings


SourceTier = Literal["A", "B", "C", "D"]


class EngagementStats(StrictModel):
    """Mechanical engagement metadata from the community search tools —
    never model-reported. Basis for the community evidence floors."""

    source: Literal["hn", "github", "reddit", "other"]
    points: int | None = None
    comments: int | None = None
    reactions: int | None = None


class Evidence(StrictModel):
    quote: str
    source_url: str
    source_title: str
    published: str | None
    kind: EvidenceKind
    status: VerificationStatus
    match_ratio: float | None
    # Stamped by reputation.py (deterministic), never by the model.
    tier: SourceTier | None = None
    recency_score: float | None = None
    # Stamped from tool metadata (deterministic); None when unavailable.
    engagement: EngagementStats | None = None


class Finding(StrictModel):
    id: str
    perspective: Perspective
    claim: str
    method_name: str
    evidence: list[Evidence]
    confidence: Confidence
    verdict: Verdict | None
    caveats: list[str]
    # Community only. Sentiment survives only with mechanical corroboration
    # (≥2 distinct qualifying threads); sample size is stamped by code.
    sentiment: Literal["positive", "mixed", "negative"] | None = None
    sentiment_sample_size: int | None = None
    how_people_test_it: list[str] = Field(default_factory=list)


class RejectedFinding(StrictModel):
    finding: Finding
    reason: str


# ----------------------------------------------------------------------- critique


class ClaimVerdict(StrictModel):
    finding_id: str
    verdict: Verdict
    fix_hint: str | None


class CritiqueResult(StrictModel):
    verdicts: list[ClaimVerdict]
    coverage_gaps: list[str]


class CoverageResult(StrictModel):
    """Output of the dedicated coverage pass (batched critics only see slices,
    so coverage is judged once against the full claim list)."""

    coverage_gaps: list[str]


# ---------------------------------------------------------------- canonicalization


class MethodGroup(StrictModel):
    canonical: str
    variants: list[str]


class CanonicalMethods(StrictModel):
    """Canonicalizer output: every input name grouped under one canonical name."""

    groups: list[MethodGroup]


# ------------------------------------------------------------------------ dossier


class ResearchDossier(StrictModel):
    perspective: Perspective
    summary: str
    findings: list[Finding]
    rejected: list[RejectedFinding]
    methods_identified: list[str]
    gaps: list[str]
    ok: bool
    failure: str | None
    web_searches: int
    web_fetches: int
    continuations: int
    # Loop-state-machine telemetry (M2): how many draft→verify→critique passes
    # ran and which loop actions were taken, e.g. ["retry", "revalidate"].
    iterations: int = 1
    loop_actions: list[str] = Field(default_factory=list)
    budget_exhausted: bool = False


# -------------------------------------------------------------------- community pulse


class MethodPulse(StrictModel):
    """Mechanical per-method aggregate of the community findings."""

    method: str
    sentiment: Sentiment
    sample_size: int  # distinct qualifying threads across the method's findings
    notable_threads: list[str]
    how_people_test_it: list[str]


# -------------------------------------------------------------------------- usage


class StageUsage(StrictModel):
    stage: str
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    web_searches: int
    web_fetches: int
    cost_usd: float


class RunUsage(StrictModel):
    stages: list[StageUsage]
    total_cost_usd: float
    wall_seconds: float


# -------------------------------------------------------------------------- result


class RunResult(StrictModel):
    schema_version: str
    topic: str
    plan: ResearchPlan
    dossiers: list[ResearchDossier]
    # Canonical method names shared across perspectives (wave-1 union after
    # canonicalization) — the vocabulary of the Method Landscape table.
    canonical_methods: list[str] = Field(default_factory=list)
    pulse: list[MethodPulse] = Field(default_factory=list)
    report_markdown: str
    usage: RunUsage
    partial: bool
    warnings: list[str]
