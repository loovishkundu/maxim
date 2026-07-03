from maxim.schemas import (
    DraftDossier,
    DraftEvidence,
    DraftFinding,
    Evidence,
    Finding,
    OutOfScope,
    ResearchBrief,
    ResearchDossier,
    ResearchPlan,
)

PAGE_TEXT = (
    "In production trials, seasonal-trend decomposition (STL) caught 92% of "
    "injected anomalies on vehicle telemetry with a false-positive rate under 2%. "
    "Prophet, by contrast, needed heavy per-signal tuning."
)
GOOD_QUOTE = "seasonal-trend decomposition (STL) caught 92% of injected anomalies"
FAKE_QUOTE = "STL is universally considered obsolete for telemetry work"
SOURCE_URL = "https://eng.example.com/anomaly-detection"


def make_brief(perspective="ai_agentic") -> ResearchBrief:
    return ResearchBrief(
        perspective=perspective,
        objective=f"Investigate {perspective} approaches",
        sub_questions=["What methods exist?", "How mature are they?"],
        seed_queries=["anomaly detection telemetry"],
        must_cover_methods=[],
        avoid=[] if perspective == "ai_agentic" else ["LLM/GenAI approaches"],
    )


def make_plan(perspectives=("ai_agentic", "statistics")) -> ResearchPlan:
    return ResearchPlan(
        topic="anomaly detection for vehicle telemetry",
        domain="time-series anomaly detection",
        rationale="testing",
        assumptions=["streaming telemetry, not batch"],
        recency_horizon_months=24,
        briefs=[make_brief(p) for p in perspectives],
        out_of_scope=[OutOfScope(perspective="community", reason="test fixture")],
    )


def _draft_finding(claim: str, method: str, quote: str = GOOD_QUOTE) -> DraftFinding:
    return DraftFinding(
        claim=claim,
        method_name=method,
        evidence=[
            DraftEvidence(
                quote=quote,
                source_url=SOURCE_URL,
                source_title="Anomaly detection at Example Corp",
                published="2026-01",
                kind="blog",
            )
        ],
        caveats=[],
    )


def make_draft_dossier() -> DraftDossier:
    """Three well-grounded findings + one fabricated quote.

    Healthy enough to pass the standard loop gate (min 3 validated) so
    pipeline tests exercise the single-pass happy path.
    """
    return DraftDossier(
        summary="STL looks strong; Prophet needs tuning.",
        findings=[
            _draft_finding(
                "STL catches most injected anomalies on vehicle telemetry.",
                "STL decomposition",
            ),
            _draft_finding("STL has a low false-positive rate.", "STL decomposition"),
            _draft_finding("Prophet needs heavy per-signal tuning.", "Prophet"),
            _draft_finding("STL is obsolete for telemetry.", "STL decomposition", FAKE_QUOTE),
        ],
        methods_identified=["STL decomposition", "Prophet"],
        gaps=["no GPU benchmarks found"],
    )


def make_finding(fid="F-ai1", perspective="ai_agentic", status="verified") -> Finding:
    return Finding(
        id=fid,
        perspective=perspective,
        claim="STL catches most injected anomalies.",
        method_name="STL decomposition",
        evidence=[
            Evidence(
                quote=GOOD_QUOTE,
                source_url=SOURCE_URL,
                source_title="Anomaly detection at Example Corp",
                published="2026-01",
                kind="blog",
                status=status,
                match_ratio=1.0 if status == "verified" else None,
                tier="B",  # eng.example.com engineering blog
                recency_score=0.8,
            )
        ],
        confidence="high",
        verdict="supported",
        caveats=[],
    )


def make_dossier(perspective="ai_agentic", fid="F-ai1") -> ResearchDossier:
    return ResearchDossier(
        perspective=perspective,
        summary="summary",
        findings=[make_finding(fid, perspective)],
        rejected=[],
        methods_identified=["STL decomposition"],
        gaps=[],
        ok=True,
        failure=None,
        web_searches=3,
        web_fetches=2,
        continuations=0,
    )
