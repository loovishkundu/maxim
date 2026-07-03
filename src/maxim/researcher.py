"""One researcher agent: GATHER → DRAFT → mechanical VERIFY → CRITIQUE.

M1 runs a single pass of each phase (the evidence-retry / re-validate / replan
loops land in M2); the mechanical verification gate and the fresh-context
critic are in from day one, so nothing unverified reaches the synthesizer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .config import Settings
from .critic import apply_critique, critique
from .llm import LLM, dump_for_prompt
from .prompts import DRAFT_INSTRUCTION, RESEARCHER_SYSTEMS
from .reputation import BLOCKED_DOMAINS, stamp_evidence
from .schemas import (
    PERSPECTIVE_ID_PREFIX,
    DraftDossier,
    Finding,
    RejectedFinding,
    ResearchBrief,
    ResearchDossier,
    ResearchPlan,
)
from .verification import verify_evidence


def _web_tools(settings: Settings) -> list[dict[str, Any]]:
    preset = settings.preset
    return [
        {
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": preset.web_search_max_uses,
            "blocked_domains": BLOCKED_DOMAINS,
        },
        {
            "type": "web_fetch_20260209",
            "name": "web_fetch",
            "max_uses": preset.web_fetch_max_uses,
            "citations": {"enabled": True},
            "max_content_tokens": settings.web_fetch_max_content_tokens,
            "blocked_domains": BLOCKED_DOMAINS,
        },
    ]


def _gather_message(brief: ResearchBrief, plan: ResearchPlan) -> str:
    return (
        f"Topic: {plan.topic}\n"
        f"Domain: {plan.domain}\n"
        f"Recency horizon: prefer sources from the last "
        f"{plan.recency_horizon_months} months (older is acceptable for stable "
        f"fundamentals — flag it).\n\n"
        f"Your research brief (JSON):\n{dump_for_prompt(brief)}\n\n"
        "Research this now using web_search and web_fetch. Fetch every page you "
        "intend to quote."
    )


def stub_dossier(
    brief: ResearchBrief,
    failure: str,
    web_searches: int = 0,
    web_fetches: int = 0,
) -> ResearchDossier:
    return ResearchDossier(
        perspective=brief.perspective,
        summary="",
        findings=[],
        rejected=[],
        methods_identified=[],
        gaps=[f"researcher failed: {failure}"],
        ok=False,
        failure=failure,
        web_searches=web_searches,
        web_fetches=web_fetches,
        continuations=0,
    )


async def run_researcher(
    brief: ResearchBrief,
    plan: ResearchPlan,
    settings: Settings,
    llm: LLM,
    progress: Callable[[str], None],
) -> ResearchDossier:
    preset = settings.preset
    stage = f"researcher:{brief.perspective}"
    system = RESEARCHER_SYSTEMS[brief.perspective]

    progress("searching…")
    gathered = await llm.run_agentic(
        stage=stage,
        system=system,
        messages=[{"role": "user", "content": _gather_message(brief, plan)}],
        tools=_web_tools(settings),
        model=settings.researcher_model,
        effort=preset.researcher_effort,
        max_tokens=preset.gather_max_tokens,
        max_continuations=preset.max_continuations,
        on_progress=progress,
    )

    progress("extracting findings…")
    draft: DraftDossier = await llm.parse(
        stage=stage,
        system=system,
        messages=gathered.messages + [{"role": "user", "content": DRAFT_INSTRUCTION}],
        output_format=DraftDossier,
        model=settings.researcher_model,
        effort=preset.researcher_effort,
    )

    prefix = PERSPECTIVE_ID_PREFIX[brief.perspective]
    survivors: list[Finding] = []
    rejected: list[RejectedFinding] = []
    for n, draft_finding in enumerate(draft.findings, 1):
        evidence = [
            stamp_evidence(
                verify_evidence(ev, gathered.source_cache, gathered.cited_quotes),
                brief.perspective,
                plan.recency_horizon_months,
            )
            for ev in draft_finding.evidence
        ]
        finding = Finding(
            id=f"F-{prefix}{n}",
            perspective=brief.perspective,
            claim=draft_finding.claim,
            method_name=draft_finding.method_name,
            evidence=evidence,
            confidence="low",
            verdict=None,
            caveats=draft_finding.caveats,
        )
        statuses = {ev.status for ev in evidence}
        if not evidence:
            # Belt and suspenders with the schema's min_length: a claim with no
            # evidence must never bypass the mechanical gate.
            rejected.append(
                RejectedFinding(finding=finding, reason="mechanical: no evidence quotes")
            )
        elif "verified" not in statuses and "failed" in statuses:
            # We HAVE the source text and the quote is not in it — the
            # anti-fabrication gate. No LLM judgment involved.
            rejected.append(
                RejectedFinding(
                    finding=finding,
                    reason="mechanical: quote(s) not found in the fetched source text",
                )
            )
        else:
            survivors.append(finding)

    validated: list[Finding] = []
    coverage_gaps: list[str] = []
    if survivors:
        progress(f"critiquing {len(survivors)} findings…")
        result = await critique(
            stage=f"critic:{brief.perspective}",
            brief=brief,
            findings=survivors,
            source_cache=gathered.source_cache,
            settings=settings,
            llm=llm,
        )
        validated, critic_rejected = apply_critique(survivors, result)
        rejected.extend(critic_rejected)
        coverage_gaps = result.coverage_gaps

    gaps = draft.gaps + coverage_gaps
    if gathered.truncated:
        gaps.append(
            "research gather stopped early (continuation/token/budget cap) — "
            "coverage may be incomplete"
        )

    searches, fetches = llm.ledger.stage_counts(stage)
    return ResearchDossier(
        perspective=brief.perspective,
        summary=draft.summary,
        findings=validated,
        rejected=rejected,
        methods_identified=draft.methods_identified,
        gaps=gaps,
        ok=True,
        failure=None,
        web_searches=searches,
        web_fetches=fetches,
        continuations=gathered.continuations,
    )
