"""One researcher agent: GATHER → (DRAFT → mechanical VERIFY → CRITIQUE) loop.

The loop is a bounded state machine routed by `loop.decide()` — pure Python,
never LLM judgment. Per iteration the critic's verdicts split findings into
frozen (supported — locked, never re-researched), pending-weak (kept but
flagged; a later repair pass may supersede them), and rejected. RETRY searches
better evidence for flagged claims only; RE-VALIDATE re-fetches exact URLs to
repair broken quotes without new searching. REPLAN abandons the failed
transcript entirely: planner.replan_task writes a revised brief and research
restarts in a fresh conversation, carrying over validated findings, the
source cache, and the list of already-tried queries.

Timeouts are graceful: the orchestrator passes a soft deadline and the loop
stops between phases, salvaging every validated finding instead of losing the
run to a hard cancel.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any

from .config import Settings
from .critic import apply_critique, critique
from .llm import LLM, CitedQuote, LLMError, SourceDoc, dump_for_prompt
from .loop import IterationOutcome, LoopState, decide
from .planner import replan_task
from .prompts import (
    COMMUNITY_DRAFT_SUFFIX,
    DRAFT_INSTRUCTION,
    REPAIR_DRAFT_INSTRUCTION,
    RESEARCHER_SYSTEMS,
    RETRY_INSTRUCTION_HEADER,
    REVALIDATE_INSTRUCTION_HEADER,
)
from .reputation import BLOCKED_DOMAINS, stamp_evidence
from .schemas import (
    PERSPECTIVE_ID_PREFIX,
    DraftDossier,
    EngagementStats,
    Finding,
    RejectedFinding,
    ResearchBrief,
    ResearchDossier,
    ResearchPlan,
)
from .sentiment import apply_sentiment_rigor, normalize_url
from .tools import client_tools_for
from .verification import verify_evidence

# Don't start another repair pass with less than this much wall-clock left —
# a gather + draft + critique that gets cancelled midway is pure waste.
DEADLINE_MARGIN_S = 60.0
# Repair turns are targeted; they never need the full gather continuation budget.
REPAIR_MAX_CONTINUATIONS = 2
REPAIR_FETCH_MAX_USES = 4

_ID_NUM = re.compile(r"(\d+)$")


def _web_tools(settings: Settings) -> list[dict[str, Any]]:
    preset = settings.preset
    return [
        {
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": preset.web_search_max_uses,
            "blocked_domains": BLOCKED_DOMAINS,
        },
        _fetch_tool(settings, preset.web_fetch_max_uses),
    ]


def _research_tools(settings: Settings, perspective: str) -> list[Any]:
    """Server web tools + the perspective's client-side specialty tools."""
    return [*_web_tools(settings), *client_tools_for(perspective)]


def _fetch_tool(settings: Settings, max_uses: int) -> dict[str, Any]:
    return {
        "type": "web_fetch_20260209",
        "name": "web_fetch",
        "max_uses": max_uses,
        "citations": {"enabled": True},
        "max_content_tokens": settings.web_fetch_max_content_tokens,
        "blocked_domains": BLOCKED_DOMAINS,
    }


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


def _retry_message(
    weak: list[Finding],
    critic_rejected: list[RejectedFinding],
    coverage_gaps: list[str],
) -> str:
    lines = [RETRY_INSTRUCTION_HEADER, ""]
    if weak:
        lines.append("PARTIALLY SUPPORTED (strengthen the evidence or refine the claim):")
        for f in weak:
            hint = next((c for c in f.caveats if c.startswith("critic:")), "")
            lines.append(f'- [{f.method_name}] "{f.claim}" {hint}'.rstrip())
    if critic_rejected:
        lines.append("REJECTED (find evidence that actually carries the claim, or drop it):")
        for r in critic_rejected:
            lines.append(f'- [{r.finding.method_name}] "{r.finding.claim}" — {r.reason}')
    if coverage_gaps:
        lines.append("UNANSWERED SUB-QUESTIONS:")
        lines.extend(f"- {gap}" for gap in coverage_gaps)
    return "\n".join(lines)


def _revalidate_message(mech_rejected: list[RejectedFinding]) -> str:
    lines = [REVALIDATE_INSTRUCTION_HEADER, ""]
    for r in mech_rejected:
        for ev in r.finding.evidence:
            if ev.status == "failed":
                lines.append(f'- "{ev.quote}"\n  claimed from: {ev.source_url}')
    return "\n".join(lines)


def _replan_gather_message(
    brief: ResearchBrief,
    plan: ResearchPlan,
    queries_tried: list[str],
    validated: list[Finding],
    rejected: list[RejectedFinding],
) -> str:
    lines = [
        _gather_message(brief, plan),
        "",
        "This brief is a REPLAN after a structurally failed pass. Constraints:",
    ]
    if queries_tried:
        lines.append(f"- Queries already tried (do not repeat): {json.dumps(queries_tried)}")
    if validated:
        lines.append(
            "- Already-validated claims (locked in — do not re-research): "
            + json.dumps([f.claim for f in validated])
        )
    if rejected:
        lines.append(
            "- Previously rejected claims (do not re-submit them or their sources): "
            + json.dumps([r.finding.claim for r in rejected])
        )
    return "\n".join(lines)


def _repair_draft_instruction(frozen: list[Finding]) -> str:
    claims = "\n".join(f'- [{f.method_name}] "{f.claim}"' for f in frozen) or "- (none yet)"
    return REPAIR_DRAFT_INSTRUCTION.format(frozen_claims=claims)


def _merge_unique(base: list[str], extra: list[str]) -> list[str]:
    seen = {item.casefold() for item in base}
    merged = list(base)
    for item in extra:
        if item.casefold() not in seen:
            seen.add(item.casefold())
            merged.append(item)
    return merged


def _norm_claim(claim: str) -> str:
    return " ".join(claim.split()).casefold()


def _id_ordinal(finding: Finding) -> int:
    match = _ID_NUM.search(finding.id)
    return int(match.group(1)) if match else 0


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


def _mechanical_gate(
    draft: DraftDossier,
    brief: ResearchBrief,
    plan: ResearchPlan,
    source_cache: dict[str, SourceDoc],
    cited_quotes: list[CitedQuote],
    engagement: dict[str, EngagementStats],
    next_id: int,
) -> tuple[list[Finding], list[RejectedFinding], int]:
    """Verify + stamp every draft finding; reject provable fabrications."""
    prefix = PERSPECTIVE_ID_PREFIX[brief.perspective]
    survivors: list[Finding] = []
    rejected: list[RejectedFinding] = []
    # Tool keys and model-cited URLs can differ in trailing slash / case /
    # fragment; floors must not be silently bypassed by spelling drift.
    engagement_by_url = {normalize_url(url): stats for url, stats in engagement.items()}
    for draft_finding in draft.findings:
        evidence = [
            stamp_evidence(
                verify_evidence(ev, source_cache, cited_quotes),
                brief.perspective,
                plan.recency_horizon_months,
            ).model_copy(update={"engagement": engagement_by_url.get(normalize_url(ev.source_url))})
            for ev in draft_finding.evidence
        ]
        finding = Finding(
            id=f"F-{prefix}{next_id}",
            perspective=brief.perspective,
            claim=draft_finding.claim,
            method_name=draft_finding.method_name,
            evidence=evidence,
            confidence="low",
            verdict=None,
            caveats=draft_finding.caveats,
            sentiment=draft_finding.sentiment,
            how_people_test_it=draft_finding.how_people_test_it,
        )
        next_id += 1
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
    return survivors, rejected, next_id


def _cap_confidence(findings: list[Finding], note: str) -> list[Finding]:
    """Exhaustion cap: work was still pending, so nothing keeps 'high'."""
    capped = []
    for f in findings:
        if f.confidence == "high":
            capped.append(
                f.model_copy(update={"confidence": "medium", "caveats": f.caveats + [note]})
            )
        else:
            capped.append(f)
    return capped


async def run_researcher(
    brief: ResearchBrief,
    plan: ResearchPlan,
    settings: Settings,
    llm: LLM,
    progress: Callable[[str], None],
    deadline: float | None = None,
    checkpoint: dict[str, ResearchDossier] | None = None,
) -> ResearchDossier:
    preset = settings.preset
    policy = preset.loop
    stage = f"researcher:{brief.perspective}"
    system = RESEARCHER_SYSTEMS[brief.perspective]
    current_brief = brief

    source_cache: dict[str, SourceDoc] = {}
    cited_quotes: list[CitedQuote] = []
    queries_tried: list[str] = []
    engagement: dict[str, EngagementStats] = {}

    progress("searching…")
    gathered = await llm.run_agentic(
        stage=stage,
        system=system,
        messages=[{"role": "user", "content": _gather_message(brief, plan)}],
        tools=_research_tools(settings, brief.perspective),
        model=settings.researcher_model,
        effort=preset.researcher_effort,
        max_tokens=preset.gather_max_tokens,
        max_continuations=preset.max_continuations,
        on_progress=progress,
    )
    transcript = gathered.messages
    source_cache.update(gathered.source_cache)
    cited_quotes.extend(gathered.cited_quotes)
    queries_tried.extend(gathered.queries)
    engagement.update(gathered.engagement)
    truncated_gather = gathered.truncated
    continuations = gathered.continuations

    state = LoopState()
    loop_actions: list[str] = []
    iterations = 0
    budget_exhausted = False
    stop_notes: list[str] = []
    stopped_with_pending_work = False

    frozen: list[Finding] = []  # verdict == supported: locked in
    pending_weak: list[Finding] = []  # partially_supported: kept, repairable
    rejected: list[RejectedFinding] = []
    coverage_gaps: list[str] = []
    summary = ""
    methods: list[str] = []
    draft_gaps: list[str] = []
    next_id = 1
    community = brief.perspective == "community"
    draft_instruction = DRAFT_INSTRUCTION + (COMMUNITY_DRAFT_SUFFIX if community else "")

    def assemble(*, capped: bool, extra_notes: tuple[str, ...] = ()) -> ResearchDossier:
        findings = sorted(frozen + pending_weak, key=_id_ordinal)
        if community:
            findings = apply_sentiment_rigor(findings)
        if capped:
            findings = _cap_confidence(
                findings, "run ended with repair work pending — confidence capped"
            )
        gaps = _merge_unique(draft_gaps, coverage_gaps)
        if truncated_gather:
            gaps.append(
                "research gather stopped early (continuation/token/budget cap) — "
                "coverage may be incomplete"
            )
        gaps.extend(stop_notes)
        gaps.extend(extra_notes)
        searches, fetches = llm.ledger.stage_counts(stage)
        return ResearchDossier(
            perspective=brief.perspective,
            summary=summary,
            findings=findings,
            rejected=rejected,
            methods_identified=methods,
            gaps=gaps,
            ok=True,
            failure=None,
            web_searches=searches,
            web_fetches=fetches,
            continuations=continuations,
            iterations=iterations,
            loop_actions=loop_actions,
            budget_exhausted=budget_exhausted,
        )

    while True:
        iterations += 1
        progress(f"extracting findings (pass {iterations})…")
        try:
            draft: DraftDossier = await llm.parse(
                stage=stage,
                system=system,
                messages=transcript + [{"role": "user", "content": draft_instruction}],
                output_format=DraftDossier,
                model=settings.researcher_model,
                effort=preset.researcher_effort,
            )
        except LLMError as exc:
            stop_notes.append(f"draft extraction failed ({exc}) — keeping partial results")
            stopped_with_pending_work = True
            break
        if draft.summary:
            summary = draft.summary
        methods = _merge_unique(methods, draft.methods_identified)
        draft_gaps = _merge_unique(draft_gaps, draft.gaps)

        survivors, mech_rejected, next_id = _mechanical_gate(
            draft, brief, plan, source_cache, cited_quotes, engagement, next_id
        )

        new_validated: list[Finding] = []
        critic_rejected: list[RejectedFinding] = []
        if survivors:
            progress(f"critiquing {len(survivors)} findings…")
            try:
                result = await critique(
                    stage=f"critic:{brief.perspective}",
                    brief=current_brief,
                    findings=survivors,
                    source_cache=source_cache,
                    settings=settings,
                    llm=llm,
                    all_claims=[f.claim for f in frozen + pending_weak + survivors],
                )
            except LLMError as exc:
                # Uncritiqued survivors are discarded — they never passed the
                # gate — but everything validated so far is kept.
                stop_notes.append(f"critique failed ({exc}) — keeping partial results")
                stopped_with_pending_work = True
                break
            new_validated, critic_rejected = apply_critique(survivors, result)
            coverage_gaps = result.coverage_gaps

        new_supported = [f for f in new_validated if f.verdict == "supported"]
        new_weak = [f for f in new_validated if f.verdict != "supported"]
        # A repaired claim supersedes the weak version it replaces (same
        # method) — even a still-weak repair, else re-drafts would pile up
        # near-duplicate findings across iterations.
        superseded = {f.method_name.casefold() for f in new_validated}
        pending_weak = [f for f in pending_weak if f.method_name.casefold() not in superseded]
        frozen.extend(new_supported)
        pending_weak.extend(new_weak)

        # Rejected-list hygiene, time-aware in both directions: an EARLIER
        # rejection repaired by THIS pass's validation leaves the list, and a
        # rejection from THIS pass evicts a stale weak twin (the later verdict
        # wins) — but never a frozen supported finding. Duplicates (same claim
        # and reason re-rejected across passes) collapse to one entry.
        newly_validated = {_norm_claim(f.claim) for f in new_validated}
        rejected = [r for r in rejected if _norm_claim(r.finding.claim) not in newly_validated]
        rejected_now = {_norm_claim(r.finding.claim) for r in mech_rejected + critic_rejected}
        pending_weak = [f for f in pending_weak if _norm_claim(f.claim) not in rejected_now]
        seen = {(_norm_claim(r.finding.claim), r.reason) for r in rejected}
        for rejection in mech_rejected + critic_rejected:
            key = (_norm_claim(rejection.finding.claim), rejection.reason)
            if key not in seen:
                seen.add(key)
                rejected.append(rejection)

        if checkpoint is not None:
            # Best-so-far snapshot: if the hard timeout backstop cancels this
            # coroutine mid-pass, the orchestrator salvages the last completed
            # pass instead of discarding every validated finding.
            checkpoint["dossier"] = assemble(
                capped=True,
                extra_notes=("hard timeout: salvaged from the last completed pass",),
            )

        outcome = IterationOutcome(
            drafted=len(draft.findings),
            validated=len(frozen) + len(pending_weak),
            weak=len(new_weak),
            unsupported=len(critic_rejected),
            mechanical_failed=len(mech_rejected),
            coverage_gaps=len(coverage_gaps),
        )
        decision = decide(outcome, state, policy)
        if decision.action == "accept":
            if decision.degraded:
                stop_notes.append(
                    "accepted with repair work pending: " + "; ".join(decision.reasons)
                )
                stopped_with_pending_work = True
            break
        if iterations >= policy.max_iterations:
            stop_notes.append(
                f"loop stopped at max_iterations={policy.max_iterations} "
                f"(pending action: {decision.action})"
            )
            stopped_with_pending_work = True
            break
        if llm.ledger.over_budget:
            budget_exhausted = True
            stop_notes.append("loop stopped: cost budget exhausted — partial results kept")
            stopped_with_pending_work = True
            break
        if deadline is not None and time.monotonic() >= deadline - DEADLINE_MARGIN_S:
            stop_notes.append("loop stopped: wall-clock deadline reached — partial results kept")
            stopped_with_pending_work = True
            break
        state = state.spend(decision.action)
        loop_actions.append(decision.action)
        if decision.action == "replan":
            # Fresh conversation on a revised brief: a structurally failed
            # transcript poisons further turns, so it is abandoned. Validated
            # findings, the source cache, and tried queries carry over.
            progress("replanning after structural failure…")
            try:
                current_brief = await replan_task(
                    current_brief,
                    plan,
                    reasons=decision.reasons,
                    queries_tried=queries_tried,
                    validated=frozen + pending_weak,
                    rejected=rejected,
                    settings=settings,
                    llm=llm,
                )
            except LLMError as exc:
                stop_notes.append(f"replan failed ({exc}) — keeping partial results")
                stopped_with_pending_work = True
                break
            progress("researching the revised brief…")
            messages = [
                {
                    "role": "user",
                    "content": _replan_gather_message(
                        current_brief, plan, queries_tried, frozen + pending_weak, rejected
                    ),
                }
            ]
            tools = _research_tools(settings, brief.perspective)
            max_continuations = preset.max_continuations
        elif decision.action == "retry":
            progress("retrying weak claims with critic hints…")
            message = _retry_message(pending_weak, critic_rejected, coverage_gaps)
            messages = transcript + [{"role": "user", "content": message}]
            tools = _research_tools(settings, brief.perspective)
            max_continuations = preset.max_continuations
        else:  # revalidate
            progress("re-fetching sources to repair broken quotes…")
            message = _revalidate_message(mech_rejected)
            messages = transcript + [{"role": "user", "content": message}]
            tools = [_fetch_tool(settings, REPAIR_FETCH_MAX_USES)]
            max_continuations = REPAIR_MAX_CONTINUATIONS
        try:
            gathered = await llm.run_agentic(
                stage=stage,
                system=system,
                messages=messages,
                tools=tools,
                model=settings.researcher_model,
                effort=preset.researcher_effort,
                max_tokens=preset.gather_max_tokens,
                max_continuations=max_continuations,
                on_progress=progress,
            )
        except LLMError as exc:
            stop_notes.append(f"repair search failed ({exc}) — keeping partial results")
            stopped_with_pending_work = True
            break
        transcript = gathered.messages
        source_cache.update(gathered.source_cache)
        cited_quotes.extend(gathered.cited_quotes)
        queries_tried.extend(gathered.queries)
        engagement.update(gathered.engagement)
        truncated_gather = truncated_gather or gathered.truncated
        continuations += gathered.continuations
        draft_instruction = _repair_draft_instruction(frozen) + (
            COMMUNITY_DRAFT_SUFFIX if community else ""
        )

    return assemble(capped=stopped_with_pending_work)
